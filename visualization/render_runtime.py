"""Opt-in, bounded process rendering of immutable, synchronously captured inputs.

Only v0.31 runner aliases of visualization.save_* functions are intercepted.
The original visualization functions and all numerical functions remain unchanged.
Workers use spawn on every OS and never read the live simulation's global state.
"""
from __future__ import annotations

from collections import OrderedDict, deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import AbstractContextManager
import functools
import importlib
import inspect
import multiprocessing
import os
from pathlib import Path
import pickle
import sys
import tempfile
from time import perf_counter, process_time

from .process_lifetime import WorkerLifetime, bind_worker_to_owner, process_diagnostics
from execution_policy import RENDER_WORKER_CHOICES, PROCESS_PRIORITY_CHOICES, apply_process_priority, read_process_priority

_active = None
_worker_meshes = OrderedDict()
_worker_execution = None
_synchronous_renderers = {("visualization.volcanic_arc", "save_volcanic_arc_maps")}


def _initialize_worker(job_name, owner_pid, process_priority):
    global _worker_execution
    bind_worker_to_owner(job_name, owner_pid)
    apply_process_priority(process_priority)
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from tectonics.cpu_runtime import CpuExecution
    # Rendering needs geometry caches, not alternative physics dispatch.
    _worker_execution = CpuExecution(1, numeric_kernels=False, single_source_cells=False)
    _worker_execution.__enter__()
    original_save = Figure.savefig

    def atomic_save(figure, destination, *args, **kwargs):
        if not isinstance(destination, (str, os.PathLike)):
            raise TypeError("Render workers require a filesystem output path")
        path = Path(destination)
        # A .part suffix keeps unfinished files out of the GUI's PNG discovery.
        descriptor, temporary = tempfile.mkstemp(prefix=".moon-render-", suffix=".part", dir=path.parent)
        os.close(descriptor)
        try:
            kwargs.setdefault("format", path.suffix.lstrip(".") or "png")
            result = original_save(figure, temporary, *args, **kwargs)
            os.replace(temporary, path)
            return result
        finally:
            Path(temporary).unlink(missing_ok=True)

    Figure.savefig = atomic_save


def _render_job(module_name, function_name, payload, mesh_slots):
    started = perf_counter()
    cpu_started = process_time()
    args, kwargs = pickle.loads(payload)
    args = list(args)
    for kind, slot, token in mesh_slots:
        container = args if kind == "arg" else kwargs
        mesh = container[slot]
        if token in _worker_meshes:
            container[slot] = _worker_meshes[token]
            _worker_meshes.move_to_end(token)
        else:
            _worker_meshes[token] = mesh
            if len(_worker_meshes) > 2:
                _worker_meshes.popitem(last=False)
    function = getattr(importlib.import_module(module_name), function_name)
    try:
        result = function(*args, **kwargs)
        if result is not None:
            raise TypeError(f"Asynchronous renderer {module_name}.{function_name} returned a value")
    finally:
        import matplotlib.pyplot as plt
        plt.close("all")
    return {"function": f"{module_name}.{function_name}", "pid": os.getpid(),
            "worker_seconds": perf_counter() - started, "cpu_seconds": process_time() - cpu_started,
            "snapshot_bytes": len(payload), "priority": read_process_priority(), **process_diagnostics()}


class RenderExecution(AbstractContextManager):
    def __init__(self, workers=1, *, process_priority="normal", max_pending=None, max_snapshot_bytes=128 * 1024 * 1024):
        if isinstance(workers, bool) or not isinstance(workers, int) or workers not in RENDER_WORKER_CHOICES:
            raise ValueError(f"Render workers must be one of {RENDER_WORKER_CHOICES}")
        if process_priority not in PROCESS_PRIORITY_CHOICES:
            raise ValueError(f"Process priority must be one of {PROCESS_PRIORITY_CHOICES}")
        self.process_priority = process_priority
        self.workers = workers
        self.max_pending = max_pending if max_pending is not None else 2 * workers
        if self.max_pending < 1 or max_snapshot_bytes < 1:
            raise ValueError("Render queue limits must be positive")
        self.max_snapshot_bytes = max_snapshot_bytes
        self.pool = None
        self.lifetime = None
        self.pending = deque()
        self.pending_bytes = 0
        self.peak_pending = 0
        self.peak_snapshot_bytes = 0
        self.serialization_seconds = 0.0
        self.wait_seconds = 0.0
        self.completed = []
        self.patches = []
        self.meshes = OrderedDict()
        self.next_mesh_token = 0
        self.entered = False

    def __enter__(self):
        global _active
        if _active is not None:
            raise RuntimeError("A render execution context is already active")
        if self.entered:
            raise RuntimeError("Render execution contexts are single-use")
        self.entered = True
        if self.workers > 1:
            self.lifetime = WorkerLifetime()
            try:
                self.pool = ProcessPoolExecutor(max_workers=self.workers,
                    mp_context=multiprocessing.get_context("spawn"), initializer=_initialize_worker,
                    initargs=(self.lifetime.name, os.getpid(), self.process_priority))
            except BaseException:
                self.lifetime.close()
                raise
        _active = self
        return self

    def install_runner_hooks(self):
        if self.pool is None:
            return
        for name, module in list(sys.modules.items()):
            if not name.startswith("run_long_evolution_v") or module is None:
                continue
            for alias, function in list(vars(module).items()):
                if (inspect.isfunction(function) and function.__module__.startswith("visualization.")
                        and function.__name__.startswith("save_")
                        and (function.__module__, function.__name__) not in _synchronous_renderers):
                    @functools.wraps(function)
                    def dispatch(*args, _function=function, **kwargs):
                        self.submit(_function, *args, **kwargs)
                    self.patches.append((module, alias, function))
                    setattr(module, alias, dispatch)

    def _mesh_slots(self, args, kwargs):
        from tectonics.mesh import SphereMesh
        slots = []
        for kind, items in (("arg", enumerate(args)), ("kwarg", kwargs.items())):
            for slot, value in items:
                if not isinstance(value, SphereMesh):
                    continue
                identity = id(value)
                if identity not in self.meshes:
                    self.next_mesh_token += 1
                    self.meshes[identity] = (value, self.next_mesh_token)
                    if len(self.meshes) > 2:
                        self.meshes.popitem(last=False)
                slots.append((kind, slot, self.meshes[identity][1]))
        return slots

    def submit(self, function, *args, **kwargs):
        if self.pool is None:
            return function(*args, **kwargs)
        # Repeated writes by the same renderer to the same destination must
        # retain submission order (e.g. the duplicated final animation frame).
        signature = inspect.signature(function).bind(*args, **kwargs)
        destinations = tuple(str(Path(value).resolve()) for name, value in signature.arguments.items()
                             if name in {"path", "out", "output", "out_dir", "outdir", "output_dir"})
        if not destinations:
            raise ValueError(f"Renderer has no declared output destination: {function.__name__}")
        key = (function.__module__, function.__name__, destinations)
        while self.pending and (len(self.pending) >= self.max_pending
                                or any(item[2] == key for item in self.pending)):
            self._finish_oldest()
        began = perf_counter()
        slots = self._mesh_slots(args, kwargs)
        # Serialize NOW, not later on ProcessPoolExecutor's feeder thread.
        payload = pickle.dumps((args, kwargs), protocol=pickle.HIGHEST_PROTOCOL)
        self.serialization_seconds += perf_counter() - began
        if len(payload) > self.max_snapshot_bytes:
            raise ValueError("One render snapshot exceeds the configured memory budget")
        while self.pending and self.pending_bytes + len(payload) > self.max_snapshot_bytes:
            self._finish_oldest()
        future = self.pool.submit(_render_job, function.__module__, function.__name__, payload, slots)
        self.pending.append((future, len(payload), key))
        self.pending_bytes += len(payload)
        self.peak_pending = max(self.peak_pending, len(self.pending))
        self.peak_snapshot_bytes = max(self.peak_snapshot_bytes, self.pending_bytes)

    def _finish_oldest(self):
        future, size, key = self.pending.popleft()
        started = perf_counter()
        try:
            self.completed.append(future.result())
        except Exception as error:
            raise RuntimeError(f"Render job failed: {key[0]}.{key[1]}") from error
        finally:
            self.wait_seconds += perf_counter() - started
            self.pending_bytes -= size

    def flush(self):
        while self.pending:
            self._finish_oldest()

    def report(self):
        return {"render_workers": self.workers, "jobs_completed": len(self.completed),
                "requested_priority": self.process_priority, "coordinator_priority": read_process_priority(),
                "peak_pending": self.peak_pending, "peak_snapshot_bytes": self.peak_snapshot_bytes,
                "serialization_seconds": self.serialization_seconds, "wait_seconds": self.wait_seconds,
                "jobs": self.completed}

    def __exit__(self, exc_type, exc, traceback):
        global _active
        try:
            if exc_type is None:
                self.flush()
        finally:
            try:
                if self.pool is not None:
                    self.pool.shutdown(wait=True, cancel_futures=True)
            finally:
                if self.lifetime is not None:
                    self.lifetime.close()
                for module, alias, original in reversed(self.patches):
                    setattr(module, alias, original)
                self.pending.clear()
                self.pending_bytes = 0
                self.meshes.clear()
                _active = None


def flush_rendering():
    """Barrier before discovering frame files; a no-op in the original runner."""
    if _active is not None:
        _active.flush()
