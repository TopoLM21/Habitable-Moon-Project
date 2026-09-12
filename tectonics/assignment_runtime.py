"""Execution choice and bounded evidence for conservative cell assignment.

This context exists only in the diagnostic runner. Without it, transport follows
the selected runner's execution policy. No checkpoint arrays are changed here.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
import json
from pathlib import Path
from threading import Lock, current_thread, get_ident, local, main_thread
from time import monotonic
from types import SimpleNamespace


_active = None
GRAPH_BYTE_LIMIT = 16 * 1024 * 1024
LOG_BYTE_LIMIT = 2 * 1024 * 1024


def current_execution():
    return _active


def plate_context(plate_id, source_cells, time_myr):
    if _active is None:
        return nullcontext()
    return _active.plate(plate_id, source_cells, time_myr)


def _write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


class AssignmentExecution:
    def __init__(self, diagnostics, *, optimized=False):
        self.diagnostics = diagnostics
        self.optimized = bool(optimized)
        self.directory = Path(diagnostics.directory)
        self.calls = 0
        self.total_seconds = 0.0
        self._stats_lock = Lock()
        self._local = local()

    def _state(self):
        state = getattr(self._local, "state", None)
        if state is None:
            directory = self.directory
            if current_thread() is not main_thread():
                directory = directory / "workers" / f"worker-{get_ident()}"
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except OSError:
                pass  # Recording below reports an unavailable diagnostic path.
            state = SimpleNamespace(directory=directory, plate_id=None, source_cells=None, time_myr=None)
            self._local.state = state
        return state

    def _update(self, **details):
        if current_thread() is main_thread():
            self.diagnostics.update(**details)
        else:
            self.diagnostics.update_worker(**details)

    def __enter__(self):
        global _active
        if _active is not None:
            raise RuntimeError("An assignment execution context is already active")
        _active = self
        return self

    def __exit__(self, *_exc):
        global _active
        _active = None

    @contextmanager
    def plate(self, plate_id, source_cells, time_myr):
        state = self._state()
        previous = state.plate_id, state.source_cells, state.time_myr
        state.plate_id, state.source_cells, state.time_myr = int(plate_id), source_cells, float(time_myr)
        try:
            yield
        finally:
            state.plate_id, state.source_cells, state.time_myr = previous

    def _record(self, record):
        try:
            path = self._state().directory / "assignment_events.jsonl"
            if path.exists() and path.stat().st_size >= LOG_BYTE_LIMIT:
                path.replace(path.with_suffix(".previous.jsonl"))
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            _write_json(path.parent / "assignment_latest.json", record)
        except OSError as error:
            self._update(assignment_log_error=str(error)[:256])

    def _save_input(self, graph, metadata):
        """Keep one bounded, atomic graph; interrupted calls retain their input."""
        import numpy as np

        state = self._state()
        byte_count = sum(array.nbytes for array in (graph.data, graph.indices, graph.indptr))
        if state.source_cells is not None:
            byte_count += state.source_cells.nbytes
        metadata["graph_array_bytes"] = int(byte_count)
        metadata["graph_saved"] = False
        if byte_count > GRAPH_BYTE_LIMIT:
            metadata["graph_note"] = "Input exceeds the 16 MiB diagnostic capture limit"
            return
        path = state.directory / "assignment_last_input.npz"
        temporary = path.with_suffix(".npz.tmp")
        try:
            with temporary.open("wb") as handle:
                arrays = dict(data=graph.data, indices=graph.indices, indptr=graph.indptr,
                              shape=np.asarray(graph.shape, dtype=np.int64),
                              assignment_call=np.asarray(metadata["assignment_call"], dtype=np.int64))
                if state.source_cells is not None:
                    arrays["source_cells"] = state.source_cells
                np.savez(handle, **arrays)
            temporary.replace(path)
            metadata["graph_saved"] = True
            metadata["graph_file"] = str(path)
            metadata["graph_note"] = "Latest attempted graph only; match assignment_call before analysis"
        except OSError as error:
            metadata["graph_note"] = f"Could not save input: {error}"

    def match(self, graph, *, candidates, attempt, solver=None, backend=None):
        import numpy as np
        from scipy.sparse.csgraph import min_weight_full_bipartite_matching

        with self._stats_lock:
            self.calls += 1
            call = self.calls
        state = self._state()
        backend = backend or ("sparse_ssp" if self.optimized else "scipy_reference")
        metadata = dict(assignment_call=call, backend=backend, plate_id=state.plate_id,
                        time_myr=state.time_myr, source_count=int(graph.shape[0]),
                        target_count=int(graph.shape[1]), edge_count=int(graph.nnz),
                        candidates=int(candidates), attempt=int(attempt))
        label = "Быстрый подбор ячеек" if backend == "sparse_ssp" else "Подбор ячеек SciPy"
        latest_progress = {}
        infeasible_error = None

        def progress(phase, **stats):
            latest_progress.clear()
            latest_progress.update(phase=phase, **stats)
            self._update(assignment_phase=phase, **stats)

        stage = (self.diagnostics.stage if current_thread() is main_thread()
                 else self.diagnostics.worker_stage)
        with stage(label, **metadata):
            self._save_input(graph, metadata)
            self._update(graph_file=metadata.get("graph_file", ""), graph_saved=metadata["graph_saved"])
            metadata["used_targets"] = int(np.unique(graph.indices).size)
            self._update(used_targets=metadata["used_targets"])
            record = dict(metadata, timestamp=datetime.now(timezone.utc).isoformat(), status="started")
            self._record(record)
            started = monotonic()
            try:
                if solver is not None:
                    result = solver(graph, progress=progress)
                elif backend == "sparse_ssp":
                    from .assignment_sparse import sparse_minimum_matching
                    result = sparse_minimum_matching(graph, progress=progress)
                else:
                    result = min_weight_full_bipartite_matching(graph)
            except ValueError as error:
                # An infeasible candidate graph is expected: transport retries
                # with larger k. Do not register it as the run's first failure.
                infeasible_error = error
                record.update(status="infeasible", exception_type=type(error).__name__,
                              exception_message=str(error)[:512])
                self._update(assignment_phase="expand_candidates")
            except BaseException as error:
                record.update(status="failed",
                              exception_type=type(error).__name__, exception_message=str(error)[:512])
                raise
            else:
                record["status"] = "completed"
                return result
            finally:
                elapsed = monotonic() - started
                with self._stats_lock:
                    self.total_seconds += elapsed
                    total_seconds = self.total_seconds
                record.update(seconds=elapsed, total_solver_seconds=total_seconds,
                              progress=latest_progress,
                              finished_utc=datetime.now(timezone.utc).isoformat())
                self._record(record)
        if infeasible_error is not None:
            raise infeasible_error
