"""CUDA ownership/validation contracts exercised without a GPU or CuPy."""
from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from tectonics import gpu_runtime
from tectonics.mesh import build_icosphere
from tectonics.sediment import SedimentParameters


class DeviceArray:
    def __init__(self, values, device=0):
        self.values = np.asarray(values)
        self.dtype = self.values.dtype
        self.shape = self.values.shape
        self.flags = self.values.flags
        self.nbytes = self.values.nbytes
        self.device = SimpleNamespace(id=device)
        self.data = SimpleNamespace(ptr=self.values.ctypes.data)


class FakeCupy:
    """Only device selection/allocation; numerical kernels must never run."""
    __version__ = "test"
    ndarray = DeviceArray
    float64 = np.float64

    def __init__(self):
        self.current = 1
        self.compile_error = None
        self.synchronize_error = None
        self.restore_error = None
        runtime = SimpleNamespace(
            getDeviceCount=lambda: 2,
            getDevice=lambda: self.current,
            getDeviceProperties=lambda _: {"name": b"fake", "major": 8, "minor": 9},
            runtimeGetVersion=lambda: 12000,
        )
        self.cuda = SimpleNamespace(
            runtime=runtime,
            Device=lambda index: SimpleNamespace(use=lambda: self.use(index)),
            get_current_stream=lambda: SimpleNamespace(synchronize=self.synchronize),
        )

    def use(self, index):
        if index == 1 and self.restore_error is not None:
            raise self.restore_error
        self.current = index

    def synchronize(self):
        if self.synchronize_error is not None:
            raise self.synchronize_error

    def RawModule(self, **_kwargs):
        if self.compile_error is not None:
            raise self.compile_error
        return SimpleNamespace(get_function=lambda _: self.unexpected_kernel)

    @staticmethod
    def unexpected_kernel(*_args):
        raise AssertionError("Validation tests must not launch numerical kernels")

    def asarray(self, values):
        return DeviceArray(np.array(values, copy=True), device=self.current)


@pytest.fixture
def fake_cuda(monkeypatch):
    cp = FakeCupy()
    monkeypatch.setitem(sys.modules, "cupy", cp)
    assert gpu_runtime.current_execution() is None
    yield cp
    assert gpu_runtime.current_execution() is None


def test_import_does_not_load_cupy_or_surface_module():
    subprocess.run(
        [sys.executable, "-c", "import sys; import tectonics.gpu_runtime; "
         "assert 'cupy' not in sys.modules; "
         "assert 'tectonics.gpu_surface' not in sys.modules"],
        check=True, capture_output=True, text=True,
    )


@pytest.mark.parametrize("device", [-1, True, 1.2, "0"])
def test_device_index_rejected_without_cuda(device):
    with pytest.raises(ValueError, match="non-negative integer"):
        gpu_runtime.GpuExecution(device)


@pytest.mark.parametrize("corruption", ["negative", "past_end", "overflow", "duplicate", "self", "float", "shape"])
def test_geometry_rejects_malformed_connectivity_before_upload(fake_cuda, corruption):
    mesh = build_icosphere(0)
    neighbors = np.array(mesh.neighbors, dtype=np.int64, copy=True)
    if corruption == "negative":
        neighbors[0, 0] = -1
    elif corruption == "past_end":
        neighbors[0, 0] = mesh.cell_count
    elif corruption == "overflow":
        neighbors[0, 0] = 2**32 + int(neighbors[0, 0])
    elif corruption == "duplicate":
        neighbors[0, 0] = neighbors[0, 1]
    elif corruption == "self":
        neighbors[0, 0] = 0
    elif corruption == "float":
        neighbors = neighbors.astype(np.float64)
    elif corruption == "shape":
        neighbors = neighbors[:, :2]
    malformed = SimpleNamespace(cell_count=mesh.cell_count, neighbors=neighbors)
    with gpu_runtime.GpuExecution() as execution:
        with pytest.raises(ValueError, match="CUDA routing"):
            execution.geometry(malformed)
        assert execution.static_transfer_bytes == 0
        assert not execution._meshes


def test_geometry_reuses_upload_and_reselects_own_device(fake_cuda):
    mesh = build_icosphere(0)
    with gpu_runtime.GpuExecution() as execution:
        item = execution.geometry(mesh)
        fake_cuda.use(1)
        assert execution.geometry(mesh) is item
        assert fake_cuda.current == 0
        assert execution.static_transfer_bytes == mesh.cell_count * 3 * 4 * 2
        incoming = item.incoming.values
        sources = incoming // 3
        positions = incoming % 3
        np.testing.assert_array_equal(np.sort(sources, axis=1), sources)
        np.testing.assert_array_equal(np.asarray(mesh.neighbors)[sources, positions],
                                      np.broadcast_to(np.arange(mesh.cell_count)[:, None], sources.shape))
    assert fake_cuda.current == 1


@pytest.mark.parametrize("corruption", ["overlap", "wrong_device", "float32", "strided", "scratch_overlap", "host_array"])
def test_device_route_rejects_invalid_buffers_before_kernel(fake_cuda, corruption):
    mesh = build_icosphere(0)
    n = mesh.cell_count
    z = DeviceArray(np.zeros(n))
    sed = DeviceArray(np.ones(n))
    mobile = DeviceArray(np.full(n, 2.0))
    scratch = None
    if corruption == "overlap":
        sed = z
    elif corruption == "wrong_device":
        mobile = DeviceArray(np.zeros(n), device=1)
    elif corruption == "float32":
        mobile = DeviceArray(np.zeros(n, dtype=np.float32))
    elif corruption == "strided":
        mobile = DeviceArray(np.zeros(n * 2)[::2])
    elif corruption == "scratch_overlap":
        scratch = mobile
    elif corruption == "host_array":
        mobile = np.zeros(n)
    with gpu_runtime.GpuExecution() as execution:
        with pytest.raises(ValueError, match="CUDA routing"):
            execution.route_mobile_device(mesh, z, sed, mobile, SedimentParameters(), 0., next_mob=scratch)
        assert execution.routing_calls == 0


def test_initialization_failure_cleans_and_restores_device(fake_cuda):
    fake_cuda.compile_error = RuntimeError("compiler failed")
    execution = gpu_runtime.GpuExecution()
    with pytest.raises(gpu_runtime.GpuUnavailableError, match="compiler failed"):
        execution.__enter__()
    assert fake_cuda.current == 1
    assert execution.cp is None
    assert execution._module is None
    assert execution._previous_device_id is None


@pytest.mark.parametrize("failure", ["body", "synchronize", "restore"])
def test_exit_failure_always_drops_gpu_references(fake_cuda, failure):
    execution = gpu_runtime.GpuExecution(surface_pipeline=True)
    with pytest.raises(RuntimeError, match="expected failure"):
        with execution:
            execution.geometry(build_icosphere(0))
            execution._surface_workspace = SimpleNamespace(report=lambda: {"calls": 7})
            if failure == "synchronize":
                fake_cuda.synchronize_error = RuntimeError("expected failure")
            elif failure == "restore":
                fake_cuda.restore_error = RuntimeError("expected failure")
            else:
                raise RuntimeError("expected failure")
    assert execution.cp is None
    assert execution._surface_workspace is None
    assert execution._module is None
    assert not execution._meshes
    assert execution._previous_device_id is None
    assert execution.report()["surface_pipeline"] == {"calls": 7}


def test_workspace_cache_replacement_and_exit_report(fake_cuda, monkeypatch):
    created = []

    def factory(execution, mesh):
        item = SimpleNamespace(mesh=mesh, report=lambda: {"cells": mesh.cell_count})
        created.append(item)
        return item

    monkeypatch.setitem(sys.modules, "tectonics.gpu_surface", SimpleNamespace(SurfaceWorkspace=factory))
    mesh_a, mesh_b = build_icosphere(0), build_icosphere(1)
    with gpu_runtime.GpuExecution(surface_pipeline=True) as execution:
        first = execution.surface_workspace(mesh_a)
        assert execution.surface_workspace(mesh_a) is first
        second = execution.surface_workspace(mesh_b)
        assert second is not first
        assert len(created) == 2
        assert execution.report()["surface_pipeline"] == {"cells": mesh_b.cell_count}
    assert execution.report()["surface_pipeline"] == {"cells": mesh_b.cell_count}
    assert "resets" in execution.report()["surface_pipeline_report_scope"]
    with pytest.raises(RuntimeError, match="not active"):
        execution.surface_workspace(mesh_b)


def test_disabled_surface_does_not_allocate_workspace(fake_cuda):
    with gpu_runtime.GpuExecution() as execution:
        with pytest.raises(RuntimeError, match="not enabled"):
            execution.surface_workspace(build_icosphere(0))
        assert execution.report()["surface_pipeline"] is None


def test_retained_workspace_cannot_touch_device_after_context_exit(fake_cuda):
    from tectonics.gpu_surface import SurfaceWorkspace

    # Avoid allocations: the closed-context check must precede all buffer access.
    workspace = SurfaceWorkspace.__new__(SurfaceWorkspace)
    with gpu_runtime.GpuExecution(surface_pipeline=True) as execution:
        workspace.execution = execution
    with pytest.raises(RuntimeError, match="not active"):
        workspace.calculate(None, None, None, None, None, 1., SedimentParameters(), None, 0.)
