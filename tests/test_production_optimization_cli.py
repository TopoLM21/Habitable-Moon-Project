"""Production opt-in switches and reports, without CUDA or simulation work."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

import run_long_evolution_v131_cpu as cpu_runner
import run_long_evolution_v131_gpu as gpu_runner
from tectonics import cpu_runtime


CASES = [
    pytest.param([], False, False, id="defaults-off"),
    pytest.param(["--no-assignment-columns", "--no-boundary-forces"], False, False,
                 id="explicit-off-off"),
    pytest.param(["--assignment-columns", "--no-boundary-forces"], True, False,
                 id="assignment-only"),
    pytest.param(["--no-assignment-columns", "--boundary-forces"], False, True,
                 id="boundary-only"),
    pytest.param(["--assignment-columns", "--boundary-forces"], True, True,
                 id="both-on"),
    pytest.param(["--assignment-columns", "--boundary-forces", "--no-assignment-columns",
                  "--no-boundary-forces"], False, False, id="last-switch-opts-out"),
    pytest.param(["--no-assignment-columns", "--no-boundary-forces", "--assignment-columns",
                  "--boundary-forces"], True, True, id="last-switch-opts-in"),
]


def _mock_runner_dependencies(monkeypatch, *, fail=False):
    """Keep the real parsers/CpuExecution and replace external work only."""
    observed = {"gpu_active": False, "render_active": False}

    class Rendering:
        def __init__(self, workers, **kwargs):
            observed["render_options"] = (workers, kwargs)

        def __enter__(self):
            observed["render_active"] = True
            return self

        def install_runner_hooks(self):
            observed["render_hooks"] = True

        def __exit__(self, *_exc):
            observed["render_active"] = False

        def report(self):
            return {"render_marker": "preserved"}

    class Gpu:
        device_name = "mock device; no CUDA"

        def __init__(self, device, **kwargs):
            observed["gpu_options"] = (device, kwargs)

        def __enter__(self):
            observed["gpu_active"] = True
            return self

        def __exit__(self, *_exc):
            observed["gpu_active"] = False

        def report(self):
            assert observed["gpu_active"]
            return {"gpu_marker": "preserved"}

    def simulation():
        execution = cpu_runtime.current_execution()
        assert execution is not None
        assert observed["render_active"] and observed["render_hooks"]
        observed["execution"] = execution
        observed["base_argv"] = list(sys.argv)
        observed["enabled"] = (execution.assignment_columns_enabled,
                               execution.boundary_forces_enabled)
        # Sentinel accounting tests report propagation without numerical work.
        if execution.assignment_columns_enabled:
            execution.assignment_calls = 3
            execution.assignment_seconds = 0.125
        if execution.boundary_forces_enabled:
            execution.boundary_calls = 2
            execution.boundary_seconds = 0.25
            execution.boundary_cached_edges = 30
            execution.boundary_cached_numeric_bytes = 1024
            execution._boundary_geometry = object()
        if fail:
            raise RuntimeError("mock simulation failed")

    monkeypatch.setattr(cpu_runner, "apply_process_priority", lambda value: {"policy": value})
    monkeypatch.setitem(sys.modules, "visualization.render_runtime",
                        SimpleNamespace(RenderExecution=Rendering))
    monkeypatch.setitem(sys.modules, "tectonics.gpu_runtime", SimpleNamespace(GpuExecution=Gpu))
    monkeypatch.setitem(sys.modules, "run_long_evolution_v131", SimpleNamespace(main=simulation))
    original_cpu_main = cpu_runner.main

    def observed_cpu_main():
        observed["cpu_argv"] = list(sys.argv)
        original_cpu_main()

    monkeypatch.setattr(cpu_runner, "main", observed_cpu_main)
    return observed


@pytest.mark.parametrize("entrypoint", ["cpu", "gpu"])
@pytest.mark.parametrize("flags,assignment,boundary", CASES)
def test_opt_in_switches_forward_and_report(monkeypatch, tmp_path, entrypoint,
                                            flags, assignment, boundary):
    assert cpu_runtime.current_execution() is None
    observed = _mock_runner_dependencies(monkeypatch)
    base_options = ["--output", str(tmp_path), "--resume", "input-checkpoint"]
    execution_options = ["--cpu-workers", "1", "--render-workers", "4"]
    gpu_options = ["--gpu-device", "2", "--gpu-surface"] if entrypoint == "gpu" else []
    monkeypatch.setattr(sys, "argv", [entrypoint, *flags, *execution_options,
                                     *gpu_options, *base_options])
    (gpu_runner if entrypoint == "gpu" else cpu_runner).main()

    assert observed["enabled"] == (assignment, boundary)
    assert observed["base_argv"] == ["run_long_evolution_v131.py", *base_options]
    assert observed["render_options"] == (4, {"process_priority": "normal"})
    assert cpu_runtime.current_execution() is None
    assert observed["execution"]._boundary_geometry is None
    assert not observed["render_active"] and not observed["gpu_active"]
    report = json.loads((tmp_path / "render_timings.json").read_text(encoding="utf-8"))
    assert report["render_marker"] == "preserved"
    numerical = report["numerical_execution"]
    assignment_report = numerical["assignment_columns"]
    assert assignment_report["enabled"] is assignment
    assert assignment_report["backend"] == "cpu_compact_columns"
    assert assignment_report["calls"] == (3 if assignment else 0)
    assert assignment_report["inclusive_seconds"] == (0.125 if assignment else 0.0)
    boundary_report = numerical["boundary_forces"]
    assert boundary_report["enabled"] is boundary
    assert boundary_report["backend"] == "prepared_cpu_boundary_forces"
    assert boundary_report["calls"] == (2 if boundary else 0)
    assert boundary_report["inclusive_seconds"] == (0.25 if boundary else 0.0)
    assert boundary_report["cached_edges"] == (30 if boundary else 0)
    assert boundary_report["cached_geometry_numeric_bytes"] == (1024 if boundary else 0)
    if entrypoint == "gpu":
        assert observed["gpu_options"] == (2, {"arc_painting": False, "surface_pipeline": True})
        assert observed["cpu_argv"] == [
            "run_long_evolution_v131_cpu.py",
            "--assignment-columns" if assignment else "--no-assignment-columns",
            "--boundary-forces" if boundary else "--no-boundary-forces",
            *execution_options, *base_options,
        ]
        assert report["gpu_execution"] == {"gpu_marker": "preserved"}
    else:
        assert "gpu_options" not in observed
        assert "gpu_execution" not in report


@pytest.mark.parametrize("entrypoint", ["cpu", "gpu"])
def test_runner_error_cleans_execution_contexts(monkeypatch, tmp_path, entrypoint):
    assert cpu_runtime.current_execution() is None
    observed = _mock_runner_dependencies(monkeypatch, fail=True)
    monkeypatch.setattr(sys, "argv", [entrypoint, "--assignment-columns", "--boundary-forces",
                                     "--output", str(tmp_path)])
    with pytest.raises(RuntimeError, match="mock simulation failed"):
        (gpu_runner if entrypoint == "gpu" else cpu_runner).main()
    assert observed["enabled"] == (True, True)
    assert cpu_runtime.current_execution() is None
    assert observed["execution"]._boundary_geometry is None
    assert not observed["render_active"] and not observed["gpu_active"]
    assert not (tmp_path / "render_timings.json").exists()


def test_gpu_help_lists_opt_in_and_opt_out_without_importing_cuda():
    code = """
import sys
import run_long_evolution_v131_gpu as runner
sys.argv = ['run_long_evolution_v131_gpu.py', '--help']
try:
    runner.main()
except SystemExit as exc:
    assert exc.code == 0
else:
    raise AssertionError('help did not exit')
assert 'cupy' not in sys.modules
assert 'tectonics.gpu_runtime' not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", code],
                            cwd=Path(__file__).resolve().parents[1],
                            check=True, capture_output=True, text=True, timeout=15)
    for flag in ("--assignment-columns", "--no-assignment-columns", "--boundary-forces",
                 "--no-boundary-forces"):
        assert flag in result.stdout
