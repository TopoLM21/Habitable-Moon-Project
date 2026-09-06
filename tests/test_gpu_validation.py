"""CPU-only guards for the GPU validation harness; no CUDA or model runs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from analysis import validate_gpu_surface as validation


def _checkpoint(path: Path, value: float = 1.0, *, time: float = 700.0, history=None) -> Path:
    path.mkdir()
    np.savez(path / "state.npz", field=np.array([value], dtype=np.float64))
    (path / "meta.json").write_text(
        json.dumps({"time_myr": time, "history": [] if history is None else history}), encoding="utf-8")
    return path


@pytest.fixture
def harness_args(tmp_path, monkeypatch):
    python = tmp_path / "python.exe"
    python.touch()
    config = tmp_path / "config.yaml"
    config.write_text("test: true\n", encoding="utf-8")
    project = tmp_path / "cpu-project"
    project.mkdir()
    (project / "run_long_evolution_v131_cpu.py").touch()
    resume = _checkpoint(tmp_path / "input")
    results = tmp_path / "results" / "gpu_surface"
    monkeypatch.setattr(validation, "RESULTS", results)
    return ["--cpu-python", str(python), "--gpu-python", str(python),
            "--cpu-project-root", str(project), "--config", str(config),
            "--resume", str(resume), "--end-time", "720"]


@pytest.mark.parametrize("target", ["outside", "base", "existing"])
def test_output_guard_rejects_unsafe_or_existing_directory(harness_args, tmp_path, target):
    results = validation.RESULTS
    if target == "outside":
        output = tmp_path / "elsewhere"
    elif target == "base":
        output = results
    else:
        output = results / "existing"
        output.mkdir(parents=True)
    with pytest.raises(SystemExit) as exc:
        validation.parse_args([*harness_args, "--output", str(output)])
    assert exc.value.code == 2


@pytest.mark.parametrize("extra", [
    ["--dt", "3"], ["--dt", "0"], ["--dt", "nan"],
    ["--midpoint", "709"], ["--midpoint", "720"],
    ["--end-time", "700"], ["--end-time", "nan"],
    ["--end-time", "724", "--dt", "8", "--frames"],
])
def test_simulation_and_frame_intervals_must_align(harness_args, extra):
    with pytest.raises(SystemExit) as exc:
        validation.parse_args([*harness_args, *extra])
    assert exc.value.code == 2


def test_valid_independent_cpu_root_midpoint_and_output(harness_args):
    output = validation.RESULTS / "new_case"
    args = validation.parse_args([*harness_args, "--midpoint", "708", "--frames", "--output", str(output)])
    assert args.start_time == 700.0 and args.midpoint == 708.0
    assert args.cpu_project_root.name == "cpu-project"
    assert args.output == output.resolve()
    assert not output.exists()


@pytest.mark.parametrize("report", [
    {},
    {"backend": "cpu", "surface_pipeline_enabled": True, "surface_pipeline": {"calls": 1}},
    {"backend": "cuda", "surface_pipeline_enabled": False, "surface_pipeline": {"calls": 1}},
    {"backend": "cuda", "surface_pipeline_enabled": True, "surface_pipeline": None},
    {"backend": "cuda", "surface_pipeline_enabled": True, "surface_pipeline": {"calls": 0}},
])
def test_gpu_provenance_requires_enabled_pipeline_and_actual_calls(report):
    with pytest.raises(RuntimeError, match="positive call count"):
        validation.require_surface_execution(report)


def test_gpu_provenance_accepts_executed_surface_pipeline():
    validation.require_surface_execution({
        "backend": "cuda", "surface_pipeline_enabled": True, "surface_pipeline": {"calls": 5}})


def test_child_environment_preserves_caches_and_external_pythonpath(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "external_dependencies")
    monkeypatch.setenv("CUPY_CACHE_DIR", "existing_cupy_cache")
    monkeypatch.setenv("CUDA_CACHE_PATH", "existing_cuda_cache")
    environment = validation.child_environment(tmp_path / "output", tmp_path / "cpu-project")
    assert environment["PYTHONPATH"] == str(tmp_path / "cpu-project") + validation.os.pathsep + "external_dependencies"
    assert environment["CUPY_CACHE_DIR"] == "existing_cupy_cache"
    assert environment["CUDA_CACHE_PATH"] == "existing_cuda_cache"


def test_exact_comparison_detects_signed_zero_and_historical_metadata(tmp_path):
    reference = _checkpoint(tmp_path / "reference", 0.0, history=[{"ledger": 1}, {"ledger": 2}])
    actual = _checkpoint(tmp_path / "actual", -0.0, history=[{"ledger": 7}, {"ledger": 2}])
    comparison = validation.compare_checkpoints(reference, actual)
    assert not comparison["exact"]
    assert set(comparison["differences"]) == {"field", "metadata/history"}


@pytest.mark.parametrize("failure", ["mismatch", "child_process"])
def test_failed_candidate_writes_summary_and_does_not_claim_speedup(harness_args, tmp_path, monkeypatch, failure):
    reference = _checkpoint(tmp_path / "reference", time=720.0)
    actual = _checkpoint(tmp_path / "actual", 2.0, time=720.0)
    output = validation.RESULTS / "failed_case"

    def fake_run(args, output, row, resume, end_time):
        row.update(wall_seconds=0.01, returncode=42 if failure == "child_process" else 0)
        if failure == "child_process":
            raise RuntimeError("child process exited 42")
        row["checkpoint"] = str(reference if row["backend"] == "cpu" else actual)

    monkeypatch.setattr(validation, "run_case", fake_run)
    assert validation.main([*harness_args, "--output", str(output)]) == 1
    report = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert report["status"] == "candidate_failed" and report["input_unchanged"]
    assert "observed_cpu_over_gpu_wall_ratio" not in report
    if failure == "mismatch":
        assert report["comparisons"][0]["exact"] is False
    else:
        assert report["runs"][0]["returncode"] == 42
