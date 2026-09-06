"""Mocked benchmark orchestration: never launches a process or accesses CUDA."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import analysis.benchmark_boundary_candidate as benchmark


@pytest.fixture
def harness(monkeypatch, tmp_path):
    config = tmp_path / "config.yaml"
    resume, reference = tmp_path / "resume", tmp_path / "reference"
    output = tmp_path / "results" / "gpu_surface" / "benchmark"
    for path, time in ((resume, 700.0), (reference, 900.0)):
        path.mkdir()
        (path / "meta.json").write_text(json.dumps({"time_myr": time}), encoding="utf-8")
    data = SimpleNamespace(mode=None, failure=None, commands=[], comparisons=[], clock=0.0,
                           output=output, resume=resume, reference=reference,
                           telemetry_calls=0, hash_calls=0)

    def source_hash(path):
        data.hash_calls += 1
        if data.failure == "source_changed" and len(data.commands) == 6:
            return "changed"
        if data.failure == "source_unreadable" and len(data.commands) == 6:
            raise OSError("source became unreadable during benchmark")
        return str(path)

    def gpu_telemetry(device):
        data.telemetry_calls += 1
        return {"device": device, "mock": True}

    def compare(left, right):
        data.comparisons.append((left, right))
        assert left == reference  # Every run must use the independent reference.
        return {"exact": data.failure != "checkpoint_mismatch", "arrays_compared": 62,
                "differences": [] if data.failure != "checkpoint_mismatch" else ["changed array"]}

    def child(command, **kwargs):
        data.commands.append(command)
        mode = "boundary_candidate" if Path(command[1]).name == "run_boundary_candidate.py" else "gpu_surface"
        data.mode = mode
        data.clock += 8.0 if mode == "boundary_candidate" else 10.0
        assert kwargs["cwd"] == tmp_path
        assert kwargs["env"] == {"mock": "environment"}
        case = Path(command[command.index("--output") + 1])
        result = {"gpu_execution": {"backend": "cuda", "surface_pipeline_enabled": True,
                                    "surface_pipeline": {"calls": 50}}}
        if data.failure == "surface_provenance":
            result["gpu_execution"]["surface_pipeline"]["calls"] = 0
        elif data.failure == "surface_partial":
            result["gpu_execution"]["surface_pipeline"]["calls"] = 49
        if mode == "boundary_candidate":
            result["boundary_candidate"] = {"backend": "prepared_cpu_boundary_forces", "calls": 50}
            if data.failure == "candidate_missing":
                del result["boundary_candidate"]
            elif data.failure == "candidate_partial":
                result["boundary_candidate"]["calls"] = 49
        (case / "render_timings.json").write_text(json.dumps(result), encoding="utf-8")
        return SimpleNamespace(returncode=7 if data.failure == "child_failed" else 0)

    monkeypatch.setattr(benchmark, "ROOT", tmp_path)
    monkeypatch.setattr(benchmark, "sha256_file", source_hash)
    monkeypatch.setattr(benchmark, "checkpoint_hashes", lambda path: {"state.npz": str(path)})
    monkeypatch.setattr(benchmark, "telemetry", gpu_telemetry)
    monkeypatch.setattr(benchmark, "child_environment", lambda *_: {"mock": "environment"})
    monkeypatch.setattr(benchmark, "compare_checkpoints", compare)
    monkeypatch.setattr(benchmark, "perf_counter", lambda: data.clock)
    monkeypatch.setattr(benchmark.subprocess, "run", child)
    monkeypatch.setattr(benchmark.sys, "argv", [
        "benchmark_boundary_candidate.py", "--config", str(config), "--resume", str(resume),
        "--reference", str(reference), "--output", str(output), "--end-time", "900",
        "--dt", "4", "--repeat", "3",
    ])
    return data


def report(harness):
    return json.loads((harness.output / "summary.json").read_text(encoding="utf-8"))


def test_whole_process_times_alternating_order_and_independent_reference(harness):
    assert benchmark.main() == 0
    result = report(harness)
    assert [row["mode"] for row in result["runs"]] == [
        "gpu_surface", "boundary_candidate", "boundary_candidate", "gpu_surface",
        "gpu_surface", "boundary_candidate",
    ]
    assert [row["repeat"] for row in result["runs"]] == [1, 1, 2, 2, 3, 3]
    assert len(harness.comparisons) == 6
    assert harness.telemetry_calls == 12
    assert harness.hash_calls == 16  # Config + seven code files, before and after.
    assert result["median_wall_seconds"] == {"gpu_surface": 10.0, "boundary_candidate": 8.0}
    assert result["observed_time_reduction_percent"] == pytest.approx(20.0)
    assert result["status"] == "exact_validation_passed"
    assert result["inputs_and_sources_unchanged"] is True
    for command in harness.commands:
        assert command[command.index("--resume") + 1] == str(harness.resume)
        assert "--gpu-surface" in command and "--cell-kernels" in command


@pytest.mark.parametrize("failure", ["checkpoint_mismatch", "surface_provenance", "surface_partial", "candidate_missing",
                                     "candidate_partial", "child_failed", "source_changed"])
def test_failure_never_publishes_speedup(harness, failure):
    harness.failure = failure
    assert benchmark.main() == 1
    result = report(harness)
    assert result["status"] == "failed"
    assert result["error"]
    assert "median_wall_seconds" not in result
    assert "observed_time_reduction_percent" not in result
    assert "fingerprints_after" in result
    assert result["inputs_and_sources_unchanged"] is (failure != "source_changed")
    if failure == "source_changed":
        # The final hash barrier must reject even six successful exact runs.
        assert len(result["runs"]) == 6
        assert all(row["comparison"]["exact"] for row in result["runs"])


def test_unreadable_final_fingerprint_persists_failure_without_speedup(harness):
    harness.failure = "source_unreadable"
    assert benchmark.main() == 1
    result = report(harness)
    assert len(result["runs"]) == 6
    assert result["status"] == "failed"
    assert "unreadable" in result["error"]
    assert result["inputs_and_sources_unchanged"] is False
    assert "median_wall_seconds" not in result
    assert "observed_time_reduction_percent" not in result


def test_existing_output_is_rejected_without_subprocess(harness):
    harness.output.mkdir(parents=True)
    marker = harness.output / "summary.json"
    marker.write_text("existing result", encoding="utf-8")
    with pytest.raises(SystemExit):
        benchmark.main()
    assert harness.commands == []
    assert marker.read_text(encoding="utf-8") == "existing result"
