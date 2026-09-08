"""Mocked assignment validation; no real child processes or CUDA access."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import analysis.validate_assignment_candidate as validation


@pytest.fixture
def harness(monkeypatch, tmp_path):
    resume, reference = tmp_path / "resume", tmp_path / "reference"
    for path, time in ((resume, 700.0), (reference, 900.0)):
        path.mkdir()
        (path / "meta.json").write_text(json.dumps({"time_myr": time}), encoding="utf-8")
    data = SimpleNamespace(failure=None, commands=[], comparisons=[], clock=0.0,
                           output=tmp_path / "results/gpu_surface/trial")

    def source_hash(path):
        return "changed" if data.failure == "hash_changed" and data.commands else str(path)

    def checkpoint(left, right):
        assert left == reference
        data.comparisons.append((left, right))
        return {"exact": data.failure != "checkpoint_mismatch", "arrays_compared": 62}

    def png(left, right):
        assert left == reference.parent
        return {"png_exact": data.failure != "png_mismatch",
                "png_count": 0 if data.failure == "png_empty" else 32}

    def child(command, **kwargs):
        data.commands.append(command)
        candidate = Path(command[1]).name == "run_assignment_candidate.py"
        assert kwargs["cwd"] == tmp_path
        assert kwargs["env"] == {"mock": "environment"}
        assert "--gpu-surface" in command and "--cell-kernels" in command
        if candidate:
            assert command[command.index("--reference") + 1] == str(reference)
        data.clock += 8.0 if candidate else 10.0
        case = Path(command[command.index("--output") + 1])
        case.mkdir()
        calls = {"surface_missing": 0, "surface_partial": 49}.get(data.failure, 50)
        execution = {"gpu_execution": {"backend": "cuda", "surface_pipeline_enabled": True,
                                       "surface_pipeline": {"calls": calls}}}
        if candidate and data.failure != "candidate_missing":
            execution["assignment_candidate"] = {"backend": "cpu_compact_columns", "calls": 439}
        (case / "render_timings.json").write_text(json.dumps(execution), encoding="utf-8")
        return SimpleNamespace(returncode=1 if data.failure == "child_failed" else 0)

    monkeypatch.setattr(validation, "ROOT", tmp_path)
    monkeypatch.setattr(validation, "__file__", str(tmp_path / "analysis/validate_assignment_candidate.py"))
    monkeypatch.setattr(validation, "sha256_file", source_hash)
    monkeypatch.setattr(validation, "checkpoint_hashes", lambda path: {"state.npz": str(path)})
    monkeypatch.setattr(validation, "telemetry", lambda device: {"mock": True})
    monkeypatch.setattr(validation, "child_environment", lambda *_: {"mock": "environment"})
    monkeypatch.setattr(validation, "compare_checkpoints", checkpoint)
    monkeypatch.setattr(validation, "compare_pngs", png)
    monkeypatch.setattr(validation, "perf_counter", lambda: data.clock)
    monkeypatch.setattr(validation.subprocess, "run", child)
    monkeypatch.setattr(validation.sys, "argv", [
        "validate_assignment_candidate.py", "--config", str(tmp_path / "config.yaml"),
        "--resume", str(resume), "--reference", str(reference), "--output", str(data.output),
        "--end-time", "900",
    ])
    return data


def report(harness):
    return json.loads((harness.output / "summary.json").read_text(encoding="utf-8"))


def test_alternating_order_and_exact_independent_reference(harness):
    assert validation.main() == 0
    result = report(harness)
    assert [(row["mode"], row["repeat"]) for row in result["runs"]] == [
        ("candidate", 2), ("original", 2), ("original", 3), ("candidate", 3),
    ]
    assert len(harness.comparisons) == 4
    assert result["status"] == "exact_validation_passed"
    assert result["inputs_and_sources_unchanged"] is True
    assert result["median_wall_seconds"] == {"original": 10.0, "candidate": 8.0}
    assert result["observed_time_reduction_percent"] == pytest.approx(20.0)


@pytest.mark.parametrize("failure", [
    "checkpoint_mismatch", "png_mismatch", "png_empty", "surface_missing",
    "surface_partial", "candidate_missing", "child_failed", "hash_changed",
])
def test_failed_validation_does_not_publish_speedup(harness, failure):
    harness.failure = failure
    assert validation.main() == 1
    result = report(harness)
    assert result["status"] == "failed" and result["error"]
    assert "median_wall_seconds" not in result
    assert "observed_time_reduction_percent" not in result
    assert result["inputs_and_sources_unchanged"] is (failure != "hash_changed")
    if failure == "hash_changed":
        assert len(result["runs"]) == 4
        assert all(row["checkpoint"]["exact"] and row["png"]["png_exact"] for row in result["runs"])


def test_candidate_only_validates_without_publishing_speedup(harness, monkeypatch):
    monkeypatch.setattr(validation.sys, "argv", [*validation.sys.argv, "--candidate-only"])
    assert validation.main() == 0
    result = report(harness)
    assert [(row["mode"], row["repeat"]) for row in result["runs"]] == [("candidate", 2), ("candidate", 3)]
    assert "median_wall_seconds" not in result
    assert "observed_time_reduction_percent" not in result


def test_existing_output_is_preserved_without_child_process(harness):
    harness.output.mkdir(parents=True)
    marker = harness.output / "summary.json"
    marker.write_text("existing result", encoding="utf-8")
    with pytest.raises(SystemExit):
        validation.main()
    assert not harness.commands
    assert marker.read_text(encoding="utf-8") == "existing result"
