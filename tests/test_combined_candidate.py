"""Combined research candidates: CPU-only context and mocked CLI safeguards."""
from contextlib import ExitStack
import json

import pytest
from scipy.sparse import csr_matrix

import analysis.run_assignment_candidate as assignment_runner
import analysis.validate_assignment_candidate as validation
from analysis.probe_assignment_columns import same
from analysis.run_boundary_candidate import BoundaryCandidateContext
import tectonics.dynamics as dynamics
import tectonics.transport as transport
from tectonics.cpu_runtime import CpuExecution
from test_assignment_validation import harness, report
from test_boundary_candidate_runner import world
from test_gpu_stage_profile import assert_exact


def graph():
    return csr_matrix(([1., 2.], ([0, 1], [3, 8])), shape=(2, 20))


def dynamics_args():
    mesh, state, system = world()
    return (mesh, state, system, system, 5287.0, 4.0, 4.0, 1.0,
            dynamics.DynamicsParameters())


@pytest.mark.parametrize("boundary_first", [False, True])
def test_real_contexts_restore_after_error_and_can_be_reused_separately(boundary_first):
    # Runner aliases must exist before the boundary context is installed.
    import run_long_evolution_v131
    import run_long_evolution_v123 as base

    original_solver = transport.min_weight_full_bipartite_matching
    original_dynamics = dynamics.update_plate_dynamics
    original_alias = base.update_plate_dynamics
    args = dynamics_args()
    assignment = assignment_runner.AssignmentCandidate()
    boundary = BoundaryCandidateContext()
    with CpuExecution(numeric_kernels=True):
        expected = original_dynamics(*args)
        with pytest.raises(ValueError, match="combined failure"):
            with ExitStack() as stack:
                for context in ((boundary, assignment) if boundary_first else (assignment, boundary)):
                    stack.enter_context(context)
                assert base.update_plate_dynamics is dynamics.update_plate_dynamics
                assert same(transport.min_weight_full_bipartite_matching(graph()), original_solver(graph()))
                assert_exact(expected, base.update_plate_dynamics(*args))
                assert assignment.calls == boundary.calls == 1
                raise ValueError("combined failure")

        assert transport.min_weight_full_bipartite_matching is original_solver
        assert dynamics.update_plate_dynamics is original_dynamics
        assert base.update_plate_dynamics is original_alias
        assert assignment.original is None
        assert boundary.geometry is None and not boundary._installed
        assert boundary.replacements == []

        with assignment:
            assert dynamics.update_plate_dynamics is original_dynamics
            assert same(transport.min_weight_full_bipartite_matching(graph()), original_solver(graph()))
        with boundary:
            assert transport.min_weight_full_bipartite_matching is original_solver
            assert_exact(expected, base.update_plate_dynamics(*args))
        assert assignment.calls == boundary.calls == 2
    assert transport.min_weight_full_bipartite_matching is original_solver
    assert dynamics.update_plate_dynamics is original_dynamics
    assert base.update_plate_dynamics is original_alias


@pytest.fixture
def mock_runner(monkeypatch, tmp_path):
    import run_long_evolution_v131
    import run_long_evolution_v123 as base
    import run_long_evolution_v131_gpu as gpu_runner

    original_solver = transport.min_weight_full_bipartite_matching
    original_dynamics = dynamics.update_plate_dynamics
    args = dynamics_args()
    data = {"calls": 0, "boundary": True, "failure": None,
            "output": tmp_path / "output", "reference": tmp_path / "reference"}

    def run():
        data["calls"] += 1
        assert "--reference" not in assignment_runner.sys.argv
        assert "--with-boundary" not in assignment_runner.sys.argv
        assert "--gpu-surface" in assignment_runner.sys.argv
        assert transport.min_weight_full_bipartite_matching is not original_solver
        assert (dynamics.update_plate_dynamics is not original_dynamics) == data["boundary"]
        assert base.update_plate_dynamics is dynamics.update_plate_dynamics
        assert same(transport.min_weight_full_bipartite_matching(graph()), original_solver(graph()))
        if data["failure"] == "runner":
            raise ValueError("mock runner failure")
        if data["failure"] != "boundary_missing":
            with CpuExecution(numeric_kernels=True):
                assert_exact(original_dynamics(*args), base.update_plate_dynamics(*args))
        data["output"].mkdir()
        (data["output"] / "render_timings.json").write_text(
            json.dumps({"existing": "retained"}), encoding="utf-8")

    def compare(left, right):
        assert left == data["reference"] and right == data["output"] / "checkpoint"
        return {"exact": True, "arrays_compared": 62}

    monkeypatch.setattr(gpu_runner, "main", run)
    monkeypatch.setattr(assignment_runner, "compare_checkpoints", compare)
    monkeypatch.setattr(assignment_runner.sys, "argv", [
        "run_assignment_candidate.py", "--reference", str(data["reference"]),
        "--output", str(data["output"]), "--gpu-surface", "--with-boundary",
    ])
    yield data
    assert transport.min_weight_full_bipartite_matching is original_solver
    assert dynamics.update_plate_dynamics is original_dynamics
    assert base.update_plate_dynamics is original_dynamics


@pytest.mark.parametrize("with_boundary", [False, True])
def test_cli_runs_once_and_records_active_boundary_cache(mock_runner, monkeypatch, with_boundary):
    mock_runner["boundary"] = with_boundary
    if not with_boundary:
        monkeypatch.setattr(assignment_runner.sys, "argv", assignment_runner.sys.argv[:-1])
    assignment_runner.main()
    result = json.loads((mock_runner["output"] / "render_timings.json").read_text(encoding="utf-8"))
    assert mock_runner["calls"] == 1 and result["existing"] == "retained"
    assert result["assignment_candidate"]["backend"] == "cpu_compact_columns"
    assert result["assignment_candidate"]["calls"] == 1
    assert result["assignment_candidate"]["comparison"]["exact"]
    if with_boundary:
        assert result["boundary_candidate"]["backend"] == "prepared_cpu_boundary_forces"
        assert result["boundary_candidate"]["calls"] == 1
        assert result["boundary_candidate"]["cached_edges"] > 0
    else:
        assert "boundary_candidate" not in result


@pytest.mark.parametrize("failure", ["runner", "boundary_missing"])
def test_cli_failure_restores_both_contexts(mock_runner, failure):
    mock_runner["failure"] = failure
    with pytest.raises((ValueError, RuntimeError)):
        assignment_runner.main()
    assert mock_runner["calls"] == 1


@pytest.fixture
def combined_harness(harness, monkeypatch):
    child = validation.subprocess.run

    def combined_child(command, **kwargs):
        result = child(command, **kwargs)
        candidate = "--reference" in command
        assert ("--with-boundary" in command) == candidate
        if candidate and harness.failure != "boundary_missing":
            path = harness.output / ("candidate_" + command[command.index("--output") + 1].rsplit("_", 1)[1])
            timing_path = path / "render_timings.json"
            execution = json.loads(timing_path.read_text(encoding="utf-8"))
            execution["boundary_candidate"] = {
                "backend": ("incorrect" if harness.failure == "boundary_backend"
                            else "prepared_cpu_boundary_forces"),
                "calls": {"boundary_zero": 0, "boundary_partial": 49,
                          "boundary_excess": 51}.get(harness.failure, 50),
            }
            timing_path.write_text(json.dumps(execution), encoding="utf-8")
        return result

    monkeypatch.setattr(validation.subprocess, "run", combined_child)
    monkeypatch.setattr(validation.sys, "argv", [*validation.sys.argv, "--with-boundary"])
    return harness


def test_combined_harness_enables_boundary_only_for_candidates(combined_harness):
    assert validation.main() == 0
    result = report(combined_harness)
    assert result["boundary_candidate_enabled"] is True
    assert result["status"] == "exact_validation_passed"
    assert len(result["runs"]) == 4
    assert result["observed_time_reduction_percent"] == pytest.approx(20.0)
    for row in result["runs"]:
        assert ("boundary_candidate" in row["execution_report"]) == (row["mode"] == "candidate")


@pytest.mark.parametrize("failure", [
    "boundary_missing", "boundary_backend", "boundary_zero", "boundary_partial", "boundary_excess",
])
def test_combined_harness_rejects_missing_or_incomplete_boundary(combined_harness, failure):
    combined_harness.failure = failure
    assert validation.main() == 1
    result = report(combined_harness)
    assert result["status"] == "failed" and "boundary" in result["error"].lower()
    assert "median_wall_seconds" not in result
    assert "observed_time_reduction_percent" not in result
