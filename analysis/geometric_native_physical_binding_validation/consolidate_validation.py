"""Seal the scoped gate, authentic physical control and independent audit."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from analysis.geometric_native_physical_binding_validation.run_binding_control import (
    PRIOR, digest, guard_sources, read, sealed, verify_prior_sources, write,
)
from analysis.geometric_native_physical_binding_validation.run_release_gates import TEST_PATHS


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def check_reference(path, expected):
    if digest(path) != expected:
        raise ValueError("Consolidation artifact changed: " + str(path))


def consolidate(gate_dir, control_dir, audit_path, out):
    gate_dir, control_dir, audit_path, out = (
        Path(path).resolve() for path in (gate_dir, control_dir, audit_path, out))
    gate = sealed(gate_dir / "receipt.json", "receipt_sha256")
    report = sealed(control_dir / "report.json", "report_sha256")
    audit = sealed(audit_path, "receipt_sha256")
    if any(row["status"] != "PASS" for row in (gate, report, audit)):
        raise ValueError("All three physical-binding evidence streams must pass")
    if (gate["exit_code"] != 0 or gate["changed_paths"]
            or gate["is_full_profile_execution"] or gate["original_770_tests_reexecuted"]
            or tuple(gate["test_paths"]) != TEST_PATHS):
        raise ValueError("Expected the declared five-file scoped gate with unchanged sources")
    if gate["counts"].get("passed", 0) <= 0:
        raise ValueError("Scoped gate has no passing tests")
    for key, name in (("log_sha256", "pytest.log"),
            ("sources_before_sha256", "sources_before.json"),
            ("sources_after_sha256", "sources_after.json")):
        check_reference(gate_dir / name, gate[key])
    for row in gate["per_file_results"]:
        if row["exit_code"] != 0:
            raise ValueError("A complete scoped test file failed")
        check_reference(gate_dir / row["log_path"], row["log_sha256"])
    current, starting = guard_sources()
    if (current != read(gate_dir / "sources_before.json")
            or current != read(gate_dir / "sources_after.json")
            or current != read(control_dir / "sources_before.json")
            or current != read(control_dir / "sources_after.json")):
        raise ValueError("Gate/control/current Python source maps differ")
    for name, expected in report["artifacts_sha256"].items():
        check_reference(control_dir / name, expected)
    for name, expected in audit["input_artifact_sha256"].items():
        check_reference(control_dir / name, expected)
    required_true = ("original_typed_checkpoint_loaded", "original_force_re_solved",
        "original_candidate_all_fields_identical", "history_builder_candidate_replayed",
        "history_dto_external_sources_replayed", "prior_material_state_replayed",
        "source_before_after_unchanged", "original_quadrature_memory_measure_work_preserved")
    if any(report[name] is not True for name in required_true):
        raise ValueError("Physical control is missing a required genuine-source replay")
    if (report["original_source_count"] != 654 or report["original_trace_count"] != 4
            or report["original_node_count"] != 16 or report["query_count"] != 48
            or report["incoming_context_count"] != 7
            or report["query_times"] != [[0, 1], [3, 8], [1, 1]]):
        raise ValueError("Physical control changed the original node/context/time coverage")
    for name in ("accepted_loaded_native_intervals", "accepted_source_work_j"):
        if report[name] != 0:
            raise ValueError("Diagnostic binding adopted a native interval or source work")
    for name in ("candidate_mechanics_adopted", "physical_pair_admission_authorized",
            "global_native_inventory_updated", "work_assigned_to_material_volume",
            "intermediate_work_computed"):
        if report[name] is not False:
            raise ValueError("Diagnostic binding changed its explicit physical scope")
    byte_pairs = []
    for first, second in report["byte_pairs"]:
        if (control_dir / first).read_bytes() != (control_dir / second).read_bytes():
            raise ValueError("Physical control replay bytes differ: " + first)
        byte_pairs.append(dict(first=first, second=second,
            sha256=digest(control_dir / first), bytes=(control_dir / first).stat().st_size,
            accepted_native_force_checkpoint=False))
    authority = verify_prior_sources()
    saved_authority = sealed(control_dir / "source_validation_receipt.json", "receipt_sha256")
    if authority != saved_authority:
        raise ValueError("Prior authoritative sources changed since the new control")
    prior_summary_path = PRIOR / "release_validation_02/validation_summary.json"
    prior_summary = sealed(prior_summary_path, "summary_sha256")
    from scripts.cloud_handoff import verify as verify_cloud_sources
    restored = verify_cloud_sources()
    if restored != dict(files=9, source_bytes=2110696, verified=True):
        raise ValueError("Nine immutable cloud sources did not verify exactly")
    out.mkdir(parents=True, exist_ok=False)
    write(out / "restored_source_receipt.json", restored)
    added = sorted(set(current) - set(starting))
    artifacts = {path.resolve() for directory in (gate_dir, control_dir)
        for path in directory.rglob("*") if path.is_file()}
    artifacts.update((audit_path, BASE / "starting_sources.json",
        out / "restored_source_receipt.json"))
    artifacts.update(ROOT / name for name in authority["source_artifacts"])
    artifacts.update(ROOT / name for name in added)
    manifest = dict(format="native-original-physical-binding-artifact-manifest-1",
        count=len(artifacts), artifacts={relative(path):dict(sha256=digest(path),
            bytes=path.stat().st_size) for path in sorted(artifacts)},
        accepted_loaded_native_intervals=0)
    write(out / "artifact_manifest.json", manifest, seal="manifest_sha256")
    plan = deepcopy(prior_summary["original_six_point_plan"])
    plan[0] = dict(number=1, status="partial",
        completed_here="whole selected owner once-only material closure plus replay-authenticated original quadrature memories/work, exact original moving tip maps and explicit incoming point bank comparisons",
        remaining="incoming physical-bank handover and measure ancestry, implicit-domain force/support integration, accepted native force interval and joint restart")
    summary = dict(format="native-original-physical-binding-validation-1", status="PASS",
        prior_authoritative_validation=dict(path=relative(prior_summary_path),
            sha256=digest(prior_summary_path), verified_artifacts=83,
            tests_passed=770, tests_reexecuted=False, full_profile_executed=False),
        fresh_scoped_gate=dict(path=relative(gate_dir / "receipt.json"),
            sha256=digest(gate_dir / "receipt.json"), tests_passed=gate["counts"]["passed"],
            counts=gate["counts"], test_files=len(TEST_PATHS), test_paths=list(TEST_PATHS),
            seconds=gate["seconds"], full_profile_executed=False),
        actual_control=dict(path=relative(control_dir / "report.json"),
            sha256=digest(control_dir / "report.json"), seconds=report["seconds"],
            original_typed_checkpoint_loaded=True, original_force_re_solved=True,
            original_candidate_all_fields_identical=True, history_builder_candidate_replayed=True,
            original_trace_count=4, original_node_count=16, query_count=48,
            query_times=report["query_times"], incoming_context_count=7, byte_pairs=byte_pairs),
        independent_audit=dict(path=relative(audit_path), sha256=digest(audit_path),
            scope="raw original source/memory/work references and high-precision point checks; numerical sampling does not prove entire interval bounds or physical handover"),
        source_count=len(current), starting_source_count=654, added_python_sources=added,
        existing_sources_changed=[], source_before_after_unchanged=True,
        restored_sources=restored, original_six_point_plan=plan,
        original_pair_history_authenticated=True, incoming_pair_handover_authorized=False,
        candidate_mechanics_adopted=False, global_native_inventory_updated=False,
        physical_pair_admission_authorized=False, accepted_loaded_native_intervals=0,
        accepted_source_work_j=0, work_assigned_to_material_volume=False,
        intermediate_work_computed=False,
        artifact_manifest=dict(path=relative(out / "artifact_manifest.json"),
            sha256=digest(out / "artifact_manifest.json"), count=len(artifacts)))
    write(out / "validation_summary.json", summary, seal="summary_sha256")
    print(json.dumps(dict(status="PASS", prior_verified_tests=770,
        fresh_scoped_tests=gate["counts"]["passed"], sources=len(current),
        unchanged_existing_sources=654, new_python_files=len(added),
        artifacts=len(artifacts), accepted_native_force_intervals=0)), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    consolidate(args.gate, args.control, args.audit, args.out)
