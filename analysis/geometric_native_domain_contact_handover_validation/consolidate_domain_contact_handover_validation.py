"""Seal whole-domain transport, continuum-work evidence and scoped regressions."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from analysis.geometric_native_domain_contact_handover_validation.run_domain_contact_handover_control import (
    PREVIOUS, digest, guard_sources, read, sealed, verify_previous_sources, write,
)
from analysis.geometric_native_domain_contact_handover_validation.run_release_gates import TEST_PATHS


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def check(path, expected):
    if digest(path) != expected:
        raise ValueError("Domain-contact evidence differs: " + str(path))


def consolidate(gate_dir, control_dir, audit_path, out):
    gate_dir, control_dir, audit_path, out = (Path(path).resolve()
        for path in (gate_dir, control_dir, audit_path, out))
    gate, report, audit = (sealed(gate_dir / "receipt.json", "receipt_sha256"),
        sealed(control_dir / "report.json", "report_sha256"), sealed(audit_path, "receipt_sha256"))
    if any(row["status"] != "PASS" for row in (gate, report, audit)):
        raise ValueError("The gate, fresh actual-source control and independent audit must all pass")
    if (gate["exit_code"] != 0 or gate["changed_paths"] or gate["is_full_profile_execution"]
            or gate["original_253_tests_reexecuted"] or gate["original_770_tests_reexecuted"]
            or not gate["prior_265_release_scope_reexecuted"]
            or tuple(gate["test_paths"]) != TEST_PATHS or gate["counts"].get("passed", 0) <= 265):
        raise ValueError("The scoped gate lacks its three-new plus eight-previous file coverage")
    for key, name in (("log_sha256", "pytest.log"), ("sources_before_sha256", "sources_before.json"),
            ("sources_after_sha256", "sources_after.json")):
        check(gate_dir / name, gate[key])
    current, starting = guard_sources()
    for directory in (gate_dir, control_dir):
        for name in ("sources_before.json", "sources_after.json"):
            if read(directory / name) != current:
                raise ValueError("The full current/gate/control source maps differ")
    if report["original_source_count"] != 684 or gate["original_source_count"] != 684:
        raise ValueError("The immutable 684-source ancestry differs")
    for name, expected in report["artifacts_sha256"].items():
        check(control_dir / name, expected)
    for name, expected in audit["input_artifact_sha256"].items():
        check(control_dir / name, expected)
    required = ("original_typed_checkpoint_loaded", "original_force_re_solved",
        "history_builder_original_force_re_solved", "prior_material_state_replayed",
        "original_candidate_all_fields_identical", "source_before_after_unchanged",
        "source_history_bytes_unchanged", "source_checkpoint_bytes_unchanged", "source_candidate_bytes_unchanged",
        "original_quadrature_measure_work_counted_once", "unchanged_global_beam_dofs_and_stiffness",
        "first_receiver_pair_6_to_3_committed", "continuum_and_original_quadrature_measures_distinct",
        "continuum_work_first_law_retains_literal_compliance_residual")
    if any(report[name] is not True for name in required):
        raise ValueError("The domain control lacks source, measure or work preservation")
    if (report["original_candidate_field_count"] != 14 or report["original_node_count"] != 16
            or report["original_continuum_trace_count"] != 4):
        raise ValueError("The complete original source coverage differs")
    false_flags = ("first_root_midpoint_used", "transported_candidate_is_force_equilibrium",
        "native_full_bank_and_clamp_support_certified", "finite_measure_assigned_to_first_entry_point",
        "nodal_work_assigned_to_first_entry_point", "candidate_mechanics_adopted",
        "physical_pair_admission_authorized", "global_native_inventory_updated")
    if any(report[name] is not False for name in false_flags):
        raise ValueError("The control exceeded its domain-reference transport scope")
    zero_fields = ("accepted_loaded_native_intervals", "accepted_source_work_j",
        "transfer_external_work_j", "transfer_dissipation_j")
    if any(report[name] != 0 for name in zero_fields):
        raise ValueError("Domain transport created a native force interval or accepted work")
    byte_pairs = []
    for first, second in report["byte_pairs"]:
        if (control_dir / first).read_bytes() != (control_dir / second).read_bytes():
            raise ValueError("Original source preservation or strict domain replay differs: " + first)
        byte_pairs.append(dict(first=first, second=second, sha256=digest(control_dir / first),
            bytes=(control_dir / first).stat().st_size, accepted_native_force_checkpoint=False))
    if len(byte_pairs) < 6:
        raise ValueError("The control must preserve source and all domain/work replay bytepairs")
    authority = verify_previous_sources()
    if authority != sealed(control_dir / "source_validation_receipt.json", "receipt_sha256"):
        raise ValueError("The 233 prior manifest records changed during the control")
    previous_path = PREVIOUS / "release_validation_01/validation_summary.json"
    previous = sealed(previous_path, "summary_sha256")
    from scripts.cloud_handoff import verify as verify_cloud_sources
    restored = verify_cloud_sources()
    if restored != dict(files=9, source_bytes=2110696, verified=True):
        raise ValueError("The original nine cloud sources changed")
    out.mkdir(parents=True, exist_ok=False)
    write(out / "restored_source_receipt.json", restored)
    added = sorted(set(current) - set(starting))
    artifacts = {path.resolve() for directory in (gate_dir, control_dir)
        for path in directory.rglob("*") if path.is_file()}
    artifacts.update((audit_path, BASE / "starting_sources.json", out / "restored_source_receipt.json"))
    artifacts.update(ROOT / name for name in authority["source_artifacts"])
    artifacts.update(ROOT / name for name in added)
    manifest = dict(format="native-domain-contact-handover-artifact-manifest-1", count=len(artifacts),
        artifacts={relative(path):dict(sha256=digest(path), bytes=path.stat().st_size)
            for path in sorted(artifacts)}, accepted_loaded_native_intervals=0)
    write(out / "artifact_manifest.json", manifest, seal="manifest_sha256")
    summary = dict(format="native-domain-contact-handover-validation-1", status="PASS",
        prior_first_entry_handover_validation=dict(path=relative(previous_path), sha256=digest(previous_path),
            verified_artifacts=48, tests_passed=265, tests_reexecuted=True,
            reexecution_scope="the same eight files within the fresh eleven-file gate", full_profile_executed=False),
        inherited_validation_ancestry=authority["ancestry"][1:], verified_prior_manifest_records=233,
        fresh_scoped_gate=dict(path=relative(gate_dir / "receipt.json"), sha256=digest(gate_dir / "receipt.json"),
            tests_passed=gate["counts"]["passed"], counts=gate["counts"], test_paths=list(TEST_PATHS),
            test_files=len(TEST_PATHS), seconds=gate["seconds"], full_profile_executed=False),
        actual_control=dict(path=relative(control_dir / "report.json"), sha256=digest(control_dir / "report.json"),
            seconds=report["seconds"], original_typed_checkpoint_loaded=True, original_force_re_solved=True,
            history_builder_original_force_re_solved=True, byte_pairs=byte_pairs,
            original_continuum_trace_count=report["original_continuum_trace_count"],
            domain_parent_count=report["domain_parent_count"], causal_mask_count=report["causal_mask_count"],
            first_root_handover_count=report["first_root_handover_count"],
            mutation_refusal_count=report["mutation_refusal_count"]),
        independent_audit=dict(path=relative(audit_path), sha256=digest(audit_path), scope=audit["scope"]),
        source_count=len(current), starting_source_count=684, added_python_sources=added,
        existing_sources_changed=[], source_before_after_unchanged=True, restored_sources=restored,
        original_six_point_plan=deepcopy(report["original_six_point_plan"]),
        remaining_work_blocks=deepcopy(report["remaining_work_blocks"]),
        remaining_work_block_count=report["remaining_work_block_count"], law=report["law"],
        **{name:False for name in false_flags}, **{name:0 for name in zero_fields},
        whole_first_entry_domain_registry_committed=report["whole_first_entry_domain_registry_committed"],
        whole_first_entry_interface_domain_numerically_integrated=report["whole_first_entry_interface_domain_numerically_integrated"],
        whole_original_interface_continuum_work_integrated=report["whole_original_interface_continuum_work_integrated"],
        per_pair_incoming_continuum_work_numerically_integrated=report["per_pair_incoming_continuum_work_numerically_integrated"],
        artifact_manifest=dict(path=relative(out / "artifact_manifest.json"),
            sha256=digest(out / "artifact_manifest.json"), count=len(artifacts)))
    write(out / "validation_summary.json", summary, seal="summary_sha256")
    print(json.dumps(dict(status="PASS", fresh_scoped_tests=gate["counts"]["passed"],
        preserved_prior_tests=[265,178,253,770], sources=len(current), unchanged_existing_sources=684,
        new_python_files=len(added), artifacts=len(artifacts), accepted_native_force_intervals=0)), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    consolidate(args.gate, args.control, args.audit, args.out)
