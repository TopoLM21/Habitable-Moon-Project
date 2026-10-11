"""Seal a narrow active-binding release against immutable complete ancestry."""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from analysis.geometric_native_active_binding_validation.validation_support import (
    digest, encoded, guard_sources, read, sealed, verify_previous_sources, write,
)


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def check(path, expected):
    if digest(path) != expected:
        raise ValueError("Active-binding evidence differs: " + str(path))


def verify_audit_only_amendment(current, gate_dir, control_dir):
    """Retain the tested freeze while declaring precisely two audit-only edits."""
    amendment_path = BASE / "audit_only_source_amendment.json"
    amendment = sealed(amendment_path, "amendment_sha256")
    archive_dir = BASE / "source_freeze_attempt_02"
    allowed = {
        relative(BASE / "independent_active_binding_audit.py"),
        relative(BASE / "consolidate_active_binding_validation.py"),
    }
    if (amendment["format"] != "native-active-binding-audit-only-source-amendment-1"
            or amendment["status"] != "PASS"
            or amendment["source_count"] != 704 or amendment["original_source_count"] != 694
            or amendment["changed_python_source_count"] != 2
            or amendment["unchanged_python_source_count"] != 702
            or amendment["original_694_sources_unchanged"] is not True
            or amendment["production_tests_gate_and_control_sources_unchanged"] is not True
            or amendment["physics_control_and_tests_reexecuted"] is not False
            or set(amendment["changes"]) != allowed):
        raise ValueError("The exact two-file audit-only source amendment differs")
    refs = {
        "execution_frozen_sources": archive_dir / "frozen_release_sources.json",
        "execution_frozen_receipt": archive_dir / "frozen_release_receipt.json",
        "execution_source_archive": archive_dir / "archive_receipt.json",
        "current_frozen_sources": BASE / "frozen_release_sources.json",
        "passing_gate": gate_dir / "receipt.json",
        "passing_control": control_dir / "report.json",
    }
    for name, path in refs.items():
        if amendment[name] != dict(path=relative(path), sha256=digest(path)):
            raise ValueError("The source amendment evidence reference differs: " + name)
    execution = read(refs["execution_frozen_sources"])
    if (set(execution) != set(current) or len(current) != 704
            or read(refs["current_frozen_sources"]) != current
            or {name for name in current if current[name] != execution[name]} != allowed):
        raise ValueError("Only the two declared audit/consolidation Python hashes may change")
    old_freeze = sealed(refs["execution_frozen_receipt"], "receipt_sha256")
    archive = sealed(refs["execution_source_archive"], "receipt_sha256")
    if (old_freeze["status"] != "PASS" or old_freeze["attempt"] != 2
            or old_freeze["source_count"] != 704 or old_freeze["original_source_count"] != 694
            or old_freeze["frozen_sources_sha256"] != digest(refs["execution_frozen_sources"])
            or archive["status"] != "PASS" or archive["source_count"] != 704
            or archive["original_source_count"] != 694
            or archive["execution_frozen_sources"] != amendment["execution_frozen_sources"]
            or archive["execution_frozen_receipt"] != amendment["execution_frozen_receipt"]
            or archive["passing_gate"] != amendment["passing_gate"]
            or archive["passing_control"] != amendment["passing_control"]):
        raise ValueError("The complete passing execution source archive differs")
    original = read(BASE / "starting_sources.json")
    if any(current.get(name) != expected or execution.get(name) != expected
            for name, expected in original.items()):
        raise ValueError("An original694 Python source changed across the audit amendment")
    for name in allowed:
        row = amendment["changes"][name]
        body = archive_dir / (Path(name).name + ".txt")
        if (set(row) != {"before_sha256", "after_sha256", "archived_before_body", "purpose"}
                or row["before_sha256"] != execution[name]
                or row["after_sha256"] != current[name]
                or row["archived_before_body"] != dict(path=relative(body), sha256=digest(body))
                or digest(body) != execution[name]
                or archive["original_bodies"][name] != dict(path=relative(body),
                    sha256=digest(body), bytes=body.stat().st_size)):
            raise ValueError("The transparent audit-only original/current source bodies differ")
    expected_maps = []
    for directory in (gate_dir, control_dir):
        for stage in ("before", "after"):
            path = directory / ("sources_" + stage + ".json")
            if read(path) != execution:
                raise ValueError("The passing physics/control execution freeze changed")
            expected_maps.append(dict(path=relative(path), sha256=digest(path)))
    if archive["execution_maps"] != expected_maps:
        raise ValueError("The four unchanged complete execution map references differ")
    return dict(path=relative(amendment_path), sha256=digest(amendment_path),
        execution_frozen_sources=amendment["execution_frozen_sources"],
        current_frozen_sources=amendment["current_frozen_sources"],
        changed_python_sources=sorted(allowed), changed_python_source_count=2,
        unchanged_python_source_count=702, original_694_sources_unchanged=True,
        production_tests_gate_and_control_sources_unchanged=True,
        physics_control_and_tests_reexecuted=False)


def consolidate(gate_dir, control_dir, audit_path, out):
    gate_dir, control_dir, audit_path, out = (Path(path).resolve()
        for path in (gate_dir, control_dir, audit_path, out))
    gate = sealed(gate_dir / "receipt.json", "receipt_sha256")
    report = sealed(control_dir / "report.json", "report_sha256")
    audit = sealed(audit_path, "receipt_sha256")
    if any(value["status"] != "PASS" for value in (gate, report, audit)):
        raise ValueError("The scoped gate, actual-source control and independent audit must pass")
    if (gate["exit_code"] != 0 or gate["changed_paths"] or gate["is_full_profile_execution"]
            or gate["prior_341_full_scope_reexecuted"]
            or gate["canonical_pytest_import_mode"] != "importlib"
            or gate["counts"].get("passed", 0) != 136):
        raise ValueError("The narrow gate or its canonical shared fixture coverage differs")
    for key, name in (("log_sha256", "pytest.log"),
            ("sources_before_sha256", "sources_before.json"),
            ("sources_after_sha256", "sources_after.json")):
        check(gate_dir / name, gate[key])
    current, original = guard_sources(require_frozen=True)
    amendment = verify_audit_only_amendment(current, gate_dir, control_dir)
    if audit["source_amendment"] != amendment:
        raise ValueError("The independent audit source amendment provenance differs")
    if gate["original_source_count"] != 694 or report["original_source_count"] != 694:
        raise ValueError("The immutable 694-source ancestry differs")
    for name, expected in report["artifacts_sha256"].items():
        check(control_dir / name, expected)
    for name, expected in audit["input_artifact_sha256"].items():
        check(control_dir / name, expected)
    if (report["accepted_loaded_native_intervals"] != 0
            or report["accepted_source_work_j"] != 0
            or report["candidate_mechanics_adopted"] is not False
            or report["physical_pair_admission_authorized"] is not False
            or report["per_receiving_pair_weighted_contact_integrals_evaluated"] is not False):
        raise ValueError("The release exceeded its active-binding reference scope")
    if report["original_candidate_field_count"] != 14:
        raise ValueError("All fourteen original actual-source candidate fields are required")
    if any(report[name] is not True for name in (
            "original_typed_checkpoint_loaded", "original_force_re_solved",
            "history_builder_original_force_re_solved", "source_before_after_unchanged",
            "frozen_original_material_label_used_for_later_geometry",
            "unique_active_binding_per_receiving_carrier", "event_transfer_increments_zero",
            "original_literal_quadrature_ledger_unchanged")):
        raise ValueError("The active-binding control lacks source or causal state preservation")
    pairs = []
    for first, second in report["byte_pairs"]:
        if (control_dir / first).read_bytes() != (control_dir / second).read_bytes():
            raise ValueError("A strict replay or source-preservation bytepair differs")
        pairs.append(dict(first=first, second=second, sha256=digest(control_dir / first),
            bytes=(control_dir / first).stat().st_size,
            accepted_native_force_checkpoint=False))
    authority = verify_previous_sources()
    from scripts.cloud_handoff import verify as verify_cloud_sources
    restored = verify_cloud_sources()
    if restored != dict(files=9, source_bytes=2110696, verified=True):
        raise ValueError("The original nine cloud sources changed")
    previous_path = BASE.parent / "geometric_native_domain_contact_handover_validation/release_validation_01/validation_summary.json"
    previous = sealed(previous_path, "summary_sha256")
    plan, remaining = deepcopy(previous["original_six_point_plan"]), deepcopy(previous["remaining_work_blocks"])
    plan[0].update(completed_here=report["completed_scope"],
        remaining="whole receiving-angle unique-binding proof and numerical receiving-pair weighted contact/work balances, native full-bank/clamp support and spatial admission, accepted intervals and joint restart")
    remaining[0].update(completed=report["completed_scope_ru"],
        remaining="доказательство единственности привязки по всей принимающей линии, численные взвешенные интегралы по действующим принимающим парам и замыкание их полного баланса")
    out.mkdir(parents=True, exist_ok=False)
    write(out / "restored_source_receipt.json", restored)
    added = sorted(set(current) - set(original))
    artifacts = {path.resolve() for directory in (gate_dir, control_dir)
        for path in directory.rglob("*") if path.is_file()}
    artifacts.update(path.resolve() for path in BASE.glob("*.json"))
    for failed in (BASE / "release_gates_01", BASE / "control_release_01",
            BASE / "source_freeze_attempt_01", BASE / "source_freeze_attempt_02"):
        if failed.is_dir():
            artifacts.update(path.resolve() for path in failed.rglob("*") if path.is_file())
    artifacts.update((audit_path, out / "restored_source_receipt.json", previous_path,
        previous_path.parent / "artifact_manifest.json"))
    artifacts.update(ROOT / name for name in added)
    manifest = dict(format="native-active-binding-artifact-manifest-1", count=len(artifacts),
        artifacts={relative(path): dict(sha256=digest(path), bytes=path.stat().st_size)
            for path in sorted(artifacts)}, accepted_loaded_native_intervals=0)
    write(out / "artifact_manifest.json", manifest, seal="manifest_sha256")
    summary = dict(format="native-active-binding-validation-1", status="PASS",
        prior_authoritative_validation=dict(path=relative(previous_path), sha256=digest(previous_path),
            tests_passed=341, verified_manifest_records=69, tests_reexecuted=False),
        verified_prior_manifest_records=302, inherited_validation_ancestry=authority["ancestry"],
        fresh_scoped_gate=dict(path=relative(gate_dir / "receipt.json"),
            sha256=digest(gate_dir / "receipt.json"), tests_passed=gate["counts"]["passed"],
            test_paths=gate["test_paths"], test_files=len(gate["test_paths"]),
            seconds=gate["seconds"], full_profile_executed=False),
        actual_control=dict(path=relative(control_dir / "report.json"),
            sha256=digest(control_dir / "report.json"), seconds=report["seconds"],
            byte_pairs=pairs, receiving_carrier_count=report["receiving_carrier_count"],
            active_transfer_event_count=report["active_transfer_event_count"],
            mutation_refusal_count=report["mutation_refusal_count"]),
        independent_audit=dict(path=relative(audit_path), sha256=digest(audit_path),
            scope=audit["scope"]), source_count=len(current), starting_source_count=694,
        added_python_sources=added, existing_sources_changed=[], source_before_after_unchanged=True,
        source_before_after_unchanged_scope="passing physics gate/control execution; later exact two-file audit-only amendment declared separately",
        execution_source_count=704, source_amendment=amendment,
        restored_sources=restored, completed_scope=report["completed_scope"],
        original_six_point_plan=plan, remaining_work_blocks=remaining, remaining_work_block_count=3,
        all_receiver_angles_scheduled=False,
        per_receiving_pair_weighted_contact_integrals_evaluated=False,
        candidate_mechanics_adopted=False, physical_pair_admission_authorized=False,
        accepted_loaded_native_intervals=0, accepted_source_work_j=0,
        artifact_manifest=dict(path=relative(out / "artifact_manifest.json"),
            sha256=digest(out / "artifact_manifest.json"), count=len(artifacts)))
    write(out / "validation_summary.json", summary, seal="summary_sha256")
    print(encoded(dict(status="PASS", fresh_scoped_tests=gate["counts"]["passed"],
        source_count=len(current), existing_sources_preserved=694, added_python_files=len(added),
        artifact_manifest_count=len(artifacts), accepted_native_force_intervals=0)).decode(), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    consolidate(args.gate, args.control, args.audit, args.out)
