"""Bind first-entry tests, real force evidence and the independent auditor."""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seal(path, key):
    body = read(path)
    expected = body.pop(key)
    assert hashlib.sha256(encoded(body)).hexdigest() == expected, (path, key)
    return body


def relative(path):
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def write(path, body, key=None):
    if key is not None:
        body = dict(body, **{key:hashlib.sha256(encoded(body)).hexdigest()})
    with path.open("xb") as stream:
        stream.write(encoded(body) + b"\n")


def verify_sources(path):
    records = read(path)
    changed = [name for name, expected in records.items()
               if digest(ROOT / name) != expected]
    assert not changed, (path, changed)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    gate_dir, control, audit_path = (path.resolve()
        for path in (args.gate, args.control, args.audit))
    gate = read(gate_dir / "receipt.json")
    report = seal(control / "report.json", "report_sha256")
    audit = seal(audit_path, "receipt_sha256")
    assert gate["status"] == report["status"] == audit["status"] == "PASS"
    assert gate["exit_code"] == 0 and not gate["changed_paths"]
    assert not gate["is_full_profile_execution"]
    assert report["original_force_re_solved"] and report["original_candidate_all_fields_identical"]
    assert report["whole_selected_owner_swept_domain_certified"]
    assert report["full_curved_extra_integrated"]
    assert report["complete_standalone_material_closure_applied"]
    assert report["actual_material_vacancies_computed"] and report["endpoint_hot_fill_applied"]
    assert report["source_parent_count"] == 8 and report["positive_archive_cohort_count"] == 3
    current = verify_sources(gate_dir / "sources_after.json")
    assert current == read(gate_dir / "sources_before.json")
    assert current == verify_sources(control / "sources_after.json") == read(control / "sources_before.json")
    assert current == verify_sources(control / "original_force/sources_after.json")
    baseline = verify_sources(BASE / "starting_sources.json")
    assert len(baseline) == 643
    assert baseline == read(ROOT / "analysis/geometric_native_full_domain_validation/release_gates_01/sources_after.json")
    assert set(baseline) <= set(current)
    added = sorted(set(current) - set(baseline))
    for name, expected in report["artifacts_sha256"].items():
        assert digest(control / name) == expected, name
    for name, expected in audit["input_artifact_sha256"].items():
        assert digest(control / name) == expected, name
    assert digest(gate_dir / "pytest.log") == gate["log_sha256"]
    assert digest(gate_dir / "sources_before.json") == gate["sources_before_sha256"]
    assert digest(gate_dir / "sources_after.json") == gate["sources_after_sha256"]
    byte_pairs = []
    for first, second in report["byte_pairs"]:
        assert (control / first).read_bytes() == (control / second).read_bytes()
        byte_pairs.append(dict(first=first, second=second,
            sha256=digest(control / first), bytes=(control / first).stat().st_size,
            accepted_native_force_checkpoint=False))
    candidate = read(control / "original_force/force_candidate.json")
    probe = ROOT / "analysis/geometric_native_underthrust_validation/force_control_release_02/force_candidate_refusal_probe.json"
    original = read(probe)
    assert all(candidate[name] == value for name, value in original["candidate"].items())
    assert candidate["admission"] == original["admission"]
    assert digest(probe) == report["source_probe_sha256"]
    state = read(control / "material_state.json")
    assert state["material_closure_applied"]
    assert not state["candidate_mechanics_adopted"]
    assert not state["physical_pair_admission_authorized"]
    assert state["native_force_intervals_accepted"] == 0
    assert len(state["parent_allocations"]) == 8 and len(state["archive_cohorts"]) == 3
    sweep = read(control / "first_entry_sweep.json")
    domains = {row["fragment_id"]:row for row in sweep["donor_domains"]}
    extra = domains["octant:5"]["extra_swept_area_bounds_sr"]
    assert Fraction(*extra["lo"]) > 0
    assert domains["octant:4"]["domain_case"] == "empty"
    assert sweep["extra_endpoint_fixed_parent_id"] == "octant:5"
    assert report["accepted_loaded_native_intervals"] == report["accepted_source_work_j"] == 0
    assert not report["global_native_inventory_updated"]
    restored = json.loads(subprocess.check_output(
        [sys.executable, "scripts/cloud_handoff.py", "verify"], cwd=ROOT))
    assert restored == dict(files=9, source_bytes=2110696, verified=True)
    write(args.out / "restored_source_receipt.json", restored)
    artifacts = {path.resolve() for directory in (gate_dir, control)
                 for path in directory.rglob("*") if path.is_file()}
    artifacts.update((audit_path, probe, BASE / "starting_sources.json",
                      args.out.resolve() / "restored_source_receipt.json"))
    artifacts.update(ROOT / path for path in added)
    manifest = dict(format="native-whole-first-entry-artifact-manifest-1",
        count=len(artifacts), artifacts={relative(path):dict(
            sha256=digest(path), bytes=path.stat().st_size) for path in sorted(artifacts)},
        accepted_loaded_native_intervals=0)
    write(args.out / "artifact_manifest.json", manifest, "manifest_sha256")
    plan = [dict(number=1, status="partial",
        completed_here="whole selected owner first-entry domains, three once-only material cohorts, eight actual vacancy domains and endpoint hot fill in separate implicit material closure",
        remaining="moving physical-bank ancestry and memory/work binding, implicit-domain force/support integration, accepted native force interval and joint restart"),
        *[dict(number=number, status="pending") for number in range(2, 7)]]
    summary = dict(format="native-whole-first-entry-validation-1", status="PASS",
        release_gate=dict(path=relative(gate_dir / "receipt.json"),
            sha256=digest(gate_dir / "receipt.json"), tests_passed=gate["counts"]["passed"],
            seconds=gate["seconds"], full_profile_executed=False),
        actual_control=dict(path=relative(control / "report.json"),
            sha256=digest(control / "report.json"), seconds=report["seconds"],
            original_force_re_solved=True, original_candidate_all_fields_identical=True,
            positive_archive_cohort_count=3, endpoint_newborn_count=report["endpoint_newborn_count"],
            point_certificate_count=report["point_certificate_count"],
            material_query_count=report["material_query_count"],
            endpoint_query_count=report["endpoint_query_count"], byte_pairs=byte_pairs),
        independent_audit=dict(path=relative(audit_path), sha256=digest(audit_path),
            scope="whole selected owner envelope and integrated curved loss; complete original material budgets and endpoint vacancies; explicit query/physics limits in receipt"),
        source_count=len(current), starting_source_count=len(baseline),
        added_python_sources=added, existing_sources_changed=[],
        source_before_after_unchanged=True, restored_sources=restored,
        full_curved_extra_area_sr_bounds=extra,
        complete_standalone_material_closure_applied=True,
        actual_material_vacancies_computed=True, endpoint_hot_fill_applied=True,
        original_live_archive_reservoirs_symbolically_conserved=True,
        implicit_domains_converted_to_endpoint_native_polygons=False,
        original_six_point_plan=plan,
        candidate_mechanics_adopted=False, global_native_inventory_updated=False,
        physical_pair_admission_authorized=False, accepted_loaded_native_intervals=0,
        accepted_source_work_j=0, continuous_hot_birth_law_declared=False,
        artifact_manifest=dict(path=relative(args.out.resolve() / "artifact_manifest.json"),
            sha256=digest(args.out / "artifact_manifest.json"), count=len(artifacts)))
    write(args.out / "validation_summary.json", summary, "summary_sha256")
    print(json.dumps(dict(status="PASS", tests=summary["release_gate"]["tests_passed"],
        sources=len(current), unchanged_existing_sources=len(baseline),
        new_python_files=len(added), artifacts=len(artifacts),
        accepted_native_force_intervals=0)), flush=True)


if __name__ == "__main__":
    main()
