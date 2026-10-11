"""Bind release tests, live force evidence and independent full-domain audit."""
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
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def seal(path, key):
    body = read(path)
    expected = body.pop(key)
    assert hashlib.sha256(encoded(body)).hexdigest() == expected, (path, key)
    return body


def write(path, body, key=None):
    if key is not None:
        body = dict(body, **{key: hashlib.sha256(encoded(body)).hexdigest()})
    with path.open("xb") as stream:
        stream.write(encoded(body) + b"\n")


def relative(path):
    path = path.resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def verify_sources(path):
    records = read(path)
    changed = [name for name, expected in records.items() if digest(ROOT / name) != expected]
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
    gate_dir, control, audit_path = (path.resolve() for path in (args.gate, args.control, args.audit))
    gate = read(gate_dir / "receipt.json")
    report = seal(control / "report.json", "report_sha256")
    audit = seal(audit_path, "receipt_sha256")
    assert gate["status"] == report["status"] == audit["status"] == "PASS"
    assert gate["exit_code"] == 0 and not gate["changed_paths"]
    assert not gate["is_full_profile_execution"]
    assert report["original_force_re_solved"] and report["source_parent_count"] == 8
    assert report["endpoint_pair_count"] == 5 and report["raw_endpoint_vacancy_count"] == 7
    assert report["positive_area_swept_only_material_missing_endpoint_loss_proved"]
    current = verify_sources(gate_dir / "sources_after.json")
    assert current == read(gate_dir / "sources_before.json")
    assert current == verify_sources(control / "sources_after.json") == read(control / "sources_before.json")
    assert current == verify_sources(control / "original_force/sources_after.json")
    baseline = verify_sources(BASE / "starting_sources.json")
    assert len(baseline) == 632
    assert baseline == read(ROOT / "analysis/geometric_native_material_history_validation/release_gates_01/sources_after.json")
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
    for first, second, scope in (
        ("full_domain_journal.json", "full_domain_journal_replayed.json", "complete_original_domain_journal"),
        ("full_swept_domain.json", "full_swept_domain_replayed.json", "whole_blocking_pair_positive_area_swept_footprint"),
        ("point_queries.json", "point_queries_replayed.json", "executable_individual_point_histories"),
        ("original_force/initial.json", "original_force/after_refusal.json", "unchanged_prepared_force_state"),
    ):
        assert (control / first).read_bytes() == (control / second).read_bytes()
        byte_pairs.append(dict(first=first, second=second, scope=scope,
            sha256=digest(control / first), bytes=(control / first).stat().st_size,
            accepted_native_force_checkpoint=False))
    candidate = read(control / "original_force/force_candidate.json")
    probe_path = ROOT / "analysis/geometric_native_underthrust_validation/force_control_release_02/force_candidate_refusal_probe.json"
    probe = read(probe_path)
    assert all(candidate[name] == value for name, value in probe["candidate"].items())
    assert candidate["admission"] == probe["admission"]
    assert digest(probe_path) == report["source_probe_sha256"]
    for key in ("accepted_loaded_native_intervals", "accepted_source_work_j"):
        assert report[key] == 0
    assert not report["global_inventory_updated"] and not report["candidate_memory_adopted"]
    assert not report["geometry_admission_authorized"]
    witness = read(control / "swept_only_material_patch.json")
    area = witness["original_patch_area_sr_bounds"]
    assert Fraction(*area["lo"]) > 0
    assert witness["patch_disjoint_from_actual_endpoint_loss"]
    assert witness["extra_material_vacancy_if_once_only_archive_adopted"]
    assert not witness["global_endpoint_loss_equals_complete_swept_loss"]
    restored = json.loads(subprocess.check_output([sys.executable, "scripts/cloud_handoff.py", "verify"], cwd=ROOT))
    assert restored == dict(files=9, source_bytes=2110696, verified=True)
    write(args.out / "restored_source_receipt.json", restored)
    artifacts = {p.resolve() for directory in (gate_dir, control) for p in directory.rglob("*") if p.is_file()}
    artifacts.update((audit_path, probe_path, BASE / "starting_sources.json",
        args.out.resolve() / "restored_source_receipt.json"))
    artifacts.update(ROOT / path for path in added)
    manifest = dict(format="native-full-source-domain-artifact-manifest-1",
        count=len(artifacts), artifacts={relative(p): dict(sha256=digest(p), bytes=p.stat().st_size)
                                       for p in sorted(artifacts)},
        accepted_loaded_native_intervals=0)
    write(args.out / "artifact_manifest.json", manifest, "manifest_sha256")
    plan = [
        dict(number=1, status="partial", scope="bending, transverse deformation, physical fault motion and spatial underthrust",
             completed_here="complete original-domain journal, exact whole blocking-pair swept footprint, individual point histories and positive swept-only material witness",
             remaining="full owner first-entry loss domains and receiver ancestry, once-only archive debits, true material vacancies and endpoint hot fill, physical material-pair memory/force binding, accepted native force interval and joint restart"),
        *[dict(number=number, status="pending") for number in range(2, 7)],
    ]
    summary = dict(format="native-full-source-domain-validation-1", status="PASS",
        release_gate=dict(path=relative(gate_dir / "receipt.json"), sha256=digest(gate_dir / "receipt.json"),
            tests_passed=gate["counts"]["passed"], seconds=gate["seconds"], full_profile_executed=False),
        actual_control=dict(path=relative(control / "report.json"), sha256=digest(control / "report.json"),
            seconds=report["seconds"], original_force_re_solved=True,
            original_force_candidate_fields_identical=True, original_force_field_count=len(probe["candidate"]),
            source_parent_count=8, endpoint_pair_count=5, raw_vacancy_count=7,
            point_certificate_count=report["point_certificate_count"], byte_pairs=byte_pairs),
        independent_audit=dict(path=relative(audit_path), sha256=digest(audit_path),
            scope="exact all-saved-point continuous proof replay and whole blocking-pair cone proof; numerical all-source endpoint geometry; explicit limits in receipt"),
        source_count=len(current), starting_source_count=len(baseline), existing_sources_changed=[],
        added_python_sources=added, source_before_after_unchanged=True,
        restored_sources=restored, original_six_point_plan=plan,
        whole_blocking_pair_positive_area_swept_footprint_proved=True,
        endpoint_loss_equals_global_complete_swept_loss=False,
        positive_area_swept_only_material_missing_endpoint_loss_proved=True,
        complete_swept_material_volume_integrated=False,
        every_original_point_history_certified=False,
        raw_gap_equals_once_only_archive_material_vacancy=False,
        continuous_hot_birth_law_declared=False, newborn_material_created=False,
        global_inventory_updated=False, candidate_mechanics_adopted=False,
        accepted_loaded_native_intervals=0, accepted_source_work_j=0,
        artifact_manifest=dict(path=relative(args.out.resolve() / "artifact_manifest.json"),
            sha256=digest(args.out / "artifact_manifest.json"), count=len(artifacts)))
    write(args.out / "validation_summary.json", summary, "summary_sha256")
    print(json.dumps(dict(status="PASS", tests=summary["release_gate"]["tests_passed"],
        sources=len(current), unchanged_existing_sources=len(baseline), new_python_files=len(added),
        artifacts=len(artifacts), accepted_native_force_intervals=0)), flush=True)


if __name__ == "__main__":
    main()
