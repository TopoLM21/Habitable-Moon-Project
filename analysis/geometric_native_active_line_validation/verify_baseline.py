"""Stdlib-only verification of the immutable release preceding line work.

This checks saved evidence; it deliberately does not rerun historical tests or
admit the diagnostic force candidate as an accepted interval.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
PRIOR = ROOT / "analysis/geometric_native_active_binding_validation"
SUMMARY = PRIOR / "release_validation_01/validation_summary.json"
MANIFEST = PRIOR / "release_validation_01/artifact_manifest.json"
EXPECTED_SUMMARY_RAW = "94a8ce6f50bb87a120adc100695506d1ccc61acc8d98d0abe6e654e7ad272781"
EXPECTED_MANIFEST_RAW = "bc46221217376dc7d25d59c072a8131e8766e6e573fb2331c32aec451286861d"


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()


def value_sha(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def sealed(path, key):
    value = read(path)
    expected = value.pop(key)
    if expected != value_sha(value):
        raise ValueError("Evidence selfseal differs: " + str(path))
    return value


def write(path, value, seal=None):
    value = deepcopy(value)
    if seal is not None:
        if seal in value:
            raise ValueError("Cannot overwrite a declared seal")
        value[seal] = value_sha(value)
    with Path(path).open("xb") as stream:
        stream.write(encoded(value) + b"\n")
    return value


def sources():
    return {str(path.relative_to(ROOT)): digest(path)
        for directory in ("tectonics", "tests", "scripts", "analysis")
        for path in sorted((ROOT / directory).rglob("*.py"))}


def verify_baseline():
    if digest(SUMMARY) != EXPECTED_SUMMARY_RAW or digest(MANIFEST) != EXPECTED_MANIFEST_RAW:
        raise ValueError("The authoritative active-binding release changed")
    summary = sealed(SUMMARY, "summary_sha256")
    if (summary["status"] != "PASS" or summary["fresh_scoped_gate"]["tests_passed"] != 136
            or summary["accepted_loaded_native_intervals"] != 0
            or summary["candidate_mechanics_adopted"] is not False
            or summary["source_count"] != 704):
        raise ValueError("Authoritative release status or scope differs")
    rows = [dict(path=str(SUMMARY.relative_to(ROOT)), sha256=digest(SUMMARY),
        manifest_path=str(MANIFEST.relative_to(ROOT)), manifest_sha256=digest(MANIFEST),
        verified_manifest_records=72, tests_passed=136), *summary["inherited_validation_ancestry"]]
    unique = set()
    count = 0
    for row in rows:
        summary_path, manifest_path = ROOT / row["path"], ROOT / row["manifest_path"]
        if digest(summary_path) != row["sha256"] or digest(manifest_path) != row["manifest_sha256"]:
            raise ValueError("An ancestry document changed")
        ancestor = sealed(summary_path, "summary_sha256")
        manifest = sealed(manifest_path, "manifest_sha256")
        if (ancestor["status"] != "PASS" or manifest["count"] != row["verified_manifest_records"]
                or len(manifest["artifacts"]) != manifest["count"]
                or ancestor["artifact_manifest"]["sha256"] != digest(manifest_path)):
            raise ValueError("An ancestry manifest count or reference differs")
        for name, record in manifest["artifacts"].items():
            path = ROOT / name
            if not path.is_file() or path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
                raise ValueError("An ancestry artifact changed: " + name)
            unique.add(name)
        for key in ("fresh_scoped_gate", "release_gate", "actual_control", "independent_audit"):
            if key in ancestor:
                reference = ancestor[key]
                if digest(ROOT / reference["path"]) != reference["sha256"]:
                    raise ValueError("An ancestry evidence reference changed: " + key)
        count += manifest["count"]
    if count != 374:
        raise ValueError("The full 374-record ancestry was not verified")
    current = sources()
    frozen = read(PRIOR / "frozen_release_sources.json")
    original = read(PRIOR / "starting_sources.json")
    if len(frozen) != 704 or len(original) != 694:
        raise ValueError("The prior source map count differs")
    changed = sorted(name for name, expected in frozen.items() if current.get(name) != expected)
    if changed:
        raise ValueError("An immutable prior Python source changed: " + repr(changed))
    if any(frozen.get(name) != expected for name, expected in original.items()):
        raise ValueError("An original 694 Python source changed")
    amendment = sealed(PRIOR / "audit_only_source_amendment.json", "amendment_sha256")
    execution = read(ROOT / amendment["execution_frozen_sources"]["path"])
    difference = {name for name in frozen if frozen[name] != execution.get(name)}
    if (difference != set(amendment["changes"]) or len(difference) != 2
            or amendment["status"] != "PASS" or amendment["source_count"] != 704
            or amendment["original_694_sources_unchanged"] is not True):
        raise ValueError("The explicit two-file audit-only amendment differs")
    for name, row in amendment["changes"].items():
        archive = ROOT / row["archived_before_body"]["path"]
        if (frozen[name] != row["after_sha256"] or execution[name] != row["before_sha256"]
                or digest(archive) != row["before_sha256"]
                or digest(archive) != row["archived_before_body"]["sha256"]):
            raise ValueError("The amended source before/after provenance differs")
    clouds = read(ROOT / "handoff/cloud_sources_manifest.json")
    total_bytes = 0
    for row in clouds["files"]:
        path = ROOT / row["path"]
        if path.stat().st_size != row["size"] or digest(path) != row["sha256"]:
            raise ValueError("A cloud source changed: " + row["path"])
        total_bytes += row["size"]
    if len(clouds["files"]) != 9 or total_bytes != 2110696:
        raise ValueError("Cloud source count or byte total differs")
    return dict(format="native-active-line-immutable-baseline-1", status="PASS",
        verified_utc=datetime.now(timezone.utc).isoformat(), prior_authoritative_validation=rows[0],
        ancestry=rows[1:], verified_manifest_records=count, verified_unique_artifacts=len(unique),
        original_python_source_count=694, immutable_python_source_count=704,
        original_694_sources_unchanged=True, immutable_704_sources_unchanged=True,
        current_observed_source_count=len(current),
        additions_since_baseline=sorted(set(current)-set(frozen)),
        original_source_map_sha256=digest(PRIOR / "starting_sources.json"),
        immutable_source_map_sha256=digest(PRIOR / "frozen_release_sources.json"),
        audit_only_source_amendment_sha256=digest(PRIOR / "audit_only_source_amendment.json"),
        verified_declared_audit_only_changes=sorted(difference),
        restored_sources=dict(files=9, source_bytes=total_bytes, verified=True, restored=False),
        prior_tests_reexecuted=False, prior_physical_source_resolved=False,
        accepted_loaded_native_intervals=0, accepted_source_work_j=0,
        scope="Read-only hash/size/self-seal/provenance verification; no new whole-line or force claim")


if __name__ == "__main__":
    receipt = write(BASE / "immutable_baseline_receipt.json", verify_baseline(), "receipt_sha256")
    print(encoded({key: receipt[key] for key in ("status", "verified_manifest_records",
        "verified_unique_artifacts", "immutable_python_source_count", "restored_sources")}).decode())
