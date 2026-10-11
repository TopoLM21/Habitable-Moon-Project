"""Read-only authentication of evidence inherited by the newborn-contact work.

Historical tests are not rerun here.  In particular, the saved second force
candidate stays refused and the original source/history bodies stay unchanged.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
PRIOR = ROOT / "analysis/geometric_native_active_line_validation"
SUMMARY = PRIOR / "release_validation_01/validation_summary.json"
MANIFEST = PRIOR / "release_validation_01/artifact_manifest.json"
EXPECTED_SUMMARY_RAW = "bd3a5f49823df99ea83e4bb57dd8597f41814c9b060c1a4d12ae22202ffd1f8c"
EXPECTED_MANIFEST_RAW = "0ab2c70496da7e67b02b6a9befa89ef6a9e7ecd2a68632a71472d5fd75854923"


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


def checked_reference(reference):
    path = ROOT / reference["path"]
    if digest(path) != reference["sha256"]:
        raise ValueError("An evidence reference changed: " + reference["path"])
    return path


def verify_baseline():
    if digest(SUMMARY) != EXPECTED_SUMMARY_RAW or digest(MANIFEST) != EXPECTED_MANIFEST_RAW:
        raise ValueError("The authoritative active-line release changed")
    summary = sealed(SUMMARY, "summary_sha256")
    manifest = sealed(MANIFEST, "manifest_sha256")
    required = {
        "status": "PASS", "source_count": 732, "accepted_loaded_native_intervals": 1,
        "entire_point_1_completed": False, "first_interval_adopted": True,
        "subsequent_candidate_solved": True, "subsequent_candidate_refused": True,
        "subsequent_interval_accepted": False, "physical_inputs_invented": False,
        "remaining_work_block_count": 1,
    }
    if any(summary.get(key) != value for key, value in required.items()):
        raise ValueError("The inherited physical status or remaining scope differs")
    if (summary["fresh_scoped_gate"]["tests_passed"] != 137
            or summary["artifact_manifest"]["sha256"] != digest(MANIFEST)
            or manifest["count"] != 1100 or manifest["unique_artifact_count"] != 1100
            or len(manifest["artifacts"]) != 1100):
        raise ValueError("The inherited scoped gate or artifact count differs")
    for name, record in manifest["artifacts"].items():
        path = ROOT / name
        if (not path.is_file() or path.stat().st_size != record["bytes"]
                or digest(path) != record["sha256"]):
            raise ValueError("An inherited artifact changed: " + name)
    references = [summary[key] for key in ("fresh_scoped_gate", "actual_control",
        "line_control", "motion_control", "immutable_baseline", "source_freeze",
        "frozen_source_map")]
    references.extend(summary["independent_audits"])
    for reference in references:
        checked_reference(reference)
    original = read(checked_reference(summary["frozen_source_map"]))
    if len(original) != 732:
        raise ValueError("The immutable Python source count differs")
    current = sources()
    changed = sorted(name for name, expected in original.items() if current.get(name) != expected)
    if changed:
        raise ValueError("An immutable inherited Python source changed: " + repr(changed))
    clouds = read(ROOT / "handoff/cloud_sources_manifest.json")
    cloud_bytes = 0
    for row in clouds["files"]:
        path = ROOT / row["path"]
        if path.stat().st_size != row["size"] or digest(path) != row["sha256"]:
            raise ValueError("A restored cloud source changed: " + row["path"])
        cloud_bytes += row["size"]
    if len(clouds["files"]) != 9 or cloud_bytes != 2110696:
        raise ValueError("The restored cloud source count or size differs")
    return dict(format="native-birth-contact-immutable-baseline-1", status="PASS",
        verified_utc=datetime.now(timezone.utc).isoformat(),
        prior_authoritative_summary=dict(path=str(SUMMARY.relative_to(ROOT)), sha256=digest(SUMMARY)),
        prior_artifact_manifest=dict(path=str(MANIFEST.relative_to(ROOT)), sha256=digest(MANIFEST)),
        verified_inherited_unique_artifacts=1100,
        inherited_python_source_count=732, inherited_python_sources_unchanged=True,
        inherited_source_map=summary["frozen_source_map"],
        observed_python_source_count=len(current), additions=sorted(set(current)-set(original)),
        inherited_fresh_tests=137, historical_tests_reexecuted=False,
        inherited_native_intervals_accepted=1, inherited_second_candidate_refused=True,
        whole_point_1_complete=False,
        restored_sources=dict(files=9, source_bytes=cloud_bytes, verified=True, restored=False),
        scope="Raw bytes, sizes, selfseals and immutable source authentication; no new force or contact adoption")


if __name__ == "__main__":
    receipt = write(BASE / "immutable_baseline_receipt.json", verify_baseline(), "receipt_sha256")
    print(encoded({key: receipt[key] for key in ("status", "verified_inherited_unique_artifacts",
        "inherited_python_source_count", "observed_python_source_count")}).decode())
