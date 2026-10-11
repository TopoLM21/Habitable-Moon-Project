"""Authenticate the complete immutable release before front completion.

This is read-only evidence verification.  It does not rerun historical tests
or convert the previously reduced two-contact experiment into an admission.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
PRIOR = ROOT / "analysis/geometric_native_birth_contact_validation"
SUMMARY = PRIOR / "release_validation_01/validation_summary.json"
MANIFEST = PRIOR / "release_validation_01/artifact_manifest.json"
EXPECTED_SUMMARY_RAW = "6b9e2bc2361ddeaff2e43199f15c5562ea29fe0b3c50ed6091f4efcae3b4b9f4"
EXPECTED_MANIFEST_RAW = "7694849110d82e2d8c7d35ff004e84e0be584cfdee0ef0b182bb6e13de806d92"


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
        raise ValueError("The authoritative newborn-contact release changed")
    summary, manifest = sealed(SUMMARY, "summary_sha256"), sealed(MANIFEST, "manifest_sha256")
    required = {"status": "PASS", "source_count": 751,
        "original_native_intervals_accepted": 1, "new_whole_native_intervals_accepted": 0,
        "explicit_birth_law_implemented": True, "genuine_reduced_current_source_force_solved": True,
        "reduced_born_contact_atoms": 2, "independent_foundation_count": 5,
        "whole_native_interval_admitted": False, "whole_point_1_completed": False,
        "remaining_work_block_count": 1}
    if any(summary.get(name) != value for name, value in required.items()):
        raise ValueError("The inherited physical scope differs")
    if (summary["fresh_scoped_gate"]["tests_passed"] != 69
            or summary["artifact_manifest"]["sha256"] != digest(MANIFEST)
            or manifest["count"] != 1271 or manifest["unique_artifact_count"] != 1271
            or len(manifest["artifacts"]) != 1271):
        raise ValueError("The inherited test or artifact count differs")
    for name, record in manifest["artifacts"].items():
        path = ROOT / name
        if (not path.is_file() or path.stat().st_size != record["bytes"]
                or digest(path) != record["sha256"]):
            raise ValueError("An inherited artifact changed: " + name)
    for name in ("fresh_scoped_gate", "same_process_suite", "actual_control",
            "local_constitutive_control", "native_reduced_control", "source_freeze", "frozen_source_map",
            "inherited_validation", "inherited_artifact_manifest"):
        checked_reference(summary[name])
    for reference in summary["independent_audits"]:
        checked_reference(reference)
    original = read(checked_reference(summary["frozen_source_map"]))
    if len(original) != 751:
        raise ValueError("The inherited Python source count differs")
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
        raise ValueError("The cloud source count or size differs")
    return dict(format="native-birth-front-immutable-baseline-1", status="PASS",
        verified_utc=datetime.now(timezone.utc).isoformat(),
        prior_authoritative_summary=dict(path=str(SUMMARY.relative_to(ROOT)), sha256=digest(SUMMARY)),
        prior_artifact_manifest=dict(path=str(MANIFEST.relative_to(ROOT)), sha256=digest(MANIFEST)),
        verified_inherited_unique_artifacts=1271, inherited_python_source_count=751,
        inherited_python_sources_unchanged=True, inherited_source_map=summary["frozen_source_map"],
        observed_python_source_count=len(current), additions=sorted(set(current)-set(original)),
        inherited_fresh_tests=69, historical_69_tests_reexecuted=False,
        historical_137_tests_reexecuted=False, inherited_native_intervals_accepted=1,
        inherited_reduced_born_atoms=2, inherited_reduced_trial_has_native_admission=False,
        whole_point_1_complete=False,
        restored_sources=dict(files=9, source_bytes=cloud_bytes, verified=True, restored=False),
        scope="Hash, size, selfseal and immutable source authentication; no fresh continuum force or temporal adoption")


if __name__ == "__main__":
    receipt = write(BASE / "immutable_baseline_receipt.json", verify_baseline(), "receipt_sha256")
    print(encoded({key: receipt[key] for key in ("status", "verified_inherited_unique_artifacts",
        "inherited_python_source_count", "observed_python_source_count")}).decode())
