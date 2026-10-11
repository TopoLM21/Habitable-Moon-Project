"""Immutable ancestry and source guards for the narrow active-binding release."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
PREVIOUS = ROOT / "analysis/geometric_native_domain_contact_handover_validation"


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()


def value_sha(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path, value, *, seal=None):
    value = deepcopy(value)
    if seal is not None:
        if seal in value:
            raise ValueError("A new seal cannot overwrite an existing declaration")
        value[seal] = value_sha(value)
    with Path(path).open("xb") as stream:
        stream.write(encoded(value) + b"\n")
    return value


def sealed(path, key):
    value = read(path)
    expected = value.pop(key)
    if expected != value_sha(value):
        raise ValueError("Evidence selfseal differs: " + str(path))
    return value


def sources():
    return {str(path.relative_to(ROOT)): digest(path)
        for directory in ("tectonics", "tests", "scripts", "analysis")
        for path in sorted((ROOT / directory).rglob("*.py"))}


def guard_sources(*, require_frozen=False):
    current, original = sources(), read(BASE / "starting_sources.json")
    if len(original) != 694:
        raise ValueError("Active binding must start with the authenticated 694-source release")
    authoritative_maps = [PREVIOUS / directory / ("sources_" + stage + ".json")
        for directory in ("release_gates_02", "control_release_01")
        for stage in ("before", "after")]
    if any(read(path) != original for path in authoritative_maps):
        raise ValueError("The four prior complete source maps differ")
    changed = sorted(name for name, expected in original.items() if current.get(name) != expected)
    if changed:
        raise ValueError("A prior Python source changed: " + repr(changed))
    frozen = BASE / "frozen_release_sources.json"
    if require_frozen and (not frozen.is_file() or read(frozen) != current):
        raise ValueError("The complete release source map must be frozen before execution")
    return current, original


def verify_previous_sources():
    receipt = sealed(BASE / "starting_validation_receipt.json", "receipt_sha256")
    if (receipt["status"] != "PASS" or receipt["original_python_source_count"] != 694
            or receipt["verified_prior_manifest_records"] != 302
            or receipt["prior_tests_reexecuted"] is not False
            or receipt["starting_sources_sha256"] != digest(BASE / "starting_sources.json")):
        raise ValueError("The sealed initial active-binding baseline differs")
    count = 0
    for row in receipt["ancestry"]:
        summary_path, manifest_path = ROOT / row["path"], ROOT / row["manifest_path"]
        if digest(summary_path) != row["sha256"] or digest(manifest_path) != row["manifest_sha256"]:
            raise ValueError("A prior release document changed")
        summary = sealed(summary_path, "summary_sha256")
        manifest = sealed(manifest_path, "manifest_sha256")
        if (summary["status"] != "PASS" or manifest["count"] != row["verified_manifest_records"]
                or len(manifest["artifacts"]) != row["verified_manifest_records"]
                or summary["artifact_manifest"]["sha256"] != digest(manifest_path)):
            raise ValueError("A prior manifest changed")
        for name, record in manifest["artifacts"].items():
            path = ROOT / name
            if not path.is_file() or path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
                raise ValueError("A prior artifact changed: " + name)
        for key in ("fresh_scoped_gate", "release_gate", "actual_control", "independent_audit"):
            if key in summary:
                reference = summary[key]
                if digest(ROOT / reference["path"]) != reference["sha256"]:
                    raise ValueError("A prior evidence reference changed: " + key)
        count += manifest["count"]
    if count != 302:
        raise ValueError("The complete 302-record ancestry was not verified")
    return receipt


def rehash_declared_seals(value):
    """Test a semantic forgery despite valid declared integrity hashes."""
    if type(value) is list:
        for item in value:
            rehash_declared_seals(item)
    elif type(value) is dict:
        for name, item in value.items():
            # The mutations in this control do not alter the independently
            # supplied original source. Its old path hashes include links
            # and source-specific payloads, not generic owning selfseals.
            if name != "source_history":
                rehash_declared_seals(item)
        if value.get("format") == "native-active-frozen-contact-segment-work-1":
            # This is a reference link, while work_sha256 is the owning seal.
            value["reference_sha256"] = value["reference"]["reference_sha256"]
            value["work_sha256"] = value_sha({name: item for name, item in value.items()
                if name != "work_sha256"})
            return
        key = {
            "native-active-receiver-contact-binding-path-1": "path_sha256",
            "native-active-frozen-contact-segment-reference-1": "reference_sha256",
            "native-active-contact-path-work-query-1": "query_sha256",
        }.get(value.get("format"))
        if key is not None and key in value:
            value[key] = value_sha({name: item for name, item in value.items() if name != key})


def refuse_mutation(raw, *, mutation, loader, label, changed_path):
    forged = deepcopy(raw)
    mutation(forged)
    rehash_declared_seals(forged)
    try:
        loader(forged)
    except (ValueError, TypeError) as error:
        return dict(label=label, changed_path=changed_path, rejected=True,
            valid_rehashed_declared_checksum=True, exception_type=type(error).__name__,
            message=str(error), mutated_dto_sha256=value_sha(forged))
    raise AssertionError("A semantic mutation was accepted: " + label)
