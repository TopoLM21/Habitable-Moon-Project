"""Assemble and read back an explicit, fresh clipped-bulk evidence package.

Standard-library metadata only: no producer imports, restores, subprocesses or
numerical verdicts.  Captures retain their actual exit codes, including FAIL.
Old large fixtures and packages are pinned and checked in place.  Current
source bytes must equal every prior/captured source pin, without exceptions.
The package manifest excludes itself and seals all literal payload members.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import shutil


FORMAT = "genesis-volume-clipped-evidence-package-0.1"
SCOPE = "metadata_only_no_numerical_or_physical_verdict"
SELF_FIELDS = ("sha256", "report_sha256", "manifest_sha256", "archive_sha256",
               "diagnosis_sha256", "packet_sha256", "fixture_sha256")
SHA = re.compile(r"[0-9a-f]{64}\Z")
THREADS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
MANIFEST = "package_manifest.json"


class EvidenceRefusal(ValueError):
    """A selected metadata/source artifact differs or has an unsafe path."""


def require(condition, message):
    if not condition:
        raise EvidenceRefusal(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key: " + key)
        result[key] = value
    return result


def read_json(path):
    return json.loads(Path(path).read_bytes(), object_pairs_hook=_unique,
        parse_constant=lambda value: (_ for _ in ()).throw(
            EvidenceRefusal("Nonfinite JSON value: " + value)))


def safe_path(path):
    path = Path(path).absolute()
    require(not any(p.is_symlink() for p in (path, *path.parents)),
            "Symlink in explicit path: " + str(path))
    return path.resolve()


def relative(path):
    require(isinstance(path, str), "Member/source name must be a string")
    value = Path(path)
    require(value.parts and not value.is_absolute() and ".." not in value.parts
            and str(value) == path and "\\" not in path,
            "Unsafe relative member/source name: " + path)
    return value


def label(value):
    require(isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.-]+", value)
            and value not in (".", ".."), "Unsafe artifact name: " + str(value))
    return value


def pin(path):
    path = safe_path(path)
    require(path.is_file(), "Missing regular file: " + str(path))
    sha, size = hashlib.sha256(), 0
    with path.open("rb") as stream:
        while data := stream.read(1 << 20):
            sha.update(data)
            size += len(data)
    return {"bytes": size, "raw_sha256": sha.hexdigest()}


def tree_inventory(directory):
    directory = safe_path(directory)
    require(directory.is_dir(), "Missing artifact directory: " + str(directory))
    result = {}
    for path in sorted(directory.rglob("*")):
        require(not path.is_symlink(), "Symlink in artifact tree: " + str(path))
        if path.is_file():
            result[str(path.relative_to(directory))] = pin(path)
        else:
            require(path.is_dir(), "Non-regular artifact entry: " + str(path))
    return result


def json_self_seal(path, field=None):
    """Check the primary canonical self seal, not foreign linked hashes."""
    value = read_json(path)
    if type(value) is not dict:
        require(field is None, "Sealed JSON must be an object: " + str(path))
        return None
    if field is None:
        field = next((name for name in SELF_FIELDS if name in value), None)
    if field is None:
        return None
    require(isinstance(value.get(field), str) and SHA.fullmatch(value[field]),
            "Invalid canonical self-seal field: " + str(path) + " / " + field)
    require(value[field] == digest({k: v for k, v in value.items() if k != field}),
            "Canonical self seal differs: " + str(path) + " / " + field)
    return {"field": field, "self_sha256": value[field]}


def _source_map(path):
    value = read_json(path)
    json_self_seal(path)
    require(type(value) is dict, "Source pin catalogue must be an object")
    names = ("source_sha256", "current_validated_source_sha256", "captured_source_sha256")
    candidates = [value[name] for name in names if name in value]
    mapping = candidates[0] if candidates else value
    require(type(mapping) is dict and mapping, "Source pin catalogue is empty")
    for name, expected in mapping.items():
        require(str(relative(name)).endswith(".py") and isinstance(expected, str)
                and SHA.fullmatch(expected), "Invalid source pin: " + str(name))
    return dict(mapping)


def _merge_sources(union, values):
    for name, expected in values.items():
        relative(name)
        require(name.endswith(".py") and isinstance(expected, str) and SHA.fullmatch(expected),
                "Invalid captured source pin: " + name)
        require(name not in union or union[name] == expected,
                "Conflicting frozen source versions: " + name)
        union[name] = expected


def _verify_sources(repository, values):
    repository = safe_path(repository)
    for name, expected in values.items():
        require(pin(repository / relative(name))["raw_sha256"] == expected,
                "Frozen/current source differs: " + name)


def _capture_record(directory, repository=None):
    directory = safe_path(directory)
    facts_path = directory / "execution_facts.json"
    self_pin = json_self_seal(facts_path, "sha256")
    facts = read_json(facts_path)
    before, after = facts.get("source_before_sha256"), facts.get("source_after_sha256")
    require(type(before) is dict and before and before == after,
            "Captured source before/after pins differ: " + str(directory))
    require(facts.get("changed_sources") == [] and facts.get("newly_added_sources") == [],
            "Capture changed or added executing sources: " + str(directory))
    sources = {}
    _merge_sources(sources, before)
    snapshots = tree_inventory(directory / "source_snapshots")
    require(set(snapshots) == set(sources), "Capture exact source snapshot inventory differs")
    for name, expected in sources.items():
        require(snapshots[name]["raw_sha256"] == expected,
                "Captured source snapshot differs: " + name)
    if repository is not None:
        _verify_sources(repository, sources)
    for name in ("stdout", "stderr"):
        require(pin(directory / (name + ".log")) == {
            "bytes": facts.get(name + "_bytes"),
            "raw_sha256": facts.get(name + "_raw_sha256")}, "Capture raw log differs: " + name)
    require(type(facts.get("actual_exit_code")) is int,
            "Capture actual exit code must be an integer")
    seconds = facts.get("actual_seconds")
    require(type(seconds) in (int, float) and math.isfinite(seconds) and seconds >= 0,
            "Capture actual elapsed time differs")
    controls = facts.get("public_numerical_thread_controls")
    if controls is not None:
        require(controls == {name: 1 for name in THREADS}
                and all(type(value) is int for value in controls.values()),
                "Numerical thread controls differ")
    return dict(facts=pin(facts_path), facts_self_sha256=self_pin["self_sha256"],
        actual_exit_code=facts["actual_exit_code"], actual_seconds=seconds,
        source_sha256=sources, numerical_result_interpreted=False)


def _catalogue_files(manifest):
    records = manifest.get("files")
    require(type(records) is list, "Package manifest file catalogue differs")
    entries = {}
    for item in records:
        require(type(item) is dict, "Package file entry must be an object")
        name = str(relative(item.get("path")))
        require(name != MANIFEST and name not in entries, "Duplicate/reserved package member: " + name)
        expected = item.get("raw_sha256", item.get("sha256"))
        require(type(item.get("bytes")) is int and item["bytes"] >= 0
                and isinstance(expected, str) and SHA.fullmatch(expected),
                "Invalid package raw member pin: " + name)
        entries[name] = {"bytes": item["bytes"], "raw_sha256": expected}
    return entries


def _prior_package(path):
    """Verify an old sealed package in place, without copying its payload."""
    path = safe_path(path)
    value = read_json(path)
    entries = _catalogue_files(value)
    actual = tree_inventory(path.parent)
    require(set(actual) == set(entries) | {path.name}, "Prior package exact inventory differs")
    for name, expected in entries.items():
        require(actual[name] == expected, "Prior package member differs: " + name)
    return {"member_count": len(entries), "total_member_bytes": sum(p["bytes"] for p in entries.values())}


def _reference_record(path):
    path = safe_path(path)
    result = {"path": str(path), **pin(path)}
    if path.suffix.lower() == ".json":
        result["json_self_seal"] = json_self_seal(path)
    if path.name == MANIFEST:
        result["prior_package_inventory"] = _prior_package(path)
    return result


def build_evidence(repository, destination, prior_source_file, *, files=None, trees=None,
        captures=None, references=None, seals=None, new_sources=(),
        expected_prior_source_count=None, helper_path=None):
    """Build one fresh package from explicit saved artifacts, no execution."""
    repository, destination = safe_path(repository), safe_path(destination)
    prior_source_file = safe_path(prior_source_file)
    require(not destination.exists(), "Fresh package destination required")
    files, trees, captures = dict(files or {}), dict(trees or {}), dict(captures or {})
    references, seals = dict(references or {}), dict(seals or {})
    prior_sources = _source_map(prior_source_file)
    require(expected_prior_source_count is None or len(prior_sources) == expected_prior_source_count,
            "Prior source count differs")
    source_union, capture_records, reference_records = dict(prior_sources), {}, {}
    for name, directory in sorted(captures.items()):
        label(name)
        record = _capture_record(directory, repository)
        _merge_sources(source_union, record["source_sha256"])
        capture_records[name] = record
    helper = safe_path(helper_path or __file__)
    require(helper.is_relative_to(repository), "Executing helper must be a literal repository source")
    require(pin(helper) == pin(__file__), "Declared executing helper differs from the actual metadata helper")
    source_names = set(new_sources) | {str(helper.relative_to(repository))}
    for name in sorted(source_names):
        relative(name)
        require(name.endswith(".py"), "New source must be Python: " + name)
        _merge_sources(source_union, {name: pin(repository / name)["raw_sha256"]})
    _verify_sources(repository, source_union)
    for name, path in sorted(references.items()):
        reference_records[label(name)] = _reference_record(path)
    planned = {}

    def add(path, name):
        name = str(relative(name))
        require(name != MANIFEST and name not in planned, "Duplicate/reserved package member: " + name)
        planned[name] = (safe_path(path), pin(path))

    def add_tree(path, prefix):
        prefix = str(relative(prefix))
        for name in tree_inventory(path):
            add(Path(path) / name, str(Path(prefix) / name))

    for name, path in sorted(files.items()):
        add(path, name)
    for prefix, path in sorted(trees.items()):
        add_tree(path, prefix)
    for name, path in sorted(captures.items()):
        add_tree(path, "captures/" + name)
    for name in sorted(source_union):
        add(repository / name, "sources/" + name)
    add(prior_source_file, "provenance/prior_source_pins.json")
    add(helper, "readback/package_evidence.py")
    require(set(seals) <= set(planned), "Self-seal specification names an absent package member")
    self_seals, unsealed_json = {}, []
    for name, (path, _) in planned.items():
        if path.suffix.lower() == ".json" or name in seals:
            self_pin = json_self_seal(path, seals.get(name))
            if self_pin is not None:
                self_seals[name] = self_pin
            else:
                unsealed_json.append(name)
    destination.mkdir(parents=True, exist_ok=False)
    for name, (path, expected) in sorted(planned.items()):
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        require(pin(path) == expected == pin(target), "Artifact changed while archiving: " + str(path))
    # Check source/reference stability again after all copies.  Failure leaves
    # a visibly incomplete fresh directory with no PASS manifest.
    _verify_sources(repository, source_union)
    for name, path in sorted(references.items()):
        require(_reference_record(path) == reference_records[name], "External reference changed during assembly: " + name)
    manifest = dict(format=FORMAT, scope=SCOPE,
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        files=[{"path": name, **expected} for name, (_, expected) in sorted(planned.items())],
        member_count=len(planned), total_member_bytes=sum(expected["bytes"] for _, expected in planned.values()),
        source_sha256=source_union, prior_source_sha256=prior_sources,
        prior_source_count=len(prior_sources), new_source_paths=sorted(set(source_union)-set(prior_sources)),
        source_version_policy="all_prior_captured_and_current_source_bytes_equal_no_version_exceptions",
        captures=capture_records, external_references=reference_records,
        json_self_seals=self_seals, unsealed_json_members=sorted(unsealed_json),
        executed_helper={"repository_path": str(helper.relative_to(repository)),
                         "member_path": "readback/package_evidence.py", **pin(helper)},
        package_manifest_excludes_itself=True,
        numerical_outcomes_reclassified=False, physical_creation_admitted=False,
        physical_release_admitted=False, automatic_goal_enabled=False)
    manifest["sha256"] = digest(manifest)
    (destination / MANIFEST).write_bytes(canonical(manifest) + b"\n")
    verify_evidence(destination, repository=repository)
    return manifest


def verify_evidence(directory, *, repository=None, check_references=True):
    """Read raw/self/source metadata only, with no producer import or replay."""
    directory = safe_path(directory)
    manifest_path = directory / MANIFEST
    json_self_seal(manifest_path, "sha256")
    manifest = read_json(manifest_path)
    require(manifest.get("format") == FORMAT and manifest.get("scope") == SCOPE,
            "Clipped package format or scope differs")
    require(manifest.get("package_manifest_excludes_itself") is True
            and manifest.get("numerical_outcomes_reclassified") is False,
            "Package metadata scope differs")
    for flag in ("physical_creation_admitted", "physical_release_admitted", "automatic_goal_enabled"):
        require(manifest.get(flag) is False, "Package cannot elevate physical admission or restart the goal")
    entries = _catalogue_files(manifest)
    actual = tree_inventory(directory)
    require(set(actual) == set(entries) | {MANIFEST}, "Package exact inventory differs")
    require(manifest.get("member_count") == len(entries)
            and manifest.get("total_member_bytes") == sum(p["bytes"] for p in entries.values()),
            "Package member totals differ")
    for name, expected in entries.items():
        require(actual[name] == expected, "Package literal member differs: " + name)
    saved_seals = manifest.get("json_self_seals", {})
    require(type(saved_seals) is dict and set(saved_seals) <= set(entries),
            "Package JSON self-seal member catalogue differs")
    actual_seals, unsealed = {}, []
    for name in entries:
        if Path(name).suffix.lower() == ".json" or name in saved_seals:
            expected = saved_seals.get(name)
            checked = json_self_seal(directory / name, expected["field"] if expected else None)
            if checked is None:
                unsealed.append(name)
            else:
                actual_seals[name] = checked
    require(actual_seals == saved_seals and sorted(unsealed) == manifest.get("unsealed_json_members"),
            "Package complete JSON self-seal catalogue differs")
    prior = _source_map(directory / "provenance/prior_source_pins.json")
    require(prior == manifest.get("prior_source_sha256") and len(prior) == manifest.get("prior_source_count"),
            "Archived prior source catalogue differs")
    union = dict(prior)
    for name, expected in manifest.get("captures", {}).items():
        label(name)
        current = _capture_record(directory / "captures" / name, repository)
        require(current == expected, "Archived capture facts/source/log binding differs: " + name)
        _merge_sources(union, current["source_sha256"])
    for name in manifest.get("new_source_paths", []):
        require(name in manifest.get("source_sha256", {}), "New source list contains an absent source")
        _merge_sources(union, {name: manifest["source_sha256"][name]})
    require(union == manifest.get("source_sha256"), "Merged source freeze catalogue differs")
    for name, expected in union.items():
        member = "sources/" + name
        require(member in entries and entries[member]["raw_sha256"] == expected,
                "Archived source freeze differs: " + name)
    if repository is not None:
        _verify_sources(repository, union)
    helper = manifest.get("executed_helper", {})
    helper_source = helper.get("repository_path")
    require(helper.get("member_path") == "readback/package_evidence.py"
            and helper_source in union and entries[helper["member_path"]] == {
                "bytes": helper.get("bytes"), "raw_sha256": helper.get("raw_sha256")}
            and helper.get("raw_sha256") == union[helper_source],
            "Archived package helper differs from frozen executing source")
    if check_references:
        for name, expected in manifest.get("external_references", {}).items():
            label(name)
            require(_reference_record(expected["path"]) == expected, "Pinned external reference differs: " + name)
    report = dict(format="genesis-volume-clipped-evidence-readback-0.1", scope=SCOPE,
        status="PASS", all_metadata_checks_pass=True,
        package_manifest=pin(manifest_path), package_manifest_self_sha256=manifest["sha256"],
        member_count=len(entries), total_member_bytes=manifest["total_member_bytes"],
        source_count=len(union), prior_source_count=len(prior),
        capture_actual_exit_codes={name: value["actual_exit_code"] for name, value in manifest.get("captures", {}).items()},
        current_repository_sources_checked=repository is not None,
        external_references_checked=bool(check_references),
        numerical_verdict_recomputed=False, physical_creation_admitted=False,
        physical_release_admitted=False, automatic_goal_enabled=False)
    report["sha256"] = digest(report)
    return report


def _named(values):
    result = {}
    for value in values:
        require("=" in value, "Expected NAME=PATH or MEMBER=FIELD: " + value)
        name, path = value.split("=", 1)
        require(name not in result and path, "Duplicate/empty named argument: " + name)
        result[name] = path
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    build = sub.add_parser("build")
    build.add_argument("--repository", type=Path, required=True)
    build.add_argument("--directory", type=Path, required=True)
    build.add_argument("--prior-sources", type=Path, required=True)
    build.add_argument("--expected-prior-source-count", type=int)
    build.add_argument("--source", action="append", default=[])
    for name in ("file", "tree", "capture", "reference", "seal"):
        build.add_argument("--" + name, action="append", default=[])
    check = sub.add_parser("check")
    check.add_argument("--directory", type=Path, required=True)
    check.add_argument("--repository", type=Path)
    check.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.mode == "build":
        manifest = build_evidence(args.repository, args.directory, args.prior_sources,
            files=_named(args.file), trees=_named(args.tree), captures=_named(args.capture),
            references=_named(args.reference), seals=_named(args.seal), new_sources=args.source,
            expected_prior_source_count=args.expected_prior_source_count)
        print(json.dumps({"status": "METADATA_PACKAGE_BUILT", "directory": str(args.directory),
            "manifest_sha256": manifest["sha256"], "member_count": manifest["member_count"],
            "source_count": len(manifest["source_sha256"]), "scope": SCOPE}, sort_keys=True))
    else:
        report = verify_evidence(args.directory, repository=args.repository)
        if args.output is not None:
            output = safe_path(args.output)
            require(not output.exists() and not output.is_relative_to(safe_path(args.directory)),
                    "Readback output must be fresh and outside the sealed package")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(canonical(report) + b"\n")
        print(json.dumps({"status": report["status"], "scope": report["scope"],
            "member_count": report["member_count"], "source_count": report["source_count"],
            "report_sha256": report["sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
