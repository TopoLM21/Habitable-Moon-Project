"""Synthetic metadata archives; no mechanics source or planet is executed.

Temporary repositories make provenance, capture retention and path safety
observable without copying or changing any existing scientific evidence.
"""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys

import pytest

from analysis.genesis_volume_clipped_validation import package_evidence


HELPER_REL = "analysis/genesis_volume_clipped_validation/package_evidence.py"
OLD_REL = "tectonics/original_source.py"
NEW_REL = "tectonics/new_source.py"


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _sealed(body, field="sha256"):
    result = dict(body)
    result[field] = hashlib.sha256(_canonical(result)).hexdigest()
    return result


def _json(path, body):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical(body)+b"\n")
    return path


def _file(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


@pytest.fixture
def repository(tmp_path):
    root = tmp_path/"repository"
    root.mkdir()
    _file(root/OLD_REL, b"raise RuntimeError('archived source must not execute')\n")
    _file(root/NEW_REL, b"raise RuntimeError('new source must not execute during metadata readback')\n")
    helper = root/HELPER_REL
    helper.parent.mkdir(parents=True)
    shutil.copyfile(package_evidence.__file__, helper)
    pins = {OLD_REL: _sha(root/OLD_REL)}
    prior = _json(tmp_path/"prior_sources.json", _sealed({"source_sha256": pins}))
    return dict(root=root, helper=helper, prior=prior, old_pins=pins,
                destination=tmp_path/"new_evidence", temporary=tmp_path)


def _build(repository, **kwargs):
    return package_evidence.build_evidence(repository["root"], repository["destination"],
        repository["prior"], helper_path=repository["helper"], **kwargs)


def _capture(repository, name="capture", exit_code=0, source_contents=None):
    directory = repository["temporary"]/name
    directory.mkdir()
    if source_contents is None:
        source_contents = {relative: (repository["root"]/relative).read_bytes()
                           for relative in (OLD_REL, NEW_REL)}
    source_pins = {}
    for relative, content in source_contents.items():
        source_pins[relative] = _sha(_file(directory/"source_snapshots"/relative, content))
    stdout = _file(directory/"stdout.log", b"synthetic saved stdout\n")
    stderr = _file(directory/"stderr.log", b"" if exit_code == 0 else b"synthetic saved failure\n")
    facts = dict(format="genesis-volume-clipped-bulk-real-subprocess-capture-0.1",
        created_utc="2026-10-11T00:00:00+00:00", command=["python", "synthetic_test_only.py"],
        source_mode="tests", actual_exit_code=exit_code, actual_seconds=.125,
        source_before_sha256=source_pins, source_after_sha256=source_pins,
        declared_source_paths=sorted(source_pins), static_local_import_closure_paths=[],
        import_inventory_scope="synthetic_metadata_fixture_no_subprocess_was_executed",
        changed_sources=[], newly_added_sources=[],
        stdout_raw_sha256=_sha(stdout), stderr_raw_sha256=_sha(stderr),
        stdout_bytes=stdout.stat().st_size, stderr_bytes=stderr.stat().st_size,
        public_numerical_thread_controls={name: 1 for name in
            ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
        thread_control_method="Synthetic fixture, no numerical subprocess")
    _json(directory/"execution_facts.json", _sealed(facts))
    return directory


def _old_package(repository):
    directory = repository["temporary"]/"old_evidence"
    payload = _file(directory/"large_previous_fixture.bin", b"previous fixture retained only in its original directory\n")
    files = [{"path": payload.name, "bytes": payload.stat().st_size, "raw_sha256": _sha(payload)}]
    manifest = _json(directory/"package_manifest.json", _sealed({
        "format": "synthetic-prior-evidence-0.1", "files": files,
        "member_count": 1, "total_member_bytes": payload.stat().st_size}))
    return directory, manifest, payload


def test_build_and_verify_metadata_union_sources_without_executing_them(repository):
    capture = _capture(repository)
    report = _json(repository["temporary"]/"report.json", _sealed({"status": "PASS", "scope": "synthetic_metadata_only"}))
    tree = repository["temporary"]/"control"
    _file(tree/"nested"/"plain.txt", b"saved computational control artifact\n")
    manifest = _build(repository, files={"reports/control.json": report},
        trees={"control": tree}, captures={"tests": capture}, new_sources=(NEW_REL,),
        expected_prior_source_count=1)
    expected = {relative: _sha(repository["root"]/relative)
                for relative in (OLD_REL, NEW_REL, HELPER_REL)}
    assert manifest["source_sha256"] == expected
    assert "package_manifest.json" not in {entry["path"] for entry in manifest["files"]}
    assert len({entry["path"] for entry in manifest["files"]}) == len(manifest["files"])
    for relative, digest in expected.items():
        assert _sha(repository["destination"]/"sources"/relative) == digest
    assert (repository["destination"]/"reports/control.json").read_bytes() == report.read_bytes()
    assert (repository["destination"]/"captures/tests/execution_facts.json").read_bytes() == (capture/"execution_facts.json").read_bytes()
    result = package_evidence.verify_evidence(repository["destination"], repository=repository["root"])
    assert result["status"] == "PASS"
    assert result["scope"] == "metadata_only_no_numerical_or_physical_verdict"
    assert result["numerical_verdict_recomputed"] is False
    assert result["physical_creation_admitted"] is False
    assert result["physical_release_admitted"] is False
    assert result["automatic_goal_enabled"] is False


@pytest.mark.parametrize("wrapper", ("raw", "source_sha256", "current_validated_source_sha256", "captured_source_sha256"))
def test_prior_source_catalogue_supported_shapes_have_same_provenance(repository, wrapper):
    pins = repository["old_pins"]
    body = pins if wrapper == "raw" else _sealed({wrapper: pins})
    _json(repository["prior"], body)
    manifest = _build(repository, expected_prior_source_count=1)
    assert manifest["source_sha256"][OLD_REL] == pins[OLD_REL]
    assert manifest["source_sha256"][HELPER_REL] == _sha(repository["helper"])
    assert package_evidence.verify_evidence(repository["destination"])["status"] == "PASS"


def test_original_references_are_checked_in_place_and_fixture_is_not_duplicated(repository):
    directory, old_manifest, payload = _old_package(repository)
    before = {path.name: path.read_bytes() for path in directory.iterdir()}
    manifest = _build(repository, references={"prior": old_manifest})
    assert before == {path.name: path.read_bytes() for path in directory.iterdir()}
    assert _sha(payload) not in {entry["raw_sha256"] for entry in manifest["files"]}
    assert package_evidence.verify_evidence(repository["destination"])["status"] == "PASS"
    payload.write_bytes(b"changed previous fixture")
    with pytest.raises((ValueError, OSError)):
        package_evidence.verify_evidence(repository["destination"])
    # Disabling original-location checks is explicit and does not certify
    # either the prior package or a numerical result.
    assert package_evidence.verify_evidence(repository["destination"], check_references=False)["scope"] == "metadata_only_no_numerical_or_physical_verdict"


def test_old_reference_extra_member_is_rejected_before_build(repository):
    directory, manifest, _ = _old_package(repository)
    _file(directory/"unlisted.bin", b"not listed in the old sealed manifest")
    with pytest.raises((ValueError, OSError)):
        _build(repository, references={"prior": manifest})


def test_failed_capture_is_retained_byte_exact_and_does_not_become_a_numerical_pass(repository):
    capture = _capture(repository, exit_code=7)
    _build(repository, captures={"failed_control": capture})
    original = json.loads((capture/"execution_facts.json").read_bytes())
    copied = json.loads((repository["destination"]/"captures/failed_control/execution_facts.json").read_bytes())
    assert copied == original
    assert copied["actual_exit_code"] == 7
    result = package_evidence.verify_evidence(repository["destination"], repository=repository["root"])
    assert result["scope"] == "metadata_only_no_numerical_or_physical_verdict"
    assert result["capture_actual_exit_codes"] == {"failed_control": 7}
    assert result["numerical_verdict_recomputed"] is False
    assert (repository["destination"]/"captures/failed_control/stderr.log").read_bytes() == b"synthetic saved failure\n"


def test_custom_selfseal_is_verified_without_treating_foreign_link_hash_as_a_seal(repository):
    body = _sealed({"status": "saved", "parent_sha256": "a"*64}, "custom_result_sha256")
    report = _json(repository["temporary"]/"custom.json", body)
    _build(repository, files={"reports/custom.json": report},
        seals={"reports/custom.json": "custom_result_sha256"})
    assert package_evidence.verify_evidence(repository["destination"])["status"] == "PASS"


def test_primary_json_selfseal_mutation_is_rejected_before_build(repository):
    report = _json(repository["temporary"]/"report.json", _sealed({"status": "FAIL"}))
    body = json.loads(report.read_bytes())
    body["status"] = "PASS"
    _json(report, body)
    with pytest.raises((ValueError, OSError)):
        _build(repository, files={"reports/result.json": report})


@pytest.mark.parametrize("mutate_report", (False, True))
def test_resealed_manifest_cannot_drop_a_primary_json_selfseal_check(repository, mutate_report):
    report = _json(repository["temporary"]/"report.json", _sealed({"status": "FAIL"}))
    _build(repository, files={"reports/result.json": report})
    destination = repository["destination"]
    member = destination/"reports/result.json"
    if mutate_report:
        body = json.loads(member.read_bytes())
        body["status"] = "PASS"  # Deliberately retain the now-invalid inner seal.
        _json(member, body)
    manifest_path = destination/"package_manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest.pop("sha256")
    for entry in manifest["files"]:
        if entry["path"] == "reports/result.json":
            entry["bytes"], entry["raw_sha256"] = member.stat().st_size, _sha(member)
    manifest["total_member_bytes"] = sum(entry["bytes"] for entry in manifest["files"])
    manifest["json_self_seals"].pop("reports/result.json")
    manifest["unsealed_json_members"].append("reports/result.json")
    manifest["unsealed_json_members"].sort()
    _json(manifest_path, _sealed(manifest))
    with pytest.raises((ValueError, OSError)):
        package_evidence.verify_evidence(destination)


@pytest.mark.parametrize("flag", (
    "physical_creation_admitted", "physical_release_admitted",
    "automatic_goal_enabled", "numerical_outcomes_reclassified",
))
def test_resealed_manifest_cannot_elevate_metadata_to_a_physical_verdict(repository, flag):
    _build(repository)
    path = repository["destination"]/"package_manifest.json"
    body = json.loads(path.read_bytes())
    body.pop("sha256")
    body[flag] = True
    _json(path, _sealed(body))
    with pytest.raises((ValueError, OSError)):
        package_evidence.verify_evidence(repository["destination"])


@pytest.mark.parametrize("mutation", ("raw_member", "manifest_selfseal", "extra_member", "missing_member"))
def test_saved_package_tampering_is_rejected(repository, mutation):
    member = _file(repository["temporary"]/"member.bin", b"literal saved result\n")
    _build(repository, files={"results/member.bin": member})
    destination = repository["destination"]
    if mutation == "raw_member":
        (destination/"results/member.bin").write_bytes(b"changed but same metadata")
    elif mutation == "manifest_selfseal":
        path = destination/"package_manifest.json"
        body = json.loads(path.read_bytes())
        body["source_sha256"][OLD_REL] = "0"*64
        _json(path, body)
    elif mutation == "extra_member":
        _file(destination/"unlisted.bin", b"extra")
    else:
        (destination/"results/member.bin").unlink()
    with pytest.raises((ValueError, OSError)):
        package_evidence.verify_evidence(destination)


def test_current_source_changes_reject_build_and_repository_bound_readback(repository):
    _build(repository)
    (repository["root"]/OLD_REL).write_bytes(b"changed after preservation baseline\n")
    with pytest.raises((ValueError, OSError)):
        package_evidence.verify_evidence(repository["destination"], repository=repository["root"])
    repository["destination"] = repository["temporary"]/"another_evidence"
    with pytest.raises((ValueError, OSError)):
        _build(repository)


def test_conflicting_capture_source_pins_are_rejected(repository):
    first = _capture(repository, name="capture1")
    second = _capture(repository, name="capture2", source_contents={
        OLD_REL: (repository["root"]/OLD_REL).read_bytes(),
        NEW_REL: b"different literal executing source\n"})
    with pytest.raises((ValueError, OSError)):
        _build(repository, captures={"first": first, "second": second})


@pytest.mark.parametrize("mutation", ("source_after", "changed_sources", "newly_added_sources", "stdout_hash", "stdout_bytes", "snapshot", "boolean_thread_control"))
def test_resealed_capture_with_false_raw_or_source_facts_is_rejected(repository, mutation):
    capture = _capture(repository)
    path = capture/"execution_facts.json"
    body = json.loads(path.read_bytes())
    body.pop("sha256")
    if mutation == "source_after":
        body["source_after_sha256"][NEW_REL] = "0"*64
    elif mutation == "changed_sources":
        body["changed_sources"] = [NEW_REL]
    elif mutation == "newly_added_sources":
        body["newly_added_sources"] = ["tectonics/hidden_source.py"]
    elif mutation == "stdout_hash":
        body["stdout_raw_sha256"] = "0"*64
    elif mutation == "stdout_bytes":
        body["stdout_bytes"] += 1
    elif mutation == "boolean_thread_control":
        body["public_numerical_thread_controls"]["OPENBLAS_NUM_THREADS"] = True
    else:
        (capture/"source_snapshots"/NEW_REL).write_bytes(b"not the captured source")
    _json(path, _sealed(body))
    with pytest.raises((ValueError, OSError)):
        _build(repository, captures={"tests": capture})


@pytest.mark.parametrize("member", ("../escaped.bin", "/absolute.bin", "safe/../../escaped.bin", "package_manifest.json"))
def test_unsafe_or_reserved_package_member_paths_are_rejected(repository, member):
    source = _file(repository["temporary"]/"input.bin", b"input")
    with pytest.raises((ValueError, OSError)):
        _build(repository, files={member: source})
    assert not (repository["temporary"]/"escaped.bin").exists()


@pytest.mark.parametrize("kind", ("input_file", "input_parent", "input_tree_member", "source_file", "saved_member"))
def test_symlinks_are_rejected_even_when_target_bytes_match(repository, kind):
    source = _file(repository["temporary"]/"input.bin", b"literal input")
    if kind == "input_file":
        alias = repository["temporary"]/"alias.bin"
        alias.symlink_to(source)
        action = lambda: _build(repository, files={"input.bin": alias})
    elif kind == "input_parent":
        original_parent = repository["temporary"]/"original_parent"
        _file(original_parent/"input.bin", b"literal input")
        alias = repository["temporary"]/"parent_alias"
        alias.symlink_to(original_parent, target_is_directory=True)
        action = lambda: _build(repository, files={"input.bin": alias/"input.bin"})
    elif kind == "input_tree_member":
        tree = repository["temporary"]/"tree"
        tree.mkdir()
        (tree/"alias.bin").symlink_to(source)
        action = lambda: _build(repository, trees={"tree": tree})
    elif kind == "source_file":
        original = repository["root"]/OLD_REL
        external = repository["temporary"]/"source_target.py"
        shutil.copyfile(original, external)
        original.unlink()
        original.symlink_to(external)
        action = lambda: _build(repository)
    else:
        _build(repository, files={"input.bin": source})
        (repository["destination"]/"input.bin").unlink()
        (repository["destination"]/"input.bin").symlink_to(source)
        action = lambda: package_evidence.verify_evidence(repository["destination"])
    with pytest.raises((ValueError, OSError)):
        action()


def test_build_refuses_existing_destination_without_touching_it(repository):
    destination = repository["destination"]
    sentinel = _file(destination/"preserve.bin", b"pre-existing data")
    with pytest.raises((ValueError, OSError)):
        _build(repository)
    assert sentinel.read_bytes() == b"pre-existing data"
    assert {path.name for path in destination.iterdir()} == {"preserve.bin"}


def test_expected_prior_count_and_source_path_traversal_are_rejected(repository):
    with pytest.raises((ValueError, OSError)):
        _build(repository, expected_prior_source_count=2)
    outside = _file(repository["temporary"]/"outside.py", b"outside source")
    _json(repository["prior"], _sealed({"source_sha256": {"../outside.py": _sha(outside)}}))
    repository["destination"] = repository["temporary"]/"unsafe_evidence"
    with pytest.raises((ValueError, OSError)):
        _build(repository)


def test_duplicate_json_provenance_keys_are_rejected(repository):
    pins = _canonical(repository["old_pins"])
    repository["prior"].write_bytes(b'{"source_sha256":'+pins+b',"source_sha256":'+pins+b'}\n')
    with pytest.raises((ValueError, OSError)):
        _build(repository)


def test_helper_identity_cannot_be_relabelled_as_an_unrelated_repository_file(repository):
    repository["helper"].write_bytes(b"raise RuntimeError('unrelated code')\n")
    with pytest.raises((ValueError, OSError)):
        _build(repository)


def test_capture_source_catalogue_cannot_include_non_python_artifacts(repository):
    unexpected = "tectonics/not_an_executing_source.txt"
    contents = b"metadata artifact, not Python source\n"
    _file(repository["root"]/unexpected, contents)
    capture = _capture(repository, source_contents={
        OLD_REL: (repository["root"]/OLD_REL).read_bytes(), unexpected: contents})
    with pytest.raises((ValueError, OSError)):
        _build(repository, captures={"tests": capture})


def test_isolated_stdlib_cli_build_and_check_preserve_saved_failure(repository):
    capture = _capture(repository, exit_code=7)
    report = _json(repository["temporary"]/"result.json", _sealed({"status": "saved"}))
    command = [sys.executable, "-I", str(repository["helper"]), "build",
        "--repository", str(repository["root"]),
        "--directory", str(repository["destination"]),
        "--prior-sources", str(repository["prior"]),
        "--expected-prior-source-count", "1", "--source", NEW_REL,
        "--file", "reports/result.json="+str(report),
        "--capture", "failed="+str(capture)]
    built = subprocess.run(command, cwd=repository["temporary"], capture_output=True, text=True)
    assert built.returncode == 0, built.stderr
    assert json.loads(built.stdout)["status"] == "METADATA_PACKAGE_BUILT"
    output = repository["temporary"]/"readback.json"
    checked = subprocess.run([sys.executable, "-I", str(repository["helper"]), "check",
        "--directory", str(repository["destination"]), "--repository", str(repository["root"]),
        "--output", str(output)], cwd=repository["temporary"], capture_output=True, text=True)
    assert checked.returncode == 0, checked.stderr
    result = json.loads(output.read_bytes())
    assert result["capture_actual_exit_codes"] == {"failed": 7}
    assert result["numerical_verdict_recomputed"] is False
    assert result["scope"] == "metadata_only_no_numerical_or_physical_verdict"
    seal = result.pop("sha256")
    assert seal == hashlib.sha256(_canonical(result)).hexdigest()
    assert json.loads(checked.stdout)["report_sha256"] == seal


@pytest.mark.parametrize("kind", ("inside_package", "existing_external"))
def test_cli_readback_refuses_output_that_would_change_saved_artifacts(repository, kind):
    _build(repository)
    if kind == "inside_package":
        output = repository["destination"]/"new_readback.json"
    else:
        output = _file(repository["temporary"]/"existing.json", b"keep existing bytes")
    before = {str(path.relative_to(repository["destination"])): path.read_bytes()
              for path in repository["destination"].rglob("*") if path.is_file()}
    checked = subprocess.run([sys.executable, "-I", str(repository["helper"]), "check",
        "--directory", str(repository["destination"]), "--output", str(output)],
        cwd=repository["temporary"], capture_output=True, text=True)
    assert checked.returncode != 0
    assert before == {str(path.relative_to(repository["destination"])): path.read_bytes()
                      for path in repository["destination"].rglob("*") if path.is_file()}
    if kind == "existing_external":
        assert output.read_bytes() == b"keep existing bytes"
    assert package_evidence.verify_evidence(repository["destination"])["status"] == "PASS"
