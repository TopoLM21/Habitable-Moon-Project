"""Scoped fresh checks and a genuine controller in one immutable-source process.

This preserves the prior release and deliberately does not rerun its 135 tests.
The controller receives an exclusive output directory and owns its physical
assertions; a passing test gate never grants material admission.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
PRIOR = ROOT / "analysis/geometric_native_birth_front_validation"
EXPECTED_SUMMARY = "33a88d35b9f945a7652aa2ad347b789ef9479ff0bbe6cde21c655da1443ee70e"
EXPECTED_MANIFEST = "abf3f7a354ba709f48c967359a7cd56c7f0e9de881f9f8de31b1819cce2834dd"
EXPECTED_SOURCE_MAP = "ce4f84e4953e394c513a1fce819871dd92fa19b52d20dcece1e9674fdbb91fcb"


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def value_digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def sealed(path, key):
    row = read(path)
    expected = row.pop(key)
    if expected != value_digest(row):
        raise ValueError("Evidence selfseal changed: " + str(path))
    return row


def source_map():
    return {str(path.relative_to(ROOT)): digest(path)
        for directory in ("tectonics", "tests", "scripts", "analysis")
        for path in sorted((ROOT / directory).rglob("*.py"))}


def verify_prior():
    summary_path = PRIOR / "release_validation_01/validation_summary.json"
    manifest_path = PRIOR / "release_validation_01/artifact_manifest.json"
    source_path = PRIOR / "frozen_release_sources.json"
    for path, expected in ((summary_path, EXPECTED_SUMMARY),
            (manifest_path, EXPECTED_MANIFEST), (source_path, EXPECTED_SOURCE_MAP)):
        if digest(path) != expected:
            raise ValueError("Authoritative prior release changed: " + str(path))
    summary = sealed(summary_path, "summary_sha256")
    manifest = sealed(manifest_path, "manifest_sha256")
    if (summary.get("status") != "PASS"
            or summary.get("source_count") != 778
            or summary.get("new_whole_native_intervals_accepted") != 0
            or summary.get("original_native_intervals_accepted") != 1
            or summary.get("whole_point_1_completed") is not False
            or len(manifest["artifacts"]) != 1633):
        raise ValueError("Prior release count or physical scope differs")
    for name, row in manifest["artifacts"].items():
        path = ROOT / name
        if path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
            raise ValueError("A prior artifact changed: " + name)
    inherited = read(source_path)
    if len(inherited) != 778:
        raise ValueError("The prior frozen Python source count differs")
    for name, expected in inherited.items():
        if digest(ROOT / name) != expected:
            raise ValueError("A prior Python source changed: " + name)
    return inherited


def parse_counts(path):
    for line in reversed(path.read_text().splitlines()):
        counts = {kind: int(number) for number, kind in re.findall(
            r"\b(\d+) (passed|failed|skipped|xfailed|xpassed|errors?|deselected)\b", line)}
        if counts:
            return line, counts
    return "", {}


def run(out, test_paths, control):
    paths = tuple(test_paths)
    if (not paths or len(set(paths)) != len(paths)
            or any(not name.startswith("tests/test_") or not name.endswith(".py")
                or not (ROOT / name).is_file() for name in paths)
            or ":" not in control):
        raise ValueError("Existing distinct test modules and module:function are required")
    inherited = verify_prior()
    if any(name in inherited for name in paths):
        raise ValueError("This gate accepts only newly added test modules")
    before = source_map()
    # Development may have imported an added file immediately before its last
    # same-size edit.  Remove only added-source bytecode caches, so the final
    # fresh interpreter imports the guarded source text rather than stale pyc.
    removed_caches = []
    for name in sorted(set(before)-set(inherited)):
        cache = Path(importlib.util.cache_from_source(str(ROOT / name)))
        if cache.is_file():
            cache.unlink()
            removed_caches.append(str(cache.relative_to(ROOT)))
    importlib.invalidate_caches()
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = dict(format="native-solid-placement-same-process-validation-1",
        status="FAIL", started_utc=datetime.now(timezone.utc).isoformat(),
        verified_prior_artifacts=1633, verified_prior_python_sources=778,
        prior_summary_raw_sha256=EXPECTED_SUMMARY, prior_manifest_raw_sha256=EXPECTED_MANIFEST,
        prior_source_map_raw_sha256=EXPECTED_SOURCE_MAP,
        source_count=len(before), source_map_before_sha256=value_digest(before),
        added_python_sources={name: before[name] for name in sorted(set(before)-set(inherited))},
        removed_added_source_bytecode_caches=removed_caches,
        historical_135_tests_reexecuted=False, test_paths=list(paths), control=control,
        execution_mode="same_python_process", pytest_import_mode="importlib",
        whole_native_interval_admitted_by_test_gate=False)
    try:
        import pytest
        command = ["-q", "-p", "no:cacheprovider", "--import-mode=importlib", *paths]
        pytest_started = time.perf_counter()
        original_directory = Path.cwd()
        try:
            os.chdir(ROOT)
            with (out / "pytest.log").open("x", buffering=1) as stream:
                with redirect_stdout(stream), redirect_stderr(stream):
                    exit_code = int(pytest.main(command))
        finally:
            os.chdir(original_directory)
        summary, counts = parse_counts(out / "pytest.log")
        report.update(pytest_exit_code=exit_code, pytest_summary=summary,
            pytest_counts=counts, pytest_seconds=time.perf_counter()-pytest_started)
        if (exit_code != 0 or counts.get("passed", 0) < 1
                or any(counts.get(name, 0) for name in
                    ("failed", "skipped", "xfailed", "xpassed", "error", "errors"))):
            raise RuntimeError("Fresh scoped pytest gate failed")
        if source_map() != before:
            raise ValueError("A Python source changed during pytest")
        module, function = control.split(":", 1)
        callback = getattr(importlib.import_module(module), function)
        control_started = time.perf_counter()
        with (out / "control_console.log").open("x", buffering=1) as stream:
            with redirect_stdout(stream), redirect_stderr(stream):
                callback(out / "native_control")
        report["native_control_seconds"] = time.perf_counter()-control_started
        report["status"] = "PASS"
    except BaseException as error:
        report["failure_type"] = type(error).__name__
        report["failure_message"] = str(error)
        report["failure_traceback"] = traceback.format_exc()
    finally:
        after = source_map()
        report["source_map_after_sha256"] = value_digest(after)
        report["changed_python_paths"] = sorted(name for name in set(before) | set(after)
            if before.get(name) != after.get(name))
        try:
            verify_prior()
            report["prior_artifacts_and_python_sources_preserved_after"] = True
        except BaseException as error:
            report["prior_artifacts_and_python_sources_preserved_after"] = False
            report["prior_after_error"] = str(error)
        if report["changed_python_paths"] or not report["prior_artifacts_and_python_sources_preserved_after"]:
            report["status"] = "FAIL"
        report["seconds"] = time.perf_counter()-started
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        report["outputs"] = {str(path.relative_to(out)): dict(
            sha256=digest(path), bytes=path.stat().st_size)
            for path in sorted(out.rglob("*")) if path.is_file()}
        report["report_sha256"] = value_digest(report)
        with (out / "report.json").open("xb") as stream:
            stream.write(encoded(report)+b"\n")
        print(encoded({key: report.get(key) for key in
            ("status", "pytest_counts", "seconds", "native_control_seconds",
             "source_count", "changed_python_paths")}).decode(), flush=True)
    if report["status"] != "PASS":
        raise RuntimeError("Validation refused; see " + str(out / "report.json"))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--control", required=True)
    parser.add_argument("test_paths", nargs="+")
    options = parser.parse_args()
    run(options.out.resolve(), options.test_paths, options.control)
