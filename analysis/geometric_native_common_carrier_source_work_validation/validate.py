"""One frozen cohesive source-handover gate; authentic saved DTOs only.

Authentic saved geometry and material DTOs are restored and validated.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[2]
PARENT = Path(__file__).resolve().parent
STAGE = Path("/workspace/carrier_source_work_baseline_20261007")
PRESERVATION_SOURCE = STAGE / "baseline_887_preservation_source.txt"
PRESERVATION_RAW = "aa191e6106ff2b397a5738afea209c626e8bd99c3e4de76718a5a46039f63f04"
OLD_FROZEN = ROOT / "analysis/geometric_native_archive_material_carrier_validation/release_validation_02_frozen_source_map_887.json"
ADDED = frozenset((
    "tectonics/geometric_native_common_carrier_source_work.py",
    "tests/test_geometric_native_common_carrier_source_work.py",
    "analysis/geometric_native_common_carrier_source_work_validation/native_control.py",
    "analysis/geometric_native_common_carrier_source_work_validation/validate.py",
))
TEST_PATH = "tests/test_geometric_native_common_carrier_source_work.py"
sys.path.insert(0, str(ROOT))


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def value_digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_map():
    return {str(path.relative_to(ROOT)): digest(path)
        for directory in ("tectonics", "tests", "scripts", "analysis")
        for path in sorted((ROOT / directory).rglob("*.py"))}


def preservation(additions, output):
    if digest(PRESERVATION_SOURCE) != PRESERVATION_RAW:
        raise ValueError("The reviewed immutable887 preservation source changed")
    module = types.ModuleType("_archive_carrier_stdlib_preservation")
    module.__file__ = str(PRESERVATION_SOURCE)
    exec(compile(PRESERVATION_SOURCE.read_bytes(), str(PRESERVATION_SOURCE), "exec"), module.__dict__)
    receipt = module.run(additions, output_path=output)
    if (receipt["status"] != "PASS_IMMUTABLE887_WITH_EXACT_FOUR_DECLARED_SOURCE_ADDITIONS"
            or receipt["current_python_source_count"] != 891
            or receipt["additional_python_source_sha256"] != additions
            or receipt["preserved_prior883_raw_dependency_pin_count"] != 341):
        raise ValueError("Incomplete previous-source preservation result")
    return receipt


class TestResults:
    def __init__(self):
        self.collected = 0
        self.passed = 0
        self.failed = 0
        self.skipped = 0

    def pytest_collection_finish(self, session):
        self.collected = len(session.items)

    def pytest_runtest_logreport(self, report):
        if report.when == "call" and report.passed:
            self.passed += 1
        if report.failed:
            self.failed += 1
        if report.skipped:
            self.skipped += 1


def run(release_name):
    if (type(release_name) is not str or not release_name.startswith("release_validation_")
            or "/" in release_name or "\\" in release_name or ".." in release_name):
        raise ValueError("A fresh simple release directory name is required")
    release = PARENT / release_name
    release.mkdir(exist_ok=False)
    started = time.monotonic()
    frozen = source_map()
    prior = json.loads(OLD_FROZEN.read_bytes())
    if (len(prior) != 887 or len(frozen) != 891
            or set(frozen) - set(prior) != ADDED
            or any(frozen.get(name) != pin for name, pin in prior.items())):
        raise ValueError("Exactly four additive sources are required, with all887 unchanged")
    frozen_path = PARENT / (release_name + "_frozen_source_map_891.json")
    with frozen_path.open("xb") as stream:
        stream.write(encoded(frozen) + b"\n")
    additions = {name: frozen[name] for name in sorted(ADDED)}
    snapshots = PARENT / "source_snapshots" / release_name
    snapshots.mkdir(parents=True, exist_ok=False)
    for name in sorted(ADDED):
        target = snapshots / (name.replace("/", "__") + ".txt")
        shutil.copyfile(ROOT / name, target)
    report = dict(format="native-common-carrier-source-work-validation-1", status="FAIL",
        started_at_utc=datetime.now(timezone.utc).isoformat(), source_count=891,
        python_executable=sys.executable, python_version=sys.version,
        frozen_source_map_sha256=value_digest(frozen),
        frozen_source_map_raw_sha256=digest(frozen_path),
        frozen_source_map_path=str(frozen_path.relative_to(ROOT)),
        additional_source_sha256=additions,
        executing_source_snapshots={str(path.relative_to(ROOT)): digest(path) for path in sorted(snapshots.iterdir())},
        previous_source_count=887, old_force_tests_or_tensor_quadrature_reexecuted=False,
        saved_geometry_and_material_restored_and_validated=False,
        current_physical_scope="DECLARED_ARCHIVE_CARRIER_REGISTERED_COMMON_COHESIVE_PREPARATION_ONLY",
        tests_exit_code=None, native_control_succeeded=False)
    before_succeeded = False
    pending_error = None
    try:
        before_path = STAGE / (release_name + "_before.json")
        before = preservation(additions, before_path)
        shutil.copyfile(before_path, PARENT / (release_name + "_prior887_before.json"))
        report["before_preservation_sha256"] = before["preservation_sha256"]
        before_succeeded = True
        import pytest
        plugin = TestResults()
        with (release / "pytest.log").open("w") as console, redirect_stdout(console), redirect_stderr(console):
            tests_started = time.monotonic()
            report["tests_exit_code"] = int(pytest.main([str(ROOT / TEST_PATH), "-q", "-p", "no:cacheprovider"], plugins=[plugin]))
            report["tests_elapsed_seconds"] = time.monotonic() - tests_started
        report["tests_collected"] = plugin.collected
        report["tests_passed"] = plugin.passed
        report["tests_failed"] = plugin.failed
        report["tests_skipped"] = plugin.skipped
        if (report["tests_exit_code"] != 0 or plugin.collected == 0
                or plugin.passed != plugin.collected or plugin.failed or plugin.skipped):
            raise ValueError("Every actually collected new carrier test must pass")
        from analysis.geometric_native_common_carrier_source_work_validation import native_control
        with (release / "control_console.log").open("w") as console, redirect_stdout(console), redirect_stderr(console):
            native_started = time.monotonic()
            native = native_control.run(release / "native_control")
            report["native_control_elapsed_seconds"] = time.monotonic() - native_started
        if native["status"] != "PASS":
            raise ValueError("New physical carrier control failed")
        report["native_control_succeeded"] = True
        report["saved_geometry_and_material_restored_and_validated"] = True
        report["native_control_report_sha256"] = native["report_sha256"]
        if source_map() != frozen:
            raise ValueError("Actual891 source map changed during the same-process gate")
        expected_payloads = {"pytest.log", "control_console.log",
                             "native_control/report.json", "native_control/source_packet.json"}
        if {str(path.relative_to(release)) for path in release.rglob("*") if path.is_file()} != expected_payloads:
            raise ValueError("Exactly four payloads must precede the sealed gate report")
        report["status"] = "PASS"
    except Exception as error:
        report["failure"] = type(error).__name__ + ": " + str(error)
        pending_error = error
    finally:
        if before_succeeded:
            try:
                after_path = STAGE / (release_name + "_after.json")
                after = preservation(additions, after_path)
                shutil.copyfile(after_path, PARENT / (release_name + "_prior887_after.json"))
                report["after_preservation_sha256"] = after["preservation_sha256"]
            except Exception as error:
                report["status"] = "FAIL"
                report["after_preservation_failure"] = type(error).__name__ + ": " + str(error)
                if pending_error is None:
                    pending_error = error
        report["elapsed_seconds"] = time.monotonic() - started
        report["output_raw_sha256"] = {str(path.relative_to(release)): digest(path)
            for path in sorted(release.rglob("*")) if path.is_file() and path != release / "report.json"}
        report["report_sha256"] = value_digest(report)
        (release / "report.json").write_bytes(encoded(report) + b"\n")
        print(json.dumps(report, sort_keys=True))
    if pending_error is not None:
        raise pending_error
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", default="release_validation_01")
    run(parser.parse_args().release)
