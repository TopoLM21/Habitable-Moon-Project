"""Fresh finite-equilibrium checks with preserved evidence and source epochs.

A genuine case may be built before the new sources are frozen.  Its caller must
then supply a same-process source-generation guard captured before that build;
passing a saved result as a fresh case is not a supported route.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import importlib
import importlib.util
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis.geometric_native_solid_placement_validation import validate as previous

SOLID = ROOT / "analysis/geometric_native_solid_placement_validation"
SOLID_RELEASE = SOLID / "release_validation_01"
EXPECTED_SOLID_REPORT = "7970483acb78f86757267ea35f29b205c63fdf8b1171f872d5d1d4f5a5384dd9"
EXPECTED_SOLID_REPORT_SEAL = "52120317399c25e1856cf3e743fff5bd3ddc7a43c07da3bd6a6c692c89a09d5c"
EXPECTED_SOLID_NATIVE_REPORT = "b4503f1ab02a4090d334c0d10307ef4459716e9baabf25f081604e6943c9bdd3"
EXPECTED_INHERITED_SOURCE_MAP = "4569e620aea3d6daf6fdca957e2499f87413f561ce11e16fb4bad0436a09db9b"
EXPECTED_DIAGNOSIS = "977d0b4697915596e4c585f79ac952070d38e686d4c47e450704b66c88eeac8b"

encoded = previous.encoded
digest = previous.digest
value_digest = previous.value_digest
read = previous.read
sealed = previous.sealed
source_map = previous.source_map


def verify_prior():
    """Verify the 786 original sources and all preserved release evidence."""
    inherited = previous.verify_prior()
    path = SOLID_RELEASE / "report.json"
    if digest(path) != EXPECTED_SOLID_REPORT:
        raise ValueError("Authoritative solid-placement report changed")
    full = read(path)
    if full.get("report_sha256") != EXPECTED_SOLID_REPORT_SEAL:
        raise ValueError("Authoritative solid-placement report seal changed")
    report = sealed(path, "report_sha256")
    if (report.get("status") != "PASS" or report.get("source_count") != 786
            or report.get("changed_python_paths") != []
            or report.get("prior_artifacts_and_python_sources_preserved_after") is not True
            or report.get("pytest_counts") != {"passed": 70}
            or report.get("source_map_before_sha256") != EXPECTED_INHERITED_SOURCE_MAP
            or report.get("source_map_after_sha256") != EXPECTED_INHERITED_SOURCE_MAP
            or len(report["added_python_sources"]) != 8
            or len(report["outputs"]) != 5):
        raise ValueError("Solid-placement release count or scope differs")
    if set(inherited) & set(report["added_python_sources"]):
        raise ValueError("Inherited source generations overlap")
    inherited.update(report["added_python_sources"])
    if len(inherited) != 786 or value_digest(inherited) != EXPECTED_INHERITED_SOURCE_MAP:
        raise ValueError("The original 786-source generation differs")
    for name, expected in inherited.items():
        if digest(ROOT / name) != expected:
            raise ValueError("An original Python source changed: " + name)
    for name, row in report["outputs"].items():
        output = SOLID_RELEASE / name
        if output.stat().st_size != row["bytes"] or digest(output) != row["sha256"]:
            raise ValueError("A solid-placement output changed: " + name)
    native_path = SOLID_RELEASE / "native_control/report.json"
    if digest(native_path) != EXPECTED_SOLID_NATIVE_REPORT:
        raise ValueError("Authoritative solid-placement native report changed")
    native = sealed(native_path, "report_sha256")
    if (native.get("status") != "PASS"
            or native.get("accepted_original_native_intervals") != 1
            or native.get("accepted_new_native_intervals") != 0
            or native.get("whole_clock_material_admission_authorized") is not False
            or native.get("original_point_1_completed") is not False):
        raise ValueError("Inherited native physical admission scope differs")
    diagnosis_path = SOLID / "radial_normalization_diagnosis.json"
    if digest(diagnosis_path) != EXPECTED_DIAGNOSIS:
        raise ValueError("Authoritative radial-normalization diagnosis changed")
    diagnosis = sealed(diagnosis_path, "diagnosis_sha256")
    if (diagnosis.get("corrected_force_solved") is not False
            or diagnosis.get("accepted_new_native_intervals") != 0
            or diagnosis.get("source_candidate_raw_sha256")
                != report["outputs"]["native_control/candidate_packet.json"]["sha256"]
            or diagnosis.get("source_candidate_sha256") != native["candidate_sha256"]):
        raise ValueError("The preserved radial diagnosis lost its source binding")
    return inherited


def capture_preloaded_case_source_guard():
    """Call before constructing a genuine case in this Python interpreter.

    The recorded map may contain new files under development.  All original 786
    files must stay immutable; final new test/controller modules are imported
    after the final source freeze in :func:`run`.
    """
    inherited = verify_prior()
    before = source_map()
    if any(before.get(name) != expected for name, expected in inherited.items()):
        raise ValueError("Cannot capture a changed original source generation")
    guard = dict(format="preloaded-native-case-source-generation-1",
        process_id=os.getpid(), interpreter_identity=id(sys.modules),
        captured_utc=datetime.now(timezone.utc).isoformat(),
        source_map=before, source_map_sha256=value_digest(before),
        original_source_map_sha256=EXPECTED_INHERITED_SOURCE_MAP,
        original_source_count=786)
    guard["guard_sha256"] = value_digest(guard)
    return guard


def verify_preloaded_case_source_guard(guard, case):
    """Check a caller's before-build source guard without replaying the case."""
    if not isinstance(guard, dict) or not isinstance(case, dict) or not case:
        raise ValueError("A genuine case and its captured source guard are required")
    # The root may have started the expensive genuine build before importing
    # this new module.  Accept its captured *original* generation as an explicit
    # caller attestation; neither an object nor a hash proves the build by itself.
    if len(guard) == 786 and all(isinstance(value, str) for value in guard.values()):
        inherited = verify_prior()
        current = source_map()
        if (guard != inherited or value_digest(guard) != EXPECTED_INHERITED_SOURCE_MAP
                or any(current.get(name) != expected for name, expected in inherited.items())):
            raise ValueError("The caller-captured original source generation differs")
        return dict(guard_sha256=value_digest(guard), captured_utc=None,
            generation_source_map_sha256=value_digest(guard),
            original_source_map_sha256=EXPECTED_INHERITED_SOURCE_MAP,
            original_source_count=786, same_python_callback_process_confirmed=True,
            genuine_case_built_before_final_new_source_freeze=True,
            case_source_binding_required_of_controller=True,
            provenance="caller-attested before-build original-786 source map; verified again after build")
    row = dict(guard)
    expected = row.pop("guard_sha256", None)
    if (expected != value_digest(row)
            or row.get("format") != "preloaded-native-case-source-generation-1"
            or row.get("process_id") != os.getpid()
            or row.get("interpreter_identity") != id(sys.modules)
            or row.get("original_source_count") != 786
            or row.get("original_source_map_sha256") != EXPECTED_INHERITED_SOURCE_MAP
            or not isinstance(row.get("source_map"), dict)
            or row.get("source_map_sha256") != value_digest(row["source_map"])):
        raise ValueError("The preloaded case source-generation guard is invalid")
    inherited = verify_prior()
    current = source_map()
    if any(row["source_map"].get(name) != expected
            or current.get(name) != expected for name, expected in inherited.items()):
        raise ValueError("The preloaded case original source generation changed")
    return dict(guard_sha256=expected, captured_utc=row["captured_utc"],
        generation_source_map_sha256=row["source_map_sha256"],
        original_source_map_sha256=EXPECTED_INHERITED_SOURCE_MAP,
        original_source_count=786, same_python_callback_process_confirmed=True,
        genuine_case_built_before_final_new_source_freeze=True,
        case_source_binding_required_of_controller=True,
        provenance="caller-captured before-build source-generation attestation")


def _check_native_control(out):
    path = out / "native_control/report.json"
    native = sealed(path, "report_sha256")
    if (native.get("status") != "PASS"
            or native.get("accepted_original_native_intervals") != 1
            or native.get("accepted_new_native_intervals") != 0
            or native.get("whole_clock_material_admission_authorized") is not False
            or native.get("original_point_1_completed") is not False):
        raise ValueError("The native controller changed this gate's physical scope")
    return dict(native_control_report_raw_sha256=digest(path),
        accepted_original_native_intervals=1, accepted_new_native_intervals=0,
        whole_clock_material_admission_authorized=False,
        original_point_1_completed=False)


def _purge_added_source_modules(added_source_paths):
    """Import final new code afresh while preserving the genuine old case types."""
    added = {str((ROOT / name).resolve()) for name in added_source_paths}
    purged = []
    for name, module in tuple(sys.modules.items()):
        if name in (__name__, "__main__"):
            continue
        filename = getattr(module, "__file__", None)
        if filename is None:
            continue
        try:
            source = str(Path(filename).resolve())
        except (TypeError, ValueError, OSError):
            continue
        if source not in added:
            continue
        del sys.modules[name]
        parent_name, _, child_name = name.rpartition(".")
        parent = sys.modules.get(parent_name)
        if parent is not None and getattr(parent, child_name, None) is module:
            delattr(parent, child_name)
        purged.append(name)
    return sorted(purged)


def run(out, test_paths, control, *, case=None, preloaded_case_source_guard=None):
    paths = tuple(test_paths)
    if (not paths or len(set(paths)) != len(paths)
            or any(not name.startswith("tests/test_") or not name.endswith(".py")
                or not (ROOT / name).is_file() for name in paths)
            or not isinstance(control, str) or ":" not in control):
        raise ValueError("Existing distinct new tests and module:function are required")
    inherited = verify_prior()
    if any(name in inherited for name in paths):
        raise ValueError("This gate accepts only newly added test modules")
    if (case is None) != (preloaded_case_source_guard is None):
        raise ValueError("A preloaded case requires its before-build source guard")
    case_guard = (None if case is None else
        verify_preloaded_case_source_guard(preloaded_case_source_guard, case))
    before = source_map()
    removed_caches = []
    for name in sorted(set(before)-set(inherited)):
        cache = Path(importlib.util.cache_from_source(str(ROOT / name)))
        if cache.is_file():
            cache.unlink()
            removed_caches.append(str(cache.relative_to(ROOT)))
    purged_modules = _purge_added_source_modules(set(before)-set(inherited))
    importlib.invalidate_caches()
    if source_map() != before:
        raise ValueError("A Python source changed while preparing final imports")
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = dict(format="native-finite-equilibrium-same-process-validation-1",
        status="FAIL", started_utc=datetime.now(timezone.utc).isoformat(),
        verified_prior_artifacts=1633, verified_solid_placement_outputs=5,
        verified_radial_normalization_diagnosis=True,
        verified_prior_python_sources=786,
        prior_summary_raw_sha256=previous.EXPECTED_SUMMARY,
        prior_manifest_raw_sha256=previous.EXPECTED_MANIFEST,
        prior_solid_report_raw_sha256=EXPECTED_SOLID_REPORT,
        prior_solid_native_report_raw_sha256=EXPECTED_SOLID_NATIVE_REPORT,
        prior_radial_diagnosis_raw_sha256=EXPECTED_DIAGNOSIS,
        prior_source_map_sha256=EXPECTED_INHERITED_SOURCE_MAP,
        source_count=len(before), source_map_before_sha256=value_digest(before),
        added_python_sources={name: before[name] for name in sorted(set(before)-set(inherited))},
        removed_added_source_bytecode_caches=removed_caches,
        purged_previously_imported_added_source_modules=purged_modules,
        historical_70_tests_reexecuted=False, historical_135_tests_reexecuted=False,
        test_paths=list(paths), control=control, execution_mode="same_python_process",
        pytest_import_mode="importlib", preloaded_genuine_case_used=case is not None,
        preloaded_case_source_generation_guard=case_guard,
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
        summary, counts = previous.parse_counts(out / "pytest.log")
        report.update(pytest_exit_code=exit_code, pytest_summary=summary,
            pytest_counts=counts, pytest_seconds=time.perf_counter()-pytest_started)
        if (exit_code != 0 or counts.get("passed", 0) < 1
                or any(counts.get(name, 0) for name in
                    ("failed", "skipped", "xfailed", "xpassed", "error", "errors"))):
            raise RuntimeError("Fresh finite-equilibrium pytest gate failed")
        if source_map() != before:
            raise ValueError("A Python source changed during pytest")
        module, function = control.split(":", 1)
        callback = getattr(importlib.import_module(module), function)
        control_started = time.perf_counter()
        with (out / "control_console.log").open("x", buffering=1) as stream:
            with redirect_stdout(stream), redirect_stderr(stream):
                callback(out / "native_control", case=case)
        report["native_control_seconds"] = time.perf_counter()-control_started
        report.update(_check_native_control(out))
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
            if case is not None:
                verify_preloaded_case_source_guard(preloaded_case_source_guard, case)
                report["preloaded_case_source_generation_preserved_after"] = True
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
