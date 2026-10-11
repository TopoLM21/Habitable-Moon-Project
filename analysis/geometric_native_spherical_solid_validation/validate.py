"""Fresh spherical-solid checks without changing the accepted source epoch.

The expensive genuine case may be built before these additive sources are
frozen. Its same-process caller then supplies the original 792-source map
captured before that build. Only the new modules are purged before final tests.
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
from analysis.geometric_native_finite_equilibrium_validation import validate as previous

PRIOR = ROOT / "analysis/geometric_native_finite_equilibrium_validation"
RELEASE = PRIOR / "release_validation_02"
FAILED_RELEASE = PRIOR / "release_validation_01"
EXPECTED_REPORT = "af6f34a7aeb5add401fcfd3effdd7842bb63407b2efc6ad759d119aa11d2ee56"
EXPECTED_REPORT_SEAL = "a52f1d3426a2b48671cdf2140f5d7e4c7d5a9ab7b42e4f545938353afd893ed3"
EXPECTED_NATIVE_REPORT = "57ca666bd7e4c8e0dda0ee1930cf4059a3c2943b063338db77111c9458ff8122"
EXPECTED_PACKET = "d5ed6f50fe09561db79d2b8398ad682d676aae60fd789289fd42a4fbe9830a25"
EXPECTED_SOURCE_MAP = "1c831bea0d5fe8f30d00c46c0c92d40fa668c579155a2827bcaceb67cceac154"
EXPECTED_FAILED_REPORT = "40f15f8afb09a33d26ac10253f24ff23cf354fccb2d704d7aa96fbf6841b8bfd"

encoded = previous.encoded
digest = previous.digest
value_digest = previous.value_digest
read = previous.read
sealed = previous.sealed
source_map = previous.source_map
parse_counts = previous.previous.parse_counts


def _verify_outputs(directory, report, count):
    if len(report.get("outputs", {})) != count:
        raise ValueError("Preserved release output count differs")
    for name, row in report["outputs"].items():
        path = directory / name
        if path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
            raise ValueError("A preserved release output changed: " + str(path))


def verify_prior():
    """Verify the 792 sources and every preserved release output, including FAIL."""
    inherited = previous.verify_prior()
    report_path = RELEASE / "report.json"
    if digest(report_path) != EXPECTED_REPORT:
        raise ValueError("Authoritative finite-equilibrium report changed")
    if read(report_path).get("report_sha256") != EXPECTED_REPORT_SEAL:
        raise ValueError("Authoritative finite-equilibrium report seal changed")
    report = sealed(report_path, "report_sha256")
    if (report.get("status") != "PASS" or report.get("source_count") != 792
            or report.get("pytest_counts") != {"passed": 49}
            or report.get("changed_python_paths") != []
            or report.get("prior_artifacts_and_python_sources_preserved_after") is not True
            or report.get("source_map_before_sha256") != EXPECTED_SOURCE_MAP
            or report.get("source_map_after_sha256") != EXPECTED_SOURCE_MAP
            or len(report.get("added_python_sources", {})) != 6
            or set(inherited) & set(report["added_python_sources"])):
        raise ValueError("Finite-equilibrium source generation or release scope differs")
    inherited.update(report["added_python_sources"])
    if len(inherited) != 792 or value_digest(inherited) != EXPECTED_SOURCE_MAP:
        raise ValueError("The original 792-source generation differs")
    for name, expected in inherited.items():
        if digest(ROOT / name) != expected:
            raise ValueError("An original Python source changed: " + name)
    _verify_outputs(RELEASE, report, 4)
    native_path = RELEASE / "native_control/report.json"
    packet_path = RELEASE / "native_control/candidate_packet.json"
    if digest(native_path) != EXPECTED_NATIVE_REPORT or digest(packet_path) != EXPECTED_PACKET:
        raise ValueError("The authoritative finite-force native evidence changed")
    native = sealed(native_path, "report_sha256")
    packet = sealed(packet_path, "packet_sha256")
    if (native.get("status") != "PASS"
            or native.get("accepted_original_native_intervals") != 1
            or native.get("accepted_new_native_intervals") != 0
            or native.get("whole_clock_material_admission_authorized") is not False
            or native.get("original_point_1_completed") is not False
            or native.get("finite_born_with_original_affine_force_equilibrium_certified") is not True
            or native.get("paid_activation_preserved_once") is not True
            or native.get("contact_foundation_and_original_clocks_preserved") is not True
            or native.get("packet_sha256") != read(packet_path)["packet_sha256"]
            or packet.get("material_admission_authorized") is not False):
        raise ValueError("The preserved finite-force physical scope or source binding differs")
    failed_path = FAILED_RELEASE / "report.json"
    if digest(failed_path) != EXPECTED_FAILED_REPORT:
        raise ValueError("The historical failed release report changed")
    failed = sealed(failed_path, "report_sha256")
    if (failed.get("status") != "FAIL"
            or failed.get("failure_type") != "TypeError"
            or failed.get("pytest_counts") != {"passed": 49}
            or failed.get("changed_python_paths") != []):
        raise ValueError("Historical failed release evidence scope differs")
    _verify_outputs(FAILED_RELEASE, failed, 3)
    return inherited


def verify_preloaded_case_source_guard(guard, case):
    """Verify the caller's before-build source map, without replaying its case.

    A map is a caller attestation rather than proof that an object was built;
    the source-bound controller must independently authenticate the case.
    """
    if not isinstance(case, dict) or not case or not isinstance(guard, dict):
        raise ValueError("A genuine case and its before-build original source map are required")
    inherited = verify_prior()
    current = source_map()
    if (guard != inherited or value_digest(guard) != EXPECTED_SOURCE_MAP
            or any(current.get(name) != expected for name, expected in inherited.items())):
        raise ValueError("The caller-captured original 792-source generation differs")
    return dict(guard_sha256=value_digest(guard), original_source_count=792,
        generation_source_map_sha256=EXPECTED_SOURCE_MAP,
        original_source_map_sha256=EXPECTED_SOURCE_MAP,
        same_python_callback_process_confirmed=True,
        genuine_case_built_before_final_new_source_freeze=True,
        case_source_binding_required_of_controller=True,
        provenance="caller-attested before-build original-792 map; verified again after build")


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
    added = set(before) - set(inherited)
    removed_caches = []
    for name in sorted(added):
        cache = Path(importlib.util.cache_from_source(str(ROOT / name)))
        if cache.is_file():
            cache.unlink()
            removed_caches.append(str(cache.relative_to(ROOT)))
    own_path = str(Path(__file__).resolve().relative_to(ROOT))
    purged_modules = previous._purge_added_source_modules(added - {own_path})
    importlib.invalidate_caches()
    if source_map() != before:
        raise ValueError("A Python source changed while preparing final imports")
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = dict(format="native-spherical-solid-same-process-validation-1",
        status="FAIL", started_utc=datetime.now(timezone.utc).isoformat(),
        verified_prior_artifacts=1633, verified_solid_placement_outputs=5,
        verified_radial_normalization_diagnosis=True, verified_prior_python_sources=792,
        verified_finite_equilibrium_outputs=4, verified_failed_release_outputs=3,
        prior_finite_report_raw_sha256=EXPECTED_REPORT,
        prior_finite_native_report_raw_sha256=EXPECTED_NATIVE_REPORT,
        prior_finite_packet_raw_sha256=EXPECTED_PACKET,
        preserved_failed_report_raw_sha256=EXPECTED_FAILED_REPORT,
        prior_source_map_sha256=EXPECTED_SOURCE_MAP,
        source_count=len(before), source_map_before_sha256=value_digest(before),
        added_python_sources={name: before[name] for name in sorted(added)},
        removed_added_source_bytecode_caches=removed_caches,
        purged_previously_imported_added_source_modules=purged_modules,
        historical_49_tests_reexecuted=False, historical_70_tests_reexecuted=False,
        historical_135_tests_reexecuted=False, test_paths=list(paths), control=control,
        execution_mode="same_python_process", pytest_import_mode="importlib",
        preloaded_genuine_case_used=case is not None,
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
        summary, counts = parse_counts(out / "pytest.log")
        report.update(pytest_exit_code=exit_code, pytest_summary=summary,
            pytest_counts=counts, pytest_seconds=time.perf_counter()-pytest_started)
        if (exit_code != 0 or counts.get("passed", 0) < 1
                or any(counts.get(name, 0) for name in
                    ("failed", "skipped", "xfailed", "xpassed", "error", "errors"))):
            raise RuntimeError("Fresh spherical-solid pytest gate failed")
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
        if (report["changed_python_paths"]
                or not report["prior_artifacts_and_python_sources_preserved_after"]):
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
    run(options.out, options.test_paths, options.control)
