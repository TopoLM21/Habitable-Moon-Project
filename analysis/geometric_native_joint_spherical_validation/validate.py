"""Fresh joint spherical checks retaining the original 800-source generation.

A genuine native case can be built in the same process before the additive
sources are frozen. Only added modules are purged; original case classes and
the 800 authenticated source bodies remain intact.
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
from analysis.geometric_native_spherical_solid_validation import validate as previous

PRIOR = ROOT / "analysis/geometric_native_spherical_solid_validation"
RELEASE = PRIOR / "release_validation_01"
EXPECTED_REPORT = "4a49e7fa73ca1f795c15e3244b083c528c1128b00ad7f6b4d92cd640dd27ac7b"
EXPECTED_REPORT_SEAL = "3b3ac04fdac5061816015935495d0fa4b711c005a4cecf0de12853af28940078"
EXPECTED_NATIVE_REPORT = "d74bd19a31312ee5dab2265be2d5ae3893c32f8fed9f08741af5aafaa56889ab"
EXPECTED_NATIVE_REPORT_SEAL = "8506c35632c0ef2a5215edf0d9e72d80702227e186dff3788e265f05018ac049"
EXPECTED_PACKET = "1518f46d53a18d78d9afc9f93b0297f8cf446026b424b0e5cd54a134bae038e2"
EXPECTED_PACKET_SEAL = "4336d85c26506622ed89e97675a5cc0d80a89cc135ce79b2e45c2dc3b71ccb7e"
EXPECTED_SOURCE_MAP = "2efdca18421932b30abb04728228e095b4e11d17fed1b1b380d750e4374da7a8"

encoded = previous.encoded
digest = previous.digest
value_digest = previous.value_digest
read = previous.read
sealed = previous.sealed
source_map = previous.source_map
parse_counts = previous.parse_counts


def verify_prior():
    """Verify all preserved releases and the original 800 Python sources."""
    inherited = previous.verify_prior()
    report_path = RELEASE / "report.json"
    if digest(report_path) != EXPECTED_REPORT:
        raise ValueError("Authoritative spherical-solid report changed")
    report = sealed(report_path, "report_sha256")
    if (read(report_path).get("report_sha256") != EXPECTED_REPORT_SEAL
            or report.get("status") != "PASS"
            or report.get("source_count") != 800
            or report.get("pytest_counts") != {"passed": 83}
            or report.get("changed_python_paths") != []
            or report.get("prior_artifacts_and_python_sources_preserved_after") is not True
            or report.get("source_map_before_sha256") != EXPECTED_SOURCE_MAP
            or report.get("source_map_after_sha256") != EXPECTED_SOURCE_MAP
            or len(report.get("added_python_sources", {})) != 8
            or set(inherited) & set(report["added_python_sources"])):
        raise ValueError("Spherical-solid source generation or release scope differs")
    inherited.update(report["added_python_sources"])
    if len(inherited) != 800 or value_digest(inherited) != EXPECTED_SOURCE_MAP:
        raise ValueError("The original 800-source generation differs")
    for name, expected in inherited.items():
        if digest(ROOT / name) != expected:
            raise ValueError("An original Python source changed: " + name)
    previous._verify_outputs(RELEASE, report, 4)
    native_path = RELEASE / "native_control/report.json"
    packet_path = RELEASE / "native_control/candidate_packet.json"
    if digest(native_path) != EXPECTED_NATIVE_REPORT or digest(packet_path) != EXPECTED_PACKET:
        raise ValueError("The authoritative spherical-solid native evidence changed")
    native = sealed(native_path, "report_sha256")
    packet = sealed(packet_path, "packet_sha256")
    if (read(native_path).get("report_sha256") != EXPECTED_NATIVE_REPORT_SEAL
            or read(packet_path).get("packet_sha256") != EXPECTED_PACKET_SEAL
            or native.get("status") != "PASS"
            or native.get("accepted_original_native_intervals") != 1
            or native.get("accepted_new_native_intervals") != 0
            or native.get("whole_clock_material_admission_authorized") is not False
            or native.get("original_point_1_completed") is not False
            or native.get("accepted_input_complete_typed_state_unchanged") is not True
            or native.get("finite_born_reference_area_with_original_affine_force_equilibrium_certified") is not True
            or native.get("distributed_curved_physical_face_force_equilibrium_certified") is not False
            or native.get("normal_column_and_contact_deformation_glued") is not False
            or native.get("whole_spherical_partition_embedding_certified") is not False
            or native.get("packet_sha256") != read(packet_path)["packet_sha256"]
            or packet.get("accepted_input_unchanged") is not True
            or packet.get("material_admission_authorized") is not False):
        raise ValueError("Preserved spherical physical scope or source binding differs")
    return inherited


def verify_preloaded_case_source_guard(guard, case):
    """Check the caller-attested before-build map; controller authenticates case."""
    if not isinstance(case, dict) or not case or not isinstance(guard, dict):
        raise ValueError("A genuine case and its before-build original source map are required")
    inherited = verify_prior()
    current = source_map()
    if (guard != inherited or value_digest(guard) != EXPECTED_SOURCE_MAP
            or any(current.get(name) != expected for name, expected in inherited.items())):
        raise ValueError("The caller-captured original 800-source generation differs")
    return dict(guard_sha256=value_digest(guard), original_source_count=800,
        generation_source_map_sha256=EXPECTED_SOURCE_MAP,
        original_source_map_sha256=EXPECTED_SOURCE_MAP,
        same_python_callback_process_confirmed=True,
        genuine_case_built_before_final_new_source_freeze=True,
        case_source_binding_required_of_controller=True,
        provenance="caller-attested before-build original-800 map; verified again after build")


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
    purged_modules = previous.previous._purge_added_source_modules(added - {own_path})
    importlib.invalidate_caches()
    if source_map() != before:
        raise ValueError("A Python source changed while preparing final imports")
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = dict(format="native-joint-spherical-same-process-validation-1",
        status="FAIL", started_utc=datetime.now(timezone.utc).isoformat(),
        verified_prior_artifacts=1633, verified_solid_placement_outputs=5,
        verified_radial_normalization_diagnosis=True, verified_prior_python_sources=800,
        verified_finite_equilibrium_outputs=4, verified_failed_release_outputs=3,
        verified_spherical_solid_outputs=4,
        prior_spherical_report_raw_sha256=EXPECTED_REPORT,
        prior_spherical_native_report_raw_sha256=EXPECTED_NATIVE_REPORT,
        prior_spherical_packet_raw_sha256=EXPECTED_PACKET,
        prior_source_map_sha256=EXPECTED_SOURCE_MAP,
        source_count=len(before), source_map_before_sha256=value_digest(before),
        added_python_sources={name: before[name] for name in sorted(added)},
        removed_added_source_bytecode_caches=removed_caches,
        purged_previously_imported_added_source_modules=purged_modules,
        historical_49_tests_reexecuted=False, historical_70_tests_reexecuted=False,
        historical_83_tests_reexecuted=False, historical_135_tests_reexecuted=False,
        test_paths=list(paths), control=control,
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
            raise RuntimeError("Fresh joint spherical pytest gate failed")
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
