"""One fresh local source-work gate preserving all closed879 evidence."""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback
import types

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
PARENT = Path(__file__).resolve().parent
OLD_PARENT = ROOT / "analysis/geometric_native_common_material_surface_transport_validation"
OLD_MAP = "4a75d4bc7d404fc11d5d252302fe6b6d961797b4a637f921c38acb679485b79c"
OLD_MAP_RAW = "1b948eace979f3125756f6f0dea803fb55d0d2e87650890a2236c07502b0c336"
PHYSICAL_SCOPE = "LOCAL_SOURCE_FORCE_WORK_WITH_DECLARED_INTERNAL_RESIDUAL_COMPONENT_ONLY"
FRESH_TEST_COUNT = 14
ADDED_SOURCE_PATHS = frozenset((
    "tectonics/geometric_native_common_local_source_work.py",
    "tests/test_geometric_native_common_local_source_work.py",
    "analysis/geometric_native_common_local_source_work_validation/native_control.py",
    "analysis/geometric_native_common_local_source_work_validation/validate.py",
))
TEST_PATH = "tests/test_geometric_native_common_local_source_work.py"
CONTROL = "analysis.geometric_native_common_local_source_work_validation.native_control:run"
BOOTSTRAP_REPORT_SEAL = "d5c033e61f88792b2539410bfbf26a64eb28e9836fa6344dfcdc58627e75a3a7"
CURRENT_EVIDENCE = {
    "baseline_879_bootstrap_source.txt": "f23843c4065d8d8bd4095a9f0b1ecbf948386e92ea1047506f09104bbf7a66f1",
    "baseline_879_bootstrap.json": "cd9654ddb97418eb5fede3c445730dee88cfeee2eaeca09d56d2ec3a03e549a9",
    "baseline_879_bootstrap_console.log": "cd9654ddb97418eb5fede3c445730dee88cfeee2eaeca09d56d2ec3a03e549a9",
    "baseline_879_bootstrap_archive_association.json": "a4612fd93eb0aa5208f840f72d515efe8233831ba01ebfa984eac33727a5da41",
    "baseline_879_reusable_source.txt": "0ab740087c8a1ca3f8aee474e2d4b114d6ab8805eaf035419bb85248c42ecb25",
}
REMAINING_CONDITIONS = (
    "Whole-source physical material and old-force/prestress/work handover, HOT law/mean chart, loads/drag and NEW mixed equilibrium",
    "Whole-clock all-pair admission including reverse C5 to C1",
    "Accepted continuation with independent path work, balances and joint restart",
)


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()


def value_digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def sealed(path, key):
    value = read(path)
    if value.get(key) != value_digest({k: v for k, v in value.items() if k != key}):
        raise ValueError("Evidence selfseal changed: " + str(path))
    return value


def source_map():
    return {str(path.relative_to(ROOT)): digest(path)
        for directory in ("tectonics", "tests", "scripts", "analysis")
        for path in sorted((ROOT / directory).rglob("*.py"))}


def exact(actual, expected, label):
    if type(actual) is not type(expected) or actual != expected:
        raise ValueError(label)
    if type(actual) in (dict, list, tuple) and encoded(actual) != encoded(expected):
        raise ValueError(label + ": recursive literal kinds differ")


def verify_prior(*, output_path, expected_source_map):
    for name, pin in CURRENT_EVIDENCE.items():
        exact(digest(PARENT / name), pin, "Immutable879 evidence: " + name)
    bootstrap = sealed(PARENT / "baseline_879_bootstrap.json", "audit_sha256")
    exact(bootstrap["audit_sha256"], BOOTSTRAP_REPORT_SEAL, "Initial879 bootstrap seal")
    exact(bootstrap["status"], "PASS_FINAL879_EVIDENCE_READBACK_ONLY", "Initial879 status")
    before = source_map()
    exact(before, expected_source_map, "Full actual883 source map before preservation readback")
    module = types.ModuleType("_private_local_source_work_stdlib879_preservation")
    module.__file__ = str(PARENT / "baseline_879_reusable_source.txt")
    exec(compile(Path(module.__file__).read_bytes(), module.__file__, "exec"), module.__dict__)
    with redirect_stdout(io.StringIO()):
        baseline = module.run(ADDED_SOURCE_PATHS, output_path=output_path, source_stage="DECLARED883",
            bootstrap_source_path=PARENT / "baseline_879_bootstrap_source.txt",
            bootstrap_report_path=PARENT / "baseline_879_bootstrap.json")
    exact(read(output_path), baseline, "Returned and saved879 preservation receipt")
    sealed(output_path, "audit_sha256")
    for key, value in (
        ("status", "PASS_IMMUTABLE879_WITH_DECLARED_SOURCE_ADDITIONS_ONLY"),
        ("baseline_source_count", 879), ("current_python_source_count", 883),
        ("source_stage", "DECLARED883"), ("additional_python_source_count", 4),
        ("all879_python_sources_unchanged", True),
        ("source_epoch", "IMMUTABLE879_TO_EXACT_AUTHORIZED_LOCAL_SOURCE_WORK883"),
        ("declared_next_component_physical_scope", PHYSICAL_SCOPE),
        ("new_component_physics_validated", False),
        ("exact_authorized_four_path_whitelist_checked", True),
        ("complete_full_actual_source_map_independently_checked_before_after", True),
        ("complete_declared883_or_frozen879_source_map_before_after_identical", True),
        ("current_python_source_map_sha256", value_digest(before)),
        ("frozen879_source_map_canonical_sha256", OLD_MAP), ("frozen879_source_map_raw_sha256", OLD_MAP_RAW),
        ("frozen879_projection_disclosed", True),
        ("nested_reader_actual879_count_and_flags_are_projected_view_only", True),
        ("full_real_caller_project_modules_independently_identity_checked_before_after", True),
        ("preexisting_project_modules_preserved", True),
        ("real_sys_modules_project_entries_deleted_or_replaced", False),
        ("reader_helper_registered_in_real_sys_modules", False),
        ("preserved_closed879_manifest_file_count", 83),
        ("preserved_old1633_artifact_chain", True),
        ("authoritative_ancestry_release_closure_count", 10),
        ("preserved_879_recorded_raw_dependency_pin_count", 271),
        ("old_final879_readback_archive_closed_file_count", 6),
        ("recorded_raw_dependency_pin_count", 277),
        ("all277_preserved_dependency_and_archive_raw_sha256_hashed_before_after", True),
        ("all_eight_current_temp_and_permanent_raw_copy_associations_verified_before_after", True),
        ("project_imports", False), ("baseline1633_reexecuted", False),
        ("math_or_solver_or_test_or_quadrature_reexecuted", False),
        ("bootstrap_reader_raw_sha256", CURRENT_EVIDENCE["baseline_879_bootstrap_source.txt"]),
        ("bootstrap_report_raw_sha256", CURRENT_EVIDENCE["baseline_879_bootstrap.json"]),
        ("bootstrap_report_selfseal", BOOTSTRAP_REPORT_SEAL),
        ("auditor_source_sha256", CURRENT_EVIDENCE["baseline_879_reusable_source.txt"]),
    ):
        exact(baseline[key], value, "Immutable879 receipt: " + key)
    exact(baseline["declared_additional_source_paths"], sorted(ADDED_SOURCE_PATHS), "Four exact883 additions")
    exact(baseline["additional_python_sources"], {path: before[path] for path in sorted(ADDED_SOURCE_PATHS)}, "Four added883 raw pins")
    old_dependencies = bootstrap["dependency_raw_sha256"]
    dependencies = baseline["dependency_raw_sha256"]
    exact(len(dependencies), 277, "Complete271+six dependency count")
    exact({name: dependencies[name] for name in old_dependencies}, old_dependencies, "Complete old271 dependency pins")
    old_archive = baseline["old_final879_readback_archive_raw_sha256"]
    exact(len(old_archive), 6, "Exact old six-file archive")
    exact(set(dependencies) - set(old_dependencies), set(old_archive), "Exactly six additional old archive pins")
    exact({name: dependencies[name] for name in old_archive}, old_archive, "Exact archived879 pins")
    inherited_path = OLD_PARENT / "frozen_source_map_879.json"
    exact(digest(inherited_path), OLD_MAP_RAW, "Frozen879 source bytes")
    inherited = read(inherited_path)
    exact(len(inherited), 879, "Frozen879 source count")
    exact(value_digest(inherited), OLD_MAP, "Frozen879 canonical source map")
    exact(source_map(), before, "Full actual883 source map after preservation readback")
    for name, pin in CURRENT_EVIDENCE.items():
        exact(digest(PARENT / name), pin, "Immutable879 evidence after: " + name)
    return inherited, baseline


def _validate_source_map(actual, inherited):
    exact(len(actual), 883, "Complete actual883 source count")
    exact(set(actual) - set(inherited), set(ADDED_SOURCE_PATHS), "Exactly four local source-work additions")
    for name, pin in inherited.items():
        exact(actual.get(name), pin, "Every old879 source unchanged: " + name)

def expected_summary():
    return dict(physical_candidate_status=PHYSICAL_SCOPE,
        global_coordinate_count=100, owner_coordinate_count=6,
        cold_coordinate_count=56, hot_coordinate_count=30,
        explicit_authentic_source_internal_coordinate_count=8,
        local_source_atom_count=16, original_material_attachment_count=32,
        archived_original_donor_attachment_count=8,
        genuine_geometry_only_surface_binding_count=5,
        complete_local_source_query_count=4, complete_source_reference_query_count=4,
        complete_prestress_second_variation_defined=True,
        body_internal_mixed_second_variations_retained=True,
        force_hyperstress_and_internal_virtual_work_defined=True,
        joint_body_internal_atom_path_work_defined=True,
        literal_global_beams_and_registered_bulk_remainders_energized_once=True,
        old_source14_structural_body_zero_first_second_variations_certified=True,
        full_nonzero_common_body_probe_is_registered_and_fixed=True,
        local_force_work_component_changes_old875_potential=True,
        geometry_only_surface_measure_and_memory_bindings_retained=True,
        source_internal_links_reduced_into_common_body_modes=False,
        unconditional92_coordinate_reduction_impossibility_proved=False,
        old875_nonzero_source_force_identity_claimed=False,
        unknown_body_zero_bulk_hessian_supplied=False,
        physical_source_material_handover_certified=False,
        full_three_dimensional_rod_endpoints_recovered=False,
        whole_archived_three_dimensional_history_recovered=False,
        whole_face_measure_or_memory_integral_certified=False,
        current_deformed_follower_normal_contact_law_defined=False,
        old_mechanical_atom_areas_smeared_into_continuous_density=False,
        continuous_face_energy_added_to_old_source_potential=False,
        hot_receiving_contact_law_inferred=False,
        new_mixed_force_equilibrium_certified=False,
        whole_clock_all_pair_admission_authorized=False,
        physical_material_admission_authorized=False,
        source_history_advanced=False, fresh_tensor_quadrature_executed=False,
        historical_force_solve_reexecuted=False,
        accepted_original_native_intervals=1, accepted_new_native_intervals=0,
        original_point_1_remaining_work_blocks=1, original_point_1_completed=False,
        remaining_physical_conditions=list(REMAINING_CONDITIONS))


def _check_native_control(out):
    from analysis.geometric_native_common_local_source_work_validation import native_control as control
    native = sealed(out / "native_control/report.json", "report_sha256")
    packet = sealed(out / "native_control/source_packet.json", "packet_sha256")
    exact(native.get("status"), "PASS", "Native local force/work status")
    exact(native.get("format"), "native-common-local-source-work-control-1", "Native control format")
    exact(packet.get("format"), "native-common-local-source-work-packet-1", "Native packet format")
    exact(native.get("source_packet_sha256"), packet["packet_sha256"], "Full packet binding")
    for key in ("local_source_work_system", "evaluation_queries", "source_reference_queries"):
        exact(native.get(key + "_sha256"), value_digest(packet[key]), "Complete native query/system binding: " + key)
    result = control.replay_packet(packet)
    exact(result, expected_summary(), "Fixed local force/work scope and complete counts")
    expected = set(result) | {"format", "status", "source_packet_sha256", "local_source_work_system_sha256",
        "evaluation_queries_sha256", "source_reference_queries_sha256", "seconds", "report_sha256"}
    exact(set(native), expected, "Complete native report schema")
    exact({key: native[key] for key in result}, result, "Complete native scope")
    return result

def _purge_added_modules(paths):
    stems = {str(Path(name).with_suffix("")).replace("/", ".") for name in paths}
    removed = []
    for name, module in tuple(sys.modules.items()):
        location = getattr(module, "__file__", None)
        exact_file = isinstance(location, str) and any(Path(location).resolve() == ROOT / path for path in paths)
        if name != "__main__" and (name in stems or exact_file):
            parent_name, _, child_name = name.rpartition(".")
            parent = sys.modules.get(parent_name)
            if parent is not None and getattr(parent, child_name, None) is module:
                delattr(parent, child_name)
            sys.modules.pop(name, None)
            removed.append(name)
    return sorted(removed)


def run(out, test_paths, control):
    if type(test_paths) not in (tuple, list) or tuple(test_paths) != (TEST_PATH,) or control != CONTROL:
        raise ValueError("Exactly the new local source-work test module and control are authorized")
    started = time.perf_counter()
    before = source_map()
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    prefix = "native_common_local_source_work_" + out.name + "_20261006"
    before_path = Path("/tmp") / (prefix + "_prior_before.json")
    after_path = Path("/tmp") / (prefix + "_prior_after.json")
    report = dict(format="native-common-local-source-work-same-process-validation-1",
        status="FAIL", physical_candidate_status="NOT_CHECKED", started_utc=datetime.now(timezone.utc).isoformat(),
        source_count=len(before), source_map_before_sha256=value_digest(before), prior_source_map_sha256=OLD_MAP,
        execution_mode="same_python_process", pytest_import_mode="importlib", historical_tests_reexecuted=False,
        baseline1633_reexecuted=False,
        old1633_preservation_readback="SAVED_RECEIPT_PLUS_271_RAW_DEPENDENCIES_SIX_OLD_ARCHIVE_FILES_AND_TEN_STRICT_CLOSURES",
        test_paths=list(test_paths), control=control, expected_fresh_test_count=FRESH_TEST_COUNT,
        verified_current_baseline_evidence=CURRENT_EVIDENCE)
    try:
        inherited, baseline = verify_prior(output_path=before_path, expected_source_map=before)
        report.update(verified_prior_python_sources=879, verified_prior_artifacts=1633,
            prior_baseline_before=baseline, prior_baseline_before_path=str(before_path),
            prior_baseline_before_raw_sha256=digest(before_path), prior_baseline_before_selfseal=baseline["audit_sha256"],
            added_python_sources={name: before[name] for name in sorted(ADDED_SOURCE_PATHS)})
        _validate_source_map(before, inherited)
        frozen = PARENT / "frozen_source_map_883.json"
        exact(read(frozen), before, "Full actual883 source freeze")
        report["frozen_source_map_raw_sha256"] = digest(frozen)
        removed = []
        for name in sorted(ADDED_SOURCE_PATHS):
            cache = Path(importlib.util.cache_from_source(str(ROOT / name)))
            if cache.is_file():
                cache.unlink()
                removed.append(str(cache.relative_to(ROOT)))
        report["removed_added_source_bytecode_caches"] = removed
        own_path = str(Path(__file__).resolve().relative_to(ROOT))
        report["purged_previously_imported_added_source_modules"] = _purge_added_modules(ADDED_SOURCE_PATHS - {own_path})
        importlib.invalidate_caches()
        exact(source_map(), before, "All source bytes before fresh imports")
        import pytest
        start = time.perf_counter()
        old_dir = Path.cwd()
        try:
            os.chdir(ROOT)
            with (out / "pytest.log").open("x", buffering=1) as stream, redirect_stdout(stream), redirect_stderr(stream):
                code = int(pytest.main(["-q", "-p", "no:cacheprovider", "--import-mode=importlib", TEST_PATH]))
        finally:
            os.chdir(old_dir)
        lines = (out / "pytest.log").read_text().splitlines()
        summary_line = lines[-1] if lines else ""
        counts = {name: int(number) for number, name in re.findall(
            r"\b(\d+)\s+(passed|failed|skipped|xfailed|xpassed|errors?|deselected)\b", summary_line)}
        report.update(pytest_exit_code=code, pytest_counts=counts, pytest_summary=summary_line,
            pytest_seconds=time.perf_counter() - start)
        exact(code, 0, "Fresh local source-work pytest exit")
        exact(counts, {"passed": FRESH_TEST_COUNT}, "Exact fresh local source-work test count")
        exact(source_map(), before, "All sources during fresh tests")
        module_name, function = control.split(":", 1)
        callback = getattr(importlib.import_module(module_name), function)
        start = time.perf_counter()
        with (out / "control_console.log").open("x", buffering=1) as stream, redirect_stdout(stream), redirect_stderr(stream):
            callback(out / "native_control")
        report["native_control_seconds"] = time.perf_counter() - start
        report.update(_check_native_control(out))
        expected = {"pytest.log", "control_console.log", "native_control/report.json", "native_control/source_packet.json"}
        exact({str(path.relative_to(out)) for path in out.rglob("*") if path.is_file()}, expected, "Exactly four closed gate outputs")
        report["status"] = "PASS"
    except BaseException as error:
        report.update(failure_type=type(error).__name__, failure_message=str(error), failure_traceback=traceback.format_exc())
    finally:
        after = source_map()
        report["source_map_after_sha256"] = value_digest(after)
        report["changed_python_paths"] = sorted(name for name in set(before) | set(after) if before.get(name) != after.get(name))
        try:
            inherited_after, baseline_after = verify_prior(output_path=after_path, expected_source_map=before)
            _validate_source_map(after, inherited_after)
            report.update(prior_artifacts_and_python_sources_preserved_after=True,
                prior_baseline_after=baseline_after, prior_baseline_after_path=str(after_path),
                prior_baseline_after_raw_sha256=digest(after_path), prior_baseline_after_selfseal=baseline_after["audit_sha256"])
        except BaseException as error:
            report.update(prior_artifacts_and_python_sources_preserved_after=False, prior_after_error=str(error))
        if report["changed_python_paths"] or not report["prior_artifacts_and_python_sources_preserved_after"]:
            report["status"] = "FAIL"
        frozen = PARENT / "frozen_source_map_883.json"
        try:
            exact(read(frozen), before, "Full actual883 frozen map after gate")
            exact(digest(frozen), report.get("frozen_source_map_raw_sha256"), "Frozen883 raw bytes after gate")
        except BaseException as error:
            report.update(status="FAIL", frozen_map_after_error=str(error))
        report["seconds"] = time.perf_counter() - started
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        report["outputs"] = {str(path.relative_to(out)): dict(sha256=digest(path), bytes=path.stat().st_size)
            for path in sorted(out.rglob("*")) if path.is_file()}
        report["report_sha256"] = value_digest(report)
        with (out / "report.json").open("xb") as stream:
            stream.write(encoded(report) + b"\n")
        print(encoded({key: report.get(key) for key in ("status", "physical_candidate_status", "pytest_counts",
            "seconds", "source_count", "changed_python_paths")}).decode(), flush=True)
    if report["status"] != "PASS":
        raise RuntimeError("Validation refused; preserved outputs in " + str(out / "report.json"))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--control", required=True)
    parser.add_argument("tests", nargs="+")
    args = parser.parse_args()
    run(args.out, args.tests, args.control)
