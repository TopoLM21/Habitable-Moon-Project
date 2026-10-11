"""Fresh finite material-surface gate; immutable875 evidence, one process."""
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
OLD_PARENT = ROOT / "analysis/geometric_native_common_strain_projection_validation"
OLD_MAP = "13c19fe10056af9ab621e2c8a23e63ad36b301c95e121a6835cead019378e385"
PHYSICAL_SCOPE = "FINITE_LOCAL_MATERIAL_SURFACE_MEASURE_AND_MEMORY_TRANSPORT_COMPONENT_ONLY"
FRESH_TEST_COUNT = 14
ADDED_SOURCE_PATHS = frozenset((
    "tectonics/geometric_native_common_material_surface_transport.py",
    "tests/test_geometric_native_common_material_surface_transport.py",
    "analysis/geometric_native_common_material_surface_transport_validation/native_control.py",
    "analysis/geometric_native_common_material_surface_transport_validation/validate.py",
))
TEST_PATH = "tests/test_geometric_native_common_material_surface_transport.py"
CONTROL = "analysis.geometric_native_common_material_surface_transport_validation.native_control:run"
OLD_MAP_RAW = "8f7608219c398a05913d96e80baf685b997dccff466ee9cf9eb1aa89b8124675"
BOOTSTRAP_REPORT_SEAL = "5076cb213fdcef2ab59666ea525c6793d75fa12ea2a5a08011b2f631613b44ca"
PRIOR_PACKET = OLD_PARENT / "release_validation_03/native_control/source_packet.json"
PRIOR_PACKET_RAW = "2ddf12415b9abecbbeead78a203ba82d7e18fea84bc9197aa789a607514939b2"
PRIOR_PACKET_SEAL = "f4c5f9519698c28bf12b9854dc05f78704f25068fca427478a0cfc98786f1a8d"
PRIOR_READBACK_PATH = Path("/tmp/native_common_material_surface_transport_gate_prior_readback_20261006.json")
CURRENT_EVIDENCE = {
    "baseline_875_bootstrap_source.txt": "85de6cb6261971ed36e78756c038455a3ec114bccac65f2a6ea3704bcc30355e",
    "baseline_875_bootstrap.json": "1bae41cd1b0f5abc11decd794878c4084742e61e68c2e537fa8026275aeaaf1d",
    "baseline_875_reusable_source.txt": "ad75a879a8485a22e21b44b97f42e89c43c408e399b4abd3a11cfcdb1adf8682",
}


REMAINING_CONDITIONS = (
    "Whole-source physical material and old-force/prestress/work handover, HOT law/mean chart, loads/drag and NEW mixed equilibrium",
    "Whole-clock all-pair admission including reverse C5 to C1",
    "Accepted continuation with independent path work, balances and joint restart",
)
EXPECTED_NATIVE_SUMMARY = dict(physical_candidate_status=PHYSICAL_SCOPE,
        genuine_cold_cut_count=5, complete_surface_query_count=10,
        complete_original_material_query_count=64, original_material_attachment_count=32,
        archived_original_donor_attachment_count=8, original_material_query_depth_count=2,
        authentic_saved_nonzero_cold_coordinate_count=56,
        finite_local_nanson_surface_measure_transport_defined=True,
        current_source_memory_material_scalars_retained=True,
        same_parent_original_material_field_extension_declared=True,
        both_deformed_material_surface_sides_stored=True,
        physical_source_material_handover_certified=False,
        whole_face_measure_or_memory_integral_certified=False,
        whole_archived_three_dimensional_history_recovered=False,
        current_donor_receiver_sheets_assumed_glued=False,
        current_deformed_follower_normal_contact_law_defined=False,
        old_mechanical_atom_areas_smeared_into_continuous_density=False,
        old_source_mechanical_potential_modified=False,
        continuous_face_energy_added_to_old_source_potential=False,
        hot_receiving_contact_law_inferred=False,
        historical_cumulative_work_reconstructed_from_current_stretch=False,
        new_mixed_force_equilibrium_certified=False,
        whole_clock_all_pair_admission_authorized=False,
        physical_material_admission_authorized=False,
        source_history_advanced=False, fresh_tensor_quadrature_executed=False,
        historical_force_solve_reexecuted=False,
        accepted_original_native_intervals=1, accepted_new_native_intervals=0,
        original_point_1_remaining_work_blocks=1, original_point_1_completed=False,
        remaining_physical_conditions=list(REMAINING_CONDITIONS))


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


def verify_prior(*, output_path=PRIOR_READBACK_PATH, expected_source_map=None):
    for name, expected in CURRENT_EVIDENCE.items():
        exact(digest(PARENT / name), expected, "Immutable875 recovery evidence changed: " + name)
    bootstrap = sealed(PARENT / "baseline_875_bootstrap.json", "audit_sha256")
    exact(bootstrap["audit_sha256"], BOOTSTRAP_REPORT_SEAL, "Immutable875 bootstrap seal")
    exact(bootstrap["status"], "PASS_FINAL875_EVIDENCE_READBACK_ONLY", "Immutable875 bootstrap status")
    before = source_map()
    if expected_source_map is not None:
        exact(before, expected_source_map, "Complete actual879 source map before prior readback")
    name = "material_surface_transport_gate_stdlib875_readback"
    module = types.ModuleType(name)
    module.__file__ = str(PARENT / "baseline_875_reusable_source.txt")
    sys.modules[name] = module
    exec(compile(Path(module.__file__).read_bytes(), module.__file__, "exec"), module.__dict__)
    with redirect_stdout(io.StringIO()):
        baseline = module.run(ADDED_SOURCE_PATHS, output_path=output_path,
            bootstrap_source_path=PARENT / "baseline_875_bootstrap_source.txt",
            bootstrap_report_path=PARENT / "baseline_875_bootstrap.json")
    exact(read(output_path), baseline, "Returned and saved immutable875 receipt")
    sealed(output_path, "audit_sha256")
    for key, value in (
        ("status", "PASS_IMMUTABLE875_WITH_DECLARED_SOURCE_ADDITIONS_ONLY"),
        ("baseline_source_count", 875), ("current_python_source_count", 879),
        ("additional_python_source_count", 4), ("all875_python_sources_unchanged", True),
        ("source_epoch", "IMMUTABLE875_TO_EXACT_AUTHORIZED_MATERIAL_SURFACE879"),
        ("declared_next_component_physical_scope", PHYSICAL_SCOPE),
        ("new_component_physics_validated", False), ("exact_authorized_four_path_whitelist_checked", True),
        ("complete_full_actual_source_map_independently_checked_before_after", True),
        ("complete_declared879_source_map_before_after_identical", True),
        ("current_python_source_map_sha256", value_digest(before)),
        ("frozen875_source_map_canonical_sha256", OLD_MAP), ("frozen875_source_map_raw_sha256", OLD_MAP_RAW),
        ("frozen875_projection_disclosed", True), ("nested_reader_actual875_count_and_flags_are_projected_view_only", True),
        ("preserved_closed875_manifest_file_count", 73), ("preserved_old1633_artifact_chain", True),
        ("authoritative_ancestry_release_closure_count", 9), ("bootstrap_evidence_preserved", True),
        ("recorded_raw_dependency_pin_count", 180),
        ("all180_recorded_dependency_raw_sha256_hashed_by_wrapper_before_and_nested_reader_after", True),
        ("saved_baseline_temp_and_permanent_copy_association_verified_before_after", True),
        ("bootstrap_reader_raw_sha256", CURRENT_EVIDENCE["baseline_875_bootstrap_source.txt"]),
        ("bootstrap_report_raw_sha256", CURRENT_EVIDENCE["baseline_875_bootstrap.json"]),
        ("bootstrap_report_selfseal", BOOTSTRAP_REPORT_SEAL), ("preexisting_project_modules_preserved", True),
        ("project_imports", False), ("baseline1633_reexecuted", False),
        ("math_or_solver_or_test_or_quadrature_reexecuted", False),
        ("auditor_source_sha256", CURRENT_EVIDENCE["baseline_875_reusable_source.txt"]),
    ):
        exact(baseline[key], value, "Immutable875 receipt: " + key)
    exact(baseline["declared_additional_source_paths"], sorted(ADDED_SOURCE_PATHS), "Exact four declared additions")
    exact(baseline["additional_python_sources"], {path: before[path] for path in sorted(ADDED_SOURCE_PATHS)}, "Exact added879 source pins")
    exact(baseline["dependency_raw_sha256"], bootstrap["dependency_raw_sha256"], "Exact preserved180 dependency map")
    inherited_path = OLD_PARENT / "frozen_source_map_875.json"
    exact(digest(inherited_path), OLD_MAP_RAW, "Old875 frozen source bytes")
    inherited = read(inherited_path)
    exact(len(inherited), 875, "Old875 source count")
    exact(value_digest(inherited), OLD_MAP, "Old875 canonical source map")
    exact(source_map(), before, "Complete actual879 source map after prior readback")
    for name, expected in CURRENT_EVIDENCE.items():
        exact(digest(PARENT / name), expected, "Immutable875 recovery evidence after: " + name)
    return inherited, baseline


def _validate_source_map(actual, inherited):
    exact(len(actual), 879, "Complete actual879 source count")
    exact(set(actual) - set(inherited), set(ADDED_SOURCE_PATHS), "Exactly four additive material-surface sources")
    for name, pin in inherited.items():
        exact(actual.get(name), pin, "Every old875 Python source unchanged: " + name)


def _check_native_control(out):
    from analysis.geometric_native_common_material_surface_transport_validation import native_control as control
    native = sealed(out / "native_control/report.json", "report_sha256")
    packet = sealed(out / "native_control/source_packet.json", "packet_sha256")
    exact(digest(PRIOR_PACKET), PRIOR_PACKET_RAW, "Completed875 predecessor packet bytes")
    exact(sealed(PRIOR_PACKET, "packet_sha256")["packet_sha256"], PRIOR_PACKET_SEAL, "Completed875 predecessor packet seal")
    for actual, expected, label in (
        (native.get("status"), "PASS", "Native finite transport status"),
        (native.get("format"), "native-common-material-surface-transport-control-1", "Native finite transport format"),
        (packet.get("format"), "native-common-material-surface-transport-packet-1", "Finite transport packet format"),
        (native.get("source_packet_sha256"), packet["packet_sha256"], "Native source packet binding"),
        (native.get("surface_queries_sha256"), control.digest(packet["surface_queries"]), "Exact surface query binding"),
        (native.get("original_material_queries_sha256"), control.digest(packet["original_material_queries"]), "Exact original material query binding"),
        (native.get("original_cold_body_contexts_sha256"), control.digest(packet["original_cold_body_contexts"]), "Exact complete original cold body context binding"),
        (packet.get("prior_875_packet_raw_sha256"), PRIOR_PACKET_RAW, "Explicit predecessor raw association"),
        (packet.get("prior_875_packet_selfseal"), PRIOR_PACKET_SEAL, "Explicit predecessor seal association"),
    ):
        exact(actual, expected, label)
    exact(type(packet["surface_queries"]), list, "Literal surface query list")
    exact(type(packet["original_material_queries"]), list, "Literal original material query list")
    exact(len(packet["surface_queries"]), 10, "Exactly ten complete surface queries")
    exact(len(packet["original_material_queries"]), 64, "Exactly64 complete original material queries")
    exact(type(packet["original_cold_body_contexts"]), list, "Literal original cold body context list")
    exact(len(packet["original_cold_body_contexts"]), 8, "Exactly eight complete original cold body contexts")
    result = control.replay_packet(packet)
    exact(result, EXPECTED_NATIVE_SUMMARY, "Complete fixed finite transport scope/metadata")
    expected_native = set(result) | {"format", "status", "source_packet_sha256", "surface_queries_sha256",
        "original_material_queries_sha256", "original_cold_body_contexts_sha256", "seconds", "report_sha256"}
    exact(set(native), expected_native, "Complete finite material-surface native report schema")
    exact({key: native.get(key) for key in result}, result, "Complete finite transport native metadata")
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
        raise ValueError("Exactly the new material-surface test module and control are authorized")
    if type(FRESH_TEST_COUNT) is not int or FRESH_TEST_COUNT <= 0:
        raise ValueError("The root must freeze the exact fresh test count")
    started = time.perf_counter()
    before = source_map()
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = dict(format="native-common-material-surface-transport-same-process-validation-1",
        status="FAIL", physical_candidate_status="NOT_CHECKED", started_utc=datetime.now(timezone.utc).isoformat(),
        source_count=len(before), source_map_before_sha256=value_digest(before), prior_source_map_sha256=OLD_MAP,
        execution_mode="same_python_process", pytest_import_mode="importlib", historical_tests_reexecuted=False,
        baseline1633_reexecuted=False,
        old1633_preservation_readback="SAVED_BEFORE_INDEPENDENT_MATH_RECEIPT_PLUS_RECORDED180_RAW_PINS_AND_NINE_RELEASE_CLOSURES",
        test_paths=list(test_paths), control=control, expected_fresh_test_count=FRESH_TEST_COUNT,
        verified_current_baseline_evidence=CURRENT_EVIDENCE)
    try:
        inherited, baseline = verify_prior(expected_source_map=before)
        report.update(verified_prior_python_sources=875, verified_prior_artifacts=1633,
            prior_baseline_before=baseline, prior_baseline_before_path=str(PRIOR_READBACK_PATH),
            prior_baseline_before_raw_sha256=digest(PRIOR_READBACK_PATH),
            prior_baseline_before_selfseal=baseline["audit_sha256"],
            added_python_sources={name: before[name] for name in sorted(ADDED_SOURCE_PATHS)})
        _validate_source_map(before, inherited)
        frozen = PARENT / "frozen_source_map_879.json"
        exact(read(frozen), before, "Complete879 source freeze must match actual source bytes")
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
        exact(source_map(), before, "All sources unchanged before fresh imports")
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
        exact(code, 0, "Fresh finite material-surface pytest exit")
        exact(counts, {"passed": FRESH_TEST_COUNT}, "Exact fresh finite material-surface test count")
        exact(source_map(), before, "All sources unchanged during fresh tests")
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
        after_path = PRIOR_READBACK_PATH.with_name(PRIOR_READBACK_PATH.stem + "_after.json")
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
        frozen = PARENT / "frozen_source_map_879.json"
        try:
            exact(read(frozen), before, "Complete879 frozen map unchanged after gate")
            exact(digest(frozen), report.get("frozen_source_map_raw_sha256"), "Complete879 frozen map raw bytes after gate")
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
