"""Fresh declared-area/serial-carrier component gate; old853 stays immutable."""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from fractions import Fraction as F
import hashlib
import importlib
import importlib.util
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis.geometric_native_common_contact_force_validation import validate as previous
from analysis.geometric_native_finite_equilibrium_validation.validate import _purge_added_source_modules
from analysis.geometric_native_common_contact_transfer_validation import native_control as control_api

PHYSICAL_SCOPE = control_api.PHYSICAL_SCOPE
FALSE_SCOPE = control_api.FALSE_SCOPE
RELEASE = ROOT / "analysis/geometric_native_common_contact_force_validation/release_validation_02"
EXPECTED_REPORT = "1d7a6ac8113a408aab6764f1c4a9fccb09f8d48184f5118434fc685574504313"
EXPECTED_SOURCE_MAP = "9d4a7a71f81b8093b09d26c74f742909839496fe18dd1429875e3a6d54c05c21"
ADDED_SOURCE_PATHS = frozenset((
    "tectonics/geometric_native_common_contact_measure.py",
    "tectonics/geometric_native_common_contact_series.py",
    "tests/test_geometric_native_common_contact_measure.py",
    "tests/test_geometric_native_common_contact_series.py",
    "analysis/geometric_native_common_contact_transfer_validation/native_control.py",
    "analysis/geometric_native_common_contact_transfer_validation/validate.py",
))
PRIOR_EVIDENCE = {
    "frozen_source_map_853.json": "f9fafa4aedce0265147d44f9338cfcbc8b49518f66dfa5a4ce0a4a0ee968420e",
    "independent_audit.json": "e4c2cb68ed6a9f102b5ec2628f7c38966e90a067df8c80d2fd055dccec87fc92",
    "independent_audit_source.txt": "8d755b719156876a87cb6f51b60221159d8d96e3dd92688e0b696f4bc12ea9e8",
    "independent_force_formula_helper_source.txt": "061505c158ae8026549788c242fbda541c0718526d61c452f07e9f32d24d50cd",
}
CURRENT_EVIDENCE = {
    "baseline_853_audit.json": "c6d9702e0684e48d88d30b3361d942116ad484f1211aed4f0bf9f32cfcdf9750",
    "baseline_853_audit_source.txt": "c92d970886f7bccdb0b570e605424c1eab5435638c8737525c360ac35b0bd748",
}
encoded, value_digest = previous.encoded, previous.value_digest
read, sealed, source_map, parse_counts = previous.read, previous.sealed, previous.source_map, previous.parse_counts


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def verify_prior():
    inherited = previous.verify_prior()
    report = sealed(RELEASE / "report.json", "report_sha256")
    if (digest(RELEASE / "report.json") != EXPECTED_REPORT
            or report.get("status") != "PASS" or report.get("pytest_counts") != {"passed": 19}
            or report.get("pytest_exit_code") != 0 or report.get("source_count") != 853
            or report.get("source_map_before_sha256") != EXPECTED_SOURCE_MAP
            or report.get("source_map_after_sha256") != EXPECTED_SOURCE_MAP
            or report.get("changed_python_paths") != []
            or report.get("physical_candidate_status") != previous.PHYSICAL_SCOPE
            or report.get("prior_artifacts_and_python_sources_preserved_after") is not True
            or set(report.get("added_python_sources", {})) != previous.ADDED_SOURCE_PATHS
            or any(report.get(name) is not False for name in previous.FALSE_SCOPE)):
        raise ValueError("The completed853 source-current contact force component differs")
    inherited.update(report["added_python_sources"])
    if len(inherited) != 853 or value_digest(inherited) != EXPECTED_SOURCE_MAP:
        raise ValueError("The immutable853 source map differs")
    for name, expected in inherited.items():
        if digest(ROOT / name) != expected:
            raise ValueError("An immutable853 Python source changed: " + name)
    expected_outputs = {"pytest.log", "control_console.log", "native_control/report.json", "native_control/source_packet.json"}
    if (set(report.get("outputs", {})) != expected_outputs
            or {str(p.relative_to(RELEASE)) for p in RELEASE.rglob("*") if p.is_file()}
               != expected_outputs | {"report.json"}):
        raise ValueError("The completed853 four-output closure changed")
    for name, item in report["outputs"].items():
        path = RELEASE / name
        if path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
            raise ValueError("An immutable853 output changed: " + name)
    for name, expected in PRIOR_EVIDENCE.items():
        if digest(RELEASE.parent / name) != expected:
            raise ValueError("Completed853 independent evidence changed: " + name)
    if read(RELEASE.parent / "frozen_source_map_853.json") != inherited:
        raise ValueError("Frozen853 differs from its authenticated ancestry")
    audit = sealed(RELEASE.parent / "independent_audit.json", "audit_packet_sha256")
    if (audit.get("status") != "PASS" or audit.get("fresh_tests") != 19
            or audit.get("final_python_sources") != 853 or audit.get("wrapper_raw_sha256") != EXPECTED_REPORT
            or audit.get("source_map_canonical_sha256") != EXPECTED_SOURCE_MAP
            or audit.get("source_packet_raw_sha256") != report["outputs"]["native_control/source_packet.json"]["sha256"]
            or audit.get("accepted_new_native_intervals") != 0
            or audit.get("physical_material_admission_authorized") is not False):
        raise ValueError("The853 independent audit gained physical admission")
    for name, expected in CURRENT_EVIDENCE.items():
        if digest(Path(__file__).resolve().parent / name) != expected:
            raise ValueError("Current853 baseline evidence changed: " + name)
    return inherited


def _check_native_control(out):
    from tectonics.geometric_native_pose import FractionInterval as I
    from tectonics.geometric_native_normal_support import load_native_normal_support_source
    from tectonics.geometric_native_common_reference import native_common_reference_from_dict, native_common_reference_to_dict
    from tectonics import geometric_native_common_contact_cuts as cuts_api
    from tectonics import geometric_native_common_contact_ports as ports_api
    from tectonics import geometric_native_common_contact_state as state_api
    from tectonics import geometric_native_common_contact_measure as area_api
    from tectonics import geometric_native_common_contact_series as series_api
    native_path, packet_path = out / "native_control/report.json", out / "native_control/source_packet.json"
    native = sealed(native_path, "report_sha256")
    packet = sealed(packet_path, "packet_sha256")
    if (native.get("status") != "PASS" or native.get("physical_candidate_status") != PHYSICAL_SCOPE
            or packet.get("physical_candidate_status") != PHYSICAL_SCOPE
            or native.get("source_packet_sha256") != read(packet_path)["packet_sha256"]
            or packet.get("prior_853_wrapper_raw_sha256") != EXPECTED_REPORT):
        raise ValueError("Declared-area/series output identity or scope differs")
    source = load_native_normal_support_source()
    if (packet.get("complete_source_dto_json") != [list(x) for x in source.complete_source_dto_json]
            or encoded(packet.get("saved_joint_reference_field")) != source.saved_field_json.encode()
            or packet.get("source_snapshot_sha256") != source.source_snapshot_sha256
            or native.get("source_snapshot_sha256") != source.source_snapshot_sha256
            or packet.get("hot_source_law_rejection") != "The true HOT curve has no authenticated source receiving contact law"):
        raise ValueError("The complete original DTO or saved128 source changed")
    reference = native_common_reference_from_dict(packet["common_reference"], source=source)
    if native_common_reference_to_dict(reference) != packet["common_reference"]:
        raise ValueError("The genuine common reference changed")
    cuts = cuts_api.common_contact_cuts_from_dict(packet["cut_set"], reference=reference)
    ports = ports_api.bind_native_common_contact_ports(reference, cuts=cuts, precision=96)
    cold = tuple(p for p in ports.ports if p.cut.receiving_trace_id is not None)
    if len(cold) != 5 or len(cuts.cuts) != 6:
        raise ValueError("The actual five cold and one HOT boundary set differs")
    expected_history = control_api.source_history_records(source)
    if any(packet.get(k) != v for k, v in expected_history.items()):
        raise ValueError("The original16 states, measures, work or clocks changed")
    rows = packet.get("contact_query_rows", ())
    keys = {(p.cut.cut_id, s) for p in cold for s in (1, 2)}
    if (len(rows) != 10 or {(r.get("cut_id"), r.get("sample")) for r in rows} != keys
            or native.get("contact_query_rows_sha256") != value_digest(rows)):
        raise ValueError("Two actual material queries per cold cut are required")
    by_id = {p.cut.cut_id: p for p in cold}
    state_count = force_count = hessian_count = 0
    for row in rows:
        if row.get("depth_profile_is_explicitly_declared_not_inferred") is not True or set(row.get("cases", {})) != {"zero", "nonzero"}:
            raise ValueError("A new depth density was silently inferred")
        port = by_id[row["cut_id"]]
        args = dict(control_api.probe_arguments(port.cut, row["sample"]), precision=96)
        first_state = first_measure = None
        for label in ("zero", "nonzero"):
            case = row["cases"][label]
            actual = dict(args)
            if label == "zero": actual.update(donor_coordinates_m=(F(0),)*7, receiver_coordinates_m=(F(0),)*7)
            if case.get("arguments") != control_api.payload(actual):
                raise ValueError("The actual source-labelled body query arguments changed")
            response = ports_api.evaluate_native_common_contact_port(reference, port, **actual)
            if (case.get("port_ingredients") != control_api.port_ingredients(response)
                    or case.get("measure_ingredients") != control_api.measure_ingredients(response)):
                raise ValueError("The genuine physical body ingredients changed")
            state = state_api.native_common_contact_state_observation_from_dict(case["source_observation"], reference=reference, port_response=response)
            declaration = area_api.native_common_contact_reference_measure_declaration_from_dict(
                case["measure_declaration"], reference=reference, port_response=response, state=state)
            measure = area_api.native_common_contact_reference_measure_response_from_dict(
                case["measure_response"], reference=reference, port_response=response, state=state, declaration=declaration)
            carrier = series_api.native_common_contact_series_carrier_declaration_from_dict(
                case["carrier_declaration"], reference=reference, port_response=response, source_observation=state)
            profile = (F(1),) if row["sample"] == 1 else (F(0), F(2))
            if (declaration.provenance != control_api.PROVENANCE or carrier.provenance != control_api.PROVENANCE
                    or declaration.depth_profile_coefficients != profile
                    or declaration.depth_profile_integral_exact != 1
                    or carrier.anchor_interpretation != series_api.ANCHOR_INTERPRETATION
                    or measure.physical_area_density_bounds_m2_per_parameter_per_physical_volume_depth_m
                        != response.geometry.area_density_bounds_m2_per_parameter_per_material_m
                    or measure.reference_area_density_bounds_m2_per_parameter_per_physical_volume_depth_m.lo <= 0
                    or state.source_current_node_epoch != "unaccepted_candidate"
                    or (state.source_memory_elapsed_time_s, state.source_native_elapsed_time_s, state.source_contact_elapsed_time_s)
                        != (F(1000000), F(1000000), F(2000000))):
                raise ValueError("The declared profile, source clocks, genuine area or serial anchor model changed")
            budgets, budget_parameters = control_api.rectangle_budgets(declaration)
            full = area_api.source_reference_rectangle_area(declaration,
                computational_angle_bounds=I(0, declaration.source_computational_extent_radians), normalized_depth_bounds=I(0, 1))
            parts = sum((area_api.source_reference_rectangle_area(declaration,
                computational_angle_bounds=I(0, declaration.source_computational_extent_radians), normalized_depth_bounds=d)
                for d in (I(0, F(1, 3)), I(F(1, 3), 1))), F(0))
            if (full != parts or case.get("source_rectangle_area_budgets") != budgets
                    or case.get("source_rectangle_budget_parameters") != budget_parameters
                    or case.get("source_budget_allocated_in_full_to_this_cut") is not False):
                raise ValueError("A pure source rectangle budget was duplicated or declared target allocation")
            if first_state is None: first_state, first_measure = state, measure
            elif state != first_state or measure != first_measure:
                raise ValueError("Fresh modal probes changed frozen source memory/reference measure")
            state_count += 1
            if set(case.get("series_responses", {})) != {"relaxed_source", "frozen_memory_predictor"}:
                raise ValueError("Both serial branches are required")
            for branch, export in case["series_responses"].items():
                value = series_api.native_common_contact_series_response_from_dict(
                    export["response"], reference=reference, port_response=response, source_observation=state, declaration=carrier)
                if export.get("reference_weighted_integrands") != control_api.reference_weighted_integrands(value, measure):
                    raise ValueError("The frozen reference-weighted local integrands changed")
                count = len(value.kinematics.coordinate_order)
                if (value.branch != branch or value.precision != 96
                        or value.energy_component_order != ("contact", "sub_column", "over_column")
                        or value.total_energy_bounds_j_m2 != sum(value.energy_component_bounds_j_m2, I(0))
                        or value.traction_bounds_pa[0].hi >= 0
                        or value.local_tangent_bounds_pa_m[0][0] != I(carrier.exact_normal_series_stiffness_pa_m)
                        or value.source_memory_proposal_applied
                        or value.old_world_or_generalized_force_handover_certified
                        or value.old_rod_three_dimensional_material_map_certified
                        or value.all_sixteen_source_carriers_physically_transferred):
                    raise ValueError("The complete serial energy lost its declared mechanical scope")
                if branch == "relaxed_source" and (value.traction_bounds_pa[1] != I(0)
                        or value.local_tangent_bounds_pa_m[1][1] != I(0)):
                    raise ValueError("The relaxed zero-strength source law gained elastic shear")
                if label == "zero" and (value.traction_bounds_pa[0] != -state.normal_pressure_bounds_pa
                        or value.total_energy_bounds_j_m2 != value.reference_total_energy_bounds_j_m2):
                    raise ValueError("Zero increments altered source preload or split reference energy")
                if count != 20 or any(value.hessian_bounds_per_reference_m2[i][j] != value.hessian_bounds_per_reference_m2[j][i] for i in range(count) for j in range(count)):
                    raise ValueError("The full20 symmetric serial Hessian is missing")
                force_count += 1; hessian_count += count*count
    summary = dict(actual_cold_contact_cut_count=5, retained_true_hot_curve_count=1,
        current_state_query_case_count=20, reference_area_density_case_count=20,
        serial_force_density_case_count=40, reference_weighted_integrand_case_count=40,
        local_coordinate_count=20, exact_symmetric_hessian_entry_replay_count=16000,
        declared_normalized_depth_profile_count=2,
        complete_sixteen_source_states_areas_work_and_clocks_preserved=True,
        variable_contact_subrod_overrod_energies_included=True,
        complete_series_compliance_not_parallel_energy_used=True,
        source_continuous_reference_measure_kept_separate_from_sixteen_atom_weights=True,
        accepted_original_native_intervals=1, accepted_new_native_intervals=0,
        original_point_1_remaining_work_blocks=1)
    if (state_count, force_count, hessian_count) != (20, 40, 16000) or any(native.get(k) != v or packet.get(k) != v for k, v in summary.items()):
        raise ValueError("The declared-area/series complete replay counts differ")
    if (any(native.get(k) is not False or packet.get(k) is not False for k in FALSE_SCOPE)
            or len(packet.get("remaining_physical_conditions", ())) != 3
            or native.get("remaining_physical_conditions") != packet.get("remaining_physical_conditions")):
        raise ValueError("The reduced carrier gained physical transfer, equilibrium or admission")
    return dict(physical_candidate_status=PHYSICAL_SCOPE, **summary, **{k: False for k in FALSE_SCOPE})


def run(out, test_paths, control):
    tests = tuple(test_paths)
    if (len(tests) != 2 or set(tests) != {p for p in ADDED_SOURCE_PATHS if p.startswith("tests/")}
            or control != "analysis.geometric_native_common_contact_transfer_validation.native_control:run"):
        raise ValueError("Exactly the two new test modules and this stage callback are required")
    inherited = verify_prior(); before = source_map(); added = set(before)-set(inherited)
    if len(before) != 859 or added != ADDED_SOURCE_PATHS:
        raise ValueError("Exactly6 additive area/series Python sources are required")
    removed = []
    for name in sorted(added):
        cache = Path(importlib.util.cache_from_source(str(ROOT/name)))
        if cache.is_file(): cache.unlink(); removed.append(str(cache.relative_to(ROOT)))
    purged = _purge_added_source_modules(added-{str(Path(__file__).resolve().relative_to(ROOT))})
    importlib.invalidate_caches()
    if source_map() != before: raise ValueError("A source changed before fresh imports")
    out = Path(out).resolve(); out.mkdir(parents=True, exist_ok=False); started = time.perf_counter()
    report = dict(format="native-common-reference-area-series-same-process-validation-1", status="FAIL",
        physical_candidate_status="NOT_CHECKED", started_utc=datetime.now(timezone.utc).isoformat(),
        source_count=len(before), source_map_before_sha256=value_digest(before), prior_source_map_sha256=EXPECTED_SOURCE_MAP,
        verified_prior_python_sources=853, verified_prior_artifacts=1633,
        added_python_sources={p: before[p] for p in sorted(added)}, execution_mode="same_python_process",
        pytest_import_mode="importlib", historical_tests_reexecuted=False, test_paths=list(tests), control=control,
        removed_added_source_bytecode_caches=removed, purged_previously_imported_added_source_modules=purged,
        verified_prior_evidence=PRIOR_EVIDENCE, verified_current_baseline_evidence=CURRENT_EVIDENCE)
    try:
        import pytest
        start = time.perf_counter(); old_dir = Path.cwd()
        try:
            os.chdir(ROOT)
            with (out/"pytest.log").open("x", buffering=1) as stream, redirect_stdout(stream), redirect_stderr(stream):
                code = int(pytest.main(["-q", "-p", "no:cacheprovider", "--import-mode=importlib", *tests]))
        finally: os.chdir(old_dir)
        summary, counts = parse_counts(out/"pytest.log")
        report.update(pytest_exit_code=code, pytest_counts=counts, pytest_summary=summary, pytest_seconds=time.perf_counter()-start)
        if code or counts.get("passed", 0)<1 or any(counts.get(k, 0) for k in ("failed", "skipped", "xfailed", "xpassed", "error", "errors")):
            raise RuntimeError("Fresh declared-area/serial-carrier tests failed")
        if source_map()!=before: raise ValueError("A source changed during fresh tests")
        module, fn = control.split(":", 1); callback = getattr(importlib.import_module(module), fn); start=time.perf_counter()
        with (out/"control_console.log").open("x", buffering=1) as stream, redirect_stdout(stream), redirect_stderr(stream): callback(out/"native_control")
        report["native_control_seconds"] = time.perf_counter()-start
        report.update(_check_native_control(out))
        outputs={"pytest.log", "control_console.log", "native_control/report.json", "native_control/source_packet.json"}
        if {str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()}!=outputs: raise ValueError("Exactly4 outputs required")
        report["status"]="PASS"
    except BaseException as error:
        report.update(failure_type=type(error).__name__, failure_message=str(error), failure_traceback=traceback.format_exc())
    finally:
        after=source_map();report["source_map_after_sha256"]=value_digest(after)
        report["changed_python_paths"]=sorted(p for p in set(before)|set(after) if before.get(p)!=after.get(p))
        try: verify_prior();report["prior_artifacts_and_python_sources_preserved_after"]=True
        except BaseException as error: report.update(prior_artifacts_and_python_sources_preserved_after=False,prior_after_error=str(error))
        if report["changed_python_paths"] or not report["prior_artifacts_and_python_sources_preserved_after"]:report["status"]="FAIL"
        report["seconds"]=time.perf_counter()-started;report["finished_utc"]=datetime.now(timezone.utc).isoformat()
        report["outputs"]={str(p.relative_to(out)):dict(sha256=digest(p),bytes=p.stat().st_size) for p in sorted(out.rglob("*")) if p.is_file()}
        report["report_sha256"]=value_digest(report);(out/"report.json").write_bytes(encoded(report)+b"\n")
        print(encoded({k:report.get(k) for k in ("status","physical_candidate_status","pytest_counts","seconds","source_count","changed_python_paths")}).decode(),flush=True)
    if report["status"]!="PASS":raise RuntimeError("Validation refused; see "+str(out/"report.json"))
    return report


if __name__ == "__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--out",required=True);parser.add_argument("--control",required=True);parser.add_argument("tests",nargs="+")
    args=parser.parse_args();run(args.out,args.tests,args.control)
