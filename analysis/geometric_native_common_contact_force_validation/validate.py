"""Fresh source-current memory and objective local contact-only force gate.

All 847 historical Python sources and their release artifacts remain byte
identical.  Local per-area forces neither allocate face memory/measure nor
transfer old rod preload, solve equilibrium, or admit a physical interval.
"""
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
from analysis.geometric_native_common_contact_validation import validate as previous
from analysis.geometric_native_finite_equilibrium_validation.validate import _purge_added_source_modules

RELEASE = ROOT / "analysis/geometric_native_common_contact_validation/release_validation_01"
EVIDENCE = RELEASE.parent
EXPECTED_REPORT = "e8ffabb1a29c1fb021e94fe53e815b6ad8d16ea14e9103a30b9f7428ac9ab847"
EXPECTED_REPORT_SEAL = "19bd6b63478c6da22d48b5606311c0473ae8d3df732753f6150051d8345f51cd"
EXPECTED_NATIVE = "66968aeb22a287eac8bdfd795cc916dded5bc3db55bed47da52f4f35d86a5dd8"
EXPECTED_PACKET = "8fa086a0d42b1cb2b81c30d01c61369d13e294d4edf6c10f7d3b8a9a3dc99308"
EXPECTED_SOURCE_MAP = "a14c79ddf7e0146ba6b5c335c800e03e442651b8a08fccc855a6ac920ca34a1c"
PHYSICAL_SCOPE = "COMMON_CONTACT_REST_LAW_COMPONENT_ONLY"
ADDED_SOURCE_PATHS = frozenset((
    "tectonics/geometric_native_common_contact_state.py",
    "tectonics/geometric_native_common_contact_constitutive.py",
    "tests/test_geometric_native_common_contact_state.py",
    "tests/test_geometric_native_common_contact_constitutive.py",
    "analysis/geometric_native_common_contact_force_validation/native_control.py",
    "analysis/geometric_native_common_contact_force_validation/validate.py",
))
STANDALONE_EVIDENCE = dict(previous.CURRENT_SOURCE_REVIEW_EVIDENCE, **{
    "frozen_source_map_847.json": "22bb02607b88935e8c93764813c2721dad0c9c72a241ffb67c2e602b8fee9779",
    "independent_audit.json": "65969e55b900d695af52025bc48718cad83bdec189fb9aab2fb5301d29be3658",
    "independent_audit_source.txt": "31b7a3535b564482f163dbc73321477d15379e7e6961b56e139d952e91df8586",
    "contact_energy_decomposition_audit.json": "b164936ac0217e4c8c4feaebc8d7495731edd218e4da93db36e1d2660a3815b0",
    "contact_energy_decomposition_audit_source.txt": "150892ef900d14bb36160af5b492d755700e071d3c2e845f18adc4b9ac7037a2",
    "contact_force_contract.txt": "51f20ac273143c1184ae2ad3256ba0b938d793ea8b539a739dadedb7f54f04da",
    "contact_force_contract.json": "b288a411e53ec56b2511d4c3e51a79f676c6fd6473dc6273f8c9b3466f4901fb",
})
CURRENT_SOURCE_REVIEW_EVIDENCE = {
    "baseline_847_audit_source.txt": "e54df4a17f7364437f991f90b7883194277e0d5287a1941d0d9ac41d2af6724e",
    "baseline_847_audit.json": "a74d392b5e92e78fd03c41d675cd26f829302c281203e22830deee625cb5daab",
}
FALSE_SCOPE = (
    "new_mixed_force_equilibrium_certified", "whole_clock_material_admission_authorized",
    "physical_material_admission_authorized", "original_point_1_completed",
    "face_depth_memory_or_measure_transfer_certified", "old_rod_prestrain_or_energy_transferred",
    "old_node_states_assigned_to_new_cuts", "source_history_advanced",
    "hot_curve_contact_law_inferred", "finite_owner_whole_clock_motion_certified",
    "full_genuine_source_reconstructed", "inherited_force_certificate_for_placement",
    "contact_energy_adopted_on_new_face", "complete_three_dimensional_strain_energy_certified",
)
encoded, value_digest = previous.encoded, previous.value_digest
read, sealed, source_map, parse_counts = previous.read, previous.sealed, previous.source_map, previous.parse_counts


def digest(path):
    """Preserve byte identity without loading the giant historical packets."""
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def verify_prior():
    inherited = previous.verify_prior()
    report = sealed(RELEASE / "report.json", "report_sha256")
    if (digest(RELEASE / "report.json") != EXPECTED_REPORT
            or read(RELEASE / "report.json").get("report_sha256") != EXPECTED_REPORT_SEAL
            or report.get("status") != "PASS"
            or report.get("pytest_counts") != {"passed": 25}
            or report.get("pytest_exit_code") != 0
            or report.get("physical_candidate_status") != previous.PHYSICAL_SCOPE
            or report.get("source_count") != 847
            or report.get("source_map_before_sha256") != EXPECTED_SOURCE_MAP
            or report.get("source_map_after_sha256") != EXPECTED_SOURCE_MAP
            or report.get("changed_python_paths") != []
            or report.get("prior_artifacts_and_python_sources_preserved_after") is not True
            or set(report.get("added_python_sources", {})) != previous.ADDED_SOURCE_PATHS
            or set(inherited) & set(report["added_python_sources"])):
        raise ValueError("The completed contact-geometry 847 source epoch differs")
    inherited.update(report["added_python_sources"])
    if len(inherited) != 847 or value_digest(inherited) != EXPECTED_SOURCE_MAP:
        raise ValueError("The immutable 847 source map differs")
    for name, expected in inherited.items():
        if digest(ROOT / name) != expected:
            raise ValueError("An original Python source changed: " + name)
    outputs = {"pytest.log", "control_console.log", "native_control/report.json", "native_control/source_packet.json"}
    if set(report.get("outputs", {})) != outputs:
        raise ValueError("The completed 847 output manifest differs")
    for name, entry in report["outputs"].items():
        path = RELEASE / name
        if path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]:
            raise ValueError("An immutable completed 847 output changed: " + name)
    if {str(p.relative_to(RELEASE)) for p in RELEASE.rglob("*") if p.is_file()} != outputs | {"report.json"}:
        raise ValueError("The sealed historical 847 output closure changed")
    if (report["outputs"]["native_control/report.json"]["sha256"] != EXPECTED_NATIVE
            or report["outputs"]["native_control/source_packet.json"]["sha256"] != EXPECTED_PACKET
            or report.get("accepted_original_native_intervals") != 1
            or report.get("accepted_new_native_intervals") != 0
            or report.get("original_point_1_remaining_work_blocks") != 1
            or any(report.get(name) is not False for name in previous.FALSE_SCOPE)):
        raise ValueError("The historical contact component gained physical acceptance")
    for name, expected in STANDALONE_EVIDENCE.items():
        if digest(EVIDENCE / name) != expected:
            raise ValueError("Immutable completed contact evidence changed: " + name)
    if read(EVIDENCE / "frozen_source_map_847.json") != inherited:
        raise ValueError("The recorded 847 source map differs from its ancestry")
    audit = sealed(EVIDENCE / "independent_audit.json", "audit_packet_sha256")
    if (audit.get("status") != "PASS" or audit.get("fresh_tests") != 25
            or audit.get("final_python_sources") != 847
            or audit.get("source_map_canonical_sha256") != EXPECTED_SOURCE_MAP
            or audit.get("wrapper_raw_sha256") != EXPECTED_REPORT
            or audit.get("native_raw_sha256") != EXPECTED_NATIVE
            or audit.get("source_packet_raw_sha256") != EXPECTED_PACKET
            or audit.get("material_admission_authorized") is not False
            or audit.get("second_certified_tensor_quadrature_performed") is not False):
        raise ValueError("The independently sealed completed 847 audit differs")
    for name, expected in CURRENT_SOURCE_REVIEW_EVIDENCE.items():
        if digest(Path(__file__).resolve().parent / name) != expected:
            raise ValueError("The current standalone baseline evidence changed: " + name)
    return inherited


def _check_native_control(out):
    from analysis.geometric_native_common_contact_force_validation import native_control as control
    from tectonics.geometric_native_pose import FractionInterval as I
    from tectonics.geometric_native_normal_support import load_native_normal_support_source
    from tectonics.geometric_native_common_reference import native_common_reference_from_dict, native_common_reference_to_dict
    from tectonics import geometric_native_common_contact_cuts as cuts_api
    from tectonics import geometric_native_common_contact_ports as ports_api
    from tectonics import geometric_native_common_contact_state as state_api
    from tectonics import geometric_native_common_contact_constitutive as force_api

    native_path, packet_path = out / "native_control/report.json", out / "native_control/source_packet.json"
    native = sealed(native_path, "report_sha256")
    packet = sealed(packet_path, "packet_sha256")
    if (native.get("status") != "PASS" or native.get("physical_candidate_status") != PHYSICAL_SCOPE
            or packet.get("physical_candidate_status") != PHYSICAL_SCOPE
            or native.get("source_packet_sha256") != read(packet_path)["packet_sha256"]
            or packet.get("authenticated_prior_contact_wrapper_raw_sha256") != EXPECTED_REPORT
            or packet.get("authenticated_prior_contact_source_packet_raw_sha256") != EXPECTED_PACKET
            or tuple(control.FALSE_SCOPE) != FALSE_SCOPE):
        raise ValueError("The local-force component lacks its sealed narrow scope")
    source = load_native_normal_support_source()
    if (packet.get("complete_source_dto_json") != [list(x) for x in source.complete_source_dto_json]
            or encoded(packet.get("saved_joint_reference_field")) != source.saved_field_json.encode()
            or packet.get("source_snapshot_sha256") != source.source_snapshot_sha256
            or native.get("source_snapshot_sha256") != source.source_snapshot_sha256):
        raise ValueError("The complete original DTO, saved128 field or source identity changed")
    reference = native_common_reference_from_dict(packet["common_reference"], source=source)
    if native_common_reference_to_dict(reference) != packet["common_reference"]:
        raise ValueError("The common runtime reference differs on source replay")
    cuts = cuts_api.common_contact_cuts_from_dict(packet["cut_set"], reference=reference)
    ports = ports_api.bind_native_common_contact_ports(reference, cuts=cuts, precision=160)
    cold = tuple(p for p in ports.ports if p.cut.receiving_trace_id is not None)
    hot = tuple(p for p in ports.ports if p.cut.receiving_trace_id is None)
    expected_pairs = {("octant:5", "octant:1"), ("octant:5", "octant:3"),
                      ("octant:6", "octant:2"), ("octant:7", "octant:2"), ("octant:7", "octant:3")}
    if (len(cuts.cuts) != 6 or len(cold) != 5 or len(hot) != 1
            or {(p.cut.donor_original_parent_id, p.cut.receiver_original_parent_id) for p in cold} != expected_pairs
            or hot[0].cut.kind != "true_curved_envelope"
            or hot[0].cut.receiver_material_kind != "hot_birth"
            or sum(c.kind == "implicit_terminal_trim" for c in cuts.cuts) != 1):
        raise ValueError("The actual five cold cuts or separate C6/HOT curve changed")
    expected_history = control.source_history_records(source)
    if any(packet.get(k) != v for k, v in expected_history.items()):
        raise ValueError("Original sixteen atom states, areas, work or differing clocks changed")
    rows = packet.get("contact_query_rows", ())
    expected_keys = {(p.cut.cut_id, sample) for p in cold for sample in (1, 2)}
    if (len(rows) != 10 or {(r.get("cut_id"), r.get("sample")) for r in rows} != expected_keys
            or native.get("contact_query_rows_sha256") != value_digest(rows)):
        raise ValueError("Exactly two strict material queries per true cold cut are required")
    by_id = {p.cut.cut_id: p for p in cold}
    state_count = force_count = hessian_count = 0
    for row in rows:
        if row.get("point_force_density_is_a_new_mixed_solution") is not False or set(row.get("cases", {})) != {"zero", "nonzero"}:
            raise ValueError("A diagnostic point force was substituted for mixed equilibrium")
        port = by_id[row["cut_id"]]
        arguments = control.probe_arguments(port.cut, row["sample"])
        original_state = None
        for label in ("zero", "nonzero"):
            case = row["cases"][label]
            actual = dict(arguments)
            if label == "zero":
                actual.update(donor_coordinates_m=(F(0),)*7, receiver_coordinates_m=(F(0),)*7)
            if case.get("arguments") != control.payload(actual):
                raise ValueError("The exact source-labelled body probe coordinates changed")
            response = ports_api.evaluate_native_common_contact_port(reference, port, **actual)
            if (case.get("port_ingredients") != control.port_ingredients(response)
                    or not response.geometry.genuine_query_support_certified
                    or not response.geometry.every_point_of_parameter_enclosure_is_material
                    or response.unresolved_parameter_bands_promoted_to_material):
                raise ValueError("Compact ingredients differ from actual source/field/owner replay")
            state = state_api.native_common_contact_state_observation_from_dict(
                case["source_observation"], reference=reference, port_response=response)
            if (state.receiving_trace_id != port.cut.receiving_trace_id
                    or state.precision != 160
                    or state.computational_angle_bounds != response.geometry.receiving_computational_angle_bounds_radians
                    or state.source_current_node_epoch != "unaccepted_candidate"
                    or state.source_time_fraction_bounds != I(1)
                    or state.source_memory_elapsed_time_s != F(1000000)
                    or state.source_native_elapsed_time_s != F(1000000)
                    or state.source_contact_elapsed_time_s != F(2000000)
                    or state.source_memory_stage_count != 1 or state.source_contact_step_count != 2
                    or not state.literal_current_memory_used
                    or state.shear_traction_bounds_pa != I(0)
                    or not state.memory_piece_intersections
                    or any(getattr(state, key) is not False for key in (
                        "current_memory_replaced_with_previous_or_dense_query", "source_node_states_assigned_to_new_cut",
                        "face_depth_memory_or_measure_transfer_certified", "old_rod_prestrain_or_energy_transferred",
                        "source_history_advanced", "contact_energy_adopted_on_new_face", "force_equilibrium_certified",
                        "physical_material_admission_authorized"))):
                raise ValueError("A current source observation lost its literal memory, receiver, clocks or scope")
            if original_state is None:
                original_state = state
            elif state != original_state:
                raise ValueError("Modal probes advanced or altered frozen source memory")
            state_count += 1
            if set(case.get("responses", {})) != set(force_api.BRANCHES):
                raise ValueError("Both relaxed and explicitly frozen predictor responses are required")
            for branch in force_api.BRANCHES:
                force = force_api.native_common_contact_constitutive_response_from_dict(
                    case["responses"][branch], reference=reference, port_response=response, source_observation=state)
                kin = force.kinematics
                count = len(kin.coordinate_order)
                if (force.branch != branch or force.receiving_trace_id != state.receiving_trace_id
                        or force.precision != 160 or kin.precision != 160
                        or force.normal_stiffness_pa_m != state.normal_stiffness_pa_m
                        or force.tangential_stiffness_pa_m != state.tangential_stiffness_pa_m
                        or kin.receiver_frame != force_api.FRAME or count != 20 or kin.body_coordinate_count != 14
                        or kin.donor_owner != port.donor_body.original_body.owner
                        or kin.receiver_owner != port.receiver_body.original_body.owner
                        or not kin.symmetric_coordinate_owner_second_derivatives
                        or not kin.full_receiver_frame_derivatives_included
                        or not kin.shared_owner_scalar_partials_cancel_symbolically
                        or kin.old_world_traction_direction_preserved
                        or force.contact_gap_bounds_m.hi >= 0
                        or not force.source_preload_geometric_hessian_included
                        or force.source_memory_proposal_applied
                        or force.frozen_predictor_is_physical_relaxed_shear_law
                        or force.angular_contact_stiffness_assigned or force.hot_receiving_law_inferred
                        or any(getattr(force, key) is not False for key in (
                            "material_memory_or_interface_measure_transferred", "old_sixteen_atoms_or_rod_energy_assigned",
                            "current_deformed_follower_normal_law_certified", "force_equilibrium_certified",
                            "physical_material_admission_authorized", "finite_owner_motion_or_whole_clock_certified",
                            "source_state_or_thermal_state_advanced"))):
                    raise ValueError("A local law changed its source stiffness, objective chart or physical scope")
                if (len(force.residual_bounds_per_reference_m2) != count
                        or len(force.hessian_bounds_per_reference_m2) != count
                        or any(len(r) != count for r in force.hessian_bounds_per_reference_m2)
                        or any(force.hessian_bounds_per_reference_m2[i][j] != force.hessian_bounds_per_reference_m2[j][i]
                               for i in range(count) for j in range(count))):
                    raise ValueError("The complete symmetric local Hessian is missing")
                if branch == "relaxed_source":
                    if (force.traction_bounds_pa[1] != I(0)
                            or force.elastic_shear_extension_bounds_m != I(0)
                            or force.local_tangent_bounds_pa_m[1][1] != I(0)):
                        raise ValueError("The authentic relaxed zero-strength shear branch gained elastic force")
                elif force.local_tangent_bounds_pa_m[1][1] != I(state.tangential_stiffness_pa_m):
                    raise ValueError("The explicit frozen-memory predictor lost its declared elastic tangent")
                if label == "zero":
                    if (kin.scalar_increment_bounds_m != (I(0),)*3
                            or force.traction_bounds_pa[0] != -state.normal_pressure_bounds_pa
                            or force.contact_gap_bounds_m != state.contact_reference_gap_bounds_m
                            or force.recoverable_energy_bounds_j_m2 != force.reference_recoverable_energy_bounds_j_m2
                            or force.traction_bounds_pa[1] != I(0)
                            or force.proposed_plastic_increment_bounds_m != I(0)
                            or not (force.proposed_plastic_slip_bounds_m.lo <= state.plastic_slip_bounds_m.lo
                                    <= state.plastic_slip_bounds_m.hi <= force.proposed_plastic_slip_bounds_m.hi)
                            or not (force.proposed_cumulative_slip_bounds_m.lo <= state.cumulative_slip_bounds_m.lo
                                    <= state.cumulative_slip_bounds_m.hi <= force.proposed_cumulative_slip_bounds_m.hi)):
                        raise ValueError("A zero increment altered source preload or plastic/cumulative memory")
                if kin.donor_owner == kin.receiver_owner:
                    if (any(x != I(0) for vector in kin.scalar_first_derivative_bounds for x in vector[14:])
                            or any(matrix[i][j] != I(0) for matrix in kin.scalar_second_derivative_bounds
                                   for i in range(count) for j in range(count) if i >= 14 or j >= 14)):
                        raise ValueError("Common rigid owner rotation did not cancel symbolically")
                force_count += 1
                hessian_count += count*count
    hot_arguments = control.probe_arguments(hot[0].cut, 1)
    hot_response = ports_api.evaluate_native_common_contact_port(reference, hot[0], **hot_arguments)
    hot_probe = packet.get("hot_geometry_only_probe", {})
    if (hot_probe.get("arguments") != control.payload(hot_arguments)
            or hot_probe.get("port_ingredients") != control.port_ingredients(hot_response)
            or hot_probe.get("hot_contact_force_or_memory_assigned") is not False):
        raise ValueError("The separate HOT curve geometry probe changed")
    hot_kin = force_api.native_common_contact_scalar_kinematics_from_dict(
        hot_probe["scalar_kinematics"], reference=reference, port_response=hot_response)
    if len(hot_kin.coordinate_order) != 19 or hot_kin.precision != 160:
        raise ValueError("The true cold/HOT curve lost its authentic 7+6 body modes")
    try:
        state_api.observe_native_common_contact_state(reference, hot_response, precision=160)
    except ValueError as error:
        if hot_probe.get("source_law_rejection") != str(error):
            raise ValueError("The retained HOT receiving-law rejection changed") from error
    else:
        raise ValueError("The HOT curve inherited an unsupported source contact family")
    summary = dict(actual_cold_contact_cut_count=5, retained_true_hot_curve_count=1,
        current_state_query_case_count=20, local_force_density_case_count=40,
        local_coordinate_count=20, exact_symmetric_hessian_entry_replay_count=16000,
        complete_sixteen_source_states_areas_work_and_clocks_preserved=True,
        literal_current_memory_not_previous_dense_substitution=True,
        source_contact_only_per_original_reference_area=True,
        actual_friction_free_relaxed_shear_zero=True, receiver_material_director_full_derivatives_included=True,
        accepted_original_native_intervals=1, accepted_new_native_intervals=0,
        original_point_1_remaining_work_blocks=1)
    if (state_count, force_count, hessian_count) != (20, 40, 16000):
        raise ValueError("The complete local current-state/force/Hessian replay counts differ")
    if any(native.get(k) != v or packet.get(k) != v for k, v in summary.items()):
        raise ValueError("The declared current-state/local-force component counts differ")
    if (native.get("remaining_physical_conditions") != packet.get("remaining_physical_conditions")
            or len(native.get("remaining_physical_conditions", [])) != 3
            or any(native.get(k) is not False or packet.get(k) is not False for k in FALSE_SCOPE)):
        raise ValueError("The local per-area law gained transfer, force equilibrium or admission")
    return dict(physical_candidate_status=PHYSICAL_SCOPE,
        native_control_report_raw_sha256=digest(native_path), native_control_source_packet_raw_sha256=digest(packet_path),
        complete_historical_dtos_saved_field_states_and_clocks_preserved=True,
        **summary, **{name: False for name in FALSE_SCOPE})


def run(out, test_paths, control):
    tests = tuple(test_paths)
    expected_tests = {name for name in ADDED_SOURCE_PATHS if name.startswith("tests/")}
    if (len(tests) != 2 or set(tests) != expected_tests
            or any(not (ROOT / name).is_file() for name in tests)
            or control != "analysis.geometric_native_common_contact_force_validation.native_control:run"):
        raise ValueError("Exactly the two new test modules and this stage's callback are required")
    inherited = verify_prior()
    before = source_map()
    added = set(before) - set(inherited)
    if len(before) != 853 or added != ADDED_SOURCE_PATHS:
        raise ValueError("Exactly six additive contact state/constitutive Python sources are required")
    removed_caches = []
    for name in sorted(added):
        cache = Path(importlib.util.cache_from_source(str(ROOT / name)))
        if cache.is_file():
            cache.unlink()
            removed_caches.append(str(cache.relative_to(ROOT)))
    own = str(Path(__file__).resolve().relative_to(ROOT))
    purged = _purge_added_source_modules(added - {own})
    importlib.invalidate_caches()
    if source_map() != before:
        raise ValueError("A Python source changed while preparing imports")
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = dict(format="native-common-contact-force-same-process-validation-1", status="FAIL",
        regression_validation_status="FAIL", physical_candidate_status="NOT_CHECKED",
        started_utc=datetime.now(timezone.utc).isoformat(), verified_prior_python_sources=847,
        verified_prior_artifacts=1633, verified_prior_contact_outputs=4,
        verified_prior_standalone_evidence=dict(STANDALONE_EVIDENCE),
        verified_current_force_review_evidence=dict(CURRENT_SOURCE_REVIEW_EVIDENCE),
        prior_contact_report_raw_sha256=EXPECTED_REPORT, prior_contact_packet_raw_sha256=EXPECTED_PACKET,
        prior_source_map_sha256=EXPECTED_SOURCE_MAP, source_count=len(before),
        source_map_before_sha256=value_digest(before), added_python_sources={name: before[name] for name in sorted(added)},
        removed_added_source_bytecode_caches=removed_caches, purged_previously_imported_added_source_modules=purged,
        historical_tests_reexecuted=False, test_paths=list(tests), control=control,
        execution_mode="same_python_process", pytest_import_mode="importlib",
        full_genuine_source_reconstructed=False, inherited_force_certificate_for_placement=False,
        whole_native_interval_admitted_by_test_gate=False)
    try:
        import pytest
        test_started = time.perf_counter()
        old_directory = Path.cwd()
        try:
            os.chdir(ROOT)
            with (out / "pytest.log").open("x", buffering=1) as stream:
                with redirect_stdout(stream), redirect_stderr(stream):
                    code = int(pytest.main(["-q", "-p", "no:cacheprovider", "--import-mode=importlib", *tests]))
        finally:
            os.chdir(old_directory)
        summary, counts = parse_counts(out / "pytest.log")
        report.update(pytest_exit_code=code, pytest_summary=summary, pytest_counts=counts,
                      pytest_seconds=time.perf_counter()-test_started)
        if (code or counts.get("passed", 0) < 1 or any(counts.get(name, 0)
                for name in ("failed", "skipped", "xfailed", "xpassed", "error", "errors"))):
            raise RuntimeError("Fresh current-state/local-force tests failed")
        if source_map() != before:
            raise ValueError("A source changed during tests")
        module, function = control.split(":", 1)
        callback = getattr(importlib.import_module(module), function)
        control_started = time.perf_counter()
        with (out / "control_console.log").open("x", buffering=1) as stream:
            with redirect_stdout(stream), redirect_stderr(stream):
                callback(out / "native_control")
        report["native_control_seconds"] = time.perf_counter()-control_started
        report.update(_check_native_control(out))
        expected_outputs = {"pytest.log", "control_console.log", "native_control/report.json", "native_control/source_packet.json"}
        if {str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()} != expected_outputs:
            raise ValueError("Exactly four declared output files are required")
        report["status"] = report["regression_validation_status"] = "PASS"
    except BaseException as error:
        report.update(failure_type=type(error).__name__, failure_message=str(error), failure_traceback=traceback.format_exc())
    finally:
        after = source_map()
        report["source_map_after_sha256"] = value_digest(after)
        report["changed_python_paths"] = sorted(name for name in set(before) | set(after) if before.get(name) != after.get(name))
        try:
            verify_prior()
            report["prior_artifacts_and_python_sources_preserved_after"] = True
        except BaseException as error:
            report["prior_artifacts_and_python_sources_preserved_after"] = False
            report["prior_after_error"] = str(error)
        if report["changed_python_paths"] or not report["prior_artifacts_and_python_sources_preserved_after"]:
            report["status"] = report["regression_validation_status"] = "FAIL"
        report["seconds"] = time.perf_counter()-started
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        report["outputs"] = {str(p.relative_to(out)): dict(sha256=digest(p), bytes=p.stat().st_size)
            for p in sorted(out.rglob("*")) if p.is_file()}
        report["report_sha256"] = value_digest(report)
        with (out / "report.json").open("xb") as stream:
            stream.write(encoded(report)+b"\n")
        print(encoded({key: report.get(key) for key in ("status", "physical_candidate_status", "pytest_counts", "seconds", "source_count", "changed_python_paths")}).decode(), flush=True)
    if report["status"] != "PASS":
        raise RuntimeError("Validation refused; see " + str(out / "report.json"))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--control", required=True)
    parser.add_argument("tests", nargs="+")
    args = parser.parse_args()
    run(args.out, args.tests, args.control)
