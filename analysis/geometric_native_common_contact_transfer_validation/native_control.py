"""Declared reference-area depth density and complete serial spring carriers.

The new model is explicit.  The original continuous and atomic measures are
distinct, and no whole-face allocation or old world-force handover is inferred.
Only frozen-reference density is used here; it produces local integrands,
without assigning a finite area to a point or adopting an interval.
"""
from __future__ import annotations

from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import time

from analysis.geometric_native_common_contact_force_validation.native_control import (
    encoded, payload, probe_arguments, source_history_records, port_ingredients,
)

PHYSICAL_SCOPE = "COMMON_REFERENCE_AREA_SERIAL_CARRIER_COMPONENT_ONLY"
FALSE_SCOPE = (
    "new_mixed_force_equilibrium_certified", "whole_clock_material_admission_authorized",
    "physical_material_admission_authorized", "original_point_1_completed",
    "face_depth_memory_or_measure_transfer_certified", "old_rod_prestrain_or_energy_transferred",
    "old_node_states_assigned_to_new_cuts", "source_history_advanced",
    "hot_curve_contact_law_inferred", "finite_owner_whole_clock_motion_certified",
    "full_genuine_source_reconstructed", "inherited_force_certificate_for_placement",
    "contact_energy_adopted_on_new_face", "complete_three_dimensional_strain_energy_certified",
    "old_world_force_pullback_preserved", "all_original_source_measure_allocated_once",
    "source_atomic_and_continuous_measures_equated", "spatial_column_endpoints_authenticated",
)
PROVENANCE = "explicit post853 normalized reference-strip and reduced serial-carrier declaration"


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def reference_weighted_integrands(series, measure):
    """Reference density, independent of modal/owner increments, never point area."""
    weight = measure.reference_area_density_bounds_m2_per_parameter_per_physical_volume_depth_m
    return payload(dict(
        energy_bounds_j_per_parameter_per_physical_volume_depth_m=series.total_energy_bounds_j_m2*weight,
        residual_bounds_per_parameter_per_physical_volume_depth_m=tuple(
            x*weight for x in series.residual_bounds_per_reference_m2),
        hessian_bounds_per_parameter_per_physical_volume_depth_m=tuple(tuple(
            x*weight for x in row) for row in series.hessian_bounds_per_reference_m2),
        finite_point_area_assigned=False, globally_integrated_force_solution=False,
        reference_density_derivatives_with_respect_to_fresh_generalized_coordinates="zero by declared material reference",
    ))


def measure_ingredients(response):
    from tectonics.geometric_native_common_contact_cuts import _RUNTIME
    geometry = response.geometry
    owner = response.port.receiver_body.embedding_bank.owner
    return payload(dict(
        source_unit_world_direction_bounds=geometry.source_unit_world_direction_bounds,
        source_unit_world_direction_parameter_derivative_bounds=geometry.source_unit_world_direction_parameter_derivative_bounds,
        receiver_owner_local_unit_direction_bounds=geometry.receiver_owner_local_unit_direction_bounds,
        receiver_source_pose_intervals=_RUNTIME[response.port.cut.cut_snapshot_sha256]["poses"][owner],
        receiver_source_pose_owner=owner,
        common_physical_volume_depth_bounds_m=geometry.common_physical_volume_depth_bounds_m,
        physical_area_density_bounds=geometry.area_density_bounds_m2_per_parameter_per_material_m,
    ))


def rectangle_budgets(declaration):
    from tectonics.geometric_native_pose import FractionInterval as I
    from tectonics.geometric_native_common_contact_measure import source_reference_rectangle_area
    extent = declaration.source_computational_extent_radians
    queries = {
        "full_source_rectangle": (I(0, extent), I(0, 1)),
        "first_depth_third": (I(0, extent), I(0, F(1, 3))),
        "remaining_depth": (I(0, extent), I(F(1, 3), 1)),
        "interior_source_rectangle": (I(extent/4, 3*extent/4), I(F(1, 5), F(4, 5))),
    }
    budgets = {name: source_reference_rectangle_area(declaration,
        computational_angle_bounds=angle, normalized_depth_bounds=depth)
        for name, (angle, depth) in queries.items()}
    parameters = {name: dict(computational_angle_bounds=angle, normalized_depth_bounds=depth)
                  for name, (angle, depth) in queries.items()}
    return payload(budgets), payload(parameters)


def run(out):
    from tectonics.geometric_native_pose import FractionInterval as I
    from tectonics.geometric_native_normal_support import load_native_normal_support_source
    from tectonics.geometric_native_common_reference import bind_native_common_reference, native_common_reference_to_dict
    from tectonics.geometric_native_common_contact_cuts import bind_common_contact_cuts, common_contact_cuts_to_dict
    from tectonics.geometric_native_common_contact_ports import bind_native_common_contact_ports, evaluate_native_common_contact_port
    from tectonics.geometric_native_common_contact_state import (
        observe_native_common_contact_state, native_common_contact_state_observation_to_dict,
    )
    from tectonics import geometric_native_common_contact_measure as area_api
    from tectonics import geometric_native_common_contact_series as series_api
    target = Path(out)
    target.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    source = load_native_normal_support_source()
    reference = bind_native_common_reference(source=source, precision=96)
    cuts = bind_common_contact_cuts(reference, precision=96)
    ports = bind_native_common_contact_ports(reference, cuts=cuts, precision=96)
    cold = tuple(p for p in ports.ports if p.cut.receiving_trace_id is not None)
    hot = tuple(p for p in ports.ports if p.cut.receiving_trace_id is None)
    if len(cold) != 5 or len(hot) != 1:
        raise ValueError("Five genuine cold boundaries and the separate HOT curve are required")
    rows = []
    for port in cold:
        for sample in (1, 2):
            arguments = dict(probe_arguments(port.cut, sample), precision=96)
            profile = (F(1),) if sample == 1 else (F(0), F(2))
            cases = {}
            for label in ("zero", "nonzero"):
                actual = dict(arguments)
                if label == "zero":
                    actual.update(donor_coordinates_m=(F(0),)*7, receiver_coordinates_m=(F(0),)*7)
                response = evaluate_native_common_contact_port(reference, port, **actual)
                state = observe_native_common_contact_state(reference, response, precision=96)
                measure_declaration = area_api.declare_native_common_contact_reference_measure(
                    reference, response, state, provenance=PROVENANCE, depth_profile_coefficients=profile)
                measure = area_api.evaluate_native_common_contact_reference_measure(
                    reference, response, state, measure_declaration, precision=96)
                budgets, budget_parameters = rectangle_budgets(measure_declaration)
                carrier = series_api.declare_native_common_contact_series_carrier(
                    reference, response, state, provenance=PROVENANCE,
                    anchor_interpretation=series_api.ANCHOR_INTERPRETATION)
                series = {}
                for branch in ("relaxed_source", "frozen_memory_predictor"):
                    value = series_api.evaluate_native_common_contact_series_carrier(
                        reference, response, state, carrier, branch=branch, precision=96)
                    series[branch] = dict(
                        response=series_api.native_common_contact_series_response_to_dict(value),
                        reference_weighted_integrands=reference_weighted_integrands(value, measure))
                cases[label] = dict(arguments=payload(actual), port_ingredients=port_ingredients(response),
                    source_observation=native_common_contact_state_observation_to_dict(state),
                    measure_declaration=area_api.native_common_contact_reference_measure_declaration_to_dict(measure_declaration),
                    measure_response=area_api.native_common_contact_reference_measure_response_to_dict(measure),
                    measure_ingredients=measure_ingredients(response),
                    source_rectangle_area_budgets=budgets,
                    source_rectangle_budget_parameters=budget_parameters,
                    source_budget_allocated_in_full_to_this_cut=False,
                    carrier_declaration=series_api.native_common_contact_series_carrier_declaration_to_dict(carrier),
                    series_responses=series)
            rows.append(dict(cut_id=port.cut.cut_id, sample=sample, cases=cases,
                             depth_profile_is_explicitly_declared_not_inferred=True))
        print("DECLARED reference-area and serial carrier", port.cut.cut_id, "PASS", flush=True)
    hot_args = dict(probe_arguments(hot[0].cut, 1), precision=96)
    hot_response = evaluate_native_common_contact_port(reference, hot[0], **hot_args)
    try:
        observe_native_common_contact_state(reference, hot_response, precision=96)
    except ValueError as error:
        hot_rejection = str(error)
    else:
        raise ValueError("The HOT curve inherited an unsupported old receiving law")
    summary = dict(physical_candidate_status=PHYSICAL_SCOPE, actual_cold_contact_cut_count=5,
        retained_true_hot_curve_count=1, current_state_query_case_count=20,
        reference_area_density_case_count=20, serial_force_density_case_count=40,
        reference_weighted_integrand_case_count=40, local_coordinate_count=20,
        exact_symmetric_hessian_entry_replay_count=16000,
        declared_normalized_depth_profile_count=2,
        complete_sixteen_source_states_areas_work_and_clocks_preserved=True,
        variable_contact_subrod_overrod_energies_included=True,
        complete_series_compliance_not_parallel_energy_used=True,
        source_continuous_reference_measure_kept_separate_from_sixteen_atom_weights=True,
        accepted_original_native_intervals=1, accepted_new_native_intervals=0,
        original_point_1_remaining_work_blocks=1,
        remaining_physical_conditions=[
            "Whole-source once-only face material/measure allocation and old-force/prestress handover, HOT law/mean chart, loads/drag and NEW mixed equilibrium",
            "Whole-clock all-pair admission including reverse C5 to C1",
            "Accepted continuation with independent path work, balances and joint restart"],
        **{name: False for name in FALSE_SCOPE})
    packet = dict(format="native-common-reference-area-serial-carrier-source-packet-1",
        prior_853_wrapper_raw_sha256="1d7a6ac8113a408aab6764f1c4a9fccb09f8d48184f5118434fc685574504313",
        complete_source_dto_json=[list(p) for p in source.complete_source_dto_json],
        saved_joint_reference_field=json.loads(source.saved_field_json),
        source_snapshot_sha256=source.source_snapshot_sha256,
        common_reference=native_common_reference_to_dict(reference), cut_set=common_contact_cuts_to_dict(cuts),
        contact_query_rows=rows, hot_source_law_rejection=hot_rejection,
        **source_history_records(source), **summary)
    packet["packet_sha256"] = digest(packet)
    (target / "source_packet.json").write_bytes(encoded(packet)+b"\n")
    report = dict(format="native-common-reference-area-serial-carrier-control-1", status="PASS",
        source_packet_sha256=packet["packet_sha256"], contact_query_rows_sha256=digest(rows),
        source_snapshot_sha256=source.source_snapshot_sha256, seconds=time.perf_counter()-started, **summary)
    report["report_sha256"] = digest(report)
    (target / "report.json").write_bytes(encoded(report)+b"\n")
    print("DECLARED area/series local integrands PASS; full-source handover and equilibrium unadopted", flush=True)
    return report
