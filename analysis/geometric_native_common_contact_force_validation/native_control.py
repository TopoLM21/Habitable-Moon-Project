"""Current-state objective contact forces, without new face-area allocation.

Complete source and common-reference data are stored once in the packet.
Compact body ingredients bind each newly replayed physical query; old volume
integrals are not re-executed. Neither old rods nor HOT laws are inferred.
"""
from __future__ import annotations

from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import time

from analysis.geometric_native_common_contact_validation.native_control import (
    encoded, payload, probe_arguments, source_history_records,
)

PHYSICAL_SCOPE = "COMMON_CONTACT_REST_LAW_COMPONENT_ONLY"
FALSE_SCOPE = (
    "new_mixed_force_equilibrium_certified", "whole_clock_material_admission_authorized",
    "physical_material_admission_authorized", "original_point_1_completed",
    "face_depth_memory_or_measure_transfer_certified", "old_rod_prestrain_or_energy_transferred",
    "old_node_states_assigned_to_new_cuts", "source_history_advanced",
    "hot_curve_contact_law_inferred", "finite_owner_whole_clock_motion_certified",
    "full_genuine_source_reconstructed", "inherited_force_certificate_for_placement",
    "contact_energy_adopted_on_new_face", "complete_three_dimensional_strain_energy_certified",
)


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def port_ingredients(response):
    """Full exact local algebra inputs, with source replay hashes and labels."""
    return payload(dict(
        port_response_binding_sha256=response.binding_sha256,
        port_binding_sha256=response.port.binding_sha256,
        reference_snapshot_sha256=response.port.reference_snapshot_sha256,
        cut_snapshot_sha256=response.port.cut.cut_snapshot_sha256,
        geometry_bounds_sha256=response.geometry.bounds_snapshot_sha256,
        donor_component_id=response.port.cut.donor_component_id,
        receiver_component_id=response.port.cut.receiver_component_id,
        donor_owner=response.port.donor_body.original_body.owner,
        receiver_owner=response.port.receiver_body.original_body.owner,
        owner_rotation_coordinate_order=response.port.owner_rotation_coordinate_order,
        mode_coordinate_order=response.mode_coordinate_order,
        parameter_bounds=response.geometry.parameter_bounds,
        physical_radius_cubed_bounds_m3=response.geometry.common_physical_radius_cubed_bounds_m3,
        receiving_computational_angle_bounds_radians=response.geometry.receiving_computational_angle_bounds_radians,
        receiving_native_angle_bounds_radians=response.geometry.receiving_native_angle_bounds_radians,
        donor_original_unit_direction_bounds=response.geometry.donor_original_unit_direction_bounds,
        receiver_original_unit_direction_bounds=response.geometry.receiver_original_unit_direction_bounds,
        world_position_bounds_m=response.geometry.world_position_bounds_m,
        reference_director_bounds=(response.geometry.receiver_inward_unit_normal_bounds,
            response.geometry.inward_unit_down_direction_bounds,response.geometry.angular_unit_tangent_bounds),
        donor_coordinates_m=response.donor_fields.coordinates_m,
        receiver_coordinates_m=response.receiver_fields.coordinates_m,
        donor_displacement_increment_bounds_m=response.donor_fields.displacement_increment_bounds_m,
        receiver_displacement_increment_bounds_m=response.receiver_fields.displacement_increment_bounds_m,
        donor_displaced_world_position_bounds_m=response.donor_fields.displaced_point_bounds_m,
        receiver_displaced_world_position_bounds_m=response.receiver_fields.displaced_point_bounds_m,
        displacement_jump_bounds_m=response.displacement_jump_bounds_m,
        mode_jump_columns=response.mode_jump_columns,
        owner_rotation_jump_columns_m=response.owner_rotation_jump_columns_m,
        owner_body_mode_cross_derivative_columns=response.owner_body_mode_cross_derivative_columns,
    ))


def run(out):
    from tectonics.geometric_native_pose import FractionInterval as I
    from tectonics.geometric_native_normal_support import load_native_normal_support_source
    from tectonics.geometric_native_common_reference import bind_native_common_reference, native_common_reference_to_dict
    from tectonics.geometric_native_common_contact_cuts import bind_common_contact_cuts, common_contact_cuts_to_dict
    from tectonics.geometric_native_common_contact_ports import bind_native_common_contact_ports, evaluate_native_common_contact_port
    from tectonics.geometric_native_common_contact_state import (
        observe_native_common_contact_state, native_common_contact_state_observation_to_dict,
    )
    from tectonics.geometric_native_common_contact_constitutive import (
        BRANCHES, evaluate_native_common_contact_constitutive, native_common_contact_constitutive_response_to_dict,
        evaluate_native_common_contact_scalar_kinematics, native_common_contact_scalar_kinematics_to_dict,
    )
    target = Path(out)
    target.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    source = load_native_normal_support_source()
    reference = bind_native_common_reference(source=source, precision=160)
    cuts = bind_common_contact_cuts(reference, precision=160)
    ports = bind_native_common_contact_ports(reference, cuts=cuts, precision=160)
    cold = [p for p in ports.ports if p.cut.receiving_trace_id is not None]
    hot = [p for p in ports.ports if p.cut.receiving_trace_id is None]
    if len(cold) != 5 or len(hot) != 1:
        raise ValueError("The complete genuine boundary set must retain five cold and one separate HOT cut")
    rows = []
    for port in cold:
        for sample in (1, 2):
            arguments = probe_arguments(port.cut, sample)
            cases = {}
            for name in ("zero", "nonzero"):
                actual = dict(arguments)
                if name == "zero":
                    actual.update(donor_coordinates_m=(F(0),)*7, receiver_coordinates_m=(F(0),)*7)
                response = evaluate_native_common_contact_port(reference, port, **actual)
                state = observe_native_common_contact_state(reference, response, precision=160)
                forces = {branch: native_common_contact_constitutive_response_to_dict(
                    evaluate_native_common_contact_constitutive(reference, response, state, branch=branch, precision=160))
                    for branch in BRANCHES}
                if name == "zero":
                    for force in forces.values():
                        if force["traction_bounds_pa"][1] != payload(I(0)):
                            raise ValueError("A zero reference increment gained artificial shear")
                cases[name] = dict(arguments=payload(actual), port_ingredients=port_ingredients(response),
                    source_observation=native_common_contact_state_observation_to_dict(state), responses=forces)
            rows.append(dict(cut_id=port.cut.cut_id, sample=sample, cases=cases,
                             point_force_density_is_a_new_mixed_solution=False))
        print("CURRENT memory/objective contact-only local force",port.cut.cut_id,"PASS",flush=True)
    hot_args = probe_arguments(hot[0].cut, 1)
    hot_response = evaluate_native_common_contact_port(reference, hot[0], **hot_args)
    try:
        observe_native_common_contact_state(reference, hot_response, precision=160)
    except ValueError as error:
        hot_rejection = str(error)
    else:
        raise ValueError("The genuine HOT curve inherited an unsupported source contact law")
    hot_probe = dict(arguments=payload(hot_args), port_ingredients=port_ingredients(hot_response),
        scalar_kinematics=native_common_contact_scalar_kinematics_to_dict(
            evaluate_native_common_contact_scalar_kinematics(reference, hot_response, precision=160)),
        source_law_rejection=hot_rejection, hot_contact_force_or_memory_assigned=False)
    summary = dict(physical_candidate_status=PHYSICAL_SCOPE, actual_cold_contact_cut_count=5,
        retained_true_hot_curve_count=1, current_state_query_case_count=20,
        local_force_density_case_count=40, local_coordinate_count=20,
        exact_symmetric_hessian_entry_replay_count=16000,
        complete_sixteen_source_states_areas_work_and_clocks_preserved=True,
        literal_current_memory_not_previous_dense_substitution=True,
        source_contact_only_per_original_reference_area=True,
        actual_friction_free_relaxed_shear_zero=True,
        receiver_material_director_full_derivatives_included=True,
        accepted_original_native_intervals=1, accepted_new_native_intervals=0,
        original_point_1_remaining_work_blocks=1,
        remaining_physical_conditions=[
            "Actual face memory/measure and old rod prestress transfer, HOT law/mean chart, source loads/drag and NEW mixed equilibrium",
            "Whole-clock all-pair admission including reverse C5 to C1",
            "Accepted continuation with independent path work, balances and joint restart"],
        **{name:False for name in FALSE_SCOPE})
    packet = dict(format="native-common-objective-contact-rest-force-source-packet-1",
        authenticated_prior_contact_wrapper_raw_sha256="e8ffabb1a29c1fb021e94fe53e815b6ad8d16ea14e9103a30b9f7428ac9ab847",
        authenticated_prior_contact_source_packet_raw_sha256="8fa086a0d42b1cb2b81c30d01c61369d13e294d4edf6c10f7d3b8a9a3dc99308",
        complete_source_dto_json=[list(p) for p in source.complete_source_dto_json],
        saved_joint_reference_field=json.loads(source.saved_field_json),
        source_snapshot_sha256=source.source_snapshot_sha256,
        common_reference=native_common_reference_to_dict(reference), cut_set=common_contact_cuts_to_dict(cuts),
        contact_query_rows=rows, hot_geometry_only_probe=hot_probe,
        **source_history_records(source), **summary)
    packet["packet_sha256"] = digest(packet)
    with (target/"source_packet.json").open("xb") as stream:
        stream.write(encoded(packet)+b"\n")
    report = dict(format="native-common-objective-contact-rest-force-control-1",status="PASS",
        source_packet_sha256=packet["packet_sha256"],contact_query_rows_sha256=digest(rows),
        source_snapshot_sha256=source.source_snapshot_sha256,seconds=time.perf_counter()-started,**summary)
    report["report_sha256"] = digest(report)
    with (target/"report.json").open("xb") as stream:
        stream.write(encoded(report)+b"\n")
    print("CURRENT contact-only local force component PASS; no face measure/rod transfer or new accepted interval",flush=True)
    return report
