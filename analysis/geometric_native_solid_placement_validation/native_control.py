"""Actual-source linkage of layer dimensions, series closure and finite forces.

This controller does not substitute a local prism for an implicit spherical
material mask.  Missing contact-zone geometry and incomplete depth coverage
remain physical refusals, even if a conservative force-error bound passes.
"""
from __future__ import annotations

from decimal import Decimal
from fractions import Fraction as F
from pathlib import Path
import hashlib
import json
import math
import time

from tectonics.geometric_native_pose import FractionInterval as I, _rounded_interval

ROOT = Path(__file__).resolve().parents[2]


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def bound_payload(value):
    """Export outward bounds; never change the process integer-digit limit."""
    if type(value) is F:
        value = I(value)
    bounded = _rounded_interval(value, 96)
    return {"lo": [bounded.lo.numerator, bounded.lo.denominator],
            "hi": [bounded.hi.numerator, bounded.hi.denominator]}


def write(path, value):
    with Path(path).open("xb") as stream:
        stream.write(encoded(value)+b"\n")


def run_control(out):
    from analysis.geometric_native_birth_front_validation.native_continuum_case import (
        build_native_continuum_case, solve_native_continuum_case, export_native_continuum_case)
    from tectonics.geometric_native_birth_continuum_motion import native_birth_continuum_frame
    from tectonics.geometric_native_birth_finite_pose import bound_born_finite_pose_fields
    from tectonics.geometric_native_birth_finite_force_bounds import (
        bound_born_finite_pose_jacobian, bound_born_finite_pose_force_error)
    from tectonics.geometric_native_solid_layer_placement import native_component_solid_thickness_m
    from tectonics.geometric_solid_contact_closure import (
        compute_series_normal_closure, geometry_to_anchor, solve_solid_tangential_closure)
    from tectonics.geometric_spatial_underthrust import _angles, _assess_interval, trig_polynomial_extrema
    from tectonics.geometric_native_physical_history import _typed_fingerprint
    from tectonics.geometric_native_birth_curved_pair_support import certify_native_cold5_receiver1_immediate_slice
    from tectonics.thermal import SECONDS_PER_MYR

    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    stages = {}
    def timed(name, callback):
        tick = time.perf_counter()
        print("SOLID control", name, "starting", flush=True)
        result = callback()
        stages[name] = time.perf_counter()-tick
        print("SOLID control", name, "finished", round(stages[name], 3), flush=True)
        return result

    case = timed("actual_source", build_native_continuum_case)
    before, support, inputs = case["previous"], case["support"], case["inputs"]
    before_fingerprint = _typed_fingerprint(before)
    candidate = timed("fresh_force", lambda: solve_native_continuum_case(case,
        provenance="Actual source finite-thickness and finite-SO3 force linkage control"))
    packet = timed("complete_force_replay", lambda: export_native_continuum_case(case, candidate))
    prior_path = ROOT/"analysis/geometric_native_birth_front_validation/control_release_01/candidate_packet.json"
    prior = json.loads(prior_path.read_bytes())
    physical = lambda raw: {name: value for name, value in raw.items()
                           if name not in ("candidate_sha256", "provenance")}
    if encoded(physical(prior)) != encoded(physical(packet)):
        raise ValueError("The genuine complete physical source/force differs from the preserved all-pair control")
    write(out/"candidate_packet.json", packet)

    thicknesses = {row.component_id: native_component_solid_thickness_m(support.source_epoch, row.component_id)
                   for row in support.source_epoch.components}
    face_coverage = []
    for carrier in support.interface_carriers:
        height = min(thicknesses[carrier.donor_component.component_id],
                     thicknesses[carrier.receiver_component.component_id])
        force_height = F.from_float(carrier.declaration.vertical_contact_height_m)
        face_coverage.append(dict(family_id=carrier.declaration.family_id,
            donor_solid_thickness_m=bound_payload(thicknesses[carrier.donor_component.component_id]),
            receiver_solid_thickness_m=bound_payload(thicknesses[carrier.receiver_component.component_id]),
            common_face_height_in_new_dipping_prism_model_m=bound_payload(height),
            existing_declared_force_height_m=bound_payload(force_height),
            whole_new_prism_face_force_coverage=force_height >= height,
            actual_spherical_face_embedding_authorized=False))
    if len(face_coverage) != 2 or any(row["whole_new_prism_face_force_coverage"] for row in face_coverage):
        raise ValueError("The actual full-face limitation differs from the genuine source")

    trace_for_point = {identity: trace for identity, trace, _ in _angles(before.initial_mechanics)}
    local_rows = []
    for identity, state in candidate.original_point_states:
        trace = trace_for_point[identity]
        normal = compute_series_normal_closure(columns=trace.columns, law=trace.law,
                                               free_gap_m=F.from_float(state.gap_free_m))
        p = -F.from_float(state.traction_normal_pa)
        check = geometry_to_anchor(normal=normal, physical_face_void_m=F(0),
            sub_column_shortening_m=p*F.from_float(trace.columns.sub_length_m)/F.from_float(trace.columns.sub_normal_modulus_pa),
            over_column_shortening_m=p*F.from_float(trace.columns.over_length_m)/F.from_float(trace.columns.over_normal_modulus_pa),
            fault_indentation_m=-F.from_float(state.gap_m), tolerance_m=F(1, 10**10))
        tangent = solve_solid_tangential_closure(columns=trace.columns, law=trace.law,
            free_jump_m=F.from_float(state.jump_free_m), plastic_slip_m=F.from_float(state.plastic_slip_m))
        if normal.physical_geometry_authorized or normal.contact_zone_geometry_declared:
            raise ValueError("Series mechanics invented an independent physical contact-zone geometry")
        local_rows.append(dict(point_id=identity, trace_id=trace.trace_id,
            free_gap_m=bound_payload(F.from_float(state.gap_free_m)),
            pressure_pa=bound_payload(normal.normal_pressure_pa),
            column_and_contact_shortening_m=bound_payload(normal.sub_column_shortening_m+
                normal.over_column_shortening_m+normal.fault_indentation_m),
            geometry_anchor_check=all(abs(value) <= check.tolerance_m for value in (
                check.anchor_residual_m, check.sub_shortening_residual_m,
                check.over_shortening_residual_m, check.fault_indentation_residual_m,
                check.physical_void_residual_m)),
            energy_density_j_m2=bound_payload(normal.total_energy_j_m2),
            tangential_elastic_traction_pa=bound_payload(tangent.shear_traction_pa),
            independent_contact_zone_geometry_declared=False))
    if len(local_rows) != 16 or not all(row["geometry_anchor_check"] for row in local_rows):
        raise ValueError("The original source elastic face decomposition failed")

    cold_binding = next(row for row in before.history.material_state.trace_bindings
                        if row.donor_fragment_id == "octant:5" and row.receiver_fragment_id == "octant:1")
    cold_certificate = next(row for row in candidate.spatial_step.certificates
                            if row.trace_id == cold_binding.trace_id)
    minimum_updip = min(trig_polynomial_extrema(row.plastic_increment,
        row.low_radians, row.high_radians)[0]-row.plastic_increment.rounding_bound
        for row in cold_certificate.pieces)
    if minimum_updip <= .003 or any(row.branch != "up_dip_plastic" for row in cold_certificate.pieces):
        raise ValueError("The actual cold reverse-sliding case differs from the preserved source")
    if _assess_interval(cold_certificate, 0., cold_certificate.extent_radians) != "no_down_dip_plastic_sliding":
        raise ValueError("The existing material gate was weakened or the physical case changed")
    immediate = timed("fresh_cold_immediate_entry", lambda: certify_native_cold5_receiver1_immediate_slice(
        before.current_surface, candidate.omega_rad_per_myr, candidate.dt_s/SECONDS_PER_MYR,
        precision=256, provenance="Actual finite-thickness closure: source-bound cold receiver identity"))
    write(out/"cold5_receiver1_immediate_slice.json", immediate)
    cold_before = next(row for row in before.transverse_displacement_m if row[0] == cold_binding.trace_id)
    cold_after = next(row for row in candidate.transverse_displacement_m if row[0] == cold_binding.trace_id)
    cold_relaxation = dict(trace_id=cold_binding.trace_id, donor_parent_id="octant:5", receiver_parent_id="octant:1",
        actual_positive_incoming_slice_proof_sha256=immediate["proof_sha256"],
        whole_trace_updip_increment_lower_m=minimum_updip,
        bank_radial_separation_before_m=cold_before[2]-cold_before[1],
        bank_radial_separation_after_m=cold_after[2]-cold_after[1],
        previous_down_dip_gate_result="no_down_dip_plastic_sliding",
        reverse_sliding_law_or_history_changed=False,
        physical_solid_first_entry_authorized=False)

    radius = F.from_float(inputs["radius_m"])
    after = tuple(F(Decimal(value)) for value in candidate.network_step.coordinates_decimal)
    prior_w = {}
    radial_before = {identity: (sub, over) for identity, sub, over in before.transverse_displacement_m}
    for index, trace in enumerate(before.initial_mechanics.fault_traces.traces):
        for bank, value in enumerate(radial_before[trace.trace_id]):
            prior_w[inputs["rotation_dof_count"]+2*index+bank] = F.from_float(value)
    for foundation in inputs["born_foundations"]:
        prior_w[foundation.radial_dof_index] = F.from_float(foundation.previous_state.radial_displacement_m)
    clock = I(1)
    dt_myr = F.from_float(candidate.dt_s/SECONDS_PER_MYR)
    axes = {owner: tuple(F.from_float(value)*dt_myr for value in row)
            for owner, row in candidate.omega_rad_per_myr}
    angular_indices = {owner: tuple(range(3*index, 3*index+3)) for index, owner in enumerate(before.plate_ids)}
    force_rows, error_bounds = [], []
    for carrier, attachment, contact_step in zip(support.interface_carriers,
            inputs["born_continuum_attachments"], candidate.network_step.continuum_contact_steps):
        declaration = attachment.declaration
        constant = tuple(Decimal(value) for value in declaration.normal_operator_coefficients[0])
        sub = next(index for index, value in enumerate(constant) if value < 0)
        over = next(index for index, value in enumerate(constant) if value > 0)
        a, b, h = native_birth_continuum_frame(carrier, precision=160)
        sine, cosine = (F.from_float(function(carrier.declaration.dip_radians))
                        for function in (math.sin, math.cos))
        shared = dict(start_unit_bounds=a, tangent_unit_bounds=b, horizontal_unit_bounds=h,
            donor_axis=axes[carrier.donor_component.owner], receiver_axis=axes[carrier.receiver_component.owner],
            radius_m=radius, donor_radial_before_m=prior_w[sub],
            donor_radial_increment_m=after[sub]*radius-prior_w[sub],
            sine_dip=sine, cosine_dip=cosine, time_fraction_bounds=clock, precision=160, order=28)
        fields = timed(declaration.reference.family_id+"_finite_fields", lambda: bound_born_finite_pose_fields(
            **shared, receiver_radial_before_m=prior_w[over],
            receiver_radial_increment_m=after[over]*radius-prior_w[over],
            birth_gap_free_m=F.from_float(declaration.reference.birth_gap_free_m),
            birth_jump_free_m=F.from_float(declaration.reference.birth_jump_free_m)))
        angle = I(F(Decimal(declaration.angle_start_decimal)),
                  max(F(Decimal(declaration.angle_end_decimal)), carrier.native_angle_bounds_radians.hi))
        jacobian = timed(declaration.reference.family_id+"_finite_jacobian", lambda: bound_born_finite_pose_jacobian(
            declaration, **shared, donor_angular_indices=angular_indices[carrier.donor_component.owner],
            receiver_angular_indices=angular_indices[carrier.receiver_component.owner],
            donor_radial_index=sub, receiver_radial_index=over, angle_bounds=angle))
        bounds = timed(declaration.reference.family_id+"_finite_force_bounds", lambda: bound_born_finite_pose_force_error(
            fields, attachment, contact_step, jacobian,
            finite_measure_density_m2_per_radian=carrier.inclined_measure_density_m2_per_radian,
            finite_angle_end_bounds=carrier.native_angle_bounds_radians))
        error_bounds.append(bounds)
        force_rows.append(dict(family_id=declaration.reference.family_id,
            endpoint_finite_gap_bounds_m=bound_payload(bounds.finite_gap_bounds_m),
            endpoint_gap_error_bound_m=bound_payload(bounds.gap_error_bound_m),
            endpoint_reaction_error_bounds=[bound_payload(value) for value in bounds.reaction_error_bounds],
            endpoint_energy_error_bound_j=bound_payload(bounds.energy_error_bound_j),
            normal_jacobian_error_bounds=[bound_payload(value) for value in jacobian.operator_error_bounds],
            native_measure_alias_bound_m2=bound_payload(bounds.measure_alias_bound_m2),
            original_active_set_assumed=False, material_admission_authorized=False))
    # Report raw errors and their ratios to actual source torque.  There is
    # no invented acceptance budget and no assertion about old finite forces.
    combined = tuple(sum((row.reaction_error_bounds[index] for row in error_bounds), F(0))
                     for index in range(len(after)))
    source_scale = sum((abs(F.from_float(value)) for value in inputs["source_torque_nm"]), F(0))
    if source_scale <= 0:
        raise ValueError("The genuine source has no nonzero external torque")
    # Every normalized coordinate has its OWN physical balancing operands.
    # A small ratio to total mantle torque can conceal a poorly bounded HOT
    # radial balance. Keep all original forces affine for this diagnostic;
    # it is never a complete finite-SO3 network certificate.
    dt = F.from_float(candidate.dt_s)
    before_q = tuple(F.from_float(value) for value in inputs["coordinates_before"])
    damping = tuple(sum((F.from_float(value)*(after[j]-before_q[j])/dt
                        for j, value in enumerate(row)), F(0)) for row in inputs["drag_nm_s"])
    elastic = [sum((F.from_float(float(value))*after[j] for j, value in enumerate(row)), F(0))
               for row in inputs["old_stiffness"]]+[F(0)]*len(inputs["born_foundations"])
    for foundation, step in zip(inputs["born_foundations"], candidate.network_step.born_foundation_steps):
        elastic[foundation.radial_dof_index] = F.from_float(step.energy_gradient_n)*radius
    old_forces = []
    for area, step, operators in zip(inputs["old_areas_m2"], candidate.network_step.original_point_steps,
                                      inputs["old_anchor_operators"]):
        old_forces.append(tuple(F.from_float(area)*sum((F.from_float(step.traction_pa[mode])*
            F.from_float(operators[mode][index]) for mode in range(2)), F(0))
            for index in range(len(inputs["old_stiffness"])))+(F(0),)*len(inputs["born_foundations"]))
    affine_born = [tuple(F(Decimal(value)) for value in step.gradient_decimal)
                   for step in candidate.network_step.continuum_contact_steps]
    all_contact_forces = old_forces+affine_born
    coordinate_rows = []
    for index, error in enumerate(combined):
        operands = (damping[index], elastic[index], F.from_float(inputs["source_torque_nm"][index]),
                    *(row[index] for row in all_contact_forces))
        scale = sum(map(abs, operands), F(0))
        residual = abs(damping[index]+elastic[index]-operands[2]+
                       sum((row[index] for row in all_contact_forces), F(0)))
        if scale == 0:
            if residual or error:
                raise ValueError("A finite force correction has no physical coordinate balance scale")
            relative = F(0)
        else:
            if residual/scale > F(5, 10**10):
                raise ValueError("Exact reconstruction of the actual affine coordinate balance failed")
            relative = (residual+error)/scale
        coordinate_rows.append(dict(coordinate_index=index, actual_affine_balance_scale=bound_payload(scale),
            absolute_affine_residual=bound_payload(residual),
            born_finite_pose_correction_relative_bound=bound_payload(relative),
            existing_coordinate_tolerance=bound_payload(F(5, 10**10)),
            born_correction_fits_existing_affine_other_force_budget=relative <= F(5, 10**10)))
    if _typed_fingerprint(before) != before_fingerprint:
        raise ValueError("The accepted original state changed during the new controls")
    body = dict(format="native-solid-layer-source-and-force-control-1", status="PASS",
        source_epoch_sha256=support.source_epoch.epoch_sha256,
        candidate_sha256=packet["candidate_sha256"],
        previous_complete_physical_candidate_identical=True,
        prior_candidate_raw_sha256=hashlib.sha256(prior_path.read_bytes()).hexdigest(),
        actual_source_solid_component_count=len(thicknesses), original_point_closures=local_rows,
        actual_cold_bending_relaxation=cold_relaxation,
        accepted_input_complete_typed_state_unchanged=True,
        physical_face_coverage=face_coverage, born_endpoint_finite_force_bounds=force_rows,
        combined_born_endpoint_reaction_error_bounds=[bound_payload(value) for value in combined],
        maximum_born_error_relative_to_total_external_torque=bound_payload(max(combined)/source_scale),
        born_endpoint_potential_error_bound_j=bound_payload(sum((row.energy_error_bound_j for row in error_bounds), F(0))),
        per_coordinate_born_force_correction_diagnostic=coordinate_rows,
        original_affine_other_forces_with_finite_born_correction_budget_passed=all(
            row["born_correction_fits_existing_affine_other_force_budget"] for row in coordinate_rows),
        full_finite_pose_network_equilibrium_certified=False,
        independent_contact_zone_geometry_available=False,
        global_spherical_solid_embedding_certified=False,
        whole_clock_material_admission_authorized=False,
        accepted_original_native_intervals=1, accepted_new_native_intervals=0,
        original_point_1_completed=False, original_point_1_remaining_work_blocks=1,
        stage_seconds=stages, seconds=time.perf_counter()-start,
        remaining_physical_conditions=[
            "source-bound whole-mask spherical layer/edge embedding and preload/column/contact-zone placement",
            "complete finite-face mechanical coverage with once-paid material and preparation ownership",
            "whole-clock all-pair physical admission, then adopted mixed continuation and joint restart"])
    body["report_sha256"] = hashlib.sha256(encoded(body)).hexdigest()
    write(out/"report.json", body)
    print("SOLID native control PASS", body["seconds"], flush=True)
    return body
