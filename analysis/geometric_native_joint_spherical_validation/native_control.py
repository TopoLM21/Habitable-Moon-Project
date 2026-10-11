"""Actual shared reference faces and variational curved contact diagnostics.

The saved motion is independently authenticated against the genuine source.
It is a probe for the new curved law, not its force-equilibrium certificate.
"""
from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import time

from analysis.geometric_native_finite_equilibrium_validation.native_control import encoded, outward, payload, write
from tectonics.geometric_native_pose import FractionInterval as I

ROOT = Path(__file__).resolve().parents[2]
PRIOR = ROOT/"analysis/geometric_native_spherical_solid_validation/release_validation_01/native_control/candidate_packet.json"
PRIOR_RAW = "1518f46d53a18d78d9afc9f93b0297f8cf446026b424b0e5cd54a134bae038e2"
FULL_PROVENANCE = "Spherical source-layer full material-depth first-birth reference declaration; unaccepted trial only"


def _point(value):
    low, high = F(*value["lo"]), F(*value["hi"])
    if low != high:
        raise ValueError("A literal source motion must not be replaced by an interval midpoint")
    return low


def prepare_source_context(case):
    """Bind previously proved trial coordinates to the current genuine source."""
    from tectonics.geometric_native_solid_face_closure import declare_native_full_face_first_birth_support
    from tectonics.geometric_native_birth_continuum_motion import build_native_birth_continuum_inputs
    from tectonics.geometric_native_birth_exact_radial import normalize_native_birth_radial_before
    from tectonics.geometric_native_birth_contact_support import native_birth_contact_support_to_dict
    from tectonics.geometric_native_implicit_underthrust_motion import native_implicit_underthrust_motion_to_dict
    from tectonics.geometric_native_implicit_mantle_loading import native_implicit_spatial_mantle_load_to_dict
    from tectonics.thermal import SECONDS_PER_MYR
    raw = PRIOR.read_bytes()
    if hashlib.sha256(raw).hexdigest() != PRIOR_RAW:
        raise ValueError("The externally authenticated original trial packet changed")
    packet = json.loads(raw)
    seal = packet.pop("packet_sha256")
    if hashlib.sha256(encoded(packet)).hexdigest() != seal:
        raise ValueError("The preserved source-motion packet seal differs")
    support, branch = declare_native_full_face_first_birth_support(case["previous"],
        case["support"], provenance=FULL_PROVENANCE)
    source = dict(before=native_implicit_underthrust_motion_to_dict(case["previous"]),
        current_load=native_implicit_spatial_mantle_load_to_dict(case["mesh"], case["previous"].current_surface, case["load"]),
        original_unaccepted_trial_birth_support=native_birth_contact_support_to_dict(case["support"]),
        full_reference_face_birth_support=native_birth_contact_support_to_dict(support))
    if any(encoded(value) != encoded(packet[name]) for name, value in source.items()):
        raise ValueError("The complete genuine before, current load or support differs from the saved motion authority")
    inputs, frames = build_native_birth_continuum_inputs(case["mesh"], case["previous"],
        case["load"], support, case["dt_s"], decimal_precision=160)
    canonical, normalization = normalize_native_birth_radial_before(inputs, case["previous"])
    trial = packet["full_reference_face_force_trial"]
    coordinates = tuple(F(Decimal(value)) for value in trial["coordinates_decimal"])
    if len(coordinates) != 19:
        raise ValueError("The preserved nineteen-coordinate trial changed")
    axes = {owner: tuple(_point(value) for value in axis) for owner, axis in trial["literal_owner_axes"]}
    dt = F.from_float(case["dt_s"]/SECONDS_PER_MYR)
    used = {owner: tuple(F.from_float(value)*dt for value in omega)
        for owner, omega in trial["omega_rad_per_myr"]}
    if axes != used:
        raise ValueError("The shared curved-face probe changed the saved literal owner motion")
    radius = F.from_float(inputs["radius_m"])
    before = {row.coordinate_index: row.physical_before_m for row in normalization.coordinates}
    after = {index: coordinates[index]*radius for index in before}
    bindings = {row.trace_id: row for row in support.source_epoch.source_material_state.trace_bindings}
    indices, selections = {}, {}
    for trace_index, trace in enumerate(case["previous"].initial_mechanics.fault_traces.traces):
        binding = bindings[trace.trace_id]
        for bank, parent, index in (("donor", binding.donor_fragment_id, inputs["rotation_dof_count"]+2*trace_index),
                ("receiver", binding.receiver_fragment_id, inputs["rotation_dof_count"]+2*trace_index+1)):
            selections.setdefault(parent, []).append((trace.trace_id, bank, index))
    foundations = {row.reference.patch_id: row.radial_dof_index for row in inputs["born_foundations"]}
    bank_sources = {}
    for component in support.source_epoch.components:
        identity = component.component_id
        if component.material_kind == "hot_birth":
            indices[identity] = foundations[identity]
            bank_sources[identity] = (identity, "whole_hot_foundation")
        else:
            matches = tuple(sorted(selections.get(component.original_parent_id, ())))
            if not matches:
                raise ValueError("An original component has no actual saved bank for its explicit uniform roof declaration")
            identity_source, bank, index = matches[0]
            indices[identity] = index
            bank_sources[identity] = (identity_source, bank)
    if len(indices) != 13 or len(before) != 13:
        raise ValueError("The genuine thirteen-component radial binding changed")
    return dict(full_case=dict(case, support=support, inputs=inputs, frames=frames),
        branch=branch, source=source, before=before, after=after, indices=indices,
        axes=axes, bank_sources=bank_sources, coordinates=coordinates,
        source_motion=trial, source_packet_seal=seal, normalization=normalization)


def audit_joint_geometry(context):
    from tectonics.geometric_native_joint_spherical_field import (
        bind_native_joint_spherical_field, validate_native_joint_spherical_field,
        bound_joint_contact_face, bound_joint_layer_point, bound_joint_component_point,
    )
    from tectonics.geometric_native_spherical_solid import native_polygon_direction_bounds, interval_cbrt
    from tectonics.geometric_native_physical_pair_time import _rotated
    from tectonics.geometric_native_first_entry_material import certify_native_first_entry_material_point
    from tectonics.geometric_native_spherical_solid import certify_native_spherical_material_point
    from tectonics.geometric_native_polygons import _ray_values
    from tectonics.geometric_native_pose import interval_sqrt
    case, before, after, indices, axes = (context[name] for name in ("full_case", "before", "after", "indices", "axes"))
    support, epoch = case["support"], case["support"].source_epoch
    tops = tuple((identity, before[indices[identity]]) for identity in sorted(indices))
    field = bind_native_joint_spherical_field(support, top_radial_displacements_m=tops,
        provenance="One source-bound saved physical-radius reference flow per authentic contact-connected group; radial motion follows saved shear", precision=128)
    rows = []
    for component in field.components:
        solid, joint = component.source_solid, component.joint_layer
        identity = component.component_id
        index = indices[identity]
        corrections = tuple(row.containing_chart for row in solid.paired_corrections
            if row.positive_component_id == identity)
        charts = solid.component.proxy_geometries+corrections
        bounds = tuple(native_polygon_direction_bounds(chart, precision=128) for chart in charts)
        direction = tuple(I(min(row[k].lo for row in bounds), max(row[k].hi for row in bounds)) for k in range(3))
        z = I(0, joint.areal_thickness_m)
        reference_position = bound_joint_layer_point(joint, direction, z, precision=128, order=28)
        reference_radius = interval_cbrt(I(joint.layer.top_radius_m**3)-3*joint.radius_m**2*z, 128)
        reference_direction = tuple(value/reference_radius for value in reference_position)
        top = I(joint.layer.top_radius_m)+I(0, 1)*(after[index]-before[index])
        cubes = top*top*top-3*joint.radius_m**2*z
        current_radius = interval_cbrt(cubes, 128)
        whole_clock = _rotated(tuple(current_radius*value for value in reference_direction),
            axes[solid.component.owner], I(0, 1), 128, 28)
        if solid.reference_volume_bounds_m3 != solid.physical_volume_bounds_m3:
            raise ValueError("The source-bound joint map changed an actual material volume")
        rows.append(dict(component_id=identity, material_kind=solid.component.material_kind,
            original_parent_id=solid.component.original_parent_id, owner=solid.component.owner,
            source_component_fingerprint_sha256=solid.source_component_fingerprint_sha256,
            material_thickness_m=outward(joint.areal_thickness_m),
            reference_volume_bounds_m3=outward(solid.reference_volume_bounds_m3),
            physical_volume_bounds_m3=outward(solid.physical_volume_bounds_m3),
            exact_material_volume_jacobian_m2=outward(joint.physical_volume_jacobian),
            radial_before_m=outward(before[index]), radial_after_m=outward(after[index]),
            actual_bank_coordinate_index=index, selected_actual_bank=context["bank_sources"][identity],
            original_whole_mask_bending_field_reconstructed=False,
            saved_reference_top_physical_volume_depth_m=outward(component.top_physical_volume_depth_m),
            saved_reference_top_angular_shift=payload(component.top_angular_shift),
            shared_reference_angular_rate_per_m=payload(component.angular_rate_per_physical_volume_depth_m),
            source_containing_chart_count=len(charts), positive_curved_correction_chart_count=len(corrections),
            charts_assign_material=False, full_depth_reference_world_bounds_m=payload(reference_position),
            full_depth_and_clock_world_bounds_m=payload(whole_clock),
            source_volume_preserved_exactly=True,
            physical_noninterpenetration_certified=False))
    faces = []
    for carrier in support.interface_carriers:
        face = bound_joint_contact_face(field, carrier.declaration.family_id, precision=128, order=28)
        uncovered = {}
        for bank, identity, depth in (("donor", carrier.donor_component.component_id, face.donor_material_depth_bounds_m),
                ("receiver", carrier.receiver_component.component_id, face.receiver_material_depth_bounds_m)):
            H = next(row.layer.areal_thickness_m for row in field.components if row.component_id == identity)
            uncovered[bank] = [outward(part) for part in (
                *((I(0, depth.lo),) if depth.lo > 0 else ()),
                *((I(depth.hi, H),) if depth.hi < H else ()))]
        faces.append(dict(family_id=carrier.declaration.family_id,
            donor_component_id=carrier.donor_component.component_id,
            receiver_component_id=carrier.receiver_component.component_id,
            common_physical_radius_cubed_bounds_m3=outward(face.common_physical_radius_cubed_bounds_m3),
            physical_radius_bounds_m=outward(face.physical_radius_bounds_m),
            common_physical_volume_depth_bounds_m=outward(face.common_physical_volume_depth_bounds_m),
            donor_material_depth_bounds_m=outward(face.donor_material_depth_bounds_m),
            receiver_material_depth_bounds_m=outward(face.receiver_material_depth_bounds_m),
            shared_reference_depth_span_m=outward(face.common_physical_volume_depth_bounds_m.width),
            unchanged_full_source_material_layers_retained=True,
            source_depth_outside_this_shared_face=uncovered,
            outside_face_declared_traction_free=False,
            shared_reference_bank_faces_match_for_every_physical_radius=True,
            whole_angle_depth_world_position_bounds_m=payload(face.world_position_bounds_m),
            actual_receiver_inward_unit_normal_bounds=payload(face.receiver_inward_unit_normal_bounds),
            actual_curved_area_density_bounds=outward(face.area_density_bounds_m2_per_radian_per_material_m),
            local_cot_dip_bounds=outward(face.local_cot_dip_bounds),
            independently_moved_or_compressed_faces_glued=False))
    # Exercise real native replay refusal, without fabricating a source fixture.
    first = field.components[0]
    wrong = replace(first, angular_rate_per_physical_volume_depth_m=tuple(
        value+(F(1, 10**6) if index == 0 else 0)
        for index, value in enumerate(first.angular_rate_per_physical_volume_depth_m)))
    forgeries = (replace(field, source_support_sha256="0"*64),
        replace(field, components=(wrong,)+field.components[1:]),
        replace(field, top_radial_displacements_m=((tops[0][0], tops[0][1]+F(1)),)+tops[1:]))
    rejected = 0
    for forged in forgeries:
        try:
            validate_native_joint_spherical_field(forged)
        except ValueError:
            rejected += 1
        else:
            raise ValueError("A forged complete native joint reference field was accepted")
    state = epoch.source_material_state
    parent = next(row for row in state.journal.source_surface.fragments
        if row.parcel.plate == state.sweep_certificate.receiver_owner)
    point = tuple(sum((F(ray.reference[k]) for ray in parent.geometry.vertices), F(0)) for k in range(3))
    chronology = certify_native_first_entry_material_point(state, parent.fragment_id, point)
    actual = next(row for row in field.components if row.source_solid.component.material_kind == "original_live"
        and row.source_solid.component.original_parent_id == parent.fragment_id)
    membership = certify_native_spherical_material_point(actual.source_solid, chronology,
        actual.layer.areal_thickness_m/2, precision=128, order=28)
    query = membership.endpoint_query
    ray = query.endpoint_ray.rotated(dict(epoch.additional_owner_histories)[actual.source_solid.component.owner])
    vector = _ray_values(ray, 128)[1]
    norm = interval_sqrt(sum((value*value for value in vector), I(0)), 136)
    direction = tuple(value/norm for value in vector)
    placed = bound_joint_component_point(field, actual.component_id, direction,
        I(actual.layer.areal_thickness_m/2), precision=128, order=28)
    result = dict(format="native-joint-saved-physical-radius-field-audit-1", status="PASS",
        source_epoch_fingerprint_sha256=field.source_epoch_fingerprint_sha256,
        source_support_sha256=field.source_support_sha256,
        groups=payload(field.groups), actual_source_continuity_graph_edges=payload(field.continuity_graph_edges),
        native_angular_constraints=payload(field.constraints),
        all_13_source_masks_and_volumes_retained=True, component_count=len(rows), components=rows,
        declared_carrier_count=len(faces), shared_reference_faces=faces,
        shared_component_axis_conflicts=0,
        native_source_field_forgeries_rejected=rejected,
        independent_actual_point_membership=dict(component_id=actual.component_id,
            chronology_sha256=chronology.point_chronology_sha256,
            material_membership_authenticated=membership.material_membership_authenticated,
            material_depth_m=outward(actual.layer.areal_thickness_m/2),
            saved_joint_reference_position_bounds_m=payload(placed), proxy_membership_used=False),
        one_depth_flow_per_authenticated_contact_connected_group=True,
        one_global_flow_for_all_source_masks=len(field.groups) == 1,
        old_surface_classification_inherited=False,
        normal_column_boundary_deformation_glued=False,
        global_physical_noninterpenetration_certified=False,
        whole_clock_material_admission_authorized=False)
    return field, result


def audit_curved_energy(context, field):
    from tectonics.geometric_native_curved_contact_energy import bound_native_joint_curved_contact_response
    before, after, indices, axes = (context[name] for name in ("before", "after", "indices", "axes"))
    rows = []
    for carrier in context["full_case"]["support"].interface_carriers:
        d, r = indices[carrier.donor_component.component_id], indices[carrier.receiver_component.component_id]
        response = bound_native_joint_curved_contact_response(field, carrier.declaration.family_id,
            donor_radial_before_m=before[d], receiver_radial_before_m=before[r],
            donor_radial_increment_m=after[d]-before[d], receiver_radial_increment_m=after[r]-before[r],
            donor_owner_axis=axes[carrier.donor_component.owner], receiver_owner_axis=axes[carrier.receiver_component.owner],
            time_fraction_bounds=I(1), angle_cells=2, depth_cells=2, precision=96, order=28)
        if (response.potential_bounds_j.lo < 0 or response.integrated_pressure_bounds_n.lo < 0
                or len(response.spatial_coordinate_gradient_bounds) != 8
                or response.source_force_certificate_reused or response.activation_history_adopted):
            raise ValueError("The new curved constitutive response has invalid energy, scope or source-gradient dimensions")
        rows.append(dict(family_id=carrier.declaration.family_id,
            actual_donor_radial_coordinate_index=d, actual_receiver_radial_coordinate_index=r,
            response=payload(response), current_physical_area_and_normal_derivatives_included=True,
            saved_angular_reference_fixed_during_actual_radial_motion=True,
            one_dimensional_column_and_fault_energy_partition_certified=True,
            conservative_endpoint_work_is_potential_change=True,
            complete_actual_angle_and_shared_reference_depth_enclosed=True,
            complete_three_dimensional_strain_energy_certified=False,
            native_nineteen_coordinate_force_equilibrium_certified=False))
    return dict(format="native-joint-curved-variational-energy-audit-1", status="PASS", contacts=rows,
        previous_reference_force_motion_used_only_as_authenticated_probe=True,
        previous_reference_force_certificate_inherited=False,
        new_curved_activation_history_adopted=False,
        whole_clock_force_and_collision_certified=False)


def run_control(out, *, case=None):
    from analysis.geometric_native_birth_front_validation.native_continuum_case import build_native_continuum_case
    from tectonics.geometric_native_physical_history import _typed_fingerprint
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    stages = {}
    def timed(name, callback):
        tick = time.perf_counter()
        print("JOINT control", name, "starting", flush=True)
        result = callback()
        stages[name] = time.perf_counter()-tick
        print("JOINT control", name, "finished", round(stages[name], 3), flush=True)
        return result
    if case is None:
        case = timed("genuine_source", build_native_continuum_case)
    objects = tuple(case[name] for name in ("previous", "load", "support"))
    preserved = tuple(_typed_fingerprint(value) for value in objects)
    context = timed("exact_genuine_source_and_saved_motion_binding", lambda: prepare_source_context(case))
    field, geometry = timed("joint_native_reference_geometry", lambda: audit_joint_geometry(context))
    energy = timed("actual_curved_variational_contact_response", lambda: audit_curved_energy(context, field))
    if tuple(_typed_fingerprint(value) for value in objects) != preserved:
        raise ValueError("The new reference field or constitutive probe changed its complete accepted source")
    packet = dict(format="native-joint-spherical-source-and-curved-energy-packet-1", **context["source"],
        authenticated_previous_motion_packet_raw_sha256=PRIOR_RAW,
        authenticated_previous_motion_packet_seal=context["source_packet_seal"],
        authenticated_previous_reference_force_motion=context["source_motion"],
        joint_reference_geometry=geometry, new_curved_variational_response=energy,
        accepted_input_unchanged=True, material_admission_authorized=False)
    packet["packet_sha256"] = hashlib.sha256(encoded(packet)).hexdigest()
    write(out/"candidate_packet.json", packet)
    body = dict(format="native-joint-spherical-source-and-curved-energy-control-1", status="PASS",
        before_state_sha256=case["previous"].state_sha256, current_load_sha256=case["load"].load_sha256,
        birth_support_sha256=field.source_support_sha256,
        complete_genuine_source_payload_sha256={name: hashlib.sha256(encoded(value)).hexdigest()
            for name, value in context["source"].items()},
        accepted_input_complete_typed_state_unchanged=True,
        packet_sha256=packet["packet_sha256"], joint_reference_geometry=geometry,
        actual_curved_variational_contact_response=energy,
        shared_component_reference_axis_conflicts_resolved=True,
        curved_area_and_normal_in_reduced_potential_gradient_certified=True,
        previous_reference_force_certificate_inherited=False,
        native_curved_force_equilibrium_certified=False,
        joint_normal_column_boundary_deformation_glued=False,
        complete_three_dimensional_strain_energy_certified=False,
        global_physical_solid_noninterpenetration_certified=False,
        all_42_surface_pairs_reexecuted=False, previous_surface_pair_classification_inherited=False,
        whole_clock_material_admission_authorized=False,
        accepted_original_native_intervals=1, accepted_new_native_intervals=0,
        original_point_1_completed=False, original_point_1_remaining_work_blocks=1,
        stage_seconds=stages, seconds=time.perf_counter()-started,
        remaining_physical_conditions=[
            "global normal-column boundary deformation and three-dimensional energy, coupled actual curved force equilibrium",
            "whole-clock all-pair material admission on the resulting literal motion",
            "adopted mixed continuation, balances and joint restart"])
    body["report_sha256"] = hashlib.sha256(encoded(body)).hexdigest()
    write(out/"report.json", body)
    print("JOINT native control PASS", body["seconds"], flush=True)
    return body
