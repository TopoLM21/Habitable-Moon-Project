"""Genuine source-mask, curved face and whole-clock embedding control.

This helper receives the caller's already reconstructed native case.  It
does not build a fixture, advance material, assign integration charts as
material, or claim that separate carrier models share a global strain field.
"""
from __future__ import annotations

from decimal import Decimal
from fractions import Fraction as F

from analysis.geometric_native_finite_equilibrium_validation.native_control import outward
from tectonics.geometric_native_pose import FractionInterval as I


def _vector_payload(values):
    return [outward(value) for value in values]


def audit_spherical_embedding(full_case, candidate):
    """Return JSON-compatible controls for the actual thirteen-mask source."""
    from tectonics.thermal import SECONDS_PER_MYR
    from tectonics.geometric_native_implicit_epoch_admission import _portable_fingerprint_sha256
    from tectonics.geometric_native_physical_history import _typed_fingerprint
    from tectonics.geometric_native_first_entry_material import (
        certify_native_first_entry_material_point,
    )
    from tectonics.geometric_native_spherical_solid import (
        bind_native_spherical_component_solid,
        bound_native_spherical_whole_mask_motion,
        declare_native_spherical_contact_solid_model,
        bound_native_spherical_contact_solid_face,
        bound_spherical_arc_face_motion,
        spherical_layer_radius_bounds,
        certify_native_spherical_material_point,
    )

    before, support = full_case["previous"], full_case["support"]
    epoch, inputs = support.source_epoch, full_case["inputs"]
    source_objects = (before, full_case["load"], support, candidate)
    fingerprints = tuple(_typed_fingerprint(item) for item in source_objects)
    if (candidate.before_state_sha256 != before.state_sha256
            or candidate.birth_support_sha256 != support.support_sha256
            or not candidate.force_equilibrium_certified
            or candidate.material_admission_authorized):
        raise ValueError("Spherical control requires the genuine unaccepted finite-force candidate")
    if len(epoch.components) != 13 or len(support.interface_carriers) != 2:
        raise ValueError("The genuine thirteen-component/two-carrier source changed")
    radius = F.from_float(inputs["radius_m"])
    physical_before = {row.coordinate_index: row.physical_before_m
        for row in candidate.exact_radial_normalization.coordinates}
    physical_after = {index: F(Decimal(value))*radius
        for index, value in enumerate(candidate.coordinates_decimal)
        if index in physical_before}
    axes = dict(candidate.literal_owner_axes)
    dt_myr = full_case["dt_s"]/SECONDS_PER_MYR
    reconstructed_axes = {owner: tuple(F.from_float(value)*F.from_float(dt_myr) for value in omega)
        for owner, omega in candidate.omega_rad_per_myr}
    if axes != reconstructed_axes:
        raise ValueError("Whole-clock embedding does not use the literal corrected candidate motion")
    angular = inputs["rotation_dof_count"]
    traces = before.initial_mechanics.fault_traces.traces
    bindings = {row.trace_id: row for row in epoch.source_material_state.trace_bindings}
    banks = {}
    for trace_index, trace in enumerate(traces):
        binding = bindings[trace.trace_id]
        for bank, parent, index in (("donor", binding.donor_fragment_id, angular+2*trace_index),
                ("receiver", binding.receiver_fragment_id, angular+2*trace_index+1)):
            banks.setdefault(parent, []).append((trace.trace_id, bank, index))
    foundation_indices = {row.reference.patch_id: row.radial_dof_index for row in inputs["born_foundations"]}
    component_indices, bank_sources = {}, {}
    for component in epoch.components:
        if component.material_kind == "hot_birth":
            component_indices[component.component_id] = foundation_indices[component.component_id]
            bank_sources[component.component_id] = (component.component_id, "whole_hot_foundation")
        else:
            matches = tuple(sorted(banks.get(component.original_parent_id, ())))
            if not matches:
                raise ValueError("A genuine cold component has no original physical bank for its declared placement")
            trace_id, bank, index = matches[0]
            component_indices[component.component_id] = index
            bank_sources[component.component_id] = (trace_id, bank)

    # These individual maps use no depth shear.  Their uniform top placement
    # is declared from an actual bank; it is not a reconstructed cantilever
    # deformation over an old component's entire spherical footprint.
    solids, mask_rows = {}, []
    volume_sum = I(0)
    for component in epoch.components:
        identity = component.component_id
        index = component_indices[identity]
        solid = bind_native_spherical_component_solid(epoch, identity,
            top_radial_displacement_m=physical_before[index],
            provenance="Explicit uniform whole-mask placement declared from one genuine physical bank; old beam field not reconstructed")
        solids[identity] = solid
        motion = bound_native_spherical_whole_mask_motion(solid,
            owner_rotation_axis=axes[component.owner],
            top_radial_increment_m=physical_after[index]-physical_before[index],
            time_fraction_bounds=I(0, 1), precision=128, order=28)
        if (solid.physical_volume_bounds_m3 != solid.reference_volume_bounds_m3
                or not motion.whole_actual_implicit_mask_depth_and_clock_enclosed
                or motion.containing_charts_assign_material):
            raise ValueError("The complete curved source-mask lift lost material or its full clock")
        volume_sum += solid.reference_volume_bounds_m3
        mask_rows.append(dict(component_id=identity, material_id=component.material_id,
            material_kind=component.material_kind, original_parent_id=component.original_parent_id,
            owner=component.owner, domain_expression=component.domain_expression,
            source_component_fingerprint_sha256=solid.source_component_fingerprint_sha256,
            actual_source_material_areal_thickness_m=outward(solid.layer.areal_thickness_m),
            physical_radial_height_bounds_m=outward(solid.layer.physical_radial_height_bounds_m),
            reference_volume_bounds_m3=outward(solid.reference_volume_bounds_m3),
            current_declared_volume_bounds_m3=outward(solid.physical_volume_bounds_m3),
            exact_volume_jacobian_m2=outward(solid.layer.volume_jacobian),
            volume_preserved_exactly=True, selected_actual_bank_id=bank_sources[identity][0],
            selected_bank_role=bank_sources[identity][1], selected_coordinate_index=index,
            radial_before_m=outward(physical_before[index]), radial_after_m=outward(physical_after[index]),
            radial_increment_m=outward(physical_after[index]-physical_before[index]),
            declared_base_depth_rotation_axis=_vector_payload(solid.layer.depth_rotation_axis),
            containing_chart_count=motion.containing_chart_count,
            positive_curved_correction_chart_count=motion.positive_curved_correction_chart_count,
            paired_curved_correction_count=len(solid.paired_corrections),
            whole_source_mask_direction_bounds=_vector_payload(motion.containing_direction_bounds),
            full_depth_clock_radial_bounds_m=outward(motion.motion.radial_bounds_m),
            full_mask_depth_clock_world_bounds_m=_vector_payload(motion.motion.world_position_bounds_m),
            whole_actual_implicit_mask_depth_and_clock_enclosed=True,
            containing_charts_assign_material=False,
            original_whole_mask_bending_field_reconstructed=False,
            whole_component_declared_embedding_injective=True,
            physical_mask_collision_excluded=False))

    face_rows, axis_assignments = [], {}
    for carrier in support.interface_carriers:
        donor_id, receiver_id = carrier.donor_component.component_id, carrier.receiver_component.component_id
        donor_index, receiver_index = component_indices[donor_id], component_indices[receiver_id]
        model = declare_native_spherical_contact_solid_model(support, carrier,
            donor_top_radial_displacement_m=physical_before[donor_index],
            receiver_top_radial_displacement_m=physical_before[receiver_index],
            provenance="Explicit common SO(3) depth flow for this actual curved carrier; source normal fields not globally glued", precision=128)
        bank_rows = []
        for bank, solid, index in (("donor", model.donor, donor_index), ("receiver", model.receiver, receiver_index)):
            identity = solid.component.component_id
            axis_assignments.setdefault(identity, []).append((carrier.declaration.family_id, solid.layer.depth_rotation_axis))
            face = bound_native_spherical_contact_solid_face(model, bank=bank, precision=128, order=28)
            moving = bound_spherical_arc_face_motion(solid.layer, carrier.edge,
                I(0, carrier.native_angle_bounds_radians.hi), I(0, model.common_material_face_depth_m),
                owner_rotation_axis=axes[solid.component.owner],
                top_radial_increment_m=physical_after[index]-physical_before[index],
                time_fraction_bounds=I(0, 1), precision=128, order=28)
            common_height = I(solid.layer.top_radius_m)-spherical_layer_radius_bounds(solid.layer,
                I(model.common_material_face_depth_m), precision=128)
            bank_rows.append(dict(bank=bank, component_id=identity,
                actual_bank_coordinate_index=index, common_material_depth_m=outward(model.common_material_face_depth_m),
                common_face_physical_radial_height_bounds_m=outward(common_height),
                physical_radial_height_difference_from_reference_m=outward(common_height-model.common_material_face_depth_m),
                declared_depth_rotation_axis=_vector_payload(solid.layer.depth_rotation_axis),
                full_angle_depth_cartesian_bounds_m=_vector_payload(face.world_position_bounds_m),
                full_angle_depth_geometric_area_density_bounds=outward(face.area_density_bounds_m2_per_radian_per_material_m),
                full_angle_depth_clock_cartesian_bounds_m=_vector_payload(moving.world_position_bounds_m),
                whole_curved_face_depth_and_clock_enclosed=True,
                actual_curved_area_force_metric_certified=False,
                physical_contact_face_collision_excluded=False))
        face_rows.append(dict(family_id=carrier.declaration.family_id,
            donor_component_id=donor_id, receiver_component_id=receiver_id,
            native_angle_bounds_radians=outward(carrier.native_angle_bounds_radians),
            reference_force_height_coordinate_m=outward(model.declared_force_vertical_contact_height_m),
            common_material_depth_m=outward(model.common_material_face_depth_m),
            declared_common_depth_rotation_rate_per_material_m=_vector_payload(model.depth_rotation_axis_per_material_m),
            source_cot_dip=outward(model.source_cot_dip),
            reference_surface_cot_dip_bounds=outward(model.reference_surface_cot_dip_bounds),
            donor_surface_cot_dip_bounds=outward(model.donor_surface_cot_dip_bounds),
            receiver_surface_cot_dip_bounds=outward(model.receiver_surface_cot_dip_bounds),
            source_depth_axis_alias_bounds_per_m=_vector_payload(model.depth_axis_declaration_error_bounds_per_m),
            banks=bank_rows, normal_column_and_contact_deformation_glued=False,
            actual_curved_area_force_metric_certified=False))
    conflicts = []
    for identity, assignments in axis_assignments.items():
        if len({axis for _, axis in assignments}) > 1:
            conflicts.append(dict(component_id=identity, source="independent carrier depth-flow models",
                carrier_axis_assignments=[dict(family_id=family, depth_rotation_axis=_vector_payload(axis))
                    for family, axis in assignments]))

    # One genuinely live original receiver point exercises independent mask
    # membership.  Its trajectory is certified from the exact original parent
    # before applying source poses; a proxy midpoint never grants membership.
    parents = epoch.source_material_state.journal.source_surface.fragments
    receiver_owner = epoch.source_material_state.sweep_certificate.receiver_owner
    parent = next(row for row in parents if row.parcel.plate == receiver_owner)
    if parent.geometry.pose or any(ray.rotations or ray.kind != "seed" for ray in parent.geometry.vertices):
        raise ValueError("The cheap original source point requires its genuine unposed exact native seed parent")
    point = tuple(sum((F(ray.reference[index]) for ray in parent.geometry.vertices), F(0)) for index in range(3))
    chronology = certify_native_first_entry_material_point(epoch.source_material_state, parent.fragment_id, point)
    component = next(row for row in epoch.components
        if row.material_kind == "original_live" and row.original_parent_id == parent.fragment_id)
    material_point = certify_native_spherical_material_point(solids[component.component_id], chronology,
        solids[component.component_id].layer.areal_thickness_m/2, precision=128, order=28)
    if not material_point.material_membership_authenticated:
        raise ValueError("The spherical source-point placement lost its actual material membership")
    if tuple(_typed_fingerprint(item) for item in source_objects) != fingerprints:
        raise ValueError("The spherical geometry audit mutated its source or force candidate")
    return dict(format="native-spherical-whole-mask-and-curved-face-control-1", status="PASS",
        source_epoch_sha256=epoch.epoch_sha256,
        complete_source_epoch_fingerprint_sha256=_portable_fingerprint_sha256(epoch),
        candidate_before_state_sha256=candidate.before_state_sha256,
        birth_support_sha256=support.support_sha256,
        corrected_candidate_literal_owner_axes=[dict(owner=owner, axis=_vector_payload(axis)) for owner, axis in candidate.literal_owner_axes],
        actual_source_component_count=len(mask_rows), actual_source_carrier_count=len(face_rows),
        source_volume_sum_bounds_m3=outward(volume_sum), component_masks=mask_rows, curved_carrier_faces=face_rows,
        actual_point_membership_control=dict(original_parent_id=parent.fragment_id,
            component_id=component.component_id, source_original_point=_vector_payload(point),
            point_chronology_sha256=chronology.point_chronology_sha256,
            membership_authenticated_by_independent_material_chronology=True,
            material_depth_m=outward(material_point.material_depth_m),
            world_position_bounds_m=_vector_payload(material_point.world_position_bounds_m),
            proxy_point_used_as_membership=False),
        all_source_masks_volumes_and_curved_corrections_retained=True,
        all_source_masks_full_depth_and_candidate_clock_enclosed=True,
        carrier_models_share_one_global_component_field=False,
        component_axis_conflicts=conflicts,
        candidate_bank_field_placement_reconstructed=False,
        old_whole_mask_bending_field_reconstructed=False,
        complete_global_normal_deformation_field_glued=False,
        actual_curved_area_force_metric_certified=False,
        global_physical_solid_noninterpenetration_certified=False,
        whole_clock_material_admission_authorized=False, accepted_new_native_intervals=0,
        complete_source_and_candidate_unchanged=True)
