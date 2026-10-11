"""Actual first-birth reference-depth and reduced Robin-placement audit.

This helper observes the genuine freshly solved trial supplied by the root
controller.  It never rebuilds the expensive accepted fixture or writes any
artifact.  Reference-chart anchors below are explicit new model declarations,
not a reconstruction of independent world anchors or a global deformation.
"""
from __future__ import annotations

from decimal import Decimal
from fractions import Fraction as F

from analysis.geometric_native_finite_equilibrium_validation.native_control import (
    outward, payload,
)
from tectonics.geometric_native_pose import FractionInterval as I


def _fingerprint(value):
    from tectonics.geometric_native_physical_history import _typed_fingerprint
    return _typed_fingerprint(value)


def _reference_robin_audit(columns, law, gap, *, geometry_source_id):
    """Check real reduced pressure, then declare a separate flat chart.

    The entire gap interval yields conservative no-collapse bounds.  The
    midpoint supplies one exact illustrative chart with independently named
    *declared* anchors; those anchors have no independent world-source proof.
    Neither the midpoint nor this chart certifies the actual spherical face.
    """
    from tectonics.geometric_solid_contact_closure import compute_series_normal_closure
    from tectonics.geometric_native_solid_face_closure import (
        place_solid_contact_robin_segments, robin_column_material_point,
    )
    if type(gap) is not I:
        raise ValueError("A real reduced pressure audit requires its whole gap enclosure")
    sub_length = F.from_float(columns.sub_length_m)
    over_length = F.from_float(columns.over_length_m)
    sub_compliance = sub_length/F.from_float(columns.sub_normal_modulus_pa)
    over_compliance = over_length/F.from_float(columns.over_normal_modulus_pa)
    fault_compliance = 1/F.from_float(law.normal_stiffness_pa_m)
    stiffness = 1/(sub_compliance+over_compliance+fault_compliance)
    pressure = -I(min(gap.lo, F(0)), min(gap.hi, F(0)))*stiffness
    fraction = F(1, 2)
    donor_stretch = I(1)-pressure*(sub_compliance+fraction*fault_compliance)/sub_length
    receiver_stretch = I(1)-pressure*(over_compliance+(1-fraction)*fault_compliance)/over_length
    if donor_stretch.lo <= 0 or receiver_stretch.lo <= 0:
        raise ValueError("The real reduced response cannot fit the declared Robin mode in its old columns")
    normal = compute_series_normal_closure(columns=columns, law=law,
        free_gap_m=gap.midpoint)
    # These exact scalars DECLARE a new flat reference chart.  Their distance
    # is not falsely asserted to have been observed in the spherical source.
    donor_anchor = F(0)
    receiver_anchor = sub_length+over_length+normal.free_gap_m
    placement = place_solid_contact_robin_segments(normal=normal,
        donor_anchor_m=donor_anchor, receiver_anchor_m=receiver_anchor,
        fault_indentation_donor_fraction=fraction,
        geometry_source_id=geometry_source_id)
    donor_tip = robin_column_material_point(placement, bank="donor",
        normal_coordinate_m=sub_length, strike_coordinate_m=F(0), depth_coordinate_m=F(0))
    receiver_tip = robin_column_material_point(placement, bank="receiver",
        normal_coordinate_m=over_length, strike_coordinate_m=F(0), depth_coordinate_m=F(0))
    if (donor_tip[0] != placement.donor_physical_face_m
            or receiver_tip[0] != placement.receiver_physical_face_m
            or placement.total_energy_j_m2 != normal.total_energy_j_m2
            or placement.additional_material_volume_m3 != 0
            or placement.added_external_work_j_m2 != 0
            or placement.activation_paid_again
            or placement.donor_local_volume_jacobian != 1
            or placement.receiver_local_volume_jacobian != 1):
        raise ValueError("The declared reference-chart placement changed source energy, volume or geometry")
    return dict(
        whole_response_gap_bounds_m=outward(gap),
        whole_response_reduced_normal_pressure_bounds_pa=outward(pressure),
        whole_response_donor_robin_stretch_bounds=outward(donor_stretch),
        whole_response_receiver_robin_stretch_bounds=outward(receiver_stretch),
        no_existing_normal_column_collapse=True,
        fault_indentation_donor_fraction=outward(fraction),
        scalar_example_gap_m=outward(normal.free_gap_m),
        scalar_example_normal_pressure_pa=outward(normal.normal_pressure_pa),
        declared_flat_reference_donor_anchor_m=outward(donor_anchor),
        declared_flat_reference_receiver_anchor_m=outward(receiver_anchor),
        scalar_example_physical_void_m=outward(placement.physical_face_void_m),
        scalar_example_bulk_shortening_m=[outward(normal.sub_column_shortening_m),
            outward(normal.over_column_shortening_m)],
        scalar_example_robin_boundary_shortening_m=[
            outward(placement.donor_robin_boundary_shortening_m),
            outward(placement.receiver_robin_boundary_shortening_m)],
        scalar_example_reduced_energy_j_per_reference_m2=outward(normal.total_energy_j_m2),
        scalar_example_deformation_gradients=payload((
            placement.donor_local_deformation_gradient,
            placement.receiver_local_deformation_gradient)),
        exact_local_material_volume_jacobians=[1, 1],
        additional_material_volume_m3=outward(F(0)),
        added_work_j_per_reference_m2=outward(F(0)),
        paid_activation_added_by_placement=False,
        geometry_source_id=geometry_source_id,
        anchor_geometry_is_explicit_new_reference_chart=True,
        independently_authenticated_world_anchor_geometry=False,
        reduced_source_energy_preserved=True,
        three_dimensional_strain_energy_certified=False,
        curved_contact_area_and_generalized_force_metric_certified=False,
        coherent_whole_mask_boundary_deformation_field_certified=False,
        global_spherical_solid_embedding_certified=False,
        material_admission_authorized=False,
    )


def audit_full_face(full_case, candidate, branch, old_support):
    """Return an exact-source, JSON-friendly audit of the new full-depth trial."""
    from tectonics.geometric_native_birth_contact_continuum import (
        activate_born_continuum_contact,
    )
    from tectonics.geometric_native_birth_contact_support import (
        native_birth_contact_support_to_dict,
    )
    from tectonics.geometric_native_solid_layer_placement import native_component_solid_thickness_m
    from tectonics.geometric_native_birth_finite_pose import born_finite_pose_angle_bounds
    from tectonics.geometric_native_solid_face_closure import (
        NativeFirstBirthFullFaceDeclaration,
    )
    previous, support, inputs = (full_case[name] for name in ("previous", "support", "inputs"))
    saved = tuple(_fingerprint(value) for value in (previous, old_support, support))
    native_birth_contact_support_to_dict(old_support)
    native_birth_contact_support_to_dict(support)
    if (type(branch) is not NativeFirstBirthFullFaceDeclaration
            or branch.before_state_sha256 != previous.state_sha256
            or branch.original_trial_support_sha256 != old_support.support_sha256
            or branch.full_face_support_sha256 != support.support_sha256
            or branch.source_epoch_sha256 != support.source_epoch.epoch_sha256
            or branch.original_accepted_contact_count != 16
            or branch.original_trial_activation_reused is not False
            or branch.fresh_first_birth_activation_required is not True
            or branch.adopted_born_history_extended is not False
            or candidate.before_state_sha256 != previous.state_sha256
            or candidate.birth_support_sha256 != support.support_sha256):
        raise ValueError("The full material-depth candidate changed its explicit first-birth source branch")
    if (_fingerprint(old_support.source_epoch) != _fingerprint(support.source_epoch)
            or _fingerprint(previous.current_surface) != _fingerprint(support.source_epoch)
            or _fingerprint(old_support.incoming_geometry) != _fingerprint(support.incoming_geometry)
            or old_support.foundation_declarations != support.foundation_declarations):
        raise ValueError("A new full-depth contact trial changed its genuine before geometry or foundations")
    old_by_family = {row.declaration.family_id: row for row in old_support.interface_carriers}
    full_ids = tuple(row.declaration.family_id for row in support.interface_carriers)
    if len(old_by_family) != len(support.interface_carriers) or set(old_by_family) != set(full_ids):
        raise ValueError("The full-depth first-birth model changed its actual carrier families")
    attachments = inputs["born_continuum_attachments"]
    if not (len(attachments) == len(candidate.responses) == len(candidate.born_contact_ledgers)
            == len(candidate.fields) == len(support.interface_carriers)):
        raise ValueError("The full-depth candidate omitted an actual response or born history")
    dt = F.from_float(full_case["dt_s"])
    height_rows, contact_rows, expected_branch_records = [], [], []
    for carrier, attachment, response, ledger, field in zip(support.interface_carriers,
            attachments, candidate.responses, candidate.born_contact_ledgers, candidate.fields):
        family = carrier.declaration.family_id
        old = old_by_family[family]
        donor = native_component_solid_thickness_m(support.source_epoch, carrier.donor_component.component_id)
        receiver = native_component_solid_thickness_m(support.source_epoch, carrier.receiver_component.component_id)
        common = min(donor, receiver)
        old_height = F.from_float(old.declaration.vertical_contact_height_m)
        full_height = F.from_float(carrier.declaration.vertical_contact_height_m)
        if (full_height != common or old_height > common
                or carrier.declaration.parameters != old.declaration.parameters
                or carrier.declaration.birth_gap_free_m != old.declaration.birth_gap_free_m
                or carrier.declaration.birth_jump_free_m != old.declaration.birth_jump_free_m
                or carrier.declaration.dip_radians != old.declaration.dip_radians
                or _fingerprint(carrier.edge) != _fingerprint(old.edge)
                or _fingerprint(carrier.receiver_inward_plane) != _fingerprint(old.receiver_inward_plane)
                or _fingerprint(carrier.donor_component) != _fingerprint(old.donor_component)
                or _fingerprint(carrier.receiver_component) != _fingerprint(old.receiver_component)
                or carrier.inclined_measure_density_m2_per_radian*old_height
                    != old.inclined_measure_density_m2_per_radian*full_height):
            raise ValueError("The full-depth branch changed preload, old columns/law, banks or native force measure")
        expected_branch_records.append((family, donor, receiver, old_height, common))
        prior = attachment.previous_state
        replay_activation = activate_born_continuum_contact(attachment.declaration,
            decimal_precision=prior.decimal_precision)
        if (_fingerprint(prior) != _fingerprint(replay_activation)
                or prior.step_count != 0 or prior.history or prior.elapsed_time_s != 0.
                or prior.stored_energy_decimal != prior.activation_energy_decimal
                or Decimal(prior.cumulative_external_work_decimal) != 0
                or Decimal(prior.cumulative_numerical_dissipation_decimal) != 0
                or _fingerprint(ledger.previous_state) != _fingerprint(prior)
                or ledger.activation_energy_decimal != prior.activation_energy_decimal
                or ledger.activation_paid_again is not False
                or ledger.step_count != 1
                or ledger.elapsed_time_s != dt
                or len(ledger.history) != 1):
            raise ValueError("The full-depth first-birth branch reused a trial history, paid twice or changed its clock")
        height_rows.append(dict(family_id=family,
            donor_source_areal_volume_thickness_m=outward(donor),
            receiver_source_areal_volume_thickness_m=outward(receiver),
            original_unadopted_trial_force_reference_height_m=outward(old_height),
            new_first_birth_force_reference_height_m=outward(full_height),
            source_common_material_depth_m=outward(common),
            full_source_material_depth_reference_coverage=True,
            new_force_reference_measure_density=outward(carrier.inclined_measure_density_m2_per_radian),
            actual_curved_face_force_metric_certified=False))
        contact_rows.append(dict(family_id=family,
            fresh_paid_activation_energy_decimal=prior.activation_energy_decimal,
            fresh_virgin_history_replayed=True,
            previous_unadopted_trial_activation_reused=False,
            full_depth_first_birth_activation_paid_once=True,
            elapsed_time_s=outward(ledger.elapsed_time_s), step_count=ledger.step_count,
            activation_representation_error_j=outward(ledger.activation_energy_representation_error_j),
            finite_stored_energy_bounds_j=outward(response.energy_bounds_j),
            conservative_path_work_bounds_j=outward(ledger.conservative_path_work_increment_bounds_j),
            generalized_coordinate_energy_closure_certified=False,
            actual_endpoint_gap_evaluation_bounds_m=[outward(
                born_finite_pose_angle_bounds(field, angle)[0])
                for angle in (I(0), carrier.native_angle_bounds_radians)],
            robin_reference_chart=_reference_robin_audit(carrier.declaration.parameters.columns,
                carrier.declaration.parameters.law, response.gap_bounds_m,
                geometry_source_id="Explicit new flat reference chart at midpoint of actual full-depth finite response "+family)))
    if tuple(expected_branch_records) != branch.family_height_records:
        raise ValueError("The full-depth first-birth receipt differs from its actual source material amounts")
    old_ids = inputs["old_point_ids"]
    old_before = dict(previous.states)
    old_after = candidate.source_affine_step.original_point_states
    if (len(old_ids) != 16 or len(old_before) != 16 or len(old_after) != 16
            or tuple(identity for identity, _ in old_after) != old_ids
            or any(_fingerprint(local) != _fingerprint(old_before[identity])
                for identity, local in zip(old_ids, inputs["old_states"]))):
        raise ValueError("The full-depth first-birth source did not retain all accepted sixteen old histories")
    for identity, current in old_after:
        prior = old_before[identity]
        if (current.elapsed_time_s != prior.elapsed_time_s+full_case["dt_s"]
                or current.step_count != prior.step_count+1
                or current.columns_signature != prior.columns_signature
                or current.law_signature != prior.law_signature
                or current.area_m2 != prior.area_m2 or current.water != prior.water):
            raise ValueError("An old contact history changed its physical binding or advanced more than once")
    foundations = inputs["born_foundations"]
    if len(foundations) != len(candidate.corrected_foundation_steps):
        raise ValueError("The full-depth force trial omitted an independent HOT foundation")
    for foundation, step in zip(foundations, candidate.corrected_foundation_steps):
        prior = foundation.previous_state
        if (prior.step_count != 0 or prior.history or prior.elapsed_time_s != 0.
                or step.state.step_count != 1
                or step.state.elapsed_time_s != full_case["dt_s"]):
            raise ValueError("The fresh first-birth support has an extended adopted foundation history or different clock")
    if tuple(_fingerprint(value) for value in (previous, old_support, support)) != saved:
        raise ValueError("The full-depth audit changed its accepted source or an immutable trial support")
    return dict(format="native-full-reference-depth-first-birth-face-audit-1", status="PASS",
        before_state_sha256=previous.state_sha256,
        old_unadopted_trial_support_sha256=old_support.support_sha256,
        new_first_birth_support_sha256=support.support_sha256,
        first_birth_branch=payload(branch), material_reference_depth_coverage=height_rows,
        actual_full_depth_born_responses_and_virgin_ledgers=contact_rows,
        accepted_original_contact_count=16, accepted_original_histories_unchanged=True,
        original_trial_contact_clocks_advanced_once=True,
        fresh_independent_foundation_count=len(foundations),
        fresh_foundation_clocks_advanced_once=True,
        existing_adopted_born_history_reinitialized=False,
        full_source_material_depth_reference_coverage=True,
        actual_curved_face_force_metric_certified=False,
        independent_world_anchor_placement_certified=False,
        whole_mask_robin_boundary_deformation_field_certified=False,
        global_three_dimensional_strain_energy_certified=False,
        whole_clock_material_admission_authorized=False,
        accepted_new_native_intervals=0)
