"""Genuine source-bound full-depth force trial and declared spherical solids.

The individual spherical embedding, the local Robin deformation and the
reference-area force balance are separate certificates. Their global gluing
and physical curved-face force balance are deliberately not inferred here.
"""
from __future__ import annotations

from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import time

from analysis.geometric_native_finite_equilibrium_validation.native_control import (
    encoded, outward, payload, write,
)

ROOT = Path(__file__).resolve().parents[2]
PROVENANCE = "Spherical source-layer full material-depth first-birth reference declaration; unaccepted trial only"


def run_control(out, *, case=None):
    from analysis.geometric_native_birth_front_validation.native_continuum_case import build_native_continuum_case
    from analysis.geometric_native_spherical_solid_validation.native_embedding_control import audit_spherical_embedding
    from analysis.geometric_native_spherical_solid_validation.native_face_control import audit_full_face
    from tectonics.geometric_native_solid_face_closure import declare_native_full_face_first_birth_support
    from tectonics.geometric_native_birth_continuum_motion import build_native_birth_continuum_inputs
    from tectonics.geometric_native_birth_finite_equilibrium import solve_native_born_finite_equilibrium
    from tectonics.geometric_native_birth_exact_radial import normalize_native_birth_radial_before
    from tectonics.geometric_native_birth_contact_continuum import birth_continuum_network_to_dict
    from tectonics.geometric_native_implicit_underthrust_motion import native_implicit_underthrust_motion_to_dict
    from tectonics.geometric_native_implicit_mantle_loading import native_implicit_spatial_mantle_load_to_dict
    from tectonics.geometric_native_birth_contact_support import native_birth_contact_support_to_dict
    from tectonics.geometric_native_physical_history import _typed_fingerprint

    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    stages = {}

    def timed(name, callback):
        tick = time.perf_counter()
        print("SPHERICAL control", name, "starting", flush=True)
        result = callback()
        stages[name] = time.perf_counter()-tick
        print("SPHERICAL control", name, "finished", round(stages[name], 3), flush=True)
        return result

    if case is None:
        case = timed("actual_source", build_native_continuum_case)
    before, load, old_support = case["previous"], case["load"], case["support"]
    sources_before = tuple(_typed_fingerprint(row) for row in (before, load, old_support))
    full_support, branch = timed("full_reference_face_first_birth_declaration",
        lambda: declare_native_full_face_first_birth_support(before, old_support, provenance=PROVENANCE))
    inputs, frames = timed("genuine_full_reference_face_inputs", lambda:
        build_native_birth_continuum_inputs(case["mesh"], before, load, full_support,
            case["dt_s"], decimal_precision=160))
    full_case = dict(case, support=full_support, inputs=inputs, frames=frames)
    candidate = timed("actual_full_reference_face_finite_equilibrium",
        lambda: solve_native_born_finite_equilibrium(full_case))
    if (not candidate.force_equilibrium_certified or candidate.material_admission_authorized
            or candidate.before_state_sha256 != before.state_sha256
            or candidate.birth_support_sha256 != full_support.support_sha256):
        raise ValueError("The actual full-depth force trial differs from its source or declared scope")
    canonical_inputs, normalization = normalize_native_birth_radial_before(inputs, before)
    if candidate.exact_radial_normalization != normalization:
        raise ValueError("Full-depth forces do not use the accepted exact physical banks")
    old_dimension = len(canonical_inputs["old_stiffness"])
    if candidate.coordinates_decimal[:old_dimension] != candidate.source_affine_step.coordinates_decimal[:old_dimension]:
        raise ValueError("Finite born correction changed the genuine full-face original affine coordinates")
    if len(candidate.residual_bounds) != 19 or len(candidate.responses) != 2:
        raise ValueError("The nineteen-coordinate/two-carrier genuine case changed")
    tolerance = F(5, 10**10)
    if any(row > tolerance for row in candidate.relative_residual_upper):
        raise ValueError("A full-depth finite-force residual exceeds its own coordinate scale")
    geometry = timed("actual_source_spherical_embedding", lambda: audit_spherical_embedding(full_case, candidate))
    faces = timed("actual_full_face_and_local_robin", lambda:
        audit_full_face(full_case, candidate, branch, old_support))
    source = timed("complete_source_and_initial_force_replay", lambda: dict(
        before=native_implicit_underthrust_motion_to_dict(before),
        current_load=native_implicit_spatial_mantle_load_to_dict(case["mesh"], before.current_surface, load),
        original_unaccepted_trial_birth_support=native_birth_contact_support_to_dict(old_support),
        full_reference_face_birth_support=native_birth_contact_support_to_dict(full_support),
        canonical_full_face_affine_initial_solve=birth_continuum_network_to_dict(
            candidate.source_affine_step, **canonical_inputs)))
    force = dict(format="full-material-reference-depth-born-finite-force-trial-1",
        coordinates_decimal=candidate.coordinates_decimal,
        omega_rad_per_myr=candidate.omega_rad_per_myr,
        literal_owner_axes=payload(candidate.literal_owner_axes),
        angular_path_error_bounds=payload(candidate.angular_path_error_bounds),
        born_responses=payload(candidate.responses),
        born_finite_fields=payload(candidate.fields),
        born_finite_jacobians=payload(candidate.jacobians),
        born_trial_ledgers=payload(candidate.born_contact_ledgers),
        corrected_foundation_steps=payload(candidate.corrected_foundation_steps),
        original_point_states=payload(candidate.source_affine_step.original_point_states),
        original_anchor_fields=payload(candidate.source_affine_step.original_anchor_fields),
        residual_bounds=payload(candidate.residual_bounds),
        operand_scales=payload(candidate.operand_scales),
        relative_residual_upper=payload(candidate.relative_residual_upper),
        iterations=candidate.iterations,
        finite_born_reference_area_with_original_affine_force_equilibrium_certified=True,
        distributed_curved_physical_face_force_equilibrium_certified=False,
        global_generalized_coordinate_energy_closure_certified=False,
        whole_clock_material_admission_authorized=False)
    packet = dict(format="native-spherical-solid-full-reference-face-source-packet-1",
        **source, full_face_first_birth_declaration=payload(branch),
        full_reference_face_force_trial=force, spherical_embedding_audit=geometry,
        local_face_deformation_audit=faces,
        accepted_input_unchanged=True, material_admission_authorized=False)
    packet["packet_sha256"] = hashlib.sha256(encoded(packet)).hexdigest()
    write(out/"candidate_packet.json", packet)
    if tuple(_typed_fingerprint(row) for row in (before, load, old_support)) != sources_before:
        raise ValueError("The unaccepted trial mutated the complete accepted source, load or original support")
    prior = json.loads((ROOT/"analysis/geometric_native_finite_equilibrium_validation/release_validation_02/native_control/candidate_packet.json").read_bytes())
    prior_motion = prior["corrected_finite_force_trial"]["omega_rad_per_myr"]
    motion_matches = encoded(candidate.omega_rad_per_myr) == encoded(prior_motion)
    rows = [dict(coordinate_index=index, actual_force_residual_bounds=outward(residual),
        proved_own_operand_scale_lower=outward(scale), relative_residual_upper=outward(relative),
        own_coordinate_tolerance=outward(tolerance), own_coordinate_budget_passed=relative <= tolerance)
        for index, (residual, scale, relative) in enumerate(zip(candidate.residual_bounds,
            candidate.operand_scales, candidate.relative_residual_upper))]
    body = dict(format="native-spherical-solid-full-reference-face-source-control-1", status="PASS",
        before_state_sha256=before.state_sha256, current_load_sha256=load.load_sha256,
        original_unaccepted_trial_support_sha256=old_support.support_sha256,
        full_reference_face_support_sha256=full_support.support_sha256,
        source_epoch_sha256=full_support.source_epoch.epoch_sha256,
        complete_actual_source_payload_sha256={name: hashlib.sha256(encoded(source[name])).hexdigest()
            for name in ("before", "current_load", "original_unaccepted_trial_birth_support", "full_reference_face_birth_support")},
        accepted_input_complete_typed_state_unchanged=True,
        packet_sha256=packet["packet_sha256"], full_face_first_birth_declaration=payload(branch),
        actual_source_spherical_geometry=geometry, actual_source_face_coverage_and_robin=faces,
        per_coordinate_actual_finite_born_force_balance=rows,
        maximum_actual_own_coordinate_relative_residual=outward(max(candidate.relative_residual_upper)),
        full_face_actual_radial_correction_iterations=candidate.iterations,
        finite_born_reference_area_with_original_affine_force_equilibrium_certified=True,
        distributed_curved_physical_face_force_equilibrium_certified=False,
        full_finite_pose_network_equilibrium_certified=False,
        global_generalized_coordinate_energy_closure_certified=False,
        literal_owner_motion_matches_previous_finite_force_trial=motion_matches,
        previous_surface_pair_classification_inherited=False,
        all_42_surface_pairs_reexecuted=False,
        whole_spherical_partition_embedding_certified=False,
        normal_column_and_contact_deformation_glued=False,
        whole_clock_material_admission_authorized=False,
        accepted_original_native_intervals=1, accepted_new_native_intervals=0,
        original_point_1_completed=False, original_point_1_remaining_work_blocks=1,
        stage_seconds=stages, seconds=time.perf_counter()-started,
        remaining_physical_conditions=[
            "joint spherical bank and normal-column/contact deformation with matching curved-face force and strain-energy law",
            "whole-clock all-pair material admission on the resulting literal motion",
            "adopted mixed continuation, balances and joint restart"])
    body["report_sha256"] = hashlib.sha256(encoded(body)).hexdigest()
    write(out/"report.json", body)
    print("SPHERICAL native control PASS", body["seconds"], flush=True)
    return body
