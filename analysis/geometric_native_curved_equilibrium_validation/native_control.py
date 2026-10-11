"""Genuine coupled curved-face forces, preserving the accepted native source.

The new endpoint force certificate covers the declared reduced potential.
It grants no whole-clock solid admission or adopted material continuation.
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
from analysis.geometric_native_joint_spherical_validation.native_control import (
    PRIOR_RAW, prepare_source_context,
)

ROOT = Path(__file__).resolve().parents[2]
JOINT_PACKET = ROOT/"analysis/geometric_native_joint_spherical_validation/release_validation_01/native_control/candidate_packet.json"
JOINT_PACKET_RAW = "8580d79ec47263c9f581a646595ab19d87ae927770d082b3016803d4c405ac45"


def prepare_reference_field(context):
    from tectonics.geometric_native_joint_spherical_field import bind_native_joint_spherical_field
    tops = tuple((identity, context["before"][context["indices"][identity]])
        for identity in sorted(context["indices"]))
    return bind_native_joint_spherical_field(context["full_case"]["support"],
        top_radial_displacements_m=tops,
        provenance="One source-bound saved physical-radius reference flow per authentic contact-connected group; radial motion follows saved shear",
        precision=128)


def _force_payload(candidate):
    return dict(format="native-coupled-current-curved-face-endpoint-forces-1",
        force_model=candidate.force_model,
        coordinates_decimal=list(candidate.coordinates_decimal),
        omega_rad_per_myr=payload(candidate.omega_rad_per_myr),
        literal_owner_axes=payload(candidate.literal_owner_axes),
        corrected_coordinate_indices=list(candidate.corrected_coordinate_indices),
        iterations=candidate.iterations, integration_precision=candidate.integration_precision,
        exact_radial_normalization=payload(candidate.exact_radial_normalization),
        responses=payload(candidate.responses),
        mapped_curved_gradient_bounds=payload(candidate.mapped_curved_gradient_bounds),
        force_operand_rows=payload(candidate.force_operand_rows),
        force_certificate=payload(candidate.force_certificate),
        force_residual_bounds=payload(candidate.residual_bounds),
        own_operand_scale_lower_bounds=payload(candidate.operand_scales),
        own_relative_residual_upper_bounds=payload(candidate.relative_residual_upper),
        relative_tolerance=outward(candidate.relative_tolerance),
        original_point_states=payload(candidate.original_point_states),
        original_contact_steps=payload(candidate.original_contact_steps),
        corrected_foundation_steps=payload(candidate.corrected_foundation_steps),
        curved_first_birth_ledgers=payload(candidate.curved_first_birth_ledgers),
        previous_flat_force_certificate_inherited=False,
        previous_flat_activation_credit_reused=False,
        new_curved_history_adopted=False,
        complete_three_dimensional_strain_energy_certified=False,
        material_admission_authorized=False)


def run(out, *, case=None):
    from analysis.geometric_native_birth_front_validation.native_continuum_case import build_native_continuum_case
    from tectonics.geometric_native_curved_equilibrium import solve_native_curved_equilibrium
    from tectonics.geometric_native_physical_history import _typed_fingerprint
    from tectonics.thermal import SECONDS_PER_MYR
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    stages = {}
    def timed(name, callback):
        tick = time.perf_counter()
        print("CURVED control", name, "starting", flush=True)
        result = callback()
        stages[name] = time.perf_counter()-tick
        print("CURVED control", name, "finished", round(stages[name], 3), flush=True)
        return result
    if case is None:
        case = timed("genuine_source", build_native_continuum_case)
    objects = tuple(case[name] for name in ("previous", "load", "support"))
    preserved = tuple(_typed_fingerprint(value) for value in objects)
    context = timed("complete_genuine_source_binding", lambda: prepare_source_context(case))
    full_case = context["full_case"]
    field = timed("saved_joint_reference_field", lambda: prepare_reference_field(context))
    prior_raw = JOINT_PACKET.read_bytes()
    if hashlib.sha256(prior_raw).hexdigest() != JOINT_PACKET_RAW:
        raise ValueError("The independently authenticated previous joint reference packet changed")
    prior_joint = json.loads(prior_raw)
    if (field.declaration_precision != 128
            or encoded(payload(field.groups)) != encoded(prior_joint["joint_reference_geometry"]["groups"])
            or any(encoded(value) != encoded(prior_joint[name]) for name, value in context["source"].items())):
        raise ValueError("The new force trial changed the saved joint reference groups or their complete source")
    def progress(family, response):
        print("CURVED integral", family, "cells", response.integration_cell_count,
            "closed", response.closed_taylor_cell_count,
            "unresolved", response.unresolved_sign_cell_count,
            "depth orders", getattr(response, "used_depth_taylor_orders", ()),
            "own widths", response.own_coordinate_widths_satisfied, flush=True)
    def iteration_progress(iteration, residuals, scales):
        ratios = tuple(max(abs(value.lo), abs(value.hi))/scale
            for value, scale in zip(residuals, scales) if scale)
        uncertain_zero_rows = tuple(index for index, (value, scale) in enumerate(zip(residuals, scales))
            if not scale and (value.lo or value.hi))
        print("CURVED iteration", iteration, "max own relative residual",
            float(max(ratios, default=F(0))), "uncertain zero rows", uncertain_zero_rows, flush=True)
    candidate = timed("actual_nineteen_coordinate_curved_equilibrium",
        lambda: solve_native_curved_equilibrium(full_case, joint_field=field,
            integration_progress=progress, iteration_progress=iteration_progress))
    tolerance = F(5, 10**10)
    if (not candidate.force_equilibrium_certified or candidate.material_admission_authorized
            or candidate.constant_height_born_force_certificate_reused
            or candidate.before_state_sha256 != case["previous"].state_sha256
            or candidate.birth_support_sha256 != full_case["support"].support_sha256
            or len(candidate.relative_residual_upper) != 19 or len(candidate.responses) != 2
            or candidate.original_accepted_contact_count != 16
            or any(value > tolerance for value in candidate.relative_residual_upper)):
        raise ValueError("The genuine coupled curved trial lacks all nineteen own-coordinate force balances")
    dt_myr = F.from_float(case["dt_s"]/SECONDS_PER_MYR)
    literal_axes = tuple((owner, tuple(F.from_float(value)*dt_myr for value in omega))
        for owner, omega in candidate.omega_rad_per_myr)
    if literal_axes != candidate.literal_owner_axes:
        raise ValueError("The curved force trial substituted a different owner clock or angular motion")
    inputs = full_case["inputs"]
    for previous, (_, current) in zip(inputs["old_states"], candidate.original_point_states):
        if (current.elapsed_time_s != previous.elapsed_time_s+case["dt_s"]
                or current.step_count != previous.step_count+1):
            raise ValueError("An original accepted contact history was reset or advanced twice")
    for attachment, step in zip(inputs["born_foundations"], candidate.corrected_foundation_steps):
        if (step.state.elapsed_time_s != attachment.previous_state.elapsed_time_s+case["dt_s"]
                or step.state.step_count != attachment.previous_state.step_count+1):
            raise ValueError("A physical HOT foundation uses a different clock")
    for ledger in candidate.curved_first_birth_ledgers:
        if (ledger.elapsed_time_s != F.from_float(case["dt_s"])
                or ledger.step_count != 1 or ledger.activation_payment_count != 1
                or ledger.flat_activation_credit_reused or ledger.adopted_history_extended):
            raise ValueError("The virgin curved activation reused credit, reset an adopted history or changed its clock")
    if tuple(_typed_fingerprint(value) for value in objects) != preserved:
        raise ValueError("The new force solve changed the complete accepted source, load or original support")
    force = _force_payload(candidate)
    reference = dict(source_support_sha256=field.source_support_sha256,
        source_epoch_fingerprint_sha256=field.source_epoch_fingerprint_sha256,
        component_count=len(field.components), groups=payload(field.groups),
        continuity_graph_edges=payload(field.continuity_graph_edges),
        top_radial_displacements_m=payload(field.top_radial_displacements_m),
        declaration_precision=field.declaration_precision, provenance=field.provenance,
        authenticated_previous_reference_packet_raw_sha256=JOINT_PACKET_RAW,
        explicit_actual_bank_sources=payload(context["bank_sources"]),
        whole_mask_bending_field_reconstructed=False,
        reference_shear_fixed_during_radial_motion=True)
    packet = dict(format="native-curved-force-equilibrium-source-packet-1", **context["source"],
        authenticated_initial_motion_packet_raw_sha256=PRIOR_RAW,
        authenticated_initial_motion_packet_seal=context["source_packet_seal"],
        saved_joint_reference_field=reference, actual_curved_force_trial=force,
        accepted_input_unchanged=True, material_admission_authorized=False)
    packet["packet_sha256"] = hashlib.sha256(encoded(packet)).hexdigest()
    write(out/"candidate_packet.json", packet)
    body = dict(format="native-coupled-curved-force-equilibrium-control-1", status="PASS",
        before_state_sha256=case["previous"].state_sha256,
        current_load_sha256=case["load"].load_sha256,
        birth_support_sha256=field.source_support_sha256,
        accepted_input_complete_typed_state_unchanged=True,
        packet_sha256=packet["packet_sha256"], saved_joint_reference_field=reference,
        actual_curved_force_trial=force,
        native_curved_force_equilibrium_certified=True,
        previous_reference_force_certificate_inherited=False,
        complete_three_dimensional_strain_energy_certified=False,
        global_physical_solid_noninterpenetration_certified=False,
        whole_clock_material_admission_authorized=False,
        all_42_surface_pairs_reexecuted=False,
        previous_surface_pair_classification_inherited=False,
        accepted_original_native_intervals=1, accepted_new_native_intervals=0,
        original_point_1_completed=False, original_point_1_remaining_work_blocks=1,
        remaining_physical_conditions=[
            "normal-column deformation placed and glued in actual masks with three-dimensional strain energy",
            "whole-clock all-pair material admission on resulting literal motion",
            "adopted mixed continuation, balances and joint restart"],
        stage_seconds=stages, seconds=time.perf_counter()-started)
    body["report_sha256"] = hashlib.sha256(encoded(body)).hexdigest()
    write(out/"report.json", body)
    print("CURVED native control PASS", body["seconds"], flush=True)
    return body
