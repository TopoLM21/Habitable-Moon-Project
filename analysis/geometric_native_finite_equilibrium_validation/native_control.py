"""Source-bound actual born finite forces, with the original affine ports.

This is a force preflight, not an adopted material state or a solid embedding.
The affine continuum result is exported only as the preserved initial solve;
the corrected five-term responses and histories have a separate format.
"""
from __future__ import annotations

from dataclasses import fields, is_dataclass
from decimal import Decimal
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import time

from tectonics.geometric_native_pose import FractionInterval as I, _rounded_interval

ROOT = Path(__file__).resolve().parents[2]


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()


def outward(value):
    bound = _rounded_interval(value if type(value) is I else I(value), 96)
    return {"lo": [bound.lo.numerator, bound.lo.denominator],
            "hi": [bound.hi.numerator, bound.hi.denominator]}


def payload(value):
    """Keep decimal coordinates literal; compress rational bounds outward."""
    if type(value) in (F, I):
        return outward(value)
    if is_dataclass(value):
        return {row.name: payload(getattr(value, row.name)) for row in fields(value)}
    if isinstance(value, (tuple, list)):
        return [payload(item) for item in value]
    if isinstance(value, dict):
        return {str(name): payload(item) for name, item in value.items()}
    return value


def write(path, value):
    with Path(path).open("xb") as stream:
        stream.write(encoded(value)+b"\n")


def run_control(out, *, case=None):
    from analysis.geometric_native_birth_front_validation.native_continuum_case import build_native_continuum_case
    from tectonics.geometric_native_birth_finite_equilibrium import solve_native_born_finite_equilibrium
    from tectonics.geometric_native_birth_exact_radial import (
        normalize_native_birth_radial_before, radial_before_normalization_to_dict)
    from tectonics.geometric_native_birth_contact_continuum import birth_continuum_network_to_dict
    from tectonics.geometric_native_implicit_underthrust_motion import native_implicit_underthrust_motion_to_dict
    from tectonics.geometric_native_implicit_mantle_loading import native_implicit_spatial_mantle_load_to_dict
    from tectonics.geometric_native_birth_contact_support import native_birth_contact_support_to_dict
    from tectonics.geometric_native_birth_contact import born_foundation_state_to_dict
    from tectonics.geometric_native_physical_history import _typed_fingerprint

    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    stages = {}
    def timed(name, callback):
        tick = time.perf_counter()
        print("FINITE control", name, "starting", flush=True)
        value = callback()
        stages[name] = time.perf_counter()-tick
        print("FINITE control", name, "finished", round(stages[name], 3), flush=True)
        return value
    if case is None:
        case = timed("actual_source", build_native_continuum_case)
    before, load, support = case["previous"], case["load"], case["support"]
    source_fingerprints = tuple(_typed_fingerprint(value) for value in (before, load, support))
    candidate = timed("actual_finite_equilibrium", lambda: solve_native_born_finite_equilibrium(case))
    if (not candidate.force_equilibrium_certified or candidate.material_admission_authorized
            or candidate.before_state_sha256 != before.state_sha256
            or candidate.birth_support_sha256 != support.support_sha256):
        raise ValueError("The corrected force candidate differs from its genuine source or scope")
    inputs, normalization = normalize_native_birth_radial_before(case["inputs"], before)
    if normalization != candidate.exact_radial_normalization:
        raise ValueError("The corrected forces do not use the actual saved radial banks")
    initial = candidate.source_affine_step
    old_dimension = len(inputs["old_stiffness"])
    if candidate.coordinates_decimal[:old_dimension] != initial.coordinates_decimal[:old_dimension]:
        raise ValueError("The bounded born radial correction changed an original authoritative coordinate")
    if len(candidate.residual_bounds) != 19 or len(candidate.responses) != 2:
        raise ValueError("The genuine nineteen-coordinate/two-arc case changed")
    tolerance = F(5, 10**10)
    if any(value > tolerance for value in candidate.relative_residual_upper):
        raise ValueError("An actual force residual exceeds its own coordinate budget")
    dt = F.from_float(case["dt_s"])
    for previous_point, (_, current_point) in zip(inputs["old_states"], initial.original_point_states):
        if (current_point.elapsed_time_s != previous_point.elapsed_time_s+case["dt_s"]
                or current_point.step_count != previous_point.step_count+1):
            raise ValueError("The original contact history was reset or advanced more than once")
    for attachment, ledger in zip(inputs["born_continuum_attachments"], candidate.born_contact_ledgers):
        prior = attachment.previous_state
        if (ledger.previous_state != prior or ledger.activation_paid_again
                or ledger.activation_energy_decimal != prior.activation_energy_decimal
                or ledger.elapsed_time_s != F.from_float(prior.elapsed_time_s)+dt
                or ledger.step_count != prior.step_count+1):
            raise ValueError("The finite contact trial reset memory, paid activation twice or changed its clock")
    for attachment, step in zip(inputs["born_foundations"], candidate.corrected_foundation_steps):
        if (step.state.elapsed_time_s != attachment.previous_state.elapsed_time_s+case["dt_s"]
                or step.state.step_count != attachment.previous_state.step_count+1):
            raise ValueError("The corrected independent foundation has a different clock")
    source = timed("complete_source_and_initial_force_replay", lambda: dict(
        before=native_implicit_underthrust_motion_to_dict(before),
        current_load=native_implicit_spatial_mantle_load_to_dict(case["mesh"], before.current_surface, load),
        birth_support=native_birth_contact_support_to_dict(support),
        canonical_before_affine_initial_solve=birth_continuum_network_to_dict(initial, **inputs)))
    finite = dict(format="native-original-affine-ports-and-born-finite-forces-1",
        force_model="unchanged original affine ports; actual finite-SO(3) born normal potentials",
        coordinates_decimal=candidate.coordinates_decimal,
        omega_rad_per_myr=candidate.omega_rad_per_myr,
        literal_owner_axes=payload(candidate.literal_owner_axes),
        angular_path_error_bounds=payload(candidate.angular_path_error_bounds),
        born_responses=payload(candidate.responses), born_finite_fields=payload(candidate.fields),
        born_finite_jacobians=payload(candidate.jacobians),
        born_finite_contact_trial_ledgers=payload(candidate.born_contact_ledgers),
        born_finite_contact_trial_histories=payload(tuple(ledger.history for ledger in candidate.born_contact_ledgers)),
        ledger_work_definition="conservative normal path work is delta potential; backward-Euler normal-gap work is recorded separately",
        corrected_foundation_states=[born_foundation_state_to_dict(step.state,
            expected_parameters=item.parameters, expected_reference=item.reference)
            for item, step in zip(inputs["born_foundations"], candidate.corrected_foundation_steps)],
        corrected_foundation_steps=payload(candidate.corrected_foundation_steps),
        original_point_states=payload(initial.original_point_states),
        original_anchor_fields=payload(initial.original_anchor_fields),
        residual_bounds=payload(candidate.residual_bounds),
        operand_scales=payload(candidate.operand_scales),
        relative_residual_upper=payload(candidate.relative_residual_upper),
        exact_radial_normalization=radial_before_normalization_to_dict(normalization),
        corrected_radial_coordinate_indices=candidate.corrected_radial_coordinate_indices,
        iterations=candidate.iterations, precision=candidate.precision,
        relative_tolerance=outward(candidate.relative_tolerance),
        finite_born_with_original_affine_force_equilibrium_certified=True,
        full_finite_pose_network_equilibrium_certified=False,
        global_generalized_coordinate_energy_closure_certified=False,
        whole_clock_material_admission_authorized=False, accepted_new_native_intervals=0)
    packet = dict(format="native-born-finite-equilibrium-source-packet-1", **source,
        corrected_finite_force_trial=finite,
        affine_initial_solve_is_corrected_finite_history=False,
        accepted_input_unchanged=True, material_admission_authorized=False)
    packet["packet_sha256"] = hashlib.sha256(encoded(packet)).hexdigest()
    write(out/"candidate_packet.json", packet)
    prior_path = ROOT/"analysis/geometric_native_birth_front_validation/control_release_01/candidate_packet.json"
    prior = json.loads(prior_path.read_bytes())
    literal_motion_matches = encoded(candidate.omega_rad_per_myr) == encoded(prior["omega_rad_per_myr"]) and case["dt_s"] == prior["dt_s"]
    if tuple(_typed_fingerprint(value) for value in (before, load, support)) != source_fingerprints:
        raise ValueError("The trial changed the accepted before state, current source load or birth support")
    rows = [dict(coordinate_index=index, actual_force_residual_bounds=outward(residual),
        proved_own_operand_scale_lower=outward(scale), relative_residual_upper=outward(relative),
        own_coordinate_tolerance=outward(tolerance), own_coordinate_budget_passed=relative <= tolerance)
        for index, (residual, scale, relative) in enumerate(zip(candidate.residual_bounds,
            candidate.operand_scales, candidate.relative_residual_upper))]
    body = dict(format="native-born-finite-equilibrium-source-control-1", status="PASS",
        before_state_sha256=before.state_sha256, current_load_sha256=load.load_sha256,
        birth_support_sha256=support.support_sha256, source_epoch_sha256=support.source_epoch.epoch_sha256,
        complete_actual_source_payload_sha256={name: hashlib.sha256(encoded(source[name])).hexdigest()
            for name in ("before", "current_load", "birth_support")},
        accepted_input_complete_typed_state_unchanged=True,
        packet_sha256=packet["packet_sha256"],
        original_authoritative_coordinates_preserved=True,
        canonical_radial_before_maximum_physical_quotient_error_m=outward(normalization.maximum_physical_quotient_error_bound_m),
        per_coordinate_actual_finite_born_force_balance=rows,
        maximum_actual_own_coordinate_relative_residual=outward(max(candidate.relative_residual_upper)),
        born_endpoint_finite_gap_bounds_m=[outward(row.gap_bounds_m) for row in candidate.responses],
        born_endpoint_finite_energy_bounds_j=[outward(row.energy_bounds_j) for row in candidate.responses],
        paid_activation_preserved_once=True, contact_foundation_and_original_clocks_preserved=True,
        actual_radial_correction_iterations=candidate.iterations,
        finite_born_with_original_affine_force_equilibrium_certified=True,
        full_finite_pose_network_equilibrium_certified=False,
        global_generalized_coordinate_energy_closure_certified=False,
        literal_owner_motion_matches_previous_all_pair_control=literal_motion_matches,
        previous_surface_pair_classification_inherited=literal_motion_matches,
        all_42_surface_pairs_reexecuted=False,
        global_spherical_solid_embedding_certified=False,
        whole_clock_material_admission_authorized=False,
        accepted_original_native_intervals=1, accepted_new_native_intervals=0,
        original_point_1_completed=False, original_point_1_remaining_work_blocks=1,
        stage_seconds=stages, seconds=time.perf_counter()-started,
        remaining_physical_conditions=[
            "source-bound whole-mask spherical layer/edge embedding with column/contact-zone placement",
            "complete physical contact face coverage and generalized-coordinate energy closure",
            "whole-clock all-pair material admission, adopted mixed continuation and joint restart"])
    body["report_sha256"] = hashlib.sha256(encoded(body)).hexdigest()
    write(out/"report.json", body)
    print("FINITE native control PASS", body["seconds"], flush=True)
    return body
