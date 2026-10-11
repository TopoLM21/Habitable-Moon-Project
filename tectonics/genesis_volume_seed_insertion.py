"""Bind a finite constrained enrichment to the calculated Genesis history.

The parent displacement locates material; it is never added again to saved
elastic memory.  The compact extra modes remain constrained to zero.  A
surface prestress seed and an endpoint bulk constraint reaction are distinct
quantities, and neither provides an accepted dynamic release or its funding.
Untrusted planetary archives must first pass the public slip checkpoint replay.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np

from .genesis_volume_checkpoint import _canonical
from .genesis_volume_embedded_partition import (
    prepare_embedded_partition, volume_embedded_partition_to_dict,
)
from .genesis_volume_embedded_prestress import (
    pull_back_embedded_prestress, embedded_prestress_pullback_to_dict,
)
from .genesis_volume_finite_seed import (
    prepare_volume_finite_seed, validate_volume_finite_seed,
    volume_finite_seed_to_dict,
)
from .genesis_volume_material_stress import (
    bind_volume_material_stress, evaluate_volume_material_stress,
    volume_material_stress_observer_to_dict, volume_material_stress_to_dict,
)
from .genesis_volume_nucleation import (
    bind_volume_nucleation_proposal, volume_nucleation_proposal_to_dict,
)
from .genesis_volume_slip_checkpoint import volume_slip_checkpoint_to_dict
from .genesis_volume_slip_dynamics import VolumeSlipModel, VolumeSlipState


FORMAT = "genesis-source-bound-finite-seed-constrained-insertion-0.1"
SCOPE = "calculated_memory_finite_material_seed_and_constrained_compact_bulk_modes"
FALSE_CLAIMS = ("physical_creation_admitted", "physical_release_admitted",
    "funding_admitted", "history_reinitialized", "material_memory_changed",
    "activation_accepted", "clock_advanced", "dynamic_bank_reaction_replaced",
    "historical_three_dimensional_memory_recovered")


class VolumeSeedInsertionRefusal(ValueError):
    """Finite support, operator or history differs from the accepted source."""


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _equal(actual, expected):
    try:
        return type(actual) is dict and _canonical(actual) == _canonical(expected)
    except (TypeError, ValueError, OverflowError):
        return False


@dataclass(frozen=True, slots=True)
class BoundVolumeSeedInsertion:
    payload_json: str
    binding_sha256: str


def _bound(body):
    raw = {**body, "seed_insertion_sha256": _digest(body)}
    payload = _canonical(raw).decode()
    return BoundVolumeSeedInsertion(payload, hashlib.sha256(payload.encode()).hexdigest())


def _source_context(model, state, observer=None):
    if type(model) is not VolumeSlipModel or type(state) is not VolumeSlipState:
        raise VolumeSeedInsertionRefusal("Seed insertion requires accepted Genesis model/state")
    expected = bind_volume_material_stress(model, state)
    if observer is not None and not _equal(volume_material_stress_observer_to_dict(observer),
            volume_material_stress_observer_to_dict(expected)):
        raise VolumeSeedInsertionRefusal("Seed insertion observer differs from accepted source recomputation")
    checkpoint = volume_slip_checkpoint_to_dict(model, state)
    context = dict(format=FORMAT, scope=SCOPE,
        parent_checkpoint_sha256=checkpoint["checkpoint_sha256"],
        formation_checkpoint_sha256=checkpoint["source_checkpoint_sha256"],
        source_identity=state.geometry.source_identity, time_myr=state.time_myr,
        accepted_steps=state.mechanics.accepted_steps,
        accepted_step_sizes_myr=list(state.accepted_step_sizes_myr),
        observer_binding_sha256=expected.binding_sha256,
        inherited_runtime_ledgers={k: v for k, v in checkpoint["runtime_state"].items()
                                  if k.endswith("_j")},
        source_historical_surface_creation_energy_law=None,
        selected_sharp_material_surface=None,
        historical_dissipation_available_for_creation_j=0.,
        creation_work_debited_from_Genesis_j=0.,
        new_accepted_intervals=0, new_material_births=0, released_physical_dofs=0,
        **{k: False for k in FALSE_CLAIMS})
    return expected, context


def _surface_enrichment(seed):
    """Trace of the SAME compact bulk basis, not two whole-cohort banks."""
    s = seed.surface
    section = s.section_records[0]
    N, lam = s.barycentric, s.radial_scale
    inner, outer = float(section.lambda_bounds[0]), float(section.lambda_bounds[1])
    t = (lam-inner)/(outer-inner)
    r2 = np.sum((s.current_position_m-seed.seed_center_world_m)**2, axis=1)
    sphere = np.maximum(1.-r2/seed.seed_radius_m**2, 0.)
    bubble = sphere**2 * (27.*np.prod(N, axis=1)) * (4.*t*(1.-t))
    if not np.isfinite(bubble).all() or np.any(bubble < -128*np.finfo(float).eps):
        raise VolumeSeedInsertionRefusal("Finite seed enrichment left its original material support")
    return bubble, bubble[:, None, None]*s.parent_trace


def _validate_material_positions(surface, samples):
    """Compare the two source placements at their arithmetic resolution.

    The section sums N@(lambda*(X+P)+(A-P)); the observer distributes
    lambda over its angular sum. Near a Cartesian cancellation, roundoff
    scales with those operands, rather than the small resulting coordinate.
    This guard follows the exact accepted-source reconstruction above it.
    """
    eps = np.finfo(float).eps
    lam, sampled_lam = surface.radial_scale, samples.radial_scale
    radial_error = np.abs(sampled_lam-lam)
    if np.any(radial_error > 128*eps*np.maximum(np.abs(lam), 1.)):
        raise VolumeSeedInsertionRefusal("Finite seed calculated material radial coordinates differ")
    for section in surface.section_records:
        selected = surface.cohort_ids == section.cohort_id
        N = np.abs(surface.barycentric[selected])
        X = np.abs(section.triangle_m)
        A, P = np.abs(section.displacement_A_m), np.abs(section.slope_P_m)
        radial = np.abs(lam[selected, None])
        delta = radial_error[selected, None]
        current_direction, reference_direction = N@(X+P), N@X
        current_scale = radial*current_direction+N@(A+P)
        reference_scale = radial*reference_direction
        for actual, expected, scale, direction in (
            (samples.current_position_m, surface.current_position_m,
             current_scale, current_direction),
            (samples.reference_position_m, surface.reference_position_m,
             reference_scale, reference_direction)):
            tolerance = 128*eps*np.maximum(scale, 1.)+delta*direction
            if np.any(np.abs(actual[selected]-expected[selected]) > tolerance):
                raise VolumeSeedInsertionRefusal("Finite seed calculated material positions differ")


def prepare_volume_seed_insertion(model, state, finite_seed, *, observer=None,
        creation_law=None, partition_quadrature_order=8):
    """Source-bound COMPUTATIONAL seed; all added bulk amplitudes stay zero.

    The explicit world seed is a probe until the prospective source gate
    selects it.  Nonzero enriched paths, additional strain memory, energy
    funding and full-time contact admission are still required for release.
    """
    observer, context = _source_context(model, state, observer)
    validate_volume_finite_seed(finite_seed)
    surface = finite_seed.surface
    if not surface.source_geometry_bound:
        raise VolumeSeedInsertionRefusal("Seed insertion requires original accepted material geometry")
    rebuilt = prepare_volume_finite_seed(state.geometry, state.displacement_m,
        surface.declared_normal_xyz, surface.declared_offset_m,
        finite_seed.seed_center_world_m, finite_seed.seed_radius_m,
        cohort_id=finite_seed.selected_cohort_id,
        quadrature_order=surface.quadrature_order)
    if not _equal(volume_finite_seed_to_dict(finite_seed), volume_finite_seed_to_dict(rebuilt)):
        raise VolumeSeedInsertionRefusal("Finite seed differs from accepted current material support")
    samples = evaluate_volume_material_stress(observer, surface.cohort_ids,
        surface.barycentric, surface.mass_coordinate_q)
    _validate_material_positions(surface, samples)
    pullback = pull_back_embedded_prestress(surface, samples.stress_world_tensor,
        creation_law=creation_law)
    assembly, basal = model._assembly(state.geometry, state.fault_history.damage)
    partition = prepare_embedded_partition(assembly, finite_seed,
        observer.elastic_constant, observer.elastic_inverse_scale, basal=basal,
        quadrature_order=partition_quadrature_order)
    partition_raw = volume_embedded_partition_to_dict(partition)
    bubble, trace = _surface_enrichment(finite_seed)
    force = np.einsum("pai,pa,p->i", trace, pullback.reference_traction_world_pa,
                      surface.reference_area_weights_m2)
    if not np.isfinite(force).all():
        raise VolumeSeedInsertionRefusal("Finite seed surface virtual force overflowed")
    bulk_force = np.asarray(partition_raw["local"]["enrichment_force_n"], float)
    # phi itself belongs to the old A/P interpolation. H*w*phi therefore
    # enriches the continuous bulk but has zero jump on Gamma. Such modes
    # must not supply purported fracture funding by releasing numerical
    # refinement without surface work. Every extra amplitude is still tied.
    section = surface.section_records[0]
    scalar_A = (section.triangle_m+section.displacement_A_m)@section.normal_xyz-section.offset_m
    scalar_P = (section.triangle_m+section.slope_P_m)@section.normal_xyz
    null_modes = np.zeros((18, 3))
    for vertex in range(3):
        for axis in range(3):
            null_modes[6*vertex+axis, axis] = scalar_A[vertex]
            null_modes[6*vertex+3+axis, axis] = scalar_P[vertex]
    null_modes /= np.linalg.norm(null_modes, axis=0)
    weighted_trace = (np.sqrt(surface.reference_area_weights_m2)[:, None, None]*trace).reshape(-1, 18)
    column_scale = np.maximum(np.linalg.norm(weighted_trace, axis=0), np.finfo(float).tiny)
    jump_rank = int(np.linalg.matrix_rank(weighted_trace/column_scale))
    null_residual = np.linalg.norm(weighted_trace@null_modes, axis=0)
    body = {**context,
        "status": "constrained_computational_material_seed_no_physical_birth",
        "geometry_provenance": "explicit_computational_seed_not_source_activation",
        "finite_seed": volume_finite_seed_to_dict(finite_seed),
        "source_stress_samples": volume_material_stress_to_dict(samples),
        "prestress_pullback": embedded_prestress_pullback_to_dict(pullback),
        "embedded_partition": partition_raw,
        "partition_quadrature_order": partition_quadrature_order,
        "surface_enrichment_bubble": bubble.tolist(),
        "surface_enrichment_trace": trace.tolist(),
        "surface_enrichment_force_n": force.tolist(),
        "analytic_surface_null_modes": null_modes.tolist(),
        "surface_null_jump_weighted_norm_m": null_residual.tolist(),
        "surface_jump_trace_rank": jump_rank,
        "surface_null_modes_remain_constrained": True,
        "volume_rank_does_not_certify_surface_opening": True,
        "null_mode_release_needs_separate_work_admission": True,
        "endpoint_elastic_constraint_reaction_n": (-bulk_force).tolist(),
        "constraint_minus_surface_force_n": (-bulk_force-force).tolist(),
        "reaction_difference_semantics": "retained_endpoint_bulk_divergence_and_quadrature_terms_not_zeroed",
        "surface_force_semantics": "positive_normal_reference_traction_dual_to_compact_jump",
        "constraint_reaction_semantics": "endpoint_elastic_added_mode_reaction_not_full_dynamic_reaction",
        "surface_force_is_discrete_bulk_reaction": False,
        "finite_positive_surface_area_prepared": True,
        "constrained_bulk_modes_prepared": True,
        "additional_memory_needed_before_nonzero_enrichment": True,
        "parameter_provenance": None if creation_law is None else
            "explicit_new_constitutive_input_not_historical_toughness"}
    return _bound(body)


def bind_proposed_volume_seed_insertion(model, state, law=None, *, quadrature_order=8,
        partition_quadrature_order=8):
    """Reach clipping/insertion only through the inherited Genesis birth gate."""
    observer, context = _source_context(model, state)
    proposal = volume_nucleation_proposal_to_dict(bind_volume_nucleation_proposal(
        model, state, law, observer=observer))
    candidate = proposal["candidate"]
    insertion = None
    if candidate is not None:
        seed = prepare_volume_finite_seed(state.geometry, state.displacement_m,
            candidate["normal_world"], candidate["plane_offset_m"],
            candidate["current_nucleation_point_m"], law.seed_radius_m,
            cohort_id=candidate["cohort_id"], quadrature_order=quadrature_order)
        insertion = volume_seed_insertion_to_dict(prepare_volume_seed_insertion(
            model, state, seed, observer=observer, creation_law=law.creation_law,
            partition_quadrature_order=partition_quadrature_order))
    return _bound({**context, "status": proposal["status"],
        "geometry_provenance": "prospective_source_gated_seed_not_physical_admission",
        "nucleation_proposal": proposal, "insertion": insertion,
        "finite_positive_surface_area_prepared": insertion is not None,
        "constrained_bulk_modes_prepared": insertion is not None,
        "additional_memory_needed_before_nonzero_enrichment": True})


def volume_seed_insertion_to_dict(value):
    if type(value) is not BoundVolumeSeedInsertion or value.binding_sha256 != hashlib.sha256(
            value.payload_json.encode()).hexdigest():
        raise VolumeSeedInsertionRefusal("Seed insertion lost its immutable source binding")
    raw = json.loads(value.payload_json)
    if raw.get("format") != FORMAT or raw.get("scope") != SCOPE or raw.get(
            "seed_insertion_sha256") != _digest({k: v for k, v in raw.items()
                                                if k != "seed_insertion_sha256"}):
        raise VolumeSeedInsertionRefusal("Seed insertion seal or scope differs")
    if any(raw.get(k) is not False for k in FALSE_CLAIMS):
        raise VolumeSeedInsertionRefusal("Constrained insertion cannot admit release, funding or changed history")
    if any(raw.get(k) != 0 for k in ("historical_dissipation_available_for_creation_j",
            "creation_work_debited_from_Genesis_j", "new_accepted_intervals",
            "new_material_births", "released_physical_dofs")) or raw.get(
            "selected_sharp_material_surface") is not None or raw.get(
            "source_historical_surface_creation_energy_law") is not None:
        raise VolumeSeedInsertionRefusal("Constrained insertion cannot invent physical creation or funding")
    return raw


def volume_seed_insertion_from_dict(raw, *, model, state, finite_seed, observer=None,
        creation_law=None, partition_quadrature_order=8):
    expected = prepare_volume_seed_insertion(model, state, finite_seed, observer=observer,
        creation_law=creation_law, partition_quadrature_order=partition_quadrature_order)
    if not _equal(raw, volume_seed_insertion_to_dict(expected)):
        raise VolumeSeedInsertionRefusal("Seed insertion differs from complete accepted-source recomputation")
    return expected


def proposed_volume_seed_insertion_from_dict(raw, *, model, state, law=None,
        quadrature_order=8, partition_quadrature_order=8):
    expected = bind_proposed_volume_seed_insertion(model, state, law,
        quadrature_order=quadrature_order, partition_quadrature_order=partition_quadrature_order)
    if not _equal(raw, volume_seed_insertion_to_dict(expected)):
        raise VolumeSeedInsertionRefusal("Proposed insertion differs from the accepted source birth gate")
    return expected
