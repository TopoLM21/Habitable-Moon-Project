"""Full accepted-interval reaction of a tied compact computational seed.

The endpoint chart is extended hypothetically over the saved last interval.
There was no historical compact seed during that interval.  Its added modes
and elastic memory are zero, while the old accepted response is preserved.
The reaction includes inertia and the saved time-average elastic strain, not
endpoint stress or surface traction.  It does not select or fund a fracture.

Untrusted planetary saves must first pass the public deterministic slip replay.
An optional typed partition is a validated immutable computational operator;
this module binds its chart, old memory, material and parent operators to that
trusted context without repeating its volume quadrature.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
import hashlib
import json
import math

import numpy as np

from .genesis import SECONDS_PER_MYR
from .genesis_volume_checkpoint import _canonical
from .genesis_volume_dynamics import coefficient_arrays
from .genesis_volume_embedded_partition import (
    GenesisVolumeEmbeddedPartition, _parent_response_binding, _plain,
    prepare_embedded_partition, validate_volume_embedded_partition)
from .genesis_volume_finite_seed import (
    prepare_volume_finite_seed, validate_volume_finite_seed,
    volume_finite_seed_to_dict)
from .genesis_volume_operator import (
    ANGULAR_BARYCENTRIC, _immutable, evaluate_genesis_volume_operator)
from .genesis_volume_slip_checkpoint import volume_slip_checkpoint_to_dict
from .genesis_volume_slip_dynamics import VolumeSlipModel, VolumeSlipState


FORMAT = "genesis-accepted-interval-tied-compact-seed-reaction-0.1"
SCOPE = "hypothetical_frozen_endpoint_compact_basis_tied_over_last_accepted_interval"
FALSE_CLAIMS = ("physical_creation_admitted", "physical_release_admitted",
    "funding_admitted", "historical_seed_claimed", "source_memory_reinitialized",
    "surface_traction_replacement_admitted", "activation_accepted")


class VolumeSeedReactionRefusal(ValueError):
    """The compact reaction lost its accepted material/interval binding."""


def _seal(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _same(actual, expected):
    return _canonical(_plain(actual)) == _canonical(_plain(expected))


def _real(value, shape, name):
    def boolean(item):
        return (isinstance(item, (bool, np.bool_)) or
                isinstance(item, (list, tuple)) and any(boolean(x) for x in item))
    if boolean(value):
        raise VolumeSeedReactionRefusal(f"{name} requires finite real values")
    raw = np.asarray(value)
    if raw.dtype.kind not in "ifu" or raw.shape != shape:
        raise VolumeSeedReactionRefusal(f"{name} has invalid real shape")
    result = np.asarray(raw, float)
    if not np.isfinite(result).all():
        raise VolumeSeedReactionRefusal(f"{name} requires finite real values")
    return result


def _exact(actual, expected, name):
    if not np.array_equal(actual, expected):
        raise VolumeSeedReactionRefusal(f"{name} differs from accepted source")


def _accepted_interval(model, state):
    if type(model) is not VolumeSlipModel or type(state) is not VolumeSlipState:
        raise VolumeSeedReactionRefusal("Reaction requires trusted accepted Genesis model/state")
    try:
        packet = volume_slip_checkpoint_to_dict(model, state)
    except (ValueError, TypeError) as error:
        raise VolumeSeedReactionRefusal(str(error)) from error
    if not state.accepted_step_sizes_myr or type(state.last_report) is not dict:
        raise VolumeSeedReactionRefusal("Reaction requires an actual accepted interval")
    report = state.last_report
    audit, slip = report.get("audit_input"), report.get("weak_plane_slip")
    if type(audit) is not dict or type(slip) is not dict:
        raise VolumeSeedReactionRefusal("Accepted interval lacks full average response witness")
    names = ("before_time_myr", "after_time_myr", "dt_myr", "requested_dt_myr")
    if any(type(report.get(k)) not in (int, float) or not math.isfinite(report[k]) for k in names):
        raise VolumeSeedReactionRefusal("Accepted duration/clock requires finite real values")
    if (report["after_time_myr"] != state.time_myr
            or report["requested_dt_myr"] != state.accepted_step_sizes_myr[-1]
            or report["dt_myr"] != state.time_myr-report["before_time_myr"]
            or report["dt_myr"] <= 0
            or not state.thermal.intervals
            or report["before_time_myr"] != state.thermal.intervals[-1].before_time_myr
            or report["after_time_myr"] != state.thermal.intervals[-1].after_time_myr
            or state.thermal.intervals[-1].requested_after_time_myr !=
                report["before_time_myr"]+report["requested_dt_myr"]):
        raise VolumeSeedReactionRefusal("Reaction interval lost accepted clock/step binding")
    dt = report["dt_myr"]*SECONDS_PER_MYR
    if not math.isfinite(dt) or dt <= 0:
        raise VolumeSeedReactionRefusal("Accepted reaction duration must be positive finite")
    parent, basal = model._assembly(state.geometry, state.fault_history.damage)
    inc = _real(audit.get("increment_m"), (parent.ndof,), "Accepted increment")
    v0 = _real(audit.get("before_velocity_m_s"), (parent.ndof,), "Accepted initial velocity")
    q0 = _real(audit.get("before_displacement_m"), (parent.ndof,), "Accepted initial displacement")
    _exact(q0+inc, state.displacement_m, "Accepted interval displacement endpoint")
    _exact(2*inc/dt-v0, state.velocity_m_s, "Accepted interval velocity endpoint")
    _exact(audit.get("basis_xyz"), parent.basis_xyz, "Accepted interval material basis")
    _exact(audit.get("elasticity_pa"), parent.C, "Accepted interval elasticity")
    _exact(audit.get("basal_traction_at_vertices_pa"), basal.basal_traction_at_vertices_pa,
           "Accepted interval external traction")
    if audit.get("basal_drag_pa_s_m") != model.contact_parameters.basal_drag_pa_s_m:
        raise VolumeSeedReactionRefusal("Accepted interval basal drag differs from source")
    if not _same(audit.get("after_cohorts"), state.geometry.cohorts):
        raise VolumeSeedReactionRefusal("Accepted interval endpoint material differs from source")
    shape = (len(parent.cohort_ids), 3, 6)
    mean0 = _real(slip.get("average_strain0"), shape, "Accepted average strain0")
    mean1 = _real(slip.get("average_strain1"), shape, "Accepted average strain1")
    declared_residual = _real(report.get("momentum_equation_relative_residual"), (),
                              "Accepted momentum residual")
    if declared_residual < 0:
        raise VolumeSeedReactionRefusal("Accepted momentum residual must be nonnegative")
    return packet, report, parent, basal, dt, inc, v0, mean0, mean1


def _source_partition(state, parent, basal, seed, partition, order):
    validate_volume_finite_seed(seed)
    s = seed.surface
    if not s.source_geometry_bound:
        raise VolumeSeedReactionRefusal("Reaction seed requires accepted original material geometry")
    rebuilt = prepare_volume_finite_seed(state.geometry, state.displacement_m,
        s.declared_normal_xyz, s.declared_offset_m, seed.seed_center_world_m,
        seed.seed_radius_m, cohort_id=seed.selected_cohort_id,
        quadrature_order=s.quadrature_order)
    if not _same(volume_finite_seed_to_dict(seed), volume_finite_seed_to_dict(rebuilt)):
        raise VolumeSeedReactionRefusal("Reaction seed differs from accepted material chart")
    e0, e1 = coefficient_arrays(state.mechanics)
    if partition is None:
        partition = prepare_embedded_partition(parent, seed, e0, e1, basal=basal,
                                                quadrature_order=order)
    if type(partition) is not GenesisVolumeEmbeddedPartition:
        raise VolumeSeedReactionRefusal("Reaction requires a typed compact partition")
    validate_volume_embedded_partition(partition)
    p, local = partition, partition.local
    ci = p.selected_cohort_index
    if (p.parent_response_binding_sha256 != _parent_response_binding(parent)
            or p.source_checkpoint_sha256 != parent.source_checkpoint_sha256
            or p.source_identity != parent.source_identity or p.time_myr != state.time_myr
            or p.parent_ndof != parent.ndof or p.ndof != parent.ndof+18
            or p.selected_cohort_id != seed.selected_cohort_id
            or not 0 <= ci < len(parent.cohort_ids)
            or parent.cohort_ids[ci] != seed.selected_cohort_id
            or local.quadrature_order != order):
        raise VolumeSeedReactionRefusal("Partition source, clock, selected cohort or order differs")
    if not _same(local.section, s.section_records[0]):
        raise VolumeSeedReactionRefusal("Partition frozen material chart differs from accepted seed")
    for actual, expected, name in (
        (p.selected_parent_dofs, parent.dofs[ci], "Partition original DOFs"),
        (p.memory0, e0, "Partition complete original memory0"),
        (p.memory1, e1, "Partition complete original memory1"),
        (local.memory0, e0[ci], "Partition selected memory0"),
        (local.memory1, e1[ci], "Partition selected memory1"),
        (local.elasticity_pa, parent.C[ci], "Partition selected elasticity"),
        (local.basis_xyz, parent.basis_xyz[ci], "Partition selected basis"),
        (local.seed_center_world_m, seed.seed_center_world_m, "Partition seed center")):
        _exact(actual, expected, name)
    if local.seed_radius_m != seed.seed_radius_m:
        raise VolumeSeedReactionRefusal("Partition seed radius differs from accepted seed")
    n = parent.ndof
    if p.M is None or p.D is None or p.external_force_n is None:
        raise VolumeSeedReactionRefusal("Reaction requires complete parent mass/drag/forcing")
    cross_m, cross_k = np.zeros((n, 18)), np.zeros((n, 18))
    cross_m[p.selected_parent_dofs] = local.mass_cross
    cross_k[p.selected_parent_dofs] = local.stiffness_cross
    endpoint = evaluate_genesis_volume_operator(parent, np.zeros(n),
        elastic_constant=e0, elastic_inverse_scale=e1)
    for actual, expected, name in (
        (p.K[:n, :n].toarray(), parent.K.toarray(), "Partition parent stiffness"),
        (p.M[:n, :n].toarray(), basal.mass_matrix.toarray(), "Partition parent mass"),
        (p.M[:n, n:].toarray(), cross_m, "Partition mass cross"),
        (p.M[n:, :n].toarray(), cross_m.T, "Partition mass transpose cross"),
        (p.M[n:, n:].toarray(), local.mass_enriched, "Partition added mass"),
        (p.K[:n, n:].toarray(), cross_k, "Partition stiffness cross"),
        (p.K[n:, :n].toarray(), cross_k.T, "Partition stiffness transpose cross"),
        (p.K[n:, n:].toarray(), local.stiffness_enriched, "Partition added stiffness"),
        (p.D[:n, :n].toarray(), basal.basal_drag_matrix.toarray(), "Partition parent drag"),
        (p.D[n:, :].toarray(), np.zeros((18, n+18)), "Partition added drag"),
        (p.D[:, n:].toarray(), np.zeros((n+18, 18)), "Partition drag cross"),
        (p.external_force_n, np.r_[basal.external_force_n, np.zeros(18)], "Partition forcing"),
        (p.initial_force_n, np.r_[endpoint.force, local.enrichment_force_n], "Partition endpoint force"),
        (p.R.toarray(), np.r_[np.eye(n), np.zeros((18, n))], "Partition tied prolongation"),
        (p.J.toarray(), np.c_[np.zeros((18, n)), np.eye(18)], "Partition added constraints")):
        _exact(actual, expected, name)
    if p.initial_elastic_energy_j != endpoint.energy_j:
        raise VolumeSeedReactionRefusal("Partition initial energy differs from accepted source")
    return p


def _average_added_force(p, mean0, mean1):
    """Integrate the archived frozen regular G with saved average memory."""
    local, ci = p.local, p.selected_cohort_index
    ld = np.longdouble
    angular = np.asarray(local.barycentric, ld)@np.asarray(np.linalg.inv(ANGULAR_BARYCENTRIC), ld)
    mean = angular@np.asarray(mean0[ci], ld)+(angular@np.asarray(mean1[ci], ld))/np.asarray(local.radial_scale, ld)[:, None]
    stress = mean@np.asarray(local.elasticity_pa.T, ld)
    return np.asarray(np.einsum("pai,pa,p->i", np.asarray(local.enrichment_strain_operator, ld),
        stress, np.asarray(local.reference_volume_weights_m3, ld)), float)


@dataclass(frozen=True, slots=True)
class GenesisVolumeSeedReaction:
    partition: GenesisVolumeEmbeddedPartition
    payload_json: str
    dt_s: float
    parent_increment_m: np.ndarray
    parent_before_velocity_m_s: np.ndarray
    interval_average_elastic_constant: np.ndarray
    interval_average_elastic_inverse_scale: np.ndarray
    enrichment_increment_m: np.ndarray
    enrichment_before_velocity_m_s: np.ndarray
    enrichment_elastic_memory_m: np.ndarray
    parent_inertia_force_n: np.ndarray
    parent_drag_force_n: np.ndarray
    parent_average_internal_force_n: np.ndarray
    parent_external_force_n: np.ndarray
    parent_residual_n: np.ndarray
    added_inertia_force_n: np.ndarray
    added_drag_force_n: np.ndarray
    added_average_internal_force_n: np.ndarray
    added_external_force_n: np.ndarray
    added_residual_n: np.ndarray
    endpoint_static_constraint_reaction_n: np.ndarray
    constraint_multipliers_n: np.ndarray
    constraint_force_n: np.ndarray
    remaining_residual_n: np.ndarray
    binding_sha256: str

    def __post_init__(self):
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, np.ndarray):
                object.__setattr__(self, item.name, _immutable(value))

    @property
    def physical_release_admitted(self):
        return False


def _body(value):
    return {**json.loads(value.payload_json), **{
        item.name: _plain(getattr(value, item.name)) for item in fields(value)
        if item.name not in {"partition", "payload_json", "binding_sha256"}}}


def prepare_volume_seed_reaction(model, state, finite_seed, *, partition=None,
                                quadrature_order=8):
    """Preserve old momentum residual and cancel only the tied added residual."""
    packet, report, parent, basal, dt, inc, v0, mean0, mean1 = _accepted_interval(model, state)
    p = _source_partition(state, parent, basal, finite_seed, partition, quadrature_order)
    inertia = 2*(basal.mass_matrix@(inc-dt*v0))/(dt*dt)
    drag = basal.basal_drag_matrix@inc/dt
    internal = evaluate_genesis_volume_operator(parent, np.zeros(parent.ndof),
        elastic_constant=mean0, elastic_inverse_scale=mean1).force
    external = basal.external_force_n
    parent_residual = inertia+drag+internal-external
    relative = float(np.linalg.norm(parent_residual)/max(np.linalg.norm(internal), np.linalg.norm(external), 1.))
    if relative > model.contact_parameters.equilibrium_tolerance:
        raise VolumeSeedReactionRefusal("Accepted average witness fails parent momentum equation")
    n = parent.ndof
    added_inertia = 2*(p.M[n:, :n]@(inc-dt*v0))/(dt*dt)
    added_internal = _average_added_force(p, mean0, mean1)
    added_drag, added_external = np.zeros(18), np.zeros(18)
    added_residual = added_inertia+added_drag+added_internal-added_external
    multiplier = -added_residual
    force = np.r_[np.zeros(n), multiplier]
    remaining = np.r_[parent_residual, added_residual]+force
    lifted_increment, lifted_velocity = p.R@inc, p.R@v0
    _exact(p.J@lifted_increment, np.zeros(18), "Canonical tied increment")
    _exact(p.J@lifted_velocity, np.zeros(18), "Canonical tied initial velocity")
    _exact(p.R.T@force, np.zeros(n), "Constraint force parent pullback")
    _exact(p.R.T@remaining, parent_residual, "Remaining parent residual pullback")
    work = float(force@lifted_increment)
    jump_work = float(multiplier@(p.J@lifted_increment))
    if work != 0. or jump_work != 0.:
        raise VolumeSeedReactionRefusal("Tied compact reaction produced work")
    runtime = packet["runtime_state"]
    context = dict(format=FORMAT, scope=SCOPE,
        accepted_parent_checkpoint_sha256=packet["checkpoint_sha256"],
        formation_checkpoint_sha256=packet["source_checkpoint_sha256"],
        source_identity=parent.source_identity, time_myr=state.time_myr,
        before_time_myr=report["before_time_myr"], after_time_myr=state.time_myr,
        accepted_step_sizes_myr=list(state.accepted_step_sizes_myr),
        accepted_final_report_sha256=_seal(runtime["last_report"]),
        full_memory_sha256=_seal(runtime["mechanics"]),
        birth_history_sha256=_seal(runtime["history"]),
        fault_history_sha256=_seal(runtime["fault_history"]),
        endpoint_geometry_sha256=_seal(runtime["geometry"]),
        finite_seed_binding_sha256=finite_seed.binding_sha256,
        embedded_partition_binding_sha256=p.binding_sha256,
        partition_quadrature_order=quadrature_order,
        selected_cohort_id=p.selected_cohort_id, parent_ndof=n,
        recomputed_parent_momentum_relative_residual=relative,
        parent_momentum_relative_residual=report["momentum_equation_relative_residual"],
        parent_operator_blocks_preserved_exactly=True,
        parent_residual_preserved_exactly=True, constraint_force_parent_pullback_exact_zero=True,
        added_residual_after_constraints_exact_zero=True,
        tied_constraint_work_j=work, exact_canonical_jump_work_j=jump_work,
        added_mode_basal_drag_and_external_force_exact_zero=True,
        mass_convention="after_mass_with_accepted_start_comoving_velocity",
        average_response_scope="saved_last_accepted_interval_not_endpoint_static_stress",
        basis_scope="accepted_endpoint_frozen_computational_basis_hypothetically_extended_over_saved_interval",
        added_memory_scope="new_zero_compact_coefficients_old_calculated_memory_unchanged",
        reaction_semantics="negative_added_inertia_plus_accepted_average_internal_residual",
        surface_force_is_dynamic_reaction=False, surface_traction_replacement=None,
        new_accepted_intervals=0, new_material_births=0, released_physical_dofs=0,
        transition_time_increment_myr=0., creation_work_debited_from_Genesis_j=0.,
        **{key: False for key in FALSE_CLAIMS})
    payload = {**context, "reaction_context_sha256": _seal(context)}
    value = GenesisVolumeSeedReaction(p, _canonical(payload).decode(), dt, inc, v0,
        mean0, mean1, np.zeros(18), np.zeros(18), np.zeros(18), inertia, drag,
        internal, external, parent_residual, added_inertia, added_drag,
        added_internal, added_external, added_residual, -p.local.enrichment_force_n,
        multiplier, force, remaining, "")
    return replace(value, binding_sha256=_seal(_body(value)))


def validate_volume_seed_reaction(value, *, partition=None):
    if (type(value) is not GenesisVolumeSeedReaction
            or value.binding_sha256 != _seal(_body(value))):
        raise VolumeSeedReactionRefusal("Seed reaction lost its immutable accepted interval binding")
    validate_volume_embedded_partition(value.partition)
    if partition is not None:
        validate_volume_embedded_partition(partition)
        if partition.binding_sha256 != value.partition.binding_sha256:
            raise VolumeSeedReactionRefusal("Seed reaction belongs to a different compact partition")
    raw = json.loads(value.payload_json)
    if (raw.get("format") != FORMAT or raw.get("scope") != SCOPE
            or raw.get("reaction_context_sha256") != _seal({k: v for k, v in raw.items() if k != "reaction_context_sha256"})
            or raw.get("embedded_partition_binding_sha256") != value.partition.binding_sha256
            or any(raw.get(k) is not False for k in FALSE_CLAIMS)
            or any(raw.get(k) != 0 for k in ("new_accepted_intervals", "new_material_births", "released_physical_dofs",
                "transition_time_increment_myr", "creation_work_debited_from_Genesis_j"))):
        raise VolumeSeedReactionRefusal("Seed reaction context cannot admit physical birth, release or funding")
    return value


def volume_seed_reaction_to_dict(value):
    validate_volume_seed_reaction(value)
    return {**_body(value), "seed_reaction_sha256": value.binding_sha256}


def volume_seed_reaction_from_dict(raw, *, model, state, finite_seed, partition=None,
                                  quadrature_order=8):
    """Verify the complete descriptor against a trusted accepted reconstruction."""
    value = prepare_volume_seed_reaction(model, state, finite_seed,
        partition=partition, quadrature_order=quadrature_order)
    try:
        equal = type(raw) is dict and _canonical(raw) == _canonical(volume_seed_reaction_to_dict(value))
    except (TypeError, ValueError, OverflowError):
        equal = False
    if not equal:
        raise VolumeSeedReactionRefusal("Seed reaction differs from complete accepted-source recomputation")
    return value
