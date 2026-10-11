"""Accepted-interval reactions and weak traction on prospective volume banks.

The complete trusted Genesis checkpoint supplies the accepted average Maxwell
and weak-plane response.  Endpoint stress is not substituted for that response.
All material, memory, forcing and clocks remain unchanged.  Supplied edge banks
and their radial surfaces are computational supports, not selected fractures.

The area-L2 traction reconstructed here is one discrete representative of the
tie reaction.  It is neither pointwise stress times a normal nor instantaneous
birth traction; it does not insert a cohesive law, release ties or admit motion.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
import hashlib
import json
import math

import numpy as np

from .genesis import SECONDS_PER_MYR
from .genesis_volume_banks import (
    GenesisVolumeBanks, _hash_value, evaluate_volume_banks, validate_volume_banks)
from .genesis_volume_checkpoint import _canonical
from .genesis_volume_fracture_surface import (
    integrate_surface_traction, validate_volume_fracture_surface)
from .genesis_volume_operator import _immutable, evaluate_genesis_volume_operator
from .genesis_volume_slip_checkpoint import volume_slip_checkpoint_to_dict
from .genesis_volume_slip_dynamics import VolumeSlipModel, VolumeSlipState


REACTION_FORMAT = "genesis-accepted-interval-volume-bank-reaction-0.1"
TRACTION_FORMAT = "genesis-accepted-interval-volume-reaction-traction-0.1"
SCOPE = "accepted_interval_average_on_computational_support_not_instantaneous_fracture_release"
FORCE_TOLERANCE = 5e-10
SVD_RCOND_FACTOR = 64.


class VolumeReactionRefusal(ValueError):
    """The supplied context cannot support this accepted-interval witness."""


class VolumeReactionRangeRefusal(VolumeReactionRefusal):
    """The material surface cannot replace the complete tied-bank reaction."""


def _digest(value):
    digest = hashlib.sha256()
    _hash_value(digest, value)
    return digest.hexdigest()


def _seal(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _real(value, shape, name):
    if isinstance(value, (bool, np.bool_)):
        raise VolumeReactionRefusal(f"{name} requires finite real values")
    if isinstance(value, (list, tuple)):
        def contains_boolean(item):
            return (isinstance(item, (bool, np.bool_)) or
                    isinstance(item, (list, tuple)) and any(contains_boolean(x) for x in item))
        if contains_boolean(value):
            raise VolumeReactionRefusal(f"{name} requires finite real values")
    raw = np.asarray(value)
    if raw.dtype.kind not in "ifu" or raw.shape != shape:
        raise VolumeReactionRefusal(f"{name} has invalid shape or real dtype")
    array = np.asarray(raw, float)
    if not np.isfinite(array).all():
        raise VolumeReactionRefusal(f"{name} requires finite real values")
    return array


def _force_families(actual, expected, *, scales=None, tolerance=FORCE_TOLERANCE):
    """Keep small depth P forces observable separately from the A forces."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or actual.ndim != 1 or actual.size % 6:
        raise VolumeReactionRefusal("Full six-DOF force families have incompatible dimensions")
    a, b = actual.reshape(-1, 6), expected.reshape(-1, 6)
    result = []
    for sl in (slice(0, 3), slice(3, 6)):
        scale = float(np.linalg.norm(b[:, sl]))
        if scales is not None:
            scale = max(scale, *(float(np.linalg.norm(np.asarray(v).reshape(-1, 6)[:, sl])) for v in scales))
        scale = max(scale, np.finfo(float).tiny)
        error = float(np.linalg.norm(a[:, sl]-b[:, sl]))
        result.append(error/scale)
        if error > tolerance*scale:
            raise VolumeReactionRefusal("Full A/P force identity differs from accepted interval")
    return tuple(result)


@dataclass(frozen=True, slots=True)
class GenesisVolumeBankReaction:
    banks: GenesisVolumeBanks
    payload_json: str
    dt_s: float
    parent_increment_m: np.ndarray
    parent_before_velocity_m_s: np.ndarray
    interval_average_elastic_constant: np.ndarray
    interval_average_elastic_inverse_scale: np.ndarray
    bank_increment_m: np.ndarray
    bank_before_velocity_m_s: np.ndarray
    parent_residual_n: np.ndarray
    bank_inertia_force_n: np.ndarray
    bank_drag_force_n: np.ndarray
    bank_average_internal_force_n: np.ndarray
    bank_external_force_n: np.ndarray
    bank_residual_n: np.ndarray
    constraint_multipliers_n: np.ndarray
    constraint_force_n: np.ndarray
    remaining_residual_n: np.ndarray
    binding_sha256: str

    def __post_init__(self):
        for item in fields(self):
            if isinstance(getattr(self, item.name), np.ndarray):
                object.__setattr__(self, item.name, _immutable(getattr(self, item.name)))

    @property
    def physical_release_admitted(self):
        return False


def prepare_volume_bank_reaction(model, state, banks):
    """Bind a trusted accepted interval, retaining its small parent residual.

    An untrusted archive must first undergo the existing complete deterministic
    checkpoint replay.  This entry point neither advances time nor activates
    faults.  The final accepted report, not a private predictor, supplies the
    average strain and thermal-first/comoving mass convention.
    """
    if type(model) is not VolumeSlipModel or type(state) is not VolumeSlipState:
        raise VolumeReactionRefusal("Reaction requires trusted accepted Genesis volume slip context")
    packet = volume_slip_checkpoint_to_dict(model, state)
    if not state.accepted_step_sizes_myr or type(state.last_report) is not dict:
        raise VolumeReactionRefusal("Reaction requires an actual accepted interval, not an initial state")
    report = state.last_report
    audit = report.get("audit_input")
    slip = report.get("weak_plane_slip")
    if type(audit) is not dict or type(slip) is not dict:
        raise VolumeReactionRefusal("Accepted interval lacks its full average response witness")
    if (report.get("after_time_myr") != state.time_myr
            or report.get("requested_dt_myr") != state.accepted_step_sizes_myr[-1]
            or type(report.get("before_time_myr")) not in (int, float)
            or type(report.get("dt_myr")) not in (int, float)
            or report["dt_myr"] != state.time_myr-report["before_time_myr"]
            or report["dt_myr"] <= 0):
        raise VolumeReactionRefusal("Reaction interval lost accepted clock/step binding")
    dt = report["dt_myr"]*SECONDS_PER_MYR
    if not math.isfinite(dt) or dt <= 0:
        raise VolumeReactionRefusal("Accepted reaction duration must be finite and positive")
    parent, basal = model._assembly(state.geometry, state.fault_history.damage)
    validate_volume_banks(banks, parent_assembly=parent)
    if banks.basal is None or _digest(banks.basal.parent_basal) != _digest(basal):
        raise VolumeReactionRefusal("Reaction banks changed accepted mass, basal drag or forcing")
    inc = _real(audit.get("increment_m"), (parent.ndof,), "Accepted increment")
    v0 = _real(audit.get("before_velocity_m_s"), (parent.ndof,), "Accepted initial velocity")
    q0 = _real(audit.get("before_displacement_m"), (parent.ndof,), "Accepted initial displacement")
    if (not np.array_equal(q0+inc, state.displacement_m)
            or not np.array_equal(2*inc/dt-v0, state.velocity_m_s)):
        raise VolumeReactionRefusal("Accepted interval increment or momenta differ from endpoint")
    shape = (len(parent.cohort_ids), 3, 6)
    mean0 = _real(slip.get("average_strain0"), shape, "Accepted average strain0")
    mean1 = _real(slip.get("average_strain1"), shape, "Accepted average strain1")
    parent_mean = evaluate_genesis_volume_operator(parent, np.zeros(parent.ndof),
        elastic_constant=mean0, elastic_inverse_scale=mean1).force
    parent_inertia = 2*(basal.mass_matrix@(inc-dt*v0))/(dt*dt)
    parent_drag = basal.basal_drag_matrix@inc/dt
    parent_residual = parent_inertia+parent_drag+parent_mean-basal.external_force_n
    parent_relative = float(np.linalg.norm(parent_residual)/max(
        np.linalg.norm(parent_mean), np.linalg.norm(basal.external_force_n), 1.))
    if parent_relative > model.contact_parameters.equilibrium_tolerance:
        raise VolumeReactionRefusal("Accepted average witness does not satisfy parent momentum equation")
    ds, v0s = banks.R@inc, banks.R@v0
    if np.any(banks.J@ds) or np.any(banks.J@v0s):
        raise VolumeReactionRefusal("Accepted bank interval is not exactly tied")
    fbar = evaluate_volume_banks(banks, np.zeros(banks.ndof),
        elastic_constant=mean0, elastic_inverse_scale=mean1).force
    inertia = 2*(banks.basal.mass_matrix@(ds-dt*v0s))/(dt*dt)
    drag = banks.basal.basal_drag_matrix@ds/dt
    external = banks.basal.external_force_n
    residual = inertia+drag+fbar-external
    parent_scales = (parent_inertia, parent_drag, parent_mean, basal.external_force_n)
    pullback_errors = _force_families(banks.R.T@residual, parent_residual,
                                    scales=parent_scales)
    J = banks.J.toarray()
    multipliers = (-np.linalg.solve(J@J.T, J@residual)
                   if len(J) else np.zeros(0))
    force = J.T@multipliers
    remaining = residual+force
    null_errors = _force_families(banks.R.T@force, np.zeros(parent.ndof),
                                scales=parent_scales)
    remaining_errors = _force_families(banks.R.T@remaining, parent_residual,
                                     scales=parent_scales)
    work = float(force@ds)
    jump_work = float(multipliers@(J@ds))
    work_scale = max(float(np.linalg.norm(force)*np.linalg.norm(ds)), 1.)
    if abs(work) > 128*np.finfo(float).eps*work_scale or jump_work != 0:
        raise VolumeReactionRefusal("Tied bank reaction produced nonzero work")
    body = dict(format=REACTION_FORMAT, scope=SCOPE,
        accepted_parent_checkpoint_sha256=packet["checkpoint_sha256"],
        formation_checkpoint_sha256=packet["source_checkpoint_sha256"],
        source_identity=parent.source_identity, time_myr=state.time_myr,
        before_time_myr=report["before_time_myr"], after_time_myr=state.time_myr,
        accepted_step_sizes_myr=list(state.accepted_step_sizes_myr),
        accepted_final_report_sha256=_seal(packet["runtime_state"]["last_report"]),
        full_memory_sha256=_seal(packet["runtime_state"]["mechanics"]),
        birth_history_sha256=_seal(packet["runtime_state"]["history"]),
        fault_history_sha256=_seal(packet["runtime_state"]["fault_history"]),
        bank_binding_sha256=banks.binding_sha256,
        cohort_ids=[int(x) for x in parent.cohort_ids],
        force_pullback_relative_errors_A_P=list(pullback_errors),
        constraint_force_null_relative_errors_A_P=list(null_errors),
        remaining_pullback_relative_errors_A_P=list(remaining_errors),
        parent_momentum_relative_residual=report["momentum_equation_relative_residual"],
        recomputed_parent_momentum_relative_residual=parent_relative,
        tied_constraint_work_j=work, exact_canonical_jump_work_j=jump_work,
        mass_convention="after_mass_with_accepted_start_comoving_velocity",
        traction_scope="last_accepted_interval_average_not_instantaneous_endpoint",
        source_calculated_creation_law=None, physical_creation_admitted=False,
        physical_release_admitted=False, calculated_physical_fracture_surface=False,
        transition_time_increment_myr=0., creation_work_debited_from_Genesis_j=0.,
        source_memory_reinitialized=False, parent_residual_preserved=True,
    )
    payload = {**body, "reaction_context_sha256": _seal(body)}
    value = GenesisVolumeBankReaction(banks, _canonical(payload).decode(), dt,
        inc, v0, mean0, mean1, ds, v0s, parent_residual, inertia, drag, fbar,
        external, residual, multipliers, force, remaining, "")
    return replace(value, binding_sha256=_digest(value))


def validate_volume_bank_reaction(reaction, *, banks=None):
    if (type(reaction) is not GenesisVolumeBankReaction
            or reaction.binding_sha256 != _digest(reaction)):
        raise VolumeReactionRefusal("Bank reaction differs from its accepted interval binding")
    validate_volume_banks(reaction.banks)
    if banks is not None:
        validate_volume_banks(banks)
        if banks.binding_sha256 != reaction.banks.binding_sha256:
            raise VolumeReactionRefusal("Reaction belongs to different source-bound banks")
    payload = json.loads(reaction.payload_json)
    if (payload.get("format") != REACTION_FORMAT or payload.get("scope") != SCOPE
            or payload.get("reaction_context_sha256") != _seal({k: v for k, v in payload.items()
                                                              if k != "reaction_context_sha256"})):
        raise VolumeReactionRefusal("Bank reaction context seal or scope differs")
    return reaction


@dataclass(frozen=True, slots=True)
class GenesisVolumeReactionTraction:
    reaction: GenesisVolumeBankReaction
    surface: object
    world_traction_pa: np.ndarray
    replacement_force_n: np.ndarray
    replacement_residual_n: np.ndarray
    canonical_trace: np.ndarray
    modal_scale: np.ndarray
    singular_values: np.ndarray
    rank: int
    svd_relative_cutoff: float
    range_relative_error_A_P: tuple
    force_relative_error_A_P: tuple
    area_l2_traction_squared_pa2_m2: float
    tied_replacement_work_j: float
    binding_sha256: str

    def __post_init__(self):
        for item in fields(self):
            if isinstance(getattr(self, item.name), np.ndarray):
                object.__setattr__(self, item.name, _immutable(getattr(self, item.name)))

    @property
    def physical_release_admitted(self):
        return False


def prepare_volume_reaction_traction(reaction, surface):
    """Replace the complete reaction by minimum-area-L2 weak traction.

    B=H J factors the physical world trace into independent canonical jumps.
    Row scaling normalises the thin-depth P amplitudes before SVD but does not
    change the physical L2 minimisation.  The unscaled complete force is checked;
    an unsupported nonzero reaction is refused rather than projected away.
    """
    validate_volume_bank_reaction(reaction)
    validate_volume_fracture_surface(surface, banks=reaction.banks)
    banks = reaction.banks
    J = banks.J.toarray()
    rows = 12*surface.patch_count
    B = np.zeros((rows, banks.ndof))
    for patch, dofs in enumerate(surface.trace_dofs):
        # Shared tips can repeat an index.  Assemble, do not overwrite columns.
        local = surface.jump_trace[patch].reshape(12, 24)
        for slot, dof in enumerate(dofs):
            B[12*patch:12*(patch+1), int(dof)] += local[:, slot]
    weights = np.repeat(surface.area_weights_m2.ravel(), 3)
    if np.any(weights <= 0) or not np.isfinite(weights).all():
        raise VolumeReactionRefusal("Reaction traction requires positive material area weights")
    Jplus = (np.linalg.solve(J@J.T, J).T if len(J) else np.zeros((banks.ndof, 0)))
    H = B@Jplus
    factor_error = np.linalg.norm(B-H@J)
    if (np.any(B@banks.R.toarray())
            or factor_error > 128*np.finfo(float).eps*max(np.linalg.norm(B), 1.)):
        raise VolumeReactionRefusal("Physical surface trace does not factor through tied bank jumps")
    if len(J):
        # Each canonical row contains duplicate +1 at the appended DOF.
        families = np.array([np.flatnonzero(row > 0)[0] % 6 >= 3 for row in J])
        depth = 1.-float(np.min(surface.lambda_bounds[:, 0])) if surface.patch_count else 0.
        scale = np.where(families, 1./depth if depth > 0 else 1., 1.)
    else:
        families, scale = np.zeros(0, bool), np.zeros(0)
    sqrt_weights = np.sqrt(weights)
    L = scale[:, None]*H.T*sqrt_weights[None, :]
    target = scale*reaction.constraint_multipliers_n
    cutoff = SVD_RCOND_FACTOR*np.finfo(float).eps*max(L.shape, default=0)
    if L.size:
        U, singular, Vt = np.linalg.svd(L, full_matrices=False)
        rank = int(np.sum(singular > cutoff*singular[0])) if singular[0] > 0 else 0
        z = (Vt[:rank].T@((U[:, :rank].T@target)/singular[:rank])
             if rank else np.zeros(rows))
    else:
        singular, rank, z = np.zeros(0), 0, np.zeros(rows)
    projected = L@z
    range_errors = []
    for mask in (~families, families):
        error, norm = float(np.linalg.norm((projected-target)[mask])), float(np.linalg.norm(target[mask]))
        # A zero target family has an absolute ROUNDING bound from the full
        # scaled target, not a tunable physical force or compliance floor.
        rounding = 256*np.finfo(float).eps*max(float(np.linalg.norm(target)), np.finfo(float).tiny)
        range_errors.append(error/max(norm, rounding))
        if error > FORCE_TOLERANCE*norm+rounding:
            raise VolumeReactionRangeRefusal("Surface trace cannot replace the complete A/P bank reaction")
    if rank == 0 and np.any(reaction.constraint_multipliers_n):
        raise VolumeReactionRangeRefusal("An empty or rank-zero surface cannot replace a nonzero reaction")
    traction = (z/sqrt_weights).reshape(surface.patch_count, 4, 3)
    force = integrate_surface_traction(surface, traction)
    residual = force-reaction.constraint_force_n
    # Work-consistent canonical comparison already prevents suppressed P
    # modes; give the unscaled A/P force errors separately as well.
    h = reaction.constraint_force_n
    force_errors = []
    for sl in (slice(0, 3), slice(3, 6)):
        error = float(np.linalg.norm(residual.reshape(-1, 6)[:, sl]))
        norm = float(np.linalg.norm(h.reshape(-1, 6)[:, sl]))
        full = float(np.linalg.norm((scale*reaction.constraint_multipliers_n)))
        family_scale = float(np.max(scale[families if sl.start == 3 else ~families], initial=1.))
        rounding = 1024*np.finfo(float).eps*max(full/family_scale, np.finfo(float).tiny)
        force_errors.append(error/max(norm, rounding))
        if error > FORCE_TOLERANCE*norm+rounding:
            raise VolumeReactionRangeRefusal("Reconstructed surface force left an unresolved A/P reaction")
    work = float(force@reaction.bank_increment_m)
    bound = 1024*np.finfo(float).eps*max(float(np.linalg.norm(force)*np.linalg.norm(reaction.bank_increment_m)), 1.)
    if abs(work) > bound:
        raise VolumeReactionRefusal("Tied reaction replacement produced nonzero work")
    l2 = float(z@z)
    value = GenesisVolumeReactionTraction(reaction, surface, traction, force, residual,
        H, scale, singular, rank, cutoff, tuple(range_errors), tuple(force_errors),
        l2, work, "")
    return replace(value, binding_sha256=_digest(value))


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_plain(x) for x in value]
    return value


def volume_bank_reaction_to_dict(reaction):
    validate_volume_bank_reaction(reaction)
    body = json.loads(reaction.payload_json)
    body.update({item.name: _plain(getattr(reaction, item.name))
                 for item in fields(reaction) if item.name not in ("banks", "payload_json", "binding_sha256")})
    body["prepared_reaction_binding_sha256"] = reaction.binding_sha256
    return {**body, "reaction_sha256": _seal(body)}


def volume_reaction_traction_to_dict(traction):
    if (type(traction) is not GenesisVolumeReactionTraction
            or traction.binding_sha256 != _digest(traction)):
        raise VolumeReactionRefusal("Reaction traction differs from its prepared material binding")
    validate_volume_bank_reaction(traction.reaction)
    validate_volume_fracture_surface(traction.surface, banks=traction.reaction.banks)
    body = dict(format=TRACTION_FORMAT, scope=SCOPE,
        reaction_binding_sha256=traction.reaction.binding_sha256,
        surface_binding_sha256=traction.surface.binding_sha256,
        accepted_parent_checkpoint_sha256=json.loads(traction.reaction.payload_json)["accepted_parent_checkpoint_sha256"],
        representative="minimum_reference_area_L2_world_traction",
        canonical_trace_semantics="world_surface_trace_equals_H_times_canonical_J",
        modal_scaling="A_one_P_reciprocal_one_minus_minimum_reference_lambda",
        physical_creation_admitted=False, physical_release_admitted=False,
        pointwise_stress_normal_recovered=False, instantaneous_birth_traction_claimed=False,
        transition_time_increment_myr=0., creation_work_debited_from_Genesis_j=0.,
        prepared_traction_binding_sha256=traction.binding_sha256)
    body.update({item.name: _plain(getattr(traction, item.name))
                 for item in fields(traction) if item.name not in ("reaction", "surface", "binding_sha256")})
    return {**body, "traction_sha256": _seal(body)}


def validate_volume_bank_reaction_payload(raw, *, expected_reaction):
    """Compare an untrusted DTO to a freshly prepared trusted interval."""
    expected = volume_bank_reaction_to_dict(expected_reaction)
    if type(raw) is not dict or _canonical(raw) != _canonical(expected):
        raise VolumeReactionRefusal("Reaction DTO differs from the accepted interval context")
    return expected


def validate_volume_reaction_traction_payload(raw, *, expected_traction):
    """A new DTO seal cannot supply a missing reaction or physical admission."""
    expected = volume_reaction_traction_to_dict(expected_traction)
    if type(raw) is not dict or _canonical(raw) != _canonical(expected):
        raise VolumeReactionRefusal("Traction DTO differs from the accepted interval/surface context")
    return expected
