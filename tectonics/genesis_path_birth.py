"""Reaction-consistent traction diagnostics before cohesive path activation.

This module never releases a constraint or initializes cohesive history. The
two endpoint tractions at a path vertex are not separately determined by its
two reaction components. We retain a bulk-stress prior and make the smallest
area-weighted correction that recovers the measured constrained reaction.
The answer is an explicit reconstruction, not a uniquely measured traction.

Extrinsic insertion must recover traction at zero physical jump; insertion at
a later, supercritical checkpoint cannot preserve both the old equilibrium
and the specified strength without a further physical model. References:
https://arxiv.org/abs/cond-mat/0106318
https://www.sandia.gov/files/sierra/SM_Development_5_26/main/cohesive/ext.html
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np

from .genesis_contact_law import ContactLawParameters
from .genesis_material import face_frames


def _owned(value):
    array = np.ascontiguousarray(value)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _real(value, shape, name):
    array = np.asarray(value)
    if (array.shape != shape or array.dtype.kind not in "fiu"
            or not np.isfinite(array).all()):
        raise ValueError(f"{name} must be finite real values with shape {shape}")
    return np.asarray(array, dtype=float)


@dataclass(frozen=True)
class TiedTractionRecovery:
    traction_pa: np.ndarray
    stress_prior_pa: np.ndarray
    correction_pa: np.ndarray
    trace_observed: np.ndarray
    target_reaction_n: np.ndarray
    recovered_reaction_n: np.ndarray
    relative_residual: float
    prior_relative_residual: float
    relative_area_weighted_correction: float
    reaction_rank: int
    traction_nullity: int
    elapsed_years: float


@dataclass(frozen=True)
class TiedOnsetDiagnostic:
    normal_ratio: np.ndarray
    shear_ratio: np.ndarray
    shear_strength_pa: np.ndarray
    trace_observed: np.ndarray
    tensile_exceeded: np.ndarray
    shear_exceeded: np.ndarray
    maximum_observed_ratio: float
    status: str
    # Even an exactly resolved strength crossing needs an extrinsic law and
    # its own constitutive/energy history. This diagnostic authorizes no birth.
    physical_birth_ready: bool = False


@dataclass(frozen=True)
class TiedCapacityDiagnostic:
    direction: np.ndarray
    required_n: np.ndarray
    capacity_n: np.ndarray
    finite_capacity: np.ndarray
    unbounded_capacity: np.ndarray
    indeterminate_capacity: np.ndarray
    ratio: np.ndarray
    violations: np.ndarray


def _accepted_tied(model, state):
    model._validate_state(state)
    if state.accepted_steps < 1:
        raise ValueError("Traction recovery requires an accepted reaction measurement")
    if state.active_interval is not None:
        raise ValueError("Traction recovery requires all support banks tied")
    if state.stopped_reason is not None:
        raise ValueError("Traction recovery requires an admissible unstopped state")


def recover_tied_tractions(model, state, stress_pa):
    """Recover a work-conjugate traction field from an accepted, all-tied state.

    ``stress_pa`` is [xx, yy, xy] Cauchy stress in each reference child chord
    frame (not engineering shear). At each interface endpoint the prior is
    the arithmetic mean of the two adjacent world stress tensors, projected
    onto the reference great-circle normal and endpoint tangent. This is an
    objective chord-stress projection, not a claim of continuum traction
    continuity. Unequal bank depths still use the model's smaller-depth
    interface area; any mismatch is visible in the reported correction.

    If J maps enrichment displacement to trace jumps, A repeats each trace
    area for both components, and R is the measured constraint reaction,
    the correction is J (J.T A J)^-1 (R - J.T A t_prior). It minimizes
    sum A |t-t_prior|^2 subject to J.T A t = R. Unobservable support tips
    retain their prior and are excluded from the observed onset maximum.

    No state, bulk stress, strength, energy, displacement or law is changed.
    """
    _accepted_tied(model, state)
    basis, geometry = model.basis, model.geometry
    mesh = basis.subdivision.mesh
    stress = _real(stress_pa, (mesh.cell_count, 3), "stress_pa")
    local = np.zeros((mesh.cell_count, 2, 2))
    local[:, 0, 0], local[:, 1, 1] = stress[:, 0], stress[:, 1]
    local[:, 0, 1] = local[:, 1, 0] = stress[:, 2]
    frames = face_frames(mesh)
    world = np.einsum("fij,fjk,flk->fil", frames, local, frames)
    average = world[basis.topology.seam_faces].mean(axis=1).repeat(2, axis=0)
    normal, tangent = geometry.interface_normal, geometry.interface_tangent
    prior = np.column_stack((np.einsum("fi,fij,fj->f", normal, average, normal),
                             np.einsum("fi,fij,fj->f", tangent, average, normal)))
    jump = model.jump_operator[:, basis.nparent:].tocsr()
    area = np.repeat(geometry.interface_area_m2, 2)
    if (not len(area) or jump.shape[1] == 0 or jump.shape[1] % 2
            or not np.isfinite(area).all() or np.any(area <= 0)):
        raise ValueError("Traction recovery needs positive areas and observable bank motion")
    weighted_transpose = jump.T.multiply(area).tocsr()
    gram = (weighted_transpose@jump).tocoo()
    # Each paired endpoint depends on just its vertex's two enrichment DOFs.
    # Enforce that contract and solve 2x2 blocks without a dense path-size solve.
    if np.any((gram.row//2 != gram.col//2) & (gram.data != 0)):
        raise ValueError("Traction recovery requires local two-component bank enrichment")
    blocks = np.zeros((jump.shape[1]//2, 2, 2))
    np.add.at(blocks, (gram.row//2, gram.row % 2, gram.col % 2), gram.data)
    eigenvalues = np.linalg.eigvalsh(blocks)
    if (not np.isfinite(eigenvalues).all()
            or np.any(eigenvalues[:, 0] <= 1e-12*eigenvalues[:, 1])):
        raise ValueError("Tied traction reaction operator is rank deficient")
    target = state.constraint_reaction_n[basis.nparent:]
    prior_force = np.asarray(weighted_transpose@prior.ravel()).ravel()
    rhs = (target-prior_force).reshape(-1, 2)
    multipliers = np.linalg.solve(blocks, rhs[..., None])[..., 0].ravel()
    correction = np.asarray(jump@multipliers).reshape(-1, 2)
    traction = prior+correction
    recovered = np.asarray(weighted_transpose@traction.ravel()).ravel()
    scale = max(np.linalg.norm(target), np.linalg.norm(prior_force), 1.)
    residual = float(np.linalg.norm(recovered-target)/scale)
    if not np.isfinite(traction).all() or not np.isfinite(residual) or residual > 1e-10:
        raise ValueError("Recovered tractions do not balance the tied reaction")
    observed = np.asarray(jump.power(2).sum(axis=1)).ravel().reshape(-1, 2).sum(axis=1) > 0
    prior_norm = np.linalg.norm(np.sqrt(area)*prior.ravel())
    correction_norm = np.linalg.norm(np.sqrt(area)*correction.ravel())
    return TiedTractionRecovery(
        *map(_owned, (traction, prior, correction, observed, target, recovered)),
        residual, float(np.linalg.norm(prior_force-target)/scale),
        float(correction_norm/max(prior_norm, 1.)), jump.shape[1],
        jump.shape[0]-jump.shape[1], float(state.elapsed_years))


def classify_tied_onset(recovery, water_per_trace, law_parameters=None, *, relative_tolerance=1e-6):
    """Compare reconstructed observed tractions to unchanged local strengths.

    These are separate tensile and cohesive-Coulomb shear diagnostics, not a
    new mixed-mode fracture criterion. An exceeded strength means that a
    future event integrator needs an earlier bracket, not a raised strength.
    Ratios at support tips are supplied but not used to classify the event.
    """
    if not isinstance(recovery, TiedTractionRecovery):
        raise ValueError("Onset classification requires a tied traction recovery")
    if (isinstance(relative_tolerance, (bool, np.bool_))
            or not isinstance(relative_tolerance, Real)
            or not np.isfinite(relative_tolerance) or not 0 <= relative_tolerance < 1):
        raise ValueError("relative_tolerance must be finite in [0, 1)")
    traction = _real(recovery.traction_pa, (len(recovery.trace_observed), 2), "traction_pa")
    observed = np.asarray(recovery.trace_observed)
    if observed.dtype.kind != "b" or observed.shape != (len(traction),) or not observed.any():
        raise ValueError("Onset classification requires observed traces")
    water = _real(water_per_trace, (len(traction),), "water_per_trace")
    if np.any((water < 0) | (water > 1)):
        raise ValueError("water_per_trace must lie in [0, 1]")
    law = law_parameters or ContactLawParameters()
    law.validate()
    normal = np.maximum(traction[:, 0], 0)/law.tensile_strength_pa
    friction = law.friction_dry+(law.friction_wet-law.friction_dry)*water
    cohesion = law.cohesion_pa*(1-(1-law.wet_cohesion_fraction)*water)
    shear_strength = cohesion+friction*np.maximum(-traction[:, 0], 0)
    absolute_shear = np.abs(traction[:, 1])
    shear = np.divide(absolute_shear, shear_strength, out=np.zeros_like(absolute_shear), where=shear_strength > 0)
    shear[(shear_strength == 0) & (absolute_shear > 0)] = np.inf
    maximum = float(max(np.max(normal[observed]), np.max(shear[observed])))
    status = ("strength_exceeded" if maximum > 1+relative_tolerance else
              "below_strength" if maximum < 1-relative_tolerance else "on_strength_surface")
    return TiedOnsetDiagnostic(*map(_owned, (normal, shear, shear_strength, observed,
        (normal > 1+relative_tolerance) & observed,
        (shear > 1+relative_tolerance) & observed)), maximum, status)


def tied_reaction_capacity(model, state, water_per_trace, law_parameters=None, *,
                           coefficient_tolerance=1e-12, relative_tolerance=1e-10):
    """Necessary, prior-independent capacity certificate for each tied vertex.

    For reaction R choose the unit virtual bank displacement d=R/|R|. Each
    incident trace has normal/shear virtual jumps (a,b)=J_trace d. Over the
    unchanged admissible traction envelope

        t_n <= S_n,  |t_s| <= C_wet + mu*max(-t_n, 0),

    the work support is finite only if a >= mu*|b|. Its maximum is then
    area*(a*S_n + |b|*C_wet). If |R| exceeds the sum, NO admissible endpoint
    traction field can carry this reaction, including any reconstruction
    nullspace choice. This directional certificate is necessary, not
    sufficient: ratios <= 1 do not establish an admissible field.

    Negative margin gives unbounded compressive support, hence no certificate
    in this direction. Nonzero margins within coefficient_tolerance of zero
    are conservatively indeterminate rather than clamped to finite support.
    Exact zero margins are allowed. Zero reaction uses d=0 and ratio=0.
    Infinite or indeterminate capacity has ratio=0 and an explicit flag;
    positive required force with exactly zero finite capacity has ratio=inf.
    The pressure-dependent envelope is an onset test, not a finite-rate
    viscous admissibility law or proof of physical cohesive nucleation.
    """
    _accepted_tied(model, state)
    for name, value in (("coefficient_tolerance", coefficient_tolerance),
                        ("relative_tolerance", relative_tolerance)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                or not np.isfinite(value) or not 0 <= value < 1):
            raise ValueError(f"{name} must be finite in [0, 1)")
    b = model.basis
    jump = model.jump_operator[:, b.nparent:].tocsr()
    jump.eliminate_zeros()
    count = len(model.geometry.interface_area_m2)
    if jump.shape != (2*count, b.ndof-b.nparent) or jump.shape[1] == 0 or jump.shape[1] % 2:
        raise ValueError("Capacity certificate requires two-component local bank enrichment")
    water = _real(water_per_trace, (count,), "water_per_trace")
    if np.any((water < 0) | (water > 1)):
        raise ValueError("water_per_trace must lie in [0, 1]")
    law = law_parameters or model.law_parameters
    law.validate()
    area = _real(model.geometry.interface_area_m2, (count,), "interface_area_m2")
    if np.any(area <= 0):
        raise ValueError("Capacity certificate requires positive trace areas")
    reaction = state.constraint_reaction_n[b.nparent:].reshape(-1, 2)
    required = np.linalg.norm(reaction, axis=1)
    if not np.isfinite(required).all():
        raise ValueError("Capacity reaction norm overflowed")
    direction = np.divide(reaction, required[:, None], out=np.zeros_like(reaction),
                          where=required[:, None] > 0)
    coefficients = np.asarray(jump@direction.ravel()).reshape(-1, 2)
    if not np.isfinite(coefficients).all():
        raise ValueError("Capacity jump projection overflowed")
    rows = jump.tocoo()
    owner = np.full(count, -1, dtype=np.int64)
    for trace, vertex in zip(rows.row//2, rows.col//2):
        if owner[trace] not in (-1, vertex):
            raise ValueError("Each capacity trace must belong to one enrichment vertex")
        owner[trace] = vertex
    observed = owner >= 0
    if not observed.any() or not np.isin(np.arange(len(required)), owner[observed]).all():
        raise ValueError("Capacity certificate requires observable bank vertices")
    friction = law.friction_dry+(law.friction_wet-law.friction_dry)*water
    cohesion = law.cohesion_pa*(1-(1-law.wet_cohesion_fraction)*water)
    a, shear = coefficients[:, 0], np.abs(coefficients[:, 1])
    margin = a-friction*shear
    tolerance = coefficient_tolerance*np.maximum(1., np.maximum(np.abs(a), friction*shear))
    unbounded = np.zeros(len(required), dtype=bool)
    indeterminate = np.zeros(len(required), dtype=bool)
    np.logical_or.at(unbounded, owner[observed], margin[observed] < -tolerance[observed])
    near_zero = (np.abs(margin) <= tolerance) & (margin != 0)
    np.logical_or.at(indeterminate, owner[observed], near_zero[observed])
    indeterminate &= ~unbounded
    finite = ~(unbounded | indeterminate)
    capacity = np.zeros(len(required))
    np.add.at(capacity, owner[observed],
              area[observed]*(a[observed]*law.tensile_strength_pa+shear[observed]*cohesion[observed]))
    capacity[~finite] = np.inf
    if np.any(capacity[finite] < 0) or not np.isfinite(capacity[finite]).all():
        raise ValueError("Finite capacity calculation overflowed or became negative")
    ratio = np.divide(required, capacity, out=np.zeros_like(required), where=capacity > 0)
    ratio[finite & (capacity == 0) & (required > 0)] = np.inf
    violation = finite & (required > capacity*(1+relative_tolerance))
    return TiedCapacityDiagnostic(*map(_owned, (direction, required, capacity, finite,
        unbounded, indeterminate, ratio, violation)))
