"""Certified Euler/contact equilibrium on a frozen spherical partition.

The Euler coordinates are small rotations in radians in the fixed source
frame. Every real contact has one declared dipping plane and fixed, explicit
preparation anchors. Mean horizontal Euler displacement is the sole Galerkin
coordinate of each contact; strain is uniform in its transported local frame.
No material is moved, and these mean strains are not pointwise rigid traces.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import hashlib
import json
import math
from numbers import Integral, Real

import numpy as np

from .genesis_contact_law import contact_return_map
from .geometric_boundary import surface_digest
from .geometric_contacts import extract_contacts
from .geometric_contact_mechanics import LocalContactState, solve_compressed_contact, validate_contact_state
from .geometric_contact_mechanics_adapter import (
    ContactExperiment, ContactMechanicsResult, ContactMechanicsSnapshot,
    _experiments, _frame, _integrated_reaction, mechanics_from_dict,
    mechanics_to_dict,
)
from .geometric_polarity import PolarityEvidence, contact_polarity_report


FORMAT = "geometric-frozen-euler-contact-dynamics-1"
COORDINATES = "fixed_frame_small_euler_rotations_rad_exact_arc_mean_horizontal_trace"


class UnresolvedContactDynamics(ValueError):
    """The declared bounded problem cannot certify an accepted next state."""


def _number(value, name, *, positive=False, nonnegative=False):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not math.isfinite(float(value))):
        raise ValueError(f"{name} must be finite and real")
    value = float(value)
    if (positive and value <= 0.) or (nonnegative and value < 0.):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'}")
    return value


def _encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _basal(surface, load):
    # The basal module owns source validation and its portable representation.
    from .geometric_basal_sources import basal_source_operators
    return basal_source_operators(surface, load)


@dataclass(frozen=True, slots=True)
class GlobalContactState:
    surface_sha256: str
    basal_source_sha256: str
    plate_ids: tuple[int, ...]
    experiments: tuple[ContactExperiment, ...]
    polarity_history: tuple[PolarityEvidence, ...]
    anchor_offsets_m: tuple[tuple[str, tuple[float, float]], ...]
    states: tuple[tuple[str, LocalContactState], ...]
    rotation_radians: tuple[tuple[float, float, float], ...]
    initial_loading_dt_s: float
    max_rotation_radians: float
    elapsed_time_s: float
    step_count: int
    preparation_work_j: float
    cumulative_source_work_j: float
    cumulative_basal_dissipation_j: float


@dataclass(frozen=True, slots=True)
class GlobalContactStep:
    state: GlobalContactState
    angular_velocity_rad_s: tuple[tuple[float, float, float], ...]
    contacts: tuple[ContactMechanicsResult, ...]
    basal_drag_torque_nm: tuple[tuple[float, float, float], ...]
    contact_torque_nm: tuple[tuple[float, float, float], ...]
    torque_residual_nm: tuple[tuple[float, float, float], ...]
    torque_relative_residual: float
    source_work_j: float
    basal_dissipation_j: float
    contact_work_j: float
    stored_energy_increment_j: float
    friction_work_j: float
    viscous_work_j: float
    numerical_dissipation_j: float
    energy_balance_residual_j: float
    energy_balance_relative_error: float
    minimum_certified_eigenvalue: float
    iterations: int
    torque_signal_relative_residual: float
    incremental_energy_balance_relative_error: float
    torque_rounding_tolerance_nm: float
    energy_rounding_tolerance_j: float


def _network(surface, experiments, plate_ids, history=()):
    experiments = _experiments(surface, experiments)
    contacts = {c.contact_id: c for c in extract_contacts(surface.fragments, surface.radius_km)}
    grouped = {identity: [] for identity in contacts}
    for experiment in experiments:
        grouped[experiment.fault.contact_id].append(experiment)
    if any(len(group) != 1 for group in grouped.values()):
        raise UnresolvedContactDynamics(
            "Every real contact requires exactly one declared plane; missing or alternative planes are unresolved")
    reports = contact_polarity_report(tuple(contacts.values()), tuple(history))
    for report in reports:
        fault = grouped[report["contact_id"]][0].fault
        expected = [[fault.subducting_plate, fault.overriding_plate]]
        if report["status"] == "conflicting" or report["directions"] != expected:
            raise UnresolvedContactDynamics("Conflicting or opposed contact history cannot be reselected")
    actual = tuple(sorted({fragment.parcel.plate for fragment in surface.fragments}))
    if plate_ids != actual:
        raise ValueError("Basal compact plate map must contain every surface owner exactly once")
    indices = {identity: index for index, identity in enumerate(plate_ids)}
    rows, areas, frames, canonical = [], [], [], []
    for experiment in experiments:
        fault = experiment.fault
        contact = contacts[fault.contact_id]
        sign = 1. if fault.overriding_plate == contact.plate_b else -1.
        mean = sign*np.asarray(contact.moment_arm_cross_normal_km2)/contact.length_km*1000.
        row = np.zeros((len(plate_ids), 3))
        row[indices[fault.subducting_plate]] = -mean
        row[indices[fault.overriding_plate]] = mean
        rows.append(row.reshape(-1))
        sine, cosine = math.sin(fault.dip_radians), math.cos(fault.dip_radians)
        areas.append(contact.length_km*1000.*experiment.contact_height_m/sine)
        frames.append((sine, cosine))
        canonical.append(contact)
    return experiments, tuple(canonical), np.asarray(rows).reshape((len(rows), 3*len(plate_ids))), np.asarray(areas), np.asarray(frames).reshape((len(rows), 2))


def _offsets(experiments, supplied):
    if not isinstance(supplied, dict) or set(supplied) != {e.fault.fault_id for e in experiments}:
        raise ValueError("Explicit preparation anchor offsets must name every declared fault")
    result = []
    for experiment in experiments:
        name = experiment.fault.fault_id
        pair = supplied[name]
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError("Each preparation anchor offset must be [gap, jump] in metres")
        gap, jump = (_number(value, "anchor offset") for value in pair)
        if gap > 0.:
            raise UnresolvedContactDynamics("Preparation normal opening is unsupported")
        result.append((name, (gap, jump)))
    return tuple(result)


def _clock_check(state):
    if (isinstance(state.step_count, (bool, np.bool_)) or not isinstance(state.step_count, Integral)
            or state.step_count < 0):
        raise ValueError("Global step count must be a nonnegative integer")
    if state.step_count > 0 and state.elapsed_time_s <= 0.:
        raise ValueError("Evolved global history must have positive elapsed time")
    expected = state.initial_loading_dt_s+state.elapsed_time_s
    for _, local in state.states:
        if local.step_count != state.step_count+1:
            raise ValueError("Global and local mechanical step counts disagree")
        scale = max(abs(expected), abs(local.elapsed_time_s), np.finfo(float).tiny)
        if abs(local.elapsed_time_s-expected) > 512*np.finfo(float).eps*scale:
            raise ValueError("Global and local mechanical clocks disagree")


def _validate_network_history(state, ids, torque, rows, areas, frames, materials, offsets, old_local, *,
                              augmented_coordinates=None, anchor_operators=None,
                              elastic_stiffness=None,
                              cumulative_elastic_numerical_dissipation_j=0.):
    """Shared reference/clock/energy checks independent of trace representation."""
    q = np.asarray(state.rotation_radians)
    if (q.shape != (len(ids), 3) or q.dtype.kind not in "iuf" or not np.isfinite(q).all()
            or any(isinstance(value, (bool, np.bool_)) for row in state.rotation_radians for value in row)):
        raise ValueError("Global rotations must be finite real compact [plate,3] coordinates")
    for name in ("initial_loading_dt_s", "max_rotation_radians"):
        _number(getattr(state, name), name, positive=True)
    for name in ("elapsed_time_s", "preparation_work_j", "cumulative_basal_dissipation_j"):
        _number(getattr(state, name), name, nonnegative=True)
    _number(state.cumulative_source_work_j, "cumulative_source_work_j")
    if np.max(np.linalg.norm(q, axis=1), initial=0.) > state.max_rotation_radians:
        raise UnresolvedContactDynamics("Stored cumulative rotation exceeds the fixed-frame validity bound")
    generalized = augmented_coordinates is not None
    if generalized:
        coordinates = np.asarray(augmented_coordinates)
        if (coordinates.ndim != 1 or coordinates.dtype.kind not in "iuf"
                or not np.isfinite(coordinates).all() or coordinates.size < q.size
                or any(isinstance(value, (bool, np.bool_)) for value in augmented_coordinates)
                or not np.array_equal(coordinates[:q.size], q.reshape(-1))):
            raise ValueError("Augmented coordinates must retain the finite Euler coordinate prefix")
        operators = _anchor_operator_array(anchor_operators, len(areas), coordinates.size)
        stiffness = _elastic_stiffness_array(elastic_stiffness, coordinates.size)
        elastic_remainder = _number(cumulative_elastic_numerical_dissipation_j,
            "cumulative_elastic_numerical_dissipation_j", nonnegative=True)
        elastic_energy = .5*float(coordinates@stiffness@coordinates)
        if not math.isfinite(elastic_energy) or elastic_energy < 0.:
            raise ValueError("Stored conservative elastic energy is invalid")
        applied = np.asarray(torque, dtype=float)
        if applied.shape == (q.size,):
            applied = np.pad(applied, (0, coordinates.size-q.size))
        if applied.shape != coordinates.shape or not np.isfinite(applied).all():
            raise ValueError("Stored source must match the augmented coordinate map")
    else:
        if anchor_operators is not None or elastic_stiffness is not None:
            raise ValueError("Generalized history operators require augmented coordinates")
        coordinates, applied = q.reshape(-1), torque
        elastic_energy = elastic_remainder = 0.
    _clock_check(state)
    prescribed_work = float(applied@np.asarray(coordinates, dtype=float))
    source_rounding_scale = max(abs(state.cumulative_source_work_j),
        math.fsum(abs(float(t*w)) for t, w in zip(applied, coordinates)), np.finfo(float).tiny)
    if abs(state.cumulative_source_work_j-prescribed_work) > 512*np.finfo(float).eps*source_rounding_scale:
        raise ValueError("Stored source work disagrees with fixed torque and accumulated Euler displacement")
    preparation = math.fsum(solve_compressed_contact(columns=experiment.columns, law=experiment.law,
        area_m2=float(area), gap_free_m=offset[0], jump_free_m=offset[1],
        dt_s=state.initial_loading_dt_s, water=experiment.water).external_work_j
        for experiment, offset, area in zip(materials, offsets, areas))
    preparation_scale = max(abs(preparation), abs(state.preparation_work_j), np.finfo(float).tiny)
    if abs(state.preparation_work_j-preparation) > 512*np.finfo(float).eps*preparation_scale:
        raise ValueError("Stored preparation work disagrees with the explicit original anchor loading")
    means = rows@np.asarray(q, dtype=float).reshape(-1) if not generalized else None
    anchors = (np.asarray(offsets)+np.einsum("ijk,k->ij", operators, coordinates)
        if generalized else np.asarray(offsets)+frames*means[:, None])
    for index, (material, offset, area, local) in enumerate(zip(materials, offsets, areas, old_local)):
        validate_contact_state(local, columns=material.columns, law=material.law, area_m2=float(area), water=material.water)
        for component, actual in enumerate((local.gap_free_m, local.jump_free_m)):
            expected = anchors[index, component]
            geometric_scale = (math.fsum(abs(float(value)) for value in operators[index, component]*coordinates)
                if generalized else abs(means[index]))
            scale = max(abs(actual), abs(expected), abs(offset[0]), abs(offset[1]),
                        geometric_scale, np.finfo(float).tiny)
            if abs(actual-expected) > 512*np.finfo(float).eps*scale:
                raise ValueError("Stored local anchors disagree with global rotations and preparation offsets")
    accounted = math.fsum((*(local.bulk_energy_j+local.contact_energy_j
                            +local.cumulative_friction_work_j+local.cumulative_viscous_work_j
                            +local.cumulative_numerical_dissipation_j for _, local in state.states),
                          state.cumulative_basal_dissipation_j))
    if generalized:
        accounted = math.fsum((accounted, elastic_energy, elastic_remainder))
    work = math.fsum((state.preparation_work_j, state.cumulative_source_work_j))
    scale = max(abs(work), abs(accounted), np.finfo(float).tiny)
    if abs(work-accounted) > 5e-10*scale:
        raise ValueError("Stored global cumulative work balance is inconsistent")
    if state.step_count == 0:
        if (state.elapsed_time_s != 0. or np.any(coordinates != 0.)
                or state.cumulative_source_work_j != 0. or state.cumulative_basal_dissipation_j != 0.
                or (generalized and elastic_remainder != 0.)):
            raise ValueError("Prepared global state must start at zero Euler rotation and global elapsed time")
        initial_work = math.fsum(local.cumulative_external_work_j for _, local in state.states)
        if state.preparation_work_j != initial_work:
            raise ValueError("Preparation work must equal the stored initial clamp work")
        if generalized:
            for material, offset, area, local in zip(materials, offsets, areas, old_local):
                prepared = solve_compressed_contact(columns=material.columns, law=material.law,
                    area_m2=float(area), gap_free_m=offset[0], jump_free_m=offset[1],
                    dt_s=state.initial_loading_dt_s, water=material.water).state
                if local != prepared:
                    raise ValueError("Prepared generalized history disagrees with the explicit initial loading")
    return state


def validate_contact_dynamics_state(surface, state, basal_load):
    if not isinstance(state, GlobalContactState):
        raise ValueError("Expected GlobalContactState")
    ids, _, torque, signature = _basal(surface, basal_load)
    if (not isinstance(state.plate_ids, tuple) or any(isinstance(value, (bool, np.bool_))
            or not isinstance(value, Integral) for value in state.plate_ids)
            or not isinstance(state.states, tuple) or not isinstance(state.rotation_radians, tuple)
            or any(not isinstance(row, tuple) for row in state.rotation_radians)):
        raise ValueError("Global plate coordinates and histories must be immutable tuples")
    if (state.surface_sha256 != surface_digest(surface) or state.basal_source_sha256 != signature
            or state.plate_ids != ids):
        raise ValueError("Global restart surface or frozen basal source changed")
    if (not isinstance(state.polarity_history, tuple)
            or any(not isinstance(item, PolarityEvidence) for item in state.polarity_history)):
        raise ValueError("Global contact polarity history must be an immutable tuple")
    canonical_history = tuple(sorted(state.polarity_history, key=lambda item: item.evidence_id))
    if canonical_history != state.polarity_history:
        raise ValueError("Global contact history must be in canonical order")
    experiments, _, rows, areas, frames = _network(surface, state.experiments, ids, state.polarity_history)
    if experiments != state.experiments:
        raise ValueError("Global experiments must be in canonical order")
    offsets = _offsets(experiments, dict(state.anchor_offsets_m))
    if offsets != state.anchor_offsets_m:
        raise ValueError("Global offsets must be unique and in canonical order")
    mechanics_to_dict(surface, ContactMechanicsSnapshot(state.surface_sha256, experiments, state.states))
    by_id = dict(state.states)
    _validate_network_history(state, ids, torque, rows, areas, frames, experiments,
        tuple(pair for _, pair in offsets), tuple(by_id[e.fault.fault_id] for e in experiments))
    return state


def initialize_contact_dynamics(surface, experiments, anchor_offsets_m, basal_load, *,
                                initial_loading_dt_s, max_rotation_radians, history=()):
    """Prepare explicit preloads before starting the frozen Euler source step.

    Preparation work is retained separately. A complete closed sphere cannot
    have all negative Euler mean gaps without preloads: its weighted mean gap
    operator sums to zero. Offsets are physical initial clamp data, not a fit.
    """
    dt = _number(initial_loading_dt_s, "initial_loading_dt_s", positive=True)
    bound = _number(max_rotation_radians, "max_rotation_radians", positive=True)
    ids, _, _, signature = _basal(surface, basal_load)
    history = tuple(history)
    experiments, _, _, areas, _ = _network(surface, experiments, ids, history)
    history = tuple(sorted(history, key=lambda item: item.evidence_id))
    offsets = _offsets(experiments, anchor_offsets_m)
    states = []
    for experiment, (_, (gap, jump)), area in zip(experiments, offsets, areas):
        result = solve_compressed_contact(columns=experiment.columns, law=experiment.law,
            area_m2=float(area), gap_free_m=gap, jump_free_m=jump, dt_s=dt, water=experiment.water)
        states.append((experiment.fault.fault_id, result.state))
    state = GlobalContactState(surface_digest(surface), signature, ids, experiments, history, offsets,
        tuple(states), tuple((0., 0., 0.) for _ in ids), dt, bound, 0., 0,
        math.fsum(local.cumulative_external_work_j for _, local in states), 0., 0.)
    return validate_contact_dynamics_state(surface, state, basal_load)


def _slope_bound(experiment, dt_s, sine, cosine):
    """Lower derivative bound of the scalar Galerkin force on every branch.

    This is a stability certificate, not a second material return algorithm.
    Normal/shear coupling has opposite signs on the two sliding directions.
    """
    law, columns = experiment.law, experiment.columns
    normal = law.normal_stiffness_pa_m/(1.+law.normal_stiffness_pa_m/columns.normal_stiffness_pa_m)
    locked = law.tangential_stiffness_pa_m/(1.+law.tangential_stiffness_pa_m/columns.shear_stiffness_pa_m)
    viscosity = law.viscosity_pa_s_m/dt_s
    if not math.isfinite(viscosity):
        raise ValueError("Contact rate coefficient overflowed")
    tangential = (law.tangential_stiffness_pa_m/(1.+law.tangential_stiffness_pa_m/viscosity)
                  if viscosity > 0. else 0.)
    condensed = tangential/(1.+tangential/columns.shear_stiffness_pa_m)
    beta = 1./((1.+tangential/columns.shear_stiffness_pa_m)
               *(1.+viscosity/law.tangential_stiffness_pa_m))
    friction = law.friction_dry+(law.friction_wet-law.friction_dry)*experiment.water
    elastic = normal*sine*sine+locked*cosine*cosine
    sliding = normal*sine*sine+condensed*cosine*cosine-beta*friction*normal*sine*cosine
    value = min(elastic, sliding)
    if not math.isfinite(value):
        raise ValueError("Contact tangent bound overflowed")
    return value


def _result(surface, contact, experiment, step):
    normal, down = _frame(contact, experiment.fault)
    tn, tt = step.traction_pa
    tensor = -tn*np.outer(normal, normal)-tt*(np.outer(down, normal)+np.outer(normal, down))
    fault = replace(experiment.fault, effective_stress_pa=tensor,
        provenance=f"solved_global_frozen_euler_contact:{experiment.provenance}:{experiment.fault.provenance}")
    force, torque = _integrated_reaction(contact, experiment.fault, experiment.contact_height_m,
                                         step.traction_pa)
    return ContactMechanicsResult(surface_digest(surface), fault.fault_id, fault.contact_id, step, fault,
        tuple(map(float, force)), tuple(map(float, -force)),
        tuple(map(float, torque)), tuple(map(float, -torque)), -step.slip_increment_m)


@dataclass(frozen=True, slots=True)
class _ContactNetworkEquilibrium:
    """Temporary result shared by mean and distributed contact wrappers."""
    q: np.ndarray
    omega: np.ndarray
    steps: tuple
    contact_force: np.ndarray
    drag_force: np.ndarray
    residual: np.ndarray
    diagnostics: dict
    elastic_force: np.ndarray | None = None


def _anchor_operator_array(value, point_count, coordinate_count):
    operators = np.asarray(value)
    if (operators.shape != (point_count, 2, coordinate_count)
            or operators.dtype.kind not in "iuf" or not np.isfinite(operators).all()):
        raise ValueError("Free normal/shear operators must be finite [point,2,coordinate] arrays")
    return np.asarray(operators, dtype=float)


def _elastic_stiffness_array(value, coordinate_count):
    if value is None:
        return np.zeros((coordinate_count, coordinate_count))
    stiffness = np.asarray(value)
    if (stiffness.shape != (coordinate_count, coordinate_count)
            or stiffness.dtype.kind not in "iuf" or not np.isfinite(stiffness).all()):
        raise ValueError("Conservative stiffness must be a finite square matrix")
    stiffness = np.asarray(stiffness, dtype=float)
    magnitude = max(float(np.linalg.norm(stiffness)), np.finfo(float).tiny)
    tolerance = 512*np.finfo(float).eps*magnitude
    if float(np.linalg.norm(stiffness-stiffness.T)) > tolerance:
        raise ValueError("Conservative stiffness must be symmetric")
    stiffness = .5*(stiffness+stiffness.T)
    if np.min(np.linalg.eigvalsh(stiffness), initial=0.) < -tolerance:
        raise ValueError("Conservative stiffness must be positive semidefinite")
    return stiffness


def _generalized_tangent_lower(material, dt, normal_row, shear_row):
    """A common Loewner lower bound for locked and either sliding sign.

    Let P contain the normal stiffness and the smaller sliding shear stiffness.
    Each sliding symmetric tangent is P +/- M. Spectral |M| +/- M is PSD,
    hence P-|M| bounds both signs; the locked tangent also exceeds P. Rows
    already use the Cholesky-whitened generalized coordinates. No assumed
    active branch or numerical penalty enters this uniqueness certificate.
    """
    law, columns = material.law, material.columns
    normal = law.normal_stiffness_pa_m/(1.+law.normal_stiffness_pa_m/columns.normal_stiffness_pa_m)
    viscosity = law.viscosity_pa_s_m/dt
    if not math.isfinite(viscosity):
        raise ValueError("Contact rate coefficient overflowed")
    shear = (law.tangential_stiffness_pa_m/(1.+law.tangential_stiffness_pa_m/viscosity)
        if viscosity > 0. else 0.)
    condensed = shear/(1.+shear/columns.shear_stiffness_pa_m)
    beta = 1./((1.+shear/columns.shear_stiffness_pa_m)
        *(1.+viscosity/law.tangential_stiffness_pa_m))
    friction = law.friction_dry+(law.friction_wet-law.friction_dry)*material.water
    positive = (normal*np.outer(normal_row, normal_row)
        +condensed*np.outer(shear_row, shear_row))
    mixed = .5*beta*friction*normal*(np.outer(normal_row, shear_row)
        +np.outer(shear_row, normal_row))
    if not np.isfinite(positive).all() or not np.isfinite(mixed).all():
        raise ValueError("Generalized contact tangent bound overflowed")
    values, vectors = np.linalg.eigh(mixed)
    absolute = (vectors*np.abs(values))@vectors.T
    return positive-absolute


def _equilibrate_contact_network(qold, drag, torque, rows, areas, frames, materials,
                                 offsets, old_local, dt, max_rotation_radians, *,
                                 domain_check=None, anchor_operators=None,
                                 elastic_stiffness=None, rotation_dof_count=None,
                                 coordinate_check=None, initial_guess=None,
                                 common_rotation_gauge=False):
    """One certified Newton/constitutive/work engine for both representations.

    Materials need columns, law and water. Geometry, source binding, physical
    resultants and persistent state are owned by the representation wrapper.
    The optional domain callback checks the complete continuous support before
    each trial; a callback rejection never supplies a mechanical barrier force.
    Optional normal/shear operators support augmented dimensionless coordinates
    and conservative zero-rest elasticity. Only the Euler prefix is checked
    against the rotation bound; the representation callback owns additional
    coordinate bounds. The original scalar-trace arithmetic remains unchanged
    when no augmented operator is supplied. An optional initial guess remains
    subject to the same domain and full force/work certificates. The optional
    common-rotation gauge evaluates a certified exact angular nullspace before
    contraction; it does not project the unknowns, velocities or source work.
    """
    generalized = anchor_operators is not None
    if not isinstance(common_rotation_gauge, bool):
        raise ValueError("Common-rotation gauge must be an explicit boolean")
    if common_rotation_gauge and not generalized:
        raise ValueError("Common-rotation gauge requires generalized contact operators")
    guess = None
    if initial_guess is not None:
        supplied_guess = np.asarray(initial_guess)
        if (supplied_guess.shape != np.shape(qold) or supplied_guess.ndim != 1
                or supplied_guess.dtype.kind not in "iuf" or not np.isfinite(supplied_guess).all()
                or any(isinstance(value, (bool, np.bool_))
                       for value in np.asarray(initial_guess, dtype=object).flat)):
            raise ValueError("Initial contact guess must be a finite real coordinate vector")
        guess = np.asarray(supplied_guess, dtype=float).copy()
    if generalized:
        operators = _anchor_operator_array(anchor_operators, len(areas), len(qold))
        stiffness = _elastic_stiffness_array(elastic_stiffness, len(qold))
        if (isinstance(rotation_dof_count, (bool, np.bool_))
                or not isinstance(rotation_dof_count, Integral)
                or rotation_dof_count < 0 or rotation_dof_count > len(qold)
                or rotation_dof_count % 3):
            raise ValueError("Generalized rotation prefix must contain complete Euler triples")
        if common_rotation_gauge:
            if rotation_dof_count == 0:
                raise ValueError("Common-rotation gauge requires an angular coordinate prefix")
            angular_blocks = operators[:, :, :rotation_dof_count].reshape(
                (len(areas), 2, rotation_dof_count//3, 3))
            if any(math.fsum(float(value) for value in component) != 0.
                   for block in angular_blocks for row in block for component in row.T):
                raise ValueError("Contact operators do not annihilate a common angular rotation exactly")
        damping = np.asarray(drag, dtype=float)
        if damping.shape != (len(qold), len(qold)) or not np.isfinite(damping).all():
            raise ValueError("Generalized damping must be a finite square matrix")
        damping_scale = max(float(np.linalg.norm(damping)), np.finfo(float).tiny)
        damping_tolerance = 512*np.finfo(float).eps*damping_scale
        if (float(np.linalg.norm(damping-damping.T)) > damping_tolerance
                or np.min(np.linalg.eigvalsh(.5*(damping+damping.T)), initial=0.) < -damping_tolerance):
            raise ValueError("Generalized damping must be symmetric positive semidefinite")
        base = drag/dt+stiffness
    else:
        if elastic_stiffness is not None or rotation_dof_count is not None:
            raise ValueError("Augmented stiffness and coordinates require normal/shear operators")
        base = drag/dt
    if not np.isfinite(base).all():
        raise ValueError("Global drag/time scale overflowed")
    try:
        chol = np.linalg.cholesky(base)
    except np.linalg.LinAlgError as error:
        raise UnresolvedContactDynamics("Combined damping/time and conservative stiffness must be positive definite") from error
    whitened_rows = (None if generalized else np.linalg.solve(chol, rows.T).T)
    if generalized:
        whitened_operators = np.linalg.solve(chol, operators.reshape((-1, len(qold))).T).T.reshape(operators.shape)
    lower = np.eye(len(qold))
    if generalized:
        for material, area, operator in zip(materials, areas, whitened_operators):
            lower += area*_generalized_tangent_lower(material, dt, *operator)
    else:
        for experiment, area, frame, row in zip(materials, areas, frames, whitened_rows):
            lower += area*_slope_bound(experiment, dt, *frame)*np.outer(row, row)
    lower = .5*(lower+lower.T)
    eigenvalues = np.linalg.eigvalsh(lower)
    minimum = float(eigenvalues[0])
    certification_tolerance = 1024*np.finfo(float).eps*len(qold)*max(float(np.max(np.abs(eigenvalues))), 1.)
    if not np.isfinite(eigenvalues).all() or minimum <= certification_tolerance:
        raise UnresolvedContactDynamics("Global Coulomb equilibrium has no positive-definite uniqueness certificate")

    def contact_coordinates(q):
        if not common_rotation_gauge:
            return q
        relative = q.copy()
        rotations = relative[:rotation_dof_count].reshape((-1, 3))
        rotations -= rotations[0].copy()
        return relative

    def evaluate(q):
        if coordinate_check is not None:
            coordinate_check(q)
        if domain_check is not None:
            domain_check(q)
        anchors = (offsets+np.einsum("ijk,k->ij", operators, contact_coordinates(q))
            if generalized else offsets+frames*(rows@q)[:, None])
        if np.any(anchors[:, 0] > 0.):
            raise UnresolvedContactDynamics("A contact requires unsupported normal opening")
        steps, slopes, horizontal, condensed_tangents = [], [], [], []
        for experiment, area, frame, anchor, local_old in zip(materials, areas, frames, anchors, old_local):
            step = solve_compressed_contact(columns=experiment.columns, law=experiment.law,
                area_m2=float(area), gap_free_m=float(anchor[0]), jump_free_m=float(anchor[1]),
                dt_s=dt, water=experiment.water, previous=local_old)
            response = contact_return_map(np.array([step.state.gap_m]), np.array([step.state.jump_m]),
                np.array([local_old.plastic_slip_m]), np.array([local_old.cumulative_slip_m]),
                np.zeros(1), dt, np.array([experiment.water]), experiment.law, old_damage=np.zeros(1))
            tangent = response["tangent_pa_m"][0]
            inverse_bulk = np.diag([1./experiment.columns.normal_stiffness_pa_m,
                                    1./experiment.columns.shear_stiffness_pa_m])
            condensed = np.linalg.solve(np.eye(2)+tangent@inverse_bulk, tangent)
            if generalized:
                condensed_tangents.append(condensed)
            slopes.append(float(frame@condensed@frame))
            horizontal.append(float(frame@np.asarray(step.traction_pa)))
            steps.append(step)
        if generalized:
            contact_force = np.einsum("i,ijk,ij->k", areas, operators,
                np.asarray([step.traction_pa for step in steps]).reshape((len(areas), 2)))
            drag_force = (drag/dt)@(q-qold)
            elastic_force = stiffness@q
            residual = drag_force+elastic_force-torque+contact_force
            contact_operands = math.fsum(float(area*math.fsum(abs(float(traction))*float(np.linalg.norm(row))
                for traction, row in zip(step.traction_pa, operator)))
                for area, operator, step in zip(areas, operators, steps))
            scale = max(float(np.linalg.norm(drag_force))+float(np.linalg.norm(elastic_force))
                +float(np.linalg.norm(torque))+contact_operands, np.finfo(float).tiny)
        else:
            contact_force = rows.T@(areas*np.asarray(horizontal))
            drag_force = base@(q-qold)
            residual = drag_force-torque+contact_force
            scale = max(float(np.linalg.norm(drag_force))+float(np.linalg.norm(torque))
                        +math.fsum(float(area*abs(value)*np.linalg.norm(row))
                                  for area, value, row in zip(areas, horizontal, rows)), np.finfo(float).tiny)
        error = float(np.linalg.norm(residual))/scale
        jacobian = np.eye(len(qold))
        if generalized:
            for area, tangent, operator in zip(areas, condensed_tangents, whitened_operators):
                jacobian += area*operator.T@tangent@operator
        else:
            for area, slope, row in zip(areas, slopes, whitened_rows):
                jacobian += area*slope*np.outer(row, row)
        return residual, error, jacobian, tuple(steps), contact_force, drag_force, anchors

    def force_certification(value, coordinates):
        residual, _, _, steps, contact_force, drag_force, _ = value
        signal = max(float(np.linalg.norm(torque)), float(np.linalg.norm(drag_force)),
                     float(np.linalg.norm(contact_force)), np.finfo(float).tiny)
        if generalized:
            elastic_magnitude = float(np.linalg.norm(stiffness@coordinates))
            signal = max(signal, elastic_magnitude)
            operands = (float(np.linalg.norm(torque))+float(np.linalg.norm(drag_force))+elastic_magnitude
                +math.fsum(float(area*math.fsum(abs(float(traction))*float(np.linalg.norm(row))
                    for traction, row in zip(step.traction_pa, operator)))
                    for area, operator, step in zip(areas, operators, steps)))
        else:
            operands = (float(np.linalg.norm(torque))+float(np.linalg.norm(drag_force))
                        +math.fsum(float(area*abs(frame@np.asarray(step.traction_pa))*np.linalg.norm(row))
                                  for area, frame, step, row in zip(areas, frames, steps, rows)))
        rounding = 512*np.finfo(float).eps*operands
        magnitude = float(np.linalg.norm(residual))
        return magnitude, signal, rounding

    q = qold.copy() if guess is None else guess
    current = evaluate(q)
    iterations = 0
    for iterations in range(81):
        residual, error, jacobian, *_ = current
        force_magnitude, force_signal, force_rounding = force_certification(current, q)
        if force_magnitude <= 2e-12*force_signal+force_rounding:
            break
        if iterations == 80:
            raise UnresolvedContactDynamics("Certified contact Newton solve did not reach force tolerance")
        white_residual = np.linalg.solve(chol, residual)
        white_step = np.linalg.solve(jacobian, -white_residual)
        increment = np.linalg.solve(chol.T, white_step)
        normal_increments = (operators[:, 0, :]@contact_coordinates(increment)
            if generalized else frames[:, 0]*(rows@increment))
        gaps = current[-1][:, 0]
        positive = normal_increments > 0.
        maximum = float(np.min(-gaps[positive]/normal_increments[positive], initial=math.inf))
        if maximum <= 0.:
            raise UnresolvedContactDynamics("Equilibrium requires motion beyond the supported closed-contact domain")
        fraction = 1. if maximum >= 1. else .99*maximum
        merit = float(np.linalg.norm(white_residual))
        accepted = False
        for _ in range(60):
            trial_q = q+fraction*increment
            if np.array_equal(trial_q, q):
                break
            try:
                trial = evaluate(trial_q)
                trial_merit = float(np.linalg.norm(np.linalg.solve(chol, trial[0])))
                magnitude, signal, rounding = force_certification(trial, trial_q)
                if magnitude <= 2e-12*signal+rounding or trial_merit <= (1.-1e-4*fraction)*merit:
                    q, current, accepted = trial_q, trial, True
                    break
            except (ValueError, RuntimeError, OverflowError):
                pass
            fraction *= .5
        if not accepted:
            raise UnresolvedContactDynamics("Closed contact line search failed; no history was committed")
    residual, error, _, steps, contact_force, drag_force, _ = current
    rotations = (q[:rotation_dof_count] if generalized else q).reshape((-1, 3))
    rotation_increment = ((q-qold)[:rotation_dof_count] if generalized else q-qold).reshape((-1, 3))
    if (np.max(np.linalg.norm(rotations, axis=1), initial=0.) > max_rotation_radians
            or np.max(np.linalg.norm(rotation_increment, axis=1), initial=0.) > max_rotation_radians):
        raise UnresolvedContactDynamics("Solved rotation exceeds the explicit fixed-frame validity bound")
    omega = (q-qold)/dt
    source_work = float(torque@(q-qold))
    basal_work = float((q-qold)@drag@omega)
    contact_work = math.fsum(step.external_work_j for step in steps)
    stored = math.fsum(step.bulk_energy_increment_j+step.contact_energy_increment_j for step in steps)
    friction = math.fsum(step.friction_work_j for step in steps)
    viscous = math.fsum(step.viscous_work_j for step in steps)
    numerical = math.fsum(step.backward_euler_remainder_j for step in steps)
    if generalized:
        delta = q-qold
        elastic_energy = .5*float(q@stiffness@q)
        old_elastic_energy = .5*float(qold@stiffness@qold)
        bending_increment = .5*float(delta@stiffness@(q+qold))
        bending_numerical = .5*float(delta@stiffness@delta)
        bending_work = float(delta@stiffness@q)
        if min(elastic_energy, old_elastic_energy, bending_numerical) < 0.:
            raise UnresolvedContactDynamics("Conservative elastic work certificate requires nonnegative stored energy and remainder")
        stored = math.fsum((stored, bending_increment))
        numerical = math.fsum((numerical, bending_numerical))
    balance = math.fsum((source_work, -basal_work, -stored, -friction, -viscous, -numerical))
    energy_scale = max(abs(source_work), abs(basal_work)+abs(stored)+friction+viscous+numerical,
                       math.fsum(step.state.bulk_energy_j+step.state.contact_energy_j
                                 +local_old.bulk_energy_j+local_old.contact_energy_j
                                 for local_old, step in zip(old_local, steps)), np.finfo(float).tiny)
    if generalized:
        energy_scale = max(energy_scale, old_elastic_energy+elastic_energy)
    energy_error = abs(balance)/energy_scale
    incremental_energy_scale = max(abs(source_work),
        abs(basal_work)+abs(stored)+friction+viscous+numerical, np.finfo(float).tiny)
    energy_rounding = 512*np.finfo(float).eps*energy_scale
    force_magnitude, force_signal, force_rounding = force_certification(current, q)
    if (not all(math.isfinite(value) for value in (source_work, basal_work, balance, energy_error))
            or basal_work < 0. or abs(balance) > 5e-10*incremental_energy_scale+energy_rounding
            or force_magnitude > 2e-10*force_signal+force_rounding):
        raise UnresolvedContactDynamics("Global force/work certification failed; no history was committed")
    diagnostics = dict(torque_relative_residual=error, source_work_j=source_work,
        basal_dissipation_j=basal_work, contact_work_j=contact_work,
        stored_energy_increment_j=stored, friction_work_j=friction, viscous_work_j=viscous,
        numerical_dissipation_j=numerical, energy_balance_residual_j=balance,
        energy_balance_relative_error=energy_error, minimum_certified_eigenvalue=minimum,
        iterations=iterations, torque_signal_relative_residual=force_magnitude/force_signal,
        incremental_energy_balance_relative_error=abs(balance)/incremental_energy_scale,
        torque_rounding_tolerance_nm=force_rounding, energy_rounding_tolerance_j=energy_rounding)
    if generalized:
        diagnostics.update(bending_energy_increment_j=bending_increment,
            bending_work_j=bending_work, bending_numerical_dissipation_j=bending_numerical)
    return _ContactNetworkEquilibrium(q, omega, steps, contact_force, drag_force, residual, diagnostics,
        stiffness@q if generalized else None)


def advance_contact_dynamics(surface, previous, basal_load, dt_s):
    """Solve the established arc-mean problem through the shared engine."""
    dt = _number(dt_s, "dt_s", positive=True)
    validate_contact_dynamics_state(surface, previous, basal_load)
    ids, drag, torque, _ = _basal(surface, basal_load)
    experiments, contacts, rows, areas, frames = _network(surface, previous.experiments, ids)
    qold = np.asarray(previous.rotation_radians, dtype=float).reshape(-1)
    offsets = np.asarray([pair for _, pair in previous.anchor_offsets_m], dtype=float).reshape((-1, 2))
    old = dict(previous.states)
    local_old = tuple(old[experiment.fault.fault_id] for experiment in experiments)
    solved = _equilibrate_contact_network(qold, drag, torque, rows, areas, frames, experiments,
        offsets, local_old, dt, previous.max_rotation_radians)
    q, steps = solved.q, solved.steps
    rotations = q.reshape((-1, 3))
    metrics = solved.diagnostics
    state = replace(previous, states=tuple((experiment.fault.fault_id, step.state)
                    for experiment, step in zip(experiments, steps)),
        rotation_radians=tuple(tuple(map(float, row)) for row in rotations),
        elapsed_time_s=math.fsum((previous.elapsed_time_s, dt)), step_count=previous.step_count+1,
        cumulative_source_work_j=math.fsum((previous.cumulative_source_work_j, metrics["source_work_j"])),
        cumulative_basal_dissipation_j=math.fsum((previous.cumulative_basal_dissipation_j, metrics["basal_dissipation_j"])))
    validate_contact_dynamics_state(surface, state, basal_load)
    results = tuple(_result(surface, contact, experiment, step)
                    for contact, experiment, step in zip(contacts, experiments, steps))
    physical = np.zeros((len(ids), 3))
    positions = {identity: index for index, identity in enumerate(ids)}
    for result, experiment in zip(results, experiments):
        physical[positions[experiment.fault.subducting_plate]] += result.torque_on_sub_nm
        physical[positions[experiment.fault.overriding_plate]] += result.torque_on_over_nm
    physical_error = float(np.linalg.norm(physical.reshape(-1)+solved.contact_force))
    physical_scale = max(float(np.linalg.norm(solved.contact_force)),
        math.fsum(float(np.linalg.norm(result.torque_on_sub_nm)) for result in results), np.finfo(float).tiny)
    if physical_error > 512*np.finfo(float).eps*physical_scale:
        raise UnresolvedContactDynamics("Integrated contact moments disagree with the global virtual-work operator")
    triples = lambda value: tuple(tuple(map(float, row)) for row in np.asarray(value).reshape((-1, 3)))
    return GlobalContactStep(state, triples(solved.omega), results, triples(solved.drag_force),
        triples(physical), triples(solved.residual), **metrics)


def contact_dynamics_to_dict(surface, state, basal_load):
    """Metadata for the existing joint material checkpoint, without new IO."""
    validate_contact_dynamics_state(surface, state, basal_load)
    result = asdict(state)
    result.pop("experiments")
    result.pop("states")
    result.update(format=FORMAT, coordinate_convention=COORDINATES,
        mechanical_history=mechanics_to_dict(surface, ContactMechanicsSnapshot(
            state.surface_sha256, state.experiments, state.states)))
    return json.loads(_encode(result))


def contact_dynamics_from_dict(raw, surface, basal_load):
    names = {field.name for field in fields(GlobalContactState)}-{"experiments", "states"}
    if (not isinstance(raw, dict) or set(raw) != names|{"format", "coordinate_convention", "mechanical_history"}
            or raw["format"] != FORMAT or raw["coordinate_convention"] != COORDINATES):
        raise ValueError("Invalid global Euler contact checkpoint schema")
    values = {name: raw[name] for name in names}
    try:
        local = mechanics_from_dict(raw["mechanical_history"], surface)
        if not isinstance(values["plate_ids"], list) or any(
                isinstance(value, bool) or not isinstance(value, int) for value in values["plate_ids"]):
            raise ValueError("Global compact plate IDs must be an integer list")
        values["plate_ids"] = tuple(values["plate_ids"])
        if not isinstance(values["polarity_history"], list) or any(
                not isinstance(item, dict) or set(item) != {field.name for field in fields(PolarityEvidence)}
                for item in values["polarity_history"]):
            raise ValueError("Global polarity history must retain every explicit evidence field")
        values["polarity_history"] = tuple(PolarityEvidence(**item) for item in values["polarity_history"])
        if not isinstance(values["anchor_offsets_m"], list) or not isinstance(values["rotation_radians"], list):
            raise ValueError("Global offsets and rotations must be lists")
        values["anchor_offsets_m"] = tuple((name, tuple(pair)) for name, pair in values["anchor_offsets_m"])
        values["rotation_radians"] = tuple(tuple(row) for row in values["rotation_radians"])
        values.update(experiments=local.experiments, states=local.states)
        state = GlobalContactState(**values)
        return validate_contact_dynamics_state(surface, state, basal_load)
    except (TypeError, KeyError) as error:
        raise ValueError("Malformed global contact checkpoint record") from error
