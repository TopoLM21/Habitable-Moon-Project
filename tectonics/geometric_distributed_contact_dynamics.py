"""Pointwise contact histories on declared, materially independent fault arcs.

Positive quadrature weights resolve transported local contact strains instead
of assigning a single mean strain to an arc. Physical trace geometry and the
quadrature are fixed throughout a history. The shared Euler/contact engine
owns constitutive updates, uniqueness certification, Newton and work checks.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import math
from numbers import Integral

import numpy as np

from .geometric_boundary import surface_digest
from .geometric_contact_dynamics import (
    UnresolvedContactDynamics, _basal, _encode, _equilibrate_contact_network,
    _number, _validate_network_history,
)
from .geometric_contact_mechanics import LocalContactState, solve_compressed_contact, state_from_dict, state_to_dict


FORMAT = "geometric-frozen-distributed-contact-dynamics-1"
COORDINATES = "fixed_frame_small_euler_rotations_rad_physical_trace_quadrature"


@dataclass(frozen=True, slots=True)
class DistributedContactState:
    surface_sha256: str
    basal_source_sha256: str
    plate_ids: tuple[int, ...]
    fault_traces: object
    polarity_history: tuple
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
class PointContactResult:
    surface_sha256: str
    point_id: str
    trace_id: str
    position: tuple[float, float, float]
    area_m2: float
    step: object
    effective_stress_pa: tuple[tuple[float, float, float], ...]
    force_on_sub_n: tuple[float, float, float]
    force_on_over_n: tuple[float, float, float]
    torque_on_sub_nm: tuple[float, float, float]
    torque_on_over_nm: tuple[float, float, float]
    down_dip_plastic_increment_m: float


@dataclass(frozen=True, slots=True)
class DistributedContactStep:
    state: DistributedContactState
    angular_velocity_rad_s: tuple[tuple[float, float, float], ...]
    points: tuple[PointContactResult, ...]
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


def _point_network(surface, trace_set, plate_ids):
    from .geometric_fault_traces import validate_fault_trace_set, fault_trace_points
    validate_fault_trace_set(surface, trace_set)
    points = fault_trace_points(surface, trace_set)
    traces = {trace.trace_id: trace for trace in trace_set.traces}
    positions = {identity: index for index, identity in enumerate(plate_ids)}
    rows, frames, materials, offsets, areas = [], [], [], [], []
    for point in points:
        trace = traces[point.trace_id]
        row = np.zeros((len(plate_ids), 3))
        lever = surface.radius_km*1000.*np.cross(point.position, point.horizontal)
        row[positions[trace.subducting_plate]] = -lever
        row[positions[trace.overriding_plate]] = lever
        rows.append(row.reshape(-1))
        frames.append((math.sin(trace.dip_radians), math.cos(trace.dip_radians)))
        materials.append(trace)
        offsets.append((trace.initial_gap_m, trace.initial_jump_m))
        areas.append(point.area_m2)
    return (points, tuple(materials), np.asarray(rows).reshape((len(points), 3*len(plate_ids))),
        np.asarray(areas), np.asarray(frames).reshape((len(points), 2)),
        np.asarray(offsets).reshape((len(points), 2)))


def continuous_contact_gap_bounds(surface, trace_set, plate_ids, rotation_radians):
    """Exact endpoint/interior extrema of each continuous free normal gap.

    The affine Euler trace is a sinusoid on the great-circle arc. Quadrature
    nodes alone cannot rule out opening at endpoints or between nodes.
    """
    q = np.asarray(rotation_radians, dtype=float).reshape((len(plate_ids), 3))
    if not np.isfinite(q).all():
        raise ValueError("Continuous trace rotation must be finite")
    positions = {identity: index for index, identity in enumerate(plate_ids)}
    result = []
    for trace in trace_set.traces:
        start, end = np.asarray(trace.start), np.asarray(trace.end)
        horizontal = np.asarray(trace.normal_sub_to_over)
        tangent = np.cross(horizontal, start)
        tangent /= np.linalg.norm(tangent)
        angle = math.atan2(float(np.linalg.norm(np.cross(start, end))), float(start@end))
        relative = q[positions[trace.overriding_plate]]-q[positions[trace.subducting_plate]]
        vector = surface.radius_km*1000.*math.sin(trace.dip_radians)*np.cross(horizontal, relative)
        a, b = float(start@vector), float(tangent@vector)
        values = [a, a*math.cos(angle)+b*math.sin(angle)]
        stationary = math.atan2(b, a)
        for k in (-2, -1, 0, 1, 2):
            position = stationary+k*math.pi
            if 0. < position < angle:
                values.append(a*math.cos(position)+b*math.sin(position))
        minimum, maximum = trace.initial_gap_m+min(values), trace.initial_gap_m+max(values)
        if not math.isfinite(minimum) or not math.isfinite(maximum):
            raise ValueError("Continuous trace gap overflowed")
        result.append((trace.trace_id, minimum, maximum))
    return tuple(result)


def _closed_support(surface, trace_set, plate_ids, q):
    bounds = continuous_contact_gap_bounds(surface, trace_set, plate_ids, q)
    if any(maximum > 0. for _, _, maximum in bounds):
        raise UnresolvedContactDynamics("Continuous physical fault support requires unsupported normal opening")


def validate_distributed_contact_dynamics_state(surface, state, basal_load):
    from .geometric_fault_traces import validate_fault_trace_history
    if not isinstance(state, DistributedContactState):
        raise ValueError("Expected DistributedContactState; arc-mean history is not a point history")
    ids, _, torque, signature = _basal(surface, basal_load)
    if (state.surface_sha256 != surface_digest(surface) or state.basal_source_sha256 != signature
            or state.plate_ids != ids):
        raise ValueError("Distributed restart surface or frozen basal source changed")
    if (not isinstance(state.plate_ids, tuple) or any(isinstance(value, (bool, np.bool_))
            or not isinstance(value, Integral) for value in state.plate_ids)
            or not isinstance(state.states, tuple) or not isinstance(state.polarity_history, tuple)
            or not isinstance(state.rotation_radians, tuple)
            or any(not isinstance(row, tuple) for row in state.rotation_radians)):
        raise ValueError("Distributed coordinates and histories must be immutable tuples")
    points, materials, rows, areas, frames, offsets = _point_network(surface, state.fault_traces, ids)
    validate_fault_trace_history(state.fault_traces, state.polarity_history)
    expected = tuple(point.point_id for point in points)
    if tuple(identity for identity, _ in state.states) != expected:
        raise ValueError("Distributed point histories must contain every fixed quadrature point in canonical order")
    old_local = tuple(local for _, local in state.states)
    _validate_network_history(state, ids, torque, rows, areas, frames, materials, offsets, old_local)
    _closed_support(surface, state.fault_traces, ids, state.rotation_radians)
    return state


def initialize_distributed_contact_dynamics(surface, fault_traces, basal_load, *,
                                            initial_loading_dt_s, max_rotation_radians, history=()):
    """Prepare every fixed physical quadrature point with explicit trace anchors."""
    from .geometric_fault_traces import bind_fault_trace_history
    dt = _number(initial_loading_dt_s, "initial_loading_dt_s", positive=True)
    bound = _number(max_rotation_radians, "max_rotation_radians", positive=True)
    ids, _, _, signature = _basal(surface, basal_load)
    points, materials, _, areas, _, offsets = _point_network(surface, fault_traces, ids)
    evidence = bind_fault_trace_history(surface, fault_traces, tuple(history))
    states = []
    for point, material, area, offset in zip(points, materials, areas, offsets):
        step = solve_compressed_contact(columns=material.columns, law=material.law,
            area_m2=float(area), gap_free_m=float(offset[0]), jump_free_m=float(offset[1]),
            dt_s=dt, water=material.water)
        states.append((point.point_id, step.state))
    state = DistributedContactState(surface_digest(surface), signature, ids, fault_traces,
        evidence, tuple(states), tuple((0., 0., 0.) for _ in ids), dt, bound, 0., 0,
        math.fsum(local.cumulative_external_work_j for _, local in states), 0., 0.)
    return validate_distributed_contact_dynamics_state(surface, state, basal_load)


def _point_result(surface, point, step):
    normal, down = np.asarray(point.normal), np.asarray(point.down_dip)
    tn, tt = step.traction_pa
    force = point.area_m2*(tn*normal+tt*down)
    torque = np.cross(surface.radius_km*1000.*np.asarray(point.position), force)
    tensor = -tn*np.outer(normal, normal)-tt*(np.outer(down, normal)+np.outer(normal, down))
    if not np.isfinite(force).all() or not np.isfinite(torque).all():
        raise ValueError("Distributed physical point reaction overflowed")
    return PointContactResult(surface_digest(surface), point.point_id, point.trace_id,
        point.position, point.area_m2, step, tuple(tuple(map(float, row)) for row in tensor),
        tuple(map(float, force)), tuple(map(float, -force)), tuple(map(float, torque)),
        tuple(map(float, -torque)), -step.slip_increment_m)


def advance_distributed_contact_dynamics(surface, previous, basal_load, dt_s):
    """Solve the point network; preserve all histories on any rejected trial."""
    dt = _number(dt_s, "dt_s", positive=True)
    validate_distributed_contact_dynamics_state(surface, previous, basal_load)
    ids, drag, torque, _ = _basal(surface, basal_load)
    points, materials, rows, areas, frames, offsets = _point_network(surface, previous.fault_traces, ids)
    local_old = tuple(local for _, local in previous.states)
    qold = np.asarray(previous.rotation_radians, dtype=float).reshape(-1)
    solved = _equilibrate_contact_network(qold, drag, torque, rows, areas, frames, materials,
        offsets, local_old, dt, previous.max_rotation_radians,
        domain_check=lambda q: _closed_support(surface, previous.fault_traces, ids, q))
    metrics = solved.diagnostics
    state = replace(previous, states=tuple((point.point_id, step.state)
        for point, step in zip(points, solved.steps)),
        rotation_radians=tuple(tuple(map(float, row)) for row in solved.q.reshape((-1, 3))),
        elapsed_time_s=math.fsum((previous.elapsed_time_s, dt)), step_count=previous.step_count+1,
        cumulative_source_work_j=math.fsum((previous.cumulative_source_work_j, metrics["source_work_j"])),
        cumulative_basal_dissipation_j=math.fsum((previous.cumulative_basal_dissipation_j, metrics["basal_dissipation_j"])))
    validate_distributed_contact_dynamics_state(surface, state, basal_load)
    results = tuple(_point_result(surface, point, step) for point, step in zip(points, solved.steps))
    physical = np.zeros((len(ids), 3))
    positions = {identity: index for index, identity in enumerate(ids)}
    for result, material in zip(results, materials):
        physical[positions[material.subducting_plate]] += result.torque_on_sub_nm
        physical[positions[material.overriding_plate]] += result.torque_on_over_nm
    scale = max(float(np.linalg.norm(solved.contact_force)),
        math.fsum(float(np.linalg.norm(result.torque_on_sub_nm)) for result in results), np.finfo(float).tiny)
    if float(np.linalg.norm(physical.reshape(-1)+solved.contact_force)) > 512*np.finfo(float).eps*scale:
        raise UnresolvedContactDynamics("Point reactions disagree with the Euler virtual-work operator")
    triples = lambda value: tuple(tuple(map(float, row)) for row in np.asarray(value).reshape((-1, 3)))
    return DistributedContactStep(state, triples(solved.omega), results, triples(solved.drag_force),
        triples(physical), triples(solved.residual), **metrics)


def distributed_contact_dynamics_to_dict(surface, state, basal_load):
    """Store point history in metadata of the established joint checkpoint."""
    import json
    from .geometric_fault_traces import fault_trace_set_to_dict, trace_history_to_dict
    validate_distributed_contact_dynamics_state(surface, state, basal_load)
    result = asdict(state)
    result["fault_traces"] = fault_trace_set_to_dict(surface, state.fault_traces)
    result["polarity_history"] = trace_history_to_dict(state.polarity_history)
    result["states"] = [[identity, state_to_dict(local)] for identity, local in state.states]
    result.update(format=FORMAT, coordinate_convention=COORDINATES)
    return json.loads(_encode(result))


def distributed_contact_dynamics_from_dict(raw, surface, basal_load):
    from .geometric_fault_traces import fault_trace_set_from_dict, trace_history_from_dict
    names = {field.name for field in fields(DistributedContactState)}
    if (not isinstance(raw, dict) or set(raw) != names|{"format", "coordinate_convention"}
            or raw["format"] != FORMAT or raw["coordinate_convention"] != COORDINATES):
        raise ValueError("Invalid distributed contact schema; no mean-history migration or automatic regridding")
    values = {name: raw[name] for name in names}
    try:
        values["fault_traces"] = fault_trace_set_from_dict(values["fault_traces"], surface)
        values["polarity_history"] = trace_history_from_dict(values["polarity_history"], values["fault_traces"])
        if (not isinstance(values["plate_ids"], list) or any(isinstance(value, bool)
                or not isinstance(value, int) for value in values["plate_ids"])):
            raise ValueError("Distributed compact owners must be an integer list")
        values["plate_ids"] = tuple(values["plate_ids"])
        if not isinstance(values["rotation_radians"], list) or not isinstance(values["states"], list):
            raise ValueError("Distributed rotations and point histories must be explicit lists")
        values["rotation_radians"] = tuple(tuple(row) for row in values["rotation_radians"])
        if any(not isinstance(item, list) or len(item) != 2 or not isinstance(item[0], str)
               for item in values["states"]):
            raise ValueError("Distributed point histories require explicit point identities")
        values["states"] = tuple((identity, state_from_dict(local)) for identity, local in values["states"])
        state = DistributedContactState(**values)
        return validate_distributed_contact_dynamics_state(surface, state, basal_load)
    except (TypeError, KeyError) as error:
        raise ValueError("Malformed distributed contact checkpoint") from error
