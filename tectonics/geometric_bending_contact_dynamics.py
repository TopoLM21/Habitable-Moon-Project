"""Frozen physical fault contacts coupled to two elastic cantilever banks.

Each bank has one outward radial tip displacement, uniform along its declared
physical trace. The exact clamped cantilever shape supplies bending stiffness;
no transverse foundation, damping or slab load is added. The shared contact
engine owns the local return map, Newton solve and incremental work checks.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import hashlib
import json
import math
from numbers import Integral

import numpy as np

from .geometric_boundary import surface_digest
from .geometric_contact_dynamics import (
    UnresolvedContactDynamics, _basal, _encode, _equilibrate_contact_network,
    _number, _validate_network_history,
)
from .geometric_contact_mechanics import (
    LocalContactState, solve_compressed_contact, state_from_dict, state_to_dict,
)
from .geometric_distributed_contact_dynamics import (
    PointContactResult, _point_network, _point_result, continuous_contact_gap_bounds,
)
from .geometric_bending import (
    FaultBendingParameters, bank_max_surface_strain, bank_max_tip_slope,
    bending_parameters_from_dict, bending_parameters_to_dict,
    bending_strip_stiffness_n_m,
)


FORMAT = "geometric-frozen-bending-contact-dynamics-1"
COORDINATES = "fixed_frame_small_euler_rotations_rad_radial_cantilever_tip_metres"


@dataclass(frozen=True, slots=True)
class BendingContactState:
    surface_sha256: str
    basal_source_sha256: str
    bending_source_sha256: str
    plate_ids: tuple[int, ...]
    fault_traces: object
    polarity_history: tuple
    bending_parameters: tuple[FaultBendingParameters, ...]
    states: tuple[tuple[str, LocalContactState], ...]
    rotation_radians: tuple[tuple[float, float, float], ...]
    transverse_displacement_m: tuple[tuple[str, float, float], ...]
    initial_loading_dt_s: float
    max_rotation_radians: float
    max_tip_slope: float
    max_bending_strain: float
    max_deflection_radius_ratio: float
    elapsed_time_s: float
    step_count: int
    preparation_work_j: float
    cumulative_source_work_j: float
    cumulative_basal_dissipation_j: float
    cumulative_bending_numerical_dissipation_j: float


@dataclass(frozen=True, slots=True)
class BendingContactStep:
    """Accepted balances with physical bank forces positive outward.

    ``bending_work_j`` is the endpoint elastic-gradient work, equal to the
    bending energy increment plus its backward Euler remainder. The work of
    the reported physical restoring forces has the opposite sign. Total
    stored energy and numerical dissipation include contact and bending.
    """
    state: BendingContactState
    angular_velocity_rad_s: tuple[tuple[float, float, float], ...]
    transverse_velocity_m_s: tuple[tuple[str, float, float], ...]
    points: tuple[PointContactResult, ...]
    basal_drag_torque_nm: tuple[tuple[float, float, float], ...]
    contact_torque_nm: tuple[tuple[float, float, float], ...]
    torque_residual_nm: tuple[tuple[float, float, float], ...]
    bending_force_n: tuple[tuple[str, float, float], ...]
    radial_contact_force_n: tuple[tuple[str, float, float], ...]
    transverse_residual_n: tuple[tuple[str, float, float], ...]
    transverse_force_relative_residual: float
    transverse_force_rounding_tolerance_n: float
    generalized_force_relative_residual: float
    bending_work_j: float
    bending_stored_energy_increment_j: float
    bending_numerical_dissipation_j: float
    bending_energy_j: float
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


def _canonical_bending_parameters(trace_set, supplied, *, immutable=False):
    if immutable and not isinstance(supplied, tuple):
        raise ValueError("Bending parameters must be an immutable tuple")
    parameters = tuple(supplied)
    if any(not isinstance(item, FaultBendingParameters) for item in parameters):
        raise ValueError("Each physical trace requires explicit FaultBendingParameters")
    canonical = tuple(sorted(parameters, key=lambda item: item.trace_id))
    expected = tuple(trace.trace_id for trace in trace_set.traces)
    if tuple(item.trace_id for item in canonical) != expected:
        raise ValueError("Bending parameters must name every physical trace exactly once")
    # The bending module owns all bank material and beam-domain validation.
    restored = bending_parameters_from_dict(bending_parameters_to_dict(canonical))
    if restored != canonical:
        raise ValueError("Bending parameters have no canonical exact representation")
    if immutable and canonical != parameters:
        raise ValueError("Bending parameters must follow canonical physical trace order")
    return canonical


def _bending_signature(surface, traces, parameters, bounds):
    payload = dict(coordinate_convention=COORDINATES,
        physical_trace_sha256=traces.trace_sha256, radius_km=surface.radius_km,
        bending_parameters=bending_parameters_to_dict(parameters),
        validity_bounds=dict(zip(("max_rotation_radians", "max_tip_slope",
            "max_bending_strain", "max_deflection_radius_ratio"), bounds)))
    return hashlib.sha256(_encode(payload)).hexdigest()


def _transverse_coordinates(trace_set, supplied):
    if (not isinstance(supplied, tuple) or any(not isinstance(row, tuple) or len(row) != 3
            for row in supplied)
            or tuple(row[0] for row in supplied) != tuple(trace.trace_id for trace in trace_set.traces)):
        raise ValueError("Transverse displacements require every physical trace in canonical immutable order")
    return np.asarray([[_number(row[1], "subducting tip displacement"),
                        _number(row[2], "overriding tip displacement")] for row in supplied])


def _augmented_point_network(surface, trace_set, plate_ids, bending_parameters):
    points, materials, rows, areas, frames, offsets = _point_network(surface, trace_set, plate_ids)
    parameters = _canonical_bending_parameters(trace_set, bending_parameters, immutable=True)
    rotation_count, radius_m = 3*len(plate_ids), surface.radius_km*1000.
    size = rotation_count+2*len(parameters)
    padded_rows = np.zeros((len(points), size))
    padded_rows[:, :rotation_count] = rows
    operators = frames[:, :, None]*padded_rows[:, None, :]
    trace_positions = {parameter.trace_id: index for index, parameter in enumerate(parameters)}
    for index, material in enumerate(materials):
        start = rotation_count+2*trace_positions[material.trace_id]
        sine, cosine = frames[index]
        operators[index, :, start] = radius_m*np.asarray((-cosine, sine))
        operators[index, :, start+1] = radius_m*np.asarray((cosine, -sine))
    stiffness = np.zeros((size, size))
    for index, (trace, parameter) in enumerate(zip(trace_set.traces, parameters)):
        angle = math.atan2(float(np.linalg.norm(np.cross(trace.start, trace.end))),
                           float(np.asarray(trace.start)@np.asarray(trace.end)))
        length_m = radius_m*angle
        start = rotation_count+2*index
        for bank_index, bank in enumerate((parameter.sub, parameter.over)):
            stiffness[start+bank_index, start+bank_index] = (
                bending_strip_stiffness_n_m(bank, length_m)*radius_m*radius_m)
    if not np.isfinite(stiffness).all() or not np.isfinite(operators).all():
        raise ValueError("Augmented cantilever stiffness or contact operator overflowed")
    return points, materials, padded_rows, areas, frames, offsets, operators, stiffness


def continuous_bending_contact_gap_bounds(surface, trace_set, plate_ids,
                                          rotation_radians, transverse_displacement_m):
    """Exact full-arc gap extrema including uniform relative radial motion."""
    transverse = _transverse_coordinates(trace_set, transverse_displacement_m)
    horizontal = continuous_contact_gap_bounds(surface, trace_set, plate_ids, rotation_radians)
    result = []
    for trace, (identity, low, high), pair in zip(trace_set.traces, horizontal, transverse):
        shift = math.cos(trace.dip_radians)*float(pair[1]-pair[0])
        low, high = low+shift, high+shift
        if not math.isfinite(low) or not math.isfinite(high):
            raise ValueError("Continuous bending contact gap overflowed")
        result.append((identity, low, high))
    return tuple(result)


def _displacements(trace_set, value):
    return tuple((trace.trace_id, float(row[0]), float(row[1]))
                 for trace, row in zip(trace_set.traces, np.asarray(value).reshape((-1, 2))))


def _bending_domain(surface, state, q_augmented):
    rotation_count = 3*len(state.plate_ids)
    rotations = np.asarray(q_augmented[:rotation_count]).reshape((-1, 3))
    transverse = np.asarray(q_augmented[rotation_count:]).reshape((-1, 2))*surface.radius_km*1000.
    if not np.isfinite(rotations).all() or not np.isfinite(transverse).all():
        raise ValueError("Bending contact coordinates must be finite")
    if np.max(np.linalg.norm(rotations, axis=1), initial=0.) > state.max_rotation_radians:
        raise UnresolvedContactDynamics("Cumulative rotation exceeds the fixed-frame validity bound")
    radius_m = surface.radius_km*1000.
    for parameter, pair in zip(state.bending_parameters, transverse):
        for bank, displacement in zip((parameter.sub, parameter.over), pair):
            if bank_max_tip_slope(bank, float(displacement)) > state.max_tip_slope:
                raise UnresolvedContactDynamics("Cantilever tip slope exceeds the explicit validity bound")
            if bank_max_surface_strain(bank, float(displacement)) > state.max_bending_strain:
                raise UnresolvedContactDynamics("Cantilever bending strain exceeds the explicit validity bound")
            if abs(float(displacement))/radius_m > state.max_deflection_radius_ratio:
                raise UnresolvedContactDynamics("Cantilever deflection exceeds the explicit radius ratio bound")
    bounds = continuous_bending_contact_gap_bounds(surface, state.fault_traces, state.plate_ids,
        rotations, _displacements(state.fault_traces, transverse))
    if any(high > 0. for _, _, high in bounds):
        raise UnresolvedContactDynamics("Continuous physical fault support requires unsupported normal opening")


def validate_bending_contact_dynamics_state(surface, state, basal_load):
    from .geometric_fault_traces import validate_fault_trace_history
    if not isinstance(state, BendingContactState):
        raise ValueError("Expected BendingContactState; frozen contact history cannot migrate automatically")
    ids, _, torque, basal_signature = _basal(surface, basal_load)
    if (state.surface_sha256 != surface_digest(surface) or state.basal_source_sha256 != basal_signature
            or state.plate_ids != ids):
        raise ValueError("Bending restart surface or frozen basal source changed")
    if (not isinstance(state.plate_ids, tuple) or any(isinstance(value, (bool, np.bool_))
            or not isinstance(value, Integral) for value in state.plate_ids)
            or not isinstance(state.states, tuple) or not isinstance(state.polarity_history, tuple)
            or not isinstance(state.rotation_radians, tuple)
            or any(not isinstance(row, tuple) for row in state.rotation_radians)):
        raise ValueError("Bending coordinates and histories must be immutable tuples")
    bounds = tuple(_number(getattr(state, name), name, positive=True) for name in (
        "max_rotation_radians", "max_tip_slope", "max_bending_strain", "max_deflection_radius_ratio"))
    _number(state.cumulative_bending_numerical_dissipation_j,
            "cumulative_bending_numerical_dissipation_j", nonnegative=True)
    parameters = _canonical_bending_parameters(state.fault_traces, state.bending_parameters, immutable=True)
    if state.bending_source_sha256 != _bending_signature(surface, state.fault_traces, parameters, bounds):
        raise ValueError("Stored bending source changed its bank parameters or validity bounds")
    network = _augmented_point_network(surface, state.fault_traces, ids, parameters)
    points, materials, rows, areas, frames, offsets, operators, stiffness = network
    validate_fault_trace_history(state.fault_traces, state.polarity_history)
    if (any(not isinstance(row, tuple) or len(row) != 2 or not isinstance(row[0], str)
            for row in state.states)
            or tuple(identity for identity, _ in state.states) != tuple(point.point_id for point in points)):
        raise ValueError("Bending point histories require every fixed quadrature point in canonical order")
    transverse = _transverse_coordinates(state.fault_traces, state.transverse_displacement_m)
    q_augmented = np.concatenate((np.asarray(state.rotation_radians).reshape(-1),
                                  transverse.reshape(-1)/(surface.radius_km*1000.)))
    augmented_torque = np.pad(torque, (0, 2*len(parameters)))
    _validate_network_history(state, ids, augmented_torque, rows, areas, frames, materials, offsets,
        tuple(local for _, local in state.states), augmented_coordinates=q_augmented,
        anchor_operators=operators, elastic_stiffness=stiffness,
        cumulative_elastic_numerical_dissipation_j=state.cumulative_bending_numerical_dissipation_j)
    if state.step_count == 0 and (np.any(transverse != 0.)
            or state.cumulative_bending_numerical_dissipation_j != 0.):
        raise ValueError("Prepared cantilevers must start at zero displacement and numerical dissipation")
    _bending_domain(surface, state, q_augmented)
    return state


def initialize_bending_contact_dynamics(surface, fault_traces, basal_load, bending_parameters, *,
                                        initial_loading_dt_s, max_rotation_radians, max_tip_slope,
                                        max_bending_strain, max_deflection_radius_ratio, history=()):
    """Prepare contact anchors while both banks are held at zero displacement."""
    from .geometric_fault_traces import bind_fault_trace_history
    dt = _number(initial_loading_dt_s, "initial_loading_dt_s", positive=True)
    bounds = tuple(_number(value, name, positive=True) for name, value in (
        ("max_rotation_radians", max_rotation_radians), ("max_tip_slope", max_tip_slope),
        ("max_bending_strain", max_bending_strain),
        ("max_deflection_radius_ratio", max_deflection_radius_ratio)))
    ids, _, _, basal_signature = _basal(surface, basal_load)
    parameters = _canonical_bending_parameters(fault_traces, bending_parameters)
    points, materials, _, areas, _, offsets, _, _ = _augmented_point_network(surface, fault_traces, ids, parameters)
    evidence = bind_fault_trace_history(surface, fault_traces, tuple(history))
    local = []
    for point, material, area, offset in zip(points, materials, areas, offsets):
        step = solve_compressed_contact(columns=material.columns, law=material.law,
            area_m2=float(area), gap_free_m=float(offset[0]), jump_free_m=float(offset[1]),
            dt_s=dt, water=material.water)
        local.append((point.point_id, step.state))
    state = BendingContactState(surface_digest(surface), basal_signature,
        _bending_signature(surface, fault_traces, parameters, bounds), ids, fault_traces,
        evidence, parameters, tuple(local), tuple((0., 0., 0.) for _ in ids),
        tuple((trace.trace_id, 0., 0.) for trace in fault_traces.traces), dt, *bounds, 0., 0,
        math.fsum(value.cumulative_external_work_j for _, value in local), 0., 0., 0.)
    return validate_bending_contact_dynamics_state(surface, state, basal_load)


def advance_bending_contact_dynamics(surface, previous, basal_load, dt_s):
    """Release the radial tips and solve Euler/contact/bending work atomically."""
    dt = _number(dt_s, "dt_s", positive=True)
    validate_bending_contact_dynamics_state(surface, previous, basal_load)
    ids, drag, torque, _ = _basal(surface, basal_load)
    network = _augmented_point_network(surface, previous.fault_traces, ids, previous.bending_parameters)
    points, materials, rows, areas, frames, offsets, operators, stiffness = network
    rotation_count, radius_m = 3*len(ids), surface.radius_km*1000.
    size = stiffness.shape[0]
    augmented_drag = np.zeros((size, size))
    augmented_drag[:rotation_count, :rotation_count] = drag
    augmented_torque = np.pad(torque, (0, size-rotation_count))
    old_transverse = _transverse_coordinates(previous.fault_traces, previous.transverse_displacement_m)
    qold = np.concatenate((np.asarray(previous.rotation_radians).reshape(-1), old_transverse.reshape(-1)/radius_m))
    solved = _equilibrate_contact_network(qold, augmented_drag, augmented_torque, rows, areas,
        frames, materials, offsets, tuple(local for _, local in previous.states), dt,
        previous.max_rotation_radians, anchor_operators=operators, elastic_stiffness=stiffness,
        rotation_dof_count=rotation_count, domain_check=lambda q: _bending_domain(surface, previous, q))
    metrics = dict(solved.diagnostics)
    bending_increment = metrics.pop("bending_energy_increment_j")
    bending_work = metrics.pop("bending_work_j")
    bending_numerical = metrics.pop("bending_numerical_dissipation_j")
    triples = lambda value: tuple(tuple(map(float, row)) for row in np.asarray(value).reshape((-1, 3)))
    state = replace(previous,
        states=tuple((point.point_id, step.state) for point, step in zip(points, solved.steps)),
        rotation_radians=triples(solved.q[:rotation_count]),
        transverse_displacement_m=_displacements(previous.fault_traces, solved.q[rotation_count:]*radius_m),
        elapsed_time_s=math.fsum((previous.elapsed_time_s, dt)), step_count=previous.step_count+1,
        cumulative_source_work_j=math.fsum((previous.cumulative_source_work_j, metrics["source_work_j"])),
        cumulative_basal_dissipation_j=math.fsum((previous.cumulative_basal_dissipation_j,
                                               metrics["basal_dissipation_j"])),
        cumulative_bending_numerical_dissipation_j=math.fsum((
            previous.cumulative_bending_numerical_dissipation_j, bending_numerical)))
    validate_bending_contact_dynamics_state(surface, state, basal_load)
    results = tuple(_point_result(surface, point, step) for point, step in zip(points, solved.steps))
    physical_torque = np.zeros((len(ids), 3))
    radial_force = np.zeros((len(previous.bending_parameters), 2))
    plate_positions = {identity: index for index, identity in enumerate(ids)}
    trace_positions = {trace.trace_id: index for index, trace in enumerate(previous.fault_traces.traces)}
    for result, material in zip(results, materials):
        physical_torque[plate_positions[material.subducting_plate]] += result.torque_on_sub_nm
        physical_torque[plate_positions[material.overriding_plate]] += result.torque_on_over_nm
        radial = float(np.asarray(result.position)@np.asarray(result.force_on_sub_n))
        radial_force[trace_positions[result.trace_id]] += (radial, -radial)
    torque_scale = max(float(np.linalg.norm(solved.contact_force[:rotation_count])),
        math.fsum(float(np.linalg.norm(point.torque_on_sub_nm)) for point in results), np.finfo(float).tiny)
    if float(np.linalg.norm(physical_torque.reshape(-1)+solved.contact_force[:rotation_count])) > 512*np.finfo(float).eps*torque_scale:
        raise UnresolvedContactDynamics("Point moments disagree with the Euler virtual-work operator")
    radial_scale = max(float(np.linalg.norm(solved.contact_force[rotation_count:])),
        radius_m*math.fsum(float(np.linalg.norm(point.force_on_sub_n)) for point in results), np.finfo(float).tiny)
    if float(np.linalg.norm(radius_m*radial_force.reshape(-1)+solved.contact_force[rotation_count:])) > 512*np.finfo(float).eps*radial_scale:
        raise UnresolvedContactDynamics("Point radial reactions disagree with the cantilever virtual-work operator")
    bending_force = -np.asarray(solved.elastic_force[rotation_count:]).reshape((-1, 2))/radius_m
    radial_residual = bending_force+radial_force
    radial_signal = max(float(np.linalg.norm(bending_force)), float(np.linalg.norm(radial_force)), np.finfo(float).tiny)
    radial_operands = float(np.linalg.norm(bending_force))+math.fsum(
        abs(float(np.asarray(point.position)@np.asarray(point.force_on_sub_n)))*math.sqrt(2.) for point in results)
    radial_rounding = 512*np.finfo(float).eps*radial_operands
    radial_error = float(np.linalg.norm(radial_residual))/radial_signal
    plate_residual = solved.residual[:rotation_count]
    plate_drag = solved.drag_force[:rotation_count]
    plate_contact = solved.contact_force[:rotation_count]
    plate_signal = max(float(np.linalg.norm(torque)), float(np.linalg.norm(plate_drag)),
        float(np.linalg.norm(plate_contact)), np.finfo(float).tiny)
    plate_operands = float(np.linalg.norm(torque))+float(np.linalg.norm(plate_drag))+math.fsum(
        math.sqrt(2.)*float(np.linalg.norm(point.torque_on_sub_nm)) for point in results)
    plate_rounding = 512*np.finfo(float).eps*plate_operands
    plate_magnitude = float(np.linalg.norm(plate_residual))
    if (float(np.linalg.norm(radial_residual)) > 2e-10*radial_signal+radial_rounding
            or plate_magnitude > 2e-10*plate_signal+plate_rounding):
        raise UnresolvedContactDynamics("Separate Euler and transverse force balances failed certification")
    generalized_error = metrics["torque_relative_residual"]
    metrics["torque_relative_residual"] = plate_magnitude/max(plate_operands, np.finfo(float).tiny)
    metrics["torque_signal_relative_residual"] = plate_magnitude/plate_signal
    metrics["torque_rounding_tolerance_nm"] = plate_rounding
    bending_energy = float(.5*solved.q@stiffness@solved.q)
    return BendingContactStep(state=state, angular_velocity_rad_s=triples(solved.omega[:rotation_count]),
        transverse_velocity_m_s=_displacements(previous.fault_traces, solved.omega[rotation_count:]*radius_m),
        points=results, basal_drag_torque_nm=triples(plate_drag), contact_torque_nm=triples(physical_torque),
        torque_residual_nm=triples(plate_residual),
        bending_force_n=_displacements(previous.fault_traces, bending_force),
        radial_contact_force_n=_displacements(previous.fault_traces, radial_force),
        transverse_residual_n=_displacements(previous.fault_traces, radial_residual),
        transverse_force_relative_residual=radial_error,
        transverse_force_rounding_tolerance_n=radial_rounding,
        generalized_force_relative_residual=generalized_error, bending_work_j=bending_work,
        bending_stored_energy_increment_j=bending_increment,
        bending_numerical_dissipation_j=bending_numerical, bending_energy_j=bending_energy, **metrics)


def bending_contact_dynamics_to_dict(surface, state, basal_load):
    """Explicit metadata for the established joint material checkpoint."""
    from .geometric_fault_traces import fault_trace_set_to_dict, trace_history_to_dict
    validate_bending_contact_dynamics_state(surface, state, basal_load)
    result = asdict(state)
    result["fault_traces"] = fault_trace_set_to_dict(surface, state.fault_traces)
    result["polarity_history"] = trace_history_to_dict(state.polarity_history)
    result["bending_parameters"] = bending_parameters_to_dict(state.bending_parameters)
    result["states"] = [[identity, state_to_dict(local)] for identity, local in state.states]
    result.update(format=FORMAT, coordinate_convention=COORDINATES)
    return json.loads(_encode(result))


def bending_contact_dynamics_from_dict(raw, surface, basal_load):
    from .geometric_fault_traces import fault_trace_set_from_dict, trace_history_from_dict
    names = {field.name for field in fields(BendingContactState)}
    if (not isinstance(raw, dict) or set(raw) != names|{"format", "coordinate_convention"}
            or raw["format"] != FORMAT or raw["coordinate_convention"] != COORDINATES):
        raise ValueError("Invalid bending contact schema; no automatic history migration")
    values = {name: raw[name] for name in names}
    try:
        values["fault_traces"] = fault_trace_set_from_dict(values["fault_traces"], surface)
        values["polarity_history"] = trace_history_from_dict(values["polarity_history"], values["fault_traces"])
        values["bending_parameters"] = bending_parameters_from_dict(values["bending_parameters"])
        if (not isinstance(values["plate_ids"], list) or any(isinstance(value, bool)
                or not isinstance(value, int) for value in values["plate_ids"])):
            raise ValueError("Bending compact owners must be an explicit integer list")
        values["plate_ids"] = tuple(values["plate_ids"])
        if (not isinstance(values["rotation_radians"], list) or any(not isinstance(row, list)
                for row in values["rotation_radians"])
                or not isinstance(values["transverse_displacement_m"], list)
                or any(not isinstance(row, list) for row in values["transverse_displacement_m"])
                or not isinstance(values["states"], list)):
            raise ValueError("Bending rotations, radial displacements and point histories require explicit lists")
        values["rotation_radians"] = tuple(tuple(row) for row in values["rotation_radians"])
        values["transverse_displacement_m"] = tuple(tuple(row) for row in values["transverse_displacement_m"])
        if any(not isinstance(row, list) or len(row) != 2 or not isinstance(row[0], str)
               for row in values["states"]):
            raise ValueError("Bending point histories require explicit point identities")
        values["states"] = tuple((identity, state_from_dict(local)) for identity, local in values["states"])
        return validate_bending_contact_dynamics_state(surface, BendingContactState(**values), basal_load)
    except (TypeError, KeyError, OverflowError) as error:
        raise ValueError("Malformed bending contact checkpoint") from error
