"""Incremental bending/contact mechanics with individually bound sources.

Every segment uses an explicit reference configuration carrying the original
material support. It may be a passive Cartesian view or a previously accepted
rigid material epoch; constructing its view adds no physical duration. The
mechanical increment carries its own source work; old loading work, plastic
memory and absolute cantilever deflections are never re-prepared. Spatial
contact rematching and physical advection are owned by the caller's separate,
certified material transaction, rather than inferred from a coordinate view.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import hashlib
import json
import math
from numbers import Integral

import numpy as np

from .geometric_basal_sources import basal_source_from_dict, basal_source_to_dict
from .geometric_bending import bank_max_surface_strain, bank_max_tip_slope
from .geometric_bending_contact_dynamics import (
    BendingContactState, BendingContactStep, _displacements,
    _transverse_coordinates, bending_contact_dynamics_from_dict,
    bending_contact_dynamics_to_dict, validate_bending_contact_dynamics_state,
)
from .geometric_bending_reference import (
    BendingReference, bending_reference_from_dict, bending_reference_to_dict,
    current_bending_reference, validate_reference_rotation,
)
from .geometric_boundary import surface_digest
from .geometric_contact_dynamics import _basal, _encode, _equilibrate_contact_network, _number, UnresolvedContactDynamics
from .geometric_contact_mechanics import state_from_dict, state_to_dict, validate_contact_state
from .geometric_distributed_contact_dynamics import _point_result
from .geometric_fault_traces import _arc_frame
from .geometric_rigid_mantle_loading import PhysicalRigidMantleLoad


FORMAT = "geometric-segmented-bending-contact-dynamics-1"
MODEL = "incremental_source_segments_certified_material_reference_configurations"
COORDINATES = "segment_cartesian_euler_increment_absolute_radial_tip_displacement"
_EPS = np.finfo(float).eps


@dataclass(frozen=True, slots=True)
class TraceAnchorHarmonics:
    """Free anchors c+a*cos(theta)+b*sin(theta) on the original minor arc."""
    trace_id: str
    gap: tuple[float, float, float]
    jump: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class SegmentedSourceReceipt:
    sequence: int
    reference_rotation: tuple[tuple[float, float, float], ...]
    current_load: object
    reference_configuration: object
    dt_s: float
    rotation_increment_radians: tuple[tuple[float, float, float], ...]
    transverse_displacement_m: tuple[tuple[str, float, float], ...]
    source_work_j: float
    basal_dissipation_j: float
    bending_numerical_dissipation_j: float
    before_mechanics_sha256: str
    after_mechanics_sha256: str
    provenance: str
    receipt_id: str
    payload_sha256: str


@dataclass(frozen=True, slots=True)
class SegmentedBendingState:
    origin_surface_sha256: str
    initial_mechanics: BendingContactState
    initial_load: object
    states: tuple
    transverse_displacement_m: tuple[tuple[str, float, float], ...]
    free_anchor_harmonics: tuple[TraceAnchorHarmonics, ...]
    rotation_path_material_radians: tuple[tuple[float, float, float], ...]
    elapsed_time_s: float
    step_count: int
    cumulative_source_work_j: float
    cumulative_basal_dissipation_j: float
    cumulative_bending_numerical_dissipation_j: float
    segments: tuple[SegmentedSourceReceipt, ...]

    @property
    def plate_ids(self):
        return self.initial_mechanics.plate_ids

    @property
    def preparation_work_j(self):
        return self.initial_mechanics.preparation_work_j

    @property
    def segment_count(self):
        return len(self.segments)


@dataclass(frozen=True, slots=True)
class SegmentedBendingStep(BendingContactStep):
    state: SegmentedBendingState
    reference_rotation: tuple[tuple[float, float, float], ...]
    rotation_increment_radians: tuple[tuple[float, float, float], ...]
    reference_surface_sha256: str


def _hash(value):
    return hashlib.sha256(_encode(value)).hexdigest()


def _mechanics_body(state):
    body = asdict(state)
    return {field.name: body[field.name] for field in fields(state)
            if field.name not in ("initial_mechanics", "initial_load", "segments")}


def _mechanics_hash(state):
    return _hash(_mechanics_body(state))


def _segment_body(receipt):
    body = asdict(receipt)
    body.pop("receipt_id")
    body.pop("payload_sha256")
    return dict(format="geometric-segmented-bending-source-receipt-1", model=MODEL, **body)


def _seal_segment(receipt):
    digest = _hash(_segment_body(receipt))
    return replace(receipt, payload_sha256=digest,
                   receipt_id=f"bending-source-segment:{receipt.sequence}:{digest}")


def _triples(value):
    return tuple(tuple(map(float, row)) for row in np.asarray(value).reshape((-1, 3)))


def _text(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Every source segment requires explicit provenance")
    return value


def _update_harmonics(seed, old, material_rotations, radial_increment):
    """Analytic virtual-displacement trace, never an interpolation of nodes."""
    positions = {owner: index for index, owner in enumerate(seed.plate_ids)}
    radius = seed.fault_traces.radius_km*1000.
    result = []
    for trace, prior, radial in zip(seed.fault_traces.traces, old, radial_increment):
        start, horizontal, tangent, _ = _arc_frame(trace.start, trace.end)
        relative = (material_rotations[positions[trace.overriding_plate]]
                    -material_rotations[positions[trace.subducting_plate]])
        vector = radius*np.cross(horizontal, relative)
        first, second = float(start@vector), float(tangent@vector)
        sine, cosine = math.sin(trace.dip_radians), math.cos(trace.dip_radians)
        z = float(radial[1]-radial[0])
        gap_delta, jump_delta = (cosine*z, sine*first, sine*second), (-sine*z, cosine*first, cosine*second)
        result.append(TraceAnchorHarmonics(trace.trace_id,
            tuple(math.fsum((a, b)) for a, b in zip(prior.gap, gap_delta)),
            tuple(math.fsum((a, b)) for a, b in zip(prior.jump, jump_delta))))
    return tuple(result)


def _seed_harmonics(seed):
    initial = tuple(TraceAnchorHarmonics(trace.trace_id,
        (trace.initial_gap_m, 0., 0.), (trace.initial_jump_m, 0., 0.))
        for trace in seed.fault_traces.traces)
    radial = _transverse_coordinates(seed.fault_traces, seed.transverse_displacement_m)
    return _update_harmonics(seed, initial, np.asarray(seed.rotation_radians), radial)


def continuous_segmented_gap_bounds(state):
    """Endpoint and interior extrema of the complete evolved free gap field."""
    result = []
    for trace, harmonic in zip(state.initial_mechanics.fault_traces.traces, state.free_anchor_harmonics):
        _, _, _, extent = _arc_frame(trace.start, trace.end)
        constant, first, second = harmonic.gap
        values = [constant+first, constant+first*math.cos(extent)+second*math.sin(extent)]
        stationary = math.atan2(second, first)
        for index in range(-2, 3):
            angle = stationary+index*math.pi
            if 0. < angle < extent:
                values.append(constant+first*math.cos(angle)+second*math.sin(angle))
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Continuous evolved free gap overflowed")
        result.append((trace.trace_id, min(values), max(values)))
    return tuple(result)


def _domain(state, material_rotations, radial):
    """Bound this segment's increment and absolute bending deformation.

    ``material_rotations`` never contains the accumulated diagnostic path.
    Finite common rotation belongs to the separately certified material pose;
    its cumulative angle is not an elastic deformation or this chart's step.
    """
    seed, radius = state.initial_mechanics, state.initial_mechanics.fault_traces.radius_km*1000.
    if (not np.isfinite(material_rotations).all() or not np.isfinite(radial).all()
            or np.max(np.linalg.norm(material_rotations, axis=1), initial=0.) > seed.max_rotation_radians):
        raise UnresolvedContactDynamics("Source segment rotation exceeds its explicit small-increment bound")
    for parameter, pair in zip(seed.bending_parameters, radial):
        for bank, displacement in zip((parameter.sub, parameter.over), pair):
            if bank_max_tip_slope(bank, float(displacement)) > seed.max_tip_slope:
                raise UnresolvedContactDynamics("Cantilever tip slope exceeds the explicit validity bound")
            if bank_max_surface_strain(bank, float(displacement)) > seed.max_bending_strain:
                raise UnresolvedContactDynamics("Cantilever bending strain exceeds the explicit validity bound")
            if abs(float(displacement))/radius > seed.max_deflection_radius_ratio:
                raise UnresolvedContactDynamics("Cantilever deflection exceeds its explicit radius ratio bound")
    old_radial = _transverse_coordinates(seed.fault_traces, state.transverse_displacement_m)
    harmonics = _update_harmonics(seed, state.free_anchor_harmonics, material_rotations, radial-old_radial)
    trial = replace(state, free_anchor_harmonics=harmonics)
    if any(high > 0. for _, _, high in continuous_segmented_gap_bounds(trial)):
        raise UnresolvedContactDynamics("Continuous evolved physical support requires unsupported normal opening")
    return harmonics


def _validate_current(origin_surface, state):
    seed = state.initial_mechanics
    reference = current_bending_reference(origin_surface, seed)
    if (not isinstance(state.states, tuple)
            or any(not isinstance(row, tuple) or len(row) != 2 for row in state.states)
            or tuple(identity for identity, _ in state.states) != tuple(point.point_id for point in reference.points)):
        raise ValueError("Segmented histories must preserve every canonical physical point")
    radial = _transverse_coordinates(seed.fault_traces, state.transverse_displacement_m)
    path = state.rotation_path_material_radians
    if (not isinstance(path, tuple) or len(path) != len(seed.plate_ids)
            or any(not isinstance(row, tuple) or len(row) != 3 for row in path)):
        raise ValueError("Material rotation path must retain compact immutable owner triples")
    for row in path:
        for component in row:
            _number(component, "Material rotation path")
    if (isinstance(state.step_count, (bool, np.bool_)) or not isinstance(state.step_count, Integral)
            or state.step_count < seed.step_count):
        raise ValueError("Segmented mechanical step counter is invalid")
    for name in ("elapsed_time_s", "cumulative_basal_dissipation_j", "cumulative_bending_numerical_dissipation_j"):
        _number(getattr(state, name), name, nonnegative=True)
    _number(state.cumulative_source_work_j, "cumulative_source_work_j")
    harmonics = state.free_anchor_harmonics
    if (not isinstance(harmonics, tuple)
            or tuple(item.trace_id for item in harmonics if isinstance(item, TraceAnchorHarmonics))
            != tuple(trace.trace_id for trace in seed.fault_traces.traces)):
        raise ValueError("Every physical trace needs its exact analytic anchor field")
    for item in harmonics:
        for pair in (item.gap, item.jump):
            if not isinstance(pair, tuple) or len(pair) != 3:
                raise ValueError("Free anchor harmonics must be immutable triples")
            for component in pair:
                _number(component, "Free anchor coefficient")
    by_trace = {trace.trace_id: trace for trace in seed.fault_traces.traces}
    by_harmonic = {item.trace_id: item for item in harmonics}
    clock = seed.initial_loading_dt_s+state.elapsed_time_s
    for point, material, area, (_, local) in zip(reference.points, reference.materials, reference.areas, state.states):
        validate_contact_state(local, columns=material.columns, law=material.law,
                               area_m2=float(area), water=material.water)
        if local.step_count != state.step_count+1:
            raise ValueError("Global and local segmented mechanical counters disagree")
        scale = max(abs(clock), abs(local.elapsed_time_s), np.finfo(float).tiny)
        if abs(local.elapsed_time_s-clock) > 512*_EPS*scale:
            raise ValueError("Global and local segmented mechanical clocks disagree")
        start, _, tangent, _ = _arc_frame(by_trace[point.trace_id].start, by_trace[point.trace_id].end)
        position = np.asarray(point.position)
        angle = math.atan2(float(position@tangent), float(position@start))
        for actual, coefficients in zip((local.gap_free_m, local.jump_free_m),
                                       (by_harmonic[point.trace_id].gap, by_harmonic[point.trace_id].jump)):
            expected = math.fsum((coefficients[0], coefficients[1]*math.cos(angle), coefficients[2]*math.sin(angle)))
            magnitude = max(abs(actual), abs(expected), math.fsum(abs(value) for value in coefficients), np.finfo(float).tiny)
            if abs(actual-expected) > 512*_EPS*magnitude:
                raise ValueError("Saved local anchors disagree with the analytic material trace")
    _domain(state, np.zeros((len(seed.plate_ids), 3)), radial)
    stiffness = np.asarray(reference.stiffness)
    coords = np.r_[np.zeros(3*len(seed.plate_ids)), radial.reshape(-1)/(seed.fault_traces.radius_km*1000.)]
    energy = .5*float(coords@stiffness@coords)
    accounted = math.fsum((*(local.cumulative_external_work_j for _, local in state.states),
        state.cumulative_basal_dissipation_j, energy, state.cumulative_bending_numerical_dissipation_j))
    supplied = math.fsum((state.preparation_work_j, state.cumulative_source_work_j))
    if abs(accounted-supplied) > 5e-10*max(abs(accounted), abs(supplied), np.finfo(float).tiny):
        raise ValueError("Segmented cumulative source/contact/bending work balance is inconsistent")


def initialize_segmented_bending_dynamics(origin_surface, initial_mechanics, initial_load):
    """Adopt a validated, possibly evolved frozen history without changing it."""
    validate_bending_contact_dynamics_state(origin_surface, initial_mechanics, initial_load)
    state = SegmentedBendingState(surface_digest(origin_surface), initial_mechanics, initial_load,
        initial_mechanics.states, initial_mechanics.transverse_displacement_m,
        _seed_harmonics(initial_mechanics), initial_mechanics.rotation_radians,
        initial_mechanics.elapsed_time_s, initial_mechanics.step_count,
        initial_mechanics.cumulative_source_work_j, initial_mechanics.cumulative_basal_dissipation_j,
        initial_mechanics.cumulative_bending_numerical_dissipation_j, ())
    _validate_current(origin_surface, state)
    return state


def _reference(origin_surface, seed, rotation, *, reference_configuration=None,
               reference_elapsed_time_myr=0., reference_surface=None, reference_roundoff_steps=0):
    if reference_configuration is not None:
        view = reference_configuration
        if not isinstance(view, BendingReference):
            raise ValueError("Source segment requires an explicit BendingReference configuration")
        if view.rotation != rotation:
            raise ValueError("Source segment reference rotation differs from its configuration")
        actual = current_bending_reference(origin_surface, seed, rotation,
            reference_elapsed_time_myr=view.reference_elapsed_time_myr,
            reference_surface=view.surface if view.uses_supplied_reference_surface else None,
            reference_roundoff_steps=view.reference_roundoff_steps)
        if view != actual:
            raise ValueError("Source segment reference configuration changed its material binding")
        return actual
    return current_bending_reference(origin_surface, seed, rotation,
        reference_elapsed_time_myr=reference_elapsed_time_myr,
        reference_surface=reference_surface, reference_roundoff_steps=reference_roundoff_steps)


def _advance(origin_surface, previous, current_load, dt, rotation, provenance, *,
             reference_configuration=None, reference_elapsed_time_myr=0.,
             reference_surface=None, reference_roundoff_steps=0):
    seed = previous.initial_mechanics
    reference = _reference(origin_surface, seed, rotation,
        reference_configuration=reference_configuration,
        reference_elapsed_time_myr=reference_elapsed_time_myr,
        reference_surface=reference_surface, reference_roundoff_steps=reference_roundoff_steps)
    ids, drag, torque, _ = _basal(reference.surface, current_load)
    if ids != previous.plate_ids:
        raise ValueError("Source segment cannot change actual material owners")
    rc, radius = 3*len(ids), seed.fault_traces.radius_km*1000.
    rows, areas, frames = map(np.asarray, (reference.rows, reference.areas, reference.frames))
    operators, stiffness = map(np.asarray, (reference.operators, reference.stiffness))
    radial_old = _transverse_coordinates(seed.fault_traces, previous.transverse_displacement_m)
    zold = np.r_[np.zeros(rc), radial_old.reshape(-1)/radius]
    old_anchors = np.asarray([(local.gap_free_m, local.jump_free_m) for _, local in previous.states])
    offsets = old_anchors-np.einsum("ijk,k->ij", operators, zold)
    augmented_drag = np.zeros(stiffness.shape)
    augmented_drag[:rc, :rc] = drag
    augmented_torque = np.pad(torque, (0, len(zold)-rc))
    matrix = np.asarray(rotation)
    def domain(z):
        return _domain(previous, np.asarray(z[:rc]).reshape((-1, 3))@matrix,
                       np.asarray(z[rc:]).reshape((-1, 2))*radius)
    guess = None
    if isinstance(current_load, PhysicalRigidMantleLoad):
        guess = zold.copy()
        guess[:rc] = np.tile(np.asarray(current_load.source.angular_velocity_rad_s)*dt, len(ids))
    solved = _equilibrate_contact_network(zold, augmented_drag, augmented_torque, rows, areas,
        frames, reference.materials, offsets, tuple(local for _, local in previous.states), dt,
        seed.max_rotation_radians, anchor_operators=operators, elastic_stiffness=stiffness,
        rotation_dof_count=rc, domain_check=domain, initial_guess=guess,
        common_rotation_gauge=True)
    radial_new = solved.q[rc:].reshape((-1, 2))*radius
    material_increment = solved.q[:rc].reshape((-1, 3))@matrix
    harmonics = _domain(previous, material_increment, radial_new)
    metrics = dict(solved.diagnostics)
    bending_increment = metrics.pop("bending_energy_increment_j")
    bending_work = metrics.pop("bending_work_j")
    bending_numerical = metrics.pop("bending_numerical_dissipation_j")
    elapsed = math.fsum((previous.elapsed_time_s, dt))
    if elapsed <= previous.elapsed_time_s:
        raise ValueError("Source segment physical clock cannot resolve this interval")
    state = replace(previous,
        states=tuple((point.point_id, step.state) for point, step in zip(reference.points, solved.steps)),
        transverse_displacement_m=_displacements(seed.fault_traces, radial_new),
        free_anchor_harmonics=harmonics,
        rotation_path_material_radians=tuple(tuple(math.fsum((a, b)) for a, b in zip(old, change))
            for old, change in zip(previous.rotation_path_material_radians, material_increment)),
        elapsed_time_s=elapsed, step_count=previous.step_count+1,
        cumulative_source_work_j=math.fsum((previous.cumulative_source_work_j, metrics["source_work_j"])),
        cumulative_basal_dissipation_j=math.fsum((previous.cumulative_basal_dissipation_j, metrics["basal_dissipation_j"])),
        cumulative_bending_numerical_dissipation_j=math.fsum((previous.cumulative_bending_numerical_dissipation_j, bending_numerical)))
    _validate_current(origin_surface, state)
    increment = _triples(solved.q[:rc])
    receipt = _seal_segment(SegmentedSourceReceipt(len(previous.segments)+1, rotation,
        current_load, reference, dt, increment, state.transverse_displacement_m, metrics["source_work_j"],
        metrics["basal_dissipation_j"], bending_numerical, _mechanics_hash(previous),
        _mechanics_hash(state), provenance, "", ""))
    state = replace(state, segments=(*previous.segments, receipt))
    results = tuple(_point_result(reference.surface, point, step) for point, step in zip(reference.points, solved.steps))
    physical_torque, radial_force = np.zeros((len(ids), 3)), np.zeros((len(seed.bending_parameters), 2))
    owners = {owner: index for index, owner in enumerate(ids)}
    traces = {trace.trace_id: index for index, trace in enumerate(seed.fault_traces.traces)}
    for result, material in zip(results, reference.materials):
        physical_torque[owners[material.subducting_plate]] += result.torque_on_sub_nm
        physical_torque[owners[material.overriding_plate]] += result.torque_on_over_nm
        radial = float(np.asarray(result.position)@np.asarray(result.force_on_sub_n))
        radial_force[traces[result.trace_id]] += (radial, -radial)
    torque_scale = max(float(np.linalg.norm(solved.contact_force[:rc])),
        math.fsum(float(np.linalg.norm(point.torque_on_sub_nm)) for point in results), np.finfo(float).tiny)
    if float(np.linalg.norm(physical_torque.reshape(-1)+solved.contact_force[:rc])) > 512*_EPS*torque_scale:
        raise UnresolvedContactDynamics("Segment point moments disagree with virtual work")
    radial_scale = max(float(np.linalg.norm(solved.contact_force[rc:])),
        radius*math.fsum(float(np.linalg.norm(point.force_on_sub_n)) for point in results), np.finfo(float).tiny)
    if float(np.linalg.norm(radius*radial_force.reshape(-1)+solved.contact_force[rc:])) > 512*_EPS*radial_scale:
        raise UnresolvedContactDynamics("Segment radial reactions disagree with virtual work")
    bending_force = -solved.elastic_force[rc:].reshape((-1, 2))/radius
    radial_residual = bending_force+radial_force
    radial_signal = max(float(np.linalg.norm(bending_force)), float(np.linalg.norm(radial_force)), np.finfo(float).tiny)
    radial_operands = float(np.linalg.norm(bending_force))+math.fsum(
        math.sqrt(2.)*abs(float(np.asarray(point.position)@np.asarray(point.force_on_sub_n))) for point in results)
    radial_rounding = 512*_EPS*radial_operands
    plate_drag, plate_residual = solved.drag_force[:rc], solved.residual[:rc]
    plate_signal = max(float(np.linalg.norm(torque)), float(np.linalg.norm(plate_drag)),
                       float(np.linalg.norm(solved.contact_force[:rc])), np.finfo(float).tiny)
    plate_operands = float(np.linalg.norm(torque))+float(np.linalg.norm(plate_drag))+math.fsum(
        math.sqrt(2.)*float(np.linalg.norm(point.torque_on_sub_nm)) for point in results)
    plate_rounding = 512*_EPS*plate_operands
    magnitude = float(np.linalg.norm(plate_residual))
    if (float(np.linalg.norm(radial_residual)) > 2e-10*radial_signal+radial_rounding
            or magnitude > 2e-10*plate_signal+plate_rounding):
        raise UnresolvedContactDynamics("Separate segmented Euler/transverse balances failed")
    generalized_error = metrics["torque_relative_residual"]
    metrics.update(torque_relative_residual=magnitude/max(plate_operands, np.finfo(float).tiny),
        torque_signal_relative_residual=magnitude/plate_signal, torque_rounding_tolerance_nm=plate_rounding)
    return SegmentedBendingStep(state=state, angular_velocity_rad_s=_triples(solved.omega[:rc]),
        transverse_velocity_m_s=_displacements(seed.fault_traces, solved.omega[rc:]*radius),
        points=results, basal_drag_torque_nm=_triples(plate_drag), contact_torque_nm=_triples(physical_torque),
        torque_residual_nm=_triples(plate_residual), bending_force_n=_displacements(seed.fault_traces, bending_force),
        radial_contact_force_n=_displacements(seed.fault_traces, radial_force),
        transverse_residual_n=_displacements(seed.fault_traces, radial_residual),
        transverse_force_relative_residual=float(np.linalg.norm(radial_residual))/radial_signal,
        transverse_force_rounding_tolerance_n=radial_rounding, generalized_force_relative_residual=generalized_error,
        bending_work_j=bending_work, bending_stored_energy_increment_j=bending_increment,
        bending_numerical_dissipation_j=bending_numerical, bending_energy_j=.5*float(solved.q@stiffness@solved.q),
        reference_rotation=rotation, rotation_increment_radians=increment,
        reference_surface_sha256=surface_digest(reference.surface), **metrics)


def validate_segmented_bending_dynamics_state(origin_surface, state):
    """Replay original preparation and every independently bound source segment."""
    if not isinstance(state, SegmentedBendingState):
        raise ValueError("Expected a SegmentedBendingState")
    if state.origin_surface_sha256 != surface_digest(origin_surface):
        raise ValueError("Segmented mechanics belongs to another original surface")
    expected = initialize_segmented_bending_dynamics(origin_surface, state.initial_mechanics, state.initial_load)
    if not isinstance(state.segments, tuple):
        raise ValueError("Source segments must be an immutable ordered tuple")
    for index, receipt in enumerate(state.segments, start=1):
        if (not isinstance(receipt, SegmentedSourceReceipt)
                or isinstance(receipt.sequence, (bool, np.bool_))
                or not isinstance(receipt.sequence, Integral) or receipt.sequence != index):
            raise ValueError("Source segment sequence is invalid")
        dt = _number(receipt.dt_s, "Source segment duration", positive=True)
        if not isinstance(receipt.reference_configuration, BendingReference):
            raise ValueError("Source segment requires an explicit reference configuration")
        rotation = validate_reference_rotation(receipt.reference_rotation,
            reference_roundoff_steps=receipt.reference_configuration.reference_roundoff_steps)
        if receipt.reference_rotation != rotation or not isinstance(receipt.reference_rotation, tuple):
            raise ValueError("Source reference view must be a canonical immutable matrix")
        _text(receipt.provenance)
        if receipt != _seal_segment(receipt):
            raise ValueError("Source segment identity or integrity mismatch")
        reproduced = _advance(origin_surface, expected, receipt.current_load, dt, rotation, receipt.provenance,
                              reference_configuration=receipt.reference_configuration)
        if receipt != reproduced.state.segments[-1]:
            raise ValueError("Source segment differs from mechanical/work replay")
        expected = reproduced.state
    _validate_current(origin_surface, state)
    if state != expected:
        raise ValueError("Saved segmented mechanics differs from source-segment replay")
    return state


def advance_segmented_bending_dynamics(origin_surface, previous, current_load, dt_s, *,
                                       reference_rotation=None, provenance,
                                       reference_elapsed_time_myr=0., reference_surface=None,
                                       reference_roundoff_steps=0):
    """Accept one incremental mechanical solve under its declared fixed source.

    The reference may describe a passive view or an already accepted material
    epoch. Constructing it does not advance the surface again. The source
    must be bound to that exact configuration; trusted assembly and material
    transaction acceptance remain the caller's responsibilities.
    """
    dt = _number(dt_s, "Source segment duration", positive=True)
    rotation = validate_reference_rotation(reference_rotation,
                                           reference_roundoff_steps=reference_roundoff_steps)
    provenance = _text(provenance)
    validate_segmented_bending_dynamics_state(origin_surface, previous)
    return _advance(origin_surface, previous, current_load, dt, rotation, provenance,
        reference_elapsed_time_myr=reference_elapsed_time_myr,
        reference_surface=reference_surface, reference_roundoff_steps=reference_roundoff_steps)


def segmented_bending_dynamics_to_dict(origin_surface, state):
    validate_segmented_bending_dynamics_state(origin_surface, state)
    result = asdict(state)
    result["initial_load"] = basal_source_to_dict(origin_surface, state.initial_load)
    result["initial_mechanics"] = bending_contact_dynamics_to_dict(origin_surface, state.initial_mechanics, state.initial_load)
    result["states"] = [[identity, state_to_dict(local)] for identity, local in state.states]
    result["segments"] = []
    for receipt in state.segments:
        row = asdict(receipt)
        reference = receipt.reference_configuration
        row["reference_configuration"] = bending_reference_to_dict(origin_surface, state.initial_mechanics, reference)
        row["current_load"] = basal_source_to_dict(reference.surface, receipt.current_load)
        result["segments"].append(row)
    result.update(format=FORMAT, model=MODEL, coordinate_convention=COORDINATES)
    return json.loads(_encode(result))


def segmented_bending_dynamics_from_dict(raw, origin_surface):
    expected = {field.name for field in fields(SegmentedBendingState)}|{"format", "model", "coordinate_convention"}
    if (not isinstance(raw, dict) or set(raw) != expected or raw["format"] != FORMAT
            or raw["model"] != MODEL or raw["coordinate_convention"] != COORDINATES):
        raise ValueError("Invalid segmented bending checkpoint schema")
    values = {field.name: raw[field.name] for field in fields(SegmentedBendingState)}
    try:
        values["initial_load"] = basal_source_from_dict(values["initial_load"], origin_surface)
        values["initial_mechanics"] = bending_contact_dynamics_from_dict(values["initial_mechanics"], origin_surface, values["initial_load"])
        for name in ("states", "transverse_displacement_m", "free_anchor_harmonics", "rotation_path_material_radians", "segments"):
            if not isinstance(values[name], list):
                raise ValueError("Segmented checkpoint collections require explicit JSON lists")
        local = []
        for row in values["states"]:
            if not isinstance(row, list) or len(row) != 2 or not isinstance(row[0], str):
                raise ValueError("Invalid segmented point history")
            local.append((row[0], state_from_dict(row[1])))
        values["states"] = tuple(local)
        for name in ("transverse_displacement_m", "rotation_path_material_radians"):
            if any(not isinstance(row, list) for row in values[name]):
                raise ValueError("Segmented coordinate rows require JSON lists")
            values[name] = tuple(tuple(row) for row in values[name])
        harmonic_rows = []
        for row in values["free_anchor_harmonics"]:
            if (not isinstance(row, dict) or set(row) != {field.name for field in fields(TraceAnchorHarmonics)}
                    or any(not isinstance(row[name], list) for name in ("gap", "jump"))):
                raise ValueError("Invalid analytic free anchor schema")
            harmonic_rows.append(TraceAnchorHarmonics(row["trace_id"], tuple(row["gap"]), tuple(row["jump"])))
        values["free_anchor_harmonics"] = tuple(harmonic_rows)
        segments = []
        names = {field.name for field in fields(SegmentedSourceReceipt)}
        for row in values["segments"]:
            if not isinstance(row, dict) or set(row) != names:
                raise ValueError("Invalid source segment receipt schema")
            row = dict(row)
            row["reference_configuration"] = bending_reference_from_dict(
                row["reference_configuration"], origin_surface, values["initial_mechanics"])
            if (not isinstance(row["reference_rotation"], list)
                    or any(not isinstance(item, list) for item in row["reference_rotation"])):
                raise ValueError("Source segment reference rotation requires an explicit JSON matrix")
            row["reference_rotation"] = validate_reference_rotation(row["reference_rotation"],
                reference_roundoff_steps=row["reference_configuration"].reference_roundoff_steps)
            for name in ("rotation_increment_radians", "transverse_displacement_m"):
                if not isinstance(row[name], list) or any(not isinstance(item, list) for item in row[name]):
                    raise ValueError("Source segment coordinates require JSON lists")
                row[name] = tuple(tuple(item) for item in row[name])
            reference = row["reference_configuration"]
            row["current_load"] = basal_source_from_dict(row["current_load"], reference.surface)
            segments.append(SegmentedSourceReceipt(**row))
        values["segments"] = tuple(segments)
        return validate_segmented_bending_dynamics_state(origin_surface, SegmentedBendingState(**values))
    except (TypeError, KeyError, IndexError, OverflowError) as error:
        raise ValueError("Malformed segmented bending checkpoint") from error
