"""Objective incremental ports on persistent original material-bank pairs.

The two tips follow their own material owners. Contact axes and point labels
follow the overriding owner. The resulting finite-column connection includes
the overriding-frame couple; replacing it with two forces at the original
common point would violate objectivity. Operators are frozen for one mechanical
increment. Saved spring/plastic histories are never rebased to a new pose.

The original beam domain, quadrature measures and material parameters remain
unchanged. A public reference therefore requires independently checked live
and buried material support, including live original clamps. This module does
not accept a material transaction or prescribe any velocity or force.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import math
from numbers import Integral, Real

import numpy as np

from .geometric_bending_contact_dynamics import _transverse_coordinates
from .geometric_bending_reference import (
    _immutable, _roundoff_steps, current_bending_reference, validate_reference_rotation,
)
from .geometric_boundary import surface_digest
from .geometric_fault_traces import _arc_frame
from .geometric_surface import GeometricSurfaceState, audit_partition


MODEL = "persistent_material_pair_objective_first_order_column_ports"
FIELD_BASIS = ("constant", "cos_theta", "sin_theta", "cos_2theta", "sin_2theta")


def _owner(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 0:
        raise ValueError("Material-pair poses require nonnegative integer owner IDs")
    return int(value)


def canonical_material_pair_poses(seed, owner_rotations, *, roundoff_steps=0):
    """Copy proper SO(3) poses for the compact actual owner set exactly once."""
    steps = _roundoff_steps(roundoff_steps)
    ids = seed.plate_ids
    if (not isinstance(ids, tuple) or tuple(sorted(set(ids))) != ids
            or any(_owner(owner) != owner for owner in ids)):
        raise ValueError("Material-pair seed must retain canonical actual owner IDs")
    if isinstance(owner_rotations, Mapping):
        records = tuple(sorted((_owner(owner), matrix) for owner, matrix in owner_rotations.items()))
    elif isinstance(owner_rotations, (tuple, list)):
        records = []
        for row in owner_rotations:
            if not isinstance(row, (tuple, list)) or len(row) != 2:
                raise ValueError("Material-pair poses require explicit (owner, SO3) records")
            records.append((_owner(row[0]), row[1]))
        records = tuple(records)
    else:
        raise ValueError("Material-pair poses require compact explicit owner records")
    if tuple(owner for owner, _ in records) != ids:
        raise ValueError("Material-pair poses must cover every actual owner exactly once in canonical order")
    if any(matrix is None for _, matrix in records):
        raise ValueError("Material-pair owner poses require explicit proper SO3 matrices")
    return tuple((owner, validate_reference_rotation(matrix,
        reference_roundoff_steps=steps)) for owner, matrix in records)


def _rotations(seed, values):
    raw = np.asarray(values)
    original = np.asarray(values, dtype=object)
    if (raw.shape != (len(seed.plate_ids), 3) or raw.dtype.kind not in "iuf"
            or any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                   for value in original.flat)):
        raise ValueError("Material-pair increments require finite compact owner-aligned triples")
    result = np.asarray(values, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError("Material-pair increments require finite compact owner-aligned triples")
    return result


def evaluate_material_pair_field(coefficients, angle):
    """Evaluate the declared five-term analytic field, without interpolation."""
    raw = np.asarray(coefficients)
    original = np.asarray(coefficients, dtype=object)
    if (raw.shape != (5,) or raw.dtype.kind not in "iuf"
            or any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                   for value in original.flat)
            or not np.isfinite(raw).all()
            or isinstance(angle, (bool, np.bool_)) or not isinstance(angle, Real)
            or not math.isfinite(float(angle))):
        raise ValueError("Material-pair fields require five finite real coefficients and a finite angle")
    angle = float(angle)
    basis = (1., math.cos(angle), math.sin(angle), math.cos(2.*angle), math.sin(2.*angle))
    return math.fsum(float(value)*term for value, term in zip(coefficients, basis))


def _linear_times_frame(first, second, axis, frame_first, frame_second):
    """Coefficients of (first*cos+second*sin).(axis+frame_first*cos+frame_second*sin)."""
    return np.asarray((
        .5*(float(first@frame_first)+float(second@frame_second)),
        float(first@axis), float(second@axis),
        .5*(float(first@frame_first)-float(second@frame_second)),
        .5*(float(first@frame_second)+float(second@frame_first))), dtype=float)


def _angular_coefficients(first, second, axis, frame_first, frame_second, radius):
    return radius*np.asarray((
        .5*(np.cross(first, frame_first)+np.cross(second, frame_second)),
        np.cross(first, axis), np.cross(second, axis),
        .5*(np.cross(first, frame_first)-np.cross(second, frame_second)),
        .5*(np.cross(first, frame_second)+np.cross(second, frame_first))))


def _trace_frame(trace, poses):
    start, horizontal, tangent, _ = _arc_frame(trace.start, trace.end)
    sub, over = poses[trace.subducting_plate], poses[trace.overriding_plate]
    return sub@start, sub@tangent, over@start, over@tangent, over@horizontal


def material_pair_increment_fields(seed, owner_rotations, radial_before,
                                   rotation_increments, radial_after, *, roundoff_steps=0):
    """Exact analytic field of the frozen objective operator times this increment.

    Coordinates are owner-aligned world rotation triples and physical radial
    displacements in the existing canonical ``(trace_id, sub, over)`` format.
    These coefficients describe the incremental port law, not exact finite
    pose strain. Terms of order two along the reference arc are essential.
    """
    records = canonical_material_pair_poses(seed, owner_rotations, roundoff_steps=roundoff_steps)
    poses = {owner: np.asarray(matrix) for owner, matrix in records}
    increments = _rotations(seed, rotation_increments)
    positions = {owner: index for index, owner in enumerate(seed.plate_ids)}
    old, new = _transverse_coordinates(seed.fault_traces, radial_before), _transverse_coordinates(seed.fault_traces, radial_after)
    radius = seed.fault_traces.radius_km*1000.
    result = []
    for trace, before, after in zip(seed.fault_traces.traces, old, new):
        sub_first, sub_second, over_first, over_second, horizontal = _trace_frame(trace, poses)
        relative = increments[positions[trace.overriding_plate]]-increments[positions[trace.subducting_plate]]
        change = after-before
        sine, cosine = math.sin(trace.dip_radians), math.cos(trace.dip_radians)
        fields = []
        for alpha, beta in ((sine, cosine), (cosine, -sine)):
            axis, frame_first, frame_second = alpha*horizontal, beta*over_first, beta*over_second
            angular = _angular_coefficients(sub_first, sub_second, axis, frame_first,
                                            frame_second, radius+float(before[0]))@relative
            sub_radial = _linear_times_frame(sub_first, sub_second, axis, frame_first, frame_second)
            over_radial = _linear_times_frame(over_first, over_second, axis, frame_first, frame_second)
            values = angular-float(change[0])*sub_radial+float(change[1])*over_radial
            if not np.isfinite(values).all():
                raise ValueError("Material-pair analytic increment overflowed")
            fields.append(tuple(map(float, values)))
        result.append((trace.trace_id, *fields))
    return tuple(result)


def _pose_fields(seed, poses, radial):
    radius = seed.fault_traces.radius_km*1000.
    result = []
    for trace, displacement in zip(seed.fault_traces.traces, radial):
        sub_first, sub_second, over_first, over_second, horizontal = _trace_frame(trace, poses)
        # Difference-form evaluation avoids subtracting two O(radius) vectors.
        with np.errstate(over="ignore", invalid="ignore"):
            first = radius*(over_first-sub_first)+float(displacement[1])*over_first-float(displacement[0])*sub_first
            second = radius*(over_second-sub_second)+float(displacement[1])*over_second-float(displacement[0])*sub_second
            if not np.isfinite(first).all() or not np.isfinite(second).all():
                raise ValueError("Material-pair finite-pose anchor field overflowed")
            sine, cosine = math.sin(trace.dip_radians), math.cos(trace.dip_radians)
            fields = tuple(tuple(map(float, _linear_times_frame(first, second,
                alpha*horizontal, beta*over_first, beta*over_second)))
                for alpha, beta in ((sine, cosine), (cosine, -sine)))
        if not np.isfinite(fields).all():
            raise ValueError("Material-pair finite-pose anchor field overflowed")
        result.append((trace.trace_id, *fields))
    return tuple(result)


def material_pair_exact_anchor_fields(seed, owner_rotations, radial_current, *, roundoff_steps=0):
    """Finite-pose anchor CHANGE from the original identity pose and seed w.

    Add this change to the saved prepared anchor field. Its difference from
    the integrated first-order port field is a diagnostic; assigning it to
    old spring anchors would insert a mechanical rebase without paid work.
    """
    records = canonical_material_pair_poses(seed, owner_rotations, roundoff_steps=roundoff_steps)
    poses = {owner: np.asarray(matrix) for owner, matrix in records}
    current = _transverse_coordinates(seed.fault_traces, radial_current)
    baseline = _transverse_coordinates(seed.fault_traces, seed.transverse_displacement_m)
    actual = _pose_fields(seed, poses, current)
    original = _pose_fields(seed, {owner: np.eye(3) for owner in seed.plate_ids}, baseline)
    return tuple((before[0], *(tuple(math.fsum((now, -old)) for now, old in zip(a, b))
        for a, b in zip(after[1:], before[1:]))) for after, before in zip(actual, original))


@dataclass(frozen=True, slots=True)
class MaterialPairPort:
    point_id: str
    trace_id: str
    subducting_plate: int
    overriding_plate: int
    sub_radial: tuple[float, float, float]
    over_radial: tuple[float, float, float]
    sub_position_m: tuple[float, float, float]
    over_position_m: tuple[float, float, float]
    normal: tuple[float, float, float]
    down_dip: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class MaterialPairReference:
    surface: GeometricSurfaceState
    points: tuple
    materials: tuple
    rows: tuple
    areas: tuple
    frames: tuple
    operators: tuple
    stiffness: tuple
    reference_points: tuple
    offsets: tuple
    plate_ids: tuple[int, ...]
    owner_rotations: tuple
    transverse_before: tuple
    pair_ports: tuple[MaterialPairPort, ...]
    reference_surface_sha256: str
    reference_trace_sha256: str
    reference_bending_sha256: str
    current_surface_sha256: str
    support_receipt: object
    roundoff_steps: int
    physical_dt_s: float = 0.
    model: str = MODEL


def _material_pair_network(origin, seed, owner_rotations, previous_transverse, *, roundoff_steps=0):
    """Pure diagnostic network; public adoption additionally proves support."""
    original = current_bending_reference(origin, seed)
    records = canonical_material_pair_poses(seed, owner_rotations, roundoff_steps=roundoff_steps)
    poses = {owner: np.asarray(matrix) for owner, matrix in records}
    radial = _transverse_coordinates(seed.fault_traces, previous_transverse)
    traces = {trace.trace_id: index for index, trace in enumerate(seed.fault_traces.traces)}
    owners = {owner: index for index, owner in enumerate(seed.plate_ids)}
    radius = origin.radius_km*1000.
    count = 3*len(seed.plate_ids)
    dimension = len(original.stiffness)
    operators = np.zeros((len(original.points), 2, dimension))
    rows = np.zeros((len(original.points), dimension))
    moved, ports = [], []
    for index, (point, material) in enumerate(zip(original.points, original.materials)):
        sub_pose, over_pose = poses[material.subducting_plate], poses[material.overriding_plate]
        sub_radial, over_radial = sub_pose@point.position, over_pose@point.position
        sub_w, over_w = radial[traces[point.trace_id]]
        sub_position, over_position = (radius+sub_w)*sub_radial, (radius+over_w)*over_radial
        normal, down_dip = over_pose@point.normal, over_pose@point.down_dip
        horizontal = over_pose@point.horizontal
        sub_index, over_index = owners[material.subducting_plate], owners[material.overriding_plate]
        radial_start = count+2*traces[point.trace_id]
        for row, axis in zip(operators[index], (normal, down_dip)):
            lever = np.cross(sub_position, axis)
            row[3*sub_index:3*sub_index+3] = -lever
            row[3*over_index:3*over_index+3] = lever
            row[radial_start:radial_start+2] = radius*np.asarray((-float(sub_radial@axis), float(over_radial@axis)))
        lever = np.cross(sub_position, horizontal)
        rows[index, 3*sub_index:3*sub_index+3], rows[index, 3*over_index:3*over_index+3] = -lever, lever
        rows[index, radial_start:radial_start+2] = radius*np.asarray((-float(sub_radial@horizontal), float(over_radial@horizontal)))
        moved.append(replace(point, position=tuple(map(float, over_radial)),
            horizontal=tuple(map(float, horizontal)), normal=tuple(map(float, normal)),
            down_dip=tuple(map(float, down_dip))))
        ports.append(MaterialPairPort(point.point_id, point.trace_id,
            material.subducting_plate, material.overriding_plate,
            tuple(map(float, sub_radial)), tuple(map(float, over_radial)),
            tuple(map(float, sub_position)), tuple(map(float, over_position)),
            tuple(map(float, normal)), tuple(map(float, down_dip))))
    if not np.isfinite(operators).all() or not np.isfinite(rows).all():
        raise ValueError("Material-pair objective operator overflowed")
    return original, records, tuple(moved), tuple(ports), _immutable(rows), _immutable(operators)


def current_material_pair_reference(origin, seed, current_surface, owner_rotations,
                                    previous_transverse, *, inventory, buried_registry,
                                    support_receipt=None, roundoff_steps=0):
    """Build the current objective port view after fresh material-support proof.

    Construction advances no clock, local return map, material budget or work
    ledger. No changed surface is accepted solely on a caller supplied hash.
    """
    from .geometric_buried_bank_support import validate_buried_bank_support
    steps = _roundoff_steps(roundoff_steps)
    if not isinstance(current_surface, GeometricSurfaceState) or current_surface.phase != "partition":
        raise ValueError("Material-pair reference requires a resolved current surface")
    audit_partition(current_surface)
    original, records, points, ports, rows, operators = _material_pair_network(origin, seed,
        owner_rotations, previous_transverse, roundoff_steps=steps)
    proof = validate_buried_bank_support(origin, seed, current_surface, inventory, records,
                                        buried_registry, roundoff_steps=steps)
    if support_receipt is not None and support_receipt != proof:
        raise ValueError("Material-pair support receipt differs from fresh material proof")
    return MaterialPairReference(current_surface, points, original.materials, rows,
        original.areas, original.frames, operators, original.stiffness,
        original.reference_points, original.offsets, original.plate_ids, records,
        tuple(previous_transverse), ports, original.reference_surface_sha256,
        original.reference_trace_sha256, original.reference_bending_sha256,
        surface_digest(current_surface), proof, steps)


def material_pair_point_reactions(port, area_m2, traction_pa):
    """Physical force pair and carrier couple reproducing objective virtual work."""
    if not isinstance(port, MaterialPairPort):
        raise ValueError("Physical pair reactions require an explicit material pair port")
    if (isinstance(area_m2, (bool, np.bool_)) or not isinstance(area_m2, Real)
            or not math.isfinite(float(area_m2)) or area_m2 <= 0.):
        raise ValueError("Material-pair reaction area must be finite and positive")
    raw, original = np.asarray(traction_pa), np.asarray(traction_pa, dtype=object)
    if (raw.shape != (2,) or raw.dtype.kind not in "iuf" or not np.isfinite(raw).all()
            or any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                   for value in original.flat)):
        raise ValueError("Material-pair reactions require two finite real traction components")
    normal, down = np.asarray(port.normal), np.asarray(port.down_dip)
    force = float(area_m2)*(float(traction_pa[0])*normal+float(traction_pa[1])*down)
    sub, over = np.asarray(port.sub_position_m), np.asarray(port.over_position_m)
    couple = np.cross(over-sub, force)
    sub_torque = np.cross(sub, force)
    over_force_torque = np.cross(over, -force)
    over_torque = -sub_torque
    radial = (float(np.asarray(port.sub_radial)@force), -float(np.asarray(port.over_radial)@force))
    values = (force, -force, sub_torque, over_torque, over_force_torque, couple)
    if any(not np.isfinite(value).all() for value in values) or not np.isfinite(radial).all():
        raise ValueError("Material-pair physical reaction overflowed")
    return dict(force_on_sub_n=tuple(map(float, force)), force_on_over_n=tuple(map(float, -force)),
        torque_on_sub_nm=tuple(map(float, sub_torque)), torque_on_over_nm=tuple(map(float, over_torque)),
        force_torque_on_over_nm=tuple(map(float, over_force_torque)),
        carrier_couple_on_over_nm=tuple(map(float, couple)), radial_force_n=radial)
