"""Reference bending banks and exact finite material-pair separation.

The reference clamp belongs to its declared material owner along the whole
inward great-circle segment, rather than just at its endpoints. Material cuts
only divide its support intervals. A finite pair audit compares the two images
of the *same* reference point; coincidence of the two boundary sets alone does
not preserve a material pairing or an existing pointwise contact history.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real

import numpy as np

from .geometric_bending import FaultBendingParameters
from .geometric_boundary import surface_digest
from .geometric_fault_traces import (
    PhysicalFaultTrace, _arc_frame, fault_trace_points,
)
from .spherical_polygons import _edge_distance


_EPS = np.finfo(float).eps
_LD = np.longdouble


class UnresolvedFaultMotionSupport(ValueError):
    """The declared reference bank or finite material pairing is unresolved."""


@dataclass(frozen=True, slots=True)
class MaterialBankSupport:
    point_id: str
    trace_id: str
    bank: str
    plate_id: int
    width_m: float
    clamp_position: tuple[float, float, float]
    fragment_intervals: tuple[tuple[str, float, float], ...]


@dataclass(frozen=True, slots=True)
class ReferenceBendingBankSupport:
    surface_sha256: str
    trace_sha256: str
    supports: tuple[MaterialBankSupport, ...]
    roundoff_merged_gap_count: int
    maximum_roundoff_gap_radians: float


def _finite_number(value, name, *, positive=False):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not math.isfinite(float(value)) or (positive and value <= 0.)):
        raise ValueError(f"{name} must be finite, real" + (" and positive" if positive else ""))
    return float(value)


def _rotation_matrix(value, name, *, rotation_roundoff_steps=0):
    raw = np.asarray(value)
    original = np.asarray(value, dtype=object)
    if (raw.shape != (3, 3) or raw.dtype.kind not in "iuf"
            or any(isinstance(item, (bool, np.bool_)) for item in original.flat)):
        raise ValueError(f"{name} must be an explicit finite SO(3) matrix")
    matrix = np.asarray(value, dtype=float)
    tolerance = max(64.*_EPS, 256.*_EPS*rotation_roundoff_steps)
    if (not np.isfinite(matrix).all()
            or np.max(np.abs(matrix.T @ matrix-np.eye(3))) > tolerance
            or abs(float(np.linalg.det(matrix))-1.) > tolerance):
        raise ValueError(f"{name} must be an explicit finite SO(3) matrix")
    return matrix


def _edge_coefficients(point, direction, a, b):
    """Filtered A,B in n.r(theta)=A cos(theta)+B sin(theta).

    The constant term uses the existing difference-form determinant predicate,
    including exact incidence at saved edge endpoints. Bounds on the tangent
    term depend on its arithmetic terms; they have no geographical floor.
    """
    a, b, direction = (np.asarray(item, dtype=_LD) for item in (a, b, direction))
    edge = b-a
    normal = np.cross(a, edge)
    norm = np.sqrt(np.sum(normal*normal))
    if norm == 0.:
        raise ValueError("Reference bank footprint contains a degenerate edge")
    constant = _LD(_edge_distance(np.asarray(point), np.asarray(a), np.asarray(b), None))
    products = normal*direction
    tangent = np.sum(products)/norm
    # Last-bit uncertainties of the saved edge and direction enter through
    # the determinant gradients, just as in the polygon predicate.
    coordinate = 4.*_EPS*(
        np.dot(np.abs(np.cross(b, direction)), np.abs(a))
        + np.dot(np.abs(np.cross(direction, a)), np.abs(b))
        + np.dot(np.abs(normal), np.abs(direction)))
    bound = (16.*_EPS*np.sum(np.abs(products))+coordinate)/norm
    if tangent != 0. and abs(tangent) <= bound:
        tangent = _LD(0.)
    return constant, tangent


def _clip_halfspace(low, high, extent, constant, tangent):
    # On an interval shorter than pi/2 a nonzero sinusoid has at most one root.
    # Endpoint signs therefore determine the complete accepted interval.
    terms = np.asarray((constant*np.cos(_LD(extent)), tangent*np.sin(_LD(extent))))
    end = np.sum(terms)
    if end != 0. and abs(end) <= 16.*_EPS*np.sum(np.abs(terms)):
        end = _LD(0.)
    if constant >= 0. and end >= 0.:
        return low, high
    if constant <= 0. and end <= 0.:
        return None
    root = float(np.arctan2(-constant, tangent) % np.arccos(_LD(-1.)))
    tolerance = 64.*_EPS*extent
    if abs(root) <= tolerance:
        root = 0.
    elif abs(root-extent) <= tolerance:
        root = extent
    if not 0. <= root <= extent:
        raise UnresolvedFaultMotionSupport("Cannot certify a reference bank halfspace root")
    if constant < 0.:
        low = max(low, root)
    else:
        high = min(high, root)
    return (low, high) if high > low else None


def _fragment_interval(point, direction, extent, polygon):
    low, high = 0., extent
    vertices = np.asarray(polygon)
    for a, b in zip(vertices, np.roll(vertices, -1, axis=0)):
        coefficients = _edge_coefficients(point, direction, a, b)
        clipped = _clip_halfspace(low, high, extent, *coefficients)
        if clipped is None:
            return None
        low, high = clipped
    return (low, high) if high > low else None


def _union_covers(intervals, extent, point_id, bank):
    cursor, gap_count, max_gap = 0., 0, 0.
    tolerance = 128.*_EPS*extent
    for _, low, high in sorted(intervals, key=lambda item: (item[1], item[2], item[0])):
        if low > cursor:
            gap = low-cursor
            if gap > tolerance:
                raise UnresolvedFaultMotionSupport(
                    f"Reference {bank} bank is not continuously supported at {point_id}")
            gap_count += 1
            max_gap = max(max_gap, gap)
        cursor = max(cursor, high)
    if extent-cursor > tolerance:
        raise UnresolvedFaultMotionSupport(
            f"Reference {bank} clamp has missing material support at {point_id}")
    if cursor < extent:
        gap_count += 1
        max_gap = max(max_gap, extent-cursor)
    return gap_count, max_gap


def validate_reference_bending_bank_support(surface, trace_set, bending_parameters):
    """Certify both complete inward bank segments at every original trace node.

    The given physical widths must satisfy ``0 < width/radius < pi/2``. Every
    positive interval must belong to its declared bank, across any number of
    material fragments. Foreign-owner intervals are rejected even if both
    endpoints lie in the correct owner. A zero-measure trench touch is allowed.
    """
    supplied = tuple(bending_parameters)
    if any(not isinstance(item, FaultBendingParameters) for item in supplied):
        raise ValueError("Explicit FaultBendingParameters are required for all physical traces")
    parameters = {item.trace_id: item for item in supplied}
    if (len(parameters) != len(supplied)
            or set(parameters) != {trace.trace_id for trace in trace_set.traces}):
        raise ValueError("Bending banks must name every physical trace exactly once")
    points = fault_trace_points(surface, trace_set)
    traces = {trace.trace_id: trace for trace in trace_set.traces}
    radius_m = surface.radius_km*1000.
    supports, gap_count, maximum_gap = [], 0, 0.
    for point in points:
        trace, parameter = traces[point.trace_id], parameters[point.trace_id]
        for bank, owner, material, sign in (
                ("sub", trace.subducting_plate, parameter.sub, -1.),
                ("over", trace.overriding_plate, parameter.over, 1.)):
            width = _finite_number(material.width_m, "Reference bank width", positive=True)
            extent = width/radius_m
            if not 0. < extent < math.pi/2.:
                raise UnresolvedFaultMotionSupport("Reference bank width/radius must lie strictly between zero and pi/2")
            origin = np.asarray(point.position)
            direction = sign*np.asarray(point.horizontal)
            intervals = []
            for fragment in surface.fragments:
                interval = _fragment_interval(origin, direction, extent, fragment.polygon)
                if interval is None:
                    continue
                if fragment.parcel.plate != owner:
                    raise UnresolvedFaultMotionSupport(
                        f"Reference {bank} bank crosses another material owner at {point.point_id}")
                intervals.append((fragment.fragment_id, *interval))
            count, gap = _union_covers(intervals, extent, point.point_id, bank)
            gap_count += count
            maximum_gap = max(maximum_gap, gap)
            clamp = origin*math.cos(extent)+direction*math.sin(extent)
            clamp /= np.sqrt(np.sum(clamp*clamp))
            supports.append(MaterialBankSupport(point.point_id, point.trace_id, bank,
                owner, width, tuple(map(float, clamp)), tuple(sorted(intervals))))
    return ReferenceBendingBankSupport(surface_digest(surface), trace_set.trace_sha256,
        tuple(supports), gap_count, maximum_gap)


def continuous_bank_pair_separation_m(trace, rotation_sub, rotation_over, radius_m, *,
                                      rotation_roundoff_steps=0):
    """Exact full-reference-arc extrema of the two paired SO(3) images.

    The squared chord is ``a+b*cos(2 theta)+c*sin(2 theta)``. All endpoints and
    interior stationary points are evaluated, including extrema between saved
    quadrature nodes. This audit does not declare material transfer or remap an
    existing local history when the pairing changes.

    ``rotation_roundoff_steps`` is the explicit number of chronological matrix
    products, after caller validation of their replay. Its arithmetic budget
    permits accumulated last-bit drift without projecting or renormalizing a
    saved map. The default retains the strict standalone SO(3) check.
    """
    if not isinstance(trace, PhysicalFaultTrace):
        raise ValueError("Finite pairing requires an explicit PhysicalFaultTrace")
    if (isinstance(rotation_roundoff_steps, (bool, np.bool_))
            or not isinstance(rotation_roundoff_steps, Integral)
            or rotation_roundoff_steps < 0):
        raise ValueError("Rotation roundoff steps must be a nonnegative integer")
    steps = int(rotation_roundoff_steps)
    sub, over = (_rotation_matrix(value, name, rotation_roundoff_steps=steps) for value, name in
                 ((rotation_sub, "Subducting rotation"), (rotation_over, "Overriding rotation")))
    radius = _finite_number(radius_m, "Sphere radius", positive=True)
    start, _, tangent, extent = _arc_frame(trace.start, trace.end)
    difference = sub.astype(_LD)-over.astype(_LD)
    u, v = difference@start.astype(_LD), difference@tangent.astype(_LD)
    b = (np.sum(u*u)-np.sum(v*v))/_LD(2.)
    c = np.sum(u*v)
    angles = [0., extent]
    if b != 0. or c != 0.:
        first = np.arctan2(c, b)/_LD(2.)
        for index in range(-2, 5):
            candidate = float(first+_LD(index)*_LD(math.pi)/_LD(2.))
            if 0. < candidate < extent:
                angles.append(candidate)
    lengths = []
    for angle in angles:
        vector = u*np.cos(_LD(angle))+v*np.sin(_LD(angle))
        length = float(_LD(radius)*np.sqrt(np.sum(vector*vector)))
        if not math.isfinite(length):
            raise ValueError("Finite material-pair separation overflowed")
        lengths.append(length)
    return min(lengths), max(lengths)
