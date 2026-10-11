"""Continuous underthrust admission on inherited native receiver arcs.

The original material-coordinate return map is replayed without changing its
mechanical nodes.  Spatial location and complete support coverage are proved
with native rays and planes; projected Cartesian endpoints never grant
permission.  Material-angle queries are outward enclosures of an already
proved positive arc, rather than a replacement for its geometry or mass.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from fractions import Fraction
from functools import cmp_to_key
import hashlib
import json
import math

import numpy as np

from .geometric_boundary import surface_digest
from .geometric_intrinsic_geometry import RotationIncrement
from .geometric_native_polygons import (
    PRECISIONS, NativeArc, NativePlane, NativePolygon, NativeRay,
    UnresolvedNativeGeometry, _arc_length, _arc_order, _atan2_positive,
    _cross, _dot, _history, _incident, _plane_values, _primitive,
    _ray_same, _ray_values, _same_plane, _side, _world, native_arc_to_dict,
    native_polygon_to_dict, native_plane_from_dict, native_ray_from_dict,
    native_intersection, validate_native_polygon,
)
from .geometric_native_pose import FractionInterval as I, interval_sqrt
from .geometric_native_surface import (
    NativeFragment, NativeSurfaceState, NativeTransportResult, extract_native_contacts,
    native_surface_to_dict,
)
from .geometric_polarity import PolarityEvidence, PolarityResolution
from .geometric_rotation_vectors import compact_rotation_vectors
from .geometric_spatial_underthrust import (
    MODEL, SpatialUnderthrustStep, _assess_interval, _real,
    _validate_memory, advance_spatial_underthrust_memory,
)
from .geometric_surface import GeometricSurfaceState
from .thermal import SECONDS_PER_MYR


FORMAT = "geometric-native-continuous-spatial-underthrust-1"
_EPS = np.finfo(float).eps


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("utf-8")).hexdigest()


def _validate_arc(arc):
    if (type(arc) is not NativeArc or type(arc.start) is not NativeRay
            or type(arc.end) is not NativeRay or type(arc.plane) is not NativePlane):
        raise ValueError("Native continuous support requires explicit inherited rays and plane")
    if ((not _incident(arc.start, arc.plane) and _side(arc.start, arc.plane) != 0)
            or (not _incident(arc.end, arc.plane) and _side(arc.end, arc.plane) != 0)):
        raise ValueError("Native continuous arc endpoints lost their inherited plane")
    if _arc_order(arc.start, arc.end, arc.plane) <= 0:
        raise ValueError("Native continuous support must be an oriented positive minor arc")
    length = _arc_length(arc.start, arc.end)
    length_value = _real(arc.length_radians, "Native continuous arc length", positive=True)
    if (not math.isfinite(length_value) or length_value <= 0.
            or arc.length_radians >= math.pi or arc.length_radians != length):
        raise ValueError("Native continuous arc length differs from its certified geometry")
    return arc


def _arc(start, end, plane):
    return _validate_arc(NativeArc(start, end, plane, _arc_length(start, end)))


def _reversed(arc):
    return NativeArc(arc.end, arc.start, arc.plane.reversed(), arc.length_radians)


def _rotated(arc, history):
    history = _history(history)
    # A common proper pose preserves the true length; the disposable length
    # estimate may differ by one final rounding bit, so recompute that value.
    return _arc(arc.start.rotated(history), arc.end.rotated(history),
                arc.plane.rotated(history))


def native_intersect_support_arcs(first, second):
    """Exact positive common minor arc, oriented like ``first``.

    Plane coincidence, membership, endpoint equality and positive extent all
    need native proofs.  There is no proximity or minimum-length rule.
    """
    first, second = _validate_arc(first), _validate_arc(second)
    if not _same_plane(first.plane, second.plane):
        return None
    if first.plane.normal == tuple(-x for x in second.plane.normal):
        second = _reversed(second)
    points = []
    for point in (first.start, first.end, second.start, second.end):
        if (all(_arc_order(start, point, first.plane) >= 0
                and _arc_order(point, end, first.plane) >= 0
                for start, end in ((first.start, first.end), (second.start, second.end)))
                and not any(_ray_same(point, own) for own in points)):
            points.append(point)
    if len(points) < 2:
        return None
    if len(points) != 2:
        raise UnresolvedNativeGeometry("Native continuous support has unproved endpoint aliases")
    start, end = points
    if _arc_order(start, end, first.plane) < 0:
        start, end = end, start
    return _arc(start, end, first.plane)


def native_support_covers_arc(actual, supports):
    """Prove the entire positive actual arc is covered without angular epsilon."""
    actual = _validate_arc(actual)
    clipped = tuple(shared for support in supports
        if (shared := native_intersect_support_arcs(actual, support)) is not None)
    if not clipped:
        return False
    ordered = sorted(clipped, key=cmp_to_key(lambda left, right:
        _arc_order(right.start, left.start, actual.plane)))
    cursor = actual.start
    for support in ordered:
        if _arc_order(cursor, support.start, actual.plane) > 0:
            return False
        if _arc_order(cursor, support.end, actual.plane) > 0:
            cursor = support.end
    return _ray_same(cursor, actual.end)


def _owner_histories(seed, supplied):
    if not isinstance(supplied, tuple) or any(not isinstance(row, tuple)
            or len(row) != 2 for row in supplied):
        raise ValueError("Native admission requires immutable original owner histories")
    if tuple(row[0] for row in supplied) != tuple(seed.plate_ids):
        raise ValueError("Native admission must retain every original mechanical owner")
    checked = []
    for owner, history in supplied:
        if isinstance(owner, bool) or not isinstance(owner, (int, np.integer)):
            raise ValueError("Native physical owner must be an explicit integer")
        if not isinstance(history, tuple):
            raise ValueError("Native physical poses require ordered immutable increments")
        checked.append((int(owner), _history(history)))
    lengths = {len(history) for _, history in checked}
    if len(lengths) != 1:
        raise ValueError("Native original owner clocks have different interval counts")
    if checked:
        clocks = tuple(step.dt_myr for step in checked[0][1])
        if any(tuple(step.dt_myr for step in history) != clocks for _, history in checked):
            raise ValueError("Native original owner histories disagree on interval clocks")
    return tuple(checked)


def _trace_arc(trace, history):
    first, last = NativeRay("seed", _primitive(trace.start)), NativeRay("seed", _primitive(trace.end))
    plane = NativePlane(_cross(first.reference, last.reference), label=trace.trace_id)
    return _rotated(_arc(first, last, plane), history)


def _fraction_float(value, direction):
    floating = float(value)
    if not math.isfinite(floating):
        raise UnresolvedNativeGeometry("Native material angle cannot be represented")
    exact = Fraction.from_float(floating)
    if (direction < 0 and exact > value) or (direction > 0 and exact < value):
        floating = math.nextafter(floating, -math.inf if direction < 0 else math.inf)
    return floating


def _angle_bounds(first, last):
    if _ray_same(first, last):
        return I(0)
    for precision in PRECISIONS:
        _, left = _ray_values(first, precision)
        _, right = _ray_values(last, precision)
        cross = _cross(left, right)
        try:
            angle = _atan2_positive(interval_sqrt(_dot(cross, cross), precision+8),
                                   _dot(left, right), precision+8)
        except (ValueError, UnresolvedNativeGeometry):
            continue
        if angle.lo > 0 and angle.width < angle.lo/Fraction(10**13):
            return angle
    raise UnresolvedNativeGeometry("Native material-angle query is not certified")


def _material_support_interval(carrier, actual, extent):
    # Positive containment is exact and established before field conversion.
    contained = native_intersect_support_arcs(carrier, actual)
    if contained is None or not native_support_covers_arc(actual, (carrier,)):
        raise ValueError("Actual native support exceeds its original physical trace")
    arithmetic_bound = 1024.*_EPS*extent
    native_extent = _angle_bounds(carrier.start, carrier.end)
    declared_extent = Fraction.from_float(extent)
    if max(abs(native_extent.lo-declared_extent), abs(native_extent.hi-declared_extent)) \
            > Fraction.from_float(arithmetic_bound):
        raise ValueError("Native carrier measure disagrees with retained material extent")
    values = []
    for point in (actual.start, actual.end):
        if _ray_same(point, carrier.start):
            values.append(I(0))
        elif _ray_same(point, carrier.end):
            # The original mechanical coordinate endpoint is defined by the
            # retained binary64 material extent, rather than a new arc fit.
            values.append(I(Fraction.from_float(extent)))
        else:
            values.append(_angle_bounds(carrier.start, point))
    low_exact = min(value.lo for value in values)
    high_exact = max(value.hi for value in values)
    low = max(0., _fraction_float(low_exact, -1))
    high = min(extent, _fraction_float(high_exact, 1))
    if max(low_exact, high_exact) > Fraction.from_float(extent+arithmetic_bound):
        raise ValueError("Native angle disagrees with retained material coordinate")
    # The mechanical field uses its originally declared binary64 endpoint.
    # For example, true pi/2 is slightly greater than that stored number.
    # Thus clamping the field query encloses the declared computational
    # coordinate within the checked frame-arithmetic budget above; it makes
    # no assertion that the stored float exactly equals the geometric angle.
    # A positive inherited arc can be shorter than an absolute-angle ULP.
    # Only its constitutive query widens; the exact native geometry is intact.
    if low >= high:
        center = min(extent, max(0., .5*(low+high)))
        low = max(0., math.nextafter(center, -math.inf))
        high = min(extent, math.nextafter(center, math.inf))
    if not low < high:
        raise UnresolvedNativeGeometry("Positive native arc has no representable field enclosure")
    return low, high


def _normalized(vector, precision):
    norm = interval_sqrt(_dot(vector, vector), precision+8)
    if norm.lo <= 0:
        raise UnresolvedNativeGeometry("Native mechanical direction is not certified nonzero")
    return tuple(value/norm for value in vector)


def _native_velocity_status(carrier, actual_before, trace, omega, radius, *, endpoint_audit=None):
    """Prove the full minor-arc sign from exact endpoint velocity enclosures.

    A minor-arc point is a positive sine-weighted sum of its normalized end
    rays.  Linear normal velocity therefore has a nonpositive sign throughout
    the arc when both endpoints do, and is negative inside when either is.
    No Cartesian midpoint, discarded angular component or sign tolerance is
    used to accept the direction.
    """
    difference = tuple(I(Fraction.from_float(float(omega[trace.overriding_plate][i]))
                        -Fraction.from_float(float(omega[trace.subducting_plate][i])))
                       for i in range(3))
    for precision in PRECISIONS:
        try:
            _, plane = _plane_values(carrier.plane, precision)
            horizontal = _normalized(plane, precision)
            vector = tuple(Fraction.from_float(radius)*value
                           for value in _cross(horizontal, difference))
            plane_norm = interval_sqrt(_dot(plane, plane), precision+8)
            raw_velocity = _cross(plane, difference)
            values = []
            for point in (actual_before.start, actual_before.end):
                _, point_values = _ray_values(point, precision)
                point_norm = interval_sqrt(_dot(point_values, point_values), precision+8)
                if plane_norm.lo <= 0 or point_norm.lo <= 0:
                    raise UnresolvedNativeGeometry("Native velocity normalization is not certified")
                # Form the homogeneous scalar triple product first.  Exact
                # dyadic endpoint cancellations then remain exact; dividing
                # components separately would erase their correlation.
                values.append(Fraction.from_float(radius)*_dot(raw_velocity, point_values)
                              /(plane_norm*point_norm))
            magnitude = interval_sqrt(_dot(vector, vector), precision+8)
        except (ValueError, UnresolvedNativeGeometry):
            continue
        if endpoint_audit is not None:
            endpoint_audit.clear()
            endpoint_audit.update(precision_bits=precision,
                endpoints=[dict(endpoint=name,
                    lower_km_per_myr=_fraction_float(value.lo, -1),
                    upper_km_per_myr=_fraction_float(value.hi, 1))
                    for name, value in zip(("start", "end"), values)])
        if all(value.hi <= 0 for value in values) and any(value.hi < 0 for value in values):
            # For negative endpoint values the sine coefficients sum to at
            # least one, so max(endpoint upper) also bounds the entire arc.
            length = _angle_bounds(actual_before.start, actual_before.end)
            lower = max(-magnitude.hi,
                min(value.lo for value in values)-magnitude.hi*length.hi)
            return "spatial_underthrust_admissible", _fraction_float(lower, -1), \
                _fraction_float(max(value.hi for value in values), 1), \
                _fraction_float(max(value.width for value in values), 1)
        if all(value.lo >= 0 for value in values):
            return "nonconvergent", _fraction_float(min(value.lo for value in values), -1), \
                _fraction_float(magnitude.hi, 1), _fraction_float(max(value.width for value in values), 1)
        if any(value.lo > 0 for value in values) and any(value.hi < 0 for value in values):
            return "mixed_convergence", _fraction_float(-magnitude.hi, -1), \
                _fraction_float(magnitude.hi, 1), _fraction_float(max(value.width for value in values), 1)
    return "unresolved_native_velocity_sign", None, None, None


def _polygon_arcs(poly):
    poly = _world(validate_native_polygon(poly))
    return tuple(_arc(start, end, plane) for start, end, plane in zip(
        poly.vertices, poly.vertices[1:]+poly.vertices[:1], poly.supports))


def _history_directions(contacts, history):
    supplied = tuple(history)
    canonical = {contact.contact_id: contact for contact in contacts}
    result, identities = {}, set()
    for item in supplied:
        if type(item) is not PolarityEvidence or item.evidence_id in identities:
            raise ValueError("Native archived polarity requires unique explicit evidence")
        identities.add(item.evidence_id)
        contact = canonical.get(item.contact_id)
        if contact is None or {contact.plate_a, contact.plate_b} != {
                item.subducting_plate, item.overriding_plate}:
            raise ValueError("Archived polarity has no matching current native contact")
        result.setdefault(item.contact_id, set()).add(
            (item.subducting_plate, item.overriding_plate))
    return result


def make_native_spatial_underthrust_resolver(surface, omega_rad_per_myr, dt_myr, *,
        origin_surface, initial_mechanics, owner_histories, step, history=()):
    """Compile native receiver-local permission from the original solved field.

    ``owner_histories`` carries each original physical bank by its complete
    analytic pose history.  Original material identity is required on both
    banks.  A caller cannot use a guessed new pair or scalar material contrast
    when this resolver returns an explicit unresolved decision.
    """
    if type(surface) is not NativeSurfaceState or surface.phase != "partition":
        raise ValueError("Native continuous admission needs a resolved native surface")
    # Reconstruct rather than trust a forged opaque object or caller checksum.
    digest = _hash(native_surface_to_dict(surface))
    if not isinstance(origin_surface, GeometricSurfaceState) or origin_surface.phase != "partition":
        raise ValueError("Native continuous admission requires original declared material banks")
    seed = initial_mechanics
    if seed.fault_traces.surface_sha256 != surface_digest(origin_surface):
        raise ValueError("Original physical weak planes belong to a different surface")
    if (surface.radius_km != origin_surface.radius_km
            or seed.fault_traces.radius_km != origin_surface.radius_km):
        raise ValueError("Native physical banks cannot change their declared planet radius")
    if type(step) is not SpatialUnderthrustStep:
        raise ValueError("Native admission requires an original continuous constitutive step")
    dt = _real(dt_myr, "Native geometric interval", positive=True)
    if not math.isclose(dt*SECONDS_PER_MYR, step.dt_s, rel_tol=8.*_EPS, abs_tol=0.):
        raise ValueError("Native constitutive and geometric clocks disagree")
    checked_histories = _owner_histories(seed, owner_histories)
    poses = dict(checked_histories)
    count = len(checked_histories[0][1]) if checked_histories else 0
    elapsed = math.fsum(increment.dt_myr*SECONDS_PER_MYR
        for increment in checked_histories[0][1]) if checked_histories else 0.
    if (count != step.previous_memory.stage_count or not math.isclose(elapsed,
            step.previous_memory.elapsed_time_s, rel_tol=8.*_EPS, abs_tol=0.)):
        raise ValueError("Native pose histories and original constitutive memory clocks disagree")
    expected_time = origin_surface.time_myr
    for increment in checked_histories[0][1] if checked_histories else ():
        expected_time += increment.dt_myr
    if expected_time != surface.time_myr:
        raise ValueError("Native surface time differs from the retained original pose histories")
    if not isinstance(omega_rad_per_myr, Mapping):
        raise ValueError("Native continuous admission needs an explicit owner velocity mapping")
    omega = compact_rotation_vectors(omega_rad_per_myr,
        tuple(fragment.parcel.plate for fragment in surface.fragments))
    _validate_memory(seed, step.previous_memory)
    _validate_memory(seed, step.memory)
    reproduced = advance_spatial_underthrust_memory(seed, step.previous_memory,
        step.anchor_fields, step.dt_s, previous_point_states=step.previous_point_states,
        current_point_states=step.current_point_states)
    if reproduced != step:
        raise ValueError("Native spatial certificate differs from authenticated node/field replay")
    if tuple(item.trace_id for item in step.certificates) != tuple(
            trace.trace_id for trace in seed.fault_traces.traces):
        raise ValueError("Native spatial certificate lost original physical traces")
    contacts = extract_native_contacts(surface.fragments, surface.radius_km)
    prior_directions = _history_directions(contacts, history)
    certificates = {item.trace_id: item for item in step.certificates}
    original_materials = {(fragment.parcel.material_id, fragment.parcel.plate)
                          for fragment in origin_surface.fragments}
    carriers = tuple((trace, _trace_arc(trace, poses[trace.overriding_plate]))
                     for trace in seed.fault_traces.traces)
    compiled = []
    for contact in contacts:
        for sub, over in ((contact.plate_a, contact.plate_b),
                          (contact.plate_b, contact.plate_a)):
            if over not in poses or sub not in poses:
                continue
            receiver_id = contact.fragment_a if contact.plate_a == over else contact.fragment_b
            material = contact.material_a if contact.plate_a == over else contact.material_b
            receiver_arc = contact.arc if contact.plate_a == over else _reversed(contact.arc)
            increment = (RotationIncrement(tuple(map(float, omega[over])), dt),)
            compiled.append((contact, sub, over, receiver_id, material,
                             _rotated(receiver_arc, increment), increment))
    overlap_rows = []
    audit = dict(format=FORMAT, model=MODEL, time_myr=surface.time_myr,
        dt_myr=dt, surface_sha256=digest,
        origin_surface_sha256=surface_digest(origin_surface),
        node_manifest_sha256=step.node_manifest_sha256,
        actual_overlap_assessments=overlap_rows,
        geometry_policy="inherited native planes, exact minor-arc membership and complete coverage",
        velocity_policy="certified full-vector endpoint signs on a positive native minor arc",
        overlap_policy="every actual receiver arc must admit the same original solved direction",
        material_policy="original declared material pairs; plastic slip never creates volume")

    def resolve(left, right, overlap):
        if type(left) is not NativeFragment or type(right) is not NativeFragment:
            raise ValueError("Native admission needs explicit original moved fragments")
        overlap = validate_native_polygon(overlap)
        if any((fragment.parcel.material_id, fragment.parcel.plate) not in original_materials
               for fragment in (left, right)):
            return PolarityResolution(0, "new_material_requires_new_physical_pair")
        edges = _polygon_arcs(overlap)
        scoped, assessments, directions = set(), [], set()
        receiver_scope = {-1: set(), 1: set()}
        for contact, sub, over, receiver_id, material, moved_arc, increment in compiled:
            receiver, donor = (left, right) if left.fragment_id == receiver_id else (right, left)
            if (receiver.fragment_id != receiver_id or receiver.parcel.material_id != material
                    or receiver.parcel.plate != over or donor.parcel.plate != sub):
                continue
            actual_arcs = tuple(shared for edge in edges
                if (shared := native_intersect_support_arcs(moved_arc, edge)) is not None)
            if not actual_arcs:
                continue
            comparison = -1 if receiver is left else 1
            scoped.add(contact.contact_id)
            receiver_scope[comparison].add(contact.contact_id)
            physical = tuple((trace, carrier) for trace, carrier in carriers
                if (trace.subducting_plate, trace.overriding_plate) == (sub, over)
                and native_intersect_support_arcs(carrier,
                    contact.arc if contact.plate_a == over else _reversed(contact.arc)) is not None)
            if not physical:
                continue
            old = prior_directions.get(contact.contact_id, set())
            for actual in actual_arcs:
                reason, support_rows, coverage = "spatial_underthrust_admissible", [], []
                if len(old) > 1:
                    reason = "conflicting_local_history"
                elif old and old != {(sub, over)}:
                    reason = "history_mechanics_mismatch"
                for trace, carrier in physical:
                    moved_carrier = _rotated(carrier, increment)
                    supported = native_intersect_support_arcs(actual, moved_carrier)
                    if supported is None:
                        continue
                    low, high = _material_support_interval(moved_carrier, supported,
                        certificates[trace.trace_id].extent_radians)
                    status = _assess_interval(certificates[trace.trace_id], low, high)
                    # The plane normal points into the original overriding
                    # material.  This exact relation authenticates fault dip.
                    if moved_carrier.plane.normal != actual.plane.normal:
                        status = "incompatible_physical_fault_orientation"
                    inverse = (RotationIncrement(tuple(-value for value in
                        increment[0].omega_rad_per_myr), dt),)
                    supported_before = _rotated(supported, inverse)
                    endpoint_velocity = {}
                    velocity_status, minimum, maximum, width = _native_velocity_status(
                        carrier, supported_before, trace, omega, surface.radius_km,
                        endpoint_audit=endpoint_velocity)
                    if velocity_status != "spatial_underthrust_admissible":
                        status = velocity_status
                    support_rows.append(dict(trace_id=trace.trace_id,
                        low_radians=low, high_radians=high, status=status,
                        normal_velocity_min_km_per_myr=minimum,
                        normal_velocity_max_km_per_myr=maximum,
                        normal_velocity_enclosure_width_km_per_myr=width,
                        endpoint_velocity_enclosures=endpoint_velocity,
                        native_arc=native_arc_to_dict(supported)))
                    coverage.append(supported)
                if not support_rows:
                    reason = "no_continuous_physical_fault_support"
                elif reason == "spatial_underthrust_admissible":
                    blocked = {row["status"] for row in support_rows}-{reason}
                    if blocked:
                        reason = next(iter(blocked)) if len(blocked) == 1 else "unresolved_local_admission"
                    elif not native_support_covers_arc(actual, coverage):
                        reason = "partial_continuous_contact_coverage"
                row = dict(contact_id=contact.contact_id, subducting_plate=sub,
                    overriding_plate=over, comparison=comparison,
                    receiver_fragment_id=receiver_id, receiver_material_id=material,
                    native_arc=native_arc_to_dict(actual), status=reason,
                    supports=support_rows)
                assessments.append(row)
                if reason == "spatial_underthrust_admissible":
                    directions.add(comparison)
        overlap_row = dict(fragment_a=left.fragment_id, fragment_b=right.fragment_id,
            native_overlap=native_polygon_to_dict(overlap),
            geometric_scope_contact_ids=sorted(scoped),
            receiver_arcs=assessments, comparison=0, accepted_evidence_ids=[])
        overlap_rows.append(overlap_row)
        if not scoped:
            return PolarityResolution(0, "no_resolved_initial_contact")
        supported_contacts = {row["contact_id"] for row in assessments}
        if not supported_contacts:
            return PolarityResolution(0, "no_continuous_physical_fault_support", (), tuple(sorted(scoped)))
        blocked = {row["status"] for row in assessments}-{"spatial_underthrust_admissible"}
        if blocked:
            return PolarityResolution(0, next(iter(blocked)) if len(blocked) == 1 else
                "unresolved_local_admission", (), tuple(sorted(supported_contacts)))
        if len(directions) != 1:
            return PolarityResolution(0, "no_unique_local_down_dip_support", (), tuple(sorted(supported_contacts)))
        comparison = next(iter(directions))
        accepted_contacts = {row["contact_id"] for row in assessments
                             if row["comparison"] == comparison}
        if receiver_scope[comparison]-accepted_contacts:
            return PolarityResolution(0, "no_continuous_physical_fault_support", (),
                tuple(sorted(receiver_scope[comparison])))
        identity = "native-spatial-underthrust:"+_hash(dict(surface=digest,
            node_manifest=step.node_manifest_sha256, actual_receiver_arcs=assessments))
        overlap_row["comparison"] = comparison
        overlap_row["accepted_evidence_ids"] = [identity]
        return PolarityResolution(comparison, "solved_spatial_underthrust_admission",
            (identity,), tuple(sorted(accepted_contacts)))

    return resolve, audit


def _arc_from_dict(raw):
    if not isinstance(raw, dict) or set(raw) != {"start", "end", "plane", "length_radians"}:
        raise ValueError("Invalid native receiver support schema")
    return _validate_arc(NativeArc(native_ray_from_dict(raw["start"]),
        native_ray_from_dict(raw["end"]), native_plane_from_dict(raw["plane"]),
        _real(raw["length_radians"], "Native receiver support extent", positive=True)))


def attach_native_underthrust_losses(before, transaction, *, origin_surface,
        initial_mechanics, owner_histories_after, step, admission_audit, history=()):
    """Bind every accepted loss to its genuine endpoint receiver boundary.

    The entire admission audit is reproduced from the recorded analytic
    histories and authenticated constitutive step.  Neither decision IDs nor
    caller receipt hashes grant support.  Each physical receiver edge of the
    actual loss then needs complete native coverage by endpoint contacts on
    the same original donor and receiver fragment lineages.

    Return ``(attached_transaction, attachment_audit)``.  The input transaction
    and admission audit remain untouched; the attached losses can subsequently
    enter the immutable native buried registry.  That registry must separately
    prove the complete loss footprint belongs to the moved donor material and
    endpoint receiver and that all original material domains still partition.
    """
    if type(before) is not NativeSurfaceState or type(transaction) is not NativeTransportResult:
        raise ValueError("Native loss attachment needs an explicit before state and transaction")
    histories_after = _owner_histories(initial_mechanics, owner_histories_after)
    if any(not history for _, history in histories_after):
        raise ValueError("Native loss attachment requires the actual accepted full pose increment")
    dt = histories_after[0][1][-1].dt_myr
    if any(history[-1].dt_myr != dt for _, history in histories_after):
        raise ValueError("Native endpoint attachment has inconsistent owner clocks")
    if transaction.state.time_myr != before.time_myr+dt:
        raise ValueError("Native endpoint attachment cannot change the transaction clock")
    omega = {owner: history[-1].omega_rad_per_myr for owner, history in histories_after}
    histories_before = tuple((owner, history[:-1]) for owner, history in histories_after)
    replay_resolver, reproduced_audit = make_native_spatial_underthrust_resolver(
        before, omega, dt, origin_surface=origin_surface,
        initial_mechanics=initial_mechanics, owner_histories=histories_before,
        step=step, history=history)
    moved = tuple(NativeFragment(fragment.fragment_id,
        fragment.geometry.rotated(omega[fragment.parcel.plate], dt),
        replace(fragment.parcel, age_myr=fragment.parcel.age_myr+dt),
        fragment.parent_fragment_id) for fragment in before.fragments)
    decisions = {}
    for index, left in enumerate(moved):
        for right in moved[index+1:]:
            overlap = native_intersection(left.geometry, right.geometry)
            if overlap is None:
                continue
            decision = replay_resolver(left, right, overlap)
            decisions[frozenset((left.fragment_id, right.fragment_id))] = decision
    if reproduced_audit != admission_audit:
        raise ValueError("Native loss attachment audit differs from fresh field/location replay")
    endpoint = extract_native_contacts(transaction.state.fragments, before.radius_km)
    if endpoint != transaction.contacts:
        raise ValueError("Native loss attachment contacts differ from the actual endpoint surface")
    original = {fragment.fragment_id: fragment for fragment in before.fragments}
    descendants = {fragment.fragment_id: fragment for fragment in (
        *transaction.state.fragments, *(loss.fragment for loss in transaction.losses))}

    def ancestor(identity):
        seen = set()
        while identity not in original:
            if identity in seen:
                raise ValueError("Native endpoint fragment ancestry contains a cycle")
            seen.add(identity)
            fragment = descendants.get(identity)
            if fragment is None or not fragment.parent_fragment_id:
                raise ValueError("Native endpoint support has no original material-fragment lineage")
            identity = fragment.parent_fragment_id
        return identity

    pose_map = dict(histories_after)
    physical_arcs = tuple((trace, _trace_arc(trace, pose_map[trace.overriding_plate]))
                          for trace in initial_mechanics.fault_traces.traces)
    attachments, losses = [], []
    for loss in transaction.losses:
        donor_id = loss.source_fragment_id
        if donor_id not in original:
            raise ValueError("Native loss changed its original donor fragment identity")
        receiver_id = ancestor(loss.receiver_fragment_id)
        donor, receiver = original[donor_id], original[receiver_id]
        if (loss.receiver_plate != receiver.parcel.plate
                or loss.fragment.parcel.plate != donor.parcel.plate
                or loss.fragment.parcel.material_id != donor.parcel.material_id
                or loss.time_myr != transaction.state.time_myr):
            raise ValueError("Native attached loss changed its original physical banks or clock")
        key = frozenset((donor_id, receiver_id))
        decision = decisions.get(key)
        if (decision is None or not decision.comparison
                or decision.basis != "solved_spatial_underthrust_admission"
                or loss.polarity_basis != decision.basis
                or loss.polarity_evidence_ids != decision.evidence_ids):
            raise ValueError("Native loss has no reproduced original solved direction")
        candidate = next((row for row in reproduced_audit["actual_overlap_assessments"]
            if frozenset((row["fragment_a"], row["fragment_b"])) == key), None)
        if candidate is None or not candidate["comparison"]:
            raise ValueError("Native loss lacks its actual candidate receiver assessment")
        receiver_rows = tuple(row for row in candidate["receiver_arcs"]
            if row["receiver_fragment_id"] == receiver_id
            and row["subducting_plate"] == donor.parcel.plate
            and row["overriding_plate"] == receiver.parcel.plate
            and row["status"] == "spatial_underthrust_admissible")
        if not receiver_rows:
            raise ValueError("Native loss direction is not supported on its own receiver bank")
        admitted = tuple(_arc_from_dict(support["native_arc"]) for row in receiver_rows
            for support in row["supports"] if support["status"] == "spatial_underthrust_admissible")
        required = []
        for edge in _polygon_arcs(loss.fragment.geometry):
            for trace, carrier in physical_arcs:
                if (trace.subducting_plate, trace.overriding_plate) != (
                        donor.parcel.plate, receiver.parcel.plate):
                    continue
                actual = native_intersect_support_arcs(edge, carrier)
                if actual is not None:
                    required.append((trace, actual))
        if not required:
            raise ValueError("Native loss has no inherited original receiver-side fault arc")
        compatible = []
        for contact in endpoint:
            if (contact.plate_a, contact.material_a, contact.plate_b, contact.material_b) == (
                    donor.parcel.plate, donor.parcel.material_id,
                    receiver.parcel.plate, receiver.parcel.material_id):
                source_endpoint, receiver_endpoint = contact.fragment_a, contact.fragment_b
            elif (contact.plate_b, contact.material_b, contact.plate_a, contact.material_a) == (
                    donor.parcel.plate, donor.parcel.material_id,
                    receiver.parcel.plate, receiver.parcel.material_id):
                source_endpoint, receiver_endpoint = contact.fragment_b, contact.fragment_a
            else:
                continue
            if ancestor(source_endpoint) == donor_id and ancestor(receiver_endpoint) == receiver_id:
                compatible.append(contact)
        bound_ids, support_rows = set(), []
        for trace, actual in required:
            if not native_support_covers_arc(actual, admitted):
                raise ValueError("Native loss receiver arc exceeds its solved continuous admission")
            covered = []
            for contact in compatible:
                shared = native_intersect_support_arcs(actual, contact.arc)
                if shared is None:
                    continue
                covered.append(shared)
                bound_ids.add(contact.contact_id)
                support_rows.append(dict(trace_id=trace.trace_id,
                    contact_id=contact.contact_id, native_arc=native_arc_to_dict(shared),
                    donor_fragment_id=donor_id, receiver_fragment_id=receiver_id,
                    donor_material_id=donor.parcel.material_id,
                    receiver_material_id=receiver.parcel.material_id))
            if not native_support_covers_arc(actual, covered):
                raise ValueError("Native underthrust loss lacks complete actual endpoint receiver support")
        attached = replace(loss, contact_ids=tuple(sorted(bound_ids)),
                           attachment_status="geometric_contact")
        losses.append(attached)
        attachments.append(dict(loss_fragment_id=loss.fragment.fragment_id,
            source_fragment_id=donor_id, receiver_fragment_id=receiver_id,
            polarity_evidence_ids=list(loss.polarity_evidence_ids),
            contact_ids=list(attached.contact_ids), supports=support_rows))
    diagnostics = dict(transaction.diagnostics,
        attached_loss_count=len(losses), unresolved_attachment_count=0,
        mechanical_attachment="authenticated original continuous constitutive field and certified native endpoint support")
    attached_transaction = replace(transaction, losses=tuple(losses), diagnostics=diagnostics)
    audit = dict(format=FORMAT, stage="native_endpoint_loss_attachment",
        before_surface_sha256=_hash(native_surface_to_dict(before)),
        after_surface_sha256=_hash(native_surface_to_dict(transaction.state)),
        admission_sha256=_hash(admission_audit), node_manifest_sha256=step.node_manifest_sha256,
        attachments=attachments)
    audit["receipt_sha256"] = _hash(audit)
    return attached_transaction, audit
