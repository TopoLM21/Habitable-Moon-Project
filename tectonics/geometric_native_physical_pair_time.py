"""Certified rational-time geometry of the unchanged original physical tips.

The stored quadrature direction and frame vectors are literal material labels:
they are never normalized or replaced with a nearby spherical direction.  The
two original tips follow their respective owners; frames follow the overriding
owner.  Radial coordinates have an explicitly diagnostic affine extension from
the prepared state to the unaccepted candidate.  Finite-pose separation changes
do not replace integrated free anchors or evolve plastic memory, work or clocks.

The relative-entry helper concerns a different, incoming material coordinate.
Mapping that coordinate to a carrier angle supplies geometry only.  No radial
tip displacement, original spring history or contact admission is assigned to it.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from fractions import Fraction
import hashlib
import json
from math import factorial

from .geometric_native_pose import (
    FractionInterval as I, _rounded_interval, interval_atan, interval_pi,
    interval_sqrt,
)
from .geometric_native_contact_events import _polynomial_bounds


FORMAT = "native-original-physical-pair-tip-time-maps-1"
MODEL = "literal_original_tip_labels_full_SO3_affine_candidate_radial_diagnostic"
SCOPE = "original_pair_geometry_only_no_anchor_rebase_no_material_handover"


def _vector(value, size=3, *, nonzero=True):
    if (type(value) is not tuple or len(value) != size
            or any(type(item) is not Fraction for item in value)):
        raise ValueError("Physical pair coordinates require immutable literal Fraction tuples")
    if nonzero and not any(value):
        raise ValueError("Physical pair direction must be nonzero")
    return value


def _owner(value):
    if type(value) is not int or value < 0:
        raise ValueError("Physical pair owners require literal nonnegative integers")
    return value


def _omega(value):
    if (type(value) is not tuple or not value or any(type(row) is not tuple
            or len(row) != 2 for row in value)):
        raise ValueError("Physical pair motion requires immutable complete owner records")
    owners = []
    for owner, vector in value:
        owners.append(_owner(owner))
        _vector(vector, nonzero=False)
    if tuple(owners) != tuple(sorted(set(owners))):
        raise ValueError("Physical pair motion owners must be unique and canonical")
    return value


def _clock(dt_myr, time):
    if type(dt_myr) is not Fraction or dt_myr <= 0:
        raise ValueError("Physical pair duration requires a positive literal Fraction")
    if type(time) is not I or not 0 <= time.lo <= time.hi <= 1:
        raise ValueError("Physical pair time must be an exact enclosure within [0,1]")
    return dt_myr, time


def _budget(precision, order):
    if type(precision) is not int or not 32 <= precision <= 1024:
        raise ValueError("Physical pair precision must be an integer from 32 to 1024")
    if type(order) is not int or not 8 <= order <= 64:
        raise ValueError("Physical pair Taylor order must be an integer from 8 to 64")


def _dot(a, b):
    return sum((x*y for x, y in zip(a, b)), Fraction(0))


def _cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2],
            a[0]*b[1]-a[1]*b[0])


def _maximum(value):
    value = value if type(value) is I else I(value)
    return max(abs(value.lo), abs(value.hi))


def _compressed(value, precision):
    return _rounded_interval(value if type(value) is I else I(value), precision+16)


def _rotation_change(vector, axis, time, precision, order):
    """Enclose (exp([axis]u)-identity) vector without a unit-size subtraction.

    Coefficient enclosures are compressed outward, preventing repeated owner
    maps from producing unbounded rational denominators.  The absolute tail
    bound uses the induced l1 matrix norm; it encloses every component.
    """
    values = tuple(item if type(item) is I else I(item) for item in vector)
    zeros = (I(0),)*3
    if not time.hi or not any(axis):
        return zeros
    if all(type(item) is Fraction for item in vector) and not any(_cross(axis, vector)):
        return zeros
    rate = sum(map(abs, axis), Fraction(0))
    ratio = rate*time.hi/Fraction(order+2)
    if ratio >= 1:
        raise ValueError("Physical pair rotation exceeds the declared Taylor remainder budget")
    scale = sum(map(_maximum, values), Fraction(0))
    tail = scale*(rate*time.hi)**(order+1)/factorial(order+1)/(1-ratio)
    tail = _compressed(I(tail), precision).hi
    coefficients = [zeros]
    coefficient = values
    for degree in range(1, order+1):
        coefficient = tuple(_compressed(item/degree, precision)
                            for item in _cross(axis, coefficient))
        coefficients.append(coefficient)
    return tuple(_polynomial_bounds(tuple(row[index] for row in coefficients),
        time, tail, precision+8) for index in range(3))


def _rotated(vector, axis, time, precision, order):
    changes = _rotation_change(vector, axis, time, precision, order)
    return tuple(_compressed(item+change, precision)
                 for item, change in zip(vector, changes))


def _axes(records, dt, sub, over):
    records = _omega(records)
    sub, over = _owner(sub), _owner(over)
    if sub == over:
        raise ValueError("Original physical pair requires two different owners")
    values = dict(records)
    if sub not in values or over not in values:
        raise ValueError("Physical pair motion lacks one of its actual owners")
    return tuple(item*dt for item in values[sub]), tuple(item*dt for item in values[over])


def _relative_change(point, donor_axis, receiver_axis, time, precision, order):
    # Q_r^-1 Q_d p-p = (Q_r^-1 p-p) + Q_r^-1 (Q_d p-p).
    # This difference form keeps the tiny physical change separate from p.
    if donor_axis == receiver_axis:
        return (I(0),)*3
    negative_receiver = tuple(-item for item in receiver_axis)
    donor_change = _rotation_change(point, donor_axis, time, precision, order)
    receiver_change = _rotation_change(point, negative_receiver, time, precision, order)
    moved_change = _rotated(donor_change, negative_receiver, time, precision, order)
    return tuple(_compressed(a+b, precision) for a, b in zip(receiver_change, moved_change))


def certify_native_relative_entry_coordinates(original_point, donor_owner,
        receiver_owner, omega_records, dt_myr, time_fraction_bounds, *,
        precision=128, order=32, normalize_direction=False):
    """Bound Q_receiver(u)^T Q_donor(u) original_point on the whole time window.

    Homogeneous input coordinates are preserved by default.  Explicit
    ``normalize_direction=True`` converts this incoming point to a unit ray;
    original physical tip maps never call this normalization operation.
    """
    point = _vector(original_point)
    dt, time = _clock(dt_myr, time_fraction_bounds)
    _budget(precision, order)
    if type(normalize_direction) is not bool:
        raise ValueError("Incoming normalization requires a literal explicit boolean")
    donor, receiver = _axes(omega_records, dt, donor_owner, receiver_owner)
    change = _relative_change(point, donor, receiver, time, precision, order)
    result = tuple(_compressed(item+delta, precision) for item, delta in zip(point, change))
    if normalize_direction:
        norm = interval_sqrt(I(_dot(point, point)), precision+16)
        result = tuple(_compressed(item/norm, precision) for item in result)
    return result


def native_direction_is_positive_scale(first, second):
    """Exact homogeneous direction identity; it is not tip-label identity."""
    first, second = _vector(first), _vector(second)
    return not any(_cross(first, second)) and _dot(first, second) > 0


def certify_native_quarter_carrier_coordinate(start, end, relative_coordinates,
                                              *, precision=128):
    """Enclose atan2(end.dot(p),start.dot(p)) on a proven axis-quarter carrier.

    This helper converts coordinates only.  Its caller must prove incidence
    in the carrier plane at the exact event; an enclosure containing that
    plane is not an incidence proof.  Endpoint singleton angles are retained.
    A window crossing an unknown quadrant refuses instead of clipping it.
    """
    start, end = _vector(start), _vector(end)
    _budget(precision, 32)
    if (any(sum(item != 0 for item in point) != 1 or _dot(point, point) != 1
            for point in (start, end)) or _dot(start, end) != 0):
        raise ValueError("Carrier coordinate requires exact orthogonal unit axis endpoints")
    if (type(relative_coordinates) is not tuple or len(relative_coordinates) != 3
            or any(type(value) is not I for value in relative_coordinates)):
        raise ValueError("Carrier coordinate requires an immutable exact interval vector")
    cosine, sine = _dot(start, relative_coordinates), _dot(end, relative_coordinates)
    if cosine.lo < 0 or sine.lo < 0:
        raise ValueError("Carrier coordinate has an unresolved or incompatible quadrant")
    if sine.hi == 0 and cosine.lo > 0:
        return I(0)
    half_pi = interval_pi(precision+16)/2
    if cosine.hi == 0 and sine.lo > 0:
        return half_pi
    if cosine.lo <= 0 and sine.lo <= 0:
        raise ValueError("Carrier coordinate lacks a certified nonzero quadrant direction")
    if cosine.lo > 0:
        return interval_atan(sine/cosine, precision+8)
    return half_pi-interval_atan(cosine/sine, precision+8)


@dataclass(frozen=True, slots=True)
class NativeOriginalPairTipMaps:
    trace_id: str
    reference_coordinates: tuple
    original_normal: tuple
    original_down_direction: tuple
    sub_owner: int
    over_owner: int
    omega_records: tuple
    dt_myr: Fraction
    radius_m: Fraction
    radial_before: tuple
    radial_candidate: tuple
    time_fraction_bounds: I
    precision: int
    order: int
    sub_direction: tuple
    over_direction: tuple
    sub_position_m: tuple
    over_position_m: tuple
    normal: tuple
    down_direction: tuple
    radial_at_time: tuple
    separation_m: tuple
    reference_gap_m: Fraction
    reference_jump_m: Fraction
    gap_m: I
    jump_m: I
    exact_gap_change_m: I
    exact_jump_change_m: I
    source_sha256: str
    maps_sha256: str

    @property
    def physical_pair_admission_authorized(self):
        return False

    @property
    def candidate_mechanics_adopted(self):
        return False

    @property
    def finite_pose_change_replaces_free_anchors(self):
        return False

    @property
    def material_handover_defined(self):
        return False


_SOURCE_NAMES = ("trace_id", "reference_coordinates", "original_normal",
    "original_down_direction", "sub_owner", "over_owner", "omega_records",
    "dt_myr", "radius_m", "radial_before", "radial_candidate",
    "time_fraction_bounds", "precision", "order")


def certify_native_original_pair_tip_maps(trace_id, reference_coordinates,
        original_normal, original_down_direction, sub_owner, over_owner,
        omega_records, dt_myr, radial_before, radial_candidate,
        time_fraction_bounds, *, radius_m, precision=128, order=32):
    """Certify unchanged original tips, frames and finite-pose diagnostics.

    The supplied full owner set is retained, including owners outside this
    pair.  No physical authority is inferred from these mathematical inputs.
    The surrounding history binder authenticates literal checkpoint labels.
    """
    if type(trace_id) is not str or not trace_id.strip():
        raise ValueError("Original physical pair requires a nonempty literal trace ID")
    point = _vector(reference_coordinates)
    normal, down = _vector(original_normal), _vector(original_down_direction)
    if not any(_cross(normal, down)):
        raise ValueError("Original physical pair frame axes must be independent")
    before = _vector(radial_before, 2, nonzero=False)
    after = _vector(radial_candidate, 2, nonzero=False)
    dt, time = _clock(dt_myr, time_fraction_bounds)
    _budget(precision, order)
    if type(radius_m) is not Fraction or radius_m <= 0:
        raise ValueError("Original tip radius requires a positive literal Fraction")
    if any(radius_m+value <= 0 for value in (*before, *after)):
        raise ValueError("Original physical tips require positive radial positions")
    sub_axis, over_axis = _axes(omega_records, dt, sub_owner, over_owner)
    radial = tuple(I(old)+time*(new-old) for old, new in zip(before, after))
    sub_change = _rotation_change(point, sub_axis, time, precision, order)
    over_change = _rotation_change(point, over_axis, time, precision, order)
    sub = tuple(_compressed(value+change, precision) for value, change in zip(point, sub_change))
    over = tuple(_compressed(value+change, precision) for value, change in zip(point, over_change))
    sub_position = tuple(_compressed((radius_m+radial[0])*value, precision) for value in sub)
    over_position = tuple(_compressed((radius_m+radial[1])*value, precision) for value in over)
    moved_normal = _rotated(normal, over_axis, time, precision, order)
    moved_down = _rotated(down, over_axis, time, precision, order)
    rotation_difference = ((I(0),)*3 if sub_axis == over_axis else
        tuple(oc-sc for oc, sc in zip(over_change, sub_change)))
    difference = tuple(_compressed((radial[1]-radial[0])*direction
        +(radius_m+radial[0])*change, precision)
        for direction, change in zip(over, rotation_difference))
    relative_change = _relative_change(point, sub_axis, over_axis, time, precision, order)
    radial_difference_change = (radial[1]-radial[0])-(before[1]-before[0])
    reference_gap = (before[1]-before[0])*_dot(normal, point)
    reference_jump = (before[1]-before[0])*_dot(down, point)
    gap_change = _compressed(radial_difference_change*_dot(normal, point)
        -(radius_m+radial[0])*_dot(normal, relative_change), precision)
    jump_change = _compressed(radial_difference_change*_dot(down, point)
        -(radius_m+radial[0])*_dot(down, relative_change), precision)
    result = NativeOriginalPairTipMaps(trace_id, point, normal, down, sub_owner,
        over_owner, omega_records, dt, radius_m, before, after, time, precision,
        order, sub, over, sub_position, over_position, moved_normal, moved_down,
        radial, difference, reference_gap, reference_jump,
        gap_change+reference_gap, jump_change+reference_jump, gap_change,
        jump_change, "", "")
    source_sha = _hash(_source_body(result))
    result = replace(result, source_sha256=source_sha)
    return replace(result, maps_sha256=_hash(_body(result)))


def _rational(value):
    if type(value) is not Fraction:
        raise ValueError("Physical pair serialization requires literal Fractions")
    return [value.numerator, value.denominator]


def _interval(value):
    if type(value) is not I:
        raise ValueError("Physical pair serialization requires exact intervals")
    return dict(lo=_rational(value.lo), hi=_rational(value.hi))


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _hash(value):
    return hashlib.sha256(_encoded(value)).hexdigest()


def _source_body(maps):
    return dict(trace_id=maps.trace_id,
        reference_coordinates=[_rational(item) for item in maps.reference_coordinates],
        original_normal=[_rational(item) for item in maps.original_normal],
        original_down_direction=[_rational(item) for item in maps.original_down_direction],
        sub_owner=maps.sub_owner, over_owner=maps.over_owner,
        omega_records=[[owner, [_rational(item) for item in vector]]
                       for owner, vector in maps.omega_records],
        dt_myr=_rational(maps.dt_myr), radius_m=_rational(maps.radius_m),
        radial_before=[_rational(item) for item in maps.radial_before],
        radial_candidate=[_rational(item) for item in maps.radial_candidate],
        time_fraction_bounds=_interval(maps.time_fraction_bounds),
        precision=maps.precision, order=maps.order)


def _body(maps):
    body = dict(format=FORMAT, model=MODEL, scope=SCOPE, source=_source_body(maps),
        source_sha256=maps.source_sha256,
        reference_gap_m=_rational(maps.reference_gap_m),
        reference_jump_m=_rational(maps.reference_jump_m),
        physical_pair_admission_authorized=False, candidate_mechanics_adopted=False,
        finite_pose_change_replaces_free_anchors=False, material_handover_defined=False,
        accepted_work_increment_j=[0, 1], accepted_native_force_intervals=0)
    for name in ("sub_direction", "over_direction", "sub_position_m", "over_position_m",
                 "normal", "down_direction", "radial_at_time", "separation_m"):
        body[name] = [_interval(value) for value in getattr(maps, name)]
    for name in ("gap_m", "jump_m", "exact_gap_change_m", "exact_jump_change_m"):
        body[name] = _interval(getattr(maps, name))
    return body


def _parameters(maps):
    return {name: getattr(maps, name) for name in _SOURCE_NAMES}


def _typed_fingerprint(value):
    kind = type(value)
    if kind in (str, int, bool, type(None)):
        return kind, value
    if kind is Fraction:
        return kind, value.numerator, value.denominator
    if kind is tuple:
        return kind, tuple(_typed_fingerprint(item) for item in value)
    if kind in (I, NativeOriginalPairTipMaps):
        return kind, tuple((field.name, _typed_fingerprint(getattr(value, field.name)))
                           for field in fields(value))
    raise ValueError("Physical pair replay requires immutable literal typed fields")


def validate_native_original_pair_tip_maps(maps):
    if type(maps) is not NativeOriginalPairTipMaps:
        raise ValueError("Expected immutable original physical pair time maps")
    reconstructed = certify_native_original_pair_tip_maps(**_parameters(maps))
    if _typed_fingerprint(maps) != _typed_fingerprint(reconstructed):
        raise ValueError("Original pair maps differ from complete exact source replay")
    return maps


def native_original_pair_tip_maps_to_dict(maps):
    maps = validate_native_original_pair_tip_maps(maps)
    return dict(_body(maps), maps_sha256=maps.maps_sha256)


def _literal_json(value):
    kind = type(value)
    if kind in (str, int, bool, type(None)):
        return
    if kind is list:
        for item in value:
            _literal_json(item)
        return
    if kind is dict and all(type(key) is str for key in value):
        for item in value.values():
            _literal_json(item)
        return
    raise ValueError("Physical pair DTO must contain literal canonical JSON field types")


def native_original_pair_tip_maps_from_dict(raw, *, expected_source):
    """Replay against caller-authenticated exact parameters, not a supplied SHA.

    ``expected_source`` is the full keyword-parameter dictionary accepted by
    ``certify_native_original_pair_tip_maps``, including precision and order.
    Native JSON records alone cannot authenticate physical checkpoint labels.
    """
    if (type(raw) is not dict or type(expected_source) is not dict
            or set(expected_source) != set(_SOURCE_NAMES)):
        raise ValueError("Original pair replay requires the complete authoritative source parameters")
    _literal_json(raw)
    result = certify_native_original_pair_tip_maps(**expected_source)
    if _encoded(raw) != _encoded(dict(_body(result), maps_sha256=result.maps_sha256)):
        raise ValueError("Saved original physical pair maps differ from authoritative exact replay")
    return result
