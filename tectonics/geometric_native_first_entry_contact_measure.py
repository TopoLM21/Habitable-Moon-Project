"""Certified carrier-to-source material Jacobian and retained interface measure.

An original unit quarter carrier is ``c(theta)=S*cos(theta)+E*sin(theta)``.
Its source-material pullback is ``q(theta,u)=Q_d(u)^T Q_r(u)c(theta)``.
The oriented solid-angle Jacobian is exactly
``q . (q_theta x q_u) = (a_d-a_r) . Q_r(u)c_theta``, where each complete
angular increment is ``a=omega*dt``.  No event-window midpoint occurs here.

This two-dimensional swept-material measure is distinct from the original
one-dimensional contact-interface measure.  The latter retains its declared
dimensional continuum reference density and original computational quarter
extent.  This density is distinct from the separately preserved literal
binary64 quadrature atoms; their totals need not agree exactly. Integrating the
Jacobian over a rectangle counts multiplicity; a first-entry chronology and
its exclusions are required before interpreting it as an archive area.  The
kernel supplies no plastic/energy/work density, quadrature reassignment,
source-material support, first-entry admission or accepted force state.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from fractions import Fraction
import hashlib
import json

from .geometric_native_pose import (
    FractionInterval as I, _rounded_interval, interval_pi,
)
from .geometric_native_contact_time_support import _trig_ranges
from .geometric_native_physical_pair_time import (
    _axes, _budget, _clock, _rotated, _vector,
)


FORMAT = "native-first-entry-carrier-contact-measure-1"
MODEL = "full_SO3_carrier_pullback_Jacobian_retained_original_interface_measure"
SCOPE = "measure_kernel_only_no_chronology_mask_or_archive_area_adoption"


def _dot(first, second):
    return sum((a*b for a, b in zip(first, second)), Fraction(0))


def _cross(first, second):
    return (first[1]*second[2]-first[2]*second[1],
            first[2]*second[0]-first[0]*second[2],
            first[0]*second[1]-first[1]*second[0])


def _compress(value, precision):
    return _rounded_interval(value if type(value) is I else I(value), precision+16)


def _positive(value, name):
    if type(value) is not Fraction or value <= 0:
        raise ValueError(f"{name} requires a positive literal Fraction")
    return value


def _interval(value, name):
    if (type(value) is not I or type(value.lo) is not Fraction
            or type(value.hi) is not Fraction or value.lo > value.hi):
        raise ValueError(f"{name} requires an immutable exact FractionInterval")
    return value


def _absolute(value):
    high = max(abs(value.lo), abs(value.hi))
    low = Fraction(0) if value.lo <= 0 <= value.hi else min(abs(value.lo), abs(value.hi))
    return I(low, high)


@dataclass(frozen=True, slots=True)
class NativeFirstEntryContactMeasure:
    trace_id: str
    start: tuple
    end: tuple
    donor_owner: int
    receiver_owner: int
    omega_records: tuple
    dt_myr: Fraction
    native_angle_bounds: I
    time_fraction_bounds: I
    source_extent_radians: Fraction
    source_interface_area_density_m2_per_radian: Fraction
    precision: int
    order: int
    required_jacobian_sign: int | None
    native_quarter_angle_enclosure: I
    native_quarter_angle_span_rad: I
    carrier_direction: tuple
    carrier_tangent: tuple
    receiver_world_direction: tuple
    receiver_world_tangent: tuple
    source_material_direction: tuple
    source_theta_derivative: tuple
    source_time_derivative: tuple
    signed_solid_angle_jacobian: I
    unsigned_solid_angle_jacobian: I
    jacobian_sign: int | None
    source_angle_scale: I
    computational_angle_bounds: I
    interface_area_density_per_native_radian_m2: I
    rectangle_interface_area_m2: I
    rectangle_multiplicity_solid_angle_sr: I
    measure_sha256: str

    @property
    def first_entry_chronology_certified(self):
        return False

    @property
    def rectangle_is_once_only_archive_area(self):
        return False

    @property
    def source_material_support_certified(self):
        return False

    @property
    def physical_pair_admission_authorized(self):
        return False

    @property
    def candidate_mechanics_adopted(self):
        return False


_SOURCE_NAMES = (
    "trace_id", "start", "end", "donor_owner", "receiver_owner", "omega_records",
    "dt_myr", "native_angle_bounds", "time_fraction_bounds", "source_extent_radians",
    "source_interface_area_density_m2_per_radian", "precision", "order",
    "required_jacobian_sign",
)
_INTERVAL_NAMES = (
    "native_quarter_angle_enclosure", "native_quarter_angle_span_rad",
    "signed_solid_angle_jacobian", "unsigned_solid_angle_jacobian",
    "source_angle_scale", "computational_angle_bounds",
    "interface_area_density_per_native_radian_m2", "rectangle_interface_area_m2",
    "rectangle_multiplicity_solid_angle_sr",
)
_VECTOR_NAMES = (
    "carrier_direction", "carrier_tangent", "receiver_world_direction",
    "receiver_world_tangent", "source_material_direction", "source_theta_derivative",
    "source_time_derivative",
)


def certify_native_first_entry_contact_measure(trace_id, start, end, donor_owner,
        receiver_owner, omega_records, dt_myr, native_angle_bounds,
        time_fraction_bounds, source_extent_radians,
        source_interface_area_density_m2_per_radian, *, precision=256, order=32,
        required_jacobian_sign=None):
    """Enclose the exact source map and both measures on the whole rectangle.

    ``source_extent_radians`` and ``source_interface_area_density...`` are
    literal coefficients of the caller's authenticated original interface.
    They specify the original dimensional continuum reference measure, which
    is distinct from the literal quadrature-atom measure. They are preserved,
    never inferred from quadrature weights or recomputed from a floating dip.
    A strict requested Jacobian sign is refused whenever
    the full rectangle fails to prove it; it is not a first-entry predicate.
    """
    if type(trace_id) is not str or not trace_id:
        raise ValueError("Contact measure requires a literal source trace identity")
    start, end = _vector(start), _vector(end)
    if (any(sum(item != 0 for item in vector) != 1 or _dot(vector, vector) != 1
            for vector in (start, end)) or _dot(start, end) != 0):
        raise ValueError("Contact measure requires exact orthogonal unit axis-quarter endpoints")
    _budget(precision, order)
    dt, time = _clock(dt_myr, time_fraction_bounds)
    theta = _interval(native_angle_bounds, "Native carrier angle")
    half_pi = interval_pi(precision+16)/2
    if theta.lo <= 0:
        # A caller may enclose the whole analytic quarter with a finer pi
        # bound than this evaluator's first budget. Refine an ambiguous upper
        # endpoint before declaring coverage; never infer it from equality of
        # rounded angles. Otherwise the interval result remains conservative.
        for extra in (32, 48, 64):
            if not half_pi.lo <= theta.hi < half_pi.hi:
                break
            half_pi = interval_pi(precision+extra)/2
    if theta.hi < 0 or theta.lo > half_pi.hi:
        raise ValueError("Native angle enclosure misses the declared native quarter domain")
    # A root-box angle can extend slightly beyond a carrier endpoint even
    # when the exact event's quarter incidence is independently proved. The
    # parameter domain here is theta in [0, analytic pi/2], intersected with
    # the supplied enclosure; this is not a new material-incidence proof.
    quarter = I(max(Fraction(0), theta.lo), min(theta.hi, half_pi.hi))
    high_end = I(min(theta.hi, half_pi.lo), min(theta.hi, half_pi.hi))
    span = I(max(Fraction(0), high_end.lo-quarter.lo),
             max(Fraction(0), high_end.hi-quarter.lo))
    extent = _positive(source_extent_radians, "Original computational quarter extent")
    density = _positive(source_interface_area_density_m2_per_radian,
                        "Original interface area density")
    if (required_jacobian_sign is not None
            and (type(required_jacobian_sign) is not int
                 or required_jacobian_sign not in (-1, 0, 1))):
        raise ValueError("Requested Jacobian sign must be literal -1, 0, 1 or None")
    donor_axis, receiver_axis = _axes(omega_records, dt, donor_owner, receiver_owner)
    # Inverse-angle bounds can have enormous exact rational denominators.
    # Compress the complete enclosure outward before the alternating series;
    # this preserves every allowed angle and avoids degree-proportional growth
    # of those source denominators. A tiny represented angle remains tiny.
    if quarter.lo >= half_pi.hi/2:
        # An implicit endpoint root may be only 1e-70 from the quarter end.
        # Evaluate its small complementary angle and use sin/cos identities;
        # subtracting floating pi/2 would erase this part of the source.
        complement = I(max(Fraction(0), half_pi.lo-quarter.hi),
                       max(Fraction(0), half_pi.hi-quarter.lo))
        trig_theta = _rounded_interval(complement, precision+24)
        cosine, sine = _trig_ranges(trig_theta.lo, trig_theta.hi, precision+16)
    else:
        trig_theta = _rounded_interval(quarter, precision+24)
        sine, cosine = _trig_ranges(trig_theta.lo, trig_theta.hi, precision+16)
    carrier = tuple(_compress(s*cosine+e*sine, precision) for s, e in zip(start, end))
    tangent = tuple(_compress(-s*sine+e*cosine, precision) for s, e in zip(start, end))
    world = _rotated(carrier, receiver_axis, time, precision, order)
    world_tangent = _rotated(tangent, receiver_axis, time, precision, order)
    negative_donor = tuple(-item for item in donor_axis)
    source = _rotated(world, negative_donor, time, precision, order)
    source_tangent = _rotated(world_tangent, negative_donor, time, precision, order)
    relative_axis = tuple(a-b for a, b in zip(donor_axis, receiver_axis))
    # This invariant formula avoids subtracting nearly equal transformed unit
    # directions. It includes every component of both angular velocities.
    signed = _compress(_dot(relative_axis, world_tangent), precision)
    world_velocity = _cross(tuple(-item for item in relative_axis), world)
    source_velocity = _rotated(world_velocity, negative_donor, time, precision, order)
    unsigned = _absolute(signed)
    sign = 1 if signed.lo > 0 else -1 if signed.hi < 0 else 0 if signed.lo == signed.hi == 0 else None
    if required_jacobian_sign is not None and sign != required_jacobian_sign:
        raise ValueError("Whole carrier/time rectangle does not prove the requested Jacobian sign")
    angle_scale = _compress(extent/half_pi, precision)
    computational_raw = _compress(quarter*angle_scale, precision)
    computational = I(max(Fraction(0), computational_raw.lo),
                      min(extent, computational_raw.hi))
    interface_density = _compress(density*angle_scale, precision)
    interface_area = (I(density*extent)
                      if quarter.lo == 0 and theta.hi >= half_pi.hi
                      else _compress(interface_density*span, precision))
    multiplicity_area = _compress(unsigned*span*time.width, precision)
    result = NativeFirstEntryContactMeasure(trace_id, start, end, donor_owner,
        receiver_owner, omega_records, dt, theta, time, extent, density, precision,
        order, required_jacobian_sign, quarter, span, carrier, tangent, world, world_tangent,
        source, source_tangent, source_velocity, signed, unsigned, sign, angle_scale,
        computational, interface_density, interface_area, multiplicity_area, "")
    return replace(result, measure_sha256=_digest(_body(result)))


def _fraction(value):
    return [value.numerator, value.denominator]


def _ib(value):
    return dict(lo=_fraction(value.lo), hi=_fraction(value.hi))


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def _digest(value):
    return hashlib.sha256(_encoded(value)).hexdigest()


def _body(result):
    source = dict(trace_id=result.trace_id, start=list(map(_fraction, result.start)),
        end=list(map(_fraction, result.end)), donor_owner=result.donor_owner,
        receiver_owner=result.receiver_owner,
        omega_records=[[owner, list(map(_fraction, axis))] for owner, axis in result.omega_records],
        dt_myr=_fraction(result.dt_myr), native_angle_bounds=_ib(result.native_angle_bounds),
        time_fraction_bounds=_ib(result.time_fraction_bounds),
        source_extent_radians=_fraction(result.source_extent_radians),
        source_interface_area_density_m2_per_radian=_fraction(
            result.source_interface_area_density_m2_per_radian),
        precision=result.precision, order=result.order,
        required_jacobian_sign=result.required_jacobian_sign)
    body = dict(format=FORMAT, model=MODEL, scope=SCOPE, source=source,
        jacobian_sign=result.jacobian_sign,
        original_interface_measure_preserved=True,
        source_density_interpretation="original_dimensional_continuum_reference_measure",
        literal_quadrature_atoms_reweighted=False,
        continuum_total_equals_literal_nodal_area_asserted=False,
        angle_domain="supplied_enclosure_intersect_declared_native_quarter",
        solid_angle_metric="reference_unit_sphere_not_deformed_radial_beam_surface",
        interface_measure_is_swept_material_measure=False,
        first_entry_chronology_certified=False,
        rectangle_is_once_only_archive_area=False,
        source_material_support_certified=False,
        nodal_work_or_energy_density_assigned=False,
        physical_pair_admission_authorized=False, candidate_mechanics_adopted=False,
        accepted_native_force_intervals=0, accepted_work_increment_j=[0, 1])
    body.update({name: _ib(getattr(result, name)) for name in _INTERVAL_NAMES})
    body.update({name: list(map(_ib, getattr(result, name))) for name in _VECTOR_NAMES})
    return body


def _typed_fingerprint(value):
    kind = type(value)
    if kind in (str, int, bool, type(None)):
        return kind, value
    if kind is Fraction:
        return kind, value.numerator, value.denominator
    if kind is tuple:
        return kind, tuple(_typed_fingerprint(item) for item in value)
    if kind in (I, NativeFirstEntryContactMeasure):
        return kind, tuple((field.name, _typed_fingerprint(getattr(value, field.name)))
                           for field in fields(value))
    raise ValueError("Contact measure replay requires immutable literal typed fields")


def validate_native_first_entry_contact_measure(result):
    if type(result) is not NativeFirstEntryContactMeasure:
        raise ValueError("Expected immutable native first-entry contact measure")
    source = {name: getattr(result, name) for name in _SOURCE_NAMES}
    replay = certify_native_first_entry_contact_measure(**source)
    if _typed_fingerprint(result) != _typed_fingerprint(replay):
        raise ValueError("Contact measure differs from complete exact source replay")
    return result


def native_first_entry_contact_measure_to_dict(result):
    result = validate_native_first_entry_contact_measure(result)
    return dict(_body(result), measure_sha256=result.measure_sha256)


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
    raise ValueError("Contact measure DTO requires canonical literal JSON without float or tuple aliases")


def native_first_entry_contact_measure_from_dict(raw, *, expected_source):
    """Recompute every field from the complete authoritative original source."""
    if (type(raw) is not dict or type(expected_source) is not dict
            or set(expected_source) != set(_SOURCE_NAMES)):
        raise ValueError("Contact measure loader requires complete authoritative source parameters")
    _literal_json(raw)
    replay = certify_native_first_entry_contact_measure(**expected_source)
    expected = dict(_body(replay), measure_sha256=replay.measure_sha256)
    if _encoded(raw) != _encoded(expected):
        raise ValueError("Saved contact measure differs from authoritative exact replay")
    return replay
