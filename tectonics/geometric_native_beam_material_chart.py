"""Exact material coordinates in the unchanged source cantilever strip.

The source shape is the existing cubic tip-load ansatz.  Its value at an
interior contact is a one-mode Ritz coupling to that source tip amplitude;
it is not a fresh free-tip displacement or a new continuum response.  The
literal material labels are homogeneous coordinates, never normalized here.

The independent beam-cut helper restricts this same cubic.  It carries the
displacement, slope, moment, shear and energy of both pieces per unit strike.
It supplies no moving-cut law, material correspondence, contact admission or
proof that the original strip and clamp have native material support.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from fractions import Fraction
from functools import lru_cache
import hashlib
import json

from .geometric_native_pose import (
    FractionInterval as I, _rounded_interval, interval_atan, interval_pi,
    interval_sqrt,
)
from .geometric_native_physical_pair_time import (
    certify_native_quarter_carrier_coordinate,
)


FORMAT = "native-source-beam-material-chart-1"
MODEL = "unchanged_source_cantilever_one_mode_Ritz_interior_coordinate"
SCOPE = "source_shape_only_no_support_or_material_handover_admission"
CUT_FORMAT = "native-source-beam-cut-history-1"
CUT_MODEL = "restriction_of_unchanged_source_cubic_per_unit_strike"
CUT_SCOPE = "fixed_cut_algebra_only_no_moving_cut_or_new_boundary_condition"


def _precision(value):
    if type(value) is not int or not 32 <= value <= 1024:
        raise ValueError("Beam chart precision must be a literal integer from 32 to 1024")
    return value


def _positive(value, name):
    if type(value) is not Fraction or value <= 0:
        raise ValueError(f"{name} requires a positive literal Fraction")
    return value


def _interval(value, name):
    if (type(value) is not I or type(value.lo) is not Fraction
            or type(value.hi) is not Fraction or value.lo > value.hi):
        raise ValueError(f"{name} requires an immutable exact FractionInterval")
    return value


def _coordinate(value):
    if type(value) is Fraction:
        return I(value)
    return _interval(value, "Beam cut distance")


def _vector(value, *, intervals=False):
    if type(value) is not tuple or len(value) != 3:
        raise ValueError("Beam labels require immutable literal three-component tuples")
    if all(type(item) is Fraction for item in value):
        return value
    if intervals and all(type(item) is I for item in value):
        for item in value:
            _interval(item, "Beam material coordinate")
        return value
    raise ValueError("Beam labels require all literal Fractions or all exact intervals")


def _dot(first, second):
    return sum((a*b for a, b in zip(first, second)), Fraction(0))


def _cross(first, second):
    return (first[1]*second[2]-first[2]*second[1],
            first[2]*second[0]-first[0]*second[2],
            first[0]*second[1]-first[1]*second[0])


def _square(value):
    value = value if type(value) is I else I(value)
    high = max(value.lo*value.lo, value.hi*value.hi)
    low = Fraction(0) if value.lo <= 0 <= value.hi else min(
        value.lo*value.lo, value.hi*value.hi)
    return I(low, high)


def _phi_point(z, precision):
    # Compress the small difference from the trench value, rather than the
    # value near one: a positive 1e-800 coordinate must still give phi < 1.
    if z <= Fraction(1, 2):
        deficit = z*(3-z*z)/2
        return 1-_rounded_interval(I(deficit), precision+16)
    return _rounded_interval(I((1-z)*(1-z)*(z+2)/2), precision+16)


def _slope_point(z, precision):
    # The analogous anchored form preserves a sub-float z**2 correction.
    if z <= Fraction(1, 2):
        return -Fraction(3, 2)+_rounded_interval(I(Fraction(3, 2)*z*z), precision+16)
    return _rounded_interval(I(-Fraction(3, 2)*(1-z)*(1+z)), precision+16)


def _shape(z, precision, *, exact=False):
    if not 0 <= z.lo <= z.hi <= 1:
        raise ValueError("The complete beam coordinate enclosure must lie inside [0,1]")
    if exact:
        phi = I((1-z.hi)**2*(z.hi+2)/2, (1-z.lo)**2*(z.lo+2)/2)
        slope = I(-Fraction(3, 2)*(1-z.lo)*(1+z.lo),
                  -Fraction(3, 2)*(1-z.hi)*(1+z.hi))
    else:
        phi = I(_phi_point(z.hi, precision).lo, _phi_point(z.lo, precision).hi)
        slope = I(_slope_point(z.lo, precision).lo, _slope_point(z.hi, precision).hi)
    return phi, slope, 3*z, I(3)


@dataclass(frozen=True, slots=True)
class NativeBeamMaterialChart:
    trace_id: str
    bank: str
    start: tuple
    end: tuple
    reference_coordinates: tuple
    radius_m: Fraction
    width_m: Fraction
    tip_displacement_m: I
    precision: int
    inward: tuple
    start_projection: I
    end_projection: I
    inward_projection: I
    in_plane_norm: I
    along_strike_angle_rad: I
    inward_angle_rad: I
    inward_distance_m: I
    x_over_width: I
    shape: I
    shape_slope: I
    shape_curvature: I
    shape_third_derivative: I
    displacement_m: I
    slope: I
    curvature_per_m: I
    third_derivative_per_m2: I
    source_sha256: str
    chart_sha256: str

    @property
    def material_labels_normalized(self):
        return False

    @property
    def native_bank_support_certified(self):
        return False

    @property
    def physical_pair_admission_authorized(self):
        return False

    @property
    def candidate_mechanics_adopted(self):
        return False

    @property
    def source_shape_is_continuum_interior_contact_solution(self):
        return False


_SOURCE_NAMES = ("trace_id", "bank", "start", "end", "reference_coordinates",
                 "radius_m", "width_m", "tip_displacement_m", "precision")


@lru_cache(maxsize=256)
def _chart_geometry(start, end, point, bank, radius, width, precision):
    # Public validation precedes this cache, so equal numeric aliases cannot
    # bypass the source's literal type or immutable-container requirements.
    if width/radius >= interval_pi(precision+16).lo/2:
        raise ValueError("Original beam width must be certified inside the inward chart hemisphere")
    sign = -1 if bank == "sub" else 1
    inward = tuple(sign*item for item in _cross(start, end))
    coordinate = tuple(I(item) if type(item) is Fraction else item for item in point)
    along_start = _dot(start, coordinate)
    along_end = _dot(end, coordinate)
    transverse = _dot(inward, coordinate)
    if along_start.lo < 0 or along_end.lo < 0 or transverse.lo < 0:
        raise ValueError("Beam material chart has an unresolved or incompatible quadrant")
    # Cancel a common positive homogeneous factor in the atan2 ratio before
    # the bounded square root.  This is algebra on the projections, not a
    # unit-ray normalization of the literal stored material coordinate.
    scale = max(along_start.hi, along_end.hi)
    if scale <= 0:
        raise ValueError("Beam material chart lacks a certified nonzero in-plane direction")
    scaled_norm = interval_sqrt(_square(along_start/scale)+_square(along_end/scale), precision+16)
    plane_norm = scale*scaled_norm
    if plane_norm.lo <= 0:
        raise ValueError("Beam material chart lacks a certified nonzero in-plane direction")
    theta = certify_native_quarter_carrier_coordinate(start, end, coordinate,
                                                      precision=precision)
    beta = interval_atan((transverse/scale)/scaled_norm, precision+16)
    x = radius*beta
    if not 0 <= x.lo <= x.hi <= width:
        raise ValueError("Complete beam material distance enclosure lies outside the original width")
    z = x/width
    phi, slope_shape, curvature_shape, third_shape = _shape(z, precision)
    return (inward, along_start, along_end, transverse, plane_norm, theta, beta,
            x, z, phi, slope_shape, curvature_shape, third_shape)


def certify_native_beam_material_chart(trace_id, bank, start, end,
        reference_coordinates, radius_m, width_m, tip_displacement_m, *,
        precision=256):
    """Certify the interior value of the original bank's source cubic.

    ``start`` and ``end`` are exact unit endpoints of a native axis quarter.
    For a sub bank the inward axis is ``-cross(start,end)``; for an over bank
    it is the positive cross product.  ``reference_coordinates`` may be a
    literal Fraction vector or an exact interval vector in this owner frame.
    All three first-quadrant signs and the whole width enclosure must prove
    their membership.  Intersection with a valid bank is not sufficient.

    The caller authenticates the source trace, its original bank width and
    its own tip amplitude.  This mathematical replay alone confers none of
    those physical authorities and cannot assign another bank's amplitude.
    """
    if type(trace_id) is not str or not trace_id.strip():
        raise ValueError("A beam material chart requires a nonempty literal trace ID")
    if type(bank) is not str or bank not in ("sub", "over"):
        raise ValueError("A beam material chart bank must be literal 'sub' or 'over'")
    start, end = _vector(start), _vector(end)
    if (any(sum(item != 0 for item in axis) != 1 or _dot(axis, axis) != 1
            for axis in (start, end)) or _dot(start, end) != 0):
        raise ValueError("Beam material charts require exact orthogonal unit axis-quarter endpoints")
    point = _vector(reference_coordinates, intervals=True)
    radius = _positive(radius_m, "Beam chart radius")
    width = _positive(width_m, "Original beam width")
    tip = _interval(tip_displacement_m, "Original beam tip displacement")
    precision = _precision(precision)
    (inward, along_start, along_end, transverse, plane_norm, theta, beta, x, z,
     phi, slope_shape, curvature_shape, third_shape) = _chart_geometry(
         start, end, point, bank, radius, width, precision)
    result = NativeBeamMaterialChart(trace_id, bank, start, end, point, radius,
        width, tip, precision, inward, along_start, along_end, transverse,
        plane_norm, theta, beta, x, z, phi, slope_shape, curvature_shape,
        third_shape, phi*tip, slope_shape*tip/width,
        curvature_shape*tip/(width*width), third_shape*tip/(width**3), "", "")
    result = replace(result, source_sha256=_hash(_source_body(result)))
    return replace(result, chart_sha256=_hash(_chart_body(result)))


@dataclass(frozen=True, slots=True)
class NativeBeamCutHistory:
    width_m: Fraction
    cut_distance_m: Fraction | I
    tip_displacement_m: I
    rigidity_nm: Fraction
    precision: int
    x_over_width: I
    restricted_cubic_coefficients: tuple
    displacement_at_cut_m: I
    slope_at_cut: I
    curvature_at_cut_per_m: I
    third_derivative_per_m2: I
    moment_per_unit_strike_n: I
    shear_per_unit_strike_n_per_m: I
    toe_cut_force_per_unit_strike_n_per_m: I
    toe_cut_couple_per_unit_strike_n: I
    retained_cut_force_per_unit_strike_n_per_m: I
    retained_cut_couple_per_unit_strike_n: I
    toe_energy_fraction: I
    retained_energy_fraction: I
    whole_energy_per_unit_strike_j_per_m: I
    toe_energy_per_unit_strike_j_per_m: I
    retained_energy_per_unit_strike_j_per_m: I
    toe_boundary_power_per_tip_rate_n_per_m: I
    retained_boundary_power_per_tip_rate_n_per_m: I
    moving_cut_configurational_power_per_cut_rate_n_per_m: I
    source_sha256: str
    cut_sha256: str

    @property
    def source_preload_reset(self):
        return False

    @property
    def new_free_tip_boundary_condition_imposed(self):
        return False

    @property
    def moving_cut_work_authorized(self):
        return False

    @property
    def physical_pair_admission_authorized(self):
        return False

    @property
    def candidate_mechanics_adopted(self):
        return False


_CUT_SOURCE_NAMES = ("width_m", "cut_distance_m", "tip_displacement_m",
                     "rigidity_nm", "precision")


def certify_native_beam_cut_history(width_m, cut_distance_m,
        tip_displacement_m, rigidity_nm, *, precision=256):
    """Restrict the old cubic at a fixed cut, with no new clamp or free tip.

    The polynomial coefficients are in powers of ``xi = x-cut_distance``.
    Interval cuts describe a static family of restrictions, not a physical
    path of a moving cut.  Moment is ``-D*w''`` and shear is ``D*w'''`` per
    unit strike.  Their opposite boundary forces/couples preserve the two
    pieces' fixed-cut energy balance.  The fractions ``(a/L)**3`` and its
    complement remain defined even when the actual energy is zero.
    """
    width = _positive(width_m, "Original beam width")
    distance = _coordinate(cut_distance_m)
    tip = _interval(tip_displacement_m, "Original beam tip displacement")
    rigidity = _positive(rigidity_nm, "Original beam flexural rigidity")
    precision = _precision(precision)
    if not 0 <= distance.lo <= distance.hi <= width:
        raise ValueError("Complete cut enclosure must lie within the unchanged source beam")
    z = distance/width
    phi, slope_shape, curvature_shape, third_shape = _shape(z, precision, exact=True)
    displacement = phi*tip
    slope = slope_shape*tip/width
    curvature = curvature_shape*tip/(width**2)
    third = third_shape*tip/(width**3)
    coefficients = (displacement, slope, curvature/2, tip/(2*width**3))
    moment, shear = -rigidity*curvature, rigidity*third
    toe_ratio = I(z.lo**3, z.hi**3)
    retained_ratio = 1-toe_ratio
    whole_energy = Fraction(3, 2)*rigidity*_square(tip)/(width**3)
    toe_power = 3*rigidity*tip*toe_ratio/(width**3)
    retained_power = 3*rigidity*tip*retained_ratio/(width**3)
    # Diagnostic coefficient for a subsequently supplied moving-cut path.
    # Its presence does not authorize a cut speed, transfer or work increment.
    configurational_power = shear*slope-rigidity*_square(curvature)/2
    result = NativeBeamCutHistory(width, cut_distance_m, tip, rigidity, precision,
        z, coefficients, displacement, slope, curvature, third, moment, shear,
        -shear, -moment, shear, moment, toe_ratio, retained_ratio, whole_energy,
        whole_energy*toe_ratio, whole_energy*retained_ratio, toe_power,
        retained_power, configurational_power, "", "")
    result = replace(result, source_sha256=_hash(_cut_source_body(result)))
    return replace(result, cut_sha256=_hash(_cut_body(result)))


def _rational(value):
    if type(value) is not Fraction:
        raise ValueError("Beam serialization requires literal Fractions")
    return [value.numerator, value.denominator]


def _interval_body(value):
    value = _interval(value, "Beam serialization value")
    return dict(lo=_rational(value.lo), hi=_rational(value.hi))


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _hash(value):
    return hashlib.sha256(_encoded(value)).hexdigest()


def _source_body(chart):
    fractional = all(type(value) is Fraction for value in chart.reference_coordinates)
    return dict(trace_id=chart.trace_id, bank=chart.bank,
        start=[_rational(value) for value in chart.start],
        end=[_rational(value) for value in chart.end],
        reference_coordinates_kind="fraction" if fractional else "interval",
        reference_coordinates=[_rational(value) if fractional else _interval_body(value)
                               for value in chart.reference_coordinates],
        radius_m=_rational(chart.radius_m), width_m=_rational(chart.width_m),
        tip_displacement_m=_interval_body(chart.tip_displacement_m),
        precision=chart.precision)


_CHART_INTERVAL_NAMES = ("start_projection", "end_projection", "inward_projection",
    "in_plane_norm", "along_strike_angle_rad", "inward_angle_rad", "inward_distance_m",
    "x_over_width", "shape", "shape_slope", "shape_curvature", "shape_third_derivative",
    "displacement_m", "slope", "curvature_per_m", "third_derivative_per_m2")


def _chart_body(chart):
    body = dict(format=FORMAT, model=MODEL, scope=SCOPE, source=_source_body(chart),
        source_sha256=chart.source_sha256, inward=[_rational(value) for value in chart.inward],
        material_labels_normalized=False, native_bank_support_certified=False,
        physical_pair_admission_authorized=False, candidate_mechanics_adopted=False,
        source_shape_is_continuum_interior_contact_solution=False,
        accepted_native_force_intervals=0, accepted_work_increment_j=[0, 1])
    body.update({name: _interval_body(getattr(chart, name)) for name in _CHART_INTERVAL_NAMES})
    return body


def _cut_source_body(cut):
    fractional = type(cut.cut_distance_m) is Fraction
    return dict(width_m=_rational(cut.width_m),
        cut_distance_kind="fraction" if fractional else "interval",
        cut_distance_m=_rational(cut.cut_distance_m) if fractional else _interval_body(cut.cut_distance_m),
        tip_displacement_m=_interval_body(cut.tip_displacement_m),
        rigidity_nm=_rational(cut.rigidity_nm), precision=cut.precision)


_CUT_INTERVAL_NAMES = ("x_over_width", "displacement_at_cut_m", "slope_at_cut",
    "curvature_at_cut_per_m", "third_derivative_per_m2", "moment_per_unit_strike_n",
    "shear_per_unit_strike_n_per_m", "toe_cut_force_per_unit_strike_n_per_m",
    "toe_cut_couple_per_unit_strike_n", "retained_cut_force_per_unit_strike_n_per_m",
    "retained_cut_couple_per_unit_strike_n", "toe_energy_fraction", "retained_energy_fraction",
    "whole_energy_per_unit_strike_j_per_m", "toe_energy_per_unit_strike_j_per_m",
    "retained_energy_per_unit_strike_j_per_m", "toe_boundary_power_per_tip_rate_n_per_m",
    "retained_boundary_power_per_tip_rate_n_per_m",
    "moving_cut_configurational_power_per_cut_rate_n_per_m")


def _cut_body(cut):
    body = dict(format=CUT_FORMAT, model=CUT_MODEL, scope=CUT_SCOPE,
        source=_cut_source_body(cut), source_sha256=cut.source_sha256,
        restricted_cubic_coefficients=[_interval_body(value) for value in cut.restricted_cubic_coefficients],
        source_preload_reset=False, new_free_tip_boundary_condition_imposed=False,
        moving_cut_work_authorized=False, physical_pair_admission_authorized=False,
        candidate_mechanics_adopted=False, accepted_native_force_intervals=0,
        accepted_work_increment_j=[0, 1])
    body.update({name: _interval_body(getattr(cut, name)) for name in _CUT_INTERVAL_NAMES})
    return body


def _typed_fingerprint(value):
    kind = type(value)
    if kind in (str, int, bool, type(None)):
        return kind, value
    if kind is Fraction:
        return kind, value.numerator, value.denominator
    if kind is tuple:
        return kind, tuple(_typed_fingerprint(item) for item in value)
    if kind in (I, NativeBeamMaterialChart, NativeBeamCutHistory):
        return kind, tuple((field.name, _typed_fingerprint(getattr(value, field.name)))
                           for field in fields(value))
    raise ValueError("Beam replay requires immutable literal typed fields")


def validate_native_beam_material_chart(chart):
    if type(chart) is not NativeBeamMaterialChart:
        raise ValueError("Expected immutable source beam material chart")
    parameters = {name: getattr(chart, name) for name in _SOURCE_NAMES}
    replay = certify_native_beam_material_chart(**parameters)
    if _typed_fingerprint(chart) != _typed_fingerprint(replay):
        raise ValueError("Beam chart differs from complete exact source replay")
    return chart


def validate_native_beam_cut_history(cut):
    if type(cut) is not NativeBeamCutHistory:
        raise ValueError("Expected immutable source beam cut history")
    parameters = {name: getattr(cut, name) for name in _CUT_SOURCE_NAMES}
    replay = certify_native_beam_cut_history(**parameters)
    if _typed_fingerprint(cut) != _typed_fingerprint(replay):
        raise ValueError("Beam cut history differs from complete exact source replay")
    return cut


def native_beam_material_chart_to_dict(chart):
    chart = validate_native_beam_material_chart(chart)
    return dict(_chart_body(chart), chart_sha256=chart.chart_sha256)


def native_beam_cut_history_to_dict(cut):
    cut = validate_native_beam_cut_history(cut)
    return dict(_cut_body(cut), cut_sha256=cut.cut_sha256)


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
    raise ValueError("Beam DTO requires literal canonical JSON fields, with no floats or tuple aliases")


def native_beam_material_chart_from_dict(raw, *, expected_source):
    """Replay all fields against the caller's complete authenticated source."""
    if (type(raw) is not dict or type(expected_source) is not dict
            or set(expected_source) != set(_SOURCE_NAMES)):
        raise ValueError("Beam chart replay requires complete authoritative source parameters")
    _literal_json(raw)
    chart = certify_native_beam_material_chart(**expected_source)
    if _encoded(raw) != _encoded(dict(_chart_body(chart), chart_sha256=chart.chart_sha256)):
        raise ValueError("Saved beam chart differs from authoritative exact replay")
    return chart


def native_beam_cut_history_from_dict(raw, *, expected_source):
    """Replay the unchanged cubic and its cut work against full source data."""
    if (type(raw) is not dict or type(expected_source) is not dict
            or set(expected_source) != set(_CUT_SOURCE_NAMES)):
        raise ValueError("Beam cut replay requires complete authoritative source parameters")
    _literal_json(raw)
    cut = certify_native_beam_cut_history(**expected_source)
    if _encoded(raw) != _encoded(dict(_cut_body(cut), cut_sha256=cut.cut_sha256)):
        raise ValueError("Saved beam cut history differs from authoritative exact replay")
    return cut
