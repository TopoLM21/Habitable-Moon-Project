"""Constitutive diagnostics on declared native contact-time windows.

This module does not grant geometric permission or adopt a material history.
It replays the original endpoint return map and declares an affine path of the
free material anchors.  At time fraction f the diagnostic return map starts
from the unchanged old plastic field and uses duration f*dt, not dt.

For an inviscid law whose preparation is already in its yield domain, a
strictly monotone branch is a continuous dense extension of that update.  A
viscous fractional backward-Euler family is not a solution of the continuous
viscoplastic ODE; its signs are diagnostics, never physical admission here.
No work, clock, preload, plastic memory or geometry is modified.

Material queries use the retained computational angle of the original
continuous fields.  For an authenticated axis-quarter trace it is related to
the native geometric angle by theta_computational = theta_native *
retained_extent/(exact pi/2).  Thus the true native endpoint maps to the
original saved endpoint, without silently extending terminal plastic pieces.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
import math
from numbers import Integral, Real

from .geometric_native_pose import FractionInterval as I, interval_pi
from .geometric_spatial_underthrust import (
    SpatialUnderthrustStep, TrigPolynomial, _anchors,
    advance_spatial_underthrust_memory, initialize_spatial_underthrust_memory,
)


MODEL = "affine_free_anchors_fractional_backward_euler"
ADMISSIBLE = "inviscid_contact_time_constitutive_admissible"
_ZERO = (Fraction(0),)*5


def _fraction(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an explicit finite number")
    if type(value) is Fraction:
        return value
    if isinstance(value, Integral):
        return Fraction(int(value))
    if isinstance(value, Real) and math.isfinite(float(value)):
        return Fraction.from_float(float(value))
    raise ValueError(f"{name} must be an explicit finite number")


def _field(value):
    return tuple(Fraction.from_float(x) for x in value.coefficients)


def _add(a, b):
    return tuple(x+y for x, y in zip(a, b))


def _scale(a, factor):
    return tuple(x*factor for x in a)


def _constant(value):
    return (value, Fraction(0), Fraction(0), Fraction(0), Fraction(0))


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class NativeContactTimePath:
    initial_mechanics: object
    step: SpatialUnderthrustStep
    previous_anchor_fields: tuple
    prior_steps: tuple[SpatialUnderthrustStep, ...]
    path_model: str
    path_sha256: str


@dataclass(frozen=True, slots=True)
class NativeContactTimeSupport:
    trace_id: str
    material_angle_bounds: tuple[Fraction, Fraction]
    time_fraction_bounds: tuple[Fraction, Fraction]
    pressure_bounds_pa: I
    down_dip_yield_excess_bounds_pa: I
    plastic_increment_bounds_m: I | None
    plastic_rate_bounds_m_s: I | None
    status: str
    path_sha256: str
    precision: int
    path_model: str = MODEL

    @property
    def admissible(self):
        return self.status == ADMISSIBLE


def build_native_contact_time_path(initial_mechanics, step, previous_anchor_fields,
                                   *, prior_steps=(), path_model=MODEL):
    """Authenticate every retained endpoint; retain the original preparation.

    A noninitial previous field needs its complete accepted spatial replay.
    Nodal checks alone cannot authenticate five polynomial coefficients from
    four Gauss nodes, so they are deliberately not used as such a substitute.
    """
    if path_model != MODEL:
        raise ValueError("An explicit supported contact-time path model is required")
    if type(step) is not SpatialUnderthrustStep or type(prior_steps) is not tuple:
        raise ValueError("Contact-time support requires immutable spatial endpoint replay")
    seed = initial_mechanics
    prepared = tuple((t.trace_id, TrigPolynomial((t.initial_gap_m, 0., 0., 0., 0.)),
        TrigPolynomial((t.initial_jump_m, 0., 0., 0., 0.)))
        for t in seed.fault_traces.traces)
    memory, states, fields = initialize_spatial_underthrust_memory(seed), seed.states, prepared
    for index, accepted in enumerate((*prior_steps, step)):
        if (type(accepted) is not SpatialUnderthrustStep
                or accepted.previous_memory != memory
                or accepted.previous_point_states != states):
            raise ValueError("Contact-time support cannot skip or reset accepted spatial history")
        reproduced = advance_spatial_underthrust_memory(seed, memory,
            accepted.anchor_fields, accepted.dt_s, previous_point_states=states,
            current_point_states=accepted.current_point_states)
        if reproduced != accepted:
            raise ValueError("Contact-time endpoint differs from original constitutive replay")
        if index == len(prior_steps):
            break
        memory, states, fields = accepted.memory, accepted.current_point_states, accepted.anchor_fields
    supplied = _anchors(seed, previous_anchor_fields)
    if supplied != fields:
        raise ValueError("Previous contact-time anchors differ from accepted original fields")
    body = dict(initial_mechanics=asdict(seed), step=asdict(step),
        previous_anchor_fields=[(identity, gap.coefficients, jump.coefficients)
                                for identity, gap, jump in supplied],
        prior_steps=[asdict(item) for item in prior_steps], path_model=path_model)
    return NativeContactTimePath(seed, step, supplied, prior_steps, path_model, _digest(body))


def _validate_path(path):
    if type(path) is not NativeContactTimePath:
        raise ValueError("A replay-authenticated contact-time path is required")
    repeated = build_native_contact_time_path(path.initial_mechanics, path.step,
        path.previous_anchor_fields, prior_steps=path.prior_steps, path_model=path.path_model)
    if repeated != path:
        raise ValueError("Contact-time path binding changed")


@lru_cache(maxsize=2048)
def _sin_cos_point(value, precision):
    """Exact alternating Taylor bounds, including tiny represented angles."""
    threshold = Fraction(1, 10**precision)

    def series(sine):
        term = value if sine else Fraction(1)
        partial, index = term, 0
        while True:
            n = 2*index+(2 if sine else 1)
            following = -term*value*value/(n*(n+1))
            # Beyond this index the alternating tail decreases monotonically.
            if value*value <= (n+2)*(n+3) and abs(following) <= threshold:
                return I(min(partial, partial+following), max(partial, partial+following))
            partial, term, index = partial+following, following, index+1
    return series(True), series(False)


def _trig_ranges(low, high, precision):
    ends = [_sin_cos_point(value, precision) for value in (low, high)]
    sine = I(min(pair[0].lo for pair in ends), max(pair[0].hi for pair in ends))
    cosine = I(min(pair[1].lo for pair in ends), max(pair[1].hi for pair in ends))
    pi = interval_pi(precision+8)
    # Inputs need only cover degree-two fields on a positive minor trace.
    for multiple, value in ((Fraction(1, 2), 1), (Fraction(3, 2), -1)):
        point = pi*multiple
        if low <= point.hi and high >= point.lo:
            sine = I(min(sine.lo, value), max(sine.hi, value))
    for multiple, value in ((0, 1), (1, -1), (2, 1)):
        point = pi*multiple
        if low <= point.hi and high >= point.lo:
            cosine = I(min(cosine.lo, value), max(cosine.hi, value))
    return sine, cosine


def _bounds(field, low, high, precision):
    sine, cosine = _trig_ranges(low, high, precision)
    double_sine, double_cosine = _trig_ranges(2*low, 2*high, precision)
    return (I(field[0])+cosine*field[1]+sine*field[2]
            +double_cosine*field[3]+double_sine*field[4])


def _affine_bounds(a, delta, low, high, first_time, last_time, precision):
    ends = [_bounds(_add(a, _scale(delta, time)), low, high, precision)
            for time in (first_time, last_time)]
    return I(min(value.lo for value in ends), max(value.hi for value in ends))


def native_quarter_material_coordinate_bounds(path, trace_id, native_angle_bounds,
                                              *, precision=80):
    """Map an already geometrically authenticated quarter-arc query.

    This is a coordinate conversion, not a geometry certificate.  The caller
    must prove that its actual native arc lies in this original carrier.  A
    coarse angle enclosure is intersected with that known carrier domain.
    The exact native endpoint maps to the exact retained binary64 extent.
    """
    _validate_path(path)
    if type(native_angle_bounds) is not I:
        raise ValueError("Native quarter coordinates require an exact rational enclosure")
    if type(precision) is not int or precision < 16:
        raise ValueError("Contact-time precision must be an integer of at least 16 digits")
    trace = next((t for t in path.initial_mechanics.fault_traces.traces
                  if t.trace_id == trace_id), None)
    if trace is None:
        raise ValueError("Unknown original physical trace")
    start = tuple(Fraction.from_float(x) for x in trace.start)
    end = tuple(Fraction.from_float(x) for x in trace.end)
    if (sum(x*y for x, y in zip(start, end)) != 0
            or sum(x*x for x in start) != 1 or sum(x*x for x in end) != 1
            or any(sum(value != 0 for value in point) != 1 for point in (start, end))):
        raise ValueError("Coordinate conversion only certifies original unit axis-quarter traces")
    extent = next(c.extent_radians for c in path.step.certificates if c.trace_id == trace_id)
    half_pi = interval_pi(precision+8)/2
    if native_angle_bounds.hi < 0 or native_angle_bounds.lo > half_pi.hi:
        raise ValueError("Native material angle enclosure misses its declared quarter carrier")
    mapped = native_angle_bounds*(Fraction.from_float(extent)/half_pi)
    return I(max(Fraction(0), mapped.lo), min(Fraction.from_float(extent), mapped.hi))


def assess_native_contact_time_support(path, trace_id, material_angle_lo,
        material_angle_hi, time_fraction_lo, time_fraction_hi, *, precision=80):
    """Bound all constitutive fields on the whole material/time rectangle.

    Only a strictly compressed, strictly down-dip inviscid rate can be
    admissible here.  Initial/branch zeros are retained algebraically; no sign,
    duration or material-angle floor is introduced.  A viscous result is
    explicitly a diagnostic of the declared fractional BE family.
    """
    _validate_path(path)
    if type(precision) is not int or precision < 16:
        raise ValueError("Contact-time precision must be an integer of at least 16 digits")
    low, high = (_fraction(v, "Original material angle")
                 for v in (material_angle_lo, material_angle_hi))
    first, last = (_fraction(v, "Contact time fraction")
                   for v in (time_fraction_lo, time_fraction_hi))
    if not 0 <= first <= last <= 1:
        raise ValueError("Contact-time windows must lie in the entire declared interval")
    seed, step = path.initial_mechanics, path.step
    position = next((index for index, trace in enumerate(seed.fault_traces.traces)
                     if trace.trace_id == trace_id), None)
    if position is None:
        raise ValueError("Unknown original physical trace")
    trace, previous = seed.fault_traces.traces[position], step.previous_memory.traces[position]
    if not 0 <= low < high <= Fraction.from_float(previous.extent_radians):
        raise ValueError("Contact-time queries must preserve a positive original material interval")
    _, gap0, jump0 = path.previous_anchor_fields[position]
    _, gap1, jump1 = step.anchor_fields[position]
    g0, j0, g1, j1 = map(_field, (gap0, jump0, gap1, jump1))
    dg, dj = _add(g1, _scale(g0, -1)), _add(j1, _scale(j0, -1))
    # Lift the unchanged endpoint implementation's effective constitutive
    # constants into exact dyadic arithmetic for these diagnostic predicates.
    kn = Fraction.from_float(1./(1./trace.columns.normal_stiffness_pa_m
                                 +1./trace.law.normal_stiffness_pa_m))
    kt = Fraction.from_float(1./(1./trace.columns.shear_stiffness_pa_m
                                 +1./trace.law.tangential_stiffness_pa_m))
    mu = Fraction.from_float(trace.law.friction_dry
        +(trace.law.friction_wet-trace.law.friction_dry)*trace.water)
    cohesion = Fraction.from_float(trace.law.cohesion_pa
        *(1.-(1.-trace.law.wet_cohesion_fraction)*trace.water))
    eta, duration = Fraction.from_float(trace.law.viscosity_pa_s_m), Fraction.from_float(step.dt_s)
    pressure = _affine_bounds(_scale(g0, -kn), _scale(dg, -kn),
        low, high, first, last, precision)
    rows = []
    for old in previous.pieces:
        left = max(low, Fraction.from_float(old.low_radians))
        right = min(high, Fraction.from_float(old.high_radians))
        if left >= right:
            continue
        slip = _field(old.plastic_slip)
        # E = -trial - strength, positive for down-dip yielding.
        e0 = _add(_scale(_add(j0, _scale(slip, -1)), -kt),
                  _add(_scale(g0, kn*mu), _constant(-cohesion)))
        e1 = _add(_scale(dj, -kt), _scale(dg, kn*mu))
        excess = _affine_bounds(e0, e1, left, right, first, last, precision)
        initial_excess = _bounds(e0, left, right, precision)
        increment = rate = None
        if pressure.lo <= 0:
            status = "unresolved_strict_compressive_support"
        elif not eta and initial_excess.hi > 0:
            status = "unresolved_inviscid_initial_relaxation"
        elif excess.hi < 0 or (excess.hi == 0 and last != 0):
            status = "no_down_dip_plastic_time_support"
        elif excess.lo < 0:
            status = "unresolved_constitutive_time_branch"
        elif eta and first == 0:
            status = "unresolved_viscous_contact_time_limit"
        else:
            if not eta:
                increment = -excess/kt
                rate = -_bounds(e1, left, right, precision)/(kt*duration)
            else:
                time = I(first, last)
                denominator = kt*duration*time+eta
                increment = -(time*duration*excess)/denominator
                numerator_field = _bounds(e0, left, right, precision)*eta
                e1_bounds = _bounds(e1, left, right, precision)
                numerator = (numerator_field+2*eta*time*e1_bounds
                             +kt*duration*time*time*e1_bounds)
                rate = -numerator/(denominator*denominator)
            if rate.hi >= 0:
                status = "no_strict_down_dip_plastic_rate"
            elif eta:
                status = "viscous_fractional_backward_euler_diagnostic_only"
            else:
                status = ADMISSIBLE
        rows.append((excess, increment, rate, status))
    if not rows:
        raise ValueError("Original material interval lacks preserved plastic memory support")
    excess = I(min(row[0].lo for row in rows), max(row[0].hi for row in rows))
    def hull(index):
        values = [row[index] for row in rows]
        return None if any(value is None for value in values) else I(
            min(value.lo for value in values), max(value.hi for value in values))
    status = next((row[3] for row in rows if row[3] != ADMISSIBLE), ADMISSIBLE)
    return NativeContactTimeSupport(trace_id, (low, high), (first, last), pressure,
        excess, hull(1), hull(2), status, path.path_sha256, precision)


def native_contact_time_support_to_dict(support):
    """Portable exact bounds; this receipt remains a constitutive diagnostic."""
    if type(support) is not NativeContactTimeSupport:
        raise ValueError("Expected native contact-time constitutive support")
    def scalar(value):
        return [value.numerator, value.denominator]
    def bounds(value):
        return None if value is None else dict(lo=scalar(value.lo), hi=scalar(value.hi))
    return dict(model=support.path_model, status=support.status,
        admissible=support.admissible, trace_id=support.trace_id,
        path_sha256=support.path_sha256, precision=support.precision,
        material_angle_bounds=[scalar(x) for x in support.material_angle_bounds],
        time_fraction_bounds=[scalar(x) for x in support.time_fraction_bounds],
        pressure_bounds_pa=bounds(support.pressure_bounds_pa),
        down_dip_yield_excess_bounds_pa=bounds(support.down_dip_yield_excess_bounds_pa),
        plastic_increment_bounds_m=bounds(support.plastic_increment_bounds_m),
        plastic_rate_bounds_m_s=bounds(support.plastic_rate_bounds_m_s),
        scope="declared_constitutive_time_path_only_no_geometry_or_memory_adoption")
