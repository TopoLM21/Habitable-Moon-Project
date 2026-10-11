"""A declared global, entry-triggered three-dimensional material carrier.

This NEW paired reference puts receiving columns above their shared radius
and donor columns below it.  It does not recover the former roof, contact
face, prestress or preparation work.  Areal-volume depth is literal:
``r0**3 = R**3 - 3*R**2*z``.  A smooth positive-gap history drives additional
down-dip settling without differentiating a singular first-entry root.
No force law, new accepted interval or all-pair admission is supplied here.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
from fractions import Fraction as F
from functools import lru_cache
from math import factorial
import hashlib
import json

from .geometric_native_pose import (
    FractionInterval as I, interval_sqrt, _rounded_interval,
    _precision_bits, _floor_log_two, _power_two, _alternating_rotation_terms,
    certify_rotation_history,
)
from .geometric_intrinsic_geometry import (
    RotationIncrement, rotation_increment_to_dict, rotation_increment_from_dict,
)
from .geometric_native_donor_sweep import _normal_coefficients
from .geometric_native_first_entry_domains import _axis_sources
from .geometric_native_first_entry_material import (
    NativeFirstEntryMaterialState, NativeFirstEntryMaterialPointChronology,
    validate_native_first_entry_material, native_first_entry_material_to_dict,
    _typed_fingerprint, certify_native_first_entry_material_point,
    validate_native_first_entry_material_point, evaluate_native_first_entry_material_point,
)
from .thermal import SECONDS_PER_MYR

FORMAT = "native-archive-physical-material-carrier-1"
RESPONSE_FORMAT = "native-archive-physical-material-point-1"
SCOPE = "DECLARED_GLOBAL_PAIRED_THREE_DIMENSIONAL_ARCHIVE_CARRIER_ONLY"
MODEL = "AREAL_VOLUME_UPPER_RECEIVER_AND_SMOOTH_CUBIC_GAP_DOWN_DIP_DONOR"
_FLAGS = dict(
    whole_three_nonempty_archive_cohort_placement_defined=True,
    full_actual_areal_volume_depth_retained=True,
    donor_live_archive_global_injectivity_certified=True,
    paired_receiver_archive_interior_separation_certified=True,
    new_receiver_reference_lift_explicitly_declared=True,
    archive_history_not_an_endpoint_polygon=True,
    first_entry_root_midpoint_used=False,
    former_receiver_roof_or_prestress_recovered=False,
    former_source_forces_or_work_handed_over=False,
    original_full_fault_face_gluing_recovered=False,
    cold_modes_globally_attached_to_new_reference=False,
    hot_constitutive_law_defined=False,
    hot_point_membership_authenticated=False,
    whole_clock_all_pair_admission_authorized=False,
    physical_material_admission_authorized=False,
    new_mixed_force_equilibrium_certified=False,
    original_point_1_completed=False,
    source_history_advanced=False,
    accepted_new_native_intervals=0,
)
_REGISTRY = {}
_CHRONOLOGY_CACHE = {}


def _payload(value):
    if type(value) is F:
        return [value.numerator, value.denominator]
    if type(value) is I:
        return dict(lo=_payload(value.lo), hi=_payload(value.hi))
    if is_dataclass(value):
        return {f.name: _payload(getattr(value, f.name)) for f in fields(value)}
    if type(value) in (tuple, list):
        return [_payload(v) for v in value]
    if type(value) is dict:
        return {k: _payload(v) for k, v in value.items()}
    return value


def _encoded(value):
    return json.dumps(_payload(value), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def _digest(value):
    return hashlib.sha256(_encoded(value)).hexdigest()


def _require_json(value):
    """A restart DTO must be decoded JSON, not a JSON-equal typed alias."""
    if type(value) is dict:
        if any(type(k) is not str for k in value):
            raise ValueError("Carrier JSON object keys must be literal strings")
        for member in value.values():
            _require_json(member)
    elif type(value) is list:
        for member in value:
            _require_json(member)
    elif type(value) not in (str, int, bool, float, type(None)):
        raise ValueError("Carrier restart requires exact decoded JSON kinds")


def _precision(value):
    if type(value) is not int or not 64 <= value <= 512:
        raise ValueError("Carrier precision must be an exact integer from64 to512")
    return value


def _exact(value, name):
    if type(value) is not F:
        raise ValueError(name + " must be an exact Fraction")
    return value


def _vector(value, *, intervals=False):
    kind = I if intervals else F
    if type(value) is not tuple or len(value) != 3 or any(type(v) is not kind for v in value):
        raise ValueError("Carrier vectors must have exact immutable three-component types")
    return value


def _interval(value):
    if (type(value) is not I or type(value.lo) is not F or type(value.hi) is not F
            or value.lo > value.hi):
        raise ValueError("Carrier intervals require canonical exact Fraction endpoints")
    return value


def _ray(value):
    if (type(value) is not tuple or len(value) != 3
            or any(type(v) not in (F, int) for v in value)):
        raise ValueError("A material ray requires an exact immutable triple")
    result = tuple(F(v) for v in value)
    if not any(result):
        raise ValueError("A material ray must be nonzero")
    return result


def _time(value):
    _exact(value, "Journal time fraction")
    if not 0 <= value <= 1:
        raise ValueError("Carrier time must belong to the complete declared journal interval")
    return value


def _dot(a, b):
    return sum((x*y for x, y in zip(a, b)), I(0))


def _cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def _mv(a, b):
    return tuple(_dot(row, b) for row in a)


def _transpose(a):
    return tuple(zip(*a))


def _mm(a, b):
    return tuple(tuple(_dot(row, col) for col in zip(*b)) for row in a)


def _identity():
    return tuple(tuple(I(int(i == j)) for j in range(3)) for i in range(3))


def _round(value, precision):
    return _rounded_interval(value, precision+8)


def _normalize(ray, precision):
    norm = interval_sqrt(I(sum(v*v for v in ray)), precision+16)
    return tuple(_round(I(v)/norm, precision) for v in ray)


def interval_cbrt(value, precision=96):
    """Monotone positive cube root on a relative integer grid, without floats."""
    _precision(precision)
    _interval(value)
    if type(value) is not I or value.lo < 0:
        raise ValueError("Carrier cube roots require a nonnegative exact interval")
    def point(x):
        if not x:
            return I(0)
        shift = _precision_bits(precision+8)-_floor_log_two(x)//3
        scale = _power_two(shift)
        scaled = x*scale*scale*scale
        target = scaled.numerator//scaled.denominator
        low, high = 0, 1 << ((target.bit_length()+2)//3+1)
        while high-low > 1:
            mid = (low+high)//2
            if mid*mid*mid <= target:
                low = mid
            else:
                high = mid
        if F(low*low*low) == scaled:
            return I(F(low)/scale)
        return I(F(low)/scale, F(low+1)/scale)
    return I(point(value.lo).lo, point(value.hi).hi)


def interval_exp(value, precision=96):
    """Outward rational exponential for the explicitly bounded carrier strain."""
    _precision(precision)
    _interval(value)
    if type(value) is not I or max(abs(value.lo), abs(value.hi)) > 1:
        raise ValueError("Carrier exponential needs the declared absolute argument bound<=1")
    def positive(x):
        if not x:
            return I(1)
        total = term = F(1)
        limit = F(1, 10**(precision+12))
        for n in range(1, 4097):
            term = term*x/n
            total += term
            following = term*x/(n+1)
            tail = following/(1-x/F(n+2))
            if tail <= limit:
                return _round(I(total, total+tail), precision)
        raise ValueError("Carrier exponential exhausted its exact series budget")
    def point(x):
        if x >= 0:
            return positive(x)
        result = 1/positive(-x)
        # Retain a strict upper bound even when the burial is much smaller
        # than the relative grid of a number close to one.
        strict = 1+x/2
        return I(result.lo, min(result.hi, strict))
    return I(point(value.lo).lo, point(value.hi).hi)


def _sin_cos(value, precision):
    if type(value) is not I or value.lo < 0 or value.hi > 2:
        raise ValueError("Carrier trigonometry requires a nonnegative small argument")
    low_cos, low_sinc = _alternating_rotation_terms(value.lo*value.lo, precision+12)
    high_cos, high_sinc = _alternating_rotation_terms(value.hi*value.hi, precision+12)
    cosine = I(high_cos.lo, low_cos.hi)
    sinc = I(high_sinc.lo, low_sinc.hi)
    return _round(value*sinc, precision), _round(cosine, precision)


@lru_cache(maxsize=128)
def _rotation(axis, time_fraction, precision):
    """Exact fractional-clock Rodrigues rotation of each saved full increment."""
    _vector(axis)
    _time(time_fraction)
    v = tuple(x*time_fraction for x in axis)
    square = sum(x*x for x in v)
    if square > F(1, 16):
        raise ValueError("Carrier rotation exceeds its declared full-increment budget")
    if not square:
        return _identity()
    cosine, sinc = _alternating_rotation_terms(square, precision+12)
    skew = ((F(0), -v[2], v[1]), (v[2], F(0), -v[0]), (-v[1], v[0], F(0)))
    squared = _mm(skew, skew)
    return tuple(tuple(_round(I(int(i == j))+sinc*skew[i][j]
                          +(1-cosine)*squared[i][j]/square, precision)
                       for j in range(3)) for i in range(3))


def areal_volume_radius(radius_m, depth_bounds_m, *, upper_thickness_m=None, precision=96):
    _exact(radius_m, "Source radius")
    _interval(depth_bounds_m)
    if type(depth_bounds_m) is not I or depth_bounds_m.lo < 0:
        raise ValueError("Areal-volume depth needs a nonnegative exact interval")
    cubic = I(radius_m**3)-3*radius_m**2*depth_bounds_m
    if upper_thickness_m is not None:
        _exact(upper_thickness_m, "Actual upper material thickness")
        cubic += 3*radius_m**2*upper_thickness_m
    if radius_m <= 0 or cubic.lo <= 0:
        raise ValueError("The complete areal-volume layer must have positive inner radius")
    return interval_cbrt(cubic, precision)


def _poly_bound(coefficients, domain, precision):
    result = I(0)
    for coefficient in reversed(coefficients):
        result = _round(result*domain+coefficient, precision)
    return result


def _poly_mul(a, b, precision):
    result = [I(0)]*(len(a)+len(b)-1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            result[i+j] = _round(result[i+j]+x*y, precision)
    return tuple(result)


def _poly_integral(a, lo, hi, precision):
    # Saved root endpoints may have thousands of digits. Exact endpoint**73
    # creates huge transient denominators although the final enclosure is
    # small. Outward interval Horner evaluates the SAME antiderivative with
    # a denominator bound after every operation; no root midpoint is used.
    primitive = tuple(_round(coefficient/F(k+1), precision)
                      for k, coefficient in enumerate(a))
    def endpoint(x):
        result = I(0)
        for coefficient in reversed(primitive):
            result = _round(result*I(x)+coefficient, precision)
        return _round(result*I(x), precision)
    return _round(endpoint(hi)-endpoint(lo), precision)


@lru_cache(maxsize=32)
def _history_coefficients(donor_axis, receiver_axis, order, precision):
    exact = _normal_coefficients((F(-1), F(0), F(0)), donor_axis, receiver_axis, order)
    # Rounded COEFFICIENT INTERVALS, not unacknowledged Taylor centres.
    coefficients = tuple(tuple(_round(I(v), precision) for v in row) for row in exact)
    gamma = sum(map(abs, donor_axis), F(0))+sum(map(abs, receiver_axis), F(0))
    tail = gamma**(order+1)/F(factorial(order+1))/(1-gamma/F(order+2))
    return coefficients, gamma, tail


@dataclass(frozen=True, slots=True)
class NativeArchiveHistoryActivation:
    beta_bounds: I
    gradient_bounds: tuple
    beta_rate_per_fraction_bounds: I
    gap_bounds: I
    gamma: F
    order: int
    cells: int


@lru_cache(maxsize=256)
def _activation(ray, donor_axis, receiver_axis, time_fraction, order, cells, precision, time_breakpoints):
    q = _normalize(ray, precision)
    coefficients, gamma, tail = _history_coefficients(donor_axis, receiver_axis, order, precision)
    gcoeff = tuple(_round(_dot(row, q), precision) for row in coefficients)
    scalar_tail = tail*3
    current = _poly_bound(gcoeff, I(time_fraction), precision)+I(-scalar_tail, scalar_tail)
    if not gamma or not time_fraction:
        return NativeArchiveHistoryActivation(I(0), (I(0),)*3, I(0), current, gamma, order, cells)
    # Whole donor pole/no-entry zones use a complete analytic interval bound,
    # so no latitude basis is constructed there.
    whole = _poly_bound(gcoeff, I(0, time_fraction), precision)+I(-scalar_tail, scalar_tail)
    if whole.hi <= 0:
        return NativeArchiveHistoryActivation(I(0), (I(0),)*3, I(0), current, gamma, order, cells)
    square = _poly_mul(gcoeff, gcoeff, precision)
    cube = _poly_mul(square, gcoeff, precision)
    gradient_poly = tuple(_poly_mul(square, tuple(row[j] for row in coefficients), precision)
                          for j in range(3))
    integral, gradient = I(0), [I(0)]*3
    boundaries = sorted(set((F(0), time_fraction)+tuple(time_fraction*F(i, cells)
                        for i in range(1, cells))+tuple(x for x in time_breakpoints
                        if 0 < x < time_fraction)))
    for lo, hi in zip(boundaries, boundaries[1:]):
        domain = I(lo, hi)
        raw = _poly_bound(gcoeff, domain, precision)
        gap = raw+I(-scalar_tail, scalar_tail)
        if gap.hi <= 0:
            continue
        maxgap = max(F(0), gap.hi)
        normals = tuple(_poly_bound(tuple(row[j] for row in coefficients), domain, precision)
                        +I(-tail, tail) for j in range(3))
        if gap.lo >= 0:
            rawmax = max(abs(raw.lo), abs(raw.hi))
            cube_error = 3*rawmax**2*scalar_tail+3*rawmax*scalar_tail**2+scalar_tail**3
            amount = _poly_integral(cube, lo, hi, precision)+I(-cube_error, cube_error)*(hi-lo)
            amount = I(max(F(0), amount.lo), max(F(0), amount.hi))
            integral = _round(integral+amount, precision)
            # Component normal tails and scalar-cube coefficient intervals
            # are separately enclosed; interval polynomial integration is exact.
            for j in range(3):
                normalmax = max(abs(normals[j].lo), abs(normals[j].hi))
                error = ((2*rawmax*scalar_tail+scalar_tail**2)*normalmax
                         +rawmax**2*tail)
                term = _poly_integral(gradient_poly[j], lo, hi, precision)+I(-error, error)*(hi-lo)
                gradient[j] = _round(gradient[j]+term, precision)
        else:
            integral = _round(integral+I(0, maxgap**3)*(hi-lo), precision)
            for j in range(3):
                gradient[j] = _round(gradient[j]+I(0, maxgap**2)*normals[j]*(hi-lo), precision)
    beta = _round(integral/(6*gamma*gamma), precision)
    grad = tuple(_round(x/(2*gamma*gamma), precision) for x in gradient)
    active = I(max(F(0), current.lo), max(F(0), current.hi))
    rate = _round(active*active*active/(6*gamma*gamma), precision)
    return NativeArchiveHistoryActivation(beta, grad, rate, current, gamma, order, cells)


def smooth_activation_history(original_ray, donor_axis, receiver_axis, time_fraction,
                              *, order=24, cells=32, precision=96, time_breakpoints=()):
    """Pure declared-law primitive; it grants no material-domain authority."""
    ray = _ray(original_ray)
    _vector(donor_axis); _vector(receiver_axis); _time(time_fraction); _precision(precision)
    if type(order) is not int or not 16 <= order <= 40:
        raise ValueError("Carrier history order must be an exact integer from16 to40")
    if type(cells) is not int or not 4 <= cells <= 256:
        raise ValueError("Carrier history cells must be an exact integer from4 to256")
    gamma = sum(map(abs, donor_axis), F(0))+sum(map(abs, receiver_axis), F(0))
    if gamma >= F(1, 4):
        raise ValueError("The complete carrier history exceeds its finite small-motion budget")
    if (type(time_breakpoints) is not tuple or any(type(x) is not F or not 0 <= x <= 1
                                                 for x in time_breakpoints)):
        raise ValueError("History breakpoints must be exact fractions on the full journal clock")
    return _activation(ray, donor_axis, receiver_axis, time_fraction, order, cells, precision, time_breakpoints)


@dataclass(frozen=True, slots=True)
class NativeArchiveWholeDomainCertificate:
    radius_m: F
    gamma: F
    common_dip_radians: F
    dip_tangent_bounds: I
    maximum_beta_bounds: I
    minimum_latitude_derivative: F
    minimum_volume_jacobian_bounds: I
    original_parent_count: int
    nonempty_cohort_count: int
    cohort_original_parent_ids: tuple
    complete_archive_domain_ids: tuple
    original_parent_thicknesses_m: tuple
    swept_only_curved_domain_retained: bool
    within_owner_parent_corner_gluing_certified: bool
    donor_radius_upper_bound_m: F
    receiving_base_radius_m: F
    declared_upper_hot_shell_support_separated: bool
    physical_all_pair_admission: bool
    constitutive_and_force_handover: bool


@dataclass(frozen=True, slots=True)
class NativeArchiveMaterialCarrier:
    source_material_state_sha256: str
    source_material_state_json: str
    donor_owner: int
    receiver_owner: int
    donor_axis: tuple
    receiver_axis: tuple
    owner_history_scope: str
    complete_ordered_owner_histories_json: str
    duration_myr: F
    duration_s: F
    radius_m: F
    common_dip_radians: F
    dip_tangent_bounds: I
    complete_original_trace_bindings_json: str
    whole_domain_certificate: NativeArchiveWholeDomainCertificate
    provenance: str
    precision: int
    history_order: int
    history_cells: int
    model_sha256: str


def _model_body(value):
    return {f.name: getattr(value, f.name) for f in fields(value) if f.name != "model_sha256"}


def _registered(value):
    if type(value) is not NativeArchiveMaterialCarrier:
        raise ValueError("A physical carrier needs the exact registered typed model")
    entry = _REGISTRY.get(value.model_sha256)
    if (entry is None or _typed_fingerprint(value) != entry[0]
            or _digest(_model_body(value)) != value.model_sha256):
        raise ValueError("Altered, aliased or resealed model lacks physical carrier authority")
    state = entry[1]
    if _typed_fingerprint(state) != entry[2]:
        raise ValueError("The complete typed material dependency changed after carrier binding")
    return state


def bind_native_archive_material_carrier(source_material_state, *, provenance,
                                        precision=96, history_order=24, history_cells=32):
    """Declare a new global paired reference on the complete authentic state."""
    if type(source_material_state) is not NativeFirstEntryMaterialState:
        raise ValueError("Carrier binding requires the authentic typed original material state")
    state = validate_native_first_entry_material(source_material_state)
    _precision(precision)
    if type(provenance) is not str or not provenance.strip():
        raise ValueError("New paired physical reference requires explicit model provenance")
    dt = F.from_float(state.journal.dt_myr)
    motions = dict(state.journal.omega_rad_per_myr)
    donor, receiver = state.sweep_certificate.donor_owner, state.sweep_certificate.receiver_owner
    # This theorem supports the exact actual single increment starting from
    # eight UNPOSED original axis octants. Full historical source poses or
    # additional differential transport require a new complete material law.
    _axis_sources(state.journal, donor, receiver)
    histories = [[owner, [dict(omega_rad_per_myr=list(motions[owner]),
                              dt_myr=state.journal.dt_myr)]] for owner in (donor, receiver)]
    dax = tuple(F.from_float(x)*dt for x in motions[donor])
    rax = tuple(F.from_float(x)*dt for x in motions[receiver])
    # Validate every budget even before a first point query.
    smooth_activation_history((F(1), F(0), F(0)), dax, rax, F(0),
                              order=history_order, cells=history_cells, precision=precision)
    gamma = sum(map(abs, dax), F(0))+sum(map(abs, rax), F(0))
    original = json.loads(state.original_mechanics_payload_json)
    traces = original["fault_traces"]["traces"]
    dips = tuple(F.from_float(t["dip_radians"]) for t in traces)
    if len(dips) != 4 or len(set(dips)) != 1:
        raise ValueError("The global carrier needs one authenticated common dip across all original traces")
    sine, cosine = _sin_cos(I(dips[0]), precision)
    if cosine.lo <= 0 or sine.lo <= 0:
        raise ValueError("The complete literal dip must have positive sine and cosine")
    tangent = sine/cosine
    if tangent.hi*gamma > F(1, 4):
        raise ValueError("The whole reference settling strain exceeds its declared bound")
    # Gamma<1/4 gives 7Gamma/6<7/24<pi/2 with no rounded angular threshold.
    if gamma >= F(1, 4):
        raise ValueError("The full donor activated belt is not proved pole-free")
    radius = F.from_float(state.journal.source_surface.radius_km)*1000
    parents = state.journal.source_surface.fragments
    heights = tuple((p.fragment_id, sum(map(F.from_float, p.parcel.specific_properties[:2]))*1000)
                    for p in parents)
    if (len(parents) != 8 or len(state.archive_cohorts) != 3
            or any(h <= 0 or 3*h >= radius for _, h in heights)):
        raise ValueError("Carrier requires all eight actual full positive areal-volume parents and three cohorts")
    receiver_heights = {dict(heights)[p.fragment_id] for p in parents if p.parcel.plate == receiver}
    if len(receiver_heights) != 1:
        raise ValueError("Within-owner upper original corner gluing needs the actual common receiving thickness")
    cohorts = tuple(sorted(state.archive_cohorts, key=lambda c: c.original_parent_id))
    if (tuple(c.original_parent_id for c in cohorts) != ("octant:5", "octant:6", "octant:7")
            or any((c.owner, c.receiver_owner) != (donor, receiver) for c in cohorts)):
        raise ValueError("The authentic three whole donor cohorts changed")
    # A swept-only ray can return outside the receiver: active latitude can
    # be as low as -2Gamma, whereas its positive upper bound is7Gamma/6.
    _, lowercos = _sin_cos(I(2*gamma), precision)
    jac = interval_exp(I(-tangent.hi*gamma/2), precision)*lowercos/2
    if jac.lo <= 0:
        raise ValueError("No positive global whole-volume Jacobian lower bound")
    certificate = NativeArchiveWholeDomainCertificate(
        radius, gamma, dips[0], tangent, I(0, gamma/6), F(1, 2), jac,
        len(parents), len(cohorts), tuple(c.original_parent_id for c in cohorts),
        tuple(c.archive_domain_id for c in cohorts), heights,
        state.sweep_certificate.extra_swept_area_bounds_sr.lo > 0, True,
        radius, radius, True, False, False)
    body = NativeArchiveMaterialCarrier(state.state_sha256,
        _encoded(native_first_entry_material_to_dict(state)).decode(), donor, receiver,
        dax, rax, "SINGLE_AUTHENTIC_JOURNAL_INCREMENT_FROM_UNPOSED_AXIS_OCTANTS",
        _encoded(histories).decode(), dt, dt*F.from_float(SECONDS_PER_MYR), radius, dips[0], tangent,
        _encoded(dict(trace_bindings=state.trace_bindings, original_traces=traces)).decode(),
        certificate, provenance, precision, history_order, history_cells, "")
    result = replace(body, model_sha256=_digest(_model_body(body)))
    _REGISTRY[result.model_sha256] = (_typed_fingerprint(result), state, _typed_fingerprint(state))
    return result


native_archive_material_carrier_bind = bind_native_archive_material_carrier


@dataclass(frozen=True, slots=True)
class NativeArchiveMaterialPoint:
    original_parent_id: str
    original_ray: tuple
    material_depth_bounds_m: I
    time_fraction: F
    precision: int
    model_sha256: str
    material_id: str
    owner: int
    placement_kind: str
    allocation: str
    contact_occupancy: str
    cohort_id: str | None
    archive_debits: int
    first_entry_time_fraction: I | None
    first_entry_receiver_fragment_ids: tuple
    beta_bounds: I
    beta_original_q_gradient_bounds: tuple
    beta_rate_per_fraction_bounds: I
    history_time_breakpoints: tuple
    position_bounds_m: tuple
    velocity_bounds_m_per_s: tuple
    deformation_gradient_bounds: tuple
    volume_jacobian_bounds: I
    radial_support_bounds_m: I
    reference_radius_bounds_m: I
    density_pullback_multiplier_bounds: I
    areal_material_thickness_m: F
    journal_time_myr: F
    world_history_json: str
    world_translation_m: tuple
    response_sha256: str


def _response_body(value):
    return {f.name: getattr(value, f.name) for f in fields(value) if f.name != "response_sha256"}


def _owner_geometry(system, q, time, precision):
    rd, rr = _rotation(system.donor_axis, time, precision), _rotation(system.receiver_axis, time, precision)
    relative = _mm(_transpose(rr), rd)
    u = _mv(relative, q)
    # d(Qr^T Qd q)/dt, t the normalized original journal clock.
    d = _mv(_transpose(rr), _cross(system.donor_axis, _mv(rd, q)))
    spin = _mv(_transpose(rr), _cross(system.receiver_axis, _mv(rd, q)))
    return rd, rr, relative, u, tuple(a-b for a, b in zip(d, spin))


def _donor_geometry(system, ray, q, depth, time, precision, time_breakpoints):
    history = smooth_activation_history(ray, system.donor_axis, system.receiver_axis, time,
        order=system.history_order, cells=system.history_cells, precision=precision,
        time_breakpoints=time_breakpoints)
    rd, rr, relative, u, ut = _owner_geometry(system, q, time, precision)
    r0 = areal_volume_radius(system.radius_m, depth, precision=precision)
    if history.beta_bounds.hi == 0:
        x = tuple(r0*v for v in _mv(rd, q))
        v = tuple(z/system.duration_s for z in _cross(system.donor_axis, x))
        return history, x, v, rd, I(1), r0, r0
    beta, grad = history.beta_bounds, history.gradient_bounds
    gap = -u[0]
    h = interval_sqrt(1-gap*gap, precision+16)
    if h.lo <= 0:
        raise ValueError("An active material point escaped the proven pole-free belt")
    exminus = (I(-1), I(0), I(0))
    e = tuple((m-gap*a)/h for m, a in zip(exminus, u))
    sine, cosine = _sin_cos(beta, precision)
    nnew = tuple(cosine*a+sine*b for a, b in zip(u, e))
    enew = tuple(-sine*a+cosine*b for a, b in zip(u, e))
    k = system.dip_tangent_bounds
    contraction = interval_exp(-k*beta, precision)
    radius = r0*contraction
    xlocal = tuple(radius*v for v in nnew)
    bt = history.beta_rate_per_fraction_bounds
    gt = -ut[0]
    et = tuple((-gt*a-gap*da)/h+b*gap*gt/(h*h) for a, da, b in zip(u, ut, e))
    nt = tuple(cosine*da+sine*db+bt*en for da, db, en in zip(ut, et, enew))
    xt = tuple(radius*(dn-k*bt*n) for dn, n in zip(nt, nnew))
    x = _mv(rr, xlocal)
    spin = _cross(system.receiver_axis, x)
    v = tuple((a+b)/system.duration_s for a, b in zip(_mv(rr, xt), spin))
    # Differentiate directly in original Cartesian X=r0*q.  No tau(q)
    # derivative and no representative interval midpoint enters this F.
    columns = []
    for j in range(3):
        dq = tuple((I(int(i == j))-q[i]*q[j])/r0 for i in range(3))
        du = _mv(relative, dq)
        dg = -du[0]
        db = _dot(grad, dq)
        de = tuple((-dg*a-gap*da)/h+b*gap*dg/(h*h) for a, da, b in zip(u, du, e))
        dn = tuple(cosine*da+sine*dea+db*en for da, dea, en in zip(du, de, enew))
        dr = contraction*(q[j]-r0*k*db)
        dx = tuple(dr*n+radius*delta for n, delta in zip(nnew, dn))
        columns.append(_mv(rr, dx))
    gradient = tuple(tuple(columns[j][i] for j in range(3)) for i in range(3))
    source_e = _mv(_transpose(relative), e)
    beta_mu = _dot(grad, source_e)
    angle_ratio = (h*cosine-gap*sine)/h
    jac = interval_exp(-3*k*beta, precision)*angle_ratio*(1+beta_mu)
    # Tighten with the independent global theorem when rectangular gradient
    # intervals lose correlations; never change the actual map or F.
    bound = system.whole_domain_certificate.minimum_volume_jacobian_bounds.lo
    if jac.hi < bound:
        raise ValueError("Point Jacobian enclosure contradicts the global injectivity proof")
    jac = I(max(bound, jac.lo), jac.hi)
    return history, tuple(_round(v, precision) for v in x), tuple(_round(v, precision) for v in v), \
        tuple(tuple(_round(v, precision) for v in row) for row in gradient), \
        _round(jac, precision), radius, r0


def _receiving_geometry(system, q, depth, height, time, precision):
    pose = _rotation(system.receiver_axis, time, precision)
    r0 = areal_volume_radius(system.radius_m, depth, precision=precision)
    radius = areal_volume_radius(system.radius_m, depth, upper_thickness_m=height, precision=precision)
    radial = r0*r0/(radius*radius)
    tangential = radius/r0
    local = tuple(tuple(tangential*I(int(i == j))+(radial-tangential)*q[i]*q[j]
                        for j in range(3)) for i in range(3))
    x = tuple(radius*v for v in _mv(pose, q))
    v = tuple(y/system.duration_s for y in _cross(system.receiver_axis, x))
    return x, v, _mm(pose, local), I(1), radius, r0


def _chronology(state, identity, ray, supplied):
    key = state.state_sha256, identity, ray
    if supplied is None:
        if key not in _CHRONOLOGY_CACHE:
            _CHRONOLOGY_CACHE[key] = certify_native_first_entry_material_point(state, identity, ray)
        return _CHRONOLOGY_CACHE[key]
    if type(supplied) is not NativeFirstEntryMaterialPointChronology:
        raise ValueError("Point chronology must have its authentic typed source")
    if (supplied.state is not state and _typed_fingerprint(supplied.state) != _typed_fingerprint(state)):
        raise ValueError("Point chronology belongs to another complete material state")
    if supplied.original_parent_id != identity or supplied.original_point != ray:
        raise ValueError("Point chronology was transplanted onto another material ray")
    checked = validate_native_first_entry_material_point(supplied)
    _CHRONOLOGY_CACHE[key] = checked
    return checked


def evaluate_native_archive_material_point(system, original_parent_id, original_ray,
        material_depth_bounds_m, time_fraction, *, precision=96, chronology=None,
        world_history=(), world_translation_m=(F(0), F(0), F(0))):
    state = _registered(system)
    _precision(precision); _time(time_fraction)
    _interval(material_depth_bounds_m)
    ray = _ray(original_ray)
    if type(original_parent_id) is not str:
        raise ValueError("Original parent identity must be a literal string")
    parent = next((p for p in state.journal.source_surface.fragments if p.fragment_id == original_parent_id), None)
    if parent is None:
        raise ValueError("Carrier point has no genuine original parent")
    height = dict(system.whole_domain_certificate.original_parent_thicknesses_m)[original_parent_id]
    if (type(material_depth_bounds_m) is not I or material_depth_bounds_m.lo < 0
            or material_depth_bounds_m.hi > height):
        raise ValueError("The complete exact depth interval must lie in its actual whole material layer")
    if (type(world_history) is not tuple or any(type(s) is not RotationIncrement for s in world_history)):
        raise ValueError("Observer objectivity uses a complete typed ordered rigid history")
    for step in world_history:
        if (type(step.omega_rad_per_myr) is not tuple or len(step.omega_rad_per_myr) != 3
                or any(type(x) is not float for x in step.omega_rad_per_myr)
                or type(step.dt_myr) is not float):
            raise ValueError("An observer increment has an altered literal saved scalar/vector kind")
    _vector(world_translation_m)
    point = _chronology(state, original_parent_id, ray, chronology)
    allocation = evaluate_native_first_entry_material_point(point, time_fraction)
    q = _normalize(ray, precision)
    first = point.first_entry_certificate
    breakpoints = (() if first is None else tuple(sorted(set(
        x for event in first.plane_chronology.events for x in
        (event.time_fraction.lo, event.time_fraction.hi)))))
    if parent.parcel.plate == system.donor_owner:
        history, x, v, gradient, jac, radius, r0 = _donor_geometry(
            system, ray, q, material_depth_bounds_m, time_fraction, precision, breakpoints)
        kind = "donor_inward_down_dip_history"
    elif parent.parcel.plate == system.receiver_owner:
        x, v, gradient, jac, radius, r0 = _receiving_geometry(
            system, q, material_depth_bounds_m, height, time_fraction, precision)
        history = NativeArchiveHistoryActivation(I(0), (I(0),)*3, I(0), I(0),
            system.whole_domain_certificate.gamma, system.history_order, system.history_cells)
        kind = "new_lifted_receiving_reference"
    else:
        raise ValueError("An original material owner is outside the declared paired model")
    observer = certify_rotation_history(world_history, precision).intervals
    x = tuple(_round(a+b, precision) for a, b in zip(_mv(observer, x), world_translation_m))
    v = tuple(_round(z, precision) for z in _mv(observer, v))
    gradient = tuple(tuple(_round(z, precision) for z in row) for row in _mm(observer, gradient))
    query = NativeArchiveMaterialPoint(original_parent_id, ray, material_depth_bounds_m,
        time_fraction, precision, system.model_sha256, parent.parcel.material_id,
        parent.parcel.plate, kind, allocation.material_allocation, allocation.contact_occupancy,
        allocation.cohort_id, allocation.archive_debits, allocation.first_entry_time_fraction,
        allocation.receiver_original_fragment_ids, history.beta_bounds, history.gradient_bounds,
        history.beta_rate_per_fraction_bounds, breakpoints, x, v, gradient, jac, radius, r0,
        _round(1/jac, precision), height,
        F.from_float(state.journal.source_surface.time_myr)+system.duration_myr*time_fraction,
        _encoded([rotation_increment_to_dict(s) for s in world_history]).decode(), world_translation_m, "")
    return replace(query, response_sha256=_digest(_response_body(query)))


def evaluate_native_archive_hot_point(system, component_id, original_ray,
                                     material_depth_bounds_m, *, precision=96):
    """Fail closed until an authentic implicit birth-point oracle is supplied.

    Upper-layer support/volume mathematics is declared globally. A point's
    HOT material membership and complete birth trajectory are separate facts.
    """
    _registered(system)
    raise ValueError("HOT point membership/birth chronology is not authenticated by this carrier component")


def native_archive_material_carrier_to_dict(system):
    _registered(system)
    return dict(format=FORMAT, scope=SCOPE, model=MODEL, **_FLAGS, carrier=_payload(system))


def native_archive_material_carrier_from_dict(raw, *, expected_source_material_state):
    _require_json(raw)
    if type(raw) is not dict or raw.get("format") != FORMAT:
        raise ValueError("Invalid physical archive carrier schema")
    try:
        value = raw["carrier"]
        result = bind_native_archive_material_carrier(expected_source_material_state,
            provenance=value["provenance"], precision=value["precision"],
            history_order=value["history_order"], history_cells=value["history_cells"])
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("Incomplete archive carrier restart fields") from error
    if _encoded(raw) != _encoded(native_archive_material_carrier_to_dict(result)):
        raise ValueError("Saved archive carrier differs from full typed external source/model replay")
    return result


def native_archive_material_point_to_dict(response, *, system):
    if type(response) is not NativeArchiveMaterialPoint:
        raise ValueError("Expected an authentic typed physical material-point response")
    if response.response_sha256 != _digest(_response_body(response)):
        raise ValueError("Physical material-point response changed")
    replay = evaluate_native_archive_material_point(system, response.original_parent_id,
        response.original_ray, response.material_depth_bounds_m, response.time_fraction,
        precision=response.precision,
        world_history=tuple(rotation_increment_from_dict(x) for x in json.loads(response.world_history_json)),
        world_translation_m=response.world_translation_m)
    if _typed_fingerprint(replay) != _typed_fingerprint(response):
        raise ValueError("Physical material point differs from its complete typed query replay")
    return dict(format=RESPONSE_FORMAT, scope=SCOPE, **_FLAGS, response=_payload(response))


def native_archive_material_point_from_dict(raw, *, system, expected_original_parent_id,
        expected_original_ray, expected_depth, expected_time_fraction, precision=96,
        world_history=(), world_translation_m=(F(0), F(0), F(0))):
    _require_json(raw)
    if type(raw) is not dict or raw.get("format") != RESPONSE_FORMAT:
        raise ValueError("Invalid physical archive point schema")
    replay = evaluate_native_archive_material_point(system, expected_original_parent_id,
        expected_original_ray, expected_depth, expected_time_fraction, precision=precision,
        world_history=world_history, world_translation_m=world_translation_m)
    if _encoded(raw) != _encoded(native_archive_material_point_to_dict(replay, system=system)):
        raise ValueError("A physical point DTO was transplanted or differs from full expected query")
    return replay


native_archive_material_carrier_response_to_dict = native_archive_material_point_to_dict
native_archive_material_carrier_response_from_dict = native_archive_material_point_from_dict
