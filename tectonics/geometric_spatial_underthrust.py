"""Continuous constitutive permission for actual geometric underthrust losses.

The fields are material-coordinate trigonometric polynomials, not an
interpolation of quadrature stresses.  Series-column compliance eliminates
the local elastic jumps exactly.  Yield crossings split the plastic memory
into polynomial pieces; the original mechanical nodes authenticate every
accepted update.  This module grants direction only, never a removed volume.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
from numbers import Integral, Real

import numpy as np
from scipy.optimize import brentq

from .geometric_boundary import surface_digest
from .geometric_contact_mechanics import LocalContactState, validate_contact_state
from .geometric_contacts import extract_contacts
from .geometric_fault_traces import PhysicalFaultTrace, _arc_frame, _interval
from .geometric_polarity import (
    PolarityEvidence, PolarityResolution, contact_polarity_report,
    rotate_points,
)
from .geometric_surface import rotations
from .spherical_polygons import _edge_cross, _shared_minor_arc, arc_length
from .thermal import SECONDS_PER_MYR


FORMAT = "geometric-continuous-spatial-underthrust-1"
MODEL = "original_material_coordinate_piecewise_series_column_return_map"
_EPS = np.finfo(float).eps


def _real(value, name, *, positive=False, nonnegative=False):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not math.isfinite(float(value))):
        raise ValueError(f"{name} must be an explicit finite real number")
    result = float(value)
    if (positive and result <= 0.) or (nonnegative and result < 0.):
        raise ValueError(f"Invalid {name}")
    return result


def _count(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class TrigPolynomial:
    """constant, cos(theta), sin(theta), cos(2 theta), sin(2 theta)."""
    coefficients: tuple[float, float, float, float, float]

    def __post_init__(self):
        if not isinstance(self.coefficients, (tuple, list)) or len(self.coefficients) != 5:
            raise ValueError("A degree-two trigonometric field requires five coefficients")
        object.__setattr__(self, "coefficients", tuple(
            _real(value, "Trigonometric coefficient") for value in self.coefficients))

    def __call__(self, theta):
        theta = _real(theta, "Material angle")
        a, b, c, d, e = self.coefficients
        return math.fsum((a, b*math.cos(theta), c*math.sin(theta),
                          d*math.cos(2.*theta), e*math.sin(2.*theta)))

    @property
    def rounding_bound(self):
        return 512.*_EPS*math.fsum(abs(value) for value in self.coefficients)

    def __add__(self, other):
        other = _field(other)
        return TrigPolynomial(tuple(math.fsum(pair) for pair in zip(
            self.coefficients, other.coefficients)))

    def __sub__(self, other):
        return self+_field(other).scaled(-1.)

    def scaled(self, factor):
        factor = _real(factor, "Field factor")
        return TrigPolynomial(tuple(factor*value for value in self.coefficients))


def _field(value):
    return value if isinstance(value, TrigPolynomial) else TrigPolynomial(value)


def _constant(value):
    return TrigPolynomial((_real(value, "Constant field"), 0., 0., 0., 0.))


def _derivative(field):
    _, b, c, d, e = _field(field).coefficients
    return TrigPolynomial((0., c, -b, 2.*e, -2.*d))


def _candidate_roots(field, low, high):
    """Quartic half-angle roots, filtered by the original trig expression.

    Near-real multiple roots are retained when their real part satisfies the
    arithmetic residual bound.  They cannot create an undetected sign pocket:
    all extrema are subsequently included in the monotone root subdivision.
    """
    field = _field(field)
    a, b, c, d, e = field.coefficients
    polynomial = np.asarray((a+b+d, 2.*c+4.*e, 2.*a-6.*d,
                             2.*c-4.*e, a-b+d), dtype=float)
    nonzero = np.flatnonzero(polynomial)
    if not len(nonzero):
        return ()
    polynomial = polynomial[:nonzero[-1]+1]
    scale = float(np.max(np.abs(polynomial)))
    roots = np.polynomial.polynomial.polyroots(polynomial/scale)
    result = []
    for root in roots:
        if root.real < 0.:
            continue
        angle = 2.*math.atan(float(root.real))
        if not low < angle < high:
            continue
        if (abs(root.imag) <= 256.*_EPS*max(1., abs(root.real))
                or abs(field(angle)) <= field.rounding_bound):
            result.append(angle)
    return tuple(sorted(result))


def trig_polynomial_extrema(field, low_radians, high_radians):
    """Evaluate all endpoints and stationary roots on a positive minor arc."""
    field = _field(field)
    low = _real(low_radians, "Interval start", nonnegative=True)
    high = _real(high_radians, "Interval end", positive=True)
    if not low < high < math.pi:
        raise ValueError("Continuous fields require 0 <= low < high < pi")
    positions = (low, high, *_candidate_roots(_derivative(field), low, high))
    values = tuple(field(position) for position in positions)
    return min(values), max(values)


def trig_polynomial_roots(field, low_radians, high_radians):
    """All crossings/touches from quartic critical points and monotone brackets."""
    field = _field(field)
    low = _real(low_radians, "Interval start", nonnegative=True)
    high = _real(high_radians, "Interval end", positive=True)
    if not low < high < math.pi:
        raise ValueError("Continuous fields require 0 <= low < high < pi")
    stationary = _candidate_roots(_derivative(field), low, high)
    knots = tuple(sorted((low, *stationary, high)))
    roots = list(_candidate_roots(field, low, high))
    for point in stationary:
        if abs(field(point)) <= field.rounding_bound:
            roots.append(point)
    for left, right in zip(knots, knots[1:]):
        first, last = field(left), field(right)
        if first*last < 0.:
            roots.append(brentq(field, left, right,
                xtol=np.nextafter(0., 1.), rtol=8.*_EPS))
    result = []
    tolerance = 256.*_EPS*max(high-low, np.finfo(float).tiny)
    for point in sorted(roots):
        if low+tolerance < point < high-tolerance and (
                not result or point-result[-1] > tolerance):
            if abs(field(point)) > 8.*field.rounding_bound:
                raise ValueError("Cannot certify a trigonometric crossing")
            result.append(point)
    return tuple(result)


@dataclass(frozen=True, slots=True)
class SpatialPlasticPiece:
    low_radians: float
    high_radians: float
    plastic_slip: TrigPolynomial
    cumulative_slip: TrigPolynomial


@dataclass(frozen=True, slots=True)
class SpatialTraceMemory:
    trace_id: str
    extent_radians: float
    pieces: tuple[SpatialPlasticPiece, ...]


@dataclass(frozen=True, slots=True)
class SpatialUnderthrustMemory:
    physical_trace_sha256: str
    traces: tuple[SpatialTraceMemory, ...]
    stage_count: int
    elapsed_time_s: float


@dataclass(frozen=True, slots=True)
class SpatialConstitutivePiece:
    low_radians: float
    high_radians: float
    pressure: TrigPolynomial
    shear_traction: TrigPolynomial
    strength: TrigPolynomial
    plastic_increment: TrigPolynomial
    branch: str


@dataclass(frozen=True, slots=True)
class SpatialTraceCertificate:
    trace_id: str
    extent_radians: float
    gap_free: TrigPolynomial
    jump_free: TrigPolynomial
    pieces: tuple[SpatialConstitutivePiece, ...]


@dataclass(frozen=True, slots=True)
class SpatialUnderthrustStep:
    previous_memory: SpatialUnderthrustMemory
    memory: SpatialUnderthrustMemory
    anchor_fields: tuple[tuple[str, TrigPolynomial, TrigPolynomial], ...]
    certificates: tuple[SpatialTraceCertificate, ...]
    dt_s: float
    node_manifest_sha256: str
    previous_point_states: tuple[tuple[str, LocalContactState], ...]
    current_point_states: tuple[tuple[str, LocalContactState], ...]


def _angles(seed):
    quadrature = seed.fault_traces.quadrature
    nodes, _ = np.polynomial.legendre.leggauss(quadrature.gauss_order)
    result = []
    for trace in seed.fault_traces.traces:
        extent = _arc_frame(trace.start, trace.end)[3]
        width = extent/quadrature.panel_count
        for panel in range(quadrature.panel_count):
            for index, node in enumerate(nodes):
                identity = f"{trace.trace_id}:panel:{panel}:gauss:{index}"
                result.append((identity, trace, (panel+.5)*width+.5*width*float(node)))
    return tuple(result)


def _states(seed, supplied):
    if not isinstance(supplied, tuple) or any(
            not isinstance(row, tuple) or len(row) != 2 for row in supplied):
        raise ValueError("Spatial authentication requires immutable original point histories")
    expected = tuple(identity for identity, _, _ in _angles(seed))
    if tuple(identity for identity, _ in supplied) != expected:
        raise ValueError("Spatial authentication cannot change original point identities")
    initial = dict(seed.states)
    for (identity, local), (_, trace, _) in zip(supplied, _angles(seed)):
        if not isinstance(local, LocalContactState):
            raise ValueError("Spatial authentication requires accepted local contact states")
        validate_contact_state(local, columns=trace.columns, law=trace.law,
            area_m2=initial[identity].area_m2, water=trace.water)
    return dict(supplied)


def initialize_spatial_underthrust_memory(initial_mechanics):
    """Retain the homogeneous physical preparation, including its plasticity."""
    seed = initial_mechanics
    states = _states(seed, seed.states)
    traces = []
    for trace in seed.fault_traces.traces:
        local = [states[identity] for identity, material, _ in _angles(seed)
                 if material.trace_id == trace.trace_id]
        retained = ("gap_free_m", "jump_free_m", "plastic_slip_m", "cumulative_slip_m")
        if not local or any(any(getattr(item, name) != getattr(local[0], name)
                for name in retained) for item in local[1:]):
            raise ValueError("Initial continuous memory requires homogeneous trace preparation")
        if (local[0].gap_free_m != trace.initial_gap_m
                or local[0].jump_free_m != trace.initial_jump_m):
            raise ValueError("Continuous memory cannot infer an already evolved plastic field")
        extent = _arc_frame(trace.start, trace.end)[3]
        piece = SpatialPlasticPiece(0., extent, _constant(local[0].plastic_slip_m),
                                    _constant(local[0].cumulative_slip_m))
        traces.append(SpatialTraceMemory(trace.trace_id, extent, (piece,)))
    return SpatialUnderthrustMemory(seed.fault_traces.trace_sha256, tuple(traces), 0, 0.)


def _validate_memory(seed, memory):
    if not isinstance(memory, SpatialUnderthrustMemory):
        raise ValueError("Explicit continuous plastic memory is required")
    if memory.physical_trace_sha256 != seed.fault_traces.trace_sha256:
        raise ValueError("Continuous memory physical trace binding changed")
    _count(memory.stage_count, "Continuous stage count")
    _real(memory.elapsed_time_s, "Continuous elapsed time", nonnegative=True)
    if (not isinstance(memory.traces, tuple) or tuple(item.trace_id for item in memory.traces)
            != tuple(trace.trace_id for trace in seed.fault_traces.traces)):
        raise ValueError("Continuous memory must retain every canonical original trace")
    for trace, stored in zip(seed.fault_traces.traces, memory.traces):
        extent = _arc_frame(trace.start, trace.end)[3]
        if not isinstance(stored, SpatialTraceMemory) or stored.extent_radians != extent:
            raise ValueError("Continuous material extent changed")
        if not isinstance(stored.pieces, tuple) or not stored.pieces:
            raise ValueError("Continuous memory requires immutable positive pieces")
        cursor = 0.
        for piece in stored.pieces:
            if (not isinstance(piece, SpatialPlasticPiece) or piece.low_radians != cursor
                    or not piece.low_radians < piece.high_radians <= extent):
                raise ValueError("Continuous plastic pieces must cover the trace exactly once")
            for field in (piece.plastic_slip, piece.cumulative_slip):
                if not isinstance(field, TrigPolynomial):
                    raise ValueError("Continuous plastic memory must contain polynomial fields")
            for field in (piece.cumulative_slip,
                    piece.cumulative_slip-piece.plastic_slip,
                    piece.cumulative_slip+piece.plastic_slip):
                if trig_polynomial_extrema(field, piece.low_radians, piece.high_radians)[0] < -field.rounding_bound:
                    raise ValueError("Continuous cumulative slip must bound signed plastic memory")
            cursor = piece.high_radians
        if cursor != extent:
            raise ValueError("Continuous plastic memory has missing trace support")
    if memory.stage_count == 0 and memory != initialize_spatial_underthrust_memory(seed):
        raise ValueError("Initial continuous plastic memory differs from preparation")
    return memory


def _piece_at(pieces, theta):
    for piece in pieces:
        if piece.low_radians <= theta <= piece.high_radians:
            return piece
    raise ValueError("Original quadrature point lacks continuous memory support")


def _close(actual, field, theta, name, *, operation_scale=0.):
    expected = field(theta)
    scale = max(abs(actual), abs(expected), math.fsum(abs(v) for v in field.coefficients),
                _real(operation_scale, "Constitutive operation scale", nonnegative=True),
                np.finfo(float).tiny)
    if abs(actual-expected) > 4096.*_EPS*scale:
        raise ValueError(f"Continuous field differs from accepted point {name}")


def _anchors(seed, supplied):
    ids = tuple(trace.trace_id for trace in seed.fault_traces.traces)
    if isinstance(supplied, Mapping):
        if set(supplied) != set(ids):
            raise ValueError("Continuous anchors require every original physical trace")
        supplied = tuple((identity, *supplied[identity]) for identity in ids)
    if (not isinstance(supplied, tuple) or tuple(row[0] for row in supplied) != ids
            or any(not isinstance(row, tuple) or len(row) != 3 for row in supplied)):
        raise ValueError("Continuous anchors require canonical trace/gap/jump records")
    return tuple((identity, _field(gap), _field(jump)) for identity, gap, jump in supplied)


def _constitutive_trace(trace, old, gap, jump, dt):
    if trig_polynomial_extrema(gap, 0., old.extent_radians)[1] > gap.rounding_bound:
        raise ValueError("Spatial underthrust requires continuously closed free normal support")
    kn = 1./(1./trace.columns.normal_stiffness_pa_m+1./trace.law.normal_stiffness_pa_m)
    kt = 1./(1./trace.columns.shear_stiffness_pa_m+1./trace.law.tangential_stiffness_pa_m)
    pressure = gap.scaled(-kn)
    mu = trace.law.friction_dry+(trace.law.friction_wet-trace.law.friction_dry)*trace.water
    cohesion = trace.law.cohesion_pa*(1.-(1.-trace.law.wet_cohesion_fraction)*trace.water)
    strength = pressure.scaled(mu)+_constant(cohesion)
    denominator = kt+trace.law.viscosity_pa_s_m/dt
    plastic, certificates = [], []
    for previous in old.pieces:
        trial = (jump-previous.plastic_slip).scaled(kt)
        positive, negative = trial-strength, trial.scaled(-1.)-strength
        boundaries = sorted({previous.low_radians, previous.high_radians,
            *trig_polynomial_roots(gap, previous.low_radians, previous.high_radians),
            *trig_polynomial_roots(positive, previous.low_radians, previous.high_radians),
            *trig_polynomial_roots(negative, previous.low_radians, previous.high_radians)})
        for low, high in zip(boundaries, boundaries[1:]):
            middle = .5*(low+high)
            if positive(middle) > positive.rounding_bound:
                increment, branch = positive.scaled(1./denominator), "up_dip_plastic"
            elif negative(middle) > negative.rounding_bound:
                increment, branch = negative.scaled(-1./denominator), "down_dip_plastic"
            else:
                increment, branch = _constant(0.), "locked"
            new_slip = previous.plastic_slip+increment
            absolute = increment.scaled(-1.) if branch == "down_dip_plastic" else increment
            plastic.append(SpatialPlasticPiece(low, high, new_slip,
                                              previous.cumulative_slip+absolute))
            certificates.append(SpatialConstitutivePiece(low, high, pressure,
                (jump-new_slip).scaled(kt), strength, increment, branch))
    return SpatialTraceMemory(trace.trace_id, old.extent_radians, tuple(plastic)), \
        SpatialTraceCertificate(trace.trace_id, old.extent_radians, gap, jump, tuple(certificates))


def advance_spatial_underthrust_memory(initial_mechanics, previous_memory, anchor_fields, dt_s, *,
                                      previous_point_states, current_point_states):
    """Authenticate a continuous constitutive update against every accepted node."""
    seed = initial_mechanics
    old = _validate_memory(seed, previous_memory)
    dt = _real(dt_s, "Spatial constitutive interval", positive=True)
    anchors = _anchors(seed, anchor_fields)
    previous = _states(seed, previous_point_states)
    current = _states(seed, current_point_states)
    old_by_trace = {item.trace_id: item for item in old.traces}
    memory, certificates = [], []
    for trace, (_, gap, jump), stored in zip(seed.fault_traces.traces, anchors, old.traces):
        updated, certificate = _constitutive_trace(trace, stored, gap, jump, dt)
        memory.append(updated)
        certificates.append(certificate)
    elapsed = math.fsum((old.elapsed_time_s, dt))
    if elapsed <= old.elapsed_time_s:
        raise ValueError("Continuous memory cannot resolve the physical interval")
    updated = SpatialUnderthrustMemory(old.physical_trace_sha256, tuple(memory), old.stage_count+1, elapsed)
    _validate_memory(seed, updated)
    new_by_trace = {item.trace_id: item for item in updated.traces}
    cert_by_trace = {item.trace_id: item for item in certificates}
    for identity, trace, theta in _angles(seed):
        before, after = previous[identity], current[identity]
        old_piece = _piece_at(old_by_trace[trace.trace_id].pieces, theta)
        new_piece = _piece_at(new_by_trace[trace.trace_id].pieces, theta)
        cert = cert_by_trace[trace.trace_id]
        piece = _piece_at(cert.pieces, theta)
        if (after.step_count != before.step_count+1
                or after.elapsed_time_s != math.fsum((before.elapsed_time_s, dt))):
            raise ValueError("Continuous and mechanical point clocks disagree")
        for actual, field, name in (
                (before.plastic_slip_m, old_piece.plastic_slip, "old plastic slip"),
                (before.cumulative_slip_m, old_piece.cumulative_slip, "old cumulative slip"),
                (after.plastic_slip_m, new_piece.plastic_slip, "new plastic slip"),
                (after.cumulative_slip_m, new_piece.cumulative_slip, "new cumulative slip"),
                (after.gap_free_m, cert.gap_free, "free normal anchor"),
                (after.jump_free_m, cert.jump_free, "free tangential anchor"),
                (-after.traction_normal_pa, piece.pressure, "normal pressure"),
                (after.traction_shear_pa, piece.shear_traction, "shear traction")):
            operation_scale = 0.
            if name == "shear traction":
                # Zero-strength, inviscid sliding subtracts equal elastic and
                # plastic jumps.  Bound the operations producing that small
                # traction, rather than the vanishing final traction alone.
                kt = 1./(1./trace.columns.shear_stiffness_pa_m
                         +1./trace.law.tangential_stiffness_pa_m)
                operation_scale = max(trace.law.tangential_stiffness_pa_m*(
                    abs(after.jump_m)+abs(after.plastic_slip_m)), kt*(
                    abs(cert.jump_free(theta))+abs(new_piece.plastic_slip(theta))))
            _close(actual, field, theta, name, operation_scale=operation_scale)
    manifest = _hash(dict(previous=[(identity, asdict(value)) for identity, value in previous_point_states],
        current=[(identity, asdict(value)) for identity, value in current_point_states]))
    return SpatialUnderthrustStep(old, updated, anchors, tuple(certificates), dt, manifest,
                                  previous_point_states, current_point_states)


def _normal_velocity_bounds(trace, omega, radius, low, high):
    origin, horizontal, tangent, _ = _arc_frame(trace.start, trace.end)
    difference = np.asarray(omega[trace.overriding_plate])-np.asarray(omega[trace.subducting_plate])
    vector = radius*np.cross(horizontal, difference)
    field = TrigPolynomial((0., float(origin@vector), float(tangent@vector), 0., 0.))
    return (*trig_polynomial_extrema(field, low, high), field.rounding_bound)


def _spatial_intersect_arcs(a0, a1, b0, b1):
    """Retain even near-endpoint positive arcs in saved Cartesian coordinates."""
    return _shared_minor_arc(a0, a1, b0, b1)


def _material_support_interval(carrier, start, end, extent):
    """Outward material-angle enclosure after positive geometric support proof.

    A physical 1e-20 arc near theta=pi/2 cannot be represented as two distinct
    absolute binary64 angles.  Only its constitutive field query is outward
    rounded; its true Cartesian geometry, length and mass remain unchanged.
    """
    values = [arc_length(carrier.start, point) for point in (start, end)]
    tolerance = 1024.*_EPS*extent
    if any(value > extent+tolerance for value in values):
        raise ValueError("Actual support exceeds its original material trace")
    low, high = max(0., min(values)), min(extent, max(values))
    if abs(high-extent) <= tolerance:
        high = extent
    if abs(low) <= tolerance and any(np.array_equal(point, carrier.start) for point in (start, end)):
        low = 0.
    if low >= high:
        angle = min(extent, max(0., .5*(low+high)))
        low = max(0., float(np.nextafter(angle, -math.inf)))
        high = min(extent, float(np.nextafter(angle, math.inf)))
    else:
        low = max(0., float(np.nextafter(low, -math.inf)))
        high = min(extent, float(np.nextafter(high, math.inf)))
    if not low < high:
        raise ValueError("Cannot enclose a positive actual material-support arc")
    return low, high


def _partial_velocity_bounds(carrier, omega, radius, start, end):
    """Signed velocity on the actual world arc, with a local zero-angle origin."""
    ld = np.longdouble
    a, b = np.asarray(start, dtype=ld), np.asarray(end, dtype=ld)
    horizontal = _edge_cross(np.asarray(carrier.start), np.asarray(carrier.end))
    horizontal /= np.sqrt(np.sum(horizontal*horizontal))
    difference = (np.asarray(omega[carrier.overriding_plate], dtype=ld)
                  -np.asarray(omega[carrier.subducting_plate], dtype=ld))
    vector = ld(radius)*np.cross(horizontal, difference)
    normal = _edge_cross(np.asarray(start), np.asarray(end))
    normal /= np.sqrt(np.sum(normal*normal))
    tangent = np.cross(normal, a)
    tangent /= np.sqrt(np.sum(tangent*tangent))
    length = arc_length(start, end)
    dot = lambda point: float(np.sum(vector*point))
    first, second = dot(a), dot(tangent)
    # Actual endpoint vectors avoid cos(pi/2)'s nonzero binary approximation.
    positions = [(first, a), (dot(b), b)]
    stationary = math.atan2(second, first)
    for index in range(-2, 3):
        position = stationary+index*math.pi
        if 0. < position < length:
            point = a*np.cos(ld(position))+tangent*np.sin(ld(position))
            positions.append((dot(point), point))
    tolerances = []
    for _, point in positions:
        # Componentwise cross/dot operations scale with the actual tiny
        # endpoint operands, not an unrelated whole-arc velocity amplitude.
        cross_terms = np.asarray((
            abs(horizontal[1]*difference[2])+abs(horizontal[2]*difference[1]),
            abs(horizontal[2]*difference[0])+abs(horizontal[0]*difference[2]),
            abs(horizontal[0]*difference[1])+abs(horizontal[1]*difference[0])))
        tolerances.append(float(512.*_EPS*ld(radius)*np.sum(cross_terms*np.abs(point))))
    return min(value for value, _ in positions), max(value for value, _ in positions), max(tolerances)


def _assess_interval(certificate, low, high):
    considered = []
    for piece in certificate.pieces:
        left, right = max(low, piece.low_radians), min(high, piece.high_radians)
        if not left < right:
            continue
        pressure = trig_polynomial_extrema(piece.pressure, left, right)
        increment = trig_polynomial_extrema(piece.plastic_increment, left, right)
        if pressure[0] < -piece.pressure.rounding_bound:
            return "effective_tension"
        if piece.branch != "down_dip_plastic" or increment[1] > piece.plastic_increment.rounding_bound:
            return "no_down_dip_plastic_sliding"
        considered.append((left, right, pressure, increment))
    if not considered or considered[0][0] != low or considered[-1][1] != high:
        return "missing_continuous_constitutive_support"
    if not any(item[2][1] > 0. and item[3][0] < 0. for item in considered):
        return "no_compressive_down_dip_sliding"
    return "spatial_underthrust_admissible"


def make_spatial_underthrust_resolver(surface, omega_rad_per_myr, dt_myr, *,
                                     initial_mechanics, carrier_traces, step, history=()):
    """Strict actual-contact permission from a certified continuous material field.

    Carrier trace poses are the images of the same original receiver-side
    physical arc.  Callers must authenticate those material maps before this
    gate.  Each true overlap is located against its surviving receiver edge;
    unknown or conflicting support never invokes a material-age fallback.
    """
    if surface.phase != "partition" or not isinstance(step, SpatialUnderthrustStep):
        raise ValueError("Spatial admission requires a resolved surface and a constitutive certificate")
    dt = _real(dt_myr, "Geometric interval", positive=True)
    if not math.isclose(dt*SECONDS_PER_MYR, step.dt_s, rel_tol=8.*_EPS, abs_tol=0.):
        raise ValueError("Spatial constitutive and geometric physical intervals disagree")
    omega = rotations(surface, omega_rad_per_myr, dt)
    seed = initial_mechanics
    _validate_memory(seed, step.previous_memory)
    _validate_memory(seed, step.memory)
    carriers = tuple(carrier_traces)
    if tuple(trace.trace_id for trace in carriers) != tuple(trace.trace_id for trace in seed.fault_traces.traces):
        raise ValueError("Carrier poses must retain every canonical original physical trace")
    for original, trace in zip(seed.fault_traces.traces, carriers):
        if not isinstance(trace, PhysicalFaultTrace) or any(getattr(original, name) != getattr(trace, name)
                for name in ("subducting_plate", "overriding_plate", "dip_radians", "columns", "law", "water", "pore_pressure_pa")):
            raise ValueError("Carrier physical material, orientation or constitutive law changed")
        if not math.isclose(_arc_frame(trace.start, trace.end)[3], _arc_frame(original.start, original.end)[3],
                            rel_tol=1024.*_EPS, abs_tol=0.):
            raise ValueError("Carrier pose changed its original material arc measure")
    if step.dt_s <= 0. or len(step.certificates) != len(carriers):
        raise ValueError("Incomplete continuous constitutive certificate")
    # Receipt checksums cannot replace constitutive replay and authentication
    # against all retained original mechanical nodes.
    reproduced = advance_spatial_underthrust_memory(seed, step.previous_memory,
        step.anchor_fields, step.dt_s, previous_point_states=step.previous_point_states,
        current_point_states=step.current_point_states)
    if reproduced != step:
        raise ValueError("Spatial constitutive certificate differs from material-field/node replay")
    contacts = extract_contacts(surface.fragments, surface.radius_km, omega)
    try:
        history = tuple(PolarityEvidence(**record) if isinstance(record, dict) else record for record in history)
    except TypeError as error:
        raise ValueError("Invalid explicit archived polarity record") from error
    previous = {row["contact_id"]: row for row in contact_polarity_report(contacts, history)}
    certificates = {item.trace_id: item for item in step.certificates}
    statuses, rows, evidence = {}, [], []
    for contact in contacts:
        supports = []
        for trace in carriers:
            clipped = _spatial_intersect_arcs(trace.start, trace.end, contact.start, contact.end)
            if clipped is None or {contact.plate_a, contact.plate_b} != {trace.subducting_plate, trace.overriding_plate}:
                continue
            low, high = _interval(trace.start, trace.end, *clipped)
            # The field uses the retained original material measure.  A
            # rotated endpoint atan2 must not change that measure by one ULP.
            # Snap only endpoint arithmetic, never rescale an interior angle.
            material_extent = certificates[trace.trace_id].extent_radians
            angle_tolerance = 1024.*_EPS*material_extent
            if abs(low) <= angle_tolerance:
                low = 0.
            if abs(high-material_extent) <= angle_tolerance:
                high = material_extent
            if low < 0. or high > material_extent:
                raise ValueError("Current physical arc exceeds retained material-coordinate support")
            if not low < high:
                continue
            status = _assess_interval(certificates[trace.trace_id], low, high)
            wanted = np.asarray(contact.normal_a_to_b)*(1. if contact.plate_b == trace.overriding_plate else -1.)
            if float(wanted@np.asarray(trace.normal_sub_to_over)) <= 0.:
                status = "incompatible_physical_fault_orientation"
            minimum, maximum, tolerance = _normal_velocity_bounds(trace, omega, surface.radius_km, low, high)
            if minimum >= -tolerance:
                status = "nonconvergent"
            elif maximum > tolerance:
                status = "mixed_convergence"
            supports.append((trace, low, high, status, minimum, maximum))
        directions = {(trace.subducting_plate, trace.overriding_plate) for trace, _, _, status, _, _ in supports
                      if status == "spatial_underthrust_admissible"}
        old = previous.get(contact.contact_id)
        old_directions = set() if old is None else {tuple(pair) for pair in old["directions"]}
        if len(old_directions) > 1:
            reason = "conflicting_local_history"
        elif not supports:
            reason = "no_continuous_physical_fault_support"
        elif any(status != "spatial_underthrust_admissible" for _, _, _, status, _, _ in supports):
            reasons = {status for _, _, _, status, _, _ in supports}
            reason = next(iter(reasons)) if len(reasons) == 1 else "unresolved_local_admission"
        elif len(directions) != 1:
            reason = "multiple_local_underthrust_directions"
        elif old_directions and old_directions != directions:
            reason = "history_mechanics_mismatch"
        else:
            # No newly merged current contact may inherit a partial old arc.
            intervals = sorted(_interval(contact.start, contact.end,
                *_spatial_intersect_arcs(trace.start, trace.end, contact.start, contact.end))
                for trace, *_ in supports)
            extent = _arc_frame(contact.start, contact.end)[3]
            cursor, complete = 0., True
            for low, high in intervals:
                if low > cursor+512.*_EPS*extent:
                    complete = False
                cursor = max(cursor, high)
            if extent-cursor > 512.*_EPS*extent:
                complete = False
            reason = "spatial_underthrust_admissible" if complete else "partial_continuous_contact_coverage"
        accepted = []
        if reason == "spatial_underthrust_admissible":
            sub, over = next(iter(directions))
            identity = "spatial-underthrust:"+_hash(dict(surface=surface_digest(surface),
                contact=contact.contact_id, node_manifest=step.node_manifest_sha256,
                traces=sorted(trace.trace_id for trace, *_ in supports), direction=(sub, over)))
            accepted.append(PolarityEvidence(identity, contact.contact_id, sub, over,
                "solved_continuous_closed_contact_down_dip_plastic_field"))
            evidence.extend(accepted)
        statuses[contact.contact_id] = reason
        rows.append(dict(contact_id=contact.contact_id, status=reason,
            supports=[dict(trace_id=trace.trace_id, low_radians=low, high_radians=high,
                status=status, normal_velocity_min_km_per_myr=minimum,
                normal_velocity_max_km_per_myr=maximum)
                for trace, low, high, status, minimum, maximum in supports],
            accepted_evidence_ids=[item.evidence_id for item in accepted]))
    # A contact may contain both opening and closing portions.  Its audit
    # above describes the entire contact, while admission below is restricted
    # to the Cartesian receiver arcs that actually bound this candidate loss.
    compiled = []
    for contact in contacts:
        for sub, over in ((contact.plate_a, contact.plate_b), (contact.plate_b, contact.plate_a)):
            receiver = contact.fragment_a if contact.plate_a == over else contact.fragment_b
            material = contact.material_a if contact.plate_a == over else contact.material_b
            moved = rotate_points((contact.start, contact.end), omega[over], dt)
            compiled.append((contact, sub, over, receiver, material, moved))
    overlap_rows = []
    def resolve(a, b, overlap):
        polygon = np.asarray(overlap)
        scoped, assessed, directions = set(), [], set()
        receiver_scope = {-1: set(), 1: set()}
        for contact, sub, over, receiver_id, material, moved in compiled:
            receiver, donor = (a, b) if a.fragment_id == receiver_id else (b, a)
            if (receiver.fragment_id != receiver_id or receiver.parcel.material_id != material
                    or receiver.parcel.plate != over or donor.parcel.plate != sub):
                continue
            actual_arcs = tuple(clipped for x, y in zip(polygon, np.roll(polygon, -1, axis=0))
                if (clipped := _spatial_intersect_arcs(moved[0], moved[1], x, y)) is not None)
            if not actual_arcs:
                continue
            scoped.add(contact.contact_id)
            comparison = -1 if receiver is a else 1
            receiver_scope[comparison].add(contact.contact_id)
            physical = tuple(trace for trace in carriers
                if (trace.subducting_plate, trace.overriding_plate) == (sub, over)
                and _spatial_intersect_arcs(trace.start, trace.end, contact.start, contact.end) is not None)
            # An opposite virtual receiver proves geometric location only;
            # it cannot create a physical material-pair port in that direction.
            if not physical:
                continue
            old = previous.get(contact.contact_id)
            old_directions = set() if old is None else {tuple(pair) for pair in old["directions"]}
            for actual_start, actual_end in actual_arcs:
                reason, support_rows, coverage = "spatial_underthrust_admissible", [], []
                extent = arc_length(actual_start, actual_end)
                if len(old_directions) > 1:
                    reason = "conflicting_local_history"
                elif old_directions and old_directions != {(sub, over)}:
                    reason = "history_mechanics_mismatch"
                for trace in physical:
                    moved_trace = rotate_points((trace.start, trace.end), omega[over], dt)
                    clipped = _spatial_intersect_arcs(*moved_trace, actual_start, actual_end)
                    if clipped is None:
                        continue
                    before_arc = rotate_points(clipped, omega[over], -dt)
                    low, high = _material_support_interval(trace, *before_arc,
                        certificates[trace.trace_id].extent_radians)
                    status = _assess_interval(certificates[trace.trace_id], low, high)
                    wanted = np.asarray(contact.normal_a_to_b)*(1. if contact.plate_b == over else -1.)
                    if float(wanted@np.asarray(trace.normal_sub_to_over)) <= 0.:
                        status = "incompatible_physical_fault_orientation"
                    minimum, maximum, tolerance = _partial_velocity_bounds(trace, omega,
                        surface.radius_km, *before_arc)
                    if minimum >= -tolerance:
                        status = "nonconvergent"
                    elif maximum > tolerance:
                        status = "mixed_convergence"
                    support_rows.append(dict(trace_id=trace.trace_id, low_radians=low,
                        high_radians=high, status=status,
                        normal_velocity_min_km_per_myr=minimum,
                        normal_velocity_max_km_per_myr=maximum,
                        normal_velocity_rounding_bound_km_per_myr=tolerance))
                    coverage.append(tuple(sorted(arc_length(actual_start, point) for point in clipped)))
                if not support_rows:
                    reason = "no_continuous_physical_fault_support"
                elif reason == "spatial_underthrust_admissible":
                    blocked = {row["status"] for row in support_rows}-{reason}
                    if blocked:
                        reason = next(iter(blocked)) if len(blocked) == 1 else "unresolved_local_admission"
                    else:
                        cursor = 0.
                        for low, high in sorted(coverage):
                            if low > cursor+512.*_EPS*extent:
                                reason = "partial_continuous_contact_coverage"
                            cursor = max(cursor, high)
                        if extent-cursor > 512.*_EPS*extent:
                            reason = "partial_continuous_contact_coverage"
                row = dict(contact_id=contact.contact_id, subducting_plate=sub,
                    overriding_plate=over, comparison=comparison,
                    actual_start=tuple(map(float, actual_start)),
                    actual_end=tuple(map(float, actual_end)), status=reason, supports=support_rows)
                assessed.append(row)
                if reason == "spatial_underthrust_admissible":
                    directions.add(comparison)
        if not scoped:
            return PolarityResolution(0, "no_resolved_initial_contact")
        supported = {row["contact_id"] for row in assessed}
        overlap_row = dict(fragment_a=a.fragment_id, fragment_b=b.fragment_id,
            geometric_scope_contact_ids=sorted(scoped), contact_ids=sorted(supported),
            receiver_arcs=assessed, comparison=0, accepted_evidence_ids=[])
        overlap_rows.append(overlap_row)
        if not supported:
            return PolarityResolution(0, "no_continuous_physical_fault_support", (), tuple(sorted(scoped)))
        blocked = {row["status"] for row in assessed}-{ "spatial_underthrust_admissible" }
        if blocked:
            return PolarityResolution(0, next(iter(blocked)) if len(blocked) == 1 else
                "unresolved_local_admission", (), tuple(sorted(supported)))
        if len(directions) != 1:
            return PolarityResolution(0, "no_unique_local_down_dip_support", (), tuple(sorted(supported)))
        comparison = next(iter(directions))
        receiver_contacts = {row["contact_id"] for row in assessed if row["comparison"] == comparison}
        if receiver_scope[comparison]-receiver_contacts:
            return PolarityResolution(0, "no_continuous_physical_fault_support", (),
                tuple(sorted(receiver_scope[comparison])))
        identity = "spatial-underthrust:"+_hash(dict(surface=surface_digest(surface),
            node_manifest=step.node_manifest_sha256, actual_receiver_arcs=assessed))
        overlap_row["comparison"] = comparison
        overlap_row["accepted_evidence_ids"] = [identity]
        return PolarityResolution(comparison, "solved_spatial_underthrust_admission",
            (identity,), tuple(sorted(supported)))
    audit = dict(format=FORMAT, model=MODEL, time_myr=surface.time_myr,
        dt_myr=dt, surface_sha256=surface_digest(surface),
        node_manifest_sha256=step.node_manifest_sha256, contacts=rows,
        contact_status_counts={reason: sum(row["status"] == reason for row in rows)
            for reason in sorted(set(statuses.values()))},
        accepted_evidence=[asdict(item) for item in evidence],
        actual_overlap_assessments=overlap_rows,
        contact_assessment_scope="whole-contact diagnostics; actual admission uses candidate receiver arcs",
        overlap_policy="every actual receiver arc must admit the same solved direction",
        material_policy="polygon overlap only; plastic slip is never converted to volume")
    return resolve, audit


def spatial_underthrust_memory_to_dict(memory):
    return dict(format=FORMAT, model=MODEL, **asdict(memory))


def spatial_underthrust_memory_from_dict(raw, initial_mechanics):
    expected = {item.name for item in fields(SpatialUnderthrustMemory)} | {"format", "model"}
    if not isinstance(raw, dict) or set(raw) != expected or raw["format"] != FORMAT or raw["model"] != MODEL:
        raise ValueError("Invalid continuous spatial memory schema")
    try:
        traces = []
        for row in raw["traces"]:
            if not isinstance(row, dict) or set(row) != {item.name for item in fields(SpatialTraceMemory)}:
                raise ValueError("Invalid continuous trace memory record")
            pieces = []
            for piece in row["pieces"]:
                if not isinstance(piece, dict) or set(piece) != {item.name for item in fields(SpatialPlasticPiece)}:
                    raise ValueError("Invalid continuous plastic piece record")
                values = dict(piece)
                for name in ("plastic_slip", "cumulative_slip"):
                    if not isinstance(values[name], dict) or set(values[name]) != {"coefficients"}:
                        raise ValueError("Invalid continuous polynomial field")
                    values[name] = TrigPolynomial(values[name]["coefficients"])
                values["low_radians"] = _real(values["low_radians"], "Piece start", nonnegative=True)
                values["high_radians"] = _real(values["high_radians"], "Piece end", positive=True)
                pieces.append(SpatialPlasticPiece(**values))
            traces.append(SpatialTraceMemory(row["trace_id"],
                _real(row["extent_radians"], "Material extent", positive=True), tuple(pieces)))
        memory = SpatialUnderthrustMemory(raw["physical_trace_sha256"], tuple(traces),
            _count(raw["stage_count"], "Continuous stage count"),
            _real(raw["elapsed_time_s"], "Continuous elapsed time", nonnegative=True))
        return _validate_memory(initial_mechanics, memory)
    except (TypeError, KeyError) as error:
        raise ValueError("Malformed continuous spatial memory") from error
