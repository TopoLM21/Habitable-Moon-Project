"""Complete continuous contact history of a positive original donor patch.

This bounded certificate uses the actual axis-octant source domains and both
complete constant Euler vectors.  A rational two-dimensional donor patch has
exactly three crossings of the receiver's -X plane; its other receiver sides
remain strictly positive for the whole interval.  The patch is not the whole
swept loss domain, and this certificate never grants force or burial adoption.
"""
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from math import factorial
import hashlib
import json

from .geometric_intrinsic_geometry import RotationIncrement
from .geometric_native_buried_support import native_pullback
from .geometric_native_contact_events import _isolate, _polynomial_bounds
from .geometric_native_polygons import (
    NativePlane, NativePolygon, NativeRay, _cross, _dot, native_intersection,
    native_polygon_from_dict, native_polygon_to_dict, native_subtract,
    validate_native_polygon,
)
from .geometric_native_pose import FractionInterval as I, certify_rotation_history

FORMAT = "native-original-donor-patch-sweep-1"
MODEL = "complete_three_event_original_donor_patch"


class UnresolvedNativeDonorSweep(ValueError):
    """The complete positive-patch contact history was not certified."""


def _encoded(raw):
    return json.dumps(raw, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _source(poly, normals):
    checked = validate_native_polygon(poly)
    if (checked.pose or len(checked.supports) != 3
            or any(p.rotations for p in checked.supports)
            or set(p.normal for p in checked.supports) != set(normals)):
        raise UnresolvedNativeDonorSweep("Donor sweep requires the authentic unposed axis-octant source")
    return checked


@lru_cache(maxsize=64)
def _normal_coefficients(normal, donor_axis, receiver_axis, order):
    """Exact coefficients of exp(-[d]u) exp([r]u) normal."""
    right = [tuple(Fraction(v) for v in normal)]
    for k in range(1, order+1):
        right.append(tuple(v/k for v in _cross(receiver_axis, right[-1])))
    result = [tuple(Fraction(0) for _ in range(3)) for _ in range(order+1)]
    negative = tuple(-v for v in donor_axis)
    for k, coefficient in enumerate(right):
        left = coefficient
        for j in range(order-k+1):
            if j:
                left = tuple(v/j for v in _cross(negative, left))
            result[k+j] = tuple(a+b for a, b in zip(result[k+j], left))
    return tuple(result)


def _scalar_coefficients(vector_coefficients, coordinates):
    return tuple(_dot(row, coordinates) for row in vector_coefficients)


def _bounds(coefficients, domain, degree, rate, scale, precision):
    n = len(coefficients)-1
    if domain.lo < 0 or domain.hi > 1:
        raise ValueError("A donor sweep query must stay inside its declared interval")
    derivative = tuple(value*Fraction(factorial(k), factorial(k-degree))
                       for k, value in enumerate(coefficients) if k >= degree)
    # The first omitted coefficient is bounded by M L^(n+1)/(n+1)!.
    # Differentiation multiplies by (n+1)!/(n+1-r)!, and the remaining
    # derivative-tail ratio is at most L*u/(n+2-r).
    ratio = rate*domain.hi/Fraction(n+2-degree)
    if ratio >= 1:
        raise UnresolvedNativeDonorSweep("Donor sweep Taylor remainder exceeds its bounded domain")
    remainder = scale*rate**(n+1)*domain.hi**(n+1-degree)
    remainder /= factorial(n+1-degree)*(1-ratio)
    return _polynomial_bounds(derivative, domain, remainder, precision)


def _patch(x, y):
    # The chart p=(x,y,-1) reverses the ordinary x/y planar orientation.
    vertices = tuple(NativeRay("seed", (a, b, Fraction(-1))) for a, b in (
        (x.lo, y.lo), (x.lo, y.hi), (x.hi, y.hi), (x.hi, y.lo)))
    supports = tuple(NativePlane(_cross(a.reference, b.reference))
                     for a, b in zip(vertices, vertices[1:]+vertices[:1]))
    return validate_native_polygon(NativePolygon(vertices, supports))


@dataclass(frozen=True, slots=True)
class NativeDonorPatchEvent:
    kind: str
    time_fraction: I
    time_myr: I


@dataclass(frozen=True, slots=True)
class NativeDonorPatchSweepCertificate:
    donor_polygon: NativePolygon
    receiver_polygon: NativePolygon
    donor_omega: tuple
    receiver_omega: tuple
    dt_myr: float
    precision: int
    order: int
    q0: Fraction
    q1: Fraction
    q2: Fraction
    turning_scale_fraction: Fraction
    height_fraction: Fraction
    width_scale_fraction: Fraction
    patch_x_bounds: I
    patch_y_bounds: I
    patch_geometry: NativePolygon
    patch_area_bounds: I
    sample_fractions: tuple[Fraction, ...]
    gap_sample_bounds: tuple[I, ...]
    first_entry_negative_fraction: Fraction
    first_entry_negative_gap_bounds: I
    gap_third_derivative_bounds: I
    receiver_positive_bounds: tuple[I, I]
    event_windows: tuple[NativeDonorPatchEvent, ...]
    endpoint_donor_domain: NativePolygon

    @property
    def patch(self):
        return self.patch_geometry

    @property
    def x_bounds(self):
        return self.patch_x_bounds

    @property
    def y_bounds(self):
        return self.patch_y_bounds

    @property
    def witness_times(self):
        return self.sample_fractions

    @property
    def gap_state_bounds(self):
        return self.gap_sample_bounds

    @property
    def main_third_derivative_bounds(self):
        return self.gap_third_derivative_bounds

    @property
    def other_support_bounds(self):
        return tuple(zip(((0, 1, 0), (0, 0, -1)), self.receiver_positive_bounds))

    @property
    def patch_area_bounds_sr(self):
        return self.patch_area_bounds


def certify_native_donor_patch_sweep(donor_polygon, receiver_polygon,
        donor_omega, receiver_omega, dt_myr, *, precision=256, order=20):
    """Prove exactly entry/exit/entry for every point of a true 2D patch.

    Three strict alternating sign brackets and g''' > 0 over the complete
    interval imply exactly three distinct roots by Rolle's theorem.  This
    proof holds for the entire rational coordinate box, including its edges;
    no sampled corner or midpoint replaces its two-dimensional proof.
    """
    if type(donor_polygon) is not NativePolygon or type(receiver_polygon) is not NativePolygon:
        raise ValueError("Donor sweep requires explicit native source polygons")
    if type(precision) is not int or not 64 <= precision <= 1024:
        raise ValueError("Donor sweep precision must be an explicit bounded integer")
    if type(order) is not int or not 8 <= order <= 64:
        raise ValueError("Donor sweep Taylor order must be an explicit bounded integer")
    if type(donor_omega) is not tuple or type(receiver_omega) is not tuple:
        raise ValueError("Donor sweep requires immutable full Euler triples")
    # Validate and normalize inputs before they become keys of the pure cache.
    ds, rs = RotationIncrement(donor_omega, dt_myr), RotationIncrement(receiver_omega, dt_myr)
    return _certify_native_donor_patch_sweep(donor_polygon, receiver_polygon,
        ds.omega_rad_per_myr, rs.omega_rad_per_myr, ds.dt_myr, precision, order)


@lru_cache(maxsize=32)
def _certify_native_donor_patch_sweep(donor_polygon, receiver_polygon,
        donor_omega, receiver_omega, dt_myr, precision, order):
    donor = _source(donor_polygon, ((1, 0, 0), (0, 1, 0), (0, 0, -1)))
    receiver = _source(receiver_polygon, ((-1, 0, 0), (0, 1, 0), (0, 0, -1)))
    ds, rs = RotationIncrement(donor_omega, dt_myr), RotationIncrement(receiver_omega, dt_myr)
    certify_rotation_history((ds,), precision)
    certify_rotation_history((rs,), precision)
    duration = Fraction.from_float(ds.dt_myr)
    da = tuple(Fraction.from_float(v)*duration for v in ds.omega_rad_per_myr)
    ra = tuple(Fraction.from_float(v)*duration for v in rs.omega_rad_per_myr)
    rate = sum(map(abs, da), Fraction(0))+sum(map(abs, ra), Fraction(0))
    coefficients = _normal_coefficients((-1, 0, 0), da, ra, order)
    if not coefficients[1][1]:
        raise UnresolvedNativeDonorSweep("Donor material patch lacks a positive entry scale")
    q0 = coefficients[1][2]/coefficients[1][1]
    q1 = (coefficients[2][2]-q0*coefficients[2][1])/coefficients[1][1]
    q2 = (coefficients[3][2]-q0*coefficients[3][1]-q1*coefficients[2][1])/coefficients[1][1]
    if q0 <= 0 or q1 <= 0 or q2 >= 0 or coefficients[1][1] <= 0:
        raise UnresolvedNativeDonorSweep("Complete donor recrossing patch recipe was not proved")
    u = -q1/(2*q2)
    height = -q1*q1/(4*q2)
    width = coefficients[1][1]*u*height
    if not 0 < 4*u < 1 or height <= 0 or width <= 0:
        raise UnresolvedNativeDonorSweep("Positive donor patch and chronological event brackets were not proved")
    x, y = I(width/1000, width/500), I(q0+height/3, q0+2*height/3)
    coordinates = (x, y, I(-1))
    scale = x.hi+y.hi+1
    scalar = _scalar_coefficients(coefficients, coordinates)
    samples = (Fraction(0), u/16, u, 4*u, Fraction(1))
    signs = tuple(_bounds(scalar, I(t), 0, rate, scale, precision) for t in samples)
    if not (signs[0].hi < 0 and signs[1].lo > 0 and signs[2].hi < 0
            and signs[3].lo > 0 and signs[4].lo > 0):
        raise UnresolvedNativeDonorSweep("Strict three-event donor patch signs were not certified")
    first_negative_time = u/4096
    first_negative = _bounds(scalar, I(first_negative_time), 0, rate, scale, precision)
    if first_negative.hi >= 0:
        raise UnresolvedNativeDonorSweep("Uniform strictly positive first-entry time was not certified")
    third = _bounds(scalar, I(0, 1), 3, rate, scale, precision)
    if third.lo <= 0:
        raise UnresolvedNativeDonorSweep("Whole-interval donor root count was not certified")
    positive = tuple(_bounds(_scalar_coefficients(
        _normal_coefficients(normal, da, ra, order), coordinates), I(0, 1), 0,
        rate, scale, precision) for normal in ((0, 1, 0), (0, 0, -1)))
    if any(bound.lo <= 0 for bound in positive):
        raise UnresolvedNativeDonorSweep("Another actual receiver side may change the patch history")
    patch = _patch(x, y)
    if native_subtract(patch, donor):
        raise UnresolvedNativeDonorSweep("Donor patch extends outside its original source domain")
    # Receiver-first clipping retains the literal support expression order of
    # the octant:2/octant:6 source candidate, as well as its exact domain.
    overlap = native_intersection(receiver.rotated(rs.omega_rad_per_myr, rs.dt_myr),
                                  donor.rotated(ds.omega_rad_per_myr, ds.dt_myr))
    if overlap is None:
        raise UnresolvedNativeDonorSweep("Donor patch lacks a positive actual endpoint overlap")
    endpoint = native_pullback(overlap, (ds,))
    if native_subtract(patch, endpoint):
        raise UnresolvedNativeDonorSweep("Donor patch is not inside the actual original-material endpoint loss")
    windows = tuple(NativeDonorPatchEvent(kind, I(a, b), I(a, b)*duration)
                    for kind, a, b in (("entry", first_negative_time, samples[1]),
                                       ("exit", samples[1], samples[2]),
                                       ("entry", samples[2], samples[3])))
    return NativeDonorPatchSweepCertificate(donor, receiver, ds.omega_rad_per_myr,
        rs.omega_rad_per_myr, ds.dt_myr, precision, order, q0, q1, q2, u, height,
        width, x, y, patch, patch.area_bounds(), samples, signs,
        first_negative_time, first_negative, third, positive,
        windows, endpoint)


def _fraction_body(value):
    return [value.numerator, value.denominator]


def _interval_body(value):
    return dict(lo=_fraction_body(value.lo), hi=_fraction_body(value.hi))


def _body(c):
    return dict(format=FORMAT, model=MODEL, trajectory=dict(
        donor_polygon=native_polygon_to_dict(c.donor_polygon),
        receiver_polygon=native_polygon_to_dict(c.receiver_polygon),
        donor_omega=list(c.donor_omega), receiver_omega=list(c.receiver_omega), dt_myr=c.dt_myr),
        precision=c.precision, order=c.order, recipe={name: _fraction_body(value)
            for name, value in (("q0", c.q0), ("q1", c.q1), ("q2", c.q2),
                                ("U", c.turning_scale_fraction), ("H", c.height_fraction),
                                ("B", c.width_scale_fraction))},
        patch_geometry=native_polygon_to_dict(c.patch_geometry),
        endpoint_donor_domain=native_polygon_to_dict(c.endpoint_donor_domain),
        certificate=dict(patch_x_bounds=_interval_body(c.patch_x_bounds),
            patch_y_bounds=_interval_body(c.patch_y_bounds),
            patch_area_bounds=_interval_body(c.patch_area_bounds),
            sample_fractions=[_fraction_body(t) for t in c.sample_fractions],
            gap_sample_bounds=[_interval_body(b) for b in c.gap_sample_bounds],
            first_entry_negative_fraction=_fraction_body(c.first_entry_negative_fraction),
            first_entry_negative_gap_bounds=_interval_body(c.first_entry_negative_gap_bounds),
            gap_third_derivative_bounds=_interval_body(c.gap_third_derivative_bounds),
            receiver_positive_bounds=[_interval_body(b) for b in c.receiver_positive_bounds],
            event_windows=[dict(kind=e.kind, time_fraction=_interval_body(e.time_fraction),
                                time_myr=_interval_body(e.time_myr)) for e in c.event_windows]),
        admission_authorized=False,
        scope="positive_original_donor_2D_patch_not_complete_swept_domain")


def validate_native_donor_patch_sweep(certificate):
    if type(certificate) is not NativeDonorPatchSweepCertificate:
        raise ValueError("Expected an explicit native donor patch sweep certificate")
    proved = certify_native_donor_patch_sweep(certificate.donor_polygon,
        certificate.receiver_polygon, certificate.donor_omega, certificate.receiver_omega,
        certificate.dt_myr, precision=certificate.precision, order=certificate.order)
    if _encoded(_body(proved)) != _encoded(_body(certificate)):
        raise ValueError("Native donor patch sweep differs from its complete analytic replay")
    return proved


def isolate_native_donor_patch_point_events(certificate, x, y):
    """Refine the three proved roots for one exact original material point."""
    c = validate_native_donor_patch_sweep(certificate)
    if type(x) is not Fraction or type(y) is not Fraction:
        raise ValueError("Original donor point coordinates must be exact Fraction values")
    if not (c.patch_x_bounds.lo <= x <= c.patch_x_bounds.hi
            and c.patch_y_bounds.lo <= y <= c.patch_y_bounds.hi):
        raise ValueError("Original donor point lies outside the proved material patch")
    duration = Fraction.from_float(c.dt_myr)
    donor = tuple(Fraction.from_float(v)*duration for v in c.donor_omega)
    receiver = tuple(Fraction.from_float(v)*duration for v in c.receiver_omega)
    coefficients = _scalar_coefficients(_normal_coefficients(
        (-1, 0, 0), donor, receiver, c.order), (I(x), I(y), I(-1)))
    rate = sum(map(abs, donor), Fraction(0))+sum(map(abs, receiver), Fraction(0))
    function = lambda u: _bounds(coefficients, u, 0, rate, x+y+1, c.precision)
    result = []
    for window in c.event_windows:
        time = _isolate(function, window.time_fraction.lo, window.time_fraction.hi,
                        increasing=window.kind == "entry")
        result.append(NativeDonorPatchEvent(window.kind, time, time*duration))
    return tuple(result)


def native_donor_patch_sweep_to_dict(certificate):
    checked = validate_native_donor_patch_sweep(certificate)
    body = _body(checked)
    return dict(body, payload_sha256=hashlib.sha256(_encoded(body)).hexdigest())


def native_donor_patch_sweep_from_dict(raw):
    required = {"format", "model", "trajectory", "precision", "order", "recipe",
                "patch_geometry", "endpoint_donor_domain", "certificate",
                "admission_authorized", "scope", "payload_sha256"}
    if type(raw) is not dict or set(raw) != required:
        raise ValueError("Invalid native donor patch sweep schema")
    trajectory = raw["trajectory"]
    if (type(trajectory) is not dict or set(trajectory) != {
            "donor_polygon", "receiver_polygon", "donor_omega", "receiver_omega", "dt_myr"}
            or any(type(trajectory[key]) is not list or len(trajectory[key]) != 3
                   or any(type(v) is not float for v in trajectory[key])
                   for key in ("donor_omega", "receiver_omega"))
            or type(trajectory["dt_myr"]) is not float):
        raise ValueError("Invalid complete native donor sweep trajectory schema")
    proved = certify_native_donor_patch_sweep(
        native_polygon_from_dict(trajectory["donor_polygon"]),
        native_polygon_from_dict(trajectory["receiver_polygon"]),
        tuple(trajectory["donor_omega"]), tuple(trajectory["receiver_omega"]),
        trajectory["dt_myr"], precision=raw["precision"], order=raw["order"])
    if _encoded(raw) != _encoded(native_donor_patch_sweep_to_dict(proved)):
        raise ValueError("Saved native donor patch sweep differs from its sealed analytic replay")
    return proved
