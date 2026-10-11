"""Complete positive-time swept donor footprint of the actual axis pair.

All donor material is represented, rather than one thin witness patch.  For
the bounded (+X,+Y,-Z)/(-X,+Y,-Z) source pair, exact full-vector Taylor bounds
prove every positive-time overlap lies in the actual endpoint donor domain.
The endpoint belongs to that union, so the two domains are equal.  The old
one-dimensional interface at time zero is retained separately.

This is a footprint proof, not uniform three-event material chronology or a
physical archive/force acceptance.  Arbitrary point history is an external
query of the same recorded trajectories; vacancies and births need their
own laws.
"""
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
import hashlib
import json

from .geometric_intrinsic_geometry import RotationIncrement
from .geometric_native_buried_support import inverse_native_history, native_pullback
from .geometric_native_contact_events import _Series, _isolate, _turning
from .geometric_native_donor_sweep import _bounds, _normal_coefficients, _source
from .geometric_native_polygons import (
    NativePlane, NativePolygon, NativeRay, _dot, _plane_values, _ray_same,
    _ray_values, _same_plane, _world, native_arc_to_dict, native_intersection,
    native_polygon_from_dict, native_polygon_to_dict, native_shared_boundary_arcs,
    native_subtract, validate_native_polygon,
)
from .geometric_native_pose import FractionInterval as I, certify_rotation_history

FORMAT = "native-complete-positive-time-swept-donor-domain-1"
MODEL = "bounded_axis_pair_swept_footprint_equals_endpoint"


class UnresolvedNativeSweptDomain(ValueError):
    """The complete swept-domain containment was not proved."""


def _encoded(raw):
    return json.dumps(raw, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _side_bounds(ray, plane, precision):
    return _dot(_ray_values(ray, precision)[1], _plane_values(plane, precision)[1])


@dataclass(frozen=True, slots=True)
class NativeSweptDomainCertificate:
    donor_polygon: NativePolygon
    receiver_polygon: NativePolygon
    donor_omega: tuple
    receiver_omega: tuple
    dt_myr: float
    precision: int
    order: int
    endpoint_world_overlap: NativePolygon
    endpoint_donor_domain: NativePolygon
    terminal_native_cone: NativePolygon
    initial_boundary_arc: object
    endpoint_area_bounds_sr: I
    c_bounds: I
    c_second_derivative_bounds: I
    main_y_first_derivative_bounds: I
    a_over_u_bounds: I
    b_over_u_bounds: I
    turning_derivative_bounds: I
    initial_turning_bounds: I
    terminal_turning_bounds: I
    ratio_turning_fraction: I
    initial_q_bounds: I
    terminal_q_bounds: I
    initial_minus_terminal_q_bounds: I
    terminal_p_bounds: I
    terminal_other_halfspace_vertex_bounds: tuple


def certify_native_swept_domain(donor_polygon, receiver_polygon, donor_omega,
        receiver_omega, dt_myr, *, precision=256, order=20):
    """Prove union(D(u), 0 < u <= 1) equals the endpoint material footprint.

    In signed donor coordinates p=(x,y,-z), x,y,z >= 0, the main receiver
    condition is y >= p(u)*x+q(u)*z, p=C/(u*A), q=-B/A.  C>0, C''<0,
    C'(0)=0 and (u*A)'>0 prove p decreases.  V=A*B'-A'*B has V'>0 and one
    zero, so q first rises then falls.  q(0)>q(1) proves q(u)>=q(1).  Thus
    every overlap satisfies the terminal main plane, irrespective of its
    other current receiver sides.  Authenticated native cone equality proves
    the terminal remaining sides add no footprint.  Time zero is separate.
    """
    if type(donor_polygon) is not NativePolygon or type(receiver_polygon) is not NativePolygon:
        raise ValueError("Swept domains require explicit native original polygons")
    if type(donor_omega) is not tuple or type(receiver_omega) is not tuple:
        raise ValueError("Swept domains require immutable complete Euler triples")
    if type(precision) is not int or not 64 <= precision <= 1024:
        raise ValueError("Swept-domain precision must be an explicit bounded integer")
    if type(order) is not int or not 8 <= order <= 64:
        raise ValueError("Swept-domain order must be an explicit bounded integer")
    ds, rs = RotationIncrement(donor_omega, dt_myr), RotationIncrement(receiver_omega, dt_myr)
    return _certify_native_swept_domain(donor_polygon, receiver_polygon,
        ds.omega_rad_per_myr, rs.omega_rad_per_myr, ds.dt_myr, precision, order)


@lru_cache(maxsize=32)
def _certify_native_swept_domain(donor_polygon, receiver_polygon, donor_omega,
        receiver_omega, dt_myr, precision, order):
    donor = _source(donor_polygon, ((1, 0, 0), (0, 1, 0), (0, 0, -1)))
    receiver = _source(receiver_polygon, ((-1, 0, 0), (0, 1, 0), (0, 0, -1)))
    ds, rs = RotationIncrement(donor_omega, dt_myr), RotationIncrement(receiver_omega, dt_myr)
    certify_rotation_history((ds,), precision)
    certify_rotation_history((rs,), precision)
    duration = Fraction.from_float(ds.dt_myr)
    da = tuple(Fraction.from_float(value)*duration for value in ds.omega_rad_per_myr)
    ra = tuple(Fraction.from_float(value)*duration for value in rs.omega_rad_per_myr)
    rate = sum(map(abs, da), Fraction(0))+sum(map(abs, ra), Fraction(0))
    normal = _normal_coefficients((-1, 0, 0), da, ra, order)
    if normal[0] != (Fraction(-1), Fraction(0), Fraction(0)) or normal[1][0] != 0:
        raise UnresolvedNativeSweptDomain("Original axis incidence and C'(0)=0 were not authenticated")
    c_coefficients = tuple(I(-row[0]) for row in normal)
    y_coefficients = tuple(I(row[1]) for row in normal)
    a = _Series(tuple(I(row[1]) for row in normal[1:]), Fraction(1), rate, precision)
    b = _Series(tuple(I(-row[2]) for row in normal[1:]), Fraction(1), rate, precision)
    whole = I(0, 1)
    c = _bounds(c_coefficients, whole, 0, rate, Fraction(1), precision)
    csecond = _bounds(c_coefficients, whole, 2, rate, Fraction(1), precision)
    yprime = _bounds(y_coefficients, whole, 1, rate, Fraction(1), precision)
    abound, bbound = a.bounds(whole), b.bounds(whole)
    if c.lo <= 0 or csecond.hi >= 0 or yprime.lo <= 0 or abound.lo <= 0 or bbound.hi >= 0:
        raise UnresolvedNativeSweptDomain("Complete swept main-cone coefficient signs were not certified")
    vprime = abound*b.bounds(whole, 2)-a.bounds(whole, 2)*bbound
    first, last = _turning(a, b, I(0)), _turning(a, b, I(1))
    if vprime.lo <= 0 or first.hi >= 0 or last.lo <= 0:
        raise UnresolvedNativeSweptDomain("Whole-interval single maximum of the cone ratio was not proved")
    turning = _isolate(lambda u: _turning(a, b, u), Fraction(0), Fraction(1), increasing=True)
    initial_q = -b.bounds(I(0))/a.bounds(I(0))
    terminal_q = -b.bounds(I(1))/a.bounds(I(1))
    delta_q = initial_q-terminal_q
    terminal_p = _bounds(c_coefficients, I(1), 0, rate, Fraction(1), precision)/a.bounds(I(1))
    if delta_q.lo <= 0 or terminal_q.lo <= 0 or terminal_p.lo <= 0:
        raise UnresolvedNativeSweptDomain("The endpoint is not the minimal complete donor cone")
    world_overlap = native_intersection(receiver.rotated(rs.omega_rad_per_myr, rs.dt_myr),
                                       donor.rotated(ds.omega_rad_per_myr, ds.dt_myr))
    if world_overlap is None:
        raise UnresolvedNativeSweptDomain("Actual endpoint overlap is not a positive native domain")
    endpoint = native_pullback(world_overlap, (ds,))
    inverse = inverse_native_history((ds,))
    source_x = next(p for p in donor.supports if p.normal == (1, 0, 0))
    source_z = next(p for p in donor.supports if p.normal == (0, 0, -1))
    receiver_x = next(p for p in receiver.supports if p.normal == (-1, 0, 0))
    final_x = receiver_x.rotated((rs,)+inverse)
    # The oriented generators are intersections of these exact native planes.
    va = NativeRay("cross", planes=(final_x, source_x), sign=1)
    vc = NativeRay("cross", planes=(source_x, source_z), sign=1)
    vb = NativeRay("cross", planes=(source_z, final_x), sign=1)
    cone = validate_native_polygon(NativePolygon((va, vc, vb), (source_x, source_z, final_x)))
    # Prove literal inward support and generator identity, not only approximate
    # area equality.  Source X/Z inverse words cancel analytically in the native
    # predicate; the receiver plane retains every component of both owners.
    transformed = _world(endpoint)
    if (len(transformed.vertices) != 3 or len(transformed.supports) != 3
            or any(not any(p.normal == wanted.normal and _same_plane(p, wanted)
                           for p in transformed.supports) for wanted in cone.supports)
            or any(not any(_ray_same(ray, wanted) for ray in transformed.vertices)
                   for wanted in cone.vertices)):
        raise UnresolvedNativeSweptDomain("Actual endpoint lost the complete terminal native cone lineage")
    if native_subtract(cone, endpoint) or native_subtract(endpoint, cone):
        raise UnresolvedNativeSweptDomain("Actual endpoint and proved terminal native cone are not equal")
    other = []
    for normal_reference in ((0, 1, 0), (0, 0, -1)):
        support = next(p for p in receiver.supports if p.normal == normal_reference).rotated((rs,)+inverse)
        values = tuple(_side_bounds(ray, support, precision) for ray in cone.vertices)
        if any(value.lo <= 0 for value in values):
            raise UnresolvedNativeSweptDomain("Another actual endpoint receiver side excludes the terminal cone")
        other.append((normal_reference, values))
    arcs = native_shared_boundary_arcs(donor, receiver)
    if len(arcs) != 1:
        raise UnresolvedNativeSweptDomain("Original one-dimensional interface was not retained separately")
    return NativeSweptDomainCertificate(donor, receiver, ds.omega_rad_per_myr,
        rs.omega_rad_per_myr, ds.dt_myr, precision, order, world_overlap, endpoint,
        cone, arcs[0], endpoint.area_bounds(), c, csecond, yprime, abound, bbound,
        vprime, first, last, turning, initial_q, terminal_q, delta_q, terminal_p,
        tuple(other))


def _fraction_body(value):
    return [value.numerator, value.denominator]


def _interval_body(value):
    return dict(lo=_fraction_body(value.lo), hi=_fraction_body(value.hi))


def _body(c):
    intervals = {name: _interval_body(getattr(c, name)) for name in (
        "endpoint_area_bounds_sr", "c_bounds", "c_second_derivative_bounds",
        "main_y_first_derivative_bounds", "a_over_u_bounds", "b_over_u_bounds",
        "turning_derivative_bounds", "initial_turning_bounds", "terminal_turning_bounds",
        "ratio_turning_fraction", "initial_q_bounds", "terminal_q_bounds",
        "initial_minus_terminal_q_bounds", "terminal_p_bounds")}
    return dict(format=FORMAT, model=MODEL, trajectory=dict(
        donor_polygon=native_polygon_to_dict(c.donor_polygon),
        receiver_polygon=native_polygon_to_dict(c.receiver_polygon),
        donor_omega=list(c.donor_omega), receiver_omega=list(c.receiver_omega), dt_myr=c.dt_myr),
        precision=c.precision, order=c.order,
        endpoint_world_overlap=native_polygon_to_dict(c.endpoint_world_overlap),
        endpoint_donor_domain=native_polygon_to_dict(c.endpoint_donor_domain),
        terminal_native_cone=native_polygon_to_dict(c.terminal_native_cone),
        initial_boundary=dict(native_arc=native_arc_to_dict(c.initial_boundary_arc),
            measure_class="one_dimensional_original_interface_zero_2D_measure",
            separate_from_positive_time_swept_domain=True, discarded=False),
        certificate=dict(intervals, c_initial_value=[1, 1], c_initial_derivative=[0, 1],
            terminal_other_halfspace_vertex_bounds=[dict(normal=list(normal),
                vertices=[_interval_body(value) for value in values])
                for normal, values in c.terminal_other_halfspace_vertex_bounds]),
        domain_identity="union_{0<u<=1} actual_original_donor_overlap(u) = endpoint_donor_domain",
        proof="decreasing_p_and_q_single_maximum_above_terminal_minimum_with_exact_native_cone_equality",
        material_time_history="arbitrary_point_query_requires_complete_all_receiver_halfspace_chronology",
        uniform_same_event_count_for_all_material=False,
        physical_archive_allocation_adopted=False, global_inventory_updated=False,
        admission_authorized=False,
        scope="complete_positive_time_swept_footprint_of_bounded_original_axis_pair")


def validate_native_swept_domain(certificate):
    if type(certificate) is not NativeSweptDomainCertificate:
        raise ValueError("Expected an explicit full native swept-domain certificate")
    proved = certify_native_swept_domain(certificate.donor_polygon, certificate.receiver_polygon,
        certificate.donor_omega, certificate.receiver_omega, certificate.dt_myr,
        precision=certificate.precision, order=certificate.order)
    if _encoded(_body(proved)) != _encoded(_body(certificate)):
        raise ValueError("Native swept-domain certificate differs from complete analytic and geometric replay")
    return proved


def native_swept_domain_to_dict(certificate):
    body = _body(validate_native_swept_domain(certificate))
    return dict(body, payload_sha256=hashlib.sha256(_encoded(body)).hexdigest())


def native_swept_domain_from_dict(raw):
    required = {"format", "model", "trajectory", "precision", "order", "endpoint_world_overlap",
        "endpoint_donor_domain", "terminal_native_cone", "initial_boundary", "certificate",
        "domain_identity", "proof", "material_time_history", "uniform_same_event_count_for_all_material",
        "physical_archive_allocation_adopted", "global_inventory_updated", "admission_authorized",
        "scope", "payload_sha256"}
    if type(raw) is not dict or set(raw) != required:
        raise ValueError("Invalid native full swept-domain schema")
    trajectory = raw["trajectory"]
    if (type(trajectory) is not dict or set(trajectory) != {
            "donor_polygon", "receiver_polygon", "donor_omega", "receiver_omega", "dt_myr"}
            or type(trajectory["dt_myr"]) is not float
            or any(type(trajectory[key]) is not list or len(trajectory[key]) != 3
                   or any(type(v) is not float for v in trajectory[key])
                   for key in ("donor_omega", "receiver_omega"))):
        raise ValueError("Invalid complete full swept-domain trajectory types")
    proved = certify_native_swept_domain(native_polygon_from_dict(trajectory["donor_polygon"]),
        native_polygon_from_dict(trajectory["receiver_polygon"]),
        tuple(trajectory["donor_omega"]), tuple(trajectory["receiver_omega"]), trajectory["dt_myr"],
        precision=raw["precision"], order=raw["order"])
    if _encoded(raw) != _encoded(native_swept_domain_to_dict(proved)):
        raise ValueError("Saved full swept-domain certificate differs from its sealed complete replay")
    return proved
