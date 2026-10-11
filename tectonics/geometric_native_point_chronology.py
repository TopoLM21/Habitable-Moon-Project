"""Complete bounded chronology of one original point and every receiver side.

The source epoch is an explicit unposed native polygon with primitive seed
vertices. Both complete Euler vectors are retained. Exact initial coefficients
and a nine-dimensional Cayley--Hamilton identity authenticate initial incidence
and all-time coincidence. Every time cell proves no root or one unique root;
an unresolved tangency, event order, or exhausted budget refuses certification.

This point certificate supplies neither a swept area nor material, polarity,
mechanical, birth, or force adoption. A closed receiver boundary is recorded
explicitly, rather than promoted to a finite material allocation.
"""
from dataclasses import dataclass, replace
from fractions import Fraction
from functools import lru_cache
from math import factorial
import hashlib
import json

from .geometric_intrinsic_geometry import RotationIncrement
from .geometric_native_contact_events import _polynomial_bounds
from .geometric_native_donor_sweep import _normal_coefficients
from .geometric_native_polygons import (
    NativePlane, NativePolygon, NativeRay, _dot, _side,
    native_plane_to_dict, native_polygon_from_dict, native_polygon_to_dict,
    validate_native_polygon,
)
from .geometric_native_pose import FractionInterval as I, certify_rotation_history

FORMAT = "native-full-point-chronology-1"
MODEL = "complete_all_receiver_halfspace_point_history"
QUERY_FORMAT = "native-full-point-chronology-query-1"


class UnresolvedNativePointChronology(ValueError):
    """The complete all-side point history was not proved within its budget."""


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _source(polygon):
    if type(polygon) is not NativePolygon:
        raise ValueError("Point chronology requires an explicit native source polygon")
    checked = validate_native_polygon(polygon)
    if (checked.pose or any(p.rotations for p in checked.supports)
            or any(ray.kind != "seed" or ray.rotations for ray in checked.vertices)):
        raise UnresolvedNativePointChronology("Point chronology currently requires an unposed primitive native source epoch")
    return checked


def _point(raw):
    if type(raw) is not tuple or len(raw) != 3:
        raise ValueError("Original point requires an immutable exact Cartesian triple")
    if any(type(value) not in (int, Fraction) for value in raw):
        raise ValueError("Original point components must be exact Fraction or integer values")
    point = tuple(Fraction(value) for value in raw)
    if not any(point):
        raise ValueError("Original point direction must be nonzero")
    return point


def _budget_value(value, low, high, name):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(name+" must be an explicit bounded integer")
    return value


def _normalize_positive(values):
    scale = max(map(abs, values))
    return tuple(Fraction(value)/scale for value in values)


def _multiply_polynomials(a, b):
    out = [Fraction(0) for _ in range(len(a)+len(b)-1)]
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i+j] += x*y
    return tuple(out)


def _characteristic(donor, receiver):
    # The tensor generator has frequencies 0, +/-|d|, +/-|r|,
    # +/-(|d|+|r|), +/-(|d|-|r|), including exact repeated frequencies.
    a, b = _dot(donor, donor), _dot(receiver, receiver)
    value = (Fraction(0), Fraction(1))
    for factor in ((a, 0, 1), (b, 0, 1), ((a-b)**2, 0, 2*(a+b), 0, 1)):
        value = _multiply_polynomials(value, factor)
    return tuple(Fraction(item) for item in value)


@dataclass(frozen=True, slots=True)
class NativePointTimeCell:
    time_fraction: I
    status: str
    sign: int
    gap_bounds: I
    derivative_bounds: I | None
    left_gap_bounds: I | None
    right_gap_bounds: I | None
    root_fraction: I | None = None


@dataclass(frozen=True, slots=True)
class NativePointPlaneEvent:
    sequence: int
    kind: str
    support_index: int
    time_fraction: I
    time_myr: I


@dataclass(frozen=True, slots=True)
class NativePointOccupancyEvent:
    sequence: int
    kind: str
    support_index: int
    time_fraction: I
    time_myr: I
    plane_event_sequence: int
    support_indices: tuple
    plane_event_sequences: tuple


@dataclass(frozen=True, slots=True)
class NativePointPlaneChronology:
    support_index: int
    support: NativePlane
    initial_zero_order: int | None
    coincident: bool
    cells: tuple
    events: tuple


@dataclass(frozen=True, slots=True)
class NativePointChronologyCertificate:
    donor_polygon: NativePolygon | None
    receiver_polygon: NativePolygon
    donor_omega: tuple
    receiver_omega: tuple
    dt_myr: float
    point: tuple
    precision: int
    order: int
    max_depth: int
    max_cells: int
    generator_characteristic_polynomial: tuple
    plane_chronologies: tuple
    plane_events: tuple
    occupancy_events: tuple
    initial_occupancy: str
    initial_boundary_supports: tuple
    processed_cells: int


@dataclass(frozen=True, slots=True)
class NativePointChronologyQuery:
    certificate: NativePointChronologyCertificate
    certificate_sha256: str
    time_fraction: Fraction
    time_myr: Fraction
    occupancy: str
    plane_signs: tuple
    active_support_indices: tuple
    coincident_support_indices: tuple
    event_sequence: tuple


@dataclass(frozen=True, slots=True)
class _FactoredSeries:
    coefficients: tuple
    zero_order: int
    scale: Fraction
    rate: Fraction
    order: int
    precision: int

    def bounds(self, domain, derivative=0):
        m, n = self.zero_order, self.order
        j = n+1-m
        if derivative > j or not 0 <= domain.lo <= domain.hi <= 1:
            raise ValueError("Factored point series query exceeds its bounded derivative domain")
        coefficients = tuple(value*Fraction(factorial(k), factorial(k-derivative))
            for k, value in enumerate(self.coefficients) if k >= derivative)
        if not coefficients:
            coefficients = (I(0),)
        first = self.scale*self.rate**(n+1)/factorial(n+1)
        first *= Fraction(factorial(j), factorial(j-derivative))*domain.hi**(j-derivative)
        ratio = self.rate*domain.hi*Fraction(j+1, (n+2)*(j+1-derivative))
        if ratio >= 1:
            raise UnresolvedNativePointChronology("Factored point Taylor tail exceeded its norm domain")
        return _polynomial_bounds(coefficients, domain, first/(1-ratio), self.precision)


class _Budget:
    def __init__(self, cells, depth):
        self.max_cells, self.max_depth, self.count = cells, depth, 0

    def tick(self, depth):
        self.count += 1
        if self.count > self.max_cells or depth > self.max_depth:
            raise UnresolvedNativePointChronology("Complete point chronology exhausted its declared cell or depth budget")


def _sign(value):
    return 1 if value.lo > 0 else -1 if value.hi < 0 else 0


def _refine_root(series, low, high, depth, budget, increasing):
    """Refine an already proved monotone sign bracket without sampled grants."""
    before, after = (-1, 1) if increasing else (1, -1)
    for _ in range(budget.max_depth+1):
        if low > 0 and high-low <= high/Fraction(10**24):
            return I(low, high)
        depth += 1
        budget.tick(depth)
        middle = (low+high)/2
        value = _sign(series.bounds(I(middle)))
        if value == before:
            low = middle
        elif value == after:
            high = middle
        else:
            left, right = (low+middle)/2, (middle+high)/2
            sl, sr = _sign(series.bounds(I(left))), _sign(series.bounds(I(right)))
            if sl == before and sr == after:
                low, high = left, right
                depth += 1
            elif sl == after:
                high = left
                depth += 1
            elif sr == before:
                low = right
                depth += 1
            else:
                raise UnresolvedNativePointChronology("A possible multiple or exact midpoint root lacks a strict enclosure")
            budget.tick(depth)
    raise UnresolvedNativePointChronology("Point root isolation exhausted its exact subdivision budget")


def _plane_chronology(index, support, donor, receiver, point, order, precision, dt, budget):
    normal = _normalize_positive(support.normal)
    vectors = _normal_coefficients(normal, donor, receiver, order)
    coefficients = tuple(_dot(vector, point) for vector in vectors)
    if all(not value for value in coefficients[:9]):
        # The exact rational tensor generator is 9 dimensional. Its degree-9
        # characteristic identity makes these nine zeros a full analytic
        # identity, including repeated or zero frequencies.
        cell = NativePointTimeCell(I(0, 1), "identically_zero", 0, I(0), I(0), I(0), I(0))
        budget.tick(0)
        return NativePointPlaneChronology(index, support, None, True, (cell,), ()), None
    zero_order = next(i for i, value in enumerate(coefficients[:9]) if value)
    scale = sum(map(abs, normal), Fraction(0))*sum(map(abs, point), Fraction(0))
    rate = sum(map(abs, donor), Fraction(0))+sum(map(abs, receiver), Fraction(0))
    series = _FactoredSeries(tuple(I(value) for value in coefficients[zero_order:]),
                             zero_order, scale, rate, order, precision)
    pending, cells, events = [(Fraction(0), Fraction(1), 0)], [], []
    if zero_order:
        events.append(NativePointPlaneEvent(0, "entry" if coefficients[zero_order] > 0 else "exit",
                                           index, I(0), I(0)))
    while pending:
        low, high, depth = pending.pop()
        budget.tick(depth)
        domain = I(low, high)
        gap = series.bounds(domain)
        sign = _sign(gap)
        if sign:
            cells.append(NativePointTimeCell(domain, "no_root", sign, gap, None, None, None))
            continue
        derivative = series.bounds(domain, 1)
        oriented = _sign(derivative)
        if oriented:
            left, right = series.bounds(I(low)), series.bounds(I(high))
            sl, sr = _sign(left), _sign(right)
            if sl and sr and sl == sr:
                cells.append(NativePointTimeCell(domain, "no_root", sl, gap, derivative, left, right))
                continue
            if (sl, sr) == ((-1, 1) if oriented > 0 else (1, -1)):
                root = _refine_root(series, low, high, depth, budget, oriented > 0)
                cells.append(NativePointTimeCell(domain, "unique_root", 0, gap, derivative, left, right, root))
                events.append(NativePointPlaneEvent(0, "entry" if oriented > 0 else "exit",
                                                   index, root, root*dt))
                continue
        if depth >= budget.max_depth:
            raise UnresolvedNativePointChronology("Unresolved point tangency, endpoint incidence or derivative sign")
        middle = (low+high)/2
        pending.extend(((middle, high, depth+1), (low, middle, depth+1)))
    cells.sort(key=lambda cell: cell.time_fraction.lo)
    if (cells[0].time_fraction.lo != 0 or cells[-1].time_fraction.hi != 1
            or any(a.time_fraction.hi != b.time_fraction.lo for a, b in zip(cells, cells[1:]))):
        raise AssertionError("Point proof time cells do not form a complete partition")
    events.sort(key=lambda event: event.time_fraction.lo)
    return NativePointPlaneChronology(index, support, zero_order, False, tuple(cells), tuple(events)), series


def _right_germ(plane):
    if plane.coincident:
        return 0
    initial = plane.events[0] if plane.initial_zero_order else None
    if initial is not None:
        return 1 if initial.kind == "entry" else -1
    return plane.cells[0].sign if plane.cells[0].status != "unique_root" else (
        -1 if plane.cells[0].derivative_bounds.lo > 0 else 1)


def _other_sign_before_event(plane, time):
    sign = _right_germ(plane)
    for event in plane.events:
        if event.time_fraction.hi < time.lo or event.time_fraction.hi == 0:
            sign = 1 if event.kind == "entry" else -1
        elif event.time_fraction.lo > time.hi:
            break
        else:
            raise UnresolvedNativePointChronology("Different support event times lack a proved strict order")
    return sign


def _occupancy_events(planes, events, initial):
    out = []
    initial_events = tuple(event for event in events if event.time_fraction.hi == 0)
    if initial_events and initial == "inside":
        right_inside = all(_right_germ(plane) >= 0 for plane in planes)
        kind = "entry" if right_inside else "exit"
        chosen = tuple(event for event in initial_events if event.kind == kind)
        if chosen:
            first = chosen[0]
            out.append(NativePointOccupancyEvent(1, kind, first.support_index, I(0), I(0),
                first.sequence, tuple(e.support_index for e in initial_events),
                tuple(e.sequence for e in initial_events)))
    for event in events:
        if event.time_fraction.hi == 0:
            continue
        others = tuple(_other_sign_before_event(plane, event.time_fraction)
                       for plane in planes if plane.support_index != event.support_index)
        if all(sign >= 0 for sign in others):
            out.append(NativePointOccupancyEvent(len(out)+1, event.kind, event.support_index,
                event.time_fraction, event.time_myr, event.sequence, (event.support_index,), (event.sequence,)))
    return tuple(out)


def _public_inputs(donor_polygon, receiver_polygon, donor_omega, receiver_omega, dt_myr,
                   point, precision, order, max_depth, max_cells):
    donor = None if donor_polygon is None else _source(donor_polygon)
    receiver = _source(receiver_polygon)
    if type(donor_omega) is not tuple or type(receiver_omega) is not tuple:
        raise ValueError("Point chronology requires immutable complete Euler triples")
    ds, rs = RotationIncrement(donor_omega, dt_myr), RotationIncrement(receiver_omega, dt_myr)
    original = _point(point)
    precision = _budget_value(precision, 32, 1024, "Point precision")
    order = _budget_value(order, 8, 64, "Point Taylor order")
    depth = _budget_value(max_depth, 1, 768, "Point depth budget")
    cells = _budget_value(max_cells, 1, 65536, "Point cell budget")
    certify_rotation_history((ds,), precision)
    certify_rotation_history((rs,), precision)
    if donor is not None and any(_side(NativeRay("seed", original), plane) < 0 for plane in donor.supports):
        raise ValueError("Point is outside its original native donor material domain")
    return donor, receiver, ds.omega_rad_per_myr, rs.omega_rad_per_myr, ds.dt_myr, original, precision, order, depth, cells


def certify_native_point_chronology(donor_polygon, receiver_polygon, donor_omega,
        receiver_omega, dt_myr, point, *, precision=256, order=24, max_depth=192, max_cells=4096):
    """Certify all receiver-side roots for a point of the original donor."""
    if donor_polygon is None:
        raise ValueError("A donor point chronology requires its original material polygon")
    return _certify(*_public_inputs(donor_polygon, receiver_polygon, donor_omega,
        receiver_omega, dt_myr, point, precision, order, max_depth, max_cells))


def certify_native_point_receiver_chronology(receiver_polygon, donor_omega,
        receiver_omega, dt_myr, point, *, precision=256, order=24, max_depth=192, max_cells=4096):
    """Certify a point/receiver path without granting donor material membership.

    A fixed-world vacancy query explicitly supplies a zero donor Euler vector.
    Its caller separately binds its world ray and original source partition.
    """
    return _certify(*_public_inputs(None, receiver_polygon, donor_omega,
        receiver_omega, dt_myr, point, precision, order, max_depth, max_cells))


@lru_cache(maxsize=128)
def _certify(donor_polygon, receiver_polygon, donor_omega, receiver_omega, dt_myr,
             original_point, precision, order, max_depth, max_cells):
    dt = Fraction.from_float(dt_myr)
    donor = tuple(Fraction.from_float(value)*dt for value in donor_omega)
    receiver = tuple(Fraction.from_float(value)*dt for value in receiver_omega)
    point = _normalize_positive(original_point)
    characteristic = _characteristic(donor, receiver)
    budget = _Budget(max_cells, max_depth)
    planes = tuple(_plane_chronology(index, support, donor, receiver, point,
        order, precision, dt, budget)[0] for index, support in enumerate(receiver_polygon.supports))
    all_events = sorted((event for plane in planes for event in plane.events),
                        key=lambda event: (event.time_fraction.lo, event.support_index))
    for first, last in zip(all_events, all_events[1:]):
        if first.time_fraction.hi >= last.time_fraction.lo and not (
                first.time_fraction.hi == last.time_fraction.hi == 0):
            raise UnresolvedNativePointChronology("Different support roots have unresolved coincidence or time order")
    all_events = tuple(replace(event, sequence=index) for index, event in enumerate(all_events, 1))
    planes = tuple(replace(plane, events=tuple(event for event in all_events
        if event.support_index == plane.support_index)) for plane in planes)
    initial_sides = tuple(_dot(_normalize_positive(plane.normal), point)
                          for plane in receiver_polygon.supports)
    initial = "inside" if all(side >= 0 for side in initial_sides) else "outside"
    boundaries = tuple(index for index, side in enumerate(initial_sides) if not side)
    occupancy = _occupancy_events(planes, all_events, initial)
    return NativePointChronologyCertificate(donor_polygon, receiver_polygon, donor_omega,
        receiver_omega, dt_myr, original_point, precision, order, max_depth, max_cells,
        characteristic, planes, all_events, occupancy, initial, boundaries, budget.count)


def _fraction_body(value):
    return [value.numerator, value.denominator]


def _interval_body(value):
    return None if value is None else dict(lo=_fraction_body(value.lo), hi=_fraction_body(value.hi))


def _event_body(event):
    raw = dict(sequence=event.sequence, kind=event.kind, support_index=event.support_index,
               time_fraction=_interval_body(event.time_fraction), time_myr=_interval_body(event.time_myr))
    if type(event) is NativePointOccupancyEvent:
        raw.update(plane_event_sequence=event.plane_event_sequence,
            support_indices=list(event.support_indices), plane_event_sequences=list(event.plane_event_sequences))
    return raw


def _body(c):
    return dict(format=FORMAT, model=MODEL, trajectory=dict(
        donor_polygon=None if c.donor_polygon is None else native_polygon_to_dict(c.donor_polygon),
        receiver_polygon=native_polygon_to_dict(c.receiver_polygon),
        donor_omega=list(c.donor_omega), receiver_omega=list(c.receiver_omega), dt_myr=c.dt_myr,
        point=[_fraction_body(value) for value in c.point]),
        precision=c.precision, order=c.order, max_depth=c.max_depth, max_cells=c.max_cells,
        generator_characteristic_polynomial=[_fraction_body(v) for v in c.generator_characteristic_polynomial],
        plane_chronologies=[dict(support_index=plane.support_index, support=native_plane_to_dict(plane.support),
            initial_zero_order=plane.initial_zero_order, coincident=plane.coincident,
            identity_proof="nine_exact_initial_derivatives_zero_cayley_hamilton" if plane.coincident
                else "exact_initial_zero_factor_and_complete_time_cell_partition",
            cells=[dict(time_fraction=_interval_body(cell.time_fraction), status=cell.status, sign=cell.sign,
                gap_bounds=_interval_body(cell.gap_bounds), derivative_bounds=_interval_body(cell.derivative_bounds),
                left_gap_bounds=_interval_body(cell.left_gap_bounds), right_gap_bounds=_interval_body(cell.right_gap_bounds),
                root_fraction=_interval_body(cell.root_fraction)) for cell in plane.cells],
            events=[_event_body(event) for event in plane.events]) for plane in c.plane_chronologies],
        plane_events=[_event_body(event) for event in c.plane_events],
        occupancy_events=[_event_body(event) for event in c.occupancy_events],
        initial_occupancy=c.initial_occupancy, initial_boundary_supports=list(c.initial_boundary_supports),
        processed_cells=c.processed_cells, boundary_convention="closed_native_receiver_halfspaces_with_explicit_boundary_events",
        admission_authorized=False, global_inventory_updated=False,
        scope="complete_original_point_chronology_not_a_swept_area_or_material_transaction")


def validate_native_point_chronology(certificate):
    if type(certificate) is not NativePointChronologyCertificate:
        raise ValueError("Expected a native original point chronology certificate")
    checked = _certify(*_public_inputs(certificate.donor_polygon, certificate.receiver_polygon,
        certificate.donor_omega, certificate.receiver_omega, certificate.dt_myr, certificate.point,
        certificate.precision, certificate.order, certificate.max_depth, certificate.max_cells))
    try:
        same = _encoded(_body(certificate)) == _encoded(_body(checked))
    except (AttributeError, TypeError) as error:
        raise ValueError("Invalid derived native point chronology fields") from error
    if not same:
        raise ValueError("Native point chronology differs from its complete analytic replay")
    return checked


def native_point_chronology_to_dict(certificate):
    body = _body(validate_native_point_chronology(certificate))
    return dict(body, payload_sha256=hashlib.sha256(_encoded(body)).hexdigest())


def native_point_chronology_from_dict(raw):
    if type(raw) is not dict or raw.get("format") != FORMAT or raw.get("model") != MODEL:
        raise ValueError("Invalid native point chronology format")
    try:
        trajectory = raw["trajectory"]
        if (type(trajectory) is not dict or set(trajectory) != {
                "donor_polygon", "receiver_polygon", "donor_omega", "receiver_omega", "dt_myr", "point"}
                or type(trajectory["dt_myr"]) is not float
                or any(type(trajectory[k]) is not list or len(trajectory[k]) != 3
                       or any(type(v) is not float for v in trajectory[k])
                       for k in ("donor_omega", "receiver_omega"))):
            raise ValueError("Invalid complete native point trajectory schema")
        point = trajectory["point"]
        if (type(point) is not list or len(point) != 3 or any(
                type(v) is not list or len(v) != 2 or any(type(x) is not int for x in v)
                or v[1] <= 0 or _fraction_body(Fraction(*v)) != v for v in point)):
            raise ValueError("Native point coordinates require canonical exact rational triples")
        donor = trajectory["donor_polygon"]
        checked = _certify(*_public_inputs(None if donor is None else native_polygon_from_dict(donor),
            native_polygon_from_dict(trajectory["receiver_polygon"]), tuple(trajectory["donor_omega"]),
            tuple(trajectory["receiver_omega"]), trajectory["dt_myr"], tuple(Fraction(*v) for v in point),
            raw["precision"], raw["order"], raw["max_depth"], raw["max_cells"]))
    except KeyError as error:
        raise ValueError("Incomplete native point chronology schema") from error
    if _encoded(raw) != _encoded(native_point_chronology_to_dict(checked)):
        raise ValueError("Saved point chronology differs from complete sealed proof replay")
    return checked


def evaluate_native_point_chronology(certificate, time_fraction):
    c = validate_native_point_chronology(certificate)
    if type(time_fraction) is not Fraction or not 0 <= time_fraction <= 1:
        raise ValueError("Point chronology time query requires an exact Fraction inside [0,1]")
    signs, pending = [], []
    for plane in c.plane_chronologies:
        if plane.coincident:
            signs.append(0)
            continue
        sign = _right_germ(plane)
        for event in plane.events:
            if event.time_fraction.lo <= time_fraction <= event.time_fraction.hi:
                sign = None
                pending.append(event.sequence)
                break
            if event.time_fraction.hi < time_fraction:
                sign = 1 if event.kind == "entry" else -1
            elif event.time_fraction.lo > time_fraction:
                break
        signs.append(sign)
    occupancy = "outside" if -1 in signs else "event_window" if None in signs else "inside"
    return NativePointChronologyQuery(c, hashlib.sha256(_encoded(_body(c))).hexdigest(), time_fraction,
        time_fraction*Fraction.from_float(c.dt_myr), occupancy, tuple(signs),
        tuple(index for index, sign in enumerate(signs) if sign in (0, None)),
        tuple(plane.support_index for plane in c.plane_chronologies if plane.coincident), tuple(sorted(pending)))


def _query_body(query):
    return dict(format=QUERY_FORMAT, model=MODEL, certificate_sha256=query.certificate_sha256,
        time_fraction=_fraction_body(query.time_fraction), time_myr=_fraction_body(query.time_myr),
        occupancy=query.occupancy, plane_signs=list(query.plane_signs),
        active_support_indices=list(query.active_support_indices),
        coincident_support_indices=list(query.coincident_support_indices), event_sequence=list(query.event_sequence),
        admission_authorized=False, global_inventory_updated=False)


def native_point_chronology_query_to_dict(query):
    if type(query) is not NativePointChronologyQuery:
        raise ValueError("Expected an exact native point chronology query")
    reproduced = evaluate_native_point_chronology(query.certificate, query.time_fraction)
    try:
        same = _encoded(_query_body(query)) == _encoded(_query_body(reproduced))
    except (AttributeError, TypeError) as error:
        raise ValueError("Invalid derived native point chronology query fields") from error
    if not same:
        raise ValueError("Native point chronology query differs from its complete original certificate replay")
    return _query_body(reproduced)


def native_point_chronology_query_from_dict(raw, *, certificate):
    if (type(raw) is not dict or raw.get("format") != QUERY_FORMAT
            or raw.get("model") != MODEL):
        raise ValueError("Invalid native point chronology query format")
    try:
        time = raw["time_fraction"]
        if (type(time) is not list or len(time) != 2 or any(type(v) is not int for v in time)
                or time[1] <= 0 or _fraction_body(Fraction(*time)) != time):
            raise ValueError("Point chronology query time must be a canonical exact rational")
        reproduced = evaluate_native_point_chronology(certificate, Fraction(*time))
    except KeyError as error:
        raise ValueError("Incomplete native point chronology query schema") from error
    if _encoded(raw) != _encoded(_query_body(reproduced)):
        raise ValueError("Saved point chronology query differs from complete native point replay")
    return reproduced
