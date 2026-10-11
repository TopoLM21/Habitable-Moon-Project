"""Native birth-carrier frames and causal incoming fronts in one SO(3) clock.

An incoming material mask is distinct from contact-interface measure and from
normal contact activation.  This module preserves that distinction: neither
a geometric overlap nor a source entry selects an unbonded constitutive branch.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal, localcontext
from fractions import Fraction as F
import hashlib
import json
import math

from .geometric_native_birth_contact_support import (
    NativeBirthInterfaceCarrier, native_birth_contact_support_to_dict,
)
from .geometric_native_implicit_epoch_admission import _portable_fingerprint_sha256
from .geometric_native_physical_history import _literal_json, _typed_fingerprint
from .geometric_native_polygons import (
    _cross, _dot, _plane_values, _ray_values, native_arc_to_dict,
    native_plane_to_dict,
)
from .geometric_native_pose import (
    FractionInterval as I, _rounded_interval, interval_atan, interval_sqrt,
)
from .geometric_native_contact_time_support import _trig_ranges
from .geometric_native_physical_pair_time import _rotated, _rotation_change
from .geometric_native_donor_sweep import _normal_coefficients, _bounds as _joint_bounds

FRAME_FORMAT = "native-birth-full-arc-source-frame-1"
FRONT_FORMAT = "native-hot-shared-arc-causal-incoming-front-1"


def _r(value):
    return [value.numerator, value.denominator]


def _i(value):
    return dict(lo=_r(value.lo), hi=_r(value.hi))


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()


def _hash(value):
    return hashlib.sha256(_encoded(value)).hexdigest()


def _unit(vector, precision):
    norm = interval_sqrt(_dot(vector, vector), precision+16)
    if norm.lo <= 0:
        raise ValueError("Native birth frame direction is not proved nonzero")
    return tuple(_rounded_interval(value/norm, precision+16) for value in vector)


def _decimal(value, precision):
    with localcontext() as context:
        context.prec = precision
        return str(Decimal(value.numerator)/Decimal(value.denominator))


@dataclass(frozen=True, slots=True)
class NativeBirthArcFrame:
    carrier: NativeBirthInterfaceCarrier
    precision: int
    start_unit_bounds: tuple
    tangent_unit_bounds: tuple
    horizontal_unit_bounds: tuple
    nominal_start: tuple
    nominal_tangent: tuple
    nominal_horizontal: tuple
    native_angle_bounds_radians: I
    measure_density_m2_per_radian: F
    frame_sha256: str


def _frame_body(frame):
    vectors = (("start", frame.start_unit_bounds, frame.nominal_start),
        ("tangent", frame.tangent_unit_bounds, frame.nominal_tangent),
        ("horizontal", frame.horizontal_unit_bounds, frame.nominal_horizontal))
    return dict(format=FRAME_FORMAT,
        complete_carrier_fingerprint_sha256=_portable_fingerprint_sha256(frame.carrier),
        family_id=frame.carrier.declaration.family_id, precision=frame.precision,
        exact_initial_shared_arc=native_arc_to_dict(frame.carrier.edge),
        receiver_inward_plane=native_plane_to_dict(frame.carrier.receiver_inward_plane),
        vectors={name: dict(native_unit_bounds=[_i(value) for value in bounds],
            nominal_fraction=[_r(value) for value in nominal],
            nominal_decimal=[_decimal(value, frame.precision) for value in nominal],
            nominal_absolute_error=[_r(max(abs(value.lo-center), abs(value.hi-center)))
                for value, center in zip(bounds, nominal)]) for name, bounds, nominal in vectors},
        native_angle_bounds_radians=_i(frame.native_angle_bounds_radians),
        fixed_constitutive_angle_radians=_r(
            frame.carrier.fixed_reference_interface_area_m2/frame.measure_density_m2_per_radian),
        fixed_constitutive_angle_decimal=_decimal(
            frame.carrier.fixed_reference_interface_area_m2/frame.measure_density_m2_per_radian, frame.precision),
        measure_density_m2_per_radian=_r(frame.measure_density_m2_per_radian),
        measure_density_decimal=_decimal(frame.measure_density_m2_per_radian, frame.precision),
        measure="C dtheta; C=R*vertical_height/literal_sin(dip)",
        native_material_area_assigned_to_point=False,
        scope="complete arc frame and native angle bounds; no material-time or constitutive admission")


def native_birth_arc_frame(carrier, precision=256):
    """Return A,B,H for c(theta)=A cos(theta)+B sin(theta), without samples.

    True native unit-vector enclosures and rational nominal coefficients are
    both exported.  Nominal coefficients define a declared constitutive model;
    they are not substituted for native sign or first-entry proofs.
    """
    if type(carrier) is not NativeBirthInterfaceCarrier:
        raise ValueError("A complete source-bound native interface carrier is required")
    if type(precision) is not int or not 128 <= precision <= 512:
        raise ValueError("Native birth frame precision must lie from 128 to 512")
    start = _unit(_ray_values(carrier.edge.start, precision)[1], precision)
    end = _unit(_ray_values(carrier.edge.end, precision)[1], precision)
    horizontal = _unit(_plane_values(carrier.receiver_inward_plane, precision)[1], precision)
    tangent = _cross(horizontal, start)
    orientation = _dot(tangent, end)
    if orientation.hi < 0:
        tangent = tuple(-value for value in tangent)
    elif orientation.lo <= 0:
        raise ValueError("Native full-arc tangent orientation is unresolved")
    tangent = _unit(tangent, precision)
    frame = NativeBirthArcFrame(carrier, precision, start, tangent, horizontal,
        tuple(value.midpoint for value in start), tuple(value.midpoint for value in tangent),
        tuple(value.midpoint for value in horizontal), carrier.native_angle_bounds_radians,
        carrier.inclined_measure_density_m2_per_radian, "")
    return replace(frame, frame_sha256=_hash(_frame_body(frame)))


def native_birth_arc_frame_to_dict(frame):
    if type(frame) is not NativeBirthArcFrame:
        raise ValueError("Expected a complete immutable native birth arc frame")
    replay = native_birth_arc_frame(frame.carrier, frame.precision)
    if _typed_fingerprint(replay) != _typed_fingerprint(frame):
        raise ValueError("Native birth frame differs from complete source-arc replay")
    return json.loads(_encoded(dict(_frame_body(frame), frame_sha256=frame.frame_sha256)))


def native_birth_arc_frame_from_dict(raw, *, expected_carrier):
    _literal_json(raw)
    if type(raw) is not dict or type(raw.get("precision")) is not int:
        raise ValueError("Expected a complete literal native birth-frame object")
    replay = native_birth_arc_frame(expected_carrier, raw["precision"])
    if _encoded(raw) != _encoded(native_birth_arc_frame_to_dict(replay)):
        raise ValueError("Native birth frame differs from independent full carrier replay")
    return replay


@dataclass(frozen=True, slots=True)
class NativeHotSharedArcFront:
    support: object
    frame: NativeBirthArcFrame
    omega_rad_per_myr: tuple
    dt_myr: F
    force_source_sha256: str
    precision: int
    order: int
    receiver_relative_omega_bounds: tuple
    initial_flux_root_angle: I
    full_time_flux_root_angle: I
    flux_angle_derivative_bounds_per_myr: I
    flux_root_time_derivative_bounds: I
    donor_upper_root_angle_bounds: I
    donor_upper_root_time_derivative_bounds: I
    positive_incoming_angle_span_bounds: I
    full_time_incoming_carrier_measure_m2: I
    first_entry_second_derivative_bounds: I
    donor_y_support_bounds: I
    other_receiver_exclusions: tuple
    front_sha256: str


def _motion(records, duration, owners):
    if (type(records) is not tuple
            or any(type(row) is not tuple or len(row) != 2 or type(row[0]) is not int
                or type(row[1]) is not tuple or len(row[1]) != 3
                or any(type(value) is not float or not math.isfinite(value) for value in row[1])
                for row in records)
            or tuple(row[0] for row in records) != owners):
        raise ValueError("Birth front requires all literal physical owner Euler vectors")
    if type(duration) is float:
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("Birth front duration must be positive")
        duration = F.from_float(duration)
    if type(duration) is not F or duration <= 0:
        raise ValueError("Birth front duration requires an explicit positive exact clock")
    return {owner: tuple(F.from_float(value) for value in vector) for owner, vector in records}, duration


def _carrier(frame, theta, precision):
    sine, cosine = _trig_ranges(theta.lo, theta.hi, precision+16)
    return tuple(_rounded_interval(a*cosine+b*sine, precision+16)
        for a, b in zip(frame.start_unit_bounds, frame.tangent_unit_bounds))


def _relative(vector, donor_axis, receiver_axis, time, precision, order):
    return _relative_series_bounds(vector, donor_axis, receiver_axis, time, precision, order, 0)


def _relative_derivative(vector, donor_axis, receiver_axis, time, precision, order):
    return _relative_series_bounds(vector, donor_axis, receiver_axis, time, precision, order, 1)


def _relative_series_bounds(vector, donor_axis, receiver_axis, time, precision, order, degree):
    # Convolve both exact generators before enclosing their shared clock.
    # Enclosing exp(-d*u) and exp(r*u) separately would treat the same u as
    # independent and lose the physical cancellation of large common spin.
    columns = tuple(_normal_coefficients(tuple(F(int(axis == component)) for axis in range(3)),
        donor_axis, receiver_axis, order) for component in range(3))
    coefficients = tuple(tuple(sum((columns[column][k][axis]*vector[column]
        for column in range(3)), I(0)) for axis in range(3)) for k in range(order+1))
    rate = sum(map(abs, donor_axis), F(0))+sum(map(abs, receiver_axis), F(0))
    scale = sum((max(abs(value.lo), abs(value.hi)) if type(value) is I else abs(value)
        for value in vector), F(0))
    return tuple(_joint_bounds(tuple(row[axis] for row in coefficients), time,
        degree, rate, scale, precision+16) for axis in range(3))


def _front_body(front):
    return dict(format=FRONT_FORMAT, source_support_sha256=front.support.support_sha256,
        complete_source_support_fingerprint_sha256=_portable_fingerprint_sha256(front.support),
        source_epoch_sha256=front.support.source_epoch.epoch_sha256,
        complete_source_epoch_fingerprint_sha256=_portable_fingerprint_sha256(front.support.source_epoch),
        frame=native_birth_arc_frame_to_dict(front.frame), force_source_sha256=front.force_source_sha256,
        omega_rad_per_myr=[[owner, list(vector)] for owner, vector in front.omega_rad_per_myr],
        exact_dt_myr=_r(front.dt_myr), precision=front.precision, order=front.order,
        receiver_relative_omega_bounds=[_i(value) for value in front.receiver_relative_omega_bounds],
        initial_flux_root_angle=_i(front.initial_flux_root_angle),
        full_time_flux_root_angle=_i(front.full_time_flux_root_angle),
        flux_angle_derivative_bounds_per_myr=_i(front.flux_angle_derivative_bounds_per_myr),
        flux_root_time_derivative_bounds=_i(front.flux_root_time_derivative_bounds),
        donor_upper_root_angle_bounds=_i(front.donor_upper_root_angle_bounds),
        donor_upper_root_time_derivative_bounds=_i(front.donor_upper_root_time_derivative_bounds),
        positive_incoming_angle_span_bounds=_i(front.positive_incoming_angle_span_bounds),
        full_time_incoming_carrier_measure_m2=_i(front.full_time_incoming_carrier_measure_m2),
        first_entry_second_derivative_bounds=_i(front.first_entry_second_derivative_bounds),
        donor_y_support_bounds=_i(front.donor_y_support_bounds),
        other_receiver_exclusions=[dict(component_id=identity, piece_sequence=piece,
            separating_plane_sequence=plane, whole_clock_support_bounds=_i(bounds))
            for identity, piece, plane, bounds in front.other_receiver_exclusions],
        incoming_mask=dict(angle_lower="atan(-By(u)/Bz(u))", angle_upper="min(native_L, atan(gamma(u)/(-delta(u))))",
            strict_angle_inequalities=True, time_fraction="0<u<=1",
            By_Bz="exp(-Omega_receiver*dt*u)(Omega_donor-Omega_receiver)",
            gamma_delta="donor_upper_plane dot exp(-Omega_donor*dt*u)exp(Omega_receiver*dt*u)(A,B)",
            source_domain="the complete actual prior-live HOT donor mask, without a polygon proxy substitution",
            first_entry="fixed material ray has strictly concave receiving-plane support; its positive entry derivative implies every earlier time is outside this receiver; every other receiver component is separately excluded"),
        root_box_midpoint_used_as_event=False, zero_flux_curve_has_zero_contact_line_measure=True,
        interface_measure_is_material_archive_area=False,
        normal_constitutive_branch_selected_by_geometry=False,
        radial_ordering_and_three_dimensional_collision_certified=False,
        complete_all_pairs_temporal_coverage=False,
        scope="complete causal positive incoming mask for the declared HOT6/HOT2 shared arc; mechanical permission remains separate")


def certify_native_hot_shared_arc_front(support, family_id, omega_rad_per_myr,
        dt_myr, *, force_source_sha256, precision=256, order=32):
    """Prove a moving positive incoming carrier and its full-time first entry.

    The source-specific axis geometry is checked explicitly.  For every time
    u, the open incoming interval is bounded below by the unique positive-flux
    root and above by the true HOT donor-support root.  All curves remain
    analytic; interval endpoints are enclosures, never selected event times.
    """
    native_birth_contact_support_to_dict(support)
    if (type(force_source_sha256) is not str or len(force_source_sha256) != 64
            or any(value not in "0123456789abcdef" for value in force_source_sha256)):
        raise ValueError("The birth front requires the independent genuine force-source digest")
    if type(order) is not int or not 16 <= order <= 40:
        raise ValueError("Birth-front Taylor order must lie from 16 to 40")
    rows = tuple(row for row in support.interface_carriers if row.declaration.family_id == family_id)
    if len(rows) != 1:
        raise ValueError("The declared birth family must bind one complete source carrier")
    carrier = rows[0]
    donor, receiver = carrier.donor_component, carrier.receiver_component
    if ((donor.material_kind, receiver.material_kind) != ("hot_birth", "hot_birth")
            or (donor.original_parent_id, receiver.original_parent_id) != ("octant:6", "octant:2")
            or (donor.owner, receiver.owner) != (1, 0)):
        raise ValueError("This causal source proof requires the actual HOT6/HOT2 material family")
    frame = native_birth_arc_frame(carrier, precision)
    axis_frame = ((F(0), F(0), F(-1)), (F(0), F(1), F(0)), (F(-1), F(0), F(0)))
    for bounds, expected in zip((frame.start_unit_bounds, frame.tangent_unit_bounds, frame.horizontal_unit_bounds), axis_frame):
        if any(value != I(coordinate) for value, coordinate in zip(bounds, expected)):
            raise ValueError("The HOT shared carrier is not the exact unrotated polar X arc")
    epoch = support.source_epoch
    if any(row.owner == receiver.owner for row in epoch.paired_corrections):
        raise ValueError("A signed receiver correction requires an additional exact material exclusion proof")
    if any(donor.component_id in (row.negative_component_id, row.positive_component_id)
            for row in epoch.paired_corrections):
        raise ValueError("A signed HOT donor correction requires its exact support, not its proxy")
    owners = tuple(owner for owner, _ in epoch.additional_owner_histories)
    omega, duration = _motion(omega_rad_per_myr, dt_myr, owners)
    donor_axis = tuple(value*duration for value in omega[donor.owner])
    receiver_axis = tuple(value*duration for value in omega[receiver.owner])
    relative_omega = tuple(a-b for a, b in zip(omega[donor.owner], omega[receiver.owner]))
    clock = I(0, 1)
    B = _rotated(relative_omega, tuple(-value for value in receiver_axis), clock, precision, order)
    if B[1].hi >= 0 or B[2].lo <= 0:
        raise ValueError("The actual fresh motion does not prove one interior polar flux root")
    root = interval_atan(-B[1]/B[2], precision+16)
    initial = interval_atan(I(-relative_omega[1]/relative_omega[2]), precision+16)
    length = frame.native_angle_bounds_radians
    if not 0 < root.lo <= root.hi < length.lo:
        raise ValueError("The moving flux root is not proved strictly inside the full native arc")
    sine, cosine = _trig_ranges(F(0), length.hi, precision+16)
    ftheta = -B[1]*sine+B[2]*cosine
    if ftheta.lo <= 0:
        raise ValueError("Whole-arc incoming flux is not proved strictly increasing in angle")
    Bprime = tuple(-value for value in _cross(receiver_axis, B))
    root_derivative = (-Bprime[1]*B[2]+B[1]*Bprime[2])/(B[1]*B[1]+B[2]*B[2])
    if root_derivative.lo <= 0:
        raise ValueError("The moving source root is not proved monotone on the actual clock")
    dg = donor.proxy_geometries[carrier.declaration.donor_piece_sequence]
    if (len(dg.support_planes) != 3
            or not any(plane.normal == (F(1), F(0), F(0)) and not plane.rotations for plane in dg.support_planes)
            or not any(plane.normal == (F(0), F(1), F(0)) and not plane.rotations for plane in dg.support_planes)):
        raise ValueError("The full actual HOT donor mask requires additional support inequalities")
    upper_planes = tuple(plane for plane in dg.support_planes
        if plane.normal == (F(-1), F(0), F(0)) and plane.rotations)
    if len(upper_planes) != 1:
        raise ValueError("The actual HOT donor has no unique inherited upper support plane")
    upper_normal = _plane_values(upper_planes[0], precision)[1]
    upper_normal = tuple(_rounded_interval(value, precision+16) for value in upper_normal)
    A, E = axis_frame[:2]
    qA, qE = (_relative(vector, donor_axis, receiver_axis, clock, precision, order) for vector in (A, E))
    gamma, delta = _dot(upper_normal, qA), _dot(upper_normal, qE)
    if gamma.lo <= 0 or delta.hi >= 0:
        raise ValueError("The complete HOT donor upper support has an unresolved angle root")
    upper_root = interval_atan(gamma/(-delta), precision+16)
    gamma_prime, delta_prime = (_dot(upper_normal,
        _relative_derivative(vector, donor_axis, receiver_axis, clock, precision, order)) for vector in (A, E))
    upper_derivative = (-gamma_prime*delta+gamma*delta_prime)/(gamma*gamma+delta*delta)
    # The true upper boundary may move in either direction.  Its explicit
    # minimum with native_L is retained instead of assuming a direction.
    span = I(min(length.lo, upper_root.lo)-root.hi,
        min(length.hi, upper_root.hi)-root.lo)
    if span.lo <= 0:
        raise ValueError("No positive full-time HOT incoming carrier span was proved")
    theta = I(root.lo, length.hi)
    c = _carrier(frame, theta, precision)
    q = _relative(c, donor_axis, receiver_axis, clock, precision, order)
    if q[1].lo <= 0:
        raise ValueError("The full actual HOT donor Y support is not strictly positive")
    # q is held fixed while v ranges independently over the whole physical
    # clock.  At its actual entry u, h(u)=0 and h'(u)>0 by the analytic mask.
    # Strict h''(v)<0 therefore proves h(v)<0 for every 0<=v<u.
    world_donor = _rotated(q, donor_axis, clock, precision, order)
    world_normal = _rotated(axis_frame[2], receiver_axis, clock, precision, order)
    relative_axis = tuple(value*duration for value in relative_omega)
    first_term = _cross(_cross(donor_axis, world_donor), world_normal)
    second_term = _cross(world_donor, _cross(receiver_axis, world_normal))
    concavity = _dot(relative_axis, tuple(a+b for a, b in zip(first_term, second_term)))
    if concavity.hi >= 0:
        raise ValueError("Complete-clock receiving support is not proved strictly concave")
    receiver_coordinates = _relative(q, receiver_axis, donor_axis, clock, precision, order)
    exclusions = []
    for component in epoch.components:
        if component.owner != receiver.owner or component.component_id == receiver.component_id:
            continue
        for piece_sequence, polygon in enumerate(component.proxy_geometries):
            candidates = tuple((index, _dot(_plane_values(plane, precision)[1], receiver_coordinates))
                for index, plane in enumerate(polygon.support_planes))
            proved = tuple((index, bound) for index, bound in candidates if bound.hi < 0)
            if not proved:
                raise ValueError("An earlier entry into another current receiver component remains unresolved: "
                    +component.original_parent_id+" "+component.material_kind+" "
                    +str(tuple((index, float(bound.lo), float(bound.hi)) for index, bound in candidates)))
            plane, bound = min(proved, key=lambda item: item[1].hi)
            exclusions.append((component.component_id, piece_sequence, plane,
                _rounded_interval(bound, precision+16)))
    rounded = lambda value: _rounded_interval(value, precision+16)
    value = NativeHotSharedArcFront(support, frame, omega_rad_per_myr, duration,
        force_source_sha256, precision, order, tuple(map(rounded, B)), rounded(initial),
        rounded(root), rounded(ftheta), rounded(root_derivative), rounded(upper_root),
        rounded(upper_derivative), rounded(span), rounded(frame.measure_density_m2_per_radian*span),
        rounded(concavity), rounded(q[1]), tuple(exclusions), "")
    return replace(value, front_sha256=_hash(_front_body(value)))


def native_hot_shared_arc_front_to_dict(front):
    if type(front) is not NativeHotSharedArcFront:
        raise ValueError("Expected a complete immutable native HOT incoming front")
    replay = certify_native_hot_shared_arc_front(front.support, front.frame.carrier.declaration.family_id,
        front.omega_rad_per_myr, front.dt_myr, force_source_sha256=front.force_source_sha256,
        precision=front.precision, order=front.order)
    if _typed_fingerprint(replay) != _typed_fingerprint(front):
        raise ValueError("Causal birth front differs from full material-source and clock replay")
    return json.loads(_encoded(dict(_front_body(front), front_sha256=front.front_sha256)))


def native_hot_shared_arc_front_from_dict(raw, *, expected_support, expected_family_id,
        expected_omega_rad_per_myr, expected_dt_myr, expected_force_source_sha256):
    _literal_json(raw)
    if type(raw) is not dict or type(raw.get("precision")) is not int or type(raw.get("order")) is not int:
        raise ValueError("Expected a complete literal native HOT front object")
    replay = certify_native_hot_shared_arc_front(expected_support, expected_family_id,
        expected_omega_rad_per_myr, expected_dt_myr, force_source_sha256=expected_force_source_sha256,
        precision=raw["precision"], order=raw["order"])
    if _encoded(raw) != _encoded(native_hot_shared_arc_front_to_dict(replay)):
        raise ValueError("Causal birth front differs from independently supplied source, actual force and clock")
    return replay


def native_hot_shared_arc_front_time_bounds(front, time_fraction_bounds):
    """Enclose both actual moving roots on a requested complete time window."""
    native_hot_shared_arc_front_to_dict(front)
    return _front_time_bounds(front, time_fraction_bounds)


def _front_time_bounds(front, time_fraction_bounds):
    if type(front) is not NativeHotSharedArcFront or type(time_fraction_bounds) is not I:
        raise ValueError("A typed causal front and exact time interval are required")
    if not 0 <= time_fraction_bounds.lo <= time_fraction_bounds.hi <= 1:
        raise ValueError("A birth-front time query must stay on its actual clock")
    omega = dict((owner, tuple(F.from_float(value) for value in vector))
        for owner, vector in front.omega_rad_per_myr)
    donor, receiver = front.frame.carrier.donor_component, front.frame.carrier.receiver_component
    donor_axis = tuple(value*front.dt_myr for value in omega[donor.owner])
    receiver_axis = tuple(value*front.dt_myr for value in omega[receiver.owner])
    difference = tuple(a-b for a, b in zip(omega[donor.owner], omega[receiver.owner]))
    B = _rotated(difference, tuple(-value for value in receiver_axis), time_fraction_bounds,
        front.precision, front.order)
    root = interval_atan(-B[1]/B[2], front.precision+16)
    polygon = donor.proxy_geometries[front.frame.carrier.declaration.donor_piece_sequence]
    plane = next(plane for plane in polygon.support_planes
        if plane.normal == (F(-1), F(0), F(0)) and plane.rotations)
    normal = tuple(_rounded_interval(value, front.precision+16)
        for value in _plane_values(plane, front.precision)[1])
    A, E = front.frame.nominal_start, front.frame.nominal_tangent
    gamma, delta = (_dot(normal, _relative(vector, donor_axis, receiver_axis,
        time_fraction_bounds, front.precision, front.order)) for vector in (A, E))
    upper = interval_atan(gamma/(-delta), front.precision+16)
    return dict(receiver_relative_omega_bounds=B, flux_root_angle=root,
        donor_upper_root_angle=upper,
        incoming_angle_span=I(min(front.frame.native_angle_bounds_radians.lo, upper.lo)-root.hi,
            min(front.frame.native_angle_bounds_radians.hi, upper.hi)-root.lo))


@dataclass(frozen=True, slots=True)
class NativeHotIncomingRectangle:
    front: NativeHotSharedArcFront
    angle_bounds_radians: I
    time_fraction_bounds: I
    flux_bounds_per_myr: I
    positive_interface_carrier_measure_m2: F
    positive_clock_carrier_measure_m2: F


def certify_native_hot_incoming_rectangle(front, angle_bounds_radians, time_fraction_bounds):
    """Certify an entire positive first-entry subdomain, with its own C measure.

    A rectangle touching a moving root is not assigned a root midpoint or a
    sign.  The exact moving mask remains authoritative for the omitted band.
    """
    native_hot_shared_arc_front_to_dict(front)
    if type(angle_bounds_radians) is not I or angle_bounds_radians.width <= 0:
        raise ValueError("A positive whole-carrier angle interval is required")
    if type(time_fraction_bounds) is not I or time_fraction_bounds.width <= 0:
        raise ValueError("A positive exact physical-clock window is required")
    bounds = _front_time_bounds(front, time_fraction_bounds)
    if (angle_bounds_radians.lo <= bounds["flux_root_angle"].hi
            or angle_bounds_radians.hi >= min(front.frame.native_angle_bounds_radians.lo,
                bounds["donor_upper_root_angle"].lo)):
        raise ValueError("The whole rectangle is not strictly inside the causal moving incoming mask")
    sine, cosine = _trig_ranges(angle_bounds_radians.lo, angle_bounds_radians.hi, front.precision+16)
    B = bounds["receiver_relative_omega_bounds"]
    flux = _rounded_interval(B[1]*cosine+B[2]*sine, front.precision+16)
    if flux.lo <= 0:
        raise ValueError("The complete incoming rectangle has an unresolved or nonpositive flux")
    area = front.frame.measure_density_m2_per_radian*angle_bounds_radians.width
    return NativeHotIncomingRectangle(front, angle_bounds_radians, time_fraction_bounds,
        flux, area, area*time_fraction_bounds.width)


def native_hot_incoming_rectangle_to_dict(rectangle):
    if type(rectangle) is not NativeHotIncomingRectangle:
        raise ValueError("Expected a complete immutable incoming carrier rectangle")
    replay = certify_native_hot_incoming_rectangle(rectangle.front,
        rectangle.angle_bounds_radians, rectangle.time_fraction_bounds)
    if _typed_fingerprint(replay) != _typed_fingerprint(rectangle):
        raise ValueError("Incoming rectangle differs from exact source, moving curves and clock replay")
    return dict(front_sha256=rectangle.front.front_sha256,
        angle_bounds_radians=_i(rectangle.angle_bounds_radians),
        time_fraction_bounds=_i(rectangle.time_fraction_bounds),
        flux_bounds_per_myr=_i(rectangle.flux_bounds_per_myr),
        positive_interface_carrier_measure_m2=_r(rectangle.positive_interface_carrier_measure_m2),
        positive_clock_carrier_measure_m2=_r(rectangle.positive_clock_carrier_measure_m2),
        earliest_owner_entry_proved=True, whole_source_donor_and_receiver_masks_bound=True,
        interface_measure_used_as_archive_area=False,
        normal_activation_or_down_dip_permission_inferred=False)


def certify_native_birth_radial_ordering(*, donor_component_id, receiver_component_id,
        donor_surface_displacement_bounds_m, receiver_surface_displacement_bounds_m,
        receiver_solid_layer_thickness_bounds_m, source_force_sha256, field_scope):
    """A strict full-footprint vertical layer-separation hook for the new law.

    The caller must provide source-authenticated bounds on the entire material
    footprints and physical clock.  Point values and free normal-contact gaps
    are insufficient.  Passing this vertical test supplies no horizontal mask
    coverage, mechanical branch, or complete three-dimensional trajectory.
    """
    if field_scope != "entire_material_footprints_and_physical_clock":
        raise ValueError("Radial ordering requires whole footprints and the physical clock")
    if (type(donor_component_id) is not str or not donor_component_id
            or type(receiver_component_id) is not str or not receiver_component_id
            or donor_component_id == receiver_component_id):
        raise ValueError("Radial ordering requires two explicit material component IDs")
    if (type(source_force_sha256) is not str or len(source_force_sha256) != 64
            or any(value not in "0123456789abcdef" for value in source_force_sha256)):
        raise ValueError("Radial ordering requires an independent force-source digest")
    values = (donor_surface_displacement_bounds_m, receiver_surface_displacement_bounds_m,
        receiver_solid_layer_thickness_bounds_m)
    if any(type(value) is not I for value in values) or receiver_solid_layer_thickness_bounds_m.lo <= 0:
        raise ValueError("Physical layer ordering requires exact displacement bounds and positive actual thickness")
    clearance = (receiver_surface_displacement_bounds_m-receiver_solid_layer_thickness_bounds_m
        -donor_surface_displacement_bounds_m)
    if clearance.lo <= 0:
        raise ValueError("The complete receiver layer bottom was not proved above the donor surface")
    return dict(donor_component_id=donor_component_id, receiver_component_id=receiver_component_id,
        source_force_sha256=source_force_sha256, field_scope=field_scope,
        strict_vertical_clearance_m=_i(clearance),
        full_three_dimensional_trajectory_collision_certified=False,
        free_contact_gap_substituted_for_layer_thickness=False)
