"""Full SO(3) bounds for the declared birth-contact anchor fields.

Normal signs at a tiny HOT contact cannot be inferred from the old banks'
absolute anchor tolerance.  These fields retain the actual owner rotations,
radial displacements and entire native arc.  They evaluate contact anchors,
not a complete three-dimensional solid collision or material admission.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction as F
import hashlib
import json

from .geometric_native_pose import FractionInterval as I, _rounded_interval
from .geometric_native_physical_pair_time import _relative_change


FORMAT = "native-born-full-SO3-contact-anchor-fields-1"


def _dot(left, right):
    return sum((a*b for a, b in zip(left, right)), I(0))


@dataclass(frozen=True, slots=True)
class BornFinitePoseFields:
    gap_coefficients: tuple
    jump_coefficients: tuple
    radial_ordering_coefficients: tuple
    time_fraction_bounds: I
    precision: int
    order: int

    def __post_init__(self):
        if any(type(row) is not tuple or len(row) != 5 or any(type(value) is not I for value in row)
                for row in (self.gap_coefficients, self.jump_coefficients, self.radial_ordering_coefficients)):
            raise ValueError("A finite-pose field requires exactly five immutable interval coefficients")
        if (type(self.time_fraction_bounds) is not I
                or not 0 <= self.time_fraction_bounds.lo <= self.time_fraction_bounds.hi <= 1):
            raise ValueError("A finite-pose field clock must remain in the actual interval")
        if type(self.precision) is not int or self.precision < 64 or type(self.order) is not int or self.order < 4:
            raise ValueError("A finite-pose field requires a literal adequate Taylor budget")


def bound_born_finite_pose_fields(*, start_unit_bounds, tangent_unit_bounds,
        horizontal_unit_bounds, donor_axis, receiver_axis, radius_m,
        donor_radial_before_m, receiver_radial_before_m, donor_radial_increment_m,
        receiver_radial_increment_m, sine_dip, cosine_dip, birth_gap_free_m,
        birth_jump_free_m, time_fraction_bounds, precision=256, order=28):
    """Bound every native angle and time using a full degree-two field.

The radial path is explicitly affine in the step fraction. Owner poses are
the ordered SO(3) path, evaluated in the receiving frame without subtracting
two unit vectors. The angular axes are the literal used Omega times dt.
"""
    vectors = (start_unit_bounds, tangent_unit_bounds, horizontal_unit_bounds)
    if any(type(row) is not tuple or len(row) != 3 or any(type(value) is not I for value in row)
            for row in vectors):
        raise ValueError("Finite-pose fields require full native unit-vector interval triples")
    if any(type(axis) is not tuple or len(axis) != 3 or any(type(value) is not F for value in axis)
            for axis in (donor_axis, receiver_axis)):
        raise ValueError("Finite-pose axes must be exact literal-path rational triples")
    numbers = (radius_m, donor_radial_before_m, receiver_radial_before_m,
        donor_radial_increment_m, receiver_radial_increment_m, sine_dip, cosine_dip,
        birth_gap_free_m, birth_jump_free_m)
    if any(type(value) is not F for value in numbers) or radius_m <= 0:
        raise ValueError("Finite-pose declarations require exact rational physical values")
    if type(time_fraction_bounds) is not I or not 0 <= time_fraction_bounds.lo <= time_fraction_bounds.hi <= 1:
        raise ValueError("Finite-pose fields need a complete time window in the actual interval")
    if type(precision) is not int or precision < 64 or type(order) is not int or order < 4:
        raise ValueError("Finite-pose bounds require an explicit adequate Taylor budget")
    a, b, horizontal = vectors
    da = _relative_change(a, donor_axis, receiver_axis, time_fraction_bounds, precision, order)
    db = _relative_change(b, donor_axis, receiver_axis, time_fraction_bounds, precision, order)
    aa, bb = _dot(a, da), _dot(b, db)
    ab = _dot(a, db)+_dot(b, da)
    radial_change = ((aa+bb)/2, I(0), I(0), (aa-bb)/2, ab/2)
    horizontal_change = (I(0), _dot(horizontal, da), _dot(horizontal, db), I(0), I(0))
    donor_w = I(donor_radial_before_m)+time_fraction_bounds*donor_radial_increment_m
    receiver_w = I(receiver_radial_before_m)+time_fraction_bounds*receiver_radial_increment_m
    displacement = time_fraction_bounds*(donor_radial_increment_m-receiver_radial_increment_m)
    lever = I(radius_m)+donor_w
    gap, jump, ordering = [], [], []
    for index, (r, h) in enumerate(zip(radial_change, horizontal_change)):
        g = -lever*(cosine_dip*r+sine_dip*h)
        j = lever*(sine_dip*r-cosine_dip*h)
        z = -lever*r
        if index == 0:
            g += I(birth_gap_free_m)-cosine_dip*displacement
            j += I(birth_jump_free_m)+sine_dip*displacement
            z += receiver_w-donor_w
        gap.append(_rounded_interval(g, precision))
        jump.append(_rounded_interval(j, precision))
        ordering.append(_rounded_interval(z, precision))
    return BornFinitePoseFields(tuple(gap), tuple(jump), tuple(ordering), time_fraction_bounds, precision, order)


def born_finite_pose_angle_bounds(fields, angle_bounds):
    """Enclose each complete field on an exact native angle interval."""
    from .geometric_native_contact_time_support import _trig_ranges
    from .geometric_native_pose import interval_pi
    if type(fields) is not BornFinitePoseFields or type(angle_bounds) is not I or angle_bounds.lo < 0:
        raise ValueError("Finite-pose evaluation requires typed fields and a native angle interval")
    if angle_bounds.hi >= interval_pi(fields.precision).lo:
        raise ValueError("Finite-pose angle bounds must describe a proved positive minor arc")
    sine, cosine = _trig_ranges(angle_bounds.lo, angle_bounds.hi, fields.precision)
    sine2, cosine2 = _trig_ranges(2*angle_bounds.lo, 2*angle_bounds.hi, fields.precision)
    return tuple(_rounded_interval(row[0]+cosine*row[1]+sine*row[2]+cosine2*row[3]+sine2*row[4], fields.precision)
        for row in (fields.gap_coefficients, fields.jump_coefficients, fields.radial_ordering_coefficients))


def _interval(value):
    return {name: [number.numerator, number.denominator]
        for name, number in (("lo", value.lo), ("hi", value.hi))}


def native_birth_candidate_finite_pose_to_dict(candidate, *, mesh, previous, load,
        support, time_fraction_bounds=I(1), precision=256, order=28):
    """Authenticate the current force, then evaluate its own native anchors."""
    import math
    from .geometric_native_birth_continuum_motion import (
        native_birth_continuum_candidate_to_dict, build_native_birth_continuum_inputs,
        native_birth_continuum_frame)
    from .thermal import SECONDS_PER_MYR
    source = native_birth_continuum_candidate_to_dict(candidate, mesh=mesh, previous=previous, load=load, support=support)
    inputs, _ = build_native_birth_continuum_inputs(mesh, previous, load, support,
        candidate.dt_s, decimal_precision=candidate.network_step.decimal_precision)
    dt = F.from_float(candidate.dt_s/SECONDS_PER_MYR)
    omega = {owner: tuple(F.from_float(value)*dt for value in values)
        for owner, values in candidate.omega_rad_per_myr}
    before = tuple(F.from_float(value) for value in inputs["coordinates_before"])
    after = tuple(F(Decimal(value)) for value in candidate.network_step.coordinates_decimal)
    radius = F.from_float(inputs["radius_m"])
    old_radial = {identity: (sub, over) for identity, sub, over in previous.transverse_displacement_m}
    actual_before = {}
    for index, trace in enumerate(previous.initial_mechanics.fault_traces.traces):
        for bank, value in enumerate(old_radial[trace.trace_id]):
            actual_before[inputs["rotation_dof_count"]+2*index+bank] = F.from_float(value)
    for foundation in inputs["born_foundations"]:
        actual_before[foundation.radial_dof_index] = F.from_float(foundation.previous_state.radial_displacement_m)
    rows = []
    for carrier, attachment in zip(support.interface_carriers, inputs["born_continuum_attachments"]):
        declaration = attachment.declaration
        # A canonical operator has exactly two radial ports in its constant
        # normal row; angular ports have no constant coefficient on this arc.
        constant = tuple(Decimal(value) for value in declaration.normal_operator_coefficients[0])
        sub = tuple(index for index, value in enumerate(constant) if value < 0)
        over = tuple(index for index, value in enumerate(constant) if value > 0)
        if len(sub) != 1 or len(over) != 1:
            raise ValueError("Canonical finite-pose birth fields lost their two radial banks")
        sub, over = sub[0], over[0]
        a, b, h = native_birth_continuum_frame(carrier, precision=precision)
        fields = bound_born_finite_pose_fields(start_unit_bounds=a, tangent_unit_bounds=b,
            horizontal_unit_bounds=h, donor_axis=omega[carrier.donor_component.owner],
            receiver_axis=omega[carrier.receiver_component.owner], radius_m=radius,
            donor_radial_before_m=actual_before[sub], receiver_radial_before_m=actual_before[over],
            donor_radial_increment_m=after[sub]*radius-actual_before[sub],
            receiver_radial_increment_m=after[over]*radius-actual_before[over],
            sine_dip=F.from_float(math.sin(carrier.declaration.dip_radians)),
            cosine_dip=F.from_float(math.cos(carrier.declaration.dip_radians)),
            birth_gap_free_m=F.from_float(declaration.reference.birth_gap_free_m),
            birth_jump_free_m=F.from_float(declaration.reference.birth_jump_free_m),
            time_fraction_bounds=time_fraction_bounds, precision=precision, order=order)
        angle = I(0, carrier.native_angle_bounds_radians.hi)
        gap, jump, ordering = born_finite_pose_angle_bounds(fields, angle)
        rows.append(dict(family_id=declaration.reference.family_id,
            gap_coefficients=[_interval(value) for value in fields.gap_coefficients],
            jump_coefficients=[_interval(value) for value in fields.jump_coefficients],
            radial_ordering_coefficients=[_interval(value) for value in fields.radial_ordering_coefficients],
            native_angle_bounds=_interval(angle), whole_arc_gap_bounds_m=_interval(gap),
            whole_arc_jump_bounds_m=_interval(jump), whole_arc_radial_ordering_bounds_m=_interval(ordering),
            positive_radial_ordering_means="receiving ray radius exceeds projected donating ray radius",
            radial_before_coordinate_alias_bounds_m={str(index):
                [error.numerator, error.denominator] for index in (sub, over)
                for error in (abs(before[index]*radius-actual_before[index]),)},
            affine_force_gap_coefficients_decimal=next(step.state.gap_coefficients_decimal
                for step in candidate.network_step.continuum_contact_steps
                if step.state.declaration.reference.family_id == declaration.reference.family_id),
            finite_pose_constitutive_force_equilibrium_proved=False,
            three_dimensional_solid_collision_excluded=False,
            material_time_admission_authorized=False))
    body = dict(format=FORMAT, force_candidate_sha256=source["candidate_sha256"],
        source_epoch_sha256=support.source_epoch.epoch_sha256,
        time_fraction_bounds=_interval(time_fraction_bounds), precision=precision, order=order,
        radial_path="explicit affine interpolation of authoritative Decimal radial coordinates",
        owner_path="full SO3 of the candidate's literal used Omega and dt",
        fields=rows, accepted_new_native_intervals=0)
    body["fields_sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return body
