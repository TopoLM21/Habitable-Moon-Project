"""Independent SO(3) material Jacobian and interface-measure controls."""
from copy import deepcopy
from dataclasses import replace
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import json

import pytest

from tectonics.geometric_native_first_entry_contact_measure import (
    certify_native_first_entry_contact_measure,
    native_first_entry_contact_measure_from_dict,
    native_first_entry_contact_measure_to_dict,
    validate_native_first_entry_contact_measure,
)
from tectonics.geometric_native_pose import FractionInterval as I, interval_pi
from tests.test_geometric_native_physical_pair_time import decimal, rotate


F = Fraction
ZERO = (F(0),) * 3


def source(**changes):
    value = dict(trace_id="declared-original-interface", start=(F(0), F(1), F(0)),
        end=(F(0), F(0), F(1)), donor_owner=1, receiver_owner=0,
        omega_records=((0, (F(1, 17), F(-2, 19), F(3, 23))),
                       (1, (F(-4, 29), F(5, 31), F(-6, 37)))),
        dt_myr=F(1, 41), native_angle_bounds=I(F(3, 7), F(7, 11)),
        time_fraction_bounds=I(F(1, 13), F(5, 7)),
        source_extent_radians=F.from_float(1.5707963267948966),
        source_interface_area_density_m2_per_radian=F(1234567, 89),
        precision=160, order=32, required_jacobian_sign=None)
    value.update(changes)
    return value


def sine_cosine(angle):
    sine, cosine = angle, Decimal(1)
    sinterm, costerm = angle, Decimal(1)
    for index in range(1, 1000):
        sinterm *= -angle * angle / Decimal((2*index)*(2*index+1))
        costerm *= -angle * angle / Decimal((2*index-1)*(2*index))
        sine += sinterm
        cosine += costerm
        if max(abs(sinterm), abs(costerm)) < Decimal("1e-560"):
            return sine, cosine
    raise AssertionError("Independent carrier trigonometry failed to converge")


def atan_small(value):
    result = term = value
    for index in range(1, 3000):
        term *= -value * value
        addition = term / Decimal(2*index+1)
        result += addition
        if abs(addition) < Decimal("1e-560"):
            return result
    raise AssertionError("Independent pi series failed to converge")


def cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def dot(a, b):
    return sum((x*y for x, y in zip(a, b)), Decimal(0))


def oracle(parameters, theta, time):
    """Differentiate the two rotations separately, then form the triple product."""
    with localcontext() as context:
        context.prec = 600
        sine, cosine = sine_cosine(decimal(theta))
        start, end = (tuple(map(decimal, parameters[name])) for name in ("start", "end"))
        carrier = tuple(s*cosine+e*sine for s, e in zip(start, end))
        tangent = tuple(-s*sine+e*cosine for s, e in zip(start, end))
        axes = {owner: tuple(decimal(x)*decimal(parameters["dt_myr"]) for x in omega)
                for owner, omega in parameters["omega_records"]}
        donor_axis, receiver_axis = axes[parameters["donor_owner"]], axes[parameters["receiver_owner"]]
        fraction = decimal(time)
        donor_rotation = tuple(-fraction*x for x in donor_axis)
        receiver_rotation = tuple(fraction*x for x in receiver_axis)
        world, world_tangent = rotate(carrier, receiver_rotation), rotate(tangent, receiver_rotation)
        material = rotate(world, donor_rotation)
        material_tangent = rotate(world_tangent, donor_rotation)
        derivative_receiver = rotate(cross(receiver_axis, world), donor_rotation)
        derivative_donor = cross(tuple(-x for x in donor_axis), material)
        material_velocity = tuple(a+b for a, b in zip(derivative_donor, derivative_receiver))
        signed = dot(material, cross(material_tangent, material_velocity))
        return dict(carrier_direction=carrier, carrier_tangent=tangent,
            receiver_world_direction=world, receiver_world_tangent=world_tangent,
            source_material_direction=material, source_theta_derivative=material_tangent,
            source_time_derivative=material_velocity, signed_solid_angle_jacobian=signed)


def enclosed(bounds, value):
    # The Decimal oracle has a much smaller tail than the production budget.
    if bounds.lo == bounds.hi:
        assert abs(bounds.lo-F(value)) <= F(1, 10**540)*max(1, abs(bounds.lo))
    else:
        assert bounds.lo <= F(value) <= bounds.hi


@pytest.mark.parametrize("theta,time", [(F(3, 7), F(1, 13)), (F(19, 35), F(3, 11)),
                                        (F(7, 11), F(5, 7))])
def test_entire_rectangle_encloses_independently_differentiated_full_rotations(theta, time):
    parameters = source()
    measure = certify_native_first_entry_contact_measure(**parameters)
    for name, value in oracle(parameters, theta, time).items():
        actual = getattr(measure, name)
        if isinstance(value, tuple):
            for bound, coordinate in zip(actual, value):
                enclosed(bound, coordinate)
        else:
            enclosed(actual, value)
    assert not measure.first_entry_chronology_certified
    assert not measure.rectangle_is_once_only_archive_area
    assert not measure.source_material_support_certified
    assert not measure.physical_pair_admission_authorized


@pytest.mark.parametrize("time", [F(0), F(1, 10**400), F(1)])
def test_singleton_jacobian_matches_oriented_triple_product_at_exact_times(time):
    parameters = source(native_angle_bounds=I(F(5, 9)), time_fraction_bounds=I(time), precision=256)
    measure = certify_native_first_entry_contact_measure(**parameters)
    independent = oracle(parameters, F(5, 9), time)
    for name in ("signed_solid_angle_jacobian", "source_time_derivative"):
        value, actual = independent[name], getattr(measure, name)
        if isinstance(value, tuple):
            for bound, coordinate in zip(actual, value):
                enclosed(bound, coordinate)
        else:
            enclosed(actual, value)
    assert measure.rectangle_interface_area_m2 == I(F(0))
    assert measure.rectangle_multiplicity_solid_angle_sr == I(F(0))


def test_zero_relative_motion_has_exact_zero_jacobian_and_retains_interface_measure():
    parameters = source(omega_records=((0, ZERO), (1, ZERO)), required_jacobian_sign=0)
    measure = certify_native_first_entry_contact_measure(**parameters)
    assert measure.jacobian_sign == 0
    assert measure.signed_solid_angle_jacobian == I(F(0))
    assert measure.rectangle_multiplicity_solid_angle_sr == I(F(0))
    assert measure.rectangle_interface_area_m2.lo > 0


def test_noncommuting_rotations_are_not_replaced_by_one_relative_exponential():
    parameters = source(native_angle_bounds=I(F(7, 10)), time_fraction_bounds=I(F(2, 5)),
        omega_records=((0, (F(-2, 100), F(5, 100), F(1, 100))),
                       (1, (F(7, 100), F(-3, 100), F(8, 100)))), dt_myr=F(1))
    measure = certify_native_first_entry_contact_measure(**parameters)
    independent = oracle(parameters, F(7, 10), F(2, 5))
    with localcontext() as context:
        context.prec = 600
        axes = dict(parameters["omega_records"])
        relative = tuple(decimal(axes[0][index]-axes[1][index])*decimal(F(2, 5))
            for index in range(3))
        shortcut = rotate(independent["carrier_direction"], relative)
        assert any(not bound.lo <= F(coordinate) <= bound.hi
            for bound, coordinate in zip(measure.source_material_direction, shortcut))
        for bound, coordinate in zip(measure.source_material_direction, independent["source_material_direction"]):
            enclosed(bound, coordinate)


def test_signed_jacobian_reversal_changes_orientation_but_preserves_unsigned_measure():
    parameters = source(omega_records=((0, ZERO), (1, (F(0), F(0), F(1, 50)))))
    positive = certify_native_first_entry_contact_measure(**dict(parameters, required_jacobian_sign=1))
    negative_source = source(omega_records=((0, ZERO), (1, (F(0), F(0), F(-1, 50)))),
        required_jacobian_sign=-1)
    negative = certify_native_first_entry_contact_measure(**negative_source)
    assert positive.jacobian_sign == 1 and negative.jacobian_sign == -1
    assert positive.unsigned_solid_angle_jacobian == negative.unsigned_solid_angle_jacobian
    assert positive.rectangle_multiplicity_solid_angle_sr == negative.rectangle_multiplicity_solid_angle_sr


def test_interface_density_preserves_retained_extent_without_replacing_it_by_native_pi():
    parameters = source()
    measure = certify_native_first_entry_contact_measure(**parameters)
    with localcontext() as context:
        context.prec = 600
        pi = 16*atan_small(Decimal(1)/5)-4*atan_small(Decimal(1)/239)
        scale = decimal(parameters["source_extent_radians"])/(pi/2)
        density = decimal(parameters["source_interface_area_density_m2_per_radian"])*scale
        enclosed(measure.source_angle_scale, scale)
        enclosed(measure.interface_area_density_per_native_radian_m2, density)
        enclosed(measure.rectangle_interface_area_m2,
                 density*decimal(parameters["native_angle_bounds"].width))
    assert measure.source_angle_scale.hi < 1
    assert measure.interface_area_density_per_native_radian_m2.hi < parameters["source_interface_area_density_m2_per_radian"]
    raw = native_first_entry_contact_measure_to_dict(measure)
    assert not raw["interface_measure_is_swept_material_measure"]
    assert not raw["nodal_work_or_energy_density_assigned"]


def test_whole_quarter_measure_contains_original_coefficient_times_retained_extent():
    pi_half = interval_pi(180)/2
    parameters = source(native_angle_bounds=I(F(0), pi_half.hi), precision=160)
    measure = certify_native_first_entry_contact_measure(**parameters)
    total = parameters["source_interface_area_density_m2_per_radian"]*parameters["source_extent_radians"]
    assert measure.rectangle_interface_area_m2.lo <= total <= measure.rectangle_interface_area_m2.hi
    assert measure.rectangle_interface_area_m2 == I(total)


def test_native_endpoint_enclosure_is_intersected_with_analytic_quarter_without_selecting_endpoint():
    parameters = source(native_angle_bounds=I(F(3, 2), F(8, 5)))
    measure = certify_native_first_entry_contact_measure(**parameters)
    assert measure.native_angle_bounds == parameters["native_angle_bounds"]
    assert measure.native_quarter_angle_enclosure.lo == F(3, 2)
    assert measure.native_quarter_angle_enclosure.hi < F(8, 5)
    assert measure.native_quarter_angle_span_rad.lo > 0
    assert measure.computational_angle_bounds.hi <= parameters["source_extent_radians"]


def test_ambiguous_jacobian_sign_is_preserved_and_requested_orientation_refused():
    # The y projection of the quarter tangent is zero at theta=0; this
    # rectangle includes that exact endpoint and therefore has no strict sign.
    parameters = source(omega_records=((0, ZERO), (1, (F(0), F(1, 50), F(0)))),
        native_angle_bounds=I(F(0), F(1, 2)))
    measure = certify_native_first_entry_contact_measure(**parameters)
    assert measure.jacobian_sign is None
    assert measure.signed_solid_angle_jacobian.lo < 0 <= measure.signed_solid_angle_jacobian.hi
    with pytest.raises(ValueError):
        certify_native_first_entry_contact_measure(**dict(parameters, required_jacobian_sign=-1))


def reseal(raw):
    raw["measure_sha256"] = hashlib.sha256(json.dumps({key: value for key, value in raw.items()
        if key != "measure_sha256"}, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    return raw


def test_complete_source_measure_roundtrip():
    parameters = source()
    original = certify_native_first_entry_contact_measure(**parameters)
    raw = native_first_entry_contact_measure_to_dict(original)
    rebuilt = native_first_entry_contact_measure_from_dict(deepcopy(raw), expected_source=parameters)
    assert rebuilt == original
    assert native_first_entry_contact_measure_to_dict(rebuilt) == raw


@pytest.mark.parametrize("mutation", ["jacobian", "derivative", "density", "extent", "source_axis", "archive_flag"])
def test_resealed_dto_cannot_replace_exact_source_or_claim_once_only_archive(mutation):
    parameters = source()
    raw = deepcopy(native_first_entry_contact_measure_to_dict(certify_native_first_entry_contact_measure(**parameters)))
    if mutation == "jacobian":
        raw["signed_solid_angle_jacobian"] = {"lo": [0, 1], "hi": [0, 1]}
    elif mutation == "derivative":
        raw["source_time_derivative"][0] = {"lo": [0, 1], "hi": [0, 1]}
    elif mutation == "density":
        raw["source"]["source_interface_area_density_m2_per_radian"] = [1, 1]
    elif mutation == "extent":
        raw["source"]["source_extent_radians"] = [1, 1]
    elif mutation == "source_axis":
        raw["source"]["omega_records"][1][1][2] = [0, 1]
    else:
        raw["rectangle_is_once_only_archive_area"] = True
    with pytest.raises(ValueError):
        native_first_entry_contact_measure_from_dict(reseal(raw), expected_source=parameters)


@pytest.mark.parametrize("changes", [dict(native_angle_bounds=I(F(-2), F(-1))),
    dict(time_fraction_bounds=I(F(0), F(2))), dict(source_extent_radians=1.5),
    dict(source_interface_area_density_m2_per_radian=F(0)), dict(precision=True),
    dict(start=(F(0), F(3, 5), F(4, 5))), dict(required_jacobian_sign=True)])
def test_invalid_or_aliased_source_is_refused(changes):
    with pytest.raises(ValueError):
        certify_native_first_entry_contact_measure(**source(**changes))


def test_mutated_frozen_derivative_cannot_pass_validation():
    value = certify_native_first_entry_contact_measure(**source())
    changed = replace(value, source_time_derivative=(I(F(0)),)*3)
    with pytest.raises(ValueError):
        validate_native_first_entry_contact_measure(changed)
