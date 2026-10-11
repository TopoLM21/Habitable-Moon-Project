"""Independent finite-pose controls for the unchanged original spring tips.

The Decimal oracle uses Rodrigues' formula, rather than the production
Taylor matrix bounds.  Literal tip labels and full transverse rotation are
retained; these geometry diagnostics cannot create a new material spring.
"""
from copy import deepcopy
from dataclasses import replace
from decimal import Decimal, getcontext, localcontext
from fractions import Fraction
import hashlib
import json

import pytest

from tectonics.geometric_native_pose import FractionInterval as I
from tectonics.geometric_native_physical_pair_time import (
    certify_native_original_pair_tip_maps,
    certify_native_quarter_carrier_coordinate,
    certify_native_relative_entry_coordinates,
    native_direction_is_positive_scale,
    native_original_pair_tip_maps_from_dict,
    native_original_pair_tip_maps_to_dict,
    validate_native_original_pair_tip_maps,
)


F = Fraction
ZERO = (F(0),) * 3


def source(**changes):
    values = dict(trace_id="literal-original-pair", reference_coordinates=(
        F(0), F(3, 5) + F(1, 2**52), F(4, 5)),
        original_normal=(F(1), F(0), F(0)),
        original_down_direction=(F(0), F(4, 5), F(-3, 5)),
        sub_owner=1, over_owner=0,
        omega_records=((0, (F(1, 17), F(-2, 19), F(3, 23))),
                       (1, (F(-4, 29), F(5, 31), F(-6, 37))),
                       (9, (F(1, 43), F(1, 47), F(1, 53)))),
        dt_myr=F(1, 40), radius_m=F(100_000),
        radial_before=(F(1, 7), F(-2, 11)),
        radial_candidate=(F(-3, 13), F(4, 17)),
        time_fraction_bounds=I(F(2, 7)), precision=160, order=32)
    values.update(changes)
    return values


def decimal(value):
    return Decimal(value.numerator) / Decimal(value.denominator)


def cross(left, right):
    return (left[1]*right[2]-left[2]*right[1],
            left[2]*right[0]-left[0]*right[2],
            left[0]*right[1]-left[1]*right[0])


def dot(left, right):
    return sum((a*b for a, b in zip(left, right)), Decimal(0))


def rotate(vector, axis):
    """Decimal Rodrigues with stable sin(theta)/theta and (1-cos)/theta²."""
    square = dot(axis, axis)
    if not square:
        return vector
    sinc, cosm, ts, tc = Decimal(1), Decimal("0.5"), Decimal(1), Decimal("0.5")
    tolerance = Decimal(10) ** (-getcontext().prec + 20)
    for k in range(1, 2000):
        ts *= -square / Decimal((2*k)*(2*k+1))
        tc *= -square / Decimal((2*k+1)*(2*k+2))
        sinc += ts
        cosm += tc
        if max(abs(ts), abs(tc)) < tolerance:
            break
    else:
        raise AssertionError("Independent Rodrigues series did not converge")
    first = cross(axis, vector)
    second = cross(axis, first)
    return tuple(p+sinc*a+cosm*b for p, a, b in zip(vector, first, second))


def oracle(parameters, time):
    with localcontext() as context:
        context.prec = 1100
        fraction = decimal(time)
        point = tuple(map(decimal, parameters["reference_coordinates"]))
        axes = {owner: tuple(decimal(x)*decimal(parameters["dt_myr"])*fraction
                             for x in vector)
                for owner, vector in parameters["omega_records"]}
        sub = rotate(point, axes[parameters["sub_owner"]])
        over = rotate(point, axes[parameters["over_owner"]])
        normal = tuple(map(decimal, parameters["original_normal"]))
        down = tuple(map(decimal, parameters["original_down_direction"]))
        moved_normal = rotate(normal, axes[parameters["over_owner"]])
        moved_down = rotate(down, axes[parameters["over_owner"]])
        radial = tuple(decimal(old) + fraction*decimal(new-old)
            for old, new in zip(parameters["radial_before"], parameters["radial_candidate"]))
        radius = decimal(parameters["radius_m"])
        sub_position = tuple((radius+radial[0])*p for p in sub)
        over_position = tuple((radius+radial[1])*p for p in over)
        separation = tuple(b-a for a, b in zip(sub_position, over_position))
        gap, jump = dot(separation, moved_normal), dot(separation, moved_down)
        reference = decimal(parameters["radial_before"][1]-parameters["radial_before"][0])
        result = dict(sub_direction=sub, over_direction=over,
            sub_position_m=sub_position, over_position_m=over_position,
            normal=moved_normal, down_direction=moved_down,
            radial_at_time=radial, separation_m=separation,
            gap_m=gap, jump_m=jump,
            exact_gap_change_m=gap-reference*dot(point, normal),
            exact_jump_change_m=jump-reference*dot(point, down))
        relative = rotate(sub, tuple(-item for item in axes[parameters["over_owner"]]))
        return result, relative


def enclosed(bounds, value):
    # The Decimal oracle exceeds the requested enclosure precision by hundreds of digits.
    if bounds.lo == bounds.hi:
        assert abs(bounds.lo-F(value)) <= F(1, 10**1000)*max(1, abs(bounds.lo))
        return
    assert bounds.lo <= F(value) <= bounds.hi


@pytest.mark.parametrize("time", [F(0), F(2, 7), F(1), F(1, 10**400)])
def test_both_original_tips_frames_and_separations_contain_independent_rodrigues(time):
    parameters = source(time_fraction_bounds=I(time))
    result = certify_native_original_pair_tip_maps(**parameters)
    independent, _ = oracle(parameters, time)
    for name, expected in independent.items():
        actual = getattr(result, name)
        if type(expected) is tuple:
            for bounds, value in zip(actual, expected):
                enclosed(bounds, value)
        else:
            enclosed(actual, expected)
    assert result.reference_coordinates == parameters["reference_coordinates"]
    assert result.omega_records == parameters["omega_records"]


def test_whole_window_encloses_both_tips_at_independent_interior_times():
    parameters = source(time_fraction_bounds=I(F(1, 5), F(4, 5)))
    result = certify_native_original_pair_tip_maps(**parameters)
    for time in (F(1, 5), F(5, 13), F(7, 11), F(4, 5)):
        independent, _ = oracle(parameters, time)
        for name in ("sub_position_m", "over_position_m", "normal", "down_direction"):
            for bounds, value in zip(getattr(result, name), independent[name]):
                enclosed(bounds, value)
        enclosed(result.exact_gap_change_m, independent["exact_gap_change_m"])
        enclosed(result.exact_jump_change_m, independent["exact_jump_change_m"])


def test_two_exact_tiny_clock_factors_retain_a_nonzero_physical_change():
    parameters = source(dt_myr=F(1, 10**400),
        time_fraction_bounds=I(F(1, 10**400)), radial_candidate=source()["radial_before"])
    result = certify_native_original_pair_tip_maps(**parameters)
    independent, _ = oracle(parameters, parameters["time_fraction_bounds"].lo)
    enclosed(result.exact_gap_change_m, independent["exact_gap_change_m"])
    assert result.exact_gap_change_m.lo * result.exact_gap_change_m.hi > 0
    assert result.dt_myr * result.time_fraction_bounds.lo == F(1, 10**800)
    assert float(result.dt_myr * result.time_fraction_bounds.lo) == 0.


def test_common_rigid_owner_rotation_keeps_exact_reference_separation_projection():
    omega = (F(2, 13), F(-3, 17), F(5, 19))
    parameters = source(omega_records=((0, omega), (1, omega)),
        radial_candidate=source()["radial_before"])
    result = certify_native_original_pair_tip_maps(**parameters)
    assert result.exact_gap_change_m == I(0)
    assert result.exact_jump_change_m == I(0)
    assert result.gap_m == I(result.reference_gap_m)
    assert result.jump_m == I(result.reference_jump_m)


def test_literal_saved_tip_label_is_never_normalized_to_a_unit_ray():
    point = (F(0), F(3), F(4))
    parameters = source(reference_coordinates=point, omega_records=((0, ZERO), (1, ZERO)),
        radial_before=(F(0), F(0)), radial_candidate=(F(0), F(0)))
    result = certify_native_original_pair_tip_maps(**parameters)
    assert result.sub_direction == result.over_direction == tuple(I(x) for x in point)
    assert result.sub_position_m == tuple(I(parameters["radius_m"]*x) for x in point)
    assert result.reference_coordinates == point


def test_relative_entry_uses_inverse_receiving_rotation_and_all_components():
    parameters = source()
    _, expected = oracle(parameters, parameters["time_fraction_bounds"].lo)
    result = certify_native_relative_entry_coordinates(parameters["reference_coordinates"],
        parameters["sub_owner"], parameters["over_owner"], parameters["omega_records"],
        parameters["dt_myr"], parameters["time_fraction_bounds"], precision=160)
    for bounds, value in zip(result, expected):
        enclosed(bounds, value)
    assert result[0].lo*result[0].hi > 0


def test_relative_entry_normalization_is_explicit_and_never_a_tip_label_change():
    point = (F(0), F(3), F(4))
    arguments = (point, 1, 0, ((0, ZERO), (1, ZERO)), F(1), I(F(1)))
    assert certify_native_relative_entry_coordinates(*arguments) == tuple(I(x) for x in point)
    normalized = certify_native_relative_entry_coordinates(*arguments, normalize_direction=True)
    for bounds, expected in zip(normalized, (F(0), F(3, 5), F(4, 5))):
        assert bounds.lo <= expected <= bounds.hi


@pytest.mark.parametrize("scale", [F(1, 10**400), F(1), F(10**400)])
def test_quarter_carrier_angle_is_independent_of_positive_homogeneous_scale(scale):
    coordinate = certify_native_quarter_carrier_coordinate(
        (F(0), F(1), F(0)), (F(0), F(0), F(1)),
        (I(0), I(3*scale), I(4*scale)), precision=160)
    with localcontext() as context:
        context.prec = 300
        # atan(4/3) = pi/4 + atan(1/7), with a rapidly convergent Decimal series.
        def atan(value):
            total = term = value
            for k in range(1, 500):
                term *= -value*value
                addition = term/Decimal(2*k+1)
                total += addition
                if abs(addition) < Decimal("1e-285"):
                    return total
            raise AssertionError("Independent atan series failed")
        pi = 16*atan(Decimal(1)/5)-4*atan(Decimal(1)/239)
        enclosed(coordinate, pi/4+atan(Decimal(1)/7))


def test_quarter_carrier_retains_endpoint_singletons_and_rejects_unknown_quadrant():
    axes = ((F(0), F(1), F(0)), (F(0), F(0), F(1)))
    assert certify_native_quarter_carrier_coordinate(*axes, (I(0), I(1), I(0))) == I(0)
    last = certify_native_quarter_carrier_coordinate(*axes, (I(0), I(0), I(1)))
    assert F(3, 2) < last.lo < last.hi < F(8, 5)
    for uncertain in ((I(0), I(-1, 1), I(1)), (I(0), I(0, 1), I(0, 1)),
                      (I(0), I(1), I(-1, 0))):
        with pytest.raises(ValueError):
            certify_native_quarter_carrier_coordinate(*axes, uncertain)


def test_homogeneous_identity_does_not_authenticate_a_different_material_label():
    first, second = (F(0), F(3), F(4)), (F(0), F(6), F(8))
    assert native_direction_is_positive_scale(first, second)
    assert first != second
    assert not native_direction_is_positive_scale(first, tuple(-x for x in second))
    assert not native_direction_is_positive_scale(first, (F(1, 10**400), F(3), F(4)))


def test_geometry_replay_retains_all_source_records_and_explicit_nonadoption():
    parameters = source()
    result = certify_native_original_pair_tip_maps(**parameters)
    raw = native_original_pair_tip_maps_to_dict(result)
    rebuilt = native_original_pair_tip_maps_from_dict(deepcopy(raw), expected_source=parameters)
    assert rebuilt == result
    assert not raw["physical_pair_admission_authorized"]
    assert not raw["candidate_mechanics_adopted"]
    assert not raw["finite_pose_change_replaces_free_anchors"]
    assert not raw["material_handover_defined"]
    assert raw["accepted_work_increment_j"] == [0, 1]
    assert raw["accepted_native_force_intervals"] == 0


@pytest.mark.parametrize("field", ["maps_sha256", "source_sha256", "gap_m",
    "physical_pair_admission_authorized", "candidate_mechanics_adopted",
    "finite_pose_change_replaces_free_anchors", "material_handover_defined",
    "accepted_native_force_intervals", "accepted_work_increment_j"])
def test_resealed_derived_data_cannot_replace_complete_source_replay(field):
    parameters = source()
    raw = native_original_pair_tip_maps_to_dict(certify_native_original_pair_tip_maps(**parameters))
    altered = deepcopy(raw)
    altered[field] = ({"lo": [0, 1], "hi": [0, 1]} if field == "gap_m" else
                      [1, 1] if field == "accepted_work_increment_j" else
                      "f"*64 if field.endswith("sha256") else 1)
    body = {key: value for key, value in altered.items() if key != "maps_sha256"}
    if field != "maps_sha256":
        altered["maps_sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True,
            separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    with pytest.raises(ValueError):
        native_original_pair_tip_maps_from_dict(altered, expected_source=parameters)


@pytest.mark.parametrize("changes", [dict(dt_myr=F(1, 41)), dict(radius_m=F(100_001)),
    dict(sub_owner=9), dict(over_owner=9), dict(radial_before=(F(0), F(0))),
    dict(reference_coordinates=(F(0), F(3, 5), F(4, 5))),
    dict(omega_records=((0, ZERO), (1, ZERO))), dict(time_fraction_bounds=I(F(3, 7)))])
def test_valid_self_consistent_alternate_source_is_rejected_by_external_identity(changes):
    authoritative = source()
    raw = native_original_pair_tip_maps_to_dict(certify_native_original_pair_tip_maps(**source(**changes)))
    with pytest.raises(ValueError):
        native_original_pair_tip_maps_from_dict(raw, expected_source=authoritative)


@pytest.mark.parametrize("changes", [dict(sub_owner=True), dict(dt_myr=1.),
    dict(reference_coordinates=[F(0), F(3), F(4)]),
    dict(radial_before=(0, F(0))), dict(time_fraction_bounds=I(F(-1, 10), F(1, 2))),
    dict(omega_records=((1, ZERO), (0, ZERO))),
    dict(omega_records=((0, ZERO), (1, ZERO), (1, ZERO))),
    dict(precision=True), dict(order=7), dict(radius_m=F(0))])
def test_ambiguous_source_types_and_invalid_clocks_are_refused(changes):
    with pytest.raises(ValueError):
        certify_native_original_pair_tip_maps(**source(**changes))


def test_mutable_or_nonliteral_derived_fields_cannot_bypass_typed_validation():
    result = certify_native_original_pair_tip_maps(**source())
    with pytest.raises(ValueError):
        validate_native_original_pair_tip_maps(replace(result, sub_direction=list(result.sub_direction)))
    with pytest.raises(ValueError):
        validate_native_original_pair_tip_maps(replace(result, gap_m=0))
