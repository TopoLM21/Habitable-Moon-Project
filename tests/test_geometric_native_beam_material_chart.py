"""Independent material-chart and restricted-cubic conservation controls."""
from copy import deepcopy
from dataclasses import replace
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import json

import pytest

from tectonics.geometric_native_pose import FractionInterval as I
from tectonics.geometric_native_beam_material_chart import (
    certify_native_beam_cut_history,
    certify_native_beam_material_chart,
    native_beam_cut_history_from_dict,
    native_beam_cut_history_to_dict,
    native_beam_material_chart_from_dict,
    native_beam_material_chart_to_dict,
    validate_native_beam_cut_history,
    validate_native_beam_material_chart,
)


F = Fraction
START = (F(0), F(1), F(0))
END = (F(0), F(0), F(1))


def chart_source(**changes):
    source = dict(trace_id="donor-own-source-beam", bank="sub", start=START,
        end=END, reference_coordinates=(F(-1, 10), F(3, 5), F(4, 5)),
        radius_m=F(100), width_m=F(20), tip_displacement_m=I(F(7)),
        precision=160)
    source.update(changes)
    return source


def cut_source(**changes):
    source = dict(width_m=F(11), cut_distance_m=F(3),
        tip_displacement_m=I(F(7, 3)), rigidity_nm=F(17), precision=160)
    source.update(changes)
    return source


def decimal(value):
    return Decimal(value.numerator)/Decimal(value.denominator)


def atan_small(value):
    result = term = value
    for index in range(1, 2000):
        term *= -value*value
        addition = term/Decimal(2*index+1)
        result += addition
        if abs(addition) < Decimal("1e-480"):
            return result
    raise AssertionError("Independent Decimal atan did not converge")


def enclosed(bounds, value):
    if bounds.lo == bounds.hi:
        assert abs(bounds.lo-F(value)) < F(1, 10**460)*max(1, abs(bounds.lo))
    else:
        assert bounds.lo <= F(value) <= bounds.hi


def independent_shape(source, point):
    with localcontext() as context:
        context.prec = 500
        x, y, z = map(decimal, point)
        transverse = -x if source["bank"] == "sub" else x
        beta = atan_small(transverse/(y*y+z*z).sqrt())
        pi = 16*atan_small(Decimal(1)/5)-4*atan_small(Decimal(1)/239)
        # The controls use the exact 3:4 in-plane ray.
        theta = pi/4+atan_small(Decimal(1)/7)
        radius, width = (decimal(source[name]) for name in ("radius_m", "width_m"))
        tip = decimal(source["tip_displacement_m"].lo)
        distance = radius*beta
        coordinate = distance/width
        phi = (1-coordinate)**2*(coordinate+2)/2
        slope = Decimal(3)*(coordinate*coordinate-1)/2
        return dict(inward_angle_rad=beta, along_strike_angle_rad=theta,
            inward_distance_m=distance, x_over_width=coordinate, shape=phi,
            shape_slope=slope, shape_curvature=3*coordinate,
            shape_third_derivative=Decimal(3), displacement_m=tip*phi,
            slope=tip*slope/width, curvature_per_m=tip*3*coordinate/(width*width),
            third_derivative_per_m2=tip*3/(width**3))


@pytest.mark.parametrize("bank,sign", [("sub", -1), ("over", 1)])
def test_material_chart_matches_independent_angles_and_original_cubic(bank, sign):
    source = chart_source(bank=bank,
        reference_coordinates=(F(sign, 10), F(3, 5), F(4, 5)))
    chart = certify_native_beam_material_chart(**source)
    for name, expected in independent_shape(source, source["reference_coordinates"]).items():
        enclosed(getattr(chart, name), expected)
    assert chart.reference_coordinates == source["reference_coordinates"]
    assert not chart.material_labels_normalized
    assert not chart.source_shape_is_continuum_interior_contact_solution
    assert not chart.native_bank_support_certified
    assert not chart.physical_pair_admission_authorized
    assert not chart.candidate_mechanics_adopted


@pytest.mark.parametrize("scale", [F(1, 10**400), F(1), F(10**400)])
def test_homogeneous_scale_preserves_material_angles_but_not_literal_label(scale):
    ordinary = chart_source()
    source = chart_source(reference_coordinates=tuple(
        scale*value for value in ordinary["reference_coordinates"]))
    chart = certify_native_beam_material_chart(**source)
    oracle = independent_shape(ordinary, ordinary["reference_coordinates"])
    for name in ("inward_angle_rad", "along_strike_angle_rad", "x_over_width", "shape"):
        enclosed(getattr(chart, name), oracle[name])
    assert chart.reference_coordinates == source["reference_coordinates"]
    if scale != 1:
        assert chart.source_sha256 != certify_native_beam_material_chart(**ordinary).source_sha256


def test_interior_material_uses_own_bank_amplitude_and_not_the_source_tip():
    donor = certify_native_beam_material_chart(**chart_source())
    other = certify_native_beam_material_chart(**chart_source(
        trace_id="different-receiver-beam", tip_displacement_m=I(F(19))))
    assert 0 < donor.shape.lo <= donor.shape.hi < 1
    assert donor.displacement_m.hi < 7
    assert donor.displacement_m.hi < other.displacement_m.lo
    assert donor.shape == other.shape
    assert donor.source_sha256 != other.source_sha256


def test_sub_float_material_offset_retains_nontrivial_source_shape():
    chart = certify_native_beam_material_chart(**chart_source(
        reference_coordinates=(F(-1, 10**800), F(3, 5), F(4, 5))))
    assert 0 < chart.x_over_width.lo < chart.x_over_width.hi
    assert chart.shape.hi < 1
    assert chart.shape_slope.lo > -F(3, 2)
    assert float(chart.inward_distance_m.lo) == 0


def test_interval_label_encloses_every_interior_sample_without_midpoint_assignment():
    source = chart_source(reference_coordinates=(I(F(-1, 9), F(-1, 10)),
        I(F(3, 5)), I(F(4, 5))))
    chart = certify_native_beam_material_chart(**source)
    for transverse in (F(1, 10), F(19, 180), F(1, 9)):
        point = (-transverse, F(3, 5), F(4, 5))
        for name, expected in independent_shape(source, point).items():
            enclosed(getattr(chart, name), expected)
    assert chart.reference_coordinates == source["reference_coordinates"]


@pytest.mark.parametrize("point", [
    (F(1, 10), F(3, 5), F(4, 5)),
    (F(-1, 10), F(-3, 5), F(4, 5)),
    (F(-1, 10), F(3, 5), F(-4, 5)),
    (F(-1), F(0), F(0)),
    (F(-1), F(3, 5), F(4, 5)),
    (I(F(-1), F(-1, 10)), I(F(3, 5)), I(F(4, 5))),
    (I(F(-1, 10), F(1, 10)), I(F(3, 5)), I(F(4, 5))),
])
def test_chart_requires_whole_quadrant_and_width_membership(point):
    with pytest.raises(ValueError):
        certify_native_beam_material_chart(**chart_source(reference_coordinates=point))


@pytest.mark.parametrize("changes", [
    dict(start=(F(0), F(3, 5), F(4, 5))),
    dict(end=START), dict(bank="incoming"), dict(radius_m=100),
    dict(width_m=F(-20)), dict(precision=True),
    dict(reference_coordinates=[F(-1, 10), F(3, 5), F(4, 5)]),
    dict(reference_coordinates=(-.1, .6, .8)),
])
def test_chart_rejects_untyped_or_incompatible_source(changes):
    with pytest.raises(ValueError):
        certify_native_beam_material_chart(**chart_source(**changes))


@pytest.mark.parametrize("distance", [F(0), F(3), F(11)])
def test_restricted_cubic_matches_independent_displacement_moment_and_energy(distance):
    source = cut_source(cut_distance_m=distance)
    cut = certify_native_beam_cut_history(**source)
    width, tip, rigidity = source["width_m"], source["tip_displacement_m"].lo, source["rigidity_nm"]
    z = distance/width
    phi = 1-F(3, 2)*z+z**3/2
    slope = tip*(-F(3, 2)+F(3, 2)*z*z)/width
    curvature = 3*tip*distance/width**3
    whole = F(3, 2)*rigidity*tip**2/width**3
    assert cut.displacement_at_cut_m == I(tip*phi)
    assert cut.slope_at_cut == I(slope)
    assert cut.curvature_at_cut_per_m == I(curvature)
    assert cut.moment_per_unit_strike_n == I(-rigidity*curvature)
    assert cut.shear_per_unit_strike_n_per_m == I(3*rigidity*tip/width**3)
    assert cut.whole_energy_per_unit_strike_j_per_m == I(whole)
    assert cut.toe_energy_per_unit_strike_j_per_m == I(whole*z**3)
    assert cut.retained_energy_per_unit_strike_j_per_m == I(whole*(1-z**3))
    assert cut.toe_energy_per_unit_strike_j_per_m+cut.retained_energy_per_unit_strike_j_per_m == I(whole)
    assert not cut.new_free_tip_boundary_condition_imposed
    assert not cut.source_preload_reset
    assert not cut.moving_cut_work_authorized


def test_cut_boundary_virtual_power_includes_couple_and_cancels_between_pieces():
    source = cut_source()
    cut = certify_native_beam_cut_history(**source)
    width, tip, rigidity = source["width_m"], source["tip_displacement_m"].lo, source["rigidity_nm"]
    z = source["cut_distance_m"]/width
    displacement_rate = 1-F(3, 2)*z+z**3/2
    slope_rate = (-F(3, 2)+F(3, 2)*z*z)/width
    toe_boundary = (cut.toe_cut_force_per_unit_strike_n_per_m*displacement_rate
        + cut.toe_cut_couple_per_unit_strike_n*slope_rate)
    retained_boundary = (cut.retained_cut_force_per_unit_strike_n_per_m*displacement_rate
        + cut.retained_cut_couple_per_unit_strike_n*slope_rate)
    assert toe_boundary+retained_boundary == I(0)
    tip_power = I(3*rigidity*tip/width**3)
    assert toe_boundary+tip_power == cut.toe_boundary_power_per_tip_rate_n_per_m
    assert retained_boundary == cut.retained_boundary_power_per_tip_rate_n_per_m
    assert toe_boundary+tip_power+retained_boundary == tip_power
    # Resetting an interior cut to a free tip would remove this nonzero couple.
    assert cut.retained_cut_couple_per_unit_strike_n.lo != 0
    assert retained_boundary != cut.retained_cut_force_per_unit_strike_n_per_m*displacement_rate


def test_moving_cut_energy_flux_is_distinct_from_fixed_cut_tip_power():
    source = cut_source()
    center, halfwidth = source["cut_distance_m"], F(1, 5)
    left = certify_native_beam_cut_history(**cut_source(cut_distance_m=center-halfwidth))
    right = certify_native_beam_cut_history(**cut_source(cut_distance_m=center+halfwidth))
    cut = certify_native_beam_cut_history(**source)
    width, tip, rigidity = source["width_m"], source["tip_displacement_m"].lo, source["rigidity_nm"]
    flux = rigidity*cut.curvature_at_cut_per_m.lo**2/2
    secant = (right.toe_energy_per_unit_strike_j_per_m.lo-left.toe_energy_per_unit_strike_j_per_m.lo)/(2*halfwidth)
    cubic_correction = F(3, 2)*rigidity*tip**2*halfwidth**2/width**6
    assert secant-cubic_correction == flux > 0
    shear = 3*rigidity*tip/width**3
    slope = tip*(-F(3, 2)+F(3, 2)*(center/width)**2)/width
    configurational_coefficient = shear*slope-flux
    assert cut.moving_cut_configurational_power_per_cut_rate_n_per_m == I(configurational_coefficient)
    assert configurational_coefficient == -F(9, 2)*rigidity*tip**2/width**4
    assert (right.toe_energy_per_unit_strike_j_per_m-left.toe_energy_per_unit_strike_j_per_m
        + right.retained_energy_per_unit_strike_j_per_m-left.retained_energy_per_unit_strike_j_per_m) == I(0)
    assert not cut.moving_cut_work_authorized


def test_interval_cut_encloses_cubic_samples_and_zero_energy_keeps_partition():
    cut = certify_native_beam_cut_history(**cut_source(cut_distance_m=I(F(2), F(4))))
    for distance in (F(2), F(3), F(4)):
        sample = certify_native_beam_cut_history(**cut_source(cut_distance_m=distance))
        for name in ("displacement_at_cut_m", "slope_at_cut", "curvature_at_cut_per_m",
                     "moment_per_unit_strike_n", "toe_energy_per_unit_strike_j_per_m",
                     "retained_energy_per_unit_strike_j_per_m"):
            outer, inner = getattr(cut, name), getattr(sample, name)
            assert outer.lo <= inner.lo <= inner.hi <= outer.hi
    zero = certify_native_beam_cut_history(**cut_source(tip_displacement_m=I(F(0))))
    assert zero.whole_energy_per_unit_strike_j_per_m == I(0)
    assert zero.toe_energy_fraction == I(F(3, 11)**3)
    assert zero.retained_energy_fraction == I(1-F(3, 11)**3)


def reseal(raw, seal):
    body = {name: value for name, value in raw.items() if name != seal}
    raw[seal] = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


@pytest.mark.parametrize("kind", ["chart", "cut"])
def test_strict_roundtrip_requires_complete_authoritative_source(kind):
    certify, dump, load, source = ((certify_native_beam_material_chart,
        native_beam_material_chart_to_dict, native_beam_material_chart_from_dict, chart_source())
        if kind == "chart" else (certify_native_beam_cut_history,
        native_beam_cut_history_to_dict, native_beam_cut_history_from_dict, cut_source()))
    value = certify(**source)
    raw = dump(value)
    assert load(deepcopy(raw), expected_source=source) == value
    with pytest.raises(ValueError):
        load(raw, expected_source={"source_sha256": value.source_sha256})
    altered_source = dict(source, tip_displacement_m=I(F(19)))
    with pytest.raises(ValueError):
        load(raw, expected_source=altered_source)


@pytest.mark.parametrize("field", ["shape", "displacement_m", "native_bank_support_certified",
    "physical_pair_admission_authorized", "material_labels_normalized", "source_sha256"])
def test_resealed_chart_cannot_forge_geometry_or_physical_authority(field):
    source = chart_source()
    raw = native_beam_material_chart_to_dict(certify_native_beam_material_chart(**source))
    raw[field] = ({"lo": [0, 1], "hi": [0, 1]} if type(raw[field]) is dict
        else True if type(raw[field]) is bool else "0"*64)
    reseal(raw, "chart_sha256")
    with pytest.raises(ValueError):
        native_beam_material_chart_from_dict(raw, expected_source=source)


@pytest.mark.parametrize("field", ["moment_per_unit_strike_n", "toe_energy_per_unit_strike_j_per_m",
    "moving_cut_work_authorized", "new_free_tip_boundary_condition_imposed", "source_sha256"])
def test_resealed_cut_cannot_reset_moment_energy_or_claim_moving_cut(field):
    source = cut_source()
    raw = native_beam_cut_history_to_dict(certify_native_beam_cut_history(**source))
    raw[field] = ({"lo": [0, 1], "hi": [0, 1]} if type(raw[field]) is dict
        else True if type(raw[field]) is bool else "0"*64)
    reseal(raw, "cut_sha256")
    with pytest.raises(ValueError):
        native_beam_cut_history_from_dict(raw, expected_source=source)


@pytest.mark.parametrize("kind", ["chart", "cut"])
def test_in_memory_numeric_alias_and_dto_integer_alias_are_rejected(kind):
    certify, validate, dump, load, source = ((certify_native_beam_material_chart,
        validate_native_beam_material_chart, native_beam_material_chart_to_dict,
        native_beam_material_chart_from_dict, chart_source()) if kind == "chart" else (
        certify_native_beam_cut_history, validate_native_beam_cut_history,
        native_beam_cut_history_to_dict, native_beam_cut_history_from_dict, cut_source()))
    value = certify(**source)
    with pytest.raises(ValueError):
        validate(replace(value, precision=160.0))
    raw = dump(value)
    raw["accepted_native_force_intervals"] = False
    with pytest.raises(ValueError):
        load(raw, expected_source=source)
    raw = dump(value)
    raw["source"]["radius_m" if kind == "chart" else "width_m"][0] = 100.0
    with pytest.raises(ValueError):
        load(raw, expected_source=source)
