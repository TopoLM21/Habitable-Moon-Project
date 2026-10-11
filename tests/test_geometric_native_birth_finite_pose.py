"""Finite SO(3) contact fields checked against independent point motions."""
from fractions import Fraction as F
from dataclasses import replace
import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tectonics.geometric_native_birth_finite_pose import (
    born_finite_pose_angle_bounds, bound_born_finite_pose_fields,
)
from tectonics.geometric_native_pose import FractionInterval as I


def _case(time=I(1), **changes):
    values = dict(start_unit_bounds=(I(0), I(0), I(-1)),
        tangent_unit_bounds=(I(0), I(1), I(0)),
        horizontal_unit_bounds=(I(-1), I(0), I(0)),
        donor_axis=(F(1, 1000), F(-2, 1000), F(13, 1000)),
        receiver_axis=(F(-3, 1000), F(1, 1000), F(7, 1000)), radius_m=F(11),
        donor_radial_before_m=F(2, 10), receiver_radial_before_m=F(-1, 10),
        donor_radial_increment_m=F(1, 10), receiver_radial_increment_m=F(-2, 10),
        sine_dip=F(3, 5), cosine_dip=F(4, 5), birth_gap_free_m=F(-1, 10),
        birth_jump_free_m=F(1, 100), time_fraction_bounds=time, precision=96, order=32)
    values.update(changes)
    return values


def _independent(values, time, angle):
    r = np.asarray((0., math.sin(angle), -math.cos(angle)))
    h = np.asarray((-1., 0., 0.))
    donor = Rotation.from_rotvec(np.asarray(values["donor_axis"], dtype=float)*time).as_matrix()
    receiver = Rotation.from_rotvec(np.asarray(values["receiver_axis"], dtype=float)*time).as_matrix()
    radius = float(values["radius_m"])
    wd = float(values["donor_radial_before_m"]+time*values["donor_radial_increment_m"])
    wr = float(values["receiver_radial_before_m"]+time*values["receiver_radial_increment_m"])
    relative = receiver.T@donor@r
    difference = (radius+wd)*relative-(radius+wr)*r
    initial_difference = float(values["donor_radial_before_m"]-values["receiver_radial_before_m"])*r
    s, c = float(values["sine_dip"]), float(values["cosine_dip"])
    gap = float(values["birth_gap_free_m"])-(s*h+c*r)@(difference-initial_difference)
    jump = float(values["birth_jump_free_m"])-(c*h-s*r)@(difference-initial_difference)
    ordering = -r@difference
    return gap, jump, ordering


@pytest.mark.parametrize("time", [0., .25, 1.])
@pytest.mark.parametrize("angle", [0., .31, .91])
def test_degree_two_fields_match_independent_finite_point_rotations(time, angle):
    values = _case(I(F.from_float(time)))
    fields = bound_born_finite_pose_fields(**values)
    bounds = born_finite_pose_angle_bounds(fields, I(F.from_float(angle)))
    actual = _independent(values, time, angle)
    for bound, value in zip(bounds, actual):
        assert float(bound.midpoint) == pytest.approx(float(value), abs=4e-15)


def test_whole_clock_and_whole_arc_enclose_interior_point_motions():
    values = _case(I(0, 1))
    fields = bound_born_finite_pose_fields(**values)
    bounds = born_finite_pose_angle_bounds(fields, I(0, F(1)))
    for time in (.01, .2, .51, .9):
        for angle in (.03, .17, .43, .92):
            for bound, value in zip(bounds, _independent(values, time, angle)):
                assert float(bound.lo)-4e-15 <= value <= float(bound.hi)+4e-15


def test_sub_ulp_polar_motion_retains_a_strict_normal_sign():
    values = _case(donor_axis=(F(0), F(1, 10**30), F(0)),
        receiver_axis=(F(0),)*3, radius_m=F(10**6), donor_radial_before_m=F(0),
        receiver_radial_before_m=F(0), donor_radial_increment_m=F(0),
        receiver_radial_increment_m=F(0), birth_gap_free_m=F(0), birth_jump_free_m=F(0))
    gap, jump, ordering = born_finite_pose_angle_bounds(bound_born_finite_pose_fields(**values), I(0))
    assert gap.hi < 0
    assert jump.hi < 0
    assert ordering.lo > 0
    assert float(gap.midpoint) == pytest.approx(-6e-25, rel=1e-14)


def test_exact_common_owner_motion_changes_only_independent_radial_banks():
    values = _case(receiver_axis=_case()["donor_axis"])
    gap, jump, ordering = born_finite_pose_angle_bounds(bound_born_finite_pose_fields(**values), I(0, F(1)))
    assert gap.lo <= F(-34, 100) <= gap.hi
    assert jump.lo <= F(19, 100) <= jump.hi
    assert ordering.lo <= F(-6, 10) <= ordering.hi


def test_untyped_axis_cannot_be_mistaken_for_an_exact_used_clock():
    with pytest.raises(ValueError, match="literal-path rational triples"):
        bound_born_finite_pose_fields(**_case(donor_axis=(0., 0., .001)))


def test_sixth_coefficient_cannot_be_retained_but_ignored_by_physics():
    fields = bound_born_finite_pose_fields(**_case())
    with pytest.raises(ValueError, match="exactly five"):
        replace(fields, gap_coefficients=fields.gap_coefficients+(I(123),))


def test_bounds_refuse_angles_outside_the_proved_minor_arc_budget():
    fields = bound_born_finite_pose_fields(**_case())
    with pytest.raises(ValueError, match="minor arc"):
        born_finite_pose_angle_bounds(fields, I(0, 7))
