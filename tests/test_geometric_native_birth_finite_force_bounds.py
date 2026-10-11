"""Finite unilateral forces checked by independent matrix derivatives and quadrature."""
from dataclasses import replace
from decimal import Decimal, localcontext
from fractions import Fraction as F
import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.linalg import expm, expm_frechet

from tests.test_geometric_native_birth_contact_network import _case
from tectonics.geometric_native_birth_contact_continuum import (
    BornContinuumAttachment, BornContinuumDeclaration, _advance_field,
    activate_born_continuum_contact,
)
from tectonics.geometric_native_birth_finite_pose import BornFinitePoseFields, bound_born_finite_pose_fields
from tectonics.geometric_native_birth_finite_force_bounds import (
    _affine_reference_bounds, bound_born_finite_pose_force_error, bound_born_finite_pose_jacobian,
    certify_born_finite_pose_force_budget,
)
from tectonics.geometric_native_pose import FractionInterval as I


def _decimal(value):
    with localcontext() as context:
        context.prec = 160
        return str(Decimal(value.numerator)/Decimal(value.denominator))


def _sources(*, axis=True, end=F(1), affine_gap=None, density=F(1), time=I(1), **changes):
    source = _case(gap=-.1)["born_attachments"][0]
    radius, before, s, c = F(11), F(1, 5), F(3, 5), F(4, 5)
    rows = [[F(0)]*8 for _ in range(3)]
    rows[0][6], rows[0][7] = -radius*c, radius*c
    rows[1][1], rows[1][4] = -(radius+before)*s, (radius+before)*s
    rows[2][2], rows[2][5] = -(radius+before)*s, (radius+before)*s
    declaration = BornContinuumDeclaration(source.parameters, source.reference, "0", _decimal(end),
        _decimal(density), tuple(tuple(_decimal(value) for value in row) for row in rows),
        (("0",)*8,)*3, "Actual two angular banks and two radial ports")
    previous = activate_born_continuum_contact(declaration, decimal_precision=160)
    attachment = BornContinuumAttachment(declaration, previous)
    values = dict(start_unit_bounds=(I(0), I(0), I(-1)), tangent_unit_bounds=(I(0), I(1), I(0)),
        horizontal_unit_bounds=(I(-1), I(0), I(0)),
        donor_axis=(F(1, 1000), F(-2, 1000), F(13, 1000)) if axis else (F(0),)*3,
        receiver_axis=(F(-3, 1000), F(1, 1000), F(7, 1000)) if axis else (F(0),)*3,
        radius_m=radius, donor_radial_before_m=before, receiver_radial_before_m=F(-1, 10),
        donor_radial_increment_m=F(1, 10) if axis else F(0),
        receiver_radial_increment_m=F(-2, 10) if axis else F(0),
        sine_dip=s, cosine_dip=c, birth_gap_free_m=F.from_float(-.1), birth_jump_free_m=F(0),
        time_fraction_bounds=time, precision=96, order=32)
    values.update(changes)
    if affine_gap is None:
        angular = values["donor_axis"]+values["receiver_axis"]
        increment = angular+(values["donor_radial_increment_m"]/radius, values["receiver_radial_increment_m"]/radius)
        affine_gap = tuple(sum((coefficient*value for coefficient, value in zip(row, increment)), F(0))
            + (values["birth_gap_free_m"] if index == 0 else F(0)) for index, row in enumerate(rows))
    with localcontext() as context:
        context.prec = 160
        step = _advance_field(previous, tuple(Decimal(_decimal(value)) for value in affine_gap),
            (Decimal(0),)*3, 1., 160)
    return declaration, attachment, step, values


def _jacobian(declaration, values, angle=I(0, F(1))):
    return bound_born_finite_pose_jacobian(declaration,
        **{name: values[name] for name in ("start_unit_bounds", "tangent_unit_bounds", "horizontal_unit_bounds",
            "donor_axis", "receiver_axis", "radius_m", "donor_radial_before_m", "donor_radial_increment_m",
            "sine_dip", "cosine_dip", "time_fraction_bounds", "precision", "order")},
        donor_angular_indices=(0, 1, 2), receiver_angular_indices=(3, 4, 5),
        donor_radial_index=6, receiver_radial_index=7, angle_bounds=angle)


def _skew(value):
    x, y, z = value
    return np.asarray(((0., -z, y), (z, 0., -x), (-y, x, 0.)))


def _independent_point(values, angle, time=1.):
    r = np.asarray((0., math.sin(angle), -math.cos(angle)))
    h = np.asarray((-1., 0., 0.))
    donor_axis = np.asarray(values["donor_axis"], dtype=float)*time
    receiver_axis = np.asarray(values["receiver_axis"], dtype=float)*time
    donor, receiver = expm(_skew(donor_axis)), expm(_skew(receiver_axis))
    s, c, radius = map(float, (values["sine_dip"], values["cosine_dip"], values["radius_m"]))
    normal = s*h+c*r
    receiving_normal = receiver@normal
    lever = radius+float(values["donor_radial_before_m"]+F.from_float(time)*values["donor_radial_increment_m"])
    receiver_radius = radius+float(values["receiver_radial_before_m"]+F.from_float(time)*values["receiver_radial_increment_m"])
    difference = lever*(receiver.T@donor@r)-receiver_radius*r
    initial = float(values["donor_radial_before_m"]-values["receiver_radial_before_m"])*r
    gap = float(values["birth_gap_free_m"])-normal@(difference-initial)
    gradients = []
    for index in range(3):
        unit = np.zeros(3)
        unit[index] = 1.
        derivative = expm_frechet(_skew(donor_axis), _skew(unit), compute_expm=False)
        gradients.append(-lever*receiving_normal@(derivative@r))
    for index in range(3):
        unit = np.zeros(3)
        unit[index] = 1.
        derivative = expm_frechet(_skew(receiver_axis), _skew(unit), compute_expm=False)
        gradients.append(-lever*(derivative@normal)@(donor@r))
    gradients += [-radius*receiving_normal@(donor@r), radius*c]
    return gap, np.asarray(gradients)


def _polynomial_value(coefficients, angle):
    basis = (1., math.cos(angle), math.sin(angle), math.cos(2*angle), math.sin(2*angle))
    return sum(float(value.midpoint)*factor for value, factor in zip(coefficients, basis))


@pytest.mark.parametrize("angle", [0., .31, .91])
def test_actual_so3_jacobian_matches_independent_frechet_matrix_derivatives(angle):
    declaration, _, _, values = _sources()
    jacobian = _jacobian(declaration, values)
    _, independent = _independent_point(values, angle)
    for index, expected in enumerate(independent):
        coefficients = tuple(row[index] for row in jacobian.coefficients)
        actual = _polynomial_value(coefficients, angle)
        assert actual == pytest.approx(expected, abs=8e-15)
        affine = sum(float(Decimal(row[index]))*factor for row, factor in zip(
            declaration.normal_operator_coefficients, (1., math.cos(angle), math.sin(angle))))
        assert abs(expected-affine) <= float(jacobian.operator_error_bounds[index])+8e-15


def test_receiver_normal_rotation_is_part_of_the_finite_angular_reaction():
    declaration, _, _, values = _sources(donor_axis=(F(0),)*3)
    jacobian = _jacobian(declaration, values)
    _, expected = _independent_point(values, .61)
    assert max(abs(expected[3:6])) > 1.
    for index in (3, 4, 5):
        assert _polynomial_value(tuple(row[index] for row in jacobian.coefficients), .61) == pytest.approx(expected[index], abs=8e-15)


def test_full_finite_force_and_energy_are_enclosed_against_independent_arc_quadrature():
    declaration, attachment, step, values = _sources()
    fields = bound_born_finite_pose_fields(**values)
    bounds = bound_born_finite_pose_force_error(fields, attachment, step, _jacobian(declaration, values),
        finite_measure_density_m2_per_radian=F(1), finite_angle_end_bounds=I(1))
    # The independently declared columns give k_eff=1/2. No production
    # pressure, arc moments, root finder, or gradient enters this quadrature.
    energy = quad(lambda angle: .25*min(_independent_point(values, angle)[0], 0.)**2, 0., 1., epsabs=1e-13)[0]
    assert float(bounds.finite_energy_bounds_j.lo) <= energy <= float(bounds.finite_energy_bounds_j.hi)
    for index, enclosure in enumerate(bounds.finite_potential_gradient_bounds):
        actual = quad(lambda angle: .5*min(_independent_point(values, angle)[0], 0.)
            *_independent_point(values, angle)[1][index], 0., 1., epsabs=1e-12)[0]
        assert float(enclosure.lo)-1e-13 <= actual <= float(enclosure.hi)+1e-13
    assert bounds.gap_error_bound_m < F(1, 100)
    assert bounds.material_admission_authorized is False


def test_exact_zero_motion_has_only_declared_outward_rounding_error():
    declaration, attachment, step, values = _sources(axis=False)
    bounds = bound_born_finite_pose_force_error(bound_born_finite_pose_fields(**values), attachment, step,
        _jacobian(declaration, values), finite_measure_density_m2_per_radian=F(1), finite_angle_end_bounds=I(1))
    assert bounds.gap_error_bound_m < F(1, 10**90)
    assert max(bounds.reaction_error_bounds) < F(1, 10**90)
    assert bounds.energy_error_bound_j < F(1, 10**90)


def test_exact_gap_match_still_pays_for_nonterminating_decimal_stiffness():
    declaration, _, _, values = _sources(axis=False, precision=256)
    law = replace(declaration.parameters.law, normal_stiffness_pa_m=.5, fracture_energy_j_m2=2.)
    declaration = replace(declaration, parameters=replace(declaration.parameters, law=law))
    previous = activate_born_continuum_contact(declaration, decimal_precision=160)
    attachment = BornContinuumAttachment(declaration, previous)
    with localcontext() as context:
        context.prec = 160
        step = _advance_field(previous, tuple(map(Decimal, previous.gap_coefficients_decimal)),
            (Decimal(0),)*3, 1., 160)
    bounds = bound_born_finite_pose_force_error(bound_born_finite_pose_fields(**values), attachment, step,
        _jacobian(declaration, values), finite_measure_density_m2_per_radian=F(1), finite_angle_end_bounds=I(1))
    # The exact gap is dyadic and unchanged, but k_eff=1/3 is not an exact
    # Decimal160 number. A zero geometric error cannot imply zero force error.
    actual_energy = F.from_float(-.1)**2/6
    assert bounds.gap_error_bound_m == 0
    assert actual_energy != F(Decimal(step.state.stored_energy_decimal))
    assert bounds.affine_energy_reference_error_bound_j > 0
    assert bounds.finite_energy_bounds_j.lo <= actual_energy <= bounds.finite_energy_bounds_j.hi


def test_missing_reported_active_pocket_cannot_be_assumed_to_have_zero_potential():
    epsilon = F(1, 10**30)
    declaration, _, step, _ = _sources(axis=False, affine_gap=(-epsilon, F(0), F(1)))
    # The independent integral deliberately gets no reported closed spans or
    # root knots. It must still retain the real narrow compression at theta=0.
    without_knots = replace(step, closed_angle_parts_decimal=(), root_bands_decimal=())
    energy, _ = _affine_reference_bounds(declaration, without_knots, 192)
    assert energy.hi > 0
    assert energy.hi >= epsilon**3/32
    assert energy.lo <= F(Decimal(step.state.stored_energy_decimal)) <= energy.hi


@pytest.mark.parametrize("finite_sign, affine_sign", [(-1, 1), (1, -1)])
def test_opening_and_closing_crossing_needs_no_shared_active_set(finite_sign, affine_sign):
    epsilon = F(1, 10**30)
    declaration, attachment, step, values = _sources(axis=False, affine_gap=(affine_sign*epsilon, F(0), F(0)))
    fields = BornFinitePoseFields((I(finite_sign*epsilon),)+(I(0),)*4, (I(0),)*5, (I(0),)*5, I(1), 96, 32)
    bounds = bound_born_finite_pose_force_error(fields, attachment, step, _jacobian(declaration, values),
        finite_measure_density_m2_per_radian=F(1), finite_angle_end_bounds=I(1))
    assert 2*epsilon <= bounds.gap_error_bound_m <= 2*epsilon*(1+F(1, 10**90))
    actual_energy = epsilon**2/4 if finite_sign < 0 else F(0)
    assert bounds.finite_energy_bounds_j.lo <= actual_energy <= bounds.finite_energy_bounds_j.hi
    if finite_sign < 0:
        assert bounds.finite_pressure_bounds_pa.lo < 0
        assert max(bounds.reaction_error_bounds) > 0
    else:
        assert bounds.finite_pressure_bounds_pa == I(0)


def test_tiny_closing_tail_has_nonzero_pressure_and_energy_budget_without_area_cutoff():
    epsilon, end = F(1, 10**30), F(1, 10**29)
    declaration, attachment, step, values = _sources(axis=False, end=end, affine_gap=(epsilon, F(0), F(0)))
    fields = BornFinitePoseFields((I(-epsilon), I(0), I(1), I(0), I(0)),
        (I(0),)*5, (I(0),)*5, I(1), 96, 32)
    bounds = bound_born_finite_pose_force_error(fields, attachment, step, _jacobian(declaration, values, I(0, end)),
        finite_measure_density_m2_per_radian=F(1), finite_angle_end_bounds=I(end))
    assert bounds.finite_area_upper_m2 == end
    assert bounds.finite_pressure_bounds_pa.lo < 0
    assert bounds.energy_error_bound_j > 0
    assert max(bounds.reaction_error_bounds) > 0
    # The tail contains theta in [0,epsilon/2], where sin(theta)<=theta,
    # and therefore its physical energy is at least epsilon^3/32.
    assert bounds.finite_energy_bounds_j.hi >= epsilon**3/32


def test_true_native_density_and_endpoint_aliases_are_paid_separately():
    declaration, attachment, step, values = _sources(axis=False)
    native_end = I(F(10001, 10000))
    bounds = bound_born_finite_pose_force_error(bound_born_finite_pose_fields(**values), attachment, step,
        _jacobian(declaration, values, I(0, native_end.hi)),
        finite_measure_density_m2_per_radian=F(2), finite_angle_end_bounds=native_end)
    assert bounds.measure_alias_bound_m2 >= F(10002, 10000)
    actual_energy = F.from_float(-.1)**2*2*native_end.lo/4
    assert bounds.finite_energy_bounds_j.lo <= actual_energy <= bounds.finite_energy_bounds_j.hi


def test_native_frame_uncertainty_survives_the_force_error_bound():
    declaration, attachment, step, values = _sources(axis=False,
        horizontal_unit_bounds=(I(-1), I(F(-1, 10**12), F(1, 10**12)), I(0)))
    bounds = bound_born_finite_pose_force_error(bound_born_finite_pose_fields(**values), attachment, step,
        _jacobian(declaration, values), finite_measure_density_m2_per_radian=F(1), finite_angle_end_bounds=I(1))
    assert max(bounds.reaction_error_bounds) >= F(1, 10**13)


def test_whole_clock_compares_to_the_stored_endpoint_energy_and_refuses_small_force_budget():
    declaration, attachment, step, values = _sources(time=I(0, 1))
    bounds = bound_born_finite_pose_force_error(bound_born_finite_pose_fields(**values), attachment, step,
        _jacobian(declaration, values), finite_measure_density_m2_per_radian=F(1), finite_angle_end_bounds=I(1))
    budget = certify_born_finite_pose_force_budget(bounds, existing_absolute_residual_bounds=(F(0),)*8,
        maximum_absolute_residual=(F(1, 10**12),)*8)
    assert budget.equilibrium_certified is False
    for time in (0., .25, .7, 1.):
        energy = quad(lambda angle: .25*min(_independent_point(values, angle, time)[0], 0.)**2, 0., 1.)[0]
        assert float(bounds.finite_energy_bounds_j.lo)-1e-14 <= energy <= float(bounds.finite_energy_bounds_j.hi)+1e-14


def test_frechet_taylor_bounds_cover_all_interior_clock_and_angle_values():
    declaration, _, _, values = _sources(time=I(F(1, 5), F(9, 10)))
    jacobian = _jacobian(declaration, values)
    for time in (.21, .51, .89):
        for angle in (.03, .47, .98):
            _, expected = _independent_point(values, angle, time)
            basis = (1., math.cos(angle), math.sin(angle), math.cos(2*angle), math.sin(2*angle))
            for index, actual in enumerate(expected):
                # Evaluate the interval coefficient enclosure at this point
                # using separate floating endpoints as an independent oracle.
                low = high = 0.
                for row, factor in zip(jacobian.coefficients, basis):
                    products = (float(row[index].lo)*factor, float(row[index].hi)*factor)
                    low += min(products)
                    high += max(products)
                assert low-2e-14 <= actual <= high+2e-14


def test_jacobian_keeps_second_harmonics_that_birth_only_ports_cannot_represent():
    declaration, _, _, values = _sources()
    jacobian = _jacobian(declaration, values)
    assert max(float(max(abs(value.lo), abs(value.hi)))
        for row in jacobian.coefficients[3:] for value in row) > 1e-4


def test_equilibrium_requires_every_own_new_error_plus_existing_residual_budget():
    declaration, attachment, step, values = _sources()
    bounds = bound_born_finite_pose_force_error(bound_born_finite_pose_fields(**values), attachment, step,
        _jacobian(declaration, values), finite_measure_density_m2_per_radian=F(1), finite_angle_end_bounds=I(1))
    old = (F(1, 100),)*8
    refuse = certify_born_finite_pose_force_budget(bounds, existing_absolute_residual_bounds=old,
        maximum_absolute_residual=old)
    assert refuse.equilibrium_certified is False
    limits = tuple(error+existing for error, existing in zip(bounds.reaction_error_bounds, old))
    accept = certify_born_finite_pose_force_budget(bounds, existing_absolute_residual_bounds=old,
        maximum_absolute_residual=limits)
    assert accept.equilibrium_certified is True
    assert accept.material_admission_authorized is False


def test_operator_uncertainty_cannot_be_erased_from_the_stored_jacobian():
    declaration, _, _, values = _sources()
    jacobian = _jacobian(declaration, values)
    with pytest.raises(ValueError, match="cannot omit"):
        replace(jacobian, operator_error_bounds=(F(0),)*8)


def test_independent_objective_norm_cannot_shrink_the_actual_bound():
    declaration, attachment, step, values = _sources()
    with pytest.raises(ValueError, match="cannot reduce"):
        bound_born_finite_pose_force_error(bound_born_finite_pose_fields(**values), attachment, step,
            _jacobian(declaration, values), finite_measure_density_m2_per_radian=F(1), finite_angle_end_bounds=I(1),
            affine_operator_norm_bounds=(F(0),)*8)


def test_tampered_contact_energy_is_refused_before_constructing_force_bounds():
    declaration, attachment, step, values = _sources()
    corrupted = replace(step, state=replace(step.state, stored_energy_decimal="123"))
    with pytest.raises(ValueError, match="complete field and energy replay"):
        bound_born_finite_pose_force_error(bound_born_finite_pose_fields(**values), attachment, corrupted,
            _jacobian(declaration, values), finite_measure_density_m2_per_radian=F(1), finite_angle_end_bounds=I(1))


def test_truncated_budget_or_repeated_bank_coordinates_cannot_grant_equilibrium():
    declaration, attachment, step, values = _sources()
    with pytest.raises(ValueError, match="distinct"):
        bound_born_finite_pose_jacobian(declaration,
            **{name: values[name] for name in ("start_unit_bounds", "tangent_unit_bounds", "horizontal_unit_bounds",
                "donor_axis", "receiver_axis", "radius_m", "donor_radial_before_m", "donor_radial_increment_m",
                "sine_dip", "cosine_dip", "time_fraction_bounds", "precision", "order")},
            donor_angular_indices=(0, 1, 2), receiver_angular_indices=(0, 4, 5), donor_radial_index=6,
            receiver_radial_index=7, angle_bounds=I(0, 1))
    bounds = bound_born_finite_pose_force_error(bound_born_finite_pose_fields(**values), attachment, step,
        _jacobian(declaration, values), finite_measure_density_m2_per_radian=F(1), finite_angle_end_bounds=I(1))
    with pytest.raises(ValueError, match="Every actual coordinate"):
        certify_born_finite_pose_force_budget(bounds, existing_absolute_residual_bounds=(F(0),)*7,
            maximum_absolute_residual=(F(0),)*8)
