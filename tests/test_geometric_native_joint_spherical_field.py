"""Independent controls for a shared physical-radius spherical deformation.

The oracle differentiates and rotates in Decimal arithmetic; it never calls
the production radial, rotation, or face-metric evaluators.  The exact linear
constraint controls include rank deficiency and inconsistent redundant rows.
"""
from dataclasses import replace
from decimal import Decimal, getcontext, localcontext
from fractions import Fraction as F

import pytest

from tectonics.geometric_native_pose import FractionInterval as I


def _d(value):
    if type(value) is Decimal:
        return value
    value = F(value)
    return Decimal(value.numerator)/Decimal(value.denominator)


def _dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def _cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2],
        a[0]*b[1]-a[1]*b[0])


def _trig(angle, cosine=False):
    term = Decimal(1) if cosine else angle
    total = term
    for index in range(1, 110):
        first = 2*index-1 if cosine else 2*index
        term *= -angle*angle/(first*(first+1))
        total += term
    return total


def _rotate(vector, axis):
    square = _dot(axis, axis)
    if not square:
        return vector
    theta = square.sqrt()
    unit = tuple(x/theta for x in axis)
    cosine, sine = _trig(theta, True), _trig(theta)
    cross, projection = _cross(unit, vector), _dot(unit, vector)
    return tuple(cosine*x+sine*y+(1-cosine)*projection*z
        for x, y, z in zip(vector, cross, unit))


def _cbrt(number):
    guess = Decimal(10)**(number.adjusted()//3)
    for _ in range(1000):
        after = (2*guess+number/guess**2)/3
        if abs(after-guess) <= Decimal(1).scaleb(4-getcontext().prec)*max(Decimal(1), abs(after)):
            return after
        guess = after
    raise AssertionError("Independent cube-root Newton iteration failed")


def _oracle(radius, roof, depth, rate, direction):
    radius, roof, depth = map(_d, (radius, roof, depth))
    r = _cbrt((radius+roof)**3-3*radius**2*depth)
    tau = (radius**3-r**3)/(3*radius**2)
    point = _rotate(tuple(map(_d, direction)), tuple(_d(a)*tau for a in rate))
    return tuple(r*x for x in point)


def _contains(bounds, expected):
    point = F(expected)
    if bounds.lo == bounds.hi:
        assert abs(bounds.lo-point) < F(1, 10**175)*max(F(1), abs(point))
    else:
        assert bounds.lo <= point <= bounds.hi


def _det(columns):
    return _dot(columns[0], _cross(columns[1], columns[2]))


def _solve(rows):
    from tectonics.geometric_native_joint_spherical_field import solve_exact_minimum_norm_angular_rate
    return solve_exact_minimum_norm_angular_rate(rows)


def test_three_independent_carrier_constraints_define_one_literal_global_rate():
    result = _solve((((F(1), F(2), F(0)), F(7)),
        ((F(0), F(1), F(3)), F(14)),
        ((F(2), F(0), F(1)), F(8))))
    assert result == (F(27, 13), F(32, 13), F(50, 13))


def test_one_carrier_minimum_norm_rate_has_no_unconstrained_spin():
    normal, target = (F(2), F(3), F(-6)), F(7)
    result = _solve(((normal, target),))
    assert result == tuple(target*x/F(49) for x in normal)
    assert _dot(normal, result) == target
    assert _cross(normal, result) == (0, 0, 0)


def test_two_distinct_carrier_axes_share_one_rate_without_arbitrary_null_rotation():
    first, second = (F(1), F(1), F(0)), (F(0), F(1), F(1))
    result = _solve(((first, F(3)), (second, F(4))))
    assert result == (F(2, 3), F(7, 3), F(5, 3))
    assert _dot(result, _cross(first, second)) == 0
    assert sum(a*a for a in result) < sum(a*a for a in (F(3), F(0), F(4)))


def test_consistent_scaled_and_zero_redundant_rows_preserve_the_same_minimum_norm():
    rows = (((F(1), F(1), F(0)), F(3)), ((F(0), F(1), F(1)), F(4)))
    redundant = (((F(2), F(2), F(0)), F(6)),
        ((F(0), F(0), F(0)), F(0)),
        ((F(-3), F(-3), F(0)), F(-9)))
    assert _solve(rows+redundant) == _solve(rows)
    assert _solve(tuple(reversed(rows+redundant))) == _solve(rows)


def test_inconsistent_redundant_carriers_refuse_even_below_float_resolution():
    rows = (((F(1), F(0), F(0)), F(1)),
        ((F(2), F(0), F(0)), F(2)+F(1, 10**600)))
    with pytest.raises(ValueError):
        _solve(rows)


def test_nonzero_zero_axis_constraint_cannot_be_averaged_into_a_rate():
    with pytest.raises(ValueError):
        _solve((((F(0), F(0), F(0)), F(1, 10**600)),))


def test_constraint_free_components_receive_exact_zero_shear():
    assert _solve(()) == (F(0), F(0), F(0))


@pytest.mark.parametrize("rows", [
    [((F(1), F(0), F(0)), F(1))],
    (((1., F(0), F(0)), F(1)),),
    (((True, F(0), F(0)), F(1)),),
    (((F(1), F(0)), F(1)),),
    (((F(1), F(0), F(0)), 1.),),
    (((F(1), F(0), F(0)), I(1)),),
])
def test_constraint_solver_refuses_mutable_or_aliased_literal_sources(rows):
    with pytest.raises((ValueError, TypeError)):
        _solve(rows)


def _layer(**changes):
    from tectonics.geometric_native_joint_spherical_field import JointSphericalLayerMap
    arguments = dict(radius_m=F(100), top_radial_displacement_m=F(0),
        areal_thickness_m=F(2),
        angular_rate_per_physical_volume_depth_m=(F(1, 100), F(-1, 120), F(1, 90)),
        provenance="Explicit shared physical-volume-radius flow")
    arguments.update(changes)
    return JointSphericalLayerMap(**arguments)


@pytest.mark.parametrize("direction", [(F(1), F(0), F(0)), (F(0), F(3, 5), F(4, 5))])
@pytest.mark.parametrize("depth", [F(0), F(1, 10**100), F(1, 3), F(2)])
def test_complete_depth_point_contains_independent_rodrigues_and_curvature_oracle(direction, depth):
    from tectonics.geometric_native_joint_spherical_field import bound_joint_layer_point
    layer = _layer(top_radial_displacement_m=F(1, 4))
    bounds = bound_joint_layer_point(layer, tuple(I(x) for x in direction), I(depth), precision=128)
    with localcontext() as context:
        context.prec = 190
        expected = _oracle(100, F(1, 4), depth,
            layer.angular_rate_per_physical_volume_depth_m, direction)
        for interval, point in zip(bounds, expected):
            _contains(interval, point)


def test_unequal_roofs_and_thicknesses_glue_at_identical_physical_radius():
    from tectonics.geometric_native_joint_spherical_field import bound_joint_layer_point, bound_joint_layer_radius_point
    donor = _layer(areal_thickness_m=F(2))
    receiver = _layer(top_radial_displacement_m=F(1), areal_thickness_m=F(4))
    cubes = F(99)**3
    donor_depth = (F(100)**3-cubes)/F(30000)
    receiver_depth = (F(101)**3-cubes)/F(30000)
    assert donor_depth != receiver_depth
    assert donor.top_physical_volume_depth_m+donor_depth == (
        receiver.top_physical_volume_depth_m+receiver_depth)
    direction = (I(F(3, 5)), I(F(4, 5)), I(0))
    donor_point = bound_joint_layer_point(donor, direction, I(donor_depth))
    receiver_point = bound_joint_layer_point(receiver, direction, I(receiver_depth))
    assert donor_point == receiver_point
    assert donor_point == bound_joint_layer_radius_point(donor, direction, I(cubes))


def test_material_depth_alias_really_separates_banks_with_different_roofs():
    from tectonics.geometric_native_joint_spherical_field import bound_joint_layer_point
    donor, receiver = _layer(), _layer(top_radial_displacement_m=F(1))
    direction, depth = (I(1), I(0), I(0)), I(F(1))
    first = bound_joint_layer_point(donor, direction, depth)
    second = bound_joint_layer_point(receiver, direction, depth)
    assert second[0].lo > first[0].hi
    assert donor.top_physical_volume_depth_m != receiver.top_physical_volume_depth_m


def test_actual_roof_direction_changes_under_nonzero_roof_displacement():
    from tectonics.geometric_native_joint_spherical_field import bound_joint_layer_point
    layer = _layer(top_radial_displacement_m=F(1),
        angular_rate_per_physical_volume_depth_m=(F(0), F(0), F(1, 100)))
    point = bound_joint_layer_point(layer, (I(1), I(0), I(0)), I(0))
    # Top tau is negative, so the common flow moves the inherited roof ray.
    assert point[1].hi < 0
    assert point[0].hi < 101
    assert point != (I(101), I(0), I(0))


def test_whole_depth_enclosure_contains_deep_interior_points_of_shared_flow():
    from tectonics.geometric_native_joint_spherical_field import bound_joint_layer_point
    layer = _layer(top_radial_displacement_m=F(1, 4))
    point = bound_joint_layer_point(layer, (I(1), I(0), I(0)), I(0, 2))
    with localcontext() as context:
        context.prec = 190
        for depth in (F(0), F(1, 13), F(7, 11), F(3, 2), F(2)):
            for interval, expected in zip(point,
                    _oracle(100, F(1, 4), depth,
                        layer.angular_rate_per_physical_volume_depth_m,
                        (F(1), F(0), F(0)))):
                _contains(interval, expected)


def test_independent_world_point_jacobian_preserves_volume_despite_depth_shear():
    layer = _layer(top_radial_displacement_m=F(1, 4))
    with localcontext() as context:
        context.prec = 190
        theta, latitude, depth = Decimal(".1"), Decimal(".2"), Decimal("1.3")
        increment = Decimal("1e-60")

        def point(a, b, z):
            ray = (_trig(b, True)*_trig(a, True),
                _trig(b, True)*_trig(a), _trig(b))
            return _oracle(100, F(1, 4), z,
                layer.angular_rate_per_physical_volume_depth_m, ray)

        columns = []
        for coordinate in range(3):
            lower, upper = [theta, latitude, depth], [theta, latitude, depth]
            lower[coordinate] -= increment
            upper[coordinate] += increment
            columns.append(tuple((b-a)/(2*increment)
                for a, b in zip(point(*lower), point(*upper))))
        physical_jacobian = _det(columns)
        expected = _d(layer.signed_volume_jacobian)*_trig(latitude, True)
        assert abs(physical_jacobian-expected) < Decimal("1e-110")
    assert layer.physical_volume_jacobian == F(10000)


def test_volume_radius_inverse_identifies_material_depth_and_reference_ray():
    from tectonics.geometric_native_joint_spherical_field import bound_joint_layer_point
    layer = _layer(top_radial_displacement_m=F(1, 4))
    direction, depth = (F(3, 5), F(4, 5), F(0)), F(7, 5)
    point = bound_joint_layer_point(layer, tuple(I(x) for x in direction), I(depth))
    with localcontext() as context:
        context.prec = 190
        expected = _oracle(100, F(1, 4), depth,
            layer.angular_rate_per_physical_volume_depth_m, direction)
        r = _dot(expected, expected).sqrt()
        reconstructed_depth = (Decimal("100.25")**3-r**3)/Decimal(30000)
        tau = (Decimal(100)**3-r**3)/Decimal(30000)
        inverse = _rotate(tuple(x/r for x in expected),
            tuple(-_d(a)*tau for a in layer.angular_rate_per_physical_volume_depth_m))
        assert abs(reconstructed_depth-_d(depth)) < Decimal("1e-175")
        for interval, position, recovered, initial in zip(point, expected, inverse, direction):
            _contains(interval, position)
            assert abs(recovered-_d(initial)) < Decimal("1e-175")


def test_positive_sub_float_depth_is_retained_in_shared_radius_model():
    from tectonics.geometric_native_joint_spherical_field import bound_joint_layer_point
    layer = _layer(angular_rate_per_physical_volume_depth_m=(F(0), F(0), F(0)))
    point = bound_joint_layer_point(layer, (I(1), I(0), I(0)), I(F(1, 10**800)), precision=900)
    assert point[0].hi < 100
    assert float(point[0].hi) == 100.
    assert point[1:] == (I(0), I(0))


@pytest.mark.parametrize("changes", [dict(radius_m=100.),
    dict(top_radial_displacement_m=True), dict(areal_thickness_m=F(0)),
    dict(areal_thickness_m=F(34)),
    dict(angular_rate_per_physical_volume_depth_m=(0., 0., 0.)),
    dict(angular_rate_per_physical_volume_depth_m=(F(1), F(0), F(0))),
    dict(provenance="")])
def test_incomplete_or_noninjective_joint_declarations_refuse(changes):
    with pytest.raises(ValueError):
        _layer(**changes)


@pytest.mark.parametrize("depth", [I(-F(1, 10**200)), I(2, 3)])
def test_full_requested_depth_must_belong_to_declared_material(depth):
    from tectonics.geometric_native_joint_spherical_field import bound_joint_layer_point
    with pytest.raises(ValueError):
        bound_joint_layer_point(_layer(), (I(1), I(0), I(0)), depth)


def test_pure_geometric_flow_cannot_authorize_physical_material_admission():
    assert _layer().physical_material_admission_authorized is False
