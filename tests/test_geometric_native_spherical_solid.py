"""Independent curvature-volume and full material-depth lift controls.

The radius oracle uses high-precision Decimal Newton iteration.  SO(3) point
checks use independent rational alternating sine/cosine series, without the
production rotation or radial evaluators.  An explicit layer map is not a
claim that neighbouring implicit material masks glue into a global body.
"""
from decimal import Decimal, localcontext
from fractions import Fraction as F
from math import factorial

import pytest

from tectonics.geometric_native_pose import FractionInterval as I
from tectonics.geometric_native_spherical_solid import (
    SphericalLayerMap, bound_spherical_arc_face, bound_spherical_arc_face_motion,
    bound_spherical_solid_motion, interval_cbrt,
    native_polygon_direction_bounds, spherical_layer_radius_bounds,
    spherical_layer_world_point_bounds,
)


def _map(**changes):
    arguments = dict(radius_m=F(100), top_radial_displacement_m=F(0),
        areal_thickness_m=F(2), volume_stretch=F(1),
        depth_rotation_axis=(F(0),)*3,
        provenance="Explicit volume-preserving whole-depth spherical lift")
    arguments.update(changes)
    return SphericalLayerMap(**arguments)


def _decimal(value):
    return Decimal(value.numerator)/Decimal(value.denominator)


def _independent_cbrt(value):
    with localcontext() as context:
        context.prec = 230
        number = _decimal(value)
        # Scale this positive Newton iteration, independently of production's
        # outward dyadic root grid.  All controls remain far above zero.
        guess = Decimal(10)**(number.adjusted()//3)
        for _ in range(1000):
            after = (2*guess+number/(guess*guess))/3
            if abs(after-guess) < Decimal("1e-225")*max(Decimal(1), abs(after)):
                return F(after)
            guess = after
        raise AssertionError("Independent cube-root Newton iteration failed")


def _trig(theta, *, cosine=False):
    # Tests use |theta|<=1/4.  The first omitted alternating term bounds the
    # exact transcendental result.  No floating world-frame projection.
    first = 0 if cosine else 1
    terms = tuple((-1)**degree*theta**(2*degree+first)/factorial(2*degree+first)
        for degree in range(90))
    center = sum(terms, F(0))
    remainder = theta**(180+first)/factorial(180+first)
    return I(center, center+remainder)


def _radius_cube(value, depth):
    return ((value.radius_m+value.top_radial_displacement_m)**3
        -3*value.radius_m**2*value.volume_stretch*depth)


def _inside(bounds, point):
    # The Decimal oracle is much more accurate than these 128-digit bounds.
    if bounds.lo == bounds.hi:
        assert abs(bounds.lo-point) < F(1, 10**220)*max(F(1), abs(point))
    else:
        assert bounds.lo <= point <= bounds.hi


@pytest.mark.parametrize("number", [F(0), F(1, 8), F(8), F(10**600), F(1, 10**600)])
def test_direct_cube_root_retains_exact_cubes_across_relative_scales(number):
    result = interval_cbrt(I(number), precision=128)
    expected = (F(0) if number == 0 else F(1, 2) if number == F(1, 8)
        else F(2) if number == 8 else F(10**200) if number > 1 else F(1, 10**200))
    assert result == I(expected)


def test_radial_curvature_corrects_depth_and_preserves_full_source_volume():
    value = _map()
    bottom = spherical_layer_radius_bounds(value, I(F(2)), precision=128)
    expected = _independent_cbrt(F(940000))
    _inside(bottom, expected)
    assert bottom.hi < 98
    # A naive physical shell r=R−H loses material volume on a sphere.
    naive_volume_per_steradian = (F(100)**3-F(98)**3)/3
    source_volume_per_steradian = F(100)**2*2
    assert naive_volume_per_steradian < source_volume_per_steradian
    assert (F(100)**3-_radius_cube(value, F(2)))/3 == source_volume_per_steradian


@pytest.mark.parametrize("depth", [F(0), F(1, 10**120), F(1, 7), F(2)])
def test_whole_material_depth_radius_encloses_independent_cube_root(depth):
    value = _map(top_radial_displacement_m=F(11, 7), volume_stretch=F(3, 2))
    result = spherical_layer_radius_bounds(value, I(depth), precision=128)
    _inside(result, _independent_cbrt(_radius_cube(value, depth)))
    if depth == 0:
        assert result == I(F(100)+F(11, 7))


def test_full_depth_interval_contains_all_interior_material_not_only_end_faces():
    value = _map(top_radial_displacement_m=F(10))
    result = spherical_layer_radius_bounds(value, I(0, 2), precision=128)
    for depth in (F(0), F(1, 1000), F(1, 3), F(7, 5), F(1999, 1000), F(2)):
        _inside(result, _independent_cbrt(_radius_cube(value, depth)))
    assert result.hi == 110
    assert result.lo < 110
    # Top displacement does not silently replace the source reference area.
    assert (F(110)**3-_radius_cube(value, F(2)))/3 == 10000*2


def test_volumetric_stretch_is_independent_physics_and_retains_curvature_factor():
    value = _map(volume_stretch=F(3, 2))
    depth1, depth2 = F(1, 7), F(11, 7)
    # Integration of r² dr with the cubic radial law, no thin-layer shortcut.
    exact_volume_per_steradian = (_radius_cube(value, depth1)-_radius_cube(value, depth2))/3
    assert exact_volume_per_steradian == value.radius_m**2*F(3, 2)*(depth2-depth1)
    assert exact_volume_per_steradian > 10000*(depth2-depth1)


def test_zero_depth_shear_places_every_material_axis_at_actual_top_radius():
    value = _map(top_radial_displacement_m=F(5, 8),
        depth_rotation_axis=(F(1, 100), F(-1, 90), F(1, 80)))
    for axis in ((F(1), F(0), F(0)), (F(0), F(1), F(0)), (F(0), F(0), F(1))):
        point = spherical_layer_world_point_bounds(value, tuple(I(x) for x in axis), I(0), precision=128)
        assert point == tuple(I((100+F(5, 8))*x) for x in axis)


def test_depth_rotation_changes_direction_at_depth_and_preserves_radial_volume():
    theta, depth = F(1, 5), F(3, 2)
    value = _map(depth_rotation_axis=(F(0), F(0), theta))
    point = spherical_layer_world_point_bounds(value,
        (I(1), I(0), I(0)), I(depth), precision=128)
    actual_theta = theta*depth/value.areal_thickness_m
    radius = _independent_cbrt(_radius_cube(value, depth))
    independent_x, independent_y = radius*_trig(actual_theta, cosine=True), radius*_trig(actual_theta)
    assert point[0].lo <= independent_x.lo <= independent_x.hi <= point[0].hi
    assert point[1].lo <= independent_y.lo <= independent_y.hi <= point[1].hi
    assert point[2] == I(0)
    # A rotation depending on depth adds cross terms to the Jacobian, but its
    # radial dot product is zero.  It cannot change the integrated shell volume.
    assert (100**3-_radius_cube(value, F(2)))/3 == 20000


def test_nonzero_sub_float_depth_is_not_rounded_onto_the_top_surface():
    value, depth = _map(), F(1, 10**800)
    radius = spherical_layer_radius_bounds(value, I(depth), precision=900)
    assert radius.hi < 100
    assert float(radius.hi) == 100.
    assert radius.lo > 0


def _native_arc():
    from tectonics.geometric_native_polygons import NativeArc, NativePlane, NativeRay
    return NativeArc(NativeRay("seed", (1, 0, 0)),
        NativeRay("seed", (0, 1, 0)), NativePlane((0, 0, 1)),
        1.5707963267948966)


def test_native_curved_face_encloses_every_depth_and_uses_curvature_area_not_flat_carrier():
    value = _map()
    result = bound_spherical_arc_face(value, _native_arc(), I(0, F(1, 10)),
        I(0, 2), precision=128)
    for depth in (F(0), F(1, 11), F(13, 7), F(2)):
        radius = _independent_cbrt(_radius_cube(value, depth))
        for theta in (F(0), F(1, 30), F(1, 10)):
            x, y = radius*_trig(theta, cosine=True), radius*_trig(theta)
            assert result.world_position_bounds_m[0].lo <= x.lo <= x.hi <= result.world_position_bounds_m[0].hi
            assert result.world_position_bounds_m[1].lo <= y.lo <= y.hi <= result.world_position_bounds_m[1].hi
        _inside(result.area_density_bounds_m2_per_radian_per_material_m, F(10000)/radius)
    bottom_face = bound_spherical_arc_face(value, _native_arc(), I(0), I(2), precision=128)
    assert bottom_face.area_density_bounds_m2_per_radian_per_material_m.lo > 100
    assert result.whole_declared_curved_face_enclosed is True
    assert result.full_spherical_footprint_and_clock_collision_excluded is False
    assert result.physical_material_admission_authorized is False


def test_curved_face_depth_shear_area_matches_independent_cross_product_metric():
    axis, theta, depth = F(1, 5), F(1, 10), F(3, 2)
    value = _map(depth_rotation_axis=(axis, F(0), F(0)))
    result = bound_spherical_arc_face(value, _native_arc(), I(theta), I(depth), precision=128)
    radius = _independent_cbrt(_radius_cube(value, depth))
    shear_theta = axis*depth/value.areal_thickness_m
    cosine, sine = _trig(theta, cosine=True), _trig(theta)
    shear_cosine, shear_sine = _trig(shear_theta, cosine=True), _trig(shear_theta)
    independent_position = (radius*cosine, radius*sine*shear_cosine, radius*sine*shear_sine)
    for bounds, expected in zip(result.world_position_bounds_m, independent_position):
        assert bounds.lo <= expected.lo <= expected.hi <= bounds.hi
    with localcontext() as context:
        context.prec = 200
        r, sin_theta, a = _decimal(radius), _decimal(sine.midpoint), _decimal(axis)
        # |dX/dtheta cross dX/dz|²: mutually orthogonal radial and shear terms.
        independent_area = F(((Decimal(10000)/r)**2
            +(r*r*a*sin_theta/Decimal(2))**2).sqrt())
    _inside(result.area_density_bounds_m2_per_radian_per_material_m, independent_area)
    assert result.area_density_bounds_m2_per_radian_per_material_m.lo > F(10000)/radius


def test_native_face_requires_full_original_angle_range_and_real_carrier_type():
    with pytest.raises(ValueError):
        bound_spherical_arc_face(_map(), _native_arc(), I(0, 2), I(0, 2))
    with pytest.raises(ValueError):
        bound_spherical_arc_face(_map(), object(), I(0), I(0))


def test_inconsistent_inherited_native_arc_plane_cannot_become_a_curved_material_face():
    from dataclasses import replace
    from tectonics.geometric_native_polygons import NativePlane
    wrong = replace(_native_arc(), plane=NativePlane((1, 0, 0)))
    with pytest.raises(ValueError):
        bound_spherical_arc_face(_map(), wrong, I(0, F(1, 10)), I(0, 2))


@pytest.mark.parametrize("changed", [dict(radius_m=100.), dict(radius_m=True),
    dict(top_radial_displacement_m=0.), dict(areal_thickness_m=F(0)),
    dict(volume_stretch=F(0)), dict(depth_rotation_axis=(0., 0., 0.)),
    dict(provenance="")])
def test_incomplete_or_aliased_solid_declarations_refuse(changed):
    with pytest.raises(ValueError):
        _map(**changed)


@pytest.mark.parametrize("depth", [I(-F(1, 10**80)), I(2, 3)])
def test_partial_overlap_with_declared_material_depth_is_not_full_membership(depth):
    with pytest.raises(ValueError):
        spherical_layer_radius_bounds(_map(), depth)


def test_layer_that_reaches_center_is_not_an_injective_positive_radius_solid():
    for thickness in (F(100, 3), F(34)):
        with pytest.raises(ValueError):
            _map(areal_thickness_m=thickness)


def test_explicit_map_does_not_invent_neighbor_gluing_contact_or_material_admission():
    value = _map()
    assert value.physical_material_admission_authorized is False
    assert value.whole_partition_noninterpenetration_certified is False
    assert value.source_contact_dip_reconstructed is False
    assert value.whole_component_spherical_embedding_certified is True


def test_whole_clock_and_depth_motion_contains_independent_noncommuting_owner_and_depth_rotations():
    depth_axis, owner_axis = F(1, 5), F(1, 6)
    layer = _map(depth_rotation_axis=(depth_axis, F(0), F(0)))
    arguments = dict(owner_rotation_axis=(F(0), F(0), owner_axis),
        top_radial_increment_m=F(1), time_fraction_bounds=I(0, 1), precision=128)
    result = bound_spherical_solid_motion(layer, (I(0), I(1), I(0)), I(0, 2), **arguments)
    for depth in (F(0), F(1, 7), F(3, 2), F(2)):
        eta = depth_axis*depth/layer.areal_thickness_m
        ce, se = _trig(eta, cosine=True), _trig(eta)
        for clock in (F(0), F(1, 3), F(4, 5), F(1)):
            psi = owner_axis*clock
            cp, sp = _trig(psi, cosine=True), _trig(psi)
            radius = _independent_cbrt((100+clock)**3-30000*depth)
            expected = (-radius*ce*sp, radius*ce*cp, radius*se)
            for bounds, component in zip(result.world_position_bounds_m, expected):
                assert bounds.lo <= component.lo <= component.hi <= bounds.hi
            _inside(result.radial_bounds_m, radius)
    assert result.whole_declared_depth_and_clock_enclosed is True
    assert result.full_spherical_footprint_and_clock_collision_excluded is False
    assert result.physical_material_admission_authorized is False


def test_moving_native_face_keeps_actual_start_tangent_and_whole_interior_clock():
    layer = _map()
    axis = (F(0), F(0), F(1, 5))
    result = bound_spherical_arc_face_motion(layer, _native_arc(), I(0, F(1, 10)), I(0, 2),
        owner_rotation_axis=axis, top_radial_increment_m=F(-1),
        time_fraction_bounds=I(0, 1), precision=128)
    for theta in (F(0), F(1, 50), F(1, 10)):
        for clock, depth in ((F(0), F(0)), (F(2, 3), F(7, 5)), (F(1), F(2))):
            angle = theta+clock/5
            radius = _independent_cbrt((100-clock)**3-30000*depth)
            independent = (radius*_trig(angle, cosine=True), radius*_trig(angle), I(0))
            for bounds, component in zip(result.world_position_bounds_m, independent):
                assert bounds.lo <= component.lo <= component.hi <= bounds.hi
    assert result.full_spherical_footprint_and_clock_collision_excluded is False


def test_whole_clock_positive_top_is_insufficient_if_complete_bottom_reaches_center():
    layer = _map()
    assert layer.bottom_radius_cubed_m3 > 0
    with pytest.raises(ValueError, match="sphere centre"):
        bound_spherical_solid_motion(layer, (I(1), I(0), I(0)), I(0, 2),
            owner_rotation_axis=(F(0),)*3, top_radial_increment_m=F(-61),
            time_fraction_bounds=I(0, 1))


def _polygon(points):
    from tectonics.geometric_native_polygons import NativePlane, NativePolygon, NativeRay, validate_native_polygon
    rays = tuple(NativeRay("seed", point) for point in points)
    def cross(first, second):
        return tuple(first[j]*second[k]-first[k]*second[j]
            for j, k in ((1, 2), (2, 0), (0, 1)))
    planes = tuple(NativePlane(cross(a.reference, b.reference))
        for a, b in zip(rays, rays[1:]+rays[:1]))
    return validate_native_polygon(NativePolygon(rays, planes))


def test_whole_polygon_bounds_include_strict_interior_axis_extremum_missing_from_all_vertices():
    points = ((1, 0, 1), (0, 1, 1), (-1, -1, 1))
    polygon = _polygon(points)
    # +z is strictly inside every inherited halfspace, whereas every vertex
    # has z<1 after unit normalization.  Vertex-only extrema miss material.
    assert all(plane.normal[2] > 0 for plane in polygon.support_planes)
    assert all(point[2]**2 < sum(value*value for value in point) for point in points)
    result = native_polygon_direction_bounds(polygon, precision=128)
    assert result[2].hi == 1
    with localcontext() as context:
        context.prec = 200
        for point in points:
            vertex_z = F(Decimal(point[2])/Decimal(sum(value*value for value in point)).sqrt())
            _inside(result[2], vertex_z)
    assert all(bounds.lo <= 0 <= bounds.hi for bounds in result[:2])


def test_whole_polygon_bounds_include_edge_interior_extremum_missing_from_its_endpoints():
    points = ((1, 0, 1), (0, 1, 1), (-1, 0, 1))
    polygon = _polygon(points)
    # The minor arc from the last vertex to the first passes through +z.
    # All three vertices have z=1/sqrt(2), strictly below that edge maximum.
    assert all(point[2]**2*2 == sum(value*value for value in point) for point in points)
    result = native_polygon_direction_bounds(polygon, precision=128)
    assert result[2].hi == 1
    with localcontext() as context:
        context.prec = 200
        _inside(result[2], F(1/Decimal(2).sqrt()))
