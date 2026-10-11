"""Independent curved-face metric and variational contact controls.

The world-space oracle uses Decimal Rodrigues rotations and finite
differences, rather than production normals, area derivatives, or forces.
Reduced series-column endpoint energy is checked without claiming a closed
three-dimensional strain-energy law.
"""
from dataclasses import replace
from decimal import Decimal, getcontext, localcontext
from fractions import Fraction as F

import pytest

from tectonics.geometric_native_pose import FractionInterval as I
from tectonics.geometric_contact_mechanics import ElasticColumnParameters
from tectonics.genesis_contact_law import ContactLawParameters


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


def _contains(bounds, expected):
    point = F(expected)
    if bounds.lo == bounds.hi:
        assert abs(bounds.lo-point) < F(1, 10**170)*max(F(1), abs(point))
    else:
        assert bounds.lo <= point <= bounds.hi


def _columns():
    return ElasticColumnParameters(1000., 1e9, 4e8, 2000., 2e9, 8e8)


def _law():
    return ContactLawParameters(normal_stiffness_pa_m=1e6,
        tangential_stiffness_pa_m=1e6, cohesion_pa=0., friction_dry=0.,
        friction_wet=0., viscosity_pa_s_m=0.)


def _oracle_geometry(*, radius, roof, reference_roof, depth, direction, tangent, rate,
        owner=(F(0), F(0), F(0))):
    """Differentiate the placed face to obtain its actual world area vector."""
    radius, roof, reference_roof, depth = map(_d, (radius, roof, reference_roof, depth))
    direction, tangent = tuple(map(_d, direction)), tuple(map(_d, tangent))
    owner = tuple(map(_d, owner))

    def point(angle, z):
        ray = tuple(_trig(angle, True)*u+_trig(angle)*t
            for u, t in zip(direction, tangent))
        r = _cbrt((radius+roof)**3-3*radius**2*z)
        # The saved joint reference deformation is held fixed while current
        # radial coordinates move.  Updating tau with current w would erase
        # the original physical radial normal-contact port.
        tau = z+(radius**3-(radius+reference_roof)**3)/(3*radius**2)
        rotated = _rotate(ray, tuple(_d(a)*tau for a in rate))
        return _rotate(tuple(r*x for x in rotated), owner)

    increment = Decimal("1e-110")
    dtheta = tuple((b-a)/(2*increment)
        for a, b in zip(point(-increment, depth), point(increment, depth)))
    ddepth = tuple((b-a)/(2*increment)
        for a, b in zip(point(Decimal(0), depth-increment), point(Decimal(0), depth+increment)))
    area_vector = _cross(dtheta, ddepth)
    area = _dot(area_vector, area_vector).sqrt()
    normal = tuple(x/area for x in area_vector)
    return point(Decimal(0), depth), area_vector, area, normal


def _base_geometry(**changes):
    arguments = dict(radius=F(100), roof=F(0), reference_roof=F(0), depth=F(6, 5),
        direction=(F(3, 5), F(4, 5), F(0)),
        tangent=(F(-4, 5), F(3, 5), F(0)),
        rate=(F(1, 1000), F(-1, 900), F(1, 800)),
        owner=(F(0), F(0), F(0)))
    arguments.update(changes)
    if "roof" in changes and "reference_roof" not in changes:
        arguments["reference_roof"] = changes["roof"]
    return arguments


def _oracle_contact(donor, receiver, reference_donor, reference_receiver, preload):
    donor_point = _oracle_geometry(**donor)[0]
    receiver_point, _, area, normal = _oracle_geometry(**receiver)
    reference_donor_point = _oracle_geometry(**reference_donor)[0]
    reference_receiver_point, _, _, reference_normal = _oracle_geometry(**reference_receiver)
    gap = (_d(preload)+_dot(normal, tuple(b-a for a, b in zip(donor_point, receiver_point)))
        -_dot(reference_normal, tuple(b-a for a, b in zip(reference_donor_point, reference_receiver_point))))
    # Independent exact source series compliance: Lsub/Esub+Lover/Eover+1/k.
    compliance = Decimal(1000)/Decimal("1e9")+Decimal(2000)/Decimal("2e9")+Decimal(1)/Decimal("1e6")
    pressure = -min(gap, Decimal(0))/compliance
    return gap, area*pressure*pressure*compliance/2


def _geometry(arguments):
    from tectonics.geometric_native_curved_contact_energy import bound_curved_face_geometry
    return bound_curved_face_geometry(radius_m=arguments["radius"],
        reference_top_displacement_m=arguments["reference_roof"],
        top_displacement_m=arguments["roof"], material_depth_bounds_m=arguments["depth"],
        direction_bounds=tuple(I(x) for x in arguments["direction"]),
        tangent_bounds=tuple(I(x) for x in arguments["tangent"]),
        normal_bounds=(I(0), I(0), I(1)),
        shared_depth_axis_per_m=arguments["rate"], owner_axis=arguments["owner"])


def _point_response(donor, receiver, reference_donor, reference_receiver, preload=F(-2, 5)):
    from tectonics.geometric_native_curved_contact_energy import bound_curved_contact_point
    return bound_curved_contact_point(*map(_geometry,
        (donor, receiver, reference_donor, reference_receiver)),
        columns=_columns(), law=_law(), source_preload_gap_m=preload)


def _contact_cases():
    return (_base_geometry(roof=F(1, 8), reference_roof=F(0)),
        _base_geometry(roof=F(-1, 5), reference_roof=F(-1, 10)),
        _base_geometry(), _base_geometry(roof=F(-1, 10)))


@pytest.mark.parametrize("changes", [dict(), dict(roof=F(1, 4)),
    dict(roof=F(-1, 3), owner=(F(1, 100), F(1, 110), F(-1, 90))),
    dict(rate=(F(0), F(0), F(0)))])
def test_real_curved_area_and_normal_contain_independent_world_cross_product(changes):
    arguments = _base_geometry(**changes)
    geometry = _geometry(arguments)
    with localcontext() as context:
        context.prec = 360
        point, vector, area, normal = _oracle_geometry(**arguments)
        for actual, expected in zip(geometry.world_position_bounds_m, point):
            _contains(actual, expected)
        for actual, expected in zip(geometry.oriented_area_vector_bounds, vector):
            _contains(actual, expected)
        for actual, expected in zip(geometry.unit_normal_bounds, normal):
            _contains(actual, expected)
        _contains(geometry.area_density_bounds_m2_per_radian_per_material_m, area)
    assert geometry.area_density_bounds_m2_per_radian_per_material_m.lo > 0


def test_area_uses_spherical_curvature_and_shared_shear_instead_of_flat_height():
    curved = _geometry(_base_geometry())
    radial = _geometry(_base_geometry(rate=(F(0), F(0), F(0))))
    assert radial.area_density_bounds_m2_per_radian_per_material_m.lo > 100
    assert curved.area_density_bounds_m2_per_radian_per_material_m.lo > (
        radial.area_density_bounds_m2_per_radian_per_material_m.hi)


def test_roof_area_and_normal_derivatives_match_independent_geometry_finite_differences():
    arguments, increment = _base_geometry(roof=F(1, 4)), F(1, 10**70)
    geometry = _geometry(arguments)
    with localcontext() as context:
        context.prec = 360
        lower = _oracle_geometry(**{**arguments, "roof": arguments["roof"]-increment})
        upper = _oracle_geometry(**{**arguments, "roof": arguments["roof"]+increment})
        scale = 2*_d(increment)
        for actual, a, b in zip(geometry.radial_position_derivative_bounds, lower[0], upper[0]):
            _contains(actual, (b-a)/scale)
        for actual, a, b in zip(geometry.radial_normal_derivative_bounds, lower[3], upper[3]):
            _contains(actual, (b-a)/scale)
        _contains(geometry.radial_area_derivative_bounds, (upper[2]-lower[2])/scale)


def test_activation_reference_retains_actual_rest_projection_and_preload_energy():
    donor, receiver = _base_geometry(), _base_geometry(roof=F(-1, 10))
    preload = F(-2, 5)
    response = _point_response(donor, receiver, donor, receiver, preload)
    assert response.free_gap_bounds_m.lo <= preload <= response.free_gap_bounds_m.hi
    independent_pressure = F(400000, 3)
    _contains(response.normal_pressure_bounds_pa, independent_pressure)
    assert response.effective_normal_stiffness_pa_m == F(1_000_000, 3)
    for energy in response.energy_partition_bounds_j_m2:
        _contains(energy, independent_pressure**2/F(2_000_000))
    assert response.source_preload_gap_m == preload


@pytest.mark.parametrize("coordinate", range(8))
def test_complete_curved_potential_gradient_matches_independent_world_energy(coordinate):
    donor, receiver, reference_donor, reference_receiver = _contact_cases()
    response = _point_response(donor, receiver, reference_donor, reference_receiver)
    increment = F(1, 10**70)
    lower, upper = [dict(donor), dict(receiver)], [dict(donor), dict(receiver)]
    if coordinate < 6:
        bank, axis = divmod(coordinate, 3)
        lower_axis, upper_axis = list(lower[bank]["owner"]), list(upper[bank]["owner"])
        lower_axis[axis] -= increment
        upper_axis[axis] += increment
        lower[bank]["owner"], upper[bank]["owner"] = tuple(lower_axis), tuple(upper_axis)
    else:
        bank = coordinate-6
        lower[bank]["roof"] -= increment
        upper[bank]["roof"] += increment
    with localcontext() as context:
        context.prec = 360
        lower_gap, lower_energy = _oracle_contact(*lower,
            reference_donor, reference_receiver, F(-2, 5))
        upper_gap, upper_energy = _oracle_contact(*upper,
            reference_donor, reference_receiver, F(-2, 5))
        scale = 2*_d(increment)
        _contains(response.gap_gradient_bounds[coordinate], (upper_gap-lower_gap)/scale)
        _contains(response.potential_gradient_bounds[coordinate], (upper_energy-lower_energy)/scale)


def test_receiver_measure_variation_is_a_real_force_term():
    donor, receiver, reference_donor, reference_receiver = _contact_cases()
    response = _point_response(donor, receiver, reference_donor, reference_receiver)
    area = _geometry(receiver).area_density_bounds_m2_per_radian_per_material_m
    pressure = response.normal_pressure_bounds_pa
    pressure_only = -area*pressure*response.gap_gradient_bounds[7]
    full = response.potential_gradient_bounds[7]
    assert full.hi < pressure_only.lo
    assert response.area_gradient_bounds[7].hi < 0


@pytest.mark.parametrize("preload", [F(0), F(1, 10), F(1, 10**600)])
def test_open_and_zero_gap_contact_have_zero_normal_force_and_potential(preload):
    geometry = _base_geometry(depth=F(0),
        direction=(F(1), F(0), F(0)), tangent=(F(0), F(1), F(0)),
        rate=(F(0), F(0), F(0)))
    response = _point_response(geometry, geometry, geometry, geometry, preload)
    assert response.normal_pressure_bounds_pa == I(0)
    assert response.potential_density_bounds_j_per_radian_per_material_m == I(0)
    assert response.potential_gradient_bounds == (I(0),)*8


def test_reduced_bulk_and_fault_endpoint_energies_close_without_new_material_energy():
    geometry = _base_geometry()
    response = _point_response(geometry, geometry, geometry, geometry)
    pressure = response.normal_pressure_bounds_pa
    for shortening in (*response.column_shortening_bounds_m, response.fault_indentation_bounds_m):
        _contains(shortening, F(2, 15))
    summed = sum(response.energy_partition_bounds_j_m2, I(0))
    expected = response.effective_normal_stiffness_pa_m*F(2, 5)**2/2
    assert summed.lo <= expected <= summed.hi
    assert response.complete_three_dimensional_strain_energy_certified is False
    assert response.native_force_equilibrium_certified is False
    assert response.physical_material_admission_authorized is False
    assert response.current_curved_area_and_normal_derivatives_included is True


def test_shared_spatial_rotation_has_zero_net_contact_spin_work():
    donor, receiver, reference_donor, reference_receiver = _contact_cases()
    response = _point_response(donor, receiver, reference_donor, reference_receiver)
    for donor_gradient, receiver_gradient in zip(response.potential_gradient_bounds[:3],
            response.potential_gradient_bounds[3:6]):
        assert donor_gradient == -receiver_gradient


def test_sub_float_compression_retains_positive_pressure_potential_and_all_source_parts():
    geometry = _base_geometry(depth=F(0),
        direction=(F(1), F(0), F(0)), tangent=(F(0), F(1), F(0)),
        rate=(F(0), F(0), F(0)))
    response = _point_response(geometry, geometry, geometry, geometry, -F(1, 10**600))
    assert response.normal_pressure_bounds_pa.lo > 0
    assert response.potential_density_bounds_j_per_radian_per_material_m.lo > 0
    assert all(energy.lo > 0 for energy in response.energy_partition_bounds_j_m2)
    assert float(response.normal_pressure_bounds_pa.lo) == 0.


@pytest.mark.parametrize("changes", [dict(area_density_bounds_m2_per_radian_per_material_m=I(1)),
    dict(unit_normal_bounds=(I(0), I(0), I(-1))),
    dict(radius_m=F(101))])
def test_changed_curved_metric_or_source_labels_cannot_replace_replayed_geometry(changes):
    from tectonics.geometric_native_curved_contact_energy import bound_curved_contact_point
    geometry = _geometry(_base_geometry())
    with pytest.raises(ValueError):
        bound_curved_contact_point(replace(geometry, **changes), geometry, geometry, geometry,
            columns=_columns(), law=_law(), source_preload_gap_m=F(-2, 5))


def test_activation_reference_cannot_silently_change_material_depth_label():
    from tectonics.geometric_native_curved_contact_energy import bound_curved_contact_point
    geometry = _geometry(_base_geometry())
    other = _geometry(_base_geometry(depth=F(1, 2)))
    with pytest.raises(ValueError):
        bound_curved_contact_point(geometry, geometry, other, geometry,
            columns=_columns(), law=_law(), source_preload_gap_m=F(-2, 5))


def test_frozen_reference_shear_retains_nonzero_original_radial_contact_port():
    arguments = _base_geometry()
    geometry = _geometry(arguments)
    response = _point_response(arguments, arguments, arguments, arguments)
    with localcontext() as context:
        context.prec = 360
        point, _, _, normal = _oracle_geometry(**arguments)
        radial_radius = _dot(point, point).sqrt()
        radial_ray = tuple(x/radial_radius for x in point)
        radial_speed = Decimal(100)**2/radial_radius**2
        normal_port = _dot(normal, radial_ray)*radial_speed
        assert abs(normal_port) > Decimal(".01")
        _contains(response.gap_gradient_bounds[6], -normal_port)
        _contains(response.gap_gradient_bounds[7], normal_port)
        _contains(sum((n*p for n, p in zip(geometry.unit_normal_bounds,
            geometry.radial_position_derivative_bounds)), I(0)), normal_port)


def test_overcompression_cannot_collapse_source_columns():
    geometry = _base_geometry()
    with pytest.raises(ValueError, match="collapse"):
        _point_response(geometry, geometry, geometry, geometry, F(-3001))


@pytest.mark.parametrize("changes", [dict(radius=100.), dict(depth=F(-1)),
    dict(roof=F(-100)), dict(rate=(0., 0., 0.)),
    dict(direction=(F(2), F(0), F(0))),
    dict(tangent=(F(3, 5), F(4, 5), F(0)))])
def test_unphysical_or_aliased_curved_geometry_refuses(changes):
    with pytest.raises(ValueError):
        _geometry(_base_geometry(**changes))
