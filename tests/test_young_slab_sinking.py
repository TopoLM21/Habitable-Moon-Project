"""Independent power and invariance checks for the finite prescribed slab."""
from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.integrate import quad

from tectonics.young_boundary import SlabForceSection
from tectonics.young_slab_sinking import slab_sinking_system


def parameters(**changes):
    values = dict(gravity_m_s2=4., young_slab_viscosity_contrast=100.,
                  young_slab_bend_radius_thickness_ratio=3.,
                  young_slab_mantle_shear_length_fraction=.5)
    values.update(changes)
    return SimpleNamespace(**values)


def section(length_km=500., width_km=600., thickness_km=50., density=60.):
    area = length_km*width_km
    return SlabForceSection("1:2", 0, 1, 0, 1, width_km, area,
        length_km, length_km*np.sin(np.deg2rad(55.)), 55., area*7.,
        area*thickness_km, area*thickness_km**3,
        density*area*thickness_km*1e9, (1., 0., 0.), (0., 0., 1.))


def build(sections=None, *, params=None, incoming=None, viscosity=1e21):
    return slab_sinking_system([section()] if sections is None else sections,
        2, 5000., viscosity, 2000., params or parameters(),
        incoming_thickness_km=incoming)


def independent_integrals(s):
    """Scalar adaptive quadrature, independent of production Gauss rules."""
    theta = np.deg2rad(s.dip_deg)
    length = s.slab_length_km*1000.
    h = (s.area_thickness_cubed_km5/s.accepted_area_km2)**(1/3)*1000.
    bend_length = 2*theta*3*h
    def angle(x):
        if x >= bend_length:
            return theta
        return theta*(x/bend_length-np.sin(2*np.pi*x/bend_length)/(2*np.pi))
    def gradient_square(x):
        if x >= bend_length:
            return 0.
        return (2*np.pi*theta/bend_length**2*np.sin(2*np.pi*x/bend_length))**2
    split = [bend_length] if bend_length < length else []
    sine = quad(lambda x: np.sin(angle(x)), 0., length, points=split, epsabs=1e-7)[0]/length
    cosine = quad(lambda x: np.cos(angle(x)), 0., length, points=split, epsabs=1e-7)[0]/length
    # Rescale the integral to avoid a tiny absolute-tolerance target.
    grad = quad(lambda x: gradient_square(x)*bend_length**3,
                0., length, points=split, epsabs=1e-11)[0]/bend_length**3
    return sine, cosine, grad, h


@pytest.mark.parametrize("length_km", [.4, 80., 500., 1800.])
def test_scalar_two_plate_operator_matches_adaptive_work_oracle(length_km):
    s = section(length_km=length_km)
    system = build([s])
    sine, cosine, gradient, h = independent_integrals(s)
    radius = 5e6
    length, width = length_km*1000., s.trench_length_km*1000.
    c_b = 1e23*width*h**3/3*gradient
    c_m = 2e21*width*length/1e6
    expected_bend = radius**2*c_b*np.array([[1., -1.], [-1., 1.]])
    expected_mantle = radius**2*c_m*np.array([[1., cosine-1.], [cosine-1., 2.-2.*cosine]])
    ids = np.array([2, 5])
    np.testing.assert_allclose(system.drag_bending_nm_s[np.ix_(ids, ids)], expected_bend, rtol=4e-12)
    np.testing.assert_allclose(system.drag_mantle_nm_s[np.ix_(ids, ids)], expected_mantle,
                               rtol=4e-12, atol=c_m*radius**2*4e-15)
    expected_force = 4.*s.density_excess_mass_kg*sine
    np.testing.assert_allclose(system.driving_torque_nm[:, 2], radius*expected_force*np.array([1., -1.]), rtol=4e-11)
    # A scalar basal resistance makes the exact two-variable problem coercive.
    matrix = (system.drag_total_nm_s[np.ix_(ids, ids)] + radius**2*2e27*np.eye(2))
    torque = system.driving_torque_nm.ravel()[ids]
    omega = np.linalg.solve(matrix, torque)
    assert omega[0] > omega[1]
    assert omega @ torque == pytest.approx(omega @ matrix @ omega, rel=3e-15)


def test_positive_dissipation_and_source_power_equal_direct_velocity_integrals():
    s = section(length_km=800.)
    system = build([s])
    rng = np.random.default_rng(7321)
    omega = rng.normal(size=(2, 3))*1e-15
    power = system.power_diagnostics(omega)
    norm = np.linalg.norm(system.drag_total_nm_s)
    np.testing.assert_allclose(system.drag_total_nm_s, system.drag_total_nm_s.T, atol=norm*1e-16)
    assert np.linalg.eigvalsh(system.drag_total_nm_s).min() >= -norm*1e-14
    assert power["slab_bending_dissipation_w"] >= 0.
    assert power["slab_mantle_dissipation_w"] >= 0.
    q = 5e6*(omega[0, 2]-omega[1, 2])
    vo_y, vo_z = 5e6*omega[1, 2], -5e6*omega[1, 1]
    sine, cosine, gradient, h = independent_integrals(s)
    width, length = s.trench_length_km*1000., s.slab_length_km*1000.
    expected_mantle = 2e21*width*length/1e6*(vo_y**2+vo_z**2+2*vo_y*q*cosine+q*q)
    expected_bending = 1e23*width*h**3/3*gradient*q*q
    assert power["slab_mantle_dissipation_w"] == pytest.approx(expected_mantle, rel=2e-14)
    assert power["slab_bending_dissipation_w"] == pytest.approx(expected_bending, rel=2e-14)
    assert power["slab_gravitational_power_w"] == pytest.approx(4.*s.density_excess_mass_kg*q*sine, rel=2e-14)
    expected_feed_drag = 2e21*width*length/1e6*(vo_y*cosine+q)
    assert np.dot(system.sections[0]["mantle_feed_drag_row_n_s"], omega.ravel()) == pytest.approx(expected_feed_drag, rel=2e-14)


def test_common_rotation_has_no_feed_or_gravity_work_but_has_absolute_mantle_drag():
    system = build()
    omega = np.tile([2e-16, -3e-16, 4e-16], (2, 1))
    power = system.power_diagnostics(omega)
    scale = abs(float(system.driving_torque_nm[0] @ omega[0]))
    assert abs(power["slab_gravitational_power_w"]) <= scale*1e-15
    assert abs(power["slab_bending_dissipation_w"]) < power["slab_mantle_dissipation_w"]*1e-14
    assert power["slab_mantle_dissipation_w"] > 0.


def test_spatial_rotation_covariance():
    original = section()
    rng = np.random.default_rng(927)
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    rotated = replace(original, midpoint=tuple(rotation@original.midpoint),
                      torque_direction=tuple(rotation@original.torque_direction))
    first, second = build([original]), build([rotated])
    transform = np.kron(np.eye(2), rotation)
    np.testing.assert_allclose(second.driving_torque_nm, first.driving_torque_nm@rotation.T,
                               atol=np.linalg.norm(first.driving_torque_nm)*2e-15)
    for name in ("drag_bending_nm_s", "drag_mantle_nm_s"):
        a, b = getattr(first, name), getattr(second, name)
        np.testing.assert_allclose(b, transform@a@transform.T, atol=np.linalg.norm(a)*2e-15)


def test_plate_relabeling_changes_only_block_order():
    original = section()
    swapped = replace(original, subducting_plate=1, overriding_plate=0)
    first, second = build([original]), build([swapped])
    indices = [3, 4, 5, 0, 1, 2]
    np.testing.assert_array_equal(second.driving_torque_nm, first.driving_torque_nm[::-1])
    np.testing.assert_array_equal(second.drag_total_nm_s, first.drag_total_nm_s[np.ix_(indices, indices)])


def test_width_partition_conserves_operator_without_cohort_hinge_duplication():
    whole = section()
    fraction = .23
    parts = []
    for k, weight in enumerate((fraction, 1.-fraction)):
        parts.append(replace(whole, contact_key=f"part{k}",
            trench_length_km=whole.trench_length_km*weight,
            accepted_area_km2=whole.accepted_area_km2*weight,
            cold_mantle_volume_km3=whole.cold_mantle_volume_km3*weight,
            oceanic_volume_km3=whole.oceanic_volume_km3*weight,
            area_thickness_cubed_km5=whole.area_thickness_cubed_km5*weight,
            density_excess_mass_kg=whole.density_excess_mass_kg*weight))
    first, second = build([whole]), build(parts)
    np.testing.assert_allclose(second.driving_torque_nm, first.driving_torque_nm, rtol=2e-15)
    np.testing.assert_allclose(second.drag_total_nm_s, first.drag_total_nm_s,
                               rtol=3e-15, atol=np.linalg.norm(first.drag_total_nm_s)*1e-17)


def test_short_slab_has_continuous_finite_geometry_and_zero_material_limit():
    long, short = build([section(length_km=.1)]), build([section(length_km=.05)])
    assert np.linalg.norm(short.driving_torque_nm)/np.linalg.norm(long.driving_torque_nm) == pytest.approx(1/16., rel=1e-5)
    assert np.linalg.norm(short.drag_bending_nm_s)/np.linalg.norm(long.drag_bending_nm_s) == pytest.approx(1/8., rel=1e-5)
    assert np.linalg.norm(short.drag_mantle_nm_s)/np.linalg.norm(long.drag_mantle_nm_s) == pytest.approx(1/2., rel=1e-5)
    assert short.sections[0]["tip_dip_deg"] < long.sections[0]["tip_dip_deg"] < 1e-7
    zero = build([replace(section(), accepted_area_km2=0.)])
    assert not zero.sections
    assert zero.feed_matrix_m.shape == (0, 6)
    assert np.count_nonzero(zero.drag_total_nm_s) == 0
    assert np.count_nonzero(zero.driving_torque_nm) == 0


def test_heating_removes_buoyancy_without_erasing_material_resistance():
    cold = section()
    warm = replace(cold, density_excess_mass_kg=0.)
    first, second = build([cold]), build([warm])
    np.testing.assert_array_equal(first.drag_total_nm_s, second.drag_total_nm_s)
    assert np.count_nonzero(second.driving_torque_nm) == 0


def test_chemical_ocean_without_cold_mantle_has_no_thermal_slab_operator():
    chemical = replace(section(), density_excess_mass_kg=0.,
                       cold_mantle_volume_km3=0., area_thickness_cubed_km5=0.)
    result = build([chemical])
    assert not result.sections
    assert np.count_nonzero(result.driving_torque_nm) == 0
    assert np.count_nonzero(result.drag_total_nm_s) == 0


def test_missing_mantle_parameters_fail_with_clear_value_error():
    with pytest.raises(ValueError, match="mantle_viscosity_pa_s"):
        build(viscosity=None)


def test_resistance_contrast_controls_feed_without_adjusting_buoyancy():
    ids = np.array([2, 5])
    def feed(contrast):
        system = build(params=parameters(young_slab_viscosity_contrast=contrast))
        matrix = system.drag_total_nm_s[np.ix_(ids, ids)]+1e41*np.eye(2)
        omega = np.linalg.solve(matrix, system.driving_torque_nm.ravel()[ids])
        return omega[0]-omega[1], system
    lower, first = feed(10.)
    higher, second = feed(1000.)
    assert 0 < higher < lower
    np.testing.assert_array_equal(first.driving_torque_nm, second.driving_torque_nm)
    np.testing.assert_array_equal(first.drag_mantle_nm_s, second.drag_mantle_nm_s)


def test_signed_buoyancy_and_negative_feed_are_reported_without_clipping():
    normal = section()
    negative = replace(normal, density_excess_mass_kg=-normal.density_excess_mass_kg)
    first, second = build([normal]), build([negative])
    np.testing.assert_array_equal(second.driving_torque_nm, -first.driving_torque_nm)
    omega = np.array([[0., 0., -1e-15], [0., 0., 0.]])
    powers = first.power_diagnostics(omega)
    np.testing.assert_allclose(first.feed_matrix_m @ omega.ravel(), [-5e-9], rtol=1e-15)
    assert powers["negative_feed_section_count"] == 1
    assert powers["slab_gravitational_power_w"] < 0
    assert powers["slab_bending_dissipation_w"] > 0
    assert powers["slab_mantle_dissipation_w"] > 0


def test_incoming_thickness_and_zero_newborn_fallback_are_explicit():
    thin = build(incoming=np.array([20., 50.]))
    newborn = build(incoming=np.array([0., 50.]))
    assert thin.sections[0]["cold_hinge_thickness_km"] == 20.
    assert thin.sections[0]["thickness_origin"] == "incoming_cold_mantle"
    assert newborn.sections[0]["cold_hinge_thickness_km"] == pytest.approx(50.)
    assert newborn.sections[0]["thickness_origin"] == "retained_cubic_mean"


@pytest.mark.parametrize("changes", [
    {"young_slab_viscosity_contrast": -1.},
    {"young_slab_bend_radius_thickness_ratio": float("nan")},
    {"young_slab_mantle_shear_length_fraction": 0.},
    {"gravity_m_s2": float("inf")},
])
def test_invalid_physical_parameters_fail_before_assembly(changes):
    with pytest.raises(ValueError):
        build(params=parameters(**changes))


def layered_section(base, intervals):
    layers = tuple(SimpleNamespace(arc_start_km=start, arc_end_km=end,
        age_myr=float(i), density_excess_mass_kg=mass) for i, (start, end, mass) in enumerate(intervals))
    values = asdict(base)
    values["buoyancy_layers"] = layers
    return SimpleNamespace(**values)


def test_uniform_mass_density_in_ordered_cohorts_recovers_legacy_gravity():
    base = section(length_km=500.)
    mass = base.density_excess_mass_kg
    layered = layered_section(base, [(0., 80., .16*mass), (80., 240., .32*mass), (240., 500., .52*mass)])
    legacy, ordered = build([base]), build([layered])
    np.testing.assert_allclose(ordered.driving_torque_nm, legacy.driving_torque_nm, rtol=3e-14)
    np.testing.assert_array_equal(ordered.drag_total_nm_s, legacy.drag_total_nm_s)
    assert ordered.sections[0]["buoyancy_distribution_model"] == "ordered_thermal_cohorts_v1"
    assert len(ordered.sections[0]["buoyancy_layers"]) == 3


def test_cold_shallow_material_is_not_given_the_dip_of_a_warm_deep_tail():
    base = section(length_km=500.)
    mass = base.density_excess_mass_kg
    shallow = build([layered_section(base, [(0., 50., mass), (50., 500., 0.)])])
    deep = build([layered_section(base, [(0., 450., 0.), (450., 500., mass)])])
    uniform = build([base])
    assert np.linalg.norm(shallow.driving_torque_nm) < .02*np.linalg.norm(uniform.driving_torque_nm)
    assert np.linalg.norm(deep.driving_torque_nm) > np.linalg.norm(uniform.driving_torque_nm)
    np.testing.assert_array_equal(shallow.drag_total_nm_s, deep.drag_total_nm_s)


def test_ordered_gravity_matches_independent_interval_quadrature_and_power():
    base = section(length_km=500.)
    mass = base.density_excess_mass_kg
    intervals = [(0., 20., .4*mass), (20., 170., .5*mass), (170., 500., .1*mass)]
    result = build([layered_section(base, intervals)])
    dip = np.deg2rad(base.dip_deg)
    bend = 2.*dip*3.*50.
    def angle(s):
        return dip if s >= bend else dip*(s/bend-np.sin(2*np.pi*s/bend)/(2*np.pi))
    expected = 0.
    for start, end, local_mass in intervals:
        breaks = [bend] if start < bend < end else []
        expected += 4.*local_mass/(end-start)*quad(lambda x:np.sin(angle(x)), start, end, points=breaks)[0]
    np.testing.assert_allclose(result.driving_torque_nm[:, 2], 5e6*expected*np.array([1., -1.]), rtol=3e-13)
    omega = np.array([[0., 0., 2e-16], [0., 0., -1e-16]])
    q = 5e6*3e-16
    assert result.power_diagnostics(omega)["slab_gravitational_power_w"] == pytest.approx(expected*q, rel=3e-13)


def test_splitting_and_reordering_a_thermal_cohort_preserves_force():
    base = section(length_km=500.)
    mass = base.density_excess_mass_kg
    first = build([layered_section(base, [(0., 200., .9*mass), (200., 500., .1*mass)])])
    second = build([layered_section(base, [(200., 500., .1*mass), (0., 40., .18*mass), (40., 200., .72*mass)])])
    np.testing.assert_allclose(second.driving_torque_nm, first.driving_torque_nm, rtol=5e-14)
    np.testing.assert_array_equal(second.drag_total_nm_s, first.drag_total_nm_s)


@pytest.mark.parametrize("intervals", [
    [(0., 200., .5), (201., 500., .5)],
    [(0., 200., .5), (199., 500., .5)],
    [(0., 200., .5), (200., 501., .5)],
    [(0., 200., .5), (200., 500., .4)],
])
def test_invalid_material_interval_partition_cannot_create_gravity(intervals):
    base = section(length_km=500.)
    scaled = [(start, end, fraction*base.density_excess_mass_kg) for start, end, fraction in intervals]
    with pytest.raises(ValueError, match="Ordered slab thermal"):
        build([layered_section(base, scaled)])
