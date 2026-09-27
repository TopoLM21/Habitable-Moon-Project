from dataclasses import replace
import math

import numpy as np
import pytest

from tectonics.genesis import GenesisParameters, M_EARTH, SECONDS_PER_MYR
from tectonics.genesis_tides import (
    G, SYNCHRONOUS_SPIN, TidalParameters, advance_tidal_orbit,
    initial_tidal_orbit, mean_motion_rad_s, tidal_diagnostics,
    tidal_heat_flux_w_m2, tidal_heat_power_w, tidal_parameters_from_config,
    tidal_strain_coefficients, tidal_strain_cycle, validate_tidal_orbit,
)
from tectonics.mesh import build_icosphere


def enabled(**kwargs):
    return TidalParameters(enabled=True, spin_state=SYNCHRONOUS_SPIN, **kwargs)


@pytest.mark.parametrize("field,value", [
    ("eccentricity", -1), ("eccentricity", .051), ("eccentricity", float("nan")),
    ("primary_mass_kg", float("inf")), ("semimajor_axis_km", 0),
    ("k2_over_q", -1), ("phase_samples", 8.5), ("phase_samples", True),
    ("enabled", 1), ("spin_state", "asynchronous"),
])
def test_invalid_parameters(field, value):
    with pytest.raises(ValueError):
        replace(enabled(), **{field: value}).validate()


def test_explicit_spin_and_canonical_body():
    with pytest.raises(ValueError, match="explicit"):
        TidalParameters(enabled=True).validate()
    thermal = replace(GenesisParameters(), radius_km=5100, mass_earth=.45)
    p = tidal_parameters_from_config({"genesis_tides": {
        "enabled": True, "spin_state": SYNCHRONOUS_SPIN, "orbit_source": "record",
        "satellite_radius_km": 10}}, thermal)
    assert p.satellite_radius_km == 5100
    assert p.satellite_mass_kg == .45*M_EARTH
    assert not tidal_parameters_from_config({}).enabled
    with pytest.raises(ValueError):
        tidal_parameters_from_config({"genesis_tides": {"surprise": 1}})


def test_heat_reference_and_scaling():
    p = enabled()
    a, r = p.semimajor_axis_km*1000, p.satellite_radius_km*1000
    n = math.sqrt(G*(p.primary_mass_kg+p.satellite_mass_kg)/a**3)
    expected = 10.5*p.k2_over_q*G*p.primary_mass_kg**2*r**5*n*p.eccentricity**2/a**6
    assert tidal_heat_power_w(p) == pytest.approx(expected, rel=1e-14)
    assert tidal_heat_power_w(replace(p, eccentricity=2*p.eccentricity)) == pytest.approx(4*expected)
    assert tidal_heat_power_w(replace(p, semimajor_axis_km=2*p.semimajor_axis_km)) == pytest.approx(expected/2**7.5)
    assert tidal_heat_flux_w_m2(p) == pytest.approx(expected/(4*math.pi*r*r))
    assert 1 < tidal_diagnostics(p)["period_days"] < 3


def test_yaml_scientific_notation_and_numeric_rejection():
    import yaml
    from pathlib import Path
    config = yaml.safe_load((Path(__file__).parents[1]/"configs/genesis_moon.yaml").read_text(encoding="utf-8"))
    assert tidal_parameters_from_config(config).primary_mass_kg == pytest.approx(9.49065e27)
    for section in ({"primary_mass_kg": True}, {"eccentricity": None}, {"phase_samples": 8.5}):
        with pytest.raises(ValueError):
            tidal_parameters_from_config({"genesis_tides": section})


@pytest.mark.parametrize("p", [TidalParameters(), enabled(eccentricity=0)])
def test_disabled_and_circular_are_exactly_zero(p):
    assert tidal_heat_power_w(p) == 0
    assert not np.any(tidal_strain_cycle(build_icosphere(1), p))
    old = initial_tidal_orbit(p)
    new, flux = advance_tidal_orbit(old, p, 100)
    assert replace(new, time_myr=0) == old
    assert flux == 0


def test_elastic_nondissipative_response():
    p = enabled(k2_over_q=0)
    assert tidal_heat_power_w(p) == 0
    assert np.max(np.abs(tidal_strain_cycle(build_icosphere(1), p))) > 0
    old = initial_tidal_orbit(p)
    new, flux = advance_tidal_orbit(old, p, 10)
    assert replace(new, time_myr=0) == old
    assert flux == 0


def test_cycle_zero_mean_engineering_shear_and_face_basis():
    p, mesh = enabled(), build_icosphere(1)
    cosine, sine = tidal_strain_coefficients(mesh, p)
    cycle = tidal_strain_cycle(mesh, p)
    assert cycle.shape == (16, mesh.cell_count, 3)
    np.testing.assert_allclose(cycle.mean(axis=0), 0, atol=1e-20)
    np.testing.assert_allclose(cycle[0], cosine)
    np.testing.assert_allclose(cycle[4], sine, atol=1e-20)
    np.testing.assert_allclose(cycle[8], -cosine, atol=1e-20)
    np.testing.assert_allclose(tidal_strain_cycle(mesh, replace(p, eccentricity=2*p.eccentricity)), 2*cycle)
    # Independent finite difference of the potential on the great circles in
    # each FEM basis checks both Hessian signs and the engineering-shear factor.
    xyz = mesh.vertices[mesh.faces[7]]
    t1 = xyz[1]-xyz[0]; t1 /= np.linalg.norm(t1)
    normal = np.cross(xyz[1]-xyz[0], xyz[2]-xyz[0]); normal /= np.linalg.norm(normal)
    t2 = np.cross(normal, t1)
    q = G*p.primary_mass_kg*p.satellite_radius_km*1000/(p.surface_gravity_m_s2*(p.semimajor_axis_km*1000)**3)
    def f(n):
        return q*p.eccentricity*3*(3*n[0]**2-1)/2
    def second(t):
        h = 1e-4
        return (f(normal*np.cos(h)+t*np.sin(h))-2*f(normal)+f(normal*np.cos(h)-t*np.sin(h)))/h**2
    h11, h22 = second(t1), second(t2)
    h12 = second((t1+t2)/math.sqrt(2))-(h11+h22)/2
    expected = [p.love_h2*f(normal)+p.love_l2*h11,
                p.love_h2*f(normal)+p.love_l2*h22, 2*p.love_l2*h12]
    np.testing.assert_allclose(cosine[7], expected, rtol=2e-6, atol=1e-15)


def test_orbital_damping_closes_energy_and_angular_momentum():
    p = enabled(eccentricity=.02)
    initial = initial_tidal_orbit(p)
    final, flux = advance_tidal_orbit(initial, p, .3)
    assert 0 < final.eccentricity < p.eccentricity
    assert final.semimajor_axis_km < p.semimajor_axis_km
    assert final.semimajor_axis_km*(1-final.eccentricity**2) == pytest.approx(p.semimajor_axis_km*(1-p.eccentricity**2), rel=1e-14)
    amin = p.semimajor_axis_km*1000*(1-p.eccentricity**2)
    expected = G*p.primary_mass_kg*p.satellite_mass_kg/(2*amin)*(p.eccentricity**2-final.eccentricity**2)
    assert final.dissipated_energy_j == pytest.approx(expected, rel=2e-11)
    assert flux*.3*SECONDS_PER_MYR*4*math.pi*(p.satellite_radius_km*1000)**2 == pytest.approx(expected, rel=2e-11)
    validate_tidal_orbit(final, p)
    part, _ = advance_tidal_orbit(initial, p, .1)
    resumed, _ = advance_tidal_orbit(part, p, .3)
    assert resumed.eccentricity == pytest.approx(final.eccentricity, rel=2e-10)
    assert resumed.dissipated_energy_j == pytest.approx(final.dissipated_energy_j, rel=2e-10)


def test_small_steps_and_exhausted_reservoir():
    p = enabled()
    initial = initial_tidal_orbit(p)
    short, flux = advance_tidal_orbit(initial, p, 1e-12)
    assert flux == pytest.approx(tidal_heat_flux_w_m2(p), rel=1e-9)
    validate_tidal_orbit(short, p)
    final, _ = advance_tidal_orbit(initial, p, 1e6)
    assert final.eccentricity == 0
    budget = G*p.primary_mass_kg*p.satellite_mass_kg/(2*p.semimajor_axis_km*1000*(1-p.eccentricity**2))*p.eccentricity**2
    assert final.dissipated_energy_j == pytest.approx(budget, rel=1e-14)
    later, flux = advance_tidal_orbit(final, p, 2e6)
    assert later.dissipated_energy_j == final.dissipated_energy_j
    assert flux == 0
    validate_tidal_orbit(later, p)


@pytest.mark.parametrize("changes", [
    {"time_myr": float("nan")}, {"eccentricity": .01},
    {"semimajor_axis_km": 700000}, {"dissipated_energy_j": 1e30},
])
def test_corrupt_orbit_rejected(changes):
    p = enabled()
    with pytest.raises(ValueError):
        validate_tidal_orbit(replace(initial_tidal_orbit(p), **changes), p)


def test_orbit_time_validation():
    p = enabled()
    state = initial_tidal_orbit(p)
    assert advance_tidal_orbit(state, p, 0) == (state, 0)
    for target in (-1, float("nan"), float("inf"), True):
        with pytest.raises(ValueError):
            advance_tidal_orbit(state, p, target)
