"""Heat inventory, phase averaging and coarse forcing checkpoint contracts."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from tectonics.genesis import GenesisParameters, diagnose
from tectonics.genesis_shell import ShellParameters, rock_enthalpy
from tectonics.genesis_starter_loading import (
    StarterLoadingModel, smooth_mantle_tensor, tidal_stress_cycle,
)
from tectonics.genesis_tides import (SYNCHRONOUS_SPIN, TidalParameters,
                                   advance_tidal_orbit, initial_tidal_orbit)
from tectonics.mesh import build_icosphere


def model(**kwargs):
    return StarterLoadingModel(GenesisParameters(), TidalParameters(
        enabled=True, spin_state=SYNCHRONOUS_SPIN), ShellParameters(**kwargs))


@pytest.fixture(scope="module")
def cooling():
    owner = model()
    initial = owner.initial()
    final, samples = owner.advance(initial, 2.)
    return owner, initial, final, samples


def test_fully_molten_initial_has_no_mechanical_lid():
    owner = model()
    state = owner.initial()
    sample = owner.sample(state)
    assert sample.thermal["surface_melt_fraction"] == 1
    assert sample.thermal["mantle_melt_fraction"] == 1
    assert sample.thermal["ocean_mass_kg"] == 0
    assert sample.lid_thickness_km == 0
    assert not sample.column_depth_limit_reached
    assert sample.column_energy_residual_j_m2 == 0
    assert np.all(state.column_enthalpy == state.column_enthalpy[0])


def test_coarse_request_resolves_cooling_and_preserves_heat_water(cooling):
    owner, initial, final, samples = cooling
    assert final.thermal.time_myr == final.orbit.time_myr == 2.
    assert len(samples) > 2
    sequence = [owner.sample(initial), *samples]
    times = np.array([sample.time_myr for sample in sequence])
    assert np.all(np.diff(times) > 0)
    assert np.max(np.diff(times)) <= .05+2e-14
    for temperature in ("surface_temperature_k", "mantle_temperature_k"):
        assert np.max(np.abs(np.diff([s.thermal[temperature] for s in sequence]))) <= 50.+1e-7
    for sample in sequence:
        row = sample.thermal
        assert row["vapor_mass_kg"]+row["ocean_mass_kg"] == pytest.approx(row["total_water_mass_kg"], rel=2e-15)
        assert abs(row["relative_energy_residual"]) < 2e-12
        assert abs(sample.column_energy_residual_j_m2)/initial.initial_column_energy_j_m2 < 2e-13
    assert samples[-1].lid_thickness_km > 0
    assert samples[-1].thermal["ocean_fraction"] > 0
    assert final.orbit.eccentricity < initial.orbit.eccentricity
    assert final.orbit.dissipated_energy_j > 0


def test_every_sample_has_matching_independent_heat_orbit_checkpoint(cooling):
    owner, _, final, samples = cooling
    for sample in samples:
        assert sample.time_myr == sample.state.thermal.time_myr == sample.state.orbit.time_myr
        check = owner.sample(sample.state)
        assert check.lid_thickness_km == sample.lid_thickness_km
        assert check.mean_lid_temperature_k == sample.mean_lid_temperature_k
        for key in ("surface_temperature_k", "mantle_temperature_k", "ocean_fraction"):
            assert check.thermal[key] == sample.thermal[key]
    assert samples[-1].state.thermal == final.thermal
    assert not np.shares_memory(samples[-1].state.column_enthalpy, final.column_enthalpy)
    assert all(not np.shares_memory(a.state.column_enthalpy, b.state.column_enthalpy)
               for a,b in zip(samples[:-1], samples[1:]))


def test_initial_is_not_mutated_by_trial(cooling):
    owner, initial, _, _ = cooling
    fresh = owner.initial()
    assert initial.thermal == fresh.thermal
    assert initial.orbit == fresh.orbit
    assert initial.boundary_energy_j_m2 == 0
    np.testing.assert_array_equal(initial.column_enthalpy, fresh.column_enthalpy)


def test_sample_checkpoint_resumes_actual_next_interval(cooling):
    owner, _, _, samples = cooling
    old = samples[len(samples)//2].state
    copy = deepcopy(old)
    first, a = owner.advance(old, old.thermal.time_myr+.05)
    second, b = owner.advance(copy, copy.thermal.time_myr+.05)
    assert first.thermal == second.thermal
    assert first.orbit == second.orbit
    np.testing.assert_array_equal(first.column_enthalpy, second.column_enthalpy)
    assert [item.thermal for item in a] == [item.thermal for item in b]


def test_thermal_event_points_are_in_the_sample_history(cooling):
    _, _, final, samples = cooling
    times = np.array([item.time_myr for item in samples])
    for time in final.thermal.events.values():
        assert np.min(np.abs(times-time)) < 2e-10


def test_uniform_column_equilibrium_generates_no_flux_or_lid():
    owner = model()
    old = owner.initial()
    before = owner.sample(old).thermal
    after = dict(before, time_myr=1.)
    following, boundary = owner._conduct(old.column_enthalpy, 0., before, after)
    np.testing.assert_array_equal(following, old.column_enthalpy)
    assert boundary == 0


def test_linear_conductive_profile_has_equal_boundary_fluxes():
    owner = model()
    top, bottom = 300., 2000.
    z = (np.arange(owner.shell.column_layers)+.5)/owner.shell.column_layers
    temperature = top+(bottom-top)*z
    h = rock_enthalpy(temperature, owner.thermal)
    before = {"time_myr": 0., "surface_temperature_k": top, "mantle_temperature_k": bottom}
    after = dict(before, time_myr=.1)
    following, boundary = owner._conduct(h, 0., before, after)
    np.testing.assert_allclose(following, h, rtol=0, atol=2e-8)
    assert abs(boundary) < .01


@pytest.mark.parametrize("argument,value", [("target_myr", 0), ("max_sample_myr", 0),
    ("max_temperature_change_k", float("nan")), ("max_thermal_step_myr", -1)])
def test_invalid_integration_controls_are_rejected(argument, value):
    owner = model()
    kwargs = {"target_myr": .1, argument: value}
    with pytest.raises(ValueError):
        owner.advance(owner.initial(), **kwargs)


def test_mismatched_clocks_and_satellite_are_rejected():
    owner = model()
    state = owner.initial()
    with pytest.raises(ValueError, match="clocks"):
        owner.sample(replace(state, orbit=replace(state.orbit, time_myr=1.)))
    with pytest.raises(ValueError, match="satellite"):
        StarterLoadingModel(GenesisParameters(), TidalParameters(satellite_radius_km=5000), ShellParameters())
    with pytest.raises(ValueError, match="prescribed"):
        StarterLoadingModel(replace(GenesisParameters(), tidal_heat_flux_w_m2=1.),
                            TidalParameters(), ShellParameters())


def test_tidal_cycle_zero_mean_and_damped_amplitude():
    owner, mesh = model(), build_icosphere(1)
    orbit = initial_tidal_orbit(owner.tides)
    first = tidal_stress_cycle(mesh, orbit, owner.tides)
    assert first.shape == (16, mesh.cell_count, 3)
    np.testing.assert_allclose(first.mean(axis=0), 0., atol=1e-8)
    following, _ = advance_tidal_orbit(orbit, owner.tides, .5)
    later = tidal_stress_cycle(mesh, following, owner.tides)
    amplitude = following.eccentricity/orbit.eccentricity*(orbit.semimajor_axis_km/following.semimajor_axis_km)**3
    np.testing.assert_allclose(later, first*amplitude, rtol=3e-14, atol=1e-8)
    assert np.max(np.abs(later)) < np.max(np.abs(first))


def test_disabled_tides_give_zero_stress():
    parameters = TidalParameters()
    assert not np.any(tidal_stress_cycle(build_icosphere(1), initial_tidal_orbit(parameters), parameters))


def test_smooth_mantle_forcing_is_bounded_seeded_and_contains_both_signs():
    mesh = build_icosphere(2)
    a = smooth_mantle_tensor(mesh, 12)
    np.testing.assert_array_equal(a, smooth_mantle_tensor(mesh, 12))
    assert not np.array_equal(a, smooth_mantle_tensor(mesh, 13))
    assert a.shape == (mesh.cell_count, 3)
    average = (a[:, 0]+a[:, 1])/2
    radius = np.hypot((a[:, 0]-a[:, 1])/2, a[:, 2])
    assert np.max(np.abs(np.column_stack((average-radius, average+radius)))) <= 1.+1e-14
    assert average.min() < 0 < average.max()
    for seed in (True, -1, 1.5):
        with pytest.raises(ValueError):
            smooth_mantle_tensor(mesh, seed)
