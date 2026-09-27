"""Independent heat, water and mechanical-lid contracts for young continuation."""
from copy import copy, deepcopy
from dataclasses import fields, is_dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tectonics.genesis import GenesisParameters
from tectonics.genesis_shell import ShellParameters
from tectonics.genesis_starter import StarterModel
from tectonics.genesis_starter_continuation import (YoungWorldCoupling,
    build_starter_continuation, project_thermal)
from tectonics.genesis_tides import (TidalParameters, SYNCHRONOUS_SPIN,
                                   advance_tidal_orbit)
from tectonics.hydrosphere import HydrosphereParameters, advance_hydrosphere
from tectonics.lithosphere import target_mantle_lithosphere_fields
from tectonics.mesh import build_icosphere
from tectonics.simulation import load_config
from tectonics.thermal import ThermalParameters
from tectonics.topology import PlateTopologyManager

ROOT = Path(__file__).resolve().parents[1]


def equal(a, b):
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    if is_dataclass(a):
        return all(equal(getattr(a, f.name), getattr(b, f.name)) for f in fields(a))
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[key], b[key]) for key in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


@pytest.fixture(scope="module")
def source():
    model = StarterModel(build_icosphere(2), GenesisParameters(),
        TidalParameters(enabled=True, spin_state=SYNCHRONOUS_SPIN),
        ShellParameters(subdivisions=2, convective_traction_pa=50000.))
    state = model.advance(model.initial_state(), 1.)
    assert state.stopped_reason == "first_partition"
    bundle, cfg, _ = build_starter_continuation(model, state,
        load_config(ROOT/"configs/canonical_moon.yaml"))
    return model, state, cfg, bundle.checkpoint


def coupling_from(source):
    model, state, cfg, checkpoint = source
    checkpoint = deepcopy(checkpoint)
    return YoungWorldCoupling(model, state, deepcopy(cfg), checkpoint), checkpoint


def test_heat_hook_matches_direct_genesis_advance_exactly(source):
    coupling, cp = coupling_from(source)
    owner = coupling.model
    original = deepcopy(coupling.source_state)
    end = cp.thermal.time_myr+1.
    expected, samples = owner.loading.advance(original.thermal_context, end,
        max_sample_myr=owner.parameters.max_loading_interval_myr)
    thermal, diag = coupling.advance_heat(cp.thermal, 1.)
    assert equal(coupling.source_state.thermal_context, expected)
    assert coupling.source_state.thermal_samples == original.thermal_samples+len(samples)
    assert thermal.time_myr == diag.time_myr == expected.thermal.time_myr
    assert thermal.system_age_myr == end+owner.thermal.system_age_at_start_myr
    assert thermal.mantle_temperature_k == samples[-1].thermal["mantle_temperature_k"]
    assert thermal.reference_convective_flux_w_m2 == cp.thermal.reference_convective_flux_w_m2
    assert diag.convective_heat_flux_w_m2 == samples[-1].thermal["mantle_to_surface_flux_w_m2"]
    assert diag.eccentricity == expected.orbit.eccentricity
    assert abs(samples[-1].thermal["relative_energy_residual"]) < 1e-12
    # Mature weakening owns these fields; the starter law must not run again.
    np.testing.assert_array_equal(coupling.source_state.damage, original.damage)
    np.testing.assert_array_equal(coupling.source_state.cooling_stress_pa, original.cooling_stress_pa)
    assert cp.thermal.time_myr == original.time_myr


def test_orbital_queries_match_direct_accepted_orbit_and_reject_outside_interval(source):
    coupling, cp = coupling_from(source)
    first = cp.thermal.time_myr
    before = coupling.source_state.thermal_context.orbit
    thermal, _ = coupling.advance_heat(cp.thermal, .1)
    for time in (first, first+.025, first+.05, thermal.time_myr):
        expected = advance_tidal_orbit(before, coupling.model.tides, time)[0].eccentricity
        assert coupling.at(time) == expected
    for time in (first-.001, thermal.time_myr+.001):
        with pytest.raises(ValueError, match="outside accepted"):
            coupling.at(time)


def test_thermal_projection_preserves_system_age_offset_and_column_lid(source):
    model, state, _, _ = source
    altered = copy(model)
    altered.thermal = replace(model.thermal, system_age_at_start_myr=42.)
    # The projection's absolute age convention does not reset the elapsed clock.
    thermal, diag = project_thermal(altered, state, ThermalParameters())
    assert thermal.time_myr == state.time_myr
    assert thermal.system_age_myr == state.time_myr+42.
    assert diag.system_age_myr == thermal.system_age_myr
    sample = model.loading.sample(state.thermal_context)
    assert thermal.thermal_lithosphere_thickness_km == sample.lid_thickness_km


def test_mechanical_hook_uses_young_column_not_reset_chemical_age(source):
    coupling, cp = coupling_from(source)
    coupling.advance_heat(cp.thermal, 2.)
    young = coupling.model.loading.sample(coupling.source_state.thermal_context)
    a, b = deepcopy(cp.state), deepcopy(cp.state)
    a.crust_age_myr[:] = 0.
    b.crust_age_myr[:] = 1000.
    coupling.mechanical_fields(a)
    coupling.mechanical_fields(b)
    expected = np.maximum(young.lid_thickness_km-a.crust_thickness_km, 0.)
    np.testing.assert_array_equal(a.mantle_lithosphere_thickness_km, expected)
    np.testing.assert_array_equal(a.mantle_lithosphere_thickness_km, b.mantle_lithosphere_thickness_km)
    np.testing.assert_array_equal(a.mantle_lithosphere_density_anomaly_kg_m3,
                                  b.mantle_lithosphere_density_anomaly_kg_m3)
    mature_age_target, _ = target_mantle_lithosphere_fields(b)
    assert np.max(np.abs(mature_age_target-expected)) > 50.
    assert np.all(expected > 0)
    np.testing.assert_array_equal(a.crust_thickness_km, cp.state.crust_thickness_km)


def isolated_runner():
    """Exercise the real hook and hydrosphere without changing legacy globals."""
    return SimpleNamespace(base=SimpleNamespace(
        advance_hydrosphere=advance_hydrosphere,
        update_plate_dynamics=lambda *args, **kwargs: None,
        PlateTopologyManager=PlateTopologyManager,
        remap_transport_state=lambda *args, **kwargs: None),
        v124=SimpleNamespace(_original_refresh_mechanical_lithosphere=None,
                             _original_advance_lithosphere=lambda *args, **kwargs: None))


def test_liquid_water_hook_condenses_only_existing_inventory(source):
    coupling, cp = coupling_from(source)
    runner = isolated_runner()
    coupling.install(runner)
    original_hydro = deepcopy(cp.hydrosphere)
    assert original_hydro.water_volume_km3 == 0.
    owner = coupling.model
    params = HydrosphereParameters(subgrid_material_hypsometry=False)
    dry, _ = runner.base.advance_hydrosphere(owner.mesh, cp.state, cp.topo,
        cp.hydrosphere, owner.thermal.radius_km, params)
    assert dry.water_volume_km3 == 0.
    thermal, _ = runner.base.advance_thermal_state(cp.thermal, 1.)
    cp.state.time_myr = cp.topo.time_myr = thermal.time_myr
    ocean, diag = runner.base.advance_hydrosphere(owner.mesh, cp.state, cp.topo,
        cp.hydrosphere, owner.thermal.radius_km, params)
    row = owner.loading.sample(coupling.source_state.thermal_context).thermal
    assert ocean.time_myr == thermal.time_myr
    assert 0 < ocean.water_volume_km3 < owner.thermal.water_volume_km3
    assert ocean.water_volume_km3 == row["ocean_fraction"]*owner.thermal.water_volume_km3
    liquid_mass = ocean.water_volume_km3*1e9*owner.thermal.water_density_kg_m3
    assert liquid_mass+row["vapor_mass_kg"] == pytest.approx(row["total_water_mass_kg"], rel=3e-15)
    assert abs(diag.relative_volume_error) < 1e-7
    assert cp.hydrosphere == original_hydro


@pytest.mark.parametrize("skew", [5e-10, 1e-5])
def test_clock_guard_uses_absolute_tolerance(source, skew):
    coupling, cp = coupling_from(source)
    bad = replace(cp.thermal, time_myr=cp.thermal.time_myr+skew)
    with pytest.raises(ValueError, match="clocks"):
        coupling.advance_heat(bad, .1)


@pytest.mark.parametrize("dt", [0., -.1, float("nan"), float("inf")])
def test_nonadvancing_or_nonfinite_time_is_rejected(source, dt):
    coupling, cp = coupling_from(source)
    before = deepcopy(coupling.source_state.thermal_context)
    with pytest.raises(ValueError):
        coupling.advance_heat(cp.thermal, dt)
    assert equal(before, coupling.source_state.thermal_context)


def injected_samples(coupling, thermal, dt, *, remelt_middle=False, remelt_end=False, cap_middle=False):
    context, samples = coupling.model.loading.advance(coupling.source_state.thermal_context,
        thermal.time_myr+dt, max_sample_myr=coupling.model.parameters.max_loading_interval_myr)
    samples = list(samples)
    assert len(samples) >= 2
    if remelt_middle:
        samples[0] = replace(samples[0], thermal=dict(samples[0].thermal, surface_melt_fraction=.1))
    if remelt_end:
        samples[-1] = replace(samples[-1], thermal=dict(samples[-1].thermal, surface_melt_fraction=.1))
    if cap_middle:
        samples[0] = replace(samples[0], column_depth_limit_reached=True)
    return context, samples


@pytest.mark.parametrize("changes,match", [({"remelt_end": True}, "remelting"),
    ({"remelt_middle": True}, "remelting"), ({"cap_middle": True}, "support limit")])
def test_unsupported_intermediate_or_endpoint_state_rejects_whole_interval(source, monkeypatch, changes, match):
    coupling, cp = coupling_from(source)
    thermal, _ = coupling.advance_heat(cp.thermal, .1)
    old_context = deepcopy(coupling.source_state.thermal_context)
    old_orbit = coupling.before_orbit
    old_count = coupling.source_state.thermal_samples
    previous_midpoint = .5*(old_orbit.time_myr+thermal.time_myr)
    previous_query = coupling.at(previous_midpoint)
    proposed = injected_samples(coupling, thermal, .1, **changes)
    monkeypatch.setattr(coupling.model.loading, "advance", lambda *args, **kwargs: proposed)
    with pytest.raises(ValueError, match=match):
        coupling.advance_heat(thermal, .1)
    assert equal(old_context, coupling.source_state.thermal_context)
    assert coupling.source_state.thermal_samples == old_count
    assert coupling.before_orbit == old_orbit
    assert coupling.at(previous_midpoint) == previous_query
