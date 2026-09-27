"""The finite starter column must not stop a valid solid-world evolution."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from tectonics.genesis import GenesisParameters
from tectonics.genesis_long_term_support import begin_mechanical_transition, mechanical_sample, refresh_matched_mechanics
from tectonics.genesis_shell import ShellParameters
from tectonics.genesis_starter import StarterModel
from tectonics.genesis_starter_continuation import YoungWorldCoupling, build_starter_continuation, project_thermal
from tectonics.genesis_tides import TidalParameters, SYNCHRONOUS_SPIN
from tectonics.mesh import build_icosphere
from tectonics.simulation import load_config
from tectonics.thermal import ThermalParameters

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def near_limit():
    model = StarterModel(build_icosphere(2), GenesisParameters(),
        TidalParameters(enabled=True, spin_state=SYNCHRONOUS_SPIN),
        ShellParameters(subdivisions=2, convective_traction_pa=50000.))
    state = model.advance(model.initial_state(), 1.)
    bundle, cfg, _ = build_starter_continuation(model, state, load_config(ROOT / "configs/canonical_moon.yaml"))
    # Move only this controlled thermal fixture near the old domain boundary;
    # real material evolution across it is covered by the CLI replay audit.
    state = deepcopy(state)
    state.thermal_context, _ = model.loading.advance(state.thermal_context, 106.,
        max_sample_myr=.25, max_thermal_step_myr=.25)
    cp = bundle.checkpoint
    cp.thermal, _ = project_thermal(model, state, ThermalParameters(**cfg["thermal"]))
    cp.state.time_myr = state.time_myr
    cp.state.crust_age_myr[:] = state.time_myr
    return model, state, cfg, cp


def test_transition_matches_thickness_temperature_without_replacing_heat_or_memory(near_limit):
    model = near_limit[0]
    state, cfg, cp = deepcopy(near_limit[1:])
    coupling = YoungWorldCoupling(model, state, cfg, cp)
    before_chemistry = cp.state.oceanic_volume_km3.copy()
    expected, samples = model.loading.advance(state.thermal_context, 108.,
        max_sample_myr=model.parameters.max_loading_interval_myr)
    thermal, _ = coupling.advance_heat(cp.thermal, 2.)
    transition = cfg["young_shell"]["mechanical_transition"]
    raw = next(s for s in samples if s.column_depth_limit_reached)
    matched = mechanical_sample(raw, transition)
    assert transition["time_myr"] == raw.time_myr
    assert matched.lid_thickness_km == pytest.approx(raw.lid_thickness_km, abs=1e-12)
    assert matched.mean_lid_temperature_k == pytest.approx(raw.mean_lid_temperature_k, abs=1e-12)
    assert matched.state is raw.state
    np.testing.assert_array_equal(coupling.source_state.thermal_context.thermal.energy, expected.thermal.energy)
    np.testing.assert_array_equal(coupling.source_state.thermal_context.column_enthalpy, expected.column_enthalpy)
    np.testing.assert_array_equal(cp.state.oceanic_volume_km3, before_chemistry)
    assert coupling.source_state.thermal_context.orbit == expected.orbit
    assert thermal.time_myr == coupling.fracture.time_myr == 108.
    assert thermal.thermal_lithosphere_thickness_km > model.shell.column_depth_km
    assert np.any(coupling.fracture.memory.water_access > 0)
    assert abs(samples[-1].thermal["relative_energy_residual"]) < 1e-10


def test_after_transition_new_crust_and_continents_use_mature_local_mechanics(near_limit):
    model = near_limit[0]
    source, cfg, cp = deepcopy(near_limit[1:])
    coupling = YoungWorldCoupling(model, source, cfg, cp)
    thermal, _ = coupling.advance_heat(cp.thermal, 2.)
    state = cp.state
    state.time_myr = thermal.time_myr
    state.continental_fraction[2] = 1.
    state.crust_age_myr[0] = 0.
    state.crust_age_myr[1] = 1.
    coupling.mechanical_fields(state, 1.)
    assert state.mantle_lithosphere_thickness_km[0] == 0.
    assert 0 < state.mantle_lithosphere_thickness_km[1] < state.mantle_lithosphere_thickness_km[3]
    assert state.mantle_lithosphere_thickness_km[3]+cfg["lithosphere"]["oceanic_thickness_km"] == pytest.approx(
        thermal.thermal_lithosphere_thickness_km)
    assert state.mantle_lithosphere_thickness_km[2] > 0


def test_mature_plate_cooling_cap_does_not_stop_global_heat(near_limit):
    model = near_limit[0]
    state, cfg, cp = deepcopy(near_limit[1:])
    coupling = YoungWorldCoupling(model, state, cfg, cp)
    thermal, _ = coupling.advance_heat(cp.thermal, 2.)
    sample = model.loading.sample(coupling.source_state.thermal_context)
    far = replace(sample, thermal=dict(sample.thermal, time_myr=4500.))
    support = mechanical_sample(far, cfg["young_shell"]["mechanical_transition"])
    assert support.lid_thickness_km == 155.
    assert not support.column_depth_limit_reached
    assert support.state is sample.state


def test_invalid_matching_parameters_and_false_limit_are_rejected(near_limit):
    model, source, _, _ = near_limit
    sample = model.loading.sample(source.thermal_context)
    with pytest.raises(ValueError, match="support limit"):
        begin_mechanical_transition(sample, model, {})
    sample = replace(sample, lid_thickness_km=model.shell.column_depth_km)
    with pytest.raises(ValueError, match="cannot continue"):
        begin_mechanical_transition(sample, model, {"oceanic_max_total_thickness_km": 20.})


def test_mixed_roots_cannot_grow_from_repeated_zero_time_refreshes(near_limit):
    state = deepcopy(near_limit[3].state)
    state.continental_fraction[:] = .5
    state.mantle_lithosphere_thickness_km[:] = 45.
    state.mantle_lithosphere_density_anomaly_kg_m3[:] = 10.
    before = deepcopy(state)
    for _ in range(3):
        refresh_matched_mechanics(state, 0., 12., 1300.)
    np.testing.assert_array_equal(state.mantle_lithosphere_thickness_km, before.mantle_lithosphere_thickness_km)
    np.testing.assert_array_equal(state.mantle_lithosphere_density_anomaly_kg_m3, before.mantle_lithosphere_density_anomaly_kg_m3)
    whole, halves = deepcopy(before), deepcopy(before)
    whole.crust_age_myr += 1.
    refresh_matched_mechanics(whole, 1., 13., 1300.)
    for age_cap in (12.5, 13.):
        halves.crust_age_myr += .5
        refresh_matched_mechanics(halves, .5, age_cap, 1300.)
    np.testing.assert_allclose(whole.mantle_lithosphere_thickness_km, halves.mantle_lithosphere_thickness_km,
        rtol=0., atol=2e-13)
    np.testing.assert_allclose(whole.mantle_lithosphere_density_anomaly_kg_m3,
        halves.mantle_lithosphere_density_anomaly_kg_m3, rtol=0., atol=2e-13)


def test_failed_fracture_step_cannot_publish_mechanical_transition(near_limit, monkeypatch):
    from tectonics.genesis_starter_fracture import YoungShellFracture
    model = near_limit[0]
    source, cfg, cp = deepcopy(near_limit[1:])
    coupling = YoungWorldCoupling(model, source, cfg, cp)
    before = deepcopy(coupling.source_state.thermal_context)
    def fail(*args, **kwargs):
        raise ValueError("injected fracture failure")
    monkeypatch.setattr(YoungShellFracture, "advance", fail)
    with pytest.raises(ValueError, match="injected"):
        coupling.advance_heat(cp.thermal, 2.)
    assert "mechanical_transition" not in cfg["young_shell"]
    assert coupling.source_state.time_myr == before.thermal.time_myr
    np.testing.assert_array_equal(coupling.source_state.thermal_context.column_enthalpy, before.column_enthalpy)
    np.testing.assert_array_equal(coupling.source_state.thermal_context.thermal.energy, before.thermal.energy)
