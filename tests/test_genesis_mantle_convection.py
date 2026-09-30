"""Energy, signed transport and convergence of the shared Genesis heat law."""
from dataclasses import replace

import numpy as np
import pytest

from tectonics.genesis import (
    ENERGY_SCALE, GenesisParameters, GenesisState, advance, diagnose, fluxes,
    initial_state, load_checkpoint, mantle_enthalpy, mantle_transport,
    outgoing_longwave_w_m2, parameters_from_config, run_genesis, save_checkpoint,
    surface_enthalpy, temperatures,
)
from tectonics.thermal import ThermalParameters, convective_state


def state_at(tm, ts, p):
    energy = [mantle_enthalpy(tm, p)/ENERGY_SCALE,
              surface_enthalpy(ts, p)/ENERGY_SCALE, 0., 0.]
    return GenesisState(0., energy, sum(energy), {})


def test_solid_branch_uses_same_physics_as_mature_at_actual_surface_temperature():
    p = GenesisParameters()
    for tm in (700., 1000., 1300., 1500.):
        for ts in (282., 350., 700.):
            row = mantle_transport(tm, ts, p)
            mature = convective_state(tm, p.radius_km, p.surface_gravity_m_s2,
                                     replace(ThermalParameters(), surface_temperature_k=ts))
            assert row["magma_transport_weight"] == 0.
            assert row["viscosity_pa_s"] == mature[0]
            if tm > ts:
                assert row["rayleigh_number"] == mature[1]
                assert row["nusselt_number"] == mature[2]
                assert row["mantle_to_surface_flux_w_m2"] == pytest.approx(mature[3], rel=3e-15)
                assert row["thermal_boundary_layer_thickness_km"] == mature[4]


def test_cooling_increases_viscosity_and_reduces_convection_and_loss():
    p = GenesisParameters()
    rows = [mantle_transport(t, 282., p) for t in (1550., 1400., 1200., 1000.)]
    viscosity = [r["viscosity_pa_s"] for r in rows]
    assert np.all(np.diff(viscosity) >= 0)
    assert viscosity[1] > viscosity[0]
    assert viscosity[-1] == p.viscosity_max_pa_s  # Existing rheology validity cap.
    assert np.all(np.diff([r["rayleigh_number"] for r in rows]) < 0)
    assert np.all(np.diff([r["nusselt_number"] for r in rows]) <= 0)
    assert np.all(np.diff([r["mantle_to_surface_flux_w_m2"] for r in rows]) < 0)
    assert np.all(np.diff([r["thermal_boundary_layer_thickness_km"] for r in rows]) >= 0)
    cold = rows[-1]
    assert cold["nusselt_number"] == 1.
    assert cold["mantle_to_surface_flux_w_m2"] == pytest.approx(
        p.thermal_conductivity_w_m_k*(1000.-282.)/(p.radius_km*1000*p.mantle_depth_fraction_radius))
    assert cold["mantle_to_surface_flux_w_m2"] < .01


@pytest.mark.parametrize("boundary", [.25, .55])
def test_melt_transition_is_continuous_in_flux_temperature_and_enthalpy(boundary):
    p = GenesisParameters()
    tm = p.solidus_k+boundary*(p.liquidus_k-p.solidus_k)
    eps = 1e-5
    probes = [tm-eps, tm, tm+eps]
    q = [mantle_transport(t, 282., p)["mantle_to_surface_flux_w_m2"] for t in probes]
    assert max(q)/min(q) < 1.000001
    left, right = (q[1]-q[0])/eps, (q[2]-q[1])/eps
    assert left == pytest.approx(right, rel=1e-3, abs=1e-7)
    for t in probes:
        s = state_at(t, 282., p)
        recovered, _ = temperatures(np.asarray(s.energy), p)
        assert recovered == pytest.approx(t, abs=1e-10)
    h = [mantle_enthalpy(t, p) for t in probes]
    assert h[2]-h[1] == pytest.approx(h[1]-h[0], rel=1e-6)


def test_small_melt_changes_cannot_jump_transport_by_a_factor():
    p = GenesisParameters()
    melts = np.linspace(.249, .551, 3021)
    q = np.array([mantle_transport(p.solidus_k+phi*(p.liquidus_k-p.solidus_k), 282., p)
                  ["mantle_to_surface_flux_w_m2"] for phi in melts])
    assert np.all(np.diff(q) > 0)
    assert np.max(q[1:]/q[:-1]) < 1.01
    for t in (1800., 2000., 2300.):
        assert mantle_transport(t, 400., p)["mantle_to_surface_flux_w_m2"] == p.magma_transfer_w_m2_k*(t-400.)


def test_isothermal_and_inverted_gradients_have_no_spurious_outward_flux():
    p = GenesisParameters()
    for t in (1000., 1560., 2300.):
        equal = mantle_transport(t, t, p)
        assert equal["mantle_to_surface_flux_w_m2"] == 0.
        assert equal["rayleigh_number"] == 0.
        assert equal["nusselt_number"] == 1.
        assert mantle_transport(t, t+10., p)["mantle_to_surface_flux_w_m2"] < 0.


def test_deprecated_transfer_does_not_change_any_new_transport():
    p = GenesisParameters()
    altered = replace(p, solid_transfer_w_m2_k=1000.)
    altered.validate()
    for tm in (1000., 1560., 1640., 1800., 2300.):
        assert mantle_transport(tm, 282., p) == mantle_transport(tm, 282., altered)
    with pytest.warns(DeprecationWarning, match="ignored"):
        parameters_from_config({"genesis": {"schema_version": 1,
            "water_volume_km3": p.water_volume_km3, "solid_transfer_w_m2_k": .0005}})


@pytest.mark.parametrize("changes", [
    {"mantle_depth_fraction_radius": 1.01}, {"viscosity_min_pa_s": 1e22},
    {"viscosity_max_pa_s": 1e20}, {"thermal_conductivity_w_m_k": 0.},
    {"nusselt_exponent": float("nan")}, {"activation_energy_j_mol": -1.},
])
def test_invalid_convection_parameters_are_rejected(changes):
    with pytest.raises(ValueError):
        replace(GenesisParameters(), **changes).validate()


@pytest.mark.parametrize("radiogenic,heats", [(5e-12, True), (0., False)])
def test_budget_can_warm_or_cool_a_cold_mantle_without_a_temperature_floor(radiogenic, heats):
    p = replace(GenesisParameters(), radiogenic_specific_power_w_kg=radiogenic)
    state = state_at(1000., 282., p)
    before = diagnose(state, p)
    assert (before["net_mantle_flux_w_m2"] > 0) == heats
    final, rows = advance(state, p, 1., max_step_myr=.1)
    assert (rows[-1]["mantle_temperature_k"] > 1000.) == heats
    assert abs(rows[-1]["relative_energy_residual"]) < 1e-11
    assert final.energy[0] != state.energy[0]


@pytest.fixture(scope="module")
def long_histories():
    p = GenesisParameters()
    return p, [run_genesis(p, duration_myr=1000., sample_interval_myr=50., max_step_myr=step)
               for step in (1., .5)]


def test_long_history_energy_and_timestep_convergence(long_histories):
    _, histories = long_histories
    coarse, fine = histories
    for _, rows in histories:
        assert max(abs(r["relative_energy_residual"]) for r in rows) < 1e-10
        for row in rows:
            assert row["net_mantle_flux_w_m2"] == pytest.approx(
                row["radiogenic_flux_w_m2"]+row["tidal_flux_w_m2"]-row["mantle_to_surface_flux_w_m2"])
    for key in ("mantle_temperature_k", "surface_temperature_k"):
        a = [r[key] for r in coarse[1] if r["time_myr"] % 50 == 0]
        b = [r[key] for r in fine[1] if r["time_myr"] % 50 == 0]
        np.testing.assert_allclose(a, b, rtol=0., atol=2e-4)


def test_long_checkpoint_resume_is_identical_on_sample_boundaries(tmp_path, long_histories):
    p, histories = long_histories
    middle, _ = run_genesis(p, duration_myr=400., sample_interval_myr=50., max_step_myr=1.)
    path = tmp_path/"heat.json"
    save_checkpoint(path, middle, p)
    restored, loaded, _ = load_checkpoint(path)
    final, _ = run_genesis(loaded, duration_myr=1000., sample_interval_myr=50., max_step_myr=1., state=restored)
    assert final == histories[0][0]


def test_hot_window_values_and_initial_energy_rate_are_unchanged():
    p = GenesisParameters()
    assert [outgoing_longwave_w_m2(t, p) for t in (2300., 2000., 1800., 1600.)] == pytest.approx(
        [13896.568980019, 1733.615851264, 372.725990704, 282.], rel=1e-14)
    q = fluxes(0., np.asarray(initial_state(p).energy), p)
    assert q["mantle_to_surface_flux_w_m2"] == 0.
    assert q["net_cooling_flux_w_m2"] > 13000.
