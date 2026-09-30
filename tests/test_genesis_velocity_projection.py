"""Young velocity projection is explicit, persistent and physically uncalibrated."""
from copy import deepcopy
from dataclasses import asdict, replace

import numpy as np
import pytest

from tectonics.dynamics import DynamicsParameters, angular_velocity_vectors, update_plate_dynamics
from tectonics.mantle import MantleFlowState, plate_mean_mantle_omega, plate_rigid_mantle_fit
from tectonics.plate_velocity_diagnostics import velocity_budget, weighted_stats
from tectonics.genesis_starter_continuation import build_starter_continuation
from test_dynamics_trace import _world
from test_genesis_starter_continuation_independent import source


def test_genesis_selects_fit_without_changing_physical_controls(source):
    model, state, config = source
    original = deepcopy(config)
    _, cfg, _ = build_starter_continuation(model, state, config)
    assert config == original
    assert cfg["plate_dynamics"]["mantle_projection"] == "velocity_least_squares"
    for name, value in original["plate_dynamics"].items():
        if name != "ridge_gpe_min_factor":  # Existing young no-ridge-floor rule.
            assert cfg["plate_dynamics"][name] == value
    config = deepcopy(config)
    config["plate_dynamics"]["mantle_projection"] = "legacy_area_mean"
    _, saved_legacy, _ = build_starter_continuation(model, state, config)
    assert saved_legacy["plate_dynamics"]["mantle_projection"] == "legacy_area_mean"


def test_real_dynamics_fit_recovers_rigid_velocity_and_default_retains_legacy():
    mesh, state, system = _world()
    vector = np.array([1.e-4, -2.e-4, 3.e-4])
    positions = mesh.centroids
    # The Genesis representation of a rigid velocity lacks radial components.
    field = np.cross(positions, np.cross(vector, positions))
    flow = MantleFlowState(0., field, float(np.linalg.norm(vector)))
    options = DynamicsParameters(force_speed_scale_deg_per_myr=0.)
    args = mesh, state, system, system, 5287., 1., 4., 1.
    trace = {}
    update_plate_dynamics(*args, options, mantle_flow=flow, trace=trace)
    average = plate_mean_mantle_omega(mesh, state.cell_plate, len(system.plates), 5287., flow)
    np.testing.assert_array_equal(trace["mantle_omega"], average)
    fixed_trace = {}
    update_plate_dynamics(*args, replace(options, mantle_projection="velocity_least_squares"),
                          mantle_flow=flow, trace=fixed_trace)
    np.testing.assert_allclose(fixed_trace["mantle_omega"], np.tile(vector, (4, 1)), atol=2.e-18)
    np.testing.assert_allclose(fixed_trace["target_omega"], .22*np.tile(vector, (4, 1)), atol=2.e-18)
    assert not np.allclose(average, vector, atol=1.e-8)
    with pytest.raises(ValueError, match="projection"):
        update_plate_dynamics(*args, replace(options, mantle_projection="typo"), mantle_flow=flow)


@pytest.mark.parametrize("mode", ["legacy_area_mean", "velocity_least_squares"])
def test_diagnostic_budget_matches_returned_velocity_without_mutating_inputs(mode):
    mesh, state, system = _world()
    positions = mesh.centroids
    field = np.cross(positions, np.array([.002, -.001, .003]))
    flow = MantleFlowState(0., field, .004)
    original = deepcopy((state, system, flow))
    trace = {}
    result = update_plate_dynamics(mesh, state, system, system, 5287., 1., 4., 1.,
        DynamicsParameters(mantle_projection=mode), mantle_flow=flow, trace=trace)
    budget = velocity_budget(mesh, state, system, flow, 5287., trace, result[2])
    returned_speed = np.linalg.norm(np.cross(angular_velocity_vectors(result[0])[state.cell_plate], positions), axis=1)*5287.
    areas = mesh.physical_cell_areas_km2(5287.)
    assert budget["stages_speed_km_per_myr"]["actual_returned"]["mean"] == pytest.approx(areas@returned_speed/areas.sum())
    assert budget["verification"]["target_vector_sum_max_error"] < 1.e-16
    assert budget["verification"]["net_rotation_relative_velocity_max_change_km_per_myr"] < 1.e-12
    fit = plate_rigid_mantle_fit(mesh, state.cell_plate, 4, 5287., flow)
    for pid, row in enumerate(budget["plates"]):
        assert row["best_fit_relative_residual"] == pytest.approx(fit.relative_residual[pid])
        assert row["represented_kinetic_proxy_fraction"] == pytest.approx(fit.represented_kinetic_fraction[pid])
    np.testing.assert_array_equal(state.cell_plate, original[0].cell_plate)
    np.testing.assert_array_equal(angular_velocity_vectors(system), angular_velocity_vectors(original[1]))
    np.testing.assert_array_equal(flow.cell_omega_rad_per_myr, original[2].cell_omega_rad_per_myr)


def test_diagnostic_percentiles_follow_area_not_cell_count():
    stats = weighted_stats(np.array([1., 100., 200.]), np.array([98., 1., 1.]))
    assert stats["median"] == stats["p90"] == 1.
    assert stats["mean"] == pytest.approx(3.98)
