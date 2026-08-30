from dataclasses import replace

import numpy as np

from tectonics.cratons import CratonParameters, initialize_craton_memory
from tectonics.lithosphere import initialize_lithosphere, refresh_mechanical_lithosphere
from tectonics.mesh import build_icosphere
from tectonics.plates import random_plate_system
from tectonics.plume_flow_coupling import (
    PlumeFlowCouplingParameters,
    diagnose_plume_flow_coupling,
    plume_parameters_with_flow_coupling,
)
from tectonics.plumes import (
    MantlePlumeParameters,
    advance_mantle_plumes,
    initialize_mantle_plumes,
    update_plume_source_flow,
)


RADIUS_KM = 5287.0


def _setup():
    mesh = build_icosphere(1)
    system = random_plate_system(mesh, 5, 3101, 0.2, 0.12, 0.35)
    lithosphere = initialize_lithosphere(
        mesh, system, 0.3, 3, radius_km=RADIUS_KM
    )
    cratons = CratonParameters()
    initialize_craton_memory(mesh, lithosphere, RADIUS_KM, cratons)
    refresh_mechanical_lithosphere(lithosphere, 0.0)
    return mesh, lithosphere, cratons


def test_uniform_mantle_flow_is_sampled_exactly_at_every_source():
    mesh, _, _ = _setup()
    params = replace(
        MantlePlumeParameters(),
        seed=3102,
        initial_plume_count=3,
        source_drift_enabled=True,
        source_flow_coupling_enabled=True,
    )
    state = initialize_mantle_plumes(mesh, 0.0, params)
    omega = np.array([0.001, -0.002, 0.003])
    field = np.broadcast_to(omega, (mesh.cell_count, 3)).copy()
    update_plume_source_flow(mesh, state, field, RADIUS_KM, params)
    assert np.allclose(state.source_flow_omega_rad_per_myr, omega, rtol=0.0, atol=1e-18)


def test_flow_and_residual_vectors_combine_before_source_motion():
    mesh, lithosphere, cratons = _setup()
    flow_params = PlumeFlowCouplingParameters(
        mantle_flow_velocity_fraction=0.5,
        residual_drift_fraction=0.0,
    )
    params = plume_parameters_with_flow_coupling(
        replace(
            MantlePlumeParameters(),
            seed=3103,
            initial_plume_count=1,
            mean_birth_interval_myr=10000.0,
            minimum_lifetime_myr=200.0,
            maximum_lifetime_myr=200.0,
            source_drift_enabled=True,
        ),
        flow_params,
    )
    state = initialize_mantle_plumes(mesh, 0.0, params)
    state.centers_unit[0] = np.array([1.0, 0.0, 0.0])
    omega = np.array([0.0, 0.0, 20.0 / RADIUS_KM])
    field = np.broadcast_to(omega, (mesh.cell_count, 3)).copy()
    update_plume_source_flow(
        mesh, state, field, RADIUS_KM, params,
        initialize_effective_velocity=True,
    )
    _, state, _ = advance_mantle_plumes(
        mesh, lithosphere, state, 4.0, RADIUS_KM, params, cratons,
        source_flow_omega_field_rad_per_myr=field,
    )
    assert np.isclose(state.last_effective_source_speeds_km_per_myr[0], 10.0)
    assert np.isclose(state.cumulative_source_distance_km[0], 40.0)
    expected = np.array([np.cos(40.0 / RADIUS_KM), np.sin(40.0 / RADIUS_KM), 0.0])
    assert np.allclose(state.centers_unit[0], expected, atol=1e-12)
    diagnostics = diagnose_plume_flow_coupling(state, RADIUS_KM, flow_params)
    assert np.isclose(diagnostics.mean_resolved_flow_speed_km_per_myr, 10.0)
    assert diagnostics.mean_residual_speed_km_per_myr == 0.0
    assert np.isclose(diagnostics.mean_effective_flow_alignment, 1.0)


def test_disabled_flow_coupling_preserves_v030_trajectory_exactly():
    mesh, lithosphere_a, cratons = _setup()
    _, lithosphere_b, _ = _setup()
    base = replace(
        MantlePlumeParameters(),
        seed=3104,
        initial_plume_count=2,
        mean_birth_interval_myr=10000.0,
        source_drift_enabled=True,
    )
    disabled = replace(base, source_flow_coupling_enabled=False)
    left = initialize_mantle_plumes(mesh, 0.0, base)
    right = initialize_mantle_plumes(mesh, 0.0, disabled)
    arbitrary = np.full((mesh.cell_count, 3), 0.02)
    lithosphere_a, left, _ = advance_mantle_plumes(
        mesh, lithosphere_a, left, 4.0, RADIUS_KM, base, cratons
    )
    lithosphere_b, right, _ = advance_mantle_plumes(
        mesh, lithosphere_b, right, 4.0, RADIUS_KM, disabled, cratons,
        source_flow_omega_field_rad_per_myr=arbitrary,
    )
    assert np.array_equal(left.centers_unit, right.centers_unit)
    assert np.array_equal(
        left.cumulative_source_distance_km,
        right.cumulative_source_distance_km,
    )
