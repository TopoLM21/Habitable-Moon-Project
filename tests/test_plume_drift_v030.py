from dataclasses import replace

import numpy as np

from tectonics.cratons import CratonParameters, initialize_craton_memory
from tectonics.lithosphere import initialize_lithosphere, refresh_mechanical_lithosphere
from tectonics.mesh import build_icosphere
from tectonics.plates import Plate, PlateSystem, random_plate_system
from tectonics.plume_drift import (
    PlumeDriftParameters,
    diagnose_plume_drift,
    source_path_rows,
    source_plate_kinematics,
)
from tectonics.plumes import (
    MantlePlumeParameters,
    advance_mantle_plumes,
    initialize_mantle_plumes,
    plume_component_flux_fields,
)


RADIUS_KM = 5287.0


def _setup():
    mesh = build_icosphere(1)
    system = random_plate_system(mesh, 5, 3001, 0.2, 0.12, 0.35)
    lithosphere = initialize_lithosphere(
        mesh,
        system,
        continental_fraction=0.3,
        continental_nuclei=3,
        initial_continental_age_myr=900.0,
        radius_km=RADIUS_KM,
    )
    cratons = CratonParameters()
    initialize_craton_memory(mesh, lithosphere, RADIUS_KM, cratons)
    refresh_mechanical_lithosphere(lithosphere, 0.0)
    return mesh, lithosphere, cratons


def test_source_drift_is_opt_in_and_legacy_center_remains_exactly_fixed():
    mesh, lithosphere, cratons = _setup()
    params = MantlePlumeParameters(
        seed=3002, initial_plume_count=1, mean_birth_interval_myr=10000.0
    )
    state = initialize_mantle_plumes(mesh, 0.0, params)
    before = state.centers_unit.copy()
    _, state, diagnostics = advance_mantle_plumes(
        mesh, lithosphere, state, 12.0, RADIUS_KM, params, cratons
    )
    assert np.array_equal(state.centers_unit, before)
    assert diagnostics.source_drift_enabled is False
    assert diagnostics.population_source_path_length_km == 0.0


def test_mobile_source_integrates_linear_path_and_reorients_deterministically():
    mesh, lithosphere_a, cratons = _setup()
    _, lithosphere_b, _ = _setup()
    params = replace(
        MantlePlumeParameters(),
        seed=3003,
        initial_plume_count=1,
        mean_birth_interval_myr=10000.0,
        minimum_lifetime_myr=200.0,
        maximum_lifetime_myr=200.0,
        source_drift_enabled=True,
        minimum_source_drift_km_per_myr=12.0,
        maximum_source_drift_km_per_myr=12.0,
        source_drift_persistence_myr=20.0,
    )
    left = initialize_mantle_plumes(mesh, 0.0, params)
    right = initialize_mantle_plumes(mesh, 0.0, params)
    start = left.centers_unit[0].copy()
    for dt in (8.0, 8.0, 8.0):
        lithosphere_a, left, _ = advance_mantle_plumes(
            mesh, lithosphere_a, left, dt, RADIUS_KM, params, cratons
        )
        lithosphere_b, right, _ = advance_mantle_plumes(
            mesh, lithosphere_b, right, dt, RADIUS_KM, params, cratons
        )
    assert np.array_equal(left.centers_unit, right.centers_unit)
    assert np.array_equal(left.source_drift_axes_unit, right.source_drift_axes_unit)
    assert np.array_equal(left.cumulative_source_distance_km, [288.0])
    assert left.population_source_distance_km == 288.0
    assert left.source_drift_segment_index[0] == 1
    assert left.cumulative_source_bend_deg[0] > 0.0
    angular_distance = np.arccos(np.clip(np.dot(start, left.centers_unit[0]), -1.0, 1.0))
    assert angular_distance > 0.0


def test_source_plate_relative_kinematics_separates_the_two_velocities():
    mesh = build_icosphere(1)
    params = MantlePlumeParameters(
        seed=3004,
        initial_plume_count=1,
        source_drift_enabled=True,
        minimum_source_drift_km_per_myr=10.0,
        maximum_source_drift_km_per_myr=10.0,
    )
    state = initialize_mantle_plumes(mesh, 0.0, params)
    state.centers_unit[0] = np.array([1.0, 0.0, 0.0])
    state.source_drift_axes_unit[0] = np.array([0.0, 0.0, 1.0])
    state.source_drift_speeds_km_per_myr[0] = 10.0
    plate = Plate(
        plate_id=7,
        seed_cell=0,
        euler_axis=np.array([0.0, 0.0, 1.0]),
        angular_speed_rad_per_myr=20.0 / RADIUS_KM,
    )
    system = PlateSystem(
        cell_plate=np.full(mesh.cell_count, 7, dtype=np.int32),
        plates=(plate,),
    )
    fields = source_plate_kinematics(mesh, state, system, RADIUS_KM)
    assert np.isclose(fields["source_speed_km_per_myr"][0], 10.0)
    assert np.isclose(fields["plate_speed_km_per_myr"][0], 20.0)
    assert np.isclose(fields["relative_speed_km_per_myr"][0], 10.0)
    assert np.isclose(fields["source_to_plate_speed_ratio"][0], 0.5)
    assert np.isclose(fields["source_motion_deflection_deg"][0], 0.0)

    drift_params = PlumeDriftParameters(
        minimum_speed_km_per_myr=10.0,
        maximum_speed_km_per_myr=10.0,
    )
    diagnostics = diagnose_plume_drift(
        mesh, state, system, RADIUS_KM, drift_params
    )
    assert diagnostics.mean_relative_track_speed_km_per_myr == 10.0
    rows = source_path_rows(mesh, state, system, RADIUS_KM)
    assert rows[0]["plume_id"] == 0
    assert rows[0]["overlying_plate_id"] == 7
    assert rows[0]["relative_track_speed_km_per_myr"] == 10.0


def test_mobile_tail_productivity_integral_is_resolution_independent():
    integrals = []
    exponent = 1.4
    for subdivision in (2, 4):
        mesh = build_icosphere(subdivision)
        params = replace(
            MantlePlumeParameters(),
            seed=3005,
            initial_plume_count=1,
            head_tail_separation_enabled=True,
            component_flux_area_normalization_enabled=True,
            component_flux_area_normalization_exponent=exponent,
        )
        state = initialize_mantle_plumes(mesh, 0.0, params)
        state.centers_unit[0] = np.array([0.31, -0.57, 0.76013157])
        state.centers_unit[0] /= np.linalg.norm(state.centers_unit[0])
        state.ages_myr[0] = 0.55 * state.lifetimes_myr[0]
        _, tail = plume_component_flux_fields(mesh, state, RADIUS_KM, params)
        areas = mesh.physical_cell_areas_km2(RADIUS_KM)
        integrals.append(float(np.sum(areas * np.power(tail, exponent))))
    assert np.isclose(integrals[0], integrals[1], rtol=1.0e-12)
