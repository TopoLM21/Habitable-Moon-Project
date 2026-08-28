from copy import deepcopy
from dataclasses import replace

import numpy as np

from tectonics.lithosphere import boundary_records_for_state, initialize_lithosphere
from tectonics.mesh import build_icosphere
from tectonics.plates import random_plate_system
from tectonics.plume_dynamic_topography import (
    PlumeDynamicTopographyParameters,
    advance_plume_dynamic_topography,
    initialize_plume_dynamic_topography,
    plume_dynamic_topography_target,
)
from tectonics.plumes import (
    MantlePlumeParameters,
    initialize_mantle_plumes,
    plume_flux_field,
)
from tectonics.topography import TopographyParameters, equilibrium_elevation


RADIUS_KM = 5287.0


def _setup_active_plume():
    mesh = build_icosphere(2)
    system = random_plate_system(mesh, 6, 2701, 0.2, 0.12, 0.35)
    lithosphere = initialize_lithosphere(
        mesh,
        system,
        continental_fraction=0.34,
        continental_nuclei=4,
        radius_km=RADIUS_KM,
    )
    plume_params = replace(
        MantlePlumeParameters(),
        seed=2702,
        initial_plume_count=1,
        mean_birth_interval_myr=10000.0,
    )
    plume = initialize_mantle_plumes(mesh, 0.0, plume_params)
    center_cell = 23
    plume.centers_unit[0] = mesh.centroids[center_cell]
    plume.ages_myr[0] = 0.5 * plume.lifetimes_myr[0]
    plume.last_flux = plume_flux_field(mesh, plume, RADIUS_KM, plume_params)
    return mesh, system, lithosphere, plume, center_cell


def _area_mean(mesh, values):
    areas = mesh.physical_cell_areas_km2(RADIUS_KM)
    return float(np.sum(areas * np.asarray(values)) / np.sum(areas))


def test_dynamic_topography_target_is_localized_and_zero_mean():
    mesh, _, _, plume, center_cell = _setup_active_plume()
    target = plume_dynamic_topography_target(
        mesh,
        plume,
        RADIUS_KM,
        PlumeDynamicTopographyParameters(),
    )
    assert int(np.argmax(target)) == center_cell
    assert target[center_cell] > 800.0
    assert float(np.min(target)) < 0.0
    assert abs(_area_mean(mesh, target)) < 1.0e-10


def test_dynamic_topography_response_is_delayed_and_reversibly_decays():
    mesh, _, _, plume, _ = _setup_active_plume()
    params = PlumeDynamicTopographyParameters(
        response_time_myr=8.0,
        decay_time_myr=20.0,
    )
    state = initialize_plume_dynamic_topography(mesh, 0.0)
    state, first = advance_plume_dynamic_topography(
        mesh, plume, state, 4.0, RADIUS_KM, params
    )
    assert 0.0 < first.maximum_realized_uplift_m < first.maximum_target_uplift_m
    first_amplitude = float(np.max(np.abs(state.realized_dynamic_topography_m)))
    plume.last_flux[:] = 0.0
    for _ in range(25):
        state, final = advance_plume_dynamic_topography(
            mesh, plume, state, 4.0, RADIUS_KM, params
        )
    assert float(np.max(np.abs(state.realized_dynamic_topography_m))) < 0.01 * first_amplitude
    assert abs(final.area_mean_realized_m) < 1.0e-10
    assert abs(final.displacement_volume_km3) < 1.0e-6


def test_disabled_dynamic_topography_remains_exactly_zero():
    mesh, _, _, plume, _ = _setup_active_plume()
    params = PlumeDynamicTopographyParameters(enabled=False)
    state = initialize_plume_dynamic_topography(mesh, 0.0)
    state, diagnostics = advance_plume_dynamic_topography(
        mesh, plume, state, 4.0, RADIUS_KM, params
    )
    assert np.count_nonzero(state.target_dynamic_topography_m) == 0
    assert np.count_nonzero(state.realized_dynamic_topography_m) == 0
    assert np.count_nonzero(state.cumulative_positive_support_m_myr) == 0
    assert not diagnostics.enabled


def test_dynamic_topography_enters_background_without_changing_lithosphere():
    mesh, system, lithosphere, _, _ = _setup_active_plume()
    before = deepcopy(lithosphere)
    boundaries = boundary_records_for_state(
        mesh, lithosphere, system, RADIUS_KM, 4.0, 1.0
    )
    params = TopographyParameters()
    dynamic = np.linspace(-80.0, 80.0, mesh.cell_count)
    baseline, _ = equilibrium_elevation(
        mesh, lithosphere, boundaries, params, RADIUS_KM
    )
    supported, _ = equilibrium_elevation(
        mesh,
        lithosphere,
        boundaries,
        params,
        RADIUS_KM,
        dynamic_topography_m=dynamic,
    )
    assert np.allclose(supported - baseline, dynamic, atol=1.0e-10)
    for name in (
        "crust_type",
        "crust_age_myr",
        "crust_thickness_km",
        "continental_fraction",
        "continental_volume_km3",
    ):
        assert np.array_equal(getattr(lithosphere, name), getattr(before, name))


def test_dynamic_topography_continuation_is_exact_for_same_steps():
    mesh, _, _, plume, _ = _setup_active_plume()
    params = PlumeDynamicTopographyParameters()
    left = initialize_plume_dynamic_topography(mesh, 0.0)
    right = initialize_plume_dynamic_topography(mesh, 0.0)
    for _ in range(3):
        left, _ = advance_plume_dynamic_topography(
            mesh, plume, left, 4.0, RADIUS_KM, params
        )
    for _ in range(2):
        right, _ = advance_plume_dynamic_topography(
            mesh, plume, right, 4.0, RADIUS_KM, params
        )
    right, _ = advance_plume_dynamic_topography(
        mesh, plume, right, 4.0, RADIUS_KM, params
    )
    assert np.array_equal(
        left.target_dynamic_topography_m,
        right.target_dynamic_topography_m,
    )
    assert np.array_equal(
        left.realized_dynamic_topography_m,
        right.realized_dynamic_topography_m,
    )
    assert np.array_equal(
        left.cumulative_positive_support_m_myr,
        right.cumulative_positive_support_m_myr,
    )
