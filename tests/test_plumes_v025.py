from dataclasses import replace

import numpy as np

from tectonics.cratons import CratonParameters, initialize_craton_memory
from tectonics.lithosphere import initialize_lithosphere, refresh_mechanical_lithosphere
from tectonics.mesh import build_icosphere
from tectonics.plates import random_plate_system
from tectonics.plumes import (
    MantlePlumeParameters,
    advance_mantle_plumes,
    initialize_mantle_plumes,
    plume_flux_field,
)


RADIUS_KM = 5287.0


def _setup():
    mesh = build_icosphere(2)
    system = random_plate_system(mesh, 6, 2501, 0.2, 0.12, 0.35)
    lithosphere = initialize_lithosphere(
        mesh,
        system,
        continental_fraction=0.32,
        continental_nuclei=4,
        initial_continental_age_myr=900.0,
        radius_km=RADIUS_KM,
    )
    cratons = CratonParameters()
    initialize_craton_memory(mesh, lithosphere, RADIUS_KM, cratons)
    refresh_mechanical_lithosphere(lithosphere, 0.0)
    return mesh, lithosphere, cratons


def test_plume_population_initialization_is_deterministic():
    mesh, _, _ = _setup()
    params = MantlePlumeParameters(seed=811, initial_plume_count=4)
    left = initialize_mantle_plumes(mesh, 0.0, params)
    right = initialize_mantle_plumes(mesh, 0.0, params)
    assert np.array_equal(left.centers_unit, right.centers_unit)
    assert np.array_equal(left.lifetimes_myr, right.lifetimes_myr)
    assert np.array_equal(left.head_radii_km, right.head_radii_km)
    assert np.array_equal(left.peak_fluxes, right.peak_fluxes)
    assert left.next_birth_time_myr == right.next_birth_time_myr


def test_plume_flux_is_localized_around_its_mantle_fixed_center():
    mesh, _, _ = _setup()
    params = MantlePlumeParameters(seed=812, initial_plume_count=1)
    plumes = initialize_mantle_plumes(mesh, 0.0, params)
    center_cell = 17
    plumes.centers_unit[0] = mesh.centroids[center_cell]
    plumes.ages_myr[0] = 0.5 * plumes.lifetimes_myr[0]
    flux = plume_flux_field(mesh, plumes, RADIUS_KM, params)
    assert int(np.argmax(flux)) == center_cell
    assert flux[center_cell] > 0.70
    farthest = int(np.argmin(mesh.centroids @ mesh.centroids[center_cell]))
    assert flux[farthest] < 1.0e-4


def test_active_plume_refertilizes_weakens_and_erodes_a_cratonic_root():
    mesh, lithosphere, cratons = _setup()
    continent = lithosphere.continental_fraction > 0.5
    cell = int(np.flatnonzero(continent)[0])
    lithosphere.continental_lithosphere_age_myr[continent] = 1800.0
    lithosphere.mantle_depletion_fraction[continent] = cratons.maximum_depletion_fraction
    initialize_craton_memory(mesh, lithosphere, RADIUS_KM, cratons)
    refresh_mechanical_lithosphere(lithosphere, 0.0)
    old_age = lithosphere.continental_lithosphere_age_myr.copy()
    old_depletion = lithosphere.mantle_depletion_fraction.copy()
    old_strength = lithosphere.craton_strength.copy()
    old_root = lithosphere.mantle_lithosphere_thickness_km.copy()

    params = replace(
        MantlePlumeParameters(),
        seed=813,
        initial_plume_count=1,
        mean_birth_interval_myr=10000.0,
    )
    plumes = initialize_mantle_plumes(mesh, 0.0, params)
    plumes.centers_unit[0] = mesh.centroids[cell]
    plumes.ages_myr[0] = 0.5 * plumes.lifetimes_myr[0] - 4.0
    lithosphere, plumes, diag = advance_mantle_plumes(
        mesh, lithosphere, plumes, 4.0, RADIUS_KM, params, cratons
    )

    assert lithosphere.continental_lithosphere_age_myr[cell] < old_age[cell]
    assert lithosphere.mantle_depletion_fraction[cell] < old_depletion[cell]
    assert lithosphere.craton_strength[cell] < old_strength[cell]
    assert lithosphere.mantle_lithosphere_thickness_km[cell] < old_root[cell]
    assert diag.mean_continental_age_loss_myr > 0.0
    assert diag.mean_continental_depletion_loss > 0.0
    assert diag.max_root_erosion_this_step_km > 0.0


def test_disabled_plumes_leave_lithosphere_memory_unchanged():
    mesh, lithosphere, cratons = _setup()
    params = MantlePlumeParameters(enabled=False)
    plumes = initialize_mantle_plumes(mesh, lithosphere.time_myr, params)
    before = (
        lithosphere.continental_lithosphere_age_myr.copy(),
        lithosphere.mantle_depletion_fraction.copy(),
        lithosphere.craton_strength.copy(),
        lithosphere.mantle_lithosphere_thickness_km.copy(),
    )
    lithosphere, plumes, diag = advance_mantle_plumes(
        mesh, lithosphere, plumes, 4.0, RADIUS_KM, params, cratons
    )
    after = (
        lithosphere.continental_lithosphere_age_myr,
        lithosphere.mantle_depletion_fraction,
        lithosphere.craton_strength,
        lithosphere.mantle_lithosphere_thickness_km,
    )
    for old, new in zip(before, after):
        assert np.array_equal(old, new)
    assert diag.active_plume_count == 0
    assert diag.max_surface_flux == 0.0


def test_plume_continuation_is_exact_for_the_same_step_sequence():
    mesh, lithosphere_a, cratons = _setup()
    _, lithosphere_b, _ = _setup()
    params = MantlePlumeParameters(seed=814, initial_plume_count=2)
    plumes_a = initialize_mantle_plumes(mesh, 0.0, params)
    plumes_b = initialize_mantle_plumes(mesh, 0.0, params)
    for dt in (4.0, 4.0, 4.0):
        lithosphere_a, plumes_a, _ = advance_mantle_plumes(
            mesh, lithosphere_a, plumes_a, dt, RADIUS_KM, params, cratons
        )
    for dt in (4.0, 4.0):
        lithosphere_b, plumes_b, _ = advance_mantle_plumes(
            mesh, lithosphere_b, plumes_b, dt, RADIUS_KM, params, cratons
        )
    # A checkpoint/resume boundary makes no state transition.  Continue with
    # the same final step and require exact deterministic equality.
    lithosphere_b, plumes_b, _ = advance_mantle_plumes(
        mesh, lithosphere_b, plumes_b, 4.0, RADIUS_KM, params, cratons
    )
    assert np.array_equal(plumes_a.last_flux, plumes_b.last_flux)
    assert np.array_equal(plumes_a.cumulative_exposure_myr, plumes_b.cumulative_exposure_myr)
    assert np.array_equal(
        lithosphere_a.continental_lithosphere_age_myr,
        lithosphere_b.continental_lithosphere_age_myr,
    )
    assert np.array_equal(
        lithosphere_a.mantle_lithosphere_thickness_km,
        lithosphere_b.mantle_lithosphere_thickness_km,
    )
