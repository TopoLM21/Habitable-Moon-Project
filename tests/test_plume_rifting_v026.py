from copy import deepcopy
from dataclasses import replace

import numpy as np

from tectonics.cratons import CratonParameters, initialize_craton_memory
from tectonics.lithosphere import initialize_lithosphere, refresh_mechanical_lithosphere
from tectonics.mesh import build_icosphere
from tectonics.plates import random_plate_system
from tectonics.plume_rifting import (
    PlumeRiftingParameters,
    advance_plume_rifting,
    initialize_plume_rifting,
    plume_rifting_fields,
)
from tectonics.plumes import (
    MantlePlumeParameters,
    advance_mantle_plumes,
    initialize_mantle_plumes,
    plume_flux_field,
)


RADIUS_KM = 5287.0


def _setup():
    mesh = build_icosphere(2)
    system = random_plate_system(mesh, 6, 2601, 0.2, 0.12, 0.35)
    lithosphere = initialize_lithosphere(
        mesh,
        system,
        continental_fraction=0.34,
        continental_nuclei=4,
        initial_continental_age_myr=1200.0,
        radius_km=RADIUS_KM,
    )
    cratons = CratonParameters()
    initialize_craton_memory(mesh, lithosphere, RADIUS_KM, cratons)
    refresh_mechanical_lithosphere(lithosphere, 0.0)
    return mesh, lithosphere, cratons


def _active_single_plume(mesh, lithosphere, *, weakening_enabled=True):
    params = replace(
        MantlePlumeParameters(),
        seed=2602,
        initial_plume_count=1,
        mean_birth_interval_myr=10000.0,
        lithosphere_weakening_enabled=weakening_enabled,
    )
    plume = initialize_mantle_plumes(mesh, lithosphere.time_myr, params)
    continent = lithosphere.continental_fraction > 0.5
    center_cell = int(np.flatnonzero(continent)[0])
    plume.centers_unit[0] = mesh.centroids[center_cell]
    plume.ages_myr[0] = 0.5 * plume.lifetimes_myr[0]
    plume.last_flux = plume_flux_field(mesh, plume, RADIUS_KM, params)
    return plume, params, center_cell


def test_plume_rifting_field_is_deterministic_and_localized():
    mesh, lithosphere, _ = _setup()
    plume, _, center_cell = _active_single_plume(mesh, lithosphere)
    params = PlumeRiftingParameters()
    left = plume_rifting_fields(mesh, lithosphere, plume, RADIUS_KM, params)
    right = plume_rifting_fields(mesh, lithosphere, plume, RADIUS_KM, params)
    for a, b in zip(left, right):
        assert np.array_equal(a, b)
    forcing, uplift, magmatic, contrast = left
    farthest = int(np.argmin(mesh.centroids @ mesh.centroids[center_cell]))
    assert forcing[center_cell] > 0.7
    assert forcing[farthest] < 1.0e-6
    assert uplift[center_cell] > uplift[farthest]
    assert magmatic[center_cell] > magmatic[farthest]
    assert np.all((contrast >= 0.0) & (contrast <= 1.0))


def test_disabled_plume_rifting_returns_zero_without_losing_time():
    mesh, lithosphere, _ = _setup()
    plume, _, _ = _active_single_plume(mesh, lithosphere)
    state = initialize_plume_rifting(mesh, lithosphere.time_myr)
    state, forcing, diag = advance_plume_rifting(
        mesh,
        lithosphere,
        plume,
        state,
        4.0,
        RADIUS_KM,
        PlumeRiftingParameters(enabled=False),
    )
    assert state.time_myr == lithosphere.time_myr + 4.0
    assert np.count_nonzero(forcing) == 0
    assert np.count_nonzero(state.cumulative_extension_impulse_myr) == 0
    assert not diag.enabled


def test_weakening_and_mechanical_forcing_are_independently_switchable():
    mesh, original, cratons = _setup()
    cases = {
        "control": (False, False),
        "weakening_only": (True, False),
        "forcing_only": (False, True),
        "combined": (True, True),
    }
    observed = {}
    for name, (weakening, rifting) in cases.items():
        lithosphere = deepcopy(original)
        plume, plume_params, center_cell = _active_single_plume(
            mesh, lithosphere, weakening_enabled=weakening
        )
        before_age = lithosphere.continental_lithosphere_age_myr.copy()
        before_root = lithosphere.mantle_lithosphere_thickness_km.copy()
        lithosphere, plume, _ = advance_mantle_plumes(
            mesh,
            lithosphere,
            plume,
            4.0,
            RADIUS_KM,
            plume_params,
            cratons,
        )
        rift_state = initialize_plume_rifting(mesh, original.time_myr)
        rift_state, forcing, _ = advance_plume_rifting(
            mesh,
            lithosphere,
            plume,
            rift_state,
            4.0,
            RADIUS_KM,
            PlumeRiftingParameters(enabled=rifting),
        )
        observed[name] = {
            "memory_changed": not np.array_equal(
                before_age, lithosphere.continental_lithosphere_age_myr
            )
            or not np.array_equal(
                before_root, lithosphere.mantle_lithosphere_thickness_km
            ),
            "forcing": float(forcing[center_cell]),
        }

    assert observed["control"] == {"memory_changed": False, "forcing": 0.0}
    assert observed["weakening_only"]["memory_changed"]
    assert observed["weakening_only"]["forcing"] == 0.0
    assert not observed["forcing_only"]["memory_changed"]
    assert observed["forcing_only"]["forcing"] > 0.0
    assert observed["combined"]["memory_changed"]
    assert observed["combined"]["forcing"] > 0.0


def test_plume_rifting_continuation_is_exact_for_same_step_sequence():
    mesh, lithosphere, _ = _setup()
    plume, _, _ = _active_single_plume(mesh, lithosphere)
    params = PlumeRiftingParameters()
    left = initialize_plume_rifting(mesh, 0.0)
    right = initialize_plume_rifting(mesh, 0.0)
    for dt in (4.0, 4.0, 4.0):
        left, _, _ = advance_plume_rifting(
            mesh, lithosphere, plume, left, dt, RADIUS_KM, params
        )
    for dt in (4.0, 4.0):
        right, _, _ = advance_plume_rifting(
            mesh, lithosphere, plume, right, dt, RADIUS_KM, params
        )
    right, _, _ = advance_plume_rifting(
        mesh, lithosphere, plume, right, 4.0, RADIUS_KM, params
    )
    assert np.array_equal(left.last_extension_forcing, right.last_extension_forcing)
    assert np.array_equal(
        left.cumulative_extension_impulse_myr,
        right.cumulative_extension_impulse_myr,
    )
    assert np.array_equal(left.last_dynamic_uplift_m, right.last_dynamic_uplift_m)
    assert np.array_equal(
        left.last_magmatic_productivity, right.last_magmatic_productivity
    )
