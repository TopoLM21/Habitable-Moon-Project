import numpy as np

from tectonics.cratons import (
    CratonParameters,
    advance_craton_memory,
    craton_extension_factor,
    initialize_craton_memory,
)
from tectonics.lithosphere import (
    advance_lithosphere,
    initialize_lithosphere,
    target_mantle_lithosphere_fields,
)
from tectonics.mesh import build_icosphere
from tectonics.plates import random_plate_system
from tectonics.tides import constant_eccentricity


RADIUS_KM = 5287.0


def _setup():
    mesh = build_icosphere(2)
    system = random_plate_system(mesh, 6, 2401, 0.2, 0.12, 0.35)
    state = initialize_lithosphere(
        mesh,
        system,
        continental_fraction=0.28,
        continental_nuclei=4,
        initial_continental_age_myr=500.0,
        radius_km=RADIUS_KM,
    )
    params = CratonParameters()
    initialize_craton_memory(mesh, state, RADIUS_KM, params)
    return mesh, system, state, params


def test_craton_memory_initializes_only_on_continental_material():
    mesh, _, state, _ = _setup()
    fraction = state.continental_fraction
    ocean = fraction <= 1e-12
    continent = fraction > 0.5
    assert state.continental_lithosphere_age_myr is not None
    assert state.mantle_depletion_fraction is not None
    assert state.craton_strength is not None
    assert np.all(state.continental_lithosphere_age_myr[ocean] == 0.0)
    assert np.all(state.mantle_depletion_fraction[ocean] == 0.0)
    assert np.all(state.craton_strength[ocean] == 0.0)
    assert np.all(state.continental_lithosphere_age_myr[continent] == 500.0)
    assert np.all((state.craton_strength >= 0.0) & (state.craton_strength <= 1.0))
    assert np.mean(state.craton_strength[continent]) > 0.35


def test_quiet_continental_lithosphere_matures_and_strengthens():
    mesh, _, state, params = _setup()
    continent = state.continental_fraction > 0.5
    old_age = state.continental_lithosphere_age_myr.copy()
    old_depletion = state.mantle_depletion_fraction.copy()
    old_strength = state.craton_strength.copy()
    pre_volume = state.continental_volume_km3.copy()
    state, diag = advance_craton_memory(
        mesh,
        state,
        100.0,
        RADIUS_KM,
        params,
        pre_cycle_continental_volume_km3=pre_volume,
    )
    assert np.all(state.continental_lithosphere_age_myr[continent] > old_age[continent])
    assert np.all(state.mantle_depletion_fraction[continent] > old_depletion[continent])
    assert np.all(state.craton_strength[continent] > old_strength[continent])
    assert diag.mean_craton_strength > 0.0
    assert diag.new_continental_material_volume_km3 == 0.0


def test_juvenile_arc_volume_dilutes_old_root_memory():
    mesh, _, state, params = _setup()
    cell = int(np.flatnonzero(state.continental_fraction > 0.5)[0])
    pre_volume = state.continental_volume_km3.copy()
    old_age = float(state.continental_lithosphere_age_myr[cell])
    state.continental_volume_km3[cell] *= 2.0
    state, diag = advance_craton_memory(
        mesh,
        state,
        4.0,
        RADIUS_KM,
        params,
        pre_cycle_continental_volume_km3=pre_volume,
    )
    assert state.continental_lithosphere_age_myr[cell] < 0.60 * old_age
    assert diag.new_continental_material_volume_km3 > 0.0


def test_rifting_rejuvenates_and_weakens_a_mature_root():
    mesh, _, state, params = _setup()
    continent = state.continental_fraction > 0.5
    state.continental_lithosphere_age_myr[continent] = 1400.0
    state.mantle_depletion_fraction[continent] = params.maximum_depletion_fraction
    initialize_craton_memory(mesh, state, RADIUS_KM, params)
    old_age = state.continental_lithosphere_age_myr.copy()
    old_strength = state.craton_strength.copy()
    state.rift_extension[continent] = 1.0
    pre_volume = state.continental_volume_km3.copy()
    state, _ = advance_craton_memory(
        mesh,
        state,
        100.0,
        RADIUS_KM,
        params,
        pre_cycle_continental_volume_km3=pre_volume,
    )
    assert np.all(state.continental_lithosphere_age_myr[continent] < old_age[continent])
    assert np.all(state.craton_strength[continent] < old_strength[continent])


def test_cratons_thicken_buoyant_roots_and_resist_extension():
    _, _, state, params = _setup()
    continent = state.continental_fraction > 0.5
    state.craton_strength[:] = 0.0
    weak_h, weak_drho = target_mantle_lithosphere_fields(state)
    state.craton_strength[continent] = 1.0
    strong_h, strong_drho = target_mantle_lithosphere_fields(state)
    assert np.all(strong_h[continent] > weak_h[continent])
    assert np.all(strong_drho[continent] < weak_drho[continent])
    factor = craton_extension_factor(
        state,
        gain=params.extension_resistance_gain,
        minimum_factor=params.minimum_extension_factor,
    )
    assert np.all(factor[continent] < 1.0)
    assert np.all(factor >= params.minimum_extension_factor)


def test_craton_memory_uses_the_lithosphere_transport_source_map():
    mesh, system, state, _ = _setup()
    marker = np.arange(mesh.cell_count, dtype=np.float64)
    state.continental_lithosphere_age_myr = marker.copy()
    state.mantle_depletion_fraction = marker / max(float(mesh.cell_count - 1), 1.0)
    state.craton_strength = marker[::-1].copy() / max(float(mesh.cell_count - 1), 1.0)
    nxt, _, _, diag = advance_lithosphere(
        mesh,
        system,
        state,
        2.0,
        RADIUS_KM,
        7.12,
        47.0,
        constant_eccentricity(0.0),
    )
    source = diag.material_source_index
    moved = source >= 0
    assert np.array_equal(
        nxt.continental_lithosphere_age_myr[moved],
        state.continental_lithosphere_age_myr[source[moved]],
    )
    assert np.array_equal(
        nxt.mantle_depletion_fraction[moved],
        state.mantle_depletion_fraction[source[moved]],
    )
    assert np.array_equal(
        nxt.craton_strength[moved],
        state.craton_strength[source[moved]],
    )
