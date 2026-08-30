from dataclasses import replace

import numpy as np

from tectonics.hotspot_tracks import (
    HotspotTrackParameters,
    advance_hotspot_tracks,
    advect_hotspot_tracks,
    initialize_hotspot_tracks,
    magmatic_extension_forcing,
    underplate_density_field,
)
from tectonics.mesh import build_icosphere
from tectonics.plume_magmatism import (
    PlumeMagmatismParameters,
    igneous_ledger_error_km3,
    initialize_plume_magmatism,
    magmatic_topography_fields,
)
from tectonics.plumes import (
    MantlePlumeParameters,
    initialize_mantle_plumes,
    plume_component_flux_fields,
)


RADIUS_KM = 5287.0


def _plume_at_cell(mesh, cell=7):
    params = replace(
        MantlePlumeParameters(),
        seed=2901,
        initial_plume_count=1,
        mean_birth_interval_myr=10000.0,
        head_tail_separation_enabled=True,
    )
    state = initialize_mantle_plumes(mesh, 0.0, params)
    state.centers_unit[0] = mesh.centroids[cell]
    return params, state


def test_broad_short_lived_head_gives_way_to_narrow_persistent_tail():
    mesh = build_icosphere(2)
    params, state = _plume_at_cell(mesh)
    lifetime = float(state.lifetimes_myr[0])
    distance = RADIUS_KM * np.arccos(
        np.clip(mesh.centroids @ state.centers_unit[0], -1.0, 1.0)
    )
    flank = int(np.argmin(np.abs(distance - 650.0)))

    state.ages_myr[0] = 0.08 * lifetime
    early_head, early_tail = plume_component_flux_fields(
        mesh, state, RADIUS_KM, params
    )
    assert early_head[flank] > early_tail[flank]
    assert early_head[flank] > 0.05

    state.ages_myr[0] = 0.55 * lifetime
    late_head, late_tail = plume_component_flux_fields(
        mesh, state, RADIUS_KM, params
    )
    assert np.count_nonzero(late_head) == 0
    assert late_tail[7] > 0.25
    assert late_tail[flank] < early_head[flank]


def test_narrow_tail_area_integral_is_resolution_independent():
    integrals = []
    for subdivision in (2, 4):
        mesh = build_icosphere(subdivision)
        params, state = _plume_at_cell(mesh, cell=7)
        params = replace(params, component_flux_area_normalization_enabled=True)
        state.centers_unit[0] = np.array([1.0, 0.0, 0.0])
        state.ages_myr[0] = 0.55 * state.lifetimes_myr[0]
        _, tail = plume_component_flux_fields(mesh, state, RADIUS_KM, params)
        areas = mesh.physical_cell_areas_km2(RADIUS_KM)
        integrals.append(float(np.sum(areas * tail)))
    assert np.isclose(integrals[0], integrals[1], rtol=1.0e-12)


def test_rift_localization_moves_underplate_share_into_dykes_conservatively():
    mesh = build_icosphere(1)
    plume_params, plume = _plume_at_cell(mesh, cell=3)
    plume.last_head_flux = np.zeros(mesh.cell_count)
    plume.last_tail_flux = np.zeros(mesh.cell_count)
    plume.last_head_flux[[3, 4]] = 1.0
    magmatic = initialize_plume_magmatism(mesh)
    tracks = initialize_hotspot_tracks(mesh)
    rift = np.zeros(mesh.cell_count)
    rift[4] = 1.0
    extension = np.ones(mesh.cell_count)
    fallback = np.zeros(mesh.cell_count)
    params = HotspotTrackParameters(underplate_evolution_enabled=False)
    mag_params = PlumeMagmatismParameters()

    tracks, magmatic, diagnostics, _ = advance_hotspot_tracks(
        mesh, tracks, magmatic, plume, rift, extension, fallback,
        4.0, RADIUS_KM, mag_params, params
    )
    total3 = (
        magmatic.extrusive_volume_km3[3]
        + magmatic.dyke_volume_km3[3]
        + magmatic.underplate_volume_km3[3]
    )
    total4 = (
        magmatic.extrusive_volume_km3[4]
        + magmatic.dyke_volume_km3[4]
        + magmatic.underplate_volume_km3[4]
    )
    assert np.isclose(magmatic.dyke_volume_km3[3] / total3, 0.21)
    assert np.isclose(magmatic.dyke_volume_km3[4] / total4, 0.51)
    assert np.isclose(magmatic.underplate_volume_km3[4] / total4, 0.40)
    assert abs(igneous_ledger_error_km3(magmatic)) < 1.0e-8
    assert diagnostics.maximum_dike_localization == 1.0


def test_new_magma_heats_plate_and_heat_produces_bounded_extension_forcing():
    mesh = build_icosphere(1)
    _, plume = _plume_at_cell(mesh, cell=5)
    plume.last_head_flux = np.zeros(mesh.cell_count)
    plume.last_tail_flux = np.zeros(mesh.cell_count)
    plume.last_tail_flux[5] = 1.0
    magmatic = initialize_plume_magmatism(mesh)
    tracks = initialize_hotspot_tracks(mesh)
    zero = np.zeros(mesh.cell_count)
    tracks, magmatic, _, _ = advance_hotspot_tracks(
        mesh, tracks, magmatic, plume, zero, zero, zero,
        4.0, RADIUS_KM, PlumeMagmatismParameters(), HotspotTrackParameters()
    )
    forcing = magmatic_extension_forcing(tracks, HotspotTrackParameters())
    assert tracks.thermal_anomaly[5] > 0.0
    assert 0.0 < forcing[5] <= 0.32
    assert np.count_nonzero(forcing) == 1


def test_old_eclogitic_underplate_founds_into_closed_deep_ledger():
    mesh = build_icosphere(1)
    _, plume = _plume_at_cell(mesh, cell=6)
    plume.last_head_flux = np.zeros(mesh.cell_count)
    plume.last_tail_flux = np.zeros(mesh.cell_count)
    magmatic = initialize_plume_magmatism(mesh)
    cell = 6
    magmatic.underplate_volume_km3[cell] = 1000.0
    magmatic.cumulative_generated_underplate_volume_km3 = 1000.0
    tracks = initialize_hotspot_tracks(mesh)
    tracks.underplate_mean_age_myr[cell] = 500.0
    tracks.underplate_eclogite_fraction[cell] = 0.65
    params = HotspotTrackParameters(
        magmatic_thermal_weakening_enabled=False,
        dike_localization_enabled=False,
    )
    zero = np.zeros(mesh.cell_count)
    tracks, magmatic, diagnostics, _ = advance_hotspot_tracks(
        mesh, tracks, magmatic, plume, zero, zero, zero,
        20.0, RADIUS_KM, PlumeMagmatismParameters(), params
    )
    assert diagnostics.delaminated_underplate_this_step_km3 > 0.0
    assert magmatic.deep_recycled_underplate_volume_km3 > 0.0
    assert magmatic.underplate_volume_km3[cell] < 1000.0
    assert abs(igneous_ledger_error_km3(magmatic)) < 1.0e-9

    density = underplate_density_field(
        tracks, PlumeMagmatismParameters(), params
    )
    _, _, support, _ = magmatic_topography_fields(
        mesh,
        magmatic,
        RADIUS_KM,
        PlumeMagmatismParameters(),
        underplate_density_kg_m3=density,
    )
    assert density[cell] > PlumeMagmatismParameters().mantle_density_kg_m3
    assert support[cell] < 0.0


def test_hotspot_memory_advection_is_exact_and_deterministic():
    mesh = build_icosphere(1)
    state = initialize_hotspot_tracks(mesh)
    state.thermal_anomaly[:3] = [0.2, 0.4, 0.6]
    state.underplate_mean_age_myr[:3] = [20.0, 40.0, 60.0]
    source = np.arange(mesh.cell_count, dtype=np.int32)
    source[0] = 2
    source[1] = 2
    source[2] = -1
    out = advect_hotspot_tracks(state, source)
    assert out.thermal_anomaly[0] == 0.6
    assert out.thermal_anomaly[1] == 0.6
    assert out.thermal_anomaly[2] == 0.0
    assert out.underplate_mean_age_myr[0] == 60.0
