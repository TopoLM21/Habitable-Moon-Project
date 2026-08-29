from copy import deepcopy
from dataclasses import replace

import numpy as np

from tectonics.lithosphere import boundary_records_for_state, initialize_lithosphere
from tectonics.mesh import build_icosphere
from tectonics.plates import PlateSystem, random_plate_system
from tectonics.plume_magmatism import (
    PlumeMagmatismParameters,
    advect_plume_magmatism,
    advance_plume_magmatism,
    igneous_ledger_error_km3,
    initialize_plume_magmatism,
    magmatic_topography_fields,
)
from tectonics.topography import TopographyParameters, topography_components


RADIUS_KM = 5287.0


def test_emplacement_partitions_volume_and_closes_ledger():
    mesh = build_icosphere(1)
    state = initialize_plume_magmatism(mesh)
    productivity = np.zeros(mesh.cell_count)
    extension = np.zeros(mesh.cell_count)
    productivity[3] = 1.0
    extension[3] = 1.0
    params = PlumeMagmatismParameters()
    state, diagnostics = advance_plume_magmatism(
        mesh, state, productivity, extension, 4.0, RADIUS_KM, params
    )
    generated = diagnostics.cumulative_generated_total_volume_km3
    assert generated > 0.0
    assert np.isclose(np.sum(state.extrusive_volume_km3), generated * 0.09)
    assert np.isclose(np.sum(state.dyke_volume_km3), generated * 0.21)
    assert np.isclose(np.sum(state.underplate_volume_km3), generated * 0.70)
    assert state.time_myr == 4.0
    assert diagnostics.global_igneous_ledger_error_km3 == 0.0


def test_disabled_magmatism_leaves_no_permanent_material():
    mesh = build_icosphere(1)
    state = initialize_plume_magmatism(mesh)
    state, diagnostics = advance_plume_magmatism(
        mesh,
        state,
        np.ones(mesh.cell_count),
        np.ones(mesh.cell_count),
        4.0,
        RADIUS_KM,
        replace(PlumeMagmatismParameters(), enabled=False),
    )
    assert state.time_myr == 4.0
    assert np.count_nonzero(state.extrusive_volume_km3) == 0
    assert diagnostics.cumulative_generated_total_volume_km3 == 0.0


def test_transport_splits_duplicate_source_and_recycles_unused_sources():
    mesh = build_icosphere(1)
    state = initialize_plume_magmatism(mesh)
    state.extrusive_volume_km3[:3] = [10.0, 20.0, 30.0]
    state.dyke_volume_km3[:3] = [1.0, 2.0, 3.0]
    state.underplate_volume_km3[:3] = [4.0, 5.0, 6.0]
    state.track_age_myr[:3] = [8.0, 12.0, 16.0]
    state.cumulative_generated_extrusive_volume_km3 = 60.0
    state.cumulative_generated_dyke_volume_km3 = 6.0
    state.cumulative_generated_underplate_volume_km3 = 15.0
    source = np.arange(mesh.cell_count, dtype=np.int32)
    source[0] = 0
    source[1] = 0
    source[2] = -1
    out = advect_plume_magmatism(state, source, 4.0)
    assert out.extrusive_volume_km3[0] == 5.0
    assert out.extrusive_volume_km3[1] == 5.0
    assert out.deep_recycled_extrusive_volume_km3 == 50.0
    assert out.track_age_myr[0] == 12.0
    assert out.track_age_myr[1] == 12.0
    assert abs(igneous_ledger_error_km3(out)) < 1.0e-12


def test_density_aware_fields_match_local_airy_limit_in_topography():
    mesh = build_icosphere(1)
    plates = random_plate_system(mesh, 4, 2801, 0.2, 0.1, 0.4)
    lithosphere = initialize_lithosphere(mesh, plates, 0.25, 2)
    system = PlateSystem(cell_plate=lithosphere.cell_plate.copy(), plates=plates.plates)
    boundaries = boundary_records_for_state(
        mesh, lithosphere, system, RADIUS_KM, 4.0, 1.0
    )
    magmatic = initialize_plume_magmatism(mesh)
    area = mesh.physical_cell_areas_km2(RADIUS_KM)
    cell = 5
    magmatic.extrusive_volume_km3[cell] = area[cell] * 1.0
    magmatic.dyke_volume_km3[cell] = area[cell] * 2.0
    magmatic.underplate_volume_km3[cell] = area[cell] * 3.0
    params = PlumeMagmatismParameters()
    extrusive, load, intrusive, total = magmatic_topography_fields(
        mesh, magmatic, RADIUS_KM, params
    )
    components = topography_components(
        mesh,
        lithosphere,
        boundaries,
        TopographyParameters(),
        RADIUS_KM,
        magmatic_extrusive_thickness_m=extrusive,
        magmatic_extrusive_load_m=load,
        magmatic_intrusive_support_m=intrusive,
    )
    expected = extrusive + load + intrusive
    assert total[cell] == 6000.0
    assert np.allclose(
        components["magmatic_net_local_isostatic_support_m"], expected
    )
    assert expected[cell] > 0.0


def test_same_step_sequence_is_bitwise_deterministic():
    mesh = build_icosphere(1)
    productivity = np.linspace(0.0, 1.0, mesh.cell_count)
    extension = np.linspace(1.0, 0.0, mesh.cell_count)
    params = PlumeMagmatismParameters()
    left = initialize_plume_magmatism(mesh)
    right = deepcopy(left)
    for _ in range(3):
        left, _ = advance_plume_magmatism(
            mesh, left, productivity, extension, 4.0, RADIUS_KM, params
        )
    for _ in range(2):
        right, _ = advance_plume_magmatism(
            mesh, right, productivity, extension, 4.0, RADIUS_KM, params
        )
    right, _ = advance_plume_magmatism(
        mesh, right, productivity, extension, 4.0, RADIUS_KM, params
    )
    assert np.array_equal(left.extrusive_volume_km3, right.extrusive_volume_km3)
    assert np.array_equal(left.dyke_volume_km3, right.dyke_volume_km3)
    assert np.array_equal(left.underplate_volume_km3, right.underplate_volume_km3)
    assert np.array_equal(left.track_age_myr, right.track_age_myr)
    assert left.cumulative_generated_underplate_volume_km3 == right.cumulative_generated_underplate_volume_km3
