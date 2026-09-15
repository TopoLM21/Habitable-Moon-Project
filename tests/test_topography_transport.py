"""Relief follows the parcel map without flooding newly moved continents."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from tectonics.hydrosphere import HydrosphereParameters, diagnose_hydrosphere, initialize_hydrosphere
from tectonics.lithosphere import CrustType, initialize_lithosphere
from tectonics.mesh import build_icosphere
from tectonics.plates import random_plate_system
from tectonics.topography import (
    TopographyParameters, TopographyState, advance_topography,
    equilibrium_elevation, transport_topography_with_material,
)

RADIUS = 5287.0


def _world():
    # Equal-area base faces isolate transport from footprint redistribution.
    mesh = build_icosphere(0)
    plates = random_plate_system(mesh, 3, 123, 0.2, 0.1, 0.3)
    state = initialize_lithosphere(mesh, plates, 0.3, 2, radius_km=RADIUS)
    state.crust_age_myr[:] = 80.0
    params = TopographyParameters(erosion_diffusion_per_myr=0.0)
    base, _ = equilibrium_elevation(mesh, state, [], params, RADIUS)
    return mesh, state, params, base


def _permute(state, source):
    moved = deepcopy(state)
    for name in state.__dataclass_fields__:
        values = getattr(state, name)
        if isinstance(values, np.ndarray):
            setattr(moved, name, values[source].copy())
    return moved


def test_no_motion_preserves_arbitrary_relief_and_does_not_mutate_input():
    mesh, state, params, base = _world()
    elevation = base + np.linspace(-700.0, 1200.0, mesh.cell_count)
    fixed = np.linspace(-100.0, 100.0, mesh.cell_count)
    before = elevation.copy()
    out = transport_topography_with_material(
        mesh, state, state, TopographyState(0.0, elevation),
        np.arange(mesh.cell_count), RADIUS, params, fixed,
    )
    np.testing.assert_allclose(out, before, atol=1e-12, rtol=0.0)
    np.testing.assert_array_equal(elevation, before)
    assert not np.shares_memory(out, elevation)


def test_moved_continent_keeps_its_relief_instead_of_inheriting_ocean_hole():
    mesh, state, params, base = _world()
    continent = int(np.flatnonzero(state.crust_type == CrustType.CONTINENTAL)[0])
    ocean = int(np.flatnonzero(state.crust_type == CrustType.OCEANIC)[0])
    source = np.arange(mesh.cell_count)
    source[[continent, ocean]] = source[[ocean, continent]]
    moved = _permute(state, source)
    previous = TopographyState(0.0, base.copy())
    previous.elevation_m[continent] += 1000.0
    fixed, _, _ = advance_topography(
        mesh, moved, [], previous, 4.0, RADIUS, params,
        previous_lithosphere=state, transport_source_index=source,
    )
    legacy, _, _ = advance_topography(mesh, moved, [], previous, 4.0, RADIUS, params)
    assert fixed.elevation_m[ocean] > 350.0
    assert legacy.elevation_m[ocean] < -1000.0
    expected = base[continent] + np.exp(-1.0) * 1000.0
    assert fixed.elevation_m[ocean] == pytest.approx(expected)


def test_reconstructs_changed_mixed_cell_composition_from_new_material():
    mesh, state, params, base = _world()
    moved = deepcopy(state)
    cell = int(np.flatnonzero(state.crust_type == CrustType.OCEANIC)[0])
    source = np.arange(mesh.cell_count)
    source[cell] = int(np.flatnonzero(state.crust_type == CrustType.OCEANIC)[1])
    areas = mesh.physical_cell_areas_km2(RADIUS)
    moved.continental_fraction[cell] = 0.8
    moved.continental_volume_km3[cell] = areas[cell] * 0.8 * 35.0
    expected, _ = equilibrium_elevation(mesh, moved, [], params, RADIUS)
    out = transport_topography_with_material(
        mesh, state, moved, TopographyState(0.0, base + 123.0),
        source, RADIUS, params,
    )
    assert out[cell] == pytest.approx(expected[cell] + 123.0)
    assert out[cell] - base[cell] > 4000.0


def test_stationary_material_cooling_and_thickening_keep_mechanical_relaxation():
    mesh, state, params, base = _world()
    changed = deepcopy(state)
    changed.crust_age_myr[:] += 40.0
    changed.continental_volume_km3[:] *= 1.1
    previous = TopographyState(0.0, base.copy())
    source = np.arange(mesh.cell_count)
    before_relaxation = transport_topography_with_material(
        mesh, state, changed, previous, source, RADIUS, params,
    )
    np.testing.assert_array_equal(before_relaxation, previous.elevation_m)
    legacy, _, target = advance_topography(mesh, changed, [], previous, 4.0, RADIUS, params)
    transported, _, _ = advance_topography(
        mesh, changed, [], previous, 4.0, RADIUS, params,
        previous_lithosphere=state, transport_source_index=source,
    )
    np.testing.assert_array_equal(transported.elevation_m, legacy.elevation_m)
    assert np.max(np.abs(transported.elevation_m - target)) > 100.0


def test_new_crust_has_no_inherited_trench_or_continental_residual():
    mesh, state, params, base = _world()
    moved = deepcopy(state)
    cell = 0
    source = np.arange(mesh.cell_count)
    source[cell] = -1
    moved.crust_type[cell] = int(CrustType.OCEANIC)
    moved.crust_age_myr[cell] = 0.0
    moved.continental_fraction[cell] = 0.0
    moved.continental_volume_km3[cell] = 0.0
    out = transport_topography_with_material(
        mesh, state, moved, TopographyState(0.0, base - 6000.0),
        source, RADIUS, params,
    )
    assert out[cell] == pytest.approx(-params.ridge_axis_depth_m)


def test_mantle_support_remains_fixed_while_parcel_relief_moves():
    mesh, state, params, base = _world()
    source = np.roll(np.arange(mesh.cell_count), 5)
    moved = _permute(state, source)
    fixed = np.linspace(-200.0, 200.0, mesh.cell_count)
    residual = np.linspace(-500.0, 1500.0, mesh.cell_count)
    out = transport_topography_with_material(
        mesh, state, moved, TopographyState(0.0, base + residual + fixed),
        source, RADIUS, params, fixed,
    )
    new_base, _ = equilibrium_elevation(mesh, moved, [], params, RADIUS)
    np.testing.assert_allclose(out - new_base, residual[source] + fixed, atol=1e-11)
    assert not np.allclose(out - new_base, (residual + fixed)[source])


def test_area_preserving_material_permutation_preserves_sea_level_and_water():
    mesh, state, params, base = _world()
    source = np.roll(np.arange(mesh.cell_count), 5)
    moved = _permute(state, source)
    previous = TopographyState(0.0, base + np.linspace(-200.0, 200.0, mesh.cell_count))
    hydro_params = HydrosphereParameters()
    hydro = initialize_hydrosphere(mesh, previous, RADIUS, hydro_params, state, params)
    out = transport_topography_with_material(mesh, state, moved, previous, source, RADIUS, params)
    diag = diagnose_hydrosphere(mesh, moved, TopographyState(4.0, out), hydro, RADIUS, hydro_params, params)
    assert diag.sea_level_m == pytest.approx(0.0, abs=hydro_params.solver_tolerance_m)
    assert abs(diag.relative_volume_error) < 1e-7


def test_disabled_transport_reproduces_original_relaxation():
    mesh, state, params, base = _world()
    previous = TopographyState(0.0, base + 1000.0)
    legacy = advance_topography(mesh, state, [], previous, 4.0, RADIUS, params)[0]
    disabled = advance_topography(
        mesh, state, [], previous, 4.0, RADIUS, replace(params, material_transport_enabled=False),
        previous_lithosphere=state, transport_source_index=np.roll(np.arange(mesh.cell_count), 2),
    )[0]
    np.testing.assert_array_equal(disabled.elevation_m, legacy.elevation_m)


@pytest.mark.parametrize("source", [np.array([-2] * 20), np.array([20] * 20), np.zeros(20), np.zeros(19, dtype=int)])
def test_invalid_material_map_rejected(source):
    mesh, state, params, base = _world()
    with pytest.raises(ValueError, match="transport_source_index"):
        transport_topography_with_material(mesh, state, state, TopographyState(0.0, base), source, RADIUS, params)
