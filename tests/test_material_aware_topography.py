import numpy as np

from tectonics.lithosphere import CrustType, initialize_lithosphere
from tectonics.mesh import build_icosphere
from tectonics.plates import random_plate_system
from tectonics.topography import (
    TopographyParameters,
    equilibrium_elevation,
    material_topography_endmembers,
)

RADIUS = 5287.0


def _world(sub=1):
    mesh = build_icosphere(sub)
    plates = random_plate_system(mesh, 4, 12345, 0.2, 0.1, 0.3)
    state = initialize_lithosphere(mesh, plates, 0.28, 2, 7.0, 35.0, 500.0, radius_km=RADIUS)
    return mesh, state


def test_material_aware_endmembers_recover_pure_ocean_and_continent():
    mesh, state = _world()
    p = TopographyParameters()
    areas = mesh.physical_cell_areas_km2(RADIUS)
    ocean = int(np.flatnonzero(state.crust_type == int(CrustType.OCEANIC))[0])
    cont = int(np.flatnonzero(state.crust_type == int(CrustType.CONTINENTAL))[0])

    target, _ = equilibrium_elevation(mesh, state, [], p, RADIUS)
    frac, h, ocean_end, cont_end = material_topography_endmembers(mesh, state, RADIUS, p)

    assert frac[ocean] == 0.0
    assert frac[cont] == 1.0
    assert np.isclose(target[ocean], ocean_end[ocean])
    assert np.isclose(target[cont], cont_end[cont])
    assert np.isclose(h[cont], state.continental_volume_km3[cont] / areas[cont])


def test_half_covered_cell_is_area_weighted_between_endmembers():
    mesh, state = _world()
    p = TopographyParameters()
    areas = mesh.physical_cell_areas_km2(RADIUS)
    cell = int(np.flatnonzero(state.crust_type == int(CrustType.OCEANIC))[0])
    state.crust_age_myr[cell] = 80.0
    state.continental_fraction[cell] = 0.5
    state.continental_volume_km3[cell] = areas[cell] * 0.5 * 35.0
    # Deliberately leave the legacy visible raster oceanic: v0.15 must use the
    # independent material fields rather than the >=50% crust label.
    state.crust_type[cell] = int(CrustType.OCEANIC)
    state.crust_thickness_km[cell] = 7.0

    target, _ = equilibrium_elevation(mesh, state, [], p, RADIUS)
    frac, h, ocean_end, cont_end = material_topography_endmembers(mesh, state, RADIUS, p)
    expected = 0.5 * ocean_end[cell] + 0.5 * cont_end[cell]

    assert np.isclose(frac[cell], 0.5)
    assert np.isclose(h[cell], 35.0)
    assert np.isclose(target[cell], expected)


def test_material_aware_relief_is_continuous_across_legacy_visibility_threshold():
    mesh, state = _world()
    p = TopographyParameters()
    areas = mesh.physical_cell_areas_km2(RADIUS)
    cells = np.flatnonzero(state.crust_type == int(CrustType.OCEANIC))[:2]
    assert len(cells) == 2
    a, b = map(int, cells)
    for cell, frac, legacy in ((a, 0.49, CrustType.OCEANIC), (b, 0.51, CrustType.CONTINENTAL)):
        state.crust_age_myr[cell] = 80.0
        state.continental_fraction[cell] = frac
        state.continental_volume_km3[cell] = areas[cell] * frac * 35.0
        state.crust_type[cell] = int(legacy)
        state.crust_thickness_km[cell] = 35.0 if legacy == CrustType.CONTINENTAL else 7.0

    target, _ = equilibrium_elevation(mesh, state, [], p, RADIUS)
    # A 2 percentage-point change in coverage should cause a modest change, not
    # the multi-kilometre binary jump of the legacy >=50% raster.
    assert 0.0 < target[b] - target[a] < 300.0


def test_thicker_material_raises_partial_continental_cell():
    mesh, state = _world()
    p = TopographyParameters()
    areas = mesh.physical_cell_areas_km2(RADIUS)
    cells = np.flatnonzero(state.crust_type == int(CrustType.OCEANIC))[:2]
    a, b = map(int, cells)
    for cell, thickness in ((a, 30.0), (b, 50.0)):
        f = 0.7
        state.crust_age_myr[cell] = 80.0
        state.continental_fraction[cell] = f
        state.continental_volume_km3[cell] = areas[cell] * f * thickness
        state.crust_type[cell] = int(CrustType.OCEANIC)
        state.crust_thickness_km[cell] = 7.0

    target, _ = equilibrium_elevation(mesh, state, [], p, RADIUS)
    assert target[b] > target[a]
