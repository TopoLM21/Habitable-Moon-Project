import numpy as np

from tectonics.lithosphere import (
    _redistribute_continental_footprint_overflow,
    advance_lithosphere,
    continental_material_fields,
    initialize_lithosphere,
)
from tectonics.mesh import build_icosphere
from tectonics.plates import random_plate_system
from tectonics.tides import constant_eccentricity
from tectonics.transport import SubgridTransportParameters, initialize_transport_state


def test_footprint_overflow_preserves_area_and_volume_without_tower():
    mesh = build_icosphere(1)
    radius = 5287.0
    areas = mesh.physical_cell_areas_km2(radius)
    frac = np.zeros(mesh.cell_count, dtype=float)
    vol = np.zeros(mesh.cell_count, dtype=float)
    src = 0
    # Two full 35-km continental footprints have landed on one discrete target.
    frac[src] = 2.0
    vol[src] = 2.0 * areas[src] * 35.0
    before_area = float(np.sum(areas * frac))
    before_volume = float(np.sum(vol))

    moved, raw_max, post_max, _ = _redistribute_continental_footprint_overflow(
        mesh, areas, frac, vol, np.array([src], dtype=np.int32)
    )

    assert moved > 0.0
    assert raw_max > 60.0  # old one-cell raster would look like a ~70 km stack
    assert post_max < 40.0
    assert np.max(frac) <= 1.0 + 1e-12
    assert np.isclose(np.sum(areas * frac), before_area, rtol=0.0, atol=1e-5)
    assert np.isclose(np.sum(vol), before_volume, rtol=0.0, atol=1e-5)


def test_conservative_step_preserves_material_area_and_volume_before_explicit_rifting():
    mesh = build_icosphere(2)
    radius = 5287.0
    system = random_plate_system(mesh, 6, 20260819, 0.2, 0.15, 0.6)
    state = initialize_lithosphere(mesh, system, 0.28, 4, radius_km=radius)
    areas = mesh.physical_cell_areas_km2(radius)
    f0, v0 = continental_material_fields(state, areas)
    a0 = float(np.sum(areas * f0))
    vv0 = float(np.sum(v0))

    ts = initialize_transport_state(len(system.plates))
    tp = SubgridTransportParameters(
        min_changed_fraction=0.0,
        min_p75_cell_spacing_fraction=0.0,
        forced_min_changed_fraction=0.0,
    )
    nxt, _, _, diag = advance_lithosphere(
        mesh, system, state, 4.0, radius, 7.12, 47.0,
        constant_eccentricity(0.00047),
        continental_extension_min_duration_myr=1e9,
        transport_state=ts,
        transport_parameters=tp,
    )
    f1, v1 = continental_material_fields(nxt, areas)
    assert np.max(f1) <= 1.0 + 1e-10
    assert abs(diag.continental_material_area_error_km2) < 1e-4
    assert abs(diag.conservative_transport_volume_error_km3) < 1e-4
    assert np.isclose(np.sum(areas * f1), a0, rtol=0.0, atol=1e-4)
    assert np.isclose(np.sum(v1), vv0, rtol=0.0, atol=1e-4)
