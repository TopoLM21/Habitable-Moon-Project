import numpy as np

from tectonics.crust import advance_oceanic_crust, initialize_oceanic_crust
from tectonics.mesh import build_icosphere
from tectonics.plates import random_plate_system


def _setup(subdivisions: int = 2):
    mesh = build_icosphere(subdivisions)
    plates = random_plate_system(
        mesh=mesh,
        plate_count=6,
        seed=20260817,
        boundary_roughness=0.2,
        min_speed_deg_per_myr=0.15,
        max_speed_deg_per_myr=0.6,
    )
    state = initialize_oceanic_crust(plates)
    return mesh, plates, state


def test_initialize_oceanic_crust_matches_plate_partition() -> None:
    mesh, plates, state = _setup(2)
    assert state.time_myr == 0.0
    assert np.array_equal(state.cell_plate, plates.cell_plate)
    assert state.crust_age_myr.shape == (mesh.cell_count,)
    assert np.all(state.crust_age_myr == 0.0)


def test_advance_preserves_full_surface_occupation() -> None:
    mesh, plates, state = _setup(2)
    nxt, diag = advance_oceanic_crust(mesh, plates, state, dt_myr=0.5, radius_km=5287.0)
    assert nxt.cell_plate.shape == (mesh.cell_count,)
    assert nxt.crust_age_myr.shape == (mesh.cell_count,)
    assert np.all(np.isfinite(nxt.crust_age_myr))
    assert np.all(nxt.crust_age_myr >= 0.0)
    assert np.all(nxt.cell_plate >= 0)
    assert np.all(nxt.cell_plate < len(plates.plates))
    assert 0.0 <= diag.pre_resolution_gap_fraction <= 1.0
    assert 0.0 <= diag.pre_resolution_overlap_fraction <= 1.0


def test_advance_is_deterministic() -> None:
    mesh, plates, state = _setup(2)
    a, da = advance_oceanic_crust(mesh, plates, state, dt_myr=0.75, radius_km=5287.0)
    b, db = advance_oceanic_crust(mesh, plates, state, dt_myr=0.75, radius_km=5287.0)
    assert np.array_equal(a.cell_plate, b.cell_plate)
    assert np.allclose(a.crust_age_myr, b.crust_age_myr)
    assert da.created_area_km2 == db.created_area_km2
    assert da.subducted_area_km2 == db.subducted_area_km2


def test_age_can_only_increase_or_reset_to_zero() -> None:
    mesh, plates, state = _setup(2)
    nxt, _ = advance_oceanic_crust(mesh, plates, state, dt_myr=0.5, radius_km=5287.0)
    unique = np.unique(np.round(nxt.crust_age_myr, 12))
    assert np.all((unique == 0.0) | np.isclose(unique, 0.5))
