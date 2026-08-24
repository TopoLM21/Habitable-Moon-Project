import numpy as np

from tectonics.lithosphere import CrustType, advance_lithosphere, initialize_lithosphere, refresh_mechanical_lithosphere
from tectonics.mesh import build_icosphere
from tectonics.plates import random_plate_system
from tectonics.tides import constant_eccentricity


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
    state = initialize_lithosphere(mesh, plates, continental_fraction=0.25, continental_nuclei=4)
    return mesh, plates, state


def test_initial_lithosphere_contains_both_crust_types() -> None:
    mesh, plates, state = _setup(2)
    assert state.cell_plate.shape == (mesh.cell_count,)
    assert np.any(state.crust_type == int(CrustType.OCEANIC))
    assert np.any(state.crust_type == int(CrustType.CONTINENTAL))
    assert np.all(state.crust_thickness_km > 0.0)


def test_backward_transport_keeps_surface_fully_occupied() -> None:
    mesh, plates, state = _setup(2)
    nxt, strain, weakening, diag = advance_lithosphere(
        mesh, plates, state, 2.0, 5287.0, 7.12, 47.0, constant_eccentricity(0.00047)
    )
    assert nxt.cell_plate.shape == (mesh.cell_count,)
    assert np.all(nxt.cell_plate >= 0)
    assert np.all(np.isfinite(nxt.crust_age_myr))
    assert np.all(nxt.crust_thickness_km > 0.0)
    assert np.all((nxt.tidal_damage >= 0.0) & (nxt.tidal_damage <= 1.0))
    assert strain.shape == (mesh.cell_count,)
    assert weakening.shape == (mesh.cell_count,)
    assert 0.0 <= diag.gap_fraction <= 1.0
    assert 0.0 <= diag.overlap_fraction <= 1.0


def test_tidal_damage_disappears_when_eccentricity_is_zero() -> None:
    mesh, plates, state = _setup(2)
    nxt, _, _, _ = advance_lithosphere(
        mesh, plates, state, 2.0, 5287.0, 7.12, 47.0, constant_eccentricity(0.0)
    )
    assert np.allclose(nxt.tidal_damage, 0.0)


def test_lithosphere_step_is_deterministic() -> None:
    mesh, plates, state = _setup(2)
    kwargs = dict(
        mesh=mesh,
        initial_system=plates,
        state=state,
        dt_myr=2.0,
        radius_km=5287.0,
        surface_gravity_m_s2=7.12,
        rotation_period_hours=47.0,
        eccentricity_history=constant_eccentricity(0.00047),
    )
    a = advance_lithosphere(**kwargs)[0]
    b = advance_lithosphere(**kwargs)[0]
    assert np.array_equal(a.cell_plate, b.cell_plate)
    assert np.array_equal(a.crust_type, b.crust_type)
    assert np.allclose(a.crust_age_myr, b.crust_age_myr)
    assert np.allclose(a.crust_thickness_km, b.crust_thickness_km)
    assert np.allclose(a.tidal_damage, b.tidal_damage)


def test_mechanical_lithosphere_is_advected_with_surface_parcels() -> None:
    mesh, plates, state = _setup(2)
    # Give each source cell a unique mechanical marker so a forgotten fixed-grid
    # field is easy to detect.  Density anomaly stays physical and positive.
    state.mantle_lithosphere_thickness_km = 20.0 + np.arange(mesh.cell_count, dtype=float) * 0.01
    state.mantle_lithosphere_density_anomaly_kg_m3 = np.full(mesh.cell_count, 60.0)
    nxt, _, _, diag = advance_lithosphere(
        mesh, plates, state, 2.0, 5287.0, 7.12, 47.0, constant_eccentricity(0.00047)
    )
    assert nxt.mantle_lithosphere_thickness_km is not None
    assert nxt.mantle_lithosphere_density_anomaly_kg_m3 is not None
    assert np.all(np.isfinite(nxt.mantle_lithosphere_thickness_km))
    # Where the surface winner came from an old parcel, the mechanical marker
    # must equal that source parcel's marker exactly.
    src = diag.material_source_index
    moved = src >= 0
    expected = state.mantle_lithosphere_thickness_km[src[moved]]
    assert np.array_equal(nxt.mantle_lithosphere_thickness_km[moved], expected)
