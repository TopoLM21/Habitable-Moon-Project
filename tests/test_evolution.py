import numpy as np

from tectonics.evolution import rotate_points_by_plate, snapshot_at_time
from tectonics.kinematics import classify_boundaries
from tectonics.mesh import build_icosphere, connected_components
from tectonics.plates import random_plate_system


def _system(subdivisions: int = 3):
    mesh = build_icosphere(subdivisions)
    plates = random_plate_system(
        mesh=mesh,
        plate_count=8,
        seed=2026,
        boundary_roughness=0.2,
        min_speed_deg_per_myr=0.1,
        max_speed_deg_per_myr=0.7,
    )
    boundaries = classify_boundaries(mesh, plates, 5287.0, 4.0, 1.0)
    return mesh, plates, boundaries


def test_zero_time_rotation_is_identity() -> None:
    mesh, plates, _ = _system(2)
    moved = rotate_points_by_plate(mesh.centroids, plates.cell_plate, plates, 0.0)
    assert np.allclose(moved, mesh.centroids, atol=1e-14)


def test_rigid_rotation_preserves_marker_norms() -> None:
    mesh, plates, _ = _system(2)
    moved = rotate_points_by_plate(mesh.centroids, plates.cell_plate, plates, 37.5)
    assert np.allclose(np.linalg.norm(moved, axis=1), 1.0, atol=1e-13)


def test_same_plate_pair_angular_distance_is_preserved() -> None:
    mesh, plates, _ = _system(2)
    moved = rotate_points_by_plate(mesh.centroids, plates.cell_plate, plates, 23.0)
    checked = 0
    for a, b, _, _ in mesh.shared_edges:
        if plates.cell_plate[a] != plates.cell_plate[b]:
            continue
        before = float(np.dot(mesh.centroids[a], mesh.centroids[b]))
        after = float(np.dot(moved[a], moved[b]))
        assert np.isclose(before, after, atol=2e-13, rtol=0.0)
        checked += 1
        if checked >= 50:
            break
    assert checked == 50


def test_plate_material_connectivity_is_invariant_by_construction() -> None:
    mesh, plates, _ = _system(3)
    # Membership of Lagrangian material markers never changes in v0.2.
    for plate_id in range(len(plates.plates)):
        cells = np.flatnonzero(plates.cell_plate == plate_id)
        assert len(connected_components(cells, mesh.neighbors)) == 1


def test_zero_time_boundary_copies_coincide() -> None:
    mesh, plates, boundaries = _system(3)
    snap = snapshot_at_time(mesh, plates, boundaries, 5287.0, 0.0)
    assert np.allclose(snap.boundary_side_a, snap.boundary_side_b, atol=1e-14)
    assert np.max(snap.boundary_separation_km) < 1e-4
    assert snap.coverage.uncovered_cell_fraction == 0.0
    assert snap.coverage.multiply_covered_cell_fraction == 0.0


def test_snapshot_is_deterministic_and_finite() -> None:
    mesh, plates, boundaries = _system(3)
    a = snapshot_at_time(mesh, plates, boundaries, 5287.0, 10.0)
    b = snapshot_at_time(mesh, plates, boundaries, 5287.0, 10.0)
    assert np.allclose(a.marker_positions, b.marker_positions)
    assert np.all(np.isfinite(a.boundary_separation_km))
    assert np.all(a.boundary_separation_km >= 0.0)
    assert 0.0 <= a.coverage.uncovered_cell_fraction <= 1.0
    assert 0.0 <= a.coverage.multiply_covered_cell_fraction <= 1.0
