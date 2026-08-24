from tectonics.kinematics import classify_boundaries, rigid_motion_residual
from tectonics.mesh import build_icosphere
from tectonics.plates import random_plate_system


def _system():
    mesh = build_icosphere(3)
    plates = random_plate_system(
        mesh=mesh,
        plate_count=8,
        seed=77,
        boundary_roughness=0.2,
        min_speed_deg_per_myr=0.1,
        max_speed_deg_per_myr=0.8,
    )
    return mesh, plates


def test_rigid_plate_internal_distance_rate_is_zero() -> None:
    mesh, plates = _system()
    assert rigid_motion_residual(mesh, plates) < 1e-15


def test_boundary_records_are_finite() -> None:
    mesh, plates = _system()
    records = classify_boundaries(mesh, plates, 5287.0, 4.0, 1.0)
    assert records
    for item in records:
        assert item.relative_speed_km_per_myr >= 0.0
        assert item.plate_a != item.plate_b
