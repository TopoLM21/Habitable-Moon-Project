from tectonics.mesh import build_icosphere, connected_components
from tectonics.plates import random_plate_system


def test_generated_plates_are_connected() -> None:
    mesh = build_icosphere(3)
    system = random_plate_system(
        mesh=mesh,
        plate_count=10,
        seed=1234,
        boundary_roughness=0.25,
        min_speed_deg_per_myr=0.1,
        max_speed_deg_per_myr=0.8,
    )
    for plate_id in range(10):
        cells = (system.cell_plate == plate_id).nonzero()[0]
        assert len(connected_components(cells, mesh.neighbors)) == 1
