import numpy as np

from tectonics.lithosphere import CrustType, initialize_lithosphere
from tectonics.mesh import build_icosphere
from tectonics.plates import Plate, PlateSystem, random_plate_system
from tectonics.transport import (
    SubgridTransportParameters,
    build_transport_map,
    initialize_transport_state,
    quaternion_angle_deg,
    remap_transport_state,
)


def _setup(subdivisions=2, speed_min=0.15, speed_max=0.6):
    mesh = build_icosphere(subdivisions)
    system = random_plate_system(mesh, 6, 20260819, 0.2, speed_min, speed_max)
    state = initialize_lithosphere(mesh, system, continental_fraction=0.28, continental_nuclei=4)
    return mesh, system, state


def test_conservative_assignment_is_one_to_one_within_each_plate():
    mesh, system, state = _setup(2)
    ts = initialize_transport_state(len(system.plates))
    params = SubgridTransportParameters(min_changed_fraction=0.0, min_p75_cell_spacing_fraction=0.0)
    out = build_transport_map(mesh, system, state, 4.0, ts, params)
    for pid, target in enumerate(out.source_to_target):
        src_count = int(np.sum(state.cell_plate == pid))
        assert len(target) == src_count
        assert len(np.unique(target)) == src_count
        assert int(np.sum(out.covered[pid])) == src_count


def test_area_scaled_parcels_preserve_continental_volume_before_interplate_resolution():
    mesh, system, state = _setup(2)
    areas = mesh.physical_cell_areas_km2(5287.0)
    ts = initialize_transport_state(len(system.plates))
    params = SubgridTransportParameters(min_changed_fraction=0.0, min_p75_cell_spacing_fraction=0.0)
    out = build_transport_map(mesh, system, state, 4.0, ts, params)

    before = float(np.sum(areas[state.crust_type == int(CrustType.CONTINENTAL)] * state.crust_thickness_km[state.crust_type == int(CrustType.CONTINENTAL)]))
    after = 0.0
    for pid, targets in enumerate(out.source_to_target):
        src = np.flatnonzero(state.cell_plate == pid)
        cont = state.crust_type[src] == int(CrustType.CONTINENTAL)
        # h_target = h_source * A_source / A_target -> A_target*h_target = A_source*h_source
        after += float(np.sum(areas[targets[cont]] * state.crust_thickness_km[src[cont]] * areas[src[cont]] / areas[targets[cont]]))
    assert np.isclose(after, before, rtol=0.0, atol=1e-6)


def test_subcell_rotation_accumulates_until_material_moves():
    mesh = build_icosphere(2)
    base = random_plate_system(mesh, 1, 7, 0.0, 0.02, 0.02)
    p = base.plates[0]
    system = PlateSystem(
        cell_plate=base.cell_plate.copy(),
        plates=(Plate(p.plate_id, p.seed_cell, p.euler_axis.copy(), np.deg2rad(0.02)),),
    )
    state = initialize_lithosphere(mesh, system, continental_fraction=0.28, continental_nuclei=1)
    ts = initialize_transport_state(1)
    params = SubgridTransportParameters(
        min_changed_fraction=0.10,
        min_p75_cell_spacing_fraction=0.20,
        max_hold_myr=200.0,
        forced_min_changed_fraction=0.02,
    )
    original = np.arange(mesh.cell_count, dtype=np.int32)
    moved = False
    residual_seen = False
    for _ in range(80):
        out = build_transport_map(mesh, system, state, 4.0, ts, params)
        residual_seen |= quaternion_angle_deg(ts.residual_quaternions[0]) > 0.01
        target = out.source_to_target[0]
        if not np.array_equal(target, original):
            moved = True
            break
    assert residual_seen
    assert moved


def test_topology_remap_preserves_parent_residual_for_split_children():
    mesh = build_icosphere(1)
    old = random_plate_system(mesh, 2, 12, 0.0, 0.2, 0.2)
    ts = initialize_transport_state(2)
    ts.residual_quaternions[0] = np.array([0.999, 0.0, 0.0, 0.0447])
    ts.residual_quaternions[0] /= np.linalg.norm(ts.residual_quaternions[0])
    cells0 = np.flatnonzero(old.cell_plate == 0)
    new_owner = old.cell_plate.copy()
    new_owner[cells0[len(cells0)//2:]] = 2
    plates = list(old.plates)
    parent = old.plates[0]
    plates.append(Plate(2, int(cells0[-1]), parent.euler_axis.copy(), parent.angular_speed_rad_per_myr))
    new = PlateSystem(new_owner, tuple(plates))
    remapped = remap_transport_state(old, new, ts)
    assert quaternion_angle_deg(remapped.residual_quaternions[0]) > 1.0
    assert quaternion_angle_deg(remapped.residual_quaternions[2]) > 1.0
