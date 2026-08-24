import numpy as np

from tectonics.continental import (
    ContinentalCycleParameters,
    advance_continental_cycle,
    initialize_continental_cycle,
)
from tectonics.kinematics import BoundaryRecord, BoundaryType
from tectonics.lithosphere import CrustType, LithosphereState
from tectonics.mesh import build_icosphere


def _state(mesh, crust_type, ages=None, thickness=None, damage=None):
    n = mesh.cell_count
    return LithosphereState(
        time_myr=4.0,
        cell_plate=np.zeros(n, dtype=np.int32),
        crust_type=np.asarray(crust_type, dtype=np.int8),
        crust_age_myr=np.zeros(n) if ages is None else np.asarray(ages, dtype=float),
        crust_thickness_km=np.full(n, 7.0) if thickness is None else np.asarray(thickness, dtype=float),
        tidal_damage=np.zeros(n) if damage is None else np.asarray(damage, dtype=float),
    )


def _boundary(mesh, a, b, rate=-50.0):
    midpoint = mesh.centroids[a] + mesh.centroids[b]
    midpoint /= np.linalg.norm(midpoint)
    return BoundaryRecord(a,b,0,1,0,1,midpoint,rate,0.0,abs(rate),BoundaryType.CONVERGENT)


def test_cycle_initializes_zero_potential():
    mesh = build_icosphere(1)
    cycle = initialize_continental_cycle(mesh)
    assert cycle.time_myr == 0.0
    assert cycle.felsic_potential.shape == (mesh.cell_count,)
    assert np.all(cycle.felsic_potential == 0.0)


def test_ocean_ocean_arc_builds_felsic_potential():
    mesh = build_icosphere(1)
    n = mesh.cell_count
    types = np.full(n, int(CrustType.OCEANIC), dtype=np.int8)
    ages = np.full(n, 20.0); ages[0] = 80.0; ages[1] = 10.0
    st = _state(mesh, types, ages=ages)
    cyc = initialize_continental_cycle(mesh)
    params = ContinentalCycleParameters(arc_maturation_rate_per_myr=0.1)
    _, new_cyc, diag = advance_continental_cycle(mesh, st, [_boundary(mesh,0,1)], cyc, 4.0, 5287.0, params)
    assert new_cyc.felsic_potential[1] > 0.0
    assert diag.max_felsic_potential > 0.0


def test_persistent_arc_can_make_juvenile_continent():
    mesh = build_icosphere(1)
    n = mesh.cell_count
    types = np.full(n, int(CrustType.OCEANIC), dtype=np.int8)
    ages = np.full(n, 20.0); ages[0] = 80.0; ages[1] = 10.0
    st = _state(mesh, types, ages=ages)
    cyc = initialize_continental_cycle(mesh)
    cyc.felsic_potential[1] = 0.95
    params = ContinentalCycleParameters(arc_maturation_rate_per_myr=0.1, juvenile_continental_thickness_km=26.0)
    out, _, diag = advance_continental_cycle(mesh, st, [_boundary(mesh,0,1)], cyc, 4.0, 5287.0, params)
    assert out.crust_type[1] == int(CrustType.CONTINENTAL)
    assert np.isclose(out.crust_thickness_km[1], 26.0)
    assert diag.juvenile_arc_area_created_km2 > 0.0


def test_continental_arc_thickens_overriding_margin():
    mesh = build_icosphere(1)
    n = mesh.cell_count
    types = np.full(n, int(CrustType.OCEANIC), dtype=np.int8); types[1] = int(CrustType.CONTINENTAL)
    thick = np.full(n, 7.0); thick[1] = 35.0
    st = _state(mesh, types, thickness=thick)
    cyc = initialize_continental_cycle(mesh)
    params = ContinentalCycleParameters(continental_arc_thickening_km_per_myr=0.1)
    out, _, diag = advance_continental_cycle(mesh, st, [_boundary(mesh,0,1)], cyc, 4.0, 5287.0, params)
    assert out.crust_thickness_km[1] > 35.0
    assert diag.arc_thickening_volume_km3 > 0.0


def test_overthick_crust_delaminates():
    mesh = build_icosphere(1)
    n = mesh.cell_count
    types = np.full(n, int(CrustType.OCEANIC), dtype=np.int8); types[3] = int(CrustType.CONTINENTAL)
    thick = np.full(n, 7.0); thick[3] = 70.0
    st = _state(mesh, types, thickness=thick)
    cyc = initialize_continental_cycle(mesh)
    params = ContinentalCycleParameters(delamination_threshold_km=60.0, delamination_target_km=54.0, delamination_rate_per_myr=0.1)
    out, _, diag = advance_continental_cycle(mesh, st, [], cyc, 4.0, 5287.0, params)
    assert out.crust_thickness_km[3] < 70.0
    assert diag.delaminated_volume_km3 > 0.0


def test_damaged_thin_continental_margin_can_be_recycled():
    mesh = build_icosphere(1)
    n = mesh.cell_count
    types = np.full(n, int(CrustType.OCEANIC), dtype=np.int8); types[1] = int(CrustType.CONTINENTAL)
    thick = np.full(n, 7.0); thick[1] = 21.2
    damage = np.zeros(n); damage[1] = 0.9
    st = _state(mesh, types, thickness=thick, damage=damage)
    cyc = initialize_continental_cycle(mesh)
    params = ContinentalCycleParameters(
        subduction_erosion_km_per_myr=1.0,
        subduction_erosion_damage_threshold=0.1,
        recycle_below_thickness_km=21.0,
    )
    out, _, diag = advance_continental_cycle(mesh, st, [_boundary(mesh,0,1)], cyc, 4.0, 5287.0, params)
    assert out.crust_type[1] == int(CrustType.OCEANIC)
    assert diag.subduction_erosion_area_km2 > 0.0
    assert diag.subduction_erosion_volume_km3 > 0.0
