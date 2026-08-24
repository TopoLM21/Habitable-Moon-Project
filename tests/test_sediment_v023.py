from pathlib import Path
import numpy as np

from tectonics.mesh import build_icosphere
from tectonics.plates import random_plate_system, PlateSystem
from tectonics.lithosphere import initialize_lithosphere, boundary_records_for_state
from tectonics.topography import TopographyState, TopographyParameters, initialize_topography
from tectonics.sediment import (
    SedimentParameters, SedimentBudgetState, initialize_sediment_budget,
    sediment_thickness_m, advance_sediments, continental_material_ledger_error_km3,
    _advect_surface_sediment,
)
from tectonics.checkpoint import RunCheckpoint, save_checkpoint, load_checkpoint
from tectonics.continental import initialize_continental_cycle
from tectonics.thermal import ThermalParameters, initialize_thermal_state
from tectonics.topology import PlateTopologyManager, PlateTopologyParameters

R=5287.0

def _world(sub=1):
    mesh=build_icosphere(sub)
    plates=random_plate_system(mesh,4,20260821,0.2,0.1,0.4)
    state=initialize_lithosphere(mesh,plates,0.45,3,radius_km=R)
    return mesh,plates,state


def test_sediment_thickness_is_volume_over_cell_area():
    mesh,_,state=_world(1)
    A=mesh.physical_cell_areas_km2(R)
    state.sediment_volume_km3=A*2.5
    th=sediment_thickness_m(mesh,state,R)
    assert np.allclose(th,2500.0)


def test_surface_parcel_transport_moves_sediment_and_recycles_lost_sources():
    mesh,_,state=_world(1)
    n=mesh.cell_count
    state.sediment_volume_km3=np.arange(1,n+1,dtype=float)
    src=np.arange(n,dtype=np.int32)
    src[0]=-1  # old source 0 disappears; target 0 is newborn
    out,deep=_advect_surface_sediment(state,src)
    assert out[0]==0.0
    assert deep==1.0
    assert abs(np.sum(out)+deep-np.sum(state.sediment_volume_km3))<1e-12


def test_erosion_routes_removed_bedrock_without_losing_mass():
    mesh,_,state=_world(1)
    n=mesh.cell_count
    # Make all cells materially continental for a clean synthetic erosion test.
    A=mesh.physical_cell_areas_km2(R)
    state.continental_fraction=np.ones(n)
    state.continental_volume_km3=A*35.0
    state.crust_type[:]=1; state.crust_thickness_km[:]=35.0
    state.sediment_volume_km3=np.zeros(n)
    previous=state
    z=np.zeros(n);z[0]=2500.0
    topo=TopographyState(4.0,z)
    src=np.arange(n,dtype=np.int32)
    budget=initialize_sediment_budget(0.0)
    state,topo,budget,diag=advance_sediments(mesh,previous,state,topo,src,budget,4.0,R,SedimentParameters())
    assert diag.eroded_bedrock_volume_km3>0.0
    assert abs(diag.surface_sediment_volume_km3-diag.eroded_bedrock_volume_km3)<1e-8
    assert abs(diag.step_surface_mass_error_km3)<1e-8


def test_global_continental_ledger_closes_for_explicit_reservoirs():
    _,_,state=_world(1)
    state.continental_volume_km3=np.array(state.continental_volume_km3,dtype=float)
    initial=float(np.sum(state.continental_volume_km3))
    state.sediment_volume_km3=np.zeros_like(state.continental_volume_km3)
    state.continental_volume_km3[0]-=10.0
    state.sediment_volume_km3[0]=4.0
    b=SedimentBudgetState(deep_recycled_sediment_volume_km3=2.0,cumulative_rift_recycled_volume_km3=3.0)
    # Remaining 1 km3 is accounted as continental-cycle recycling.
    err=continental_material_ledger_error_km3(initial,0.0,state,b,1.0)
    assert abs(err)<1e-12


def test_checkpoint_roundtrip_preserves_sediment_budget_and_field(tmp_path: Path):
    mesh,plates,state=_world(1)
    A=mesh.physical_cell_areas_km2(R)
    state.sediment_volume_km3=0.125*A
    cycle=initialize_continental_cycle(mesh)
    thermal=initialize_thermal_state(0.5,R,7.12,ThermalParameters())
    system=PlateSystem(state.cell_plate.copy(),plates.plates)
    bounds=boundary_records_for_state(mesh,state,system,R,4.0,1.0)
    topo=initialize_topography(mesh,state,bounds,TopographyParameters(),R)
    manager=PlateTopologyManager(PlateTopologyParameters())
    budget=SedimentBudgetState(12.0,5.0,2.0,3.0,4.0)
    cp=RunCheckpoint(state,cycle,thermal,topo,system,system,manager,0.4,float(np.sum(state.continental_volume_km3)),[],[],[],[],[],[],[],sediment_budget=budget,sediment_rows=[{'time_myr':12.0,'surface_sediment_volume_km3':7.0}])
    save_checkpoint(tmp_path/'cp',cp)
    got=load_checkpoint(tmp_path/'cp',PlateTopologyManager(PlateTopologyParameters()))
    assert np.array_equal(got.state.sediment_volume_km3,state.sediment_volume_km3)
    assert got.sediment_budget==budget
    assert got.sediment_rows==cp.sediment_rows
