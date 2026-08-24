import numpy as np

from tectonics.flexure import (
    FlexureParameters,
    effective_elastic_thickness_km,
    solve_flexural_response,
)
from tectonics.lithosphere import initialize_lithosphere
from tectonics.mesh import build_icosphere
from tectonics.plates import random_plate_system

RADIUS_KM=5287.0
GRAV=8.8


def _state(subdiv=2):
    mesh=build_icosphere(subdiv)
    plates=random_plate_system(mesh,6,20260820,0.2,0.15,0.6)
    state=initialize_lithosphere(mesh,plates,0.25,4,7.0,35.0,500.0)
    n=mesh.cell_count
    state.mantle_lithosphere_thickness_km=np.full(n,80.0)
    state.mantle_lithosphere_density_anomaly_kg_m3=np.full(n,30.0)
    return mesh,state


def test_effective_elastic_thickness_tracks_mechanical_thickness_and_damage():
    mesh,state=_state(1)
    p=FlexureParameters()
    state.crust_thickness_km[:]=7.0
    state.mantle_lithosphere_thickness_km[:]=40.0
    Te0=effective_elastic_thickness_km(state,p)
    state.mantle_lithosphere_thickness_km[:]=140.0
    Te1=effective_elastic_thickness_km(state,p)
    assert float(np.mean(Te1)) > float(np.mean(Te0))
    state.rift_extension[:]=1.0
    Te2=effective_elastic_thickness_km(state,p)
    assert float(np.mean(Te2)) < float(np.mean(Te1))
    assert np.min(Te2) >= p.min_elastic_thickness_km-1e-12


def test_flexure_preserves_area_weighted_mean_load():
    mesh,state=_state(2)
    p=FlexureParameters()
    src=np.sin(np.arange(mesh.cell_count)*0.137)*700.0+125.0
    rsp,diag,_,_=solve_flexural_response(mesh,state,src,RADIUS_KM,GRAV,p)
    areas=mesh.physical_cell_areas_km2(RADIUS_KM)
    ms=float(np.sum(areas*src)/np.sum(areas))
    mr=float(np.sum(areas*rsp)/np.sum(areas))
    assert abs(ms-mr) < 1e-7
    assert abs(diag.area_mean_source_m-diag.area_mean_response_m) < 1e-7
    assert diag.cg_converged


def test_stiffer_plate_spreads_same_local_load_more_broadly():
    mesh,state=_state(3)
    src=np.zeros(mesh.cell_count); src[0]=3000.0
    p=FlexureParameters(min_elastic_thickness_km=4.0,max_elastic_thickness_km=50.0)
    state.crust_thickness_km[:]=7.0
    state.mantle_lithosphere_thickness_km[:]=20.0
    soft,_,_,_=solve_flexural_response(mesh,state,src,RADIUS_KM,GRAV,p)
    state.mantle_lithosphere_thickness_km[:]=150.0
    stiff,_,_,_=solve_flexural_response(mesh,state,src,RADIUS_KM,GRAV,p)
    # A stiffer plate suppresses the local peak and transfers more of the
    # response into neighboring cells.
    assert stiff[0] < soft[0]
    near=np.asarray(mesh.neighbors[0],dtype=int)
    assert float(np.mean(np.abs(stiff[near]))) > float(np.mean(np.abs(soft[near])))


def test_fourth_order_response_has_opposite_sign_side_lobe():
    mesh,state=_state(4)
    state.crust_thickness_km[:]=7.0
    state.mantle_lithosphere_thickness_km[:]=145.0
    p=FlexureParameters(min_elastic_thickness_km=30.0,max_elastic_thickness_km=50.0)
    src=np.zeros(mesh.cell_count); src[0]=4000.0
    rsp,diag,_,_=solve_flexural_response(mesh,state,src,RADIUS_KM,GRAV,p)
    assert rsp[0] > 0.0
    # A biharmonic elastic plate has a damped oscillatory Green response;
    # the negative peripheral lobe distinguishes it from simple diffusion.
    assert float(np.min(rsp)) < -1e-3
    assert diag.cg_converged
