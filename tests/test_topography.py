import numpy as np

from tectonics.kinematics import classify_boundaries
from tectonics.lithosphere import CrustType, initialize_lithosphere
from tectonics.mesh import build_icosphere
from tectonics.plates import random_plate_system
from tectonics.topography import TopographyParameters, advance_topography, equilibrium_elevation, initialize_topography


def setup_world(subdiv=2):
    mesh=build_icosphere(subdiv)
    plates=random_plate_system(mesh,6,20260817,0.2,0.15,0.6)
    lith=initialize_lithosphere(mesh,plates,0.28,4,7.0,35.0,500.0)
    bounds=classify_boundaries(mesh,plates,5287.0,4.0,1.0)
    return mesh,plates,lith,bounds


def test_ocean_age_makes_base_bathymetry_deeper():
    mesh,plates,lith,bounds=setup_world(2)
    p=TopographyParameters()
    ocean=np.flatnonzero(lith.crust_type==int(CrustType.OCEANIC))
    assert len(ocean)>2
    lith.crust_age_myr[ocean[0]]=0.0
    lith.crust_age_myr[ocean[1]]=100.0
    target,_=equilibrium_elevation(mesh,lith,[],p)
    assert target[ocean[1]] < target[ocean[0]]


def test_thicker_continent_is_higher_without_boundary_forcing():
    mesh,plates,lith,bounds=setup_world(2)
    p=TopographyParameters()
    cont=np.flatnonzero(lith.crust_type==int(CrustType.CONTINENTAL))
    assert len(cont)>1
    lith.crust_thickness_km[cont[0]]=35.0
    lith.crust_thickness_km[cont[1]]=60.0
    target,_=equilibrium_elevation(mesh,lith,[],p)
    assert target[cont[1]] > target[cont[0]]


def test_initial_topography_is_finite_and_bounded():
    mesh,plates,lith,bounds=setup_world(2)
    p=TopographyParameters()
    topo=initialize_topography(mesh,lith,bounds,p)
    assert topo.elevation_m.shape==(mesh.cell_count,)
    assert np.all(np.isfinite(topo.elevation_m))
    assert np.min(topo.elevation_m)>=p.min_elevation_m-1e-9
    assert np.max(topo.elevation_m)<=p.max_elevation_m+1e-9


def test_advance_topography_is_deterministic():
    mesh,plates,lith,bounds=setup_world(2)
    p=TopographyParameters()
    topo=initialize_topography(mesh,lith,bounds,p)
    a,da,ta=advance_topography(mesh,lith,bounds,topo,2.0,5287.0,p)
    b,db,tb=advance_topography(mesh,lith,bounds,topo,2.0,5287.0,p)
    assert np.allclose(a.elevation_m,b.elevation_m)
    assert np.allclose(ta,tb)
    assert da.eroded_volume_km3==db.eroded_volume_km3


def test_positive_peak_erodes():
    mesh,plates,lith,bounds=setup_world(1)
    p=TopographyParameters(erosion_diffusion_per_myr=0.1,max_erosion_fraction_per_step=0.5,isostatic_relaxation_myr=1e12)
    topo=initialize_topography(mesh,lith,[],p)
    cell=0
    topo.elevation_m[:]=100.0
    topo.elevation_m[cell]=5000.0
    nxt,diag,_=advance_topography(mesh,lith,[],topo,1.0,5287.0,p)
    assert nxt.elevation_m[cell] < 5000.0
    assert diag.eroded_volume_km3 > 0.0

from tectonics.kinematics import BoundaryRecord, BoundaryType
from tectonics.topography import oceanic_plate_depth_m, tectonic_forcing, trench_extra_depth_m


def _fake_convergent(face_a, face_b, plate_a, plate_b, normal_rate=-60.0):
    return BoundaryRecord(
        face_a=int(face_a), face_b=int(face_b), vertex_u=0, vertex_v=1,
        plate_a=int(plate_a), plate_b=int(plate_b), midpoint=np.array([1.0,0.0,0.0]),
        normal_rate_km_per_myr=float(normal_rate), tangential_rate_km_per_myr=0.0,
        relative_speed_km_per_myr=abs(float(normal_rate)), boundary_type=BoundaryType.CONVERGENT,
    )


def test_plate_model_old_ocean_asymptotes_without_hard_75km_cap():
    p=TopographyParameters()
    d=np.asarray(oceanic_plate_depth_m(np.array([0.0,20.0,100.0,500.0]),p))
    assert np.all(np.diff(d)>0.0)
    assert 6200.0 < d[-1] < 6450.0
    assert d[-1] < 7000.0


def test_trench_is_deeper_for_older_faster_subduction_but_bounded():
    mesh,plates,lith,_=setup_world(2)
    ocean=np.flatnonzero(lith.crust_type==int(CrustType.OCEANIC))
    a,b=int(ocean[0]),int(ocean[1])
    lith.crust_age_myr[a]=10.0
    lith.crust_age_myr[b]=120.0
    p=TopographyParameters()
    slow=_fake_convergent(a,b,int(lith.cell_plate[a]),int(lith.cell_plate[b]),-15.0)
    fast=_fake_convergent(b,a,int(lith.cell_plate[b]),int(lith.cell_plate[a]),-90.0)
    ds=trench_extra_depth_m(lith,slow,a,p)
    df=trench_extra_depth_m(lith,fast,b,p)
    assert df>ds
    assert p.trench_min_extra_depth_m <= ds <= p.trench_max_extra_depth_m
    assert p.trench_min_extra_depth_m <= df <= p.trench_max_extra_depth_m


def test_duplicate_boundary_edges_do_not_stack_trench_depth():
    mesh,plates,lith,_=setup_world(2)
    ocean=np.flatnonzero(lith.crust_type==int(CrustType.OCEANIC))
    # Find neighboring ocean cells on different plates.
    pair=None
    for a,b,_,_ in mesh.shared_edges:
        if (lith.crust_type[a]==int(CrustType.OCEANIC) and lith.crust_type[b]==int(CrustType.OCEANIC)
                and lith.cell_plate[a]!=lith.cell_plate[b]):
            pair=(a,b);break
    assert pair is not None
    a,b=pair
    lith.crust_age_myr[a]=100.0; lith.crust_age_myr[b]=20.0
    rec=_fake_convergent(a,b,int(lith.cell_plate[a]),int(lith.cell_plate[b]),-70.0)
    p=TopographyParameters()
    f1,_,c1=tectonic_forcing(mesh,lith,[rec],p)
    f3,_,c3=tectonic_forcing(mesh,lith,[rec,rec,rec],p)
    assert np.allclose(f1,f3)
    assert np.allclose(c1['trench_depth'],c3['trench_depth'])


def test_normal_ocean_plus_max_trench_stays_above_numerical_safety_floor():
    p=TopographyParameters()
    oldest=float(oceanic_plate_depth_m(1000.0,p))
    deepest=-(oldest+p.trench_max_extra_depth_m)
    assert deepest > p.numerical_min_elevation_m + 5000.0
