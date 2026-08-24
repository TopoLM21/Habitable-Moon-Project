import numpy as np

from tectonics.continental import ContinentalCycleParameters, advance_continental_cycle, initialize_continental_cycle
from tectonics.lithosphere import CrustType, LithosphereState
from tectonics.mesh import build_icosphere
from tectonics.subduction_memory import SlabZone, initialize_subduction_memory
from tectonics.topography import TopographyParameters, equilibrium_elevation
from tectonics.volcanic_arc import (
    VolcanicArcParameters, compute_volcanic_arc_forcing, projected_arc_center,
    target_slab_depth_km, trench_arc_distance_km,
)

R=6000.0


def _state(mesh, continental_fraction=0.0):
    n=mesh.cell_count; areas=mesh.physical_cell_areas_km2(R)
    frac=np.full(n,float(continental_fraction))
    legacy=np.where(frac>=.5,int(CrustType.CONTINENTAL),int(CrustType.OCEANIC)).astype(np.int8)
    return LithosphereState(
        time_myr=40.0,
        cell_plate=np.where(mesh.centroids[:,1]>=0.0,1,0).astype(np.int32),
        crust_type=legacy,
        crust_age_myr=np.full(n,80.0),
        crust_thickness_km=np.where(legacy==int(CrustType.CONTINENTAL),35.0,7.0),
        tidal_damage=np.zeros(n),
        continental_fraction=frac,
        continental_volume_km3=areas*frac*35.0,
    )


def _zone():
    return SlabZone(
        0,1,active=True,active_age_myr=80.0,slab_length_km=1000.0,slab_depth_km=750.0,
        dip_deg=45.0,trench_length_km=1000.0,convergence_rate_km_per_myr=50.0,
        trench_midpoint=np.array([1.0,0.0,0.0]),torque_axis=np.array([0.0,0.0,1.0]),
    )


def test_arc_distance_follows_slab_depth_over_tan_dip():
    p=VolcanicArcParameters(reference_slab_depth_km=105.0,convergence_depth_sensitivity_km=0.0)
    z=_zone(); depth=target_slab_depth_km(50.0,p); dist=trench_arc_distance_km(z.dip_deg,depth,p)
    center,d2,x2=projected_arc_center(z,R,p)
    assert np.isclose(depth,105.0) and np.isclose(dist,105.0)
    assert np.isclose(d2,depth) and np.isclose(x2,dist)
    ang=np.arccos(np.clip(center@z.trench_midpoint,-1,1))*R
    assert np.isclose(ang,dist)
    assert center[1]>0.0


def test_arc_forcing_is_on_overriding_plate_and_not_trench_cell():
    mesh=build_icosphere(3); state=_state(mesh); mem=initialize_subduction_memory(); z=_zone();mem.zones[z.key()]=z
    p=VolcanicArcParameters(arc_half_width_km=100.0,arc_outer_width_km=260.0)
    field,d=compute_volcanic_arc_forcing(mesh,state,mem,R,p)
    assert d.active_arc_zones==1 and np.max(field)>0
    assert np.all(field[state.cell_plate==0]==0.0)
    peak=int(np.argmax(field)); assert state.cell_plate[peak]==1
    trench_dist=np.arccos(np.clip(mesh.centroids[peak]@z.trench_midpoint,-1,1))*R
    assert trench_dist>20.0


def test_post_breakoff_magmatic_pulse_decays():
    mesh=build_icosphere(3);state=_state(mesh);p=VolcanicArcParameters()
    def peak(age):
        mem=initialize_subduction_memory();z=_zone();z.active=False;z.broken_off=True;z.post_breakoff_age_myr=age;mem.zones[z.key()]=z
        f,d=compute_volcanic_arc_forcing(mesh,state,mem,R,p);return float(np.max(f)),d
    a,d0=peak(0.0);b,d1=peak(20.0)
    assert d0.post_breakoff_pulses==1 and d1.post_breakoff_pulses==1
    assert a>b>0.0


def test_external_arc_topography_places_uplift_where_field_is_nonzero():
    mesh=build_icosphere(1);state=_state(mesh);p=TopographyParameters(arc_uplift_m=1200.0)
    forcing=np.zeros(mesh.cell_count);cell=int(np.flatnonzero(state.cell_plate==1)[0]);forcing[cell]=1.0
    base,_=equilibrium_elevation(mesh,state,[],p,R)
    arc,_=equilibrium_elevation(mesh,state,[],p,R,forcing)
    assert np.isclose(arc[cell]-base[cell],1200.0)
    other=int(np.flatnonzero((state.cell_plate==1)&(np.arange(mesh.cell_count)!=cell))[0])
    assert np.isclose(arc[other]-base[other],0.0)


def test_external_arc_forcing_feeds_existing_felsic_cycle_without_boundary_arc():
    mesh=build_icosphere(1);state=_state(mesh);cycle=initialize_continental_cycle(mesh)
    p=ContinentalCycleParameters(arc_maturation_rate_per_myr=.02,arc_potential_decay_myr=1e12,juvenile_threshold=10.0)
    forcing=np.zeros(mesh.cell_count);cell=int(np.flatnonzero(state.cell_plate==1)[0]);forcing[cell]=0.75
    _,new_cycle,_=advance_continental_cycle(mesh,state,[],cycle,4.0,R,p,volcanic_arc_forcing=forcing)
    assert np.isclose(new_cycle.felsic_potential[cell],0.75*.02*4.0,rtol=0,atol=1e-12)
    assert np.count_nonzero(new_cycle.felsic_potential)>0
