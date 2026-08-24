import numpy as np

from tectonics.breakoff import SlabBreakoffParameters, advance_slab_breakoff, slab_pull_multiplier_for_pair
from tectonics.lithosphere import CrustType, LithosphereState
from tectonics.mesh import build_icosphere
from tectonics.subduction_memory import SlabZone, initialize_subduction_memory, residual_pull_by_plate, SubductionMemoryParameters


def make_state(frac=1.0):
    mesh=build_icosphere(1); n=mesh.cell_count
    state=LithosphereState(
        time_myr=0.0,
        cell_plate=np.zeros(n,dtype=np.int32),
        crust_type=np.full(n,int(CrustType.CONTINENTAL if frac>=.5 else CrustType.OCEANIC),dtype=np.int8),
        crust_age_myr=np.full(n,80.0),
        crust_thickness_km=np.full(n,35.0 if frac>=.5 else 7.0),
        tidal_damage=np.zeros(n),
        continental_fraction=np.full(n,float(frac)),
        continental_volume_km3=mesh.physical_cell_areas_km2(6000.0)*float(frac)*35.0,
    )
    return mesh,state


def mature_zone():
    return SlabZone(
        0,1,active=False,active_age_myr=120.0,slab_length_km=1500.0,slab_depth_km=900.0,
        buoyancy_factor=1.2,trench_length_km=1200.0,trench_midpoint=np.array([1.0,0.0,0.0]),
        torque_axis=np.array([0.0,0.0,1.0]),
    )


def test_sustained_continental_arrival_breaks_mature_slab_after_finite_time():
    mesh,state=make_state(1.0); mem=initialize_subduction_memory(); z=mature_zone();mem.zones[z.key()]=z
    p=SlabBreakoffParameters()
    for _ in range(8):
        mem,d=advance_slab_breakoff(mesh,state,mem,6000.0,4.0,p)
        state.time_myr += 4.0
        if z.broken_off: break
    assert z.broken_off
    assert mem.breakoffs==1
    assert 8.0 <= z.breakoff_time_myr <= 32.0
    assert slab_pull_multiplier_for_pair(mem,0,1)==0.0


def test_brief_continental_contact_relaxes_instead_of_instant_breakoff():
    mesh,state=make_state(1.0);mem=initialize_subduction_memory();z=mature_zone();mem.zones[z.key()]=z;p=SlabBreakoffParameters()
    for _ in range(2):
        mem,_=advance_slab_breakoff(mesh,state,mem,6000.0,4.0,p);state.time_myr+=4.0
    damage=z.breakoff_damage
    assert 0.0 < damage < 1.0
    state.continental_fraction[:] = 0.0
    state.crust_type[:] = int(CrustType.OCEANIC)
    for _ in range(5):
        mem,_=advance_slab_breakoff(mesh,state,mem,6000.0,4.0,p);state.time_myr+=4.0
    assert not z.broken_off
    assert z.breakoff_damage < damage


def test_shallow_short_slab_does_not_break_even_under_continent():
    mesh,state=make_state(1.0);mem=initialize_subduction_memory();z=mature_zone();z.slab_length_km=400;z.slab_depth_km=150;mem.zones[z.key()]=z;p=SlabBreakoffParameters()
    for _ in range(10):
        mem,_=advance_slab_breakoff(mesh,state,mem,6000.0,4.0,p);state.time_myr+=4.0
    assert not z.broken_off and z.breakoff_damage==0.0


def test_broken_slab_has_no_residual_pull_and_tombstone_expires():
    mesh,state=make_state(1.0);mem=initialize_subduction_memory();z=mature_zone();z.broken_off=True;z.active=False;mem.zones[z.key()]=z
    sp=SubductionMemoryParameters();v,f=residual_pull_by_plate(mem,2,sp,1.25,1.85)
    assert np.all(v==0.0) and np.all(f==0.0)
    bp=SlabBreakoffParameters(post_breakoff_cooldown_myr=12.0)
    for _ in range(3):
        mem,_=advance_slab_breakoff(mesh,state,mem,6000.0,4.0,bp);state.time_myr+=4.0
    assert (0,1) not in mem.zones


def test_breakoff_state_json_roundtrip_is_covered_by_existing_memory_serializer():
    from tectonics.subduction_memory import memory_to_json,memory_from_json
    m=initialize_subduction_memory();z=mature_zone();z.breakoff_damage=.7;z.continental_collision_age_myr=12;z.last_front_continental_fraction=.8;z.broken_off=True;z.post_breakoff_age_myr=4;z.breakoff_time_myr=44;m.zones[z.key()]=z;m.breakoffs=2
    q=memory_from_json(memory_to_json(m));zz=q.zones[(0,1)]
    assert q.breakoffs==2 and zz.broken_off and zz.breakoff_damage==.7 and zz.breakoff_time_myr==44
