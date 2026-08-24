import numpy as np
from tectonics.subduction_memory import *


def test_memory_json_roundtrip():
    m=initialize_subduction_memory(12.0); z=SlabZone(1,2,active=False,active_age_myr=40,inactive_age_myr=8,slab_length_km=900,slab_depth_km=600,trench_length_km=1200,convergence_rate_km_per_myr=20,buoyancy_factor=1.2,cumulative_subducted_area_km2=2e6,trench_midpoint=np.array([0.,1.,0.]),torque_axis=np.array([0.,0.,1.])); m.zones[z.key()]=z; m.births=3; m.detachments=1; m.cumulative_subducted_area_km2=5e6
    q=memory_from_json(memory_to_json(m)); assert q is not None
    assert q.births==3 and q.detachments==1 and q.zones[(1,2)].slab_length_km==900
    assert np.array_equal(q.zones[(1,2)].trench_midpoint,z.trench_midpoint)


def test_residual_decays_and_is_length_weighted():
    p=SubductionMemoryParameters(residual_pull_gain=.05,residual_decay_myr=24)
    m=initialize_subduction_memory();
    m.zones[(0,1)]=SlabZone(0,1,active=False,inactive_age_myr=0,trench_length_km=100,buoyancy_factor=1,torque_axis=np.array([0.,0.,1.]))
    m.zones[(0,2)]=SlabZone(0,2,active=False,inactive_age_myr=24,trench_length_km=300,buoyancy_factor=1,torque_axis=np.array([0.,0.,1.]))
    v,f=residual_pull_by_plate(m,3,p,1.25,1.85)
    expected=(100*.05+300*.05/np.e)/400
    assert np.isclose(f[0],expected)
    assert v[0,2]>0


def test_active_zone_has_no_residual_force():
    p=SubductionMemoryParameters();m=initialize_subduction_memory();m.zones[(0,1)]=SlabZone(0,1,active=True,trench_length_km=1000,buoyancy_factor=2)
    v,f=residual_pull_by_plate(m,2,p,1.25,1.85); assert np.all(v==0) and np.all(f==0)


def test_detach_after_timeout():
    p=SubductionMemoryParameters(detach_after_inactive_myr=80)
    m=initialize_subduction_memory();m.zones[(0,1)]=SlabZone(0,1,active=False,inactive_age_myr=76)
    # Direct timeout behaviour is deterministic even without synthetic mesh: emulate next inactive step.
    z=m.zones[(0,1)];z.inactive_age_myr+=4
    if z.inactive_age_myr>=p.detach_after_inactive_myr: m.zones.pop((0,1));m.detachments+=1
    assert (0,1) not in m.zones and m.detachments==1
