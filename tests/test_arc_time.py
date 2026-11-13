from copy import deepcopy
from dataclasses import replace,asdict
import math
import pytest
from tectonics.arc_time import arc_time_slices
from tectonics.subduction_memory import SlabZone,SubductionMemoryState,SubductionMemoryParameters
from tectonics.volcanic_arc import VolcanicArcParameters


def constant_dip_state(age,speed=20.,dip=35.):
    length=age*speed*.9
    z=SlabZone(0,1,active_age_myr=age,slab_length_km=length,dip_deg=dip,
               slab_depth_km=length*math.sin(math.radians(dip)),convergence_rate_km_per_myr=speed)
    return SubductionMemoryState(time_myr=age,zones={(0,1):z})


@pytest.mark.parametrize('dt',[4.,3.,2.,1.])
@pytest.mark.parametrize('activation',[8.,10.3])
def test_frozen_age_and_depth_gate_have_no_early_or_missing_source(dt,activation):
    slab=SubductionMemoryParameters(initial_dip_deg=35.,mature_dip_deg=35.)
    arc=VolcanicArcParameters(min_active_age_myr=8.,min_active_slab_depth_km=activation*20*.9*math.sin(math.radians(35.)))
    t=0.;dose=0.;first=None
    while t<20.:
        h=min(dt,20.-t);before=constant_dip_state(t);after=constant_dip_state(t+h)
        saved=(deepcopy(before),deepcopy(after))
        pieces=arc_time_slices(before,after,h,slab,arc)
        assert sum(hi-lo for lo,hi,_ in pieces)==pytest.approx(h,abs=1e-13)
        for lo,hi,sample in pieces:
            z=sample.zones[0,1]
            if z.active_age_myr>=arc.min_active_age_myr and z.slab_depth_km>=arc.min_active_slab_depth_km:
                assert t+lo>=activation-3e-13
                dose+=hi-lo
                if first is None:first=t+lo
        assert before.zones[0,1].active_age_myr==saved[0].zones[0,1].active_age_myr
        assert after.zones[0,1].active_age_myr==saved[1].zones[0,1].active_age_myr
        t+=h
    assert first==pytest.approx(activation,abs=3e-13)
    assert dose==pytest.approx(20.-activation,abs=3e-13)


def test_gate_exactly_at_endpoint_has_zero_production_in_preceding_interval():
    before=constant_dip_state(4.);after=constant_dip_state(8.)
    pieces=arc_time_slices(before,after,4.,SubductionMemoryParameters(mature_dip_deg=35.),VolcanicArcParameters())
    assert all(sample.zones[0,1].active_age_myr<8. for _,_,sample in pieces)


def test_saturated_depth_and_length_use_existing_law_without_false_activation():
    slab=SubductionMemoryParameters(initial_dip_deg=35.,mature_dip_deg=35.,slab_length_cap_km=100.,slab_depth_cap_km=40.)
    before=constant_dip_state(6.);after=constant_dip_state(10.)
    for m in (before,after):m.zones[0,1].slab_length_km=100.;m.zones[0,1].slab_depth_km=40.
    pieces=arc_time_slices(before,after,4.,slab,VolcanicArcParameters(min_active_slab_depth_km=55.))
    assert all(sample.zones[0,1].slab_depth_km==40. for _,_,sample in pieces)


def test_new_detachment_rejects_unknown_event_time_and_corrupt_endpoint_rejects():
    before=constant_dip_state(4.);after=constant_dip_state(8.);slab=SubductionMemoryParameters(mature_dip_deg=35.)
    after.zones[0,1].broken_off=True
    with pytest.raises(ValueError,match='breakoff'):arc_time_slices(before,after,4.,slab,VolcanicArcParameters())
    after.zones[0,1].broken_off=False;after.zones[0,1].slab_depth_km+=1
    with pytest.raises(ValueError,match='endpoint'):arc_time_slices(before,after,4.,slab,VolcanicArcParameters())
