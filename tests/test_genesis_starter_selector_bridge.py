from copy import deepcopy
import json
import numpy as np
import pytest
from analysis.genesis_starter_selector_bridge.adapter import eligible,select,save,load
from tectonics.mesh import build_icosphere,connected_components
from tectonics.plates import Plate,PlateSystem
from tectonics.genesis_starter_topology import split_starter_band


def fixture():
    mesh=build_icosphere(3);n=mesh.cell_count;weak=np.abs(mesh.centroids[:,0])<.12
    history=dict(damage=np.where(weak,.7,.04).tolist(),fault_candidate_age_myr=np.where(weak,.05,0.).tolist(),fault_active=np.zeros(n,dtype=bool).tolist(),membrane_established=True,time_myr=.9)
    p=dict(geometry=dict(vertices_m=(mesh.vertices*5287000).tolist(),faces=mesh.faces.tolist(),owner_ids_per_face=['original-body']*n),state=dict(displacement_m=np.zeros(mesh.vertex_count*6).tolist(),time_myr=.9,fault_history=history,source_parameters=dict(faults=dict(enabled=True,activation_damage=.65,activation_persistence_myr=.05))),selector_parameters=dict(radius_km=5287.,min_child_area_km2=3.1e6,min_band_span_km=1000.))
    return mesh,weak,p


def test_positive_synthetic_selection_split_and_restart(tmp_path):
    mesh,weak,p=fixture();before=deepcopy(p);result=select(p)
    assert result['status']=='CANDIDATE_REQUIRES_PHYSICAL_CUT_MODEL'
    cut=np.array(result['cut']);assert np.all(weak[cut]);assert len(cut)>0
    system=PlateSystem(np.zeros(mesh.cell_count,dtype=np.int32),(Plate(0,0,np.array([0.,0.,1.]),0.),))
    split,event=split_starter_band(mesh,system,cut,**p['selector_parameters'],time_myr=.9)
    assert len(split.plates)==2 and event.kind=='split'
    assert all(len(connected_components(np.flatnonzero(split.cell_plate==i),mesh.neighbors))==1 for i in range(2))
    assert all(x.angular_speed_rad_per_myr==0 for x in split.plates)
    save(tmp_path/'one.json',p);restored=load(tmp_path/'one.json');save(tmp_path/'two.json',restored)
    assert (tmp_path/'one.json').read_bytes()==(tmp_path/'two.json').read_bytes()
    assert select(restored)==result and p==before


@pytest.mark.parametrize('mode',['no_damage','no_persistence','disabled','no_membrane'])
def test_source_eligibility_cannot_be_replaced_by_stress_or_damage_alone(mode):
    _,_,p=fixture();h=p['state']['fault_history'];fp=p['state']['source_parameters']['faults']
    if mode=='no_damage':h['damage']=[.04]*len(h['damage'])
    if mode=='no_persistence':h['fault_candidate_age_myr']=[0.]*len(h['damage'])
    if mode=='disabled':fp['enabled']=False
    if mode=='no_membrane':h['membrane_established']=False
    assert not eligible(h,fp).any();assert select(p)['status']=='NO_CANDIDATE'


def test_corrupt_restart_rejected(tmp_path):
    _,_,p=fixture();path=tmp_path/'one.json';save(path,p);obj=json.loads(path.read_bytes());obj['payload']['state']['time_myr']=2.;path.write_text(json.dumps(obj))
    with pytest.raises(ValueError,match='RESTART_CHECKSUM'):load(path)
