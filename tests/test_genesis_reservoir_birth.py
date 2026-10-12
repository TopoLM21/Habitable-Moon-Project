import numpy as np
import pytest
from tectonics.mesh import build_icosphere
from tectonics.genesis_bulk_birth import initial_bulk_birth
from tectonics.genesis_reservoir_birth import material_proposal,require_loading_request,digest
from tectonics.genesis_shell import ShellParameters,lid_geometry
from tectonics.genesis import GenesisParameters


def origin():
    mesh=build_icosphere(0);h=initial_bulk_birth(mesh,np.full((len(mesh.faces),2),10.),1.,{'fixture':'manufactured'},1400.)
    return mesh,h


@pytest.mark.parametrize('front,per_face_mass,count_per_face',[(0.,0.,0),(.25,5.,1),(1.,20.,2)])
def test_no_partial_full_birth(front,per_face_mass,count_per_face):
    mesh,h=origin();new,births,removed=material_proposal(h,[front]*len(mesh.faces),1.,mesh,1.)
    assert not removed
    assert sum(b['mass_kg'] for b in births)==pytest.approx(per_face_mass*len(mesh.faces))
    assert len(births)==count_per_face*len(mesh.faces)
    assert [c.id for c in new.cohorts]==list(range(1,len(new.cohorts)+1))
    assert h.cohorts==()


def test_old_ids_records_unchanged_under_growth():
    mesh,h=origin();h,_,_=material_proposal(h,[.25]*len(mesh.faces),1.,mesh,1.)
    new,births,_=material_proposal(h,[.75]*len(mesh.faces),2.,mesh,1.)
    assert new.cohorts[:len(h.cohorts)]==h.cohorts
    assert all(b['id']>len(h.cohorts) for b in births)
    assert sum(b['mass_kg'] for b in births)==pytest.approx(10.*len(mesh.faces))


def test_remelt_retires_without_destroying_or_reusing_ids():
    mesh,h=origin();h,_,_=material_proposal(h,[.75]*len(mesh.faces),1.,mesh,1.)
    new,births,removed=material_proposal(h,[.125]*len(mesh.faces),2.,mesh,1.)
    assert not births and len(new.cohorts)==len(h.cohorts)
    assert sum(x['mass_kg'] for x in removed)==pytest.approx(12.5*len(mesh.faces))
    assert [c.id for c in new.cohorts]==[c.id for c in h.cohorts]
    regrown,births,_=material_proposal(new,[.75]*len(mesh.faces),3.,mesh,1.)
    assert regrown.cohorts[:len(new.cohorts)]==new.cohorts
    assert all(b['id']>len(h.cohorts) for b in births)


def test_bulk_cooling_does_not_imply_connected_birth():
    shell=ShellParameters(column_layers=2);thermal=GenesisParameters()
    hot=np.array([[1800.,1900.]]);cool=np.array([[1700.,1800.]])
    assert lid_geometry(hot,1500.,1800.,shell,thermal)[0][0]==0
    assert lid_geometry(cool,1500.,1800.,shell,thermal)[0][0]==0
    assert lid_geometry(np.array([[1300.,1300.]]),1300.,1300.,shell,thermal)[0][0]==shell.column_depth_km


def test_remelt_guard_even_on_resealed_ready_label():
    body={'status':'READY_FOR_LOADING_PREPARATION','proposed_retirements':[{'mass_kg':1.}], 'mechanical_loading_request':{'ids':[1]}}
    packet={**body,'sha256':digest(body)}
    with pytest.raises(ValueError,match='remelt'):require_loading_request(packet)


def test_full_adapter_growth_prepares_temperatures_without_changing_memory():
    from dataclasses import asdict
    import hashlib,json,math
    from tectonics.genesis_reservoir_birth import prepare
    from tectonics.genesis_bulk_birth import bulk_birth_to_dict
    from tectonics.genesis_reservoir_thermal import Reservoir,CoupledThermalState,specific_h
    mesh,h=origin();h,_,_=material_proposal(h,[.125]*len(mesh.faces),1.,mesh,1.)
    p=GenesisParameters();shell=ShellParameters(column_layers=2)
    memory={'state':{'immutable_fixture':[1.,2.,3.]}}
    root={'runtime_state':{'history':bulk_birth_to_dict(h),'mechanics':memory},'source_packet':{'source':{'parameters':{'thermal':asdict(p),'shell':asdict(shell)}},'current_geometry':{'vertices_unit_xyz':mesh.vertices.tolist(),'faces':mesh.faces.tolist(),'radius_km':1.}}}
    raw=json.dumps(root).encode();origin_json=json.dumps({'legacy_raw_sha256':hashlib.sha256(raw).hexdigest()})
    def state(time,ts,t0):
        cells=[Reservoir('deep_mantle','silicate',100.,100*specific_h(1700.,'silicate',p)),Reservoir('surface_water','water',100.,100*specific_h(ts,'water',p))]
        for i in range(len(mesh.faces)):
            for j,T in enumerate((t0,1800.)):cells.append(Reservoir(f'column:{i}:{j}','silicate',10.,10*specific_h(T,'silicate',p)))
        return CoupledThermalState(time,tuple(cells),json.dumps(asdict(p)),origin_json,math.fsum(c.energy_j for c in cells))
    from dataclasses import replace
    before=state(1.,1300.,1500.);after=state(2.,1290.,1490.)
    cells=list(after.cells);cells[0]=replace(cells[0],energy_j=cells[0].energy_j+before.reference_total_j-after.reference_total_j)
    after=replace(after,cells=tuple(cells),reference_total_j=before.reference_total_j)
    packet=prepare(before,after,raw)
    request=require_loading_request(packet)
    assert packet['status']=='READY_FOR_LOADING_PREPARATION'
    assert len(packet['newborn_requests'])==len(mesh.faces)
    assert len(request['ids'])==2*len(mesh.faces)
    assert packet['parent_memory_sha256']==digest(memory)
    assert packet['latent_heat_source_added'] is False
