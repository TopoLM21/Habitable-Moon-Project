from copy import deepcopy
from pathlib import Path
import math
import numpy as np
import pytest
from tectonics.experimental_geological_parcels import remap,transport,COMPOSITION,layout
from tectonics.simplified_genesis import SimplifiedGenesis
from tectonics.genesis_geological_thermal import ThermalGeologicalContinuation,eligibility
from tectonics.genesis_transfer_bridge import encoded


def test_ragged_remap_preserves_each_material_identity():
    parts=[dict(id=str(i),composition=COMPOSITION,mass_kg=m,energy_j=h) for i,(m,h) in enumerate([(2.,10.),(3.,90.),(5.,20.)])]
    result=remap(parts,[1,2]);assert list(map(len,result))==[2,2]
    for p in parts:
        pieces=[q for row in result for q in row if q['id']==p['id']]
        assert sum(x['mass_kg'] for x in pieces)==pytest.approx(p['mass_kg'])
        assert sum(x['energy_j'] for x in pieces)==pytest.approx(p['energy_j'])
        assert all(x['composition']==COMPOSITION for x in pieces)


def test_ragged_transport_new_material_and_recycling():
    M=np.array([2.,3.,1.,2.,3.,100.,1.]);E=M*np.array([10.,20.,30.,40.,50.,60.,70.])
    nodes=[[0,1],[2,3,4]]
    p=dict(nodes=nodes,weights=[[.4,.6],[1/6,2/6,3/6]],node_parts={str(i):[dict(id=str(i),mass_kg=M[i],energy_j=E[i],composition=COMPOSITION)] for i in range(5)},recycled_history=[],generation=0)
    beforeM=M.sum();beforeE=E.sum();report=transport(p,M,E,np.array([1,-1]),np.array([6.,4.]),1.)
    assert M.sum()==pytest.approx(beforeM) and E.sum()==pytest.approx(beforeE)
    assert {x['id'] for i in nodes[0] for x in p['node_parts'][str(i)]}=={'2','3','4'}
    assert report['recycled_mass']==5 and report['created_mass']==4
    assert M[nodes[1]].sum()==pytest.approx(4)
    assert all(x['composition']==COMPOSITION for row in p['node_parts'].values() for x in row)
    with pytest.raises(ValueError,match='duplication'):transport(p,M,E,np.array([0,0]),np.ones(2),2.)


@pytest.fixture
def late():
    p=Path('/workspace/genesis-moving-basal-pilot-01/checkpoint.json')
    if not p.exists():pytest.skip('Authenticated local source only')
    return SimplifiedGenesis.load(p)


def handoff(m):
    side=m.mesh.centroids[:,2]>0;mask=np.array([side[f]!=side[g] for f,g,_,_ in m.mesh.shared_edges])
    return ThermalGeologicalContinuation.from_source(m,experimental_nodal_retirement=True,fracture_edges=mask,experimental_parcel_mapping=True)


def test_real_late_profile_import_and_native_restart(late,tmp_path):
    before=encoded(dict(config=late.config,state=late.state));g=handoff(late)
    assert g.record['eligibility']['mapped_solid_node_count']==240
    np.testing.assert_array_equal(g.record['thermal_inventory']['mass_kg'],late.config['mass'])
    np.testing.assert_array_equal(g.record['thermal_inventory']['energy_j'],late.state['energy'])
    assert g.record['mapping']['energy']['mapping_heat_j']==0
    g.step(.25e-6);g.save(tmp_path/'restart');h=ThermalGeologicalContinuation.load(tmp_path/'restart')
    a=g.step(.25e-6);b=h.step(.25e-6);assert encoded(g.record)==encoded(h.record)
    assert abs(a['thermal']['total_mass_residual_kg'])<=a['thermal']['mass_roundoff_allowance_kg']
    assert a['thermal']['mechanical_heat_j']==0
    assert encoded(dict(config=late.config,state=late.state))==before


def test_unequal_profile_counts_and_guard(late):
    # Manufactured admission mask, not a physical evolution or new source.
    late.state['born_material']=[p for p in late.state['born_material'] if p['face']%2]
    nodes=layout(late);assert set(map(len,nodes))=={2,3}
    g=handoff(late);assert g.record['eligibility']['mapped_solid_node_count']==200
    original=deepcopy(late.state['born_material'][0]);late.state['born_material'].append(original)
    with pytest.raises(ValueError,match='PARCEL_BIRTH_ID'):layout(late)


def test_default_still_requires_explicit_parcel_mode(late):
    result=eligibility(late,experimental_nodal_retirement=True,partition_owner=(late.mesh.centroids[:,2]>0).astype(int))
    assert any('newborn' in r for r in result['reasons'])


def test_additional_newborn_layer_and_transaction_rollback(late,monkeypatch):
    admitted={p['cell_id'] for p in late.state['born_material']}
    cell=next(c for c in late.config['phase_cells'] if c['face']==0 and c['id'] not in admitted)
    node=cell['node'];late.state['energy'][node]=late.config['mass'][node]*late.config['cp'][node]*(late.config['solidus']-1)
    late.state['born_material'].append(dict(cell_id=cell['id'],node=node,face=0,mass_kg=cell['mass_kg'],volume_m3=cell['volume_m3'],inertia=cell['inertia'],memory=[0.]*6,birth_time_myr=late.state['time_myr'],birth_enthalpy_j=late.state['energy'][node],specific_enthalpy_j_kg=late.state['energy'][node]/cell['mass_kg']))
    late.validate();g=handoff(late);assert len(g.record['material_parcels']['nodes'][0])==4
    before=encoded(g.record)
    def fail(*args,**kwargs):raise ValueError('manufactured port rejection')
    monkeypatch.setattr('tectonics.genesis_geological_thermal.apply_ports',fail)
    with pytest.raises(ValueError,match='port rejection'):g.step(.25e-6)
    assert encoded(g.record)==before


def test_native_transport_and_creation_use_parcel_branch():
    from analysis.genesis_geological_handoff.thermal_experiment import thermal_source
    m,_=thermal_source();mantle=np.zeros((m.nf,3));mantle[:,2]=.04*m.mesh.centroids[:,0]
    g=ThermalGeologicalContinuation.from_source(m,mantle_omega=mantle,experimental_parcel_mapping=True)
    rows=[g.step(10.) for _ in range(3)]  # Manufactured native fixture, not planetary evolution.
    assert sum(r['transported_winning_cells'] for r in rows)>0
    assert sum(r['thermal']['created_mass_kg'] for r in rows)>0
    assert sum(r['thermal']['recycled_mass_kg'] for r in rows)>0
    for r in rows:assert abs(r['thermal']['global_residual_j'])<=r['thermal']['global_allowance_j']
