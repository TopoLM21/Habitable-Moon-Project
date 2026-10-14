from types import SimpleNamespace
from copy import deepcopy
import math
import numpy as np
import pytest
from tectonics.geological_material import transport_cycle,CONVERSION as C
from tectonics.experimental_geological_parcels import initialize

def state(b,c):return SimpleNamespace(oceanic_volume_km3=np.array(b,dtype=float),continental_volume_km3=np.array(c,dtype=float))

def test_actual_retired_H_and_arc_donor_H_with_provenance():
    old=state([2,3],[0,0]);lith=state([2,7],[0,0]);final=state([2,0],[0,26])
    mass=np.array([1,1,1.5,1.5,1e6,1e3])*C
    energy=mass*np.array([1e6,2e6,3e6,4e6,5e6,1.26e6]);M=math.fsum(mass);H=math.fsum(energy)
    parcel=initialize(SimpleNamespace(config={'mass':mass.tolist()},state={'energy':energy.tolist()}),[[0,1],[2,3]])
    trace=dict(winner_source=np.array([0,-1]),rigid_transport={0:dict(source=np.array([0]),target=np.array([0])),1:dict(source=np.array([1]),target=np.array([0]))})
    cycle=dict(stages=[('initial',lith.oceanic_volume_km3,lith.continental_volume_km3),('juvenile_arc',final.oceanic_volume_km3,final.continental_volume_km3)])
    r=transport_cycle(parcel,mass,energy,old,lith,final,trace,cycle,1.)
    assert r['created_mass']==pytest.approx(33*C)
    assert r['recycled_mass']==pytest.approx(10*C)
    assert r['recycled_H']==pytest.approx((3*3.5e6+7*5e6)*C)
    assert math.fsum(mass)==pytest.approx(M,rel=1e-15)
    assert math.fsum(energy)==pytest.approx(H,rel=1e-15)
    parts=[p for i in (2,3) for p in parcel['node_parts'][str(i)]]
    assert all(p['material_phase']=='felsic' and p['process']=='juvenile_arc' for p in parts)
    assert sum(p['mass_kg'] for p in parts)==pytest.approx(26*C)

def test_multi_source_felsic_spreading_keeps_both_original_identities():
    old=state([2,3],[1,2]);final=state([2,7],[2,1]);mass=np.array([2,1,3,2,1e6,1e3])*C;energy=mass*np.arange(1,7)*1e6
    parcel=initialize(SimpleNamespace(config={'mass':mass.tolist()},state={'energy':energy.tolist()}),[[0,1],[2,3]])
    for i in (0,2):parcel['node_parts'][str(i)][0]['material_phase']='mafic'
    for i in (1,3):parcel['node_parts'][str(i)][0]['material_phase']='felsic'
    H=math.fsum(energy)
    trace=dict(winner_source=np.array([0,-1]),rigid_transport={0:dict(source=np.array([0]),target=np.array([0])),1:dict(source=np.array([1]),target=np.array([0]))},continental_spreading=[(0,1,1.)])
    cycle=dict(stages=[('initial',final.oceanic_volume_km3,final.continental_volume_km3)])
    transport_cycle(parcel,mass,energy,old,final,final,trace,cycle,1.)
    for f in range(2):
        parts=[p for i in parcel['nodes'][f] for p in parcel['node_parts'][str(i)] if p['material_phase']=='felsic']
        assert {p['id'] for p in parts}=={'source-node-1','source-node-3'}
    assert math.fsum(energy)==pytest.approx(H,rel=1e-15)

def test_sub_ulp_fragment_survives_mass_coordinate_mapping():
    from tectonics.experimental_geological_parcels import remap
    parts=[dict(id='large',mass_kg=1e20,energy_j=2e26),dict(id='tiny',mass_kg=1e-10,energy_j=3e-4),dict(id='tail',mass_kg=1e20,energy_j=4e26)]
    out=remap(parts,[.5,.5]);tiny=[p for row in out for p in row if p['id']=='tiny']
    assert math.fsum(p['mass_kg'] for p in tiny)==1e-10
    assert math.fsum(p['energy_j'] for p in tiny)==3e-4
    assert tiny[0]['mass_coordinate_point_assignment']

def test_node_coalescing_preserves_identity_composition_and_enthalpy():
    from tectonics.geological_material import coalesce_node
    parts=[dict(id='a',material_phase='mafic',mass_kg=2.,energy_j=3.),dict(id='b',material_phase='felsic',mass_kg=4.,energy_j=5.),dict(id='a',material_phase='mafic',mass_kg=7.,energy_j=11.)]
    out=coalesce_node(parts)
    assert len(out)==2 and out[0]['mass_kg']==9 and out[0]['energy_j']==14
    assert out[1]==parts[1]
