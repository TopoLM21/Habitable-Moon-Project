from copy import deepcopy
import numpy as np
import pytest
from tectonics.simplified_genesis import SimplifiedGenesis
from tectonics.simplified_genesis_phase import solid_inventory,subdivide_uniform_cell
from tectonics.genesis_transfer_bridge import encoded
from analysis.simplified_genesis_validation.synthetic import synthetic


from analysis.simplified_genesis_validation.phase_synthetic import freezing_model


def test_freezing_conserves_inventory_and_has_no_latent_heat_double_count(tmp_path):
    model,node=freezing_model();mass0=sum(model.mass);E0=np.array(model.state['energy']);I0=model.material_inertia().sum(axis=0)
    model.advance(25.);s=model.state;phase=s['last_ledger']['phase']
    assert phase['new_load_bearing_cells']==1 and phase['new_load_bearing_kg']==3e12
    assert np.isclose(phase['new_phase_solid_kg'],3e12*10/600,rtol=1e-10)
    assert phase['added_latent_heat_j']==0 and sum(model.mass)==mass0
    assert s['born_material'][0]['memory']==[0.]*6 and s['owner']==[0]*model.nf and s['events']==[]
    assert model.material_inertia().sum()>I0.sum()
    T=model.temperature(s['energy']);base_sigma=model.constitutive(T,np.array(s['damage']),np.array(s['memory']),T,0.)[1]
    expected=base_sigma[0]*model.G[0,0,0]/(model.G[0,0,0]+s['born_material'][0]['volume_m3'])
    np.testing.assert_allclose(s['stress_pa'][0],expected,rtol=1e-12,atol=1e-8)
    assert phase['capture_angular_momentum_relative']<1e-12
    assert abs(s['last_ledger']['whole_balance_j'])<=s['last_ledger']['whole_allowance_j']
    model.save(tmp_path/'born.json');copy=SimplifiedGenesis.load(tmp_path/'born.json')
    model.advance(25.);copy.advance(25.);assert encoded(model.state)==encoded(copy.state)
    assert len(model.state['born_material'])==1 and model.state['last_ledger']['phase']['new_load_bearing_cells']==0
    assert np.linalg.norm(model.state['born_material'][0]['memory'])>0


def test_repartition_of_same_enthalpy_does_not_birth_material():
    m=3e12;h=1200*1600+4e5/3
    cell=dict(id='original',face=0,layer=0,q_lo=.1,q_hi=.2,mass_kg=m,energy_j=m*h,cp=1200.,latent=4e5)
    parts=subdivide_uniform_cell(cell,.37)
    assert np.isclose(sum(x['energy_j'] for x in parts),cell['energy_j'],rtol=1e-15)
    assert np.isclose(sum(x['mass_kg'] for x in parts),m,rtol=1e-15)
    assert np.isclose(solid_inventory(parts,1400,2000),solid_inventory([cell],1400,2000),rtol=2e-15)
    assert {x['id'] for x in parts}=={'original'}


def test_unsupported_remelt_rolls_back():
    model,node=freezing_model();model.advance(25.)
    model.config['synthetic_power_w'][node]*=-100
    before=encoded(model.state)
    with pytest.raises(ValueError,match='REMELT'):model.advance(25.)
    assert encoded(model.state)==before


def test_growth_across_existing_fracture_stops_before_commit():
    model,node=freezing_model();model.state['consumed'][0]=True;before=encoded(model.state)
    with pytest.raises(ValueError,match='ACROSS_FRACTURE'):model.advance(25.)
    assert encoded(model.state)==before


def test_mass_capture_preserves_nonzero_angular_momentum_and_pays_heat():
    model,node=freezing_model();model.face_drag[:]=0.;model.face_torque[:]=0.
    model.config['face_drag']=model.face_drag.tolist();model.config['face_torque']=model.face_torque.tolist()
    oldL=model.material_inertia().sum(axis=0)@np.array(model.state['omega'][0]);model.advance(25.)
    newL=model.material_inertia().sum(axis=0)@np.array(model.state['omega'][0])
    np.testing.assert_allclose(newL,oldL,rtol=1e-14,atol=1.)
    assert model.state['last_ledger']['phase']['capture_heat_j']>1e-6


def test_legacy_fixed_catalog_cannot_silently_admit_new_material():
    model,node=freezing_model();model.config.pop('phase_cells');before=encoded(model.state)
    with pytest.raises(ValueError,match='UNMAPPED_SOLIDIFICATION'):model.advance(25.)
    assert encoded(model.state)==before
