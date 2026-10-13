"""Source invariants, minimum-variance initialization and independent restart."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import pytest
from tectonics.genesis_transfer_bridge import encoded,digest
from analysis.genesis_geological_handoff.source_angular import SourceAngular


@pytest.fixture(scope='module')
def model():
    path=Path('/workspace/genesis-moving-basal-source-01.json')
    if not path.exists():pytest.skip('Authenticated local source only')
    return SourceAngular(json.loads(path.read_bytes()))


def test_primary_preserves_radial_moments_and_inventory(model):
    s=model.initial();m=model.m;e=np.array(s['memory']).mean(axis=1).reshape(m.nf,2,6)
    np.testing.assert_allclose(np.einsum('fij,fjk->fik',m.G,e),np.einsum('fij,fjk->fik',m.G,np.array(m.state['memory'])),rtol=1e-13,atol=1e-8)
    np.testing.assert_array_equal(s['velocity'],m.state['nodal_velocity'])
    np.testing.assert_array_equal(s['displacement'],m.state['nodal_displacement'])
    assert digest(encoded(dict(config=m.config,state=m.state)))==model.envelope['sha256']
    assert len(m.config['source_material_catalogue'])==800
    assert sum(x['cohorts'] for x in model.projection_audit['supports'])==800
    for support in model.projection_audit['supports']:
        f=support['face']
        assert support['volume_m3']==pytest.approx(m.G[f,0,0],rel=1e-13)
        assert support['solid_nodes']==m.solid_nodes[f].tolist()
        assert m.mass[m.solid_nodes[f]].sum()==pytest.approx(support['volume_m3']*m.config['nodal']['density'],rel=1e-13)
    assert abs(s['mapping']['elastic_relative_change'])<1e-10
    assert s['mapping']['force_relative_change']<1e-10
    assert s['mapping']['mapping_heat_j']==0 and s['mapping']['generalized_momentum_change']==0
    # Independent stress reconstruction through source thermal radial nodes,
    # rather than reusing the new moment-mean implementation.
    sampled=np.einsum('fri,fij->frj',m.P,np.asarray(m.state['memory']))
    source_sigma=np.einsum('fr,fij,frj->fi',m.weights,m.C*m.scale(m.state['damage'])[:,None,None],sampled)/m.weights.sum(axis=1)[:,None]
    source_tensor=np.zeros((m.nf,3,3));source_tensor[:,range(3),range(3)]=source_sigma[:,:3]
    for k,(i,j) in enumerate(((0,1),(0,2),(1,2)),3):source_tensor[:,i,j]=source_tensor[:,j,i]=source_sigma[:,k]
    expected=np.einsum('fai,fij,fbj->fab',model.frames,source_tensor,model.frames)
    actual=model.energy_force_stress(s['memory'],s['edge_damage'])[2]
    assert np.linalg.norm(actual-expected)/np.linalg.norm(expected)<1e-12


def test_alternative_is_zero_mean_and_accounts_for_extra_energy(model):
    primary=model.initial();other=model.initial(True)
    delta=np.array(other['memory'])-np.array(primary['memory'])
    np.testing.assert_allclose(delta.mean(axis=1),0,atol=1e-19)
    added=model.energy_force_stress(delta,primary['edge_damage'])[0]
    assert other['mapping']['signed_energy_difference_j']==pytest.approx(added,rel=1e-10)
    assert 0<other['mapping']['elastic_relative_change']<=.01+1e-12
    assert other['heat_buffer_j']==0 and other['initialization']['mapping_heat_j']==0


def test_full_source_immutable_restart_and_heat_buffer(model,tmp_path):
    archived=encoded(model.envelope);body=encoded(dict(config=model.m.config,state=model.m.state))
    s=model.initial();before=deepcopy(s)
    s,row=model.step(s,.0625)
    assert before==model.initial()
    path=tmp_path/'checkpoint.json';model.save(s,path)
    restored,loaded=SourceAngular.load(path)
    a,ra=model.step(s,.0625);b,rb=restored.step(loaded,.0625)
    assert encoded(a)==encoded(b) and ra==rb
    assert encoded(model.envelope)==archived and encoded(dict(config=model.m.config,state=model.m.state))==body
    assert model.m.state['energy']==model.envelope['payload']['state']['energy']
    assert model.m.state['damage']==model.envelope['payload']['state']['damage']
    assert a['heat_buffer_j']==pytest.approx(a['cumulative']['maxwell']+a['cumulative']['drag'],rel=1e-13)
    assert a['reserve_j']>=s['reserve_j'] and row['global_energy_relative']<5e-8
    with pytest.raises(FileExistsError):model.save(s,path)
    envelope=json.loads(path.read_bytes());envelope['payload']['state']['memory'][0][0][0]+=1
    path.write_text(json.dumps(envelope))
    with pytest.raises(ValueError,match='CHECKSUM'):SourceAngular.load(path)


def test_newborn_is_rejected_before_partial_mapping(model):
    envelope=deepcopy(model.envelope)
    # A real late snapshot exercises actual catalogue, masses and birth records.
    late=Path('/workspace/genesis-moving-basal-pilot-01/checkpoint.json')
    if not late.exists():pytest.skip('Local late source not available')
    with pytest.raises(ValueError,match='NEWBORN_ANGULAR_MAPPING_NOT_IMPLEMENTED'):
        SourceAngular(json.loads(late.read_bytes()))
