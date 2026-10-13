from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
from tectonics.simplified_genesis_basal import enable,facet,exchange,geometry,assemble
from tectonics.simplified_genesis import SimplifiedGenesis
from tectonics.genesis_transfer_bridge import encoded
from analysis.genesis_aggregate_coupled.core import step


@pytest.fixture
def model():
    path=Path('/workspace/genesis-nodal-source-01.json')
    if not path.is_file():pytest.skip('Authenticated local nodal source required; no retrieval')
    return enable(SimplifiedGenesis.load(path))


def test_common_motion_has_no_drag_and_relative_work_closes():
    tri=np.array([[1.,0.,0.],[1.,1.,0.],[1.,0.,1.]])
    vm=np.tile([0.,1e-8,0.],(3,1));face=facet(tri,-.1,1e14,np.zeros((3,3)),vm)
    v=np.column_stack((vm,np.zeros((3,3)))).ravel()
    common=exchange(face,v,2.)
    assert common['heat_j']==0 and common['crust_drag_work_j']==0
    np.testing.assert_allclose(face['Fm']-face['D']@v,0.,atol=1e-9)
    r=exchange(face,2*v,2.)
    assert r['heat_j']>0 and r['crust_drag_work_j']<0 and r['mantle_drag_work_j']>0
    assert abs(r['heat_j']+r['crust_drag_work_j']+r['mantle_drag_work_j'])<1e-15
    block=dict(H=np.eye(1),L=np.zeros((1,18)),R=np.zeros((1,1)),old=np.zeros(1),theta=np.zeros(1),dofs=np.arange(18))
    result=step(np.eye(18)*1e15,face['D'],face['Fm'],2*v,2.,[block])
    q=exchange(face,result['velocity'],2.);ledger=result['ledger']
    residual=ledger['kinetic_after']-ledger['kinetic_before']+ledger['numerical_kinetic']+q['heat_j']+q['mantle_drag_work_j']
    assert abs(residual)<1e-15


def test_growth_moves_support_without_front_velocity_or_impulse(model):
    m=model;before=geometry(m);s=deepcopy(m.state);cell=m.config['phase_cells'][0]
    s['born_material']=[dict(cell_id=cell['id'],face=cell['face'],node=cell['node'])]
    after=geometry(m,s);f=cell['face']
    assert after['depth'][f]>before['depth'][f] and after['thermal_nodes'][f]==cell['node']
    assert after['area'][f]<before['area'][f]
    face=facet(after['bottom'][f],after['h'][f],1e14,np.zeros((3,3)),np.zeros((3,3)))
    result=exchange(face,np.zeros(18),25*31557600.)
    assert result['heat_j']==0
    np.testing.assert_array_equal(result['mantle_reaction_impulse_ns'],0.)


def test_shrink_rotation_normals_area_and_material_velocity(model):
    from scipy.spatial.transform import Rotation
    m=model;before=geometry(m);s=deepcopy(m.state);X=np.asarray(m.config['nodal']['X'])
    # Include the accepted source displacement in the common affine transform.
    U=np.asarray(s['nodal_displacement']).reshape(-1,6);R=Rotation.from_rotvec([.001,.002,-.001]).as_matrix();scale=.999
    a=(X+U[:,:3])@R.T*scale-X;p=(X+U[:,3:])@R.T*scale-X
    s['nodal_displacement']=np.column_stack((a,p)).ravel().tolist();after=geometry(m,s)
    np.testing.assert_allclose(after['area'],before['area']*scale**2,rtol=2e-15)
    np.testing.assert_allclose(after['normal'],before['normal']@R.T,atol=3e-15)
    np.testing.assert_allclose(after['depth'],before['depth']*scale,rtol=3e-11)
    f=0;vel=np.column_stack((X[m.mesh.faces[f]]*1e-15,X[m.mesh.faces[f]]*1e-15))
    vm=vel[:,:3]+after['h'][f]*vel[:,3:]
    face=facet(after['bottom'][f],after['h'][f],1e14,np.zeros((3,3)),vm)
    assert exchange(face,vel.ravel(),1.)['heat_j']==0


def test_opt_in_preserves_state_and_restart_balance(model,tmp_path):
    source=SimplifiedGenesis.load('/workspace/genesis-nodal-source-01.json')
    assert encoded(source.state)==encoded(model.state)
    before=model.mass.copy();model.advance(1.)
    B=model.state['last_ledger']['basal'];L=model.state['last_ledger']
    assert B['beta_pa_s_m']==1e14 and B['mantle_drag_work_j']==0
    assert not B['phase_front_velocity_used_as_mantle_velocity']
    assert abs(L['whole_balance_j'])<=L['whole_allowance_j']
    model.save(tmp_path/'state.json');restored=SimplifiedGenesis.load(tmp_path/'state.json')
    model.advance(1.);restored.advance(1.)
    assert encoded(model.state)==encoded(restored.state)
    np.testing.assert_array_equal(model.mass,before)
    assert assemble(model)['geometry']['depth'].min()>0
