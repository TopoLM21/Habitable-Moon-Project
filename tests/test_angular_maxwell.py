"""Full angular history: material, thermal, rotation, damage and disk restart."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
import pytest
from scipy import sparse
from analysis.genesis_geological_handoff.angular_maxwell import AngularMaxwell, maxwell_update, SCHEMA
from analysis.genesis_geological_handoff.edge_spatial_audit import AuditGrid, ROOT
from analysis.genesis_geological_handoff.integrated_vector_operator import first_step
from analysis.genesis_geological_handoff.base_operator_audit import rotation
from tectonics.simplified_genesis import SimplifiedGenesis


@pytest.fixture
def grid():
    source=Path('/workspace/genesis-moving-basal-source-01.json')
    if not source.exists():pytest.skip('Local authenticated fixture source required; no fetch')
    coeff=np.array(json.loads((ROOT/'edge_spatial_control.json').read_text())['coefficients'])
    return AuditGrid(1,False,coeff)


def model(grid, feedback=False):
    alpha=SimplifiedGenesis.load('/workspace/genesis-moving-basal-source-01.json').shell['linear_expansion_per_k']
    return AngularMaxwell(grid,damage_feedback=feedback,expansion_per_k=alpha)


def test_first_step_matches_integrated_operator_and_is_transactional(grid):
    m=model(grid);state=m.initial();before=deepcopy(state)
    new,row=m.step(state,.5)
    stress=np.einsum('fij,fqj->fqi',m.Cbase,np.array(new['memory']).reshape(m.nf,6,6))
    expected,_=first_step(grid)
    np.testing.assert_allclose(m.world_stress(stress),expected,atol=1e-5,rtol=1e-10)
    assert state==before and row['balance_relative']<5e-8
    with pytest.raises(ValueError):m.step(state,-1)
    assert state==before


def test_material_point_maxwell_exact_be_and_positive_dissipation():
    old=np.arange(1,7)*1e-6;rate=6e10/1e24;dt=1/rate
    new=maxwell_update(old,np.zeros(6),np.zeros(6),dt,rate)
    np.testing.assert_array_equal(new,old/2)
    H=np.diag([3.,3.,3.,1.,1.,1.]);de=new-old
    elastic_change=.5*new@H@new-.5*old@H@old
    diss=dt*rate*new@H@new;numerical=.5*de@H@de
    assert diss>0 and numerical>0
    assert abs(elastic_change+diss+numerical)<1e-25


def test_clamped_thermal_load_and_relaxation_preserve_all_angular_samples(grid):
    grid.a.free=np.array([],dtype=int);grid.a.force[:]=0.
    m=model(grid)
    memory=np.ones((m.nf,2,3,6))*1e-6
    memory[:,:,1,:]*=2;memory[:,:,2,:]*=3
    state=m.initial(memory=memory,description='known three-valued angular patch')
    delta=np.broadcast_to(np.array([.01,-.02,.03]),(m.nf,2,3))
    new,row=m.step(state,.5,temperature_increment_k=delta)
    theta=m.alpha*delta[...,None]*np.array([1,1,1,0,0,0])
    expected=(memory-theta)/(1+.5*31557600*m.rate)
    np.testing.assert_allclose(new['memory'],expected,rtol=1e-15)
    L=new['last_ledger']
    stress=np.einsum('fij,frqj->frqi',m.Cbase,expected)
    thermal=-np.einsum('frqi,frqi,frq->',stress,theta,m.w.reshape(m.nf,2,3))
    assert L['thermal_work_j']==pytest.approx(thermal,rel=1e-13)
    assert L['maxwell_j']>0 and L['numerical_memory_j']>0
    assert L['fracture_heat_j']==0 and row['balance_relative']<1e-12


def test_checkpoint_version_fingerprint_restart_and_no_overwrite(grid,tmp_path):
    m=model(grid,True);state=m.initial()
    for _ in range(3):state,_=m.step(state,.5,temperature_increment_k=.01)
    path=tmp_path/'new.json';m.save(state,path)
    loaded=model(grid,True).load(path)
    continued,row=m.step(state,.5);restarted,row2=m.step(loaded,.5)
    assert continued==restarted and row==row2
    with pytest.raises(FileExistsError):m.save(state,path)
    with pytest.raises(ValueError,match='OPERATOR_MISMATCH'):model(grid,False).load(path)
    with pytest.raises(ValueError,match='HISTORY_REQUIRED'):m.validate({'schema':'old'})


def test_lift_is_explicit_and_cannot_recover_angular_variance(grid):
    m=model(grid)
    with pytest.raises(ValueError,match='Describe'):
        m.initial(memory=np.zeros((m.nf,2,3,6)))
    reduced=np.full((m.nf,2,6),1e-6)
    with pytest.raises(ValueError,match='NOT_IDENTIFIABLE'):m.lift_reduced_fixture(reduced,assumption='exact')
    lifted=m.lift_reduced_fixture(reduced,assumption='angular_uniform_zero_variance')
    np.testing.assert_allclose(np.array(lifted['memory']).mean(axis=2),reduced,rtol=1e-15)
    assert abs(lifted['initialization']['modeled_energy_difference_j'])/lifted['initial_elastic_j']<1e-13
    alternate=np.array(lifted['memory']);alternate[:,:,0,3]+=1e-5;alternate[:,:,1,3]-=1e-5
    np.testing.assert_allclose(alternate.mean(axis=2),reduced,atol=1e-20)
    assert m.energy(alternate,lifted['edge_damage'])>lifted['initial_elastic_j']


def test_damage_release_is_single_reserve_transfer_without_heat(grid):
    m=model(grid,True);s=m.initial();initial=m.energy(s['memory'],s['edge_damage'])
    for _ in range(8):
        before=s;s,row=m.step(s,.5)
        L=s['last_ledger'];difference=s['fracture_reserve_j']-before['fracture_reserve_j']
        assert difference==pytest.approx(L['fracture_release_j'],rel=1e-13)
        assert L['fracture_heat_j']==0 and difference>=0
        assert np.all(np.array(s['edge_damage'])>=before['edge_damage'])
        assert m.energy(s['memory'],s['edge_damage'])==pytest.approx(row['elastic_j'],rel=1e-13)
    assert s['fracture_reserve_j']>0
    c=s['cumulative'];residual=(row['elastic_j']-initial+row['kinetic_j']+s['fracture_reserve_j']
        +c['maxwell_j']+c['numerical_memory_j']+c['numerical_kinetic_j']-c['external_work_j']-c['thermal_work_j'])
    assert abs(residual)/max(abs(c['external_work_j']),1)<5e-8


def test_saturated_damage_cannot_release_twice(grid):
    grid.a.free=np.array([],dtype=int);grid.a.force[:]=0.
    m=model(grid,True);s=m.initial(memory=np.full((m.nf,2,3,6),1e-4),description='known strained saturated patch')
    s['edge_damage']=np.ones(len(m.mesh.shared_edges)).tolist()
    for _ in range(2):
        s,_=m.step(s,.5)
        assert s['last_ledger']['fracture_release_j']==0
        assert s['fracture_reserve_j']==0 and s['last_ledger']['fracture_heat_j']==0


def test_rotated_multistep_mechanics_and_memory_are_covariant(grid):
    grid.a.free=np.arange(grid.a.nd)
    rotated=deepcopy(grid);Q=rotation()
    transform=sparse.kron(sparse.eye(grid.a.nd//3),Q,format='csr')
    rotated.mesh=replace(grid.mesh,vertices=grid.mesh.vertices@Q.T,centroids=grid.mesh.centroids@Q.T)
    rotated.a.mesh=rotated.mesh
    rotated.frames=np.einsum('ij,fjk->fik',Q,grid.frames)
    rotated.a.M=(transform@grid.a.M@transform.T).tocsr();rotated.a.force=transform@grid.a.force
    a,b=model(grid),model(rotated);s,t=a.initial(),b.initial()
    for _ in range(3):s,_=a.step(s,.5);t,_=b.step(t,.5)
    # Local frames rotate with the material, so strain components are invariant.
    x,y=np.array(s['memory']),np.array(t['memory'])
    assert np.linalg.norm(x-y)/np.linalg.norm(x)<1e-6
    assert a.energy(x,s['edge_damage'])==pytest.approx(b.energy(y,t['edge_damage']),rel=1e-6)
