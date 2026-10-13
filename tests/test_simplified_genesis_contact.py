import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from tectonics.simplified_genesis_contact import (unilateral_projection,closed_partition_step,swept_certificate,placement_certificate,ContactPathUnsupported)
from tectonics.simplified_genesis import SimplifiedGenesis
from tectonics.genesis_transfer_bridge import encoded
from tectonics.spherical_polygons import intersect_convex,polygon_area
from analysis.simplified_genesis_validation.synthetic import synthetic,hemisphere_synthetic


def test_unilateral_opening_closure_and_tangential_slide():
    B=np.eye(4);J=np.array([[-1.,0.,1.,0.]])
    for rhs in [np.array([-1.,3.,1.,-2.]),np.array([0.,3.,0.,-2.])]:
        v,lam=unilateral_projection(B,rhs,J)
        np.testing.assert_array_equal(v,rhs);assert lam[0]==0
    v,lam=unilateral_projection(B,np.array([1.,3.,-1.,-2.]),J)
    np.testing.assert_allclose(v,[0.,3.,0.,-2.],atol=1e-14);assert lam[0]>0
    assert abs(lam@(J@v))<1e-14


def test_original_jagged_split_locks_without_changing_overlap_metric():
    m=synthetic();m.advance(25.);m.advance(25.)
    assert len(m.state['omega'])==2 and m.state['contact']['accepted_relative_omega_norm']==0
    assert m.state['contact']['free_relative_omega_norm']>1e-16
    assert any(c['normal_impulse_n_s']>0 for c in m.state['contact']['normal_impulses'])
    X=m.mesh.vertices;F=m.mesh.faces;owner=np.array(m.state['owner']);R=np.array(m.state['rotation'])
    a,b=122,123;inter=intersect_convex(X[F[a]]@R[owner[a]].T,X[F[b]]@R[owner[b]].T)
    assert len(inter)==0 or polygon_area(inter)==0
    assert m.state['contact']['whole_step']['all_domain_pairs']


def test_birth_two_domains_free_sliding_and_restart(tmp_path):
    m=hemisphere_synthetic();assert len(m.state['omega'])==1 and max(m.state['damage'])==0
    m.advance(25.);assert len(m.state['omega'])==2 and len(m.state['events'])==1
    m.advance(25.);assert np.linalg.norm(np.diff(m.state['omega'],axis=0))>1e-16
    proof=m.state['contact']['whole_step'];assert proof['kind']=='invariant_opposite_hemispheres_all_pairs_all_time'
    m.save(tmp_path/'state.json');copy=SimplifiedGenesis.load(tmp_path/'state.json');m.advance(25.);copy.advance(25.)
    assert encoded(m.state)==encoded(copy.state)
    owner=np.array(m.state['owner']);R=np.array(m.state['rotation']);X=m.mesh.vertices;F=m.mesh.faces
    for a,b,_,_ in m.mesh.shared_edges:
        if owner[a]==owner[b]:continue
        overlap=intersect_convex(X[F[a]]@R[owner[a]].T,X[F[b]]@R[owner[b]].T)
        assert len(overlap)==0 or polygon_area(overlap)==0
    assert abs(m.state['last_ledger']['whole_balance_j'])<=m.state['last_ledger']['whole_allowance_j']


def test_endpoints_do_not_authorize_between_endpoint_collision():
    m=hemisphere_synthetic();m.advance(25.);owner=np.array(m.state['owner']);poses=np.repeat(np.eye(3)[None],2,axis=0)
    # A whole revolution returns to disjoint endpoints; halfway the two
    # hemispheres occupy the same side. The path must be refused.
    w=np.array([[2*np.pi,0.,0.],[0.,0.,0.]])
    end=np.array([Rotation.from_rotvec(x).as_matrix() for x in w]);placement_certificate(m.mesh,owner,end)
    with pytest.raises(ContactPathUnsupported,match='NONCOMMUTING'):swept_certificate(m.mesh,owner,poses,w,1.)


def test_multiple_domains_joint_normal_reactions_conserve_momentum():
    m=hemisphere_synthetic();mesh=m.mesh
    # Three spherical lunes: manufactured simultaneous-contact test, no
    # assumption that independent pair projections compose correctly.
    angle=np.mod(np.arctan2(mesh.centroids[:,1],mesh.centroids[:,0]),2*np.pi);owner=np.minimum((angle/(2*np.pi/3)).astype(int),2)
    I=np.repeat(np.eye(3)[None],3,axis=0);D=.1*I;tau=np.array([[1.,2.,0.],[-2.,0.,1.],[0.,-1.,0.]])
    w,R,L,p=closed_partition_step(mesh,owner,np.repeat(np.eye(3)[None],3,axis=0),I,D,tau,np.zeros((3,3)),.1,1.)
    np.testing.assert_allclose(p['net_internal_angular_impulse'],0.,atol=1e-12)
    assert min(x['normal_impulse_n_s'] for x in p['normal_impulses'])>=0
    assert L['momentum_relative']<1e-10 and abs(L['rotation_balance'])<1e-12
    assert np.array_equal(w,np.repeat(w[:1],3,axis=0))


def test_saved_overlapping_placement_is_not_repaired_by_relabeling():
    m=synthetic();m.advance(25.);s=m.state
    s['rotation']=[Rotation.from_rotvec([0,0,1e-5]).as_matrix().tolist(),Rotation.from_rotvec([0,0,-1e-5]).as_matrix().tolist()]
    before=encoded(s)
    with pytest.raises(ContactPathUnsupported,match='INITIAL_PLACEMENT'):m.advance(25.)
    assert encoded(m.state)==before


def test_inelastic_contact_impulse_has_explicit_energy_sink():
    m=synthetic();m.advance(25.);owner=np.array(m.state['owner']);I=np.repeat(np.eye(3)[None],2,axis=0)
    old=np.array([[0.,0.,1.],[0.,0.,-1.]])
    w,R,L,p=closed_partition_step(m.mesh,owner,np.repeat(np.eye(3)[None],2,axis=0),I,0*I,np.zeros((2,3)),old,.01,1.)
    np.testing.assert_allclose(w,0.,atol=1e-14)
    assert np.isclose(L['numerical_rotation'],L['kinetic_before'])
    assert abs(p['contact_work_j'])<1e-14 and abs(L['rotation_balance'])<1e-14
    total=np.zeros(3)
    for item in p['normal_impulses']:
        impulse=np.array(item['normal_a_to_b'])*item['normal_impulse_n_s']
        total-=impulse;total+=impulse
    np.testing.assert_array_equal(total,0.)
