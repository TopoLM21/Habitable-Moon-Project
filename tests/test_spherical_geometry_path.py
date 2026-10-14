"""Spherical affine completeness, physical work, positive mass and restart."""
from pathlib import Path
import numpy as np
import pytest
from tectonics.genesis_transfer_bridge import encoded
from analysis.genesis_geological_handoff.spherical_geometry_path import SphericalPath,evaluation


def test_spherical_map_translation_rotation_and_affine_strain():
    tri=np.array([[0.,0.,1.],[.4,0.,1.],[0.,.3,1.]])
    tri=tri/np.linalg.norm(tri,axis=1)[:,None]*5e6
    N=np.array([[.2,.3,.5],[.6,.1,.3]])
    S,L,lam=evaluation(tri,N,5e6-70)
    B=np.array([[1.,.2,-.3],[.4,-.8,.7],[.1,.5,.6]])*1e-9
    velocity=np.tile((tri@B.T)[:,None,:],(1,2,1)).ravel()
    x=lam[:,None]*(N@tri)
    np.testing.assert_allclose(np.einsum('qij,j->qi',S,velocity),x@B.T,rtol=1e-14)
    expected=np.array([B[0,0],B[1,1],B[2,2],B[0,1]+B[1,0],B[0,2]+B[2,0],B[1,2]+B[2,1]])
    np.testing.assert_allclose(np.einsum('qij,j->qi',L,velocity),np.broadcast_to(expected,(2,6)),rtol=1e-13,atol=1e-22)
    trans=np.zeros((3,6));trans[:,:3]=[1.,2.,3.]
    np.testing.assert_allclose(np.einsum('qij,j->qi',L,trans.ravel()),0,atol=1e-21)


@pytest.fixture(scope='module')
def model():
    if not Path('/workspace/genesis-angular-coupled-01/initial.json').exists():pytest.skip('Authenticated source required')
    return SphericalPath(1,5,load_field='analytic_tangent',angular_partition=2)


def test_mass_work_and_positive_operators(model):
    a=model
    assert a.audit['new_load_bearing_mass_kg']==pytest.approx(a.audit['old_load_bearing_mass_kg'],rel=1e-14)
    diag=1/np.sqrt(np.diag(a.M));np.linalg.cholesky(a.M*diag[:,None]*diag[None,:])
    rng=np.random.default_rng(7)
    for _ in range(3):
        v=rng.normal(size=a.nd)
        assert v@a.M@v>0 and v@a.D@v>=0
    assert np.all(a.weights>0) and min(np.linalg.eigvalsh(a.Cs[0]))>0
    omega=np.array([.3,-.2,.1])*1e-12
    rotation=np.tile(np.cross(omega,a.X)[:,None,:],(1,2,1)).ravel()
    assert abs(a.F@rotation-omega@a.torque)<1e-9*max(abs(omega@a.torque),1.)


def test_maxwell_restart_exclusive_save_and_guard(model,tmp_path):
    from copy import deepcopy
    a=deepcopy(model);before=encoded(a.state())
    with pytest.raises(ValueError):a.advance(0)
    assert encoded(a.state())==before
    L=a.advance(.0625)
    assert L['maxwell']>0 and L['drag']>0 and L['elastic_after']>0
    p=tmp_path/'new_perturbation.json';a.save(p)
    with pytest.raises(FileExistsError):a.save(p)
    b=SphericalPath.load(p);a.advance(.0625);b.advance(.0625)
    assert encoded(a.state())==encoded(b.state())
    assert a.heat>0 and a.cumulative_external>0
