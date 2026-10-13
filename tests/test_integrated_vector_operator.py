"""Patch, nullspace, quadrature and covariance tests independent of damage."""
from pathlib import Path
import json
import numpy as np
import pytest
from tectonics.mesh import build_icosphere
from analysis.genesis_volume_continuation_validation import independent_audit as reference
from analysis.genesis_geological_handoff.integrated_vector_operator import blocks, first_step
from analysis.genesis_geological_handoff.base_operator_audit import rotation, R, INNER, EPS, field


def patch():
    mesh=build_icosphere(2)
    return mesh.vertices[mesh.faces[7]]*R


def stiffness(triangle, high_order=False):
    args=reference.angular_rule(4) if high_order else ()
    B,w,_,_=blocks(triangle,INNER,*args)
    C=reference.isotropic_matrix(6e10,.25,.13,.02)
    return np.einsum('raki,kl,ralj,ra->ij',B,C,B,w)


def test_three_point_stiffness_matches_independent_sixteen_point_quadrature():
    expected=stiffness(patch(),True)
    np.testing.assert_allclose(stiffness(patch()),expected,rtol=2e-13,atol=np.linalg.norm(expected)*1e-14)


def test_full_integration_removes_extra_zero_energy_mode_without_rigid_penalty():
    B,w,_,_=blocks(patch(),INNER)
    mean=B.mean(axis=1).reshape(-1,18)
    full=B.reshape(-1,18)
    assert np.linalg.matrix_rank(mean,tol=np.linalg.norm(mean)*1e-10)==11
    assert np.linalg.matrix_rank(full,tol=np.linalg.norm(full)*1e-10)==12
    C=reference.isotropic_matrix(6e10,.25,.13,.02)
    reduced=np.einsum('rki,kl,rlj,r->ij',mean.reshape(2,6,18),C,mean.reshape(2,6,18),w.sum(axis=1))
    exact=stiffness(patch())
    # Restored variance is positive semidefinite, not arbitrary stabilization.
    assert np.linalg.eigvalsh(exact-reduced).min()>-np.linalg.norm(exact)*1e-13
    for axis in np.eye(3):
        rigid=np.concatenate((np.cross(axis,patch()),np.cross(axis,patch())),axis=1).ravel()
        assert np.linalg.norm(full@rigid)<1e-12
        translation=np.tile(np.r_[axis,np.zeros(3)],3)
        assert np.linalg.norm(full@translation)*R<1e-12


def test_affine_patch_and_covariant_rotation():
    tri=patch();Q=rotation();G=EPS*np.array([[1.,.4,.2],[-.1,-.5,.3],[.2,.1,.7]])
    for triangle,gradient in ((tri,G),(tri@Q.T,Q@G@Q.T)):
        B,w,_,frame=blocks(triangle,INNER)
        nodal=triangle@gradient.T
        dof=np.concatenate((nodal,nodal),axis=1).ravel()
        expected=reference.engineering_gradient(gradient,frame)
        np.testing.assert_allclose(B@dof,np.broadcast_to(expected,(2,3,6)),atol=EPS*1e-13)
    transform=np.kron(np.eye(6),Q)
    K=stiffness(tri);rotated=stiffness(tri@Q.T)
    np.testing.assert_allclose(rotated,transform@K@transform.T,atol=np.linalg.norm(K)*1e-13,rtol=1e-12)


def test_manufactured_gradient_against_cartesian_finite_difference():
    x=np.array([[.4,.7,.5]])*R
    delta=R*1e-5
    for degree in (0,1):
        _,gradient=field(x,degree)
        numerical=np.stack([(field(x+axis*delta,degree)[0]-field(x-axis*delta,degree)[0])/(2*delta) for axis in np.eye(3)],axis=-1)
        np.testing.assert_allclose(gradient,numerical,atol=EPS*1e-9,rtol=1e-8)


def test_passive_legacy_crosscheck_and_integrated_energy():
    if not Path('/workspace/genesis-moving-basal-source-01.json').is_file():pytest.skip('Authenticated local source required')
    from analysis.genesis_geological_handoff.edge_spatial_audit import AuditGrid,ROOT
    coeff=np.array(json.loads((ROOT/'edge_spatial_control.json').read_text())['coefficients'])
    g=AuditGrid(2,False,coeff)
    source_material=g.a.C.copy();state=g.initial()
    g.step(state,.5)
    legacy,_=first_step(g,integrated=False)
    np.testing.assert_allclose(legacy,g.last_stress,atol=1e-5,rtol=1e-10)
    _,row=first_step(g,integrated=True)
    assert row['momentum']<1e-8 and row['energy_relative']<5e-8
    np.testing.assert_array_equal(source_material,g.a.C)
    assert state==g.initial()
