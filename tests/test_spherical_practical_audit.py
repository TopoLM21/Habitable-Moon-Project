"""Norm decomposition must not confuse tensor shear with scalar shear."""
import numpy as np
from analysis.genesis_geological_handoff.spherical_practical_audit import decompose,tensor


def test_radial_tangential_decomposition_is_orthogonal():
    rng=np.random.default_rng(23);T=rng.normal(size=(30,3,3));T=(T+T.transpose(0,2,1))/2
    n=rng.normal(size=(30,3));n/=np.linalg.norm(n,axis=1)[:,None]
    parts=decompose(T,n)
    np.testing.assert_allclose(sum(parts),np.sum(T*T,axis=(1,2)),rtol=1e-13)
    Q,_=np.linalg.qr(rng.normal(size=(3,3)))
    rotated=np.einsum('ij,qjk,lk->qil',Q,T,Q)
    for a,b in zip(parts,decompose(rotated,n@Q.T)):
        np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-14)


def test_tensor_stress_shear_contributes_twice():
    T=tensor(np.array([[0.,0.,0.,3.,0.,0.]]))
    rr,mixed,tt=decompose(T,np.array([[1.,0.,0.]]))
    np.testing.assert_array_equal(rr,[0.]);np.testing.assert_array_equal(mixed,[18.]);np.testing.assert_array_equal(tt,[0.])
