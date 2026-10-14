"""Independent P1 facet load/work identities used in the paired control."""
import numpy as np
from tectonics.simplified_genesis_basal import facet
from analysis.genesis_geological_handoff.coupled_load_audit import prescribed,basal,ORIGINAL


def test_nested_p1_preserves_virtual_work_force_and_moment():
    X=np.array([[1.,0.,2.],[4.,1.,2.],[0.,3.,2.]])
    T=np.array([[2.,-3.,5.],[7.,11.,-1.],[-2.,4.,8.]])
    P=np.array([[1,0,0],[0,1,0],[0,0,1],[.5,.5,0],[0,.5,.5],[.5,0,.5]])
    faces=np.array([[0,3,5],[3,1,4],[5,4,2],[3,4,5]])
    h=-.012;old=facet(X,h,1e14,T,np.zeros((3,3)))
    F=np.zeros(36);D=np.zeros((36,36));total=np.zeros(3);torque=np.zeros(3)
    for ids in faces:
        row=facet((P@X)[ids],h,1e14,(P@T)[ids],np.zeros((3,3)))
        ix=(ids[:,None]*6+np.arange(6)).ravel()
        np.add.at(F,ix,row['Ft']);D[np.ix_(ix,ix)]+=row['D']
        nodal=row['Ft'].reshape(3,6)[:,:3]
        total+=nodal.sum(0);torque+=np.cross((P@X)[ids],nodal).sum(0)
    P6=np.kron(P,np.eye(6))
    np.testing.assert_allclose(P6.T@F,old['Ft'],rtol=1e-14,atol=1e-13)
    np.testing.assert_allclose(P6.T@D@P6,old['D'],rtol=1e-14,atol=.1)
    nodal=old['Ft'].reshape(3,6)[:,:3]
    np.testing.assert_allclose(total,nodal.sum(0),rtol=1e-14)
    np.testing.assert_allclose(torque,np.cross(X,nodal).sum(0),rtol=1e-14)
    # Independent constant and linear test displacements are rigid motions.
    assert np.linalg.norm(total)>0 and np.linalg.norm(torque)>0


def test_analysis_override_restores_after_exception():
    try:
        with prescribed(np.zeros((3,3))):
            assert basal.assemble is not ORIGINAL
            raise RuntimeError('deliberate')
    except RuntimeError:pass
    assert basal.assemble is ORIGINAL


def test_frozen_response_is_linear_and_preserves_source():
    import pytest
    from tectonics.experimental_angular_coupled import AngularCoupled
    from tectonics.genesis_transfer_bridge import encoded
    from analysis.genesis_geological_handoff.coupled_load_audit import SOURCE,vertex_traction
    from analysis.genesis_geological_handoff.coupled_load_linear import response
    if not SOURCE.exists():pytest.skip('Authenticated local source required')
    a=AngularCoupled.load(SOURCE);before=encoded(a.model.state)
    t,_=vertex_traction(a.model)
    stress,v,K,L=response(a,t);half,vhalf,Khalf,Lhalf=response(a,.5*t)
    np.testing.assert_allclose(stress,2*half,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(v,2*vhalf,rtol=1e-12,atol=1e-30)
    np.testing.assert_array_equal(K,Khalf)
    assert encoded(a.model.state)==before
    assert L['balance_relative']<1e-10 and Lhalf['balance_relative']<1e-10
