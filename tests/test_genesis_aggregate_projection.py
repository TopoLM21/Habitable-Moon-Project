import numpy as np
from analysis.genesis_aggregate_projection.projection import moments,thermal_bins,rigid_motion


def test_uniform_stress_retains_force_work_and_zero_bending():
    z=np.linspace(-2,2,9);w=np.arange(1,10,dtype=float);C=np.diag([3.,5.,8.]);e=np.tile([.1,-.2,.3],(9,1))
    p=moments(w,z,e,C)
    np.testing.assert_allclose(p['fit'],e,atol=1e-16)
    assert p['energy_residual']<1e-28
    np.testing.assert_allclose(C@p['mean'],C@e[0])
    np.testing.assert_allclose(p['slope'],0,atol=1e-16)


def test_uniform_thermal_enthalpy_conserves_inventory():
    m=np.arange(1,9,dtype=float);p=thermal_bins(m,m*1200,np.repeat([0,1],4),2)
    np.testing.assert_array_equal(p['specific_enthalpy'],[1200,1200])
    assert p['mass_residual']==p['energy_residual']==0
    np.testing.assert_array_equal(p['residual_enthalpy'],0)


def test_rigid_motion_preserves_both_momenta_without_residual():
    rng=np.random.default_rng(8);X=rng.normal(size=(12,3));omega=np.array([.2,.3,-.4]);v=(np.cross(omega,X)+[.4,-.2,.7]).ravel()
    p=rigid_motion(np.eye(X.size),X,v)
    np.testing.assert_allclose(p['fit'],v,atol=1e-15)
    assert p['kinetic_residual']<1e-28


def test_nonrigid_and_stress_residuals_are_explicit_not_heat():
    rng=np.random.default_rng(20);X=rng.normal(size=(12,3));v=rng.normal(size=X.size)
    p=rigid_motion(np.eye(X.size),X,v)
    np.testing.assert_allclose(p['moment_residual'],0,atol=1e-14)
    assert p['kinetic_residual']>0
    assert abs(p['kinetic_before']-p['kinetic_retained']-p['kinetic_residual'])<1e-13
    z=np.linspace(-3,3,15);w=np.ones(15);e=np.column_stack((z*z,z+z*z*z));q=moments(w,z,e,np.eye(2))
    assert q['energy_residual']>0
    np.testing.assert_allclose(q['zeroth_error'],0,atol=1e-12)
    np.testing.assert_allclose(q['first_error'],0,atol=1e-12)
    assert abs(q['energy_balance'])<1e-10
    virtual=np.column_stack((2+3*z,-1+.5*z))
    assert abs(np.sum(w[:,None]*virtual*(e-q['fit'])))<1e-10
