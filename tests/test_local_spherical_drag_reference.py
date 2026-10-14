"""Independent area, resultant/work and manufactured-solution checks."""
import numpy as np
from analysis.genesis_geological_handoff.local_spherical_drag_reference import quadrature,assemble,metrics,run

TRI=np.array([[0.,0.,1.],[.4,0.,1.],[0.,.3,1.]])
TRI=TRI/np.linalg.norm(TRI,axis=1)[:,None]*5.3e6


def test_curved_area_against_exact_solid_angle():
    x,n,w,N=quadrature(TRI,10);u=TRI/5.3e6
    solid=2*np.arctan2(abs(np.linalg.det(u)),1+u[0]@u[1]+u[1]@u[2]+u[2]@u[0])
    assert abs(w.sum()/(solid*5.3e6**2)-1)<1e-11
    assert np.all(w>0)
    np.testing.assert_allclose(np.linalg.norm(x,axis=1),5.3e6,rtol=1e-15)


def test_same_unprojected_force_torque_and_work_for_both_geometries():
    a=assemble(TRI,1e14,10);v=np.array([1.,-.7,.2,.3])*1e-12
    traction=np.column_stack((100+20*a['n'][:,0],50*a['n'][:,2],-30*a['n'][:,1]))
    F,row=metrics(a,traction,v)
    np.testing.assert_allclose(F[:3]*5.3e6,row['moment_nm'],rtol=1e-13)
    assert abs(F[3]-np.dot(a['normal'],row['force_n']))<1e-12*np.linalg.norm(row['force_n'])
    assert row['virtual_work_identity_relative']<1e-13
    # F has no dependence on either drag projector; no traction projection.
    np.testing.assert_allclose(F,np.einsum('q,qik,qi->k',a['w'],a['B'],traction),rtol=0,atol=0)


def test_transverse_geometric_null_mode_and_known_solution():
    r=run(TRI,1e14,700.,10)
    for row in r['cases'].values():assert row['curved_relative_solution_error']<1e-12
    t=r['cases']['transverse']
    assert t['flat_equation_unsolvable'] and t['flat_power_w']==0
    assert t['curved_power_w']>0 and t['flat_known_field_residual_relative']==1
    assert min(r['curved_eigenvalues'])>0
