"""Independent affine kinematics and projected-traction equilibrium."""
import numpy as np
import pytest
from tectonics.experimental_angular_coupled import AngularCoupled
from tectonics.simplified_genesis_basal import facet
from analysis.genesis_geological_handoff.coupled_load_audit import SOURCE,vertex_traction,with_traction
from analysis.genesis_geological_handoff.linear_response_resolution import prepare,stiffness,scaled_solve,YEAR


@pytest.fixture(scope='module')
def prepared():
    if not SOURCE.exists():pytest.skip('Authenticated local source required')
    a=AngularCoupled.load(SOURCE);t,_=vertex_traction(a.model)
    return a,t,prepare(a,t)


def test_analytic_affine_strain_not_matrix_generated(prepared):
    a,t,p=prepared;m=p['m'];X=np.array(m.config['nodal']['X']);frames=np.array(m.config['nodal']['frames'])
    B=np.array([[1.,.2,-.3],[.4,-.8,.7],[.1,.5,.6]])*1e-17
    v=np.tile((X@B.T)[:,None,:],(1,2,1)).ravel()
    for i,b in enumerate(p['blocks']):
        g=frames[i//3].T@B@frames[i//3]
        expected=np.r_[g[0,0],g[1,1],g[2,2],g[0,1]+g[1,0],g[0,2]+g[2,0],g[1,2]+g[2,1],np.zeros(6)]
        np.testing.assert_allclose(b['L']@v[b['dofs']],expected,atol=1e-29,rtol=1e-12)


def test_known_pure_drag_field_matches_independent_facet_projection(prepared):
    a,t,p=prepared;b=with_traction(a.model,t);F=np.zeros_like(p['F'])
    for f,idx in enumerate(b['dofs']):
        old=b['faces'][f];projected=t[a.model.mesh.faces[f]]@old['P']
        row=facet(old['bottom'],old['h'],old['beta'],projected,np.zeros((3,3)))
        np.add.at(F,idx,row['Ft'])
    v=np.zeros((a.model.mesh.vertex_count,6));v[:,:3]=t/1e14
    assert np.linalg.norm(p['D']@v.ravel()-F)/np.linalg.norm(F)<1e-14


def test_manufactured_full_system_recovers_affine_velocity(prepared):
    _,_,p=prepared;K,_=stiffness(p,.0625*YEAR);A=p['M']/(.0625*YEAR)+p['D']+.0625*YEAR*K
    X=np.array(p['m'].config['nodal']['X']);u=np.tile((X/np.linalg.norm(X,axis=1)[:,None])[:,None,:],(1,2,1)).ravel()*1e-10
    actual,report=scaled_solve(A,A@u)
    assert np.linalg.norm(actual-u)/np.linalg.norm(u)<1e-9
    assert report['relative_residual']<1e-12
