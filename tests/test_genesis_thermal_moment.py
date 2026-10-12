import json
import numpy as np
import pytest
from analysis.genesis_thermal_moment.adapter import moment_nodes,split_enthalpy,positive_chain
from analysis.genesis_aggregate_coupled.core import step
from tectonics.genesis_implicit_enthalpy import solve


def test_first_moment_and_linear_temperature_are_preserved():
    G=np.diag([12.,4.]);nodes,P,w=moment_nodes(G)
    np.testing.assert_allclose(P.T@(w[:,None]*P),G,rtol=1e-15,atol=1e-15)
    rho=3.;cp=1200.;coeff=np.array([1380.,8.]);mass=rho*G[0,0]
    moments=rho*cp*G@coeff
    m,E=split_enthalpy(moments[0],moments[1],mass,nodes)
    np.testing.assert_allclose(E/m/cp,P@coeff,rtol=1e-15)
    assert E.sum()==moments[0]
    # First moment subtracts large positive energies; use their ULP scale.
    np.testing.assert_allclose(nodes@E,moments[1],rtol=0,atol=8*np.finfo(float).eps*np.sum(np.abs(nodes*E)))
    np.testing.assert_allclose(np.linalg.solve(P,E/m/cp),coeff,rtol=0,atol=16*np.finfo(float).eps*np.max(np.abs(coeff)))


def test_linear_fourier_profile_is_stationary_and_resistance_exact():
    nodes,P,w=moment_nodes(np.diag([2.,2/3]))
    # Physical depth runs opposite to the reference outward normal.
    positions=1-nodes;edges=positive_chain([0,1],positions,2.,4.,2,3)
    T=np.r_[1400-10*positions,1400.,1380.]
    np.testing.assert_allclose([G*(T[i]-T[j]) for i,j,G in edges],40.,rtol=1e-13)
    assert abs(sum(1/G for _,_,G in edges)-.5)<1e-15
    assert all(G>0 for _,_,G in edges)


def test_positive_graph_step_preserves_heat_and_restart(tmp_path):
    nodes,_,_=moment_nodes(np.diag([2.,2/3]));edges=positive_chain([0,1],1-nodes,2.,4.,2,3)
    m=np.array([1.,1.,5.,5.]);cp=np.ones(4);T=np.array([1382.,1390.,1370.,1395.]);h=T.copy()
    args=(m,h,cp,np.zeros(4),np.full(4,1400.),np.full(4,2000.),np.zeros(4),edges,.1)
    r=solve(*args);assert r['temperature_k'].min()>=T.min() and r['temperature_k'].max()<=T.max()
    assert abs(m@r['specific_h']-m@h)<1e-11
    path=tmp_path/'state.json';path.write_text(json.dumps(dict(h=r['specific_h'].tolist())))
    direct=solve(m,r['specific_h'],*args[2:]);restored=solve(m,np.array(json.loads(path.read_text())['h']),*args[2:])
    np.testing.assert_array_equal(direct['temperature_k'],restored['temperature_k'])


def test_thermal_moment_work_and_positive_maxwell_heat_match_mechanics():
    G=np.diag([2.,2/3]);nodes,P,w=moment_nodes(G);C=np.diag([2.,3.,4.,1.,1.,1.]);H=np.kron(G,C)
    rates=np.array([.1,.2]);R=np.kron(np.linalg.solve(G,P.T@((w*rates)[:,None]*P)),np.eye(6))
    old=np.arange(12,dtype=float)*.001;theta=(np.array([.01,.02])[:,None]*np.array([1.,1.,1.,0.,0.,0.])).ravel()
    b=dict(H=H,L=np.zeros((12,3)),R=R,old=old,theta=theta,dofs=np.arange(3));r=step(np.eye(3),np.eye(3),np.zeros(3),np.zeros(3),.1,[b])
    e=P@r['memory'][0].reshape(2,6);stress=e@C
    Q=.1*w*rates*np.einsum('ni,ni->n',stress,e)
    W=-w*np.einsum('ni,ni->n',stress,P@theta.reshape(2,6))
    assert np.all(Q>=0)
    assert np.isclose(Q.sum(),r['ledger']['maxwell'],rtol=1e-14)
    assert np.isclose(W.sum(),r['ledger']['thermal_work'],rtol=1e-14)
    assert r['ledger']['balance_relative']<1e-14


def test_invalid_moment_rejected_without_clipping():
    with pytest.raises(ValueError):split_enthalpy(1.,100.,1.,np.array([1.,-1.]))
    with pytest.raises(ValueError):positive_chain([0,1],[1.,.5],2.,1.,2,3)
