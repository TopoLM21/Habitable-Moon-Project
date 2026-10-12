"""Saved-forcing material diagnostic only. Never advances the planet."""
import argparse,json,math
from pathlib import Path
import numpy as np
from tectonics.genesis_volume_loading import coefficient_inner_product,RadialMoments
from tectonics.genesis_volume_operator import conical_radial_moments,ANGULAR_BARYCENTRIC
from analysis.genesis_volume_continuation_validation import independent_audit as q
from analysis.genesis_be_candidate.audit import face_driver
from analysis.genesis_volume_fault_validation.independent_audit import independent_damage_law
from analysis.genesis_connected_step_validation.run import PARENT,PARENT_SHA
from analysis.genesis_coupled_split_control.run import pinned
from .exponential import weights


def material(root,p,state_key='state',inputs_key='inputs'):
    s=p[state_key];inp=p[inputs_key];g=s['geometry'];source=root['source_packet'];shell=source['source']['parameters']['shell']
    X=np.array(g['vertices_m']);faces=np.array(g['faces']);dt=(s['thermal']['time_myr']-root['runtime_state']['thermal']['time_myr'])*365.25*86400*1e6
    before_ids=[c['id'] for c in root['runtime_state']['geometry']['cohorts']]
    after_ids=[c['id'] for c in g['cohorts']]
    assert len(set(after_ids))==800 and set(after_ids)==set(before_ids)
    assert after_ids==[c['id'] for c in s['mechanics']['cohorts']]
    C=np.array(inp['elasticity_pa']);z=np.array(inp['maxwell_ratio']);old0=np.array(inp['old0']);old1=np.array(inp['old1']);th=np.array(inp['thermal0']);du=np.array(inp['increment_m'])
    inc0=[];inc1=[];mom=[]
    for i,c in enumerate(g['cohorts']):
        f=c['face_index'];B0,B1=q.independent_strain_operators(X[faces[f]],np.array(inp['basis_xyz'])[i],ANGULAR_BARYCENTRIC)
        idx=(6*faces[f,:,None]+np.arange(6)).ravel();inc0.append(B0@du[idx]);inc1.append(B1@du[idx]);mom.append(conical_radial_moments(c['outer_scale'],c['inner_scale']))
    inc0=np.array(inc0);inc1=np.array(inc1);mom=RadialMoments(tuple(c['volume_m3'] for c in g['cohorts']),tuple(x[0] for x in mom),tuple(x[1] for x in mom))
    inner=lambda a,b,c,d:coefficient_inner_product(a,b,c,d,C,mom)
    be0=np.array([c['strain0'] for c in s['mechanics']['cohorts']]);be1=np.array([c['strain1'] for c in s['mechanics']['cohorts']]);zero=np.zeros_like(be0)
    assert np.allclose((1+z[:,None,None])*be0,old0+inc0-th,rtol=2e-10,atol=2e-12)
    assert np.allclose((1+z[:,None,None])*be1,old1+inc1,rtol=2e-10,atol=2e-12)
    pieces={'deformation':(inc0,inc1),'cooling':(-th,zero),'relaxation':(-z[:,None,None]*be0,-z[:,None,None]*be1)}
    diag={k:float(.5*inner(*v,*v).sum()) for k,v in pieces.items()};cross={}
    keys=list(pieces)
    for i,k in enumerate(keys):
        for j in keys[i+1:]:cross[k+'__'+j]=float(inner(*pieces[k],*pieces[j]).sum())
    localN=.5*inner(be0-old0,be1-old1,be0-old0,be1-old1);localQ=z*inner(be0,be1,be0,be1)
    N=float(localN.sum());Q=float(localQ.sum())
    assert math.isclose(math.fsum([*diag.values(),*cross.values()]),N,rel_tol=3e-12,abs_tol=1e3)
    coeff=np.array([weights(x) for x in z]);a,b,c,A,B,D=coeff.T
    f0=inc0-th;f1=inc1
    ex0=a[:,None,None]*old0+b[:,None,None]*f0;ex1=a[:,None,None]*old1+b[:,None,None]*f1
    avg0=b[:,None,None]*old0+c[:,None,None]*f0;avg1=b[:,None,None]*old1+c[:,None,None]*f1
    shifted0=old0+(B/A)[:,None,None]*f0;shifted1=old1+(B/A)[:,None,None]*f1
    heat=z*(A*inner(shifted0,shifted1,shifted0,shifted1)+(D-B*B/A)*inner(f0,f1,f0,f1))
    localUbe=.5*inner(be0,be1,be0,be1)
    local_memory_difference=np.sqrt(inner(ex0-be0,ex1-be1,ex0-be0,ex1-be1)/(2*localUbe))
    U0=.5*inner(old0,old1,old0,old1);U1=.5*inner(ex0,ex1,ex0,ex1);W=inner(avg0,avg1,inc0,inc1);Wt=-inner(avg0,avg1,th,zero)
    identity=U1-U0-W-Wt+heat
    assert np.max(np.abs(identity)/(1e3+3e-10*np.maximum(U0,U1)))<=1
    # Independent 24-point temporal quadrature of physical Maxwell heat.
    nodes,ws=np.polynomial.legendre.leggauss(24);quad=np.zeros(len(z))
    for t,w in zip((nodes+1)/2,ws/2):
        at=np.exp(-z*t);bt=np.array([weights(x*t)[1] for x in z])*t
        x0=at[:,None,None]*old0+bt[:,None,None]*f0;x1=at[:,None,None]*old1+bt[:,None,None]*f1
        quad+=w*z*inner(x0,x1,x0,x1)
    assert np.allclose(quad,heat,rtol=3e-12,atol=1e3)
    alt=json.loads(json.dumps(s));alt['mechanics']['elastic_energy_j']=float(U1.sum())
    for i,row in enumerate(alt['mechanics']['cohorts']):row['strain0']=ex0[i].tolist();row['strain1']=ex1[i].tolist()
    be_driver=face_driver(s,source);ex_driver=face_driver(alt,source)
    def stats(v):return dict(min=float(np.min(v)),median=float(np.median(v)),max=float(np.max(v)))
    out=dict(dt_years=dt/(365.25*86400),cohorts=len(z),newborn_count=0,z_dt_over_tau=stats(z),tau_years=stats(dt/z/(365.25*86400)),temperature_increment_k=stats(th[:,0,0]/shell['linear_expansion_per_k']),
        numerical_memory_j=N,physical_maxwell_BE_j=Q,ratio_N_over_Q=N/Q,local_N_over_Q=stats(localN/localQ),local_N_over_elastic=stats(localN/localUbe),change_norm_terms_j=diag,signed_cross_terms_j=cross,
        exact_frozen_forcing=dict(elastic_after_j=float(U1.sum()),elastic_difference_j=float(U1.sum()-s['mechanics']['elastic_energy_j']),physical_heat_j=float(heat.sum()),physical_heat_difference_j=float(heat.sum()-Q),thermal_work_j=float(Wt.sum()),internal_work_j=float(W.sum()),material_energy_residual_j=float(identity.sum()),endpoint_momentum_signed_coupling_defect_j=float(inner(ex0-avg0,ex1-avg1,inc0,inc1).sum()),endpoint_thermal_work_j=float(-inner(ex0,ex1,th,zero).sum()),
            max_face_stress_difference_pa=float(np.max(np.abs(ex_driver['stress_pa']-be_driver['stress_pa']))),face_stress_relative_l2=float(np.linalg.norm(ex_driver['stress_pa']-be_driver['stress_pa'])/np.linalg.norm(be_driver['stress_pa'])),local_memory_relative_energy_norm=stats(local_memory_difference),max_memory_coefficient_difference=float(max(np.max(abs(ex0-be0)),np.max(abs(ex1-be1))))),
        caveat='Constant total deformation/temperature rates and frozen endpoint rheology/damage only. Does not solve alternative momentum, thermal feedback or planetary evolution.')
    rows=[dict(id=cohort['id'],dt_over_tau=float(z[i]),tau_years=float(dt/z[i]/(365.25*86400)),BE_heat_j=float(localQ[i]),exact_heat_j=float(heat[i]),BE_numerical_memory_j=float(localN[i]),BE_elastic_after_j=float(localUbe[i]),exact_elastic_after_j=float(U1[i]),local_memory_relative_energy_norm=float(local_memory_difference[i])) for i,cohort in enumerate(g['cohorts'])]
    return out,alt,ex_driver,rows


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    root=pinned(PARENT,PARENT_SHA)
    paths={'forced25':('/workspace/genesis-forced-coupled-25-01/candidate.json','e6ad0ed1372b18a73c3852f639edafd959776e6cb0d3a747aa75bcffe8c365a5'),
           'unforced50':('/workspace/genesis-cutcell-mechanical-02/candidate.json','5a0cfcd3edfd57f4f05b59e8b408b09c25ac2a704068609e22029ab865230f2e')}
    paths['unforced25']=('/workspace/genesis-coupled-split-25-01/step_1.json','5a0a1f53c9a01237f626b03c7119f98475d8392e0e84349b11bfd1e50c50b9ed')
    results={};details={}
    for name,(path,sha) in paths.items():
        p=pinned(path,sha);report,alt,driver,rows=material(root,p);results[name]=report;details[name]=rows
        if name=='forced25':
            preport,palt,pdriver,_=material(root,p,'predictor','predictor_inputs')
            payload={**p['predictor_driver'],'stress_pa':pdriver['stress_pa'].tolist()};before=root['runtime_state'];expected,law=independent_damage_law(before['fault_history'],payload,root['source_packet'],before['thermal']['orbit'],p['orbit_after'],p['thermal']['time_myr'])
            actual=np.array(p['state']['fault_history']['damage']);alt_damage=np.array(expected['damage'])
            report['damage_predictor_probe']=dict(max_absolute_difference=float(np.max(abs(alt_damage-actual))),max_original=float(actual.max()),max_with_exponential_stress=float(alt_damage.max()),activation_threshold=root['source_packet']['source']['parameters']['faults']['activation_damage'],note='Reevaluate inherited damage law using exponential predictor stress at saved displacement/temperature; do not accept or re-solve final state.')
            report['adaptive_cost_estimate']={str(tol):dict(max_step_years=report['dt_years']*tol/report['ratio_N_over_Q'],steps_for_25_years=math.ceil(report['ratio_N_over_Q']/tol),basis='Frozen observed rate scaling N~dt^2,Q~dt; estimate, not a convergence proof') for tol in (.05,.01)}
    result=dict(status='PASS_SAVED_FORCING_DIAGNOSIS',new_planetary_steps=0,inputs={k:dict(path=p,sha256=h) for k,(p,h) in paths.items()},canonical=dict(path=str(PARENT),sha256=PARENT_SHA),results=results,per_cohort=details)
    with args.output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='per_cohort'},indent=2))


if __name__=='__main__':main()
