"""One saved-operator BE/SDIRK comparison and final numerical decision; no planet."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from .validate import real_patch,ROOT,SAVED,SAVED_SHA,PARENT,PARENT_SHA
from .thin_diagnosis import step,solve_stage,LD
from .audit import audit
from .core import ENERGY_MATRIX
from analysis.genesis_adaptive_control.controller import time_limit


def energy_bound(m,r):
    norms=[]
    for k in r['rates']:
        v=k[m.nt:m.nt+m.nv]
        norms.append(float(v@m.M@v+sum(k[s]@b.H@k[s] for b,s in zip(m.blocks,m.slices))))
    bound=.5*r['dt']**2*np.max(np.abs(np.linalg.eigvalsh(ENERGY_MATRIX)))*sum(norms)
    return dict(stage_rate_energy_norms=norms,absolute_signed_numerical_bound_j=float(bound))


def assembly_check():
    from tectonics.genesis_volume_basal import local_consistent_mass_matrix,local_basal_drag_and_force
    p=json.loads(SAVED.read_bytes())['packet'];c=p['state']['geometry']['cohorts'][0]
    X=np.array(c['top_triangle_m'],LD);outer=LD(c['outer_scale']);inner=LD(c['inner_scale'])
    mid=(outer+inner)/2;half=(outer-inner)/2;ratio=(half/mid)**2
    mean=((outer-1)+(inner-1))/2+2*mid*ratio/(3+ratio)
    var=half**2*(4+(1-ratio)**2)/(15*(1+ratio/3)**2)
    angular=np.full((3,3),LD(1)/12);np.fill_diagonal(angular,LD(1)/6)
    mass=LD(c['mass_kg'])*np.kron(angular,np.kron(np.array([[1,mean],[mean,mean*mean+var]],LD),np.eye(3,dtype=LD)))
    cross=np.cross(X[1]-X[0],X[2]-X[0]);length=np.sqrt(cross@cross);normal=cross/length
    tangent=np.eye(3,dtype=LD)-np.outer(normal,normal);h=inner-1
    drag=LD(p['inputs']['basal_drag_pa_s_m'])*length*inner**2/2*np.kron(angular,np.kron(np.array([[1,h],[h,h*h]],LD),tangent))
    m64=local_consistent_mass_matrix(c['mass_kg'],float(outer),float(inner))
    d64,_,_=local_basal_drag_and_force(np.array(X,float),float(inner),np.zeros((3,3)),p['inputs']['basal_drag_pa_s_m'])
    norm=lambda a:np.sqrt(np.sum(a*a))
    return dict(mass_relative_assembly_difference=float(norm(np.asarray(m64,LD)-mass)/norm(mass)),
        drag_relative_assembly_difference=float(norm(np.asarray(d64,LD)-drag)/norm(drag)),
        note='Diagnostic reassembly only; original float64 matrices used unchanged in both solves. Relative matrix error is not a forward-solution bound.')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    with time_limit(120):
        setup=time.perf_counter();m,exact,scale,p=real_patch();setup=time.perf_counter()-setup
        y=exact(0.);h=12.5
        start=time.perf_counter();s=step(m,0.,y,h,scale);s_seconds=time.perf_counter()-start
        start=time.perf_counter();be,be_detail=solve_stage(m,y,h,h,scale);be_seconds=time.perf_counter()-start
        start=time.perf_counter();a=audit(m,y,s);audit_seconds=time.perf_counter()-start
        truth=exact(h)
        # Score normalized state errors only, NOT the real controller's admission metric.
        def errors(z):
            delta=z-truth
            return dict(temperature_k=float(abs(delta[0])/(m.mass[0]*m.cp[0])),
                velocity_max_m_s=float(np.max(abs(delta[1:19]))),memory_max=float(np.max(abs(delta[19:]))),
                scaled_l2=float(np.linalg.norm(delta/scale)))
        eb,es=errors(be),errors(s['after'])
        impulse=np.linalg.norm(a['impulse_residual'])
        v0=y[1:19];v1=s['after'][1:19]
        # Cancellation-aware backward scale; do not label as relative accuracy.
        impulse_scale=np.linalg.norm(m.M@(v1-v0))
        # Use actual tableau weights, independently from the solver helper.
        from .core import B,C
        for j,w in enumerate(B):
            Y=s['stages'][j];Q,F=m.ports(C[j]*h,m.thermal(Y[:1]),Y[1:19],[Y[19:]])
            impulse_scale+=h*w*(np.linalg.norm(F)+np.linalg.norm(m.D@Y[1:19])+np.linalg.norm(m.blocks[0].B.T@m.blocks[0].H@Y[19:]))
        energy_ulps=8*(abs(np.spacing(m.energy(y)))+abs(np.spacing(m.energy(s['after']))))
        bound=energy_bound(m,s);N=a['signed_numerical_j']
        assert max(x['original_scaled_residual'] for x in s['details'])<=2e-10
        assert be_detail['original_scaled_residual']<=2e-10
        assert abs(a['balance_residual_j'])<=energy_ulps
        assert impulse/max(impulse_scale,1.)<2e-10
        assert abs(N)<=bound['absolute_signed_numerical_bound_j']*(1+1e-12)
        previous=json.loads((Path(__file__).parent/'validation.json').read_text())
        old=previous['negative_numerical_example'];oldN=old['signed_numerical_j']
        # Reconstruct scalar rates analytically for the previously recorded example.
        gamma=1-1/np.sqrt(2);dt=.5;norms=[]
        for decay,weight,start_value in ((.3,1.,.2),(2.,2.,.3)):
            y1=start_value/(1+gamma*dt*decay)
            y2=(start_value-dt*(1-gamma)*decay*y1)/(1+gamma*dt*decay)
            norms.append((weight*(decay*y1)**2,weight*(decay*y2)**2))
        oldbound=.5*dt**2*np.max(np.abs(np.linalg.eigvalsh(ENERGY_MATRIX)))*sum(sum(x) for x in norms)
        report=dict(decision='NOT_SUITABLE_UNDER_UNCHANGED_STRICT_PASSIVITY',new_planetary_steps=0,
            source=p,setup_seconds=setup,assembly=assembly_check(),sdirk_stage_details=s['details'],be_stage_detail=be_detail,
            one_saved_operator_comparison=dict(scope='one 12.5s manufactured step on unchanged saved cohort operators, not actual trajectory forcing',
                be_errors=eb,sdirk_errors=es,error_reduction=eb['scaled_l2']/es['scaled_l2'],
                be_seconds=be_seconds,sdirk_seconds=s_seconds,cost_ratio=s_seconds/be_seconds,
                sdirk_audit_seconds=audit_seconds,accuracy_metric='same explicit state scales; not planetary admission'),
            energy=dict(**bound,signed_numerical_j=N,physical_drag_j=a['physical_drag_j'],physical_maxwell_j=a['physical_maxwell_j'],
                source_heat_j=a['source_heat_j'],external_work_j=a['external_work_j'],residual_j=a['balance_residual_j'],
                stored_energy_8ulp_allowance_j=float(energy_ulps),negative_over_drag=abs(N)/a['physical_drag_j'],
                negative_over_abs_source=abs(N)/abs(a['source_heat_j'])),
            impulse_backward_residual=impulse/max(impulse_scale,1.),
            previous_unforced_example=dict(signed_numerical_j=oldN,absolute_bound_j=float(oldbound),
                negative_over_maxwell=abs(oldN)/old['physical_maxwell_j'],
                negative_over_drag_plus_maxwell=abs(oldN)/(old['physical_drag_j']+old['physical_maxwell_j']),
                explanation='No external force and zero thermal expansion/B: negative defect exists in scalar decay. Structural lack of algebraic stability, not coupling error.'),
            conclusion='Thin-block solve repaired; negative numerical energy is structural and cannot be removed by scaling/refinement without changing scheme or acceptance. Stop development of this prototype.',
            practical_alternative='Keep audited BE plus invariant cache and event/error-limited growth. Larger dt only where output tests pass; roughly first-order work versus second-order, no guaranteed kyr step.')
        report['code_sha256']={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in (Path(__file__),Path(__file__).with_name('thin_diagnosis.py'))}
        assert hashlib.sha256(SAVED.read_bytes()).hexdigest()==SAVED_SHA
        assert hashlib.sha256(PARENT.read_bytes()).hexdigest()==PARENT_SHA
        with args.output.open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
        print(json.dumps(report,indent=2))


if __name__=='__main__':main()
