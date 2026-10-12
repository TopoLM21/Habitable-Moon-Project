"""Fixed local cohesive-law sensitivity study, not an accepted Genesis step.

Four scalar patch paths in three fixed cases. Composite trapezoid work uses
only traction outputs; an independent rational primitive supplies the oracle.
"""
from fractions import Fraction as F
from pathlib import Path
import argparse
import hashlib
import json
import math
import shutil
import time
import numpy as np
from tectonics.genesis_volume_creation_law import ProspectiveCreationLaw,evaluate_creation_law

GC=1_000_000;T=4_000_000;K=1_000_000_000;KC=2_000_000_000
LEVELS=(8,32,128);EPSILONS=(1e-4,2e-6,1e-8)
AREA=1.;POINT_CAP=694


def primitive(x):
    """Independent monotone 1D tensile/shear traction primitive, exact rational."""
    x=abs(F(x));d0=F(T,K);df=F(2*GC,T)
    if x<=d0:return F(K,2)*x*x
    if x>=df:return F(GC)
    y=x-d0
    return F(K,2)*d0*d0+T*y-F(T,2)*(y*y)/(df-d0)


def oracle(old,new):
    old,new=F(old),F(new);d0=F(T,K);df=F(2*GC,T)
    if max(abs(old),abs(new))<=d0:
        return K*new,F(K,2)*(new*new-old*old)
    if old!=0:raise ValueError('Oracle scope: elastic paths or monotone loading from zero')
    x=abs(new)
    magnitude=K*x if x<=d0 else (T*(df-x)/(df-d0) if x<df else F())
    return magnitude*(1 if new>=0 else -1),primitive(new)


class Counter:
    def __init__(self):self.points=0;self.calls=0
    def evaluate(self,law,old,new,axis):
        values=np.asarray(new,float);count=values.size
        if self.points+count>POINT_CAP:raise ValueError('Fixed constitutive-point cap reached')
        a=np.zeros(values.shape+(3,));b=np.zeros_like(a);a[...,axis]=old;b[...,axis]=values
        self.points+=count;self.calls+=1
        return evaluate_creation_law(law,a,b,abs(old))


def diagnostic(error,value,scale,epsilon=2e-6):
    pure=epsilon*abs(value);mixed=epsilon*(scale+abs(value))
    return dict(pure_allowance=pure,mixed_allowance=mixed,pure_pass=error<=pure,mixed_pass=error<=mixed,
                physical_scale=scale,absolute_term=epsilon*scale,relative_term=epsilon*abs(value))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise ValueError('Fresh output required')
    free=shutil.disk_usage('/workspace').free
    if free<16*1024**2:raise ValueError('Need 16 MiB output reserve')
    law=ProspectiveCreationLaw(GC,T,K,KC);d0=law.peak_separation_m
    # Both cancelling paths stay strictly elastic; no invented damaged history.
    cases=[('ordinary',[(0.,.0123,0)]),
           ('cancellation',[(0.,.001,1),(-math.sqrt(2)*.001,-.001*(1-1e-8),1)]),
           ('near_peak',[(0.,d0*(1+1e-6),0)])]
    counter=Counter();start=time.monotonic();results=[]
    for name,paths in cases:
        patches=[]
        for old,new,axis in paths:
            endpoint=counter.evaluate(law,old,new,axis);traction,work=oracle(old,new)
            energy=float(endpoint.stored_energy_j_m2-endpoint.old_stored_energy_j_m2+endpoint.fracture_dissipation_increment_j_m2)*AREA
            force=float(endpoint.traction_pa[axis])*AREA
            estimates=[]
            for n in LEVELS:
                positions=np.linspace(old,new,n+1)
                response=counter.evaluate(law,old,positions,axis)
                tr=response.traction_pa[...,axis]
                integral=float((new-old)/n*(.5*tr[0]+np.sum(tr[1:-1])+.5*tr[-1]))*AREA
                estimates.append(integral)
            patches.append(dict(old_gap_m=old,new_gap_m=new,axis=axis,area_m2=AREA,
                prescribed_not_predicted=True,traction_pa=float(endpoint.traction_pa[axis]),force_n=force,
                oracle_force_n=str(traction*F(AREA)),oracle_work_j=str(work*F(AREA)),
                force_error_n=float(abs(F(force)-traction*F(AREA))),
                stored_energy_change_j=float(endpoint.stored_energy_j_m2-endpoint.old_stored_energy_j_m2)*AREA,
                dissipation_increment_j=float(endpoint.fracture_dissipation_increment_j_m2)*AREA,
                reported_work_j=float(endpoint.mechanical_work_j_m2)*AREA,
                independent_oracle_vs_energy_error_j=float(abs(work*F(AREA)-F(energy))),
                quadrature_work_j=estimates))
        truth=sum(F(p['oracle_work_j']) for p in patches);true_force=sum(F(p['oracle_force_n']) for p in patches)
        force=sum(p['force_n'] for p in patches);energy=sum(p['stored_energy_change_j']+p['dissipation_increment_j'] for p in patches)
        work_scale=len(patches)*AREA*GC;force_scale=len(patches)*AREA*T
        levels=[]
        for index,n in enumerate(LEVELS):
            estimate=sum(p['quadrature_work_j'][index] for p in patches);error=float(abs(F(estimate)-truth))
            levels.append(dict(subintervals_per_patch=n,work_j=estimate,true_error_j=error,
                 relative_true_error=float(abs(F(estimate)-truth)/abs(truth)) if truth else None,
                 independent_energy_balance_residual_j=estimate-energy,
                 work_sign_matches_oracle=(estimate>0)==(truth>0) and (estimate<0)==(truth<0),
                 diagnostics_only=diagnostic(error,float(truth),work_scale)))
        sensitivities=[]
        for eps in EPSILONS:
            # Scenario budgets, NOT measured errors or certified downstream bounds.
            force_budget=eps*force_scale;work_budget=eps*work_scale
            sensitivities.append(dict(epsilon=eps,assumed_force_error_n=force_budget,assumed_work_error_j=work_budget,
                force_sign_resolved=abs(float(true_force))>force_budget,work_sign_resolved=abs(float(truth))>work_budget))
        results.append(dict(case=name,patches=patches,reference_work_j=str(truth),reference_force_n=str(true_force),
            net_force_n=force,net_force_error_n=float(abs(F(force)-true_force)),work_scale_j=work_scale,force_scale_n=force_scale,
            traction_scale_pa=T,levels=levels,assumed_budget_scenarios=sensitivities,
            force_discretization_note='Endpoint force is independent of path-work quadrature; no claim of coupled force convergence'))
    center=d0*(1+1e-6);threshold=[]
    for eps in EPSILONS:
        delta=eps*d0;lo,hi=center-delta,center+delta
        responses=[counter.evaluate(law,0.,x,0) for x in (lo,hi)]
        threshold.append(dict(epsilon=eps,prescribed_gap_center_m=center,assumed_gap_error_m=delta,
            peak_separation_m=d0,gap_margin_m=center-d0,interval_m=[lo,hi],
            damages=[float(r.damage) for r in responses],tractions_pa=[float(r.traction_pa[0]) for r in responses],
            softening_decision_resolved=lo>d0,scenario_only=True))
    assert counter.points==POINT_CAP and counter.calls==22
    paths=[Path(__file__),Path('tectonics/genesis_volume_creation_law.py'),Path('tests/test_genesis_volume_creation_law.py')]
    report=dict(scope='Prescribed 1D local prospective cohesive kernel; not calibrated Genesis, not bulk stress or evolution',
        base_commit='632e0fbe4e1aa84941eac2df0b519235e88f52e1',parameters=dict(Gc_j_m2=GC,T_pa=T,k_pa_m=K,compression_k_pa_m=KC),
        predeclared_levels=list(LEVELS),predeclared_epsilon_scenarios=list(EPSILONS),
        mixed_criterion='E <= epsilon * physical_scale + epsilon * abs(observable); diagnostic proposal only',
        scale_policy='sum of area*peak traction for force; sum of area*Gc for work; peak separation for gap; fixed before results',
        cases=results,threshold_scenarios=threshold,constitutive_points=counter.points,constitutive_api_calls=counter.calls,
        radial_calls=0,new_angular_integrations=0,seconds=time.monotonic()-start,free_bytes_before=free,
        free_bytes_before_output=shutil.disk_usage('/workspace').free,working_tolerance_changed=False,
        source_pins={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths})
    with args.output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(dict(points=counter.points,calls=counter.calls,seconds=report['seconds'],cases=[
        dict(case=c['case'],force=float(F(c['reference_force_n'])),work=float(F(c['reference_work_j'])),
             work_errors=[r['true_error_j'] for r in c['levels']],relative=[r['relative_true_error'] for r in c['levels']]) for c in results],
        threshold_resolved=[r['softening_decision_resolved'] for r in threshold])))


if __name__=='__main__':main()
