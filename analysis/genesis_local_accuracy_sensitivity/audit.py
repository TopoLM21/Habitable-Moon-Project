"""Analytical cross-check of the saved prescribed-path study; no kernel calls."""
from fractions import Fraction as F
from pathlib import Path
import argparse
import json
import hashlib


def check(report):
    assert report['parameters']==dict(Gc_j_m2=1000000,T_pa=4000000,k_pa_m=1000000000,compression_k_pa_m=2000000000)
    k=F(1000000000);T=F(4000000);d0=T/k;df=F(1,2)
    assert report['predeclared_levels']==[8,32,128]
    assert report['constitutive_points']==694 and report['constitutive_api_calls']==22
    assert report['radial_calls']==report['new_angular_integrations']==0
    rows=[]
    for case in report['cases']:
        expected=[];max_energy_error=F();expected_trapezoid_errors=[F(),F(),F()]
        for patch in case['patches']:
            old,new=F(patch['old_gap_m']),F(patch['new_gap_m']);x=abs(new)
            assert patch['area_m2']==1 and patch['prescribed_not_predicted']
            if max(abs(old),x)<=d0:
                work=k*(new*new-old*old)/2;force=k*new
            else:
                assert old==0 and d0<x<df
                final=T*(df-x)/(df-d0);work=T*d0/2+(T+final)*(x-d0)/2
                force=final*(1 if new>0 else -1)
                for i,n in enumerate((8,32,128)):
                    h=x/n;left=(d0//h)*h;right=left+h
                    expected_trapezoid_errors[i]+=(k+T/(df-d0))*(d0-left)*(right-d0)/2
            assert work==F(patch['oracle_work_j']) and force==F(patch['oracle_force_n'])
            energy=F(patch['stored_energy_change_j'])+F(patch['dissipation_increment_j'])
            max_energy_error=max(max_energy_error,abs(work-energy));expected.append(work)
        truth=sum(expected);assert truth==F(case['reference_work_j'])
        for row,theory in zip(case['levels'],expected_trapezoid_errors):
            # Exact trapezoid kink error for ideal uniform nodes; difference
            # here measures implementation roundoff, not a physical acceptance rule.
            measured=truth-F(row['work_j'])
            assert abs(measured-theory)<F(1,10**7)
            e=float(abs(measured));assert abs(e-row['true_error_j'])<1e-15*max(1,e)
            d=row['diagnostics_only'];assert d['pure_pass']==(e<=d['pure_allowance'])
            assert d['mixed_pass']==(e<=d['mixed_allowance'])
        rows.append(dict(case=case['case'],work_reference_j=str(truth),
                         exact_uniform_grid_trapezoid_errors_j=list(map(str,expected_trapezoid_errors)),
                         maximum_oracle_vs_energy_error_j=float(max_energy_error)))
    thresholds=report['threshold_scenarios']
    assert [r['softening_decision_resolved'] for r in thresholds]==[False,False,True]
    for r in thresholds:
        assert r['softening_decision_resolved']==(r['interval_m'][0]>r['peak_separation_m'])
    return dict(cases=rows,independent_piecewise_traction_area_check=True,independent_trapezoid_kink_error_check=True,
                additional_constitutive_kernel_calls=0,additional_radial_calls=0,
                scope='Local scalar prescribed paths; analytical roundoff cross-check, not certified global mechanics')


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=check(json.loads(a.input.read_bytes()));result['input_sha256']=hashlib.sha256(a.input.read_bytes()).hexdigest()
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
