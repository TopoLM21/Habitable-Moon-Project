"""Independent arithmetic checks of the same saved component comparisons."""
from fractions import Fraction as F
from pathlib import Path
import argparse
import hashlib
import json


def audit_rows(rows):
    assert len(rows)<=100 and len({r['name'] for r in rows})==len(rows)
    for r in rows:
        lo,hi=map(F,r['reference']);assert lo<=hi
        for value_key,error_key in [('old_value','old_error'),('corrected_value','corrected_error')]:
            v=F(r[value_key]);e=r[error_key];a,b=map(F,e['signed'])
            assert a+hi==v and b+lo==v
            assert F(e['absolute_upper'])==max(abs(v-lo),abs(v-hi))
            expected_lower=0 if lo<=v<=hi else min(abs(v-lo),abs(v-hi))
            assert F(e['absolute_lower'])==expected_lower
            assert F(e['own_relative_allowance'])==abs(v)/500000
            assert e['passes_selected_radial_2e6']==(F(e['absolute_upper'])<=abs(v)/500000)
    return len(rows)


def audit(report):
    count=audit_rows(report['rows']);assert count==report['comparisons']==78
    by_name={r['name']:r for r in report['rows']}
    assert report['radial_evaluations']==report['mechanical_solves']==report['angular_integrations']==0
    for p in report['probes']:
        prefix='probe.'+p['probe']+'.'
        names=[prefix+k for k in ('cross_parent_work_j','cross_added_work_j','self_work_j')]
        L=sum(F(by_name[n]['reference'][0]) for n in names);U=sum(F(by_name[n]['reference'][1]) for n in names)
        assert [L,U]==list(map(F,p['work_reference']))
        for label,value_key in [('old','old_value'),('corrected','corrected_value')]:
            total=sum(F(by_name[n][value_key]) for n in names);saved=p['versions'][label]
            dot=F()
            for group,vector in [('added_parent_force_n','parent_increment_m'),('enrichment_force_n','enrichment_increment_m')]:
                dot+=sum(F(a)*F(by_name[f'{prefix}{group}[{i}]'][value_key]) for i,a in enumerate(p['definition'][vector]))
            assert total==F(saved['work_density']) and dot==F(saved['force_dot_given_increment'])
            assert dot-total==F(saved['duality_residual'])
            assert F(saved['error_against_independent_work']['absolute_upper'])==max(abs(total-L),abs(total-U))
    return dict(unique_components=count,exact_interval_residuals_verified=True,fixed_probe_work_reduction_verified=True,
                independent_of_producer_energy_identity=True,additional_radial_evaluations=0,mechanical_solves=0,
                scope='Verification of arithmetic against saved independent certificates, not regeneration of their physical integrals')


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=audit(json.loads(a.input.read_bytes()));result['input_sha256']=hashlib.sha256(a.input.read_bytes()).hexdigest()
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
