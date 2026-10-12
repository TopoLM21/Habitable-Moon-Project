"""At most 78 saved-component comparisons; no numerical producer imports."""
from fractions import Fraction as F
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import time

ROOT=Path(__file__).resolve().parents[2]
PROBES=('A_only','saved_accepted_interval_mean')
WORK=('cross_parent_work_j','cross_added_work_j','self_work_j')


def decode(v):return F(*map(int,v))
def pin(p):return dict(path=str(p),bytes=p.stat().st_size,raw_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def plain(x):
    if isinstance(x,F):return dict(fraction=[str(x.numerator),str(x.denominator)])
    if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [plain(v) for v in x]
    return x
def digest(x):return hashlib.sha256(json.dumps(plain(x),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def sign(lo,hi):
    if lo>0:return 'positive'
    if hi<0:return 'negative'
    if lo==hi==0:return 'exact_zero'
    return 'unresolved'


def error(value,lo,hi):
    a,b=value-hi,value-lo;upper=max(abs(a),abs(b));lower=F() if a<=0<=b else min(abs(a),abs(b))
    allowed=F(1,500000)*abs(value)
    return dict(signed=[str(a),str(b)],absolute_lower=str(lower),absolute_upper=str(upper),
                own_relative_allowance=str(allowed),passes_selected_radial_2e6=upper<=allowed,
                sign_matches_reference=sign(value,value)==sign(lo,hi))


def load():
    base=ROOT/'analysis/genesis_volume_integral_validation/evidence/actual_control/case_000_fixed_100m_clipped_support'
    paths=dict(old=base/'runtime_integral_bounds.json',truth=base/'true_source_certificate.json',
        new=Path('/workspace/genesis-precision-full-01/case_000/selected.json'),
        fixture=ROOT/'analysis/genesis_volume_exact_lambda_validation/evidence/actual_control/source_fixture.json',
        line=ROOT/'analysis/genesis_volume_exact_lambda_validation/evidence/actual_control/case_000_fixed_100m_clipped_support.json')
    data={k:json.loads(p.read_bytes()) for k,p in paths.items()}
    old,new,truth,line=(data[k] for k in ('old','new','truth','line'))
    case=data['fixture']['cases'][0];support=case['supports']['fixed_100m_clipped_support']
    assert line['source_case_sha256']==case['source_case_sha256']
    assert line['source_support_sha256']==support['support_sha256']
    assert line['source_fixture_sha256']==data['fixture']['source_fixture_sha256']
    assert line['original_computational_support']==support
    assert old['source_binding_sha256']==line['line_witness']['source_binding_sha256']
    assert old['component_names']==new['component_names']==truth['component_names']
    assert old['original_N_exact_binary_ratios']==new['original_N']
    N=list(map(decode,new['original_N']))
    assert N==[F(v) for v in line['line_witness_barycentric_exact_binary_ratios']]
    raw={**case['source_inputs'],'seed_center_world_m':support['seed_center_world_m'],'seed_radius_m':support['seed_radius_m']}
    options=dict(added_basis=support['original_declared_added_basis'],parent_basis=support['original_declared_parent_basis'],probes=support['declared_probes'])
    assert digest(dict(raw=raw,N=N,witness=line['line_witness'],binary_places=256,**options))==truth['input_binding_sha256']
    assert new['saved_independent_certificate_pin']=={k:pin(paths['truth'])[k] for k in ('bytes','raw_sha256')}
    probes={p['name']:p for p in support['declared_probes']}
    assert probes['saved_accepted_interval_mean']['mean_e0']==raw['accepted_interval_mean_e0']
    assert probes['saved_accepted_interval_mean']['mean_e1']==raw['accepted_interval_mean_e1']
    return data,probes,paths


def compare():
    clock=time.monotonic();free=shutil.disk_usage('/workspace').free;data,probes,paths=load()
    old,new,truth=(data[k] for k in ('old','new','truth'))
    ov=dict(zip(old['component_names'],map(decode,old['native_values_exact_binary_ratios'])))
    nv=dict(zip(new['component_names'],map(decode,new['values'])))
    nl=dict(zip(new['component_names'],map(decode,new['lower'])));nu=dict(zip(new['component_names'],map(decode,new['upper'])))
    names=[f'probe.{p}.{k}[{i}]' for p in PROBES for k in ('added_parent_force_n','enrichment_force_n') for i in range(18)]
    names += [f'probe.{p}.{k}' for p in PROBES for k in WORK]
    assert len(names)==len(set(names))==78<=100
    rows=[];by_name={}
    for name in names:
        source=truth['components'][name];target=source['true_source_integral_enclosure'];lo,hi=decode(target['lower']),decode(target['upper'])
        assert nl[name]<=lo<=hi<=nu[name]
        eold=error(ov[name],lo,hi);enew=error(nv[name],lo,hi)
        residual=source['saved_native_minus_true_source_enclosure']
        assert list(map(F,eold['signed']))==[decode(residual['lower']),decode(residual['upper'])]
        row=dict(name=name,units='N per dN1 dN2' if '_force_n[' in name else 'J per dN1 dN2',
                 old_value=str(ov[name]),corrected_value=str(nv[name]),reference=[str(lo),str(hi)],
                 old_error=eold,corrected_error=enew,reference_sign=sign(lo,hi))
        rows.append(row);by_name[name]=row
    summaries=[]
    for p in PROBES:
        prefix=f'probe.{p}.';worknames=[prefix+k for k in WORK]
        lows=[F(by_name[n]['reference'][0]) for n in worknames];highs=[F(by_name[n]['reference'][1]) for n in worknames]
        L,U=sum(lows),sum(highs);terms=[]
        for group,vector in [('added_parent_force_n','parent_increment_m'),('enrichment_force_n','enrichment_increment_m')]:
            for i,increment in enumerate(probes[p][vector]):
                row=by_name[f'{prefix}{group}[{i}]'];a=F(increment);lo,hi=map(F,row['reference'])
                terms.append((min(a*lo,a*hi),max(a*lo,a*hi)))
        positive=sum(lo for lo,hi in terms if lo>0);negative=sum(hi for lo,hi in terms if hi<0)
        abs_lower=sum(min(abs(lo),abs(hi)) if not lo<=0<=hi else F() for lo,hi in terms)
        denom=max(abs(L),abs(U));factor=abs_lower/denom if denom else None
        dotL,dotU=sum(lo for lo,hi in terms),sum(hi for lo,hi in terms)
        assert max(dotL,L)<=min(dotU,U)  # Consistency only; not independent accuracy proof.
        versions={}
        for label,values in [('old',ov),('corrected',nv)]:
            total=sum(values[n] for n in worknames)
            dot=sum(F(a)*values[f'{prefix}{group}[{i}]'] for group,vector in
                    [('added_parent_force_n','parent_increment_m'),('enrichment_force_n','enrichment_increment_m')]
                    for i,a in enumerate(probes[p][vector]))
            versions[label]=dict(work_density=str(total),error_against_independent_work=error(total,L,U),
                                force_dot_given_increment=str(dot),duality_residual=str(dot-total),
                                duality_is_not_accuracy_certificate=True)
        selected=[r for r in rows if r['name'].startswith(prefix)]
        summaries.append(dict(probe=p,definition=probes[p],work_reference=[str(L),str(U)],work_sign=sign(L,U),
            nonzero_work_terms=[k for k in WORK if by_name[prefix+k]['reference_sign']!='exact_zero'],
            dot_reference=[str(dotL),str(dotU)],positive_dot_contribution_lower=str(positive),negative_dot_contribution_upper=str(negative),
            has_proven_opposing_dot_terms=positive>0 and negative<0,
            dot_absolute_sum_to_total_lower=str(factor) if factor is not None else None,
            versions=versions,exact_zero_components=sum(r['reference_sign']=='exact_zero' for r in selected),
            old_failed_radial_components=sum(not r['old_error']['passes_selected_radial_2e6'] for r in selected),
            corrected_failed_radial_components=sum(not r['corrected_error']['passes_selected_radial_2e6'] for r in selected)))
    return dict(scope='One actual source radial line; densities per original dN1 dN2, not integrated regional forces/work',
        base_commit='0e5031904e3a2fb5185440a6045213c8d3aa762a',N=new['original_N'],comparisons=len(rows),rows=rows,probes=summaries,
        compatibility=dict(same_line=True,same_component_order=True,exact_source_probe_witness_binding=True,
            corrected_result_bound_to_same_independent_certificate=True,saved_mean_matches_source=True,
            source_inputs_and_probes_frozen=True),input_pins={k:pin(p) for k,p in paths.items()},
        radial_evaluations=0,mechanical_solves=0,angular_integrations=0,seconds=time.monotonic()-clock,
        free_bytes_before=free,free_bytes_after_comparison=shutil.disk_usage('/workspace').free,
        working_tolerance_changed=False,physical_threshold_invented=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Fresh output required')
    r=compare()
    with a.output.open('x') as f:json.dump(r,f,indent=2);f.write('\n')
    for p in r['probes']:
        print(json.dumps(dict(probe=p['probe'],nonzero_work_terms=p['nonzero_work_terms'],work=[float(F(v)) for v in p['work_reference']],
            old_fail=p['old_failed_radial_components'],new_fail=p['corrected_failed_radial_components'],
            opposing=p['has_proven_opposing_dot_terms'],cancellation_factor=float(F(p['dot_absolute_sum_to_total_lower'])) if p['dot_absolute_sum_to_total_lower'] else None,
            versions={k:dict(work=float(F(v['work_density'])),error_upper=float(F(v['error_against_independent_work']['absolute_upper'])),
                 duality_residual=float(F(v['duality_residual']))) for k,v in p['versions'].items()})))


if __name__=='__main__':main()
