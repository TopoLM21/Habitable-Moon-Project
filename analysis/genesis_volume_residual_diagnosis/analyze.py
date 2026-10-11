"""Read-only localization and matched-parent comparison of saved experiments.

No experiment/evaluator import, no source evaluation, no new adaptive run.
"""
from fractions import Fraction as F
from pathlib import Path
import argparse
import hashlib
import json


def pin(path):
    return dict(path=str(path), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def key(c):
    return (str(c['triangle']), *(str(F(*map(int,c[k]))) for k in ('s0','s1','t0','t1')))


def normalized(values, allowances):
    return max((v/a if a else 0) for v,a in zip(values, allowances))


def summarize(report):
    last=report['trace'][-1]; allowances=last['allowances']
    if any('observed' not in c for c in report['cell_records']):
        return dict(nodes=report['radial_line_evaluations'],leaves=len(report['cell_records']),
                    per_cell_observations_saved=False,
                    embedded_to_allowance=[e/a if a else 0 for e,a in zip(last['embedded_observations'],allowances)])
    bank=max(range(2), key=lambda b: last['embedded_observations'][b]/allowances[b] if allowances[b] else 0)
    cells=sorted(report['cell_records'],key=lambda c:c['observed'][bank],reverse=True)
    total=sum(c['observed'][bank] for c in cells)
    cumulative=0; count80=0
    for c in cells:
        cumulative+=c['observed'][bank];count80+=1
        if cumulative>=.8*total:break
    triangles={str(i):sum(c['observed'][bank] for c in cells if c['triangle']==i)/total for i in range(3)}
    top=[dict(cell=key(c),fraction=c['observed'][bank]/total,
              observed=c['observed'],direction=c.get('directions',c.get('diagnostic'))) for c in cells[:3]]
    return dict(nodes=report['radial_line_evaluations'],leaves=len(cells),worst_bank=bank,
                embedded_to_allowance=[e/a if a else 0 for e,a in zip(last['embedded_observations'],allowances)],
                top3_fraction=sum(c['fraction'] for c in top),cells_for_80_percent=count80,
                triangle_fractions=triangles,top3=top)


def refinements(report, known=None):
    result={}
    for i,r in enumerate(report['refinements']):
        parent=r.get('parent_observed')
        if parent is None and r.get('parent_diagnostic'):
            parent=[8*v for v in r['parent_diagnostic']['diagonal']]
        if parent is None and known and tuple(r['parent']) in known:
            parent=known[tuple(r['parent'])]['parent']
        if parent is None:continue
        # Independent reconstruction from consecutive global observation sums.
        before,after=report['trace'][i:i+2]
        child=[a-b+p for a,b,p in zip(after['embedded_observations'],before['embedded_observations'],parent)]
        if 'child_observed' in r:
            assert all(abs(a-b)<1e-10*max(abs(p),1e-30) for a,b,p in zip(child,r['child_observed'],parent))
        result[tuple(r['parent'])]=dict(axis=r['axis'],parent=parent,children=child,
                                      new_nodes=r['new_nodes'])
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise ValueError('Fresh output required')
    cases=[];pins=[]
    for slug in ('half-ball','small-half-ball','tangent-Gamma'):
        reports={}
        for variant,prefix in [('reference','genesis-ray-enclosure'),('eager','genesis-directional'),
                               ('lazy','genesis-directional-lazy'),('reuse','genesis-reused-direction'),
                               ('diagnose','genesis-direction-diagnose')]:
            path=Path('/workspace')/f'{prefix}-{slug}-01'/'report.json'
            pins.append(pin(path));reports[variant]=json.loads(path.read_bytes())
        summaries={v:summarize(r) for v,r in reports.items()}
        maps={v:refinements(reports[v]) for v in ('lazy','reuse')}
        maps['eager']=refinements(reports['eager'],{**maps['lazy'],**maps['reuse']})
        comparisons=[]
        for left,right in [('eager','reuse'),('lazy','reuse')]:
            for k in sorted(maps[left].keys() & maps[right].keys()):
                a,b=maps[left][k],maps[right][k]
                if a['axis']==b['axis']:continue
                allowance=reports['reuse']['trace'][-1]['allowances']
                an,bn=normalized(a['children'],allowance),normalized(b['children'],allowance)
                comparisons.append(dict(cell=k,left_variant=left,right_variant=right,
                                        left=a,right=b,right_to_left_child_error=bn/an))
        d=reports['diagnose'];bank=summaries['diagnose']['worst_bank']
        dc=[c for c in d['cell_records'] if c['diagnostic']]
        total=sum(c['observed'][bank] for c in d['cell_records'])
        tmajor=sum(c['observed'][bank] for c in dc if c['diagnostic']['t'][bank]>c['diagnostic']['s'][bank])/total
        bands={}
        for c in dc:
            for b in c['diagnostic']['band_indicators']:
                target=bands.setdefault(b['band'],dict(s=0.,t=0.))
                for axis in ('s','t'):target[axis]+=b[axis][bank]
        cases.append(dict(case=slug,variants=summaries,matched_opposite_directions=comparisons,
                          baseline_observed_fraction_with_t_dominant=tmajor,
                          baseline_mixed_indicator_sums_by_band=bands,
                          saved_nodal_values_available=False))
    result=dict(base_commit='0fa57a9c45df464491fcb551d27e147cdfdc8ffd',cases=cases,input_pins=pins,
                new_radial_evaluations=0,new_adaptive_runs=0,
                status='NO_PHYSICAL_RESIDUAL_TEST_WITHOUT_MISSING_NODE_VALUES',
                scope='Saved scalar fixture results; no proof of optimal mesh or impossibility under cap')
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    for c in cases:
        print(c['case'],json.dumps(dict(reuse=c['variants']['reuse'],
            matched_opposite_directions=[dict(cell=x['cell'],left=x['left_variant'],
                axis=x['left']['axis'],reuse_axis=x['right']['axis'],ratio=x['right_to_left_child_error'])
                for x in c['matched_opposite_directions']],tmajor=c['baseline_observed_fraction_with_t_dominant'],
                bands=c['baseline_mixed_indicator_sums_by_band'])))


if __name__=='__main__':main()
