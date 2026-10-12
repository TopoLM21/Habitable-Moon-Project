"""Read only the two saved regional certificates; no numerical producer calls."""
from fractions import Fraction as F
from pathlib import Path
import hashlib
import json


def diagnose(path):
    r=json.loads(path.read_bytes());groups={};ranked=[]
    for i in r['leaves']:
        row=r['records'][i];lo,hi=map(F,row['bounds']);width=hi-lo
        gamma=list(map(F,row['ranges']['gamma']));density=list(map(F,row['ranges']['density_without_H']))
        key=row['state']+('_gamma_uncertain' if gamma[0]<=0<=gamma[1] else '_gamma_fixed')
        g=groups.setdefault(key,dict(count=0,width=F(),density_contains_zero=0))
        g['count']+=1;g['width']+=width;g['density_contains_zero']+=int(density[0]<=0<=density[1])
        ranked.append(dict(id=i,group=key,width=width,bounds=row['bounds'],box=row['box'],ranges=row['ranges']))
    ranked.sort(key=lambda x:(-x['width'],x['id']));total=sum(x['width'] for x in ranked)
    concentration={}
    for pct in (50,80,95):
        cumulative=F()
        for n,x in enumerate(ranked,1):
            cumulative+=x['width']
            if cumulative>=total*F(pct,100):concentration[str(pct)]=n;break
    # Two widest leaves in each nonzero class: eight fixed real blocks.
    chosen=[]
    for key in sorted(groups):
        if groups[key]['width']:chosen.extend(x['id'] for x in [v for v in ranked if v['group']==key][:2])
    for g in groups.values():g['width_j']=str(g.pop('width'));g['share']=float(F(g['width_j'])/total)
    for x in ranked:x['width_j']=str(x.pop('width'))
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),groups=groups,
                total_width_j=str(total),leaves_for_percent_width=concentration,top_blocks=ranked[:12],fixed_pilot_ids=chosen,
                limitation='Classification is not an additive causal decomposition. A density interval containing zero does not prove true cancellation.')


if __name__=='__main__':
    results=[diagnose(Path('/workspace')/name) for name in ['genesis-regional-work-01.json','genesis-regional-work-centered-01.json']]
    path=Path('/workspace/genesis-regional-moment-diagnosis-01.json')
    with path.open('x') as f:json.dump(results,f,indent=2);f.write('\n')
    for r in results:
        print(r['path']);print({k:dict(count=v['count'],width_j=float(F(v['width_j'])),share=v['share']) for k,v in r['groups'].items()});print(r['leaves_for_percent_width']);print('pilot',r['fixed_pilot_ids'])
        print('top',[(x['id'],x['group'],float(F(x['width_j']))) for x in r['top_blocks'][:8]])
