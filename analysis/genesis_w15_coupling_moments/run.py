"""One gated moment series on the existing 796-leaf cover, within 120 s."""
from fractions import Fraction as F
from pathlib import Path
from math import prod
import argparse,hashlib,json,time,sys
from analysis.genesis_regional_work_validation.identity import expand
from analysis.genesis_regional_moment_validation.moments import coefficients,enclose_certificates
from analysis.genesis_w15_coupling_validation.core import contract,observed
sys.set_int_max_str_digits(30000)


def integral(poly,box):
    """Exact moments by three successive contractions, not quadrature."""
    p=poly
    for axis,(lo,hi) in enumerate(box):
        powers={k[axis] for k in p};weights={n:(hi**(n+1)-lo**(n+1))/F(n+1) for n in powers}
        out={}
        for k,v in p.items():
            key=list(k);key[axis]=0;key=tuple(key)
            out[key]=out.get(key,F())+v*weights[k[axis]]
        p=out
    return p.get((0,0,0),F())


def intersection(a,b):
    result=(max(a[0],b[0]),min(a[1],b[1]))
    if result[0]>result[1]:raise ValueError('Incompatible certified enclosures')
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise ValueError('Fresh output required')
    start=time.monotonic();paths={
        'vector':Path('/workspace/genesis-w15-coupling-vector-01.json'),
        'vector_audit':Path('/workspace/genesis-w15-coupling-vector-audit-02.json'),
        'geometry':Path('/workspace/genesis-w15-mode1-joint-01.json')}
    loaded={k:json.loads(p.read_bytes()) for k,p in paths.items()};old=loaded['vector'];geometry=loaded['geometry']
    pins={k:dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for k,p in paths.items()}
    assert pins['vector']['sha256']==loaded['vector_audit']['input_sha256']
    assert pins['geometry']['sha256']==old['input_pins']['source']['sha256']
    polys,_=expand(old['graph']);keys=list(old['totals']);byid={r['id']:r for r in old['records']}
    leaves=list(byid);assert len(leaves)==796
    width=lambda i:sum(F(byid[i]['bounds']['b_'+str(j)][1])-F(byid[i]['bounds']['b_'+str(j)][0]) for j in range(18))
    state=lambda i:geometry['work']['records'][i]['state']
    inside=sorted((i for i in leaves if state(i)=='inside_gamma_fixed'),key=lambda i:(-width(i),i))
    addressed=inside[:2];assert len(addressed)==2
    rest=sorted((i for i in leaves if i not in addressed),key=lambda i:(0 if state(i)=='outside' else 1 if state(i)=='inside_gamma_fixed' else 2,-width(i),i))
    rows={};gate=[];transforms=0;exact_moments=0
    for position,i in enumerate(addressed+rest):
        fallback=time.monotonic()-start>=120 or (position>=2 and not all(g['strictly_narrower_components']==18 for g in gate))
        before={k:tuple(map(F,v)) for k,v in byid[i]['bounds'].items()}
        row=dict(id=i,method='prior_verified_bound',bounds={k:list(map(str,v)) for k,v in before.items()})
        if not fallback:
            box=tuple(tuple(map(F,b)) for b in geometry['work']['records'][i]['box'])
            geo={k:coefficients(polys[k],box) for k in ('sphere','gamma')};transforms+=2
            q=list(map(F,geo['sphere']['coefficients']));g=list(map(F,geo['gamma']['coefficients']))
            row['geometry_certificates']=geo;fresh={};certs={}
            if min(q)>0:
                row['method']='exact_outside';fresh={k:(F(),F()) for k in keys}
            elif max(q)<=0 and (min(g)>=0 or max(g)<=0):
                sign=1 if min(g)>=0 else -1;assert min(g)!=0 or max(g)!=0
                row['method']='exact_inside_moments'
                for k in keys:
                    value=sign*integral(polys[k],box)/2;fresh[k]=(value,value);exact_moments+=1
                if position<2:
                    # Gate checks the established Bernstein mean against the fast moments.
                    for k in keys:
                        cert=coefficients(polys[k],box);transforms+=1;certs[k]=cert
                        value=sign*prod(hi-lo for lo,hi in box)*sum(map(F,cert['coefficients']))/len(cert['coefficients'])/2
                        assert fresh[k]==(value,value)
            else:
                row['method']='signed_bernstein_mask'
                for k in keys:
                    cert=coefficients(polys[k],box);transforms+=1;certs[k]=cert
                    enclosure=enclose_certificates({**geo,'density_without_H':cert},box)
                    fresh[k]=tuple(map(F,enclosure['bounds']))
            row['density_certificates']=certs
            row['moment_bounds']={k:list(map(str,v)) for k,v in fresh.items()}
            row['bounds']={k:list(map(str,intersection(before[k],fresh[k]))) for k in keys}
        rows[i]=row
        if position<2:
            improved=sum(F(row['bounds']['b_'+str(j)][1])-F(row['bounds']['b_'+str(j)][0]) < before['b_'+str(j)][1]-before['b_'+str(j)][0] for j in range(18))
            gate.append(dict(id=i,strictly_narrower_components=improved,seconds=time.monotonic()-start))
            print(json.dumps(dict(addressed_leaf=i,improved_components=improved,seconds=gate[-1]['seconds'])),flush=True)
        elif position in (151,234,383,795):print(json.dumps(dict(visited=position+1,seconds=time.monotonic()-start)),flush=True)
    ordered=[rows[i] for i in leaves];totals={k:[sum(F(row['bounds'][k][e]) for row in ordered) for e in (0,1)] for k in keys}
    forward=contract([totals['b_'+str(i)] for i in range(18)]);adjoint=totals['adjoint_work']
    result=dict(format='w15-coupling-moment-fixed-cover-1',input_pins=pins,gate=gate,vector_control_enabled=all(x['strictly_narrower_components']==18 for x in gate),
        records=ordered,totals={k:list(map(str,v)) for k,v in totals.items()},
        forward_work_j=list(map(str,forward)),mutual_work_intersection_j=list(map(str,intersection(forward,adjoint))),
        observables={k:observed(v) for k,v in totals.items()},seconds=time.monotonic()-start,
        max_compute_seconds=120,max_leaves=796,external_process_limit_seconds=180,audit_process_limit_seconds=240,
        polynomial_transforms=transforms,exact_moments=exact_moments,
        fallback_leaves=sum(row['method']=='prior_verified_bound' for row in ordered),
        mechanical_solves=0,new_partition_splits=0,physical_admission=False,source_law_selected=False)
    with args.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(seconds=result['seconds'],fallback=result['fallback_leaves'],transforms=transforms,
        adjoint_work_j=list(map(float,adjoint)),signs=[result['observables']['b_'+str(i)]['sign'] for i in range(18)])),flush=True)


if __name__=='__main__':main()
