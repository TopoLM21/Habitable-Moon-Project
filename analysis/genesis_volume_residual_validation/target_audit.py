"""Audit serialized target nodes and propagate their uncertainty independently.

No fiber evaluator, experimental planner or residual implementation imported.
"""
from fractions import Fraction as F
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np


def audit(path):
    report=json.loads(path.read_bytes())
    samples={tuple(map(F,r['N'])):r for r in report['samples']}
    assert len(samples)==len(report['samples'])==33
    raypath=Path('/workspace/genesis-reused-direction-tangent-Gamma-01/ray_certificates.json')
    decode=lambda x:F(*map(int,x))
    rays={(r['triangle'],decode(r['t'])):r for r in json.loads(raypath.read_bytes())}
    rules={n:tuple(zip(map(lambda v:F(float(v)),(np.polynomial.legendre.leggauss(n)[0]+1)/2),
                       map(lambda v:F(float(v)),np.polynomial.legendre.leggauss(n)[1]/2))) for n in (3,5)}
    area=F(5,18)  # det in barycentric N1,N2 of center,(1,0,0),(0,1,0).
    values={3:{},5:{}};errors={3:{},5:{}};seen=set()
    for request in report['node_requests']:
        n,t,s=(request[k] for k in ('order','t','s'))
        xt,tw=rules[n][t];xs,sw=rules[n][s]
        ray=rays[0,F(1,2)+xt/2]
        bands=[b for b in ray['intervals'] if b['integrated'] and b['kind']!='root_box']
        assert len(bands)==1
        lo,hi=decode(bands[0]['lo']),decode(bands[0]['hi']);ss=F(1,2)+xs/2
        rho=lo+(hi-lo)*(1-(1-ss)**2)
        start=list(map(decode,ray['start']));end=list(map(decode,ray['end']))
        N=tuple((1-rho)*a+rho*b for a,b in zip(start,end));seen.add(N)
        jac=area*rho*(hi-lo)*2*(1-ss)/4
        assert N==tuple(map(F,request['N'])) and jac==F(request['jacobian'])
        assert jac*sw*tw==F(request['weight'])
        r=samples[N];lower,upper=F(r['lower'][0]),F(r['upper'][0]);assert lower<=upper
        assert F(r['lower'][1])==F(r['upper'][1])==0
        values[n][t,s]=jac*(lower+upper)/2;errors[n][t,s]=jac*(upper-lower)/2
    assert seen==samples.keys() and len(report['node_requests'])==34
    assert len(values[3])==9 and len(values[5])==25
    basis=[]
    for x,_ in rules[3]:
        row=[]
        for j,(xj,_) in enumerate(rules[5]):
            v=F(1)
            for k,(xk,_) in enumerate(rules[5]):
                if j!=k:v*=(x-xk)/(xj-xk)
            row.append(v)
        basis.append(row)
    residual={};bounds={}
    for t in range(3):
        for s in range(3):
            residual[t,s]=values[3][t,s]-sum(basis[t][j]*basis[s][i]*values[5][j,i] for j in range(5) for i in range(5))
            bounds[t,s]=errors[3][t,s]+sum(abs(basis[t][j]*basis[s][i])*errors[5][j,i] for j in range(5) for i in range(5))
    weight_sum=sum(w for x,w in rules[3]);w=[w/weight_sum for x,w in rules[3]]
    mean=sum(w[t]*w[s]*residual[t,s] for t in range(3) for s in range(3))
    mean_bound=sum(w[t]*w[s]*bounds[t,s] for t in range(3) for s in range(3))
    sv=sum(w[s]*abs(sum(w[t]*residual[t,s] for t in range(3))-mean) for s in range(3))
    tv=sum(w[t]*abs(sum(w[s]*residual[t,s] for s in range(3))-mean) for t in range(3))
    assert sv==F(report['directions']['s'][0]) and tv==F(report['directions']['t'][0])
    assert tv-sv>4*mean_bound  # Each centered marginal L1 has error <=2*mean_bound.
    assert report['diagnostic_radial_evaluations']==33 and report['diagnostic_scalar_bank_calls']==66
    return dict(target_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                ray_records_sha256=hashlib.sha256(raypath.read_bytes()).hexdigest(),
                unique_nodes=33,requests=34,cache_hits=1,scalar_bank_calls=66,
                exact_node_geometry_and_weights=True,all_saved_enclosures_ordered=True,
                s=float(sv),t=float(tv),t_to_s=float(tv/sv),axis_gap=float(tv-sv),
                axis_gap_uncertainty_bound=float(4*mean_bound),selected_axis_robust_to_saved_uncertainty=True,
                additional_source_evaluator_calls=0,
                scope='Node geometry, cost and residual propagation from saved intervals; source interval validity inherits verified radial implementation')


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=audit(a.input)
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
