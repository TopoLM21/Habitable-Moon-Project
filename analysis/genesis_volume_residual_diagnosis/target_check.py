"""One authorized 33-ray parent replay, preserving every nodal enclosure."""
from fractions import Fraction as F
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np

from analysis.genesis_volume_directional_validation.experiment import (
    DirectionalExperiment, Cell, BASE, cell_key, exact, fixture,
)
from analysis.genesis_volume_residual_diagnosis.residual import residual


def grids(experiment, cell):
    result={}; records=[]
    for n in (3,5):
        groups={}
        for N,weight,label in experiment.plans[cell_key(cell),(n,n)]:
            entries=groups.setdefault(label,[])
            t,s=divmod(len(entries),n)
            sw=experiment.rules[n][s][1];tw=experiment.rules[n][t][1]
            jac=weight/(sw*tw);sample=experiment.samples[N]
            mids=[(exact(lo)+exact(hi))/2 for lo,hi in zip(sample.lower,sample.upper)]
            entries.append([jac*m for m in mids])
            records.append(dict(order=n,t=t,s=s,band=label,N=list(map(str,N)),
                                weight=str(weight),jacobian=str(jac)))
        result[n]={label:[[values[t*n+s] for s in range(n)] for t in range(n)]
                   for label,values in groups.items() if len(values)==n*n}
        if len(result[n])!=len(groups):raise ValueError('Incomplete stable band')
    if result[3].keys()!=result[5].keys():raise ValueError('Different band topology')
    return result,records


def diagnose(experiment, cell):
    before=(experiment.nodes,experiment.requests,len(experiment.planner.cache))
    data,records=grids(experiment,cell)
    x5,_=zip(*experiment.rules[5]);x3,w3=zip(*experiment.rules[3])
    directions={a:[F(),F()] for a in ('s','t')};bands=[]
    for label in data[3]:
        sides=[]
        for side in range(2):
            r=residual([[v[side] for v in row] for row in data[5][label]],
                       [[v[side] for v in row] for row in data[3][label]],x5,x3,w3)
            for axis in directions:directions[axis][side]+=r[axis]
            sides.append({k:[[str(v) for v in row] for row in val] if k=='grid' else str(val)
                          for k,val in r.items()})
        bands.append(dict(band=label,sides=sides))
    assert before==(experiment.nodes,experiment.requests,len(experiment.planner.cache))
    return directions,bands,records


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise ValueError('Fresh output required')
    if np.finfo(np.longdouble).nmant!=63:raise ValueError('nmant=63 required')
    e=DirectionalExperiment(fixture(.25,9.25))
    c=Cell(0,F(1,2),F(1),F(1,2),F(1),2)
    assert e.cost([c],BASE)==33
    c=e.compute(c,BASE)
    assert e.nodes==33 and e.requests==34 and e.mixed_new_nodes==0
    directions,bands,requests=diagnose(e,c)
    # Positive bank is exactly empty for this tangent fixture.
    assert directions['s'][1]==directions['t'][1]==0
    selected=max(('s','t'),key=lambda a:directions[a][0])
    samples=[dict(N=list(map(str,N)),lower=[str(exact(v)) for v in sample.lower],
                  upper=[str(exact(v)) for v in sample.upper]) for N,sample in e.samples.items()]
    payload=dict(base_commit='64a8913055d90c0afb1ab1ca7bfb5b589ed0ffe1',
                 source_geometry=e.raw,parent=['0','1/2','1','1/2','1'],
                 diagnostic_radial_evaluations=e.nodes,diagnostic_scalar_bank_calls=2*e.nodes,
                 requests=e.requests,cache_hits=e.requests-e.nodes,
                 indicator_additional_radial_calls=0,selected_axis=selected,
                 required_axis_from_prior_children='t',addressed_gate_passed=selected=='t',
                 directions={a:list(map(str,values)) for a,values in directions.items()},
                 directions_float={a:list(map(float,values)) for a,values in directions.items()},
                 coarse=list(map(str,map(exact,c.coarse))),fine=list(map(str,map(exact,c.fine))),
                 observed=list(map(float,c.observed)),root_upper=list(map(str,e.root_upper(c))),
                 bands=bands,node_requests=requests,samples=samples)
    paths=[Path(__file__),Path('analysis/genesis_volume_residual_diagnosis/residual.py'),
           Path('analysis/genesis_volume_directional_validation/experiment.py')]
    payload['source_sha256']={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    with args.output.open('x') as f:json.dump(payload,f,indent=2);f.write('\n')
    print(json.dumps({k:payload[k] for k in ('diagnostic_radial_evaluations','indicator_additional_radial_calls',
                       'selected_axis','addressed_gate_passed','directions_float')}))


if __name__=='__main__':main()
