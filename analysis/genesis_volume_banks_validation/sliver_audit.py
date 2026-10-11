"""Prove whether midpoint quadrature can ever sample a surviving Gamma sliver."""
import argparse,json,hashlib
from fractions import Fraction as F
from pathlib import Path
import numpy as np
from analysis.genesis_volume_integral_validation.source_control import load_authenticated_inputs,support_source_inputs
from analysis.genesis_volume_banks_validation.diagnose import positive_bank
from tectonics.genesis_volume_bank_bounds import PreparedBankBounds
from tectonics.genesis_volume_event_partition import source_events,ratio
from tectonics.genesis_volume_rational_fields import fraction

def pin(p):return dict(path=str(p),bytes=p.stat().st_size,raw_sha256=hashlib.sha256(p.read_bytes()).hexdigest())

def weight_minimum(orders):
    weights=[]
    for order in orders:
        nodes=tuple(map(fraction,(np.polynomial.legendre.leggauss(order)[0]+1)/2))
        for s in nodes:
            for t in nodes:weights.extend((1-s,s*(1-t),s*t))
    return min(weights)

def endpoint_node_upper(values,remaining,wmin):
    # Midpoint descendants have barycentric vertices on a 2^-depth lattice.
    # The nonnegative original vertices span at most an edge. Any nondegenerate
    # leaf therefore has a vertex with negative-side barycentric mass >=2^-depth.
    # Every Duffy node gives each leaf vertex at least wmin. Consequently its
    # mass on negative original vertices is >= wmin*2^-remaining.
    negative=[-v for v in values if v<0]
    if not negative or max(values)<=0:return None
    delta=max(values);m=min(negative)
    return delta-(delta+m)*wmin/F(2**remaining)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--control',type=Path,required=True)
    parser.add_argument('--case',type=int,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();assert 0<=args.case<4 and not args.output.exists()
    item=load_authenticated_inputs()['supports'][args.case];raw=support_source_inputs(item['case'],item['label'])
    g=PreparedBankBounds(raw);events,_=source_events(raw)
    path=args.control/f'case_{args.case:03d}'/'angular.json';a=json.loads(path.read_bytes())
    wmin=weight_minimum(a['angular_orders']);rows=[]
    for index,cell in enumerate(a['final_cells']):
        vertices=tuple(tuple(F(*map(int,v)) for v in row) for row in cell['vertices'])
        remaining=a['max_depth']-cell['depth']
        for k,side in enumerate((-1,1)):
            if F(*map(int,cell['fine_bank_volume'][k]))!=0 or F(*map(int,cell['upper_bank_volume'][k]))<=0:continue
            endpoints={name:[side*events[name].evaluate(n) for n in vertices] for name in ('Gamma_cohort_lower','Gamma_cohort_upper')}
            proofs=[]
            for name,values in endpoints.items():
                other=next(v for n,v in endpoints.items() if n!=name)
                bound=endpoint_node_upper(values,remaining,wmin)
                if bound is not None and bound<=0 and max(other)<=0:
                    proofs.append(dict(event=name,vertex_values=list(map(ratio,values)),
                        other_endpoint_vertex_values=list(map(ratio,other)),
                        all_future_quadrature_endpoint_values_upper=ratio(bound)))
            witness=next((n for n in vertices if positive_bank(g,n,side)),None)
            rows.append(dict(cell=index,side=side,remaining_midpoint_depth=remaining,
                positive_source_witness=None if witness is None else list(map(ratio,witness)),
                cannot_sample_this_bank_under_remaining_midpoint_budget=bool(proofs),proofs=proofs))
    result=dict(format='genesis-midpoint-Gamma-sliver-reachability-1',case=args.case,
        angular_pin=pin(path),auditor_pin=pin(Path(__file__)),minimum_literal_Duffy_vertex_weight=ratio(wmin),
        rows=rows,missing_banks=len(rows),
        proven_unreachable_positive_banks=sum(bool(r['proofs']) and r['positive_source_witness'] is not None for r in rows),
        applies_only_to_the_recorded_mesh_and_midpoint_descendants=True,
        positive_banks_not_reclassified_as_empty=True)
    args.output.mkdir()
    with (args.output/'report.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','minimum_literal_Duffy_vertex_weight')}))

if __name__=='__main__':main()
