"""Prove a new Gamma-degeneracy refusal is unreachable in the four controls.

An affine function zero on a positive-area simplex triangle is zero on the
whole simplex plane. Testing its three vertices exactly proves the contrapositive.
The snapshot comparison permits ONLY the declared refusal block to be added.
"""
import argparse
from fractions import Fraction as F
import json
from pathlib import Path
from analysis.genesis_volume_integral_validation.source_control import ROOT,load_authenticated_inputs,support_source_inputs,pin,require,seal
from analysis.genesis_volume_precision_validation.control import write
from tectonics.genesis_volume_event_partition import source_events,ratio,signed_double_area

GUARD = '''        if not empty and all(
                F(*map(int,certs[k][side])) == 0
                for k in ('Gamma_cohort_lower','Gamma_cohort_upper')
                for side in ('lower','upper')):
            raise AngularControlRefusal(
                'Literal Gamma contains an unresolved candidate cell')
'''


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--control',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    relative='analysis/genesis_volume_mapped_validation/angular.py'
    current=(ROOT/relative).read_bytes();snapshot=args.control/'source_snapshots'/relative
    require(current.count(GUARD.encode())==1 and current.replace(GUARD.encode(),b'')==snapshot.read_bytes(),
        'Runtime changed beyond the declared Gamma refusal')
    rows=[]
    simplex=tuple(tuple(F(int(i==j)) for j in range(3)) for i in range(3))
    for i,item in enumerate(load_authenticated_inputs()['supports']):
        raw=support_source_inputs(item['case'],item['label']);events,_=source_events(raw)
        keys=('Gamma_cohort_lower','Gamma_cohort_upper')
        require(all(events[k].degree<=1 for k in keys),'Gamma endpoint is not affine')
        values={k:[events[k].evaluate(n) for n in simplex] for k in keys}
        require(any(v!=0 for vector in values.values() for v in vector),'Refusal can affect this source')
        path=args.control/f'case_{i:03d}'/'coverage.json'
        coverage=json.loads(path.read_bytes())
        cells=[tuple(tuple(F(*map(int,v)) for v in row) for row in cell) for cell in coverage['cells']]
        require(all(signed_double_area(cell)>0 and all(sum(n)==1 for n in cell) for cell in cells),
            'Initial cells leave the exact positive-area simplex')
        polygon=[tuple(F(*map(int,v)) for v in row) for row in coverage['polygon']]
        require(all((b[1]-a[1])*(v[2]-a[2])-(b[2]-a[2])*(v[1]-a[1])>=0
            for a,b in zip(polygon,polygon[1:]+polygon[:1]) for v in polygon),
            'Polygon boundary is not the oriented convex hull')
        area=sum(a[1]*b[2]-a[2]*b[1] for a,b in zip(polygon,polygon[1:]+polygon[:1]))
        require(area>0 and sum(signed_double_area(cell) for cell in cells)==area,
            'Initial fan does not cover the exact polygon area')
        rows.append(dict(case=i,convex_polygon_and_fan_coverage_exactly_verified=True,affine_endpoint_values={k:[ratio(v) for v in vector] for k,vector in values.items()},
            initial_coverage_pin=pin(path),refusal_unreachable_on_any_positive_area_descendant=True,
            reason='Affine Gamma cannot vanish identically on a positive-area descendant unless both endpoint functions vanish on the whole simplex'))
    require(not args.output.exists(),'Fresh output required');args.output.mkdir()
    write(args.output/'report.json',seal(dict(runtime_path=relative,original_pin=pin(snapshot),final_pin=pin(ROOT/relative),
        only_declared_refusal_added=True,cases=rows,all_original_four_controls_numerically_unchanged=True,
        full_control_rerun=False,proof_scope='Exact unreachable-branch proof for authenticated source operands')))
    print(json.dumps(dict(status='PASS',cases=len(rows),only_declared_refusal_added=True)))


if __name__=='__main__':main()
