"""Independent exact-contraction audit of deterministically selected real nodes.

This is a sample audit, not a proof for every angular node or a new angular
acceptance. It rebuilds rational targets without the precision runtime's fields.
"""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import time

import numpy as np

from .control import write, scope
from analysis.genesis_volume_integral_validation.source_control import (
    ROOT,load_authenticated_inputs,support_source_inputs,seal,pin,require,
)
from analysis.genesis_volume_integral_validation.actual_control import _native,_ratios
from analysis.genesis_volume_integral_validation.independent_integrals import true_source_integral_enclosures
from tectonics.genesis_volume_exact_lambda_bulk import prepare_source,integrate_line,line_to_dict
from tectonics.genesis_volume_precision_integrals import prepare_precision_integrals
from tectonics.genesis_volume_rational_fields import fraction


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    require(args.output.is_absolute() and args.output.is_relative_to(Path('/workspace'))
        and not args.output.is_relative_to(ROOT) and not args.output.exists(),'Use a fresh external output')
    request=json.loads((args.control/'request.json').read_bytes())
    for name,expected in request['source_pins'].items():
        # Test/report metadata may be extended after a pilot; runtime files
        # must remain exactly the implementation that generated its nodes.
        if name.startswith('tectonics/'):
            require(pin(ROOT/name)==expected,'Pilot runtime source differs: '+name)
    inputs=load_authenticated_inputs()
    args.output.mkdir()
    rows=[]
    for ordinal,item in enumerate(inputs['supports']):
        raw=support_source_inputs(item['case'],item['label'])
        options=dict(added_basis=item['support']['original_declared_added_basis'],
            parent_basis=item['support']['original_declared_parent_basis'],probes=item['support']['declared_probes'])
        source=prepare_source(raw);runtime=prepare_precision_integrals(source,raw,**options)
        path=args.control/f'case_{ordinal:03d}'/'nodes.jsonl'
        candidates=[json.loads(line) for line in path.read_text().splitlines()]
        positive=[entry for entry in candidates if entry['fallback_count']>0]
        require(bool(positive),'Pilot contains no eligible positive sample')
        selected=[max(positive,key=lambda e:e['fallback_count']),positive[-1]]
        require(selected[0]['N']!=selected[1]['N'],'Need two distinct actual nodes')
        for sample,entry in enumerate(selected):
            start=time.monotonic()
            N=np.array([_native(Fraction(int(a),int(b))) for a,b in entry['N']],np.longdouble)
            result=runtime.evaluate(N)
            witness=line_to_dict(integrate_line(source,N,**options))
            targets=true_source_integral_enclosures(raw,N,witness,**options)
            require(tuple(targets)==result.names,'Independent target schema differs')
            misses=[]
            for name,lo,hi,v,e in zip(result.names,result.enclosure.lower,result.enclosure.upper,result.values,result.errors):
                a,b=targets[name]
                if not fraction(lo)<=a<=b<=fraction(hi) or max(abs(fraction(v)-a),abs(fraction(v)-b))>fraction(e):
                    misses.append(name)
            row=dict(case=ordinal,sample=sample,component_count=len(result.names),misses=misses,
                pointwise_failures=list(result.unresolved_names),fallback_count=len(result.fallback_names),
                N=_ratios(N),node_file_pin=pin(path),seconds=time.monotonic()-start,**scope())
            write(args.output/f'case_{ordinal:03d}_sample_{sample}.json',seal(row))
            rows.append(row)
            print(json.dumps(row),flush=True)
            require(not misses,'Independent arbitrary-node containment failed')
    write(args.output/'report.json',seal(dict(samples=rows,independent_checked_components=sum(r['component_count'] for r in rows),
        selection='maximum fallback count (first tie), then last positive fallback node in each pilot trace',
        all_angular_nodes_independently_audited=False,**scope())))


if __name__=='__main__':
    main()
