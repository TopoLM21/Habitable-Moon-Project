"""Exact event partition on original supports after the source precision gate."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import time

import numpy as np

from analysis.genesis_volume_integral_validation.source_control import (
    ROOT,load_authenticated_inputs,support_source_inputs,pin,require,seal,
)
from analysis.genesis_volume_precision_validation.control import write
from analysis.genesis_volume_clipped_validation.independent_geometry_audit import read_source,_angular_support
from analysis.genesis_volume_operand_validation.independent_geometry_bounds import source_line
from tectonics.genesis_volume_event_partition import source_events,partition_events,verify_partition,literal_cell


def direct_events(line):
    A,B,E=line['uu'],line['uv'],line['vv']-line['radius']**2
    g,h=line['slope'],line['intercept']
    values=dict(A=A,B=B,E=E,Delta=B*B-A*E,g=g,h=h,
                Gamma_sphere_resultant=A*h*h-2*B*h*g+E*g*g)
    for label,x in zip(('lower','upper'),line['bounds']):
        values['sphere_cohort_'+label]=A*x*x+2*B*x+E
        values['sphere_derivative_half_'+label]=A*x+B
        values['Gamma_cohort_'+label]=g*x+h
    values.update({f'N_{i}':v for i,v in enumerate(line['n'])})
    return values


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();output=args.output.absolute()
    require(output.is_relative_to(Path('/workspace')) and not output.is_relative_to(ROOT)
        and not output.exists(),'Fresh external output required')
    require(shutil.disk_usage('/workspace').free>=128*1024**2,'Need 128 MiB output reserve')
    precision=ROOT/'analysis/genesis_volume_precision_validation/summary.json'
    previous=json.loads(precision.read_bytes())
    require(previous['source_precision_status']=='PASS_ON_COMPLETED_ORIGINAL_CONTROLS',
            'Source precision must pass before the geometry lane')
    manifest=json.loads((ROOT/'analysis/genesis_volume_integral_validation/evidence/package_manifest.json').read_bytes())
    sources=dict(manifest['source_sha256'])
    for relative,expected in previous['source_pins'].items():
        require(pin(ROOT/relative)==expected,'Preceding precision implementation changed: '+relative)
        sources[relative]=expected['raw_sha256']
    for relative in ('tectonics/genesis_volume_event_partition.py',
                     'tests/test_genesis_volume_event_partition.py',
                     'analysis/genesis_volume_event_partition_validation/control.py'):
        sources[relative]=pin(ROOT/relative)['raw_sha256']
    require(all(pin(ROOT/k)['raw_sha256']==v for k,v in sources.items()),'Source byte mismatch')
    inputs=load_authenticated_inputs()
    prototype_path=Path('/workspace/genesis_volume_integral_event_prototype_20261011.facts.json')
    anchor_manifest=json.loads((ROOT/'artifact_checkpoints/20261011-genesis/manifest.json').read_bytes())
    anchor=next(e for e in anchor_manifest['files'] if e['path'].endswith('/'+prototype_path.name))
    require(pin(prototype_path)==dict(bytes=anchor['size'],raw_sha256=anchor['sha256']),
            'Preserved event prototype receipt changed')
    prototype=json.loads(prototype_path.read_bytes())
    output.mkdir()
    write(output/'request.json',dict(max_depth=2,max_cells=256,estimated_output_upper_bytes=128*1024**2,
        preceding_precision_summary_pin=pin(precision),source_sha256=sources,
        preserved_prototype_receipt_pin=pin(prototype_path)))
    write(output/'authentication.json',inputs['authentication'])
    for relative in ('tectonics/genesis_volume_event_partition.py','tests/test_genesis_volume_event_partition.py',
                     'analysis/genesis_volume_event_partition_validation/control.py'):
        target=output/'source_snapshots'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as stream:stream.write((ROOT/relative).read_bytes())
    rows=[]
    for ordinal,item in enumerate(inputs['supports']):
        start=time.monotonic();raw=support_source_inputs(item['case'],item['label'])
        roots=tuple(literal_cell(v) for v in _angular_support(read_source(raw)))
        events,_=source_events(raw);checks=0
        require({name:p.metadata() for name,p in events.items()}==prototype['cases'][ordinal]['source_events'],
                'Runtime event polynomials differ from preserved prototype')
        for N in dict.fromkeys(vertex for root in roots for vertex in root):
            expected=direct_events(source_line(raw,N))
            require(events.keys()==expected.keys(),'Event schema differs')
            for name,polynomial in events.items():
                require(polynomial.evaluate(N)==expected[name],'Independent event equation differs: '+name)
                checks+=1
        result=partition_events(raw,roots,max_depth=2,max_cells=256)
        require(verify_partition(raw,roots,result),'Canonical event partition differs')
        target=output/f'case_{ordinal:03d}.json';write(target,result)
        row=dict(case=ordinal,original_radius_m=item['support']['seed_radius_m'],root_cells=len(roots),
            leaf_cells=result['leaf_count'],split_count=result['split_count'],
            event_free_leaves=result['event_free_leaf_count'],unresolved_leaves=result['unresolved_leaf_count'],
            radial_support_classifications=dict(Counter(leaf['radial_support_classification'] for leaf in result['leaves'])),
            independent_source_event_equalities=checks,canonical_partition_readback=True,
            exact_polynomials_equal_preserved_prototype=True,
            exact_partition_of_supplied_cells=result['exact_partition_of_supplied_cells'],discarded_cells=0,
            output=dict(path=str(target),**pin(target)),seconds=time.monotonic()-start)
        rows.append(row);print(json.dumps(row),flush=True)
    require(all(pin(ROOT/k)['raw_sha256']==v for k,v in sources.items()),'Source changed during geometry control')
    body=seal(dict(format='genesis-event-partition-control-1',cases=rows,
        preserved_source_count=len(sources),all_source_bytes_unchanged=True,
        source_time_myr=inputs['authentication']['source_time_myr'],cohort_count=800,accepted_interval_count=4,
        exact_partition_of_supplied_cells=True,source_physical_support_coverage_certified=False,
        candidate_roots_use_previous_shared_geometry=True,native_node_mapping_accuracy_certified=False,
        angular_run_performed=False,angular_accuracy_certified=False,
        physical_creation_admitted=False,physical_release_admitted=False,automated_goal_enabled=False,
        history_advanced=False,history_reinitialized=False,point_one_complete=False))
    write(output/'report.json',body)


if __name__=='__main__':main()
