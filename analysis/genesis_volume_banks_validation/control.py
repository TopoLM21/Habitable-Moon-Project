"""Four original spatial controls with bank bounds and measure-preserving event cuts."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
from analysis.genesis_volume_integral_validation.source_control import (
    ROOT,load_authenticated_inputs,support_source_inputs,canonical,seal,pin,require,
)
from analysis.genesis_volume_integral_validation.actual_control import ANGULAR_OPTIONS
from analysis.genesis_volume_precision_validation.control import write,scope,PACKAGE,MANIFEST_RAW
from analysis.genesis_volume_banks_validation.angular import integrate_angular
from analysis.genesis_volume_banks_validation.independent_bank import audit_certificates
from tectonics.genesis_volume_bank_bounds import verify_bank_certificate
from tectonics.genesis_volume_event_mesh import event_mesh
from analysis.genesis_volume_mapped_validation.independent import independent_targets
from tectonics.genesis_volume_exact_lambda_bulk import prepare_source
from tectonics.genesis_volume_mapped_integrals import PreparedMappedIntegrals
from tectonics.genesis_volume_source_coverage import source_coverage,verify_coverage
from tectonics.genesis_volume_event_partition import partition_events,verify_partition,ratio
from tectonics.genesis_volume_rational_fields import fraction


def ratios(a):
    return [ratio(fraction(x)) for x in a]


def case(ordinal,item,output):
    start=time.monotonic(); directory=Path(output)/f'case_{ordinal:03d}';directory.mkdir()
    raw=support_source_inputs(item['case'],item['label'])
    options=dict(added_basis=item['support']['original_declared_added_basis'],
        parent_basis=item['support']['original_declared_parent_basis'],probes=item['support']['declared_probes'])
    source=prepare_source(raw)
    parent=(source.parent_stiffness.tobytes(),source.parent_mass.tobytes())
    saved=tuple(np.asarray(raw['previous_local_bulk_blocks'][k],float).tobytes() for k in ('parent_stiffness','parent_mass'))
    require(parent==saved,'Parent K/M differs')
    runtime=PreparedMappedIntegrals(source,raw,**options)
    cells,coverage=source_coverage(raw);verify_coverage(raw,coverage)
    write(directory/'coverage.json',coverage)
    _,partition=event_mesh(raw,cells,max_depth=24)
    require(event_mesh(raw,cells,max_depth=24)[1]==partition,'Event mesh replay differs')
    write(directory/'partition.json',partition)
    counts=dict(completed=0,positive=0,pointwise_failed_nodes=0,fallback_components=0)
    trace=hashlib.sha256();samples=[]
    with (directory/'nodes.jsonl').open('x') as stream:
        def callback(N):
            line=runtime.evaluate(N)
            counts['completed']+=1;counts['positive']+=int(line.interval is not None)
            counts['pointwise_failed_nodes']+=int(bool(line.unresolved_names))
            counts['fallback_components']+=len(line.fallback_names)
            literal=dict(N=ratios(N),values=ratios(line.values),errors=ratios(line.errors),
                lower=ratios(line.enclosure.lower),upper=ratios(line.enclosure.upper))
            trace.update(canonical(literal)+b'\n')
            stream.write(json.dumps(dict(N=literal['N'],positive=line.interval is not None,
                unresolved=list(line.unresolved_names),fallback_count=len(line.fallback_names),
                value_error_sha256=hashlib.sha256(canonical(literal)).hexdigest()))+'\n')
            if line.interval is not None:
                if not samples:samples.append(N)
                elif len(samples)==1:samples.append(N)
                else:samples[-1]=N
            if counts['completed']%100==0:
                stream.flush();print(json.dumps(dict(event='nodes',case=ordinal,**counts,seconds=time.monotonic()-start)),flush=True)
            return line.values,line.errors
        angular=integrate_angular(raw,callback,runtime.names,**ANGULAR_OPTIONS)
    write(directory/'angular.json',seal(angular))
    bank_certificates=[c['bank_bound_certificate'] for c in angular['final_cells']]+angular['discarded_cells']
    for cert in bank_certificates:verify_bank_certificate(raw,cert)
    bank_audit=audit_certificates(raw,bank_certificates)
    write(directory/'bank_audit.json',seal(bank_audit))
    audits=[]
    for N in samples:
        line=runtime.evaluate(N);targets=independent_targets(raw,N,line.interval,**options)
        require(tuple(targets)==line.names,'Independent target names differ')
        misses=[]
        for name,lo,hi,v,e in zip(line.names,line.enclosure.lower,line.enclosure.upper,line.values,line.errors):
            a,b=targets[name]
            if not fraction(lo)<=a<=b<=fraction(hi) or max(abs(fraction(v)-a),abs(fraction(v)-b))>fraction(e):misses.append(name)
        audits.append(dict(N=ratios(N),components=len(line.names),misses=misses))
        require(not misses,'Independent mapped source containment failed')
    require(parent==(source.parent_stiffness.tobytes(),source.parent_mass.tobytes()),'Parent K/M changed')
    report=seal(dict(case=ordinal,original_radius_m=item['support']['seed_radius_m'],
        coverage_pin=pin(directory/'coverage.json'),partition_pin=pin(directory/'partition.json'),
        angular=angular,node_counts=counts,node_trace_sha256=trace.hexdigest(),
        independent_bank_audit=bank_audit,
        independent_samples=audits,independent_selection='first and last positive actual quadrature node',
        all_nodes_independently_audited=False,parent_K_M_bytes_unchanged=True,
        seconds=time.monotonic()-start,**scope()))
    write(directory/'report.json',report)
    print(json.dumps(dict(event='case_complete',case=ordinal,counts=counts,
        angular_failed=angular['failed_component_count'],seconds=time.monotonic()-start)),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();output=args.output.absolute()
    require(output.is_relative_to(Path('/workspace')) and not output.is_relative_to(ROOT) and not output.exists(),'Fresh external output required')
    reserve=256*1024**2;free=shutil.disk_usage('/workspace').free
    require(free>reserve,'Need 256 MiB output reserve')
    require(np.finfo(np.longdouble).nmant==63,'Requires 63 fraction bits')
    require(pin(PACKAGE/'package_manifest.json')['raw_sha256']==MANIFEST_RAW,'Frozen manifest changed')
    frozen=json.loads((PACKAGE/'package_manifest.json').read_bytes())['source_sha256']
    require(len(frozen)==469 and all(pin(ROOT/k)['raw_sha256']==v for k,v in frozen.items()),'Frozen source changed')
    paths=[ROOT/'tectonics'/('genesis_volume_'+n+'.py') for n in ('precision_integrals','rational_fields','source_coverage','mapped_integrals','event_partition','bank_bounds','event_mesh')]
    paths+=list((ROOT/'analysis/genesis_volume_banks_validation').glob('*.py'))
    paths+=[ROOT/'analysis/genesis_volume_mapped_validation/independent.py']
    paths+=[ROOT/'tests/test_genesis_volume_bank_bounds.py']
    paths+=[ROOT/'analysis/genesis_volume_banks_validation/radial_exclusion_fixture.json']
    sources={str(p.relative_to(ROOT)):pin(p) for p in paths}
    inputs=load_authenticated_inputs();output.mkdir()
    snapshots=output/'source_snapshots';snapshots.mkdir()
    for p in paths:
        dest=snapshots/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
    write(output/'authentication.json',inputs['authentication'])
    write(output/'request.json',dict(source_pins=sources,original_angular_options=ANGULAR_OPTIONS,
        free_bytes_before=free,estimated_output_upper_bytes=reserve,python_version=sys.version,
        numpy_version=np.__version__,longdouble_nmant=63,**scope()))
    items=[{k:v for k,v in item.items() if k!='prior_operand_proofs'} for item in inputs['supports']]
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(case,i,item,str(output)) for i,item in enumerate(items)]
        reports=[f.result() for f in as_completed(futures)]
    require(all(pin(ROOT/k)['raw_sha256']==v for k,v in frozen.items()),'Frozen source changed during run')
    require(all(pin(ROOT/k)==v for k,v in sources.items()),'New source changed during run')
    write(output/'report.json',seal(dict(cases=sorted(reports,key=lambda c:c['case']),
        frozen_469_sources_unchanged=True,new_sources_unchanged=True,**scope())))


if __name__=='__main__':main()
