"""Fresh precision control using authenticated original Genesis, never evolution.

Archived independent exact targets are hash-bound through the integral package.
The angular rule, radii and 2e-6 allowance remain the original control's values.
Only a pilot may use a lower work cap, and cannot claim full-control completion.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

from analysis.genesis_volume_integral_validation.source_control import (
    ROOT, load_authenticated_inputs, support_source_inputs, canonical, seal, pin, require,
)
from analysis.genesis_volume_integral_validation.actual_control import ANGULAR_OPTIONS, _native, _ratios
from analysis.genesis_volume_integral_validation.angular_control import integrate_angular, NumericalLineBoundRefusal
from analysis.genesis_volume_operand_validation.independent_coefficients import exact_source_coefficients
from tectonics.genesis_volume_exact_lambda_bulk import prepare_source
from tectonics.genesis_volume_precision_integrals import prepare_precision_integrals, MASS_REPAIR_TRIGGER
from tectonics.genesis_volume_rational_fields import fraction

PACKAGE=ROOT/'analysis/genesis_volume_integral_validation/evidence'
MANIFEST_RAW='48fc182155882931d607846fca6ae6594e8fa4cfe90d8516970aa9c0fdfa1634'


def write(path,body):
    with path.open('xb') as stream:
        stream.write(canonical(body)+b'\n')


def scope():
    return dict(history_advanced=False,history_reinitialized=False,automated_goal_enabled=False,
        physical_creation_admitted=False,physical_release_admitted=False,angular_accuracy_certified=False,
        source_bounds_added_to_acceptance_allowance=False)


def case(ordinal,item,output,cap):
    start=time.monotonic()
    directory=Path(output)/f'case_{ordinal:03d}'
    directory.mkdir()
    raw=support_source_inputs(item['case'],item['label'])
    options=dict(added_basis=item['support']['original_declared_added_basis'],
        parent_basis=item['support']['original_declared_parent_basis'],probes=item['support']['declared_probes'])
    source=prepare_source(raw)
    parent=(source.parent_stiffness.tobytes(),source.parent_mass.tobytes())
    saved=tuple(np.asarray(raw['previous_local_bulk_blocks'][k],np.float64).tobytes()
                for k in ('parent_stiffness','parent_mass'))
    require(parent==saved,'Retained parent K/M differs')
    evaluator=prepare_precision_integrals(source,raw,**options)
    N=np.array([_native(Fraction(v)) for v in item['report']['line_witness_barycentric_exact_binary_ratios']],np.longdouble)
    selected=evaluator.evaluate(N)
    require(selected.interval is not None,'Selected source line became empty')
    actual=evaluator.fields.evaluate(selected.interval)
    expected=exact_source_coefficients(raw,N,fraction(selected.interval.lambda_mid),fraction(selected.interval.lambda_halfwidth))
    require(all(np.array_equal(actual[k],expected[k]) for k in actual),'Independent source coefficient mismatch')
    manifest=json.loads((PACKAGE/'package_manifest.json').read_bytes())
    proof_rel=f'actual_control/case_{ordinal:03d}_{item["label"]}/true_source_certificate.json'
    entry=next(e for e in manifest['files'] if e['path']==proof_rel)
    proof_path=PACKAGE/proof_rel
    require(pin(proof_path)=={k:entry[k] for k in ('bytes','raw_sha256')},'Saved independent target hash differs')
    proof=json.loads(proof_path.read_bytes())
    misses=[]
    for name,lo,hi,v,error in zip(selected.names,selected.enclosure.lower,selected.enclosure.upper,selected.values,selected.errors):
        target=proof['components'][name]['true_source_integral_enclosure']
        a,b=(Fraction(int(target[k][0]),int(target[k][1])) for k in ('lower','upper'))
        if not fraction(lo)<=a<=b<=fraction(hi) or max(abs(fraction(v)-a),abs(fraction(v)-b))>fraction(error):
            misses.append(name)
    require(not misses,'Independent source target containment failed: '+str(misses[:3]))
    selected_report=dict(component_count=len(selected.names),source_target_misses=misses,
        independent_coefficient_equality=True,own_relative_failures=list(selected.unresolved_names),
        fallback_names=list(selected.fallback_names),original_N=_ratios(N),component_names=list(selected.names),
        values=_ratios(selected.values),errors=_ratios(selected.errors),
        lower=_ratios(selected.enclosure.lower),upper=_ratios(selected.enclosure.upper),
        saved_independent_certificate_pin=pin(proof_path),seconds=time.monotonic()-start,**scope())
    write(directory/'selected.json',seal(selected_report))
    print(json.dumps(dict(event='selected',case=ordinal,components=len(selected.names),
        misses=len(misses),failures=len(selected.unresolved_names),fallback=len(selected.fallback_names),
        seconds=time.monotonic()-start)),flush=True)
    angular=None
    counts=dict(attempted=0,completed=0,positive=0,pointwise_failed_nodes=0,fallback_components=0)
    failed_names=set()
    digest=hashlib.sha256()
    if cap:
        with (directory/'nodes.jsonl').open('x') as nodes:
            def callback(node):
                counts['attempted']+=1
                try:
                    line=evaluator.evaluate(node)
                except ValueError as error:
                    raise NumericalLineBoundRefusal(str(error),barycentric=node) from error
                counts['completed']+=1
                counts['positive']+=int(line.interval is not None)
                counts['pointwise_failed_nodes']+=int(bool(line.unresolved_names))
                counts['fallback_components']+=len(line.fallback_names)
                failed_names.update(line.unresolved_names)
                literal=dict(N=_ratios(node),values=_ratios(line.values),errors=_ratios(line.errors),
                             lower=_ratios(line.enclosure.lower),upper=_ratios(line.enclosure.upper))
                digest.update(canonical(literal)+b'\n')
                nodes.write(json.dumps(dict(N=literal['N'],unresolved=list(line.unresolved_names),
                    fallback_count=len(line.fallback_names),value_error_sha256=hashlib.sha256(canonical(literal)).hexdigest()))+'\n')
                if counts['completed']%100==0:
                    nodes.flush()
                    print(json.dumps(dict(event='nodes',case=ordinal,**counts,seconds=time.monotonic()-start)),flush=True)
                return line.values,line.errors
            try:
                angular=integrate_angular(raw,callback,evaluator.names,
                    **{**ANGULAR_OPTIONS,'max_line_evaluations':cap}).to_dict()
            except NumericalLineBoundRefusal as refusal:
                angular=refusal.to_dict()
            write(directory/'angular.json',seal(angular))
    require(parent==(source.parent_stiffness.tobytes(),source.parent_mass.tobytes()),'Parent K/M changed')
    report=seal(dict(case=ordinal,selected=selected_report,angular=angular,node_counts=counts,
        pointwise_unresolved_names=sorted(failed_names),node_trace_sha256=digest.hexdigest(),
        max_line_evaluations=cap,full_original_work_cap=(cap==3400),
        original_radius_m=item['support']['seed_radius_m'],seconds=time.monotonic()-start,**scope()))
    write(directory/'report.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--angular-cap',type=int,default=0,choices=(0,340,3400))
    args=parser.parse_args()
    output=args.output.absolute()
    require(output.is_relative_to(Path('/workspace')) and not output.is_relative_to(ROOT)
        and not output.exists(),'Use a fresh external output directory')
    require(np.finfo(np.longdouble).nmant==63,'Requires 63 fraction bits')
    require(shutil.disk_usage('/workspace').free>512*1024**2,'Need 512 MiB output reserve')
    require(pin(PACKAGE/'package_manifest.json')['raw_sha256']==MANIFEST_RAW,'Frozen integral manifest differs')
    manifest=json.loads((PACKAGE/'package_manifest.json').read_bytes())
    frozen=manifest['source_sha256']
    require(len(frozen)==469 and all(pin(ROOT/k)['raw_sha256']==v for k,v in frozen.items()),'Frozen source changed')
    sources={str(p.relative_to(ROOT)):pin(p) for family in ('tectonics','analysis/genesis_volume_precision_validation')
             for p in (ROOT/family).glob('*precision*.py')}
    sources['tectonics/genesis_volume_rational_fields.py']=pin(ROOT/'tectonics/genesis_volume_rational_fields.py')
    sources['analysis/genesis_volume_precision_validation/control.py']=pin(Path(__file__))
    sources['tests/test_genesis_volume_precision_integrals.py']=pin(ROOT/'tests/test_genesis_volume_precision_integrals.py')
    inputs=load_authenticated_inputs()
    output.mkdir()
    write(output/'authentication.json',inputs['authentication'])
    write(output/'request.json',dict(source_pins=sources,angular_cap=args.angular_cap,
        original_angular_options=ANGULAR_OPTIONS,estimated_output_upper_bytes=512*1024**2,
        python_version=sys.version,numpy_version=np.__version__,longdouble_nmant=np.finfo(np.longdouble).nmant,
        internal_mass_repair_trigger_ratio=[str(MASS_REPAIR_TRIGGER.numerator),str(MASS_REPAIR_TRIGGER.denominator)],
        mass_fallback_component_cap_per_line=64,**scope()))
    items=[{k:v for k,v in item.items() if k!='prior_operand_proofs'} for item in inputs['supports']]
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(case,i,item,str(output),args.angular_cap) for i,item in enumerate(items)]
        reports=[future.result() for future in as_completed(futures)]
    require(all(pin(ROOT/k)['raw_sha256']==v for k,v in frozen.items()),'Frozen source changed during run')
    require(all(pin(ROOT/k)==v for k,v in sources.items()),'New source changed during run')
    write(output/'report.json',seal(dict(cases=sorted(reports,key=lambda c:c['case']),
        frozen_469_sources_unchanged=True,new_sources_unchanged=True,**scope())))


if __name__=='__main__':
    main()
