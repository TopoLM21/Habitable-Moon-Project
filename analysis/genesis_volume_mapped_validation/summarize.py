"""Build a small completion report from the real four-control run and audits."""
import argparse
import hashlib
from fractions import Fraction as F
import json
from pathlib import Path
from analysis.genesis_volume_integral_validation.source_control import ROOT,pin,require


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--control',type=Path,required=True)
    parser.add_argument('--coverage-audit',type=Path,required=True);parser.add_argument('--tests',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--guard-proof',type=Path);args=parser.parse_args()
    request=json.loads((args.control/'request.json').read_bytes())
    result=json.loads((args.control/'report.json').read_bytes())
    coverage=json.loads((args.coverage_audit/'report.json').read_bytes())
    auth=json.loads((args.control/'authentication.json').read_bytes())
    require(len(result['cases'])==4 and len(coverage['cases'])==4,'Need all four original cases')
    guard=None if args.guard_proof is None else json.loads(args.guard_proof.read_bytes())
    for k,v in request['source_pins'].items():
        actual=pin(ROOT/k)
        if actual==v:continue
        require(guard is not None and k==guard['runtime_path'] and guard['original_pin']==v
            and guard['final_pin']==actual and guard['only_declared_refusal_added']
            and guard['all_original_four_controls_numerically_unchanged'],
            'Current numerical sources differ without an exact compatibility proof')
        from analysis.genesis_volume_mapped_validation.guard_compatibility import GUARD
        current=(ROOT/k).read_bytes();snapshot=(args.control/'source_snapshots'/k).read_bytes()
        require(current.count(GUARD.encode())==1 and current.replace(GUARD.encode(),b'')==snapshot,
            'Runtime differs beyond the proved refusal')
    rows=[]
    for case,audit in zip(result['cases'],coverage['cases']):
        require(case['case']==audit['case'] and audit['initial_support_completeness_proof_verified']
            and case['coverage_pin']==audit['certificate_pin'],'Coverage audit mismatch')
        a=case['angular']
        directory=args.control/f"case_{case['case']:03d}"
        c=json.loads((directory/'coverage.json').read_bytes())
        bounds=[tuple(F(*map(int,q)) for q in b) for b in c['coordinate_bounds']]
        node_count=non_native=maximum_bits=0
        for text in (directory/'nodes.jsonl').read_text().splitlines():
            node=json.loads(text);n=tuple(F(*map(int,q)) for q in node['N'])
            require(len(n)==3 and sum(n)==1 and min(n)>=0,'Recorded exact node leaves simplex')
            require(all(lo<=v<=hi for v,(lo,hi) in zip(n,bounds)),'Recorded node leaves initial support')
            bits=max(v.numerator.bit_length() for v in n)
            maximum_bits=max(maximum_bits,bits);non_native+=int(bits>64);node_count+=1
        require(node_count==case['node_counts']['completed'],'Node trace count differs')
        rows.append(dict(case=case['case'],radius_m=case['original_radius_m'],
            nodes=case['node_counts']['completed'],
            all_recorded_nodes_exactly_in_initial_simplex=True,node_file_pin=pin(directory/'nodes.jsonl'),
            non_native_64bit_nodes=non_native,
            maximum_exact_coordinate_numerator_bits=maximum_bits,positive_nodes=case['node_counts']['positive'],
            pointwise_failed_nodes=case['node_counts']['pointwise_failed_nodes'],
            initial_source_coverage_independently_verified=True,
            exact_node_mapping_certified=a['angular_node_mapping_accuracy_certified'],
            event_partition_initial_leaves=a['event_partition_initial_leaves'],
            source_budget_failures=a['source_uncertainty_failed_component_count'],
            observed_angular_failures=a['observed_angular_failed_component_count'],
            failed_components=a['failed_component_count'],status=a['status'],reason=a['reason'],
            maximum_depth=a['maximum_evaluated_depth'],
            independent_checked_components=sum(s['components'] for s in case['independent_samples']),
            independent_misses=sum(len(s['misses']) for s in case['independent_samples']),
            seconds=case['seconds']))
    history={k:auth[k] for k in ('source_time_myr','cohort_count','accepted_interval_count','all_800_cohort_memory_sha256')}
    history['raw_history_files_rechecked']={}
    for name,record in auth['authoritative_raw_history_pins'].items():
        require(pin(Path(record['path']))=={k:record[k] for k in ('bytes','raw_sha256')},'Accepted history changed')
        history['raw_history_files_rechecked'][name]=record
    manifest=json.loads((ROOT/'analysis/genesis_volume_integral_validation/evidence/package_manifest.json').read_bytes())
    require(all(hashlib.sha256((ROOT/k).read_bytes()).hexdigest()==v for k,v in manifest['source_sha256'].items()),'Frozen evidence source changed')
    summary=dict(format='genesis-certified-cover-exact-map-validation-1',
        status='COMPLETED_FOUR_ORIGINAL_CONTROLS',cases=rows,history=history,
        total_nodes=sum(r['nodes'] for r in rows),
        independent_checked_components=sum(r['independent_checked_components'] for r in rows),
        frozen_source_count=len(manifest['source_sha256']),frozen_sources_unchanged=True,
        original_tolerance=2e-6,original_options=request['original_angular_options'],
        final_runtime_source_pins={k:pin(ROOT/k) for k in request['source_pins']},
        additional_validation_source_pins={str(p.relative_to(ROOT)):pin(p) for p in
            [*sorted((ROOT/'analysis/genesis_volume_mapped_validation').glob('*.py')),
             ROOT/'tests/test_genesis_volume_mapped_semantics.py']},
        postrun_guard_compatibility=None if guard is None else dict(
            proof_path=str(args.guard_proof),proof_pin=pin(args.guard_proof),
            original_pin=guard['original_pin'],final_pin=guard['final_pin'],
            all_four_controls_unchanged_by_exact_unreachable_branch_proof=True,
            full_control_not_rerun_after_guard=True),
        initial_source_coverage_and_partition_coverage_are_separate=True,
        node_mapping='Exact rational Duffy coordinates passed unchanged into source coefficients and roots',
        old_native_mapper_certified=False,ideal_Gauss_root_errors_certified=False,
        angular_accuracy_certified=False,physical_creation_admitted=False,
        history_advanced=False,history_reinitialized=False,automated_goal_enabled=False,
        evidence={str(p):pin(p) for p in (args.control/'report.json',args.control/'request.json',
            args.coverage_audit/'report.json',args.tests)},tests=args.tests.read_text().splitlines()[-1],
        next_step='Resolve event-boundary angular convergence and establish an angular remainder bound without changing 2e-6 or physical source',
        interrupted_attempts=['/workspace/genesis-mapped-full-01: initial-depth accounting corrected',
            '/workspace/genesis-mapped-full-02: volume-envelope aggregation made outward'],
        limitations=['Embedded (3,5) differences are observations, not certified angular remainders',
            'Whole-cohort candidate volume envelope may be loose',
            'Independent integral audits sample first and last positive actual nodes per case',
            'Stage 1 and four spatial controls remain OPEN unless separately admitted'])
    require(not args.output.exists(),'Do not overwrite an existing completion report')
    args.output.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(total_nodes=summary['total_nodes'],cases=rows),ensure_ascii=False))


if __name__=='__main__':main()
