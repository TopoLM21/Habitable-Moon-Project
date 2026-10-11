"""Create a small reviewable summary only after the full precision run ends."""
import argparse
import json
from pathlib import Path

from .control import write, scope
from analysis.genesis_volume_integral_validation.source_control import pin, seal, require


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control',required=True,type=Path)
    parser.add_argument('--audit',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    request=json.loads((args.control/'request.json').read_bytes())
    report=json.loads((args.control/'report.json').read_bytes())
    audit=json.loads((args.audit/'report.json').read_bytes())
    authentication=json.loads((args.control/'authentication.json').read_bytes())
    require(request['angular_cap']==3400 and len(report['cases'])==4,'Requires the completed original work cap')
    rows=[]
    for case in report['cases']:
        angular=case['angular']
        source_failures=angular.get('source_uncertainty_failed_component_count')
        rows.append(dict(case=case['case'],original_radius_m=case['original_radius_m'],
            selected_components=case['selected']['component_count'],
            selected_source_target_misses=len(case['selected']['source_target_misses']),
            selected_own_relative_failures=len(case['selected']['own_relative_failures']),
            node_counts=case['node_counts'],pointwise_unresolved_names=case['pointwise_unresolved_names'],
            angular_status=angular['status'],source_uncertainty_failed_components=source_failures,
            angular_observed_failed_components=angular.get('observed_angular_failed_component_count'),
            source_precision_check_completed=source_failures==0 and case['node_counts']['attempted']==case['node_counts']['completed']
                and case['node_counts']['pointwise_failed_nodes']==0
                and not case['selected']['source_target_misses']
                and not case['selected']['own_relative_failures'],
            max_line_evaluations=case['max_line_evaluations'],seconds=case['seconds']))
    body=dict(format='genesis-source-precision-summary-1',
        source_precision_status='PASS_ON_COMPLETED_ORIGINAL_CONTROLS' if
            all(r['source_precision_check_completed'] for r in rows)
            and all(not s['misses'] and not s['pointwise_failures'] for s in audit['samples']) else 'OPEN',
        global_angular_and_physical_status='OPEN',cases=rows,
        source_time_myr=authentication['source_time_myr'],cohort_count=authentication['cohort_count'],
        accepted_interval_count=authentication['accepted_interval_count'],
        all_800_cohort_memory_sha256=authentication['all_800_cohort_memory_sha256'],
        authoritative_raw_history_pins=authentication['authoritative_raw_history_pins'],
        independent_additional_node_samples=len(audit['samples']),
        independent_additional_components=audit['independent_checked_components'],
        independent_additional_misses=sum(len(s['misses']) for s in audit['samples']),
        frozen_469_sources_unchanged=report['frozen_469_sources_unchanged'],
        new_sources_unchanged=report['new_sources_unchanged'],source_pins=request['source_pins'],
        full_control=dict(path=str(args.control/'report.json'),**pin(args.control/'report.json')),
        independent_audit=dict(path=str(args.audit/'report.json'),**pin(args.audit/'report.json')),
        allowance='Original signed own-component 2e-6; neither source errors nor scale floors enlarge it',
        internal_mass_repair_trigger_ratio=request['internal_mass_repair_trigger_ratio'],
        mass_fallback_component_cap_per_line=request['mass_fallback_component_cap_per_line'],
        original_archived_node_sequence_replayed=False,
        all_nodes_independently_recomputed_by_oracle=False,
        scope_note='Fresh source precision at evaluated literal nodes and their finite-rule sums only; no angular remainder or mapping certificate',
        point_one_complete=False,**scope())
    write(args.output,seal(body))


if __name__=='__main__':
    main()
