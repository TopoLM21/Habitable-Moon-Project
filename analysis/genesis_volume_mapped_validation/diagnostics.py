"""Compare completed original controls and locate remaining angular failures."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import median
from analysis.genesis_volume_integral_validation.source_control import ROOT,pin,require


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--control',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    previous_summary=json.loads((ROOT/'analysis/genesis_volume_precision_validation/summary.json').read_bytes())
    old_pin=previous_summary['full_control'];old_path=Path(old_pin['path'])
    require(pin(old_path)=={k:old_pin[k] for k in ('bytes','raw_sha256')},'Previous completed control hash differs')
    previous=json.loads(old_path.read_bytes());current=json.loads((args.control/'report.json').read_bytes())
    rows=[]
    for old,new in zip(previous['cases'],current['cases']):
        require(old['case']==new['case'],'Case order differs')
        a,b=old['angular'],new['angular']
        require(a['component_names']==b['component_names'],'Component schema differs')
        require(a['relative_tolerance']==b['relative_tolerance']==2e-6,'Tolerance changed')
        groups=defaultdict(lambda:dict(components=0,failed_before=0,failed_after=0,source_failures_after=0,ratios=[]))
        improvements=[]
        for i,name in enumerate(b['component_names']):
            key=name.split('[')[0];g=groups[key];g['components']+=1
            ea,eb=a['observed_embedded_errors'][i],b['observed_embedded_errors'][i]
            aa,ab=a['component_allowances'][i],b['component_allowances'][i]
            g['failed_before']+=int(ea>aa);g['failed_after']+=int(eb>ab)
            source=b['coarse_source_integral_and_aggregation_error_bounds'][i]+b['fine_source_integral_and_aggregation_error_bounds'][i]
            g['source_failures_after']+=int(source>ab)
            if ab>0:g['ratios'].append(eb/ab)
            if ea>0 and aa>0 and ab>0 and eb>0:improvements.append((ea/aa)/(eb/ab))
        for g in groups.values():
            ratios=g.pop('ratios');g['median_embedded_error_over_own_allowance']=median(ratios) if ratios else None
            g['maximum_embedded_error_over_own_allowance']=max(ratios) if ratios else None
        rows.append(dict(case=new['case'],failed_before=a['failed_component_count'],failed_after=b['failed_component_count'],
            source_budget_failures_after=b['source_uncertainty_failed_component_count'],
            median_improvement_of_normalized_embedded_error=median(improvements) if improvements else None,
            unresolved_positive_candidate_cells_before=a['unresolved_positive_candidate_cells'],
            unresolved_positive_candidate_cells_after=b['unresolved_positive_candidate_cells'],
            original_bank_volume_estimates=a['fine_values'][:2],new_bank_volume_estimates=b['fine_values'][:2],
            maximum_depth_before=a['maximum_evaluated_depth'],maximum_depth_after=b['maximum_evaluated_depth'],
            groups=dict(groups)))
    result=dict(format='genesis-mapped-angular-diagnostics-1',cases=rows,
        old_control_pin=old_pin,new_control_pin=pin(args.control/'report.json'),
        comparison_is_between_fresh_adaptive_controls_not_identical_nodes=True,
        embedded_error_is_not_a_proven_remainder=True,physical_admission=False)
    require(not args.output.exists(),'Fresh diagnostic file required')
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps([{k:v for k,v in row.items() if k!='groups'} for row in rows],ensure_ascii=False))


if __name__=='__main__':main()
