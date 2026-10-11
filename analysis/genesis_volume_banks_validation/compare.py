"""Read-only comparison and compact evidence for the completed bank control."""
import json,hashlib,re
from pathlib import Path
import numpy as np

ROOT=Path('/workspace/Habitable-Moon-Project')
OLD=Path('/workspace/genesis-mapped-full-03')
NEW=Path('/workspace/genesis-banks-full-02')
DIAG=Path('/workspace/genesis-banks-diagnostic-04')
DEST=ROOT/'analysis/genesis_volume_banks_validation'

def read(p):return json.loads(p.read_bytes())
def pin(p):return dict(path=str(p),bytes=p.stat().st_size,raw_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def relative(a):
 e=np.array(a['observed_embedded_errors']);t=np.array(a['component_allowances'])
 return np.divide(e,t,out=np.full_like(e,np.inf),where=t>0)

new=read(NEW/'report.json');old=read(OLD/'report.json');diagnosis=read(DIAG/'report.json')
old_summary=read(ROOT/'analysis/genesis_volume_mapped_validation/summary.json')
old_expected=old_summary['evidence'][str(OLD/'report.json')]
assert all(pin(OLD/'report.json')[k]==v for k,v in old_expected.items())
rows=[];groups=[]
for prior,after,diag in zip(old['cases'],new['cases'],diagnosis['cases']):
 a,b=prior['angular'],after['angular'];i=after['case']
 assert a['component_names']==b['component_names']
 for key in ('relative_tolerance','angular_orders','max_cells','max_depth','max_line_evaluations'):
  assert a[key]==b[key],key
 assert prior['original_radius_m']==after['original_radius_m']
 assert b['exact_initial_parameter_measure_preserved']
 r0,r1=relative(a),relative(b);valid=np.isfinite(r0)&np.isfinite(r1)&(r0>0)&(r1>0)
 rows.append(dict(case=i,radius_m=after['original_radius_m'],
  nodes_before=a['radial_line_evaluations'],nodes_after=b['radial_line_evaluations'],
  positive_nodes=after['node_counts']['positive'],pointwise_failed_nodes=after['node_counts']['pointwise_failed_nodes'],
  failed_before=a['failed_component_count'],failed_after=b['failed_component_count'],
  source_budget_failures=b['source_uncertainty_failed_component_count'],
  unresolved_cells_before=a['unresolved_positive_candidate_cells'],unresolved_cells_after=b['unresolved_positive_candidate_cells'],
  old_missing_bank_classification=diag['missing_bank_counts'],
  unresolved_bank_volume_before=a['unresolved_positive_candidate_volume_m3'],unresolved_bank_volume_after=b['unresolved_positive_candidate_volume_m3'],
  bank_volume_estimates_before=a['fine_values'][:2],bank_volume_estimates_after=b['fine_values'][:2],
  bank_volume_lower_after=b['reference_side_volume_envelope_lower_m3'],bank_volume_upper_after=b['reference_side_volume_envelope_upper_m3'],
  median_normalized_embedded_error_improvement=float(np.median(r0[valid]/r1[valid])) if valid.any() else None,
  median_absolute_embedded_error_improvement=float(np.median(np.array(a['observed_embedded_errors'])[valid]/np.array(b['observed_embedded_errors'])[valid])) if valid.any() else None,
  median_old_over_new_own_allowance=float(np.median(np.array(a['component_allowances'])[valid]/np.array(b['component_allowances'])[valid])) if valid.any() else None,
  status=b['status'],reason=b['reason'],depth_before=a['maximum_evaluated_depth'],depth_after=b['maximum_evaluated_depth'],
  certified_empty_cells_skipped=b['certified_empty_cells_skipped'],
  measure_preserved=b['exact_initial_parameter_measure_preserved'],
  bank_audit=after['independent_bank_audit'],
  independent_tensor_components=sum(s['components'] for s in after['independent_samples']),
  independent_tensor_misses=sum(len(s['misses']) for s in after['independent_samples']),
  parent_K_M_bytes_unchanged=after['parent_K_M_bytes_unchanged'],
  node_file_pin=pin(NEW/f'case_{i:03d}'/'nodes.jsonl'),seconds=after['seconds']))
 reach=read(Path(f'/workspace/genesis-banks-sliver-audit-case{i}/report.json'))
 mesh_audit=read(Path(f'/workspace/genesis-banks-partition-audit-case{i}/report.json'))
 rows[-1]['proven_positive_banks_unreachable_by_remaining_midpoint_depth']=reach['proven_unreachable_positive_banks']
 rows[-1]['sliver_audit_pin']=pin(Path(f'/workspace/genesis-banks-sliver-audit-case{i}/report.json'))
 rows[-1]['independent_mesh_audit']=mesh_audit['cases'][0]
 rows[-1]['independent_mesh_audit_pin']=pin(Path(f'/workspace/genesis-banks-partition-audit-case{i}/report.json'))
 names={re.sub(r'\[.*','',n) for n in b['component_names']}
 detail={}
 for name in sorted(names):
  idx=np.array([j for j,n in enumerate(b['component_names']) if re.sub(r'\[.*','',n)==name]);mask=valid[idx]
  detail[name]=dict(components=len(idx),failed_before=int(np.sum(r0[idx]>1)),failed_after=int(np.sum(r1[idx]>1)),
   median_normalized_error_before=float(np.median(r0[idx][mask])) if mask.any() else None,
   median_normalized_error_after=float(np.median(r1[idx][mask])) if mask.any() else None)
 groups.append(dict(case=i,groups=detail))

history=read(ROOT/'analysis/genesis_volume_mapped_validation/summary.json')['history']
for record in history['raw_history_files_rechecked'].values():
 assert pin(Path(record['path']))['raw_sha256']==record['raw_sha256']
frozen=read(ROOT/'analysis/genesis_volume_integral_validation/evidence/package_manifest.json')['source_sha256']
assert len(frozen)==469 and all(pin(ROOT/k)['raw_sha256']==v for k,v in frozen.items())
request=read(NEW/'request.json')
for path,record in request['source_pins'].items():
 actual=pin(ROOT/path);assert actual['raw_sha256']==record['raw_sha256'] and actual['bytes']==record['bytes']
summary=dict(format='genesis-bank-bounds-event-integration-validation-1',base_commit='dcbc8cf889f58b0a0c2673032bcc662fbd6c4f80',
 status='COMPLETED_FOUR_ORIGINAL_CONTROLS',cases=rows,history=history,
 event_mesh_strategy_verdict='REJECTED_AS_A_GENERAL_ANGULAR_CONVERGENCE_FIX',
 next_step='Exact affine Gamma cuts without rounded-edge slivers; replace unconditional missing-first priority, then curved sphere event integration and angular remainder at unchanged 2e-6',
 total_nodes=sum(r['nodes_after'] for r in rows),
 frozen_469_sources_unchanged=True,numerical_sources_match_completed_run=True,
 original_angular_options=request['original_angular_options'],
 full_report_pin=pin(NEW/'report.json'),old_report_pin=pin(OLD/'report.json'),diagnosis_pin=pin(DIAG/'report.json'),
 test_log_pin=pin(Path('/workspace/genesis-banks-regression-final-02.log')),
 stopped_pilot_pin=pin(Path('/workspace/genesis-banks-full-01/STOPPED.json')),
 tests=dict(main_regression='67 passed in 13.62s',midpoint_sliver_proof='5 passed in 1.01s'),
 sliver_test_log_pin=pin(Path('/workspace/genesis-banks-sliver-tests-01.log')),
 analytic_sphere_control=dict(
  before_pin=pin(Path('/workspace/genesis-banks-sphere-oracle-baseline-02/report.json')),
  after_pin=pin(Path('/workspace/genesis-banks-sphere-oracle-current-01/report.json')),
  relative_errors_before=read(Path('/workspace/genesis-banks-sphere-oracle-baseline-02/report.json'))['fine_relative_errors'],
  relative_errors_after=read(Path('/workspace/genesis-banks-sphere-oracle-current-01/report.json'))['fine_relative_errors'],
  interpretation='Missing-bank correction alone does not resolve curved-sphere angular error at the unchanged cap'),
 source_pins=request['source_pins'],
 additional_validation_source_pins={str(p.relative_to(ROOT)):pin(p) for p in (
  ROOT/'analysis/genesis_volume_banks_validation/partition_audit.py',
  ROOT/'analysis/genesis_volume_banks_validation/sliver_audit.py',
  ROOT/'analysis/genesis_volume_banks_validation/sphere_oracle.py',
  ROOT/'analysis/genesis_volume_banks_validation/compare.py',
  ROOT/'tests/test_genesis_volume_bank_sliver_audit.py')},
 scope=dict(whole_cell_bank_volume_bounds_certified=True,exact_domain_partition=True,
  exact_event_alignment=False,angular_remainder_certified=False,ideal_Gauss_parameters_certified=False,
  independent_tensor_audit_is_sampled=True,physical_creation_admitted=False,physical_release_admitted=False,
  new_accepted_intervals=0,new_material_cohorts=0,stage_1_open=True,
  angular_controls_open=[r['case'] for r in rows if r['status']=='OPEN']))
for name,value in (('summary.json',summary),('angular_comparison.json',dict(cases=groups))):
 path=DEST/name
 with path.open('x') as stream:json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')
print(json.dumps(rows,indent=2,ensure_ascii=False))
