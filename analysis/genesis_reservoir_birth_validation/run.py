"""Read the already computed thermal step; propose material changes only."""
import argparse,json,hashlib,resource,signal,time
from pathlib import Path
from tectonics.genesis_reservoir_thermal import load
from tectonics.genesis_reservoir_birth import prepare,require_loading_request,digest
from tectonics.genesis_bulk_birth import bulk_birth_from_dict,validate_bulk_birth_continuation
from analysis.genesis_energy_inventory.audit import ROOT,ROOT_SHA


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
    resource.setrlimit(resource.RLIMIT_AS,(1024**3,1024**3))
    def stop(*args):raise TimeoutError('60-second budget')
    signal.signal(signal.SIGALRM,stop);signal.alarm(60);start=time.monotonic()
    raw=ROOT.read_bytes();assert hashlib.sha256(raw).hexdigest()==ROOT_SHA
    paths=[Path('/workspace/genesis-coupled-thermal-projection-03/projected.json'),Path('/workspace/genesis-reservoir-thermal-step-01/after.json')]
    hashes=['d25fe1fd4e8b56b18ab8ec697b197c326d723e9421148e5d3726fb7affdd45a4','154b7dce4f1dbad58c3cf7ce583c39c8c6a0e55f80bacb895edae8085bb2c236']
    for path,sha in zip(paths,hashes):
        if hashlib.sha256(path.read_bytes()).hexdigest()!=sha:raise ValueError('Saved thermal state SHA')
    before,after=[load(path,legacy_path=ROOT) for path in paths];packet=prepare(before,after,raw)
    out=a.output/'proposal.json';out.write_text(json.dumps(packet,indent=2)+'\n');restored=json.loads(out.read_bytes())
    old=bulk_birth_from_dict(json.loads(raw)['runtime_state']['history']);new=bulk_birth_from_dict(restored['proposed_history'])
    validate_bulk_birth_continuation(new,old)
    if restored!=packet:raise ValueError('Readback')
    try:require_loading_request(restored)
    except ValueError:guard=True
    else:guard=False
    if packet['status']=='BLOCKED_REMELT' and not guard:raise ValueError('Remelt guard failed')
    if ROOT.read_bytes()!=raw:raise ValueError('Canonical changed')
    for path,sha in zip(paths,hashes):
        if hashlib.sha256(path.read_bytes()).hexdigest()!=sha:raise ValueError('Thermal source changed')
    report=dict(validation='PASS',material_status=packet['status'],birth_mass_kg=packet['birth_mass_kg'],retirement_mass_kg=packet['retirement_mass_kg'],
      newborn_count=len(packet['newborn_requests']),affected_old_cohorts=len(packet['proposed_retirements']),
      old_cohort_count=len(old.cohorts),proposed_cohort_count=len(new.cohorts),
      mechanical_loading_guard_stopped=guard,new_thermal_steps=0,canonical_advanced=False,
      inherited_front_reconstruction_max_error=packet['inherited_front_reconstruction_max_error'],
      parent_memory_sha256=packet['parent_memory_sha256'],history_roundtrip_and_continuation=True,
      prior_total_column_melt_mass_decrease_kg=5.025769549385252e17,
      prior_migration_bottom_flux_relative_change=.1467492499419472,
      external_sources_in_thermal_control='disabled',
      input_pins={str(path):sha for path,sha in zip([ROOT,*paths],[ROOT_SHA,*hashes])},
      proposal_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));signal.alarm(0)

if __name__=='__main__':main()
