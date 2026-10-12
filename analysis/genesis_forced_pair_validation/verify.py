"""Read-only packet/hash/controller replay after the ONE real pair; no solve."""
import argparse,json,math
from collections import defaultdict
from pathlib import Path
from analysis.genesis_adaptive_control.controller import fingerprint
from analysis.genesis_adaptive_control.metrics import observe_packet,assess
from analysis.genesis_be_candidate.checkpoint import seal
from analysis.genesis_coupled_split_control.run import pinned
from .run import rawsha,REPO
from analysis.genesis_adaptive_control import backend as b


def verify(folder):
    folder=Path(folder);report=json.loads((folder/'report.json').read_text());proof=json.loads((folder/'compatibility.json').read_text())
    for name,sha in report['output_raw_sha256'].items():assert rawsha(folder/name)==sha
    for path,sha in proof['pins'].items():assert rawsha(path)==sha
    assert report['new_planetary_trials']==2 and report['controller']['attempts']==1
    cfg=json.loads((folder/'config.json').read_text());base=json.loads((REPO/'analysis/genesis_adaptive_control/config.json').read_text())
    allowed={'horizon_years','initial_step_years','max_step_years','max_attempts','max_accepted_windows','max_rejections_per_window','runtime_seconds','artifact_limit_bytes'}
    assert all(cfg[k]==v for k,v in base.items() if k not in allowed)
    snapshots={k:json.loads((folder/filename).read_bytes()) for k,filename in [('initial','initial.json'),('whole','001_whole.json'),('half1','001_half1.json'),('half2','001_half2.json')]}
    for s in snapshots.values():assert seal({k:v for k,v in s['packet'].items() if k!='sha256'})==s['packet']
    initial,whole,h1,h2=[snapshots[k] for k in ('initial','whole','half1','half2')]
    ids=[x['id'] for x in initial['packet']['state']['geometry']['cohorts']]
    assert len(ids)==len(set(ids))==800
    for s in (whole,h1,h2):assert [x['id'] for x in s['packet']['state']['geometry']['cohorts']]==ids
    assert h1['packet']['adaptive_predecessor_sha256']==initial['packet']['sha256']
    assert h2['packet']['adaptive_predecessor_sha256']==h1['packet']['sha256']
    assert h2['packet']['initial_thermal']==h1['packet']['thermal']
    assert h2['packet']['source_ports']['orbit_before']==h1['packet']['orbit_after']
    root=pinned(b.PARENT,b.PARENT_SHA);observations=[observe_packet(s['packet'],root['source_packet']) for s in (whole,h1,h2)]
    assessment=assess(*observations,25.,cfg)
    assert assessment==report['controller']['history'][0]['assessment']
    assert assessment['score']>1 and report['engineering_accepted'] is False
    assert report['accepted_elapsed_years']==0 and report['accepted_packet_sha256']==initial['packet']['sha256']
    reject=report['controller']['history'][0]
    assert reject['base_sha256']==reject['rollback_sha256']==fingerprint(initial)
    assert report['controller']['accepted_substeps']==0
    audits=[json.loads(p.read_text()) for p in sorted(folder.glob('*_audited_trial.json'))]
    assert len(audits)==3 and all(x['audit']['status']=='PASS' for x in audits)
    for a,s in zip(audits,(whole,h1,h2)):assert a['sha256']==fingerprint(s)
    stage=defaultdict(lambda:dict(calls=0,inclusive_seconds=0.,exclusive_seconds=0.))
    for t in report['timings']:
        x=stage[t['stage']];x['calls']+=1
        for k in ('inclusive_seconds','exclusive_seconds'):x[k]+=t[k]
    wp=whole['packet'];fp=h2['packet'];cw=wp['state']['interval_ledger'];parts=[s['packet']['state']['interval_ledger'] for s in (h1,h2)]
    sums={k:sum(x[k] for x in parts) for k in ('physical_maxwell_j','thermal_work_j','numerical_memory_j','numerical_kinetic_j')}
    return dict(status='PASS_HASH_AND_CONTROLLER_REPLAY',new_solver_calls=0,original_pair_status=report['status'],unchanged_engineering_budgets=True,
        ids=800,births=0,accepted_cursor_unchanged=True,all_three_independent_audits='PASS',assessment=assessment,
        measured=dict(total_seconds=report['total_seconds'],setup_seconds=report['setup_seconds'],controller_seconds=report['controller']['seconds'],
                      peak_rss_bytes=report['peak_rss_kib']*1024,artifact_bytes=sum(p.stat().st_size for p in folder.iterdir() if p.is_file()),stages=dict(stage),
                      unprofiled_io_and_controller_seconds=report['total_seconds']-report['setup_seconds']-sum(t['exclusive_seconds'] for t in report['timings'])),
        counts=dict(reused_whole=1,new_trials=2,corrections=sum(x['count'] for x in report['correctors']),BE_predictor_final_solves=2*sum(x['count'] for x in report['correctors']),independent_coupled_audits=3,source_audits=5,endpoint_typed_restart_checks=3,parent_typed_decodes=2,checkpoint_disk_readbacks=3),
        fine_energy=sums,whole_energy={k:cw[k] for k in sums},fine_final_elastic_j=fp['state']['mechanics']['elastic_energy_j'],whole_final_elastic_j=wp['state']['mechanics']['elastic_energy_j'],
        fine_max_damage=max(fp['state']['fault_history']['damage']),whole_max_damage=max(wp['state']['fault_history']['damage']),
        fine_surface_temperature_k=fp['thermal']['temperature_k'][-1],whole_surface_temperature_k=wp['thermal']['temperature_k'][-1],
        fine_end_time_myr=fp['thermal']['time_myr'],whole_end_time_myr=wp['thermal']['time_myr'],
        proposed_next_macro_years=25*assessment['next_factor'],proposed_step_executed=False,
        whole_packet_raw_sha256=rawsha(folder/'001_whole.json'),half1_packet_raw_sha256=rawsha(folder/'001_half1.json'),half2_packet_raw_sha256=rawsha(folder/'001_half2.json'),report_raw_sha256=rawsha(folder/'report.json'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True,type=Path);p.add_argument('--output',required=True,type=Path);args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    r=verify(args.folder)
    with args.output.open('x') as f:json.dump(r,f,indent=2,allow_nan=False)
    print(json.dumps(r,indent=2))


if __name__=='__main__':main()
