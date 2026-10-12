"""One isolated 100-year candidate; source physics and canonical parent untouched."""
from pathlib import Path
from contextlib import contextmanager
import argparse,hashlib,json,math,resource,shutil,signal,time
import numpy as np

PARENT = Path('analysis/genesis_volume_creation_validation/evidence/actual_continuation/direct_after_second.json')
PARENT_SHA = '62579a0bad128830a66e2f44579cf2b1f5d3835b691225b8d431102f3e4486d7'
FORMATION_SHA = 'cd79ee2a5139c4b91abdb1b4714a761c9153cc9f4490f5aa63fa2facba458c89'


class GateBoundary(ValueError):
    pass


def gate(history, parameters):
    arrays = [np.asarray(getattr(history,k)) for k in ('damage','fault_candidate_age_myr','fault_active')]
    damage,age,active = arrays
    if any(a.shape!=(80,) for a in arrays) or active.dtype.kind!='b':
        raise GateBoundary('Exactly all 80 source faces and boolean active flags required')
    if not np.isfinite(damage).all() or not np.isfinite(age).all() or np.any((damage<0)|(damage>1)) or np.any(age<0):
        raise GateBoundary('Invalid saved damage/age')
    threshold = parameters.activation_damage
    persistence = parameters.activation_persistence_myr
    eligible = (damage>=threshold)
    ready = eligible & (age>=persistence) & bool(parameters.enabled) & bool(history.membrane_established)
    result=dict(faces=80,active=np.flatnonzero(active).tolist(),damage_candidates=np.flatnonzero(eligible).tolist(),
        persistent_candidates=np.flatnonzero(age>0).tolist(),ready=np.flatnonzero(ready).tolist(),
        max_damage=float(damage.max()),max_candidate_age_myr=float(age.max()),
        activation_damage=threshold,activation_persistence_myr=persistence)
    # Conservative diagnostic stop before any candidate, not a new physical gate.
    if any(result[k] for k in ('active','damage_candidates','persistent_candidates','ready')):
        raise GateBoundary(json.dumps(result,sort_keys=True))
    return result


def guarded_step(model,state,save_candidate):
    gate(state.fault_history,model.fault_parameters)
    candidate=model.step(state,.0001)
    save_candidate(candidate)  # retain evidence even if endpoint reaches gate
    gate(candidate.fault_history,model.fault_parameters)
    return candidate


@contextmanager
def deadline(seconds):
    def timeout(*_):raise TimeoutError('Predeclared wall-time limit reached')
    previous=signal.signal(signal.SIGALRM,timeout)
    signal.setitimer(signal.ITIMER_REAL,seconds)
    try:yield
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(exist_ok=False)
    start=time.monotonic();report=dict(status='RUNNING',canonical_history_advanced=False,
        numerical_future_intervals=0,total_wall_limit_seconds=600,step_wall_limit_seconds=120,
        stages={},physical_creation_admitted=False,temporal_convergence_claimed=False)
    def write(name,value):
        with (args.output/name).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
    def stage(name,cap,fn):
        report['current_stage']=name;remaining=600-(time.monotonic()-start)
        if remaining<=0:raise TimeoutError('Total wall-time budget exhausted')
        print(json.dumps(dict(stage=name,event='start',elapsed=time.monotonic()-start)),flush=True)
        t=time.monotonic()
        try:
            with deadline(min(cap,remaining)):return fn()
        finally:
            report['stages'][name]=time.monotonic()-t
            print(json.dumps(dict(stage=name,event='end',seconds=report['stages'][name])),flush=True)
    parent_bytes=PARENT.read_bytes()
    try:
        if hashlib.sha256(parent_bytes).hexdigest()!=PARENT_SHA:raise ValueError('Wrong accepted parent bytes')
        if shutil.disk_usage(args.output).free<512*1024**2:raise ValueError('512 MiB disk reserve required')
        if np.finfo(np.longdouble).nmant!=63:raise ValueError('Required longdouble precision unavailable')
        resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
        from tectonics.genesis_volume_slip_checkpoint import load_volume_slip_checkpoint,save_volume_slip_checkpoint,volume_slip_checkpoint_to_dict
        from tectonics.genesis_volume_slip_dynamics import VolumeSlipModel
        from analysis.genesis_volume_slip_validation.actual_control import audit_state
        from analysis.genesis_volume_slip_validation.independent_audit import audit_actual_interval
        from analysis.genesis_volume_creation_validation.independent_audit import audit_endpoint_energy
        canonical=lambda v:json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)
        before=json.loads(parent_bytes)
        model,state=stage('load_existing_parent',240,lambda:load_volume_slip_checkpoint(PARENT,expected_source_checkpoint_sha256=FORMATION_SHA))
        if type(model) is not VolumeSlipModel:raise ValueError('Model substitution refused')
        if abs(state.time_myr-.8684)>1e-14 or len(state.geometry.cohorts)!=800:raise ValueError('Wrong source clock/catalogue')
        if canonical(volume_slip_checkpoint_to_dict(model,state))!=canonical(before):raise ValueError('Parent replay differs')
        report['before_gate']=gate(state.fault_history,model.fault_parameters)
        report['before_state_audit']=stage('before_state_audit',60,lambda:audit_state(model,state))
        saved={}
        def retain(candidate):
            saved['packet']=save_volume_slip_checkpoint(args.output/'candidate.json',model,candidate)
        candidate=stage('single_future_step',120,lambda:guarded_step(model,state,retain))
        report['numerical_future_intervals']=1
        after=saved['packet'];report['after_gate']=gate(candidate.fault_history,model.fault_parameters)
        if abs(candidate.time_myr-.8685)>1e-14:raise ValueError('Unexpected candidate time')
        report['after_state_audit']=stage('after_state_audit',60,lambda:audit_state(model,candidate))
        independent=stage('independent_interval_audit',180,lambda:audit_actual_interval(before,after))
        write('independent_interval_audit.json',independent)
        report['endpoint_energy_audit']=stage('independent_endpoint_energy',60,lambda:audit_endpoint_energy(after))
        restored_model,restored=stage('candidate_restart_replay',240,lambda:load_volume_slip_checkpoint(args.output/'candidate.json',expected_source_checkpoint_sha256=FORMATION_SHA))
        save_volume_slip_checkpoint(args.output/'restart.json',restored_model,restored)
        if (args.output/'candidate.json').read_bytes()!=(args.output/'restart.json').read_bytes():raise ValueError('Restart bytes differ')
        if PARENT.read_bytes()!=parent_bytes:raise ValueError('Accepted parent changed')
        old_ids={c.id for c in state.geometry.cohorts};new_ids={c.id for c in candidate.geometry.cohorts}
        if not old_ids<=new_ids:raise ValueError('Existing material IDs lost')
        if after['source_packet']!=before['source_packet']:raise ValueError('Source/parameters changed')
        report.update(status='PASS',candidate_time_myr=candidate.time_myr,cohorts=len(new_ids),
            retained_cohorts=len(old_ids),newborn_cohorts=len(new_ids-old_ids),restart_byte_identical=True,
            immutable_parent_byte_identical=True,source_parameters_unchanged=True,
            max_elastic_principal_strain=candidate.last_report['maximum_elastic_principal_strain'],
            momentum_relative_residual=candidate.last_report['momentum_equation_relative_residual'],
            whole_interval_geometry_certified=candidate.last_report['whole_interval_all_pair_geometry_certified'])
    except Exception as error:
        report.update(status='STOPPED',error_type=type(error).__name__,error=str(error),
            immutable_parent_byte_identical=PARENT.read_bytes()==parent_bytes)
    finally:
        report['seconds']=time.monotonic()-start
        report['peak_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report['parent_raw_sha256']=PARENT_SHA
        report['artifacts']={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in args.output.iterdir() if p.is_file()}
        write('report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k in ['status','seconds','stages','error','cohorts','candidate_time_myr','peak_rss_kib']}),flush=True)


if __name__=='__main__':main()
