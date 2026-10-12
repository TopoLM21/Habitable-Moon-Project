"""Bounded 2x50 and 4x25 intact branches; retain the existing 1x100 result."""
from pathlib import Path
import argparse,hashlib,json,resource,shutil,time
import numpy as np
from analysis.genesis_connected_step_validation.run import PARENT,PARENT_SHA,FORMATION_SHA,gate,deadline
from .metrics import extract,compare


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();args.output.mkdir(exist_ok=False)
    start=time.monotonic();report=dict(status='RUNNING',stages={},branches={},comparisons={},max_wall_seconds=720,
        canonical_history_advanced=False,pure_time_convergence_claimed=False,physical_parameters_changed=False)
    def write(path,data):
        with path.open('x') as f:json.dump(data,f,indent=2,allow_nan=False);f.write('\n')
    def stage(name,cap,fn):
        remaining=720-(time.monotonic()-start)
        if remaining<=0:raise TimeoutError('Total budget exhausted')
        report['current_stage']=name;t=time.monotonic();print(json.dumps(dict(stage=name,event='start',elapsed=t-start)),flush=True)
        try:
            with deadline(min(cap,remaining)):return fn()
        finally:report['stages'][name]=time.monotonic()-t
    raw=PARENT.read_bytes()
    try:
        if hashlib.sha256(raw).hexdigest()!=PARENT_SHA:raise ValueError('Parent pin changed')
        if shutil.disk_usage(args.output).free<1024**3:raise ValueError('1 GiB disk reserve required')
        if np.finfo(np.longdouble).nmant!=63:raise ValueError('longdouble precision unavailable')
        resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
        from tectonics.genesis_volume_slip_checkpoint import load_volume_slip_checkpoint,save_volume_slip_checkpoint,volume_slip_checkpoint_to_dict
        from tectonics.genesis_volume_slip_dynamics import VolumeSlipModel
        from analysis.genesis_volume_slip_validation.actual_control import audit_state
        from analysis.genesis_volume_slip_validation.independent_audit import audit_actual_interval
        from analysis.genesis_volume_creation_validation.independent_audit import audit_endpoint_energy
        parent=json.loads(raw)
        previous=json.loads(Path('analysis/genesis_connected_step_validation/summary.json').read_bytes())
        pin=previous['artifacts']['candidate.json'];baseline_bytes=Path(pin['path']).read_bytes()
        if hashlib.sha256(baseline_bytes).hexdigest()!=pin['sha256'] or previous['report']['status']!='PASS':raise ValueError('Baseline certificate mismatch')
        baseline=json.loads(baseline_bytes);metrics={'100':extract(baseline,parent)}
        report['baseline_pin']=pin
        model,initial=stage('load_parent',180,lambda:load_volume_slip_checkpoint(PARENT,expected_source_checkpoint_sha256=FORMATION_SHA))
        if type(model) is not VolumeSlipModel:raise ValueError('Wrong model')
        gate(initial.fault_history,model.fault_parameters)
        if volume_slip_checkpoint_to_dict(model,initial)!=parent:raise ValueError('Parent replay changed')
        for label,years,count in [('50',50,2),('25',25,4)]:
            folder=args.output/label;folder.mkdir();state=initial;before=parent
            branch=dict(years_per_step=years,requested_steps=count,steps=[]);report['branches'][label]=branch
            for index in range(count):
                gate(state.fault_history,model.fault_parameters)
                def advance():return model.step(state,years/1e6)
                candidate=stage(label+'_step_'+str(index+1),90,advance)
                target=folder/('step_'+str(index+1)+'.json')
                after=save_volume_slip_checkpoint(target,model,candidate)
                gate_status=gate(candidate.fault_history,model.fault_parameters)
                if after['source_packet']!=parent['source_packet']:raise ValueError('Source changed')
                state_audit=stage(label+'_state_'+str(index+1),30,lambda:audit_state(model,candidate))
                independent=stage(label+'_audit_'+str(index+1),90,lambda:audit_actual_interval(before,after))
                write(folder/('audit_'+str(index+1)+'.json'),independent)
                branch['steps'].append(dict(index=index+1,gate=gate_status,state_audit=state_audit,
                    material_cohorts=len(candidate.geometry.cohorts),independent_balance=independent['discrete_dynamic_balance'],
                    geometry=independent['geometry'],maximum_cohort_relative_work_residual=independent['maximum_cohort_relative_work_residual']))
                state,before=candidate,after
            if abs(state.time_myr-.8685)>1e-14:raise ValueError('Endpoint time changed')
            branch['endpoint_energy_audit']=stage(label+'_energy',30,lambda:audit_endpoint_energy(before))
            metrics[label]=extract(before,parent)
            restored_model,restored=stage(label+'_restart',180,lambda:load_volume_slip_checkpoint(target,expected_source_checkpoint_sha256=FORMATION_SHA))
            save_volume_slip_checkpoint(folder/'restart.json',restored_model,restored)
            if target.read_bytes()!=(folder/'restart.json').read_bytes():raise ValueError('Restart differs')
            branch.update(status='PASS',restart_byte_identical=True,cohorts=len(state.geometry.cohorts))
            del restored_model,restored
            print(json.dumps(dict(branch=label,status='PASS',elapsed=time.monotonic()-start,cohorts=len(state.geometry.cohorts))),flush=True)
        report['comparisons']={a+'_vs_'+b:compare(metrics[a],metrics[b]) for a,b in [('100','50'),('50','25'),('100','25')]}
        write(args.output/'metrics.json',metrics)
        if PARENT.read_bytes()!=raw:raise ValueError('Canonical parent changed')
        report.update(status='PASS',parent_byte_unchanged=True)
    except Exception as e:report.update(status='STOPPED',error_type=type(e).__name__,error=str(e),parent_byte_unchanged=PARENT.read_bytes()==raw)
    finally:
        report['seconds']=time.monotonic()-start;report['peak_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report['artifacts']={str(p.relative_to(args.output)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in args.output.rglob('*') if p.is_file()}
        write(args.output/'report.json',report)
    print(json.dumps({k:report[k] for k in ['status','seconds','peak_rss_kib','current_stage']}),flush=True)


if __name__=='__main__':main()
