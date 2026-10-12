"""ONE isolated 100-year BE candidate, hard bounded; never promote or push."""
import argparse,json,hashlib,time,resource,shutil,traceback
from pathlib import Path
import numpy as np
from analysis.genesis_connected_step_validation.run import PARENT,PARENT_SHA,FORMATION_SHA,deadline,gate
from .adapter import transition,encode,dumps,state_payload
from .checkpoint import seal,restore
from .audit import audit


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();args.output.mkdir(exist_ok=False)
    start=time.monotonic();report=dict(status='RUNNING',total_limit_s=300,step_limit_s=120,stages={},new_physical_intervals=0)
    def write(name,value):(args.output/name).write_text(dumps(value))
    def stage(name,limit,fn):
        remaining=300-(time.monotonic()-start)
        if remaining<=0:raise TimeoutError('Total 300 seconds exceeded')
        t=time.monotonic();print(json.dumps(dict(stage=name,event='start',elapsed=t-start)),flush=True)
        try:
            with deadline(min(limit,remaining)):return fn()
        finally:
            report['stages'][name]=time.monotonic()-t
            print(json.dumps(dict(stage=name,event='end',seconds=report['stages'][name])),flush=True)
    raw=PARENT.read_bytes()
    try:
        if hashlib.sha256(raw).hexdigest()!=PARENT_SHA:raise ValueError('Canonical source changed')
        if shutil.disk_usage(args.output).free<512*1024**2:raise ValueError('Need 512 MiB disk reserve')
        if np.finfo(np.longdouble).nmant!=63:raise ValueError('Precision environment')
        resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
        midpath=Path('/workspace/genesis-connected-step-08685-01/candidate.json')
        if hashlib.sha256(midpath.read_bytes()).hexdigest()!='4fa1eb80eed9d19ddb44a08c747dfe3fe267a84bd953380f649d4d3834c60e98':raise ValueError('Midpoint pin')
        midpoint=json.loads(midpath.read_bytes());before=json.loads(raw)
        from tectonics.genesis_volume_slip_checkpoint import load_volume_slip_checkpoint
        model,parent=stage('load_pinned_parent',120,lambda:load_volume_slip_checkpoint(PARENT,expected_source_checkpoint_sha256=FORMATION_SHA))
        report['before_gate']=gate(parent.fault_history,model.fault_parameters)
        candidate,detail=stage('single_BE_interval',120,lambda:transition(model,parent));report['new_physical_intervals']=1
        packet=seal(encode(model,candidate,detail));write('candidate.json',packet)
        report['after_gate']=gate(candidate.fault_history,model.fault_parameters)
        independent=stage('independent_audit',90,lambda:audit(before,packet,midpoint));write('audit.json',independent)
        restored,recheck=stage('audited_state_restart',90,lambda:restore(packet,before,midpoint))
        restarted={**packet,'state':state_payload(restored)};write('restart.json',restarted)
        if (args.output/'candidate.json').read_bytes()!=(args.output/'restart.json').read_bytes():raise ValueError('Restart bytes differ')
        report.update(status='PASS',restart_byte_identical=True,cohorts=len(candidate.geometry.cohorts),time_myr=candidate.time_myr,
            independent_final=independent['final'],max_damage=float(max(candidate.fault_history.damage)))
    except Exception as e:
        report.update(status='STOPPED',error_type=type(e).__name__,error=str(e));(args.output/'failure.txt').write_text(traceback.format_exc())
    finally:
        report['parent_unchanged']=PARENT.read_bytes()==raw
        report['seconds']=time.monotonic()-start;report['peak_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report['artifacts']={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in args.output.iterdir() if p.is_file()}
        write('report.json',report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
