"""Audit/restart a retained candidate ONLY: no model load or mechanical step."""
import argparse,hashlib,json,time,resource
from pathlib import Path
from analysis.genesis_connected_step_validation.run import PARENT,PARENT_SHA,deadline
from .checkpoint import verify,restore
from .adapter import state_payload,dumps
from .audit import audit


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);a=p.parse_args();d=a.input
    oldraw=(d/'report.json').read_bytes();report=json.loads(oldraw);candidate_raw=(d/'candidate.json').read_bytes()
    if hashlib.sha256(candidate_raw).hexdigest()!=report['artifacts']['candidate.json']['sha256']:raise ValueError('Candidate changed since first run')
    if hashlib.sha256(PARENT.read_bytes()).hexdigest()!=PARENT_SHA:raise ValueError('Parent changed')
    midpath=Path('/workspace/genesis-connected-step-08685-01/candidate.json')
    if hashlib.sha256(midpath.read_bytes()).hexdigest()!='4fa1eb80eed9d19ddb44a08c747dfe3fe267a84bd953380f649d4d3834c60e98':raise ValueError('Baseline changed')
    before=json.loads(PARENT.read_bytes());mid=json.loads(midpath.read_bytes());packet=json.loads(candidate_raw);verify(packet)
    resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3));start=time.monotonic()
    def save(name,obj):
        with (d/name).open('x') as f:f.write(dumps(obj))
    with deadline(min(180,300-report['seconds'])):
        print('Independent audit of retained candidate',flush=True);audit_result=audit(before,packet,mid);save('audit.json',audit_result)
        print('Audited typed-state restart',flush=True);state,recheck=restore(packet,before,mid)
        save('restart.json',{**packet,'state':state_payload(state)})
    if (d/'restart.json').read_bytes()!=candidate_raw:raise ValueError('Restart bytes differ')
    seconds=time.monotonic()-start
    final={k:v for k,v in report.items() if k not in ['error','error_type','artifacts']}
    final.update(status='PASS',initial_audit_failure=report.get('error'),initial_report_sha256=hashlib.sha256(oldraw).hexdigest(),
        seconds=report['seconds']+seconds,audit_recovery_seconds=seconds,restart_byte_identical=True,
        independent_final=audit_result['final'],time_myr=state.time_myr,cohorts=len(state.geometry.cohorts),
        max_damage=float(max(state.fault_history.damage)),parent_unchanged=hashlib.sha256(PARENT.read_bytes()).hexdigest()==PARENT_SHA,
        peak_rss_kib=max(report['peak_rss_kib'],resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        artifacts={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in d.iterdir() if p.is_file()})
    save('verified_report.json',final)
    print(json.dumps({k:final[k] for k in ['status','seconds','audit_recovery_seconds','cohorts','peak_rss_kib','restart_byte_identical']}),flush=True)


if __name__=='__main__':main()
