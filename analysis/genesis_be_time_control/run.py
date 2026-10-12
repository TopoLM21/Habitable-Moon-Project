"""Exactly 2x50 then, only if PASS, 4x25 on the SAME canonical interval."""
import argparse,json,hashlib,time,resource,shutil,traceback
from pathlib import Path
import numpy as np
from analysis.genesis_connected_step_validation.run import PARENT,PARENT_SHA,FORMATION_SHA,deadline,gate
from analysis.genesis_be_candidate.adapter import dumps,state_payload
from .chain import transition
from .audit import audit_step,verify_sum
from .checkpoint import restore


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
    start=time.monotonic();r=dict(status='RUNNING',total_budget_seconds=600,new_substeps=0,stages={},branches={})
    def write(name,value):
        path=a.output/name;path.parent.mkdir(exist_ok=True)
        with path.open('x') as f:f.write(dumps(value))
        if sum(p.stat().st_size for p in a.output.rglob('*') if p.is_file())>1024**3:raise RuntimeError('1 GiB output budget exceeded')
    def stage(name,cap,fn):
        remaining=600-(time.monotonic()-start)
        if remaining<=0:raise TimeoutError('Total 600-second budget exhausted')
        t=time.monotonic();print(json.dumps(dict(stage=name,event='start',elapsed=t-start)),flush=True)
        try:
            with deadline(min(cap,remaining)):return fn()
        finally:
            r['stages'][name]=time.monotonic()-t
            print(json.dumps(dict(stage=name,event='end',seconds=r['stages'][name])),flush=True)
    raw=PARENT.read_bytes()
    try:
        if hashlib.sha256(raw).hexdigest()!=PARENT_SHA:raise ValueError('Root pin')
        if np.finfo(np.longdouble).nmant!=63:raise ValueError('Longdouble')
        if shutil.disk_usage(a.output).free<1024**3:raise ValueError('Need 1 GiB free')
        resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
        baseline=Path('/workspace/genesis-be-candidate-01/candidate.json')
        if hashlib.sha256(baseline.read_bytes()).hexdigest()!='b919855cd5a27fcec4f49eedaadab4a490775b801b10c54a667f6d4d14c6b892':raise ValueError('BE100 changed')
        from tectonics.genesis_volume_slip_checkpoint import load_volume_slip_checkpoint
        root=json.loads(raw);model,initial=stage('load_parent',120,lambda:load_volume_slip_checkpoint(PARENT,expected_source_checkpoint_sha256=FORMATION_SHA))
        gate(initial.fault_history,model.fault_parameters)
        for label,n,dt in [('50',2,5e-5),('25',4,2.5e-5)]:
            branch_start=time.monotonic();state=initial;previous=None;packets=[];audits=[];b=dict(status='RUNNING',steps=[]);r['branches'][label]=b
            for i in range(1,n+1):
                if r['new_substeps']>=6:raise ValueError('Six-substep budget')
                r['new_substeps']+=1
                state,packet=stage(label+'_step_'+str(i),90,lambda:transition(model,state,previous,dt))
                write(label+'/step_'+str(i)+'.json',packet)
                result,*_=stage(label+'_audit_'+str(i),90,lambda:audit_step(root,previous,packet))
                write(label+'/audit_'+str(i)+'.json',result);packets.append(packet);audits.append(result)
                cumulative=verify_sum(root,packets,audits)
                b['steps'].append(dict(index=i,gate=gate(state.fault_history,model.fault_parameters),
                    independent_final=result['final'],cumulative=cumulative))
                previous=packet
                # Continue the actual branch FROM a decoded, fully audited mid-state.
                if i==n//2 or i==n:
                    state,total=stage(label+'_restart_'+str(i),120,lambda:restore(root,packets))
                    restarted={**packet,'state':state_payload(state)}
                    write(label+'/restart_'+str(i)+'.json',restarted)
                    if (a.output/label/('step_'+str(i)+'.json')).read_bytes()!=(a.output/label/('restart_'+str(i)+'.json')).read_bytes():raise ValueError('Restart bytes differ')
            if abs(state.time_myr-.8685)>1e-14:raise ValueError('Different endpoint')
            b.update(status='PASS',seconds=time.monotonic()-branch_start,cohorts=len(state.geometry.cohorts),restart_byte_identical=True,cumulative=cumulative)
            print(json.dumps(dict(branch=label,status='PASS',seconds=b['seconds'],cohorts=b['cohorts'])),flush=True)
        r['status']='PASS'
    except Exception as e:
        r.update(status='STOPPED',error_type=type(e).__name__,error=str(e));(a.output/'failure.txt').write_text(traceback.format_exc())
    finally:
        r['seconds']=time.monotonic()-start;r['peak_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;r['canonical_parent_unchanged']=PARENT.read_bytes()==raw
        r['implementation_pins']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ['analysis/genesis_be_time_control','analysis/genesis_be_candidate'] for p in Path(folder).glob('*.py')}
        r['artifacts']={str(p.relative_to(a.output)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in a.output.rglob('*') if p.is_file()}
        write('report.json',r)
    print(json.dumps({k:r[k] for k in ['status','seconds','new_substeps','peak_rss_kib']} | ({'error':r['error']} if 'error' in r else {})),flush=True)


if __name__=='__main__':main()
