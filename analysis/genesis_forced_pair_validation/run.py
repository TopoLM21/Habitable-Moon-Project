"""ONE forced25 versus 2x12.5 pair, reuse authenticated whole, no retries."""
import argparse,copy,hashlib,json,os,resource,shutil,subprocess,time,sys
from collections import defaultdict
from dataclasses import asdict,replace
from pathlib import Path
from unittest.mock import patch
import numpy as np
from analysis.genesis_adaptive_control import backend as b
from analysis.genesis_adaptive_control.controller import integrate,load_config,fingerprint,time_limit
from analysis.genesis_adaptive_control.metrics import assess
from analysis.genesis_be_candidate.adapter import dumps
from analysis.genesis_be_candidate.checkpoint import seal
from analysis.genesis_coupled_split_control.run import pinned,decode
from tectonics.genesis import GenesisParameters
from tectonics.genesis_tides import TidalParameters,TidalOrbitState
from tectonics.genesis_cutcell_loading import prepare_saved_loading
from tectonics.genesis_volume_slip_checkpoint import load_volume_slip_checkpoint

REPO=Path(__file__).resolve().parents[2]
PRODUCER='30b7154c4aa251a0ccdd4c73e20eb691608a3592'
THERMAL_AUDIT='/workspace/genesis-forced-coupled-25-01/thermal_audit.json'
THERMAL_AUDIT_SHA='7426c2a1f0e7497377d7f38e53801702b0b65bc92b455678c5f5e1a6bf8a9b12'
TH50='/workspace/genesis-implicit-cutcell-01/after.json'
TH50_SHA='632354594ccda5aaffeb8933d9ed95be96cb53037a2ce67d18cf407562d6f2e8'
DIRS=['tectonics','analysis/genesis_be_candidate','analysis/genesis_coupled_split_control','analysis/genesis_forced_coupled_validation','analysis/genesis_backward_euler_prototype']


def git(*args):return subprocess.check_output(['git',*args],cwd=REPO,text=True).strip()
def rawsha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compatible():
    if git('diff','--name-only',PRODUCER,'--',*DIRS):raise ValueError('COMPUTE_SOURCE_CHANGED: cannot reuse whole')
    root=pinned(b.PARENT,b.PARENT_SHA);whole=pinned(b.SEED,b.SEED_SHA);th=pinned(TH50,TH50_SHA)
    ta=pinned(THERMAL_AUDIT,THERMAL_AUDIT_SHA)
    assert seal({k:v for k,v in whole.items() if k!='sha256'})==whole
    params=root['source_packet']['source']['parameters'];ph=fingerprint(params)
    assert ph==whole['source_parameter_sha256']
    assert whole['source_ports']['orbit_before']==root['runtime_state']['thermal']['orbit']
    assert whole['initial_thermal']['time_myr']==root['runtime_state']['thermal']['time_myr']
    assert whole['thermal_packet']['fixed_cut_geometry']==th['fixed_cut_geometry']
    # Reconstruct original migration input from the same pinned cut/projection files.
    cutpath='/workspace/genesis-cutcell-feasibility-02/candidate.json';cutsha='6006521e7b3e22ce23894e63248d0145ee15d3065333c949701c730c6cab47ad'
    projpath='/workspace/genesis-coupled-thermal-projection-03/projected.json';projsha='d25fe1fd4e8b56b18ab8ec697b197c326d723e9421148e5d3726fb7affdd45a4'
    cut=pinned(cutpath,cutsha);proj=pinned(projpath,projsha);cells={c['identity']:c for c in proj['state']['cells']}
    initial=dict(time_myr=root['runtime_state']['thermal']['time_myr'],mass_kg=th['mass_kg'],energy_j=[c['mass_kg']*c['specific_h'] for c in cut['cells']]+[(cells[k]['energy_j']/cells[k]['mass_kg'])*cells[k]['mass_kg'] for k in ('deep_mantle','surface_water')])
    assert initial==whole['initial_thermal']
    assert whole['inputs']['before_displacement_m']==root['runtime_state']['displacement_m']
    assert whole['inputs']['before_velocity_m_s']==root['runtime_state']['velocity_m_s']
    paths=git('ls-tree','-r','--name-only',PRODUCER,'--',*DIRS).splitlines()
    hashes={p:rawsha(REPO/p) for p in paths if p.endswith('.py')}
    config=REPO/'analysis/genesis_adaptive_control/config.json'
    proof=dict(status='COMPATIBLE_REUSE_WHOLE',producer_commit=PRODUCER,controller_base_commit=git('rev-parse','HEAD'),
        whole_raw_sha256=b.SEED_SHA,canonical_raw_sha256=b.PARENT_SHA,physical_parameter_sha256=ph,initial_thermal_sha256=fingerprint(initial),
        fixed_cuts_sha256=fingerprint(th['fixed_cut_geometry']),coarse_solver_inputs_sha256=fingerprint(whole['inputs']),
        unchanged_compute_python_files=len(hashes),compute_manifest_sha256=fingerprint(hashes),compute_file_hashes=hashes,
        engineering_config_raw_sha256=rawsha(config),thermal_audit_raw_sha256=THERMAL_AUDIT_SHA,
        runner_raw_sha256=rawsha(__file__),runtime=dict(python=sys.version,numpy=np.__version__,openblas_threads=os.environ.get('OPENBLAS_NUM_THREADS'),omp_threads=os.environ.get('OMP_NUM_THREADS')),
        source_note='Same physical inputs and compute code. New controller budgets are post-trial tests, not trajectory parameters; wrapper/seal metadata differs.',
        pins={str(b.PARENT):b.PARENT_SHA,b.SEED:b.SEED_SHA,THERMAL_AUDIT:THERMAL_AUDIT_SHA,TH50:TH50_SHA,cutpath:cutsha,projpath:projsha})
    return proof,root,whole,th,ta


class PairBackend(b.ForcedBackend):
    def __init__(self,write,out,root,whole,th,ta):
        self.write=write;self.out=out;self.root=root;self.source=root['source_packet'];self.whole=whole;self.ta=ta
        params=self.source['source']['parameters'];self.gp=GenesisParameters(**params['thermal']);self.tides=TidalParameters(**params['tides'])
        self.cuts=th['fixed_cut_geometry'];self.moments=prepare_saved_loading(b.PARENT.read_bytes(),th).radial_moments
        self.model,self.parent=load_volume_slip_checkpoint(b.PARENT,expected_source_checkpoint_sha256=b.FORMATION_SHA)
        r=root['runtime_state'];s={k:r[k] for k in ('geometry','fault_history','displacement_m','velocity_m_s')}
        s.update(mechanics=r['mechanics']['state'],thermal={'time_myr':r['thermal']['time_myr']})
        p=seal(dict(format='genesis-forced-pair-pinned-start-0.1',canonical_raw_sha256=b.PARENT_SHA,state=s,thermal=whole['initial_thermal'],orbit_after=r['thermal']['orbit']))
        self.initial=dict(elapsed_years=0.,packet=p);self.origin=r['thermal']['time_myr'];self.new_trials=0;self.correctors=[];self.label='setup'
        restored=decode(self.parent,p,self.source)
        for key in ('source_identity','time_myr','accepted_steps','elastic_energy_j','cohorts'):
            assert dumps(asdict(restored.mechanics)[key])==dumps(asdict(self.parent.mechanics)[key])

    def advance(self,before,h,label,cap):
        self.label=label
        if label!='001_whole':
            if label not in ('001_half1','001_half2') or h!=12.5:raise RuntimeError('EXACT_ONE_PAIR_ONLY')
            self.new_trials+=1;result=super().advance(before,h,label,cap)
            self.correctors.append(dict(label=label,count=len(result['packet']['correctors'])))
        else:
            if h!=25 or before['elapsed_years']!=0:raise RuntimeError('WHOLE_WINDOW_MISMATCH')
            p=copy.deepcopy(self.whole);p.pop('sha256')
            p.update(format='genesis-adaptive-forced-BE-trial-0.1',adaptive_thermal_audit=self.ta,
                     adaptive_predecessor_sha256=before['packet']['sha256'],parent_sha256=before['packet']['sha256'])
            p=json.loads(dumps(seal(p)))
            for key in ('state','inputs','predictor','predictor_inputs','thermal','thermal_packet','source_ports','exchanges'):
                assert p[key]==self.whole[key]
            restored=decode(self.parent,p,self.source)
            assert json.loads(dumps(asdict(restored.mechanics)))==p['state']['mechanics']
            restored=replace(restored,thermal=replace(restored.thermal,orbit=TidalOrbitState(**p['orbit_after'])))
            assert asdict(restored.thermal.orbit)==p['orbit_after']
            result=dict(elapsed_years=25.,packet=p);self.write(label+'.json',result)
        # The next half really consumes the disk-decoded checkpoint, not a live
        # solver object. No redundant byte-identical restart copy is written.
        loaded=json.loads((self.out/(label+'.json')).read_bytes())
        assert fingerprint(loaded)==fingerprint(result)
        return loaded


class Timings:
    def __init__(self,backend):self.backend=backend;self.entries=[];self.stack=[]
    def wrap(self,name,fn):
        def run(*args,**kw):
            row=dict(label=self.backend.label,stage=name);t=time.perf_counter();self.stack.append([row,0.])
            try:return fn(*args,**kw)
            finally:
                _,children=self.stack.pop();elapsed=time.perf_counter()-t
                row.update(inclusive_seconds=elapsed,exclusive_seconds=elapsed-children);self.entries.append(row)
                if self.stack:self.stack[-1][1]+=elapsed
        return run


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path);args=parser.parse_args()
    out=args.output;out.mkdir(exist_ok=False);start=time.perf_counter();used=0;counter=0;raw=b.PARENT.read_bytes();backend=None;timings=None
    c=load_config(REPO/'analysis/genesis_adaptive_control/config.json')
    c.update(horizon_years=25.,initial_step_years=25.,max_step_years=25.,max_attempts=1,max_accepted_windows=1,max_rejections_per_window=1,runtime_seconds=600.,artifact_limit_bytes=200000000)
    report=dict(status='STOPPED',new_planetary_trials=0,canonical_advanced=False,limits=dict(seconds=600,artifact_bytes=200000000,address_space_bytes=4*1024**3),original_long_run_estimate=dict(trials=[36,42],correctors_at_four_per_trial=[144,168],BE_linear_solves_at_two_per_corrector=[288,336],independent_coupled_audits=[36,42],typed_endpoint_restarts=[36,42],seconds_per_old_complete_trial=102.8103))
    def write(name,value):
        nonlocal used
        data=(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
        if used+len(data)>199000000:raise RuntimeError('PAIR_ARTIFACT_LIMIT')
        if shutil.disk_usage(out).free-len(data)<2000000000:raise RuntimeError('DISK_RESERVE')
        with (out/name).open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        used+=len(data)
    def emit(kind,event):
        nonlocal counter
        counter+=1;write(f'event_{counter:02d}_{kind}.json',event)
    try:
      with time_limit(600):
        resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
        if np.finfo(np.longdouble).nmant!=63:raise ValueError('LONGDOUBLE_PRECISION')
        proof,root,whole,th,ta=compatible();write('compatibility.json',proof);write('config.json',c)
        mem={k:int(v) for k,v in (line.split() for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())}
        write('resources.json',dict(free_disk_bytes=shutil.disk_usage(out).free,cgroup_memory_max=Path('/sys/fs/cgroup/memory.max').read_text().strip(),cgroup_memory_current=int(Path('/sys/fs/cgroup/memory.current').read_text()),cgroup_memory_stat=mem,meminfo=Path('/proc/meminfo').read_text()))
        backend=PairBackend(write,out,root,whole,th,ta);write('initial.json',backend.initial)
        report['setup_seconds']=time.perf_counter()-start;timings=Timings(backend)
        names={'apply_ports':'sources','audit_ports':'source_audit','conduct':'conduction_with_audit','mechanical':'mechanical_correctors','decode':'typed_decode','audit_forced':'independent_coupled_audit','observe_packet':'observable_extraction'}
        from contextlib import ExitStack
        with ExitStack() as stack:
            for function,name in names.items():stack.enter_context(patch.object(b,function,timings.wrap(name,getattr(b,function))))
            adjusted={**c,'runtime_seconds':600-(time.perf_counter()-start)}
            result=integrate(backend.initial,adjusted,backend.advance,backend.audit,backend.observe,assess,emit)
        accepted=result.pop('accepted');report.update(controller=result,engineering_accepted=result['status']=='PASS',accepted_elapsed_years=accepted['elapsed_years'],accepted_packet_sha256=accepted['packet']['sha256'])
        complete=backend.new_trials==2 and (out/'001_half2.json').exists() and bool(result['history'])
        report['status']='PAIR_COMPLETE_ACCEPTED' if result['status']=='PASS' else ('PAIR_COMPLETE_REJECTED' if complete else 'PAIR_STOPPED_INCOMPLETE')
        report['working_fine_packet']='001_half2.json' if (out/'001_half2.json').exists() else None
        report['input_hashes_unchanged']=all(rawsha(p)==sha for p,sha in proof['pins'].items())
    except Exception as exc:
        report.update(status='STOPPED',error=f'{type(exc).__name__}: {exc}')
        import traceback
        with (out/'failure.txt').open('x') as f:f.write(traceback.format_exc())
    finally:
        report.update(total_seconds=time.perf_counter()-start,artifact_bytes_before_report=used,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,canonical_unchanged=b.PARENT.read_bytes()==raw)
        if backend is not None:report.update(new_planetary_trials=backend.new_trials,whole_reused=True,correctors=backend.correctors)
        if timings is not None:report['timings']=timings.entries
        report['output_raw_sha256']={p.name:rawsha(p) for p in out.iterdir() if p.is_file()}
        with (out/'report.json').open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in report.items() if k not in ('controller','timings','output_raw_sha256')},indent=2),flush=True)


if __name__=='__main__':main()
