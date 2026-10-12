"""Read-only existing trial audits and controller replay. No solver invocation."""
import argparse
import hashlib
import json
from pathlib import Path
from .controller import load_config,integrate
from .metrics import observe_packet,assess,work_budget
from .backend import SEED,SEED_SHA
from analysis.genesis_connected_step_validation.run import PARENT,PARENT_SHA
from analysis.genesis_coupled_split_control.run import pinned,root_from_previous,audit_step
from analysis.genesis_forced_coupled_validation.audit import audit_ports
from tectonics.genesis import GenesisParameters
from tectonics.genesis_tides import TidalParameters

PATHS={
 'whole':('/workspace/genesis-cutcell-mechanical-02/candidate.json','5a0cfcd3edfd57f4f05b59e8b408b09c25ac2a704068609e22029ab865230f2e'),
 'half1':('/workspace/genesis-coupled-split-25-01/step_1.json','5a0a1f53c9a01237f626b03c7119f98475d8392e0e84349b11bfd1e50c50b9ed'),
 'half2':('/workspace/genesis-coupled-split-25-01/step_2.json','ad8810c6f6c537e6252deeaccafda569dcc403431a3442651934b7086992ad6d'),
 'forced':(SEED,SEED_SHA)}


def replay(c):
    root=pinned(PARENT,PARENT_SHA);source=root['source_packet'];packets={k:pinned(*v) for k,v in PATHS.items()}
    thermal50=pinned('/workspace/genesis-implicit-cutcell-01/after.json','632354594ccda5aaffeb8933d9ed95be96cb53037a2ce67d18cf407562d6f2e8')
    audits={}
    for key in ('whole','half1','half2','forced'):
        r=root_from_previous(root,packets['half1']) if key=='half2' else root
        thermal=thermal50 if key=='whole' else packets[key]['thermal_packet']
        audits[key]=audit_step(r,thermal,packets[key],source)
    params=source['source']['parameters'];f=packets['forced']
    source_audit=audit_ports(f['initial_thermal'],f['source_endpoint'],f['source_ports'],GenesisParameters(**params['thermal']),TidalParameters(**params['tides']))
    observations={k:observe_packet(p,source) for k,p in packets.items()}
    checks=assess(observations['whole'],observations['half1'],observations['half2'],50.,c)
    events=[]
    def saved(before,h,label,cap):
        name=label.split('_',1)[1]
        if (before['elapsed_years'],h)!={'whole':(0.,50.),'half1':(0.,25.),'half2':(25.,25.)}[name]:raise RuntimeError('REPLAY_NO_SAVED_TRIAL_FOR_REQUESTED_DT')
        return dict(elapsed_years=before['elapsed_years']+h,key=name)
    rc={**c,'initial_step_years':50.,'max_step_years':50.,'horizon_years':50.,'max_attempts':1}
    run=integrate(dict(elapsed_years=0.),rc,saved,lambda b,a:audits[a['key']],lambda a:observations[a['key']],assess,lambda k,v:events.append(dict(kind=k,event=v)))
    return dict(status='PASS_REPLAY_EXECUTED',new_planetary_solver_calls=0,canonical_sha256=PARENT_SHA,
        inputs={k:dict(path=p,sha256=s) for k,(p,s) in PATHS.items()},unforced_step_doubling=checks,controller_replay=run,
        forced25_single_trial_work=work_budget([observations['forced']],25.,c),forced_step_doubling='UNAVAILABLE: no saved forced whole/half pair; do not fabricate one',
        independent_audits=audits,forced_source_audit=source_audit,events=events,
        input_hashes_unchanged=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==s for p,s in PATHS.values()))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    c=load_config(Path(__file__).with_name('config.json'));r=replay(c)
    with args.output.open('x') as f:json.dump(r,f,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in r.items() if k not in ('events','independent_audits','forced_source_audit')},indent=2))


if __name__=='__main__':main()
