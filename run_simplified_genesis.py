"""Explicit opt-in simplified Genesis pilot; no canonical/default changes."""
import argparse,json,time
from pathlib import Path
from tectonics.simplified_genesis import SimplifiedGenesis
from tectonics.genesis_transfer_bridge import digest,encoded
from analysis.genesis_adaptive_control.controller import time_limit


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--restart',type=Path);p.add_argument('--steps',type=int,default=3);p.add_argument('--years',type=float,default=25.);args=p.parse_args()
    if not 1<=args.steps<=100:raise ValueError('Pilot permits 1..100 steps')
    args.output.mkdir(exist_ok=False);start=time.perf_counter();report=dict(status='STOPPED',canonical_advanced=False,completed_steps=0)
    try:
      with time_limit(280):
        if args.restart:model=SimplifiedGenesis.load(args.restart)
        else:
            from analysis.simplified_genesis_validation.import_source import build
            model=build()
        model.save(args.output/'initial.json');report['start_myr']=model.state['time_myr'];report['initial_domains']=len(model.state['omega'])
        for i in range(args.steps):
            model.advance(args.years);model.save(args.output/f'step_{i+1}.json');report['completed_steps']+=1
        # Resume one of the requested steps from checkpoint and compare against
        # its already-computed endpoint; no extra interval is appended.
        checkpoint=args.output/('initial.json' if args.steps==1 else f'step_{args.steps-1}.json')
        restored=SimplifiedGenesis.load(checkpoint);restored.advance(args.years)
        assert encoded(restored.state)==encoded(model.state)
        model.save(args.output/'final.json');report.update(status='COMPLETED_EFFECTIVE_MODEL_PILOT',end_myr=model.state['time_myr'],domains=len(model.state['omega']),events=model.state['events'],last_ledger=model.state['last_ledger'],restart_next_step_exact=True,replayed_steps=1,mature_compatibility=model.mature_compatibility(),source_pins=model.config.get('source_pins',{}),passive_reserve_j=model.state['passive_reserve_j'])
        for name,pin in model.config.get('source_pins',{}).items():assert digest(Path(name).read_bytes())==pin
    except Exception as exc:
        report['error']=type(exc).__name__+': '+str(exc)
    report.update(seconds=time.perf_counter()-start,artifact_bytes=sum(x.stat().st_size for x in args.output.iterdir()),output=str(args.output))
    with (args.output/'report.json').open('xb') as f:f.write(encoded(report))
    print(json.dumps(report,indent=2))
    if report['status']=='STOPPED':raise SystemExit(1)

if __name__=='__main__':main()
