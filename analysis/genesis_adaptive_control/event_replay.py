"""Synthetic-only event/controller CLI. Never loads a planetary model."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from .controller import integrate, load_config
from .events import Event, EventPolicy


def policy_from_config(config):
    required={'source_sha256','valid_years','resolution_years','max_evaluations','seconds','events'}
    if set(config)!=required:raise ValueError('EVENT_CONFIG_FIELDS')
    formulas={};events=[]
    for row in config['events']:
        if set(row)!={'kind','identity','units','transition','formula','center','offset'}:raise ValueError('EVENT_ROW_FIELDS')
        e=Event(**{k:row[k] for k in ('kind','identity','units','transition')})
        if row['formula'] not in ('decreasing','touch','constant'):raise ValueError('SYNTHETIC_FORMULA')
        if any(isinstance(row[k],bool) or not isinstance(row[k],(int,float)) or not math.isfinite(row[k]) for k in ('center','offset')):raise ValueError('SYNTHETIC_COEFFICIENT')
        formulas[(e.kind,e.identity)]=row;events.append(e)
    def enclosure(state,e,a,b):
        r=formulas[e.kind,e.identity];x=r['center'];offset=r['offset']
        if r['formula']=='constant':return (offset,offset)
        if r['formula']=='decreasing':return (x-b+offset,x-a+offset)
        return ((0. if a<=x<=b else min((a-x)**2,(b-x)**2))+offset,
                max((a-x)**2,(b-x)**2)+offset)
    # Include coefficients, not just function identity, in the restart binding.
    raw=Path(__file__).read_bytes()+json.dumps(config,sort_keys=True,allow_nan=False).encode()
    return EventPolicy(events,enclosure,source_sha256=config['source_sha256'],
        provider_sha256=hashlib.sha256(raw).hexdigest(),
        assumptions='Manufactured analytic affine/constant/squared margins, same prescribed path for all trials; no physical inference.',
        **{k:config[k] for k in required-{'source_sha256','events'}})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--controller-config',type=Path,required=True)
    p.add_argument('--event-config',type=Path,required=True)
    p.add_argument('--resume',type=Path,help='Previous synthetic report, unchanged configs required')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();c=load_config(a.controller_config)
    policy=policy_from_config(json.loads(a.event_config.read_text()))
    initial=dict(elapsed_years=0.,source_sha256=policy.contract['source_sha256'],value=0.)
    resume=json.loads(a.resume.read_text())['controller_state'] if a.resume else None
    def advance(s,h,*_):return {**s,'elapsed_years':s['elapsed_years']+h,'value':s['value']+h}
    # Exact manufactured linear output: zero step-doubling error. Real controller
    # metric/energy tests remain separate; no fabricated real-planet accuracy PASS.
    def assess(w,h1,h2,h,c):
        error=abs(w['value']-h2['value'])
        return dict(score=0. if error==0 else 2.,next_factor=c['max_factor'] if error==0 else c['min_factor'])
    result=integrate(initial,c,advance,lambda *_:{'status':'PASS'},lambda s:s,assess,
        event_policy=policy,resume=resume)
    result['synthetic_only']=True
    with a.output.open('x') as f:
        json.dump(result,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    print(json.dumps(dict(status=result['status'],reason=result['reason'],new_planetary_steps=0)))


if __name__=='__main__':main()
