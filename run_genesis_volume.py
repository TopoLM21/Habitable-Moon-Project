"""Bounded, headless VolumeSlipModel runtime; midpoint stays the default.

inspect/pack never advance mechanics. advance is an explicit separate action.
"""
import argparse,json,sys,resource
from pathlib import Path


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['inspect','pack','advance'])
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path)
    p.add_argument('--integrator',choices=['midpoint','be50','be25'])
    p.add_argument('--config',type=Path)
    p.add_argument('--chain',type=Path,nargs='+')
    p.add_argument('--steps',type=int)
    p.add_argument('--wall-seconds',type=float)
    a=p.parse_args(argv)
    try:
        c={} if a.config is None else json.loads(a.config.read_text())
        if type(c)!=dict or set(c)-{'integrator','steps','wall_seconds'}:raise ValueError('Config only accepts integrator, steps, wall_seconds; physics stays in checkpoint')
        mode=a.integrator if a.integrator is not None else c.get('integrator','midpoint')
        steps=a.steps if a.steps is not None else c.get('steps',1)
        wall=a.wall_seconds if a.wall_seconds is not None else c.get('wall_seconds',300)
        if mode not in ('midpoint','be50','be25'):raise ValueError('Unknown integrator')
        if type(steps)!=int or not 1<=steps<=4:raise ValueError('steps must be an integer from 1 to 4')
        if type(wall) not in (int,float) or not 30<=wall<=600:raise ValueError('wall_seconds must be finite, 30..600')
        if a.action!='inspect' and a.output is None:raise ValueError('Explicit new output required')
        if a.output is not None and a.output.exists():raise ValueError('Output exists; overwrite is forbidden')
        if a.action=='pack' and (mode=='midpoint' or not a.chain):raise ValueError('pack needs explicit BE mode and saved chain')
        if a.action!='pack' and a.chain:raise ValueError('--chain is only valid for pack')
        from tectonics.genesis_volume_runtime import load_runtime
        from analysis.genesis_connected_step_validation.run import deadline
        resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
        with deadline(wall):
            runtime=load_runtime(a.input,integrator=mode)
            if a.action=='pack':runtime.attach_chain([json.loads(x.read_bytes()) for x in a.chain])
            if a.action=='advance':
                for _ in range(steps):runtime.advance()
            if a.output is not None:runtime.save(a.output)
            print(json.dumps({'status':'PASS','action':a.action,**runtime.describe()},sort_keys=True),flush=True)
        return 0
    except (ValueError,OSError,RuntimeError,KeyError,TypeError,TimeoutError) as e:
        print('Volume runtime error: '+str(e),file=sys.stderr,flush=True);return 1


if __name__=='__main__':raise SystemExit(main())
