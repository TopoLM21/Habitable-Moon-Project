"""Explicit opt-in CLI for a future bounded run. Not invoked by replay/tests."""
from .controller import time_limit
import argparse
import json
import resource
import platform
import subprocess
import numpy as np
import os
import shutil
import time
from pathlib import Path
from .controller import load_config,integrate,StopRun
from .metrics import assess
from .backend import ForcedBackend
from analysis.genesis_connected_step_validation.run import PARENT


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();c=load_config(args.config);out=args.output
    # All output is new; refuse symlinks/existing paths. Retain rejected trials.
    out.mkdir(exist_ok=False);used=0;counter=0;start=time.monotonic();raw=PARENT.read_bytes()
    def write(name,value):
        nonlocal used
        data=(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
        if used+len(data)>c['artifact_limit_bytes']-1000000:raise StopRun('ARTIFACT_BYTE_LIMIT')
        if shutil.disk_usage(out).free-len(data)<c['free_reserve_bytes']:raise StopRun('FREE_DISK_RESERVE')
        with (out/name).open('xb') as f:
            f.write(data);f.flush();os.fsync(f.fileno())
        used+=len(data)
    def emit(kind,event):
        nonlocal counter
        counter+=1;write(f'event_{counter:04d}_{kind}.json',event)
    report=dict(status='STOPPED',canonical_advanced=False)
    try:
        if platform.system()!='Linux' or np.finfo(np.longdouble).nmant!=63:raise StopRun('UNSUPPORTED_PLATFORM_PRECISION')
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[2],text=True).strip()
        write('provenance.json',dict(code_commit=code_commit,precision_nmant=63))
        write('config.json',c)
        resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
        # Single outer deadline includes setup/audits, even if a callback hangs.
        with time_limit(c['runtime_seconds']):
            backend=ForcedBackend(write);write('seed.json',backend.initial)
            adjusted={**c,'runtime_seconds':c['runtime_seconds']-(time.monotonic()-start)}
            if adjusted['runtime_seconds']<=0:raise StopRun('SETUP_RUNTIME_LIMIT')
            report=integrate(backend.initial,adjusted,backend.advance,backend.audit,backend.observe,assess,emit)
            accepted=report.pop('accepted');report['accepted_elapsed_years']=accepted['elapsed_years']
            report['accepted_packet_sha256']=accepted['packet']['sha256']
            # Cursor references archived half2; no unrequested canonical write.
    except Exception as exc:
        report.update(status='STOPPED',reason=f'{type(exc).__name__}: {exc}')
    finally:
        report.update(total_seconds=time.monotonic()-start,artifact_bytes=used,canonical_unchanged=PARENT.read_bytes()==raw,
                      peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # Small final stop record must survive the data-byte limit; never overwrite.
        with (out/'report.json').open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in report.items() if k!='history'},indent=2))


if __name__=='__main__':main()
