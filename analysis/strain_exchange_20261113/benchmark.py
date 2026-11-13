"""Fresh process timing (does not require /usr/bin/time)."""
import json
import os
import resource
import subprocess
import sys
import time
from verify import HERE
from tectonics.strain_exchange import StrainExchange

start=time.perf_counter()
with (HERE/'native_fresh100_run.log').open('x') as log:
    completed=subprocess.run([sys.executable,str(HERE/'verify.py'),'native_fresh100','--end','100'],
        stdout=log,stderr=subprocess.STDOUT,check=False,env=os.environ.copy())
elapsed=time.perf_counter()-start;usage=resource.getrusage(resource.RUSAGE_CHILDREN)
result=dict(wall_s=elapsed,user_s=usage.ru_utime,system_s=usage.ru_stime,
    peak_rss_mib=usage.ru_maxrss/1024,exit_code=completed.returncode,
    scope='fresh 1280-cell linked 100 Myr, including startup, setup, save/load and coverage',
    thread_environment={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')})
if completed.returncode==0:
    fresh=StrainExchange.load(HERE/'states/native_fresh100')
    reference=StrainExchange.load(HERE/'states/native_g3_100')
    assert fresh.checkpoint()==reference.checkpoint()
    result['exact_reference_checkpoint']=True
with (HERE/'fresh_resources.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result),flush=True)
if completed.returncode:raise SystemExit(completed.returncode)
