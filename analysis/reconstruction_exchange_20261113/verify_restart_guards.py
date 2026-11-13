"""Expected failures: do not invent or silently discard saved enthalpy history."""
import json,os,subprocess,sys
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
base=yaml.safe_load((HERE/'inventory_control.yaml').read_text());results=[]
for name,resume,enabled,message in [
    ('guard_disabled',ROOT/'outputs_v131_exchange/corrected_g2_dt4/restart',False,'requires its finite inventory mode'),
    ('guard_invented',ROOT/'outputs_v131_reconstruction/legacy_original/restart',True,'no recoverable material enthalpy'),
]:
    cfg=dict(base);cfg['material_exchange']={**base['material_exchange'],'enabled':enabled}
    parameter=HERE/f'{name}.yaml';out=ROOT/'outputs_v131_exchange'/name
    if out.exists() or parameter.exists():raise FileExistsError(name)
    with parameter.open('x') as f:yaml.safe_dump(cfg,f,sort_keys=False)
    with (HERE/f'{name}.log').open('x') as log:
        p=subprocess.run([sys.executable,'run_long_evolution_v131.py','--config',str(parameter),
                          '--resume',str(resume),'--output',str(out),'--end-time','104','--dt','4'],
                         cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=30,
                         env={**os.environ,'MPLCONFIGDIR':'/tmp/hm_reconstruction_mpl'})
    assert p.returncode!=0 and message in (HERE/f'{name}.log').read_text()
    assert not list(out.rglob('state.npz'))
    results.append({'case':name,'returncode':p.returncode,'expected_message':message,'no_new_checkpoint':True})
with (HERE/'restart_guards.json').open('x') as f:json.dump(results,f,indent=2)
print(json.dumps(results))
