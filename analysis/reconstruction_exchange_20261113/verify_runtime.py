"""Bounded new runtime checks. No overwrite, no automatic failed-case retry.

Run from repository root: .venv/bin/python analysis/reconstruction_exchange_20261113/verify_runtime.py
Each child has a 120 s ceiling; cases run sequentially for comparable timings.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tectonics.checkpoint import load_checkpoint
from tectonics.topology import PlateTopologyManager,PlateTopologyParameters
from tectonics.material_exchange import MaterialExchangeBudget,MaterialExchangeParameters
from tectonics.thermal import ThermalParameters

HERE=Path(__file__).resolve().parent
OUT=ROOT/'outputs_v131_exchange'
BASE=yaml.safe_load((HERE/'inventory_control.yaml').read_text())
OLD=yaml.safe_load((ROOT/'analysis/reconstruction_20261113/coarse_thermal.yaml').read_text())


def run_case(name,sub,dt,end=100.,resume=None,mode='inventory'):
    output=OUT/name;parameter=HERE/f'{name}.yaml'
    if output.exists() or parameter.exists():raise FileExistsError(name)
    cfg=deepcopy(BASE if mode=='inventory' else OLD)
    cfg['mesh']['subdivisions']=sub
    if mode=='maturation_only':cfg['continental_cycle']['analytic_maturation']=True
    if mode=='legacy':cfg['thermal_integration']['enabled']=False
    parameter.write_text(yaml.safe_dump(cfg,sort_keys=False))
    command=[sys.executable,'run_long_evolution_v131.py','--config',str(parameter),
             '--output',str(output),'--checkpoint',str(output/'restart'),'--dt',str(dt),'--end-time',str(end)]
    if resume:command+=['--resume',str(OUT/resume/'restart')]
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    start=time.perf_counter()
    with (HERE/f'{name}.log').open('x') as log:
        process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                                 env={**os.environ,'MPLCONFIGDIR':'/tmp/hm_reconstruction_mpl'})
        while True:
            pid,status,usage=os.wait4(process.pid,os.WNOHANG)
            if pid:
                process.returncode=os.waitstatus_to_exitcode(status);break
            if time.perf_counter()-start>120:
                process.kill();process.wait()
                raise TimeoutError(f'{name}: exceeded 120 seconds, inspect saved log')
            time.sleep(.05)
    resources={'max_rss_kib':usage.ru_maxrss,'user_s':usage.ru_utime,'system_s':usage.ru_stime,
               'wall_s':time.perf_counter()-start,'returncode':process.returncode}
    (HERE/f'{name}_resources.json').write_text(json.dumps(resources,indent=2)+'\n')
    if process.returncode:raise RuntimeError(f'{name} failed; no automatic retry')
    cp=load_checkpoint(output/'restart',PlateTopologyManager(PlateTopologyParameters()))
    item={'case':name,'source_commit':source,'mode':mode,'subdivisions':sub,'dt_myr':dt,'end_myr':end,
          'parameters_sha256':hashlib.sha256(parameter.read_bytes()).hexdigest(),'resources':resources,
          'plate_count':len(cp.system.plates),'topology_events':cp.events,
          'continental_generated_km3':cp.cycle.cumulative_generated_volume_km3,
          'juvenile_area_km2':cp.cycle.cumulative_generated_area_km2,
          'arc_thickening_km3':sum(x['arc_thickening_volume_km3'] for x in cp.cycle_rows),
          'transport_commits':cp.transport_state.cumulative_commit_count,
          'continental_ledger_error_km3':max(abs(x['global_continental_ledger_error_km3']) for x in cp.sediment_rows),
          'thickness_min_km':float(cp.state.crust_thickness_km.min()),
          'thickness_max_km':float(cp.state.crust_thickness_km.max()),
          'temperature_k':cp.thermal.mantle_temperature_k,'full_energy_closure':False}
    if (output/'runtime_stats.json').exists():item['runtime']=json.loads((output/'runtime_stats.json').read_text())
    if cp.material_exchange:
        data=cp.material_exchange;budget=MaterialExchangeBudget.restore(data,MaterialExchangeParameters(**data['options']))
        item['material_budget']=budget.diagnostics(ThermalParameters(**cfg['thermal']))
        item['max_energy_residual_j']=max(abs(x['energy_residual_j']) for x in data['history'])
        item['max_origin_residual_kg']=max(abs(v) for x in data['history'] for v in x['origin_mass_residual_kg'])
        item['ocean_created_km3']=sum(x['oceanic_created_volume_km3'] for x in cp.lithosphere_rows)+sum(x['oceanic_created_volume_km3'] for x in cp.cycle_rows)
        item['ocean_recycled_km3']=sum(x['oceanic_recycled_volume_km3'] for x in cp.lithosphere_rows)+sum(x['oceanic_recycled_volume_km3'] for x in cp.cycle_rows)
    (HERE/f'{name}_result.json').write_text(json.dumps(item,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:item[k] for k in ('case','continental_generated_km3','temperature_k','plate_count','resources')}),flush=True)


def compare(first,second,name):
    with np.load(first/'state.npz') as a,np.load(second/'state.npz') as b:
        diffs=[k for k in a.files if k not in b.files or not np.array_equal(a[k],b[k])]
        samekeys=set(a.files)==set(b.files);count=len(a.files)
    ma,mb=[json.loads((p/'meta.json').read_text()) for p in (first,second)]
    result={'array_count':count,'same_keys':samekeys,'array_differences':diffs,'metadata_equal':ma==mb,
            'metadata_differences':[k for k in ma if ma[k]!=mb.get(k)]}
    (HERE/f'{name}.json').write_text(json.dumps(result,indent=2)+'\n')
    assert not diffs and samekeys and ma==mb,result


if __name__=='__main__':
    for name,sub,dt,end,resume,mode in [
        ('finite_g2_dt4',2,4.,100.,None,'inventory'),('finite_g2_dt2',2,2.,100.,None,'inventory'),
        ('finite_g2_dt1',2,1.,100.,None,'inventory'),('finite_g1_dt2',1,2.,100.,None,'inventory'),
        ('finite_g3_dt2',3,2.,100.,None,'inventory'),('finite_first48',2,4.,48.,None,'inventory'),
        ('finite_restart100',2,4.,100.,'finite_first48','inventory'),
        ('maturation_g2_dt4',2,4.,100.,None,'maturation_only'),('maturation_g2_dt2',2,2.,100.,None,'maturation_only'),
        ('legacy20',2,4.,20.,None,'legacy'),
    ]:run_case(name,sub,dt,end,resume,mode)
    compare(OUT/'finite_g2_dt4/restart',OUT/'finite_restart100/restart','restart_comparison')
    compare(ROOT/'outputs_v131_reconstruction/legacy_original/restart',OUT/'legacy20/restart','legacy_comparison')
