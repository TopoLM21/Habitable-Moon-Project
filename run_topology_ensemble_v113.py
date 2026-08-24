#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json,subprocess,sys
from pathlib import Path
from collections import Counter
import numpy as np,yaml
from tectonics.checkpoint import load_checkpoint
from tectonics.mesh import build_icosphere
from tectonics.topology import PlateTopologyManager,PlateTopologyParameters

def args():
 p=argparse.ArgumentParser();p.add_argument('--config',default='configs/canonical_moon.yaml');p.add_argument('--output',default='ensemble_v113_topology');p.add_argument('--seeds',default='20260806,20260807,20260808,20260809,20260810');p.add_argument('--subdivisions',default='3,4');p.add_argument('--end-time',type=float,default=500.0);p.add_argument('--dt',type=float,default=4.0);return p.parse_args()
def main():
 a=args();base=yaml.safe_load(Path(a.config).read_text());out=Path(a.output);out.mkdir(parents=True,exist_ok=True);rows=[]
 for sub in [int(x) for x in a.subdivisions.split(',') if x.strip()]:
  for seed in [int(x) for x in a.seeds.split(',') if x.strip()]:
   d=out/f'sub{sub}'/f'seed_{seed}';d.mkdir(parents=True,exist_ok=True);cfg=json.loads(json.dumps(base));cfg['mesh']['subdivisions']=sub;cfg['plates']['seed']=seed;cp=d/'cp';cfgp=d/'config.yaml';cfgp.write_text(yaml.safe_dump(cfg,sort_keys=False))
   subprocess.run([sys.executable,'run_long_evolution_v113.py','--config',str(cfgp),'--output',str(d/'run'),'--end-time',str(a.end_time),'--dt',str(a.dt),'--checkpoint',str(cp)],check=True)
   pp=PlateTopologyParameters(**{k:v for k,v in cfg['plate_topology'].items() if k in PlateTopologyParameters.__dataclass_fields__});c=load_checkpoint(cp,PlateTopologyManager(pp));mesh=build_icosphere(sub);areas=mesh.physical_cell_areas_km2(float(cfg['moon']['radius_km']));pa=np.bincount(c.state.cell_plate,weights=areas,minlength=len(c.system.plates));ev=Counter(e.get('kind','?') for e in c.events)
   rows.append({'subdivision':sub,'seed':seed,'plate_count':len(c.system.plates),'largest_plate_fraction':float(pa.max()/areas.sum()),'event_count':len(c.events),'merge':ev['merge'],'disconnect_split':ev['disconnect_split'],'absorb':ev['absorb'],'vanish':ev['vanish']})
 with (out/'summary.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
