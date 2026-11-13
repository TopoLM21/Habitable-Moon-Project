"""Wall-clock native LP calls: cProfile does not capture their native runtime."""
import json,time,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import tectonics.regional_material as module
from tectonics.checkpoint import load_checkpoint
from tectonics.topology import PlateTopologyManager,PlateTopologyParameters
from tectonics.mesh import build_icosphere
HERE=Path(__file__).resolve().parent
cp=load_checkpoint(ROOT/'outputs_v131_regional/m3_g2_dt4/restart',PlateTopologyManager(PlateTopologyParameters()))
inv=module.RegionalMaterialInventory.restore(cp.regional_material,module.RegionalMaterialParameters(**cp.regional_material['parameters']))
original=module.linprog;calls=[]
def timed(*args,**kwargs):
    start=time.perf_counter();result=original(*args,**kwargs)
    calls.append(dict(wall_s=time.perf_counter()-start,variables=len(args[0]),status=result.status))
    return result
before=inv.checkpoint();module.linprog=timed;start=time.perf_counter()
try:state,cycle,diag=inv.project(build_icosphere(2),cp.state,cp.cycle,5287.)
finally:module.linprog=original
wall=time.perf_counter()-start
fields=['cell_plate','crust_type','crust_thickness_km','crust_age_myr','continental_fraction','continental_volume_km3',
        'oceanic_volume_km3','oceanic_age_moment_km3_myr']
assert all(np.array_equal(getattr(state,k),getattr(cp.state,k)) for k in fields)
assert inv.checkpoint()==before
result=dict(wall_s=wall,linprog_calls=calls,linprog_wall_fraction=sum(x['wall_s'] for x in calls)/wall,
            material_history_unchanged=True,projected_fields_identical=True,diagnostics=diag)
with (HERE/'projection_timing.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result),flush=True)
