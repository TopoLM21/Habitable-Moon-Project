"""One identical projection of the saved 100 Myr state; no dynamics rerun."""
import cProfile,pstats,io,json,sys,time,resource
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tectonics.checkpoint import load_checkpoint
from tectonics.topology import PlateTopologyManager,PlateTopologyParameters
from tectonics.regional_material import RegionalMaterialInventory,RegionalMaterialParameters
from tectonics.mesh import build_icosphere
HERE=Path(__file__).resolve().parent
cp=load_checkpoint(ROOT/'outputs_v131_regional/m3_g2_dt4/restart',PlateTopologyManager(PlateTopologyParameters()))
inv=RegionalMaterialInventory.restore(cp.regional_material,RegionalMaterialParameters(**cp.regional_material['parameters']))
before=inv.checkpoint();mesh=build_icosphere(2);prof=cProfile.Profile();start=time.perf_counter()
prof.enable();state,cycle,diagnostics=inv.project(mesh,cp.state,cp.cycle,5287.);prof.disable();wall=time.perf_counter()-start
fields=['cell_plate','crust_type','crust_thickness_km','crust_age_myr','continental_fraction','continental_volume_km3',
        'oceanic_volume_km3','oceanic_age_moment_km3_myr']
diff=[k for k in fields if not np.array_equal(getattr(state,k),getattr(cp.state,k))]
assert not diff and inv.checkpoint()==before
stream=io.StringIO();stats=pstats.Stats(prof,stream=stream).sort_stats('cumtime');stats.print_stats(18)
(HERE/'projection_profile.txt').write_text(stream.getvalue())
highs=sum(value[3] for key,value in stats.stats.items() if key[2]=='_highs_wrapper')
result=dict(wall_s=wall,max_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
    material_columns=1280,receiver_cells=320,highs_cumulative_s=highs,
    material_history_unchanged=True,reprojected_state_fields_equal=fields,diagnostics=diagnostics)
with (HERE/'projection_profile.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result),flush=True)
