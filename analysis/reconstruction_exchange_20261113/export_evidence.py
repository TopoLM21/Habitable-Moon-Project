"""Save compact independent restart copies, comparison table and scientific plot.

Run once; saved states are real copies, never archive hard links. Source commits
are carried in the result records. Old evidence is not overwritten.
"""
import csv,hashlib,json,math,shutil,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tectonics.checkpoint import load_checkpoint
from tectonics.topology import PlateTopologyManager,PlateTopologyParameters
from tectonics.mesh import build_icosphere
from visualization.raster import rasterize_cells
HERE=Path(__file__).resolve().parent;OUT=ROOT/'outputs_v131_exchange'
names=['corrected_g2_dt4','corrected_g2_dt2','corrected_g2_dt1','corrected_g1_dt2','corrected_g3_dt2']
rows=[]
for name in names:
    x=json.loads((HERE/f'{name}_result.json').read_text());b=x['material_budget']
    with np.load(OUT/name/'restart/state.npz') as z:
        continent=z['continental_fraction'];area=build_icosphere(x['subdivisions']).physical_cell_areas_km2(5287.)
        area_fraction=float(area@continent/sum(area))
    row={'case':name,'cells':x['runtime']['mesh_cells'],'dt_myr':x['dt_myr'],
         'continental_generated_million_km3':x['continental_generated_km3']/1e6,
         'juvenile_area_million_km2':x['juvenile_area_km2']/1e6,
         'arc_thickening_million_km3':x['arc_thickening_km3']/1e6,
         'basalt_created_million_km3':x['ocean_created_km3']/1e6,
         'basalt_returned_million_km3':x['ocean_recycled_km3']/1e6,
         'continental_area_fraction':area_fraction,'max_crust_thickness_km':x['thickness_max_km'],
         'temperature_mantle_k':b['mantle_temperature_k'],'temperature_crust_k':b['crust_temperature_k'],
         'loop_s':x['runtime']['integration_wall_seconds'],'process_s':x['resources']['wall_s'],
         'max_rss_mib':x['resources']['max_rss_kib']/1024,
         'model_myr_per_loop_s':100/x['runtime']['integration_wall_seconds'],
         'max_thermal_residual_j':x['max_energy_residual_j'],'max_origin_residual_kg':x['max_origin_residual_kg'],
         'continental_volume_error_km3':x['continental_ledger_error_km3'],
         'unfilled_footprint_fraction':x['max_unfilled_ocean_footprint_km2']/(4*math.pi*5287.**2),
         'transport_commits':x['transport_commits'],'source_commit':x['source_commit']}
    rows.append(row)
with (HERE/'comparison.csv').open('x',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
with (HERE/'summary.json').open('x') as f:json.dump(rows,f,indent=2,allow_nan=False)

saved=[]
for name in ('corrected_g2_dt4','corrected_g2_dt2','corrected_first48'):
    dest=HERE/'saved_states'/name;dest.mkdir(parents=True,exist_ok=False)
    for file in ('meta.json','state.npz'):
        source=OUT/name/'restart'/file;target=dest/file
        shutil.copy2(source,target)
        assert target.stat().st_nlink==1 and target.stat().st_ino!=source.stat().st_ino
        saved.append({'path':str(target.relative_to(ROOT)),'bytes':target.stat().st_size,
                      'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
                      'source_commit':json.loads((HERE/f'{name}_result.json').read_text())['source_commit']})
with (HERE/'saved_states_manifest.json').open('x') as f:json.dump(saved,f,indent=2)

fig=plt.figure(figsize=(12,8.6),layout='constrained')
gs=fig.add_gridspec(2,2)
mesh=build_icosphere(2);cps={}
for name in names[:3]:
    cps[name]=load_checkpoint(OUT/name/'restart',PlateTopologyManager(PlateTopologyParameters()))
for col,name in enumerate((names[0],names[2])):
    ax=fig.add_subplot(gs[0,col],projection='mollweide')
    xe,ye,data=rasterize_cells(mesh,cps[name].state.continental_fraction,width=360,height=180)
    im=ax.pcolormesh(xe,ye,data,cmap='YlGnBu_r',vmin=0,vmax=1,shading='auto',rasterized=True)
    ax.grid(alpha=.25);ax.tick_params(labelsize=7)
    ax.set_title(f'100 Myr, 320 cells, dt = {rows[col*2]["dt_myr"]:g} Myr',fontsize=11)
    fig.colorbar(im,ax=ax,orientation='horizontal',fraction=.065,pad=.07,label='Continental material fraction (cell sampling)')
ax=fig.add_subplot(gs[1,0]);colors=['#23769c','#c57728','#596799']
ax.bar([0,1,2],[r['continental_generated_million_km3'] for r in rows[:3]],color=colors,width=.6)
ax.set_xticks([0,1,2],['4 Myr','2 Myr','1 Myr']);ax.set_xlabel('Outer step, identical initial state')
ax.set_ylabel('Produced continental crust, million km³');ax.set_title('Production is NOT time converged',fontsize=11)
ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
for i,r in enumerate(rows[:3]):ax.text(i,r['continental_generated_million_km3']+2,f'{r["continental_generated_million_km3"]:.1f}',ha='center')
ax.set_ylim(0,170)
ax=fig.add_subplot(gs[1,1])
for color,name,r in zip(colors,names[:3],rows):
    history=cps[name].material_exchange['history']
    ax.plot([x['time_myr'] for x in history],[x['mantle_temperature_k'] for x in history],
            label=f'dt = {r["dt_myr"]:g} Myr',color=color)
ax.set_xlabel('Time, Myr');ax.set_ylabel('Mean mantle temperature, K');ax.set_title('Finite two-box thermal inventory',fontsize=11)
ax.grid(alpha=.2);ax.legend(frameon=False)
fig.suptitle('EXPERIMENTAL reconstruction — existing initial regions, no paid fracture\nMass/sensible heat conserved; geology and phase/mechanical energy remain unvalidated',fontsize=13)
fig.savefig(HERE/'comparison.png',dpi=135);plt.close(fig)
print(json.dumps({'rows':len(rows),'saved_state_bytes':sum(x['bytes'] for x in saved),'figure':'comparison.png'},indent=2))
