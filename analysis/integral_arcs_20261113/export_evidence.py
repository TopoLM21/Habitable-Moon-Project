"""Compact, independent checks and separately copied native restart evidence."""
import csv
from decimal import Decimal,localcontext
import hashlib,json,math,shutil,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tectonics.checkpoint import load_checkpoint
from tectonics.topology import PlateTopologyManager,PlateTopologyParameters
from tectonics.mesh import build_icosphere
from tectonics.kinematics import cell_velocities_km_per_myr
from visualization.raster import rasterize_cells
HERE=Path(__file__).resolve().parent;OUT=ROOT/'outputs_v131_integral_arcs';R=5287.

def read(name):return load_checkpoint(OUT/name/'restart',PlateTopologyManager(PlateTopologyParameters()))
def ds(values):return sum((Decimal.from_float(float(x)) for x in values),Decimal(0))

names=['final_g2_dt4','final_g2_dt2','final_g2_dt1','final_g1_dt2','final_g3_dt2',
       'reference_g2_dt4','reference_g2_dt2','reference_g2_dt1']
rows=[];independent=[];cps={}
with localcontext() as ctx:
    ctx.prec=60
    for name in names:
        x=json.loads((HERE/f'{name}_result.json').read_text());cp=read(name);cps[name]=cp
        mesh=build_icosphere(x['subdivisions']);a=mesh.physical_cell_areas_km2(R)
        init=read(f'initial_same_g{x["subdivisions"]}')
        budget=cp.material_exchange;d=x['material_budget']
        # Independent Decimal sums from raw geometric arrays and saved process rows.
        f=lambda c:ds(c.state.continental_volume_km3)+ds(c.state.sediment_volume_km3)
        frecycled=ds([cp.cycle.cumulative_recycled_volume_km3,cp.sediment_budget.deep_recycled_sediment_volume_km3,
                      cp.sediment_budget.cumulative_rift_recycled_volume_km3])
        fr=f(cp)-f(init)-ds([cp.cycle.cumulative_generated_volume_km3])+frecycled
        born=ds(z['oceanic_created_volume_km3'] for z in cp.lithosphere_rows+cp.cycle_rows)
        returned=ds(z['oceanic_recycled_volume_km3'] for z in cp.lithosphere_rows+cp.cycle_rows)
        ores=ds(cp.state.oceanic_volume_km3)-ds(init.state.oceanic_volume_km3)-born+returned
        eres=ds(budget['energy_m_j']+budget['energy_c_j'])-ds([budget['initial_energy_j']])-ds(budget['radiogenic_input_j'])-ds(budget['orbital_input_j'])+ds(budget['surface_output_j'])
        origins=[ds([budget['origin_hi'][i][j] for i in range(3)]+[budget['origin_lo'][i][j] for i in range(3)])-ds([budget['origin_mass_kg'][j]]) for j in range(3)]
        assert abs(fr)<Decimal('0.0001') and abs(ores)<Decimal('0.0001')
        independent.append({'case':name,'continental_volume_residual_km3':str(fr),'basalt_volume_residual_km3':str(ores),
            'sensible_energy_residual_j':str(eres),'origin_mass_residuals_kg':list(map(str,origins)),
            'method':'60-digit Decimal.from_float sums of native arrays and gross exchanges; not the runtime diagnostic'})
        row={'case':name,'cells':mesh.cell_count,'dt_myr':x['dt_myr'],
            'continental_produced_million_km3':x['continental_generated_km3']/1e6,
            'arc_thickening_million_km3':x['arc_thickening_km3']/1e6,'juvenile_area_million_km2':x['juvenile_area_km2']/1e6,
            'basalt_created_million_km3':x['ocean_created_km3']/1e6,'basalt_returned_million_km3':x['ocean_recycled_km3']/1e6,
            'h_min_km':x['thickness_min_km'],'h_max_km':x['thickness_max_km'],'plates':x['plate_count'],
            'mantle_k':d['mantle_temperature_k'],'crust_k':d['crust_temperature_k'],
            'max_unfilled_footprint_fraction':x['max_unfilled_ocean_footprint_km2']/(4*math.pi*R**2),
            'final_area_rms_speed_mm_per_year':float(np.sqrt(np.sum(a*np.sum(cell_velocities_km_per_myr(mesh,cp.system,R)**2,axis=1))/sum(a))),
            'transport_commits':x['transport_commits'],'sensible_energy_residual_j':float(eres),
            'global_mechanical_energy_residual_j':None,'process_s':x['resources']['wall_s'],
            'loop_s':x['runtime']['integration_wall_seconds'],'rss_mib':x['resources']['max_rss_kib']/1024,
            'source_commit':x['source_commit']}
        rows.append(row)
with (HERE/'comparison.csv').open('x',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
with (HERE/'summary.json').open('x') as f:json.dump(rows,f,indent=2,allow_nan=False)
with (HERE/'independent_balances.json').open('x') as f:json.dump(independent,f,indent=2)

saved=[]
for name in ('initial_same_g1','initial_same_g2','initial_same_g3','final_first48','final_g2_dt4','final_g2_dt2','final_g3_dt2'):
    dest=HERE/'saved_states'/name;dest.mkdir(parents=True,exist_ok=False)
    for file in ('meta.json','state.npz'):
        source=OUT/name/'restart'/file;target=dest/file;shutil.copy2(source,target)
        assert target.stat().st_nlink==1 and target.stat().st_ino!=source.stat().st_ino
        saved.append({'path':str(target.relative_to(ROOT)),'bytes':target.stat().st_size,
            'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
            'source_commit':rows[0]['source_commit'] if not name.startswith('initial_') else '3bc3b1f14c1235ded4326a691cceda9705af0fbd'})
with (HERE/'saved_states_manifest.json').open('x') as f:json.dump(saved,f,indent=2)

fig=plt.figure(figsize=(12,8.8),layout='constrained');gs=fig.add_gridspec(2,2)
for col,name in enumerate(('final_g2_dt2','final_g3_dt2')):
    sub=2+col;mesh=build_icosphere(sub);cp=cps[name]
    ax=fig.add_subplot(gs[0,col],projection='mollweide')
    xe,ye,data=rasterize_cells(mesh,cp.state.continental_fraction,width=360,height=180)
    im=ax.pcolormesh(xe,ye,data,cmap='YlGnBu_r',vmin=0,vmax=1,shading='auto',rasterized=True)
    ax.grid(alpha=.2);ax.tick_params(labelsize=7)
    ax.set_title(f'100 Myr, {mesh.cell_count} cells, dt = 2 Myr',fontsize=11)
    fig.colorbar(im,ax=ax,orientation='horizontal',fraction=.06,pad=.06,label='Continental material fraction (raster map)')
ax=fig.add_subplot(gs[1,0]);colors=['#39789a','#cc8534','#768a4c']
for color,nameset,label in zip(colors[:2],[names[:3],names[5:]],['Physical integrals','Old source/raster exchange']):
    rs=[rows[names.index(n)] for n in nameset]
    ax.plot([r['dt_myr'] for r in rs],[r['continental_produced_million_km3'] for r in rs],'o-',label=label,color=color)
ax.set_xlabel('Outer timestep, Myr');ax.set_ylabel('Produced continental crust, million km³')
ax.set_title('Matched 320-cell initial geometry; different source laws',fontsize=10)
ax.grid(alpha=.2);ax.legend(frameon=False,fontsize=8)
ax=fig.add_subplot(gs[1,1]);rs=[rows[3],rows[1],rows[4]]
ax.bar(range(3),[r['arc_thickening_million_km3'] for r in rs],label='Thickening existing material',color=colors[0])
ax.bar(range(3),[r['continental_produced_million_km3']-r['arc_thickening_million_km3'] for r in rs],
       bottom=[r['arc_thickening_million_km3'] for r in rs],label='Cell-average maturation events',color=colors[1])
ax.set_xticks(range(3),['80','320','1280']);ax.set_xlabel('Cells, identical initial regions; dt = 2 Myr')
ax.set_ylabel('Produced continental crust, million km³');ax.set_title('Birth threshold remains mesh dependent',fontsize=10)
ax.legend(frameon=False,fontsize=8);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
fig.suptitle('EXPERIMENTAL reconstruction: finite material / sensible heat inventories\nPre-existing regions; no paid fracture, no validated lunar history',fontsize=13)
fig.savefig(HERE/'comparison.png',dpi=135);plt.close(fig)
print(json.dumps({'cases':len(rows),'saved_state_bytes':sum(x['bytes'] for x in saved),'figure':'comparison.png'}),flush=True)
