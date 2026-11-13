"""Compact comparison table and scientific map; no result overwrites."""
import csv
import json
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from verify import HERE, setup
from tectonics.strain_exchange import StrainExchange

rows=[]
for p in sorted(HERE.glob('*.json')):
    d=json.loads(p.read_text())
    if not isinstance(d,dict) or 'final' not in d:continue
    f=d['final']
    rows.append(dict(case=d['case'],cells=d['cells'],dt_myr=d['dt_myr'],time_myr=f['time_myr'],
        created_km3=f['basalt_created_m3']/1e9,returned_km3=f['basalt_returned_m3']/1e9,
        pressure_mpa=f['pressure_pa']/1e6,reservoir_energy_j=f['reservoir_energy_j'],
        pressure_work_j=f['surface_pressure_work_j'],hydraulic_heat_j=f['hydraulic_heat_j'],
        total_mechanical_heat_j=f['heat_generated_j'],combined_residual_j=f['combined_energy_residual_j'],
        residual_over_mechanical_exchange=f['combined_energy_residual_j']/max(f['heat_generated_j'],1.),
        tallest_km=f['max_height_m']/1000,crust_temperature_k=f['crust_temperature_k'],
        mantle_temperature_k=f['mantle_temperature_k'],speed_rms_mm_yr=d['final_strain']['speed_rms_mm_yr'],
        wall_s=d['wall_s'],status=d['status']))
with (HERE/'comparison.csv').open('x',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
with (HERE/'comparison.json').open('x') as f:json.dump(rows,f,indent=2)

def painted(ax,s,values):
    tri=s.vertices[s.faces]
    lon=np.degrees(np.arctan2(tri[:,:,1],tri[:,:,0]));lat=np.degrees(np.arcsin(tri[:,:,2]))
    lon=lon[:,:1]+(lon-lon[:,:1]+180)%360-180
    polys=[];colors=[]
    for shift in (-360,0,360):
        polys.extend(np.stack([lon+shift,lat],axis=-1));colors.extend(values)
    c=PolyCollection(polys,array=np.asarray(colors),cmap='viridis',edgecolors='none',clim=(0,36))
    ax.add_collection(c);ax.set(xlim=(-180,180),ylim=(-90,90),xlabel='Longitude',ylabel='Latitude')
    ax.set_xticks([-180,0,180]);ax.set_yticks([-90,0,90]);return c

fig,axes=plt.subplots(2,3,figsize=(12,6),constrained_layout=True)
for row,(scenario,pressure) in enumerate([('native',0.),('smooth',5e8)]):
    initial=setup(scenario=scenario,pressure=pressure).coupled.surface
    final=StrainExchange.load(HERE/'states'/f'{scenario}_g3_100').coupled.surface
    for col,s in enumerate([initial,final]):
        h=s.evaluate(s.vertices)['height_m'].max(axis=1)/1000
        color=painted(axes[row,col],s,h)
        axes[row,col].set_title(f'{scenario}: '+('initial' if col==0 else '100 Myr'))
    d=json.loads((HERE/f'{scenario}_g3_100.json').read_text())
    t=[x['time_myr'] for x in d['history']]
    axes[row,2].plot(t,[x['basalt_created_m3']/1e15 for x in d['history']],label='Inserted')
    axes[row,2].plot(t,[x['basalt_returned_m3']/1e15 for x in d['history']],label='Returned')
    axes[row,2].set(xlabel='Time, Myr',ylabel='Cumulative volume, million km³',title=f'Initial pressure {pressure/1e6:g} MPa')
    axes[row,2].legend();axes[row,2].grid(alpha=.2)
fig.colorbar(color,ax=axes[:,:2],label='Fixed-base column height, km',shrink=.8)
fig.suptitle('Conditional coarse exchange on a welded sphere\nFinite pressure / sensible heat; prescribed regions, no melting or fracture',fontsize=12)
fig.savefig(HERE/'linked100.png',dpi=140);plt.close(fig)

fig,axes=plt.subplots(1,3,figsize=(11,3.5),constrained_layout=True)
for scenario,key,label in [('native','returned_km3','Native return'),('smooth','created_km3','Smooth insertion')]:
    subset=[next(x for x in rows if x['case']==f'{scenario}_g{s}_20') for s in (2,3,4)]
    v=np.array([x[key] for x in subset]);axes[0].plot([x['cells'] for x in subset],v/v[-1],'-o',label=label)
    vals=[next(x for x in rows if x['case']==n) for n in [f'{scenario}_g3_20',f'{scenario}_g3_dt0.5_20',f'{scenario}_g3_dt0.25_20']]
    v=np.array([x[key] for x in vals]);axes[1].plot([x['dt_myr'] for x in vals],(v/v[-1]-1)*100,'-o',label=label)
    record=json.loads((HERE/f'{scenario}_g3_100.json').read_text())
    axes[2].plot([x['time_myr'] for x in record['history']],[x['pressure_pa']/1e6 for x in record['history']],label=label)
axes[0].set(xlabel='Cells',ylabel='Flow / finest-grid flow',xscale='log',title='20 Myr spatial comparison')
axes[1].set(xlabel='Outer step, Myr',ylabel='Difference from 0.25 Myr, %',title='Temporal sensitivity')
axes[2].set(xlabel='Time, Myr',ylabel='Store pressure, MPa',title='Finite pressure response')
for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.savefig(HERE/'convergence.png',dpi=140);plt.close(fig)
print(json.dumps(dict(rows=len(rows),figures=['linked100.png','convergence.png'])))
