"""Sample exact spherical face containment; avoid polygon-display pole gaps."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from verify import HERE, setup
from tectonics.strain_exchange import StrainExchange

lon=np.linspace(-180,180,301);lat=np.linspace(-90,90,151)
x,y=np.meshgrid(np.radians((lon[:-1]+lon[1:])/2),np.radians((lat[:-1]+lat[1:])/2))
points=np.c_[np.cos(y).ravel()*np.cos(x).ravel(),np.cos(y).ravel()*np.sin(x).ravel(),np.sin(y).ravel()]
fig,axes=plt.subplots(2,3,figsize=(12,6),constrained_layout=True);checks=[]
for row,(scenario,pressure) in enumerate([('native',0.),('smooth',5e8)]):
    surfaces=[setup(scenario=scenario,pressure=pressure).coupled.surface,
              StrainExchange.load(HERE/'states'/f'{scenario}_g3_100').coupled.surface]
    for col,s in enumerate(surfaces):
        tri=s.vertices[s.faces];normals=np.stack([np.cross(tri[:,i],tri[:,(i+1)%3]) for i in range(3)],axis=1)
        index=np.full(len(points),-1);count=np.zeros(len(points),int)
        for start in range(0,len(tri),32):
            mask=np.all(np.einsum('fkj,pj->fkp',normals[start:start+32],points)>=-1e-13,axis=1)
            hit=np.any(mask,axis=0);index[hit]=start+np.argmax(mask[:,hit],axis=0);count+=mask.sum(axis=0)
        assert np.all(count==1)
        checks.append(dict(scenario=scenario,time_myr=s.data['time_myr'],points=len(points),uncovered=0,overlap=0))
        h=s.evaluate(s.vertices)['height_m'].max(axis=1)/1000
        color=axes[row,col].pcolormesh(lon,lat,h[index].reshape(x.shape),cmap='viridis',vmin=0,vmax=36,rasterized=True)
        axes[row,col].set(xlabel='Longitude',ylabel='Latitude',title=f'{scenario}: '+('initial' if col==0 else '100 Myr'))
        axes[row,col].set_xticks([-180,0,180]);axes[row,col].set_yticks([-90,0,90])
    d=json.loads((HERE/f'{scenario}_g3_100.json').read_text());t=[v['time_myr'] for v in d['history']]
    ax=axes[row,2]
    for k,label in [('basalt_created_m3','Inserted'),('basalt_returned_m3','Returned')]:
        ax.plot(t,[v[k]/1e15 for v in d['history']],label=label)
    ax.set(xlabel='Time, Myr',ylabel='Cumulative volume, million km³',title=f'Initial pressure {pressure/1e6:g} MPa')
    ax.legend();ax.grid(alpha=.2)
fig.colorbar(color,ax=axes[:,:2],label='Fixed-base column height, km',shrink=.8)
fig.suptitle('Conditional coarse exchange on a welded sphere\nFinite pressure / sensible heat; prescribed regions, no melting or fracture',fontsize=12)
fig.savefig(HERE/'linked100_raster.png',dpi=140)
with (HERE/'map_coverage.json').open('x') as f:json.dump(checks,f,indent=2)
