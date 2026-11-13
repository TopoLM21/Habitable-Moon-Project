"""Compact tables and inspected scientific figures from saved controls only."""
import csv,json,math
from pathlib import Path
import numpy as np
from verify import HERE,setup
from tectonics.deformation_exchange import DeformationExchange
from tectonics.regional_material import SphereReceiver


def tables():
    rows=[]
    for path in sorted(HERE.glob('*.json')):
        j=json.loads(path.read_text())
        if 'case' not in j or 'final' not in j:continue
        d=j['final'];q=d['heat_generated_j'];w=d['external_work_j']
        rows.append(dict(case=j['case'],end_myr=d['time_myr'],cells=j.get('cells',1280),
            length_km=j.get('length_m',1e6)/1000,step_myr=j.get('dt_myr'),
            heat_j=q,external_work_j=w,residual_j=d.get('combined_energy_residual_j',d['energy_residual_j']),
            residual_on_mechanical_exchange=d.get('combined_energy_residual_j',d['energy_residual_j'])/max(q,w,1.),
            thermal_residual_j=d.get('thermal_energy_residual_j'),
            max_height_km=d['max_height_m']/1000,crust_temperature_k=d.get('crust_temperature_k'),
            mantle_temperature_k=d.get('mantle_temperature_k'),
            wall_s=j['wall_s'],speed_rms_mm_yr=j.get('final_strain',{}).get('speed_rms_mm_yr')))
    with (HERE/'comparison.json').open('x') as f:json.dump(rows,f,indent=2)
    with (HERE/'comparison.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    return rows


def figures(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    by={r['case']:r for r in rows}
    fig,axs=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for label,names in [('Dilation length 0 km',['native_zero_g2_20','native_length0_20','native_zero_g4_20']),
                        ('Dilation length 1000 km',[f'native_g{i}_20' for i in (2,3,4)])]:
        r=[by[n] for n in names]
        axs[0].plot([x['cells'] for x in r],[x['heat_j']/1e25 for x in r],'o-',label=label)
    axs[0].set(xlabel='Material cells',ylabel='20 Myr dissipation, 10²⁵ J');axs[0].legend(fontsize=8)
    times=[by[n] for n in ('native_g3_20','native_g3_dt0.5_20','native_g3_dt0.25_20')]
    axs[1].loglog([r['step_myr'] for r in times],[r['residual_on_mechanical_exchange'] for r in times],'o-')
    axs[1].set(xlabel='Maximum outer step, Myr',ylabel='Combined residual / mechanical heat')
    names=['native_length0_20','native_length500000_20','native_g3_20','native_length2e+06_20']
    r=[by[n] for n in names]
    axs[2].plot([x['length_km'] for x in r],[x['speed_rms_mm_yr'] for x in r],'o-')
    axs[2].set(xlabel='Physical dilatational length, km',ylabel='20 Myr RMS speed, mm/yr')
    for ax in axs:ax.grid(alpha=.25)
    fig.suptitle('Same initial material; changing length changes the constitutive model, not just accuracy\nSharp-profile spatial convergence remains incomplete; no coefficients fitted to events',fontsize=11)
    fig.savefig(HERE/'convergence.png',dpi=140);plt.close(fig)
    final=DeformationExchange.load(HERE/'states/native_g3_100');initial=setup('native',3,1e6,1.)
    lon=np.linspace(-math.pi,math.pi,181);lat=np.linspace(-math.pi/2,math.pi/2,91)
    ll,pp=np.meshgrid((lon[:-1]+lon[1:])/2,(lat[:-1]+lat[1:])/2)
    points=np.c_[np.cos(pp).ravel()*np.cos(ll).ravel(),np.cos(pp).ravel()*np.sin(ll).ravel(),np.sin(pp).ravel()]
    fig=plt.figure(figsize=(12,8),layout='constrained');grid=fig.add_gridspec(2,2)
    for i,model in enumerate((initial,final)):
        obj=model.surface;ev=obj.evaluate(obj.vertices)
        h=(ev['height_m'][:,0]*obj.arrays['felsic_fraction']+ev['height_m'][:,1]*(1-obj.arrays['felsic_fraction']))/1000
        cell=SphereReceiver(obj.mesh()).locate(points)
        ax=fig.add_subplot(grid[0,i],projection='mollweide')
        im=ax.pcolormesh(lon,lat,h[cell].reshape(ll.shape),vmin=0,vmax=40,cmap='terrain',shading='flat')
        lines=[]
        for fa,fb,u,v in obj.edges:
            if obj.arrays['owner'][fa]==obj.arrays['owner'][fb]:continue
            t=np.linspace(0,1,5);x=(1-t[:,None])*obj.vertices[u]+t[:,None]*obj.vertices[v];x/=np.linalg.norm(x,axis=1)[:,None]
            xy=np.c_[np.arctan2(x[:,1],x[:,0]),np.arcsin(x[:,2])]
            lines.extend(np.array([a,b]) for a,b in zip(xy[:-1],xy[1:]) if abs(a[0]-b[0])<math.pi)
        ax.add_collection(LineCollection(lines,colors='k',linewidths=.35));ax.grid(alpha=.2)
        ax.set_title('Initial material partition' if i==0 else '100 Myr: force-driven deformation')
        fig.colorbar(im,ax=ax,orientation='horizontal',shrink=.8,label='Area-mean column height, km')
    history=final.data['history'];t=[0.,*(h['time_myr'] for h in history)]
    ax=fig.add_subplot(grid[1,0]);ax.plot(t,[initial.diagnostics()['crust_temperature_k'],*(h['crust_temperature_k'] for h in history)])
    ax.set(xlabel='Time, Myr',ylabel='Finite crust reservoir temperature, K');ax.grid(alpha=.25)
    hist=final.surface.data['history'];t=[0.,*(h['time_myr'] for h in hist)]
    U=[final.surface.data['initial_potential_j'],*(h['potential_j'] for h in hist)]
    Q=np.r_[0.,np.cumsum([h['heat_j'] for h in hist])]
    ax=fig.add_subplot(grid[1,1]);ax.plot(t,(U[0]-np.array(U))/1e26,label='Potential released');ax.plot(t,Q/1e26,'--',label='Heat deposited once')
    ax.set(xlabel='Time, Myr',ylabel='Mechanical exchange, 10²⁶ J');ax.grid(alpha=.25);ax.legend()
    fig.suptitle('Coarse welded dilation + existing finite sensible-heat reservoirs\n12 prescribed regions remain 12; no slip, thermal stress, automatic production or fracture',fontsize=12)
    fig.savefig(HERE/'coupled100.png',dpi=140);plt.close(fig)


if __name__=='__main__':
    rows=tables();figures(rows)
    print(json.dumps(dict(cases=len(rows),reported_case_wall_sum_s=sum(r['wall_s'] for r in rows)),indent=2))
