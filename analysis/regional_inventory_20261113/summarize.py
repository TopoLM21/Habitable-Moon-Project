"""Compact, independent accounting and figures from completed native states."""
import csv,json,math,sys
from decimal import Decimal,localcontext
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
HERE=Path(__file__).resolve().parent;OUT=ROOT/'outputs_v131_regional'
from tectonics.mesh import build_icosphere


def decimal_residual(data):
    with localcontext() as ctx:
        ctx.prec=80
        dec=Decimal.from_float
        terms=[*data['energy_m_j'],*data['energy_c_j'],-data['initial_energy_j'],
               *(-x for x in data['radiogenic_input_j']),*(-x for x in data['orbital_input_j']),*data['surface_output_j']]
        energy=sum((dec(float(x)) for x in terms),Decimal(0))
        origin=[]
        for j in range(3):
            origin.append(float(sum((dec(float(data[key][i][j])) for key in ('origin_hi','origin_lo') for i in range(3)),Decimal(0))-dec(data['origin_mass_kg'][j])))
        return float(energy),origin


def extract():
    rows=[]
    for path in sorted(HERE.glob('*_result.json')):
        result=json.loads(path.read_text());name=result['case']
        if name=='m3_restart100':continue # known pre-fix restart, retained separately
        target=OUT/name/'restart';meta=json.loads((target/'meta.json').read_text())
        if 'regional_material' not in meta:continue
        with np.load(target/'state.npz') as z:
            a={k:z['regional_material_'+k] for k in meta['regional_material']['array_names']}
        hist=meta['regional_material']['history'];data=meta['material_exchange']
        residual,origin=decimal_residual(data)
        rows.append(dict(case=name,source_commit=result['source_commit'],end_myr=result['end_myr'],dt_myr=result['dt_myr'],
            receiver_cells=20*4**result['subdivisions'],material_columns=len(a['area']),
            felsic_created_km3=result['continental_generated_km3'],basalt_created_km3=result['ocean_created_km3'],
            basalt_returned_km3=result['ocean_recycled_km3'],matured_area_km2=result['juvenile_area_km2'],
            total_felsic_km3=math.fsum(a['felsic_volume']),total_basalt_km3=math.fsum(a['basalt_volume']),
            material_max_felsic_thickness_km=float(max(np.divide(a['felsic_volume'],a['felsic_area'],out=np.zeros_like(a['area']),where=a['felsic_area']>0))),
            receiver_max_thickness_km=result['thickness_max_km'],max_maturation=float(max(a['potential'])),
            mantle_temperature_k=result['temperature_k'],crust_temperature_k=result['material_budget']['crust_temperature_k'],
            plate_count=result['plate_count'],wall_s=result['resources']['wall_s'],max_rss_mib=result['resources']['max_rss_kib']/1024,
            final_projection_rms_km=hist[-1]['rms_projection_distance_km'],
            max_projection_rms_km=max(x['rms_projection_distance_km'] for x in hist),
            max_projection_distance_km=max(x['max_projection_distance_km'] for x in hist),
            receiver_area_relative_error=max(x['max_receiver_area_relative_error'] for x in hist),
            sensible_heat_residual_80digit_j=residual,origin_residual_80digit_kg=max(abs(x) for x in origin),
            rounded_pool_mass_residual_kg=result['material_budget']['mass_residual_kg'],
            relative_rounded_pool_mass_error=result['material_budget']['mass_residual_kg']/data['total_mass_kg'],
            material_ledger_error_km3=result['continental_ledger_error_km3']))
    with (HERE/'comparison.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    (HERE/'comparison.json').write_text(json.dumps(rows,indent=2)+'\n')
    return rows


def figures(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    by={r['case']:r for r in rows}
    fig,axs=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    sets=[['short_m3_g2_dt4','short_m3_g2_dt2','short_m3_g2_dt1'],
          ['short_m2_g2_dt4','short_m3_g2_dt4','short_m4_g2_dt4'],
          ['short_m3_g1_dt4','short_m3_g2_dt4','short_m3_g3_dt4']]
    for ax,names,key,label in zip(axs.flat,sets,['dt_myr','material_columns','receiver_cells'],['Outer step, Myr','Material columns (receiver: 320)','Receiver cells (material: 1280)']):
        selected=[by[n] for n in names if n in by]
        ax.plot([r[key] for r in selected],[r['felsic_created_km3']/1e6 for r in selected],'o-',label='Felsic creation / thickening')
        ax.set(xlabel=label,ylabel='20 Myr production, million km³');ax.grid(alpha=.2)
    op=json.loads((HERE/'operator_verified.json').read_text());ax=axs[1,1]
    for orientation in ('aligned','oblique'):
        selected=[r for r in op['spatial'] if r['orientation']==orientation]
        ax.plot([r['material_columns'] for r in selected],[r['area_km2']/1e6 for r in selected],'o-',label=orientation)
    ax.axhline(op['exact_continuum']['area_km2']/1e6,color='k',ls='--',label='Independent continuum integral')
    ax.set(xscale='log',xlabel='Material quadrature columns',ylabel='Matured area, million km²');ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle('Experimental regional inventory: remaining resolution dependence\nSame physical initial state; frozen-strip test at right is independent of coupled runs',fontsize=12)
    fig.savefig(HERE/'convergence.png',dpi=145);plt.close(fig)
    target=OUT/'m3_g2_dt4/restart';meta=json.loads((target/'meta.json').read_text())
    with np.load(target/'state.npz') as z:
        a={k:z['regional_material_'+k] for k in meta['regional_material']['array_names']}
        mapped=z['crust_thickness_km']
    mesh=build_icosphere(2)
    fig=plt.figure(figsize=(12,8),layout='constrained');gs=fig.add_gridspec(2,2)
    for col,points,values,title,size in [(0,a['points'],(a['felsic_volume']+a['basalt_volume'])/a['area'],'Authoritative moving columns (fixed area weights)',10),
                                        (1,mesh.centroids,mapped,'Conservative coarse receiver map',25)]:
        ax=fig.add_subplot(gs[0,col],projection='mollweide');lon=np.arctan2(points[:,1],points[:,0]);lat=np.arcsin(points[:,2])
        sc=ax.scatter(lon,lat,c=values,s=size,cmap='viridis',vmin=0,vmax=50,rasterized=True);ax.grid(alpha=.3);ax.set_title(title)
        fig.colorbar(sc,ax=ax,orientation='horizontal',shrink=.8,label='Crust thickness, km (map uses dominant material)')
    ax=fig.add_subplot(gs[1,0]);hist=meta['material_exchange']['history']
    ax.plot([x['time_myr'] for x in hist],[x['mantle_temperature_k'] for x in hist],label='Mantle')
    ax.set(xlabel='Time, Myr',ylabel='Mantle temperature, K');ax.grid(alpha=.2)
    ax=fig.add_subplot(gs[1,1]);hist=meta['regional_material']['history'];t=[x['time_myr'] for x in hist]
    for key,label in [('born_basalt_km3','Basalt created'),('returned_basalt_km3','Basalt recycled'),('born_felsic_km3','Felsic thickening')]:
        ax.plot(t,np.cumsum([x[key] for x in hist])/1e9,label=label)
    ax.set(xlabel='Time, Myr',ylabel='Cumulative volume, billion km³');ax.grid(alpha=.2);ax.legend()
    fig.suptitle('100 Myr reconstructed experiment: 12 prescribed regions remain 12\nNo juvenile maturation event; no paid fracture. Projection is not a physical contact solution.',fontsize=12)
    fig.savefig(HERE/'regional100.png',dpi=145);plt.close(fig)


if __name__=='__main__':
    rows=extract();figures(rows)
    print(json.dumps(rows),flush=True)
