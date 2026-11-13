"""Independent accounting from native states; no run or old artifact mutation."""
import csv,json,math,sys
from decimal import Decimal,localcontext
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
HERE=Path(__file__).resolve().parent;OUT=ROOT/'outputs_v131_regional_area'


def residuals(data):
    with localcontext() as ctx:
        ctx.prec=80;dec=lambda x:Decimal.from_float(float(x))
        terms=[*data['energy_m_j'],*data['energy_c_j'],-data['initial_energy_j'],
               *(-x for x in data['radiogenic_input_j']),*(-x for x in data['orbital_input_j']),*data['surface_output_j']]
        energy=sum(map(dec,terms),Decimal(0))
        origin=[sum((dec(data[key][i][j]) for key in ('origin_hi','origin_lo') for i in range(3)),Decimal(0))
                -dec(data['origin_mass_kg'][j]) for j in range(3)]
        return float(energy),float(max(map(abs,origin)))


def extract():
    rows=[]
    for path in sorted(HERE.glob('area*_result.json')):
        result=json.loads(path.read_text());name=result['case'];target=OUT/name/'restart'
        meta=json.loads((target/'meta.json').read_text());history=meta['regional_material']['history'];data=meta['material_exchange']
        with np.load(target/'state.npz') as z:
            a={key:z['regional_material_leaf_'+key] for key in ('area','felsic_area','felsic_volume','basalt_volume','potential')}
        with np.load(OUT/f"initial_same_g{result['subdivisions']}/restart/state.npz") as z:
            initial=[math.fsum(z['continental_volume_km3']),math.fsum(z['oceanic_volume_km3'])]
        totals=[math.fsum(a['felsic_volume']),math.fsum(a['basalt_volume'])]
        ledger=[]
        for label,start,total in zip(('felsic','basalt'),initial,totals):
            ledger.append(math.fsum([total,-start,*(-x['born_'+label+'_km3'] for x in history),
                                     *(x['returned_'+label+'_km3'] for x in history)]))
        heat,origin=residuals(data)
        row=dict(case=name,source_commit=result['source_commit'],end_myr=result['end_myr'],dt_myr=result['dt_myr'],
            receiver_cells=20*4**result['subdivisions'],material_parents=1280,
            max_edge_km=meta['regional_material']['area_parameters']['max_edge_km'],leaves=len(a['area']),
            felsic_created_km3=result['continental_generated_km3'],basalt_created_km3=result['ocean_created_km3'],
            basalt_returned_km3=result['ocean_recycled_km3'],matured_area_km2=result['juvenile_area_km2'],
            material_max_felsic_thickness_km=float(max(np.divide(a['felsic_volume'],a['felsic_area'],out=np.zeros_like(a['area']),where=a['felsic_area']>0))),
            receiver_max_thickness_km=result['thickness_max_km'],max_maturation=float(max(a['potential'])),
            plate_count=result['plate_count'],mantle_temperature_k=result['temperature_k'],crust_temperature_k=result['material_budget']['crust_temperature_k'],
            wall_s=result['resources']['wall_s'],loop_wall_s=result['runtime']['integration_wall_seconds'],
            max_rss_mib=result['resources']['max_rss_kib']/1024,
            max_source_residual_km2=max(abs(x['source_deposition_residual_km2']) for x in history),
            max_source_relative_residual=max(abs(x['source_deposition_residual_km2'])/max(x['source_integral_km2'],1.) for x in history),
            max_source_order_difference_relative=max(x['coarea_order_difference_estimate_km2']/max(x['source_integral_km2'],1.) for x in history),
            max_support_area_relative_residual=max(abs(x['support_area_residual_km2'])/max(x['source_supported_area_sum_km2'],1.) for x in history),
            max_refinement_geometry_relative_error=max(x['max_subdivision_geometry_relative_error'] for x in history),
            felsic_volume_residual_km3=ledger[0],basalt_volume_residual_km3=ledger[1],
            sensible_heat_residual_80digit_j=heat,origin_residual_80digit_kg=origin,
            rounded_pool_mass_relative_error=result['material_budget']['mass_residual_kg']/data['total_mass_kg'],
            final_projection_rms_km=history[-1]['rms_projection_distance_km'],
            max_projection_distance_km=max(x['max_projection_distance_km'] for x in history),
            receiver_area_relative_error=max(x['max_receiver_area_relative_error'] for x in history),
            native_state_bytes=sum((target/n).stat().st_size for n in ('state.npz','meta.json')))
        rows.append(row)
    with (HERE/'comparison.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    with (HERE/'comparison.json').open('x') as f:json.dump(rows,f,indent=2)
    return rows


def figures(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from tectonics.mesh import build_icosphere
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    by={r['case']:r for r in rows};old=json.loads((HERE.parent/'regional_inventory_20261113/comparison.json').read_text())
    old={r['case']:r for r in old}
    fig,axs=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    groups=[('max_edge_km',['area20_edge250','area20','area20_edge62'],'Maximum active material edge, km'),
            ('dt_myr',['area20','area20_dt2','area20_dt1'],'Outer step, Myr'),
            ('receiver_cells',['area20_g1','area20','area20_g3'],'Receiver map cells')]
    for ax,(key,names,label) in zip(axs.flat,groups):
        selected=sorted([by[n] for n in names if n in by],key=lambda r:r[key])
        ax.plot([r[key] for r in selected],[r['felsic_created_km3']/1e6 for r in selected],'o-',label='Actual material areas')
        if key=='receiver_cells':
            prev=[old['short_m3_g'+str(n)+'_dt4'] for n in (1,2,3)]
            ax.plot([r[key] for r in prev],[r['felsic_created_km3']/1e6 for r in prev],'s--',label='Previous point inventory');ax.legend()
        ax.set(xlabel=label,ylabel='Felsic thickening in 20 Myr, million km³');ax.grid(alpha=.2)
        if key=='max_edge_km':
            ax.set_ylim(1.60,1.68)
            ax.text(.03,.92,'Integral change < 3 × 10⁻¹⁵ relative',transform=ax.transAxes)
    probe=json.loads((HERE/'coarea_probe.json').read_text());ax=axs[1,1]
    for orientation in ('aligned','oblique'):
        selected=[r for r in probe['rows'] if r['orientation']==orientation and r['order']==16]
        ax.plot([r['max_edge_km'] for r in selected],[100*r['area_relative_error'] for r in selected],'o-',label=orientation)
    ax.set(xscale='log',xlabel='Active material edge, km',ylabel='Matured area error vs continuum, %');ax.axhline(0,color='k',lw=.7);ax.legend();ax.grid(alpha=.2)
    fig.suptitle('Reconstructed experiment: source integration improves; coupled convergence remains limited\nFrozen-strip threshold at right is a separate operator test, not a physical fracture',fontsize=12)
    fig.savefig(HERE/'convergence.png',dpi=145);plt.close(fig)
    target=OUT/'area20/restart';meta=json.loads((target/'meta.json').read_text())
    with np.load(target/'state.npz') as z:
        points=z['regional_material_leaf_points'];values=(z['regional_material_leaf_felsic_volume']+z['regional_material_leaf_basalt_volume'])/z['regional_material_leaf_area'];mapped=z['crust_thickness_km']
    fig=plt.figure(figsize=(12,8),layout='constrained');grid=fig.add_gridspec(2,2)
    for col,p,v,title,size in [(0,points,values,'Persistent material-area samples',2),
                              (1,build_icosphere(2).centroids,mapped,'Conservative receiver map (320 cells)',25)]:
        ax=fig.add_subplot(grid[0,col],projection='mollweide')
        sc=ax.scatter(np.arctan2(p[:,1],p[:,0]),np.arcsin(np.clip(p[:,2],-1,1)),c=v,s=size,cmap='viridis',vmin=0,vmax=50,rasterized=True)
        ax.grid(alpha=.3);ax.set_title(title);fig.colorbar(sc,ax=ax,orientation='horizontal',shrink=.8,label='Crust thickness, km; map uses dominant material')
    hist=meta['regional_material']['history'];t=[x['time_myr'] for x in hist]
    ax=fig.add_subplot(grid[1,0])
    for key,label in [('born_basalt_km3','Basalt created'),('returned_basalt_km3','Basalt returned')]:
        ax.plot(t,np.cumsum([x[key] for x in hist])/1e6,'o-',label=label)
    ax.set(xlabel='Time, Myr',ylabel='Cumulative volume, million km³');ax.legend();ax.grid(alpha=.2)
    ax=fig.add_subplot(grid[1,1]);ax.plot(t,[x['leaves_after'] for x in hist],'o-')
    ax.set(xlabel='Time, Myr',ylabel='Persistent material elements (not plates)');ax.grid(alpha=.2)
    fig.suptitle('20 Myr experimental evolution; 12 prescribed regions remain 12\nNo juvenile maturation or paid fracture; regional overlaps and nonlocal map are not a contact solution',fontsize=12)
    fig.savefig(HERE/'area20.png',dpi=145);plt.close(fig)


if __name__=='__main__':
    rows=extract();figures(rows)
    print(json.dumps(rows,indent=2))
