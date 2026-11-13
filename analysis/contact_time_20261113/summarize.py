"""Independent finite-ledger accounting and compact comparison plots."""
import csv,importlib.util,json,math,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
HERE=Path(__file__).resolve().parent;OUT=ROOT/'outputs_v131_contact_time'
s=importlib.util.spec_from_file_location('accounting',ROOT/'analysis/regional_area_20261113/summarize.py')
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)


def extract():
    rows=[]
    for p in sorted(HERE.glob('*_result.json')):
        r=json.loads(p.read_text());name=r['case'];target=OUT/name/'restart';meta=json.loads((target/'meta.json').read_text())
        history=meta['regional_material']['history'];parts=[part for step in history for part in step.get('source_intervals',[step])]
        first=next((part.get('start_myr',part['time_myr']-r['dt_myr']) if 'time_myr' in part else part['start_myr'] for part in parts if part['source_integral_km2']>0),None)
        data=meta['material_exchange'];heat,origin=m.residuals(data)
        with np.load(target/'state.npz') as z:
            a={k:z['regional_material_leaf_'+k] for k in ('area','felsic_area','felsic_volume','basalt_volume','potential')}
        with np.load(OUT/f"initial_same_g{r['subdivisions']}/restart/state.npz") as z:
            initial=[math.fsum(z['continental_volume_km3']),math.fsum(z['oceanic_volume_km3'])]
        ledger=[math.fsum([math.fsum(a[label+'_volume']),-start,*(-h['born_'+label+'_km3'] for h in history),*(h['returned_'+label+'_km3'] for h in history)]) for label,start in zip(('felsic','basalt'),initial)]
        rows.append(dict(case=name,end_myr=r['end_myr'],dt_myr=r['dt_myr'],receiver_cells=20*4**r['subdivisions'],
            felsic_created_km3=r['continental_generated_km3'],basalt_created_km3=r['ocean_created_km3'],basalt_returned_km3=r['ocean_recycled_km3'],
            matured_area_km2=r['juvenile_area_km2'],material_owners=r['plate_count'],leaves=len(a['area']),source_intervals=len(parts),first_source_interval_start_myr=first,
            max_potential=float(max(a['potential'])),max_felsic_thickness_km=float(max(np.divide(a['felsic_volume'],a['felsic_area'],out=np.zeros_like(a['area']),where=a['felsic_area']>0))),
            mantle_temperature_k=r['temperature_k'],crust_temperature_k=r['material_budget']['crust_temperature_k'],
            felsic_ledger_residual_km3=ledger[0],basalt_ledger_residual_km3=ledger[1],
            sensible_heat_residual_80digit_j=heat,origin_residual_80digit_kg=origin,
            pool_mass_relative_error=r['material_budget']['mass_residual_kg']/data['total_mass_kg'],
            max_source_relative_residual=max(abs(x['source_deposition_residual_km2'])/max(x['source_integral_km2'],1.) for x in parts),
            wall_s=r['resources']['wall_s'],loop_s=r['runtime']['integration_wall_seconds'],max_rss_mib=r['resources']['max_rss_kib']/1024,
            native_state_bytes=sum((target/n).stat().st_size for n in ('state.npz','meta.json'))))
    with (HERE/'comparison.json').open('x') as f:json.dump(rows,f,indent=2)
    with (HERE/'comparison.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    return rows


def figures(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from tectonics.mesh import build_icosphere
    s=importlib.util.spec_from_file_location('contacts',HERE/'contact_diagnosis.py');c=importlib.util.module_from_spec(s);s.loader.exec_module(c)
    old=json.loads((ROOT/'analysis/regional_area_20261113/comparison.json').read_text());old={r['case']:r for r in old};new={r['case']:r for r in rows}
    fig,axs=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for ax,key,keys_old,keys_new,label in [(axs[0],'dt_myr',['area20_dt1','area20_dt2','area20'],['gate20_dt1','gate20_dt2','gate20'],'Outer time step, Myr'),
                                         (axs[1],'receiver_cells',['area20_g1','area20','area20_g3'],['gate20_g1','gate20','gate20_g3'],'Receiver cells')]:
        for label_series,source,names in [('Endpoint source',old,keys_old),('Localized gates',new,keys_new)]:
            sel=[source[n] for n in names if n in source];ax.plot([r[key] for r in sel],[r['felsic_created_km3']/1e6 for r in sel],'o-',label=label_series)
        ax.set(xlabel=label,ylabel='20 Myr felsic thickening, million km³');ax.grid(alpha=.25);ax.legend()
        if key=='receiver_cells' and 'gate20_g3' not in new:
            ax.text(.98,.08,'Localized 1280-cell attempt has no result;\nnot repeated or treated as a completed comparison',ha='right',transform=ax.transAxes,fontsize=8)
    fig.suptitle('Timing defect corrected; legacy contact-map dependence remains\nThese are numerical proxy controls, not a physically closed tectonic history')
    fig.savefig(HERE/'source_convergence.png',dpi=145);plt.close(fig)
    cp,inv=c.load('gate20');diag,mesh,count=c.coverage(inv,4)
    fig=plt.figure(figsize=(11,5),layout='constrained');ax=fig.add_subplot(111,projection='mollweide')
    colors=np.where(count==0,0,np.where(count==1,1,2))
    from matplotlib.colors import ListedColormap,BoundaryNorm
    cmap=ListedColormap(['#d86b48','#d8e5df','#526eb5']);norm=BoundaryNorm([-.5,.5,1.5,2.5],3)
    p=mesh.centroids;sc=ax.scatter(np.arctan2(p[:,1],p[:,0]),np.arcsin(p[:,2]),c=colors,s=7,cmap=cmap,norm=norm)
    ax.grid(alpha=.25);bar=fig.colorbar(sc,ax=ax,orientation='horizontal',shrink=.65,ticks=[0,1,2]);bar.ax.set_xticklabels(['Uncovered','Covered once','Overlap'])
    fig.suptitle('Material geometry at 20 Myr: approximately 10.7% gaps and 10.6% overlap\n5120-point diagnostic; finite inventory is conserved, but a closed physical surface is missing')
    fig.savefig(HERE/'material_coverage20.png',dpi=145);plt.close(fig)


if __name__=='__main__':
    rows=extract();figures(rows);print(json.dumps(rows,indent=2))
