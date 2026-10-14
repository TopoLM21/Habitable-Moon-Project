"""Collect bounded contact evidence and plot recorded, not reconstructed, histories."""
import json,math,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent


def summarize():
    results=[]
    cases=[('legacy_hypothesis',dt,'/workspace/geological-contact-02') for dt in (1,.5,.25)] + [('area_corrected',dt,'/workspace/geological-contact-area') for dt in (1,.5)]
    for mode,dt,base in cases:
        path=Path(base,f'dt{dt}','result.json')
        r=json.loads(path.read_bytes());steps=r['steps'];rep=[s['contact_repair'] for s in steps if s.get('contact_repair')]
        volume=sum(s['material']['generated_volume_km3'] for s in steps)
        recycled=sum(s['material']['recycled_volume_km3'] for s in steps)
        saved=path.parent/'accepted'
        record=json.loads((saved/'record.json').read_bytes());source=json.loads((saved/'source.json').read_bytes())
        identity={}
        for parts in record['material_parcels']['node_parts'].values():
            for part in parts:identity.setdefault(part['id'],[]).append(part['mass_kg'])
        for retired in record['material_parcels']['recycled_history']:
            for part in retired['parts']:identity.setdefault(part['id'],[]).append(part['mass_kg'])
        identity_error=max(abs(math.fsum(identity.get('source-node-'+str(i),[]))-m)/m for i,m in enumerate(source['config']['mass'][:-2]))
        summary=dict(displacement_mode=mode,original640_identity_relative_error=identity_error,dt=dt,initial=r['initial'],final=r['final'],history=r['history'],balances=r['balances'],
            elapsed_seconds=r['elapsed_seconds'],including_restart_seconds=r['total_seconds_including_restart'],restart_exact=r.get('restart_exact'),
            generated_volume_km3=volume,recycled_volume_km3=recycled,
            max_water_relative_error=max(abs(s['water']['relative_volume_error']) for s in steps),
            max_material_volume_residual_km3=max(abs(s['material']['volume_residual_km3']) for s in steps),
            max_native_transfer_residual_kg=max(abs(v) for s in steps for v in s['native_transfer_residual_kg'].values()),
            promotions=sum(x['promoted'] for x in rep),repairs=len(rep),reassigned_cells=sum(x['reassigned_cells'] for x in rep),
            all_final_domains_connected=all(h['owner_components']==h['plates'] for h in r['history']),
            all_pure_children_inherit_omega=max((x['max_pure_child_omega_error'] for x in rep),default=0),
            max_thin_shell_momentum_relative_error=max((x['thin_shell_momentum_relative_residual'] for x in rep),default=0),
            max_ownership_velocity_change_km_per_myr=max((x['max_ownership_velocity_change_km_per_myr'] for x in rep),default=0),
            raw_footprint_extras=sum(s['geometry']['raw_footprint_extra_components'] for s in steps),
            orphan_newborns=sum(len(s['contact_inventory']['orphan_newborn_cells']) for s in steps),
            source_duplicates=sum(s['contact_inventory']['source_duplicates'] for s in steps),
            winner_duplicates=sum(s['contact_inventory']['winner_duplicates'] for s in steps),
            missing_sources=sum(s['contact_inventory']['missing_sources'] for s in steps),
            inherited_owner_mismatches=sum(s['contact_inventory']['inherited_owner_mismatches'] for s in steps),
            max_raster_footprint_area_discrepancy_km2=max(abs(s['contact_inventory']['retained_raster_area_minus_source_area_km2']) for s in steps),
            detached_components=[dict(time_myr=s['time_myr'],**c) for s in steps for c in s.get('contact_repair',{}).get('detached_components',[])],
            result_path=str(path),result_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        results.append(summary)
    off=json.loads(Path('/workspace/geological-contact-controls/off160/result.json').read_bytes())
    on=json.loads(Path('/workspace/geological-contact-02/dt1/result.json').read_bytes())
    common=min(len(off['history']),len(on['history']))
    controls=dict(without_rule_error=off['error'],without_rule_last=off['final'],
        common_history_samples=common,max_common_land_difference=max(abs(a['land_area_fraction']-b['land_area_fraction']) for a,b in zip(off['history'],on['history'])),
        max_common_sea_difference_m=max(abs(a['sea_level_m']-b['sea_level_m']) for a,b in zip(off['history'],on['history'])))
    legacy=json.loads(Path('/workspace/geological-contact-01/legacy1/result.json').read_bytes())
    controls['guarded_dt1_history_equals_legacy']=legacy['history']==on['history']
    controls['guarded_dt1_balances_equal_legacy']=legacy['balances']==on['balances']
    data=dict(runs=results[:3],corrected_runs=results[3:],controls=controls,scope='one common saved state at160.8684 to200.8684Myr; not a new full500Myr convergence claim')
    (ROOT/'comparison.json').write_text(json.dumps(data,indent=2))
    return data


def plots(data):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(10,7))
    for r,color in zip(data['runs'],['#147d92','#b45114','#75469a']):
        h=r['history'];t=[v['time_myr'] for v in h]
        for ax,key,scale in [(axes[0,0],'land_area_fraction',100),(axes[0,1],'continental_area_fraction',100),(axes[1,0],'mean_speed_km_per_myr',1),(axes[1,1],'plates',1)]:
            ax.plot(t,[scale*v[key] for v in h],color=color,label=f"dt={r['dt']} Myr")
    for ax,title in zip(axes.ravel(),['Exposed land (%)','Continental material area (%)','Area-weighted surface speed (km/Myr)','Connected domains']):
        ax.set_title(title);ax.set_xlabel('Model time (Myr)');ax.grid(alpha=.2)
    axes[0,0].legend();fig.suptitle('Common saved start; contact topology remains step-sensitive');fig.tight_layout();fig.savefig(ROOT/'contact_history.png',dpi=150)
    fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True)
    for r,color in zip(data['corrected_runs'],['#147d92','#b45114']):
        h=r['history'];t=[v['time_myr'] for v in h]
        for ax,key,scale in zip(axes,['land_area_fraction','continental_area_fraction','mean_speed_km_per_myr'],[100,100,1]):
            ax.plot(t,[scale*v[key] for v in h],color=color,label=f"dt={r['dt']} Myr")
    for ax,title in zip(axes,['Exposed land (%)','Continental material (%)','Surface speed (km/Myr)']):ax.set_ylabel(title);ax.grid(alpha=.2)
    axes[0].legend();axes[-1].set_xlabel('Model time (Myr)');fig.suptitle('Area-corrected displacement: same saved start');fig.tight_layout();fig.savefig(ROOT/'corrected_history.png',dpi=150)
    fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True)
    long=[]
    for dt,path,color in [(1,'/workspace/geological-continuation-04/dt1/final','#147d92'),(.5,'/workspace/geological-continuation-06/dt05/final','#b45114')]:
        r=json.loads(Path(path,'record.json').read_bytes());steps=r['steps'];t=[s['time_myr'] for s in steps]
        land=[s['water']['land_area_fraction']*100 for s in steps]
        cont=[100*(s['water']['exposed_continental_material_area_fraction']+s['water']['submerged_continental_material_area_fraction']) for s in steps]
        speed=[s['dynamics']['mean_speed_deg_per_myr'] for s in steps]
        for ax,y in zip(axes,[land,cont,speed]):ax.plot(t,y,color=color,label=f'dt={dt} Myr')
        late=[s for s in steps if s['time_myr']>=450]
        long.append(dict(dt=dt,late450_500_min_mean_angular_speed_deg_myr=min(s['dynamics']['mean_speed_deg_per_myr'] for s in late),
            late450_500_max_mean_angular_speed_deg_myr=max(s['dynamics']['mean_speed_deg_per_myr'] for s in late),
            time_myr=t,land_percent=land,continental_percent=cont,mean_plate_angular_speed_deg_myr=speed))
    for ax,title in zip(axes,['Exposed land (%)','Continental material area (%)','Mean plate omega\n(deg/Myr, before contact)']):ax.set_ylabel(title);ax.grid(alpha=.2)
    axes[0].legend();axes[-1].set_xlabel('Model time (Myr)');fig.suptitle('Preserved 500 Myr histories (legacy displacement rule)');fig.tight_layout();fig.savefig(ROOT/'preserved_500_history.png',dpi=150)
    (ROOT/'preserved_500_history.json').write_text(json.dumps(long,indent=2))

if __name__=='__main__':plots(summarize())
