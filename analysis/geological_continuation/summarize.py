"""Collect actual saved histories and common reference observables."""
import json,math,hashlib
from pathlib import Path
import numpy as np
from tectonics.unified_geological_genesis import UnifiedGeologicalGenesis as U
from tectonics.mesh import connected_components
from .run import snap

ROOT=Path(__file__).resolve().parent
PATHS=['/workspace/geological-continuation-04/dt1','/workspace/geological-continuation-06/dt05']

def collect():
    runs=[]
    for root,at160 in zip(PATHS,['/workspace/geological-continuation-03/dt1/at160','/workspace/geological-continuation-03/dt05/at160']):
        r=json.loads(Path(root,'result.json').read_bytes());e=U.load(Path(root,'final'));cp=e.checkpoint;record=e.record
        identity={}
        for row in record['material_parcels']['node_parts'].values():
            for p in row:identity.setdefault(p['id'],[]).append(p['mass_kg'])
        for row in record['material_parcels']['recycled_history']:
            for p in row['parts']:identity.setdefault(p['id'],[]).append(p['mass_kg'])
        errors=[abs(math.fsum(identity.get('source-node-'+str(i),[]))-m)/m for i,m in enumerate(e.source['config']['mass'][:-2])]
        areas=e.mesh.physical_cell_areas_km2(record['radius_km']);components=connected_components(np.flatnonzero(cp.state.continental_fraction>=.5),e.mesh.neighbors)
        r.update(checkpoint=str(Path(root,'final')),at160=snap(U.load(at160)),original_source_sha256=e.source['original_sha256'],
            original_surface_identity_count=len(errors),max_original_identity_mass_relative_error=max(errors),
            promoted_contact_fragments=sum(x['connectivity_promoted_components'] for x in cp.topology_rows),
            reassigned_contact_cells=sum(x['connectivity_reassigned_cells'] for x in cp.topology_rows),
            continental_generated_volume_km3=cp.cycle.cumulative_generated_volume_km3,
            continental_cycle_recycled_volume_km3=cp.cycle.cumulative_recycled_volume_km3,
            first_juvenile_arc_time_myr=next((x['time_myr'] for x in cp.cycle_rows if x['juvenile_arc_area_created_km2']>0),None),
            continental_rift_recycled_volume_km3=sum(x['rift_recycled_volume_km3'] for x in cp.lithosphere_rows),
            displaced_basalt_volume_km3=sum(x.get('oceanic_continental_displacement_volume_km3',0) for x in cp.lithosphere_rows),
            max_native_transfer_residual_kg=max(abs(v) for row in record['steps'] for v in row.get('native_transfer_residual_kg',{}).values()),
            visible_continental_clusters=len(components),largest_visible_continental_cluster_share=float(max((sum(areas[c]*cp.state.continental_fraction[c]) for c in components),default=0)/sum(areas*cp.state.continental_fraction)),
            geological_split_times_myr=[x['time_myr'] for x in record['events'] if x['kind']=='geological_weak_band_split'],
            hydrosphere_history=cp.hydrosphere_rows,topology_history=cp.topology_rows,
            water_phase_assumption=record['water_initialization'],composition_model=record['composition_model'],
            continental_parameters=record['continental_parameters'],subduction_parameters=record['subduction_parameters'],
            thermal_ledger_last_step=record['steps'][-1]['thermal'])
        # Hydrosphere diagnostics are stored in step audits in this adapter.
        r['hydrosphere_history']=[record['initial_water']]+[row['water'] for row in record['steps']]
        runs.append(r)
    reference=json.loads((ROOT/'reference_comparison.json').read_bytes())
    for item,label in zip(reference,['dt1','dt05']):
        m=json.loads(Path('/workspace/geological-reference-01',label,'checkpoints/checkpoint_0500_Myr/meta.json').read_bytes())
        item.update(hydrosphere_history=m['hydrosphere_rows'],topology_history=m['topology_rows'],continental_cycle=m['continental_cycle'])
    comparison={}
    for key in ['mean_speed_km_per_myr','plates','continental_volume_km3','land_area_fraction','sea_level_m','largest_plate_fraction']:
        a,b=(r['final'][key] for r in runs);comparison[key]=dict(coarse=a,fine=b,absolute_difference=abs(a-b),relative_difference=abs(a-b)/max(abs(b),1e-30))
    for key in ['generated_volume_km3','recycled_volume_km3']:
        a,b=(r[key] for r in runs);comparison[key]=dict(coarse=a,fine=b,relative_difference=abs(a-b)/max(abs(b),1e-30))
    result=dict(runs=runs,reference=reference,comparison=comparison,
        caveat='Different young/mature initial states and parameters; within-scenario dt pairs, not an isolated causal effect of one repair. No spatial or statistical convergence certified.')
    (ROOT/'comparison.json').write_text(json.dumps(result,indent=2))
    return result

def plot(data):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    labels=['Young start dt1','Young start dt0.5','Mature b0ba7e5 dt1','Mature b0ba7e5 dt0.5']
    colors=['#147d92','#b45114','#6f7780','#b1a5c0']
    for r,label,color in zip(data['runs']+data['reference'],labels,colors):
        w=r['hydrosphere_history'];t=[x['time_myr'] for x in w]
        axes[0,0].plot(t,[100*x['land_area_fraction'] for x in w],label=label,color=color)
        axes[0,1].plot(t,[x['sea_level_m'] for x in w],color=color)
        h=r['topology_history'];axes[1,1].step([x['time_myr'] for x in h],[x['plate_count_after'] for x in h],color=color)
    coarse=data['runs'][0]['history'];fine=[]
    for path in ['/workspace/geological-continuation-04/dt05/last_accepted','/workspace/geological-continuation-05/dt05/last_accepted']:
        fine.append(snap(U.load(path)))
    fine+=data['runs'][1]['history']+[data['runs'][1]['final']]
    for h,color,label in [(coarse,colors[0],labels[0]),(fine,colors[1],labels[1])]:axes[1,0].plot([x['time_myr'] for x in h],[x['mean_speed_km_per_myr'] for x in h],'.-',color=color,label=label)
    ref=json.loads((ROOT/'reference_fine_observations.json').read_bytes())['history'];axes[1,0].plot([x['time_myr'] for x in ref],[x['mean_speed_km_per_myr'] for x in ref],color=colors[3])
    axes[1,0].scatter([500],[data['reference'][0]['mean_speed_km_per_myr']],color=colors[2],s=35)
    for ax,title in zip(axes.ravel(),['Exposed land (%)','Sea level relative to datum (m)','Area-weighted surface speed (km/Myr; saved samples)','Connected plate domains']):
        ax.set_title(title,fontsize=10);ax.set_xlabel('Model time (Myr)');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8);fig.suptitle('500 Myr: water conserved; topology remains sensitive to dt',fontsize=13);fig.tight_layout();fig.savefig(ROOT/'comparison.png',dpi=160)

if __name__=='__main__':plot(collect())
