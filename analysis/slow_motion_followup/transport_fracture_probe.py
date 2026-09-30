"""Read-only transport/fracture diagnostics; frozen predictions are labelled."""
from pathlib import Path
import sys, json, math
import numpy as np
from scipy.spatial import cKDTree
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tectonics.genesis_starter_continuation import _load_cp, load_starter_source
from tectonics.genesis_starter_fracture import YoungShellFracture
from tectonics.simulation import load_config
from tectonics.transport import (_median_cell_spacing_rad, rotate_by_quaternion,
    quaternion_angle_deg, quaternion_multiply, quaternion_from_axis_angle,
    quaternion_conjugate, build_transport_map, SubgridTransportParameters)
from tectonics.kinematics import classify_boundaries
from tectonics.plates import Plate, PlateSystem
from tectonics.plate_velocity_diagnostics import weighted_stats
from tectonics.mantle import plate_rigid_mantle_fit

def load(path):
    cfg = load_config(path/'mature_config.yaml')
    model, source, _ = load_starter_source(path/'young_context/starter_checkpoint.npz')
    cp = _load_cp(path/'mature_checkpoint', cfg)
    fracture = YoungShellFracture.load(model, path/'young_context/fracture_memory.npz')
    return cfg, model, source, cp, fracture

def probe(path):
    cfg, model, source, cp, fracture = load(path)
    mesh, radius, areas = model.mesh, model.thermal.radius_km, model.areas
    spacing = _median_cell_spacing_rad(mesh)*radius
    tree=cKDTree(mesh.centroids)
    rows=[]
    def residual_metrics(pid, q):
        mask=cp.state.cell_plate==pid
        pos=mesh.centroids[mask]; rot=rotate_by_quaternion(pos,q)
        dist=radius*np.arccos(np.clip(np.einsum('ij,ij->i',pos,rot),-1,1))
        _, near=tree.query(rot)
        changed=float(np.mean(near!=np.flatnonzero(mask)))
        return dict(displacement_km=weighted_stats(dist,areas[mask]),
            p75_displacement_km=float(np.quantile(dist,.75)),changed_fraction=changed,
            residual_rotation_deg=quaternion_angle_deg(q))
    for pid,plate in enumerate(cp.system.plates):
        q=cp.transport_state.residual_quaternions[pid]
        row=dict(plate=pid,hold_myr=float(cp.transport_state.hold_age_myr[pid]),**residual_metrics(pid,q))
        # Keep present Euler motion, mesh, and owner fixed; this is NOT a forward model run.
        def meets(t):
            qt=quaternion_multiply(quaternion_from_axis_angle(plate.euler_axis,plate.angular_speed_rad_per_myr*t),q)
            z=residual_metrics(pid,qt)
            return ((z['changed_fraction']>=.18 and z['p75_displacement_km']>=.30*spacing)
                or (row['hold_myr']+t>=120 and z['changed_fraction']>=.04)), z
        upper=1.
        while upper<100000 and not meets(upper)[0]: upper*=2
        lo=0.
        if upper<100000:
            for _ in range(28):
                mid=(lo+upper)/2
                if meets(mid)[0]: upper=mid
                else: lo=mid
            row['frozen_extra_myr_to_commit']=upper
            row['frozen_commit_metrics']=meets(upper)[1]
        rows.append(row)
    sample=model.loading.sample(source.thermal_context)
    h=sample.lid_thickness_km
    available=fracture.memory.eligible & ~fracture.memory.consumed_band
    fit=plate_rigid_mantle_fit(mesh,cp.state.cell_plate,len(cp.system.plates),radius,cp.mantle_flow)
    local=np.cross(cp.mantle_flow.cell_omega_rad_per_myr,mesh.centroids)*radius
    best=np.cross(fit.omega_rad_per_myr[cp.state.cell_plate],mesh.centroids)*radius
    actual=np.cross(np.array([p.euler_axis*p.angular_speed_rad_per_myr for p in cp.system.plates])[cp.state.cell_plate],mesh.centroids)*radius
    norm=lambda a: float(np.sqrt(np.sum(areas[:,None]*a*a)/areas.sum()))
    result=dict(path=str(path.relative_to(ROOT)),age=cp.state.time_myr,cell_count=mesh.cell_count,
        spacing_km=spacing,normal_p75_trigger_km=.3*spacing,
        transport_parameters=cfg['subgrid_transport'],commits=cp.transport_state.cumulative_commit_count,
        plates=rows,fracture=fracture.diagnose(),fracture_events=fracture.events,
        thickness_km=h,mantle_stress_scale_pa=model.shell.convective_traction_pa*model.parameters.mantle_stress_length_km/h*(-math.expm1(-h/model.shell.traction_coupling_depth_km)),
        strength_pa=weighted_stats(fracture.memory.strength_pa,areas),yield_ratio=weighted_stats(fracture.memory.yield_ratio,areas),
        local_rms_km_myr=norm(local),best_rigid_rms_km_myr=norm(best),actual_rms_km_myr=norm(actual),
        best_residual_rms_km_myr=norm(local-best),actual_residual_rms_km_myr=norm(local-actual),
        available_rupture_cells=int(available.sum()),available_rupture_area_fraction=float(areas@available/areas.sum()))
    return result,(cfg,model,source,cp,fracture)

def integrate_normals(folder,loaded):
    cfg,model,source,cp,fracture=loaded
    mesh,radius=model.mesh,model.thermal.radius_km
    for age in (50,100,200):
        prior=load(folder/f'elapsed_{age:04d}')[3]
        if not np.array_equal(prior.state.cell_plate,cp.state.cell_plate):
            raise ValueError('Boundary integral requires unchanged saved plate ownership')
    template=classify_boundaries(mesh,cp.system,radius,**cfg['classification'])
    opening=np.zeros(len(template)); closure=np.zeros(len(template)); steps=0; first=None; last=None
    # The final four-plate owner remains unchanged, as verified against all later saved states.
    for file in sorted(folder.glob('*.dynamics.jsonl')):
        for line in file.open(encoding='utf8'):
            row=json.loads(line)
            if row['plate_count']!=len(cp.system.plates):continue
            w=np.asarray(row['trace']['final_omega']); ns=np.linalg.norm(w,axis=1)
            plates=tuple(Plate(pid,p.seed_cell,w[pid]/ns[pid],ns[pid]) for pid,p in enumerate(cp.system.plates))
            boundaries=classify_boundaries(mesh,PlateSystem(cp.system.cell_plate,plates),radius,**cfg['classification'])
            n=np.array([b.normal_rate_km_per_myr for b in boundaries]);dt=row['dt_myr']
            opening+=np.maximum(n,0)*dt;closure+=np.maximum(-n,0)*dt
            first=row['state_time_myr'] if first is None else min(first,row['state_time_myr'])
            last=row['state_time_myr'] if last is None else max(last,row['state_time_myr'])
            steps+=1
    lengths=np.array([radius*np.arccos(np.clip(mesh.vertices[b.vertex_u]@mesh.vertices[b.vertex_v],-1,1)) for b in template])
    return dict(steps=steps,first_input_age=first,last_input_age=last,
        note='Integral of actual returned Euler normal velocities on unadvected boundary grid after four plates exist; geometric trial displacement, not actual consumed material or slab.',
        opening_km=weighted_stats(opening,lengths),closure_km=weighted_stats(closure,lengths),
        opening_area_proxy_km2=float(opening@lengths),closure_area_proxy_km2=float(closure@lengths),
        hypothetical_slab_max_length_km=float(.9*closure.max()),hypothetical_slab_max_activation_fraction=float(.9*closure.max()/1800))

def first_commit_replay(root,experiment):
    cfg,model,source,cp,fracture=load(experiment/'elapsed_0400')
    rows=[]
    for line in (root/'transport_405.dynamics.jsonl').open(encoding='utf8'):
        row=json.loads(line);w=np.asarray(row['trace']['final_omega']);ns=np.linalg.norm(w,axis=1)
        system=PlateSystem(cp.state.cell_plate,tuple(Plate(pid,p.seed_cell,w[pid]/ns[pid],ns[pid]) for pid,p in enumerate(cp.system.plates)))
        total=[quaternion_multiply(quaternion_from_axis_angle(p.euler_axis,p.angular_speed_rad_per_myr),q)
            for p,q in zip(system.plates,cp.transport_state.residual_quaternions)]
        mapped=build_transport_map(model.mesh,system,cp.state,1.,cp.transport_state,SubgridTransportParameters(**cfg['subgrid_transport']))
        stats=[]
        for pid,targets in enumerate(mapped.source_to_target):
            sources=np.flatnonzero(cp.state.cell_plate==pid)
            qfit=quaternion_multiply(quaternion_conjugate(mapped.state.residual_quaternions[pid]),total[pid])
            stats.append(dict(plate=pid,cells_moved=int(np.sum(targets!=sources)),plate_cells=len(sources),
                represented_rotation_deg=quaternion_angle_deg(qfit),total_trial_rotation_deg=quaternion_angle_deg(total[pid]),
                residual_rotation_deg=quaternion_angle_deg(mapped.state.residual_quaternions[pid])))
        rows.append(dict(age_end=row['state_time_myr']+1,commits=mapped.diagnostics.committed_plates,plates=stats))
        if mapped.diagnostics.committed_plates:break
    return rows

def post_commit_mechanics(root):
    cfg,model,source,cp,fracture=load(root/'transport_405')
    newborn=cp.state.crust_age_myr<10
    keys=['time_myr','gap_fraction','overlap_fraction','created_oceanic_area_km2','subducted_oceanic_area_km2',
        'conservative_transport_commits','transport_cumulative_commit_count','oceanic_created_volume_km3','oceanic_subducted_volume_km3']
    fields={name:{'newborn_values':np.unique(getattr(cp.state,name)[newborn]).tolist(),
        'old_values':np.unique(getattr(cp.state,name)[~newborn]).tolist()}
        for name in ['crust_age_myr','crust_thickness_km','mantle_lithosphere_thickness_km','mantle_lithosphere_density_anomaly_kg_m3']}
    return dict(age=cp.state.time_myr,newborn_cell_count=int(newborn.sum()),mechanical_transition=cfg.get('young_shell',{}).get('mechanical_transition'),
        fields=fields,last_material_rows=[{k:r[k] for k in keys} for r in cp.lithosphere_rows[-5:]],slab_zones=len(cp.subduction_memory.zones),
        subduction_memory_cumulative_area_km2=cp.subduction_memory_rows[-1]['cumulative_subducted_area_km2'])

def hemisphere_probe(experiment):
    cfg,model,source,cp,fracture=load(experiment/'elapsed_0005')
    x=model.mesh.centroids;a=model.areas;owner=cp.state.cell_plate
    anti=cKDTree(x).query(-x)[1]
    rows=[]
    for pid in range(2):
        mask=owner==pid;center=np.sum(a[mask,None]*x[mask],axis=0);center/=np.linalg.norm(center)
        rows.append(dict(plate=pid,area_fraction=float(a@mask/a.sum()),
            hemisphere_mismatch_area=float(a@(mask!=(x@center>0))/a.sum()),centroid_direction=center.tolist()))
    local=np.cross(cp.mantle_flow.cell_omega_rad_per_myr,x)*model.thermal.radius_km
    # Non-axis-aligned cut avoids an equatorial centroid tie on this symmetric mesh.
    axis=np.array([1.,2,3])/np.sqrt(14);labels=(x@axis>0).astype(int)
    fit=plate_rigid_mantle_fit(model.mesh,labels,2,model.thermal.radius_km,cp.mantle_flow)
    represented=1-np.sum(a[:,None]*(local-np.cross(fit.omega_rad_per_myr[labels],x)*model.thermal.radius_km)**2)/np.sum(a[:,None]*local**2)
    return dict(actual_partition=rows,antipodal_same_owner_area_fraction=float(a@(owner==owner[anti])/a.sum()),
        synthetic_hemisphere_fit_same_field=dict(axis=axis.tolist(),represented_kinetic_fraction=float(represented),omega=fit.omega_rad_per_myr.tolist()))

if __name__=='__main__':
    experiment=ROOT/'analysis/plate_velocity_validation/experiments/velocity_least_squares'
    rows=[]; loaded=None
    for age in [5,50,100,200,400]:
        row,loaded=probe(experiment/f'elapsed_{age:04d}');rows.append(row)
    observed,_=probe(ROOT/'results/gui_runs/genesis_20260928_192334_470716/gui_checkpoint_0000110p8781_Myr')
    output=dict(corrected=rows,observed=observed,integrated_four_plate_normal_motion=integrate_normals(experiment,loaded))
    output['hemisphere_probe']=hemisphere_probe(experiment)
    if (Path(__file__).parent/'transport_405').exists():
        output['post_first_commit_mechanics']=post_commit_mechanics(Path(__file__).parent)
        output['first_commit_replay']=first_commit_replay(Path(__file__).parent,experiment)
    target=Path(__file__).with_suffix('.json')
    target.write_text(json.dumps(output,indent=2)+'\n',encoding='utf8')
    print(json.dumps(output,indent=2))
