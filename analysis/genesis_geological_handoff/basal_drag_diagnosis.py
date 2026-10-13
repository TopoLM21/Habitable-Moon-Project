"""Read-only matrix/geometry diagnosis. No time step, quadrature audit or restore."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from tectonics.simplified_genesis import SimplifiedGenesis
from tectonics.simplified_genesis_nodal import operators,material_mass,blocks,rate
from tectonics.genesis_shell import principal_tensile


def stats(x):
    x=np.asarray(x);return dict(min=float(x.min()),max=float(x.max()),mean=float(x.mean()))


def diagnose(path):
    start=time.monotonic();p=Path(path);pin=hashlib.sha256(p.read_bytes()).hexdigest()
    m=SimplifiedGenesis.load(p);c,s=m.config,m.state;o=operators(m);M=material_mass(m)
    X=o['X'];tri=X[m.mesh.faces];cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);area=np.linalg.norm(cross,axis=1)/2
    n=cross/(2*area[:,None]);height=np.einsum('fi,fi->f',tri[:,0],n)
    q0=np.zeros(m.nf)
    for x in c['source_material_catalogue']:q0[x['face_index']]=max(q0[x['face_index']],x['live_q_hi'])
    qnow=q0.copy();catalog={x['id']:x for x in c['phase_cells']}
    for x in s.get('born_material',[]):qnow[x['face']]=max(qnow[x['face']],catalog[x['cell_id']]['q_hi'])
    coefficient=6*np.array(c['nodal']['column_mass'])/(c['nodal']['density']*np.linalg.det(tri.transpose(0,2,1)))
    l0=np.cbrt(1-coefficient*q0);lnow=np.cbrt(1-coefficient*qnow)
    report_path=Path(__file__).parents[1]/'genesis_thermal_moment/report.json'
    source_report=json.loads(report_path.read_bytes())
    beta=source_report['contact_parameters']['basal_drag_pa_s_m']
    load_path=Path('/workspace/genesis-forced-coupled-25-01/candidate.json')
    assert hashlib.sha256(load_path.read_bytes()).hexdigest()==source_report['input_sha256'][str(load_path)]
    original_inputs=json.loads(load_path.read_bytes())['inputs']
    assert original_inputs['basal_drag_pa_s_m']==beta
    assembled=np.zeros_like(o['D']);normal_leak=[]
    for f,idx in enumerate(o['dofs']):
        assembled[np.ix_(idx,idx)]+=o['face_D'][f]
        vn=np.tile(np.r_[n[f],np.zeros(3)],3)
        normal_leak.append(float(np.linalg.norm(o['face_D'][f]@vn)/max(np.linalg.norm(o['face_D'][f])*np.linalg.norm(vn),1.)))
    assert np.allclose(assembled,o['D'],rtol=1e-14,atol=0.)
    T=m.temperature(s['energy']);damage=np.array(s['damage']);bb=blocks(m,T,damage,T)
    K=np.zeros_like(M);force=np.zeros(len(M));energy0=0.
    for b in bb:
        idx=b['dofs'];K[np.ix_(idx,idx)]+=b['L'].T@b['H']@b['L']
        np.add.at(force,idx,b['L'].T@b['H']@b['old']);energy0+=.5*b['old']@b['H']@b['old']
    # z is physical displacement for a unit homothetic dilation: A=P=X.
    z=np.concatenate([np.r_[x,x] for x in X]);k=float(z@K@z);d=float(z@o['D']@z);mass=float(z@M@z)
    v=np.array(s['nodal_velocity']);drag=o['D']@v
    power=sum(float(v[idx]@fd@v[idx]) for idx,fd in zip(o['dofs'],o['face_D']))
    ratio=power/float(v@o['D']@v)
    assert abs(ratio-1)<1e-12
    sigma=np.array(s['stress_pa']);tensile=principal_tensile(sigma)
    raw=np.maximum(tensile/m.shell['tensile_strength_pa']-1,0)**2/m.shell['damage_timescale_myr']
    smooth=m.smoother.apply(raw,np.ones(m.nf,bool));hot=np.clip((T[m.solid_nodes].mean(axis=1)-700)/650,0,1)
    heal=1/m.shell['cold_healing_timescale_myr']+hot**4/m.shell['hot_healing_timescale_myr']
    times=1/rate(m,T[m.solid_nodes]);unittri=tri/np.linalg.norm(tri,axis=2)[:,:,None]
    radial_tangent=np.sqrt(np.maximum(0,1-np.einsum('fvi,fi->fv',unittri,n)**2))
    source=SimplifiedGenesis.load('/workspace/genesis-nodal-source-01.json')
    assert source.config['nodal']['D']==c['nodal']['D']
    result=dict(path=path,sha256=pin,time_myr=s['time_myr'],beta_pa_s_m=beta,
        radius_m=stats(np.linalg.norm(X,axis=1)),top_area_m2=float(area.sum()),basal_reference_area_m2=float(np.sum(area*l0*l0)),spherical_area_m2=float(4*np.pi*(c['radius_km']*1000)**2),
        initial_normal_thickness_m=stats(height*(1-l0)),current_front_normal_thickness_m=stats(height*(1-lnow)),basal_reference_scale=stats(l0),
        nodal_spherical_radial_tangent_fraction=stats(radial_tangent),facet_normal_drag_relative_leak=max(normal_leak),assembled_D_relative_error=float(np.linalg.norm(assembled-o['D'])/np.linalg.norm(o['D'])),heat_sum_over_matrix_drag_power=ratio,basal_operator_unchanged_since_import=True,
        homothetic_mode=dict(stiffness_j=k,drag_j_s=d,inertia_kg_m2=mass,drag_over_stiffness_years=d/k/31557600.,inertial_oscillation_scale_seconds=float(np.sqrt(mass/k)),inertia_over_drag_seconds=mass/d,unit_thermal_cooling_generalized_force_j=k*m.shell['linear_expansion_per_k'],stored_internal_generalized_force_j=float(z@force),stored_drag_generalized_force_j=float(z@drag),external_generalized_force_j=float(z@o['F']),instantaneous_radial_rate_per_year=float(np.mean(np.einsum('ni,ni->n',v.reshape(-1,6)[:,:3],X)/np.sum(X*X,axis=1)))*31557600.,caveat='Rayleigh scales for prescribed homothetic mode, not an eigenmode or a forecast; no evolution'),
        generalized_force_norm_n=dict(internal=float(np.linalg.norm(force)),drag=float(np.linalg.norm(drag)),external=float(np.linalg.norm(o['F'])),unsteady_residual=float(np.linalg.norm(force+drag-o['F']))),
        last_accepted_momentum_residual=s['last_ledger']['momentum_relative'],drag_power_w=float(v@drag),elastic_energy_j=energy0,
        maxwell_time_years=stats(times/31557600.),damage=stats(damage),stiffness_fraction=stats(m.scale(damage)),tensile_pa=stats(tensile),instantaneous_no_tide_damage_drive_per_myr=stats(smooth),healing_per_myr=stats(heal),
        initial_maxwell_time_years=stats((1/rate(source,source.temperature(source.state['energy'])[source.solid_nodes]))/31557600.),
        prescribed_traction_pa=stats(np.linalg.norm(original_inputs['basal_traction_at_vertices_pa'],axis=1)),
        traction_source=dict(amplitude_pa=m.shell['convective_traction_pa'],coupling_depth_km=m.shell['traction_coupling_depth_km'],coupling_fraction_at_initial_H=stats(-np.expm1(-height*(1-l0)/1000/m.shell['traction_coupling_depth_km']))),
        beta_source=str(report_path),beta_matches_pinned_original_inputs=True,
        viscosity_limits_pa_s=[m.shell['viscosity_min_pa_s'],m.shell['viscosity_max_pa_s']],
        seconds=time.monotonic()-start,no_new_advance=True)
    assert hashlib.sha256(p.read_bytes()).hexdigest()==pin
    return result


if __name__=='__main__':
    result=diagnose('/workspace/genesis-nodal-pilot-01/checkpoint.json')
    out=Path(__file__).with_name('basal_drag_diagnosis.json');out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
