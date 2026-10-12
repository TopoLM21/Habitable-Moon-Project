"""Independent saved-endpoint balance/restart inspection; no mechanical solve."""
import argparse,json,math,time
from pathlib import Path
import numpy as np
from .adapter import Adapter
from tectonics.genesis_transfer_bridge import encoded,digest
from tectonics.genesis import SECONDS_PER_MYR
from tectonics.genesis_shell import principal_tensile
from analysis.genesis_be_candidate.audit import face_driver
from analysis.genesis_volume_continuation_validation import independent_audit as q

ROOT=Path(__file__).resolve().parents[2]
OUT=Path('/workspace/genesis-aggregate-coupled-25-02')


def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=OUT)
    OUT=parser.parse_args().directory
    started=time.perf_counter();report=json.loads((OUT/'report.json').read_bytes())
    for name,pin in report['input_sha256'].items(): assert digest(Path(name).read_bytes())==pin
    p=json.loads(Path('/workspace/genesis-exploratory-100-01/continuation/initial.json').read_bytes())['packet'];root=json.loads((ROOT/'analysis/genesis_volume_creation_validation/evidence/actual_continuation/direct_after_second.json').read_bytes());full=json.loads(Path('/workspace/genesis-forced-coupled-25-01/candidate.json').read_bytes());r=json.loads((OUT/'after.json').read_bytes())
    a=Adapter(p,root['source_packet'],full['thermal_packet']['fixed_cut_geometry'],full['inputs'],report['contact_parameters'],thermal_reduction=r.get('thermal_reduction','projected_graph'))
    dt=(r['time_myr']-a.time)*SECONDS_PER_MYR;v=np.array(r['velocity_m_s']);du=np.array(r['displacement_m'])-a.u0;e=np.array(r['memory']);damage=np.array(r['fault_history']['damage']);T=np.array(r['temperature_k']);E=np.array(r['energy_j'])
    force=np.zeros_like(v);Q=[];W=[];U=0.;oldnewC=0.;numerical=0.;memory_error=0.
    for f,b in enumerate(a.blocks):
        C=a.C(f,damage);H=np.kron(b['G'],C);temp=T[a.solid_group[f]];sp=a.shell
        eta=np.clip(sp['viscosity_reference_pa_s']*np.exp(np.clip(sp['activation_energy_j_mol']/8.314462618*(1/temp-1/sp['viscosity_reference_temperature_k']),-60,60)),sp['viscosity_min_pa_s'],sp['viscosity_max_pa_s'])
        ratio=dt*sp['young_modulus_pa']/eta
        theta=((np.array([temp,0.])-b['old_temperature_coeff'])[:,None]*sp['linear_expansion_per_k']*np.array([1,1,1,0,0,0])[None,:]).ravel()
        err=(1+ratio)*e[f]-a.old[f]-b['L']@du[b['dofs']]+theta;memory_error=max(memory_error,float(np.max(np.abs(err))))
        np.add.at(force,b['dofs'],b['L'].T@H@e[f]);U+=float(.5*e[f]@H@e[f]);oldnewC+=float(.5*a.old[f]@H@a.old[f]);numerical+=float(.5*(e[f]-a.old[f])@H@(e[f]-a.old[f]))
        Q.append(float(ratio*e[f]@H@e[f]));W.append(float(-e[f]@H@theta))
    Q=np.array(Q);W=np.array(W);base=E.copy();np.add.at(base,a.solid_group,-Q+W);Tc=a.temperature(base)
    sourced=a.E0.copy()
    for row in r['ports']['ports']:sourced[row['recipient']]+=row['energy_into_thermal_j']
    # Match the actual source endpoint expression for the water/atmosphere node.
    sourced[-1]=a.m[-1]*(a.gp.water_heat_capacity_j_kg_k*r['ports']['source_surface_endpoint_k']+a.gp.water_latent_heat_j_kg)
    residual=base.astype(np.longdouble)-sourced.astype(np.longdouble)
    allow=8*(np.spacing(base)+np.spacing(sourced))+3e-11*a.m*a.cp
    for i,j,G in a.edges:
        heat=dt*G*(Tc[i]-Tc[j]);residual[i]+=heat;residual[j]-=heat
        allowance=16*dt*G*(math.ulp(float(Tc[i]))+math.ulp(float(Tc[j])));allow[i]+=allowance;allow[j]+=allowance
    thermal_ratio=float(np.max(np.abs(residual)/allow))
    nodal=a.M@(v-a.v0)/dt+a.D@v+force-a.F;momentum_relative=float(np.linalg.norm(nodal)/max(np.linalg.norm(force),np.linalg.norm(a.F),1.))
    kinetic0=float(.5*a.v0@a.M@a.v0);kinetic=float(.5*v@a.M@v);numkin=float(.5*(v-a.v0)@a.M@(v-a.v0));drag=float(dt*v@a.D@v);work=float(a.F@du)
    change=float(np.sum(E.astype(np.longdouble)-a.E0.astype(np.longdouble)));source_energy=math.fsum(x['energy_into_thermal_j'] for x in r['ports']['ports'])
    balance=math.fsum([change,kinetic-kinetic0,U-a.active_initial,-source_energy,-work,drag,numerical,numkin,-r['damage_energy_change_j']])
    # Global conservation uses only reservoir ULPs, not a sum of node Newton
    # tolerances; keep the same stricter global convention as full conduct().
    thermal_global_allow=float(np.sum(8*(np.spacing(base)+np.spacing(sourced))))
    feedback_allow=float(np.sum(8*(np.spacing(base[a.solid_group])+np.spacing(E[a.solid_group])))+np.sum(1e3+3e-8*(np.abs(Q)+np.abs(W))))
    global_allow=thermal_global_allow+feedback_allow+r['ports']['endpoint_roundoff_allowance_j']
    assert momentum_relative<=a.contact_parameters['equilibrium_tolerance'] and thermal_ratio<=1
    assert abs(balance)<=global_allow and memory_error<2e-12
    # Same original nodal mass operator, checked against separate quadrature.
    Mq=q.mass_matrix(a.X,a.faces,a.g['column_mass_kg'],a.g['density_kg_m3'],a.g['cohorts']);mass_error=float(np.max(np.abs(Mq-a.M))/np.max(np.abs(a.M)));assert mass_error<3e-12
    vf=np.array(full['state']['velocity_m_s']);physical=[];delta=v-vf
    for c in a.g['cohorts']:
        dd=delta.reshape(-1,6)[a.faces[c['face_index']]]
        for lam in (c['outer_scale'],c['inner_scale']):physical.extend(np.linalg.norm(dd[:,:3]+(lam-1)*dd[:,3:],axis=1))
    fullforce=a.F-a.M@(vf-a.v0)/dt-a.D@vf
    force_difference=float(np.linalg.norm(force-fullforce)/np.linalg.norm(fullforce))
    source_stress=face_driver(full['state'],root['source_packet'])['stress_pa']
    inherited=principal_tensile(source_stress)>=a.shell['tensile_strength_pa'];aggregated=principal_tensile(np.array(r['stress_pa']))>=a.shell['tensile_strength_pa']
    # Bound geometry movement, but do not call this a full-path contact certificate.
    displacement=np.array(r['displacement_m'])-a.u0;corners=[]
    for c in a.g['cohorts']:
        dd=displacement.reshape(-1,6)[a.faces[c['face_index']]]
        for lam in (c['outer_scale'],c['inner_scale']):corners.extend(np.linalg.norm(dd[:,:3]+(lam-1)*dd[:,3:],axis=1))
    surface_ratios=[]
    for f,group in enumerate(a.solid_group):
        first=next(i for i,c in enumerate(a.cuts) if c['face']==f)
        surface_ratios.append(1. if a.thermal_reduction=='series_resistance' else a.m[group]/p['thermal']['mass_kg'][first])
    a.restore(r)
    assert np.array_equal(a.old,e) and np.array_equal(a.v0,v) and np.array_equal(a.E0,E) and a.reserve==r['unresolved_reserve_j']
    result=dict(status='SAVED_BALANCES_AND_ACTIVE_RESTART_PASS_NOT_PHYSICAL_ADMISSION',new_time_steps=0,
        input_after_sha256=digest((OUT/'after.json').read_bytes()),mass_operator_relative_difference=mass_error,
        memory_equation_max_absolute_error=memory_error,momentum_relative=momentum_relative,thermal_node_balance_over_allowance=thermal_ratio,
        whole_balance_residual_j=balance,whole_balance_allowance_j=global_allow,unresolved_reserve_unchanged_j=a.reserve,
        physical_corner_velocity_difference_max_m_s=float(max(physical)),internal_force_difference_relative_l2=force_difference,
        full_force_comparison_method='Full endpoint force inferred from saved full-mode nodal impulse equation; not independently re-integrated stress.',
        full_tensile_above_12MPa=int(inherited.sum()),aggregate_tensile_above_12MPa=int(aggregated.sum()),changed_tensile_classification_faces=np.flatnonzero(inherited!=aggregated).tolist(),
        maximum_physical_corner_motion_m=float(max(corners)),
        collapsed_surface_conductance_over_whole_solid_cell_conductance_min_max=[float(min(surface_ratios)),float(max(surface_ratios))],
        active_restart_restored=True,full_path_geometry_event_certificate=False,
        no_second_trial=True,seconds=time.perf_counter()-started)
    with (OUT/'audit_strict.json').open('xb') as f:f.write(encoded(result))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
