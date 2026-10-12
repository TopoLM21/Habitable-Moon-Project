"""Read-only common-material kinetic diagnostic. Never loads a runtime model."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from analysis.genesis_volume_continuation_validation.independent_audit import (
    mass_matrix, angular_rule, conical_scales, basal_drag_and_force)

PARENT = Path('analysis/genesis_volume_creation_validation/evidence/actual_continuation/direct_after_second.json')
PARENT_SHA = '62579a0bad128830a66e2f44579cf2b1f5d3835b691225b8d431102f3e4486d7'
SECONDS_PER_MYR = 365.25 * 86400 * 1e6
SUPPORT = ('face_index', 'q_lo', 'live_q_hi', 'mass_kg', 'material_id', 'owner_id')


def read(path, pin=None):
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if pin is not None and sha != pin:
        raise ValueError(f'Artifact hash mismatch: {path}')
    return json.loads(raw), {'path': str(path), 'sha256': sha}


def fronts(g):
    out = np.zeros(len(g['faces']))
    for face in range(len(out)):
        end = 0.
        for c in sorted((c for c in g['cohorts'] if c['face_index'] == face), key=lambda c:c['q_lo']):
            if c['q_lo'] != end or c['live_q_hi'] <= end:
                raise ValueError('Material gap/overlap')
            end = c['live_q_hi']
        out[face] = end
    return out


def bands(lo, hi):
    if np.any(np.asarray(hi) < lo):
        raise ValueError('Material retreat is outside this diagnostic')
    return [dict(face_index=i, q_lo=float(a), live_q_hi=float(b))
            for i, (a, b) in enumerate(zip(lo, hi)) if b > a]


def matrix(g, rows):
    return mass_matrix(g['vertices_m'], g['faces'], g['column_mass_kg'], g['density_kg_m3'], rows)


def kinetic(M, v):
    return float(.5 * v @ M @ v)


def rms(M, v, mass):
    return float(np.sqrt(max(0., 2*kinetic(M, v)/mass)))


def direct_energy(g, rows, v, order=64):
    """Pointwise physical velocity integration, no assembled mass matrix.

    Aggregate face bands remove cohort partition dependence. Angular integrand
    is quadratic; 6-point-per-axis Duffy quadrature is exact for it. Radial
    order comparison is a numerical cross-check, not a new physical tolerance.
    """
    N, wa = angular_rule(6)
    x, w = np.polynomial.legendre.leggauss(order)
    X = np.asarray(g['vertices_m']); faces = np.asarray(g['faces'])
    velocity = np.asarray(v).reshape(-1, 6); result = 0.
    for c in rows:
        i, lo, hi = c['face_index'], c['q_lo'], c['live_q_hi']
        q = lo + (hi-lo)*(x+1)/2
        _, lm1 = conical_scales(X[faces[i]], g['column_mass_kg'][i], g['density_kg_m3'], q)
        va = N @ velocity[faces[i], :3]; vp = N @ velocity[faces[i], 3:]
        values = va[:, None, :] + lm1[None, :, None]*vp[:, None, :]
        result += .5*g['column_mass_kg'][i]*(hi-lo)*np.einsum('a,r,ark,ark->', wa, w/2, values, values)
    return float(result)


def compare_velocities(M, mass, a, b):
    difference = b-a
    norm_product = np.sqrt((a @ M @ a)*(b @ M @ b))
    return dict(mass_weighted_cosine=float(a @ M @ b)/norm_product if norm_product else None,
                kinetic_a_j=kinetic(M,a), kinetic_b_j=kinetic(M,b),
                signed_energy_difference_j=kinetic(M,b)-kinetic(M,a),
                cross_work_j=float(((a+b)/2) @ M @ difference),
                difference_field_energy_j=kinetic(M,difference),
                rms_a_m_s=rms(M,a,mass), rms_b_m_s=rms(M,b,mass),
                rms_difference_m_s=rms(M,difference,mass),
                relative_rms_difference=rms(M,difference,mass)/max(rms(M,a,mass),rms(M,b,mass)))


def same_time(a, b):
    # Only machine-rounding allowance on stored clocks; no physical time tolerance.
    if abs(a-b) > 8*np.finfo(float).eps*max(abs(a),abs(b)):
        raise ValueError('Different physical endpoint times')


def run(directory):
    summary, _ = read(Path('analysis/genesis_intact_time_sensitivity/summary.json'))
    report, report_pin = read(directory/'report.json', summary['full_report']['sha256'])
    if report['status'] != 'PASS': raise ValueError('Series did not pass')
    for name, pin in report['artifacts'].items(): read(directory/name, pin['sha256'])
    parent, parent_pin = read(PARENT, PARENT_SHA)
    baseline, baseline_pin = read(Path(report['baseline_pin']['path']), report['baseline_pin']['sha256'])
    packets = {'parent':parent, '100_1':baseline}; pins = {'parent':parent_pin, '100_1':baseline_pin}
    for label, n in [('50',2),('25',4)]:
        for i in range(1,n+1):
            key=f'{label}_{i}'; name=f'{label}/step_{i}.json'
            packets[key], pins[key] = read(directory/name, report['artifacts'][name]['sha256'])
    states = {k:p['runtime_state'] for k,p in packets.items()}
    for packet in packets.values():
        if packet['source_packet']!=parent['source_packet']:raise ValueError('Source packet changed')
    g0=states['parent']['geometry']; old={c['id']:c for c in g0['cohorts']}; f0=fronts(g0)
    M0=matrix(g0,g0['cohorts']); oldmass=sum(c['mass_kg'] for c in old.values())
    endpoints={}; matrices={}; times={}
    for key,s in states.items():
        g=s['geometry']; rows={c['id']:c for c in g['cohorts']}
        if len(rows)!=len(g['cohorts']):raise ValueError('Duplicate material ID')
        for field in ('vertices_m','faces','column_mass_kg','density_kg_m3'):
            if g[field]!=g0[field]:raise ValueError('Reference mass chart changed')
        for i,c in old.items():
            if i not in rows or any(rows[i][f]!=c[f] for f in SUPPORT):raise ValueError('Old material support changed')
        for field in ('thermal','fault_history'):
            same_time(s[field]['time_myr'],s['thermal']['time_myr'])
        f=fronts(g); new=bands(f0,f); full=bands(np.zeros(len(f)),f)
        Mnew=matrix(g,new); Mfull=matrix(g,g['cohorts']); matrices[key]=(Mnew,Mfull)
        v=np.asarray(s['velocity_m_s']); K=kinetic(Mfull,v)
        direct=direct_energy(g,full,v); direct16=direct_energy(g,full,v,16)
        old_direct=direct_energy(g,bands(np.zeros(len(f0)),f0),v)
        if abs(old_direct-kinetic(M0,v))>1e-6+5e-10*abs(old_direct):raise ValueError('Old mass integration differs')
        for check in (direct,direct16,kinetic(M0,v)+kinetic(Mnew,v)):
            if abs(K-check)>1e-6+5e-10*abs(K):raise ValueError('Independent mass integration differs')
        newmass=float(np.dot(g['column_mass_kg'],f-f0))
        # Reuse the established endpoint audit's computational acceptance test.
        if abs(K-s['kinetic_energy_j'])>1e-6+5e-10*abs(K):raise ValueError('Saved kinetic energy differs')
        endpoints[key]=dict(time_myr=s['thermal']['time_myr'],cohorts=len(rows),
            old_kinetic_j=kinetic(M0,v),newborn_kinetic_j=kinetic(Mnew,v),total_kinetic_j=K,
            old_rms_speed_m_s=rms(M0,v,oldmass),
            newborn_mass_kg=newmass,
            newborn_rms_speed_m_s=rms(Mnew,v,newmass) if newmass else None,
            old_velocity_vs_parent=compare_velocities(M0,oldmass,np.asarray(states['parent']['velocity_m_s']),v),
            total_mass_kg=sum(c['mass_kg'] for c in rows.values()),
            partition_energy_residual_j=K-kinetic(M0,v)-kinetic(Mnew,v),
            direct_pointwise_energy_j=direct,matrix_minus_direct_j=K-direct,old_matrix_minus_direct_j=kinetic(M0,v)-old_direct,
            radial_16_minus_64_j=direct16-direct,
            max_damage=max(s['fault_history']['damage']))
        times[key]=s['thermal']['time_myr']
    steps={}
    for label,n in [('100',1),('50',2),('25',4)]:
        previous='parent'; cumulative_birth=0.
        for i in range(1,n+1):
            key=f'{label}_{i}'; a,b=states[previous],states[key]; r=b['last_report']; inp=r['audit_input']
            same_time(times[previous],r['before_time_myr']);same_time(times[key],r['after_time_myr'])
            dt=(times[key]-times[previous])*SECONDS_PER_MYR
            v0=np.asarray(a['velocity_m_s']);v1=np.asarray(b['velocity_m_s']);inc=np.asarray(inp['increment_m'])
            if inp['before_velocity_m_s']!=a['velocity_m_s'] or inp['before_displacement_m']!=a['displacement_m']:
                raise ValueError('Saved step predecessor differs')
            midpoint=inc/dt; alternating=(v1-v0)/2
            g=b['geometry'];M=matrices[key][1]
            born=bands(fronts(a['geometry']),fronts(g)); Mb=matrix(g,born)
            faces=np.asarray(g['faces']);D,F=basal_drag_and_force(g['vertices_m'],g['faces'],g['column_mass_kg'],g['density_kg_m3'],g['cohorts'],inp['basal_drag_pa_s_m'],np.asarray(inp['basal_traction_at_vertices_pa'])[faces])
            deltaK=kinetic(M,v1)-kinetic(M,v0)
            force_inertia=M@(v1-v0)/dt
            cumulative_birth+=kinetic(Mb,v0)
            endpoints[key]['cumulative_birth_flux_j']=cumulative_birth
            steps[key]=dict(dt_years=dt/(365.25*86400),
                velocity_identity_max_abs_m_s=float(np.max(np.abs(v1-(2*midpoint-v0)))),
                displacement_identity_max_abs_m=float(np.max(np.abs(np.asarray(b['displacement_m'])-np.asarray(a['displacement_m'])-inc))),
                old_midpoint_kinetic_j=kinetic(M0,midpoint),old_alternating_kinetic_j=kinetic(M0,alternating),
                old_midpoint_alternating_cross_j=float(midpoint@M0@alternating),
                birth_flux_j=kinetic(Mb,v0),saved_birth_flux_j=r['birth_kinetic_energy_j'],
                birth_flux_difference_j=kinetic(Mb,v0)-r['birth_kinetic_energy_j'],
                inertial_work_j=float(force_inertia@inc),fixed_after_mass_delta_kinetic_j=deltaK,
                inertial_work_identity_residual_j=float(force_inertia@inc)-deltaK,
                drag_work_j=float(midpoint@D@midpoint*dt),saved_drag_work_j=r['basal_dissipation_j'],
                drag_work_relative_difference=(float(midpoint@D@midpoint*dt)-r['basal_dissipation_j'])/r['basal_dissipation_j'],
                inertial_force_norm_n=float(np.linalg.norm(force_inertia)),external_force_norm_n=float(np.linalg.norm(F)),
                drag_force_norm_n=float(np.linalg.norm(D@midpoint)),
                old_midpoint_rms_m_s=rms(M0,midpoint,oldmass),old_alternating_rms_m_s=rms(M0,alternating,oldmass))
            previous=key
    comparisons={}
    for ka,kb,start in [('50_1','25_2','parent'),('50_2','25_4','50_1'),('100_1','50_2','parent'),('100_1','25_4','parent')]:
        a,b=states[ka],states[kb];same_time(times[ka],times[kb])
        va=np.asarray(a['velocity_m_s']);vb=np.asarray(b['velocity_m_s'])
        fa,fb=fronts(a['geometry']),fronts(b['geometry']);common=bands(f0,np.minimum(fa,fb));Mc=matrix(g0,common)
        # Equal physical window for averaged velocities, including 2x25 vs 1x50.
        sa=states[start]; sb=states['25_2'] if start=='50_1' else sa
        same_time(sa['thermal']['time_myr'],sb['thermal']['time_myr'])
        dt=(times[ka]-sa['thermal']['time_myr'])*SECONDS_PER_MYR
        ua=(np.asarray(a['displacement_m'])-np.asarray(sa['displacement_m']))/dt
        ub=(np.asarray(b['displacement_m'])-np.asarray(sb['displacement_m']))/dt
        total=endpoints[kb]['total_kinetic_j']-endpoints[ka]['total_kinetic_j']
        olddelta=kinetic(M0,vb)-kinetic(M0,va);newdelta=kinetic(Mc,vb)-kinetic(Mc,va)
        unmatched=kinetic(matrix(g0,bands(np.minimum(fa,fb),fb)),vb)-kinetic(matrix(g0,bands(np.minimum(fa,fb),fa)),va)
        comparisons[ka+'_vs_'+kb]=dict(time_a_myr=times[ka],time_b_myr=times[kb],
            total_signed_difference_j=total,total_relative_difference=abs(total)/max(endpoints[ka]['total_kinetic_j'],endpoints[kb]['total_kinetic_j']),
            old_signed_difference_j=olddelta,common_newborn_signed_difference_j=newdelta,
            unmatched_material_signed_difference_j=unmatched,decomposition_residual_j=total-olddelta-newdelta-unmatched,
            old_endpoint_velocity=compare_velocities(M0,oldmass,va,vb),
            common_newborn_endpoint_velocity=compare_velocities(Mc,float(np.dot(g0['column_mass_kg'],np.minimum(fa,fb)-f0)),va,vb),
            common_window_years=dt/(365.25*86400),old_window_average_velocity=compare_velocities(M0,oldmass,ua,ub),
            old_endpoint_minus_window_average=compare_velocities(M0,oldmass,va-ua,vb-ub),
            max_damage_difference=float(np.max(np.abs(np.asarray(a['fault_history']['damage'])-b['fault_history']['damage']))),
            max_face_tension_difference_pa=float(np.max(np.abs(np.asarray(a['last_report']['fault_evolution']['final_face_tensile_pa'])-b['last_report']['fault_evolution']['final_face_tensile_pa']))),
            total_difference_over_elastic_energy=abs(total)/max(a['mechanics']['state']['elastic_energy_j'],b['mechanics']['state']['elastic_energy_j']))
    checks=[s for branch in report['branches'].values() for s in branch['steps']]
    return dict(status='READ_ONLY_DIAGNOSTIC_COMPLETE',base_code_sha='c1ec56c4b43bf4ce9fc2ed02209496e5734736cb',
        inputs=pins,series_report=report_pin,common_old_cohorts=len(old),common_old_mass_kg=oldmass,
        endpoints=endpoints,steps=steps,comparisons=comparisons,
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        prior_audits=dict(max_relative_balance=max(s['independent_balance']['relative_global_balance_residual'] for s in checks),
            max_relative_coordinate_momentum=max(s['independent_balance']['relative_max_coordinate_momentum_residual'] for s in checks),
            absolute_balance_residuals_j=[s['independent_balance']['global_balance_residual_j'] for s in checks]),
        new_evolution=False,physical_error_bound_claimed=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=run(a.input)
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({'status':result['status'],'comparisons':result['comparisons']}))
