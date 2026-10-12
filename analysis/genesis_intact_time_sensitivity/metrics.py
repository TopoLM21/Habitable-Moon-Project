"""Comparable source-bound endpoint metrics; newborn IDs are not matched."""
import numpy as np


LEDGERS=('external_work_j','basal_dissipation_j','physical_mechanical_work_j')
MECHANICS=('thermal_work_j','viscous_dissipation_j','damage_energy_change_j','remelt_energy_removed_j')


def material(packet,old_ids):
    r=packet['runtime_state'];g=r['geometry'];cohorts=g['cohorts']
    mass=np.zeros(80);newmass=np.zeros(80);front=np.zeros(80);groups=[[] for _ in range(80)]
    for c in cohorts:
        i=c['face_index'];mass[i]+=c['mass_kg'];front[i]=max(front[i],c['live_q_hi']);groups[i].append((c['q_lo'],c['live_q_hi']))
        if c['id'] not in old_ids:newmass[i]+=c['mass_kg']
    for group in groups:
        end=0.
        for lo,hi in sorted(group):
            if abs(lo-end)>1e-14 or hi<lo:raise ValueError('Material gap/overlap prevents common-q comparison')
            end=hi
    return dict(cohorts=len(cohorts),mass_kg=float(mass.sum()),newborn_mass_kg=float(newmass.sum()),
        mass_by_face_kg=mass.tolist(),newborn_mass_by_face_kg=newmass.tolist(),front_q=front.tolist(),
        column_mass_kg=g['column_mass_kg'],geometry_vertices_m=g['vertices_m'],common_q_comparison_valid=True)


def extract(packet,parent):
    r=packet['runtime_state'];p=parent['runtime_state'];old_ids={c['id'] for c in p['geometry']['cohorts']}
    old_memory={c['id']:c for c in r['mechanics']['state']['cohorts'] if c['id'] in old_ids}
    if set(old_memory)!=old_ids:raise ValueError('Old material/memory IDs lost')
    source=r['last_report']['fault_evolution']['predictor_driver']
    out=material(packet,old_ids)
    out.update(time_myr=r['thermal']['time_myr'],damage=r['fault_history']['damage'],
        A_m=np.asarray(r['displacement_m']).reshape(-1,6)[:,:3].tolist(),
        P_m=np.asarray(r['displacement_m']).reshape(-1,6)[:,3:].tolist(),
        velocity_A_m_s=np.asarray(r['velocity_m_s']).reshape(-1,6)[:,:3].tolist(),
        face_tensile_pa=r['last_report']['fault_evolution']['final_face_tensile_pa'],
        old_cohort_temperature_k=[old_memory[i]['temperature_k'] for i in sorted(old_ids)],
        old_cohort_strain0=[old_memory[i]['strain0'] for i in sorted(old_ids)],
        old_cohort_strain1=[old_memory[i]['strain1'] for i in sorted(old_ids)],
        elastic_energy_j=r['mechanics']['state']['elastic_energy_j'],kinetic_energy_j=r['kinetic_energy_j'],
        column_enthalpy=r['thermal']['column_enthalpy'],global_thermal_energy=r['thermal']['global_state']['energy'],
        surface_temperature_k=source['surface_temperature_k'],
        boundary_heat_j=r['thermal']['boundary_energy_j']-p['thermal']['boundary_energy_j'],
        tidal_heat_j=r['thermal']['tidal_heat_received_j']-p['thermal']['tidal_heat_received_j'],
        maxwell_ratio_last_step=r['last_report']['audit_input']['maxwell_ratio'][0])
    for k in LEDGERS:out[k]=r[k]-p[k]
    for k in MECHANICS:out[k]=r['mechanics']['state'][k]-p['mechanics']['state'][k]
    return out


def difference(a,b):
    x,y=np.asarray(a,float),np.asarray(b,float)
    if x.shape!=y.shape:raise ValueError('Incomparable observables')
    absolute=float(np.max(np.abs(y-x)))
    scale=float(max(np.max(np.abs(x)),np.max(np.abs(y))))
    return dict(max_abs_difference=absolute,symmetric_relative_max=absolute/scale if scale else 0.)


def compare(a,b):
    if abs(a['time_myr']-b['time_myr'])>1e-14:raise ValueError('Different endpoint times')
    if a['column_mass_kg']!=b['column_mass_kg'] or a['geometry_vertices_m']!=b['geometry_vertices_m']:
        raise ValueError('Different material mass chart/reference mesh')
    columns=np.asarray(a['column_mass_kg']);fa,fb=np.asarray(a['front_q']),np.asarray(b['front_q'])
    keys=['mass_kg','newborn_mass_kg','mass_by_face_kg','newborn_mass_by_face_kg','front_q','A_m','P_m','velocity_A_m_s',
        'face_tensile_pa','damage','old_cohort_temperature_k','old_cohort_strain0','old_cohort_strain1',
        'elastic_energy_j','kinetic_energy_j','column_enthalpy','global_thermal_energy','surface_temperature_k','boundary_heat_j','tidal_heat_j',*LEDGERS,*MECHANICS]
    return dict(observables={k:difference(a[k],b[k]) for k in keys},
        common_live_material_mass_kg=float(np.sum(columns*np.minimum(fa,fb))),
        unmatched_live_material_mass_kg=float(np.sum(columns*np.abs(fa-fb))),
        cohort_counts=[a['cohorts'],b['cohorts']],newborn_ids_compared=False,
        pure_time_convergence_claimed=False,error_bound_claimed=False)
