"""Independent BE chain audit; integrate damage delta-C before summing.

Derived from the existing direct quadrature audit. Tolerances unchanged;
subtracting two huge separately summed energies is avoided for damage work.
"""
import hashlib,json
import numpy as np
from analysis.genesis_volume_continuation_validation import independent_audit as q
from analysis.genesis_volume_fault_validation import independent_audit as fault


def close(a,b,name,rtol=3e-10,atol=1e-8):
    a,b=np.asarray(a),np.asarray(b)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError(name+' dimensions/finite')
    if a.dtype.kind=='b' or b.dtype.kind=='b':
        if a.dtype.kind!='b' or b.dtype.kind!='b' or not np.array_equal(a,b):raise ValueError(name+' boolean mismatch')
        return 0.
    err=float(np.max(np.abs(a-b),initial=0));scale=float(np.max(np.abs(b),initial=0))
    if err>atol+rtol*scale:raise ValueError(name+' mismatch')
    return err/max(scale,atol,np.finfo(float).tiny)


def block(triangle,frame,column_mass,rho,cohort,C,old0,old1,new0,new1,thermal0,ratio,du):
    N,wa=q.angular_rule();interp=N@np.linalg.inv(q.SOURCE_ANGULAR_POINTS)
    radial,wr=q.radial_rule(cohort['q_lo'],cohort['live_q_hi']);lam,_=q.conical_scales(triangle,column_mass,rho,radial)
    B0,B1=q.independent_strain_operators(triangle,frame,N)
    B=B0[:,None,:,:]+B1[:,None,:,:]/lam[None,:,None,None]
    value=lambda e0,e1:(interp@np.asarray(e0))[:,None,:]+(interp@np.asarray(e1))[:,None,:]/lam[None,:,None]
    old=value(old0,old1);new=value(new0,new1);thermal=(interp@thermal0)[:,None,:];inc=B@du
    memory_error=close(new,(old+inc-thermal)/(1+ratio),'BE memory',rtol=2e-9,atol=2e-12)
    weights=wa[:,None]*wr[None,:]*column_mass*(cohort['live_q_hi']-cohort['q_lo'])/rho
    inner=lambda a,b:float(np.einsum('ar,ari,ij,arj->',weights,a,C,b))
    force=np.einsum('ar,arik,ij,arj->k',weights,B,C,new)
    return dict(elastic_after_j=.5*inner(new,new),elastic_before_after_damage_j=.5*inner(old,old),
        internal_work_j=inner(new,inc),thermal_work_j=-inner(new,thermal),physical_maxwell_j=ratio*inner(new,new),
        numerical_memory_j=.5*inner(new-old,new-old)),force,memory_error


def trial(before,after,inputs,source):
    g=after['geometry'];g0=before['geometry'];X=np.asarray(g['vertices_m']);faces=np.asarray(g['faces']);mass=np.asarray(g['column_mass_kg']);rho=g['density_kg_m3']
    shell=source['source']['parameters']['shell'];old={c['id']:c for c in before['mechanics']['state']['cohorts']};new={c['id']:c for c in after['mechanics']['cohorts']}
    u0=np.asarray(before['displacement_m']);u1=np.asarray(after['displacement_m']);v0=np.asarray(before['velocity_m_s']);v1=np.asarray(after['velocity_m_s'])
    dt=(after['thermal']['time_myr']-before['thermal']['time_myr'])*365.25*86400*1e6
    du=u1-u0;close(v1,du/dt,'BE endpoint equals path velocity',rtol=3e-10,atol=1e-20)
    close(inputs['before_displacement_m'],u0,'predecessor u',0,0);close(inputs['before_velocity_m_s'],v0,'predecessor v',0,0)
    oldgeom={c['id']:c for c in g0['cohorts']};newgeom={c['id']:c for c in g['cohorts']}
    if len(new)!=len(after['mechanics']['cohorts']) or set(new)!=set(newgeom) or not set(oldgeom)<=set(newgeom):raise ValueError('Material IDs')
    for k in ['remelt_elastic_outflow_j','remelt_kinetic_outflow_j']:
        if after['interval_ledger'][k]!=0:raise ValueError('Unsupported remelt ledger')
    for i,c in oldgeom.items():
        for k in ['face_index','q_lo','live_q_hi','mass_kg','owner_id','material_id']:
            if c[k]!=newgeom[i][k]:raise ValueError('Old material changed')
    for k in ['vertices_m','faces','column_mass_kg','density_kg_m3']:
        if g[k]!=g0[k]:raise ValueError('Reference chart changed')
    totals={};force=np.zeros(len(u0));maxmemory=0.;initial=0.;damage_change=0.;damage=after['fault_history']['damage']
    for i,c in enumerate(g['cohorts']):
        f=c['face_index'];triangle=X[faces[f]];frame=fault.physical_face_geometry(X,faces[f][None,:])[0][0]
        cid=c['id'];eold=old.get(cid);zeros=np.zeros((3,6));old0=zeros if eold is None else np.asarray(eold['strain0']);old1=zeros if eold is None else np.asarray(eold['strain1'])
        close(inputs['old0'][i],old0,'original memory0',0,0);close(inputs['old1'][i],old1,'original memory1',0,0)
        temp=new[cid]['temperature_k'];eta=np.clip(shell['viscosity_reference_pa_s']*np.exp(np.clip(shell['activation_energy_j_mol']/8.314462618*(1/temp-1/shell['viscosity_reference_temperature_k']),-60,60)),shell['viscosity_min_pa_s'],shell['viscosity_max_pa_s'])
        ratio=0. if eold is None else dt*shell['young_modulus_pa']/eta
        close(inputs['maxwell_ratio'][i],ratio,'source relaxation',rtol=2e-10,atol=1e-15)
        C=q.isotropic_matrix(shell['young_modulus_pa'],shell['poisson_ratio'],damage[f],shell['residual_stiffness'])
        close(inputs['elasticity_pa'][i],C,'source elasticity',rtol=2e-14,atol=1e-6)
        idx=(6*faces[f,:,None]+np.arange(6)).ravel()
        values,local,err=block(triangle,frame,mass[f],rho,c,C,old0,old1,np.asarray(new[cid]['strain0']),np.asarray(new[cid]['strain1']),np.asarray(inputs['thermal0'][i]),ratio,du[idx])
        for k,v in values.items():totals[k]=totals.get(k,0.)+v
        np.add.at(force,idx,local);maxmemory=max(maxmemory,err)
        if eold is not None:
            C0=q.isotropic_matrix(shell['young_modulus_pa'],shell['poisson_ratio'],before['fault_history']['damage'][f],shell['residual_stiffness'])
            initial+=q.coefficient_energy(triangle,mass[f],rho,c,old0,old1,C0)
            damage_change+=damage_delta(triangle,mass[f],rho,c,old0,old1,C0,C)
    born=[c for c in g['cohorts'] if c['id'] not in oldgeom]
    M0=q.mass_matrix(X,faces,mass,rho,g0['cohorts']);M=q.mass_matrix(X,faces,mass,rho,g['cohorts']);Min=q.mass_matrix(X,faces,mass,rho,born)
    traction=q.prescribed_traction(X,faces,mass,rho,g['cohorts'],source['current_geometry']['radius_km'],shell)
    close(inputs['basal_traction_at_vertices_pa'],traction,'external traction',rtol=2e-10,atol=1e-8)
    D,F=q.basal_drag_and_force(X,faces,mass,rho,g['cohorts'],inputs['basal_drag_pa_s_m'],traction[faces])
    close(after['mechanics']['elastic_energy_j'],totals['elastic_after_j'],'stored memory energy',rtol=5e-12,atol=1e4)
    if after['mechanics']['source_identity']!=before['mechanics']['state']['source_identity']:raise ValueError('Memory source identity')
    K0=float(.5*v0@M0@v0);K1=float(.5*v1@M@v1);birth=float(.5*v0@Min@v0)
    totals.update(kinetic_before_j=K0,kinetic_after_j=K1,birth_kinetic_inflow_j=birth,elastic_before_j=initial,
        damage_energy_change_j=damage_change,external_work_j=float(F@du),
        physical_drag_j=float(dt*v1@D@v1),numerical_kinetic_j=float(.5*(v1-v0)@M@(v1-v0)))
    for k,val in totals.items():
        small=k in ['kinetic_before_j','kinetic_after_j','birth_kinetic_inflow_j','numerical_kinetic_j']
        close(after['interval_ledger'][k],val,k,rtol=5e-10 if small else 3e-8,atol=1e-6 if small else 1e3)
    balance=K1+totals['elastic_after_j']-K0-initial-birth-totals['damage_energy_change_j']-totals['external_work_j']-totals['thermal_work_j']+sum(totals[k] for k in ['physical_drag_j','physical_maxwell_j','numerical_kinetic_j','numerical_memory_j'])
    residual=M@(v1-v0)/dt+D@v1+force-F
    impulse=float(np.max(np.abs(residual))/max(np.max(np.abs(F)),np.max(np.abs(force)),1.))
    relbalance=abs(balance)/max(abs(x) for x in totals.values())
    if impulse>1e-8 or relbalance>5e-10:raise ValueError('Independent BE momentum/energy balance')
    for k in ['physical_drag_j','physical_maxwell_j','numerical_kinetic_j','numerical_memory_j']:
        if totals[k]<0:raise ValueError('Negative dissipation')
    return dict(status='PASS',ledger=totals,balance_residual_j=balance,balance_relative=relbalance,
        coordinate_momentum_relative=impulse,max_memory_relative=maxmemory,born_cohorts=len(born))



def damage_delta(triangle,column_mass,rho,cohort,old0,old1,before_C,after_C):
    """Integrate the signed stiffness change before any global summation."""
    return q.coefficient_energy(triangle,column_mass,rho,cohort,old0,old1,np.asarray(after_C)-np.asarray(before_C))
