"""Real fixed-cut heat graph checks, with explicitly limited certification scope."""
import numpy as np
from analysis.genesis_coupled_split_control.thermal import fields
from tectonics.genesis_implicit_enthalpy import invert
from analysis.genesis_adaptive_control.metrics import assess as verified_assess,observe_packet


def thermal_check(packet,cuts,root,gp):
    th=packet['thermal'];s=packet['state'];g=s['geometry'];mass=np.array(th['mass_kg']);E=np.array(th['energy_j']);n=len(mass)
    cp,L,off=fields(n,gp);T=invert(E/mass,cp,L,np.full(n,gp.solidus_k),np.full(n,gp.liquidus_k),off)
    X=np.array(g['vertices_m'])+np.array(s['displacement_m']).reshape(-1,6)[:,:3]
    tri=X[np.array(g['faces'])];areas=.5*np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)
    shell=root['source_packet']['source']['parameters']['shell'];edges=[]
    for face,area in enumerate(areas):
        ids=[i for i,c in enumerate(cuts) if c['face']==face];dz=mass[ids]/(shell['density_kg_m3']*area);kA=shell['conductivity_w_m_k']*area
        edges.append((n-1,ids[0],2*kA/dz[0]))
        edges.extend((i,j,kA/(.5*(dz[k]+dz[k+1]))) for k,(i,j) in enumerate(zip(ids,ids[1:])))
        edges.append((ids[-1],n-2,2*kA/dz[-1]))
    flux=np.zeros(n);boundary_upper=np.zeros(n);upper=float(T.max())
    for i,j,G in edges:
        q=G*(T[j]-T[i]);flux[i]+=q;flux[j]-=q
        # Global maximum principle bounds conduction alone. At an old node's
        # solidus boundary the worst allowed neighbor is this global upper bound.
        boundary_upper[i]+=G*(upper-gp.solidus_k);boundary_upper[j]+=G*(upper-gp.solidus_k)
    old=np.array([c['material_id'] is not None for c in cuts]);idx=np.flatnonzero(old);new=np.flatnonzero(~old)
    capacity=mass*(cp+L/(gp.liquidus_k-gp.solidus_k)*((T>gp.solidus_k)&(T<gp.liquidus_k)))
    rates=flux/capacity*365.25*86400
    closest=int(idx[np.argmax(T[idx])]);damage=np.array(s['fault_history']['damage']);activation=root['source_packet']['source']['parameters']['faults']['activation_damage']
    unsupported=[]
    if np.any(T[idx]>=gp.solidus_k):unsupported.append('OLD_MATERIAL_AT_OR_ABOVE_SOLIDUS')
    if np.any(T[new]<=gp.solidus_k):unsupported.append('UNSUPPORTED_NEW_SOLID_COHORT')
    if T[-1]<647.096:unsupported.append('UNSUPPORTED_WATER_PHASE')
    if np.any(damage>=activation-1e-4) or any(s['fault_history']['fault_active']) or any(s['fault_history']['fault_candidate_age_myr']):unsupported.append('GATE_BOUNDARY')
    return dict(status='BLOCKED' if unsupported else 'UNCERTIFIED_DIAGNOSTIC',unsupported=unsupported,
        closest_old_cut=closest,closest_old_id=cuts[closest]['material_id'],old_margin_k=float(gp.solidus_k-T[closest]),
        initial_conduction_rate_at_closest_k_year=float(rates[closest]),maximum_old_conduction_rate_k_year=float(rates[idx].max()),
        conduction_only_global_upper_k=upper,old_boundary_flux_upper_w=float(boundary_upper[closest]),
        old_one_sided_conduction_bound=bool(np.all(boundary_upper[idx]<=0)),
        birth_margin_k=float(np.min(T[new]-gp.solidus_k)),water_margin_k=float(T[-1]-647.096),
        max_damage=float(damage.max()),damage_margin=float(activation-damage.max()),
        full_path_certified=False,
        limitation='Real graph maximum principle cannot bound old nodes below solidus; source/mechanical-feedback rates and stress-dependent damage lack full-path envelopes. Endpoint rates are diagnostics, not extrapolation bounds.')


def physical_displacement(p):
    s=p['state'];g=s['geometry'];q=np.array(s['displacement_m']).reshape(-1,6);faces=np.array(g['faces']);out=[]
    for c in g['cohorts']:
        v=q[faces[c['face_index']]]
        out.append([v[:,:3]+(lam-1)*v[:,3:] for lam in (c['outer_scale'],c['inner_scale'])])
    return np.array(out).reshape(len(out),6,3)


def observe(p,source,initial):
    o=observe_packet(p,source);o['displacement_increment_m']=physical_displacement(p)-physical_displacement(initial)
    return o


def assess(whole,half1,half2,h,c):
    r=verified_assess(whole,half1,half2,h,c)
    if r.get('fatal'):return r
    a=whole['displacement_increment_m'];b=half2['displacement_increment_m']
    delta=np.linalg.norm(a-b,axis=-1);scale=np.maximum(np.linalg.norm(a,axis=-1),np.linalg.norm(b,axis=-1))
    ratio=float(np.max(delta/(c['displacement_atol_m']+c['displacement_rtol']*scale)))
    global_ratio=float(np.sqrt(np.mean(delta**2))/(c['displacement_atol_m']+c['displacement_rtol']*np.sqrt(np.mean(scale**2))))
    r['checks']['displacement_m']=dict(local_ratio=ratio,global_ratio=global_ratio,max_difference=float(delta.max()),rms_difference=float(np.sqrt(np.mean(delta**2))))
    r['score']=max(r['score'],ratio,global_ratio)
    return r
