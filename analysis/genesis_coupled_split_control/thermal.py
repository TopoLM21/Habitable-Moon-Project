"""Fixed-material conduction and source-bound independent endpoint audit."""
import math
import numpy as np
from tectonics.genesis_implicit_enthalpy import solve,invert
from tectonics.genesis import SECONDS_PER_MYR
from analysis.genesis_implicit_cutcell_validation.run import seal


def fields(n,p):
    cp=np.full(n,p.silicate_heat_capacity_j_kg_k);L=np.full(n,p.silicate_latent_heat_j_kg);off=np.zeros(n)
    cp[-1]=p.water_heat_capacity_j_kg_k;L[-1]=0.;off[-1]=p.water_latent_heat_j_kg
    return cp,L,off


def phase_guard(cuts,T,p,initialT):
    if T[-1]<647.096:raise ValueError('UNSUPPORTED_WATER_PHASE')
    for i,c in enumerate(cuts):
        if c['material_id'] is not None and T[i]>=p.solidus_k:raise ValueError('UNSUPPORTED_OLD_MATERIAL_MELTING')
        if c['material_id'] is None and initialT[i]>p.solidus_k and T[i]<=p.solidus_k:
            raise ValueError('UNSUPPORTED_NEW_SOLID_COHORT: face '+str(c['face'])+' layer '+str(c['layer']))


def conduct(before,cuts,root,p,dt_myr):
    m=np.array(before['mass_kg']);E0=np.array(before['energy_j']);n=len(m);cp,L,off=fields(n,p)
    initialT=invert(E0/m,cp,L,np.full(n,p.solidus_k),np.full(n,p.liquidus_k),off)
    r=root['runtime_state'];g=r['geometry'];shell=root['source_packet']['source']['parameters']['shell']
    X=np.array(g['vertices_m'])+np.array(r['displacement_m']).reshape(-1,6)[:,:3];tri=X[np.array(g['faces'])];area=.5*np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)
    edges=[]
    for face,A in enumerate(area):
        ids=[i for i,c in enumerate(cuts) if c['face']==face];dz=m[ids]/(shell['density_kg_m3']*A);kA=shell['conductivity_w_m_k']*A
        edges.append((3361,ids[0],2*kA/dz[0]));edges.extend((i,j,kA/(.5*(dz[t]+dz[t+1]))) for t,(i,j) in enumerate(zip(ids,ids[1:])));edges.append((ids[-1],3360,2*kA/dz[-1]))
    dt=dt_myr*SECONDS_PER_MYR;result=solve(m,E0/m,cp,L,np.full(n,p.solidus_k),np.full(n,p.liquidus_k),off,edges,dt)
    T=result['temperature_k'];phase_guard(cuts,T,p,initialT)
    # Independent enthalpy/edge endpoint balance, not the nonlinear residual.
    h=cp*T+L*np.clip((T-p.solidus_k)/(p.liquidus_k-p.solidus_k),0,1)+off;E=m*h
    balance=E.astype(np.longdouble)-E0.astype(np.longdouble);delta=balance.copy();ulps=8*(np.spacing(E)+np.spacing(E0));allow=ulps+3e-11*m*cp;pairs=[]
    for i,j,G in edges:
        Q=dt*G*(T[i]-T[j]);balance[i]+=np.longdouble(Q);balance[j]-=np.longdouble(Q)
        a=16*dt*G*(math.ulp(float(T[i]))+math.ulp(float(T[j])));allow[i]+=a;allow[j]+=a;pairs.append([i,j,float(Q),float(-Q)])
    ratio=float(np.max(np.abs(balance)/allow));assert ratio<=1 and abs(delta.sum())<=ulps.sum()
    packet=seal(dict(time_myr=before['time_myr']+dt_myr,mass_kg=m.tolist(),energy_j=E.tolist(),temperature_k=T.tolist(),fixed_cut_geometry=cuts,paired_exchanges=pairs))
    audit=dict(status='PASS',node_balance_over_allowance=ratio,global_residual_j=float(delta.sum()),global_allowance_j=float(ulps.sum()),Newton_iterations=result['iterations'])
    return packet,audit
