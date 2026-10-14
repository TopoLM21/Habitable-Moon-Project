"""Conservative initial-data correction: fast drag velocities, zero memory edit.

The overdamped elastic/Maxwell spectrum does not certify a fast memory split.
Only mass-normalized velocity modes with drag rate >100 times the largest
elastic frequency are replaced by their frozen-force terminal velocity. This
also requires >100 inverse global shear-wave crossing times. There is no time
advance, heat, position, damage, phase, material or memory change.
"""
from copy import deepcopy
from dataclasses import replace
import time
import numpy as np
from scipy.linalg import cholesky, solve_triangular, eigh

from tectonics.genesis_transfer_bridge import encoded,digest
from tectonics.experimental_edge_weakening import scale
from .equilibrated_start import CompatibleProjection,rigid_modes,equilibrium_balance,point_stress
from .spherical_coupled import to_tensor

VERSION='spherical-fast-drag-velocity-only/v1'
SEPARATION=100.


def stress_inventory(a):
    """Recompute active stress thresholds, not a stale saved stress diagnostic."""
    from tectonics.genesis_shell import principal_tensile
    from tectonics.genesis_tides import TidalParameters,tidal_strain_cycle
    m=a.model;_,w,f,_,e=a.arrays();s=point_stress(a);tensor=to_tensor(s)
    eig=np.linalg.eigvalsh(tensor);tensile=np.maximum(eig[:,-1],0);shear=(eig[:,-1]-eig[:,0])/2
    world=np.zeros((m.nf,3,3));volume=np.zeros(m.nf)
    np.add.at(world,f,w[:,None,None]*tensor);np.add.at(volume,f,w);world/=volume[:,None,None]
    local=np.einsum('fai,fab,fbj->fij',a.frames,world,a.frames)
    plane_stress=np.stack((local[:,0,0],local[:,1,1],local[:,0,1]),axis=1)
    tp=replace(TidalParameters(**m.config['tidal_parameters']),eccentricity=m.state['orbit']['eccentricity'],semimajor_axis_km=m.state['orbit']['semimajor_axis_km'])
    nu=m.shell['poisson_ratio'];plane=np.array([[1.,nu,0.],[nu,1.,0.],[0.,0.,(1-nu)/2]])/(1-nu**2)
    tide=np.einsum('ij,pfj->pfi',plane,tidal_strain_cycle(m.mesh,tp))*m.shell['young_modulus_pa']*scale(m,m.state['damage'])[None,:,None]
    peaks=np.array([principal_tensile(plane_stress+t) for t in tide]).max(0)
    edges=[]
    for ff,gg,i,j in m.mesh.shared_edges:
        va,vb=m.mesh.vertices[[i,j]];n=(va+vb)/np.linalg.norm(va+vb);t=(vb-va)/np.linalg.norm(vb-va);normal=np.cross(t,n)
        traction=.5*(world[ff]+world[gg])@normal;edges.append(np.linalg.norm(traction-normal*(traction@normal)))
    edges=np.array(edges);total=float(a.base_energy()@scale(m,m.state['damage']))
    bulk=(m.C[f,0,0]+2*m.C[f,0,1])/3*scale(m,m.state['damage'])[f]
    volumetric=float(.5*np.sum(w*bulk*e[:,:3].sum(1)**2))
    return dict(elastic_j=total,volumetric_j=volumetric,deviatoric_j=total-volumetric,
                point_tensile_mean_pa=float(w@tensile/w.sum()),point_tensile_max_pa=float(tensile.max()),
                point_shear_mean_pa=float(w@shear/w.sum()),point_shear_max_pa=float(shear.max()),
                tidal_face_peak_max_pa=float(peaks.max()),faces_above_tensile=int((peaks>m.shell['tensile_strength_pa']).sum()),
                tensile_margin_min_pa=float(m.shell['tensile_strength_pa']-peaks.max()),
                edge_shear_max_pa=float(edges.max()),edges_above_shear=int((edges>m.config['experimental_geological_edge_fatigue']['shear_strength_pa']).sum()),
                shear_margin_min_pa=float(m.config['experimental_geological_edge_fatigue']['shear_strength_pa']-edges.max()))


def protected_state(a):
    state=deepcopy(a.model.state)
    state.pop('nodal_velocity');state.pop('omega')
    return digest(encoded(dict(config=a.model.config,state=state)))


def mass_whitener(M):
    diagonal=1/np.sqrt(np.diag(M))
    C=cholesky(M*diagonal[:,None]*diagonal[None,:],lower=True)
    return diagonal[:,None]*solve_triangular(C.T,np.eye(len(M)),lower=False)


def terminal_fast_modes(M,Dbar,Kbar,T,force,v0,crossing_seconds):
    """No tunable target dt or energy fraction. Cutoff from physical operators."""
    omega2=eigh((Kbar+Kbar.T)/2,eigvals_only=True)
    omega_max=float(np.sqrt(max(omega2[-1],0.)))
    rates,V=eigh((Dbar+Dbar.T)/2)
    cutoff=SEPARATION*max(omega_max,1/crossing_seconds)
    selected=rates>cutoff
    # z=T^-1 v, obtained through T^T M instead of a second ill-conditioned solve.
    z0=T.T@M@v0;rhs=T.T@force;c=V.T@z0;g=V.T@rhs
    target=c.copy();target[selected]=g[selected]/rates[selected]
    # Reconstruct the target directly: v0 + delta would cancel a large old fast
    # velocity against its correction and lose the small terminal component.
    v=T@(V@target)
    return v,dict(rates=rates,V=V,selected=selected,cutoff=cutoff,omega_max=omega_max,
                  original=c,target=target,force=g)


def prepare(a):
    """Read-only operator analysis. Returns proposed velocity and diagnostics."""
    tick=time.monotonic();m=a.model;M=a.material_mass();T=mass_whitener(M)
    L,w,f,_,e=a.arrays();H=w[:,None,None]*m.C[f]*scale(m,m.state['damage'])[f,None,None]
    op=CompatibleProjection(L,H,a.ids[f],M,rigid_modes(a.path.X,a.R))
    Kbar=T.T@op.K@T;basal=a.basal();Dbar=np.zeros_like(M)
    for face,row in enumerate(basal['rows']):
        B=(np.sqrt(a.beta*row['area'])[:,None,None]*row['tangent']).reshape(-1,18)@T[a.ids[face]]
        Dbar+=B.T@B
    force=basal['F']-a.internal_force();v0=np.array(m.state['nodal_velocity'])
    shear_speed=np.sqrt(m.shell['young_modulus_pa']/(2*(1+m.shell['poisson_ratio'])*a.rho))
    crossing=a.R/shear_speed
    v,data=terminal_fast_modes(M,Dbar,Kbar,T,force,v0,crossing)
    mask=data['selected'];V=data['V'];rates=data['rates'];cut=data['cutoff']
    # Independent dense assembly representation must select the same well-gapped
    # subspace. This checks numerical stability, not nonlinear physics accuracy.
    other=T.T@basal['D']@T;other=(other+other.T)/2
    rr,VV=eigh(other);mm=rr>cut
    projector_error=float(np.linalg.norm(V[:,mask]@V[:,mask].T-VV[:,mm]@VV[:,mm].T,2))
    white_error=float(np.linalg.norm(T.T@M@T-np.eye(len(M)),2))
    residual=a.drag_action(basal,v)-force;transformed=V.T@T.T@residual
    fast_relative=float(np.linalg.norm(transformed[mask])/max(np.linalg.norm(data['force'][mask]),1.))
    slow_change=V[:,~mask].T@T.T@M@(v-v0)
    fast_min=float(rates[mask].min()) if mask.any() else None
    slow_max=float(rates[~mask].max()) if (~mask).any() else None
    # Refuse an ambiguous cutoff rather than tuning it to discard more energy.
    if (not mask.any() or int(mm.sum())!=int(mask.sum()) or projector_error>1e-7
            or white_error>1e-7 or fast_relative>1e-7
            or fast_min/cut<10 or (slow_max is not None and slow_max>cut/10)):
        raise ValueError(f'VELOCITY_SPECTRAL_SEPARATION_UNRESOLVED fast={fast_min} slow={slow_max} cutoff={cut} projector={projector_error} whitening={white_error} residual={fast_relative}')
    acceleration= T@(T.T@(force-a.drag_action(basal,v)))
    before_acceleration=T@(T.T@(force-a.drag_action(basal,v0)))
    result=dict(version=VERSION,seconds=time.monotonic()-tick,method='Fast drag velocity terminal projection; all memory and positions exactly preserved',
        separation_ratio=SEPARATION,global_shear_crossing_seconds=float(crossing),max_elastic_frequency_per_s=data['omega_max'],
        cutoff_rate_per_s=cut,selected_modes=int(mask.sum()),preserved_velocity_modes=int((~mask).sum()),
        selected_rate_minmax_per_s=[fast_min,float(rates[mask].max())],excluded_rate_max_per_s=slow_max,
        selected_decay_time_max_s=1/fast_min,projector_dense_factored_error=projector_error,mass_whitening_error=white_error,
        fast_force_relative=fast_relative,slow_velocity_mass_norm_change=float(np.linalg.norm(slow_change)),
        generalized_force_residual_norm_n=float(np.linalg.norm(residual)),
        surface_acceleration_before_max_m_s2=float(np.linalg.norm(before_acceleration.reshape(-1,6)[:,:3],axis=1).max()),
        surface_acceleration_after_max_m_s2=float(np.linalg.norm(acceleration.reshape(-1,6)[:,:3],axis=1).max()),
        equilibrium_claim=False,memory_spectral_filter=False,thermal_heat_j=0.,elastic_change_j=0.,time_advanced_years=0.)
    return v,result


def initialize(a):
    if 'equilibrated_start' in a.origin:
        raise ValueError('WEAKENED_SCENARIO_NOT_A_PRIMARY_SOURCE')
    if 'velocity_only_start' in a.origin:
        raise ValueError('ALREADY_VELOCITY_PREPARED')
    before=deepcopy(a.model.state);old_origin=deepcopy(a.origin);protected=protected_state(a)
    M=a.material_mass();J=rigid_modes(a.path.X,a.R);v0=np.array(before['nodal_velocity'])
    balance0=equilibrium_balance(a)
    try:
        v,ledger=prepare(a)
        a.model.state['nodal_velocity']=v.tolist()
        a.model.state['omega']=[(np.linalg.solve(J[:,3:].T@M@J[:,3:],J[:,3:].T@M@v)/a.R).tolist()]
        if protected_state(a)!=protected:raise ValueError('VELOCITY_INIT_PROTECTED_STATE_CHANGED')
        def momenta(x):
            p=J.T@M@x
            return dict(linear_kg_m_s=p[:3].tolist(),angular_kg_m2_s=(a.R*p[3:]).tolist())
        K0=float(.5*v0@M@v0);K1=float(.5*v@M@v)
        ledger.update(protected_sha256=protected,kinetic_before_j=K0,kinetic_after_j=K1,
                      signed_initialization_exchange_j=K1-K0,momentum_before=momenta(v0),momentum_after=momenta(v),momentum_change=momenta(v-v0),
                      balance_before=balance0,balance_after=equilibrium_balance(a))
        a.origin['velocity_only_start']=ledger;a.validate();a.model.validate()
        return ledger
    except Exception:
        a.model.state=before;a.origin=old_origin;raise
