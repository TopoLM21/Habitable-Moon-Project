"""Frozen directional damage tangent along existing contrast; no advance."""
import hashlib,json,time
from pathlib import Path
from dataclasses import replace
import numpy as np
from tectonics.simplified_genesis import SimplifiedGenesis
from tectonics.simplified_genesis_nodal import mechanical,material_mass
from tectonics.simplified_genesis_basal import assemble
from tectonics.genesis_tides import TidalParameters,tidal_strain_cycle
from tectonics.genesis_shell import principal_tensile
from tectonics.genesis_transfer_bridge import encoded
from analysis.genesis_adaptive_control.controller import time_limit


def stats(x):
    x=np.asarray(x);return dict(min=float(x.min()),max=float(x.max()),mean=float(x.mean()))


def verify(result):
    for state in result['states']:
        a,b=state['probes'];error=abs(a['net_rate_projection_per_myr']-b['net_rate_projection_per_myr'])/max(abs(b['net_rate_projection_per_myr']),1e-12)
        assert error<1e-4
        state['finite_difference_relative_disagreement']=error
        state['finite_difference_consistent']=True
        if state['damage_drive_per_myr']['max']==0:
            healing=state['healing_per_myr']['mean']
            assert abs(b['net_rate_projection_per_myr']+healing)<1e-8
            assert abs(b['frozen_map_contrast_gain']-np.exp(-healing*state['frozen_response_years']/1e6))<1e-8
            state['zero_drive_healing_identity_verified']=True
    return result


def probe(path):
    p=Path(path);blob=p.read_bytes();m=SimplifiedGenesis.load(p);before=encoded(m.state)
    T=m.temperature(m.state['energy']);D=np.asarray(m.state['damage']);M=material_mass(m);basal=assemble(m)
    weights=m.mesh.areas_unit_sphere;weights=weights/weights.sum()
    center=lambda x:x-weights@x
    contrast=center(D);norm2=float(weights@(contrast*contrast))
    if norm2<=0:raise ValueError('No inherited spatial contrast to test')
    years=m.config['max_step_years'];dt=years*31557600.;dm=years/1e6
    orbit=m.state['orbit'];tp=replace(TidalParameters(**m.config['tidal_parameters']),eccentricity=orbit['eccentricity'],semimajor_axis_km=orbit['semimajor_axis_km'])
    nu=m.shell['poisson_ratio'];plane=np.array([[1.,nu,0.],[nu,1.,0.],[0.,0.,(1-nu)/2]])/(1-nu**2)
    cycle=np.einsum('ij,pfj->pfi',plane,tidal_strain_cycle(m.mesh,tp))*m.shell['young_modulus_pa']
    hot=np.clip((T[m.solid_nodes].mean(axis=1)-700)/650,0,1)
    heal=1/m.shell['cold_healing_timescale_myr']+hot**4/m.shell['hot_healing_timescale_myr']
    def evaluate(d):
        if np.any(d<0) or np.any(d>1):raise ValueError('Probe would leave physical damage bounds')
        r=mechanical(m,T,d,T,dt,M,basal)
        tensile=np.array([principal_tensile(r['stress']+z*m.scale(d)[:,None]) for z in cycle])
        raw=np.mean(np.maximum(tensile/m.shell['tensile_strength_pa']-1,0)**2,axis=0)/m.shell['damage_timescale_myr']
        load=m.smoother.apply(raw,np.ones(m.nf,bool));total=load+heal
        rate=(1-d)*load-heal*d
        endpoint=load/total+(d-load/total)*np.exp(-total*dm)
        return dict(load=load,rate=rate,endpoint=endpoint,tensile=tensile,momentum=r['ledger']['momentum_relative'])
    base=evaluate(D);results=[]
    for epsilon in (1e-3,5e-4):
        plus=evaluate(D+epsilon*contrast);minus=evaluate(D-epsilon*contrast)
        derivative={k:(plus[k]-minus[k])/(2*epsilon) for k in ('load','rate','endpoint')}
        results.append(dict(epsilon=epsilon,max_absolute_damage_probe=float(epsilon*abs(contrast).max()),
            drive_projection_per_myr=float(weights@(contrast*center(derivative['load']))/norm2),
            net_rate_projection_per_myr=float(weights@(contrast*center(derivative['rate']))/norm2),
            frozen_map_contrast_gain=float(np.sqrt(weights@(center(derivative['endpoint'])**2)/norm2)),
            plus_overstress_faces=int(np.any(plus['tensile']>m.shell['tensile_strength_pa'],axis=0).sum()),minus_overstress_faces=int(np.any(minus['tensile']>m.shell['tensile_strength_pa'],axis=0).sum())))
    assert encoded(m.state)==before and p.read_bytes()==blob
    return dict(path=str(path),sha256=hashlib.sha256(blob).hexdigest(),time_myr=m.state['time_myr'],frozen_response_years=years,
        scope='Existing contrast only; T, memory, geometry, M, D_basal, F, orbital phase cycle fixed; no cooling increment or planetary step.',
        damage=stats(D),inherited_contrast_weighted_rms=float(np.sqrt(norm2)),tensile_pa=stats(base['tensile']),overstress_faces=int(np.any(base['tensile']>m.shell['tensile_strength_pa'],axis=0).sum()),
        damage_drive_per_myr=stats(base['load']),healing_per_myr=stats(heal),base_net_rate_per_myr=stats(base['rate']),
        base_contrast_rate_projection_per_myr=float(weights@(contrast*center(base['rate']))/norm2),
        probes=results,source_and_state_unchanged=True,
        limitation='Directional tangent is not a full eigenvalue spectrum, thermal-feedback test or a proof of absence of future localization.')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args();start=time.monotonic()
    with time_limit(110):
        states=[probe(path) for path in ['/workspace/genesis-moving-basal-source-01.json','/workspace/genesis-moving-basal-pilot-01/checkpoint.json']]
    result=verify(dict(status='COMPLETED_WITHOUT_EVOLUTION',seconds=time.monotonic()-start,states=states))
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
