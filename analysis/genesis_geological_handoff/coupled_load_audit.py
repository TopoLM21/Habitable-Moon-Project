"""Paired-load diagnostics; no changes to production assembly or stored sources."""
from copy import deepcopy
from contextlib import contextmanager
from pathlib import Path
import json,time
import numpy as np
from tectonics import simplified_genesis_basal as basal
from tectonics.basal_coupling import mantle_source_pattern,basal_coupling_fraction
from tectonics.experimental_angular_coupled import AngularCoupled
from tectonics.genesis_transfer_bridge import digest
from .coupled_spatial_check import field,restrict_fields,compare
from .edge_spatial_audit import ROOT

OUT=Path('/workspace/genesis-coupled-load-audit-03')
SOURCE=Path('/workspace/genesis-angular-coupled-01/initial.json')
FINE=Path('/workspace/genesis-coupled-spatial-01/fine_initial.json')


def vertex_traction(m):
    g=basal.geometry(m);num=np.zeros(m.mesh.vertex_count);den=num.copy()
    c=basal_coupling_fraction(g['depth']/1000,m.shell['traction_coupling_depth_km'])
    np.add.at(num,m.mesh.faces.ravel(),np.repeat(g['area']*c,3))
    np.add.at(den,m.mesh.faces.ravel(),np.repeat(g['area'],3))
    return m.shell['convective_traction_pa']*(num/den)[:,None]*mantle_source_pattern(m.mesh,m.shell['seed']),num/den


def prolongation(coarse,fine):
    P=np.zeros((fine.vertex_count,coarse.vertex_count));P[:coarse.vertex_count]=np.eye(coarse.vertex_count)
    for _,_,i,j in coarse.shared_edges:
        x=coarse.vertices[i]+coarse.vertices[j];x/=np.linalg.norm(x)
        k=np.argmin(np.linalg.norm(fine.vertices-x,axis=1));P[k,i]=P[k,j]=.5
    assert np.all(P.sum(axis=1)==1)
    return P


def with_traction(m,traction):
    """Same current geometry/drag; explicit prescribed P1 reference traction."""
    b=ORIGINAL(m);F=np.zeros_like(b['F'])
    for f,idx in enumerate(b['dofs']):
        old=b['faces'][f]
        row=basal.facet(old['bottom'],old['h'],old['beta'],traction[m.mesh.faces[f]],old['vm'])
        assert np.array_equal(row['D'],old['D'])
        b['faces'][f]=row;np.add.at(F,idx,row['Ft'])
    b['Ft']=F;b['F']=F+b['Fm'];return b

ORIGINAL=basal.assemble

@contextmanager
def prescribed(traction):
    # Process-local analysis override, always restored. No altered-law checkpoints.
    basal.assemble=lambda m:with_traction(m,traction)
    try:yield
    finally:basal.assemble=ORIGINAL


def load_summary(b):
    force=np.zeros(3);moment=np.zeros(3);force_scale=moment_scale=0.
    for r in b['faces']:
        nodal=r['Ft'].reshape(3,6)[:,:3]
        torques=np.cross(r['bottom'],nodal)
        force+=nodal.sum(0);moment+=torques.sum(0)
        force_scale+=np.linalg.norm(nodal,axis=1).sum();moment_scale+=np.linalg.norm(torques,axis=1).sum()
    return dict(force_n=force.tolist(),moment_nm=moment.tolist(),sum_nodal_force_norm_n=force_scale,sum_nodal_moment_norm_nm=moment_scale,
        generalized_force_euclidean_n=float(np.linalg.norm(b['F'])))


def rms(s,V):return float(np.sqrt(np.einsum('f,fij,fij->',V,s,s)/V.sum()))


def main():
    if OUT.exists():raise FileExistsError('Preserve existing audit')
    start=time.monotonic();OUT.mkdir();hashes={str(p):digest(p.read_bytes()) for p in (SOURCE,FINE)}
    a=AngularCoupled.load(SOURCE);b=AngularCoupled.load(FINE);P=prolongation(a.model.mesh,b.model.mesh);P6=np.kron(P,np.eye(6))
    tc,cc=vertex_traction(a.model);tf,cf=vertex_traction(b.model);shared=P@tc
    bc=ORIGINAL(a.model);bf=ORIGINAL(b.model);bs=with_traction(b.model,shared)
    # Isolate vertex smoothing from resampling the analytic spherical pattern.
    patternfine=mantle_source_pattern(b.model.mesh,b.model.shell['seed'])
    pattern_only=b.model.shell['convective_traction_pa']*(P@cc)[:,None]*patternfine
    bp=with_traction(b.model,pattern_only)
    norm=lambda x:float(np.linalg.norm(x))
    diagnostics=dict(coarse=load_summary(bc),fine_native=load_summary(bf),fine_shared=load_summary(bs),
        native_restriction_relative=norm(P6.T@bf['F']-bc['F'])/norm(bc['F']),
        shared_restriction_relative=norm(P6.T@bs['F']-bc['F'])/norm(bc['F']),
        pattern_resampling_component_relative=norm(P6.T@(bp['F']-bs['F']))/norm(bc['F']),
        coupling_resmoothing_component_relative=norm(P6.T@(bf['F']-bp['F']))/norm(bc['F']),
        coupling_max_difference=float(abs(cf-P@cc).max()),
        note='Generalized Euclidean norm after dual restriction, not net force or pointwise traction norm; components need not add as norms.')
    for label in ('fine_native','fine_shared'):
        d=diagnostics[label];c=diagnostics['coarse']
        d['force_difference_n']=norm(np.array(d['force_n'])-c['force_n'])
        d['moment_difference_nm']=norm(np.array(d['moment_nm'])-c['moment_nm'])
        d['force_difference_over_absolute_sum']=d['force_difference_n']/c['sum_nodal_force_norm_n']
        d['moment_difference_over_absolute_sum']=d['moment_difference_nm']/c['sum_nodal_moment_norm_nm']
    assert diagnostics['shared_restriction_relative']<1e-10
    print(json.dumps(dict(load_diagnostics=diagnostics)),flush=True)
    fields={};runs={};deadline=time.monotonic()+120
    for name,seed,traction in [('coarse_shared',a,tc),('fine_shared',b,shared),('coarse_half',a,.5*tc),('fine_half',b,.5*shared),('fine_native_half',b,.5*tf)]:
        model=deepcopy(seed);trace={};rows=[]
        print(json.dumps(dict(start_run=name)),flush=True)
        with prescribed(traction):
            for i in range(2):
                if time.monotonic()>deadline:raise TimeoutError('120second solve budget')
                tick=time.monotonic();L=model.advance(.0625);m=model.model
                rows.append(dict(seconds=time.monotonic()-tick,whole_allowance_fraction=abs(L['whole_balance_j'])/L['whole_allowance_j'],
                    external_work_j=L['external_work'],births=len(m.state.get('born_material',[])),certificate=m.state['experimental_edge_fatigue']['reason']))
                assert rows[-1]['whole_allowance_fraction']<=1 and rows[-1]['certificate']=='NOT_SEPARATING'
                trace[str((i+1)*.0625)]=field(m)
        runs[name]=rows;fields[name]=trace
        np.savez_compressed(OUT/(name+'.npz'),**{f'{t}_{k}':v for t,f in trace.items() for k,v in f.items()})
    saved=np.load('/workspace/genesis-coupled-spatial-01/fields.npz')
    for name in ('coarse_full','fine_full','fine_alternative'):
        fields[name]={t:{k:saved[f'{name}_{t}_{k}'] for k in ('stress','T','H','oldD','faceD')} for t in ('0.0625','0.125')}
    V=a.model.G[:,0,0];comparisons={}
    for t in ('0.0625','0.125'):
        cc=fields['coarse_shared'][t];ff=restrict_fields(fields['fine_shared'][t]);cz=fields['coarse_half'][t];fz=restrict_fields(fields['fine_half'][t])
        cn=fields['coarse_full'][t];fn=restrict_fields(fields['fine_full'][t]);ca=fields['fine_alternative'][t]
        response_c=2*(cc['stress']-cz['stress']);response_f=2*(ff['stress']-fz['stress'])
        native_c=2*(cn['stress']-cz['stress']);native_f=2*(fn['stress']-restrict_fields(fields['fine_native_half'][t])['stress'])
        comparisons[t]=dict(shared_load_total=compare(cc,ff,V),half_load_total=compare(cz,fz,V),
            shared_load_response_rms_pa_coarse=rms(response_c,V),shared_load_response_rms_pa_fine=rms(response_f,V),
            shared_load_response_difference_relative=rms(response_c-response_f,V)/rms(response_f,V),
            native_load_response_difference_relative=rms(native_c-native_f,V)/rms(native_f,V),
            native_load_response_rms_pa_fine=rms(native_f,V),
            full_stress_rms_pa_fine=rms(fn['stress'],V),
            half_load_stress_rms_pa_fine=rms(fz['stress'],V),
            fine_load_sampling_effect_rms_pa=rms(fn['stress']-ff['stress'],V),
            coarse_freezing_effect_rms_pa=rms(cn['stress']-cc['stress'],V),
            initialization_effect_rms_pa=rms(ca['stress']-fields['fine_full'][t]['stress'],b.model.G[:,0,0]),
            native_total_grid_difference_rms_pa=rms(cn['stress']-fn['stress'],V))
    np.savez_compressed(OUT/'fields.npz',**{f'{name}_{t}_{k}':v for name,trace in fields.items() for t,f in trace.items() for k,v in f.items()})
    assert all(digest(Path(p).read_bytes())==h for p,h in hashes.items())
    result=dict(code_base='3f24fd1efc5c1f988ba90e4864399bda1d93ae9c',source_hashes=hashes,load_diagnostics=diagnostics,runs=runs,comparisons=comparisons,
        seconds=time.monotonic()-start,output=str(OUT),fields_sha256=digest((OUT/'fields.npz').read_bytes()),
        scope='Frozen initial P1 transmitted traction and half-traction paired controls; reported response is twice full-minus-half, not a zero-memory evolution; no altered-law trajectory checkpoints; native physical amplitudes unchanged')
    (ROOT/'coupled_load_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(seconds=result['seconds'],comparisons=comparisons)),flush=True)

if __name__=='__main__':main()
