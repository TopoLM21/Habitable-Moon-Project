"""Bounded 80/320 load-only Maxwell series and independent saved restart."""
from copy import deepcopy
from pathlib import Path
import json,time
import numpy as np
from tectonics.genesis_transfer_bridge import encoded,digest
from .spherical_geometry_path import SphericalPath,YEAR,step
from .coupled_load_audit import rms
from .edge_spatial_audit import ROOT
OUT=Path('/workspace/genesis-spherical-geometry-01')


def main(load_field="parent_p1",angular_partition=None):
    global OUT
    suffix="" if load_field=="parent_p1" else "_analytic"
    if suffix:OUT=Path("/workspace/genesis-spherical-geometry-analytic-01")
    if angular_partition==2:
        suffix+="_common";OUT=Path("/workspace/genesis-spherical-geometry-analytic-common-01")
    if OUT.exists():raise FileExistsError('Preserve geometric starts')
    start=time.monotonic();OUT.mkdir();seeds={1:SphericalPath(1,5,load_field=load_field,angular_partition=angular_partition),2:SphericalPath(2,5,load_field=load_field,angular_partition=angular_partition)}
    audit={str(k):v.audit for k,v in seeds.items()}
    coarse,fine=seeds.values();V=coarse.volumes
    restricted=lambda s:np.einsum('fk,fkij->fij',fine.volumes.reshape(80,4),s.reshape(80,4,3,3))/V[:,None,None]
    force=dict(coarse_n=coarse.force.tolist(),fine_n=fine.force.tolist(),coarse_nm=coarse.torque.tolist(),fine_nm=fine.torque.tolist(),
        force_difference_n=float(np.linalg.norm(coarse.force-fine.force)),moment_difference_nm=float(np.linalg.norm(coarse.torque-fine.torque)))
    patch={}
    for label,a in [('coarse',coarse),('fine',fine)]:
        X=a.X;omega=np.array([.3,-.7,.5])*1e-12/a.audit['radius_m'];translation=np.array([.2,-.1,.4])*1e-12
        rot=np.tile(np.cross(omega,X)[:,None,:],(1,2,1)).reshape(-1);trans=np.zeros((len(X),6));trans[:,:3]=translation;trans=trans.ravel()
        # Exact spherical rigid solution with its inertial BE load; no stiffness work.
        F=a.M@rot/(.0625*YEAR)+a.D@rot
        r=step(a.M,a.D,F,np.zeros(a.nd),.0625*YEAR,a.blocks)
        patch[label]=dict(rotation_solution_relative=float(np.linalg.norm(r['velocity']-rot)/np.linalg.norm(rot)),
            rotation_load_work_relative=float(abs(a.F@rot-omega@a.torque)/max(abs(omega@a.torque),1.)),
            translation_load_work_relative=float(abs(a.F@trans-translation@a.force)/max(abs(translation@a.force),1.)),
            rotation_spurious_elastic_j=r['ledger']['elastic_after'],rotation_external_j=r['ledger']['external_work'],
            random_mass_positive=bool(np.random.default_rng(13).normal(size=a.nd)@a.M@np.random.default_rng(13).normal(size=a.nd)>0))
        assert patch[label]['rotation_solution_relative']<1e-6
        a.save(OUT/(label+'_initial.json'))
    runs={};fields={};deadline=time.monotonic()+90;restart=False
    for name,key,dt in [('coarse',1,.0625),('fine',2,.0625),('fine_half_step',2,.03125)]:
        a=deepcopy(seeds[key]);rows=[];trace={}
        for i in range(round(.125/dt)):
            if time.monotonic()>deadline:raise TimeoutError('90s solve budget')
            tick=time.monotonic();L=a.advance(dt)
            rows.append(dict(years=a.elapsed,seconds=time.monotonic()-tick,**L))
            trace[str(a.elapsed)]=a.stress()
            if name=='fine' and i==0:
                a.save(OUT/'fine_restart.json');resumed=SphericalPath.load(OUT/'fine_restart.json')
            elif name=='fine' and i==1:
                resumed.advance(dt);restart=encoded(a.state())==encoded(resumed.state());assert restart
        a.save(OUT/(name+'_final.json'));runs[name]=rows;fields[name]=trace
        print(json.dumps(dict(run=name,solve_seconds=sum(x['seconds'] for x in rows),stress_rms_pa=rms(a.stress(),a.volumes))),flush=True)
    comparisons={}
    for t in ('0.0625','0.125'):
        c=fields['coarse'][t];f=restricted(fields['fine'][t]);small=fields['fine_half_step'][t]
        comparisons[t]=dict(coarse_rms_pa=rms(c,V),fine_restricted_rms_pa=rms(f,V),
            absolute_difference_pa=rms(c-f,V),symmetric_relative=rms(c-f,V)/max(rms(c,V),rms(f,V),1e-12),
            temporal_absolute_pa=rms(fields['fine'][t]-small,fine.volumes),temporal_relative=rms(fields['fine'][t]-small,fine.volumes)/rms(small,fine.volumes))
    # Single quadrature check on coarse grid; no wider spatial refinement.
    q=SphericalPath(1,7,load_field=load_field,angular_partition=angular_partition);L=q.advance(.0625)
    quadrature=dict(force_relative=float(np.linalg.norm(q.F-coarse.F)/np.linalg.norm(q.F)),
        stress_absolute_pa=rms(q.stress()-fields['coarse']['0.0625'],V),stress_relative=rms(q.stress()-fields['coarse']['0.0625'],V)/rms(q.stress(),V))
    np.savez_compressed(OUT/'fields.npz',**{f'{name}_{t}':s for name,trace in fields.items() for t,s in trace.items()})
    result=dict(audit=audit,resultants=force,patch=patch,runs=runs,comparisons=comparisons,quadrature_check=quadrature,restart_exact=restart,
        seconds=time.monotonic()-start,output=str(OUT),files={p.name:dict(bytes=p.stat().st_size,sha256=digest(p.read_bytes())) for p in OUT.iterdir()})
    (ROOT/('spherical_geometry_check'+suffix+'.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('runs','files')},indent=2))

if __name__=='__main__':main()
