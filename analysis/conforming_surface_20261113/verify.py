"""Compact shared-surface controls. Existing states are always read-only."""
from dataclasses import replace
from functools import lru_cache
import hashlib,json,math,resource,subprocess,sys,time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tectonics.conforming_surface import ConformingSurface,SurfaceParameters,SurfaceControl
from tectonics.mesh import build_icosphere
from tectonics.checkpoint import load_checkpoint
from tectonics.topology import PlateTopologyManager,PlateTopologyParameters
from tectonics.regional_area import AreaMaterialInventory,ActiveAreaParameters
from tectonics.regional_material import RegionalMaterialParameters
from tectonics.kinematics import angular_velocity_vectors
HERE=Path(__file__).resolve().parent
NATIVE=ROOT/'analysis/regional_inventory_20261113/saved_states/initial_same_g2'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


@lru_cache(None)
def fine_fields(scenario):
    mesh=build_icosphere(5);p=SurfaceParameters();area=mesh.areas_unit_sphere*p.radius_m**2
    axis=np.array([0.,0.,1.]) if scenario=='smooth' else np.array([1.,2.,3.])/np.sqrt(14)
    amplitude=.1 if scenario=='smooth' else .25
    h=20000*(1+amplitude*(3*(mesh.centroids@axis)**2-1)/2)
    volume=area*(h+h*h/p.radius_m+h**3/(3*p.radius_m**2))
    return volume,axis


def synthetic(scenario,sub,params,control):
    mesh=build_icosphere(sub);fine,axis=fine_fields(scenario)
    volume=np.array([math.fsum(v) for v in fine.reshape(mesh.cell_count,-1)])
    zero=np.zeros(mesh.cell_count);owner=(mesh.centroids[:,0]<0).astype(int)+2*(mesh.centroids[:,1]<0)+4*(mesh.centroids[:,2]<0)
    arrays=dict(owner=owner,felsic_fraction=zero,felsic_volume_m3=zero,basalt_volume_m3=volume,
        felsic_age_moment_m3_myr=zero,basalt_age_moment_m3_myr=volume*40,potential=zero)
    obj=ConformingSurface(mesh,arrays,params,control,provenance=dict(scenario=scenario,common_reference_subdivision=5,
        initialization='conservative aggregation of same analytic profile on common fine quadrature',axis=axis.tolist()))
    return obj


def native(params,control):
    cp=load_checkpoint(NATIVE,PlateTopologyManager(PlateTopologyParameters()))
    d=cp.regional_material
    inv=AreaMaterialInventory.restore(d,RegionalMaterialParameters(**d['parameters']),ActiveAreaParameters(**d['area_parameters']))
    obj=ConformingSurface.from_area_inventory(inv,params,control,cp.material_exchange,dict(
        source_files={str((NATIVE/n).relative_to(ROOT)):sha(NATIVE/n) for n in ('meta.json','state.npz')}))
    return obj,cp,inv


def sampled_coverage(obj,samples=2048):
    k=np.arange(samples);z=1-2*(k+.5)/samples;phi=k*math.pi*(3-math.sqrt(5))
    points=np.c_[np.sqrt(1-z*z)*np.cos(phi),np.sqrt(1-z*z)*np.sin(phi),z]
    count=np.zeros(samples,int);tri=obj.vertices[obj.faces]
    normals=np.stack([np.cross(tri[:,i],tri[:,(i+1)%3]) for i in range(3)],axis=1)
    for first in range(0,len(tri),128):
        dots=np.einsum('fkj,pj->fkp',normals[first:first+128],points)
        count+=np.sum(np.all(dots>=-1e-13,axis=1),axis=0)
    return dict(samples=samples,uncovered=int(np.sum(count==0)),multiply_covered=int(np.sum(count>1))),points


def metrics(obj):
    b=obj.budget();e=obj.evaluate(obj.vertices);a=e['area_m2'];centers=obj.mesh().centroids
    axis=np.array(obj.data['provenance'].get('axis',[0.,0.,1.]));y=(3*(centers@axis)**2-1)/2
    b.update(height_l2_m=float(np.sum(a*y*e['height_m'][:,1])/np.sum(a*y*y)),
        area_rms_change_fraction=float(np.sqrt(np.sum(obj.reference_area*(a/obj.reference_area-1)**2)/np.sum(obj.reference_area))),
        min_area_fraction=float(np.min(a/obj.reference_area)),
        relative_energy_error_on_exchange=abs(b['energy_residual_j'])/max(abs(b['heat_generated_j']),abs(b['external_work_j']),1.),
        accepted_steps=obj.data['accepted_steps'],rejected_steps=obj.data['rejected_steps'])
    return b


def run(name,scenario='smooth',sub=2,dt=1.,end=100.,beta=1e17,adaptive=False,resume=None):
    output=HERE/f'{name}.json';saved=HERE/'states'/name
    if output.exists() or saved.exists():raise FileExistsError(name)
    begin=time.perf_counter();params=replace(SurfaceParameters(),basal_drag_pa_s_m=beta)
    control=replace(SurfaceControl(),max_step_myr=dt,max_trials=2000)
    obj=ConformingSurface.load(HERE/'states'/resume) if resume else (native(params,control)[0] if scenario=='native' else synthetic(scenario,sub,params,control))
    initial=obj.checkpoint();initial_budget=metrics(obj);start=time.perf_counter();status='complete';failure=None
    try:
        if adaptive:obj.advance(end)
        else:
            while obj.data['time_myr']<end:obj.step_fixed(min(dt,end-obj.data['time_myr']))
    except (ValueError,RuntimeError) as exc:status='validity_or_controller_limit';failure=str(exc)
    elapsed=time.perf_counter()-start;obj.save(saved)
    checks={k:obj.arrays[k].tolist()==initial['arrays'][k] for k in ('owner','felsic_fraction','felsic_volume_m3','basalt_volume_m3','potential')}
    assert all(checks.values())
    restored=ConformingSurface.load(saved);assert restored.checkpoint()==obj.checkpoint()
    report=dict(case=name,scenario=scenario,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        runtime_sha256=sha(ROOT/'tectonics/conforming_surface.py'),status=status,failure=failure,requested_end_myr=end,
        fixed_step_myr=None if adaptive else dt,parameters=initial['parameters'],control=initial['control'],
        initial=initial_budget,final=metrics(obj),material_unchanged=checks,native_restore_equal=True,
        coverage=sampled_coverage(obj)[0],integration_wall_s=elapsed,wall_s=time.perf_counter()-begin,
        process_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        snapshot_bytes=sum(p.stat().st_size for p in saved.iterdir()),scope='new welded fixed-base gravity/drag control; no cooling, production, recycling, slip or fracture')
    with output.open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(dict(case=name,status=status,end_myr=obj.data['time_myr'],wall_s=report['wall_s'],steps=obj.data['accepted_steps'],energy_error=report['final']['relative_energy_error_on_exchange'])),flush=True)


def kinematics():
    name='prescribed_native20';path=HERE/f'{name}.json'
    if path.exists() or (HERE/'states'/name).exists():raise FileExistsError(name)
    obj,cp,inv=native(SurfaceParameters(),SurfaceControl());omega=angular_velocity_vectors(cp.system)
    begin=time.perf_counter();initial=metrics(obj)
    for _ in range(20):obj.prescribed_step(omega,1.)
    shared,points=sampled_coverage(obj)
    count=np.zeros(len(points),int)
    for owner,w in enumerate(omega):
        reference=Rotation.from_rotvec(w*20).inv().apply(points)
        root=inv.receiver.locate(reference);count+=inv.coarse.arrays['owner'][root]==owner
    old=dict(samples=len(points),uncovered=int(np.sum(count==0)),multiply_covered=int(np.sum(count>1)))
    obj.save(HERE/'states'/name)
    report=dict(case=name,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        runtime_sha256=sha(ROOT/'tectonics/conforming_surface.py'),initial=initial,final=metrics(obj),
        shared_coverage=shared,independent_rigid_coverage=old,wall_s=time.perf_counter()-begin,
        input_omega_rad_myr=omega.tolist(),scope='same imposed initial Euler rotations over 20 Myr; welded projection changes velocities and explicitly accounts external work; not the previous dynamically driven 20 Myr history')
    with path.open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    mode=sys.argv[1]
    if mode=='smooth':
        for sub in (1,2,3):
            for dt in (2.,1.,.5):run(f'smooth_g{sub}_dt{dt:g}',sub=sub,dt=dt)
    elif mode=='oblique':
        for sub in (1,2,3):run(f'oblique_g{sub}',scenario='oblique',sub=sub)
    elif mode=='native':
        run('native20',scenario='native',end=20.,adaptive=True)
        run('native100',scenario='native',end=100.,adaptive=True)
        run('native100_dt_half',scenario='native',dt=.5,end=100.,adaptive=True)
    elif mode=='restart':
        # Stop at an actual accepted time of the reference, so restart is not
        # confused with changing the controller's prescribed endpoint.
        ref=ConformingSurface.load(HERE/'states/native100')
        stop=next(h['time_myr'] for h in ref.data['history'] if h['time_myr']>=50.)
        run('native50',scenario='native',end=stop,adaptive=True)
        run('native_resume100',scenario='native',end=100.,adaptive=True,resume='native50')
        resumed=ConformingSurface.load(HERE/'states/native_resume100')
        assert ref.checkpoint()==resumed.checkpoint()
        with (HERE/'restart_comparison.json').open('x') as f:json.dump(dict(equal_arrays_and_metadata=True,stop_myr=stop,end_myr=100.),f,indent=2)
    elif mode=='sensitivity':
        for beta in (1e16,1e18):run(f'native_beta{beta:g}',scenario='native',end=100.,beta=beta,adaptive=True)
    elif mode=='kinematics':kinematics()
    else:raise ValueError(mode)
