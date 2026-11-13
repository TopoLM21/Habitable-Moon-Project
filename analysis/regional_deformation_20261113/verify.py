"""Bounded reproducible comparisons; never overwrite an earlier result."""
from dataclasses import asdict,replace
import hashlib,importlib.util,json,math,resource,subprocess,sys,time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tectonics.conforming_surface import SurfaceParameters,SurfaceControl
from tectonics.regional_deformation import DeformingSurface,DilationParameters
from tectonics.deformation_exchange import DeformationExchange,CoupledParameters
from tectonics.material_exchange import MaterialExchangeBudget,MaterialExchangeParameters
from tectonics.thermal import ThermalParameters
from tectonics.tides import constant_eccentricity

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('prior_shared_controls',ROOT/'analysis/conforming_surface_20261113/verify.py')
prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)


def hashes():
    return {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
        ('tectonics/regional_deformation.py','tectonics/deformation_exchange.py',
         'tectonics/conforming_surface.py','tectonics/material_exchange.py')}


def setup(scenario,sub,length,dt):
    params=SurfaceParameters();control=replace(SurfaceControl(),max_step_myr=dt,max_trials=2000)
    if scenario=='native':
        base,cp,_=prior.native(params,control,sub)
        surface=DeformingSurface.from_surface(base,DilationParameters(length))
        material=MaterialExchangeBudget.restore(cp.material_exchange,MaterialExchangeParameters(**cp.material_exchange['options']))
        return DeformationExchange(surface,material,cp.thermal,ThermalParameters(),constant_eccentricity(0.),
                                   CoupledParameters(step_myr=dt))
    base=prior.synthetic(scenario,sub,params,control)
    return DeformingSurface.from_surface(base,DilationParameters(length))


def strain_summary(surface):
    e=surface.evaluate(surface.vertices);rate=e['area_rate_m2_s']/e['area_m2']
    return dict(dissipation_w=e['dissipation_w'],basal_dissipation_w=e['basal_dissipation_w'],
        dilation_dissipation_w=e['dilation_dissipation_w'],power_residual_w=e['power_residual_w'],
        linear_relative_residual=e['linear_relative_residual'],
        positive_area_rate_m2_s=math.fsum(np.maximum(e['area_rate_m2_s'],0)),
        negative_area_rate_m2_s=math.fsum(np.minimum(e['area_rate_m2_s'],0)),
        area_strain_rate_rms_per_s=float(np.sqrt(np.dot(e['area_m2'],rate*rate)/sum(e['area_m2']))),
        speed_rms_mm_yr=float(np.sqrt(np.dot(e['weights_m2'],np.sum(e['velocity_per_s']**2,axis=1))/sum(e['weights_m2']))*surface.params.radius_m*31557600*1e3))


def run(name,scenario='native',sub=3,length=1e6,dt=1.,end=20.,resume=None):
    path=HERE/f'{name}.json';saved=HERE/'states'/name
    if path.exists() or saved.exists():raise FileExistsError(name)
    started=time.perf_counter()
    obj=(DeformationExchange.load(HERE/'states'/resume) if resume else setup(scenario,sub,length,dt))
    surface=obj.surface if isinstance(obj,DeformationExchange) else obj
    initial=obj.diagnostics() if isinstance(obj,DeformationExchange) else obj.budget()
    initial_strain=strain_summary(surface);initial_state=obj.checkpoint()
    then=time.perf_counter();status='complete';failure=None
    try:
        if isinstance(obj,DeformationExchange):obj.advance(end)
        else:
            while obj.data['time_myr']<end:obj.step_fixed(min(dt,end-obj.data['time_myr']))
    except (ValueError,RuntimeError) as exc:status='stopped';failure=str(exc)
    integration=time.perf_counter()-then
    surface=obj.surface if isinstance(obj,DeformationExchange) else obj
    obj.save(saved)
    restored=type(obj).load(saved);assert restored.checkpoint()==obj.checkpoint()
    final=obj.diagnostics() if isinstance(obj,DeformationExchange) else obj.budget()
    coverage=prior.sampled_coverage(surface)[0]
    report=dict(case=name,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        runtime_sha256=hashes(),status=status,failure=failure,scenario=scenario,cells=len(surface.faces),
        length_m=length,dt_myr=dt,requested_end_myr=end,initial=initial,final=final,
        initial_strain=initial_strain,final_strain=strain_summary(surface),coverage=coverage,
        exact_native_restart=True,integration_wall_s=integration,wall_s=time.perf_counter()-started,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        parameters=asdict(surface.params),control=asdict(surface.control),
        snapshot_bytes=sum(p.stat().st_size for p in saved.iterdir()),
        scope='dilation/drag gravity control; native case additionally uses existing sensible-heat reservoirs; no automatic creation, recycling, slip or fracture')
    if isinstance(obj,DeformationExchange):
        for key in ('surface_volumes_km3','origin_hi','origin_lo'):
            assert obj.material.data[key]==initial_state['material'][key]
        report['thermal_parameters']=asdict(obj.thermal_params)
        report['coupling_parameters']=asdict(obj.options)
    with path.open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps({k:report[k] for k in ('case','status','failure','wall_s','integration_wall_s')}),flush=True)
    return report


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('--scenario',default='native',choices=['native','smooth','oblique'])
    p.add_argument('--sub',type=int,default=3);p.add_argument('--length',type=float,default=1e6)
    p.add_argument('--dt',type=float,default=1.);p.add_argument('--end',type=float,default=20.);p.add_argument('--resume')
    a=p.parse_args();run(a.name,scenario=a.scenario,sub=a.sub,length=a.length,dt=a.dt,end=a.end,resume=a.resume)
