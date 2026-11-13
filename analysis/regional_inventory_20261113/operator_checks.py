"""Independent continuum strip and finite-donor checks; no parameter fitting.

The analytic integral is over physical coordinates, not the material mesh.
Rotating the whole source probes quadrature phase error at identical physics.
No coupled long-run output is overwritten by this diagnostic.
"""
from copy import deepcopy
from dataclasses import replace
import json,math,subprocess,sys,time
from pathlib import Path
import numpy as np
import yaml
from scipy.integrate import quad
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tectonics.regional_material import RegionalMaterialInventory,RegionalMaterialParameters,arc_field_at_points
from tectonics.integral_arcs import ArcSegment
from tectonics.mesh import build_icosphere
from tectonics.continental import ContinentalCycleParameters
from tectonics.checkpoint import load_checkpoint
from tectonics.topology import PlateTopologyManager,PlateTopologyParameters
from tectonics.material_exchange import MaterialExchangeBudget,MaterialExchangeParameters
from tectonics.thermal import ThermalParameters
HERE=Path(__file__).resolve().parent;R=5287.;END=100.
CFG=yaml.safe_load((HERE/'control.yaml').read_text())
PARAMS=ContinentalCycleParameters(**CFG['continental_cycle'])


def continuum(params):
    field=lambda d:.7*math.exp(-.5*((d-105.)/140.)**2)
    integral=R*quad(lambda d:math.cos(d/R)*field(d),0,425.,epsabs=1e-8)[0]
    peak=.7*params.arc_maturation_rate_per_myr*params.arc_potential_decay_myr*(-math.expm1(-END/params.arc_potential_decay_myr))
    if peak<=params.juvenile_threshold:return dict(source_km2=integral,area_km2=0.,volume_km3=0.,age_moment_km3_myr=0.)
    width=140*math.sqrt(2*math.log(peak/params.juvenile_threshold))
    lo=max(0.,105-width);hi=min(425.,105+width)
    def remaining(d):
        return END+params.arc_potential_decay_myr*math.log1p(-params.juvenile_threshold/(field(d)*params.arc_maturation_rate_per_myr*params.arc_potential_decay_myr))
    def volume(d):return params.juvenile_continental_thickness_km+params.continental_arc_thickening_km_per_myr*field(d)*remaining(d)
    def age(d):return params.juvenile_continental_thickness_km*remaining(d)+params.continental_arc_thickening_km_per_myr*field(d)*remaining(d)**2/2
    return dict(source_km2=integral,area_km2=R*R*(math.sin(hi/R)-math.sin(lo/R)),
                volume_km3=R*quad(lambda d:math.cos(d/R)*volume(d),lo,hi)[0],
                age_moment_km3_myr=R*quad(lambda d:math.cos(d/R)*age(d),lo,hi)[0])


def point_case(sub,rotation,params,dt):
    mesh=build_icosphere(sub);area=mesh.physical_cell_areas_km2(R);n=len(area)
    inv=RegionalMaterialInventory(dict(points=mesh.centroids,owner=np.zeros(n,int),area=area,
        felsic_area=np.zeros(n),felsic_volume=np.zeros(n),felsic_age_moment=np.zeros(n),
        basalt_volume=area*7,basalt_age_moment=area*700,potential=np.zeros(n)),RegionalMaterialParameters(material_subdivisions=sub))
    a,b,t=[rotation@v for v in (np.array([1.,0,0]),np.array([math.cos(1),math.sin(1),0]),np.array([0.,0,1.]))]
    field,_=arc_field_at_points(inv.arrays['points'],inv.arrays['owner'],[ArcSegment(a,b,t,0,.7,105.)],R,140.,320.)
    born=[];returned=[];first=None
    for _ in range(round(END/dt)):
        step=inv.react(dt,field,params);born.append(step['born_felsic_km3']);returned.append(step['returned_basalt_km3'])
        if first is None:first=step['first_event_myr']
    a=inv.arrays
    return dict(material_columns=n,dt_myr=dt,source_km2=math.fsum(field*area),area_km2=math.fsum(a['felsic_area']),
        volume_km3=math.fsum(a['felsic_volume']),age_moment_km3_myr=math.fsum(a['felsic_age_moment']),
        returned_basalt_km3=math.fsum(returned),first_event_myr=first,
        felsic_volume_residual_km3=math.fsum([*a['felsic_volume'],*(-np.asarray(born))]),
        basalt_volume_residual_km3=math.fsum([*a['basalt_volume'],*returned,*(-area*7)]))


def finite_exchange_check():
    cp=load_checkpoint(ROOT/'outputs_v131_regional/initial_same_g2/restart',PlateTopologyManager(PlateTopologyParameters()))
    data=cp.material_exchange;opts=MaterialExchangeParameters(**data['options']);thermal=ThermalParameters(**CFG['thermal'])
    budget=MaterialExchangeBudget.restore(data,opts)
    inv=RegionalMaterialInventory.initialize(build_icosphere(2),cp.state,R,RegionalMaterialParameters())
    # Uniform imposed source is an operator test, not a computed lunar arc.
    before=deepcopy(budget.checkpoint());tm,tc=budget.temperatures(thermal)
    x=inv.react(END,np.full(len(inv.arrays['area']),.7),PARAMS)
    born=[x['born_felsic_km3'],x['born_basalt_km3']];returned=[x['returned_felsic_km3'],x['returned_basalt_km3']]
    target=[math.fsum(inv.arrays['felsic_volume']),math.fsum(inv.arrays['basalt_volume'])]
    mass_created=math.fsum(np.array(born)*1e9*np.array([opts.continental_rock_density_kg_m3,opts.basalt_density_kg_m3]))
    expected_q=mass_created*thermal.mantle_heat_capacity_j_kg_k*tm
    budget.exchange(born,returned,target,thermal)
    observed_q=math.fsum(budget.data['advected_out_j'])-math.fsum(before['advected_out_j'])
    diag=budget.diagnostics(thermal)
    restored=MaterialExchangeBudget.restore(json.loads(json.dumps(budget.checkpoint())),opts)
    assert restored.checkpoint()==budget.checkpoint()
    # Unaffordable request must reject atomically, with no partial debit.
    saved=budget.checkpoint();too_much=budget.pool_masses()[0]/(opts.continental_rock_density_kg_m3*1e9)*1.01
    rejected=False
    try:budget.exchange([too_much,0],[0,0],[target[0]+too_much,target[1]],thermal)
    except ValueError:rejected=True
    assert rejected and budget.checkpoint()==saved
    return dict(reaction=x,diagnostics=diag,advected_out_expected_j=expected_q,advected_out_observed_j=observed_q,
                advected_out_relative_error=(observed_q-expected_q)/expected_q,unaffordable_request_atomic=True,
                json_restart_equal=True,scope='finite mass and sensible enthalpy only; no latent/chemical/work energy')


def run(filename='operator_checks.json'):
    start=time.perf_counter();rows=[]
    axis=np.array([1.,2.,3.]);axis/=np.linalg.norm(axis)
    rotations={'aligned':np.eye(3),'oblique':Rotation.from_rotvec(axis*math.radians(43)).as_matrix()}
    exact=continuum(PARAMS)
    for orientation,rotation in rotations.items():
        for sub in (2,3,4,5):
            result=point_case(sub,rotation,PARAMS,100.)
            rows.append(dict(orientation=orientation,**result,relative_error={k:result[k]/exact[k]-1 for k in exact}))
    time_rows=[point_case(3,rotations['oblique'],PARAMS,dt) for dt in (4.,2.,1.)]
    sensitivity=[]
    for factor in (.5,1.,2.):
        p=replace(PARAMS,arc_maturation_rate_per_myr=PARAMS.arc_maturation_rate_per_myr*factor)
        sensitivity.append(dict(rate_factor=factor,analytic=continuum(p),numerical=point_case(5,rotations['oblique'],p,100.)))
    result=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        exact_continuum=exact,spatial=rows,time_partition=time_rows,parameter_sensitivity=sensitivity,
        finite_exchange=finite_exchange_check(),wall_s=time.perf_counter()-start,
        scope='synthetic frozen forcing; numerical resolution versus effective-law sensitivity, not lunar calibration')
    with (HERE/filename).open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result),flush=True)

if __name__=='__main__':run(sys.argv[1] if len(sys.argv)>1 else 'operator_checks.json')
