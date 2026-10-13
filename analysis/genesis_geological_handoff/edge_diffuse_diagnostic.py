"""Read saved endpoints; two 25-year controls and frozen linear load attribution.

No trajectory continuation, threshold search, or change to physical defaults.
"""
import hashlib,json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import numpy as np
from tectonics.simplified_genesis import SimplifiedGenesis
from tectonics.simplified_genesis_nodal import operators,blocks,material_mass
from tectonics.simplified_genesis_basal import assemble
from tectonics.experimental_genesis_edge_rule import update
from tectonics.genesis_transfer_bridge import encoded
from tectonics.genesis import SECONDS_PER_MYR
from tectonics.genesis_tides import TidalParameters, tidal_strain_cycle
from tectonics.genesis_shell import principal_tensile
from analysis.genesis_aggregate_coupled.core import step
from .nodal_retirement_check import OPTION

ROOT=Path(__file__).parent
SOURCE=Path('/workspace/genesis-moving-basal-source-01.json')
END=Path('/workspace/genesis-edge-retirement-pilot-01/checkpoint.json')


def stats(x):
    x=np.asarray(x);return dict(min=float(x.min()),mean=float(x.mean()),max=float(x.max()),rms=float(np.sqrt(np.mean(x*x))))


def tensors(m,memory=None,damage=None):
    e=np.einsum('fnk,fki->fni',m.P,np.asarray(m.state['memory'] if memory is None else memory).reshape(m.nf,2,6))
    stress=np.einsum('fij,fnj->fni',m.C*m.scale(m.state['damage'] if damage is None else damage)[:,None,None],e)
    sigma=np.einsum('fn,fni->fi',m.weights,stress)/m.weights.sum(axis=1)[:,None]
    t=np.zeros((m.nf,3,3));t[:,range(3),range(3)]=sigma[:,:3]
    for k,(i,j) in enumerate(((0,1),(0,2),(1,2)),3):t[:,i,j]=t[:,j,i]=sigma[:,k]
    return t


def shear_vectors(m,t):
    frame=operators(m)['frames'];world=np.einsum('fai,fij,fbj->fab',frame,t,frame);out=[]
    for f,g,i,j in m.mesh.shared_edges:
        a,b=m.mesh.vertices[[i,j]];radial=a+b;radial/=np.linalg.norm(radial)
        tangent=b-a;tangent/=np.linalg.norm(tangent);normal=np.cross(tangent,radial)
        traction=.5*(world[f]+world[g])@normal;out.append(traction-normal*(traction@normal))
    return np.array(out)


def shear_summary(m,t):
    v=shear_vectors(m,t);n=np.linalg.norm(v,axis=1)
    return dict(pa=stats(n),above_1MPa=int(sum(n>1e6)),above_3MPa=int(sum(n>=3e6)))


def endpoint(m):
    t=tensors(m);w=m.weights.sum(axis=1);w=w/w.sum()
    # Common local axisymmetric shell stress: tangential mean + radial mean.
    tang=(t[:,0,0]+t[:,1,1])/2;rad=t[:,2,2]
    common=np.tile(np.diag([w@tang,w@tang,w@rad]),(m.nf,1,1))
    hydro=np.eye(3)[None]*np.trace(t,axis1=1,axis2=2)[:,None,None]/3
    d=np.asarray(m.state['damage']);T=m.temperature(m.state['energy'])[m.solid_nodes]
    shear=np.linalg.norm(shear_vectors(m,t),axis=1)
    return dict(time_myr=m.state['time_myr'],old_damage=stats(d),old_stiffness_scale=stats(m.scale(d)),
        temperature_k=stats(T),thermal_depth_node_spatial_range_k=np.ptp(T,axis=0).tolist(),
        domains=len(set(m.state['owner'])),newborn=len(m.state.get('born_material',[])),
        common_tangential_pa=float(w@tang),common_radial_pa=float(w@rad),
        full=shear_summary(m,t),common_axisymmetric=shear_summary(m,common),
        departure_from_common=shear_summary(m,t-common),hydrostatic=shear_summary(m,hydro),
        fixed_memory_without_old_damage=shear_summary(m,tensors(m,damage=np.zeros(m.nf))),
        fixed_memory_uniform_old_damage=shear_summary(m,tensors(m,damage=np.full(m.nf,w@d))),
        shear_minus_threshold_pa=stats(shear-1e6),
        frozen_load_time_to_D1_years=stats(100/(np.maximum(shear[shear>1e6]/1e6-1,0)**2)))


def main():
    pins={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (SOURCE,END)}
    assert pins[str(SOURCE)]=='94fbbb3cb88a15147d33ab04c8f791470dd92e3d73cf8cba4afc0f361cd583fc'
    assert pins[str(END)]=='6f7052512a38696c76a84f5894ac30ddaf6b5bc02a9c36de3c129fce9df9c9c1'
    src=SimplifiedGenesis.load(SOURCE);end=SimplifiedGenesis.load(END)
    c=deepcopy(src.config);c['experimental_geological_edge_fatigue']=OPTION
    low=SimplifiedGenesis(c,src.state);high=SimplifiedGenesis(c,src.state)
    high.state['experimental_edge_fatigue']={'damage':[.8]*len(src.mesh.shared_edges)}
    low.advance(25.);high.advance(25.)
    a=deepcopy(low.state);b=deepcopy(high.state);a.pop('experimental_edge_fatigue');b.pop('experimental_edge_fatigue')
    assert encoded(a)==encoded(b)
    # Match production reconstruction independently at endpoint.
    np.testing.assert_allclose(np.linalg.norm(shear_vectors(end,tensors(end)),axis=1),end.state['experimental_edge_fatigue']['shear_traction_pa'],rtol=2e-14)
    T0=src.temperature(src.state['energy']);T1=low.temperature(low.state['energy']);D=np.array(low.state['damage'])
    bs=blocks(src,T1,D,T0);M=material_mass(src);basal=assemble(src);dt=(low.state['time_myr']-src.state['time_myr'])*SECONDS_PER_MYR
    full=step(M,basal['D'],basal['F'],np.array(src.state['nodal_velocity']),dt,bs)
    delta=(T1-T0)[src.solid_nodes];weights=np.array(src.config['mass'])[src.solid_nodes]
    mean=(delta*weights).sum(axis=0)/weights.sum(axis=0)
    unit=np.array([1,1,1,0,0,0]);alpha=src.shell['linear_expansion_per_k']
    common_theta=[(np.linalg.solve(src.P[f],mean)[:,None]*alpha*unit).ravel() for f in range(src.nf)]
    modes={};sum_memory=np.zeros_like(full['memory']);sum_v=np.zeros_like(full['velocity'])
    for name in ['initial_memory','common_thermal_increment','spatial_thermal_increment','basal_traction','initial_velocity']:
        local=deepcopy(bs)
        for f,b in enumerate(local):
            b['old']=b['old'] if name=='initial_memory' else np.zeros_like(b['old'])
            b['theta']=common_theta[f] if name=='common_thermal_increment' else (b['theta']-common_theta[f] if name=='spatial_thermal_increment' else np.zeros_like(b['theta']))
        r=step(M,basal['D'],basal['F'] if name=='basal_traction' else np.zeros_like(basal['F']),np.array(src.state['nodal_velocity']) if name=='initial_velocity' else np.zeros_like(full['velocity']),dt,local)
        modes[name]=shear_summary(src,tensors(src,r['memory'],D));sum_memory+=r['memory'];sum_v+=r['velocity']
    memory_error=float(np.linalg.norm(sum_memory-full['memory'])/np.linalg.norm(full['memory']))
    assert memory_error<1e-10
    production_error=float(np.linalg.norm(full['memory'].reshape(src.nf,2,6)-np.array(low.state['memory']))/np.linalg.norm(full['memory']))
    assert production_error<1e-8
    tp=replace(TidalParameters(**src.config['tidal_parameters']),eccentricity=low.state['orbit']['eccentricity'],semimajor_axis_km=low.state['orbit']['semimajor_axis_km'])
    nu=src.shell['poisson_ratio'];plane=np.array([[1.,nu,0.],[nu,1.,0.],[0.,0.,(1-nu)/2]])/(1-nu**2)
    tide=np.einsum('ij,pfj->pfi',plane,tidal_strain_cycle(src.mesh,tp))*src.shell['young_modulus_pa']*src.scale(src.state['damage'])[None,:,None]
    predblocks=blocks(src,T1,np.array(src.state['damage']),T0)
    pred=step(M,basal['D'],basal['F'],np.array(src.state['nodal_velocity']),dt,predblocks)
    tensor=tensors(src,pred['memory']);inplane=tensor[:,[0,1,0],[0,1,1]]
    without=np.maximum(principal_tensile(inplane)/src.shell['tensile_strength_pa']-1,0)**2
    with_tide=np.mean([np.maximum(principal_tensile(inplane+t)/src.shell['tensile_strength_pa']-1,0)**2 for t in tide],axis=0)
    tide_record=dict(cycle_stress_component_abs_max_pa=float(abs(tide).max()),old_damage_unsmoothed_load_without_tide=stats(without),old_damage_unsmoothed_load_with_tide=stats(with_tide),note='Tidal cycle enters old tensile damage only, not the edge fatigue stress reconstruction directly')
    rows=json.loads((ROOT/'nodal_retirement_result.json').read_text())['pilot']['rows'];previous=0;timeline=[]
    for i,row in enumerate(rows):
        timeline.append(dict(years=25*(i+1),saturated=row['failed_edges'],newly_saturated=row['failed_edges']-previous,max_shear_pa=row['max_shear_pa']))
        previous=row['failed_edges']
    d=np.array(end.state['experimental_edge_fatigue']['damage']);unsat=d<1
    late=SimplifiedGenesis.load('/workspace/genesis-moving-basal-pilot-01/checkpoint.json')
    from tectonics.genesis_geological_thermal import eligibility
    late_reasons=eligibility(late,experimental_nodal_retirement=True,partition_owner=(late.mesh.centroids[:,2]>0).astype(int))
    result=dict(source_pins=pins,start=endpoint(src),first_step=endpoint(low),endpoint=endpoint(end),
        parameters=OPTION,old_shell_parameters=src.shell,old_fault_parameters=src.faults,tidal_first_step=tide_record,timeline=timeline,
        first_step_saturation_shear_threshold_pa=3e6,allowed_saturated_edges=42,first_excess_edges=48-42,end_excess_edges=84-42,
        unsaturated_count=int(unsat.sum()),unsaturated_damage=stats(d[unsat]),unsaturated_margin_to_1=stats(1-d[unsat]),
        feedback_control=dict(initial_edge_damage=[0,.8],all_other_state_bitwise_identical=True,
            saturated_counts=[sum(low.state['experimental_edge_fatigue']['failed_edges']),sum(high.state['experimental_edge_fatigue']['failed_edges'])]),
        frozen_first_step_attribution=dict(scope='linear decomposition at fixed converged T and old-material damage; vector components add, norms do not; not separate coupled histories',
            common_temperature_change_k=mean.tolist(),spatial_temperature_change_k=stats(delta-mean),components=modes,
            memory_superposition_relative=memory_error,velocity_superposition_relative=float(np.linalg.norm(sum_v-full['velocity'])/np.linalg.norm(full['velocity'])),production_memory_relative=production_error),
        late_handoff=dict(time_myr=late.state['time_myr'],newborn=len(late.state['born_material']),readiness_with_manufactured_partition=late_reasons))
    for p in (SOURCE,END):assert hashlib.sha256(p.read_bytes()).hexdigest()==pins[str(p)]
    (ROOT/'edge_diffuse_diagnostic.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
