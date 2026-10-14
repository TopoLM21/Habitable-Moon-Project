"""Predeclared short native-thermal spherical pilot plus isolated admission."""
from pathlib import Path
import json,time
import numpy as np
from tectonics.genesis_transfer_bridge import encoded,digest
from tectonics.experimental_edge_weakening import scale
from .spherical_coupled import SphericalCoupled
from .coupled_spatial_check import field,compare
from .edge_spatial_audit import ROOT
START=Path('/workspace/spherical-coupled-preflight-initial.json')
OUT=Path('/workspace/genesis-spherical-coupled-01')


def rigid_momenta(before_path,after_path):
    result={}
    for name,path in [('old',before_path),('new',after_path)]:
        b=json.loads(Path(path).read_bytes())['payload'];X=np.array(b['config']['nodal']['X'])
        M=np.array(b['config']['nodal']['M']);v=np.array(b['state']['nodal_velocity']);p=(M@v).reshape(-1,6)
        result[name]=dict(linear_kg_m_s=p[:,:3].sum(0).tolist(),angular_kg_m2_s=np.cross(X,p[:,:3]+p[:,3:]).sum(0).tolist())
    for k in result['old']:
        result[k+'_relative']=float(np.linalg.norm(np.array(result['new'][k])-result['old'][k])/np.linalg.norm(result['old'][k]))
    return result


def main():
    if OUT.exists():raise FileExistsError('Preserve prior pilot')
    start=time.monotonic();OUT.mkdir();input_sha=digest(START.read_bytes());runs={};fields={};restart=False;solve_time=0.
    for name,dt in [('coarse',.03125),('fine',.015625)]:
        a=SphericalCoupled.load(START);m=a.model;initial=field(m);mass=m.mass.copy();rows=[];trace={}
        if name=='coarse':a.save(OUT/'initial.json');mapping=a.origin
        for i in range(round(.0625/dt)):
            if solve_time>180 or time.monotonic()-start>360:raise TimeoutError('Predeclared budget')
            tick=time.monotonic();L=a.advance(dt);seconds=time.monotonic()-tick;solve_time+=seconds;snap=field(m)
            U=float(a.base_energy()@scale(m,m.state['damage']));K=float(.5*np.array(m.state['nodal_velocity'])@a.material_mass()@np.array(m.state['nodal_velocity']))
            assert abs(U-L['elastic_after'])<1e-12*max(U,1.)
            assert abs(K-L['kinetic_after'])<1e-8*max(K,1.)
            np.testing.assert_array_equal(m.mass,mass)
            rows.append(dict(years=(i+1)*dt,seconds=seconds,whole_balance_j=L['whole_balance_j'],whole_allowance_j=L['whole_allowance_j'],
                whole_allowance_fraction=abs(L['whole_balance_j'])/L['whole_allowance_j'],mechanical_balance_relative=L['moment_balance_relative'],momentum_relative=L['momentum_relative'],
                maxwell_j=L['maxwell'],thermal_work_j=L['thermal_work'],damage_energy_change_j=L['damage_energy_change'],drag_heat_j=L['basal']['heat_j'],
                numerical_memory_j=L['numerical_memory'],numerical_kinetic_j=L['numerical_kinetic'],external_work_j=L['external_work'],
                source_energy_j=L['source_energy'],elastic_j=U,kinetic_j=K,edge_release_j=L['edge_fracture_release_j'],edge_heat_j=L['edge_damage_heat_j'],
                births=len(m.state.get('born_material',[])),new_phase_solid_kg=L['phase']['new_phase_solid_kg'],max_temperature_change_k=float(abs(snap['T']-initial['T']).max()),
                max_old_damage_change=float(abs(snap['oldD']-initial['oldD']).max()),max_face_damage=float(snap['faceD'].max()),corrector_iterations=L['corrector_iterations'],
                certificate=m.state['experimental_edge_fatigue']['reason']))
            if abs((i+1)*dt/.03125-round((i+1)*dt/.03125))<1e-10:trace[str((i+1)*dt)]=snap
            if name=='fine' and i==0:
                a.save(OUT/'restart_before.json');b=SphericalCoupled.load(OUT/'restart_before.json')
            elif name=='fine' and i==1:
                tick=time.monotonic();b.advance(dt);solve_time+=time.monotonic()-tick;restart=encoded(b.model.state)==encoded(m.state);assert restart
            print(json.dumps(dict(run=name,step=i+1,seconds=seconds,whole_fraction=rows[-1]['whole_allowance_fraction'])),flush=True)
            if m.state['experimental_edge_fatigue']['partition_owner'] is not None:raise RuntimeError('Certified split: pilot stopped')
        a.save(OUT/(name+'_final.json'));runs[name]=rows;fields[name]=trace
    comparisons={t:compare(fields['coarse'][t],fields['fine'][t],m.G[:,0,0]) for t in fields['coarse']}
    control=SphericalCoupled.load(START);cm=control.model;cell=cm.config['phase_cells'][0];node=cell['node'];mass=cm.mass.copy();before=cm.state['energy'][node]
    cm.state['energy'][node]=float(cm.mass[node]*cm.cp[node]*(cm.config['solidus']-1)+cm.mass[node]*cm.offset[node]);delta=cm.state['energy'][node]-before
    tick=time.monotonic();L1=control.advance(.015625);solve_time+=time.monotonic()-tick
    born=cm.state['born_material'];assert len(born)==1 and born[0]['cell_id']==cell['id'];assert not np.any(cm.state['spherical_born_memory'][cell['id']])
    control.save(OUT/'controlled_birth.json');resumed=SphericalCoupled.load(OUT/'controlled_birth.json')
    tick=time.monotonic();L2=control.advance(.015625);resumed.advance(.015625);solve_time+=time.monotonic()-tick
    assert encoded(cm.state)==encoded(resumed.model.state);np.testing.assert_array_equal(cm.mass,mass)
    assert np.linalg.norm(cm.state['spherical_born_memory'][cell['id']])>0
    control.save(OUT/'controlled_after_birth.json')
    controlled=dict(kind='Artificial thermal preparation, not spontaneous real-run birth',cell_id=cell['id'],node=node,imposed_enthalpy_change_j=delta,
        first_phase=L1['phase'],second_phase=L2['phase'],newborn_memory_initial_zero=True,next_memory_norm=float(np.linalg.norm(cm.state['spherical_born_memory'][cell['id']])),
        restart_exact=True,whole_allowance_fractions=[abs(L['whole_balance_j'])/L['whole_allowance_j'] for L in (L1,L2)])
    (OUT/'controlled_preparation.json').write_text(json.dumps(controlled,indent=2)+'\n')
    np.savez_compressed(OUT/'fields.npz',**{f'{name}_{t}_{k}':v for name,trace in fields.items() for t,s in trace.items() for k,v in s.items()})
    assert digest(START.read_bytes())==input_sha
    result=dict(code_base='9bb3f0178b6e8d32ff36367ed313d7948e38d379',mapping=mapping,initial_sha256=input_sha,
        runs=runs,comparisons=comparisons,restart_exact=restart,controlled=controlled,solve_seconds=solve_time,seconds=time.monotonic()-start,output=str(OUT),
        files={p.name:dict(bytes=p.stat().st_size,sha256=digest(p.read_bytes())) for p in OUT.iterdir()})
    (ROOT/'spherical_coupled_check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('runs','files')},indent=2))

if __name__=='__main__':main()
