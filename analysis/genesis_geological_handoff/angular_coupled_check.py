"""Bounded native-thermal/phase plus full-angular memory continuation."""
import json,time
from pathlib import Path
import numpy as np
from tectonics.experimental_angular_coupled import AngularCoupled,base_face_energy
from tectonics.experimental_edge_weakening import scale
from tectonics.simplified_genesis_nodal import material_mass
from tectonics.simplified_genesis_basal import geometry
from tectonics.genesis_transfer_bridge import encoded,digest
from .edge_spatial_audit import ROOT

SOURCE=Path('/workspace/genesis-source-angular-01/initial.json')
OUT=Path('/workspace/genesis-angular-coupled-01')


def world_stress(m):
    C=m.C*scale(m,m.state['damage'])[:,None,None]
    mean=np.einsum('fi,fij->fj',m.G[:,0,:],np.asarray(m.state['memory']))
    weighted=np.einsum('fij,fj->fi',C,mean);volume=m.G[:,0,0].copy()
    for b in m.state.get('born_material',[]):
        weighted[b['face']]+=b['volume_m3']*C[b['face']]@np.asarray(b['memory']);volume[b['face']]+=b['volume_m3']
    stress=weighted/volume[:,None];tensor=np.zeros((m.nf,3,3));tensor[:,range(3),range(3)]=stress[:,:3]
    for k,(i,j) in enumerate(((0,1),(0,2),(1,2)),3):tensor[:,i,j]=tensor[:,j,i]=stress[:,k]
    frames=np.array(m.config['nodal']['frames'])
    return np.einsum('fai,fij,fbj->fab',frames,tensor,frames)


def snapshot(m):
    return dict(stress=world_stress(m),T=m.temperature(m.state['energy']),old_damage=np.array(m.state['damage']),
        edge_damage=np.array(m.state['experimental_edge_fatigue']['damage']),born=len(m.state.get('born_material',[])))


def mechanical_energy(m):
    v=np.asarray(m.state['nodal_velocity'])
    return float(base_face_energy(m)@scale(m,m.state['damage'])),float(.5*v@material_mass(m)@v)


def main():
    if OUT.exists():raise FileExistsError('Keep existing coupled results')
    blob=SOURCE.read_bytes();start=time.monotonic();deadline=start+120;OUT.mkdir()
    runs={};fields={};restart=False
    for name,dt in [('coarse',.125),('fine',.0625)]:
        a=AngularCoupled.from_source_checkpoint(SOURCE);m=a.model;initial=snapshot(m);mass=m.mass.copy()
        rows=[];samples={};Hstart=np.asarray(m.state['energy'],np.longdouble);Ustart,Kstart=mechanical_energy(m);Rstart=m.state['experimental_edge_fracture_reserve_j']
        external=source=numerical=allowance=0.
        if name=='coarse':a.save(OUT/'initial.json')
        for i in range(round(1/dt)):
            if time.monotonic()>deadline:raise TimeoutError('120second coupled budget')
            oldH=np.asarray(m.state['energy'],np.longdouble);tick=time.monotonic();L=a.advance(dt);elapsed=time.monotonic()-tick
            snap=snapshot(m);dH=float(np.sum(np.asarray(m.state['energy'],np.longdouble)-oldH))
            heat_expected=L['source_energy']+L['maxwell']-L['thermal_work']-L['damage_energy_change']+L['basal']['heat_j']+L['phase']['capture_heat_j']
            external+=L['external_work'];source+=L['source_energy'];numerical+=L['numerical_memory']+L['numerical_kinetic'];allowance+=L['whole_allowance_j']
            U,K=mechanical_energy(m)
            whole=float(np.sum(np.asarray(m.state['energy'],np.longdouble)-Hstart))+U-Ustart+K-Kstart+m.state['experimental_edge_fracture_reserve_j']-Rstart-source-external+numerical
            assert abs(whole)<=allowance
            row=dict(years=(i+1)*dt,seconds=elapsed,whole_balance_j=L['whole_balance_j'],whole_allowance_j=L['whole_allowance_j'],
                whole_allowance_fraction=abs(L['whole_balance_j'])/L['whole_allowance_j'],cumulative_allowance_fraction=abs(whole)/allowance,
                thermal_identity_residual_j=dH-heat_expected,source_energy_j=L['source_energy'],maxwell_j=L['maxwell'],
                thermal_work_j=L['thermal_work'],old_damage_energy_change_j=L['damage_energy_change'],basal_heat_j=L['basal']['heat_j'],
                capture_heat_j=L['phase']['capture_heat_j'],edge_release_j=L['edge_fracture_release_j'],numerical_memory_j=L['numerical_memory'],
                numerical_kinetic_j=L['numerical_kinetic'],mechanical_balance_relative=L['moment_balance_relative'],
                momentum_relative=L['momentum_relative'],temperature_max_change_k=float(abs(snap['T']-initial['T']).max()),
                old_damage_max_change=float(abs(snap['old_damage']-initial['old_damage']).max()),max_edge_damage=float(snap['edge_damage'].max()),
                saturated=int(sum(snap['edge_damage']>=1)),born=snap['born'],new_phase_solid_kg=L['phase']['new_phase_solid_kg'],
                certificate=m.state['experimental_edge_fatigue']['reason'],corrector_iterations=L['corrector_iterations'])
            rows.append(row)
            if abs((i+1)*dt/.125-round((i+1)*dt/.125))<1e-10:samples[str(round((i+1)*dt,6))]=snap
            if name=='fine' and i==6:
                a.save(OUT/'restart_before.json');b=AngularCoupled.load(OUT/'restart_before.json');b.advance(dt)
            if name=='fine' and i==7:
                restart=encoded(b.model.state)==encoded(m.state);assert restart
            if m.state['experimental_edge_fatigue']['partition_owner'] is not None:break
        np.testing.assert_array_equal(m.mass,mass)
        a.save(OUT/(name+'_final.json'));fields[name]=samples
        runs[name]=dict(dt=dt,rows=rows,final_time_myr=m.state['time_myr'],mass_exact=True,
            cumulative_source_energy_j=source,cumulative_external_work_j=external,cumulative_numerical_j=numerical,
            cumulative_whole_balance_j=whole,cumulative_whole_allowance_j=allowance,
            final_fracture_reserve_j=m.state['experimental_edge_fracture_reserve_j'])
        print(json.dumps(dict(run=name,seconds=time.monotonic()-start,final=rows[-1])),flush=True)
    comparison=[];area=m.mesh.areas_unit_sphere
    for t in sorted(set(fields['coarse'])&set(fields['fine']),key=float):
        c,f=fields['coarse'][t],fields['fine'][t];delta=c['stress']-f['stress']
        comparison.append(dict(years=float(t),stress_relative=float(np.sqrt(np.einsum('f,fij,fij->',area,delta,delta)/np.einsum('f,fij,fij->',area,f['stress'],f['stress']))),
            edge_damage_absolute=float(abs(c['edge_damage']-f['edge_damage']).max()),old_damage_absolute=float(abs(c['old_damage']-f['old_damage']).max()),
            T_absolute_k=float(abs(c['T']-f['T']).max()),birth_counts_equal=c['born']==f['born']))
    # Controlled birth transition, explicitly separated from the real trajectory.
    control=AngularCoupled.from_source_checkpoint(SOURCE);cm=control.model;cell=cm.config['phase_cells'][0];node=cell['node']
    before_front=geometry(cm)
    before_h=cm.state['energy'][node];cm.state['energy'][node]=float(cm.mass[node]*cm.cp[node]*(cm.config['solidus']-1.))
    imposed_h=cm.state['energy'][node]-before_h;L1=control.advance(.0625)
    zero=np.array(cm.state['angular_born_memory'][cell['id']]);assert np.max(abs(zero))==0
    after_front=geometry(cm)
    control.save(OUT/'controlled_birth.json');again=AngularCoupled.load(OUT/'controlled_birth.json')
    L2=control.advance(.0625);again.advance(.0625);assert encoded(cm.state)==encoded(again.model.state)
    control.save(OUT/'controlled_after_birth.json')
    controlled=dict(kind='Artificial thermal initial condition only; not actual source trajectory birth',cell_id=cell['id'],node=node,
        imposed_enthalpy_change_j=imposed_h,first_phase=L1['phase'],birth_angular_memory_exact_zero=True,
        front_q_before=float(before_front['fronts'][cell['face']]),front_q_after=float(after_front['fronts'][cell['face']]),
        depth_m_before=float(before_front['depth'][cell['face']]),depth_m_after=float(after_front['depth'][cell['face']]),
        next_memory_norm=float(np.linalg.norm(cm.state['angular_born_memory'][cell['id']])),second_phase=L2['phase'],restart_exact=True,
        whole_allowance_fractions=[abs(x['whole_balance_j'])/x['whole_allowance_j'] for x in (L1,L2)])
    with (OUT/'controlled_preparation.json').open('x') as stream:
        json.dump(dict(format='controlled-thermal-preparation/v1',source_checkpoint_sha256=digest(blob),
            preparation=controlled,checkpoints={name:digest((OUT/name).read_bytes()) for name in ('controlled_birth.json','controlled_after_birth.json')}),stream,indent=2)
        stream.write('\n')
    np.savez_compressed(OUT/'fields.npz',**{f'{name}_{t}_{key}':value for name,series in fields.items() for t,values in series.items() for key,value in values.items()})
    assert SOURCE.read_bytes()==blob
    result=dict(code_base='1f7176f2c8fce97039cf1570ea4a82245192ff05',source_checkpoint_sha256=digest(blob),source_unchanged=True,
        runs=runs,comparison=comparison,restart_exact=restart,controlled_birth=controlled,
        temporal_gate_passed=all(x['stress_relative']<.05 and x['edge_damage_absolute']<.02 and x['old_damage_absolute']<.002 and x['T_absolute_k']<.05 and x['birth_counts_equal'] for x in comparison),
        output=str(OUT),files={p.name:dict(bytes=p.stat().st_size,sha256=digest(p.read_bytes())) for p in OUT.iterdir()},seconds=time.monotonic()-start)
    (ROOT/'angular_coupled_check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('runs','files')},indent=2))


if __name__=='__main__':main()
