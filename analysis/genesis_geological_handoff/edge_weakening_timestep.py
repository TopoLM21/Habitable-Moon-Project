"""Predeclared same-source 25-year refinement, no tuning or long continuation."""
from copy import deepcopy
from pathlib import Path
import hashlib,json,time
import numpy as np
from tectonics.simplified_genesis import SimplifiedGenesis
from tectonics.experimental_edge_weakening import MODE,face_damage
from tectonics.genesis_transfer_bridge import encoded
from .nodal_retirement_check import SOURCE,OPTION
from .edge_diffuse_diagnostic import tensors,shear_vectors

ROOT=Path(__file__).parent
OUTPUT=Path('/workspace/genesis-edge-weakening-dt-01')
STEPS=(25.,5.,1.,.5)
HORIZON=25.
CAP_SECONDS=120.


def field(m):
    d=np.array(m.state['experimental_edge_fatigue']['damage'])
    effective=np.maximum(m.state['damage'],face_damage(m.mesh,d))
    return d,np.linalg.norm(shear_vectors(m,tensors(m,damage=effective)),axis=1)


def comparison(a,b):
    da=np.array(a['damage']);db=np.array(b['damage']);ta=np.array(a['shear_after_update_pa']);tb=np.array(b['shear_after_update_pa'])
    sa=set(np.flatnonzero(da>=1));sb=set(np.flatnonzero(db>=1));ha=set(np.argsort(da)[-30:]);hb=set(np.argsort(db)[-30:])
    ratio=round(a['dt_years']/b['dt_years']);samples=[]
    for i,r in enumerate(a['rows']):
        fine=b['rows'][(i+1)*ratio-1]
        samples.append(dict(years=r['years'],max_damage_difference=abs(r['max_damage']-fine['max_damage']),
            peak_shear_relative_difference=abs(r['max_shear_after_update_pa']-fine['max_shear_after_update_pa'])/fine['max_shear_after_update_pa']))
    return dict(coarse_dt=a['dt_years'],fine_dt=b['dt_years'],worst_common_time_damage=max(samples,key=lambda r:r['max_damage_difference']),worst_common_time_peak_shear=max(samples,key=lambda r:r['peak_shear_relative_difference']),damage_rms_difference=float(np.sqrt(np.mean((da-db)**2))),
        damage_max_difference=float(np.max(abs(da-db))),damage_correlation=float(np.corrcoef(da,db)[0,1]),
        top_quarter_overlap=len(ha&hb)/30,saturated_symmetric_difference=len(sa^sb),
        shear_relative_l2=float(np.linalg.norm(ta-tb)/max(np.linalg.norm(tb),1.)),
        shear_max_difference_pa=float(np.max(abs(ta-tb))),reserve_relative_difference=(a['reserve_j']-b['reserve_j'])/b['reserve_j'])


def main():
    start=time.monotonic();assert not OUTPUT.exists(),'Refinement output exists; no automatic rerun'
    blob=SOURCE.read_bytes();pin=hashlib.sha256(blob).hexdigest()
    assert pin=='94fbbb3cb88a15147d33ab04c8f791470dd92e3d73cf8cba4afc0f361cd583fc'
    OUTPUT.mkdir();declaration=dict(dt_years=STEPS,horizon_years=HORIZON,cap_seconds=CAP_SECONDS,source_sha256=pin,parameters={**OPTION,'mode':MODE})
    (OUTPUT/'declaration.json').write_text(json.dumps(declaration,indent=2)+'\n')
    source=SimplifiedGenesis.load(SOURCE);target=source.state['time_myr']+HORIZON/1e6
    configs=deepcopy(source.config);configs['experimental_geological_edge_fatigue']=declaration['parameters'];runs=[]
    for dt in STEPS:
        m=SimplifiedGenesis(configs,source.state);rows=[];status='COMPLETED';last=None;first_diffuse=None;run_start=time.monotonic()
        for i in range(round(HORIZON/dt)):
            if time.monotonic()-start>CAP_SECONDS:status='BUDGET_STOP';break
            years=(target-m.state['time_myr'])*1e6 if i==round(HORIZON/dt)-1 and dt<25 else dt
            before=deepcopy(m.state)
            try:m.advance(years)
            except ValueError as e:
                assert encoded(before)==encoded(m.state);status='GUARD_STOP: '+str(e);break
            last=(before,years);s=m.state;L=s['last_ledger'];f=s['experimental_edge_fatigue'];d,shear=field(m)
            deltaH=np.sum(np.array(s['energy'],np.longdouble)-np.array(before['energy'],np.longdouble));deltaR=s['experimental_edge_fracture_reserve_j']-before.get('experimental_edge_fracture_reserve_j',0.)
            residual=deltaH+L['elastic_after']-L['elastic_before']+L['kinetic_after']-L['kinetic_before']+deltaR-L['source_energy']-L['external_work']+L['numerical_memory']+L['numerical_kinetic']
            assert abs(residual)<=L['whole_allowance_j'];assert L['edge_damage_heat_j']==0
            elapsed=(s['time_myr']-source.state['time_myr'])*1e6
            if f['reason']=='DIFFUSE_NETWORK_FAILURE' and first_diffuse is None:first_diffuse=elapsed
            rows.append(dict(years=elapsed,actual_dt_years=years,saturated=int(sum(d>=1)),reason=f['reason'],max_damage=float(d.max()),mean_damage=float(d.mean()),
                max_shear_before_update_pa=max(f['shear_traction_pa']),max_shear_after_update_pa=float(shear.max()),
                max_damage_increment=float(np.max(d-np.array(before.get('experimental_edge_fatigue',{}).get('damage',np.zeros(len(d)))))),
                energy_fraction=float(abs(residual)/L['whole_allowance_j']),release_j=L['edge_fracture_release_j']))
            if f['partition_owner'] is not None:status='HANDOFF_PENDING';break
        restart=None
        if last:
            replay=SimplifiedGenesis(configs,last[0]);replay.advance(last[1]);restart=encoded(m.state)==encoded(replay.state);assert restart
        name=f'dt-{dt:g}';m.save(OUTPUT/(name+'.json'));d,shear=field(m)
        if status=='COMPLETED':assert m.state['time_myr']==target
        runs.append(dict(dt_years=dt,status=status,accepted_steps=len(rows),start_time_myr=source.state['time_myr'],end_time_myr=m.state['time_myr'],
            first_diffuse_years=first_diffuse,domains=len(set(m.state['owner'])),newborn=len(m.state['born_material']),
            damage=d.tolist(),shear_after_update_pa=shear.tolist(),saturated_indices=np.flatnonzero(d>=1).tolist(),
            max_damage=float(d.max()),mean_damage=float(d.mean()),max_shear_after_pa=float(shear.max()),
            reserve_j=m.state['experimental_edge_fracture_reserve_j'],max_energy_fraction=max(x['energy_fraction'] for x in rows),
            restart_exact=restart,mass_exact=m.config['mass']==source.config['mass'],checkpoint_sha256=hashlib.sha256((OUTPUT/(name+'.json')).read_bytes()).hexdigest(),
            seconds=time.monotonic()-run_start,rows=rows))
        print(json.dumps({k:v for k,v in runs[-1].items() if k not in ('rows','damage','shear_after_update_pa') }),flush=True)
        if status=='BUDGET_STOP':break
    assert SOURCE.read_bytes()==blob
    completed=[r for r in runs if r['status']=='COMPLETED']
    result=dict(declaration=declaration,target_time_myr=target,source_unchanged=True,runs=runs,
        comparisons=[comparison(a,b) for a,b in zip(completed,completed[1:])],seconds=time.monotonic()-start)
    (ROOT/'edge_weakening_timestep.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
