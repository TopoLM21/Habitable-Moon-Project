"""Close a finite short-time accuracy question; bounded refinement, fixed physics."""
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
OUTPUT=Path('/workspace/genesis-edge-short-accuracy-01')
BUDGET=180.
CRITERION=dict(traction_max_vector_difference_over_fine_max=.05,damage_max_absolute_difference=.02,
    saturated_masks_equal=True,topology_status_equal=True,scope='all common temporal nodes, fixed25year experiment; no universal physical accuracy claim')


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fields(m):
    d=np.array(m.state['experimental_edge_fatigue']['damage']);eff=np.maximum(m.state['damage'],face_damage(m.mesh,d))
    return d,shear_vectors(m,tensors(m,damage=eff))


def run(dt,source,c,target,deadline):
    m=SimplifiedGenesis(c,source.state);start=time.monotonic();rows=[];ds=[];ts=[];last=None;status='COMPLETED'
    count=round(25/dt)
    for i in range(count):
        if time.monotonic()>deadline:status='BUDGET_STOP';break
        years=(target-m.state['time_myr'])*1e6 if i==count-1 else dt
        before=deepcopy(m.state)
        try:m.advance(years)
        except ValueError as e:
            assert encoded(before)==encoded(m.state);status='GUARD_STOP: '+str(e);break
        last=(before,years);s=m.state;L=s['last_ledger'];d,t=fields(m);ds.append(d);ts.append(t)
        deltaH=np.sum(np.array(s['energy'],np.longdouble)-np.array(before['energy'],np.longdouble));deltaR=s['experimental_edge_fracture_reserve_j']-before.get('experimental_edge_fracture_reserve_j',0.)
        residual=deltaH+L['elastic_after']-L['elastic_before']+L['kinetic_after']-L['kinetic_before']+deltaR-L['source_energy']-L['external_work']+L['numerical_memory']+L['numerical_kinetic']
        assert abs(residual)<=L['whole_allowance_j'];assert L['edge_damage_heat_j']==0
        rows.append(dict(years=(i+1)*dt,time_myr=s['time_myr'],saturated=int(sum(d>=1)),reason=s['experimental_edge_fatigue']['reason'],
            max_damage=float(d.max()),mean_damage=float(d.mean()),max_shear_pa=float(np.linalg.norm(t,axis=1).max()),
            reserve_j=s['experimental_edge_fracture_reserve_j'],energy_fraction=float(abs(residual)/L['whole_allowance_j'])))
        if s['experimental_edge_fatigue']['partition_owner'] is not None:status='HANDOFF_PENDING';break
    restart=None
    if last:
        replay=SimplifiedGenesis(c,last[0]);replay.advance(last[1]);restart=encoded(replay.state)==encoded(m.state);assert restart
    if status=='COMPLETED':assert m.state['time_myr']==target
    name=f'dt-{dt:g}';checkpoint=OUTPUT/(name+'.json');m.save(checkpoint)
    trace=OUTPUT/(name+'-fields.npz');np.savez_compressed(trace,damage=np.array(ds),shear_vectors_pa=np.array(ts),years=np.array([r['years'] for r in rows]))
    source_match=None
    if dt==.5:
        saved=SimplifiedGenesis.load('/workspace/genesis-edge-weakening-dt-01/dt-0.5.json')
        source_match=encoded(m.state)==encoded(saved.state) and encoded(m.config)==encoded(saved.config);assert source_match
    result=dict(dt_years=dt,status=status,accepted_steps=len(rows),start_time_myr=source.state['time_myr'],end_time_myr=m.state['time_myr'],
        checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),trace=str(trace),trace_sha256=digest(trace),
        restart_exact=restart,previous_halfyear_endpoint_exact=source_match,mass_exact=m.config['mass']==source.config['mass'],
        domains=len(set(m.state['owner'])),newborn=len(m.state['born_material']),max_energy_fraction=max(r['energy_fraction'] for r in rows),
        seconds=time.monotonic()-start,rows=rows)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
    return result,np.array(ds),np.array(ts)


def compare(a,b):
    ar,ad,at=a;br,bd,bt=b;ratio=round(ar['dt_years']/br['dt_years']);rows=[]
    for i,r in enumerate(ar['rows']):
        j=(i+1)*ratio-1;fine=br['rows'][j];den=max(float(np.linalg.norm(bt[j],axis=1).max()),1.)
        damage=float(np.max(abs(ad[i]-bd[j])));traction=float(np.linalg.norm(at[i]-bt[j],axis=1).max()/den)
        mask=bool(np.array_equal(ad[i]>=1,bd[j]>=1));topology=r['reason']==fine['reason']
        rows.append(dict(years=r['years'],damage_normalized_difference=damage,traction_normalized_difference=traction,
            saturated_masks_equal=mask,topology_status_equal=topology,
            reserve_relative_difference=abs(r['reserve_j']-fine['reserve_j'])/max(abs(fine['reserve_j']),1.)))
    wd=max(rows,key=lambda r:r['damage_normalized_difference']);wt=max(rows,key=lambda r:r['traction_normalized_difference'])
    passed=(wd['damage_normalized_difference']<=CRITERION['damage_max_absolute_difference']
        and wt['traction_normalized_difference']<=CRITERION['traction_max_vector_difference_over_fine_max']
        and all(r['saturated_masks_equal'] and r['topology_status_equal'] for r in rows))
    return dict(coarse_dt=ar['dt_years'],fine_dt=br['dt_years'],criterion_pass=passed,worst_damage=wd,worst_traction=wt,
        max_reserve_relative_difference=max(r['reserve_relative_difference'] for r in rows),common_time_count=len(rows),rows=rows)


def main():
    assert not OUTPUT.exists(),'Do not overwrite or automatically repeat this finite study'
    start=time.monotonic();deadline=start+BUDGET;OUTPUT.mkdir()
    declaration=dict(base_commit='8c588d60cf8a1384b06c326860a6cf2b32599d64',sequence=[.5,.25],conditional_final_dt=.125,
        criterion=CRITERION,horizon_years=25.,budget_seconds=BUDGET,parameters={**OPTION,'mode':MODE})
    (OUTPUT/'declaration.json').write_text(json.dumps(declaration,indent=2)+'\n')
    source_pin=digest(SOURCE);assert source_pin=='94fbbb3cb88a15147d33ab04c8f791470dd92e3d73cf8cba4afc0f361cd583fc'
    old='/workspace/genesis-edge-weakening-dt-01/dt-0.5.json';old_pin=digest(old)
    source=SimplifiedGenesis.load(SOURCE);c=deepcopy(source.config);c['experimental_geological_edge_fatigue']=declaration['parameters'];target=source.state['time_myr']+25/1e6
    a=run(.5,source,c,target,deadline);b=run(.25,source,c,target,deadline);runs=[a[0],b[0]];pairs=[]
    if a[0]['status']==b[0]['status']=='COMPLETED':
        first=compare(a,b);pairs.append(first);print(json.dumps({k:v for k,v in first.items() if k!='rows'}),flush=True)
        if not first['criterion_pass'] and time.monotonic()<deadline:
            final=run(.125,source,c,target,deadline);runs.append(final[0])
            if final[0]['status']=='COMPLETED':pairs.append(compare(b,final))
    assert digest(SOURCE)==source_pin and digest(old)==old_pin
    result=dict(declaration=declaration,source_sha256=source_pin,originals_unchanged=True,target_time_myr=target,
        runs=runs,comparisons=pairs,seconds=time.monotonic()-start)
    (ROOT/'edge_short_accuracy.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(seconds=result['seconds'],final_comparison={k:v for k,v in pairs[-1].items() if k!='rows'} if pairs else None)),flush=True)


if __name__=='__main__':main()
