"""Bounded real-source initialization, temporal and one-field sensitivity check."""
from pathlib import Path
import json,time
import numpy as np
from tectonics.genesis_transfer_bridge import encoded,digest
from .source_angular import SourceAngular
from .edge_spatial_audit import ROOT,SOURCE
from .base_operator_audit import SOURCE_SHA256

OUT=Path('/workspace/genesis-source-angular-01')


def pointwise_stress_change(model,initial,alternative):
    m=model.m
    def stress(s):
        coefficients=np.asarray(s['memory']).reshape(m.nf,3,2,6)
        strain=np.einsum('fri,faij->farj',m.P,coefficients)
        C=m.C*m.scale(m.state['damage'])[:,None,None]
        return np.einsum('fij,farj->fari',C,strain)
    base=stress(initial);difference=stress(alternative)-base
    norm=lambda a:np.einsum('fr,fari,i,fari->',m.weights,a,np.array([1,1,1,2,2,2]),a)/3
    return dict(alternative_relative_full_tensor_l2=float(np.sqrt(norm(difference)/norm(base))),
        alternative_max_component_change_pa=float(abs(difference).max()),
        scope='Source radial nodes x three angular points; tensor Frobenius, original volume weights; mean may remain unchanged')


def main():
    if OUT.exists():raise FileExistsError('Preserve existing source-derived checkpoints')
    blob=SOURCE.read_bytes();assert digest(blob)==SOURCE_SHA256
    start=time.monotonic();deadline=start+120
    model=SourceAngular(json.loads(blob));OUT.mkdir()
    initial=model.initial();alternate=model.initial(True)
    model.save(initial,OUT/'initial.json');model.save(alternate,OUT/'alternative_initial.json')
    runs={};profiles={};restart_exact=False
    for name,dt,state in [('coarse',.125,initial),('fine',.0625,initial),('alternative',.0625,alternate)]:
        rows=[];fields={}
        for i in range(round(1/dt)):
            if time.monotonic()>deadline:raise TimeoutError('Predeclared120second budget')
            before=state;clock=time.monotonic();state,row=model.step(state,dt);row['seconds']=time.monotonic()-clock;rows.append(row)
            if not runs and i==0:
                print(json.dumps(dict(first_step_seconds=row['seconds'],estimated_40steps_seconds=row['seconds']*40,
                    M_L_memory_bytes=model.M.nbytes+model.L.nbytes+np.asarray(state['memory']).nbytes)),flush=True)
            if abs((i+1)*dt/.125-round((i+1)*dt/.125))<1e-10:
                fields[str(round(state['elapsed_years'],6))]=(model.energy_force_stress(state['memory'],state['edge_damage'])[2],np.array(state['edge_damage']))
            if name=='fine' and i==7:
                model.save(before,OUT/'restart_before.json');resumed,old=SourceAngular.load(OUT/'restart_before.json')
                new,_=resumed.step(old,dt);restart_exact=encoded(new)==encoded(state);assert restart_exact
            if state['certificate']=='CLOSED_CONNECTED_BOUNDARIES':break
        model.save(state,OUT/(name+'_final.json'));profiles[name]=fields
        runs[name]=dict(dt=dt,rows=rows,final_time_myr=state['time_myr'],final_memory_angular_variance_energy_j=(model.energy_force_stress(state['memory'],state['edge_damage'])[0]-model.energy_force_stress(np.repeat(np.asarray(state['memory']).mean(axis=1)[:,None],3,axis=1),state['edge_damage'])[0]))
    comparisons={}
    for name in ('coarse','alternative'):
        common=sorted(set(profiles[name])&set(profiles['fine']),key=float);rows=[]
        for t in common:
            stress,damage=profiles[name][t];reference,refdamage=profiles['fine'][t]
            area=model.m.mesh.areas_unit_sphere
            relative=float(np.sqrt(np.einsum('f,fij,fij->',area,stress-reference,stress-reference)/np.einsum('f,fij,fij->',area,reference,reference)))
            rows.append(dict(years=float(t),stress_relative=relative,damage_max_absolute=float(abs(damage-refdamage).max())))
        comparisons[name]=rows
    assert SOURCE.read_bytes()==blob
    assert digest(encoded(dict(config=model.m.config,state=model.m.state)))==model.envelope['sha256']
    result=dict(code_base='81e064eb7ec94238de69370073617bd62182363f',source_sha256=SOURCE_SHA256,
        source_unchanged=True,scope='Real-source initialized frozen thermal/old-D/front mechanical slice; not full Genesis evolution',
        projection=model.projection_audit,primary_mapping=initial['mapping'],alternative_mapping=alternate['mapping'],
        pointwise_stress_mapping=pointwise_stress_change(model,initial,alternate),
        alternative_initialization=alternate['initialization'],runs=runs,comparisons=comparisons,restart_exact=restart_exact,
        temporal_gate_passed=max(x['stress_relative'] for x in comparisons['coarse'])<.05 and max(x['damage_max_absolute'] for x in comparisons['coarse'])<.02,
        files={p.name:dict(bytes=p.stat().st_size,sha256=digest(p.read_bytes())) for p in OUT.glob('*.json')},
        output=str(OUT),seconds=time.monotonic()-start)
    (ROOT/'source_angular_check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('projection','runs','files')},indent=2))


if __name__=='__main__':main()
