"""Predeclared bounded check: 320 faces, ten half-year steps, one thermal cycle.

Physical traction/material/edge law unchanged. Temperature is prescribed solely
as a mechanical eigenstrain control. No real planetary checkpoint is advanced.
"""
import hashlib,json,time
from pathlib import Path
import numpy as np
from tectonics.simplified_genesis import SimplifiedGenesis
from .angular_maxwell import AngularMaxwell
from .edge_spatial_audit import AuditGrid, ROOT
from .base_operator_audit import SOURCE,SOURCE_SHA256

OUT=Path('/workspace/genesis-angular-maxwell-check-01')


def collapse_evidence(model,state):
    memory=np.asarray(state['memory'])
    collapsed=np.repeat(memory.mean(axis=2)[:,:,None,:],3,axis=2)
    full=model.energy(memory,state['edge_damage'])
    reduced=model.energy(collapsed,state['edge_damage'])
    return dict(full_energy_j=full,angular_mean_only_energy_j=reduced,
        lost_energy_j=full-reduced,lost_fraction=(full-reduced)/full,
        scope='Known new fixture only; this does not estimate unknown variance in the old real source')


def main():
    if OUT.exists():raise FileExistsError('Preserve existing angular checkpoint directory')
    assert hashlib.sha256(Path(SOURCE).read_bytes()).hexdigest()==SOURCE_SHA256
    start=time.monotonic();deadline=start+60
    source=SimplifiedGenesis.load(SOURCE)
    coeff=np.array(json.loads((ROOT/'edge_spatial_control.json').read_text())['coefficients'])
    grid=AuditGrid(2,False,coeff)
    m=AngularMaxwell(grid,damage_feedback=True,expansion_per_k=source.shell['linear_expansion_per_k'])
    s=m.initial();OUT.mkdir();m.save(s,OUT/'initial.json')
    # Known smooth angular temperature increment; exactly reversed on step2.
    x=m.points/np.linalg.norm(m.points,axis=-1)[...,None]
    temperature=.01*(1+x[...,0]*x[...,1])
    rows=[];restart_exact=False;restart_state=None
    for i in range(10):
        if time.monotonic()>deadline:raise TimeoutError('Predeclared60second check cap')
        tick=time.monotonic()
        delta=temperature if i==0 else -temperature if i==1 else 0.
        s,row=m.step(s,.5,temperature_increment_k=delta)
        elapsed=time.monotonic()-tick
        cumulative=s['cumulative']
        global_residual=(m.energy(s['memory'],s['edge_damage'])-s['initial_elastic_j']
            +.5*np.asarray(s['velocity'])@(m.M@np.asarray(s['velocity']))-s['initial_kinetic_j']
            +s['fracture_reserve_j']+cumulative['maxwell_j']+cumulative['numerical_memory_j']
            +cumulative['numerical_kinetic_j']-cumulative['external_work_j']-cumulative['thermal_work_j'])
        global_relative=abs(global_residual)/max(abs(cumulative['external_work_j']),abs(cumulative['thermal_work_j']),1.)
        assert global_relative<5e-8
        row.update(time_years=s['time_years'],seconds=elapsed,global_balance_relative=float(global_relative),
            reserve_j=s['fracture_reserve_j'],thermal_work_j=s['last_ledger']['thermal_work_j'])
        rows.append(row)
        if i==0:
            # Extrapolation only. Factor16 is face count; factor64 allows fill growth.
            estimate=dict(faces=5120,measured_320_first_step_seconds=elapsed,
                estimated_step_seconds=[elapsed*16,elapsed*64],
                B_weights_Kbase_Cbase_bytes=5120*(6*6*18+6+18*18+6*6)*8,
                memory_velocity_displacement_temperature_damage_bytes=(5120*(36+6)+2*(6*(5120//2+2))+3*5120//2)*8,
                limitation='Not peak RSS; retained GridControl, sparse matrices/factorization and JSON temporaries excluded. No5120run.')
            print(json.dumps(dict(preflight_5120_estimate=estimate)),flush=True)
        if i==4:
            m.save(s,OUT/'step5.json');restart_state=m.load(OUT/'step5.json')
        if i==5:
            restarted,_=m.step(restart_state,.5)
            restart_exact=restarted==s;assert restart_exact
    assert np.max(abs(np.array(s['temperature_change_k'])))==0
    assert s['fracture_reserve_j']>0 and s['cumulative']['fracture_heat_j']==0
    m.save(s,OUT/'final.json')
    result=dict(code_base='85bc8585cdecd5d0c05ec74b2c2b26131f243eab',source_sha256=SOURCE_SHA256,
        scope='Known zero initial memory/velocity;320faces,10x.5years; thermal cycle +/-0.01*(1+x*y)K; same fixed traction and damage law',
        schema=s['schema'],operator_sha256=m.fingerprint,rows=rows,restart_exact=restart_exact,
        thermal_cycle_closed_exact=True,temperature_is_prescribed_not_enthalpy_evolution=True,
        estimate_5120=estimate,checkpoint_files={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in OUT.glob('*.json')},
        output_directory=str(OUT),cumulative=s['cumulative'],collapse_evidence=collapse_evidence(m,s),seconds=time.monotonic()-start)
    (ROOT/'angular_maxwell_check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
