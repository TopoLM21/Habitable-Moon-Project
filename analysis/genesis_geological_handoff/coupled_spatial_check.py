"""One nested mesh pair, matched short interval, corrected/reduced bracket."""
from copy import deepcopy
from pathlib import Path
import json,time
import numpy as np
from tectonics.experimental_angular_coupled import AngularCoupled,base_face_energy
from tectonics.experimental_edge_weakening import scale,face_damage
from tectonics.genesis_transfer_bridge import encoded,digest
from .coupled_refine import refine
from .angular_coupled_check import world_stress
from .edge_spatial_audit import ROOT

OUT=Path('/workspace/genesis-coupled-spatial-01')
SOURCE=Path('/workspace/genesis-angular-coupled-01/initial.json')
ALT=Path('/workspace/genesis-source-angular-01/alternative_initial.json')


def field(m):
    return dict(stress=world_stress(m),T=m.temperature(m.state['energy']),H=np.array(m.state['energy']),
        oldD=np.array(m.state['damage']),faceD=face_damage(m.mesh,m.state['experimental_edge_fatigue']['damage']))


def restrict_fields(fine):
    n=(len(fine['T'])-2)//4
    return dict(stress=fine['stress'].reshape(80,4,3,3).mean(axis=1),
        T=np.r_[fine['T'][:4*n].reshape(n,4).mean(axis=1),fine['T'][-2:]],
        H=np.r_[fine['H'][:4*n].reshape(n,4).sum(axis=1),fine['H'][-2:]],
        oldD=fine['oldD'].reshape(80,4).mean(axis=1),faceD=fine['faceD'].reshape(80,4).mean(axis=1))


def compare(coarse,fine,volume):
    delta=coarse['stress']-fine['stress'];den=np.einsum('f,fij,fij->',volume,fine['stress'],fine['stress'])
    return dict(world_stress_relative=float(np.sqrt(np.einsum('f,fij,fij->',volume,delta,delta)/den)),
        temperature_max_difference_k=float(abs(coarse['T']-fine['T']).max()),
        thermal_H_relative_l2=float(np.linalg.norm(coarse['H']-fine['H'])/np.linalg.norm(fine['H'])),
        old_damage_max_difference=float(abs(coarse['oldD']-fine['oldD']).max()),
        face_damage_max_difference=float(abs(coarse['faceD']-fine['faceD']).max()))


def energy(m):return float(base_face_energy(m)@scale(m,m.state['damage']))


def main():
    if OUT.exists():raise FileExistsError('Preserve previous spatial results')
    start=time.monotonic();OUT.mkdir();blob=SOURCE.read_bytes();altblob=ALT.read_bytes()
    a=AngularCoupled.load(SOURCE);coarse_initial=field(a.model);U0=energy(a.model)
    fine,P,audit=refine(a);fine_initial=field(fine.model)
    audit.update(initial_fields=compare(coarse_initial,restrict_fields(fine_initial),a.model.G[:,0,0]),
        elastic_relative=(energy(fine.model)-U0)/U0,source_sha256=digest(blob))
    assert max(audit['mass_matrix_restriction_relative'],abs(audit['elastic_relative']),audit['initial_fields']['world_stress_relative'])<1e-10
    assert audit['mass_sum_relative']==0 and audit['enthalpy_sum_relative']==0
    fine.save(OUT/'fine_initial.json')
    snapshots={};runs={};fields={};deadline=time.monotonic()+180
    # The reduced branch is an isolated diagnostic, never saved as a production restart.
    for name,seed,reduced in [('coarse_full',a,False),('fine_full',fine,False),('coarse_reduced',a,True),('fine_reduced',fine,True)]:
        model=deepcopy(seed);m=model.model
        if reduced:model.reference.L[:]=model.reference.L.mean(axis=1)[:,None];m._angular_reference=model.reference
        rows=[];trace={}
        for i in range(2):
            if time.monotonic()>deadline:raise TimeoutError('180second solve budget')
            tick=time.monotonic();L=model.advance(.0625);elapsed=time.monotonic()-tick
            rows.append(dict(years=(i+1)*.0625,seconds=elapsed,whole_allowance_fraction=abs(L['whole_balance_j'])/L['whole_allowance_j'],
                mechanical_balance=L['moment_balance_relative'],momentum=L['momentum_relative'],births=len(m.state.get('born_material',[])),
                edge_saturated=sum(m.state['experimental_edge_fatigue']['failed_edges']),certificate=m.state['experimental_edge_fatigue']['reason']))
            trace[str((i+1)*.0625)]=field(m)
            if i==0:print(json.dumps(dict(run=name,first_step_seconds=elapsed,estimated_two_steps=2*elapsed)),flush=True)
        runs[name]=dict(faces=m.nf,reduced_diagnostic=reduced,rows=rows);fields[name]=trace
        if name=='fine_full':
            model.save(OUT/'fine_final.json');fine_direct=model
    # One predeclared alternative, already documented in source initialization audit.
    alternative=AngularCoupled.from_source_checkpoint(ALT);altfine,_,altaudit=refine(alternative)
    altaudit['initial_difference_same_fine']=compare(field(altfine.model),fine_initial,fine.model.G[:,0,0])
    altaudit['elastic_relative_transfer']=(energy(altfine.model)-energy(alternative.model))/energy(alternative.model)
    rows=[];trace={}
    for i in range(2):
        if time.monotonic()>deadline:raise TimeoutError('180second solve budget')
        tick=time.monotonic();L=altfine.advance(.0625)
        rows.append(dict(years=(i+1)*.0625,seconds=time.monotonic()-tick,whole_allowance_fraction=abs(L['whole_balance_j'])/L['whole_allowance_j'],births=len(altfine.model.state.get('born_material',[]))))
        trace[str((i+1)*.0625)]=field(altfine.model)
    fields['fine_alternative']=trace;runs['fine_alternative']=dict(faces=320,rows=rows)
    altfine.save(OUT/'fine_alternative_final.json')
    comparisons={}
    for t in fields['coarse_full']:
        comparisons[t]=dict(corrected=compare(fields['coarse_full'][t],restrict_fields(fields['fine_full'][t]),a.model.G[:,0,0]),
            reduced=compare(fields['coarse_reduced'][t],restrict_fields(fields['fine_reduced'][t]),a.model.G[:,0,0]),
            same_fine_initialization=compare(fields['fine_alternative'][t],fields['fine_full'][t],fine.model.G[:,0,0]),
            integration_change_coarse=compare(fields['coarse_reduced'][t],fields['coarse_full'][t],a.model.G[:,0,0]),
            integration_change_fine=compare(fields['fine_reduced'][t],fields['fine_full'][t],fine.model.G[:,0,0]))
    # Fresh-file restart of the corrected fine solution, one extra step only.
    resumed=AngularCoupled.load(OUT/'fine_final.json');direct=fine_direct
    resumed.advance(.0625);direct.advance(.0625);restart=encoded(resumed.model.state)==encoded(direct.model.state);assert restart
    np.savez_compressed(OUT/'fields.npz',**{f'{name}_{t}_{k}':v for name,trace in fields.items() for t,f in trace.items() for k,v in f.items()})
    spatial_gate=all(x['corrected']['world_stress_relative']<.05 and x['corrected']['face_damage_max_difference']<.02 and x['corrected']['temperature_max_difference_k']<.05 for x in comparisons.values())
    spatial_gate = spatial_gate and restart and all(
        a['births']==4*b['births'] and a['whole_allowance_fraction']<=1 and b['whole_allowance_fraction']<=1
        for a,b in zip(runs['fine_full']['rows'],runs['coarse_full']['rows']))
    assert SOURCE.read_bytes()==blob and ALT.read_bytes()==altblob
    result=dict(code_base='77ef6c3d80565d2d43035724d159eabd5d66a3a0',scope='80/320 nested coplanar physical geometry, coupled.125year dt.0625, fixed laws',
        primary_transfer=audit,alternative_transfer=altaudit,runs=runs,comparisons=comparisons,
        spatial_gate_passed=spatial_gate,restart_exact=restart,longer_pilot_run=False,
        files={p.name:dict(bytes=p.stat().st_size,sha256=digest(p.read_bytes())) for p in OUT.iterdir()},output=str(OUT),seconds=time.monotonic()-start)
    (ROOT/'coupled_spatial_check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('files','runs')},indent=2))


if __name__=='__main__':main()
