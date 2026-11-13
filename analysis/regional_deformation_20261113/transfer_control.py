"""Externally funded exchange controls, kept separate from unforced histories."""
from decimal import Decimal,localcontext
from dataclasses import replace
import json,time,subprocess,math
import numpy as np
from verify import setup,HERE,hashes
from tectonics.deformation_exchange import DeformationExchange


def total_energy(obj):
    with localcontext() as ctx:
        ctx.prec=70
        return sum(Decimal(x) for key in ('energy_m_j','energy_c_j') for x in obj.material.data[key])+Decimal(obj.surface.budget()['potential_j'])


def control(name,dt):
    output=HERE/f'{name}.json';saved=HERE/'states'/name
    if output.exists() or saved.exists():raise FileExistsError(name)
    started=time.perf_counter();obj=setup('native',3,1e6,dt);obj.advance(2.)
    initial=obj.diagnostics();snapshot=obj.checkpoint();energy_before=total_energy(obj)
    volume=np.stack([obj.surface.arrays[k+'_volume_m3'] for k in ('felsic','basalt')],axis=1)
    # Two explicit, spatially heterogeneous but grid-independent factors.
    axis=np.array([1.,2.,3.])/np.sqrt(14)
    factor=(1+obj.surface.mesh().centroids@axis)/2
    born=volume*.02*factor[:,None];returned=volume*.01*(1-factor[:,None])
    try:obj.material_request(born,returned)
    except ValueError as exc:rejection=str(exc)
    else:raise AssertionError('Unfunded creation accepted')
    assert obj.checkpoint()==snapshot
    event=obj.material_request(born,returned,external_work_available_j=1e28)
    difference=total_energy(obj)-energy_before
    high_precision_work_residual=float(difference-Decimal(event['external_placement_work_j']))
    obj.save(HERE/'states'/f'{name}_event')
    recovered=DeformationExchange.load(HERE/'states'/f'{name}_event')
    assert recovered.checkpoint()==obj.checkpoint()
    obj.advance(10.);recovered.advance(10.)
    assert recovered.checkpoint()==obj.checkpoint()
    obj.save(saved)
    report=dict(case=name,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        runtime_sha256=hashes(),dt_myr=dt,initial=initial,event=event,unfunded_rejection=rejection,
        no_mutation_on_rejection=True,high_precision_event_work_residual_j=high_precision_work_residual,
        final=obj.diagnostics(),restart_exact=True,wall_s=time.perf_counter()-started,
        scope='prescribed regional transfer at 2 Myr with explicit external placement work; not spontaneous crust production or subduction')
    with output.open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps(dict(case=name,wall_s=report['wall_s'],work_j=event['external_placement_work_j'],
        event_residual_j=high_precision_work_residual,final_residual_j=report['final']['combined_energy_residual_j'])),flush=True)


if __name__=='__main__':
    control('transfer_dt1',1.)
    control('transfer_dt0.5',.5)
