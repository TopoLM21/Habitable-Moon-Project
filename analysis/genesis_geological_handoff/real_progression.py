"""One bounded real-source progression. No parameter edits or forced splitting."""
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np

from analysis.genesis_adaptive_control.controller import time_limit
from analysis.genesis_geological_handoff.experiment import exact_saved_state
from tectonics.genesis_transfer_bridge import encoded
from tectonics.genesis_implicit_enthalpy import phase
from tectonics.genesis_geological_thermal import ThermalGeologicalContinuation, eligibility
from tectonics.simplified_genesis import SimplifiedGenesis
from tectonics.mesh import connected_components
from tectonics.genesis import GenesisParameters,vapor_column_kg_m2


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def observe(model, elapsed):
    s,c=model.state,model.config; T=model.temperature(s['energy']); D=np.array(s['damage'])
    nodes=model.solid_nodes; above=D>=c['faults']['activation_damage']
    persistent=above & (np.array(s['candidate_age'])>=c['faults']['activation_persistence_myr']) & ~np.array(s['consumed'])
    components=connected_components(np.flatnonzero(above),model.mesh.neighbors)
    areas=model.mesh.physical_cell_areas_km2(c['radius_km'])
    crystals=model.mass*(1-phase(T,c['solidus'],c['liquidus']))
    L=s.get('last_ledger',{}); P=L.get('phase',{})
    p=GenesisParameters(**c['thermal_parameters'])
    liquid=model.mass[-1]-vapor_column_kg_m2(T[-1],p)*p.area_m2
    extra={}
    if c.get('mechanical_mode')=='shared_nodal_moments':
        X=np.asarray(c['nodal']['X']);du=(np.asarray(s['nodal_displacement'])-np.asarray(c['nodal']['initial_displacement'])).reshape(-1,6)[:,:3]
        radial=np.einsum('ni,ni->n',du,X)/np.sum(X*X,axis=1)
        extra.update(radial_strain_mean=float(radial.mean()),radial_strain_min=float(radial.min()),radial_strain_max=float(radial.max()),nodal_momentum_relative=L.get('momentum_relative'))
        if 'stress_pa' in s:
            from tectonics.genesis_shell import principal_tensile
            tensile=principal_tensile(np.asarray(s['stress_pa']))
            extra.update(tensile_mean_pa=float(tensile.mean()),tensile_std_pa=float(tensile.std()),tensile_max_pa=float(tensile.max()))
    return dict(**extra,liquid_water_mass_kg=float(liquid),persistent_covers_all_faces=bool(persistent.all()),time_myr=s['time_myr'],wall_seconds=elapsed,steps=s['steps'],domains=len(s['omega']),
        solid_min_k=float(T[nodes].min()),solid_mean_k=float(T[nodes].mean()),solid_max_k=float(T[nodes].max()),
        deep_k=float(T[-2]),water_k=float(T[-1]),
        resolved_crystal_mass_kg=float(crystals[:-2].sum()),deep_crystal_mass_kg=float(crystals[-2]),
        load_bearing_mass_kg=math.fsum(c['face_mass_kg'])+math.fsum(x['mass_kg'] for x in s.get('born_material',[])),
        total_silicate_mass_kg=float(model.mass[:-1].sum()),born_cells=len(s.get('born_material',[])),
        damage_min=float(D.min()),damage_mean=float(D.mean()),damage_max=float(D.max()),
        above_threshold_cells=int(above.sum()),persistent_eligible_cells=int(persistent.sum()),
        above_threshold_components=len(components),
        largest_above_threshold_component_area_km2=max((float(areas[x].sum()) for x in components),default=0.),
        max_candidate_age_years=float(max(s['candidate_age'])*1e6),
        whole_energy_residual_j=L.get('whole_balance_j'),whole_energy_allowance_j=L.get('whole_allowance_j'),
        moment_balance_relative=L.get('moment_balance_relative'),
        new_solid_crystal_mass_kg=P.get('new_phase_solid_kg'),mush_remelted_kg=P.get('mush_remelted_kg'),
        new_load_bearing_cells=P.get('new_load_bearing_cells'),added_latent_heat_j=P.get('added_latent_heat_j'))


def independent_phase_audit(model,before):
    c,s=model.config,model.state; rows=c.get('phase_cells',[])
    catalog={r['id']:r for r in rows}; previous={r['cell_id'] for r in before.get('born_material',[])}
    born=s.get('born_material',[]); ids=[r['cell_id'] for r in born]
    if len(set(ids))!=len(ids) or not previous.issubset(ids):raise ValueError('Independent phase ID audit')
    T=model.temperature(s['energy']); oldT=model.temperature(before['energy'])
    for r in born:
        origin=catalog[r['cell_id']]
        if r['mass_kg']!=origin['mass_kg'] or r['node']!=origin['node'] or T[r['node']]>c['solidus']:
            raise ValueError('Independent unsupported remelt/false solid birth')
        if r['cell_id'] not in previous:
            if r['birth_time_myr']!=s['time_myr'] or oldT[r['node']]<=c['solidus']:
                raise ValueError('Independent false whole-cell birth timing')
    changed=math.fsum(r['mass_kg']*(float(phase(oldT[r['node']],c['solidus'],c['liquidus']))-
                     float(phase(T[r['node']],c['solidus'],c['liquidus']))) for r in rows)
    ledger=s['last_ledger']['phase']; expected=ledger['new_phase_solid_kg']-ledger['mush_remelted_kg']
    # Floating point phase inventory audit, not a time-discretization certificate.
    allowance=64*np.finfo(float).eps*math.fsum(r['mass_kg'] for r in rows)
    if abs(changed-expected)>allowance:raise ValueError('Independent crystal mass change mismatch')
    if ledger['new_load_bearing_cells']!=len(set(ids)-previous) or ledger['added_latent_heat_j']!=0:
        raise ValueError('Independent duplicated phase birth/latent heat')
    if abs(s['last_ledger']['whole_balance_j'])>s['last_ledger']['whole_allowance_j']:
        raise ValueError('Whole energy audit failed')
    return dict(crystal_mass_residual_kg=changed-expected,crystal_mass_allowance_kg=allowance,
                unique_born_ids=True,no_unsupported_remelt=True,no_added_latent_heat=True)


def checkpoint(model,root):
    # Only rotate files owned by this fresh experiment; no previous experiment touched.
    temp=root/'checkpoint.pending.json';model.save(temp);temp.replace(root/'checkpoint.json')


def run(source,output,budget_seconds=280.,*,condensation=False,step_years=None):
    if not 5<budget_seconds<=280:raise ValueError('Budget must be >5 and <=280 seconds')
    source=Path(source);root=Path(output);root.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();source_hash=sha(source);model=SimplifiedGenesis.load(source)
    if model.config['source_mode']!='genesis':raise ValueError('A real inherited Genesis source is required')
    original_config=deepcopy(model.config)
    if condensation:model=model.with_condensation()
    config_hash=hashlib.sha256(encoded(model.config)).hexdigest()
    first=observe(model,0.);last_before=None;history=[first];phase_rows=[];count=0
    years=float(model.config['max_step_years'] if step_years is None else step_years)
    if not 0<years<=model.config['max_step_years']:raise ValueError('Step exceeds inherited bounds')
    deadline=start+budget_seconds
    report=dict(source_file=str(source),source_sha256=source_hash,config_sha256=config_hash,
        budget_seconds=budget_seconds,step_years=years,condensation_opt_in=condensation,
        only_authorized_config_change=all(model.config[k]==v for k,v in original_config.items() if k!='water_eos'),
        step_policy='stored max_step_years; existing phase/contact/corrector/energy guards; stop on refusal, no threshold edits or blind retries',
        temporal_error_certified=False,canonical_advanced=False,fixed_reference_thermal_graph=True,
        status='RUNNING',handoff_performed=False,initial=first)
    checkpoint(model,root)
    stream=(root/'history.jsonl').open('x');stream.write(json.dumps(first)+'\n');stream.flush()
    try:
        with time_limit(max(.1,deadline-time.monotonic()-4)):
            # Leave ~12 s for a final exact replay/checkpoint. No artificial model horizon.
            while time.monotonic()<deadline-16:
                before=deepcopy(model.state);step_start=time.monotonic()
                try:
                    model.advance(years)
                    audit=independent_phase_audit(model,before)
                except BaseException:
                    model.state=before
                    raise
                last_before=before;count+=1;phase_rows.append(audit)
                row=observe(model,time.monotonic()-start);row['step_seconds']=time.monotonic()-step_start
                row['independent_phase']=audit;history.append(row)
                stream.write(json.dumps(row)+'\n');stream.flush()
                if count%100==0:
                    checkpoint(model,root)
                    print(json.dumps({'progress':row}),flush=True)
                if len(model.state['omega'])>len(before['omega']):
                    report['status']='FIRST_SELECTOR_SPLIT';report['eligibility']=eligibility(model)
                    checkpoint(model,root)
                    if report['eligibility']['eligible']:
                        mature=ThermalGeologicalContinuation.from_source(model);mature.save(root/'mature-import')
                        dt=years/1e6 # inherited short interval, no jump to synthetic 10 Myr steps
                        mature.step(dt);mature.step(dt);mature.save(root/'mature-restart')
                        resumed=ThermalGeologicalContinuation.load(root/'mature-restart')
                        mature.step(dt);resumed.step(dt)
                        a=mature.save(root/'mature-direct');b=resumed.save(root/'mature-resumed')
                        exact_saved_state(a,b)
                        report.update(handoff_performed=True,mature_steps=3,mature_dt_myr=dt,mature_restart_exact=True,
                                      mature_rows=mature.record['steps'])
                    break
            else:report['status']='COMPUTE_BUDGET_REACHED_BEFORE_SELECTOR_EVENT'
            if last_before is not None and time.monotonic()<deadline-7:
                replay=SimplifiedGenesis(model.config,last_before);replay.advance(years)
                report['last_real_step_restart_exact']=encoded(replay.state)==encoded(model.state)
                if not report['last_real_step_restart_exact']:raise ValueError('Last real step restart mismatch')
    except TimeoutError as e:
        report.update(status='COMPUTE_BUDGET_REACHED',stop_reason=str(e))
    except Exception as e:
        report.update(status='STOPPED_AT_UNSUPPORTED_EVENT',stop_reason=type(e).__name__+': '+str(e),
                      attempted_step_years=years,attempted_endpoint_myr=model.state['time_myr']+years/1e6)
    finally:
        stream.close();checkpoint(model,root)
    report.update(completed_real_steps=count,final=observe(model,time.monotonic()-start),
        model_years_advanced=(model.state['time_myr']-first['time_myr'])*1e6,
        source_unchanged=sha(source)==source_hash,parameters_unchanged=hashlib.sha256(encoded(model.config)).hexdigest()==config_hash,
        elapsed_seconds=time.monotonic()-start,final_checkpoint_sha256=sha(root/'checkpoint.json'),
        phase_audit_all_passed=len(phase_rows)==count,
        max_whole_energy_residual_over_allowance=max((abs(x['whole_energy_residual_j'])/x['whole_energy_allowance_j'] for x in history[1:]),default=None),
        peak_damage_max=max(x['damage_max'] for x in history),
        peak_above_threshold_cells=max(x['above_threshold_cells'] for x in history),
        minimum_solid_temperature_k=min(x['solid_min_k'] for x in history),
        artifact_bytes=sum(p.stat().st_size for p in root.rglob('*') if p.is_file()))
    if not report['source_unchanged'] or not report['parameters_unchanged']:raise AssertionError('Source/parameter mutation')
    (root/'report.json').write_bytes(encoded(report));print(json.dumps(report,indent=2),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    p.add_argument('--budget-seconds',type=float,default=280.)
    p.add_argument('--condensation',action='store_true');p.add_argument('--step-years',type=float);args=p.parse_args()
    run(args.source,args.output,args.budget_seconds,condensation=args.condensation,step_years=args.step_years)
