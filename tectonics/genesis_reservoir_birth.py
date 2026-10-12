"""Source-bound thermal→birth proposal. Never advances mechanical memory."""
from dataclasses import asdict
from types import SimpleNamespace
import hashlib,json,math
import numpy as np
from .genesis import GenesisParameters
from .genesis_shell import ShellParameters,lid_geometry
from .genesis_reservoir_thermal import validate,temperature
from .genesis_bulk_birth import (bulk_birth_from_dict,bulk_birth_to_dict,advance_bulk_birth,validate_bulk_birth_continuation)
from .genesis_cohort_mechanics import cohort_temperature


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def phase_profile(state,history,shell,thermal):
    validate(state);cells={c.identity:c for c in state.cells};mass=np.asarray(history.layer_mass_kg)
    if not np.allclose(mass,mass[:,:1],rtol=2e-13,atol=0):raise ValueError('Inherited depth/front mapping requires equal-mass layers per column')
    expected={'deep_mantle','surface_water'}|{f'column:{i}:{j}' for i,row in enumerate(mass) for j in range(len(row))}
    if set(cells)!=expected:raise ValueError('Thermal ownership changed')
    for i,row in enumerate(mass):
        for j,m in enumerate(row):
            if cells[f'column:{i}:{j}'].mass_kg!=m:raise ValueError('Material layer mass changed')
    T=np.array([[temperature(cells[f'column:{i}:{j}'],thermal) for j in range(len(row))] for i,row in enumerate(mass)])
    ts=temperature(cells['surface_water'],thermal);tm=temperature(cells['deep_mantle'],thermal)
    front=lid_geometry(T,ts,tm,shell,thermal)[0]/shell.column_depth_km
    return front,T,ts,tm


def material_proposal(history,after_front,after_time,mesh,radius):
    """Existing append/retirement law; retirement is diagnostic, not BE admission."""
    candidate=advance_bulk_birth(history,before_front_fraction=history.front_fraction,after_front_fraction=after_front,
      before_time_myr=history.time_myr,after_time_myr=after_time,mesh=mesh,radius_km=radius,solidus_k=history.solidus_k)
    validate_bulk_birth_continuation(candidate,history)
    totals=[math.fsum(row) for row in history.layer_mass_kg];old={c.id:c for c in history.cohorts}
    births=[];retirements=[]
    for c in candidate.cohorts:
        if c.id not in old:births.append(dict(id=c.id,face=c.face_index,layer=c.layer_index,q_lo=c.q_lo,q_hi=c.live_q_hi,mass_kg=totals[c.face_index]*(c.live_q_hi-c.q_lo),birth_temperature_k=c.birth_temperature_k))
        elif c.live_q_hi<old[c.id].live_q_hi:retirements.append(dict(id=c.id,face=c.face_index,layer=c.layer_index,q_lo=c.live_q_hi,q_hi=old[c.id].live_q_hi,mass_kg=totals[c.face_index]*(old[c.id].live_q_hi-c.live_q_hi)))
    expected=math.fsum(t*(b-a) for t,a,b in zip(totals,history.front_fraction,candidate.front_fraction))
    actual=math.fsum(x['mass_kg'] for x in births)-math.fsum(x['mass_kg'] for x in retirements)
    if not math.isclose(actual,expected,rel_tol=2e-12,abs_tol=1e-9):raise ValueError('Front mass partition')
    return candidate,births,retirements


def prepare(before,after,legacy_raw):
    sha=hashlib.sha256(legacy_raw).hexdigest();root=json.loads(legacy_raw);r=root['runtime_state'];params=root['source_packet']['source']['parameters']
    for state in (before,after):
        if json.loads(state.origin_json)['legacy_raw_sha256']!=sha or json.loads(state.parameters_json)!=params['thermal']:raise ValueError('Thermal source binding')
    if before.origin_json!=after.origin_json or before.reference_total_j!=after.reference_total_j:raise ValueError('Thermal migration/reference changed across interval')
    h=bulk_birth_from_dict(r['history']);shell=ShellParameters(**params['shell']);thermal=GenesisParameters(**params['thermal'])
    if before.time_myr!=h.time_myr or after.time_myr<=before.time_myr:raise ValueError('Shared clock')
    f0,T0,ts0,tm0=phase_profile(before,h,shell,thermal);f1,T1,ts1,tm1=phase_profile(after,h,shell,thermal)
    mismatch=float(np.max(np.abs(f0-np.asarray(h.front_fraction))))
    if mismatch>5e-12:raise ValueError('Migration changed inherited solidus front')
    # Keep exact accepted fractions; only the recomputed endpoint is appended.
    geometry=root['source_packet']['current_geometry']
    mesh=SimpleNamespace(vertices=np.array(geometry['vertices_unit_xyz']),faces=np.array(geometry['faces']))
    candidate,births,retirements=material_proposal(h,f1.tolist(),after.time_myr,mesh,geometry['radius_km'])
    cells={c.identity:c for c in after.cells}
    for row in births+retirements:
        cell=cells[f"column:{row['face']}:{row['layer']}"]
        row['column_enthalpy_subset_j']=row['mass_kg']*cell.energy_j/cell.mass_kg
        row['enthalpy_semantics']='diagnostic subset of existing finite-volume cell; never added as heat'
    old_memory=r['mechanics'];old_ids={c.id for c in h.cohorts}
    live=[c for c in candidate.cohorts if c.live_q_hi>c.q_lo]
    before_t=cohort_temperature(candidate,T0,ts0,tm0);after_t=cohort_temperature(candidate,T1,ts1,tm1)
    payload=dict(format='genesis-reservoir-birth-proposal-0.1',status='BLOCKED_PHASE_MAPPING' if retirements else 'READY_FOR_LOADING_PREPARATION',
      legacy_raw_sha256=sha,before_thermal_state_sha256=digest(asdict(before)),after_thermal_state_sha256=digest(asdict(after)),
      before_time_myr=before.time_myr,after_time_myr=after.time_myr,parent_mechanical_time_myr=h.time_myr,
      parent_memory_sha256=digest(old_memory),parent_memory_preserved=True,
      old_cohort_formed_records_preserved=True,newborn_requests=births,proposed_retirements=retirements,
      birth_mass_kg=math.fsum(x['mass_kg'] for x in births),retirement_mass_kg=math.fsum(x['mass_kg'] for x in retirements),
      inherited_front_reconstruction_max_error=mismatch,front_reconstruction_tolerance=5e-12,
      proposed_history=bulk_birth_to_dict(candidate),history_admitted_to_mechanics=False,
      mechanical_loading_request=None if retirements else dict(ids=[c.id for c in live],before_temperature_k=before_t.tolist(),after_temperature_k=after_t.tolist(),birth_temperature_k=[c.birth_temperature_k for c in live],old_memory_source='unchanged parent_memory_sha256',newborn_memory='existing stress-free-at-solidus policy; to be evaluated by prepare_volume_loading'),
      diagnostic_retained_temperatures=dict(ids=[c.id for c in live],before_temperature_k=before_t.tolist(),after_temperature_k=after_t.tolist()),
      physical_melt_export_mass_kg=None,retirement_semantics='legacy connected fully-solid envelope, not demonstrated liquid mass',
      remelt_elastic_outflow_j=None,remelt_kinetic_outflow_j=None,latent_heat_source_added=False,new_thermal_steps=0,
      blocker='Conservative phase/enthalpy mapping required before interpreting envelope recession as physical remelt; BE no_remelt remains closed' if retirements else None)
    return {**payload,'sha256':digest(payload)}


def require_loading_request(packet):
    body={k:v for k,v in packet.items() if k!='sha256'}
    if digest(body)!=packet.get('sha256'):raise ValueError('Birth proposal seal')
    if packet['status']!='READY_FOR_LOADING_PREPARATION' or packet['proposed_retirements'] or packet['mechanical_loading_request'] is None:
        raise ValueError('STOP: remelt proposal cannot enter current BE loading')
    return packet['mechanical_loading_request']
