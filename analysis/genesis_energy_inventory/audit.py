"""Read-only ledger arithmetic; no thermal/mechanical solve or evolution."""
import argparse,hashlib,json,math,resource,signal,time
from pathlib import Path
import numpy as np
from tectonics.genesis import GenesisParameters,ENERGY_SCALE,SECONDS_PER_MYR

ROOT=Path('analysis/genesis_volume_creation_validation/evidence/actual_continuation/direct_after_second.json')
ROOT_SHA='62579a0bad128830a66e2f44579cf2b1f5d3835b691225b8d431102f3e4486d7'
STEPS=[('step_1.json','044c7780b4f612b8445bd2198bd068c45feef80bd7a2b8225d1a5cb9ed697ea3'),
       ('step_2.json','0536be1e28221249c4ea92603b6c19a3a1f16c8c54fbfa241273332f525ee97d')]
LD=np.longdouble


def read_pinned(path,expected):
    raw=path.read_bytes();sha=hashlib.sha256(raw).hexdigest()
    if sha!=expected:raise ValueError('Raw input hash mismatch: '+str(path))
    return json.loads(raw),dict(path=str(path),bytes=len(raw),sha256=sha)


def residual_record(residual,throughput,tolerance=1e-9):
    relative=float(abs(residual)/max(abs(throughput),1))
    return dict(residual_j=float(residual),relative_to_throughput=relative,tolerance=tolerance,
                status='PASS' if math.isfinite(relative) and relative<=tolerance else 'FAIL')


def thermal_budget(before,after,parameters):
    p=GenesisParameters(**parameters);area=LD(p.area_m2)
    y0=np.asarray(before['global_state']['energy'],dtype=LD);y1=np.asarray(after['global_state']['energy'],dtype=LD)
    delta=(y1-y0)*LD(ENERGY_SCALE)*area
    dt=LD(after['time_myr'])-LD(before['time_myr']);seconds=dt*LD(SECONDS_PER_MYR)
    tide=LD(after['orbit']['dissipated_energy_j'])-LD(before['orbit']['dissipated_energy_j'])
    received=LD(after['tidal_heat_received_j'])-LD(before['tidal_heat_received_j'])
    decay=LD(math.log(2))/LD(p.radiogenic_half_life_myr)
    rad=(LD(p.silicate_column_kg_m2)*area*LD(p.radiogenic_specific_power_w_kg)*
         np.exp(-decay*(LD(before['time_myr'])+LD(p.system_age_at_start_myr)))*(-np.expm1(-decay*dt))/decay*LD(SECONDS_PER_MYR))
    stellar=area*LD(p.absorbed_stellar_flux_w_m2)*seconds
    giant=area*LD(p.giant_absorbed_flux_w_m2)*seconds
    mass0=np.asarray(before['layer_mass_kg'],dtype=LD);mass1=np.asarray(after['layer_mass_kg'],dtype=LD)
    if not np.array_equal(mass0,mass1):raise ValueError('Column mass changed; moving-mass audit required')
    dh=np.asarray(after['column_enthalpy'],dtype=LD)-np.asarray(before['column_enthalpy'],dtype=LD)
    column_delta=np.sum(mass0*dh,dtype=LD)
    boundary=LD(after['boundary_energy_j'])-LD(before['boundary_energy_j'])
    checks=dict(global_thermal=residual_record(delta[0]+delta[1]-delta[2]+delta[3],abs(delta[2])+abs(delta[3])),
      source_decomposition=residual_record(delta[2]-rad-tide-stellar-giant,delta[2]),
      orbit_heat_transfer=residual_record(received-tide,max(abs(tide),abs(received))),
      columns=residual_record(column_delta-boundary,abs(boundary)))
    return dict(dt_myr=float(dt),global_units='J (saved energy * 1e12 J/m2 * spherical reference area)',
      global_delta_mantle_j=float(delta[0]),global_delta_surface_j=float(delta[1]),
      global_input_j=float(delta[2]),global_OLR_out_j=float(delta[3]),
      sources_j=dict(radiogenic_analytic=float(rad),orbital_dissipation=float(tide),absorbed_stellar=float(stellar),absorbed_giant=float(giant)),
      mantle_to_surface_inferred_j=float(rad+tide-delta[0]),
      mantle_to_surface_independently_recorded_j=None,
      column_delta_j=float(column_delta),column_net_boundary_in_j=float(boundary),
      column_top_boundary_in_j=None,column_bottom_boundary_in_j=None,checks=checks)


def mechanical_budget(L):
    # Required fields: absence raises, never substitutes zero.
    delta=LD(L['kinetic_after_j'])+LD(L['elastic_after_j'])-LD(L['kinetic_before_j'])-LD(L['elastic_before_j'])
    incoming=sum((LD(L[k]) for k in ['external_work_j','thermal_work_j','birth_kinetic_inflow_j','damage_energy_change_j']),LD(0))
    outgoing=sum((LD(L[k]) for k in ['physical_drag_j','physical_maxwell_j','numerical_kinetic_j','numerical_memory_j','remelt_kinetic_outflow_j','remelt_elastic_outflow_j']),LD(0))
    return dict(delta_K_plus_U_j=float(delta),signed_inputs_j=float(incoming),sinks_j=float(outgoing),
      ledger_j=L,check=residual_record(delta-incoming+outgoing,abs(delta)+abs(incoming)+outgoing,1e-11),
      thermal_counterentry_j=None,mantle_traction_reservoir_debit_j=None,birth_donor_kinetic_debit_j=None,
      damage_energy_destination_j=None,numerical_heat_conversion='forbidden')


def audit_pair(before,after,parameters):
    thermal=thermal_budget(before['thermal'],after['thermal'],parameters)
    mechanics=mechanical_budget(after['interval_ledger'])
    old_ids={c['id'] for c in before['geometry']['cohorts']}
    newborn=[c for c in after['geometry']['cohorts'] if c['id'] not in old_ids]
    return dict(thermal=thermal,mechanics=mechanics,birth=dict(cohorts=len(newborn),mass_kg=math.fsum(c['mass_kg'] for c in newborn),enthalpy_transfer_j=None,latent_heat_already_in_column_enthalpy=True))


def run(directory):
    root,pin=read_pinned(ROOT,ROOT_SHA);before=root['runtime_state'];params=root['source_packet']['source']['parameters']['thermal']
    records=[];pins=[pin];predecessor=ROOT_SHA;first=before
    for name,sha in STEPS:
        packet,pin=read_pinned(directory/name,sha);pins.append(pin)
        if packet['predecessor_sha256']!=predecessor:raise ValueError('Chain predecessor')
        after=packet['state'];records.append(audit_pair(before,after,params))
        before=after;predecessor=packet['sha256']
    window_thermal=thermal_budget(first['thermal'],before['thermal'],params)
    window_mechanics=mechanical_budget(before['be_accumulated_ledger'])
    checks=[v for row in records for v in row['thermal']['checks'].values()]+[r['mechanics']['check'] for r in records]
    checks+=list(window_thermal['checks'].values())+[window_mechanics['check']]
    for key in ['external_work_j','thermal_work_j','birth_kinetic_inflow_j','damage_energy_change_j','physical_drag_j','physical_maxwell_j','numerical_kinetic_j','numerical_memory_j']:
        summed=math.fsum(r['mechanics']['ledger_j'][key] for r in records)
        if not math.isclose(summed,before['be_accumulated_ledger'][key],rel_tol=1e-12,abs_tol=1e-9):raise ValueError('Cumulative ledger mismatch: '+key)
    return dict(status='PASS' if all(c['status']=='PASS' for c in checks) else 'FAIL',
      meaning='Separate saved-ledger consistency only, not global coupled closure',new_Genesis_steps=0,
      units='J unless explicitly stated',intervals=records,
      full_window_thermal=window_thermal,full_window_mechanics=window_mechanics,
      combined_global_residual_j=None,combined_global_status='UNDEFINED: overlapping reservoirs and missing counterentries',
      input_pins=pins,canonical_unchanged=hashlib.sha256(ROOT.read_bytes()).hexdigest()==ROOT_SHA)


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=Path('/workspace/genesis-be-time-control-01/50'));p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Output exists')
    resource.setrlimit(resource.RLIMIT_AS,(1024**3,1024**3))
    def stop(*args):raise TimeoutError('60-second audit budget')
    signal.signal(signal.SIGALRM,stop);signal.alarm(60);start=time.monotonic();report=run(a.input)
    report.update(seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
      audit_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with a.output.open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:report[k] for k in ['status','seconds','peak_rss_kib','full_window_thermal','full_window_mechanics']},indent=2));signal.alarm(0)
    return 0 if report['status']=='PASS' else 1

if __name__=='__main__':raise SystemExit(main())
