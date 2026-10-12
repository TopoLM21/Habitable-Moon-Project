"""Exactly one isolated aggregate 25-year trial, bounded by 280 seconds."""
import argparse,json,time,resource,traceback
from pathlib import Path
import numpy as np
from tectonics.genesis_transfer_bridge import encoded,digest
from analysis.genesis_adaptive_control.controller import time_limit
from analysis.genesis_be_candidate.audit import face_driver
from analysis.genesis_aggregate_coupled.adapter import Adapter

ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--thermal-reduction',choices=['projected_graph','series_resistance'],required=True)
    args=p.parse_args()
    out=args.output;out.mkdir(exist_ok=False);start=time.perf_counter();report=dict(status='STOPPED',canonical_advanced=False,new_trials=0)
    def write(name,x):
        with (out/name).open('xb') as f:f.write(encoded(x))
    try:
      with time_limit(280):
        resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
        assert np.finfo(np.longdouble).nmant==63
        paths=[Path('/workspace/genesis-exploratory-100-01/continuation/initial.json'),ROOT/'analysis/genesis_volume_creation_validation/evidence/actual_continuation/direct_after_second.json',Path('/workspace/genesis-forced-coupled-25-01/candidate.json')]
        pins=['4c0427d81bcb4b2a1dd8df2021e5ae1f6fedf983a9b8b2c165d69128a716bc95','62579a0bad128830a66e2f44579cf2b1f5d3835b691225b8d431102f3e4486d7','e6ad0ed1372b18a73c3852f639edafd959776e6cb0d3a747aa75bcffe8c365a5']
        raw=[path.read_bytes() for path in paths]
        for data,pin in zip(raw,pins):assert digest(data)==pin
        initial=json.loads(raw[0])['packet'];root=json.loads(raw[1]);full=json.loads(raw[2]);source=root['source_packet']
        assert initial['thermal']==full['initial_thermal']
        assert initial['orbit_after']==full['source_ports']['orbit_before']
        assert initial['state']['mechanics']['cohorts']==root['runtime_state']['mechanics']['state']['cohorts']
        report.update(input_sha256={str(path):pin for path,pin in zip(paths,pins)},comparison='same original physical initial state/parameters and 25-year interval; reduced initial projection is the intentional change',source_parameter_sha256=full['source_parameter_sha256'])
        contact=json.loads(root['runtime_state']['contact_parameters_json'])
        a=Adapter(initial,source,full['thermal_packet']['fixed_cut_geometry'],full['inputs'],contact,thermal_reduction=args.thermal_reduction)
        report['thermal_reduction']=args.thermal_reduction
        report.update(setup_seconds=time.perf_counter()-start,active_memory_scalars_before=800*36,active_memory_scalars_after=int(a.old.size),thermal_unknowns_before=len(initial['thermal']['mass_kg']),thermal_unknowns_after=a.n,nodal_velocity_dofs=len(a.v0),unresolved_reserve_j=a.reserve,active_initial_elastic_j=float(a.active_initial),contact_parameters=contact)
        # Portable projected initial payload. Original cohorts remain only in the pinned source.
        write('initial.json',dict(time_myr=a.time,memory=a.old.tolist(),mass_kg=a.m.tolist(),energy_j=a.E0.tolist(),velocity_m_s=a.v0.tolist(),displacement_m=a.u0.tolist(),reserve_j=a.reserve,source_pins=report['input_sha256']))
        print(json.dumps(report),flush=True);report['new_trials']=1;trial_start=time.perf_counter()
        result=a.advance(25.);report['trial_seconds']=time.perf_counter()-trial_start
        write('after.json',result);restored=json.loads((out/'after.json').read_bytes());assert encoded(restored)==encoded(result)
        assert result['time_myr']==full['thermal']['time_myr']
        Tg=np.array(result['temperature_k']);Tfine=Tg[a.map];Tf=np.array(full['thermal']['temperature_k']);sig=np.array(result['stress_pa']);sf=face_driver(full['state'],source)['stress_pa']
        damage=np.array(result['fault_history']['damage']);df=np.array(full['state']['fault_history']['damage'])
        report.update(status='COMPLETED_AGGREGATE_TRIAL_NOT_CANONICAL',restart_payload_exact=True,ledger=result['ledger'],correctors=result['correctors'],damage_energy_change_j=result['damage_energy_change_j'],new_events=result['new_events'],
            comparison_metrics=dict(temperature_max_difference_k=float(np.max(np.abs(Tfine-Tf))),temperature_mass_rms_k=float(np.sqrt(np.average((Tfine-Tf)**2,weights=initial['thermal']['mass_kg']))),stress_max_difference_pa=float(np.max(np.abs(sig-sf))),stress_relative_l2=float(np.linalg.norm(sig-sf)/np.linalg.norm(sf)),damage_max_difference=float(np.max(np.abs(damage-df))),velocity_max_difference_m_s=float(np.max(np.abs(np.array(result['velocity_m_s'])-full['state']['velocity_m_s']))),elastic_active_after_j=result['ledger']['elastic_after'],full_elastic_after_j=full['state']['mechanics']['elastic_energy_j'],reserve_j=a.reserve,thermal_total_difference_j=float(np.sum(np.array(result['energy_j']),dtype=np.longdouble)-np.sum(np.array(full['thermal']['energy_j']),dtype=np.longdouble))))
        assert all(path.read_bytes()==data for path,data in zip(paths,raw))
    except Exception as exc:
        report.update(error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
    finally:
        report.update(total_seconds=time.perf_counter()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,artifact_bytes=sum(p.stat().st_size for p in out.iterdir() if p.is_file()))
        write('report.json',report);print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
