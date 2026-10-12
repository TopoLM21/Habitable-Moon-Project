"""One real-source 25-year coupled step; total 600 seconds, no retry/sweep."""
import argparse,json,hashlib,resource,time,traceback
from dataclasses import asdict
from pathlib import Path
import numpy as np
from tectonics.genesis import GenesisParameters
from tectonics.genesis_tides import TidalParameters,TidalOrbitState,validate_tidal_orbit
from tectonics.genesis_thermal_ports import apply_ports
from tectonics.genesis_energy_ports import boundary_budget
from tectonics.genesis_cutcell_loading import prepare_saved_loading
from tectonics.genesis_volume_loading import VolumeCohortMemory
from tectonics.genesis_volume_fault_evolution import volume_fault_history_from_dict,volume_fault_history_to_dict
from tectonics.genesis_volume_slip_checkpoint import load_volume_slip_checkpoint
from analysis.genesis_connected_step_validation.run import PARENT,PARENT_SHA,FORMATION_SHA,deadline
from analysis.genesis_coupled_split_control.run import pinned,audit_step
from analysis.genesis_coupled_split_control.thermal import conduct,phase_guard
from analysis.genesis_coupled_split_control.mechanical import advance
from analysis.genesis_be_candidate.adapter import dumps
from analysis.genesis_be_candidate.checkpoint import seal
from .audit import audit_ports
from .preflight import inspect


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True,type=Path);out=parser.parse_args().output;out.mkdir(exist_ok=False)
    start=time.monotonic();report=dict(status='RUNNING',new_coupled_steps_started=0,new_coupled_steps_passed=0,canonical_advanced=False,compute_limit_seconds=600)
    def write(name,x):
        with (out/name).open('x') as f:f.write(dumps(x))
    raw=PARENT.read_bytes()
    try:
      with deadline(600):
        resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
        root=pinned(PARENT,PARENT_SHA);source=root['source_packet'];params=source['source']['parameters'];p=GenesisParameters(**params['thermal']);tides=TidalParameters(**params['tides'])
        write('source_preflight.json',inspect())
        cut=pinned('/workspace/genesis-cutcell-feasibility-02/candidate.json','6006521e7b3e22ce23894e63248d0145ee15d3065333c949701c730c6cab47ad')
        oldthermal=pinned('/workspace/genesis-implicit-cutcell-01/after.json','632354594ccda5aaffeb8933d9ed95be96cb53037a2ce67d18cf407562d6f2e8')
        proj=pinned('/workspace/genesis-coupled-thermal-projection-03/projected.json','d25fe1fd4e8b56b18ab8ec697b197c326d723e9421148e5d3726fb7affdd45a4');cells={c['identity']:c for c in proj['state']['cells']}
        before=dict(time_myr=root['runtime_state']['thermal']['time_myr'],mass_kg=oldthermal['mass_kg'],energy_j=[c['mass_kg']*c['specific_h'] for c in cut['cells']]+[(cells[k]['energy_j']/cells[k]['mass_kg'])*cells[k]['mass_kg'] for k in ('deep_mantle','surface_water')])
        template=prepare_saved_loading(raw,oldthermal);cuts=oldthermal['fixed_cut_geometry'];assert len(cuts)==3360
        print('load authenticated mechanical parent',flush=True);model,parent=load_volume_slip_checkpoint(PARENT,expected_source_checkpoint_sha256=FORMATION_SHA)
        orbit=TidalOrbitState(**root['runtime_state']['thermal']['orbit']);target=before['time_myr']+2.5e-5
        report['new_coupled_steps_started']=1
        print('real source ports and orbital debit',flush=True);sourced,end_orbit,ports=apply_ports(before,p,tides,orbit,target)
        write('initial_thermal.json',before);write('source_endpoint.json',sourced);write('ports.json',ports)
        source_audit=audit_ports(before,sourced,ports,p,tides);write('source_audit.json',source_audit)
        print('implicit conduction and phase guards',flush=True);thermal,ta=conduct(sourced,cuts,root,p,2.5e-5);write('thermal.json',thermal);write('thermal_audit.json',ta)
        print('mechanical feedback with actual endpoint orbit',flush=True);result=advance(model,parent,root,thermal,template.radial_moments,orbit_after=end_orbit)
        phase_guard(cuts,np.array(result['thermal']['temperature_k']),p,np.array(thermal['temperature_k']))
        result.pop('sha256');result.update(format='genesis-forced-cutcell-coupled-0.1',source_ports=ports,source_endpoint=sourced,initial_thermal=before,thermal_packet=thermal,source_parameter_sha256=hashlib.sha256(json.dumps(params,sort_keys=True,separators=(',',':')).encode()).hexdigest())
        result=seal(result);write('candidate.json',result)
        print('independent coupled audit',flush=True);aud=audit_step(root,thermal,result,source);write('coupled_audit.json',aud)
        delta=float(np.sum(np.array(result['thermal']['energy_j'],dtype=np.longdouble)-np.array(before['energy_j'],dtype=np.longdouble)))
        open_mech=boundary_budget(result['state']['interval_ledger'],delta)
        residual=open_mech['balance_residual_j']-source_audit['net_source_j']
        allowance=source_audit['source_allowance_j']+ta['global_allowance_j']+aud['feedback']['combined_allowance_j']
        if abs(residual)>allowance:raise ValueError('Full forced energy budget')
        # Including orbit cancels tidal heat/debit; other external ports remain.
        external_without_tide=source_audit['net_source_j']-ports['orbital_debit_j']
        orbital_included_residual=open_mech['balance_residual_j']-ports['orbital_debit_j']-external_without_tide
        assert abs(orbital_included_residual)<=allowance
        restored=json.loads((out/'candidate.json').read_bytes());assert seal({k:v for k,v in restored.items() if k!='sha256'})==restored
        restored['state']['mechanics']['cohorts']=[asdict(VolumeCohortMemory(**c)) for c in restored['state']['mechanics']['cohorts']]
        decoded_orbit=TidalOrbitState(**restored['orbit_after']);validate_tidal_orbit(decoded_orbit,tides);assert asdict(decoded_orbit)==restored['source_ports']['orbit_after']
        faults=volume_fault_history_from_dict(restored['state']['fault_history'],source);assert volume_fault_history_to_dict(faults)==restored['state']['fault_history']
        assert dumps(restored).encode()==(out/'candidate.json').read_bytes()
        audit_ports(restored['initial_thermal'],restored['source_endpoint'],restored['source_ports'],p,tides)
        write('restart.json',restored);assert (out/'candidate.json').read_bytes()==(out/'restart.json').read_bytes()
        old= np.array(oldthermal['temperature_k']);T=np.array(result['thermal']['temperature_k'])
        report.update(status='PASS',new_coupled_steps_passed=1,correctors=len(result['correctors']),old_ids=800,births=0,retirements=0,restart_byte_identical=True,source_audit=source_audit,thermal_audit=ta,mechanical_audit=aud['mechanical'],feedback_audit=aud['feedback'],net_external_thermal_input_j=source_audit['net_source_j'],full_budget_residual_j=residual,full_budget_allowance_j=allowance,orbit_included_residual_j=orbital_included_residual,mechanical_boundary_budget=open_mech,max_damage=float(max(faults.damage)),minimum_temperature_k=float(T.min()),maximum_temperature_k=float(T.max()),surface_temperature_k=float(T[-1]),source_surface_temperature_k=ports['source_surface_endpoint_k'],ledger=result['state']['interval_ledger'],candidate_sha256=hashlib.sha256((out/'candidate.json').read_bytes()).hexdigest())
    except Exception as exc:
        report.update(status='STOPPED',error_type=type(exc).__name__,error=str(exc));(out/'failure.txt').write_text(traceback.format_exc())
    finally:
        report.update(seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,parent_unchanged=PARENT.read_bytes()==raw);write('report.json',report)
    print(dumps({k:report[k] for k in ('status','new_coupled_steps_started','new_coupled_steps_passed','seconds')}|({'error':report['error']} if 'error' in report else {})))


if __name__=='__main__':main()
