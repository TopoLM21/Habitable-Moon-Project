"""Opt-in adapter for the existing forced BE pipeline; no solver changes."""
from .controller import time_limit
from dataclasses import replace,asdict
import json
import math
import numpy as np
from tectonics.genesis import GenesisParameters
from tectonics.genesis_tides import TidalParameters,TidalOrbitState,validate_tidal_orbit
from tectonics.genesis_thermal_ports import apply_ports
from tectonics.genesis_energy_ports import boundary_budget
from tectonics.genesis_cutcell_loading import prepare_saved_loading
from tectonics.genesis_volume_slip_checkpoint import load_volume_slip_checkpoint
from analysis.genesis_connected_step_validation.run import PARENT,PARENT_SHA,FORMATION_SHA
from analysis.genesis_coupled_split_control.run import pinned,audit_step,decode,root_from_previous
from analysis.genesis_coupled_split_control.thermal import conduct,phase_guard
from analysis.genesis_coupled_split_control.mechanical import advance as mechanical
from analysis.genesis_forced_coupled_validation.audit import audit_ports
from analysis.genesis_be_candidate.adapter import dumps
from analysis.genesis_be_candidate.checkpoint import seal
from .metrics import observe_packet

SEED='/workspace/genesis-forced-coupled-25-01/candidate.json'
SEED_SHA='e6ad0ed1372b18a73c3852f639edafd959776e6cb0d3a747aa75bcffe8c365a5'


def continued_root(root,p):
    # The historical unforced helper advances only the orbit clock. Real forcing
    # must carry the actual eccentricity, semimajor axis AND cumulative debit.
    r=root_from_previous(root,p)
    r['runtime_state']['thermal']={**r['runtime_state']['thermal'],'orbit':p['orbit_after']}
    return r


def audit_forced(root,p):
    source=root['source_packet'];params=source['source']['parameters']
    gp=GenesisParameters(**params['thermal']);tides=TidalParameters(**params['tides'])
    sa=audit_ports(p['initial_thermal'],p['source_endpoint'],p['source_ports'],gp,tides)
    a=audit_step(root,p['thermal_packet'],p,source)
    before=np.array(p['initial_thermal']['energy_j'],dtype=np.longdouble);after=np.array(p['thermal']['energy_j'],dtype=np.longdouble)
    b=boundary_budget(p['state']['interval_ledger'],float(np.sum(after-before)))
    ta=p['adaptive_thermal_audit']
    if ta['status']!='PASS' or not 0<=ta['node_balance_over_allowance']<=1:raise ValueError('THERMAL_AUDIT')
    residual=b['balance_residual_j']-sa['net_source_j']
    allowance=sa['source_allowance_j']+ta['global_allowance_j']+a['feedback']['combined_allowance_j']
    if abs(residual)>allowance:raise ValueError('FORCED_WHOLE_BUDGET')
    orbit=TidalOrbitState(**p['orbit_after']);validate_tidal_orbit(orbit,tides)
    if p['orbit_after']!=p['source_ports']['orbit_after']:raise ValueError('ORBIT_RESTART_MISMATCH')
    return dict(status='PASS',sources=sa,mechanical=a['mechanical'],feedback=a['feedback'],
                thermal=ta,whole_residual_j=residual,whole_allowance_j=allowance)


class ForcedBackend:
    def __init__(self,write):
        self.write=write;self.root=pinned(PARENT,PARENT_SHA);self.source=self.root['source_packet']
        self.seed=pinned(SEED,SEED_SHA);self.initial=dict(elapsed_years=0.,packet=self.seed)
        params=self.source['source']['parameters'];self.gp=GenesisParameters(**params['thermal']);self.tides=TidalParameters(**params['tides'])
        self.cuts=self.seed['thermal_packet']['fixed_cut_geometry']
        th=pinned('/workspace/genesis-implicit-cutcell-01/after.json','632354594ccda5aaffeb8933d9ed95be96cb53037a2ce67d18cf407562d6f2e8')
        self.moments=prepare_saved_loading(PARENT.read_bytes(),th).radial_moments
        self.model,self.parent=load_volume_slip_checkpoint(PARENT,expected_source_checkpoint_sha256=FORMATION_SHA)
        # Authenticate the starting private state, including its original audits.
        audit_step(self.root,self.seed['thermal_packet'],self.seed,self.source)
        audit_ports(self.seed['initial_thermal'],self.seed['source_endpoint'],self.seed['source_ports'],self.gp,self.tides)
        self.origin=self.seed['thermal']['time_myr']

    def advance(self,before,h,label,cap):
        with time_limit(cap):
            p0=before['packet'];root=continued_root(self.root,p0);parent=decode(self.parent,p0,self.source)
            parent=replace(parent,thermal=replace(parent.thermal,orbit=TidalOrbitState(**p0['orbit_after'])))
            thermal0=p0['thermal'];target=thermal0['time_myr']+h/1e6
            # Carry only authoritative source-stage energy fields; do not attach
            # stale temperature/phase metadata to the newly dosed energies.
            source_input={k:thermal0[k] for k in ('time_myr','mass_kg','energy_j')}
            sourced,orbit,ports=apply_ports(source_input,self.gp,self.tides,parent.thermal.orbit,target)
            audit_ports(thermal0,sourced,ports,self.gp,self.tides)
            thermal,ta=conduct(sourced,self.cuts,root,self.gp,h/1e6)
            p=mechanical(self.model,parent,root,thermal,self.moments,orbit_after=orbit)
            phase_guard(self.cuts,np.array(p['thermal']['temperature_k']),self.gp,np.array(thermal['temperature_k']))
            p.pop('sha256');p.update(format='genesis-adaptive-forced-BE-trial-0.1',source_ports=ports,source_endpoint=sourced,
                initial_thermal=thermal0,thermal_packet=thermal,adaptive_thermal_audit=ta,
                adaptive_predecessor_sha256=p0['sha256'],parent_sha256=p0['sha256'])
            p=json.loads(dumps(seal(p)))
            # Decode checkpoint, carrying actual orbit and typed mechanical state.
            restored=decode(self.parent,p,self.source)
            restored=replace(restored,thermal=replace(restored.thermal,orbit=TidalOrbitState(**p['orbit_after'])))
            if not math.isclose(restored.time_myr,target,rel_tol=0,abs_tol=1e-14):raise ValueError('RESTART_CLOCK')
            if json.loads(dumps(asdict(restored.mechanics)))!=p['state']['mechanics']:raise ValueError('TYPED_MEMORY_RESTART')
            if asdict(restored.thermal.orbit)!=p['orbit_after']:raise ValueError('TYPED_ORBIT_RESTART')
            snapshot=dict(elapsed_years=before['elapsed_years']+h,packet=p)
            self.write(label+'.json',snapshot)
            return snapshot

    def audit(self,before,after):
        p=after['packet'];p0=before['packet'];root=continued_root(self.root,p0)
        if p['initial_thermal']!=p0['thermal']:raise ValueError('THERMAL_PREDECESSOR')
        if p['source_ports']['orbit_before']!=p0['orbit_after']:raise ValueError('ORBIT_PREDECESSOR')
        if p['adaptive_predecessor_sha256']!=p0['sha256']:raise ValueError('PREDECESSOR_MISMATCH')
        if not math.isclose(p['thermal']['time_myr'],self.origin+after['elapsed_years']/1e6,rel_tol=0,abs_tol=2e-14):raise ValueError('ABSOLUTE_CLOCK')
        return audit_forced(root,p)

    def observe(self,snapshot):
        return observe_packet(snapshot['packet'],self.source)
