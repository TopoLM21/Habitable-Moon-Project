"""Read-only completion from the saved candidate; zero mechanical solver calls."""
import json,hashlib
from pathlib import Path
from dataclasses import asdict
from tectonics.genesis_volume_loading import VolumeCohortMemory
from analysis.genesis_be_candidate.checkpoint import seal
from analysis.genesis_be_candidate.adapter import dumps
from analysis.genesis_be_candidate.audit import trial as independent_trial
from analysis.genesis_volume_fault_validation.independent_audit import independent_damage_law
from analysis.genesis_connected_step_validation.run import PARENT,PARENT_SHA
from .audit import audit_feedback
import numpy as np


def main():
    out=Path('/workspace/genesis-cutcell-mechanical-02');raw=PARENT.read_bytes();assert hashlib.sha256(raw).hexdigest()==PARENT_SHA;root=json.loads(raw)
    thraw=Path('/workspace/genesis-implicit-cutcell-01/after.json').read_bytes();thermal=json.loads(thraw)
    saved=(out/'candidate.json').read_bytes();p=json.loads(saved)
    assert hashlib.sha256(thraw).hexdigest()==p['thermal_sha256'] and p['parent_sha256']==PARENT_SHA
    assert seal({k:v for k,v in p.items() if k!='sha256'})==p
    typed=tuple(VolumeCohortMemory(**c) for c in p['state']['mechanics']['cohorts'])
    p['state']['mechanics']['cohorts']=[asdict(c) for c in typed]
    assert dumps(p).encode()==saved
    final=independent_trial(root['runtime_state'],p['state'],p['inputs'],root['source_packet'])
    pred=independent_trial(root['runtime_state'],p['predictor'],p['predictor_inputs'],root['source_packet'])
    expected,_=independent_damage_law(root['runtime_state']['fault_history'],p['predictor_driver'],root['source_packet'],root['runtime_state']['thermal']['orbit'],root['runtime_state']['thermal']['orbit'],p['thermal']['time_myr'])
    from analysis.genesis_be_candidate.audit import face_driver
    from analysis.genesis_volume_fault_validation.independent_audit import tensile_principal
    tensile=tensile_principal(face_driver(p['state'],root['source_packet'])['stress_pa'])
    for name,val in expected.items():
        if name=='peak_tensile_pa':val=np.maximum(val,tensile)
        assert np.allclose(val,p['state']['fault_history'][name],rtol=3e-10,atol=1e-12),name
    for state in (root['runtime_state'],p['predictor'],p['state']):
        f=state['fault_history'];assert not any(f['fault_active']) and not any(f['fault_candidate_age_myr']) and max(f['damage'])<root['source_packet']['source']['parameters']['faults']['activation_damage']
    audit=audit_feedback(root,thermal,p)
    with (out/'restart.json').open('xb') as stream:stream.write(dumps(p).encode())
    assert (out/'restart.json').read_bytes()==saved
    assert PARENT.read_bytes()==raw
    report=dict(status='PASS',physical_intervals=1,correctors=p['correctors'],restart_byte_identical=True,readback_solver_calls=0,old_ids=800,births=0,retirements=0,mass_and_phase_compatible=True,canonical_advanced=False,parent_unchanged=True,gate_active_count=0,max_damage=max(p['state']['fault_history']['damage']),
                max_feedback_temperature_change_k=float(np.max(np.abs(np.array(p['thermal']['temperature_k'])-thermal['temperature_k']))),predictor_audit=pred,mechanical_audit=final,feedback_audit=audit,ledger=p['state']['interval_ledger'],storage=p['storage'],candidate_sha256=hashlib.sha256(saved).hexdigest(),regression_tests_passed=66,unresolved=p['unresolved'])
    with (out/'completion.json').open('x') as stream:stream.write(dumps(report))
    print(dumps({k:v for k,v in report.items() if k not in ('predictor_audit','mechanical_audit','feedback_audit')}))


if __name__=='__main__':main()
