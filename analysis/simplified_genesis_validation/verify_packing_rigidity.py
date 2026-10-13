"""Bounded negative end-to-end test; does not claim general moving plates."""
import json,time
from pathlib import Path
import numpy as np
from tectonics.simplified_genesis import SimplifiedGenesis
from tectonics.simplified_genesis_contact import packing_rigidity,boundary_rows
from tectonics.genesis_transfer_bridge import encoded,digest
from .synthetic import irregular_three_domain_synthetic


def main():
    start=time.perf_counter();out=Path('/workspace/simplified-genesis-rigidity-01');out.mkdir(exist_ok=False)
    model=irregular_three_domain_synthetic();counts=[len(model.state['omega'])]
    for _ in range(2):model.advance(25.);counts.append(len(model.state['omega']))
    assert counts==[1,2,3]
    model.save(out/'after_two_splits.json');restart=SimplifiedGenesis.load(out/'after_two_splits.json')
    owner=np.array(model.state['owner']);poses=np.array(model.state['rotation']);old=np.array(model.state['omega'])
    witness=packing_rigidity(model.mesh,owner,poses);assert witness['common_rotation_only']
    J,_=boundary_rows(model.mesh,owner,poses);singular=np.linalg.svd(J,compute_uv=False)
    assert singular[5]>5. and singular[6]<1e-12
    I=np.array(model.config['face_inertia']);subdivisions=[]
    for depth in [0,4,12,24,40]:
        dt=25*365.25*86400/(2**depth)
        free=np.array([np.linalg.solve(I[owner==k].sum(axis=0)+dt*model.face_drag[owner==k].sum(axis=0),
            I[owner==k].sum(axis=0)@old[k]+dt*model.face_torque[owner==k].sum(axis=0)) for k in range(3)])
        gaps=J@free.ravel();assert min(gaps)<-1e-20
        subdivisions.append(dict(depth=depth,candidate_step_seconds=dt,
            minimum_initial_normal_velocity_m_s=float(min(gaps)*model.config['radius_km']*1000),
            free_candidate_contact_free_prefix_seconds=0.,executed_free_steps=0))
    model.advance(25.);restart.advance(25.)
    assert encoded(model.state)==encoded(restart.state)
    model.save(out/'common_motion_endpoint.json');L=model.state['last_ledger'];p=model.state['contact']
    result=dict(status='BLOCKED_BY_RIGID_FULL_SPHERE_PACKING',positive_relative_motion_scenario=False,
        new_real_intervals=0,initial_damage=0.,owners_after_each_split=counts,
        split_times_myr=[e['time_myr'] for e in model.state['events']],rigidity=witness,
        normal_constraint_singular_values=singular.tolist(),subdivision_diagnostics=subdivisions,
        whole_interval_certificate=p['whole_step'],common_omega_rad_s=model.state['omega'][0],
        accepted_relative_omega=p['accepted_relative_omega_norm'],free_relative_omega=p['free_relative_omega_norm'],
        momentum_relative=L['momentum_relative'],rotation_balance_j=L['rotation_balance'],
        contact_work_j=p['contact_work_j'],whole_balance_j=L['whole_balance_j'],whole_allowance_j=L['whole_allowance_j'],
        restart_exact=True,seconds=time.perf_counter()-start,
        output_bytes=sum(f.stat().st_size for f in out.iterdir()),artifacts={f.name:digest(f.read_bytes()) for f in out.iterdir()})
    assert result['accepted_relative_omega']==0 and abs(L['whole_balance_j'])<=L['whole_allowance_j']
    path=Path(__file__).with_name('packing_rigidity_result.json')
    with path.open('xb') as handle:handle.write(encoded(result))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
