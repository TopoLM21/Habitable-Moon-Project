"""Bounded, explicitly confined laboratory test; no real Genesis intervals."""
from pathlib import Path
import json,time
import numpy as np
from tectonics.simplified_genesis import SimplifiedGenesis
from tectonics.simplified_genesis_edges import enable
from tectonics.genesis_material import rebuild_material_mesh
from tectonics.genesis_transfer_bridge import encoded,digest
from .synthetic import deformable_edge_synthetic


def main():
    start=time.perf_counter();out=Path('/workspace/simplified-genesis-edges-01');out.mkdir(exist_ok=False)
    m=deformable_edge_synthetic();counts=[len(m.state['omega'])]
    for _ in range(2):m.advance(25.);counts.append(len(m.state['omega']))
    assert counts==[1,2,3]
    born=SimplifiedGenesis(m.config,m.state);initial_area=m.mesh.areas_unit_sphere.copy()
    enable(m,1);first=m.advance(.01);m.save(out/'edge_step_1.json')
    copy=SimplifiedGenesis.load(out/'edge_step_1.json');second=m.advance(.01);copy.advance(.01)
    assert encoded(m.state)==encoded(copy.state);m.save(out/'edge_step_2.json')
    def summary(ledger):return {k:v for k,v in ledger.items() if k!='normal_impulses'}
    sensitivity=[]
    for width in (1,2):
        test=SimplifiedGenesis(born.config,born.state);enable(test,width);record=test.advance(.01)
        sensitivity.append(dict(width_rings=width,strip_faces=sum(test.config['edge_strips']['strip_faces']),
            core_vertices=[len(v) for v in test.config['edge_strips']['core_vertices']],ledger=summary(record)))
    fine=SimplifiedGenesis(born.config,born.state);enable(fine,1);fine_first=fine.advance(.005);fine.advance(.005)
    coarse=SimplifiedGenesis.load(out/'edge_step_1.json')
    timestep=dict(coarse_years=.01,fine_years=.005,horizon_years=.01,
        maximum_vertex_difference=float(np.max(abs(np.asarray(coarse.state['edge_material']['vertices'])-fine.state['edge_material']['vertices']))),
        coarse_finite_work_relative=abs(first['finite_geometry_work_defect_j'])/first['work_scale_j'],
        fine_first_finite_work_relative=abs(fine_first['finite_geometry_work_defect_j'])/fine_first['work_scale_j'])
    unconfined=SimplifiedGenesis(born.config,born.state);unconfined.config['edge_prescribed_compressive_pressure_pa']=0.;enable(unconfined,1);pin=encoded(unconfined.state)
    try:unconfined.advance(.01)
    except ValueError as e:unconfined_refusal=str(e)
    else:raise AssertionError('Unconfined control unexpectedly admitted')
    assert encoded(unconfined.state)==pin
    mesh=rebuild_material_mesh(m.mesh,np.asarray(m.state['edge_material']['vertices']))
    area_change=mesh.areas_unit_sphere/initial_area-1
    result=dict(status='CONFINED_SMALL_INCREMENT_EDGE_MOTION',general_plate_motion_ready=False,new_real_intervals=0,
        owners_after_birth_steps=counts,initial_damage=0.,birth_times_myr=[e['time_myr'] for e in born.state['events']],
        mechanical_steps_years=[.01,.01],test_loads=dict(boundary_pressure_pa=m.config['edge_prescribed_compressive_pressure_pa'],
            thermal_expansion_per_k=0.,synthetic_geometry='seed17 tangent perturbation +/-0.02 radians of icosphere3'),
        first=summary(first),second=summary(second),width_sensitivity=sensitivity,timestep_check=timestep,
        restart_next_step_exact=True,maximum_relative_face_area_change=float(max(abs(area_change))),
        total_area_relative_error=float(mesh.areas_unit_sphere.sum()/(4*np.pi)-1),
        no_pressure_control_refused=unconfined_refusal,refusal_transactional=True,
        sources_and_sinks=False,damage_frozen_after_edge_enable=True,
        seconds=time.perf_counter()-start,output_bytes=sum(p.stat().st_size for p in out.iterdir()),
        artifacts={p.name:digest(p.read_bytes()) for p in out.iterdir()})
    assert first['relative_core_omega_rad_s']>1e-15 and second['relative_core_omega_rad_s']>1e-15
    assert abs(first['whole_balance_j'])<first['whole_allowance_j']
    path=Path(__file__).with_name('edge_strips_result.json')
    with path.open('xb') as handle:handle.write(encoded(result))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
