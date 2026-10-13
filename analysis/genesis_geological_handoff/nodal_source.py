"""Attach authenticated pre-reduction nodal operators to the earliest snapshot."""
from copy import deepcopy
import hashlib,json
from pathlib import Path
import numpy as np
from tectonics.simplified_genesis import SimplifiedGenesis
from analysis.genesis_thermal_moment.adapter import MomentAdapter
from analysis.genesis_volume_continuation_validation import independent_audit as q

ROOT=Path(__file__).resolve().parents[2]
SOURCE=Path('/workspace/simplified-genesis-phase-pilot-01/initial.json')


def build():
    report=json.loads((ROOT/'analysis/genesis_thermal_moment/report.json').read_bytes())
    for path,pin in report['input_sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=pin:raise ValueError('NODAL_SOURCE_PIN')
    packet=json.loads(Path('/workspace/genesis-exploratory-100-01/continuation/initial.json').read_bytes())['packet']
    source=json.loads((ROOT/'analysis/genesis_volume_creation_validation/evidence/actual_continuation/direct_after_second.json').read_bytes())['source_packet']
    full=json.loads(Path('/workspace/genesis-forced-coupled-25-01/candidate.json').read_bytes())
    a=MomentAdapter(packet,source,full['thermal_packet']['fixed_cut_geometry'],full['inputs'],report['contact_parameters'])
    m=SimplifiedGenesis.load(SOURCE);c,s=deepcopy(m.config),deepcopy(m.state)
    assert s['time_myr']==a.time==.8684000000000005 and s['steps']==0
    assert not s.get('born_material') and not s['events']
    np.testing.assert_array_equal(s['energy'],a.E0)
    np.testing.assert_array_equal(np.array(s['memory']).reshape(80,12),a.old)
    np.testing.assert_array_equal(s['damage'],a.history.damage)
    np.testing.assert_array_equal(c['G'],[b['G'] for b in a.blocks])
    np.testing.assert_array_equal(c['mass'],a.m)
    face_D=[];traction=np.asarray(full['inputs']['basal_traction_at_vertices_pa'])[a.faces]
    for f,tri in enumerate(a.X[a.faces]):
        cohorts=[dict(x,face_index=0) for x in a.g['cohorts'] if x['face_index']==f]
        D,_=q.basal_drag_and_force(tri,np.array([[0,1,2]]),[a.g['column_mass_kg'][f]],a.g['density_kg_m3'],cohorts,full['inputs']['basal_drag_pa_s_m'],traction[f:f+1])
        face_D.append(D)
    nodal=dict(M=a.M.tolist(),D=a.D.tolist(),F=a.F.tolist(),X=a.X.tolist(),frames=a.frames.tolist(),L=[b['L'].tolist() for b in a.blocks],face_D=np.array(face_D).tolist(),column_mass=a.g['column_mass_kg'],density=a.g['density_kg_m3'],initial_displacement=a.u0.tolist(),source_file=str(SOURCE),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest())
    restored=float(.5*a.v0@a.M@a.v0);old=float(.5*np.asarray(s['omega'][0])@np.sum(c['face_inertia'],axis=0)@np.asarray(s['omega'][0]))
    assert abs(restored-old-c['passive_kinetic_projection_j'])<1e-10*max(restored,1.)
    nodal['kinetic_import']=dict(full_kinetic_j=restored,old_rigid_kinetic_j=old,reactivated_passive_kinetic_j=c['passive_kinetic_projection_j'])
    c.update(mechanical_mode='shared_nodal_moments',nodal=nodal,passive_kinetic_projection_j=0.,approximation='shared nodal two-moment Maxwell on fixed reference chart; pre-partition only; fixed thermal graph; passive elastic reserve retained')
    s.update(nodal_velocity=a.v0.tolist(),nodal_displacement=a.u0.tolist())
    return SimplifiedGenesis(c,s).with_condensation()


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    build().save(args.output)
