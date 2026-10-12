"""One-time, pinned import into the explicitly different simplified model."""
import json
from pathlib import Path
from dataclasses import asdict
import numpy as np
from tectonics.simplified_genesis import SimplifiedGenesis,VERSION
from tectonics.genesis_transfer_bridge import digest
from tectonics.genesis_volume_basal import local_consistent_mass_matrix
from analysis.genesis_volume_continuation_validation import independent_audit as q
from analysis.genesis_thermal_moment.adapter import MomentAdapter
from analysis.genesis_starter_selector_bridge.adapter import mesh_view

ROOT=Path(__file__).resolve().parents[2]

def cross_matrix(x):
    a,b,c=x;return np.array([[0.,-c,b],[c,0.,-a],[-b,a,0.]])


def build():
    report=json.loads((ROOT/'analysis/genesis_thermal_moment/report.json').read_bytes())
    for path,pin in report['input_sha256'].items():assert digest(Path(path).read_bytes())==pin
    p=json.loads(Path('/workspace/genesis-exploratory-100-01/continuation/initial.json').read_bytes())['packet']
    source=json.loads((ROOT/'analysis/genesis_volume_creation_validation/evidence/actual_continuation/direct_after_second.json').read_bytes())['source_packet']
    full=json.loads(Path('/workspace/genesis-forced-coupled-25-01/candidate.json').read_bytes())
    a=MomentAdapter(p,source,full['thermal_packet']['fixed_cut_geometry'],full['inputs'],report['contact_parameters'])
    mesh=mesh_view(a.X,a.faces);Is=[];Ds=[];taus=[];Ls=[];masses=[];forces=[];dragforces=[];momentummaps=[]
    traction=np.array(full['inputs']['basal_traction_at_vertices_pa'])[a.faces]
    for f,tri in enumerate(a.X[a.faces]):
        cohorts=[dict(c,face_index=0) for c in a.g['cohorts'] if c['face_index']==f]
        M=sum((local_consistent_mass_matrix(c['mass_kg'],c['outer_scale'],c['inner_scale']) for c in cohorts),np.zeros((18,18)))
        D,F=q.basal_drag_and_force(tri,np.array([[0,1,2]]),[a.g['column_mass_kg'][f]],a.g['density_kg_m3'],cohorts,full['inputs']['basal_drag_pa_s_m'],traction[f:f+1])
        J=np.concatenate([np.vstack((-cross_matrix(x),-cross_matrix(x))) for x in tri])
        translate=np.tile(np.vstack((np.eye(3),np.zeros((3,3)))),(3,1))
        forces.append(translate.T@F);dragforces.append(translate.T@D@J);momentummaps.append(translate.T@M@J)
        Is.append(J.T@M@J);Ds.append(J.T@D@J);taus.append(J.T@F);Ls.append(J.T@M@a.v0[a.blocks[f]['dofs']]);masses.append(sum(c['mass_kg'] for c in cohorts))
    I=np.sum(Is,axis=0);L=np.sum(Ls,axis=0);omega=np.linalg.solve(I,L)
    residual_kinetic=float(.5*a.v0@a.M@a.v0-.5*omega@I@omega)
    assert residual_kinetic>=-1e-8
    cfg=dict(mesh=dict(vertices=mesh.vertices.tolist(),faces=mesh.faces.tolist(),centroids=mesh.centroids.tolist(),areas=mesh.areas_unit_sphere.tolist(),neighbors=mesh.neighbors,shared_edges=mesh.shared_edges),
        mass=a.m.tolist(),cp=a.cp.tolist(),latent=a.latent.tolist(),offset=a.off.tolist(),edges=a.edges,
        G=[b['G'].tolist() for b in a.blocks],P=np.array(a.P).tolist(),weights=np.array(a.weights).tolist(),C=[a.C(f,np.zeros(80)).tolist() for f in range(80)],
        solid_nodes=np.column_stack((a.solid_group,a.solid_second)).tolist(),profile_limits=[np.column_stack((np.ones(2),bounds)).tolist() for bounds in a.phi_bounds],
        face_inertia=np.array(Is).tolist(),face_drag=np.array(Ds).tolist(),face_torque=np.array(taus).tolist(),face_mass_kg=masses,face_force=np.array(forces).tolist(),face_drag_force=np.array(dragforces).tolist(),face_linear_momentum_map=np.array(momentummaps).tolist(),
        shell=a.shell,faults=a.parameters['faults'],radius_km=a.gp.radius_km,regularization_km=a.parameters['onset']['regularization_km'],solidus=a.gp.solidus_k,liquidus=a.gp.liquidus_k,
        thermal_parameters=asdict(a.gp),tidal_parameters=asdict(a.tides),source_mode='genesis',
        selector=dict(radius_km=a.gp.radius_km,min_child_area_km2=3.1e6,min_band_span_km=1000.),max_step_years=25.,max_rotation_rad=1e-3,
        passive_reserve_j=a.reserve,passive_kinetic_projection_j=residual_kinetic,source_pins=report['input_sha256'],source_material_catalogue=[{k:c[k] for k in ['id','material_id','owner_id','face_index','mass_kg','q_lo','live_q_hi']} for c in a.g['cohorts']],
        approximation='constrained thermal strain; rigid rotational dynamics; passive unresolved reserve; effective cut; fixed chart limited to 1e-3 rad, no contact/remesh',original_fault_history=p['state']['fault_history'])
    s=dict(format=VERSION,time_myr=a.time,energy=a.E0.tolist(),memory=a.old.reshape(80,2,6).tolist(),damage=a.history.damage.tolist(),candidate_age=a.history.fault_candidate_age_myr.tolist(),consumed=[False]*80,owner=[0]*80,omega=[omega.tolist()],rotation=[np.eye(3).tolist()],membrane_established=a.history.membrane_established,passive_reserve_j=a.reserve,events=[],orbit=asdict(a.orbit),steps=0)
    return SimplifiedGenesis(cfg,s)
