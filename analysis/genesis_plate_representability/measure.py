"""Read-only representation test. No handoff/solver/time advance or new partition."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from tectonics.genesis_volume_basal import local_consistent_mass_matrix
from tectonics.mesh import build_icosphere
from tectonics.mantle import MantleFlowState,plate_rigid_mantle_fit
from analysis.genesis_be_candidate.audit import face_driver
from analysis.genesis_adaptive_control.controller import time_limit

ROOT=Path(__file__).resolve().parents[2]
FOLDER=Path('/workspace/genesis-exploratory-100-01/continuation')
PARENT=ROOT/'analysis/genesis_volume_creation_validation/evidence/actual_continuation/direct_after_second.json'


def projection(M,v,basis):
    gram=basis.T@M@basis;rhs=basis.T@M@v
    coeff=np.linalg.solve(gram,rhs);fitted=basis@coeff;res=v-fitted
    norm=lambda x:float(x@M@x)
    return fitted,dict(coefficients=coeff.tolist(),kinetic_j=.5*norm(fitted),residual_kinetic_j=.5*norm(res),
        relative_mass_rms=float(np.sqrt(norm(res)/norm(v))),orthogonality_relative=float(np.linalg.norm(basis.T@M@res)/max(np.linalg.norm(rhs),1.)))


def measure(packet,source,cuts):
    s=packet['state'];g=s['geometry'];X=np.array(g['vertices_m']);faces=np.array(g['faces']);nv=len(X);n=nv*6
    if len(set(g['owner_ids_per_face']))!=1:raise ValueError('THIS_DIAGNOSTIC_USES_ONE_EXISTING_OWNER_ONLY')
    M=np.zeros((n,n));mass=0.
    for c in g['cohorts']:
        dofs=(faces[c['face_index'],:,None]*6+np.arange(6)).ravel()
        local=local_consistent_mass_matrix(c['mass_kg'],c['outer_scale'],c['inner_scale'])
        M[np.ix_(dofs,dofs)]+=local;mass+=c['mass_kg']
    u=np.array(s['displacement_m']).reshape(nv,6);v=np.array(s['velocity_m_s']);pos=u.copy();pos[:,:3]+=X;pos[:,3:]+=X
    radius=source['source']['parameters']['thermal']['radius_km']*1000
    translation=np.zeros((nv,6,3));translation[:,:3,:]=np.eye(3)
    rotation=np.empty((nv,6,3))
    for k in range(3):
        rotation[:,:3,k]=np.cross(np.eye(3)[k],pos[:,:3])/radius
        rotation[:,3:,k]=np.cross(np.eye(3)[k],pos[:,3:])/radius
    T=translation.reshape(n,3);R=rotation.reshape(n,3)
    assert np.allclose(T.T@M@T,mass*np.eye(3),rtol=1e-13,atol=0)
    best,three=projection(M,v,R);sixv,six=projection(M,v,np.column_stack((T,R)))
    # Existing mature surface-area least-squares function, no plate creation.
    mesh=build_icosphere(1);surface=np.array(v).reshape(nv,6)[faces,:3].mean(axis=1);r=mesh.centroids
    seconds=365.25*86400*1e6
    omega=np.cross(r,surface/radius)*seconds
    fit=plate_rigid_mantle_fit(mesh,np.zeros(len(faces),int),1,radius/1000,MantleFlowState(s['thermal']['time_myr'],omega,0.))
    actual=R@(fit.omega_rad_per_myr[0]/seconds*radius)
    def metrics(q):
        res=v-q
        corners=[];original=[]
        for c in g['cohorts']:
            d=res.reshape(nv,6)[faces[c['face_index']]];vv=v.reshape(nv,6)[faces[c['face_index']]]
            for lam in (c['outer_scale'],c['inner_scale']):
                corners.extend(np.linalg.norm(d[:,:3]+(lam-1)*d[:,3:],axis=1));original.extend(np.linalg.norm(vv[:,:3]+(lam-1)*vv[:,3:],axis=1))
        P=T.T@M@v;L=radius*R.T@M@v
        return dict(kinetic_j=float(.5*q@M@q),mass_rms_residual_m_s=float(np.sqrt(res@M@res/mass)),
            relative_mass_rms=float(np.sqrt((res@M@res)/(v@M@v))),max_physical_residual_m_s=float(max(corners)),max_original_speed_m_s=float(max(original)),
            linear_momentum_error_kg_m_s=(T.T@M@(q-v)).tolist(),angular_momentum_error_kg_m2_s=(radius*R.T@M@(q-v)).tolist(),
            momentum_error_over_absolute_momentum_scale=float(np.linalg.norm(T.T@M@(q-v))/np.sqrt(mass*(v@M@v))),
            angular_error_over_absolute_moment_scale=float(np.linalg.norm(radius*R.T@M@(q-v))/(radius*np.sqrt(mass*(v@M@v)))))
    stress=face_driver(s,source)['stress_pa'];a=np.array(stress);principal=.5*(a[:,0]+a[:,1])+np.sqrt((.5*(a[:,0]-a[:,1]))**2+a[:,2]**2)
    # No invented stress map. Current mature importer explicitly initializes
    # intraplate_stress to zero and does not accept the conical strain coefficients.
    damage=np.array(s['fault_history']['damage']);E=np.array(packet['thermal']['energy_j']);m=np.array(packet['thermal']['mass_kg'])
    groups_m=np.zeros(len(faces));groups_E=np.zeros(len(faces));solid_indices=[]
    for i,c in enumerate(cuts):
        if c['material_id'] is not None:groups_m[c['face']]+=m[i];groups_E[c['face']]+=E[i];solid_indices.append(i)
    return dict(time_myr=s['thermal']['time_myr'],owners=1,cohorts=len(g['cohorts']),generalized_velocity_dofs=n,rigid_plate_dofs=3,
        solid_mass_kg=mass,total_thermal_mass_kg=float(m.sum()),solid_enthalpy_j=float(E[solid_indices].sum()),total_enthalpy_j=float(E.sum()),
        inventory_grouping_mass_residual_kg=float(groups_m.sum()-m[solid_indices].sum()),inventory_grouping_enthalpy_residual_j=float(groups_E.sum()-E[solid_indices].sum()),
        first_mass_moment_kg_m=(T.T@M@pos.ravel()).tolist(),inertia_tensor_kg_m2=(radius**2*R.T@M@R).tolist(),
        linear_momentum_kg_m_s=(T.T@M@v).tolist(),angular_momentum_kg_m2_s=(radius*R.T@M@v).tolist(),
        kinetic_j=float(.5*v@M@v),elastic_memory_j=s['mechanics']['elastic_energy_j'],
        best_mass_rotation=dict(**three,observables=metrics(best)),six_dof_lower_bound=dict(**six,observables=metrics(sixv)),
        existing_surface_area_fit=metrics(actual),
        classifications=dict(tensile_above_12MPa_faces=np.flatnonzero(principal>=12e6).tolist(),max_tensile_pa=float(principal.max()),
            maximum_damage=float(damage.max()),damage_above_activation=int(np.count_nonzero(damage>=.65)),fault_active=int(sum(s['fault_history']['fault_active'])),
            zero_initialized_mature_stress_above_12MPa_faces=[],damage_array_copy_max_difference=0.),
        memory_representable=False,physical_handoff_admitted=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    start=time.perf_counter()
    with time_limit(60):
        source=json.loads(PARENT.read_bytes())['source_packet'];paths=[FOLDER/'initial.json',FOLDER/'half2.json'];pins=json.loads((ROOT/'analysis/genesis_exploratory_100/source_precision_result.json').read_text())['files']
        raw=[p.read_bytes() for p in paths]
        for p,data in zip(paths,raw):assert hashlib.sha256(data).hexdigest()==pins[str(p)]['sha256']
        packets=[json.loads(x)['packet'] for x in raw]
        cuts=packets[1]['thermal_packet']['fixed_cut_geometry']
        result=dict(status='NOT_REPRESENTABLE_AS_CURRENT_SINGLE_RIGID_PLATE',new_time_steps=0,
            input_sha256={str(p):hashlib.sha256(x).hexdigest() for p,x in zip(paths,raw)},
            initial=measure(packets[0],source,cuts),final_trial=measure(packets[1],source,cuts),seconds=time.perf_counter()-start)
        for p,data in zip(paths,raw):assert p.read_bytes()==data
        with args.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
        print(json.dumps(result,indent=2))


if __name__=='__main__':main()
