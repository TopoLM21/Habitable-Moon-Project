"""Explicit coarse retirement of nodal mechanics; not continuation of contact dynamics."""
import numpy as np
from .experimental_edge_fracture import Parameters,certify


def prepare_nodal_retirement(model,fracture_edges):
    from .simplified_genesis_nodal import operators,material_mass,blocks
    if model.config.get('mechanical_mode')!='shared_nodal_moments':raise ValueError('NODAL_RETIREMENT_REQUIRES_NODAL_SOURCE')
    if fracture_edges is None:raise ValueError('CERTIFIED_FRACTURE_EDGES_REQUIRED')
    failed=np.asarray(fracture_edges)
    if failed.dtype.kind!='b' or failed.shape!=(len(model.mesh.shared_edges),):raise ValueError('FRACTURE_EDGE_SHAPE')
    owner,reason=certify(model.mesh,failed,Parameters())
    if owner is None:raise ValueError(reason)
    o=operators(model);X=o['X'];v=np.array(model.state['nodal_velocity']);M=material_mass(model)
    rigid=np.zeros((len(v),3));translation=np.zeros((len(v),3))
    for i,x in enumerate(X):
        cross=np.array([[0.,-x[2],x[1]],[x[2],0.,-x[0]],[-x[1],x[0],0.]])
        rigid[6*i:6*i+3]=-cross;rigid[6*i+3:6*i+6]=-cross;translation[6*i:6*i+3]=np.eye(3)
    T=model.temperature(model.state['energy']);bb=blocks(model,T,np.array(model.state['damage']),T)
    elastic=sum(.5*b['old']@b['H']@b['old'] for b in bb)
    points=X[model.mesh.faces].mean(axis=1);velocity=v.reshape(-1,6)[model.mesh.faces,:3].mean(axis=1)
    weights=np.array(model.config['face_mass_kg']);omegas=[];errors=[]
    for p in range(owner.max()+1):
        ids=owner==p;r=points[ids];u=velocity[ids];w=weights[ids]
        normal=np.eye(3)*np.sum(w*np.sum(r*r,axis=1))-np.einsum('f,fi,fj->ij',w,r,r)
        rhs=np.sum(w[:,None]*np.cross(r,u),axis=0);omega=np.linalg.solve(normal,rhs)
        residual=u-np.cross(omega,r)
        errors.append(float(np.sqrt(np.sum(w*np.sum(residual**2,axis=1))/np.sum(w))))
        omegas.append(omega.tolist())
    return dict(owner=owner.tolist(),fracture_edges=failed.tolist(),boundary_certificate=reason,
        omega_rad_s=omegas,fit_velocity_rms_error_m_s=errors,
        angular_momentum_kg_m2_s=(rigid.T@M@v).tolist(),linear_momentum_kg_m_s=(translation.T@M@v).tolist(),
        kinetic_j=float(.5*v@M@v),elastic_j=float(elastic),
        geometry='canonical face-ID replacement; nodal displacement and slope archived, not evolved after transition',
        velocity_fit='mass-weighted least-squares top-face velocity fit; discarded deformation/radial motion recorded in model change',
        no_mapping_heat=True,SI_mechanical_conservation_claim=False)
