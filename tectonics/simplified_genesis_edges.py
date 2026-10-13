"""Opt-in small-increment material edge strips; no birth, sink or mature export.

Normal traces are independent on each side. A conforming whole-path geometry
certificate is available only when the solved corner traces coincide. Tangential
sliding that requires changing the contact tessellation is declined, not tied.
"""
from copy import deepcopy
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.linalg import qr
from scipy.optimize import nnls
from .genesis_shell import Membrane
from .genesis_material import rebuild_material_mesh,move_mesh,face_deformation,polar_increment
from .genesis import SECONDS_PER_MYR
from .genesis_volume_operator import engineering_strain_to_tensor,tensor_to_engineering_strain,isotropic_elasticity


def physical_face_inertia(mesh,mass,radius_m):
    x=mesh.vertices[mesh.faces]
    return np.asarray(mass)[:,None,None]*radius_m**2/3*np.sum(np.eye(3)-x[:,:,:,None]*x[:,:,None,:],axis=1)


def enable(model,width_rings):
    if type(width_rings) is not int or width_rings<1:raise ValueError('EDGE_WIDTH_RINGS')
    s=model.state;c=model.config
    if c['source_mode']!='synthetic_prescribed_power' or np.any(c['synthetic_power_w']) or s.get('born_material') or c.get('phase_cells'):
        raise ValueError('EDGE_FIRST_EXPERIMENT_REQUIRES_NO_SOURCES_OR_SINKS')
    if model.shell['linear_expansion_per_k']!=0:
        raise ValueError('EDGE_FIRST_EXPERIMENT_REQUIRES_ZERO_THERMAL_EXPANSION')
    if not np.allclose(model.C,isotropic_elasticity(model.shell['young_modulus_pa'],model.shell['poisson_ratio']),rtol=1e-12,atol=0):
        raise ValueError('EDGE_OBJECTIVE_MEMORY_REQUIRES_ISOTROPIC_ELASTICITY')
    if not np.array_equal(np.asarray(s['rotation']),np.repeat(np.asarray(s['rotation'])[:1],len(s['omega']),axis=0)):
        raise ValueError('EDGE_INITIAL_PARTITION_MUST_BE_CONFORMING')
    if 'edge_strips' in c:raise ValueError('EDGE_ALREADY_ENABLED')
    mesh=rebuild_material_mesh(model.mesh,model.mesh.vertices@np.asarray(s['rotation'][0]).T)
    owner=np.array(s['owner']);strip=np.zeros(model.nf,bool)
    for a,b,_,_ in mesh.shared_edges:
        if owner[a]!=owner[b]:strip[[a,b]]=True
    for _ in range(width_rings-1):
        grown=strip.copy()
        for f in np.flatnonzero(strip):grown[list(mesh.neighbors[f])]=True
        strip=grown
    core=[]
    for k in range(len(s['omega'])):
        inside=set(mesh.faces[owner==k].ravel())-set(mesh.faces[strip].ravel())
        if len(inside)<3:raise ValueError('EDGE_WIDTH_LEAVES_NO_RIGID_CORE')
        core.append([int(v) for v in sorted(inside)])
    mass=np.asarray(c['face_mass_kg']);R=c['radius_km']*1000
    expected=physical_face_inertia(model.mesh,mass,R)
    if not np.allclose(expected,model.face_inertia,rtol=1e-12,atol=0):raise ValueError('EDGE_REQUIRES_MATERIAL_CONSISTENT_INERTIA')
    gamma=float(np.sum(model.face_drag*expected)/np.sum(expected*expected))
    if gamma<0 or not np.allclose(model.face_drag,gamma*expected,rtol=1e-12,atol=1.):raise ValueError('EDGE_FIRST_EXPERIMENT_REQUIRES_MASS_PROPORTIONAL_BASAL_DRAG')
    velocities=[]
    for f in range(model.nf):velocities.append(np.cross(np.asarray(s['omega'][owner[f]]),mesh.vertices[mesh.faces[f]])*R)
    model.config['edge_strips']=dict(width_rings=width_rings,core_vertices=core,strip_faces=strip.tolist(),basal_drag_rate_s=gamma,
        max_increment_rad=1e-5,relative_work_tolerance=1e-5,relative_momentum_tolerance=1e-5)
    model.state['edge_material']=dict(vertices=mesh.vertices.tolist(),face_velocity_m_s=np.asarray(velocities).tolist(),steps=0)
    # Inactive bookkeeping strain in formerly zero-stiffness cells carried no
    # elastic energy. Their restored BULK response starts relaxed; the interface
    # remains disconnected. Never resurrect the already released fracture work.
    memory=np.asarray(model.state['memory']).copy();memory[np.asarray(s['consumed'],bool)]=0
    model.state['memory']=memory.tolist()


def validate(model):
    cfg=model.config['edge_strips'];s=model.state['edge_material']
    if type(cfg['width_rings']) is not int or cfg['width_rings']<1:raise ValueError('EDGE_WIDTH_RINGS')
    rebuild_material_mesh(model.mesh,np.asarray(s['vertices']))
    v=np.asarray(s['face_velocity_m_s'])
    if v.shape!=(model.nf,3,3) or not np.isfinite(v).all():raise ValueError('EDGE_VELOCITY_STATE')
    if len(cfg['core_vertices'])!=len(model.state['omega']):raise ValueError('EDGE_CORE_STATE')


def swept_mesh_certificate(mesh,delta):
    """Positive Bernstein bounds for every triangle and edge for all t in [0,1].

    Vertices follow normalize(x+t*delta). Positive oriented triangles on the
    closed initial degree-one triangulation preserve the spherical embedding.
    Normalization has a positive denominator and cannot change these signs.
    """
    x=mesh.vertices[mesh.faces];d=delta[mesh.faces]
    det=lambda a,b,c:np.einsum('fi,fi->f',a,np.cross(b,c))
    a,b,c=x[:,0],x[:,1],x[:,2];u,v,w=d[:,0],d[:,1],d[:,2]
    p0=det(a,b,c);p1=det(u,b,c)+det(a,v,c)+det(a,b,w)
    p2=det(u,v,c)+det(u,b,w)+det(a,v,w);p3=det(u,v,w)
    bounds=np.stack([p0,p0+p1/3,p0+2*p1/3+p2/3,p0+p1+p2+p3])
    margin=256*np.finfo(float).eps*(np.abs(p0)+np.abs(p1)+np.abs(p2)+np.abs(p3)+1)
    if np.any(bounds<=margin):raise ValueError('EDGE_SWEPT_FACE_DEGENERACY')
    edge_min=1.
    for i,j in [(0,1),(1,2),(2,0)]:
        p0=np.sum(x[:,i]*x[:,j],axis=1);p1=np.sum(x[:,i]*d[:,j]+d[:,i]*x[:,j],axis=1);p2=np.sum(d[:,i]*d[:,j],axis=1)
        edge_min=min(edge_min,float(np.min([p0,p0+p1/2,p0+p1+p2])))
    if edge_min<=256*np.finfo(float).eps:raise ValueError('EDGE_SWEPT_HEMISPHERE_EDGE')
    return dict(kind='positive_bernstein_spherical_isotopy',all_times=True,endpoint_sampling=False,
                minimum_oriented_determinant_bound=float(bounds.min()),minimum_edge_dot_bound=edge_min)


def _assembly(model,mesh):
    """Duplicate material vertices by owner; core rotations are condensed DOFs."""
    cfg=model.config['edge_strips'];owner=np.asarray(model.state['owner']);n=len(model.state['omega']);mem=Membrane(mesh,model.shell['poisson_ratio'])
    cores=[set(x) for x in cfg['core_vertices']];keys=sorted({(int(owner[f]),int(v)) for f,face in enumerate(mesh.faces) for v in face})
    offsets={};nd=3*n
    for k,v in keys:
        if v not in cores[k]:offsets[k,v]=nd;nd+=2
    maps={}
    for k,v in keys:
        if (k,v) in offsets:
            ids=np.arange(offsets[k,v],offsets[k,v]+2);P=mem.vertex_basis[v]
        else:
            ids=np.arange(3*k,3*k+3);P=np.cross(np.eye(3),mesh.vertices[v]).T
        maps[k,v]=(ids,P)
    rows=[];cols=[];values=[]
    br=[];bc=[];bv=[]
    for f,face in enumerate(mesh.faces):
        xyz=mesh.vertices[face];a,b,c=xyz;det=float(a@np.cross(b,c));den=1+a@b+b@c+c@a
        gradients=np.array([2*(den*np.cross(b,c)-det*(b+c))/(den*den+det*det),
                            2*(den*np.cross(c,a)-det*(c+a))/(den*den+det*det),
                            2*(den*np.cross(a,b)-det*(a+b))/(den*den+det*det)])/mesh.areas_unit_sphere[f]
        for j,v in enumerate(face):
            ids,P=maps[int(owner[f]),int(v)]
            for xyzaxis in range(3):
                for col,val in zip(ids,P[xyzaxis]):rows.append(9*f+3*j+xyzaxis);cols.append(col);values.append(val)
            planar=mem.b[f,:,2*j:2*j+2]@mem.vertex_basis[v].T@P
            local=np.zeros((6,len(ids)));local[[0,1,3]]=planar;local[2]=-gradients[j]@P
            for comp in range(6):
                for col,val in zip(ids,local[comp]):br.append(6*f+comp);bc.append(col);bv.append(val)
    P=sparse.coo_matrix((values,(rows,cols)),shape=(9*model.nf,nd)).tocsr()
    B=sparse.coo_matrix((bv,(br,bc)),shape=(6*model.nf,nd)).tocsr()
    jr=[];jc=[];jv=[];contacts=[]
    for fa,fb,u,v in mesh.shared_edges:
        ka,kb=int(owner[fa]),int(owner[fb])
        if ka==kb:continue
        normal=np.cross(mesh.vertices[u],mesh.vertices[v]);normal/=np.linalg.norm(normal)
        if normal@mesh.centroids[fb]<0:normal=-normal
        for vertex in (u,v):
            length=float(np.arccos(np.clip(mesh.vertices[u]@mesh.vertices[v],-1,1)))
            row=len(contacts);contacts.append(dict(owners=[ka,kb],vertex=int(vertex),normal=normal.tolist(),faces=[int(fa),int(fb)],arc_radians=length))
            for k,sign in [(ka,-1),(kb,1)]:
                ids,A=maps[k,int(vertex)]
                for col,val in zip(ids,sign*normal@A):jr.append(row);jc.append(col);jv.append(val)
    J=sparse.coo_matrix((jv,(jr,jc)),shape=(len(contacts),nd)).toarray()
    return mem,P,B,J,maps,contacts


def advance(model,years):
    """One bounded linear-increment mechanics step, with measured finite defects."""
    s=model.state;cfg=model.config['edge_strips'];R=model.config['radius_km']*1000;dt=years/1e6*SECONDS_PER_MYR
    mesh=rebuild_material_mesh(model.mesh,np.asarray(s['edge_material']['vertices']));mem,P,B,J,maps,contacts=_assembly(model,mesh)
    nd=P.shape[1];mass=np.asarray(model.config['face_mass_kg']);lump=np.repeat(mass/3,9);M=P.T@sparse.diags(lump*R*R)@P;D=cfg['basal_drag_rate_s']*M
    old=np.asarray(s['memory']);T=model.temperature(s['energy']);p=model.shell
    nodes=T[model.solid_nodes];eta=np.clip(p['viscosity_reference_pa_s']*np.exp(np.clip(p['activation_energy_j_mol']/8.314462618*(1/nodes-1/p['viscosity_reference_temperature_k']),-60,60)),p['viscosity_min_pa_s'],p['viscosity_max_pa_s']);rates=p['young_modulus_pa']/eta
    relax=np.linalg.solve(model.G,np.einsum('fni,fn,fnj->fij',model.P,model.weights*rates,model.P));A=np.linalg.inv(np.eye(2)[None]+dt*relax)
    # A cut disconnects the two traces, not the bulk material on either side.
    # Consumed cells enter with the zero memory already left by effective split;
    # their damaged bulk stiffness now responds to NEW deformation only.
    C=model.C*model.scale(s['damage'])[:,None,None]
    Aold=A@old;factor=np.einsum('fi,fi->f',model.G[:,0,:],A[:,:,0]);weighted=np.einsum('fi,fij->fj',model.G[:,0,:],Aold)
    blocks=list(C*factor[:,None,None]);blocks[0]=sparse.coo_matrix(blocks[0])
    K=B.T@sparse.block_diag(blocks)@B
    internal=B.T@np.einsum('fij,fj->fi',C,weighted).ravel()
    faceI=physical_face_inertia(mesh,mass,R);acc=np.linalg.solve(faceI,model.face_torque[...,None])[...,0]
    force=mass[:,None,None]/3*np.cross(acc[:,None,:],R*mesh.vertices[mesh.faces])
    external=R*(P.T@force.ravel());oldv=np.asarray(s['edge_material']['face_velocity_m_s'])
    pressure=float(model.config.get('edge_prescribed_compressive_pressure_pa',0.))
    if not np.isfinite(pressure) or pressure<0:raise ValueError('EDGE_PRESSURE_LOAD')
    thickness=model.G[:,0,0]/(mesh.areas_unit_sphere*R*R)
    imposed=np.array([pressure*R*R*c['arc_radians']*float(np.mean(thickness[c['faces']]))/2 for c in contacts])
    # Explicit laboratory loading toward the interface, opposite to contact.
    # It is a prescribed load, not a stiffness, tensile reaction or hidden tie.
    # Eliminate this known constraint load analytically from the primal solve:
    # it shifts the normal multipliers but does no work on a closed trace.
    # This avoids subtracting large pressure terms to recover small motion.
    rhs=R*(P.T@(lump*oldv.ravel()))/dt+external-internal
    H=(M/dt**2+D/dt+K).tocsc();scale=float(abs(H.diagonal()).max())
    # Rank reduction removes duplicate normal rows only; no tangent tie is added.
    _,qrR,piv=qr(J.T,mode='economic',pivoting=True);rank=int(np.sum(abs(np.diag(qrR))>1e-11))
    independent=J[piv[:rank]]
    system=sparse.bmat([[H/scale,-sparse.csc_matrix(independent.T)],[sparse.csc_matrix(independent),None]],format='csc')
    solution=spsolve(system,np.r_[rhs/scale,np.zeros(rank)]);q=solution[:nd]
    if not np.isfinite(q).all():raise ValueError('EDGE_MECHANICAL_SOLVE')
    reaction=H@q-rhs+J.T@imposed
    # Recover normal-only compressive reactions, rejecting tensile solutions.
    reaction_scale=max(float(np.linalg.norm(reaction)),1.)
    lam,_=nnls(J.T,reaction/reaction_scale,maxiter=max(100,20*len(J)));lam*=reaction_scale/scale
    contact_error=float(np.linalg.norm(J.T@lam-reaction/scale)/max(np.linalg.norm(rhs/scale),1e-30))
    if contact_error>1e-8:raise ValueError(f'EDGE_NEEDS_OPENING_OR_CONTACT_CHANGE {contact_error}')
    face_delta=(P@q).reshape(model.nf,3,3)
    delta=np.zeros_like(mesh.vertices);counts=np.zeros(mesh.vertex_count)
    for face,d in zip(mesh.faces,face_delta):np.add.at(delta,face,d);np.add.at(counts,face,1)
    delta/=counts[:,None]
    trace_error=float(np.max(abs(face_delta-delta[mesh.faces])))
    if trace_error>1e-10*max(float(np.max(abs(delta))),1e-12):raise ValueError('EDGE_SLIDING_REQUIRES_NONCONFORMING_PATH_CERTIFICATE')
    if np.max(np.linalg.norm(delta,axis=1))>cfg['max_increment_rad']:raise ValueError('EDGE_SMALL_INCREMENT_LIMIT')
    certificate=swept_mesh_certificate(mesh,delta)
    newmesh=move_mesh(mesh,mem.vertex_basis,np.einsum('vij,vi->vj',mem.vertex_basis,delta))
    # Actual geometric strains are measured independently of linear FEM strains.
    polar,hencky=polar_increment(face_deformation(mesh,newmesh,model.config['radius_km'],model.config['radius_km']))
    strain=(B@q).reshape(model.nf,6);actual=np.zeros_like(strain);actual[:,[0,1,3]]=hencky;actual[:,2]=-np.log(newmesh.areas_unit_sphere/mesh.areas_unit_sphere)
    drive=np.zeros_like(old);drive[:,0,:]=strain;new=A@(old+drive)
    node_strain=np.einsum('fni,fij->fnj',model.P,new);stress=np.einsum('fij,fnj->fni',C,node_strain)
    heat=dt*model.weights*rates*np.sum(node_strain*stress,axis=2)
    energy=lambda e:.5*np.einsum('fai,fab,fij,fbj->',e,model.G,C,e)
    U0=float(energy(old));U1=float(energy(new));Nmemory=float(energy(new-old))
    transform=np.repeat(np.eye(3)[None],model.nf,axis=0);transform[:,:2,:2]=polar
    tensors=engineering_strain_to_tensor(new)
    transported=tensor_to_engineering_strain(transform[:,None]@tensors@transform[:,None].transpose(0,1,3,2))
    rotation_energy_defect=float(energy(transported)-U1)
    dsq=np.sum(delta*delta,axis=1);norm=np.sqrt(1+dsq)
    physical_delta=(delta-mesh.vertices*(dsq/(norm+1))[:,None])/norm[:,None]
    physical_v=R*physical_delta[mesh.faces]/dt
    linear_v=R*face_delta/dt
    K0=float(.5*np.sum(mass[:,None,None]/3*oldv**2));K1=float(.5*np.sum(mass[:,None,None]/3*linear_v**2))
    kinetic_defect=float(.5*np.sum(mass[:,None,None]/3*(physical_v**2-linear_v**2)))
    work=float(external@q-imposed@(J@q));drag=float(q@(D@q)/dt);Nkin=float(.5*np.sum(mass[:,None,None]/3*(linear_v-oldv)**2))
    contact_work=float(scale*lam@(J@q));balance=K1-K0+U1-U0+float(heat.sum())+drag+Nmemory+Nkin-work-contact_work
    mechanical_scale=max(abs(work),abs(U1-U0),float(heat.sum()),1.)
    geometric_work=float(np.einsum('fi,fi->',actual-strain,np.einsum('fij,fj->fi',C,np.einsum('fi,fij->fj',model.G[:,0,:],new))))
    if np.min(heat)<0 or max(abs(balance),abs(geometric_work),abs(kinetic_defect),abs(rotation_energy_defect))>cfg['relative_work_tolerance']*mechanical_scale:raise ValueError('EDGE_WORK_OR_GEOMETRY_DEFECT')
    angular=lambda x,v:np.sum(np.cross(R*x[mesh.faces],mass[:,None,None]/3*v),axis=(0,1))
    L0=angular(mesh.vertices,oldv);L1=angular(newmesh.vertices,physical_v)
    drag_impulse=-dt*cfg['basal_drag_rate_s']*mass[:,None,None]/3*linear_v
    applied=dt*force+drag_impulse
    torque=np.sum(np.cross(R*mesh.vertices[mesh.faces],applied),axis=(0,1))
    angular_error=L1-L0-torque
    momentum_relative=float(np.linalg.norm(angular_error)/max(np.linalg.norm(L0),np.linalg.norm(L1),np.linalg.norm(torque),1.))
    if momentum_relative>cfg['relative_momentum_tolerance']:raise ValueError('EDGE_PHYSICAL_ANGULAR_MOMENTUM_DEFECT')
    nxt=deepcopy(s);E=np.asarray(s['energy']).copy()
    dragface=dt*cfg['basal_drag_rate_s']*mass/3*np.sum(linear_v**2,axis=(1,2))
    for f,ids in enumerate(model.solid_nodes):E[ids]+=heat[f]+dragface[f]/2
    thermal_change=float(np.sum(E.astype(np.longdouble)-np.asarray(s['energy'],np.longdouble)))
    whole=thermal_change+U1-U0+K1+kinetic_defect-K0+Nmemory+Nkin-work-contact_work
    allowance=cfg['relative_work_tolerance']*mechanical_scale+float(4*np.sum(np.spacing(E)+np.spacing(np.asarray(s['energy']))))
    if abs(whole)>allowance:raise ValueError('EDGE_WHOLE_ENERGY_BALANCE')
    model.phase_guard(model.temperature(E))
    nxt['energy']=E.tolist();nxt['memory']=transported.tolist();nxt['time_myr']+=years/1e6;nxt['steps']+=1
    nxt['edge_material']=dict(vertices=newmesh.vertices.tolist(),face_velocity_m_s=physical_v.tolist(),steps=s['edge_material']['steps']+1)
    nxt['omega']=(q[:3*len(s['omega'])].reshape(-1,3)/dt).tolist()
    core_faces=np.array([all(int(v) in cfg['core_vertices'][int(s['owner'][f])] for v in face) for f,face in enumerate(mesh.faces)])
    for item,value in zip(contacts,lam):
        item['point_unit']=mesh.vertices[item['vertex']].tolist()
        item['normal_impulse_n_s']=float(value*scale*dt/R)
    nxt['edge_ledger']=dict(external_work_j=work,drag_heat_j=drag,maxwell_heat_j=float(heat.sum()),elastic_before_j=U0,elastic_after_j=U1,
        numerical_memory_j=Nmemory,numerical_kinetic_j=Nkin,contact_work_j=contact_work,linear_balance_j=balance,
        finite_geometry_work_defect_j=geometric_work,physical_kinetic_defect_j=kinetic_defect,work_scale_j=mechanical_scale,
        tensor_rotation_energy_defect_j=rotation_energy_defect,whole_balance_j=whole,whole_allowance_j=allowance,
        prescribed_pressure_pa=pressure,minimum_normal_multiplier_j=float(np.min(lam*scale)),
        angular_momentum_relative=momentum_relative,compressive_reaction_residual=contact_error,
        angular_momentum_before=L0.tolist(),angular_momentum_after=L1.tolist(),applied_angular_impulse=torque.tolist(),angular_momentum_defect=angular_error.tolist(),
        center_support_impulse_n_s=np.sum(mass[:,None,None]/3*(physical_v-oldv)-applied,axis=(0,1)).tolist(),
        normal_impulses=contacts,max_core_geometric_strain=float(np.max(abs(actual[core_faces]))) if core_faces.any() else None,
        relative_core_omega_rad_s=float(np.linalg.norm(np.diff(np.asarray(nxt['omega']),axis=0))),
        max_strain=float(np.max(abs(actual))),normal_trace_residual_rad=trace_error,
        total_mass_kg=float(mass.sum()),surface_area_km2=float(newmesh.areas_unit_sphere.sum()*(R/1000)**2),
        minimum_thickness_m=float(np.min(model.G[:,0,0]/(newmesh.areas_unit_sphere*R*R))),
        whole_interval=certificate,normal_impulse_count=int(np.count_nonzero(lam)),width_rings=cfg['width_rings'])
    model.state=nxt
    try:model.validate()
    except Exception:model.state=s;raise
    return nxt['edge_ledger']
