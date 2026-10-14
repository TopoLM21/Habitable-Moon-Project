"""Conservative nested 80->320 refinement of the SAME conical polyhedron.

No normalization of physical midpoint X; spherical directions are diagnostic.
Only early, uncut, zero-edge-damage checkpoints are supported here.
"""
from copy import deepcopy
import numpy as np
from tectonics.mesh import build_icosphere
from tectonics.simplified_genesis import SimplifiedGenesis,VERSION as SOURCE_VERSION
from tectonics.simplified_genesis_basal import assemble
from tectonics.genesis_volume_basal import local_consistent_mass_matrix
from tectonics.genesis_transfer_bridge import encoded,digest
from tectonics.experimental_angular_coupled import AngularCoupled,VERSION
from .source_angular import SourceAngular
from .integrated_vector_operator import ANGULAR_N,frame_for
from analysis.genesis_volume_continuation_validation import independent_audit as q


def rotate_strain(e,old_frame,new_frame):
    tensor=np.zeros(e.shape[:-1]+(3,3));tensor[...,range(3),range(3)]=e[...,:3]
    for k,(i,j) in enumerate(((0,1),(0,2),(1,2)),3):tensor[...,i,j]=tensor[...,j,i]=e[...,k]/2
    Q=new_frame.T@old_frame
    t=np.einsum('ij,...jk,lk->...il',Q,tensor,Q)
    return np.stack([t[...,0,0],t[...,1,1],t[...,2,2],2*t[...,0,1],2*t[...,0,2],2*t[...,1,2]],axis=-1)


def raw_operators(m):
    c=m.config;X=np.asarray(c['nodal']['X']);frames=np.asarray(c['nodal']['frames']);result=[]
    for f,ids in enumerate(m.mesh.faces):
        tri=X[ids];height=tri[0]@frames[f,:,2]
        rows=[r for r in c['source_material_catalogue'] if r['face_index']==f]
        bounds,_=q.conical_scales(tri,c['nodal']['column_mass'][f],c['nodal']['density'],np.array([0.,max(x['live_q_hi'] for x in rows)]))
        z=height*(bounds-1);phi=np.asarray(c['profile_limits'][f])[:,1]
        scale=(z[1]-z[0])/(phi[1]-phi[0]);center=z[0]-scale*phi[0]
        b0,b1=q.independent_strain_operators(tri,frames[f],ANGULAR_N);LB=np.zeros((3,2,6,18))
        for row in rows:
            r,w=q.radial_rule(row['q_lo'],row['live_q_hi']);lam,_=q.conical_scales(tri,c['nodal']['column_mass'][f],c['nodal']['density'],r)
            p=np.column_stack((np.ones(len(r)),(height*(lam-1)-center)/scale))
            B=b0[:,None]+b1[:,None]/lam[None,:,None,None]
            LB+=np.einsum('r,rk,arij->akij',w*row['mass_kg']/c['nodal']['density'],p,B)
        result.append([np.linalg.solve(m.G[f],x.reshape(2,-1)).reshape(12,18) for x in LB])
    return np.asarray(result)


def refine(a):
    m=a.model;c=deepcopy(m.config);s=deepcopy(m.state)
    if m.nf!=80 or s.get('born_material') or s['events'] or any(s.get('experimental_edge_fatigue',{}).get('damage',[])):
        raise ValueError('REFINE_ONLY_EARLY_UNCUT_ZERO_EDGE_SOURCE')
    fine=build_icosphere(2);coarse=m.mesh;oldX=np.asarray(c['nodal']['X']);nv=coarse.vertex_count;nd=6*fine.vertex_count
    vertexP=np.zeros((fine.vertex_count,nv));vertexP[:nv]=np.eye(nv)
    for _,_,i,j in coarse.shared_edges:
        direction=coarse.vertices[i]+coarse.vertices[j];direction/=np.linalg.norm(direction)
        k=np.argmin(np.linalg.norm(fine.vertices-direction,axis=1));assert k>=nv
        vertexP[k,i]=vertexP[k,j]=.5
    assert np.all(vertexP.sum(axis=1)==1)
    Pvec=np.kron(vertexP,np.eye(6));X=vertexP@oldX;frames=np.array([frame_for(t) for t in X[fine.faces]])
    n=len(m.mass);ng=4*(n-2)+2
    node=lambda i,k:4*int(i)+k if i<n-2 else 4*(n-2)+(int(i)-(n-2))
    owner=np.full(n-2,-1,dtype=int)
    for f,ids in enumerate(m.solid_nodes):owner[ids]=f
    for row in c['phase_cells']:owner[row['node']]=row['face']
    assert np.all(owner>=0)
    for name in ('mass','cp','latent','offset'):
        value=np.asarray(c[name]);prefix=np.repeat(value[:-2],4)
        if name=='mass':prefix=prefix/4
        c[name]=np.r_[prefix,value[-2:]].tolist()
    s['energy']=np.r_[np.repeat(np.asarray(s['energy'])[:-2]/4,4),s['energy'][-2:]].tolist()
    edges=[]
    for i,j,G in c['edges']:
        if i<n-2 and j<n-2:assert owner[i]==owner[j]
        if i>=n-2 and j>=n-2:edges.append([node(i,0),node(j,0),G])
        else:edges.extend([[node(i,k),node(j,k),G/4] for k in range(4)])
    c['edges']=edges
    c['mesh']=dict(vertices=fine.vertices.tolist(),faces=fine.faces.tolist(),centroids=fine.centroids.tolist(),areas=fine.areas_unit_sphere.tolist(),neighbors=[list(x) for x in fine.neighbors],shared_edges=[list(x) for x in fine.shared_edges])
    for name in ('G','P','weights','C','profile_limits','face_mass_kg','face_inertia','face_drag','face_drag_force','face_force','face_torque','face_linear_momentum_map'):
        values=np.repeat(np.asarray(c[name]),4,axis=0)
        if name not in ('P','C','profile_limits'):values=values/4
        c[name]=values.tolist()
    c['solid_nodes']=[[node(i,k) for i in m.solid_nodes[f]] for f in range(m.nf) for k in range(4)]
    for name in ('damage','candidate_age','consumed','owner'):s[name]=np.repeat(s[name],4).tolist()
    s['experimental_edge_fatigue']=dict(damage=np.zeros(len(fine.shared_edges)).tolist())
    s['nodal_velocity']=(Pvec@np.asarray(s['nodal_velocity'])).tolist();s['nodal_displacement']=(Pvec@np.asarray(s['nodal_displacement'])).tolist()
    old_frames=np.asarray(m.config['nodal']['frames']);oldmemory=np.asarray(m.state['angular_memory']);newmemory=[]
    for f in range(m.nf):
        for k in range(4):
            cf=4*f+k
            parent_coordinates=vertexP[fine.faces[cf]][:,coarse.faces[f]]
            interp=(ANGULAR_N@parent_coordinates)@np.linalg.inv(ANGULAR_N)
            e=np.einsum('ai,irj->arj',interp,oldmemory[f])
            newmemory.append(rotate_strain(e,old_frames[f],frames[cf]))
    s['angular_memory']=np.array(newmemory).tolist();s['memory']=np.array(newmemory).mean(axis=1).tolist();s['angular_born_memory']={}
    for name in ('stress_pa',):
        s.pop(name,None)
    cat=[]
    for row in c['source_material_catalogue']:
        for k in range(4):
            x=deepcopy(row);x.update(face_index=4*row['face_index']+k,id=len(cat)+1,mass_kg=row['mass_kg']/4,
                material_id=row['material_id']+f'/angular-child:{k}',parent_material_id=row['material_id']);cat.append(x)
    c['source_material_catalogue']=cat
    c['nodal']['X']=X.tolist();c['nodal']['frames']=frames.tolist();c['nodal']['column_mass']=np.repeat(np.array(m.config['nodal']['column_mass'])/4,4).tolist()
    c['nodal']['initial_displacement']=(Pvec@np.asarray(m.config['nodal']['initial_displacement'])).tolist()
    ids=(fine.faces[:,:,None]*6+np.arange(6)).reshape(-1,18)
    def local_mass(f,lo,hi,mass):
        scales,_=q.conical_scales(X[fine.faces[f]],c['nodal']['column_mass'][f],c['nodal']['density'],np.array([lo,hi]))
        return local_consistent_mass_matrix(mass,*scales)
    def inertia(f,M):
        J=np.zeros((18,3));translation=np.tile(np.vstack((np.eye(3),np.zeros((3,3)))),(3,1))
        for j,v in enumerate(X[fine.faces[f]]):
            cross=np.array([[0.,-v[2],v[1]],[v[2],0.,-v[0]],[-v[1],v[0],0.]])
            J[6*j:6*j+3]=J[6*j+3:6*j+6]=-cross
        return J.T@M@J,translation.T@M@J,J,translation
    M=np.zeros((nd,nd));faceM=np.zeros((fine.cell_count,18,18))
    for row in cat:faceM[row['face_index']]+=local_mass(row['face_index'],row['q_lo'],row['live_q_hi'],row['mass_kg'])
    for f,idx in enumerate(ids):M[np.ix_(idx,idx)]+=faceM[f]
    c['face_inertia']=[inertia(f,x)[0].tolist() for f,x in enumerate(faceM)]
    c['face_linear_momentum_map']=[inertia(f,x)[1].tolist() for f,x in enumerate(faceM)]
    phases=[]
    for row in m.config['phase_cells']:
        for k in range(4):
            f=4*row['face']+k;x=deepcopy(row)
            x.update(id=row['id']+f'/angular-child:{k}',parent_cell_id=row['id'],face=f,node=node(row['node'],k),mass_kg=row['mass_kg']/4,volume_m3=row['volume_m3']/4,
                inertia=inertia(f,local_mass(f,row['q_lo'],row['q_hi'],row['mass_kg']/4))[0].tolist());phases.append(x)
    c['phase_cells']=phases
    c['nodal'].update(M=M.tolist(),D=np.zeros_like(M).tolist(),F=np.zeros(nd).tolist(),L=np.zeros((fine.cell_count,12,18)).tolist(),face_D=np.zeros_like(faceM).tolist())
    c['approximation']='Nested angular FE refinement on unchanged source polyhedron; conservative thermal/material fragments; provenance in parent fields'
    refined=SimplifiedGenesis(c,s)
    raw=raw_operators(refined);refined.config['nodal']['L']=raw.mean(axis=1).tolist()
    basal=assemble(refined)
    refined.config['nodal'].update(D=basal['D'].tolist(),F=basal['F'].tolist(),face_D=[x['D'].tolist() for x in basal['faces']])
    for f,row in enumerate(basal['faces']):
        _,_,J,tr=inertia(f,faceM[f])
        for name,value in [('face_drag',J.T@row['D']@J),('face_drag_force',tr.T@row['D']@J),('face_force',tr.T@row['Ft']),('face_torque',J.T@row['Ft'])]:refined.config[name][f]=value.tolist()
    # Reference recipe retains the original moments for each material fragment.
    body=dict(config=refined.config,state=refined.state);envelope=dict(format=SOURCE_VERSION,payload=body,sha256=digest(encoded(body)))
    reference=SourceAngular(envelope)
    result=AngularCoupled(reference,refined,{**deepcopy(a.origin), 'nested_refinement':dict(parent_faces=80,faces=320,physical_geometry='unchanged planar facets',thermal_partition='quarters, shared global reservoirs',parent_origin=a.origin)})
    oldM=np.array(m.config['nodal']['M']);restricted=Pvec.T@M@Pvec
    oldbasal=assemble(m)
    audit=dict(mass_matrix_restriction_relative=float(np.linalg.norm(restricted-oldM)/np.linalg.norm(oldM)),
        mass_sum_relative=float(abs(np.sum(np.asarray(refined.mass,np.longdouble))-np.sum(np.asarray(m.mass,np.longdouble)))/np.sum(np.asarray(m.mass,np.longdouble))),
        enthalpy_sum_relative=float(abs(np.sum(np.asarray(s['energy'],np.longdouble))-np.sum(np.asarray(m.state['energy'],np.longdouble)))/np.sum(np.asarray(m.state['energy'],np.longdouble))),
        kinetic_before_j=float(.5*np.array(m.state['nodal_velocity'])@oldM@np.array(m.state['nodal_velocity'])),
        kinetic_after_j=float(.5*np.array(s['nodal_velocity'])@M@np.array(s['nodal_velocity'])),
        restricted_force_relative=float(np.linalg.norm(Pvec.T@basal['F']-oldbasal['F'])/np.linalg.norm(oldbasal['F'])),
        drag_restriction_relative=float(np.linalg.norm(Pvec.T@basal['D']@Pvec-oldbasal['D'])/np.linalg.norm(oldbasal['D'])),
        scalar_nodes_before=n,scalar_nodes_after=ng,source_cohorts=len(m.config['source_material_catalogue']),fragments=len(cat))
    return result,Pvec,audit
