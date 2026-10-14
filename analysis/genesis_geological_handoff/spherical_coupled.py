"""Opt-in adapter of spherical quadrature memory to the existing native thermal path."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json,time
import numpy as np
from tectonics.experimental_angular_coupled import AngularCoupled,base_face_energy as old_energy
from tectonics.simplified_genesis import SimplifiedGenesis
from tectonics import simplified_genesis_nodal as native
from tectonics.experimental_edge_weakening import scale,face_damage
from tectonics.basal_coupling import basal_coupling_fraction,mantle_source_pattern
from tectonics.genesis_transfer_bridge import encoded,digest
from .spherical_geometry_path import SphericalPath,evaluation,solid_angle,SOURCE
from .local_spherical_drag_reference import quadrature
from .coupled_refine import refine,rotate_strain
from .integrated_vector_operator import frame_for,ANGULAR_N
VERSION='spherical-native-thermal-angular/v1'


def inertia(X,M):
    J=np.zeros((18,3))
    for i,x in enumerate(X):
        cross=np.array([[0.,-x[2],x[1]],[x[2],0.,-x[0]],[-x[1],x[0],0.]])
        J[6*i:6*i+3]=J[6*i+3:6*i+6]=-cross
    return J.T@M@J


def to_tensor(e):
    t=np.zeros(e.shape[:-1]+(3,3));t[...,range(3),range(3)]=e[...,:3]
    for k,(i,j) in enumerate(((0,1),(0,2),(1,2)),3):t[...,i,j]=t[...,j,i]=e[...,k]
    return t


class SphericalCoupled:
    def __init__(self,m,path,origin):
        self.model=m;self.path=path;self.origin=origin;self.births={};self.geometry=[]
        self.frames=np.array(m.config['nodal']['frames']);self.R=path.audit['radius_m'];self.rho=m.config['nodal']['density'];self.beta=m.config['nodal']['moving_basal']['beta_pa_s_m']
        self.ids=path.ids;self.baseL=np.array([b['L'] for b in path.blocks]);self.f=path.face_index;self.w=path.weights
        self.nodes=np.repeat(m.solid_nodes,25,axis=1).reshape(-1)
        for face,ids in enumerate(m.mesh.faces):
            tri=path.X[ids];x,n,w,N=quadrature(tri,5);omega=solid_angle(tri);wa=w*(omega/w.sum())
            self.geometry.append(dict(tri=tri,n=n,N=N,wa=wa,omega=omega))
        self.validate()

    @classmethod
    def initialize(cls):
        tick=time.monotonic();a=AngularCoupled.load(SOURCE);original=a.model
        refined,_,_=refine(a);c=deepcopy(refined.model.config);s=deepcopy(refined.model.state)
        path=SphericalPath(2,5,load_field='analytic_tangent',angular_partition=2)
        # Replace equal quarters by exact spherical-area fractions throughout the
        # existing conservative thermal/material refinement, reservoirs unchanged.
        volumes=path.volumes;fractions=volumes.reshape(80,4)/volumes.reshape(80,4).sum(1)[:,None];factors=4*fractions.ravel()
        n=len(original.mass);owner=np.full(n-2,-1,dtype=int)
        for f,ids in enumerate(original.solid_nodes):owner[ids]=f
        for row in original.config['phase_cells']:owner[row['node']]=row['face']
        nodefactor=factors.reshape(80,4)[owner].ravel();c['mass'][:4*(n-2)]=(np.array(c['mass'][:4*(n-2)])*nodefactor).tolist()
        s['energy'][:4*(n-2)]=(np.array(s['energy'][:4*(n-2)])*nodefactor).tolist()
        for edge in c['edges']:
            i,j,_=edge;idx=int(i) if i<4*(n-2) else int(j)
            if idx<4*(n-2):edge[2]*=nodefactor[idx]
        for name in ('G','weights','face_mass_kg'):
            ar=np.array(c[name]);c[name]=(ar*factors.reshape((-1,)+(1,)*(ar.ndim-1))).tolist()
        for row in c['source_material_catalogue']:row['mass_kg']*=factors[row['face_index']]
        for row in c['phase_cells']:
            factor=factors[row['face']];row['mass_kg']*=factor;row['volume_m3']*=factor
        c['nodal']['column_mass']=(np.array(c['nodal']['column_mass'])*factors).tolist()
        c['nodal']['X']=path.X.tolist();c['nodal']['M']=path.M.tolist()
        frames=np.array([frame_for(t) for t in path.X[path.mesh.faces]]);c['nodal']['frames']=frames.tolist()
        c['mesh']['vertices']=(path.X/path.audit['radius_m']).tolist();centers=path.X[path.mesh.faces].mean(1);centers/=np.linalg.norm(centers,axis=1)[:,None]
        c['mesh']['centroids']=centers.tolist();c['mesh']['areas']=[solid_angle(t) for t in path.X[path.mesh.faces]]
        # Geometric prolongation: exact value and radial derivative of old field.
        P=np.zeros((path.nd,len(original.state['nodal_velocity'])));oldX=np.array(original.config['nodal']['X'])
        for v,x in enumerate(path.X):
            cf=int(np.flatnonzero(np.any(path.mesh.faces==v,axis=1))[0]);f=cf//4;idx=original.mesh.faces[f]
            N=x@np.linalg.inv(oldX[idx]);lam=N.sum();N/=lam
            for j,node in enumerate(idx):
                P[6*v:6*v+3,6*node:6*node+3]=N[j]*np.eye(3)
                P[6*v:6*v+3,6*node+3:6*node+6]=N[j]*(lam-1)*np.eye(3)
                P[6*v+3:6*v+6,6*node+3:6*node+6]=N[j]*lam*np.eye(3)
        oldM=np.array(original.config['nodal']['M']);oldv=np.array(original.state['nodal_velocity']);v0=P@oldv
        target=oldM@oldv;delta=target-P.T@path.M@v0;v=v0+P@np.linalg.solve(P.T@path.M@P,delta)
        s['nodal_velocity']=v.tolist();s['nodal_displacement']=(P@np.array(original.state['nodal_displacement'])).tolist()
        c['nodal']['initial_displacement']=(P@np.array(original.config['nodal']['initial_displacement'])).tolist()
        eold=np.array(original.state['angular_memory']);oldframes=np.array(original.config['nodal']['frames']);mapped=[]
        for f,idx in enumerate(path.mesh.faces):
            p=f//4;_,normals,_,_=quadrature(path.X[idx],5);N=normals@np.linalg.inv(oldX[original.mesh.faces[p]]);N/=N.sum(1)[:,None]
            interp=N@np.linalg.inv(ANGULAR_N);angular=np.einsum('ai,irj->arj',interp,eold[p])
            radial=np.einsum('ri,aij->arj',original.P[p],angular)
            world=rotate_strain(radial,oldframes[p],np.eye(3))
            mapped.extend(world.transpose(1,0,2).reshape(-1,6))
        s['spherical_memory']=np.array(mapped).tolist();s['spherical_born_memory']={}
        s.pop('angular_memory',None);s.pop('angular_born_memory',None)
        c['experimental_angular_memory']=VERSION;c['experimental_spherical_geometry']=VERSION
        # Native diagnostics retain two radial moments, in the new local frames.
        for f in range(320):
            sample=np.array(mapped).reshape(320,2,25,6)[f].mean(1)
            local=rotate_strain(sample,np.eye(3),frames[f]);s['memory'][f]=np.linalg.solve(np.array(c['P'][f]),local).tolist()
        # Native mean-L fallback is forbidden by the geometry-provider guard.
        c['nodal']['L']=np.zeros((320,12,18)).tolist()
        # Initial inertia derives from exactly the same spherical material mass.
        faceM=np.zeros((320,18,18))
        for f,idx in enumerate(path.mesh.faces):
            _,normals,w,N=quadrature(path.X[idx],5);omega=solid_angle(path.X[idx]);w=w/w.sum()*path.volumes[f]/2
            for sign in (-1,1):
                z=.5+sign/(2*np.sqrt(3));r=path.audit['radius_m']*np.cbrt(1-3*path.volumes[f]*z/(omega*path.audit['radius_m']**3))
                S,_,_=evaluation(path.X[idx],N,r);faceM[f]+=c['nodal']['density']*np.einsum('q,qki,qkj->ij',w,S,S)
        c['face_inertia']=[inertia(path.X[idx],faceM[f]).tolist() for f,idx in enumerate(path.mesh.faces)]
        m=SimplifiedGenesis(c,s);obj=cls(m,path,dict(source=str(SOURCE),sha256=digest(SOURCE.read_bytes())))
        obj.collapse();basal=obj.basal();m.config['nodal']['D']=basal['D'].tolist();m.config['nodal']['F']=basal['F'].tolist()
        before=float(old_energy(original)@scale(original,original.state['damage']));after=float(obj.base_energy()@scale(m,m.state['damage']))
        oldforce=np.zeros(len(oldv))
        for f,ids in enumerate(a.reference.ids):
            for j in range(3):
                H=np.kron(original.G[f],original.C[f]*scale(original,original.state['damage'])[f])/3
                np.add.at(oldforce,ids,a.reference.L[f,j].T@H@eold[f,j].ravel())
        newforce=obj.internal_force();obj.origin['mapping']=dict(seconds=time.monotonic()-tick,elastic_before_j=before,elastic_after_j=after,signed_elastic_change_j=after-before,
            elastic_relative_change=(after-before)/before,kinetic_before_j=float(.5*oldv@oldM@oldv),kinetic_after_j=float(.5*v@path.M@v),
            restricted_momentum_relative=float(np.linalg.norm(P.T@path.M@v-target)/np.linalg.norm(target)),
            internal_force_relative=float(np.linalg.norm(P.T@newforce-oldforce)/np.linalg.norm(oldforce)),
            external_force_relative=float(np.linalg.norm(P.T@basal['F']-np.array(original.config['nodal']['F']))/np.linalg.norm(original.config['nodal']['F'])),
            mass_relative=float((np.sum(m.mass.astype(np.longdouble))-np.sum(original.mass.astype(np.longdouble)))/np.sum(original.mass.astype(np.longdouble))),
            enthalpy_relative=float((np.sum(np.array(s['energy'],np.longdouble))-np.sum(np.array(original.state['energy'],np.longdouble)))/np.sum(np.array(original.state['energy'],np.longdouble))),
            mapping_heat_j=0.,history='Existing minimum-angular-variance reconstruction; mapping energy recorded, not thermalized; source immutable')
        return obj

    def validate(self):
        m=self.model;e=np.asarray(m.state['spherical_memory'])
        if e.shape!=(16000,6) or not np.isfinite(e).all():raise ValueError('SPHERICAL_MEMORY')
        for value in m.state['spherical_born_memory'].values():
            ar=np.asarray(value)
            if ar.shape!=(50,6) or not np.isfinite(ar).all():raise ValueError('SPHERICAL_BORN_SHAPE')
        if set(m.state['spherical_born_memory'])!={x['cell_id'] for x in m.state.get('born_material',[])}:raise ValueError('SPHERICAL_BORN_MEMORY')

    def layer(self,cell):
        key=cell['id']
        if key not in self.births:
            f=cell['face'];g=self.geometry[f];m=self.model;masscol=m.config['nodal']['column_mass'][f]
            rtop=self.R*np.cbrt(1-3*masscol*cell['q_lo']/(self.rho*g['omega']*self.R**3))
            L=[];weights=[];M=np.zeros((18,18))
            for sign in (-1,1):
                z=.5+sign/(2*np.sqrt(3));r=np.cbrt(rtop**3-3*cell['volume_m3']*z/g['omega'])
                S,B,_=evaluation(g['tri'],g['N'],r);w=g['wa']/g['omega']*cell['volume_m3']/2
                M+=self.rho*np.einsum('q,qki,qkj->ij',w,S,S);L.extend(B);weights.extend(w)
            self.births[key]=dict(L=np.array(L),w=np.array(weights),M=M,face=f,node=cell['node'])
        return self.births[key]

    def birth_operator(self,cell):
        row=self.layer(cell);return row['L'].mean(0),row['M']

    def prepare_phase(self,T):
        for c in self.model.config['phase_cells']:
            if T[c['node']]<=self.model.config['solidus']:
                c['inertia']=inertia(self.geometry[c['face']]['tri'],self.layer(c)['M']).tolist()

    def material_mass(self):
        M=self.path.M.copy();catalog={c['id']:c for c in self.model.config['phase_cells']}
        for b in self.model.state.get('born_material',[]):
            row=self.layer(catalog[b['cell_id']]);idx=self.ids[b['face']];M[np.ix_(idx,idx)]+=row['M']
        return M

    def basal(self):
        m=self.model;D=np.zeros_like(self.path.M);F=np.zeros(self.path.nd);rows=[];depth=[];area=[];nodes=m.solid_nodes[:,1].copy()
        mass=np.array(m.config['face_mass_kg']);born=m.state.get('born_material',[])
        for b in born:mass[b['face']]+=b['mass_kg'];nodes[b['face']]=b['node']
        U=np.array(m.state['nodal_displacement']);unit=self.path.X/self.R
        for f,g in enumerate(self.geometry):
            rb=self.R*np.cbrt(1-3*mass[f]/(self.rho*g['omega']*self.R**3));S,_,lam=evaluation(g['tri'],g['N'],rb)
            St,_,_=evaluation(g['tri'],g['N'],self.R);u=U[self.ids[f]].reshape(3,6)
            inv=np.linalg.inv(g['tri'].T);plane=inv.sum(0);grad=np.zeros((25,3,3))
            for j in range(3):
                changing=inv[j]-g['N'][:,j,None]*plane
                ga=changing/lam[:,None];gp=inv[j]-ga
                grad+=np.einsum('i,qj->qij',u[j,:3],ga)+np.einsum('i,qj->qij',u[j,3:],gp)
            A=grad+np.eye(3);det=np.linalg.det(A);cofactor=det[:,None]*np.linalg.solve(A.transpose(0,2,1),g['n'][...,None])[...,0]
            jac=np.linalg.norm(cofactor,axis=1);normal=cofactor/jac[:,None]
            if np.any(det<=0):raise ValueError('SPHERICAL_BASAL_INVERSION')
            bottom=rb*g['n']+np.einsum('qij,j->qi',S,U[self.ids[f]])
            top=self.R*g['n']+np.einsum('qij,j->qi',St,U[self.ids[f]])
            ar=g['wa']*rb**2*jac;dep=np.sum(ar*np.einsum('qi,qi->q',top-bottom,normal))/ar.sum()
            if dep<=0:raise ValueError('SPHERICAL_BASAL_DEPTH')
            tangent=S-normal[:,:,None]*np.einsum('qi,qij->qj',normal,S)[:,None,:]
            local=self.beta*np.einsum('q,qki,qkj->ij',ar,tangent,tangent)
            D[np.ix_(self.ids[f],self.ids[f])]+=local;rows.append(dict(S=S,tangent=tangent,normal=normal,area=ar,D=local,n=g['n'],bottom=bottom));depth.append(dep);area.append(ar.sum())
        coupling=basal_coupling_fraction(np.array(depth)/1000,m.shell['traction_coupling_depth_km']);num=np.zeros(len(unit));den=num.copy()
        np.add.at(num,m.mesh.faces.ravel(),np.repeat(np.array(area)*coupling,3));np.add.at(den,m.mesh.faces.ravel(),np.repeat(area,3))
        cv=num/den
        for f,row in enumerate(rows):
            g=self.geometry[f];tau=m.shell['convective_traction_pa']*(g['N']@cv[m.mesh.faces[f]])[:,None]*mantle_source_pattern(SimpleNamespace(vertices=row['n']),m.shell['seed'])
            row['Ft']=np.einsum('q,qki,qk->i',row['area'],row['S'],tau);np.add.at(F,self.ids[f],row['Ft'])
        return dict(D=D,F=F,Ft=F,rows=rows,nodes=nodes,depth=depth)

    def drag_action(self,basal,v):
        force=np.zeros(self.path.nd)
        for f,row in enumerate(basal['rows']):
            tv=np.einsum('qij,j->qi',row['tangent'],v[self.ids[f]])
            local=self.beta*np.einsum('q,qki,qk->i',row['area'],row['tangent'],tv)
            np.add.at(force,self.ids[f],local)
        return force

    def arrays(self):
        m=self.model;L=self.baseL;w=self.w;f=self.f;nodes=self.nodes;e=np.array(m.state['spherical_memory']);catalog={c['id']:c for c in m.config['phase_cells']}
        for b in m.state.get('born_material',[]):
            row=self.layer(catalog[b['cell_id']]);L=np.concatenate((L,row['L']));w=np.r_[w,row['w']];f=np.r_[f,np.full(50,b['face'])];nodes=np.r_[nodes,np.full(50,b['node'])];e=np.concatenate((e,m.state['spherical_born_memory'][b['cell_id']]))
        return L,w,f,nodes,e

    def base_energy(self):
        m=self.model;_,w,f,_,e=self.arrays();values=.5*w*np.einsum('qi,qij,qj->q',e,m.C[f],e);out=np.zeros(m.nf);np.add.at(out,f,values);return out

    def internal_force(self):
        m=self.model;L,w,f,_,e=self.arrays();sigma=np.einsum('qij,qj->qi',m.C[f]*scale(m,m.state['damage'])[f,None,None],e);F=np.zeros(self.path.nd)
        local=np.einsum('q,qki,qk->qi',w,L,sigma)
        np.add.at(F,self.ids[f].ravel(),local.ravel());return F

    def collapse(self):
        m=self.model;e=np.array(m.state['spherical_memory']).reshape(320,2,25,6);w=self.w.reshape(320,2,25)
        rad=np.sum(w[:,:,:,None]*e,axis=2)/w.sum(2)[:,:,None]
        for f in range(320):m.state['memory'][f]=np.linalg.solve(m.P[f],rotate_strain(rad[f],np.eye(3),self.frames[f])).tolist()
        for b in m.state.get('born_material',[]):
            row=self.layer(next(c for c in m.config['phase_cells'] if c['id']==b['cell_id']));e=np.array(m.state['spherical_born_memory'][b['cell_id']]);avg=np.average(e,axis=0,weights=row['w']);b['memory']=rotate_strain(avg,np.eye(3),self.frames[b['face']]).tolist()

    def mechanical(self,m,T,D,oldT,dt,M,basal):
        L,w,f,nodes,e=self.arrays();current=scale(m,D);previous=scale(m,m.state['damage']);rates=native.rate(m,T[nodes]);unit=np.array([1.,1.,1.,0.,0.,0.]);blocks=[]
        for j in range(len(w)):
            blocks.append(dict(H=w[j]*m.C[f[j]]*current[f[j]],L=L[j],R=np.eye(6)*rates[j],old=e[j],theta=m.shell['linear_expansion_per_k']*(T[nodes[j]]-oldT[nodes[j]])*unit,dofs=self.ids[f[j]]))
        r=native.step(M,basal['D'],basal['F'],np.array(m.state['nodal_velocity']),dt,blocks,refinement_steps=3,drag_action=lambda v:self.drag_action(basal,v))
        if r['ledger']['balance_relative']>5e-9 or r['ledger']['momentum_relative']>1e-8:raise ValueError(f"SPHERICAL_MECHANICAL_BALANCE energy={r['ledger']['balance_relative']} momentum={r['ledger']['momentum_relative']}")
        new=np.array(r['memory']);delta=.5*w*(current[f]-previous[f])*np.einsum('qi,qij,qj->q',e,m.C[f],e)
        Q=np.array([x['maxwell'] for x in r['rows']]);W=np.array([x['thermal_work'] for x in r['rows']]);heat=np.zeros(len(T));np.add.at(heat,nodes,Q-W-delta)
        drag=[]
        for face,row in enumerate(basal['rows']):
            tv=np.einsum('qij,j->qi',row['tangent'],r['velocity'][self.ids[face]]);value=float(dt*self.beta*np.einsum('q,qi,qi->',row['area'],tv,tv));drag.append(value);heat[basal['nodes'][face]]+=value
        if min(drag)<-1e-6:raise ValueError('SPHERICAL_NEGATIVE_DRAG')
        sigma=np.einsum('qij,qj->qi',m.C[f]*current[f,None,None],new);summed=np.zeros((320,6));volume=np.zeros(320);np.add.at(summed,f,w[:,None]*sigma);np.add.at(volume,f,w)
        world=to_tensor(summed/volume[:,None]);local=np.einsum('fai,fab,fbj->fij',self.frames,world,self.frames);stress=np.stack((local[:,0,0],local[:,1,1],local[:,0,1]),axis=1)
        legacy=[]
        for face in range(320):
            sample=new[:16000].reshape(320,2,25,6)[face];ww=self.w.reshape(320,2,25)[face];avg=np.sum(ww[:,:,None]*sample,axis=1)/ww.sum(1)[:,None]
            legacy.append(np.linalg.solve(m.P[face],rotate_strain(avg,np.eye(3),self.frames[face])).ravel())
        born=deepcopy(m.state.get('born_material',[]));bmem={}
        for j,b in enumerate(born):
            z=new[16000+50*j:16000+50*(j+1)];bmem[b['cell_id']]=z.tolist();avg=np.average(z,axis=0,weights=w[16000+50*j:16000+50*(j+1)]);b['memory']=rotate_strain(avg,np.eye(3),self.frames[b['face']]).tolist()
        r.update(memory=legacy,stress=stress,heat=heat,damage_energy_change=float(delta.sum()),born=born,blocks=blocks,spherical_memory=new[:16000].tolist(),spherical_born_memory=bmem,
            basal_budget=dict(heat_j=sum(drag),physical_external_work_j=float(dt*basal['F']@r['velocity']),prescribed_traction_work_j=float(dt*basal['F']@r['velocity']),mantle_drag_work_j=0.,beta_pa_s_m=self.beta,depth_min_m=min(basal['depth']),depth_max_m=max(basal['depth'])))
        return r

    def finalize(self,m,previous,result):
        m.state['spherical_memory']=result['spherical_memory'];m.state['spherical_born_memory']=result['spherical_born_memory']
        for b in m.state.get('born_material',[]):
            if b['cell_id'] not in m.state['spherical_born_memory']:m.state['spherical_born_memory'][b['cell_id']]=np.zeros((50,6)).tolist()
        self.collapse();self.validate()

    def commit(self,m,new):
        previous=np.array(m.state['experimental_edge_fatigue']['damage']);face_damage(m.mesh,new)
        if np.any(new<previous):raise ValueError('EDGE_HEALING_NOT_SUPPORTED')
        before=scale(m,m.state['damage'],previous);after=scale(m,m.state['damage'],new);base=self.base_energy();release=float(base@(before-after))
        if release<0:raise ValueError('SPHERICAL_EDGE_RELEASE')
        m.state['experimental_edge_fracture_reserve_j']+=release;m.state['experimental_edge_fatigue']['damage']=np.array(new).tolist()
        return dict(release_j=release,elastic_before_j=float(base@before),elastic_after_j=float(base@after),scale_before=before,scale_after=after)

    def advance(self,years):
        if not np.isfinite(years) or not 0<years<=1:raise ValueError('SPHERICAL_STEP_RANGE')
        previous=deepcopy(self.model.state)
        try:
            native.advance(self.model,years,mechanical_solver=self.mechanical,state_finalizer=self.finalize,edge_commit_solver=self.commit,geometry_provider=self)
            self.validate()
        except Exception:self.model.state=previous;raise
        return deepcopy(self.model.state['last_ledger'])

    def save(self,path):
        self.validate();body=dict(version=VERSION,origin=self.origin,recipe=self.path.recipe,config=self.model.config,state=self.model.state)
        with Path(path).open('xb') as f:f.write(encoded(dict(payload=body,sha256=digest(encoded(body)))))

    @classmethod
    def load(cls,path):
        obj=json.loads(Path(path).read_bytes());b=obj['payload'];r=b['recipe']
        if digest(encoded(b))!=obj['sha256'] or b['version']!=VERSION:raise ValueError('SPHERICAL_COUPLED_CHECKSUM')
        if digest(Path(r['source']).read_bytes())!=r['source_sha256']:raise ValueError('SPHERICAL_SOURCE_CHANGED')
        p=SphericalPath(r['subdivisions'],r['order'],r['source'],r['load_field'],r.get('angular_partition'))
        return cls(SimplifiedGenesis(b['config'],b['state']),p,b['origin'])
