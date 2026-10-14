"""Isolated spherical-map, six-DOF, quadrature-Maxwell perturbation path.

No source history import, thermal evolution, damage evolution or default hook.
"""
from pathlib import Path
import json,time
from types import SimpleNamespace
from tectonics.basal_coupling import mantle_source_pattern
import numpy as np
from tectonics.mesh import build_icosphere
from tectonics.experimental_angular_coupled import AngularCoupled,base_face_energy
from tectonics.experimental_edge_weakening import scale
from tectonics.simplified_genesis_nodal import rate
from tectonics.genesis_transfer_bridge import encoded,digest
from analysis.genesis_aggregate_coupled.core import step
from analysis.genesis_volume_continuation_validation.independent_audit import independent_strain_operators
from .coupled_load_audit import SOURCE,vertex_traction
from .local_spherical_drag_reference import quadrature
VERSION='spherical-map-load-perturbation/v1'
YEAR=31557600.


def solid_angle(tri):
    u=tri/np.linalg.norm(tri,axis=1)[:,None]
    return 2*np.arctan2(abs(np.linalg.det(u)),1+u[0]@u[1]+u[1]@u[2]+u[2]@u[0])


def evaluation(triangle,N,radius):
    y=N@triangle;lam=radius/np.linalg.norm(y,axis=1)
    S=np.zeros((len(N),3,18))
    for i in range(3):
        S[:,:,6*i:6*i+3]=N[:,i,None,None]*np.eye(3)
        S[:,:,6*i+3:6*i+6]=(N[:,i]*(lam-1))[:,None,None]*np.eye(3)
    b0,b1=independent_strain_operators(triangle,np.eye(3),N)
    B=b0+b1/lam[:,None,None]
    return S,B,lam


class SphericalPath:
    def __init__(self,subdivisions=1,order=5,source=SOURCE,load_field="parent_p1",angular_partition=None):
        if load_field not in ("parent_p1","analytic_tangent"):raise ValueError("LOAD_FIELD")
        if subdivisions not in (1,2):raise ValueError('Bounded80/320_only')
        if angular_partition not in (None,2):raise ValueError('ANGULAR_PARTITION')
        start=time.monotonic();self.source=Path(source);self.sha=digest(self.source.read_bytes());a=AngularCoupled.load(source);m=a.model
        self.recipe=dict(version=VERSION,subdivisions=subdivisions,order=order,load_field=load_field,angular_partition=angular_partition,source=str(source),source_sha256=self.sha)
        self.mesh=build_icosphere(subdivisions);self.nf=self.mesh.cell_count;self.nd=6*self.mesh.vertex_count
        R=float(np.mean(np.linalg.norm(m.config['nodal']['X'],axis=1)));self.X=self.mesh.vertices*R
        oldX=np.array(m.config['nodal']['X']);self.X[:len(oldX)]=oldX
        if subdivisions==2:
            base=build_icosphere(1)
            for _,_,i,j in base.shared_edges:
                direction=base.vertices[i]+base.vertices[j];direction/=np.linalg.norm(direction)
                k=np.argmin(np.linalg.norm(self.mesh.vertices-direction,axis=1))
                self.X[k]=R*(oldX[i]+oldX[j])/np.linalg.norm(oldX[i]+oldX[j])
        refmesh=None
        if subdivisions==1 and angular_partition==2:
            refmesh=build_icosphere(2);refX=refmesh.vertices*R;refX[:len(oldX)]=oldX;base=build_icosphere(1)
            for _,_,i,j in base.shared_edges:
                direction=base.vertices[i]+base.vertices[j];direction/=np.linalg.norm(direction)
                k=np.argmin(np.linalg.norm(refmesh.vertices-direction,axis=1))
                refX[k]=R*(oldX[i]+oldX[j])/np.linalg.norm(oldX[i]+oldX[j])
        parents=np.arange(self.nf) if subdivisions==1 else np.arange(self.nf)//4
        parent_tri=np.array(m.config['nodal']['X'])[m.mesh.faces];parent_omega=np.array([solid_angle(t) for t in parent_tri])
        rho=m.config['nodal']['density'];parent_volume=np.asarray(m.config['face_mass_kg'])/rho;beta=m.config['nodal']['moving_basal']['beta_pa_s_m']
        # Stable shell thickness from conserved source volume, no guessed thickness.
        dr=R*np.expm1(np.log1p(-3*parent_volume/(parent_omega*R**3))/3);bottom=R+dr
        traction,coupling=vertex_traction(m);rates=rate(m,m.temperature(m.state['energy'])[m.solid_nodes]);Cs=m.C*scale(m,m.state['damage'])[:,None,None]
        self.M=np.zeros((self.nd,self.nd));self.D=self.M.copy();self.F=np.zeros(self.nd);self.blocks=[];self.face_index=[];self.weights=[];self.Cs=[]
        self.ids=(self.mesh.faces[:,:,None]*6+np.arange(6)).reshape(self.nf,18)
        self.volumes=np.zeros(self.nf);self.force=np.zeros(3);self.torque=np.zeros(3);self.quad_area_errors=[]
        self.patch_error=0.;self.rotation_drag_work=0.;self.rotation_force_work=0.
        for f,ids in enumerate(self.mesh.faces):
            p=parents[f];tri=self.X[ids];omega=solid_angle(tri)
            if refmesh is None:x,n,w,N=quadrature(tri,order)
            else:
                chunks=[]
                for child in range(4):
                    t=refX[refmesh.faces[4*f+child]];xx,nn,ww,_=quadrature(t,order)
                    ww*=R*R*solid_angle(t)/ww.sum();chunks.append((xx,nn,ww))
                x=np.concatenate([z[0] for z in chunks]);n=np.concatenate([z[1] for z in chunks]);w=np.concatenate([z[2] for z in chunks])
                N=n@np.linalg.inv(tri);N/=N.sum(1)[:,None]
            self.quad_area_errors.append(float(w.sum()/(R*R*omega)-1));w=w*(R*R*omega/w.sum())
            angle_weights=w/(R*R);volume=parent_volume[p]*omega/parent_omega[p];self.volumes[f]=volume
            dofs=self.ids[f];Mlocal=np.zeros((18,18))
            # Two conserved-volume radial samples, same inherited Maxwell rates.
            for radial in range(2):
                fraction=.5+(-1 if radial==0 else 1)/(2*np.sqrt(3))
                r=R*np.cbrt(1-3*parent_volume[p]*fraction/(parent_omega[p]*R**3))
                S,B,lam=evaluation(tri,N,r)
                vw=angle_weights/omega*volume/2
                Mlocal+=rho*np.einsum('q,qki,qkj->ij',vw,S,S)
                for j in range(len(N)):
                    self.blocks.append(dict(H=vw[j]*Cs[p],L=B[j],R=np.eye(6)*rates[p,radial],old=np.zeros(6),theta=np.zeros(6),dofs=dofs))
                    self.face_index.append(f);self.weights.append(vw[j]);self.Cs.append(Cs[p])
                # Independent rigid rotation has u=A+(lambda-1)P=omega cross x.
                v=np.tile(np.cross(np.array([.3,-.7,.5]),tri/R)[:,None,:],(1,2,1)).reshape(18)
                self.patch_error=max(self.patch_error, R*float(np.max(abs(np.einsum('qij,j->qi',B,v)))))
            self.M[np.ix_(dofs,dofs)]+=Mlocal
            S,_,_=evaluation(tri,N,bottom[p]);xb=n*bottom[p];area=angle_weights*bottom[p]**2
            tangent=S-n[:,:,None]*np.einsum('qi,qij->qj',n,S)[:,None,:]
            Dlocal=beta*np.einsum('q,qki,qkj->ij',area,tangent,tangent)
            self.D[np.ix_(dofs,dofs)]+=Dlocal
            # One frozen physical vector field on parent spherical patches.
            bary=n@np.linalg.inv(parent_tri[p]);bary/=bary.sum(1)[:,None]
            tau=bary@traction[m.mesh.faces[p]]
            if load_field=='analytic_tangent':
                tau=m.shell['convective_traction_pa']*(bary@coupling[m.mesh.faces[p]])[:,None]*mantle_source_pattern(SimpleNamespace(vertices=n),m.shell['seed'])
            force=area[:,None]*tau;local=np.einsum('qki,qk->i',S,force)
            np.add.at(self.F,dofs,local);self.force+=force.sum(0);self.torque+=np.cross(xb,force).sum(0)
        self.face_index=np.array(self.face_index);self.weights=np.array(self.weights);self.Cs=np.array(self.Cs)
        self.memory=np.zeros((len(self.blocks),6));self.velocity=np.zeros(self.nd);self.elapsed=0.;self.heat=0.;self.cumulative_external=0.;self.cumulative_numerical=0.
        old_mass=float(sum(m.config['face_mass_kg']));self.audit=dict(setup_seconds=time.monotonic()-start,faces=self.nf,ndof=self.nd,blocks=len(self.blocks),radius_m=R,
            old_load_bearing_mass_kg=old_mass,new_load_bearing_mass_kg=float(rho*self.volumes.sum()),
            old_moment_volume_mass_relative=float((rho*m.G[:,0,0].sum()-old_mass)/old_mass),source_elastic_energy_not_imported_j=float(base_face_energy(m)@scale(m,m.state['damage'])),
            thickness_min_m=float((-dr).min()),thickness_max_m=float((-dr).max()),
            rigid_rotation_dimensionless_strain_error=self.patch_error,max_raw_area_quadrature_relative=float(max(abs(np.array(self.quad_area_errors)))),
            initialization='zero perturbation velocity/memory only; source damage/material/rates retained and frozen; no claim of geological-state transfer')
        if subdivisions==1:
            self.audit.update(source_mass_matrix_relative=float(np.linalg.norm(self.M-np.array(m.config['nodal']['M']))/np.linalg.norm(m.config['nodal']['M'])),
                source_force_relative=float(np.linalg.norm(self.F-np.array(m.config['nodal']['F']))/np.linalg.norm(m.config['nodal']['F'])))

    def advance(self,years):
        if not 0<years<=1:raise ValueError('bounded_time_step')
        for block,e in zip(self.blocks,self.memory):block['old']=e
        result=step(self.M,self.D,self.F,self.velocity,years*YEAR,self.blocks);L=result['ledger']
        if L['balance_relative']>5e-9 or L['momentum_relative']>1e-8:raise ValueError('SPHERICAL_BALANCE_GUARD')
        self.memory=np.array(result['memory']);self.velocity=result['velocity'];self.elapsed+=years
        self.heat+=L['maxwell']+L['drag'];self.cumulative_external+=L['external_work'];self.cumulative_numerical+=L['numerical_memory']+L['numerical_kinetic']
        return L

    def stress(self):
        sigma=np.einsum('qij,qj->qi',self.Cs,self.memory);mean=np.zeros((self.nf,6))
        np.add.at(mean,self.face_index,self.weights[:,None]*sigma);mean/=self.volumes[:,None]
        t=np.zeros((self.nf,3,3));t[:,range(3),range(3)]=mean[:,:3]
        for k,(i,j) in enumerate(((0,1),(0,2),(1,2)),3):t[:,i,j]=t[:,j,i]=mean[:,k]
        return t

    def state(self):return dict(memory=self.memory.tolist(),velocity=self.velocity.tolist(),elapsed=self.elapsed,heat=self.heat,external=self.cumulative_external,numerical=self.cumulative_numerical)

    def save(self,path):
        body=dict(recipe=self.recipe,state=self.state())
        with Path(path).open('xb') as f:f.write(encoded(dict(payload=body,sha256=digest(encoded(body)))))

    @classmethod
    def load(cls,path):
        obj=json.loads(Path(path).read_bytes());p=obj['payload'];r=p['recipe']
        if digest(encoded(p))!=obj['sha256'] or r['version']!=VERSION:raise ValueError('SPHERICAL_CHECKSUM')
        if digest(Path(r['source']).read_bytes())!=r['source_sha256']:raise ValueError('SOURCE_CHANGED')
        a=cls(r['subdivisions'],r['order'],r['source'],r.get('load_field','parent_p1'),r.get('angular_partition'));s=p['state'];a.memory=np.array(s['memory']);a.velocity=np.array(s['velocity']);a.elapsed=s['elapsed'];a.heat=s['heat'];a.cumulative_external=s['external'];a.cumulative_numerical=s['numerical'];return a
