"""Explicit minimum-variance initialization of REAL source radial moments.

Mechanical slice only: source thermal inventory, old damage and phase front are
frozen. Maxwell and basal heat enter a separate buffer, not source enthalpy.
"""
from copy import copy,deepcopy
from pathlib import Path
import json
import numpy as np
from tectonics.simplified_genesis import SimplifiedGenesis
from tectonics.simplified_genesis_nodal import material_mass, rate
from tectonics.simplified_genesis_basal import assemble
from tectonics.experimental_edge_weakening import face_damage
from tectonics.experimental_edge_fracture import certify, Parameters
from tectonics.genesis_transfer_bridge import encoded, digest
from analysis.genesis_aggregate_coupled.core import step as maxwell_step
from analysis.genesis_volume_continuation_validation import independent_audit as q
from .integrated_vector_operator import ANGULAR_N

VERSION='real-source-angular-moments/v1'
YEAR=31557600.


class SourceAngular:
    def __init__(self,envelope):
        body=envelope['payload']
        if digest(encoded(body))!=envelope['sha256']:raise ValueError('SOURCE_CHECKSUM')
        self.envelope=deepcopy(envelope)
        self.m=m=SimplifiedGenesis(body['config'],body['state'])
        if m.state.get('born_material'):raise ValueError('NEWBORN_ANGULAR_MAPPING_NOT_IMPLEMENTED')
        if m.config['mechanical_mode']!='shared_nodal_moments' or not m.config['nodal'].get('moving_basal'):raise ValueError('SOURCE_MODE')
        if len(m.state['omega'])!=1 or m.state['events'] or any(m.state['consumed']):raise ValueError('SOURCE_TOPOLOGY')
        self.nf=m.nf;self.M=material_mass(m);self.nd=len(self.M)
        self.ids=(m.mesh.faces[:,:,None]*6+np.arange(6)).reshape(m.nf,18)
        self.oldL=np.asarray(m.config['nodal']['L']);self.frames=np.asarray(m.config['nodal']['frames'])
        self.L,self.projection_audit=self.angular_operators()
        T=m.temperature(m.state['energy']);rates=rate(m,T[m.solid_nodes])
        self.R=np.linalg.solve(m.G,np.einsum('fni,fn,fnj->fij',m.P,m.weights*rates,m.P))
        self.signature=digest(encoded(dict(source=envelope['sha256'],L=self.L.tolist(),version=VERSION)))

    def angular_operators(self):
        m=self.m;X=np.asarray(m.config['nodal']['X']);L=[];gerr=[];lerr=[];supports=[]
        for f,ids in enumerate(m.mesh.faces):
            tri=X[ids];frame=self.frames[f];height=tri[0]@frame[:,2]
            rows=[c for c in m.config['source_material_catalogue'] if c['face_index']==f]
            lo=min(c['q_lo'] for c in rows);hi=max(c['live_q_hi'] for c in rows)
            bounds,_=q.conical_scales(tri,m.config['nodal']['column_mass'][f],m.config['nodal']['density'],np.array([lo,hi]))
            z=height*(bounds-1);phi=np.array(m.config['profile_limits'][f])[:,1]
            scale=(z[1]-z[0])/(phi[1]-phi[0]);center=z[0]-scale*phi[0]
            B0,B1=q.independent_strain_operators(tri,frame,ANGULAR_N)
            G=np.zeros((2,2));LB=np.zeros((3,2,6,18));volume=0.
            for c in rows:
                radial,wr=q.radial_rule(c['q_lo'],c['live_q_hi'])
                lam,_=q.conical_scales(tri,m.config['nodal']['column_mass'][f],m.config['nodal']['density'],radial)
                P=np.column_stack((np.ones(len(lam)),(height*(lam-1)-center)/scale))
                w=wr*c['mass_kg']/m.config['nodal']['density'];volume+=w.sum()
                G+=P.T@(w[:,None]*P)
                B=B0[:,None]+B1[:,None]/lam[None,:,None,None]
                LB+=np.einsum('r,rk,arij->akij',w,P,B)
            raw=np.array([np.linalg.solve(m.G[f],x.reshape(2,-1)).reshape(12,18) for x in LB])
            gerr.append(np.linalg.norm(G-m.G[f])/np.linalg.norm(m.G[f]))
            lerr.append(np.linalg.norm(raw.mean(axis=0)-self.oldL[f])/np.linalg.norm(self.oldL[f]))
            L.append(raw-raw.mean(axis=0)+self.oldL[f])
            supports.append(dict(face=f,cohorts=len(rows),q_lo=lo,q_hi=hi,volume_m3=volume,
                solid_nodes=m.solid_nodes[f].tolist(),center_m=float(center),scale_m=float(scale)))
        if max(gerr)>1e-8 or max(lerr)>1e-8:raise ValueError(f'RADIAL_PROJECTION_MISMATCH {max(gerr)} {max(lerr)}')
        return np.array(L),dict(max_G_relative=max(gerr),max_mean_L_relative=max(lerr),supports=supports)

    def stiffness(self,damage):
        m=self.m
        scale=m.scale(np.maximum(m.state['damage'],face_damage(m.mesh,damage)))
        return np.array([np.kron(m.G[f],m.C[f]*scale[f]) for f in range(m.nf)])

    def energy_force_stress(self,memory,damage):
        e=np.asarray(memory).reshape(self.nf,3,12);H=self.stiffness(damage)
        energy=float(np.einsum('fai,fij,faj->',e,H,e)/6)
        force=np.zeros(self.nd)
        np.add.at(force,self.ids.ravel(),(np.einsum('faij,fik,fak->fj',self.L,H,e)/3).ravel())
        coeff=e.mean(axis=1).reshape(self.nf,2,6)
        mean=np.einsum('fi,fij->fj',self.m.G[:,0,:],coeff)/self.m.G[:,0,0,None]
        C=self.m.C*self.m.scale(np.maximum(self.m.state['damage'],face_damage(self.m.mesh,damage)))[:,None,None]
        local=np.einsum('fij,fj->fi',C,mean);tensor=np.zeros((self.nf,3,3));tensor[:,range(3),range(3)]=local[:,:3]
        for k,(i,j) in enumerate(((0,1),(0,2),(1,2)),3):tensor[:,i,j]=tensor[:,j,i]=local[:,k]
        world=np.einsum('fai,fij,fbj->fab',self.frames,tensor,self.frames)
        return energy,force,world

    def initial(self,alternative=False):
        m=self.m;edge=np.array(m.state.get('experimental_edge_fatigue',{}).get('damage',np.zeros(len(m.mesh.shared_edges))))
        if digest(encoded(dict(config=m.config,state=m.state)))!=self.envelope['sha256']:raise ValueError('SOURCE_MUTATED')
        e=np.repeat(np.asarray(m.state['memory']).reshape(m.nf,1,12),3,axis=1)
        baseline=e.copy();scale=0.
        old_energy=self.energy_force_stress(e,edge)[0]
        if alternative:
            u=np.asarray(m.state['nodal_displacement'])
            fluctuation=np.einsum('faij,fj->fai',self.L-self.oldL[:,None],u[self.ids])
            fluctuation-=fluctuation.mean(axis=1)[:,None]
            variance=self.energy_force_stress(fluctuation,edge)[0]
            scale=min(1.,np.sqrt(.01*old_energy/variance)) if variance else 0.
            e+=scale*fluctuation
        v=np.asarray(m.state['nodal_velocity'])
        state=dict(version=VERSION,operator_sha256=self.signature,elapsed_years=0.,memory=e.tolist(),
            source_time_myr=m.state['time_myr'],time_myr=m.state['time_myr'],source_file_sha256=digest(encoded(self.envelope)),
            velocity=v.tolist(),displacement=deepcopy(m.state['nodal_displacement']),edge_damage=edge.tolist(),
            reserve_j=float(m.state.get('experimental_edge_fracture_reserve_j',0.)),heat_buffer_j=0.,cumulative={},
            initialization=dict(method='displacement-direction capped at1percent energy' if alternative else 'minimum-energy angular-uniform extension',
                angular_history_reconstructed_not_saved=True,alternative_amplitude=scale,mapping_heat_j=0.),
            certificate='NOT_SEPARATING')
        E,F,S=self.energy_force_stress(e,edge);H=self.stiffness(edge);old=np.asarray(m.state['memory']).reshape(m.nf,12)
        original_energy=float(.5*np.einsum('fi,fij,fj->',old,H,old))
        oldF=np.zeros(self.nd);np.add.at(oldF,self.ids.ravel(),np.einsum('fij,fik,fk->fj',self.oldL,H,old).ravel())
        state.update(initial_elastic_j=E,initial_kinetic_j=float(.5*v@(self.M@v)),initial_reserve_j=state['reserve_j'])
        referenceS=self.energy_force_stress(baseline,edge)[2]
        state['mapping']=dict(elastic_before_j=original_energy,elastic_after_j=E,signed_energy_difference_j=E-original_energy,
            elastic_relative_change=(E-original_energy)/max(original_energy,1.),
            force_relative_change=float(np.linalg.norm(F-oldF)/max(np.linalg.norm(oldF),1.)),
            mean_stress_relative_change=float(np.linalg.norm(S-referenceS)/max(np.linalg.norm(referenceS),1.)),
            radial_coefficients_max_error=float(abs(e.mean(axis=1)-old).max()),
            generalized_momentum_change=float(np.linalg.norm(self.M@np.asarray(state['velocity'])-self.M@v)),
            kinetic_change_j=float(.5*np.asarray(state['velocity'])@self.M@np.asarray(state['velocity'])-.5*v@self.M@v),mass_enthalpy_catalogue_exact=True,
            retained_source_payload_sha256=self.envelope['sha256'],unknown_angular_energy='not identifiable; primary is minimum variance',mapping_heat_j=0.)
        if not alternative and max(abs(state['mapping']['elastic_relative_change']),state['mapping']['force_relative_change'])>1e-10:raise ValueError('INITIALIZATION_REJECTED')
        return state

    def validate(self,s):
        if s.get('version')!=VERSION or s.get('operator_sha256')!=self.signature:raise ValueError('SOURCE_ANGULAR_RESTART_MISMATCH')
        for key,shape in [('memory',(self.nf,3,12)),('velocity',(self.nd,)),('displacement',(self.nd,))]:
            a=np.asarray(s[key])
            if a.shape!=shape or not np.isfinite(a).all():raise ValueError('SOURCE_ANGULAR_STATE')
        face_damage(self.m.mesh,s['edge_damage'])
        if min(s['heat_buffer_j'],s['reserve_j'])<0:raise ValueError('NEGATIVE_RESERVOIR')

    def step(self,s,years):
        self.validate(s)
        if s['certificate']=='CLOSED_CONNECTED_BOUNDARIES':raise ValueError('PARTITION_REQUIRES_HANDOFF')
        if not np.isfinite(years) or years<=0:raise ValueError('Positive years')
        dt=years*YEAR;m=copy(self.m);m.state=deepcopy(self.m.state)
        m.state['nodal_displacement']=deepcopy(s['displacement']);basal=assemble(m)
        e=np.asarray(s['memory']);H=self.stiffness(s['edge_damage']);blocks=[]
        for f in range(self.nf):
            for a in range(3):blocks.append(dict(H=H[f]/3,L=self.L[f,a],R=np.kron(self.R[f],np.eye(6)),
                old=e[f,a],theta=np.zeros(12),dofs=self.ids[f]))
        r=maxwell_step(self.M,basal['D'],basal['F'],np.asarray(s['velocity']),dt,blocks)
        memory=r['memory'].reshape(self.nf,3,12);before_commit,_,stress=self.energy_force_stress(memory,s['edge_damage'])
        shear=[]
        for f,g,i,j in m.mesh.shared_edges:
            n=np.cross(m.mesh.vertices[i],m.mesh.vertices[j]);n/=np.linalg.norm(n)
            traction=.5*(stress[f]+stress[g])@n;shear.append(np.linalg.norm(traction-n*(traction@n)))
        damage=np.minimum(1.,np.asarray(s['edge_damage'])+years/100*np.maximum(np.array(shear)/1e6-1,0)**2)
        Hnew=self.stiffness(damage)
        release=float(np.einsum('fai,fij,faj->',memory,H-Hnew,memory)/6)
        if release < -1e-12*max(before_commit,1.):raise ValueError('NEGATIVE_RELEASE')
        release=max(0.,release)
        out=deepcopy(s);out.update(memory=memory.tolist(),velocity=r['velocity'].tolist(),
            displacement=(np.asarray(s['displacement'])+r['displacement_increment']).tolist(),
            edge_damage=damage.tolist(),elapsed_years=s['elapsed_years']+years,
            time_myr=s['source_time_myr']+(s['elapsed_years']+years)/1e6,
            reserve_j=s['reserve_j']+release,heat_buffer_j=s['heat_buffer_j']+r['ledger']['maxwell']+r['ledger']['drag'])
        _,reason=certify(m.mesh,damage>=1,Parameters());out['certificate']=reason
        for name in ('external_work','numerical_memory','numerical_kinetic','maxwell','drag'):
            out['cumulative'][name]=out['cumulative'].get(name,0.)+r['ledger'][name]
        E,_,_=self.energy_force_stress(memory,damage);K=float(.5*r['velocity']@(self.M@r['velocity']));c=out['cumulative']
        balance=E-out['initial_elastic_j']+K-out['initial_kinetic_j']+out['reserve_j']-out['initial_reserve_j']+out['heat_buffer_j']+c['numerical_memory']+c['numerical_kinetic']-c['external_work']
        den=max(out['initial_elastic_j'],abs(c['external_work']),E,1.)
        if abs(balance)/den>5e-8:raise ValueError('GLOBAL_ENERGY')
        self.validate(out)
        return out,dict(elapsed_years=out['elapsed_years'],global_energy_relative=abs(balance)/den,
            momentum_relative=r['ledger']['momentum_relative'],max_damage=float(damage.max()),saturated=int(sum(damage>=1)),
            max_shear_pa=max(shear),elastic_j=E,reserve_j=out['reserve_j'],heat_buffer_j=out['heat_buffer_j'],
            force_norm_n=float(np.linalg.norm(basal['F'])),basal_depth_min_m=float(basal['geometry']['depth'].min()),certificate=reason)

    def save(self,s,path):
        self.validate(s);payload=dict(source_envelope=self.envelope,state=s,scope='thermal inventory/old-D/front frozen; dissipation in heat buffer')
        with Path(path).open('xb') as stream:stream.write(encoded(dict(format=VERSION,sha256=digest(encoded(payload)),payload=payload)))

    @classmethod
    def load(cls,path):
        obj=json.loads(Path(path).read_bytes())
        if obj['format']!=VERSION or digest(encoded(obj['payload']))!=obj['sha256']:raise ValueError('SOURCE_ANGULAR_CHECKSUM')
        model=cls(obj['payload']['source_envelope']);s=obj['payload']['state'];model.validate(s);return model,s
