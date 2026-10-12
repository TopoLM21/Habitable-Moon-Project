"""Opt-in effective Genesis: constrained thermal moments and rotating domains.

This is a new model, not continuation of the resolved volume equations.
Material topology is fixed between cuts; small rotations have no contact/remesh.
"""
from copy import deepcopy
from dataclasses import asdict,replace
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .genesis import GenesisParameters,SECONDS_PER_MYR
from .genesis_tides import TidalParameters,TidalOrbitState,tidal_strain_cycle
from .genesis_thermal_ports import apply_ports
from .genesis_implicit_enthalpy import solve,invert
from .genesis_onset_support import NonlocalLoading
from .genesis_shell import principal_tensile
from .genesis_starter_topology import select_starter_cut,split_starter_band
from .plates import Plate,PlateSystem
from .mesh import SphereMesh
from .genesis_transfer_bridge import encoded,digest
from .simplified_genesis_phase import promote

VERSION='simplified-genesis-0.1'


def rotational_step(I,D,tau,omega,dt):
    """BE angular momentum with exact positive discrete energy accounting."""
    new=np.linalg.solve(I+dt*D,I@omega+dt*tau)
    d=new-omega;K0=.5*omega@I@omega;K1=.5*new@I@new
    work=dt*tau@new;drag=dt*new@D@new;numerical=.5*d@I@d
    residual=I@d+dt*(D@new-tau)
    return new,dict(kinetic_before=float(K0),kinetic_after=float(K1),external_work=float(work),drag=float(drag),numerical_rotation=float(numerical),rotation_balance=float(K1-K0-work+drag+numerical),momentum_relative=float(np.linalg.norm(residual)/max(np.linalg.norm(I@d),dt*np.linalg.norm(tau),1.)))


def constrained_force_reaction(momentum_map,force,drag_force,old_omega,new_omega,dt):
    """Center constraint closes linear momentum; reaction at origin does no work.

    The first version admits rotations only. This is an explicit support force,
    not an assertion that the unconstrained translational resultant vanishes.
    """
    rate=momentum_map@(new_omega-old_omega)/dt
    external=force-drag_force@new_omega
    return rate-external


class SimplifiedGenesis:
    def __init__(self,config,state):
        self.config=deepcopy(config);self.state=deepcopy(state)
        m=config['mesh'];self.mesh=SphereMesh(np.array(m['vertices']),np.array(m['faces']),np.array(m['centroids']),np.array(m['areas']),tuple(tuple(x) for x in m['neighbors']),tuple(tuple(x) for x in m['shared_edges']))
        self.nf=self.mesh.cell_count
        for name in ['mass','cp','latent','offset','G','P','weights','C','solid_nodes','face_inertia','face_drag','face_torque']:
            setattr(self,name,np.array(config[name]))
        self.solid_nodes=self.solid_nodes.astype(int);self.shell=config['shell'];self.faults=config['faults']
        self.smoother=NonlocalLoading(self.mesh,config['radius_km'],config['regularization_km'])
        self.validate()

    def validate(self):
        s=self.state;n=self.nf
        if s['format']!=VERSION:raise ValueError('SIMPLIFIED_VERSION')
        for name,shape in [('energy',(len(self.mass),)),('memory',(n,2,6)),('damage',(n,)),('candidate_age',(n,)),('consumed',(n,)),('owner',(n,))]:
            a=np.asarray(s[name]);
            if a.shape!=shape or not np.isfinite(a).all():raise ValueError('SIMPLIFIED_SHAPE '+name)
        if np.any(self.mass<=0) or np.any(np.asarray(s['energy'])<=0):raise ValueError('POSITIVE_INVENTORY')
        if np.any((np.asarray(s['damage'])<0)|(np.asarray(s['damage'])>1)):raise ValueError('DAMAGE_RANGE')
        owner=np.asarray(s['owner']);ids=np.unique(owner)
        if owner.dtype.kind not in 'iu' or not np.array_equal(ids,np.arange(len(ids))):raise ValueError('OWNER_IDS')
        for name,shape in [('omega',(len(ids),3)),('rotation',(len(ids),3,3))]:
            if np.asarray(s[name]).shape!=shape or not np.isfinite(s[name]).all():raise ValueError('DOMAIN_SHAPE')
        if s['passive_reserve_j']!=self.config['passive_reserve_j']:raise ValueError('PASSIVE_RESERVE_CHANGED')
        catalog=self.config.get('phase_cells',[]);cells={x['id']:x for x in catalog};seen=set()
        if len(cells)!=len(catalog) or len({x['node'] for x in catalog})!=len(catalog):raise ValueError('DUPLICATE_PHASE_SUPPORT')
        for c in catalog:
            if c['mass_kg']!=self.mass[c['node']] or c['node'] in self.solid_nodes or not 0<=c['face']<n:raise ValueError('PHASE_SUPPORT_MASS')
        for f in range(n):
            rows=sorted((x for x in catalog if x['face']==f),key=lambda x:x['q_lo'])
            if any(b['q_lo']<a['q_hi'] for a,b in zip(rows,rows[1:])):raise ValueError('OVERLAPPING_PHASE_SUPPORT')
        for x in s.get('born_material',[]):
            if x['cell_id'] in seen or x['cell_id'] not in cells:raise ValueError('MATERIAL_IDENTITY')
            c=cells[x['cell_id']]
            if any(x[k]!=c[k] for k in ['node','face','mass_kg','volume_m3','inertia']):raise ValueError('MATERIAL_INVENTORY')
            if np.asarray(x['memory']).shape!=(6,) or not np.isfinite(x['memory']).all():raise ValueError('BIRTH_MEMORY')
            if x['birth_time_myr']>s['time_myr']:raise ValueError('BIRTH_CLOCK')
            seen.add(x['cell_id'])
        if self.config['source_mode']=='genesis' and s['orbit']['time_myr']!=s['time_myr']:raise ValueError('CLOCK')

    def temperature(self,E):
        return invert(np.asarray(E)/self.mass,self.cp,self.latent,np.full(len(E),self.config['solidus']),np.full(len(E),self.config['liquidus']),self.offset)

    def scale(self,damage):
        return self.shell['residual_stiffness']+(1-self.shell['residual_stiffness'])*(1-np.asarray(damage))**2

    def phase_guard(self,T):
        coeff=np.linalg.solve(self.P,T[self.solid_nodes][...,None])[...,0];ends=np.einsum('fij,fj->fi',np.array(self.config['profile_limits']),coeff)
        if not np.isfinite(T).all() or np.min(T)<=0 or np.max(ends)>=self.config['solidus'] or np.min(ends)<=0:raise ValueError('PHASE_FRONT_REQUIRES_NEXT_MODEL')
        nonsolid=np.ones(len(T),bool);nonsolid[self.solid_nodes.ravel()]=False;nonsolid[-2:]=False
        supported={x['node'] for x in self.config.get('phase_cells',[])}
        if any(T[i]<=self.config['solidus'] and i not in supported for i in np.flatnonzero(nonsolid)):raise ValueError('UNMAPPED_SOLIDIFICATION_UNSUPPORTED')
        if any(T[x['node']]>self.config['solidus'] for x in self.state.get('born_material',[])):raise ValueError('LOAD_BEARING_REMELT_UNSUPPORTED')
        if self.config['source_mode']=='genesis' and T[-1]<647.096:raise ValueError('SURFACE_PHASE_REQUIRES_NEXT_MODEL')

    def constitutive(self,T,damage,old,oldT,dt):
        p=self.shell;nodes=T[self.solid_nodes]
        eta=np.clip(p['viscosity_reference_pa_s']*np.exp(np.clip(p['activation_energy_j_mol']/8.314462618*(1/nodes-1/p['viscosity_reference_temperature_k']),-60,60)),p['viscosity_min_pa_s'],p['viscosity_max_pa_s'])
        rates=p['young_modulus_pa']/eta
        R=np.linalg.solve(self.G,np.einsum('fni,fn,fnj->fij',self.P,self.weights*rates,self.P))
        dcoeff=np.linalg.solve(self.P,(nodes-oldT[self.solid_nodes])[...,None])[...,0];theta=dcoeff[:,:,None]*p['linear_expansion_per_k']*np.array([1,1,1,0,0,0.])
        new=np.linalg.solve(np.eye(2)[None,:,:]+dt*R,old-theta)
        C=self.C*(self.scale(damage)*~np.asarray(self.state['consumed'],bool))[:,None,None];strain=np.einsum('fni,fij->fnj',self.P,new);stress=np.einsum('fij,fnj->fni',C,strain)
        Q=dt*self.weights*rates*np.sum(strain*stress,axis=2)
        W=-self.weights*p['linear_expansion_per_k']*(nodes-oldT[self.solid_nodes])*stress[:,:,:3].sum(axis=2)
        H=np.einsum('fij,fkl->fikjl',self.G,C).reshape(self.nf,12,12)
        energy=lambda e:.5*np.einsum('fi,fij,fj->f',e.reshape(self.nf,12),H,e.reshape(self.nf,12))
        mean=np.einsum('fi,fij->fj',self.G[:,0,:],new)/self.G[:,0,0,None]
        sigma=np.einsum('fij,fj->fi',C,mean)[:,[0,1,3]]
        return new,sigma,Q,W,energy(new),energy(old),energy(new-old)

    def material_inertia(self):
        value=self.face_inertia.copy()
        for x in self.state.get('born_material',[]):value[x['face']]+=np.array(x['inertia'])
        return value

    def born_response(self,T,damage,oldT,dt):
        rows=[];heat=np.zeros(len(T));tot={k:np.zeros(self.nf) for k in ['U','Uold','Uinitial','N','Q','W','dU','volume']};stress=np.zeros((self.nf,3))
        p=self.shell
        for entry in self.state.get('born_material',[]):
            x=deepcopy(entry);f=x['face'];i=x['node'];V=x['volume_m3'];old=np.array(x['memory'])
            C=self.C[f]*self.scale(damage)[f]*(not self.state['consumed'][f]);C0=self.C[f]*self.scale(self.state['damage'])[f]*(not self.state['consumed'][f])
            eta=np.clip(p['viscosity_reference_pa_s']*np.exp(np.clip(p['activation_energy_j_mol']/8.314462618*(1/T[i]-1/p['viscosity_reference_temperature_k']),-60,60)),p['viscosity_min_pa_s'],p['viscosity_max_pa_s'])
            ratio=dt*p['young_modulus_pa']/eta;theta=p['linear_expansion_per_k']*(T[i]-oldT[i])*np.array([1.,1.,1.,0.,0.,0.]);new=(old-theta)/(1+ratio)
            U=.5*V*new@C@new;Uold=.5*V*old@C@old;Uinitial=.5*V*old@C0@old;N=.5*V*(new-old)@C@(new-old);Q=ratio*V*new@C@new;W=-V*new@C@theta;dU=Uold-Uinitial
            heat[i]+=Q-W-dU;stress[f]+=V*(C@new)[[0,1,3]]
            for key,value in dict(U=U,Uold=Uold,Uinitial=Uinitial,N=N,Q=Q,W=W,dU=dU,volume=V).items():tot[key][f]+=value
            x['memory']=new.tolist();rows.append(x)
        return rows,heat,tot,stress

    def advance(self,years):
        """Transactional physical-model step. Fixed parameters; no retry/dt sweep."""
        if not np.isfinite(years) or not 0<years<=self.config['max_step_years']:raise ValueError('STEP_RANGE')
        s=self.state;nexts=deepcopy(s);inertia=self.material_inertia();target=s['time_myr']+years/1e6;dt=(target-s['time_myr'])*SECONDS_PER_MYR
        E0=np.array(s['energy']);oldT=self.temperature(E0);old=np.array(s['memory']);oldD=np.array(s['damage']);owner=np.array(s['owner'])
        if self.config['source_mode']=='genesis':
            sourced,orbit,ports=apply_ports(dict(time_myr=s['time_myr'],mass_kg=self.mass.tolist(),energy_j=E0.tolist()),GenesisParameters(**self.config['thermal_parameters']),TidalParameters(**self.config['tidal_parameters']),TidalOrbitState(**s['orbit']),target)
            Esource=np.array(sourced['energy_j']);nexts['orbit']=asdict(orbit)
        else:
            Esource=E0+dt*np.array(self.config['synthetic_power_w']);ports=dict(net_thermal_input_j=float(np.sum(Esource-E0)))
        thermal=solve(self.mass,Esource/self.mass,self.cp,self.latent,np.full(len(E0),self.config['solidus']),np.full(len(E0),self.config['liquidus']),self.offset,self.config['edges'],dt)
        base=self.mass*thermal['specific_h'];T=self.temperature(base)
        # Angular dynamics on the saved material chart. Sphere fixes translation;
        # net force is supported by a zero-work radial/centering reaction.
        omega=[];rotation=[];rotledger={};dragface=np.zeros(self.nf);reactions=[]
        for k in range(len(s['omega'])):
            ids=owner==k;I=inertia[ids].sum(axis=0);D=self.face_drag[ids].sum(axis=0);tau=self.face_torque[ids].sum(axis=0)
            w,L=rotational_step(I,D,tau,np.array(s['omega'][k]),dt)
            pose=Rotation.from_rotvec(dt*w).as_matrix()@np.array(s['rotation'][k])
            if Rotation.from_matrix(pose).magnitude()>self.config['max_rotation_rad']:raise ValueError('FINITE_CONTACT_REMESH_LIMIT')
            if 'face_force' in self.config:
                def group(name):return np.array(self.config[name])[ids].sum(axis=0)
                reactions.append(constrained_force_reaction(group('face_linear_momentum_map'),group('face_force'),group('face_drag_force'),np.array(s['omega'][k]),w,dt).tolist())
            omega.append(w.tolist());rotation.append(pose.tolist());dragface[ids]=dt*np.einsum('i,fij,j->f',w,self.face_drag[ids],w)
            for key,value in L.items():rotledger[key]=max(rotledger.get(key,0.),value) if key=='momentum_relative' else rotledger.get(key,0.)+value
        heat=np.zeros_like(base)
        for f,ids in enumerate(self.solid_nodes):heat[ids]+=dragface[f]/2
        base=base+heat
        tide=np.zeros((1,self.nf,3))
        if self.config['source_mode']=='genesis':
            tp=replace(TidalParameters(**self.config['tidal_parameters']),eccentricity=orbit.eccentricity,semimajor_axis_km=orbit.semimajor_axis_km)
            nu=self.shell['poisson_ratio'];plane=np.array([[1.,nu,0.],[nu,1.,0.],[0.,0.,(1-nu)/2]])/(1-nu**2)
            tide=np.einsum('ij,pfj->pfi',plane,tidal_strain_cycle(self.mesh,tp))*self.shell['young_modulus_pa']*self.scale(oldD)[None,:,None]
        oldU=self.constitutive(oldT,oldD,old,oldT,0.)[4]
        for iteration in range(8):
            _,pred,_,_,_,_,_=self.constitutive(T,oldD,old,oldT,dt)
            _,_,bp,bs=self.born_response(T,oldD,oldT,dt)
            pred=(pred*self.G[:,0,0,None]+bs)/(self.G[:,0,0]+bp['volume'])[:,None]
            tensile=np.array([principal_tensile(pred+t) for t in tide]);load=np.mean(np.maximum(tensile/self.shell['tensile_strength_pa']-1,0)**2,axis=0)/self.shell['damage_timescale_myr']
            # Positive source-style regularization; hot-water branch has no wet weakening.
            load=self.smoother.apply(load,np.ones(self.nf,bool))
            hot=np.clip((T[self.solid_nodes].mean(axis=1)-700)/650,0,1)
            heal=1/self.shell['cold_healing_timescale_myr']+hot**4/self.shell['hot_healing_timescale_myr'];rate=load+heal;eq=load/rate
            damage=eq+(oldD-eq)*np.exp(-rate*years/1e6)
            new,sigma,Q,W,U,Uold,N=self.constitutive(T,damage,old,oldT,dt)
            # Healing is funded by thermal energy; damage releases heat. No
            # passive-reserve change and no numerical sink returned as heat.
            dU=Uold-oldU;E=base.copy()
            for f,ids in enumerate(self.solid_nodes):E[ids]+=Q[f]-W[f]-dU[f]/2
            born,extra_heat,bt,bs=self.born_response(T,damage,oldT,dt)
            E+=extra_heat
            sigma=(sigma*self.G[:,0,0,None]+bs)/(self.G[:,0,0]+bt['volume'])[:,None]
            U+=bt['U'];Uold+=bt['Uold'];N+=bt['N'];Q+=bt['Q'][:,None]/2;W+=bt['W'][:,None]/2;dU+=bt['dU']
            Tnew=self.temperature(E)
            if np.max(abs(Tnew-T))<1e-8:break
            T=Tnew
        else:raise ValueError('THERMAL_MECHANICAL_CORRECTOR_LIMIT')
        memory_residual=U-Uold-W.sum(axis=1)+Q.sum(axis=1)+N
        memory_scale=np.maximum(abs(U)+abs(Uold)+abs(W).sum(axis=1)+Q.sum(axis=1)+N,1.)
        if np.max(abs(memory_residual)/memory_scale)>1e-10 or np.any(Q<0):raise ValueError('MOMENT_ENERGY_IDENTITY')
        self.phase_guard(Tnew)
        age=np.where(damage>=self.faults['activation_damage'],np.array(s['candidate_age'])+years/1e6,0.)
        eligible=(damage>=self.faults['activation_damage'])&(age>=self.faults['activation_persistence_myr'])&~np.array(s['consumed'],bool)
        if not self.faults['enabled'] or not s['membrane_established']:eligible[:]=False
        plates=tuple(Plate(i,int(np.flatnonzero(owner==i)[0]),np.array([0.,0.,1.]),0.) for i in range(len(omega)))
        system=PlateSystem(owner,plates);args=self.config['selector'];cut=select_starter_cut(self.mesh,system,eligible,damage,**args)
        fracture=0.;split_momentum_error=0.
        if cut is not None:
            output,event=split_starter_band(self.mesh,system,cut,**args,time_myr=target)
            if output is None:raise ValueError('SELECTED_SPLIT_FAILED')
            newowner=output.cell_plate;newomega=[];newrotation=[]
            # Each child's own inertia inherits the parent's angular velocity;
            # sum(I_child omega_parent) = I_parent omega_parent, no kick.
            Lbefore=sum((inertia[owner==k].sum(axis=0)@omega[k] for k in range(len(omega))),np.zeros(3))
            for k in range(len(output.plates)):
                parents=np.unique(owner[newowner==k]);assert len(parents)==1;parent=int(parents[0])
                newomega.append(omega[parent]);newrotation.append(rotation[parent])
            Lafter=sum((inertia[newowner==k].sum(axis=0)@newomega[k] for k in range(len(newomega))),np.zeros(3))
            split_momentum_error=float(np.linalg.norm(Lafter-Lbefore)/max(np.linalg.norm(Lbefore),1.))
            # Effective fracture: release ONLY active elastic energy on cut cells.
            fracture=float(U[cut].sum());new[cut]=0.
            for f in cut:E[self.solid_nodes[f]]+=U[f]/2
            for row in born:
                if row['face'] in cut:row['memory']=[0.]*6
            U[cut]=0.;sigma[cut]=0.;consumed=np.array(s['consumed'],bool);consumed[cut]=True
            nexts['consumed']=consumed.tolist();nexts['owner']=newowner.tolist();omega=newomega;rotation=newrotation
            nexts['events'].append(dict(kind='effective_split',time_myr=target,cut=cut.tolist(),parents=list(event.parents),children=list(event.children),elastic_release_j=fracture,angular_momentum_relative=split_momentum_error))
        nexts.update(time_myr=target,born_material=born)
        omega,phase_report,new_inertia=promote(self.config,s,nexts,oldT,self.temperature(E),E,inertia,omega)
        omega=omega.tolist()
        new_volume=np.zeros(self.nf)
        old_ids={x['cell_id'] for x in s.get('born_material',[])}
        for row in nexts['born_material']:
            if row['cell_id'] not in old_ids:new_volume[row['face']]+=row['volume_m3']
        sigma*=((self.G[:,0,0]+bt['volume'])/(self.G[:,0,0]+bt['volume']+new_volume))[:,None]
        phase_report['load_bearing_mass_total_kg']=float(sum(self.config['face_mass_kg'])+sum(x['mass_kg'] for x in nexts['born_material']))
        phase_report['thermal_silicate_mass_kg']=float(sum(self.mass[:-2]))
        phase_report['not_load_bearing_mass_kg']=phase_report['thermal_silicate_mass_kg']-phase_report['load_bearing_mass_total_kg']
        if phase_report['not_load_bearing_mass_kg'] < -1e-12*phase_report['thermal_silicate_mass_kg']:raise ValueError('DOUBLE_COUNTED_MATERIAL_MASS')
        Tnew=self.temperature(E);self.phase_guard(Tnew)
        if any(Tnew[x['node']]>self.config['solidus'] for x in nexts['born_material']):raise ValueError('CAPTURE_REHEATS_NEW_SOLID')
        rotledger['kinetic_after']-=phase_report['capture_heat_j']
        ledger=dict(maxwell=float(Q.sum()),thermal_work=float(W.sum()),damage_energy_change=float(dU.sum()),numerical_memory=float(N.sum()),elastic_before=float(oldU.sum()+bt['Uinitial'].sum()),elastic_after=float(U.sum()),fracture_heat=fracture,source_energy=float(ports['net_thermal_input_j']),**rotledger)
        change=float(np.sum(E.astype(np.longdouble)-E0.astype(np.longdouble)))
        balance=change+ledger['elastic_after']-ledger['elastic_before']+ledger['kinetic_after']-ledger['kinetic_before']-ledger['source_energy']-ledger['external_work']+ledger['numerical_memory']+ledger['numerical_rotation']
        allowance=float(np.sum(32*(np.spacing(E)+np.spacing(E0))))+1e-8*max(abs(ledger['thermal_work']),ledger['maxwell'],abs(ledger['external_work']),1.)
        if abs(balance)>allowance:raise ValueError('SIMPLIFIED_WHOLE_ENERGY_BALANCE')
        ledger.update(phase=phase_report,whole_balance_j=balance,whole_allowance_j=allowance,moment_balance_relative=float(np.max(abs(memory_residual)/memory_scale)))
        if reactions:ledger['center_constraint_reaction_n']=reactions
        nexts.update(time_myr=target,energy=E.tolist(),memory=new.tolist(),damage=damage.tolist(),candidate_age=age.tolist(),omega=omega,rotation=rotation,stress_pa=sigma.tolist(),last_ledger=ledger,steps=s['steps']+1)
        self.state=nexts
        try:self.validate()
        except Exception:
            self.state=s
            raise
        return deepcopy(nexts)

    def save(self,path):
        body=dict(config=self.config,state=self.state);blob=encoded(body)
        with Path(path).open('xb') as f:f.write(encoded(dict(format=VERSION,sha256=digest(blob),payload=body)))

    @classmethod
    def load(cls,path):
        obj=json.loads(Path(path).read_bytes());body=obj['payload']
        if obj['format']!=VERSION or digest(encoded(body))!=obj['sha256']:raise ValueError('SIMPLIFIED_RESTART_CHECKSUM')
        return cls(body['config'],body['state'])

    def mature_compatibility(self):
        return dict(direct_native_mature=False,reason='moment memory, passive reserve and nonrigid source history need an explicit export; no silent default conversion',domains=len(self.state['omega']))
