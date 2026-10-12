"""Opt-in fixed-catalogue aggregate adapter for the existing source/conduction/BE split.

No new damage law. Constant residual energy is a declared unresolved reserve,
not an evolved Maxwell field. Geometry/phase events stop this bounded adapter.
"""
from dataclasses import replace,asdict
from types import SimpleNamespace
import numpy as np
from analysis.genesis_volume_continuation_validation import independent_audit as q
from analysis.genesis_volume_fault_validation.independent_audit import physical_face_geometry
from analysis.genesis_connected_step_validation.run import gate
from analysis.genesis_coupled_split_control.mechanical import healing_guard
from tectonics.genesis_volume_fault_evolution import volume_fault_history_from_dict,evaluate_volume_fault_evolution,volume_fault_history_to_dict
from tectonics.genesis_implicit_enthalpy import solve,invert
from tectonics.genesis_thermal_ports import apply_ports
from tectonics.genesis_volume_basal import local_consistent_mass_matrix
from tectonics.genesis import GenesisParameters,SECONDS_PER_MYR
from tectonics.genesis_tides import TidalParameters,TidalOrbitState
from tectonics.genesis_shell import principal_tensile
from .core import step


def series_column_edges(groups,mass,density,area,conductivity,water,deep):
    """Existing Fourier law between aggregate cell mass-centres.

    Constant source density/k and face area: series resistances add exactly.
    Boundary nodes retain the original zero half-width bath convention.
    """
    ids=list(groups)
    if not ids or len(set(ids))!=len(ids) or min(density,area,conductivity)<=0:
        raise ValueError('CONTIGUOUS_UNIQUE_COLUMN_REQUIRED')
    dz=np.asarray(mass)[ids]/(density*area)
    if not np.isfinite(dz).all() or np.any(dz<=0):raise ValueError('POSITIVE_COLUMN_MASS_REQUIRED')
    ka=conductivity*area
    return [(water,ids[0],2*ka/dz[0]),*[(i,j,ka/(.5*(dz[k]+dz[k+1]))) for k,(i,j) in enumerate(zip(ids,ids[1:]))],(ids[-1],deep,2*ka/dz[-1])]


class Adapter:
    def __init__(self,packet,source,cuts,full_inputs,contact_parameters,*,thermal_reduction='projected_graph'):
        self.source=source;self.packet=packet;self.cuts=cuts;self.contact_parameters=contact_parameters
        if thermal_reduction not in ('projected_graph','series_resistance'):raise ValueError('EXPLICIT_THERMAL_REDUCTION')
        self.thermal_reduction=thermal_reduction
        p=source['source']['parameters'];self.parameters=p;self.shell=p['shell'];self.gp=GenesisParameters(**p['thermal']);self.tides=TidalParameters(**p['tides'])
        s=packet['state'];g=s['geometry'];self.g=g;self.X=np.array(g['vertices_m']);self.faces=np.array(g['faces']);nf=len(self.faces)
        assert nf==80 and len(g['cohorts'])==800
        self.frames,self.areas=physical_face_geometry(self.X,self.faces)
        self.history=volume_fault_history_from_dict(s['fault_history'],source)
        self.fault_parameters=SimpleNamespace(**p['faults']);gate(self.history,self.fault_parameters)
        self.v0=np.array(s['velocity_m_s']);self.u0=np.array(s['displacement_m']);self.time=packet['thermal']['time_myr'];self.orbit=TidalOrbitState(**packet['orbit_after'])
        # Strict fixed-input binding to the saved full-mode operator data.
        assert full_inputs['before_displacement_m']==s['displacement_m'] and full_inputs['before_velocity_m_s']==s['velocity_m_s']
        self.M=np.zeros((len(self.v0),len(self.v0)))
        for c in g['cohorts']:
            idx=(self.faces[c['face_index'],:,None]*6+np.arange(6)).ravel()
            self.M[np.ix_(idx,idx)]+=local_consistent_mass_matrix(c['mass_kg'],c['outer_scale'],c['inner_scale'])
        traction=np.asarray(full_inputs['basal_traction_at_vertices_pa'])[self.faces]
        self.D,self.F=q.basal_drag_and_force(self.X,self.faces,g['column_mass_kg'],g['density_kg_m3'],g['cohorts'],full_inputs['basal_drag_pa_s_m'],traction)
        labels=[(c['face'],c['layer'],c['material_id'] is None) for c in cuts]
        unique=sorted(set(labels));lookup={x:i for i,x in enumerate(unique)}
        self.map=np.array([lookup[x] for x in labels]+[len(unique),len(unique)+1]);self.n=len(unique)+2
        self.m=np.bincount(self.map,weights=packet['thermal']['mass_kg'],minlength=self.n)
        self.E0=np.bincount(self.map,weights=packet['thermal']['energy_j'],minlength=self.n)
        self.cp=np.full(self.n,self.gp.silicate_heat_capacity_j_kg_k);self.latent=np.full(self.n,self.gp.silicate_latent_heat_j_kg);self.off=np.zeros(self.n)
        self.cp[-1]=self.gp.water_heat_capacity_j_kg_k;self.latent[-1]=0.;self.off[-1]=self.gp.water_latent_heat_j_kg
        self.solid_group=np.array([lookup[(f,0,False)] for f in range(nf)])
        self.old_material_T={c['id']:c['temperature_k'] for c in s['mechanics']['cohorts']}
        self.old_cohort_groups={c['material_id']:self.map[i] for i,c in enumerate(cuts) if c['material_id'] is not None}
        assert all(self.old_cohort_groups[c['id']]==self.solid_group[c['face_index']] for c in g['cohorts'])
        self.edges=self.thermal_edges(packet)
        self.blocks,self.old,self.reserve=self.project(s)
        self.active_initial=sum(.5*e@np.kron(b['G'],self.C(f,self.history.damage))@e for f,(b,e) in enumerate(zip(self.blocks,self.old)))

    def C(self,f,damage):
        p=self.shell
        return q.isotropic_matrix(p['young_modulus_pa'],p['poisson_ratio'],damage[f],p['residual_stiffness'])

    def thermal_edges(self,packet):
        m=np.array(packet['thermal']['mass_kg']);X=self.X+self.u0.reshape(-1,6)[:,:3];_,areas=physical_face_geometry(X,self.faces)
        edges={}
        for f,area in enumerate(areas*1e6):
            ids=[i for i,c in enumerate(self.cuts) if c['face']==f];dz=m[ids]/(self.shell['density_kg_m3']*area);ka=self.shell['conductivity_w_m_k']*area
            if self.thermal_reduction=='series_resistance':
                groups=[]
                for i in ids:
                    if not groups or groups[-1]!=self.map[i]:groups.append(int(self.map[i]))
                for i,j,G in series_column_edges(groups,self.m,self.shell['density_kg_m3'],area,self.shell['conductivity_w_m_k'],self.n-1,self.n-2):
                    key=tuple(sorted((i,j)));edges[key]=edges.get(key,0.)+G
                continue
            fine=[(len(m)-1,ids[0],2*ka/dz[0]),(ids[-1],len(m)-2,2*ka/dz[-1])]
            fine.extend((i,j,ka/(.5*(dz[t]+dz[t+1]))) for t,(i,j) in enumerate(zip(ids,ids[1:])))
            for i,j,G in fine:
                a,b=sorted((int(self.map[i]),int(self.map[j])))
                if a!=b:edges[a,b]=edges.get((a,b),0.)+G
        return [(i,j,G) for (i,j),G in edges.items()]

    def project(self,s):
        N,wa=q.angular_rule();interp=N@np.linalg.inv(q.SOURCE_ANGULAR_POINTS);rows={c['id']:c for c in s['mechanics']['cohorts']}
        blocks=[];old=[];reserve=0.
        for f in range(80):
            B0,B1=q.independent_strain_operators(self.X[self.faces[f]],self.frames[f],N)
            height=float(self.X[self.faces[f,0]]@self.frames[f,:,2]);parts=[]
            for c in self.g['cohorts']:
                if c['face_index']!=f:continue
                radial,wr=q.radial_rule(c['q_lo'],c['live_q_hi']);lam,_=q.conical_scales(self.X[self.faces[f]],self.g['column_mass_kg'][f],self.g['density_kg_m3'],radial)
                z=np.broadcast_to(height*(lam-1),(len(wa),len(wr))).ravel();w=(c['mass_kg']/self.g['density_kg_m3']*wa[:,None]*wr[None,:]).ravel()
                r=rows[c['id']];e=(interp@np.array(r['strain0']))[:,None,:]+(interp@np.array(r['strain1']))[:,None,:]/lam[None,:,None]
                B=B0[:,None,:,:]+B1[:,None,:,:]/lam[None,:,None,None]
                parts.append((c['id'],w,z,e.reshape(-1,6),B.reshape(-1,6,18)))
            volume=sum(x[1].sum() for x in parts);center=sum(x[1]@x[2] for x in parts)/volume;scale=max(np.max(np.abs(x[2]-center)) for x in parts)
            G=np.zeros((2,2));rhs=np.zeros((2,6));LB=np.zeros((2,6,18));energy=0.;temp_rhs=np.zeros(2)
            for cid,w,z,e,B in parts:
                phi=np.column_stack((np.ones(len(w)),(z-center)/scale));G+=phi.T@(w[:,None]*phi);rhs+=phi.T@(w[:,None]*e)
                LB+=np.einsum('n,na,nbi->abi',w,phi,B)
                temp_rhs+=phi.T@(w*self.old_material_T[cid]);C=self.C(f,self.history.damage);energy+=float(.5*np.einsum('n,ni,ij,nj->',w,e,C,e))
            e=np.linalg.solve(G,rhs).ravel();L=np.linalg.solve(G,LB.reshape(2,-1)).reshape(12,18)
            reserve+=energy-.5*e@np.kron(G,C)@e
            blocks.append(dict(G=G,L=L,old_temperature_coeff=np.linalg.solve(G,temp_rhs),dofs=(self.faces[f,:,None]*6+np.arange(6)).ravel(),center=center,scale=scale));old.append(e)
        return blocks,np.array(old),float(reserve)

    def temperature(self,E):
        return invert(E/self.m,self.cp,self.latent,np.full(self.n,self.gp.solidus_k),np.full(self.n,self.gp.liquidus_k),self.off)

    def phase_guard(self,T,T0):
        if T[-1]<647.096 or np.any(T[self.solid_group]>=self.gp.solidus_k):raise ValueError('UNSUPPORTED_PHASE_TRANSITION')
        solid=set(self.solid_group)
        if any(i not in solid and T0[i]>self.gp.solidus_k and T[i]<=self.gp.solidus_k for i in range(self.n-2)):raise ValueError('UNSUPPORTED_NEW_SOLID')

    def mechanical(self,T,damage,dt):
        blocks=[]
        for f,b in enumerate(self.blocks):
            temp=T[self.solid_group[f]];p=self.shell
            eta=np.clip(p['viscosity_reference_pa_s']*np.exp(np.clip(p['activation_energy_j_mol']/8.314462618*(1/temp-1/p['viscosity_reference_temperature_k']),-60,60)),p['viscosity_min_pa_s'],p['viscosity_max_pa_s'])
            theta=(np.array([temp,0.])-b['old_temperature_coeff'])[:,None]*p['linear_expansion_per_k']*np.array([1.,1.,1.,0.,0.,0.])[None,:]
            blocks.append(dict(H=np.kron(b['G'],self.C(f,damage)),L=b['L'],R=np.eye(12)*p['young_modulus_pa']/eta,old=self.old[f],theta=theta.ravel(),dofs=b['dofs']))
        result=step(self.M,self.D,self.F,self.v0,dt,blocks)
        if result['ledger']['momentum_relative']>self.contact_parameters['equilibrium_tolerance']:raise ValueError('AGGREGATE_MOMENTUM_GATE')
        if result['ledger']['balance_relative']>5e-10:raise ValueError('AGGREGATE_ENERGY_GATE')
        X=self.X+(self.u0+result['displacement_increment']).reshape(-1,6)[:,:3];frames,_=physical_face_geometry(X,self.faces)
        stress=[]
        for f,(e,b) in enumerate(zip(result['memory'],self.blocks)):
            mean=(b['G'][0]@e.reshape(2,6))/b['G'][0,0];v=self.C(f,damage)@mean
            sigma=np.array([[v[0],v[3],v[4]],[v[3],v[1],v[5]],[v[4],v[5],v[2]]]);world=self.frames[f]@sigma@self.frames[f].T;local=frames[f].T@world@frames[f]
            stress.append([local[0,0],local[1,1],local[0,1]])
        result.update(stress=np.array(stress),vertices=X)
        return result

    def advance(self,years=25.):
        if years!=25.:raise ValueError('ONE_25_YEAR_TRIAL_ONLY')
        if self.time!=self.packet['thermal']['time_myr']:
            raise ValueError('FROZEN_RESERVE_NOT_ADMITTED_FOR_FURTHER_EVOLUTION')
        target=self.time+years/1e6;dt=(target-self.time)*SECONDS_PER_MYR
        initial=dict(time_myr=self.time,mass_kg=self.m.tolist(),energy_j=self.E0.tolist())
        sourced,orbit,ports=apply_ports(initial,self.gp,self.tides,self.orbit,target)
        Esource=np.array(sourced['energy_j']);T0=self.temperature(Esource)
        thermal=solve(self.m,Esource/self.m,self.cp,self.latent,np.full(self.n,self.gp.solidus_k),np.full(self.n,self.gp.liquidus_k),self.off,self.edges,dt)
        base=self.m*thermal['specific_h'] if 'specific_h' in thermal else self.m*(self.cp*thermal['temperature_k']+self.latent*np.clip((thermal['temperature_k']-self.gp.solidus_k)/(self.gp.liquidus_k-self.gp.solidus_k),0,1)+self.off)
        T=self.temperature(base);self.phase_guard(T,T0);previous=None;correctors=[]
        for iteration in range(6):
            pred=self.mechanical(T,self.history.damage,dt)
            faults,evolution=evaluate_volume_fault_evolution(self.history,self.source,after_time_myr=target,physical_vertices_m=pred['vertices'],face_stress_pa=pred['stress'],face_mean_temperature_k=T[self.solid_group],retained_fraction=np.ones(80),active_material=np.ones(80,dtype=bool),surface_temperature_k=float(T[-1]),ocean_fraction=0.,orbit_before=self.orbit,orbit_after=orbit,weak_plane_slip_available=True)
            gate(faults,self.fault_parameters);healing_guard(self.history.damage,faults.damage)
            final=self.mechanical(T,faults.damage,dt);Q=np.array([r['maxwell'] for r in final['rows']]);W=np.array([r['thermal_work'] for r in final['rows']])
            E=base.copy();np.add.at(E,self.solid_group,Q-W);nextT=self.temperature(E);self.phase_guard(nextT,T0)
            err=float(np.max(np.abs(nextT-T)));exerr=None if previous is None else float(max(np.max(np.abs(Q-previous[0])/(1e3+3e-8*np.abs(Q))),np.max(np.abs(W-previous[1])/(1e3+3e-8*np.abs(W)))))
            correctors.append(dict(index=iteration+1,temperature_error_k=err,exchange_error_over_budget=exerr))
            if err<=1e-8 and exerr is not None and exerr<=1:break
            previous=(Q,W);T=nextT
        else:raise ValueError('SIX_CORRECTOR_BUDGET')
        faults=replace(faults,peak_tensile_pa=np.maximum(faults.peak_tensile_pa,principal_tensile(final['stress'])))
        damage_change=final['ledger']['elastic_before']-self.active_initial
        if damage_change>0:raise ValueError('UNFUNDED_DAMAGE_ENERGY')
        return dict(format='genesis-active-aggregate-trial-1',time_myr=target,mass_kg=self.m.tolist(),energy_j=E.tolist(),temperature_k=nextT.tolist(),
            memory=final['memory'].tolist(),velocity_m_s=final['velocity'].tolist(),displacement_m=(self.u0+final['displacement_increment']).tolist(),
            fault_history=volume_fault_history_to_dict(faults),stress_pa=final['stress'].tolist(),internal_force_n=final['internal_force'].tolist(),
            ledger=final['ledger'],damage_energy_change_j=float(damage_change),unresolved_reserve_j=self.reserve,
            reserve_policy='frozen for this one trial; not heat, not active memory, no long-term guarantee',
            correctors=correctors,orbit=asdict(orbit),ports=ports,source_parameters=self.parameters,
            thermal_edges=len(self.edges),thermal_reduction=self.thermal_reduction,new_events=False,canonical_advanced=False)

    def restore(self,saved):
        """Restore active aggregate state on the same pinned reference catalogue.

        Rebuilding an Adapter authenticates the original source/operator data.
        This does not promote the trial to accepted or canonical history.
        """
        if saved['format']!='genesis-active-aggregate-trial-1' or saved['source_parameters']!=self.parameters:
            raise ValueError('AGGREGATE_RESTART_SOURCE')
        if saved.get('thermal_reduction','projected_graph')!=self.thermal_reduction:
            raise ValueError('AGGREGATE_RESTART_THERMAL_REDUCTION')
        if not np.array_equal(saved['mass_kg'],self.m) or saved['unresolved_reserve_j']!=self.reserve:
            raise ValueError('AGGREGATE_RESTART_INVENTORY')
        memory=np.asarray(saved['memory'],float);velocity=np.asarray(saved['velocity_m_s'],float)
        displacement=np.asarray(saved['displacement_m'],float);energy=np.asarray(saved['energy_j'],float)
        if memory.shape!=self.old.shape or velocity.shape!=self.v0.shape or displacement.shape!=self.u0.shape or energy.shape!=self.E0.shape:
            raise ValueError('AGGREGATE_RESTART_SHAPE')
        if not all(np.isfinite(x).all() for x in (memory,velocity,displacement,energy)) or np.any(energy<0):
            raise ValueError('AGGREGATE_RESTART_FINITE')
        history=volume_fault_history_from_dict(saved['fault_history'],self.source)
        gate(history,self.fault_parameters)
        if saved['time_myr']!=history.time_myr or saved['time_myr']!=saved['orbit']['time_myr']:
            raise ValueError('AGGREGATE_RESTART_CLOCK')
        temperature=self.temperature(energy)
        if not np.array_equal(temperature,np.asarray(saved['temperature_k'])):
            raise ValueError('AGGREGATE_RESTART_ENTHALPY')
        self.old=memory.copy();self.v0=velocity.copy();self.u0=displacement.copy();self.E0=energy.copy()
        self.time=saved['time_myr'];self.history=history;self.orbit=TidalOrbitState(**saved['orbit'])
        for f,b in enumerate(self.blocks): b['old_temperature_coeff']=np.array([temperature[self.solid_group[f]],0.])
        self.active_initial=sum(.5*e@np.kron(b['G'],self.C(f,history.damage))@e for f,(b,e) in enumerate(zip(self.blocks,self.old)))
        self.edges=self.thermal_edges(self.packet)
        return self
