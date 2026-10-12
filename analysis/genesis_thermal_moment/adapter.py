"""Two positive thermal quadrature nodes per live solid group, +80 scalars.

Mean enthalpy and its first material-depth moment are the active variables in
nodal form. All fluxes use source Fourier k and actual distances. No clipping,
new strength, secular heating of the unresolved reserve, or phase conversion.
"""
import numpy as np
from analysis.genesis_aggregate_coupled.adapter import Adapter
from analysis.genesis_volume_continuation_validation import independent_audit as q
from analysis.genesis_volume_fault_validation.independent_audit import physical_face_geometry


def moment_nodes(G):
    G=np.asarray(G,float);volume=G[0,0];mean=G[0,1]/volume;variance=G[1,1]/volume-mean*mean
    if not np.isfinite(G).all() or volume<=0 or variance<=0:raise ValueError('POSITIVE_THERMAL_MOMENT_REQUIRED')
    # Top (positive reference normal) first, deeper node second.
    nodes=mean+np.array([1.,-1.])*np.sqrt(variance)
    P=np.column_stack((np.ones(2),nodes));weights=np.full(2,volume/2)
    return nodes,P,weights


def split_enthalpy(total,first,mass,nodes):
    """Two positive masses retain total H and integral(phi H) exactly to roundoff."""
    mu=float(np.mean(nodes));sigma=float((nodes[0]-nodes[1])/2)
    if mass<=0 or sigma<=0:raise ValueError('THERMAL_MOMENT_BOUNDS')
    upper=.5*(total+(first-mu*total)/sigma);E=np.array([upper,total-upper])
    if not np.isfinite(E).all() or np.any(E<=0):raise ValueError('NONPOSITIVE_MOMENT_ENTHALPY')
    return np.full(2,mass/2),E


def positive_chain(ids,positions,total_depth,kA,water,deep):
    pos=np.asarray(positions,float);lengths=np.diff(np.r_[0.,pos,total_depth])
    if len(ids)!=len(pos) or len(set(ids))!=len(ids) or np.any(lengths<=0) or not np.isfinite(lengths).all() or kA<=0:raise ValueError('POSITIVE_THERMAL_DISTANCES_REQUIRED')
    chain=[water,*ids,deep]
    return [(i,j,float(kA/d)) for i,j,d in zip(chain,chain[1:],lengths)]


class MomentAdapter(Adapter):
    def __init__(self,packet,source,cuts,full_inputs,contact_parameters):
        self._moment_ready=False
        super().__init__(packet,source,cuts,full_inputs,contact_parameters,thermal_reduction='series_resistance')
        old_n=self.n;old_m=self.m.copy();old_E=self.E0.copy();self.solid_second=np.arange(old_n-2,old_n-2+80)
        self.n=old_n+80;self.m=np.r_[old_m[:-2],np.zeros(80),old_m[-2:]];self.E0=np.r_[old_E[:-2],np.zeros(80),old_E[-2:]]
        self.map[-2:]=[self.n-2,self.n-1]
        self.cp=np.r_[self.cp[:-2],np.full(80,self.gp.silicate_heat_capacity_j_kg_k),self.cp[-2:]]
        self.latent=np.r_[self.latent[:-2],np.full(80,self.gp.silicate_latent_heat_j_kg),self.latent[-2:]]
        self.off=np.r_[self.off[:-2],np.zeros(80),self.off[-2:]]
        self.node_phi=[];self.P=[];self.weights=[];self.node_q=[];self.phi_bounds=[];self.cut_phi={};self.initial_moment=[]
        raw_E=np.asarray(packet['thermal']['energy_j']);byid={c['material_id']:i for i,c in enumerate(cuts) if c['material_id'] is not None}
        for f,b in enumerate(self.blocks):
            nodes,P,w=moment_nodes(b['G']);self.node_phi.append(nodes);self.P.append(P);self.weights.append(w)
            height=float(self.X[self.faces[f,0]]@self.frames[f,:,2]);tri=self.X[self.faces[f]]
            coefficient=6*self.g['column_mass_kg'][f]/(self.g['density_kg_m3']*np.linalg.det(tri.T))
            z=b['center']+b['scale']*nodes
            node_q=-np.expm1(3*np.log1p(z/height))/coefficient;self.node_q.append(node_q)
            first=0.;lo=1.;hi=0.
            for c in self.g['cohorts']:
                if c['face_index']!=f:continue
                rad,wr=q.radial_rule(c['q_lo'],c['live_q_hi']);lam,_=q.conical_scales(tri,self.g['column_mass_kg'][f],self.g['density_kg_m3'],rad)
                phi=(height*(lam-1)-b['center'])/b['scale'];mean=float(wr@phi);i=byid[c['id']]
                self.cut_phi[i]=(f,mean);first+=raw_E[i]*mean;lo=min(lo,c['q_lo']);hi=max(hi,c['live_q_hi'])
            if lo!=0 or not lo<node_q[0]<node_q[1]<hi:raise ValueError('THERMAL_NODES_OUTSIDE_LIVE_SOLID')
            limits,_=q.conical_scales(tri,self.g['column_mass_kg'][f],self.g['density_kg_m3'],np.array([lo,hi]))
            self.phi_bounds.append((height*(limits-1)-b['center'])/b['scale'])
            ids=[self.solid_group[f],self.solid_second[f]];mm,ee=split_enthalpy(old_E[ids[0]],first,old_m[ids[0]],nodes)
            self.m[ids]=mm;self.E0[ids]=ee;self.initial_moment.append(first)
        self.thermal_reduction='thermal_moment';self._moment_ready=True;self.edges=self.thermal_edges(packet)
        self.phase_guard(self.temperature(self.E0),self.temperature(self.E0))

    def temperature_coefficients(self,T,f):
        return np.linalg.solve(self.P[f],T[[self.solid_group[f],self.solid_second[f]]])

    def solid_mean_temperature(self,T):
        return .5*(T[self.solid_group]+T[self.solid_second])

    def expand_temperature(self,T):
        out=np.asarray(T)[self.map].copy()
        for i,(f,phi) in self.cut_phi.items():out[i]=np.array([1.,phi])@self.temperature_coefficients(T,f)
        return out

    def thermal_edges(self,packet):
        if not self._moment_ready:return super().thermal_edges(packet)
        X=self.X+self.u0.reshape(-1,6)[:,:3];_,areas=physical_face_geometry(X,self.faces);edges=[]
        for f,area in enumerate(areas*1e6):
            old_ids=[i for i,c in enumerate(self.cuts) if c['face']==f];groups=[]
            for i in old_ids:
                if not groups or groups[-1]!=self.map[i]:groups.append(int(self.map[i]))
            if groups[0]!=self.solid_group[f] or len(set(groups))!=len(groups):raise ValueError('CONTIGUOUS_SOLID_PREFIX_REQUIRED')
            # Same mass-depth convention as the original cut-cell Fourier graph.
            nodes=[int(self.solid_group[f]),int(self.solid_second[f])]
            positions=list(self.node_q[f]*self.g['column_mass_kg'][f]/(self.g['density_kg_m3']*area))
            depth=(self.m[nodes[0]]+self.m[nodes[1]])/(self.g['density_kg_m3']*area)
            for group in groups[1:]:
                dz=self.m[group]/(self.g['density_kg_m3']*area);nodes.append(group);positions.append(depth+dz/2);depth+=dz
            edges.extend(positive_chain(nodes,positions,depth,self.shell['conductivity_w_m_k']*area,self.n-1,self.n-2))
        return edges

    def phase_guard(self,T,T0):
        if T[-1]<647.096 or np.any(T<=0) or not np.isfinite(T).all():raise ValueError('UNSUPPORTED_THERMAL_PHASE')
        for f,bounds in enumerate(self.phi_bounds):
            coeff=self.temperature_coefficients(T,f);ends=np.column_stack((np.ones(2),bounds))@coeff
            if np.min(ends)<=0 or np.max(ends)>=self.gp.solidus_k:raise ValueError('RECONSTRUCTED_SOLID_PHASE_BOUNDARY: face '+str(f))
        solid=set(self.solid_group)|set(self.solid_second)
        if any(i not in solid and T0[i]>self.gp.solidus_k and T[i]<=self.gp.solidus_k for i in range(self.n-2)):raise ValueError('UNSUPPORTED_NEW_SOLID')

    def rate(self,T):
        p=self.shell
        eta=np.clip(p['viscosity_reference_pa_s']*np.exp(np.clip(p['activation_energy_j_mol']/8.314462618*(1/T-1/p['viscosity_reference_temperature_k']),-60,60)),p['viscosity_min_pa_s'],p['viscosity_max_pa_s'])
        return p['young_modulus_pa']/eta

    def mechanical_blocks(self,T,damage):
        blocks=[];unit=np.array([1.,1.,1.,0.,0.,0.])
        for f,b in enumerate(self.blocks):
            temps=T[[self.solid_group[f],self.solid_second[f]]];P=self.P[f];w=self.weights[f]
            Rsmall=np.linalg.solve(b['G'],P.T@((w*self.rate(temps))[:,None]*P))
            theta=(self.temperature_coefficients(T,f)-b['old_temperature_coeff'])[:,None]*self.shell['linear_expansion_per_k']*unit
            blocks.append(dict(H=np.kron(b['G'],self.C(f,damage)),L=b['L'],R=np.kron(Rsmall,np.eye(6)),old=self.old[f],theta=theta.ravel(),dofs=b['dofs']))
        return blocks

    def node_work(self,T,damage,memory,dt):
        Q=[];W=[];indices=[];unit=np.array([1.,1.,1.,0.,0.,0.])
        for f,b in enumerate(self.blocks):
            ids=[self.solid_group[f],self.solid_second[f]];temps=T[ids];P=self.P[f];w=self.weights[f]
            e=P@memory[f].reshape(2,6);C=self.C(f,damage);stress=e@C
            heat=dt*w*self.rate(temps)*np.einsum('ni,ni->n',stress,e)
            delta=temps-P@b['old_temperature_coeff'];work=-w*self.shell['linear_expansion_per_k']*delta*(stress@unit)
            if np.any(heat<0):raise ValueError('NEGATIVE_MAXWELL_HEAT')
            Q.extend(heat);W.extend(work);indices.extend(ids)
        return np.array(indices),np.array(Q),np.array(W)

    def mechanical(self,T,damage,dt):
        r=super().mechanical(T,damage,dt);ids,Q,W=self.node_work(T,damage,r['memory'],dt)
        for val,name in [(Q.sum(),'maxwell'),(W.sum(),'thermal_work')]:
            if abs(val-r['ledger'][name])>1e3+3e-8*abs(val):raise ValueError('MOMENT_WORK_EXCHANGE_MISMATCH')
        r['moment_exchanges']=(ids,Q,W)
        return r

    def heat_feedback(self,base,result):
        ids,Q,W=result['moment_exchanges'];E=base.copy();np.add.at(E,ids,Q-W)
        return E,Q,W
