"""Opt-in angular memory with the EXISTING Genesis thermal/phase integrator.

No new heat law: three angular contributions feed the same thermal supports.
Newly admitted material starts stress-free at all angular points. Old newborn
history without angular data is never silently lifted.
"""
from copy import deepcopy
from pathlib import Path
import json
import numpy as np
from .simplified_genesis import SimplifiedGenesis
from . import simplified_genesis_nodal as native
from .simplified_genesis_basal import budget
from .experimental_edge_weakening import scale,face_damage,MODE
from .genesis_transfer_bridge import encoded,digest
from analysis.genesis_geological_handoff.source_angular import SourceAngular
from analysis.genesis_geological_handoff.integrated_vector_operator import ANGULAR_N
from analysis.genesis_volume_continuation_validation import independent_audit as q

VERSION='coupled-source-angular-moments/v1'


def validate(m):
    if m.config.get('experimental_angular_memory')!=VERSION:raise ValueError('ANGULAR_COUPLED_VERSION')
    e=np.asarray(m.state['angular_memory'])
    if e.shape!=(m.nf,3,2,6) or not np.isfinite(e).all():raise ValueError('ANGULAR_COUPLED_MEMORY')
    if not np.allclose(e.mean(axis=1),m.state['memory'],rtol=1e-12,atol=1e-18):raise ValueError('ANGULAR_MEAN_MISMATCH')
    born=m.state['angular_born_memory']
    if set(born)!={x['cell_id'] for x in m.state.get('born_material',[])}:raise ValueError('ANGULAR_BORN_IDENTITY')
    for x in m.state.get('born_material',[]):
        b=np.asarray(born[x['cell_id']])
        if b.shape!=(3,6) or not np.isfinite(b).all() or not np.allclose(b.mean(axis=0),x['memory'],rtol=1e-12,atol=1e-18):raise ValueError('ANGULAR_BORN_MEMORY')


def born_operator(m,cell):
    if not hasattr(m,'_angular_birth'):m._angular_birth={}
    if cell['id'] not in m._angular_birth:
        f=cell['face'];o=native.operators(m);tri=o['X'][m.mesh.faces[f]]
        B0,B1=q.independent_strain_operators(tri,o['frames'][f],ANGULAR_N)
        radial,wr=q.radial_rule(cell['q_lo'],cell['q_hi'])
        lam,_=q.conical_scales(tri,m.config['nodal']['column_mass'][f],m.config['nodal']['density'],radial)
        raw=B0+B1*float(wr@(1/lam))
        original,_=native.birth_operator(m,cell)
        if np.linalg.norm(raw.mean(axis=0)-original)/np.linalg.norm(original)>1e-10:raise ValueError('ANGULAR_BIRTH_OPERATOR')
        m._angular_birth[cell['id']]=raw-raw.mean(axis=0)+original
    return m._angular_birth[cell['id']]


def mechanical(m,T,D,oldT,dt,M,basal):
    validate(m);o=native.operators(m);full=np.asarray(m.state['angular_memory'])
    current=scale(m,D);previous=scale(m,m.state['damage']);unit=np.array([1.,1.,1.,0.,0.,0.]);blocks=[]
    for f in range(m.nf):
        nodes=m.solid_nodes[f];P=m.P[f];w=m.weights[f]
        R=np.linalg.solve(m.G[f],P.T@((w*native.rate(m,T[nodes]))[:,None]*P))
        theta=np.linalg.solve(P,T[nodes]-oldT[nodes])[:,None]*m.shell['linear_expansion_per_k']*unit
        for a in range(3):blocks.append(dict(H=np.kron(m.G[f],m.C[f]*current[f])/3,L=m._angular_reference.L[f,a],
            R=np.kron(R,np.eye(6)),old=full[f,a].ravel(),theta=theta.ravel(),dofs=o['dofs'][f]))
    catalogue={x['id']:x for x in m.config['phase_cells']}
    for x in m.state.get('born_material',[]):
        f=x['face'];i=x['node'];L=born_operator(m,catalogue[x['cell_id']])
        for a in range(3):blocks.append(dict(H=x['volume_m3']*m.C[f]*current[f]/3,L=L[a],R=np.eye(6)*native.rate(m,T[i]),
            old=np.asarray(m.state['angular_born_memory'][x['cell_id']][a]),theta=m.shell['linear_expansion_per_k']*(T[i]-oldT[i])*unit,dofs=o['dofs'][f]))
    r=native.step(M,o['D'] if basal is None else basal['D'],o['F'] if basal is None else basal['F'],np.asarray(m.state['nodal_velocity']),dt,blocks)
    if r['ledger']['momentum_relative']>1e-8 or r['ledger']['balance_relative']>5e-10:raise ValueError('ANGULAR_OPERATOR_BALANCE')
    new=np.asarray(r['memory'][:3*m.nf]).reshape(m.nf,3,2,6)
    heat=np.zeros(len(T));sigma=np.zeros((m.nf,3));volume=m.G[:,0,0].copy();dU=0.;exchanges=np.zeros(2)
    for f in range(m.nf):
        nodes=m.solid_nodes[f];C=m.C[f]*current[f]
        e=np.einsum('ri,aij->arj',m.P[f],new[f]);stress=e@C
        old=np.einsum('ri,aij->arj',m.P[f],full[f])
        Q=dt*m.weights[f]*native.rate(m,T[nodes])*np.einsum('ari,ari->r',stress,e)/3
        W=-m.weights[f]*m.shell['linear_expansion_per_k']*(T[nodes]-oldT[nodes])*stress[:,:,:3].sum(axis=(0,2))/3
        delta=.5*m.weights[f]*np.einsum('ari,ij,arj->r',old,m.C[f]*(current[f]-previous[f]),old)/3
        heat[nodes]+=Q-W-delta;exchanges+=[Q.sum(),W.sum()];dU+=delta.sum()
        sigma[f]=np.einsum('r,ari->i',m.weights[f],stress)[[0,1,3]]/3
    born=deepcopy(m.state.get('born_material',[]));angular_born={};collapsed=[x.mean(axis=0).ravel() for x in new]
    for j,x in enumerate(born):
        f=x['face'];i=x['node'];start=3*m.nf+3*j
        e=np.asarray(r['memory'][start:start+3]);old=np.asarray(m.state['angular_born_memory'][x['cell_id']])
        delta=.5*x['volume_m3']*np.einsum('ai,ij,aj->',old,m.C[f]*(current[f]-previous[f]),old)/3
        Q=sum(row['maxwell'] for row in r['rows'][start:start+3]);W=sum(row['thermal_work'] for row in r['rows'][start:start+3])
        heat[i]+=Q-W-delta;exchanges+=[Q,W];dU+=delta
        sigma[f]+=x['volume_m3']*(m.C[f]*current[f]@e.mean(axis=0))[[0,1,3]];volume[f]+=x['volume_m3']
        x['memory']=e.mean(axis=0).tolist();angular_born[x['cell_id']]=e.tolist();collapsed.append(e.mean(axis=0))
    for value,key in zip(exchanges,('maxwell','thermal_work')):
        if abs(value-r['ledger'][key])>1e3+3e-8*abs(value):raise ValueError('ANGULAR_NODE_WORK_IDENTITY')
    if basal is None:
        for f,idx in enumerate(o['dofs']):heat[m.solid_nodes[f]]+=dt*(r['velocity'][idx]@o['face_D'][f]@r['velocity'][idx])/2
    else:
        exchange=budget(basal,r['velocity'],dt)
        np.add.at(heat,basal['geometry']['thermal_nodes'],[x['heat_j'] for x in exchange['rows']])
        r['basal_budget']={k:v for k,v in exchange.items() if k!='rows'}
    r.update(memory=collapsed,angular_memory=new.tolist(),angular_born_memory=angular_born,
        stress=sigma/volume[:,None],heat=heat,damage_energy_change=float(dU),born=born,blocks=blocks)
    return r


def finalize(m,previous,result):
    m.state['angular_memory']=result['angular_memory']
    m.state['angular_born_memory']=deepcopy(result['angular_born_memory'])
    for x in m.state.get('born_material',[]):
        if x['cell_id'] not in m.state['angular_born_memory']:
            if x['birth_time_myr']!=m.state['time_myr'] or np.any(x['memory']):raise ValueError('UNKNOWN_ANGULAR_BIRTH_HISTORY')
            m.state['angular_born_memory'][x['cell_id']]=np.zeros((3,6)).tolist()
    validate(m)


def base_face_energy(m):
    e=np.asarray(m.state['angular_memory']).reshape(m.nf,3,12)
    H=np.array([np.kron(m.G[f],m.C[f]) for f in range(m.nf)])
    out=np.einsum('fai,fij,faj->f',e,H,e)/6
    for x in m.state.get('born_material',[]):
        b=np.asarray(m.state['angular_born_memory'][x['cell_id']])
        out[x['face']]+=.5*x['volume_m3']*np.einsum('ai,ij,aj->',b,m.C[x['face']],b)/3
    return out


def commit_damage(m,new):
    previous=np.asarray(m.state.get('experimental_edge_fatigue',{}).get('damage',np.zeros(len(m.mesh.shared_edges))))
    new=np.asarray(new);face_damage(m.mesh,new)
    if np.any(new<previous):raise ValueError('EDGE_HEALING_NOT_SUPPORTED')
    before=scale(m,m.state['damage'],previous);after=scale(m,m.state['damage'],new);base=base_face_energy(m)
    per_face=base*(before-after)
    if np.any(per_face<0):raise ValueError('ANGULAR_EDGE_RELEASE_SIGN')
    release=float(per_face.sum())
    m.state['experimental_edge_fracture_reserve_j']=m.state.get('experimental_edge_fracture_reserve_j',0.)+release
    m.state.setdefault('experimental_edge_fatigue',{})['damage']=new.tolist()
    return dict(release_j=release,elastic_before_j=float(base@before),elastic_after_j=float(base@after),
        face_release_j=per_face.tolist(),scale_before=before,scale_after=after)


class AngularCoupled:
    def __init__(self,reference,model,origin):
        self.reference=reference;self.model=model;self.origin=deepcopy(origin)
        self.model._angular_reference=reference;validate(self.model)

    @classmethod
    def from_source_checkpoint(cls,path):
        ref,s=SourceAngular.load(path)
        if s['elapsed_years']!=0 or s['heat_buffer_j']!=0:raise ValueError('UNALLOCATED_THERMAL_HISTORY: use initialized source checkpoint')
        c=deepcopy(ref.m.config);state=deepcopy(ref.m.state)
        c['experimental_angular_memory']=VERSION
        c['experimental_geological_edge_fatigue']=dict(shear_strength_pa=1e6,damage_time_years=100.,mode=MODE)
        state.update(angular_memory=np.asarray(s['memory']).reshape(ref.nf,3,2,6).tolist(),angular_born_memory={},
            memory=np.asarray(s['memory']).mean(axis=1).reshape(ref.nf,2,6).tolist(),
            nodal_velocity=s['velocity'],nodal_displacement=s['displacement'],
            experimental_edge_fatigue=dict(damage=s['edge_damage']),experimental_edge_fracture_reserve_j=s['reserve_j'])
        origin=dict(path=str(path),sha256=digest(Path(path).read_bytes()),initialization=s['initialization'],mapping=s['mapping'],source_file_sha256=s['source_file_sha256'])
        return cls(ref,SimplifiedGenesis(c,state),origin)

    def advance(self,years):
        if not np.isfinite(years) or not 0<years<=self.model.config['max_step_years']:raise ValueError('STEP_RANGE')
        before=deepcopy(self.model.state)
        try:
            native.advance(self.model,years,mechanical_solver=mechanical,state_finalizer=finalize,edge_commit_solver=commit_damage)
            validate(self.model)
        except Exception:
            self.model.state=before
            raise
        return deepcopy(self.model.state['last_ledger'])

    def save(self,path):
        validate(self.model)
        payload=dict(origin=self.origin,reference_source=self.reference.envelope,config=self.model.config,state=self.model.state)
        with Path(path).open('xb') as stream:stream.write(encoded(dict(format=VERSION,sha256=digest(encoded(payload)),payload=payload)))

    @classmethod
    def load(cls,path):
        obj=json.loads(Path(path).read_bytes());p=obj['payload']
        if obj['format']!=VERSION or digest(encoded(p))!=obj['sha256']:raise ValueError('ANGULAR_COUPLED_CHECKSUM')
        return cls(SourceAngular(p['reference_source']),SimplifiedGenesis(p['config'],p['state']),p['origin'])
