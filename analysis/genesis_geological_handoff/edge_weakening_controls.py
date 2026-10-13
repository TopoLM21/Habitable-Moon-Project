"""Bounded vector-shell controls before the one real feedback pilot."""
from copy import deepcopy
from pathlib import Path
import json,time,tempfile
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from tectonics.simplified_genesis import SimplifiedGenesis
from tectonics.simplified_genesis_nodal import mechanical,material_mass
from tectonics.experimental_edge_weakening import MODE,face_damage,scale,commit_damage
from tectonics.genesis_transfer_bridge import encoded
from .nodal_retirement_check import SOURCE,OPTION
from .driven_shell import Driven,CONFIG

ROOT=Path(__file__).parent
YEAR=31557600.


def local_controls():
    m=SimplifiedGenesis.load(SOURCE);m.config['experimental_geological_edge_fatigue']={**OPTION,'mode':MODE}
    inventory=encoded(dict(mass=m.config['mass'],H=m.state['energy']));T=m.temperature(m.state['energy']);D=np.array(m.state['damage'])
    n=deepcopy(m);edge=np.zeros(len(m.mesh.shared_edges));edge[0]=.8
    change=commit_damage(n,edge);assert change['release_j']>0
    assert abs(change['elastic_after_j']-change['elastic_before_j']+change['release_j'])/change['elastic_before_j']<1e-14
    assert encoded(dict(mass=n.config['mass'],H=n.state['energy']))==inventory
    a=mechanical(m,T,D,T,25*YEAR,material_mass(m));b=mechanical(n,T,D,T,25*YEAR,material_mass(n))
    f,g,*_=m.mesh.shared_edges[0];ids=[f,g];outside=np.ones(m.nf,bool);outside[ids]=False
    x=np.linalg.norm(a['stress'],axis=1);y=np.linalg.norm(b['stress'],axis=1)
    assert np.all(y[ids]<x[ids]) and np.max((y-x)[outside])>0
    neighbors=sorted({h for k in ids for h in m.mesh.neighbors[k] if h not in ids})
    assert np.max((y-x)[neighbors])>0
    zero=deepcopy(m);zero.state['memory']=np.zeros_like(zero.state['memory']).tolist();zero.state['nodal_velocity']=np.zeros_like(zero.state['nodal_velocity']).tolist()
    z=commit_damage(zero,np.full(len(edge),.8));assert z['release_j']==0
    from analysis.genesis_aggregate_coupled.core import step
    from tectonics.simplified_genesis_nodal import operators,blocks
    o=operators(zero);bb=blocks(zero,T,D,T);result=step(material_mass(zero),o['D'],np.zeros(len(o['F'])),np.zeros(len(o['F'])),YEAR,bb)
    assert np.max(abs(result['velocity']))==0 and np.max(abs(result['memory']))==0
    uniform=deepcopy(m);commit_damage(uniform,np.full(len(edge),.8))
    np.testing.assert_array_equal(scale(uniform,D),uniform.scale(np.maximum(D,.8)))
    # Old damage already stronger: edge growth must neither weaken twice nor release twice.
    overlap=deepcopy(m);overlap.state['damage']=np.full(m.nf,.9).tolist()
    assert commit_damage(overlap,np.full(len(edge),.8))['release_j']==0
    before=encoded(n.state);again=commit_damage(n,edge);assert again['release_j']==0 and encoded(n.state)==before
    # Actual coupled first step, then disk restart; no saved planetary checkpoint overwritten.
    m.advance(25.)
    with tempfile.TemporaryDirectory(prefix='edge-weakening-control-') as tmp:
        m.save(Path(tmp)/'restart.json');loaded=SimplifiedGenesis.load(Path(tmp)/'restart.json')
        m.advance(25.);loaded.advance(25.);assert encoded(m.state)==encoded(loaded.state)
    L=m.state['last_ledger'];assert abs(L['whole_balance_j'])<=L['whole_allowance_j'];assert L['edge_damage_heat_j']==0
    return dict(source_patch_edge=0,patch_faces=ids,patch_stress_ratio=(y[ids]/x[ids]).tolist(),
        adjacent_faces=neighbors,adjacent_stress_change_pa=(y-x)[neighbors].tolist(),
        outside_max_increase_pa=float(np.max((y-x)[outside])),velocity_relative_change=float(np.linalg.norm(b['velocity']-a['velocity'])/np.linalg.norm(a['velocity'])),
        instantaneous_release_j=change['release_j'],inventory_exact=True,unloaded_exact=True,
        uniform_max_competition_exact=True,no_double_release=True,restart_exact=True,
        coupled_energy_fraction=abs(L['whole_balance_j'])/L['whole_allowance_j'],coupled_fracture_heat_j=L['edge_damage_heat_j'])


class GridControl:
    """Same damage law on independent refined vector operators; no thermal history."""
    def __init__(self,level):
        self.a=a=Driven(level);self.mesh=a.mesh
        # Same frozen old-damage field inherited from existing driven diagnostic.
        self.old=a.damage;self.base_scale=.02+.98*(1-self.old)**2
        frames=[]
        for ids in self.mesh.faces:
            tri=self.mesh.vertices[ids];t=tri[1]-tri[0];t/=np.linalg.norm(t)
            normal=np.cross(tri[1]-tri[0],tri[2]-tri[0]);normal/=np.linalg.norm(normal)
            frames.append(np.column_stack((t,np.cross(normal,t),normal)))
        self.frames=np.array(frames)
    def ratio(self,edge):
        d=np.maximum(self.old,face_damage(self.mesh,edge));return (.02+.98*(1-d)**2)/self.base_scale
    def shear(self,stress):
        sigma=stress.reshape(-1,2,6).mean(axis=1);t=np.zeros((self.a.nf,3,3));t[:,range(3),range(3)]=sigma[:,:3]
        for k,(i,j) in enumerate(((0,1),(0,2),(1,2)),3):t[:,i,j]=t[:,j,i]=sigma[:,k]
        w=np.einsum('fai,fij,fbj->fab',self.frames,t,self.frames);out=[]
        for f,g,i,j in self.mesh.shared_edges:
            u,v=self.mesh.vertices[[i,j]];rad=u+v;rad/=np.linalg.norm(rad);tangent=v-u;tangent/=np.linalg.norm(tangent)
            n=np.cross(tangent,rad);traction=.5*(w[f]+w[g])@n;out.append(np.linalg.norm(traction-n*(traction@n)))
        return np.array(out)
    def initial(self):
        return dict(e=np.zeros((len(self.a.w),6)).tolist(),v=np.zeros(self.a.nd).tolist(),d=np.zeros(len(self.mesh.shared_edges)).tolist(),reserve=0.,time=0.)
    def step(self,s,years):
        a=self.a;dt=years*YEAR;e0=np.array(s['e']);v0=np.array(s['v']);d=np.array(s['d'])
        C=a.C*np.repeat(self.ratio(d),2)[:,None,None];r=CONFIG['young_pa']/CONFIG['maxwell_viscosity_pa_s'];relax=1+dt*r
        local=np.einsum('qki,qkl,qlj,q->qij',a.B,C,a.B,a.w).reshape(a.nf,2,18,18).sum(axis=1)/relax
        K=sparse.coo_matrix((local.ravel(),(a.row,a.col)),shape=a.M.shape).tocsr();A=a.M/dt**2+K
        stress0=np.einsum('qij,qj->qi',C,e0)/relax;internal=np.zeros(a.nd)
        np.add.at(internal,a.ids.ravel(),np.einsum('qij,qi,q->qj',a.B,stress0,a.w).ravel())
        rhs=a.M@v0/dt+a.force-internal;u=np.zeros(a.nd);Af=A[a.free][:,a.free];scl=1/np.sqrt(Af.diagonal())
        u[a.free]=scl*spsolve(sparse.diags(scl)@Af@sparse.diags(scl),scl*rhs[a.free])
        v=u/dt;e=(e0+np.einsum('qij,qj->qi',a.B,u[a.ids]))/relax;stress=np.einsum('qij,qj->qi',C,e)
        shear=self.shear(stress);dn=np.minimum(1,d+years/100*np.maximum(shear/1e6-1,0)**2)
        Cn=a.C*np.repeat(self.ratio(dn),2)[:,None,None]
        U0=.5*np.einsum('qi,qij,qj,q->',e0,C,e0,a.w);Up=.5*np.einsum('qi,qi,q->',e,stress,a.w);U1=.5*np.einsum('qi,qij,qj,q->',e,Cn,e,a.w)
        release=Up-U1;maxwell=dt*r*2*Up;num=.5*np.einsum('qi,qij,qj,q->',e-e0,C,e-e0,a.w)
        K0=.5*v0@(a.M@v0);K1=.5*v@(a.M@v);numK=.5*(v-v0)@(a.M@(v-v0));work=float(a.force@u)
        balance=U1-U0+K1-K0+release+maxwell+num+numK-work;den=max(U0,U1,K0,K1,abs(work),1.)
        assert release>=-1e-12*den and abs(balance)/den<5e-8,(balance/den,release)
        momentum=np.linalg.norm((A@u-rhs)[a.free])/max(np.linalg.norm(rhs[a.free]),1.)
        assert momentum<1e-8,momentum
        return dict(e=e.tolist(),v=v.tolist(),d=dn.tolist(),reserve=s['reserve']+release,time=s['time']+years),dict(energy_relative=abs(balance)/den,momentum=momentum,max_damage=float(dn.max()),max_shear_pa=float(shear.max()))


def grid_controls():
    out=[]
    for level,dt in [(2,2.),(2,1.),(3,2.)]:
        start=time.monotonic();g=GridControl(level);s=g.initial();rows=[];restart=False
        for i in range(round(20/dt)):
            before=deepcopy(s);s,row=g.step(s,dt);rows.append(row)
            if i==2:restart=g.step(json.loads(json.dumps(before)),dt)[0]==s;assert restart
        out.append(dict(level=level,faces=g.a.nf,dt_years=dt,years=20.,max_damage=max(s['d']),mean_damage=float(np.mean(s['d'])),
            saturated=int(sum(np.array(s['d'])>=1)),reserve_j=s['reserve'],max_energy_relative=max(x['energy_relative'] for x in rows),
            max_momentum=max(x['momentum'] for x in rows),restart_exact=restart,seconds=time.monotonic()-start))
    return out


if __name__=='__main__':
    start=time.monotonic();result=dict(local=local_controls(),grid=grid_controls(),seconds=time.monotonic()-start)
    (ROOT/'edge_weakening_controls.json').write_text(json.dumps(result,indent=2,default=lambda x:x.item())+'\n');print(json.dumps(result,indent=2,default=lambda x:x.item()))
