"""Predeclared sparse, externally driven diagnostic; never advances the planet."""
import json,time,hashlib
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.linalg import qr
from tectonics.mesh import build_icosphere
from tectonics.simplified_genesis import SimplifiedGenesis
from tectonics.basal_coupling import mantle_source_pattern
from tectonics.genesis_onset_support import NonlocalLoading
from tectonics.genesis_volume_basal import local_consistent_mass_matrix,CONSISTENT_ANGULAR_MASS
from analysis.genesis_volume_continuation_validation import independent_audit as audit
from .shell_j2_diagnostic import j2,PARAM as J2,YEAR,SOURCE

ROOT=Path(__file__).parent
CONFIG=dict(radius_km=5287.,thickness_km=20.,density=3300.,young_pa=6e10,poisson=.25,
            maxwell_viscosity_pa_s=1e24,ell_km=4500.,traction_pa=20000.,end_years=2.)


class Driven:
    def __init__(self,level):
        self.mesh=mesh=build_icosphere(level);self.nf=mesh.cell_count;self.nd=6*mesh.vertex_count
        source=SimplifiedGenesis.load(SOURCE);D=np.repeat(source.state['damage'],4**(level-1))
        X=mesh.vertices*CONFIG['radius_km']*1000;inner=1-CONFIG['thickness_km']/CONFIG['radius_km'];rho=CONFIG['density']
        ids=(mesh.faces[:,:,None]*6+np.arange(6)).reshape(-1,18);self.ids=np.repeat(ids,2,axis=0)
        row=np.repeat(ids,18,axis=1).ravel();col=np.tile(ids,(1,18)).ravel();self.row=row;self.col=col
        B=[];weights=[];C=[];mass=[];force=np.zeros(self.nd);N,wa=audit.angular_rule()
        q=np.array([.5-.5/np.sqrt(3),.5+.5/np.sqrt(3)])
        lam=np.cbrt(1-q*(1-inner**3));self.P=np.column_stack((np.ones(2),[1.,-1.]))
        pattern=mantle_source_pattern(mesh,source.shell['seed'])*CONFIG['traction_pa']
        for f,vertices in enumerate(mesh.faces):
            tri=X[vertices];normal=np.cross(tri[1]-tri[0],tri[2]-tri[0]);area=np.linalg.norm(normal)/2;normal/=2*area
            t=tri[1]-tri[0];t/=np.linalg.norm(t);frame=np.column_stack((t,np.cross(normal,t),normal))
            b0,b1=audit.independent_strain_operators(tri,frame,N)
            # Frozen reduced-integration path: averaging before B.T C B loses
            # angular strain variance (local rank 11 rather than 12). New
            # passive controls use integrated_vector_operator.first_step;
            # do not silently reinterpret existing memory/checkpoint evidence.
            b0=np.einsum('p,pij->ij',wa,b0);b1=np.einsum('p,pij->ij',wa,b1)
            B.extend([b0+b1/l for l in lam]);volume=np.linalg.det(tri.T)*(1-inner**3)/6
            weights.extend([volume/2]*2);cc=audit.isotropic_matrix(CONFIG['young_pa'],CONFIG['poisson'],D[f],.02)
            C.extend([cc]*2);mass.append(local_consistent_mass_matrix(rho*volume,1.,inner))
            traction=pattern[vertices];traction-=np.outer(traction@normal,normal)
            load=area*inner**2*CONSISTENT_ANGULAR_MASS@traction
            local=np.concatenate((load,(inner-1)*load),axis=1).ravel();np.add.at(force,ids[f],local)
        self.B=np.array(B);self.C=np.array(C);self.w=np.array(weights);self.vol=self.w.reshape(-1,2).sum(axis=1)
        self.M=sparse.coo_matrix((np.array(mass).ravel(),(row,col)),shape=(self.nd,self.nd)).tocsr()
        rigid=np.zeros((self.nd,6))
        for i in range(3):
            rigid.reshape(-1,6,6)[:,:3,i]=np.eye(3)[i]
            rot=np.cross(np.eye(3)[i],mesh.vertices)
            rigid.reshape(-1,6,6)[:,:3,i+3]=rot;rigid.reshape(-1,6,6)[:,3:,i+3]=rot
        Q=np.linalg.qr(rigid)[0];self.force=force-Q@(Q.T@force)
        _,_,piv=qr(rigid.T,pivoting=True);self.fixed=piv[:6];self.free=np.setdiff1d(np.arange(self.nd),self.fixed)
        self.filter=NonlocalLoading(mesh,CONFIG['radius_km'],CONFIG['ell_km'])
        self.area=mesh.physical_cell_areas_km2(CONFIG['radius_km']);self.area/=self.area.sum()
        self.damage=D;self.initial=dict(memory=np.zeros((self.nf,2,6)).tolist(),kappa=np.zeros(2*self.nf).tolist(),
            velocity=np.zeros(self.nd).tolist(),time=0.,ledger={})
        rotation=np.tile(np.cross(np.array([1.,2.,3.]),mesh.vertices),(1,2)).ravel()
        assert np.max(abs(np.einsum('qij,qj->qi',self.B,rotation[self.ids])))<1e-12
        assert np.linalg.norm(Q.T@self.force)/max(np.linalg.norm(self.force),1)<1e-12

    def step(self,state,weak,years,deadline,newton_tolerance=1e-9):
        dt=years*YEAR;r=CONFIG['young_pa']/CONFIG['maxwell_viscosity_pa_s'];relax=np.full(len(self.w),dt*r)
        e0=np.einsum('ij,fjk->fik',self.P,np.array(state['memory'])).reshape(-1,6)
        k0=np.array(state['kappa']);v0=np.array(state['velocity']);base=self.M/dt**2;rhs=self.M@v0/dt+self.force
        du=dt*v0;guess=k0.copy();count=0
        def averaged(k):
            face=(k*self.w).reshape(-1,2).sum(axis=1)/self.vol
            return np.repeat(self.filter.apply(face,np.ones(self.nf,bool)),2)
        for outer in range(40):
            bar=averaged(guess);fac=.4+.6*np.maximum(1-bar/J2['equivalent_critical_strain'],0) if weak else np.ones_like(bar)
            Y=np.sqrt(3)*J2['shear_yield_pa']*fac
            def evaluate(u,stiff=False):
                raw=e0+np.einsum('qij,qj->qi',self.B,u[self.ids])
                ans=j2(raw,self.C,relax,dt,Y);e,s,dep,dl,tan=ans
                forces=np.einsum('qij,qi,q->qj',self.B,s,self.w);internal=np.zeros(self.nd);np.add.at(internal,self.ids.ravel(),forces.ravel())
                res=base@u-rhs+internal;scale=max(np.linalg.norm(internal[self.free]),np.linalg.norm(self.force[self.free]),1.)
                if not stiff:return ans,res,scale
                cb=np.einsum('qij,qjk->qik',tan,self.B)
                local=np.einsum('qki,qkj,q->qij',self.B,cb,self.w).reshape(self.nf,2,18,18).sum(axis=1)
                mat=base+sparse.coo_matrix((local.ravel(),(self.row,self.col)),shape=(self.nd,self.nd)).tocsr()
                return ans,res,scale,mat
            for it in range(25):
                if time.monotonic()>deadline:raise TimeoutError('run exceeded 300s')
                ans,res,scale,K=evaluate(du,True);count+=1;error=np.linalg.norm(res[self.free])/scale
                if error<newton_tolerance:break
                k=K[self.free][:,self.free];d=1/np.sqrt(k.diagonal());delta=d*spsolve(sparse.diags(d)@k@sparse.diags(d),-res[self.free]*d)
                for back in range(16):
                    u=du.copy();u[self.free]+=delta*2**(-back);_,rr,ss=evaluate(u)
                    if np.linalg.norm(rr[self.free])/ss<error:du=u;break
                else:raise RuntimeError('line search failed')
            else:raise RuntimeError('Newton limit')
            e,s,dep,dl,tan=ans;newk=k0+dl
            if np.max(abs(newk-guess))/J2['equivalent_critical_strain']<1e-9:break
            guess=newk
        else:raise RuntimeError('nonlocal iteration limit')
        bar=averaged(newk);fac=.4+.6*np.maximum(1-bar/J2['equivalent_critical_strain'],0) if weak else np.ones_like(bar)
        checked=j2(e0+np.einsum('qij,qj->qi',self.B,du[self.ids]),self.C,relax,dt,np.sqrt(3)*J2['shear_yield_pa']*fac)
        closure=np.max(abs(checked[3]-dl))/J2['equivalent_critical_strain'];assert closure<1e-9,f'plastic closure {closure}'
        v=du/dt;de=e-e0;U0=.5*np.einsum('qi,qij,qj,q->',e0,self.C,e0,self.w);U1=.5*np.einsum('qi,qij,qj,q->',e,self.C,e,self.w)
        K0=.5*v0@(self.M@v0);K1=.5*v@(self.M@v)
        ledger=dict(maxwell=float(dt*r*np.einsum('qi,qi,q->',s,e,self.w)),plastic=float(np.einsum('qi,qi,q->',s,dep,self.w)),
            numerical_memory=float(.5*np.einsum('qi,qij,qj,q->',de,self.C,de,self.w)),numerical_kinetic=float(.5*(v-v0)@(self.M@(v-v0))),
            elastic_change=float(U1-U0),kinetic_change=float(K1-K0),external=float(self.force@du),damage=0.,drag=0.,thermal_work=0.)
        bal=sum(ledger.values())-2*ledger['external'];den=max(U0,U1,K0,K1,abs(ledger['external']),1)
        assert abs(bal)/den<5e-10,f'energy relative {abs(bal)/den}, momentum {error}'
        assert error<1e-8,f'momentum relative {error}'
        iso=float(np.max(abs(dep[:,:3].sum(axis=1))));assert iso<1e-14
        assert min(ledger[x] for x in ('plastic','maxwell','numerical_memory','numerical_kinetic'))>=0
        memory=np.einsum('ij,fjk->fik',np.linalg.inv(self.P),e.reshape(self.nf,2,6))
        new=dict(memory=memory.tolist(),kappa=newk.tolist(),velocity=v.tolist(),time=state['time']+years,
            ledger={k:state['ledger'].get(k,0)+value for k,value in ledger.items()})
        flow=(dl*self.w).reshape(-1,2).sum(axis=1)/self.vol/years
        return new,dict(time=new['time'],momentum=error,energy=abs(bal)/den,closure=float(closure),isochoric=iso,
            outer=outer+1,newton=count,metrics=self.metrics(flow),flow=flow.tolist())

    def metrics(self,field):
        a=self.area;mean=float(a@field);cv=float(np.sqrt(a@(field-mean)**2)/mean) if mean else 0
        order=np.argsort(field)[::-1];areas=a[order];flux=areas*field[order];total=flux.sum()
        def share(limit):
            remaining=limit;out=0.
            for ar,val in zip(areas,field[order]):
                use=min(remaining,ar);out+=use*val;remaining-=use
                if remaining<=0:break
            return float(out/total) if total else 0.
        fraction50=1.
        if total:
            target=.5*total;fraction50=0
            for ar,val in zip(areas,field[order]):
                use=min(ar,target/max(val,1e-100));fraction50+=use;target-=use*val
                if target<=1e-14*total:break
        return dict(mean=mean,cv=cv,participation_area=1/(1+cv*cv),area_for_half_flow=float(fraction50),
            top20_share=share(.2),above_twice_mean_area=float(a[field>2*mean].sum()),
            equivalent_global_belt_width_km=float(fraction50*2*CONFIG['radius_km']))


def run(level,weak,dt,benchmark=False,newton_tolerance=1e-9):
    start=time.monotonic();deadline=start+300;model=Driven(level);s=model.initial;rows=[];saved=None
    steps=1 if benchmark else round(CONFIG['end_years']/dt)
    try:
        for i in range(steps):
            s,row=model.step(s,weak,dt,deadline,newton_tolerance);rows.append(row)
            if i+1==steps//2:saved=json.loads(json.dumps(s))
        restart=None
        if not benchmark:
            # One additional step from an actual saved midpoint; avoids replaying half a long run.
            path=ROOT/f'driven_restart_{level}_{int(weak)}_{dt}.json'
            path.write_text(json.dumps(dict(config=CONFIG,J2=J2,level=level,weak=weak,dt=dt,state=saved)))
            a,_=model.step(saved,weak,dt,deadline,newton_tolerance);payload=json.loads(path.read_text())
            assert payload['config']==CONFIG and payload['J2']==J2 and payload['dt']==dt
            b,_=model.step(payload['state'],weak,dt,deadline,newton_tolerance);restart=a==b;assert restart
        k=(np.array(s['kappa'])*model.w).reshape(-1,2).sum(axis=1)/model.vol
        tail=rows[-max(1,steps//4):]
        # Keep every scalar diagnostic, but only five spatial snapshots.
        keep={0,steps//4-1,steps//2-1,3*steps//4-1,steps-1}
        for i,row in enumerate(rows):
            if i not in keep:row.pop('flow',None)
        return dict(status='PASS',seconds=time.monotonic()-start,level=level,faces=model.nf,unknowns=model.nd,weak=weak,dt=dt,newton_tolerance=newton_tolerance,
            restart=restart,completed_years=s['time'],ledger=s['ledger'],kappa=k.tolist(),kappa_metrics=model.metrics(k),
            rows=rows,criterion=all(r['metrics']['top20_share']>=.5 and r['metrics']['participation_area']<=.3 for r in tail),
            ell_over_max_cell_size=CONFIG['ell_km']/float(np.sqrt(model.mesh.physical_cell_areas_km2(CONFIG['radius_km'])).max()))
    except (RuntimeError,AssertionError,TimeoutError) as exc:
        return dict(status='FAILED',error=str(exc),seconds=time.monotonic()-start,level=level,weak=weak,dt=dt,completed_years=s['time'],rows=rows)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--level',type=int,required=True);p.add_argument('--weak',type=int,required=True);p.add_argument('--dt',type=float,default=.05);p.add_argument('--benchmark',action='store_true');p.add_argument('--newton-tolerance',type=float,default=1e-9);args=p.parse_args()
    r=run(args.level,bool(args.weak),args.dt,args.benchmark,args.newton_tolerance)
    path=ROOT/f'driven_{args.level}_{args.weak}_{args.dt}_{"bench" if args.benchmark else "run"}.json'
    path.write_text(json.dumps(dict(config=CONFIG,J2=J2,result=r),indent=2)+'\n')
    print(json.dumps({k:v for k,v in r.items() if k not in ('rows','kappa')},indent=2))
