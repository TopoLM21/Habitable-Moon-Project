"""Isolated fixed-chart J2 experiment. Never calls planetary advance/save."""
from copy import deepcopy
import hashlib,json,time
from pathlib import Path
import numpy as np
from scipy.linalg import solve
from tectonics.simplified_genesis import SimplifiedGenesis
from tectonics.simplified_genesis_nodal import blocks,operators,material_mass,rate
from tectonics.simplified_genesis_basal import assemble
from tectonics.genesis_onset_support import NonlocalLoading

ROOT=Path(__file__).parent
SOURCE=Path('/workspace/genesis-moving-basal-source-01.json')
YEAR=31557600.
PARAM=dict(shear_yield_pa=1e6,residual_fraction=.4,
           equivalent_critical_strain=2.5e-4/np.sqrt(3),
           equivalent_plastic_viscosity_pa_s=3*6.31152e16,
           ell_km=4500.,dt_years=.025,steps=8)


def j2(e,C,relax,dt,Y):
    """Vectorized implicit Maxwell + radial J2 return, fixed current yield.

    Engineering strain Voigt, physical stress Voigt. C isotropic, damage frozen.
    relax=dt*E/eta inherited; plastic viscosity is q/lambda_dot.
    """
    Ce=C/(1+relax[:,None,None]);trial=np.einsum('qij,qj->qi',Ce,e)
    mean=trial[:,:3].mean(axis=1);dev=trial.copy();dev[:,:3]-=mean[:,None]
    q=np.sqrt(1.5*np.sum(dev[:,:3]**2,axis=1)+3*np.sum(dev[:,3:]**2,axis=1))
    G=Ce[:,3,3];eta=PARAM['equivalent_plastic_viscosity_pa_s']
    den=3*G+eta/dt;dl=np.maximum(q-Y,0)/den
    direction=np.zeros_like(dev);positive=q>0
    direction[positive]=1.5*dev[positive]/q[positive,None]
    direction[:,3:]*=2 # tensor -> engineering shear strain
    dep=dl[:,None]*direction
    elastic=(e-dep)/(1+relax[:,None]);stress=np.einsum('qij,qj->qi',C,elastic)
    tangent=Ce.copy();active=q>Y
    for i in np.flatnonzero(active):
        K=Ce[i,:3,:3].sum()/9;bulk=np.zeros((6,6));bulk[:3,:3]=K
        f=1-3*G[i]*dl[i]/q[i]
        tangent[i]=bulk+f*(Ce[i]-bulk)-9*G[i]**2*Y[i]/(den[i]*q[i]**3)*np.outer(dev[i],dev[i])
    return elastic,stress,dep,dl,tangent


class Shell:
    def __init__(self):
        m=SimplifiedGenesis.load(SOURCE);self.m=m;o=operators(m)
        T=m.temperature(m.state['energy']);D=np.array(m.state['damage'])
        self.blocks=blocks(m,T,D,T);self.M=material_mass(m);basal=assemble(m)
        self.drag=basal['D'];self.force=basal['F'];self.nd=len(self.M)
        B=[];C=[];w=[];rates=[];face=[];maps=[];old=[];projection_error=0.
        for j,b in enumerate(self.blocks):
            if j<m.nf:
                P=m.P[j];ww=m.weights[j];cc=m.C[j]*m.scale(D)[j]
                rr=rate(m,T[m.solid_nodes[j]]);f=j
            else:
                born=m.state['born_material'][j-m.nf];f=born['face'];P=np.ones((1,1))
                ww=[born['volume_m3']];cc=m.C[f]*m.scale(D)[f];rr=rate(m,np.array([T[born['node']]]))
            transform=np.kron(P,np.eye(6));local=(transform@b['L']).reshape(len(P),6,18)
            full=np.zeros((len(P),6,self.nd));full[:,:,b['dofs']]=local
            B.extend(full);C.extend([cc]*len(P));w.extend(ww);rates.extend(rr);face.extend([f]*len(P))
            old.extend((transform@b['old']).reshape(-1,6));maps.append(P)
            projected=transform.T@np.kron(np.diag(ww),cc)@transform
            norm=1/np.sqrt(np.diag(b['H']))
            # Cross moments nearly cancel: compare in the elastic-energy norm.
            error=float(np.max(abs((projected-b['H'])*norm[:,None]*norm[None,:])))
            assert error<1e-12;projection_error=max(projection_error,error)
        self.B=np.array(B);self.C=np.array(C);self.w=np.array(w);self.rates=np.array(rates)
        self.faces=np.array(face);self.maps=maps;self.vol=np.bincount(self.faces,weights=self.w,minlength=m.nf)
        self.filter=NonlocalLoading(m.mesh,m.config['radius_km'],PARAM['ell_km'])
        self.initial=dict(memory=[b['old'].tolist() for b in self.blocks],kappa=np.zeros(len(w)).tolist(),
            velocity=list(m.state['nodal_velocity']),elapsed_years=0.,ledger={})
        self.old=np.array(old)
        self.projection_error=projection_error

    def expand(self,memory):
        return np.concatenate([P@np.array(e).reshape(len(P),6) for P,e in zip(self.maps,memory)])

    def pack(self,e):
        out=[];k=0
        for P in self.maps:
            out.append(np.linalg.solve(P,e[k:k+len(P)]).ravel().tolist());k+=len(P)
        return out

    def averaged(self,k):
        face=np.bincount(self.faces,weights=self.w*k,minlength=len(self.vol))/self.vol
        return self.filter.apply(face,np.ones(len(face),bool))[self.faces]

    def step(self,state,weakening,deadline):
        dt=PARAM['dt_years']*YEAR;e0=self.expand(state['memory']);k0=np.array(state['kappa']);v0=np.array(state['velocity'])
        base=self.M/dt**2+self.drag/dt;rhs=self.M@v0/dt+self.force
        du=dt*v0;guess=k0.copy();inner_count=0;max_momentum=0
        for outer in range(40):
            bar=self.averaged(guess)
            factor=PARAM['residual_fraction']+(1-PARAM['residual_fraction'])*np.maximum(1-bar/PARAM['equivalent_critical_strain'],0) if weakening else np.ones_like(bar)
            Y=np.sqrt(3)*PARAM['shear_yield_pa']*factor
            def evaluate(u,stiff=False):
                raw=e0+np.einsum('qij,j->qi',self.B,u)
                result=j2(raw,self.C,dt*self.rates,dt,Y);e,s,dep,dl,tan=result
                internal=np.einsum('qij,qi,q->j',self.B,s,self.w)
                residual=base@u-rhs+internal
                scale=max(np.linalg.norm(internal),np.linalg.norm(self.force),1.)
                if stiff:
                    CB=np.einsum('qij,qjk->qik',tan,self.B)*self.w[:,None,None]
                    K=base+self.B.reshape(-1,self.nd).T@CB.reshape(-1,self.nd)
                    return result,residual,scale,K
                return result,residual,scale
            for inner in range(25):
                if time.monotonic()>deadline:raise TimeoutError('300 second diagnostic budget')
                result,res,scale,K=evaluate(du,True);relative=np.linalg.norm(res)/scale;inner_count+=1
                if relative<1e-9:break
                diagonal=1/np.sqrt(np.diag(K));delta=diagonal*solve(K*diagonal[:,None]*diagonal[None,:],-res*diagonal,assume_a='sym')
                for back in range(16):
                    candidate=du+delta*2**(-back);_,r2,sc2=evaluate(candidate)
                    if np.linalg.norm(r2)/sc2<relative:du=candidate;break
                else:raise RuntimeError('Newton line search did not reduce residual')
            else:raise RuntimeError('Newton iteration limit')
            e,s,dep,dl,tan=result;newk=k0+dl
            # Absolute strain tolerance scaled by prescribed critical strain.
            change=np.max(abs(newk-guess))/PARAM['equivalent_critical_strain']
            if change<1e-9:break
            guess=newk
        else:raise RuntimeError('Nonlocal iteration limit')
        max_momentum=relative
        # Check the constitutive residual with endpoint, not lagged, weakening.
        truebar=self.averaged(newk)
        fac=PARAM['residual_fraction']+(1-PARAM['residual_fraction'])*np.maximum(1-truebar/PARAM['equivalent_critical_strain'],0) if weakening else np.ones_like(truebar)
        verified=j2(e0+np.einsum('qij,j->qi',self.B,du),self.C,dt*self.rates,dt,np.sqrt(3)*PARAM['shear_yield_pa']*fac)
        cerr=float(np.max(abs(verified[3]-dl))/PARAM['equivalent_critical_strain']);assert cerr<1e-9
        v=du/dt;de=e-e0
        U0=.5*np.einsum('qi,qij,qj,q->',e0,self.C,e0,self.w);U1=.5*np.einsum('qi,qij,qj,q->',e,self.C,e,self.w)
        Qm=dt*np.einsum('qi,qi,q,q->',s,e,self.rates,self.w)
        Qp=np.einsum('qi,qi,q->',s,dep,self.w)
        Nm=.5*np.einsum('qi,qij,qj,q->',de,self.C,de,self.w)
        K0=.5*v0@self.M@v0;K1=.5*v@self.M@v;Nk=.5*(v-v0)@self.M@(v-v0)
        drag=dt*v@self.drag@v;work=self.force@du
        balance=U1-U0+K1-K0+Qm+Qp+Nm+Nk+drag-work
        denom=max(abs(U0),abs(U1),abs(K0),abs(K1),abs(work),Qm,Qp,drag,1.)
        assert abs(balance)/denom<5e-10 and max_momentum<1e-8
        assert min(Qm,Qp,Nm,Nk,drag)>=0 and np.max(abs(dep[:,:3].sum(axis=1)))<1e-14
        ledger=dict(maxwell=Qm,plastic=Qp,damage=0.,drag=drag,thermal_work=0.,external=work,
                    numerical_memory=Nm,numerical_kinetic=Nk,elastic_change=U1-U0,kinetic_change=K1-K0)
        new=deepcopy(state);new.update(memory=self.pack(e),kappa=newk.tolist(),velocity=v.tolist(),elapsed_years=state['elapsed_years']+PARAM['dt_years'])
        new['ledger']={key:state['ledger'].get(key,0)+value for key,value in ledger.items()}
        return new,dict(outer_iterations=outer+1,newton_evaluations=inner_count,momentum_relative=max_momentum,
            energy_relative=abs(balance)/denom,constitutive_relative=cerr,plastic_j=Qp,maxwell_j=Qm,
            active_points=int(np.sum(dl>0)),kappa_max=float(newk.max()))


def controls(shell):
    G=2e10;K=4e10;C=np.zeros((6,6));C[:3,:3]=K-2*G/3;C[np.arange(3),np.arange(3)]+=2*G;C[3:,3:]=np.eye(3)*G
    dt=.025*YEAR;eta=PARAM['equivalent_plastic_viscosity_pa_s'];Y=np.sqrt(3)*1e6
    raw=np.array([[0.,0.,0.,2e-4,0.,0.]])
    e,s,dep,dl,tan=j2(raw,C[None],np.array([0.]),dt,np.array([Y]))
    qtrial=np.sqrt(3)*G*raw[0,3];expected=(qtrial-Y)/(3*G+eta/dt)
    assert abs(dl[0]-expected)<1e-18 and abs(dep[0,:3].sum())<1e-18
    assert abs(s[0]@dep[0]-(np.sqrt(3)*s[0,3])*dl[0])<1e-10
    assert s[0]@dep[0]>0
    hydro=j2(np.array([[1e-4]*3+[0.]*3]),C[None],np.array([0.]),dt,np.array([Y]));assert hydro[3][0]==0
    # Finite-difference tangent, including engineering shear factors.
    eps=1e-8;numeric=np.zeros((6,6))
    for i in range(6):
        dr=np.eye(6)[i]*eps
        numeric[:,i]=(j2(raw+dr,C[None],np.array([0.]),dt,np.array([Y]))[1][0]-j2(raw-dr,C[None],np.array([0.]),dt,np.array([Y]))[1][0])/(2*eps)
    tangent_error=float(np.linalg.norm(numeric-tan[0])/np.linalg.norm(tan[0]));assert tangent_error<1e-7
    # Rigid infinitesimal rotation has A=P=omega cross X in conical kinematics.
    X=operators(shell.m)['X'];omega=np.array([2e-6,-1e-6,3e-6]);u=np.tile(np.cross(omega,X),(1,2)).ravel()
    rotation=np.einsum('qij,j->qi',shell.B,u);rotation_error=float(np.max(abs(rotation)))
    assert rotation_error<1e-12
    rr=j2(rotation,shell.C,np.zeros(len(shell.w)),dt,np.full(len(shell.w),Y));assert np.max(rr[3])==0
    from analysis.genesis_aggregate_coupled.core import step as maxwell_step
    legacy=maxwell_step(shell.M,shell.drag,shell.force,np.array(shell.initial['velocity']),dt,shell.blocks)
    raw=shell.old+np.einsum('qij,j->qi',shell.B,legacy['displacement_increment'])
    inactive=j2(raw,shell.C,dt*shell.rates,dt,np.full(len(shell.w),1e99))
    reference=shell.expand(legacy['memory'])
    maxwell_error=float(np.max(abs(inactive[0]-reference))/max(np.max(abs(reference)),1e-30))
    assert maxwell_error<1e-10 and np.max(inactive[3])==0
    return dict(status='PASS',pure_shear_lambda_error=float(abs(dl[0]-expected)),hydrostatic_plastic_zero=True,
                tangent_relative=tangent_error,rotation_strain_max=rotation_error,isochoric_trace=float(dep[0,:3].sum()),positive_plastic_work=True,
                old_Maxwell_limit_relative=maxwell_error,projection_energy_norm_error=shell.projection_error)


def main():
    start=time.monotonic();deadline=start+300
    protected=[SOURCE,Path('/workspace/genesis-moving-basal-pilot-01/checkpoint.json')]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    shell=Shell();result=dict(parameters=PARAM,source_hashes=hashes,controls=controls(shell),cases=[])
    for weak in (False,True):
        s=deepcopy(shell.initial);rows=[];case_start=time.monotonic();saved=None
        try:
            for j in range(PARAM['steps']):
                s,row=shell.step(s,weak,deadline);rows.append(row)
                if j+1==PARAM['steps']//2:saved=deepcopy(s)
            path=ROOT/('shell_j2_restart_B.json' if weak else 'shell_j2_restart_A.json')
            path.write_text(json.dumps(dict(parameters=PARAM,source_hashes=hashes,weakening=weak,state=saved),indent=2)+'\n')
            reload=json.loads(path.read_text());assert reload['parameters']==PARAM
            restored=reload['state']
            for _ in range(PARAM['steps']//2,PARAM['steps']):restored,_=shell.step(restored,weak,deadline)
            exact=json.dumps(restored,sort_keys=True)==json.dumps(s,sort_keys=True);assert exact
            face=np.bincount(shell.faces,weights=shell.w*np.array(s['kappa']),minlength=len(shell.vol))/shell.vol
            weights=shell.vol/shell.vol.sum();mean=float(weights@face);std=float(np.sqrt(weights@(face-mean)**2))
            result['cases'].append(dict(status='PASS',weakening=weak,seconds=time.monotonic()-case_start,restart_exact=exact,
                rows=rows,ledger=s['ledger'],face_kappa=face.tolist(),volume_mean_kappa=mean,volume_std_kappa=std,
                cv=std/mean if mean else 0,participation_fraction=mean**2/float(weights@face**2) if mean else 0,
                active_faces=int(np.sum(face>0))))
        except (RuntimeError,AssertionError,TimeoutError) as exc:
            result['cases'].append(dict(status='FAILED',weakening=weak,error=str(exc),accepted_steps=len(rows),rows=rows));break
    for p in protected:assert hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p)]
    kernel=shell.filter.apply(np.eye(shell.m.nf),np.ones(shell.m.nf,bool))
    support=1/np.sum(kernel**2,axis=1);h=np.sqrt(shell.m.mesh.physical_cell_areas_km2(shell.m.config['radius_km']))
    assert np.max(abs(kernel.sum(axis=1)-1))<1e-12 and kernel.min()>0
    result.update(seconds=time.monotonic()-start,unknowns=shell.nd,quadrature_points=len(shell.w),
                  retained_memory_coefficients=sum(len(b['old']) for b in shell.blocks),source_unchanged=True)
    result['filter_support']=dict(cell_size_km_min=float(h.min()),cell_size_km_max=float(h.max()),
        ell_over_cell_size_min=float(PARAM['ell_km']/h.max()),effective_cells_min=float(support.min()),
        effective_cells_max=float(support.max()),support_faces=int(shell.m.nf))
    if len(result['cases'])==2 and all(c['status']=='PASS' for c in result['cases']):
        a,b=result['cases'];result['B_over_A']=dict(mean_kappa=b['volume_mean_kappa']/a['volume_mean_kappa'],
            std_kappa=b['volume_std_kappa']/a['volume_std_kappa'],cv=b['cv']/a['cv'])
    (ROOT/'shell_j2_result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({**result,'cases':[{k:v for k,v in c.items() if k not in ('face_kappa','rows')} for c in result['cases']]},indent=2))


if __name__=='__main__':main()
