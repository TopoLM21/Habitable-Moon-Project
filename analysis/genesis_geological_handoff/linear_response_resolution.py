"""Bounded matrix/patch diagnosis on the unchanged nested physical geometry."""
from copy import deepcopy
import json,time
import numpy as np
from scipy.linalg import solve,eigvalsh
from tectonics.experimental_angular_coupled import AngularCoupled,mechanical
from tectonics.simplified_genesis_nodal import material_mass
from .coupled_load_audit import SOURCE,FINE,vertex_traction,prolongation,with_traction,rms
from .angular_coupled_check import world_stress
from .edge_spatial_audit import ROOT
YEAR=31557600.


def prepare(a,traction):
    m=deepcopy(a.model);m.state['angular_memory']=np.zeros_like(m.state['angular_memory']).tolist()
    m.state['memory']=np.zeros_like(m.state['memory']).tolist();m.state['nodal_velocity']=np.zeros_like(m.state['nodal_velocity']).tolist()
    T=m.temperature(m.state['energy']);b=with_traction(m,traction);M=material_mass(m)
    r=mechanical(m,T,np.array(m.state['damage']),T,.0625*YEAR,M,b)
    return dict(m=m,M=M,D=b['D'],F=b['F'],blocks=r['blocks'])


def stiffness(p,dt):
    K=np.zeros_like(p['M']);prepared=[]
    for b in p['blocks']:
        S=np.linalg.solve(np.eye(len(b['R']))+dt*b['R'],np.eye(len(b['R'])))
        B=S@b['L'];idx=b['dofs'];K[np.ix_(idx,idx)]+=b['L'].T@b['H']@B
        prepared.append((b,B))
    return K,prepared


def stress(p,prepared,v,dt):
    memory=np.array([B@(dt*v[b['dofs']]) for b,B in prepared]).reshape(p['m'].nf,3,2,6)
    m=p['m'];m.state['angular_memory']=memory.tolist();m.state['memory']=memory.mean(1).tolist()
    return world_stress(m)


def scaled_solve(A,F):
    s=1/np.sqrt(np.diag(A));C=A*s[:,None]*s[None,:]
    v=s*solve(C,s*F,assume_a='pos')
    return v,dict(relative_residual=float(np.linalg.norm(A@v-F)/np.linalg.norm(F)),
        scaled_relative_residual=float(np.linalg.norm(s*(A@v-F))/np.linalg.norm(s*F)))


def norm(v,A):return float(np.sqrt(max(float(v@A@v),0.)))


def compare_case(pc,pf,P,dt,kind='full'):
    kc,bc=stiffness(pc,dt);kf,bf=stiffness(pf,dt)
    def system(p,K):
        if kind=='no_mass':return p['D']+dt*K
        if kind=='no_drag':return p['M']/dt+dt*K
        return p['M']/dt+p['D']+dt*K
    Ac=system(pc,kc);Af=system(pf,kf)
    vc,rc=scaled_solve(Ac,pc['F']);vf,rf=scaled_solve(Af,pf['F'])
    projected=P.T@Af@P;vp,rp=scaled_solve(projected,P.T@pf['F'])
    cs=stress(pc,bc,vc,dt);fs=stress(pf,bf,vf,dt);fs=fs.reshape(80,4,3,3).mean(1)
    V=pc['m'].G[:,0,0];err=rms(cs-fs,V);cn=rms(cs,V);fn=rms(fs,V)
    diff=vf-P@vc
    row=dict(kind=kind,dt_years=dt/YEAR,coarse_stress_rms_pa=cn,fine_stress_rms_pa=fn,
        absolute_stress_difference_pa=err,symmetric_relative_stress=err/max(cn,fn,1e-12),
        coarse_residual=rc,fine_residual=rf,restricted_system_relative=float(np.linalg.norm(projected-Ac)/np.linalg.norm(Ac)),
        constrained_fine_velocity_relative=float(np.linalg.norm(vp-vc)/np.linalg.norm(vc)),
        velocity_l2_relative=float(np.linalg.norm(diff)/np.linalg.norm(vf)),
        velocity_energy_relative=norm(diff,Af)/norm(vf,Af),
        fine_external_power_w=float(pf['F']@vf),coarse_external_power_w=float(pc['F']@vc),
        fine_velocity_terms={name:float(vf@op@vf) for name,op in [('mass',pf['M']/dt),('drag',pf['D']),('elastic',dt*kf)]},
        correction_terms={name:float(diff@op@diff) for name,op in [('mass',pf['M']/dt),('drag',pf['D']),('elastic',dt*kf)]})
    return row,(Ac,Af,kc,kf,bc,bf,vc,vf)


def main():
    out=ROOT/'linear_response_resolution.json'
    if out.exists():raise FileExistsError('Preserve completed diagnosis')
    start=time.monotonic();a=AngularCoupled.load(SOURCE);b=AngularCoupled.load(FINE)
    Pv=prolongation(a.model.mesh,b.model.mesh);P=np.kron(Pv,np.eye(6));tc,_=vertex_traction(a.model)
    pc=prepare(a,tc);pf=prepare(b,Pv@tc)
    print(json.dumps(dict(setup_seconds=time.monotonic()-start,ndofs=[len(pc['F']),len(pf['F'])])),flush=True)
    tick=time.monotonic();base,ops=compare_case(pc,pf,P,.0625*YEAR);first=time.monotonic()-tick
    print(json.dumps(dict(first_pair_seconds=first,base=base)),flush=True)
    rows=[base];deadline=time.monotonic()+90
    for years,kind in [(1/256,'full'),(1.,'full'),(.0625,'no_mass'),(.0625,'no_drag')]:
        if time.monotonic()>deadline:raise TimeoutError('90second bounded control budget')
        try:row,_=compare_case(pc,pf,P,years*YEAR,kind);rows.append(row)
        except np.linalg.LinAlgError as e:rows.append(dict(kind=kind,dt_years=years,error=str(e)))
    Ac,Af,kc,kf,bc,bf,vc,vf=ops
    patches={};X=np.asarray(pc['m'].config['nodal']['X']);R=np.mean(np.linalg.norm(X,axis=1))
    # These known FE velocities are fixed in physical coordinates, independent of
    # each matrix. Manufactured forcing is A*v, kept separate from mantle load.
    for name,B,translation in [('translation',np.zeros((3,3)),np.array([1.,-.4,.7])),
        ('rotation',np.array([[0.,-.7,.4],[.7,0.,-.2],[-.4,.2,0.]]),np.zeros(3)),
        ('affine_shear',np.array([[0.,1.,0.],[1.,0.,0.],[0.,0.,0.]]),np.zeros(3))]:
        exact=np.zeros((len(X),6));exact[:,:3]=X@B.T/R+translation;exact[:,3:]=X@B.T/R;exact=exact.ravel()*1e-10
        ef=P@exact;solc,_=scaled_solve(Ac,Ac@exact);solf,_=scaled_solve(Af,Af@ef)
        sc=stress(pc,bc,exact,.0625*YEAR);sf=stress(pf,bf,ef,.0625*YEAR).reshape(80,4,3,3).mean(1)
        patches[name]=dict(coarse_relative_velocity_error=float(np.linalg.norm(solc-exact)/np.linalg.norm(exact)),
            fine_relative_velocity_error=float(np.linalg.norm(solf-ef)/np.linalg.norm(ef)),
            patch_stress_difference_pa=rms(sc-sf,pc['m'].G[:,0,0]),stress_rms_pa=rms(sc,pc['m'].G[:,0,0]),
            coarse_elastic_power_w=float(exact@kc@exact),fine_elastic_power_w=float(ef@kf@ef))
    spectra={}
    for name,A in [('coarse',Ac),('fine',Af)]:
        s=1/np.sqrt(np.diag(A));e=eigvalsh(A*s[:,None]*s[None,:]);spectra[name]=dict(min=float(e[0]),max=float(e[-1]),condition=float(e[-1]/e[0]))
    result=dict(code_base='bff90555eb4136bcc6eb9738ec3db9927765f90f',rows=rows,patches=patches,scaled_spectrum=spectra,
        boundary_conditions='No pinned vertices, no eliminated dofs. Closed shared-nodal body; inertia and tangential drag retain rigid motions dynamically.',
        restrictions={name:float(np.linalg.norm(P.T@fine@P-coarse)/np.linalg.norm(coarse)) for name,coarse,fine in [('mass',pc['M'],pf['M']),('drag',pc['D'],pf['D']),('relaxed_stiffness',kc,kf)]},
        seconds=time.monotonic()-start,first_pair_seconds=first)
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
