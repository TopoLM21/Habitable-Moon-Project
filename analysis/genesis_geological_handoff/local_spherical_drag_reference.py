"""Small manufactured geometry reference for the unchanged tangential drag law.

Only 3 rigid-rotation modes and one transverse mode. No shell evolution,
new constitutive law, traction projection, or replacement of source geometry.
"""
import json,time
from pathlib import Path
import numpy as np
from scipy.linalg import solve,eigvalsh
from tectonics.basal_coupling import basal_coupling_fraction
from .edge_spatial_audit import ROOT
SOURCE=Path('/workspace/genesis-angular-coupled-01/initial.json')


def quadrature(triangle,order):
    """Radial map of one triangle to its circumsphere; physical area Jacobian."""
    z,w=np.polynomial.legendre.leggauss(order);z=(z+1)/2;w=w/2
    u,v=np.meshgrid(z,z,indexing='ij');wu,wv=np.meshgrid(w,w,indexing='ij')
    u=u.ravel();v=v.ravel();N=np.column_stack((1-u,u*(1-v),u*v))
    y=N@triangle;R=np.mean(np.linalg.norm(triangle,axis=1));r=np.linalg.norm(y,axis=1);n=y/r[:,None]
    du=(1-v)[:,None]*(triangle[1]-triangle[0])+v[:,None]*(triangle[2]-triangle[0]);dv=u[:,None]*(triangle[2]-triangle[1])
    dxdu=R/r[:,None]*(du-n*np.sum(n*du,axis=1)[:,None]);dxdv=R/r[:,None]*(dv-n*np.sum(n*dv,axis=1)[:,None])
    area=np.linalg.norm(np.cross(dxdu,dxdv),axis=1)*(wu*wv).ravel()
    return R*n,n,area,N


def assemble(triangle,beta,order):
    x,n,w,N=quadrature(triangle,order);R=np.mean(np.linalg.norm(triangle,axis=1))
    normal=np.cross(triangle[1]-triangle[0],triangle[2]-triangle[0]);normal/=np.linalg.norm(normal)
    # Rotation fields exact on the physical sphere. Fourth mode is translation
    # normal to the reference flat facet, the smallest transverse enrichment.
    B=np.stack([np.cross(np.broadcast_to(e,x.shape),x/R) for e in np.eye(3)]+[np.broadcast_to(normal,x.shape)],axis=2)
    tangent=B-n[:,:,None]*np.einsum('qi,qij->qj',n,B)[:,None,:]
    flat=B-normal[None,:,None]*np.einsum('i,qij->qj',normal,B)[:,None,:]
    flat[:,:,3]=0. # Exact null mode, avoiding a manufactured roundoff drag.
    curved=beta*np.einsum('q,qik,qil->kl',w,tangent,tangent)
    planar=beta*np.einsum('q,qik,qil->kl',w,flat,flat)
    return dict(x=x,n=n,w=w,B=B,tangent=tangent,curved=curved,planar=planar,normal=normal)


def metrics(a,traction,coefficients):
    nodal_force=a['w'][:,None]*traction;velocity=np.einsum('qik,k->qi',a['B'],coefficients)
    F=np.einsum('q,qik,qi->k',a['w'],a['B'],traction)
    return F,dict(force_n=nodal_force.sum(0).tolist(),moment_nm=np.cross(a['x'],nodal_force).sum(0).tolist(),
        external_power_w=float(np.einsum('qi,qi->',nodal_force,velocity)),
        virtual_work_identity_relative=float(abs(F@coefficients-np.einsum('qi,qi->',nodal_force,velocity))/max(abs(F@coefficients),1.)),
        traction_rms_pa=float(np.sqrt(np.einsum('q,qi,qi->',a['w'],traction,traction)/a['w'].sum())))


def run(triangle,beta,amplitude,order):
    a=assemble(triangle,beta,order);cases={}
    for name,exact in [('rotation',np.array([.3,-.7,.5,0.])),('transverse',np.array([0.,0.,0.,1.]))]:
        # Fixed analytic manufactured field/traction on the SAME sphere for both
        # projectors. Same amplitude for every order; no fitting to either answer.
        exact=exact*(amplitude/beta)
        traction=beta*np.einsum('qik,k->qi',a['tangent'],exact)
        F,summary=metrics(a,traction,exact)
        size=3 if name=='rotation' else 4
        D=a['curved'][:size,:size];scale=1/np.sqrt(np.diag(D))
        solution=scale*solve(D*scale[:,None]*scale[None,:],scale*F[:size],assume_a='pos')
        residual=a['planar'][:size,:size]@exact[:size]-F[:size]
        row=dict(exact_coefficients=exact.tolist(),resultants_and_work=summary,
            curved_relative_solution_error=float(np.linalg.norm(solution-exact[:size])/np.linalg.norm(exact[:size])),
            flat_known_field_residual_relative=float(np.linalg.norm(residual)/np.linalg.norm(F[:size])),
            curved_power_w=float(exact@a['curved']@exact),flat_power_w=float(exact@a['planar']@exact),
            flat_residual_norm=float(np.linalg.norm(residual)),generalized_force_norm=float(np.linalg.norm(F[:size])))
        if name=='rotation':
            flat=solve(a['planar'][:3,:3],F[:3],assume_a='pos')
            verror=np.einsum('qik,k->qi',a['B'][:,:,:3],flat-exact[:3])
            vexact=np.einsum('qik,k->qi',a['B'][:,:,:3],exact[:3])
            row.update(flat_velocity_rms_error_m_s=float(np.sqrt(np.einsum('q,qi,qi->',a['w'],verror,verror)/a['w'].sum())),
                flat_velocity_relative=float(np.sqrt(np.einsum('q,qi,qi->',a['w'],verror,verror)/np.einsum('q,qi,qi->',a['w'],vexact,vexact))))
        else:row.update(flat_transverse_drag_exact_zero=True,flat_equation_unsolvable=bool(F[3]!=0),transverse_generalized_force_n=float(F[3]))
        cases[name]=row
    return dict(order=order,quadrature_points=order**2,area_m2=float(a['w'].sum()),cases=cases,
        curved_eigenvalues=eigvalsh(a['curved']).tolist(),matrix_bytes=a['curved'].nbytes+a['planar'].nbytes)


def main():
    out=ROOT/'local_spherical_drag_reference.json'
    if out.exists():raise FileExistsError('Preserve prior reference')
    start=time.monotonic();source=json.loads(SOURCE.read_bytes())['payload'];c=source['config']
    X=np.array(c['nodal']['X']);faces=np.array(c['mesh']['faces']);triangle=X[faces[0]]
    beta=c['nodal']['moving_basal']['beta_pa_s_m']
    # Source traction scale, not chosen to make the outcome small or large.
    shell=c['shell'];h=float((triangle[0]@np.array(c['nodal']['frames'])[0,:,2]))
    qhi=max(x['live_q_hi'] for x in c['source_material_catalogue'] if x['face_index']==0)
    coeff=6*c['nodal']['column_mass'][0]/(c['nodal']['density']*np.linalg.det(triangle.T))
    depth=h*(-np.expm1(np.log1p(-coeff*qhi)/3))
    amplitude=float(shell['convective_traction_pa']*basal_coupling_fraction(depth/1000,shell['traction_coupling_depth_km']))
    tick=time.monotonic();low=run(triangle,beta,amplitude,6);cost=time.monotonic()-tick
    print(json.dumps(dict(benchmark_seconds=cost,planned='one order10 quadrature check, four modes maximum',expected_seconds=cost*4)),flush=True)
    high=run(triangle,beta,amplitude,10)
    result=dict(code_base='a47272d7b00574e784cf2433837f59ec6888c54b',scope='Local manufactured basal operator reference; same physical sphere, quadrature traction, force, moment and virtual-work load functional for both geometry representations',
        beta_pa_s_m=beta,source_transmitted_amplitude_pa=amplitude,source_face_depth_m=depth,benchmark_seconds=cost,low=low,high=high,seconds=time.monotonic()-start)
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
