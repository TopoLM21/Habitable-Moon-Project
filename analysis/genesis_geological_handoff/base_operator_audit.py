"""Finite analytic audit and three passive single-step spatial probes."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from tectonics.mesh import build_icosphere
from analysis.genesis_volume_continuation_validation import independent_audit as independent
from .integrated_vector_operator import blocks, first_step, frame_for
from .edge_spatial_audit import AuditGrid, ROOT, metric
from .driven_shell import CONFIG
from .nodal_retirement_check import SOURCE

R = CONFIG['radius_km']*1000
INNER = 1-CONFIG['thickness_km']/CONFIG['radius_km']
EPS = 1e-6
SOURCE_SHA256 = '94fbbb3cb88a15147d33ab04c8f791470dd92e3d73cf8cba4afc0f361cd583fc'
CODE_BASE = '510aef4bbf33cba9ee7d157a9706d4a09660a9ef'


def field(x, degree):
    """u_i=eps*R^(1-p)*x_j*x_k/|x|^(2-p), p=0 or 1."""
    r = np.linalg.norm(x,axis=-1)
    exponent = 2-degree
    pairs = ((0,1),(1,2),(2,0))
    u = np.stack([x[...,i]*x[...,j]/r**exponent for i,j in pairs],axis=-1)*EPS*R**(1-degree)
    gradient = np.zeros(x.shape[:-1]+(3,3))
    for k,(i,j) in enumerate(pairs):
        for b in range(3):
            gradient[...,k,b] = ((int(i==b)*x[...,j]+int(j==b)*x[...,i])/r**exponent
                 -exponent*x[...,i]*x[...,j]*x[...,b]/r**(exponent+2))*EPS*R**(1-degree)
    return u,gradient


def rotation():
    axis=np.array([1.,2.,3.]);axis/=np.linalg.norm(axis);theta=.371
    W=np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
    return np.eye(3)+np.sin(theta)*W+(1-np.cos(theta))*(W@W)


def analytic(level, rotate_grid=False):
    mesh=build_icosphere(level);Q=rotation() if rotate_grid else np.eye(3)
    triangles=(mesh.vertices@Q.T)[mesh.faces]*R
    N,wa=independent.angular_rule(4)
    C=independent.isotropic_matrix(CONFIG['young_pa'],CONFIG['poisson'],0.,.02)
    errors={p:[0.,0.,0.,0.] for p in (0,1)}
    affine_error=0.;rigid_error=0.
    for tri in triangles:
        B,w,points,frame=blocks(tri,INNER,N,wa)
        meanB=np.einsum('a,raij->rij',wa,B)
        G=EPS*np.array([[1.,.4,.2],[-.1,-.5,.3],[.2,.1,.7]])
        affine=np.concatenate((tri@G.T,tri@G.T),axis=1).ravel()
        exact=independent.engineering_gradient(G,frame)
        affine_error=max(affine_error,float(abs(B@affine-exact).max()/EPS))
        omega=EPS*np.array([.3,.7,-.2]);turn=np.cross(omega,tri)
        rigid=np.concatenate((turn,turn),axis=1).ravel()
        translation=np.tile(np.r_[np.array([1.,2.,3.]),np.zeros(3)],3)
        rigid_error=max(rigid_error,float(abs(B@rigid).max()/EPS),float(abs(B@translation).max()*R))
        for p in (0,1):
            nodal,_=field(tri,p);dof=np.concatenate((nodal,p*nodal),axis=1).ravel()
            _,gradient=field(points,p);truth=independent.engineering_gradient(gradient,frame)
            numerical=B@dof;difference=numerical-truth
            errors[p][0]+=np.einsum('rai,ij,raj,ra->',difference,C,difference,w)
            errors[p][1]+=np.einsum('rai,ij,raj,ra->',truth,C,truth,w)
            errors[p][2]+=np.einsum('rai,ij,raj,ra->',numerical,C,numerical,w)
            mean=meanB@dof
            errors[p][3]+=np.einsum('ri,ij,rj,r->',mean,C,mean,w.sum(axis=1))
    # Representative patch: reduced null vector with positive full energy.
    tri=triangles[0];B,w,_,frame=blocks(tri,INNER,N,wa);mean=np.einsum('a,raij->rij',wa,B)
    U,s,V=np.linalg.svd(mean.reshape(-1,18),full_matrices=True)
    null=V[11:].T;full=B.reshape(-1,18)
    eigen,vec=np.linalg.eigh(null.T@full.T@full@null);mode=null@vec[:,-1]
    C=independent.isotropic_matrix(CONFIG['young_pa'],CONFIG['poisson'],0.,.02)
    mode_full=np.einsum('raij,j->rai',B,mode);mode_mean=mean@mode
    energy_full=np.einsum('rai,ij,raj,ra->',mode_full,C,mode_full,w)
    energy_mean=np.einsum('ri,ij,rj,r->',mode_mean,C,mode_mean,w.sum(axis=1))
    return dict(faces=mesh.cell_count,rotated_grid=rotate_grid,affine_strain_relative_max=affine_error,
        rigid_strain_scaled_max=rigid_error,fields={str(p):dict(energy_norm_strain_error=float(np.sqrt(v[0]/v[1])),
            reduced_energy_loss_fraction=float(1-v[3]/v[2])) for p,v in errors.items()},
        local_rank_reduced=int(np.linalg.matrix_rank(mean.reshape(-1,18),tol=np.linalg.norm(mean)*1e-10)),
        local_rank_full=int(np.linalg.matrix_rank(full,tol=np.linalg.norm(full)*1e-10)),
        spurious_mode_reduced_to_full_energy=float(energy_mean/energy_full))


def main():
    assert hashlib.sha256(Path(SOURCE).read_bytes()).hexdigest()==SOURCE_SHA256
    start=time.monotonic();deadline=start+120
    bench=time.monotonic();first=analytic(1);elapsed=time.monotonic()-bench
    print(json.dumps(dict(benchmark_80_faces_seconds=elapsed,estimated_80_320_1280_plus_rotated_seconds=elapsed*37,
        cap_seconds=120)),flush=True)
    if elapsed*37>60:raise RuntimeError('Analytic audit estimate exceeds60seconds')
    analytic_rows=[first,analytic(2),analytic(3),analytic(2,True)]
    coeff=np.array(json.loads((ROOT/'edge_spatial_control.json').read_text())['coefficients'])
    probes=[];profiles={}
    for level in (2,3,4):
        if time.monotonic()>deadline:raise TimeoutError('120s cap')
        tick=time.monotonic();g=AuditGrid(level,False,coeff)
        for integrated in (False,True):
            if level==4 and not integrated:continue
            stress,row=first_step(g,integrated=integrated)
            profiles[(level,integrated)]=stress
            probes.append(dict(level=level,integrated=integrated,**row))
        print(json.dumps(dict(level=level,faces=g.a.nf,setup_and_solve_seconds=time.monotonic()-tick)),flush=True)
    comparisons={}
    for level in (2,3):
        fine=build_icosphere(level+1)
        comparisons[f'integrated_{level}_{level+1}']=metric(profiles[(level,True)],profiles[(level+1,True)],fine.areas_unit_sphere)
    for level in (2,3):
        area=build_icosphere(level).areas_unit_sphere;delta=profiles[(level,True)]-profiles[(level,False)]
        comparisons[f'quadrature_change_{level}']=float(np.sqrt(np.einsum('f,fij,fij->',area,delta,delta)/np.einsum('f,fij,fij->',area,profiles[(level,True)],profiles[(level,True)])))
    result=dict(code_base=CODE_BASE,source_sha256=SOURCE_SHA256,
        analytic=analytic_rows,probes=probes,stress_comparisons=comparisons,seconds=time.monotonic()-start,
        scope='Analytic kinematic fields, then at most five passive first-step solves; no damage advancement')
    (ROOT/'base_operator_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
