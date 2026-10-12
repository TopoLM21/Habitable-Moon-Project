"""Bounded manufactured uncut stiffness control; no Genesis advancement."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import time
from types import SimpleNamespace as NS
import numpy as np
from tectonics.genesis_volume_operator import _construct_operator, isotropic_elasticity

TOL = 1e-11
E, NU = 60e9, .25


def physical_oracle(X, inner, outer):
    """Direct physical Jacobian and tensor Hooke law; no production B/C/moments."""
    z, w = np.polynomial.legendre.leggauss(4)
    z, w = (z+1)/2, w/2
    mu, lame = E/(2*(1+NU)), E*NU/((1+NU)*(1-2*NU))
    K = np.zeros((18,18)); volume = 0.
    dn = np.array([[-1.,1.,0.],[-1.,0.,1.]])
    for r,wr in zip(z,w):
        for t,wt in zip(z,w):
            s=(1-r)*t; N=np.array([1-r-s,r,s])
            for b,wb in zip(z,w):
                lam=inner+(outer-inner)*b
                J=np.column_stack((lam*dn[0]@X,lam*dn[1]@X,N@X))
                det=np.linalg.det(J)
                if det<=0:raise ValueError('Manufactured cell orientation')
                weight=wr*wt*(1-r)*wb*(outer-inner)*det
                inverse=np.linalg.inv(J); strains=[]
                for vertex in range(3):
                    for director in (False,True):
                        param=np.r_[dn[:,vertex]*(lam-1 if director else 1),N[vertex] if director else 0]
                        gradient=param@inverse
                        for axis in np.eye(3):
                            G=np.outer(axis,gradient);strains.append((G+G.T)/2)
                strains=np.array(strains);trace=np.trace(strains,axis1=1,axis2=2)
                K+=weight*(lame*np.outer(trace,trace)+2*mu*np.einsum('aij,bij->ab',strains,strains))
                volume+=weight
    return K,volume


def cases():
    Q,_=np.linalg.qr(np.array([[1.,2.,3.],[-2.,4.,1.],[3.,1.,-2.]]))
    if np.linalg.det(Q)<0:Q[:,0]*=-1
    return [('thick',np.array([[0.,0.,10.],[2.,0.,10.],[0.,2.,10.]]),np.eye(3),.4,1.),
            ('rotated_skew',np.array([[-2.,1.,8.],[3.,1.,8.],[-1.,4.,8.]])@Q.T,Q,.7,.93),
            ('planet_scale_thin',1e6*np.array([[0.,0.,1.],[.2,0.,1.],[.03,.15,1.]]),np.eye(3),.999999999,1.)]


def check_case(name,X,Q,inner,outer):
    # Stable exact geometric volume, independent of production radial moments.
    V=np.linalg.det(X)*(outer-inner)*(outer*outer+outer*inner+inner*inner)/6
    cohort=NS(id=1,face_index=0,volume_m3=V,top_triangle_m=X,outer_scale=outer,inner_scale=inner)
    geometry=NS(cohorts=[cohort],faces=[[0,1,2]],vertices_m=X,source_checkpoint_sha256='manufactured',source_identity='manufactured',time_myr=0.)
    operator=_construct_operator(geometry,isotropic_elasticity(E,NU)[None],Q[None])
    K=operator.K.toarray(); reference,oracle_volume=physical_oracle(X,inner,outer)
    norm=np.linalg.norm(reference)
    errors={'stiffness_frobenius':float(np.linalg.norm(K-reference)/norm),
            'volume':float(abs(oracle_volume-V)/V), 'symmetry':float(np.linalg.norm(K-K.T)/norm)}
    rigid=[]
    for axis in np.eye(3):
        rigid.append(np.column_stack((np.tile(axis,(3,1)),np.zeros((3,3)))).ravel())
        rotation=np.cross(axis,X);rigid.append(np.column_stack((rotation,rotation)).ravel())
    errors['rigid_nullspace']=max(float(np.linalg.norm(K@q)/(norm*np.linalg.norm(q))) for q in rigid)
    affine=[]
    for i,j in [(0,0),(1,1),(2,2),(0,1),(0,2),(1,2)]:
        G=np.zeros((3,3));G[i,j]=G[j,i]=1 if i==j else .5
        q=np.column_stack((X@G.T,X@G.T)).ravel()
        mu=E/(2*(1+NU));lame=E*NU/((1+NU)*(1-2*NU))
        expected=V*(mu*np.sum(G*G)+.5*lame*np.trace(G)**2)
        affine.append(float(abs(.5*q@K@q-expected)/expected))
    errors['six_affine_energies']=max(affine)
    # Full matrix agreement controls all 18 directions, not one chosen displacement.
    passed=all(np.isfinite(v) and v<=TOL for v in errors.values())
    return dict(case=name,status='PASS' if passed else 'FAIL',errors=errors,nodes=64,ndof=18)


def run():
    results=[check_case(*case) for case in cases()]
    return dict(status='PASS' if all(r['status']=='PASS' for r in results) else 'FAIL',
                criterion_relative=TOL,cases=results,total_oracle_nodes=192,new_Genesis_steps=0,
                scope='Uncut element stiffness and affine consistency only; no mesh convergence or thermal closure')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Output exists')
    resource.setrlimit(resource.RLIMIT_AS,(1024**3,1024**3))
    import signal
    def stop(*args):raise TimeoutError('60-second control budget')
    signal.signal(signal.SIGALRM,stop);signal.alarm(60)
    start=time.monotonic();report=run()
    report.update(seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
      sources={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in [__file__,'tectonics/genesis_volume_operator.py']})
    with a.output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(report,indent=2));signal.alarm(0)
    return 0 if report['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
