"""Bounded fixed-coefficient stage elimination, no new scheme/physical runner."""
import numpy as np
from scipy.linalg import lu_factor,lu_solve
from .core import A,B,C,GAMMA
LD=np.longdouble


def refined_solve(matrix,rhs):
    """Same linear system, symmetric diagonal scaling and extended residuals."""
    matrix=np.asarray(matrix,LD);rhs=np.asarray(rhs,LD)
    if not np.isfinite(matrix).all() or not np.isfinite(rhs).all() or np.any(np.diag(matrix)==0):raise ValueError('INVALID_LINEAR_SYSTEM')
    scale=1/np.sqrt(np.abs(np.diag(matrix)))
    scaled=matrix*scale[:,None]*scale[None,:];right=rhs*scale[:,None]
    lu=lu_factor(np.array(scaled,float));x=np.asarray(lu_solve(lu,np.array(right,float)),LD)
    errors=[]
    for _ in range(9):
        residual=right-scaled@x
        den=np.abs(right)+np.abs(scaled)@np.abs(x)
        error=float(np.max(np.abs(residual)/np.maximum(den,np.finfo(LD).tiny)))
        if not np.isfinite(error):raise ValueError('NONFINITE_REFINEMENT')
        errors.append(error)
        if error<32*np.finfo(LD).eps:break
        x+=np.asarray(lu_solve(lu,np.array(residual,float)),LD)
    if errors[-1]>=32*np.finfo(LD).eps:raise ValueError('REFINEMENT_LIMIT')
    return scale[:,None]*x,dict(scaled_condition=float(np.linalg.cond(np.array(scaled,float))),backward_errors=errors)


def solve_stage(model,base,t,s,scale):
    """One-node, one-block, smooth fixed-phase, state-independent ports ONLY.

    Caller supplies frozen Q,F explicitly via the saved manufactured provider.
    This verification helper is not a general replacement for model.rhs.
    """
    if model.nt!=1 or len(model.blocks)!=1 or model.edges or model.regions!=('solid',):raise ValueError('DIAGNOSTIC_SCOPE')
    m=model.blocks[0];nv=model.nv
    base=np.asarray(base,LD);cap=LD(model.mass[0])*LD(model.cp[0]);s=LD(s)
    E0=base[0];v0=base[1:1+nv];e0=base[1+nv:]
    Q,F=model.ports(float(t),model.thermal(np.array([E0],float)),np.array(v0,float),[np.array(e0,float)])
    Q=LD(Q[0]);F=np.asarray(F,LD)
    M,D,H,Bm,U=map(lambda x:np.asarray(x,LD),(model.M,model.D,m.H,m.B,m.expansion))
    g=LD(m.relaxation);a=1/(1+s*g);BH=Bm.T@H
    matrix=M+s*D+s*s*a*(BH@Bm)
    rhs=np.column_stack((M@v0+s*F-s*a*(BH@e0),s*a*(BH@U)))
    vv,linear=refined_solve(matrix,rhs)
    ec=a*(e0+s*Bm@vv[:,0]);et=a*(s*Bm@vv[:,1]-U)
    # Solve the exact reduced quadratic stably, using x=deltaT (not huge E).
    qa=-(et@H@U)-s*g*(et@H@et)
    qb=cap-(ec@H@U)-2*s*g*(ec@H@et)
    qc=-s*(Q+g*(ec@H@ec))
    disc=qb*qb-4*qa*qc
    if qb<=0 or disc<0:raise ValueError('REDUCED_THERMAL_BRANCH')
    x=-2*qc/(qb+np.sqrt(disc))
    Y=np.concatenate(([E0+cap*x],vv[:,0]+vv[:,1]*x,ec+et*x))
    model.guard(np.array(Y,float))
    # Audit the original residual, with exactly its old normalization/threshold.
    residual=model.implicit_residual(float(t),np.array(Y,float),np.array(base,float),float(s),scale)
    return np.array(Y,float),dict(linear=linear,original_scaled_residual=float(np.max(np.abs(residual))),
        thermal_residual=float(abs(residual[0])),momentum_residual=float(np.max(abs(residual[1:1+nv]))),
        memory_residual=float(np.max(abs(residual[1+nv:]))))


def step(model,t,y,h,scale):
    stages=[];rates=[];details=[]
    for i in range(2):
        base=np.asarray(y,LD).copy()
        for j in range(i):base+=LD(h)*LD(A[i,j])*np.asarray(rates[j],LD)
        Y,detail=solve_stage(model,base,t+C[i]*h,GAMMA*h,scale)
        if detail['original_scaled_residual']>2e-10:raise ValueError('ORIGINAL_STAGE_TOLERANCE: '+str(detail))
        stages.append(Y);rates.append(np.array((np.asarray(Y,LD)-base)/(LD(GAMMA)*LD(h)),float));details.append(detail)
    return dict(after=stages[-1],stages=np.array(stages),rates=np.array(rates),time=t,dt=h,details=details)
