"""Two-stage stiffly accurate SDIRK2; fixed coefficients are not production defaults."""
import numpy as np
from scipy.optimize import root

GAMMA=1-1/np.sqrt(2.)
A=np.array([[GAMMA,0.],[1-GAMMA,GAMMA]])
B=A[1].copy()
C=A.sum(axis=1)
ENERGY_MATRIX=B[:,None]*A+B[None,:]*A.T-np.outer(B,B)


def step(rhs,t,y,h,scale,guard,*,implicit_residual=None):
    y=np.asarray(y,float);scale=np.asarray(scale,float)
    if not np.isfinite(h) or h<=0 or scale.shape!=y.shape or np.any(scale<=0) or not np.isfinite(scale).all():
        raise ValueError('STEP_SCALE')
    guard(y);stages=[];rates=[];residuals=[];evaluations=[]
    for i in range(2):
        base=y.copy()
        for j in range(i):base+=h*A[i,j]*rates[j]
        def residual(z):
            Y=scale*z
            if implicit_residual is not None:return implicit_residual(t+C[i]*h,Y,base,GAMMA*h,scale)
            return (Y-base)/scale-GAMMA*h*np.asarray(rhs(t+C[i]*h,Y))/scale
        guess=base/scale
        result=root(residual,guess,method='hybr',options={'xtol':1e-10,'maxfev':2000})
        defect=residual(result.x)
        if not np.isfinite(defect).all() or np.max(np.abs(defect))>2e-10:
            raise ValueError(f'STAGE_SOLVE_FAILED: stage={i} defect={np.max(np.abs(defect)):.6g} nfev={result.nfev} message={result.message}')
        Y=scale*result.x;guard(Y)
        stages.append(Y);rates.append((Y-base)/(GAMMA*h) if implicit_residual is not None else np.asarray(rhs(t+C[i]*h,Y)))
        residuals.append(float(np.max(np.abs(defect))));evaluations.append(result.nfev)
    return dict(after=stages[-1].copy(),stages=np.array(stages),rates=np.array(rates),
                time=t,dt=h,stage_residuals=residuals,evaluations=evaluations)
