"""Exact frozen-coefficient Maxwell response to a constant increment rate."""
import math
import numpy as np


def weights(z):
    z=float(z)
    if not math.isfinite(z) or z<0:raise ValueError('Nonnegative dt/tau required')
    if z<1e-3:
        b=sum((-z)**k/math.factorial(k+1) for k in range(12))
        c=sum((-z)**k/math.factorial(k+2) for k in range(12))
        A=sum((-2*z)**k/math.factorial(k+1) for k in range(12))
        B=sum((-1)**(k+1)*(1-2**(k+1))*z**k/math.factorial(k+2) for k in range(12))
        C=sum((-1)**k*(2**(k+2)-2)*z**k/math.factorial(k+3) for k in range(12))
    else:
        b=-math.expm1(-z)/z;c=(1-b)/z;A=-math.expm1(-2*z)/(2*z)
        B=(b-A)/z;C=(1-2*b+A)/z**2
    return math.exp(-z),b,c,A,B,C


def update(old,increment,H,z):
    """Return endpoint, time mean, and exact dissipated joules for de/ds=f-z e.

    H is the existing energy metric; s=t/dt, f is total strain minus thermal
    eigenstrain increment. No new constitutive coefficients or physical solve.
    """
    old=np.asarray(old,dtype=float);f=np.asarray(increment,dtype=float);H=np.asarray(H,dtype=float)
    if old.ndim!=1 or f.shape!=old.shape or H.shape!=(len(old),len(old)) or not all(np.isfinite(x).all() for x in (old,f,H)):raise ValueError('Finite material fields')
    if not np.allclose(H,H.T,rtol=1e-13,atol=0) or np.linalg.eigvalsh(H).min()<=0:raise ValueError('Positive energy metric')
    a,b,c,A,B,C=weights(z);end=a*old+b*f;mean=b*old+c*f
    shifted=old+(B/A)*f
    heat=float(z*(A*(shifted@H@shifted)+(C-B*B/A)*(f@H@f)))
    if heat<0:raise ValueError('Negative exact physical heat')
    return dict(endpoint=end,mean=mean,heat_j=heat,elastic_before_j=float(.5*old@H@old),elastic_after_j=float(.5*end@H@end),forcing_work_j=float(mean@H@f))
