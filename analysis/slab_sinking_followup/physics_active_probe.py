from pathlib import Path
import numpy as np
z=np.load(Path(__file__).with_name('physics_support_failure.npz'))
E,v,c=(z[k] for k in ('equality','value','candidate'))
s=c>2e-12*max(np.linalg.norm(c),1.); current=np.where(s,c,0.)
for i in range(100):
 p=np.zeros_like(c); p[s]=np.linalg.lstsq(E[:,s],v,rcond=2e-12)[0]
 negative=s&(p<0.)
 if negative.any():
  ratios=np.full_like(c,np.inf); ratios[negative]=current[negative]/(current[negative]-p[negative]); idx=np.argmin(ratios); alpha=ratios[idx]
  current+=alpha*(p-current); current[idx]=0.;s[idx]=False
  print(i,'blocked',idx,alpha,'minp',p.min());continue
 current=p; y=np.linalg.lstsq(E[:,s].T,p[s],rcond=2e-12)[0]; grad=p-E.T@y
 tol=2e-10*max(np.linalg.norm(p),np.linalg.norm(E.T@y),1.)
 if grad[~s].min(initial=0)<-tol:
  idx=np.argmin(np.where(s,np.inf,grad));s[idx]=True;print(i,'add',idx);continue
 print('done',i,np.flatnonzero(s),current[s],np.linalg.norm(E@current-v),'mingrad',grad[~s].min(initial=0));break
