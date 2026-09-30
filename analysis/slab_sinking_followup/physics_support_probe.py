from pathlib import Path
import numpy as np
z=np.load(Path(__file__).with_name('physics_support_failure.npz'))
E,v,c=(z[k] for k in ('equality','value','candidate'))
for tol in (2e-12,2e-11,2e-10,1e-8):
 s=c>tol; p=np.linalg.lstsq(E[:,s],v,rcond=2e-12)[0]; y=np.linalg.lstsq(E[:,s].T,p,rcond=2e-12)[0]; grad=np.where(s,c,0)-E.T@y
 print(tol,np.flatnonzero(s),p,'rank',np.linalg.svd(E[:,s],compute_uv=False),'res',np.linalg.norm(E[:,s]@p-v),'dual',grad[~s].min(initial=0),'activegrad',np.linalg.norm(p-E[:,s].T@y))
