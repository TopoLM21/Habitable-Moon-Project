from pathlib import Path
import sys, json
import numpy as np
root=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(root))
import tectonics.young_slab_constraints as m
path=root/'analysis/slab_sinking_followup/runs/ordered_sub5_dt1/elapsed_0050.solver_failure.npz'
data=np.load(path)
old=m._polish_reaction_support

def capture(E,v,c):
    np.savez(Path(__file__).with_suffix('.npz'), equality=E,value=v,candidate=c)
    mask=c>2e-12*max(np.linalg.norm(c),1.)
    sol=np.linalg.lstsq(E[:,mask],v,rcond=2e-12)[0]
    print(json.dumps(dict(shape=E.shape, value=v.tolist(),candidate=c.tolist(), mask=np.flatnonzero(mask).tolist(),polished=sol.tolist(),residual=float(np.linalg.norm(E@c-v)), singular=np.linalg.svd(E[:,mask],compute_uv=False).tolist()),indent=2))
    return old(E,v,c)
m._polish_reaction_support=capture
m.solve_no_eduction(data['drag_nm_s'],data['driving_torque_nm'],data['feed_matrix_m'],reaction_weights=data['reaction_weights'])
