"""Located joule exchanges; no latent, tidal, drag or numerical heating."""
import numpy as np
from .genesis_volume_loading import coefficient_inner_product


def local_work(loading, e0, e1):
    inner = lambda a,b,c,d: coefficient_inner_product(a,b,c,d,loading.elasticity_pa,loading.radial_moments)
    maxwell = loading.ratio*inner(e0,e1,e0,e1)
    thermal = -inner(e0,e1,loading.thermal0,np.zeros_like(e0))
    if not np.isfinite(maxwell).all() or not np.isfinite(thermal).all() or np.any(maxwell<0):
        raise ValueError('Invalid local joule exchange')
    return maxwell, thermal


def apply_exchange(base_energy, cell_ids, material_ids, maxwell_j, thermal_work_j):
    """Q>=0: elastic -> same material heat. W signed: heat -> elastic.

    Always apply once to the original conduction endpoint, never to a previous
    corrector. Mechanical solver already includes -Q+W; do not debit it twice.
    """
    E=np.asarray(base_energy,dtype=float).copy()
    ids=list(material_ids); Q=np.asarray(maxwell_j); W=np.asarray(thermal_work_j)
    present=[x for x in cell_ids if x is not None]
    if len(ids)!=len(set(ids)) or len(present)!=len(set(present)) or set(ids)!=set(present):
        raise ValueError('Exact material coverage required')
    if Q.shape!=(len(ids),) or W.shape!=Q.shape or not np.isfinite(Q).all() or not np.isfinite(W).all() or np.any(Q<0):
        raise ValueError('Invalid exchange arrays')
    if len(E)!=len(cell_ids) or not np.isfinite(E).all() or np.any(E<0):raise ValueError('Energy coverage')
    byid={cid:i for i,cid in enumerate(cell_ids) if cid is not None};entries=[]
    for cid,q,w in zip(ids,Q,W):
        i=byid[cid];E[i]+=q-w
        entries.append(dict(material_id=int(cid),cutcell=i,maxwell_heat_j=float(q),thermal_work_to_mechanics_j=float(w),
                            thermal_entry_j=float(q-w),mechanical_entry_j=float(w-q)))
    if np.any(E<0) or not np.isfinite(E).all():raise ValueError('Thermal reservoir exhausted')
    return E,entries
