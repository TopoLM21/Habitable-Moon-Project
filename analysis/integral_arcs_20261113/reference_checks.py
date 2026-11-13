"""Matched old-law comparison from the same NEW zero-time initial geometry.

Changing laws is permitted only here at initialization, never as an evolved
restart conversion. This compares two approximations, not an exact solution.
"""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import verify_runtime as v


if __name__=='__main__':
    src=v.OUT/'initial_same_g2/restart';dest=v.OUT/'initial_reference_g2/restart'
    dest.mkdir(parents=True,exist_ok=False)
    m=json.loads((src/'meta.json').read_text());assert m['time_myr']==0
    m.pop('arc_integration');m['version']='0.33-material-exchange-reconstruction'
    with np.load(src/'state.npz') as z:arrays={k:z[k] for k in z.files if k!='oceanic_age_moment_km3_myr'}
    np.savez_compressed(dest/'state.npz',**arrays)
    (dest/'meta.json').write_text(json.dumps(m,indent=2)+'\n')
    v.h.BASE=deepcopy(v.h.BASE)
    v.h.BASE['volcanic_arc']['integral_production']=False
    v.h.BASE['lithosphere']['integral_ocean_fluxes']=False
    for dt in (4.,2.,1.):v.h.run_case(f'reference_g2_dt{int(dt)}',2,dt,100.,'initial_reference_g2')
    v.h.run_case('legacy20',2,4.,20.,None,'legacy')
    v.h.compare(v.ROOT/'outputs_v131_reconstruction/legacy_original/restart',v.OUT/'legacy20/restart','legacy_comparison')
