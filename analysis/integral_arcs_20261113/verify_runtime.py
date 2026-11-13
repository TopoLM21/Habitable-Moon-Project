"""Bounded same-initial-geometry checks; all outputs are new, never overwritten.

Run `prepare` once after initial_g1's zero-time run; then named cases or `suite`.
All cases run sequentially with the inherited 120 s process ceiling. The reused
harness is committed evidence from the preceding stage, not a physics adapter.
"""
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tectonics.integral_arcs import SphereReceiver
from tectonics.mesh import build_icosphere

HERE=Path(__file__).resolve().parent;OUT=ROOT/'outputs_v131_integral_arcs'
spec=importlib.util.spec_from_file_location('previous_checks',ROOT/'analysis/reconstruction_exchange_20261113/verify_runtime.py')
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
h.HERE=HERE;h.OUT=OUT;h.BASE=h.yaml.safe_load((HERE/'control.yaml').read_text())


def prepare():
    original=OUT/'initial_g1/restart'
    meta=json.loads((original/'meta.json').read_text())
    assert meta['time_myr']==0 and not meta['arc_rows'] and not meta['events']
    with np.load(original/'state.npz') as f:arrays={k:f[k] for k in f.files}
    coarse=build_icosphere(1);r=float(h.BASE['moon']['radius_km']);ca=coarse.physical_cell_areas_km2(r)
    extensive={k for k in arrays if k.endswith('_volume_km3')}|{'oceanic_age_moment_km3_myr'}
    results=[]
    for sub in (1,2,3):
        mesh=build_icosphere(sub);areas=mesh.physical_cell_areas_km2(r)
        parent=np.repeat(np.arange(80),4**(sub-1))
        assert np.array_equal(SphereReceiver(coarse).locate(mesh.centroids),parent)
        area_error=np.max(np.abs(np.bincount(parent,weights=areas)-ca))
        assert area_error<1e-7
        children={}
        for key,x in arrays.items():
            y=x[parent].copy() if x.shape[0]==80 else x.copy()
            if key in extensive:y*=areas/ca[parent]
            children[key]=y
        inventory_error={k:math.fsum(children[k])-math.fsum(arrays[k]) for k in extensive}
        for k,e in inventory_error.items():assert abs(e)<=8*math.ulp(max(abs(math.fsum(arrays[k])),1.))
        boundary_length=math.fsum(r*math.atan2(np.linalg.norm(np.cross(mesh.vertices[u],mesh.vertices[v])),mesh.vertices[u]@mesh.vertices[v])
            for a,b,u,v in mesh.shared_edges if children['state_cell_plate'][a]!=children['state_cell_plate'][b])
        m=deepcopy(meta)
        m['arc_integration']['initialization']={
            'kind':'new zero-time material refinement; not an evolved history migration',
            'source_state_sha256':hashlib.sha256((original/'state.npz').read_bytes()).hexdigest(),
            'base_cells':80,'cellwise_fields':'piecewise constant parent; extensive fields split by exact spherical area'}
        dest=OUT/f'initial_same_g{sub}'/'restart';dest.mkdir(parents=True,exist_ok=False)
        np.savez_compressed(dest/'state.npz',**children)
        (dest/'meta.json').write_text(json.dumps(m,indent=2)+'\n')
        results.append({'subdivisions':sub,'cells':mesh.cell_count,'max_parent_area_error_km2':area_error,
                        'extensive_sum_errors':inventory_error,'boundary_length_km':boundary_length})
    assert max(x['boundary_length_km'] for x in results)-min(x['boundary_length_km'] for x in results)<1e-7
    (HERE/'initial_refinement.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results),flush=True)


CASES={
    'same_g2_dt4':(2,4.,100.,'initial_same_g2'),
    'same_g2_dt2':(2,2.,100.,'initial_same_g2'),
    'same_g2_dt1':(2,1.,100.,'initial_same_g2'),
    'same_g1_dt2':(1,2.,100.,'initial_same_g1'),
    'same_g3_dt2':(3,2.,100.,'initial_same_g3'),
    'same_first48':(2,4.,48.,'initial_same_g2'),
    'same_restart100':(2,4.,100.,'same_first48'),
}
FINAL_CASES={name.replace('same_','final_'):(sub,dt,end,resume.replace('same_first','final_first'))
             for name,(sub,dt,end,resume) in CASES.items()}

if __name__=='__main__':
    arg=sys.argv[1]
    if arg=='prepare':prepare()
    elif arg=='suite':
        for name,args in CASES.items():h.run_case(name,*args)
        h.compare(OUT/'same_g2_dt4/restart',OUT/'same_restart100/restart','restart_comparison')
    elif arg=='final_suite':
        for name,args in FINAL_CASES.items():h.run_case(name,*args)
        h.compare(OUT/'final_g2_dt4/restart',OUT/'final_restart100/restart','final_restart_comparison')
    else:h.run_case(arg,*(CASES|FINAL_CASES)[arg])
