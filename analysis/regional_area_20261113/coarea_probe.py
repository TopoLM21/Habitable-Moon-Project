"""Matched nonlinear area refinement; independent continuum reference retained."""
from dataclasses import replace
import importlib.util,json,math,sys,time,subprocess
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'tests')]
from test_regional_area import inventory,source,R
from tectonics.continental import ContinentalCycleParameters
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('reference',ROOT/'analysis/regional_inventory_20261113/operator_checks.py')
ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
params=ref.PARAMS;exact=ref.continuum(params);rows=[]
for order in (8,16):
    for orientation in ('aligned','oblique'):
        s=source()
        if orientation=='oblique':
            axis=np.array([1.,2.,3.]);axis/=np.linalg.norm(axis);rot=Rotation.from_rotvec(axis*math.radians(43))
            s=replace(s,start=rot.apply(s.start),end=rot.apply(s.end),toward=rot.apply(s.toward))
        for edge in (250.,125.,62.5,31.25):
            start=time.perf_counter();inv=inventory(edge=edge)
            inv.area_params=replace(inv.area_params,strip_rule='coarea',quadrature_order=order)
            refinement=inv.refine([s],[],R,140.,320.)
            try:
                field,diag=inv.integrated_field([s],[],R,140.,320.)
                step=inv.leaves.react(100.,field,params)
                row=dict(order=order,orientation=orientation,max_edge_km=edge,leaves=len(inv.roots),
                    mature_area_km2=step['matured_area_km2'],area_relative_error=step['matured_area_km2']/exact['area_km2']-1,
                    felsic_volume_km3=step['born_felsic_km3'],volume_relative_error=step['born_felsic_km3']/exact['volume_km3']-1,
                    first_event_myr=step['first_event_myr'],source=diag,refinement=refinement,wall_s=time.perf_counter()-start)
            except ValueError as exc:row=dict(order=order,orientation=orientation,max_edge_km=edge,failure=str(exc),wall_s=time.perf_counter()-start)
            rows.append(row);print(json.dumps(row),flush=True)
with (HERE/'coarea_probe.json').open('x') as f:json.dump(dict(exact_continuum=exact,rows=rows,
    source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
    scope='new coarea implementation in working tree; file hashes preserved with final evidence'),f,indent=2)
