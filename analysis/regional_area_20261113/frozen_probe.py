"""Independent steady-strip mature-area check; keep quadrature error visible."""
from dataclasses import replace
import json,math,sys,time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'tests')]
from test_regional_area import inventory,source,R
from tectonics.continental import ContinentalCycleParameters
HERE=Path(__file__).resolve().parent
params=ContinentalCycleParameters(arc_maturation_rate_per_myr=.018,juvenile_threshold=.75,
    continental_arc_thickening_km_per_myr=.045,analytic_maturation=True)
maximum=.7*.018*params.arc_potential_decay_myr*(-math.expm1(-100/params.arc_potential_decay_myr))
width=140*math.sqrt(2*math.log(maximum/.75))
exact=R*R*(math.sin(min(425.,105+width)/R)-math.sin(max(0.,105-width)/R))
rows=[]
for order in (16,32):
    for orientation in ('aligned','oblique'):
        s=source()
        if orientation=='oblique':
            axis=np.array([1.,2.,3.]);axis/=np.linalg.norm(axis)
            r=Rotation.from_rotvec(axis*math.radians(43))
            s=replace(s,start=r.apply(s.start),end=r.apply(s.end),toward=r.apply(s.toward))
        for edge in (500.,250.,125.,62.5):
            start=time.perf_counter();inv=inventory(edge=edge)
            inv.area_params=replace(inv.area_params,quadrature_order=order)
            refine=inv.refine([s],[],R,140.,320.)
            field,diag=inv.integrated_field([s],[],R,140.,320.)
            step=inv.leaves.react(100.,field,params)
            row=dict(order=order,orientation=orientation,max_edge_km=edge,leaves=len(inv.roots),
                mature_area_km2=step['matured_area_km2'],mature_area_relative_error=step['matured_area_km2']/exact-1,
                first_event_myr=step['first_event_myr'],source=diag,refinement=refine,wall_s=time.perf_counter()-start)
            rows.append(row);print(json.dumps(row),flush=True)
with (HERE/'frozen_probe.json').open('x') as f:json.dump(dict(exact_area_km2=exact,rows=rows),f,indent=2)
