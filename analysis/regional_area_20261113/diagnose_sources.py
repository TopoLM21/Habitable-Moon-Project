"""Read-only separation of material quadrature and receiver-boundary geometry."""
import json,math,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tectonics.mesh import build_icosphere
from tectonics.checkpoint import load_checkpoint
from tectonics.topology import PlateTopologyManager,PlateTopologyParameters
from tectonics.kinematics import classify_boundaries,BoundaryType
HERE=Path(__file__).resolve().parent;OUT=ROOT/'outputs_v131_regional_area';R=5287.


def geometry(name,sub):
    cp=load_checkpoint(OUT/name/'restart',PlateTopologyManager(PlateTopologyParameters()))
    mesh=build_icosphere(sub);bounds=classify_boundaries(mesh,cp.system,R,4.,1.)
    lengths=np.array([R*math.atan2(np.linalg.norm(np.cross(mesh.vertices[b.vertex_u],mesh.vertices[b.vertex_v])),
                                   mesh.vertices[b.vertex_u]@mesh.vertices[b.vertex_v]) for b in bounds])
    normal=np.array([b.normal_rate_km_per_myr for b in bounds]);kind=np.array([int(b.boundary_type) for b in bounds])
    owner=cp.state.cell_plate;adj=[[] for _ in owner]
    for a,b,_,_ in mesh.shared_edges:
        if owner[a]==owner[b]:adj[a].append(b);adj[b].append(a)
    unseen=set(range(len(owner)));components=[]
    while unseen:
        first=unseen.pop();stack=[first];count=1
        while stack:
            for neighbor in adj[stack.pop()]:
                if neighbor in unseen:unseen.remove(neighbor);stack.append(neighbor);count+=1
        components.append((int(owner[first]),count))
    return dict(case=name,time_myr=cp.state.time_myr,receiver_cells=len(owner),boundary_segments=len(bounds),
                boundary_length_km=math.fsum(lengths),
                convergent_length_km=math.fsum(lengths[kind==int(BoundaryType.CONVERGENT)]),
                convergence_line_flux_km2_per_myr=math.fsum(lengths[kind==int(BoundaryType.CONVERGENT)]*np.maximum(-normal[kind==int(BoundaryType.CONVERGENT)],0)),
                mean_angular_speed_deg_per_myr=math.degrees(np.mean([abs(p.angular_speed_rad_per_myr) for p in cp.system.plates])),
                connected_components=len(components),components_per_owner={str(i):int(sum(p==i for p,n in components)) for i in sorted(set(owner))})


if __name__=='__main__':
    geom=[];hist=[]
    for sub,name in [(1,'area20_g1'),(2,'area20'),(3,'area20_g3')]:
        geom.extend([geometry(f'initial_same_g{sub}',sub),geometry(name,sub)])
        m=json.loads((OUT/name/'restart/meta.json').read_text())
        hist.append(dict(case=name,source_history=[{key:x[key] for key in ('time_myr','source_integral_km2','deposited_source_km2','source_deposition_residual_km2','born_felsic_km3')} for x in m['regional_material']['history']]))
    timing=[]
    for name in ('area20','area20_dt2','area20_dt1'):
        m=json.loads((OUT/name/'restart/meta.json').read_text());h=m['regional_material']['history']
        first=next(x for x in h if x['source_integral_km2']>0);dt=h[0]['time_myr']
        timing.append(dict(case=name,outer_dt_myr=dt,first_source_interval_start_myr=first['time_myr']-dt,
                           first_source_interval_end_myr=first['time_myr'],minimum_active_age_myr=8.,
                           first_interval_felsic_km3=first['born_felsic_km3']))
    result=dict(geometry=geom,source_history=hist,time_gate=timing,
                scope='Observed discrete boundary/source change, not a complete causal attribution of mechanical evolution. Endpoint gating remains in inherited operator.')
    with (HERE/'source_diagnosis_verified.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
