"""Compare saved nested-grid runs on the common fine-grid sampling."""
import json,sys,hashlib
from pathlib import Path
import numpy as np
from tectonics.unified_geological_genesis import UnifiedGeologicalGenesis as U
from tectonics.dynamics import angular_velocity_vectors
from tectonics.topography import material_subgrid_surface_elevations,TopographyParameters
from analysis.geological_continuation.run import snap

def land(e):
    c=e.checkpoint;s=snap(e)
    f,o,t=material_subgrid_surface_elevations(e.mesh,c.state,c.topo,e.record['radius_km'],TopographyParameters())
    return (1-f)*(o>=s['sea_level_m'])+f*(t>=s['sea_level_m'])

def compare(root,output):
    root=Path(root);results={k:json.loads((root/k/'result.json').read_text()) for k in ('coarse','fine')}
    assert all(v['error'] is None and v['restart_exact'] for v in results.values())
    diagnostics={}
    for stage in ('initial','accepted'):
        c=U.load(root/'coarse'/stage);f=U.load(root/'fine'/stage)
        parent=np.arange(f.mesh.cell_count)//4;a=f.mesh.physical_cell_areas_km2(f.record['radius_km'])
        wc=angular_velocity_vectors(c.checkpoint.system);wf=angular_velocity_vectors(f.checkpoint.system)
        assert wc.shape==wf.shape
        velocity_delta=np.cross((wf-wc)[c.checkpoint.system.cell_plate[parent]],f.mesh.centroids*f.record['radius_km'])
        cs=snap(c);fs=snap(f)
        dry_c=c.checkpoint.topo.elevation_m>=cs['sea_level_m'];dry_f=f.checkpoint.topo.elevation_m>=fs['sea_level_m']
        diagnostics[stage]=dict(land_fraction_L1_area_km2=float(np.sum(a*abs(land(c)[parent]-land(f)))),
            mean_height_coast_disagreement_area_km2=float(a[dry_c[parent]!=dry_f].sum()),
            relief_RMS_difference_m=float(np.sqrt(np.average((c.checkpoint.topo.elevation_m[parent]-f.checkpoint.topo.elevation_m)**2,weights=a))),
            plate_owner_disagreement_area_km2=float(a[c.checkpoint.system.cell_plate[parent]!=f.checkpoint.system.cell_plate].sum()),
            velocity_RMS_difference_km_per_myr=float(np.sqrt(np.average(np.sum(velocity_delta**2,axis=1),weights=a))),
            per_plate_omega_difference_deg_per_myr=np.rad2deg(np.linalg.norm(wf-wc,axis=1)).tolist(),
            coarse_rotation_deg=np.rad2deg(np.linalg.norm(wc,axis=1)*results['coarse']['dt']).tolist(),
            fine_rotation_deg=np.rad2deg(np.linalg.norm(wf,axis=1)*results['fine']['dt']).tolist())
    result=dict(runs=results,comparison=diagnostics,
        runtime_ratio=results['fine']['advance_seconds']/results['coarse']['advance_seconds'],
        cost_seconds_per_100_myr_at_fixed_step={k:v['advance_seconds']*100/v['dt'] for k,v in results.items()},
        code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
        caveat='Single-step cost extrapolation only; fragmentation and topology can increase cost nonlinearly. This is not a convergence proof or remesh of the latest fractional checkpoint.')
    Path(output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['comparison'],indent=2))

if __name__=='__main__':compare(*sys.argv[1:])
