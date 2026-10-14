"""Observe unmodified b0ba7e5 runner; wrappers do not alter inputs or results."""
import sys,json,time,importlib.util,math
from pathlib import Path
repo,config,output,dt=sys.argv[1:];sys.path.insert(0,repo)
import numpy as np
import run_long_evolution_v123 as runner
from tectonics.mesh import connected_components
import tectonics.transport as transport
spec=importlib.util.spec_from_file_location('common_diagnostics',Path(__file__).with_name('diagnostics.py'))
diagnostics=importlib.util.module_from_spec(spec);spec.loader.exec_module(diagnostics)
history=[];geometry=dict(raw_extra_components=0,post_overlap_extra_components=0,max_assignment_error_km=0.)
last={};original_topology=runner.PlateTopologyManager.update
def topology(self,mesh,state,system,*args,**kwargs):
    geometry['post_overlap_extra_components']+=sum(max(len(connected_components(np.flatnonzero(state.cell_plate==p),mesh.neighbors))-1,0) for p in range(len(system.plates)))
    out=original_topology(self,mesh,state,system,*args,**kwargs);last.update(mesh=mesh,state=state,system=out[0]);return out
runner.PlateTopologyManager.update=topology
original_hydro=runner.advance_hydrosphere
def hydro(mesh,state,topo,previous,radius,*args,**kwargs):
    out=original_hydro(mesh,state,topo,previous,radius,*args,**kwargs)
    if round(state.time_myr/20)*20==state.time_myr:
        history.append(diagnostics.snapshot(mesh,state,last['system'],topo,out[1],radius))
    return out
runner.advance_hydrosphere=hydro
original_map=transport.build_transport_map
def mapping(mesh,system,state,step,memory,*args,**kwargs):
    oldq=memory.residual_quaternions.copy();out=original_map(mesh,system,state,step,memory,*args,**kwargs)
    for p,plate in enumerate(system.plates):
        source=np.flatnonzero(state.cell_plate==p);target=out.source_to_target[p]
        if not len(source):continue
        geometry['raw_extra_components']+=max(len(connected_components(target,mesh.neighbors))-1,0)
        q=transport.quaternion_multiply(transport.quaternion_from_axis_angle(plate.euler_axis,plate.angular_speed_rad_per_myr*step),oldq[p])
        desired=transport.rotate_by_quaternion(mesh.centroids[source],q)
        error=np.arccos(np.clip(np.sum(desired*mesh.centroids[target],axis=1),-1,1))*5287.
        geometry['max_assignment_error_km']=max(geometry['max_assignment_error_km'],float(error.max()))
    return out
transport.build_transport_map=mapping
root=Path(output);root.mkdir(parents=True,exist_ok=False)
sys.argv=['reference','--config',config,'--output',output,'--dt',dt,'--end-time','500']
started=time.monotonic();runner.main()
(root/'observations.json').write_text(json.dumps(dict(reference_commit='b0ba7e51da0177447aec4fb632cec1c154e9504d',dt_myr=float(dt),history=history,geometry=geometry,elapsed_seconds=time.monotonic()-started),indent=2))
