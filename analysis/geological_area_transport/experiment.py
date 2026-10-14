"""Local prescribed rotation/collision, dt1/.5; no geological evolution."""
import json,math,time
from dataclasses import asdict
from pathlib import Path
import numpy as np
from tectonics.mesh import build_icosphere
from tectonics.geological_area_transport import MaterialPatch,transfer_rigid_patches,inspect_two_owner_contact
from tectonics.transport import _optimal_assignment,SubgridTransportParameters,quaternion_from_axis_angle,rotate_by_quaternion


def fixture(subdivisions=1):
    mesh=build_icosphere(subdivisions);radius=5287.;areas=mesh.physical_cell_areas_km2(radius)
    from tectonics.spherical_polygons import clip_hemisphere,polygon_area
    patches=[]
    # Exact geometric hemispheres, including triangles cut by x=0. This gives
    # an independent spherical-lune oracle for collision area under +/- z motion.
    for i,face in enumerate(mesh.faces):
        for owner,normal in [(0,[1.,0.,0.]),(1,[-1.,0.,0.])]:
            poly=clip_hemisphere(mesh.vertices[face],normal)
            if not len(poly):continue
            area=polygon_area(poly)*radius**2
            patches.append(MaterialPatch(f'source-{i:03}-{owner}',owner,poly,
                float(area*7*1e9*3000),float(area*7*1e9*3000*(8e5+i*1000)),'mafic'))
    omega={0:np.array([0.,0.,.01]),1:np.array([0.,0.,-.01])}
    return mesh,radius,patches,omega


def run(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    mesh,r,patches,omega=fixture();runs=[]
    for dt in (1.,.5):
        start=time.monotonic();history=[]
        for i in range(round(2/dt)):
            t=(i+1)*dt
            result=transfer_rigid_patches(mesh,patches,omega,t,r)
            contact=inspect_two_owner_contact(mesh,result,r)
            history.append(dict(time_myr=t,**result.diagnostics,contact=contact,analytic_lune_area_km2=4*.01*t*r*r,collision_minus_analytic_km2=contact['collision_area_km2']-4*.01*t*r*r))
        # Fragment ownership belongs to the original plate, not the target.
        original={p.identity:p for p in patches}
        assert all(p.owner==original[p.identity].owner and p.composition==original[p.identity].composition for p in result.pieces)
        runs.append(dict(dt_myr=dt,elapsed_seconds=time.monotonic()-start,history=history,
            max_global_mass_relative_error=max(abs(h['total_mass_residual_kg'])/math.fsum(p.mass_kg for p in patches) for h in history),
            max_global_H_relative_error=max(abs(h['total_H_residual_j'])/math.fsum(p.enthalpy_j for p in patches) for h in history),
            stable_source_owners=True,final_parts=[dict(identity=p.identity,owner=p.owner,target=p.target,area_km2=p.area_km2,mass_kg=p.mass_kg,enthalpy_j=p.enthalpy_j) for p in result.pieces]))
    # An accumulated rotation exposes unequal-area raster assignment directly.
    a=mesh.physical_cell_areas_km2(r);src=np.flatnonzero(mesh.centroids[:,0]>=0)
    q=quaternion_from_axis_angle(np.array([.2,.5,.7])/np.linalg.norm([.2,.5,.7]),.24)
    target=_optimal_assignment(mesh,rotate_by_quaternion(mesh.centroids[src],q),SubgridTransportParameters(connected_footprint=True))
    old=dict(rotation_rad=.24,source_cells=src.tolist(),target_cells=target.tolist(),
        source_area_km2=float(a[src].sum()),target_area_km2=float(a[target].sum()),
        total_area_error_km2=float(a[target].sum()-a[src].sum()),max_identity_area_relative_error=float(np.max(abs(a[target]-a[src])/a[src])))
    axis=np.array([.2,.5,.7])/np.linalg.norm([.2,.5,.7])
    subset=[MaterialPatch(f'unequal-{i}',0,mesh.vertices[mesh.faces[i]],a[i]*7e9*3000,a[i]*7e9*3000*8e5,'mafic') for i in src]
    check=transfer_rigid_patches(mesh,subset,{0:axis*.24,1:axis*.24},1.,r)
    old['intersection_transfer_diagnostics']=check.diagnostics
    result=dict(radius_km=r,mesh_cells=mesh.cell_count,source_patch_count=len(patches),source_area_ratio=float(a.max()/a.min()),
        motion_rad_per_myr={str(k):v.tolist() for k,v in omega.items()},runs=runs,legacy_unequal_area=old,
        final_partition_exact_equal=runs[0]['final_parts']==runs[1]['final_parts'],
        hypothesis='none: prescribed rigid geometry, overlaps pending, no basalt retirement or new material')
    (output/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='runs'},indent=2))
    return result

if __name__=='__main__':
    import sys
    run(sys.argv[1])


def fragment_event():
    """Expose the existing pickup law in isolation; no contact law is added."""
    from tectonics.plates import Plate,PlateSystem
    from tectonics.lithosphere import LithosphereState
    from tectonics.topology import PlateTopologyManager,PlateTopologyParameters
    mesh=build_icosphere(2);owner=np.ones(mesh.cell_count,dtype=int)
    owner[np.abs(mesh.centroids[:,2])>.6]=0;crumb=int(np.argmax(mesh.centroids[:,0]));owner[crumb]=0
    system=PlateSystem(owner.copy(),tuple(Plate(i,int(np.flatnonzero(owner==i)[0]),np.array([0.,0.,1.]),.003+.001*i) for i in range(2)))
    areas=mesh.physical_cell_areas_km2(5287.);n=mesh.cell_count
    state=LithosphereState(500.,owner.copy(),np.zeros(n,dtype=np.int8),np.ones(n)*100,np.ones(n)*7,np.zeros(n),oceanic_volume_km3=areas*7)
    saved=state.oceanic_volume_km3.copy();trace={}
    manager=PlateTopologyManager(PlateTopologyParameters(split_enabled=False,merge_enabled=False,
        max_events_per_step=0,min_plate_area_km2=10e6))
    final,diag,events=manager.update(mesh,state,system,[],5287.,1.,connectivity_trace=trace)
    return dict(crumb=crumb,diagnostics=asdict(diag),contact_trace=trace,
        material_volume_exact_unchanged=bool(np.array_equal(saved,state.oceanic_volume_km3)),
        note='existing pickup observed, not approved as a new mechanism; angular momentum proxy is uniform thin shell')
