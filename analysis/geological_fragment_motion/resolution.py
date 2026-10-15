"""Bounded nested-grid experiment; not a remesher for fractional checkpoints."""
import argparse, json, math, time, subprocess, signal
from copy import deepcopy
from dataclasses import fields, is_dataclass, replace, asdict
from pathlib import Path
import numpy as np
from tectonics.unified_geological_genesis import UnifiedGeologicalGenesis as U
from tectonics.mesh import build_icosphere
from tectonics.plates import PlateSystem
from tectonics.dynamics import angular_velocity_vectors
from tectonics.genesis_transfer_bridge import encoded
from tectonics.genesis_geological_handoff import sha
from analysis.geological_continuation.run import snap
from analysis.genesis_geological_handoff.experiment import exact_saved_state

SOURCE='/workspace/geological-contact-area/dt1/accepted'

def totals(e):
    return [math.fsum(e.record['thermal_inventory'][k]) for k in ('mass_kg','energy_j')]

def stocks(e):
    out={}
    for parts in e.record['material_parcels']['node_parts'].values():
        for p in parts:
            out.setdefault(p['id'],[[],[]])
            for j,k in enumerate(('mass_kg','energy_j')):out[p['id']][j].append(p[k])
    return {k:[math.fsum(v[0]),math.fsum(v[1])] for k,v in out.items()}

def refine(e):
    """Piecewise-constant nested prolongation of a nonfractional state.

    Extensive columns and every material ID split by exact child spherical area.
    No interpolation of ages, relief or chemistry; no fractional history reset.
    """
    if 'fractional_surface' in e.record:raise ValueError('Fractional remesh unsupported')
    if e.checkpoint.subduction_memory.young_boundary_state is not None:
        raise ValueError('Young boundary remesh unsupported')
    old=deepcopy(e);n=e.mesh.cell_count;level=e.record['subdivisions']+1
    mesh=build_icosphere(level);parent=np.arange(mesh.cell_count)//4
    a=mesh.physical_cell_areas_km2(e.record['radius_km'])
    w=(a.reshape(n,4)/a.reshape(n,4).sum(1)[:,None]).ravel()
    assert np.allclose(a.reshape(n,4).sum(1),e.mesh.physical_cell_areas_km2(e.record['radius_km']),rtol=2e-14)
    # Explicit surface-field containers; plate-sized arrays are not resampled.
    names=['state','cycle','topo','mantle_flow','plume_state','plume_rifting_state',
           'plume_dynamic_topography_state','plume_magmatism_state','hotspot_track_state']
    for name in names:
        obj=getattr(e.checkpoint,name)
        for f in fields(obj):
            v=getattr(obj,f.name)
            if isinstance(v,np.ndarray) and v.shape and v.shape[0]==n:
                new=v[parent].copy()
                if f.name.endswith('volume_km3'):new*=w
                setattr(obj,f.name,new)
    for name in ('system','baseline'):
        system=getattr(e.checkpoint,name)
        setattr(e.checkpoint,name,PlateSystem(system.cell_plate[parent].copy(),
            tuple(replace(p,seed_cell=p.seed_cell*4) for p in system.plates)))
    manager=e.checkpoint.manager
    manager.collision_contact_faces={k:tuple(4*i+j for i in v for j in range(4)) for k,v in manager.collision_contact_faces.items()}
    # Keep physical areas/spans and equivalent minimum sampled area unchanged.
    params=asdict(manager.params)
    for k in ('min_plate_cells','split_min_child_cells','disconnect_min_child_cells'):params[k]*=4
    params['split_min_rift_cells']*=2 # edge-length sampling doubles
    manager.params=type(manager.params)(**params);e.record['topology_parameters']=params
    e.record['parameters']['min_child_cells']*=4
    p=e.record['material_parcels'];oldnodes=p['nodes'];nodes=[];parts={};mass=[];energy=[]
    for child,par in enumerate(parent):
        row=[]
        for node in oldnodes[par]:
            i=len(mass);row.append(i)
            q=[dict(x,mass_kg=x['mass_kg']*w[child],energy_j=x['energy_j']*w[child]) for x in p['node_parts'][str(node)]]
            parts[str(i)]=q;mass.append(math.fsum(x['mass_kg'] for x in q));energy.append(math.fsum(x['energy_j'] for x in q))
        nodes.append(row)
    mass+=e.record['thermal_inventory']['mass_kg'][-2:];energy+=e.record['thermal_inventory']['energy_j'][-2:]
    p['weights']=[deepcopy(p['weights'][i]) for i in parent];p['nodes']=nodes;p['node_parts']=parts
    e.record['thermal_inventory']=dict(mass_kg=mass,energy_j=energy)
    for k in ('fracture_damage','material_strength'):e.record[k]=np.asarray(e.record[k])[parent].tolist()
    e.source['config'].update(mass=mass,solid_nodes=nodes);e.source['state']['energy']=energy
    e.source['resolution_parent_source_sha256']=old.record['source_sha256']
    e.record['source_sha256']=sha(encoded(e.source));e.record['subdivisions']=level;e.mesh=mesh
    before=stocks(old);after=stocks(e)
    assert before.keys()==after.keys()
    err=max(abs(after[k][j]-v)/max(abs(v),1.) for k,vals in before.items() for j,v in enumerate(vals))
    assert err<2e-14
    m0,h0=totals(old);m,h=totals(e)
    assert abs(m-m0)/m0<2e-14 and abs(h-h0)/h0<2e-14
    for k in ('oceanic_volume_km3','continental_volume_km3','sediment_volume_km3'):
        assert math.isclose(math.fsum(getattr(old.checkpoint.state,k)),math.fsum(getattr(e.checkpoint.state,k)),rel_tol=2e-14,abs_tol=1e-12)
    assert np.array_equal(angular_velocity_vectors(e.checkpoint.system),angular_velocity_vectors(old.checkpoint.system))
    return dict(parent_cells=n,cells=mesh.cell_count,material_ids=len(before),max_id_relative_error=err,
                mass_relative_error=(m-m0)/m0,H_relative_error=(h-h0)/h0,
                fractional_remesh=False,physical_thresholds_preserved=True)

def run(root,fine,dt):
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    e=U.load(SOURCE);audit=refine(e) if fine else {'cells':e.mesh.cell_count}
    e.enable_fractional_transport();e.save(root/'initial')
    result=dict(source=SOURCE,sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                dt=dt,duration=dt,initialization=audit,initial=snap(e),history=[snap(e)],error=None)
    m0,h0=totals(e);export=e.record['cumulative_exported_heat_j'];water=e.record['water_mass_kg']
    omega0=angular_velocity_vectors(e.checkpoint.system);start=time.monotonic()
    def timeout(*_):raise TimeoutError('Single-arm 240-second hard wall budget')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(240)
    try:
        e.step(dt);result['advance_seconds']=time.monotonic()-start
        e.save(root/'accepted');result['history'].append(snap(e));m,h=totals(e)
        result.update(final=snap(e),mass_relative_error=(m-m0)/m0,
            H_relative_error=(h-h0+e.record['cumulative_exported_heat_j']-export)/h0,
            water_change_kg=e.record['water_mass_kg']-water,
            initial_omega_rad_per_myr=omega0.tolist(),final_omega_rad_per_myr=angular_velocity_vectors(e.checkpoint.system).tolist(),
            surface_patches=len(e.record['fractional_surface']))
        repeat=U.load(root/'initial');t=time.monotonic();repeat.step(dt)
        result['repeat_seconds']=time.monotonic()-t
        exact_saved_state(root/'accepted',repeat.save(root/'replayed'));result['restart_exact']=True
    except Exception as exc:
        import traceback
        result['error']=repr(exc);(root/'failure.txt').write_text(traceback.format_exc())
        e.save(root/'last_accepted')
    finally:signal.alarm(0)
    result['total_seconds']=time.monotonic()-start
    (root/'result.json').write_bytes(encoded(result));print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');p.add_argument('--fine',action='store_true');p.add_argument('--dt',type=float,default=.25)
    a=p.parse_args();run(a.output,a.fine,a.dt)
