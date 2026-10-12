"""Source-bound column conductances for the opt-in finite reservoirs."""
import hashlib,json,math
import numpy as np
from .genesis_reservoir_thermal import validate,conduct
from .genesis_volume_thermal import _PhysicalAreaMesh


def column_network(state,legacy_raw):
    validate(state);origin=json.loads(state.origin_json)
    if hashlib.sha256(legacy_raw).hexdigest()!=origin['legacy_raw_sha256']:raise ValueError('Legacy geometry SHA')
    root=json.loads(legacy_raw);r=root['runtime_state'];params=root['source_packet']['source']['parameters']
    if json.loads(state.parameters_json)!=params['thermal']:raise ValueError('Thermal parameters changed')
    g=r['geometry'];X=np.array(g['vertices_m'])+np.array(r['displacement_m']).reshape(-1,6)[:,:3]
    mesh=_PhysicalAreaMesh(X,g['faces']);area=mesh.area_m2
    masses=np.asarray(r['thermal']['layer_mass_kg']);rho=params['shell']['density_kg_m3'];k=params['shell']['conductivity_w_m_k']
    dz=masses/(rho*area[:,None]);cells={c.identity:c for c in state.cells}
    expected={'deep_mantle','surface_water'}|{f'column:{i}:{j}' for i,row in enumerate(masses) for j in range(len(row))}
    thermal={c.identity for c in state.cells if c.material in ('water','silicate')}
    if thermal!=expected:raise ValueError('Overlapping/missing thermal material ownership')
    edges=[]
    for i,row in enumerate(masses):
        for j,m in enumerate(row):
            if cells[f'column:{i}:{j}'].mass_kg!=m:raise ValueError('Column mass changed')
        edges.append(('surface_water',f'column:{i}:0',float(2*k*area[i]/dz[i,0])))
        for j in range(len(row)-1):edges.append((f'column:{i}:{j}',f'column:{i}:{j+1}',float(k*area[i]/(.5*(dz[i,j]+dz[i,j+1])))))
        edges.append((f'column:{i}:{len(row)-1}','deep_mantle',float(2*k*area[i]/dz[i,-1])))
    return edges,dict(area_m2=area.tolist(),dz_m=dz.tolist(),conductivity_w_m_k=k,density_kg_m3=rho,
       legacy_geometry_sha256=origin['legacy_raw_sha256'],geometry_scope='frozen_actual_top_chord_areas_at_legacy_time',geometry_time_myr=r['thermal']['time_myr'])


def step(state,legacy_raw,dt_seconds):
    edges,geometry=column_network(state,legacy_raw)
    # Thermal isolated step cannot silently advance old mechanical memory.
    if any(c.material in ('elastic','kinetic') for c in state.cells):raise ValueError('Mechanical attachment requires coupled driver')
    after,journal=conduct(state,edges,dt_seconds)
    return after,dict(geometry=geometry,exchanges=journal,dt_seconds=dt_seconds,
      external_forcing='disabled_for_closed_isolated_control',mechanics_advanced=False,
      latent_heat='inside_enthalpy_only_no_extra_birth_source')
