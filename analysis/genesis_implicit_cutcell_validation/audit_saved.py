"""Read-only audit of the saved real endpoint; never calls the thermal solver."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from tectonics.genesis_cutcell_loading import prepare_saved_loading
from tectonics.genesis import SECONDS_PER_MYR
from analysis.genesis_energy_inventory.audit import ROOT, ROOT_SHA


def audit(directory=Path('/workspace/genesis-implicit-cutcell-01')):
    raw = ROOT.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == ROOT_SHA
    root = json.loads(raw)
    c_raw = Path('/workspace/genesis-cutcell-feasibility-02/candidate.json').read_bytes()
    assert hashlib.sha256(c_raw).hexdigest() == '6006521e7b3e22ce23894e63248d0145ee15d3065333c949701c730c6cab47ad'
    c = json.loads(c_raw)
    saved = (directory / 'after.json').read_bytes()
    assert saved == (directory / 'restart.json').read_bytes()
    assert hashlib.sha256(saved).hexdigest() == '632354594ccda5aaffeb8933d9ed95be96cb53037a2ce67d18cf407562d6f2e8'
    p = json.loads(saved)
    prepare_saved_loading(raw, p)  # Validates seals, IDs, old T and parent memory.
    cuts = c['cells']
    assert len(cuts) == 3360
    assert p['fixed_cut_geometry'] == [{k:x[k] for k in ('face','layer','q_lo','q_hi','material_id','mass_kg')} for x in cuts]
    assert not p['mechanical_memory_updated'] and not p['retirements_applied']
    assert p['historical_opening_balance'] is None
    T = np.array(p['temperature_k']); m = np.array(p['mass_kg'])
    assert np.array_equal(m[:3360], [x['mass_kg'] for x in cuts])
    assert np.isfinite(T).all() and T.min() > 0
    params = root['source_packet']['source']['parameters']
    thermal = params['thermal']; shell = params['shell']
    # Parameters are also pinned by the authenticated projection.
    from tectonics.genesis_reservoir_thermal import load
    from tectonics.genesis import GenesisParameters
    state = load('/workspace/genesis-coupled-thermal-projection-03/projected.json', legacy_path=ROOT)
    gp = GenesisParameters(**json.loads(state.parameters_json))
    reservoirs = {x.identity:x for x in state.cells}
    assert list(m[-2:]) == [reservoirs[k].mass_kg for k in ('deep_mantle','surface_water')]
    E0 = np.array([x['mass_kg']*x['specific_h'] for x in cuts] + [reservoirs[k].energy_j for k in ('deep_mantle','surface_water')])
    cp = np.full(len(T), gp.silicate_heat_capacity_j_kg_k); cp[-1] = gp.water_heat_capacity_j_kg_k
    phi = np.clip((T-gp.solidus_k)/(gp.liquidus_k-gp.solidus_k),0,1)
    h = cp*T + gp.silicate_latent_heat_j_kg*phi
    h[-1] = cp[-1]*T[-1]+gp.water_latent_heat_j_kg
    assert T[-1] > 647.096
    assert np.array_equal(h, p['specific_h'])
    assert np.array_equal(m*h, p['energy_j'])
    assert np.array_equal(1-phi[:-1], p['solid_fraction_silicates'])
    g = root['runtime_state']['geometry']
    X = np.array(g['vertices_m'])+np.array(root['runtime_state']['displacement_m']).reshape(-1,6)[:,:3]
    tri = X[np.array(g['faces'])]
    areas = .5*np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)
    edges = []
    for face,A in enumerate(areas):
        ids = [i for i,x in enumerate(cuts) if x['face']==face]
        dz = m[ids]/(shell['density_kg_m3']*A); kA = shell['conductivity_w_m_k']*A
        edges.append((3361,ids[0],2*kA/dz[0]))
        edges.extend((i,j,kA/(.5*(dz[t]+dz[t+1]))) for t,(i,j) in enumerate(zip(ids,ids[1:])))
        edges.append((ids[-1],3360,2*kA/dz[-1]))
    assert len(edges)==len(p['paired_exchanges'])
    dt = 5e-5*SECONDS_PER_MYR
    assert p['time_myr'] == state.time_myr+5e-5
    E1 = m*h; delta = E1.astype(np.longdouble)-E0.astype(np.longdouble)
    balance = delta.copy(); ulps = 8*(np.spacing(E0)+np.spacing(E1)); allowance = ulps+3e-11*m*cp
    for (i,j,G),record in zip(edges,p['paired_exchanges']):
        Q = dt*G*(T[i]-T[j]); assert record==[i,j,Q,-Q]
        balance[i]+=np.longdouble(Q);balance[j]-=np.longdouble(Q)
        a=16*dt*G*(math.ulp(float(T[i]))+math.ulp(float(T[j])))
        allowance[i]+=a;allowance[j]+=a
    assert np.max(np.abs(balance)/allowance)<=1
    assert abs(delta.sum())<=ulps.sum()
    assert all(phi[i]==0 for i,x in enumerate(cuts) if x['material_id'] is not None)
    assert ROOT.read_bytes()==raw
    return dict(status='PASS',solver_calls=0,old_ids=800,paired_fluxes=len(edges),
                max_local_balance_over_allowance=float(np.max(np.abs(balance)/allowance)),
                global_residual_j=float(delta.sum()),restart_byte_identical=True,
                existing_prepare_volume_loading='PASS')


if __name__=='__main__':
    print(json.dumps(audit(),indent=2))
