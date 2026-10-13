"""Bounded manufactured integration experiment; never loads canonical Genesis."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from analysis.simplified_genesis_validation.synthetic import irregular_three_domain_synthetic
from tectonics.simplified_genesis import SimplifiedGenesis
from tectonics.genesis_geological_handoff import GeologicalContinuation
from tectonics.genesis_transfer_bridge import encoded


def born_synthetic():
    model = irregular_three_domain_synthetic()
    # Declared initial synthetic forcing: stationary source geometry, no kick.
    # Independent mantle flow is supplied only at the geological transition.
    c, s = model.config, model.state
    c['face_torque'] = np.zeros_like(c['face_torque']).tolist()
    temperature = model.temperature(s['energy'])
    c['mass'][-2] = 4e24
    s['energy'][-2] = c['mass'][-2]*c['cp'][-2]*temperature[-2]
    model = SimplifiedGenesis(c,s)
    counts = [len(np.unique(model.state['owner']))]
    for _ in range(2):
        model.advance(25.)
        counts.append(len(np.unique(model.state['owner'])))
    return model, counts


def mapped(model):
    mantle = np.zeros((model.mesh.cell_count,3))
    mantle[:,2] = .04*model.mesh.centroids[:,0]
    return GeologicalContinuation.from_synthetic(model,
        composition='all_solid_nodes_basalt_reference_density_3000',
        mantle_omega=mantle, water_volume_km3=0.)


def exact_saved_state(a,b):
    assert json.loads((a/'mature/meta.json').read_bytes()) == json.loads((b/'mature/meta.json').read_bytes())
    with np.load(a/'mature/state.npz') as x, np.load(b/'mature/state.npz') as y:
        assert set(x.files) == set(y.files)
        for name in x.files:
            np.testing.assert_array_equal(x[name],y[name],err_msg=name)
    for name in ['record.json','source.json']:
        assert (a/name).read_bytes() == (b/name).read_bytes(), name


def run(root):
    started = time.monotonic(); root = Path(root); root.mkdir(exist_ok=False,parents=True)
    model, counts = born_synthetic(); frozen = encoded(dict(config=model.config,state=model.state))
    engine = mapped(model); engine.save(root/'mapped')
    for _ in range(3): engine.step(10.)
    engine.save(root/'split'); resumed = GeologicalContinuation.load(root/'split')
    engine.step(10.); resumed.step(10.)
    a = engine.save(root/'direct'); b = resumed.save(root/'resumed'); exact_saved_state(a,b)
    assert encoded(dict(config=model.config,state=model.state)) == frozen
    rows = engine.record['steps']
    result = dict(selector_owner_counts=counts, mature_steps=len(rows), dt_myr=10.,
        max_relative_omega_rad_per_myr=max(x['relative_omega_rad_per_myr'] for x in rows),
        transported_winning_cells=sum(x['transported_winning_cells'] for x in rows),
        transported_volume_step_sum_km3=sum(x['transported_oceanic_volume_km3'] for x in rows),
        generated_volume_step_sum_km3=sum(x['material']['generated_volume_km3'] for x in rows),
        recycled_volume_step_sum_km3=sum(x['material']['recycled_volume_km3'] for x in rows),
        max_material_residual_km3=max(abs(x['material']['volume_residual_km3']) for x in rows),
        material_balanced=all(x['material']['balanced'] for x in rows), restart_exact=True,
        source_unchanged=True, synthetic_only=True, physical_handoff_certified=False,
        elapsed_seconds=time.monotonic()-started)
    assert counts == [1,2,3], counts
    assert result['max_relative_omega_rad_per_myr'] > 0
    assert result['transported_winning_cells'] > 0
    assert result['material_balanced']
    (root/'result.json').write_bytes(encoded(result)); print(json.dumps(result,indent=2))
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('output'); run(p.parse_args().output)
