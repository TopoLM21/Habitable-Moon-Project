"""Positive native integration plus refusal and transactional restart boundaries."""
from copy import deepcopy
import numpy as np
import pytest

from analysis.genesis_geological_handoff.experiment import born_synthetic, mapped, run
from analysis.simplified_genesis_validation.synthetic import synthetic
from tectonics.genesis_geological_handoff import GeologicalContinuation
from tectonics.genesis_transfer_bridge import encoded


@pytest.fixture(scope='module')
def source():
    return born_synthetic()[0]


def test_real_native_motion_material_and_exact_restart(tmp_path):
    result = run(tmp_path/'experiment')
    assert result['selector_owner_counts'] == [1,2,3]
    assert result['generated_volume_step_sum_km3'] > 0
    assert result['recycled_volume_step_sum_km3'] > 0
    assert result['max_material_residual_km3'] < 1e-6
    restored = GeologicalContinuation.load(tmp_path/'experiment'/'direct')
    # Full enthalpy/memory archives survive; they were not advanced or thermalized.
    initial = GeologicalContinuation.load(tmp_path/'experiment'/'mapped')
    assert encoded(restored.source) == encoded(initial.source)
    assert restored.checkpoint.thermal == initial.checkpoint.thermal
    assert restored.record['mapping']['Maxwellmemory']['released_as_heat_j'] == 0
    assert not restored.record['mapping']['H']['material_exchange_enthalpy_closed']


def test_not_ready_states_refused(source):
    with pytest.raises(ValueError, match='Selector'):
        mapped(synthetic())
    real = deepcopy(source); real.config['source_mode'] = 'genesis'
    with pytest.raises(ValueError, match='REAL_HANDOFF_NOT_READY'):
        mapped(real)
    rotated = deepcopy(source); rotated.state['rotation'][0][0][0] += .01
    with pytest.raises(ValueError, match='remapping'):
        mapped(rotated)
    with pytest.raises(ValueError, match='composition'):
        GeologicalContinuation.from_synthetic(source, composition='unknown',
            mantle_omega=np.zeros((source.nf,3)), water_volume_km3=0.)


def test_exhausted_reservoir_does_not_commit(source):
    engine = mapped(source)
    engine.step(10.)  # Native residual transport holds the first step; the next creates crust.
    engine.record['interior_reference_mass_kg'] = 0.
    before = encoded(engine.record); old = engine.checkpoint
    with pytest.raises(ValueError, match='exhausted'):
        engine.step(10.)
    assert engine.checkpoint is old
    assert encoded(engine.record) == before
    assert old.transport_state.cumulative_commit_count == 0


@pytest.mark.parametrize('name',['mature/state.npz','mature/meta.json','source.json','record.json'])
def test_corrupt_restart_refused_before_native_load(source,tmp_path,name):
    root = mapped(source).save(tmp_path/'saved')
    target = root/name
    target.write_bytes(target.read_bytes()+b'bad')
    with pytest.raises(ValueError,match='checksum'):
        GeologicalContinuation.load(root)


def test_save_cannot_overwrite(source,tmp_path):
    engine = mapped(source); root = engine.save(tmp_path/'saved')
    original = (root/'checksums.json').read_bytes()
    with pytest.raises(FileExistsError): engine.save(root)
    assert original == (root/'checksums.json').read_bytes()


def test_edge_geometry_needs_an_explicit_map(source):
    edges = deepcopy(source); edges.config['edge_strips'] = {'enabled': True}
    with pytest.raises(ValueError, match='edge mapping'):
        mapped(edges)
