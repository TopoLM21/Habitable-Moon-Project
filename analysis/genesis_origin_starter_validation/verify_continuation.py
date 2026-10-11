"""Independent file comparison of actual continuous and resumed young runs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def enc(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False).encode()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(output):
    direct = Path('/workspace/genesis_origin_young_control_20261009_direct02')
    resumed = Path('/workspace/genesis_origin_young_control_20261009_resumed02')
    first = Path('/workspace/genesis_origin_starter_control_20261009_partition_01')
    report_a = json.loads((direct / 'continuation.json').read_bytes())
    report_b = json.loads((resumed / 'continuation.json').read_bytes())
    assert report_a['status'] == report_b['status'] == 'completed'
    assert report_a['checks'] == report_b['checks'] and all(report_a['checks'].values())
    assert report_a['clocks'] == report_b['clocks']
    assert len(set(report_a['clocks'].values())) == 1
    assert report_a['final_time_myr'] == report_b['final_time_myr']
    assert report_a['primary_solidification_history'] == report_b['primary_solidification_history']
    assert report_a['physical_handoff_certified'] is report_b['physical_handoff_certified'] is False
    npz_equalities = {}
    for rel in ('mature_checkpoint/state.npz', 'young_context/starter_checkpoint.npz',
                'young_context/fracture_memory.npz'):
        with np.load(direct / rel, allow_pickle=False) as a, np.load(resumed / rel, allow_pickle=False) as b:
            assert a.files == b.files
            for key in a.files:
                assert np.array_equal(a[key], b[key]), (rel, key)
            npz_equalities[rel] = a.files
    origin = json.loads((first / 'report.json').read_bytes())
    initial_age = origin['final_diagnosis']['primary_mean_age_myr']
    with np.load(direct / 'mature_checkpoint/state.npz', allow_pickle=False) as state:
        np.testing.assert_allclose(state['crust_age_myr'], initial_age+.2, rtol=0, atol=2e-15)
        assert np.all(state['crust_age_myr'] > .2)
        assert not np.any(state['continental_volume_km3'])
        final_age = float(np.mean(state['crust_age_myr']))
    original_history = origin['actual_solidification_history']
    current_history = report_a['primary_solidification_history']
    assert current_history['cohorts'] == original_history['cohorts']
    assert current_history['parameters_sha256'] == original_history['parameters_sha256']
    assert current_history['intervals'][:len(original_history['intervals'])] == original_history['intervals']
    assert len(current_history['intervals']) > len(original_history['intervals'])
    assert report_a['ocean_fraction'] > .99
    result = dict(format='genesis-origin-young-independent-restart-comparison-1', status='PASS',
        direct_directory=str(direct), resumed_directory=str(resumed),
        complete_npz_payload_equalities=npz_equalities, all_joint_checks_passed=report_a['checks'],
        absolute_clocks=report_a['clocks'], retained_primary_age_myr=final_age,
        actual_primary_birth_history_preserved=True, liquid_ocean_fraction=report_a['ocean_fraction'],
        original_cohort_count=len(original_history['cohorts']),
        current_interval_count=len(current_history['intervals']),
        new_3d_physical_handoff_certified=False,
        raw_report_sha256=dict(direct=sha(direct / 'continuation.json'),
                              resumed=sha(resumed / 'continuation.json')))
    result['sha256'] = hashlib.sha256(enc(result)).hexdigest()
    output = Path(output)
    with output.open('xb') as stream:
        stream.write(enc(result)+b'\n')
    print(enc(result).decode(), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    run(parser.parse_args().output)
