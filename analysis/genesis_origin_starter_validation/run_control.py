"""Real molten Genesis -> solidification history -> first partition import.

No synthetic cold state, benchmark octants, or mature initialization is used.
The stronger loading control is explicit and distinct from canonical physics.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tectonics.genesis import parameters_from_config
from tectonics.genesis_checkpoint_compat import MODEL_VERSION
from tectonics.genesis_shell import shell_parameters_from_config
from tectonics.genesis_solidification import solidification_history_to_dict
from tectonics.genesis_starter import StarterModel, StarterParameters, VERSION
from tectonics.genesis_starter_continuation import build_starter_continuation
from tectonics.genesis_starter_material import primary_material_inventory
from tectonics.genesis_tides import tidal_parameters_from_config
from tectonics.mesh import build_icosphere
from tectonics.simulation import load_config


def enc(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False).encode()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_catalogue():
    paths = sorted((ROOT / 'tectonics').glob('*.py'))
    paths += [ROOT / 'configs/genesis_moon.yaml', ROOT / 'configs/canonical_moon.yaml',
              Path(__file__).resolve()]
    return {str(path.relative_to(ROOT)): sha(path) for path in paths}


def write_json(path, value):
    with Path(path).open('xb') as stream:
        stream.write(enc(value)+b'\n')


def equal_npz(left, right):
    with np.load(left, allow_pickle=False) as a, np.load(right, allow_pickle=False) as b:
        assert a.files == b.files
        for key in a.files:
            assert np.array_equal(a[key], b[key]), key


def run(output, traction, require_partition):
    started = time.monotonic()
    output = Path(output).resolve()
    assert output.is_relative_to('/workspace') and not output.exists()
    output.mkdir()
    before_sources = source_catalogue()
    write_json(output / 'sources_before.json', before_sources)
    config = load_config(ROOT / 'configs/genesis_moon.yaml')
    thermal = parameters_from_config(config)
    tides = tidal_parameters_from_config(config, thermal)
    shell = replace(shell_parameters_from_config(config), subdivisions=2)
    canonical_traction = shell.convective_traction_pa
    if traction is not None:
        shell = replace(shell, convective_traction_pa=traction)
    model = StarterModel(build_icosphere(2), thermal, tides, shell, StarterParameters())
    initial = model.initial_state()
    initial_sample = model.loading.sample(initial.thermal_context)
    initial_material = primary_material_inventory(model, initial)
    assert initial.time_myr == 0. and initial.solidification_history.cohorts == ()
    assert initial.solidification_history.solid_mass_kg == 0.
    assert not np.any(initial_material.primary_mass_kg)
    assert initial_sample.thermal['ocean_fraction'] == 0.
    assert initial_sample.thermal['surface_melt_fraction'] == 1.
    assert initial_sample.thermal['mantle_melt_fraction'] == 1.
    assert not np.any(initial.damage) and not np.any(initial.cooling_stress_pa)
    model.save_state(output / 'molten_checkpoint.npz', initial)

    # Genuine partial solidification, before any first partition.
    partial = model.advance(initial, .65)
    partial_material = primary_material_inventory(model, partial)
    assert partial.stopped_reason is None
    assert 0. < partial_material.surface_solid_fraction < 1.
    assert partial.solidification_history.solid_mass_kg > 0.
    assert partial.solidification_history.mean_age_myr > 0.
    assert initial.solidification_history.cohorts == ()  # Trial does not mutate parent.
    model.save_state(output / 'partial_checkpoint.npz', partial)
    restored = model.load_state(output / 'partial_checkpoint.npz')
    assert restored.solidification_history == partial.solidification_history
    target = 2. if require_partition else 1.2
    direct = model.advance(partial, target)
    resumed = model.advance(restored, target)
    model.save_state(output / 'direct_checkpoint.npz', direct)
    model.save_state(output / 'resumed_checkpoint.npz', resumed)
    equal_npz(output / 'direct_checkpoint.npz', output / 'resumed_checkpoint.npz')
    history = direct.solidification_history
    material = primary_material_inventory(model, direct)
    np.testing.assert_allclose(history.solid_mass_kg, material.primary_mass_kg.sum(), rtol=2e-12)
    assert 0. < history.mean_birth_time_myr < direct.time_myr
    assert 0. < history.mean_age_myr < direct.time_myr
    assert history.mean_age_myr == direct.time_myr-history.mean_birth_time_myr
    assert history.time_myr == direct.time_myr
    import_checks = None
    if require_partition:
        assert direct.stopped_reason == 'first_partition'
        assert len(direct.system.plates) > 1
        canonical = load_config(ROOT / 'configs/canonical_moon.yaml')
        bundle, _, import_report = build_starter_continuation(model, direct, canonical)
        np.testing.assert_array_equal(bundle.checkpoint.state.crust_age_myr,
                                      np.full(model.mesh.cell_count, history.mean_age_myr))
        np.testing.assert_array_equal(bundle.checkpoint.state.cell_plate, direct.system.cell_plate)
        assert not np.any(bundle.checkpoint.state.continental_volume_km3)
        assert bundle.checkpoint.state.time_myr == direct.time_myr
        np.testing.assert_allclose(bundle.checkpoint.state.oceanic_volume_km3,
            material.primary_mass_kg/(material.primary_density_kg_m3*1e9), rtol=2e-15)
        import_checks = dict(material_age_not_reset=True, material_mass_preserved=True,
            actual_domain_labels_preserved=True, absolute_clock_preserved=True,
            no_continental_material_invented=True, physical_handoff_certified=False,
            import_report=import_report)
        # Sibling parameters permit the real isolated continuation CLI to consume
        # this actual first-partition checkpoint without a second initialization.
        model.save_state(output / 'starter_checkpoint.npz', direct)
        write_json(output / 'parameters.json', dict(format='genesis-starter-run-0.1',
            thermal_model_version=MODEL_VERSION, thermal=asdict(thermal), tides=asdict(tides),
            shell=asdict(shell), starter=asdict(model.parameters), step_myr=.1,
            provenance=dict(config_path=str(ROOT / 'configs/genesis_moon.yaml'),
                config_sha256=sha(ROOT / 'configs/genesis_moon.yaml'),
                numerical_subdivisions=2, explicit_traction_override_pa=traction,
                origin='actual molten initial_state, continued through accepted thermal samples')))
    after_sources = source_catalogue()
    write_json(output / 'sources_after.json', after_sources)
    assert before_sources == after_sources, 'Sources changed during actual control'
    report = dict(format='genesis-origin-starter-actual-control-1', status='PASS',
        model_version=VERSION, config_sha256=sha(ROOT / 'configs/genesis_moon.yaml'),
        canonical_traction_pa=canonical_traction, actual_traction_pa=shell.convective_traction_pa,
        explicit_physical_control=traction is not None, mesh_subdivisions=2,
        radius_km=thermal.radius_km, initial_sample=initial_sample.thermal,
        partial_time_myr=partial.time_myr,
        partial_solid_fraction=partial_material.surface_solid_fraction,
        final_diagnosis=model.diagnose(direct),
        actual_solidification_history=solidification_history_to_dict(history),
        direct_vs_restart_complete_npz_payload_identical=True,
        sources_unchanged=True, first_partition_import=import_checks,
        continental_differentiation_executed=False, new_COMMON_initialization_adopted=False,
        seconds=time.monotonic()-started)
    report['sha256'] = hashlib.sha256(enc(report)).hexdigest()
    write_json(output / 'report.json', report)
    print(enc(report).decode(), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--traction-pa', type=float)
    parser.add_argument('--require-partition', action='store_true')
    args = parser.parse_args()
    run(args.output, args.traction_pa, args.require_partition)
