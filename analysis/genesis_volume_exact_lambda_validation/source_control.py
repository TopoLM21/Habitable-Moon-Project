"""Compact, authenticated original Genesis inputs for exact-lambda integration.

The birth-memory fixture and accepted parent remain authoritative.  This
reader reuses their frozen source checks and the compact literal comparison
archive for the existing computational bases and probes.  It does not read
the large clipped point shards, restore or initialize a planet, reassemble
parent matrices, integrate a domain, or advance accepted history.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analysis.genesis_volume_clipped_validation import actual_control as original

PACKAGE = ROOT / 'analysis/genesis_volume_clipped_validation/evidence'
PACKAGE_MANIFEST_PIN = dict(bytes=1108090,
    raw_sha256='646adad680383eb672f2ac403976c21d2698c7f95e8fe656ce274ff14ea8cca5')
PACKAGE_MANIFEST_SELF = '319639683bae6984a480f4d6315ed6923fe52840ee6c6c08c8e374caf5d71bf0'
CLIPPED_READBACK = Path('/workspace/genesis_volume_clipped_final_readback_20261011.json')
CLIPPED_READBACK_SELF = 'a1bed65b16c440156148a431f759114ce5299cd54ead84ca102c624c9ae1e582'
ORIGINALS = Path('/workspace/genesis_volume_exact_lambda_originals_20261011')
BASELINE_SELF = '25bef200cfe1afb296749e58869d70a3ff8fd82dedc6bf84ba4498c59001da67'
UNION_SELF = '7f0ccfdd55ace0a54c84bbd1c9778bbd4a9041eefa6a1741c5525d523f8eef9f'
MONITOR = 'monitoring/genesis_volume_clipped_control_monitor_receipt_20261011_02.json'
NEGATIVES = 'independent/resealed_negatives'
FORMAT = 'genesis-exact-lambda-authenticated-compact-source-0.1'
GEOMETRY_KEYS = ('triangle_m', 'displacement_A_m', 'slope_P_m', 'lambda_bounds',
    'normal_xyz', 'offset_m', 'cohort_id', 'face_index', 'owner_id', 'material_id',
    'mass_kg', 'volume_m3', 'q_bounds', 'conical_mass_coefficient')
HEAVY_SOURCE_KEYS = ('literal_original_parent_blocks',
    'original_source_memory_birth', 'original_accepted_interval_reaction')


def checked_json(path, key, expected=None):
    return original.check_seal(json.loads(Path(path).read_bytes()), key, expected)


def source_member(manifest, relative, key):
    entries = [item for item in manifest['files'] if item['path'] == relative]
    original.require(len(entries) == 1, 'Compact provenance member is absent or duplicate: ' + relative)
    expected = {name: entries[0][name] for name in ('bytes', 'raw_sha256')}
    path = PACKAGE / relative
    original.require(original.pin(path) == expected, 'Compact provenance raw pin differs: ' + relative)
    value = checked_json(path, key)
    return value, dict(path=str(path), **expected, self_seal_key=key, self_sha256=value[key])


def _support_packet(case, label, support, source):
    comparison = original.check_seal(support['literal_comparison_packet'], 'comparison_sha256')
    packets = comparison['comparison_packets']
    original.require(packets and comparison['physical_contraction_certified'] is False
        and comparison['available_for_creation_funding'] is False,
        'Original declared basis acquired physical admission')
    first = original.check_seal(packets[0], 'binding_sha256')
    W, probes, parent_basis = first['declared_added_basis'], first['probes'], first['declared_parent_basis']
    original.require(np.asarray(W).shape == (18, 15)
        and (parent_basis is None or np.array_equal(parent_basis, np.eye(18))) and len(probes) == 5,
        'Original full coordinates, operative basis or probe coverage differs')
    for packet in packets:
        original.check_seal(packet, 'binding_sha256')
        original.require(packet['declared_added_basis'] == W and packet['declared_parent_basis'] == parent_basis
            and packet['probes'] == probes and packet['projection_before_Gram'] is True
            and packet['relative_tolerance'] == 2e-6,
            'Original sequential basis, probes, arithmetic or tolerance differs')
    original.require(probes[:4] == list(original.declared_probes())
        and probes[4] == dict(name='saved_accepted_interval_mean',
            parent_increment_m=probes[0]['parent_increment_m'],
            enrichment_increment_m=probes[0]['enrichment_increment_m'],
            mean_e0=source['accepted_interval_mean_e0'], mean_e1=source['accepted_interval_mean_e1']),
        'Original virtual directions or calculated interval mean were replaced')
    definition = source['previous_seed_definition']
    radius = 100.0 if label == 'fixed_100m_clipped_support' else definition['seed_radius_m']
    return original.seal(dict(label=label, seed_center_world_m=definition['seed_center_world_m'],
        seed_radius_m=radius, original_declared_added_basis=W, original_declared_parent_basis=parent_basis,
        resolved_parent_basis=np.eye(18).tolist(), original_parent_basis_default_identity=(parent_basis is None),
        declared_probes=probes, source_cohort_id=case['descriptor']['original_cohort_id'],
        original_virtual_surface_basis_sha256=comparison['operative_basis_virtual_sha256'],
        original_comparison_sha256=comparison['comparison_sha256'],
        original_comparison_packet_sha256=[packet['binding_sha256'] for packet in packets],
        original_clipped_case_descriptor=case['descriptor'],
        basis_scope='literal_existing_computational_virtual_Gamma_basis_not_physical_history',
        original_kernel_coordinates_tied=True, parent_coordinate_count=18, added_coordinate_count=18,
        operative_coordinate_count=15, kernel_coordinate_count=3, radius_unchanged=True,
        physical_Gamma_selected=False, physical_creation_admitted=False, physical_release_admitted=False,
        physical_contraction_certified=False, available_for_creation_funding=False), 'support_sha256')


def load_authentic_source_fixture():
    """Return three sealed compact DTOs; perform source binding only."""
    started = time.perf_counter()
    original.require(original.pin(PACKAGE / 'package_manifest.json') == PACKAGE_MANIFEST_PIN,
                     'Last accepted clipped package manifest raw anchor differs')
    manifest = checked_json(PACKAGE / 'package_manifest.json', 'sha256', PACKAGE_MANIFEST_SELF)
    readback = checked_json(CLIPPED_READBACK, 'sha256', CLIPPED_READBACK_SELF)
    original.require(readback['status'] == 'PASS' and readback['source_count'] == 434
        and readback['package_manifest'] == PACKAGE_MANIFEST_PIN
        and readback['package_manifest_self_sha256'] == PACKAGE_MANIFEST_SELF,
        'Last metadata readback does not authenticate the old source package')
    baseline = checked_json(ORIGINALS / 'baseline_manifest.json', 'sha256', BASELINE_SELF)
    union = checked_json(ORIGINALS / 'unchanged_previous_434_source_pins.json', 'sha256', UNION_SELF)
    old_sources = union['source_sha256']
    original.require(baseline['accepted_previous_source_count'] == union['source_count'] == len(old_sources) == 434
        and manifest['source_sha256'] == old_sources,
        'Phase original source catalogue differs from the accepted package')
    before = {relative: original.pin(ROOT / relative)['raw_sha256'] for relative in old_sources}
    original.require(before == old_sources, 'An accepted old source changed before compact extraction')
    original.require(original.pin(original.PARENT)['raw_sha256'] == original.PARENT_RAW
        and original.pin(original.PREVIOUS)['raw_sha256'] == original.PREVIOUS_RAW,
        'Accepted parent or genuine birth-memory raw anchor differs')
    parent = checked_json(original.PARENT, 'checkpoint_sha256', original.PARENT_SELF)
    previous = checked_json(original.PREVIOUS, 'fixture_sha256', original.PREVIOUS_SELF)
    runtime, geometry = original.authenticate_inputs(parent, previous)
    for path, raw, expected in ((original.MEMORY_READBACK, original.MEMORY_READBACK_RAW, original.MEMORY_READBACK_SELF),
                               (original.RELEASE_READBACK, original.RELEASE_READBACK_RAW, original.RELEASE_READBACK_SELF)):
        original.require(original.pin(path)['raw_sha256'] == raw, 'Previous public reconstruction readback differs')
        original.require(checked_json(path, 'sha256', expected)['status'] == 'PASS', 'Previous source readback is not PASS')
    release = checked_json(original.RELEASE_REPORT, 'report_sha256', original.RELEASE_REPORT_SELF)
    original.require(release['status'] == 'PASS' and release['parent_raw_sha256'] == original.PARENT_RAW
        and release['previous_memory_fixture_pin']['raw_sha256'] == original.PREVIOUS_RAW
        and 'accepted_Genesis_has_no_new_source_activation' in release['physical_refusals'],
        'Original no-activation source gate differs')
    monitor, monitor_pin = source_member(manifest, MONITOR, 'sha256')
    original.require(monitor['captured_source_union_count'] == 434
        and monitor['current_source_sha256'] == old_sources
        and monitor['numerical_convergence_status'] == 'OPEN' and len(monitor['cases']) == 3,
        'Compact literal comparison provenance differs')
    cases, minimal_pins, parent_blocks = [], [], None
    for index, prior in enumerate(previous['cases']):
        selected = {key: prior[key] for key in ('name', 'seed_definition', 'embedded_partition',
            'accepted_interval_reaction', 'source_memory_birth', 'birth_memory')}
        section, C, Q, e0, e1, source = original.bind_case(parent, selected)
        minimal, minimal_pin = source_member(manifest, f'{NEGATIVES}/original_source_case_{index:03d}.json', 'case_sha256')
        original.require(minimal['name'] == prior['name'] and minimal['source_inputs'] == source,
            'Source extraction differs from previously independently audited original inputs')
        minimal_pins.append(minimal_pin)
        if parent_blocks is None:
            parent_blocks = source['literal_original_parent_blocks']
        original.require(source['literal_original_parent_blocks'] == parent_blocks,
            'Original old-parent CSR differs between source cases')
        compact = {key: value for key, value in source.items() if key not in HEAVY_SOURCE_KEYS}
        compact['geometry_source'] = {key: source['section'][key] for key in GEOMETRY_KEYS}
        compact['original_source_memory_birth_sha256'] = source['original_source_memory_birth']['source_memory_birth_sha256']
        compact['original_accepted_interval_reaction_sha256'] = source['original_accepted_interval_reaction']['seed_reaction_sha256']
        observed = monitor['cases'][index]
        original.require(observed['descriptor']['name'] == prior['name']
            and observed['descriptor']['original_cohort_id'] == section.cohort_id,
            'Literal computational basis belongs to another original cohort')
        supports = {'fixed_100m_clipped_support': _support_packet(observed, 'fixed_100m_clipped_support',
            observed['fixed_100m_clipped_support'], source)}
        if index == 0:
            supports['original_radius_baseline'] = _support_packet(observed, 'original_radius_baseline',
                observed['original_radius_baseline'], source)
        else:
            original.require(observed['original_radius_baseline'] is None, 'Unexpected original radius baseline')
        cases.append(original.seal(dict(name=prior['name'], original_cohort_id=section.cohort_id,
            original_face_index=section.face_index, source_inputs=compact, supports=supports,
            source_minimal_archive_pin=minimal_pin), 'source_case_sha256'))
    memory_by_id = {item['id']: item for item in runtime['mechanics']['state']['cohorts']}
    memories = [memory_by_id[item['id']] for item in geometry['cohorts']]
    original.require(len(memories) == len(memory_by_id) == 800, 'Full original mechanics cohort coverage differs')
    shared = dict(parent_ndof=252, literal_original_parent_blocks=parent_blocks,
        original_parent_block_sha256={name: original.digest(value) for name, value in parent_blocks.items()},
        all_800_cohort_memory_sha256=original.digest(memories), all_800_cohort_memory_count=800,
        all_800_memory0_sha256=original.digest([item['strain0'] for item in memories]),
        all_800_memory1_sha256=original.digest([item['strain1'] for item in memories]),
        ordered_original_cohort_ids=[item['id'] for item in geometry['cohorts']],
        original_parent_mechanics_sha256=original.digest(runtime['mechanics']),
        old_parent_K_M_D_retained_without_reassembly=True, old_800_cohort_memory_retained_exactly=True)
    original.require(all(block['shape'] == [252, 252] for block in parent_blocks.values()), 'Old parent CSR shape differs')
    after = {relative: original.pin(ROOT / relative)['raw_sha256'] for relative in old_sources}
    original.require(after == before == old_sources and original.pin(original.PARENT)['raw_sha256'] == original.PARENT_RAW
        and original.pin(original.PREVIOUS)['raw_sha256'] == original.PREVIOUS_RAW,
        'Authoritative history or old source changed during extraction')
    pins = {name: dict(path=str(path), **original.pin(path)) for name, path in (
        ('accepted_parent', original.PARENT), ('genuine_birth_memory_fixture', original.PREVIOUS),
        ('clipped_package_manifest', PACKAGE / 'package_manifest.json'), ('clipped_final_readback', CLIPPED_READBACK),
        ('memory_final_readback', original.MEMORY_READBACK), ('release_final_readback', original.RELEASE_READBACK),
        ('release_source_gate_report', original.RELEASE_REPORT), ('phase_baseline', ORIGINALS / 'baseline_manifest.json'),
        ('phase_prior_sources', ORIGINALS / 'unchanged_previous_434_source_pins.json'))}
    return original.seal(dict(format=FORMAT, scope='original_birth_source_and_existing_virtual_bases_only_no_domain_integration',
        cases=cases, shared_source_retention=shared, source_pins=pins,
        compact_comparison_archive_pin=monitor_pin, original_minimal_source_pins=minimal_pins,
        previous_memory_fixture_sha256=original.PREVIOUS_SELF, parent_checkpoint_sha256=original.PARENT_SELF,
        formation_checkpoint_sha256=original.FORMATION, parent_raw_sha256=original.PARENT_RAW,
        source_identity=geometry['source_identity'], source_time_myr=geometry['time_myr'],
        cohort_count=800, accepted_interval_count=4, previous_source_count=434,
        previous_source_sha256=old_sources, all_previous_source_bytes_unchanged=True,
        source_inputs_checked_directly_against_raw_accepted_state=True,
        source_selected_prospective_law=None, actual_nucleation_candidate=None,
        physical_Gamma_selected=False, physical_creation_admitted=False, physical_release_admitted=False,
        physical_contraction_certified=False, available_for_creation_funding=False, funding_admitted=False,
        history_reinitialized=False, accepted_planet_memory_changed=False, actual_dynamic_reaction_replaced=False,
        automated_goal_enabled=False, point_one_complete=False, new_Genesis_time_steps=0,
        new_material_cohorts=0, creation_work_debited_from_Genesis_j=0.0, public_planetary_restore_count=0,
        clipped_point_shards_read=False, parent_matrices_reassembled=False,
        status='AUTHENTICATED_SOURCE_EXTRACTION_ONLY', numerical_convergence_status='NOT_EVALUATED',
        extraction_seconds=time.perf_counter()-started), 'source_fixture_sha256')


def support_source_inputs(case, support_name='fixed_100m_clipped_support'):
    """Provide the agreed flat raw numerical input, preserving original bytes."""
    original.check_seal(case, 'source_case_sha256')
    support = original.check_seal(case['supports'][support_name], 'support_sha256')
    return {**case['source_inputs'], 'seed_center_world_m': support['seed_center_world_m'],
            'seed_radius_m': support['seed_radius_m']}


def run(output):
    output = Path(output).resolve()
    original.require(output.is_relative_to(Path('/workspace')) and not output.is_relative_to(ROOT)
        and not output.exists(), 'Source extraction output must be fresh and external')
    output.mkdir(exist_ok=False)
    fixture = load_authentic_source_fixture()
    path = output / 'source_fixture.json'
    path.write_bytes(original.canonical(fixture) + b'\n')
    print(json.dumps(dict(status=fixture['status'], source_fixture=str(path), pin=original.pin(path),
        source_fixture_sha256=fixture['source_fixture_sha256'], extraction_seconds=fixture['extraction_seconds'],
        source_cases=3, computational_supports=4, clipped_point_shards_read=False,
        physical_contraction_certified=False, public_planetary_restore_count=0), sort_keys=True), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    run(parser.parse_args().output)
