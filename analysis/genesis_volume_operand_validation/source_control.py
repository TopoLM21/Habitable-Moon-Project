"""Authenticate compact original Genesis lines for an additive operand audit.

The accepted full birth-memory and parent raw bytes remain authoritative.
The unchanged preceding exact-lambda archive supplies literal compact inputs,
four radial witnesses, and the previously audited computational bases. This
reader performs no numerical assembly, replay, restoration, or initialization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT/'analysis/genesis_volume_exact_lambda_validation/evidence'
MANIFEST_PIN = dict(bytes=1229699, raw_sha256='e3d4ec76e0a3681488db9a664a4137ffccdea5875b2692372968b52f1a245d90')
MANIFEST_SELF = '1d29c40bb70c7009312e4f84a62013489ebc0cfef226e081fd2bdea536834c5a'
READBACK = Path('/workspace/genesis_volume_exact_lambda_final_readback_20261011.json')
READBACK_SELF = 'd06359423c2a324ad5c662cc17b9385746fc766f50b304b056cb5550d1f6c669'
PARENT_RAW = '62579a0bad128830a66e2f44579cf2b1f5d3835b691225b8d431102f3e4486d7'
BIRTH_RAW = 'a98ff7cc89aeb59eb004c05b8ac87f75f2e39a4f10d4454fdf93c288153c9c0e'
FORMAT = 'genesis-volume-operand-authenticated-compact-inputs-0.1'
SUPPORTS = ((0,'fixed_100m_clipped_support'), (0,'original_radius_baseline'),
            (1,'fixed_100m_clipped_support'), (2,'fixed_100m_clipped_support'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),ensure_ascii=False,allow_nan=False).encode()


def seal(value, key='sha256'):
    require(key not in value, 'Body already has requested seal')
    return {**value,key:hashlib.sha256(canonical(value)).hexdigest()}


def check_seal(value, key='sha256', expected=None):
    require(isinstance(value,dict) and isinstance(value.get(key),str),'Missing JSON seal '+key)
    body = dict(value)
    actual = body.pop(key)
    require(hashlib.sha256(canonical(body)).hexdigest() == actual and (expected is None or actual == expected),
            'JSON self seal differs: '+key)
    return value


def pin(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(),'Missing or symlink literal file: '+str(path))
    size,digest = 0,hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(1024*1024):
            size += len(chunk)
            digest.update(chunk)
    return dict(bytes=size,raw_sha256=digest.hexdigest())


def checked_json(path, key='sha256', expected=None):
    return check_seal(json.loads(Path(path).read_bytes()),key,expected)


def _member(manifest, relative, key):
    entries = [entry for entry in manifest['files'] if entry['path'] == relative]
    require(len(entries) == 1,'Absent or duplicate compact source member: '+relative)
    expected = {name:entries[0][name] for name in ('bytes','raw_sha256')}
    path = PACKAGE/relative
    require(pin(path) == expected,'Compact source raw pin differs: '+relative)
    value = checked_json(path,key)
    return value,dict(path=str(path),**expected,self_seal_key=key,self_sha256=value[key])


def load_authenticated_inputs():
    """Return literal source fixture and four archived source/line pairs."""
    started = time.perf_counter()
    require(pin(PACKAGE/'package_manifest.json') == MANIFEST_PIN,'Prior manifest raw anchor differs')
    manifest = checked_json(PACKAGE/'package_manifest.json',expected=MANIFEST_SELF)
    readback = checked_json(READBACK,expected=READBACK_SELF)
    require(readback['status'] == 'PASS' and readback['source_count'] == 447
            and readback['package_manifest'] == MANIFEST_PIN
            and readback['package_manifest_self_sha256'] == MANIFEST_SELF
            and readback['all_metadata_checks_pass'] is True
            and readback['automatic_goal_enabled'] is False,'Prior source readback differs')
    sources = manifest['source_sha256']
    require(len(sources) == 447,'Prior current-source union differs')
    before = {relative:pin(ROOT/relative)['raw_sha256'] for relative in sources}
    require(before == sources,'Accepted old source changed before extraction')
    # Stream each authoritative large history file exactly once here. Their
    # unchanged raw hashes bind the compact previously audited source DTOs.
    raw_history = {}
    for name,expected_raw in (('accepted_parent',PARENT_RAW),('genuine_birth_memory_fixture',BIRTH_RAW)):
        reference = manifest['external_references'][name]
        actual = pin(Path(reference['path']))
        require(actual == {key:reference[key] for key in ('bytes','raw_sha256')}
                and actual['raw_sha256'] == expected_raw,'Authoritative Genesis history raw pin differs: '+name)
        raw_history[name] = dict(path=reference['path'],**actual)
    fixture,fixture_pin = _member(manifest,'actual_control/source_fixture.json','source_fixture_sha256')
    require(fixture['parent_raw_sha256'] == PARENT_RAW and fixture['cohort_count'] == 800
            and fixture['source_time_myr'] == 0.8684000000000005
            and fixture['accepted_interval_count'] == 4 and len(fixture['cases']) == 3
            and fixture['previous_source_count'] == 434
            and fixture['shared_source_retention']['all_800_cohort_memory_count'] == 800
            and fixture['shared_source_retention']['all_800_cohort_memory_sha256']
                == '1216a018a75ca788b718d774f2159cdc754222ec3d0b037aa25af390ea09c1bc',
            'Compact genuine source coverage or original 434 audit identity differs')
    require(fixture['automated_goal_enabled'] is False and fixture['history_reinitialized'] is False
            and fixture['accepted_planet_memory_changed'] is False and fixture['new_Genesis_time_steps'] == 0
            and fixture['new_material_cohorts'] == 0 and fixture['public_planetary_restore_count'] == 0
            and fixture['physical_contraction_certified'] is False
            and fixture['available_for_creation_funding'] is False,'Compact source acquired new physics/history')
    supports,report_pins = [],[]
    for case_index,label in SUPPORTS:
        case = check_seal(fixture['cases'][case_index],'source_case_sha256')
        support = check_seal(case['supports'][label],'support_sha256')
        require(case['original_cohort_id'] == (1,41,80)[case_index]
                and case['original_face_index'] == (0,40,79)[case_index]
                and support['original_declared_parent_basis'] is None
                and len(support['original_declared_added_basis']) == 18
                and all(len(row) == 15 for row in support['original_declared_added_basis'])
                and len(support['declared_probes']) == 5,'Original case/bases/probe coverage differs')
        relative = f'actual_control/case_{case_index:03d}_{label}.json'
        report,report_pin = _member(manifest,relative,'support_report_sha256')
        require(report['source_fixture_sha256'] == fixture['source_fixture_sha256']
                and report['source_case_sha256'] == case['source_case_sha256']
                and report['source_support_sha256'] == support['support_sha256']
                and report['original_computational_support'] == support
                and report['cohort_id'] == case['original_cohort_id']
                and report['support'] == label,'Archived line belongs to another genuine input')
        line = check_seal(report['line_witness'],'archive_sha256')
        source = check_seal(report['source'],'archive_sha256')
        require(line['source_binding_sha256'] == source['source_binding_sha256']
                and len(line['component_names']) == len(line['values']) == 2534
                and line['geometry_operand_accuracy_certified'] is False
                and line['physical_contraction_certified'] is False
                and report['physical_creation_admitted'] is False
                and report['physical_release_admitted'] is False
                and report['original_whole_parent_K_M_byte_equal'] is True
                and report['accepted_steps'] == 0,'Original witness arithmetic/scope differs')
        require(len(report['line_witness_barycentric_exact_binary_ratios']) == 3,
                'Original exact angular source chart is absent')
        supports.append(dict(case_index=case_index,label=label,case=case,support=support,
                             report=report,line_witness=line,archived_source=source,report_pin=report_pin))
        report_pins.append(report_pin)
    after = {relative:pin(ROOT/relative)['raw_sha256'] for relative in sources}
    require(after == before == sources and pin(PACKAGE/'package_manifest.json') == MANIFEST_PIN,
            'Accepted old source changed during extraction')
    authentication = seal(dict(format=FORMAT,prior_package_manifest=MANIFEST_PIN,
        prior_package_manifest_self_sha256=MANIFEST_SELF,prior_readback=dict(path=str(READBACK),**pin(READBACK),self_sha256=READBACK_SELF),
        previous_source_count=447,previous_source_sha256=sources,all_previous_source_bytes_unchanged=True,
        compact_fixture_original_previous_source_count=434,
        compact_source_fixture_pin=fixture_pin,compact_line_report_pins=report_pins,
        authoritative_raw_history_pins=raw_history,source_time_myr=fixture['source_time_myr'],
        cohort_count=800,accepted_interval_count=4,computational_support_count=4,
        all_800_cohort_memory_sha256=fixture['shared_source_retention']['all_800_cohort_memory_sha256'],
        scope='Source authentication only; original radial witnesses retained without numerical reclassification.',
        full_previous_package_payloads_reread=False,clipped_point_shards_read=False,
        history_reinitialized=False,history_advanced=False,public_planetary_restore_count=0,
        automated_goal_enabled=False,physical_contraction_certified=False,
        physical_creation_admitted=False,physical_release_admitted=False,available_for_creation_funding=False,
        extraction_seconds=time.perf_counter()-started))
    return dict(source_fixture=fixture,supports=supports,authentication=authentication)


def support_source_inputs(case,support_name='fixed_100m_clipped_support'):
    check_seal(case,'source_case_sha256')
    support = check_seal(case['supports'][support_name],'support_sha256')
    return {**case['source_inputs'],'seed_center_world_m':support['seed_center_world_m'],
            'seed_radius_m':support['seed_radius_m']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    output = parser.parse_args().output.resolve()
    require(output.is_relative_to(Path('/workspace')) and not output.is_relative_to(ROOT) and not output.exists(),
            'Authentication output must be fresh and external')
    output.mkdir()
    inputs = load_authenticated_inputs()
    (output/'authentication.json').write_bytes(canonical(inputs['authentication'])+b'\n')
    # Preserve the existing DTO seal; do not manufacture a replacement source.
    (output/'source_fixture.json').write_bytes(canonical(inputs['source_fixture'])+b'\n')
    print(json.dumps(dict(status='AUTHENTICATED_SOURCE_ONLY',authentication_sha256=inputs['authentication']['sha256'],
                         extraction_seconds=inputs['authentication']['extraction_seconds'],supports=4),sort_keys=True))


if __name__ == '__main__':
    main()
