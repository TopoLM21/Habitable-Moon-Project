"""Authenticate retained Genesis inputs and preceding selected operand proofs.

The prior frozen source reader retains the original four exact-lambda lines
and streams each authoritative raw history once. This additive layer binds
those same literals to the latest 458-source operand package and its selected
proofs. It performs no replay, restore, initialization, or history advance.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
from pathlib import Path
import time

from analysis.genesis_volume_operand_validation.source_control import (
    canonical, seal, check_seal, pin, checked_json, require,
    load_authenticated_inputs as _load_original_lines,
    support_source_inputs,
)

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT/'analysis/genesis_volume_operand_validation/evidence'
MANIFEST_PIN = dict(bytes=749993,raw_sha256='dcd532e19b09c03f6c511d0204781a4d654f8f9798cff4ef1300f225e7131cc4')
MANIFEST_SELF = '125c27427e1561165612e7f888f459c92efe328ed088a0c12e9b8df1d7cc6dcd'
READBACK = Path('/workspace/genesis_volume_operand_final_readback_20261011.json')
READBACK_SELF = 'f505a98280b1cfba7fd93028fccebbbca7e243011c62ab61dad203919f9110a9'
PROOF_KEYS = dict(coefficient_proof='coefficient_proof_sha256',
    geometry_proof='geometry_proof_sha256',directed_bounds='directed_bounds_sha256',
    contraction_certificate='certificate_sha256',case_report='case_report_sha256')


def _member(manifest,relative,key):
    entries = [entry for entry in manifest['files'] if entry['path'] == relative]
    require(len(entries) == 1,'Absent or duplicate operand member: '+relative)
    expected = {name:entries[0][name] for name in ('bytes','raw_sha256')}
    path = PACKAGE/relative
    require(pin(path) == expected,'Operand member raw pin differs: '+relative)
    value = checked_json(path,key)
    return value,dict(path=str(path),**expected,self_seal_key=key,self_sha256=value[key])


def load_authenticated_inputs():
    """Return the same genuine fixture, four original lines, and prior proofs."""
    started = time.perf_counter()
    require(pin(PACKAGE/'package_manifest.json') == MANIFEST_PIN,'Operand manifest raw anchor differs')
    manifest = checked_json(PACKAGE/'package_manifest.json',expected=MANIFEST_SELF)
    readback = checked_json(READBACK,expected=READBACK_SELF)
    require(readback['status'] == 'PASS' and readback['source_count'] == 458
        and readback['prior_source_count'] == 447 and readback['package_manifest'] == MANIFEST_PIN
        and readback['package_manifest_self_sha256'] == MANIFEST_SELF
        and readback['all_metadata_checks_pass'] is True
        and readback['automatic_goal_enabled'] is False,'Operand metadata readback differs')
    sources = manifest['source_sha256']
    require(len(sources) == 458 and manifest['prior_source_count'] == 447,'Frozen 458-source union differs')
    before = {name:pin(ROOT/name)['raw_sha256'] for name in sources}
    require(before == sources,'A prior source changed before extraction')
    original = _load_original_lines()
    fixture,fixture_pin = _member(manifest,'actual_control/source_fixture.json','source_fixture_sha256')
    require(fixture == original['source_fixture'],'Latest compact fixture replaced original Genesis source')
    prior_auth,prior_auth_pin = _member(manifest,'actual_control/authentication.json','sha256')
    require(prior_auth['previous_source_count'] == 447
        and prior_auth['prior_package_manifest_self_sha256'] == original['authentication']['prior_package_manifest_self_sha256']
        and prior_auth['compact_source_fixture_pin'] == original['authentication']['compact_source_fixture_pin']
        and prior_auth['compact_line_report_pins'] == original['authentication']['compact_line_report_pins']
        and prior_auth['authoritative_raw_history_pins'] == original['authentication']['authoritative_raw_history_pins']
        and prior_auth['automated_goal_enabled'] is False,'Prior operand source chain differs')
    old_control,control_pin = _member(manifest,'actual_control/control_report.json','report_sha256')
    require(old_control['selected_line_count'] == len(old_control['cases']) == 4
        and old_control['source_fixture_sha256'] == fixture['source_fixture_sha256']
        and old_control['source_authentication_sha256'] == prior_auth['sha256']
        and old_control['selected_operand_proof_status'] == 'PASS'
        and old_control['angular_accuracy_certified'] is False
        and old_control['full_contraction_uniform_forward_bounds_certified'] is False
        and old_control['source_geometry_displacement_propagated'] is False
        and old_control['physical_creation_admitted'] is False
        and old_control['physical_release_admitted'] is False
        and old_control['new_Genesis_time_steps'] == 0,'Prior operand control scope differs')
    proof_pins=[]
    for ordinal,item in enumerate(original['supports']):
        prefix=f"actual_control/case_{ordinal:03d}_{item['label']}"
        proofs,pins={},{}
        for stem,key in PROOF_KEYS.items():
            proofs[stem],pins[stem]=_member(manifest,prefix+'/'+stem+'.json',key)
        case=proofs['case_report']
        require(case == old_control['cases'][ordinal]
            and case['original_report_pin'] == item['report_pin']
            and case['original_cohort_id'] == item['case']['original_cohort_id']
            and case['original_face_index'] == item['case']['original_face_index']
            and case['support'] == item['label']
            and [Fraction(int(n),int(d)) for n,d in case['original_N_exact_binary_ratios']]
                == [Fraction(v) for v in item['report']['line_witness_barycentric_exact_binary_ratios']],
            'Prior selected proof belongs to another original line')
        require(case['geometry_source_error_enclosures_certified'] is True
            and case['angular_accuracy_certified'] is False
            and case['source_geometry_displacement_propagated'] is False
            and case['physical_creation_admitted'] is False
            and case['physical_release_admitted'] is False,'Prior proof scope reclassified')
        item['prior_operand_proofs']=proofs
        item['prior_operand_proof_pins']=pins
        proof_pins.append(dict(case_ordinal=ordinal,support=item['label'],members=pins))
    after={name:pin(ROOT/name)['raw_sha256'] for name in sources}
    require(after == before == sources and pin(PACKAGE/'package_manifest.json') == MANIFEST_PIN,
        'A frozen prior source changed during extraction')
    authentication=seal(dict(format='genesis-volume-integral-authenticated-compact-inputs-0.1',
        prior_package_manifest=MANIFEST_PIN,prior_package_manifest_self_sha256=MANIFEST_SELF,
        prior_readback=dict(path=str(READBACK),**pin(READBACK),self_sha256=READBACK_SELF),
        previous_source_count=458,previous_source_sha256=sources,all_previous_source_bytes_unchanged=True,
        compact_fixture_original_previous_source_count=434,compact_source_fixture_pin=fixture_pin,
        compact_line_report_pins=original['authentication']['compact_line_report_pins'],
        preceding_operand_authentication_pin=prior_auth_pin,preceding_operand_control_pin=control_pin,
        preceding_selected_operand_proof_pins=proof_pins,
        original_line_authentication=original['authentication'],
        authoritative_raw_history_pins=original['authentication']['authoritative_raw_history_pins'],
        source_time_myr=fixture['source_time_myr'],cohort_count=800,accepted_interval_count=4,
        computational_support_count=4,all_800_cohort_memory_sha256=prior_auth['all_800_cohort_memory_sha256'],
        scope='Source and prior selected proof authentication only; original radial witnesses retained without numerical reclassification.',
        full_previous_package_payloads_reread=False,clipped_point_shards_read=False,
        history_reinitialized=False,history_advanced=False,public_planetary_restore_count=0,
        automated_goal_enabled=False,physical_contraction_certified=False,
        physical_creation_admitted=False,physical_release_admitted=False,available_for_creation_funding=False,
        extraction_seconds=time.perf_counter()-started))
    return dict(source_fixture=fixture,supports=original['supports'],authentication=authentication)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    output=parser.parse_args().output.resolve()
    require(output.is_relative_to(Path('/workspace')) and not output.is_relative_to(ROOT) and not output.exists(),
        'Authentication output must be fresh and external')
    output.mkdir()
    inputs=load_authenticated_inputs()
    (output/'authentication.json').write_bytes(canonical(inputs['authentication'])+b'\n')
    (output/'source_fixture.json').write_bytes(canonical(inputs['source_fixture'])+b'\n')
    print('Authenticated the same four original Genesis lines; no numerical or physical reclassification.')


if __name__ == '__main__':
    main()
