"""Source authority and scope controls for the new initialization context."""
from dataclasses import replace
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path

import pytest

from tectonics import geometric_native_common_initialization_source as api

ROOT = Path(__file__).resolve().parents[1]
PATHS = (
    'analysis/geometric_native_common_general_force_validation/development_native03/native_control.json',
    'analysis/geometric_native_common_source_carrier_validation/release_validation_02/native_control/source_packet.json',
    'analysis/geometric_native_common_mixed_force_validation/release_validation_01/native_control/source_packet.json',
    'analysis/geometric_native_common_cold_material_carrier_validation/release_validation_02/native_control/source_packet.json',
)
PROVENANCE = 'Authenticated accepted original interval and explicit unaccepted reference935 initialization controls'


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',', ':'),
                            ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def fraction(raw):
    return F(*raw)


@pytest.fixture(scope='module')
def source():
    packets = tuple(json.loads((ROOT/path).read_bytes()) for path in PATHS)
    context = api.bind_native_common_initialization_source(*packets,provenance=PROVENANCE)
    return packets, context


def test_receipt_authorizes_outer_accepted_state_despite_frozen_inner_candidate_names(source):
    _, context = source
    accepted = api.native_common_initialization_source_evidence(context)['accepted_source']
    assert accepted['accepted_outer_state_sha256'] == '617cc7c5bd032cf521524aa5fd4ac6c10c56f9470455eb5c31ae059a09718f9f'
    receipt = accepted['receipt']
    assert receipt['sequence'] == 1
    assert receipt['receipt_sha256'] == digest({k:v for k,v in receipt.items() if k!='receipt_sha256'})
    assert receipt['source_history_sha256'] == accepted['original_history_dto_sha256']
    assert receipt['before_sha256'] == accepted['preceding_prepared_dto_sha256']
    assert receipt['admission_sha256'] == accepted['temporal_admission_sha256']
    assert receipt['after_material_epoch_sha256'] == accepted['accepted_material_epoch_sha256']
    assert accepted['pre_admission_history_flags']['candidate_mechanics_adopted'] is False
    assert accepted['outer_receipt_authorizes_original_interval'] is True


def test_accepted_clocks_and_frozen_reference_clocks_keep_different_meanings(source):
    _, context = source
    evidence = api.native_common_initialization_source_evidence(context)
    clocks = evidence['accepted_source']['accepted_state_clocks_s']
    assert {k:fraction(v) for k,v in clocks.items()} == dict(
        source_geometry=F(1000000),contact=F(2000000),memory=F(1000000),material_geometry=F(1000000))
    mechanical = evidence['source_mechanical_reference']
    assert fraction(mechanical['archival_reference_geometry_elapsed_time_s']) == 0
    assert fraction(mechanical['reconstructed_current_geometry_elapsed_time_s']) == 1000000
    assert evidence['accepted_source']['contact_step_count'] == 2
    assert evidence['accepted_source']['memory_stage_count'] == 1


def test_all_sixteen_literal_force_producers_and_both_reconstructed_bases_are_distinct(source):
    _, context = source
    mechanical = api.native_common_initialization_source_evidence(context)['source_mechanical_reference']
    assert len(mechanical['atoms']) == 16
    assert len({tuple(atom['original_source_work_lineage_id']) for atom in mechanical['atoms']}) == 16
    for atom in mechanical['atoms']:
        old, current = atom['archival_old_reference'], atom['current_objective']
        assert fraction(old['source_geometry_elapsed_time_s']) == 0
        assert fraction(current['source_geometry_elapsed_time_s']) == 1000000
        assert old['source_traction_world_reactions'] != current['source_traction_world_reactions']
        assert atom['archival_literal_reaction_replay_certified'] is True
        assert len(atom['archival_mathematical_minus_literal_reaction_bounds']) == 7
    assert len(set(mechanical['force_basis_descriptions'].values())) == 3


def test_all_original_material_attachments_and_internal_parent_links_survive(source):
    packets, context = source
    evidence = api.native_common_initialization_source_evidence(context)
    original_zero = next(row['response'] for row in packets[0]['queries']
                         if row['label']=='zero_source_zero_rates')
    original_sides = {}
    for raw in original_zero['contact_side_receipts_json']:
        original = json.loads(raw)
        original_sides[original['point_id'],original['side']] = original
    coordinate_order = evidence['target_reference']['zero_operator_excerpt']['coordinate_order']
    assert len(coordinate_order) == 101
    assert coordinate_order == original_zero['coordinate_order']

    def assert_original_sparse_jets(retained, original):
        if type(retained) is dict:
            if 'first_derivative_bounds' in retained:
                first = retained['first_derivative_bounds']
                assert first == original['first_derivative_bounds']
                assert all(type(pair) is list and len(pair)==2 and type(pair[0]) is int
                           for pair in first)
                indices = [pair[0] for pair in first]
                assert len(indices) == len(set(indices))
                assert all(0 <= index < len(coordinate_order) for index in indices)
                assert 'second_derivative_bounds' not in retained
            for name, value in retained.items():
                assert_original_sparse_jets(value,original[name])
        elif type(retained) is list:
            assert len(retained) == len(original)
            for value, source_value in zip(retained,original):
                assert_original_sparse_jets(value,source_value)

    sides = evidence['target_reference']['sides']
    assert len(sides) == 32
    assert len({(r['zero_side']['point_id'],r['zero_side']['side'])for r in sides}) == 32
    archived = [row for row in sides if row['source_side']['field']['original_label_membership']=='ARCHIVED_ORIGINAL_MATERIAL']
    assert len(archived) == 8
    for row in sides:
        field, side = row['source_side']['field'], row['zero_side']
        assert field['original_parent_id'] == side['original_parent_id'] == row['zero_point']['original_parent_id']
        assert field['original_material_id'] == side['original_material_id']
        assert field['residual_coordinate_index'] == side['internal_radial_coordinate_index']
        assert_original_sparse_jets(side,original_sides[side['point_id'],side['side']])


def test_accepted_work_is_not_relabelled_common_preparation_or_continuous_density(source):
    _, context = source
    evidence = api.native_common_initialization_source_evidence(context)
    accepted = evidence['accepted_source']
    ledgers = accepted['accepted_work_ledgers']
    assert ledgers['preparation_work_j'] == 4188790204786.3916
    assert ledgers['cumulative_source_work_j'] == 8.469879637103078e16
    assert ledgers['cumulative_basal_dissipation_j'] == 8.469878939936646e16
    assert ledgers['cumulative_bending_numerical_dissipation_j'] == 251841262157.0276
    assert accepted['original_preparation_work_is_common_lambda_preparation_work'] is False
    assert accepted['continuous_contact_work_added_to_mechanical_ledger'] is False
    assert len(accepted['states']) == 16
    assert len(accepted['spatial_memory']['traces']) == 4
    assert len(evidence['source_mechanical_reference']['ledger_candidate']) == 15


def test_beam_literals_keep_binary_rounding_residuals_against_mathematical_beams(source):
    _, context = source
    evidence = api.native_common_initialization_source_evidence(context)
    beams = evidence['source_mechanical_reference']['beams']
    assert len(beams) == 8
    assert {b['target_internal_coordinate_index']for b in beams} == set(range(92,100))
    assert len({b['original_parent_id']for b in beams}) == 8
    literal_sum = sum((fraction(b['literal_energy_j'])for b in beams),F(0))
    old_aggregate = F(dict(evidence['source_mechanical_reference']['ledger_candidate'])['bending_energy_increment_j'])
    assert literal_sum-old_aggregate == -F(3,131072)
    for beam in beams:
        k,w = fraction(beam['stiffness_n_m']),fraction(beam['source_displacement_m'])
        assert fraction(beam['mathematical_source_force_n']) == k*w
        assert fraction(beam['mathematical_source_energy_j']) == k*w*w/2


def test_known_full_packets_and_unselected_diagnostic_rows_are_authenticated(source):
    packets, _ = source
    packet = packets[0]
    previous = packet['queries'][-1]['label']
    packet['queries'][-1]['label'] = previous+' changed outside selected zero query'
    try:
        with pytest.raises(ValueError,match='complete source packet bytes'):
            api.bind_native_common_initialization_source(*packets,provenance=PROVENANCE)
    finally:
        packet['queries'][-1]['label'] = previous


def test_authenticated_packet_extraction_cannot_alias_the_callers_live_input(source):
    packets, _ = source
    packet = packets[1]
    snapshot = api._packet(packet,api.SOURCE_PINS[1])
    original = packet['carrier_system']['carriers'][0]['original_area_m2']
    packet['carrier_system']['carriers'][0]['original_area_m2'] = [1,1]
    try:
        assert snapshot['carrier_system']['carriers'][0]['original_area_m2'] == original
        assert snapshot['carrier_system']['carriers'][0]['original_area_m2'] != [1,1]
    finally:
        packet['carrier_system']['carriers'][0]['original_area_m2'] = original


@pytest.mark.parametrize('index', range(4))
def test_resealed_or_foreign_authority_cannot_replace_any_known_full_packet(source,index):
    packets, _ = source
    packet = packets[index]
    key = api.SOURCE_PINS[index][1]
    old = packet[key]
    packet[key] = '0'*64
    try:
        with pytest.raises(ValueError,match='known source packet seal'):
            api.bind_native_common_initialization_source(*packets,provenance=PROVENANCE)
    finally:
        packet[key] = old


def test_registered_common_probe_is_not_adopted_physical_initialization(source):
    _, context = source
    evidence = api.native_common_initialization_source_evidence(context)
    values = evidence['target_reference']['old_registered_coordinate_increments']
    assert len(values) == 100
    assert sum(fraction(v)!=0 for v in values) == 56
    assert evidence['physical_initialization_ready'] is False
    assert evidence['accepted_original_native_intervals'] == 1
    assert evidence['accepted_new_native_intervals'] == 0
    assert evidence['original_point_1_completed'] is False
    assert len(evidence['remaining_conditions']) == 3
    assert all(value is None for value in evidence['unknown_physical_inputs'].values())


def test_evidence_reads_are_isolated_and_compact_without_full_source_copies(source):
    _, context = source
    evidence = api.native_common_initialization_source_evidence(context)
    evidence['unknown_physical_inputs'][api.UNKNOWN_PHYSICAL_INPUTS[0]] = [0,1]
    evidence['accepted_source']['states'][0][1]['plastic_slip_m'] = 0
    fresh = api.native_common_initialization_source_evidence(context)
    assert fresh['unknown_physical_inputs'][api.UNKNOWN_PHYSICAL_INPUTS[0]] is None
    assert fresh['accepted_source']['states'][0][1]['plastic_slip_m'] != 0
    assert len(context.evidence_json) < 40000000
    assert all('source_context_json' not in bank for bank in fresh['target_reference']['whole_cold_banks'])


@pytest.mark.parametrize('field,value', (
    ('accepted_new_native_intervals',1),
    ('physical_initialization_ready',True),
    ('source_clock_state_memory_or_work_advanced',True),
    ('accepted_original_native_intervals',True),
))
def test_resealed_scope_or_boolean_counter_alias_has_no_registered_authority(source,field,value):
    _, context = source
    altered = replace(context,**{field:value})
    altered = replace(altered,binding_sha256=api._digest(api._body(altered)))
    with pytest.raises(ValueError,match='authenticated initialization source'):
        api.validate_native_common_initialization_source(altered)


def test_complete_context_restart_reauthenticates_sources(source):
    packets, context = source
    dto = json.loads(json.dumps(api.native_common_initialization_source_to_dict(context)))
    restored = api.native_common_initialization_source_from_dict(dto,*packets)
    assert api.native_common_initialization_source_to_dict(restored) == api.native_common_initialization_source_to_dict(context)
    assert restored is not context


def test_restart_rejects_even_selfsealed_zero_filled_missing_history(source):
    packets, context = source
    dto = api.native_common_initialization_source_to_dict(context)
    evidence = json.loads(dto['evidence_json'])
    evidence['unknown_physical_inputs']['historical_normal_damage_and_maximum_opening'] = [0,1]
    dto['evidence_json'] = api._json(evidence)
    dto['binding_sha256'] = digest({k:v for k,v in dto.items() if k not in ('binding_sha256','format')})
    with pytest.raises(ValueError,match='resealed initialization DTO'):
        api.native_common_initialization_source_from_dict(dto,*packets)


@pytest.mark.parametrize('value', ('', ' ', None, True))
def test_provenance_cannot_be_missing_or_retyped(source,value):
    packets, _ = source
    with pytest.raises(ValueError,match='explicit provenance'):
        api.bind_native_common_initialization_source(*packets,provenance=value)
