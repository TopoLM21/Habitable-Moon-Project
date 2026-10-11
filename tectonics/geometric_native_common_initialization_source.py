"""Authenticated source evidence for an unaccepted COMMON initialization.

The accepted original Gauss interval, its frozen force producer and later
COMMON diagnostic references are distinct authorities.  This module reads
their complete known packet seals and retains compact immutable evidence;
it neither replays the original solve nor fills absent physical fields.
"""
from dataclasses import dataclass, fields, replace
from fractions import Fraction
import hashlib
import json
import math

FORMAT = 'native-common-initialization-source-context-1'
SOURCE_PINS = (
    ('935', 'sha256', 'edb636a5e5c177d10f674c7424e0c0a76c57ae3e971c5bc69d99a89a57e8cf37',
     'a4f14ce71ecef16959685091a0c0b15d2a2e5c134a8023e9d3f98b8f8a7f0c0e'),
    ('867', 'packet_sha256', 'a46fe79aeabef70ec01672ad4c33c737bfc06d5323a112edcfeb2616f7b148a3',
     '265206430b4c864db32bce1be0a9a75b71fa180158f3d81e4f52131bebdf320c'),
    ('907', 'packet_sha256', 'bba7f1f4913d6bc47f5f41beff0a279be3c203993ddd0b0b06979c719abeda1a',
     '5e5555712e57fd69156ee475fe5feb161baef3cb96dd911a443228d160f53005'),
    ('895', 'packet_sha256', '27cbc05b687509fa22c00aa4339dd66d461c4def73ed37d4eae3acd62847da70',
     'fbafd092a4eb98472e60f79665e1e0fe8e9e1c2c2b9d3679344a32e59593cc89'),
)
UNKNOWN_PHYSICAL_INPUTS = (
    'historical_full_three_dimensional_cold_strain',
    'historical_full_three_dimensional_cold_prestress',
    'historical_full_three_dimensional_archive_placement_and_motion',
    'historical_normal_damage_and_maximum_opening',
    'physical_reference_preparation_path_and_paid_work',
    'physical_hot_birth_strain_and_prestress',
)
_CONTEXTS, _GUARDS = {}, {}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _json(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(',', ':'),
                          ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError('Initialization evidence needs finite canonical JSON values') from exc


def _digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _same(left, right, message):
    _require(_json(left) == _json(right), message)


def _fraction(raw):
    _require(type(raw) is list and len(raw) == 2 and all(type(x) is int for x in raw)
             and raw[1] > 0, 'A source fraction must have exact integer endpoints')
    result = Fraction(*raw)
    _require([result.numerator, result.denominator] == raw, 'A source fraction must be canonical')
    return result


def _literal(raw):
    _require(type(raw) in (int, float) and (type(raw) is int or math.isfinite(raw)),
             'A source literal must be finite and must not be a bool')
    return Fraction(raw)


def _rational(value):
    return [value.numerator, value.denominator]


def _packet(raw, pin):
    name, seal_key, seal, _ = pin
    _require(type(raw) is dict and raw.get(seal_key) == seal,
             'The complete known source packet seal changed: '+name)
    canonical = _json({k: v for k, v in raw.items() if k != seal_key}).encode()
    _require(hashlib.sha256(canonical).hexdigest() == seal,
             'The complete source packet bytes were altered: '+name)
    # Extraction reads precisely the bytes just authenticated.  The caller's
    # mutable dict cannot change authority after the complete packet hash.
    authenticated = json.loads(canonical)
    authenticated[seal_key] = seal
    return authenticated


def _compact(value):
    """Retain values and first variations, without dense second-jet matrices."""
    if type(value) is dict:
        return {k: _compact(v) for k, v in value.items() if k != 'second_derivative_bounds'}
    if type(value) is list:
        return [_compact(v) for v in value]
    return value


def _accepted_source(carrier):
    before = json.loads(dict(carrier['complete_source_dto_json'])['before'])
    snapshot, history = before['snapshot'], before['original_physical_history']
    candidate, previous = history['actual_candidate'], history['accepted_before']
    _require(before['format'] == 'native-implicit-temporal-original-Gauss-underthrust-motion-1',
             'The accepted outer source DTO was replaced by a diagnostic reference')
    _require(before['state_sha256'] == _digest(snapshot), 'The complete accepted source state hash changed')
    _require(history['binding_sha256'] == _digest({k: v for k, v in history.items() if k != 'binding_sha256'}),
             'The complete frozen original history binding changed')
    _same(json.loads(carrier['complete_original_physical_history_json']), history,
          'The mechanical carriers lost their complete original history')
    receipts = snapshot['receipts']
    _require(type(snapshot['accepted_loaded_native_intervals']) is int
             and snapshot['accepted_loaded_native_intervals'] == 1 and len(receipts) == 1,
             'Exactly one genuinely accepted original interval is required')
    receipt = receipts[0]
    _require(type(receipt['sequence']) is int and receipt['sequence'] == 1
             and receipt['receipt_sha256'] == _digest({k: v for k, v in receipt.items() if k != 'receipt_sha256'}),
             'The original accepted interval receipt is invalid')
    admission = before['temporal_admission']
    epoch, material = snapshot['current_material_epoch'], snapshot['current_material_epoch']['source_material_state']
    _require(receipt['source_history_sha256'] == _digest(history) == admission['source_history_sha256']
             and receipt['before_sha256'] == _digest(previous)
             and receipt['admission_sha256'] == _digest(admission) == snapshot['admission_sha256']
             and receipt['after_material_epoch_sha256'] == epoch['epoch_sha256'] == carrier['source_epoch_sha256'],
             'The accepted receipt does not bind its actual history, previous state, admission and material epoch')
    _require(epoch['epoch_sha256'] == _digest({k: v for k, v in epoch.items()
                    if k not in ('epoch_sha256', 'source_material_state')}),
             'The accepted material epoch seal changed')
    _same(material, history['material_state'], 'The accepted epoch substituted the original material transaction')
    _require(epoch['source_material_state_sha256'] == material['state_sha256'] == admission['material_state_sha256'],
             'The original material identity is not bound to temporal admission')
    _same(snapshot['states'], candidate['states'], 'The accepted states differ from the genuine original candidate')
    _same(snapshot['transverse_displacement_m'], candidate['transverse_displacement_m'],
          'The accepted radial values differ from the genuine original candidate')
    _same(snapshot['spatial_memory'], candidate['spatial_step']['memory'],
          'The accepted memory differs from the genuine original candidate')
    plain_anchors = [[row[0], row[1]['coefficients'], row[2]['coefficients']]
                     for row in candidate['spatial_step']['anchor_fields']]
    _same(snapshot['free_anchor_fields'], plain_anchors, 'The accepted free anchor fields were reset')
    _same(json.loads(carrier['complete_current_source_states_json']), snapshot['states'],
          'The mechanical carriers changed accepted source states')
    _same(json.loads(carrier['complete_current_source_memory_json']), snapshot['spatial_memory'],
          'The mechanical carriers changed accepted source memory')
    duration = _literal(candidate['dt_s'])
    _require(_literal(receipt['dt_s']) == duration == _literal(history['literal_dt_s'])
             and _literal(snapshot['elapsed_time_s']) == _literal(previous['snapshot']['elapsed_time_s'])+duration,
             'The accepted source duration and geometry clock are inconsistent')
    owners = history['original_reference']['plate_ids']
    rates = dict(candidate['omega_rad_per_myr'])
    old_histories = {row['owner']: row['history'] for row in previous['snapshot']['owner_histories']}
    expected_pose = [[owner, old_histories[owner]+[dict(dt_myr=history['literal_dt_myr'],
                                            omega_rad_per_myr=rates[owner])]] for owner in owners]
    _same(snapshot['owner_histories'], expected_pose, 'The accepted owner histories were replaced by a COMMON probe')
    _require(len(snapshot['states']) == 16 and len({row[0] for row in snapshot['states']}) == 16,
             'Every accepted original material atom is required')
    state_clocks = {_literal(state['elapsed_time_s']) for _, state in snapshot['states']}
    state_steps = {state['step_count'] for _, state in snapshot['states']}
    _require(state_clocks == {_fraction(carrier['source_contact_elapsed_time_s'])}
             and state_steps == {carrier['source_contact_step_count']}
             and all(type(state['step_count']) is int for _, state in snapshot['states']),
             'The complete contact clock or step count changed')
    _require(_literal(snapshot['elapsed_time_s']) == _fraction(carrier['current_geometry_elapsed_time_s'])
             and _literal(snapshot['spatial_memory']['elapsed_time_s']) == _fraction(carrier['source_memory_elapsed_time_s'])
             and snapshot['spatial_memory']['stage_count'] == carrier['source_memory_stage_count'],
             'Source geometry and memory clocks must retain their separate authority')
    candidate_ledger = dict(history['unaccepted_candidate_work_ledgers'])
    previous_snapshot = previous['snapshot']
    for output, incoming in (('cumulative_source_work_j', 'source_work_j'),
                            ('cumulative_basal_dissipation_j', 'basal_dissipation_j'),
                            ('cumulative_bending_numerical_dissipation_j', 'bending_numerical_dissipation_j')):
        # This source has a zero preceding cumulative ledger.  Exact dyadics
        # retain its original literal operation rather than renaming a budget.
        _require(_literal(previous_snapshot[output]) == 0
                 and _literal(snapshot[output]) == _literal(candidate_ledger[incoming]) == _literal(receipt[incoming]),
                 'The accepted global work receipt differs from its literal source ledger')
    _same(json.loads(carrier['complete_before_work_ledgers_json']), history['before_work_ledgers'],
          'The preceding work ledger was changed')
    _same(json.loads(carrier['complete_unaccepted_candidate_work_ledgers_json']), history['unaccepted_candidate_work_ledgers'],
          'The retained pre-admission candidate ledger was changed')
    _require(snapshot['continuous_contact_work_added_to_mechanical_ledger'] is False,
             'Continuous reconstructed work was added to literal Gauss work')
    evidence = dict(
        accepted_outer_state_sha256=before['state_sha256'], original_history_binding_sha256=history['binding_sha256'],
        original_history_dto_sha256=_digest(history), preceding_prepared_dto_sha256=_digest(previous),
        temporal_admission_sha256=_digest(admission), accepted_material_epoch_sha256=epoch['epoch_sha256'],
        accepted_material_state_sha256=material['state_sha256'], original_checkpoint_sha256=history['original_checkpoint_sha256'],
        receipt=receipt, original_literal_duration_s=_rational(duration),
        accepted_state_clocks_s=dict(source_geometry=carrier['current_geometry_elapsed_time_s'],
            contact=carrier['source_contact_elapsed_time_s'], memory=carrier['source_memory_elapsed_time_s'],
            material_geometry=carrier['current_geometry_elapsed_time_s']),
        contact_step_count=carrier['source_contact_step_count'], memory_stage_count=carrier['source_memory_stage_count'],
        owner_histories=snapshot['owner_histories'], transverse_displacement_m=snapshot['transverse_displacement_m'],
        states=snapshot['states'], spatial_memory=snapshot['spatial_memory'], free_anchor_fields=snapshot['free_anchor_fields'],
        original_material_state=material,
        accepted_work_ledgers={k:snapshot[k] for k in ('preparation_work_j','cumulative_source_work_j',
                        'cumulative_basal_dissipation_j','cumulative_bending_numerical_dissipation_j')},
        pre_admission_history_flags={k:history[k] for k in ('candidate_mechanics_adopted',
                    'incoming_pair_handover_authorized','mechanical_work_assigned_to_material_volume')},
        pre_admission_material_native_force_intervals_accepted=material['native_force_intervals_accepted'],
        outer_receipt_authorizes_original_interval=True,
        inner_candidate_names_are_retained_pre_admission_provenance=True,
        continuous_contact_work_added_to_mechanical_ledger=False,
        original_preparation_work_is_common_lambda_preparation_work=False,
    )
    return evidence, snapshot, history


def _extract(source935, source867, source907, source895):
    general, carrier = source935['general_system'], source867['carrier_system']
    cold = source895['cold_material_carrier_system']
    _require(source935['status'] == 'PASS_NATIVE935_DECLARED_GENERAL_STATE_OPERATOR'
             and source935['coordinate_count'] == 101 and source935['clocks_advanced'] is False,
             'The target must be the unchanged declared reference935')
    for name, pin in (('source907', SOURCE_PINS[2]), ('source895', SOURCE_PINS[3])):
        _require(source935['authority_reference'][name]['raw_sha256'] == pin[3],
                 'The target reference changed its full source authority')
    _require(general['cold_material_carrier_binding_sha256'] == cold['binding_sha256'],
             'The whole cold material chart differs from reference935')
    authorities = [row['authority'] for row in source907['contact_field_authorities']
                   if row['label'] == 'prepared_zero_cold_reference']
    _require(len(authorities) == 1, 'The whole prepared-zero original32 authority is required')
    authority = authorities[0]
    _require(authority['binding_sha256'] == general['contact_zero_authority_binding_sha256'],
             'The contact zero authority differs from reference935')
    zeros = [row['response'] for row in source935['queries'] if row['label'] == 'zero_source_zero_rates']
    _require(len(zeros) == 1, 'The full reference935 zero query is required')
    zero = zeros[0]
    _require(len(zero['coordinate_values']) == len(zero['coordinate_rates']) == 101
             and all(_fraction(v) == 0 for v in zero['coordinate_values']+zero['coordinate_rates']),
             'Reference935 must retain its exact zero state and zero rates')
    _same(zero['coordinate_order'], general['coordinate_order'], 'The zero coordinate catalogue changed')
    accepted, snapshot, history = _accepted_source(carrier)
    atom_map = {atom['point_id']: atom for atom in carrier['carriers']}
    _require(len(atom_map) == len(carrier['carriers']) == 16, 'All original sixteen mechanical carriers are required')
    original_map = {row['point_id']: row for row in history['quadrature_histories']}
    _require(set(original_map) == set(atom_map), 'An accepted original atom lost its source carrier')
    for point_id, atom in atom_map.items():
        _same(json.loads(atom['complete_original_atom_history_json']), original_map[point_id],
              'A mechanical carrier substituted its complete accepted atom ancestry')
        _same(json.loads(atom['source_local_state_json']), dict(snapshot['states'])[point_id],
              'An original carrier changed its accepted contact state')
        _same(json.loads(atom['archival_literal_world_reactions_json']),
              history['actual_candidate']['physical_points'][list(original_map).index(point_id)],
              'The actual accepted force producer was replaced by a current objective reconstruction')
        _require(atom['original_source_work_lineage_id'] == [history['original_checkpoint_sha256'], point_id],
                 'An original work lineage was reallocated')
    old_points = {(row['point_id'],row['side']): row for row in authority['point_receipts']}
    sides = []
    for raw in zero['contact_side_receipts_json']:
        side = json.loads(raw); key = side['point_id'], side['side']
        _require(key in old_points and key[1] in ('sub','over') and key[0] in atom_map,
                 'A target contact attachment has no complete original source authority')
        point, atom = old_points[key], atom_map[key[0]]
        saved, field = point['original_saved_attachment'], point['original_saved_attachment']['field']
        current = json.loads(side['complete_source_affine_field_json'])
        _same(current, point['current'], 'The target side substituted its actual full cold point')
        _same(current, point['prepared_zero'], 'The zero target authority is not its prepared-zero material point')
        parent = atom['donor_original_parent_id'] if key[1]=='sub' else atom['receiver_original_parent_id']
        owner = atom['sub_owner'] if key[1]=='sub' else atom['over_owner']
        _require(side['original_parent_id'] == field['original_parent_id'] == current['original_parent_id'] == parent
                 and side['owner'] == field['owner'] == current['owner'] == owner
                 and side['original_material_id'] == field['original_material_id'] == current['original_material_id'],
                 'The target side changed original parent, owner or material identity')
        _require(type(side['internal_radial_coordinate_index']) is int
                 and side['internal_radial_coordinate_index'] == field['residual_coordinate_index']
                 and 92 <= side['internal_radial_coordinate_index'] < 100,
                 'The original internal material mechanism was reassigned')
        _same(saved['original_ray'], [_rational(_literal(v)) for v in original_map[key[0]]['original_point']['position']],
              'The original literal material ray was normalized or changed')
        sides.append(dict(source_side=saved, zero_side=_compact(side), zero_point=current))
    _require(len(sides) == len(old_points) == 32 and len({(r['zero_side']['point_id'],r['zero_side']['side'])for r in sides}) == 32,
             'Exactly thirty-two authentic original attachments are required')
    target_atoms = []
    for raw in zero['contact_atoms_json']:
        atom = json.loads(raw); point_id = atom['point_id']; original = json.loads(atom['complete_original_atom_json'])
        _require(point_id in atom_map, 'An original source atom was born or dropped')
        _same(original, atom_map[point_id], 'The target contact replaced an original force/state/work carrier')
        _require(atom['source_original_area_included_once'] is True and atom['current_area_energy_multiplier_added'] is False
                 and atom['source_zero_traction_calibration_is_physical_prestress_handover'] is False,
                 'A discrete source contact was silently converted into physical continuum prestress')
        row = _compact(atom); row['original_atom'] = original; target_atoms.append(row)
    _require(len(target_atoms) == 16 and {r['point_id']for r in target_atoms} == set(atom_map),
             'All target contacts must keep their single original energy and work owner')
    internal_beams = source907['mixed_force_system']['internal_beams']
    _require(len(internal_beams) == 8, 'All eight original beam mechanisms are required')
    order14, radius = carrier['coordinate_order'], _fraction(carrier['source_radius_m'])
    radial = {row[0]:row[1:] for row in snapshot['transverse_displacement_m']}
    target_cold = source935['cold_energy_system']; parents = target_cold['original_parent_ids']
    beams = []
    for beam in internal_beams:
        original = json.loads(beam['complete_original_beam_json'])
        parent, trace, side = beam['original_parent_id'], beam['trace_id'], beam['side']
        index = order14.index(['radial',trace,side]); w = _literal(radial[trace][0 if side=='sub' else 1])
        stiffness = _literal(history['original_reference']['stiffness'][index][index])/(radius*radius)
        pi = parents.index(parent)
        _require(_fraction(beam['stiffness_exact_n_m']) == stiffness == _fraction(target_cold['tip_stiffnesses_n_m'][pi]),
                 'The accepted beam stiffness differs from its target material calibration')
        _same(beam['literal_reference_force_n'], target_cold['literal_reference_forces_n'][pi], 'The beam source force changed')
        _same(beam['literal_reference_energy_j'], target_cold['literal_reference_energies_j'][pi], 'The literal beam offset changed')
        _require(_fraction(original['reference_displacement_m']) == w,
                 'The beam displacement differs from the accepted original state')
        indices = {r['zero_side']['internal_radial_coordinate_index'] for r in sides
                   if r['zero_side']['original_parent_id']==parent}
        _require(indices == {beam['global_coordinate_index']}, 'The beam and material arm parent binding differ')
        beams.append(dict(original_parent_id=parent, trace_id=trace, side=side, source_coordinate_index=index,
            target_internal_coordinate_index=beam['global_coordinate_index'], target_tip_index=target_cold['tip_coordinate_indices'][pi],
            literal_force_n=beam['literal_reference_force_n'], literal_energy_j=beam['literal_reference_energy_j'],
            stiffness_n_m=_rational(stiffness), source_displacement_m=_rational(w),
            complete_original_beam=original, mathematical_source_force_n=_rational(stiffness*w),
            mathematical_source_energy_j=_rational(stiffness*w*w/2)))
    cold_zero, hot_zero = json.loads(zero['cold_response_json']), json.loads(zero['hot_response_json'])
    cold_fields = ('energy_j','beam_energy_j','nonlocal_orthogonal_energy_j','half_remainder_norm_j',
                   'remainder_projection_n','tip_projection_m','gradient57','preparation_fraction',
                   'whole_original_integral_certified','authentic_original_volume_included_once')
    hot_fields = ('energy_bounds_j','gradient_bounds_n','coordinate_increments','whole_original_hot_material_count',
                  'complete_hot30_operator_certified','physical_source_force_prestress_work_handover_certified',
                  'prepared_reference_full_system_work_certified')
    hot_selected = {k:hot_zero[k] for k in hot_fields}
    hot_selected['body_responses'] = [{k:_compact(v) for k,v in json.loads(raw).items()
                if 'hessian' not in k} for raw in hot_zero['body_responses_json']]
    banks = [{k:v for k,v in bank.items() if k!='source_context_json'} | dict(
             source_context_sha256=hashlib.sha256(bank['source_context_json'].encode()).hexdigest()) for bank in cold['banks']]
    _require(len(banks) == 8 and len({b['original_parent_id']for b in banks}) == 8,
             'The whole material reference must retain all eight original parent banks')
    return dict(
        authority=dict(known_packet_seals={p[0]:p[2] for p in SOURCE_PINS}, known_raw_artifact_sha256={p[0]:p[3] for p in SOURCE_PINS},
                       complete_known_packet_canonical_seals_authenticated=True, old_scientific_mains_executed=False),
        accepted_source=accepted,
        source_mechanical_reference=dict(coordinate_order=order14, radius_m=carrier['source_radius_m'], atoms=carrier['carriers'],
            beams=beams, ledger_before=json.loads(carrier['complete_before_work_ledgers_json']),
            ledger_candidate=json.loads(carrier['complete_unaccepted_candidate_work_ledgers_json']),
            carrier_system_binding_sha256=carrier['binding_sha256'],
            force_basis_descriptions=dict(literal_accepted_producer='LITERAL_BINARY64_FROZEN_ORIGINAL_GEOMETRY_ZERO',
                exact_archival_reference='EXACT_LITERAL_VECTOR_OPERATOR_WITH_SIGNED_BINARY64_RESIDUALS',
                current_objective='RECONSTRUCTED_OBJECTIVE_OPERATOR_AT_ACCEPTED_POSE_NOT_ACTUAL_PRODUCER'),
            archival_reference_geometry_elapsed_time_s=carrier['archival_geometry_elapsed_time_s'],
            reconstructed_current_geometry_elapsed_time_s=carrier['current_geometry_elapsed_time_s']),
        target_reference=dict(sides=sides, atoms=target_atoms, source_clocks_s=general['source_clocks_s'],
            whole_domain_certificate=cold['whole_domain_certificate'], coordinate_envelope_abs_m=cold['coordinate_envelope_abs_m'],
            whole_cold_banks=banks, preparation_binding_sha256s=general['preparation_binding_sha256s'],
            old_registered_coordinate_order=source907['mixed_force_system']['original_coordinate_order'],
            old_registered_coordinate_increments=source907['mixed_force_system']['original_bound_coordinate_increments'],
            zero_operator_excerpt=dict(coordinate_order=zero['coordinate_order'], coordinate_values=zero['coordinate_values'],
                coordinate_rates=zero['coordinate_rates'], conservative_gradient_bounds=zero['conservative_gradient_bounds'],
                energy_bounds_j=zero['energy_bounds_j'], energy_partitions=json.loads(zero['energy_partitions_json']),
                cold_response_selected={k:cold_zero[k] for k in cold_fields}, hot_response_selected=hot_selected)),
        unknown_physical_inputs={name:None for name in UNKNOWN_PHYSICAL_INPUTS},
        remaining_conditions=general['remaining_conditions'], accepted_original_native_intervals=1, accepted_new_native_intervals=0,
        physical_initialization_ready=False, historical_archived_three_dimensional_prestress_recovered=False,
        original_point_1_completed=False, original_point_1_remaining_work_blocks=1,
        no_source_clock_state_memory_or_work_advanced=True,
    )


@dataclass(frozen=True, slots=True)
class NativeCommonInitializationSource:
    evidence_json: str
    source_packet_seals: tuple
    provenance: str
    binding_sha256: str
    accepted_original_native_intervals: int = 1
    accepted_new_native_intervals: int = 0
    physical_initialization_ready: bool = False
    source_clock_state_memory_or_work_advanced: bool = False


def _body(context):
    return {f.name:getattr(context,f.name) for f in fields(context) if f.name!='binding_sha256'}


def bind_native_common_initialization_source(source935, source867, source907, source895, *, provenance):
    _require(type(provenance) is str and bool(provenance.strip()), 'Initialization source needs explicit provenance')
    packets = [_packet(raw,pin) for raw,pin in zip((source935,source867,source907,source895),SOURCE_PINS)]
    evidence = _extract(*packets)
    context = NativeCommonInitializationSource(_json(evidence), tuple((p[0],p[2])for p in SOURCE_PINS), provenance, '')
    context = replace(context, binding_sha256=_digest(_body(context)))
    _CONTEXTS[id(context)] = context
    _GUARDS[id(context)] = _json(_body(context))
    return context


def validate_native_common_initialization_source(context):
    _require(type(context) is NativeCommonInitializationSource and _CONTEXTS.get(id(context)) is context,
             'A complete authenticated initialization source context is required')
    _require(context.binding_sha256 == _digest(_body(context)) and _GUARDS[id(context)] == _json(_body(context)),
             'An altered or resealed initialization context has no source authority')
    return context


def native_common_initialization_source_evidence(context):
    return json.loads(validate_native_common_initialization_source(context).evidence_json)


def native_common_initialization_source_to_dict(context):
    return dict(format=FORMAT, **_body(validate_native_common_initialization_source(context)),
                binding_sha256=context.binding_sha256)


def native_common_initialization_source_from_dict(raw, source935, source867, source907, source895):
    _require(type(raw) is dict and raw.get('format') == FORMAT, 'A complete initialization context DTO is required')
    context = bind_native_common_initialization_source(source935,source867,source907,source895,provenance=raw.get('provenance'))
    _same(raw, native_common_initialization_source_to_dict(context), 'A resealed initialization DTO changed source evidence or scope')
    return context
