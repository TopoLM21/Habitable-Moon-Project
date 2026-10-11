"""Necessary preparation work for the authenticated, declared COMMON101 law.

The bound relaxes every mechanical coordinate.  It uses the zero modal
complement and exact completion of all eight original beam squares; it does
not replay an integral, find an equilibrium, or certify a physical path.
"""
from dataclasses import dataclass, fields, replace
from fractions import Fraction as F
import hashlib
import json
import math

FORMAT = 'native-common-preparation-feasibility-system-1'
RESPONSE_FORMAT = 'native-common-preparation-feasibility-response-1'
MEASURE = 'AUTHENTIC_ORIGINAL_AREAL_VOLUME_R_SQUARED_DOMEGA_DZ_ONCE'
_SYSTEMS, _GUARDS = {}, {}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _fraction(raw):
    _require(type(raw) is list and len(raw) == 2 and all(type(x) is int for x in raw)
             and raw[1] > 0, 'A source scalar must retain its exact rational type')
    value = F(*raw)
    _require([value.numerator, value.denominator] == raw, 'Source fractions must be canonical')
    return value


def _literal(raw):
    _require(type(raw) in (int, float) and (type(raw) is int or math.isfinite(raw)),
             'An original material literal must be a finite number, never a bool')
    return F(raw)


def _interval(raw):
    _require(type(raw) is dict and set(raw) == {'lo', 'hi'}, 'An interval needs both original endpoints')
    lo, hi = _fraction(raw['lo']), _fraction(raw['hi'])
    _require(lo <= hi, 'A source interval is reversed')
    return lo, hi


def _payload(x):
    if type(x) is F:
        return [x.numerator, x.denominator]
    if hasattr(type(x), '__dataclass_fields__'):
        return {f.name: _payload(getattr(x, f.name)) for f in fields(x)}
    if type(x) in (tuple, list):
        return [_payload(v) for v in x]
    if type(x) is dict:
        return {k: _payload(v) for k, v in x.items()}
    if type(x) in (str, int, bool) or x is None:
        return x
    if type(x) is float and math.isfinite(x):
        # Complete original atom provenance contains authentic binary64
        # literals. Evaluated bound inputs/results themselves are Fractions.
        return x
    raise ValueError('An undeclared feasibility DTO type was supplied')


def _json(raw):
    return json.dumps(_payload(raw), sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False)


def _seal(value):
    return hashlib.sha256(_json({f.name: getattr(value, f.name) for f in fields(value)
                                if f.name != 'binding_sha256'}).encode()).hexdigest()


def _power_api():
    from . import geometric_native_common_basal_work_bound as api
    return api


def _extract_preparation_energy_evidence(source935, source899, source923, source903):
    """Extract theorem premises AFTER the common port authenticates all packets.

    These cross checks retain the whole support, actual inverse Gram, source
    prestress literals, contact offsets and original parent/internal binding.
    This private extractor is not a replacement for raw packet authentication.
    """
    cold = source935['cold_energy_system']
    general = source935['general_system']
    continuum = source899['cold_continuum_energy_system']
    integral = source923['integral']
    hot = source903['hot_mean_system']
    _require(cold['continuum_system_binding_sha256'] == continuum['binding_sha256']
             == integral['continuum_system_binding_sha256'], 'The whole cold material binding changed')
    _require(cold['cold_carrier_binding_sha256'] == integral['cold_system_binding_sha256']
             == continuum['cold_material_carrier_binding_sha256'], 'The authentic cold support changed')
    _require(cold['archive_carrier_model_sha256'] == integral['archive_carrier_model_sha256']
             == continuum['archive_carrier_model_sha256'], 'The actual donor collar changed')
    _require(cold['preparation_binding_sha256s'] == integral['preparation_binding_sha256s']
             == general['preparation_binding_sha256s'], 'The prepared reference changed')
    _require(cold['binding_sha256'] == general['cold_energy_system_binding_sha256']
             and hot['binding_sha256'] == general['hot_mean_system_binding_sha256'],
             'The general operator changed a constitutive port')
    _require(cold['material_reference_measure'] == integral['material_reference_measure']
             == continuum['material_reference_measure'] == MEASURE, 'The original material measure changed')
    for name in ('whole_original_integral_certified', 'authentic_original_volume_included_once',
                 'complete_modal_zero_clamp_complement_integrated',
                 'inactive_complement_tip_strain_identically_zero',
                 'isotropic_radial_twist_cubic_moment_exactly_zero',
                 'original_gram_matrix_and_inverse_reused_without_quadrature',
                 'complete_original_live_and_archive_material_retained'):
        _require(integral[name] is True, 'A full orthogonal-support theorem premise is missing: '+name)
    _require(source935['coupled_cold_whole_original_volume_integral_certified'] is True
             and source935['frozen_owner_load_potential_used'] is False,
             'A frozen owner potential or incomplete cold support cannot enter the work theorem')
    banks = continuum['banks']
    parents = [b['original_parent_id'] for b in banks]
    _require(len(banks) == 8 and len(set(parents)) == 8 and parents == cold['original_parent_ids']
             == integral['original_parent_ids'], 'The eight original physical parents changed')
    energies = [b['source_reference_energy_j'] for b in banks]
    forces = [b['source_reference_force_n'] for b in banks]
    stiffnesses = [b['source_old_tip_stiffness_n_m'] for b in banks]
    _require(energies == cold['literal_reference_energies_j']
             and forces == cold['literal_reference_forces_n']
             and stiffnesses == cold['tip_stiffnesses_n_m'], 'A literal beam ledger was rounded or substituted')
    _require(all(_fraction(k) > 0 for k in stiffnesses), 'Every original beam stiffness must be positive')
    gram = source899['evaluation_queries'][0]['response']['gram_inverse_certificate']
    _require(gram['positive_definite_certified'] is True and gram['global_eight_tip_overlap_retained'] is True
             and gram['inverse_enclosure_method'] == 'ACTUAL_INTERIOR_GRAM_COERCIVITY_AND_RIGHT_RESIDUAL_NORM',
             'The global orthogonal projection needs its actual positive Gram and certified inverse')
    _require(len(gram['gram_bounds_n_m']) == len(gram['inverse_gram_bounds_m_n']) == 8,
             'The full eight-tip Gram cannot be replaced by a diagonal approximation')
    for query in source899['evaluation_queries']:
        _require(_json(query['response']['gram_inverse_certificate']) == _json(gram),
                 'The source Gram certificate changed between queries')
    tensor_rows, clamp_rows = [], []
    for bank in banks:
        E, nu = _fraction(bank['upper_young_modulus_pa']), _fraction(bank['upper_poisson_ratio'])
        mu = E/(2*(1+nu)); lame = E*nu/((1+nu)*(1-2*nu))
        lowE, lowMu = _fraction(bank['lower_young_modulus_pa']), _fraction(bank['lower_shear_modulus_pa'])
        lowNu = lowE/(2*lowMu)-1
        lowLame = 2*lowMu*lowNu/(1-2*lowNu)
        _require(E > 0 and -1 < nu < F(1,2) and mu > 0 and 3*lame+2*mu > 0
                 and lowMu > 0 and 3*lowLame+2*lowMu > 0, 'The source strain inner product must be positive')
        original = json.loads(bank['complete_original_body_context_json'])['original_body']
        beta = _fraction(original['beta_cut_radians'])
        _require(original['original_parent_id'] == bank['original_parent_id'] and beta > 0,
                 'A modal clamp cannot be inferred from a different material')
        tensor_rows.append(dict(original_parent_id=bank['original_parent_id'], upper_young_modulus_pa=bank['upper_young_modulus_pa'],
            upper_poisson_ratio=bank['upper_poisson_ratio'], lower_young_modulus_pa=bank['lower_young_modulus_pa'],
            lower_shear_modulus_pa=bank['lower_shear_modulus_pa']))
        clamp_rows.append(dict(original_parent_id=bank['original_parent_id'], beta_cut_radians=original['beta_cut_radians'],
            complete_source_clamp_support_json=original['complete_source_clamp_support_json'],
            complete_source_clamp_row_json=original['source_clamp_row_json'],
            displacement_and_gradient_identically_zero_at_and_beyond_clamp=True))
    zero_queries = [q for q in source935['queries'] if q['label'] == 'zero_source_zero_rates']
    _require(len(zero_queries) == 1, 'The literal prepared-zero source is required')
    zero = zero_queries[0]['response']
    _require(len(zero['coordinate_values']) == 101 and all(_fraction(v) == 0 for v in zero['coordinate_values']),
             'The work baseline must be the complete prepared-zero endpoint')
    cold_zero, hot_zero = json.loads(zero['cold_response_json']), json.loads(zero['hot_response_json'])
    _require(cold_zero['system_binding_sha256'] == cold['binding_sha256']
             and _interval(hot_zero['energy_bounds_j']) == (F(0), F(0)), 'The prepared-zero HOT energy changed')
    cells = [c for c in cold_zero['cell_receipts'] if c['kind'] == 'zero_modal_complement_preparation']
    _require(len(cold_zero['cell_receipts']) == 408 and len(cells) == 192,
             'Every whole complement cell must be retained, without replacing the actual donor collar')
    compact_cells = []
    seen = set()
    clampmap = {c['original_parent_id']: _fraction(c['beta_cut_radians']) for c in clamp_rows}
    for c in cells:
        for name in ('active_coordinate_indices', 'global_tip_local_indices', 'global_projection_tip_indices',
                     'mode_displacement_columns', 'relative_gradient_columns_per_m'):
            _require(type(c[name]) is list and not c[name], 'A supposedly inactive complement retains modal support')
        _require(c['local_coordinate_indices'] == [56] and c['whole_material_box_enclosure_certified'] is True
                 and c['upper_beta_is_exact_pole'] is True and c['activation_history_replaced_by_zero'] is False,
                 'A point sample, removed pole or zeroed archive cannot replace the full complement')
        _require(_interval(c['beta_lo']) == (clampmap[c['original_parent_id']],)*2
                 and _interval(c['source_volume_bounds_m3'])[0] > 0,
                 'The complementary domain must begin at its exact original clamp with positive volume')
        key = _json([c[k] for k in ('original_parent_id','layer','theta_piece','integration_angle_lo','integration_angle_hi','depth_bounds_m')])
        _require(key not in seen, 'An original complement cell was duplicated')
        seen.add(key)
        names = ('original_parent_id','original_material_id','component_id','owner','layer','theta_piece',
                 'integration_angle_lo','integration_angle_hi','beta_lo','beta_hi','depth_bounds_m',
                 'upper_beta_is_exact_pole','source_volume_bounds_m3','active_coordinate_indices',
                 'global_tip_local_indices','global_projection_tip_indices','mode_displacement_columns',
                 'relative_gradient_columns_per_m','local_coordinate_indices','activation_history_replaced_by_zero',
                 'whole_material_box_enclosure_certified','inactive_complement_half_norm_coefficients_j_m3')
        compact_cells.append({k:c[k] for k in names})
    for parent in parents:
        _require(sum(c['original_parent_id'] == parent for c in cells) == 24,
                 'An original parent lost a layer or seam of its modal complement')
    side_map = {}
    sides = [json.loads(s) for s in zero['contact_side_receipts_json']]
    _require(len(sides) == 32, 'All original material attachment sides are required')
    for side in sides:
        parent, index = side['original_parent_id'], side['internal_radial_coordinate_index']
        _require(type(index) is int and 92 <= index < 100 and parent in parents,
                 'An original internal arm was assigned a different coordinate')
        _require(parent not in side_map or side_map[parent] == index, 'One parent has inconsistent source internal arms')
        side_map[parent] = index
    beam_rows = [json.loads(s) for s in zero['internal_beam_receipts_json']]
    _require(len(beam_rows) == 8 and set(side_map) == set(parents), 'All eight physical internal elements are required')
    compact_beams = []
    for i, parent in enumerate(parents):
        rows = [b for b in beam_rows if b['original_parent_id'] == parent]
        _require(len(rows) == 1, 'An original beam was added or omitted')
        b = rows[0]; index = b['global_internal_coordinate_index']
        _require(type(index) is int and index == side_map[parent]
                 and general['coordinate_order'][index][0] == 'source_internal'
                 and b['literal_reference_force_n'] == forces[i] and b['tip_stiffness_n_m'] == stiffnesses[i]
                 and b['original_beam_energy_added_again'] is False,
                 'The exact square needs the original parent/internal binding and each literal beam only once')
        compact_beams.append(dict(original_parent_id=parent, global_coordinate_index=index,
            coordinate_label=general['coordinate_order'][index], literal_reference_force_n=forces[i],
            tip_stiffness_n_m=stiffnesses[i]))
    contacts, contact_literal = [], F(0)
    for encoded in zero['contact_atoms_json']:
        receipt = json.loads(encoded); atom = json.loads(receipt['complete_original_atom_json'])
        material = json.loads(atom['complete_original_atom_history_json'])['original_material']
        law, columns = material['law'], material['columns']
        kn, T, gamma = (_literal(law[n]) for n in ('normal_stiffness_pa_m','tensile_strength_pa','fracture_energy_j_m2'))
        C = (_literal(columns['sub_length_m'])/_literal(columns['sub_normal_modulus_pa'])
             +_literal(columns['over_length_m'])/_literal(columns['over_normal_modulus_pa']))
        K = 1/(1/kn+C); tau = _fraction(atom['literal_source_tractions_pa'][0])
        state = json.loads(atom['source_local_state_json'])
        area = _fraction(atom['original_area_m2'])
        literal = sum((_fraction(v) for v in atom['stored_source_energy_components_j']), F(0))
        _require(kn > 0 and T > 0 and gamma > 0 and C >= 0 and 2*gamma/T > T/K
                 and K == _fraction(atom['series_stiffness_exact_pa_m'][0]) and tau < 0 and area > 0,
                 'The nonnegative series potential needs its authentic positive, nonsnapback law')
        _require(_literal(state['gap_m']) < 0 and _literal(state.get('damage',0)) == 0
                 and _literal(state.get('max_opening_m',0)) == 0
                 and all(_literal(law[n]) == 0 for n in ('cohesion_pa','friction_dry','friction_wet','viscosity_pa_s_m')),
                 'A contact cannot invent a different virgin normal or shear history')
        _require(receipt['original_area_m2'] == atom['original_area_m2']
                 and _fraction(receipt['literal_source_energy_j']) == literal
                 and receipt['source_original_area_included_once'] is True
                 and receipt['current_area_energy_multiplier_added'] is False,
                 'The original contact measure or energy offset changed')
        contacts.append(dict(point_id=atom['point_id'], original_area_m2=atom['original_area_m2'],
            literal_source_traction_pa=_payload(tau), series_stiffness_exact_pa_m=_payload(K),
            literal_source_energy_j=_payload(literal), normal_stiffness_pa_m=_payload(kn),
            column_compliance_m_pa=_payload(C), tensile_strength_pa=_payload(T), fracture_energy_j_m2=_payload(gamma),
            source_gap_m=_payload(_literal(state['gap_m'])), source_damage=_payload(F(0)),
            source_max_opening_m=_payload(F(0)), original_atom=atom))
        contact_literal += literal
    _require(len(contacts) == 16 and len({c['point_id'] for c in contacts}) == 16,
             'Every original contact must enter the relaxation cap exactly once')
    hot_rows = []
    for b in hot['bodies']:
        mu, lame, rho, g, birth = (_fraction(b[n]) for n in ('shear_modulus_pa','lame_modulus_pa',
            'density_kg_m3','gravity_m_s2','source_birth_radial_displacement_m'))
        _require(mu > 0 and lame >= 0 and rho > 0 and g > 0 and birth == 0
                 and _interval(b['original_reference_volume_bounds_m3'])[0] > 0
                 and _interval(b['actual_area_bounds_sr'])[0] > 0,
                 'Whole HOT bulk/base nonnegativity needs positive source measure and actual coefficients')
        hot_rows.append({n:b[n] for n in ('component_id','original_parent_id','original_reference_volume_bounds_m3',
            'actual_area_bounds_sr','shear_modulus_pa','lame_modulus_pa','density_kg_m3','gravity_m_s2',
            'source_birth_radial_displacement_m')})
    _require(len(hot_rows) == 5, 'All five HOT bodies must enter the nonnegative law')
    baseline = sum((_fraction(v) for v in energies), F(0))+contact_literal
    _require(_interval(zero['energy_bounds_j']) == (baseline, baseline),
             'The reference potential must retain all literal energies and offsets exactly')
    S2, S4 = integral['inactive_complement_S2_bounds_j'], integral['inactive_complement_S4_bounds_j']
    _require(_interval(S2)[0] > 0 and _interval(S4)[0] > 0, 'The full complement energy bound must remain positive')
    energy = dict(S2_complement_bounds_j=S2, S4_complement_bounds_j=S4,
        literal_reference_energies_j=energies, literal_reference_forces_n=forces, tip_stiffnesses_n_m=stiffnesses,
        original_parent_ids=parents, internal_beams=compact_beams, contacts16=contacts, hot_bodies=hot_rows,
        prepared_zero_energy_j=_payload(baseline), prepared_zero_hot_energy_j=[0,1])
    int_names = ('binding_sha256','cold_system_binding_sha256','continuum_system_binding_sha256',
        'archive_carrier_model_sha256','preparation_binding_sha256s','material_reference_measure',
        'whole_original_integral_certified','authentic_original_volume_included_once',
        'complete_modal_zero_clamp_complement_integrated','inactive_complement_tip_strain_identically_zero',
        'isotropic_radial_twist_cubic_moment_exactly_zero','original_gram_matrix_and_inverse_reused_without_quadrature',
        'complete_original_live_and_archive_material_retained','whole_source_volume_bounds_m3',
        'covered_source_volume_bounds_m3','original_parent_ids','nonempty_archive_domain_ids')
    theorem = dict(gram_inverse_certificate=gram, preparation_integral={n:integral[n] for n in int_names},
        complement_cell_receipts=compact_cells, cold_energy_system=cold, cold_tensor_materials=tensor_rows,
        authentic_clamp_support=clamp_rows,
        complete_source_carrier_certificate=json.loads(continuum['complete_source_carrier_certificate_json']),
        original_internal_attachment_bindings=[dict(original_parent_id=s['original_parent_id'],
            global_coordinate_index=s['internal_radial_coordinate_index'], point_id=s['point_id'],side=s['side']) for s in sides],
        frozen_owner_load_potential_used=False,
        orthogonal_residual_identity='half_norm(E-H*p)_C_squared; p=c_tip+actual_inverse_Gram*r',
        full_internal_square_identity='E_literal+f*(p+r)+K*(p+r)^2/2',
        complement_modal_displacement_gradient_and_tip_strain_identically_zero=True,
        global_bound_relaxes_all_100_mechanical_coordinates=True,
        contact_energy_offsets_retained_exactly=True,
        donor_collar_excluded_from_positive_floor_but_retained_in_nonnegative_remainder=True)
    return json.loads(_json(energy)), json.loads(_json(theorem))


@dataclass(frozen=True)
class NativeCommonPreparationFeasibility:
    source_context_binding_sha256: str
    basal_power_system_binding_sha256: str
    source_raw_pins: tuple
    preparation_energy_evidence_sha256: str
    full_support_theorem_evidence_sha256: str
    inactive_complement_S2_lower_j: F
    inactive_complement_S4_lower_j: F
    eight_beam_relaxation_cap_j: F
    sixteen_contact_relaxation_cap_j: F
    prepared_zero_potential_j: F
    maximum_net_power_w: F
    source_clocks_s: tuple
    remaining_conditions: tuple
    provenance: str
    binding_sha256: str
    mechanical_coordinate_count: int = 100
    preparation_coordinate_count: int = 1
    original_point_1_remaining_work_blocks: int = 1
    accepted_old_native_intervals: int = 1
    accepted_new_native_intervals: int = 0
    scope: str = 'DECLARED_COMMON101_GLOBAL_RELAXATION_NECESSARY_WORK_BOUND_ONLY'


def bind_native_common_preparation_feasibility(source_context, power_system, *, provenance):
    api = _power_api()
    api.validate_native_common_preparation_work_source_context(source_context)
    api.validate_native_common_basal_work_bound(power_system)
    _require(type(provenance) is str and bool(provenance.strip()), 'The necessary bound needs explicit provenance')
    _require(power_system.source_context_binding_sha256 == source_context.binding_sha256,
             'Energy and basal work must use the same authenticated reference source')
    evidence = api.native_common_preparation_work_source_context_evidence(source_context)
    energy, theorem = evidence['energy_evidence'], evidence['theorem_evidence']
    beam = sum((_fraction(f)**2/(2*_fraction(k)) for f,k in zip(
        energy['literal_reference_forces_n'],energy['tip_stiffnesses_n_m'])), F(0))
    contact = sum((_fraction(a['original_area_m2'])*_fraction(a['literal_source_traction_pa'])**2
                   /(2*_fraction(a['series_stiffness_exact_pa_m'])) for a in energy['contacts16']), F(0))
    result = NativeCommonPreparationFeasibility(source_context.binding_sha256, power_system.binding_sha256,
        tuple(tuple(x) for x in source_context.source_raw_pins), hashlib.sha256(_json(energy).encode()).hexdigest(),
        hashlib.sha256(_json(theorem).encode()).hexdigest(), _interval(energy['S2_complement_bounds_j'])[0],
        _interval(energy['S4_complement_bounds_j'])[0], beam, contact, _fraction(energy['prepared_zero_energy_j']),
        power_system.maximum_net_power_w, tuple(_fraction(x) for x in evidence['source_clocks_s']),
        tuple(evidence['remaining_conditions']), provenance, '')
    _require(result.maximum_net_power_w > 0 and len(result.remaining_conditions) == 3,
             'The bound needs positive net-power capacity and the unchanged three physical conditions')
    result = replace(result, binding_sha256=_seal(result))
    _SYSTEMS[result.binding_sha256] = result
    _GUARDS[result.binding_sha256] = _json(result)
    return result


def validate_native_common_preparation_feasibility(system):
    _require(type(system) is NativeCommonPreparationFeasibility and system.binding_sha256 in _SYSTEMS
             and _seal(system) == system.binding_sha256 and _json(system) == _GUARDS.get(system.binding_sha256),
             'The feasibility system must retain its authenticated immutable source and theorem premises')
    return system


@dataclass(frozen=True)
class NativeCommonPreparationFeasibilityResponse:
    system_binding_sha256: str
    preparation_fraction: F
    duration_s: F
    assumed_initial_kinetic_release_upper_j: F
    global_relaxed_potential_increase_lower_j: F
    necessary_net_external_work_lower_j: F
    maximum_net_basal_work_j: F
    necessary_minimum_duration_s: F
    necessary_work_deficit_lower_j: F
    duration_satisfies_necessary_bound: bool
    inactive_complement_energy_lower_j: F
    eight_beam_relaxation_cap_j: F
    sixteen_contact_relaxation_cap_j: F
    prepared_zero_potential_j: F
    maximum_net_power_w: F
    source_clocks_s: tuple
    remaining_conditions: tuple
    binding_sha256: str
    all_100_mechanical_coordinates_relaxed_in_global_bound: bool = True
    full_orthogonal_support_and_literal_offset_premises_authenticated: bool = True
    declared_necessary_work_bound_certified: bool = True
    declared_necessary_duration_bound_certified: bool = True
    quasistatic_R101_zero_path_required: bool = True
    no_additional_preparation_lambda_drive_required: bool = True
    prepared_zero935_potential_baseline_required: bool = True
    no_net_initial_kinetic_energy_release_required: bool = True
    supplied_initial_kinetic_release_budget_must_not_be_exceeded: bool = True
    actual_initial_kinetic_energy_budget_known: bool = False
    initial_kinetic_budget_is_user_supplied_conditional_assumption: bool = True
    physical_inertial_extension_certified: bool = False
    necessary_condition_is_sufficient: bool = False
    negative_lower_bound_replaced_by_zero_work_claim: bool = False
    actual_equilibrium_or_preparation_path_computed: bool = False
    full_system_preparation_work_certified: bool = False
    full_system_temporal_path_work_certified: bool = False
    reference_preparation_work_paid: bool = False
    physical_source_force_prestress_work_handover_certified: bool = False
    historical_archived_three_dimensional_prestress_recovered: bool = False
    general_state_full_physics_operator_certified: bool = False
    new_mixed_force_equilibrium_certified: bool = False
    physical_material_admission_authorized: bool = False
    whole_clock_all_pair_admission_authorized: bool = False
    clocks_advanced: bool = False
    original_point_1_completed: bool = False
    original_point_1_remaining_work_blocks: int = 1
    accepted_old_native_intervals: int = 1
    accepted_new_native_intervals: int = 0
    scope: str = 'CONDITIONAL_DECLARED_QUASISTATIC_PREPARED_ZERO935_NECESSITY_ONLY'


def evaluate_native_common_preparation_feasibility(system, preparation_fraction=F(1), *, duration_s,
        initial_kinetic_budget_j=F(0)):
    validate_native_common_preparation_feasibility(system)
    _require(type(preparation_fraction) is F and 0 <= preparation_fraction <= 1,
             'Preparation must be an explicit Fraction in[0,1]')
    _require(type(duration_s) is F and duration_s >= 0,
             'Duration must be an explicit nonnegative physical Fraction, never inferred from old clocks')
    _require(type(initial_kinetic_budget_j) is F and initial_kinetic_budget_j >= 0,
             'A hypothetical initial kinetic-release budget must be an explicit nonnegative Fraction')
    lam = preparation_fraction
    complement = lam**2*system.inactive_complement_S2_lower_j+lam**4*system.inactive_complement_S4_lower_j
    lower = complement-system.eight_beam_relaxation_cap_j-system.sixteen_contact_relaxation_cap_j
    external_lower = lower-initial_kinetic_budget_j
    maximum_work = system.maximum_net_power_w*duration_s
    minimum_duration = max(external_lower,F(0))/system.maximum_net_power_w
    response = NativeCommonPreparationFeasibilityResponse(system.binding_sha256, lam, duration_s,
        initial_kinetic_budget_j, lower, external_lower,
        maximum_work, minimum_duration, max(external_lower-maximum_work,F(0)), duration_s >= minimum_duration,
        complement, system.eight_beam_relaxation_cap_j, system.sixteen_contact_relaxation_cap_j,
        system.prepared_zero_potential_j, system.maximum_net_power_w, system.source_clocks_s,
        system.remaining_conditions, '')
    response = replace(response, no_net_initial_kinetic_energy_release_required=initial_kinetic_budget_j == 0)
    return replace(response, binding_sha256=_seal(response))


def native_common_preparation_feasibility_to_dict(system):
    validate_native_common_preparation_feasibility(system)
    return dict(format=FORMAT, **_payload(system))


def native_common_preparation_feasibility_from_dict(raw, source_context, power_system):
    _require(type(raw) is dict and type(raw.get('provenance')) is str, 'A full feasibility restart DTO is required')
    expected = bind_native_common_preparation_feasibility(source_context, power_system, provenance=raw['provenance'])
    _require(_json(raw) == _json(native_common_preparation_feasibility_to_dict(expected)),
             'A feasibility restart changed a source, theorem premise, scalar, clock or exact type')
    return expected


def native_common_preparation_feasibility_response_to_dict(response):
    _require(type(response) is NativeCommonPreparationFeasibilityResponse and _seal(response) == response.binding_sha256
             and response.system_binding_sha256 in _SYSTEMS, 'A full authenticated feasibility response is required')
    expected = evaluate_native_common_preparation_feasibility(_SYSTEMS[response.system_binding_sha256],
        response.preparation_fraction, duration_s=response.duration_s,
        initial_kinetic_budget_j=response.assumed_initial_kinetic_release_upper_j)
    _require(_json(response) == _json(expected), 'The conditional bound response changed its result or physical scope')
    return dict(format=RESPONSE_FORMAT, **_payload(response))


def native_common_preparation_feasibility_response_from_dict(raw, *, system, expected_preparation_fraction,
        expected_duration_s, expected_initial_kinetic_budget_j=F(0)):
    expected = evaluate_native_common_preparation_feasibility(system, expected_preparation_fraction,
        duration_s=expected_duration_s, initial_kinetic_budget_j=expected_initial_kinetic_budget_j)
    _require(type(raw) is dict and _json(raw) == _json(native_common_preparation_feasibility_response_to_dict(expected)),
             'A necessary-bound restart changed a result, exact input or energy-balance assumption')
    return expected
