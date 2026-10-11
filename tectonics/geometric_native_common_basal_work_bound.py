"""Authenticated necessary NET-work envelope of the frozen 68-point source rule.

This module reads complete saved source packets once to register compact immutable
evidence.  It executes no previous geometry, quadrature, force or audit main.
The envelope is already after isotropic drag, and uses the original positive
reference weights once.  It neither advances a clock nor authorizes a path.
"""
from dataclasses import dataclass, fields, replace
from fractions import Fraction as F
import hashlib
import json


CONTEXT_FORMAT = 'native-common-preparation-work-source-context-1'
FORMAT = 'native-common-basal-net-work-bound-system-1'
RESPONSE_FORMAT = 'native-common-basal-net-work-bound-response-1'
SOURCE_SEALS = (
    ('935', 'sha256', 'edb636a5e5c177d10f674c7424e0c0a76c57ae3e971c5bc69d99a89a57e8cf37'),
    ('899', 'packet_sha256', 'c863be05c9d641025666a1a4c3929701c7d472f9d574085926d64a2d08b4c439'),
    ('923', 'sha256', '9312c861d7238c1e443181f9928b63bf58060da13a609aa58a1a450c26b732dc'),
    ('903', 'packet_sha256', 'c53e4cd75219328eeac974f3b9982aa33ad78eaf2e07d823feafb9f07b289e6c'),
)
SOURCE_RAW_PINS = (
    ('935', 'a4f14ce71ecef16959685091a0c0b15d2a2e5c134a8023e9d3f98b8f8a7f0c0e'),
    ('899', '4a5c059176cef063bf12ff7bd98df830395504d6c12051f269a177fcbc375fec'),
    ('923', '33b6a3585b6d8cd644e29d30bf2e648e2479f6e87393b7122061dda56d08a5d4'),
    ('903', '0f8a0ee8ae082e3434b136bab7515944354eaf059db95965d0e87259f6791035'),
)
_CONTEXTS, _SYSTEMS, _RESPONSES = {}, {}, {}


def _payload(value):
    if type(value) is F:
        return [value.numerator, value.denominator]
    if isinstance(value, tuple):
        return [_payload(v) for v in value]
    if isinstance(value, list):
        return [_payload(v) for v in value]
    if isinstance(value, dict):
        return {k: _payload(v) for k, v in value.items()}
    return value


def _encoded(value):
    return json.dumps(_payload(value), sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode()


def _digest(value):
    return hashlib.sha256(_encoded(value)).hexdigest()


def _json(value):
    return _encoded(value).decode()


def _fraction(value):
    if (type(value) is not list or len(value) != 2
            or any(type(v) is not int for v in value) or value[1] <= 0):
        raise ValueError('A canonical literal Fraction pair is required')
    result = F(*value)
    if _payload(result) != value:
        raise ValueError('A source Fraction pair must be reduced and canonical')
    return result


def _interval(value):
    if type(value) is not dict or set(value) != {'lo', 'hi'}:
        raise ValueError('A complete canonical source interval is required')
    lo, hi = _fraction(value['lo']), _fraction(value['hi'])
    if lo > hi:
        raise ValueError('An original source interval is reversed')
    return lo, hi


def _literal(value):
    if type(value) is float:
        return F.from_float(value)
    if type(value) is int:
        return F(value)
    raise ValueError('A genuine numeric source literal is required')


def _body(value):
    return {f.name: getattr(value, f.name) for f in fields(value)
            if f.name != 'binding_sha256'}


def _remember(value, registry):
    value = replace(value, binding_sha256=_digest(_body(value)))
    previous = registry.get(value.binding_sha256)
    if previous is not None:
        return _guard(previous, type(value), registry)
    registry[value.binding_sha256] = value
    return value


def _guard(value, cls, registry):
    if (type(value) is not cls or registry.get(value.binding_sha256) is not value
            or _digest(_body(value)) != value.binding_sha256):
        raise ValueError('An altered, resealed or unregistered work-bound object is unauthorized')
    return value


def _same(left, right, message):
    # Canonical serialization distinguishes Python's otherwise equal bool/int.
    if _encoded(left) != _encoded(right):
        raise ValueError(message)


def _packet(packet, authority):
    label, seal_key, expected = authority
    if type(packet) is not dict or packet.get(seal_key) != expected:
        raise ValueError('The complete authentic '+label+' source packet is required')
    try:
        # Saved authorities already contain canonical JSON values. Avoid
        # recursively copying the complete 336 MB source for Fraction export.
        encoded = json.dumps({k: v for k, v in packet.items() if k != seal_key},
            sort_keys=True, separators=(',', ':'), ensure_ascii=False,
            allow_nan=False).encode()
    except (TypeError, ValueError) as exc:
        raise ValueError('A complete JSON source authority is required') from exc
    if hashlib.sha256(encoded).hexdigest() != expected:
        raise ValueError('An altered or resealed full '+label+' source is unauthorized')
    # Derive evidence from the very bytes just authenticated. Caller-owned
    # mutable dictionaries (or unusual nested Python subclasses) cannot
    # change the source between authentication and compact registration.
    authenticated = json.loads(encoded)
    authenticated[seal_key] = expected
    return authenticated


@dataclass(frozen=True, slots=True)
class NativeCommonPreparationWorkSourceContext:
    source_packet_seals: tuple
    source_raw_pins: tuple
    evidence_json: str
    provenance: str
    binding_sha256: str


def _power_evidence(source935, source903):
    trace = source935['trace_system']
    hot = source903['hot_mean_system']
    load = json.loads(hot['complete_current_load_json'])
    _same(trace['complete_source903_load_json'], hot['complete_current_load_json'],
        'The full original 903 load changed in the frozen 935 trace')
    _same(trace['source_mantle_json'], _json(load['source']),
        'The original spatial mantle source changed')
    if (trace['source903_load_sha256'] != load['load_sha256']
            or trace['hot_system_binding_sha256'] != hot['binding_sha256']
            or trace['complete903_source_binding_sha256'] != SOURCE_SEALS[3][2]
            or source935['general_system']['basal_trace_system_binding_sha256'] != trace['binding_sha256']
            or trace['positive_source_material_cell_row_count'] != 68
            or trace['original_live_source_material_cell_row_count'] != 59
            or trace['hot_birth_source_material_cell_row_count'] != 9
            or trace['paired_correction_metadata_count'] != 1):
        raise ValueError('All frozen 931/935 basal source associations must agree')
    if (load['archived_material_receives_basal_loading'] is not False
            or load['geometric_measure_midpoint_is_material_debit'] is not False
            or len(load['correction_cell_measures']) != 1):
        raise ValueError('The fixed measure, archive exclusion and paired correction must be preserved')
    radius = _fraction(trace['source_radius_m'])
    if radius != 100000:
        raise ValueError('The fixed full original source radius changed')
    rows = load['material_cell_measures']
    witnesses = [json.loads(v) for v in trace['source_row_witnesses_json']]
    if (len(rows) != 68 or len(witnesses) != 68
            or sum(r['material_kind'] == 'original_live' for r in rows) != 59
            or sum(r['material_kind'] == 'hot_birth' for r in rows) != 9):
        raise ValueError('Every original positive source material-cell row is required once')
    compact = []
    for index, (row, witness) in enumerate(zip(rows, witnesses)):
        lo, hi = _interval(row['area_bounds_m2'])
        if lo <= 0:
            raise ValueError('Every original source weight must remain strictly positive')
        _same(witness['complete_source_material_cell_row'], row,
            'A frozen trace changed its original material-cell row')
        _same(witness['source_weight_bounds_m2'], row['area_bounds_m2'],
            'A frozen trace changed its exact original positive weight')
        if (witness['source_material_cell_index'] != index
                or witness['corrected_source_weight_included_once'] is not True
                or witness['paired_correction_added_as_extra_weight'] is not False
                or witness['source_material_id'] != row['material_id']
                or witness['source_component_id'] != row['component_id']
                or witness['source_original_parent_id'] != row['original_parent_id']
                or witness['owner'] != row['owner']):
            raise ValueError('The original material provenance or once-only correction changed')
        compact.append(dict(source_material_cell_index=index, original_row=row,
            source_binding_sha256=witness['source_binding_sha256'],
            original_reference_weight_bounds_m2=_payload((lo, hi))))
    literal = load['source']
    beta = _literal(literal['basal_drag_pa_s_m'])
    amplitude = _literal(literal['amplitude_constant'])
    gradient = tuple(_literal(v) for v in literal['amplitude_gradient'])
    omega = tuple(_literal(v) for v in literal['angular_velocity_rad_s'])
    if (beta != 120000000000000 or amplitude != 1
            or gradient != (F.from_float(.17), F(0), F(0))
            or omega != (F(0), F(0), F.from_float(1.1e-12))):
        raise ValueError('The complete literal prescribed force and independent drag changed')
    return dict(fixed_source_radius_m=_payload(radius), basal_drag_pa_s_m=_payload(beta),
        amplitude_constant=_payload(amplitude), amplitude_gradient=_payload(gradient),
        angular_velocity_rad_s=_payload(omega), source903_load_sha256=load['load_sha256'],
        source_trace_system_binding_sha256=trace['binding_sha256'],
        source_mantle_json=trace['source_mantle_json'], original_positive_reference_rows=compact,
        source_correction_metadata=load['correction_cell_measures'],
        current_unit_direction_is_position_over_its_norm=True,
        nonzero_material_position_required=True, source_reference_weights_have_no_state_derivatives=True,
        actual_source_rule_trace_count=68, original_live_trace_count=59, hot_birth_trace_count=9,
        paired_correction_included_once=True, continuum_whole_base_quadrature_certified=False)


def bind_native_common_preparation_work_source_context(source935, source899, source923, source903,
        *, provenance='Complete frozen 935/899/923/903 necessary-work source evidence'):
    if type(provenance) is not str or not provenance.strip():
        raise ValueError('Explicit source-context provenance is required')
    source935, source899, source923, source903 = tuple(_packet(packet, authority)
        for packet, authority in zip((source935, source899, source923, source903), SOURCE_SEALS))
    cold = source935['cold_energy_system']
    if (cold['source899_packet_sha256'] != SOURCE_SEALS[1][2]
            or cold['source923_packet_sha256'] != SOURCE_SEALS[2][2]
            or cold['source899_raw_sha256'] != SOURCE_RAW_PINS[1][1]
            or cold['source923_raw_sha256'] != SOURCE_RAW_PINS[2][1]
            or source935['status'] != 'PASS_NATIVE935_DECLARED_GENERAL_STATE_OPERATOR'
            or source935['coordinate_count'] != 101
            or source935['accepted_new_native_intervals'] != 0
            or source935['clocks_advanced'] is not False
            or source935['frozen_owner_load_potential_used'] is not False):
        raise ValueError('The authentic declared 935 source context or source links changed')
    order = source935['general_system']['coordinate_order']
    if len(order) != 101 or order[100] != ['preparation', 'cold_depth_twist', 'lambda']:
        raise ValueError('The complete original 100-coordinate catalogue plus lambda is required')
    _same(order, source935['trace_system']['coordinate_order'], 'The source coordinate catalogues differ')
    power = _power_evidence(source935, source903)
    # Both new siblings consume one registered compact authority.  This late
    # import runs only the new source extraction, never historical math mains.
    from .geometric_native_common_preparation_feasibility import _extract_preparation_energy_evidence
    energy, theorem = _extract_preparation_energy_evidence(source935, source899, source923, source903)
    evidence = dict(power_evidence=power, energy_evidence=energy, theorem_evidence=theorem,
        coordinate_order=order, source_clocks_s=source935['general_system']['source_clocks_s'],
        remaining_conditions=source935['general_system']['remaining_conditions'],
        accepted_original_native_intervals=source935['accepted_original_native_intervals'],
        accepted_new_native_intervals=source935['accepted_new_native_intervals'])
    return _remember(NativeCommonPreparationWorkSourceContext(
        tuple((a[0], a[2]) for a in SOURCE_SEALS), SOURCE_RAW_PINS, _json(evidence), provenance, ''), _CONTEXTS)


def validate_native_common_preparation_work_source_context(context):
    return _guard(context, NativeCommonPreparationWorkSourceContext, _CONTEXTS)


def native_common_preparation_work_source_context_evidence(context):
    return json.loads(validate_native_common_preparation_work_source_context(context).evidence_json)


def native_common_preparation_work_source_context_to_dict(context):
    return dict(format=CONTEXT_FORMAT, **_payload(_body(validate_native_common_preparation_work_source_context(context))),
        binding_sha256=context.binding_sha256)


def native_common_preparation_work_source_context_from_dict(raw, source935, source899, source923, source903):
    if type(raw) is not dict or raw.get('format') != CONTEXT_FORMAT:
        raise ValueError('A complete source-bound preparation-work context DTO is required')
    expected = bind_native_common_preparation_work_source_context(source935, source899, source923, source903,
        provenance=raw.get('provenance'))
    _same(raw, native_common_preparation_work_source_context_to_dict(expected),
        'The preparation-work context restart changed its complete source evidence')
    return expected


@dataclass(frozen=True, slots=True)
class NativeCommonBasalWorkBoundSystem:
    source_context_binding_sha256: str
    source_trace_system_binding_sha256: str
    fixed_reference_weight_sum_bounds_m2: tuple
    maximum_net_power_density_w_m2: F
    maximum_net_power_w: F
    fixed_reference_trace_count: int
    original_positive_reference_rows_json: str
    provenance: str
    binding_sha256: str
    scope: str = 'NECESSARY_NET_WORK_BOUND_OF_DECLARED_FIXED_68_REFERENCE_SOURCE_RULE'
    normalized_position_unit_direction_proved: bool = True
    full_101_velocity_chain_rule_including_lambda_retained: bool = True
    generalized_and_spatial_power_identical_by_chain_rule: bool = True
    isotropic_drag_already_subtracted: bool = True
    maximum_power_attainability_certified: bool = False
    continuum_whole_base_quadrature_certified: bool = False
    complete_actual_bottom_support_certified: bool = False
    general_state_full_physics_operator_certified: bool = False
    physical_material_admission_authorized: bool = False
    new_mixed_force_equilibrium_certified: bool = False


def bind_native_common_basal_work_bound(source_context, *, provenance):
    if type(provenance) is not str or not provenance.strip():
        raise ValueError('Explicit necessary-work bound provenance is required')
    evidence = native_common_preparation_work_source_context_evidence(source_context)['power_evidence']
    beta = _fraction(evidence['basal_drag_pa_s_m'])
    radius = _fraction(evidence['fixed_source_radius_m'])
    amplitude = _fraction(evidence['amplitude_constant'])
    gradient = tuple(_fraction(v) for v in evidence['amplitude_gradient'])
    omega = tuple(_fraction(v) for v in evidence['angular_velocity_rad_s'])
    rows = evidence['original_positive_reference_rows']
    weights = tuple(tuple(_fraction(v) for v in row['original_reference_weight_bounds_m2']) for row in rows)
    if (len(weights) != 68 or any(lo <= 0 or lo > hi for lo, hi in weights)
            or gradient[0] <= 0 or gradient[1:] != (F(0), F(0))
            or omega[:2] != (F(0), F(0)) or beta <= 0):
        raise ValueError('The authenticated positive fixed source rule is required')
    # q=X/|X| gives |q|=1 independently of material-position magnitude.
    # (1+k qx)^2(1-qz^2) <= (1+k)^2 for the authentic positive k.
    density = beta*radius**2*omega[2]**2*(amplitude+gradient[0])**2/4
    total = (sum((lo for lo, _ in weights), F(0)), sum((hi for _, hi in weights), F(0)))
    result = _remember(NativeCommonBasalWorkBoundSystem(source_context.binding_sha256,
        evidence['source_trace_system_binding_sha256'], total, density, total[1]*density, 68,
        _json(rows), provenance, ''), _SYSTEMS)
    return result


def validate_native_common_basal_work_bound(system):
    system = _guard(system, NativeCommonBasalWorkBoundSystem, _SYSTEMS)
    context = _CONTEXTS.get(system.source_context_binding_sha256)
    validate_native_common_preparation_work_source_context(context)
    return system


def native_common_basal_work_bound_to_dict(system):
    system = validate_native_common_basal_work_bound(system)
    return dict(format=FORMAT, **_payload(_body(system)), binding_sha256=system.binding_sha256)


def native_common_basal_work_bound_from_dict(raw, source_context):
    if type(raw) is not dict or raw.get('format') != FORMAT:
        raise ValueError('A complete source-bound NET-work system DTO is required')
    expected = bind_native_common_basal_work_bound(source_context, provenance=raw.get('provenance'))
    _same(raw, native_common_basal_work_bound_to_dict(expected), 'The exact NET-work system restart changed')
    return expected


@dataclass(frozen=True, slots=True)
class NativeCommonBasalWorkBoundResponse:
    system_binding_sha256: str
    duration_s: F
    maximum_net_power_w: F
    maximum_net_work_j: F
    binding_sha256: str
    physical_duration_is_explicit_parameter: bool = True
    original_clocks_used_as_physical_duration: bool = False
    net_power_is_after_drag: bool = True
    drag_dissipation_subtracted_twice: bool = False
    additional_preparation_actuator_work_included: bool = False
    full_system_temporal_path_work_certified: bool = False
    physical_material_admission_authorized: bool = False
    new_mixed_force_equilibrium_certified: bool = False
    clocks_advanced: bool = False
    accepted_new_native_intervals: int = 0
    original_point_1_completed: bool = False


def evaluate_native_common_basal_work_bound(system, *, duration_s):
    system = validate_native_common_basal_work_bound(system)
    if type(duration_s) is not F or duration_s < 0:
        raise ValueError('A nonnegative explicit literal Fraction physical duration T is required')
    return _remember(NativeCommonBasalWorkBoundResponse(system.binding_sha256, duration_s,
        system.maximum_net_power_w, system.maximum_net_power_w*duration_s, ''), _RESPONSES)


def native_common_basal_work_bound_response_to_dict(response):
    response = validate_native_common_basal_work_bound_response(response)
    return dict(format=RESPONSE_FORMAT, **_payload(_body(response)), binding_sha256=response.binding_sha256)


def validate_native_common_basal_work_bound_response(response, *, system=None):
    response = _guard(response, NativeCommonBasalWorkBoundResponse, _RESPONSES)
    actual = _SYSTEMS.get(response.system_binding_sha256)
    validate_native_common_basal_work_bound(actual)
    if system is not None and validate_native_common_basal_work_bound(system) is not actual:
        raise ValueError('The NET-work response belongs to a different authenticated source system')
    return response


def native_common_basal_work_bound_response_from_dict(raw, *, system, expected_duration_s):
    if type(raw) is not dict or raw.get('format') != RESPONSE_FORMAT:
        raise ValueError('A complete exact-duration NET-work response DTO is required')
    expected = evaluate_native_common_basal_work_bound(system, duration_s=expected_duration_s)
    _same(raw, native_common_basal_work_bound_response_to_dict(expected),
        'The NET-work restart changed its exact explicit duration or source bound')
    return expected
