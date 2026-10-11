"""Source-bound fifteen-coordinate energy on the cold preparation slice.

The frozen923 volume polynomial, HOT903 zero state, actual prepared source16
contacts and the complete frozen owner6 potential are included once. Eight
internal coordinates shift the existing calibrated cold beam drivers. The
other86 body coordinates are explicitly fixed; their physical basal loading
and a general mixed equilibrium are not supplied by this slice.
"""
from dataclasses import dataclass, fields, replace
from fractions import Fraction as F
import hashlib
import json

from .geometric_native_pose import FractionInterval as I
from .geometric_native_common_carrier_source_work import _payload
from .geometric_native_common_cold_variational_work import (
    _fraction, _saved, _round, _square,
)
from .geometric_native_common_cold_preparation_energy import (
    _coefficients, _polynomial, _work, _difference,
)

FORMAT = 'native-common-fifteen-preparation-balance-system-1'
RESPONSE_FORMAT = 'native-common-fifteen-preparation-balance-response-1'
SCOPE = 'DECLARED_OWNER_INTERNAL_PREPARATION_SLICE_WITH_COLD56_HOT30_FIXED_ZERO'
AUTHORITY_PINS = (
    ('923', '33b6a3585b6d8cd644e29d30bf2e648e2479f6e87393b7122061dda56d08a5d4',
     '9312c861d7238c1e443181f9928b63bf58060da13a609aa58a1a450c26b732dc', 'sha256'),
    ('899', '4a5c059176cef063bf12ff7bd98df830395504d6c12051f269a177fcbc375fec',
     'c863be05c9d641025666a1a4c3929701c7d472f9d574085926d64a2d08b4c439', 'packet_sha256'),
    ('903', '0f8a0ee8ae082e3434b136bab7515944354eaf059db95965d0e87259f6791035',
     'c53e4cd75219328eeac974f3b9982aa33ad78eaf2e07d823feafb9f07b289e6c', 'packet_sha256'),
    ('907', '5e5555712e57fd69156ee475fe5feb161baef3cb96dd911a443228d160f53005',
     'bba7f1f4913d6bc47f5f41beff0a279be3c203993ddd0b0b06979c719abeda1a', 'packet_sha256'),
)
_SYSTEMS, _CONTEXTS, _RESPONSES, _EVALUATIONS = {}, {}, {}, {}


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False)


def _digest(value):
    return hashlib.sha256(_json(_payload(value)).encode()).hexdigest()


def _body(value):
    return {f.name: getattr(value, f.name) for f in fields(value) if f.name != 'binding_sha256'}


def _seal(value):
    # Immutable complete source strings are committed by their exact hashes.
    return _digest({k: hashlib.sha256(v.encode()).hexdigest() if type(v) is str and k.endswith('_json') else v
                    for k, v in _body(value).items()})


def _remember(value, registry):
    value = replace(value, binding_sha256=_seal(value))
    previous = registry.get(value.binding_sha256)
    if previous is not None:
        return _guard(previous, type(value), registry)
    registry[value.binding_sha256] = value
    return value


def _guard(value, cls, registry):
    if type(value) is not cls or registry.get(value.binding_sha256) is not value or _seal(value) != value.binding_sha256:
        raise ValueError('An altered, resealed or unregistered preparation balance is not authorized')
    return value


def _precision(value):
    if type(value) is not int or not 96 <= value <= 512:
        raise ValueError('Balance precision is a literal integer from96 to512')
    return value


def _bounds(value):
    return value if type(value) is I else I(value)


def _sum(values):
    return sum(values, F(0))


def _authenticate(packet, authority):
    label, raw, seal, seal_key = authority
    if type(packet) is not dict or packet.get(seal_key) != seal:
        raise ValueError('The entire frozen'+label+' authority with its original self seal is required')
    canonical = _json(packet).encode()
    if hashlib.sha256(canonical+b'\n').hexdigest() != raw:
        raise ValueError('The complete frozen'+label+' canonical raw bytes changed')
    if hashlib.sha256(_json({k:v for k,v in packet.items() if k != seal_key}).encode()).hexdigest() != seal:
        raise ValueError('The complete frozen'+label+' self seal changed')


def _owner_relaxation_certificate(torque, drag, time_scale):
    """A rigorous lower bound for the NEW full frozen owner potential."""
    if (type(torque) is not tuple or len(torque) != 6 or type(drag) is not tuple or len(drag) != 6
            or any(type(row) is not tuple or len(row) != 6 for row in drag)
            or type(time_scale) is not F or time_scale <= 0
            or any(type(x) not in (F,I) for x in torque)
            or any(type(x) not in (F,I) for row in drag for x in row)):
        raise ValueError('The relaxation proof needs all six torques and all36 drag entries')
    symmetric = tuple(tuple((drag[i][j]+drag[j][i])/2 for j in range(6)) for i in range(6))
    absupper = lambda x: max(abs(_bounds(x).lo), abs(_bounds(x).hi))
    rows = tuple(_bounds(symmetric[i][i]).lo-_sum(absupper(symmetric[i][j])
                 for j in range(6) if j != i) for i in range(6))
    alpha = min(rows)
    if alpha <= 0:
        raise ValueError('The full owner drag must have a strictly positive certified Gershgorin lower bound')
    norm2 = _sum(absupper(x)**2 for x in torque)
    return symmetric, rows, alpha, time_scale*norm2/(2*alpha)


def _cold_source_context(cold899):
    """Reconstruct the old source binding hash by pure saved-field arithmetic."""
    system = cold899['cold_continuum_energy_system']
    banks, queries = system['banks'], cold899['evaluation_queries']
    if len(banks) != 8 or len(queries) != 2 or system['source_beams_included_once'] is not True:
        raise ValueError('All eight complete cold ledgers must remain once')
    certificate = queries[0]['response']['gram_inverse_certificate']
    for query in queries:
        response, integral = query['response'], query['response']['integrals']
        proof = integral['quadrature_certificate']
        if (response['gram_inverse_certificate'] != certificate
                or proof['whole_domain_enclosure_certified'] is not True
                or proof['authentic_original_volume_included_once'] is not True
                or proof['complete_original_parent_ids'] != ['octant:'+str(i) for i in range(8)]
                or len(proof['nonempty_archive_domain_ids']) != 3
                or integral['gram_bounds_n_m'] != certificate['gram_bounds_n_m']):
            raise ValueError('The authentic whole cold Gram and archive domain proof changed')
    context = dict(source_raw_sha256=AUTHORITY_PINS[1][1], source_packet_sha256=AUTHORITY_PINS[1][2],
        complete_source_system_json=hashlib.sha256(_json(system).encode()).hexdigest(),
        complete_saved_queries_json=hashlib.sha256(_json(queries).encode()).hexdigest(),
        coordinate_order=tuple(tuple(v) for v in system['coordinate_order']),
        tip_coordinate_indices=tuple(v['tip_coordinate_index'] for v in banks),
        literal_reference_energies_j=tuple(_fraction(v['source_reference_energy_j']) for v in banks),
        literal_reference_forces_n=tuple(_fraction(v['source_reference_force_n']) for v in banks),
        tip_stiffnesses_n_m=tuple(_fraction(v['source_old_tip_stiffness_n_m']) for v in banks),
        gram_bounds_n_m=tuple(tuple(_saved(v) for v in row) for row in certificate['gram_bounds_n_m']),
        inverse_gram_bounds_m_n=tuple(tuple(_saved(v) for v in row) for row in certificate['inverse_gram_bounds_m_n']),
        complete_gram_inverse_certificate_json=hashlib.sha256(_json(certificate).encode()).hexdigest())
    return system, banks, context, _digest(context)


@dataclass(frozen=True, slots=True)
class NativeCommonPreparationBalanceSystem:
    source_authority_raw_sha256: tuple
    source_authority_packet_sha256: tuple
    cold_variational_source_binding_sha256: str
    cold_preparation_energy_binding_sha256: str
    cold_preparation_integral_binding_sha256: str
    contacts_system_binding_sha256: str
    complete_frozen923_native_json: str
    complete_current907_catalog_json: str
    complete_current_owner_load_json: str
    complete_saved_hot_zero_response_json: str
    complete_original_cold_beam_catalog_json: str
    registered_source_response_binding_sha256: str
    registered_source_energy_bounds_j: I
    prepared_zero_total_energy_j: F
    original_coordinate_order: tuple
    coordinate_order: tuple
    active_original100_indices: tuple
    frozen_original100_indices: tuple
    frozen_body_coordinate_values: tuple
    coordinate_units: tuple
    gradient_units: tuple
    hessian_units: tuple
    original_beam_internal_indices: tuple
    cold_energy_polynomial_coefficients_j: tuple
    cold_tip_projection_linear_m: tuple
    cold_tip_projection_quadratic_m: tuple
    source_beam_forces_n: tuple
    source_beam_stiffnesses_n_m: tuple
    source_beam_energies_j: tuple
    inactive_complement_half_norm_coefficients_j: tuple
    eight_literal_beam_relaxation_cap_j: F
    source_contact_relaxation_lower_bound_j: F
    source_contact_literal_energy_j: F
    hot_fixed_energy_j: F
    owner_torque_bounds_nm: tuple
    owner_drag_bounds_nm_s: tuple
    owner_drag_time_scale_s: F
    owner_drag_gershgorin_row_lower_bounds_nm_s: tuple
    owner_drag_gershgorin_lower_bound_nm_s: F
    owner_potential_relaxation_cap_j: F
    precision: int
    provenance: str
    binding_sha256: str
    scope: str = SCOPE
    owner_coordinate_values_exact_zero: bool = True
    original_body86_coordinates_fixed_zero: bool = True
    complete_declared_preparation_component_sum_certified: bool = True
    source_beams_included_once: bool = True
    internal_residuals_shift_existing_beams_once: bool = True
    old891_complete_beam_bulk_operator_added_again: bool = False
    old_scalar_hot_foundation_added_again: bool = False
    complete_modal_basal_loading_defined: bool = False
    full_physical_preparation_balance_certified: bool = False
    general_state_mixed_operator_certified: bool = False
    new_mixed_force_equilibrium_certified: bool = False
    historical_archived_three_dimensional_prestress_recovered: bool = False
    born_hot_receiving_contact_law_certified: bool = False
    reference_preparation_work_paid: bool = False
    complete_physical_system_preparation_work_paid: bool = False
    full_system_temporal_path_work_certified: bool = False
    whole_clock_all_pair_admission_authorized: bool = False
    physical_material_admission_authorized: bool = False
    accepted_original_native_intervals: int = 1
    accepted_new_native_intervals: int = 0
    source_geometry_elapsed_time_s: F = F(0)
    contact_geometry_elapsed_time_s: F = F(2000000)
    memory_elapsed_time_s: F = F(1000000)
    material_geometry_elapsed_time_s: F = F(1000000)
    clocks_advanced: bool = False
    original_point_1_completed: bool = False
    original_point_1_remaining_work_blocks: int = 1


def _hessian_units(old, active):
    def unit(i,j):
        if i == 14:
            return 'J/rad' if j < 6 else ('N' if j < 14 else 'J')
        if j == 14:
            return 'N*m' if i < 6 else 'N'
        return old[active[i]][active[j]]
    return tuple(tuple(unit(i,j) for j in range(15)) for i in range(15))


def bind_native_common_preparation_balance(cold923, cold899, hot903, current907, *,
                                           contacts_system, provenance, precision=96):
    _precision(precision)
    if type(provenance) is not str or not provenance.strip():
        raise ValueError('The declared preparation balance requires provenance')
    from .geometric_native_common_preparation_contacts import validate_native_common_preparation_contacts
    validate_native_common_preparation_contacts(contacts_system)
    for packet, authority in zip((cold923,cold899,hot903,current907), AUTHORITY_PINS):
        _authenticate(packet, authority)
    cold, banks, source, source_binding = _cold_source_context(cold899)
    energy, integral = cold923['energy'], cold923['integral']
    if (cold923['status'] != 'PASS_NEW_WHOLE_COLD_CONSTITUTIVE_PREPARATION_WORK'
            or cold923['complete_energy_restart_equal'] is not True
            or cold923['complete_integral_restart_equal'] is not True
            or integral['source_binding_sha256'] != source_binding
            or energy['source_binding_sha256'] != source_binding
            or energy['integrals_binding_sha256'] != integral['binding_sha256']
            or energy['source_beams_included_once'] is not True
            or integral['whole_original_integral_certified'] is not True
            or integral['inactive_complement_tip_strain_identically_zero'] is not True
            or integral['relative_prepared_cartesian_green_strain_used'] is not True
            or integral['complete_modal_zero_clamp_complement_integrated'] is not True
            or cold923['source_context'] != _payload({k:v for k,v in source.items()
                if not k.startswith('complete_') and k != 'coordinate_order'}) | {'source_binding_sha256':source_binding}):
        raise ValueError('The complete saved923 volume moments and authentic899 association changed')
    if _digest({k:v for k,v in energy.items() if k not in ('format','binding_sha256')}) != energy['binding_sha256']:
        raise ValueError('The saved preparation energy typed seal changed')
    parsevec = lambda raw: tuple(_saved(x) for x in raw)
    coefficients,p1,p2,residual = _coefficients(source['inverse_gram_bounds_m_n'],
        _sum(source['literal_reference_energies_j']), source['literal_reference_forces_n'],
        source['tip_stiffnesses_n_m'], parsevec(integral['r1_bounds_n']), parsevec(integral['r2_bounds_n']),
        _saved(integral['S2_bounds_j']), _saved(integral['S3_bounds_j']), _saved(integral['S4_bounds_j']), energy['precision'])
    if (_payload(coefficients) != energy['energy_polynomial_coefficients_j']
            or _payload(p1) != energy['tip_projection_linear_m']
            or _payload(p2) != energy['tip_projection_quadratic_m']
            or _payload(residual) != energy['orthogonal_residual_polynomial_coefficients_j']):
        raise ValueError('The saved preparation polynomial must reproduce from every saved whole-volume moment')
    hot = hot903['hot_mean_system']; mixed = current907['mixed_force_system']
    hotzero = [q['response'] for q in hot903['evaluation_queries']
               if q['response']['coordinate_increments'] == [[0,1]]*30]
    if (len(hotzero) != 1 or hotzero[0]['energy_bounds_j'] != {'lo':[0,1],'hi':[0,1]}
            or hot['cold_continuum_component_binding_sha256'] != cold['binding_sha256']
            or hot['cold_material_carrier_binding_sha256'] != integral['cold_system_binding_sha256']
            or mixed['complete_current_load_json'] != hot['complete_current_load_json']
            or mixed['cold_continuum_system_binding_sha256'] != cold['binding_sha256']
            or mixed['hot_mean_system_binding_sha256'] != hot['binding_sha256']
            or current907['source_literal_dt_s'] != 1000000.0
            or mixed['owner_drag_time_scale_s'] != [1000000,1]
            or contacts_system.source907_raw_sha256 != AUTHORITY_PINS[3][1]
            or contacts_system.cold_material_carrier_binding_sha256 != integral['cold_system_binding_sha256']
            or contacts_system.preparation_binding_sha256s != tuple(integral['preparation_binding_sha256s'])):
        raise ValueError('The complete HOT zero, owner load and actual cold preparation references must agree')
    order = tuple(tuple(x) for x in mixed['coordinate_order'])
    active = tuple(range(6))+tuple(range(92,100)); frozen = tuple(range(6,92))
    neworder = tuple(order[i] for i in active)+(('preparation','cold_depth_twist','lambda'),)
    if (len(order) != 100 or len(set(order)) != 100
            or contacts_system.original_coordinate_order != order
            or contacts_system.active_original100_indices != active
            or contacts_system.coordinate_order != neworder):
        raise ValueError('The complete original catalogue and actual fifteen-coordinate contact injection changed')
    load = json.loads(hot['complete_current_load_json'])
    if load.get('geometric_measure_midpoint_is_material_debit') is not False:
        raise ValueError('The full load catalogue cannot debit midpoint material measures')
    torque = parsevec(load['torque_bounds_nm'])
    drag = tuple(parsevec(row) for row in load['drag_bounds_nm_s'])
    dt = _fraction(mixed['owner_drag_time_scale_s'])
    _, rows, alpha, ownercap = _owner_relaxation_certificate(torque,drag,dt)
    beam_catalog = tuple(json.loads(b['complete_source_beam_json']) for b in banks)
    internalindices = tuple(active.index(b['global_coordinate_index']) for b in beam_catalog)
    if set(internalindices) != set(range(6,14)):
        raise ValueError('Every original beam must shift its own distinct existing internal driver')
    cap = _sum(f*f/(2*K) for f,K in zip(source['literal_reference_forces_n'],source['tip_stiffnesses_n_m']))
    if _fraction(energy['eight_literal_beam_relaxation_cap_j']) != cap:
        raise ValueError('The original beam relaxation cap cannot be changed or counted twice')
    complement = parsevec(energy['inactive_complement_half_norm_coefficients_j'])
    if min(_bounds(x).lo for x in complement) < 0 or len(complement) != 2:
        raise ValueError('The whole inactive complement retains both nonnegative squared-strain coefficients')
    catalog = {k:mixed[k] for k in ('coordinate_order','original_coordinate_order','coordinate_units',
        'gradient_units','hessian_units','bound_coordinate_increments','original_bound_coordinate_increments',
        'source_atoms_complete_sha256','internal_beams','owner_load_model','owner_drag_time_scale_is_new_declared_calibration')}
    registered = [q['response'] for q in current907['evaluation_queries']
                  if q['response']['coordinate_increments'] == mixed['bound_coordinate_increments']]
    if len(registered) != 1 or registered[0]['system_binding_sha256'] != mixed['binding_sha256']:
        raise ValueError('Exactly one full authenticated registered COMMON100 source response is required')
    source_energy = _saved(registered[0]['energy_bounds_j'])
    prepared_zero_energy = _sum(source['literal_reference_energies_j'])+_sum(contacts_system.literal_source_energies_j)
    result = NativeCommonPreparationBalanceSystem(tuple(v[1] for v in AUTHORITY_PINS),
        tuple(v[2] for v in AUTHORITY_PINS),source_binding,energy['binding_sha256'],integral['binding_sha256'],
        contacts_system.binding_sha256,_json(cold923),_json(catalog),hot['complete_current_load_json'],
        _json(hotzero[0]),_json(beam_catalog),registered[0]['binding_sha256'],_bounds(source_energy),
        prepared_zero_energy,order,neworder,active,frozen,(F(0),)*86,
        tuple(mixed['coordinate_units'][i] for i in active)+('1',),
        tuple(mixed['gradient_units'][i] for i in active)+('J',),_hessian_units(mixed['hessian_units'],active),
        internalindices,coefficients,p1,p2,source['literal_reference_forces_n'],source['tip_stiffnesses_n_m'],
        source['literal_reference_energies_j'],complement,cap,contacts_system.source_contact_relaxation_lower_bound_j,
        _sum(contacts_system.literal_source_energies_j),F(0),torque,drag,dt,rows,alpha,ownercap,precision,provenance,'')
    result = _remember(result,_SYSTEMS)
    _CONTEXTS[result.binding_sha256] = contacts_system
    return result


def validate_native_common_preparation_balance(system):
    _guard(system,NativeCommonPreparationBalanceSystem,_SYSTEMS)
    from .geometric_native_common_preparation_contacts import validate_native_common_preparation_contacts
    contacts = _CONTEXTS.get(system.binding_sha256)
    if contacts is None or contacts.binding_sha256 != system.contacts_system_binding_sha256:
        raise ValueError('The complete preparation balance lost its actual source16 context')
    validate_native_common_preparation_contacts(contacts)
    return system


def native_common_preparation_balance_to_dict(system):
    return dict(format=FORMAT,**_payload(validate_native_common_preparation_balance(system)))


def native_common_preparation_balance_from_dict(raw,cold923,cold899,hot903,current907,*,contacts_system):
    if type(raw) is not dict or raw.get('format') != FORMAT:
        raise ValueError('A complete source-bound preparation balance DTO is required')
    value = bind_native_common_preparation_balance(cold923,cold899,hot903,current907,
        contacts_system=contacts_system,provenance=raw.get('provenance'),precision=raw.get('precision'))
    if _json(native_common_preparation_balance_to_dict(value)) != _json(raw):
        raise ValueError('A resealed preparation balance cannot change its complete source context or physical scope')
    return value


def _coordinates(preparation_fraction, internal_coordinates):
    if type(preparation_fraction) is not F or not 0 <= preparation_fraction <= 1:
        raise ValueError('The actual preparation slice uses a literal Fraction lambda in[0,1]')
    internal = (F(0),)*8 if internal_coordinates is None else internal_coordinates
    if type(internal) is not tuple or len(internal) != 8 or any(type(x) is not F for x in internal):
        raise ValueError('All eight internal coordinates must be literal immutable Fractions')
    return internal


def _cold_twojet(system, lam, internal, precision):
    coefficients = system.cold_energy_polynomial_coefficients_j
    base = _polynomial(coefficients,lam,precision)
    g, H = [I(0)]*15, [[I(0) for _ in range(15)] for _ in range(15)]
    g[14] = _polynomial(tuple(k*c for k,c in enumerate(coefficients[1:],1)),lam,precision)
    H[14][14] = _polynomial(tuple(k*(k-1)*coefficients[k] for k in range(2,5)),lam,precision)
    correction = F(0); receipts=[]
    for k,(force,K,p1,p2,index) in enumerate(zip(system.source_beam_forces_n,
            system.source_beam_stiffnesses_n_m,system.cold_tip_projection_linear_m,
            system.cold_tip_projection_quadratic_m,system.original_beam_internal_indices)):
        p = _round(lam*p1+lam*lam*p2,precision)
        dp, ddp = _round(p1+2*lam*p2,precision),2*p2
        r = internal[index-6]
        energy = _round((force+K*p)*r+K*r*r/2,precision)
        correction += energy
        g[index] = _round(force+K*(p+r),precision)
        H[index][index] = I(K)
        mixed = _round(K*dp,precision)
        H[index][14] = H[14][index] = mixed
        g[14] += K*r*dp; H[14][14] += K*r*ddp
        receipts.append(_json(_payload(dict(cold_bank_index=k,active_internal_coordinate_index=index,
            original_internal_coordinate_index=system.active_original100_indices[index],
            source_force_n=force,source_stiffness_n_m=K,tip_projection_m=p,
            tip_projection_lambda_first_m=dp,tip_projection_lambda_second_m=ddp,
            internal_increment_m=r,energy_increment_bounds_j=energy,
            old_beam_reference_energy_added_again=False))))
    return base,_round(correction,precision),tuple(_round(x,precision) for x in g),tuple(
        tuple(_round(x,precision) for x in row) for row in H),tuple(receipts)


@dataclass(frozen=True, slots=True)
class NativeCommonPreparationBalanceResponse:
    system_binding_sha256: str
    contact_response_binding_sha256: str
    coordinate_order: tuple
    coordinate_values: tuple
    preparation_fraction: F
    internal_coordinates: tuple
    energy_bounds_j: I
    potential_path_work_from_prepared_zero_bounds_j: I
    physical_reaction_work_from_prepared_zero_bounds_j: I
    gradient_bounds: tuple
    hessian_bounds: tuple
    coordinate_units: tuple
    gradient_units: tuple
    hessian_units: tuple
    energy_partitions_json: str
    internal_beam_corrections_json: tuple
    common_owner_rotation_gradient_bounds_nm: tuple
    independent_fixed_owner_potential_work_lower_bound_j: F
    independent_owner_relaxed_potential_work_lower_bound_j: F
    registered_source_potential_energy_difference_bounds_j: I
    independent_registered_source_potential_energy_difference_lower_bound_j: F
    precision: int
    binding_sha256: str
    complete_declared_preparation_component_sum_certified: bool = True
    complete_declared_active_fifteen_twojet_certified: bool = True
    source_beams_included_once: bool = True
    internal_residuals_shift_existing_beams_once: bool = True
    hot30_cold56_coordinates_fixed_zero: bool = True
    original_body86_loads_replaced_by_zero: bool = False
    common_owner_z_rotation_strict_descent_certified: bool = True
    clamped_owner_slice_is_free_mixed_equilibrium: bool = False
    owner_motion_equilibrium_excluded: bool = False
    complete_modal_basal_loading_defined: bool = False
    full_physical_preparation_balance_certified: bool = False
    general_state_mixed_operator_certified: bool = False
    new_mixed_force_equilibrium_certified: bool = False
    reference_preparation_work_paid: bool = False
    complete_physical_system_preparation_work_paid: bool = False
    historical_archived_three_dimensional_prestress_recovered: bool = False
    born_hot_receiving_contact_law_certified: bool = False
    full_system_temporal_path_work_certified: bool = False
    physical_material_admission_authorized: bool = False
    whole_clock_all_pair_admission_authorized: bool = False
    accepted_new_native_intervals: int = 0
    clocks_advanced: bool = False


def evaluate_native_common_preparation_balance(system,*,preparation_fraction=F(0),
                                               internal_coordinates=None,precision=96):
    validate_native_common_preparation_balance(system); _precision(precision)
    internal = _coordinates(preparation_fraction,internal_coordinates)
    key = system.binding_sha256,preparation_fraction,internal,precision
    previous = _EVALUATIONS.get(key)
    if previous is not None:
        return _guard(previous,NativeCommonPreparationBalanceResponse,_RESPONSES)
    from .geometric_native_common_preparation_contacts import evaluate_native_common_preparation_contacts
    contacts = evaluate_native_common_preparation_contacts(_CONTEXTS[system.binding_sha256],
        preparation_fraction=preparation_fraction,internal_coordinates=internal,precision=precision)
    if (contacts.coordinate_order != system.coordinate_order or len(contacts.gradient_bounds) != 15
            or len(contacts.hessian_bounds) != 15 or any(len(row) != 15 for row in contacts.hessian_bounds)
            or contacts.simultaneous_owner_rotation_directional_first_derivative_bounds != (I(0),)*3
            or contacts.original_atom_count != 16 or contacts.original_material_point_count != 32):
        raise ValueError('The actual source16 contact component must retain its complete fifteen-coordinate twojet')
    cold,correction,coldg,coldH,receipts = _cold_twojet(system,preparation_fraction,internal,precision)
    g,H = [coldg[i]+contacts.gradient_bounds[i] for i in range(15)],[
        [coldH[i][j]+contacts.hessian_bounds[i][j] for j in range(15)] for i in range(15)]
    for i in range(6):
        g[i] -= system.owner_torque_bounds_nm[i]
        for j in range(6):
            H[i][j] += (system.owner_drag_bounds_nm_s[i][j]+system.owner_drag_bounds_nm_s[j][i])/(2*system.owner_drag_time_scale_s)
    # Component objectivity and the exact contact global-moment identity give
    # the common-spin derivative without subtracting wide individual forces.
    spin = tuple(_round(-system.owner_torque_bounds_nm[i]-system.owner_torque_bounds_nm[i+3],precision) for i in range(3))
    if _bounds(spin[2]).hi >= 0:
        raise ValueError('The authentic owner z load must prove strict common-spin descent')
    total = _round(cold+correction+contacts.energy_bounds_j+system.hot_fixed_energy_j,precision)
    coldwork = _work(system.cold_energy_polynomial_coefficients_j,preparation_fraction,precision)
    contactwork = contacts.energy_bounds_j-system.source_contact_literal_energy_j
    work = _round(coldwork+correction+contactwork,precision)
    if preparation_fraction == 0 and not any(internal):
        work = F(0)
    lower2,lower4 = (_bounds(x).lo for x in system.inactive_complement_half_norm_coefficients_j)
    fixedlower = preparation_fraction**2*lower2+preparation_fraction**4*lower4-system.eight_literal_beam_relaxation_cap_j+system.source_contact_relaxation_lower_bound_j
    partitions = dict(cold_whole_material_and_existing_eight_beams_bounds_j=_payload(cold),
        same_eight_internal_beam_driver_correction_bounds_j=_payload(correction),
        frozen_hot30_energy_bounds_j=_payload(system.hot_fixed_energy_j),
        actual_source16_contact_loading_potential_bounds_j=_payload(contacts.energy_bounds_j),
        actual_source16_contact_recoverable_bounds_j=_payload(contacts.recoverable_energy_bounds_j),
        actual_source16_contact_fracture_work_bounds_j=_payload(contacts.fracture_work_bounds_j),
        frozen_owner6_load_drag_potential_bounds_j=_payload(F(0)),total_bounds_j=_payload(total),
        old891_complete_beam_bulk_operator_added_again=False,old_scalar_hot_foundation_added_again=False,
        hot_owner_parameter_derivatives_exact_zero=True,owner_load_is_new_frozen_increment_potential=True,
        physical_loading_work_recovered=False)
    result = NativeCommonPreparationBalanceResponse(system.binding_sha256,contacts.binding_sha256,
        system.coordinate_order,(F(0),)*6+internal+(preparation_fraction,),preparation_fraction,internal,
        _bounds(total),_bounds(work),_bounds(-work),tuple(_bounds(_round(x,precision)) for x in g),
        tuple(tuple(_bounds(_round(x,precision)) for x in row) for row in H),system.coordinate_units,
        system.gradient_units,system.hessian_units,_json(partitions),receipts,spin,fixedlower,
        fixedlower-system.owner_potential_relaxation_cap_j,_bounds(_round(total-system.registered_source_energy_bounds_j,precision)),
        fixedlower-(system.registered_source_energy_bounds_j.hi-system.prepared_zero_total_energy_j),precision,'')
    result = _remember(result,_RESPONSES); _EVALUATIONS[key] = result
    return result


def native_common_preparation_balance_response_to_dict(response):
    return dict(format=RESPONSE_FORMAT,**_payload(_guard(response,NativeCommonPreparationBalanceResponse,_RESPONSES)))


def native_common_preparation_balance_response_from_dict(raw,*,system,expected_preparation_fraction,
                                                        expected_internal_coordinates=None,precision=96):
    if type(raw) is not dict or raw.get('format') != RESPONSE_FORMAT:
        raise ValueError('A complete fifteen-coordinate preparation response DTO is required')
    value = evaluate_native_common_preparation_balance(system,preparation_fraction=expected_preparation_fraction,
        internal_coordinates=expected_internal_coordinates,precision=precision)
    if _json(native_common_preparation_balance_response_to_dict(value)) != _json(raw):
        raise ValueError('A saved preparation response cannot change its exact expected query, sources or work scope')
    return value


def compare_native_common_preparation_balance_states(system,before_fraction,after_fraction,*,
        before_internal_coordinates=None,after_internal_coordinates=None,precision=96):
    """Signed declared potential differences, not physical reverse unloading."""
    before = evaluate_native_common_preparation_balance(system,preparation_fraction=before_fraction,
        internal_coordinates=before_internal_coordinates,precision=precision)
    after = evaluate_native_common_preparation_balance(system,preparation_fraction=after_fraction,
        internal_coordinates=after_internal_coordinates,precision=precision)
    if before_fraction == after_fraction and before.internal_coordinates == after.internal_coordinates:
        work = F(0)
    else:
        cold = _difference(system.cold_energy_polynomial_coefficients_j,before_fraction,after_fraction,precision)
        bpart,apart = json.loads(before.energy_partitions_json),json.loads(after.energy_partitions_json)
        work = _round(cold+_saved(apart['same_eight_internal_beam_driver_correction_bounds_j'])
            -_saved(bpart['same_eight_internal_beam_driver_correction_bounds_j'])
            +_saved(apart['actual_source16_contact_loading_potential_bounds_j'])
            -_saved(bpart['actual_source16_contact_loading_potential_bounds_j']),precision)
    result = dict(format='native-common-preparation-balance-state-difference-1',
        system_binding_sha256=system.binding_sha256,before_response_binding_sha256=before.binding_sha256,
        after_response_binding_sha256=after.binding_sha256,before_preparation_fraction=_payload(before_fraction),
        after_preparation_fraction=_payload(after_fraction),before_internal_coordinates=_payload(before.internal_coordinates),
        after_internal_coordinates=_payload(after.internal_coordinates),potential_energy_change_bounds_j=_payload(_bounds(work)),
        opposite_reaction_potential_difference_bounds_j=_payload(_bounds(-work)),
        declared_loading_potential_state_difference_computed=True,physical_reverse_unloading_authorized=False,
        temporal_path_work_certified=False,reference_preparation_work_paid=False,clocks_advanced=False,
        accepted_new_native_intervals=0,precision=precision)
    return dict(**result,sha256=_digest(result))


def native_common_preparation_balance_state_difference_from_dict(raw,*,system,before_fraction,after_fraction,
        before_internal_coordinates=None,after_internal_coordinates=None,precision=96):
    value = compare_native_common_preparation_balance_states(system,before_fraction,after_fraction,
        before_internal_coordinates=before_internal_coordinates,after_internal_coordinates=after_internal_coordinates,
        precision=precision)
    if type(raw) is not dict or _json(raw) != _json(value):
        raise ValueError('A state difference cannot change its complete expected endpoints or physical scope')
    return value


def require_native_common_preparation_full_physical_solve(system):
    validate_native_common_preparation_balance(system)
    raise ValueError('A full physical mixed solve requires authentic cold56/HOT30 basal-modal load jets, '
                     'physical preparation work, born HOT contacts and the general-state material operator; '
                     'the missing86 loads cannot be filled with zero')
