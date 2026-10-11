"""Source-bound nonlocal cold strain energy and preparation-coordinate work.

The complete declared899 law is retained, including its global eight-tip
projection, original material measure, literal beam ledgers and prestress.
For E=H*t+R, r_i=integral H_i:C:R and S=integral R:C:R/2:

    p=t+A*r; U=U_literal+f_literal*p+K*p*p/2+S-r*A*r/2.

Supplied new moment two-jets are declarations, not integration certificates.
The saved zero-to-bound modal path is physical modal motion on the fixed895
reference. It is not reference preparation, elapsed time, or admission.
"""
from dataclasses import dataclass, fields, replace
from fractions import Fraction as F
import hashlib
import json

from .geometric_native_pose import FractionInterval as I, _rounded_interval

SOURCE_RAW_SHA256 = '4a5c059176cef063bf12ff7bd98df830395504d6c12051f269a177fcbc375fec'
SOURCE_PACKET_SHA256 = 'c863be05c9d641025666a1a4c3929701c7d472f9d574085926d64a2d08b4c439'
SOURCE_PATH = 'analysis/geometric_native_common_cold_continuum_energy_validation/release_validation_01/native_control/source_packet.json'
_SOURCES, _MOMENTS, _RESPONSES, _SAVED_ROWS, _SAVED_MOMENTS = {}, {}, {}, {}, {}


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def _payload(value):
    if type(value) is F:
        return [value.numerator, value.denominator]
    if type(value) is I:
        return {'lo': _payload(value.lo), 'hi': _payload(value.hi)}
    if type(value) is tuple:
        return [_payload(x) for x in value]
    if hasattr(type(value), '__dataclass_fields__'):
        return {f.name: _payload(getattr(value, f.name)) for f in fields(value)}
    if type(value) in (str, int, bool) or value is None:
        return value
    raise ValueError('Undeclared cold variational scalar or container kind')


def _seal(value):
    body = {f.name: _payload(getattr(value, f.name)) for f in fields(value) if f.name != 'binding_sha256'}
    return hashlib.sha256(_json(body).encode()).hexdigest()


def _source_seal(value):
    # Large immutable complete source JSON is committed by its exact raw
    # string hash, avoiding JSON escaping121MB on each source guard.
    body = {f.name: (hashlib.sha256(getattr(value, f.name).encode()).hexdigest()
            if f.name.endswith('_json') else _payload(getattr(value, f.name)))
            for f in fields(value) if f.name != 'binding_sha256'}
    return hashlib.sha256(_json(body).encode()).hexdigest()


def _fraction(raw):
    if type(raw) is not list or len(raw) != 2 or any(type(x) is not int for x in raw) or raw[1] <= 0:
        raise ValueError('A literal canonical source fraction is required')
    value = F(*raw)
    if [value.numerator, value.denominator] != raw:
        raise ValueError('Noncanonical source fraction')
    return value


def _scalar(value):
    if type(value) not in (F, I):
        raise ValueError('Cold moments require exact fractions or outward intervals')
    return value


def _saved(raw):
    if type(raw) is dict and set(raw) == {'lo', 'hi'}:
        return I(_fraction(raw['lo']), _fraction(raw['hi']))
    return _fraction(raw)


def _vector(value, n):
    if type(value) is not tuple or len(value) != n:
        raise ValueError('Complete immutable moment vector required')
    return tuple(_scalar(x) for x in value)


def _matrix(value, n, m):
    if type(value) is not tuple or len(value) != n:
        raise ValueError('Complete immutable moment matrix required')
    return tuple(_vector(row, m) for row in value)


def _symmetric(value, n):
    result = _matrix(value, n, n)
    if any(result[i][j] != result[j][i] for i in range(n) for j in range(i)):
        raise ValueError('Potential moment second variations must be symmetric')
    return result


def _order(value):
    if (type(value) is not tuple or not value or any(type(x) is not tuple or not x
            or any(type(y) is not str for y in x) for x in value) or len(set(value)) != len(value)):
        raise ValueError('All distinct literal coordinate labels are required')
    return value


def _sum(values):
    return sum(values, F(0))


def _dot(a, b):
    return _sum(x*y for x, y in zip(a, b))


def _mv(a, b):
    return tuple(_dot(row, b) for row in a)


def _round(value, precision):
    return _rounded_interval(value, precision+16) if type(value) is I else value


def _square(value):
    if type(value) is F:
        return value*value
    return I(F(0) if value.lo <= 0 <= value.hi else min(value.lo**2, value.hi**2),
             max(value.lo**2, value.hi**2))


@dataclass(frozen=True, slots=True)
class NativeColdVariationalSource:
    source_raw_sha256: str
    source_packet_sha256: str
    complete_source_system_json: str
    complete_saved_queries_json: str
    coordinate_order: tuple
    tip_coordinate_indices: tuple
    literal_reference_energies_j: tuple
    literal_reference_forces_n: tuple
    tip_stiffnesses_n_m: tuple
    gram_bounds_n_m: tuple
    inverse_gram_bounds_m_n: tuple
    complete_gram_inverse_certificate_json: str
    binding_sha256: str


def bind_native_cold_variational_source(packet):
    """Authenticate the complete frozen899 packet; no old physics is run."""
    if type(packet) is not dict or packet.get('packet_sha256') != SOURCE_PACKET_SHA256:
        raise ValueError('The complete immutable899 source is required')
    canonical = _json(packet).encode()
    if hashlib.sha256(canonical+b'\n').hexdigest() != SOURCE_RAW_SHA256:
        raise ValueError('Complete899 source bytes differ from their frozen authority')
    if hashlib.sha256(_json({k:v for k,v in packet.items() if k != 'packet_sha256'}).encode()).hexdigest() != SOURCE_PACKET_SHA256:
        raise ValueError('The complete899 source seal changed')
    if (packet.get('complete_cold_continuum_operator_certified') is not True
            or packet.get('complete_declared_cold_continuum_restart_verified') is not True):
        raise ValueError('Complete whole-domain899 constitutive authority required')
    system = packet['cold_continuum_energy_system']
    queries = packet['evaluation_queries']
    banks = system['banks']
    if (len(banks) != 8 or len(queries) != 2 or system['source_beams_included_once'] is not True
            or system['material_reference_measure'] != 'AUTHENTIC_ORIGINAL_AREAL_VOLUME_R_SQUARED_DOMEGA_DZ_ONCE'):
        raise ValueError('All eight literal ledgers and the original measure must remain once')
    order = _order(tuple(tuple(x) for x in system['coordinate_order']))
    cert = queries[0]['response']['gram_inverse_certificate']
    for query in queries:
        response = query['response']; raw = response['integrals']; proof = raw['quadrature_certificate']
        if (response['system_binding_sha256'] != system['binding_sha256']
                or raw['system_binding_sha256'] != system['binding_sha256']
                or response['coordinate_order'] != system['coordinate_order']
                or raw['coordinate_increments'] != response['coordinate_increments']
                or proof['whole_domain_enclosure_certified'] is not True
                or proof['authentic_original_volume_included_once'] is not True
                or proof['complete_original_parent_ids'] != ['octant:'+str(i) for i in range(8)]
                or len(proof['nonempty_archive_domain_ids']) != 3
                or response['gram_inverse_certificate'] != cert
                or cert['positive_definite_certified'] is not True
                or cert['global_eight_tip_overlap_retained'] is not True
                or raw['gram_bounds_n_m'] != cert['gram_bounds_n_m']):
            raise ValueError('Whole source moments, material masks or global Gram authority changed')
    value = NativeColdVariationalSource(SOURCE_RAW_SHA256, SOURCE_PACKET_SHA256, _json(system),
        _json(queries), order, tuple(b['tip_coordinate_index'] for b in banks),
        tuple(_fraction(b['source_reference_energy_j']) for b in banks),
        tuple(_fraction(b['source_reference_force_n']) for b in banks),
        tuple(_fraction(b['source_old_tip_stiffness_n_m']) for b in banks),
        tuple(tuple(_saved(x) for x in row) for row in cert['gram_bounds_n_m']),
        tuple(tuple(_saved(x) for x in row) for row in cert['inverse_gram_bounds_m_n']), _json(cert), '')
    value = replace(value, binding_sha256=_source_seal(value))
    existing = _SOURCES.get(value.binding_sha256)
    if existing is not None:
        return _source(existing)
    _SOURCES[value.binding_sha256] = value
    _SAVED_ROWS[value.binding_sha256] = json.loads(value.complete_saved_queries_json)
    return value


def _source(value):
    if (type(value) is not NativeColdVariationalSource or _SOURCES.get(value.binding_sha256) is not value
            or _source_seal(value) != value.binding_sha256):
        raise ValueError('An altered or unauthenticated whole cold source is not authorized')
    return value


@dataclass(frozen=True, slots=True)
class NativeColdStrainMomentTwoJet:
    source_binding_sha256: str
    coordinate_order: tuple
    tip_coefficients_m: tuple
    tip_first: tuple
    tip_second: tuple
    remainder_projection_n: tuple
    remainder_projection_first: tuple
    remainder_projection_second: tuple
    half_remainder_norm_j: object
    half_remainder_norm_first: tuple
    half_remainder_norm_second: tuple
    provenance: str
    authentic_saved_label: str | None
    whole_original_integral_certified: bool
    binding_sha256: str


def _moments(source, coordinate_order, tip_coefficients_m, tip_first, tip_second,
        remainder_projection_n, remainder_projection_first, remainder_projection_second,
        half_remainder_norm_j, half_remainder_norm_first, half_remainder_norm_second,
        provenance, label=None):
    _source(source); order = _order(coordinate_order); n = len(order)
    if type(provenance) is not str or not provenance.strip():
        raise ValueError('The source and any new strain moment declaration need provenance')
    t = _vector(tip_coefficients_m, 8); dt = _matrix(tip_first, 8, n)
    r = _vector(remainder_projection_n, 8); dr = _matrix(remainder_projection_first, 8, n)
    if type(tip_second) is not tuple or len(tip_second) != 8 or type(remainder_projection_second) is not tuple or len(remainder_projection_second) != 8:
        raise ValueError('All eight projected second variations are required')
    ddt = tuple(_symmetric(x, n) for x in tip_second)
    ddr = tuple(_symmetric(x, n) for x in remainder_projection_second)
    norm = _scalar(half_remainder_norm_j)
    if (norm.hi if type(norm) is I else norm) < 0:
        raise ValueError('A strain squared norm cannot be negative')
    dnorm = _vector(half_remainder_norm_first, n); ddnorm = _symmetric(half_remainder_norm_second, n)
    value = NativeColdStrainMomentTwoJet(source.binding_sha256, order, t, dt, ddt, r, dr, ddr,
        norm, dnorm, ddnorm, provenance, label, label is not None, '')
    value = replace(value, binding_sha256=_seal(value))
    existing = _MOMENTS.get(value.binding_sha256)
    if existing is not None:
        return _moment(source, existing)
    _MOMENTS[value.binding_sha256] = value
    return value


def declare_native_cold_strain_moments(source, *, coordinate_order, tip_coefficients_m,
        tip_first, tip_second, remainder_projection_n, remainder_projection_first,
        remainder_projection_second, half_remainder_norm_j, half_remainder_norm_first,
        half_remainder_norm_second, provenance):
    """Declare new integrated moments; this does not certify their integration."""
    return _moments(source, coordinate_order, tip_coefficients_m, tip_first, tip_second,
        remainder_projection_n, remainder_projection_first, remainder_projection_second,
        half_remainder_norm_j, half_remainder_norm_first, half_remainder_norm_second, provenance)


def saved_native_cold_strain_moments(source, label):
    """Use authentic whole899 moments, retaining the exact tip/Gram relation."""
    _source(source)
    if type(label) is not str:
        raise ValueError('Exactly one authentic saved899 label is required')
    cached = _SAVED_MOMENTS.get((source.binding_sha256, label))
    if cached is not None:
        return _moment(source, cached)
    rows = [q for q in _SAVED_ROWS[source.binding_sha256] if q['label'] == label]
    if type(label) is not str or len(rows) != 1:
        raise ValueError('Exactly one authentic saved899 label is required')
    response = rows[0]['response']; raw = response['integrals']; n = len(source.coordinate_order)
    coordinates = tuple(_fraction(x) for x in response['coordinate_increments'])
    zero = ((F(0),)*n,)*n
    value = _moments(source, source.coordinate_order,
        tuple(coordinates[i] for i in source.tip_coordinate_indices),
        tuple(tuple(F(int(i == tip)) for i in range(n)) for tip in source.tip_coordinate_indices),
        (zero,)*8, tuple(_saved(x) for x in raw['projection_remainder_n']),
        tuple(tuple(_saved(x) for x in row) for row in raw['projection_remainder_gradient_n_m']),
        tuple(tuple(tuple(_saved(x) for x in row) for row in block) for block in raw['projection_hessian_n_m2']),
        _saved(raw['remainder_strain_inner_product_j'])/2,
        tuple(_saved(x) for x in raw['remainder_local_gradient_n']),
        tuple(tuple(_saved(x) for x in row) for row in raw['remainder_local_hessian_n_m']),
        'Authentic complete whole899 moments: '+label, label)
    _SAVED_MOMENTS[source.binding_sha256, label] = value
    return value


def _moment(source, value):
    _source(source)
    if (type(value) is not NativeColdStrainMomentTwoJet or _MOMENTS.get(value.binding_sha256) is not value
            or value.source_binding_sha256 != source.binding_sha256 or _seal(value) != value.binding_sha256):
        raise ValueError('A complete registered two-jet of this source is required')
    return value


@dataclass(frozen=True, slots=True)
class NativeColdVariationalWorkResponse:
    source_binding_sha256: str
    moments_binding_sha256: str
    coordinate_order: tuple
    tip_projection_m: tuple
    energy_j: object
    gradient: tuple
    hessian: tuple
    global_projection_curvature_hessian: tuple
    authentic_saved_label: str | None
    whole_original_integral_certified: bool
    binding_sha256: str
    source_beams_included_once: bool = True
    historical_three_dimensional_prestress_recovered: bool = False
    reference_preparation_work_paid: bool = False
    whole_clock_all_pair_admission_authorized: bool = False
    clocks_advanced: bool = False
    accepted_new_native_intervals: int = 0


def evaluate_native_cold_variational_work(source, moments, *, precision=96):
    """Evaluate the complete nonlocal potential and any number of mode channels."""
    _moment(source, moments)
    if type(precision) is not int or not 96 <= precision <= 512:
        raise ValueError('Variational precision is a literal integer from96 to512')
    rnd = lambda x: _round(x, precision)
    A = source.inverse_gram_bounds_m_n; n = len(moments.coordinate_order)
    r = moments.remainder_projection_n; dr = moments.remainder_projection_first
    pr = tuple(rnd(x) for x in _mv(A, r))
    p = tuple(rnd(t+x) for t,x in zip(moments.tip_coefficients_m, pr))
    ar = tuple(tuple(rnd(_sum(A[k][l]*dr[l][i] for l in range(8))) for i in range(n)) for k in range(8))
    dp = tuple(tuple(rnd(moments.tip_first[k][i]+ar[k][i]) for i in range(n)) for k in range(8))
    f = tuple(rnd(f0+K*x) for f0,K,x in zip(source.literal_reference_forces_n, source.tip_stiffnesses_n_m, p))
    c = tuple(rnd(x-y) for x,y in zip(_mv(A, f), pr))
    beam = rnd(_sum(E0+f0*x+K*_square(x)/2 for E0,f0,K,x in zip(
        source.literal_reference_energies_j, source.literal_reference_forces_n, source.tip_stiffnesses_n_m, p)))
    orthogonal = rnd(moments.half_remainder_norm_j-_dot(r, pr)/2)
    if type(orthogonal) is I:
        if orthogonal.hi < 0:
            raise ValueError('Integrated moments contradict positive orthogonal strain energy')
        orthogonal = I(max(F(0), orthogonal.lo), orthogonal.hi)
    elif orthogonal < 0:
        raise ValueError('Integrated moments contradict positive orthogonal strain energy')
    energy = rnd(beam+orthogonal)
    gradient = tuple(rnd(moments.half_remainder_norm_first[i]+_sum(
        f[k]*dp[k][i]-pr[k]*dr[k][i] for k in range(8))) for i in range(n))
    H = [[None]*n for _ in range(n)]; curvature = [[None]*n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            curve = rnd(_sum(f[k]*moments.tip_second[k][i][j]
                +c[k]*moments.remainder_projection_second[k][i][j] for k in range(8)))
            value = rnd(moments.half_remainder_norm_second[i][j]+curve+_sum(
                source.tip_stiffnesses_n_m[k]*dp[k][i]*dp[k][j]-dr[k][i]*ar[k][j] for k in range(8)))
            H[i][j] = H[j][i] = value; curvature[i][j] = curvature[j][i] = curve
    result = NativeColdVariationalWorkResponse(source.binding_sha256, moments.binding_sha256,
        moments.coordinate_order, p, energy, gradient, tuple(tuple(row) for row in H),
        tuple(tuple(row) for row in curvature), moments.authentic_saved_label,
        moments.whole_original_integral_certified, '')
    result = replace(result, binding_sha256=_seal(result))
    existing = _RESPONSES.get(result.binding_sha256)
    if existing is not None:
        native_cold_variational_work_to_dict(existing)
        return existing
    _RESPONSES[result.binding_sha256] = result
    return result


def native_cold_variational_source_to_dict(source):
    _source(source)
    return {'format': 'native-cold-variational-source-1', **_payload(source)}


def native_cold_variational_source_from_dict(raw, *, source_packet):
    source = bind_native_cold_variational_source(source_packet)
    if type(raw) is not dict or _json(raw) != _json(native_cold_variational_source_to_dict(source)):
        raise ValueError('A complete899 source restart changed')
    return source


def native_cold_strain_moments_to_dict(source, moments):
    _moment(source, moments)
    return {'format': 'native-cold-strain-moment-twojet-1', **_payload(moments)}


def native_cold_strain_moments_from_dict(raw, *, source):
    if type(raw) is not dict or raw.get('format') != 'native-cold-strain-moment-twojet-1':
        raise ValueError('A complete cold moment DTO is required')
    if raw.get('authentic_saved_label') is not None:
        value = saved_native_cold_strain_moments(source, raw['authentic_saved_label'])
    else:
        parsevector = lambda row: tuple(_saved(x) for x in row)
        parsematrix = lambda rows: tuple(parsevector(row) for row in rows)
        parseblocks = lambda blocks: tuple(parsematrix(rows) for rows in blocks)
        value = declare_native_cold_strain_moments(source,
            coordinate_order=tuple(tuple(x) for x in raw['coordinate_order']),
            tip_coefficients_m=parsevector(raw['tip_coefficients_m']), tip_first=parsematrix(raw['tip_first']),
            tip_second=parseblocks(raw['tip_second']), remainder_projection_n=parsevector(raw['remainder_projection_n']),
            remainder_projection_first=parsematrix(raw['remainder_projection_first']),
            remainder_projection_second=parseblocks(raw['remainder_projection_second']),
            half_remainder_norm_j=_saved(raw['half_remainder_norm_j']),
            half_remainder_norm_first=parsevector(raw['half_remainder_norm_first']),
            half_remainder_norm_second=parsematrix(raw['half_remainder_norm_second']), provenance=raw['provenance'])
    if _json(raw) != _json(native_cold_strain_moments_to_dict(source, value)):
        raise ValueError('A cold moment restart changed its source, fields or literal kinds')
    return value


def native_cold_variational_work_to_dict(response):
    if (type(response) is not NativeColdVariationalWorkResponse
            or _RESPONSES.get(response.binding_sha256) is not response
            or _seal(response) != response.binding_sha256):
        raise ValueError('A complete cold work response is required')
    return {'format': 'native-cold-variational-work-response-1', **_payload(response)}


def native_cold_variational_work_from_dict(raw, *, source, moments, precision=96):
    result = evaluate_native_cold_variational_work(source, moments, precision=precision)
    if type(raw) is not dict or _json(raw) != _json(native_cold_variational_work_to_dict(result)):
        raise ValueError('A cold work response restart changed its full operator or source')
    return result


def compare_saved_native_cold_state_work(source, before_label, after_label, *, precision=96):
    """State energy difference; a temporal path or preparation payment is not inferred."""
    before = evaluate_native_cold_variational_work(source, saved_native_cold_strain_moments(source, before_label), precision=precision)
    after = evaluate_native_cold_variational_work(source, saved_native_cold_strain_moments(source, after_label), precision=precision)
    work = _round(after.energy_j-before.energy_j, precision)
    return {'format': 'native-cold-saved-state-work-1', 'source_binding_sha256': source.binding_sha256,
        'before_label': before_label, 'after_label': after_label, 'potential_energy_change_j': _payload(work),
        'physical_reaction_work_j': _payload(-work), 'temporal_path_certified': False,
        'reference_preparation_work_paid': False, 'clocks_advanced': False, 'accepted_new_native_intervals': 0}


def evaluate_native_cold_affine_modal_path_work(source, lambda_fraction, *, precision=96):
    """Whole original-material zero-to-bound affine modal path, from frozen moments.

    R(lambda)=lambda*R1+lambda**2*R2. Endpoint directional derivatives
    recover its projected quadratic and squared-norm quartic coefficients.
    All lambda values remain in the fixed895 coordinate envelope. Their
    integral coverage is inherited899; physical temporal admission is absent.
    """
    _source(source)
    if type(lambda_fraction) is not F or not 0 <= lambda_fraction <= 1:
        raise ValueError('The actual affine modal path uses a literal fraction in[0,1]')
    if type(precision) is not int or not 96 <= precision <= 512:
        raise ValueError('Path precision is a literal integer from96 to512')
    rnd = lambda x: _round(x, precision)
    rows = {q['label']:q['response'] for q in _SAVED_ROWS[source.binding_sha256]}
    endpoint = rows['registered_cold_endpoint']; zero = rows['prepared_zero_cold_reference']
    a = tuple(_fraction(x) for x in endpoint['coordinate_increments'])
    if any(_fraction(x) for x in zero['coordinate_increments']):
        raise ValueError('The authenticated path must start at prepared zero')
    moments = saved_native_cold_strain_moments(source, 'registered_cold_endpoint')
    r = moments.remainder_projection_n
    rp = tuple(rnd(_dot(row,a)) for row in moments.remainder_projection_first)
    r2 = tuple(rnd(x-y) for x,y in zip(rp,r))
    r1 = tuple(rnd(2*y-x) for x,y in zip(rp,r))
    S = moments.half_remainder_norm_j
    Sp = rnd(_dot(moments.half_remainder_norm_first,a))
    Spp = rnd(_sum(a[i]*moments.half_remainder_norm_second[i][j]*a[j] for i in range(56) for j in range(56)))
    # S=c2*l^2+c3*l^3+c4*l^4; zero second derivative is not needed.
    c2 = rnd(6*S-3*Sp+Spp/2); c3 = rnd(5*Sp-Spp-8*S); c4 = rnd((Spp-4*Sp+6*S)/2)
    A = source.inverse_gram_bounds_m_n
    p1 = tuple(rnd(a[i]+x) for i,x in zip(source.tip_coordinate_indices,_mv(A,r1)))
    p2 = tuple(rnd(x) for x in _mv(A,r2))
    u0 = _sum(source.literal_reference_energies_j)
    u1 = rnd(_dot(source.literal_reference_forces_n,p1))
    u2 = rnd(_dot(source.literal_reference_forces_n,p2)+_sum(K*_square(x)/2 for K,x in zip(source.tip_stiffnesses_n_m,p1))
        +c2-_dot(r1,_mv(A,r1))/2)
    u3 = rnd(_sum(K*x*y for K,x,y in zip(source.tip_stiffnesses_n_m,p1,p2))+c3
        -(_dot(r1,_mv(A,r2))+_dot(r2,_mv(A,r1)))/2)
    u4 = rnd(_sum(K*_square(x)/2 for K,x in zip(source.tip_stiffnesses_n_m,p2))+c4-_dot(r2,_mv(A,r2))/2)
    coefficients = (u0,u1,u2,u3,u4); lam = lambda_fraction
    energy = rnd(_sum(value*lam**k for k,value in enumerate(coefficients)))
    work = rnd(_sum(value*lam**k for k,value in enumerate(coefficients[1:],1)))
    power = rnd(_sum(k*value*lam**(k-1) for k,value in enumerate(coefficients) if k))
    second = rnd(_sum(k*(k-1)*value*lam**(k-2) for k,value in enumerate(coefficients) if k > 1))
    return {'format':'native-cold-affine-modal-path-work-1', 'source_binding_sha256':source.binding_sha256,
        'lambda_fraction':_payload(lam), 'energy_polynomial_coefficients_j':_payload(coefficients),
        'remainder_projection_linear_n':_payload(r1), 'remainder_projection_quadratic_n':_payload(r2),
        'half_remainder_norm_coefficients_j':_payload((c2,c3,c4)),
        'tip_projection_linear_m':_payload(p1), 'tip_projection_quadratic_m':_payload(p2),
        'energy_j':_payload(energy), 'potential_path_work_j':_payload(work),
        'physical_reaction_path_work_j':_payload(-work), 'potential_path_power_j':_payload(power),
        'potential_path_second_derivative_j':_payload(second),
        'all_eight_original_parents_and_three_archive_cohorts_covered':True,
        'fixed895_coordinate_envelope_preserved':True, 'source_beams_included_once':True,
        'physical_modal_motion_defined':True, 'whole_original_integral_certified':True,
        'physical_temporal_path_admission_certified':False, 'reference_preparation_work_paid':False,
        'historical_three_dimensional_prestress_recovered':False, 'clocks_advanced':False,
        'accepted_new_native_intervals':0}


def native_cold_affine_modal_path_work_from_dict(raw, *, source, expected_lambda_fraction,
        precision=96):
    """Replay the complete source-bound path work at its explicit fixed label."""
    expected = evaluate_native_cold_affine_modal_path_work(source, expected_lambda_fraction,
        precision=precision)
    if type(raw) is not dict or _json(raw) != _json(expected):
        raise ValueError('A modal path work restart changed its source, parameter or work')
    return expected
