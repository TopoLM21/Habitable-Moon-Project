"""Whole-source work of the physical cold depth-twist preparation.

The frozen899 nonlocal law is evaluated on a registered, newly integrated
T(lambda) path. At zero cold modes its relative Green strain is
R=lambda*L+lambda**2*Q; no prepared-reference metric or old energy is reset.
The complete eight-tip inverse Gram, literal forces and beam energies remain
once. Only the cold constitutive contribution is calculated here; activation,
HOT/contact/load work, historical3D prestress and temporal admission are not
certified by this operator.
"""
from dataclasses import dataclass, fields, replace
from fractions import Fraction as F
import hashlib

from .geometric_native_pose import FractionInterval as I
from .geometric_native_common_cold_variational_work import (
    _source, _scalar, _sum, _dot, _mv, _round, _square, _payload, _json,
)
from .geometric_native_common_cold_preparation_integrals import (
    NativeColdPreparationIntegral, validate_native_common_cold_preparation_integral,
)

_ENERGIES, _RESPONSES = {}, {}


def _seal(value):
    body = {f.name: _payload(getattr(value, f.name)) for f in fields(value)
            if f.name != 'binding_sha256'}
    return hashlib.sha256(_json(body).encode()).hexdigest()


def _precision(value):
    if type(value) is not int or not 96 <= value <= 512:
        raise ValueError('Preparation work precision is a literal integer from96 to512')
    return value


def _parameter(value):
    if type(value) not in (F, I):
        raise ValueError('The physical preparation parameter requires a Fraction or outward interval')
    if type(value) is I and (type(value.lo) is not F or type(value.hi) is not F):
        raise ValueError('Preparation interval endpoints require exact literal Fractions')
    lo, hi = (value.lo, value.hi) if type(value) is I else (value, value)
    if not 0 <= lo <= hi <= 1:
        raise ValueError('The complete preparation parameter must lie in[0,1]')
    return value


def _bounds(value):
    return value if type(value) is I else I(value)


def _coefficients(A, u0, forces, stiffnesses, r1, r2, S2, S3, S4, precision):
    """Pure polynomial arithmetic, carrying no material integration authority."""
    rnd = lambda x: _round(x, precision)
    p1, p2 = tuple(rnd(x) for x in _mv(A, r1)), tuple(rnd(x) for x in _mv(A, r2))
    residual = (rnd(S2-_dot(r1, p1)/2),
        rnd(S3-(_dot(r1, p2)+_dot(r2, p1))/2), rnd(S4-_dot(r2, p2)/2))
    u1 = rnd(_dot(forces, p1))
    u2 = rnd(_dot(forces, p2)+_sum(K*_square(p)/2 for K,p in zip(stiffnesses,p1))+residual[0])
    u3 = rnd(_sum(K*x*y for K,x,y in zip(stiffnesses,p1,p2))+residual[1])
    u4 = rnd(_sum(K*_square(p)/2 for K,p in zip(stiffnesses,p2))+residual[2])
    return (u0,u1,u2,u3,u4), p1, p2, residual


def _polynomial(coefficients, parameter, precision):
    if not coefficients:
        return F(0)
    result = coefficients[-1]
    for coefficient in reversed(coefficients[:-1]):
        result = _round(coefficient+parameter*result, precision)
    return result


def _work(coefficients, parameter, precision):
    if _bounds(parameter).hi == 0:
        return F(0)
    return _round(parameter*_polynomial(coefficients[1:],parameter,precision),precision)


def _difference(coefficients, before, after, precision):
    """Exact divided-difference recurrence; the fixed constant cancels first."""
    if _bounds(before).width == _bounds(after).width == 0 and _bounds(before) == _bounds(after):
        return F(0)
    powers_before, powers_after = [F(1)], [F(1)]
    for _ in range(3):
        powers_before.append(_round(powers_before[-1]*before,precision))
        powers_after.append(_round(powers_after[-1]*after,precision))
    divided = _sum(c*_sum(powers_after[k-1-j]*powers_before[j] for j in range(k))
        for k,c in enumerate(coefficients[1:],1))
    return _round((after-before)*divided,precision)


def _sign(value):
    bounds = _bounds(value)
    if bounds.lo > 0:
        return 'POSITIVE'
    if bounds.hi < 0:
        return 'NEGATIVE'
    return 'ZERO' if bounds.lo == bounds.hi == 0 else 'UNRESOLVED'


def _positive_work_bracket(S2_lower, S4_lower, relaxation_cap, precision):
    """Bracket the zero of the independent lower bound, not the actual work."""
    if min(S2_lower,S4_lower,relaxation_cap) < 0:
        raise ValueError('The inactive complement and beam relaxation cap must be nonnegative')
    def phi(lam):
        return lam*lam*S2_lower+lam**4*S4_lower-relaxation_cap
    if phi(F(1)) < 0 or S2_lower+S4_lower == 0:
        return None
    if not relaxation_cap:
        return I(0)
    if not phi(F(1)):
        return I(1)
    lo, hi = F(0), F(1)
    for _ in range(4*precision):
        middle = (lo+hi)/2
        value = phi(middle)
        if not value:
            return I(middle)
        if value > 0:
            hi = middle
        else:
            lo = middle
    return I(lo,hi)


@dataclass(frozen=True, slots=True)
class NativeColdPreparationEnergy:
    source_binding_sha256: str
    integrals_binding_sha256: str
    original_parent_ids: tuple
    nonempty_archive_domain_ids: tuple
    energy_polynomial_coefficients_j: tuple
    tip_projection_linear_m: tuple
    tip_projection_quadratic_m: tuple
    orthogonal_residual_polynomial_coefficients_j: tuple
    inactive_complement_half_norm_coefficients_j: tuple
    eight_literal_beam_relaxation_cap_j: F
    independent_positive_work_parameter_bracket: I | None
    precision: int
    binding_sha256: str
    strain_tensor_frame: str = 'FROZEN899_PREPARED_RELATIVE_WORLD_GREEN_STRAIN'
    source_beams_included_once: bool = True
    authentic_original_volume_included_once: bool = True
    whole_original_integral_certified: bool = True
    cold_constitutive_preparation_work_computed: bool = True
    historical_three_dimensional_prestress_recovered: bool = False
    reference_preparation_work_paid: bool = False
    complete_physical_system_preparation_work_paid: bool = False
    whole_clock_all_pair_admission_authorized: bool = False
    general_state_mixed_operator_certified: bool = False
    clocks_advanced: bool = False
    accepted_new_native_intervals: int = 0


def bind_native_common_cold_preparation_energy(source, integral, *, precision=96):
    """Bind only a genuinely computed whole-material preparation integral."""
    _source(source); _precision(precision)
    if type(integral) is not NativeColdPreparationIntegral:
        raise ValueError('Supplied moment arrays cannot authorize whole preparation work')
    validate_native_common_cold_preparation_integral(source,integral)
    if (integral.source_binding_sha256 != source.binding_sha256
            or integral.original_parent_ids != tuple('octant:'+str(i) for i in range(8))
            or len(integral.nonempty_archive_domain_ids) != 3
            or len(set(integral.nonempty_archive_domain_ids)) != 3
            or integral.whole_original_integral_certified is not True
            or integral.authentic_original_volume_included_once is not True
            or integral.complete_modal_zero_clamp_complement_integrated is not True
            or integral.relative_prepared_cartesian_green_strain_used is not True
            or integral.inactive_complement_tip_strain_identically_zero is not True):
        raise ValueError('All authentic original parents and archive cohorts must remain once')
    u0 = _sum(source.literal_reference_energies_j)
    coefficients,p1,p2,residual = _coefficients(source.inverse_gram_bounds_m_n,u0,
        source.literal_reference_forces_n,source.tip_stiffnesses_n_m,
        integral.r1_bounds_n,integral.r2_bounds_n,integral.S2_bounds_j,
        integral.S3_bounds_j,integral.S4_bounds_j,precision)
    complement = tuple(_scalar(x) for x in (integral.inactive_complement_S2_bounds_j,
        integral.inactive_complement_S4_bounds_j))
    lower2,lower4 = tuple(_bounds(x).lo for x in complement)
    cap = _sum(f*f/(2*K) for f,K in zip(source.literal_reference_forces_n,source.tip_stiffnesses_n_m))
    bracket = _positive_work_bracket(lower2,lower4,cap,precision)
    result = NativeColdPreparationEnergy(source.binding_sha256,integral.binding_sha256,
        integral.original_parent_ids,integral.nonempty_archive_domain_ids,coefficients,
        p1,p2,residual,complement,cap,bracket,precision,'')
    result = replace(result,binding_sha256=_seal(result))
    old = _ENERGIES.get(result.binding_sha256)
    if old is not None:
        return _energy(old[0])
    _ENERGIES[result.binding_sha256] = result,source,integral
    return result


def _energy(value):
    if type(value) is not NativeColdPreparationEnergy:
        raise ValueError('A registered whole-source preparation energy is required')
    stored = _ENERGIES.get(value.binding_sha256)
    if stored is None or stored[0] is not value or _seal(value) != value.binding_sha256:
        raise ValueError('An altered or resealed preparation operator is not authorized')
    _source(stored[1]); validate_native_common_cold_preparation_integral(stored[1],stored[2])
    return value


@dataclass(frozen=True, slots=True)
class NativeColdPreparationWorkResponse:
    energy_binding_sha256: str
    preparation_fraction: object
    tip_projection_m: tuple
    energy_j: object
    potential_path_work_j: object
    physical_reaction_path_work_j: object
    potential_path_power_j: object
    physical_reaction_path_power_j: object
    potential_path_second_derivative_j: object
    unintersected_orthogonal_residual_energy_j: object
    orthogonal_residual_energy_j: object
    independent_potential_path_work_lower_bound_j: F
    independent_physical_reaction_work_upper_bound_j: F
    raw_potential_work_interval_sign: str
    certified_potential_work_sign: str
    binding_sha256: str
    orthogonal_residual_nonnegative_by_fixed_gram_projection: bool = True
    independent_work_lower_bound_uses_complete_inactive_complement: bool = True
    cold_constitutive_preparation_work_computed: bool = True
    source_beams_included_once: bool = True
    reference_preparation_work_paid: bool = False
    complete_physical_system_preparation_work_paid: bool = False
    historical_three_dimensional_prestress_recovered: bool = False
    physical_temporal_path_admission_certified: bool = False
    whole_clock_all_pair_admission_authorized: bool = False
    clocks_advanced: bool = False
    accepted_new_native_intervals: int = 0


def evaluate_native_common_cold_preparation_work(energy, lambda_fraction, *, precision=96):
    """Cold path energy, dimensionless-parameter power and signed work."""
    _energy(energy); lam = _parameter(lambda_fraction); _precision(precision)
    coefficients = energy.energy_polynomial_coefficients_j
    work = _work(coefficients,lam,precision)
    current_energy = coefficients[0] if _bounds(lam).hi == 0 else _round(coefficients[0]+work,precision)
    power = _polynomial(tuple(k*c for k,c in enumerate(coefficients) if k),lam,precision)
    second = _polynomial(tuple(k*(k-1)*c for k,c in enumerate(coefficients) if k>1),lam,precision)
    tip = tuple(_round(lam*(a+lam*b),precision) for a,b in
        zip(energy.tip_projection_linear_m,energy.tip_projection_quadratic_m))
    raw_residual = F(0) if _bounds(lam).hi == 0 else _round(lam*lam*
        _polynomial(energy.orthogonal_residual_polynomial_coefficients_j,lam,precision),precision)
    rb = _bounds(raw_residual)
    if rb.hi < 0:
        raise ValueError('Whole integrated moments contradict the fixed Gram projection proof')
    residual = F(0) if _bounds(lam).hi == 0 else I(max(F(0),rb.lo),rb.hi)
    lo = _bounds(lam).lo
    lower2,lower4 = tuple(_bounds(x).lo for x in energy.inactive_complement_half_norm_coefficients_j)
    lower = (F(0) if _bounds(lam).hi == 0 else lo*lo*lower2+lo**4*lower4
        -energy.eight_literal_beam_relaxation_cap_j)
    if _bounds(work).hi < lower:
        raise ValueError('Integrated work contradicts the independent inactive-complement lower bound')
    raw_sign = _sign(work)
    sign = 'POSITIVE' if lower>0 else raw_sign
    result = NativeColdPreparationWorkResponse(energy.binding_sha256,lam,tip,current_energy,
        work,-work,power,-power,second,raw_residual,residual,lower,-lower,raw_sign,sign,'')
    result = replace(result,binding_sha256=_seal(result))
    existing = _RESPONSES.get(result.binding_sha256)
    if existing is not None:
        native_common_cold_preparation_work_to_dict(existing)
        return existing
    _RESPONSES[result.binding_sha256] = result
    return result


def compare_native_common_cold_preparation_path_work(energy, before_fraction, after_fraction,
        *, precision=96):
    """Signed cold constitutive work between explicitly declared parameters."""
    _energy(energy); before=_parameter(before_fraction); after=_parameter(after_fraction)
    _precision(precision)
    work = _difference(energy.energy_polynomial_coefficients_j,before,after,precision)
    return {'format':'native-common-cold-preparation-path-work-1',
        'energy_binding_sha256':energy.binding_sha256,'before_fraction':_payload(before),
        'after_fraction':_payload(after),'potential_path_work_j':_payload(work),
        'physical_reaction_path_work_j':_payload(-work),'potential_work_interval_sign':_sign(work),
        'cold_constitutive_preparation_work_computed':True,'source_beams_included_once':True,
        'reference_preparation_work_paid':False,'complete_physical_system_preparation_work_paid':False,
        'historical_three_dimensional_prestress_recovered':False,
        'physical_temporal_path_admission_certified':False,'whole_clock_all_pair_admission_authorized':False,
        'clocks_advanced':False,'accepted_new_native_intervals':0}


def native_common_cold_preparation_energy_to_dict(energy):
    _energy(energy)
    return {'format':'native-common-cold-preparation-energy-1',**_payload(energy)}


def native_common_cold_preparation_energy_from_dict(raw, *, source, integral, precision=96):
    expected=bind_native_common_cold_preparation_energy(source,integral,precision=precision)
    if type(raw) is not dict or _json(raw)!=_json(native_common_cold_preparation_energy_to_dict(expected)):
        raise ValueError('A complete preparation energy restart changed its source, moments or flags')
    return expected


def native_common_cold_preparation_work_to_dict(response):
    if (type(response) is not NativeColdPreparationWorkResponse
            or _RESPONSES.get(response.binding_sha256) is not response
            or _seal(response)!=response.binding_sha256):
        raise ValueError('A registered complete preparation work response is required')
    stored = _ENERGIES.get(response.energy_binding_sha256)
    if stored is None:
        raise ValueError('A preparation response has no registered energy authority')
    _energy(stored[0])
    return {'format':'native-common-cold-preparation-work-response-1',**_payload(response)}


def native_common_cold_preparation_work_from_dict(raw, *, energy, expected_lambda_fraction,
        precision=96):
    expected=evaluate_native_common_cold_preparation_work(energy,expected_lambda_fraction,precision=precision)
    if type(raw) is not dict or _json(raw)!=_json(native_common_cold_preparation_work_to_dict(expected)):
        raise ValueError('A preparation response restart changed its parameter, work or literal types')
    return expected


def native_common_cold_preparation_path_work_from_dict(raw, *, energy, expected_before_fraction,
        expected_after_fraction, precision=96):
    expected=compare_native_common_cold_preparation_path_work(energy,expected_before_fraction,
        expected_after_fraction,precision=precision)
    if type(raw) is not dict or _json(raw)!=_json(expected):
        raise ValueError('A signed preparation path restart changed its declared endpoints or work')
    return expected
