"""Continuous work of an authenticated original inviscid contact field.

The saved local solver uses an endpoint backward-Euler work ledger.  Its
sixteen literal area/work atoms remain in ``source_field_query.history``;
they are never divided by area to manufacture a continuous work field.
Here the already authenticated affine free anchors define a separate dense
inviscid branch.  Work is its actual reaction one-form, -P dg_free + tau
dj_free, and plastic dissipation is tau dp.  The original binary64 series
stiffnesses are retained exactly.  Their small signed compliance balance
residual is explicit rather than silently replacing them by rational series
stiffnesses.  No first-root clock, finite material area or native force step
is selected or adopted by these observations.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json

from .geometric_native_contact_field_history import (
    NativeContactFieldHistoryQuery, _compressed_sin_cos_point, _field_affine_bounds, _field_bounds,
    _hull, _outward, _portable, native_contact_field_history_to_dict,
    query_native_contact_field_history,
)
from .geometric_native_contact_time_support import ADMISSIBLE, _add, _field, _scale
from .geometric_native_physical_history import (
    _dyadic, _encoded, _literal_json, _typed_fingerprint,
)
from .geometric_native_pose import FractionInterval as I


FORMAT = "native-original-continuous-contact-field-work-query-1"
INTEGRAL_FORMAT = "native-original-continuous-contact-field-work-integral-1"
PRIMITIVE_FORMAT = "native-original-continuous-contact-field-work-primitive-1"
INTERFACE_FORMAT = "native-original-continuous-contact-field-work-interface-integral-1"
MODEL = "authenticated_inviscid_down_dip_reaction_one_form_and_literal_compliance_residual"
WORK_MEASURE = "joules_per_square_metre_of_original_interface_area"
RATE_MEASURE = "joules_per_square_metre_per_original_time_fraction"
FIRST_LAW = "source_work = bulk_energy_increment + contact_energy_increment + plastic_dissipation + signed_literal_compliance_residual"


@dataclass(frozen=True, slots=True)
class NativeContactFieldWorkPieceQuery:
    previous_piece_index: int
    computational_angle_bounds: I
    free_gap_rate_bounds_m_per_fraction: I
    free_jump_rate_bounds_m_per_fraction: I
    plastic_slip_rate_bounds_m_per_fraction: I
    cumulative_slip_rate_bounds_m_per_fraction: I
    source_work_rate_bounds_j_m2_per_fraction: I
    friction_dissipation_rate_bounds_j_m2_per_fraction: I
    bulk_energy_rate_bounds_j_m2_per_fraction: I
    contact_energy_rate_bounds_j_m2_per_fraction: I
    total_energy_rate_bounds_j_m2_per_fraction: I
    compliance_balance_residual_rate_bounds_j_m2_per_fraction: I
    source_work_rate_bounds_w_m2: I
    friction_dissipation_rate_bounds_w_m2: I
    bulk_energy_rate_bounds_w_m2: I
    contact_energy_rate_bounds_w_m2: I
    total_energy_rate_bounds_w_m2: I
    compliance_balance_residual_rate_bounds_w_m2: I


_RATE_FIELDS = tuple(name for name in NativeContactFieldWorkPieceQuery.__dataclass_fields__
                     if name not in ("previous_piece_index", "computational_angle_bounds"))


@dataclass(frozen=True, slots=True)
class NativeContactFieldWorkQuery:
    source_field_query: NativeContactFieldHistoryQuery
    pieces: tuple[NativeContactFieldWorkPieceQuery, ...]
    free_gap_rate_bounds_m_per_fraction: I
    free_jump_rate_bounds_m_per_fraction: I
    plastic_slip_rate_bounds_m_per_fraction: I
    cumulative_slip_rate_bounds_m_per_fraction: I
    source_work_rate_bounds_j_m2_per_fraction: I
    friction_dissipation_rate_bounds_j_m2_per_fraction: I
    bulk_energy_rate_bounds_j_m2_per_fraction: I
    contact_energy_rate_bounds_j_m2_per_fraction: I
    total_energy_rate_bounds_j_m2_per_fraction: I
    compliance_balance_residual_rate_bounds_j_m2_per_fraction: I
    source_work_rate_bounds_w_m2: I
    friction_dissipation_rate_bounds_w_m2: I
    bulk_energy_rate_bounds_w_m2: I
    contact_energy_rate_bounds_w_m2: I
    total_energy_rate_bounds_w_m2: I
    compliance_balance_residual_rate_bounds_w_m2: I

    @property
    def computational_angle_bounds(self):
        return self.source_field_query.computational_angle_bounds

    @property
    def time_fraction_bounds(self):
        return self.source_field_query.time_fraction_bounds

    @property
    def finite_point_area_m2(self):
        return None

    @property
    def cumulative_nodal_work_density_j_m2(self):
        return None

    @property
    def candidate_mechanics_adopted(self):
        return False


@dataclass(frozen=True, slots=True)
class NativeContactFieldWorkPieceIntegral:
    previous_piece_index: int
    computational_angle_bounds: I
    source_work_density_bounds_j_m2: I
    friction_dissipation_density_bounds_j_m2: I
    bulk_energy_increment_density_bounds_j_m2: I
    contact_energy_increment_density_bounds_j_m2: I
    total_energy_increment_density_bounds_j_m2: I
    compliance_balance_residual_density_bounds_j_m2: I


_INTEGRAL_FIELDS = tuple(name for name in NativeContactFieldWorkPieceIntegral.__dataclass_fields__
                         if name not in ("previous_piece_index", "computational_angle_bounds"))


@dataclass(frozen=True, slots=True)
class NativeContactFieldWorkIntegral:
    source_field_query: NativeContactFieldHistoryQuery
    initial_time_fraction: Fraction
    final_time_fraction: Fraction
    pieces: tuple[NativeContactFieldWorkPieceIntegral, ...]
    source_work_density_bounds_j_m2: I
    friction_dissipation_density_bounds_j_m2: I
    bulk_energy_increment_density_bounds_j_m2: I
    contact_energy_increment_density_bounds_j_m2: I
    total_energy_increment_density_bounds_j_m2: I
    compliance_balance_residual_density_bounds_j_m2: I

    @property
    def computational_angle_bounds(self):
        return self.source_field_query.computational_angle_bounds

    @property
    def finite_point_area_m2(self):
        return None

    @property
    def cumulative_nodal_work_density_j_m2(self):
        return None

    @property
    def candidate_mechanics_adopted(self):
        return False


@dataclass(frozen=True, slots=True)
class NativeContactFieldWorkPrimitive:
    source_field_query: NativeContactFieldHistoryQuery
    initial_time_fraction: Fraction
    final_time_fraction_bounds: I
    pieces: tuple[NativeContactFieldWorkPieceIntegral, ...]
    source_work_density_bounds_j_m2: I
    friction_dissipation_density_bounds_j_m2: I
    bulk_energy_increment_density_bounds_j_m2: I
    contact_energy_increment_density_bounds_j_m2: I
    total_energy_increment_density_bounds_j_m2: I
    compliance_balance_residual_density_bounds_j_m2: I

    @property
    def computational_angle_bounds(self):
        return self.source_field_query.computational_angle_bounds

    @property
    def finite_point_area_m2(self):
        return None

    @property
    def candidate_mechanics_adopted(self):
        return False


@dataclass(frozen=True, slots=True)
class NativeContactFieldWorkInterfaceIntegral:
    source_time_integral: NativeContactFieldWorkIntegral
    original_interface_area_density_m2_per_computational_radian: Fraction
    continuum_interface_area_m2: Fraction
    original_nodal_interface_area_m2: Fraction
    continuum_minus_original_nodal_interface_area_m2: Fraction
    initial_bulk_energy_bounds_j: I
    initial_contact_energy_bounds_j: I
    initial_total_energy_bounds_j: I
    final_bulk_energy_bounds_j: I
    final_contact_energy_bounds_j: I
    final_total_energy_bounds_j: I
    source_work_bounds_j: I
    friction_dissipation_bounds_j: I
    bulk_energy_increment_bounds_j: I
    contact_energy_increment_bounds_j: I
    total_energy_increment_bounds_j: I
    compliance_balance_residual_bounds_j: I

    @property
    def candidate_mechanics_adopted(self):
        return False

    @property
    def original_nodal_ledgers_replaced(self):
        return False


def _fraction_clock(value, name):
    if type(value) is not Fraction or not 0 <= value <= 1:
        raise ValueError(f"{name} requires an explicit exact Fraction in the original clock")
    return value


def _support_interval(raw):
    return I(*(Fraction(*raw[end]) for end in ("lo", "hi")))


def _intersection(a, b, name):
    low, high = max(a.lo, b.lo), min(a.hi, b.hi)
    if low > high:
        raise ValueError(f"Dense {name} contradicts the complete authenticated constitutive support")
    return I(low, high)


def _family(source):
    if source.constitutive_status != ADMISSIBLE:
        raise ValueError("Continuous work requires the proved whole-time inviscid down-dip branch")
    path, trace_id = source.history.contact_time_path, source.trace_id
    position = next(index for index, trace in enumerate(path.initial_mechanics.fault_traces.traces)
                    if trace.trace_id == trace_id)
    trace = path.initial_mechanics.fault_traces.traces[position]
    if _dyadic(trace.law.viscosity_pa_s_m):
        raise ValueError("Fractional backward-Euler viscosity has no continuous-work authority")
    knb, ktb = map(_dyadic, (trace.columns.normal_stiffness_pa_m,
                            trace.columns.shear_stiffness_pa_m))
    knc, ktc = map(_dyadic, (trace.law.normal_stiffness_pa_m,
                            trace.law.tangential_stiffness_pa_m))
    kn = _dyadic(1. / (1. / trace.columns.normal_stiffness_pa_m
                       + 1. / trace.law.normal_stiffness_pa_m))
    kt = _dyadic(1. / (1. / trace.columns.shear_stiffness_pa_m
                       + 1. / trace.law.tangential_stiffness_pa_m))
    mu = _dyadic(trace.law.friction_dry
        + (trace.law.friction_wet - trace.law.friction_dry) * trace.water)
    cohesion = _dyadic(trace.law.cohesion_pa
        * (1. - (1. - trace.law.wet_cohesion_fraction) * trace.water))
    _, gap0, jump0 = path.previous_anchor_fields[position]
    _, gap1, jump1 = path.step.anchor_fields[position]
    g0, j0, g1, j1 = map(_field, (gap0, jump0, gap1, jump1))
    dg, dj = _add(g1, _scale(g0, -1)), _add(j1, _scale(j0, -1))
    # tau = -C + mu*kn*g, and p = j - tau/kt on this branch.
    dtau = _scale(dg, mu * kn)
    dp = _add(dj, _scale(dtau, -1 / kt))
    an_bulk, an_contact = knb * (kn / knc - 1)**2, kn * kn / knc
    at_bulk, at_contact = ktb * (1 / ktc - 1 / kt)**2, 1 / ktc
    support = json.loads(source.whole_source_constitutive_support_payload_json)
    return dict(kn=kn, kt=kt, mu=mu, cohesion=cohesion,
        g0=g0, dg=dg, dj=dj, dtau=dtau, dp=dp,
        an_bulk=an_bulk, an_contact=an_contact,
        at_bulk=at_bulk, at_contact=at_contact,
        an=an_bulk + an_contact, at=at_bulk + at_contact,
        support_pressure=_support_interval(support["pressure_bounds_pa"]),
        support_dp=_support_interval(support["plastic_rate_bounds_m_s"])
            * _dyadic(source.history.literal_dt_s),
        duration=_dyadic(source.history.literal_dt_s))


def _piece_values(source, row, family, time, precision):
    angle = row.computational_angle_bounds
    bound = lambda field: _outward(_field_bounds(field, angle.lo, angle.hi, precision), precision)
    gap = _outward(_field_affine_bounds(family["g0"], family["dg"],
        angle.lo, angle.hi, time.lo, time.hi, precision), precision)
    pressure = _intersection(-family["kn"] * gap, family["support_pressure"], "pressure")
    shear = _outward(-family["cohesion"] - family["mu"] * pressure, precision)
    dg, dj, dtau = map(bound, (family["dg"], family["dj"], family["dtau"]))
    dp = _intersection(bound(family["dp"]), family["support_dp"], "plastic rate")
    normal_product, shear_product = _outward(gap * dg, precision), _outward(shear * dtau, precision)
    bulk = _outward(family["an_bulk"] * normal_product
                     + family["at_bulk"] * shear_product, precision)
    contact = _outward(family["an_contact"] * normal_product
                        + family["at_contact"] * shear_product, precision)
    work = _outward(family["kn"] * normal_product + shear * dj, precision)
    dissipation = _outward(shear * dp, precision)
    residual = _outward((family["kn"] - family["an"]) * normal_product
                         + (1 / family["kt"] - family["at"]) * shear_product, precision)
    # This sign follows the complete original source support.  It is not a
    # tolerance, a floor applied to a guessed branch or a selected endpoint.
    if dissipation.lo < 0:
        raise ValueError("Authenticated down-dip work has contradictory dissipation signs")
    return dict(dg=dg, dj=dj, dp=dp, work=work, dissipation=dissipation,
                bulk=bulk, contact=contact, total=_outward(bulk + contact, precision),
                residual=residual)


def query_native_contact_field_work(history, trace_id, computational_angle_bounds,
        time_fraction_bounds, *, expected_source_history_dto,
        source_expression_context=None, precision=256):
    """Enclose reaction power on every source piece and the complete clock box.

    ``time_fraction_bounds`` bounds expressions at all its clocks.  It does
    not prescribe an accumulated-work duration.  Rates with respect to the
    dimensionless original fraction are divided by the literal source dt to
    report physical watts per square metre as well.
    """
    source = query_native_contact_field_history(history, trace_id,
        computational_angle_bounds, time_fraction_bounds,
        expected_source_history_dto=expected_source_history_dto,
        source_expression_context=source_expression_context, precision=precision)
    family, rows = _family(source), []
    for row in source.previous_memory_intersections:
        values = _piece_values(source, row, family, source.time_fraction_bounds, source.precision)
        rates = {"source_work": values["work"], "friction_dissipation": values["dissipation"],
                 "bulk_energy": values["bulk"], "contact_energy": values["contact"],
                 "total_energy": values["total"], "compliance_balance_residual": values["residual"]}
        fields = {name + "_rate_bounds_j_m2_per_fraction": value for name, value in rates.items()}
        fields.update({name + "_rate_bounds_w_m2": _outward(value / family["duration"], source.precision)
                       for name, value in rates.items()})
        rows.append(NativeContactFieldWorkPieceQuery(row.piece_index,
            row.computational_angle_bounds, values["dg"], values["dj"], values["dp"], -values["dp"],
            **fields))
    bounds = {name: _hull([getattr(row, name) for row in rows]) for name in _RATE_FIELDS}
    return NativeContactFieldWorkQuery(source, tuple(rows), **bounds)


def integrate_native_contact_field_work(history, trace_id, computational_angle_bounds,
        initial_time_fraction, final_time_fraction, *, expected_source_history_dto,
        source_expression_context=None, precision=256):
    """Analytically integrate time between two explicit original clock limits.

    The arithmetic endpoint mean is an identity for the integral of an
    affine function.  It is never selected as a physical event or sampled
    contact state.  Angle intervals remain expression enclosures, not finite
    interface areas; angular integration belongs to an explicit measure.
    """
    first = _fraction_clock(initial_time_fraction, "Initial work clock")
    last = _fraction_clock(final_time_fraction, "Final work clock")
    if first > last:
        raise ValueError("Continuous work integration cannot reverse its original clock")
    source = query_native_contact_field_history(history, trace_id,
        computational_angle_bounds, I(first, last),
        expected_source_history_dto=expected_source_history_dto,
        source_expression_context=source_expression_context, precision=precision)
    family, rows = _family(source), []
    average_clock = I((first + last) / 2)
    duration = last - first
    for row in source.previous_memory_intersections:
        values = _piece_values(source, row, family, average_clock, source.precision)
        fields = dict(source_work_density_bounds_j_m2=values["work"] * duration,
            friction_dissipation_density_bounds_j_m2=values["dissipation"] * duration,
            bulk_energy_increment_density_bounds_j_m2=values["bulk"] * duration,
            contact_energy_increment_density_bounds_j_m2=values["contact"] * duration,
            total_energy_increment_density_bounds_j_m2=values["total"] * duration,
            compliance_balance_residual_density_bounds_j_m2=values["residual"] * duration)
        rows.append(NativeContactFieldWorkPieceIntegral(row.piece_index,
            row.computational_angle_bounds, **{name: _outward(value, source.precision)
                                              for name, value in fields.items()}))
    bounds = {name: _hull([getattr(row, name) for row in rows]) for name in _INTEGRAL_FIELDS}
    return NativeContactFieldWorkIntegral(source, first, last, tuple(rows), **bounds)


def query_native_contact_field_work_primitive(history, trace_id, computational_angle_bounds,
        initial_time_fraction, final_time_fraction_bounds, *, expected_source_history_dto,
        source_expression_context=None, precision=256):
    """Enclose a primitive at every possible final clock, including a root.

    The final interval bounds the primitive expression.  Its width is not an
    adopted physical duration.  The implicit expression identity is retained
    in the source context; only the material-domain caller authenticates it.
    """
    first = _fraction_clock(initial_time_fraction, "Initial primitive clock")
    last = final_time_fraction_bounds
    if (type(last) is not I or type(last.lo) is not Fraction or type(last.hi) is not Fraction
            or not first <= last.lo <= last.hi <= 1):
        raise ValueError("Final primitive clock requires its exact original interval")
    source = query_native_contact_field_history(history, trace_id, computational_angle_bounds,
        I(first, last.hi), expected_source_history_dto=expected_source_history_dto,
        source_expression_context=source_expression_context, precision=precision)
    family, rows = _family(source), []
    average_clock, duration = (I(first) + last) / 2, last - first
    for row in source.previous_memory_intersections:
        values = _piece_values(source, row, family, average_clock, source.precision)
        fields = dict(source_work_density_bounds_j_m2=values["work"] * duration,
            friction_dissipation_density_bounds_j_m2=values["dissipation"] * duration,
            bulk_energy_increment_density_bounds_j_m2=values["bulk"] * duration,
            contact_energy_increment_density_bounds_j_m2=values["contact"] * duration,
            total_energy_increment_density_bounds_j_m2=values["total"] * duration,
            compliance_balance_residual_density_bounds_j_m2=values["residual"] * duration)
        rows.append(NativeContactFieldWorkPieceIntegral(row.piece_index,
            row.computational_angle_bounds, **{name: _outward(value, source.precision)
                                              for name, value in fields.items()}))
    bounds = {name: _hull([getattr(row, name) for row in rows]) for name in _INTEGRAL_FIELDS}
    return NativeContactFieldWorkPrimitive(source, first, last, tuple(rows), **bounds)


def _harmonic(field):
    """Exact complex Fourier coefficients as pairs of rational real parts."""
    result = {0: (field[0], Fraction(0))}
    for frequency, cosine, sine in ((1, field[1], field[2]), (2, field[3], field[4])):
        result[frequency] = (cosine / 2, -sine / 2)
        result[-frequency] = (cosine / 2, sine / 2)
    return {frequency: value for frequency, value in result.items() if any(value)}


def _harmonic_add(a, b):
    result = {}
    for frequency in a.keys() | b.keys():
        aa, bb = a.get(frequency, (Fraction(0), Fraction(0))), b.get(frequency, (Fraction(0), Fraction(0)))
        value = (aa[0] + bb[0], aa[1] + bb[1])
        if any(value):
            result[frequency] = value
    return result


def _harmonic_scale(a, scalar):
    return {frequency: (real * scalar, imaginary * scalar)
            for frequency, (real, imaginary) in a.items() if scalar and (real or imaginary)}


def _harmonic_product(a, b):
    result = {}
    for first, (ar, ai) in a.items():
        for last, (br, bi) in b.items():
            frequency = first + last
            old = result.get(frequency, (Fraction(0), Fraction(0)))
            result[frequency] = (old[0] + ar * br - ai * bi, old[1] + ar * bi + ai * br)
    return {frequency: value for frequency, value in result.items() if any(value)}


def _harmonic_integral(field, low, high, precision):
    if low == high:
        return I(0)
    constant = field.get(0, (Fraction(0), Fraction(0)))
    if constant[1]:
        raise ValueError("Physical contact work must retain real harmonic coefficients")
    result = I(constant[0] * (high - low))
    for frequency in sorted(value for value in field if value > 0):
        real, imaginary = field[frequency]
        if field.get(-frequency, (Fraction(0), Fraction(0))) != (real, -imaginary):
            raise ValueError("Physical contact work lost its exact conjugate harmonics")
        sin_low, cos_low = _compressed_sin_cos_point(frequency * low, precision)
        sin_high, cos_high = _compressed_sin_cos_point(frequency * high, precision)
        result = _outward(result + (sin_high - sin_low) * (2 * real / frequency)
            + (cos_low - cos_high) * (-2 * imaginary / frequency), precision)
    return result


def _work_harmonics(family, first, last):
    mean_gap = _harmonic(_add(family["g0"], _scale(family["dg"], (first + last) / 2)))
    mean_shear = _harmonic_scale(mean_gap, family["mu"] * family["kn"])
    mean_shear = _harmonic_add(mean_shear, {0: (-family["cohesion"], Fraction(0))})
    normal = _harmonic_product(mean_gap, _harmonic(family["dg"]))
    shear = _harmonic_product(mean_shear, _harmonic(family["dtau"]))
    combine = lambda normal_coefficient, shear_coefficient: _harmonic_scale(
        _harmonic_add(_harmonic_scale(normal, normal_coefficient),
                      _harmonic_scale(shear, shear_coefficient)), last - first)
    bulk, contact = combine(family["an_bulk"], family["at_bulk"]), combine(family["an_contact"], family["at_contact"])
    work = _harmonic_scale(_harmonic_add(_harmonic_scale(normal, family["kn"]),
        _harmonic_product(mean_shear, _harmonic(family["dj"]))), last - first)
    dissipation = _harmonic_scale(_harmonic_product(mean_shear, _harmonic(family["dp"])), last - first)
    residual = combine(family["kn"] - family["an"], 1 / family["kt"] - family["at"])
    # Check the first law in exact symbolic coefficients, before any bound.
    accounted = _harmonic_add(_harmonic_add(bulk, contact), _harmonic_add(dissipation, residual))
    if work != accounted:
        raise ValueError("Continuous contact work violates its exact harmonic first law")
    return dict(source_work=work, friction_dissipation=dissipation,
        bulk_energy_increment=bulk, contact_energy_increment=contact,
        total_energy_increment=_harmonic_add(bulk, contact), compliance_balance_residual=residual)


def _energy_harmonics(family, time):
    gap = _harmonic(_add(family["g0"], _scale(family["dg"], time)))
    shear = _harmonic_add(_harmonic_scale(gap, family["mu"] * family["kn"]),
                          {0: (-family["cohesion"], Fraction(0))})
    normal_square, shear_square = _harmonic_product(gap, gap), _harmonic_product(shear, shear)
    combine = lambda an, at: _harmonic_scale(_harmonic_add(_harmonic_scale(normal_square, an),
        _harmonic_scale(shear_square, at)), Fraction(1, 2))
    bulk, contact = combine(family["an_bulk"], family["at_bulk"]), combine(family["an_contact"], family["at_contact"])
    return dict(bulk_energy=bulk, contact_energy=contact, total_energy=_harmonic_add(bulk, contact))


def integrate_native_contact_field_work_over_original_interface(history, trace_id,
        initial_computational_angle, final_computational_angle,
        initial_time_fraction, final_time_fraction, *, expected_source_history_dto,
        source_expression_context=None, precision=256):
    """Integrate the dense source over an explicit unchanged interface measure.

    Products of the retained degree-two source fields have degree at most
    four; all angular integrals use their exact harmonic antiderivatives.
    The measure is R_m H/saved_sin(dip) dtheta_computational, authenticated
    from original frame literals.  It is separate from the sixteen saved
    rounded Gauss areas.  This integrates the declared original trace, never
    a swept archive domain or a newly admitted material contact.
    """
    low, high = initial_computational_angle, final_computational_angle
    if type(low) is not Fraction or type(high) is not Fraction or not 0 <= low <= high:
        raise ValueError("Original interface integration requires exact ordered Fraction limits")
    time_integral = integrate_native_contact_field_work(history, trace_id, I(low, high),
        initial_time_fraction, final_time_fraction,
        expected_source_history_dto=expected_source_history_dto,
        source_expression_context=source_expression_context, precision=precision)
    source, family = time_integral.source_field_query, _family(time_integral.source_field_query)
    trace = next(trace for trace in history.initial_mechanics.fault_traces.traces if trace.trace_id == trace_id)
    nodes = tuple(node for node in history.quadrature_histories if node.trace_id == trace_id)
    if not nodes or any(node.original_frame != nodes[0].original_frame for node in nodes):
        raise ValueError("Original interface measure lacks one authenticated saved fault frame")
    sine = _dyadic(nodes[0].original_frame[0])
    radius = _dyadic(history.initial_mechanics.fault_traces.radius_km) * 1000
    height = _dyadic(trace.contact_height_m)
    if min(sine, radius, height) <= 0:
        raise ValueError("Original interface measure must retain positive literal dimensions")
    coefficient = radius * height / sine
    # Nodal total is retained independently for this trace.  It is neither
    # rescaled to the requested angle interval nor used to infer a density.
    nodal_area = sum((_dyadic(node.original_area_m2) for node in nodes), Fraction(0))
    first, last = time_integral.initial_time_fraction, time_integral.final_time_fraction
    work = _work_harmonics(family, first, last)
    fields = {name + "_bounds_j": _outward(_harmonic_integral(polynomial, low, high, source.precision)
        * coefficient, source.precision) for name, polynomial in work.items()}
    for prefix, time in (("initial", first), ("final", last)):
        fields.update({prefix + "_" + name + "_bounds_j": _outward(
            _harmonic_integral(polynomial, low, high, source.precision) * coefficient,
            source.precision) for name, polynomial in _energy_harmonics(family, time).items()})
    continuum_area = (high - low) * coefficient
    return NativeContactFieldWorkInterfaceIntegral(time_integral, coefficient,
        continuum_area, nodal_area, continuum_area - nodal_area, **fields)


def validate_native_contact_field_work(query, *, history, expected_source_history_dto):
    if type(query) is not NativeContactFieldWorkQuery:
        raise ValueError("Expected an immutable continuous work rate query")
    source = query.source_field_query
    repeated = query_native_contact_field_work(history, source.trace_id,
        source.computational_angle_bounds, source.time_fraction_bounds,
        expected_source_history_dto=expected_source_history_dto,
        source_expression_context=source.source_expression_context, precision=source.precision)
    if _typed_fingerprint(query) != _typed_fingerprint(repeated):
        raise ValueError("Continuous work query differs from the complete original source")
    return query


def validate_native_contact_field_work_integral(integral, *, history, expected_source_history_dto):
    if type(integral) is not NativeContactFieldWorkIntegral:
        raise ValueError("Expected an immutable continuous work time integral")
    source = integral.source_field_query
    repeated = integrate_native_contact_field_work(history, source.trace_id,
        source.computational_angle_bounds, integral.initial_time_fraction, integral.final_time_fraction,
        expected_source_history_dto=expected_source_history_dto,
        source_expression_context=source.source_expression_context, precision=source.precision)
    if _typed_fingerprint(integral) != _typed_fingerprint(repeated):
        raise ValueError("Continuous work integral differs from the complete original source")
    return integral


def validate_native_contact_field_work_primitive(primitive, *, history, expected_source_history_dto):
    if type(primitive) is not NativeContactFieldWorkPrimitive:
        raise ValueError("Expected an immutable continuous work primitive query")
    source = primitive.source_field_query
    repeated = query_native_contact_field_work_primitive(history, source.trace_id,
        source.computational_angle_bounds, primitive.initial_time_fraction,
        primitive.final_time_fraction_bounds, expected_source_history_dto=expected_source_history_dto,
        source_expression_context=source.source_expression_context, precision=source.precision)
    if _typed_fingerprint(primitive) != _typed_fingerprint(repeated):
        raise ValueError("Continuous work primitive differs from the complete original source")
    return primitive


def validate_native_contact_field_work_interface_integral(integral, *, history, expected_source_history_dto):
    if type(integral) is not NativeContactFieldWorkInterfaceIntegral:
        raise ValueError("Expected an immutable original-interface work integral")
    time_integral = integral.source_time_integral
    source, angle = time_integral.source_field_query, time_integral.computational_angle_bounds
    repeated = integrate_native_contact_field_work_over_original_interface(history, source.trace_id,
        angle.lo, angle.hi, time_integral.initial_time_fraction, time_integral.final_time_fraction,
        expected_source_history_dto=expected_source_history_dto,
        source_expression_context=source.source_expression_context, precision=source.precision)
    if _typed_fingerprint(integral) != _typed_fingerprint(repeated):
        raise ValueError("Original-interface work integral differs from its complete source and measure")
    return integral


def _dto(value, *, history, expected_source_history_dto, integral):
    validator = validate_native_contact_field_work_integral if integral else validate_native_contact_field_work
    validator(value, history=history, expected_source_history_dto=expected_source_history_dto)
    body = {name: _portable(getattr(value, name)) for name in value.__dataclass_fields__
            if name != "source_field_query"}
    body.update(format=INTEGRAL_FORMAT if integral else FORMAT, model=MODEL,
        source_field_query=native_contact_field_history_to_dict(value.source_field_query,
            history=history, expected_source_history_dto=expected_source_history_dto),
        work_measure=WORK_MEASURE, rate_measure=RATE_MEASURE, first_law=FIRST_LAW,
        angular_integration_performed=False, finite_point_area_m2=None,
        original_nodal_work_density_j_m2=None, endpoint_backward_euler_ledgers_replaced=False,
        binary64_compliance_coefficients_preserved=True, signed_compliance_residual_suppressed=False,
        native_candidate_mechanics_adopted=False, geometry_expression_authenticated_by_this_component=False,
        material_transfer_work_adopted=False, material_transfer_dissipation_adopted=False)
    body["work_sha256"] = hashlib.sha256(_encoded(body)).hexdigest()
    return json.loads(_encoded(body))


def native_contact_field_work_to_dict(query, *, history, expected_source_history_dto):
    return _dto(query, history=history, expected_source_history_dto=expected_source_history_dto, integral=False)


def native_contact_field_work_integral_to_dict(integral, *, history, expected_source_history_dto):
    return _dto(integral, history=history, expected_source_history_dto=expected_source_history_dto, integral=True)


def native_contact_field_work_primitive_to_dict(primitive, *, history, expected_source_history_dto):
    validate_native_contact_field_work_primitive(primitive, history=history,
                                               expected_source_history_dto=expected_source_history_dto)
    body = {name: _portable(getattr(primitive, name)) for name in primitive.__dataclass_fields__
            if name != "source_field_query"}
    body.update(format=PRIMITIVE_FORMAT, model=MODEL,
        source_field_query=native_contact_field_history_to_dict(primitive.source_field_query,
            history=history, expected_source_history_dto=expected_source_history_dto),
        work_measure=WORK_MEASURE, first_law=FIRST_LAW,
        final_clock_interval_is_primitive_expression_enclosure=True,
        root_midpoint_selected=False, finite_point_area_m2=None,
        angular_integration_performed=False, endpoint_backward_euler_ledgers_replaced=False,
        binary64_compliance_coefficients_preserved=True, signed_compliance_residual_suppressed=False,
        native_candidate_mechanics_adopted=False, geometry_expression_authenticated_by_this_component=False,
        material_transfer_work_adopted=False, material_transfer_dissipation_adopted=False)
    body["work_sha256"] = hashlib.sha256(_encoded(body)).hexdigest()
    return json.loads(_encoded(body))


def native_contact_field_work_interface_integral_to_dict(integral, *, history, expected_source_history_dto):
    validate_native_contact_field_work_interface_integral(integral, history=history,
                                                        expected_source_history_dto=expected_source_history_dto)
    body = {name: _portable(getattr(integral, name)) for name in integral.__dataclass_fields__
            if name != "source_time_integral"}
    body.update(format=INTERFACE_FORMAT, model=MODEL,
        source_time_integral=native_contact_field_work_integral_to_dict(integral.source_time_integral,
            history=history, expected_source_history_dto=expected_source_history_dto),
        first_law=FIRST_LAW, angular_integration_model="exact_degree_four_harmonic_antiderivative",
        measure_model="original_saved_frame_radius_height_over_sine_times_computational_angle",
        continuum_measure_is_original_nodal_gauss_measure=False,
        continuum_work_is_original_endpoint_backward_euler_work=False,
        original_nodal_ledgers_replaced=False, signed_compliance_residual_suppressed=False,
        native_candidate_mechanics_adopted=False,
        swept_archive_measure_integrated=False, incoming_material_contact_area_adopted=False,
        material_transfer_work_adopted=False, material_transfer_dissipation_adopted=False)
    body["work_sha256"] = hashlib.sha256(_encoded(body)).hexdigest()
    return json.loads(_encoded(body))


def native_contact_field_work_from_dict(raw, *, history, trace_id, computational_angle_bounds,
        time_fraction_bounds, expected_source_history_dto, source_expression_context=None, precision=256):
    _literal_json(raw)
    query = query_native_contact_field_work(history, trace_id, computational_angle_bounds,
        time_fraction_bounds, expected_source_history_dto=expected_source_history_dto,
        source_expression_context=source_expression_context, precision=precision)
    if type(raw) is not dict or _encoded(raw) != _encoded(native_contact_field_work_to_dict(
            query, history=history, expected_source_history_dto=expected_source_history_dto)):
        raise ValueError("Continuous work DTO differs from the independent full source and arguments")
    return query


def native_contact_field_work_integral_from_dict(raw, *, history, trace_id, computational_angle_bounds,
        initial_time_fraction, final_time_fraction, expected_source_history_dto,
        source_expression_context=None, precision=256):
    _literal_json(raw)
    integral = integrate_native_contact_field_work(history, trace_id, computational_angle_bounds,
        initial_time_fraction, final_time_fraction, expected_source_history_dto=expected_source_history_dto,
        source_expression_context=source_expression_context, precision=precision)
    if type(raw) is not dict or _encoded(raw) != _encoded(native_contact_field_work_integral_to_dict(
            integral, history=history, expected_source_history_dto=expected_source_history_dto)):
        raise ValueError("Continuous work integral DTO differs from the independent full source and clocks")
    return integral


def native_contact_field_work_primitive_from_dict(raw, *, history, trace_id, computational_angle_bounds,
        initial_time_fraction, final_time_fraction_bounds, expected_source_history_dto,
        source_expression_context=None, precision=256):
    _literal_json(raw)
    primitive = query_native_contact_field_work_primitive(history, trace_id, computational_angle_bounds,
        initial_time_fraction, final_time_fraction_bounds,
        expected_source_history_dto=expected_source_history_dto,
        source_expression_context=source_expression_context, precision=precision)
    if type(raw) is not dict or _encoded(raw) != _encoded(native_contact_field_work_primitive_to_dict(
            primitive, history=history, expected_source_history_dto=expected_source_history_dto)):
        raise ValueError("Continuous work primitive DTO differs from its independent source and root interval")
    return primitive


def native_contact_field_work_interface_integral_from_dict(raw, *, history, trace_id,
        initial_computational_angle, final_computational_angle,
        initial_time_fraction, final_time_fraction, expected_source_history_dto,
        source_expression_context=None, precision=256):
    _literal_json(raw)
    integral = integrate_native_contact_field_work_over_original_interface(history, trace_id,
        initial_computational_angle, final_computational_angle,
        initial_time_fraction, final_time_fraction,
        expected_source_history_dto=expected_source_history_dto,
        source_expression_context=source_expression_context, precision=precision)
    if type(raw) is not dict or _encoded(raw) != _encoded(native_contact_field_work_interface_integral_to_dict(
            integral, history=history, expected_source_history_dto=expected_source_history_dto)):
        raise ValueError("Original-interface work DTO differs from the independent complete source and measure")
    return integral
