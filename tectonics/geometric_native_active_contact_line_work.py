"""Enclosed active-pair work on the original, one-dimensional interface.

The measure is the retained ``C dtheta_computational``.  A contact segment's
time integral is its normal-potential difference; no Gauss atom, swept
material area, source-affine work ledger, root midpoint or centroid is used.
The unchanged carrier cores have exact mixed-frequency antiderivatives.
Every positive-width seam strip is retained with a rigorous interval budget.
The latter budgets enclose all selector paths and inherit the evolved state:
there is no work or energy increment at a change of reference.

This operator does not accept a force trajectory.  It also refuses to call a
line integral closed when its externally authenticated selector leaves a set
of positive contact measure unresolved.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction as F
from functools import lru_cache
import hashlib
import json

from .geometric_native_pose import FractionInterval as I, interval_pi
from .geometric_native_contact_field_history import (
    _compressed_trig_ranges, _outward, _portable, _square,
)
from .geometric_native_contact_time_support import _field
from .geometric_native_contact_reference_transport import _parameters, _proof_key
from .geometric_native_physical_history import (
    _dyadic, _literal_json, native_physical_history_binding_to_dict,
    validate_native_physical_history_binding,
)
from .geometric_native_active_contact_work import _family, _joint_relative_change


FORMAT = "native-active-original-contact-line-work-1"
MODEL = "full_SO3_frozen_reference_mixed_frequency_core_and_positive_measure_strip_enclosures"
MEASURE = "C_dtheta_computational_du_on_original_contact_interface"
FIRST_LAW = "actual_work = bulk_energy_increment + contact_energy_increment + zero_dissipation + signed_literal_compliance_residual"
_BUDGET_FIELDS = (
    "source_work_bounds_j", "friction_dissipation_bounds_j",
    "bulk_energy_increment_bounds_j", "contact_energy_increment_bounds_j",
    "total_energy_increment_bounds_j", "compliance_balance_residual_bounds_j",
)


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def _source(history, expected):
    validate_native_physical_history_binding(history)
    if type(expected) is not dict:
        raise ValueError("Line work requires the complete independent original source DTO")
    _literal_json(expected)
    result = _encoded(expected)
    if result != _encoded(native_physical_history_binding_to_dict(history)):
        raise ValueError("Line work source differs from the full original physical replay")
    return result


def _dot(first, second):
    return sum((a*b for a, b in zip(first, second)), I(0))


def _cross(first, second):
    return tuple(first[j]*second[k]-first[k]*second[j]
                 for j, k in ((1, 2), (2, 0), (0, 1)))


def _trace(history, identity):
    matches = tuple(row for row in history.initial_mechanics.fault_traces.traces
                    if row.trace_id == identity)
    if len(matches) != 1:
        raise ValueError("Line work needs an identified original physical trace")
    return matches[0]


def _tips(history, identity, index):
    first = next(row[index] for row in history.accepted_before.transverse_displacement_m
                 if row[0] == identity)
    last = next(row[index] for row in history.actual_candidate.transverse_displacement_m
                if row[0] == identity)
    return _dyadic(first), _dyadic(last)


def _geometry_literals(history, trace_id):
    nodes = tuple(row for row in history.quadrature_histories if row.trace_id == trace_id)
    if not nodes or any(row.original_frame != nodes[0].original_frame
            or row.original_point.horizontal != nodes[0].original_point.horizontal for row in nodes):
        raise ValueError("Original line has no consistent retained fault frame")
    sine, cosine = map(_dyadic, nodes[0].original_frame)
    horizontal = tuple(map(_dyadic, nodes[0].original_point.horizontal))
    trace = _trace(history, trace_id)
    radius = _dyadic(history.origin_surface.radius_km)*1000
    coefficient = radius*_dyadic(trace.contact_height_m)/sine
    if min(radius, sine, cosine, coefficient) <= 0 or horizontal != (F(-1), F(0), F(0)):
        raise ValueError("Line work requires the actual positive inclined X-port frame")
    return radius, sine, cosine, horizontal, coefficient


def _hadd(first, second, precision):
    result = {}
    for key in first.keys() | second.keys():
        a, b = first.get(key, (I(0), I(0))), second.get(key, (I(0), I(0)))
        result[key] = tuple(_outward(x+y, precision) for x, y in zip(a, b))
    return result


def _hscale(field, scalar, precision):
    return {key: tuple(_outward(x*scalar, precision) for x in value)
            for key, value in field.items()}


def _hproduct(first, second, precision):
    result = {}
    for (am, an), (ar, ai) in first.items():
        for (bm, bn), (br, bi) in second.items():
            key = am+bm, an+bn
            old = result.get(key, (I(0), I(0)))
            result[key] = (_outward(old[0]+ar*br-ai*bi, precision),
                           _outward(old[1]+ar*bi+ai*br, precision))
    return result


def _harmonic(field, *, native=False):
    field = _as_intervals(field)
    result = {(0, 0): (field[0], I(0))}
    for frequency, cosine, sine in ((1, field[1], field[2]), (2, field[3], field[4])):
        if any((cosine.lo, cosine.hi, sine.lo, sine.hi)):
            key = (0, frequency) if native else (frequency, 0)
            opposite = tuple(-item for item in key)
            result[key] = (cosine/2, -sine/2)
            result[opposite] = (result[key][0], -result[key][1])
    return result


def _as_intervals(field):
    return tuple(x if type(x) is I else I(x) for x in field)


def _trig_ranges_signed(low, high, precision):
    # The original range helper enumerates nonnegative critical points.
    # Reflect negative-frequency endpoints before using it, preserving
    # possible exact extrema at -pi/2 and -pi.
    if high <= 0:
        sine, cosine = _compressed_trig_ranges(-high, -low, precision)
        return -sine, cosine
    if low >= 0:
        return _compressed_trig_ranges(low, high, precision)
    left = _trig_ranges_signed(low, F(0), precision)
    right = _compressed_trig_ranges(F(0), high, precision)
    return tuple(I(min(a.lo, b.lo), max(a.hi, b.hi)) for a, b in zip(left, right))


@lru_cache(maxsize=1024)
def _integrated_trig(frequency, low, high, precision):
    """Certified integrals, including the nearly cancelling mixed frequency."""
    if frequency == I(0):
        return I(high-low), I(0)
    magnitude = max(abs(frequency.lo), abs(frequency.hi))*max(abs(low), abs(high))
    if magnitude < F(1, 1000):
        # A denominator Gamma different from exact pi/2 produces genuine
        # frequencies near zero.  Taylor kernels avoid subtracting close
        # endpoint sines or dividing by an enclosure containing zero.
        cosine, sine = I(high-low), frequency*((high**2-low**2)/2)
        squared = _square(frequency)
        power_c, power_s = I(1), frequency
        fact_c, fact_s = 1, 1
        sign = 1
        for degree in range(1, 32):
            sign = -sign
            power_c = _outward(power_c*squared, precision)
            power_s = _outward(power_s*squared, precision)
            fact_c *= (2*degree-1)*(2*degree)
            fact_s *= (2*degree)*(2*degree+1)
            cosine = _outward(cosine+power_c*(sign*(high**(2*degree+1)-low**(2*degree+1))
                / F(fact_c*(2*degree+1))), precision)
            sine = _outward(sine+power_s*(sign*(high**(2*degree+2)-low**(2*degree+2))
                / F(fact_s*(2*degree+2))), precision)
        # Absolute entire-series remainder, integrated over the domain.
        tail_c = (high-low)*magnitude**64/F(fact_c*63*64)/(1-magnitude)
        tail_s = (high-low)*magnitude**65/F(fact_s*64*65)/(1-magnitude)
        return (_outward(cosine+I(-tail_c, tail_c), precision),
                _outward(sine+I(-tail_s, tail_s), precision))
    if frequency.lo <= 0 <= frequency.hi:
        raise ValueError("Mixed-frequency integral has an unresolved zero denominator")
    at_low, at_high = frequency*low, frequency*high
    sl, cl = _trig_ranges_signed(at_low.lo, at_low.hi, precision)
    sh, ch = _trig_ranges_signed(at_high.lo, at_high.hi, precision)
    return (_outward((sh-sl)/frequency, precision),
            _outward((cl-ch)/frequency, precision))


def _hintegral(field, low, high, native_scale, precision):
    result = I(0)
    for (computational, native), (real, imaginary) in sorted(field.items()):
        frequency = computational+native*native_scale
        cosine, sine = _integrated_trig(frequency, low, high, precision)
        result = _outward(result+real*cosine-imaginary*sine, precision)
    return result


def _core_fields(history, receiver, donor, precision, order):
    trace = _trace(history, receiver)
    radius, sine, cosine, horizontal, coefficient = _geometry_literals(history, receiver)
    axes = {owner: tuple(item*history.dt_myr for item in omega)
            for owner, omega in history.omega_records}
    donor_trace = _trace(history, donor)
    ad, ar = axes[donor_trace.subducting_plate], axes[trace.overriding_plate]
    start, end = (tuple(map(_dyadic, endpoint)) for endpoint in (trace.start, trace.end))
    ds, de = (_joint_relative_change(tuple(I(x) for x in vector), ad, ar, I(1), precision, order)
              for vector in (start, end))
    aa, bb = _dot(start, ds), _dot(end, de)
    ab = _dot(start, de)+_dot(end, ds)
    zh = (I(0), _dot(horizontal, ds), _dot(horizontal, de), I(0), I(0))
    zc = ((aa+bb)/2, I(0), I(0), (aa-bb)/2, ab/2)
    ws0, ws1 = _tips(history, donor, 1)
    wo0, wo1 = _tips(history, receiver, 2)
    geometry = tuple(_outward(-((radius+ws1)*(sine*h+cosine*c)), precision)
                     for h, c in zip(zh, zc))
    geometry = (geometry[0]+cosine*((wo1-wo0)-(ws1-ws0)),)+geometry[1:]
    fields = next(row for row in history.contact_time_path.previous_anchor_fields if row[0] == receiver)
    gap0 = _harmonic(_as_intervals(_field(fields[1])))
    gap1 = _hadd(gap0, _harmonic(geometry, native=True), precision)
    square0, square1 = _hproduct(gap0, gap0, precision), _hproduct(gap1, gap1, precision)
    potential = _hscale(_hadd(square1, _hscale(square0, -1, precision), precision), F(1, 2), precision)
    return gap0, gap1, square0, square1, potential, coefficient


@dataclass(frozen=True, slots=True)
class NativeActiveContactLineBranchProof:
    receiver_trace_id: str
    initial_gap_bounds_m: I
    whole_line_gap_rate_bounds_m_per_fraction: I
    whole_line_jump_rate_bounds_m_per_fraction: I
    whole_line_gap_bounds_m: I
    final_gap_bounds_m: I
    relative_rotation_rate_remainder_bounds_per_fraction: I
    original_shape_bounds: I
    carried_state_at_switch: bool
    compression_proved: bool
    monotone_down_dip_proved: bool


def _branch_proof(history, quarter, precision):
    """Bound every admissible frozen segment, without knowing its root clock.

    For unit c, v=Qr(u)^T Qd(u)q and v(s)=c, |v-c|_2<=D(u-s).
    Its derivative is (Qr(u)^T(ad-ar)) cross v.  Consequently its difference
    from (ad-ar) cross c is bounded by 2*A*D + D^2 componentwise; the
    slightly larger rational expression below also bounds the L1 rotation
    series.  This preserves all axes and is uniform over every switch time.
    """
    receiver = quarter.receiver_trace_id
    trace = _trace(history, receiver)
    radius, sine, cosine, horizontal, _ = _geometry_literals(history, receiver)
    possible = tuple(_trace(history, identity) for identity in quarter.possible_donor_trace_ids)
    axes = {owner: tuple(item*history.dt_myr for item in omega)
            for owner, omega in history.omega_records}
    ar = axes[trace.overriding_plate]
    if len({row.subducting_plate for row in possible}) != 1:
        raise ValueError("Whole-line conservative bound requires the same actual donor owner")
    ad = axes[possible[0].subducting_plate]
    difference = tuple(d-r for d, r in zip(ad, ar))
    a, d = sum(map(abs, ar), F(0)), sum(map(abs, difference), F(0))
    if a+d >= F(1, 4):
        raise ValueError("Whole-line rotation exceeds the explicit small-increment proof budget")
    error = 4*(a+d)*d/(1-a)
    start, end = (tuple(map(_dyadic, endpoint)) for endpoint in (trace.start, trace.end))
    # Every native quarter has the signs of its two unit-axis endpoints.
    cbox = tuple(I(min(F(0), x, y), max(F(0), x, y)) for x, y in zip(start, end))
    leading_horizontal = _dot(horizontal, _cross(difference, cbox))
    dh = _outward(leading_horizontal+I(-error, error), precision)
    # c dot ((ad-ar) cross c)=0 exactly.  Do not box that cancelling term.
    dc = I(-2*error, 2*error)
    ws = [_tips(history, row.trace_id, 1) for row in possible]
    wo0, wo1 = _tips(history, receiver, 2)
    if any(first != 0 or last > 0 for first, last in ws) or wo0 != 0 or wo1 <= 0:
        raise ValueError("Line branch proof needs the actual zero-tip preparation and signed donor/receiver rates")
    donor_rate = I(min(last for _, last in ws), F(0))
    radial = I(radius+min(last for _, last in ws), radius)
    zh, zc = I(-d, d), I(-d*d/2, F(0))
    # phi in [0,1], ws<=0.  Both bounds include every own donor Ritz strip.
    gap_rate = _outward(cosine*(wo1-donor_rate)
        -donor_rate*(sine*zh+cosine*zc)-radial*(sine*dh+cosine*dc), precision)
    jump_rate = _outward(-sine*(wo1-donor_rate)
        -donor_rate*(cosine*zh-sine*zc)-radial*(cosine*dh-sine*dc), precision)
    fields = next(row for row in history.contact_time_path.previous_anchor_fields if row[0] == receiver)
    g0, j0 = _field(fields[1]), _field(fields[2])
    if any(g0[1:]) or any(j0):
        raise ValueError("Global branch proof needs the unchanged actual constant-gap zero-jump preparation")
    memory = next(row for row in history.previous_memory.traces if row.trace_id == receiver)
    if any(any(_field(piece.plastic_slip)) or any(_field(piece.cumulative_slip))
            for piece in memory.pieces):
        raise ValueError("Whole-line source must retain its actual zero plastic and cumulative preparation")
    # A newly frozen label q=Qd(s)^T Qr(s)c has |qx|<=D, while the positive
    # owner gate supplies qx>=0.  asin(D)<=2D for this small-increment domain.
    # This certifies its own Ritz strip, instead of borrowing a receiver tip.
    widths = [_dyadic(next(row.sub.width_m for row in history.initial_mechanics.bending_parameters
                          if row.trace_id == donor.trace_id)) for donor in possible]
    z = 2*radius*d/min(widths)
    if min(widths) <= 0 or z >= 1:
        raise ValueError("A whole-line frozen donor label leaves its own original Ritz strip")
    shape = I((1-z)**2*(z+2)/2, F(1))
    initial = I(g0[0])
    if gap_rate.lo <= 0 or jump_rate.hi >= 0:
        raise ValueError("Whole-line segments do not prove opening-toward-zero and monotone down-dip rates")
    final = _outward(initial+gap_rate, precision)
    whole = I(initial.lo, final.hi)
    if whole.hi >= 0:
        raise ValueError("Whole-line carried state does not prove strict compression")
    return NativeActiveContactLineBranchProof(receiver, initial, gap_rate, jump_rate, whole,
        final, I(-error, error), shape, True, True, True)


@dataclass(frozen=True, slots=True)
class NativeActiveContactLinePairBudget:
    receiver_fragment_id: str
    receiver_trace_id: str
    donor_fragment_id: str
    donor_trace_id: str
    core_original_interface_area_m2: F
    enclosed_strip_original_interface_area_m2: F
    core_potential_increment_integral_bounds_m4: I
    strip_potential_increment_integral_bounds_m4: I
    potential_increment_integral_bounds_m4: I
    literal_normal_coefficients_pa_m: tuple
    source_work_bounds_j: I
    friction_dissipation_bounds_j: I
    bulk_energy_increment_bounds_j: I
    contact_energy_increment_bounds_j: I
    total_energy_increment_bounds_j: I
    compliance_balance_residual_bounds_j: I


@dataclass(frozen=True, slots=True)
class NativeActiveContactLineQuarterBudget:
    receiver_fragment_id: str
    receiver_trace_id: str
    original_interface_area_density_m2_per_computational_radian: F
    whole_original_interface_area_m2: F
    core_original_interface_area_m2: F
    strip_original_interface_area_m2: F
    native_angle_per_computational_radian_bounds: I
    branch_proof: NativeActiveContactLineBranchProof
    initial_normal_potential_integral_bounds_m4: I
    final_normal_potential_integral_bounds_m4: I
    independent_potential_increment_integral_bounds_m4: I


@dataclass(frozen=True, slots=True)
class NativeActiveContactLineWorkIntegral:
    history: object
    line_certificate: object
    precision: int
    order: int
    quarters: tuple
    receiving_pair_budgets: tuple
    original_interface_area_m2: F
    explicitly_enclosed_strip_interface_area_m2: F
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
    independently_enclosed_source_work_bounds_j: I
    first_law_residual_bounds_j: I
    source_history_payload_json: bytes
    integral_sha256: str

    @property
    def receiving_pair_weighted_contact_integrals_numerically_enclosed(self):
        return True

    @property
    def all_receiver_angles_scheduled(self):
        return True

    @property
    def candidate_mechanics_adopted(self):
        return False

    @property
    def accepted_native_force_intervals(self):
        return 0

    @property
    def physical_pair_admission_authorized(self):
        return False


def _intersection(first, second, message):
    low, high = max(first.lo, second.lo), min(first.hi, second.hi)
    if low > high:
        raise ValueError(message)
    return I(low, high)


def integrate_native_active_contact_line_work(history, *, expected_source_history_dto,
        line_certificate, precision=128, order=24):
    """Enclose all 16 original receiving pairs on their original line measure."""
    from .geometric_native_active_contact_line import validate_native_active_contact_line

    payload = _source(history, expected_source_history_dto)
    if type(precision) is not int or not 96 <= precision <= 512 or type(order) is not int or not 16 <= order <= 64:
        raise ValueError("Line work needs bounded explicit literal proof precision and order")
    validate_native_active_contact_line(line_certificate)
    if _proof_key(line_certificate.history) != _proof_key(history):
        raise ValueError("Line selector and work have different original sources")
    if (not line_certificate.whole_line_selection_proved
            or line_certificate.unresolved_positive_measure
            or not line_certificate.closed_axis_ancestry_is_retained
            or line_certificate.maximum_quarter_changes_per_carrier > 2):
        raise ValueError("Cannot close contact budgets on a positive-measure unresolved selector")
    banks = {row.trace_id: row for row in history.material_state.trace_bindings}
    if len(line_certificate.quarters) != 4 or len(banks) != 4:
        raise ValueError("Line work must retain all four original interfaces")
    quarter_budgets, pair_budgets = [], []
    initial_b, initial_c, final_b, final_c, independent_work = (I(0),)*5
    whole_area, strip_area = F(0), F(0)
    half_pi = interval_pi(precision+16)/2
    for quarter in line_certificate.quarters:
        receiver, donor = quarter.receiver_trace_id, quarter.original_donor_trace_id
        trace = _trace(history, receiver)
        for identity in quarter.possible_donor_trace_ids:
            other = _trace(history, identity)
            _parameters(trace, other.columns, other.law, other.water)
        branch = _branch_proof(history, quarter, precision)
        extent = quarter.computational_extent_radians
        low, high = quarter.core_computational_angle_bounds.lo, quarter.core_computational_angle_bounds.hi
        strips = (quarter.start_strip_computational_angle_bounds, quarter.end_strip_computational_angle_bounds)
        if (strips[0] != I(F(0), low) or strips[1] != I(high, extent)
                or not 0 < low < high < extent):
            raise ValueError("Core and nonzero strips must partition the entire original computational carrier")
        scale = half_pi/extent
        _, _, square0, square1, potential, coefficient = _core_fields(history, receiver, donor, precision, order)
        core = _outward(_hintegral(potential, low, high, scale, precision)*coefficient, precision)
        core0 = _outward(_hintegral(square0, low, high, scale, precision)*coefficient/2, precision)
        core1 = _outward(_hintegral(square1, low, high, scale, precision)*coefficient/2, precision)
        strip_measure = coefficient*(low+extent-high)
        # Every admitted segment has increasing negative gap.  Its potential
        # increment is nonpositive; the sum on ANY pair is bounded by the
        # receiver's whole potential decrement.  Switching changes no state.
        initial_strip, final_strip, strip_increment = (I(0),)*3
        strip_by_donor = {identity: I(0) for identity in banks}
        strip_area_by_donor = {identity: F(0) for identity in banks}
        for strip, sign, allowed_donors in zip(strips,
                (quarter.start_axis_sign, quarter.end_axis_sign),
                (quarter.start_strip_possible_donor_trace_ids, quarter.end_strip_possible_donor_trace_ids)):
            area = coefficient*strip.width
            before_strip = _outward(_hintegral(square0, strip.lo, strip.hi, scale, precision)*coefficient/2, precision)
            if sign < 0:
                # A proved negative owner-entry crossing preserves its old
                # frozen label.  The entire strip has the original harmonic
                # path, including the portion that changes geometric bank.
                after_strip = _outward(_hintegral(square1, strip.lo, strip.hi, scale, precision)*coefficient/2, precision)
                amount = _outward(_hintegral(potential, strip.lo, strip.hi, scale, precision)*coefficient, precision)
                strip_by_donor[donor] = _outward(strip_by_donor[donor]+amount, precision)
                strip_area_by_donor[donor] += area
            else:
                after_strip = _outward(_square(branch.final_gap_bounds_m)*area/2, precision)
                amount = _outward(after_strip-before_strip, precision)
                possible_increment = I(min(F(0), amount.lo), F(0))
                for identity in allowed_donors:
                    strip_by_donor[identity] = _outward(strip_by_donor[identity]+possible_increment, precision)
                    strip_area_by_donor[identity] += area
            initial_strip = _outward(initial_strip+before_strip, precision)
            final_strip = _outward(final_strip+after_strip, precision)
            strip_increment = _outward(strip_increment+amount, precision)
        coefficients = _family(trace)
        kn, ab, ac = coefficients
        amounts_coefficients = (kn, F(0), ab, ac, ab+ac, kn-ab-ac)
        for identity in sorted(banks):
            own = identity == donor
            core_amount, strip_amount = core if own else I(0), strip_by_donor[identity]
            potential_amount = _outward(core_amount+strip_amount, precision)
            amounts = tuple(_outward(item*potential_amount, precision) for item in amounts_coefficients)
            pair_budgets.append(NativeActiveContactLinePairBudget(
                quarter.receiver_fragment_id, receiver, banks[identity].donor_fragment_id, identity,
                coefficient*(high-low) if own else F(0), strip_area_by_donor[identity],
                core_amount, strip_amount, potential_amount, coefficients, *amounts))
        first, last = _outward(core0+initial_strip, precision), _outward(core1+final_strip, precision)
        independent_potential = _outward(core+strip_increment, precision)
        quarter_budgets.append(NativeActiveContactLineQuarterBudget(
            quarter.receiver_fragment_id, receiver, coefficient, coefficient*extent,
            coefficient*(high-low), strip_measure, scale, branch, first, last, independent_potential))
        initial_b, initial_c = (_outward(initial_b+ab*first, precision), _outward(initial_c+ac*first, precision))
        final_b, final_c = (_outward(final_b+ab*last, precision), _outward(final_c+ac*last, precision))
        independent_work = _outward(independent_work+kn*independent_potential, precision)
        whole_area, strip_area = whole_area+coefficient*extent, strip_area+strip_measure
    totals = tuple(_outward(sum((getattr(row, field) for row in pair_budgets), I(0)), precision)
                   for field in _BUDGET_FIELDS)
    work, dissipation, bulk_increment, contact_increment, total_increment, residual = totals
    _intersection(work, independent_work, "Receiving-pair work disagrees with independently enclosed endpoints")
    final_b = _intersection(final_b, initial_b+bulk_increment, "Bulk state disagrees with carried segment increments")
    final_c = _intersection(final_c, initial_c+contact_increment, "Contact state disagrees with carried segment increments")
    balance = _outward(work-bulk_increment-contact_increment-dissipation-residual, precision)
    if not balance.lo <= 0 <= balance.hi or dissipation != I(0):
        raise ValueError("Original line budgets do not enclose their full signed first law")
    result = NativeActiveContactLineWorkIntegral(history, line_certificate, precision, order,
        tuple(quarter_budgets), tuple(pair_budgets), whole_area, strip_area,
        initial_b, initial_c, _outward(initial_b+initial_c, precision),
        final_b, final_c, _outward(final_b+final_c, precision), *totals,
        independent_work, balance, payload, "")
    return replace(result, integral_sha256=hashlib.sha256(_encoded(_body(result))).hexdigest())


def _body(integral):
    from .geometric_native_active_contact_line import native_active_contact_line_to_dict

    result = {name: _portable(getattr(integral, name)) for name in integral.__dataclass_fields__
              if name not in ("history", "line_certificate", "source_history_payload_json", "integral_sha256")}
    result.update(format=FORMAT, model=MODEL, measure=MEASURE, first_law=FIRST_LAW,
        complete_original_source_history_sha256=hashlib.sha256(integral.source_history_payload_json).hexdigest(),
        line_selector=native_active_contact_line_to_dict(integral.line_certificate),
        temporal_quadrature_used=False, root_midpoint_selected=False,
        strip_centroid_or_representative_selected=False, tiny_strip_work_zeroed=False,
        original_nodal_ledgers_summed_with_continuous_work=False,
        original_affine_source_work_included=False, swept_material_area_used_as_contact_measure=False,
        transfer_work_j=[0, 1], transfer_energy_increment_j=[0, 1], transfer_dissipation_j=[0, 1],
        whole_line_compression_proved=True, whole_line_monotone_down_dip_proved=True,
        constitutive_source_scope="actual_constant_gap_zero_jump_zero_plastic_zero_cumulative_zero_tip_preparation",
        energy_and_work_share_each_pair_potential_enclosure=True,
        first_law_holds_as_exact_rational_coefficient_identity=True,
        proved_negative_owner_seam_strips_integrated_on_unchanged_frozen_reference=True,
        receiving_pair_weighted_contact_integrals_numerically_enclosed=True,
        all_receiver_angles_scheduled=True, candidate_mechanics_adopted=False,
        physical_pair_admission_authorized=False, accepted_native_force_intervals=0)
    return result


def validate_native_active_contact_line_work(integral, *, expected_source_history_dto):
    if type(integral) is not NativeActiveContactLineWorkIntegral:
        raise ValueError("Line work retains its typed source, selector and complete pair budgets")
    actual = integrate_native_active_contact_line_work(integral.history,
        expected_source_history_dto=expected_source_history_dto,
        line_certificate=integral.line_certificate, precision=integral.precision, order=integral.order)
    if actual != integral:
        raise ValueError("Original line work differs from full source and selector replay")


def native_active_contact_line_work_to_dict(integral, *, expected_source_history_dto):
    validate_native_active_contact_line_work(integral, expected_source_history_dto=expected_source_history_dto)
    result = _body(integral)
    result["integral_sha256"] = integral.integral_sha256
    return result


def native_active_contact_line_work_from_dict(raw, *, history, expected_source_history_dto,
        expected_line_certificate, precision=128, order=24):
    if type(raw) is not dict:
        raise ValueError("Line work restart requires its complete JSON object")
    _literal_json(raw)
    result = integrate_native_active_contact_line_work(history,
        expected_source_history_dto=expected_source_history_dto,
        line_certificate=expected_line_certificate, precision=precision, order=order)
    if _encoded(raw) != _encoded(native_active_contact_line_work_to_dict(result,
            expected_source_history_dto=expected_source_history_dto)):
        raise ValueError("Line work restart differs from external source, selector, measure or signed budget")
    return result
