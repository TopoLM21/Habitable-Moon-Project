"""Whole-angle, full-Euler geometry of the original four receiving quarters.

The proof is a shared-clock Taylor certificate for the complete pullback
M(u)=exp(-[Ad]u) exp([Ar]u).  Positive diagonal entries and the two
off-diagonal ratios enclose every angular carrier, including the extremely
small cross-axis sectors.  Each ratio has one strict turning point.  Thus an
open carrier has at most two original-quarter changes, and only one of its
Y/Z supports can change.  None of these conclusions comes from point samples.

This certificate distinguishes geometry from the positive fresh-owner gate.
A negative owner-entry seam cannot authorize a reassignment.  Tangencies do
not change quarters; closed-axis endpoints keep their original two ancestries.
The supplied transport law, full original source, and all saved Euler
components remain external authorities.  No force interval is admitted here.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction as F
from functools import lru_cache
from math import factorial
import hashlib
import json

from .geometric_native_pose import FractionInterval as I, interval_pi
from .geometric_native_contact_events import _polynomial_bounds, _isolate
from .geometric_native_physical_pair_time import _axes, _cross, _rotated
from .geometric_native_physical_history import _dyadic, _literal_json
from .geometric_native_first_entry_contact_domain_handover import _source
from .geometric_native_first_entry_interface_integral import _quarters
from .geometric_native_contact_reference_transport import _proof_key, _parameters
from .geometric_native_active_contact_binding import _axis_support


FORMAT = "native-active-receiver-contact-whole-line-geometry-1"
MODEL = "shared_clock_full_Euler_quarter_ratio_turning_and_positive_owner_gate"
_AUTHENTICATED = {}
_DERIVATIONS = {}
_AUTHENTICATED_SELECTIONS = {}


class UnresolvedNativeActiveContactLine(ValueError):
    """The original source does not satisfy this whole-angle proof domain."""


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()


def _fraction(value):
    return [value.numerator, value.denominator]


def _interval(value):
    return dict(lo=_fraction(value.lo), hi=_fraction(value.hi))


def _maximum(value):
    return max(abs(value.lo), abs(value.hi))


@dataclass(frozen=True, slots=True)
class NativeActiveContactLineSeries:
    """Exact coefficients for one column of the actual shared-clock map."""
    initial_coordinate_index: int
    coefficients: tuple
    rate_bound: F
    precision: int
    order: int

    def bounds(self, coordinate, clock=I(0, 1), derivative=0, divide_clock=False):
        if type(coordinate) is not int or coordinate not in (0, 1, 2):
            raise ValueError("A whole-line series needs an original Cartesian coordinate")
        if type(clock) is not I or not 0 <= clock.lo <= clock.hi <= 1:
            raise ValueError("Whole-line bounds need an exact original-clock interval")
        if type(derivative) is not int or not 0 <= derivative <= 2:
            raise ValueError("Whole-line series supports literal derivative orders zero to two")
        if type(divide_clock) is not bool or divide_clock and derivative:
            raise ValueError("Exact division by the original clock precedes differentiation")
        rows = tuple(row[coordinate] for row in self.coefficients)
        if divide_clock:
            if rows[0] != 0:
                raise ValueError("Only an authenticated zero initial component can be divided by clock")
            rows = rows[1:]
        coefficients = tuple(I(value*F(factorial(k), factorial(k-derivative)))
            for k, value in enumerate(rows) if k >= derivative)
        # The induced l1 norm bounds each coefficient by L^n/n!.
        # Division by u removes an exact zero and lowers the polynomial
        # degree; its omitted tail is still bounded uniformly at u=0.
        first = self.rate_bound**(self.order+1)/factorial(self.order+1-derivative)
        ratio = self.rate_bound/F(self.order+2-derivative)
        if ratio >= 1:
            raise UnresolvedNativeActiveContactLine("Whole-line Taylor remainder exceeds its source domain")
        tail = first/(1-ratio)
        return _polynomial_bounds(coefficients, clock, tail, self.precision)


@lru_cache(maxsize=16)
def _column_series(donor_axis, receiver_axis, coordinate, precision, order):
    """Convolve both exact generators before any interval enclosure."""
    unit = tuple(F(int(index == coordinate)) for index in range(3))
    receiver_terms = [unit]
    for degree in range(1, order+1):
        receiver_terms.append(tuple(value/degree for value in
            _cross(receiver_axis, receiver_terms[-1])))
    result = [[F(0)]*3 for _ in range(order+1)]
    negative_donor = tuple(-value for value in donor_axis)
    for receiver_degree, first in enumerate(receiver_terms):
        term = first
        for donor_degree in range(order-receiver_degree+1):
            degree = receiver_degree+donor_degree
            result[degree] = [a+b for a, b in zip(result[degree], term)]
            if degree < order:
                term = tuple(value/(donor_degree+1) for value in
                    _cross(negative_donor, term))
    rate = sum(map(abs, donor_axis), F(0))+sum(map(abs, receiver_axis), F(0))
    return NativeActiveContactLineSeries(coordinate, tuple(map(tuple, result)),
        rate, precision, order)


@dataclass(frozen=True, slots=True)
class NativeActiveContactLineAxisProof:
    axis_coordinate_index: int
    seam_coordinate_index: int
    diagonal_bounds: I
    offdiagonal_bounds: I
    offdiagonal_ratio_absolute_bound: F
    turning_numerator_derivative_bounds: I
    turning_numerator_at_zero_bounds: I
    turning_numerator_at_one_bounds: I
    unique_turning_clock_bounds: I
    seam_ratio_at_turning_bounds: I
    seam_ratio_at_one_bounds: I
    positive_axis_other_support_bounds: I
    positive_axis_owner_depth_over_clock_bounds: I
    positive_axis_fresh_owner_derivative_bounds: I

    @property
    def maximum_quarter_changes_per_carrier(self):
        return 2

    @property
    def negative_axis_fresh_owner_derivative_bounds(self):
        return -self.positive_axis_fresh_owner_derivative_bounds


@dataclass(frozen=True, slots=True)
class NativeActiveContactLineQuarter:
    receiver_fragment_id: str
    receiver_trace_id: str
    original_donor_fragment_id: str
    original_donor_trace_id: str
    signs_y_z: tuple
    computational_extent_radians: F
    native_extent_radians: I
    core_computational_angle_bounds: I
    start_strip_computational_angle_bounds: I
    end_strip_computational_angle_bounds: I
    start_axis_coordinate_index: int
    end_axis_coordinate_index: int
    start_axis_sign: int
    end_axis_sign: int
    possible_donor_trace_ids: tuple
    start_strip_possible_donor_trace_ids: tuple
    end_strip_possible_donor_trace_ids: tuple


@dataclass(frozen=True, slots=True)
class NativeActiveContactLineCertificate:
    history: object
    law: object
    quarters: tuple
    columns: tuple
    axes: tuple
    donor_axis: tuple
    receiver_axis: tuple
    common_computational_strip_width_radians: F
    native_strip_width_upper_bound_radians: F
    precision: int
    order: int
    source_history_payload_json: bytes
    certificate_sha256: str

    @property
    def all_open_receiver_angles_geometrically_classified(self):
        return True

    @property
    def whole_line_selection_proved(self):
        return True

    @property
    def unresolved_positive_measure(self):
        return False

    @property
    def maximum_quarter_changes_per_carrier(self):
        return 2

    @property
    def closed_axis_ancestry_is_retained(self):
        return True

    @property
    def physical_pair_admission_authorized(self):
        return False

    @property
    def candidate_mechanics_adopted(self):
        return False

    @property
    def accepted_native_force_intervals(self):
        return 0


@dataclass(frozen=True, slots=True)
class NativeActiveContactLineImplicitBranch:
    """One exact monotone inverse expression, conditional on its open range.

    The condition is part of the selected mathematical expression.  Its
    boundary retains the old binding: at a turn there is no sign change,
    and at the terminal clock there is no positive-duration right segment.
    An endpoint enclosure is evidence about that expression, not its value.
    """
    sequence: int
    axis_coordinate_index: int
    seam_coordinate_index: int
    carrier_ratio: F
    time_branch: str
    exact_existence_condition: str
    initial_ratio_bound: I
    final_ratio_bound: I
    incoming_seam_sign: int
    incoming_donor_fragment_id: str
    incoming_donor_trace_id: str
    owner_depth_derivative_bounds: I
    known_existence: str


@dataclass(frozen=True, slots=True)
class NativeActiveContactLineCarrierSelection:
    certificate: NativeActiveContactLineCertificate
    receiver_fragment_id: str
    receiver_trace_id: str
    receiver_reference_coordinates: tuple
    original_donor_fragment_id: str
    original_donor_trace_id: str
    classification: str
    candidate_axis_coordinate_index: int | None
    candidate_axis_sign: int | None
    implicit_branches: tuple
    closed_endpoint_donor_fragment_ids: tuple
    closed_endpoint_donor_trace_ids: tuple
    selection_sha256: str

    @property
    def maximum_material_handovers(self):
        return sum(branch.known_existence != "proved_absent" for branch in self.implicit_branches)

    @property
    def frozen_label_between_admissible_events(self):
        return True

    @property
    def single_endpoint_ancestry_selected(self):
        return False

    @property
    def physical_pair_admission_authorized(self):
        return False

    @property
    def accepted_native_force_intervals(self):
        return 0


def _axis_proof(series, other_series, donor_axis, receiver_axis):
    axis = series.initial_coordinate_index
    seam = 3-axis
    # q_seam = M[seam,axis] c_axis + M[seam,seam] c_seam.
    # The denominator is the other column's seam diagonal, rather than
    # this column's axis diagonal: the complete rotation is not symmetric.
    a = lambda clock=I(0, 1), derivative=0: other_series.bounds(seam, clock, derivative)
    b = lambda clock=I(0, 1), derivative=0: series.bounds(seam, clock, derivative)
    diagonal, offdiagonal = a(), b()
    if diagonal.lo <= 0:
        raise UnresolvedNativeActiveContactLine("Whole-line pullback diagonal is not strictly positive")
    width = _maximum(offdiagonal)/diagonal.lo
    numerator = lambda clock: b(clock, 1)*a(clock)-b(clock)*a(clock, 1)
    derivative = b(derivative=2)*a()-b()*a(derivative=2)
    first, last = numerator(I(0)), numerator(I(1))
    increasing = derivative.lo > 0
    if not increasing and derivative.hi >= 0:
        raise UnresolvedNativeActiveContactLine("The whole-angle seam ratio lacks a strict turning law")
    if (increasing and not first.hi < 0 < last.lo
            or not increasing and not last.hi < 0 < first.lo):
        raise UnresolvedNativeActiveContactLine("The whole-angle seam ratio lacks its unique interior turn")
    turning = _isolate(numerator, F(0), F(1), increasing=increasing,
        relative_digits=24)
    ratio_at_turn = -b(turning)/a(turning)
    ratio_at_one = -b(I(1))/a(I(1))
    # Any root has |c_seam/c_axis| <= width.  The entire corresponding
    # unit carrier is enclosed without choosing an angular representative.
    radius = width*2
    if radius >= F(1, 4):
        raise UnresolvedNativeActiveContactLine("Cross-axis support strips are not disjoint")
    c_axis, c_seam = I(1-radius*radius, 1), I(-radius, radius)
    q = tuple(series.bounds(k)*c_axis+other_series.bounds(k)*c_seam
        for k in range(3))
    q = tuple(I(0) if k == seam else value for k, value in enumerate(q))
    depth_over_clock = series.bounds(0, divide_clock=True)*c_axis \
        +other_series.bounds(0, divide_clock=True)*c_seam
    normal = _rotated((F(1), F(0), F(0)), receiver_axis, I(0, 1),
        series.precision, series.order)
    moved = _rotated(q, donor_axis, I(0, 1), series.precision, series.order)
    fresh = sum(((ad-ar)*component for ad, ar, component in
        zip(donor_axis, receiver_axis, _cross(normal, moved))), I(0))
    if q[axis].lo <= 0 or depth_over_clock.lo <= 0 or fresh.lo <= 0:
        raise UnresolvedNativeActiveContactLine("A positive-axis seam lacks a whole-domain first-owner proof")
    return NativeActiveContactLineAxisProof(axis, seam, diagonal, offdiagonal,
        width, derivative, first, last, turning, ratio_at_turn, ratio_at_one,
        q[axis], depth_over_clock, fresh)


def certify_native_active_contact_line(history, *, law, expected_source_history_dto,
        precision=128, order=16):
    """Certify the geometry and owner gate over all four original quarters.

    The core has no original-quarter changes.  Each endpoint strip has at
    most two changes across that endpoint's support, preserving its other
    support.  A positive-axis crossing is a unique fresh owner entry;
    antipodal crossings fail that gate and preserve the currently frozen
    binding.  Retention follows from the same law that permits reassignment
    only at proved fresh positive quarter changes; it admits no new contact.
    A tangent root makes no quarter change, and endpoint ancestry is not
    resolved by an identifier.  An inconclusive proof still raises.
    """
    payload = _source(history, expected_source_history_dto, law)
    if type(precision) is not int or not 96 <= precision <= 1024:
        raise ValueError("Whole-line precision must be a literal integer from 96 to 1024")
    if type(order) is not int or not 16 <= order <= 48:
        raise ValueError("Whole-line Taylor order must be a literal integer from 16 to 48")
    key = _proof_key((history, law, precision, order))
    if key in _DERIVATIONS:
        return _DERIVATIONS[key]
    rows = _quarters(history)
    for _, trace, donor_supports, receiver_supports in rows:
        start, end = tuple(map(_dyadic, trace.start)), tuple(map(_dyadic, trace.end))
        signs = {1: int(start[1]+end[1]), 2: int(start[2]+end[2])}
        if (len(donor_supports) != 3 or len(receiver_supports) != 3
                or dict(_axis_support(normal) for normal in donor_supports) != {0: 1, **signs}
                or dict(_axis_support(normal) for normal in receiver_supports) != {0: -1, **signs}):
            raise UnresolvedNativeActiveContactLine("The original banks are not the exact opposite-X axis-quarter partition")
    owners = {(bank.subducting_plate, bank.overriding_plate) for bank, _, _, _ in rows}
    if len(owners) != 1:
        raise UnresolvedNativeActiveContactLine("The original four quarters do not share one physical owner pair")
    donor_axis, receiver_axis = _axes(history.omega_records, history.dt_myr,
        *next(iter(owners)))
    columns = tuple(_column_series(donor_axis, receiver_axis, index, precision, order)
        for index in (1, 2))
    axes = (_axis_proof(columns[0], columns[1], donor_axis, receiver_axis),
            _axis_proof(columns[1], columns[0], donor_axis, receiver_axis))
    half_pi = interval_pi(precision+16)/2
    extents = {item.trace_id: _dyadic(item.extent_radians)
        for item in history.contact_time_path.step.certificates}
    if len({extents[trace.trace_id] for _, trace, _, _ in rows}) != 1:
        raise UnresolvedNativeActiveContactLine("The original quarter computational extents differ")
    extent = extents[rows[0][1].trace_id]
    ratio_width = max(axis.offdiagonal_ratio_absolute_bound for axis in axes)
    # atan(r) <= r.  The factor two gives a strict no-switch core, with
    # separate exact computational and native angular coordinates.
    strip = 2*extent*ratio_width/half_pi.lo
    native_strip_upper = strip*half_pi.hi/extent
    if 2*strip >= extent or native_strip_upper >= F(1, 4):
        raise UnresolvedNativeActiveContactLine("The two endpoint strips do not leave a proved complete core")
    by_signs = {}
    for bank, trace, _, _ in rows:
        start, end = tuple(map(_dyadic, trace.start)), tuple(map(_dyadic, trace.end))
        by_signs[(int(start[1]+end[1]), int(start[2]+end[2]))] = (bank, trace)
    quarters = []
    for bank, trace, _, _ in rows:
        start, end = tuple(map(_dyadic, trace.start)), tuple(map(_dyadic, trace.end))
        signs = (int(start[1]+end[1]), int(start[2]+end[2]))
        start_axis = next(index for index in (1, 2) if start[index])
        end_axis = 3-start_axis
        start_neighbor = list(signs)
        start_neighbor[(3-start_axis)-1] *= -1
        end_neighbor = list(signs)
        end_neighbor[(3-end_axis)-1] *= -1
        start_trace = by_signs[tuple(start_neighbor)][1].trace_id
        end_trace = by_signs[tuple(end_neighbor)][1].trace_id
        for donor_signs in (signs, tuple(start_neighbor), tuple(end_neighbor)):
            donor_trace = by_signs[donor_signs][1]
            _parameters(trace, donor_trace.columns, donor_trace.law, donor_trace.water)
        quarters.append(NativeActiveContactLineQuarter(bank.receiver_fragment_id,
            trace.trace_id, bank.donor_fragment_id, trace.trace_id, signs,
            extent, half_pi, I(strip, extent-strip), I(0, strip),
            I(extent-strip, extent), start_axis, end_axis, int(start[start_axis]),
            int(end[end_axis]), (trace.trace_id, start_trace, end_trace),
            (trace.trace_id, start_trace), (trace.trace_id, end_trace)))
    result = NativeActiveContactLineCertificate(history, law, tuple(quarters), columns,
        axes, donor_axis, receiver_axis, strip, native_strip_upper, precision, order,
        payload, "")
    body = _encoded(_body(result))
    result = replace(result, certificate_sha256=hashlib.sha256(body).hexdigest())
    if len(_DERIVATIONS) >= 8:
        _DERIVATIONS.clear()
    if len(_AUTHENTICATED) >= 16:
        _AUTHENTICATED.clear()
    _DERIVATIONS[key], _AUTHENTICATED[_proof_key(result)] = result, body
    return result


def _body(value):
    return dict(format=FORMAT, model=MODEL,
        source_history=json.loads(value.source_history_payload_json),
        explicit_law=dict(model=value.law.model, provenance=value.law.provenance),
        precision=value.precision, order=value.order,
        donor_axis=[_fraction(v) for v in value.donor_axis],
        receiver_axis=[_fraction(v) for v in value.receiver_axis],
        common_computational_strip_width_radians=_fraction(value.common_computational_strip_width_radians),
        native_strip_width_upper_bound_radians=_fraction(value.native_strip_width_upper_bound_radians),
        columns=[dict(initial_coordinate_index=column.initial_coordinate_index,
            coefficients=[[_fraction(v) for v in row] for row in column.coefficients],
            rate_bound=_fraction(column.rate_bound), remainder_model="l1_shared_clock_exponential_tail")
            for column in value.columns],
        axes=[{name: (_interval(item) if type(item) is I else _fraction(item)
            if type(item) is F else item) for name, item in
            ((name, getattr(axis, name)) for name in axis.__dataclass_fields__)}
            for axis in value.axes],
        quarters=[{name: (_interval(item) if type(item) is I else _fraction(item)
            if type(item) is F else list(item) if type(item) is tuple else item)
            for name, item in ((name, getattr(quarter, name)) for name in quarter.__dataclass_fields__)}
            for quarter in value.quarters],
        all_open_receiver_angles_geometrically_classified=True,
        whole_line_selection_proved=True, unresolved_positive_measure=False,
        maximum_quarter_changes_per_carrier=2,
        only_one_axis_can_change_for_each_carrier=True,
        original_opposite_X_hemisphere_axis_quarter_partition_proved=True,
        no_quarter_change_in_each_core=True,
        tangency_policy="no_support_sign_change_no_handover",
        tangent_carrier_measure="at_most_four_isolated_angles_zero_original_interface_measure",
        terminal_time_seam_policy="no_incoming_positive_duration_segment_inside_original_clock_no_transfer",
        positive_axis_seams="unique_positive_first_owner_entry",
        fresh_owner_proof="D(q_star,s)=0_by_full_carrier_identity_and_D_v_positive_on_complete_0_to_s",
        distinct_accepted_root_clocks_have_distinct_original_donor_material=True,
        same_original_donor_material_reentry_allocated_again=False,
        negative_axis_seams="proved_negative_owner_entry_retain_current_frozen_binding",
        selection_law="switch_only_at_unique_positive_fresh_first_owner_quarter_change_else_preserve_binding",
        inconclusive_owner_entry_proof_retained_as_if_negative=False,
        geometric_crossings_identified_with_material_handovers=False,
        closed_axis_endpoint_policy="retain_both_original_closed_ancestries_null_interface_measure",
        original_interface_measure="C_dtheta_computational_du_or_C_Gamma_over_native_half_pi_dtheta_native_du",
        geometric_proof_from_point_samples=False,
        tiny_cross_axis_strips_discarded=False,
        true_native_quarter_equals_saved_binary64_extent=False,
        material_label_reprojected_during_segment=False,
        root_midpoint_selected=False, identifier_tie_break_used=False,
        physical_pair_admission_authorized=False, candidate_mechanics_adopted=False,
        accepted_native_force_intervals=0)


def validate_native_active_contact_line(value):
    if type(value) is not NativeActiveContactLineCertificate:
        raise ValueError("A whole-line certificate needs its immutable complete source derivation")
    key = _proof_key(value)
    if key not in _AUTHENTICATED:
        rebuilt = certify_native_active_contact_line(value.history, law=value.law,
            expected_source_history_dto=json.loads(value.source_history_payload_json),
            precision=value.precision, order=value.order)
        if _proof_key(rebuilt) != key:
            raise ValueError("Whole-line geometry differs from its complete original-source replay")
    return value


def native_active_contact_line_to_dict(value):
    validate_native_active_contact_line(value)
    result = json.loads(_AUTHENTICATED[_proof_key(value)])
    result["certificate_sha256"] = value.certificate_sha256
    return result


def native_active_contact_line_from_dict(raw, *, expected_history, expected_law,
        expected_source_history_dto):
    if type(raw) is not dict:
        raise ValueError("Whole-line restart requires its complete literal JSON object")
    _literal_json(raw)
    try:
        result = certify_native_active_contact_line(expected_history, law=expected_law,
            expected_source_history_dto=expected_source_history_dto,
            precision=raw["precision"], order=raw["order"])
    except KeyError as error:
        raise ValueError("Whole-line restart omits its literal proof budgets") from error
    if _encoded(raw) != _encoded(native_active_contact_line_to_dict(result)):
        raise ValueError("Whole-line restart differs from the full source, law, or geometric proof")
    return result


def _existence(ratio, first, last):
    lower = I(min(first.lo, last.lo), min(first.hi, last.hi))
    upper = I(max(first.lo, last.lo), max(first.hi, last.hi))
    if lower.hi < ratio < upper.lo:
        return "proved_present"
    if ratio <= lower.lo or ratio >= upper.hi:
        return "proved_absent"
    # The exact analytic predicate remains the operative selector.  This
    # finite-precision bracket is never promoted to equality or a midpoint.
    return "exact_analytic_condition"


def select_native_active_contact_line_carrier(certificate, receiver_fragment_id,
        receiver_reference_coordinates, *, expected_source_history_dto):
    """Select an arbitrary exact original carrier by the whole-angle rule.

    Positive-axis candidates retain their two ordered, unique implicit
    inverse branches and exact open-range predicates.  This representation
    handles an exact tangent or terminal-clock equality without approximating
    it as a crossing.  Negative-axis candidates provably cannot pass the
    fresh-owner gate and keep the original label.  Closed endpoints retain
    both source ancestries with zero one-dimensional interface measure.
    """
    validate_native_active_contact_line(certificate)
    _source(certificate.history, expected_source_history_dto, certificate.law)
    if type(receiver_fragment_id) is not str:
        raise ValueError("Whole-line carrier selection requires an original receiving fragment")
    value = receiver_reference_coordinates
    if (type(value) is not tuple or len(value) != 3
            or any(type(item) not in (int, F) for item in value)):
        raise ValueError("Whole-line carrier coordinates must be an exact immutable Cartesian triple")
    carrier = tuple(F(item) for item in value)
    if carrier[0] != 0 or not any(carrier):
        raise ValueError("Whole-line selection requires a nonzero original X=0 ray")
    quarters = tuple(quarter for quarter in certificate.quarters
        if quarter.receiver_fragment_id == receiver_fragment_id)
    if len(quarters) != 1:
        raise ValueError("The exact carrier must name one original receiving quarter")
    quarter = quarters[0]
    if any(carrier[index]*quarter.signs_y_z[index-1] < 0 for index in (1, 2)):
        raise ValueError("The exact original carrier lies outside its declared receiving quarter")
    closed_fragments, closed_traces, branches = (), (), ()
    candidate, sign = None, None
    if not carrier[1] or not carrier[2]:
        containing = tuple(item for item in certificate.quarters if all(
            carrier[index]*item.signs_y_z[index-1] >= 0 for index in (1, 2)))
        if len(containing) != 2:
            raise UnresolvedNativeActiveContactLine("An original axis endpoint lacks both closed source ancestries")
        closed_fragments = tuple(item.original_donor_fragment_id for item in containing)
        closed_traces = tuple(item.original_donor_trace_id for item in containing)
        classification = "closed_original_endpoint_both_ancestries_zero_interface_measure"
    else:
        candidates = tuple(axis for axis in certificate.axes if abs(
            carrier[axis.seam_coordinate_index]/carrier[axis.axis_coordinate_index])
            <= axis.offdiagonal_ratio_absolute_bound)
        if len(candidates) > 1:
            raise UnresolvedNativeActiveContactLine("The exact carrier contradicts the disjoint whole-line strip proof")
        if not candidates:
            classification = "original_frozen_binding_no_quarter_change"
        else:
            axis = candidates[0]
            candidate = axis.axis_coordinate_index
            sign = 1 if carrier[candidate] > 0 else -1
            if sign < 0:
                classification = "proved_negative_owner_entry_retain_original_frozen_binding"
            else:
                ratio = carrier[axis.seam_coordinate_index]/carrier[candidate]
                turning, terminal = axis.seam_ratio_at_turning_bounds, axis.seam_ratio_at_one_bounds
                increasing_turning_numerator = axis.turning_numerator_derivative_bounds.lo > 0
                # r'=-W/A^2.  W increases through zero near +Y, and
                # decreases through zero near +Z.  These are actual right
                # germs, rather than a choice among closed seam identifiers.
                first_incoming = -1 if increasing_turning_numerator else 1
                first_condition = "min(0,r(t_turn)) < lambda < max(0,r(t_turn))"
                second_condition = "min(r(t_turn),r(1)) < lambda < max(r(t_turn),r(1))"
                implicit = []
                for sequence, time_branch, condition, first, last, incoming in (
                        (1, "0 < root < exact_unique_turning_clock", first_condition,
                            I(0), turning, first_incoming),
                        (2, "exact_unique_turning_clock < root < 1", second_condition,
                            turning, terminal, -first_incoming)):
                    signs = list(quarter.signs_y_z)
                    signs[axis.seam_coordinate_index-1] = incoming
                    donors = tuple(item for item in certificate.quarters if item.signs_y_z == tuple(signs))
                    if len(donors) != 1:
                        raise UnresolvedNativeActiveContactLine("A positive seam right germ lacks its unique original donor quarter")
                    donor = donors[0]
                    implicit.append(NativeActiveContactLineImplicitBranch(sequence, candidate,
                        axis.seam_coordinate_index, ratio, time_branch, condition, first, last,
                        incoming, donor.original_donor_fragment_id, donor.original_donor_trace_id,
                        axis.positive_axis_fresh_owner_derivative_bounds,
                        _existence(ratio, first, last)))
                branches = tuple(implicit)
                classification = ("positive_fresh_owner_unique_monotone_implicit_branches"
                    if any(branch.known_existence != "proved_absent" for branch in branches)
                    else "original_frozen_binding_no_quarter_change")
    result = NativeActiveContactLineCarrierSelection(certificate, receiver_fragment_id,
        quarter.receiver_trace_id, carrier, quarter.original_donor_fragment_id,
        quarter.original_donor_trace_id, classification, candidate, sign, branches,
        closed_fragments, closed_traces, "")
    body = _encoded(_selection_body(result))
    result = replace(result, selection_sha256=hashlib.sha256(body).hexdigest())
    if len(_AUTHENTICATED_SELECTIONS) >= 64:
        _AUTHENTICATED_SELECTIONS.clear()
    _AUTHENTICATED_SELECTIONS[_proof_key(result)] = body
    return result


def _selection_body(value):
    certificate = value.certificate
    return dict(format="native-active-receiver-contact-whole-line-carrier-selection-1",
        model=MODEL, certificate_sha256=certificate.certificate_sha256,
        receiver_fragment_id=value.receiver_fragment_id, receiver_trace_id=value.receiver_trace_id,
        receiver_reference_coordinates=[_fraction(item) for item in value.receiver_reference_coordinates],
        original_donor_fragment_id=value.original_donor_fragment_id,
        original_donor_trace_id=value.original_donor_trace_id,
        classification=value.classification,
        candidate_axis_coordinate_index=value.candidate_axis_coordinate_index,
        candidate_axis_sign=value.candidate_axis_sign,
        implicit_branches=[{name: (_interval(item) if type(item) is I else _fraction(item)
            if type(item) is F else item) for name, item in
            ((name, getattr(branch, name)) for name in branch.__dataclass_fields__)}
            for branch in value.implicit_branches],
        exact_seam_equation="lambda = r(u) = -M[seam,axis](u)/M[seam,seam](u)",
        exact_turning_expression="unique W(u)=0 where W=B_prime*A-B*A_prime",
        exact_frozen_event_reference="exp(-Ad*root)*exp(Ar*root)*unit(original_receiver_c)",
        initial_frozen_reference="unit(original_receiver_c)",
        implicit_branch_condition_refers_to_exact_expressions=True,
        interval_endpoint_or_midpoint_promoted_to_exact_root=False,
        positive_branch_handover="iff_exact_open_range_condition_true_consume_predecessor_once",
        branches_ordered_by_exact_turning_clock=True,
        closed_endpoint_donor_fragment_ids=list(value.closed_endpoint_donor_fragment_ids),
        closed_endpoint_donor_trace_ids=list(value.closed_endpoint_donor_trace_ids),
        frozen_label_between_admissible_events=True,
        original_closed_endpoint_has_finite_interface_area=False,
        identifier_tie_break_used=False, numerical_tolerance_used_for_branch_acceptance=False,
        geometric_crossings_identified_with_material_handovers=False,
        physical_pair_admission_authorized=False, candidate_mechanics_adopted=False,
        accepted_native_force_intervals=0)


def native_active_contact_line_carrier_to_dict(value, *, expected_source_history_dto):
    if type(value) is not NativeActiveContactLineCarrierSelection:
        raise ValueError("Whole-line carrier selection requires its authenticated typed expression")
    rebuilt = select_native_active_contact_line_carrier(value.certificate, value.receiver_fragment_id,
        value.receiver_reference_coordinates, expected_source_history_dto=expected_source_history_dto)
    if _proof_key(rebuilt) != _proof_key(value):
        raise ValueError("Whole-line carrier selection differs from its source-bound analytic rule")
    result = json.loads(_AUTHENTICATED_SELECTIONS[_proof_key(value)])
    result["selection_sha256"] = value.selection_sha256
    return result


def native_active_contact_line_carrier_from_dict(raw, *, expected_certificate,
        expected_receiver_fragment_id, expected_receiver_reference_coordinates,
        expected_source_history_dto):
    if type(raw) is not dict:
        raise ValueError("Whole-line carrier restart requires a complete literal JSON object")
    _literal_json(raw)
    result = select_native_active_contact_line_carrier(expected_certificate, expected_receiver_fragment_id,
        expected_receiver_reference_coordinates, expected_source_history_dto=expected_source_history_dto)
    if _encoded(raw) != _encoded(native_active_contact_line_carrier_to_dict(result,
            expected_source_history_dto=expected_source_history_dto)):
        raise ValueError("Whole-line carrier restart differs from the exact external source, quarter or ray")
    return result
