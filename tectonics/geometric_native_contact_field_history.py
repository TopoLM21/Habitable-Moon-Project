"""Original continuous contact configuration at an implicit angle and clock.

The source is the complete replay-authenticated physical history, including
every previous and unaccepted endpoint polynomial piece.  The observed state
is the original inviscid dense return-map family; it is not an interpolation
of Gauss nodes or a finite local state chosen inside a first-root enclosure.
Energy is per unit original interface area.  This module never invents a work
density from the retained nodal endpoint ledgers and never adopts the native
force candidate.  A declared expression context retains an implicit root's
identity, but geometric authentication of that expression belongs to its
material-port caller.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
import hashlib
import json

from .geometric_native_contact_time_support import (
    ADMISSIBLE, MODEL as CONTACT_TIME_MODEL, _add, _field, _scale,
)
from .geometric_native_physical_history import (
    NativePhysicalHistoryBinding, _dyadic, _encoded, _literal_json,
    _precision, _typed_fingerprint, native_physical_history_binding_to_dict,
    validate_native_physical_history_binding,
)
from .geometric_native_pose import FractionInterval as I, _rounded_interval, interval_pi


FORMAT = "native-original-continuous-contact-field-query-1"
MODEL = "original_inviscid_dense_return_map_entire_piecewise_history"
PIECE_SELECTION = "all_closed_original_piece_intersections_no_node_or_midpoint_selection"
ENERGY_MEASURE = "joules_per_square_metre_of_original_interface_area"


def _interval(value, name, *, maximum=None):
    if (type(value) is not I or type(value.lo) is not Fraction
            or type(value.hi) is not Fraction or value.lo < 0
            or value.lo > value.hi or maximum is not None and value.hi > maximum):
        raise ValueError(f"{name} must retain its literal ordered Fraction interval")
    return value


def _source(history, expected):
    validate_native_physical_history_binding(history)
    if type(expected) is not dict:
        raise ValueError("Continuous field history requires the complete independent source DTO")
    _literal_json(expected)
    payload = _encoded(expected)
    if payload != _encoded(native_physical_history_binding_to_dict(history)):
        raise ValueError("Continuous field source differs from the full original physical replay")
    return payload


def _context(value):
    if value is not None and (type(value) is not dict or not value):
        raise ValueError("An implicit source expression needs its complete nonempty JSON context")
    _literal_json(value)
    return _encoded(value)


def _outward(value, precision):
    # The extra relative binary grid only compresses outward.  It does not
    # round a root, select a midpoint, impose a floor, or grant a branch sign.
    return _rounded_interval(value, precision + 16)


def _hull(values):
    return I(min(value.lo for value in values), max(value.hi for value in values))


def _positive(value):
    return I(max(Fraction(0), value.lo), max(Fraction(0), value.hi))


def _absolute(value):
    return (I(0, max(-value.lo, value.hi)) if value.lo <= 0 <= value.hi
            else -value if value.hi < 0 else value)


def _square(value):
    # Interval multiplication x*x would introduce negative energy when x
    # contains zero.  The exact image of squaring has a nonnegative lower end.
    return I(0 if value.lo <= 0 <= value.hi else min(value.lo**2, value.hi**2),
             max(value.lo**2, value.hi**2))


@lru_cache(maxsize=2048)
def _compressed_sin_cos_point(value, precision):
    """Certified alternating Taylor sums on a bounded-denominator grid.

    Implicit angles carry hundreds of exact denominator bits.  Raising those
    denominators through an uncompressed rational Taylor sum is unnecessary:
    every recurrence and partial sum is enclosed outward, and the decreasing
    alternating remainder is bounded by the next term's exact enclosure.
    """
    if not value:
        return I(0), I(1)
    squared = value * value
    threshold = Fraction(1, 10**precision)

    def series(sine):
        term = I(value if sine else Fraction(1))
        partial, index = term, 0
        while True:
            n = 2 * index + (2 if sine else 1)
            following = _outward(-term * squared / (n * (n + 1)), precision + 8)
            tail = max(abs(following.lo), abs(following.hi))
            if squared <= (n + 2) * (n + 3) and tail <= threshold:
                return _outward(partial + I(-tail, tail), precision)
            partial = _outward(partial + following, precision + 8)
            term, index = following, index + 1
    return series(True), series(False)


@lru_cache(maxsize=2048)
def _compressed_trig_ranges(low, high, precision):
    ends = [_compressed_sin_cos_point(value, precision) for value in (low, high)]
    sine = _hull([pair[0] for pair in ends])
    cosine = _hull([pair[1] for pair in ends])
    pi = interval_pi(precision + 8)
    for multiple, value in ((Fraction(1, 2), 1), (Fraction(3, 2), -1)):
        point = pi * multiple
        if low <= point.hi and high >= point.lo:
            sine = I(min(sine.lo, value), max(sine.hi, value))
    for multiple, value in ((0, 1), (1, -1), (2, 1)):
        point = pi * multiple
        if low <= point.hi and high >= point.lo:
            cosine = I(min(cosine.lo, value), max(cosine.hi, value))
    return sine, cosine


def _field_bounds(field, low, high, precision):
    result = I(field[0])
    if field[1] or field[2]:
        sine, cosine = _compressed_trig_ranges(low, high, precision)
        result = _outward(result + cosine * field[1] + sine * field[2], precision)
    if field[3] or field[4]:
        sine, cosine = _compressed_trig_ranges(2 * low, 2 * high, precision)
        result = _outward(result + cosine * field[3] + sine * field[4], precision)
    return result


def _field_affine_bounds(a, delta, low, high, first_time, last_time, precision):
    return _hull([_field_bounds(_add(a, _scale(delta, time)), low, high, precision)
                  for time in (first_time, last_time)])


@dataclass(frozen=True, slots=True)
class NativeContactFieldMemoryIntersection:
    piece_index: int
    computational_angle_bounds: I
    original_piece: object


@dataclass(frozen=True, slots=True)
class NativeContactFieldPieceQuery:
    previous_piece_index: int
    computational_angle_bounds: I
    free_gap_bounds_m: I
    free_jump_bounds_m: I
    previous_plastic_slip_bounds_m: I
    previous_cumulative_slip_bounds_m: I
    plastic_increment_bounds_m: I
    plastic_slip_bounds_m: I
    cumulative_slip_bounds_m: I
    normal_pressure_bounds_pa: I
    shear_traction_bounds_pa: I
    contact_gap_bounds_m: I
    contact_jump_bounds_m: I
    bulk_normal_strain_bounds_m: I
    bulk_shear_strain_bounds_m: I
    contact_elastic_shear_strain_bounds_m: I
    bulk_energy_density_bounds_j_m2: I
    contact_energy_density_bounds_j_m2: I
    total_energy_density_bounds_j_m2: I
    constitutive_status: str


@dataclass(frozen=True, slots=True)
class NativeContactFieldHistoryQuery:
    history: NativePhysicalHistoryBinding
    source_history_payload_json: bytes
    source_expression_context_payload_json: bytes
    trace_id: str
    computational_angle_bounds: I
    time_fraction_bounds: I
    elapsed_time_s_bounds: I
    native_elapsed_time_s_bounds: I
    physical_time_myr_bounds: I
    free_gap_bounds_m: I
    free_jump_bounds_m: I
    previous_plastic_slip_bounds_m: I
    previous_cumulative_slip_bounds_m: I
    plastic_increment_bounds_m: I
    plastic_slip_bounds_m: I
    cumulative_slip_bounds_m: I
    normal_pressure_bounds_pa: I
    shear_traction_bounds_pa: I
    contact_gap_bounds_m: I
    contact_jump_bounds_m: I
    bulk_normal_strain_bounds_m: I
    bulk_shear_strain_bounds_m: I
    contact_elastic_shear_strain_bounds_m: I
    bulk_energy_density_bounds_j_m2: I
    contact_energy_density_bounds_j_m2: I
    total_energy_density_bounds_j_m2: I
    previous_memory_intersections: tuple[NativeContactFieldMemoryIntersection, ...]
    unaccepted_candidate_memory_intersections: tuple[NativeContactFieldMemoryIntersection, ...]
    pieces: tuple[NativeContactFieldPieceQuery, ...]
    whole_source_constitutive_support_payload_json: bytes
    constitutive_status: str
    precision: int

    @property
    def source_expression_context(self):
        return json.loads(self.source_expression_context_payload_json)

    @property
    def source_contact_time_path(self):
        return self.history.contact_time_path

    @property
    def source_previous_memory(self):
        return self.history.previous_memory

    @property
    def source_unaccepted_candidate_memory(self):
        return self.history.unaccepted_candidate_memory

    @property
    def original_pair_history_authenticated(self):
        return True

    @property
    def finite_point_area_m2(self):
        return None

    @property
    def finite_point_work_j(self):
        return None

    @property
    def cumulative_work_density_j_m2(self):
        return None

    @property
    def candidate_mechanics_adopted(self):
        return False

    @property
    def incoming_pair_handover_authorized(self):
        return False


_BOUND_FIELDS = tuple(name for name in NativeContactFieldPieceQuery.__dataclass_fields__
                      if name.endswith("_bounds_m") or name.endswith("_bounds_pa")
                      or name.endswith("_bounds_j_m2"))


def _intersections(memory, angle):
    rows = []
    for index, piece in enumerate(memory.pieces):
        low = max(angle.lo, _dyadic(piece.low_radians))
        high = min(angle.hi, _dyadic(piece.high_radians))
        if low <= high:
            rows.append(NativeContactFieldMemoryIntersection(index, I(low, high), piece))
    if not rows or rows[0].computational_angle_bounds.lo != angle.lo:
        raise ValueError("Continuous field query lacks its unchanged original memory support")
    cursor = angle.lo
    for row in rows:
        if row.computational_angle_bounds.lo > cursor:
            raise ValueError("Continuous field query would skip an original memory piece")
        cursor = max(cursor, row.computational_angle_bounds.hi)
    if cursor != angle.hi:
        raise ValueError("Continuous field query exceeds its original memory pieces")
    return tuple(rows)


def _piece_query(trace, row, g0, dg, j0, dj, time, support_status, precision):
    angle, piece = row.computational_angle_bounds, row.original_piece
    bound = lambda polynomial: _outward(_field_bounds(polynomial, angle.lo, angle.hi, precision), precision)
    gap = _outward(_field_affine_bounds(g0, dg, angle.lo, angle.hi,
                                   time.lo, time.hi, precision), precision)
    jump = _outward(_field_affine_bounds(j0, dj, angle.lo, angle.hi,
                                    time.lo, time.hi, precision), precision)
    old_p, old_c = bound(_field(piece.plastic_slip)), bound(_field(piece.cumulative_slip))
    kn_column, kt_column = map(_dyadic, (trace.columns.normal_stiffness_pa_m,
                                       trace.columns.shear_stiffness_pa_m))
    kn_contact, kt_contact = map(_dyadic, (trace.law.normal_stiffness_pa_m,
                                         trace.law.tangential_stiffness_pa_m))
    # Keep the original binary64 series-compliance constants, rather than
    # substituting an exact rational evaluation of a different compliance law.
    kn = _dyadic(1. / (1. / trace.columns.normal_stiffness_pa_m
                       + 1. / trace.law.normal_stiffness_pa_m))
    kt = _dyadic(1. / (1. / trace.columns.shear_stiffness_pa_m
                       + 1. / trace.law.tangential_stiffness_pa_m))
    mu = _dyadic(trace.law.friction_dry
                 + (trace.law.friction_wet - trace.law.friction_dry) * trace.water)
    cohesion = _dyadic(trace.law.cohesion_pa
                       * (1. - (1. - trace.law.wet_cohesion_fraction) * trace.water))
    pressure = _outward(-kn * gap, precision)
    strength = _outward(cohesion + mu * pressure, precision)
    if support_status == ADMISSIBLE:
        # The authenticated whole-rectangle predicate proves the down-dip
        # branch, including its initial algebraic zero.  These are identities
        # of that branch and avoid dependency cancellation of j-p in traction.
        increment = _outward(jump - old_p + strength / kt, precision)
        plastic = _outward(jump + strength / kt, precision)
        cumulative = _outward(old_c - increment, precision)
        shear = -strength
        status = ADMISSIBLE
    else:
        trial = _outward(kt * (jump - old_p), precision)
        increment = _outward((_positive(trial - strength)
                              - _positive(-trial - strength)) / kt, precision)
        plastic = _outward(old_p + increment, precision)
        cumulative = _outward(old_c + _absolute(increment), precision)
        shear = _outward(kt * (jump - plastic), precision)
        status = "inviscid_dense_return_map_enclosure_without_whole_time_admission"
    contact_gap = _outward(-pressure / kn_contact, precision)
    elastic_shear = _outward(shear / kt_contact, precision)
    contact_jump = _outward(plastic + elastic_shear, precision)
    # These are exact identities of the retained series-compliance family.
    # Direct subtraction of interval anchors would lose the shared j/g
    # dependency, creating spurious strain even for a zero-strength spring.
    # Keep kt as the original binary64 lift: replacing this coefficient with
    # 1/kt_column would silently change the authenticated endpoint family.
    bulk_normal = _outward(gap * (kn / kn_contact - 1), precision)
    bulk_shear = _outward(shear * (1 / kt_contact - 1 / kt), precision)
    bulk_energy = _outward(Fraction(1, 2) * (kn_column * _square(bulk_normal)
                                            + kt_column * _square(bulk_shear)), precision)
    contact_energy = _outward(Fraction(1, 2) * (kn_contact * _square(contact_gap)
                                               + kt_contact * _square(elastic_shear)), precision)
    total_energy = _outward(bulk_energy + contact_energy, precision)
    return NativeContactFieldPieceQuery(row.piece_index, angle,
        gap, jump, old_p, old_c, increment, plastic, cumulative, pressure, shear,
        contact_gap, contact_jump, bulk_normal, bulk_shear, elastic_shear,
        bulk_energy, contact_energy, total_energy, status)


def query_native_contact_field_history(history, trace_id, computational_angle_bounds,
        time_fraction_bounds, *, expected_source_history_dto,
        source_expression_context=None, precision=256):
    """Enclose the entire original field without choosing a material or time.

    Angle and time are exact intervals used to bound expressions, not selected
    physical coordinates.  The optional expression context is retained in
    full so a material-port certificate can identify its actual implicit root.
    Every closed intersecting previous and candidate memory piece survives;
    shared boundaries retain both expressions as a conservative union.
    """
    payload = _source(history, expected_source_history_dto)
    precision = _precision(precision)
    if type(trace_id) is not str or not trace_id:
        raise ValueError("Continuous field query requires an original physical trace identity")
    path = history.contact_time_path
    position = next((index for index, trace in enumerate(path.initial_mechanics.fault_traces.traces)
                     if trace.trace_id == trace_id), None)
    if position is None:
        raise ValueError("Continuous field query cannot borrow a neighbouring trace's history")
    trace = path.initial_mechanics.fault_traces.traces[position]
    if _dyadic(trace.law.viscosity_pa_s_m):
        raise ValueError("Viscous fractional backward-Euler observations cannot supply a continuous physical field")
    extent = _dyadic(path.step.previous_memory.traces[position].extent_radians)
    angle = _interval(computational_angle_bounds, "Original computational angle", maximum=extent)
    time = _interval(time_fraction_bounds, "Original contact clock", maximum=Fraction(1))
    context = _context(source_expression_context)
    previous = _intersections(path.step.previous_memory.traces[position], angle)
    candidate = _intersections(path.step.memory.traces[position], angle)
    _, old_gap, old_jump = path.previous_anchor_fields[position]
    _, new_gap, new_jump = path.step.anchor_fields[position]
    g0, j0, g1, j1 = map(_field, (old_gap, old_jump, new_gap, new_jump))
    dg, dj = _add(g1, _scale(g0, -1)), _add(j1, _scale(j0, -1))
    supports = json.loads(history.whole_time_support_payload_json)["certificates"]
    support = next(row for row in supports if row["trace_id"] == trace_id)
    pieces = tuple(_piece_query(trace, row, g0, dg, j0, dj,
                               time, support["status"], precision) for row in previous)
    bounds = {name: _hull([getattr(piece, name) for piece in pieces]) for name in _BOUND_FIELDS}
    old_nodes = [node.previous_state for node in history.quadrature_histories
                 if node.trace_id == trace_id]
    if not old_nodes or any(node.elapsed_time_s != old_nodes[0].elapsed_time_s for node in old_nodes):
        raise ValueError("Continuous field cannot infer a local clock from inconsistent original histories")
    duration = _dyadic(history.literal_dt_s)
    status = next((piece.constitutive_status for piece in pieces
                   if piece.constitutive_status != ADMISSIBLE), ADMISSIBLE)
    return NativeContactFieldHistoryQuery(history, payload, context, trace_id, angle, time,
        I(_dyadic(old_nodes[0].elapsed_time_s)) + duration * time,
        I(_dyadic(history.accepted_before.elapsed_time_s)) + duration * time,
        I(_dyadic(float(history.origin_surface.time_myr))) + _dyadic(history.literal_dt_myr) * time,
        **bounds, previous_memory_intersections=previous,
        unaccepted_candidate_memory_intersections=candidate, pieces=pieces,
        whole_source_constitutive_support_payload_json=_encoded(support),
        constitutive_status=status, precision=precision)


def _portable(value):
    if type(value) is Fraction:
        return [value.numerator, value.denominator]
    if type(value) is I:
        return dict(lo=_portable(value.lo), hi=_portable(value.hi))
    if hasattr(type(value), "__dataclass_fields__"):
        return {name: _portable(getattr(value, name)) for name in value.__dataclass_fields__}
    if type(value) in (tuple, list):
        return [_portable(item) for item in value]
    return value


def validate_native_contact_field_history(query, *, history, expected_source_history_dto):
    if type(query) is not NativeContactFieldHistoryQuery:
        raise ValueError("Expected an immutable complete continuous contact-field query")
    repeated = query_native_contact_field_history(history, query.trace_id,
        query.computational_angle_bounds, query.time_fraction_bounds,
        expected_source_history_dto=expected_source_history_dto,
        source_expression_context=query.source_expression_context, precision=query.precision)
    if _typed_fingerprint(query) != _typed_fingerprint(repeated):
        raise ValueError("Continuous field query differs from the complete original field replay")
    return query


def native_contact_field_history_to_dict(query, *, history, expected_source_history_dto):
    validate_native_contact_field_history(query, history=history,
                                         expected_source_history_dto=expected_source_history_dto)
    body = {name: _portable(getattr(query, name))
            for name in query.__dataclass_fields__ if name not in (
                "history", "source_history_payload_json", "source_expression_context_payload_json",
                "whole_source_constitutive_support_payload_json")}
    body.update(format=FORMAT, model=MODEL, source_history=json.loads(query.source_history_payload_json),
        source_expression_context=query.source_expression_context,
        whole_source_constitutive_support=json.loads(query.whole_source_constitutive_support_payload_json),
        constitutive_path_model=CONTACT_TIME_MODEL, piece_selection_model=PIECE_SELECTION,
        energy_measure=ENERGY_MEASURE, finite_point_area_m2=None, finite_point_work_j=None,
        cumulative_work_density_j_m2=None, candidate_mechanics_adopted=False,
        incoming_pair_handover_authorized=False, original_pair_history_authenticated=True,
        geometry_expression_authenticated_by_this_component=False,
        endpoint_scope="complete retained original before and unaccepted candidate fields; dense family does not replace literal nodal endpoints")
    body["query_sha256"] = hashlib.sha256(_encoded(body)).hexdigest()
    return json.loads(_encoded(body))


def native_contact_field_history_from_dict(raw, *, history, trace_id,
        computational_angle_bounds, time_fraction_bounds, expected_source_history_dto,
        source_expression_context=None, precision=256):
    """Re-evaluate independently supplied field arguments before DTO adoption.

    A rehashed source, angle, root-window or bound does not become authority:
    every literal DTO field must match the complete external source query.
    """
    _literal_json(raw)
    query = query_native_contact_field_history(history, trace_id,
        computational_angle_bounds, time_fraction_bounds,
        expected_source_history_dto=expected_source_history_dto,
        source_expression_context=source_expression_context, precision=precision)
    if type(raw) is not dict or _encoded(raw) != _encoded(native_contact_field_history_to_dict(
            query, history=history, expected_source_history_dto=expected_source_history_dto)):
        raise ValueError("Continuous field DTO differs from full independently supplied source and arguments")
    return query
