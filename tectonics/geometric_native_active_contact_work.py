"""Frozen material contact segments and their genuine conservative work.

The original receiver owns one continuous contact state.  A proved change of
its donor bank changes a reference, not that state.  Each segment follows its
unchanged material label with both complete SO(3) rotations and the original
donor's Ritz displacement.  The actual source used here is inviscid and has
zero friction and cohesion.  Work is its reaction one-form, including the
signed residual of the retained binary64 series stiffness.  No quadrature
area, force acceptance or diagnostic affine source-work replacement follows.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from collections import OrderedDict
from fractions import Fraction as F
from functools import lru_cache
from math import factorial
import hashlib
import json

from .geometric_native_pose import (
    FractionInterval as I, _rounded_interval, interval_atan, interval_sqrt,
)
from .geometric_native_physical_history import (
    _dyadic, _literal_json, native_physical_history_binding_to_dict,
    validate_native_physical_history_binding,
)
from .geometric_native_contact_reference_transport import _proof_key, _parameters
from .geometric_native_contact_field_history import (
    NativeContactFieldHistoryQuery, _portable, _square,
    query_native_contact_field_history, native_contact_field_history_to_dict,
)
from .geometric_native_contact_time_support import (
    ADMISSIBLE, native_quarter_material_coordinate_bounds,
)
from .geometric_native_physical_pair_time import (
    _rotated, _cross, _dot,
    certify_native_quarter_carrier_coordinate,
)
from .geometric_native_contact_events import _polynomial_bounds
from .geometric_native_point_chronology import (
    validate_native_point_chronology, native_point_chronology_to_dict,
)


REFERENCE_FORMAT = "native-active-frozen-contact-segment-reference-1"
WORK_FORMAT = "native-active-frozen-contact-segment-work-1"
MODEL = "actual_full_SO3_frozen_material_Ritz_contact_zero_strength_inviscid"
FIRST_LAW = "actual_work = bulk_energy_increment + contact_energy_increment + zero_dissipation + signed_literal_compliance_residual"
_AUTHENTICATED_REFERENCES = OrderedDict()
_AUTHENTICATED_QUERIES = OrderedDict()
_REFERENCES = OrderedDict()
_QUERIES = OrderedDict()


def _remember(cache, key, value, limit):
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > limit:
        cache.popitem(last=False)


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def _out(value, precision):
    return _rounded_interval(value if type(value) is I else I(value), precision+16)


def _clock(value, name):
    if type(value) is not I or not 0 <= value.lo <= value.hi <= 1:
        raise ValueError(name+" requires an exact interval in the original clock")
    return value


def _source(history, expected):
    validate_native_physical_history_binding(history)
    if type(expected) is not dict:
        raise ValueError("Active contact work needs the entire independent physical source DTO")
    _literal_json(expected)
    payload = _encoded(expected)
    if payload != _encoded(native_physical_history_binding_to_dict(history)):
        raise ValueError("Active contact work differs from its complete original source")
    return payload


def _direction(value):
    if (type(value) is not tuple or len(value) != 3
            or any(type(x) is not F for x in value) or not any(value)
            or value[0] != 0):
        raise ValueError("Receiver contact needs an exact nonzero homogeneous X=0 carrier label")
    return value


def _trace(history, identity):
    matches = tuple(t for t in history.initial_mechanics.fault_traces.traces
                    if t.trace_id == identity)
    if len(matches) != 1:
        raise ValueError("Active contact cannot borrow an unidentified original trace")
    return matches[0]


def _bank(history, identity):
    return next(b for b in history.material_state.trace_bindings if b.trace_id == identity)


def _unit(values, precision):
    norm = interval_sqrt(I(sum((x*x for x in values), F(0))), precision+24)
    return tuple(_out(I(x)/norm, precision) for x in values)


def _tip(history, identity, index, time):
    old = next(r[index] for r in history.accepted_before.transverse_displacement_m
               if r[0] == identity)
    new = next(r[index] for r in history.actual_candidate.transverse_displacement_m
               if r[0] == identity)
    start, end = _dyadic(old), _dyadic(new)
    return I(start)+(end-start)*time, end-start


def _family(trace):
    if (_dyadic(trace.law.viscosity_pa_s_m)
            or _dyadic(trace.law.friction_dry) or _dyadic(trace.law.friction_wet)
            or _dyadic(trace.law.cohesion_pa)):
        raise ValueError("Active conservative contact supports only the explicit zero-strength inviscid source")
    kb, kc = map(_dyadic, (trace.columns.normal_stiffness_pa_m,
                          trace.law.normal_stiffness_pa_m))
    kn = _dyadic(1./(1./trace.columns.normal_stiffness_pa_m
                     + 1./trace.law.normal_stiffness_pa_m))
    return kn, kb*(kn/kc-1)**2, kn*kn/kc


@dataclass(frozen=True, slots=True)
class NativeActiveContactSegmentReference:
    history: object
    receiver_trace_id: str
    donor_trace_id: str
    receiver_original_direction: tuple
    switch_time_fraction_bounds: I
    switch_chronology: object | None
    switch_event_sequence: int | None
    root_proof_context_payload_json: bytes
    initial_state: object
    source_field_query: NativeContactFieldHistoryQuery
    receiver_unit_direction_bounds: tuple
    receiver_native_angle_bounds: I
    computational_angle_bounds: I
    frozen_original_material_coordinates: tuple
    seam_coordinate_index: int | None
    frozen_shape_bounds: I
    inward_distance_bounds_m: I
    radius_m: F
    source_gap_at_switch_bounds_m: I
    source_jump_at_switch_bounds_m: I
    source_plastic_slip_at_switch_bounds_m: I
    source_cumulative_slip_at_switch_bounds_m: I
    initial_bulk_energy_density_bounds_j_m2: I
    initial_contact_energy_density_bounds_j_m2: I
    initial_total_energy_density_bounds_j_m2: I
    revision: int
    precision: int
    order: int
    source_history_payload_json: bytes
    reference_sha256: str

    @property
    def physical_pair_admission_authorized(self):
        return False

    @property
    def candidate_mechanics_adopted(self):
        return False


@dataclass(frozen=True, slots=True)
class NativeActiveContactSegmentWorkQuery:
    reference: NativeActiveContactSegmentReference
    current_time_fraction_bounds: I
    at_reference_event: bool
    receiver_relative_direction_change_bounds: tuple
    projected_gap_change_bounds_m: I
    projected_jump_change_bounds_m: I
    free_gap_bounds_m: I
    free_jump_bounds_m: I
    plastic_slip_bounds_m: I
    cumulative_slip_bounds_m: I
    normal_pressure_bounds_pa: I
    shear_traction_bounds_pa: I
    bulk_energy_density_bounds_j_m2: I
    contact_energy_density_bounds_j_m2: I
    total_energy_density_bounds_j_m2: I
    free_gap_rate_bounds_m_per_fraction: I
    free_jump_rate_bounds_m_per_fraction: I
    plastic_slip_rate_bounds_m_per_fraction: I
    cumulative_slip_rate_bounds_m_per_fraction: I
    whole_segment_gap_bounds_m: I
    whole_segment_jump_rate_bounds_m_per_fraction: I
    source_work_density_bounds_j_m2: I
    friction_dissipation_density_bounds_j_m2: I
    bulk_energy_increment_density_bounds_j_m2: I
    contact_energy_increment_density_bounds_j_m2: I
    total_energy_increment_density_bounds_j_m2: I
    compliance_balance_residual_density_bounds_j_m2: I
    accumulated_segment_work_density_bounds_j_m2: I
    accumulated_segment_dissipation_density_bounds_j_m2: I
    accumulated_segment_bulk_energy_increment_density_bounds_j_m2: I
    accumulated_segment_contact_energy_increment_density_bounds_j_m2: I
    accumulated_segment_total_energy_increment_density_bounds_j_m2: I
    accumulated_segment_compliance_balance_residual_density_bounds_j_m2: I
    constitutive_status: str
    work_sha256: str

    @property
    def finite_point_area_m2(self):
        return None

    @property
    def finite_point_work_j(self):
        return None

    @property
    def candidate_mechanics_adopted(self):
        return False

    @property
    def physical_pair_admission_authorized(self):
        return False


def _event(history, receiver_trace_id, donor_trace_id, direction, switch,
           chronology, sequence):
    if switch == I(0):
        if chronology is not None or sequence is not None:
            raise ValueError("The original zero-clock reference has no material switch event")
        donor = _bank(history, donor_trace_id)
        polygon = next(f.geometry for f in history.material_state.journal.source_surface.fragments
                       if f.fragment_id == donor.donor_fragment_id)
        if any(_dot(p.normal, direction) < 0 for p in polygon.supports):
            raise ValueError("Initial frozen carrier label is outside its declared original donor")
        return None
    chronology = validate_native_point_chronology(chronology)
    receiver, donor = _bank(history, receiver_trace_id), _bank(history, donor_trace_id)
    polygons = {f.fragment_id: f.geometry for f in history.material_state.journal.source_surface.fragments}
    omegas = dict(history.omega_records)
    if (_proof_key(chronology.donor_polygon) != _proof_key(polygons[receiver.receiver_fragment_id])
            or _proof_key(chronology.receiver_polygon) != _proof_key(polygons[donor.donor_fragment_id])
            or chronology.donor_omega != omegas[receiver.overriding_plate]
            or chronology.receiver_omega != omegas[donor.subducting_plate]
            or chronology.dt_myr != history.dt_myr or chronology.point != direction):
        raise ValueError("Frozen reference root does not authenticate its actual original carrier and owners")
    if type(sequence) is not int:
        raise ValueError("A material switch requires its explicit incoming occupancy event")
    matches = tuple(e for e in chronology.occupancy_events if e.sequence == sequence)
    if len(matches) != 1 or matches[0].kind != "entry" or matches[0].time_fraction != switch:
        raise ValueError("Frozen reference clock is not the exact proved incoming donor event")
    event = matches[0]
    if len(event.support_indices) != 1:
        raise ValueError("A frozen reference requires one independently proved quarter seam")
    support = chronology.receiver_polygon.supports[event.support_index]
    indexes = tuple(i for i, x in enumerate(support.normal) if x)
    if len(indexes) != 1 or indexes[0] == 0:
        raise ValueError("A donor-quarter switch must authenticate its exact Y or Z seam")
    return indexes[0]


def make_native_active_contact_segment_reference(history, receiver_trace_id,
        donor_trace_id, receiver_original_direction, switch_time_fraction_bounds, *,
        switch_chronology=None, switch_event_sequence=None, root_proof_context,
        initial_state=None, expected_source_history_dto, precision=256, order=32):
    """Freeze a real material reference and inherit one continuous state.

    Only the original zero-clock segment can seed itself from the original
    dense source.  Every positive-clock switch inherits an already evolved
    segment at that same implicit event; its diagnostic source clock cannot
    replace the transported state.
    """
    payload = _source(history, expected_source_history_dto)
    if (type(precision) is not int or not 96 <= precision <= 1024
            or type(order) is not int or not 16 <= order <= 64):
        raise ValueError("Active contact requires bounded literal proof budgets")
    direction = _direction(receiver_original_direction)
    switch = _clock(switch_time_fraction_bounds, "Frozen reference clock")
    if type(root_proof_context) is not dict or not root_proof_context:
        raise ValueError("Frozen reference requires the complete independent selector context")
    _literal_json(root_proof_context)
    context_payload = _encoded(root_proof_context)
    seam = _event(history, receiver_trace_id, donor_trace_id, direction, switch,
                  switch_chronology, switch_event_sequence)
    key = _proof_key((history, receiver_trace_id, donor_trace_id, direction, switch,
        switch_chronology, switch_event_sequence, context_payload, initial_state, precision, order))
    if key in _REFERENCES:
        result = _REFERENCES[key]
        _REFERENCES.move_to_end(key)
        _remember(_AUTHENTICATED_REFERENCES, _proof_key(result),
                  _encoded(_reference_body(result)), 64)
        return result
    receiver, donor = _trace(history, receiver_trace_id), _trace(history, donor_trace_id)
    _parameters(receiver, donor.columns, donor.law, donor.water)
    _family(receiver)
    nodes = tuple(n for n in history.quadrature_histories if n.trace_id == receiver_trace_id)
    if not nodes or any(n.original_frame != nodes[0].original_frame
            or n.original_point.horizontal != nodes[0].original_point.horizontal for n in nodes):
        raise ValueError("The original receiving contact frame is inconsistent")
    native_angle = certify_native_quarter_carrier_coordinate(
        tuple(map(_dyadic, receiver.start)), tuple(map(_dyadic, receiver.end)),
        tuple(I(x) for x in direction), precision=precision)
    angle = native_quarter_material_coordinate_bounds(history.contact_time_path,
        receiver_trace_id, native_angle, precision=precision)
    unit = _unit(direction, precision)
    bank_d, bank_r = _bank(history, donor_trace_id), _bank(history, receiver_trace_id)
    axes = {owner: tuple(x*history.dt_myr for x in omega)
            for owner, omega in history.omega_records}
    q = _rotated(_rotated(unit, axes[bank_r.overriding_plate], switch, precision, order),
        tuple(-x for x in axes[bank_d.subducting_plate]), switch, precision, order)
    if switch == I(0):
        q = unit
    if seam is not None:
        if not q[seam].lo <= 0 <= q[seam].hi:
            raise ValueError("Frozen direction contradicts its authenticated seam root")
        q = tuple(I(0) if i == seam else x for i, x in enumerate(q))
    if not 0 <= q[0].lo <= q[0].hi < 1:
        raise ValueError("Frozen incoming material has no proved positive original donor beam coordinate")
    # q is an exact unit expression.  The Cartesian enclosure is not a new
    # independently chosen point, so sqrt(1-qx²) preserves that identity.
    inward = interval_atan(q[0]/interval_sqrt(1-_square(q[0]), precision+24), precision+16)
    radius = _dyadic(history.origin_surface.radius_km)*1000
    width = _dyadic(next(p.sub.width_m for p in history.initial_mechanics.bending_parameters
                        if p.trace_id == donor_trace_id))
    distance, z = _out(radius*inward, precision), _out(radius*inward/width, precision)
    if not 0 <= z.lo <= z.hi < 1:
        raise ValueError("Frozen material lies outside its own original donor Ritz strip")
    shape = _out(I((1-z.hi)**2*(z.hi+2)/2, (1-z.lo)**2*(z.lo+2)/2), precision)
    if type(initial_state) is NativeActiveContactSegmentWorkQuery:
        validate_native_active_contact_segment_work(initial_state)
        before = initial_state.reference
        if (switch == I(0) or initial_state.current_time_fraction_bounds != switch
                or _proof_key(before.history) != _proof_key(history)
                or before.receiver_trace_id != receiver_trace_id
                or before.receiver_original_direction != direction):
            raise ValueError("A switched contact must inherit the same evolved receiver at its exact event")
        source, revision = before.source_field_query, before.revision+1
    else:
        if switch != I(0):
            raise ValueError("A positive-clock switch cannot reset to the original diagnostic source field")
        source = query_native_contact_field_history(history, receiver_trace_id, angle, I(0),
            expected_source_history_dto=expected_source_history_dto,
            source_expression_context=root_proof_context, precision=precision)
        if initial_state is not None and (type(initial_state) is not NativeContactFieldHistoryQuery
                or _proof_key(initial_state) != _proof_key(source)):
            raise ValueError("Original segment seed differs from the complete unchanged source at zero")
        initial_state, revision = source, 0
    if source.constitutive_status != ADMISSIBLE:
        raise ValueError("Initial active state needs its original proved inviscid source branch")
    if (initial_state.plastic_slip_bounds_m != initial_state.free_jump_bounds_m
            or initial_state.shear_traction_bounds_pa != I(0)):
        raise ValueError("Zero-strength active state needs its unchanged relaxed shear and plastic reference")
    values = dict(source_gap_at_switch_bounds_m=initial_state.free_gap_bounds_m,
        source_jump_at_switch_bounds_m=initial_state.free_jump_bounds_m,
        source_plastic_slip_at_switch_bounds_m=initial_state.plastic_slip_bounds_m,
        source_cumulative_slip_at_switch_bounds_m=initial_state.cumulative_slip_bounds_m,
        initial_bulk_energy_density_bounds_j_m2=initial_state.bulk_energy_density_bounds_j_m2,
        initial_contact_energy_density_bounds_j_m2=initial_state.contact_energy_density_bounds_j_m2,
        initial_total_energy_density_bounds_j_m2=initial_state.total_energy_density_bounds_j_m2)
    result = NativeActiveContactSegmentReference(history, receiver_trace_id, donor_trace_id,
        direction, switch, switch_chronology, switch_event_sequence, context_payload,
        initial_state, source, unit, native_angle, angle, q, seam, shape, distance, radius,
        **values, revision=revision, precision=precision, order=order,
        source_history_payload_json=payload, reference_sha256="")
    body = _encoded(_reference_body(result))
    result = replace(result, reference_sha256=hashlib.sha256(body).hexdigest())
    _remember(_AUTHENTICATED_REFERENCES, _proof_key(result), body, 64)
    _remember(_REFERENCES, key, result, 32)
    return result


@lru_cache(maxsize=128)
def _joint_relative_coefficients(vector, donor_axis, receiver_axis, precision, order):
    """Merge both rotation generators before bounding their shared clock.

    Independently boxing exp(-Ar*delta) and exp(Ad*delta) would lose the
    cancellation in (Ad-Ar)*delta on a whole time interval.  The coefficient
    of delta**n is sum_{i+j=n} (-Ar)**j Ad**i vector/(j! i!).
    All components of both original generators remain in this expression.
    """
    values = tuple(x if type(x) is I else I(x) for x in vector)
    donor_terms = [values]
    for degree in range(1, order+1):
        donor_terms.append(tuple(_out(x/degree, precision)
            for x in _cross(donor_axis, donor_terms[-1])))
    coefficients = [[I(0), I(0), I(0)] for _ in range(order+1)]
    negative_receiver = tuple(-x for x in receiver_axis)
    for donor_degree, term in enumerate(donor_terms):
        for receiver_degree in range(order-donor_degree+1):
            degree = donor_degree+receiver_degree
            if degree >= 2:
                coefficients[degree] = [_out(a+b, precision)
                    for a, b in zip(coefficients[degree], term)]
            if degree < order:
                term = tuple(_out(x/(receiver_degree+1), precision)
                    for x in _cross(negative_receiver, term))
    # Preserve the exact common-clock cancellation even when the vector is
    # an implicit-root interval expression with nonzero enclosure widths.
    coefficients[1] = list(_cross(
        tuple(a-b for a, b in zip(donor_axis, receiver_axis)), values))
    return tuple(tuple(_out(x, precision) for x in row) for row in coefficients)


def _joint_relative_change(vector, donor_axis, receiver_axis, delta, precision, order):
    """Enclose [exp(-Ar*delta) exp(Ad*delta)-I] vector jointly."""
    if not delta.hi:
        return (I(0),)*3
    coefficients = _joint_relative_coefficients(
        tuple(vector), tuple(donor_axis), tuple(receiver_axis), precision, order)
    bound = sum(map(abs, donor_axis), F(0))+sum(map(abs, receiver_axis), F(0))
    scale = sum((max(abs(x.lo), abs(x.hi)) if type(x) is I else abs(x)
                 for x in vector), F(0))
    size = bound*delta.hi
    ratio = size/F(order+2)
    if ratio >= 1:
        raise ValueError("Joint frozen rotation exceeds its explicit Taylor remainder budget")
    tail = _out(scale*size**(order+1)/factorial(order+1)/(1-ratio), precision).hi
    return tuple(_polynomial_bounds(tuple(row[index] for row in coefficients),
        delta, tail, precision+8) for index in range(3))


def _geometry(reference, current, *, same_event=False):
    history, precision, order = reference.history, reference.precision, reference.order
    s, c = reference.switch_time_fraction_bounds, reference.receiver_unit_direction_bounds
    d, r = _bank(history, reference.donor_trace_id), _bank(history, reference.receiver_trace_id)
    axes = {owner: tuple(x*history.dt_myr for x in omega)
            for owner, omega in history.omega_records}
    ad, ar = axes[d.subducting_plate], axes[r.overriding_plate]
    delta = I(0) if same_event else I(max(F(0), current.lo-s.hi), current.hi-s.lo)
    at_switch = _rotated(c, ar, s, precision, order)
    inner = _joint_relative_change(at_switch, ad, ar, delta, precision, order)
    changes = _rotated(inner, tuple(-x for x in ar), s, precision, order)
    if same_event:
        changes = (I(0),)*3
    relative = tuple(_out(x+y, precision) for x, y in zip(c, changes))
    nodes = tuple(n for n in history.quadrature_histories if n.trace_id == reference.receiver_trace_id)
    sine, cosine = map(_dyadic, nodes[0].original_frame)
    horizontal = tuple(map(_dyadic, nodes[0].original_point.horizontal))
    zh, zc = _out(_dot(horizontal, changes), precision), _out(_dot(c, changes), precision)
    ws, rate_s = _tip(history, reference.donor_trace_id, 1, current)
    wo, rate_o = _tip(history, reference.receiver_trace_id, 2, current)
    phi = reference.frozen_shape_bounds
    radial = reference.radius_m+phi*ws
    delta_ws, delta_wo = rate_s*delta, rate_o*delta
    gap_change = _out(cosine*(delta_wo-phi*delta_ws)-radial*(sine*zh+cosine*zc), precision)
    jump_change = _out(-sine*(delta_wo-phi*delta_ws)-radial*(cosine*zh-sine*zc), precision)
    # d/du [Qr(u)^T Qd(u)q*] = Qr(u)^T[(ad-ar) x Qd(u)q*].
    donor_world = _rotated(relative, ar, current, precision, order)
    derivative = _rotated(_cross(tuple(x-y for x, y in zip(ad, ar)), donor_world),
        tuple(-x for x in ar), current, precision, order)
    dh, dc = _out(_dot(horizontal, derivative), precision), _out(_dot(c, derivative), precision)
    gap_rate = _out(cosine*(rate_o-phi*rate_s)
        -phi*rate_s*(sine*zh+cosine*zc)-radial*(sine*dh+cosine*dc), precision)
    jump_rate = _out(-sine*(rate_o-phi*rate_s)
        -phi*rate_s*(cosine*zh-sine*zc)-radial*(cosine*dh-sine*dc), precision)
    return changes, gap_change, jump_change, gap_rate, jump_rate


def query_native_active_contact_segment_work(reference, current_time_fraction_bounds, *,
        expected_source_history_dto, at_reference_event=False):
    """Evaluate an evolved frozen segment without replaying an old source clock."""
    validate_native_active_contact_segment_reference(reference)
    _source(reference.history, expected_source_history_dto)
    current = _clock(current_time_fraction_bounds, "Actual segment query clock")
    if type(at_reference_event) is not bool:
        raise ValueError("The same-reference event identity needs an explicit literal boolean")
    if at_reference_event:
        if current != reference.switch_time_fraction_bounds:
            raise ValueError("A common event identity cannot refer to a different clock")
    elif current.lo < reference.switch_time_fraction_bounds.hi:
        raise ValueError("An evolved segment query must lie after its complete frozen-event window")
    key = _proof_key((reference, current, at_reference_event))
    if key in _QUERIES:
        result = _QUERIES[key]
        _QUERIES.move_to_end(key)
        _remember(_AUTHENTICATED_QUERIES, _proof_key(result), _encoded(_work_body(result)), 128)
        return result
    precision = reference.precision
    changes, dg, dj, gap_rate, jump_rate = _geometry(reference, current,
                                                   same_event=at_reference_event)
    whole_clock = I(reference.switch_time_fraction_bounds.lo, current.hi)
    _, whole_dg, _, _, whole_jump_rate = _geometry(reference, whole_clock,
                                                  same_event=at_reference_event)
    whole_gap = _out(reference.source_gap_at_switch_bounds_m+whole_dg, precision)
    if whole_gap.hi > 0 or whole_jump_rate.hi > 0:
        raise ValueError("The complete frozen segment does not prove compressive monotone down-dip evolution")
    kn, ab, ac = _family(_trace(reference.history, reference.receiver_trace_id))
    gap = _out(reference.source_gap_at_switch_bounds_m+dg, precision)
    jump = _out(reference.source_jump_at_switch_bounds_m+dj, precision)
    cumulative = _out(reference.source_cumulative_slip_at_switch_bounds_m-dj, precision)
    potential_change = _out(dg*(2*reference.source_gap_at_switch_bounds_m+dg)/2, precision)
    amounts = tuple(_out(coefficient*potential_change, precision)
                    for coefficient in (kn, F(0), ab, ac, ab+ac, kn-ab-ac))
    bulk = _out(reference.initial_bulk_energy_density_bounds_j_m2+amounts[2], precision)
    contact = _out(reference.initial_contact_energy_density_bounds_j_m2+amounts[3], precision)
    total = _out(reference.initial_total_energy_density_bounds_j_m2+amounts[4], precision)
    if at_reference_event:
        dg, dj, potential_change = I(0), I(0), I(0)
        gap, jump = reference.source_gap_at_switch_bounds_m, reference.source_jump_at_switch_bounds_m
        cumulative = reference.source_cumulative_slip_at_switch_bounds_m
        bulk, contact, total = (reference.initial_bulk_energy_density_bounds_j_m2,
            reference.initial_contact_energy_density_bounds_j_m2,
            reference.initial_total_energy_density_bounds_j_m2)
        amounts = (I(0),)*6
    pressure = (reference.initial_state.normal_pressure_bounds_pa if at_reference_event
                else _out(-kn*gap, precision))
    predecessor = reference.initial_state
    if type(predecessor) is NativeActiveContactSegmentWorkQuery:
        previous = tuple(getattr(predecessor, name) for name in _ACCUMULATED_FIELDS)
    else:
        previous = (I(0),)*6
    accumulated = previous if at_reference_event else tuple(
        _out(a+b, precision) for a, b in zip(previous, amounts))
    result = NativeActiveContactSegmentWorkQuery(reference, current, at_reference_event,
        changes, dg, dj, gap, jump, jump, cumulative, pressure, I(0),
        bulk, contact, total, gap_rate, jump_rate, jump_rate, -jump_rate,
        whole_gap, whole_jump_rate, *amounts, *accumulated, ADMISSIBLE, "")
    body = _encoded(_work_body(result))
    result = replace(result, work_sha256=hashlib.sha256(body).hexdigest())
    _remember(_AUTHENTICATED_QUERIES, _proof_key(result), body, 128)
    _remember(_QUERIES, key, result, 128)
    return result


_ACCUMULATED_FIELDS = (
    "accumulated_segment_work_density_bounds_j_m2",
    "accumulated_segment_dissipation_density_bounds_j_m2",
    "accumulated_segment_bulk_energy_increment_density_bounds_j_m2",
    "accumulated_segment_contact_energy_increment_density_bounds_j_m2",
    "accumulated_segment_total_energy_increment_density_bounds_j_m2",
    "accumulated_segment_compliance_balance_residual_density_bounds_j_m2",
)


def _reference_body(value):
    source = value.source_field_query
    is_previous = type(value.initial_state) is NativeActiveContactSegmentWorkQuery
    state_sha = value.initial_state.work_sha256 if is_previous else None
    return dict(format=REFERENCE_FORMAT, model=MODEL,
        complete_original_source_history_sha256=hashlib.sha256(value.source_history_payload_json).hexdigest(),
        receiver_trace_id=value.receiver_trace_id, donor_trace_id=value.donor_trace_id,
        receiver_original_direction=_portable(value.receiver_original_direction),
        switch_time_fraction_bounds=_portable(value.switch_time_fraction_bounds),
        switch_chronology_sha256=None if value.switch_chronology is None
            else native_point_chronology_to_dict(value.switch_chronology)["payload_sha256"],
        switch_event_sequence=value.switch_event_sequence,
        root_proof_context=json.loads(value.root_proof_context_payload_json),
        initial_state_kind="validated_previous_actual_segment" if is_previous else "original_dense_source_at_zero",
        initial_state_sha256=state_sha,
        original_source_field_reference=dict(trace_id=source.trace_id,
            computational_angle_bounds=_portable(source.computational_angle_bounds),
            time_fraction_bounds=_portable(source.time_fraction_bounds),
            source_expression_context=source.source_expression_context,
            contact_time_path_sha256=source.history.contact_time_path.path_sha256),
        **{name: _portable(getattr(value, name)) for name in (
            "receiver_unit_direction_bounds", "receiver_native_angle_bounds", "computational_angle_bounds",
            "frozen_original_material_coordinates", "seam_coordinate_index", "frozen_shape_bounds",
            "inward_distance_bounds_m", "radius_m", "source_gap_at_switch_bounds_m",
            "source_jump_at_switch_bounds_m", "source_plastic_slip_at_switch_bounds_m",
            "source_cumulative_slip_at_switch_bounds_m", "initial_bulk_energy_density_bounds_j_m2",
            "initial_contact_energy_density_bounds_j_m2", "initial_total_energy_density_bounds_j_m2",
            "revision", "precision", "order")},
        source_expression_clock_replayed_after_reference=False,
        frozen_material_label_reprojected_during_segment=False,
        initial_source_work_density_inferred_from_literal_atoms=False,
        switch_additional_work_density_j_m2=[0, 1], switch_energy_change_density_j_m2=[0, 1],
        no_finite_point_area_or_work=True, candidate_mechanics_adopted=False,
        physical_pair_admission_authorized=False)


def _work_body(value):
    fields = {name: _portable(getattr(value, name))
        for name in value.__dataclass_fields__ if name not in ("reference", "work_sha256")}
    return dict(format=WORK_FORMAT, model=MODEL, reference_sha256=value.reference.reference_sha256,
        reference=native_active_contact_segment_reference_to_dict(value.reference),
        **fields, first_law_identity=FIRST_LAW,
        work_measure="joules_per_square_metre_of_once_owned_original_receiver_interface",
        finite_point_area_m2=None, finite_point_work_j=None,
        original_literal_area_energy_work_atoms_unchanged=True,
        original_affine_diagnostic_work_summed_with_actual_segment_work=False,
        original_mechanical_candidate_work_replaced=False,
        full_saved_rotation_axes_preserved=True, candidate_mechanics_adopted=False,
        physical_pair_admission_authorized=False, accepted_loaded_native_intervals=0)


def validate_native_active_contact_segment_reference(value):
    if type(value) is not NativeActiveContactSegmentReference:
        raise ValueError("Frozen segment reference was not derived from its complete authenticated sources")
    key = _proof_key(value)
    if key not in _AUTHENTICATED_REFERENCES:
        rebuilt = make_native_active_contact_segment_reference(value.history,
            value.receiver_trace_id, value.donor_trace_id, value.receiver_original_direction,
            value.switch_time_fraction_bounds, switch_chronology=value.switch_chronology,
            switch_event_sequence=value.switch_event_sequence,
            root_proof_context=json.loads(value.root_proof_context_payload_json),
            initial_state=value.initial_state,
            expected_source_history_dto=json.loads(value.source_history_payload_json),
            precision=value.precision, order=value.order)
        if _proof_key(rebuilt) != key:
            raise ValueError("Frozen reference differs from its complete independently reconstructed arguments")
    _AUTHENTICATED_REFERENCES.move_to_end(key)
    return value


def validate_native_active_contact_segment_work(value, *, expected_source_history_dto=None):
    if type(value) is not NativeActiveContactSegmentWorkQuery:
        raise ValueError("Active segment work was not derived from its complete authenticated reference")
    key = _proof_key(value)
    if key not in _AUTHENTICATED_QUERIES:
        rebuilt = query_native_active_contact_segment_work(value.reference,
            value.current_time_fraction_bounds,
            expected_source_history_dto=json.loads(value.reference.source_history_payload_json),
            at_reference_event=value.at_reference_event)
        if _proof_key(rebuilt) != key:
            raise ValueError("Active work differs from its full independently reconstructed reference")
    _AUTHENTICATED_QUERIES.move_to_end(key)
    if expected_source_history_dto is not None:
        _source(value.reference.history, expected_source_history_dto)
    return value


def native_active_contact_segment_reference_to_dict(value):
    validate_native_active_contact_segment_reference(value)
    raw = json.loads(_AUTHENTICATED_REFERENCES[_proof_key(value)])
    raw["reference_sha256"] = value.reference_sha256
    return raw


def native_active_contact_segment_reference_from_dict(raw, *, expected_history,
        expected_source_history_dto, expected_receiver_trace_id, expected_donor_trace_id,
        expected_receiver_original_direction, expected_switch_time_fraction_bounds,
        expected_switch_chronology=None, expected_switch_event_sequence=None,
        expected_root_proof_context, expected_initial_state=None, precision=256, order=32):
    if type(raw) is not dict:
        raise ValueError("Frozen segment reference DTO requires an object")
    _literal_json(raw)
    rebuilt = make_native_active_contact_segment_reference(expected_history,
        expected_receiver_trace_id, expected_donor_trace_id, expected_receiver_original_direction,
        expected_switch_time_fraction_bounds, switch_chronology=expected_switch_chronology,
        switch_event_sequence=expected_switch_event_sequence,
        root_proof_context=expected_root_proof_context, initial_state=expected_initial_state,
        expected_source_history_dto=expected_source_history_dto, precision=precision, order=order)
    if _encoded(raw) != _encoded(native_active_contact_segment_reference_to_dict(rebuilt)):
        raise ValueError("Frozen reference DTO differs from its full independently authenticated reconstruction")
    return rebuilt


def native_active_contact_segment_work_to_dict(value, *, expected_source_history_dto=None):
    validate_native_active_contact_segment_work(value, expected_source_history_dto=expected_source_history_dto)
    raw = json.loads(_AUTHENTICATED_QUERIES[_proof_key(value)])
    raw["work_sha256"] = value.work_sha256
    return raw


def native_active_contact_segment_work_from_dict(raw, *, expected_reference,
        expected_source_history_dto, expected_current_time_fraction_bounds,
        expected_at_reference_event=False):
    if type(raw) is not dict:
        raise ValueError("Active segment work DTO requires an object")
    _literal_json(raw)
    rebuilt = query_native_active_contact_segment_work(expected_reference,
        expected_current_time_fraction_bounds, expected_source_history_dto=expected_source_history_dto,
        at_reference_event=expected_at_reference_event)
    if _encoded(raw) != _encoded(native_active_contact_segment_work_to_dict(rebuilt)):
        raise ValueError("Active work DTO differs from its full independently authenticated reconstruction")
    return rebuilt
