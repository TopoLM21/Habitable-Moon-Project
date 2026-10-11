"""Unique quarter-change contact paths for one exact receiving carrier.

The receiving material ray is fixed.  Four complete, reversed point
chronologies prove the instantaneous donor-quarter changes of Qd^-1 Qr c.
Only such a change can trigger a handover; the infinitely many fresh owner
entries inside a quarter do not replace the frozen active material label.
The incoming event must additionally prove a unique positive first owner
entry.  Closed seam ancestry is resolved by its proved incoming right germ,
never by an identifier, a sampled clock, or a geometric tolerance.

This is a declared reduced correspondence law, not spatial force admission.
It selects a complete path for the supplied carrier, not all carrier angles.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction as F
import hashlib
import json

from .geometric_native_pose import FractionInterval as I, interval_sqrt
from .geometric_native_physical_history import _dyadic, _literal_json
from .geometric_native_first_entry_contact_domain_handover import _source
from .geometric_native_first_entry_interface_integral import _quarters
from .geometric_native_contact_reference_transport import _proof_key, _parameters
from .geometric_native_physical_pair_time import (
    _rotated, _axes, _cross, certify_native_quarter_carrier_coordinate,
)
from .geometric_native_contact_time_support import native_quarter_material_coordinate_bounds
from .geometric_native_point_chronology import (
    certify_native_point_chronology, native_point_chronology_to_dict,
    UnresolvedNativePointChronology,
)


FORMAT = "native-active-receiver-contact-binding-path-1"
MODEL = "fixed_receiver_quarter_exit_unique_fresh_right_germ_frozen_material"
_DERIVATIONS = {}
_AUTHENTICATED = {}


class UnresolvedNativeActiveContactBinding(ValueError):
    """A unique causal handover could not be proved within the source budgets."""


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def _fraction(value):
    return [value.numerator, value.denominator]


def _ib(value):
    return dict(lo=_fraction(value.lo), hi=_fraction(value.hi))


def _iv(values):
    return [_ib(value) for value in values]


def _carrier(value):
    if (type(value) is not tuple or len(value) != 3
            or any(type(v) not in (int, F) for v in value)):
        raise ValueError("The fixed receiver carrier must be an exact immutable Cartesian triple")
    result = tuple(F(v) for v in value)
    # Endpoints retain closed ancestry in the whole-domain registry.  This
    # single-valued path accepts open quarters, rather than choosing that
    # endpoint's physical ancestry by a sorted identifier.
    if result[0] != 0 or result[1] == 0 or result[2] == 0:
        raise UnresolvedNativeActiveContactBinding("The unique carrier path requires a nonzero open receiving quarter")
    return result


def _axis_support(normal):
    values = tuple(F(v) for v in normal)
    indices = tuple(i for i, v in enumerate(values) if v)
    if len(indices) != 1 or abs(values[indices[0]]) != 1:
        raise ValueError("The original source must retain its exact primitive axis-quarter supports")
    return indices[0], 1 if values[indices[0]] > 0 else -1


@dataclass(frozen=True, slots=True)
class NativeActiveContactBindingEvent:
    sequence: int
    old_donor_fragment_id: str
    new_donor_fragment_id: str
    old_donor_trace_id: str
    new_donor_trace_id: str
    time_fraction_bounds: I
    seam_coordinate_index: int
    incoming_sign: int
    donor_reference_coordinates_bounds: tuple
    fresh_owner_depth_derivative_bounds: I
    incoming_chronology: object
    incoming_chronology_event_sequence: int
    incoming_plane_event_sequence: int
    incoming_occupancy_event_sequence: int
    consumed_binding_id: str
    active_binding_id: str

    @property
    def first_entry_proved(self):
        return True

    @property
    def material_reentry(self):
        return False

    @property
    def frozen_reference_expression(self):
        return "Q_d(first_quarter_change)^T Q_r(first_quarter_change) unit(original_receiver_c)"


@dataclass(frozen=True, slots=True)
class NativeActiveContactBindingSegment:
    sequence: int
    start_time_fraction_bounds: I
    end_time_fraction_bounds: I
    donor_fragment_id: str
    donor_trace_id: str
    frozen_reference_coordinates_bounds: tuple
    anchor_event_sequence: int | None
    active_binding_id: str

    @property
    def donor_material_label_frozen(self):
        return True


@dataclass(frozen=True, slots=True)
class NativeActiveContactBindingPath:
    history: object
    law: object
    receiver_fragment_id: str
    receiver_trace_id: str
    receiver_reference_coordinates: tuple
    receiver_unit_direction_bounds: tuple
    receiver_native_angle_bounds: I
    receiver_computational_angle_bounds: I
    donor_chronologies: tuple
    events: tuple
    segments: tuple
    precision: int
    order: int
    max_depth: int
    max_cells: int
    source_history_payload_json: bytes
    path_sha256: str

    @property
    def initial_donor_fragment_id(self):
        return self.segments[0].donor_fragment_id

    @property
    def initial_donor_trace_id(self):
        return self.segments[0].donor_trace_id

    @property
    def physical_pair_admission_authorized(self):
        return False

    @property
    def candidate_mechanics_adopted(self):
        return False

    @property
    def accepted_native_force_intervals(self):
        return 0


def _quarter_events(chronologies):
    # Each +/- axis support is the same analytic scalar up to its explicit
    # orientation.  Root equality comes from this exact source identity,
    # not from the overlap of two unrelated numerical root enclosures.
    groups = {}
    for identity, chronology in chronologies:
        for event in chronology.plane_events:
            index, orientation = _axis_support(chronology.receiver_polygon.supports[event.support_index].normal)
            if index == 0 or event.time_fraction.hi == 0:
                continue
            if event.kind not in ("entry", "exit"):
                raise UnresolvedNativeActiveContactBinding("A quarter tangency is not a handover trigger")
            sign = orientation*(1 if event.kind == "entry" else -1)
            key = (index, event.time_fraction.lo, event.time_fraction.hi)
            groups.setdefault(key, []).append((identity, sign, event))
    result = []
    for (index, low, high), rows in groups.items():
        if (len(rows) != 4 or len({row[0] for row in rows}) != 4
                or len({row[1] for row in rows}) != 1):
            raise UnresolvedNativeActiveContactBinding("Shared quarter roots lack a complete four-bank right-germ identity")
        result.append((I(low, high), index, rows[0][1]))
    result.sort(key=lambda value: (value[0].lo, value[0].hi))
    if any(a[0].hi >= b[0].lo for a, b in zip(result, result[1:])):
        raise UnresolvedNativeActiveContactBinding("Quarter changes lack a proved strict chronological order")
    return tuple(result)


def _fresh_derivative(unit_q, donor_axis, receiver_axis, time, precision, order):
    domain = I(0, time.hi)
    normal = _rotated((F(1), F(0), F(0)), receiver_axis, domain, precision, order)
    moved = _rotated(unit_q, donor_axis, domain, precision, order)
    value = sum(((a-b)*component for a, b, component in
                 zip(donor_axis, receiver_axis, _cross(normal, moved))), I(0))
    if value.lo <= 0:
        raise UnresolvedNativeActiveContactBinding("Incoming material lacks a proved unique positive first owner entry")
    return value


def certify_native_active_contact_binding(history, receiver_fragment_id,
        receiver_reference_coordinates, *, law, expected_source_history_dto,
        precision=256, order=32, max_depth=192, max_cells=4096):
    """Select every causal quarter change for one exact fixed receiver ray.

    There is no minimum fresh event inside an unchanged quarter: fresh owner
    entries are continuous there.  The explicitly declared exit trigger is
    what permits a material label to remain frozen between handovers.
    """
    payload = _source(history, expected_source_history_dto, law)
    if type(receiver_fragment_id) is not str:
        raise ValueError("The receiving fragment must be an original physical source identifier")
    c = _carrier(receiver_reference_coordinates)
    for value, low, high, name in ((precision, 96, 1024, "precision"),
            (order, 16, 48, "order"), (max_depth, 1, 768, "depth"),
            (max_cells, 1, 65536, "cells")):
        if type(value) is not int or not low <= value <= high:
            raise ValueError("Active contact "+name+" must be a bounded literal integer")
    key = _proof_key((history, law, receiver_fragment_id, c, precision, order, max_depth, max_cells))
    if key in _DERIVATIONS:
        return _DERIVATIONS[key]
    source_rows = _quarters(history)
    receiving = tuple(row for row in source_rows if row[0].receiver_fragment_id == receiver_fragment_id)
    if len(receiving) != 1:
        raise ValueError("The fixed receiving ray must name exactly one original contact quarter")
    receiver_bank, receiver_trace, _, receiver_supports = receiving[0]
    if any(sum(n*v for n, v in zip(normal, c)) < 0 for normal in receiver_supports):
        raise ValueError("The exact ray lies outside its declared original receiving quarter")
    fragments = {row.fragment_id: row for row in history.material_state.journal.source_surface.fragments}
    omega = dict(history.actual_candidate.omega_rad_per_myr)
    ad, ar = _axes(history.omega_records, history.dt_myr,
                   receiver_bank.subducting_plate, receiver_bank.overriding_plate)
    norm = interval_sqrt(I(sum((v*v for v in c), F(0))), precision+16)
    unit_c = tuple(I(v)/norm for v in c)
    angle = certify_native_quarter_carrier_coordinate(tuple(map(_dyadic, receiver_trace.start)),
        tuple(map(_dyadic, receiver_trace.end)), tuple(I(v) for v in c), precision=precision)
    computational = native_quarter_material_coordinate_bounds(history.contact_time_path,
        receiver_trace.trace_id, angle, precision=precision)
    chronologies = tuple((row[0].donor_fragment_id, certify_native_point_chronology(
        fragments[receiver_fragment_id].geometry, fragments[row[0].donor_fragment_id].geometry,
        omega[receiver_bank.overriding_plate], omega[receiver_bank.subducting_plate],
        float(history.dt_myr), c, precision=precision, order=order,
        max_depth=max_depth, max_cells=max_cells)) for row in source_rows)
    by_id = dict(chronologies)
    quarter_rows = {}
    for row in source_rows:
        signs = dict(_axis_support(normal) for normal in row[2])
        if signs.get(0) != 1 or 1 not in signs or 2 not in signs:
            raise ValueError("Original donor ancestry must be the complete positive-X hemisphere quarters")
        quarter_rows[(signs[1], signs[2])] = row
    signs = [1 if c[k] > 0 else -1 for k in (1, 2)]
    original_row = quarter_rows[tuple(signs)]
    if original_row[0].donor_fragment_id != receiver_bank.donor_fragment_id:
        raise ValueError("The initial frozen attachment differs from the original receiving bank")
    prefix = "active-carrier:"+history.original_checkpoint_sha256+":"+hashlib.sha256(
        _encoded([receiver_fragment_id, [_fraction(v) for v in c]])).hexdigest()
    active_id = prefix+":binding:0"
    events, segments = [], []
    current_row, current_q, start, anchor = original_row, unit_c, I(0), None
    for time, coordinate, incoming_sign in _quarter_events(chronologies):
        if time.lo <= 0 or time.hi >= 1:
            raise UnresolvedNativeActiveContactBinding("A positive interior quarter-change root must be strictly enclosed")
        old_sign = signs[coordinate-1]
        if old_sign != -incoming_sign:
            raise UnresolvedNativeActiveContactBinding("The incoming germ does not consume the unique preceding quarter")
        signs[coordinate-1] = incoming_sign
        incoming_row = quarter_rows[tuple(signs)]
        bank, trace, _, _ = incoming_row
        _parameters(receiver_trace, trace.columns, trace.law, trace.water)
        q = list(_rotated(_rotated(unit_c, ar, time, precision, order),
                          tuple(-v for v in ad), time, precision, order))
        if not q[coordinate].lo <= 0 <= q[coordinate].hi:
            raise ValueError("The complete seam chronology contradicts the frozen event coordinate")
        q[coordinate] = I(0)  # Authenticated exact axis-plane event identity.
        q = tuple(q)
        other = 3-coordinate
        if q[0].lo <= 0:
            raise UnresolvedNativeActiveContactBinding("The quarter change has no strictly incoming original donor bank")
        oriented_other = signs[other-1]*q[other]
        if oriented_other.lo <= 0:
            raise UnresolvedNativeActiveContactBinding("The new bank's other material support remains unresolved")
        derivative = _fresh_derivative(q, ad, ar, time, precision, order)
        chronology = by_id[bank.donor_fragment_id]
        plane_events = tuple(event for event in chronology.plane_events
            if event.time_fraction == time and event.kind == "entry"
            and _axis_support(chronology.receiver_polygon.supports[event.support_index].normal)[0] == coordinate)
        occupancy = tuple(event for event in chronology.occupancy_events
                          if event.time_fraction == time and event.kind == "entry")
        if len(plane_events) != 1 or len(occupancy) != 1:
            raise UnresolvedNativeActiveContactBinding("The incoming germ is not the unique complete donor occupancy entry")
        next_id = prefix+":binding:"+str(len(events)+1)
        segments.append(NativeActiveContactBindingSegment(len(segments), start, time,
            current_row[0].donor_fragment_id, current_row[1].trace_id, current_q, anchor, active_id))
        events.append(NativeActiveContactBindingEvent(len(events)+1,
            current_row[0].donor_fragment_id, bank.donor_fragment_id,
            current_row[1].trace_id, trace.trace_id, time, coordinate, incoming_sign,
            q, derivative, chronology, occupancy[0].sequence, plane_events[0].sequence, occupancy[0].sequence,
            active_id, next_id))
        current_row, current_q, start, anchor, active_id = incoming_row, q, time, len(events), next_id
    segments.append(NativeActiveContactBindingSegment(len(segments), start, I(1),
        current_row[0].donor_fragment_id, current_row[1].trace_id, current_q, anchor, active_id))
    path = NativeActiveContactBindingPath(history, law, receiver_fragment_id,
        receiver_trace.trace_id, c, unit_c, angle, computational, chronologies,
        tuple(events), tuple(segments), precision, order, max_depth, max_cells, payload, "")
    body = _encoded(_body(path))
    path = replace(path, path_sha256=hashlib.sha256(body).hexdigest())
    if len(_DERIVATIONS) >= 8:
        _DERIVATIONS.clear()
    _DERIVATIONS[key] = path
    if len(_AUTHENTICATED) >= 16:
        _AUTHENTICATED.clear()
    _AUTHENTICATED[_proof_key(path)] = body
    return path


def _body(path):
    return dict(format=FORMAT, model=MODEL,
        source_history=json.loads(path.source_history_payload_json),
        explicit_law=dict(model=path.law.model, provenance=path.law.provenance),
        receiver_fragment_id=path.receiver_fragment_id, receiver_trace_id=path.receiver_trace_id,
        receiver_reference_coordinates=[_fraction(v) for v in path.receiver_reference_coordinates],
        receiver_unit_direction_bounds=_iv(path.receiver_unit_direction_bounds),
        receiver_native_angle_bounds=_ib(path.receiver_native_angle_bounds),
        receiver_computational_angle_bounds=_ib(path.receiver_computational_angle_bounds),
        precision=path.precision, order=path.order, max_depth=path.max_depth, max_cells=path.max_cells,
        donor_chronologies=[dict(donor_fragment_id=identity,
            chronology=native_point_chronology_to_dict(value)) for identity, value in path.donor_chronologies],
        events=[dict(sequence=e.sequence, old_donor_fragment_id=e.old_donor_fragment_id,
            new_donor_fragment_id=e.new_donor_fragment_id, old_donor_trace_id=e.old_donor_trace_id,
            new_donor_trace_id=e.new_donor_trace_id, time_fraction_bounds=_ib(e.time_fraction_bounds),
            seam_coordinate_index=e.seam_coordinate_index, incoming_sign=e.incoming_sign,
            donor_reference_coordinates_bounds=_iv(e.donor_reference_coordinates_bounds),
            fresh_owner_depth_derivative_bounds=_ib(e.fresh_owner_depth_derivative_bounds),
            incoming_chronology_ref=e.new_donor_fragment_id,
            incoming_chronology_event_sequence=e.incoming_chronology_event_sequence,
            incoming_plane_event_sequence=e.incoming_plane_event_sequence,
            incoming_occupancy_event_sequence=e.incoming_occupancy_event_sequence,
            consumed_binding_id=e.consumed_binding_id, active_binding_id=e.active_binding_id,
            frozen_reference_expression=e.frozen_reference_expression,
            exact_seam_incidence_from_source_chronology=True, first_entry_proved=True,
            same_material_reentry=False) for e in path.events],
        segments=[dict(sequence=s.sequence, start_time_fraction_bounds=_ib(s.start_time_fraction_bounds),
            end_time_fraction_bounds=_ib(s.end_time_fraction_bounds), donor_fragment_id=s.donor_fragment_id,
            donor_trace_id=s.donor_trace_id, frozen_reference_coordinates_bounds=_iv(s.frozen_reference_coordinates_bounds),
            anchor_event_sequence=s.anchor_event_sequence, active_binding_id=s.active_binding_id,
            donor_material_label_frozen=True) for s in path.segments],
        selection_trigger="exit of instantaneous original donor quarter; unique positive incoming germ; fresh first owner entry",
        fresh_proof="D(q*,s)=0 by full carrier identity and D_v>0 on [0,s] with positive right germ; each distinct first clock has distinct original material",
        preceding_binding_consumed_once=True, reentry_allocates_again=False,
        receiving_carrier_reprojected_after_switch=False, root_midpoint_selected=False,
        interface_or_work_weight_assigned_to_point=False,
        all_receiver_angles_scheduled=False, physical_pair_admission_authorized=False,
        candidate_mechanics_adopted=False, accepted_native_force_intervals=0)


def validate_native_active_contact_binding(path):
    if type(path) is not NativeActiveContactBindingPath:
        raise ValueError("Expected the complete immutable active receiving-carrier path")
    key = _proof_key(path)
    if key not in _AUTHENTICATED:
        expected = certify_native_active_contact_binding(path.history, path.receiver_fragment_id,
            path.receiver_reference_coordinates, law=path.law,
            expected_source_history_dto=json.loads(path.source_history_payload_json),
            precision=path.precision, order=path.order, max_depth=path.max_depth, max_cells=path.max_cells)
        if _proof_key(expected) != key:
            raise ValueError("Active carrier path differs from the complete source chronology replay")
        _AUTHENTICATED[key] = _encoded(_body(expected))
    return path


def native_active_contact_binding_to_dict(path):
    path = validate_native_active_contact_binding(path)
    body = json.loads(_AUTHENTICATED[_proof_key(path)])
    return dict(body, path_sha256=path.path_sha256)


def native_active_contact_binding_from_dict(raw, *, expected_history, expected_law,
        expected_source_history_dto, expected_receiver_fragment_id,
        expected_receiver_reference_coordinates):
    if type(raw) is not dict or raw.get("format") != FORMAT:
        raise ValueError("Expected a complete active-carrier restart DTO")
    _literal_json(raw)
    try:
        result = certify_native_active_contact_binding(expected_history,
            expected_receiver_fragment_id, expected_receiver_reference_coordinates,
            law=expected_law, expected_source_history_dto=expected_source_history_dto,
            precision=raw["precision"], order=raw["order"],
            max_depth=raw["max_depth"], max_cells=raw["max_cells"])
    except KeyError as error:
        raise ValueError("The active-carrier restart omits its complete proof budgets") from error
    if _encoded(raw) != _encoded(native_active_contact_binding_to_dict(result)):
        raise ValueError("Active-carrier restart differs from its independently supplied physical source and ray")
    return result
