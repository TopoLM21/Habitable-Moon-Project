"""Bind first material entry to the preserved original physical bank evidence.

An incoming bulk label and a saved spring tip are different material objects.
This binding keeps both original trace pairs, certifies the incoming relative
coordinate on its first root window, and tests original carrier incidence.
It does not assign a saved tip's bending displacement, plastic field or nodal
work to incoming material.  A new material correspondence needs a handover law.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
import hashlib
import json

from .geometric_native_first_entry_material import (
    NativeFirstEntryMaterialPointChronology,
    validate_native_first_entry_material_point,
    evaluate_native_first_entry_material_point,
)
from .geometric_native_pose import FractionInterval as I
from .geometric_native_physical_pair_time import (
    certify_native_relative_entry_coordinates,
    certify_native_quarter_carrier_coordinate,
)
from .geometric_native_physical_history import _typed_fingerprint


FORMAT = "native-first-entry-original-physical-bank-binding-1"
MODEL = "preserved_original_pairs_incoming_material_handover_unresolved"
_CACHE = {}


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode("utf-8")


def _fraction(value):
    if type(value) is not Fraction:
        raise ValueError("Physical entry coordinates require literal Fractions")
    return [value.numerator, value.denominator]


def _interval(value):
    if type(value) is not I:
        raise ValueError("Physical entry requires an exact interval")
    return dict(lo=_fraction(value.lo), hi=_fraction(value.hi))


def _dot(first, second):
    return sum((x*y for x, y in zip(first, second)), Fraction(0))


def _cross(first, second):
    return (first[1]*second[2]-first[2]*second[1],
            first[2]*second[0]-first[0]*second[2],
            first[0]*second[1]-first[1]*second[0])


def _literal_vector(values):
    return tuple(Fraction.from_float(value) for value in values)


@dataclass(frozen=True, slots=True)
class NativeFirstEntryPhysicalBankRow:
    receiver_fragment_id: str
    donor_trace_id: str
    receiver_trace_id: str
    desired_fragment_pair: tuple
    original_donor_trace_pair: tuple
    original_receiver_trace_pair: tuple
    same_original_fragment_pair: bool
    incoming_original_label_on_donor_carrier: bool
    donor_carrier_plane_value: Fraction
    receiver_carrier_plane: tuple
    relative_entry_coordinates: tuple
    receiver_native_angle_bounds: I | None
    receiver_coordinate_status: str
    original_donor_point_ids: tuple
    original_receiver_point_ids: tuple
    handover_status: str = "unresolved_material_handover"


@dataclass(frozen=True, slots=True)
class NativeFirstEntryPhysicalBinding:
    chronology: NativeFirstEntryMaterialPointChronology
    history: object
    time_fraction_bounds: I | None
    first_receiver_ancestry_status: str
    rows: tuple[NativeFirstEntryPhysicalBankRow, ...]
    status: str
    precision: int
    order: int
    binding_sha256: str

    @property
    def physical_pair_admission_authorized(self):
        return False


def _body(value):
    from .geometric_native_physical_history import native_physical_history_binding_to_dict
    history_raw = native_physical_history_binding_to_dict(value.history)
    return dict(format=FORMAT, model=MODEL,
        material_state_sha256=value.chronology.state.state_sha256,
        point_chronology_sha256=value.chronology.point_chronology_sha256,
        physical_history_sha256=history_raw["binding_sha256"],
        original_parent_id=value.chronology.original_parent_id,
        original_point=[_fraction(x) for x in value.chronology.original_point],
        time_fraction_bounds=None if value.time_fraction_bounds is None else _interval(value.time_fraction_bounds),
        first_receiver_ancestry_status=value.first_receiver_ancestry_status,
        status=value.status, precision=value.precision, order=value.order,
        rows=[dict(receiver_fragment_id=row.receiver_fragment_id,
            donor_trace_id=row.donor_trace_id, receiver_trace_id=row.receiver_trace_id,
            desired_fragment_pair=list(row.desired_fragment_pair),
            original_donor_trace_pair=list(row.original_donor_trace_pair),
            original_receiver_trace_pair=list(row.original_receiver_trace_pair),
            same_original_fragment_pair=row.same_original_fragment_pair,
            incoming_original_label_on_donor_carrier=row.incoming_original_label_on_donor_carrier,
            donor_carrier_plane_value=_fraction(row.donor_carrier_plane_value),
            receiver_carrier_plane=[_fraction(x) for x in row.receiver_carrier_plane],
            relative_entry_coordinates=[_interval(x) for x in row.relative_entry_coordinates],
            receiver_native_angle_bounds=None if row.receiver_native_angle_bounds is None else _interval(row.receiver_native_angle_bounds),
            receiver_coordinate_status=row.receiver_coordinate_status,
            original_donor_point_ids=list(row.original_donor_point_ids),
            original_receiver_point_ids=list(row.original_receiver_point_ids),
            handover_status=row.handover_status) for row in value.rows],
        first_root_midpoint_used=False,
        original_nodal_history_assigned_to_incoming_material=False,
        finite_material_reservoir_assigned_to_point=False,
        nodal_work_distributed_to_incoming_material=False,
        candidate_mechanics_adopted=False,
        physical_pair_admission_authorized=False,
        accepted_loaded_native_intervals=0, accepted_source_work_j=0)


def _remember(value, encoded):
    if len(_CACHE) >= 64:
        _CACHE.clear()
    _CACHE[_typed_fingerprint(value)] = encoded


def certify_native_first_entry_physical_binding(chronology, history, *,
                                               precision=256, order=32):
    """Retain exact first-entry timing and prove which old banks it touches.

    Carrier angle conversion is conditional on the independently proved owner
    plane root and receiver ancestry.  It never proves a new spring identity.
    Off-carrier original labels and changed fragment pairs require handover.
    """
    from .geometric_native_physical_history import validate_native_physical_history_binding
    chronology = validate_native_first_entry_material_point(chronology)
    history = validate_native_physical_history_binding(history)
    if _typed_fingerprint(chronology.state) != _typed_fingerprint(history.material_state):
        raise ValueError("Physical bank binding needs the actual authenticated material state")
    if (type(precision) is not int or not 32 <= precision <= 1024
            or type(order) is not int or not 8 <= order <= 64):
        raise ValueError("Physical bank binding needs bounded literal proof budgets")
    query = evaluate_native_first_entry_material_point(chronology, Fraction(1))
    window = query.first_entry_time_fraction
    rows = []
    if window is not None:
        traces = {trace.trace_id: trace for trace in history.initial_mechanics.fault_traces.traces}
        bindings = {binding.trace_id: binding for binding in chronology.state.trace_bindings}
        point_ids = {identity: tuple(point.point_id for point in history.original_points
            if point.trace_id == identity) for identity in traces}
        for donor_id in query.original_trace_ids:
            donor_binding, donor = bindings[donor_id], traces[donor_id]
            donor_start, donor_end = _literal_vector(donor.start), _literal_vector(donor.end)
            donor_plane = _cross(donor_start, donor_end)
            plane_value = _dot(donor_plane, chronology.original_point)
            on_carrier = (plane_value == 0
                and _dot(donor_start, chronology.original_point) >= 0
                and _dot(donor_end, chronology.original_point) >= 0)
            for receiver_id in query.receiver_trace_ids:
                receiver_binding, receiver = bindings[receiver_id], traces[receiver_id]
                desired = (chronology.original_parent_id, receiver_binding.receiver_fragment_id)
                donor_pair = (donor_binding.donor_fragment_id, donor_binding.receiver_fragment_id)
                receiver_pair = (receiver_binding.donor_fragment_id, receiver_binding.receiver_fragment_id)
                relative = certify_native_relative_entry_coordinates(chronology.original_point,
                    donor_binding.subducting_plate, receiver_binding.overriding_plate,
                    history.omega_records, history.dt_myr, window,
                    precision=precision, order=order)
                start, end = _literal_vector(receiver.start), _literal_vector(receiver.end)
                plane = _cross(start, end)
                angle, coordinate_status = None, "unresolved_carrier_coordinate"
                # The existing first-entry proof authenticates its X=0 owner
                # root.  A containing interval alone would not authenticate
                # incidence on any other carrier plane.
                axis_owner_plane = (plane[0] != 0 and plane[1:] == (Fraction(0), Fraction(0)))
                if query.first_entry_receiver_ancestry_status == "proved" and axis_owner_plane:
                    incidence = _dot(plane, relative)
                    if not incidence.lo <= 0 <= incidence.hi:
                        raise ValueError("Physical incoming coordinates contradict the certified owner root")
                    try:
                        angle = certify_native_quarter_carrier_coordinate(start, end, relative,
                            precision=precision)
                    except ValueError:
                        # Ambiguous quadrants remain unresolved; the full
                        # root window and coordinates are retained for replay.
                        pass
                    else:
                        coordinate_status = "first_owner_root_receiver_carrier_coordinate_proved"
                rows.append(NativeFirstEntryPhysicalBankRow(receiver_binding.receiver_fragment_id,
                    donor_id, receiver_id, desired, donor_pair, receiver_pair,
                    desired == donor_pair == receiver_pair, on_carrier, plane_value,
                    plane, relative, angle, coordinate_status, point_ids[donor_id], point_ids[receiver_id]))
        if not rows:
            raise ValueError("First entry lost the original physical donor or receiver references")
    status = "no_first_entry" if window is None else "unresolved_material_handover"
    unsealed = NativeFirstEntryPhysicalBinding(chronology, history, window,
        query.first_entry_receiver_ancestry_status, tuple(rows), status, precision, order, "")
    body = _encoded(_body(unsealed))
    result = replace(unsealed, binding_sha256=hashlib.sha256(body).hexdigest())
    _remember(result, body)
    return result


def validate_native_first_entry_physical_binding(value):
    if type(value) is not NativeFirstEntryPhysicalBinding:
        raise ValueError("Expected an immutable original physical bank binding")
    fingerprint = _typed_fingerprint(value)
    if fingerprint not in _CACHE:
        rebuilt = certify_native_first_entry_physical_binding(value.chronology, value.history,
            precision=value.precision, order=value.order)
        if _typed_fingerprint(value) != _typed_fingerprint(rebuilt):
            raise ValueError("Physical bank binding differs from complete source replay")
    return value


def native_first_entry_physical_binding_to_dict(value):
    value = validate_native_first_entry_physical_binding(value)
    body = json.loads(_CACHE[_typed_fingerprint(value)])
    body["binding_sha256"] = value.binding_sha256
    return body


def native_first_entry_physical_binding_from_dict(raw, *, chronology, history):
    if type(raw) is not dict:
        raise ValueError("Physical bank binding requires an object")
    _literal_json(raw)
    rebuilt = certify_native_first_entry_physical_binding(chronology, history,
        precision=raw.get("precision"), order=raw.get("order"))
    if _encoded(raw) != _encoded(native_first_entry_physical_binding_to_dict(rebuilt)):
        raise ValueError("Saved physical bank binding differs from complete source replay")
    return rebuilt


def _literal_json(value):
    kind = type(value)
    if kind in (str, int, bool, type(None)):
        return
    if kind is list:
        for item in value:
            _literal_json(item)
        return
    if kind is dict and all(type(key) is str for key in value):
        for item in value.values():
            _literal_json(item)
        return
    raise ValueError("Physical bank DTO requires literal canonical JSON field types")
