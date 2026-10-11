"""Authentic CURRENT receiver memory and a contact-only scalar reference.

This observes the literal endpoint ``step.memory`` at the genuine receiver
label.  Dense return-map queries built from previous memory are a different
expression and are not substituted for the stored current polynomials.
Neither a two-dimensional memory density, first-entry attachment, old rod
prestrain, new area allocation nor acceptance is inferred from this receipt.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from fractions import Fraction as F
from functools import lru_cache
import hashlib
import json

from .geometric_native_pose import FractionInterval as I, _rounded_interval
from .geometric_native_common_reference import validate_native_common_reference
from .geometric_native_normal_support import load_native_normal_support_source
from .geometric_native_common_contact_ports import (
    _payload, _encoded, validate_native_common_contact_port_response,
)
from .geometric_native_contact_field_history import _field_bounds, _square

FORMAT = "native-common-current-contact-state-observation-1"
MODEL = "literal_current_endpoint_memory_contact_only_scalar_calibration"
SCOPE = "SOURCE_CURRENT_STATE_OBSERVATION_NO_FACE_MEMORY_ALLOCATION"
_REGISTRY = {}


def _digest(value):
    return hashlib.sha256(_encoded(value)).hexdigest()


def _seal(value):
    return _digest({f.name: getattr(value, f.name) for f in fields(value)
                    if f.name != "binding_sha256"})


def _literal(value):
    if type(value) is not float:
        raise ValueError("The current contact source requires its literal binary64 coefficients")
    return F.from_float(value)


def _round(value, precision):
    return _rounded_interval(value, precision + 16)


def _hull(values):
    return I(min(x.lo for x in values), max(x.hi for x in values))


@lru_cache(maxsize=8)
def _source(snapshot):
    source = load_native_normal_support_source()
    if source.source_snapshot_sha256 != snapshot:
        raise ValueError("The contact state belongs to another complete physical source")
    before = json.loads(dict(source.complete_source_dto_json)["before"])
    history = before["original_physical_history"]
    path = history["contact_time_path"]
    states = before["snapshot"]["states"]
    if (before["snapshot"]["spatial_memory"] != path["step"]["memory"]
            or len(states) != 16 or len(history["quadrature_histories"]) != 16
            or dict(states) != {row["point_id"]: row["unaccepted_candidate_state"]
                                for row in history["quadrature_histories"]}):
        raise ValueError("The current endpoint memory or original sixteen states differ")
    return before, history, path


@dataclass(frozen=True, slots=True)
class NativeCommonCurrentMemoryPiece:
    piece_index: int
    computational_angle_bounds: I
    original_piece_json: str
    plastic_slip_bounds_m: I
    cumulative_slip_bounds_m: I


@dataclass(frozen=True, slots=True)
class NativeCommonContactStateObservation:
    source_snapshot_sha256: str
    source_epoch_sha256: str
    reference_snapshot_sha256: str
    cut_snapshot_sha256: str
    geometry_bounds_sha256: str
    port_binding_sha256: str
    receiving_trace_id: str
    source_history_binding_sha256: str
    source_current_memory_sha256: str
    source_current_atom_states_sha256: str
    source_trace_json: str
    source_endpoint_anchor_fields_json: str
    source_current_memory_trace_json: str
    source_current_node_epoch: str
    computational_angle_bounds: I
    source_time_fraction_bounds: I
    source_memory_elapsed_time_s: F
    source_memory_stage_count: int
    source_native_elapsed_time_s: F
    source_contact_elapsed_time_s: F
    source_contact_step_count: int
    memory_piece_intersections: tuple
    normal_stiffness_pa_m: F
    tangential_stiffness_pa_m: F
    original_column_normal_stiffness_pa_m: F
    original_column_shear_stiffness_pa_m: F
    original_series_normal_stiffness_pa_m: F
    original_series_shear_stiffness_pa_m: F
    source_interface_height_m: F
    source_water: F
    free_gap_bounds_m: I
    free_jump_bounds_m: I
    plastic_slip_bounds_m: I
    cumulative_slip_bounds_m: I
    normal_pressure_bounds_pa: I
    shear_traction_bounds_pa: I
    contact_reference_gap_bounds_m: I
    contact_reference_jump_bounds_m: I
    contact_reference_energy_density_bounds_j_m2: I
    precision: int
    binding_sha256: str
    source_relaxed_friction_free_branch: bool = True
    compressed_source_observation: bool = True
    literal_current_memory_used: bool = True
    current_memory_replaced_with_previous_or_dense_query: bool = False
    source_node_states_assigned_to_new_cut: bool = False
    face_depth_memory_or_measure_transfer_certified: bool = False
    old_rod_prestrain_or_energy_transferred: bool = False
    source_history_advanced: bool = False
    contact_energy_adopted_on_new_face: bool = False
    force_equilibrium_certified: bool = False
    physical_material_admission_authorized: bool = False


def observe_native_common_contact_state(reference, port_response, *, precision=160):
    """Observe all closed CURRENT pieces at an authentic geometric label.

    Pressure uses the original binary64 series coefficient and final free
    anchor; p/c use actual stored current memory, including its rounding.
    Zero shear is the source friction-free relaxed identity, not an interval
    subtraction of independently enclosed j and p.  No source clock advances.
    """
    if type(precision) is not int or not 96 <= precision <= 512:
        raise ValueError("Current-state precision must be an integer from 96 to 512")
    reference = validate_native_common_reference(reference)
    response = validate_native_common_contact_port_response(port_response, reference=reference)
    cut, geometry = response.port.cut, response.geometry
    if cut.receiving_trace_id is None or cut.receiver_material_kind != "original_live":
        raise ValueError("The true HOT curve has no authenticated source receiving contact law")
    if not geometry.genuine_query_support_certified:
        raise ValueError("An unresolved material band cannot observe contact memory")
    angle = geometry.receiving_computational_angle_bounds_radians
    if type(angle) is not I:
        raise ValueError("The genuine cold receiver lacks its original computational label")
    before, history, path = _source(reference.source_snapshot_sha256)
    traces = path["initial_mechanics"]["fault_traces"]["traces"]
    matches = [row for row in traces if row["trace_id"] == cut.receiving_trace_id]
    if len(matches) != 1:
        raise ValueError("A current cut cannot borrow a donor or neighbouring family")
    trace = matches[0]
    law, columns = trace["law"], trace["columns"]
    if any(_literal(law[k]) for k in ("friction_dry", "friction_wet", "cohesion_pa", "viscosity_pa_s_m")):
        raise ValueError("Only the authentic compressive friction-free source branch is supported")
    kn, kt = _literal(law["normal_stiffness_pa_m"]), _literal(law["tangential_stiffness_pa_m"])
    if min(kn, kt) <= 0:
        raise ValueError("The source contact stiffness must be positive")
    # Preserve the original PROPERTY arithmetic before lifting its result.
    knc = 1. / (columns["sub_length_m"] / columns["sub_normal_modulus_pa"]
                + columns["over_length_m"] / columns["over_normal_modulus_pa"])
    ktc = 1. / (columns["sub_length_m"] / columns["sub_shear_modulus_pa"]
                + columns["over_length_m"] / columns["over_shear_modulus_pa"])
    kns, kts = _literal(1. / (1. / knc + 1. / float(kn))), _literal(1. / (1. / ktc + 1. / float(kt)))
    current = next(row for row in path["step"]["memory"]["traces"] if row["trace_id"] == cut.receiving_trace_id)
    extent = _literal(current["extent_radians"])
    if not 0 <= angle.lo <= angle.hi <= extent:
        raise ValueError("The receiving label leaves its literal current-memory extent")
    pieces = []
    for index, piece in enumerate(current["pieces"]):
        lo, hi = max(angle.lo, _literal(piece["low_radians"])), min(angle.hi, _literal(piece["high_radians"]))
        if lo > hi:
            continue
        p, c = (tuple(_literal(x) for x in piece[k]["coefficients"])
                for k in ("plastic_slip", "cumulative_slip"))
        if len(p) != 5 or len(c) != 5:
            raise ValueError("Every source current memory piece needs all five literal coefficients")
        pieces.append(NativeCommonCurrentMemoryPiece(index, I(lo, hi), _encoded(piece).decode(),
            _round(_field_bounds(p, lo, hi, precision), precision),
            _round(_field_bounds(c, lo, hi, precision), precision)))
    if not pieces:
        raise ValueError("No actual current memory piece covers the receiving material label")
    coverage = sorted((p.computational_angle_bounds.lo, p.computational_angle_bounds.hi) for p in pieces)
    reached = angle.lo
    for lo, hi in coverage:
        if lo > reached:
            raise ValueError("The current memory has an uncovered material interval")
        reached = max(reached, hi)
    if reached != angle.hi:
        raise ValueError("The entire receiving interval is not covered by current memory")
    anchors = next(row for row in path["step"]["anchor_fields"] if row[0] == cut.receiving_trace_id)
    gap, jump = (_round(_field_bounds(tuple(_literal(x) for x in v["coefficients"]), angle.lo, angle.hi, precision), precision)
                 for v in anchors[1:])
    if gap.hi >= 0:
        raise ValueError("The complete current source query must remain strictly compressed")
    pressure = _round(-kns * gap, precision)
    plastic = _hull([p.plastic_slip_bounds_m for p in pieces])
    cumulative = _hull([p.cumulative_slip_bounds_m for p in pieces])
    if cumulative.lo < 0:
        raise ValueError("Current cumulative slip lacks a nonnegative source enclosure")
    states = [row[1] for row in before["snapshot"]["states"] if row[0].startswith(cut.receiving_trace_id + ":")]
    if len(states) != 4 or len({(s["elapsed_time_s"], s["step_count"]) for s in states}) != 1:
        raise ValueError("The original receiving atoms do not have one preserved contact clock")
    if any(s["traction_shear_pa"] != 0 for s in states):
        raise ValueError("The current source branch is not the literal zero-shear family")
    memory = path["step"]["memory"]
    value = NativeCommonContactStateObservation(
        reference.source_snapshot_sha256, reference.source_epoch_sha256, reference.reference_snapshot_sha256,
        cut.cut_snapshot_sha256, geometry.bounds_snapshot_sha256, response.port.binding_sha256,
        cut.receiving_trace_id, history["binding_sha256"], _digest(memory), _digest(before["snapshot"]["states"]),
        _encoded(trace).decode(), _encoded(anchors).decode(), _encoded(current).decode(), "unaccepted_candidate",
        angle, I(1), _literal(memory["elapsed_time_s"]), memory["stage_count"],
        _literal(before["snapshot"]["elapsed_time_s"]), _literal(states[0]["elapsed_time_s"]), states[0]["step_count"],
        tuple(pieces), kn, kt, _literal(knc), _literal(ktc), kns, kts, _literal(trace["contact_height_m"]),
        _literal(trace["water"]), gap, jump, plastic, cumulative, pressure, I(0),
        _round(-pressure / kn, precision), plastic, _round(_square(pressure) / (2 * kn), precision), precision, "")
    value = replace(value, binding_sha256=_seal(value))
    _REGISTRY[value.binding_sha256] = _encoded(value)
    return value


def validate_native_common_contact_state_observation(value, *, reference=None, port_response=None):
    if (type(value) is not NativeCommonContactStateObservation or value.binding_sha256 != _seal(value)
            or _REGISTRY.get(value.binding_sha256) != _encoded(value)):
        raise ValueError("An altered or re-sealed contact state lacks genuine source replay")
    if reference is not None:
        reference = validate_native_common_reference(reference)
        if value.reference_snapshot_sha256 != reference.reference_snapshot_sha256:
            raise ValueError("Contact reference and current source observation disagree")
    if port_response is not None:
        response = validate_native_common_contact_port_response(port_response, reference=reference)
        if (response.port.binding_sha256 != value.port_binding_sha256
                or response.geometry.bounds_snapshot_sha256 != value.geometry_bounds_sha256):
            raise ValueError("A source state cannot be attached to another physical material query")
    return value


def native_common_contact_state_observation_to_dict(value):
    value = validate_native_common_contact_state_observation(value)
    return dict(format=FORMAT, model=MODEL, scope=SCOPE, **_payload(value))


def native_common_contact_state_observation_from_dict(raw, *, reference, port_response):
    if type(raw) is not dict or type(raw.get("precision")) is not int:
        raise ValueError("A complete contact state observation needs its source precision")
    replay = observe_native_common_contact_state(reference, port_response, precision=raw["precision"])
    if _encoded(raw) != _encoded(native_common_contact_state_observation_to_dict(replay)):
        raise ValueError("Serialized current contact state differs from source, labels, memory or clocks")
    return replay
