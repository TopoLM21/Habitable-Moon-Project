"""Source-bound frozen mechanical predictions with exact finite material maps.

A predictor advances the existing bending/contact law once. Its Euler increment
then moves material labels over that same physical interval by exact SO(3).
The moved geometry remains an unaccepted candidate: spatial material pairing,
admission and changing-source work have not been solved. The original surface,
source and passive archive remain the reference of every force/work receipt.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import hashlib
import json
import math
from numbers import Integral

import numpy as np

from .basal_coupling import PrescribedBasalParameters
from .geometric_basal_loading import (
    assemble_geometric_basal_load, validate_geometric_basal_load,
)
from .geometric_bending_contact_dynamics import (
    BendingContactState, BendingContactStep, advance_bending_contact_dynamics,
    bending_contact_dynamics_from_dict, bending_contact_dynamics_to_dict,
    validate_bending_contact_dynamics_state,
)
from .geometric_boundary import (
    GeometricBoundaryState, boundary_from_dict, boundary_to_dict, surface_digest,
)
from .geometric_boundary_io import _validate_pair
from .geometric_contact_dynamics import _encode, _number
from .geometric_fault_motion_support import (
    continuous_bank_pair_separation_m, validate_reference_bending_bank_support,
)
from .geometric_fault_transport import (
    FaultMaterialAtlas, advance_fault_material_atlas,
    fault_material_atlas_from_dict, fault_material_atlas_to_dict,
    initialize_fault_material_atlas, material_atlas_surface,
    validate_fault_material_atlas,
)
from .geometric_surface import audit_partition
from .thermal import SECONDS_PER_MYR


FORMAT = "geometric-bending-fault-transport-predictor-1"
MODEL = "frozen_mechanical_predictor_unaccepted_finite_geometry"
RECEIPT_FORMAT = "geometric-bending-fault-transport-receipt-1"


@dataclass(frozen=True, slots=True)
class PredictorTransportReceipt:
    sequence: int
    dt_s: float
    before_elapsed_time_s: float
    after_elapsed_time_s: float
    before_step_count: int
    after_step_count: int
    before_mechanics_sha256: str
    after_mechanics_sha256: str
    before_atlas_sha256: str
    after_atlas_sha256: str
    reference_basal_source_sha256: str
    rotation_vectors: tuple[tuple[int, tuple[float, float, float]], ...]
    source_work_j: float
    basal_dissipation_j: float
    bending_numerical_dissipation_j: float
    provenance: str
    receipt_id: str
    payload_sha256: str


@dataclass(frozen=True, slots=True)
class BendingFaultTransportState:
    origin_surface_sha256: str
    reference_basal_source_sha256: str
    reference_inventory: GeometricBoundaryState
    initial_mechanics: BendingContactState
    current_mechanics: BendingContactState
    material_atlas: FaultMaterialAtlas
    receipts: tuple[PredictorTransportReceipt, ...]


@dataclass(frozen=True, slots=True)
class PointGeometryDefect:
    """Kinematics only, expressed in the original material carrier frame.

    Exact relative tip motion, including both rotated radial displacements,
    is pulled back by the overriding SO(3) map and measured from the initial
    relative tips. Finite versus linear motion supplies the defect. These
    values neither replace saved free anchors nor drive a second solve.
    """
    point_id: str
    trace_id: str
    geometric_relative_displacement_m: tuple[float, float, float]
    linear_relative_displacement_m: tuple[float, float, float]
    geometric_minus_linear_displacement_m: tuple[float, float, float]
    geometric_gap_increment_m: float
    geometric_jump_increment_m: float
    frozen_gap_increment_m: float
    frozen_jump_increment_m: float
    gap_anchor_defect_m: float
    jump_anchor_defect_m: float


@dataclass(frozen=True, slots=True)
class BendingFaultTransportStep:
    state: BendingFaultTransportState
    predictor: BendingContactStep
    candidate_surface: object
    point_geometry_defects: tuple[PointGeometryDefect, ...]
    trace_material_pair_separation_m: tuple[tuple[str, float, float], ...]
    unresolved_material_pairing: bool
    model: str = MODEL


def _hash(value):
    return hashlib.sha256(_encode(value)).hexdigest()


def _mechanics_hash(state):
    return _hash(asdict(state))


def _receipt_body(receipt):
    body = asdict(receipt)
    body.pop("receipt_id")
    body.pop("payload_sha256")
    return dict(format=RECEIPT_FORMAT, model=MODEL, **body)


def _seal_receipt(receipt):
    digest = _hash(_receipt_body(receipt))
    return replace(receipt, payload_sha256=digest,
                   receipt_id=f"frozen-predictor:{receipt.sequence}:{digest}")


def _rotation_increment(before, after):
    difference = np.asarray(after.rotation_radians)-np.asarray(before.rotation_radians)
    return tuple((owner, tuple(map(float, vector)))
                 for owner, vector in zip(before.plate_ids, difference))


def _receipt(sequence, before, predictor, old_atlas, atlas, dt, provenance):
    return _seal_receipt(PredictorTransportReceipt(sequence, dt,
        before.elapsed_time_s, predictor.state.elapsed_time_s,
        before.step_count, predictor.state.step_count,
        _mechanics_hash(before), _mechanics_hash(predictor.state),
        old_atlas.payload_sha256, atlas.payload_sha256,
        before.basal_source_sha256, _rotation_increment(before, predictor.state),
        predictor.source_work_j, predictor.basal_dissipation_j,
        predictor.bending_numerical_dissipation_j, provenance, "", ""))


def _text(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Finite material predictions require explicit provenance")
    return value


def _reference(origin_surface, inventory, mechanics, reference_load):
    validate_geometric_basal_load(origin_surface, reference_load)
    validate_bending_contact_dynamics_state(origin_surface, mechanics, reference_load)
    if not isinstance(inventory, GeometricBoundaryState):
        raise ValueError("Predictor requires the original passive boundary inventory")
    _validate_pair(origin_surface, inventory)
    if boundary_from_dict(boundary_to_dict(inventory)) != inventory:
        raise ValueError("Reference passive inventory is not canonical")
    audit_partition(origin_surface)
    validate_reference_bending_bank_support(origin_surface, mechanics.fault_traces,
                                           mechanics.bending_parameters)


def initialize_bending_fault_transport(origin_surface, inventory, mechanics, reference_load):
    """Keep mechanics at its original epoch; declare new material-map elapsed zero."""
    _reference(origin_surface, inventory, mechanics, reference_load)
    atlas = initialize_fault_material_atlas(origin_surface, mechanics.fault_traces,
        mechanics.bending_parameters, mechanics.polarity_history)
    return BendingFaultTransportState(surface_digest(origin_surface),
        mechanics.basal_source_sha256, inventory, mechanics, mechanics, atlas, ())


def validate_bending_fault_transport_state(origin_surface, state, reference_load):
    """Reproduce both the mechanical and material-map receipt chains exactly.

    This replay is intentionally stricter than a self-consistent checksum. A
    rehashed duration, constitutive state or independent atlas rotation cannot
    become a verified predictor by merely changing its declared final fields.
    """
    if not isinstance(state, BendingFaultTransportState):
        raise ValueError("Expected a source-bound BendingFaultTransportState")
    if (state.origin_surface_sha256 != surface_digest(origin_surface)
            or state.reference_basal_source_sha256 != state.initial_mechanics.basal_source_sha256):
        raise ValueError("Finite predictor origin or reference source changed")
    _reference(origin_surface, state.reference_inventory, state.initial_mechanics, reference_load)
    if not isinstance(state.receipts, tuple):
        raise ValueError("Predictor receipts must be an immutable tuple")
    expected_atlas = initialize_fault_material_atlas(origin_surface,
        state.initial_mechanics.fault_traces, state.initial_mechanics.bending_parameters,
        state.initial_mechanics.polarity_history)
    expected_mechanics = state.initial_mechanics
    receipt_ids = set()
    for sequence, receipt in enumerate(state.receipts, start=1):
        if not isinstance(receipt, PredictorTransportReceipt):
            raise ValueError("Explicit immutable predictor receipts are required")
        if (isinstance(receipt.sequence, (bool, np.bool_))
                or not isinstance(receipt.sequence, Integral) or receipt.sequence != sequence):
            raise ValueError("Predictor receipt sequence mismatch")
        if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
               for value in (receipt.before_step_count, receipt.after_step_count)):
            raise ValueError("Predictor step counters must be explicit integers")
        dt = _number(receipt.dt_s, "Predictor interval", positive=True)
        provenance = _text(receipt.provenance)
        if receipt != _seal_receipt(receipt) or receipt.receipt_id in receipt_ids:
            raise ValueError("Predictor receipt identity, integrity or uniqueness mismatch")
        receipt_ids.add(receipt.receipt_id)
        predictor = advance_bending_contact_dynamics(origin_surface, expected_mechanics,
                                                     reference_load, dt)
        if predictor.state.elapsed_time_s <= expected_mechanics.elapsed_time_s:
            raise ValueError("Predictor physical clock cannot resolve this interval")
        moved = advance_fault_material_atlas(origin_surface, expected_atlas,
            _rotation_increment(expected_mechanics, predictor.state), dt/SECONDS_PER_MYR,
            provenance=provenance)
        reproduced = _receipt(sequence, expected_mechanics, predictor,
                              expected_atlas, moved, dt, provenance)
        if receipt != reproduced:
            raise ValueError("Predictor receipt disagrees with source-bound mechanical replay")
        expected_mechanics, expected_atlas = predictor.state, moved
    validate_fault_material_atlas(origin_surface, state.material_atlas)
    validate_bending_contact_dynamics_state(origin_surface, state.current_mechanics, reference_load)
    if (state.current_mechanics != expected_mechanics or state.material_atlas != expected_atlas
            or state.current_mechanics.basal_source_sha256 != state.reference_basal_source_sha256):
        raise ValueError("Predictor mechanics or finite material geometry differs from receipt replay")
    if state.current_mechanics.step_count-state.initial_mechanics.step_count != len(state.receipts):
        raise ValueError("Predictor epoch and material step counts disagree")
    # Conversion and subtraction need not share a final binary64 rounding bit.
    duration = math.fsum(receipt.dt_s for receipt in state.receipts)
    elapsed = state.current_mechanics.elapsed_time_s-state.initial_mechanics.elapsed_time_s
    clock_scale = max(abs(state.current_mechanics.elapsed_time_s),
                      abs(state.initial_mechanics.elapsed_time_s), duration, np.finfo(float).tiny)
    if (abs(elapsed-duration) > 64*np.finfo(float).eps*clock_scale
            or state.material_atlas.elapsed_time_myr != math.fsum(
                receipt.dt_s/SECONDS_PER_MYR for receipt in state.receipts)):
        raise ValueError("Predictor mechanical epoch and material elapsed time disagree")
    return state


def _geometry_defects(state):
    initial, current, atlas = state.initial_mechanics, state.current_mechanics, state.material_atlas
    rotations = {owner: np.asarray(matrix) for owner, matrix in atlas.owner_rotations}
    q = {owner: np.asarray(new)-np.asarray(old) for owner, old, new
         in zip(initial.plate_ids, initial.rotation_radians, current.rotation_radians)}
    traces = {trace.trace_id: trace for trace in atlas.reference_traces.traces}
    old_radial = {identity: (sub, over) for identity, sub, over in initial.transverse_displacement_m}
    radial = {identity: (sub, over) for identity, sub, over in current.transverse_displacement_m}
    old_local, new_local = dict(initial.states), dict(current.states)
    radius = atlas.radius_km*1000.
    result = []
    for point in atlas.reference_points:
        trace, position = traces[point.trace_id], np.asarray(point.position)
        sub, over = trace.subducting_plate, trace.overriding_plate
        old_sub, old_over = old_radial[point.trace_id]
        new_sub, new_over = radial[point.trace_id]
        relative_sub = rotations[over].T@(rotations[sub]@position)
        # Expand the radii to avoid cancellation of two O(radius) vectors.
        # This equals Qover.T @ ((R+w_over)Qover p-(R+w_sub)Qsub p)
        # minus the original relative tip vector (w_over0-w_sub0)*p.
        geometric = (radius*(rotations[over].T@(rotations[over]@position-rotations[sub]@position))
                     +new_over*position-new_sub*relative_sub-(old_over-old_sub)*position)
        radial_increment = (new_over-old_over)-(new_sub-old_sub)
        linear = radius*np.cross(q[over]-q[sub], position)+radial_increment*position
        exact_gap = float(geometric@np.asarray(point.normal))
        exact_jump = float(geometric@np.asarray(point.down_dip))
        frozen_gap = new_local[point.point_id].gap_free_m-old_local[point.point_id].gap_free_m
        frozen_jump = new_local[point.point_id].jump_free_m-old_local[point.point_id].jump_free_m
        result.append(PointGeometryDefect(point.point_id, point.trace_id,
            tuple(map(float, geometric)), tuple(map(float, linear)),
            tuple(map(float, geometric-linear)), exact_gap, exact_jump,
            frozen_gap, frozen_jump, exact_gap-frozen_gap, exact_jump-frozen_jump))
    return tuple(result)


def advance_bending_fault_transport(origin_surface, previous, reference_load, dt_s, *, provenance):
    """Predict mechanics once and compose exact candidate maps for the same dt."""
    dt = _number(dt_s, "Predictor interval", positive=True)
    provenance = _text(provenance)
    validate_bending_fault_transport_state(origin_surface, previous, reference_load)
    predictor = advance_bending_contact_dynamics(origin_surface, previous.current_mechanics,
                                                 reference_load, dt)
    if predictor.state.elapsed_time_s <= previous.current_mechanics.elapsed_time_s:
        raise ValueError("Predictor physical clock cannot resolve this interval")
    atlas = advance_fault_material_atlas(origin_surface, previous.material_atlas,
        _rotation_increment(previous.current_mechanics, predictor.state), dt/SECONDS_PER_MYR,
        provenance=provenance)
    receipt = _receipt(len(previous.receipts)+1, previous.current_mechanics,
                       predictor, previous.material_atlas, atlas, dt, provenance)
    state = replace(previous, current_mechanics=predictor.state, material_atlas=atlas,
                    receipts=(*previous.receipts, receipt))
    # The accepted predictor and atlas constructors already validate this new
    # edge. Existing receipts were replayed above; no extra constitutive step.
    matrices = dict(atlas.owner_rotations)
    separation = tuple((trace.trace_id, *continuous_bank_pair_separation_m(trace,
        matrices[trace.subducting_plate], matrices[trace.overriding_plate], atlas.radius_km*1000.,
        rotation_roundoff_steps=len(atlas.motion_receipts)))
        for trace in atlas.reference_traces.traces)
    unresolved = any(not np.array_equal(matrices[trace.subducting_plate],
                                       matrices[trace.overriding_plate])
                     for trace in atlas.reference_traces.traces)
    return BendingFaultTransportStep(state, predictor, material_atlas_surface(origin_surface, atlas),
                                    _geometry_defects(state), separation, unresolved)


def reassemble_candidate_basal_load(mesh, origin_surface, state, reference_load):
    """Independently integrate a resolved candidate under the original source.

    An overlapping/vacant candidate requires spatial material events first.
    The returned load is evidence for future source segments and is never
    substituted into the stored frozen mechanics or its accumulated work.
    """
    validate_bending_fault_transport_state(origin_surface, state, reference_load)
    validate_geometric_basal_load(origin_surface, reference_load, mesh=mesh)
    candidate = replace(material_atlas_surface(origin_surface, state.material_atlas), phase="partition")
    audit_partition(candidate)
    return assemble_geometric_basal_load(mesh, candidate,
        PrescribedBasalParameters(**dict(reference_load.source_parameters)),
        dict(reference_load.thermal_lid_by_fragment), reference_load.provenance)


def bending_fault_transport_to_dict(origin_surface, state, reference_load):
    validate_bending_fault_transport_state(origin_surface, state, reference_load)
    result = dict(format=FORMAT, model=MODEL,
        origin_surface_sha256=state.origin_surface_sha256,
        reference_basal_source_sha256=state.reference_basal_source_sha256,
        reference_inventory=boundary_to_dict(state.reference_inventory),
        initial_mechanics=bending_contact_dynamics_to_dict(origin_surface, state.initial_mechanics, reference_load),
        current_mechanics=bending_contact_dynamics_to_dict(origin_surface, state.current_mechanics, reference_load),
        material_atlas=fault_material_atlas_to_dict(origin_surface, state.material_atlas),
        receipts=[asdict(receipt) for receipt in state.receipts])
    return json.loads(_encode(result))


def bending_fault_transport_from_dict(raw, origin_surface, reference_load):
    expected = {field.name for field in fields(BendingFaultTransportState)}|{"format", "model"}
    if (not isinstance(raw, dict) or set(raw) != expected
            or raw["format"] != FORMAT or raw["model"] != MODEL):
        raise ValueError("Invalid frozen predictor/material transport schema")
    values = {field.name: raw[field.name] for field in fields(BendingFaultTransportState)}
    try:
        values["reference_inventory"] = boundary_from_dict(values["reference_inventory"])
        for name in ("initial_mechanics", "current_mechanics"):
            values[name] = bending_contact_dynamics_from_dict(values[name], origin_surface, reference_load)
        values["material_atlas"] = fault_material_atlas_from_dict(values["material_atlas"], origin_surface)
        if not isinstance(values["receipts"], list):
            raise ValueError("Predictor receipts require an explicit JSON list")
        receipts = []
        names = {field.name for field in fields(PredictorTransportReceipt)}
        for record in values["receipts"]:
            if not isinstance(record, dict) or set(record) != names:
                raise ValueError("Invalid predictor receipt schema")
            record = dict(record)
            if not isinstance(record["rotation_vectors"], list):
                raise ValueError("Predictor owner increments require an explicit JSON list")
            vectors = []
            for row in record["rotation_vectors"]:
                if (not isinstance(row, list) or len(row) != 2
                        or not isinstance(row[1], list) or len(row[1]) != 3):
                    raise ValueError("Invalid predictor owner increment record")
                vectors.append((row[0], tuple(row[1])))
            record["rotation_vectors"] = tuple(vectors)
            receipts.append(PredictorTransportReceipt(**record))
        values["receipts"] = tuple(receipts)
        return validate_bending_fault_transport_state(origin_surface,
            BendingFaultTransportState(**values), reference_load)
    except (TypeError, KeyError, OverflowError, IndexError) as error:
        raise ValueError("Malformed frozen predictor/material transport checkpoint") from error
