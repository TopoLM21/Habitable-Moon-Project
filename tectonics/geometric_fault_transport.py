"""Exact finite material transport candidates anchored to a frozen origin.

This atlas is geometric evidence, not an accepted mechanical step. Its weak
plane follows the explicitly named overriding-bank material frame; the two
bank edge labels follow their own owners and may cease to coincide. No
contact pairing, birth, loss, force, energy or history remapping is inferred.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import hashlib
import json
import math
from numbers import Integral, Real

import numpy as np
from scipy.spatial.transform import Rotation

from .geometric_bending import (FaultBendingParameters, canonical_bending_parameters,
    bending_parameters_to_dict, bending_parameters_from_dict)
from .geometric_boundary import surface_digest
from .geometric_fault_traces import (FaultTraceSet, FaultTracePoint, PhysicalFaultTrace,
    TracePolarityEvidence, fault_trace_points, fault_trace_set_to_dict,
    fault_trace_set_from_dict, trace_history_to_dict, trace_history_from_dict,
    validate_fault_trace_set, validate_fault_trace_history)
from .geometric_surface import GeometricSurfaceState, audit_partition
from .geometric_fault_motion_support import (
    validate_reference_bending_bank_support, continuous_bank_pair_separation_m,
)
from .spherical_polygons import rotate_polygon

FORMAT = "geometric-fault-material-atlas-1"
CARRIER = "overriding_plate_material_frame"
STATUS = "unaccepted_transport_candidate"
COORDINATES = "global_unit_sphere_column_vectors_chronological_left_SO3_composition"
_EPS = np.finfo(float).eps


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def _number(value, name, *, nonnegative=False, positive=False):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not math.isfinite(float(value))):
        raise ValueError(f"{name} must be finite real")
    result = float(value)
    if (positive and result <= 0.) or (nonnegative and result < 0.):
        raise ValueError(f"Invalid {name}")
    return result


def _owner(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 0:
        raise ValueError("Material owner IDs must be nonnegative integers")
    return int(value)


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be explicit nonempty text")
    return value


def _sha(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(v not in "0123456789abcdef" for v in value):
        raise ValueError(f"Invalid {name}")
    return value


def _vector(value, name):
    if not isinstance(value, (tuple, list, np.ndarray)) or len(value) != 3:
        raise ValueError(f"{name} must contain three Cartesian components")
    return tuple(_number(component, name) for component in value)


def _unit(value, name):
    result = _vector(value, name)
    if abs(math.fsum(v*v for v in result)-1.) > 64*_EPS:
        raise ValueError(f"{name} must be a unit vector")
    return result


def _matrix(value, *, steps=0):
    if not isinstance(value, (tuple, list, np.ndarray)) or len(value) != 3:
        raise ValueError("Owner rotation must be an SO3 matrix")
    result = tuple(_vector(row, "Owner rotation") for row in value)
    array = np.asarray(result)
    tolerance = 256*_EPS*max(1, steps)
    if (np.linalg.norm(array.T@array-np.eye(3), ord=np.inf) > tolerance
            or abs(np.linalg.det(array)-1.) > tolerance):
        raise ValueError("Owner rotation must be a proper SO3 matrix")
    return result


def _rows(array):
    return tuple(tuple(map(float, row)) for row in array)


def _rotated(matrix, vector):
    # Reference coordinates are never repeatedly advected or checkpoint-normalized.
    if np.array_equal(matrix, np.eye(3)):
        return tuple(vector)
    return tuple(map(float, np.asarray(matrix)@np.asarray(vector)))


def _rotations(raw, owners):
    if not isinstance(raw, (tuple, list)):
        raise ValueError("Explicit compact owner rotation vectors are required")
    result = []
    for row in raw:
        if not isinstance(row, (tuple, list)) or len(row) != 2:
            raise ValueError("Rotation vectors require (owner, vector) records")
        result.append((_owner(row[0]), _vector(row[1], "Incremental rotation vector")))
    result = tuple(result)
    if tuple(owner for owner, _ in result) != owners:
        raise ValueError("Rotation vectors must cover canonical actual owners exactly once")
    return result


@dataclass(frozen=True, slots=True)
class MaterialBankClampReference:
    point_id: str
    trace_id: str
    subducting_plate: int
    overriding_plate: int
    edge: tuple[float, float, float]
    sub_clamp: tuple[float, float, float]
    over_clamp: tuple[float, float, float]
    sub_width_m: float
    over_width_m: float


@dataclass(frozen=True, slots=True)
class MaterialMotionReceipt:
    step_index: int
    start_time_myr: float
    end_time_myr: float
    dt_myr: float
    rotation_vectors: tuple[tuple[int, tuple[float, float, float]], ...]
    provenance: str
    previous_motion_sha256: str
    motion_sha256: str


@dataclass(frozen=True, slots=True)
class FaultMaterialAtlas:
    origin_surface_sha256: str
    reference_trace_sha256: str
    origin_time_myr: float
    radius_km: float
    plate_ids: tuple[int, ...]
    reference_traces: FaultTraceSet
    bending_parameters: tuple[FaultBendingParameters, ...]
    polarity_history: tuple[TracePolarityEvidence, ...]
    reference_points: tuple[FaultTracePoint, ...]
    reference_bank_clamps: tuple[MaterialBankClampReference, ...]
    owner_rotations: tuple[tuple[int, tuple[tuple[float, float, float], ...]], ...]
    motion_receipts: tuple[MaterialMotionReceipt, ...]
    elapsed_time_myr: float
    time_myr: float
    current_surface_sha256: str
    carrier_convention: str
    coordinate_convention: str
    status: str
    payload_sha256: str


@dataclass(frozen=True, slots=True)
class TransportedBankPoint:
    point_id: str
    trace_id: str
    subducting_plate: int
    overriding_plate: int
    sub_edge: tuple[float, float, float]
    over_edge: tuple[float, float, float]
    sub_clamp: tuple[float, float, float]
    over_clamp: tuple[float, float, float]
    sub_radial: tuple[float, float, float]
    over_radial: tuple[float, float, float]
    bank_separation_m: float


@dataclass(frozen=True, slots=True)
class TransportedTraceMismatch:
    trace_id: str
    bank_maps_identical: bool
    endpoint_separation_m: tuple[float, float]
    maximum_bank_separation_m: float


@dataclass(frozen=True, slots=True)
class TransportedFaultGeometry:
    origin_surface_sha256: str
    current_surface_sha256: str
    carrier_convention: str
    status: str
    traces: tuple[PhysicalFaultTrace, ...]
    points: tuple[FaultTracePoint, ...]
    bank_points: tuple[TransportedBankPoint, ...]
    history: tuple[TracePolarityEvidence, ...]
    trace_mismatches: tuple[TransportedTraceMismatch, ...]


def _check_reference_records(atlas):
    if not isinstance(atlas.reference_points, tuple) or not isinstance(atlas.reference_bank_clamps, tuple):
        raise ValueError("Material atlas reference records must be immutable")
    for point in atlas.reference_points:
        if not isinstance(point, FaultTracePoint):
            raise ValueError("Explicit physical reference points are required")
        _text(point.point_id, "Reference point ID")
        _text(point.trace_id, "Reference trace ID")
        for value in (point.panel_index, point.node_index):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 0:
                raise ValueError("Reference point indices must be nonnegative integers")
        for name in ("position", "horizontal", "normal", "down_dip"):
            _unit(getattr(point, name), name)
        _number(point.line_weight_m, "Reference line weight", positive=True)
        _number(point.area_m2, "Reference contact area", positive=True)
    for clamp in atlas.reference_bank_clamps:
        if not isinstance(clamp, MaterialBankClampReference):
            raise ValueError("Explicit reference bank clamp records are required")
        _text(clamp.point_id, "Clamp point ID")
        _text(clamp.trace_id, "Clamp trace ID")
        _owner(clamp.subducting_plate)
        _owner(clamp.overriding_plate)
        for name in ("edge", "sub_clamp", "over_clamp"):
            _unit(getattr(clamp, name), name)
        _number(clamp.sub_width_m, "Subducting clamp width", positive=True)
        _number(clamp.over_width_m, "Overriding clamp width", positive=True)


def _references(surface, trace_set, parameters):
    validate_reference_bending_bank_support(surface, trace_set, parameters)
    points = fault_trace_points(surface, trace_set)
    traces = {trace.trace_id: trace for trace in trace_set.traces}
    banks = {item.trace_id: item for item in parameters}
    radius = surface.radius_km*1000.
    clamps = []
    for point in points:
        trace, pair = traces[point.trace_id], banks[point.trace_id]
        widths = pair.sub.width_m, pair.over.width_m
        if any(width/radius >= math.pi/2. for width in widths):
            raise ValueError("Material bank clamps require geodesic width smaller than a hemisphere")
        position, horizontal = np.asarray(point.position), np.asarray(point.horizontal)
        sub = math.cos(widths[0]/radius)*position-math.sin(widths[0]/radius)*horizontal
        over = math.cos(widths[1]/radius)*position+math.sin(widths[1]/radius)*horizontal
        clamps.append(MaterialBankClampReference(point.point_id, point.trace_id,
            trace.subducting_plate, trace.overriding_plate, point.position,
            _unit(sub, "Subducting clamp"), _unit(over, "Overriding clamp"), *widths))
    return points, tuple(clamps)


def _body(atlas):
    result = asdict(atlas)
    result.pop("payload_sha256")
    return dict(format=FORMAT, **result)


def _seal(atlas):
    return replace(atlas, payload_sha256=_hash(_body(atlas)))


def _receipt_body(receipt):
    body = asdict(receipt)
    body.pop("motion_sha256")
    return dict(format="geometric-material-motion-1", **body)


def _surface(origin, atlas):
    rotations = dict(atlas.owner_rotations)
    reference = rotations[atlas.plate_ids[0]]
    common = all(np.array_equal(reference, rotations[owner]) for owner in atlas.plate_ids)
    moved = []
    for fragment in origin.fragments:
        matrix = np.asarray(rotations[fragment.parcel.plate])
        if np.array_equal(matrix, np.eye(3)):
            polygon = fragment.polygon
        else:
            # One fresh rotation of the original polygon. The established
            # spherical routine validates and rounds unit coordinates safely.
            vector = Rotation.from_matrix(matrix).as_rotvec()
            polygon = _rows(rotate_polygon(fragment.polygon, vector, 1.))
        moved.append(replace(fragment, polygon=polygon,
            parcel=replace(fragment.parcel, age_myr=fragment.parcel.age_myr+atlas.elapsed_time_myr)))
    return replace(origin, time_myr=atlas.time_myr, fragments=tuple(moved),
                   phase="partition" if common else "advected_candidates")


def initialize_fault_material_atlas(surface, trace_set, bending_parameters, history=()):
    """Declare immutable material labels and an explicit overriding-bank carrier."""
    if not isinstance(surface, GeometricSurfaceState) or surface.phase != "partition":
        raise ValueError("Material atlas requires a resolved origin surface")
    _number(surface.time_myr, "Origin clock", nonnegative=True)
    _number(surface.radius_km, "Origin radius", positive=True)
    audit_partition(surface)
    validate_fault_trace_set(surface, trace_set)
    parameters = canonical_bending_parameters(bending_parameters,
        trace_ids=tuple(trace.trace_id for trace in trace_set.traces))
    records = validate_fault_trace_history(trace_set, history)
    points, clamps = _references(surface, trace_set, parameters)
    owners = tuple(sorted({_owner(fragment.parcel.plate) for fragment in surface.fragments}))
    digest = surface_digest(surface)
    atlas = FaultMaterialAtlas(digest, trace_set.trace_sha256, surface.time_myr,
        surface.radius_km, owners, trace_set, parameters, records, points, clamps,
        tuple((owner, _rows(np.eye(3))) for owner in owners), (), 0., surface.time_myr,
        digest, CARRIER, COORDINATES, STATUS, "")
    return _seal(atlas)


def validate_fault_material_atlas(origin_surface, atlas):
    """Replay the entire motion chain; reject rehashed geometry or clock edits."""
    if not isinstance(atlas, FaultMaterialAtlas):
        raise ValueError("Expected a FaultMaterialAtlas")
    if (atlas.carrier_convention != CARRIER or atlas.coordinate_convention != COORDINATES
            or atlas.status != STATUS):
        raise ValueError("Material atlas carrier, coordinates and candidate status are immutable")
    if not isinstance(origin_surface, GeometricSurfaceState) or origin_surface.phase != "partition":
        raise ValueError("Material atlas requires its resolved origin surface")
    _sha(atlas.origin_surface_sha256, "origin surface checksum")
    _sha(atlas.reference_trace_sha256, "reference trace checksum")
    _sha(atlas.current_surface_sha256, "current surface checksum")
    _sha(atlas.payload_sha256, "material atlas checksum")
    if atlas.payload_sha256 != _hash(_body(atlas)):
        raise ValueError("Material atlas payload integrity mismatch")
    if (atlas.origin_surface_sha256 != surface_digest(origin_surface)
            or atlas.origin_time_myr != origin_surface.time_myr
            or atlas.radius_km != origin_surface.radius_km):
        raise ValueError("Material atlas origin geometry, clock or radius mismatch")
    _number(atlas.origin_time_myr, "Origin clock", nonnegative=True)
    _number(atlas.radius_km, "Origin radius", positive=True)
    owners = tuple(sorted({_owner(fragment.parcel.plate) for fragment in origin_surface.fragments}))
    if not isinstance(atlas.plate_ids, tuple) or tuple(_owner(owner) for owner in atlas.plate_ids) != owners:
        raise ValueError("Material atlas requires canonical compact actual owners")
    validate_fault_trace_set(origin_surface, atlas.reference_traces)
    if atlas.reference_trace_sha256 != atlas.reference_traces.trace_sha256:
        raise ValueError("Material atlas physical reference signature mismatch")
    parameters = canonical_bending_parameters(atlas.bending_parameters,
        trace_ids=tuple(trace.trace_id for trace in atlas.reference_traces.traces))
    if parameters != atlas.bending_parameters or not isinstance(atlas.bending_parameters, tuple):
        raise ValueError("Material atlas bending parameters must be immutable and canonical")
    if not isinstance(atlas.polarity_history, tuple):
        raise ValueError("Material atlas history must be immutable")
    validate_fault_trace_history(atlas.reference_traces, atlas.polarity_history)
    _check_reference_records(atlas)
    points, clamps = _references(origin_surface, atlas.reference_traces, parameters)
    if atlas.reference_points != points or atlas.reference_bank_clamps != clamps:
        raise ValueError("Material atlas reference point labels or bank clamps changed")
    if not isinstance(atlas.motion_receipts, tuple):
        raise ValueError("Material motion receipts must be immutable")
    cumulative = {owner: np.eye(3) for owner in owners}
    durations = []
    previous_sha = _hash(dict(origin_surface_sha256=atlas.origin_surface_sha256,
        reference_trace_sha256=atlas.reference_trace_sha256, carrier_convention=CARRIER))
    clock = atlas.origin_time_myr
    for index, receipt in enumerate(atlas.motion_receipts, start=1):
        if not isinstance(receipt, MaterialMotionReceipt):
            raise ValueError("Explicit material motion receipts are required")
        if (isinstance(receipt.step_index, (bool, np.bool_))
                or not isinstance(receipt.step_index, Integral) or receipt.step_index != index):
            raise ValueError("Material motion receipt index mismatch")
        dt = _number(receipt.dt_myr, "Motion duration", positive=True)
        _number(receipt.start_time_myr, "Motion start clock", nonnegative=True)
        _number(receipt.end_time_myr, "Motion end clock", nonnegative=True)
        durations.append(dt)
        end = atlas.origin_time_myr+math.fsum(durations)
        if receipt.start_time_myr != clock or receipt.end_time_myr != end or end <= clock:
            raise ValueError("Material motion receipt clock does not advance consistently")
        vectors = _rotations(receipt.rotation_vectors, owners)
        if not isinstance(receipt.rotation_vectors, tuple) or vectors != receipt.rotation_vectors:
            raise ValueError("Material motion rotation records must be immutable")
        _text(receipt.provenance, "Motion provenance")
        _sha(receipt.previous_motion_sha256, "previous motion checksum")
        _sha(receipt.motion_sha256, "motion checksum")
        if receipt.previous_motion_sha256 != previous_sha or receipt.motion_sha256 != _hash(_receipt_body(receipt)):
            raise ValueError("Material motion receipt chain integrity mismatch")
        for owner, vector in vectors:
            cumulative[owner] = Rotation.from_rotvec(vector).as_matrix()@cumulative[owner]
        clock, previous_sha = end, receipt.motion_sha256
    elapsed = _number(atlas.elapsed_time_myr, "Elapsed material time", nonnegative=True)
    _number(atlas.time_myr, "Current material clock", nonnegative=True)
    if elapsed != math.fsum(durations) or atlas.time_myr != clock:
        raise ValueError("Material atlas elapsed time or clock mismatch")
    if not isinstance(atlas.owner_rotations, tuple) or len(atlas.owner_rotations) != len(owners):
        raise ValueError("Material atlas requires canonical compact owner matrices")
    saved = []
    for row in atlas.owner_rotations:
        if not isinstance(row, tuple) or len(row) != 2:
            raise ValueError("Material owner matrices require immutable records")
        owner, matrix = _owner(row[0]), _matrix(row[1], steps=len(durations))
        saved.append((owner, matrix))
    expected = tuple((owner, _rows(cumulative[owner])) for owner in owners)
    if tuple(saved) != expected or atlas.owner_rotations != expected:
        raise ValueError("Material atlas owner rotations disagree with chronological SO3 replay")
    if atlas.current_surface_sha256 != surface_digest(_surface(origin_surface, atlas)):
        raise ValueError("Material atlas current surface geometry checksum mismatch")
    return atlas


def advance_fault_material_atlas(origin_surface, atlas, rotation_vectors, dt_myr, *, provenance):
    """Compose finite owner rotations; preserve original mechanical references."""
    validate_fault_material_atlas(origin_surface, atlas)
    vectors = _rotations(rotation_vectors, atlas.plate_ids)
    dt = _number(dt_myr, "Motion duration", positive=True)
    provenance = _text(provenance, "Motion provenance")
    elapsed = math.fsum((*[item.dt_myr for item in atlas.motion_receipts], dt))
    end = atlas.origin_time_myr+elapsed
    if end <= atlas.time_myr:
        raise ValueError("Material motion must advance the physical clock")
    previous_sha = (atlas.motion_receipts[-1].motion_sha256 if atlas.motion_receipts else
        _hash(dict(origin_surface_sha256=atlas.origin_surface_sha256,
            reference_trace_sha256=atlas.reference_trace_sha256, carrier_convention=CARRIER)))
    receipt = MaterialMotionReceipt(len(atlas.motion_receipts)+1, atlas.time_myr, end,
        dt, vectors, provenance, previous_sha, "")
    receipt = replace(receipt, motion_sha256=_hash(_receipt_body(receipt)))
    old = dict(atlas.owner_rotations)
    matrices = tuple((owner, _rows(Rotation.from_rotvec(vector).as_matrix()@np.asarray(old[owner])))
                     for owner, vector in vectors)
    result = replace(atlas, owner_rotations=matrices, motion_receipts=(*atlas.motion_receipts, receipt),
                     elapsed_time_myr=elapsed, time_myr=end, current_surface_sha256="", payload_sha256="")
    result = replace(result, current_surface_sha256=surface_digest(_surface(origin_surface, result)))
    result = _seal(result)
    validate_fault_material_atlas(origin_surface, result)
    return result


def material_atlas_surface(origin_surface, atlas):
    validate_fault_material_atlas(origin_surface, atlas)
    return _surface(origin_surface, atlas)


def _trace_mismatch(trace, sub, over, radius, *, rotation_roundoff_steps):
    _, maximum = continuous_bank_pair_separation_m(trace, sub, over, radius,
        rotation_roundoff_steps=rotation_roundoff_steps)
    delta = np.asarray(sub)-np.asarray(over)
    endpoints = tuple(radius*float(np.linalg.norm(delta@np.asarray(position)))
                      for position in (trace.start, trace.end))
    return TransportedTraceMismatch(trace.trace_id, np.array_equal(sub, over), endpoints, maximum)


def transported_fault_geometry(origin_surface, atlas):
    """Return distinct bank labels and a plane carried by the overriding bank."""
    validate_fault_material_atlas(origin_surface, atlas)
    rotations = dict(atlas.owner_rotations)
    traces = {trace.trace_id: trace for trace in atlas.reference_traces.traces}
    moved_traces = tuple(replace(trace,
        start=_rotated(rotations[trace.overriding_plate], trace.start),
        end=_rotated(rotations[trace.overriding_plate], trace.end)) for trace in atlas.reference_traces.traces)
    points = []
    for point in atlas.reference_points:
        matrix = rotations[traces[point.trace_id].overriding_plate]
        points.append(replace(point, **{name: _rotated(matrix, getattr(point, name))
            for name in ("position", "horizontal", "normal", "down_dip")}))
    bank_points = []
    radius = atlas.radius_km*1000.
    for reference in atlas.reference_bank_clamps:
        sub, over = rotations[reference.subducting_plate], rotations[reference.overriding_plate]
        sub_edge, over_edge = _rotated(sub, reference.edge), _rotated(over, reference.edge)
        bank_points.append(TransportedBankPoint(reference.point_id, reference.trace_id,
            reference.subducting_plate, reference.overriding_plate, sub_edge, over_edge,
            _rotated(sub, reference.sub_clamp), _rotated(over, reference.over_clamp),
            sub_edge, over_edge, radius*float(np.linalg.norm(np.asarray(sub_edge)-over_edge))))
    history = tuple(replace(item,
        start=_rotated(rotations[traces[item.trace_id].overriding_plate], item.start),
        end=_rotated(rotations[traces[item.trace_id].overriding_plate], item.end))
        for item in atlas.polarity_history)
    mismatches = tuple(_trace_mismatch(trace, rotations[trace.subducting_plate],
        rotations[trace.overriding_plate], radius,
        rotation_roundoff_steps=len(atlas.motion_receipts)) for trace in atlas.reference_traces.traces)
    return TransportedFaultGeometry(atlas.origin_surface_sha256, atlas.current_surface_sha256,
        CARRIER, STATUS, moved_traces, tuple(points), tuple(bank_points), history, mismatches)


def fault_material_atlas_to_dict(origin_surface, atlas):
    validate_fault_material_atlas(origin_surface, atlas)
    # Canonical JSON types, without mutating saved binary64 coordinates.
    return json.loads(json.dumps(dict(format=FORMAT, **asdict(atlas)), allow_nan=False))


def _strict(raw, cls):
    if not isinstance(raw, dict) or set(raw) != {field.name for field in fields(cls)}:
        raise ValueError(f"Invalid {cls.__name__} checkpoint schema")
    return dict(raw)


def fault_material_atlas_from_dict(raw, origin_surface):
    expected = {field.name for field in fields(FaultMaterialAtlas)} | {"format"}
    if not isinstance(raw, dict) or set(raw) != expected or raw["format"] != FORMAT:
        raise ValueError("Invalid fault material atlas checkpoint schema")
    data = dict(raw)
    data.pop("format")
    try:
        for name in ("plate_ids", "bending_parameters", "polarity_history", "reference_points",
                     "reference_bank_clamps", "owner_rotations", "motion_receipts"):
            if not isinstance(data[name], list):
                raise ValueError("Material atlas collections must be explicit JSON lists")
        traces_raw = data["reference_traces"]
        _strict(traces_raw, FaultTraceSet)
        from .geometric_fault_traces import FORMAT as TRACE_FORMAT, FRAME, EVOLUTION
        trace_set = fault_trace_set_from_dict(dict(format=TRACE_FORMAT,
            coordinate_convention=FRAME, evolution=EVOLUTION, **traces_raw), origin_surface)
        data["reference_traces"] = trace_set
        data["bending_parameters"] = bending_parameters_from_dict(data["bending_parameters"])
        data["polarity_history"] = trace_history_from_dict(data["polarity_history"], trace_set)
        point_rows = []
        for row in data["reference_points"]:
            values = _strict(row, FaultTracePoint)
            for name in ("position", "horizontal", "normal", "down_dip"):
                values[name] = _vector(values[name], name)
            point_rows.append(FaultTracePoint(**values))
        data["reference_points"] = tuple(point_rows)
        clamp_rows = []
        for row in data["reference_bank_clamps"]:
            values = _strict(row, MaterialBankClampReference)
            for name in ("edge", "sub_clamp", "over_clamp"):
                values[name] = _unit(values[name], name)
            clamp_rows.append(MaterialBankClampReference(**values))
        data["reference_bank_clamps"] = tuple(clamp_rows)
        data["plate_ids"] = tuple(data["plate_ids"])
        matrices = []
        for row in data["owner_rotations"]:
            if not isinstance(row, list) or len(row) != 2:
                raise ValueError("Owner matrix record must retain its owner and matrix")
            matrices.append((_owner(row[0]), _matrix(row[1], steps=len(data["motion_receipts"]))))
        data["owner_rotations"] = tuple(matrices)
        receipts = []
        for row in data["motion_receipts"]:
            values = _strict(row, MaterialMotionReceipt)
            values["rotation_vectors"] = _rotations(values["rotation_vectors"], data["plate_ids"])
            receipts.append(MaterialMotionReceipt(**values))
        data["motion_receipts"] = tuple(receipts)
        result = FaultMaterialAtlas(**data)
        return validate_fault_material_atlas(origin_surface, result)
    except (TypeError, KeyError, IndexError) as error:
        raise ValueError("Malformed fault material atlas checkpoint") from error
