"""Passive SO(3) views of an immutable bending reference configuration.

A view changes Cartesian representation at zero physical duration. Material
all reservoirs, contact quadrature measures, beam lengths and scalar mechanical
histories remain those of the certified original configuration. An explicitly
supplied accepted physical epoch carries its actual clock and material ages;
constructing the view never adds another physical step.
It does not adopt a finite-transport candidate or redefine a basal source.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import hashlib
import json
from numbers import Integral, Real

import numpy as np
from scipy.spatial.transform import Rotation

from .geometric_bending_contact_dynamics import (
    BendingContactState, _augmented_point_network, _bending_signature,
)
from .geometric_boundary import surface_digest
from .geometric_contact_dynamics import _number
from .geometric_contact_mechanics import validate_contact_state
from .geometric_fault_motion_support import validate_reference_bending_bank_support
from .geometric_surface import GeometricFragment, GeometricSurfaceState, audit_partition
from .fractional_surface import SurfaceParcel
from .spherical_polygons import rotate_polygon

FORMAT = "geometric-bending-reference-view-1"
INTERPRETATION = "passive_cartesian_SO3_coordinate_view_zero_physical_duration"
EPOCH_INTERPRETATION = "reference_view_of_previously_accepted_physical_epoch_no_additional_step"


def _roundoff_steps(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 0:
        raise ValueError("Reference roundoff steps must be an explicit nonnegative integer")
    if 256*np.finfo(float).eps*(int(value)+1) >= 1.:
        raise ValueError("Reference roundoff budget cannot certify a rotation")
    return int(value)


def validate_reference_rotation(rotation=None, *, reference_roundoff_steps=0):
    """Return a canonical proper SO(3) matrix; bools and inferred rotations reject."""
    steps = _roundoff_steps(reference_roundoff_steps)
    tolerance = max(64*np.finfo(float).eps, 256*np.finfo(float).eps*steps)
    if rotation is None:
        rotation = np.eye(3)
    raw = np.asarray(rotation)
    values = np.asarray(rotation, dtype=object)
    if (raw.shape != (3, 3) or raw.dtype.kind not in "iuf"
            or any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                   for value in values.flat)):
        raise ValueError("Reference rotation must be an explicit finite proper SO3 matrix")
    matrix = np.asarray(rotation, dtype=float)
    if (not np.isfinite(matrix).all()
            or np.max(np.abs(matrix.T@matrix-np.eye(3))) > tolerance
            or abs(float(np.linalg.det(matrix))-1.) > tolerance):
        raise ValueError("Reference rotation must be an explicit finite proper SO3 matrix")
    return tuple(tuple(map(float, row)) for row in matrix)


def _immutable(array):
    value = np.asarray(array)
    if value.ndim == 1:
        return tuple(map(float, value))
    return tuple(_immutable(row) for row in value)


def _rotate(matrix, vector):
    if np.array_equal(matrix, np.eye(3)):
        return tuple(vector)
    return tuple(map(float, matrix@np.asarray(vector)))


@dataclass(frozen=True, slots=True)
class BendingReference:
    surface: GeometricSurfaceState
    points: tuple
    materials: tuple
    rows: tuple
    areas: tuple
    frames: tuple
    operators: tuple
    stiffness: tuple
    reference_points: tuple
    offsets: tuple
    plate_ids: tuple[int, ...]
    rotation: tuple[tuple[float, float, float], ...]
    reference_surface_sha256: str
    reference_trace_sha256: str
    reference_bending_sha256: str
    current_surface_sha256: str
    physical_dt_s: float = 0.
    interpretation: str = INTERPRETATION
    reference_elapsed_time_myr: float = 0.
    reference_roundoff_steps: int = 0
    uses_supplied_reference_surface: bool = False


def current_bending_reference(origin_surface, initial_mechanics, rotation=None, *,
        reference_elapsed_time_myr=0., reference_surface=None, reference_roundoff_steps=0):
    """Rotate the original operators while carrying their exact material measure.

    Source and mechanical ledgers are validated by the calling engine. Here
    geometry, frozen material binding and bank support are validated before
    constructing a passive view. No new local return-map step is performed.
    """
    elapsed = _number(reference_elapsed_time_myr, "Reference elapsed time", nonnegative=True)
    steps = _roundoff_steps(reference_roundoff_steps)
    canonical = validate_reference_rotation(rotation, reference_roundoff_steps=steps)
    matrix = np.asarray(canonical)
    if not isinstance(origin_surface, GeometricSurfaceState) or origin_surface.phase != "partition":
        raise ValueError("Bending reference requires a resolved original surface")
    if not isinstance(initial_mechanics, BendingContactState):
        raise ValueError("Bending reference requires an explicit initial mechanical configuration")
    _number(origin_surface.time_myr, "Reference surface clock", nonnegative=True)
    _number(origin_surface.radius_km, "Reference radius", positive=True)
    if elapsed > 0. and origin_surface.time_myr+elapsed <= origin_surface.time_myr:
        raise ValueError("Reference accepted epoch must resolve an advancing physical clock")
    original_digest = surface_digest(origin_surface)
    if initial_mechanics.surface_sha256 != original_digest:
        raise ValueError("Bending reference mechanical origin surface changed")
    bounds = tuple(_number(getattr(initial_mechanics, name), name, positive=True) for name in (
        "max_rotation_radians", "max_tip_slope", "max_bending_strain", "max_deflection_radius_ratio"))
    if initial_mechanics.bending_source_sha256 != _bending_signature(origin_surface,
            initial_mechanics.fault_traces, initial_mechanics.bending_parameters, bounds):
        raise ValueError("Bending reference parameters or original trace signature changed")
    owners = tuple(sorted({fragment.parcel.plate for fragment in origin_surface.fragments}))
    if (not isinstance(initial_mechanics.plate_ids, tuple)
            or any(isinstance(owner, (bool, np.bool_)) or not isinstance(owner, Integral)
                   or owner < 0 for owner in initial_mechanics.plate_ids)
            or initial_mechanics.plate_ids != owners):
        raise ValueError("Bending reference needs canonical compact actual material owners")
    audit_partition(origin_surface)
    network = _augmented_point_network(origin_surface, initial_mechanics.fault_traces,
                                       owners, initial_mechanics.bending_parameters)
    points, materials, rows, areas, frames, offsets, operators, stiffness = network
    if tuple(identity for identity, _ in initial_mechanics.states) != tuple(point.point_id for point in points):
        raise ValueError("Bending reference must preserve every original material point identity")
    for material, area, (_, local) in zip(materials, areas, initial_mechanics.states):
        validate_contact_state(local, columns=material.columns, law=material.law,
                               area_m2=float(area), water=material.water)
    validate_reference_bending_bank_support(origin_surface, initial_mechanics.fault_traces,
                                           initial_mechanics.bending_parameters)
    dimension = operators.shape[-1]
    transform = np.eye(dimension)
    for index in range(len(owners)):
        transform[3*index:3*index+3, 3*index:3*index+3] = matrix
    if np.array_equal(matrix, np.eye(3)):
        surface, moved_points = origin_surface, points
        moved_rows, moved_operators = rows, operators
    else:
        vector = Rotation.from_matrix(matrix).as_rotvec()
        surface = replace(origin_surface, fragments=tuple(replace(fragment,
            polygon=tuple(map(tuple, rotate_polygon(fragment.polygon, vector, 1.))))
            for fragment in origin_surface.fragments))
        moved_points = tuple(replace(point, **{name: _rotate(matrix, getattr(point, name))
            for name in ("position", "horizontal", "normal", "down_dip")}) for point in points)
        moved_rows, moved_operators = rows@transform.T, operators@transform.T
    if elapsed:
        surface = replace(surface, time_myr=origin_surface.time_myr+elapsed,
            fragments=tuple(replace(fragment, parcel=replace(fragment.parcel,
                age_myr=fragment.parcel.age_myr+elapsed)) for fragment in surface.fragments))
    if reference_surface is not None:
        _validate_supplied_surface(origin_surface, reference_surface, matrix, elapsed, steps)
        surface = reference_surface
    # K has only scalar radial-bank entries and is therefore unchanged exactly.
    # Recomputing physical arc lengths from rotated endpoints can change a bit.
    return BendingReference(surface, moved_points, materials, _immutable(moved_rows),
        _immutable(areas), _immutable(frames), _immutable(moved_operators), _immutable(stiffness),
        points, _immutable(offsets), owners, canonical, original_digest,
        initial_mechanics.fault_traces.trace_sha256, initial_mechanics.bending_source_sha256,
        surface_digest(surface), 0.,
        EPOCH_INTERPRETATION if elapsed else INTERPRETATION, elapsed, steps,
        reference_surface is not None)


def _validate_supplied_surface(origin, supplied, matrix, elapsed, steps):
    if not isinstance(supplied, GeometricSurfaceState) or supplied.phase != "partition":
        raise ValueError("Supplied reference surface must be a resolved accepted partition")
    _number(supplied.time_myr, "Supplied reference clock", nonnegative=True)
    _number(supplied.radius_km, "Supplied reference radius", positive=True)
    if (supplied.radius_km != origin.radius_km or supplied.version != origin.version
            or supplied.known_material_ids != origin.known_material_ids):
        raise ValueError("Supplied reference radius, version or material origins changed")
    budget = 256*np.finfo(float).eps*(steps+1)
    expected_time = origin.time_myr+elapsed
    if (elapsed == 0. and supplied.time_myr != origin.time_myr
            or elapsed > 0. and supplied.time_myr <= origin.time_myr
            or abs(supplied.time_myr-expected_time) > budget*max(
                abs(expected_time), abs(supplied.time_myr), elapsed, np.finfo(float).tiny)):
        raise ValueError("Supplied reference surface has a different accepted physical epoch")
    old = {fragment.fragment_id: fragment for fragment in origin.fragments}
    new = {fragment.fragment_id: fragment for fragment in supplied.fragments}
    if set(old) != set(new):
        raise ValueError("Supplied reference surface cannot change original material fragment IDs")
    for identity, fragment in old.items():
        current = new[identity]
        if (current.parent_fragment_id != fragment.parent_fragment_id
                or replace(current.parcel, age_myr=fragment.parcel.age_myr) != fragment.parcel):
            raise ValueError("Supplied reference surface changed immutable material properties or reservoirs")
        expected_age = fragment.parcel.age_myr+elapsed
        if (elapsed == 0. and current.parcel.age_myr != fragment.parcel.age_myr
                or current.parcel.age_myr < fragment.parcel.age_myr
                or abs(current.parcel.age_myr-expected_age) > budget*max(
                    expected_age, current.parcel.age_myr, elapsed, np.finfo(float).tiny)):
            raise ValueError("Supplied reference material age differs from its accepted epoch")
        expected = np.asarray(fragment.polygon)@matrix.T
        actual = np.asarray(current.polygon)
        if actual.shape != expected.shape:
            raise ValueError("Supplied reference surface changed polygon material vertex labels")
        scale = np.abs(np.asarray(fragment.polygon))@np.abs(matrix.T)+np.abs(actual)
        if np.any(np.abs(actual-expected) > budget*scale):
            raise ValueError("Supplied reference surface is not the certified rigid image of its origin")
    audit_partition(supplied)


def _surface_from_dict(raw):
    if not isinstance(raw, dict) or set(raw) != {field.name for field in fields(GeometricSurfaceState)}:
        raise ValueError("Invalid supplied reference surface schema")
    if not isinstance(raw["fragments"], list) or not isinstance(raw["known_material_ids"], list):
        raise ValueError("Supplied reference surface requires explicit material lists")
    fragments = []
    for row in raw["fragments"]:
        if not isinstance(row, dict) or set(row) != {field.name for field in fields(GeometricFragment)}:
            raise ValueError("Invalid supplied reference fragment schema")
        parcel = row["parcel"]
        if not isinstance(parcel, dict) or set(parcel) != {field.name for field in fields(SurfaceParcel)}:
            raise ValueError("Invalid supplied reference material schema")
        if (not isinstance(row["polygon"], list)
                or any(not isinstance(point, list) or len(point) != 3 for point in row["polygon"])):
            raise ValueError("Supplied reference polygon needs explicit Cartesian vertex lists")
        for point in row["polygon"]:
            for value in point:
                _number(value, "Supplied reference vertex")
        fragments.append(GeometricFragment(row["fragment_id"], tuple(map(tuple, row["polygon"])),
            SurfaceParcel(**parcel), row["parent_fragment_id"]))
    return GeometricSurfaceState(raw["time_myr"], raw["radius_km"], tuple(fragments),
        tuple(raw["known_material_ids"]), raw["phase"], raw["version"])


def _receipt(view):
    return dict(format=FORMAT, interpretation=view.interpretation, physical_dt_s=0.,
        rotation=[list(row) for row in view.rotation],
        reference_surface_sha256=view.reference_surface_sha256,
        reference_trace_sha256=view.reference_trace_sha256,
        reference_bending_sha256=view.reference_bending_sha256,
        current_surface_sha256=view.current_surface_sha256,
        reference_elapsed_time_myr=view.reference_elapsed_time_myr,
        reference_roundoff_steps=view.reference_roundoff_steps,
        reference_surface=(json.loads(json.dumps(asdict(view.surface), allow_nan=False))
                           if view.uses_supplied_reference_surface else None))


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def bending_reference_to_dict(origin_surface, initial_mechanics, reference):
    if (not isinstance(reference, BendingReference)
            or isinstance(reference.physical_dt_s, (bool, np.bool_))
            or not isinstance(reference.physical_dt_s, Real) or reference.physical_dt_s != 0.
            or not isinstance(reference.uses_supplied_reference_surface, bool)):
        raise ValueError("Expected an immutable BendingReference")
    actual = current_bending_reference(origin_surface, initial_mechanics, reference.rotation,
        reference_elapsed_time_myr=reference.reference_elapsed_time_myr,
        reference_surface=reference.surface if reference.uses_supplied_reference_surface else None,
        reference_roundoff_steps=reference.reference_roundoff_steps)
    if reference != actual:
        raise ValueError("Bending reference geometry or original material measure changed")
    result = _receipt(reference)
    result["payload_sha256"] = _hash(result)
    return result


def bending_reference_from_dict(raw, origin_surface, initial_mechanics):
    expected = {"format", "interpretation", "physical_dt_s", "rotation", "reference_surface_sha256",
        "reference_trace_sha256", "reference_bending_sha256", "current_surface_sha256",
        "reference_elapsed_time_myr", "reference_roundoff_steps", "reference_surface", "payload_sha256"}
    if (not isinstance(raw, dict) or set(raw) != expected or raw["format"] != FORMAT
            or raw["interpretation"] not in (INTERPRETATION, EPOCH_INTERPRETATION)
            or isinstance(raw["physical_dt_s"], (bool, np.bool_)) or raw["physical_dt_s"] != 0.):
        raise ValueError("Invalid zero-duration passive bending-reference schema")
    body = {name: value for name, value in raw.items() if name != "payload_sha256"}
    if _hash(body) != raw["payload_sha256"]:
        raise ValueError("Bending reference receipt integrity mismatch")
    if not isinstance(raw["rotation"], list) or any(not isinstance(row, list) for row in raw["rotation"]):
        raise ValueError("Bending reference rotation must be an explicit JSON matrix")
    supplied = None if raw["reference_surface"] is None else _surface_from_dict(raw["reference_surface"])
    view = current_bending_reference(origin_surface, initial_mechanics, raw["rotation"],
        reference_elapsed_time_myr=raw["reference_elapsed_time_myr"], reference_surface=supplied,
        reference_roundoff_steps=raw["reference_roundoff_steps"])
    if body != _receipt(view):
        raise ValueError("Bending reference receipt differs from original configuration replay")
    return view
