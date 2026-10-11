"""Explicit portable rigid epochs of a strict material partition.

The reference rays describe each material footprint in its own saved epoch.
Ordered, full angular-velocity increments describe its subsequent pose.  A
rounded world projection is a view, never the next footprint or an area oracle.

This format currently accepts common motion only.  Differential transport must
first preserve native clipping planes and cut lineage; passing its projection
to the ordinary world-coordinate transport would lose the thin geometry again.
The existing geometric surface/checkpoint contract is deliberately separate.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from numbers import Real
from pathlib import Path

import numpy as np

from .fractional_surface import EXTENSIVE_FIELDS, SurfaceParcel
from .geometric_intrinsic_geometry import (
    IntrinsicGeometry, RotationIncrement, rotation_increment_from_dict,
    rotation_increment_to_dict,
)
from .geometric_rotation_vectors import compact_rotation_vectors
from .geometric_surface import (
    AREA_REL_TOL, GeometricFragment, GeometricSurfaceState, audit_partition,
)

VERSION = "geometric-intrinsic-surface-1"
FORMAT = "geometric-intrinsic-surface-checkpoint-1"


def _real(value, label, *, positive=False):
    message = f"{label} must be a finite {'positive' if positive else 'nonnegative'} real number"
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(message)
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(message) from error
    if (not math.isfinite(converted)
            or (converted <= 0. if positive else converted < 0.)):
        raise ValueError(message)
    return converted


def _strict_reference(reference):
    if type(reference) is not GeometricSurfaceState or reference.phase != "partition":
        raise ValueError("Intrinsic geometry requires a strict resolved reference partition")
    _real(reference.time_myr, "Reference clock")
    _real(reference.radius_km, "Reference radius", positive=True)
    # Revalidate before trusting an object, including one made through object
    # mutation/deserialization.  No uncertainty allowance enters this import.
    fragments = tuple(GeometricFragment(fragment.fragment_id, fragment.polygon,
        SurfaceParcel(**asdict(fragment.parcel)), fragment.parent_fragment_id)
        for fragment in reference.fragments)
    checked = GeometricSurfaceState(reference.time_myr, reference.radius_km,
        fragments, reference.known_material_ids, reference.phase, reference.version)
    audit_partition(checked)
    return checked


@dataclass(frozen=True, slots=True)
class IntrinsicFragment:
    fragment_id: str
    geometry: IntrinsicGeometry
    parcel: SurfaceParcel
    parent_fragment_id: str | None = None


class IntrinsicSurfaceState:
    """A material partition derived from a reference and one common SO(3) pose.

    Current clocks, ages, footprints and reservoirs are derived, so a loader
    cannot accept an unrelated current material snapshot merely by hashing it.
    """

    # Intentionally opaque to generic dataclass writers.  The ordinary legacy
    # checkpoint writer uses asdict before opening a path; accepting this state
    # there would otherwise create an unreadable checkpoint with the old label.
    __slots__ = ("reference_surface", "rotations", "version", "_fragments", "_time_myr")

    def __init__(self, reference_surface, rotations=(), version=VERSION):
        object.__setattr__(self, "reference_surface", reference_surface)
        object.__setattr__(self, "rotations", rotations)
        object.__setattr__(self, "version", version)
        self.__post_init__()

    def __setattr__(self, name, value):
        raise AttributeError("Intrinsic surface and its material pose are immutable")

    def __delattr__(self, name):
        raise AttributeError("Intrinsic surface and its material pose are immutable")

    def __eq__(self, other):
        if type(other) is not IntrinsicSurfaceState:
            return NotImplemented
        return all(getattr(self, name) == getattr(other, name) for name in self.__slots__)

    def __post_init__(self):
        if self.version != VERSION:
            raise ValueError("Unsupported intrinsic surface version")
        reference = _strict_reference(self.reference_surface)
        try:
            history = tuple(self.rotations)
        except TypeError as error:
            raise ValueError("Intrinsic pose requires explicit rotation increments") from error
        if any(type(increment) is not RotationIncrement for increment in history):
            raise ValueError("Intrinsic pose requires explicit rotation increments")
        # Reconstruct to validate even a forged dataclass instance.
        history = tuple(RotationIncrement(increment.omega_rad_per_myr, increment.dt_myr)
                        for increment in history)
        time = reference.time_myr
        for increment in history:
            following = time + increment.dt_myr
            if not math.isfinite(following) or following <= time:
                raise ValueError("Intrinsic rotation must advance the physical clock")
            time = following
        fragments = []
        for original in reference.fragments:
            geometry = IntrinsicGeometry(original.polygon, history)
            if not math.isclose(geometry.area_sr * reference.radius_km**2,
                                original.parcel.area_km2,
                                rel_tol=AREA_REL_TOL, abs_tol=0.):
                raise ValueError("Intrinsic material area and reference footprint disagree")
            age = original.parcel.age_myr
            for increment in history:
                age += increment.dt_myr
                if not math.isfinite(age):
                    raise ValueError("Intrinsic material age must remain finite")
            parcel = replace(original.parcel, age_myr=age)
            fragments.append(IntrinsicFragment(original.fragment_id, geometry, parcel,
                                               original.parent_fragment_id))
        object.__setattr__(self, "reference_surface", reference)
        object.__setattr__(self, "rotations", history)
        object.__setattr__(self, "_fragments", tuple(fragments))
        object.__setattr__(self, "_time_myr", time)

    @property
    def time_myr(self):
        return self._time_myr

    @property
    def radius_km(self):
        return self.reference_surface.radius_km

    @property
    def fragments(self):
        return self._fragments

    @property
    def known_material_ids(self):
        return self.reference_surface.known_material_ids

    @property
    def phase(self):
        return "partition"


def enroll_intrinsic_surface(surface):
    """Explicitly adopt validated saved rays as the intrinsic reference epoch."""
    return IntrinsicSurfaceState(surface)


def advance_intrinsic_surface(state, omega_rad_per_myr, dt_myr):
    """Accept a common physical rigid step, preserving every material measure.

    No source, mechanical work, cooling, birth or loss is invented here.  This
    operation is a geometry control, distinct from a loaded underthrust step.
    """
    if type(state) is not IntrinsicSurfaceState:
        raise ValueError("Intrinsic advance requires an explicit intrinsic surface")
    checked = IntrinsicSurfaceState(state.reference_surface, state.rotations, state.version)
    if checked != state:
        raise ValueError("Intrinsic derived material state disagrees with its reference and pose")
    dt = _real(dt_myr, "Intrinsic timestep", positive=True)
    if not isinstance(omega_rad_per_myr, Mapping):
        raise ValueError("Intrinsic motion requires an explicit material-owner mapping")
    omega = compact_rotation_vectors(omega_rad_per_myr,
        (fragment.parcel.plate for fragment in state.fragments))
    owners = tuple(omega)
    vector = omega[owners[0]]
    if any(not np.array_equal(vector, omega[owner]) for owner in owners[1:]):
        raise ValueError("Differential intrinsic motion requires native clipping and cut lineage")
    increment = RotationIncrement(tuple(map(float, vector)), dt)
    return IntrinsicSurfaceState(state.reference_surface, state.rotations + (increment,))


def intrinsic_surface_to_dict(state):
    if type(state) is not IntrinsicSurfaceState:
        raise ValueError("Expected an explicit intrinsic surface")
    # Validate before creating/writing a destination or returning an envelope.
    checked = IntrinsicSurfaceState(state.reference_surface, state.rotations, state.version)
    if checked != state:
        raise ValueError("Intrinsic derived material state disagrees with its reference and pose")
    reference = json.loads(_encoded(asdict(checked.reference_surface)))
    return dict(version=checked.version, reference_surface=reference,
                rotations=[rotation_increment_to_dict(step) for step in checked.rotations])


def intrinsic_surface_from_dict(raw):
    if (not isinstance(raw, dict)
            or set(raw) != {"version", "reference_surface", "rotations"}
            or raw["version"] != VERSION or not isinstance(raw["rotations"], list)):
        raise ValueError("Invalid intrinsic surface schema or version")
    reference = raw["reference_surface"]
    if not isinstance(reference, dict):
        raise ValueError("Invalid intrinsic reference schema")
    _real(reference.get("time_myr"), "Reference clock")
    _real(reference.get("radius_km"), "Reference radius", positive=True)
    # The legacy reader validates its exact schema; additionally reject type
    # coercions in this new explicit format (strings/bools are not physics).
    rows = reference.get("fragments")
    if not isinstance(rows, list):
        raise ValueError("Intrinsic reference requires explicit fragment arrays")
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("parcel"), dict):
            raise ValueError("Invalid intrinsic reference material schema")
        parcel = row["parcel"]
        for key in (*EXTENSIVE_FIELDS, "age_myr"):
            _real(parcel.get(key), "Reference material " + key,
                  positive=key in ("area_km2", "oceanic_volume_km3"))
        properties = parcel.get("specific_properties")
        if not isinstance(properties, list) or len(properties) != 3:
            raise ValueError("Intrinsic reference requires three explicit specific properties")
        for value in properties:
            _real(value, "Reference specific property")
        material_fields = parcel.get("material_fields")
        if not isinstance(material_fields, list):
            raise ValueError("Intrinsic reference requires explicit material fields")
        for pair in material_fields:
            if (not isinstance(pair, list) or len(pair) != 2
                    or not isinstance(pair[0], str) or not pair[0]
                    or isinstance(pair[1], bool) or not isinstance(pair[1], Real)):
                raise ValueError("Intrinsic material fields require named finite real values")
            try:
                finite = math.isfinite(float(pair[1]))
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError("Intrinsic material fields require finite real values") from error
            if not finite:
                raise ValueError("Intrinsic material fields require finite real values")
    from .geometric_bending_reference import _surface_from_dict
    original = _surface_from_dict(reference)
    return IntrinsicSurfaceState(original, tuple(rotation_increment_from_dict(row)
                                               for row in raw["rotations"]), raw["version"])


def _encoded(payload):
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def save_intrinsic_checkpoint(path, state, *, provenance=None):
    payload = dict(format=FORMAT, state=intrinsic_surface_to_dict(state),
                   provenance={} if provenance is None else provenance)
    payload["payload_sha256"] = hashlib.sha256(_encoded(payload)).hexdigest()
    encoded = _encoded(payload) + b"\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(encoded)
    return path


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Intrinsic checkpoint contains duplicate JSON keys")
        result[key] = value
    return result


def load_intrinsic_checkpoint(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"),
                         object_pairs_hook=_unique_object)
    if (not isinstance(payload, dict)
            or set(payload) != {"format", "state", "provenance", "payload_sha256"}
            or payload["format"] != FORMAT):
        raise ValueError("Expected an intrinsic surface checkpoint schema")
    saved = payload.pop("payload_sha256")
    if saved != hashlib.sha256(_encoded(payload)).hexdigest():
        raise ValueError("Intrinsic checkpoint integrity mismatch")
    return intrinsic_surface_from_dict(payload["state"]), payload["provenance"]
