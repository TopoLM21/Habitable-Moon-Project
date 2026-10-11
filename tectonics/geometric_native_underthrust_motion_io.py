"""Exclusive joint checkpoints for force-driven native material support.

The initial legacy epoch is an explicit mechanical seed.  The current surface
and buried registry use native geometry throughout; no projected world polygon
is stored as an authoritative current material domain.
"""
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

from .geometric_bending_reference import _surface_from_dict
from .geometric_native_buried_support import (
    native_buried_registry_from_dict, native_buried_registry_to_dict,
)
from .geometric_native_surface import (
    _parcel_from_dict, enroll_native_surface, native_surface_from_dict,
    native_surface_to_dict,
)
from .geometric_surface import GeometricSurfaceState


FORMAT = "geometric-native-force-generated-underthrust-checkpoint-1"
_KEYS = {"format", "origin_surface", "native_underthrust_motion", "current_surface",
         "current_inventory", "provenance", "payload_sha256"}


def _encoded(payload):
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Native underthrust checkpoint contains duplicate JSON keys")
        result[key] = value
    return result


def _nonfinite_constant(value):
    raise ValueError("Native underthrust checkpoint contains a nonfinite JSON number")


def _origin_from_dict(raw):
    """Read the original seed without numeric coercion or polygon rewriting."""
    if not isinstance(raw, dict):
        raise ValueError("Native underthrust checkpoint needs an explicit original seed")
    for key in ("time_myr", "radius_km"):
        value = raw.get(key)
        try:
            valid = (type(value) in (int, float) and math.isfinite(value)
                     and (value > 0. if key == "radius_km" else value >= 0.))
        except (TypeError, ValueError, OverflowError):
            valid = False
        if not valid:
            raise ValueError("Native underthrust original clock/radius requires explicit physical numbers")
    fragments = raw.get("fragments")
    if not isinstance(fragments, list):
        raise ValueError("Native underthrust original material requires explicit fragments")
    for row in fragments:
        if not isinstance(row, dict):
            raise ValueError("Native underthrust original fragment schema mismatch")
        _parcel_from_dict(row.get("parcel"))
    origin = _surface_from_dict(raw)
    if _encoded(asdict(origin)) != _encoded(raw):
        raise ValueError("Native underthrust original seed changed during typed reconstruction")
    # Native admission audits the exact saved rays rather than inheriting the
    # floating partition's tolerance contract.
    enroll_native_surface(origin)
    return origin


def _write_exclusive(path, encoded):
    """Publish complete bytes atomically, retaining any existing destination."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent,
                prefix="." + path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def save_native_underthrust_motion_checkpoint(path, mesh, origin_surface, state, *,
                                             provenance=None):
    """Validate mechanics and the native current pair before touching a path."""
    if provenance is not None and not isinstance(provenance, dict):
        raise ValueError("Native underthrust checkpoint provenance must be a JSON object")
    _encoded({} if provenance is None else provenance)
    if type(origin_surface) is not GeometricSurfaceState:
        raise TypeError("Native underthrust checkpoint requires its explicit original legacy seed")
    origin_raw = json.loads(_encoded(asdict(origin_surface)))
    _origin_from_dict(origin_raw)
    from .geometric_native_underthrust_motion import native_underthrust_motion_to_dict

    motion = native_underthrust_motion_to_dict(mesh, origin_surface, state)
    payload = dict(format=FORMAT, origin_surface=origin_raw,
        native_underthrust_motion=motion,
        current_surface=native_surface_to_dict(state.current_surface),
        current_inventory=native_buried_registry_to_dict(state.current_inventory),
        provenance={} if provenance is None else provenance)
    payload["payload_sha256"] = hashlib.sha256(_encoded(payload)).hexdigest()
    return _write_exclusive(path, _encoded(payload) + b"\n")


def load_native_underthrust_motion_checkpoint(path, *, mesh):
    """Replay every physical source interval and audit the redundant joint pair."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=_unique_object, parse_constant=_nonfinite_constant)
    if (not isinstance(payload, dict) or set(payload) != _KEYS
            or payload["format"] != FORMAT or not isinstance(payload["provenance"], dict)):
        raise ValueError("Expected an explicit native underthrust checkpoint schema")
    saved = payload.pop("payload_sha256")
    if (not isinstance(saved, str) or len(saved) != 64
            or saved != hashlib.sha256(_encoded(payload)).hexdigest()):
        raise ValueError("Native underthrust checkpoint integrity mismatch")
    origin = _origin_from_dict(payload["origin_surface"])
    current = native_surface_from_dict(payload["current_surface"])
    inventory = native_buried_registry_from_dict(payload["current_inventory"],
        enroll_native_surface(origin), current)
    from .geometric_native_underthrust_motion import native_underthrust_motion_from_dict

    state = native_underthrust_motion_from_dict(payload["native_underthrust_motion"], mesh, origin)
    if (_encoded(payload["current_surface"]) != _encoded(native_surface_to_dict(state.current_surface))
            or _encoded(payload["current_inventory"]) != _encoded(
                native_buried_registry_to_dict(state.current_inventory))
            or _encoded(payload["current_surface"]) != _encoded(native_surface_to_dict(current))
            or _encoded(payload["current_inventory"]) != _encoded(native_buried_registry_to_dict(inventory))):
        raise ValueError("Native checkpoint current surface/archive differs from complete physical replay")
    return origin, state, payload["provenance"]
