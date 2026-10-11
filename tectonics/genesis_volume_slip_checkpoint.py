"""Strict, replay-validated saves of actual Genesis finite-width plane slip.

The inherited plane is a constitutive shear band in connected material. Its
world directions and separate physical-work ledgers belong to the prospective
history; neither a checksum nor an editable state can supply that history.
Only the complete state rebuilt from the embedded Genesis source is returned.
"""
from __future__ import annotations

from dataclasses import fields
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from .genesis_bulk_birth import bulk_birth_to_dict
from .genesis_contact import ContactParameters
from .genesis_volume_checkpoint import (
    _SPECIAL, _canonical, _geometry_claims, _jsonable,
    _reject_duplicate_keys, _reject_nonfinite_constant, _validate_runtime_schema)
from .genesis_volume_dynamics import VolumeState
from .genesis_volume_fault_dynamics import DAMAGE_COUPLING
from .genesis_volume_fault_evolution import VolumeFaultHistory, volume_fault_history_to_dict
from .genesis_volume_loading import volume_mechanics_to_dict
from .genesis_volume_slip_dynamics import (
    MECHANICAL_WORK_CLOSURE, PLANE_TRANSPORT, SLIP_CLOSURE,
    VolumeSlipModel, VolumeSlipState)
from .genesis_volume_temporal import cone_homothetic_to_dict
from .genesis_volume_thermal import thermal_to_dict


FORMAT = "genesis-volume-slip-checkpoint-0.1"
SCOPE = "actual_genesis_source_replayed_connected_volume_finite_width_weak_plane_slip"
_TOP_FIELDS = {"format", "scope", "source_packet", "source_checkpoint_sha256",
               "runtime_state", "checkpoint_sha256"}
_CLOSURES = {
    "damage_coupling": DAMAGE_COUPLING,
    "slip_closure": SLIP_CLOSURE,
    "plane_transport": PLANE_TRANSPORT,
    "mechanical_work_closure": MECHANICAL_WORK_CLOSURE,
}
_LEDGERS = {"physical_mechanical_work_j", "fault_friction_dissipation_j",
            "fault_viscous_dissipation_j"}
_PLANES = {"plane_normal_world", "slip_direction_world"}


def _slip_state_payload(state):
    """Include every runtime field; encode source/history only once."""
    result = {f.name: _jsonable(getattr(state, f.name))
              for f in fields(VolumeSlipState) if f.name not in _SPECIAL | {"fault_history"}}
    result.update(
        geometry=_geometry_claims(state.geometry),
        thermal=thermal_to_dict(state.thermal, include_source=False),
        history=bulk_birth_to_dict(state.history),
        mechanics=volume_mechanics_to_dict(state.mechanics),
        last_certificate=(None if state.last_certificate is None
                          else cone_homothetic_to_dict(state.last_certificate)),
        fault_history=volume_fault_history_to_dict(state.fault_history),
    )
    return _jsonable(result)


def volume_slip_checkpoint_to_dict(model, state):
    """Seal a validated state without changing memory, planes or ledgers."""
    if type(model) is not VolumeSlipModel or type(state) is not VolumeSlipState:
        raise ValueError("A source-bound volume slip model and state are required")
    model.validate_state(state)
    body = _jsonable(dict(
        format=FORMAT, scope=SCOPE, source_packet=model.source_packet,
        source_checkpoint_sha256=state.geometry.source_checkpoint_sha256,
        runtime_state=_slip_state_payload(state)))
    return {**body, "checkpoint_sha256": hashlib.sha256(_canonical(body)).hexdigest()}


def _slip_runtime_schema(model, runtime):
    expected = {f.name for f in fields(VolumeSlipState)}
    if type(runtime) is not dict or set(runtime) != expected:
        raise ValueError("Volume slip checkpoint runtime fields mismatch")
    for name, value in _CLOSURES.items():
        if type(runtime[name]) is not str or runtime[name] != value:
            raise ValueError(f"Volume slip checkpoint {name} changed its physical semantics")
    for name in _LEDGERS:
        if type(runtime[name]) not in (int, float) or not np.isfinite(runtime[name]):
            raise ValueError(f"Volume slip checkpoint {name} must be a finite real")
    nface = len(model.source_operator.geometry.faces)
    for name in _PLANES:
        value = runtime[name]
        if (type(value) is not list or len(value) != nface
                or any(type(row) is not list or len(row) != 3
                       or any(type(x) not in (int, float) or not np.isfinite(x) for x in row)
                       for row in value)):
            raise ValueError(f"Volume slip checkpoint {name} must be a finite face vector array")
    base_names = {f.name for f in fields(VolumeState)}
    sequence = _validate_runtime_schema(model, {name: runtime[name] for name in base_names})
    history = runtime["fault_history"]
    if type(history) is not dict or set(history) != {f.name for f in fields(VolumeFaultHistory)} | {"format"}:
        raise ValueError("Volume slip checkpoint inherited fault history fields mismatch")
    # A JSON schema check is not provisional trust in the candidate's physics.
    # Full literal values, including inherited work offsets, must equal replay.
    _canonical(history)
    return sequence


def volume_slip_checkpoint_from_dict(raw, *, expected_source_checkpoint_sha256=None):
    """Rebuild every accepted interval and return only the rebuilt state."""
    if type(raw) is not dict or set(raw) != _TOP_FIELDS:
        raise ValueError("Volume slip checkpoint top-level fields mismatch")
    body = {k: v for k, v in raw.items() if k != "checkpoint_sha256"}
    if (raw["format"] != FORMAT or raw["scope"] != SCOPE
            or raw["checkpoint_sha256"] != hashlib.sha256(_canonical(body)).hexdigest()):
        raise ValueError("Volume slip checkpoint format, scope or seal mismatch")
    pin, source = raw["source_checkpoint_sha256"], raw["source_packet"]
    if (type(pin) is not str or len(pin) != 64 or any(c not in "0123456789abcdef" for c in pin)
            or (expected_source_checkpoint_sha256 is not None and expected_source_checkpoint_sha256 != pin)
            or type(source) is not dict or type(source.get("source")) is not dict
            or source["source"].get("checkpoint_sha256") != pin):
        raise ValueError("Volume slip checkpoint source binding mismatch")
    runtime = raw["runtime_state"]
    if type(runtime) is not dict or type(runtime.get("contact_parameters_json")) is not str:
        raise ValueError("Volume slip checkpoint contact parameters are missing")
    try:
        p = json.loads(runtime["contact_parameters_json"], object_pairs_hook=_reject_duplicate_keys,
                       parse_constant=_reject_nonfinite_constant)
        if type(p) is not dict or set(p) != {f.name for f in fields(ContactParameters)}:
            raise ValueError("Volume slip checkpoint contact fields mismatch")
        contact = ContactParameters(**p)
        contact.validate()
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Volume slip checkpoint contact parameters invalid") from exc
    model = VolumeSlipModel(source, contact_parameters=contact)
    sequence = _slip_runtime_schema(model, runtime)
    replay = model.initial_state()
    for dt in sequence:
        replay = model.step(replay, dt)
    if _canonical(_slip_state_payload(replay)) != _canonical(runtime):
        raise ValueError("Volume slip checkpoint differs from full deterministic source-to-current replay")
    return model, replay


def save_volume_slip_checkpoint(path, model, state):
    """Write strict JSON atomically to a path, or to a writable text stream."""
    raw = volume_slip_checkpoint_to_dict(model, state)
    encoded = json.dumps(raw, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if hasattr(path, "write"):
        path.write(encoded)
    else:
        target, temporary = Path(path), None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=target.parent,
                    prefix=f".{target.name}.", suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return raw


def load_volume_slip_checkpoint(path, *, expected_source_checkpoint_sha256=None):
    """Read duplicate-free, finite JSON and replay its full prospective history."""
    encoded = path.read() if hasattr(path, "read") else Path(path).read_text(encoding="utf-8")
    raw = json.loads(encoded, object_pairs_hook=_reject_duplicate_keys,
                     parse_constant=_reject_nonfinite_constant)
    return volume_slip_checkpoint_from_dict(raw,
        expected_source_checkpoint_sha256=expected_source_checkpoint_sha256)
