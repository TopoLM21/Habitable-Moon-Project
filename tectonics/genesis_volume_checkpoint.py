"""Replay-validated checkpoints of the actual Genesis volume continuation.

A checksum detects accidental byte changes; it does not authenticate a
constitutive history. Loading therefore repeats every accepted mechanical
step from the embedded, strictly validated Genesis source and compares the
complete runtime state. Jointly resealing a changed strain and energy/work
ledger cannot turn that changed state into a continuation.
"""
from __future__ import annotations

from dataclasses import asdict, fields
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from .genesis_bulk_birth import bulk_birth_to_dict
from .genesis_contact import ContactParameters
from .genesis_volume_loading import volume_mechanics_to_dict
from .genesis_volume_temporal import cone_homothetic_to_dict


FORMAT = "genesis-volume-continuation-checkpoint-0.1"
SCOPE = "actual_genesis_source_and_replayed_prospective_volume_continuation"
_SPECIAL = {"geometry", "thermal", "history", "mechanics", "last_certificate"}
_ARRAYS = {"displacement_m", "velocity_m_s"}
_TOP_FIELDS = {"format", "scope", "source_packet", "source_checkpoint_sha256",
               "runtime_state", "checkpoint_sha256"}


def _jsonable(value):
    if isinstance(value, np.ndarray):
        if not np.isfinite(value).all():
            raise ValueError("Volume checkpoint contains a nonfinite array")
        return value.tolist()
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, dict):
        if any(type(key) is not str for key in value):
            raise ValueError("Volume checkpoint object keys must be strings")
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) is float and np.isfinite(value):
        return value
    raise ValueError("Volume checkpoint contains a non-JSON value")


def _canonical(value):
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _geometry_claims(geometry):
    claims = asdict(geometry)
    # The producer bytes and runtime birth history occur only once in the
    # checkpoint. All placement/ownership/mass claims are independently rebuilt.
    claims.pop("source_packet_json")
    claims.pop("runtime_birth_history_json")
    return _jsonable(claims)


def _state_payload(state):
    from .genesis_volume_thermal import thermal_to_dict
    result = {field.name: _jsonable(getattr(state, field.name))
              for field in fields(state) if field.name not in _SPECIAL}
    result.update(
        geometry=_geometry_claims(state.geometry),
        thermal=thermal_to_dict(state.thermal, include_source=False),
        history=bulk_birth_to_dict(state.history),
        mechanics=volume_mechanics_to_dict(state.mechanics),
        last_certificate=(None if state.last_certificate is None
                          else cone_homothetic_to_dict(state.last_certificate)),
    )
    return _jsonable(result)


def volume_checkpoint_to_dict(model, state):
    """Archive a mathematically validated current state without resetting it."""
    from .genesis_volume_dynamics import VolumeModel, VolumeState
    if type(model) is not VolumeModel or type(state) is not VolumeState:
        raise ValueError("A source-bound VolumeModel and VolumeState are required")
    model.validate_state(state)
    body = dict(format=FORMAT, scope=SCOPE, source_packet=model.source_packet,
                source_checkpoint_sha256=state.geometry.source_checkpoint_sha256,
                runtime_state=_state_payload(state))
    body = _jsonable(body)
    return {**body, "checkpoint_sha256": hashlib.sha256(_canonical(body)).hexdigest()}


def _finite_vector(raw, size, name):
    if (type(raw) is not list or len(raw) != size
            or any(type(x) not in (int, float) for x in raw)):
        raise ValueError(f"Invalid volume checkpoint {name} array")
    value = np.asarray(raw, dtype=float)
    if not np.isfinite(value).all():
        raise ValueError(f"Nonfinite volume checkpoint {name} array")
    return value


def _validate_runtime_schema(model, raw):
    from .genesis_volume_dynamics import ACTIVATION, KINEMATICS, VELOCITY_CLOSURE, VolumeState
    expected = {field.name for field in fields(VolumeState)}
    if type(raw) is not dict or set(raw) != expected:
        raise ValueError("Volume checkpoint runtime state fields mismatch")
    for name in _ARRAYS:
        _finite_vector(raw[name], model.source_operator.ndof, name)
    semantics = dict(velocity_closure=VELOCITY_CLOSURE, kinematics=KINEMATICS,
                     activation_closure=ACTIVATION)
    for name, value in semantics.items():
        if type(raw[name]) is not str or raw[name] != value:
            raise ValueError(f"Volume checkpoint {name} changed its physical semantics")
    for name in expected-_SPECIAL-_ARRAYS-set(semantics)-{
            "contact_parameters_json", "last_report", "accepted_step_sizes_myr"}:
        if type(raw[name]) not in (int, float) or not np.isfinite(raw[name]):
            raise ValueError(f"Volume checkpoint {name} must be a finite real")
    for name in _SPECIAL-{"last_certificate"}:
        if type(raw[name]) is not dict:
            raise ValueError(f"Volume checkpoint {name} must be a complete object")
    if raw["last_certificate"] is not None and type(raw["last_certificate"]) is not dict:
        raise ValueError("Volume checkpoint temporal certificate must be an object")
    if raw["last_report"] is not None and type(raw["last_report"]) is not dict:
        raise ValueError("Volume checkpoint step report must be an object")
    step_sizes = raw["accepted_step_sizes_myr"]
    if (type(step_sizes) is not list
            or any(type(dt) not in (int, float) or not np.isfinite(dt) or dt <= 0
                   for dt in step_sizes)):
        raise ValueError("Volume checkpoint accepted mechanical step sequence is invalid")
    return tuple(step_sizes)


def volume_checkpoint_from_dict(raw, *, expected_source_checkpoint_sha256=None):
    """Load only a complete deterministic continuation of the embedded source.

    All source-backed geometry, thermal history, constitutive memory, clocks,
    motion, balances, reports and temporal certificates must equal replay.
    """
    from .genesis_volume_dynamics import VolumeModel
    if type(raw) is not dict or set(raw) != _TOP_FIELDS:
        raise ValueError("Volume checkpoint top-level fields mismatch")
    body = {name: raw[name] for name in _TOP_FIELDS-{"checkpoint_sha256"}}
    if (raw["format"] != FORMAT or raw["scope"] != SCOPE
            or raw["checkpoint_sha256"] != hashlib.sha256(_canonical(body)).hexdigest()):
        raise ValueError("Volume checkpoint format, scope or seal mismatch")
    pin = raw["source_checkpoint_sha256"]
    source_metadata = (raw["source_packet"].get("source")
                       if type(raw["source_packet"]) is dict else None)
    if (type(pin) is not str or len(pin) != 64
            or any(c not in "0123456789abcdef" for c in pin)
            or (expected_source_checkpoint_sha256 is not None
                and pin != expected_source_checkpoint_sha256)
            or type(source_metadata) is not dict
            or source_metadata.get("checkpoint_sha256") != pin):
        raise ValueError("Volume checkpoint source byte binding mismatch")
    runtime = raw["runtime_state"]
    if type(runtime) is not dict or type(runtime.get("contact_parameters_json")) is not str:
        raise ValueError("Volume checkpoint contact parameters are missing")
    try:
        parameters = json.loads(runtime["contact_parameters_json"])
        if type(parameters) is not dict or set(parameters) != {f.name for f in fields(ContactParameters)}:
            raise ValueError("Contact parameter fields mismatch")
        contact = ContactParameters(**parameters)
        contact.validate()
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid volume checkpoint contact parameters") from exc
    model = VolumeModel(raw["source_packet"], contact_parameters=contact)
    # Archive constitutive/thermal objects never receive provisional trust.
    # Their complete serialized values are checked against actual replay.
    step_sizes = _validate_runtime_schema(model, runtime)
    replay = model.initial_state()
    for dt_myr in step_sizes:
        replay = model.step(replay, dt_myr)
    if _canonical(_state_payload(replay)) != _canonical(runtime):
        raise ValueError("Volume checkpoint differs from full deterministic source-to-current replay")
    # Return the actual replay, not objects assembled from untrusted ledgers.
    return model, replay


def save_volume_checkpoint(path, model, state):
    """Save the strict JSON packet to a path or writable text stream."""
    raw = volume_checkpoint_to_dict(model, state)
    encoded = json.dumps(raw, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)+"\n"
    if hasattr(path, "write"):
        path.write(encoded)
    else:
        target = Path(path)
        temporary = None
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


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate volume checkpoint JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value):
    raise ValueError(f"Nonfinite volume checkpoint JSON number: {value}")


def load_volume_checkpoint(path, *, expected_source_checkpoint_sha256=None):
    """Read a strict archive and replay its complete prospective history."""
    encoded = path.read() if hasattr(path, "read") else Path(path).read_text(encoding="utf-8")
    raw = json.loads(encoded, object_pairs_hook=_reject_duplicate_keys,
                     parse_constant=_reject_nonfinite_constant)
    return volume_checkpoint_from_dict(raw,
        expected_source_checkpoint_sha256=expected_source_checkpoint_sha256)
