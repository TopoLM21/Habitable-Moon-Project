"""Replay-validated saves of calculated volume damage and weak-plane timers."""
from __future__ import annotations

from dataclasses import fields
import hashlib
import json
import os
from pathlib import Path
import tempfile

from .genesis_bulk_birth import bulk_birth_to_dict
from .genesis_contact import ContactParameters
from .genesis_volume_checkpoint import (
    _ARRAYS, _SPECIAL, _canonical, _geometry_claims, _jsonable,
    _reject_duplicate_keys, _reject_nonfinite_constant, _validate_runtime_schema)
from .genesis_volume_dynamics import VolumeState
from .genesis_volume_fault_dynamics import DAMAGE_COUPLING, VolumeFaultModel, VolumeFaultState
from .genesis_volume_fault_evolution import VolumeFaultHistory, volume_fault_history_to_dict
from .genesis_volume_loading import volume_mechanics_to_dict
from .genesis_volume_temporal import cone_homothetic_to_dict
from .genesis_volume_thermal import thermal_to_dict


FORMAT = "genesis-volume-fault-checkpoint-0.1"
SCOPE = "actual_genesis_source_replayed_connected_volume_damage_until_weak_plane_gate"
_TOP_FIELDS = {"format", "scope", "source_packet", "source_checkpoint_sha256",
               "runtime_state", "checkpoint_sha256"}


def _fault_state_payload(state):
    result = {f.name: _jsonable(getattr(state, f.name))
              for f in fields(VolumeState) if f.name not in _SPECIAL}
    result.update(geometry=_geometry_claims(state.geometry),
        thermal=thermal_to_dict(state.thermal, include_source=False),
        history=bulk_birth_to_dict(state.history),
        mechanics=volume_mechanics_to_dict(state.mechanics),
        last_certificate=(None if state.last_certificate is None
                          else cone_homothetic_to_dict(state.last_certificate)),
        fault_history=volume_fault_history_to_dict(state.fault_history),
        damage_coupling=state.damage_coupling)
    return _jsonable(result)


def volume_fault_checkpoint_to_dict(model, state):
    if type(model) is not VolumeFaultModel or type(state) is not VolumeFaultState:
        raise ValueError("A source-bound volume fault model and state are required")
    model.validate_state(state)
    body = _jsonable(dict(format=FORMAT, scope=SCOPE, source_packet=model.source_packet,
        source_checkpoint_sha256=state.geometry.source_checkpoint_sha256,
        runtime_state=_fault_state_payload(state)))
    return {**body, "checkpoint_sha256": hashlib.sha256(_canonical(body)).hexdigest()}


def _fault_runtime_schema(model, runtime):
    expected = {f.name for f in fields(VolumeFaultState)}
    if type(runtime) is not dict or set(runtime) != expected:
        raise ValueError("Volume fault checkpoint runtime fields mismatch")
    if type(runtime["damage_coupling"]) is not str or runtime["damage_coupling"] != DAMAGE_COUPLING:
        raise ValueError("Volume fault checkpoint changed its damage semantics")
    base_names = {f.name for f in fields(VolumeState)}
    sequence = _validate_runtime_schema(model, {name: runtime[name] for name in base_names})
    history = runtime["fault_history"]
    if type(history) is not dict or set(history) != {f.name for f in fields(VolumeFaultHistory)} | {"format"}:
        raise ValueError("Volume fault checkpoint inherited history schema mismatch")
    # Validate literal finite JSON without constructing an untrusted candidate
    # physics/history. The complete field values must subsequently equal replay.
    _canonical(history)
    return sequence


def volume_fault_checkpoint_from_dict(raw, *, expected_source_checkpoint_sha256=None):
    if type(raw) is not dict or set(raw) != _TOP_FIELDS:
        raise ValueError("Volume fault checkpoint top-level fields mismatch")
    body = {k: v for k, v in raw.items() if k != "checkpoint_sha256"}
    if (raw["format"] != FORMAT or raw["scope"] != SCOPE
            or raw["checkpoint_sha256"] != hashlib.sha256(_canonical(body)).hexdigest()):
        raise ValueError("Volume fault checkpoint format, scope or seal mismatch")
    pin = raw["source_checkpoint_sha256"]
    source = raw["source_packet"]
    if (type(pin) is not str or len(pin) != 64 or any(c not in "0123456789abcdef" for c in pin)
            or (expected_source_checkpoint_sha256 is not None and expected_source_checkpoint_sha256 != pin)
            or type(source) is not dict or type(source.get("source")) is not dict
            or source["source"].get("checkpoint_sha256") != pin):
        raise ValueError("Volume fault checkpoint source binding mismatch")
    runtime = raw["runtime_state"]
    if type(runtime) is not dict or type(runtime.get("contact_parameters_json")) is not str:
        raise ValueError("Volume fault checkpoint contact parameters are missing")
    try:
        p = json.loads(runtime["contact_parameters_json"], object_pairs_hook=_reject_duplicate_keys,
                       parse_constant=_reject_nonfinite_constant)
        if type(p) is not dict or set(p) != {f.name for f in fields(ContactParameters)}:
            raise ValueError("Volume fault checkpoint contact fields mismatch")
        contact = ContactParameters(**p)
        contact.validate()
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Volume fault checkpoint contact parameters invalid") from exc
    model = VolumeFaultModel(source, contact_parameters=contact)
    sequence = _fault_runtime_schema(model, runtime)
    replay = model.initial_state()
    for dt in sequence:
        replay = model.step(replay, dt)
    if _canonical(_fault_state_payload(replay)) != _canonical(runtime):
        raise ValueError("Volume fault checkpoint differs from full deterministic source-to-current replay")
    return model, replay


def save_volume_fault_checkpoint(path, model, state):
    raw = volume_fault_checkpoint_to_dict(model, state)
    encoded = json.dumps(raw, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)+'\n'
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


def load_volume_fault_checkpoint(path, *, expected_source_checkpoint_sha256=None):
    encoded = path.read() if hasattr(path, "read") else Path(path).read_text(encoding="utf-8")
    raw = json.loads(encoded, object_pairs_hook=_reject_duplicate_keys,
                     parse_constant=_reject_nonfinite_constant)
    return volume_fault_checkpoint_from_dict(raw,
        expected_source_checkpoint_sha256=expected_source_checkpoint_sha256)
