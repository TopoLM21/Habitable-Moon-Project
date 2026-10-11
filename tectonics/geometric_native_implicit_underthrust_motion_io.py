"""Exclusive joint restart of temporal implicit material and original forces."""
from __future__ import annotations

from pathlib import Path
import hashlib
import json

from .geometric_native_contact_reference_transport import ContactHistoryTransportLaw
from .geometric_native_first_entry_material import native_first_entry_material_from_dict
from .geometric_native_implicit_material_epoch import native_implicit_material_epoch_to_dict
from .geometric_native_implicit_underthrust_motion import (
    native_implicit_underthrust_motion_from_dict, native_implicit_underthrust_motion_to_dict,
)
from .geometric_native_mantle_loading import native_spatial_mantle_load_from_dict
from .geometric_native_physical_history import native_physical_history_binding_from_dict
from .geometric_native_underthrust_motion import (
    UnresolvedNativeUnderthrustMotion, advance_native_underthrust_motion,
    native_underthrust_motion_from_dict,
)
from .geometric_native_underthrust_motion_io import (
    _nonfinite_constant, _origin_from_dict, _unique_object, _write_exclusive,
)


FORMAT = "native-implicit-temporal-underthrust-joint-checkpoint-1"


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()


def save_native_implicit_underthrust_motion_checkpoint(path, state, *, provenance=None):
    """Publish only a fully authenticated loaded state and exact material epoch."""
    if provenance is not None and type(provenance) is not dict:
        raise ValueError("Implicit joint checkpoint provenance must be an explicit JSON object")
    motion = native_implicit_underthrust_motion_to_dict(state)
    payload = dict(format=FORMAT, implicit_underthrust_motion=motion,
        current_material_epoch=native_implicit_material_epoch_to_dict(state.current_surface),
        provenance={} if provenance is None else provenance)
    payload["payload_sha256"] = hashlib.sha256(_encoded(payload)).hexdigest()
    return _write_exclusive(path, _encoded(payload)+b"\n")


def load_native_implicit_underthrust_motion_checkpoint(path, *, mesh):
    """Re-solve the source, masks, field, clamp and mechanical work before loading."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=_unique_object, parse_constant=_nonfinite_constant)
    if (type(raw) is not dict or set(raw) != {"format", "implicit_underthrust_motion",
            "current_material_epoch", "provenance", "payload_sha256"}
            or raw["format"] != FORMAT or type(raw["provenance"]) is not dict):
        raise ValueError("Expected a complete explicit implicit joint checkpoint schema")
    payload = {key: value for key, value in raw.items() if key != "payload_sha256"}
    if raw["payload_sha256"] != hashlib.sha256(_encoded(payload)).hexdigest():
        raise ValueError("Implicit joint checkpoint integrity mismatch")
    try:
        motion = raw["implicit_underthrust_motion"]
        source = motion["original_physical_history"]
        origin = _origin_from_dict(source["origin_surface"])
        before = native_underthrust_motion_from_dict(source["accepted_before"], mesh, origin)
        load = native_spatial_mantle_load_from_dict(source["current_load"],
            before.current_surface, mesh=mesh)
        recorded = source["actual_candidate"]
        try:
            advance_native_underthrust_motion(mesh, origin, before, load,
                recorded["dt_s"], provenance=recorded["provenance"])
        except UnresolvedNativeUnderthrustMotion as refusal:
            candidate = refusal.candidate
        else:
            raise ValueError("Original solved evidence changed its authenticated spatial contract")
        material = native_first_entry_material_from_dict(source["material_state"])
        history = native_physical_history_binding_from_dict(source, mesh, origin,
            before, candidate, material, current_load=load,
            original_checkpoint_sha256=source["original_checkpoint_sha256"])
        law = ContactHistoryTransportLaw(**motion["transport_law"])
        state = native_implicit_underthrust_motion_from_dict(motion,
            expected_history=history, expected_source_history_dto=source, expected_law=law)
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("Implicit joint checkpoint lacks its complete force and material source") from error
    if _encoded(raw["current_material_epoch"]) != _encoded(
            native_implicit_material_epoch_to_dict(state.current_surface)):
        raise ValueError("Joint implicit material epoch differs from complete loaded replay")
    return origin, state, raw["provenance"]
