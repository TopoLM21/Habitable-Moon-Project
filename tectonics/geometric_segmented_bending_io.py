"""Joint original surface/archive checkpoints with a replayed source ledger.

The core's reference configurations are source integration views. They are not
substituted for the genuine joint surface/archive pair. Accepted physical
material motion has its own archive transaction/checkpoint layer.
"""
from collections.abc import Mapping
import hashlib
import json

from .geometric_basal_sources import basal_source_from_dict, basal_source_to_dict
from .geometric_boundary import boundary_to_dict
from .geometric_boundary_io import load_boundary_checkpoint, save_boundary_checkpoint
from .geometric_segmented_bending_dynamics import (
    segmented_bending_dynamics_from_dict, segmented_bending_dynamics_to_dict,
    validate_segmented_bending_dynamics_state,
)


FORMAT = "geometric-segmented-bending-checkpoint-1"
_SOURCE_OPTIONS = {"mesh", "parameters", "thermal_lid_by_fragment", "provenance", "source"}


def _archive_hash(inventory):
    return hashlib.sha256(json.dumps(boundary_to_dict(inventory), sort_keys=True,
        separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def save_segmented_bending_checkpoint(path, origin_surface, inventory, state, *, provenance=None):
    """Save the original joint pair and all accepted source-segment receipts."""
    validate_segmented_bending_dynamics_state(origin_surface, state)
    if provenance is not None and not isinstance(provenance, dict):
        raise ValueError("Segmented checkpoint provenance must be a JSON object")
    metadata = dict(format=FORMAT, reference_inventory_sha256=_archive_hash(inventory),
        segmented_bending=segmented_bending_dynamics_to_dict(origin_surface, state),
        user_provenance={} if provenance is None else provenance)
    return save_boundary_checkpoint(path, origin_surface, inventory, provenance=metadata)


def _verify_sources(origin, state, source_inputs):
    if source_inputs is None:
        return
    if not isinstance(source_inputs, Mapping):
        raise ValueError("Trusted source inputs require a load-checksum mapping")
    loads = [(origin, state.initial_load)]
    loads.extend((receipt.reference_configuration.surface, receipt.current_load)
                 for receipt in state.segments)
    expected = {load.load_sha256 for _, load in loads}
    if set(source_inputs) != expected:
        raise ValueError("Trusted source inputs must cover every distinct saved load exactly")
    # A source may recur in multiple receipts; it is checked on every bound
    # configuration so a reused key never bypasses exact surface binding.
    for surface, load in loads:
        options = source_inputs[load.load_sha256]
        if not isinstance(options, Mapping) or not set(options) <= _SOURCE_OPTIONS:
            raise ValueError("Invalid trusted physical source reassembly options")
        rebuilt = basal_source_from_dict(json.loads(json.dumps(basal_source_to_dict(surface, load))),
                                        surface, **dict(options))
        if rebuilt != load:
            raise ValueError("Saved source differs from its trusted reassembly")


def load_segmented_bending_checkpoint(path, *, source_inputs=None):
    """Restore original material/archive and replay every mechanical segment.

    If supplied, ``source_inputs`` must cover all distinct ``load_sha256``
    values. Each value contains the explicit trusted kwargs of the basal
    source loader. The original quadratic source rebuilds operators when its
    mesh, parameters, lid and provenance are all supplied; rigid mantle
    loading rebuilds its full physical sample quadrature whenever mesh is
    supplied. Partial kwargs retain their narrower source-identity checks.
    """
    origin, inventory, metadata = load_boundary_checkpoint(path)
    expected = {"format", "reference_inventory_sha256", "segmented_bending", "user_provenance"}
    if (not isinstance(metadata, dict) or set(metadata) != expected or metadata["format"] != FORMAT
            or not isinstance(metadata["user_provenance"], dict)):
        raise ValueError("Expected explicit segmented bending checkpoint metadata")
    if metadata["reference_inventory_sha256"] != _archive_hash(inventory):
        raise ValueError("Segmented checkpoint original passive archive binding mismatch")
    state = segmented_bending_dynamics_from_dict(metadata["segmented_bending"], origin)
    _verify_sources(origin, state, source_inputs)
    return origin, inventory, state, metadata["user_provenance"]
