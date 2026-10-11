"""Restart the genuine current material pair and complete underthrust history."""
from dataclasses import asdict

from .geometric_bending_reference import _surface_from_dict
from .geometric_boundary_io import _encoded, load_boundary_checkpoint, save_boundary_checkpoint


FORMAT = "geometric-force-generated-underthrust-checkpoint-1"


def save_underthrust_motion_checkpoint(path, mesh, origin_surface, state, *, provenance=None):
    """Save actual current surface/archive with a replayable original epoch.

    Source and mechanical histories, material-bank support and geometric
    transactions are validated before the exclusive writer opens a file.
    """
    if provenance is not None and not isinstance(provenance, dict):
        raise ValueError("Physical underthrust checkpoint provenance must be a JSON object")
    from .geometric_underthrust_motion import (
        underthrust_motion_to_dict, validate_underthrust_motion_state,
    )
    validate_underthrust_motion_state(mesh, origin_surface, state)
    metadata = dict(format=FORMAT, origin_surface=asdict(origin_surface),
        underthrust_motion=underthrust_motion_to_dict(mesh, origin_surface, state),
        user_provenance={} if provenance is None else provenance)
    return save_boundary_checkpoint(path, state.current_surface, state.current_inventory,
                                    provenance=metadata)


def load_underthrust_motion_checkpoint(path, *, mesh):
    """Reproduce force, source, bank-support and material transactions."""
    from .geometric_underthrust_motion import underthrust_motion_from_dict
    current_surface, current_inventory, metadata = load_boundary_checkpoint(path)
    expected = {"format", "origin_surface", "underthrust_motion", "user_provenance"}
    if (not isinstance(metadata, dict) or set(metadata) != expected
            or metadata["format"] != FORMAT or not isinstance(metadata["user_provenance"], dict)):
        raise ValueError("Expected explicit accepted underthrust checkpoint metadata")
    origin = _surface_from_dict(metadata["origin_surface"])
    state = underthrust_motion_from_dict(metadata["underthrust_motion"], mesh, origin)
    if (_encoded(asdict(current_surface)) != _encoded(asdict(state.current_surface))
            or _encoded(asdict(current_inventory)) != _encoded(asdict(state.current_inventory))):
        raise ValueError("Checkpoint current surface/archive differs from underthrust trajectory replay")
    return origin, state, metadata["user_provenance"]
