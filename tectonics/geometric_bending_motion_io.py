"""Checkpoints of the actual moved surface/archive and complete force history."""
from dataclasses import asdict

from .geometric_bending_motion import (
    bending_motion_from_dict, bending_motion_to_dict, validate_bending_motion_state,
)
from .geometric_bending_reference import _surface_from_dict
from .geometric_boundary_io import load_boundary_checkpoint, save_boundary_checkpoint


FORMAT = "geometric-force-generated-bending-checkpoint-1"


def save_bending_motion_checkpoint(path, mesh, origin_surface, state, *, provenance=None):
    """Save the genuine current joint pair and its replayable original epoch."""
    validate_bending_motion_state(mesh, origin_surface, state)
    if provenance is not None and not isinstance(provenance, dict):
        raise ValueError("Physical bending checkpoint provenance must be a JSON object")
    metadata = dict(format=FORMAT, origin_surface=asdict(origin_surface),
        bending_motion=bending_motion_to_dict(mesh, origin_surface, state),
        user_provenance={} if provenance is None else provenance)
    return save_boundary_checkpoint(path, state.current_surface, state.current_inventory,
                                    provenance=metadata)


def load_bending_motion_checkpoint(path, *, mesh):
    """Replay every force, source and material transaction against the mesh."""
    current_surface, current_inventory, metadata = load_boundary_checkpoint(path)
    expected = {"format", "origin_surface", "bending_motion", "user_provenance"}
    if (not isinstance(metadata, dict) or set(metadata) != expected
            or metadata["format"] != FORMAT or not isinstance(metadata["user_provenance"], dict)):
        raise ValueError("Expected explicit accepted physical bending checkpoint metadata")
    origin = _surface_from_dict(metadata["origin_surface"])
    state = bending_motion_from_dict(metadata["bending_motion"], mesh, origin)
    if current_surface != state.current_surface or current_inventory != state.current_inventory:
        raise ValueError("Checkpoint current surface/archive differs from physical trajectory replay")
    return origin, state, metadata["user_provenance"]
