"""Restart a finite-geometry proposal with its original validated references.

The existing joint checkpoint binds the original surface and slab archive.
An unaccepted candidate is stored as explicit metadata and reconstructed from
its material atlas. It never masquerades as the surface of the old archive.
"""
from .geometric_basal_loading import basal_load_from_dict, basal_load_to_dict
from .geometric_bending_fault_transport import (
    bending_fault_transport_from_dict, bending_fault_transport_to_dict,
    validate_bending_fault_transport_state,
)
from .geometric_boundary_io import load_boundary_checkpoint, save_boundary_checkpoint


FORMAT = "geometric-bending-fault-transport-checkpoint-1"


def save_bending_fault_transport_checkpoint(path, origin_surface, state, reference_load, *, provenance=None):
    """Save the unaccepted transport state beside its genuine reference pair."""
    validate_bending_fault_transport_state(origin_surface, state, reference_load)
    if provenance is not None and not isinstance(provenance, dict):
        raise ValueError("Transport checkpoint provenance must be a JSON object")
    metadata = dict(format=FORMAT,
        reference_basal_load=basal_load_to_dict(origin_surface, reference_load),
        fault_transport=bending_fault_transport_to_dict(origin_surface, state, reference_load),
        user_provenance={} if provenance is None else provenance)
    return save_boundary_checkpoint(path, origin_surface, state.reference_inventory,
                                    provenance=metadata)


def load_bending_fault_transport_checkpoint(path, *, mesh=None, parameters=None,
        thermal_lid_by_fragment=None, source_provenance=None):
    """Load original mechanics/source and replay the finite transport proposal.

    Optional trusted source inputs retain the established basal loader's
    independent reassembly check. The source is never rebound to a candidate
    geometry and its previously certified work is not rewritten.
    """
    origin, inventory, metadata = load_boundary_checkpoint(path)
    expected = {"format", "reference_basal_load", "fault_transport", "user_provenance"}
    if (not isinstance(metadata, dict) or set(metadata) != expected
            or metadata["format"] != FORMAT or not isinstance(metadata["user_provenance"], dict)):
        raise ValueError("Expected explicit finite-transport proposal metadata")
    load = basal_load_from_dict(metadata["reference_basal_load"], origin, mesh=mesh,
        parameters=parameters, thermal_lid_by_fragment=thermal_lid_by_fragment,
        provenance=source_provenance)
    state = bending_fault_transport_from_dict(metadata["fault_transport"], origin, load)
    if state.reference_inventory != inventory:
        raise ValueError("Transport reference archive differs from its original joint checkpoint")
    return origin, state, load, metadata["user_provenance"]
