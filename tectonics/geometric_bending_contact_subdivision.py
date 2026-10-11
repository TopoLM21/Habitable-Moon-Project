"""Conservative material cuts preserve physical bending and contact history."""
from dataclasses import replace

from .geometric_boundary import surface_digest
from .geometric_contact_dynamics import _basal
from .geometric_bending_contact_dynamics import validate_bending_contact_dynamics_state
from .geometric_fault_traces import declare_fault_trace_set, fault_trace_points
from .geometric_material_subdivision import (
    validate_material_subdivision, validate_subdivision_basal_rebind,
)


def rebind_bending_contact_subdivision(before, after, previous, old_load, new_load):
    """Rebind an evolved state without changing physical strips or any ledger.

    Clamp distances, elastic thicknesses and tip deflections belong to physical
    traces, not fragment IDs. Both loads must be assembled from their trusted
    source inputs. The unchanged source-work and bending certificates must
    pass after the ordinary conservative subdivision gates; no work is fitted.
    """
    validate_bending_contact_dynamics_state(before, previous, old_load)
    parents = validate_material_subdivision(before, after)
    validate_subdivision_basal_rebind(before, after, old_load, new_load, parents)
    traces = declare_fault_trace_set(after, previous.fault_traces.traces,
                                    previous.fault_traces.quadrature)
    if (traces.trace_sha256 != previous.fault_traces.trace_sha256
            or fault_trace_points(before, previous.fault_traces)
            != fault_trace_points(after, traces)):
        raise ValueError("Material subdivision cannot change physical bending support or quadrature")
    rebound = replace(previous, surface_sha256=surface_digest(after),
        basal_source_sha256=_basal(after, new_load)[3], fault_traces=traces)
    return validate_bending_contact_dynamics_state(after, rebound, new_load)


def rebind_bending_boundary_subdivision(before, after, previous, old_load, new_load, inventory):
    """Return matching immutable mechanics/archive only after both checks pass."""
    from .geometric_boundary_subdivision import rebind_boundary_for_material_subdivision
    state = rebind_bending_contact_subdivision(before, after, previous, old_load, new_load)
    boundary = rebind_boundary_for_material_subdivision(before, after, inventory)
    return state, boundary
