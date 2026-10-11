"""Explicit conservative rebinding of fixed physical point histories.

This operation changes material bookkeeping only. The physical weak traces,
quadrature, local reference displacements and all mechanical ledgers remain
unchanged. Raw checkpoint loading onto another surface remains prohibited.
"""
from dataclasses import replace

from .geometric_boundary import surface_digest
from .geometric_contact_dynamics import _basal
from .geometric_distributed_contact_dynamics import validate_distributed_contact_dynamics_state
from .geometric_fault_traces import declare_fault_trace_set, fault_trace_points
from .geometric_material_subdivision import (
    validate_material_subdivision, validate_subdivision_basal_rebind,
)


def rebind_distributed_contact_subdivision(before, after, previous, old_load, new_load):
    """Preserve evolved point history through a verified direct material split.

    Both prescribed loads must be independently assembled and bound to their
    surfaces. The subdivision validator checks every donor, owner, material
    history, reservoir and inherited full thermal lid. Only integration
    roundoff is allowed in the physical basal operators. A new physical
    trace, changed quadrature or fitted load cannot migrate the old history.

    Retained history names physical intervals and keeps original contact
    evidence as provenance. It is not regenerated from the new cut contacts.
    Work and preparation references are never adjusted to fit the new source;
    their ordinary independent validation must still pass after rebinding.
    """
    validate_distributed_contact_dynamics_state(before, previous, old_load)
    mapping = validate_material_subdivision(before, after)
    validate_subdivision_basal_rebind(before, after, old_load, new_load, mapping)
    traces = declare_fault_trace_set(after, previous.fault_traces.traces,
                                    previous.fault_traces.quadrature)
    if traces.trace_sha256 != previous.fault_traces.trace_sha256:
        raise ValueError("Material subdivision cannot change physical traces or quadrature")
    if fault_trace_points(before, previous.fault_traces) != fault_trace_points(after, traces):
        raise ValueError("Material subdivision changed a fixed physical quadrature point")
    state = replace(previous, surface_sha256=surface_digest(after),
                    basal_source_sha256=_basal(after, new_load)[3], fault_traces=traces)
    return validate_distributed_contact_dynamics_state(after, state, new_load)


def rebind_distributed_boundary_subdivision(before, after, previous, old_load, new_load, inventory):
    """Atomically prepare matching mechanics and passive archive after a cut.

    Existing inputs are immutable. The archive keeps every accepted loss and
    transaction, remaps only geometric support, and records a zero-time change
    of representation. If either validation fails, no new pair is returned.
    """
    from .geometric_boundary_subdivision import rebind_boundary_for_material_subdivision
    state = rebind_distributed_contact_subdivision(before, after, previous, old_load, new_load)
    boundary = rebind_boundary_for_material_subdivision(before, after, inventory)
    return state, boundary
