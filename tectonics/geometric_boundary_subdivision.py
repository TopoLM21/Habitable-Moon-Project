"""Rebind passive slab support after a strict zero-time material subdivision.

Only the representation of current surface material and contact support changes.
Accepted loss payloads, material reservoirs and physical transaction identities
remain unchanged. A separate representation registry connects the old and new
surface signatures without inventing a new acceptance or advancing a clock.
"""
from __future__ import annotations

from dataclasses import replace
import math

import numpy as np

from .geometric_boundary import (
    GeometricBoundaryState, _representation_record, _transfer_cohort,
    _validate_contacts, boundary_from_dict, boundary_to_dict, surface_digest,
)
from .geometric_boundary_io import _validate_pair
from .geometric_contacts import extract_contacts
from .geometric_material_subdivision import validate_material_subdivision
from .spherical_polygons import arc_length


def rebind_boundary_for_material_subdivision(old_surface, new_surface, inventory):
    """Preserve one archived cohort per accepted loss through a pure cut.

    The strict subdivision gate proves donor containment, unchanged ownership,
    frozen histories, and all extensive budgets. Existing actual support is
    transferred with a zero-time identity rotation in each owner's frame. Its
    total length must remain fully covered; even a previously unresolved cohort
    cannot silently lose an additional strip of saved support.

    Splitting one supported contact into several retains the archive's existing
    ``unresolved_lateral_allocation`` classification: no slab material is copied
    or assigned to the child arcs. Existing parent support lineage remains on
    each descendant. Mechanics must preserve its explicit physical trace
    evidence instead of selecting a new polarity from child iteration order.
    """
    if not isinstance(inventory, GeometricBoundaryState):
        raise ValueError("Material subdivision requires a geometric boundary inventory")
    # Validate physical acceptance, current support, and any earlier zero-time
    # history before constructing an immutable replacement.
    boundary_from_dict(boundary_to_dict(inventory))
    _validate_pair(old_surface, inventory)
    mapping = validate_material_subdivision(old_surface, new_surface)
    if surface_digest(old_surface) == surface_digest(new_surface):
        return inventory

    contacts = _validate_contacts(new_surface,
        extract_contacts(new_surface.fragments, new_surface.radius_km))
    by_contact = {contact.contact_id: contact for contact in contacts}
    old_contacts = {contact.contact_id: contact for contact in inventory.contacts}
    # A compact owner dictionary avoids allocating through the largest plate ID.
    omega = {fragment.parcel.plate: np.zeros(3) for fragment in old_surface.fragments}
    cohorts = []
    for previous in inventory.cohorts:
        current = _transfer_cohort(previous, old_surface, new_surface,
            by_contact, omega, 0., old_contacts)
        old_length = math.fsum(arc_length(item.start, item.end) for item in previous.supports)
        new_length = math.fsum(arc_length(item.start, item.end) for item in current.supports)
        tolerance = 512.*np.finfo(float).eps*max(old_length, new_length)
        if (abs(new_length-old_length) > tolerance
                or current.unresolved_support != previous.unresolved_support):
            raise ValueError("Material subdivision lost or duplicated saved slab support")
        if (current.loss != previous.loss or current.event_id != previous.event_id
                or current.payload_sha256 != previous.payload_sha256):
            raise ValueError("Material subdivision changed an accepted slab payload")
        if len(current.connectivity_history) > len(previous.connectivity_history):
            # This is bookkeeping, not a new moving trench or physical breakoff.
            change = replace(current.connectivity_history[-1], reason="material_subdivision")
            current = replace(current,
                connectivity_history=previous.connectivity_history+(change,))
        cohorts.append(current)
    record = _representation_record(old_surface, new_surface, mapping)
    if any(item.record_id == record.record_id for item in inventory.representations):
        raise ValueError("Duplicate material representation; replay rejected")
    rebound = replace(inventory, surface_digest=surface_digest(new_surface),
        cohorts=tuple(cohorts), contacts=contacts,
        representations=inventory.representations+(record,))
    boundary_from_dict(boundary_to_dict(rebound))
    _validate_pair(new_surface, rebound)
    return rebound
