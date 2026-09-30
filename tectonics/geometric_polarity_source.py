"""Explicit import of legacy contact direction, never of slab material or force.

The raster producer's tie conventions remain part of this inherited boundary
condition. Calling this adapter opts into that provenance; it does not certify
that the old model resolved physical initiation. Opposing histories survive as
opposing evidence. Only the initial mesh-triangle surface can be translated.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
import math

import numpy as np

from .geometric_contacts import extract_contacts
from .geometric_polarity import PolarityEvidence, contact_polarity_report

PROVENANCE = "legacy_raster_accepted_history_not_resolved_initiation"
_COORDINATE_ROUNDOFF = 64.*np.finfo(float).eps


def _same_vertices(left, right):
    """Unordered original vertices, allowing only normalization roundoff."""
    left, right = np.asarray(left), np.asarray(right)
    if left.shape != right.shape:
        return False
    distance = np.max(np.abs(left[:, None, :]-right[None, :, :]), axis=2)
    matches = distance <= _COORDINATE_ROUNDOFF
    return bool(np.all(matches.sum(axis=0) == 1) and np.all(matches.sum(axis=1) == 1))


def _initial_fragments(mesh, surface):
    if surface.phase != "partition" or len(surface.fragments) != mesh.cell_count:
        raise ValueError("Legacy polarity requires an initial pure mesh-triangle surface")
    by_cell = {}
    for fragment in surface.fragments:
        cell = fragment.parcel.cell
        if (cell in by_cell or cell < 0 or cell >= mesh.cell_count or
                not _same_vertices(fragment.polygon, mesh.vertices[mesh.faces[cell]])):
            raise ValueError("Legacy polarity cannot map a mixed, moved or noninitial footprint")
        by_cell[cell] = fragment
    return by_cell


def _inventory_digest(inventory):
    # Empty legacy inventories can legitimately contain a -inf transaction
    # sentinel. It is only hashed, never copied into the JSON audit or evidence.
    encoded = json.dumps(None if inventory is None else asdict(inventory),
                         sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _retained(segment):
    area, ocean, cold, mass = [], [], [], []
    for cohort in segment.thermal_cohorts:
        values = (cohort.deep_transfer_fraction, cohort.accepted_area_km2,
                  cohort.oceanic_volume_km3, cohort.cold_mantle_volume_km3,
                  cohort.initial_density_excess_mass_kg)
        if (not all(math.isfinite(x) for x in values) or not 0. <= values[0] <= 1.
                or any(x < 0. for x in values[1:])):
            return None
        fraction = 1.-values[0]
        for group, value in zip((area, ocean, cold, mass), values[1:]):
            group.append(fraction*value)
    return dict(retained_area_km2=math.fsum(area), retained_oceanic_volume_km3=math.fsum(ocean),
                retained_initial_cold_volume_km3=math.fsum(cold),
                retained_initial_excess_mass_kg=math.fsum(mass))


def legacy_polarity_evidence(mesh, surface, legacy_inventory):
    """Translate exact local legacy directions without choosing conflicts.

    Returns immutable evidence records and a JSON-compatible exclusion/mapping
    audit. No surface, contact, segment, cohort, ledger or mesh is modified.
    This must be called at initial import, before any fragment rotation/split.
    """
    by_cell = _initial_fragments(mesh, surface)
    before = _inventory_digest(legacy_inventory)
    contacts = extract_contacts(surface.fragments, surface.radius_km)
    by_pair = defaultdict(list)
    for contact in contacts:
        by_pair[frozenset((contact.fragment_a, contact.fragment_b))].append(contact)
    segments = () if legacy_inventory is None else tuple(legacy_inventory.segments.values())
    if len({segment.key for segment in segments}) != len(segments):
        raise ValueError("Legacy segment identities must be unique")
    evidence, excluded, mapping = [], [], []
    for segment in sorted(segments, key=lambda item: item.key):
        reasons = []
        retained = _retained(segment)
        if not segment.attached:
            reasons.append("detached_segment")
        if retained is None:
            reasons.append("invalid_retained_cohort")
        elif retained["retained_area_km2"] <= 0. or retained["retained_oceanic_volume_km3"] <= 0.:
            reasons.append("no_retained_material")
        contact = legacy_inventory.contacts.get(segment.contact_key)
        geometric = None
        if contact is None or not contact.present:
            reasons.append("absent_legacy_contact")
        else:
            if contact.key != segment.contact_key:
                reasons.append("legacy_contact_identity_mismatch")
            if ({contact.plate_a, contact.plate_b} !=
                    {segment.subducting_plate, segment.overriding_plate} or
                    segment.subducting_plate == segment.overriding_plate):
                reasons.append("legacy_owner_pair_mismatch")
            if (contact.face_a not in by_cell or contact.face_b not in by_cell
                    or contact.face_a == contact.face_b):
                reasons.append("invalid_legacy_faces")
            else:
                a, b = by_cell[contact.face_a], by_cell[contact.face_b]
                if (a.parcel.plate, b.parcel.plate) != (contact.plate_a, contact.plate_b):
                    reasons.append("current_owner_mismatch")
                shared = sorted(set(mesh.faces[contact.face_a]) & set(mesh.faces[contact.face_b]))
                if len(shared) != 2 or segment.contact_key != f"{shared[0]}:{shared[1]}":
                    reasons.append("not_exact_source_mesh_edge")
                else:
                    candidates = [candidate for candidate in
                        by_pair.get(frozenset((a.fragment_id, b.fragment_id)), ())
                        if _same_vertices((candidate.start, candidate.end), mesh.vertices[shared])]
                    if len(candidates) == 1:
                        geometric = candidates[0]
                    else:
                        reasons.append("no_unique_exact_geometric_arc")
        if reasons:
            excluded.append(dict(segment_key=segment.key, contact_key=segment.contact_key,
                                 reasons=sorted(set(reasons))))
            continue
        item = PolarityEvidence(f"legacy:{segment.key}", geometric.contact_id,
            segment.subducting_plate, segment.overriding_plate, PROVENANCE)
        evidence.append(item)
        mapping.append(dict(segment_key=segment.key, legacy_contact_key=segment.contact_key,
            geometric_contact_id=geometric.contact_id, evidence_id=item.evidence_id,
            subducting_plate=item.subducting_plate, overriding_plate=item.overriding_plate,
            source_face=contact.face_a if contact.plate_a == item.subducting_plate else contact.face_b,
            receiver_face=contact.face_b if contact.plate_a == item.subducting_plate else contact.face_a,
            legacy_retained_quantities_not_imported=retained))
    reports = contact_polarity_report(contacts, evidence)
    after = _inventory_digest(legacy_inventory)
    if after != before:
        raise RuntimeError("Legacy polarity import modified its source inventory")
    audit = dict(format="legacy-geometric-polarity-import-1", provenance=PROVENANCE,
        interpretation="Explicit inherited boundary condition; no resolved initiation or material/force import",
        counts=dict(legacy_contacts=0 if legacy_inventory is None else len(legacy_inventory.contacts),
                    legacy_segments=len(segments), attached_segments=sum(s.attached for s in segments),
                    geometric_contacts=len(contacts), geometry_matched_segments=len(evidence),
                    geometry_matched_contacts=len(reports), excluded_segments=len(excluded),
                    oriented_contacts=sum(r["status"] == "oriented" for r in reports),
                    conflicting_contacts=sum(r["status"] == "conflicting" for r in reports)),
        exclusions_by_reason=dict(sorted(Counter(reason for row in excluded for reason in row["reasons"]).items())),
        excluded_segments=excluded, mapping=mapping, contact_reports=reports,
        source_inventory_sha256_before=before, source_inventory_sha256_after=after,
        source_inventory_unchanged=True, imported_material=False, imported_force=False)
    return tuple(evidence), audit
