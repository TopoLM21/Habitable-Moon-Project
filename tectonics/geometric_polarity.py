"""Local, oriented boundary conditions for geometric material transactions.

An existing slab direction or explicit oriented seed is evidence, not a new
subduction-initiation law. Scalar damage and arbitrary plate IDs cannot choose
a fault dip. Contact histories that disagree remain explicitly unresolved.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math

import numpy as np

from .geometric_contacts import extract_contacts
from .geometric_surface import rotations
from .spherical_polygons import arc_length
from .spherical_polygons import intersect_convex, rotate_polygon


@dataclass(frozen=True, slots=True)
class PolarityEvidence:
    evidence_id: str
    contact_id: str
    subducting_plate: int
    overriding_plate: int
    provenance: str
    origin_evidence_id: str = ""

    def __post_init__(self):
        if any(not isinstance(x, str) or not x for x in
               (self.evidence_id, self.contact_id, self.provenance)):
            raise ValueError("Polarity evidence needs explicit identity and provenance")
        if (any(isinstance(x, bool) or not isinstance(x, (int, np.integer)) or x < 0
                for x in (self.subducting_plate, self.overriding_plate))
                or self.subducting_plate == self.overriding_plate):
            raise ValueError("Polarity evidence requires two distinct plate owners")
        if not isinstance(self.origin_evidence_id, str):
            raise ValueError("Polarity origin identity must be a string")
        if not self.origin_evidence_id:
            object.__setattr__(self, "origin_evidence_id", self.evidence_id)


@dataclass(frozen=True, slots=True)
class PolarityResolution:
    comparison: int
    basis: str
    evidence_ids: tuple[str, ...] = ()
    contact_ids: tuple[str, ...] = ()

    def __post_init__(self):
        if (isinstance(self.comparison, bool) or not isinstance(self.comparison, (int, np.integer))
                or self.comparison not in (-1, 0, 1)
                or not isinstance(self.basis, str) or not self.basis
                or any(not isinstance(x, str) or not x for x in (*self.evidence_ids, *self.contact_ids))):
            raise ValueError("Invalid explicit polarity resolution")
        object.__setattr__(self, "evidence_ids", tuple(self.evidence_ids))
        object.__setattr__(self, "contact_ids", tuple(self.contact_ids))


def rotate_points(points, omega, dt):
    """Rigid rotation for arc endpoints, without inventing a polygon interior."""
    points = np.asarray(points, dtype=float)
    omega = np.asarray(omega, dtype=float)
    magnitude = float(np.linalg.norm(omega))
    if magnitude == 0.:
        return points.copy()
    axis, angle = omega/magnitude, magnitude*dt
    result = (points*math.cos(angle)+np.cross(axis, points)*math.sin(angle)
              +np.outer(points@axis, axis)*(2.*math.sin(angle/2.)**2))
    return result/np.linalg.norm(result, axis=1)[:, None]


def intersect_arcs(a0, a1, b0, b1):
    """Positive-length shared minor arc, oriented from a0 towards a1.

    Plane coincidence is filtered at arithmetic precision. Point contact alone
    gives no support. No geographic nearest-neighbor or minimum length rule.
    """
    a0, a1, b0, b1 = [np.asarray(point, dtype=float) for point in (a0, a1, b0, b1)]
    length_a, length_b = arc_length(a0, a1), arc_length(b0, b1)
    if not (0. < length_a < math.pi and 0. < length_b < math.pi):
        return None
    x, y = (a0, a1) if length_a >= length_b else (b0, b1)
    normal = np.cross(x, y-x)
    normal /= np.linalg.norm(normal)
    probes = (b0, b1) if length_a >= length_b else (a0, a1)
    if any(abs(float(np.dot(point, normal))) > 32.*np.finfo(float).eps*float(
            np.sum(np.abs(point*normal))) for point in probes):
        return None
    tangent = np.cross(normal, a0)
    if float(np.dot(tangent, a1)) < 0.:
        tangent = -tangent
    start, end = [math.atan2(float(np.dot(tangent, p)), float(np.dot(a0, p))) for p in (b0, b1)]
    if end-start > math.pi:
        end -= 2.*math.pi
    elif end-start < -math.pi:
        end += 2.*math.pi
    for shift in (-2.*math.pi, 0., 2.*math.pi):
        low, high = max(0., min(start, end)+shift), min(length_a, max(start, end)+shift)
        if high <= low:
            continue
        candidates = ((0., a0), (length_a, a1), (start+shift, b0), (end+shift, b1))
        first = min(candidates, key=lambda pair: abs(pair[0]-low))[1]
        last = min(candidates, key=lambda pair: abs(pair[0]-high))[1]
        if not np.array_equal(first, last):
            return tuple(map(float, first)), tuple(map(float, last))
    return None


def contact_polarity_report(contacts, evidence):
    contacts = {c.contact_id: c for c in contacts}
    grouped, identities = {}, set()
    for item in evidence:
        if not isinstance(item, PolarityEvidence):
            raise ValueError("Expected explicit PolarityEvidence records")
        if item.evidence_id in identities:
            raise ValueError("Polarity evidence IDs must be unique")
        identities.add(item.evidence_id)
        contact = contacts.get(item.contact_id)
        if contact is None or {contact.plate_a, contact.plate_b} != {
                item.subducting_plate, item.overriding_plate}:
            raise ValueError("Polarity evidence has no matching current local contact")
        grouped.setdefault(item.contact_id, []).append(item)
    result = []
    for identity, items in sorted(grouped.items()):
        directions = sorted({(e.subducting_plate, e.overriding_plate) for e in items})
        result.append(dict(contact_id=identity, status="oriented" if len(directions) == 1 else "conflicting",
            directions=[list(pair) for pair in directions], evidence_ids=sorted(e.evidence_id for e in items),
            provenance=sorted({e.provenance for e in items})))
    return result


def make_polarity_resolver(before, omega_rad_per_myr, dt_myr, *, contacts, evidence):
    """Compile local evidence against an actual pre-step geometric partition.

    The old arc follows its overriding material. Evidence applies only when
    that same receiver fragment bounds a positive part of the candidate loss.
    Incoming donor histories may change; disconnected arcs of the same pair
    cannot inherit through pair identity or cell occupancy alone.
    """
    dt = float(dt_myr)
    omega = rotations(before, omega_rad_per_myr, dt)
    contacts, evidence = tuple(contacts), tuple(evidence)
    canonical = {c.contact_id: c for c in extract_contacts(before.fragments, before.radius_km)}
    # Kinematic fields can differ because an inventory need not store the next
    # prescribed velocity. All spatial sides/arcs must be current and genuine.
    for contact in contacts:
        expected = canonical.get(contact.contact_id)
        if expected is None or any(getattr(contact, name) != getattr(expected, name) for name in
                ("fragment_a", "fragment_b", "material_a", "material_b", "plate_a", "plate_b",
                 "start", "end", "normal_a_to_b", "length_km")):
            raise ValueError("Polarity contact geometry is stale or not part of the surface")
    by_contact = {c.contact_id: c for c in contacts}
    reports = contact_polarity_report(contacts, evidence)
    compiled = []
    for record in reports:
        contact = by_contact[record["contact_id"]]
        for sub, over in record["directions"]:
            receiver = contact.fragment_a if contact.plate_a == over else contact.fragment_b
            material = contact.material_a if contact.plate_a == over else contact.material_b
            arc = rotate_points((contact.start, contact.end), omega[over], dt)
            compiled.append((record, sub, over, receiver, material, arc))

    def resolve(a, b, overlap):
        matches, choices = {}, set()
        polygon = np.asarray(overlap)
        for record, sub, over, receiver_id, material, arc in compiled:
            receiver, donor = (a, b) if a.fragment_id == receiver_id else (b, a)
            if (receiver.fragment_id != receiver_id or receiver.parcel.material_id != material
                    or receiver.parcel.plate != over or donor.parcel.plate != sub):
                continue
            if not any(intersect_arcs(arc[0], arc[1], x, y) is not None
                       for x, y in zip(polygon, np.roll(polygon, -1, axis=0))):
                continue
            matches[record["contact_id"]] = record
            choices.add(-1 if receiver is a else 1)
        if not matches:
            return None
        conflict = len(choices) != 1 or any(r["status"] == "conflicting" for r in matches.values())
        return PolarityResolution(0 if conflict else next(iter(choices)),
            "conflicting_local_history" if conflict else "oriented_local_history",
            tuple(sorted({e for record in matches.values() for e in record["evidence_ids"]})),
            tuple(sorted(matches)))
    return resolve


def transport_polarity_evidence(before, after, omega_rad_per_myr, dt_myr, *,
                               contacts_before, contacts_after, evidence):
    """Carry explicit boundary conditions by receiver material and shared arcs.

    These records contain no slab mass. Losing a geometric support records
    unresolved lineage rather than declaring physical breakoff or moving the
    condition to another same-pair contact somewhere else on the sphere.
    """
    omega = rotations(before, omega_rad_per_myr, float(dt_myr))
    if not math.isclose(after.time_myr-before.time_myr, dt_myr, rel_tol=0., abs_tol=8.*math.ulp(after.time_myr)):
        raise ValueError("Polarity lineage clocks disagree")
    contacts_before, contacts_after, evidence = tuple(contacts_before), tuple(contacts_after), tuple(evidence)
    contact_polarity_report(contacts_before, evidence)
    old_contacts = {c.contact_id: c for c in contacts_before}
    old_fragments = {f.fragment_id: f for f in before.fragments}
    new_fragments = {f.fragment_id: f for f in after.fragments}
    propagated, unresolved, coverage = {}, [], {}
    for item in evidence:
        previous = old_contacts[item.contact_id]
        receiver_id = previous.fragment_a if previous.plate_a == item.overriding_plate else previous.fragment_b
        receiver = old_fragments[receiver_id]
        moved_arc = rotate_points((previous.start, previous.end), omega[item.overriding_plate], dt_myr)
        moved_receiver = rotate_polygon(receiver.polygon, omega[item.overriding_plate], dt_myr)
        matched = []
        for contact in contacts_after:
            if {contact.plate_a, contact.plate_b} != {item.subducting_plate, item.overriding_plate}:
                continue
            new_receiver_id = contact.fragment_a if contact.plate_a == item.overriding_plate else contact.fragment_b
            new_receiver = new_fragments[new_receiver_id]
            if new_receiver.parcel.material_id != receiver.parcel.material_id:
                continue
            shared = intersect_arcs(*moved_arc, contact.start, contact.end)
            if shared is None:
                continue
            if not len(intersect_convex(moved_receiver, new_receiver.polygon)):
                continue
            suffix = hashlib.sha256(contact.contact_id.encode()).hexdigest()[:24]
            identity = item.evidence_id if contact.contact_id == item.contact_id else f"{item.origin_evidence_id}@contact:{suffix}"
            value = replace(item, evidence_id=identity, contact_id=contact.contact_id)
            key = (item.origin_evidence_id, contact.contact_id)
            propagated[key] = value
            origin = np.asarray(contact.start)
            tangent = np.cross(contact.normal_a_to_b, origin)
            tangent /= np.linalg.norm(tangent)
            if float(tangent@(np.asarray(contact.end)-origin)) < 0.:
                tangent = -tangent
            parameters = [math.atan2(float(tangent@(np.asarray(p)-origin)), float(origin@p)) for p in shared]
            extent = arc_length(contact.start, contact.end)
            coverage.setdefault(key, []).append((max(0., min(parameters)), min(extent, max(parameters)), extent))
            matched.append(contact.contact_id)
        if not matched:
            unresolved.append(dict(evidence_id=item.evidence_id, origin_evidence_id=item.origin_evidence_id,
                previous_contact_id=item.contact_id, subducting_plate=item.subducting_plate,
                overriding_plate=item.overriding_plate, time_myr=after.time_myr,
                reason="unresolved_polarity_lineage", transported_arc=moved_arc.tolist(),
                provenance=item.provenance))
    # A condition names a whole contact. Do not expand a partial inherited
    # arc over a newly merged contact. Split siblings with the same origin can
    # jointly establish complete coverage; their evidence is still one record.
    for key, intervals in coverage.items():
        merged = []
        for low, high, extent in sorted(intervals):
            if merged and low <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], high)
            else:
                merged.append([low, high])
        covered = math.fsum(high-low for low, high in merged)
        if not math.isclose(covered, extent, rel_tol=512.*np.finfo(float).eps, abs_tol=0.):
            item = propagated.pop(key)
            unresolved.append(dict(evidence_id=item.evidence_id, origin_evidence_id=item.origin_evidence_id,
                current_contact_id=item.contact_id, subducting_plate=item.subducting_plate,
                overriding_plate=item.overriding_plate, time_myr=after.time_myr,
                reason="partial_polarity_contact_coverage", coverage_fraction=covered/extent,
                provenance=item.provenance))
    return tuple(propagated[key] for key in sorted(propagated)), tuple(unresolved)
