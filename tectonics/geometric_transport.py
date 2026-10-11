"""Experimental endpoint transactions on persisted spherical footprints.

This fixed-motion geometry experiment freezes thermal/damage properties.
An explicit buoyancy/age ordering resolves real overlaps; exact physical ties
remain unresolved. Actual vacancies inherit the pre-step local owner's hot
newborn material. This is a declared departure-footprint closure, not a ridge
axis or subduction-initiation model. No archived loss exerts a force here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from collections.abc import Mapping
import heapq
import math
from numbers import Real

import numpy as np
from scipy.spatial import cKDTree

from .fractional_surface import EXTENSIVE_FIELDS, SurfaceParcel
from .geometric_contacts import extract_contacts
from .geometric_surface import (GeometricFragment, GeometricSurfaceState,
    audit_partition, candidate_pairs, caps, rotate_surface, rotations, split_geometry, totals)
from .spherical_polygons import (_reverse_polygon, _shared_minor_arc, GeometryDiagnostics,
    intersect_convex, polygon_area, rotate_polygon, shared_boundary_arcs, subtract_convex)


@dataclass(frozen=True, slots=True)
class GeometricOverlap:
    fragment_a: str
    fragment_b: str
    plate_a: int
    plate_b: int
    polygon: tuple[tuple[float, float, float], ...]
    area_km2: float
    reason: str


class UnresolvedPolarityError(ValueError):
    def __init__(self, overlaps):
        self.overlaps = tuple(overlaps)
        super().__init__(f"{len(self.overlaps)} real overlaps need an explicit polarity decision")


@dataclass(frozen=True, slots=True)
class GeometricLoss:
    fragment: GeometricFragment
    source_fragment_id: str
    receiver_fragment_id: str
    receiver_plate: int
    time_myr: float
    contact_ids: tuple[str, ...] = ()
    attachment_status: str = "unresolved_attachment"
    polarity_basis: str = "material_contrast"
    polarity_evidence_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class GeometricTransportResult:
    state: GeometricSurfaceState
    losses: tuple[GeometricLoss, ...]
    births: tuple[GeometricFragment, ...]
    contacts: tuple
    diagnostics: dict


def _compare_polarity(a, b):
    """Greater negative buoyancy/area loses; age breaks arithmetic mass ties."""
    # A*H*rho/A can differ by ULPs even for identical physical inputs. This is
    # an arithmetic bound inherited from parcel accounting, not tunable physics.
    for left, right in ((a.parcel.specific_properties[2], b.parcel.specific_properties[2]),
                        (a.parcel.age_myr, b.parcel.age_myr)):
        if not math.isclose(left, right, rel_tol=64.*np.finfo(float).eps, abs_tol=0.):
            return -1 if left < right else 1
    return 0


class _Index:
    """Conservative bounding-cap index; true polygon predicates decide overlap."""
    def __init__(self, fragments):
        self.fragments = tuple(fragments)
        self.centers, self.radii = caps([f.polygon for f in fragments])
        self.tree = cKDTree(self.centers)
        self.maximum = float(np.max(self.radii))

    def query(self, polygon):
        centers, radii = caps([polygon])
        center, radius = centers[0], radii[0]
        padding = 64.*np.finfo(float).eps
        chord = 2.*math.sin(min(math.pi, radius+self.maximum)/2.)+padding
        return [self.fragments[j] for j in sorted(self.tree.query_ball_point(center, chord))
                if np.linalg.norm(center-self.centers[j]) <=
                2.*math.sin((radius+self.radii[j])/2.)+padding]


def _arc_meets_contact(start, end, contact):
    """Positive overlap of two known boundary arcs, using contact's stable plane."""
    return _shared_minor_arc(contact.start, contact.end, start, end) is not None


def _attach_losses(losses, contacts, retained, descendants, diag):
    by_id = {f.fragment_id: f for f in retained}
    by_pair = {}
    for contact in contacts:
        pair = frozenset((contact.fragment_a, contact.fragment_b))
        by_pair.setdefault(pair, []).append(contact)
    output = []
    for loss in losses:
        attached = set()
        for child_id in descendants[loss.source_fragment_id]:
            pair = frozenset((child_id, loss.receiver_fragment_id))
            possible = by_pair.get(pair, ())
            if not possible:
                continue
            arcs = shared_boundary_arcs(loss.fragment.polygon, by_id[child_id].polygon, diagnostics=diag)
            for contact in possible:
                if any(_arc_meets_contact(a, b, contact) for a, b in arcs):
                    attached.add(contact.contact_id)
        output.append(replace(loss, contact_ids=tuple(sorted(attached)),
            attachment_status="geometric_contact" if attached else "unresolved_attachment"))
    return tuple(output)


def _diagnostics(before, state, losses, births, contacts, diag):
    initial, remaining = totals(before.fragments), totals(state.fragments)
    removed, born = totals(loss.fragment for loss in losses), totals(births)
    residual = {key: (remaining[key]+removed[key]-born[key]-initial[key]) /
        max(abs(initial[key]), abs(born[key]), 1.) for key in EXTENSIVE_FIELDS}
    if any(abs(value) > 5e-12 for value in residual.values()):
        raise RuntimeError("Geometric material transaction failed its extensive ledger")
    return dict(geometry=asdict(diag), retained=remaining, removed=removed, born=born,
        relative_residuals=residual, contacts=len(contacts),
        attached_loss_count=sum(bool(loss.contact_ids) for loss in losses),
        unresolved_attachment_count=sum(not loss.contact_ids for loss in losses),
        polarity_policy="greater excess mass/area loses, then greater age; 64eps arithmetic ties unresolved",
        birth_policy="actual vacancy intersected with pre-step footprints; endpoint hot birth",
        time_scheme="endpoint_geometry_v1", thermal_evolution="frozen",
        mechanical_attachment="geometric adjacency only; no forces or slab attachment state")


def advance_geometric_surface(mesh, state, omega_rad_per_myr, dt_myr, *, birth_factory,
                              polarity_resolver=None):
    """Return one atomic, budgeted endpoint transaction; never mutate input.

    A pure common rotation needs no contact policy. Differential motion first
    rejects physically tied overlaps before creating any newborn material.
    """
    if state.phase != "partition":
        raise ValueError("Geometric advance requires a resolved partition")
    if not callable(birth_factory):
        raise ValueError("Geometric advance requires an explicit birth factory")
    if polarity_resolver is not None and not callable(polarity_resolver):
        raise ValueError("Geometric polarity resolver must be callable")
    if any(f.parcel.cell >= mesh.cell_count for f in state.fragments):
        raise ValueError("Source material references an invalid birth-provenance cell")
    if isinstance(omega_rad_per_myr, Mapping) and (
            isinstance(dt_myr, (bool, np.bool_)) or not isinstance(dt_myr, Real)):
        raise ValueError("Compact transport requires an explicit real timestep")
    dt = float(dt_myr)
    omega = rotations(state, omega_rad_per_myr, dt)
    diag = GeometryDiagnostics()
    audit_partition(state, diagnostics=diag)
    moved = rotate_surface(state, omega, dt)
    if moved.phase == "partition":
        contacts = extract_contacts(moved.fragments, state.radius_km, omega)
        return GeometricTransportResult(moved, (), (), contacts,
            _diagnostics(state, moved, (), (), contacts, diag))

    def bound_geometry(shape, fragment):
        # Preserve the executed rotation/cut's scratch edge planes, while
        # binding every coordinate to the actual public fragment.  Constructor
        # orientation reversal is the only permitted representation change.
        saved = np.asarray(fragment.polygon)
        if np.array_equal(shape, saved):
            return shape
        if np.array_equal(shape[::-1], saved):
            return _reverse_polygon(shape)
        raise RuntimeError("Scratch geometry does not match its material footprint")

    # Rodrigues supports exist before its final binary64 projection.  Keeping
    # them through this transaction prevents rounded cut endpoints from being
    # reused as newly invented receiver/vacancy planes.  They are recomputed
    # from the same saved input and full actual motion after a restart, and
    # never enter the public state or checkpoint schema.
    geometry = {old.fragment_id: bound_geometry(
        rotate_polygon(old.polygon, omega[old.parcel.plate], dt), new)
        for old, new in zip(state.fragments, moved.fragments)}

    # Only genuinely overlapping original candidates can compete. Disjoint
    # equal materials may be iterated deterministically without choosing sides.
    neighbours = {f.fragment_id: [] for f in moved.fragments}
    by_id = {f.fragment_id: f for f in moved.fragments}
    successors = {identity: [] for identity in by_id}
    indegree = {identity: 0 for identity in by_id}
    overlap_records = []
    decisions = {}
    from .geometric_polarity import PolarityResolution
    unresolved = []
    for i, j in candidate_pairs(moved.fragments):
        a, b = moved.fragments[i], moved.fragments[j]
        overlap = intersect_convex(a.polygon, b.polygon, diagnostics=diag)
        if not len(overlap):
            continue
        if a.parcel.plate == b.parcel.plate:
            raise ValueError("Rigidly moved fragments of the same plate overlap")
        decision = None if polarity_resolver is None else polarity_resolver(a, b, overlap)
        if decision is None:
            decision = PolarityResolution(_compare_polarity(a, b), "material_contrast")
        if not isinstance(decision, PolarityResolution):
            raise ValueError("Polarity resolver must return a PolarityResolution or None")
        comparison = decision.comparison
        decisions[frozenset((a.fragment_id, b.fragment_id))] = decision
        record = GeometricOverlap(a.fragment_id, b.fragment_id,
            a.parcel.plate, b.parcel.plate, tuple(map(tuple, overlap)),
            polygon_area(overlap)*state.radius_km**2,
            "equal_buoyancy_and_age" if decision.basis == "material_contrast" else decision.basis)
        overlap_records.append(record)
        if comparison == 0:
            unresolved.append(record)
        else:
            winner, loser = (a, b) if comparison < 0 else (b, a)
            successors[winner.fragment_id].append(loser.fragment_id)
            indegree[loser.fragment_id] += 1
        neighbours[a.fragment_id].append(b.fragment_id)
        neighbours[b.fragment_id].append(a.fragment_id)
    if unresolved:
        raise UnresolvedPolarityError(unresolved)

    # Roundoff-aware comparison need not be transitive. Build the actual
    # overlap ordering explicitly rather than passing a nontransitive
    # comparator to sort. IDs order only unrelated ready fragments.
    ready = [identity for identity, degree in indegree.items() if degree == 0]
    heapq.heapify(ready)
    ordered = []
    while ready:
        identity = heapq.heappop(ready)
        ordered.append(by_id[identity])
        for loser in successors[identity]:
            indegree[loser] -= 1
            if indegree[loser] == 0:
                heapq.heappush(ready, loser)
    if len(ordered) != len(moved.fragments):
        unresolved = [replace(record, reason="priority_cycle") for record in overlap_records
                      if indegree[record.fragment_a] and indegree[record.fragment_b]]
        raise UnresolvedPolarityError(unresolved)

    retained, losses, descendants = [], [], {}
    accepted = {}
    for source in ordered:
        pieces, removed = [geometry[source.fragment_id]], []
        for neighbour in sorted(neighbours[source.fragment_id]):
            for receiver in accepted.get(neighbour, ()):
                remaining = []
                for polygon in pieces:
                    overlap = intersect_convex(polygon, geometry[receiver.fragment_id], diagnostics=diag)
                    if len(overlap):
                        removed.append((overlap, receiver, decisions[frozenset((source.fragment_id, neighbour))]))
                        remaining.extend(subtract_convex(polygon, geometry[receiver.fragment_id], diagnostics=diag))
                    else:
                        remaining.append(polygon)
                pieces = remaining
        if not removed:
            children = (source,)
        else:
            parts = split_geometry(source, pieces+[p for p, _, _ in removed],
                                   label=f"step:{moved.time_myr.hex()}")
            for fragment, shape in zip(parts, pieces+[p for p, _, _ in removed]):
                geometry[fragment.fragment_id] = bound_geometry(shape, fragment)
            children = parts[:len(pieces)]
            for fragment, (_, receiver, decision) in zip(parts[len(pieces):], removed):
                losses.append(GeometricLoss(fragment, source.fragment_id,
                    receiver.fragment_id, receiver.parcel.plate, moved.time_myr,
                    polarity_basis=decision.basis, polarity_evidence_ids=decision.evidence_ids))
        retained.extend(children)
        accepted[source.fragment_id] = children
        descendants[source.fragment_id] = tuple(child.fragment_id for child in children)

    # Old footprints partition the whole sphere. Their uncovered parts give
    # both the exact vacancy and the explicitly chosen departing local owner.
    # This closure is independent of the integration mesh's current triangles.
    index = _Index(retained)
    births, known = [], set(state.known_material_ids)
    for departed in state.fragments:
        gaps = [np.asarray(departed.polygon)]
        for covered in index.query(departed.polygon):
            remaining = []
            for polygon in gaps:
                overlap = intersect_convex(polygon, geometry[covered.fragment_id], diagnostics=diag)
                if len(overlap):
                    remaining.extend(subtract_convex(polygon, geometry[covered.fragment_id], diagnostics=diag))
                else:
                    remaining.append(polygon)
            gaps = remaining
            if not gaps:
                break
        for polygon in gaps:
            area = polygon_area(polygon)*state.radius_km**2
            parcel = birth_factory(departed.parcel.cell, departed.parcel.plate,
                                   area, moved.time_myr, len(births))
            if (not isinstance(parcel, SurfaceParcel) or parcel.plate != departed.parcel.plate
                    or parcel.cell != departed.parcel.cell or parcel.age_myr != 0.
                    or parcel.cold_mantle_volume_km3 != 0. or parcel.density_excess_mass_kg != 0.
                    or not math.isclose(parcel.area_km2, area, rel_tol=2e-13, abs_tol=0.)
                    or parcel.material_id in known):
                raise ValueError("Birth factory must supply fresh, hot material for the actual local vacancy")
            known.add(parcel.material_id)
            births.append(GeometricFragment(f"new-fragment:{moved.time_myr.hex()}:{len(births)}",
                tuple(map(tuple, polygon)), parcel, departed.fragment_id))

    resolved = GeometricSurfaceState(moved.time_myr, state.radius_km, tuple(retained+births), tuple(known))
    audit_partition(resolved, diagnostics=diag)
    contacts = extract_contacts(resolved.fragments, state.radius_km, omega)
    losses = _attach_losses(losses, contacts, retained, descendants, diag)
    diagnostics = _diagnostics(state, resolved, losses, births, contacts, diag)
    diagnostics["polarity_resolution_counts"] = {basis: sum(d.basis == basis for d in decisions.values())
                                                for basis in sorted({d.basis for d in decisions.values()})}
    return GeometricTransportResult(resolved, losses, tuple(births), contacts, diagnostics)
