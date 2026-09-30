"""Physical shared arcs between persisted spherical material fragments.

Cell occupancy is deliberately absent from this interface.  Contacts exist
only where two different owners' convex polygons have the same boundary arc
and opposite interiors.  Material histories on one plate may meet without
forming a plate contact.  This module supplies geometry and kinematics, not
ridge forces, subduction polarity, or mechanical attachment.

Identifiers name a segment of a pair of persisted fragments.  They survive
rigid rotation and input reordering.  A geometric split creates new fragment
identities, and consequently new segment identities; tracking mechanical
attachment through that event requires an explicit transaction lineage.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Iterable, Protocol

import numpy as np
from scipy.spatial import cKDTree

from .fractional_surface import SurfaceParcel
from .spherical_polygons import (arc_length, normalize_polygon,
                                 shared_boundary_arcs)


class Fragment(Protocol):
    fragment_id: str
    polygon: object
    parcel: SurfaceParcel


@dataclass(frozen=True, slots=True)
class GeometricContact:
    """One minor great-circle arc; normal points from side A into side B.

    Relative velocities are B minus A.  A positive normal velocity separates
    the sides; a negative value converges.  ``normal_area_rate_km2_per_myr``
    is the exact line integral of that velocity, not midpoint speed times
    arc length.  ``moment_arm_cross_normal_km2`` is the exact integral of
    r cross normal along the arc.  Both remain additive after arc splitting.
    """
    contact_id: str
    fragment_a: str
    fragment_b: str
    material_a: str
    material_b: str
    plate_a: int
    plate_b: int
    edge_a: int
    edge_b: int
    start: tuple[float, float, float]
    end: tuple[float, float, float]
    midpoint: tuple[float, float, float]
    tangent: tuple[float, float, float]
    normal_a_to_b: tuple[float, float, float]
    length_km: float
    normal_velocity_km_per_myr: float
    tangential_velocity_km_per_myr: float
    normal_area_rate_km2_per_myr: float
    integrated_position_unit_km: tuple[float, float, float]
    moment_arm_cross_normal_km2: tuple[float, float, float]


def _vector(values):
    return tuple(float(value) for value in values)


def _radius(value):
    if isinstance(value, bool) or not math.isfinite(float(value)) or value <= 0.:
        raise ValueError("Contact radius must be finite and positive")
    return float(value)


def _omega(values, owners):
    if values is None:
        return np.zeros((max(owners, default=-1)+1, 3), dtype=float)
    array = np.asarray(values, dtype=float)
    if (array.ndim != 2 or array.shape[1] != 3 or not np.isfinite(array).all()
            or any(owner < 0 or owner >= len(array) for owner in owners)):
        raise ValueError("Contact angular velocities require finite [plate, 3] values")
    return array


def _edge_index(polygon, midpoint):
    """Find the supporting edge of a known returned shared minor arc.

    The primitive has already tested arc overlap and opposite interiors.
    We only recover provenance here; endpoint proximity is not a contact test.
    """
    best = None
    for i, (start, end) in enumerate(zip(polygon, np.roll(polygon, -1, axis=0))):
        normal = np.cross(start, end-start)
        normal /= np.linalg.norm(normal)
        # Half-open endpoints are irrelevant because an arc midpoint lies in
        # an edge interior.  The angular excess rejects other coplanar edges.
        excess = abs(arc_length(start, midpoint)+arc_length(midpoint, end)
                     - arc_length(start, end))
        error = abs(float(np.dot(midpoint, normal)))+excess
        candidate = (error, i)
        if best is None or candidate < best:
            best = candidate
    if best is None:
        raise ValueError("Contact arc has no polygon edge")
    return best[1]


def _contact(a, b, polygon_a, polygon_b, start, end, radius, omega):
    theta = arc_length(start, end)
    midpoint = start+end
    midpoint /= np.linalg.norm(midpoint)
    edge_a = _edge_index(polygon_a, midpoint)
    edge_b = _edge_index(polygon_b, midpoint)
    # A tiny overlap arc may lie on a much longer source edge.  Its supporting
    # plane is more accurately represented by that longer edge than by taking
    # the cross product of almost coincident contact endpoints.
    a0, a1 = polygon_a[edge_a], polygon_a[(edge_a+1) % len(polygon_a)]
    b0, b1 = polygon_b[edge_b], polygon_b[(edge_b+1) % len(polygon_b)]
    if arc_length(a0, a1) >= arc_length(b0, b1):
        inward_a = np.cross(a0, a1-a0)
    else:
        inward_a = -np.cross(b0, b1-b0)
    inward_a /= np.linalg.norm(inward_a)
    normal = -inward_a
    tangent = np.cross(inward_a, midpoint)
    identity = (a.fragment_id, b.fragment_id, edge_a, edge_b)
    encoded = json.dumps(identity, separators=(",", ":"), ensure_ascii=False)
    contact_id = "contact:"+hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:32]
    delta = omega[b.parcel.plate]-omega[a.parcel.plate]
    velocity = radius*np.cross(delta, midpoint)
    integral_position = radius*2.*math.sin(.5*theta)*midpoint
    moment = radius*np.cross(integral_position, normal)
    return GeometricContact(
        contact_id=contact_id, fragment_a=a.fragment_id, fragment_b=b.fragment_id,
        material_a=a.parcel.material_id, material_b=b.parcel.material_id,
        plate_a=a.parcel.plate, plate_b=b.parcel.plate, edge_a=edge_a, edge_b=edge_b,
        start=_vector(start), end=_vector(end), midpoint=_vector(midpoint),
        tangent=_vector(tangent), normal_a_to_b=_vector(normal),
        length_km=radius*theta,
        normal_velocity_km_per_myr=float(np.dot(velocity, normal)),
        tangential_velocity_km_per_myr=float(np.dot(velocity, tangent)),
        normal_area_rate_km2_per_myr=float(np.dot(delta, moment)),
        integrated_position_unit_km=_vector(integral_position),
        moment_arm_cross_normal_km2=_vector(moment))


def _between(a, b, polygon_a, polygon_b, radius, omega):
    if a.parcel.plate == b.parcel.plate:
        return ()
    return tuple(_contact(a, b, polygon_a, polygon_b, np.asarray(start).copy(),
                          np.asarray(end).copy(), radius, omega)
                 for start, end in shared_boundary_arcs(polygon_a, polygon_b))


def contacts_between(a: Fragment, b: Fragment, radius_km: float,
                     omega_rad_per_myr=None) -> tuple[GeometricContact, ...]:
    """Contacts of a fragment pair, independent of their mesh-cell labels."""
    radius = _radius(radius_km)
    if a.fragment_id == b.fragment_id:
        raise ValueError("Contact fragments require distinct stable identities")
    a, b = sorted((a, b), key=lambda item: item.fragment_id)
    omega = _omega(omega_rad_per_myr, (a.parcel.plate, b.parcel.plate))
    return _between(a, b, normalize_polygon(a.polygon), normalize_polygon(b.polygon),
                    radius, omega)


def _cap(polygon):
    center = np.sum(polygon, axis=0)
    center /= np.linalg.norm(center)
    angles = np.arctan2(np.linalg.norm(np.cross(polygon, center), axis=1),
                        polygon@center)
    radius = float(np.max(angles))
    # Convex caps of radius below pi/2 contain every minor boundary arc and
    # the polygon interior.  A hemisphere-sized polygon uses an all-pairs
    # broad phase rather than incorrectly rejecting a possible neighbor.
    return center, radius


def extract_contacts(fragments: Iterable[Fragment], radius_km: float,
                     omega_rad_per_myr=None) -> tuple[GeometricContact, ...]:
    """Extract every actual material contact using a conservative broad phase.

    Spherical caps only reject distant pairs; exact shared-edge tests decide
    all returned contacts.  No mesh edges, area fractions, nearest-owner
    choice, or material-density ordering define the geometry.
    """
    radius = _radius(radius_km)
    fragments = tuple(sorted(fragments, key=lambda item: item.fragment_id))
    if len({item.fragment_id for item in fragments}) != len(fragments):
        raise ValueError("Contact fragments require unique stable identities")
    omega = _omega(omega_rad_per_myr, tuple(item.parcel.plate for item in fragments))
    if len(fragments) < 2:
        return ()
    polygons = tuple(normalize_polygon(item.polygon) for item in fragments)
    caps = tuple(_cap(polygon) for polygon in polygons)
    centers = np.asarray([item[0] for item in caps])
    radii = np.asarray([item[1] for item in caps])
    max_radius = float(np.max(radii))
    padding = 128.*np.finfo(float).eps
    if max_radius >= .5*math.pi:
        pairs = ((i, j) for i in range(len(fragments)) for j in range(i+1, len(fragments)))
    else:
        chord = min(2., 2.*math.sin(max_radius)+padding)
        pairs = sorted(cKDTree(centers).query_pairs(chord))
    result = []
    for i, j in pairs:
        if fragments[i].parcel.plate == fragments[j].parcel.plate:
            continue
        separation = math.atan2(float(np.linalg.norm(np.cross(centers[i], centers[j]))),
                                float(np.dot(centers[i], centers[j])))
        if max_radius < .5*math.pi and separation > radii[i]+radii[j]+padding:
            continue
        result.extend(_between(fragments[i], fragments[j], polygons[i], polygons[j],
                               radius, omega))
    return tuple(sorted(result, key=lambda item: item.contact_id))
