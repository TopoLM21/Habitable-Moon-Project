"""Persisted spherical material footprints, independent of the integration mesh.

Fractions alone have no spatial arrangement. Import therefore requires one
complete material history per source triangle. Rotated footprints can be
projected into mixed cells without reconstructing or changing their geometry.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .fractional_surface import SurfaceParcel, EXTENSIVE_FIELDS, split_parcel
from .spherical_polygons import (GeometryDiagnostics, normalize_polygon, polygon_area,
                                 rotate_polygon, intersect_convex)

VERSION = "geometric-surface-1"
AREA_REL_TOL = 2e-10


@dataclass(frozen=True, slots=True)
class GeometricFragment:
    fragment_id: str
    polygon: tuple[tuple[float, float, float], ...]
    parcel: SurfaceParcel
    parent_fragment_id: str | None = None

    def __post_init__(self):
        if not isinstance(self.fragment_id, str) or not self.fragment_id:
            raise ValueError("Geometry needs a stable nonempty fragment ID")
        if not isinstance(self.parcel, SurfaceParcel):
            raise ValueError("Geometry needs explicit material properties")
        raw = np.asarray(self.polygon, dtype=float)
        checked = normalize_polygon(raw)
        if len(checked) < 3 or raw.shape != checked.shape:
            raise ValueError("Material footprint must have positive area and no repeated vertices")
        if not np.allclose(np.linalg.norm(raw, axis=1), 1., rtol=0., atol=16.*np.finfo(float).eps):
            raise ValueError("Material footprint vertices must be unit vectors")
        # Validate with normalized coordinates, but preserve saved binary64
        # values. Repeated checkpoint reads must not renormalize their bits.
        # Use a fan of difference-form determinants.  A short first edge makes
        # cross(raw[0],raw[1]) subtract nearly equal O(1) products, so its rounded
        # sign can flip a valid narrow footprint.  No coordinates are changed.
        orientation = math.fsum(float(np.dot(raw[0], np.cross(
            raw[index]-raw[0], raw[index+1]-raw[0])))
            for index in range(1, len(raw)-1))
        if orientation < 0.:
            raw = raw[::-1]
        object.__setattr__(self, "polygon", tuple(tuple(map(float, row)) for row in raw))
        if self.parent_fragment_id is not None and (
                not isinstance(self.parent_fragment_id, str) or not self.parent_fragment_id):
            raise ValueError("Fragment parent must be an explicit nonempty ID")


@dataclass(frozen=True, slots=True)
class GeometricSurfaceState:
    time_myr: float
    radius_km: float
    fragments: tuple[GeometricFragment, ...]
    known_material_ids: tuple[str, ...] = ()
    phase: str = "partition"
    version: str = VERSION

    def __post_init__(self):
        if (self.version != VERSION or self.phase not in ("partition", "advected_candidates")
                or not math.isfinite(self.time_myr) or self.time_myr < 0.
                or not math.isfinite(self.radius_km) or self.radius_km <= 0.):
            raise ValueError("Unsupported geometric surface or invalid physical clock/radius")
        fragments = tuple(self.fragments)
        if not fragments or any(not isinstance(f, GeometricFragment) for f in fragments):
            raise ValueError("Geometric surface requires material fragments")
        ids = [f.fragment_id for f in fragments]
        if len(ids) != len(set(ids)):
            raise ValueError("Geometric fragment IDs must be unique")
        for fragment in fragments:
            measured = polygon_area(fragment.polygon)*self.radius_km**2
            if not math.isclose(measured, fragment.parcel.area_km2, rel_tol=AREA_REL_TOL, abs_tol=0.):
                raise ValueError("Material area and its spherical footprint disagree")
        area = math.fsum(f.parcel.area_km2 for f in fragments)
        if not math.isclose(area, 4.*math.pi*self.radius_km**2, rel_tol=AREA_REL_TOL, abs_tol=0.):
            raise ValueError("Surface material area must equal the sphere area")
        origins = {f.parcel.material_id for f in fragments}
        if any(not isinstance(identity, str) or not identity for identity in self.known_material_ids):
            raise ValueError("Historical material origin IDs must be nonempty strings")
        known = set(self.known_material_ids)
        if known and not origins.issubset(known):
            raise ValueError("Geometry is missing material origin IDs")
        object.__setattr__(self, "fragments", fragments)
        object.__setattr__(self, "known_material_ids", tuple(sorted(known | origins)))


def from_fractional_surface(mesh, surface, radius_km):
    """Import an explicitly pure triangle partition, never invent mixed shapes."""
    geometric = mesh.physical_cell_areas_km2(radius_km)
    grouped = [[] for _ in range(mesh.cell_count)]
    if surface.cell_count != mesh.cell_count or not np.allclose(
            surface.cell_areas_km2, geometric, rtol=2e-13, atol=0.):
        raise ValueError("Fractional source does not match the geometric mesh/radius")
    for parcel in surface.parcels:
        grouped[parcel.cell].append(parcel)
    if any(len(values) != 1 for values in grouped):
        raise ValueError("Mixed fractions do not determine geometry; import requires one complete parcel per cell")
    fragments = tuple(GeometricFragment(f"source-fragment:{cell}",
        tuple(map(tuple, mesh.vertices[mesh.faces[cell]])), values[0]) for cell, values in enumerate(grouped))
    return GeometricSurfaceState(surface.time_myr, float(radius_km), fragments, surface.known_material_ids)


def rotations(state, omega, dt):
    values = np.asarray(omega, dtype=float)
    if (not math.isfinite(dt) or dt <= 0. or state.time_myr+dt <= state.time_myr
            or values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all()
            or any(f.parcel.plate >= len(values) for f in state.fragments)):
        raise ValueError("Geometric rotation requires finite plate vectors and a positive advancing timestep")
    return values


def rotate_surface(state, omega_rad_per_myr, dt_myr):
    dt = float(dt_myr)
    omega = rotations(state, omega_rad_per_myr, dt)
    owners = sorted({f.parcel.plate for f in state.fragments})
    common = all(np.array_equal(omega[p], omega[owners[0]]) for p in owners)
    moved = tuple(replace(f, polygon=tuple(map(tuple, rotate_polygon(f.polygon, omega[f.parcel.plate], dt))),
                          parcel=replace(f.parcel, age_myr=f.parcel.age_myr+dt)) for f in state.fragments)
    phase = state.phase if common else "advected_candidates"
    return replace(state, time_myr=state.time_myr+dt, fragments=moved, phase=phase)


def caps(polygons):
    centers, radii = [], []
    for polygon in polygons:
        points = np.asarray(polygon)
        center = np.sum(points, axis=0)
        center /= np.linalg.norm(center)
        angles = np.arctan2(np.linalg.norm(np.cross(points, center), axis=1), points@center)
        radius = float(np.max(angles))
        if radius >= math.pi/2:
            raise ValueError("Geometric broad phase requires small convex footprints")
        centers.append(center)
        radii.append(radius)
    return np.asarray(centers), np.asarray(radii)


def candidate_pairs(fragments):
    centers, radii = caps([f.polygon for f in fragments])
    tree = cKDTree(centers)
    max_radius = float(np.max(radii))
    for i, (center, radius) in enumerate(zip(centers, radii)):
        chord = 2.*math.sin(min(math.pi, radius+max_radius)/2.)+64.*np.finfo(float).eps
        for j in sorted(tree.query_ball_point(center, chord)):
            if j > i and np.linalg.norm(center-centers[j]) <= 2.*math.sin((radius+radii[j])/2.)+64.*np.finfo(float).eps:
                yield i, j


def split_geometry(fragment, polygons, *, cells=None, label="part"):
    """Partition one donor by one normalized geometric measure, including roundoff."""
    polygons = tuple(polygons)
    measured = np.asarray([polygon_area(p) for p in polygons])
    total = math.fsum(measured)
    original = polygon_area(fragment.polygon)
    if (not len(polygons) or np.any(measured <= 0.) or
            not math.isclose(total, original, rel_tol=AREA_REL_TOL, abs_tol=0.)):
        raise ValueError("Geometric children do not partition their source footprint")
    weights = measured/total
    # A single floating-point correction remains inside this same donor.
    weights[int(np.argmax(weights))] += 1.-math.fsum(weights)
    result = []
    for index, (polygon, weight) in enumerate(zip(polygons, weights)):
        parcel = split_parcel(fragment.parcel, float(weight),
            target_cell=fragment.parcel.cell if cells is None else int(cells[index]))
        result.append(GeometricFragment(f"{fragment.fragment_id}/{label}:{index}",
            tuple(map(tuple, polygon)), parcel, fragment.fragment_id))
    return tuple(result)


def project_to_mesh(mesh, state, *, diagnostics=None):
    """Conservative integration view; it never changes persisted footprints/IDs."""
    diag = diagnostics if diagnostics is not None else GeometryDiagnostics()
    polygons = tuple(mesh.vertices[face] for face in mesh.faces)
    centers, radii = caps(polygons)
    tree = cKDTree(centers)
    source_centers, source_radii = caps([f.polygon for f in state.fragments])
    max_radius = float(np.max(radii))
    result = []
    for fragment, center, radius in zip(state.fragments, source_centers, source_radii):
        chord = 2.*math.sin(min(math.pi, radius+max_radius)/2.)+64.*np.finfo(float).eps
        children, cells = [], []
        for cell in sorted(tree.query_ball_point(center, chord)):
            if np.linalg.norm(center-centers[cell]) > 2.*math.sin((radius+radii[cell])/2.)+64.*np.finfo(float).eps:
                continue
            polygon = intersect_convex(fragment.polygon, polygons[cell], diagnostics=diag)
            if len(polygon):
                children.append(polygon)
                cells.append(cell)
        result.extend(split_geometry(fragment, children, cells=cells, label="view"))
    return tuple(result)


def totals(fragments):
    fragments = tuple(fragments)
    return {key: math.fsum(getattr(f.parcel, key) for f in fragments) for key in EXTENSIVE_FIELDS}


def audit_partition(state, *, diagnostics=None):
    """Check actual nonoverlap; total sphere area is also checked by the state."""
    if state.phase != "partition":
        raise ValueError("Advected candidates have not resolved geometric overlap and vacancy")
    diag = diagnostics if diagnostics is not None else GeometryDiagnostics()
    for i, j in candidate_pairs(state.fragments):
        overlap = intersect_convex(state.fragments[i].polygon, state.fragments[j].polygon, diagnostics=diag)
        if len(overlap):
            raise ValueError("Geometric partition contains overlapping footprints")
    return asdict(diag)


def _digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                      allow_nan=False).encode("utf-8")).hexdigest()


def save_geometric_checkpoint(path, state, *, provenance=None):
    payload = dict(format=VERSION, state=asdict(state), provenance=provenance or {})
    payload["payload_sha256"] = _digest(payload)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
    return path


def load_geometric_checkpoint(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format") != VERSION:
        raise ValueError("Expected a checkpoint with persisted spherical geometry")
    saved = payload.pop("payload_sha256", None)
    if saved != _digest(payload):
        raise ValueError("Geometric checkpoint integrity mismatch")
    data = payload["state"]
    fragments = tuple(GeometricFragment(f["fragment_id"], tuple(map(tuple, f["polygon"])),
        SurfaceParcel(**f["parcel"]), f.get("parent_fragment_id")) for f in data.pop("fragments"))
    state = GeometricSurfaceState(fragments=fragments, **data)
    if state.phase == "partition":
        audit_partition(state)
    return state, payload.get("provenance", {})


def advance_geometric_surface(*args, **kwargs):
    from .geometric_transport import advance_geometric_surface as advance
    return advance(*args, **kwargs)
