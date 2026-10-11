"""Independent prescribed spatial traction on certified native footprints.

The source and cell-centroid quadrature retain the existing physical meaning.
Only the temporary integration footprints change: intersections are certified
native polygons, and no projected world-coordinate polygon enters a predicate.
Every positive intersection remains a sample; unrepresentable weights refuse
the assembly instead of removing material.  A saved load requires the trusted
mesh for native reassembly when it is read.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from fractions import Fraction
from functools import lru_cache
import json
import math

import numpy as np

from .geometric_basal_loading import mesh_fingerprint
from .geometric_intrinsic_geometry import IntrinsicGeometry
from .geometric_native_polygons import NativePolygon, native_intersection
from .geometric_native_surface import (
    NativeSurfaceState, _audit_native_coverage, native_surface_to_dict,
)
from .geometric_rigid_mantle_loading import (
    _EPS, _hash, _integer, _integrate, _number, _sha, _text, _vector,
)
from .geometric_spatial_mantle_loading import (
    MODEL, SOURCE_FRAME, PrescribedSpatialMantleSource,
    SpatialMantleQuadratureSample, _source_traction,
)
from .geometric_surface import AREA_REL_TOL


FORMAT = "geometric-native-physical-spatial-mantle-loading-1"
QUADRATURE = "certified_native_fragment_mesh_intersections_physical_cell_centroids_v1"


@dataclass(frozen=True, slots=True)
class NativePhysicalSpatialMantleLoad:
    surface_sha256: str
    mesh_sha256: str
    plate_ids: tuple[int, ...]
    drag_nm_s: tuple[tuple[float, ...], ...]
    torque_nm: tuple[float, ...]
    source: PrescribedSpatialMantleSource
    samples: tuple[SpatialMantleQuadratureSample, ...]
    load_sha256: str

    @property
    def provenance(self):
        return self.source.provenance


def native_surface_digest(surface):
    """Bind the complete native material state and all authoritative recipes."""
    if type(surface) is not NativeSurfaceState or surface.phase != "partition":
        raise ValueError("Native mantle loading requires a resolved native partition")
    return _hash(native_surface_to_dict(surface))


def _body(load):
    body = asdict(load)
    body.pop("load_sha256")
    return dict(format=FORMAT, model=MODEL, quadrature=QUADRATURE,
                source_frame=SOURCE_FRAME, transmission=1., **body)


def _canonical_source(source):
    if type(source) is not PrescribedSpatialMantleSource:
        raise ValueError("Explicit independently prescribed spatial mantle source required")
    checked = PrescribedSpatialMantleSource(**asdict(source))
    if checked != source:
        raise ValueError("Native spatial mantle source is not canonical")
    return checked


def _traction(source, radius_m, position):
    # Retain the inherited physical centroid rule exactly within its numerical
    # domain. Intermediate underflow in either the cross product or beta*R*f
    # cannot erase a component whose complete physical traction is nonzero.
    omega = tuple(Fraction.from_float(float(value)) for value in source.angular_velocity_rad_s)
    point = tuple(Fraction.from_float(float(value)) for value in position)
    exact = (omega[1]*point[2]-omega[2]*point[1],
             omega[2]*point[0]-omega[0]*point[2],
             omega[0]*point[1]-omega[1]*point[0])
    actual = np.cross(np.asarray(source.angular_velocity_rad_s), position)
    tolerance = Fraction.from_float(64*_EPS)
    for expected, component in zip(exact, actual):
        if (not math.isfinite(float(component))
                or abs(Fraction.from_float(float(component))-expected) > tolerance*abs(expected)):
            raise ValueError("Native physical traction cannot resolve the full prescribed source vector")
    traction = _source_traction(source, radius_m, position)
    _vector(traction, "Native independently prescribed physical traction")
    amplitude = Fraction.from_float(float(source.amplitude_constant)) + sum(
        Fraction.from_float(float(gradient))*component
        for gradient, component in zip(source.amplitude_gradient, point))
    factor = (Fraction.from_float(float(source.basal_drag_pa_s_m))
              *Fraction.from_float(float(radius_m))*amplitude)
    for crossing, component in zip(exact, traction):
        expected = factor*crossing
        if abs(Fraction.from_float(float(component))-expected) > tolerance*abs(expected):
            raise ValueError("Native physical traction cannot resolve the full prescribed traction product")
    return traction


@lru_cache(maxsize=8)
def _native_cells(vertices, faces):
    cells = tuple(NativePolygon.from_intrinsic(IntrinsicGeometry(
        tuple(vertices[index] for index in face))) for face in faces)
    if not cells or not math.isclose(math.fsum(cell.area_sr for cell in cells),
                                    4.*math.pi, rel_tol=AREA_REL_TOL, abs_tol=0.):
        raise ValueError("Native source integration mesh must cover the complete sphere")
    for index, left in enumerate(cells):
        for right in cells[index+1:]:
            if native_intersection(left, right) is not None:
                raise ValueError("Native source integration mesh contains overlapping cells")
    _audit_native_coverage(cells)
    return cells


def _mesh_cells(mesh):
    # Fingerprinting checks all physical mesh arrays before integer indexing.
    mesh_fingerprint(mesh)
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    centroids = np.asarray(mesh.centroids)
    areas = np.asarray(mesh.areas_unit_sphere)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2
            or faces.shape[1] != 3 or faces.dtype.kind not in "iu"
            or centroids.shape != (len(faces), 3) or areas.shape != (len(faces),)
            or np.any(faces < 0) or np.any(faces >= len(vertices))):
        raise ValueError("Native integration mesh requires explicit triangular cells")
    if any(abs(math.hypot(*map(float, vertex))-1.) > 16*_EPS for vertex in vertices):
        raise ValueError("Native integration mesh requires unit physical vertex rays")
    centers = vertices[faces].sum(axis=1)
    norms = np.linalg.norm(centers, axis=1, keepdims=True)
    if np.any(norms == 0.) or not np.isfinite(norms).all():
        raise ValueError("Native integration mesh requires well-defined physical cell centroids")
    centers = centers/norms
    for position in centroids:
        unit = _vector(tuple(map(float, position)), "Native physical cell centroid")
        if abs(math.fsum(value*value for value in unit)-1.) > 64*_EPS:
            raise ValueError("Native source integration requires unit physical cell centroids")
    # This is solely the arithmetic check for the existing physical quadrature
    # rule, not a geometric membership or clipping tolerance.
    if np.any(np.linalg.norm(centroids-centers, axis=1) > 64*_EPS):
        raise ValueError("Native physical cell centroids disagree with their actual face vertices")
    cells = _native_cells(tuple(tuple(map(float, row)) for row in vertices),
                          tuple(tuple(map(int, row)) for row in faces))
    for saved, geometry in zip(areas, cells):
        saved = _number(saved, "Native saved physical cell area", positive=True)
        if not math.isclose(saved, geometry.area_sr, rel_tol=AREA_REL_TOL, abs_tol=0.):
            raise ValueError("Native saved cell area disagrees with its certified face footprint")
    return cells


def _sample_identity(cell, identity):
    # Explicit length makes the cell/origin pairing unambiguous for arbitrary
    # stable fragment names. No owner or target motion determines this key.
    return f"native-view:{cell}:{len(identity.encode('utf-8'))}:{identity}"


def _native_samples(mesh, surface, source):
    cells = _mesh_cells(mesh)
    radius_m = _number(surface.radius_km, "Native surface radius", positive=True)*1000.
    if not math.isfinite(radius_m):
        raise ValueError("Native physical source radius is not representable")
    samples, by_cell = [], [[] for _ in cells]
    for fragment in surface.fragments:
        intersections = []
        for cell, geometry in enumerate(cells):
            overlap = native_intersection(fragment.geometry, geometry)
            if overlap is not None:
                # area_sr has a certified positive lower bound. It never
                # exposes a vanished binary64 display polygon as a footprint.
                intersections.append((cell, overlap.area_sr))
        measured = math.fsum(area for _, area in intersections)
        if (not intersections or not math.isfinite(measured) or measured <= 0.
                or not math.isclose(measured, fragment.geometry.area_sr,
                                    rel_tol=AREA_REL_TOL, abs_tol=0.)):
            raise ValueError("Native integration intersections do not cover their material fragment")
        expected = fragment.parcel.area_km2*1e6
        _number(expected, "Native source fragment area", positive=True)
        # Normalize only within this material's actual certified intersection
        # domain. The residual is assigned to its largest physical sample.
        weights = [area/measured for _, area in intersections]
        values = [expected*weight for weight in weights]
        largest = max(range(len(values)), key=values.__getitem__)
        values[largest] += expected-math.fsum(values)
        for (cell, _), area in zip(intersections, values):
            _number(area, "Native positive source intersection area", positive=True)
            position = tuple(map(float, mesh.centroids[cell]))
            sample = SpatialMantleQuadratureSample(cell,
                _sample_identity(cell, fragment.fragment_id), fragment.fragment_id,
                fragment.parcel.plate, area, position,
                _traction(source, radius_m, position))
            samples.append(sample)
            by_cell[cell].append(area)
    radius_squared_m2 = radius_m*radius_m
    for geometry, areas in zip(cells, by_cell):
        expected = geometry.area_sr*radius_squared_m2
        _number(expected, "Native source cell area", positive=True)
        if not areas or not math.isclose(math.fsum(areas), expected,
                                        rel_tol=AREA_REL_TOL, abs_tol=0.):
            raise ValueError("Native source sample areas do not conservatively fill their local cell")
    return tuple(sorted(samples, key=lambda sample:
                        (sample.cell, sample.source_fragment_id, sample.fragment_id)))


def assemble_native_physical_spatial_mantle_load(mesh, surface, source):
    surface_sha256 = native_surface_digest(surface)
    source = _canonical_source(source)
    samples = _native_samples(mesh, surface, source)
    owners = tuple(sorted({fragment.parcel.plate for fragment in surface.fragments}))
    radius_m = surface.radius_km*1000.
    drag, torque = _integrate(samples, owners, source, radius_m)
    temporary = NativePhysicalSpatialMantleLoad(surface_sha256, mesh_fingerprint(mesh),
        owners, drag, torque, source, samples, "")
    load = NativePhysicalSpatialMantleLoad(temporary.surface_sha256, temporary.mesh_sha256,
        owners, drag, torque, source, samples, _hash(_body(temporary)))
    return validate_native_physical_spatial_mantle_load(surface, load)


def validate_native_physical_spatial_mantle_load(surface, load, *, mesh=None,
                                               source=None, provenance=None):
    digest = native_surface_digest(surface)
    if type(load) is not NativePhysicalSpatialMantleLoad:
        raise ValueError("Expected an explicit NativePhysicalSpatialMantleLoad")
    _canonical_source(load.source)
    for name in ("surface_sha256", "mesh_sha256", "load_sha256"):
        _sha(getattr(load, name), name)
    if load.surface_sha256 != digest:
        raise ValueError("Native spatial mantle load is bound to another exact material surface")
    owners = tuple(sorted({fragment.parcel.plate for fragment in surface.fragments}))
    if (not isinstance(load.plate_ids, tuple)
            or tuple(_integer(owner, "Native compact material owner")
                     for owner in load.plate_ids) != owners):
        raise ValueError("Native spatial mantle load requires every canonical actual owner")
    if not isinstance(load.samples, tuple) or not load.samples:
        raise ValueError("Native mantle source requires immutable physical quadrature samples")
    fragments = {fragment.fragment_id: fragment for fragment in surface.fragments}
    per_fragment = {identity: [] for identity in fragments}
    keys, identities = [], set()
    radius_m = _number(surface.radius_km, "Native surface radius", positive=True)*1000.
    for sample in load.samples:
        if type(sample) is not SpatialMantleQuadratureSample:
            raise ValueError("Invalid native spatial mantle physical sample")
        cell = _integer(sample.cell, "Native source integration cell")
        _text(sample.fragment_id, "Native projected fragment identity")
        _text(sample.source_fragment_id, "Native original fragment identity")
        owner = _integer(sample.plate_id, "Native sample material owner")
        if (sample.fragment_id in identities or sample.source_fragment_id not in fragments
                or sample.fragment_id != _sample_identity(cell, sample.source_fragment_id)):
            raise ValueError("Repeated or foreign native spatial mantle sample identity")
        identities.add(sample.fragment_id)
        if owner != fragments[sample.source_fragment_id].parcel.plate:
            raise ValueError("Native spatial mantle sample changed material ownership")
        area = _number(sample.area_m2, "Native sample physical area", positive=True)
        position = _vector(sample.position, "Native source sample position")
        if abs(math.fsum(component*component for component in position)-1.) > 64*_EPS:
            raise ValueError("Native source samples require unit physical positions")
        _vector(sample.source_traction_pa, "Native prescribed source traction")
        if sample.source_traction_pa != _traction(load.source, radius_m, position):
            raise ValueError("Saved native traction disagrees with the independently prescribed field")
        per_fragment[sample.source_fragment_id].append(area)
        keys.append((cell, sample.source_fragment_id, sample.fragment_id))
    if keys != sorted(keys):
        raise ValueError("Native spatial mantle samples require canonical cell/origin order")
    for identity, areas in per_fragment.items():
        expected = fragments[identity].parcel.area_km2*1e6
        _number(expected, "Native material source area", positive=True)
        if not areas or not math.isclose(math.fsum(areas), expected,
                                        rel_tol=AREA_REL_TOL, abs_tol=0.):
            raise ValueError("Native source sample areas do not cover their original material fragment")
    if (not isinstance(load.drag_nm_s, tuple)
            or any(not isinstance(row, tuple) for row in load.drag_nm_s)
            or not isinstance(load.torque_nm, tuple)):
        raise ValueError("Native spatial operators must be immutable compact tuples")
    size = 3*len(owners)
    if (len(load.drag_nm_s) != size or any(len(row) != size for row in load.drag_nm_s)
            or len(load.torque_nm) != size):
        raise ValueError("Native spatial operators require three Euler degrees per actual owner")
    for row in load.drag_nm_s:
        for value in row:
            _number(value, "Native physical basal drag entry")
    for value in load.torque_nm:
        _number(value, "Native physical spatial source torque entry")
    drag, torque = _integrate(load.samples, owners, load.source, radius_m)
    if load.drag_nm_s != drag or load.torque_nm != torque:
        raise ValueError("Native spatial operators disagree with independent physical sample integration")
    for index in range(len(owners)):
        block = np.asarray(drag)[3*index:3*index+3, 3*index:3*index+3]
        scale = float(np.max(np.abs(block)))
        normalized = block/scale if scale else block
        if (not math.isfinite(scale) or scale == 0.
                or np.linalg.eigvalsh(normalized)[0] <= 64*_EPS*float(np.linalg.norm(normalized))):
            raise ValueError("Native mantle quadrature cannot resolve all material Euler degrees of freedom")
    if load.load_sha256 != _hash(_body(load)):
        raise ValueError("Native spatial mantle load integrity mismatch")
    if source is not None and source != load.source:
        raise ValueError("Prescribed native spatial mantle physical source changed")
    if provenance is not None and provenance != load.source.provenance:
        raise ValueError("Native spatial mantle source provenance changed")
    if mesh is not None:
        if mesh_fingerprint(mesh) != load.mesh_sha256:
            raise ValueError("Native spatial mantle source integration mesh changed")
        if assemble_native_physical_spatial_mantle_load(mesh, surface, load.source) != load:
            raise ValueError("Native spatial samples disagree with trusted native footprint reassembly")
    return load


def native_spatial_mantle_load_to_dict(surface, load):
    validate_native_physical_spatial_mantle_load(surface, load)
    # The native writer returns JSON-native lists, not an in-memory tuple DTO.
    return json.loads(json.dumps(dict(**_body(load), load_sha256=load.load_sha256),
                                 allow_nan=False))


def native_spatial_mantle_load_from_dict(raw, surface, *, mesh=None,
                                        source=None, provenance=None):
    if mesh is None:
        raise ValueError("Native spatial mantle loading requires trusted mesh reassembly on load")
    names = {field.name for field in fields(NativePhysicalSpatialMantleLoad)}
    metadata = {"format", "model", "quadrature", "source_frame", "transmission"}
    if (not isinstance(raw, dict) or set(raw) != names|metadata
            or any(raw[key] != value for key, value in (("format", FORMAT),
                ("model", MODEL), ("quadrature", QUADRATURE), ("source_frame", SOURCE_FRAME)))
            or isinstance(raw["transmission"], (bool, np.bool_)) or raw["transmission"] != 1.):
        raise ValueError("Invalid explicit native spatial mantle loading schema")
    try:
        data = {key: raw[key] for key in names}
        source_data = data["source"]
        if (not isinstance(source_data, dict)
                or set(source_data) != {field.name for field in fields(PrescribedSpatialMantleSource)}):
            raise ValueError("Invalid native spatial mantle source schema")
        source_data = dict(source_data)
        for name in ("angular_velocity_rad_s", "amplitude_gradient"):
            if not isinstance(source_data[name], list):
                raise ValueError("Native mantle source vectors require explicit JSON lists")
            source_data[name] = tuple(source_data[name])
        data["source"] = PrescribedSpatialMantleSource(**source_data)
        for name in ("plate_ids", "drag_nm_s", "torque_nm", "samples"):
            if not isinstance(data[name], list):
                raise ValueError("Native mantle collections require explicit JSON lists")
        data["plate_ids"] = tuple(data["plate_ids"])
        if any(not isinstance(row, list) for row in data["drag_nm_s"]):
            raise ValueError("Native spatial mantle drag requires explicit JSON rows")
        data["drag_nm_s"] = tuple(tuple(row) for row in data["drag_nm_s"])
        data["torque_nm"] = tuple(data["torque_nm"])
        samples = []
        sample_names = {field.name for field in fields(SpatialMantleQuadratureSample)}
        for item in data["samples"]:
            if not isinstance(item, dict) or set(item) != sample_names:
                raise ValueError("Invalid native spatial mantle source sample schema")
            item = dict(item)
            for name in ("position", "source_traction_pa"):
                if not isinstance(item[name], list):
                    raise ValueError("Native mantle sample vectors require explicit JSON lists")
                item[name] = tuple(item[name])
            samples.append(SpatialMantleQuadratureSample(**item))
        data["samples"] = tuple(samples)
        return validate_native_physical_spatial_mantle_load(surface,
            NativePhysicalSpatialMantleLoad(**data), mesh=mesh,
            source=source, provenance=provenance)
    except (KeyError, TypeError, IndexError, OverflowError) as error:
        raise ValueError("Malformed native spatial mantle physical source checkpoint") from error
