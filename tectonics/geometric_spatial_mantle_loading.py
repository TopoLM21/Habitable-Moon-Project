"""Explicit positive spatial mantle traction prescribed before plate motion.

The manufactured control field is tau_source(x) = beta R f(x) (Omega cross x),
with f(x) = constant + gradient dot x > 0 on the entire sphere. The independent
linear resistance is tau_drag = -beta v_plate. Neither source reads an owner
identity, a solved plate velocity, or a target torque. Persisted material
footprints determine centroid quadrature; source samples generate T and drag
samples independently generate D. This is a separate, explicitly selected
control source, without changing the Genesis quadratic/thermal source.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import math

import numpy as np

from .geometric_basal_loading import mesh_fingerprint
from .geometric_boundary import surface_digest
from .geometric_rigid_mantle_loading import (
    QUADRATURE, _EPS, _hash, _integer, _integrate, _number, _sha, _text, _vector,
)
from .geometric_surface import GeometricSurfaceState, project_to_mesh


FORMAT = "geometric-physical-spatial-mantle-loading-1"
MODEL = "manufactured_prescribed_positive_spatial_mantle_traction_independent_linear_drag_v1"
SOURCE_FRAME = "fixed_global_cartesian_prescribed_mantle_angular_velocity_and_positive_affine_unit_position_amplitude"


@dataclass(frozen=True, slots=True)
class PrescribedSpatialMantleSource:
    angular_velocity_rad_s: tuple[float, float, float]
    basal_drag_pa_s_m: float
    amplitude_constant: float
    amplitude_gradient: tuple[float, float, float]
    provenance: str

    def __post_init__(self):
        _vector(self.angular_velocity_rad_s, "Prescribed spatial mantle angular velocity")
        _number(self.basal_drag_pa_s_m, "Explicit basal drag", positive=True)
        constant = _number(self.amplitude_constant, "Spatial amplitude constant", positive=True)
        gradient = _vector(self.amplitude_gradient, "Spatial amplitude gradient")
        if constant <= math.hypot(*gradient):
            raise ValueError("Spatial amplitude constant must exceed the gradient norm for global positivity")
        _text(self.provenance, "Spatial mantle physical source provenance")


@dataclass(frozen=True, slots=True)
class SpatialMantleQuadratureSample:
    cell: int
    fragment_id: str
    source_fragment_id: str
    plate_id: int
    area_m2: float
    position: tuple[float, float, float]
    source_traction_pa: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class PhysicalSpatialMantleLoad:
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


def _body(load):
    body = asdict(load)
    body.pop("load_sha256")
    return dict(format=FORMAT, model=MODEL, quadrature=QUADRATURE,
                source_frame=SOURCE_FRAME, transmission=1., **body)


def _source_traction(source, radius_m, position):
    amplitude = math.fsum((source.amplitude_constant,
        *(a*float(x) for a, x in zip(source.amplitude_gradient, position))))
    return tuple(map(float, source.basal_drag_pa_s_m*radius_m*amplitude
        *np.cross(np.asarray(source.angular_velocity_rad_s), position)))


def validate_physical_spatial_mantle_load(surface, load, *, mesh=None, source=None, provenance=None):
    if not isinstance(surface, GeometricSurfaceState) or surface.phase != "partition":
        raise ValueError("Spatial mantle loading requires a resolved geometric partition")
    if not isinstance(load, PhysicalSpatialMantleLoad) or not isinstance(load.source, PrescribedSpatialMantleSource):
        raise ValueError("Expected an explicit PhysicalSpatialMantleLoad")
    source_values = PrescribedSpatialMantleSource(**asdict(load.source))
    if source_values != load.source:
        raise ValueError("Spatial mantle source is not canonical")
    for name in ("surface_sha256", "mesh_sha256", "load_sha256"):
        _sha(getattr(load, name), name)
    if load.surface_sha256 != surface_digest(surface):
        raise ValueError("Spatial mantle load is bound to another exact material surface")
    expected_owners = tuple(sorted({fragment.parcel.plate for fragment in surface.fragments}))
    if (not isinstance(load.plate_ids, tuple)
            or tuple(_integer(owner, "Compact material owner") for owner in load.plate_ids) != expected_owners):
        raise ValueError("Spatial mantle load requires every canonical actual owner")
    if not isinstance(load.samples, tuple) or not load.samples:
        raise ValueError("Spatial mantle source requires immutable physical quadrature samples")
    by_id = {fragment.fragment_id: fragment for fragment in surface.fragments}
    per_fragment = {key: [] for key in by_id}
    sample_keys, identities = [], set()
    radius = _number(surface.radius_km, "Surface radius", positive=True)*1000.
    for sample in load.samples:
        if not isinstance(sample, SpatialMantleQuadratureSample):
            raise ValueError("Invalid spatial mantle physical sample")
        cell = _integer(sample.cell, "Source integration cell")
        _text(sample.fragment_id, "Projected fragment identity")
        _text(sample.source_fragment_id, "Original fragment identity")
        owner = _integer(sample.plate_id, "Sample material owner")
        if sample.fragment_id in identities or sample.source_fragment_id not in by_id:
            raise ValueError("Repeated or foreign spatial mantle sample identity")
        identities.add(sample.fragment_id)
        if owner != by_id[sample.source_fragment_id].parcel.plate:
            raise ValueError("Spatial mantle sample changed material ownership")
        area = _number(sample.area_m2, "Sample physical area", positive=True)
        position = _vector(sample.position, "Source sample position")
        if abs(math.fsum(component*component for component in position)-1.) > 64*_EPS:
            raise ValueError("Spatial mantle source samples require unit physical positions")
        _vector(sample.source_traction_pa, "Prescribed physical spatial source traction")
        if sample.source_traction_pa != _source_traction(load.source, radius, position):
            raise ValueError("Saved traction does not match the independently prescribed spatial mantle field")
        per_fragment[sample.source_fragment_id].append(area)
        sample_keys.append((cell, sample.source_fragment_id, sample.fragment_id))
    if sample_keys != sorted(sample_keys):
        raise ValueError("Spatial mantle samples must follow canonical projection order")
    for identity, areas in per_fragment.items():
        expected = by_id[identity].parcel.area_km2*1e6
        if not areas or abs(math.fsum(areas)-expected) > 2e-10*expected:
            raise ValueError("Spatial mantle sample areas do not cover their original material fragment")
    if (not isinstance(load.drag_nm_s, tuple) or any(not isinstance(row, tuple) for row in load.drag_nm_s)
            or not isinstance(load.torque_nm, tuple)):
        raise ValueError("Spatial mantle operators must be immutable compact tuples")
    for row in load.drag_nm_s:
        for value in row:
            _number(value, "Physical basal drag entry")
    for value in load.torque_nm:
        _number(value, "Physical spatial source torque entry")
    expected_drag, expected_torque = _integrate(load.samples, load.plate_ids, load.source, radius)
    if load.drag_nm_s != expected_drag or load.torque_nm != expected_torque:
        raise ValueError("Spatial mantle operators disagree with independent physical sample integration")
    drag = np.asarray(load.drag_nm_s)
    for index in range(len(load.plate_ids)):
        block = drag[3*index:3*index+3, 3*index:3*index+3]
        scale = float(np.linalg.norm(block))
        if not math.isfinite(scale) or scale == 0. or np.linalg.eigvalsh(block)[0] <= 64*_EPS*scale:
            raise ValueError("Spatial mantle quadrature cannot resolve all material Euler degrees of freedom")
    if load.load_sha256 != _hash(_body(load)):
        raise ValueError("Spatial mantle load integrity mismatch")
    if mesh is not None and mesh_fingerprint(mesh) != load.mesh_sha256:
        raise ValueError("Spatial mantle source integration mesh changed")
    if source is not None and source != load.source:
        raise ValueError("Prescribed spatial mantle physical source changed")
    if provenance is not None and provenance != load.source.provenance:
        raise ValueError("Spatial mantle source provenance changed")
    if mesh is not None:
        actual = assemble_physical_spatial_mantle_load(mesh, surface, load.source)
        if actual != load:
            raise ValueError("Spatial mantle samples do not match trusted geometric source reassembly")
    return load


def assemble_physical_spatial_mantle_load(mesh, surface, source):
    if not isinstance(surface, GeometricSurfaceState) or surface.phase != "partition":
        raise ValueError("Spatial mantle loading requires a resolved geometric partition")
    if not isinstance(source, PrescribedSpatialMantleSource):
        raise ValueError("Explicit independently prescribed spatial mantle source required")
    # Also validate objects whose frozen dataclass constructor was bypassed.
    PrescribedSpatialMantleSource(**asdict(source))
    radius = _number(surface.radius_km, "Surface radius", positive=True)*1000.
    view = sorted(project_to_mesh(mesh, surface),
        key=lambda item: (item.parcel.cell, item.parent_fragment_id, item.fragment_id))
    samples = tuple(SpatialMantleQuadratureSample(item.parcel.cell, item.fragment_id,
        item.parent_fragment_id, item.parcel.plate, item.parcel.area_km2*1e6,
        tuple(map(float, mesh.centroids[item.parcel.cell])),
        _source_traction(source, radius, mesh.centroids[item.parcel.cell])) for item in view)
    owners = tuple(sorted({fragment.parcel.plate for fragment in surface.fragments}))
    drag, torque = _integrate(samples, owners, source, radius)
    temporary = PhysicalSpatialMantleLoad(surface_digest(surface), mesh_fingerprint(mesh),
        owners, drag, torque, source, samples, "")
    load = PhysicalSpatialMantleLoad(temporary.surface_sha256, temporary.mesh_sha256,
        owners, drag, torque, source, samples, _hash(_body(temporary)))
    return validate_physical_spatial_mantle_load(surface, load)


def spatial_mantle_load_to_dict(surface, load):
    validate_physical_spatial_mantle_load(surface, load)
    return dict(**_body(load), load_sha256=load.load_sha256)


def spatial_mantle_load_from_dict(raw, surface, *, mesh=None, source=None, provenance=None):
    names = {field.name for field in fields(PhysicalSpatialMantleLoad)}
    metadata = {"format", "model", "quadrature", "source_frame", "transmission"}
    if (not isinstance(raw, dict) or set(raw) != names|metadata
            or any(raw[key] != value for key, value in (("format", FORMAT), ("model", MODEL),
                ("quadrature", QUADRATURE), ("source_frame", SOURCE_FRAME)))
            or isinstance(raw["transmission"], bool) or raw["transmission"] != 1.):
        raise ValueError("Invalid explicit spatial mantle loading schema")
    data = {key: raw[key] for key in names}
    try:
        source_data = data["source"]
        if not isinstance(source_data, dict) or set(source_data) != {field.name for field in fields(PrescribedSpatialMantleSource)}:
            raise ValueError("Invalid spatial mantle physical source schema")
        source_data = dict(source_data)
        for name in ("angular_velocity_rad_s", "amplitude_gradient"):
            if not isinstance(source_data[name], list):
                raise ValueError("Prescribed spatial mantle vectors require explicit JSON lists")
            source_data[name] = tuple(source_data[name])
        data["source"] = PrescribedSpatialMantleSource(**source_data)
        for name in ("plate_ids", "drag_nm_s", "torque_nm", "samples"):
            if not isinstance(data[name], list):
                raise ValueError("Spatial mantle collections require explicit JSON lists")
        data["plate_ids"] = tuple(data["plate_ids"])
        if any(not isinstance(row, list) for row in data["drag_nm_s"]):
            raise ValueError("Spatial mantle drag requires explicit compact JSON rows")
        data["drag_nm_s"] = tuple(tuple(row) for row in data["drag_nm_s"])
        data["torque_nm"] = tuple(data["torque_nm"])
        samples = []
        sample_names = {field.name for field in fields(SpatialMantleQuadratureSample)}
        for item in data["samples"]:
            if not isinstance(item, dict) or set(item) != sample_names:
                raise ValueError("Invalid spatial mantle source sample schema")
            item = dict(item)
            for name in ("position", "source_traction_pa"):
                if not isinstance(item[name], list):
                    raise ValueError("Spatial source sample vectors require explicit JSON lists")
                item[name] = tuple(item[name])
            samples.append(SpatialMantleQuadratureSample(**item))
        data["samples"] = tuple(samples)
        return validate_physical_spatial_mantle_load(surface, PhysicalSpatialMantleLoad(**data),
            mesh=mesh, source=source, provenance=provenance)
    except (KeyError, TypeError, IndexError, OverflowError) as error:
        raise ValueError("Malformed spatial mantle physical source checkpoint") from error
