"""Explicit manufactured rigid mantle flow, independent of solved plate motion.

This source has full transmission and a separately declared positive drag:
tau_source(x) = beta R (Omega_m cross x), tau_drag = -beta v_plate.
It is not the quadratic-potential/thermal-transmission Genesis source. Every
physical centroid sample and traction is saved; driving torque and drag are
integrated independently, without fitting either to a solved angular speed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
from numbers import Integral, Real

import numpy as np

from .geometric_basal_loading import mesh_fingerprint
from .geometric_boundary import surface_digest
from .geometric_surface import GeometricSurfaceState, project_to_mesh


FORMAT = "geometric-physical-rigid-mantle-loading-1"
MODEL = "manufactured_prescribed_rigid_mantle_traction_independent_linear_drag_v1"
QUADRATURE = "persisted_fragment_projection_physical_centroid_samples_v1"
SOURCE_FRAME = "fixed_global_cartesian_prescribed_mantle_angular_velocity_rad_s"
_EPS = np.finfo(float).eps


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def _number(value, name, *, positive=False):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not math.isfinite(float(value)) or (positive and value <= 0.)):
        raise ValueError(f"{name} must be a finite {'positive ' if positive else ''}real")
    return float(value)


def _vector(value, name):
    if not isinstance(value, tuple) or len(value) != 3:
        raise ValueError(f"{name} must be an immutable Cartesian three-vector")
    return tuple(_number(item, name) for item in value)


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be explicit nonempty text")
    return value


def _integer(value, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 0):
        raise ValueError(f"{name} must be a nonnegative actual integer")
    return int(value)


def _sha(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"Invalid {name}")


@dataclass(frozen=True, slots=True)
class PrescribedRigidMantleSource:
    angular_velocity_rad_s: tuple[float, float, float]
    basal_drag_pa_s_m: float
    provenance: str

    def __post_init__(self):
        _vector(self.angular_velocity_rad_s, "Prescribed mantle angular velocity")
        _number(self.basal_drag_pa_s_m, "Explicit basal drag", positive=True)
        _text(self.provenance, "Rigid mantle physical source provenance")


@dataclass(frozen=True, slots=True)
class RigidMantleQuadratureSample:
    cell: int
    fragment_id: str
    source_fragment_id: str
    plate_id: int
    area_m2: float
    position: tuple[float, float, float]
    source_traction_pa: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class PhysicalRigidMantleLoad:
    surface_sha256: str
    mesh_sha256: str
    plate_ids: tuple[int, ...]
    drag_nm_s: tuple[tuple[float, ...], ...]
    torque_nm: tuple[float, ...]
    source: PrescribedRigidMantleSource
    samples: tuple[RigidMantleQuadratureSample, ...]
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
    # This independently prescribed spatial field never reads a plate velocity.
    return tuple(map(float, source.basal_drag_pa_s_m*radius_m
                     *np.cross(np.asarray(source.angular_velocity_rad_s), position)))


def _integrate(samples, owners, source, radius_m):
    index = {owner: slot for slot, owner in enumerate(owners)}
    size = 3*len(owners)
    drag = np.zeros((size, size), dtype=np.longdouble)
    torque = np.zeros(size, dtype=np.longdouble)
    radius, beta = np.longdouble(radius_m), np.longdouble(source.basal_drag_pa_s_m)
    for sample in samples:
        start = 3*index[sample.plate_id]
        area, x = np.longdouble(sample.area_m2), np.asarray(sample.position, dtype=np.longdouble)
        # The velocity/lever use the same physical position, so C(x)^T C(x)
        # preserves virtual work even in the final rounded unit coordinates.
        cross = np.array(((0., -x[2], x[1]), (x[2], 0., -x[0]),
                          (-x[1], x[0], 0.)), dtype=np.longdouble)
        drag[start:start+3, start:start+3] += beta*radius*radius*area*(cross.T@cross)
        traction = np.asarray(sample.source_traction_pa, dtype=np.longdouble)
        torque[start:start+3] += radius*area*np.cross(x, traction)
    return tuple(tuple(map(float, row)) for row in drag), tuple(map(float, torque))


def validate_physical_rigid_mantle_load(surface, load, *, mesh=None, source=None, provenance=None):
    if not isinstance(surface, GeometricSurfaceState) or surface.phase != "partition":
        raise ValueError("Rigid mantle loading requires a resolved geometric partition")
    if not isinstance(load, PhysicalRigidMantleLoad) or not isinstance(load.source, PrescribedRigidMantleSource):
        raise ValueError("Expected an explicit PhysicalRigidMantleLoad")
    # Recheck immutable values even when a caller bypassed dataclass construction.
    source_values = PrescribedRigidMantleSource(**asdict(load.source))
    if source_values != load.source:
        raise ValueError("Rigid mantle source is not canonical")
    for name in ("surface_sha256", "mesh_sha256", "load_sha256"):
        _sha(getattr(load, name), name)
    if load.surface_sha256 != surface_digest(surface):
        raise ValueError("Rigid mantle load is bound to another exact material surface")
    expected_owners = tuple(sorted({fragment.parcel.plate for fragment in surface.fragments}))
    if (not isinstance(load.plate_ids, tuple)
            or tuple(_integer(owner, "Compact material owner") for owner in load.plate_ids) != expected_owners):
        raise ValueError("Rigid mantle load requires every canonical actual owner")
    if not isinstance(load.samples, tuple) or not load.samples:
        raise ValueError("Rigid mantle source requires immutable physical quadrature samples")
    by_id = {fragment.fragment_id: fragment for fragment in surface.fragments}
    per_fragment = {key: [] for key in by_id}
    sample_keys, identities = [], set()
    radius = _number(surface.radius_km, "Surface radius", positive=True)*1000.
    for sample in load.samples:
        if not isinstance(sample, RigidMantleQuadratureSample):
            raise ValueError("Invalid rigid mantle physical sample")
        cell = _integer(sample.cell, "Source integration cell")
        _text(sample.fragment_id, "Projected fragment identity")
        _text(sample.source_fragment_id, "Original fragment identity")
        owner = _integer(sample.plate_id, "Sample material owner")
        if sample.fragment_id in identities or sample.source_fragment_id not in by_id:
            raise ValueError("Repeated or foreign rigid mantle sample identity")
        identities.add(sample.fragment_id)
        if owner != by_id[sample.source_fragment_id].parcel.plate:
            raise ValueError("Rigid mantle sample changed material ownership")
        area = _number(sample.area_m2, "Sample physical area", positive=True)
        position = _vector(sample.position, "Source sample position")
        if abs(math.fsum(component*component for component in position)-1.) > 64*_EPS:
            raise ValueError("Rigid mantle source samples require unit physical positions")
        _vector(sample.source_traction_pa, "Prescribed physical source traction")
        if sample.source_traction_pa != _source_traction(load.source, radius, position):
            raise ValueError("Saved traction does not match the independently prescribed mantle field")
        per_fragment[sample.source_fragment_id].append(area)
        sample_keys.append((cell, sample.source_fragment_id, sample.fragment_id))
    if sample_keys != sorted(sample_keys):
        raise ValueError("Rigid mantle samples must follow canonical projection order")
    for identity, areas in per_fragment.items():
        expected = by_id[identity].parcel.area_km2*1e6
        if not areas or abs(math.fsum(areas)-expected) > 2e-10*expected:
            raise ValueError("Rigid mantle sample areas do not cover their original material fragment")
    if (not isinstance(load.drag_nm_s, tuple) or any(not isinstance(row, tuple) for row in load.drag_nm_s)
            or not isinstance(load.torque_nm, tuple)):
        raise ValueError("Rigid mantle operators must be immutable compact tuples")
    for row in load.drag_nm_s:
        for value in row:
            _number(value, "Physical basal drag entry")
    for value in load.torque_nm:
        _number(value, "Physical source torque entry")
    expected_drag, expected_torque = _integrate(load.samples, load.plate_ids, load.source, radius)
    if load.drag_nm_s != expected_drag or load.torque_nm != expected_torque:
        raise ValueError("Rigid mantle operators disagree with independent physical sample integration")
    drag = np.asarray(load.drag_nm_s)
    for index in range(len(load.plate_ids)):
        block = drag[3*index:3*index+3, 3*index:3*index+3]
        scale = float(np.linalg.norm(block))
        if not math.isfinite(scale) or scale == 0. or np.linalg.eigvalsh(block)[0] <= 64*_EPS*scale:
            raise ValueError("Rigid mantle quadrature cannot resolve all material Euler degrees of freedom")
    if load.load_sha256 != _hash(_body(load)):
        raise ValueError("Rigid mantle load integrity mismatch")
    if mesh is not None and mesh_fingerprint(mesh) != load.mesh_sha256:
        raise ValueError("Rigid mantle source integration mesh changed")
    if source is not None and source != load.source:
        raise ValueError("Prescribed rigid mantle physical source changed")
    if provenance is not None and provenance != load.source.provenance:
        raise ValueError("Rigid mantle source provenance changed")
    if mesh is not None:
        # Geometry-aware authenticity: a rehashed altered sample cannot replace
        # the trusted mesh projection of these actual persisted footprints.
        actual = assemble_physical_rigid_mantle_load(mesh, surface, load.source)
        if actual != load:
            raise ValueError("Rigid mantle samples do not match trusted geometric source reassembly")
    return load


def assemble_physical_rigid_mantle_load(mesh, surface, source):
    if not isinstance(surface, GeometricSurfaceState) or surface.phase != "partition":
        raise ValueError("Rigid mantle loading requires a resolved geometric partition")
    if not isinstance(source, PrescribedRigidMantleSource):
        raise ValueError("Explicit independently prescribed rigid mantle source required")
    radius = _number(surface.radius_km, "Surface radius", positive=True)*1000.
    view = sorted(project_to_mesh(mesh, surface),
                  key=lambda item: (item.parcel.cell, item.parent_fragment_id, item.fragment_id))
    samples = tuple(RigidMantleQuadratureSample(item.parcel.cell, item.fragment_id,
        item.parent_fragment_id, item.parcel.plate, item.parcel.area_km2*1e6,
        tuple(map(float, mesh.centroids[item.parcel.cell])),
        _source_traction(source, radius, mesh.centroids[item.parcel.cell])) for item in view)
    owners = tuple(sorted({fragment.parcel.plate for fragment in surface.fragments}))
    drag, torque = _integrate(samples, owners, source, radius)
    temporary = PhysicalRigidMantleLoad(surface_digest(surface), mesh_fingerprint(mesh),
        owners, drag, torque, source, samples, "")
    load = PhysicalRigidMantleLoad(temporary.surface_sha256, temporary.mesh_sha256,
        owners, drag, torque, source, samples, _hash(_body(temporary)))
    return validate_physical_rigid_mantle_load(surface, load)


def rigid_mantle_load_to_dict(surface, load):
    validate_physical_rigid_mantle_load(surface, load)
    return dict(**_body(load), load_sha256=load.load_sha256)


def rigid_mantle_load_from_dict(raw, surface, *, mesh=None, source=None, provenance=None):
    names = {field.name for field in fields(PhysicalRigidMantleLoad)}
    metadata = {"format", "model", "quadrature", "source_frame", "transmission"}
    if (not isinstance(raw, dict) or set(raw) != names|metadata
            or any(raw[key] != value for key, value in (("format", FORMAT), ("model", MODEL),
                ("quadrature", QUADRATURE), ("source_frame", SOURCE_FRAME)))
            or isinstance(raw["transmission"], bool) or raw["transmission"] != 1.):
        raise ValueError("Invalid explicit rigid mantle loading schema")
    data = {key: raw[key] for key in names}
    try:
        source_data = data["source"]
        if not isinstance(source_data, dict) or set(source_data) != {field.name for field in fields(PrescribedRigidMantleSource)}:
            raise ValueError("Invalid rigid mantle physical source schema")
        source_data = dict(source_data)
        if not isinstance(source_data["angular_velocity_rad_s"], list):
            raise ValueError("Prescribed mantle angular velocity requires an explicit JSON list")
        source_data["angular_velocity_rad_s"] = tuple(source_data["angular_velocity_rad_s"])
        data["source"] = PrescribedRigidMantleSource(**source_data)
        for name in ("plate_ids", "drag_nm_s", "torque_nm", "samples"):
            if not isinstance(data[name], list):
                raise ValueError("Rigid mantle collections require explicit JSON lists")
        data["plate_ids"] = tuple(data["plate_ids"])
        if any(not isinstance(row, list) for row in data["drag_nm_s"]):
            raise ValueError("Rigid mantle drag requires explicit compact JSON rows")
        data["drag_nm_s"] = tuple(tuple(row) for row in data["drag_nm_s"])
        data["torque_nm"] = tuple(data["torque_nm"])
        samples = []
        sample_names = {field.name for field in fields(RigidMantleQuadratureSample)}
        for item in data["samples"]:
            if not isinstance(item, dict) or set(item) != sample_names:
                raise ValueError("Invalid rigid mantle source sample schema")
            item = dict(item)
            for name in ("position", "source_traction_pa"):
                if not isinstance(item[name], list):
                    raise ValueError("Source sample vectors require explicit JSON lists")
                item[name] = tuple(item[name])
            samples.append(RigidMantleQuadratureSample(**item))
        data["samples"] = tuple(samples)
        return validate_physical_rigid_mantle_load(surface, PhysicalRigidMantleLoad(**data),
            mesh=mesh, source=source, provenance=provenance)
    except (KeyError, TypeError, IndexError, OverflowError) as error:
        raise ValueError("Malformed rigid mantle physical source checkpoint") from error
