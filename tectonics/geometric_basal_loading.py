"""Source-bound SI basal operators through a temporary conservative mesh view.

Persisted spherical fragment footprints and material histories are untouched.
Every projected component keeps its owner and explicit total thermal lid.
The existing conditional-component vertex-transmission/cell-centroid quadrature
is reused; this is not exact integration over a fragment footprint. Driving
transmission and independent linear basal drag keep their existing meanings.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, replace
import hashlib
import json
import math
from numbers import Integral, Real

import numpy as np

from .basal_coupling import PrescribedBasalParameters
from .fractional_dynamics import MODEL, QUADRATURE, solve_fractional_basal_dynamics
from .fractional_surface import EXTENSIVE_FIELDS, FractionalSurfaceState
from .geometric_boundary import surface_digest
from .geometric_surface import GeometricSurfaceState, project_to_mesh, totals

FORMAT = "geometric-frozen-basal-loading-1"
SOURCE_FRAME = "fixed_mesh_prescribed_physical_source_no_rotation_removal"
_DIAGNOSTICS = ("input_fragment_count", "projected_fragment_count",
                "material_origin_count", "integrated_area_m2",
                "max_relative_material_residual", "max_relative_cell_capacity_residual")


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def _number(value, name, *, positive=False, nonnegative=False):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not math.isfinite(float(value))):
        raise ValueError(f"{name} must be a finite real physical number")
    value = float(value)
    if (positive and value <= 0.) or (nonnegative and value < 0.):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'}")
    return value


def mesh_fingerprint(mesh):
    """Fingerprint every mesh array used by projection and basal quadrature."""
    data = {}
    for name in ("vertices", "faces", "centroids", "areas_unit_sphere"):
        raw = np.asarray(getattr(mesh, name))
        if raw.dtype.kind not in "iuf" or not np.isfinite(raw).all():
            raise ValueError("Basal mesh must contain finite real geometry")
        data[name] = raw.tolist()
    return _hash(data)


def _parameters(parameters):
    if not isinstance(parameters, PrescribedBasalParameters):
        raise ValueError("Explicit prescribed basal source parameters are required")
    return tuple((item.name, int(getattr(parameters, item.name)) if item.name == "seed"
                  else float(getattr(parameters, item.name))) for item in fields(parameters))


def _lid(surface, supplied):
    if not isinstance(supplied, Mapping):
        raise ValueError("An explicit thermal lid map per persisted fragment is required")
    expected = {item.fragment_id for item in surface.fragments}
    if set(supplied) != expected:
        raise ValueError("Thermal lid map must contain every current fragment exactly once")
    return tuple((key, _number(supplied[key], "Total thermal lid thickness", nonnegative=True))
                 for key in sorted(expected))


@dataclass(frozen=True, slots=True)
class GeometricBasalLoad:
    surface_sha256: str
    mesh_sha256: str
    plate_ids: tuple[int, ...]
    drag_nm_s: tuple[tuple[float, ...], ...]
    torque_nm: tuple[float, ...]
    source_parameters: tuple[tuple[str, int | float], ...]
    thermal_lid_by_fragment: tuple[tuple[str, float], ...]
    thermal_lid_sha256: str
    provenance: str
    projection_diagnostics: tuple[tuple[str, int | float], ...]
    load_sha256: str


def _body(load):
    values = asdict(load)
    del values["load_sha256"]
    parameters = PrescribedBasalParameters(**dict(load.source_parameters))
    return dict(format=FORMAT, model=MODEL, quadrature=QUADRATURE, source_frame=SOURCE_FRAME,
                source_metadata=parameters.metadata(), **values)


def _validate(load):
    if not isinstance(load, GeometricBasalLoad):
        raise ValueError("Expected an explicit GeometricBasalLoad")
    for name in ("surface_sha256", "mesh_sha256", "thermal_lid_sha256", "load_sha256"):
        value = getattr(load, name)
        if (not isinstance(value, str) or len(value) != 64
                or any(letter not in "0123456789abcdef" for letter in value)):
            raise ValueError(f"Invalid basal {name}")
    ids = load.plate_ids
    if (not isinstance(ids, tuple) or not ids
            or any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
                   or value < 0 for value in ids)
            or tuple(sorted(set(ids))) != ids):
        raise ValueError("Basal plate IDs must be sorted unique nonnegative actual owners")
    if not isinstance(load.provenance, str) or not load.provenance.strip():
        raise ValueError("Basal load source requires explicit provenance")
    parameter_names = tuple(item.name for item in fields(PrescribedBasalParameters))
    if (not isinstance(load.source_parameters, tuple)
            or any(not isinstance(item, tuple) or len(item) != 2 for item in load.source_parameters)
            or tuple(name for name, _ in load.source_parameters) != parameter_names):
        raise ValueError("Basal source must explicitly retain every physical parameter")
    parameters = PrescribedBasalParameters(**dict(load.source_parameters))
    if (not isinstance(load.thermal_lid_by_fragment, tuple) or not load.thermal_lid_by_fragment
            or any(not isinstance(item, tuple) or len(item) != 2 for item in load.thermal_lid_by_fragment)):
        raise ValueError("Basal load requires the explicit per-fragment thermal lid")
    lid_ids = []
    for key, value in load.thermal_lid_by_fragment:
        if not isinstance(key, str) or not key:
            raise ValueError("Thermal lid keys must identify persisted fragments")
        lid_ids.append(key)
        _number(value, "Stored thermal lid", nonnegative=True)
    if tuple(sorted(set(lid_ids))) != tuple(lid_ids):
        raise ValueError("Thermal lid entries must be unique and sorted")
    if _hash(load.thermal_lid_by_fragment) != load.thermal_lid_sha256:
        raise ValueError("Stored thermal lid checksum changed")
    if (not isinstance(load.projection_diagnostics, tuple)
            or any(not isinstance(item, tuple) or len(item) != 2 for item in load.projection_diagnostics)
            or tuple(name for name, _ in load.projection_diagnostics) != _DIAGNOSTICS):
        raise ValueError("Basal projection diagnostics are incomplete")
    for name, value in load.projection_diagnostics:
        _number(value, f"Stored {name}", nonnegative=True)
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
           or value <= 0 for _, value in load.projection_diagnostics[:3]):
        raise ValueError("Basal projection counts must be positive integers")
    size = 3*len(ids)
    if (not isinstance(load.drag_nm_s, tuple)
            or any(not isinstance(row, tuple) for row in load.drag_nm_s)
            or not isinstance(load.torque_nm, tuple)):
        raise ValueError("Basal operators must be immutable tuples")
    try:
        drag = np.asarray(load.drag_nm_s, dtype=float)
        torque = np.asarray(load.torque_nm, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("Invalid basal operator shape") from error
    if drag.shape != (size, size) or torque.shape != (size,):
        raise ValueError("Basal operators must have compact three-DOF owner dimensions")
    for row in load.drag_nm_s:
        for value in row:
            _number(value, "Basal drag entry")
    for value in load.torque_nm:
        _number(value, "Basal torque entry")
    for index in range(len(ids)):
        block = drag[3*index:3*index+3, 3*index:3*index+3]
        scale = float(np.linalg.norm(block))
        if (not math.isfinite(scale) or not scale
                or np.linalg.norm(block-block.T) > 64*np.finfo(float).eps*scale):
            raise ValueError("Basal drag blocks must be finite symmetric positive definite")
        eig = np.linalg.eigvalsh(.5*(block+block.T))
        if not np.isfinite(eig).all() or eig[0] <= 64*np.finfo(float).eps*scale:
            raise ValueError("Degenerate basal plate quadrature cannot resolve all Euler degrees of freedom")
        if np.any(drag[3*index:3*index+3, :3*index] != 0.) or np.any(drag[3*index:3*index+3, 3*index+3:] != 0.):
            raise ValueError("Basal drag must not contain hidden inter-plate contact forces")
    if _hash(_body(load)) != load.load_sha256:
        raise ValueError("Stored basal load checksum changed")
    return parameters


def validate_geometric_basal_load(surface, load, *, mesh=None, parameters=None,
                                 thermal_lid_by_fragment=None, provenance=None):
    """Validate a physical source record without evolving the frozen surface."""
    if not isinstance(surface, GeometricSurfaceState) or surface.phase != "partition":
        raise ValueError("Basal loading requires a resolved geometric partition")
    _validate(load)
    if load.surface_sha256 != surface_digest(surface):
        raise ValueError("Basal load is bound to another exact material surface")
    if load.plate_ids != tuple(sorted({item.parcel.plate for item in surface.fragments})):
        raise ValueError("Basal compact owners do not match the surface")
    if set(dict(load.thermal_lid_by_fragment)) != {item.fragment_id for item in surface.fragments}:
        raise ValueError("Basal thermal lid does not cover the current fragments")
    if mesh is not None and load.mesh_sha256 != mesh_fingerprint(mesh):
        raise ValueError("Basal restart integration mesh changed")
    if parameters is not None and load.source_parameters != _parameters(parameters):
        raise ValueError("Basal restart prescribed source changed")
    if thermal_lid_by_fragment is not None and load.thermal_lid_by_fragment != _lid(surface, thermal_lid_by_fragment):
        raise ValueError("Basal restart thermal lid changed")
    if provenance is not None and load.provenance != provenance:
        raise ValueError("Basal restart load provenance changed")
    return load


def assemble_geometric_basal_load(mesh, surface, parameters, thermal_lid_by_fragment, provenance):
    """Reuse component quadrature, without raster ownership or material edits.

    The thermal lid is required per persisted fragment even when other scalar
    material fields exist. Cell-centroid sampling and conditional vertex
    transmission are inherited approximations, not resolved subcell stresses.
    """
    if not isinstance(surface, GeometricSurfaceState) or surface.phase != "partition":
        raise ValueError("Basal loading requires a resolved geometric partition")
    source = _parameters(parameters)
    lid_entries = _lid(surface, thermal_lid_by_fragment)
    if not isinstance(provenance, str) or not provenance.strip():
        raise ValueError("Prescribed load needs explicit physical source provenance")
    ids = tuple(sorted({item.parcel.plate for item in surface.fragments}))
    lookup = {pid: index for index, pid in enumerate(ids)}
    lid = dict(lid_entries)
    # This is an integration view only: its descendants never replace the
    # persisted fragments, their identities, inventories or ages.
    # Fix reduction order by integration cell and persisted identity. This is
    # addressing only: no owner or material is selected or discarded. It makes
    # a tuple reorder of the same bound surface reproduce checkpoint bytes.
    view = tuple(sorted(project_to_mesh(mesh, surface),
                        key=lambda item: (item.parcel.cell, item.parent_fragment_id, item.fragment_id)))
    h = np.asarray([lid[item.parent_fragment_id] for item in view])
    parcels = tuple(replace(item.parcel, plate=lookup[item.parcel.plate]) for item in view)
    capacities = tuple(float(v) for v in mesh.physical_cell_areas_km2(surface.radius_km))
    fractional = FractionalSurfaceState(surface.time_myr, capacities, parcels,
                                       known_material_ids=surface.known_material_ids)
    result = solve_fractional_basal_dynamics(mesh, fractional, surface.radius_km,
        parameters, h, plate_count=len(ids), source_frame="fixed_mesh", remove_net_rotation=False)
    size = 3*len(ids)
    drag = np.zeros((size, size))
    for index, block in enumerate(result.basal_drag_tensor_nm_s):
        drag[3*index:3*index+3, 3*index:3*index+3] = block
    before, after = totals(surface.fragments), totals(view)
    material_residual = max(abs(after[key]-before[key])/max(abs(before[key]), np.finfo(float).tiny)
                            for key in EXTENSIVE_FIELDS)
    by_cell = np.bincount([item.parcel.cell for item in view],
                         weights=[item.parcel.area_km2 for item in view], minlength=mesh.cell_count)
    cell_residual = float(np.max(np.abs(by_cell-np.asarray(capacities))/np.asarray(capacities)))
    diagnostics = tuple(zip(_DIAGNOSTICS, (len(surface.fragments), len(view),
        len(surface.known_material_ids), sum(capacities)*1e6, material_residual, cell_residual)))
    values = dict(surface_sha256=surface_digest(surface), mesh_sha256=mesh_fingerprint(mesh),
        plate_ids=ids, drag_nm_s=tuple(tuple(map(float, row)) for row in drag),
        torque_nm=tuple(map(float, result.basal_driving_torque_nm.ravel())),
        source_parameters=source, thermal_lid_by_fragment=lid_entries,
        thermal_lid_sha256=_hash(lid_entries), provenance=provenance,
        projection_diagnostics=diagnostics)
    temporary = GeometricBasalLoad(**values, load_sha256="0"*64)
    load = GeometricBasalLoad(**values, load_sha256=_hash(_body(temporary)))
    return validate_geometric_basal_load(surface, load)


def basal_load_to_dict(surface, load):
    validate_geometric_basal_load(surface, load)
    return dict(**_body(load), load_sha256=load.load_sha256)


def basal_load_from_dict(raw, surface, *, mesh=None, parameters=None,
                         thermal_lid_by_fragment=None, provenance=None):
    """Restore a bound source record, optionally verifying trusted reassembly.

    The checksum checks content consistency, not source authenticity. Supplying
    all four trusted source inputs also rebuilds the quadrature and verifies
    operators instead of trusting a recomputed checksum on fabricated forces.
    """
    expected = {item.name for item in fields(GeometricBasalLoad)} | {
        "format", "model", "quadrature", "source_frame", "source_metadata"}
    if not isinstance(raw, dict) or set(raw) != expected:
        raise ValueError("Basal checkpoint has missing or unknown fields")
    if any(raw[name] != value for name, value in (("format", FORMAT), ("model", MODEL),
               ("quadrature", QUADRATURE), ("source_frame", SOURCE_FRAME))):
        raise ValueError("Basal checkpoint uses another physical model or source frame")
    values = {item.name: raw[item.name] for item in fields(GeometricBasalLoad)}
    try:
        for name in ("source_parameters", "thermal_lid_by_fragment", "projection_diagnostics"):
            values[name] = tuple(tuple(item) for item in values[name])
        values["plate_ids"] = tuple(values["plate_ids"])
        values["drag_nm_s"] = tuple(tuple(row) for row in values["drag_nm_s"])
        values["torque_nm"] = tuple(values["torque_nm"])
        load = GeometricBasalLoad(**values)
        restored_parameters = _validate(load)
        if raw["source_metadata"] != restored_parameters.metadata():
            raise ValueError("Basal checkpoint interpretation disagrees with its physical parameters")
        validate_geometric_basal_load(surface, load, mesh=mesh, parameters=parameters,
            thermal_lid_by_fragment=thermal_lid_by_fragment, provenance=provenance)
        if all(value is not None for value in (mesh, parameters, thermal_lid_by_fragment, provenance)):
            actual = assemble_geometric_basal_load(mesh, surface, parameters, thermal_lid_by_fragment, provenance)
            if actual != load:
                raise ValueError("Stored basal operators do not match trusted source reassembly")
        return load
    except (TypeError, KeyError, IndexError) as error:
        raise ValueError("Invalid basal checkpoint record") from error
