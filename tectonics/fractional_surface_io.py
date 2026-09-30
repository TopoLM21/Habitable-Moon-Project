"""Explicit import/checkpoint/refinement for the experimental fractional surface.

These files are not mature/Genesis continuation checkpoints. The sparse state
retains minority material owners and ages that a winner-only raster cannot
represent. Import is one-way; no lossy export into LithosphereState is provided.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .fractional_surface import (
    FractionalSurfaceState, SurfaceParcel, state_from_json, state_to_json,
)


FORMAT = "fractional-surface-checkpoint-1"
_AREA_QUADRATURE_RTOL = 2e-13


def _payload_digest(data):
    encoded = json.dumps(data, sort_keys=True, ensure_ascii=False,
                         allow_nan=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def mesh_fingerprint(mesh):
    digest = hashlib.sha256()
    for value, dtype in ((mesh.vertices, "<f8"), (mesh.faces, "<i8"),
                         (mesh.areas_unit_sphere, "<f8")):
        array = np.ascontiguousarray(value, dtype=dtype)
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _field(value, count, name):
    array = np.asarray(value)
    if array.shape != (count,) or array.dtype.kind not in "bifu" or not np.isfinite(array).all():
        raise ValueError(f"Invalid fractional import field {name}")
    return array


def surface_from_lithosphere(mesh, state, radius_km, *, fracture_memory=None):
    """Copy an all-ocean raster into independent, explicitly owned components.

    Each cell initially contains one component. Later mixtures must remain in
    FractionalSurfaceState: averaging age before a nonlinear cooling law is not
    an equivalent representation. Mechanical fields are frozen import values;
    this adapter neither advances heat nor creates an energy reservoir.
    """
    if not math.isfinite(radius_km) or radius_km <= 0:
        raise ValueError("Radius must be finite and positive")
    count = mesh.cell_count
    areas = mesh.physical_cell_areas_km2(radius_km)
    owner = _field(state.cell_plate, count, "cell_plate")
    if owner.dtype.kind not in "iu" or np.any(owner < 0):
        raise ValueError("Fractional import requires nonnegative integer owners")
    from .lithosphere import CrustType
    crust_type = _field(state.crust_type, count, "crust_type")
    if np.any(crust_type != int(CrustType.OCEANIC)):
        raise ValueError("Fractional experiment currently supports oceanic material only")
    # Even an invisible continental fringe cannot be discarded by this import.
    for name in ("continental_fraction", "continental_volume_km3", "sediment_volume_km3"):
        value = getattr(state, name)
        if value is not None and np.any(_field(value, count, name) != 0):
            raise ValueError(f"Fractional oceanic experiment cannot import nonzero {name}")
    required = ("oceanic_volume_km3", "mantle_lithosphere_thickness_km",
                "mantle_lithosphere_density_anomaly_kg_m3")
    if any(getattr(state, name) is None for name in required):
        raise ValueError("Fractional import requires explicit ocean volume and mechanical mantle fields")
    basalt, cold_h, density = (_field(getattr(state, name), count, name) for name in required)
    if np.any(basalt <= 0) or np.any(cold_h < 0) or np.any(density < 0):
        raise ValueError("Fractional oceanic import requires positive basalt and nonnegative cold support")
    age = _field(state.crust_age_myr, count, "crust_age_myr")
    if np.any(age < 0):
        raise ValueError("Material age must be nonnegative")
    names = ("tidal_damage", "rift_extension", "extension_age_myr",
             "collision_seam_weakness", "intraplate_stress", "supercontinent_heat",
             "continental_lithosphere_age_myr", "mantle_depletion_fraction", "craton_strength")
    memory = {name: _field(getattr(state, name), count, name)
              for name in names if getattr(state, name) is not None}
    if fracture_memory is not None:
        for name in ("damage", "cooling_stress_pa", "water_access", "yield_ratio",
                     "strength_pa", "eligible", "consumed_band"):
            memory["fracture_" + name] = _field(getattr(fracture_memory, name), count, name)
        if not np.array_equal(memory["fracture_damage"], memory["tidal_damage"]):
            raise ValueError("Fracture and surface damage disagree")
    parcels = tuple(SurfaceParcel(
        cell=i, plate=int(owner[i]), material_id=f"source:{i}",
        area_km2=float(areas[i]), oceanic_volume_km3=float(basalt[i]),
        cold_mantle_volume_km3=float(areas[i]*cold_h[i]),
        density_excess_mass_kg=float(areas[i]*cold_h[i]*density[i]*1e9),
        age_myr=float(age[i]),
        material_fields=tuple((name, float(values[i])) for name, values in sorted(memory.items())),
    ) for i in range(count))
    return FractionalSurfaceState(float(state.time_myr), tuple(map(float, areas)), parcels)


def save_fractional_checkpoint(path, mesh, state, radius_km, *, provenance=None):
    """Write a new independent checkpoint; never overwrite a source experiment."""
    path = Path(path)
    payload = state_to_json(state)
    expected = mesh.physical_cell_areas_km2(radius_km)
    if (not math.isfinite(radius_km) or radius_km <= 0 or
            expected.shape != (state.cell_count,) or
            not np.allclose(expected, state.cell_areas_km2, rtol=_AREA_QUADRATURE_RTOL, atol=0.)):
        raise ValueError("Fractional checkpoint area/radius does not match its mesh")
    envelope = dict(format=FORMAT, mesh_sha256=mesh_fingerprint(mesh),
                    radius_km=float(radius_km), state=payload, provenance=provenance or {})
    envelope["payload_sha256"] = _payload_digest(envelope)
    text = json.dumps(envelope, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)
    return path


def load_fractional_checkpoint(path, mesh, radius_km):
    """Load only the independent fractional format and verify its geometry."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("format") != FORMAT:
        raise ValueError("Expected an independent fractional surface checkpoint")
    saved_digest = data.pop("payload_sha256", None)
    if saved_digest != _payload_digest(data):
        raise ValueError("Fractional checkpoint payload integrity mismatch")
    if (data.get("mesh_sha256") != mesh_fingerprint(mesh) or
            not math.isfinite(radius_km) or radius_km <= 0 or data.get("radius_km") != radius_km):
        raise ValueError("Fractional checkpoint geometry differs from requested mesh/radius")
    state = state_from_json(data["state"])
    expected = mesh.physical_cell_areas_km2(radius_km)
    if (expected.shape != (state.cell_count,) or
            not np.allclose(expected, state.cell_areas_km2, rtol=_AREA_QUADRATURE_RTOL, atol=0.)):
        raise ValueError("Fractional checkpoint has inconsistent cell areas")
    return state, data.get("provenance", {})


def refine_fractional_surface(state, source_mesh, target_mesh, radius_km):
    """Split every component among its actual canonical triangular descendants.

    Fractional ownership and distinct ages survive even when the raster's
    majority owner would be unchanged. This is conservative initialization on
    a finer mesh, not additional resolved information or an overlap remesher.
    """
    from .mesh import build_icosphere
    def subdivision(mesh):
        level, count = 0, 20
        while count < mesh.cell_count:
            level, count = level + 1, count * 4
        canonical = build_icosphere(level)
        if count != mesh.cell_count or mesh_fingerprint(canonical) != mesh_fingerprint(mesh):
            raise ValueError("Fractional refinement requires canonical icospheres")
        return level
    source_level, target_level = subdivision(source_mesh), subdivision(target_mesh)
    if target_level < source_level:
        raise ValueError("Fractional coarsening is not implemented")
    geometric_old = source_mesh.physical_cell_areas_km2(radius_km)
    old_areas = np.asarray(state.cell_areas_km2)
    if (old_areas.shape != geometric_old.shape or
            not np.allclose(geometric_old, old_areas, rtol=_AREA_QUADRATURE_RTOL, atol=0.)):
        raise ValueError("Fractional state does not match source geometry")
    if target_level == source_level:
        return state
    geometric_new = target_mesh.physical_cell_areas_km2(radius_km)
    factor = 4 ** (target_level-source_level)
    children = geometric_new.reshape(source_mesh.cell_count, factor)
    if not np.allclose(children.sum(axis=1), old_areas, rtol=_AREA_QUADRATURE_RTOL, atol=0):
        raise ValueError("Refinement children do not cover source cells")
    # Spherical triangle quadrature evaluates parent and children separately;
    # their sums can differ by a few ulps. Use ONE inherited area measure for
    # both capacity and material. Correcting only material creates artificial
    # single-plate overlaps/vacancies that transport would interpret as flux.
    weights = children/children.sum(axis=1)[:, None]
    capacities = old_areas[:, None]*weights
    largest = np.argmax(weights, axis=1)
    capacities[np.arange(source_mesh.cell_count), largest] += old_areas-capacities.sum(axis=1)
    new_areas = capacities.ravel()
    if not np.allclose(new_areas, geometric_new, rtol=_AREA_QUADRATURE_RTOL, atol=0.):
        raise ValueError("Conservative child capacities exceed geometric quadrature roundoff")
    parcels = []
    extensive = ("area_km2", "oceanic_volume_km3", "cold_mantle_volume_km3", "density_excess_mass_kg")
    for parcel in state.parcels:
        fractions = weights[parcel.cell]
        quantities = {}
        for name in extensive:
            values = float(getattr(parcel, name)) * fractions
            # Return numerical roundoff to a child of this same parcel only.
            values[np.argmax(fractions)] += float(getattr(parcel, name))-float(values.sum())
            quantities[name] = values
        for offset in range(factor):
            parcels.append(replace(parcel, cell=parcel.cell*factor+offset,
                           **{name: float(values[offset]) for name, values in quantities.items()}))
    return FractionalSurfaceState(state.time_myr, tuple(map(float, new_areas)), tuple(parcels),
                                  material_model=state.material_model, version=state.version,
                                  known_material_ids=state.known_material_ids)
