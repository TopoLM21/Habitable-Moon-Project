"""Conservative hierarchical refinement of a paired Genesis continuation.

Each triangular parent is exactly covered by its 4**delta spherical children.
Intensive material memory is piecewise constant; extensive volumes are divided
by child area. Refinement adds sampling points, not new physical information.
Plate IDs, motion, subgrid rotations, slab histories and global budgets survive.
Coarsening is deliberately unsupported: a coarse cell cannot represent several
plate owners or preserve resolved fracture topology in the present solver.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, fields, replace
from numbers import Integral

import numpy as np

from .genesis_starter import StarterModel
from .genesis_starter_fracture import YoungShellFracture
from .mesh import build_icosphere, connected_components
from .plates import Plate, PlateSystem


# Explicit schemas distinguish physical cell totals from cell averages and
# plume-population arrays. Unknown arrays fail closed instead of losing memory.
CELL_FIELDS = {
    "state": ("cell_plate", "crust_type", "crust_age_myr", "crust_thickness_km",
        "tidal_damage", "rift_extension", "extension_age_myr", "collision_seam_weakness",
        "intraplate_stress", "supercontinent_heat", "continental_fraction",
        "continental_volume_km3", "mantle_lithosphere_thickness_km",
        "mantle_lithosphere_density_anomaly_kg_m3", "sediment_volume_km3",
        "continental_lithosphere_age_myr", "mantle_depletion_fraction", "craton_strength",
        "oceanic_volume_km3"),
    "cycle": ("felsic_potential",),
    "topo": ("elevation_m",),
    "mantle_flow": ("cell_omega_rad_per_myr",),
    "plume_state": ("last_flux", "cumulative_exposure_myr", "cumulative_root_erosion_km",
        "last_head_flux", "last_tail_flux"),
    "plume_rifting_state": ("last_extension_forcing", "cumulative_extension_impulse_myr",
        "last_dynamic_uplift_m", "last_magmatic_productivity"),
    "plume_dynamic_topography_state": ("target_dynamic_topography_m",
        "realized_dynamic_topography_m", "cumulative_positive_support_m_myr"),
    "plume_magmatism_state": ("extrusive_volume_km3", "dyke_volume_km3",
        "underplate_volume_km3", "track_age_myr", "last_emplacement_productivity"),
    "hotspot_track_state": ("thermal_anomaly", "underplate_mean_age_myr",
        "underplate_eclogite_fraction", "last_dike_localization", "last_head_productivity",
        "last_tail_productivity"),
}
VOLUME_FIELDS = {
    "state": {"continental_volume_km3", "oceanic_volume_km3", "sediment_volume_km3"},
    "plume_magmatism_state": {"extrusive_volume_km3", "dyke_volume_km3", "underplate_volume_km3"},
}
PLUME_POPULATION_FIELDS = {
    "centers_unit", "ages_myr", "lifetimes_myr", "head_radii_km", "peak_fluxes", "plume_ids",
    "source_drift_axes_unit", "source_drift_speeds_km_per_myr", "source_drift_segment_index",
    "cumulative_source_distance_km", "cumulative_source_bend_deg", "source_flow_omega_rad_per_myr",
    "last_effective_source_axes_unit", "last_effective_source_speeds_km_per_myr",
}
STARTER_FIELDS = ("damage", "cooling_stress_pa", "water_access", "yield_ratio",
                  "strength_pa", "eligible", "split_band")
FRACTURE_FIELDS = ("damage", "cooling_stress_pa", "water_access", "yield_ratio",
                   "strength_pa", "eligible", "consumed_band")


@dataclass
class RemeshedContinuation:
    model: StarterModel
    starter_state: object
    fracture: YoungShellFracture
    checkpoint: object
    config: dict
    report: dict


def _system(source, owner, mesh, target, ancestors, factor):
    if not np.array_equal(np.unique(owner), np.arange(len(source.plates))):
        raise ValueError("Refinement requires compact plate IDs with nonempty ownership")
    plates = []
    for pid, plate in enumerate(source.plates):
        if plate.plate_id != pid or not 0 <= plate.seed_cell < len(owner) or owner[plate.seed_cell] != pid:
            raise ValueError("Plate seed must belong to its saved plate")
        if len(connected_components(np.flatnonzero(owner == pid), mesh.neighbors)) != 1:
            raise ValueError("Refinement requires connected source plates")
        first = plate.seed_cell * factor
        # The seed is metadata, not a new partition. Select a descendant of the
        # same source seed, closest to its old spherical position.
        offsets = target.centroids[first:first + factor] @ mesh.centroids[plate.seed_cell]
        seed = first + int(np.argmax(offsets))
        plates.append(Plate(pid, seed, np.asarray(plate.euler_axis).copy(), plate.angular_speed_rad_per_myr))
    return PlateSystem(np.asarray(owner)[ancestors].copy(), tuple(plates))


def _cell_array(value, ancestors, count, label):
    array = np.asarray(value)
    if array.ndim < 1 or array.shape[0] != count or array.dtype.kind not in "bifu":
        raise ValueError(f"Invalid remesh cell field {label}")
    if not np.isfinite(array).all():
        raise ValueError(f"Nonfinite remesh cell field {label}")
    return array[ancestors].copy()


def _volume(value, weights, factor, count, label):
    array = np.asarray(value)
    if array.shape != (count,) or not np.isfinite(array).all() or np.any(array < 0):
        raise ValueError(f"Invalid extensive volume {label}")
    result = np.asarray(array, dtype=np.float64)[:, None] * weights.reshape(count, factor)
    # Correct only floating-point roundoff per parent; this is not a global
    # rescaling that could transfer material between unrelated source cells.
    largest = np.argmax(weights.reshape(count, factor), axis=1)
    result[np.arange(count), largest] += array - result.sum(axis=1)
    if np.any(result < 0):
        raise ValueError(f"Refinement produced a negative volume in {label}")
    return result.ravel()


def _refine_record(name, source, ancestors, weights, factor, count, conservation):
    if source is None:
        return None
    result = deepcopy(source)
    for entry in fields(source):
        value = getattr(source, entry.name)
        label = f"{name}.{entry.name}"
        if value is None:
            continue
        if entry.name in CELL_FIELDS[name]:
            if entry.name in VOLUME_FIELDS.get(name, ()):
                mapped = _volume(value, weights, factor, count, label)
                grouped = mapped.reshape(count, factor).sum(axis=1)
                scale = max(float(np.abs(value).sum()), 1.)
                conservation[label] = {
                    "source_total_km3": float(np.sum(value)), "target_total_km3": float(mapped.sum()),
                    "relative_total_residual": float((mapped.sum() - np.sum(value)) / scale),
                    "max_parent_residual_km3": float(np.max(np.abs(grouped - value))),
                }
            else:
                mapped = _cell_array(value, ancestors, count, label)
            setattr(result, entry.name, mapped)
        elif isinstance(value, np.ndarray):
            if name != "plume_state" or entry.name not in PLUME_POPULATION_FIELDS:
                raise ValueError(f"No explicit remesh rule for array {label}")
    return result


def _contact_footprints(manager, old_mesh, new_mesh, old_owner, new_owner, ancestors):
    """Keep remembered seam footprints on descendants of actual source edges."""
    new_edges = {}
    for a, b, _, _ in new_mesh.shared_edges:
        ca, cb = int(ancestors[a]), int(ancestors[b])
        if ca == cb or new_owner[a] == new_owner[b]:
            continue
        key = tuple(sorted((ca, cb)))
        new_edges.setdefault(key, set()).update((a, b))
    footprints = {}
    for pair, saved in manager.collision_contact_faces.items():
        saved = set(map(int, saved))
        if any(face < 0 or face >= len(old_owner) for face in saved):
            raise ValueError("Collision footprint references an unknown source cell")
        result = set()
        matched = set()
        for a, b, _, _ in old_mesh.shared_edges:
            if a in saved and b in saved and tuple(sorted((int(old_owner[a]), int(old_owner[b])))) == pair:
                result.update(new_edges.get((a, b), ()))
                matched.update((a, b))
        # Refuse ambiguous/stale footprints instead of silently erasing the
        # contact's spatial history or spreading it over a whole coarse cell.
        if matched != saved:
            raise ValueError("Collision footprint is not a complete current plate-boundary footprint")
        footprints[pair] = tuple(sorted(result))
    return footprints


def remesh_continuation_state(model, starter_state, fracture, checkpoint, config, target_subdivisions):
    """Return independent paired state on the same or finer canonical icosphere.

    Input objects are never modified. Save the returned model.configuration in
    young parameters metadata and save the returned fracture with that model;
    this supplies their new, consistent grid/configuration fingerprints.
    Histories keep their historical source-grid counts and are not rewritten.
    """
    if isinstance(target_subdivisions, bool) or not isinstance(target_subdivisions, Integral):
        raise ValueError("Target subdivisions must be an integer")
    if not 1 <= target_subdivisions <= 8:
        raise ValueError("Target subdivisions must be in 1..8")
    source_subdivisions = model.shell.subdivisions
    if target_subdivisions < source_subdivisions:
        raise ValueError("Coarsening is unsupported: it can erase plate ownership and fracture memory; choose the same or finer mesh")
    model._validate(starter_state)
    fracture._validate()
    if fracture.model.fingerprint != model.fingerprint:
        raise ValueError("Fracture memory belongs to another source model")
    if (abs(fracture.time_myr - starter_state.time_myr) > 1e-10
            or abs(checkpoint.state.time_myr - starter_state.time_myr) > 1e-10):
        raise ValueError("Paired continuation clocks disagree")
    if not np.array_equal(fracture.damage, checkpoint.state.tidal_damage):
        raise ValueError("Fracture memory and mature material damage disagree")
    mesh = model.mesh
    canonical = build_icosphere(source_subdivisions)
    if not (np.array_equal(mesh.vertices, canonical.vertices) and np.array_equal(mesh.faces, canonical.faces)):
        raise ValueError("Hierarchical refinement requires the canonical unrotated icosphere ordering")
    if config["mesh"]["subdivisions"] != source_subdivisions:
        raise ValueError("Saved mature configuration and paired source mesh disagree")
    if not np.array_equal(checkpoint.state.cell_plate, checkpoint.system.cell_plate):
        raise ValueError("Mature state and plate ownership disagree")
    target = canonical if target_subdivisions == source_subdivisions else build_icosphere(target_subdivisions)
    factor = 4 ** (target_subdivisions - source_subdivisions)
    count = mesh.cell_count
    ancestors = np.arange(target.cell_count, dtype=np.int64) // factor
    grouped_area = target.areas_unit_sphere.reshape(count, factor).sum(axis=1)
    if not np.allclose(grouped_area, mesh.areas_unit_sphere, rtol=2e-13, atol=1e-15):
        raise ValueError("Target children do not cover their spherical source parents")
    weights = target.areas_unit_sphere / grouped_area[ancestors]
    new_model = StarterModel(target, model.thermal, model.tides,
                             replace(model.shell, subdivisions=int(target_subdivisions)), model.parameters)
    young = deepcopy(starter_state)
    for name in STARTER_FIELDS:
        setattr(young, name, _cell_array(getattr(starter_state, name), ancestors, count, f"starter.{name}"))
    young.system = _system(starter_state.system, starter_state.system.cell_plate, mesh, target, ancestors, factor)
    new_model._validate(young)
    live = deepcopy(fracture)
    live.model = new_model
    for name in FRACTURE_FIELDS:
        setattr(live.memory, name, _cell_array(getattr(fracture.memory, name), ancestors, count, f"fracture.{name}"))
    live._validate()
    cp = deepcopy(checkpoint)
    conservation = {}
    for name in CELL_FIELDS:
        setattr(cp, name, _refine_record(name, getattr(checkpoint, name), ancestors, weights, factor, count, conservation))
    cp.system = _system(checkpoint.system, checkpoint.system.cell_plate, mesh, target, ancestors, factor)
    cp.baseline = _system(checkpoint.baseline, checkpoint.baseline.cell_plate, mesh, target, ancestors, factor)
    cp.manager.collision_contact_faces = _contact_footprints(checkpoint.manager, mesh, target,
        checkpoint.system.cell_plate, cp.system.cell_plate, ancestors)
    cfg = deepcopy(config)
    cfg["mesh"]["subdivisions"] = int(target_subdivisions)
    cfg["plates"]["count"] = len(cp.system.plates)
    if any(abs(row["relative_total_residual"]) > 2e-13 for row in conservation.values()):
        raise ValueError("Conservative refinement failed its volume ledger")
    report = {
        "format": "genesis-continuation-refinement-0.1", "time_myr": starter_state.time_myr,
        "source_subdivisions": int(source_subdivisions), "target_subdivisions": int(target_subdivisions),
        "source_cell_count": count, "target_cell_count": target.cell_count,
        "children_per_source_cell": factor, "source_fingerprint": model.fingerprint,
        "target_fingerprint": new_model.fingerprint, "volume_conservation": conservation,
        "method": "exact icosphere ancestry; cell totals split by spherical child area; cell averages inherited",
        "cooling_stress": "scalar isotropic in-plane stress; no basis-dependent persisted tensor components",
        "spatial_forcing": "same seeded smooth forcing re-evaluated on target mesh; discretization and quadrature may change",
        "historical_rows": "unchanged historical source-grid counts, events and budgets; not new-grid observations",
        "limitations": ["Refinement does not add resolved initial physical detail or prove spatial convergence.",
                        "Coarsening is unsupported because the solver stores one plate owner per cell."],
    }
    return RemeshedContinuation(new_model, young, live, cp, cfg, report)


__all__ = ["RemeshedContinuation", "remesh_continuation_state"]
