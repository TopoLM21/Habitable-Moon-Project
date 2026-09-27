"""Experimental, explicit-state import into the existing v0.31 plate model.

This is a representation adapter, not a physical handoff criterion.  In
particular, a split contact mesh and a cold mechanical lid are not respectively
a mature raster and differentiated chemical crust.  Callers must resolve those
questions before supplying the fields below.  No plate/continent generation,
ocean-age seeding, mantle-flow reconstruction, or equilibrium relief is used.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .checkpoint import RunCheckpoint, load_checkpoint, save_checkpoint
from .continental import ContinentalCycleState
from .hotspot_tracks import initialize_hotspot_tracks
from .hydrosphere import HydrosphereState
from .lithosphere import CrustType, LithosphereState
from .mantle import MantleFlowState
from .mesh import SphereMesh, build_icosphere, connected_components
from .plates import PlateSystem
from .plume_dynamic_topography import initialize_plume_dynamic_topography
from .plume_flow_coupling import PlumeFlowCouplingParameters, diagnose_plume_flow_coupling
from .plume_magmatism import initialize_plume_magmatism
from .plume_rifting import initialize_plume_rifting
from .plumes import MantlePlumeState
from .sediment import initialize_sediment_budget
from .subduction_memory import initialize_subduction_memory
from .thermal import ThermalState
from .topography import TopographyState
from .topology import PlateTopologyManager, PlateTopologyParameters
from .transport import initialize_transport_state


@dataclass(slots=True)
class GenesisMatureImport:
    checkpoint: RunCheckpoint
    reservoir_arrays: dict[str, np.ndarray]
    report: dict[str, Any]


def _scalar(name: str, value: float, *, minimum: float = 0.0) -> float:
    result = float(value)
    if not np.isfinite(result) or result < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return result


def _field(name: str, value, shape: tuple[int, ...], *, minimum=None, maximum=None):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite array of shape {shape}")
    if minimum is not None and np.any(result < minimum):
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and np.any(result > maximum):
        raise ValueError(f"{name} must be <= {maximum}")
    return result.copy()


def _canonical_subdivisions(mesh: SphereMesh) -> int:
    # The current runner reconstructs its mesh from config, not the checkpoint.
    count, subdivisions = mesh.cell_count, 0
    while count > 20 and count % 4 == 0:
        count //= 4
        subdivisions += 1
    if count != 20:
        raise ValueError("mature resume requires the canonical icosphere mesh")
    canonical = build_icosphere(subdivisions)
    fields = ("vertices", "faces", "centroids", "areas_unit_sphere")
    if any(not np.array_equal(getattr(mesh, name), getattr(canonical, name)) for name in fields):
        raise ValueError("mature resume requires unchanged canonical icosphere geometry and face order")
    if mesh.neighbors != canonical.neighbors or mesh.shared_edges != canonical.shared_edges:
        raise ValueError("mature resume requires closed canonical mesh connectivity")
    return subdivisions


def _validated_system(mesh: SphereMesh, system: PlateSystem) -> PlateSystem:
    owner = np.asarray(system.cell_plate)
    count = len(system.plates)
    if owner.shape != (mesh.cell_count,) or owner.dtype.kind not in "iu" or count == 0:
        raise ValueError("plate ownership must be an integer field matching the mesh")
    if not np.array_equal(np.unique(owner), np.arange(count)):
        raise ValueError("plate labels must be compact and every plate must own cells")
    for index, plate in enumerate(system.plates):
        if plate.plate_id != index:
            raise ValueError("plate tuple order must match compact plate IDs")
        cells = np.flatnonzero(owner == index)
        if len(connected_components(cells, mesh.neighbors)) != 1:
            raise ValueError("each imported plate must be connected")
        if plate.seed_cell not in cells:
            raise ValueError("each plate seed must belong to its plate")
        axis = _field("Euler axis", plate.euler_axis, (3,))
        if not np.isclose(np.linalg.norm(axis), 1.0, rtol=0.0, atol=1e-12):
            raise ValueError("Euler axes must be unit vectors")
        if not np.isfinite(plate.angular_speed_rad_per_myr):
            raise ValueError("plate angular speeds must be finite")
    result = deepcopy(system)
    result.cell_plate = owner.astype(np.int32, copy=True)
    return result


def build_experimental_genesis_mature_import(
    mesh: SphereMesh,
    *,
    radius_km: float,
    system: PlateSystem,
    crust_age_myr: np.ndarray,
    crust_thickness_km: np.ndarray,
    tidal_damage: np.ndarray,
    mantle_lithosphere_thickness_km: np.ndarray,
    mantle_lithosphere_density_anomaly_kg_m3: np.ndarray,
    thermal: ThermalState,
    elevation_m: np.ndarray,
    water_volume_km3: float,
    mantle_cell_omega_rad_per_myr: np.ndarray,
    source_mass_kg: np.ndarray,
    source_enthalpy_j: np.ndarray,
    next_plume_birth_time_myr: float,
    source_metadata: dict[str, Any] | None = None,
    topology_parameters: PlateTopologyParameters | None = None,
    continental_fraction: np.ndarray | None = None,
    continental_volume_km3: np.ndarray | None = None,
) -> GenesisMatureImport:
    """Build a v0.31-compatible checkpoint without manufacturing initial geology.

    Required preconditions belong to the caller: a coupled, physically mature
    state; verified rigid plates; conservative remapping onto the EXACT canonical
    icosphere; differentiated crust (not total cold lid); an explicit independent
    mantle flow; and a physically justified scalar ThermalState. Its ``time_myr``
    supplies the absolute clock for every subsystem. Surface water here is ONLY
    liquid ocean volume; vapour/ice cannot silently be counted as liquid water.

    The mature configuration must use the reported mesh subdivision and radius,
    compatible thermal/orbital parameters and times, and a sufficiently short
    first step. This constructor cannot certify those physical conditions.

    Source mass and enthalpy are per-cell extensive arrays after remapping. They
    are retained exactly in a separate archive because the mature solver has no
    equivalent full-column reservoirs. That archive is NOT evolved, and its
    preservation is NOT a claim of coupled mass/energy conservation. Source
    parameters, clocks and extra reservoir totals should be recorded in metadata.

    ``crust_thickness_km`` is the area-mean TOTAL chemical crust thickness. For
    mixed cells the explicit continental volume is subtracted from that total
    to obtain basalt volume. The mature legacy thickness field is converted to
    the visible endmember thickness; both material volumes remain independent.
    Continental fields must be supplied together or both omitted (zero). New
    geological memories start at zero, including subduction, arcs, sediments and
    cratons. There are no inherited plumes; the first future birth time must be
    explicit. Existing genesis contact snapshots are not sufficient inputs.
    """
    subdivisions = _canonical_subdivisions(mesh)
    radius = _scalar("radius_km", radius_km, minimum=np.finfo(float).tiny)
    n = mesh.cell_count
    areas = mesh.physical_cell_areas_km2(radius)
    system = _validated_system(mesh, system)
    for name, value in asdict(thermal).items():
        _scalar(f"thermal.{name}", value)
    time = float(thermal.time_myr)
    if thermal.system_age_myr < time or thermal.mantle_temperature_k <= 0.0:
        raise ValueError("thermal system age must cover simulation time and temperature must be positive")
    if thermal.reference_convective_flux_w_m2 <= 0.0 or thermal.thermal_lithosphere_thickness_km <= 0.0:
        raise ValueError("thermal reference flux and thermal thickness must be positive")
    birth_time = _scalar("next_plume_birth_time_myr", next_plume_birth_time_myr, minimum=time)
    age = _field("crust_age_myr", crust_age_myr, (n,), minimum=0.0, maximum=thermal.system_age_myr)
    thickness = _field("crust_thickness_km", crust_thickness_km, (n,), minimum=0.0)
    if np.any(thickness <= 0.0):
        raise ValueError("every mature cell must have positive chemical crust thickness")
    damage = _field("tidal_damage", tidal_damage, (n,), minimum=0.0, maximum=1.0)
    mantle_thickness = _field("mantle_lithosphere_thickness_km", mantle_lithosphere_thickness_km, (n,), minimum=0.0)
    mantle_density = _field("mantle_lithosphere_density_anomaly_kg_m3", mantle_lithosphere_density_anomaly_kg_m3, (n,))
    elevation = _field("elevation_m", elevation_m, (n,))
    if np.any(radius + elevation / 1000.0 <= 0.0):
        raise ValueError("surface elevation reaches nonpositive radius")
    water = _scalar("water_volume_km3", water_volume_km3)
    omega = _field("mantle_cell_omega_rad_per_myr", mantle_cell_omega_rad_per_myr, (n, 3))
    mass = _field("source_mass_kg", source_mass_kg, (n,), minimum=0.0)
    enthalpy = _field("source_enthalpy_j", source_enthalpy_j, (n,))
    if (continental_fraction is None) != (continental_volume_km3 is None):
        raise ValueError("continental fraction and volume must be supplied together")
    fraction = np.zeros(n) if continental_fraction is None else _field("continental_fraction", continental_fraction, (n,), minimum=0.0, maximum=1.0)
    volume = np.zeros(n) if continental_volume_km3 is None else _field("continental_volume_km3", continental_volume_km3, (n,), minimum=0.0)
    if np.any((fraction == 0.0) != (volume == 0.0)):
        raise ValueError("continental footprint and volume must be consistently present")
    if np.any(volume > areas * thickness * (1.0 + 1e-12)):
        raise ValueError("continental volume cannot exceed total chemical crust volume")
    oceanic_volume = areas * thickness - volume
    # A tiny subtraction residual is harmless only on a fully continental cell.
    full_continent = fraction == 1.0
    if np.any(full_continent & (np.abs(oceanic_volume) > areas * thickness * 1e-12)):
        raise ValueError("oceanic volume requires a nonzero oceanic footprint")
    oceanic_volume[full_continent] = 0.0
    if np.any(oceanic_volume < 0.0):
        raise ValueError("oceanic volume cannot be negative")
    if np.any((~full_continent) & (oceanic_volume == 0.0)):
        raise ValueError("oceanic footprint requires positive chemical crust volume")
    visible_continent = fraction >= 0.5
    visible_thickness = np.empty(n)
    visible_thickness[visible_continent] = volume[visible_continent] / (areas[visible_continent] * fraction[visible_continent])
    visible_thickness[~visible_continent] = oceanic_volume[~visible_continent] / (areas[~visible_continent] * (1.0 - fraction[~visible_continent]))
    visible_thickness[fraction == 0.0] = thickness[fraction == 0.0]
    metadata = json.loads(json.dumps(source_metadata or {}, allow_nan=False))
    state = LithosphereState(
        time_myr=time, cell_plate=system.cell_plate.copy(),
        crust_type=np.where(fraction >= 0.5, CrustType.CONTINENTAL, CrustType.OCEANIC).astype(np.int8),
        crust_age_myr=age, crust_thickness_km=visible_thickness, tidal_damage=damage,
        rift_extension=np.zeros(n), extension_age_myr=np.zeros(n),
        collision_seam_weakness=np.zeros(n), intraplate_stress=np.zeros(n),
        supercontinent_heat=np.zeros(n), continental_fraction=fraction,
        continental_volume_km3=volume, mantle_lithosphere_thickness_km=mantle_thickness,
        mantle_lithosphere_density_anomaly_kg_m3=mantle_density,
        sediment_volume_km3=np.zeros(n), continental_lithosphere_age_myr=np.zeros(n),
        mantle_depletion_fraction=np.zeros(n), craton_strength=np.zeros(n),
        oceanic_volume_km3=oceanic_volume,
    )
    # Explicitly empty population prevents v0.25+ resume wrappers from seeding it.
    plume = MantlePlumeState(
        time_myr=time, centers_unit=np.empty((0, 3)), ages_myr=np.empty(0),
        lifetimes_myr=np.empty(0), head_radii_km=np.empty(0), peak_fluxes=np.empty(0),
        next_plume_id=0, next_birth_time_myr=birth_time,
        last_flux=np.zeros(n), cumulative_exposure_myr=np.zeros(n),
        cumulative_root_erosion_km=np.zeros(n), last_head_flux=np.zeros(n),
        last_tail_flux=np.zeros(n), plume_ids=np.empty(0, dtype=np.int64),
        source_drift_axes_unit=np.empty((0, 3)), source_drift_speeds_km_per_myr=np.empty(0),
        source_drift_segment_index=np.empty(0, dtype=np.int32),
        cumulative_source_distance_km=np.empty(0), cumulative_source_bend_deg=np.empty(0),
        source_flow_omega_rad_per_myr=np.empty((0, 3)), last_effective_source_axes_unit=np.empty((0, 3)),
        last_effective_source_speeds_km_per_myr=np.empty(0),
    )
    checkpoint = RunCheckpoint(
        state=state, cycle=ContinentalCycleState(time, np.zeros(n)),
        thermal=deepcopy(thermal), topo=TopographyState(time, elevation),
        system=system, baseline=deepcopy(system),
        manager=PlateTopologyManager(deepcopy(topology_parameters or PlateTopologyParameters())),
        initial_continental_area_fraction=float(np.sum(areas * fraction) / np.sum(areas)),
        initial_continental_volume_km3=float(np.sum(volume)),
        topology_rows=[], lithosphere_rows=[], relief_rows=[], cycle_rows=[], thermal_rows=[],
        events=[{"kind": "experimental_genesis_import", "time_myr": time,
                 "radius_km": radius, "mesh_subdivisions": subdivisions}],
        mantle_flow=MantleFlowState(time, omega, float(np.sqrt(np.mean(np.sum(omega * omega, axis=1))))),
        transport_state=initialize_transport_state(len(system.plates)),
        hydrosphere=HydrosphereState(time, water),
        subduction_memory=initialize_subduction_memory(time),
        sediment_budget=initialize_sediment_budget(time),
        plume_state=plume,
        plume_rifting_state=initialize_plume_rifting(mesh, time),
        plume_dynamic_topography_state=initialize_plume_dynamic_topography(mesh, time),
        plume_magmatism_state=initialize_plume_magmatism(mesh, time),
        hotspot_track_state=initialize_hotspot_tracks(mesh, time),
        plume_flow_coupling_rows=[asdict(diagnose_plume_flow_coupling(plume, radius, PlumeFlowCouplingParameters()))],
    )
    report = {
        "format": "experimental_genesis_mature_import", "version": 1,
        "physical_handoff_certified": False,
        "mesh_subdivisions": subdivisions, "cell_count": n, "radius_km": radius,
        "time_myr": time, "plate_count": len(system.plates),
        "water_volume_km3": water,
        "chemical_crust_volume_km3": float(np.sum(areas * thickness)),
        "continental_volume_km3": float(np.sum(volume)),
        "oceanic_volume_km3": float(np.sum(oceanic_volume)),
        "archived_source_mass_kg": float(np.sum(mass)),
        "archived_source_enthalpy_j": float(np.sum(enthalpy)),
        "source_metadata": metadata,
        "unsupported_state": [
            "Source column mass and enthalpy are archived, not evolved by the mature solver.",
            "Contact opening, stress, cohesive/frictional memory and deformation are not represented.",
            "A scalar thermal state cannot retain the depth-resolved enthalpy and melt profile.",
            "Steam and ice are not represented by the mature liquid-water inventory.",
            "Young-moon thermal, orbital and mechanical applicability needs independent validation.",
        ],
        "new_zero_memories": ["subduction", "felsic_potential", "sediments", "cratons", "plumes", "plate_transport"],
        "topology_parameters": asdict(checkpoint.manager.params),
    }
    return GenesisMatureImport(checkpoint, {"source_mass_kg": mass, "source_enthalpy_j": enthalpy}, report)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_experimental_genesis_mature_import(path: str | Path, bundle: GenesisMatureImport) -> Path:
    """Write a new checkpoint plus an untranslated-reservoir archive; never overwrite."""
    root = Path(path)
    root.mkdir(parents=True, exist_ok=False)
    save_checkpoint(root / "mature_checkpoint", bundle.checkpoint)
    np.savez_compressed(root / "source_reservoirs.npz", **bundle.reservoir_arrays)
    manifest = deepcopy(bundle.report)
    manifest["sha256"] = {
        name: _sha256(root / name)
        for name in ("mature_checkpoint/meta.json", "mature_checkpoint/state.npz", "source_reservoirs.npz")
    }
    (root / "handoff.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return root


def load_experimental_genesis_mature_import(path: str | Path) -> GenesisMatureImport:
    """Verify the complete archive before loading the mature checkpoint."""
    root = Path(path)
    report = json.loads((root / "handoff.json").read_text(encoding="utf-8"))
    if report.get("format") != "experimental_genesis_mature_import" or report.get("version") != 1:
        raise ValueError("unsupported experimental genesis import")
    for name in ("mature_checkpoint/meta.json", "mature_checkpoint/state.npz", "source_reservoirs.npz"):
        if report.get("sha256", {}).get(name) != _sha256(root / name):
            raise ValueError(f"genesis import checksum mismatch: {name}")
    params = PlateTopologyParameters(**report["topology_parameters"])
    cp = load_checkpoint(root / "mature_checkpoint", PlateTopologyManager(params))
    mesh = build_icosphere(int(report["mesh_subdivisions"]))
    with np.load(root / "source_reservoirs.npz", allow_pickle=False) as values:
        mass = _field("source_mass_kg", values["source_mass_kg"], (mesh.cell_count,), minimum=0.0)
        enthalpy = _field("source_enthalpy_j", values["source_enthalpy_j"], (mesh.cell_count,))
    checks = {
        "radius_km": cp.events[0]["radius_km"],
        "mesh_subdivisions": cp.events[0]["mesh_subdivisions"],
        "cell_count": mesh.cell_count, "time_myr": cp.state.time_myr,
        "plate_count": len(cp.system.plates), "water_volume_km3": cp.hydrosphere.water_volume_km3,
        "archived_source_mass_kg": float(np.sum(mass)),
        "archived_source_enthalpy_j": float(np.sum(enthalpy)),
        "continental_volume_km3": float(np.sum(cp.state.continental_volume_km3)),
        "oceanic_volume_km3": float(np.sum(cp.state.oceanic_volume_km3)),
        "chemical_crust_volume_km3": float(np.sum(cp.state.oceanic_volume_km3 + cp.state.continental_volume_km3)),
    }
    for name, value in checks.items():
        # Endmember subtraction/recombination can differ by one rounding unit.
        same = np.isclose(report.get(name, np.nan), value, rtol=2e-15, atol=0.0)
        if not same:
            raise ValueError(f"genesis import manifest mismatch: {name}")
    return GenesisMatureImport(cp, {"source_mass_kg": mass, "source_enthalpy_j": enthalpy}, report)
