"""Checkpoint serialization for long v0.9.3 tectonic integrations.

The format intentionally avoids pickle:
- dense numerical fields live in ``state.npz``;
- small structured state and diagnostic history live in ``meta.json``.

A checkpoint contains enough information to continue a run deterministically:
lithosphere, topography, continental-cycle memory, thermal reservoir, dynamic
plate system, post-topology baseline plate system, topology-manager memory,
and accumulated diagnostic/event histories.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np

from .continental import ContinentalCycleState
from .lithosphere import LithosphereState
from .hydrosphere import HydrosphereState
from .mantle import MantleFlowState
from .transport import PlateTransportState
from .plates import Plate, PlateSystem
from .thermal import ThermalState
from .topography import TopographyState
from .topology import PlateTopologyManager
from .subduction_memory import SubductionMemoryState, memory_to_json, memory_from_json
from .sediment import SedimentBudgetState
from .plumes import MantlePlumeState
from .plume_rifting import PlumeRiftingState
from .plume_dynamic_topography import PlumeDynamicTopographyState


@dataclass(slots=True)
class RunCheckpoint:
    state: LithosphereState
    cycle: ContinentalCycleState
    thermal: ThermalState
    topo: TopographyState
    system: PlateSystem
    baseline: PlateSystem
    manager: PlateTopologyManager
    initial_continental_area_fraction: float
    initial_continental_volume_km3: float
    topology_rows: list[dict[str, Any]]
    lithosphere_rows: list[dict[str, Any]]
    relief_rows: list[dict[str, Any]]
    cycle_rows: list[dict[str, Any]]
    thermal_rows: list[dict[str, Any]]
    events: list[dict[str, Any]]
    late_rows: list[dict[str, Any]] = field(default_factory=list)
    mantle_flow: MantleFlowState | None = None
    transport_state: PlateTransportState | None = None
    hydrosphere: HydrosphereState | None = None
    hydrosphere_rows: list[dict[str, Any]] = field(default_factory=list)
    subduction_memory: SubductionMemoryState | None = None
    subduction_memory_rows: list[dict[str, Any]] = field(default_factory=list)
    rollback_rows: list[dict[str, Any]] = field(default_factory=list)
    breakoff_rows: list[dict[str, Any]] = field(default_factory=list)
    arc_rows: list[dict[str, Any]] = field(default_factory=list)
    sediment_budget: SedimentBudgetState | None = None
    sediment_rows: list[dict[str, Any]] = field(default_factory=list)
    craton_rows: list[dict[str, Any]] = field(default_factory=list)
    plume_state: MantlePlumeState | None = None
    plume_rows: list[dict[str, Any]] = field(default_factory=list)
    plume_rifting_state: PlumeRiftingState | None = None
    plume_rifting_rows: list[dict[str, Any]] = field(default_factory=list)
    plume_dynamic_topography_state: PlumeDynamicTopographyState | None = None
    plume_dynamic_topography_rows: list[dict[str, Any]] = field(default_factory=list)


def _plate_dict(p: Plate) -> dict[str, Any]:
    return {
        "plate_id": int(p.plate_id),
        "seed_cell": int(p.seed_cell),
        "euler_axis": [float(x) for x in np.asarray(p.euler_axis, dtype=float)],
        "angular_speed_rad_per_myr": float(p.angular_speed_rad_per_myr),
    }


def _plates_from_dict(items: list[dict[str, Any]]) -> tuple[Plate, ...]:
    return tuple(
        Plate(
            plate_id=int(x["plate_id"]),
            seed_cell=int(x["seed_cell"]),
            euler_axis=np.asarray(x["euler_axis"], dtype=np.float64),
            angular_speed_rad_per_myr=float(x["angular_speed_rad_per_myr"]),
        )
        for x in items
    )


def _jsonable_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out=[]
    for row in rows:
        clean={}
        for k,v in row.items():
            if isinstance(v, np.ndarray):
                continue
            if isinstance(v, (np.integer,)): v=int(v)
            elif isinstance(v, (np.floating,)): v=float(v)
            elif isinstance(v, (np.bool_,)): v=bool(v)
            clean[str(k)]=v
        out.append(clean)
    return out


def save_checkpoint(path: str | Path, cp: RunCheckpoint) -> Path:
    root=Path(path)
    root.mkdir(parents=True, exist_ok=True)
    arrays = dict(
        state_cell_plate=np.asarray(cp.state.cell_plate, dtype=np.int32),
        crust_type=np.asarray(cp.state.crust_type, dtype=np.int8),
        crust_age_myr=np.asarray(cp.state.crust_age_myr, dtype=np.float64),
        crust_thickness_km=np.asarray(cp.state.crust_thickness_km, dtype=np.float64),
        tidal_damage=np.asarray(cp.state.tidal_damage, dtype=np.float64),
        rift_extension=np.zeros_like(cp.state.crust_thickness_km, dtype=np.float64) if cp.state.rift_extension is None else np.asarray(cp.state.rift_extension, dtype=np.float64),
        extension_age_myr=np.zeros_like(cp.state.crust_thickness_km, dtype=np.float64) if cp.state.extension_age_myr is None else np.asarray(cp.state.extension_age_myr, dtype=np.float64),
        collision_seam_weakness=np.zeros_like(cp.state.crust_thickness_km, dtype=np.float64) if cp.state.collision_seam_weakness is None else np.asarray(cp.state.collision_seam_weakness, dtype=np.float64),
        intraplate_stress=np.zeros_like(cp.state.crust_thickness_km, dtype=np.float64) if cp.state.intraplate_stress is None else np.asarray(cp.state.intraplate_stress, dtype=np.float64),
        supercontinent_heat=np.zeros_like(cp.state.crust_thickness_km, dtype=np.float64) if cp.state.supercontinent_heat is None else np.asarray(cp.state.supercontinent_heat, dtype=np.float64),
        felsic_potential=np.asarray(cp.cycle.felsic_potential, dtype=np.float64),
        elevation_m=np.asarray(cp.topo.elevation_m, dtype=np.float64),
        system_cell_plate=np.asarray(cp.system.cell_plate, dtype=np.int32),
        baseline_cell_plate=np.asarray(cp.baseline.cell_plate, dtype=np.int32),
    )
    if cp.state.continental_fraction is not None:
        arrays["continental_fraction"] = np.asarray(cp.state.continental_fraction, dtype=np.float64)
    if cp.state.continental_volume_km3 is not None:
        arrays["continental_volume_km3"] = np.asarray(cp.state.continental_volume_km3, dtype=np.float64)
    if cp.state.mantle_lithosphere_thickness_km is not None:
        arrays["mantle_lithosphere_thickness_km"] = np.asarray(cp.state.mantle_lithosphere_thickness_km, dtype=np.float64)
    if cp.state.mantle_lithosphere_density_anomaly_kg_m3 is not None:
        arrays["mantle_lithosphere_density_anomaly_kg_m3"] = np.asarray(cp.state.mantle_lithosphere_density_anomaly_kg_m3, dtype=np.float64)
    if cp.state.sediment_volume_km3 is not None:
        arrays["sediment_volume_km3"] = np.asarray(cp.state.sediment_volume_km3, dtype=np.float64)
    if cp.state.continental_lithosphere_age_myr is not None:
        arrays["continental_lithosphere_age_myr"] = np.asarray(cp.state.continental_lithosphere_age_myr, dtype=np.float64)
    if cp.state.mantle_depletion_fraction is not None:
        arrays["mantle_depletion_fraction"] = np.asarray(cp.state.mantle_depletion_fraction, dtype=np.float64)
    if cp.state.craton_strength is not None:
        arrays["craton_strength"] = np.asarray(cp.state.craton_strength, dtype=np.float64)
    if cp.mantle_flow is not None:
        arrays["mantle_cell_omega_rad_per_myr"] = np.asarray(cp.mantle_flow.cell_omega_rad_per_myr, dtype=np.float64)
    if cp.transport_state is not None:
        arrays["transport_residual_quaternions"] = np.asarray(cp.transport_state.residual_quaternions, dtype=np.float64)
        arrays["transport_hold_age_myr"] = np.asarray(cp.transport_state.hold_age_myr, dtype=np.float64)
    if cp.plume_state is not None:
        arrays["plume_centers_unit"] = np.asarray(cp.plume_state.centers_unit, dtype=np.float64)
        arrays["plume_ages_myr"] = np.asarray(cp.plume_state.ages_myr, dtype=np.float64)
        arrays["plume_lifetimes_myr"] = np.asarray(cp.plume_state.lifetimes_myr, dtype=np.float64)
        arrays["plume_head_radii_km"] = np.asarray(cp.plume_state.head_radii_km, dtype=np.float64)
        arrays["plume_peak_fluxes"] = np.asarray(cp.plume_state.peak_fluxes, dtype=np.float64)
        arrays["plume_last_flux"] = np.asarray(cp.plume_state.last_flux, dtype=np.float64)
        arrays["plume_cumulative_exposure_myr"] = np.asarray(cp.plume_state.cumulative_exposure_myr, dtype=np.float64)
        arrays["plume_cumulative_root_erosion_km"] = np.asarray(cp.plume_state.cumulative_root_erosion_km, dtype=np.float64)
    if cp.plume_rifting_state is not None:
        arrays["plume_rifting_last_extension_forcing"] = np.asarray(cp.plume_rifting_state.last_extension_forcing, dtype=np.float64)
        arrays["plume_rifting_cumulative_extension_impulse_myr"] = np.asarray(cp.plume_rifting_state.cumulative_extension_impulse_myr, dtype=np.float64)
        arrays["plume_rifting_last_dynamic_uplift_m"] = np.asarray(cp.plume_rifting_state.last_dynamic_uplift_m, dtype=np.float64)
        arrays["plume_rifting_last_magmatic_productivity"] = np.asarray(cp.plume_rifting_state.last_magmatic_productivity, dtype=np.float64)
    if cp.plume_dynamic_topography_state is not None:
        arrays["plume_dynamic_topography_target_m"] = np.asarray(cp.plume_dynamic_topography_state.target_dynamic_topography_m, dtype=np.float64)
        arrays["plume_dynamic_topography_realized_m"] = np.asarray(cp.plume_dynamic_topography_state.realized_dynamic_topography_m, dtype=np.float64)
        arrays["plume_dynamic_topography_cumulative_positive_support_m_myr"] = np.asarray(cp.plume_dynamic_topography_state.cumulative_positive_support_m_myr, dtype=np.float64)
    np.savez_compressed(root/"state.npz", **arrays)
    collision=[
        [int(a), int(b), float(age)]
        for (a,b),age in sorted(cp.manager.collision_age_myr.items())
    ]
    quiet_weld=[
        [int(a), int(b), float(age)]
        for (a,b),age in sorted(cp.manager.quiet_weld_age_myr.items())
    ]
    meta={
        "format":"moon_tectonics_checkpoint",
        "version":("0.27-plume-dynamic-topography" if cp.plume_dynamic_topography_state is not None else ("0.26-plume-rifting" if cp.plume_rifting_state is not None else ("0.25-mantle-plumes" if cp.plume_state is not None else ("0.24-cratonic-memory" if cp.state.craton_strength is not None else ("0.23-conservative-sediments" if cp.sediment_budget is not None else ("0.22-flexural-isostasy" if (cp.relief_rows and "mean_elastic_thickness_km" in cp.relief_rows[-1]) else ("0.21-slab-geometry-arcs" if cp.arc_rows else ("0.20-slab-breakoff" if cp.breakoff_rows else ("0.19-rollback" if cp.rollback_rows else ("0.18-subduction-memory" if cp.subduction_memory is not None else ("0.16-lithosphere-split" if cp.state.mantle_lithosphere_thickness_km is not None else ("0.14-hydrosphere" if cp.hydrosphere is not None else ("0.11-material" if cp.state.continental_fraction is not None else ("0.10-reconstructed" if (cp.mantle_flow is not None or cp.transport_state is not None) else "0.9.5")))))))))))))),
        "time_myr":float(cp.state.time_myr),
        "lithosphere_time_myr":float(cp.state.time_myr),
        "continental_cycle":{
            "time_myr":float(cp.cycle.time_myr),
            "cumulative_generated_area_km2":float(cp.cycle.cumulative_generated_area_km2),
            "cumulative_recycled_area_km2":float(cp.cycle.cumulative_recycled_area_km2),
            "cumulative_generated_volume_km3":float(cp.cycle.cumulative_generated_volume_km3),
            "cumulative_recycled_volume_km3":float(cp.cycle.cumulative_recycled_volume_km3),
        },
        "thermal":asdict(cp.thermal),
        "topography_time_myr":float(cp.topo.time_myr),
        "system_plates":[_plate_dict(p) for p in cp.system.plates],
        "baseline_plates":[_plate_dict(p) for p in cp.baseline.plates],
        "topology_manager":{
            "collision_age_myr":collision,
            "quiet_weld_age_myr":quiet_weld,
            "small_plate_age_myr":[[int(pid),float(age)] for pid,age in sorted(cp.manager.small_plate_age_myr.items())],
            "last_split_time_myr":float(cp.manager.last_split_time_myr),
        },
        "initial_continental_area_fraction":float(cp.initial_continental_area_fraction),
        "initial_continental_volume_km3":float(cp.initial_continental_volume_km3),
        "topology_rows":_jsonable_rows(cp.topology_rows),
        "lithosphere_rows":_jsonable_rows(cp.lithosphere_rows),
        "relief_rows":_jsonable_rows(cp.relief_rows),
        "cycle_rows":_jsonable_rows(cp.cycle_rows),
        "thermal_rows":_jsonable_rows(cp.thermal_rows),
        "late_rows":_jsonable_rows(cp.late_rows),
        "events":cp.events,
        "hydrosphere": None if cp.hydrosphere is None else {
            "time_myr": float(cp.hydrosphere.time_myr),
            "water_volume_km3": float(cp.hydrosphere.water_volume_km3),
            "reference_sea_level_m": float(cp.hydrosphere.reference_sea_level_m),
        },
        "hydrosphere_rows": _jsonable_rows(cp.hydrosphere_rows),
        "mantle_flow": None if cp.mantle_flow is None else {
            "time_myr": float(cp.mantle_flow.time_myr),
            "formation_rms_rad_per_myr": float(cp.mantle_flow.formation_rms_rad_per_myr),
        },
        "subduction_memory": memory_to_json(cp.subduction_memory),
        "subduction_memory_rows": _jsonable_rows(cp.subduction_memory_rows),
        "rollback_rows": _jsonable_rows(cp.rollback_rows),
        "breakoff_rows": _jsonable_rows(cp.breakoff_rows),
        "arc_rows": _jsonable_rows(cp.arc_rows),
        "sediment_budget": None if cp.sediment_budget is None else asdict(cp.sediment_budget),
        "sediment_rows": _jsonable_rows(cp.sediment_rows),
        "craton_rows": _jsonable_rows(cp.craton_rows),
        "plume_state": None if cp.plume_state is None else {
            "time_myr": float(cp.plume_state.time_myr),
            "next_plume_id": int(cp.plume_state.next_plume_id),
            "next_birth_time_myr": float(cp.plume_state.next_birth_time_myr),
        },
        "plume_rows": _jsonable_rows(cp.plume_rows),
        "plume_rifting_state": None if cp.plume_rifting_state is None else {
            "time_myr": float(cp.plume_rifting_state.time_myr),
        },
        "plume_rifting_rows": _jsonable_rows(cp.plume_rifting_rows),
        "plume_dynamic_topography_state": None if cp.plume_dynamic_topography_state is None else {
            "time_myr": float(cp.plume_dynamic_topography_state.time_myr),
        },
        "plume_dynamic_topography_rows": _jsonable_rows(cp.plume_dynamic_topography_rows),
        "transport_state": None if cp.transport_state is None else {
            "cumulative_commit_count": int(cp.transport_state.cumulative_commit_count),
            "max_hold_age_myr": float(cp.transport_state.max_hold_age_myr),
        },
    }
    with (root/"meta.json").open("w",encoding="utf-8") as h:
        json.dump(meta,h,ensure_ascii=False,indent=2)
    return root


def load_checkpoint(path: str | Path, manager: PlateTopologyManager) -> RunCheckpoint:
    root=Path(path)
    with (root/"meta.json").open("r",encoding="utf-8") as h:
        meta=json.load(h)
    if meta.get("format")!="moon_tectonics_checkpoint":
        raise ValueError("Not a moon tectonics checkpoint")
    if meta.get("version") not in {"0.9.1", "0.9.2", "0.9.3", "0.9.4", "0.9.5", "0.10-reconstructed", "0.11-material", "0.14-hydrosphere", "0.16-lithosphere-split", "0.18-subduction-memory", "0.19-rollback", "0.20-slab-breakoff", "0.21-slab-geometry-arcs", "0.22-flexural-isostasy", "0.23-conservative-sediments", "0.24-cratonic-memory", "0.25-mantle-plumes", "0.26-plume-rifting", "0.27-plume-dynamic-topography"}:
        raise ValueError(f"Unsupported checkpoint version: {meta.get('version')}")
    with np.load(root/"state.npz", allow_pickle=False) as z:
        state=LithosphereState(
            time_myr=float(meta["lithosphere_time_myr"]),
            cell_plate=z["state_cell_plate"].copy(),
            crust_type=z["crust_type"].copy(),
            crust_age_myr=z["crust_age_myr"].copy(),
            crust_thickness_km=z["crust_thickness_km"].copy(),
            tidal_damage=z["tidal_damage"].copy(),
            rift_extension=z["rift_extension"].copy() if "rift_extension" in z.files else np.zeros_like(z["crust_thickness_km"], dtype=np.float64),
            extension_age_myr=z["extension_age_myr"].copy() if "extension_age_myr" in z.files else np.zeros_like(z["crust_thickness_km"], dtype=np.float64),
            collision_seam_weakness=z["collision_seam_weakness"].copy() if "collision_seam_weakness" in z.files else np.zeros_like(z["crust_thickness_km"], dtype=np.float64),
            intraplate_stress=z["intraplate_stress"].copy() if "intraplate_stress" in z.files else np.zeros_like(z["crust_thickness_km"], dtype=np.float64),
            supercontinent_heat=z["supercontinent_heat"].copy() if "supercontinent_heat" in z.files else np.zeros_like(z["crust_thickness_km"], dtype=np.float64),
            continental_fraction=z["continental_fraction"].copy() if "continental_fraction" in z.files else None,
            continental_volume_km3=z["continental_volume_km3"].copy() if "continental_volume_km3" in z.files else None,
            mantle_lithosphere_thickness_km=z["mantle_lithosphere_thickness_km"].copy() if "mantle_lithosphere_thickness_km" in z.files else None,
            mantle_lithosphere_density_anomaly_kg_m3=z["mantle_lithosphere_density_anomaly_kg_m3"].copy() if "mantle_lithosphere_density_anomaly_kg_m3" in z.files else None,
            sediment_volume_km3=z["sediment_volume_km3"].copy() if "sediment_volume_km3" in z.files else np.zeros_like(z["crust_thickness_km"],dtype=np.float64),
            continental_lithosphere_age_myr=z["continental_lithosphere_age_myr"].copy() if "continental_lithosphere_age_myr" in z.files else None,
            mantle_depletion_fraction=z["mantle_depletion_fraction"].copy() if "mantle_depletion_fraction" in z.files else None,
            craton_strength=z["craton_strength"].copy() if "craton_strength" in z.files else None,
        )
        cc=meta["continental_cycle"]
        cycle=ContinentalCycleState(
            time_myr=float(cc["time_myr"]),
            felsic_potential=z["felsic_potential"].copy(),
            cumulative_generated_area_km2=float(cc["cumulative_generated_area_km2"]),
            cumulative_recycled_area_km2=float(cc["cumulative_recycled_area_km2"]),
            cumulative_generated_volume_km3=float(cc["cumulative_generated_volume_km3"]),
            cumulative_recycled_volume_km3=float(cc["cumulative_recycled_volume_km3"]),
        )
        thermal=ThermalState(**{k:float(v) for k,v in meta["thermal"].items()})
        topo=TopographyState(time_myr=float(meta["topography_time_myr"]), elevation_m=z["elevation_m"].copy())
        system=PlateSystem(cell_plate=z["system_cell_plate"].copy(), plates=_plates_from_dict(meta["system_plates"]))
        baseline=PlateSystem(cell_plate=z["baseline_cell_plate"].copy(), plates=_plates_from_dict(meta["baseline_plates"]))
        mantle_flow = None
        mf = meta.get("mantle_flow")
        if mf is not None and "mantle_cell_omega_rad_per_myr" in z.files:
            mantle_flow = MantleFlowState(
                time_myr=float(mf["time_myr"]),
                cell_omega_rad_per_myr=z["mantle_cell_omega_rad_per_myr"].copy(),
                formation_rms_rad_per_myr=float(mf["formation_rms_rad_per_myr"]),
            )
        transport_state = None
        ts = meta.get("transport_state")
        if ts is not None and "transport_residual_quaternions" in z.files:
            transport_state = PlateTransportState(
                residual_quaternions=z["transport_residual_quaternions"].copy(),
                hold_age_myr=z["transport_hold_age_myr"].copy(),
                cumulative_commit_count=int(ts.get("cumulative_commit_count", 0)),
                max_hold_age_myr=float(ts.get("max_hold_age_myr", 0.0)),
            )
        plume_state = None
        ps = meta.get("plume_state")
        if ps is not None and "plume_centers_unit" in z.files:
            plume_state = MantlePlumeState(
                time_myr=float(ps["time_myr"]),
                centers_unit=z["plume_centers_unit"].copy(),
                ages_myr=z["plume_ages_myr"].copy(),
                lifetimes_myr=z["plume_lifetimes_myr"].copy(),
                head_radii_km=z["plume_head_radii_km"].copy(),
                peak_fluxes=z["plume_peak_fluxes"].copy(),
                next_plume_id=int(ps["next_plume_id"]),
                next_birth_time_myr=float(ps["next_birth_time_myr"]),
                last_flux=z["plume_last_flux"].copy(),
                cumulative_exposure_myr=z["plume_cumulative_exposure_myr"].copy(),
                cumulative_root_erosion_km=z["plume_cumulative_root_erosion_km"].copy(),
            )
        plume_rifting_state = None
        prs = meta.get("plume_rifting_state")
        if prs is not None and "plume_rifting_last_extension_forcing" in z.files:
            plume_rifting_state = PlumeRiftingState(
                time_myr=float(prs["time_myr"]),
                last_extension_forcing=z["plume_rifting_last_extension_forcing"].copy(),
                cumulative_extension_impulse_myr=z["plume_rifting_cumulative_extension_impulse_myr"].copy(),
                last_dynamic_uplift_m=z["plume_rifting_last_dynamic_uplift_m"].copy(),
                last_magmatic_productivity=z["plume_rifting_last_magmatic_productivity"].copy(),
            )
        plume_dynamic_topography_state = None
        pdts = meta.get("plume_dynamic_topography_state")
        if pdts is not None and "plume_dynamic_topography_realized_m" in z.files:
            plume_dynamic_topography_state = PlumeDynamicTopographyState(
                time_myr=float(pdts["time_myr"]),
                target_dynamic_topography_m=z["plume_dynamic_topography_target_m"].copy(),
                realized_dynamic_topography_m=z["plume_dynamic_topography_realized_m"].copy(),
                cumulative_positive_support_m_myr=z["plume_dynamic_topography_cumulative_positive_support_m_myr"].copy(),
            )
    manager.collision_age_myr={
        (int(a),int(b)):float(age) for a,b,age in meta["topology_manager"]["collision_age_myr"]
    }
    manager.quiet_weld_age_myr={
        (int(a),int(b)):float(age) for a,b,age in meta["topology_manager"].get("quiet_weld_age_myr", [])
    }
    manager.small_plate_age_myr={
        int(pid):float(age) for pid,age in meta["topology_manager"].get("small_plate_age_myr", [])
    }
    manager.last_split_time_myr=float(meta["topology_manager"]["last_split_time_myr"])
    hydro_meta=meta.get("hydrosphere")
    hydrosphere=None if hydro_meta is None else HydrosphereState(
        time_myr=float(hydro_meta["time_myr"]),
        water_volume_km3=float(hydro_meta["water_volume_km3"]),
        reference_sea_level_m=float(hydro_meta.get("reference_sea_level_m",0.0)),
    )
    sb=meta.get("sediment_budget")
    sediment_budget=None if sb is None else SedimentBudgetState(**{k:float(v) for k,v in sb.items()})
    return RunCheckpoint(
        state=state,cycle=cycle,thermal=thermal,topo=topo,system=system,baseline=baseline,manager=manager,
        initial_continental_area_fraction=float(meta["initial_continental_area_fraction"]),
        initial_continental_volume_km3=float(meta["initial_continental_volume_km3"]),
        topology_rows=list(meta.get("topology_rows",[])), lithosphere_rows=list(meta.get("lithosphere_rows",[])), relief_rows=list(meta.get("relief_rows",[])),
        cycle_rows=list(meta.get("cycle_rows",[])), thermal_rows=list(meta.get("thermal_rows",[])),
        late_rows=list(meta.get("late_rows",[])),
        events=list(meta.get("events",[])),
        mantle_flow=mantle_flow,
        transport_state=transport_state,
        hydrosphere=hydrosphere,
        hydrosphere_rows=list(meta.get("hydrosphere_rows",[])),
        subduction_memory=memory_from_json(meta.get("subduction_memory")),
        subduction_memory_rows=list(meta.get("subduction_memory_rows",[])),
        rollback_rows=list(meta.get("rollback_rows",[])),
        breakoff_rows=list(meta.get("breakoff_rows",[])),
        arc_rows=list(meta.get("arc_rows",[])),
        sediment_budget=sediment_budget,
        sediment_rows=list(meta.get("sediment_rows",[])),
        craton_rows=list(meta.get("craton_rows",[])),
        plume_state=plume_state,
        plume_rows=list(meta.get("plume_rows",[])),
        plume_rifting_state=plume_rifting_state,
        plume_rifting_rows=list(meta.get("plume_rifting_rows",[])),
        plume_dynamic_topography_state=plume_dynamic_topography_state,
        plume_dynamic_topography_rows=list(meta.get("plume_dynamic_topography_rows",[])),
    )
