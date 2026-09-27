"""Synthetic receiving-side validation of the active v0.31 CPU runner.

This deliberately does not consume a real genesis/contact snapshot or certify
a physical handoff. It checks that explicitly imported young crust, water and
clocks survive the complete existing loop and a segmented restart.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import yaml

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from tectonics.checkpoint import load_checkpoint
from tectonics.genesis_mature import build_experimental_genesis_mature_import, save_experimental_genesis_mature_import
from tectonics.mesh import build_icosphere
from tectonics.plates import Plate, PlateSystem
from tectonics.thermal import ThermalParameters, initialize_thermal_state
from tectonics.topology import PlateTopologyManager, PlateTopologyParameters


def _run(root: Path, name: str, config: Path, source: Path, end: float) -> Path:
    checkpoint = root / name / "checkpoint"
    command = [sys.executable, str(PROJECT / "run_long_evolution_v131_cpu.py"),
               "--config", str(config), "--output", str(root / name),
               "--resume", str(source), "--end-time", str(end), "--dt", "0.0001",
               "--checkpoint", str(checkpoint), "--cpu-workers", "1", "--render-workers", "1"]
    print(f"Running {name}: target {end:.4f} Myr", flush=True)
    result = subprocess.run(command, cwd=PROJECT, text=True, encoding="utf-8", errors="replace",
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
    (root / f"{name}.log").write_text(result.stdout, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"{name} failed ({result.returncode}); see {root / (name + '.log')}:\n{result.stdout[-5000:]}")
    return checkpoint


def _load(path: Path):
    return load_checkpoint(path, PlateTopologyManager(PlateTopologyParameters()))


def validate(output: Path) -> dict:
    root = output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    config = yaml.safe_load((PROJECT / "configs/canonical_moon.yaml").read_text(encoding="utf-8"))
    config["mesh"]["subdivisions"] = 1
    config["plates"]["count"] = 2
    config["plates"]["seed"] = 123
    config["moon"]["rotation_period_hours"] = 47.136
    config["thermal"]["system_age_at_start_myr"] = 2.0
    config["thermal"]["initial_mantle_temperature_k"] = 1550.0
    config["thermal_evolution"]["time_step_myr"] = 0.0001
    config["tides"]["eccentricity_rms"] = 0.0002
    config["tides"]["eccentricity_history_csv"] = None
    for key in config["output"]:
        if key.endswith("dpi"):
            config["output"][key] = 60
    config_path = root / "synthetic_receiving_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    # A changed unused fresh-run seed makes accidental regeneration detectable.
    config["plates"]["seed"] = 92777
    alternate_path = root / "synthetic_receiving_alternate_seed.yaml"
    alternate_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    mesh = build_icosphere(1)
    n = mesh.cell_count
    radius = float(config["moon"]["radius_km"])
    owner = (mesh.centroids[:, 2] < 0).astype(np.int32)
    system = PlateSystem(owner, tuple(
        Plate(pid, int(np.flatnonzero(owner == pid)[0]), np.array([0., 0., 1.]), speed)
        for pid, speed in enumerate((0.001, -0.002))))
    thermal = initialize_thermal_state(0.5, radius, 7.12, ThermalParameters(**config["thermal"]))
    thermal.time_myr = 2.0
    thickness = np.linspace(3.0, 6.0, n)
    age = np.linspace(0.0, 0.2, n)
    areas = mesh.physical_cell_areas_km2(radius)
    bundle = build_experimental_genesis_mature_import(
        mesh, radius_km=radius, system=system, crust_age_myr=age,
        crust_thickness_km=thickness, tidal_damage=np.linspace(0.0, 0.4, n),
        mantle_lithosphere_thickness_km=np.linspace(10.0, 20.0, n),
        mantle_lithosphere_density_anomaly_kg_m3=np.linspace(0.0, 30.0, n),
        thermal=thermal, elevation_m=np.linspace(-800.0, -100.0, n),
        water_volume_km3=1126719620.5563111,
        mantle_cell_omega_rad_per_myr=np.tile([0.0003, -0.0005, 0.0001], (n, 1)),
        source_mass_kg=areas * thickness * 1e9 * 2900.0,
        source_enthalpy_j=areas * thickness * 1e9 * 2900.0 * 1200.0 * 1550.0,
        next_plume_birth_time_myr=12.0,
        source_metadata={"kind": "synthetic receiving-side validation only", "real_genesis_handoff": False,
                         "thermal_parameters": config["thermal"], "time_myr": 2.0},
        topology_parameters=PlateTopologyParameters(**config["plate_topology"]),
    )
    imported = save_experimental_genesis_mature_import(root / "synthetic_import", bundle) / "mature_checkpoint"
    whole_path = _run(root, "uninterrupted", config_path, imported, 2.0003)
    first_path = _run(root, "first_step", alternate_path, imported, 2.0001)
    split_path = _run(root, "resumed", alternate_path, first_path, 2.0003)
    whole, first, split = _load(whole_path), _load(first_path), _load(split_path)

    with np.load(whole_path / "state.npz", allow_pickle=False) as a, np.load(split_path / "state.npz", allow_pickle=False) as b:
        arrays_equal = set(a.files) == set(b.files) and all(np.array_equal(a[key], b[key]) for key in a.files)
        changed_arrays = [key for key in a.files if key not in b.files or not np.array_equal(a[key], b[key])]
        array_count = len(a.files)
    whole_meta = json.loads((whole_path / "meta.json").read_text(encoding="utf-8"))
    split_meta = json.loads((split_path / "meta.json").read_text(encoding="utf-8"))
    changed_meta = [key for key in set(whole_meta) | set(split_meta) if whole_meta.get(key) != split_meta.get(key)]
    initial_volume = float(np.sum(bundle.checkpoint.state.oceanic_volume_km3))
    snapshots = {}
    for name, cp in (("first_step", first), ("uninterrupted", whole), ("resumed", split)):
        times = {key: float(value.time_myr) for key, value in (
            ("lithosphere", cp.state), ("thermal", cp.thermal), ("continental_cycle", cp.cycle),
            ("topography", cp.topo), ("hydrosphere", cp.hydrosphere), ("mantle", cp.mantle_flow),
            ("plumes", cp.plume_state), ("hotspot_tracks", cp.hotspot_track_state))}
        elapsed = len(cp.lithosphere_rows) * 0.0001
        snapshots[name] = {
            "clocks_myr": times, "system_age_myr": cp.thermal.system_age_myr,
            "oceanic_volume_km3": float(np.sum(cp.state.oceanic_volume_km3)),
            "oceanic_volume_change_km3": float(np.sum(cp.state.oceanic_volume_km3)) - initial_volume,
            "max_crust_thickness_change_km": float(np.max(np.abs(cp.state.crust_thickness_km - thickness))),
            "max_crust_age_error_myr": float(np.max(np.abs(cp.state.crust_age_myr - age - elapsed))),
            "water_volume_km3": cp.hydrosphere.water_volume_km3,
            "plate_count": len(cp.system.plates), "plume_count": len(cp.plume_state.ages_myr),
            "continental_volume_km3": float(np.sum(cp.state.continental_volume_km3)),
            "transport_commits": cp.transport_state.cumulative_commit_count,
            "mantle_temperature_k": cp.thermal.mantle_temperature_k,
            "max_mechanical_mantle_thickness_change_km": float(np.max(np.abs(cp.state.mantle_lithosphere_thickness_km - bundle.checkpoint.state.mantle_lithosphere_thickness_km))),
        }
    target_times = {"first_step": 2.0001, "uninterrupted": 2.0003, "resumed": 2.0003}
    passed = arrays_equal and not changed_meta and all(
        info["oceanic_volume_change_km3"] == 0.0
        and info["water_volume_km3"] == bundle.checkpoint.hydrosphere.water_volume_km3
        and info["max_crust_thickness_change_km"] < 1e-12
        and info["max_crust_age_error_myr"] < 1e-12
        and info["plate_count"] == 2 and info["plume_count"] == 0
        and info["continental_volume_km3"] == 0.0
        and abs(info["system_age_myr"] - target_times[name]) < 1e-12
        and all(abs(value - target_times[name]) < 1e-12 for value in info["clocks_myr"].values())
        and max(info["clocks_myr"].values()) - min(info["clocks_myr"].values()) < 1e-12
        for name, info in snapshots.items())
    report = {
        "kind": "synthetic_receiving_side_validation", "real_genesis_handoff": False,
        "runner": "run_long_evolution_v131_cpu.py", "steps": 3, "dt_myr": 0.0001,
        "passed": bool(passed), "restart_arrays_exact": bool(arrays_equal),
        "restart_array_count": array_count, "restart_changed_arrays": changed_arrays,
        "restart_metadata_exact": not changed_meta, "restart_changed_metadata": changed_meta,
        "fresh_plate_seed_varied": [123, 92777], "initial_oceanic_volume_km3": initial_volume,
        "snapshots": snapshots,
        "limits": ["No material remap commits occur in this short fixture.",
                   "Depth-resolved mass and energy archives are not evolved.",
                   "Mechanical mantle thickness follows the receiving engine's age law; this fixture does not certify physical continuity."],
    }
    (root / "validation.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not validate(args.output)["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
