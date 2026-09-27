"""Seven bounded molten-start controls; numerical sensitivity, not calibration."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, fields, is_dataclass, replace
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_genesis_starter import FORMAT, LIMITATIONS, _write_json
from tectonics.genesis import parameters_from_config
from tectonics.genesis_shell import shell_parameters_from_config
from tectonics.genesis_starter import StarterModel, StarterParameters
from tectonics.genesis_tides import tidal_parameters_from_config
from tectonics.mesh import build_icosphere, connected_components
from tectonics.simulation import load_config
from visualization.genesis_starter import save_starter_snapshot


CASES = (
    dict(name="canonical_20kpa_5120_step1", traction=20000., subdivisions=4, duration=10., step=1.),
    dict(name="strong_50kpa_5120_step1", traction=50000., subdivisions=4, duration=10., step=1.),
    dict(name="strong_50kpa_5120_step05", traction=50000., subdivisions=4, duration=10., step=.5),
    dict(name="strong_50kpa_1280_step1", traction=50000., subdivisions=3, duration=10., step=1.),
    dict(name="strong_50kpa_5120_loading0025", traction=50000., subdivisions=4, duration=10., step=1., loading=.025),
    dict(name="intact_5120", traction=0., subdivisions=4, duration=3., step=1., intact=True),
    dict(name="seed20260928_50kpa_5120", traction=50000., subdivisions=4, duration=3., step=1., seed=20260928),
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def same(a, b):
    """Exact arrays/scalars, with JSON events compared in their wire format."""
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if is_dataclass(a):
        return type(a) is type(b) and all(same(getattr(a, f.name), getattr(b, f.name)) for f in fields(a))
    if isinstance(a, dict):
        # TopologyEvent parents/children are tuples in memory and JSON arrays
        # on disk. The checkpoint's event contract is ordered JSON content;
        # its conversion is not a change in any mechanical state or history.
        if a.get("kind") == "split":
            return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(same(x, y) for x,y in zip(a,b))
    return a == b


def run_case(spec, config, config_path, directory):
    directory.mkdir()
    thermal = parameters_from_config(config)
    tides = tidal_parameters_from_config(config, thermal)
    shell = replace(shell_parameters_from_config(config), subdivisions=spec["subdivisions"],
                    convective_traction_pa=spec["traction"])
    parameters = StarterParameters(seed=spec.get("seed", 20260927),
        max_loading_interval_myr=spec.get("loading", .05))
    if spec.get("intact"):
        parameters = replace(parameters, cooling_contrast_fraction=0., tidal_mechanics=False)
    setup_start = perf_counter()
    model = StarterModel(build_icosphere(shell.subdivisions), thermal, tides, shell, parameters)
    state = model.initial_state()
    setup_seconds = perf_counter()-setup_start
    initial = model.diagnose(state)
    rows = [initial]
    wall_seconds = 0.
    while state.time_myr < spec["duration"] and not state.stopped_reason:
        target = min(spec["duration"], state.time_myr+spec["step"])
        before = perf_counter()
        state = model.advance(state, target)
        wall_seconds += perf_counter()-before
        rows.append(model.diagnose(state))
    model.save_state(directory/"starter_checkpoint.npz", state)
    restored = model.load_state(directory/"starter_checkpoint.npz")
    restart_exact = same(state, restored)
    next_step_exact = None
    if not state.stopped_reason:
        next_step_exact = same(model.advance(state, state.time_myr+.05),
                               model.advance(restored, restored.time_myr+.05))
    owner = state.system.cell_plate
    child_areas = [float(model.areas[owner == p].sum()) for p in range(len(state.system.plates))]
    thermal_sample = model.loading.sample(state.thermal_context)
    water = thermal_sample.thermal
    checks = {
        "initial_fully_molten_no_lid_damage_continents_relief": bool(
            initial["plate_count"] == 0 and initial["domain_count"] == 1
            and initial["mantle_melt_fraction"] == 1 and initial["lid_thickness_km"] == 0
            and initial["max_damage"] == 0 and initial["continental_volume_km3"] == 0
            and initial["initial_relief_m"] == 0),
        "duration_or_explicit_stop": state.time_myr == spec["duration"] or state.stopped_reason is not None,
        "ownership_covers_each_cell_once": bool(owner.shape == (model.mesh.cell_count,)
            and np.array_equal(np.unique(owner), np.arange(len(state.system.plates)))),
        "total_area_conserved": bool(np.isclose(sum(child_areas), model.areas.sum(), rtol=2e-15, atol=0)),
        "each_domain_connected": all(len(connected_components(np.flatnonzero(owner == pid), model.mesh.neighbors)) == 1
                                      for pid in range(len(state.system.plates))),
        "all_split_cells_eligible": bool(np.all(state.eligible[state.split_band])),
        "children_meet_physical_area_threshold": all(a >= parameters.min_child_area_km2 for a in child_areas),
        "thermal_orbital_clock_matches": state.time_myr == state.thermal_context.orbit.time_myr,
        "thermal_energy_accounted": max(abs(row["thermal_energy_relative_residual"]) for row in rows) < 1e-10,
        "column_energy_accounted_separately": max(abs(row["column_energy_relative_residual"]) for row in rows) < 1e-10,
        "water_inventory_conserved": bool(np.isclose(water["ocean_mass_kg"]+water["vapor_mass_kg"],
                                                     water["total_water_mass_kg"], rtol=2e-15)),
        "terminal_restart_exact": restart_exact,
        "actual_following_step_exact_if_unstopped": next_step_exact is not False,
        "mature_handoff_not_claimed": not rows[-1]["mature_handoff_ready"],
    }
    if spec.get("intact"):
        checks["intact_control_has_no_fracture"] = len(state.system.plates) == 1 and not np.any(state.damage)
    provenance = {"config_path": str(config_path), "config_sha256": digest(config_path),
        "control": "intact" if spec.get("intact") else None,
        "control_description": "All mechanical drivers disabled; orbital heating retained." if spec.get("intact") else None}
    metadata = {"format": FORMAT, **model.configuration, "step_myr": spec["step"], "provenance": provenance}
    _write_json(directory/"parameters.json", metadata)
    summary = {"format": FORMAT, "status": state.stopped_reason or "completed", "case": spec,
        "requested_end_time_myr": spec["duration"], "final": rows[-1], "events": state.events,
        "mature_handoff": False, "candidate_partition": state.stopped_reason == "first_partition",
        "limitations": LIMITATIONS, "provenance": provenance, "checks": checks,
        "wall_model_advance_seconds": wall_seconds, "wall_model_setup_seconds": setup_seconds,
        "runtime_scope": "Sum of model.advance calls only; excludes setup, diagnosis, checkpoint audit and rendering.",
        "actual_following_step_exact": next_step_exact,
        "restart_equality_definition": "All numerical arrays and scalar state fields exact; event history exact as canonical ordered JSON (tuple/list wire conversion allowed).",
        "restart_following_step_note": "Terminal stopped states are restored exactly but cannot be advanced." if state.stopped_reason else None,
        "child_areas_km2": child_areas, "row_scope": "All accepted macro/output endpoints, including the physical stopping sample.",
        "rows": rows}
    _write_json(directory/"summary.json", summary)
    with (directory/"history.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(directory/"starter_fields.npz", vertices=model.mesh.vertices,
        faces=model.mesh.faces, centroids=model.mesh.centroids, damage=state.damage,
        yield_ratio=state.yield_ratio, domain_id=owner, eligible=state.eligible,
        split_band=state.split_band, cell_area_km2=model.areas)
    print(json.dumps({"case": spec["name"], "wall_seconds": wall_seconds,
        "final": rows[-1], "checks": checks}, ensure_ascii=False), flush=True)
    save_starter_snapshot(model.mesh, state, directory/"genesis_starter.png", rows, summary)
    return summary, model, state


def compare(first, second, *, nearest=False):
    a, ma, sa = first
    b, mb, sb = second
    result = {"first": a["case"]["name"], "second": b["case"]["name"],
        "comparison_scope": "Numerical sensitivity of terminal candidate states; not a physical convergence claim.",
        "end_age_difference_myr": sb.time_myr-sa.time_myr,
        "first_partition_in_both": sa.stopped_reason == sb.stopped_reason == "first_partition",
        "maximum_damage_difference": float(sb.damage.max()-sa.damage.max()),
        "advance_runtime_ratio_second_over_first": b["wall_model_advance_seconds"]/a["wall_model_advance_seconds"],
        "area_weighted_label_permutation_agreement": None}
    if result["first_partition_in_both"] and len(sa.system.plates) == len(sb.system.plates):
        labels = sb.system.cell_plate
        if nearest:
            indices = cKDTree(mb.mesh.centroids).query(ma.mesh.centroids)[1]
            labels = labels[indices]
            result["map_sampling"] = "Second mesh nearest centroid at first mesh cell centers; weights are first mesh areas."
        else:
            if not np.array_equal(ma.mesh.faces, mb.mesh.faces):
                raise ValueError("Identical meshes required unless nearest-cell comparison is explicit")
            result["map_sampling"] = "Same cell identities, first mesh area weights."
        count = len(sa.system.plates)
        overlap = np.zeros((count, count))
        np.add.at(overlap, (sa.system.cell_plate, labels), ma.areas)
        x, y = linear_sum_assignment(-overlap)
        result["area_weighted_label_permutation_agreement"] = float(overlap[x,y].sum()/ma.areas.sum())
        result["overlap_area_km2"] = overlap.tolist()
        result["label_assignment"] = [[int(i), int(j)] for i,j in zip(x,y)]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"results/genesis_runs/starter_validation_20260927")
    parser.add_argument("--config", type=Path, default=ROOT/"configs/genesis_moon.yaml")
    args = parser.parse_args()
    output, config_path = args.output.resolve(), args.config.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Validation output must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    source_paths = sorted((ROOT/"tectonics").glob("*.py"))+[
        Path(__file__).resolve(), ROOT/"run_genesis_starter.py",
        ROOT/"visualization/genesis_starter.py", ROOT/"visualization/raster.py"]
    source_hashes = {str(path.relative_to(ROOT)): digest(path) for path in source_paths}
    config_hash = digest(config_path)
    config = load_config(config_path)
    results = {}
    for spec in CASES:
        results[spec["name"]] = run_case(spec, config, config_path, output/spec["name"])
    baseline = results["strong_50kpa_5120_step1"]
    comparisons = {
        "macro_output_step_1_vs_05_myr": compare(baseline, results["strong_50kpa_5120_step05"]),
        "loading_interval_005_vs_0025_myr": compare(baseline, results["strong_50kpa_5120_loading0025"]),
        "mesh_5120_vs_1280_cells": compare(baseline, results["strong_50kpa_1280_step1"], nearest=True),
    }
    checks = {"all_case_contracts_pass": all(all(value[0]["checks"].values()) for value in results.values()),
        "source_code_unchanged_during_runs": all(digest(ROOT/name) == value for name,value in source_hashes.items()),
        "config_unchanged_during_runs": digest(config_path) == config_hash}
    report = {"scope": __doc__, "cases": {name: value[0] for name,value in results.items()},
        "comparisons": comparisons, "checks": checks,
        "source_sha256": source_hashes, "config_sha256": config_hash,
        "limitations": LIMITATIONS+["Parameter sensitivity and mesh dependence are reported without tuning any case to force a split."],
        "artifact_sha256": {str(path.relative_to(output)): digest(path)
                           for path in output.rglob("*") if path.is_file()}}
    _write_json(output/"validation.json", report)
    print(json.dumps({"comparisons": comparisons, "checks": checks}, indent=2), flush=True)
    if not all(checks.values()):
        raise SystemExit("Starter validation did not pass every requested contract")


if __name__ == "__main__":
    main()
