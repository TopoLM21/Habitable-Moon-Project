"""Independent public-CLI continuations of the young-shell fracture law.

The matrix measures naturally resulting partitions; it never prescribes a
desired plate count or adds rift/continent histories. Each mature continuation
runs in its own process because the legacy runner installs global hooks.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.genesis_starter_continuation_validation import (
    SOURCE_ROOT, code_hashes, compare_npz, differences, digest, run_case,
    write_json,
)
from tectonics.mesh import build_icosphere, connected_components


CASES = (
    {"name": "strong_5120_10myr_step1", "source": "strong_50kpa_5120_step1", "duration": 10., "step": 1.},
    {"name": "strong_5120_50myr_step1", "source": "strong_50kpa_5120_step1", "duration": 50., "step": 1.},
    {"name": "strong_5120_10myr_step05", "source": "strong_50kpa_5120_step1", "duration": 10., "step": .5},
    {"name": "whole_1280_3myr_step1", "source": "strong_50kpa_1280_step1", "duration": 3., "step": 1.},
    {"name": "first_1280_1myr_step1", "source": "strong_50kpa_1280_step1", "duration": 1., "step": 1.},
    {"name": "resumed_1280_total3myr_step1", "resume": "first_1280_1myr_step1", "duration": 3., "step": 1.},
)


def source_code_hashes():
    result = code_hashes()
    result[str(Path(__file__).resolve().relative_to(ROOT)).replace("\\", "/")] = digest(__file__)
    return result


def inspect_actual_result(directory, result):
    """Inspect final ownership and events independently of runtime checks."""
    report = result["report"]
    metadata = json.loads((directory / "mature_checkpoint/meta.json").read_text(encoding="utf-8"))
    events = metadata["events"]
    origin = report["import"]["origin_time_myr"]
    new_events = [event for event in events if float(event.get("time_myr", origin)) > origin + 1e-10]
    with np.load(directory / "mature_checkpoint/state.npz", allow_pickle=False) as archive:
        fields = {key: archive[key] for key in archive.files}
    owner = fields["state_cell_plate"]
    subdivisions = 0
    while 20 * 4**subdivisions < len(owner):
        subdivisions += 1
    mesh = build_icosphere(subdivisions)
    if mesh.cell_count != len(owner):
        raise ValueError("Unexpected continuation mesh size")
    radius = float(events[0]["radius_km"])
    areas = mesh.physical_cell_areas_km2(radius)
    compact = np.array_equal(np.unique(owner), np.arange(report["final_plate_count"]))
    connected = all(len(connected_components(np.flatnonzero(owner == pid), mesh.neighbors)) == 1
                    for pid in np.unique(owner))
    finite = all(np.isfinite(value).all() for value in fields.values() if value.dtype.kind in "fiu")
    memory_path = directory / "young_context/fracture_memory.npz"
    if not memory_path.is_file():
        memory_path = directory / "fracture_memory.npz"
    result["checks"].update(
        final_ownership_compact=bool(compact), final_domains_connected=bool(connected),
        mature_numeric_fields_finite=bool(finite), fracture_memory_written=memory_path.is_file(),
    )
    result["observed"] = {
        "post_import_events": new_events,
        "post_import_event_counts": dict(Counter(str(event.get("kind")) for event in new_events)),
        "final_plate_areas_km2": [float(areas[owner == pid].sum()) for pid in np.unique(owner)],
        "final_continental_volume_km3": float(fields["continental_volume_km3"].sum()),
        "final_oceanic_volume_km3": float(fields["oceanic_volume_km3"].sum()),
        "final_mean_damage": float(areas @ fields["tidal_damage"] / areas.sum()),
        "final_max_damage": float(fields["tidal_damage"].max()),
        "young_fracture": report.get("young_fracture"),
        "young_fracture_events": report.get("young_fracture_events", []),
        "fracture_memory_path": str(memory_path.relative_to(directory)),
    }
    if memory_path.is_file():
        with np.load(memory_path, allow_pickle=False) as memory:
            result["observed"]["fracture_memory_fields"] = {
                key: {"shape": list(memory[key].shape), "dtype": str(memory[key].dtype)}
                for key in memory.files
            }
            result["checks"]["fracture_numeric_fields_finite"] = bool(all(
                np.isfinite(memory[key]).all() for key in memory.files
                if memory[key].dtype.kind in "fiu"))
    return result


def restart_comparison(output):
    left = output / "whole_1280_3myr_step1"
    right = output / "resumed_1280_total3myr_step1"
    a = json.loads((left / "continuation.json").read_text(encoding="utf-8"))
    b = json.loads((right / "continuation.json").read_text(encoding="utf-8"))
    mature = compare_npz(left / "mature_checkpoint/state.npz", right / "mature_checkpoint/state.npz")
    young = compare_npz(left / "young_context/starter_checkpoint.npz", right / "young_context/starter_checkpoint.npz")
    memory_relative = Path("young_context/fracture_memory.npz")
    if not (left / memory_relative).is_file():
        memory_relative = Path("fracture_memory.npz")
    memory = compare_npz(left / memory_relative, right / memory_relative)
    meta_diff = differences(
        json.loads((left / "mature_checkpoint/meta.json").read_text(encoding="utf-8")),
        json.loads((right / "mature_checkpoint/meta.json").read_text(encoding="utf-8")), "mature_meta")
    history_diff = differences(a["history"], b["history"], "history")
    totals = ("final_time_myr", "clocks", "cumulative_mantle_material_transfer_kg", "remaining_mantle_material_mass_kg",
              "final_plate_count", "transport_commits", "maximum_residual_rotation_deg", "final_mean_surface_speed_km_myr",
              "final_max_surface_speed_km_myr", "ocean_fraction", "young_fracture", "young_fracture_events")
    total_diff = differences({key: a[key] for key in totals}, {key: b[key] for key in totals}, "cumulative_final")
    return {
        "mature_arrays": mature, "young_arrays_and_metadata": young,
        "fracture_memory_arrays_and_metadata": memory,
        "mature_metadata_differences": meta_diff, "history_differences": history_diff,
        "cumulative_final_differences": total_diff,
        "all_exact": bool(mature["all_exact"] and young["all_exact"] and memory["all_exact"]
                          and not meta_diff and not history_diff and not total_diff),
        "note": "All numerical checkpoint fields, JSON metadata, fracture memory, events and full histories are compared; segment-local ledgers cover different intervals.",
    }


def step_comparison(output):
    left = output / "strong_5120_10myr_step1"
    right = output / "strong_5120_10myr_step05"
    a = json.loads((left / "continuation.json").read_text(encoding="utf-8"))
    b = json.loads((right / "continuation.json").read_text(encoding="utf-8"))
    fields = ("final_plate_count", "transport_commits", "maximum_residual_rotation_deg", "final_mean_surface_speed_km_myr",
              "final_max_surface_speed_km_myr", "ocean_fraction", "cumulative_mantle_material_transfer_kg")
    diagnostics = {}
    for name in fields:
        x, y = a[name], b[name]
        diagnostics[name] = {"step1": x, "step05": y, "difference": y - x,
                             "relative_difference_vs_step1": (y - x) / abs(x) if x else None}
    return {"same_final_time": a["final_time_myr"] == b["final_time_myr"], "final_diagnostics": diagnostics,
            "fracture_events_step1": a.get("young_fracture_events", []),
            "fracture_events_step05": b.get("young_fracture_events", []),
            "final_fracture_step1": a.get("young_fracture"),
            "final_fracture_step05": b.get("young_fracture"),
            "mature_field_comparison": compare_npz(left / "mature_checkpoint/state.npz", right / "mature_checkpoint/state.npz"),
            "note": "Measured sensitivity to mechanical/event decision intervals, not an asserted physical convergence pass. Changes in topology and event time are retained."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sources", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--case-timeout-seconds", type=float, default=900.)
    args = parser.parse_args(argv)
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error("Output must be new or empty")
    if not np.isfinite(args.case_timeout_seconds) or args.case_timeout_seconds <= 0:
        parser.error("Case timeout must be finite and positive")
    sources = {}
    for name in sorted({spec["source"] for spec in CASES if "source" in spec}):
        for filename in ("starter_checkpoint.npz", "parameters.json"):
            path = args.sources / name / filename
            sources[str(path.resolve())] = digest(path)
    before = source_code_hashes()
    args.output.mkdir(parents=True, exist_ok=True)
    result = {"format": "genesis-starter-fracture-validation-0.1", "source_hashes": sources,
              "code_hashes_before": before, "cases": {}, "checks": {},
              "interpretation": "No desired plate count is imposed; intact or slowly moving outcomes are valid numerical outcomes and are reported honestly."}
    write_json(args.output / "validation.json", result)
    for spec in CASES:
        if "resume" in spec and result["cases"][spec["resume"]]["returncode"] != 0:
            result["cases"][spec["name"]] = {"returncode": None, "skipped": "source continuation failed", "checks": {"source_available": False}}
        else:
            try:
                case = run_case(spec, args.output, args.sources, args.case_timeout_seconds)
                if case["returncode"] == 0 and "report" in case:
                    case = inspect_actual_result(args.output / spec["name"], case)
                result["cases"][spec["name"]] = case
            except (OSError, ValueError, KeyError, subprocess.TimeoutExpired) as exc:
                result["cases"][spec["name"]] = {"returncode": None, "error": str(exc), "checks": {"completed": False}}
        write_json(args.output / "validation.json", result)
    all_cases = all(all(case["checks"].values()) for case in result["cases"].values())
    result["checks"]["all_case_checks"] = all_cases
    result["checks"]["source_files_unchanged"] = all(digest(Path(path)) == value for path, value in sources.items())
    result["code_hashes_after"] = source_code_hashes()
    result["checks"]["code_unchanged_during_matrix"] = before == result["code_hashes_after"]
    if all_cases:
        result["restart_comparison"] = restart_comparison(args.output)
        result["checks"]["restart_exact"] = result["restart_comparison"]["all_exact"]
        result["step_comparison"] = step_comparison(args.output)
        result["checks"]["same_step_comparison_endpoint"] = result["step_comparison"]["same_final_time"]
    write_json(args.output / "validation.json", result)
    print(json.dumps(result["checks"], indent=2), flush=True)
    return 0 if all(result["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
