#!/usr/bin/env python3
"""Build and visualize the first tectonic-plate kinematics prototype."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from tectonics.kinematics import BoundaryType
from tectonics.simulation import build_prototype, load_config
from visualization.maps import save_boundary_map, save_plate_map, save_velocity_histogram


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/canonical_moon.yaml")
    parser.add_argument("--output", default=None, help="Override output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    result = build_prototype(config)

    output_dir = Path(args.output or config["output"]["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    dpi = int(config["output"].get("dpi", 180))

    save_plate_map(result, output_dir / "plate_map.png", dpi=dpi)
    save_boundary_map(result, output_dir / "boundary_map.png", dpi=dpi)
    save_velocity_histogram(result, output_dir / "normal_rate_histogram.png", dpi=dpi)

    counts = Counter(BoundaryType(b.boundary_type).name.lower() for b in result.boundaries)
    area_sum = float(result.mesh.areas_unit_sphere.sum())
    physical_area = float(result.mesh.physical_cell_areas_km2(config["moon"]["radius_km"]).sum())

    summary = {
        "moon": config["moon"]["name"],
        "mesh_cells": result.mesh.cell_count,
        "mesh_vertices": result.mesh.vertex_count,
        "plate_count": len(result.plates.plates),
        "boundary_edge_count": len(result.boundaries),
        "boundary_types": dict(sorted(counts.items())),
        "unit_sphere_area": area_sum,
        "expected_unit_sphere_area": 4.0 * 3.141592653589793,
        "surface_area_km2": physical_area,
        "rigid_motion_residual": result.rigid_residual,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved diagnostics to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
