"""Read-only checks of historical ownership and corrected continuation files.

Example (from the project root):
  python analysis/validate_tectonic_repairs.py --run results/gui_runs/v031_20260913_212557 \
      --combined-checkpoint results/tectonic_repair_validation/combined_cp4520 \
      --boundary-half-width-km 400 --output results/tectonic_repair_validation

The boundary normal comparison is a diagnostic only; it never changes fluxes,
classification, the input checkpoint, or any running simulation.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tectonics.boundary_geometry import estimate_boundary_normals
from tectonics.connectivity import repair_plate_connectivity
from tectonics.checkpoint import load_checkpoint
from tectonics.kinematics import classify_boundaries, BoundaryType
from tectonics.mesh import build_icosphere, connected_components
from tectonics.topology import PlateTopologyManager, PlateTopologyParameters


def ownership_metrics(mesh, owner, radius):
    areas = mesh.physical_cell_areas_km2(radius)
    components = [connected_components(np.flatnonzero(owner == pid), mesh.neighbors)
                  for pid in np.unique(owner)]
    detached = sum(sum(len(c) for c in group) - max(map(len, group)) for group in components)
    return {
        "plates": len(components),
        "components": sum(map(len, components)),
        "detached_cells": detached,
        "largest_plate_area_fraction": float(max(np.sum(areas[owner == pid])
                                                 for pid in np.unique(owner)) / areas.sum()),
    }


def sea_metrics(meta, start):
    hydro = [r for r in meta['hydrosphere_rows'] if r['time_myr'] >= start]
    sea = np.array([r['sea_level_m'] for r in hydro])
    time = np.array([r['time_myr'] for r in hydro]) - start
    ocean = np.array([r['ocean_area_fraction'] for r in hydro])
    residual = sea - np.polyval(np.polyfit(time, sea, 1), time)
    ledger = [abs(r['global_continental_ledger_error_km3']) for r in meta['sediment_rows']
              if r['time_myr'] >= start]
    return {
        'interval_myr': [start, float(meta['time_myr'])],
        'max_abs_sea_step_m': float(np.max(np.abs(np.diff(sea)))),
        'detrended_sea_rms_m': float(np.sqrt(np.mean(residual**2))),
        'max_ocean_area_step_percentage_points': float(100*np.max(np.abs(np.diff(ocean)))),
        'sea_level_m': sea.tolist(),
        'times_myr': (time + start).tolist(),
        'water_inventory_range_km3': float(np.ptp([r['water_volume_km3'] for r in hydro])),
        'max_abs_water_relative_error': max(abs(r['relative_volume_error']) for r in hydro),
        'max_abs_continental_ledger_error_km3': max(ledger, default=0.),
    }


def ridge_continuity(mesh, checkpoint, radius):
    cp = load_checkpoint(checkpoint, PlateTopologyManager(
        PlateTopologyParameters(connected_collision_contacts=False)))
    bounds = classify_boundaries(mesh, cp.system, radius, 4., 1.)
    fraction = cp.state.continental_fraction
    edges = [b for b in bounds if b.boundary_type == BoundaryType.DIVERGENT
             and fraction[b.face_a] < .5 and fraction[b.face_b] < .5]
    at_vertex = defaultdict(list)
    for i, edge in enumerate(edges):
        pair = tuple(sorted((edge.plate_a, edge.plate_b)))
        for vertex in (edge.vertex_u, edge.vertex_v):
            at_vertex[pair, vertex].append(i)
    seen, lengths = set(), []
    for start in range(len(edges)):
        if start in seen:
            continue
        stack, length = [start], 0.
        seen.add(start)
        while stack:
            edge = edges[stack.pop()]
            pair = tuple(sorted((edge.plate_a, edge.plate_b)))
            length += radius*np.arccos(np.clip(
                mesh.vertices[edge.vertex_u] @ mesh.vertices[edge.vertex_v], -1., 1.))
            for vertex in (edge.vertex_u, edge.vertex_v):
                for i in at_vertex[pair, vertex]:
                    if i not in seen:
                        seen.add(i)
                        stack.append(i)
        lengths.append(float(length))
    return {'boundary_edges': len(bounds), 'oceanic_divergent_edges': len(edges),
            'oceanic_divergent_components': len(lengths),
            'longest_oceanic_divergent_component_km': max(lengths, default=0.),
            'total_oceanic_divergent_length_km': sum(lengths)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('configs/canonical_moon.yaml'))
    parser.add_argument('--times', nargs='+', type=int, default=[500, 1000, 3500, 4000, 4500])
    parser.add_argument('--combined-checkpoint', type=Path)
    parser.add_argument('--comparison-checkpoint', type=Path)
    parser.add_argument('--start-time', type=float, default=4480.)
    parser.add_argument('--boundary-half-width-km', type=float, default=0.)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    radius = float(cfg['moon']['radius_km'])
    topology = cfg['plate_topology']
    mesh = None
    rows = []
    for time in args.times:
        checkpoint = args.run / f'gui_checkpoint_{time:06d}_Myr'
        with np.load(checkpoint / 'state.npz') as arrays:
            owner = arrays['state_cell_plate']
        subdivision = int(round(np.log(len(owner)/20.) / np.log(4.)))
        if 20*4**subdivision != len(owner):
            raise ValueError(f'Unexpected mesh cell count: {len(owner)}')
        if mesh is None or mesh.cell_count != len(owner):
            mesh = build_icosphere(subdivision)
        result = repair_plate_connectivity(mesh, owner, radius,
            minimum_independent_area_km2=topology.get('min_plate_area_km2'),
            minimum_independent_cells=int(topology['min_plate_cells']))
        before = ownership_metrics(mesh, owner, radius)
        after = ownership_metrics(mesh, result.cell_plate, radius)
        assert after['plates'] == after['components']
        row = {'time_myr': time, 'before': before, 'after': after,
               'promoted_components': result.promoted_components,
               'reassigned_cells': result.reassigned_cells}
        if args.boundary_half_width_km > 0 and time == args.times[-1]:
            geometry = estimate_boundary_normals(mesh, result.cell_plate, radius,
                half_width_km=args.boundary_half_width_km)
            angle = np.degrees(np.arccos(np.clip(np.sum(
                geometry.raw_normals*geometry.normals, axis=1), -1., 1.)))
            row['experimental_geometry_diagnostic'] = {
                'half_width_km': args.boundary_half_width_km,
                'boundary_edges': len(angle),
                'protected_edges': int(np.count_nonzero(geometry.protected_edges)),
                'mean_normal_change_degrees': float(np.mean(angle)),
                'note': 'Change from raw normals, not an accuracy estimate. No physical use.'}
        rows.append(row)
    report = {'source_run': str(args.run.resolve()), 'saved_states': rows}
    if args.combined_checkpoint:
        meta = json.loads((args.combined_checkpoint/'meta.json').read_text(encoding='utf-8'))
        with np.load(args.combined_checkpoint/'state.npz') as arrays:
            owner = arrays['state_cell_plate']
            assert np.array_equal(owner, arrays['system_cell_plate'])
        combined = ownership_metrics(mesh, owner, radius)
        assert combined['components'] == combined['plates']
        combined.update(sea_metrics(meta, args.start_time))
        combined['repair_steps'] = [r for r in meta['topology_rows'] if r['time_myr'] > args.start_time]
        combined['checkpoint'] = str(args.combined_checkpoint.resolve())
        combined['ridge_continuity'] = ridge_continuity(mesh, args.combined_checkpoint, radius)
        report['combined_continuation'] = combined
    if args.comparison_checkpoint:
        report['comparison_ridge_continuity'] = ridge_continuity(mesh, args.comparison_checkpoint, radius)
        report['ridge_continuity_method'] = ('Same-pair divergent edges sharing vertices; both cells '
            'continental fraction < 0.5; normal/inactive thresholds 4/1 km/Myr. '
            'Connectivity is not a resolved ridge-axis or physical-accuracy metric.')
    path = args.output/'repair_validation.json'
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(path.resolve())


if __name__ == '__main__':
    main()
