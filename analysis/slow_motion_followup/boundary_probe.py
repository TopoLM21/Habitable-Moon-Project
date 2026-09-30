"""Frozen-state counterfactuals: never a forward integration or production fix.

Retains actual signed boundary velocities and all physical/config parameters.
Only temporary copies of boundary labels and slab memory are changed. Slab
length is integrated at the frozen convergence rate for the stated interval;
all thermal/material/plate geometry remains frozen. No ledger claim is made.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import itertools
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tectonics import dynamics
from tectonics.genesis_starter_continuation import _load_cp, load_starter_source
from tectonics.genesis_starter_slab import young_slab_pull
from tectonics.kinematics import BoundaryType, classify_boundaries
from tectonics.plate_velocity_diagnostics import jsonable, weighted_stats, boundary_budget
from tectonics.simulation import load_config
from tectonics.subduction_memory import SubductionMemoryParameters, initialize_subduction_memory, advance_subduction_memory


def diagnose(root):
    root = Path(root).resolve()
    saved = json.loads((root/'continuation.json').read_text(encoding='utf-8'))
    for relative, expected in saved['checkpoint_sha256'].items():
        assert hashlib.sha256((root/relative).read_bytes()).hexdigest() == expected
    cfg = load_config(root/'mature_config.yaml')
    model, _, _ = load_starter_source(root/'young_context/starter_checkpoint.npz')
    cp = _load_cp(root/'mature_checkpoint', cfg)
    radius, mesh = model.thermal.radius_km, model.mesh
    area = mesh.physical_cell_areas_km2(radius)
    owner = cp.state.cell_plate
    params = dynamics.DynamicsParameters(**cfg['plate_dynamics'])
    params = replace(params, force_speed_scale_deg_per_myr=params.force_speed_scale_deg_per_myr*cp.thermal.tectonic_activity_factor)
    subp = SubductionMemoryParameters(**cfg['subduction_memory'])
    boundaries = classify_boundaries(mesh, cp.system, radius, **cfg['classification'])
    signed = [replace(b, boundary_type=BoundaryType.CONVERGENT if b.normal_rate_km_per_myr < 0 else BoundaryType.DIVERGENT if b.normal_rate_km_per_myr > 0 else BoundaryType.INACTIVE) for b in boundaries]
    convergence_only = [replace(b, boundary_type=BoundaryType.INACTIVE) if b.boundary_type == BoundaryType.DIVERGENT else b for b in signed]
    speed = lambda w: weighted_stats(np.linalg.norm(np.cross(w[owner], mesh.centroids)*radius, axis=1), area)
    def force(bds, memory):
        trace = {}
        with patch.object(dynamics, 'classify_boundaries', return_value=bds), young_slab_pull(subp):
            dynamics.update_plate_dynamics(mesh, cp.state, cp.system, cp.baseline, radius, 1., **cfg['classification'], params=params, mantle_flow=cp.mantle_flow, thermal_lithosphere_thickness_km=cp.thermal.thermal_lithosphere_thickness_km, subduction_memory=memory, subduction_memory_params=subp, trace=trace)
        factor = trace['drive_scale_rad_per_myr']
        return dict(mantle=speed(trace['common_mantle']), slab=speed(factor*trace['slab_drive_normalized']), ridge=speed(factor*trace['ridge_drive_normalized']), target=speed(trace['target_omega']), one_myr_relaxed=speed(trace['relaxed_omega']), one_myr_after_gauge=speed(trace['post_gauge_omega']), boundary_normalization_km=trace['boundary_weight_km'], slab_edges=trace['slab_edges'], ridge_factors=trace['ridge_push_factors'], raw_slab_drive=trace['slab_drive_raw'], normalized_slab_drive=trace['slab_drive_normalized'], raw_ridge_drive=trace['ridge_drive_raw'], normalized_ridge_drive=trace['ridge_drive_normalized'])
    stages = []
    for interval in (0., 1e-9, 1., 50., 100., 400.):
        mem = initialize_subduction_memory(cp.state.time_myr)
        if interval:
            mem, _ = advance_subduction_memory(mesh, cp.state, convergence_only, radius, interval, mem, subp)
        zones = []
        for key, z in sorted(mem.zones.items()):
            subset = [b for b in convergence_only if b.boundary_type == BoundaryType.CONVERGENT and dynamics._choose_subducting_side(cp.state, b) == key[0] and {b.plate_a, b.plate_b} == set(key)]
            lengths = np.array([dynamics._boundary_length_km(mesh,b,radius) for b in subset])
            torques = np.array([np.cross(b.midpoint, dynamics._normal_ab(mesh,b)*(1 if b.plate_a == key[0] else -1)) for b in subset])
            coherent = np.linalg.norm(np.sum(lengths[:,None]*torques,axis=0))/lengths.sum()
            zones.append(dict(**asdict(z), development_fraction=z.slab_length_km/subp.slab_length_cap_km, raw_torque_coherence=coherent, frozen_myr_to_full_length=subp.slab_length_cap_km/(subp.slab_length_growth_efficiency*z.convergence_rate_km_per_myr), frozen_myr_to_rollback_length=cfg['rollback']['min_slab_length_km']/(subp.slab_length_growth_efficiency*z.convergence_rate_km_per_myr)))
        stages.append(dict(frozen_interval_myr=interval, zones=zones, counterfactual_subducted_area_km2=mem.cumulative_subducted_area_km2, slab_only=force(convergence_only,mem), naive_signed_ridge_and_slab=force(signed,mem)))
    relabel_rows = []
    for perm_tuple in itertools.permutations(range(len(cp.system.plates))):
        perm = np.asarray(perm_tuple)
        labelled_state = deepcopy(cp.state)
        labelled_state.cell_plate = perm[owner]
        labelled_boundaries = [replace(b,plate_a=int(perm[b.plate_a]),plate_b=int(perm[b.plate_b])) for b in convergence_only]
        memory = initialize_subduction_memory(cp.state.time_myr)
        memory, _ = advance_subduction_memory(mesh,labelled_state,labelled_boundaries,radius,400.,memory,subp)
        tr = {}
        with young_slab_pull(subp):
            terms = dynamics._boundary_force_terms_reference(mesh,labelled_state,labelled_boundaries,radius,len(perm),params,cp.mantle_flow,cp.thermal.thermal_lithosphere_thickness_km,memory,trace=tr)
        drive, weights = terms[:2]
        nz = weights > 0
        drive[nz] /= weights[nz,None]
        physical_omega = np.deg2rad(params.force_speed_scale_deg_per_myr)*drive[perm]
        plate_weights = np.bincount(owner,weights=area,minlength=len(perm))
        gauge_omega = physical_omega-(plate_weights@physical_omega/plate_weights.sum())[None,:]
        relative_normals = [np.cross(physical_omega[b.plate_b]-physical_omega[b.plate_a],b.midpoint)@dynamics._normal_ab(mesh,b)*radius for b in boundaries]
        relabel_rows.append(dict(old_to_new=perm_tuple,slab_only_speed=speed(physical_omega),slab_only_speed_after_gauge=speed(gauge_omega),slab_omega_physical_order=physical_omega.tolist(),slab_relative_normal_velocity_km_myr=relative_normals))
    h = np.asarray(cp.state.mantle_lithosphere_thickness_km)
    rho = np.asarray(cp.state.mantle_lithosphere_density_anomaly_kg_m3)
    reference_normals = np.asarray(relabel_rows[0]['slab_relative_normal_velocity_km_myr'])
    for row in relabel_rows:
        row['max_relative_normal_change_from_identity_km_myr'] = float(np.max(np.abs(np.asarray(row.pop('slab_relative_normal_velocity_km_myr'))-reference_normals)))
    return jsonable(dict(source=str(root), time_myr=cp.state.time_myr, note=__doc__, current_boundary_budget=boundary_budget(mesh,boundaries,radius), actual=force(boundaries,deepcopy(cp.subduction_memory)), signed_boundary_budget=boundary_budget(mesh,signed,radius), physical_parameters=dict(subduction=asdict(subp), dynamics=asdict(params), rollback=cfg['rollback']), mantle_lithosphere_thickness_km=weighted_stats(h,area), density_anomaly_kg_m3=weighted_stats(rho,area), buoyancy_raw_ratio=weighted_stats(h*rho/6000,area), ridge_gpe_proxy=weighted_stats(rho*h*h,area), stages=stages, slab_400_myr_plate_id_permutations=relabel_rows))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    result=diagnose(args.checkpoint)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    compact=dict(source=result['source'], time_myr=result['time_myr'], actual=result['actual']['one_myr_after_gauge'], signed=result['signed_boundary_budget'], stages=[dict(interval=s['frozen_interval_myr'], zone_count=len(s['zones']), maximum_length=max((z['slab_length_km'] for z in s['zones']),default=0.), slab_speed=s['slab_only']['slab'], slab_only_target=s['slab_only']['target'], naive_ridge_target=s['naive_signed_ridge_and_slab']['target']) for s in result['stages']])
    print(json.dumps(compact,indent=2))
