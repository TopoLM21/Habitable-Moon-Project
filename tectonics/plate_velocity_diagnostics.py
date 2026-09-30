"""Read-only velocity budget for a saved world or an actual dynamics call.

All speed statistics are area weighted (boundary statistics length weighted).
Vector contributions add; their norms generally do not. A frozen checkpoint
probe is labelled separately from a captured real integration step.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import numpy as np

from . import dynamics
from .kinematics import BoundaryType, classify_boundaries
from .mantle import plate_mean_mantle_omega, plate_rigid_mantle_fit, mantle_flow_rms_rad_per_myr


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    return value


def weighted_stats(values, weights):
    """Weighted inverse-CDF quantiles, with explicit empty-population zeros."""
    values, weights = np.asarray(values), np.asarray(weights)
    if not len(values) or weights.sum() <= 0:
        return dict(mean=0., median=0., p90=0., rms=0., max=0.)
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order]) / weights.sum()
    quantile = lambda q: float(values[order[min(np.searchsorted(cumulative, q), len(values)-1)]])
    return dict(mean=float(weights @ values / weights.sum()), median=quantile(.5), p90=quantile(.9),
                rms=float(np.sqrt(weights @ (values*values) / weights.sum())), max=float(values.max()))


def boundary_budget(mesh, boundaries, radius_km):
    lengths = np.array([dynamics._boundary_length_km(mesh, b, radius_km) for b in boundaries])
    total = float(lengths.sum())
    normal = np.array([b.normal_rate_km_per_myr for b in boundaries])
    speed = np.array([b.relative_speed_km_per_myr for b in boundaries])
    kinds = np.array([int(b.boundary_type) for b in boundaries])
    by_kind = {}
    for kind in BoundaryType:
        mask = kinds == int(kind)
        by_kind[kind.name] = dict(count=int(mask.sum()), length_km=float(lengths[mask].sum()),
            length_fraction=float(lengths[mask].sum()/total) if total else 0.)
    return dict(count=len(boundaries), total_length_km=total, by_type=by_kind,
        active_length_fraction=1.-by_kind["INACTIVE"]["length_fraction"] if total else 0.,
        relative_speed_km_per_myr=weighted_stats(speed, lengths),
        absolute_normal_speed_km_per_myr=weighted_stats(np.abs(normal), lengths),
        max_convergent_normal_speed_km_per_myr=float(np.maximum(-normal, 0.).max(initial=0.)),
        max_divergent_normal_speed_km_per_myr=float(np.maximum(normal, 0.).max(initial=0.)),
        positive_normal_length_km=float(lengths[normal > 0.].sum()),
        negative_normal_length_km=float(lengths[normal < 0.].sum()))


def velocity_budget(mesh, state, system, mantle, radius_km, trace, boundaries, *,
                    subduction_memory=None, slab_reference_length_km=1800.,
                    evaluation="captured_dynamics_step"):
    """Summarize a real ``update_plate_dynamics(trace=...)`` without mutating it.

    ``state`` and ``system`` are the inputs at that force evaluation, before
    any later material transport/topology changes. ``trace.final_omega`` is
    the returned dynamics result, never a reconstructed historical velocity.
    """
    owner, x = np.asarray(state.cell_plate), mesh.centroids
    n, radius = len(system.plates), float(radius_km)
    areas = mesh.physical_cell_areas_km2(radius)
    local = np.cross(mantle.cell_omega_rad_per_myr, x)*radius
    local_speed = np.linalg.norm(local, axis=1)
    from .lithosphere import mantle_lithosphere_negative_buoyancy_proxy
    negative_buoyancy = mantle_lithosphere_negative_buoyancy_proxy(state)
    fit = plate_rigid_mantle_fit(mesh, owner, n, radius, mantle)
    average = plate_mean_mantle_omega(mesh, owner, n, radius, mantle)
    def speeds(omega):
        return np.linalg.norm(np.cross(np.asarray(omega)[owner], x)*radius, axis=1)
    scale, drag = trace["drive_scale_rad_per_myr"], trace["drag_factor"][:, None]
    contributions = dict(mantle=trace["common_mantle"],
        ridge=scale*trace["ridge_drive_normalized"],
        slab=scale*trace["slab_drive_normalized"],
        residual_slab=scale*trace["residual_slab_drive"],
        gpe=scale*trace["gpe_component"], rollback=trace["rollback_omega"],
        collision_resistance=trace["collision_resistance_omega"],
        transform_resistance=trace["transform_resistance_omega"])
    if "slab_constraint_omega" in trace:
        contributions["slab_constraint"] = trace["slab_constraint_omega"]
    stages = dict(best_rigid_fit=fit.omega_rad_per_myr, simple_area_average=average,
        selected_projection=trace["mantle_omega"], after_memory=trace["common_mantle"],
        target=trace["target_omega"], current=trace["current_omega"],
        after_relaxation=trace["relaxed_omega"], after_net_rotation=trace["post_gauge_omega"],
        actual_returned=trace["final_omega"])
    global_stages = {name: weighted_stats(speeds(w), areas) for name, w in stages.items()}
    global_stages["local_mantle"] = weighted_stats(local_speed, areas)
    global_contributions = {name: weighted_stats(speeds(w), areas) for name, w in contributions.items()}
    plate_rows = []
    for pid in range(n):
        mask = owner == pid
        local_weight = areas[mask]
        row = dict(plate_id=int(system.plates[pid].plate_id), area_fraction=float(local_weight.sum()/areas.sum()),
            actual_angular_speed_rad_per_myr=float(np.linalg.norm(trace["current_omega"][pid])),
            local_mantle_speed_km_per_myr=weighted_stats(local_speed[mask], local_weight),
            best_fit_relative_residual=float(fit.relative_residual[pid]),
            represented_kinetic_proxy_fraction=float(fit.represented_kinetic_fraction[pid]),
            residual_rms_km_per_myr=float(fit.residual_rms_speed_km_per_myr[pid]),
            fit_moment_rank=int(fit.moment_rank[pid]),
            relaxation_alpha=float(trace["alpha"]), drag_factor=float(drag[pid, 0]),
            boundary_normalization_length_km=float(trace["boundary_weight_km"][pid]),
            collision_length_weighted_km=float(trace["collision_length_weighted_km"][pid]),
            transform_length_weighted_km=float(trace["transform_length_weighted_km"][pid]),
            ridge_gpe_factor=float(trace["ridge_push_factors"][pid]))
        row["available_negative_buoyancy_km_kg_m3"] = weighted_stats(negative_buoyancy[mask], local_weight)
        row["stages"] = {name: dict(omega_rad_per_myr=w[pid].tolist(),
            surface_speed_km_per_myr=weighted_stats(speeds(w)[mask], local_weight)) for name, w in stages.items()}
        row["contributions"] = {name: dict(omega_rad_per_myr=w[pid].tolist(),
            surface_speed_km_per_myr=weighted_stats(speeds(w)[mask], local_weight)) for name, w in contributions.items()}
        plate_rows.append(row)
    residual_energy = np.sum(areas[:, None]*(local-np.cross(fit.omega_rad_per_myr[owner], x)*radius)**2)
    local_energy = np.sum(areas[:, None]*local**2)
    before, after = trace["relaxed_omega"], trace["post_gauge_omega"]
    changes = [np.linalg.norm(np.cross((before[b.plate_b]-before[b.plate_a])
                 -(after[b.plate_b]-after[b.plate_a]), b.midpoint)*radius) for b in boundaries]
    zones = []
    from .genesis_starter_slab import slab_development_fraction
    for zone in (() if subduction_memory is None else subduction_memory.zones.values()):
        zones.append({**asdict(zone), "development_fraction": slab_development_fraction(zone, slab_reference_length_km)})
    rms = mantle_flow_rms_rad_per_myr(mantle)
    vector_sum = sum(contributions.values(), np.zeros_like(trace["target_omega"]))
    predicted = trace["current_omega"]+trace["alpha"]*(trace["target_omega"]-trace["current_omega"])
    return jsonable(dict(format="plate-velocity-budget-1", evaluation=evaluation, time_myr=float(state.time_myr),
        plate_count=n, cell_count=mesh.cell_count, weighting="cell area; boundary edge length; inverse-CDF quantiles",
        units="omega rad/Myr, surface speed km/Myr = mm/yr; raw boundary drives km * dimensionless proxy",
        note="Magnitudes are not additive. Raw vectors in trace reconstruct the target exactly. Current and returned are different step endpoints.",
        mantle=dict(local_velocity_km_per_myr=weighted_stats(local_speed, areas),
            rms_omega_rad_per_myr=rms, formation_rms_omega_rad_per_myr=float(mantle.formation_rms_rad_per_myr),
            realised_thermal_amplitude_fraction=rms/max(mantle.formation_rms_rad_per_myr, 1e-30),
            rigid_fit_relative_residual=float(np.sqrt(residual_energy/local_energy)) if local_energy else 0.,
            represented_kinetic_proxy_fraction=float(1.-residual_energy/local_energy) if local_energy else 0.),
        stages_speed_km_per_myr=global_stages, contributions_speed_km_per_myr=global_contributions,
        plates=plate_rows, boundaries=boundary_budget(mesh, boundaries, radius), slabs=zones,
        slab_zero_reason=("No convergence-integrated slab geometry exists" if not zones else
            "Inspect stored length, activity, breakoff and per-edge multiplier in trace"),
        verification=dict(target_vector_sum_max_error=float(np.max(np.abs(vector_sum-trace["target_omega"]))),
            relaxation_prediction_max_error=float(np.max(np.abs(predicted-trace["relaxed_omega"]))),
            net_rotation_relative_velocity_max_change_km_per_myr=max(changes, default=0.)), trace=trace))


def diagnose_checkpoint(path, *, step_myr=1., projection=None):
    """Probe saved forces with frozen heat/mantle/material; leave disk untouched."""
    from .genesis_starter_continuation import _load_cp, load_starter_source
    from .genesis_starter_slab import young_slab_pull
    from .simulation import load_config
    from .rollback import RollbackParameters, advance_rollback
    from .subduction_memory import SubductionMemoryParameters
    root = Path(path).resolve()
    if root.name == "mature_checkpoint":
        root = root.parent
    saved = json.loads((root/"continuation.json").read_text(encoding="utf-8"))
    for relative, expected in saved["checkpoint_sha256"].items():
        if hashlib.sha256((root/relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Checkpoint integrity mismatch: {relative}")
    cfg = load_config(root/"mature_config.yaml")
    model, young, metadata = load_starter_source(root/"young_context/starter_checkpoint.npz")
    cp = _load_cp(root/"mature_checkpoint", cfg)
    from .genesis_young_mechanics import corrected_mechanics, transmitted_mantle_flow
    flow = cp.mantle_flow
    source_flow = flow
    if corrected_mechanics(cfg):
        from .genesis_starter_continuation import YoungWorldCoupling
        fracture = None
        if cfg["plate_dynamics"].get("young_slab_force_model") == "viscous_sinking_v1":
            from .genesis_starter_fracture import YoungShellFracture
            fracture = YoungShellFracture.load(model, root/"young_context/fracture_memory.npz")
        coupling = YoungWorldCoupling(model, young, cfg, cp, fracture=fracture)
        material = deepcopy(cp.state)
        coupling.mechanical_fields(material)
        local = getattr(coupling, "local_mechanical_diagnostics", None)
        thickness = (local["local_total_lid_thickness_km"] if local is not None
            else material.mantle_lithosphere_thickness_km + material.crust_thickness_km)
        flow = transmitted_mantle_flow(model, source_flow, thickness)
        cp.state = material
    params = dynamics.DynamicsParameters(**cfg["plate_dynamics"])
    if corrected_mechanics(cfg):
        params = coupling.dynamics_parameters(params)
    params = replace(params, force_speed_scale_deg_per_myr=params.force_speed_scale_deg_per_myr*cp.thermal.tectonic_activity_factor)
    if projection is not None:
        params = replace(params, mantle_projection=projection)
    subp = SubductionMemoryParameters(**cfg["subduction_memory"])
    # Rollback's geometry is evaluated on copies; its distance ledger is discarded.
    rb, _, _ = advance_rollback(model.mesh, deepcopy(cp.state), deepcopy(cp.subduction_memory),
        model.thermal.radius_km, step_myr, RollbackParameters(**cfg["rollback"]))
    trace = {}
    force_options = ({"young_slab_strength_pa": coupling.fracture.memory.strength_pa.copy()}
        if params.young_slab_force_model == "viscous_sinking_v1" else {})
    with young_slab_pull(subp):
        _, _, boundaries, _ = dynamics.update_plate_dynamics(model.mesh, cp.state, cp.system, cp.baseline,
            model.thermal.radius_km, step_myr, **cfg["classification"], params=params,
            mantle_flow=flow, thermal_lithosphere_thickness_km=cp.thermal.thermal_lithosphere_thickness_km,
            subduction_memory=cp.subduction_memory, subduction_memory_params=subp,
            rollback_omega_rad_per_myr=rb, trace=trace, **force_options)
    report = velocity_budget(model.mesh, cp.state, cp.system, flow, model.thermal.radius_km,
        trace, boundaries, subduction_memory=cp.subduction_memory,
        slab_reference_length_km=subp.slab_length_cap_km, evaluation="frozen_checkpoint_force_probe")
    report.update(source=str(root), step_myr=step_myr,
        controls=dict(shell=metadata["shell"], starter=metadata["starter"], classification=cfg["classification"],
                      dynamics=asdict(params), mantle_flow=cfg["mantle_flow"]),
        thermal=asdict(cp.thermal), ledger_checks=saved["checks"],
        transport_commits=cp.transport_state.cumulative_commit_count,
        thermal_budget=model.loading.sample(young.thermal_context).thermal)
    # This is an audit of the missing feedback, not a new force in the solver.
    from .genesis import SECONDS_PER_MYR
    owner, x = cp.state.cell_plate, model.mesh.centroids
    local = np.cross(flow.cell_omega_rad_per_myr, x)*model.thermal.radius_km
    actual = np.cross(trace["current_omega"][owner], x)*model.thermal.radius_km
    shear = np.linalg.norm(local-actual, axis=1)*1000./SECONDS_PER_MYR*model.parameters.basal_drag_pa_s_m
    report["residual_basal_shear_diagnostic_pa"] = weighted_stats(shear, model.areas)
    report["residual_basal_shear_note"] = "beta*(u_m-u_plate), diagnostic only; not wired into evolved young-shell stress"
    if corrected_mechanics(cfg):
        report["force_model"] = params.force_model
        report["uncoupled_prescribed_source_speed_km_per_myr"] = weighted_stats(
            np.linalg.norm(np.cross(source_flow.cell_omega_rad_per_myr, x)*model.thermal.radius_km, axis=1), model.areas)
        report["si_force_budget"] = {key: trace[key] for key in (
            "basal_driving_torque_nm", "ridge_torque_nm", "slab_torque_nm",
            "target_torque_residual_nm", "transient_torque_residual_nm",
            "basal_drag_dissipation_w", "slab_bending_dissipation_w", "slab_mantle_dissipation_w",
            "slab_constraint_reaction_torque_nm", "slab_constraint_power_w",
            "total_dissipation_w", "total_source_power_w", "transient_net_power_w")}
        report["residual_basal_shear_note"] = (
            "beta*(transmitted_traction/beta-u_plate); used by SI rigid torque balance, "
            "not yet coupled to the transported Maxwell fracture stress")
    return jsonable(report)
