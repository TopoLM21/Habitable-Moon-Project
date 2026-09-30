"""SI basal torque balance with explicitly resolved young-world forces.

This mode uses prescribed transmitted traction and independent linear basal
drag. It does not reinterpret the calibrated mature speed proxies as forces.
The finite relaxation is an explicit temporal filter, not a torque equilibrium
claim about the returned transient velocity. The fixed prescribed-source frame
is retained; subtracting only the plate mean would change physical basal slip.
"""
from __future__ import annotations

import math
import numpy as np

from .genesis import SECONDS_PER_MYR
from .kinematics import classify_boundaries, BoundaryType
from .lithosphere import CrustType
from .mantle import mantle_flow_rms_rad_per_myr


def basal_torque_system(mesh, owner, count, radius_km, flow, beta):
    """Return drag tensors [N m s], driving moments [N m], and cell u [m/s]."""
    if not math.isfinite(beta) or beta <= 0 or not math.isfinite(radius_km) or radius_km <= 0:
        raise ValueError("SI dynamics requires positive finite radius and basal drag")
    x = np.asarray(mesh.centroids)
    area = mesh.physical_cell_areas_km2(radius_km) * 1e6
    r = radius_km * 1000.
    field = np.asarray(flow.cell_omega_rad_per_myr)
    if field.shape != x.shape or not np.isfinite(field).all():
        raise ValueError("SI dynamics requires a finite transmitted mantle field")
    u = np.cross(field, x) * r / SECONDS_PER_MYR
    drag = np.zeros((count, 3, 3))
    torque = np.zeros((count, 3))
    rhs = np.cross(x, u)
    for a in range(3):
        torque[:, a] = np.bincount(owner, weights=beta*r*area*rhs[:, a], minlength=count)
        for b in range(3):
            drag[:, a, b] = np.bincount(owner,
                weights=beta*r*r*area*((a == b)-x[:, a]*x[:, b]), minlength=count)
    return drag, torque, u


def ridge_torques_nm(mesh, state, boundaries, radius_km, gravity, count):
    """Thin-layer hydrostatic contrast between a boundary axis and its flank.

    F/length = g/2 * (mean_flank(drho H^2) - axis(drho H^2)). Only actual
    younger oceanic material at the boundary can be a ridge axis. This passive
    flank approximation is explicit; a spatial pressure solution is not claimed.
    No display-speed gate, positive floor or plate-average force without a
    contrast is used. Uniform fields give exactly zero.
    """
    result = np.zeros((count, 3))
    length_active = 0.
    if state.mantle_lithosphere_thickness_km is None or state.mantle_lithosphere_density_anomaly_kg_m3 is None:
        return result, length_active
    h = np.maximum(state.mantle_lithosphere_thickness_km, 0.) * 1000.
    rho = np.maximum(state.mantle_lithosphere_density_anomaly_kg_m3, 0.)
    proxy = rho * h*h
    ocean = np.asarray(state.crust_type) == int(CrustType.OCEANIC)
    areas = mesh.physical_cell_areas_km2(radius_km) * ocean
    owner = np.asarray(state.cell_plate)
    sums = np.bincount(owner, weights=areas, minlength=count)
    means = np.divide(np.bincount(owner, weights=areas*proxy, minlength=count), sums,
                      out=np.zeros(count), where=sums > 0)
    ages = np.asarray(state.crust_age_myr)
    mean_age = np.divide(np.bincount(owner, weights=areas*ages, minlength=count), sums,
                        out=np.zeros(count), where=sums > 0)
    from .dynamics import _normal_ab, _boundary_length_km
    for edge in boundaries:
        normal = _normal_ab(mesh, edge)
        length = _boundary_length_km(mesh, edge, radius_km)
        active = False
        for face, pid, sign in ((edge.face_a, edge.plate_a, -1.), (edge.face_b, edge.plate_b, 1.)):
            if not ocean[face] or ages[face] >= mean_age[pid] - 1e-10:
                continue
            contrast = max(0., means[pid] - proxy[face])
            # Eliminate summation roundoff for uniform GPE fields.
            if contrast <= 1e-12*max(abs(means[pid]), abs(proxy[face]), 1.):
                continue
            force = .5*gravity*contrast * length*1000.
            result[pid] += radius_km*1000. * force * np.cross(edge.midpoint, sign*normal)
            active = True
        if active:
            length_active += length
    return result, length_active


def update_young_plate_dynamics(mesh, state, current_system, baseline_system,
        radius_km, dt_myr, normal_threshold_km_per_myr, inactive_speed_km_per_myr,
        params, *, mantle_flow, subduction_memory=None, subduction_memory_params=None,
        young_slab_strength_pa=None, trace=None):
    from .dynamics import (DynamicsDiagnostics, angular_velocity_vectors,
                           system_from_omega, _boundary_length_km)
    if not math.isfinite(dt_myr) or dt_myr <= 0:
        raise ValueError("dt_myr must be positive and finite")
    if params.young_velocity_response_model not in ("relaxed", "quasistatic"):
        raise ValueError("Unknown young velocity response model")
    if (params.young_velocity_response_model == "relaxed"
            and (not math.isfinite(params.velocity_relaxation_myr) or params.velocity_relaxation_myr <= 0)):
        raise ValueError("SI velocity relaxation must be positive and finite")
    if mantle_flow is None:
        raise ValueError("Young SI mechanics requires an explicit basal source")
    if not math.isfinite(params.gravity_m_s2) or params.gravity_m_s2 <= 0:
        raise ValueError("SI dynamics requires positive finite gravity")
    count = len(current_system.plates)
    owner = np.asarray(state.cell_plate)
    boundaries = classify_boundaries(mesh, system_from_omega(owner, current_system,
        angular_velocity_vectors(current_system)), radius_km,
        normal_threshold_km_per_myr, inactive_speed_km_per_myr)
    drag, basal, u = basal_torque_system(mesh, owner, count, radius_km, mantle_flow,
                                       params.basal_drag_pa_s_m)
    ridge, ridge_length = ridge_torques_nm(mesh, state, boundaries, radius_km,
                                         params.gravity_m_s2, count)
    slab = np.zeros_like(basal)
    if params.young_slab_force_model not in ("disabled_pending_closure", "full_transmission_upper_bound", "viscous_sinking_v1"):
        raise ValueError("Unknown young slab force closure")
    if (subduction_memory is not None
            and params.young_slab_force_model == "full_transmission_upper_bound"):
        from .young_boundary import accepted_slab_torques_nm
        slab = accepted_slab_torques_nm(subduction_memory, count, radius_km, params.gravity_m_s2)

    basal_global = np.zeros((3*count, 3*count))
    for pid in range(count):
        basal_global[3*pid:3*pid+3, 3*pid:3*pid+3] = drag[pid]
    bending_drag = np.zeros_like(basal_global)
    sinking_drag = np.zeros_like(basal_global)
    slab_sections = ()
    sinking = None
    constrained = None
    reaction = np.zeros_like(basal)
    neck_failures = []
    if params.young_slab_force_model == "viscous_sinking_v1":
        from .young_boundary import accepted_slab_force_sections
        from .young_slab_sinking import slab_sinking_system
        from .young_slab_constraints import solve_no_eduction
        if params.young_velocity_response_model != "quasistatic":
            raise ValueError("Sinking with finite attachment strength requires quasistatic velocities")
        strength = np.asarray(young_slab_strength_pa, dtype=float)
        if strength.shape != owner.shape or not np.isfinite(strength).all() or np.any(strength < 0):
            raise ValueError("Sinking requires the live finite nonnegative tensile strength field")
        sections = accepted_slab_force_sections(subduction_memory, params=subduction_memory_params,
            buoyancy_model=params.young_slab_buoyancy_model)
        cold_h = state.mantle_lithosphere_thickness_km
        incoming = None if cold_h is None else np.asarray(cold_h)
        for iteration in range(len(sections)+1):
            sinking = slab_sinking_system(sections, count, radius_km,
                params.young_slab_mantle_viscosity_pa_s, params.young_slab_mantle_depth_km,
                params, incoming_thickness_km=incoming)
            bending_drag, sinking_drag = sinking.drag_bending_nm_s, sinking.drag_mantle_nm_s
            slab, slab_sections = sinking.driving_torque_nm, sinking.sections
            cross_sections = np.array([s['trench_length_km']*s['cold_hinge_thickness_km']*1e6
                                       for s in slab_sections])
            constrained = solve_no_eduction(basal_global+bending_drag+sinking_drag,
                basal+ridge+slab, sinking.feed_matrix_m, reaction_weights=cross_sections)
            omega_si = constrained.omega_rad_s.ravel()
            source_faces = {(s.contact_key, s.subducting_plate, s.overriding_plate): s.source_face
                            for s in sections}
            failed = set()
            for i, section in enumerate(slab_sections):
                key = (section['contact_key'], section['subducting_plate'], section['overriding_plate'])
                q = constrained.feed_m_s[i]
                tension = (section['gravitational_feed_force_n']
                    -section['bending_coefficient_n_s_m']*q
                    -np.dot(section['mantle_feed_drag_row_n_s'], omega_si)
                    +constrained.normal_forces_n[i])
                capacity = strength[source_faces[key]]*cross_sections[i]
                section.update(feed_m_s=float(q), neck_tension_n=float(tension),
                    neck_capacity_n=float(capacity), neck_strength_pa=float(strength[source_faces[key]]),
                    no_eduction_reaction_n=float(constrained.normal_forces_n[i]))
                if tension > capacity*(1.+1e-10):
                    failed.add(key)
                    neck_failures.append(dict(contact_key=key[0], subducting_plate=key[1],
                        overriding_plate=key[2], tension_n=float(tension), capacity_n=float(capacity),
                        source_face=int(source_faces[key]), feed_m_s=float(q),
                        unconstrained_feed_m_s=float(constrained.unconstrained_feed_m_s[i]),
                        gravitational_feed_force_n=float(section['gravitational_feed_force_n']),
                        bending_feed_resistance_n=float(section['bending_coefficient_n_s_m']*q),
                        mantle_feed_resistance_n=float(np.dot(section['mantle_feed_drag_row_n_s'], omega_si)),
                        no_eduction_reaction_n=float(constrained.normal_forces_n[i]),
                        neck_strength_pa=float(strength[source_faces[key]]),
                        cold_hinge_thickness_km=float(section['cold_hinge_thickness_km']),
                        trench_length_km=float(section['trench_length_km']),
                        iteration=iteration))
            if not failed:
                reaction = constrained.reaction_torque_nm
                break
            sections = tuple(s for s in sections if
                (s.contact_key, s.subducting_plate, s.overriding_plate) not in failed)
        else:
            raise RuntimeError("Slab attachment failure iteration did not converge")
    total_drag = basal_global + bending_drag + sinking_drag

    def solve(torque):
        if params.young_slab_force_model == "viscous_sinking_v1":
            # Coupled plate pairs must be solved together: both the trench
            # velocity and the material feeding it enter slab dissipation.
            scale = float(np.linalg.norm(total_drag))
            result = np.linalg.solve(total_drag/scale, torque.ravel()/scale)
            return result.reshape(count, 3)*SECONDS_PER_MYR
        result = np.zeros_like(torque)
        for pid in range(count):
            scale = np.linalg.norm(drag[pid])
            if scale > 0:
                result[pid] = np.linalg.lstsq(drag[pid]/scale, torque[pid]/scale, rcond=None)[0]
        return result * SECONDS_PER_MYR

    mantle_omega, ridge_omega, slab_omega = (solve(t) for t in (basal, ridge, slab))
    reaction_omega = solve(reaction)
    target = (mantle_omega + ridge_omega + slab_omega if constrained is None
              else constrained.omega_rad_s*SECONDS_PER_MYR)
    current = angular_velocity_vectors(current_system)
    alpha = (1. if params.young_velocity_response_model == "quasistatic"
        else -math.expm1(-dt_myr/max(params.velocity_relaxation_myr, 1e-9)))
    final = current + alpha*(target-current)
    areas = mesh.physical_cell_areas_km2(radius_km)
    plate_areas = np.bincount(owner, weights=areas, minlength=count)
    mean_rotation = np.sum(final*plate_areas[:, None], axis=0)/plate_areas.sum()
    torque_total = basal + ridge + slab
    target_si, final_si = target.ravel()/SECONDS_PER_MYR, final.ravel()/SECONDS_PER_MYR
    residual_target = torque_total + reaction - (total_drag@target_si).reshape(count, 3)
    residual_transient = torque_total + reaction - (total_drag@final_si).reshape(count, 3)
    bending_dissipation = float(final_si@bending_drag@final_si)
    sinking_dissipation = float(final_si@sinking_drag@final_si)
    sinking_power = {} if sinking is None else sinking.power_diagnostics(final_si)
    if constrained is not None:
        sinking_power.update(no_eduction_active_count=int(np.sum(constrained.active)),
            unconstrained_negative_feed_section_count=int(np.sum(constrained.unconstrained_feed_m_s < 0.)),
            unconstrained_minimum_feed_m_s=float(np.min(constrained.unconstrained_feed_m_s, initial=0.)),
            constraint_relative_residual=constrained.relative_residual,
            constraint_complementarity_relative_error=constrained.complementarity_relative_error,
            neck_failures_this_evaluation=len(neck_failures))
    plate_u = np.cross(final[owner], mesh.centroids)*radius_km*1000./SECONDS_PER_MYR
    dissipation = params.basal_drag_pa_s_m*np.sum(plate_u*plate_u, axis=1)
    input_power = params.basal_drag_pa_s_m*np.sum(u*plate_u, axis=1)
    zero = np.zeros_like(target)
    if trace is not None:
        trace.clear()
        trace.update(force_model="young_si_v1", mantle_projection="velocity_least_squares",
            young_slab_force_model=params.young_slab_force_model,
            young_slab_buoyancy_model=params.young_slab_buoyancy_model,
            young_velocity_response_model=params.young_velocity_response_model,
            current_omega=current.copy(), baseline_omega=angular_velocity_vectors(baseline_system),
            mantle_omega=mantle_omega, common_mantle=mantle_omega.copy(),
            ridge_drive_raw=ridge_omega.copy(), slab_drive_raw=slab_omega.copy(),
            ridge_drive_normalized=ridge_omega.copy(), slab_drive_normalized=slab_omega.copy(),
            slab_drive_potential_raw=slab_omega.copy(), slab_edges=[], ridge_push_factors=np.zeros(count),
            boundary_drive_raw=ridge_omega+slab_omega, boundary_weight_km=np.zeros(count),
            boundary_drive_normalized=ridge_omega+slab_omega,
            residual_slab_drive=zero.copy(), total_drive_normalized=ridge_omega+slab_omega,
            gpe_drive_raw=zero.copy(), gpe_weight=np.zeros(count), gpe_drive_normalized=zero.copy(),
            gpe_component=zero.copy(), rollback_omega=zero.copy(), drive_scale_rad_per_myr=1.,
            relative_drive=ridge_omega+slab_omega, total_boundary_weight_km=np.zeros(count),
            collision_length_weighted_km=np.zeros(count), transform_length_weighted_km=np.zeros(count),
            collision_ratio=np.zeros(count), transform_ratio=np.zeros(count), drag_factor=np.ones(count),
            resistance_omega=zero.copy(), collision_resistance_omega=zero.copy(), transform_resistance_omega=zero.copy(),
            target_omega=target, alpha=alpha, relaxed_omega=final.copy(), post_gauge_omega=final.copy(),
            final_omega=final.copy(), plate_areas_km2=plate_areas, mean_rotation=mean_rotation,
            remove_net_rotation=False, frame="fixed_prescribed_source",
            basal_drag_tensor_nm_s=drag, basal_driving_torque_nm=basal,
            total_drag_tensor_nm_s=total_drag,
            slab_bending_drag_tensor_nm_s=bending_drag,
            slab_mantle_drag_tensor_nm_s=sinking_drag,
            slab_sections=slab_sections,
            slab_sinking_diagnostics=sinking_power,
            slab_neck_failures=neck_failures,
            slab_constraint_omega=reaction_omega,
            slab_constraint_reaction_torque_nm=reaction,
            slab_constraint_power_w=float(np.sum(reaction*final/SECONDS_PER_MYR)),
            slab_bending_resistance_torque_nm=-(bending_drag@final_si).reshape(count, 3),
            slab_mantle_resistance_torque_nm=-(sinking_drag@final_si).reshape(count, 3),
            ridge_torque_nm=ridge, slab_torque_nm=slab, target_torque_residual_nm=residual_target,
            transient_torque_residual_nm=residual_transient,
            basal_drag_dissipation_w=float(areas@(dissipation*1e6)),
            slab_bending_dissipation_w=bending_dissipation,
            slab_mantle_dissipation_w=sinking_dissipation,
            total_dissipation_w=float(areas@(dissipation*1e6))+bending_dissipation+sinking_dissipation,
            basal_source_power_w=float(areas@(input_power*1e6)),
            ridge_power_w=float(np.sum(ridge*final/SECONDS_PER_MYR)),
            slab_power_w=float(np.sum(slab*final/SECONDS_PER_MYR)),
            total_source_power_w=float(np.sum(torque_total*final/SECONDS_PER_MYR)),
            transient_net_power_w=float(np.sum(residual_transient*final/SECONDS_PER_MYR)),
            force_trace_units="SI torque fields; component drives are rad/Myr")
    speeds = np.linalg.norm(final, axis=1)
    oldspeeds = np.linalg.norm(current, axis=1)
    turns = np.zeros(count)
    mask = (speeds > 1e-14) & (oldspeeds > 1e-14)
    turns[mask] = np.rad2deg(np.arccos(np.clip(np.sum(final[mask]*current[mask], axis=1)/(speeds[mask]*oldspeeds[mask]), -1., 1.)))
    lengths = {kind: sum(_boundary_length_km(mesh,b,radius_km) for b in boundaries if b.boundary_type==kind) for kind in BoundaryType}
    diag = DynamicsDiagnostics(float(state.time_myr), float(np.rad2deg(speeds).mean()),
        float(np.rad2deg(speeds).max()), float(turns.mean()), float(turns.max()),
        ridge_length, lengths[BoundaryType.CONVERGENT], 0., lengths[BoundaryType.TRANSFORM],
        1., float(np.rad2deg(np.linalg.norm(mean_rotation))),
        mantle_rms_speed_deg_per_myr=float(np.rad2deg(mantle_flow_rms_rad_per_myr(mantle_flow))),
        mean_plate_mantle_slip_deg_per_myr=float(np.rad2deg(np.linalg.norm(final-mantle_omega,axis=1)).mean()),
        mean_ridge_push_factor=0., min_ridge_push_factor=0., max_ridge_push_factor=0.)
    return system_from_omega(owner, current_system, final), diag, boundaries, ridge_omega+slab_omega
