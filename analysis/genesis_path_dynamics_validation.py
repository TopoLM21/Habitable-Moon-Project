"""Real-source tied insertion and explicitly prescribed cohesive path motion.

Tied controls share the existing column, thermal, orbit and Maxwell predictor.
Released controls deliberately freeze temperature/phase/orbit and advance only
mechanical elapsed years. They do not predict nucleation or geological growth.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, fields, is_dataclass, replace
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tectonics.genesis_contact_growth import empty_cohorts
from tectonics.genesis_coupled import CoupledModel, CoupledState
from tectonics.genesis_coupled_thermal import advance_thermal_loading
from tectonics.genesis_crack_path import CrackInterval, ReferenceCrackPath
from tectonics.genesis_mobile import _external_force
from tectonics.genesis_path_basis import EmbeddedPathBasis
from tectonics.genesis_path_dynamics import PathLoading, PathMechanics
from tectonics.genesis_path_mesh import insert_crack_path
from tectonics.genesis_shell import Membrane, mantle_traction

BASE = ROOT/"results/genesis_runs"
SOURCE = BASE/"fine_fault_audit_20260924/strong_5120/fault_checkpoint.npz"
TRACE = BASE/"ridge_tracking_20260924/source_trace_verified/trace_5120_cumulative_shear_divided_by_0p1.npz"


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _relative(first, second):
    first, second = np.asarray(first), np.asarray(second)
    return float(np.linalg.norm(first-second)/max(np.linalg.norm(first), np.linalg.norm(second), 1e-300))


def _array_hash(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256(str((value.dtype.str, value.shape)).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _exact(first, second):
    if is_dataclass(first):
        return all(_exact(getattr(first, item.name), getattr(second, item.name)) for item in fields(first))
    if isinstance(first, np.ndarray):
        return np.array_equal(first, second)
    return first == second


def _empty_state(coupled):
    """A diagnostic empty-cut state, without invoking legacy seam selection."""
    source = coupled.source.source_state
    return CoupledState(source.time_myr, coupled.source.initial(), empty_cohorts(),
        np.empty((0, 2), dtype=np.int64), np.empty(0), np.empty(0),
        source.column_enthalpy.copy(), source.elastic_strain.copy(), source.damage.copy(),
        source.water_access.copy(), source.fault_active.copy(), source.plane_normal.copy(),
        source.fault_candidate_age_myr.copy(), np.zeros_like(source.velocity_km_myr),
        source.tidal_stress_mpa.copy(), source.boundary_energy_j, source.tidal_heat_received_j,
        source.initial_column_energy_j)


def _external(coupled, basis, depth_km):
    """Old background work plus fine loads split by the same bank drag areas."""
    p = coupled.source_model.p
    parent = coupled._external(coupled.source, depth_km)
    fine_mesh = basis.subdivision.mesh
    fine_membrane = Membrane(fine_mesh, p.poisson_ratio)
    traction = mantle_traction(fine_mesh, p, basis.subdivision.intensive(depth_km))
    force = _external_force(fine_membrane, traction, basis.radius_m/1000., p.young_modulus_pa)
    force *= p.young_modulus_pa*basis.radius_m*1000.
    area = basis.fine_drag_area_m2[:-1:2]
    parent_vertex = basis.topology.parent_vertex
    summed = np.bincount(parent_vertex, weights=area, minlength=fine_mesh.vertex_count)
    split_force = np.zeros(basis.membrane.ndof)
    split_force[:-1] = (force[:-1].reshape(-1, 2)[parent_vertex]
        *(area/summed[parent_vertex])[:, None]).ravel()
    split_force[-1] = force[-1]
    return parent, basis.external(parent, split_force)


def _row(model, state, release_year=None):
    values = model.geometry_metrics(state)
    values.update(elapsed_years=state.elapsed_years,
        years_since_prescribed_release=None if release_year is None else state.elapsed_years-release_year,
        equilibrium_residual=state.equilibrium_residual,
        constraint_reaction_norm_n=float(np.linalg.norm(state.constraint_reaction_n)),
        released_reaction_norm_n=state.released_reaction_norm_n,
        released_reaction_measured=getattr(state, "released_reaction_measured", state.accepted_steps > 0),
        drag_work_j=state.drag_work_j, external_work_j=state.external_work_j,
        bulk_work_j=state.bulk_work_j, bulk_loading_correction_j=state.bulk_loading_correction_j,
        mechanical_remainder_j=state.mechanical_remainder_j,
        max_cohesive_damage=float(np.max(state.cohorts.damage, initial=0)),
        damaged_cohort_count=int(np.count_nonzero(state.cohorts.damage > 0)),
        fully_broken_cohort_count=int(np.count_nonzero(state.cohorts.damage >= 1)),
        friction_work_j=float(state.cohorts.friction_work_j.sum()),
        viscous_work_j=float(state.cohorts.viscous_work_j.sum()),
        fracture_work_j=float(state.cohorts.fracture_work_j.sum()),
        accepted_steps=state.accepted_steps, rejected_steps=state.rejected_steps,
        last_step_years=state.last_step_years, stopped_reason=state.stopped_reason)
    return values


def tied_control(coupled, model, output):
    basis, projection = model.basis, model.basis.subdivision
    state = _empty_state(coupled)
    path_state = model.initial(projection.tensor(state.elastic_strain, engineering=True))
    thermal, orbit = coupled.source.thermal_state, coupled.source.orbit
    p = coupled.source_model.p
    rows = []
    started = perf_counter()
    for requested_years in (1., 7., 13.):
        mesh, radius, _, _, _, _ = coupled._phase_fields(state, thermal)
        loading = advance_thermal_loading(coupled.source_model, mesh=mesh, radius_km=radius,
            layer_mass_kg=coupled.layer_mass_kg, column_enthalpy=state.column_enthalpy,
            elastic_strain=state.elastic_strain, damage=state.damage, water_access=state.water_access,
            boundary_energy_j=state.boundary_energy_j, thermal_state=thermal, orbit=orbit,
            target_myr=thermal.time_myr+requested_years/1e6)
        volume = coupled.layer_mass_kg.sum(axis=1)*loading.fraction/p.density_kg_m3
        degradation = p.residual_stiffness+(1-p.residual_stiffness)*(1-loading.damage0)**2
        elasticity = (p.young_modulus_pa*degradation)[:, None, None]*coupled.original_membrane.d
        parent_external, external = _external(coupled, basis, loading.depth_km)
        path_loading = PathLoading(loading.dt_myr*1e6, projection.extensive(volume),
            projection.intensive(elasticity), projection.tensor(loading.memory, engineering=True),
            projection.intensive(loading.effective_b), external,
            projection.intensive(loading.water_access))
        history, elastic, _, work, loss, cohorts = coupled._solve(state, loading, loading.damage0)
        before_path = path_state
        path_state = model.trial(path_state, path_loading)
        projected_elastic = projection.tensor(elastic, engineering=True)
        row = _row(model, path_state)
        column_heat = float(np.sum(coupled.layer_mass_kg*loading.column_enthalpy))
        row.update(requested_step_years=requested_years, actual_step_years=loading.dt_myr*1e6,
            thermal_time_myr=loading.thermal_state.time_myr, orbit_time_myr=loading.orbit.time_myr,
            parent_displacement_relative_error=_relative(path_state.displacement_m[:basis.nparent], history.displacement_m),
            parent_displacement_max_absolute_error_m=float(np.max(np.abs(path_state.displacement_m[:basis.nparent]-history.displacement_m))),
            elastic_relative_error=_relative(path_state.elastic_strain, projected_elastic),
            elastic_max_absolute_error=float(np.max(np.abs(path_state.elastic_strain-projected_elastic))),
            drag_work_relative_error=_relative(path_state.drag_work_j, history.drag_work_j),
            external_work_relative_error=_relative(path_state.external_work_j, history.external_work_j),
            increment_bulk_work_relative_error=_relative(path_state.bulk_work_j-before_path.bulk_work_j, work),
            increment_loading_correction_relative_error=_relative(path_state.bulk_loading_correction_j-before_path.bulk_loading_correction_j, loss),
            clock_difference_years=abs(path_state.elapsed_years-(loading.thermal_state.time_myr-coupled.source_time_myr)*1e6),
            parent_equilibrium_residual=history.equilibrium_residual,
            max_locked_enrichment_m=float(np.max(np.abs(path_state.displacement_m[basis.nparent:]), initial=0)),
            cohort_traction_max_pa=float(np.max(np.abs(path_state.cohorts.traction_pa), initial=0)),
            parent_force_norm_n=float(np.linalg.norm(parent_external)),
            relative_force_norm_n=float(np.linalg.norm(external[basis.nparent:])),
            column_heat_relative_residual=(column_heat-state.initial_column_energy_j-loading.boundary_energy_j)/max(abs(state.initial_column_energy_j), 1.))
        rows.append(row)
        state = replace(state, time_myr=loading.thermal_state.time_myr, contact=history,
            cohorts=cohorts, column_enthalpy=loading.column_enthalpy, elastic_strain=elastic,
            damage=loading.damage0, water_access=loading.water_access,
            boundary_energy_j=loading.boundary_energy_j)
        thermal, orbit = loading.thermal_state, loading.orbit
    model.save_state(output/"tied_mechanics_checkpoint.npz", path_state)
    np.savez_compressed(output/"tied_loading_context.npz", parent_displacement_m=state.contact.displacement_m,
        parent_elastic_strain=state.elastic_strain, column_enthalpy=state.column_enthalpy,
        layer_mass_kg=coupled.layer_mass_kg, damage=state.damage, water_access=state.water_access,
        metadata=np.array(json.dumps({"format": "diagnostic_loading_context_not_coupled_checkpoint",
            "thermal": asdict(thermal), "orbit": asdict(orbit)}, default=lambda x: x.tolist(), allow_nan=False)))
    return {"scope": "three_existing_thermal_orbit_predictors_and_mechanical_solves_no_new_damage_or_cut_selection",
        "wall_seconds": perf_counter()-started, "rows": rows,
        "checks": {"same_background": all(r["parent_displacement_relative_error"] < 2e-8 for r in rows),
            "same_elastic_memory": all(r["elastic_relative_error"] < 2e-8 for r in rows),
            "same_drag_work": all(r["drag_work_relative_error"] < 2e-8 for r in rows),
            "same_external_work": all(r["external_work_relative_error"] < 2e-8 for r in rows),
            "same_clocks": all(r["thermal_time_myr"] == r["orbit_time_myr"] and r["clock_difference_years"] < 1e-8 for r in rows),
            "locked_contact_inactive": all(r["max_locked_enrichment_m"] == r["cohort_traction_max_pa"] == r["max_cohesive_damage"] == r["fracture_work_j"] == 0 for r in rows)}}


def released_control(coupled, model, output):
    basis, projection = model.basis, model.basis.subdivision
    source, p = coupled.source.source_state, coupled.source_model.p
    initial = model.initial(projection.tensor(source.elastic_strain, engineering=True))
    degradation = p.residual_stiffness+(1-p.residual_stiffness)*(1-source.damage)**2
    elasticity = projection.intensive((p.young_modulus_pa*degradation)[:, None, None]*coupled.original_membrane.d)
    temp = coupled.source.source_fields["temperature_k"]
    eta = np.clip(p.viscosity_reference_pa_s*np.exp(np.clip(p.activation_energy_j_mol/8.314462618
        *(1/np.maximum(temp, 1)-1/p.viscosity_reference_temperature_k), -60, 60)),
        p.viscosity_min_pa_s, p.viscosity_max_pa_s)
    viscosity = projection.intensive(eta)
    water = projection.intensive(source.water_access)
    _, external = _external(coupled, basis, coupled.source.depth_m/1000.)
    volume = model.reference_volume_m3
    def factory(state, dt_years):
        return model.isothermal_loading(state, dt_years, volume, elasticity,
            viscosity, p.young_modulus_pa, external, water)
    started = perf_counter()
    tied = model.advance(initial, 1., factory, max_step_years=1.)
    if tied.stopped_reason:
        return {"scope": "isothermal_fixed_phase_mechanics_only", "tied_initialization": _row(model, tied),
                "wall_seconds": perf_counter()-started, "release_performed": False}
    path_length = basis.insertion.path.length_m
    interval = CrackInterval(.25*path_length, .75*path_length)
    release = model.release(tied, interval)
    source_clock = source.time_myr
    milestones, histories = [], {}
    state = release
    for years in (1., 10., 100.):
        state = model.advance(state, release.elapsed_years+years, factory, max_step_years=10.)
        row = _row(model, state, release.elapsed_years)
        row.update(requested_years_since_release=years,
            thermal_time_myr=source_clock, orbit_time_myr=source_clock)
        milestones.append(row)
        if state.stopped_reason:
            break
    model.save_state(output/"released_mechanics_checkpoint.npz", state)
    np.savez_compressed(output/"released_loading_context.npz", volume_m3=volume,
        elasticity=elasticity, viscosity_pa_s=viscosity, external_force_n=external,
        water_access=water, metadata=np.array(json.dumps({"format": "fixed_isothermal_mechanics_loading",
            "source_sha256": _hash(SOURCE), "source_time_myr": source_clock,
            "global_thermal_orbit_clocks_frozen": True, "young_modulus_pa": p.young_modulus_pa}, allow_nan=False)))
    convergence_states, convergence = [], []
    for maximum in (10., 5., 2.5):
        current = release
        sequence = [_row(model, current, release.elapsed_years)]
        for end in (10., 20.):
            current = model.advance(current, release.elapsed_years+end, factory, max_step_years=maximum)
            sequence.append(_row(model, current, release.elapsed_years))
            if current.stopped_reason:
                break
        convergence_states.append(current)
        histories[str(maximum)] = sequence
        convergence.append({"max_step_years": maximum, **sequence[-1]})
    finest = convergence_states[-1]
    for row, current in zip(convergence, convergence_states):
        row["same_final_time_as_finest"] = current.elapsed_years == finest.elapsed_years
        row["displacement_relative_to_finest"] = _relative(current.displacement_m, finest.displacement_m)
        row["enrichment_relative_to_finest"] = _relative(current.displacement_m[basis.nparent:], finest.displacement_m[basis.nparent:])
        row["elastic_relative_to_finest"] = _relative(current.elastic_strain, finest.elastic_strain)
        row["friction_work_relative_to_finest"] = _relative(current.cohorts.friction_work_j, finest.cohorts.friction_work_j)
        row["fracture_work_relative_to_finest"] = _relative(current.cohorts.fracture_work_j, finest.cohorts.fracture_work_j)
    halfway = model.advance(release, release.elapsed_years+10., factory, max_step_years=5.)
    model.save_state(output/"restart_halfway.npz", halfway)
    restored = model.load_state(output/"restart_halfway.npz")
    uninterrupted = model.advance(halfway, release.elapsed_years+20., factory, max_step_years=5.)
    resumed = model.advance(restored, release.elapsed_years+20., factory, max_step_years=5.)
    model.save_state(output/"restart_resumed.npz", resumed)
    final_jump = (model.jump_operator@state.displacement_m).reshape(-1, 2)
    np.savez_compressed(output/"released_fields.npz", reference_vertices=basis.topology.mesh.vertices,
        faces=basis.topology.mesh.faces, parent_face=projection.parent_face,
        path_arclength_m=basis.insertion.path_arclength_m,
        path_vertex_ids=basis.insertion.path_vertex_ids, cut_edges=basis.topology.cut_edges,
        fine_displacement_m=basis.displacement_operator@state.displacement_m,
        constitutive_strain=basis.strain(state.displacement_m),
        geometric_strain=basis.geometric_strain(state.displacement_m), jump_m=final_jump,
        normal_traction_pa=state.cohorts.traction_pa[:, 0], shear_traction_pa=state.cohorts.traction_pa[:, 1],
        interface_damage=state.cohorts.damage)
    return {"scope": "prescribed_cohesive_release_not_nucleation_fixed_temperature_phase_damage_water_orbit",
        "release_performed": True, "wall_seconds": perf_counter()-started,
        "source_global_time_myr": source_clock, "mechanical_clock_is_elapsed_years_only": True,
        "prescribed_interval_m": asdict(interval), "prescribed_length_km": (interval.right_m-interval.left_m)/1000.,
        "min_maxwell_time_years": float(np.min(eta)/p.young_modulus_pa/(365.25*86400.)),
        "max_maxwell_time_years": float(np.max(eta)/p.young_modulus_pa/(365.25*86400.)),
        "tied_initialization": _row(model, tied), "release": _row(model, release, release.elapsed_years),
        "milestones": milestones, "timestep_comparison": convergence, "timestep_histories": histories,
        "restart": {"checkpoint_roundtrip_exact": _exact(halfway, restored),
            "continued_state_exact": _exact(uninterrupted, resumed),
            "halfway_elapsed_years": halfway.elapsed_years, "final_elapsed_years": resumed.elapsed_years,
            "stopped_reason": resumed.stopped_reason},
        "checks": {"release_does_not_reset_bulk_or_motion": np.array_equal(release.displacement_m, tied.displacement_m)
                and np.array_equal(release.elastic_strain, tied.elastic_strain),
            "release_does_not_add_work": release.external_work_j == tied.external_work_j
                and release.drag_work_j == tied.drag_work_j,
            "reaction_release_reported": release.released_reaction_norm_n > 0,
            "restart_exact": _exact(halfway, restored) and _exact(uninterrupted, resumed),
            "finite_final_fields": bool(np.isfinite(state.displacement_m).all() and np.isfinite(state.elastic_strain).all())}}


def plot_result(report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), constrained_layout=True)
    tied = report["tied_control"]["rows"]
    t = [r["elapsed_years"] for r in tied]
    for key, label in (("parent_displacement_relative_error", "parent displacement"),
                       ("elastic_relative_error", "elastic strain"),
                       ("drag_work_relative_error", "drag work")):
        axes[0, 0].semilogy(t, [max(r[key], 1e-18) for r in tied], "o-", label=label)
    axes[0, 0].set(xlabel="Shared elapsed physical years", ylabel="Relative error", title="Tied support vs existing mechanics")
    axes[0, 0].legend(fontsize=8)
    released = report["released_control"]
    if released.get("release_performed"):
        rows = released["milestones"]
        t = [r["years_since_prescribed_release"] for r in rows]
        axes[0, 1].plot(t, [r["max_opening_m"] for r in rows], "o-", label="normal opening")
        axes[0, 1].plot(t, [r["max_slip_m"] for r in rows], "s-", label="tangential jump")
        axes[0, 1].set(xlabel="Isothermal mechanical years after release", ylabel="Metres", title="Prescribed 25–75% cohesive interval")
        axes[0, 1].legend(fontsize=8)
        for key, limit, label in (("constitutive_added_strain", .005, "assumed strain / limit"),
                                 ("geometric_added_strain", .005, "compatible strain / limit"),
                                 ("motion_edge_fraction", .02, "motion / limit")):
            axes[1, 0].semilogy(t, [max(r[key]/limit, 1e-12) for r in rows], "o-", label=label)
        axes[1, 0].axhline(1., color="black", linestyle=":", label="existing guard")
        stop = rows[-1]["stopped_reason"]
        subtitle = "Reference geometry checks"
        if stop:
            subtitle = f"Stopped at {t[-1]:.3f} yr: {stop.removeprefix('path_').replace('_', ' ')}"
        axes[1, 0].set(xlabel="Isothermal mechanical years after release", ylabel="Fraction of existing limit", title=subtitle)
        axes[1, 0].legend(fontsize=8)
        for maximum, rows in released["timestep_histories"].items():
            axes[1, 1].plot([r["years_since_prescribed_release"] for r in rows],
                [r["max_opening_m"] for r in rows], "o-", label=f"max step {maximum} yr")
        axes[1, 1].set(xlabel="Isothermal mechanical years after release", ylabel="Maximum opening, m", title="Admissibility-controlled timestep comparison")
        axes[1, 1].legend(fontsize=8)
    for axis in axes.ravel():
        axis.grid(alpha=.25)
    fig.suptitle("5120-cell source + 1532 km prescribed support: dynamics validation\nReleased experiment freezes thermal/orbit evolution; this is not natural crack growth", fontsize=12)
    fig.savefig(output/"validation.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Validation output must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    code_paths = [Path(__file__), ROOT/"tectonics/genesis_path_basis.py", ROOT/"tectonics/genesis_path_dynamics.py",
        ROOT/"tectonics/genesis_path_mesh.py", ROOT/"tectonics/genesis_path_material.py",
        ROOT/"tectonics/genesis_contact_geometry.py", ROOT/"tectonics/genesis_contact_growth.py",
        ROOT/"tectonics/genesis_contact_law.py", ROOT/"tectonics/genesis_coupled.py",
        ROOT/"tectonics/genesis_coupled_thermal.py"]
    code_hashes = {str(p.relative_to(ROOT)): _hash(p) for p in code_paths}
    source_hashes = {str(path.relative_to(ROOT)): _hash(path) for path in (SOURCE, TRACE)}
    coupled = CoupledModel(SOURCE.read_bytes(), source_path=str(SOURCE))
    source = coupled.source.source_state
    source_array_hashes = {item.name: _array_hash(getattr(source, item.name)) for item in fields(source)
                           if isinstance(getattr(source, item.name), np.ndarray)}
    with np.load(TRACE, allow_pickle=False) as data:
        path = ReferenceCrackPath(data["points_xyz"], source.radius_km)
    inserted = insert_crack_path(coupled.original_mesh, path,
        front_coordinates_m=path.length_m*np.array([.25, .5, .75]))
    basis = EmbeddedPathBasis(coupled.original_mesh, inserted, coupled.radius_m,
                              coupled.source_model.p.poisson_ratio)
    model = PathMechanics(basis, basis.subdivision.intensive(coupled.source.depth_m),
                          coupled.contact_parameters, coupled.law_parameters)
    material = basis.subdivision.fault_fields(source)
    np.savez_compressed(output/"source_material_history.npz", **material,
        metadata=np.array(json.dumps({"format": "diagnostic_inherited_history_no_interface_conversion",
            "source_sha256": source_hashes, "source_array_sha256": source_array_hashes,
            "source_time_myr": source.time_myr}, allow_nan=False)))
    tied = tied_control(coupled, model, output)
    print(json.dumps({"stage": "tied", "checks": tied["checks"], "last": tied["rows"][-1]}, allow_nan=False), flush=True)
    released = released_control(coupled, model, output)
    print(json.dumps({"stage": "released", "checks": released.get("checks"),
        "milestones": released.get("milestones")}, allow_nan=False), flush=True)
    code_unchanged = all(_hash(ROOT/name) == digest for name, digest in code_hashes.items())
    unchanged = (all(_hash(ROOT/name) == digest for name, digest in source_hashes.items())
        and all(_array_hash(getattr(source, name)) == digest for name, digest in source_array_hashes.items()))
    report = {"interpretation": "tied physical-time equivalence and separately prescribed cohesive release",
        "source_sha256": source_hashes, "source_array_sha256": source_array_hashes,
        "sources_unchanged": unchanged, "code_sha256": code_hashes, "code_unchanged_during_run": code_unchanged,
        "source_cells": coupled.original_mesh.cell_count, "child_cells": basis.topology.mesh.cell_count,
        "parent_dofs": basis.nparent, "generalized_dofs": basis.ndof, "fine_split_dofs": basis.membrane.ndof,
        "path_length_km": path.length_m/1000., "smallest_refined_edge_m": model.shortest_edge_m,
        "source_age_myr": source.time_myr, "source_diffuse_active_cells": int(source.fault_active.sum()),
        "inherited_diffuse_friction_work_j": float(source.friction_work_cell_j.sum()),
        "inherited_diffuse_viscous_work_j": float(source.viscous_work_cell_j.sum()),
        "mechanics_fingerprint": model.fingerprint, "tied_control": tied, "released_control": released}
    report["artifact_sha256"] = {p.name: _hash(p) for p in output.glob("*.npz")}
    def serial(value):
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(type(value).__name__)
    (output/"validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                                    allow_nan=False, default=serial)+"\n", encoding="utf-8")
    plot_result(report, output)
    if not unchanged or not code_unchanged or not all(tied["checks"].values()) or not all(released.get("checks", {}).values()):
        raise RuntimeError("One or more validation checks failed; inspect the written report")
    print(json.dumps({"report": str(output/"validation.json"), "sources_unchanged": unchanged}), flush=True)


if __name__ == "__main__":
    main()
