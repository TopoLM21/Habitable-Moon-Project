"""Thermal-gate precursor for a prescribed, material-transported path.

The legacy moving-shell solver imposes static equilibrium when the last face
becomes load bearing. This diagnostic enters the existing rate-dependent path
solver immediately BEFORE that solve. It never resets an equilibrated stress
history. Path selection is retrospective, and bulk damage feedback is omitted
over this bounded onset experiment. This is not a production coupled restart.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.genesis_path_dynamics_validation import SOURCE, TRACE
from tectonics.genesis import temperatures
from tectonics.genesis_coupled_thermal import advance_thermal_loading, evolve_coupled_damage
from tectonics.genesis_crack_path import ReferenceCrackPath
from tectonics.genesis_faults import load_fault_checkpoint, save_fault_checkpoint
from tectonics.genesis_mobile import _external_force
from tectonics.genesis_onset import advance_orbit_thermal
from tectonics.genesis_path_basis import EmbeddedPathBasis
from tectonics.genesis_path_birth import recover_tied_tractions, classify_tied_onset
from tectonics.genesis_path_dynamics import PathLoading, PathMechanics, _PathRetry
from tectonics.genesis_path_geometry import PathGeometryParameters
from tectonics.genesis_path_mesh import insert_crack_path
from tectonics.genesis_path_onset_event import TiedOnsetSample, TiedOnsetBracket
from tectonics.genesis_shell import Membrane, mantle_traction


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def material_support(late_mesh, earlier_mesh, points):
    """Pull back control points through fixed parent-face material ancestry.

    Normalized chord barycentric interpolation is the explicit convention.
    Subsequent minor-arc segments remain a prescribed observation support.
    """
    candidates = cKDTree(late_mesh.centroids).query(points, k=16)[1]
    parent, barycentric = [], []
    for point, choices in zip(points, candidates):
        for face in choices:
            weights = np.linalg.solve(late_mesh.vertices[late_mesh.faces[face]].T, point)
            weights /= weights.sum()
            if weights.min() >= -1e-10:
                parent.append(int(face))
                barycentric.append(weights)
                break
        else:
            raise ValueError("Path control point was not contained in the searched material faces")
    parent, barycentric = np.asarray(parent), np.asarray(barycentric)
    mapped = np.einsum("pi,pij->pj", barycentric,
        earlier_mesh.vertices[earlier_mesh.faces[parent]])
    mapped /= np.linalg.norm(mapped, axis=1)[:, None]
    return mapped, parent, barycentric


def thermal_gate_duration(source_model, state, thermal, orbit, upper_myr):
    """Localize the existing whole-membrane thickness gate without mechanics."""
    mesh = source_model.mesh_for(state)
    p = source_model.p
    def thickness(dt):
        following, _, _, _ = advance_orbit_thermal(deepcopy(thermal), orbit,
            source_model.thermal, source_model.tides_p, state.time_myr+dt)
        tm0, ts0 = temperatures(thermal.energy, source_model.thermal)
        tm1, ts1 = temperatures(following.energy, source_model.thermal)
        enthalpy, _ = source_model._conduct(state, mesh, dt, ts0, ts1, tm0, tm1)
        fraction, _, _ = source_model._phase(enthalpy, ts1, tm1)
        return float(np.min(fraction*source_model._column_geometry(state, mesh)))
    if state.membrane_established or thickness(upper_myr) < p.min_load_bearing_thickness_km:
        raise ValueError("Thermal gate needs an unestablished lower state and an established upper bound")
    lower, upper = 0., float(upper_myr)
    for _ in range(40):
        middle = .5*(lower+upper)
        if thickness(middle) >= p.min_load_bearing_thickness_km:
            upper = middle
        else:
            lower = middle
    return upper


@dataclass
class ThermalContext:
    thermal: object
    orbit: object
    column_enthalpy: np.ndarray
    elastic_parent: np.ndarray
    damage: np.ndarray
    water_access: np.ndarray
    boundary_energy_j: float


class PrecursorCase:
    """Independent tied mechanics plus the existing thermal/orbit predictor."""

    def __init__(self, source_model, before, gate_loading, late_mesh, points):
        if before.membrane_established or np.any(before.elastic_strain) or np.any(before.damage):
            raise ValueError("Entry must inherit an unestablished, unstressed membrane")
        self.source_model, self.before = source_model, before
        self.mesh = source_model.mesh_for(before)
        self.parent_membrane = Membrane(self.mesh, source_model.p.poisson_ratio)
        self.radius_m = before.radius_km*1000.
        mapped, self.path_parent, self.path_barycentric = material_support(late_mesh, self.mesh, points)
        path = ReferenceCrackPath(mapped, before.radius_km)
        inserted = insert_crack_path(self.mesh, path,
            front_coordinates_m=path.length_m*np.array([.25, .5, .75]))
        self.basis = EmbeddedPathBasis(self.mesh, inserted, self.radius_m, source_model.p.poisson_ratio)
        projection = self.basis.subdivision
        self.model = PathMechanics(self.basis, projection.intensive(gate_loading.depth_km*1000),
            geometry_parameters=PathGeometryParameters())
        # All earlier partial rafts were mechanically inactive by the existing
        # global gate. The inherited stress is zero; no loaded history is erased.
        self.initial = self.model.initial(projection.tensor(before.elastic_strain, engineering=True))
        self.context = ThermalContext(gate_loading.thermal_state, gate_loading.orbit,
            gate_loading.column_enthalpy, before.elastic_strain.copy(), before.damage.copy(),
            gate_loading.water_access, gate_loading.boundary_energy_j)
        self.gate_loading = gate_loading

    def current_section(self, state, loading):
        """Measured force divided by the ACTUAL solid section at this time.

        Before birth every cohort is only a zero-jump, zero-work placeholder.
        Updating its section for a diagnostic copy does not change mechanics
        or transfer interface history. This operation refuses any active or
        loaded interface; it is not a general solidification/cohort adapter.
        """
        if state.active_interval is not None or any(np.any(getattr(state.cohorts, name)) for name in
            ("traction_pa", "damage", "max_opening_m", "plastic_slip_m", "cumulative_slip_m",
             "friction_work_j", "viscous_work_j", "fracture_work_j", "shear_remainder_j")):
            raise ValueError("Current-section diagnostic requires entirely tied, zero-work placeholders")
        depth = loading.volume_m3/(self.basis.subdivision.mesh.areas_unit_sphere*self.radius_m**2)
        model = PathMechanics(self.basis, depth, self.model.parameters, self.model.law_parameters,
            geometry_parameters=self.model.geometry_parameters)
        cohorts = replace(deepcopy(state.cohorts), z_hi_ref_m=model.trace_depth_m.copy(),
            area_ref_m2=model.geometry.interface_area_m2.copy())
        return model, replace(deepcopy(state), cohorts=cohorts)

    def external(self, depth_km):
        p, basis = self.source_model.p, self.basis
        def force(mesh, membrane, depth):
            traction = mantle_traction(mesh, p, depth)
            result = _external_force(membrane, traction, self.radius_m/1000, p.young_modulus_pa)
            return result*(p.young_modulus_pa*self.radius_m*1000)
        parent = force(self.mesh, self.parent_membrane, depth_km)
        fine = basis.subdivision.mesh
        fine_force = force(fine, Membrane(fine, p.poisson_ratio), basis.subdivision.intensive(depth_km))
        area, ancestry = basis.fine_drag_area_m2[:-1:2], basis.topology.parent_vertex
        summed = np.bincount(ancestry, weights=area, minlength=fine.vertex_count)
        split = np.zeros(basis.membrane.ndof)
        split[:-1] = (fine_force[:-1].reshape(-1, 2)[ancestry]
            *(area/summed[ancestry])[:, None]).ravel()
        split[-1] = fine_force[-1]
        return basis.external(parent, split)

    def trial(self, state, context, years):
        if state.active_interval is not None:
            raise ValueError("This diagnostic thermal adapter owns tied states only")
        p, projection = self.source_model.p, self.basis.subdivision
        loading = advance_thermal_loading(self.source_model, mesh=self.mesh,
            radius_km=self.radius_m/1000, layer_mass_kg=self.before.layer_mass_kg,
            column_enthalpy=context.column_enthalpy, elastic_strain=context.elastic_parent,
            damage=context.damage, water_access=context.water_access,
            boundary_energy_j=context.boundary_energy_j,
            thermal_state=context.thermal, orbit=context.orbit,
            target_myr=context.thermal.time_myr+years/1e6)
        if abs(loading.dt_myr*1e6-years) > max(2e-9, years*1e-9):
            raise ValueError("Thermal clock did not advance the requested mechanical interval")
        volume = self.before.layer_mass_kg.sum(axis=1)*loading.fraction/p.density_kg_m3
        degradation = p.residual_stiffness+(1-p.residual_stiffness)*(1-loading.damage0)**2
        elasticity = projection.intensive((p.young_modulus_pa*degradation)[:, None, None]*self.parent_membrane.d)
        mechanical = PathLoading(years, projection.extensive(volume), elasticity,
            projection.tensor(loading.memory, engineering=True), projection.intensive(loading.effective_b),
            self.external(loading.depth_km), projection.intensive(loading.water_access))
        candidate = self.model.trial(state, mechanical)
        delta = candidate.displacement_m[:self.basis.nparent]-state.displacement_m[:self.basis.nparent]
        strain = np.einsum("fai,fi->fa", self.parent_membrane.b,
            delta[self.parent_membrane.dofs])/self.radius_m
        elastic = loading.memory+loading.effective_b[:, None]*strain
        following = ThermalContext(loading.thermal_state, loading.orbit, loading.column_enthalpy,
            elastic, loading.damage0, loading.water_access, loading.boundary_energy_j)
        water = mechanical.water_access[self.basis.topology.seam_faces].mean(axis=1).repeat(2)
        stress = np.einsum("fij,fj->fi", elasticity, candidate.elastic_strain)
        return candidate, stress, water, following, mechanical, loading


def staged_onset(case, *, max_step_years=500., time_tolerance_years=1e-7):
    """Accept physical thermal history up to the last subthreshold predictor."""
    if not np.isfinite(max_step_years) or max_step_years <= 0:
        raise ValueError("Maximum predictor step must be finite and positive")
    model, p = case.model, case.source_model.p
    state, stress, water, context, mechanical, loading = case.trial(case.initial, case.context, .001)
    first = (state, stress, water, context, mechanical, loading)
    rows = []
    maximum_damage_increment = 0.
    def sample(values, duration):
        section, copy = case.current_section(values[0], values[4])
        recovery = recover_tied_tractions(section, copy, values[1])
        onset = classify_tied_onset(recovery, values[2], section.law_parameters)
        score = np.maximum(onset.normal_ratio, onset.shear_ratio)
        governing = np.flatnonzero(recovery.trace_observed & (score >= onset.maximum_observed_ratio-1e-6))
        return TiedOnsetSample(duration, copy, recovery, onset, governing)
    for _ in range(10000):
        accepted = (state, stress, water, context, mechanical, loading)
        start_sample = sample(accepted, 0.)
        onset = start_sample.onset
        rows.append({"elapsed_years": state.elapsed_years, "strength_ratio": onset.maximum_observed_ratio})
        if onset.maximum_observed_ratio >= 1:
            raise ValueError("Accepted precursor unexpectedly exceeded strength")
        try:
            trial = case.trial(state, context, max_step_years)
        except _PathRetry as exc:
            bracket = TiedOnsetBracket("stopped_before_crossing", start_sample, start_sample,
                0, time_tolerance_years, 1e-6)
            return first, accepted, bracket, accepted, accepted, rows, maximum_damage_increment, str(exc)
        candidate, candidate_stress, candidate_water, following, _, thermal_loading = trial
        # This audit is necessary because the adapter deliberately does not
        # apply the legacy bulk-damage corrector. A nonzero result invalidates
        # the claim that the omitted corrector is inactive on this trajectory.
        degradation = p.residual_stiffness+(1-p.residual_stiffness)*(1-thermal_loading.damage0)**2
        parent_stress = np.einsum("ij,fj->fi", case.parent_membrane.d, following.elastic_parent)
        parent_stress *= (p.young_modulus_pa*degradation)[:, None]
        predicted_damage, _ = evolve_coupled_damage(case.source_model, thermal_loading,
            case.mesh, case.radius_m/1000, parent_stress)
        maximum_damage_increment = max(maximum_damage_increment,
            float(np.max(np.abs(predicted_damage-thermal_loading.damage0))))
        end_sample = sample(trial, max_step_years)
        candidate_onset = end_sample.onset
        if candidate_onset.maximum_observed_ratio >= 1:
            # The production locator owns one fixed contact section. Here
            # each speculative thermal state has a different physical depth,
            # so use a bounded analysis-only bracket with its own model view.
            lower_sample, upper_sample, calls = start_sample, end_sample, 1
            while upper_sample.duration_years-lower_sample.duration_years > time_tolerance_years:
                middle = .5*(lower_sample.duration_years+upper_sample.duration_years)
                probe = sample(case.trial(state, context, middle), middle)
                calls += 1
                if calls > 64:
                    raise RuntimeError("Changing-section onset bracket did not converge")
                if probe.onset.maximum_observed_ratio < 1:
                    lower_sample = probe
                else:
                    upper_sample = probe
            bracket = TiedOnsetBracket("bracketed", lower_sample, upper_sample, calls,
                time_tolerance_years, 1e-6)
            lower = case.trial(state, context, lower_sample.duration_years)
            upper = case.trial(state, context, upper_sample.duration_years)
            return first, (state, stress, water, context, mechanical, loading), bracket, lower, upper, rows, maximum_damage_increment, None
        state, stress, water, context, mechanical, loading = trial
        if len(rows) % 40 == 0:
            print("Accepted thermal trajectory:", state.elapsed_years,
                "yr; current-section strength ratio", candidate_onset.maximum_observed_ratio, flush=True)
        if state.elapsed_years > 100000.:
            raise RuntimeError("No crossing within the bounded 100000-year diagnostic")
    raise RuntimeError("Precursor predictor count exceeded its bounded limit")


def build_case(pre_gate_checkpoint):
    source_model, state, thermal, orbit, meta = load_fault_checkpoint(pre_gate_checkpoint)
    for index, margin in enumerate((1., .001, .000001)):
        duration = thermal_gate_duration(source_model, state, thermal, orbit,
            .002 if index == 0 else .00001)
        target = state.time_myr+duration-margin/1e6
        if target > state.time_myr:
            state, thermal, orbit, _ = source_model.step(state, thermal, orbit, target)
        if state.membrane_established:
            raise ValueError("Pre-gate refinement inadvertently entered legacy static mechanics")
    duration = thermal_gate_duration(source_model, state, thermal, orbit, 1e-8)
    gate = advance_thermal_loading(source_model, mesh=source_model.mesh_for(state),
        radius_km=state.radius_km, layer_mass_kg=state.layer_mass_kg,
        column_enthalpy=state.column_enthalpy, elastic_strain=state.elastic_strain,
        damage=state.damage, water_access=state.water_access,
        boundary_energy_j=state.boundary_energy_j, thermal_state=thermal, orbit=orbit,
        target_myr=state.time_myr+duration+1e-12)
    if gate.depth_km.min() < source_model.p.min_load_bearing_thickness_km:
        raise ValueError("Thermal entry did not reach the existing whole-shell thickness gate")
    late_model, late_state, _, _, _ = load_fault_checkpoint(SOURCE)
    for field in ("p", "thermal", "onset_p", "tides_p", "mobile_p", "fault_p"):
        if asdict(getattr(source_model, field)) != asdict(getattr(late_model, field)):
            raise ValueError("Precursor and late reference must use identical physical parameters")
    with np.load(TRACE, allow_pickle=False) as archive:
        points = archive["points_xyz"]
    return PrecursorCase(source_model, state, gate, late_model.mesh_for(late_state), points)


def _save_context(path, context, case, loading):
    np.savez_compressed(path, column_enthalpy=context.column_enthalpy,
        elastic_parent=context.elastic_parent, damage=context.damage, water_access=context.water_access,
        layer_mass_kg=case.before.layer_mass_kg, path_points_xyz=case.basis.insertion.path.points_xyz,
        path_parent_face=case.path_parent, path_barycentric=case.path_barycentric,
        volume_m3=loading.volume_m3, elasticity_pa=loading.elasticity,
        external_force_n=loading.external_force, fine_water_access=loading.water_access,
        metadata=np.array(json.dumps({"format": "diagnostic-path-thermal-context-0.1",
            "thermal": asdict(context.thermal), "orbit": asdict(context.orbit),
            "boundary_energy_j": context.boundary_energy_j,
            "production_restart": False}, allow_nan=False)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pre-gate-checkpoint", type=Path)
    parser.add_argument("--max-step-years", type=float, default=500.)
    args = parser.parse_args()
    code_paths = [Path(__file__)] + [ROOT/"tectonics"/name for name in (
        "genesis_path_birth.py", "genesis_path_dynamics.py", "genesis_path_basis.py",
        "genesis_coupled_thermal.py", "genesis_contact_geometry.py", "genesis_contact_growth.py",
        "genesis_contact_law.py", "genesis_faults.py", "genesis_mobile.py", "genesis_shell.py")]
    code_hashes = {str(path.relative_to(ROOT)): _hash(path) for path in code_paths}
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    if args.pre_gate_checkpoint is None:
        source_model, _, _, _, meta = load_fault_checkpoint(SOURCE)
        state, thermal, orbit = source_model.initial()
        for index in range(1, 477):
            state, thermal, orbit, _ = source_model.step(state, thermal, orbit, index*.002)
            if state.stopped_reason or thermal.stopped_reason:
                raise RuntimeError("Replay stopped before requested pre-membrane state")
        checkpoint = output/"pre_gate_fault_checkpoint.npz"
        save_fault_checkpoint(checkpoint, source_model, state, thermal, orbit,
            meta["controls"], {"scope": "fresh replay with original physical parameters and 0.002 Myr requested grid", "source_sha256": _hash(SOURCE)})
    else:
        checkpoint = args.pre_gate_checkpoint.resolve()
    source_hashes = {str(path): _hash(path) for path in (SOURCE, TRACE, checkpoint)}
    case = build_case(checkpoint)
    model = case.model
    first, accepted, bracket, lower, higher, probes, maximum_damage_increment, stop_reason = staged_onset(
        case, max_step_years=args.max_step_years)
    candidate, stress, water, context, mechanical, _ = first
    first_model, first_state = case.current_section(candidate, mechanical)
    recovery = recover_tied_tractions(first_model, first_state, stress)
    start_onset = classify_tied_onset(recovery, water, model.law_parameters)
    for label, values in (("subthreshold", first), ("last_accepted", accepted)) + (
            (("onset_lower", lower), ("onset_upper", higher)) if stop_reason is None else ()):
        section, measured = case.current_section(values[0], values[4])
        section.save_state(output/(label+"_mechanics.npz"), measured)
    _save_context(output/"subthreshold_context.npz", context, case, mechanical)
    _save_context(output/"last_accepted_context.npz", accepted[3], case, accepted[4])
    if stop_reason is None:
        _save_context(output/"onset_lower_context.npz", lower[3], case, lower[4])
        _save_context(output/"onset_upper_context.npz", higher[3], case, higher[4])
    np.savez_compressed(output/"measured_tractions.npz", lower_pa=bracket.lower.recovery.traction_pa,
        upper_pa=bracket.upper.recovery.traction_pa, observed=bracket.upper.recovery.trace_observed,
        water= higher[2], governing=bracket.upper.governing_trace_indices)
    report = {"scope": "prescribed material support, thermal/orbit/Maxwell loading after original whole-shell thermal gate, before legacy instantaneous equilibrium; no damage corrector or automatic birth",
        "source_cells": case.mesh.cell_count, "source_parameters_unchanged": True,
        "thermal_gate_age_myr": case.context.thermal.time_myr,
        "path_length_km": case.basis.insertion.path.length_m/1000,
        "first_predictor_years": candidate.elapsed_years,
        "first_predictor_strength_ratio": start_onset.maximum_observed_ratio,
        "max_step_years": args.max_step_years,
        "traction_section": "current solid depth, diagnostic views of entirely tied zero-work placeholders; not frozen thermal-gate area",
        "current_depth_m_quantiles": np.quantile(case.current_section(higher[0], higher[4])[0].trace_depth_m, [0, .5, 1]).tolist(),
        "maximum_omitted_damage_corrector_increment": maximum_damage_increment,
        "probes": probes, "bracket_status": bracket.status, "stopped_reason": stop_reason,
        "onset_lower_elapsed_years": bracket.lower.state.elapsed_years if stop_reason is None else None,
        "onset_upper_elapsed_years": bracket.upper.state.elapsed_years if stop_reason is None else None,
        "last_accepted_elapsed_years": higher[0].elapsed_years,
        "bracket_width_years": bracket.time_width_years if stop_reason is None else None,
        "lower_ratio": bracket.lower.onset.maximum_observed_ratio,
        "upper_ratio": bracket.upper.onset.maximum_observed_ratio,
        "upper_strength_overshoot": bracket.upper_strength_overshoot if stop_reason is None else None,
        "governing_trace_indices": bracket.upper.governing_trace_indices.tolist() if stop_reason is None else [],
        "largest_ratio_trace_indices": bracket.upper.governing_trace_indices.tolist(),
        "event_resolution": {"time_tolerance_met": bracket.time_width_years <= 1e-7,
            "strength_tolerance_met": bracket.upper_within_strength_tolerance} if stop_reason is None else None,
        "callback_count": bracket.callback_count,
        "upper_geometry": model.geometry_metrics(higher[0]),
        "source_sha256": source_hashes,
        "code_sha256": code_hashes,
        "production_restart": False,
        "law_parameters": asdict(model.law_parameters),
        "wall_seconds": perf_counter()-started}
    report["checks"] = {"entry_was_unestablished_and_unstressed": not case.before.membrane_established and not np.any(case.before.elastic_strain),
        "thermal_gate_reached": bool(np.all(case.gate_loading.depth_km >= case.source_model.p.min_load_bearing_thickness_km)),
        "first_predictor_subthreshold": start_onset.maximum_observed_ratio < 1,
        "crossing_bracketed_or_admissibility_stop_explicit": (bracket.lower.onset.maximum_observed_ratio < 1 <= bracket.upper.onset.maximum_observed_ratio
            if stop_reason is None else bracket.status == "stopped_before_crossing" and bracket.upper.onset.maximum_observed_ratio < 1),
        "event_resolution_met_or_explicitly_absent": (bracket.time_width_years <= 1e-7
            and bracket.upper_within_strength_tolerance) if stop_reason is None else report["event_resolution"] is None,
        "no_interface_birth": higher[0].active_interval is None and not np.any(higher[0].cohorts.traction_pa),
        "source_files_unchanged": all(_hash(name)==digest for name,digest in source_hashes.items()),
        "code_unchanged_during_run": all(_hash(ROOT/name)==digest for name,digest in code_hashes.items()),
        "current_interface_area_used": bool(np.allclose(case.current_section(higher[0], higher[4])[0].reference_volume_m3,
            higher[4].volume_m3, rtol=1e-14, atol=0)),
        "same_thermal_and_orbit_clocks": higher[3].thermal.time_myr == higher[3].orbit.time_myr,
        "omitted_bulk_damage_corrector_inactive": maximum_damage_increment == 0.,
        "equilibrium_balanced": higher[0].equilibrium_residual < 1e-8}
    report["artifact_sha256"] = {path.name: _hash(path) for path in output.glob("*.npz")}
    (output/"validation.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, allow_nan=False), flush=True)
    if not all(report["checks"].values()):
        raise RuntimeError("Precursor validation failed")


if __name__ == "__main__":
    main()
