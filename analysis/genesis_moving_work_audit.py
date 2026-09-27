"""Explain the moving-shell mechanical remainder without changing its ledgers.

This bounded audit reproduces the first 500 physical years and one 500-year
step from the saved 10,000-year thermo-mechanical restart. It reconstructs the
thermal/Maxwell predictor and finite kinematics, then compares the endpoint
force work with the stored trapezoidal stress work. The quadratic difference
is an algorithmic quadrature term, not physical heat or fracture work. Neither
this decomposition nor a small force residual closes global thermal energy.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.genesis_moving_genesis_validation import MovingGenesisCase, PRE_GATE, build_case
from analysis.genesis_path_dynamics_validation import SOURCE, TRACE
from tectonics.genesis_coupled_thermal import advance_thermal_loading
from tectonics.genesis_material import face_deformation, polar_increment, rotate_tensor
from tectonics.genesis_moving_tied import MovingTiedLoading
from tectonics.genesis_shell import Membrane


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exactly_equal(first, second):
    for item in fields(first):
        left, right = getattr(first, item.name), getattr(second, item.name)
        if isinstance(left, np.ndarray):
            if not np.array_equal(left, right):
                return False
        elif left != right:
            return False
    return True


def close(left, right, scale):
    return abs(left-right) <= 2e-13*max(abs(scale), 1.)


def audit_step(case, old, context, dt_years, label):
    """No ledger repair: calculate independent scalars from saved physical data."""
    old_mesh = case.model.mesh_for(old)
    parameters = case.source_model.p
    thermal = advance_thermal_loading(case.source_model,
        mesh=old_mesh, radius_km=old.radius_m/1000., layer_mass_kg=case.mass,
        column_enthalpy=context.column_enthalpy, elastic_strain=old.elastic_strain,
        damage=context.damage, water_access=context.water_access,
        boundary_energy_j=context.boundary_energy_j,
        thermal_state=context.thermal, orbit=context.orbit,
        target_myr=context.thermal.time_myr+dt_years/1e6)
    volume = case.mass.sum(axis=1)*thermal.fraction/parameters.density_kg_m3
    young = parameters.young_modulus_pa*(parameters.residual_stiffness
        +(1-parameters.residual_stiffness)*(1-thermal.damage0)**2)
    loading = MovingTiedLoading(dt_years, volume, young, thermal.memory,
        thermal.effective_b, case.force)
    new = case.model.trial(old, loading)
    owner_result, _ = case.trial(old, context, dt_years)

    mesh = case.model.mesh_for(new)
    membrane = Membrane(mesh, case.model.poisson_ratio)
    rotation, increment = polar_increment(face_deformation(old_mesh, mesh,
        old.radius_m/1000., new.radius_m/1000.))
    stress = (new.elastic_strain@membrane.d.T)*young[:, None]
    start_stress = (thermal.memory@membrane.d.T)*young[:, None]
    stress_old_frame = rotate_tensor(stress, rotation.transpose(0, 2, 1))
    elastic_from_predictor = rotate_tensor(thermal.memory
        +thermal.effective_b[:, None]*increment, rotation, engineering=True)
    reconstructed_trapezoid = float(np.sum(np.einsum("fi,fi->f",
        .5*(start_stress+stress_old_frame), increment)*volume))
    energy_change = .5*float(np.sum((np.einsum("fi,fi->f", new.elastic_strain, stress)
        -np.einsum("fi,fi->f", thermal.memory, start_stress))*volume))
    endpoint_hencky = float(np.sum(np.einsum("fi,fi->f", stress_old_frame, increment)*volume))
    quadratic = .5*float(np.sum(np.einsum("fi,ij,fj->f", increment, membrane.d,
        increment)*young*thermal.effective_b*volume))

    # Assemble current internal force without the kernel's assembly/residual
    # helper, and obtain the actual endpoint external force from its callback.
    internal = np.zeros(membrane.ndof)
    for face in range(mesh.cell_count):
        internal[membrane.dofs[face]] += membrane.b[face].T@stress[face]*volume[face]/new.radius_m
    external = case.force(mesh, new.radius_m,
        volume/(mesh.areas_unit_sphere*new.radius_m**2), membrane)
    q = new.last_increment_current_m
    endpoint_internal = float(internal@q)
    residual_dot = float((internal+new.last_drag_force_n-external)@q)
    geometric = endpoint_hencky-endpoint_internal
    decomposition = quadratic-geometric-residual_dot
    work = {name: float(getattr(new, name)-getattr(old, name)) for name in
        ("external_work_j", "drag_work_j", "bulk_work_j", "bulk_loading_correction_j",
         "mechanical_remainder_j")}
    scale = max(abs(value) for value in work.values())
    checks = {
        "matches_thermal_owner_step_exactly": exactly_equal(new, owner_result),
        "predictor_memory_applied_once": bool(np.allclose(new.elastic_strain,
            elastic_from_predictor, rtol=0., atol=3e-15)),
        "external_force_reconstructed": bool(np.array_equal(external, new.last_external_force_n)),
        "trapezoidal_bulk_work_reconstructed": close(reconstructed_trapezoid, work["bulk_work_j"], scale),
        "loading_energy_correction_reconstructed": close(reconstructed_trapezoid-energy_change,
            work["bulk_loading_correction_j"], scale),
        "endpoint_minus_trapezoid_equals_quadratic": close(endpoint_hencky-reconstructed_trapezoid,
            quadratic, scale),
        "external_endpoint_work_reconstructed": close(float(external@q), work["external_work_j"], scale),
        "drag_endpoint_work_reconstructed": close(float(new.last_drag_force_n@q), work["drag_work_j"], scale),
        "remainder_decomposition_matches": close(decomposition, work["mechanical_remainder_j"], scale),
        "quadratic_term_nonnegative": quadratic >= 0.,
    }
    return {"label": label, "start_elapsed_years": old.elapsed_years,
        "end_elapsed_years": new.elapsed_years, "dt_years": dt_years,
        "ledger_increment_j": work,
        "endpoint_internal_force_work_j": endpoint_internal,
        "endpoint_stress_hencky_work_j": endpoint_hencky,
        "reconstructed_trapezoid_bulk_work_j": reconstructed_trapezoid,
        "elastic_energy_change_from_post_thermal_memory_j": energy_change,
        "quadratic_algorithmic_term_j": quadratic,
        "geometric_conjugacy_difference_j": geometric,
        "equilibrium_residual_dot_increment_j": residual_dot,
        "decomposed_remainder_j": decomposition,
        "decomposition_error_j": decomposition-work["mechanical_remainder_j"],
        "decomposition_error_relative_to_step_work": (decomposition-work["mechanical_remainder_j"])/scale,
        "maximum_absolute_hencky_increment": float(np.max(np.abs(increment))),
        "effective_b_range": [float(thermal.effective_b.min()), float(thermal.effective_b.max())],
        "checks": checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
        default=ROOT/"results/genesis_runs/moving_work_audit_20260927")
    parser.add_argument("--restart-directory", type=Path,
        default=ROOT/"results/genesis_runs/moving_genesis500_20260927")
    parser.add_argument("--pre-gate-checkpoint", type=Path, default=PRE_GATE)
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Audit output must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    source_paths = [SOURCE, TRACE, arguments.pre_gate_checkpoint,
        arguments.restart_directory/"restart_mechanics.npz",
        arguments.restart_directory/"restart_thermal.npz"]
    source_hashes = {str(path.resolve()): digest(path) for path in source_paths}
    code_paths = [Path(__file__), ROOT/"analysis/genesis_moving_genesis_validation.py",
        ROOT/"analysis/genesis_path_precursor_validation.py"]
    code_paths += [ROOT/"tectonics"/name for name in (
        "genesis_moving_tied.py", "genesis_coupled_thermal.py", "genesis_material.py",
        "genesis_shell.py", "genesis_mobile.py", "genesis_onset.py", "genesis_tides.py")]
    code_hashes = {str(path.relative_to(ROOT)): digest(path) for path in code_paths}
    case = MovingGenesisCase(build_case(arguments.pre_gate_checkpoint))
    first = audit_step(case, case.initial, case.context, 500., "thermal_gate_to_500_years")
    restart = case.model.load_state(arguments.restart_directory/"restart_mechanics.npz")
    context = case.load_context(arguments.restart_directory/"restart_thermal.npz")
    if restart.elapsed_years != 10000.:
        raise ValueError("This bounded audit requires the saved 10000-year checkpoint")
    second = audit_step(case, restart, context, 500., "10000_to_10500_years")
    checks = {"all_step_identities": all(all(row["checks"].values()) for row in (first, second)),
        "sources_unchanged": all(digest(path)==value for path, value in source_hashes.items()),
        "code_unchanged": all(digest(ROOT/path)==value for path, value in code_hashes.items())}
    report = {"scope": __doc__, "gate_age_myr": case.gate_age_myr,
        "identity": "remainder = quadratic_algorithmic_term - geometric_conjugacy_difference - equilibrium_residual_dot_increment",
        "definitions": {
            "quadratic_algorithmic_term": "0.5 sum(volume * effective_b * Hencky_increment : C : Hencky_increment)",
            "geometric_conjugacy_difference": "endpoint stress in old frame : Hencky increment * volume, minus current internal force dot arrival-frame increment",
            "equilibrium_residual_dot_increment": "(current internal force + current drag - current external force) dot arrival-frame increment",
            "bulk_loading_correction": "trapezoidal stress work minus elastic-energy change from supplied post-thermal/Maxwell memory"},
        "interpretation": "The positive quadratic term measures endpoint-versus-trapezoid time quadrature. It is not assigned to heat, fracture, or a physical dissipation law. Endpoint force work would close the algebraic endpoint-force identity, not prove a physical integral or closed global thermal energy budget. The thermal owner retains separate column/global ledgers.",
        "steps": [first, second], "checks": checks, "source_sha256": source_hashes,
        "code_sha256": code_hashes, "wall_seconds": perf_counter()-started}
    (output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": str(output), "checks": checks,
        "steps": [{"label": row["label"], "remainder_j": row["ledger_increment_j"]["mechanical_remainder_j"],
            "quadratic_j": row["quadratic_algorithmic_term_j"], "identity_error_j": row["decomposition_error_j"]}
            for row in report["steps"]], "wall_seconds": report["wall_seconds"]}, indent=2))
    if not all(checks.values()):
        raise SystemExit("Moving work audit failed")


if __name__ == "__main__":
    main()
