"""Independent force/work reconstruction of accepted physical coupled steps.

This observes the production solver; it does not replace its constitutive law
with a static energy minimum. Only the final mechanical correction of an
accepted thermal/topology trial is retained. Rejected trials contribute nothing.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tectonics.genesis import SECONDS_PER_MYR
from tectonics.genesis_contact_growth import cohort_energy
from tectonics.genesis_coupled import CoupledModel, load_coupled_checkpoint
from analysis.genesis_coupled_validation import _fingerprint


@dataclass
class StepBalance:
    summary: dict
    arrays: dict


def reconstruct_balance(model, before, loading, damage, solved):
    """Assemble accepted forces from final stress and individual cohort forces.

    ``before`` is the trial's prepared state, AFTER topology/material birth.
    ``solved`` is the six-value result of CoupledModel._solve. Loading retains
    the actual accepted interval and old geometry used for the mantle traction.
    An averaged display traction is deliberately not used for force assembly.
    """
    contact, elastic, _, bulk_work, bulk_loss, cohorts = solved
    geometry = model._geometry(before.cut_edges)
    membrane, p = geometry.membrane, model.source_model.p
    delta = contact.displacement_m-before.contact.displacement_m
    volume = model.layer_mass_kg.sum(axis=1)*loading.fraction/p.density_kg_m3
    degradation = p.residual_stiffness+(1-p.residual_stiffness)*(1-damage)**2
    elasticity = (p.young_modulus_pa*degradation)[:, None, None]*membrane.d
    stress = np.einsum("fij,fj->fi", elasticity, elastic)
    initial_stress = np.einsum("fij,fj->fi", elasticity, loading.memory)
    local = np.einsum("fai,fa,f->fi", membrane.b, stress, volume/model.radius_m)
    bulk = np.zeros(membrane.ndof)
    np.add.at(bulk, membrane.dofs.ravel(), local.ravel())
    traces = np.zeros((geometry.jump_operator.shape[0]//2, 2))
    np.add.at(traces, cohorts.trace_index, cohorts.traction_pa*cohorts.area_ref_m2[:, None])
    interface = np.asarray(geometry.jump_operator.T@traces.ravel())
    drag_coefficient = (geometry.drag_area_m2*model.contact_parameters.basal_drag_pa_s_m
                        /(loading.dt_myr*SECONDS_PER_MYR))
    drag = drag_coefficient*delta
    external = model._external(geometry, loading.depth_km)
    residual = bulk+interface+drag-external
    scale = max(np.linalg.norm(bulk), np.linalg.norm(interface), np.linalg.norm(external),
                p.young_modulus_pa*float(volume.sum())/model.radius_m*1e-10)
    strain_increment = geometry._strain(delta)
    midpoint = float(np.sum(np.einsum("fi,fi->f", .5*(initial_stress+stress), strain_increment)*volume))
    bulk_quadrature = float(np.sum(np.einsum("fi,fi->f", .5*(stress-initial_stress), strain_increment)*volume))
    old_jump = (geometry.jump_operator@before.contact.displacement_m).reshape(-1, 2)
    new_jump = (geometry.jump_operator@contact.displacement_m).reshape(-1, 2)
    interface_energy_change = (cohort_energy(cohorts, *new_jump.T, model.law_parameters)
                              -cohort_energy(before.cohorts, *old_jump.T, model.law_parameters))
    contact_loss = sum(float(np.sum(getattr(cohorts, key)-getattr(before.cohorts, key)))
                       for key in ("friction_work_j", "viscous_work_j", "fracture_work_j"))
    contact_endpoint = float(interface@delta)
    contact_quadrature = contact_endpoint-interface_energy_change-contact_loss
    external_work, drag_work = float(external@delta), float(drag@delta)
    remainder = external_work-bulk_work-interface_energy_change-contact_loss-drag_work
    residual_work = float(residual@delta)
    throughput = max(abs(external_work), abs(midpoint), abs(contact_endpoint), drag_work, 1.)
    summary = {
        "time_myr": loading.thermal_state.time_myr,
        "step_years": loading.dt_myr*1e6,
        "seam_count": len(before.cut_edges), "cohort_count": len(cohorts.trace_index),
        "force_balance_relative_residual": float(np.linalg.norm(residual)/scale),
        "static_relative_residual_without_drag": float(np.linalg.norm(bulk+interface-external)/scale),
        "solver_reported_residual": contact.equilibrium_residual,
        "bulk_force_norm_n": float(np.linalg.norm(bulk)),
        "contact_force_norm_n": float(np.linalg.norm(interface)),
        "drag_force_norm_n": float(np.linalg.norm(drag)),
        "external_force_norm_n": float(np.linalg.norm(external)),
        "max_increment_displacement_m": float(np.max(np.abs(delta))),
        "external_work_j": external_work, "drag_work_j": drag_work,
        "bulk_midpoint_work_j": midpoint, "bulk_loading_correction_j": bulk_loss,
        "interface_energy_change_j": interface_energy_change,
        "contact_dissipation_j": contact_loss,
        "bulk_endpoint_quadrature_j": bulk_quadrature,
        "contact_endpoint_quadrature_j": contact_quadrature,
        "mechanical_remainder_j": remainder, "residual_work_j": residual_work,
        "work_throughput_scale_j": throughput,
        "remainder_identity_relative_error": abs(remainder-bulk_quadrature-contact_quadrature+residual_work)/throughput,
        "bulk_work_ledger_relative_error": abs(midpoint-bulk_work)/throughput,
        "external_work_ledger_relative_error": abs(external_work-(contact.external_work_j-before.contact.external_work_j))/throughput,
        "drag_work_ledger_relative_error": abs(drag_work-(contact.drag_work_j-before.contact.drag_work_j))/throughput,
    }
    arrays = dict(displacement_increment_m=delta, bulk_force_n=bulk,
                  contact_force_n=interface, drag_force_n=drag,
                  external_force_n=external, residual_force_n=residual)
    if not all(np.isfinite(value) for value in summary.values()) or not all(np.isfinite(a).all() for a in arrays.values()):
        raise ValueError("Nonfinite reconstructed physical-step balance")
    return StepBalance(summary, arrays)


def advance_with_balance_audit(model, state, thermal, orbit, target_myr, *, max_step_myr=.001):
    """Observe a normal production advance, restoring methods even on failure.

    Call on an exclusively owned model. This diagnostic temporarily wraps two
    instance methods; it is not a concurrent observer or a checkpoint format.
    """
    original_solve, original_trial = model._solve, model._trial
    previous_attributes = {name: model.__dict__.get(name) for name in ("_solve", "_trial")}
    present = {name: name in model.__dict__ for name in previous_attributes}
    accepted, pending = [], []

    def solve(before, loading, damage):
        result = original_solve(before, loading, damage)
        pending.append(reconstruct_balance(model, before, loading, damage, result))
        return result

    def trial(*args):
        pending.clear()
        result = original_trial(*args)
        if not pending:
            raise RuntimeError("Accepted coupled trial supplied no mechanical state")
        # Damage evolution may request a second solve; only its final state is
        # committed. A later rejected trial never reaches this append.
        row = pending[-1].summary
        committed = result[0].mechanical_energy_remainder_j-args[0].mechanical_energy_remainder_j
        row["committed_mechanical_remainder_j"] = committed
        row["mechanical_remainder_ledger_relative_error"] = abs(committed-row["mechanical_remainder_j"])/row["work_throughput_scale_j"]
        row["interface_birth_energy_increment_j"] = result[0].interface_birth_energy_j-args[0].interface_birth_energy_j
        accepted.append(pending[-1])
        return result

    try:
        model._solve, model._trial = solve, trial
        result = model.step(state, thermal, orbit, target_myr, max_step_myr=max_step_myr)
    finally:
        for name, value in previous_attributes.items():
            if present[name]:
                setattr(model, name, value)
            else:
                delattr(model, name)
    if len(accepted) != result[0].accepted_steps-state.accepted_steps:
        raise RuntimeError("Mechanical audit missed an accepted physical interval")
    return result, accepted


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--years", type=float, default=10.)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not np.isfinite(args.years) or args.years <= 0:
        parser.error("years must be finite and positive")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("output must be new or empty")
    source_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    with np.load(args.checkpoint, allow_pickle=False) as source:
        kind = json.loads(str(source["metadata"]))["format"]
    if kind == "genesis-faults-0.1":
        model, state, thermal, orbit = CoupledModel.from_fault_checkpoint(args.checkpoint)
    else:
        model, state, thermal, orbit = load_coupled_checkpoint(args.checkpoint)
    source_state = _fingerprint((state, thermal, orbit))
    started = perf_counter()
    result, balances = advance_with_balance_audit(model, state, thermal, orbit,
                                                 state.time_myr+args.years/1e6)
    elapsed = perf_counter()-started
    if source_state != _fingerprint((state, thermal, orbit)):
        raise RuntimeError("Audit mutated its initial state")
    if source_hash != hashlib.sha256(args.checkpoint.read_bytes()).hexdigest():
        raise RuntimeError("Source checkpoint changed during audit")
    args.output.mkdir(parents=True, exist_ok=True)
    arrays = args.output/"last_step_forces.npz"
    if balances:
        np.savez_compressed(arrays, **balances[-1].arrays)
    steps = [balance.summary for balance in balances]
    report = dict(source=str(args.checkpoint.resolve()), source_sha256=source_hash,
        requested_years=args.years, wall_seconds=elapsed, source_unchanged=True,
        initial_state_unchanged=True, accepted_steps=len(steps),
        reached_requested_duration=bool(abs(result[0].time_myr-(state.time_myr+args.years/1e6)) <= 128*np.spacing(max(abs(result[0].time_myr), 1.))),
        final=model.diagnostics(*result[:3]), steps=steps,
        checks={"accepted_physical_steps": bool(steps),
            "force_balance": bool(steps) and all(r["force_balance_relative_residual"] <= model.contact_parameters.equilibrium_tolerance for r in steps),
            "work_identity": bool(steps) and all(r["remainder_identity_relative_error"] < 1e-9 for r in steps),
            "work_ledgers": bool(steps) and all(max(r[k] for k in ("bulk_work_ledger_relative_error", "external_work_ledger_relative_error", "drag_work_ledger_relative_error", "mechanical_remainder_ledger_relative_error")) < 1e-9 for r in steps)},
        limitations=["Force closure and the work identity do not certify fracture localization or a plate handoff.",
            "Omitting basal drag is a different static problem, not a valid source-state acceptance test.",
            "Work quadrature remainders are not crack release energy or deposited heat."],
        implementation_sha256={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
            for name in ("tectonics/genesis_coupled.py", "tectonics/genesis_coupled_thermal.py",
                         "tectonics/genesis_contact_growth.py", "analysis/genesis_coupled_balance_audit.py")},
        forces_sha256=hashlib.sha256(arrays.read_bytes()).hexdigest() if balances else None)
    (args.output/"balance.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "checks": report["checks"], "accepted_steps": len(steps), "wall_seconds": elapsed}), flush=True)
    return 0 if all(report["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
