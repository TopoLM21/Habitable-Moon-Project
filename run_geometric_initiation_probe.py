"""One prescribed-motion step with explicit, state-bound weak-fault loading.

This is a forced-underthrust admissibility experiment. It has no stress solver,
thermal evolution or force feedback, and never infers fault dip from scalar
damage, plate age or plate number. Run a new invocation with a fresh loading
snapshot to attempt another step.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

import numpy as np

from run_fractional_transport_probe import DEFAULT_SOURCE, ROOT, digest, load_probe_source, make_birth_factory
from tectonics.fractional_surface import EXTENSIVE_FIELDS
from tectonics.fractional_surface_io import surface_from_lithosphere
from tectonics.geometric_boundary import (initialize_boundary, consume_geometric_transaction,
    boundary_diagnostics, polarity_evidence, surface_digest)
from tectonics.geometric_boundary_io import load_boundary_checkpoint, save_boundary_checkpoint
from tectonics.geometric_initiation_adapter import (make_forced_underthrust_resolver,
    fault_snapshot_from_dict)
from tectonics.geometric_polarity import (PolarityEvidence, contact_polarity_report,
    transport_polarity_evidence)
from tectonics.geometric_surface import from_fractional_surface, totals
from tectonics.geometric_transport import advance_geometric_surface, UnresolvedPolarityError

FORMAT = "geometric-forced-underthrust-probe-1"
SCOPE = ("Single prescribed-motion step; explicitly oriented weak faults and frozen effective stress; "
         "forced-underthrust admissibility, not spontaneous initiation; passive slab accounting; no heat or forces")


def _code_hashes():
    paths = [Path(__file__), ROOT/"run_fractional_transport_probe.py",
             ROOT/"tectonics/spherical_polygons.py",
             *sorted((ROOT/"tectonics").glob("geometric_*.py"))]
    return {path.relative_to(ROOT).as_posix(): digest(path) for path in paths}


def _evidence(inventory, external):
    return tuple(external)+tuple(item if isinstance(item, PolarityEvidence) else PolarityEvidence(**item)
        for item in polarity_evidence(inventory))


def _compatible_experiment(saved, current):
    # A joint passive-slab checkpoint can enter the stricter initiation mode;
    # its immutable source, motion, step and inheritance flag must still agree.
    ignored = {"format", "scope"}
    return ({key: value for key, value in saved.items() if key not in ignored}
            == {key: value for key, value in current.items() if key not in ignored})


def _validate_resume(state, inventory, saved, source_state, experiment):
    if not _compatible_experiment(saved.get("experiment", {}), experiment):
        raise ValueError("Resume requires identical source, motion, time step and polarity initialization")
    required = ("initial_totals", "cumulative_losses", "cumulative_births", "history",
                "polarity_initialization", "external_polarity_evidence", "unresolved_polarity_lineage")
    if any(name not in saved for name in required):
        raise ValueError("Resume provenance is missing geometric transaction history")
    if state.time_myr < source_state.time_myr:
        raise ValueError("Resume precedes its immutable source")
    history = saved["history"]
    if (not isinstance(history, list) or len(history) != len(inventory.transactions)
            or any(item.get("time_myr") != transaction.end_time_myr
                   for item, transaction in zip(history, inventory.transactions))
            or (history and history[-1]["time_myr"] != state.time_myr)
            or (not history and state.time_myr != source_state.time_myr)):
        raise ValueError("Resume history and surface chronology disagree")
    source_hash = surface_digest(source_state)
    if inventory.transactions:
        first = inventory.transactions[0]
        if (first.start_time_myr != source_state.time_myr
                or first.before_surface_digest != source_hash):
            raise ValueError("Resume transaction chain does not start at its immutable source geometry")
    elif surface_digest(state) != source_hash:
        raise ValueError("Resume without transactions must equal its immutable source geometry")
    initial, retained = totals(source_state.fragments), totals(state.fragments)
    for field in EXTENSIVE_FIELDS:
        archived = math.fsum(getattr(cohort.parcel, field) for cohort in inventory.cohorts)
        try:
            origin, lost, born = (saved[name][field] for name in
                ("initial_totals", "cumulative_losses", "cumulative_births"))
        except (KeyError, TypeError) as error:
            raise ValueError("Resume material ledger is incomplete") from error
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0.
               for x in (origin, lost, born)):
            raise ValueError("Resume material ledger must contain finite nonnegative amounts")
        tolerance = 5e-12*max(abs(initial[field]), abs(born), 1.)
        if (abs(origin-initial[field]) > tolerance or abs(lost-archived) > tolerance
                or abs(retained[field]+lost-born-origin) > tolerance):
            raise ValueError("Resume material ledger does not match source and passive archive")


def execute_probe(source, output, dt_myr=.25, *, resume=None, fault_snapshot=None,
                  common_omega=None, inherit_legacy_polarity=False):
    """Attempt exactly one atomic step; blocked attempts publish no checkpoint."""
    if isinstance(dt_myr, bool) or not math.isfinite(float(dt_myr)) or dt_myr <= 0.:
        raise ValueError("Probe requires a positive finite time step")
    dt_myr = float(dt_myr)
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Probe output must be a new directory")
    code = _code_hashes()
    mesh, checkpoint, fracture, model, experiment = load_probe_source(source)
    experiment = dict(experiment)
    omega = np.asarray([p.euler_axis*p.angular_speed_rad_per_myr for p in checkpoint.system.plates], dtype=float)
    if common_omega is not None:
        vector = np.asarray(common_omega, dtype=float)
        if vector.shape != (3,) or not np.isfinite(vector).all():
            raise ValueError("Common rotation requires three finite components")
        omega[:] = vector
    experiment.update(format=FORMAT, scope=SCOPE, fixed_omega_rad_per_myr=omega.tolist(),
                      dt_myr=dt_myr, inherit_legacy_polarity=bool(inherit_legacy_polarity))
    original = surface_from_lithosphere(mesh, checkpoint.state, experiment["radius_km"],
                                       fracture_memory=fracture.memory)
    factory = make_birth_factory(original, model, experiment)
    source_state = from_fractional_surface(mesh, original, experiment["radius_km"])
    input_hashes = dict(experiment["source_sha256"])
    if resume is None:
        state, inventory = source_state, initialize_boundary(source_state)
        external, unresolved_lineage = (), []
        initialization = dict(imported_material=False, imported_force=False)
        if inherit_legacy_polarity:
            from tectonics.geometric_polarity_source import legacy_polarity_evidence
            memory = checkpoint.subduction_memory
            external, initialization = legacy_polarity_evidence(mesh, state,
                None if memory is None else memory.young_boundary_state)
        origin = totals(state.fragments)
        lost = {key: 0. for key in EXTENSIVE_FIELDS}
        born, history = dict(lost), []
    else:
        resume = Path(resume).resolve()
        input_hashes[str(resume)] = digest(resume)
        state, inventory, saved = load_boundary_checkpoint(resume)
        _validate_resume(state, inventory, saved, source_state, experiment)
        external = tuple(PolarityEvidence(**item) for item in saved["external_polarity_evidence"])
        origin, lost, born, history, initialization, unresolved_lineage = (saved[name] for name in
            ("initial_totals", "cumulative_losses", "cumulative_births", "history", "polarity_initialization",
             "unresolved_polarity_lineage"))

    snapshot_metadata, faults = None, ()
    if fault_snapshot is not None:
        snapshot_path = Path(fault_snapshot).resolve()
        snapshot_hash = digest(snapshot_path)
        input_hashes[str(snapshot_path)] = snapshot_hash
        faults = fault_snapshot_from_dict(json.loads(snapshot_path.read_text(encoding="utf-8")), state)
        snapshot_metadata = dict(sha256=snapshot_hash, surface_sha256=surface_digest(state),
                                 time_myr=state.time_myr, reuse_permitted=False)
    evidence = _evidence(inventory, external)
    initial_polarity = contact_polarity_report(inventory.contacts, evidence)
    resolver, initiation = make_forced_underthrust_resolver(state, omega, dt_myr,
        faults=faults, contacts=inventory.contacts, history=evidence)
    segment_start, history_start = state.time_myr, len(history)
    output.mkdir(parents=True)
    started, unresolved = time.perf_counter(), None
    try:
        result = advance_geometric_surface(mesh, state, omega, dt_myr,
            birth_factory=factory, polarity_resolver=resolver)
    except UnresolvedPolarityError as error:
        unresolved = dict(attempted_step=1, last_valid_time_myr=state.time_myr,
                          overlaps=[asdict(item) for item in error.overlaps])
    else:
        advanced_inventory = consume_geometric_transaction(state, result, inventory, omega, dt_myr)
        used = {identity for loss in result.losses for identity in loss.polarity_evidence_ids}
        accepted = tuple(PolarityEvidence(**item) for item in initiation["accepted_evidence"]
                         if item["evidence_id"] in used)
        # Existing zero-mass conditions stay local; only fault directions that
        # actually produced accepted loss enter memory. Stress never does.
        remembered = {item.evidence_id: item for item in (*external, *accepted)}
        advanced_external, missing = transport_polarity_evidence(state, result.state, omega, dt_myr,
            contacts_before=inventory.contacts, contacts_after=advanced_inventory.contacts,
            evidence=tuple(remembered.values()))
        state, inventory, external = result.state, advanced_inventory, advanced_external
        unresolved_lineage.extend(missing)
        for key, value in totals(loss.fragment for loss in result.losses).items():
            lost[key] += value
        for key, value in totals(result.births).items():
            born[key] += value
        retained = totals(state.fragments)
        residuals = {key: (retained[key]+lost[key]-born[key]-origin[key])/
                     max(abs(origin[key]), abs(born[key]), 1.) for key in EXTENSIVE_FIELDS}
        if any(abs(value) > 5e-12 for value in residuals.values()):
            raise RuntimeError("Surface/passive slab cumulative ledger does not close")
        history.append(dict(time_myr=state.time_myr, fragments=len(state.fragments),
            cohorts=len(inventory.cohorts), relative_residuals=residuals,
            transport=result.diagnostics, boundary=boundary_diagnostics(inventory),
            polarity=contact_polarity_report(inventory.contacts, _evidence(inventory, external)),
            initiation=dict(policy=FORMAT, fault_snapshot=snapshot_metadata,
                            used_fault_evidence_ids=sorted(item.evidence_id for item in accepted))))

    saved = dict(experiment=experiment, initial_totals=origin, cumulative_losses=lost,
        cumulative_births=born, history=history, polarity_initialization=initialization,
        external_polarity_evidence=[asdict(item) for item in external],
        unresolved_polarity_lineage=unresolved_lineage)
    report = dict(format=FORMAT, status="unresolved_initiation" if unresolved else "complete", **saved,
        initial_polarity=initial_polarity, initiation=initiation, fault_snapshot=snapshot_metadata,
        segment_start_time_myr=segment_start, completed_steps=len(history)-history_start, requested_steps=1,
        passive_boundary=boundary_diagnostics(inventory), unresolved=unresolved,
        resumed_from=None if resume is None else str(resume),
        limitations=["Only declared oriented faults are assessed; no fault nucleation or stress solution",
            "Effective stress is a single-step frozen external snapshot, never restored for another step",
            "Directional history alone cannot admit a mechanically locked contact",
            "No old slab material or forces are imported",
            "Missing or conflicting directional/loading evidence blocks the atomic step",
            "Endpoint geometry and age with frozen heat/damage; no self-sustaining subduction or force feedback"])
    if any(digest(path) != expected for path, expected in input_hashes.items()):
        raise RuntimeError("Immutable source, resume checkpoint or fault snapshot changed during the probe")
    report["checkpoint_file"], report["checkpoint_sha256"] = None, None
    if unresolved is None:
        filename = "boundary_checkpoint.json"
        save_boundary_checkpoint(output/filename, state, inventory, provenance=saved)
        report["checkpoint_file"], report["checkpoint_sha256"] = filename, digest(output/filename)
    report.update(source_unchanged=True, input_sha256=input_hashes,
                  production_sha256=code, production_sha256_after_run=_code_hashes(),
                  wall_seconds=time.perf_counter()-started)
    report["code_changed_during_run"] = code != report["production_sha256_after_run"]
    (output/"report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dt-myr", type=float, default=.25)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--fault-snapshot", type=Path,
                        help="Explicit weak-fault orientations/loading bound to the current surface and time")
    parser.add_argument("--common-omega", nargs=3, type=float)
    parser.add_argument("--inherit-legacy-polarity", action="store_true",
                        help="Inherit local directional conditions without importing old material or forces")
    args = parser.parse_args(argv)
    try:
        report = execute_probe(args.source, args.out, args.dt_myr, resume=args.resume,
            fault_snapshot=args.fault_snapshot, common_omega=args.common_omega,
            inherit_legacy_polarity=args.inherit_legacy_polarity)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    print(json.dumps(dict(output=str(args.out.resolve()), status=report["status"],
        completed_steps=report["completed_steps"], passive_boundary=report["passive_boundary"],
        wall_seconds=report["wall_seconds"]), indent=2))
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
