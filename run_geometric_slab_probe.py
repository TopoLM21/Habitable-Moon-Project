"""Fixed-motion geometric surface with local polarity and passive slab cohorts.

No slab/ridge force or thermal evolution is activated. Inherited raster polarity
is an explicit opt-in boundary condition, not solved subduction initiation.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from numbers import Integral
from pathlib import Path
import time

import numpy as np

from run_fractional_transport_probe import DEFAULT_SOURCE, ROOT, digest, load_probe_source, make_birth_factory
from tectonics.fractional_surface import EXTENSIVE_FIELDS
from tectonics.fractional_surface_io import surface_from_lithosphere
from tectonics.geometric_boundary import (initialize_boundary, consume_geometric_transaction,
    boundary_diagnostics, polarity_evidence)
from tectonics.geometric_boundary_io import save_boundary_checkpoint, load_boundary_checkpoint
from tectonics.geometric_polarity import (PolarityEvidence, contact_polarity_report,
    make_polarity_resolver, transport_polarity_evidence)
from tectonics.geometric_surface import from_fractional_surface, totals
from tectonics.geometric_transport import advance_geometric_surface, UnresolvedPolarityError

FORMAT = "geometric-passive-slab-probe-1"
SCOPE = "Fixed motion; endpoint geometry and age; frozen thermal/damage properties; passive slab accounting; no forces"


def _code_hashes():
    return {path.relative_to(ROOT).as_posix(): digest(path) for path in
            [Path(__file__), ROOT/"tectonics/spherical_polygons.py", *sorted((ROOT/"tectonics").glob("geometric_*.py"))]}


def _evidence(inventory, external):
    result = list(external)
    for record in polarity_evidence(inventory):
        result.append(record if isinstance(record, PolarityEvidence) else PolarityEvidence(**record))
    return tuple(result)


def execute_probe(source, output, dt_myr, steps, *, resume=None, common_omega=None,
                  inherit_legacy_polarity=False):
    if (isinstance(dt_myr, bool) or not math.isfinite(float(dt_myr)) or dt_myr <= 0.
            or isinstance(steps, bool) or not isinstance(steps, Integral) or steps <= 0):
        raise ValueError("Probe requires positive finite dt and positive integer steps")
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Probe output must be a new directory")
    dt_myr = float(dt_myr)
    code = _code_hashes()
    mesh, checkpoint, fracture, model, experiment = load_probe_source(source)
    experiment = dict(experiment)
    omega = np.asarray([p.euler_axis*p.angular_speed_rad_per_myr for p in checkpoint.system.plates], dtype=float)
    if common_omega is not None:
        vector = np.asarray(common_omega, dtype=float)
        if vector.shape != (3,) or not np.isfinite(vector).all():
            raise ValueError("Common rotation requires three finite components")
        omega[:] = vector
    experiment.update(format=FORMAT, scope=SCOPE, fixed_omega_rad_per_myr=omega.tolist(), dt_myr=dt_myr,
                      inherit_legacy_polarity=bool(inherit_legacy_polarity))
    original = surface_from_lithosphere(mesh, checkpoint.state, experiment["radius_km"],
                                        fracture_memory=fracture.memory)
    factory = make_birth_factory(original, model, experiment)
    if resume is None:
        state = from_fractional_surface(mesh, original, experiment["radius_km"])
        inventory = initialize_boundary(state)
        external, unresolved_lineage, initialization = (), [], {"imported_material": False, "imported_force": False}
        if inherit_legacy_polarity:
            from tectonics.geometric_polarity_source import legacy_polarity_evidence
            memory = checkpoint.subduction_memory
            old_inventory = None if memory is None else memory.young_boundary_state
            external, initialization = legacy_polarity_evidence(mesh, state, old_inventory)
        origin = totals(state.fragments)
        lost = {key: 0. for key in EXTENSIVE_FIELDS}
        born, history = dict(lost), []
    else:
        state, inventory, saved = load_boundary_checkpoint(resume)
        if saved.get("experiment") != experiment:
            raise ValueError("Resume requires identical source, motion, time step and polarity initialization")
        external = tuple(PolarityEvidence(**item) for item in saved["external_polarity_evidence"])
        origin, lost, born, history, initialization, unresolved_lineage = (saved[name] for name in
            ("initial_totals", "cumulative_losses", "cumulative_births", "history", "polarity_initialization",
             "unresolved_polarity_lineage"))
    history_start = len(history)
    segment_start = state.time_myr
    output.mkdir(parents=True)
    started = time.perf_counter()
    unresolved = None
    initial_polarity = contact_polarity_report(inventory.contacts, _evidence(inventory, external))
    for index in range(steps):
        evidence = _evidence(inventory, external)
        resolver = make_polarity_resolver(state, omega, dt_myr, contacts=inventory.contacts, evidence=evidence) if evidence else None
        try:
            result = advance_geometric_surface(mesh, state, omega, dt_myr,
                                               birth_factory=factory, polarity_resolver=resolver)
        except UnresolvedPolarityError as error:
            unresolved = dict(attempted_step=index+1, last_valid_time_myr=state.time_myr,
                              overlaps=[asdict(item) for item in error.overlaps])
            break
        advanced_inventory = consume_geometric_transaction(state, result, inventory, omega, dt_myr)
        advanced_evidence, missing = transport_polarity_evidence(state, result.state, omega, dt_myr,
            contacts_before=inventory.contacts, contacts_after=advanced_inventory.contacts, evidence=external)
        state, inventory, external = result.state, advanced_inventory, advanced_evidence
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
            polarity=contact_polarity_report(inventory.contacts, _evidence(inventory, external))))
    saved = dict(experiment=experiment, initial_totals=origin, cumulative_losses=lost,
                 cumulative_births=born, history=history, polarity_initialization=initialization,
                 external_polarity_evidence=[asdict(e) for e in external],
                 unresolved_polarity_lineage=unresolved_lineage)
    report = dict(format=FORMAT, status="unresolved_polarity" if unresolved else "complete", **saved,
        initial_polarity=initial_polarity, segment_start_time_myr=segment_start,
        completed_steps=len(history)-history_start, requested_steps=int(steps),
        passive_boundary=boundary_diagnostics(inventory), unresolved=unresolved,
        resumed_from=None if resume is None else str(Path(resume).resolve()),
        limitations=["Local oriented memory and passive accounting, not solved subduction initiation",
            "Inherited raster direction is an explicitly selected legacy boundary condition",
            "No old slab mass is imported; the legacy reservoir remains in the immutable source",
            "Ambiguous lateral mass allocation cannot exert a force",
            "Missing geometric lineage is unresolved connectivity, not physical breakoff",
            "Endpoint geometry with frozen heat/damage; gauss2 thermal coupling and force feedback remain separate"])
    if any(digest(path) != expected for path, expected in experiment["source_sha256"].items()):
        raise RuntimeError("Saved source changed during the probe")
    if unresolved is None or len(history) > history_start:
        filename = "boundary_checkpoint.json" if unresolved is None else "last_valid_boundary_checkpoint.json"
        save_boundary_checkpoint(output/filename, state, inventory, provenance=saved)
        report["checkpoint_file"], report["checkpoint_sha256"] = filename, digest(output/filename)
    else:
        report["checkpoint_file"], report["checkpoint_sha256"] = None, None
    report.update(source_unchanged=True, production_sha256=code, production_sha256_after_run=_code_hashes(),
                  wall_seconds=time.perf_counter()-started)
    report["code_changed_during_run"] = code != report["production_sha256_after_run"]
    (output/"report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dt-myr", type=float, default=.25)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--common-omega", nargs=3, type=float)
    parser.add_argument("--inherit-legacy-polarity", action="store_true",
                        help="Explicitly inherit local raster-history direction, retaining all conflicts; no old mass import")
    args = parser.parse_args(argv)
    try:
        result = execute_probe(args.source, args.out, args.dt_myr, args.steps, resume=args.resume,
            common_omega=args.common_omega, inherit_legacy_polarity=args.inherit_legacy_polarity)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    print(json.dumps(dict(output=str(args.out.resolve()), status=result["status"],
        completed_steps=result["completed_steps"], passive_boundary=result["passive_boundary"],
        wall_seconds=result["wall_seconds"]), indent=2))
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
