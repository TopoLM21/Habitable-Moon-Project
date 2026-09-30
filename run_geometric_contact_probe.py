"""Independent spherical contact geometry experiment with fixed plate motion.

Material footprints, actual contact arcs and their material ledger are saved.
Heat, forces, fracture and the GUI are not advanced by this experiment.
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

from run_fractional_transport_probe import (DEFAULT_SOURCE, EXTENSIVE, ROOT,
    digest, load_probe_source, make_birth_factory)


FORMAT = "geometric-contact-probe-1"
SCOPE = ("Geometry only: prescribed fixed plate rotations; material age advances; "
         "thermal properties and damage are frozen; no force or fracture evolution")


def _write_report(output, report):
    (output/"report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def _verify_source(provenance):
    if any(digest(path) != expected for path, expected in provenance["source_sha256"].items()):
        raise RuntimeError("Geometric probe source changed during the experiment")


def _zero_totals():
    return {name: 0. for name in EXTENSIVE}


def _production_hashes():
    paths = [Path(__file__), ROOT/"tectonics/spherical_polygons.py",
             *sorted((ROOT/"tectonics").glob("geometric_*.py"))]
    return {str(path.relative_to(ROOT)).replace("\\", "/"): digest(path) for path in paths}


def projection_summary(mesh, state):
    """Integrate the actual footprints into cells without changing contact geometry."""
    from tectonics.geometric_surface import project_to_mesh, totals
    from tectonics.spherical_polygons import GeometryDiagnostics
    diagnostics = GeometryDiagnostics()
    view = project_to_mesh(mesh, state, diagnostics=diagnostics)
    areas = np.zeros(mesh.cell_count)
    owners = [set() for _ in range(mesh.cell_count)]
    counts = np.zeros(mesh.cell_count, dtype=int)
    for fragment in view:
        cell = fragment.parcel.cell
        areas[cell] += fragment.parcel.area_km2
        owners[cell].add(fragment.parcel.plate)
        counts[cell] += 1
    expected = mesh.physical_cell_areas_km2(state.radius_km)
    before, after = totals(state.fragments), totals(view)
    material_errors = {}
    for name in EXTENSIVE:
        if before[name] == 0.:
            if after[name] != 0.:
                raise ValueError("Geometric projection creates material from an empty reservoir")
            material_errors[name] = 0.
        else:
            material_errors[name] = (after[name]-before[name])/abs(before[name])
    maximum_cell_error = float(np.max(np.abs(areas-expected)/expected))
    global_error = float((math.fsum(areas)-math.fsum(expected))/math.fsum(expected))
    if (maximum_cell_error > 5e-12 or abs(global_error) > 5e-12
            or any(abs(value) > 5e-12 for value in material_errors.values())):
        raise ValueError("Geometric projection coverage or material budget does not close")
    return dict(projected_fragment_count=len(view),
                mixed_plate_cells=sum(len(values) > 1 for values in owners),
                multi_fragment_cells=int(np.count_nonzero(counts > 1)),
                maximum_fragments_per_cell=int(np.max(counts)),
                uncovered_cells=int(np.count_nonzero(counts == 0)),
                maximum_cell_relative_coverage_error=maximum_cell_error,
                global_relative_coverage_error=global_error,
                material_relative_projection_errors=material_errors,
                arithmetic_diagnostics=asdict(diagnostics))


def execute_probe(source, output, dt_myr, steps, *, resume=None, common_omega=None):
    """Run fixed motion, returning a report; unresolved polarity has no final checkpoint."""
    startup_production = _production_hashes()
    from tectonics.fractional_surface_io import surface_from_lithosphere
    from tectonics.geometric_contacts import extract_contacts
    from tectonics.geometric_surface import (advance_geometric_surface, audit_partition, from_fractional_surface,
        load_geometric_checkpoint, rotate_surface, save_geometric_checkpoint, totals)
    from tectonics.geometric_transport import UnresolvedPolarityError

    if (isinstance(dt_myr, bool) or not math.isfinite(float(dt_myr)) or dt_myr <= 0.
            or isinstance(steps, bool) or not isinstance(steps, Integral) or steps <= 0):
        raise ValueError("Geometric probe requires a finite positive dt and positive integer steps")
    dt_myr, steps = float(dt_myr), int(steps)
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Probe output must be new; existing results are never overwritten")
    mesh, checkpoint, fracture, model, provenance = load_probe_source(source)
    provenance = dict(provenance)
    radius = provenance["radius_km"]
    omega = np.asarray([p.euler_axis*p.angular_speed_rad_per_myr for p in checkpoint.system.plates], dtype=float)
    if common_omega is not None:
        vector = np.asarray(common_omega, dtype=float)
        if vector.shape != (3,) or not np.isfinite(vector).all():
            raise ValueError("Common rotation requires three finite components in rad/Myr")
        omega[:] = vector
    provenance.update(scope=SCOPE, fixed_omega_rad_per_myr=omega.tolist(), dt_myr=dt_myr,
                      experiment_format=FORMAT,
                      motion_mode="explicit_common_rotation" if common_omega is not None else "source_plate_rotations")
    source_surface = surface_from_lithosphere(mesh, checkpoint.state, radius, fracture_memory=fracture.memory)
    if resume is None:
        state = from_fractional_surface(mesh, source_surface, radius)
        initial_totals, lost, born = totals(state.fragments), _zero_totals(), _zero_totals()
        history, losses_archive, births_archive = [], [], []
    else:
        state, saved = load_geometric_checkpoint(resume)
        if saved.get("experiment") != provenance:
            raise ValueError("Geometric resume requires identical source, motion, timestep and birth parameters")
        if state.phase != "partition" or state.time_myr < provenance["source_time_myr"]:
            raise ValueError("Geometric resume requires a valid partition at or after the source time")
        initial_totals, lost, born, history, losses_archive, births_archive = (
            saved[name] for name in ("initial_totals", "cumulative_losses", "cumulative_births",
                                    "history", "losses", "births"))
        if state.radius_km != radius:
            raise ValueError("Geometric resume radius differs from the source")
    factory = make_birth_factory(source_surface, model, provenance)
    segment_start = state.time_myr
    initial_history_count = len(history)
    start = time.perf_counter()
    output.mkdir(parents=True)
    unresolved = None
    contacts = ()
    for index in range(steps):
        try:
            if common_omega is not None:
                state = rotate_surface(state, omega, dt_myr)
                losses, births = (), ()
                diagnostics = dict(common_rotation=True)
            else:
                result = advance_geometric_surface(mesh, state, omega, dt_myr, birth_factory=factory)
                state = result.state
                losses, births = result.losses, result.births
                diagnostics = result.diagnostics
                contacts = result.contacts
        except UnresolvedPolarityError as error:
            unresolved = dict(message=str(error), attempted_step=index+1,
                              last_valid_time_myr=state.time_myr,
                              overlaps=[asdict(overlap) for overlap in error.overlaps])
            break
        for name, value in totals(loss.fragment for loss in losses).items():
            lost[name] += value
        for name, value in totals(births).items():
            born[name] += value
        losses_archive.extend(asdict(loss) for loss in losses)
        births_archive.extend(asdict(fragment) for fragment in births)
        remaining = totals(state.fragments)
        residuals = {name: (remaining[name]+lost[name]-born[name]-initial_totals[name]) /
                     max(abs(initial_totals[name]), abs(born[name]), 1.) for name in EXTENSIVE}
        if any(abs(value) > 5e-12 for value in residuals.values()):
            raise RuntimeError("Geometric material ledger does not close")
        history.append(dict(time_myr=state.time_myr, fragment_count=len(state.fragments), retained=remaining,
                            cumulative_losses=dict(lost), cumulative_births=dict(born),
                            relative_residuals=residuals, diagnostics=diagnostics))
    _verify_source(provenance)
    saved = dict(experiment=provenance, initial_totals=initial_totals,
                 cumulative_losses=lost, cumulative_births=born,
                 history=history, losses=losses_archive, births=births_archive)
    report = dict(format=FORMAT, status="unresolved_polarity" if unresolved else "complete", **saved,
                  segment_start_time_myr=segment_start, dt_myr=dt_myr, requested_steps=steps,
                  completed_steps=len(history)-initial_history_count,
                  resumed_from=None if resume is None else str(Path(resume).resolve()),
                  source_unchanged=True,
                  limitations=[
                      "Independent fixed-motion geometry experiment; not a coupled speed prediction",
                      "Material fractions without persisted geometry cannot be reconstructed exactly",
                      "Heat and damage are frozen; newborn material is created at the end of each geometric step",
                      "Physical ties in overlap polarity remain explicit unresolved cases",
                      "Contact segment identities survive rigid motion; mechanical split/merge attachment is not implemented",
                      "Loss archive is passive and does not activate slab or ridge forces"])
    if unresolved is not None:
        report["unresolved"] = unresolved
        report["checkpoint_sha256"] = None
        if len(history) > initial_history_count:
            audit_partition(state)
            saved["contacts"] = [asdict(contact) for contact in extract_contacts(state.fragments, radius, omega)]
            last_valid = output/"last_valid_geometric_checkpoint.json"
            save_geometric_checkpoint(last_valid, state, provenance=saved)
            report["last_valid_checkpoint_sha256"] = digest(last_valid)
    else:
        audit_partition(state)
        if common_omega is not None:
            contacts = extract_contacts(state.fragments, radius, omega)
        records = [asdict(contact) for contact in contacts]
        report["projection"] = projection_summary(mesh, state)
        report["contacts"] = records
        report["contact_count"] = len(records)
        report["total_contact_length_km"] = math.fsum(contact.length_km for contact in contacts)
        report["attached_loss_count"] = sum(bool(loss["contact_ids"]) for loss in losses_archive)
        report["unresolved_attachment_loss_count"] = sum(not loss["contact_ids"] for loss in losses_archive)
        saved["contacts"] = records
        save_geometric_checkpoint(output/"geometric_checkpoint.json", state, provenance=saved)
        report["checkpoint_sha256"] = digest(output/"geometric_checkpoint.json")
    _verify_source(provenance)
    report["wall_seconds"] = time.perf_counter()-start
    report["production_sha256"] = startup_production
    report["production_sha256_after_run"] = _production_hashes()
    report["code_changed_during_run"] = report["production_sha256_after_run"] != startup_production
    _write_report(output, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dt-myr", type=float, default=.25)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--common-omega", nargs=3, type=float, metavar=("X", "Y", "Z"),
                        help="Give every plate this rigid rotation in rad/Myr; geometry conservation control")
    parser.add_argument("--resume", type=Path,
                        help="Geometric checkpoint; source, timestep, motion and birth parameters must match")
    args = parser.parse_args(argv)
    try:
        report = execute_probe(args.source, args.out, args.dt_myr, args.steps,
                               resume=args.resume, common_omega=args.common_omega)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    print(json.dumps(dict(output=str(args.out.resolve()), status=report["status"],
                          completed_steps=report["completed_steps"], contact_count=report.get("contact_count"),
                          projection=report.get("projection"), source_unchanged=report["source_unchanged"],
                          wall_seconds=report["wall_seconds"]), ensure_ascii=False, indent=2))
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
