"""Independent native force candidate and atomic kinematic-refusal control.

The genuine declared affine source produces a mixed normal-velocity sign on
one actual finite-SO(3) receiver arc. The attempted loaded interval refuses;
accepted native loaded intervals remain zero. Work below belongs solely to
the unaccepted candidate and never enters the persistent material/work ledger.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from tectonics.fractional_surface import EXTENSIVE_FIELDS
from tectonics.thermal import SECONDS_PER_MYR


def encoded(raw):
    return json.dumps(raw, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def write_json(path, raw):
    with Path(path).open("xb") as stream:
        stream.write(encoded(raw) + b"\n")


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_snapshot():
    paths = set()
    for directory in ("tectonics", "tests", "scripts"):
        paths.update((ROOT / directory).rglob("*.py"))
    paths.add(Path(__file__).resolve())
    paths.add(ROOT / "analysis/geometric_underthrust_validation/underthrust_case.py")
    return {str(path.relative_to(ROOT)): file_sha256(path) for path in sorted(paths)}


def independent_source_work(surface, load, increments):
    """Integrate traction work directly, without assembled torque/operators."""
    radius = np.longdouble(surface.radius_km * 1000.)
    motion = {owner: np.asarray(row, dtype=np.longdouble)
              for owner, row in zip(load.plate_ids, increments)}
    contributions = []
    for sample in load.samples:
        displacement = radius * np.cross(motion[sample.plate_id],
            np.asarray(sample.position, dtype=np.longdouble))
        contributions.append(np.longdouble(sample.area_m2) * (
            np.asarray(sample.source_traction_pa, dtype=np.longdouble) @ displacement))
    return float(sum(contributions, np.longdouble(0.)))


def independent_basal_dissipation(surface, load, increments, dt_s):
    """Integrate positive drag work directly over the physical source samples."""
    radius = np.longdouble(surface.radius_km * 1000.)
    coefficient = np.longdouble(load.source.basal_drag_pa_s_m)
    motion = {owner: np.asarray(row, dtype=np.longdouble)
              for owner, row in zip(load.plate_ids, increments)}
    contributions = []
    for sample in load.samples:
        displacement = radius * np.cross(motion[sample.plate_id],
            np.asarray(sample.position, dtype=np.longdouble))
        contributions.append(coefficient * np.longdouble(sample.area_m2)
                             * (displacement @ displacement) / np.longdouble(dt_s))
    return float(sum(contributions, np.longdouble(0.)))


def parcel_ledger(parcels):
    groups = {}
    for parcel in parcels:
        for field in EXTENSIVE_FIELDS:
            groups.setdefault((parcel.material_id, field), []).append(getattr(parcel, field))
    return {key: math.fsum(values) for key, values in groups.items()}


def independent_material_ledger(original, retained, losses, births):
    ledgers = [parcel_ledger(parcels) for parcels in (original, retained, losses, births)]
    rows = []
    maximum = 0.
    for material, field in sorted(set().union(*(ledger.keys() for ledger in ledgers))):
        old, live, buried, born = [ledger.get((material, field), 0.) for ledger in ledgers]
        left, right = math.fsum((live, buried)), math.fsum((old, born))
        if not math.isclose(left, right, rel_tol=5e-12, abs_tol=0.):
            raise AssertionError(f"Independent original/born/live/buried budget failed: {material}, {field}")
        residual = abs(left-right) / max(abs(left), abs(right)) if left or right else 0.
        maximum = max(maximum, residual)
        rows.append(dict(material_id=material, field=field, initial=old, live=live,
                         buried=buried, born=born, relative_residual=residual))
    return dict(maximum_relative_residual=maximum, reservoirs=rows)


def relative_error(actual, expected):
    return abs(actual-expected) / max(abs(actual), abs(expected), np.finfo(float).tiny)


def state_snapshot(state):
    from tectonics.geometric_native_buried_support import native_buried_registry_to_dict
    from tectonics.geometric_native_surface import native_surface_to_dict
    return encoded(dict(surface=native_surface_to_dict(state.current_surface),
        registry=native_buried_registry_to_dict(state.current_inventory),
        initial_inventory=asdict(state.initial_inventory), initial_load=asdict(state.initial_load),
        initial_mechanics=asdict(state.initial_mechanics),
        owner_histories=[dict(owner=owner, history=[asdict(step) for step in history])
                         for owner, history in state.owner_histories],
        nodes=[(identity, asdict(local)) for identity, local in state.states],
        radial=state.transverse_displacement_m, anchor_fields=state.free_anchor_fields,
        spatial_memory=asdict(state.spatial_memory), receipts=[asdict(row) for row in state.receipts],
        elapsed_time_s=state.elapsed_time_s, source_work_j=state.cumulative_source_work_j,
        basal_dissipation_j=state.cumulative_basal_dissipation_j,
        bending_dissipation_j=state.cumulative_bending_numerical_dissipation_j))


def attempt_control(mesh, origin, state, load, dt_s):
    from tectonics.geometric_native_polygons import native_polygon_to_dict
    from tectonics.geometric_native_underthrust_motion import (
        UnresolvedNativeUnderthrustMotion, advance_native_underthrust_motion,
    )
    before = state_snapshot(state)
    try:
        advance_native_underthrust_motion(mesh, origin, state, load, dt_s,
            provenance="Independent affine source drives the original native material pairs")
    except UnresolvedNativeUnderthrustMotion as refusal:
        candidate = refusal.candidate
        unresolved = refusal.overlaps
    else:
        raise AssertionError("The genuine mixed-convergence candidate was unexpectedly accepted")
    after = state_snapshot(state)
    if before != after or state.receipts or state.current_inventory.records:
        raise AssertionError("An unaccepted force candidate changed the persistent native trajectory")
    metrics = dict(candidate.candidate_diagnostics)
    increments = np.asarray(candidate.rotation_increment_radians)
    if np.linalg.norm(increments[0]-increments[1]) <= 0. or not np.any(increments[:, :2] != 0.):
        raise AssertionError("Full force-generated differential Euler components were changed")
    work = independent_source_work(state.current_surface, load, increments)
    dissipation = independent_basal_dissipation(state.current_surface, load, increments, dt_s)
    work_error = relative_error(work, metrics["source_work_j"])
    dissipation_error = relative_error(dissipation, metrics["basal_dissipation_j"])
    if work_error >= 3e-14 or dissipation_error >= 3e-14:
        raise AssertionError("Independent candidate sample work disagrees with the solved physical forces")
    for name in ("cumulative_energy_balance_relative_error", "material_reaction_relative_error",
                 "angular_signal_relative_residual", "transverse_force_relative_residual"):
        if metrics[name] >= 5e-10:
            raise AssertionError(f"Unaccepted physical candidate balance failed: {name}")
    if metrics["anchor_transport_error_m"] > state.max_anchor_transport_error_m:
        raise AssertionError("Candidate finite-pose anchor error exceeds its declared physical bound")
    overlaps = [dict(fragment_a=item.fragment_a, fragment_b=item.fragment_b,
        plate_a=item.plate_a, plate_b=item.plate_b, area_km2=item.area_km2, reason=item.reason,
        geometry=native_polygon_to_dict(item.geometry)) for item in unresolved]
    if len(overlaps) != 1 or overlaps[0]["reason"] != "mixed_convergence":
        raise AssertionError("The true finite-kinematic refusal changed its physical reason")
    ledger = independent_material_ledger(
        (fragment.parcel for fragment in origin.fragments),
        (fragment.parcel for fragment in state.current_surface.fragments), (), ())
    if ledger["maximum_relative_residual"] != 0.:
        raise AssertionError("Refusal changed an accepted original material reservoir")
    return dict(status="expected_atomic_refusal", accepted_loaded_native_intervals=0,
        before_state_sha256=hashlib.sha256(before).hexdigest(), input_unchanged=True,
        candidate=dict(before_surface_sha256=candidate.before_surface_sha256,
            current_load_sha256=candidate.current_load_sha256, dt_s=candidate.dt_s,
            provenance=candidate.provenance,
            rotation_increment_radians=candidate.rotation_increment_radians,
            angular_velocity_rad_s=candidate.angular_velocity_rad_s,
            omega_rad_per_myr=candidate.omega_rad_per_myr,
            states=[(identity, asdict(local)) for identity, local in candidate.states],
            transverse_displacement_m=candidate.transverse_displacement_m,
            spatial_step=asdict(candidate.spatial_step), diagnostics=metrics,
            physical_points=candidate.points),
        candidate_source_work_j=work, independent_candidate_source_work_relative_error=work_error,
        candidate_basal_dissipation_j=dissipation,
        independent_candidate_basal_dissipation_relative_error=dissipation_error,
        accepted_material_ledger=ledger,
        admission=candidate.admission_audit,
        omega={str(owner):list(value) for owner,value in candidate.omega_rad_per_myr},
        unresolved=overlaps)


def run_control(out):
    from analysis.geometric_underthrust_validation.underthrust_case import (
        ANCHOR_ERROR_BOUND_M, DT_S, SOURCE_INTERVALS, physical_inputs,
    )
    from tectonics.geometric_native_mantle_loading import (
        assemble_native_physical_spatial_mantle_load, native_spatial_mantle_load_to_dict,
    )
    from tectonics.geometric_native_underthrust_motion import initialize_native_underthrust_motion
    from tectonics.geometric_native_underthrust_motion_io import (
        load_native_underthrust_motion_checkpoint, save_native_underthrust_motion_checkpoint,
    )
    from tectonics.geometric_underthrust_motion import HotBirthParameters

    mesh, origin, inventory, seed, initial_load = physical_inputs()
    write_json(out/"literal_legacy_origin.json", asdict(origin))
    write_json(out/"declared_sources.json", [asdict(source) for source in SOURCE_INTERVALS])
    metadata = dict(control="native force candidate with exact finite-kinematic refusal",
        accepted_loaded_native_intervals=0, cooling="frozen", source_selected_before_force=True)
    state = initialize_native_underthrust_motion(mesh, origin, inventory, seed, initial_load,
        birth_parameters=HotBirthParameters(2., 0., 0.,
            "Declared hot departure-footprint closure for the native physical control"),
        max_anchor_transport_error_m=ANCHOR_ERROR_BOUND_M)
    initial = save_native_underthrust_motion_checkpoint(out/"initial.json", mesh, origin, state,
                                                       provenance=metadata)
    print("Assembling the independently prescribed source on certified native footprints", flush=True)
    load = assemble_native_physical_spatial_mantle_load(mesh, state.current_surface, SOURCE_INTERVALS[0])
    write_json(out/"initial_native_source.json", native_spatial_mantle_load_to_dict(state.current_surface, load))
    print("Solving the full force candidate and proving the real mixed-convergence refusal", flush=True)
    direct = attempt_control(mesh, origin, state, load, DT_S)
    probe = out/"force_candidate_refusal_probe.json"
    write_json(probe, direct)
    after_direct = save_native_underthrust_motion_checkpoint(out/"after_direct_refusal.json", mesh,
        origin, state, provenance=metadata)
    print("Replaying the prepared joint checkpoint and reproducing the full atomic refusal", flush=True)
    restored_origin, restored, restored_metadata = load_native_underthrust_motion_checkpoint(initial,
                                                                                           mesh=mesh)
    resaved = save_native_underthrust_motion_checkpoint(out/"initial_resaved.json", mesh, restored_origin,
        restored, provenance=restored_metadata)
    restored_load = assemble_native_physical_spatial_mantle_load(mesh, restored.current_surface,
                                                                SOURCE_INTERVALS[0])
    resumed = attempt_control(mesh, restored_origin, restored, restored_load, DT_S)
    write_json(out/"resumed_force_candidate_refusal_probe.json", resumed)
    if encoded(direct) != encoded(resumed):
        raise AssertionError("Direct and resumed candidate force/admission/refusal evidence differs")
    after_resumed = save_native_underthrust_motion_checkpoint(out/"after_resumed_refusal.json", mesh,
        restored_origin, restored, provenance=restored_metadata)
    pairs = []
    for right in (resaved, after_direct, after_resumed):
        if initial.read_bytes() != right.read_bytes():
            raise AssertionError("Prepared joint checkpoint bytes changed on replay or rejected force attempt")
        pairs.append(dict(left=initial.name, right=right.name, byte_equal=True,
                          bytes=len(initial.read_bytes()), sha256=file_sha256(initial)))
    return dict(accepted_loaded_native_intervals=0, attempted_loaded_native_intervals=2,
        physical_control_outcome="exact finite-kinematic admission refuses the genuine force candidate",
        candidate_source_work_relative_error=direct["independent_candidate_source_work_relative_error"],
        candidate_basal_dissipation_relative_error=direct["independent_candidate_basal_dissipation_relative_error"],
        accepted_material_budget_max_relative_residual=0.,
        accepted_source_work_j=state.cumulative_source_work_j,
        accepted_native_receipt_count=len(state.receipts), joint_initial_checkpoint_byte_pairs=pairs,
        force_candidate_refusal_probe=probe.name,
        force_candidate_refusal_probe_sha256=file_sha256(probe),
        unresolved_overlaps=[{key:value for key,value in row.items() if key!="geometry"}
                             for row in direct["unresolved"]],
        direct_resumed_full_candidate_refusal_equal=True,
        next_point_1_requirement="certified swept contact entry and recrossing times with persistent constitutive support")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.out.mkdir(parents=True, exist_ok=False)
    before, started = code_snapshot(), time.monotonic()
    write_json(arguments.out/"code_before.json", before)
    report = dict(status="running", started_utc=datetime.now(timezone.utc).isoformat(),
        command=sys.argv, accepted_loaded_native_intervals=0)
    failure = None
    try:
        report.update(run_control(arguments.out))
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", failure=dict(type=type(error).__name__, message=str(error)))
        failure = error
    after = code_snapshot()
    write_json(arguments.out/"code_after.json", after)
    report.update(finished_utc=datetime.now(timezone.utc).isoformat(),
        wall_seconds=time.monotonic()-started, code_files=len(before), code_unchanged=before==after,
        code_before_sha256=file_sha256(arguments.out/"code_before.json"),
        code_after_sha256=file_sha256(arguments.out/"code_after.json"), runner_sha256=file_sha256(__file__))
    report["final_eligible"] = report["status"] == "passed" and report["code_unchanged"]
    write_json(arguments.out/"report.json", report)
    if failure is not None:
        raise failure
    if not report["code_unchanged"]:
        raise RuntimeError("Native control sources changed while the receipt was running")
    print(json.dumps({key:report[key] for key in ("status", "accepted_loaded_native_intervals",
        "wall_seconds", "final_eligible")}), flush=True)


if __name__ == "__main__":
    main()
