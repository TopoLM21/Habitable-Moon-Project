"""Portable differential geometry controls; no loaded underthrust advance.

The full-vector millisecond case keeps the measured SOURCE omega components.
New hot-material ties use an explicit geometric ordering, with the ordinary
contrast policy retained whenever it resolves polarity.  This declaration is
not a physical material-pair law.  A separate coplanar control exercises cuts
on an exactly invariant supporting plane.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from tectonics.fractional_surface import EXTENSIVE_FIELDS, SurfaceParcel
from tectonics.geometric_native_polygons import native_arc_to_dict, native_polygon_to_dict
from tectonics.geometric_native_surface import (
    advance_native_surface, enroll_native_surface, load_native_checkpoint,
    native_surface_to_dict, save_native_checkpoint,
)
from tectonics.geometric_polarity import PolarityResolution
from tectonics.geometric_transport import UnresolvedPolarityError
from test_geometric_precision import NEXT_DT_MYR, NEXT_OMEGA, thin_partition


def _encoded(raw):
    return json.dumps(raw, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def _write(path, raw):
    with Path(path).open("xb") as stream:
        stream.write(_encoded(raw) + b"\n")


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _snapshot_code():
    paths = set()
    for directory in ("tectonics", "tests", "scripts"):
        paths.update((ROOT / directory).rglob("*.py"))
    paths.add(Path(__file__).resolve())
    return {str(path.relative_to(ROOT)): _hash(path) for path in sorted(paths)}


def _source():
    source = thin_partition()
    return replace(source, fragments=tuple(replace(fragment,
        parcel=replace(fragment.parcel, age_myr=10.)) if fragment.parcel.plate == 1
        else fragment for fragment in source.fragments))


def _birth(cell, plate, area, time_myr, serial):
    return SurfaceParcel(cell, plate, f"native-control-birth:{time_myr.hex()}:{serial}",
                         area, 2. * area, 0., 0., 0., (("damage", 0.),))


def _tie_policy(left, right, geometry):
    for a, b in ((left.parcel.specific_properties[2], right.parcel.specific_properties[2]),
                 (left.parcel.age_myr, right.parcel.age_myr)):
        if not math.isclose(a, b, rel_tol=64. * math.ulp(1.), abs_tol=0.):
            return None
    return PolarityResolution(-1 if left.parcel.plate == 1 else 1,
        "declared_geometric_control_hot_tie_owner_1_remains_surface")


def _coplanar_policy(left, right, geometry):
    return PolarityResolution(-1 if left.parcel.plate == 1 else 1,
                             "declared_geometric_coplanar_control_owner_1_remains_surface")


def _ledger(fragments):
    groups = {}
    for fragment in fragments:
        for field in EXTENSIVE_FIELDS:
            groups.setdefault((fragment.parcel.material_id, field), []).append(
                getattr(fragment.parcel, field))
    return {key: math.fsum(values) for key, values in groups.items()}


def _check_ledgers(before, retained, losses, births):
    original = _ledger(before)
    current = _ledger(retained)
    removed = _ledger(loss.fragment for loss in losses)
    born = _ledger(births)
    rows, maximum = [], 0.
    for key in sorted(original.keys() | current.keys() | removed.keys() | born.keys()):
        left = current.get(key, 0.) + removed.get(key, 0.)
        right = original.get(key, 0.) + born.get(key, 0.)
        if not math.isclose(left, right, rel_tol=5e-12, abs_tol=0.):
            raise AssertionError(f"Independent material reservoir mismatch: {key}")
        residual = abs(left - right) / max(abs(left), abs(right)) if left or right else 0.
        maximum = max(maximum, residual)
        rows.append(dict(material_id=key[0], field=key[1], initial=original.get(key, 0.),
            retained=current.get(key, 0.), lost=removed.get(key, 0.), born=born.get(key, 0.),
            relative_residual=residual))
    return dict(maximum_relative_residual=maximum, reservoirs=rows)


def _step(state, mode):
    owners = {fragment.parcel.plate for fragment in state.fragments}
    if mode == "coplanar":
        omega = {owner: (0., 0., .01 if owner == 1 else 0.) for owner in owners}
        dt, policy = 1e-5, _coplanar_policy
    else:
        omega = {owner: NEXT_OMEGA[0][owner] for owner in owners}
        dt = NEXT_DT_MYR * (1e-9 if mode == "full_vector_millisecond" else 1.)
        policy = _tie_policy if mode == "full_vector_millisecond" else None
    result = advance_native_surface(state, omega, dt, birth_factory=_birth,
                                    polarity_resolver=policy)
    history = dict(omega_rad_per_myr={str(owner): list(omega[owner]) for owner in sorted(omega)},
        dt_myr=dt, start_time_myr=state.time_myr, end_time_myr=result.state.time_myr,
        polarity_declaration="none" if policy is None else policy.__name__)
    return result, history


def _run_case(out, mode, steps):
    directory = out / mode
    directory.mkdir()
    source = _source()
    _write(directory / "literal_legacy_source.json", asdict(source))
    state = enroll_native_surface(source)
    provenance = dict(control=mode, loaded_underthrust_steps=0,
        physical_material_pair_law="not supplied", cooling="frozen", mesh_capacity="not claimed")
    initial_path = save_native_checkpoint(directory / "initial.json", state, provenance=provenance)
    initial_snapshot = _encoded(native_surface_to_dict(state))
    all_losses, all_births, records, midpoint, default_refusal = [], [], [], None, None
    middle = max(1, steps // 2)
    for index in range(steps):
        before = state
        if mode == "full_vector_millisecond" and index == 1:
            before_snapshot = _encoded(native_surface_to_dict(state))
            calls = []
            def forbidden_birth(*args):
                calls.append(args)
                raise AssertionError("Unresolved polarity reached a birth callback")
            try:
                advance_native_surface(state, NEXT_OMEGA[0], NEXT_DT_MYR * 1e-9,
                                       birth_factory=forbidden_birth)
            except UnresolvedPolarityError as refusal:
                default_refusal = dict(reason="unresolved physical hot-material polarity",
                    positive_overlaps=len(refusal.overlaps), birth_calls=len(calls),
                    input_unchanged=_encoded(native_surface_to_dict(state)) == before_snapshot,
                    state_sha256=hashlib.sha256(before_snapshot).hexdigest())
            else:
                raise AssertionError("The declared default-policy refusal was not reproduced")
            if calls or not default_refusal["input_unchanged"]:
                raise AssertionError("Default-policy refusal was not atomic")
        result, history = _step(state, mode)
        ledger = _check_ledgers(before.fragments, result.state.fragments, result.losses, result.births)
        if not result.losses or not result.births:
            raise AssertionError("Differential control did not exercise positive loss and birth")
        if mode != "full_vector_whole_interval":
            cold_children = [fragment for fragment in result.state.fragments
                if fragment.parcel.material_id == "thin" and fragment.parcel.cold_mantle_volume_km3 > 0.]
            if not cold_children or not result.contacts:
                raise AssertionError("Repeated control lost its original thin cold material or native contact")
        state = result.state
        all_losses.extend(result.losses)
        all_births.extend(result.births)
        record = dict(step=index + 1, history=history, fragment_count=len(state.fragments),
            loss_count=len(result.losses), birth_count=len(result.births), contact_count=len(result.contacts),
            minimum_positive_area_km2=min(fragment.parcel.area_km2 for fragment in state.fragments),
            independent_material_ledger=ledger, diagnostics=result.diagnostics,
            losses=[dict(source_fragment_id=loss.source_fragment_id,
                receiver_fragment_id=loss.receiver_fragment_id, receiver_plate=loss.receiver_plate,
                time_myr=loss.time_myr, attachment_status=loss.attachment_status,
                polarity_basis=loss.polarity_basis, fragment_id=loss.fragment.fragment_id,
                parcel=asdict(loss.fragment.parcel), geometry=native_polygon_to_dict(loss.fragment.geometry))
                for loss in result.losses],
            contacts=[dict(contact_id=contact.contact_id, fragment_a=contact.fragment_a,
                fragment_b=contact.fragment_b, material_a=contact.material_a, material_b=contact.material_b,
                plate_a=contact.plate_a, plate_b=contact.plate_b, length_km=contact.length_km,
                arc=native_arc_to_dict(contact.arc)) for contact in result.contacts])
        _write(directory / f"step_{index + 1:02d}.json", record)
        records.append(record)
        if index + 1 == middle:
            midpoint = state
    if _encoded(native_surface_to_dict(enroll_native_surface(source))) != initial_snapshot:
        raise AssertionError("Initial source changed during the control")
    cumulative = _check_ledgers(enroll_native_surface(source).fragments, state.fragments,
                                all_losses, all_births)
    partial_path = save_native_checkpoint(directory / "partial.json", midpoint, provenance=provenance)
    resumed, metadata = load_native_checkpoint(partial_path)
    partial_resaved = save_native_checkpoint(directory / "partial_resaved.json", resumed, provenance=metadata)
    for _ in range(steps - middle):
        resumed, _ = _step(resumed, mode)
        resumed = resumed.state
    direct = save_native_checkpoint(directory / "direct_final.json", state, provenance=provenance)
    resumed_path = save_native_checkpoint(directory / "resumed_final.json", resumed, provenance=metadata)
    restored, metadata = load_native_checkpoint(direct)
    resaved = save_native_checkpoint(directory / "final_resaved.json", restored, provenance=metadata)
    pairs = [(partial_path, partial_resaved), (direct, resumed_path), (direct, resaved)]
    equality = [dict(left=left.name, right=right.name,
        byte_equal=left.read_bytes() == right.read_bytes(), sha256=_hash(left)) for left, right in pairs]
    if not all(row["byte_equal"] for row in equality):
        raise AssertionError("Direct and restored native checkpoint bytes differ")
    return dict(mode=mode, accepted_geometric_steps=steps, loaded_underthrust_steps=0,
        initial_fragments=len(source.fragments), final_fragments=len(state.fragments),
        total_loss_cohorts=len(all_losses), total_hot_births=len(all_births),
        native_contact_counts=[record["contact_count"] for record in records],
        step_histories=[record["history"] for record in records],
        step_material_errors=[record["independent_material_ledger"]["maximum_relative_residual"]
                              for record in records],
        cumulative_material_ledger=cumulative, checkpoint_byte_pairs=equality,
        default_policy_next_interval_refusal=default_refusal,
        initial_checkpoint_sha256=_hash(initial_path), final_checkpoint_sha256=_hash(direct))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    before, start = _snapshot_code(), time.monotonic()
    args.out.mkdir(parents=True, exist_ok=False)
    _write(args.out / "code_before.json", before)
    report = dict(started_utc=datetime.now(timezone.utc).isoformat(),
        command=sys.argv, cases=[], status="running", loaded_underthrust_steps=0)
    failure = None
    try:
        for mode, steps in (("full_vector_whole_interval", 1),
                            ("full_vector_millisecond", 4), ("coplanar", 4)):
            print(f"Starting {mode}: {steps} differential geometric steps", flush=True)
            report["cases"].append(_run_case(args.out, mode, steps))
            print(f"Passed {mode}", flush=True)
        report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["failure"] = dict(type=type(error).__name__, message=str(error))
        failure = error
    after = _snapshot_code()
    _write(args.out / "code_after.json", after)
    report.update(finished_utc=datetime.now(timezone.utc).isoformat(),
        wall_seconds=time.monotonic() - start, code_files=len(before), code_unchanged=before == after,
        code_before_sha256=_hash(args.out / "code_before.json"),
        code_after_sha256=_hash(args.out / "code_after.json"), runner_sha256=_hash(__file__))
    report["final_eligible"] = report["status"] == "passed" and report["code_unchanged"]
    _write(args.out / "report.json", report)
    manifest = {str(path.relative_to(args.out)): _hash(path)
                for path in sorted(args.out.rglob("*")) if path.is_file()}
    _write(args.out / "artifact_manifest.json", manifest)
    if failure is not None:
        raise failure
    if not report["final_eligible"]:
        raise RuntimeError("Control code changed during execution")
    print(json.dumps(dict(status=report["status"], wall_seconds=report["wall_seconds"],
                          code_files=report["code_files"], report=str(args.out / "report.json"))), flush=True)


if __name__ == "__main__":
    main()
