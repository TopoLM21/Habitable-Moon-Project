"""Re-solve the original force and replay a true donor material chronology.

This bounded component transfers one certified positive 2D donor patch into
the existing once-only passive archive closure. Contact departure/reentry is
recorded separately. It does not adopt a global force/surface interval.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analysis.geometric_native_contact_event_validation.run_time_support_control import run_control as run_force_control
from tectonics.geometric_native_donor_sweep import (
    certify_native_donor_patch_sweep, native_donor_patch_sweep_to_dict,
)
from tectonics.geometric_native_material_history import (
    make_native_material_memory_reference, initialize_native_material_history,
    evaluate_native_material_history, native_material_history_to_dict,
    native_material_history_from_dict,
    evaluate_native_material_point_history, native_material_point_history_to_dict,
    native_material_point_history_from_dict,
)
from tectonics.geometric_native_surface import native_surface_from_dict


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open("xb") as stream:
        stream.write(encoded(value)+b"\n")


def sources():
    paths=set()
    for directory in ("tectonics", "tests", "scripts", "analysis"):
        paths.update((ROOT/directory).rglob("*.py"))
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}


def memory_payloads(prepared, candidate, support):
    old = prepared["native_underthrust_motion"]["snapshot"]
    before = {key: old[key] for key in (
        "states", "spatial_memory", "free_anchor_fields", "elapsed_time_s",
        "cumulative_source_work_j", "cumulative_basal_dissipation_j",
        "cumulative_bending_numerical_dissipation_j")}
    after = dict(states=candidate["states"],
        spatial_memory=candidate["spatial_step"]["memory"],
        free_anchor_fields=candidate["spatial_step"]["anchor_fields"],
        dt_s=candidate["dt_s"], candidate_diagnostics=candidate["diagnostics"],
        candidate_sha256=hashlib.sha256(encoded(candidate)).hexdigest(),
        whole_time_support=deepcopy(support), candidate_adopted=False)
    return before, after


def run_control(out):
    started = time.perf_counter()
    if out.exists():
        raise FileExistsError("Material history requires a fresh output directory")
    out.mkdir(parents=True)
    source_before = sources()
    write(out/"sources_before.json",source_before)
    force_dir=out/"original_force"
    force_dir.mkdir()
    force_report=run_force_control(force_dir)
    prepared=json.loads((force_dir/"initial.json").read_bytes())
    candidate=json.loads((force_dir/"force_candidate.json").read_bytes())
    support=json.loads((force_dir/"constitutive_time_support.json").read_bytes())
    historical_probe_path=ROOT/"analysis/geometric_native_underthrust_validation/force_control_release_02/force_candidate_refusal_probe.json"
    historical_probe_bytes=historical_probe_path.read_bytes()
    historical_probe=json.loads(historical_probe_bytes)
    if any(candidate[k]!=v for k,v in historical_probe["candidate"].items()):
        raise AssertionError("Material chronology must retain the actual original force candidate")
    if candidate["admission"]!=historical_probe["admission"]:
        raise AssertionError("The original native force geometry has changed")
    surface=native_surface_from_dict(prepared["current_surface"])
    original={f.fragment_id:f for f in surface.fragments}
    donor,receiver=original["octant:6"],original["octant:2"]
    omega=dict(candidate["omega_rad_per_myr"])
    duration=candidate["admission"]["dt_myr"]
    print("Certifying true original donor material against all receiver planes",flush=True)
    sweep=certify_native_donor_patch_sweep(donor.geometry,receiver.geometry,
        tuple(omega[donor.parcel.plate]),tuple(omega[receiver.parcel.plate]),duration)
    write(out/"donor_patch_sweep.json",native_donor_patch_sweep_to_dict(sweep))
    before_payload,after_payload=memory_payloads(prepared,candidate,support)
    reference=make_native_material_memory_reference(before_payload,after_payload,
        constitutive_path_sha256=support["path_sha256"])
    print("Transferring one real patch once; recording contact exit and reentry separately",flush=True)
    history=initialize_native_material_history(donor,receiver,sweep,
        material_epoch_myr=surface.time_myr,memory_reference=reference)
    first=native_material_history_to_dict(history)
    loaded=native_material_history_from_dict(deepcopy(first),expected_memory_reference=reference)
    replay=native_material_history_to_dict(loaded)
    repeated=native_material_history_to_dict(initialize_native_material_history(
        donor,receiver,sweep,material_epoch_myr=surface.time_myr,memory_reference=reference))
    if encoded(first)!=encoded(replay) or encoded(first)!=encoded(repeated):
        raise AssertionError("Material allocation, ancestry or mechanical references changed after replay")
    write(out/"material_history_checkpoint.json",first)
    write(out/"material_history_replayed.json",replay)
    write(out/"material_history_repeated.json",repeated)
    # Snapshot serialization belongs to the local component, not a global
    # force epoch. The saved evidence retains exact Fraction time queries.
    from tectonics.geometric_native_material_history import native_material_history_snapshot_to_dict
    times=tuple(sweep.sample_fractions)
    queries=tuple(dict(time_fraction=[t.numerator,t.denominator],
        snapshot=native_material_history_snapshot_to_dict(evaluate_native_material_history(history,t)))
        for t in times)
    replay_queries=tuple(dict(time_fraction=[t.numerator,t.denominator],
        snapshot=native_material_history_snapshot_to_dict(evaluate_native_material_history(loaded,t)))
        for t in times)
    if encoded(queries)!=encoded(replay_queries):
        raise AssertionError("Exact-time material allocations changed after local restart")
    write(out/"material_time_queries.json",queries)
    write(out/"material_time_queries_replayed.json",replay_queries)
    if [q["snapshot"]["contact_occupancy"] for q in queries]!=["outside","inside","outside","inside","inside"]:
        raise AssertionError("The actual patch contact chronology was not preserved")
    if [q["snapshot"]["material_allocation"] for q in queries]!=["live","archived","archived","archived","archived"]:
        raise AssertionError("Contact exit or return allocated material more than once")
    if len({q["snapshot"]["archive_cohort_id"] for q in queries})!=1:
        raise AssertionError("A contact event created a second donor cohort")
    if sum(e.operation=="live_to_passive_archive_once" for e in history.events)!=1:
        raise AssertionError("The passive closure did not debit the original patch exactly once")
    # Execute the material-time function at spatially distinct points. They
    # must retain different entry times at one common exact query time.
    points=(("earlier_entry",sweep.patch_x_bounds.lo,sweep.patch_y_bounds.hi),
            ("later_entry",sweep.patch_x_bounds.hi,sweep.patch_y_bounds.lo))
    finals=[evaluate_native_material_point_history(history,x,y,1) for _,x,y in points]
    if finals[0].events[0].time_fraction.hi>=finals[1].events[0].time_fraction.lo:
        raise AssertionError("Spatial entry times collapsed to one fictitious timestamp")
    common=(finals[0].events[0].time_fraction.hi+finals[1].events[0].time_fraction.lo)/2
    point_queries,replayed_point_queries=[],[]
    for label,x,y in points:
        for instant in (common,*times):
            raw=native_material_point_history_to_dict(evaluate_native_material_point_history(history,x,y,instant))
            replayed=native_material_point_history_to_dict(native_material_point_history_from_dict(raw,history=loaded))
            if encoded(raw)!=encoded(replayed):
                raise AssertionError("Material point entry-time function changed after restart")
            point_queries.append(dict(label=label,query=raw))
            replayed_point_queries.append(dict(label=label,query=replayed))
    at_common=[q["query"]["material_allocation"] for q in point_queries if q["query"]["time_fraction"]==[common.numerator,common.denominator]]
    if at_common!=["archived","live"]:
        raise AssertionError("Distinct original material points lost their distinct first-entry histories")
    write(out/"material_point_queries.json",point_queries)
    write(out/"material_point_queries_replayed.json",replayed_point_queries)
    if (force_dir/"initial.json").read_bytes()!=(force_dir/"after_refusal.json").read_bytes():
        raise AssertionError("The local chronology changed the actual force state")
    source_after=sources()
    if source_before!=source_after or historical_probe_path.read_bytes()!=historical_probe_bytes:
        raise AssertionError("Source code or authoritative candidate input changed during the control")
    write(out/"sources_after.json",source_after)
    artifact_hashes={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob("*")) if p.is_file()}
    report=dict(format="native-true-donor-material-history-control-1",status="PASS",
        seconds=time.perf_counter()-started,source_count=len(source_before),sources_unchanged=True,
        source_probe=str(historical_probe_path.relative_to(ROOT)),source_probe_sha256=sha(historical_probe_path),
        original_force_candidate_fields_identical=True,
        full_original_owner_components_retained=True,
        accepted_loaded_native_intervals=0,accepted_source_work_j=0.,candidate_memory_adopted=False,
        geometry_admission_authorized=False,
        patch_scope="certified positive original donor 2D patch; not entire swept material domain",
        contact_sequence=["entry","exit","entry"],
        archive_law="once_only_passive_archive_first_entry",
        archive_transfer_count=1,contact_reentry_creates_new_archive_allocation=False,
        whole_parent_symbolic_reservoir_conservation=True,
        mechanical_memory_is_referenced_without_allocation_to_2d_mass=True,
        prepared_force_checkpoint_unchanged=True,
        local_checkpoint_byte_exact=True,exact_time_queries_byte_exact=True,
        executable_per_material_point_time_function=True,spatially_distinct_first_entry_times_preserved=True,
        local_checkpoint_sha256=sha(out/"material_history_checkpoint.json"),
        force_report_sha256=sha(force_dir/"report.json"),
        force_refusal_reasons=force_report["refusal_reasons"],artifacts_sha256=artifact_hashes,
        remaining_within_point_1="complete swept domains including transverse fringes, vacancies and births; original-pair contact law and accepted native interval")
    report["report_sha256"]=hashlib.sha256(encoded(report)).hexdigest()
    write(out/"report.json",report)
    print(json.dumps(dict(status="PASS",source_count=len(source_before),
        local_archive_transfers=1,accepted_loaded_native_intervals=0)),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    run_control(args.out)


if __name__=="__main__":
    main()
