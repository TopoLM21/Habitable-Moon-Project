"""Persist and resume native rigid geometry from the accepted physical epoch.

This is a common-motion geometry control, not another loaded underthrust step.
The input remains the unchanged previously certified physical checkpoint.
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

from tectonics.fractional_surface import EXTENSIVE_FIELDS
from tectonics.geometric_bending_reference import _surface_from_dict
from tectonics.geometric_intrinsic_surface import (
    advance_intrinsic_surface, enroll_intrinsic_surface,
    intrinsic_surface_to_dict, load_intrinsic_checkpoint, save_intrinsic_checkpoint,
)

DT_MYR = 3.168808781402895e-8
STEPS = (
    ((4.171092555597449e-18, 2.4581194098898567e-19, 34.86996946470772), DT_MYR),
    ((.023, -.037, .049), .71),
    ((3.5029521325109036e-18, -7.03581045461479e-20, 31.616288107981852), DT_MYR),
    ((-.16, .09, .13), .29),
) * 2


def encoded(payload):
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ledger(fragments):
    result = {}
    for fragment in fragments:
        for key in EXTENSIVE_FIELDS:
            result.setdefault(fragment.parcel.material_id, {}).setdefault(key, []).append(
                getattr(fragment.parcel, key))
    return {material: {key: math.fsum(values) for key, values in reservoirs.items()}
            for material, reservoirs in sorted(result.items())}


def advance(state, steps, records=None):
    for vector, dt in steps:
        owners = {fragment.parcel.plate: vector for fragment in state.fragments}
        state = advance_intrinsic_surface(state, owners, dt)
        if records is not None:
            projected_counts = []
            for fragment in state.fragments:
                projection = fragment.geometry.world_polygon()
                assert all(math.isfinite(value) for ray in projection for value in ray)
                projected_counts.append(len(set(projection)))
            records.append(dict(accepted_common_geometry_steps=len(state.rotations),
                time_myr=state.time_myr, fragment_count=len(state.fragments),
                min_material_area_km2=min(f.parcel.area_km2 for f in state.fragments),
                min_distinct_projected_vertices=min(projected_counts),
                projected_lower_dimensional_fragments=sum(n < 3 for n in projected_counts),
                material_ledger_sha256=hashlib.sha256(encoded(ledger(state.fragments))).hexdigest()))
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT/
        "analysis/geometric_underthrust_validation/cloud_control/direct_final.json")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    started = datetime.now(timezone.utc).isoformat()
    clock = time.perf_counter()
    before_sha = sha(args.source)
    original = json.loads(args.source.read_text(encoding="utf-8"))
    digest = original.pop("payload_sha256")
    if hashlib.sha256(encoded(original)).hexdigest() != digest:
        raise ValueError("Source physical checkpoint integrity mismatch")
    source = _surface_from_dict(original["surface"])
    initial = enroll_intrinsic_surface(source)
    initial_ledger = ledger(source.fragments)
    provenance = dict(model="common_motion_intrinsic_geometry_control",
        source_checkpoint=args.source.relative_to(ROOT).as_posix(),
        source_checkpoint_sha256=before_sha, new_loaded_underthrust_steps=0)
    args.out.mkdir(parents=True, exist_ok=False)
    save_intrinsic_checkpoint(args.out/"initial.json", initial, provenance=provenance)
    direct_records = []
    direct = advance(initial, STEPS, direct_records)
    direct_path = save_intrinsic_checkpoint(args.out/"direct_final.json", direct,
                                          provenance=provenance)
    partial = advance(initial, STEPS[:4])
    partial_path = save_intrinsic_checkpoint(args.out/"partial.json", partial,
                                           provenance=provenance)
    restored, saved_provenance = load_intrinsic_checkpoint(partial_path)
    resaved_partial = save_intrinsic_checkpoint(args.out/"partial_resaved.json", restored,
                                               provenance=saved_provenance)
    resumed = advance(restored, STEPS[4:])
    resumed_path = save_intrinsic_checkpoint(args.out/"resumed_final.json", resumed,
                                           provenance=saved_provenance)
    restored_final, final_provenance = load_intrinsic_checkpoint(direct_path)
    resaved_final = save_intrinsic_checkpoint(args.out/"final_resaved.json", restored_final,
                                              provenance=final_provenance)
    pairs = ((partial_path, resaved_partial), (direct_path, resumed_path),
             (direct_path, resaved_final))
    assert all(first.read_bytes() == second.read_bytes() for first, second in pairs)
    assert encoded(intrinsic_surface_to_dict(direct)) == encoded(intrinsic_surface_to_dict(resumed))
    assert ledger(direct.fragments) == initial_ledger
    assert all(current.geometry.body_rays == old.polygon and
               all(getattr(current.parcel, key) == getattr(old.parcel, key)
                   for key in EXTENSIVE_FIELDS)
               for old, current in zip(source.fragments, direct.fragments))
    assert before_sha == sha(args.source)
    baseline = json.loads((ROOT/
        "analysis/geometric_underthrust_validation/parallel_portable_final/code_and_tests_after.json")
        .read_text(encoding="utf-8"))
    changed = [name for name, digest in baseline.items() if sha(ROOT/name) != digest]
    assert changed == []
    report = dict(status="passed_common_motion_intrinsic_geometry_control",
        started_utc=started, finished_utc=datetime.now(timezone.utc).isoformat(),
        seconds=time.perf_counter()-clock, accepted_common_geometry_steps=len(STEPS),
        new_loaded_underthrust_steps=0, source_checkpoint_sha256=before_sha,
        source_checkpoint_unchanged=True, source_time_myr=source.time_myr,
        final_time_myr=direct.time_myr, source_fragment_count=len(source.fragments),
        source_material_count=len(initial_ledger),
        min_material_area_km2=min(f.parcel.area_km2 for f in direct.fragments),
        all_positive_fragments_and_four_reservoirs_preserved=True,
        body_rays_unchanged=True, legacy_baseline_path_count=len(baseline),
        legacy_baseline_changed_paths=changed,
        legacy_baseline_is_reused_not_a_new_full_profile_run=True,
        trajectory=direct_records, byte_pairs=[dict(first=a.name, second=b.name,
            equal=True, sha256=sha(a), bytes=a.stat().st_size) for a, b in pairs],
        checkpoint_sha256={path.name: sha(path) for path in sorted(args.out.glob("*.json"))},
        code_sha256={name: sha(ROOT/name) for name in (
            "tectonics/geometric_intrinsic_geometry.py", "tectonics/geometric_intrinsic_surface.py",
            "analysis/geometric_intrinsic_validation/run_intrinsic_control.py")},
        remaining_original_plan_point_1=["native differential cuts, supports and contact lineage",
                                         "new physical material pairs and mechanical memory"],
        original_plan_completed_points=[], original_plan_partial_points=[1],
        original_plan_pending_points=[2,3,4,5,6])
    with (args.out/"report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: report[key] for key in ("status", "seconds",
        "accepted_common_geometry_steps", "source_fragment_count", "min_material_area_km2")},
        ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
