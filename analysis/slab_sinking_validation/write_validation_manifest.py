"""Capture final scientific provenance without altering any saved checkpoint."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def main():
    from tectonics.genesis_starter_continuation import mechanics_limitations
    source = ROOT / "results/genesis_runs/starter_20260928_192306_547551/starter_checkpoint.npz"
    expected = "b4584dd66208ff80aea18a178c38da2b96e353c723663251091886dee49e87f6"
    actual = sha(source)
    if actual != expected:
        raise ValueError("Original Starter archive hash changed")
    current = {str(path.relative_to(ROOT)).replace("\\", "/"):sha(path)
               for path in sorted((ROOT / "tectonics").glob("*.py"))}
    corrections, manifests = [], []
    for path in sorted((HERE / "runs").glob("validated_*/elapsed_*/continuation.json")):
        folder = path.parent
        report, metrics = read(path), read(folder.with_suffix(".metrics.json"))
        integrity = {name:sha(folder / name) == value for name,value in report["checkpoint_sha256"].items()}
        cfg = yaml.safe_load((folder / "mature_config.yaml").read_text(encoding="utf-8"))
        correct = mechanics_limitations(cfg)
        corrections.append(dict(continuation=str(path), original_continuation_sha256=sha(path),
            correction_needed=report["limitations"] != correct,
            saved_limits=report["limitations"], corrected_current_limits=correct))
        manifests.append(dict(path=str(folder), age_myr=report["duration_myr"],
            mechanics_model_version=report["mechanics_model_version"], slab_model=report["young_slab_force_model"],
            checkpoint_file_integrity=integrity,
            production_files_different_from_current=[name for name in current.keys() | metrics["production_sha256"].keys()
                if current.get(name) != metrics["production_sha256"].get(name)],
            production_files_changed_during_run=metrics["production_files_changed_during_run"]))
    write(HERE / "saved_report_limitations_correction.json", dict(
        note="Description-only sidecar. Original reports and all checkpoints remain byte-for-byte unchanged. Current0.4 report wording replaces the earlier stale0.3 slab-disabled limitation.",
        source_function="tectonics.genesis_starter_continuation.mechanics_limitations", cases=corrections))
    final = HERE / "runs/validated_sinking_fixed2_sub4_dt1/elapsed_0400"
    metrics = read(final.with_suffix(".metrics.json"))
    mismatches = [name for name in current.keys() | metrics["production_sha256"].keys()
                  if current.get(name) != metrics["production_sha256"].get(name)]
    scientific_hashes_match = mismatches == ["tectonics/genesis_starter_continuation.py"] or not mismatches
    if not scientific_hashes_match:
        raise ValueError(f"Unexpected production changes after final main run: {mismatches}")
    metadata = read(final / "mature_checkpoint/meta.json")
    failures = metadata["subduction_memory"]["young_boundary_state"]["mechanical_detachments"]
    report = read(final / "continuation.json")
    audit = read(HERE / "step_audit.json")
    result = dict(original_starter=str(source), original_starter_sha256=actual,
        original_starter_matches_expected=True, final_checkpoint=str(final / "mature_checkpoint"),
        final_continuation=str(final / "continuation.json"),
        final_continuation_sha256=sha(final / "continuation.json"),
        final_checks=report["checks"],
        final_material_relative_volume_residual=report["material_ledger"]["relative_volume_residual"],
        final_thermal_energy_relative_residual=report["history"][-1]["thermal_energy_relative_residual"],
        final_production_hashes_match_current_except_descriptive_report_fix=scientific_hashes_match,
        final_production_files_different_from_current=mismatches,
        descriptive_late_change="Only genesis_starter_continuation.py: mechanics_limitations(config) and report limitations selection. No scientific equation, saved state, parameter or continuation file changed.",
        numerical_changes="Original0–100 Myr completed before two dual-solver numerical repairs. Resumed100–400 after BVLS and exact positive-support KKT polishing. Same convex objective and physical coefficients; saved107/186 matrices reproduced and tested. Fresh50 replay preserves every scientific array and physical report; only detached-neck tension diagnostic roundoff differs.",
        current_production_sha256=current, completed_cases=manifests,
        all_saved_checkpoint_files_match=all(all(row["checkpoint_file_integrity"].values()) for row in manifests),
        trace_audit=dict(segments=len(audit["segments"]), steps=sum(row["trace_steps"] for row in audit["segments"]),
            all_checks_passed=audit["all_checks_passed"],
            maximum_relative_torque_residual=max(row["maxima"]["torque_relative_residual"] for row in audit["segments"]),
            maximum_relative_power_residual=max(row["maxima"]["power_relative_residual"] for row in audit["segments"])),
        final_inventory=report["accepted_slab_inventory"],
        cumulative_mechanical_detachment_retained_oceanic_volume_km3=sum(row["retained_oceanic_volume_km3"] for row in failures),
        last100_myr_statistics=read(HERE / "last100_myr_statistics.json"),
        cpu_parity=read(HERE / "cpu_parity.json"),
        legacy03_resume=str(HERE / "runs/legacy03_resume_0401/elapsed_0401"),
        accepted_material_remesh_resume=str(HERE / "runs/validated_accepted_remesh_sub5_dt1/elapsed_0051"))
    write(HERE / "validation.json", result)
    print(json.dumps(dict(source_sha256=actual, checkpoint_integrity=result["all_saved_checkpoint_files_match"],
        scientific_code_hashes_current=scientific_hashes_match,
        described_file_exceptions=mismatches, audit=result["trace_audit"])))


if __name__ == "__main__":
    main()
