"""Inspect which stored contact trace limits a completed geometry experiment.

Only saved arrays are compared. No mechanics is advanced, no admissibility
policy changes, and a traction-free limiting trace is not permission to remove
the geometry bound; future re-contact would still depend on its frames.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.genesis_path_dynamics_validation import CoupledModel, SOURCE


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    files = [args.input/name for name in ("validation.json", "local_final_geometry.npz",
                                          "local_final_checkpoint.npz")]+[SOURCE]
    hashes = {str(path.resolve()): _hash(path) for path in files}
    report = json.loads(files[0].read_text(encoding="utf-8"))
    coupled = CoupledModel(SOURCE.read_bytes(), source_path=str(SOURCE))
    onset = coupled.law_parameters.damage_onset_opening_m
    with np.load(files[1], allow_pickle=False) as data:
        linear, actual = data["linear_jump_m"], data["mean_corotated_actual_jump_m"]
        error = np.abs(actual-linear)/np.maximum(np.abs(linear), onset)
        trace, component = map(int, np.unravel_index(np.argmax(error), error.shape))
        linear_point, actual_point = linear[trace], actual[trace]
    with np.load(files[2], allow_pickle=False) as data:
        ids = np.flatnonzero(data["cohort_trace_index"] == trace)
        damage = data["cohort_damage"][ids]
        traction = data["cohort_traction_pa"][ids]
        areas = data["cohort_area_ref_m2"][ids]
        age = json.loads(str(data["metadata"]))["elapsed_years"]
    checks = {
        "primary_artifact_hashes_match": all(_hash(args.input/name) == digest
            for name, digest in report["artifact_sha256"].items()),
        "primary_validation_script_unchanged": _hash(ROOT/"analysis/genesis_path_local_geometry_validation.py")
            == report["code_sha256"][str(Path("analysis")/"genesis_path_local_geometry_validation.py")],
        "critical_metric_matches_primary": bool(np.isclose(error[trace, component],
            report["milestones"][-1]["relative_contact_jump_error"], rtol=0, atol=1e-9)),
        "critical_cohorts_exist": bool(len(ids)),
        "inputs_unchanged": all(_hash(path) == digest for path, digest in hashes.items()),
    }
    result = {"scope": "saved_state_contact_frame_diagnostic_no_mechanical_evolution",
        "mechanical_elapsed_years": age, "trace_index": trace,
        "limiting_component": "normal" if component == 0 else "tangential",
        "relative_jump_error": float(error[trace, component]),
        "normalization_onset_m": onset,
        "normalization_denominator_m": float(max(abs(linear_point[component]), onset)),
        "linear_jump_m": linear_point.tolist(), "mean_corotated_jump_m": actual_point.tolist(),
        "difference_m": (actual_point-linear_point).tolist(),
        "cohort_indices": ids.tolist(), "cohort_damage": damage.tolist(),
        "cohort_traction_pa": traction.tolist(), "cohort_reference_area_m2": areas.tolist(),
        "limiting_trace_currently_traction_free": bool(len(ids) and np.all(traction == 0)),
        "limiting_trace_fully_decohered": bool(len(ids) and np.all(damage == 1)),
        "global_same_history_traction_comparison": report["contact_frame_traction_diagnostic"],
        "interpretation": "Stop bounds validity of retained paired reference geometry; it is not a physical stopping condition or a demonstrated 2% force error. The limiting trace currently carries zero traction. No tolerance was weakened.",
        "input_sha256": hashes, "code_sha256": _hash(__file__), "checks": checks}
    output = args.input/"contact_frame_diagnostic.json"
    if output.exists():
        raise ValueError("Supplementary diagnostic already exists")
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))
    if not all(checks.values()):
        raise SystemExit("Supplementary checks failed")


if __name__ == "__main__":
    main()
