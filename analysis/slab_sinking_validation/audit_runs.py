"""Audit every captured integration step, not only the final saved endpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=HERE / "runs")
    parser.add_argument("--arm-glob", default="final_*")
    parser.add_argument("--output", type=Path, default=HERE / "step_audit.json")
    parser.add_argument("--relative-tolerance", type=float, default=1e-7)
    args = parser.parse_args()
    summaries = []
    for arm in sorted(args.runs.glob(args.arm_glob)):
        for path in sorted(arm.glob("elapsed_*.metrics.json")):
            saved = json.loads(path.read_text(encoding="utf-8"))
            prefix = Path(str(path).removesuffix(".metrics.json"))
            trace = [json.loads(line) for line in prefix.with_suffix(".dynamics.jsonl").read_text(encoding="utf-8").splitlines()]
            limits = {name:0. for name in ("torque_relative_residual", "power_relative_residual",
                "reaction_power_fraction", "velocity_reconstruction_relative_error",
                "drag_relative_asymmetry", "neck_tension_to_capacity")}
            min_eigenvalue, minimum_feed, max_sections, active_max, failures = 1., 0., 0, 0, 0
            violations = []
            for row in trace:
                metrics = row["metrics"]
                names = {"torque_relative_residual":"target_torque_relative_residual",
                    "power_relative_residual":"net_power_balance_relative_residual",
                    "reaction_power_fraction":"slab_constraint_power_fraction",
                    "velocity_reconstruction_relative_error":"target_velocity_component_relative_error",
                    "drag_relative_asymmetry":"total_drag_relative_asymmetry",
                    "neck_tension_to_capacity":"maximum_attached_neck_tension_to_capacity"}
                for output_key, input_key in names.items():
                    value = metrics.get(input_key, 0.)
                    # Capacity is tensile: negative neck force is compression,
                    # not a violation of the tensile fracture criterion.
                    if output_key != "neck_tension_to_capacity":
                        value = abs(value)
                    limits[output_key] = max(limits[output_key], value)
                min_eigenvalue = min(min_eigenvalue, metrics.get("total_drag_min_normalized_eigenvalue", 1.))
                sinking = row["trace"].get("slab_sinking_diagnostics", {})
                feed = sinking.get("minimum_feed_m_s", 0.)
                feed_scale = max(abs(sinking.get("maximum_feed_m_s", 0.)), abs(feed), 1e-20)
                minimum_feed = min(minimum_feed, feed)
                max_sections = max(max_sections, len(row["trace"].get("slab_sections", [])))
                active_max = max(active_max, sinking.get("no_eduction_active_count", 0))
                failures += sinking.get("neck_failures_this_evaluation", 0)
                if feed < -max(1e-20, args.relative_tolerance*feed_scale):
                    violations.append(dict(time_myr=row["state_time_myr"], issue="negative_feed", value=feed))
                for key, value in metrics.items():
                    if key.endswith("_nonnegative") and not value:
                        violations.append(dict(time_myr=row["state_time_myr"], issue=key))
            checks = dict(
                every_continuation_check=all(saved["checks"].values()), source_unchanged=saved["source_unchanged"],
                torque_balance=limits["torque_relative_residual"] <= args.relative_tolerance,
                power_balance=limits["power_relative_residual"] <= args.relative_tolerance,
                reaction_does_no_work=limits["reaction_power_fraction"] <= args.relative_tolerance,
                component_reconstruction=limits["velocity_reconstruction_relative_error"] <= args.relative_tolerance,
                symmetric_positive_drag=limits["drag_relative_asymmetry"] < 1e-12 and min_eigenvalue > 0,
                finite_neck_capacity=limits["neck_tension_to_capacity"] <= 1.+2e-10,
                feasible_feed_and_positive_dissipation=not violations)
            summaries.append(dict(arm=arm.name, age_myr=saved["duration_myr"],
                trace_steps=len(trace), checks=checks, maxima=limits,
                min_normalized_drag_eigenvalue=min_eigenvalue, minimum_feed_m_s=minimum_feed,
                maximum_attached_force_sections=max_sections, maximum_active_feed_constraints=active_max,
                total_neck_failures_evaluated=failures, violations=violations,
                production_files_changed_during_run=saved["production_files_changed_during_run"]))
    result = dict(scope="Every captured integration step of completed segments; material ledgers at saved endpoints.",
        relative_tolerance=args.relative_tolerance, segments=summaries,
        all_checks_passed=bool(summaries) and all(all(row["checks"].values()) for row in summaries))
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(dict(output=str(args.output), segments=len(summaries), all_checks_passed=result["all_checks_passed"])))
    if not result["all_checks_passed"]:
        raise SystemExit("At least one step audit check failed or there were no completed segments")


if __name__ == "__main__":
    main()
