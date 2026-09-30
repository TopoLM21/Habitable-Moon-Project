"""Independent frozen probes of one unchanged, versioned mechanical state."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Sweep output must be new")
    args.output.mkdir(parents=True)
    cases = {"disabled": ("disabled_pending_closure", [], 1.),
        "upper_bound": ("full_transmission_upper_bound", [], 1.),
        "sinking_default": ("viscous_sinking_v1", [], 1.),
        "sinking_dt0p5": ("viscous_sinking_v1", [], .5)}
    for key, values in (("viscosity_contrast", (10., 1000.)),
            ("bend_radius_thickness_ratio", (2., 5.)),
            ("mantle_shear_length_fraction", (.25, 1.))):
        for value in values:
            label = key + "_" + f"{value:g}".replace(".", "p")
            cases[label] = ("viscous_sinking_v1", [f"plate_dynamics.young_slab_{key}={value}"], 1.)
    records = {}
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "MPLBACKEND": "Agg",
        "CUPY_CACHE_DIR": str(ROOT / ".venv/cupy-cache")}
    for label, (model, parameters, step) in cases.items():
        output = args.output / (label + ".json")
        command = [sys.executable, str(HERE / "run_case.py"), "frozen", "--source", str(args.source),
            "--output", str(output), "--slab-model", model, "--step-myr", str(step)]
        for item in parameters:
            command += ["--parameter", item]
        process = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8")
        (args.output / (label + ".log")).write_text(process.stdout + process.stderr, encoding="utf-8")
        if process.returncode:
            raise SystemExit(f"Frozen case {label} failed; see {args.output / (label + '.log')}")
        records[label] = json.loads(output.read_text(encoding="utf-8"))
        print(json.dumps(dict(case=label, metrics=records[label]["metrics"])), flush=True)
    base = records["sinking_default"]
    # Contrast and shear length vary resistance at fixed force loads. Bend
    # radius also changes the smooth shape and its gravitational projection;
    # those cases remain sensitivity data, without a monotonicity assertion.
    rhs_names = ("basal_driving_torque_nm", "ridge_torque_nm", "slab_torque_nm")
    working = [name for name in records if name not in ("disabled", "upper_bound")
               and not name.startswith("bend_radius_")]
    fixed_rhs = {name:all(np.array_equal(records[name]["trace"][key], base["trace"][key])
        for key in rhs_names) for name in working}
    def power(name):
        return records[name]["metrics"]["total_source_power_w"]
    def ordered(low, high):
        return power(low) <= power(high) + 1e-10*max(abs(power(low)), abs(power(high)), 1.)
    def resistance_order(low, middle, high):
        # Finite-strength failure can change the attached section set. That
        # changes gravity, hence is a response experiment rather than a
        # fixed-load passivity test. Record an inapplicable check as null.
        if not all(fixed_rhs[name] for name in (low, middle, high)):
            return None
        return ordered(low, middle) and ordered(middle, high)
    checks = dict(source_unchanged=all(row["source_unchanged"] for row in records.values()),
        contrast_higher_resistance_lower_source_work=resistance_order("viscosity_contrast_1000", "sinking_default", "viscosity_contrast_10"),
        larger_shear_length_lower_resistance_higher_work=resistance_order("mantle_shear_length_fraction_0p25", "sinking_default", "mantle_shear_length_fraction_1"),
        quasistatic_target_dt_independent=np.array_equal(base["trace"]["target_omega"], records["sinking_dt0p5"]["trace"]["target_omega"]),
        quasistatic_returned_dt_independent=np.array_equal(base["trace"]["final_omega"], records["sinking_dt0p5"]["trace"]["final_omega"]))
    summary = dict(source=str(args.source.resolve()), evaluation="frozen_counterfactual_force_probes_no_evolution",
        monotonicity_quantity="Equilibrium source work b^T omega* at fixed force RHS and admissible cone, not individual speeds. Bend-radius cases change gravity projection. Checks are null if finite-strength failure changes the RHS.",
        fixed_rhs_relative_to_default=fixed_rhs,
        checks=checks, cases={name:dict(parameter_overrides=row["parameter_overrides"],
            speeds_mm_yr=row["stages_speed_km_per_myr"], metrics=row["metrics"],
            inventory=row["accepted_slab_inventory"]) for name,row in records.items()})
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(checks), flush=True)
    if any(value is False for value in checks.values()):
        raise SystemExit("Frozen sensitivity checks did not all pass")


if __name__ == "__main__":
    main()
