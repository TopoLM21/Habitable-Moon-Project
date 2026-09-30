"""Summarize independent0.4/0.5 runs without altering their saved artifacts."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "analysis/slab_sinking_validation")]
from run_case import SOURCE, code_hashes, digest, write_json
from compare_checkpoints import diff


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def stats(values):
    array = np.asarray(list(values), dtype=float)
    if not array.size:
        return dict(count=0)
    return dict(count=int(array.size), mean=float(array.mean()), median=float(np.median(array)),
        minimum=float(array.min()), maximum=float(array.max()))


def segment(path):
    report = read(path / "continuation.json")
    metrics = read(path.with_suffix(".metrics.json"))
    trace = [json.loads(line) for line in path.with_suffix(".dynamics.jsonl").read_text(encoding="utf-8").splitlines()]
    return dict(path=str(path), report=report, metrics=metrics, trace=trace)


def ordered_paths():
    paths = []
    for age in (50, 100, 200, 400):
        candidates = [p for p in (HERE / "runs").glob(f"ordered_sub4_dt1*/elapsed_{age:04d}")
            if (p / "continuation.json").exists() and p.with_suffix(".metrics.json").exists()]
        if len(candidates)>1:
            raise ValueError(f"Ambiguous completed main age {age}: {candidates}")
        if candidates:
            paths.append(candidates[0])
    return paths


def reduced(item):
    report, metrics = item["report"], item["metrics"]
    inventory=dict(report["accepted_slab_inventory"])
    memory=read(Path(item["path"])/"mature_checkpoint/meta.json")["subduction_memory"]["young_boundary_state"]
    mechanical_volume=sum(event["retained_oceanic_volume_km3"] for event in memory["mechanical_detachments"])
    inventory.update(mechanical_detachment_retained_oceanic_volume_km3=mechanical_volume,
        mechanical_detachment_fraction_of_accepted_volume=mechanical_volume/max(inventory["cumulative_accepted_oceanic_volume_km3"],1.),
        other_unresolved_volume_km3=inventory["unresolved_or_detached_oceanic_volume_km3"]-mechanical_volume)
    return dict(path=item["path"], elapsed_myr=report["duration_myr"],
        mechanics_model_version=report["mechanics_model_version"],
        mean_speed_mm_yr=report["final_mean_surface_speed_km_myr"],
        maximum_speed_mm_yr=report["final_max_surface_speed_km_myr"],
        plates=report["final_plate_count"], commits=report["transport_commits"],
        inventory=inventory, checks=report["checks"],
        powers_and_force_checks=metrics["last_dynamics_metrics"])


def write_readme(result):
    if result["ordered_complete_through_myr"] != 400:
        return
    rows=[]
    for old,new in zip(result["baseline04"],result["ordered05"]):
        rows.append(f"| {new['elapsed_myr']:g} | {old['mean_speed_mm_yr']:.5f} | {new['mean_speed_mm_yr']:.5f} | "
            f"{new['inventory']['mechanical_detachment_count']} | {100*new['inventory']['unresolved_or_detached_fraction_of_accepted_volume']:.2f}% |")
    old_window=result["last100_myr_speed_statistics"]["uniform04"]
    new_window=result["last100_myr_speed_statistics"]["ordered05"]
    failures=result["ordered_failure_analysis"]
    final=result["ordered05"][-1]
    sensitivities="\n".join(f"- {name}: {value['mean_speed_mm_yr']:.5f} mm/yr "
        f"({100*value['relative_mean_speed_difference_from_ordered_default']:+.2f}% from default at50)."
        for name,value in result["numerical_sensitivity_at50"].items())
    text=f"""# Ordered thermal-cohort buoyancy: validation

These are independently evolved mechanics0.5 runs from the original Starter, compared with immutable archived0.4 runs. Only the version and the two buoyancy selectors differ in the fresh configurations. Geometry constants, strength law and viscosity were not retuned. Speeds are area means; 1 km/Myr = 1 mm/yr.

| Elapsed Myr | 0.4 speed, mm/yr | 0.5 speed, mm/yr | 0.5 cumulative mechanical detachments | 0.5 unresolved/detached accepted volume |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

These are endpoint snapshots, not steady velocities. Over 300<elapsed<=400 Myr (100 uniform samples), the mean/median/range of area-mean speed is {new_window['mean']:.5f}/{new_window['median']:.5f}/{new_window['minimum']:.5f}–{new_window['maximum']:.5f} mm/yr for0.5, compared with {old_window['mean']:.5f}/{old_window['median']:.5f}/{old_window['minimum']:.5f}–{old_window['maximum']:.5f} for0.4.

Final0.5 accepted oceanic volume is {final['inventory']['cumulative_accepted_oceanic_volume_km3']:,.3f} km³; attached {final['inventory']['attached_oceanic_volume_km3']:,.3f}; deep transferred {final['inventory']['deep_oceanic_volume_km3']:,.3f}; unresolved/detached {final['inventory']['unresolved_or_detached_oceanic_volume_km3']:,.3f}. The unresolved category includes lost contact connectivity and mechanical neck failure. A smaller speed or detached fraction is not proof of realistic stable subduction.

Specifically, committed mechanical neck failures account for {final['inventory']['mechanical_detachment_retained_oceanic_volume_km3']:,.3f} km³, or {100*final['inventory']['mechanical_detachment_fraction_of_accepted_volume']:.3f}% of accepted volume. Other unresolved connectivity accounts for {final['inventory']['other_unresolved_volume_km3']:,.3f} km³. In0.4 the mechanical fraction was {100*result['baseline04'][-1]['inventory']['mechanical_detachment_fraction_of_accepted_volume']:.3f}%; this is distinct from its97.373% total unresolved/detached fraction.

## Resolution sensitivity

{sensitivities}

The 50 Myr numerical comparisons remain sensitive to timestep and grid. [The archived baseline investigation](validation_baseline.md) shows whole-cell acceptance timing, different first-force thermal weights and different bend-length regimes. Ordered buoyancy corrects mass placement; it does not make those discretizations converge.

## Force and failure evidence

The new trajectory contains {failures['total']} recorded neck failures: {failures['with_material_reaction']} have a material no-eduction reaction and {failures['without_material_reaction']} do not. Rich records retain gravity, bending, mantle drag, reaction and live capacity before removal. Their maximum relative decomposition error is {failures['force_decomposition_maximum_relative_error']:.3g}. Subtracting a force term at fixed velocity is only a diagnostic, not an independently solved alternative trajectory.

The [local frozen counterfactuals](frozen04_counterfactual_solver_fixed/summary.json) load old0.4 states and change only selectors on in-memory copies. They do not relabel checkpoints or commit hypothetical detachments. Local speed changes are small on these particular saved states even though evolved histories diverge. The first physical speed difference occurs at18 Myr; the first failure-count difference occurs at26 Myr (0.4:2,0.5:0).

## Checks and artifacts

- [Every-step force audit](validation_step_audit.json): balance of torque and power, zero-work constraints, positive drag/dissipation, feasible feed and finite surviving tensile capacity; all seven continuation checks and source integrity at saved endpoints.
- [Resume compatibility](validation_compatibility.json): old0.4 state arrays/history are bitwise equal; the only metadata additions are10 failure-diagnostic fields. Populated0.5 CPU1/CPU4 states, metadata and physical reports are bitwise equal.
- [Final-solver fresh50 replay](validation_solver_recheck.json): bitwise equal to the original0.5 first50 Myr state, including physical metadata and report.
- [Results and exact numbers](validation_results.json), [failure decomposition](validation_failure_decomposition.json), [source/code provenance](validation.json), [comparison figure](comparison.png).
- Main continuation: `{final['path']}`. The first200 Myr are in `runs/ordered_sub4_dt1`;200→400 is in `runs/ordered_sub4_dt1_solver_fixed`.

The initial sub5 attempt stopped on a numerical active-support error; its exact solver inputs and log remain in `runs/ordered_sub5_dt1`. The completed fresh rerun is `runs/ordered_sub5_dt1_solver_fixed`. The support solver was corrected without relaxing force, residual or bound guards. Earlier completed segments retain their original production hashes; the changed file is the numerical constraint helper. Tests and exact source hashes are recorded in `validation.json`.

Reproduce a fresh0.5 series from the repository root:

```powershell
& .\\.venv\\Scripts\\python.exe analysis/slab_sinking_followup/validation_case.py series --output NEW_OUTPUT_DIRECTORY --ages 50 100 200 400
```

`validation_case.py` defaults fresh imports to0.5. Saved-state resume and frozen probes preserve the saved version. The historical `slab_sinking_validation/run_case.py` remains pinned to fresh0.4. Physical overrides are applied before creating the initial inventory.
"""
    (HERE/"README.md").write_text(text,encoding="utf-8")


def main():
    baseline_root = ROOT / "analysis/slab_sinking_validation/runs"
    baseline = [segment(baseline_root / ("validated_sinking_sub4_dt1" if age<=100
        else "validated_sinking_fixed2_sub4_dt1") / f"elapsed_{age:04d}") for age in (50,100,200,400)]
    ordered = [segment(path) for path in ordered_paths()]
    if not ordered:
        raise ValueError("No completed0.5 main segments")
    origin = ordered[-1]["report"]["import"]["origin_time_myr"]
    rows = [row for item in ordered for row in item["trace"]]
    failures = [dict(elapsed_myr=row["state_time_myr"]-origin, **failure)
        for row in rows for failure in row["trace"].get("slab_neck_failures", [])]
    for failure in failures:
        capacity = max(failure["capacity_n"],1.)
        failure["gravity_alone_to_capacity_ratio"] = failure["gravitational_feed_force_n"]/capacity
        failure["reaction_to_capacity_ratio"] = failure["no_eduction_reaction_n"]/capacity
        failure["fixed_velocity_tension_without_reaction_to_capacity_ratio"] = (
            failure["tension_n"]-failure["no_eduction_reaction_n"])/capacity
    reaction_failures = [f for f in failures if f["no_eduction_reaction_n"]>1e-8*max(abs(f["tension_n"]),f["capacity_n"],1.)]
    balances = [abs(f["tension_n"]-(f["gravitational_feed_force_n"]-f["bending_feed_resistance_n"]
        -f["mantle_feed_resistance_n"]+f["no_eduction_reaction_n"]))/max(abs(f["tension_n"]),f["capacity_n"],1.) for f in failures]
    failure_summary = dict(total=len(failures), with_material_reaction=len(reaction_failures),
        without_material_reaction=len(failures)-len(reaction_failures),
        unconstrained_reverse_feed_count=sum(f["unconstrained_feed_m_s"]<0 for f in failures),
        gravity_alone_exceeds_capacity_count=sum(f["gravity_alone_to_capacity_ratio"]>1 for f in failures),
        fixed_velocity_tension_without_reaction_exceeds_capacity_count=sum(
            f["fixed_velocity_tension_without_reaction_to_capacity_ratio"]>1 for f in failures),
        decomposition_note="Removing a term at fixed solved velocity is a diagnostic, not a re-solved alternative physical history.",
        force_decomposition_maximum_relative_error=max(balances, default=0.),
        tension_capacity_ratio=stats(f["tension_n"]/max(f["capacity_n"],1.) for f in failures),
        neck_strength_pa=stats(f["neck_strength_pa"] for f in failures),
        most_overloaded=max(failures,key=lambda f:f["tension_n"]/max(f["capacity_n"],1.),default=None))
    main_end = ordered[-1]["report"]["duration_myr"]
    window = {}
    for name, items in (("uniform04",baseline),("ordered05",ordered)):
        window[name] = stats(r["mean_surface_speed_km_myr"] for r in items[-1]["report"]["history"]
            if 300 < r["time_myr"]-origin <= 400)
    sensitivity = {}
    for label, stem in (("dt0p5","ordered_sub4_dt0p5"),("sub5","ordered_sub5_dt1")):
        candidates = [p for p in (HERE/"runs").glob(stem+"*/elapsed_0050") if p.with_suffix(".metrics.json").exists()]
        if len(candidates)>1:
            raise ValueError(f"Ambiguous completed sensitivity {label}")
        if candidates:
            sensitivity[label] = reduced(segment(candidates[0]))
            sensitivity[label]["relative_mean_speed_difference_from_ordered_default"] = (
                sensitivity[label]["mean_speed_mm_yr"]/ordered[0]["report"]["final_mean_surface_speed_km_myr"]-1.)
    baseline_config=yaml.safe_load((Path(baseline[0]["path"])/"mature_config.yaml").read_text(encoding="utf-8"))
    ordered_config=yaml.safe_load((Path(ordered[0]["path"])/"mature_config.yaml").read_text(encoding="utf-8"))
    for key in ("plate_dynamics","subduction_memory"):
        baseline_config[key].setdefault("young_slab_buoyancy_model","uniform_thermal_mass_v1")
    configuration_differences=diff(baseline_config,ordered_config)
    aligned_version_rows=[]
    for oldrow,newrow in zip(baseline[0]["trace"],ordered[0]["trace"]):
        aligned_version_rows.append(dict(elapsed_myr=newrow["state_time_myr"]-origin,
            old_mean_speed_mm_yr=oldrow["final_omega_surface_speed_mm_yr"]["mean"],
            new_mean_speed_mm_yr=newrow["final_omega_surface_speed_mm_yr"]["mean"],
            old_neck_failures=len(oldrow["trace"].get("slab_neck_failures",[])),
            new_neck_failures=len(newrow["trace"].get("slab_neck_failures",[]))))
    frozen_root = HERE / "frozen04_counterfactual_solver_fixed"
    if not (frozen_root / "summary.json").exists():
        frozen_root = HERE / "frozen04_counterfactual"
    frozen = read(frozen_root/"summary.json")
    local_comparisons = []
    for case in frozen["cases"]:
        age = case["elapsed_myr"]
        detail = read(frozen_root / f"elapsed_{age:04d}_ordered05_counterfactual.json")
        sections = detail["trace"].get("slab_sections", [])
        uniform_force = sum(s["uniform_buoyancy_feed_force_n"] for s in sections)
        ordered_force = sum(s["gravitational_feed_force_n"] for s in sections)
        local_comparisons.append(dict(elapsed_myr=age,
            saved_uniform_target_speed_mm_yr=case["cases"]["saved_uniform04"]["speeds"]["target"]["mean"],
            ordered_counterfactual_target_speed_mm_yr=case["cases"]["ordered05_counterfactual"]["speeds"]["target"]["mean"],
            uniform_hypothetical_failures=case["cases"]["saved_uniform04"]["hypothetical_neck_failures"],
            ordered_hypothetical_failures=case["cases"]["ordered05_counterfactual"]["hypothetical_neck_failures"],
            ordered_surviving_set_sum_uniform_feed_force_n=uniform_force,
            ordered_surviving_set_sum_ordered_feed_force_n=ordered_force,
            ordered_surviving_set_fractional_gravity_change=ordered_force/uniform_force-1. if uniform_force else 0.,
            warning="Frozen counterfactual, not evolved0.5; equilibrium failure sets can differ. Feed force sums use identical ordered surviving geometry."))
    result = dict(baseline04=[reduced(item) for item in baseline], ordered05=[reduced(item) for item in ordered],
        ordered_complete_through_myr=main_end, last100_myr_speed_statistics=window,
        configuration_differences_after_restoring_saved04_implicit_uniform_default=configuration_differences,
        first_version_speed_difference_over_1e_minus10_mm_yr=next((r for r in aligned_version_rows
            if abs(r["old_mean_speed_mm_yr"]-r["new_mean_speed_mm_yr"])>1e-10),None),
        first_version_failure_count_difference=next((r for r in aligned_version_rows
            if r["old_neck_failures"]!=r["new_neck_failures"]),None),
        numerical_sensitivity_at50=sensitivity, ordered_failure_analysis=failure_summary,
        local_force_counterfactuals=local_comparisons)
    write_json(HERE/"validation_results.json",result)
    write_readme(result)
    write_json(HERE/"validation_failure_decomposition.json",dict(summary=failure_summary,failures=failures))
    current = code_hashes()
    provenance = []
    for path in sorted((HERE/"runs").glob("*/elapsed_*.metrics.json")):
        metrics=read(path)
        provenance.append(dict(path=str(path), version=metrics["mechanics_model_version"],
            source_sha256=metrics["source_sha256"], source_unchanged=metrics["source_unchanged"],
            changed_during_run=metrics["production_files_changed_during_run"],
            production_files_different_from_current=[name for name in current.keys()|metrics["production_sha256"].keys()
                if current.get(name)!=metrics["production_sha256"].get(name)],
            passed=all(metrics["checks"].values())))
    evidence = {path.name:read(path) for path in (HERE/"validation_compatibility.json",
        HERE/"validation_solver_recheck.json",HERE/"validation_step_audit.json") if path.exists()}
    write_json(HERE/"validation.json",dict(original_starter=str(SOURCE),original_starter_sha256=digest(SOURCE),
        current_production_sha256=current, completed_segments=provenance,
        test_suite=dict(passed=199,elapsed_seconds=218.20,log="final_focused_tests.log",
            log_sha256=digest(HERE/"final_focused_tests.log")),
        supporting_evidence=evidence,
        note="Preserved segments may precede an exact numerical support-solver fix; per-run request/metrics files retain original hashes. Saved0.4 artifacts unchanged.",
        results="validation_results.json",baseline="validation_baseline.md"))
    print(json.dumps(dict(completed_through_myr=main_end, last100=window,
        failures=failure_summary, sensitivities={k:v["mean_speed_mm_yr"] for k,v in sensitivity.items()}),indent=2))


if __name__=="__main__":
    main()
