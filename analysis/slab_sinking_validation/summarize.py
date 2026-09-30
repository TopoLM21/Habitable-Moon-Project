"""Summarize completed arms and explicitly separate frozen force experiments."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def differences(first, second, prefix=""):
    if isinstance(first, dict) and isinstance(second, dict):
        result = {}
        for key in sorted(first.keys() | second.keys()):
            result.update(differences(first.get(key), second.get(key), f"{prefix}.{key}" if prefix else key))
        return result
    return {} if first == second else {prefix:[first, second]}


def compact_run(path):
    report = read(path / "continuation.json")
    metrics_path = path.with_suffix(".metrics.json")
    harness = read(metrics_path) if metrics_path.exists() else {}
    inventory = report.get("accepted_slab_inventory") or {}
    history = report["history"]
    last100 = [row["mean_surface_speed_km_myr"] for row in history
               if report["final_time_myr"]-100. < row["time_myr"] <= report["final_time_myr"]+1e-9]
    return dict(path=str(path), age_myr=report["duration_myr"],
        step_myr=report["step_myr"], mechanics_model_version=report.get("mechanics_model_version"),
        slab_force_model=report.get("young_slab_force_model"),
        status=report["status"], checks=report["checks"],
        mean_speed_mm_yr=report["final_mean_surface_speed_km_myr"],
        max_speed_mm_yr=report["final_max_surface_speed_km_myr"],
        plates=report["final_plate_count"], transport_commits=report["transport_commits"],
        inventory=inventory, final_mantle_temperature_k=history[-1]["mantle_temperature_k"],
        thermal_energy_relative_residual=history[-1]["thermal_energy_relative_residual"],
        material_relative_volume_residual=report["material_ledger"]["relative_volume_residual"],
        first_transport_age_myr=next((row["time_myr"]-report["import"]["origin_time_myr"]
            for row in history if row["transport_commits"]), None),
        last100_myr_mean_speed_temporal_statistics=None if report["duration_myr"] < 100. else dict(
            sample_count=len(last100), time_mean_mm_yr=float(np.mean(last100)),
            time_median_mm_yr=float(np.median(last100)), minimum_mm_yr=float(np.min(last100)),
            maximum_mm_yr=float(np.max(last100))),
        last_force_metrics=harness.get("last_dynamics_metrics"),
        production_files_changed_during_run=harness.get("production_files_changed_during_run"),
        source_unchanged=harness.get("source_unchanged"),
        parameter_overrides=harness.get("parameter_overrides"))


def compact_frozen(path):
    report = read(path)
    return dict(path=str(path), source=report["source"], time_myr=report["time_myr"],
        evaluation=report["evaluation"], parameter_overrides=report["parameter_overrides"],
        source_unchanged=report["source_unchanged"],
        speeds_mm_yr=report["stages_speed_km_per_myr"],
        component_speeds_mm_yr=report["contributions_speed_km_per_myr"],
        metrics=report["metrics"], inventory=report["accepted_slab_inventory"])


def trace_ancestry(path):
    """Include earlier saved segments when a numerical repair used a new arm folder."""
    result, visited = [], set()
    while path.is_dir() and path.resolve() not in visited:
        visited.add(path.resolve())
        trace = path.with_suffix(".dynamics.jsonl")
        if trace.exists():
            result.append(trace)
        metrics = path.with_suffix(".metrics.json")
        if not metrics.exists():
            break
        source = read(metrics).get("source")
        if not source:
            break
        path = Path(source)
    return list(reversed(result))


def plot(arms, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for name, cases in arms.items():
        if not cases:
            continue
        path = Path(cases[-1]["path"])
        report = read(path / "continuation.json")
        history = report["history"]
        age = [row["time_myr"]-report["import"]["origin_time_myr"] for row in history]
        match = re.search(r"sub(\d+)_dt([\dp]+)", name)
        label = "Погружение выключено" if "disabled" in name else "Погружение и изгиб"
        if match:
            label += f", сетка {match[1]}, шаг {match[2].replace('p', '.')}"
        line, = axes[0, 0].plot(age, [row["mean_surface_speed_km_myr"] for row in history], label=label)
        axes[0, 1].plot(age, [row["max_surface_speed_km_myr"] for row in history], color=line.get_color())
        axes[1, 0].plot(age, [row["transport_commits"] for row in history], color=line.get_color())
        trace_files = trace_ancestry(path)
        force_rows = [json.loads(line) for trace_file in trace_files if trace_file.exists()
                      for line in trace_file.read_text(encoding="utf-8").splitlines()]
        if force_rows:
            axes[1, 1].plot([row["state_time_myr"]-report["import"]["origin_time_myr"] for row in force_rows],
                [row["metrics"].get("target_torque_relative_residual", np.nan) for row in force_rows],
                color=line.get_color())
    for ax in axes.flat:
        ax.grid(alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("Время после первого разделения, млн лет")
    axes[0, 0].set_ylabel("Средняя скорость, мм/год")
    axes[0, 1].set_ylabel("Максимальная скорость, мм/год")
    axes[1, 0].set_ylabel("Число растровых переносов")
    axes[1, 1].set_ylabel("Относительный остаток целевого баланса сил")
    axes[1, 1].set_yscale("log")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 2), frameon=False)
    fig.suptitle("Проверка механики погружения: скорость, перенос и баланс сил")
    fig.tight_layout(rect=(0, .07, 1, .96))
    fig.savefig(output, dpi=170)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=HERE / "runs")
    parser.add_argument("--arm-glob", default="*", help="Use final_* to exclude rejected interim arms")
    parser.add_argument("--frozen", type=Path, default=HERE / "frozen")
    parser.add_argument("--output", type=Path, default=HERE / "comparison.json")
    parser.add_argument("--plot", type=Path)
    parser.add_argument("--plot-arms", nargs="+", help="Exact arm folder names to draw; summary retains all matching arms")
    args = parser.parse_args()
    arms = {}
    for folder in sorted(args.runs.glob(args.arm_glob)):
        if not folder.is_dir():
            continue
        cases = [compact_run(path.parent) for path in folder.glob("elapsed_*/continuation.json")]
        if cases:
            arms[folder.name] = sorted(cases, key=lambda case: case["age_myr"])
    frozen = [compact_frozen(path) for path in sorted(args.frozen.rglob("*.json"))
              if "trace" in read(path)]
    disabled = {case["age_myr"]:case for name,cases in arms.items() if "disabled" in name for case in cases}
    for cases in arms.values():
        for case in cases:
            reference = disabled.get(case["age_myr"])
            if reference is None:
                continue
            case["mantle_temperature_difference_from_disabled_k"] = case["final_mantle_temperature_k"]-reference["final_mantle_temperature_k"]
            original_cfg = yaml.safe_load((Path(reference["path"])/"mature_config.yaml").read_text(encoding="utf-8"))
            current_cfg = yaml.safe_load((Path(case["path"])/"mature_config.yaml").read_text(encoding="utf-8"))
            keys = ("plate_dynamics", "subduction_memory", "young_shell", "mesh", "thermal_evolution")
            case["config_differences_from_disabled"] = differences({key:original_cfg.get(key) for key in keys},
                {key:current_cfg.get(key) for key in keys})
    historical_path = ROOT / "analysis/young_mechanics_validation/final_after_seed_fix/corrected_mechanics/elapsed_0400"
    historical = compact_run(historical_path) if historical_path.exists() else None
    result = dict(scope="Independent evolving arms and counterfactual frozen probes are separate evidence.",
        original_starter_sha256=hashlib.sha256((ROOT / "results/genesis_runs/starter_20260928_192306_547551/starter_checkpoint.npz").read_bytes()).hexdigest(),
        arms=arms, frozen=frozen, historical_03_disabled_relaxed_400=historical,
        note="A small target residual alone does not establish physical closure of a relaxed transient velocity. Inspect source power, all positive dissipations, and transient residual work together.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    if args.plot and arms:
        selected = arms if args.plot_arms is None else {name:arms[name] for name in args.plot_arms}
        plot(selected, args.plot)
    print(json.dumps(dict(output=str(args.output), arms={name:len(cases) for name,cases in arms.items()}, frozen_probes=len(frozen))))


if __name__ == "__main__":
    main()
