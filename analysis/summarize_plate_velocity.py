"""Build compact, reproducible comparison tables and figures from audited runs."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT/"analysis/plate_velocity_validation"


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def main():
    from tectonics.genesis_starter_continuation import load_starter_source
    from tectonics.genesis_starter_material import independent_mantle_omega
    from tectonics.plate_velocity_diagnostics import weighted_stats, diagnose_checkpoint
    from tectonics.genesis import SECONDS_PER_MYR
    source = ROOT/"results/genesis_runs/starter_20260928_192306_547551/starter_checkpoint.npz"
    model, state, metadata = load_starter_source(source)
    sample = model.loading.sample(state.thermal_context)
    coupling = 1.-np.exp(-sample.lid_thickness_km/model.shell.traction_coupling_depth_km)
    field = independent_mantle_omega(model, state)
    local = np.linalg.norm(np.cross(field, model.mesh.centroids), axis=1)*model.thermal.radius_km
    dump(OUT/"source.json", dict(source=str(source), sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        time_myr=state.time_myr, plate_count=len(state.system.plates),
        traction_pa=model.shell.convective_traction_pa, basal_drag_pa_s_m=model.parameters.basal_drag_pa_s_m,
        mechanical_lid_thickness_km=sample.lid_thickness_km, traction_coupling=coupling,
        nominal_traction_over_drag_km_myr=model.shell.convective_traction_pa/model.parameters.basal_drag_pa_s_m*SECONDS_PER_MYR/1000.,
        actual_local_velocity_km_myr=weighted_stats(local, model.areas),
        hypothetical_saturated_coupling_velocity_km_myr=weighted_stats(local/coupling, model.areas),
        first_partition_fit_relative_residual=state.kinematic_fit_relative_residual,
        first_partition_represented_kinetic_fraction=1.-state.kinematic_fit_relative_residual**2))
    replay = OUT/"experiments/observed_replay_110"
    original = ROOT/"results/gui_runs/genesis_20260928_192334_470716/gui_checkpoint_0000110p8781_Myr"
    matched = {}
    for relative in ("mature_checkpoint/state.npz", "young_context/starter_checkpoint.npz", "young_context/fracture_memory.npz"):
        with np.load(replay/relative) as a, np.load(original/relative) as b:
            matched[relative] = a.files == b.files and all(np.array_equal(a[k], b[k]) for k in a.files)
    lines = replay.with_suffix(".dynamics.jsonl").read_text(encoding="utf-8").splitlines()
    recorded = json.loads(lines[-1])
    budget = recorded["velocity_budget"]
    dump(OUT/"observed_actual_step.json", dict(input_time_myr=recorded["state_time_myr"],
        output_time_myr=recorded["state_time_myr"]+recorded["dt_myr"],
        replay_arrays_bitwise_equal=matched, **budget))
    rows, histories = [], {}
    snapshots = OUT/"comparison_checkpoints"
    snapshots.mkdir(exist_ok=True)
    for mode in ("legacy", "velocity_least_squares"):
        mode_history = []
        for p in sorted((OUT/"experiments"/mode).glob("*.metrics.json")):
            report = json.loads(p.read_text(encoding="utf-8"))
            run = Path(report["output"])
            history = report["history"]
            last = history[-1]
            detailed = diagnose_checkpoint(run, projection="legacy_area_mean" if mode == "legacy" else mode)
            dump(snapshots/f"{mode}_{last['time_myr']:.8f}.json", detailed)
            b = detailed["boundaries"]
            row = dict(mode=mode, elapsed_myr=last["time_myr"]-state.time_myr,
                age_myr=last["time_myr"], plates=report["final_plate_count"],
                mean_speed_km_myr=report["final_mean_surface_speed_km_myr"],
                max_speed_km_myr=report["final_max_surface_speed_km_myr"],
                mantle_rms_km_myr=detailed["mantle"]["local_velocity_km_per_myr"]["rms"],
                fit_relative_residual=detailed["mantle"]["rigid_fit_relative_residual"],
                boundary_count=b["count"], active_boundary_length_fraction=b["active_length_fraction"],
                max_relative_boundary_speed_km_myr=b["relative_speed_km_per_myr"]["max"],
                ridge_mean_speed_contribution_km_myr=detailed["contributions_speed_km_per_myr"]["ridge"]["mean"],
                slab_mean_speed_contribution_km_myr=detailed["contributions_speed_km_per_myr"]["slab"]["mean"],
                slab_zones=len(detailed["slabs"]), transport_commits=report["transport_commits"],
                continental_volume_km3=last["continental_volume_km3"],
                mantle_temperature_k=last["mantle_temperature_k"],
                thermal_energy_relative_residual=last["thermal_energy_relative_residual"],
                material_relative_residual=report["material_ledger"]["relative_volume_residual"],
                all_checks=all(report["checks"].values()))
            rows.append(row)
            mode_history = history
        histories[mode] = mode_history
    with (OUT/"old_vs_new.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    paired = {}
    for old, new in zip(histories["legacy"], histories["velocity_least_squares"]):
        for key in ("mantle_temperature_k", "surface_temperature_k", "mantle_to_surface_flux_w_m2",
                    "thermal_energy_relative_residual", "ocean_fraction"):
            paired[key] = max(paired.get(key, 0.), abs(old[key]-new[key]))
    dump(OUT/"comparison.json", dict(rows=rows, maximum_thermal_history_absolute_differences=paired,
        observed_replay_arrays_bitwise_equal=matched,
        note="Both comparison arms use original subdivision4; actual user replay uses saved subdivision5. Physical coefficients unchanged; mode is the only equation selection."))
    make_plots(budget, histories, rows)
    print(json.dumps(dict(comparison_rows=len(rows), replay_arrays_equal=all(matched.values()),
                         thermal_difference=paired), indent=2))


def make_plots(budget, histories, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    stages = budget["stages_speed_km_per_myr"]
    keys = ["local_mantle", "best_rigid_fit", "simple_area_average", "after_memory", "target", "after_relaxation", "after_net_rotation"]
    labels = ["Локальный поток", "Оптимальный rigid fit", "Старое усреднение", "После ×0,22", "Target (прочие силы = 0)", "После релаксации", "После удаления вращения"]
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    y = np.arange(len(keys))
    ax.barh(y-.16, [stages[k]["mean"] for k in keys], height=.3, label="Средний модуль", color="#3671a7")
    ax.barh(y+.16, [stages[k]["rms"] for k in keys], height=.3, label="RMS", color="#b3cedf")
    for i, k in enumerate(keys):
        ax.text(stages[k]["mean"]+.012, i-.16, f"{stages[k]['mean']:.6f}", va="center", fontsize=9)
    ax.set(yticks=y, yticklabels=labels, xlabel="Скорость, км / млн лет (= мм/год)",
           title="Фактический шаг 109,878 → 110,878 млн лет\nОптимальный fit показан как сравнение; старый прогон использовал усреднение")
    ax.invert_yaxis(); ax.legend(loc="lower right"); ax.grid(axis="x", alpha=.18)
    fig.savefig(OUT/"observed_speed_decomposition.png", dpi=160); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for mode, label in (("legacy", "Старое усреднение"), ("velocity_least_squares", "Velocity least squares")):
        history = histories[mode]
        t = [r["time_myr"] for r in history]
        axes[0, 0].plot(t, [r["mean_surface_speed_km_myr"] for r in history], label=label)
        axes[0, 1].plot(t, [r["max_surface_speed_km_myr"] for r in history], label=label)
        axes[1, 0].step(t, [r["plate_count"] for r in history], where="post", label=label)
        axes[1, 1].plot(t, [r["mantle_temperature_k"] for r in history], label=label)
    for ax, title, ylabel in zip(axes.flat, ["Средняя скорость", "Максимальная скорость", "Число плит", "Одинаковая тепловая история"],
                                ["км / млн лет", "км / млн лет", "Плиты", "K"]):
        ax.set(title=title, xlabel="Возраст, млн лет", ylabel=ylabel); ax.grid(alpha=.2)
    axes[0, 0].legend(); axes[1, 0].set_ylim(1.5, 4.5)
    fig.suptitle("Тот же источник Genesis, 0,08 МПа; изменена только математическая проекция")
    fig.savefig(OUT/"old_vs_new.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    main()
