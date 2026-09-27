"""Reproduce the actual coupled stopping event and summarize long continuations."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.genesis_coupled_validation import _fingerprint
from tectonics.genesis_coupled import load_coupled_checkpoint
from tectonics.genesis_shell_release import FrozenShell

RUNS = ROOT/"results/genesis_runs"
COUPLED = ("coupled_growth_final_20260924", "long_time_coupled_20260925", "long_time_coupled_limit_20260925")
SOURCES = {
    "strong320": "fault_damage_final_20260922/fault_checkpoint.npz",
    "strong1280": "fine_fault_audit_20260924/strong_1280/fault_checkpoint.npz",
    "strong5120": "fine_fault_audit_20260924/strong_5120/fault_checkpoint.npz",
    "user5120": "genesis_20260924_194255_087614/fault_checkpoint.npz",
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("output must be new or empty")
    args.output.mkdir(parents=True, exist_ok=True)
    sources, snapshots, rows, provenance = {}, {}, [], {}
    for name in COUPLED:
        path = RUNS/name/"coupled_checkpoint.npz"
        provenance[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        model, state, thermal, orbit = load_coupled_checkpoint(path)
        snapshots[name] = model.diagnostics(state, thermal, orbit)
        with (RUNS/name/"coupled_history.csv").open(encoding="utf-8") as handle:
            rows.extend(dict(row) for row in csv.DictReader(handle))
    rows = list({float(row["elapsed_years"]): row for row in rows}.values())
    rows.sort(key=lambda row: float(row["elapsed_years"]))
    # Repeat precisely the same output targets and original saved controls.
    model, state, thermal, orbit = load_coupled_checkpoint(RUNS/COUPLED[1]/"coupled_checkpoint.npz")
    expected = load_coupled_checkpoint(RUNS/COUPLED[2]/"coupled_checkpoint.npz")[1:]
    start, started = state.time_myr, perf_counter()
    for index in range(1, 51):
        state, thermal, orbit, _ = model.step(state, thermal, orbit, start+index*.002,
                                              max_step_myr=.002)
        if state.stopped_reason:
            break
    restart = {"bitwise_equal_state_heat_orbit": _fingerprint((state, thermal, orbit)) == _fingerprint(expected),
               "status": state.stopped_reason, "wall_seconds": perf_counter()-started}
    for name, relative in SOURCES.items():
        path = RUNS/relative
        provenance[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        shell = FrozenShell.from_fault_checkpoint(path)
        equilibrium = shell.solve(np.empty((0, 2), dtype=np.int64))
        f, g = equilibrium.external_force, equilibrium.initial_bulk_force
        scale = max(np.linalg.norm(f), np.linalg.norm(g))
        sources[name] = {
            "inherited_intact_force_residual": float(np.linalg.norm(g-f)/scale),
            "full_load_equilibrium_residual": equilibrium.equilibrium_residual,
            "full_load_added_strain": equilibrium.max_added_strain,
            "full_load_max_displacement_m": float(np.max(np.abs(equilibrium.displacement_m))),
            "admissible_intact_equilibrium": equilibrium.admissible_open_crack,
        }
    balance = {}
    for name in ("strong5120", "user5120", "evolved320"):
        path = RUNS/"physical_balance_verified_20260925"/name/"balance.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        balance[name] = {"checks": report["checks"], "reached_requested_duration": report["reached_requested_duration"],
            "accepted_steps": report["accepted_steps"],
            "max_force_residual": max(r["force_balance_relative_residual"] for r in report["steps"]),
            "max_residual_omitting_drag": max(r["static_relative_residual_without_drag"] for r in report["steps"])}
        provenance[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    final = snapshots[COUPLED[2]]
    checks = {
        "restart_reproduces_limit": restart["bitwise_equal_state_heat_orbit"],
        "stopped_on_total_reference_strain": final["stopped_reason"] == "coupled_reference_geometry_limit" and .99999 < final["added_strain_utilization"] <= 1.,
        "other_recorded_guards_below_limit": max(final[k] for k in ("motion_utilization", "small_sliding_utilization", "penetration_utilization", "elastic_strain_utilization", "interface_area_utilization", "cohort_count_utilization")) < 1.,
        "cohesive_shell_still_connected": final["cohesive_component_count"] == 1,
        "mass_thermal_orbit_budgets": all(abs(final[k]) < 1e-10 for k in ("relative_mass_residual", "relative_global_energy_residual", "relative_column_energy_residual", "orbit_heat_transfer_relative_residual")),
        "intact_source_force_balance": all(r["inherited_intact_force_residual"] < 1e-6 for r in sources.values()),
        "intact_full_load_equilibria_valid": all(r["admissible_intact_equilibrium"] for r in sources.values()),
        "physical_step_balance": all(all(r["checks"].values()) and r["reached_requested_duration"] for r in balance.values()),
        "sources_unchanged": all(hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == digest for path, digest in provenance.items()),
    }
    report = {"snapshots": snapshots, "restart": restart, "intact_sources": sources,
        "physical_step_balance": balance, "checks": checks, "source_sha256": provenance,
        "limitations": ["This is one 320-cell strong-load scenario, not a grid-converged physical lifetime.",
            "Total strain .005 is the supported reference-geometry threshold, not a physical arrest mechanism.",
            "Legacy edge selection still makes a two-cell fragment; apparent components are not validated plates.",
            "The shrinking final step cannot remove accumulated reference distortion.",
            "The separate default-load long run has different parameters and tests insufficient elapsed time before activation."],
        "implementation_sha256": {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
            for name in ("tectonics/genesis_coupled.py", "tectonics/genesis_coupled_thermal.py", "tectonics/genesis_coupled_checkpoint.py",
                         "tectonics/genesis_contact_growth.py", "tectonics/genesis_shell_release.py", "analysis/genesis_long_time_audit.py")}}
    (args.output/"audit.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    plot(rows, snapshots, args.output/"long_coupled.png")
    print(json.dumps({"checks": checks, "restart": restart, "final_elapsed_years": final["elapsed_years"]}), flush=True)
    return 0 if all(checks.values()) else 1


def plot(rows, snapshots, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    time = np.array([float(row["elapsed_years"]) for row in rows])/1000
    with plt.rc_context({"font.size": 10, "figure.facecolor": "#f8fafc"}):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for key, label in (("max_opening_m", "Раскрытие"), ("max_abs_jump_m", "Сдвиг")):
            axes[0, 0].plot(time, [float(r[key])/1000 for r in rows], label=label)
        axes[0, 0].set(ylabel="км", title="Движение продолжается")
        for key, label in (("seam_count", "Все разрезанные рёбра"), ("fully_decohered_seam_count", "Полностью утратившие сцепление")):
            available = [r for r in rows if key in r]
            axes[0, 1].plot([float(r["elapsed_years"])/1000 for r in available], [float(r[key]) for r in available], label=label)
        axes[0, 1].set(title="Оболочка остаётся связной", ylabel="Число рёбер")
        times = [r["elapsed_years"]/1000 for r in snapshots.values()]
        for key, label in (("added_strain_utilization", "Накопленная деформация"), ("small_sliding_utilization", "Смещение берегов"), ("cohort_count_utilization", "Число контактных слоёв")):
            axes[1, 0].plot(times, [r[key] for r in snapshots.values()], "o-", label=label)
        axes[1, 0].axhline(1, color="#b91c1c", ls="--", label="Предел модели")
        axes[1, 0].set(title="Что остановило продолжение", ylabel="Доля допустимого предела")
        stepped = [r for r in rows if float(r["last_step_years"]) > 0]
        axes[1, 1].semilogy([float(r["elapsed_years"])/1000 for r in stepped],
                            [float(r["last_step_years"]) for r in stepped], ".-")
        axes[1, 1].set(title="Последний принятый шаг в точке вывода", ylabel="лет")
        for ax in axes.flat:
            ax.set_xlabel("Время после начала контакта, тыс. лет")
            ax.grid(alpha=.2)
        for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
            ax.legend(fontsize=8)
        fig.suptitle("Продлённый генезис: 320 ячеек, нагрузка 50 кПа", fontsize=15)
        fig.text(.5, .02, "Остановка при 102,6 тыс. лет — предел опорной геометрии, не физическая остановка тектоники.", ha="center", fontsize=10)
        fig.tight_layout(rect=(0, .045, 1, .96))
        fig.savefig(output, dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
