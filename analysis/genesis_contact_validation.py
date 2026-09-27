"""Step and normal-penalty sensitivity for one frozen paired-bank snapshot."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tectonics.genesis_contact import CONTACT_VERSION, ContactModel, ContactParameters, save_contact_checkpoint
from tectonics.genesis_contact_law import ContactLawParameters


def sample(model, state):
    row = model.diagnostics(state)
    bulk_release = model.initial_elastic_energy_j-row["bulk_elastic_energy_j"]
    energy_scale = abs(bulk_release)+abs(row["external_work_j"])
    row.update(bulk_elastic_energy_released_j=bulk_release,
               energy_release_scale_j=energy_scale,
               mechanical_energy_remainder_relative_release=(
                   row["mechanical_energy_remainder_j"]/energy_scale if energy_scale > 0 else 0.))
    return row


def simulate(source, source_path, parameters, law, dt, duration, name, output):
    model = ContactModel(source, parameters, law, str(source_path.resolve()))
    state = model.initial()
    history = [sample(model, state)]
    for index in range(1, math.ceil(duration/dt)+1):
        state = model.step(state, min(index*dt, duration))
        row = sample(model, state)
        history.append(row)
        if index % max(1, round(200/dt)) == 0 or state.stopped_reason:
            print(f"{name}: t={state.elapsed_years:g} yr, opening={row['max_opening_m']:.6g} m, "
                  f"penetration={row['max_penetration_m']:.6g} m, "
                  f"status={state.stopped_reason or 'running'}", flush=True)
        if state.stopped_reason:
            break
    save_contact_checkpoint(output/f"{name}_checkpoint.npz", model, state)
    return {
        "model_version": CONTACT_VERSION,
        "parameters": {"contact": asdict(parameters), "law": asdict(law)},
        "controls": {"requested_step_years": dt, "requested_duration_years": duration},
        "status": state.stopped_reason or "completed",
        "source_hash": model.source_hash,
        "initial_bulk_elastic_energy_j": model.initial_elastic_energy_j,
        "frozen_column_energy_j": model.column_energy_j,
        "material_mass_kg": model.initial_mass_kg,
        "material_face_count": model.topology.mesh.cell_count,
        "final": history[-1],
        "maximum_sampled_penetration_m": max(row["max_penetration_m"] for row in history),
        "maximum_sampled_equilibrium_residual": max(row["equilibrium_residual"] for row in history),
        "history": history,
    }


def compare(cases):
    indexed = {name: {round(row["elapsed_years"], 9): row for row in case["history"]}
               for name, case in cases.items()}
    common = sorted(set.intersection(*(set(rows) for rows in indexed.values())))
    age = common[-1]
    metrics = ("max_opening_m", "max_abs_jump_m", "max_penetration_m", "max_plastic_slip_m",
               "fractured_endpoint_fraction", "drag_work_j", "friction_work_j", "fracture_work_j",
               "mechanical_energy_remainder_relative_release")
    values = {name: {key: rows[age][key] for key in metrics} for name, rows in indexed.items()}
    reference = values["step_10yr"]
    return {
        "latest_common_elapsed_years": age,
        "latest_common_values": values,
        "differences_from_step_10yr": {
            name: {key: {"signed_difference": value-reference[key],
                         "relative_difference": ((value-reference[key])/abs(reference[key])
                                                  if reference[key] != 0 else None)}
                   for key, value in case.items()}
            for name, case in values.items() if name != "step_10yr"
        },
        "method": "Compare stored samples at identical elapsed times without interpolation.",
        "interpretation": "Time-step and normal-penalty sensitivity only; neither a mesh convergence proof nor a physical calibration.",
    }


def plot(cases, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {"step_10yr": "Шаг 10 лет · Kn = 10 МПа/м",
              "step_5yr": "Шаг 5 лет · Kn = 10 МПа/м",
              "normal_penalty_double": "Шаг 10 лет · Kn = 20 МПа/м"}
    colors = {"step_10yr": "#356ca9", "step_5yr": "#d27935", "normal_penalty_double": "#8a58a7"}
    panels = (("max_opening_m", "Раскрытие независимых берегов", "Максимальное раскрытие, м"),
              ("max_abs_jump_m", "Относительное скольжение", "Максимальный сдвиг, м"),
              ("max_penetration_m", "Податливость нормального контакта", "Максимальное проникновение, м"),
              ("mechanical_energy_remainder_relative_release", "Невязка механического бюджета", "% от высвобождения энергии"))
    with plt.rc_context({"font.size": 10, "axes.titlesize": 11, "figure.facecolor": "#f7f9fc"}):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
        try:
            for ax, (key, title, ylabel) in zip(axes.flat, panels):
                for name, case in cases.items():
                    factor = 100 if key.endswith("relative_release") else 1
                    ax.plot([row["elapsed_years"] for row in case["history"]],
                            [factor*row[key] for row in case["history"]],
                            color=colors[name], label=labels[name], linewidth=1.7,
                            linestyle="--" if name == "step_5yr" else "-")
                ax.set(title=title, xlabel="Время после разделения берегов, лет", ylabel=ylabel)
                ax.grid(alpha=.22)
                if key.endswith("relative_release"):
                    ax.axhline(0., color="#697585", linewidth=.8)
                else:
                    ax.set_ylim(bottom=0.)
            handles, names = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, names, loc="upper center", bbox_to_anchor=(.5, .925), frameon=False, ncol=1)
            fig.suptitle("Контакт берегов · проверка шага и жёсткости штрафного контакта", fontsize=14, y=.98)
            reference = cases["step_10yr"]
            face_count = reference["material_face_count"]
            source_age = reference["final"]["source_time_myr"]
            fig.text(.5, .025, f"Одна исходная сетка: {face_count} ячеек, возраст {source_age:g} млн лет; тепло, орбита и реология заморожены.\n"
                     "Невязка нормирована на |убыль упругой энергии оболочки| + |внешняя работа|; это не нагрев.",
                     ha="center", fontsize=9)
            fig.subplots_adjust(left=.085, right=.975, bottom=.12, top=.78, hspace=.42, wspace=.28)
            fig.savefig(path, dpi=150)
        finally:
            plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--duration-years", type=float, default=1000.)
    args = parser.parse_args(argv)
    if not math.isfinite(args.duration_years) or args.duration_years <= 0:
        parser.error("Duration must be finite and positive")
    if args.output.exists():
        parser.error("Output must be a new directory")
    source = args.checkpoint.read_bytes()
    parameters, law = ContactParameters(), ContactLawParameters()
    scenarios = (("step_10yr", law, 10.), ("step_5yr", law, 5.),
                 ("normal_penalty_double", replace(law, normal_stiffness_pa_m=2*law.normal_stiffness_pa_m), 10.))
    args.output.mkdir(parents=True)
    cases = {}
    for name, local_law, step in scenarios:
        cases[name] = simulate(source, args.checkpoint, parameters, local_law, step,
                               args.duration_years, name, args.output)
        (args.output/f"{name}.json").write_text(json.dumps(cases[name], indent=2, allow_nan=False)+"\n", encoding="utf-8")
    report = {
        "provenance": {"source_path": str(args.checkpoint.resolve()),
                       "source_sha256": hashlib.sha256(source).hexdigest(),
                       "contact_module_sha256": hashlib.sha256((ROOT/"tectonics/genesis_contact.py").read_bytes()).hexdigest(),
                       "contact_law_sha256": hashlib.sha256((ROOT/"tectonics/genesis_contact_law.py").read_bytes()).hexdigest()},
        "energy_accounting": "Friction, viscosity, Mode-I fracture and basal drag are separate from thermal energy. The mechanical remainder includes numerical/time-discretization and softening-path effects, not a second physical heat source.",
        "cases": cases, "comparison": compare(cases),
    }
    (args.output/"validation.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    plot(cases, args.output/"comparison.png")
    print("CONTACT_VALIDATION_COMPLETE", args.output.resolve(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
