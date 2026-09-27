"""Validate seeded energetic crack growth on a displacement-controlled DCB.

The spherical support stores material coordinates only. The mechanical oracle
is a straight, small-deflection Euler--Bernoulli double-cantilever beam, not a
curved planetary shell. A finite initial notch is prescribed. This experiment
neither chooses a planetary nucleation site nor inserts contacts into its mesh.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tectonics.genesis_crack_energy import DCBOracle
from tectonics.genesis_crack_front import CrackFrontModel, FrontParameters
from tectonics.genesis_crack_path import CrackInterval, ReferenceCrackPath


# Independent benchmark specification. Do not derive the analytic reference
# from controller events, its energy residual, or a fitted numerical response.
YOUNG_PA = 70e9
ARM_THICKNESS_M = .002
WIDTH_M = .025
TOUGHNESS_J_M2 = 5.
SEED_M = .03
SUPPORT_M = .15
TARGET_M = .09
REFINEMENTS = ((40, .002), (80, .001), (160, .0005), (320, .00025), (640, .000125))


def _opening_for_length(length_m):
    return float(np.sqrt(16*TOUGHNESS_J_M2*length_m**4/(3*YOUNG_PA*ARM_THICKNESS_M**3)))


def _equilibrium_length(opening_m):
    return np.maximum(SEED_M,
        (3*YOUNG_PA*ARM_THICKNESS_M**3*np.asarray(opening_m)**2/(16*TOUGHNESS_J_M2))**.25)


def _model(extension_m, *, support_m=SUPPORT_M):
    # Radius = one metre. This arc is only a material-coordinate support;
    # no spherical curvature is inserted into the straight-beam oracle.
    angles = np.linspace(0., support_m, 7)
    points = np.column_stack((np.cos(angles), np.sin(angles), np.zeros_like(angles)))
    path = ReferenceCrackPath(points, radius_km=.001)
    oracle = DCBOracle(young_pa=YOUNG_PA, arm_height_m=ARM_THICKNESS_M,
                       width_m=WIDTH_M, fracture_energy_j_m2=TOUGHNESS_J_M2)
    return CrackFrontModel(path, oracle, CrackInterval(0., SEED_M),
                           FrontParameters(extension_m=extension_m))


def _snapshot(model, state):
    return {
        "load_step": int(state.load_step), "opening_m": float(state.opening_m),
        "crack_length_m": float(state.interval.length_m),
        "stored_energy_j": float(state.stored_energy_j),
        "external_work_j": float(state.external_work_j),
        "fracture_work_j": float(state.fracture_work_j),
        "unresolved_release_j": float(state.unresolved_release_j),
        "energy_residual_j": float(model.energy_residual_j(state)),
        "event_count": len(state.events), "stop_reason": state.stop_reason,
    }


def _run(model, openings):
    state = model.initial()
    history = [_snapshot(model, state)]
    for opening in openings:
        previous = state
        state = model.advance(state, float(opening))
        if state.interval.left_m != previous.interval.left_m or state.interval.right_m < previous.interval.right_m:
            raise AssertionError("The seeded right-front experiment must never heal or move its left tip")
        if (state.fracture_work_j < previous.fracture_work_j
                or state.unresolved_release_j < previous.unresolved_release_j):
            raise AssertionError("Irreversible energy ledgers must never decrease")
        tolerance = 1e-10*max(abs(state.external_work_j), 1.)
        if abs(model.energy_residual_j(state)) > tolerance:
            raise AssertionError("Front energy ledger failed to close")
        history.append(_snapshot(model, state))
    return state, history


def _joint_refinement():
    final_opening = _opening_for_length(TARGET_M)
    initial_onset_energy = TOUGHNESS_J_M2*WIDTH_M*SEED_M/3
    reference = {
        "target_crack_length_m": TARGET_M, "final_opening_m": final_opening,
        "critical_seed_opening_m": _opening_for_length(SEED_M),
        "seed_energy_at_onset_j": initial_onset_energy,
        "stored_energy_j": TOUGHNESS_J_M2*WIDTH_M*TARGET_M/3,
        "fracture_work_j": TOUGHNESS_J_M2*WIDTH_M*(TARGET_M-SEED_M),
        "external_work_j": initial_onset_energy+(4/3)*TOUGHNESS_J_M2*WIDTH_M*(TARGET_M-SEED_M),
        "unresolved_release_j": 0.,
    }
    cases = []
    for steps, extension in REFINEMENTS:
        model = _model(extension)
        state, history = _run(model, np.linspace(0., final_opening, steps+1)[1:])
        analytic_lengths = _equilibrium_length([row["opening_m"] for row in history])
        length_errors = np.asarray([row["crack_length_m"] for row in history])-analytic_lengths
        final = _snapshot(model, state)
        final.update(name=f"ramp_{steps}_da_{extension:g}", loading_steps=steps,
                     extension_m=extension, history=history,
                     crack_length_error_m=abs(state.interval.length_m-TARGET_M),
                     loading_path_max_length_error_m=float(np.abs(length_errors).max()),
                     loading_path_rms_length_error_m=float(np.sqrt(np.mean(length_errors**2))),
                     external_work_error_j=abs(state.external_work_j-reference["external_work_j"]))
        if final["crack_length_error_m"] > extension*(1+1e-9):
            raise AssertionError("Final front lies outside its finite-extension resolution of the analytic solution")
        cases.append(final)
    if (cases[-1]["external_work_error_j"] >= cases[0]["external_work_error_j"]
            or cases[-1]["unresolved_release_j"] >= cases[0]["unresolved_release_j"]
            or cases[-1]["loading_path_max_length_error_m"] >= cases[0]["loading_path_max_length_error_m"]):
        raise AssertionError("Joint loading/extension refinement did not reduce energetic overdrive")
    return reference, cases


def _single_jump(reference):
    cases = []
    for _, extension in REFINEMENTS:
        model = _model(extension)
        state, history = _run(model, [reference["final_opening_m"]])
        final = _snapshot(model, state)
        final.update(extension_m=extension, history=history)
        cases.append(final)
    # At fixed opening U = E b h^3 delta^2 / (16 a^3). One instantaneous
    # load jump stores 0.10125 J in the 30 mm seed. Even infinitely fine growth
    # releases 0.09 J beyond the fracture work. That is not numerical closure.
    loaded_seed_energy = (YOUNG_PA*WIDTH_M*ARM_THICKNESS_M**3
                          *reference["final_opening_m"]**2/(16*SEED_M**3))
    exact_excess = loaded_seed_energy-reference["stored_energy_j"]-reference["fracture_work_j"]
    if not cases[-1]["unresolved_release_j"] > .99*exact_excess:
        raise AssertionError("One large load jump incorrectly lost its finite unresolved release")
    return {"continuous_extension_limit_unresolved_release_j": exact_excess,
            "load_jump_work_j": loaded_seed_energy, "cases": cases}


def _cycles_and_controls(reference, output):
    model = _model(.00025)
    seed_critical = reference["critical_seed_opening_m"]
    zero, zero_history = _run(model, [0.])
    subcritical, subcritical_history = _run(model, [.9*seed_critical])
    if zero.events or subcritical.events or zero.interval.length_m != SEED_M or subcritical.interval.length_m != SEED_M:
        raise AssertionError("Zero and subcritical opening must not create crack growth")

    peak = reference["final_opening_m"]
    cycle_openings = np.r_[np.linspace(0., peak, 161)[1:],
                          np.linspace(peak, 0., 61)[1:],
                          np.linspace(0., .95*peak, 61)[1:],
                          np.linspace(.95*peak, _opening_for_length(.11), 81)[1:]]
    state, history = _run(model, cycle_openings)
    peak_row, unload_row, reload_row = history[160], history[220], history[280]
    for row in (unload_row, reload_row):
        for key in ("crack_length_m", "fracture_work_j", "unresolved_release_j", "event_count"):
            if row[key] != peak_row[key]:
                raise AssertionError("Unloading and reloading below the previous peak changed irreversible history")
    if history[-1]["crack_length_m"] <= peak_row["crack_length_m"]:
        raise AssertionError("A higher loading peak should restart growth")

    # Resume halfway through the unloading branch, then compare the full
    # immutable state and event ledger, not just its front coordinate.
    checkpoint_step = 190
    prior, _ = _run(model, cycle_openings[:checkpoint_step])
    checkpoint = output/"restart_front.json"
    model.save_checkpoint(checkpoint, prior)
    restored_model, restored = CrackFrontModel.load_checkpoint(checkpoint)
    if json.dumps(asdict(restored), sort_keys=True, allow_nan=False) != json.dumps(asdict(prior), sort_keys=True, allow_nan=False):
        raise AssertionError("Checkpoint roundtrip changed the prescribed front state")
    for opening in cycle_openings[checkpoint_step:]:
        restored = restored_model.advance(restored, float(opening))
    if json.dumps(asdict(restored), sort_keys=True, allow_nan=False) != json.dumps(asdict(state), sort_keys=True, allow_nan=False):
        raise AssertionError("Restart changed the uninterrupted front history")

    exhausted_model = _model(.002, support_m=.055)
    exhausted, exhausted_history = _run(exhausted_model, [peak])
    if not np.isclose(exhausted.interval.right_m, exhausted_model.path.length_m, atol=1e-13, rtol=0):
        raise AssertionError("A sufficiently loaded crack must reach the end of its supplied support")
    if exhausted.stop_reason != "support_exhausted":
        raise AssertionError("Exhausting the supplied support must retain an explicit stop reason")
    return {
        "zero": zero_history[-1], "subcritical": subcritical_history[-1],
        "cycle": {"history": history, "peak_step": 160, "unloaded_step": 220,
                  "below_peak_reloaded_step": 280, "higher_peak_step": len(cycle_openings)},
        "restart": {"checkpoint": checkpoint.name, "checkpoint_step": checkpoint_step,
                    "roundtrip_state_exact": True, "continuation_state_and_events_exact": True,
                    "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()},
        "support_exhaustion": {**exhausted_history[-1], "support_length_m": .055},
        "checks": {"zero_load_no_growth": True, "subcritical_no_growth": True,
                   "unloading_no_healing": True, "below_peak_reloading_no_growth": True,
                   "higher_peak_restarts_growth": True, "support_limit_explicit": True},
    }


def _draw_report(output, report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter, FuncFormatter

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    ramp = report["joint_refinement"]
    reference = report["analytic_reference"]
    ax = axes[0, 0]
    opening = np.linspace(0., reference["final_opening_m"], 400)
    ax.plot(opening*1000, _equilibrium_length(opening)*1000,
            color="black", linestyle="--", label="Аналитическое равновесие")
    for row, color in ((ramp[0], "#e18422"), (ramp[-1], "#197aa3")):
        ax.step([item["opening_m"]*1000 for item in row["history"]],
                [item["crack_length_m"]*1000 for item in row["history"]], where="post", color=color,
                label=f"{row['loading_steps']} нагрузок; Δa = {row['extension_m']*1000:g} мм")
    ax.set(xlabel="Заданное раскрытие δ, мм", ylabel="Длина трещины a, мм",
           title="Рост заданного начального надреза")
    ax.legend(fontsize=9)

    ax = axes[0, 1]
    counts = [row["loading_steps"] for row in ramp]
    exact_work = reference["external_work_j"]
    ax.loglog(counts, [row["external_work_error_j"]/exact_work*100 for row in ramp],
              "o-", color="#197aa3", label="Ошибка внешней работы")
    ax.loglog(counts, [row["unresolved_release_j"]/exact_work*100 for row in ramp],
              "s-", color="#af524a", label="Неразрешённое высвобождение")
    ax.set(xlabel="Число приращений нагрузки (Δa также уменьшается)",
           ylabel="Доля аналитической внешней работы, %",
           title="Совместное уточнение нагрузки и продвижения")
    ax.legend(fontsize=9)
    ax.set_xticks(counts, labels=[str(value) for value in counts])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda value, _: f"{value:g}"))

    ax = axes[1, 0]
    jumps = report["single_load_jump"]
    ax.semilogx([row["extension_m"]*1000 for row in jumps["cases"]],
                [row["unresolved_release_j"] for row in jumps["cases"]], "o-", color="#af524a")
    ax.axhline(jumps["continuous_extension_limit_unresolved_release_j"],
               color="black", linestyle="--", label="Предел при единственном скачке нагрузки")
    ax.set(xlabel="Шаг продвижения Δa, мм", ylabel="Неразрешённое высвобождение, Дж",
           title="Мелкий шаг трещины не исправляет крупный скачок нагрузки")
    ax.set_ylim(0., .1)
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    steps_mm = [row["extension_m"]*1000 for row in jumps["cases"]]
    ax.set_xticks(steps_mm, labels=[f"{value:g}" for value in steps_mm])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    history = report["controls"]["cycle"]["history"]
    line1, = ax.plot([row["load_step"] for row in history],
                     [row["crack_length_m"]*1000 for row in history], color="#197aa3", label="Длина трещины")
    second = ax.twinx()
    line2, = second.plot([row["load_step"] for row in history],
                         [row["opening_m"]*1000 for row in history], color="#e18422", linestyle="--", label="Раскрытие")
    ax.set(xlabel="Номер приращения нагрузки", ylabel="Длина трещины a, мм",
           title="Разгрузка, повторная нагрузка и новый максимум")
    second.set_ylabel("Заданное раскрытие δ, мм")
    ax.legend(handles=[line1, line2], fontsize=9, loc="upper left")
    for ax in axes.ravel():
        ax.grid(alpha=.22)
    fig.suptitle("Энергетическое продвижение фронта: контрольный балочный образец\n"
                 "Начальный надрез и путь заданы; расчёт не моделирует планетное зарождение", fontsize=15)
    fig.savefig(output/"validation.png", dpi=160)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("Validation output must be a new or empty directory")
    output.mkdir(parents=True, exist_ok=True)

    reference, ramp = _joint_refinement()
    jumps = _single_jump(reference)
    controls = _cycles_and_controls(reference, output)
    source_names = ("tectonics/genesis_crack_energy.py", "tectonics/genesis_crack_path.py",
                    "tectonics/genesis_crack_front.py", "analysis/genesis_crack_front_validation.py")
    report = {
        "format": "genesis-crack-front-validation-0.1",
        "interpretation": "Prescribed straight-beam elastic DCB benchmark; no planetary G, time, nucleation or contacts",
        "mechanical_specification": {"young_modulus_pa": YOUNG_PA, "arm_thickness_m": ARM_THICKNESS_M,
                                     "width_m": WIDTH_M, "fracture_energy_j_m2": TOUGHNESS_J_M2,
                                     "seed_length_m": SEED_M, "reference_support_length_m": SUPPORT_M},
        "source_sha256": {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in source_names},
        "analytic_reference": reference, "joint_refinement": ramp,
        "single_load_jump": jumps, "controls": controls,
        "limitations": [
            "The finite seed notch and candidate support are prescribed, not selected by a physical nucleation law.",
            "Beam dimensions and 5 J/m2 toughness are explicit verification controls, not calibrated planetary properties.",
            "The straight Euler--Bernoulli beam oracle ignores support-sphere curvature; the arc supplies coordinates only.",
            "Loading increments and extension distances are not elapsed time or physical crack speed.",
            "Unresolved excess release is retained separately, not relabeled fracture work, heat, or kinetic energy.",
            "A small algebraic energy residual does not establish convergence of loading or fracture increments.",
            "The 90 mm final length aligns with every tested extension grid; length convergence is assessed along the full loading path.",
            "No ridge-derived planetary energy release, embedded-contact insertion, branching, or production checkpoint changes.",
        ],
    }
    (output/"validation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    _draw_report(output, report)
    print(json.dumps({"output": str(output), "joint_refinement_cases": len(ramp),
                      "single_jump_cases": len(jumps["cases"]), "controls": controls["checks"],
                      "restart_exact": controls["restart"]["continuation_state_and_events_exact"],
                      "finest_work_error_j": ramp[-1]["external_work_error_j"],
                      "finest_unresolved_release_j": ramp[-1]["unresolved_release_j"]}, ensure_ascii=False), flush=True)
    return report


if __name__ == "__main__":
    main()
