"""Compare saved moving-shell trajectories without running a new simulation.

Common-time values are piecewise-linear interpolations of saved observations,
never extrapolations or new mechanical states. Different event endpoints are
reported separately. Two timesteps show sensitivity, not an established order
of convergence or a resolution-independent prediction of planetary fracture.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXED = ROOT / "results/genesis_runs/path_precursor_current250_20260926/validation.json"
FIELDS = ("strength_ratio", "accumulated_hencky_strain", "radius_m", "radius_fraction",
          "max_material_travel_m", "elastic_strain", "external_work_j", "drag_work_j",
          "bulk_work_j", "bulk_loading_correction_j", "mechanical_remainder_j")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _key(value):
    return str(value).replace("\\", "/")


def report_path(path):
    path = Path(path).resolve()
    return path / "validation.json" if path.is_dir() else path


def audit_report(path, report):
    """Verify executed code using an explicit archived-source map when present."""
    archived = {_key(k): v for k, v in report.get("execution_source_copies", {}).items()}
    rows = []
    for group, base in (("source_sha256", ROOT), ("code_sha256", ROOT),
                        ("artifact_sha256", path.parent),
                        ("execution_source_sha256", path.parent)):
        mapping = report.get(group, {})
        if group != "execution_source_sha256" and not mapping:
            raise ValueError(f"Missing {group} provenance in {path}")
        for name, expected in mapping.items():
            archive = archived.get(_key(name)) if group == "code_sha256" else None
            resolved = (path.parent / archive if archive else base / Path(name)).resolve()
            actual = digest(resolved) if resolved.is_file() else None
            rows.append({"group": group, "declared_path": name,
                "verified_path": str(resolved),
                "source_kind": "archived_execution_source" if archive else "declared_file",
                "expected_sha256": expected, "actual_sha256": actual,
                "matches": actual == expected})
    # An unused archive mapping would hide a misspelled code provenance key.
    declared_code = {_key(k) for k in report.get("code_sha256", {})}
    if not set(archived).issubset(declared_code):
        raise ValueError(f"Archived execution source has no declared code hash in {path}")
    return {"report_path": str(path), "report_sha256": digest(path),
            "all_matches": all(row["matches"] for row in rows), "files": rows}


def load_trajectory(path):
    path = report_path(path)
    report = json.loads(path.read_text(encoding="utf-8"))
    audit = audit_report(path, report)
    if not audit["all_matches"]:
        bad = [row["verified_path"] for row in audit["files"] if not row["matches"]]
        raise ValueError(f"Trajectory provenance failed: {bad}")
    if not report.get("checks") or not all(v is True for v in report["checks"].values()):
        raise ValueError(f"Trajectory does not have all validation checks passing: {path}")
    event = report.get("event")
    if not event or not event.get("activation_performed"):
        raise ValueError(f"Comparison requires a finished local-birth event: {path}")
    if report.get("stopped_reason") is not None:
        raise ValueError(f"Trajectory stopped on a numerical guard: {path}")
    rows = report.get("rows", [])
    time = np.asarray([row["elapsed_years"] for row in rows], dtype=float)
    if len(time) < 2 or not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
        raise ValueError(f"Trajectory times must be finite and strictly increasing: {path}")
    for field in FIELDS:
        if not np.isfinite([row[field] for row in rows]).all():
            raise ValueError(f"Non-finite trajectory values in {field}: {path}")
    if time[-1] != event["lower_elapsed_years"]:
        raise ValueError(f"Last observation must be the accepted lower event state: {path}")
    return report, audit


def interpolate(report, times, field):
    source_times = np.asarray([r["elapsed_years"] for r in report["rows"]])
    times = np.asarray(times, dtype=float)
    if np.any(times < source_times[0]) or np.any(times > source_times[-1]):
        raise ValueError("Common-time comparison never extrapolates")
    return np.interp(times, source_times, [r[field] for r in report["rows"]])


def peak(report, field):
    values = np.asarray([abs(r[field]) for r in report["rows"]])
    index = int(np.argmax(values))
    return {"absolute_value": float(values[index]),
            "elapsed_years": report["rows"][index]["elapsed_years"]}


def trajectory_summary(report):
    last = report["rows"][-1]
    initial_radius = report["rows"][0]["radius_m"] / report["rows"][0]["radius_fraction"]
    residual_fields = ("force_raw_relative_residual", "force_residual", "force_residual_norm_n",
        "parent_force_projection_error", "parent_elastic_energy_projection_error",
        "column_heat_relative_residual")
    return {"max_step_years": report["max_step_years"],
        "event_elapsed_years": last["elapsed_years"], "event_age_myr": last["age_myr"],
        "event_bracket_width_years": report["event"]["width_years"],
        "event_strength_ratio": last["strength_ratio"],
        "initial_radius_m": initial_radius, "event_radius_m": last["radius_m"],
        "event_radius_contraction_m": initial_radius - last["radius_m"],
        "event_accumulated_hencky_strain": last["accumulated_hencky_strain"],
        "event_depth_m_quantiles": last["depth_m_quantiles"],
        "event_force_replacement_relative_error": report["event"]["force_replacement_relative_error"],
        "event_interface_energy_change_j": report["event"]["interface_energy_change_j"],
        "maxima_over_saved_observations": {name: peak(report, name) for name in residual_fields},
        "layer_mass_unchanged": report["layer_mass_unchanged"],
        "restart_exact": report["restart_exact"],
        "birth_restart_exact": report["event"]["birth_restart_exact"],
        "final_work_ledgers_j": {key: last[key] for key in FIELDS if key.endswith("_j")},
        "accepted_steps": last["accepted_steps"], "rejected_steps": last["rejected_steps"],
        "wall_seconds": report["wall_seconds"]}


def comparison(coarse, fine):
    if coarse["max_step_years"] <= fine["max_step_years"]:
        raise ValueError("Coarse maximum step must exceed the fine maximum step")
    if coarse["source_cells"] != fine["source_cells"] or not np.isclose(
            coarse["gate_age_myr"], fine["gate_age_myr"], rtol=0., atol=1e-12):
        raise ValueError("Comparison requires the same resolution and thermal gate")
    if coarse["source_sha256"] != fine["source_sha256"]:
        raise ValueError("Comparison requires identical declared source files and hashes")
    ca, fa = coarse["rows"], fine["rows"]
    first = max(ca[0]["elapsed_years"], fa[0]["elapsed_years"])
    last = min(ca[-1]["elapsed_years"], fa[-1]["elapsed_years"])
    if last <= first:
        raise ValueError("Trajectories have no common physical-time interval")
    times = np.unique(np.r_[[r["elapsed_years"] for r in ca],
                           [r["elapsed_years"] for r in fa]])
    times = times[(times >= first) & (times <= last)]
    sampled_times = sorted(set([first, last] + [t for t in (10000., 50000., 95000., 100000.)
                                               if first <= t <= last]))
    common_rows, metrics = [], {}
    for t in sampled_times:
        a = {field: float(interpolate(coarse, t, field)) for field in FIELDS}
        b = {field: float(interpolate(fine, t, field)) for field in FIELDS}
        common_rows.append({"elapsed_years": t, "coarse": a, "fine": b,
                           "fine_minus_coarse": {field: b[field]-a[field] for field in FIELDS}})
    for field in FIELDS:
        a, b = interpolate(coarse, times, field), interpolate(fine, times, field)
        difference = b-a
        index = int(np.argmax(np.abs(difference)))
        metrics[field] = {"max_absolute_difference": float(abs(difference[index])),
            "elapsed_years_at_maximum": float(times[index]),
            "max_difference_divided_by_max_fine_magnitude": float(
                np.max(np.abs(difference)) / max(float(np.max(np.abs(b))), 1e-300)),
            "difference_at_common_end": float(difference[-1])}
    delta = fa[-1]["elapsed_years"]-ca[-1]["elapsed_years"]
    return {"scope": __doc__, "source_cells": coarse["source_cells"],
        "gate_age_myr": coarse["gate_age_myr"],
        "coarse": trajectory_summary(coarse), "fine": trajectory_summary(fine),
        "event_comparison": {"fine_minus_coarse_years": delta,
            "absolute_difference_years": abs(delta),
            "relative_difference_to_fine_elapsed_time": abs(delta)/fa[-1]["elapsed_years"],
            "fine_minus_coarse_age_myr": fa[-1]["age_myr"]-ca[-1]["age_myr"],
            "radius_difference_at_respective_event_times_m": fa[-1]["radius_m"]-ca[-1]["radius_m"],
            "radius_difference_interpretation": "Different event times; this is not a same-time spatial error."},
        "common_time_comparison": {"method": "Piecewise-linear interpolation between saved observations; no extrapolation or new simulation.",
            "first_elapsed_years": first, "last_elapsed_years": last,
            "union_sample_count": len(times), "metrics": metrics, "selected_observations": common_rows},
        "force_residual_interpretation": "Raw relative residual is norm(R) / max(norm(internal), norm(external), 1 N). The solver's normalized residual additionally uses a finite-geometry roundoff floor. Both are reported; they are different criteria.",
        "mechanical_work_interpretation": "Mechanical remainder is a separately retained algorithmic ledger, not heat or fracture dissipation. It includes endpoint-versus-trapezoid work quadrature and geometric work-conjugacy differences. A small force residual and closed column heat ledger do not establish a closed global thermo-mechanical energy budget.",
        "limitations": ["The event is a sampled crossing on one prescribed transported support, not the globally earliest planetary fracture.",
            "The run stops after saving local birth; active contact evolution on moving geometry is not part of this comparison.",
            "Two maximum timesteps are a sensitivity check, not a demonstrated convergence order.",
            "Crossing the former accumulated 0.005 strain guard does not remove individual-increment and mesh-validity guards."]}


def plot(coarse, fine, fixed, result, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.4), constrained_layout=True)
    for report, color in ((coarse, "tab:blue"), (fine, "tab:orange")):
        rows = report["rows"]
        time = np.asarray([r["elapsed_years"] for r in rows])/1000.
        label = f"Шаг до {report['max_step_years']:g} лет"
        axes[0, 0].plot(time, [r["strength_ratio"] for r in rows], color=color, label=label)
        axes[0, 0].plot(time[-1], rows[-1]["strength_ratio"], "o", color=color, markersize=5)
        axes[0, 1].plot(time, [r["accumulated_hencky_strain"] for r in rows], color=color, label=label)
        initial_radius = rows[0]["radius_m"]/rows[0]["radius_fraction"]
        axes[1, 0].plot(time, [(initial_radius-r["radius_m"])/1000. for r in rows], color=color, label=label)
        axes[1, 1].semilogy(time, [r["force_raw_relative_residual"] for r in rows],
            color=color, label=f"{report['max_step_years']:g} лет: по силам")
        axes[1, 1].semilogy(time, [r["force_residual"] for r in rows],
            color=color, linestyle="--", label=f"{report['max_step_years']:g} лет: с порогом округления")
    if fixed is not None:
        probes = fixed["probes"]
        axes[0, 0].plot(np.asarray([r["elapsed_years"] for r in probes])/1000.,
            [r["strength_ratio"] for r in probes], color=".55", linewidth=1.4,
            label="Прежняя фиксированная геометрия")
        old_stop = fixed["last_accepted_elapsed_years"]/1000.
    else:
        old_stop = 95.
    axes[0, 0].axvline(old_stop, color=".5", linestyle=":", label="Прежняя остановка ≈95 тыс. лет")
    axes[0, 0].axhline(1., color="black", linewidth=1, linestyle="--", label="Порог разрушения")
    delta = result["event_comparison"]["absolute_difference_years"]
    axes[0, 0].text(.03, .96, f"Разница времени события: {delta:.2f} года",
        transform=axes[0, 0].transAxes, va="top", fontsize=9)
    axes[0, 0].set(title="Достижение прочности на заданной линии", ylabel="Нагрузка / прочность", ylim=(0., 1.10))
    axes[0, 1].axhline(.005, color="black", linewidth=1, linestyle="--", label="Прежний суммарный предел 0,005")
    axes[0, 1].set(title="Накопленная деформация при обновлении геометрии",
        ylabel="Максимальная деформация Хенки")
    axes[0, 1].text(.03, .96, "Ограничение отдельного шага сохранено",
        transform=axes[0, 1].transAxes, va="top", fontsize=9)
    axes[1, 0].set(title="Сжатие оболочки по мере остывания", ylabel="Уменьшение радиуса, км")
    axes[1, 1].set(title="Невязка равновесия: два способа нормировки", ylabel="Относительная невязка")
    for axis in axes.ravel():
        axis.set_xlabel("Тысячи лет после образования несущей оболочки")
        axis.grid(alpha=.25)
        axis.legend(fontsize=7.5, loc="best")
    axes[0, 0].legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(0., .87))
    axes[0, 1].legend(fontsize=7.5, loc="lower right")
    fig.suptitle(f"Сетка {coarse['source_cells']} ячеек: остывание до рождения локального контакта\n"
        "Заданная линия разлома; каждый расчёт заканчивается в своём времени события", fontsize=13)
    fig.savefig(output/"comparison.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coarse", type=Path, required=True)
    parser.add_argument("--fine", type=Path, required=True)
    parser.add_argument("--fixed", type=Path, default=DEFAULT_FIXED,
                        help="Optional historical fixed-geometry report; an absent file omits that curve")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Comparison output must be new or empty")
    coarse, caudit = load_trajectory(args.coarse)
    fine, faudit = load_trajectory(args.fine)
    result = comparison(coarse, fine)
    fixed = None
    fixed_path = report_path(args.fixed)
    if fixed_path.is_file():
        fixed = json.loads(fixed_path.read_text(encoding="utf-8"))
        if not np.isclose(fixed["thermal_gate_age_myr"], coarse["gate_age_myr"], rtol=0., atol=1e-12):
            raise ValueError("Historical fixed-geometry reference has a different thermal gate")
        result["historical_fixed_reference"] = {"path": str(fixed_path), "sha256": digest(fixed_path),
            "last_accepted_elapsed_years": fixed["last_accepted_elapsed_years"],
            "last_strength_ratio": fixed["probes"][-1]["strength_ratio"],
            "stopped_reason": fixed["stopped_reason"],
            "scope": "Historical curve for context; not included in moving-kernel timestep convergence or current-code audit."}
    result["input_provenance"] = {"coarse": caudit, "fine": faudit}
    result["comparison_code_sha256"] = {str(Path(__file__).resolve()): digest(__file__)}
    output.mkdir(parents=True, exist_ok=True)
    plot(coarse, fine, fixed, result, output)
    result["artifact_sha256"] = {"comparison.png": digest(output/"comparison.png")}
    result["checks"] = {"all_input_provenance_verified": caudit["all_matches"] and faudit["all_matches"],
        "both_trajectories_passed_validation": True,
        "both_local_births_saved": True,
        "common_time_comparison_has_no_extrapolation": True,
        "input_reports_unchanged": digest(caudit["report_path"]) == caudit["report_sha256"]
            and digest(faudit["report_path"]) == faudit["report_sha256"]}
    if fixed is not None:
        result["checks"]["historical_reference_unchanged"] = digest(fixed_path) == result["historical_fixed_reference"]["sha256"]
    (output/"comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": str(output), "event_comparison": result["event_comparison"],
        "common_end_radius_difference_m": result["common_time_comparison"]["metrics"]["radius_m"]["difference_at_common_end"],
        "checks": result["checks"]}, indent=2))
    if not all(result["checks"].values()):
        raise SystemExit("Moving comparison audit failed")


if __name__ == "__main__":
    main()
