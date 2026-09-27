"""Audit and compare saved moving-contact runs without running mechanics.

Comparison uses the shared physical-time interval after the same saved birth.
Piecewise-linear interpolation of recorded diagnostics is not a new solution;
there is no extrapolation and no convergence-order claim from two timesteps.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SECONDS_PER_YEAR = 365.25*86400.
WORK_FIELDS = (
    "external_work_j", "drag_work_j", "bulk_work_j",
    "bulk_loading_correction_j", "mechanical_remainder_j",
    "contact_geometric_work_j", "cohort_parameter_energy_j",
)
FIELDS = (
    "radius_m", "max_opening_m", "max_slip_m", "penetration_m",
    "initial_cohort_damage_max", "initial_max_opening_m",
    "new_cohort_area_m2", "initial_cohort_area_m2", "solid_volume_m3",
    "max_solid_volume_growth_fraction", "force_relative_residual",
    "force_raw_relative_residual", "force_residual_norm_n",
    "contact_potential_j", "column_heat_relative_residual",
    "fracture_work_j", "friction_work_j", "viscous_work_j",
    "shear_remainder_j", "max_held_strength_ratio",
    "held_traction_recovery_residual",
) + tuple("post_birth_"+name for name in WORK_FIELDS)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def report_path(path):
    path = Path(path).resolve()
    return path/"validation.json" if path.is_dir() else path


def audit_report(path, report):
    """Check every declared source, code and artifact hash, including archives."""
    normalized = lambda key: str(key).replace("\\", "/")
    archived = {normalized(k): v for k, v in report.get("execution_source_copies", {}).items()}
    files = []
    for group, base in (("source_sha256", ROOT), ("code_sha256", ROOT),
                        ("artifact_sha256", path.parent),
                        ("execution_source_sha256", path.parent)):
        mapping = report.get(group, {})
        if group != "execution_source_sha256" and not mapping:
            raise ValueError(f"Missing {group} provenance: {path}")
        for name, expected in mapping.items():
            archived_path = archived.get(normalized(name)) if group == "code_sha256" else None
            resolved = (path.parent/archived_path if archived_path else base/Path(name)).resolve()
            actual = digest(resolved) if resolved.is_file() else None
            files.append({"group": group, "declared_path": name,
                "verified_path": str(resolved), "expected_sha256": expected,
                "actual_sha256": actual, "matches": actual == expected,
                "archived_execution_source": archived_path is not None})
    if not set(archived).issubset({normalized(k) for k in report.get("code_sha256", {})}):
        raise ValueError("Archived source map has an undeclared code entry")
    return {"report_path": str(path), "report_sha256": digest(path),
            "all_matches": all(item["matches"] for item in files), "files": files}


def _time(report):
    return np.asarray([r["years_after_birth"] for r in report["rows"]], dtype=float)


def load_trajectory(path):
    path = report_path(path)
    report = json.loads(path.read_text(encoding="utf-8"))
    audit = audit_report(path, report)
    if not audit["all_matches"]:
        failed = [row["verified_path"] for row in audit["files"] if not row["matches"]]
        raise ValueError(f"Input provenance failed: {failed}")
    if not report.get("checks") or not all(v is True for v in report["checks"].values()):
        raise ValueError(f"Not every trajectory validation check passed: {path}")
    if report.get("stopped_reason") is not None:
        raise ValueError(f"Trajectory stopped on a numerical guard: {path}")
    rows = report.get("rows", [])
    time = _time(report)
    if len(time) < 2 or not np.isfinite(time).all() or time[0] != 0 or np.any(np.diff(time) <= 0):
        raise ValueError(f"Saved times must start at birth and increase strictly: {path}")
    for field in FIELDS+WORK_FIELDS+("age_myr", "elapsed_years"):
        if not np.isfinite([row[field] for row in rows]).all():
            raise ValueError(f"Invalid saved diagnostic {field}: {path}")
    if not np.allclose([r["elapsed_years"]-report["birth_elapsed_years"] for r in rows],
                       time, rtol=0, atol=1e-9):
        raise ValueError("Saved physical clocks disagree")
    area = np.asarray([row["reference_to_current_area_ratio_range"] for row in rows])
    if (area.shape != (len(rows), 2) or not np.isfinite(area).all()
            or np.any(area <= 0) or np.any(area[:, 1] < area[:, 0])):
        raise ValueError("Invalid reference/current interface-area diagnostic")
    event = report.get("next_held_event")
    if event is not None:
        if (event["lower_years_after_birth"] != time[-1]
                or event["upper_years_after_birth"] < time[-1]
                or not event["lower_ratio"] < 1 <= event["upper_ratio"]):
            raise ValueError("Next held strength event is not an admissible bracket")
    elif time[-1] < report["requested_duration_years"]:
        raise ValueError("Trajectory reached neither requested duration nor a next strength event")
    return report, audit


def interpolate(report, times, field):
    source_time = _time(report)
    times = np.asarray(times, dtype=float)
    if np.any(times < source_time[0]) or np.any(times > source_time[-1]):
        raise ValueError("Common-time comparison never extrapolates")
    return np.interp(times, source_time, [r[field] for r in report["rows"]])


def final_material(path, report):
    """Audit cohort bookkeeping directly from the already-hashed checkpoint."""
    with np.load(path.parent/"final_mechanics.npz", allow_pickle=False) as data:
        area = data["added_area_ref_m2"]
        bonded = data["added_bonded"]
        initial_area = data["initial_area_ref_m2"]
        if (area.shape != bonded.shape or bonded.dtype.kind != "b"
                or np.any(area <= 0) or not np.isfinite(area).all()):
            raise ValueError("Invalid added material checkpoint arrays")
        last = report["rows"][-1]
        if (len(area) != last["new_cohort_count"]
                or int(np.count_nonzero(~bonded)) != last["new_unbonded_cohort_count"]
                or float(area.sum()) != last["new_cohort_area_m2"]
                or float(initial_area.sum()) != last["initial_cohort_area_m2"]):
            raise ValueError("Final cohort material disagrees with saved observations")
        return {"cohort_count": len(area), "bonded_cohort_count": int(np.count_nonzero(bonded)),
            "unbonded_cohort_count": int(np.count_nonzero(~bonded)),
            "added_area_ref_m2": float(area.sum()),
            "bonded_added_area_ref_m2": float(area[bonded].sum()),
            "unbonded_added_area_ref_m2": float(area[~bonded].sum()),
            "initial_area_ref_m2": float(initial_area.sum()),
            "cohort_count_interpretation": "Record counts depend on accepted timestep count; areas are the material measure. No count interpolation or count convergence claim is made."}


def peak(report, field):
    values = np.asarray([abs(r[field]) for r in report["rows"]])
    index = int(np.argmax(values))
    return {"absolute_value": float(values[index]),
            "years_after_birth": report["rows"][index]["years_after_birth"]}


def trajectory_summary(report):
    rows = report["rows"]
    time = _time(report)
    geometry = np.asarray([row["post_birth_contact_geometric_work_j"] for row in rows])
    powers = np.diff(geometry)/(np.diff(time)*SECONDS_PER_YEAR)
    peak_index = int(np.argmax(np.abs(powers)))
    area = np.asarray([row["reference_to_current_area_ratio_range"] for row in rows])
    last = rows[-1]
    return {"maximum_step_years": report["maximum_step_years"],
        "requested_duration_years": report["requested_duration_years"],
        "last_years_after_birth": last["years_after_birth"],
        "last_age_myr": last["age_myr"],
        "radius_contraction_since_birth_m": rows[0]["radius_m"]-last["radius_m"],
        "termination": "next_held_strength_event" if report.get("next_held_event") else "requested_duration",
        "next_held_event": report.get("next_held_event"),
        "last_observation": {name: last[name] for name in FIELDS},
        "reference_to_current_area_ratio_range_over_saved_observations": [float(area[:, 0].min()), float(area[:, 1].max())],
        "reference_to_current_area_ratio_range_at_end": last["reference_to_current_area_ratio_range"],
        "peak_absolute_interval_average_contact_geometric_power_w": float(abs(powers[peak_index])),
        "signed_geometric_power_at_peak_w": float(powers[peak_index]),
        "geometric_power_peak_interval_years_after_birth": [float(time[peak_index]), float(time[peak_index+1])],
        "geometric_power_definition": "Difference of cumulative contact_geometric_work_j divided by saved interval duration in seconds; this is an interval average, not instantaneous physical dissipation.",
        "maxima_over_saved_observations": {name: peak(report, name) for name in (
            "force_raw_relative_residual", "force_relative_residual", "force_residual_norm_n",
            "column_heat_relative_residual", "held_traction_recovery_residual")},
        "accepted_steps_after_birth": last["accepted_steps"]-rows[0]["accepted_steps"],
        "rejected_steps_after_birth": last["rejected_steps"]-rows[0]["rejected_steps"],
        "child_column_depth_change_fraction_at_transfer": report["child_column_depth_change_fraction_at_transfer"],
        "wall_seconds": report["wall_seconds"]}


def comparison(coarse, fine):
    if coarse["maximum_step_years"] <= fine["maximum_step_years"]:
        raise ValueError("Coarse maximum timestep must be larger than fine maximum timestep")
    for field in ("parent_cells", "material_cells", "birth_elapsed_years", "birth_age_myr",
                  "source_sha256", "code_sha256"):
        if coarse[field] != fine[field]:
            raise ValueError(f"Comparison requires identical {field}")
    for field in FIELDS:
        if coarse["rows"][0][field] != fine["rows"][0][field]:
            raise ValueError(f"The saved birth states disagree in {field}")
    first = max(_time(coarse)[0], _time(fine)[0])
    last = min(_time(coarse)[-1], _time(fine)[-1])
    if last <= first:
        raise ValueError("No common physical-time interval")
    times = np.unique(np.r_[_time(coarse), _time(fine)])
    times = times[(times >= first) & (times <= last)]
    metrics, common_rows = {}, []
    for field in FIELDS:
        a, b = interpolate(coarse, times, field), interpolate(fine, times, field)
        difference = b-a
        index = int(np.argmax(np.abs(difference)))
        magnitude = float(np.max(np.abs(b)))
        metrics[field] = {"maximum_absolute_difference": float(abs(difference[index])),
            "years_after_birth_at_maximum": float(times[index]),
            "maximum_difference_divided_by_maximum_fine_magnitude": (
                float(np.max(np.abs(difference))/magnitude) if magnitude else
                0. if np.max(np.abs(difference)) == 0 else None),
            "fine_minus_coarse_at_common_end": float(difference[-1])}
    selected = sorted(set([first, last]+[t for t in (10., 50., 100., 250., 500., 1000.) if first <= t <= last]))
    for time in selected:
        a = {field: float(interpolate(coarse, time, field)) for field in FIELDS}
        b = {field: float(interpolate(fine, time, field)) for field in FIELDS}
        common_rows.append({"years_after_birth": time, "coarse": a, "fine": b,
            "fine_minus_coarse": {field: b[field]-a[field] for field in FIELDS}})
    coarse_geometric = interpolate(coarse, times, "post_birth_contact_geometric_work_j")
    fine_geometric = interpolate(fine, times, "post_birth_contact_geometric_work_j")
    durations = np.diff(times)*SECONDS_PER_YEAR
    coarse_power, fine_power = np.diff(coarse_geometric)/durations, np.diff(fine_geometric)/durations
    power_difference = fine_power-coarse_power
    power_index = int(np.argmax(np.abs(power_difference)))
    ce, fe = coarse.get("next_held_event"), fine.get("next_held_event")
    event_comparison = None
    if ce is not None and fe is not None:
        event_comparison = {
            "coarse_held_trace": ce["held_trace"], "fine_held_trace": fe["held_trace"],
            "same_governing_trace": ce["held_trace"] == fe["held_trace"],
            "fine_minus_coarse_lower_bracket_years": fe["lower_years_after_birth"]-ce["lower_years_after_birth"],
            "event_time_interpretation": "Lower bracket samples from separate trajectories; endpoint state differences are not same-time errors."}
    return {"scope": __doc__, "parent_cells": coarse["parent_cells"],
        "material_cells": coarse["material_cells"], "birth_age_myr": coarse["birth_age_myr"],
        "birth_elapsed_years": coarse["birth_elapsed_years"],
        "coarse": trajectory_summary(coarse), "fine": trajectory_summary(fine),
        "next_held_event_comparison": event_comparison,
        "common_time_comparison": {
            "method": "Piecewise-linear interpolation between saved observations, restricted to their shared physical-time range; no extrapolation.",
            "first_years_after_birth": float(first), "last_years_after_birth": float(last),
            "union_sample_count": len(times), "metrics": metrics,
            "contact_geometric_power_comparison": {
                "method": "Slopes of piecewise-linear cumulative geometric work over common saved-time intervals; no instantaneous physical power claim.",
                "maximum_absolute_difference_w": float(abs(power_difference[power_index])),
                "interval_at_maximum_years_after_birth": [float(times[power_index]), float(times[power_index+1])],
                "coarse_interval_average_w": float(coarse_power[power_index]),
                "fine_interval_average_w": float(fine_power[power_index])},
            "selected_observations": common_rows},
        "force_residual_interpretation": "Raw residual is norm(R_free) divided by max(norm(internal_free), norm(contact_free), norm(external_free), 1 N). Solver-normalized residual also includes a roundoff force floor; the two criteria differ.",
        "material_measure_interpretation": "Cohort areas are frozen material measures from contact-birth edge lengths and volume/birth-face-area depth. They are not rescaled by current geometry. Reference/current area ratios are reported separately; initial extrinsic material histories retain their original areas.",
        "work_interpretation": "Post-birth cumulative work subtracts the inherited pre-birth ledger. Mechanical remainder retains endpoint/trapezoid quadrature and constitutive/geometric discrepancies; contact_geometric_work separately measures endpoint force work versus traction times local jump change. Neither is heat nor physical fracture energy. Small force and column-heat residuals do not close a global thermomechanical energy budget.",
        "limitations": [
            "One prescribed active vertex on a prescribed material path; crack-front release and a plate network are not simulated.",
            "The next held-strength crossing, if present, stops the run at an admissible lower bracket; no second contact is born.",
            "Two maximum timesteps measure sensitivity, not a demonstrated convergence order or a resolution-independent planetary prediction.",
            "New cohort counts depend on timestep partitioning; compare material area and histories, not counts as a convergence observable.",
            "Global mantle evolution and column heat accounting remain separate; the full thermal-mechanical energy budget is not closed."]}


def plot(coarse, fine, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    maximum = max(row["max_opening_m"] for report in (coarse, fine) for row in report["rows"])
    scale, unit = ((1e6, "мкм") if maximum < 1e-3 else
                   (1e3, "мм") if maximum < 1. else (1., "м"))
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 9.2), constrained_layout=True)
    for report, color in ((coarse, "tab:blue"), (fine, "tab:orange")):
        time = _time(report)
        rows = report["rows"]
        label = f"Шаг до {report['maximum_step_years']:g} лет"
        axes[0, 0].plot(time, [r["max_opening_m"]*scale for r in rows], color=color, label=label)
        axes[0, 1].plot(time, [r["initial_cohort_damage_max"] for r in rows], color=color, label=label)
        axes[1, 0].plot(time, [r["max_held_strength_ratio"] for r in rows], color=color, label=label)
        for field, style, name in (
                ("post_birth_mechanical_remainder_j", "-", "численный остаток"),
                ("fracture_work_j", "--", "разрушение"),
                ("post_birth_contact_geometric_work_j", ":", "геометрическая работа")):
            axes[1, 1].plot(time, [r[field] for r in rows], color=color, linestyle=style,
                            label=f"{report['maximum_step_years']:g} лет: {name}")
    axes[0, 0].set(title="Раскрытие первого контакта", ylabel=f"Максимальное раскрытие, {unit}", ylim=(0, None))
    contractions = [report["rows"][0]["radius_m"]-report["rows"][-1]["radius_m"] for report in (coarse, fine)]
    axes[0, 0].text(.02, .96,
        f"Уменьшение радиуса за каждый опыт: {contractions[0]:.1f} / {contractions[1]:.1f} м\n"
        "Раскрытие контакта показано отдельно",
        transform=axes[0, 0].transAxes, fontsize=8, va="top",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": .8})
    axes[0, 1].set(title="Повреждение исходного сечения", ylabel="Максимальное повреждение, доля", ylim=(0, None))
    axes[1, 0].axhline(1., color="black", linewidth=1, linestyle="--", label="Следующий порог прочности")
    axes[1, 0].set(title="Нагрузка на ещё связанные участки", ylabel="Нагрузка / прочность", ylim=(0, 1.05))
    axes[1, 1].set(title="Работа после рождения: разные статьи баланса", ylabel="Дж; симметричная логарифмическая шкала")
    axes[1, 1].set_yscale("symlog", linthresh=1.)
    axes[1, 1].axhline(0., color=".5", linewidth=.6)
    axes[1, 1].text(.02, .04, "Численный остаток и геометрическая работа\nне считаются теплом или энергией разрушения",
        transform=axes[1, 1].transAxes, fontsize=8, va="bottom",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": .8})
    for axis in axes.flat:
        axis.set_xlabel("Годы после рождения контакта")
        axis.set_xlim(left=0)
        axis.grid(alpha=.25)
        axis.legend(fontsize=8, loc="best")
    axes[0, 0].legend(fontsize=8, loc="lower right")
    axes[1, 1].legend(fontsize=7, loc="upper left")
    fig.suptitle(f"Сетка {coarse['parent_cells']} ячеек: раскрытие контакта при дальнейшем остывании\n"
        "Одна заданная активная область; движение фронтов и сеть плит ещё не моделируются", fontsize=13)
    fig.savefig(output/"comparison.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coarse", type=Path, required=True)
    parser.add_argument("--fine", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Comparison output must be new or empty")
    coarse, coarse_audit = load_trajectory(args.coarse)
    fine, fine_audit = load_trajectory(args.fine)
    result = comparison(coarse, fine)
    result["coarse"]["final_material"] = final_material(report_path(args.coarse), coarse)
    result["fine"]["final_material"] = final_material(report_path(args.fine), fine)
    result["input_provenance"] = {"coarse": coarse_audit, "fine": fine_audit}
    result["comparison_code_sha256"] = {str(Path(__file__).resolve()): digest(__file__)}
    output.mkdir(parents=True, exist_ok=True)
    plot(coarse, fine, output)
    result["artifact_sha256"] = {"comparison.png": digest(output/"comparison.png")}
    result["checks"] = {
        "all_source_code_and_artifact_hashes_verified": coarse_audit["all_matches"] and fine_audit["all_matches"],
        "both_trajectories_passed_validation": True,
        "identical_birth_inputs_and_executed_code": True,
        "common_physical_times_without_extrapolation": True,
        "final_cohort_material_matches_observations": True,
        "input_reports_unchanged": digest(coarse_audit["report_path"]) == coarse_audit["report_sha256"]
            and digest(fine_audit["report_path"]) == fine_audit["report_sha256"],
    }
    (output/"comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    end = result["common_time_comparison"]["selected_observations"][-1]
    print(json.dumps({"output": str(output), "common_end_years_after_birth": end["years_after_birth"],
        "coarse_opening_m": end["coarse"]["max_opening_m"],
        "fine_opening_m": end["fine"]["max_opening_m"],
        "opening_difference_m": end["fine_minus_coarse"]["max_opening_m"],
        "checks": result["checks"]}, indent=2))
    if not all(result["checks"].values()):
        raise SystemExit("Moving contact comparison audit failed")


if __name__ == "__main__":
    main()
