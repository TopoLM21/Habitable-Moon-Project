"""Validate a diagnostic ridge tracer against fixed physical analytic bands.

The NPZ outputs are sampled geometric paths, not simulation checkpoints. No
contact is inserted and no physical fracture, work, time or plates are inferred.
The Gaussian sigma, reconstruction support and integration step stay fixed
across meshes; coarse cases may explicitly decline to return a path.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.genesis_ridge_reference import RidgeReference
from tectonics.genesis_ridge import RidgeField, RidgeParameters, RidgeUnavailable
from tectonics.mesh import build_icosphere


# Fixed generic orientations avoid placing the reference along symmetry axes
# of an icosphere. The same physical fields are sampled on every refinement.
ROTATION_VECTORS = ((.41, -.22, .53), (-.37, .62, .19), (.73, .31, -.48))


def _coverage(reference, points):
    local = points @ reference.rotation
    longitude = np.unwrap(np.arctan2(local[:, 1], local[:, 0]))
    angle_span = float(np.ptp(longitude))
    if reference.kind == "great_circle":
        extent = reference.radius_km * longitude
        return {
            "analytic_length_covered_km": angle_span*reference.radius_km,
            "analytic_coverage_fraction": angle_span*reference.radius_km/(2*reference.half_length_km),
            "longitudinal_end_positions_km": [float(extent[0]), float(extent[-1])],
            "flat_plateau_fraction": (reference.half_length_km-reference.taper_km)/reference.half_length_km,
        }
    radius = reference.radius_km*np.sin(np.deg2rad(reference.small_circle_radius_degrees))
    return {
        "analytic_length_covered_km": angle_span*radius,
        "analytic_coverage_fraction": angle_span/(2*np.pi),
    }


def evaluate_case(mesh, reference, parameters, *, name, output, group, rotation_index):
    """Run one deterministic trace and preserve its points with exact errors."""
    started = time.perf_counter()
    samples = reference.sample(mesh.centroids)
    record = {
        "name": name, "group": group, "rotation_index": rotation_index,
        "cell_count": mesh.cell_count,
        "reference": reference.metadata(), "parameters": asdict(parameters),
        "median_cell_scale_km": float(np.median(np.sqrt(mesh.areas_unit_sphere))*reference.radius_km),
        "active_cell_count": int(samples.active.sum()),
    }
    field = RidgeField(mesh, samples.values, samples.plane_normals, reference.radius_km,
                       active=samples.active, parameters=parameters)
    try:
        path = field.trace(reference.seed)
    except RidgeUnavailable as exc:
        record.update(status="unavailable", reason=exc.reason,
                      elapsed_seconds=time.perf_counter()-started)
        print(json.dumps({"name": name, "status": "unavailable", "reason": exc.reason}), flush=True)
        return record

    if reference.kind == "uniform":
        # Retain an unexpected result so that the report explicitly fails the
        # negative control, rather than hiding it behind distance evaluation.
        error = np.full(len(path.points_xyz), np.nan)
        coverage = {}
    else:
        error = reference.distance_km(path.points_xyz)
        coverage = _coverage(reference, path.points_xyz)
    artifact_name = name + ".npz"
    np.savez_compressed(output / artifact_name,
        format=np.array("genesis-ridge-geometric-path-0.1"),
        interpretation=np.array("Geometric diagnostic only; not a physical checkpoint"),
        points_xyz=path.points_xyz, arclength_km=path.arclength_km,
        values=path.values, transverse_contrast=path.transverse_contrast,
        exact_distance_km=error,
        signed_cross_distance_km=(reference.signed_distance_km(path.points_xyz)
                                  if reference.kind != "uniform" else error),
        seed_index=np.array(path.seed_index), seed_xyz=reference.seed,
        left_stop=np.array(path.left_stop), right_stop=np.array(path.right_stop),
        closed=np.array(path.closed),
        reference_json=np.array(json.dumps(reference.metadata(), allow_nan=False)),
        parameters_json=np.array(json.dumps(asdict(parameters), allow_nan=False)))
    record.update(status="traced", point_count=len(path.points_xyz),
                  seed_index=path.seed_index, left_stop=path.left_stop,
                  right_stop=path.right_stop, closed=path.closed,
                  path_length_km=float(path.arclength_km[-1]),
                  error_rms_km=(float(np.sqrt(np.mean(error**2))) if reference.kind != "uniform" else None),
                  error_max_km=(float(error.max()) if reference.kind != "uniform" else None),
                  error_p95_km=(float(np.quantile(error, .95)) if reference.kind != "uniform" else None),
                  minimum_transverse_contrast=float(path.transverse_contrast.min()),
                  artifact=artifact_name, elapsed_seconds=time.perf_counter()-started, **coverage)
    print(json.dumps({key: record[key] for key in ("name", "status", "point_count", "error_rms_km",
                                                  "left_stop", "right_stop", "closed")}), flush=True)
    return record


def _draw_report(output, records):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {320: "#a3a5aa", 1280: "#b079a8", 5120: "#d78a2d", 20480: "#247ca5"}
    fig, axes = plt.subplots(2, 3, figsize=(18, 10.5), constrained_layout=True)
    matrix = [row for row in records if row["group"] == "mesh_matrix"]
    for col, kind in enumerate(("great_circle", "small_circle")):
        ax = axes[0, col]
        chosen = [row for row in matrix if row["reference"]["kind"] == kind and row["rotation_index"] == 0]
        ref = RidgeReference(kind=kind, rotation=Rotation.from_rotvec(ROTATION_VECTORS[0]).as_matrix())
        exact = ref.centerline(801) @ ref.rotation
        if kind == "great_circle":
            project = lambda p: (ref.radius_km*np.arctan2(p[:, 1], p[:, 0]),
                                 ref.radius_km*np.arcsin(np.clip(p[:, 2], -1., 1.)))
        else:
            def project(p):
                angle = np.arctan2(p[:, 1], p[:, 0])
                radius = ref.radius_km*np.arccos(np.clip(p[:, 2], -1., 1.))
                return radius*np.cos(angle), radius*np.sin(angle)
        ax.plot(*project(exact), color="#42474d", lw=2, ls="--", label="Точная центролиния")
        for row in chosen:
            if row["status"] != "traced":
                continue
            with np.load(output / row["artifact"], allow_pickle=False) as data:
                local = data["points_xyz"] @ ref.rotation
            ax.plot(*project(local), color=colors[row["cell_count"]], lw=1.7,
                    label=f"{row['cell_count']} ячеек")
        if kind == "great_circle":
            ax.set(xlabel="Вдоль большого круга, км", ylabel="Поперечное смещение, км",
                   title="Прямая полоса: остановка до концов дуги", ylim=(-3, 3))
            ax.axvspan(-4500, -3900, color="#777777", alpha=.1)
            ax.axvspan(3900, 4500, color="#777777", alpha=.1, label="Продольное затухание")
        else:
            ax.set(xlabel="Полярная проекция x, км", ylabel="Полярная проекция y, км",
                   title="Кривая полоса: ограниченная длина трассы", aspect="equal")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)

    ax = axes[0, 2]
    for kind, marker, label in (("great_circle", "o", "Прямая полоса"),
                                ("small_circle", "s", "Кривая полоса")):
        subset = [row for row in matrix if row["reference"]["kind"] == kind and row["status"] == "traced"]
        for rotation in range(len(ROTATION_VECTORS)):
            rows = [row for row in subset if row["rotation_index"] == rotation]
            ax.plot([row["cell_count"] for row in rows], [row["error_rms_km"] for row in rows],
                    marker=marker, color="#247ca5" if kind == "great_circle" else "#d78a2d",
                    alpha=.8, label=label if rotation == 0 else None)
    ax.set(xlabel="Число ячеек (логарифмическая шкала)", ylabel="RMS отклонения от центролинии, км",
           title="Три ориентации одного поля относительно сетки", xscale="log")
    ax.set_xticks([320, 1280, 5120, 20480], ["320", "1280", "5120", "20480"])
    ax.text(.02, .97, "320 / 1280: результат может быть\nотклонён из-за разрешения", transform=ax.transAxes,
            va="top", fontsize=9)
    ax.grid(alpha=.2)
    ax.legend(loc="center left", fontsize=9)

    ax = axes[1, 0]
    sensitivity = [row for row in records if row["group"] == "sensitivity" and row["status"] == "traced"]
    for support, color in ((900., "#8d60a0"), (1200., "#247ca5")):
        rows = sorted([row for row in sensitivity if row["parameters"]["fit_radius_km"] == support],
                      key=lambda row: row["parameters"]["step_km"])
        ax.plot([row["parameters"]["step_km"] for row in rows],
                [row["error_rms_km"] for row in rows], "o-", color=color,
                label=f"Радиус восстановления {support:.0f} км")
    ax.set(xlabel="Шаг вдоль трассы, км", ylabel="RMS отклонения, км",
           title="20 480 ячеек: шаг и радиус восстановления")
    ax.set_xticks([50, 100])
    ax.grid(alpha=.2)
    ax.legend(fontsize=9)

    ax = axes[1, 1]
    loop = next(row for row in records if row["group"] == "closed_loop")
    ax.axis("off")
    loop_text = "Полный обход кривой полосы\n\n"
    if loop["status"] == "traced":
        loop_text += (f"20 480 ячеек; замкнута: {'да' if loop['closed'] else 'нет'}\n"
                      f"Длина: {loop['path_length_km']:,.0f} км\n"
                      f"Покрытие аналитической окружности: {100*loop['analytic_coverage_fraction']:.2f}%\n"
                      f"RMS: {loop['error_rms_km']:.2f} км\n"
                      f"Стоп: {loop['right_stop']}\n\n")
    else:
        loop_text += f"Результат отклонён: {loop['reason']}\n\n"
    loop_text += "Постоянное поле — отрицательный контроль\n\n"
    for row in records:
        if row["group"] == "uniform_control":
            loop_text += f"{row['cell_count']:>5} ячеек: {row.get('reason', 'НЕОЖИДАННАЯ ТРАССА')}\n"
    ax.text(0., .98, loop_text, va="top", fontsize=10, linespacing=1.7)

    ax = axes[1, 2]
    ax.axis("off")
    ax.text(0., .98,
        "Что проверяет этот опыт\n\n"
        "Известная полоса задана аналитически.\n"
        "Её ширина σ = 600 км не зависит от сетки.\n"
        "Начальная точка задана явно.\n"
        "Параметры одинаковы на разных сетках.\n\n"
        "Получена геометрическая линия максимумов.\n"
        "Это не физическое распространение трещины.\n"
        "Время, энергия и число плит не вычисляются.\n"
        "Контакты и сохранения симуляции не меняются.\n\n"
        "Уменьшение шага не убирает смещение,\n"
        "вносимое пространственным восстановлением.\n"
        "Продольное затухание может остановить трассу\n"
        "раньше конца заданной аналитической дуги.",
        va="top", fontsize=10, linespacing=1.7)
    fig.suptitle("Прослеживание локализованных полос: геометрическая проверка", fontsize=17)
    fig.savefig(output / "validation.png", dpi=170)
    plt.close(fig)


def _check_artifacts(output, records):
    """Check saved geometry and the actual, corrected arclength budgets."""
    count = 0
    for row in records:
        if row["status"] != "traced":
            continue
        with np.load(output / row["artifact"], allow_pickle=False) as data:
            points = data["points_xyz"]
            np.testing.assert_allclose(np.linalg.norm(points, axis=1), 1., atol=3e-15)
            arc = data["arclength_km"]
            if not np.all(np.diff(arc) > 0):
                raise AssertionError("Saved arclength must increase strictly")
            if row["reference"]["kind"] != "uniform" and not np.isfinite(data["exact_distance_km"]).all():
                raise AssertionError("Saved reference errors must be finite")
            if bool(data["closed"]):
                np.testing.assert_array_equal(points[0], points[-1])
            else:
                seed = int(data["seed_index"])
                limit = row["parameters"]["max_branch_length_km"]
                if arc[seed]-arc[0] > limit+1e-6 or arc[-1]-arc[seed] > limit+1e-6:
                    raise AssertionError("Corrected branch exceeds its length budget")
        count += 1
    return {"npz_count": count, "unit_points": True,
            "strictly_increasing_arclength": True, "branch_length_budget": True,
            "closed_endpoints_exact": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Validation output must be a new or empty directory")
    output.mkdir(parents=True, exist_ok=True)
    meshes = {subdivision: build_icosphere(subdivision) for subdivision in (2, 3, 4, 5)}
    default = RidgeParameters()
    records = []
    for kind in ("great_circle", "small_circle"):
        for index, vector in enumerate(ROTATION_VECTORS):
            reference = RidgeReference(kind=kind, rotation=Rotation.from_rotvec(vector).as_matrix())
            for subdivision, mesh in meshes.items():
                records.append(evaluate_case(mesh, reference, default,
                    name=f"{kind}_rotation{index}_{mesh.cell_count}", output=output,
                    group="mesh_matrix", rotation_index=index))
    rotation = Rotation.from_rotvec(ROTATION_VECTORS[0]).as_matrix()
    reference = RidgeReference(kind="uniform", rotation=rotation)
    for mesh in meshes.values():
        records.append(evaluate_case(mesh, reference, default,
            name=f"uniform_{mesh.cell_count}", output=output, group="uniform_control", rotation_index=0))
    reference = RidgeReference(kind="small_circle", rotation=rotation)
    for support in (900., 1200.):
        for step in (50., 100.):
            parameters = replace(default, fit_radius_km=support, step_km=step)
            records.append(evaluate_case(meshes[5], reference, parameters,
                name=f"sensitivity_support{support:.0f}_step{step:.0f}", output=output,
                group="sensitivity", rotation_index=0))
    records.append(evaluate_case(meshes[5], reference,
        replace(default, max_branch_length_km=30000.), name="closed_loop_20480",
        output=output, group="closed_loop", rotation_index=0))
    report = {
        "format": "genesis-ridge-validation-0.1",
        "interpretation": "Geometric seeded-path benchmark; no physical crack growth or plate certification",
        "rotation_vectors_radians": ROTATION_VECTORS,
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                          for name in ("tectonics/genesis_ridge.py",
                                       "analysis/genesis_ridge_reference.py",
                                       "analysis/genesis_ridge_validation.py")},
        "parameters": asdict(default), "records": records,
        "artifact_integrity_checks": _check_artifacts(output, records),
        "uniform_control_passed": all(row["status"] == "unavailable"
                                      for row in records if row["group"] == "uniform_control"),
        "limitations": [
            "Reference seeds and physical band width are supplied rather than inferred from a damage simulation.",
            "Fixed-support scalar reconstruction can bias curved ridges even after mesh refinement.",
            "Trace integration step measures geometric arclength, not years or fracture propagation speed.",
            "Longitudinal taper can stop the trace before the end of the reference great-circle segment.",
            "No sharp-contact insertion, propagation-energy law, network branching or production checkpoint changes.",
        ],
    }
    (output / "validation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
                                             encoding="utf-8")
    _draw_report(output, records)
    print(json.dumps({"output": str(output), "cases": len(records),
                      "uniform_control_passed": report["uniform_control_passed"]}), flush=True)
    return report


if __name__ == "__main__":
    main()
