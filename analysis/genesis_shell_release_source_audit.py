"""Read-only virtual-cut diagnostics on saved planetary fault states.

Every case assumes an arbitrary, already open half-base-edge notch and asks
about a quarter-edge extension. Material vertex IDs come from the nested
fixture; actual source coordinates, elastic memory, softened stiffness, solid
depth and loads come from the checkpoint. The selected path is neither the
1532 km localization ridge nor a spontaneous fracture/nucleation prediction.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.genesis_shell_release_fixture import make_fixture
from tectonics.genesis_shell_release import FrozenShell, VERSION


CASES = (
    ("strong_320", 2, "results/genesis_runs/fault_damage_final_20260922/fault_checkpoint.npz"),
    ("strong_1280", 3, "results/genesis_runs/fine_fault_audit_20260924/strong_1280/fault_checkpoint.npz"),
    ("strong_5120", 4, "results/genesis_runs/fine_fault_audit_20260924/strong_5120/fault_checkpoint.npz"),
    ("user_5120", 4, "results/genesis_runs/genesis_20260924_194255_087614/fault_checkpoint.npz"),
)
IMPLEMENTATION = (
    "analysis/genesis_shell_release_source_audit.py",
    "analysis/genesis_shell_release_fixture.py",
    "tectonics/genesis_shell_release.py", "tectonics/genesis_shell.py",
    "tectonics/genesis_seams.py", "tectonics/genesis_contact.py",
    "tectonics/genesis_faults.py", "tectonics/mesh.py",
)


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stats(values):
    values = np.asarray(values)
    return {"min": float(values.min()), "median": float(np.median(values)),
            "mean": float(values.mean()), "max": float(values.max())}


def _equilibrium(equilibrium, shell):
    q = equilibrium.displacement_m
    tangent_motion = np.linalg.norm(q[:-1].reshape(-1, 2), axis=1)
    radial_fraction = abs(float(q[-1]))/shell.radius_m
    gates = {
        "equilibrium": {"value": float(equilibrium.equilibrium_residual),
                        "limit": shell.equilibrium_tolerance,
                        "passed": bool(equilibrium.equilibrium_residual <= shell.equilibrium_tolerance)},
        "rotation_constraint": {"value": float(equilibrium.constraint_residual),
                                "limit": shell.equilibrium_tolerance,
                                "passed": bool(equilibrium.constraint_residual <= shell.equilibrium_tolerance)},
        "added_strain": {"value": float(equilibrium.max_added_strain),
                         "limit": shell.max_strain,
                         "passed": bool(equilibrium.max_added_strain <= shell.max_strain)},
        "radial_strain": {"value": radial_fraction, "limit": shell.max_strain,
                          "passed": bool(radial_fraction <= shell.max_strain)},
        "motion_to_shortest_reference_edge": {
            "value": float(equilibrium.max_motion_edge_fraction),
            "limit": shell.max_motion_edge_fraction,
            "passed": bool(equilibrium.max_motion_edge_fraction <= shell.max_motion_edge_fraction)},
        "no_free_bank_interpenetration": {
            "minimum_gap_m": float(equilibrium.min_gap_m),
            "allowed_negative_gap_m": shell.penetration_tolerance_m,
            "passed": bool(equilibrium.min_gap_m >= -shell.penetration_tolerance_m)},
    }
    return {
        "stored_energy_j": float(equilibrium.stored_energy_j),
        "external_potential_work_j": float(equilibrium.external_potential_work_j),
        "potential_energy_j": float(equilibrium.potential_energy_j),
        "reduced_potential_j": float(equilibrium.reduced_potential_j),
        "max_added_strain": float(equilibrium.max_added_strain),
        "max_motion_edge_fraction": float(equilibrium.max_motion_edge_fraction),
        "max_tangent_displacement_m": float(tangent_motion.max()),
        "common_radial_displacement_m": float(q[-1]),
        "max_total_displacement_m": float(np.sqrt(tangent_motion**2+q[-1]**2).max()),
        "minimum_normal_gap_m": float(equilibrium.min_gap_m),
        "maximum_normal_gap_m": float(np.max(equilibrium.normal_gap_m, initial=0.)),
        "maximum_abs_tangential_jump_m": float(np.max(abs(equilibrium.tangential_jump_m), initial=0.)),
        "cut_edge_count": len(equilibrium.topology.cut_edges),
        "split_vertex_count": equilibrium.topology.mesh.vertex_count,
        "admissible_open_crack": bool(equilibrium.admissible_open_crack),
        "rejection_reasons": list(equilibrium.rejection_reasons), "gates": gates,
    }


def _geometry(shell, fixture, state):
    path = fixture.path_vertices
    a, b = shell.mesh.vertices[path[:-1]], shell.mesh.vertices[path[1:]]
    lengths = shell.radius_m*np.arctan2(np.linalg.norm(np.cross(a, b), axis=1),
                                      np.sum(a*b, axis=1))
    end = fixture.seed_edge_count+fixture.extension_edge_count
    owners = {tuple(sorted((u, v))): (fa, fb) for fa, fb, u, v in shell.mesh.shared_edges}
    adjacent = np.asarray([owners[tuple(sorted(edge))] for edge in fixture.trial_cuts])
    active = state["fault_active"]
    return {
        "selection": "Prescribed material-vertex path 0 to 1; first half already cut, next quarter tested",
        "not_a_localization_trace": True, "assumes_preexisting_free_notch": True,
        "path_vertices": path.tolist(), "path_points_xyz": shell.mesh.vertices[path].tolist(),
        "seed_cuts": fixture.seed_cuts.tolist(), "trial_cuts": fixture.trial_cuts.tolist(),
        "seed_edge_count": fixture.seed_edge_count,
        "extension_edge_count": fixture.extension_edge_count,
        "edge_lengths_m": lengths.tolist(),
        "seed_length_m": float(lengths[:fixture.seed_edge_count].sum()),
        "extension_length_m": float(lengths[fixture.seed_edge_count:end].sum()),
        "trial_length_m": float(lengths[:end].sum()),
        "trial_adjacent_material_faces": adjacent.tolist(),
        "trial_adjacent_unique_active_fault_cells": int(np.count_nonzero(active[np.unique(adjacent)])),
        "trial_edges_with_two_active_fault_cells": int(np.count_nonzero(active[adjacent].all(axis=1))),
        "trial_adjacent_damage": _stats(state["damage"][np.unique(adjacent)]),
    }


def _save_fields(output, label, shell, fixture, report):
    filename = output/f"{label}_equilibria.npz"
    fields = dict(reference_vertices_xyz=shell.mesh.vertices, reference_faces=shell.mesh.faces,
                  radius_m=np.array(shell.radius_m), depth_m=shell.depth_m,
                  elasticity_pa=shell.elasticity_pa, source_elastic_strain=shell.elastic_strain,
                  source_vertex_force_xyz_n=shell.vertex_force_xyz_n,
                  path_vertices=fixture.path_vertices,
                  seed_edge_count=np.array(fixture.seed_edge_count),
                  extension_edge_count=np.array(fixture.extension_edge_count))
    for prefix, equilibrium in (("before", report.before), ("after", report.after)):
        fields.update({f"{prefix}_{name}": value for name, value in (
            ("displacement_m", equilibrium.displacement_m),
            ("normal_gap_m", equilibrium.normal_gap_m),
            ("tangential_jump_m", equilibrium.tangential_jump_m),
            ("cut_edges", equilibrium.topology.cut_edges),
            ("parent_vertex", equilibrium.topology.parent_vertex),
            ("split_faces", equilibrium.topology.mesh.faces),
        )})
    np.savez_compressed(filename, **fields)
    return {"file": filename.name, "sha256": _hash(filename),
            "meaning": "Virtual-equilibrium diagnostic arrays; not a physical front or checkpoint"}


def run_case(label, subdivisions, source, output):
    started = time.perf_counter()
    data = source.read_bytes()
    source_hash = hashlib.sha256(data).hexdigest()
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        state = {name: archive[name].copy() for name in ("fault_active", "damage")}
    shell = FrozenShell.from_fault_checkpoint(source)
    if shell.source_hash != source_hash:
        raise RuntimeError("Checkpoint changed while constructing the frozen source")
    fixture = make_fixture(subdivisions, radius_m=shell.radius_m)
    if not np.array_equal(shell.mesh.faces, fixture.mesh.faces):
        raise ValueError("Fixture material IDs do not match source topology")
    report = shell.compare_extension(fixture.seed_cuts, fixture.trial_cuts)
    fields = (
        "release_j", "added_area_m2", "mean_release_j_m2", "potential_difference_j",
        "relaxation_energy_j", "embedding_energy_error_j", "force_pullback_relative_error",
        "prestress_pullback_relative_error", "stiffness_pullback_relative_error",
        "gauge_pullback_relative_error", "relative_release_identity_error",
        "material_area_relative_error", "material_volume_relative_error",
    )
    row = {name: float(getattr(report, name)) for name in fields}
    row.update(
        label=label, checkpoint=str(source.relative_to(ROOT)), checkpoint_sha256_before=source_hash,
        cell_count=shell.mesh.cell_count, subdivisions=subdivisions,
        source_time_myr=float(metadata["state"]["time_myr"]),
        convective_traction_parameter_pa=float(metadata["parameters"]["shell"]["convective_traction_pa"]),
        active_fault_cell_count=int(np.count_nonzero(state["fault_active"])),
        active_fault_area_fraction=float(np.average(state["fault_active"], weights=shell.mesh.areas_unit_sphere)),
        radius_m=shell.radius_m, solid_depth_m=_stats(shell.depth_m),
        geometry=_geometry(shell, fixture, state), source_fingerprint=shell.fingerprint,
        before=_equilibrium(report.before, shell), after=_equilibrium(report.after, shell),
        admissible_open_crack=bool(report.admissible_open_crack),
        rejection_reasons=list(report.rejection_reasons),
        energy_status="admissible_only_for_assumed_free_notch" if report.admissible_open_crack
                      else "inadmissible_equilibrium_diagnostic_only",
        elapsed_seconds=time.perf_counter()-started,
    )
    if shell.mesh.cell_count == 5120:
        row["field_archive"] = _save_fields(output, label, shell, fixture, report)
    row["checkpoint_sha256_after"] = _hash(source)
    if row["checkpoint_sha256_after"] != source_hash:
        raise RuntimeError("Read-only diagnostic changed its source checkpoint")
    print(json.dumps({key: row[key] for key in (
        "label", "source_time_myr", "active_fault_cell_count", "mean_release_j_m2",
        "admissible_open_crack", "rejection_reasons", "elapsed_seconds")}), flush=True)
    return row


def draw_report(output, report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    rows = report["cases"]
    labels = [f"50 кПа\n{row['cell_count']} яч." if row["label"].startswith("strong")
              else "20 кПа\n5120 яч." for row in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for ax, key, title in (
        (axes[0, 0], "added_strain", "Добавленная деформация / предел 0,5%"),
        (axes[0, 1], "motion_to_shortest_reference_edge", "Смещение / предел 2% длины ребра"),
    ):
        for prefix, shift, color, label in (
            ("before", -.18, "#728899", "Начальная заданная трещина"),
            ("after", .18, "#258eb0", "После пробного удлинения"),
        ):
            values = [row[prefix]["gates"][key]["value"]/row[prefix]["gates"][key]["limit"]
                      for row in rows]
            bars = ax.bar(x+shift, values, .36, color=color, label=label)
            for bar, row in zip(bars, rows):
                if not row[prefix]["admissible_open_crack"]:
                    bar.set_hatch("xxx")
                    bar.set_edgecolor("#8b3131")
        ax.axhline(1., color="#a52d32", linestyle="--", linewidth=1.4)
        ax.set_yscale("log")
        ax.set(title=title, xticks=x, xticklabels=labels, ylabel="Доля допустимого предела; выше 1 — превышение")
        ax.grid(axis="y", alpha=.22)
        ax.legend(fontsize=8)

    ax = axes[1, 0]
    bars = ax.bar(x, [row["mean_release_j_m2"] for row in rows], color="#258eb0", width=.62)
    for bar, row in zip(bars, rows):
        if not row["admissible_open_crack"]:
            bar.set_hatch("xxx")
            bar.set_edgecolor("#8b3131")
            bar.set_facecolor("#edd6d4")
        ax.annotate(f"{row['mean_release_j_m2']:.2e}",
                    (bar.get_x()+bar.get_width()/2, bar.get_height()),
                    xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8)
    ax.set_yscale("log")
    ax.set(title="Средняя энергия пробного продвижения ΔΠ / ΔA",
           ylabel="Дж/м²; штриховка — недопустимое равновесие", xticks=x, xticklabels=labels)
    ax.margins(y=.2)
    ax.grid(axis="y", alpha=.22)
    ax.legend(handles=[Patch(facecolor="#edd6d4", edgecolor="#8b3131", hatch="xxx",
                             label="Энергия не разрешает рост при нарушенных пределах")], fontsize=8)

    ax = axes[1, 1]
    ax.axis("off")
    header = ["Источник", "Активные\nячейки", "max |u|\nпосле, км", "max зазор\nпосле, км"]
    body = [[label.replace("\n", " / "), str(row["active_fault_cell_count"]),
             f"{row['after']['max_total_displacement_m']/1000:.3g}",
             f"{row['after']['maximum_normal_gap_m']/1000:.3g}"]
            for label, row in zip(labels, rows)]
    table = ax.table(cellText=body, colLabels=header, loc="upper center", cellLoc="center",
                     bbox=[0, .53, 1., .44], colWidths=[.34, .19, .23, .24])
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    seed_lengths = [row["geometry"]["seed_length_m"]/1000 for row in rows]
    extension_lengths = [row["geometry"]["extension_length_m"]/1000 for row in rows]
    ax.text(0., .44, "Во всех опытах трещина задана заранее:\n"
            f"{min(seed_lengths):.0f}–{max(seed_lengths):.0f} км разреза + "
            f"{min(extension_lengths):.0f}–{max(extension_lengths):.0f} км пробы.\n\n"
            "У источника 20 кПа нет активированных слабых плоскостей.\n"
            "Допустимость искусственного надреза не означает,\n"
            "что такая трещина зародилась или вырастет сама.\n\n"
            "Это не извлечённая ранее линия локализации 1532 км.\n"
            "Время, контакт и когезионное разрушение здесь не развиваются.",
            transform=ax.transAxes, va="top", fontsize=9.5, linespacing=1.35)
    fig.suptitle("Сохранённые состояния: виртуальное удлинение произвольно заданного разреза", fontsize=14)
    target = output/"source_audit.png"
    fig.savefig(target, dpi=160)
    plt.close(fig)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("Output must be a new or empty directory")
    output.mkdir(parents=True, exist_ok=True)
    implementation_before = {path: _hash(ROOT/path) for path in IMPLEMENTATION}
    sources_before = {path: _hash(ROOT/path) for _, _, path in CASES}
    rows = [run_case(label, subdivisions, ROOT/path, output) for label, subdivisions, path in CASES]
    sources_after = {path: _hash(ROOT/path) for path in sources_before}
    implementation_after = {path: _hash(ROOT/path) for path in IMPLEMENTATION}
    if sources_before != sources_after or implementation_before != implementation_after:
        raise RuntimeError("Sources or implementation changed while the audit was running")
    result = {
        "format": "genesis-shell-release-source-audit-0.1", "solver_version": VERSION,
        "purpose": "Read-only frozen-shell energy diagnostic under a prescribed pre-existing notch",
        "path_is_arbitrary_not_ridge": True, "spontaneous_nucleation_tested": False,
        "growth_or_time_advanced": False, "fracture_toughness_applied": False,
        "coupled_contact_or_heat_evolved": False,
        "sources_sha256_before": sources_before, "sources_sha256_after": sources_after,
        "implementation_sha256_before": implementation_before,
        "implementation_sha256_after": implementation_after, "cases": rows,
    }
    draw_report(output, result)
    (output/"source_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2,
                                                    allow_nan=False), encoding="utf-8")
    lines = ["# Виртуальные разрезы в сохранённых состояниях", "",
             "Здесь произвольно задан существующий разрез вдоль материального пути 0→1; "
             "это не след локализации и не опыт зарождения трещины. Источники не изменены, "
             "время не продвинуто. Энергия недопустимых равновесий — только диагностическое число.", "",
             "| Источник | Время, млн лет | Активные ячейки | Начальный разрез, км | Проба, км | ΔΠ/ΔA, Дж/м² | Допустимость |",
             "|---|---:|---:|---:|---:|---:|---|"]
    for row in rows:
        decision = "Только при заданном свободном надрезе" if row["admissible_open_crack"] else ", ".join(row["rejection_reasons"])
        lines.append(f"| {row['label']} | {row['source_time_myr']:.3g} | {row['active_fault_cell_count']} | "
                     f"{row['geometry']['seed_length_m']/1000:.3f} | {row['geometry']['extension_length_m']/1000:.3f} | "
                     f"{row['mean_release_j_m2']:.6g} | {decision} |")
    lines.extend(["", "Допустимость навязанного большого разреза в источнике с 20 кПа не означает "
                  "самопроизвольное разрушение: число активированных слабых плоскостей в источнике остаётся нулевым.",
                  "", "Полные пределы, причины отказа, невязки, глубины, геометрия и контрольные суммы — в source_audit.json."])
    (output/"source_audit.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
