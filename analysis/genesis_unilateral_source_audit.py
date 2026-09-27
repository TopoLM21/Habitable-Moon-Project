"""Read-only hard-contact and bounded load continuation on saved fault states.

The cut is prescribed, and only the multiplier of the frozen external force
changes. Prestress is fixed even at multiplier zero. Neither geological time
nor the material reference geometry advances, and no crack is nucleated.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
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
from analysis.genesis_shell_release_source_audit import CASES, _equilibrium, _geometry, _hash, _stats
from tectonics.genesis_shell_release import FrozenShell
from tectonics.genesis_unilateral import UnilateralShell, LoadPathParameters, VERSION


IMPLEMENTATION = (
    "analysis/genesis_unilateral_source_audit.py",
    "analysis/genesis_shell_release_source_audit.py",
    "analysis/genesis_shell_release_fixture.py",
    "tectonics/genesis_unilateral.py", "tectonics/genesis_shell_release.py",
    "tectonics/genesis_shell.py", "tectonics/genesis_seams.py",
    "tectonics/genesis_contact.py", "tectonics/genesis_contact_law.py",
    "tectonics/genesis_faults.py", "tectonics/genesis_fault_law.py",
    "tectonics/genesis_mobile.py", "tectonics/genesis_material.py",
    "tectonics/genesis.py", "tectonics/genesis_onset.py", "tectonics/genesis_tides.py",
    "tectonics/genesis_seam_diagnostics.py", "tectonics/mesh.py",
)


def _contact_equilibrium(equilibrium, model):
    """Reuse the scalar mechanical report, with explicit unilateral semantics."""
    row = _equilibrium(equilibrium, model.shell)
    row["admissible_contact"] = row.pop("admissible_open_crack")
    row["gates"]["impenetrability"] = row["gates"].pop("no_free_bank_interpenetration")
    row["gates"]["reference_strain"] = row["gates"].pop("added_strain")
    row["gates"]["complementarity"] = {
        "value": float(equilibrium.complementarity_relative_error),
        "limit": model.parameters.complementarity_tolerance,
        "passed": bool(equilibrium.complementarity_relative_error <= model.parameters.complementarity_tolerance),
    }
    row.update(
        load_factor=float(equilibrium.load_factor),
        active_contact_count=int(equilibrium.active_contact_count),
        dual_rank=int(equilibrium.dual_rank),
        contact_work_j=float(equilibrium.contact_work_j),
        complementarity_relative_error=float(equilibrium.complementarity_relative_error),
        minimum_normal_reaction_n=float(np.min(equilibrium.normal_reaction_n, initial=0.)),
        maximum_normal_reaction_n=float(np.max(equilibrium.normal_reaction_n, initial=0.)),
        normal_contact_nodal_force_norm_n=float(np.linalg.norm(equilibrium.normal_operator.T @ equilibrium.normal_reaction_n)),
        reference_strain_meaning="Displacement from the original frozen material reference; not a reset at each load increment",
    )
    return row


def _save_fields(output, label, shell, fixture, report, continuation):
    filename = output/f"{label}_contact_fields.npz"
    data = dict(
        reference_vertices_xyz=shell.mesh.vertices, reference_faces=shell.mesh.faces,
        radius_m=np.array(shell.radius_m), depth_m=shell.depth_m,
        elasticity_pa=shell.elasticity_pa, source_elastic_strain=shell.elastic_strain,
        source_vertex_force_xyz_n=shell.vertex_force_xyz_n,
        path_vertices=fixture.path_vertices, seed_cuts=fixture.seed_cuts,
        trial_cuts=fixture.trial_cuts,
        load_attempts_json=np.array(json.dumps([asdict(item) for item in continuation.attempts], allow_nan=False)),
        continuation_stop_reason=np.array(continuation.stop_reason),
        continuation_reached_target=np.array(continuation.reached_target),
        has_last_accepted=np.array(continuation.last_accepted is not None),
    )
    equilibria = [("seed_target", report.before), ("trial_target", report.after),
                  ("trial_initial", continuation.initial)]
    if continuation.last_accepted is not None:
        equilibria.append(("trial_last_accepted", continuation.last_accepted))
    for prefix, equilibrium in equilibria:
        operator = equilibrium.normal_operator.tocsr()
        data.update({f"{prefix}_{name}": value for name, value in (
            ("load_factor", np.array(equilibrium.load_factor)),
            ("displacement_m", equilibrium.displacement_m),
            ("normal_gap_m", equilibrium.normal_gap_m),
            ("tangential_jump_m", equilibrium.tangential_jump_m),
            ("normal_reaction_n", equilibrium.normal_reaction_n),
            ("normal_contact_nodal_force_n", operator.T @ equilibrium.normal_reaction_n),
            ("normal_operator_data", operator.data),
            ("normal_operator_indices", operator.indices),
            ("normal_operator_indptr", operator.indptr),
            ("normal_operator_shape", np.array(operator.shape)),
            ("initial_bulk_force_n", equilibrium.initial_bulk_force),
            ("external_force_n", equilibrium.external_force),
            ("cut_edges", equilibrium.topology.cut_edges),
            ("parent_vertex", equilibrium.topology.parent_vertex),
            ("split_faces", equilibrium.topology.mesh.faces),
        )})
    np.savez_compressed(filename, **data)
    return {"file": filename.name, "sha256": _hash(filename),
            "meaning": "Diagnostic equilibria and load attempts; trial_target can be rejected. Not a physical front or restart checkpoint",
            "normal_operator_format": "CSR, Jq=normal_gap_m; nodal contact force=J.T @ normal_reaction_n"}


def run_case(label, subdivisions, source, output):
    started = time.perf_counter()
    raw = source.read_bytes()
    source_hash = hashlib.sha256(raw).hexdigest()
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        state = {name: archive[name].copy() for name in ("fault_active", "damage")}
    shell = FrozenShell.from_fault_checkpoint(source)
    if shell.source_hash != source_hash:
        raise RuntimeError("Checkpoint changed while constructing the frozen source")
    fixture = make_fixture(subdivisions, radius_m=shell.radius_m)
    if not np.array_equal(shell.mesh.faces, fixture.mesh.faces):
        raise ValueError("Fixture material IDs do not match source topology")
    model = UnilateralShell(shell)
    report = model.compare_extension(fixture.seed_cuts, fixture.trial_cuts)
    parameters = LoadPathParameters()
    continuation = model.continue_loading(fixture.trial_cuts, parameters=parameters)
    fields = (
        "release_j", "added_area_m2", "mean_release_j_m2", "potential_difference_j",
        "relaxation_energy_j", "embedding_energy_error_j", "force_pullback_relative_error",
        "prestress_pullback_relative_error", "stiffness_pullback_relative_error",
        "gauge_pullback_relative_error", "relative_release_identity_error",
        "material_area_relative_error", "material_volume_relative_error",
        "contact_release_term_j", "lifted_min_gap_m", "energy_roundoff_bound_j",
    )
    row = {name: float(getattr(report, name)) for name in fields}
    row.update(
        label=label, checkpoint=str(source.relative_to(ROOT)),
        checkpoint_sha256_before=source_hash, cell_count=shell.mesh.cell_count,
        subdivisions=subdivisions, source_time_myr=float(metadata["state"]["time_myr"]),
        convective_traction_parameter_pa=float(metadata["parameters"]["shell"]["convective_traction_pa"]),
        active_fault_cell_count=int(np.count_nonzero(state["fault_active"])),
        active_fault_area_fraction=float(np.average(state["fault_active"], weights=shell.mesh.areas_unit_sphere)),
        radius_m=shell.radius_m, solid_depth_m=_stats(shell.depth_m),
        geometry=_geometry(shell, fixture, state), source_fingerprint=shell.fingerprint,
        contact_solve_parameters=asdict(model.parameters), load_path_parameters=asdict(parameters),
        seed_target=_contact_equilibrium(report.before, model),
        trial_target=_contact_equilibrium(report.after, model),
        admissible_contact_extension=bool(report.admissible_open_crack),
        released_energy_resolved=bool(report.released_energy_resolved),
        rejection_reasons=list(report.rejection_reasons),
        energy_status="admissible_only_for_assumed_frictionless_notch" if report.admissible_open_crack
                      else "inadmissible_equilibrium_diagnostic_only",
        continuation={
            "initial": _contact_equilibrium(continuation.initial, model),
            "last_accepted": (_contact_equilibrium(continuation.last_accepted, model)
                              if continuation.last_accepted is not None else None),
            "attempts": [asdict(item) for item in continuation.attempts],
            "accepted_step_count": sum(item.accepted for item in continuation.attempts),
            "rejected_attempt_count": sum(not item.accepted for item in continuation.attempts),
            "reached_target": bool(continuation.reached_target),
            "stop_reason": continuation.stop_reason,
        },
        elapsed_seconds=time.perf_counter()-started,
    )
    # Contact banks can close: inherited free-notch terminology is overridden.
    row["geometry"].pop("assumes_preexisting_free_notch", None)
    row["geometry"]["assumes_preexisting_frictionless_crack"] = True
    if shell.mesh.cell_count == 5120:
        row["field_archive"] = _save_fields(output, label, shell, fixture, report, continuation)
    row["checkpoint_sha256_after"] = _hash(source)
    if row["checkpoint_sha256_after"] != source_hash:
        raise RuntimeError("Read-only diagnostic changed its source checkpoint")
    print(json.dumps({
        "label": label, "source_time_myr": row["source_time_myr"],
        "initial_active_contact_count": continuation.initial.active_contact_count,
        "target_active_contact_count": report.after.active_contact_count,
        "initial_minimum_gap_m": continuation.initial.min_gap_m,
        "target_admissible": row["admissible_contact_extension"],
        "load_path_stop_reason": continuation.stop_reason,
        "accepted_steps": row["continuation"]["accepted_step_count"],
        "elapsed_seconds": row["elapsed_seconds"],
    }), flush=True)
    return row


def draw_report(output, report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    rows = report["cases"]
    labels = [f"50 кПа\n{row['cell_count']} яч." if row["label"].startswith("strong")
              else "20 кПа\n5120 яч." for row in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for ax, key, title in (
        (axes[0, 0], "reference_strain", "Деформация от исходной геометрии / предел 0,5%"),
        (axes[0, 1], "motion_to_shortest_reference_edge", "Смещение / предел 2% длины ребра"),
    ):
        for stage, shift, color, label in (
            ("initial", -.18, "#748b9c", "α=0: только сохранённое напряжение"),
            ("target", .18, "#258eb0", "α=1: полная внешняя нагрузка"),
        ):
            eqs = [row["continuation"]["initial"] if stage == "initial" else row["trial_target"]
                   for row in rows]
            values = [eq["gates"][key]["value"]/eq["gates"][key]["limit"] for eq in eqs]
            bars = ax.bar(x+shift, values, .36, color=color, label=label)
            for bar, eq in zip(bars, eqs):
                if not eq["admissible_contact"]:
                    bar.set_hatch("xxx")
                    bar.set_edgecolor("#8b3131")
            for bar, value in zip(bars, values):
                ax.annotate(f"{value:.2g}", (bar.get_x()+bar.get_width()/2, bar.get_height()),
                            xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
        ax.axhline(1., color="#a52d32", linestyle="--", linewidth=1.4)
        ax.set_yscale("log")
        ax.set(title=title, xticks=x, xticklabels=labels, ylabel="Доля предела; штриховка — отказ")
        ax.grid(axis="y", alpha=.22)
        ax.legend(fontsize=8)
        ax.margins(y=.22)

    ax = axes[1, 0]
    for stage, shift, color, label in (
        ("initial", -.18, "#748b9c", "α=0"),
        ("target", .18, "#258eb0", "α=1, в том числе отклонённые решения"),
    ):
        values = [(row["continuation"]["initial"] if stage == "initial" else row["trial_target"])["active_contact_count"]
                  for row in rows]
        bars = ax.bar(x+shift, values, .36, color=color, label=label)
        for bar, value in zip(bars, values):
            ax.annotate(str(value), (bar.get_x()+bar.get_width()/2, value),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
    ax.set(title="Контактные реакции в концах рёбер заданного разреза",
           ylabel="Число ненулевых реакций; не число разломов", xticks=x, xticklabels=labels)
    ax.set_ylim(0, max(1, ax.get_ylim()[1])*1.2)
    ax.yaxis.get_major_locator().set_params(integer=True)
    ax.grid(axis="y", alpha=.22)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.axis("off")
    body = []
    for label, row in zip(labels, rows):
        path = row["continuation"]
        last = path["last_accepted"]
        body.append([label.replace("\n", " / "), str(path["accepted_step_count"]),
                     "—" if last is None else f"{last['load_factor']:.3g}",
                     "Достигнута α=1" if path["reached_target"] else "Предел при α=0" if last is None else "Предел"])
    table = ax.table(cellText=body, colLabels=["Источник", "Принято\nшагов", "Последняя\nпринятая α", "Результат"],
                     bbox=[0, .58, 1., .4], cellLoc="center", colWidths=[.32, .17, .22, .29])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    ax.text(0., .51, "α — множитель внешней силы, а не время.\n"
            "Сохранённое напряжение при α=0 не обнуляется.\n"
            "Шаги не сбрасывают суммарную деформацию.\n\n"
            "Контакт исключает взаимное проникновение берегов,\n"
            "но не устраняет чрезмерное свободное раскрытие.\n"
            "Недопустимая исходная релаксация останавливает расчёт.\n\n"
            "Разрез задан заранее: это не зарождение трещины\n"
            "и не найденная линия локализации 1532 км.\n"
            "Нет трения, когезии, роста трещины или прогрева.",
            transform=ax.transAxes, va="top", fontsize=9, linespacing=1.3)
    fig.suptitle("Сохранённые состояния: контакт берегов и ограниченные приращения нагрузки", fontsize=14)
    path = output/"source_audit.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


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
        "format": "genesis-unilateral-source-audit-0.1", "solver_version": VERSION,
        "purpose": "Read-only frozen-shell unilateral-contact diagnostic under a prescribed pre-existing notch",
        "load_factor_meaning": "Dimensionless multiplier of frozen external dead load only; frozen prestress remains unscaled",
        "path_is_arbitrary_not_ridge": True, "spontaneous_nucleation_tested": False,
        "growth_or_time_advanced": False, "material_reference_updated": False,
        "fracture_toughness_applied": False, "friction_cohesion_or_heat_evolved": False,
        "sources_sha256_before": sources_before, "sources_sha256_after": sources_after,
        "implementation_sha256_before": implementation_before,
        "implementation_sha256_after": implementation_after, "cases": rows,
    }
    draw_report(output, result)
    (output/"source_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# Контакт берегов и приращения нагрузки в сохранённых состояниях", "",
             "Трещина задана заранее вдоль материального пути 0→1. Это не след локализации и не опыт зарождения. "
             "α масштабирует только внешнюю силу; сохранённое напряжение не масштабируется. Геологическое время, "
             "материальная геометрия, трение, когезия и температура не развиваются. Источники не изменены.", "",
             "| Источник | Возраст, млн лет | Активные слабые ячейки | Реакции α=0 → α=1 | Принято шагов | Последняя принятая α | Результат |",
             "|---|---:|---:|---:|---:|---:|---|"]
    for row in rows:
        path = row["continuation"]
        last = path["last_accepted"]
        last_alpha = "—" if last is None else f"{last['load_factor']:.6g}"
        lines.append(f"| {row['label']} | {row['source_time_myr']:.3g} | {row['active_fault_cell_count']} | "
                     f"{path['initial']['active_contact_count']} → {row['trial_target']['active_contact_count']} | "
                     f"{path['accepted_step_count']} | {last_alpha} | {path['stop_reason']} |")
    lines.extend(["", "Контакт предотвращает проникновение берегов, но не ограничивает раскрытие свободной трещины. "
                  "Уменьшение приращений не исправляет нарушение пределов относительно исходной геометрии. "
                  "Нарушение уже при α=0 означает недопустимую релаксацию сохранённого напряжения; "
                  "принятого начального состояния в таком опыте нет.", "",
                  "Числа при α=1 сохранены и для отклонённых состояний исключительно как диагностика; "
                  "это не продолжение принятого физического пути. Энергия отклонённого равновесия не разрешает рост трещины.", "",
                  "Допустимость искусственного большого разреза у источника с 20 кПа не означает самопроизвольное разрушение: "
                  "число активированных слабых плоскостей остаётся нулевым.", "",
                  "Полные невязки, энергетический член контактных реакций, попытки нагружения, контрольные суммы "
                  "и геометрия — в source_audit.json. Для обоих источников на 5120 ячейках NPZ содержит q, зазоры, "
                  "реакции λ, CSR-оператор J и узловую силу Jᵀλ при α=0 и α=1, а также последний принятый шаг, если он есть."])
    (output/"source_audit.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
