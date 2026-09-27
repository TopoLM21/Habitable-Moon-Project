"""Audit finite virtual crack extension in a frozen, dead-loaded elastic shell.

The initial notch, geodesic candidate path, and load are prescribed controls.
This runner computes equilibrium potential differences. It does not advance a
physical front, apply a calibrated fracture toughness, or run geological time.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tectonics.genesis_shell_release import FrozenShell
from analysis.genesis_shell_release_fixture import make_fixture


RADIUS_M = 5.3e6
DEPTH_M = 1e4
YOUNG_PA = 60e9
POISSON_RATIO = .25
TRACTION_PA = 1000.
SUBDIVISIONS = (2, 3, 4, 5)


def _rotation():
    axis = np.array([1., -2., 3.])
    axis /= np.linalg.norm(axis)
    angle = .731
    cross = np.array([[0., -axis[2], axis[1]],
                      [axis[2], 0., -axis[0]],
                      [-axis[1], axis[0], 0.]])
    return np.eye(3)*np.cos(angle)+(1-np.cos(angle))*np.outer(axis, axis)+np.sin(angle)*cross


def _fixture(subdivisions, *, factor=1., rotation=None):
    return make_fixture(subdivisions, rotation=rotation, traction_pa=factor*TRACTION_PA,
                        radius_m=RADIUS_M, depth_m=DEPTH_M)


def _model(fixture):
    return FrozenShell.from_uniform(fixture.mesh, radius_m=RADIUS_M, depth_m=DEPTH_M,
                                    young_pa=YOUNG_PA, poisson_ratio=POISSON_RATIO,
                                    traction_xyz_pa=fixture.traction_xyz_pa)


def _relative_difference(first, second, *, floor=1e-30):
    return float(abs(first-second)/max(abs(first), abs(second), floor))


def _equilibrium_summary(equilibrium):
    return {
        "stored_energy_j": float(equilibrium.stored_energy_j),
        "potential_energy_j": float(equilibrium.potential_energy_j),
        "reduced_potential_j": float(equilibrium.reduced_potential_j),
        "external_potential_work_j": float(equilibrium.external_potential_work_j),
        "equilibrium_residual": float(equilibrium.equilibrium_residual),
        "constraint_residual": float(equilibrium.constraint_residual),
        "max_added_strain": float(equilibrium.max_added_strain),
        "max_motion_edge_fraction": float(equilibrium.max_motion_edge_fraction),
        "min_gap_m": float(equilibrium.min_gap_m),
        "max_gap_m": float(np.max(equilibrium.normal_gap_m, initial=0.)),
        "max_abs_tangential_jump_m": float(np.max(np.abs(equilibrium.tangential_jump_m), initial=0.)),
        "admissible_open_crack": bool(equilibrium.admissible_open_crack),
        "rejection_reasons": list(equilibrium.rejection_reasons),
        "cut_edge_count": len(equilibrium.topology.cut_edges),
        "split_vertex_count": equilibrium.topology.mesh.vertex_count,
    }


def _report_summary(report):
    scalar_names = (
        "release_j", "added_area_m2", "mean_release_j_m2", "potential_difference_j",
        "relaxation_energy_j", "embedding_energy_error_j", "force_pullback_relative_error",
        "stiffness_pullback_relative_error", "gauge_pullback_relative_error",
        "prestress_pullback_relative_error", "material_area_relative_error",
        "material_volume_relative_error", "relative_release_identity_error",
    )
    result = {name: float(getattr(report, name)) for name in scalar_names}
    result.update(before=_equilibrium_summary(report.before), after=_equilibrium_summary(report.after),
                  admissible_open_crack=bool(report.admissible_open_crack),
                  rejection_reasons=list(report.rejection_reasons))
    scale = max(abs(report.release_j), abs(report.potential_difference_j),
                abs(report.relaxation_energy_j), 1.)
    if report.release_j < -1e-9*scale:
        raise AssertionError("Releasing displacement continuity increased the minimized potential")
    if (abs(report.release_j-report.potential_difference_j) > 1e-7*scale
            or abs(report.release_j-report.relaxation_energy_j) > 1e-7*scale):
        raise AssertionError("Potential difference and elastic relaxation energy disagree")
    if not np.isclose(report.release_j, report.mean_release_j_m2*report.added_area_m2, rtol=1e-12, atol=1e-12):
        raise AssertionError("Mean release is not based on the added projected fracture area")
    for name in ("force_pullback_relative_error", "stiffness_pullback_relative_error",
                 "gauge_pullback_relative_error", "prestress_pullback_relative_error",
                 "material_area_relative_error", "material_volume_relative_error"):
        if result[name] > 1e-10:
            raise AssertionError(f"The virtual extension changed inherited material/loading: {name}")
    for equilibrium in (report.before, report.after):
        if equilibrium.equilibrium_residual > 1e-7 or equilibrium.constraint_residual > 1e-7:
            raise AssertionError("The equilibrium solve failed its residual checks")
    return result


def _run_case(subdivisions, *, factor=1., rotation=None, one_edge=False):
    started = time.perf_counter()
    fixture = _fixture(subdivisions, factor=factor, rotation=rotation)
    model = _model(fixture)
    end = fixture.seed_edge_count+(1 if one_edge else fixture.extension_edge_count)
    report = model.compare_extension(fixture.seed_cuts, fixture.cuts(end))
    result = _report_summary(report)
    result.update(cell_count=fixture.mesh.cell_count, subdivisions=subdivisions,
                  traction_factor=factor, extension_kind="one_mesh_edge" if one_edge else "fixed_quarter_base_arc",
                  frozen_snapshot_fingerprint=model.fingerprint,
                  seed_length_m=float(fixture.length_m(fixture.seed_edge_count)),
                  trial_length_m=float(fixture.length_m(end)),
                  extension_length_m=float(fixture.length_m(end)-fixture.length_m(fixture.seed_edge_count)),
                  elapsed_seconds=time.perf_counter()-started)
    geometric_added_area=result["extension_length_m"]*DEPTH_M
    if not np.isclose(report.added_area_m2, geometric_added_area, rtol=2e-12, atol=1e-6):
        raise AssertionError("Trial fracture area differs from the independently prescribed physical path and depth")
    # This independent identity is specific to the fixture with zero inherited
    # elastic strain. Under fixed dead loads the stored energy INCREASES when
    # the crack relaxes the constraint. Using its drop would give a wrong sign.
    stored_increase=report.after.stored_energy_j-report.before.stored_energy_j
    work_increase=report.after.external_potential_work_j-report.before.external_potential_work_j
    result["release_vs_stored_increase_relative_error"]=_relative_difference(
        report.release_j, stored_increase, floor=1.)
    result["external_work_increment_identity_relative_error"]=_relative_difference(
        2*report.release_j, work_increase, floor=1.)
    if max(result["release_vs_stored_increase_relative_error"],
           result["external_work_increment_identity_relative_error"]) > 1e-7:
        raise AssertionError("The unstrained dead-load benchmark violated its energy/work identity")
    print(json.dumps({"case": result["extension_kind"], "cells": fixture.mesh.cell_count,
                      "traction_factor": factor, "release_j": result["release_j"],
                      "mean_release_j_m2": result["mean_release_j_m2"],
                      "admissible": result["admissible_open_crack"],
                      "seconds": result["elapsed_seconds"]}), flush=True)
    return result, fixture, report


def _cartesian_displacement(equilibrium):
    dofs = np.asarray(equilibrium.displacement_m)
    tangent = np.einsum("vij,vj->vi", equilibrium.membrane.vertex_basis,
                        dofs[:-1].reshape(-1, 2))
    return tangent+dofs[-1]*equilibrium.topology.mesh.vertices


def _load_controls(reference):
    controls = []
    for factor in (0., .5, 1., 2., -1.):
        if factor == 1.:
            row = reference
        else:
            row, _, _ = _run_case(3, factor=factor)
        expected=reference["release_j"]*factor**2
        row = dict(row, expected_quadratic_release_j=expected,
                   quadratic_scaling_relative_error=_relative_difference(row["release_j"], expected, floor=1.))
        if row["quadratic_scaling_relative_error"] > 2e-8:
            raise AssertionError("Linear frozen-shell release must scale quadratically with the dead load")
        if factor == 0. and (abs(row["release_j"]) > 1e-12
                             or abs(row["after"]["stored_energy_j"]) > 1e-12):
            raise AssertionError("The unstrained unloaded shell must not supply fracture energy")
        controls.append(row)
    compressed=next(row for row in controls if row["traction_factor"] == -1.)
    if compressed["admissible_open_crack"]:
        raise AssertionError("The compressive, interpenetrating free-bank control was not rejected")
    if not all(row["admissible_open_crack"] for row in controls if row["traction_factor"] >= 0.):
        raise AssertionError("The intended small-strain opening controls were unexpectedly inadmissible")
    return controls


def _rotation_controls(references):
    rotation = _rotation()
    controls = []
    for subdivisions in (2, 3):
        original_row, _, original_report = references[subdivisions]
        row, _, rotated_report = _run_case(subdivisions, rotation=rotation)
        original_xyz = _cartesian_displacement(original_report.after)
        rotated_xyz = _cartesian_displacement(rotated_report.after)@rotation
        displacement_error=float(np.linalg.norm(rotated_xyz-original_xyz)
                                 /max(np.linalg.norm(original_xyz), 1e-30))
        energy_error=_relative_difference(row["release_j"], original_row["release_j"])
        gap_error=float(np.linalg.norm(rotated_report.after.normal_gap_m-original_report.after.normal_gap_m)
                        /max(np.linalg.norm(original_report.after.normal_gap_m), 1e-30))
        if max(displacement_error, energy_error, gap_error) > 1e-7:
            raise AssertionError("Rigidly rotating the mesh, prescribed crack, and load changed the solution")
        if row["after"]["admissible_open_crack"] != original_row["after"]["admissible_open_crack"]:
            raise AssertionError("The admissibility decision changed under a common rigid rotation")
        controls.append({"subdivisions": subdivisions, "cell_count": row["cell_count"],
                         "rotation_matrix": rotation.tolist(), "release_relative_error": energy_error,
                         "cartesian_displacement_relative_error": displacement_error,
                         "normal_gap_relative_error": gap_error})
    return controls


def _save_geometry(output, fixture, report):
    path=output/"finest_equilibria.npz"
    np.savez_compressed(path, reference_vertices_xyz=fixture.mesh.vertices,
                        reference_faces=fixture.mesh.faces,
                        path_vertices=fixture.path_vertices,
                        seed_edge_count=np.array(fixture.seed_edge_count),
                        extension_edge_count=np.array(fixture.extension_edge_count),
                        traction_xyz_pa=fixture.traction_xyz_pa,
                        before_displacement_m=report.before.displacement_m,
                        after_displacement_m=report.after.displacement_m,
                        before_normal_gap_m=report.before.normal_gap_m,
                        after_normal_gap_m=report.after.normal_gap_m,
                        before_tangential_jump_m=report.before.tangential_jump_m,
                        after_tangential_jump_m=report.after.tangential_jump_m,
                        before_cut_edges=report.before.topology.cut_edges,
                        after_cut_edges=report.after.topology.cut_edges,
                        before_parent_vertex=report.before.topology.parent_vertex,
                        after_parent_vertex=report.after.topology.parent_vertex,
                        before_split_faces=report.before.topology.mesh.faces,
                        after_split_faces=report.after.topology.mesh.faces)
    return {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "meaning": "Read-only equilibrium diagnostic, not a front state or a simulation checkpoint"}


def _draw_report(output, report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes=plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    fixed, single=report["fixed_physical_extension"], report["one_edge_extension"]
    counts=[row["cell_count"] for row in fixed]
    ax=axes[0, 0]
    for rows, label, color, marker in (
        (fixed, "Одинаковое физическое продвижение", "#197aa3", "o"),
        (single, "Продвижение на одно ребро (длина уменьшается)", "#bc6535", "s")):
        ax.semilogx(counts, [row["mean_release_j_m2"] for row in rows],
                    marker+"-", color=color, label=label)
    ax.set(xlabel="Число ячеек", ylabel="Среднее высвобождение энергии, Дж/м²",
           title="Чувствительность к сетке и длине продвижения")
    ax.set_xticks(counts, labels=[str(value) for value in counts])
    ax.legend(fontsize=8)

    ax=axes[0, 1]
    for key, label, color in (
        ("relative_release_identity_error", "Потенциал ↔ энергия релаксации", "#197aa3"),
        ("stiffness_pullback_relative_error", "Сохранение упругого оператора", "#bc6535"),
        ("force_pullback_relative_error", "Сохранение заданной силы", "#5b8a54")):
        ax.loglog(counts, [max(row[key], 1e-17) for row in fixed], "o-", color=color, label=label)
    ax.set(xlabel="Число ячеек", ylabel="Относительная невязка (нули показаны на 10⁻¹⁷)",
           title="Проверки энергетики и сохранения материала")
    ax.set_xticks(counts, labels=[str(value) for value in counts])
    ax.legend(fontsize=8)

    ax=axes[1, 0]
    controls=sorted(report["load_controls"], key=lambda row: row["traction_factor"])
    reference=next(row["release_j"] for row in controls if row["traction_factor"] == 1.)
    factors=np.linspace(-1., 2., 200)
    ax.plot(factors, factors**2, "--", color="gray", label="Квадратичное масштабирование")
    for row in controls:
        good=row["admissible_open_crack"]
        ax.scatter(row["traction_factor"], row["release_j"]/reference,
                   marker="o" if good else "x", color="#197aa3" if good else "#b74343", s=65)
    ax.scatter([], [], color="#197aa3", label="Допустимые свободные берега")
    ax.scatter([], [], marker="x", color="#b74343", label="Недопустимое пересечение берегов")
    ax.set(xlabel="Множитель заданной нагрузки", ylabel="Высвобождение / значение при +1",
           title="Энергия сама по себе не разрешает пересечение")
    ax.legend(fontsize=8)

    ax=axes[1, 1]
    x=np.arange(len(controls))
    ax.bar(x-.18, [row["after"]["min_gap_m"] for row in controls], .36,
           label="Минимальное раскрытие", color="#b74343")
    ax.bar(x+.18, [row["after"]["max_gap_m"] for row in controls], .36,
           label="Максимальное раскрытие", color="#197aa3")
    ax.axhline(0., color="black", linewidth=.8)
    ax.set_xticks(x, [f"{row['traction_factor']:g}" for row in controls])
    ax.set(xlabel="Множитель заданной нагрузки", ylabel="Раскрытие берегов, м",
           title="Геометрический контроль на 1280 ячейках")
    ax.legend(fontsize=8)
    for ax in axes.ravel():
        ax.grid(alpha=.2)
    fig.suptitle("Энергия конечного размыкания заданного разлома в оболочке\n"
                 "Замороженная упругость и постоянная сила; без роста фронта и геологического времени", fontsize=14)
    fig.savefig(output/"validation.png", dpi=160)
    plt.close(fig)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args=parser.parse_args(argv)
    output=args.output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("Validation output must be a new or empty directory")
    output.mkdir(parents=True, exist_ok=True)

    references={subdivisions: _run_case(subdivisions) for subdivisions in SUBDIVISIONS}
    fixed=[references[subdivisions][0] for subdivisions in SUBDIVISIONS]
    single=[_run_case(subdivisions, one_edge=True)[0] for subdivisions in SUBDIVISIONS]
    for rows in (fixed, single):
        for index, row in enumerate(rows):
            row["relative_change_from_coarser_mesh"]=(None if index == 0 else
                _relative_difference(row["mean_release_j_m2"], rows[index-1]["mean_release_j_m2"]))
    if not np.allclose([row["extension_length_m"] for row in fixed], fixed[0]["extension_length_m"],
                       rtol=2e-12, atol=1e-7):
        raise AssertionError("The fixed-physical-extension experiment changed crack length with mesh refinement")
    controls=_load_controls(references[3][0])
    rotations=_rotation_controls(references)
    artifact=_save_geometry(output, references[5][1], references[5][2])
    source_names=("tectonics/genesis_shell_release.py", "tectonics/genesis_shell.py",
                  "tectonics/genesis_seams.py", "analysis/genesis_shell_release_fixture.py",
                  "analysis/genesis_shell_release_validation.py")
    report={
        "format": "genesis-shell-release-validation-0.1",
        "interpretation": "Prescribed seeded crack in a frozen elastic spherical membrane under fixed dead load; no propagation",
        "mechanical_specification": {"radius_m": RADIUS_M, "depth_m": DEPTH_M, "young_pa": YOUNG_PA,
                                     "poisson_ratio": POISSON_RATIO, "traction_scale_pa": TRACTION_PA,
                                     "fracture_toughness_j_m2": None,
                                     "fracture_toughness_calibrated": False},
        "source_sha256": {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in source_names},
        "fixed_physical_extension": fixed, "one_edge_extension": single,
        "load_controls": controls, "rotation_controls": rotations,
        "finest_equilibria_artifact": artifact,
        "checks": {"inherited_material_and_load_preserved": True, "variational_release_identity": True,
                   "fixed_deadload_work_identity": True,
                   "projected_fracture_area": True, "zero_load_no_release": True,
                   "quadratic_load_scaling": True, "compression_rejected": True,
                   "rigid_rotation_objectivity": True},
        "limitations": [
            "A finite notch and a geodesic path coincident with mesh edges are prescribed; this is not a nucleation law or an embedded ridge.",
            "Fixed-load potential contains external-force work; its release is not the drop in stored elastic energy alone.",
            "The projected added fracture area counts interface depth times arc length once, not both crack banks twice.",
            "All candidate cuts are perfectly free; cohesive work, friction, unilateral contact, and history transfer are absent.",
            "Positive potential release in an interpenetrating free-bank solution is inadmissible for an opening crack.",
            "No calibrated planetary fracture toughness is assumed and no release is booked as fracture work or heat.",
            "Fixed physical extension is an interval-average energetic quantity; a single mesh edge changes physical extension with refinement.",
            "There is no analytic reference for this spherical-shell experiment; differences between successive meshes are sensitivity, not certified error.",
            "The shell is small-strain, frozen, and quasi-static; no temperature, tidal/orbital evolution, crack speed, or geological time is advanced.",
        ],
    }
    (output/"validation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    _draw_report(output, report)
    print(json.dumps({"output": str(output), "fixed_extension_cases": len(fixed), "one_edge_cases": len(single),
                      "checks": report["checks"], "finest_mean_release_j_m2": fixed[-1]["mean_release_j_m2"],
                      "last_fixed_extension_relative_change": fixed[-1]["relative_change_from_coarser_mesh"]},
                     ensure_ascii=False), flush=True)
    return report


if __name__ == "__main__":
    main()
