"""Frozen spherical-shell controls for frictionless unilateral crack banks.

The crack and its extension are prescribed. Loading fractions are numerical
continuation parameters, not years. No fracture toughness, heat, irreversible
front propagation, or production checkpoint is introduced by this audit.
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

from analysis.genesis_shell_release_fixture import make_fixture, opening_traction
from tectonics.genesis_shell_release import FrozenShell
from tectonics.genesis_unilateral import UnilateralShell, LoadPathParameters


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
    return (np.eye(3)*np.cos(angle)+(1-np.cos(angle))*np.outer(axis, axis)
            + np.sin(angle)*cross)


def _fixture(subdivisions, *, rotation=None):
    return make_fixture(subdivisions, rotation=rotation, traction_pa=TRACTION_PA,
                        radius_m=RADIUS_M, depth_m=DEPTH_M)


def _traction(fixture, kind, factor=1.):
    """A fixed quadratic spherical potential field, sampled at material vertices.

    The mixed field superposes compression and an off-diagonal quadratic field.
    Its normal/shear components decouple by reflection symmetry in this fixture.
    Every axis rotates with the mesh; no fixed world-axis load is introduced.
    """
    normal = fixture.normal_xyz
    midpoint = fixture.mesh.vertices[fixture.path_vertices[fixture.seed_edge_count]]
    tangent = np.cross(normal, midpoint)
    plus = (normal+tangent)/np.sqrt(2.)
    minus = (normal-tangent)/np.sqrt(2.)
    opening = fixture.traction_xyz_pa
    shear = (opening_traction(fixture.mesh.vertices, plus, TRACTION_PA)
             - opening_traction(fixture.mesh.vertices, minus, TRACTION_PA))
    if kind == "opening":
        traction = opening
    elif kind == "compression":
        traction = -opening
    elif kind == "mixed":
        traction = -opening+.5*shear
    elif kind == "shear":
        traction = .5*shear
    else:
        raise ValueError(f"Unknown loading control {kind!r}")
    return float(factor)*traction


def _model(fixture, kind, factor=1.):
    return FrozenShell.from_uniform(fixture.mesh, radius_m=RADIUS_M, depth_m=DEPTH_M,
                                    young_pa=YOUNG_PA, poisson_ratio=POISSON_RATIO,
                                    traction_xyz_pa=_traction(fixture, kind, factor))


def _relative(first, second, floor=1.):
    return float(abs(first-second)/max(abs(first), abs(second), floor))


def _cartesian(equilibrium):
    q = equilibrium.displacement_m
    tangent = np.einsum("vij,vj->vi", equilibrium.membrane.vertex_basis,
                        q[:-1].reshape(-1, 2))
    return tangent+q[-1]*equilibrium.topology.mesh.vertices


def _cartesian_contact_force(equilibrium):
    force = equilibrium.normal_operator.T@equilibrium.normal_reaction_n
    return np.einsum("vij,vj->vi", equilibrium.membrane.vertex_basis,
                     force[:-1].reshape(-1, 2))


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _equilibrium_summary(equilibrium):
    result = {name: float(getattr(equilibrium, name)) for name in (
        "stored_energy_j", "potential_energy_j", "reduced_potential_j",
        "external_potential_work_j", "equilibrium_residual", "constraint_residual",
        "max_added_strain", "max_motion_edge_fraction", "min_gap_m",
        "complementarity_relative_error", "contact_work_j", "load_factor")}
    result.update(max_gap_m=float(np.max(equilibrium.normal_gap_m, initial=0.)),
                  max_abs_tangential_jump_m=float(np.max(np.abs(equilibrium.tangential_jump_m), initial=0.)),
                  max_normal_reaction_n=float(np.max(equilibrium.normal_reaction_n, initial=0.)),
                  min_normal_reaction_n=float(np.min(equilibrium.normal_reaction_n, initial=0.)),
                  active_contact_count=int(equilibrium.active_contact_count),
                  dual_rank=int(equilibrium.dual_rank),
                  admissible_contact=bool(equilibrium.admissible_contact),
                  rejection_reasons=list(equilibrium.rejection_reasons))
    scale = max(abs(equilibrium.stored_energy_j), abs(equilibrium.reduced_potential_j), 1.)
    if (equilibrium.min_gap_m < -1e-6
            or equilibrium.equilibrium_residual > 1e-7
            or equilibrium.constraint_residual > 1e-7
            or equilibrium.complementarity_relative_error > 1e-7
            or abs(equilibrium.contact_work_j) > 1e-7*scale):
        raise AssertionError("Unilateral equilibrium failed feasibility, stationarity or contact-work controls")
    reaction_scale = max(result["max_normal_reaction_n"], 1.)
    if result["min_normal_reaction_n"] < -1e-8*reaction_scale:
        raise AssertionError("The unilateral normal reaction became tensile")
    return result


def _run_case(subdivisions, kind, *, factor=1., rotation=None):
    started = time.perf_counter()
    fixture = _fixture(subdivisions, rotation=rotation)
    base = _model(fixture, kind, factor)
    free = base.compare_extension(fixture.seed_cuts, fixture.trial_cuts)
    report = UnilateralShell(base).compare_extension(fixture.seed_cuts, fixture.trial_cuts)
    result = {name: float(getattr(report, name)) for name in (
        "release_j", "added_area_m2", "mean_release_j_m2", "potential_difference_j",
        "relaxation_energy_j", "contact_release_term_j", "lifted_min_gap_m",
        "embedding_energy_error_j", "force_pullback_relative_error",
        "stiffness_pullback_relative_error", "gauge_pullback_relative_error",
        "prestress_pullback_relative_error", "material_area_relative_error",
        "material_volume_relative_error", "relative_release_identity_error",
        "energy_roundoff_bound_j")}
    result.update(before=_equilibrium_summary(report.before), after=_equilibrium_summary(report.after),
                  cell_count=fixture.mesh.cell_count, subdivisions=subdivisions,
                  load_kind=kind, traction_factor=factor,
                  released_energy_resolved=bool(report.released_energy_resolved),
                  frozen_snapshot_fingerprint=base.fingerprint,
                  free_release_j=float(free.release_j),
                  free_mean_release_j_m2=float(free.mean_release_j_m2),
                  free_min_gap_m=float(free.after.min_gap_m),
                  seed_length_m=fixture.length_m(fixture.seed_edge_count),
                  trial_length_m=fixture.length_m(fixture.seed_edge_count+fixture.extension_edge_count),
                  elapsed_seconds=time.perf_counter()-started)
    result["extension_length_m"] = result["trial_length_m"]-result["seed_length_m"]
    if not np.isclose(report.added_area_m2, result["extension_length_m"]*DEPTH_M, rtol=2e-12):
        raise AssertionError("Projected fracture area differs from the fixed physical extension")
    energy_scale = max(abs(free.release_j), abs(report.release_j), 1.)
    if report.release_j < -report.energy_roundoff_bound_j:
        raise AssertionError("Relaxing tangential continuity increased the minimized potential")
    if abs(report.release_j-report.potential_difference_j) > 1e-7*energy_scale:
        raise AssertionError("Contact release differs from the complete potential difference")
    if abs(report.release_j-report.relaxation_energy_j-report.contact_release_term_j) > 1e-7*energy_scale:
        raise AssertionError("Contact reaction contribution is missing from the release identity")
    if not (report.before.admissible_contact and report.after.admissible_contact):
        raise AssertionError("The small prescribed control unexpectedly violated a reference-geometry guard")
    if kind == "opening":
        result["free_displacement_relative_difference"] = float(
            np.linalg.norm(report.after.displacement_m-free.after.displacement_m)
            / max(np.linalg.norm(free.after.displacement_m), 1.))
        result["free_release_relative_difference"] = _relative(report.release_j, free.release_j)
        if max(result["free_displacement_relative_difference"],
               result["free_release_relative_difference"]) > 1e-7:
            raise AssertionError("Inactive contact changed the freely opening solution")
    if kind == "compression":
        intact = base.solve([])
        parent = report.after.topology.parent_vertex
        lifted = np.r_[intact.displacement_m[:-1].reshape(-1, 2)[parent].ravel(),
                       intact.displacement_m[-1]]
        result["intact_displacement_relative_difference"] = float(
            np.linalg.norm(report.after.displacement_m-lifted)/max(np.linalg.norm(lifted), 1.))
        result["release_relative_to_free_invalid_control"] = abs(report.release_j)/energy_scale
        if (result["intact_displacement_relative_difference"] > 1e-7
                or abs(report.release_j) > report.energy_roundoff_bound_j):
            raise AssertionError("Pure symmetric compression should recover the intact normal solution")
        if free.after.min_gap_m >= -1e-3:
            raise AssertionError("The penetration negative control did not activate")
    print(json.dumps({"control": kind, "cells": fixture.mesh.cell_count,
                      "release_j": report.release_j, "min_gap_m": report.after.min_gap_m,
                      "active_contacts": report.after.active_contact_count,
                      "seconds": result["elapsed_seconds"]}), flush=True)
    return result, fixture, report


def _rotation_control(original):
    row, fixture, report = _run_case(3, "mixed", rotation=_rotation())
    original_row, _, original_report = original
    xyz = _cartesian(report.after)@_rotation()
    original_xyz = _cartesian(original_report.after)
    displacement_error = float(np.linalg.norm(xyz-original_xyz)/np.linalg.norm(original_xyz))
    energy_error = _relative(row["release_j"], original_row["release_j"])
    # Coincident endpoint constraints admit multiple dual multiplier vectors.
    # Objectivity applies to their physical assembled force, not to an arbitrary
    # allocation between duplicate rows of the contact operator.
    original_force = _cartesian_contact_force(original_report.after)
    rotated_force = _cartesian_contact_force(report.after)@_rotation()
    reaction_error = float(np.linalg.norm(rotated_force-original_force)
                           / max(np.linalg.norm(original_force), 1.))
    if max(displacement_error, energy_error, reaction_error) > 1e-7:
        raise AssertionError("Rigidly rotating the crack, mesh and load changed the contact solution")
    return {"cell_count": fixture.mesh.cell_count, "rotation_matrix": _rotation().tolist(),
            "cartesian_displacement_relative_error": displacement_error,
            "release_relative_error": energy_error, "assembled_contact_force_relative_error": reaction_error}


def _continuation_control():
    fixture = _fixture(3)
    base = _model(fixture, "opening", factor=3000.)
    contact = UnilateralShell(base)
    limits = LoadPathParameters(max_load_increment=.125, min_load_increment=1e-6,
                                max_incremental_strain=.0005,
                                max_incremental_motion_edge_fraction=.001)
    direct = contact.solve(fixture.trial_cuts, load_factor=1.)
    ramp = contact.continue_loading(fixture.trial_cuts, target_load_factor=1., parameters=limits)
    if direct.admissible_contact or ramp.reached_target:
        raise AssertionError("Subdividing the load bypassed a total reference-geometry guard")
    if not ramp.last_accepted.admissible_contact:
        raise AssertionError("The continuation committed an inadmissible state")
    accepted = [attempt for attempt in ramp.attempts if attempt.accepted]
    if not accepted:
        raise AssertionError("Continuation did not resolve any admissible intermediate equilibrium")
    accepted_factor = float(ramp.last_accepted.load_factor)
    direct_accepted = contact.solve(fixture.trial_cuts, load_factor=accepted_factor)
    agreement = float(np.linalg.norm(direct_accepted.displacement_m-ramp.last_accepted.displacement_m)
                      / max(np.linalg.norm(direct_accepted.displacement_m), 1.))
    if agreement > 1e-10:
        raise AssertionError("Load continuation reset reference displacement or changed the endpoint equilibrium")
    small = contact.continue_loading(fixture.trial_cuts, target_load_factor=.1, parameters=limits)
    if not small.reached_target or not small.last_accepted.admissible_contact:
        raise AssertionError("The admissible bounded load path did not reach its endpoint")
    attempts = [{"load_factor": float(item.load_factor), "accepted": bool(item.accepted),
                 "reason": item.reason, "increment_strain": float(item.increment_strain),
                 "increment_motion_edge_fraction": float(item.increment_motion_edge_fraction),
                 "max_added_strain": float(item.total_strain),
                 "max_motion_edge_fraction": float(item.total_motion_edge_fraction)} for item in ramp.attempts]
    return {"cell_count": fixture.mesh.cell_count, "traction_multiplier": 3000.,
            "requested_factor": 1., "last_accepted_factor": accepted_factor,
            "reached_target": bool(ramp.reached_target), "stop_reason": ramp.stop_reason,
            "direct_rejection_reasons": list(direct.rejection_reasons),
            "direct_max_added_strain": float(direct.max_added_strain),
            "direct_max_motion_edge_fraction": float(direct.max_motion_edge_fraction),
            "max_strain": float(base.max_strain),
            "max_motion_edge_fraction": float(base.max_motion_edge_fraction),
            "max_incremental_strain": float(limits.max_incremental_strain),
            "max_incremental_motion_edge_fraction": float(limits.max_incremental_motion_edge_fraction),
            "accepted_endpoint_displacement_relative_error": agreement,
            "small_target_factor": .1, "small_target_reached": True,
            "attempts": attempts, "accepted_steps": [row for row in attempts if row["accepted"]]}


def _save_geometry(output, fixture, contact_report, name):
    path = output/f"{name}.npz"
    np.savez_compressed(path, reference_vertices_xyz=fixture.mesh.vertices,
                        reference_faces=fixture.mesh.faces, path_vertices=fixture.path_vertices,
                        before_displacement_m=contact_report.before.displacement_m,
                        after_displacement_m=contact_report.after.displacement_m,
                        before_external_force=contact_report.before.external_force,
                        after_external_force=contact_report.after.external_force,
                        before_normal_gap_m=contact_report.before.normal_gap_m,
                        after_normal_gap_m=contact_report.after.normal_gap_m,
                        before_tangential_jump_m=contact_report.before.tangential_jump_m,
                        after_tangential_jump_m=contact_report.after.tangential_jump_m,
                        before_normal_reaction_n=contact_report.before.normal_reaction_n,
                        after_normal_reaction_n=contact_report.after.normal_reaction_n,
                        before_cut_edges=contact_report.before.topology.cut_edges,
                        after_cut_edges=contact_report.after.topology.cut_edges,
                        before_split_faces=contact_report.before.topology.mesh.faces,
                        after_split_faces=contact_report.after.topology.mesh.faces,
                        before_parent_vertex=contact_report.before.topology.parent_vertex,
                        after_parent_vertex=contact_report.after.topology.parent_vertex)
    return {"file": path.name, "sha256": _sha256(path),
            "meaning": "Frozen equilibrium diagnostic; not a front state or production checkpoint"}


def _draw_report(output, report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    opening = report["opening_mesh_controls"]
    compressed = report["compression_mesh_controls"]
    counts = [row["cell_count"] for row in opening]
    ax = axes[0, 0]
    ax.semilogx(counts, [row["mean_release_j_m2"] for row in opening], "o-",
                color="#197aa3", label="Растяжение, односторонний контакт")
    ax.semilogx(counts, [row["free_mean_release_j_m2"] for row in opening], "x--",
                color="#bd7834", label="Растяжение, свободные берега")
    ax.set(xlabel="Число ячеек", ylabel="Высвобождение энергии, Дж/м²",
           title="Одинаковое заданное удлинение разреза")
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    ax.semilogx(counts, [row["free_min_gap_m"] for row in compressed], "o-",
                color="#b74343", label="Свободные берега")
    ax.semilogx(counts, [row["after"]["min_gap_m"] for row in compressed], "o-",
                color="#218666", label="Односторонний контакт")
    ax.axhline(0., color="black", linewidth=.8)
    ax.set(xlabel="Число ячеек", ylabel="Минимальное раскрытие, м",
           title="Сжатие: контакт исключает проникновение")
    ax.legend(fontsize=8)
    ax = axes[1, 0]
    controls = [opening[1], compressed[1], report["mixed_control"]]
    x = np.arange(len(controls))
    ax.bar(x, [row["release_j"] for row in controls], color=["#197aa3", "#b74343", "#9465a5"])
    ax.set_xticks(x, ["Растяжение", "Сжатие", "Сжатие + сдвиг"])
    ax.set(ylabel="Высвобождение энергии, Дж", title="1280 ячеек; контакт без трения")
    ax = axes[1, 1]
    continuation = report["continuation_control"]
    steps = continuation["accepted_steps"]
    utilization = [max(row["max_motion_edge_fraction"]/continuation["max_motion_edge_fraction"],
                       row["max_added_strain"]/continuation["max_strain"]) for row in steps]
    direct_utilization = max(continuation["direct_max_motion_edge_fraction"]/continuation["max_motion_edge_fraction"],
                             continuation["direct_max_added_strain"]/continuation["max_strain"])
    ax.plot([row["load_factor"] for row in steps], utilization, "o-", color="#218666")
    ax.axhline(1., color="#b74343", linestyle="--", label="Предел общей деформации / смещения")
    ax.scatter([continuation["requested_factor"]], [direct_utilization],
               marker="x", s=70, color="#b74343", label="Недопустимый конечный уровень")
    ax.set(xlabel="Численный множитель нагрузки", ylabel="Доля допустимого общего изменения",
           title="Малые приращения не обходят общий предел")
    ax.legend(fontsize=8)
    for ax in axes.ravel():
        ax.grid(alpha=.2)
    fig.suptitle("Контакт берегов в замороженной упругой оболочке\n"
                 "Заданный разрез и нагрузка; без роста фронта и геологического времени", fontsize=14)
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
    opening = {sub: _run_case(sub, "opening") for sub in SUBDIVISIONS}
    compression = {sub: _run_case(sub, "compression") for sub in SUBDIVISIONS}
    mixed = _run_case(3, "mixed")
    shear = _run_case(3, "shear")
    mixed_row, _, mixed_report = mixed
    shear_error = _relative(mixed_row["release_j"], shear[0]["release_j"])
    if mixed_row["release_j"] <= 0 or shear_error > 1e-7:
        raise AssertionError("Frictionless mixed compression should retain the positive shear relaxation")
    mixed_row["pure_shear_release_relative_difference"] = shear_error
    mixed_row["pure_shear_release_j"] = shear[0]["release_j"]
    if (mixed_report.after.active_contact_count == 0
            or mixed_row["after"]["max_abs_tangential_jump_m"] <= .01):
        raise AssertionError("The mixed control did not engage contact together with tangential slip")
    scaled = _run_case(3, "mixed", factor=.5)
    scaling_error = _relative(scaled[0]["release_j"], .25*mixed_row["release_j"])
    if scaling_error > 1e-7:
        raise AssertionError("Frictionless contact on a zero-gap cone must preserve quadratic load scaling")
    for table in (opening, compression):
        rows = [table[sub][0] for sub in SUBDIVISIONS]
        for index, row in enumerate(rows):
            row["relative_change_from_coarser_mesh"] = (None if index == 0 or not row["released_energy_resolved"] else
                _relative(row["mean_release_j_m2"], rows[index-1]["mean_release_j_m2"]))
        if not np.allclose([row["extension_length_m"] for row in rows], rows[0]["extension_length_m"],
                           rtol=2e-12, atol=1e-7):
            raise AssertionError("Mesh refinement changed the physical crack extension")
    rotation = _rotation_control(mixed)
    continuation = _continuation_control()
    artifacts = [_save_geometry(output, opening[5][1], opening[5][2], "opening_finest"),
                 _save_geometry(output, compression[5][1], compression[5][2], "compression_finest"),
                 _save_geometry(output, mixed[1], mixed[2], "mixed_1280")]
    sources = ("tectonics/genesis_unilateral.py", "tectonics/genesis_shell_release.py",
               "tectonics/genesis_shell.py", "tectonics/genesis_seams.py",
               "analysis/genesis_shell_release_fixture.py", "analysis/genesis_unilateral_validation.py")
    report = {
        "format": "genesis-unilateral-validation-0.1",
        "interpretation": "Frozen frictionless Signorini contact on a prescribed edge crack; no front advancement",
        "mechanical_specification": {"radius_m": RADIUS_M, "depth_m": DEPTH_M,
                                     "young_pa": YOUNG_PA, "poisson_ratio": POISSON_RATIO,
                                     "traction_scale_pa": TRACTION_PA, "mixed_shear_fraction": .5},
        "source_sha256": {name: _sha256(ROOT/name) for name in sources},
        "opening_mesh_controls": [opening[sub][0] for sub in SUBDIVISIONS],
        "compression_mesh_controls": [compression[sub][0] for sub in SUBDIVISIONS],
        "mixed_control": mixed_row,
        "quadratic_scaling_control": {"factor": .5, "relative_error": scaling_error,
                                       "scaled_release_j": scaled[0]["release_j"]},
        "rotation_control": rotation, "continuation_control": continuation,
        "equilibria_artifacts": artifacts,
        "checks": {"free_opening_preserved": True, "no_interpenetration": True,
                   "compressive_normal_reactions": True, "complementarity": True,
                   "contact_work_zero": True, "pure_normal_compression_recovers_intact": True,
                   "mixed_compression_releases_shear_energy": True, "quadratic_load_scaling": True,
                   "rigid_rotation_objectivity": True, "total_geometry_guard_not_reset": True,
                   "admissible_load_path_reaches_target": True},
        "limitations": [
            "The notch and nested edge-aligned extension are prescribed; neither nucleation nor ridge insertion is solved.",
            "Zero initial gap, frictionless, nonadhesive, reversible contact is assumed; water pressure, cohesion and friction history are absent.",
            "Duplicate endpoint constraints can have nonunique individual multipliers; the assembled contact force is the physical quantity checked under rotation.",
            "Closed banks can still release tangential elastic energy without friction; zero release is asserted only for this pure-normal symmetric control.",
            "Fixed dead-load potential includes external-force work; release is not the drop in stored energy alone.",
            "No fracture toughness is calibrated, no fracture/heat energy is booked, and no physical crack-front decision is made.",
            "Load fractions and substeps are numerical continuation, not elapsed geological time or a speed law.",
            "Total reference strain/motion limits are retained; accepted small increments do not make a large final displacement physically valid.",
            "Contact normals, pairing and material geometry remain fixed; large sliding, new collisions and finite rotations are unsupported.",
            "The spherical benchmark has no analytical absolute release value; successive-mesh differences measure sensitivity, not certified physical error.",
        ],
    }
    (output/"validation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    _draw_report(output, report)
    print(json.dumps({"output": str(output), "checks": report["checks"],
                      "finest_opening_mean_release_j_m2": opening[5][0]["mean_release_j_m2"],
                      "last_mesh_relative_change": opening[5][0]["relative_change_from_coarser_mesh"],
                      "mixed_release_j": mixed_row["release_j"],
                      "large_load_stop_reason": continuation["stop_reason"]}, ensure_ascii=False), flush=True)
    return report


if __name__ == "__main__":
    main()
