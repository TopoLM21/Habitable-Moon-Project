"""Read-only localization audit for a stopped moving-contact strength event.

The original mechanics/checkpoints and reports remain unchanged. Only held
material vertices enter an independent weighted least-norm reconstruction.
This diagnoses a strength event on the prescribed line; it does not release
another contact, select a fracture path, or claim a connected plate network.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.genesis_moving_contact_validation import MovingContactCase
from tectonics.genesis_material import face_frames


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def provenance(directory, report):
    rows = []
    copies = report.get("execution_source_copies", {})
    for kind, entries in (("source", report["source_sha256"]),
                          ("code", report["code_sha256"]),
                          ("artifact", report["artifact_sha256"])):
        for name, expected in entries.items():
            path = (Path(name) if kind == "source" else directory/name
                    if kind == "artifact" else directory/copies[name]
                    if name in copies else ROOT/name)
            actual = digest(path)
            rows.append({"kind": kind, "recorded_name": name,
                "read_path": str(path.resolve()), "sha256": actual,
                "matches_recorded": actual == expected})
    if not all(item["matches_recorded"] for item in rows):
        raise ValueError("Input provenance mismatch; refuse reconstruction using different sources")
    return rows


def independent_recovery(case, state, context):
    model = case.model
    basis = model.basis_for(state)
    geometry, jump = model.geometry_for(state)
    material_vertices = basis.topology.cut_edges.reshape(-1)
    active_indices = (model.free[model.free >= basis.nparent]-basis.nparent)//2
    active_vertices = np.unique(basis.enrichment_vertices[active_indices])
    eligible = (np.isin(material_vertices, basis.enrichment_vertices)
        & ~np.isin(material_vertices, active_vertices))
    selected = np.flatnonzero(eligible)
    if not len(selected):
        raise ValueError("No held material vertices remain observable")
    held_columns = np.setdiff1d(np.arange(basis.nparent, basis.ndof), model.free)
    rows = (2*selected[:, None]+np.arange(2)).ravel()
    local_jump = jump[rows][:, held_columns].toarray()
    areas = np.repeat(geometry.interface_area_m2[selected], 2)
    stress = model.stress(state)
    tensor = np.zeros((len(stress), 2, 2))
    tensor[:, 0, 0], tensor[:, 1, 1] = stress[:, 0], stress[:, 1]
    tensor[:, 0, 1] = tensor[:, 1, 0] = stress[:, 2]
    frames = face_frames(basis.subdivision.mesh)
    world = frames @ tensor @ frames.transpose(0, 2, 1)
    average = np.repeat(world[basis.topology.seam_faces].mean(axis=1), 2, axis=0)
    normal, tangent = geometry.interface_normal, geometry.interface_tangent
    prior = np.column_stack((np.einsum("ti,tij,tj->t", normal, average, normal),
        np.einsum("ti,tij,tj->t", tangent, average, normal)))
    # Independent constrained least-norm solve in area-whitened coordinates;
    # no active-vertex columns or traces enter this matrix.
    matrix = local_jump.T * np.sqrt(areas)[None, :]
    target = state.constraint_reaction_n[held_columns]
    residual = target-local_jump.T@(areas*prior[selected].ravel())
    correction, _, rank, _ = np.linalg.lstsq(matrix, residual, rcond=1e-13)
    traction = prior[selected]+(correction/np.sqrt(areas)).reshape(-1, 2)
    recovered = local_jump.T@(areas*traction.ravel())
    balance_error = float(np.linalg.norm(recovered-target)/max(np.linalg.norm(target), 1.))
    water = np.repeat(context.water_access[basis.topology.seam_faces].mean(axis=1), 2)[selected]
    law = model.law_parameters
    friction = law.friction_dry+(law.friction_wet-law.friction_dry)*water
    cohesion = law.cohesion_pa*(1-(1-law.wet_cohesion_fraction)*water)
    shear_strength = cohesion+friction*np.maximum(-traction[:, 0], 0)
    normal_ratio = np.maximum(traction[:, 0], 0)/law.tensile_strength_pa
    shear_ratio = np.divide(np.abs(traction[:, 1]), shear_strength,
        out=np.zeros(len(selected)), where=shear_strength > 0)
    if np.any((shear_strength <= 0) & (traction[:, 1] != 0)):
        raise ValueError("Undefined finite shear strength ratio in audited state")
    ratios = np.maximum(normal_ratio, shear_ratio)
    index = int(np.argmax(ratios))
    trace = int(selected[index])
    vertex = int(material_vertices[trace])
    support_indices = np.flatnonzero(basis.insertion.path_vertex_ids == vertex)
    if len(support_indices) != 1:
        raise ValueError("Governing material vertex is not unique on the path")
    support_index = int(support_indices[0])
    left, right = model.active_support_indices
    arc = basis.insertion.path_arclength_m
    if support_index in (left, right):
        relation = "existing_material_front"
    elif support_index < left or support_index > right:
        relation = "separate_held_material_vertex"
    else:
        raise ValueError("Held event unexpectedly lies inside the released interval")
    distance = max(float(arc[left]-arc[support_index]),
        float(arc[support_index]-arc[right]), 0.)
    star_left, star_right = support_index-1, support_index+1
    minimal_star_distance = max(float(arc[left]-arc[star_right]),
        float(arc[star_left]-arc[right]), 0.)
    diagnostic = case.held_strength(state, context)
    forces = model.force_diagnostics(state)
    required = (state.last_external_force_n-forces["internal_force_n"]
        -forces["contact_force_n"]-state.last_drag_force_n)[held_columns]
    measured_reaction_error = float(np.linalg.norm(required-target)
        /max(np.linalg.norm(target), 1.))
    by_vertex = {int(v): float(s) for v, s in zip(basis.insertion.path_vertex_ids, arc)}
    trace_arclength = [by_vertex[int(v)] for v in material_vertices]
    return {"governing_trace": trace, "governing_material_vertex_id": vertex,
        "governing_support_index": support_index,
        "governing_arclength_m": float(arc[support_index]),
        "governing_mode": "tensile" if normal_ratio[index] >= shear_ratio[index] else "shear",
        "normal_ratio": float(normal_ratio[index]), "shear_ratio": float(shear_ratio[index]),
        "normal_traction_pa": float(traction[index, 0]),
        "shear_traction_pa": float(traction[index, 1]),
        "shear_strength_pa": float(shear_strength[index]), "water_access": float(water[index]),
        "held_trace_count": len(selected), "held_reaction_rank": int(rank),
        "independent_recovery_relative_residual": balance_error,
        "actual_force_to_saved_reaction_relative_error": measured_reaction_error,
        "owner_strength_ratio_difference": float(ratios[index]-diagnostic["max_held_strength_ratio"]),
        "owner_governing_trace_matches": trace == diagnostic["governing_held_trace"],
        "active_material_vertex_ids": active_vertices.tolist(),
        "current_active_front_support_indices": [left, right],
        "current_active_front_vertex_ids": list(model.active_tip_vertex_ids),
        "current_active_interval_m": [float(arc[left]), float(arc[right])],
        "distance_to_current_active_interval_m": distance,
        "minimal_candidate_star_support_indices": [star_left, star_right],
        "minimal_candidate_star_interval_m": [float(arc[star_left]), float(arc[star_right])],
        "distance_between_candidate_star_and_active_interval_m": minimal_star_distance,
        "event_relation": relation,
        "directly_adjacent_front_event": relation == "existing_material_front",
        "plot_data": {"held_trace_indices": selected.tolist(),
            "trace_arclength_m": trace_arclength,
            "held_normal_ratio": normal_ratio.tolist(), "held_shear_ratio": shear_ratio.tolist(),
            "trace_opening_m": model.jump(state)[:, 0].tolist()},
        "inference_scope": "A reconstructed strength crossing on a prescribed material path; no new birth or propagation has been performed."}


def audit(directory):
    directory = Path(directory).resolve()
    report_path = directory/"validation.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not report.get("next_held_event") or not all(report["checks"].values()):
        raise ValueError("Require a completed validated next-event run")
    verified = provenance(directory, report)
    input_hashes = {str(path): digest(path) for path in
        (report_path, directory/"final_mechanics.npz", directory/"final_thermal.npz")}
    case = MovingContactCase(report["birth_directory"])
    state = case.model.load_state(directory/"final_mechanics.npz")
    context = case.load_context(directory/"final_thermal.npz")
    result = independent_recovery(case, state, context)
    expected_age = case.gate_age_myr+state.elapsed_years/1e6
    if abs(context.thermal.time_myr-expected_age) > 256*np.finfo(float).eps*max(abs(expected_age), 1):
        raise ValueError("Mechanics and thermal checkpoint clocks do not agree")
    result.update(run_directory=str(directory), maximum_step_years=report["maximum_step_years"],
        years_after_first_birth=state.elapsed_years-case.birth_elapsed_years,
        planet_age_myr=context.thermal.time_myr, event_bracket=report["next_held_event"],
        first_contact_max_opening_m=float(case.model.jump(state)[:, 0].max()),
        input_sha256=input_hashes, verified_provenance=verified)
    result["checks"] = {
        "held_only_reconstruction_balances_reaction": result["independent_recovery_relative_residual"] < 1e-10,
        "saved_reaction_matches_actual_forces": result["actual_force_to_saved_reaction_relative_error"] < 1e-10,
        "owner_strength_diagnostic_matches": result["owner_governing_trace_matches"]
            and abs(result["owner_strength_ratio_difference"]) < 1e-10,
        "bracket_trace_matches_lower_checkpoint": result["governing_trace"] == report["next_held_event"]["held_trace"],
        "inputs_unchanged": all(digest(path) == checksum for path, checksum in input_hashes.items())}
    return result


def plot_event(run, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = run["plot_data"]
    x = np.asarray(data["trace_arclength_m"])/1000.
    held = np.asarray(data["held_trace_indices"])
    active = np.asarray(run["current_active_interval_m"])/1000.
    event_x = run["governing_arclength_m"]/1000.
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, constrained_layout=True)
    axes[0].scatter(x[held], data["held_normal_ratio"], s=24, label="Растяжение / прочность")
    axes[0].scatter(x[held], data["held_shear_ratio"], s=24, marker="x", label="Сдвиг / прочность")
    axes[0].axhline(1., color="firebrick", linestyle="--", linewidth=1., label="Порог")
    axes[0].set(ylabel="Отношение к прочности", ylim=(0., 1.12))
    axes[0].legend(loc="best", fontsize=9)
    axes[1].scatter(x, 1000*np.asarray(data["trace_opening_m"]), s=24, color="teal")
    axes[1].set(xlabel="Расстояние вдоль заданной линии, км", ylabel="Раскрытие, мм")
    for axis in axes:
        axis.axvspan(*active, color="teal", alpha=.12)
        axis.axvline(event_x, color="firebrick", linestyle=":", linewidth=1.5)
        axis.grid(alpha=.25)
    axes[0].annotate("Следующее наблюдаемое событие",
        xy=(event_x, max(run["normal_ratio"], run["shear_ratio"])),
        xytext=(8, -30), textcoords="offset points", fontsize=9,
        arrowprops={"arrowstyle": "->", "color": "firebrick"})
    fig.suptitle("Один раскрывающийся контакт и следующий порог на заданной линии\n"
        "Затенена уже активная область; это ещё не сеть разломов планеты")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, action="append")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Audit output must be new or empty")
    runs = [audit(path) for path in args.run]
    comparison = None
    if len(runs) == 2:
        coarse, fine = sorted(runs, key=lambda item: item["maximum_step_years"], reverse=True)
        comparison = {"event_time_difference_years": fine["years_after_first_birth"]-coarse["years_after_first_birth"],
            "same_material_event": all(coarse[name] == fine[name] for name in
                ("governing_trace", "governing_material_vertex_id", "governing_support_index", "governing_mode", "event_relation")),
            "current_arclength_distance_difference_m": fine["distance_to_current_active_interval_m"]
                -coarse["distance_to_current_active_interval_m"]}
    report = {"scope": __doc__, "audit_source_sha256": {str(Path(__file__).resolve()): digest(__file__)},
        "runs": runs, "comparison": comparison,
        "checks_passed": all(all(item["checks"].values()) for item in runs)}
    output.mkdir(parents=True, exist_ok=True)
    plot_event(max(runs, key=lambda item: item["maximum_step_years"]), output/"event_location.png")
    report["artifact_sha256"] = {"event_location.png": digest(output/"event_location.png")}
    (output/"audit.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"runs": [{k:v for k,v in item.items() if k not in
        ("verified_provenance", "input_sha256", "plot_data")} for item in runs], "comparison": comparison,
        "checks_passed": report["checks_passed"]}, indent=2))
    if not report["checks_passed"]:
        raise SystemExit("Event audit failed; inspect audit.json")


if __name__ == "__main__":
    main()
