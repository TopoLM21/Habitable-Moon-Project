"""Independent file readback of the actual Genesis bulk-birth control.

Only the Python standard library and NumPy are used.  No producer validator,
physics module, checkpoint loader, or simulation is imported or executed.
"""
from __future__ import annotations

import argparse
import base64
from bisect import bisect_right
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
FORMAT = "genesis-bulk-birth-independent-file-readback-0.1"


def require(condition, description):
    if not condition:
        raise ValueError(description)


def canonical(value, *, ascii=True):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=ascii, allow_nan=False).encode()


def digest(value, *, ascii=True):
    return hashlib.sha256(canonical(value, ascii=ascii)).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files if name != "metadata"}
        metadata = json.loads(str(archive["metadata"]))
    return metadata, arrays


def same_json(a, b, label):
    require(canonical(a) == canonical(b), label)


def same_array(a, b, label):
    left, right = np.asarray(a), np.asarray(b)
    require(left.shape == right.shape and np.array_equal(left, right), label)


def close(a, b, label, *, rtol=2e-13, atol=0.):
    require(np.allclose(a, b, rtol=rtol, atol=atol), label)


def face_geometry(vertices, faces, radius_km):
    vertices, faces = np.asarray(vertices), np.asarray(faces)
    require(vertices.ndim == 2 and vertices.shape[1] == 3, "Vertex geometry is not three dimensional")
    close(np.linalg.norm(vertices, axis=1), 1., "Material vertices are not unit directions", rtol=0, atol=2e-12)
    require(faces.ndim == 2 and faces.shape[1] == 3 and faces.dtype.kind in "iu",
            "Material faces are not ordered integer triangles")
    require(np.all((faces >= 0) & (faces < len(vertices))), "A material face references an absent vertex")
    triangles = vertices[faces]
    a, b, c = triangles.transpose(1, 0, 2)
    determinant = np.einsum("fi,fi->f", a, np.cross(b, c))
    require(np.all(determinant > 0), "A material face is inverted")
    area = 2*np.arctan2(determinant,
        1+np.einsum("fi,fi->f", a, b)+np.einsum("fi,fi->f", b, c)+np.einsum("fi,fi->f", c, a))
    close(math.fsum(area), 4*math.pi, "Material mesh does not cover the sphere", rtol=5e-11)
    first, second = b-a, c-a
    normal = np.cross(first, second)
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    e1 = first/np.linalg.norm(first, axis=1)[:, None]
    basis = np.stack((e1, np.cross(normal, e1), normal), axis=2)
    edges = np.stack((first, second), axis=2)*(1000*radius_km)
    metric = edges.transpose(0, 2, 1)@edges
    return area, basis, metric


def replay_front(mass, intervals):
    """Split growth by sorted cuts; retire only extant deepest live slices.

    This event-table algorithm is independent of the producer's per-layer
    growth iterator and its replay validator.  Retired rows remain in the
    table; re-freezing appends new identities instead of extending old rows.
    """
    mass = np.asarray(mass, dtype=float)
    require(mass.ndim == 2 and np.isfinite(mass).all() and np.all(mass > 0),
            "Fixed layer masses are not positive finite values")
    count, layers = mass.shape
    totals = [math.fsum(row) for row in mass]
    boundaries = [[0.]+[math.fsum(row[:k])/totals[face] for k in range(1, layers)]+[1.]
                  for face, row in enumerate(mass)]
    require(all(all(a < b for a, b in zip(row, row[1:])) for row in boundaries),
            "Fixed mass layer boundaries are unresolved")
    front, clock, table, live_by_face = [0.]*count, 0., [], [[] for _ in range(count)]
    for interval in intervals:
        require(interval["before_time_myr"] == clock
                and interval["before_front_fraction"] == front,
                "Accepted solid-front intervals have a clock or material gap")
        end, next_front = interval["after_time_myr"], interval["after_front_fraction"]
        require(type(end) is float and math.isfinite(end) and end > clock,
                "Accepted solid-front interval fails to advance")
        require(len(next_front) == count and all(type(x) is float and math.isfinite(x)
                and 0 <= x <= 1 for x in next_front), "Accepted front vector is invalid")
        for face in range(count):
            lower, upper = front[face], next_front[face]
            if upper < lower:
                survivors = []
                for index in live_by_face[face]:
                    row = table[index]
                    if row["q_lo"] >= upper:
                        row["live_q_hi"] = row["q_lo"]
                    elif row["live_q_hi"] > upper:
                        row["live_q_hi"] = upper
                    if row["live_q_hi"] > row["q_lo"]:
                        survivors.append(index)
                live_by_face[face] = survivors
            elif upper > lower:
                cuts = [lower]+[x for x in boundaries[face] if lower < x < upper]+[upper]
                for lo, hi in zip(cuts, cuts[1:]):
                    layer = bisect_right(boundaries[face], lo)-1
                    table.append({"id": len(table)+1, "face_index": face, "layer_index": layer,
                        "q_lo": lo, "q_hi": hi, "live_q_hi": hi,
                        "birth_before_time_myr": clock, "birth_after_time_myr": end})
                    live_by_face[face].append(len(table)-1)
        front, clock = list(next_front), end
    formed, live = np.zeros_like(mass), np.zeros_like(mass)
    for row in table:
        face, layer = row["face_index"], row["layer_index"]
        formed[face, layer] += totals[face]*(row["q_hi"]-row["q_lo"])
        live[face, layer] += totals[face]*(row["live_q_hi"]-row["q_lo"])
    for face, indices in enumerate(live_by_face):
        ordered = sorted((table[i]["q_lo"], table[i]["live_q_hi"]) for i in indices)
        endpoints = [0.]+[hi for _, hi in ordered]
        require(all(lo == endpoints[index] for index, (lo, _) in enumerate(ordered))
                and endpoints[-1] == front[face], "Independent live material has a gap or overlap")
    close(live.sum(axis=1), np.asarray(totals)*front,
          "Independent live masses differ from accepted front inventory")
    return {"table": table, "time_myr": clock, "front_fraction": front,
            "formed_mass_by_face_layer_kg": formed,
            "live_mass_by_face_layer_kg": live,
            "remelted_mass_by_face_layer_kg": formed-live}


def verify_history(history, *, exact_birth_geometry=False):
    require(history["format"] == "genesis-bulk-birth-0.1"
            and history["front_closure"] == "surface_connected_solidus_column_mass_coordinate"
            and history["time_closure"] == "accepted_endpoint_birth_bracket"
            and history["geometry_closure"] == "thermal_first_old_mechanical_geometry"
            and history["remelt_closure"] == "retire_deepest_reference_mass_overlap",
            "The exported bulk-birth model changed")
    body = {key: value for key, value in history.items() if key != "sha256"}
    require(digest(body) == history["sha256"], "Bulk history seal changed")
    context = {key: history[key] for key in ("configuration_json", "initial_vertices",
        "initial_faces", "initial_radius_km", "solidus_k", "layer_mass_kg")}
    context.update(format=history["format"], front_closure=history["front_closure"])
    require(digest(context) == history["source_identity"], "Material birth source identity changed")
    configuration = json.loads(history["configuration_json"])
    require(canonical(configuration).decode() == history["configuration_json"],
            "Birth-source configuration is not its complete canonical record")
    _, initial_basis, initial_metric = face_geometry(history["initial_vertices"],
        history["initial_faces"], history["initial_radius_km"])
    replay = replay_front(history["layer_mass_kg"], history["intervals"])
    require(replay["time_myr"] == history["time_myr"]
            and replay["front_fraction"] == history["front_fraction"],
            "Bulk history fails to reach the current material endpoint")
    require(len(replay["table"]) == len(history["cohorts"]),
            "Actual cohort count differs from independent accepted-front replay")
    for expected, cohort in zip(replay["table"], history["cohorts"]):
        for name, value in expected.items():
            require(type(cohort[name]) is type(value) and cohort[name] == value,
                    "Cohort identity, birth bracket or live mass interval differs: "+name)
        require(cohort["birth_temperature_k"] == history["solidus_k"],
                "A cohort changed its solidus birth boundary law")
        metric, basis = np.asarray(cohort["birth_metric_m2"]), np.asarray(cohort["birth_frame_xyz"])
        require(metric.shape == (2, 2) and np.array_equal(metric, metric.T)
                and np.all(np.linalg.eigvalsh(metric) > 0), "A birth chart metric is not SPD")
        close(basis.T@basis, np.eye(3), "A birth frame is not orthonormal", rtol=0, atol=2e-12)
        close(np.linalg.det(basis), 1., "A birth frame is not proper", rtol=0, atol=2e-12)
        if exact_birth_geometry:
            face = cohort["face_index"]
            require(cohort["birth_radius_km"] == history["initial_radius_km"],
                    "Thermal precursor birth radius changed before mechanical activation")
            close(metric, initial_metric[face], "Recorded precursor birth metric differs from actual geometry",
                  rtol=2e-15)
            same_array(basis, initial_basis[face], "Recorded precursor birth frame differs from actual geometry")
    replay["source_identity"] = history["source_identity"]
    return replay


def verify_prefix(before, after):
    require(before["source_identity"] == after["source_identity"], "Continuation replaced original material identity")
    source_count, source_cohorts = len(before["intervals"]), len(before["cohorts"])
    same_json(before["intervals"], after["intervals"][:source_count],
              "Continuation rewrote accepted birth intervals")
    require(len(after["cohorts"]) >= source_cohorts, "Continuation discarded source birth cohorts")
    for original, current in zip(before["cohorts"], after["cohorts"]):
        require(current["live_q_hi"] <= original["live_q_hi"], "Continuation revived retired source material")
        inherited = {**current, "live_q_hi": original["live_q_hi"]}
        same_json(original, inherited, "Continuation rewrote original cohort birth geometry or identity")


def algorithm_control():
    positions = [0., .3, .9, .4, .8]
    intervals = [{"before_time_myr": float(i), "after_time_myr": float(i+1),
        "before_front_fraction": [positions[i]], "after_front_fraction": [positions[i+1]]}
        for i in range(4)]
    replay = replay_front([[1., 3.]], intervals)
    close(replay["formed_mass_by_face_layer_kg"].sum(), 5.2,
          "Independent growth/remelt/refreezing control formed mass failed")
    close(replay["live_mass_by_face_layer_kg"].sum(), 3.2,
          "Independent growth/remelt/refreezing control live mass failed")
    close(replay["remelted_mass_by_face_layer_kg"].sum(), 2.,
          "Independent growth/remelt/refreezing control retirement failed")
    require(len(replay["table"]) == 4 and replay["table"][-1]["id"] == 4
            and replay["table"][-1]["q_lo"] == .4,
            "Independent re-freezing control reused retired material")
    return {"unequal_layer_splitting": True, "deep_retirement": True,
            "refreezing_new_identity": True, "cohort_count": 4}


def compare_baseline(directory, baseline):
    results = []
    for name in ("born_fault_checkpoint.npz", "coupled_before.npz", "coupled_after.npz",
                 "coupled_restart_continued.npz"):
        old_meta, old_arrays = read_npz(baseline/name)
        new_meta, new_arrays = read_npz(directory/name)
        old_names, new_names = set(old_arrays)-{"source_checkpoint"}, set(new_arrays)-{"source_checkpoint"}
        require(old_names == new_names, "Existing physics array catalogue changed: "+name)
        for field in old_names:
            same_array(old_arrays[field], new_arrays[field], "Existing physics array changed: "+name+":"+field)
        before = {key: value for key, value in old_meta["state"].items() if key != "bulk_birth_history"}
        after = {key: value for key, value in new_meta["state"].items() if key != "bulk_birth_history"}
        same_json(before, after, "Existing physics scalar state changed: "+name)
        for key in ("parameters", "parameter_hash", "thermal_state", "orbit"):
            same_json(old_meta[key], new_meta[key], "Existing physics metadata changed: "+name+":"+key)
        if "contact_state" in old_meta:
            same_json(old_meta["contact_state"], new_meta["contact_state"], "Existing contact scalar state changed: "+name)
        results.append({"file": name, "baseline_sha256": file_sha(baseline/name),
            "actual_sha256": file_sha(directory/name), "physics_arrays_equal": len(old_names),
            "scalar_thermal_orbit_equal": True,
            "excluded_new_fields": ["format", "bulk_birth_history", "embedded_source_bytes/hash"]})
    return results


def verify(directory, *, baseline=None):
    imported = [name for name in sys.modules if name == "tectonics" or name.startswith("tectonics.")]
    require(not imported, "Independent readback imported project physics")
    execution_report = read_json(directory/"report.json")
    require(execution_report["result"] == "verified_real_genesis_bulk_birth_continuation",
            "The actual control did not complete")
    fault_meta, fault_arrays = read_npz(directory/"born_fault_checkpoint.npz")
    precursor_meta, precursor_arrays = read_npz(directory/"precursor_fault_checkpoint.npz")
    checkpoints = {name: read_npz(directory/name) for name in (
        "coupled_before.npz", "coupled_after.npz", "coupled_restart_continued.npz")}
    fault_history = read_json(directory/"bulk_history_fault.json")
    final_history = read_json(directory/"bulk_history_coupled.json")
    same_json(fault_history, fault_meta["state"]["bulk_birth_history"], "Standalone Fault bulk record differs from checkpoint")
    same_json(final_history, checkpoints["coupled_restart_continued.npz"][0]["state"]["bulk_birth_history"],
              "Standalone final bulk record differs from checkpoint")
    trace = read_json(directory/"formation_trace.json")
    require(trace[0]["time_myr"] == 0. and trace[0]["surface_melt_fraction"] == 1.
            and trace[0]["ocean_mass_kg"] == 0. and trace[0]["steam_pressure_bar"] > 100.
            and trace[0]["min_solid_depth_km"] == trace[0]["max_solid_depth_km"] == 0.,
            "Actual trajectory did not start from a molten planet without solid surface/ocean")
    require(all(not row["membrane_established"] for row in trace[:-1])
            and trace[-1]["membrane_established"], "Control did not stop at first membrane activation")
    histories = [("precursor", precursor_meta["state"]["bulk_birth_history"]),
                 ("fault", fault_history)]
    histories += [(name, meta["state"]["bulk_birth_history"]) for name, (meta, _) in checkpoints.items()]
    replay_results = {}
    for name, history in histories:
        # The last outer formation target can contain several accepted
        # mechanical substeps after activation.  Only the saved pre-membrane
        # precursor has the initial geometry at every recorded birth.
        replay = verify_history(history, exact_birth_geometry=name == "precursor")
        replay_results[name] = replay
        checkpoint_meta = (precursor_meta if name == "precursor" else fault_meta if name == "fault"
                           else checkpoints[name][0])
        times = [checkpoint_meta[key]["time_myr"] for key in ("state", "thermal_state", "orbit")]
        require(times == [history["time_myr"]]*3, "Saved material/thermal/orbit clocks disagree: "+name)
    same_array(fault_arrays["layer_mass_kg"], fault_history["layer_mass_kg"], "Source fixed mass columns changed")
    require(not precursor_meta["state"]["membrane_established"] and precursor_meta["state"]["bulk_birth_history"]["cohorts"],
            "Saved precursor has no genuinely earlier solid birth")
    same_array(precursor_arrays["elastic_strain"], 0.*precursor_arrays["elastic_strain"],
               "Precursor contains invented mechanical raft strain")
    verify_prefix(precursor_meta["state"]["bulk_birth_history"], fault_history)
    same_json(checkpoints["coupled_before.npz"][0]["state"]["bulk_birth_history"], fault_history,
              "Fault bulk record was not transferred exactly into Coupled")
    previous = fault_history
    for name, (meta, arrays) in checkpoints.items():
        require(arrays["source_checkpoint"].dtype == np.uint8
                and arrays["source_checkpoint"].tobytes() == (directory/"born_fault_checkpoint.npz").read_bytes(),
                "Coupled embedded source bytes changed: "+name)
        require(meta["source_hash"] == file_sha(directory/"born_fault_checkpoint.npz"), "Coupled source SHA changed: "+name)
        current = meta["state"]["bulk_birth_history"]
        verify_prefix(previous, current)
        previous = current
    inherited_names = ("column_enthalpy", "elastic_strain", "damage", "water_access", "fault_active",
                       "plane_normal", "fault_candidate_age_myr", "velocity_km_myr", "tidal_stress_mpa")
    for name in inherited_names:
        same_array(fault_arrays[name], checkpoints["coupled_before.npz"][1][name], "Fault field not inherited exactly: "+name)
    packet = read_json(directory/"material_export.json")
    require(packet["packet_sha256"] == digest({k: v for k, v in packet.items() if k != "packet_sha256"}, ascii=False),
            "Material export packet seal changed")
    require(base64.b64decode(packet["source"]["checkpoint_base64"], validate=True)
            == (directory/"born_fault_checkpoint.npz").read_bytes(), "Export embedded source bytes changed")
    require(packet["source"]["checkpoint_sha256"] == file_sha(directory/"born_fault_checkpoint.npz"),
            "Export source checksum changed")
    same_json(packet["source"]["parameters"], fault_meta["parameters"], "Export producer parameters changed")
    require(packet["source"]["parameter_hash"] == fault_meta["parameter_hash"], "Export parameter checksum changed")
    require(packet["birth_source_identity"] == fault_history["source_identity"], "Export material source identity changed")
    same_json(packet["bulk_birth_history"], fault_history, "Export bulk history differs from source")
    for export_name, source_name in (("layer_mass_kg", "layer_mass_kg"), ("specific_enthalpy_j_kg", "column_enthalpy")):
        same_array(packet["current_columns"][export_name], fault_arrays[source_name], "Export source column changed: "+source_name)
    same_array(packet["current_columns"]["connected_surface_solid_front_fraction"], fault_history["front_fraction"],
               "Export conductive front differs from source")
    same_array(packet["current_geometry"]["vertices_unit_xyz"], fault_arrays["vertices"], "Export material vertices changed")
    same_array(packet["current_geometry"]["faces"], fault_history["initial_faces"], "Export material faces changed")
    require(packet["current_geometry"]["radius_km"] == fault_meta["state"]["radius_km"], "Export material radius changed")
    area, basis, _ = face_geometry(fault_arrays["vertices"], fault_history["initial_faces"], fault_meta["state"]["radius_km"])
    close(packet["current_geometry"]["face_area_sr"], area, "Export material areas changed", rtol=2e-15)
    same_array(packet["current_geometry"]["face_frames_xyz"], basis[:, :, :2], "Export current chord frames changed")
    for target, source in (("elastic_strain_xx_yy_gamma_xy", "elastic_strain"),
                           ("velocity_km_myr_world", "velocity_km_myr"), ("damage", "damage"), ("water_access", "water_access")):
        same_array(packet["current_shell_memory"][target], fault_arrays[source], "Export current shell field changed: "+source)
    require(all(value is None for value in packet["unknown_fields"].values())
            and packet["capabilities"]["physical_force_work_admission"] is False
            and packet["capabilities"]["full_historical_three_dimensional_memory_present"] is False,
            "Export asserted unknown history or physical force/work admission")
    mass = fault_arrays["layer_mass_kg"]
    layer_catalogue = packet["original_material_layer_catalogue"]
    require(len(layer_catalogue) == mass.size and len({row["material_id"] for row in layer_catalogue}) == mass.size,
            "Original layer catalogue changed or repeated an identity")
    for row, (face, layer) in zip(layer_catalogue, np.ndindex(mass.shape)):
        require(row["original_face_index"] == face and row["layer_index"] == layer
                and row["layer_mass_kg"] == mass[face, layer] and row["current_owner"] is None
                and row["material_id"] == f"genesis:{fault_history['source_identity']}:face:{face}:layer:{layer}",
                "Original material layer identity/mass/ownership changed")
    cohort_catalogue = packet["solid_birth_cohort_material_catalogue"]
    require(len(cohort_catalogue) == len(fault_history["cohorts"])
            and len({row["material_id"] for row in cohort_catalogue}) == len(cohort_catalogue),
            "Birth cohort material catalogue has missing or duplicate identities")
    for row, cohort in zip(cohort_catalogue, fault_history["cohorts"]):
        face, layer, identity = cohort["face_index"], cohort["layer_index"], cohort["id"]
        require(row["source_cohort_id"] == identity and row["current_owner"] is None
                and row["material_id"] == f"genesis:{fault_history['source_identity']}:face:{face}:layer:{layer}:cohort:{identity}",
                "Birth cohort source identity or ownership changed")
        close(row["formed_mass_kg"], mass[face].sum()*(cohort["q_hi"]-cohort["q_lo"]), "Export cohort formed mass changed")
        close(row["live_mass_kg"], mass[face].sum()*(cohort["live_q_hi"]-cohort["q_lo"]), "Export cohort live mass changed")
    stress = packet["instantaneous_plane_stress_3d"]
    require(stress is not None and stress["historical_three_dimensional_memory_recovered"] is False,
            "Established shell export lacks its explicit instantaneous closure")
    shell = fault_meta["parameters"]["shell"]
    E = shell["young_modulus_pa"]*(shell["residual_stiffness"]
        +(1-shell["residual_stiffness"])*(1-fault_arrays["damage"])**2)
    nu, engineering = shell["poisson_ratio"], fault_arrays["elastic_strain"]
    elastic = np.zeros((len(engineering), 3, 3))
    elastic[:, 0, 0], elastic[:, 1, 1] = engineering[:, 0], engineering[:, 1]
    elastic[:, 0, 1] = elastic[:, 1, 0] = engineering[:, 2]/2
    elastic[:, 2, 2] = -nu/(1-nu)*(engineering[:, 0]+engineering[:, 1])
    lame, shear = E*nu/((1+nu)*(1-2*nu)), E/(2*(1+nu))
    sigma = (lame*np.trace(elastic, axis1=1, axis2=2))[:, None, None]*np.eye(3)+2*shear[:, None, None]*elastic
    close(stress["elastic_strain_local"], elastic, "Plane-stress elastic closure changed", rtol=2e-14, atol=1e-18)
    close(stress["stress_local_pa"], sigma, "Independent 3D Hooke stress differs from exported closure", rtol=2e-14, atol=2e-8)
    close(stress["elastic_strain_world"], basis@elastic@basis.transpose(0, 2, 1),
          "World elastic strain has incorrect basis", rtol=2e-14, atol=1e-18)
    close(stress["stress_world_pa"], basis@sigma@basis.transpose(0, 2, 1),
          "World stress has incorrect basis", rtol=2e-14, atol=2e-8)
    D = np.array([[1., nu, 0.], [nu, 1., 0.], [0., 0., (1-nu)/2]])/(1-nu**2)
    energy = .5*np.sum(engineering*(engineering@D.T)*E[:, None], axis=1)
    world_energy = .5*np.einsum("fij,fij->f", stress["elastic_strain_world"], stress["stress_world_pa"])
    close(world_energy, energy, "World tensor energy differs from independently reconstructed shell energy")
    close(stress["elastic_energy_density_j_m3"], energy, "Export current elastic energy density changed")
    solid_volume = mass.sum(axis=1)*np.asarray(fault_history["front_fraction"])/shell["density_kg_m3"]
    close(stress["solid_volume_m3"], solid_volume, "Plane-stress closure uses incorrect solid material volume")
    close(stress["elastic_energy_j"], float(np.dot(energy, solid_volume)), "Export total current elastic energy changed")
    require(execution_report["bulk_source_identity"] == fault_history["source_identity"]
            and execution_report["bulk_cohort_count"] == len(fault_history["cohorts"])
            and execution_report["source_accepted_birth_intervals"] == len(fault_history["intervals"])
            and execution_report["final_accepted_birth_intervals"] == len(final_history["intervals"]),
            "Actual report material identity/counts differ from independent readback")
    close(execution_report["bulk_live_mass_kg"], replay_results["fault"]["live_mass_by_face_layer_kg"].sum(),
          "Actual report solid mass differs from independent integration")
    require(execution_report["source_fault_sha256"] == file_sha(directory/"born_fault_checkpoint.npz")
            and execution_report["material_export_sha256"] == file_sha(directory/"material_export.json")
            and execution_report["source_config_sha256"] == file_sha(directory/"source_config.yaml"),
            "Actual report output/config byte hashes changed")
    source_hashes = read_json(directory/"physics_source_hashes.json")
    for module, pin in source_hashes.items():
        require(file_sha(ROOT/(module.replace(".", "/")+".py")) == pin,
                "Actual control physics source bytes changed: "+module)
    baseline_results = compare_baseline(directory, baseline) if baseline is not None else []
    files = {path.name: file_sha(path) for path in sorted(directory.iterdir()) if path.is_file()}
    report = {"format": FORMAT, "result": "PASS_INDEPENDENT_GENESIS_BULK_BIRTH_READBACK",
        "directory": str(directory.resolve()), "verifier_source_sha256": file_sha(__file__),
        "physics_modules_imported": imported, "actual_output_file_sha256": files,
        "independent_front_replay": [{"checkpoint": name, "time_myr": replay["time_myr"],
            "accepted_intervals": len(dict(histories)[name]["intervals"]), "cohorts": len(replay["table"]),
            "formed_mass_kg": float(replay["formed_mass_by_face_layer_kg"].sum()),
            "live_mass_kg": float(replay["live_mass_by_face_layer_kg"].sum()),
            "remelted_mass_kg": float(replay["remelted_mass_by_face_layer_kg"].sum())}
            for name, replay in replay_results.items()],
        "independent_algorithm_control": algorithm_control(),
        "material_birth_source_identity": fault_history["source_identity"],
        "actual_source_and_export_bytes_bound": True, "bulk_transfer_exact": True,
        "continuation_preserves_birth_prefix_and_material_identity": True,
        "source_shell_fields_and_geometry_exact": True,
        "birth_chart_metric_spd_and_frames_proper": True,
        "precursor_birth_geometry_reconstructed": True,
        "independent_current_plane_stress_hooke_and_world_energy_equal": True,
        "current_shell_elastic_energy_j": float(np.dot(energy, solid_volume)),
        "historical_three_dimensional_memory_known": False,
        "physical_force_work_admission": False,
        "baseline_existing_physics_comparison": baseline_results,
        "restart_scope": "Producer control checks exact replay; this verifier independently checks saved source/history/physics fields"}
    report["report_sha256"] = digest(report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--baseline", type=Path,
        default=ROOT/"analysis/genesis_origin_starter_validation/evidence/fault_canonical")
    args = parser.parse_args()
    report = verify(args.directory, baseline=args.baseline)
    args.report.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
