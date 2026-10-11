"""Independent readback of actual Genesis cohort Maxwell continuation.

Only the Python standard library and NumPy are used.  No producer validator,
physics module, checkpoint loader, or simulation is imported or executed.
"""
from __future__ import annotations

import argparse
import base64
from bisect import bisect_right
import hashlib
import io
import json
import math
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
FORMAT = "genesis-cohort-mechanics-independent-file-readback-0.1"


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


def thermal_endpoints(metadata, parameters):
    """Invert the recorded enthalpy with bisection, independent of SciPy.

    The IAPWS coefficients and reservoir definitions are the declared source
    law. Bisection uses this law directly instead of importing its inverter.
    """
    p = parameters["thermal"]
    energy = metadata["thermal_state"]["energy"]
    area = 4*math.pi*(1000*p["radius_km"])**2
    water = p["water_volume_km3"]*1e9*p["water_density_kg_m3"]/area
    surface = p["surface_layer_depth_m"]*p["surface_layer_density_kg_m3"]
    mantle = 5.9722e24*p["mass_earth"]*p["mantle_mass_fraction"]/area-surface

    def rock(t):
        fraction = min(1., max(0., (t-p["solidus_k"])/(p["liquidus_k"]-p["solidus_k"])))
        return p["silicate_heat_capacity_j_kg_k"]*t+p["silicate_latent_heat_j_kg"]*fraction

    def vapor(t):
        if t >= 647.096:
            return water
        if t < 273.16:
            pressure = 611.657*math.exp(5420.*(1/273.16-1/max(t, 1.)))
        else:
            theta = 1-t/647.096
            pressure = 22.064e6*math.exp(647.096/t*sum(a*theta**b for a, b in
                ((-7.85951783, 1), (1.84408259, 1.5), (-11.7866497, 3),
                 (22.6807411, 3.5), (-15.9618719, 4), (1.80122502, 7.5))))
        amount = min(water, pressure/p["surface_gravity_m_s2"])
        if water*p["surface_gravity_m_s2"] > 22.064e6:
            x = np.clip((647.096-t)/p["critical_transition_width_k"], 0., 1.)
            amount = water+x*x*(3-2*x)*(amount-water)
        return amount

    def invert(target, law):
        lo, hi = 0., max(4000., p["initial_temperature_k"]*2)
        require(law(hi) > target >= law(lo), "Thermal inversion energy outside finite bracket")
        for _ in range(90):
            middle = (lo+hi)/2
            if law(middle) < target:
                lo = middle
            else:
                hi = middle
        return (lo+hi)/2

    tm = invert(float(energy[0])*1e12, lambda t: mantle*rock(t))
    ts = invert(float(energy[1])*1e12, lambda t: surface*rock(t)
        +water*p["water_heat_capacity_j_kg_k"]*t+p["water_latent_heat_j_kg"]*vapor(t))
    return tm, ts


def column_temperatures(enthalpy, p):
    h = np.asarray(enthalpy, dtype=float)
    cp, latent = p["silicate_heat_capacity_j_kg_k"], p["silicate_latent_heat_j_kg"]
    solidus, liquidus = p["solidus_k"], p["liquidus_k"]
    width = liquidus-solidus
    return np.where(h <= cp*solidus, h/cp, np.where(h >= cp*liquidus+latent,
        (h-latent)/cp, (h+latent*solidus/width)/(cp+latent/width)))


def profile_mean(mass, temperature, surface, mantle, lo, hi):
    bounds = np.concatenate(([0.], np.cumsum(mass)/sum(mass)))
    nodes = np.concatenate(([0.], .5*(bounds[:-1]+bounds[1:]), [1.]))
    values = np.concatenate(([surface], temperature, [mantle]))
    integral = 0.
    for a, b, ta, tb in zip(nodes[:-1], nodes[1:], values[:-1], values[1:]):
        left, right = max(a, lo), min(b, hi)
        if right > left:
            integral += (right-left)*(ta+(tb-ta)/(b-a)*((left+right)/2-a))
    return integral/(hi-lo)


def verify_mechanics(raw, history, metadata, arrays, parameters):
    require(set(raw) == {"format", "energy_scope", "state", "packet_sha256"}
        and raw["format"] == "genesis-cohort-mechanics-0.1"
        and raw["energy_scope"] == "activated_loadbearing_plane_stress_membrane",
        "Cohort constitutive contract changed")
    require(digest({k: v for k, v in raw.items() if k != "packet_sha256"})
            == raw["packet_sha256"], "Cohort mechanics seal changed")
    state = raw["state"]
    require(state["source_identity"] == history["source_identity"]
            and state["time_myr"] == history["time_myr"] == metadata["state"]["time_myr"]
            and state["accepted_intervals"] == len(history["intervals"]),
            "Mechanical memory lost original material identity or accepted clock")
    for owner in ("thermal_state", "orbit"):
        require(metadata[owner]["time_myr"] == state["time_myr"], "A physics owner has a different time")
    births = [c for c in history["cohorts"] if c["live_q_hi"] > c["q_lo"]]
    records = state["cohorts"]
    require(len(births) == len(records), "Mechanical records differ from live material coverage")
    mass = np.asarray(history["layer_mass_kg"], float)
    p = parameters["shell"]
    temperature = column_temperatures(arrays["column_enthalpy"], parameters["thermal"])
    tm, ts = thermal_endpoints(metadata, parameters)
    n = len(mass)
    volume_sum, average_sum = np.zeros(n), np.zeros((n, 3))
    energy, all_volumes, all_strains, all_faces = 0., [], [], []
    gate_times = set()
    by_face = {}
    for birth, record in zip(births, records):
        for key in ("id", "face_index", "layer_index", "q_lo", "live_q_hi"):
            require(record[key] == birth[key], "A mechanical cohort changed source material identity")
        face = record["face_index"]
        expected_t = profile_mean(mass[face], temperature[face], ts, tm,
                                 record["q_lo"], record["live_q_hi"])
        close(record["temperature_k"], expected_t, "Cohort temperature is not its live source-profile mean",
              rtol=2e-12, atol=5e-8)
        if not state["established"]:
            require(all(record[k] is None for k in ("elastic_strain", "activation_time_myr",
                "activation_temperature_k", "activation_closure", "activation_metric_m2",
                "activation_frame_xyz")), "Unmodelled solid rafts gained fabricated mechanical history")
            continue
        require(record["activation_time_myr"] >= birth["birth_after_time_myr"]
                and record["activation_time_myr"] <= state["time_myr"],
                "Mechanical activation is outside its accepted material lifetime")
        if record["activation_closure"] == "whole_membrane_gate_endpoint_temperature":
            gate_times.add(record["activation_time_myr"])
        else:
            require(record["activation_closure"] == "solidus_thermal_first_endpoint_cooling"
                and record["activation_time_myr"] == birth["birth_after_time_myr"]
                and record["activation_temperature_k"] == birth["birth_temperature_k"],
                "Post-gate birth replaced its declared stress-free solidus activation")
            same_json(record["activation_metric_m2"], birth["birth_metric_m2"], "Birth activation metric changed")
            same_json(record["activation_frame_xyz"], birth["birth_frame_xyz"], "Birth activation frame changed")
        basis = np.asarray(record["activation_frame_xyz"])
        close(basis.T@basis, np.eye(3), "Activation basis is not orthonormal", rtol=0, atol=2e-12)
        close(np.linalg.det(basis), 1., "Activation basis is not proper", rtol=0, atol=2e-12)
        require(np.all(np.linalg.eigvalsh(record["activation_metric_m2"]) > 0), "Activation metric is not positive")
        strain = np.asarray(record["elastic_strain"], float)
        require(strain.shape == (3,) and np.isfinite(strain).all(), "Active cohort strain is not finite in-plane tensor")
        volume = sum(mass[face])*(record["live_q_hi"]-record["q_lo"])/p["density_kg_m3"]
        nu = p["poisson_ratio"]
        degradation = p["residual_stiffness"]+(1-p["residual_stiffness"])*(1-arrays["damage"][face])**2
        c = p["young_modulus_pa"]*degradation/(1-nu**2)*np.array(
            [[1., nu, 0.], [nu, 1., 0.], [0., 0., (1-nu)/2]])
        energy += volume*.5*strain@c@strain
        volume_sum[face] += volume
        average_sum[face] += volume*strain
        all_volumes.append(volume); all_strains.append(strain); all_faces.append(face)
        by_face.setdefault(face, []).append(record)
    require(len(gate_times) <= 1, "Original cohorts activated at different whole-membrane gates")
    mean = np.divide(average_sum, volume_sum[:, None], out=np.zeros_like(average_sum),
                     where=volume_sum[:, None] > 0)
    close(mean, arrays["elastic_strain"], "Saved face memory is not live volume-weighted cohort strain",
          rtol=2e-12, atol=2e-17)
    close(energy, state["elastic_energy_j"], "Physical energy is not the sum of live cohort energies",
          rtol=3e-12, atol=1e-7)
    expected = (state["mechanical_work_j"]-state["plastic_work_j"]+state["thermal_work_j"]
        -state["viscous_dissipation_j"]+state["damage_energy_change_j"]-state["remelt_energy_removed_j"])
    scale = max(abs(energy), abs(expected), abs(state["mechanical_work_j"]), abs(state["thermal_work_j"]), 1.)
    residual = energy-expected
    require(abs(residual) < 7e-12*scale, "Independent membrane work identity does not close")
    require(abs(residual-state["balance_residual_j"]) < 7e-12*scale, "Recorded work residual differs from independent result")
    require(state["viscous_dissipation_j"] >= 0 and state["remelt_energy_removed_j"] >= 0,
            "A passive membrane loss is negative")
    if all(c["live_q_hi"] == c["q_hi"] for c in history["cohorts"]):
        require(state["remelt_energy_removed_j"] == 0.,
                "The mechanics ledger removed energy although no material ever remelted")
    if not state["established"]:
        require(all(state[key] == 0 for key in ("elastic_energy_j", "mechanical_work_j",
            "thermal_work_j", "viscous_dissipation_j", "plastic_work_j",
            "damage_energy_change_j", "remelt_energy_removed_j", "balance_residual_j")),
            "Pre-gate rafts acquired calculated work or energy")
    contrast_t = max((max(c["temperature_k"] for c in values)-min(c["temperature_k"] for c in values)
                      for values in by_face.values()), default=0.)
    contrast_e = max((np.max(np.ptp([c["elastic_strain"] for c in values], axis=0))
                      for values in by_face.values()), default=0.)
    return {"established": state["established"], "live_cohorts": len(records),
            "independent_elastic_energy_j": energy, "relative_work_balance_residual": abs(residual)/scale,
            "max_same_face_temperature_contrast_k": float(contrast_t),
            "max_same_face_strain_component_contrast": float(contrast_e),
            "boundary_temperature_k": {"mantle": tm, "surface": ts}}


def compare_checkpoints(directory, first, second):
    left_meta, left_arrays = read_npz(directory/first)
    right_meta, right_arrays = read_npz(directory/second)
    same_json(left_meta, right_meta, "Restart checkpoint metadata differs: "+first)
    require(set(left_arrays) == set(right_arrays), "Restart checkpoint array members differ")
    for name in left_arrays:
        same_array(left_arrays[name], right_arrays[name], "Restart array differs: "+name)
    return {"first": first, "second": second, "exact_metadata": True,
            "exact_array_members": len(left_arrays), "zip_container_bytes_compared": False}


def verify_activation_prefix(source, current):
    by_id = {c["id"]: c for c in current["state"]["cohorts"]}
    for record in source["state"]["cohorts"]:
        if record["id"] not in by_id:
            continue  # Full remelting may retire a mechanical record.
        inherited = by_id[record["id"]]
        for key in ("activation_time_myr", "activation_temperature_k", "activation_closure",
                    "activation_metric_m2", "activation_frame_xyz"):
            same_json(record[key], inherited[key], "Continuation replaced original cohort activation: "+key)


def verify(directory):
    directory = Path(directory)
    report = read_json(directory/"report.json")
    require(report["result"] == "verified_real_genesis_cohort_mechanics_continuation",
            "The captured physics control did not accept actual formation continuation")
    formation = read_json(directory/"formation_trace.json")
    birth = formation[0]
    require(birth["time_myr"] == 0 and birth["surface_melt_fraction"] == 1
            and birth["ocean_mass_kg"] == 0 and birth["steam_pressure_bar"] > 100
            and not birth["membrane_established"] and birth["min_solid_depth_km"] == 0,
            "Source control did not begin with the recorded molten steam planet")
    checkpoint_names = ("precursor_fault_checkpoint.npz", "born_fault_checkpoint.npz",
        "fault_after.npz", "fault_restart_continued.npz", "coupled_before.npz",
        "coupled_after.npz", "coupled_restart_continued.npz")
    snapshots = []
    source_meta, source_arrays = read_npz(directory/"born_fault_checkpoint.npz")
    for name in checkpoint_names:
        metadata, arrays = read_npz(directory/name)
        history = metadata["state"]["bulk_birth_history"]
        verify_history(history, exact_birth_geometry=name == "precursor_fault_checkpoint.npz")
        parameters = metadata["parameters"]
        if name.startswith("coupled"):
            source_bytes = bytes(arrays["source_checkpoint"])
            require(source_bytes == (directory/"born_fault_checkpoint.npz").read_bytes(),
                    "Coupled continuation changed embedded Genesis producer bytes")
            source_meta, _ = read_npz(io.BytesIO(source_bytes))
            parameters = source_meta["parameters"]
            inherited_work = source_meta["state"]["cohort_mechanics"]["state"]["mechanical_work_j"]
            current_work = metadata["state"]["cohort_mechanics"]["state"]["mechanical_work_j"]
            increment_work = metadata["state"]["bulk_increment_work_j"]
            scale = max(abs(inherited_work), abs(current_work), abs(increment_work), 1.)
            require(abs(current_work-inherited_work-increment_work) < scale*5e-13,
                    "Coupled bulk increment work differs from its cohort time-path integral")
        if name not in ("precursor_fault_checkpoint.npz", "born_fault_checkpoint.npz"):
            verify_prefix(source_meta["state"]["bulk_birth_history"], history)
            verify_activation_prefix(source_meta["state"]["cohort_mechanics"],
                                     metadata["state"]["cohort_mechanics"])
        facts = verify_mechanics(metadata["state"]["cohort_mechanics"], history,
                                 metadata, arrays, parameters)
        snapshots.append({"file": name, "sha256": file_sha(directory/name), **facts})
    require(not snapshots[0]["established"] and snapshots[1]["established"],
            "Formation did not preserve pending precursor then activate membrane")
    require(snapshots[-1]["max_same_face_temperature_contrast_k"] > 0
            and snapshots[-1]["max_same_face_strain_component_contrast"] > 0,
            "Final cohort memory has no calculated depth heterogeneity")
    restart_checks = [compare_checkpoints(directory, a, b) for a, b in (
        ("born_fault_checkpoint.npz", "precursor_resumed_born_fault.npz"),
        ("fault_direct_continued.npz", "fault_restart_continued.npz"),
        ("coupled_direct_continued.npz", "coupled_restart_continued.npz"))]
    fault_meta, fault_arrays = read_npz(directory/"born_fault_checkpoint.npz")
    coupled_meta, coupled_arrays = read_npz(directory/"coupled_before.npz")
    same_json(fault_meta["state"]["cohort_mechanics"], coupled_meta["state"]["cohort_mechanics"],
              "Initial Coupled changed actual prospective cohort memory")
    for name in ("column_enthalpy", "elastic_strain", "damage", "water_access", "velocity_km_myr"):
        same_array(fault_arrays[name], coupled_arrays[name], "Fault-to-Coupled material field changed: "+name)
    packet = read_json(directory/"material_export.json")
    require(digest({k: v for k, v in packet.items() if k != "packet_sha256"}, ascii=False)
            == packet["packet_sha256"], "Strict material packet seal changed")
    source_bytes = base64.b64decode(packet["source"]["checkpoint_base64"], validate=True)
    require(source_bytes == (directory/"born_fault_checkpoint.npz").read_bytes()
            and hashlib.sha256(source_bytes).hexdigest() == packet["source"]["checkpoint_sha256"],
            "Material export changed source bytes")
    same_json(packet["cohort_mechanics"], fault_meta["state"]["cohort_mechanics"],
              "Material export changed prospective depth memory")
    same_json(packet["bulk_birth_history"], fault_meta["state"]["bulk_birth_history"],
              "Material export changed material birth history")
    same_array(packet["current_geometry"]["vertices_unit_xyz"], fault_arrays["vertices"],
               "Export changed actual source vertices")
    require(packet["current_geometry"]["radius_km"] == fault_meta["state"]["radius_km"],
            "Export changed actual source radius")
    for packet_name, array_name in (("layer_mass_kg", "layer_mass_kg"),
                                    ("specific_enthalpy_j_kg", "column_enthalpy")):
        same_array(packet["current_columns"][packet_name], fault_arrays[array_name],
                   "Export changed actual source conductive column: "+array_name)
    for packet_name, array_name in (("elastic_strain_xx_yy_gamma_xy", "elastic_strain"),
        ("velocity_km_myr_world", "velocity_km_myr"), ("damage", "damage"), ("water_access", "water_access")):
        same_array(packet["current_shell_memory"][packet_name], fault_arrays[array_name],
                   "Export changed actual source shell field: "+array_name)
    for json_name, checkpoint_name, member in (
        ("cohort_mechanics_fault.json", "born_fault_checkpoint.npz", "cohort_mechanics"),
        ("cohort_mechanics_fault_continued.json", "fault_restart_continued.npz", "cohort_mechanics"),
        ("cohort_mechanics_coupled.json", "coupled_restart_continued.npz", "cohort_mechanics"),
        ("bulk_history_fault.json", "born_fault_checkpoint.npz", "bulk_birth_history"),
        ("bulk_history_coupled.json", "coupled_restart_continued.npz", "bulk_birth_history")):
        metadata, _ = read_npz(directory/checkpoint_name)
        same_json(read_json(directory/json_name), metadata["state"][member],
                  "Standalone material JSON differs from actual checkpoint: "+json_name)
    verify_export_tensors(packet)
    require(not packet["capabilities"]["physical_force_work_admission"]
            and not packet["capabilities"]["full_historical_three_dimensional_memory_present"]
            and not packet["capabilities"]["native943_source_adopted"]
            and all(v is None for v in packet["unknown_fields"].values()),
            "Export promoted a membrane closure into unavailable 3D formation history")
    sources = read_json(directory/"physics_source_hashes.json")
    for name, expected in sources.items():
        path = ROOT/Path(*name.split(".")).with_suffix(".py")
        require(file_sha(path) == expected, "Physics source changed since actual control: "+name)
    result = {"format": FORMAT, "verified": True, "producer_physics_imports": False,
        "source_directory": str(directory.resolve()), "source_control_report_sha256": file_sha(directory/"report.json"),
        "algorithm_control": algorithm_control(), "snapshots": snapshots, "exact_restart_checks": restart_checks,
        "source_cohort_memory_inherited_exact": True, "source_material_fields_inherited_exact": True,
        "independent_profile_temperatures_verified": True, "cohort_volume_weighted_strain_verified": True,
        "independent_cohort_elastic_energy_verified": True, "independent_membrane_work_identity_verified": True,
        "unavailable_pre_gate_motion_retained_as_none": True, "strict_source_bytes_and_export_tensors_verified": True,
        "physics_source_hashes_verified": len(sources), "physical_force_work_admission": False,
        "full_historical_three_dimensional_memory_recovered": False,
        "limits": ["Accepted constitutive state and cumulative identity readback; not a replay of every mechanical interval",
                   "Plane-stress activated membrane with common face damage; no early isolated-raft dynamics",
                   "This standalone membrane identity is separate from thermal, mantle, contact and fault friction reservoirs"]}
    result["report_sha256"] = digest(result)
    return result


def verify_export_tensors(packet):
    """Use full 3D Hooke law independently of the source 2D extension."""
    values = packet["instantaneous_cohort_plane_stress_3d"]
    records = packet["cohort_mechanics"]["state"]["cohorts"]
    require(values is not None and values["source_cohort_ids"] == [r["id"] for r in records],
            "Export cohort tensors lost live source identities")
    p = packet["source"]["parameters"]["shell"]
    mass = np.asarray(packet["current_columns"]["layer_mass_kg"])
    damage = np.asarray(packet["current_shell_memory"]["damage"])
    basis = np.asarray(values["basis_xyz"])
    strain = np.asarray(values["elastic_strain_local"])
    nu = p["poisson_ratio"]
    energy = 0.
    for index, record in enumerate(records):
        face = record["face_index"]
        e = np.asarray(record["elastic_strain"])
        expected = np.array([[e[0], e[2]/2, 0.], [e[2]/2, e[1], 0.],
                             [0., 0., -nu/(1-nu)*(e[0]+e[1])]])
        close(strain[index], expected, "Export strain is not explicit plane-stress extension", rtol=2e-13, atol=1e-18)
        effective = p["young_modulus_pa"]*(p["residual_stiffness"]
            +(1-p["residual_stiffness"])*(1-damage[face])**2)
        lame = effective*nu/((1+nu)*(1-2*nu))
        shear = effective/(2*(1+nu))
        stress = lame*np.trace(expected)*np.eye(3)+2*shear*expected
        close(values["stress_local_pa"][index], stress, "Export stress differs from full 3D Hooke law", rtol=2e-12, atol=1e-6)
        close(values["elastic_strain_world"][index], basis[index]@expected@basis[index].T,
              "Export world strain changed the material chart", rtol=3e-12, atol=1e-18)
        close(values["stress_world_pa"][index], basis[index]@stress@basis[index].T,
              "Export world stress changed the material chart", rtol=3e-12, atol=1e-6)
        volume = sum(mass[face])*(record["live_q_hi"]-record["q_lo"])/p["density_kg_m3"]
        close(values["live_volume_m3"][index], volume, "Export changed live cohort volume", rtol=2e-13)
        density = .5*np.sum(expected*stress)
        close(values["elastic_energy_density_j_m3"][index], density, "Export tensor energy differs", rtol=2e-13)
        energy += volume*density
    close(values["elastic_energy_j"], energy, "Export lost cohort energy variance", rtol=3e-12)
    close(energy, packet["cohort_mechanics"]["state"]["elastic_energy_j"], "Export physical energy differs from source", rtol=3e-12)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.directory)
    if args.output is not None:
        args.output.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False)+"\n")
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()


