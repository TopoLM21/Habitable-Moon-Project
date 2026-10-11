"""Producer-free active coupled-volume test oracle.

The initial weak plane, water seed and extra force are explicitly synthetic.
The authentic formation source is retained as immutable context; this control
does not establish planetary fault activation or a physical sharp cut.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from analysis.genesis_volume_slip_validation import independent_audit as slip


volume, faults = slip.volume, slip.faults
FORMAT = "genesis-volume-slip-coupled-fixture-0.1"
SCOPE = "declared_synthetic_active_coupled_volume_test_not_actual_genesis"
PACKET_FORMAT = "synthetic-volume-slip-runtime-view-0.1"


def _close(actual, expected, name, *, rtol=3e-9, atol=1e-10):
    volume._assert_close(actual, expected, "coupled synthetic "+name, rtol=rtol, atol=atol)


def _memory(state):
    return {int(row["id"]): row for row in state["mechanics"]["state"]["cohorts"]}


def _packet(raw):
    volume._canonical(raw)
    if (raw.get("format") != PACKET_FORMAT or raw.get("scope") != SCOPE
            or "checkpoint_sha256" in raw or raw.get("authentic_checkpoint_claimed") is not False):
        raise ValueError("Coupled active test must use a declared synthetic runtime view")
    volume.source_dto_binding(raw)
    if raw["source_checkpoint_sha256"] != faults.SOURCE_SHA256:
        raise ValueError("Coupled synthetic test changed its immutable source context")
    state = raw["runtime_state"]
    if (state.get("slip_closure") != slip.SLIP_CLOSURE
            or state.get("plane_transport") != slip.PLANE_TRANSPORT
            or state.get("mechanical_work_closure") != slip.WORK_CLOSURE):
        raise ValueError("Coupled synthetic state changed its declared work/plane closure")
    volume.endpoint_clock_binding(state, state["geometry"]["time_myr"])
    source, history, thermal = raw["source_packet"], state["fault_history"], state["thermal"]
    count = state["mechanics"]["state"]["accepted_steps"]
    if (type(count) is not int or count < 0 or history["accepted_steps"] != count
            or thermal["accepted_steps"] != count or history["time_myr"] != state["geometry"]["time_myr"]
            or history["source_checkpoint_sha256"] != faults.SOURCE_SHA256
            or thermal["source_checkpoint_sha256"] != faults.SOURCE_SHA256
            or history["source_identity"] != source["birth_source_identity"]
            or thermal["source_identity"] != source["birth_source_identity"]
            or state["mechanics"]["state"]["source_identity"] != source["birth_source_identity"]):
        raise ValueError("Coupled synthetic state changed source identities or common counters/clocks")
    _close(thermal["damage"], source["current_shell_memory"]["damage"], "literal thermal damage baseline", rtol=0, atol=0)
    _close(thermal["layer_mass_kg"], source["current_columns"]["layer_mass_kg"], "literal source column mass", rtol=0, atol=0)
    original, birth = source["bulk_birth_history"], state["history"]
    for key in set(original)-{"cohorts", "intervals", "front_fraction", "time_myr", "sha256"}:
        if volume._canonical(birth[key]) != volume._canonical(original[key]):
            raise ValueError("Coupled synthetic state replaced original material birth context")
    if volume._canonical(birth["intervals"][:len(original["intervals"])]) != volume._canonical(original["intervals"]):
        raise ValueError("Coupled synthetic state replaced original accepted birth prefix")
    current = {int(c["id"]):c for c in birth["cohorts"]}
    for c in original["cohorts"]:
        actual = current.get(int(c["id"]))
        if actual is None or any(volume._canonical(actual[key]) != volume._canonical(value)
                for key, value in c.items() if key != "live_q_hi"):
            raise ValueError("Coupled synthetic state replaced an original formation material cohort")
        if not c["q_lo"] <= volume._high(actual) <= volume._high(c):
            raise ValueError("Coupled synthetic state invented old formation material")
    if len(state["accepted_step_sizes_myr"]) != count:
        raise ValueError("Coupled synthetic state changed its accepted journal count")
    return state


def _initial_binding(raw, seed):
    state = _packet(raw)
    # This verifies the genuine tensor/rate/birth import; the declared seed is
    # separately checked below and never passed off as formation history.
    initial = volume.audit_initial(raw)
    nf = len(state["geometry"]["faces"])
    face = seed["face_index"]
    if type(face) is not int or not 0 <= face < nf:
        raise ValueError("Synthetic seed face is invalid")
    if seed["fault_active"] is not True or seed["activation_time_myr"] != state["geometry"]["time_myr"]:
        raise ValueError("Synthetic seed activation must declare the test start clock")
    history = state["fault_history"]
    mask = np.asarray(history["fault_active"])
    expected = np.zeros(nf, bool); expected[face] = True
    if mask.dtype.kind != "b" or not np.array_equal(mask, expected):
        raise ValueError("Synthetic initial active mask differs from the declared seed")
    source = raw["source_packet"]
    metadata, source_fault_arrays, _ = faults.formation_fault_data(raw)
    seeded_fields = {"fault_active", "plane_normal", "activation_time_myr", "water_access"}
    for key, original in source_fault_arrays.items():
        actual = np.asarray(history[key])
        if key in seeded_fields:
            original = original.copy()
            original[face] = {"fault_active":True, "plane_normal":seed["plane_normal2"],
                "activation_time_myr":seed["activation_time_myr"], "water_access":seed["water_access"]}[key]
        if actual.shape != original.shape or not np.array_equal(actual, original):
            raise ValueError("Synthetic initial history changed an undeclared source field: "+key)
    if (history["source_checkpoint_sha256"] != faults.SOURCE_SHA256
            or history["source_identity"] != source["birth_source_identity"]
            or history["initial_time_myr"] != source["time_myr"]
            or history["membrane_established"] != metadata["state"]["membrane_established"]):
        raise ValueError("Synthetic initial fault context differs from its authentic source")
    for key in ("first_fracture_time_myr", "friction_work_j", "viscous_fault_work_j"):
        if history[key] != metadata["state"][key]:
            raise ValueError("Synthetic initial history changed an undeclared source scalar: "+key)
    _close(history["damage"], source["current_shell_memory"]["damage"], "unaltered initial damage", rtol=0, atol=0)
    water = np.zeros(nf); water[face] = seed["water_access"]
    _close(history["water_access"], water, "declared initial water seed", rtol=0, atol=0)
    normals = np.asarray(history["plane_normal"], float)
    _close(normals[face], seed["plane_normal2"], "declared initial plane seed", rtol=0, atol=0)
    if not math.isclose(float(normals[face]@normals[face]), 1., rel_tol=0, abs_tol=2e-12):
        raise ValueError("Synthetic seed plane must be a unit covector")
    for key in ("signed_shear", "cumulative_shear", "friction_work_cell_j", "viscous_work_cell_j"):
        if np.any(np.asarray(history[key]) != 0):
            raise ValueError("Synthetic seed invented previous slip work or shear")
    for key in ("physical_mechanical_work_j", "fault_friction_dissipation_j", "fault_viscous_dissipation_j"):
        if state[key] != 0:
            raise ValueError("Synthetic initial state invented previous runtime work")
    return {**initial, "formation_material_memory_unchanged": True,
        "initial_active_plane_and_water_are_explicit_synthetic_seed": True,
        "planetary_activation_claimed": False}


def _slip_fields_binding(actual, expected, name):
    for key in ("shear_increment", "trial_shear_stress_pa", "shear_stress_pa", "normal_stress_pa", "yield_strength_pa",
                "friction_work_cell_j", "viscous_work_cell_j", "friction_work_j", "viscous_fault_work_j",
                "plastic_work_j", "physical_mechanical_work_j"):
        is_work = "work" in key
        _close(actual[key], expected[key], name+" "+key, rtol=4e-8 if is_work else 3e-9,
               atol=1024. if is_work else (1e-12 if key == "shear_increment" else 1e-5))
    if not np.array_equal(actual["yielding"], expected["yielding"]):
        raise ValueError("Coupled synthetic "+name+" yielding mask differs")
    _close(actual["net_constitutive_driver_work_j"], expected["mechanical_work_j"], name+" net constitutive work", rtol=4e-8, atol=1024.)
    for key in ("average_strain0", "average_strain1", "plastic_increment0"):
        _close(actual[key], [row[key] for row in expected["cohorts"]], name+" "+key, rtol=3e-9, atol=2e-12)


def _world_binding(state):
    geometry = state["geometry"]
    X = volume._array(geometry["vertices_m"])+volume._array(state["displacement_m"]).reshape(-1, 6)[:, :3]
    frame, _ = faults.physical_face_geometry(X, np.asarray(geometry["faces"], int))
    normal2 = volume._array(state["fault_history"]["plane_normal"])
    active = np.asarray(state["fault_history"]["fault_active"], bool)
    n = np.einsum("fij,fj->fi", frame[:, :, :2], normal2)
    tangent2 = np.stack((-normal2[:, 1], normal2[:, 0]), axis=1)
    t = np.einsum("fij,fj->fi", frame[:, :, :2], tangent2)
    n, t = np.where(active[:, None], n, 0.), np.where(active[:, None], t, 0.)
    _close(state["plane_normal_world"], n, "endpoint world plane", rtol=0, atol=3e-12)
    _close(state["slip_direction_world"], t, "endpoint world tangent", rtol=0, atol=3e-12)
    return n, t, X, frame


def _transport_binding(before, after, old_n):
    _, _, X0, frame0 = _world_binding(before)
    n1, _, X1, frame1 = _world_binding(after)
    faces = np.asarray(after["geometry"]["faces"], int)
    old2 = np.einsum("fij,fj->fi", frame0[:, :, :2].transpose(0, 2, 1), old_n)
    edges = []
    for X, frame in ((X0, frame0), (X1, frame1)):
        tri = X[faces]
        E = np.stack((tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=2)
        edges.append(frame[:, :, :2].transpose(0, 2, 1)@E)
    deformation = edges[1]@np.linalg.inv(edges[0])
    transported = np.linalg.solve(deformation.transpose(0, 2, 1), old2[..., None])[..., 0]
    active = np.asarray(before["fault_history"]["fault_active"], bool)
    transported[active] /= np.linalg.norm(transported[active], axis=1)[:, None]
    expected = np.einsum("fij,fj->fi", frame1[:, :, :2], transported)
    _close(after["fault_history"]["plane_normal"], transported, "all accepted 2-D plane covectors", rtol=0, atol=4e-12)
    _close(n1[active], expected[active], "accepted top-face covector transport", rtol=0, atol=4e-12)


def audit_coupled_interval(before_packet, after_packet, synthetic_force_n):
    """Independently integrate one accepted active interval and all ledgers."""
    before, after = _packet(before_packet), _packet(after_packet)
    source = after_packet["source_packet"]
    if volume._canonical(before_packet["source_packet"]) != volume._canonical(source):
        raise ValueError("Coupled synthetic interval changed its full immutable formation context")
    geometry = after["geometry"]
    for key in ("vertices_m", "faces", "column_mass_kg", "density_kg_m3", "source_identity", "source_checkpoint_sha256"):
        if volume._canonical(before["geometry"][key]) != volume._canonical(geometry[key]):
            raise ValueError("Coupled synthetic interval changed its fixed reference: "+key)
    report, inputs, actual = after["last_report"], after["last_report"]["audit_input"], after["last_report"]["weak_plane_slip"]
    faults.declared_drag_binding(before, after)
    if (actual["closure"] != slip.SLIP_CLOSURE or actual["mechanical_work_closure"] != slip.WORK_CLOSURE
            or actual["plane_transport"] != slip.PLANE_TRANSPORT
            or actual["traction_closure"] != "exact_interval_average_constant_rate_Maxwell_path"):
        raise ValueError("Coupled synthetic slip report changed constitutive semantics")
    for key in ("sharp_banks_created", "exact_through_depth_finite_plane_transport_claimed", "physical_fault_contact_admission_claimed"):
        if actual[key] is not False:
            raise ValueError("Coupled synthetic weak plane made an unsupported physical claim: "+key)
    X, faces = volume._array(geometry["vertices_m"]), np.asarray(geometry["faces"], int)
    mass, rho = volume._array(geometry["column_mass_kg"]), float(geometry["density_kg_m3"])
    shell, thermal_params = (source["source"]["parameters"][key] for key in ("shell", "thermal"))
    bcohorts, acohorts = before["geometry"]["cohorts"], geometry["cohorts"]
    bmap, amap = ({int(c["id"]): c for c in rows} for rows in (bcohorts, acohorts))
    born = [c for c in acohorts if int(c["id"]) not in bmap]
    removed = []
    for c in bcohorts:
        survivor = amap.get(int(c["id"]))
        lower = c["q_lo"] if survivor is None else volume._high(survivor)
        if lower < volume._high(c):
            removed.append({**c, "q_lo": lower})
    for key, expected in (("before_cohorts", bcohorts), ("after_cohorts", acohorts)):
        if volume._canonical(inputs[key]) != volume._canonical(expected):
            raise ValueError("Coupled synthetic loading catalogue differs from accepted endpoint")
    q0, q1, v0, v1 = (volume._array(state[key]) for state, key in
        ((before, "displacement_m"), (after, "displacement_m"), (before, "velocity_m_s"), (after, "velocity_m_s")))
    dq = q1-q0
    dt = (report["after_time_myr"]-report["before_time_myr"])*slip.SECONDS_PER_MYR
    volume.endpoint_clock_binding(before, report["before_time_myr"])
    volume.endpoint_clock_binding(after, report["after_time_myr"])
    if dt <= 0 or after["accepted_step_sizes_myr"][:-1] != before["accepted_step_sizes_myr"]:
        raise ValueError("Coupled synthetic interval changed its accepted clock/journal")
    _close(inputs["before_displacement_m"], q0, "accepted start displacement", rtol=0, atol=0)
    _close(inputs["before_velocity_m_s"], v0, "accepted start velocity", rtol=0, atol=0)
    _close(inputs["increment_m"], dq, "accepted displacement increment", rtol=2e-12, atol=1e-14)
    active = np.asarray(before["fault_history"]["fault_active"])
    if active.dtype.kind != "b" or not np.array_equal(active, after["fault_history"]["fault_active"]):
        raise ValueError("Coupled synthetic test changed its explicitly seeded active mask")
    n, t, _, _ = _world_binding(before)
    _close(actual["frozen_start_plane_normal_world"], n, "frozen start normal", rtol=0, atol=3e-12)
    _close(actual["frozen_start_slip_direction_world"], t, "frozen start tangent", rtol=0, atol=3e-12)
    _transport_binding(before, after, n)
    damage0, damage1 = (volume._array(s["fault_history"]["damage"]) for s in (before, after))
    water = volume._array(before["fault_history"]["water_access"])
    _close(actual["damage"], damage1, "final mechanical damage", rtol=0, atol=0)
    _close(actual["water"], water, "lagged water", rtol=0, atol=0)
    _close(actual["endpoint_water_access"], after["fault_history"]["water_access"], "separate endpoint water", rtol=0, atol=0)
    if actual["water_coupling"] != "accepted_start_lagged_water_fixed_over_both_private_solves":
        raise ValueError("Coupled synthetic report changed its lagged-water closure")
    # Inactive faces have no plane. The private material argument needs unit
    # vectors, so harmless orthogonal placeholders are masked out by its law.
    pn, pt = n.copy(), t.copy(); pn[~active] = (1., 0., 0.); pt[~active] = (0., 1., 0.)
    bm, am = _memory(before), _memory(after)
    tm0, ts0 = volume.global_temperatures(before["thermal"]["global_state"], thermal_params)
    tm1, ts1 = volume.global_temperatures(after["thermal"]["global_state"], thermal_params)
    T0, T1 = (volume.conductive_temperatures(s["thermal"], thermal_params) for s in (before, after))
    layer_mass = volume._array(after["thermal"]["layer_mass_kg"])
    frames, _ = faults.physical_face_geometry(X, faces)
    rows, damage_energy, removed_energy, before_energy = [], 0., 0., 0.
    for cohorts, name in ((bcohorts, "before"), (removed, "removed")):
        energy = 0.
        for c in cohorts:
            cid, face = int(c["id"]), int(c["face_index"])
            C = volume.isotropic_matrix(shell["young_modulus_pa"], shell["poisson_ratio"], damage0[face], shell["residual_stiffness"])
            energy += volume.coefficient_energy(X[faces[face]], mass[face], rho, c, bm[cid]["strain0"], bm[cid]["strain1"], C)
        if name == "before": before_energy = energy
        else: removed_energy = energy
    _close(before["mechanics"]["state"]["elastic_energy_j"], before_energy, "start material energy", rtol=3e-12, atol=1024.)
    for index, c in enumerate(acohorts):
        cid, face = int(c["id"]), int(c["face_index"])
        old = bm.get(cid)
        old0, old1 = (np.zeros((3, 6)) if old is None else volume._array(old[key]) for key in ("strain0", "strain1"))
        lo, hi = float(c["q_lo"]), volume._high(c)
        previous_T = thermal_params["solidus_k"] if old is None else volume.profile_mean(layer_mass[face], T0[face], ts0, tm0, lo, hi)
        temperature = volume.profile_mean(layer_mass[face], T1[face], ts1, tm1, lo, hi)
        thermal0 = np.zeros((3, 6)); thermal0[:, :3] = shell["linear_expansion_per_k"]*(temperature-previous_T)
        viscosity = np.clip(shell["viscosity_reference_pa_s"]*np.exp(np.clip(shell["activation_energy_j_mol"]/8.314462618*(1/temperature-1/shell["viscosity_reference_temperature_k"]), -60, 60)), shell["viscosity_min_pa_s"], shell["viscosity_max_pa_s"])
        ratio = 0. if old is None else dt*shell["young_modulus_pa"]/viscosity
        C = volume.isotropic_matrix(shell["young_modulus_pa"], shell["poisson_ratio"], damage1[face], shell["residual_stiffness"])
        for key, expected, tol in (("old0", old0, 0.), ("old1", old1, 0.), ("thermal0", thermal0, 2e-13), ("basis_xyz", frames[face], 2e-12), ("elasticity_pa", C, 1e-6)):
            _close(inputs[key][index], expected, "independent loading "+key, rtol=2e-7 if key == "thermal0" else 3e-12, atol=tol)
        _close(inputs["maxwell_ratio"][index], ratio, "independent Maxwell rate", rtol=2e-10, atol=1e-15)
        _close(am[cid]["temperature_k"], temperature, "accepted temperature", rtol=2e-12, atol=1e-8)
        if old is not None:
            oldC = volume.isotropic_matrix(shell["young_modulus_pa"], shell["poisson_ratio"], damage0[face], shell["residual_stiffness"])
            damage_energy += volume.coefficient_energy(X[faces[face]], mass[face], rho, c, old0, old1, C-oldC)
        dofs = (6*faces[face, :, None]+np.arange(6)).ravel()
        rows.append(dict(face_index=face, triangle_m=X[faces[face]].tolist(), basis_xyz=frames[face].tolist(), geometry_cohort=c,
            column_mass_kg=mass[face], density_kg_m3=rho, displacement_increment18_m=dq[dofs].tolist(),
            old0=old0.tolist(), old1=old1.tolist(), thermal0=thermal0.tolist(), maxwell_ratio=float(ratio)))
    fixture = dict(format=slip.FIXTURE_FORMAT, scope=slip.FIXTURE_SCOPE,
        plane_normal_world=pn.tolist(), slip_direction_world=pt.tolist(), damage=damage1.tolist(), water=water.tolist(),
        active=active.tolist(), dt_myr=dt/slip.SECONDS_PER_MYR, elastic_parameters=shell,
        fault_parameters=source["source"]["parameters"]["faults"], cohorts=rows)
    expected = slip.independent_material_response(fixture)
    _slip_fields_binding(actual, expected, "accepted integrated slip")
    if not np.array_equal(actual["yielding"], expected["yielding"]) or not np.array_equal(actual["active"], active):
        raise ValueError("Coupled synthetic active/yielding mask differs")
    _close(actual["net_constitutive_driver_work_j"], expected["mechanical_work_j"], "net constitutive work", rtol=4e-8, atol=1024.)
    force = np.zeros(len(q0))
    for index, (c, result) in enumerate(zip(acohorts, expected["cohorts"])):
        cid, face = int(c["id"]), int(c["face_index"])
        for key in ("strain0", "strain1"):
            _close(am[cid][key], result[key], "accepted plastic coefficient "+key, rtol=3e-9, atol=2e-12)
        for key in ("average_strain0", "average_strain1", "plastic_increment0"):
            _close(actual[key][index], result[key], "full cohort "+key, rtol=3e-9, atol=2e-12)
        dofs = (6*faces[face, :, None]+np.arange(6)).ravel()
        np.add.at(force, dofs, result["average_force_n"])
    totals = {key: expected[key] for key in ("mechanical_work_j", "thermal_work_j", "viscous_dissipation_j", "elastic_energy_j", "physical_mechanical_work_j", "plastic_work_j")}
    _close(report["mechanical_work_j"], totals["physical_mechanical_work_j"], "top physical mechanical work", rtol=4e-8, atol=1024.)
    _close(report["net_constitutive_driver_work_j"], totals["mechanical_work_j"], "top net constitutive work", rtol=4e-8, atol=1024.)
    _close(report["fault_friction_dissipation_j"], expected["friction_work_j"], "top interval friction dissipation", rtol=4e-8, atol=1024.)
    _close(report["fault_viscous_dissipation_j"], expected["viscous_fault_work_j"], "top interval viscous slip dissipation", rtol=4e-8, atol=1024.)
    for key in ("mechanical_work_j", "thermal_work_j", "viscous_dissipation_j"):
        if key != "mechanical_work_j":
            _close(report[key], totals[key], "interval reported "+key, rtol=4e-8, atol=1024.)
        _close(after["mechanics"]["state"][key]-before["mechanics"]["state"][key], totals[key], "cumulative "+key, rtol=4e-8, atol=8192.)
    _close(after["mechanics"]["state"]["elastic_energy_j"], expected["elastic_energy_j"], "accepted endpoint energy", rtol=3e-9, atol=1024.)
    for statekey, value in (("physical_mechanical_work_j", expected["physical_mechanical_work_j"]), ("fault_friction_dissipation_j", expected["friction_work_j"]), ("fault_viscous_dissipation_j", expected["viscous_fault_work_j"])):
        _close(after[statekey]-before[statekey], value, "cumulative "+statekey, rtol=4e-8, atol=8192.)
    for key, value in (("damage_energy_change_j", damage_energy), ("remelt_energy_removed_j", removed_energy)):
        _close(after["mechanics"]["state"][key]-before["mechanics"]["state"][key], value, "signed cumulative "+key, rtol=4e-8, atol=8192.)
    elastic_residual = expected["elastic_energy_j"]-before_energy-(expected["physical_mechanical_work_j"]+expected["thermal_work_j"]-expected["plastic_work_j"]-expected["viscous_dissipation_j"]+damage_energy-removed_energy)
    if abs(elastic_residual) > 3e-9*max(before_energy, expected["elastic_energy_j"], 1.):
        raise ValueError("Independent coupled slip/Maxwell/damage/remelt energy balance failed")
    history0, history1 = before["fault_history"], after["fault_history"]
    _close(history1["activation_time_myr"], history0["activation_time_myr"], "unchanged declared seed activation times", rtol=0, atol=0)
    if actual["activation_event"] is not None:
        raise ValueError("Synthetic two-interval control invented an undeclared activation event")
    for key, resultkey in (("last_shear_increment", "shear_increment"), ("shear_stress_pa", "shear_stress_pa"),
            ("normal_stress_pa", "normal_stress_pa"), ("shear_strength_pa", "yield_strength_pa")):
        value = np.where(active, expected[resultkey], np.asarray(history0[key]))
        _close(history1[key], value, "accepted last traction/shear history "+key, rtol=3e-9, atol=1e-12 if key == "last_shear_increment" else 1e-5)
    for key, increment in (("signed_shear", expected["shear_increment"]), ("cumulative_shear", np.abs(expected["shear_increment"])), ("friction_work_cell_j", expected["friction_work_cell_j"]), ("viscous_work_cell_j", expected["viscous_work_cell_j"]), ("friction_work_j", expected["friction_work_j"]), ("viscous_fault_work_j", expected["viscous_fault_work_j"])):
        _close(np.asarray(history1[key])-np.asarray(history0[key]), increment, "accepted inherited slip ledger "+key, rtol=4e-8, atol=8192. if "work" in key else 1e-12)
    traction = volume.prescribed_traction(X, faces, mass, rho, acohorts, source["current_geometry"]["radius_km"], shell)
    _close(inputs["basal_traction_at_vertices_pa"], traction, "source prescribed basal traction", rtol=2e-10, atol=1e-8)
    D, F = volume.basal_drag_and_force(X, faces, mass, rho, acohorts, inputs["basal_drag_pa_s_m"], traction[faces])
    _close(inputs["source_basal_external_force_n"], F, "unchanged source basal force", rtol=3e-10, atol=1.)
    _close(inputs["synthetic_additional_external_force_n"], synthetic_force_n, "declared fixed synthetic force", rtol=0, atol=0)
    F += volume._array(synthetic_force_n, F.shape)
    _close(inputs["total_external_force_n"], F, "source plus declared external force", rtol=3e-10, atol=1.)
    M0, M1, Min, Mout = (volume.mass_matrix(X, faces, mass, rho, rows) for rows in (bcohorts, acohorts, born, removed))
    balance = volume.dynamic_balance(M0, M1, Min, Mout, D, F, force, q0, q1, v0, v1, dt, expected["physical_mechanical_work_j"])
    # Independently rebuild the unpublished predictor from the same ORIGINAL
    # memory/thermal trial, then calculate its current-axis stress. Its slip
    # work is checked, but is never added to any accepted cumulative ledger.
    evolution = report["fault_evolution"]
    driver = evolution["predictor_driver"]
    retained = faults.retained_fraction(bcohorts, acohorts, mass)
    predictor_damage = damage0*retained
    _close(driver["retained_fraction"], retained, "predictor actual retained material", rtol=3e-12, atol=2e-14)
    _close(driver["damage"], predictor_damage, "predictor retained start damage", rtol=3e-12, atol=2e-14)
    p_q = volume._array(driver["displacement_m"])
    p_dq = p_q-q0
    p_rows = []
    for row in rows:
        dofs = (6*faces[row["face_index"], :, None]+np.arange(6)).ravel()
        p_rows.append({**row, "displacement_increment18_m":p_dq[dofs].tolist()})
    predictor = slip.independent_material_response({**fixture, "damage":predictor_damage.tolist(), "cohorts":p_rows})
    p_actual = driver["weak_plane_slip"]
    _slip_fields_binding(p_actual, predictor, "private predictor slip")
    _close(p_actual["water"], water, "private predictor same lagged water", rtol=0, atol=0)
    _close(p_actual["endpoint_water_access"], water, "private predictor unchanged endpoint water", rtol=0, atol=0)
    _close(p_actual["frozen_start_plane_normal_world"], n, "private predictor frozen start plane", rtol=0, atol=3e-12)
    _close(p_actual["frozen_start_slip_direction_world"], t, "private predictor frozen start tangent", rtol=0, atol=3e-12)
    pe0, pe1 = ([row[key] for row in predictor["cohorts"]] for key in ("strain0", "strain1"))
    _close(driver["elastic_strain0"], pe0, "predictor endpoint coefficient0", rtol=3e-9, atol=2e-12)
    _close(driver["elastic_strain1"], pe1, "predictor endpoint coefficient1", rtol=3e-9, atol=2e-12)
    p_X = X+p_q.reshape(-1, 6)[:, :3]
    _close(driver["physical_vertices_m"], p_X, "predictor physical vertices", rtol=0, atol=1e-8)
    temperatures = [am[int(c["id"])]["temperature_k"] for c in acohorts]
    p_response = faults.face_stress_from_coefficients(X, faces, mass, rho, acohorts, pe0, pe1,
        temperatures, predictor_damage, p_X, shell)
    for key in ("stress_pa", "mean_temperature_k"):
        _close(driver[key], p_response[key], "predictor independently calculated "+key,
               rtol=3e-9, atol=1e-5 if key == "stress_pa" else 1e-8)
    if not np.array_equal(driver["active_material"], p_response["active_material"]):
        raise ValueError("Coupled synthetic predictor changed its live material mask")
    _close(driver["surface_temperature_k"], ts1, "predictor surface temperature", rtol=3e-12, atol=1e-8)
    _close(driver["ocean_fraction"], faults.ocean_fraction(ts1, thermal_params), "predictor ocean fraction", rtol=3e-12, atol=1e-12)
    damage_law, damage_report = faults.independent_damage_law(before["fault_history"], driver, source,
        before["thermal"]["orbit"], after["thermal"]["orbit"], report["after_time_myr"])
    final_response = faults.face_stress_from_coefficients(X, faces, mass, rho, acohorts,
        [c["strain0"] for c in expected["cohorts"]], [c["strain1"] for c in expected["cohorts"]],
        temperatures, damage1, X+q1.reshape(-1, 6)[:, :3], shell)
    final_peak = faults.tensile_principal(final_response["stress_pa"])
    damage_law["peak_tensile_pa"] = np.maximum(damage_law["peak_tensile_pa"], final_peak)
    _close(evolution["final_face_tensile_pa"], final_peak, "final independently calculated peak", rtol=3e-9, atol=1e-5)
    for key, value in damage_law.items():
        _close(after["fault_history"][key], value, "calculated damage/water/history "+key,
            rtol=3e-9, atol=1e-5 if key == "peak_tensile_pa" else 1e-12)
    first_fracture = before["fault_history"]["first_fracture_time_myr"]
    if first_fracture is None and np.any(p_response["active_material"] & (damage_law["damage"] >= shell["damage_threshold"])):
        first_fracture = report["after_time_myr"]
    if after["fault_history"]["first_fracture_time_myr"] != first_fracture:
        raise ValueError("Coupled synthetic state changed its calculated diffuse fracture clock")
    for key, value in damage_report.items():
        _close(evolution[key], value, "damage evolution report "+key, rtol=3e-9, atol=1e-12)
    p_force = np.zeros(len(q0))
    for c, response in zip(acohorts, predictor["cohorts"]):
        dofs = (6*faces[int(c["face_index"]), :, None]+np.arange(6)).ravel()
        np.add.at(p_force, dofs, response["average_force_n"])
    p_velocity = 2*p_dq/dt-v0
    p_balance = volume.dynamic_balance(M0, M1, Min, Mout, D, F, p_force, q0, p_q, v0, p_velocity, dt,
        predictor["physical_mechanical_work_j"])
    for independent, reported in (("kinetic_birth_inflow_j", "birth_kinetic_energy_j"), ("kinetic_remelt_outflow_j", "remelt_kinetic_energy_j"), ("drag_dissipation_j", "basal_dissipation_j"), ("external_work_j", "external_work_j")):
        _close(report[reported], balance[independent], "full-force "+reported, rtol=4e-8, atol=8192.)
    for key, value in (("external_work_j", balance["external_work_j"]), ("basal_dissipation_j", balance["drag_dissipation_j"]), ("phase_kinetic_energy_change_j", balance["kinetic_birth_inflow_j"]-balance["kinetic_remelt_outflow_j"])):
        _close(after[key]-before[key], value, "accepted cumulative "+key, rtol=4e-8, atol=8192.)
    _close(after["kinetic_energy_j"], balance["kinetic_after_j"], "accepted kinetic energy", rtol=3e-10, atol=1e-5)
    m = after["mechanics"]["state"]
    cumulative_expected = m["initial_energy_j"]+after["initial_kinetic_energy_j"]+after["external_work_j"]+m["thermal_work_j"]-m["viscous_dissipation_j"]-after["basal_dissipation_j"]-after["fault_friction_dissipation_j"]-after["fault_viscous_dissipation_j"]+m["damage_energy_change_j"]-m["remelt_energy_removed_j"]+after["phase_kinetic_energy_change_j"]
    cumulative_residual = expected["elastic_energy_j"]+balance["kinetic_after_j"]-cumulative_expected
    energy_scale = max(m["initial_energy_j"], expected["elastic_energy_j"], balance["kinetic_after_j"], 1.)
    if abs(cumulative_residual) > 3e-9*energy_scale or abs(after["global_balance_residual_j"]) > 3e-9*energy_scale:
        raise ValueError("Independent full cumulative coupled work balance failed")
    geometry_audit = volume.geometry_binding(after["last_certificate"], X, faces, q0, q1, bcohorts+acohorts, mass, rho, report["before_time_myr"], report["after_time_myr"])
    return dict(before_time_myr=report["before_time_myr"], after_time_myr=report["after_time_myr"],
        live_cohort_count=len(acohorts), newborn_count=len(born), removed_band_count=len(removed),
        yielding_face_count=int(expected["yielding"].sum()), maximum_absolute_shear_increment=float(np.max(np.abs(expected["shear_increment"]))),
        material_quadrature={**totals, "damage_energy_change_j":damage_energy, "remelt_energy_removed_j":removed_energy,
            "full_elastic_balance_residual_j":elastic_residual},
        maximum_cohort_relative_work_residual=max(c["relative_balance_residual"] for c in expected["cohorts"]),
        discrete_dynamic_balance=balance, cumulative_full_work_relative_residual=abs(cumulative_residual)/energy_scale,
        predictor_full_coordinate_force_and_work=p_balance, predictor_stress_and_damage_law_independently_recomputed=True,
        geometry=geometry_audit, independently_checked_full_18_coordinate_force=True,
        exact_through_depth_finite_plane_transport_claimed=False, actual_planetary_activation_claimed=False)


def audit_coupled_fixture(raw):
    volume._canonical(raw)
    if raw.get("format") != FORMAT or raw.get("scope") != SCOPE:
        raise ValueError("Active coupled control must declare its synthetic test scope")
    for key in ("actual_source_replaced", "actual_planet_activation_claimed", "sharp_banks_created", "physical_fault_contact_admission_claimed", "original_point_1_completed"):
        if raw.get(key) is not False:
            raise ValueError("Synthetic coupled control made an unsupported claim: "+key)
    if raw.get("initial_memory_unchanged") is not True:
        raise ValueError("Synthetic coupled control must preserve literal initial material memory")
    packets = [raw[key] for key in ("initial_packet", "after_first_packet", "after_second_packet")]
    initial = _initial_binding(packets[0], raw["synthetic_seed"])
    counts = [packet["runtime_state"]["mechanics"]["state"]["accepted_steps"] for packet in packets]
    if counts != [0, 1, 2]:
        raise ValueError("Synthetic coupled oracle requires exactly two accepted intervals")
    intervals = [audit_coupled_interval(packets[i], packets[i+1], raw["synthetic_force_n"]) for i in range(2)]
    if not all(interval["yielding_face_count"] > 0 for interval in intervals):
        raise ValueError("Synthetic active coupled control did not yield in both accepted intervals")
    restarted = raw.get("restarted_after_second_packet")
    restart = False
    if restarted is not None:
        _packet(restarted)
        if volume._canonical(restarted) != volume._canonical(packets[-1]):
            raise ValueError("Complete synthetic direct/restarted runtime packets differ")
        restart = True
    report = dict(format="genesis-volume-slip-coupled-independent-audit-0.1", status="PASS", producer_imports=False,
        scope=SCOPE, initial_formation_material_binding=initial, intervals=intervals,
        two_active_coupled_accepted_intervals_independently_checked=True, complete_runtime_restart_pair_equal=restart,
        quadrature="Cartesian_Duffy4x4_Gauss32_massq_Gauss32_adaptive_time_bisection",
        initial_active_plane_water_and_extra_force_are_explicit_synthetic_inputs=True,
        actual_source_replaced=False, actual_planet_activation_claimed=False, sharp_banks_created=False,
        physical_fault_contact_admission_claimed=False, original_point_1_completed=False,
        whole_heat_mechanical_reservoir_closed=False)
    return {**report, "report_sha256":hashlib.sha256(volume._canonical(report)).hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    source_bytes = args.fixture.read_bytes()
    report = audit_coupled_fixture(json.loads(source_bytes))
    body = {key:value for key, value in report.items() if key != "report_sha256"}
    body["input_fixture_sha256"] = hashlib.sha256(source_bytes).hexdigest()
    report = {**body, "report_sha256":hashlib.sha256(volume._canonical(body)).hexdigest()}
    encoded = json.dumps(report, sort_keys=True, indent=2, allow_nan=False)+"\n"
    if args.output is not None: args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
