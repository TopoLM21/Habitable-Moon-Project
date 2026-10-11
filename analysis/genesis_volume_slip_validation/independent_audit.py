"""Producer-free quadrature for declared 3-D material weak-plane slip.

The numerical oracle integrates the conical field in angular coordinates,
material mass and physical time.  Its scalar yield solve uses bisection rather
than the producer's algebraic return map.  Synthetic material controls remain
separate from the actual molten-formation source and never admit sharp banks.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from analysis.genesis_volume_continuation_validation import independent_audit as volume
from analysis.genesis_volume_fault_validation import independent_audit as faults


FIXTURE_FORMAT = "genesis-volume-slip-material-fixture-0.1"
FIXTURE_SCOPE = "declared_synthetic_3d_material_weak_plane_test_not_actual_genesis"
SECONDS_PER_MYR = 365.25*86400*1e6
FORMAT = "genesis-volume-slip-checkpoint-0.1"
SCOPE = "actual_genesis_source_replayed_connected_volume_finite_width_weak_plane_slip"
SLIP_CLOSURE = "volume_mean_time_average_traction_with_constant_rate_face_shear"
PLANE_TRANSPORT = "top_face_covector_endpoint_transport_with_frozen_interval_flow"
WORK_CLOSURE = "base_mechanics_net_driver_and_explicit_total_slip_work"


def _vectors(normal, tangent):
    normal, tangent = volume._array(normal), volume._array(tangent)
    if (normal.ndim != 2 or normal.shape[-1] != 3 or tangent.shape != normal.shape
            or not np.allclose(np.linalg.norm(normal, axis=1), 1., rtol=0, atol=2e-11)
            or not np.allclose(np.linalg.norm(tangent, axis=1), 1., rtol=0, atol=2e-11)
            or not np.allclose(np.einsum("fi,fi->f", normal, tangent), 0., rtol=0, atol=2e-11)):
        raise ValueError("Independent slip planes require orthonormal 3-D normal/tangent")
    return normal, tangent


def engineering_flow(frame, normal_world, tangent_world):
    """Rotate the actual symmetric world tensor, including all three shears."""
    Q = volume._array(frame, (3, 3))
    if (not np.allclose(Q.T@Q, np.eye(3), rtol=0, atol=2e-11)
            or not np.isclose(np.linalg.det(Q), 1., rtol=0, atol=2e-11)):
        raise ValueError("Independent cohort basis must be proper orthonormal")
    n, t = volume._array(normal_world, (3,)), volume._array(tangent_world, (3,))
    world = .5*(np.outer(t, n)+np.outer(n, t))
    local = Q.T@world@Q
    flow = np.array((local[0, 0], local[1, 1], local[2, 2],
                     2*local[0, 1], 2*local[0, 2], 2*local[1, 2]))
    normal_tensor = Q.T@np.outer(n, n)@Q
    projector = np.array((normal_tensor[0, 0], normal_tensor[1, 1], normal_tensor[2, 2],
        2*normal_tensor[0, 1], 2*normal_tensor[0, 2], 2*normal_tensor[1, 2]))
    if abs(flow[:3].sum()) > 3e-11:
        raise ValueError("Independent 3-D slip is not nondilatant")
    return flow, projector


def _cohort_fields(row, C, normal, tangent):
    triangle = volume._array(row["triangle_m"], (3, 3))
    frame = volume._array(row["basis_xyz"], (3, 3))
    cohort = row["geometry_cohort"]
    N, wa = volume.angular_rule()
    q, wr = volume.radial_rule(float(cohort["q_lo"]), volume._high(cohort))
    lam, _ = volume.conical_scales(triangle, row["column_mass_kg"], row["density_kg_m3"], q)
    weights = (wa[:, None]*wr[None, :]).ravel()
    V = row["column_mass_kg"]*(volume._high(cohort)-float(cohort["q_lo"]))/row["density_kg_m3"]
    B0, B1 = volume.independent_strain_operators(triangle, frame, N)
    B = (B0[:, None]+B1[:, None]/lam[None, :, None, None]).reshape(-1, 6, 18)
    dq = volume._array(row["displacement_increment18_m"], (18,))
    inc = B@dq
    interpolation = N@np.linalg.inv(volume.SOURCE_ANGULAR_POINTS)
    old0, old1, thermal0 = (volume._array(row[k], (3, 6)) for k in ("old0", "old1", "thermal0"))
    old = ((interpolation@old0)[:, None]+(interpolation@old1)[:, None]/lam[None, :, None]).reshape(-1, 6)
    thermal = np.repeat(interpolation@thermal0, len(lam), axis=0)
    flow, projector = engineering_flow(frame, normal, tangent)
    if abs(projector@C@flow) > 5e-11*max(np.max(np.abs(C)), 1.):
        raise ValueError("Independent isotropic shear changed plane-normal traction")
    ratio = float(row["maxwell_ratio"])
    a, b, aa, ab, bb = volume.temporal_factors(ratio)
    driver = inc-thermal
    stress = (a*old+b*driver)@C.T
    return dict(row=row, C=C, B=B, weights=weights, volume=V, old=old, thermal=thermal,
        increment=inc, flow=flow, projector=projector, ratio=ratio,
        factors=(a, b, aa, ab, bb), trial_average_stress=stress)


def _integral(field, values):
    return float(field["volume"]*np.dot(field["weights"], values))


def _material_response(field, gamma):
    """Integrate the ODE's energy and conjugate work without producer moments."""
    C, old = field["C"], field["old"]
    plastic = gamma*field["flow"]
    driver = field["increment"]-field["thermal"]-plastic
    a, b, aa, ab, bb = field["factors"]
    stress = (a*old+b*driver)@C.T
    inner = lambda first, second: np.einsum("pi,ij,pj->p", first, C, second)
    work = _integral(field, np.einsum("pi,pi->p", stress, field["increment"]))
    thermal_work = -_integral(field, np.einsum("pi,pi->p", stress, field["thermal"]))
    plastic_work = _integral(field, stress@plastic)
    loss = _integral(field, aa*inner(old, old)+ab*inner(old, driver)+bb*inner(driver, driver))
    r = field["ratio"]
    beta = 1. if r == 0 else -math.expm1(-r)/r
    endpoint = math.exp(-r)*old+beta*driver
    E0, E1 = .5*_integral(field, inner(old, old)), .5*_integral(field, inner(endpoint, endpoint))
    residual = E1-E0-work-thermal_work+plastic_work+loss
    scale = max(E0, E1, abs(work), abs(thermal_work), abs(plastic_work), abs(loss), 1.)
    if abs(residual) > 3e-11*scale or loss < -3e-13*scale:
        raise ValueError("Independent 3-D slip/Maxwell work identity failed")
    row = field["row"]
    B0, B1 = volume.independent_strain_operators(volume._array(row["triangle_m"]),
        volume._array(row["basis_xyz"]), volume.SOURCE_ANGULAR_POINTS)
    dq = volume._array(row["displacement_increment18_m"])
    inc0, inc1 = B0@dq, B1@dq
    old0, old1, thermal0 = (volume._array(row[k]) for k in ("old0", "old1", "thermal0"))
    return dict(strain0=math.exp(-r)*old0+beta*(inc0-thermal0-plastic),
        strain1=math.exp(-r)*old1+beta*inc1,
        average_strain0=a*old0+b*(inc0-thermal0-plastic),
        average_strain1=a*old1+b*inc1,
        plastic_increment0=np.tile(plastic, (3, 1)),
        elastic_energy_j=E1, before_energy_j=E0,
        physical_mechanical_work_j=work, thermal_work_j=thermal_work,
        plastic_work_j=plastic_work, viscous_dissipation_j=loss,
        balance_residual_j=residual, relative_balance_residual=abs(residual)/scale,
        average_force_n=field["volume"]*np.einsum("p,pai,pa->i", field["weights"], field["B"], stress))


def independent_material_response(raw):
    """Evaluate a declared fixture; a material control is never planetary data."""
    if raw.get("format") != FIXTURE_FORMAT or raw.get("scope") != FIXTURE_SCOPE:
        raise ValueError("Independent slip fixture must declare its synthetic material scope")
    normal, tangent = _vectors(raw["plane_normal_world"], raw["slip_direction_world"])
    nf = len(normal)
    damage, water = (volume._array(raw[k], (nf,)) for k in ("damage", "water"))
    active = np.asarray(raw["active"])
    if active.shape != (nf,) or active.dtype.kind != "b":
        raise ValueError("Independent material slip active mask must be boolean")
    if np.any((damage < 0)|(damage > 1)|(water < 0)|(water > 1)):
        raise ValueError("Independent material slip fractions must be in [0,1]")
    dt = raw["dt_myr"]
    if type(dt) not in (float, int) or not math.isfinite(dt) or dt <= 0:
        raise ValueError("Independent material slip time must be positive")
    shell, params = raw["elastic_parameters"], raw["fault_parameters"]
    V, trial, pressure, H = (np.zeros(nf) for _ in range(4))
    fields = []
    for row in raw["cohorts"]:
        face = row["face_index"]
        if type(face) is not int or not 0 <= face < nf:
            raise ValueError("Independent material slip face identity is invalid")
        C = volume.isotropic_matrix(shell["young_modulus_pa"], shell["poisson_ratio"],
            damage[face], shell["residual_stiffness"])
        field = _cohort_fields(row, C, normal[face], tangent[face])
        # An inactive declared plane has no shear/normal traction observable;
        # the underlying bulk elastic field and its work remain unchanged.
        if not active[face]:
            field["flow"] = np.zeros(6)
            field["projector"] = np.zeros(6)
        fields.append(field)
        V[face] += field["volume"]
        trial[face] += _integral(field, field["trial_average_stress"]@field["flow"])
        pressure[face] += _integral(field, field["trial_average_stress"]@field["projector"])
        H[face] += field["volume"]*field["factors"][1]*(field["flow"]@C@field["flow"])
    if np.any(V <= 0):
        raise ValueError("Independent material slip requires live cohorts on every fixture face")
    trial, pressure, H = trial/V, pressure/V, H/V
    friction = params["friction_dry"]+(params["friction_wet"]-params["friction_dry"])*water
    cohesion = (params["cohesion_pa"]*(params["residual_cohesion_fraction"]
        +(1-params["residual_cohesion_fraction"])*(1-damage))
        *(1-(1-params["wet_cohesion_fraction"])*water))
    strength = cohesion+friction*np.maximum(-pressure, 0.)
    zeta = params["viscosity_pa_s"]/(dt*SECONDS_PER_MYR)
    gamma = np.zeros(nf)
    yielding = active & params["enabled"] & (np.abs(trial) > strength)
    for face in np.flatnonzero(yielding):
        sign = np.sign(trial[face])
        low, high = 0., abs(trial[face])/H[face]
        # Monotone residual of the integrated traction and viscous resistance.
        for _ in range(90):
            middle = (low+high)/2
            residual = abs(trial[face])-H[face]*middle-strength[face]-zeta*middle
            if residual > 0:
                low = middle
            else:
                high = middle
        gamma[face] = sign*(low+high)/2
    if np.max(np.abs(gamma)) > params["max_shear_increment"]:
        raise ValueError("Independent material slip exceeds the source shear increment boundary")
    average_shear = trial-H*gamma
    friction_work = V*strength*np.abs(gamma)
    viscous_work = V*zeta*gamma**2
    responses = [_material_response(field, gamma[field["row"]["face_index"]]) for field in fields]
    plastic_work = sum(row["plastic_work_j"] for row in responses)
    if abs(plastic_work-friction_work.sum()-viscous_work.sum()) > 3e-11*max(abs(plastic_work), 1.):
        raise ValueError("Independent integrated plastic traction work disagrees with dissipation")
    return dict(shear_increment=gamma, trial_shear_stress_pa=trial,
        shear_stress_pa=average_shear, normal_stress_pa=pressure, yield_strength_pa=strength,
        yielding=yielding, friction_work_cell_j=friction_work, viscous_work_cell_j=viscous_work,
        friction_work_j=float(friction_work.sum()), viscous_fault_work_j=float(viscous_work.sum()),
        plastic_work_j=float(plastic_work), face_volume_m3=V,
        physical_mechanical_work_j=sum(row["physical_mechanical_work_j"] for row in responses),
        thermal_work_j=sum(row["thermal_work_j"] for row in responses),
        viscous_dissipation_j=sum(row["viscous_dissipation_j"] for row in responses),
        elastic_energy_j=sum(row["elastic_energy_j"] for row in responses),
        mechanical_work_j=sum(row["physical_mechanical_work_j"]-row["plastic_work_j"] for row in responses),
        balance_residual_j=sum(row["balance_residual_j"] for row in responses), cohorts=responses)


def audit_material_fixture(raw):
    volume._canonical(raw)
    expected = independent_material_response(raw)
    actual = raw["producer_result"]
    for key, value in expected.items():
        if key in ("cohorts", "face_volume_m3"):
            continue
        if key == "yielding":
            if np.asarray(actual[key]).dtype.kind != "b" or not np.array_equal(actual[key], value):
                raise ValueError("Independent material slip yielding mask differs")
        elif key == "balance_residual_j":
            scale = max(expected["elastic_energy_j"], abs(expected["physical_mechanical_work_j"]), 1.)
            if abs(actual[key]) > 5e-11*scale:
                raise ValueError("Independent reported material slip work balance failed")
        else:
            volume._assert_close(actual[key], value, "material slip "+key, rtol=3e-10,
                atol=1e-11 if "shear_increment" == key else 1e-5)
    for index, result in enumerate(expected["cohorts"]):
        for key in ("strain0", "strain1", "average_strain0", "average_strain1", "plastic_increment0"):
            volume._assert_close(actual[key][index], result[key], "material slip cohort "+key, rtol=3e-9, atol=2e-12)
        volume._assert_close(actual["average_force_cohort_n"][index], result["average_force_n"],
            "material slip full 18-coordinate force", rtol=3e-9, atol=1e-2)
    report = dict(format="genesis-volume-slip-material-independent-audit-0.1", status="PASS",
        producer_imports=False, fixture_scope=FIXTURE_SCOPE,
        quadrature="Cartesian_Duffy4x4_Gauss32_massq_Gauss32_adaptive_time_bisection",
        cohort_count=len(expected["cohorts"]), yielding_face_count=int(expected["yielding"].sum()),
        maximum_absolute_shear_increment=float(np.max(np.abs(expected["shear_increment"]))),
        plastic_work_j=expected["plastic_work_j"], friction_work_j=expected["friction_work_j"],
        viscous_fault_work_j=expected["viscous_fault_work_j"],
        physical_mechanical_work_j=expected["physical_mechanical_work_j"],
        net_mechanical_work_j=expected["mechanical_work_j"],
        viscous_maxwell_dissipation_j=expected["viscous_dissipation_j"],
        maximum_relative_cohort_balance_residual=max(c["relative_balance_residual"] for c in expected["cohorts"]),
        all_3d_shear_components_retained=True, actual_planet_activation_claimed=False,
        full_coordinate_force_independently_integrated=True,
        sharp_banks_created=False, physical_contact_or_underthrust_admitted=False,
        full_historical_3d_recovery_claimed=False)
    return {**report, "report_sha256": hashlib.sha256(volume._canonical(report)).hexdigest()}


def audit_material_directory(directory):
    directory = Path(directory)
    reports, inputs = [], []
    names = ("positive_3d_slip.json", "negative_3d_slip.json", "inactive_plane.json", "pure_compression.json")
    for name in names:
        path = directory/name
        encoded = path.read_bytes()
        reports.append({"name": name, **audit_material_fixture(json.loads(encoded))})
        inputs.append(dict(name=name, raw_sha256=hashlib.sha256(encoded).hexdigest()))
    report = dict(format="genesis-volume-slip-material-controls-independent-audit-0.1",
        status="PASS", producer_imports=False, fixture_scope=FIXTURE_SCOPE,
        input_files=inputs, controls=reports, actual_planet_activation_claimed=False,
        synthetic_material_state_substituted_for_real_formation_history=False,
        sharp_banks_created=False, physical_contact_or_underthrust_admitted=False,
        original_point_1_completed=False)
    return {**report, "report_sha256": hashlib.sha256(volume._canonical(report)).hexdigest()}


def actual_state_binding(packet):
    """Bind genuine fault-zero source data before any private quadrature view."""
    if packet.get("format") != FORMAT or packet.get("scope") != SCOPE:
        raise ValueError("Independent actual volume-slip checkpoint scope changed")
    binding = faults.inherited_fault_binding(packet)
    state = packet["runtime_state"]
    if (state.get("slip_closure") != SLIP_CLOSURE
            or state.get("plane_transport") != PLANE_TRANSPORT
            or state.get("mechanical_work_closure") != WORK_CLOSURE):
        raise ValueError("Independent actual volume-slip constitutive closure changed")
    nf = len(state["geometry"]["faces"])
    for name in ("plane_normal_world", "slip_direction_world"):
        value = volume._array(state[name], (nf, 3))
        if np.any(value != 0):
            raise ValueError("Actual inactive formation source gained a fabricated 3-D plane")
    for name in ("fault_friction_dissipation_j", "fault_viscous_dissipation_j"):
        if type(state[name]) not in (int, float) or state[name] != 0:
            raise ValueError("Actual inactive formation source gained fabricated slip loss")
    volume._assert_close(state["physical_mechanical_work_j"],
        state["mechanics"]["state"]["mechanical_work_j"],
        "actual zero-slip total versus net mechanical work", rtol=0, atol=0)
    return {**binding, "work_closure": WORK_CLOSURE,
        "actual_source_active_weak_planes": 0,
        "runtime_3d_inactive_planes_and_slip_loss_remain_exact_zero": True}


def inactive_slip_report_binding(before_packet, after_packet, report, displacement, damage, water, endpoint_water):
    """Reconstruct all zero-slip reports, including the unpublished predictor."""
    before, after = before_packet["runtime_state"], after_packet["runtime_state"]
    if (report["closure"] != SLIP_CLOSURE
            or report["traction_closure"] != "exact_interval_average_constant_rate_Maxwell_path"
            or report["plane_transport"] != PLANE_TRANSPORT
            or report["mechanical_work_closure"] != WORK_CLOSURE
            or report["water_coupling"] != "accepted_start_lagged_water_fixed_over_both_private_solves"):
        raise ValueError("Independent inactive slip report changed constitutive semantics")
    if report["finite_width_continuous_material"] is not True:
        raise ValueError("Independent slip report omitted its continuous-material scope")
    if type(report["newton_iterations"]) is not int or report["newton_iterations"] != 0:
        raise ValueError("Independent inactive slip report invented Newton iterations")
    for name in ("sharp_banks_created", "exact_through_depth_finite_plane_transport_claimed",
                 "physical_fault_contact_admission_claimed"):
        if report[name] is not False:
            raise ValueError("Independent slip report makes an unsupported physical claim: "+name)
    geometry = after["geometry"]
    nf, nc = len(geometry["faces"]), len(geometry["cohorts"])
    for name in ("frozen_start_plane_normal_world", "frozen_start_slip_direction_world"):
        volume._assert_close(report[name], np.zeros((nf, 3)), "actual inactive slip "+name, rtol=0, atol=0)
    for name in ("shear_increment", "shear_stress_pa", "normal_stress_pa", "trial_shear_stress_pa",
                 "friction_work_cell_j", "viscous_work_cell_j"):
        volume._assert_close(report[name], np.zeros(nf), "actual inactive slip "+name, rtol=0, atol=0)
    for name in ("yielding", "active"):
        mask = np.asarray(report[name])
        if mask.shape != (nf,) or mask.dtype.kind != "b" or mask.any():
            raise ValueError("Independent inactive slip boolean mask changed: "+name)
    for name in ("friction_work_j", "viscous_fault_work_j", "plastic_work_j"):
        if type(report[name]) not in (float, int) or report[name] != 0:
            raise ValueError("Independent inactive slip invented work: "+name)
    volume._assert_close(report["plastic_increment0"], np.zeros((nc, 3, 6)),
        "actual inactive plastic coefficient field", rtol=0, atol=0)
    damage, water = volume._array(damage), volume._array(water)
    volume._assert_close(report["damage"], damage, "actual slip mechanical damage", rtol=0, atol=0)
    volume._assert_close(report["water"], water, "actual slip water closure", rtol=0, atol=0)
    volume._assert_close(report["endpoint_water_access"], endpoint_water,
        "actual slip separately evolved endpoint water", rtol=0, atol=0)
    params = after_packet["source_packet"]["source"]["parameters"]["faults"]
    cohesion = (params["cohesion_pa"]*(params["residual_cohesion_fraction"]
        +(1-params["residual_cohesion_fraction"])*(1-damage))
        *(1-(1-params["wet_cohesion_fraction"])*water))
    volume._assert_close(report["yield_strength_pa"], cohesion, "actual inactive source cohesion", rtol=3e-14, atol=1e-7)
    X, faces = volume._array(geometry["vertices_m"]), np.asarray(geometry["faces"], int)
    mass, density = volume._array(geometry["column_mass_kg"]), geometry["density_kg_m3"]
    q0, q1 = volume._array(before["displacement_m"]), volume._array(displacement)
    dq = q1-q0
    inputs = after["last_report"]["audit_input"]
    shell = after_packet["source_packet"]["source"]["parameters"]["shell"]
    means0, means1, physical_work, force = [], [], 0., np.zeros(len(q0))
    for index, cohort in enumerate(geometry["cohorts"]):
        face = cohort["face_index"]
        frame, _ = faults.physical_face_geometry(X, faces[face][None])
        frame = frame[0]
        dofs = (6*faces[face, :, None]+np.arange(6)).ravel()
        B0, B1 = volume.independent_strain_operators(X[faces[face]], frame, volume.SOURCE_ANGULAR_POINTS)
        old0, old1, thermal0 = (volume._array(inputs[name][index]) for name in ("old0", "old1", "thermal0"))
        ratio = inputs["maxwell_ratio"][index]
        beta, theta = volume.temporal_factors(ratio)[:2]
        means0.append(beta*old0+theta*(B0@dq[dofs]-thermal0))
        means1.append(beta*old1+theta*(B1@dq[dofs]))
        C = volume.isotropic_matrix(shell["young_modulus_pa"], shell["poisson_ratio"], damage[face], shell["residual_stiffness"])
        result = volume.quadrature_cohort(X[faces[face]], frame, mass[face], density, cohort,
            dq[dofs], old0, old1, thermal0, ratio, C)
        physical_work += result["mechanical_work_j"]
        np.add.at(force, dofs, result["average_force_n"])
    volume._assert_close(report["average_strain0"], means0, "actual zero-slip average coefficient0", rtol=3e-9, atol=2e-12)
    volume._assert_close(report["average_strain1"], means1, "actual zero-slip average coefficient1", rtol=3e-9, atol=2e-12)
    for name in ("physical_mechanical_work_j", "net_constitutive_driver_work_j"):
        volume._assert_close(report[name], physical_work, "actual zero-slip reported "+name, rtol=2e-7, atol=1024.)
    traction = volume.prescribed_traction(X, faces, mass, density, geometry["cohorts"],
        after_packet["source_packet"]["current_geometry"]["radius_km"], shell)
    D, F = volume.basal_drag_and_force(X, faces, mass, density, geometry["cohorts"], inputs["basal_drag_pa_s_m"], traction[faces])
    M = volume.mass_matrix(X, faces, mass, density, geometry["cohorts"])
    dt = (after["geometry"]["time_myr"]-before["geometry"]["time_myr"])*SECONDS_PER_MYR
    residual = 2*(M@(dq-dt*volume._array(before["velocity_m_s"])))/(dt*dt)+D@dq/dt+force-F
    relative = float(np.linalg.norm(residual)/max(np.linalg.norm(force), np.linalg.norm(F), 1.))
    volume._assert_close(report["coupled_momentum_relative_residual"], relative,
        "actual inactive slip full-coordinate force residual", rtol=3e-3, atol=3e-10)
    energy_scale = max(before["mechanics"]["state"]["elastic_energy_j"], after["mechanics"]["state"]["elastic_energy_j"], 1.)
    if abs(report["work_identity_residual_j"]) > 5e-12*energy_scale:
        raise ValueError("Independent inactive slip report lost its work identity")
    return dict(full_coordinate_momentum_relative_residual=relative,
        independently_integrated_physical_mechanical_work_j=physical_work,
        all_zero_slip_fields_and_full_average_memory_bound=True,
        frozen_interval_plane_inactive=True, through_depth_finite_transport_claimed=False)


def audit_actual_interval(before_packet, after_packet):
    """Audit a real source continuation which has not reached an active plane.

    This scope deliberately does not use the synthetic material experiment as
    proof of an activated planetary trajectory.  The complete numerical
    volume/predictor/damage tests apply directly to the genuine raw packets.
    """
    before, after = before_packet["runtime_state"], after_packet["runtime_state"]
    actual_state_binding(before_packet)
    actual_state_binding(after_packet)
    if volume._canonical(before_packet["source_packet"]) != volume._canonical(after_packet["source_packet"]):
        raise ValueError("Independent actual slip continuation changed its complete formation source")
    faults.declared_drag_binding(before, after)
    count = before["fault_history"]["accepted_steps"]+1
    if (after["fault_history"]["accepted_steps"] != count
            or after["accepted_step_sizes_myr"][:-1] != before["accepted_step_sizes_myr"]
            or after["last_report"]["requested_dt_myr"] != after["accepted_step_sizes_myr"][-1]):
        raise ValueError("Independent actual slip interval changed its accepted-step journal")
    report = after["last_report"]["fault_evolution"]
    if (report["format"] != faults.HISTORY_FORMAT
            or report["birth_source_identity"] != after_packet["source_packet"]["birth_source_identity"]
            or report["before_time_myr"] != before["fault_history"]["time_myr"]
            or report["after_time_myr"] != after["fault_history"]["time_myr"]
            or report["dt_myr"] != report["after_time_myr"]-report["before_time_myr"]):
        raise ValueError("Independent actual fault report lost source/schema/shared-clock binding")
    for key in ("sharp_banks_created", "physical_fault_contact_admission_claimed",
            "combined_heat_mechanical_energy_closed", "combined_heat_mechanical_reservoir_closed",
            "physical_banks_admitted"):
        if report[key] is not False:
            raise ValueError("Independent actual slip continuation makes an unsupported claim: "+key)
    if report["source_spherical_filter_equals_current_chord_filter_claimed"] is not False:
        raise ValueError("Independent actual fault report equated different nonlocal geometries")
    for key in ("activation_gate_checked_at_next_start", "source_fault_work_preserved",
                "mechanical_damage_energy_accounted_by_full_cohort_caller"):
        if report[key] is not True:
            raise ValueError("Independent actual fault report removed its closure declaration: "+key)
    if (report["future_volume_plastic_slip_implemented"] is not True
            or report["damage_coupling"] != faults.DAMAGE_COUPLING
            or report["stress_scope"] != faults.STRESS_SCOPE
            or report["nonlocal_geometry_closure"] != faults.NONLOCAL_GEOMETRY
            or report["source_checkpoint_sha256"] != faults.SOURCE_SHA256):
        raise ValueError("Independent actual slip interval changed its numerical/source closure")
    for key in ("static_equilibrium_claimed", "old_C5_to_C1_admission_claimed", "full_historical_3d_recovery_claimed"):
        if after["last_report"][key] is not False:
            raise ValueError("Independent actual slip report makes an unsupported claim: "+key)
    predictor = faults.predictor_binding(before_packet, after_packet)
    source = after_packet["source_packet"]
    params = source["source"]["parameters"]["faults"]
    ready = ((np.asarray(before["fault_history"]["damage"]) >= params["activation_damage"])
        & (np.asarray(before["fault_history"]["fault_candidate_age_myr"]) >= params["activation_persistence_myr"]))
    if params["enabled"] and before["fault_history"]["membrane_established"] and ready.any():
        raise ValueError("Actual inactive source continuation ignored a ready weak-plane gate")
    expected, damage_report = faults.independent_damage_law(before["fault_history"],
        report["predictor_driver"], source, before["thermal"]["orbit"], after["thermal"]["orbit"],
        after["fault_history"]["time_myr"])
    final_tensile = faults.final_stress_binding(after_packet)
    expected["peak_tensile_pa"] = np.maximum(expected["peak_tensile_pa"], final_tensile)
    for name, value in expected.items():
        volume._assert_close(after["fault_history"][name], value, "actual source-bound fault law "+name,
            rtol=3e-10, atol=1e-5 if name == "peak_tensile_pa" else 1e-12)
    for name, value in damage_report.items():
        volume._assert_close(report[name], value, "actual fault report "+name, rtol=3e-10, atol=1e-12)
    availability = np.asarray(report["mean_solid_temperature_available"])
    if (availability.dtype.kind != "b"
            or not np.array_equal(availability, report["predictor_driver"]["active_material"])):
        raise ValueError("Independent actual slip temperature availability differs from live material")
    first = before["fault_history"]["first_fracture_time_myr"]
    if first is None and np.any(np.asarray(report["predictor_driver"]["active_material"], bool)
            & (expected["damage"] >= source["source"]["parameters"]["shell"]["damage_threshold"])):
        first = after["fault_history"]["time_myr"]
    if after["fault_history"]["first_fracture_time_myr"] != first:
        raise ValueError("Independent actual diffuse fracture event clock differs")
    result = volume.audit_interval(faults.mechanical_view(before_packet), faults.mechanical_view(after_packet))
    volume._assert_close(after["last_report"]["net_constitutive_driver_work_j"],
        result["quadrature"]["mechanical_work_j"], "actual reported zero-slip net driver work", rtol=2e-7, atol=1024.)
    for name in ("fault_friction_dissipation_j", "fault_viscous_dissipation_j"):
        if type(after["last_report"][name]) not in (int, float) or after["last_report"][name] != 0:
            raise ValueError("Independent actual top-level report invented slip dissipation: "+name)
    for name in ("mechanical_work_j", "thermal_work_j", "viscous_dissipation_j",
                 "damage_energy_change_j", "remelt_energy_removed_j"):
        delta = after["mechanics"]["state"][name]-before["mechanics"]["state"][name]
        volume._assert_close(delta, result["quadrature"][name], "actual cumulative volume "+name,
            rtol=3e-7, atol=8192.)
    volume._assert_close(report["damage_energy_change_j"], result["quadrature"]["damage_energy_change_j"],
        "actual signed full-cohort damage energy", rtol=3e-7, atol=8192.)
    final_slip = inactive_slip_report_binding(before_packet, after_packet, after["last_report"]["weak_plane_slip"],
        after["displacement_m"], after["fault_history"]["damage"], before["fault_history"]["water_access"],
        after["fault_history"]["water_access"])
    if after["last_report"]["weak_plane_slip"]["activation_event"] is not None:
        raise ValueError("Actual inactive source fabricated a weak-plane activation event")
    driver = report["predictor_driver"]
    predictor_slip = inactive_slip_report_binding(before_packet, after_packet, driver["weak_plane_slip"],
        driver["displacement_m"], driver["damage"], before["fault_history"]["water_access"],
        before["fault_history"]["water_access"])
    return {**result, "predictor": predictor,
        "fault_evolution": {"independently_recomputed_water_damage_persistence_and_tides": True,
            "maximum_damage": damage_report["max_damage"],
            "actual_active_weak_plane_count": 0, "actual_slip_work_j": 0.,
            "all_3d_inactive_planes_remain_exact_zero": True,
            "peak_includes_predictor_and_final_accepted_stress": True},
        "slip_work_closure": WORK_CLOSURE,
        "accepted_inactive_slip_report": final_slip, "private_predictor_inactive_slip_report": predictor_slip,
        "active_synthetic_fixture_substituted_for_real_formation_history": False}


def audit_directory(directory):
    directory = Path(directory)
    snapshots, bindings = [], []
    for path in sorted(directory.glob("*.json")):
        raw = json.loads(path.read_text())
        if raw.get("format") != FORMAT:
            continue
        if (set(raw) != {"format", "scope", "source_packet", "source_checkpoint_sha256", "runtime_state", "checkpoint_sha256"}
                or raw["scope"] != SCOPE):
            raise ValueError("Independent actual slip checkpoint schema/scope changed")
        body = {key: value for key, value in raw.items() if key != "checkpoint_sha256"}
        if hashlib.sha256(volume._canonical(body)).hexdigest() != raw["checkpoint_sha256"]:
            raise ValueError("Independent actual slip checkpoint seal mismatch")
        bindings.append({"name": path.name, **actual_state_binding(raw)})
        snapshots.append((raw["runtime_state"]["mechanics"]["state"]["accepted_steps"], path, raw))
    if not snapshots:
        raise ValueError("No authentic Genesis volume-slip checkpoints found")
    grouped = {}
    for count, path, raw in snapshots:
        grouped.setdefault(count, []).append((path, raw))
    if sorted(grouped) != list(range(max(grouped)+1)) or max(grouped) == 0:
        raise ValueError("Independent actual slip audit needs all endpoints and an accepted interval")
    restarts = []
    for count, rows in grouped.items():
        for path, raw in rows[1:]:
            if volume._canonical(raw) != volume._canonical(rows[0][1]):
                raise ValueError("Complete direct/restarted actual slip checkpoints differ")
            restarts.append(dict(accepted_steps=count, first=rows[0][0].name,
                second=path.name, complete_json_equal=True))
    initial = volume.audit_initial(grouped[0][0][1])
    intervals = [audit_actual_interval(grouped[i-1][0][1], grouped[i][0][1]) for i in range(1, max(grouped)+1)]
    report = dict(format="genesis-volume-slip-actual-independent-audit-0.1", status="PASS",
        producer_imports=False, source_checkpoint_sha256=faults.SOURCE_SHA256,
        quadrature="Cartesian_Duffy4x4_Gauss32_massq_Gauss32_adaptive_time",
        input_files=[dict(name=path.name, raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for _, path, _ in snapshots],
        literal_formation_and_fault_source_bindings=bindings, initial_source_audit=initial,
        intervals=intervals, restart_pairs=restarts,
        actual_weak_plane_activation_observed=False, synthetic_material_fixtures_substituted_for_actual_source=False,
        sharp_banks_created=False, physical_fault_contact_admission_claimed=False,
        old_C5_to_C1_admission_claimed=False, original_point_1_completed=False,
        global_heat_and_mechanical_reservoir_closed=False)
    return {**report, "report_sha256": hashlib.sha256(volume._canonical(report)).hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--material-fixture", type=Path)
    group.add_argument("--material-directory", type=Path)
    group.add_argument("--actual-directory", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.material_fixture is not None:
        fixture_bytes = arguments.material_fixture.read_bytes()
        report = audit_material_fixture(json.loads(fixture_bytes))
        body = {key: value for key, value in report.items() if key != "report_sha256"}
        body["input_fixture_sha256"] = hashlib.sha256(fixture_bytes).hexdigest()
        report = {**body, "report_sha256": hashlib.sha256(volume._canonical(body)).hexdigest()}
    elif arguments.material_directory is not None:
        report = audit_material_directory(arguments.material_directory)
    else:
        report = audit_directory(arguments.actual_directory)
    encoded = json.dumps(report, sort_keys=True, indent=2, allow_nan=False)+"\n"
    if arguments.output is not None:
        arguments.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
