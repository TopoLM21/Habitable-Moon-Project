"""Numerical and forgery controls of the producer-free fault volume audit."""
import ast
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from analysis.genesis_volume_fault_validation import independent_audit as audit


REPOSITORY = Path(__file__).resolve().parents[1]
SEALED_INITIAL = REPOSITORY/"analysis/genesis_volume_continuation_validation/evidence/control/initial.json"


@pytest.fixture(scope="module")
def initial_fault_packet():
    """Literal prior checked initial state, with separately decoded fault history."""
    packet = json.loads(SEALED_INITIAL.read_text())
    metadata, arrays, _ = audit.formation_fault_data(packet)
    packet["format"] = audit.FORMAT
    packet["scope"] = audit.SCOPE
    source = packet["source_packet"]
    history = {name: value.tolist() for name, value in arrays.items()}
    history.update(format=audit.HISTORY_FORMAT, source_checkpoint_sha256=audit.SOURCE_SHA256,
        source_identity=source["birth_source_identity"], initial_time_myr=source["time_myr"],
        time_myr=source["time_myr"], accepted_steps=0,
        membrane_established=metadata["state"]["membrane_established"],
        first_fracture_time_myr=metadata["state"]["first_fracture_time_myr"],
        friction_work_j=metadata["state"]["friction_work_j"],
        viscous_fault_work_j=metadata["state"]["viscous_fault_work_j"])
    packet["runtime_state"].update(fault_history=history, damage_coupling=audit.DAMAGE_COUPLING)
    return packet


def test_independent_modules_do_not_import_producer_physics():
    paths = (Path(audit.__file__), Path(audit.volume.__file__))
    for path in paths:
        for node in ast.walk(ast.parse(path.read_text())):
            names = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            assert not any(name == "tectonics" or name.startswith("tectonics.") for name in names)


def test_all_original_inactive_fault_fields_are_bound(initial_fault_packet):
    report = audit.inherited_fault_binding(initial_fault_packet)
    assert report["literal_inactive_plane_slip_and_work_preserved"]
    assert report["source_active_plane_count"] == 0
    assert report["original_birth_prefix_preserved"]


@pytest.mark.parametrize("field,value", [
    ("cumulative_shear", 1e-8), ("activation_time_myr", 0.),
    ("normal_stress_pa", 3.), ("viscous_work_cell_j", 1.),
])
def test_inactive_history_forgery_is_refused(initial_fault_packet, field, value):
    packet = copy.deepcopy(initial_fault_packet)
    packet["runtime_state"]["fault_history"][field][0] = value
    with pytest.raises(ValueError, match="reset or forged"):
        audit.inherited_fault_binding(packet)


def test_thermal_baseline_damage_cannot_be_replaced_by_runtime_damage(initial_fault_packet):
    packet = copy.deepcopy(initial_fault_packet)
    packet["runtime_state"]["thermal"]["damage"][0] += 1e-6
    with pytest.raises(ValueError, match="literal thermal source damage"):
        audit.inherited_fault_binding(packet)


def test_birth_prefix_forgery_is_refused(initial_fault_packet):
    packet = copy.deepcopy(initial_fault_packet)
    packet["runtime_state"]["history"]["cohorts"][0]["birth_temperature_k"] += 2
    with pytest.raises(ValueError, match="literal Genesis material cohort"):
        audit.inherited_fault_binding(packet)


def test_runtime_thermal_source_identity_cannot_be_replaced(initial_fault_packet):
    packet = copy.deepcopy(initial_fault_packet)
    packet["runtime_state"]["thermal"]["source_identity"] = "0"*64
    with pytest.raises(ValueError, match="runtime thermal/mechanical source context"):
        audit.inherited_fault_binding(packet)


def test_current_top_axes_project_world_tensor_without_resetting_memory():
    X = np.eye(3)*1e5
    faces = np.array(((0, 1, 2),))
    frame, _ = audit.physical_face_geometry(X, faces)
    frame = frame[0]
    angle = .4
    local_rotation = np.array(((np.cos(angle), -np.sin(angle), 0.),
                              (np.sin(angle), np.cos(angle), 0.), (0., 0., 1.)))
    R = frame@local_rotation@frame.T
    current = X@R.T
    shell = dict(young_modulus_pa=6e10, poisson_ratio=.25, residual_stiffness=.02)
    e = np.array((1e-4, -2e-5, 0., 3e-5, 0., 0.))
    e0 = np.tile(e, (2, 3, 1)); e0[1] *= 2
    cohorts = [dict(id=1, face_index=0, q_lo=.01, q_hi=.02),
               dict(id=2, face_index=0, q_lo=.02, q_hi=.04)]
    result = audit.face_stress_from_coefficients(X, faces, np.array((1e12,)), 3000., cohorts,
        e0, np.zeros_like(e0), np.array((900., 1200.)), np.array((.1,)), current, shell)
    stress = audit.volume.isotropic_matrix(6e10, .25, .1, .02)@(e*5/3)
    tensor = np.array(((stress[0], stress[3], stress[4]), (stress[3], stress[1], stress[5]),
                       (stress[4], stress[5], stress[2])))
    expected = local_rotation.T@tensor@local_rotation
    np.testing.assert_allclose(result["stress_pa"][0], (expected[0, 0], expected[1, 1], expected[0, 1]), rtol=3e-15)
    assert result["mean_temperature_k"][0] == pytest.approx(1100., rel=2e-15)
    assert not np.allclose(result["stress_pa"][0], stress[[0, 1, 3]], rtol=.01)


def test_radial_stress_mean_preserves_exact_material_volume_weight():
    X = np.eye(3)*1e5
    faces = np.array(((0, 1, 2),))
    mass, rho, lo, hi = 2e17, 3000., .1, .4
    coefficient = 6*mass/(rho*np.linalg.det(X.T))
    inverse_mean = 3/(2*coefficient)*((1-coefficient*lo)**(2/3)-(1-coefficient*hi)**(2/3))/(hi-lo)
    shell = dict(young_modulus_pa=6e10, poisson_ratio=.25, residual_stiffness=.02)
    e1 = np.tile((1e-4, 2e-5, 0., 0., 0., 0.), (1, 3, 1))
    result = audit.face_stress_from_coefficients(X, faces, np.array((mass,)), rho,
        [dict(id=1, face_index=0, q_lo=lo, q_hi=hi)], np.zeros_like(e1), e1, np.array((1000.,)), np.zeros(1), X, shell)
    expected = audit.volume.isotropic_matrix(6e10, .25, 0., .02)@(e1[0, 0]*inverse_mean)
    np.testing.assert_allclose(result["stress_pa"][0], expected[[0, 1, 3]], rtol=2e-14, atol=1e-8)


def test_retained_mass_dilution_counts_newbirth_and_remelt_directly():
    old = [dict(id=1, face_index=0, q_lo=0., q_hi=.3), dict(id=2, face_index=1, q_lo=0., q_hi=.4)]
    new = [dict(id=1, face_index=0, q_lo=0., q_hi=.2), dict(id=3, face_index=0, q_lo=.2, q_hi=.4)]
    np.testing.assert_array_equal(audit.retained_fraction(old, new, np.array((2e12, 5e12))), (.5, 0.))


def test_nonlocal_constant_and_current_area_conservation(initial_fault_packet):
    source = initial_fault_packet["source_packet"]
    X, faces = source["current_geometry"]["vertices_unit_xyz"], source["current_geometry"]["faces"]
    n = len(faces)
    area = np.linspace(1e6, 2e6, n)
    active = np.ones(n, bool)
    constant = audit.nonlocal_loading(X, faces, area, active, np.full(n, 7.), 800.)
    np.testing.assert_allclose(constant, 7., rtol=7e-16)
    raw = np.arange(n, dtype=float)**2
    loading = audit.nonlocal_loading(X, faces, area, active, raw, 800.)
    assert area@loading == pytest.approx(area@raw, rel=2e-15)
    assert loading.min() >= 0
    spherical_mass = np.ones(n)*area.mean()
    spherical_result = audit.nonlocal_loading(X, faces, spherical_mass, active, raw, 800.)
    assert np.linalg.norm(loading-spherical_result) > 1.


def test_nonlocal_inactive_faces_are_no_flux_boundaries(initial_fault_packet):
    source = initial_fault_packet["source_packet"]
    X, faces = source["current_geometry"]["vertices_unit_xyz"], source["current_geometry"]["faces"]
    n = len(faces)
    active = np.zeros(n, bool); active[0] = True
    raw = np.arange(n, dtype=float)+2
    loading = audit.nonlocal_loading(X, faces, np.ones(n)*1e6, active, raw, 800.)
    np.testing.assert_array_equal(loading[1:], np.zeros(n-1))
    assert loading[0] == 2.


def test_tidal_hessian_uses_current_face_axes_and_both_orbit_endpoints():
    X = np.array(((1., 0., 1.), (2., 0., 1.), (1., 1., 1.)))
    faces = np.array(((0, 1, 2),))
    p = dict(enabled=True, primary_mass_kg=9e27, satellite_radius_km=5287., surface_gravity_m_s2=7.,
             love_h2=.6, love_l2=.08, phase_samples=4)
    orbit0 = dict(eccentricity=.001, semimajor_axis_km=800000.)
    orbit1 = dict(eccentricity=.003, semimajor_axis_km=1000000.)
    cycle = audit.current_tidal_strain(X, faces, p, orbit0, orbit1)
    scale = 6.67430e-11*p["primary_mass_kg"]*(p["satellite_radius_km"]*1000)/(p["surface_gravity_m_s2"]*(900000.*1000)**3)*.002
    np.testing.assert_allclose(cycle[0, 0], scale*np.array((2*.08*3+(.6-2*.08)*(-1.5),
        2*.08*(-1.5)+(.6-2*.08)*(-1.5), 0.)), rtol=2e-15)
    np.testing.assert_allclose(cycle[1, 0], scale*np.array((0., 0., 12*.08)), rtol=2e-15, atol=1e-20)
    np.testing.assert_allclose(cycle[2], -cycle[0], rtol=2e-15, atol=1e-20)


def law_control(initial_fault_packet, surface, ocean):
    source = copy.deepcopy(initial_fault_packet["source_packet"])
    source["source"]["parameters"]["tides"]["enabled"] = False
    n = len(source["current_geometry"]["faces"])
    before = dict(time_myr=1., membrane_established=True, damage=np.full(n, .2), water_access=np.full(n, .4),
        fault_candidate_age_myr=np.full(n, .1), weak_duration_myr=np.full(n, .2), peak_tensile_pa=np.zeros(n))
    driver = dict(retained_fraction=np.full(n, .75), active_material=np.ones(n, bool),
        surface_temperature_k=surface, ocean_fraction=ocean, stress_pa=np.zeros((n, 3)), mean_temperature_k=np.full(n, 700.),
        physical_vertices_m=np.asarray(source["current_geometry"]["vertices_unit_xyz"])*5e6)
    orbit = dict(eccentricity=.001, semimajor_axis_km=800000.)
    return before, driver, source, orbit


def test_hot_steam_dilutes_and_loses_water_without_adding_ocean(initial_fault_packet):
    before, driver, source, orbit = law_control(initial_fault_packet, 800., 0.)
    result, report = audit.independent_damage_law(before, driver, source, orbit, orbit, 1.01)
    loss = source["source"]["parameters"]["onset"]["loss_timescale_myr"]
    np.testing.assert_allclose(result["water_access"], .4*.75*np.exp(-.01/loss), rtol=2e-15)
    cold = source["source"]["parameters"]["shell"]["cold_healing_timescale_myr"]
    np.testing.assert_allclose(result["damage"], .2*.75*np.exp(-.01/cold), rtol=2e-15)
    assert np.max(report["raw_loading_per_myr"]) == 0.
    np.testing.assert_array_equal(result["fault_candidate_age_myr"], np.zeros(len(result["damage"])))


def test_liquid_water_weakening_uses_retained_damage_permeability(initial_fault_packet):
    before, driver, source, orbit = law_control(initial_fault_packet, 600., .005)
    result, _ = audit.independent_damage_law(before, driver, source, orbit, orbit, 1.01)
    onset = source["source"]["parameters"]["onset"]
    rate = (onset["intact_permeability_fraction"]+(1-onset["intact_permeability_fraction"])*.15**2)/onset["access_timescale_myr"]*.5
    np.testing.assert_allclose(result["water_access"], 1-(1-.3)*np.exp(-rate*.01), rtol=2e-15)


def test_inactive_material_has_no_constitutive_temperature_loading_or_timer(initial_fault_packet):
    before, driver, source, orbit = law_control(initial_fault_packet, 800., 0.)
    driver["active_material"][:] = False
    driver["mean_temperature_k"][:] = 0.
    driver["retained_fraction"][:] = 0.
    driver["stress_pa"][:] = 1e11
    source["source"]["parameters"]["faults"]["enabled"] = False
    result, report = audit.independent_damage_law(before, driver, source, orbit, orbit, 1.01)
    for name in ("damage", "water_access", "fault_candidate_age_myr", "peak_tensile_pa", "tidal_stress_mpa"):
        np.testing.assert_array_equal(result[name], np.zeros_like(result[name]))
    for name in ("raw_loading_per_myr", "nonlocal_loading_per_myr", "healing_per_myr"):
        np.testing.assert_array_equal(report[name], np.zeros_like(report[name]))


def test_missing_temperature_is_refused_on_live_material(initial_fault_packet):
    before, driver, source, orbit = law_control(initial_fault_packet, 800., 0.)
    driver["mean_temperature_k"][0] = 0.
    with pytest.raises(ValueError, match="actual temperatures on active material"):
        audit.independent_damage_law(before, driver, source, orbit, orbit, 1.01)


def test_tensile_loading_matches_symmetric_tensor_eigenvalues():
    stress = np.array(((4., -2., 3.), (-5., -3., .2), (0., 0., 2.)))
    expected = [max(0., np.linalg.eigvalsh(((xx, xy), (xy, yy)))[-1]) for xx, yy, xy in stress]
    np.testing.assert_allclose(audit.tensile_principal(stress), expected, rtol=2e-15)


def test_mechanical_damage_view_is_private_and_leaves_literal_thermal_untouched(initial_fault_packet):
    packet = copy.deepcopy(initial_fault_packet)
    packet["runtime_state"]["fault_history"]["damage"][0] += .001
    raw_thermal = copy.deepcopy(packet["runtime_state"]["thermal"])
    view = audit.mechanical_view(packet)
    assert packet["runtime_state"]["thermal"] == raw_thermal
    assert view["runtime_state"]["thermal"]["damage"][0] != raw_thermal["damage"][0]
    assert view["runtime_state"]["fault_history"] == packet["runtime_state"]["fault_history"]


def test_initial_only_archive_cannot_be_reported_as_advanced_control(initial_fault_packet, tmp_path):
    raw = copy.deepcopy(initial_fault_packet)
    body = {key: value for key, value in raw.items() if key != "checkpoint_sha256"}
    raw["checkpoint_sha256"] = hashlib.sha256(audit.volume._canonical(body)).hexdigest()
    (tmp_path/"initial.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="actually advanced interval"):
        audit.audit_directory(tmp_path)


def test_tidal_disabled_control_is_exact_zero():
    X = np.eye(3)*1e5
    p = dict(enabled=False, phase_samples=8)
    orbit = dict(eccentricity=.01, semimajor_axis_km=800000.)
    np.testing.assert_array_equal(audit.current_tidal_strain(X, [[0, 1, 2]], p, orbit, orbit), np.zeros((8, 1, 3)))


def test_reported_basal_drag_cannot_replace_the_declared_physical_parameter():
    before = dict(contact_parameters_json=json.dumps(dict(basal_drag_pa_s_m=1e14)))
    after = {**before, "last_report": {"audit_input": {"basal_drag_pa_s_m": 1e14}}}
    audit.declared_drag_binding(before, after)
    forged = copy.deepcopy(after)
    forged["last_report"]["audit_input"]["basal_drag_pa_s_m"] *= 2
    with pytest.raises(ValueError, match="actual declared basal drag"):
        audit.declared_drag_binding(before, forged)
    forged = copy.deepcopy(after)
    forged["contact_parameters_json"] = json.dumps(dict(basal_drag_pa_s_m=2e14))
    with pytest.raises(ValueError, match="actual contact parameters"):
        audit.declared_drag_binding(before, forged)


@pytest.fixture(scope="module")
def independently_calculated_real_predictor(initial_fault_packet):
    """Separate NumPy solve on a sealed actual thermal interval; no Genesis rerun."""
    before = copy.deepcopy(initial_fault_packet)
    after = json.loads((SEALED_INITIAL.parent/"after_first.json").read_text())
    b, a = before["runtime_state"], after["runtime_state"]
    source, geometry = after["source_packet"], a["geometry"]
    v = audit.volume
    X, faces = np.asarray(geometry["vertices_m"]), np.asarray(geometry["faces"], int)
    mass, rho = np.asarray(geometry["column_mass_kg"]), geometry["density_kg_m3"]
    cohorts = geometry["cohorts"]
    shell = source["source"]["parameters"]["shell"]
    retained = audit.retained_fraction(b["geometry"]["cohorts"], cohorts, mass)
    damage = np.asarray(b["fault_history"]["damage"])*retained
    dt = (geometry["time_myr"]-b["geometry"]["time_myr"])*365.25*86400*1e6
    inputs = a["last_report"]["audit_input"]
    ndof = len(b["displacement_m"])
    tangent, average_initial = np.zeros((ndof, ndof)), np.zeros(ndof)
    N, angular_weights = v.angular_rule()
    all_B, all_dofs = [], []
    for i, c in enumerate(cohorts):
        face = int(c["face_index"])
        frame, _ = audit.physical_face_geometry(X, faces[face][None, :])
        frame = frame[0]
        B0, B1 = v.independent_strain_operators(X[faces[face]], frame, N)
        q, radial_weights = v.radial_rule(c["q_lo"], v._high(c))
        lam, _ = v.conical_scales(X[faces[face]], mass[face], rho, q)
        B = (B0[:, None, :, :]+B1[:, None, :, :]/lam[None, :, None, None]).reshape(-1, 6, 18)
        weights = (angular_weights[:, None]*radial_weights[None, :]).ravel()
        volume = mass[face]*(v._high(c)-c["q_lo"])/rho
        C = v.isotropic_matrix(shell["young_modulus_pa"], shell["poisson_ratio"], damage[face], shell["residual_stiffness"])
        theta = v.temporal_factors(inputs["maxwell_ratio"][i])[1]
        stiffness = volume*theta*np.einsum("p,pai,ab,pbj->ij", weights, B, C, B)
        dofs = (6*faces[face, :, None]+np.arange(6)).ravel()
        tangent[np.ix_(dofs, dofs)] += stiffness
        result = v.quadrature_cohort(X[faces[face]], frame, mass[face], rho, c, np.zeros(18),
            inputs["old0"][i], inputs["old1"][i], inputs["thermal0"][i], inputs["maxwell_ratio"][i], C)
        np.add.at(average_initial, dofs, result["average_force_n"])
        all_B.append(v.independent_strain_operators(X[faces[face]], frame, v.SOURCE_ANGULAR_POINTS))
        all_dofs.append(dofs)
    M = v.mass_matrix(X, faces, mass, rho, cohorts)
    traction = v.prescribed_traction(X, faces, mass, rho, cohorts, source["current_geometry"]["radius_km"], shell)
    D, F = v.basal_drag_and_force(X, faces, mass, rho, cohorts, inputs["basal_drag_pa_s_m"], traction[faces])
    tangent += 2*M/dt**2+D/dt
    scale = 1/np.sqrt(np.diag(tangent))
    rhs = F-average_initial+2*(M@np.asarray(b["velocity_m_s"]))/dt
    dq = scale*np.linalg.solve(tangent*scale[:, None]*scale[None, :], scale*rhs)
    q1 = np.asarray(b["displacement_m"])+dq
    e0, e1 = [], []
    for i, (operators, dofs) in enumerate(zip(all_B, all_dofs)):
        ratio = inputs["maxwell_ratio"][i]
        beta = 1. if ratio == 0 else -np.expm1(-ratio)/ratio
        e0.append(np.exp(-ratio)*np.asarray(inputs["old0"][i])+beta*(operators[0]@dq[dofs]-inputs["thermal0"][i]))
        e1.append(np.exp(-ratio)*np.asarray(inputs["old1"][i])+beta*(operators[1]@dq[dofs]))
    temperatures = [c["temperature_k"] for c in a["mechanics"]["state"]["cohorts"]]
    vertices = X+q1.reshape(-1, 6)[:, :3]
    response = audit.face_stress_from_coefficients(X, faces, mass, rho, cohorts, e0, e1,
        temperatures, damage, vertices, shell)
    _, surface = v.global_temperatures(a["thermal"]["global_state"], source["source"]["parameters"]["thermal"])
    driver = dict(physical_vertices_m=vertices.tolist(), stress_pa=response["stress_pa"].tolist(),
        mean_temperature_k=response["mean_temperature_k"].tolist(), retained_fraction=retained.tolist(),
        active_material=response["active_material"].tolist(), surface_temperature_k=surface,
        ocean_fraction=audit.ocean_fraction(surface, source["source"]["parameters"]["thermal"]),
        elastic_strain0=np.asarray(e0).tolist(), elastic_strain1=np.asarray(e1).tolist(),
        displacement_m=q1.tolist(), damage=damage.tolist())
    a["last_report"]["fault_evolution"] = dict(predictor_driver=driver)
    return before, after


def test_real_predictor_force_binding_refuses_forged_current_stress(independently_calculated_real_predictor):
    before, after = independently_calculated_real_predictor
    report = audit.predictor_binding(before, after)
    assert report["full_predictor_momentum_and_work"]["relative_max_coordinate_momentum_residual"] < 1e-8
    forged = copy.deepcopy(after)
    forged["runtime_state"]["last_report"]["fault_evolution"]["predictor_driver"]["stress_pa"][0][0] += 1e6
    with pytest.raises(ValueError, match="current-axis predictor stress_pa"):
        audit.predictor_binding(before, forged)


def test_real_predictor_memory_binding_refuses_forged_cohort_strain(independently_calculated_real_predictor):
    before, after = independently_calculated_real_predictor
    forged = copy.deepcopy(after)
    forged["runtime_state"]["last_report"]["fault_evolution"]["predictor_driver"]["elastic_strain0"][0][0][0] += 1e-4
    with pytest.raises(ValueError, match="predictor constant coefficient memory"):
        audit.predictor_binding(before, forged)


def test_final_accepted_stress_diagnostic_is_independently_bound(independently_calculated_real_predictor):
    before, predecessor = independently_calculated_real_predictor
    packet = copy.deepcopy(predecessor)
    state = packet["runtime_state"]
    state["fault_history"] = copy.deepcopy(before["runtime_state"]["fault_history"])
    geometry = state["geometry"]
    X = np.asarray(geometry["vertices_m"])
    q = np.asarray(state["displacement_m"]).reshape(-1, 6)
    memories = state["mechanics"]["state"]["cohorts"]
    response = audit.face_stress_from_coefficients(X, geometry["faces"], np.asarray(geometry["column_mass_kg"]),
        geometry["density_kg_m3"], geometry["cohorts"], [c["strain0"] for c in memories],
        [c["strain1"] for c in memories], [c["temperature_k"] for c in memories],
        np.asarray(state["fault_history"]["damage"]), X+q[:, :3], packet["source_packet"]["source"]["parameters"]["shell"])
    tensile = audit.tensile_principal(response["stress_pa"])
    state["last_report"]["fault_evolution"]["final_face_tensile_pa"] = tensile.tolist()
    np.testing.assert_allclose(audit.final_stress_binding(packet), tensile, rtol=0, atol=0)
    state["last_report"]["fault_evolution"]["final_face_tensile_pa"][0] += 1e6
    with pytest.raises(ValueError, match="final accepted current-axis tensile stress"):
        audit.final_stress_binding(packet)
