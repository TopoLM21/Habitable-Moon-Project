"""Independent mathematical identities; no producer or planetary initialization."""
from __future__ import annotations

import ast
import copy
from pathlib import Path

import numpy as np
import pytest

from analysis.genesis_volume_nucleation_validation import independent_audit as audit


def manufactured_parent(*, pose=True, inner=.5):
    X = np.array(((-1., 0., 1.), (1., 0., 1.), (0., 2., 1.)))
    coefficient, density = .9, 2.
    displacement = np.array(((.03, -.02, .01, -.01, .04, .02),
        (-.01, .03, .02, .03, -.02, .01), (.02, .01, -.03, -.02, .01, -.04))) if pose else np.zeros((3, 6))
    e0 = np.array(((.01, .02, -.01, .03, -.02, .04),
        (.02, -.01, .03, .02, .01, -.03), (-.01, .03, .02, -.02, .04, .01)))
    e1 = .2*e0+np.arange(18).reshape(3, 6)*.0001
    cohort = dict(id=5, face_index=0, q_lo=0., live_q_hi=(1-inner**3)/coefficient,
        inner_scale=inner, outer_scale=1., owner_id="manufactured-body", material_id="manufactured-material")
    geometry = dict(vertices_m=X.tolist(), faces=[[0, 1, 2]],
        column_mass_kg=[coefficient*density*np.linalg.det(X.T)/6], density_kg_m3=density,
        cohorts=[cohort], source_identity="manufactured_only_not_planet", time_myr=.1)
    return dict(checkpoint_sha256="manufactured_only", source_checkpoint_sha256=audit.FORMATION_SHA256,
        source_packet=dict(current_geometry=dict(face_frames_xyz=np.eye(3)[:, :2][None].tolist()),
            source=dict(parameters=dict(shell=dict(young_modulus_pa=71., poisson_ratio=.23, residual_stiffness=.02)))),
        runtime_state=dict(displacement_m=displacement.ravel().tolist(), velocity_m_s=np.zeros(18).tolist(),
            fault_history=dict(damage=[.17]), geometry=geometry, accepted_step_sizes_myr=[.0001]*4,
            mechanics=dict(state=dict(accepted_steps=4, cohorts=[dict(id=5,
                strain0=(audit.SOURCE_ANGULAR_POINTS@e0).tolist(), strain1=(audit.SOURCE_ANGULAR_POINTS@e1).tolist())]))))


def surface(parent=None, *, normal=(1., 0., 0.), offset=.1, order=8):
    return audit.embedded_quadrature(manufactured_parent() if parent is None else parent,
        normal, offset, face_indices=[0], order=order)


def plain(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, dict): return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [plain(v) for v in value]
    if isinstance(value, np.generic): return value.item()
    return value


def sealed(body, name):
    return {**body, name: audit.digest(body)}


def sample_packet(parent, surf, observer):
    result = audit.material_samples(parent, surf["cohort_ids"], surf["barycentric"], surf["mass_coordinate_q"])
    body = dict(format="genesis-volume-material-stress-samples-0.1",
        parent_checkpoint_sha256=parent["checkpoint_sha256"], formation_checkpoint_sha256=parent["source_checkpoint_sha256"],
        source_identity=parent["runtime_state"]["geometry"]["source_identity"],
        time_myr=parent["runtime_state"]["geometry"]["time_myr"], observer_binding_sha256=observer["binding_sha256"],
        supplied_normal_is_fracture_orientation=False, normal_world=None, traction_world_pa=None,
        **audit.stress_claims(), **plain(result))
    return sealed(body, "sample_sha256")


def surface_packet(parent, surf):
    body = plain(surf)
    body.update(format="genesis-original-material-embedded-surface-trace-0.1",
        scope="explicit_current_plane_material_quadrature_paired_virtual_trace_no_physical_insertion",
        source_checkpoint_sha256=audit.FORMATION_SHA256,
        source_identity=parent["runtime_state"]["geometry"]["source_identity"],
        time_myr=parent["runtime_state"]["geometry"]["time_myr"], source_geometry_bound=True,
        declared_normal_xyz=[1., 0., 0.], declared_offset_m=.1,
        declared_plane_input_semantics="literal_numeric_constructor_coefficients_before_normalization",
        material_sections_binding_sha256="0"*64,
        quadrature_rule="tensor_Gauss_on_analytic_lambda_pieces_and_Duffy_triangle",
        original_parent_modes="A_xyz_and_P_xyz_metres_per_original_vertex",
        paired_trace_scope="two_virtual_parent_fields_not_created_or_released_mesh_DOFs",
        material_support_semantics="birth_pullback_N_q_frozen_not_world_plane_reintersection",
        area_conventions="current_and_reference_separate_interface_counted_once",
        support_scope="relative_interior_cohort_sheets_boundary_only_support_has_zero_area",
        numerical_integration_scope="approximate_geometry_integrals_exact_discrete_trace_work_at_declared_order",
        geometry_integrals_exact=False,
        recorded_metrics_semantics="fixed_declared_plane_pullback_snapshot_not_transported_after_follow",
        total_current_area_m2=float(surf["current_area_weights_m2"].sum()),
        total_reference_area_m2=float(surf["reference_area_weights_m2"].sum()),
        creation_work_debited_from_Genesis_j=0., **{k: False for k in audit.SURFACE_FALSE})
    return sealed(body, "embedded_surface_sha256")


def test_oracle_has_no_producer_imports_dynamic_loading_or_replay_calls():
    source = Path(audit.__file__).read_text(); tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(not name.name.startswith("tectonics") for name in node.names)
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("tectonics")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in {"__import__", "eval", "exec", "load_volume_slip_checkpoint"}


def test_spectral_envelope_uses_full_shear_not_diagonal_tension():
    stress = np.array(((-10e6, 20e6, 0), (20e6, -10e6, 0), (0, 0, -10e6)))
    result = audit.nucleation_criterion(stress, audit.DECLARED_LAW)
    np.testing.assert_allclose(result["principal_stress_pa"], [-30e6, -10e6, 10e6])
    assert result["yield_score"] > 0
    assert result["tensile_score"] < 0


def test_spectral_envelope_is_objective_under_proper_rotation():
    angle = .51
    Q = np.array(((np.cos(angle), -np.sin(angle), 0), (np.sin(angle), np.cos(angle), 0), (0, 0, 1)))
    S = np.array(((8e6, 3e6, 1e6), (3e6, -5e6, -2e6), (1e6, -2e6, 2e6)))
    first = audit.nucleation_criterion(S, audit.DECLARED_LAW)
    rotated = audit.nucleation_criterion(Q@S@Q.T, audit.DECLARED_LAW)
    for name in first: np.testing.assert_allclose(first[name], rotated[name], rtol=3e-15, atol=1e-8)


def test_classical_unclamped_Coulomb_convex_envelope_coefficients():
    result = audit.nucleation_criterion(np.diag((20e6, -2e6, -8e6)), audit.DECLARED_LAW)
    mu = .6; h = np.hypot(1., mu)
    expected = ((h+mu)*20e6-(h-mu)*(-8e6))/(2*4e6)-1
    assert result["shear_score"] == pytest.approx(expected, rel=4e-16)
    assert result["tensile_score"] == pytest.approx(20/12-1)


def test_six_corner_hull_certifies_convex_maximum_of_saved_material_field():
    parent = manufactured_parent(); corners = audit.saved_corner_catalogue(parent, audit.DECLARED_LAW)
    maximum = corners["yield_score"].max()
    c = parent["runtime_state"]["geometry"]["cohorts"][0]
    N, q = [], []
    for r in np.linspace(.03, .97, 9):
        for s in np.linspace(.03, .97, 7):
            N.append(((1-r)*(1-s), r, (1-r)*s)); q.append(c["live_q_hi"]*(r+s)/2)
    stress = audit.material_samples(parent, [5]*len(q), N, q)["stress_world_tensor"]
    assert audit.nucleation_criterion(stress, audit.DECLARED_LAW)["yield_score"].max() <= maximum+1e-14


def test_stress_fields_preserve_saved_angular_and_inverse_lambda_memory():
    parent = manufactured_parent()
    values = audit.material_samples(parent, [5, 5], [[1., 0., 0.], [0., 1., 0.]], [.2, .7])
    assert np.linalg.norm(values["stress_world_tensor"][0]-values["stress_world_tensor"][1]) > 1
    wrong = np.mean(parent["runtime_state"]["mechanics"]["state"]["cohorts"][0]["strain0"], axis=0)
    assert np.linalg.norm(values["elastic_strain_local6"][0]-wrong) > .01


def test_vertical_undeformed_cut_area_is_exact_trapezoid():
    s = surface(manufactured_parent(pose=False), offset=0.)
    assert s["current_area_weights_m2"].sum() == pytest.approx(1-.5**2, rel=8e-16)
    np.testing.assert_array_equal(s["current_area_weights_m2"], s["reference_area_weights_m2"])


def test_horizontal_interior_cut_area_is_single_triangle():
    s = surface(manufactured_parent(pose=False), normal=(0., 0., 1.), offset=.8)
    assert s["patch_kinds"] == ["constant_lambda_triangle"]
    assert s["current_area_weights_m2"].sum() == pytest.approx(2*.8**2, rel=8e-16)
    assert len(s["cohort_ids"]) == 64


@pytest.mark.parametrize("offset", (.5, 1.))
def test_closed_radial_boundary_is_not_an_interior_embedded_sheet(offset):
    s = surface(manufactured_parent(pose=False), normal=(0., 0., 1.), offset=offset)
    assert len(s["cohort_ids"]) == 0
    assert s["current_area_weights_m2"].sum() == 0


def test_shared_angular_boundary_does_not_create_paired_embedded_sheet():
    s = surface(manufactured_parent(pose=False), normal=(0., 1., 0.), offset=0.)
    assert s["selected_cohort_ids"] == [5]
    assert len(s["cohort_ids"]) == 0


def test_material_quadrature_recovers_plane_and_full_current_pose():
    parent = manufactured_parent(); s = surface(parent)
    np.testing.assert_allclose(s["current_position_m"][:, 0], .1, atol=3e-16)
    reproduced = s["reference_position_m"]+audit.point_trace(s, parent["runtime_state"]["displacement_m"])
    np.testing.assert_allclose(reproduced, s["current_position_m"], rtol=2e-15, atol=2e-16)
    pose = np.asarray(parent["runtime_state"]["displacement_m"]).reshape(3, 6)
    assert np.linalg.norm(s["parent_trace"][:, :, 3:6]) > .01
    assert np.linalg.norm(pose[:, 3:]) > .01


def test_fixed_N_q_follow_does_not_reintersect_a_new_world_plane():
    parent = manufactured_parent(); s = surface(parent)
    translated = np.asarray(parent["runtime_state"]["displacement_m"]).reshape(3, 6).copy()
    translated[:, :3] += (.02, -.01, .04)
    followed = s["reference_position_m"]+audit.point_trace(s, translated.ravel())
    np.testing.assert_allclose(followed-s["current_position_m"], np.broadcast_to([.02, -.01, .04], followed.shape), atol=3e-16)
    assert abs(np.mean(followed[:, 0])-.1) > .019


def test_full_material_Jacobians_are_positive_and_include_P():
    s = surface()
    assert np.all(s["current_volume_determinant_m3"] > 0)
    assert np.all(s["reference_volume_determinant_m3"] > 0)
    np.testing.assert_allclose(s["current_to_reference_volume_ratio"],
        np.linalg.det(s["current_volume_jacobian_m"])/np.linalg.det(s["reference_volume_jacobian_m"]), rtol=4e-16)
    assert np.max(abs(s["current_to_reference_volume_ratio"]-1)) > .01


def test_whole_cohort_positive_certificate_checks_interior_radial_extrema():
    # determinant at one vertex can be positive at endpoints but negative
    # inside because the two angular columns reverse at different depths.
    X = np.eye(3)
    Y = np.array(((0., 0., 1.), (1., 0., 1.), (0., 1., 1.)))
    Z = np.array(((0., 0., 0.), (-.7, 0., 0.), (0., -.8, 0.)))
    c = dict(X=X, Y=Y, Z=Z)
    with pytest.raises(ValueError, match="whole cohort"):
        audit.whole_cohort_chart_positive(c, .5, 1.)


def test_paired_force_is_equal_opposite_and_work_dual_in_all_modes():
    s = surface(); t = np.tile([3., -2., 4.], (len(s["cohort_ids"]), 1))
    increments = np.arange(36).reshape(2, 18)*.001
    f = audit.paired_force(s, t)
    np.testing.assert_array_equal(f[0], -f[1])
    work = audit.paired_virtual_work(s, t, increments)
    assert work == pytest.approx(np.sum(f*increments), rel=3e-16)
    assert np.linalg.norm(f.reshape(2, 3, 6)[:, :, 3:]) > .1


def test_equal_virtual_parent_fields_have_zero_jump_and_zero_work():
    s = surface(); u = np.arange(18)*.003
    t = np.tile([3., -2., 4.], (len(s["cohort_ids"]), 1))
    assert audit.paired_virtual_work(s, t, np.stack((u, u))) == 0.


def test_reference_work_requires_reference_normal_and_reference_measure():
    parent = manufactured_parent(); s = surface(parent); p = audit.prestress_fields(parent, s)
    converted = audit.paired_force(s, p["born_current_equivalent_traction_world_pa"], reference=False)
    np.testing.assert_allclose(converted, p["paired_force_n"], rtol=3e-15, atol=2e-16)
    difference = np.linalg.norm(p["wrong_current_normal_measure_paired_force_n"]-p["paired_force_n"])
    assert difference/np.linalg.norm(p["paired_force_n"]) > .005


def test_reference_Gc_seed_is_zero_initial_debit_not_available_work():
    law = dict(fracture_energy_j_m2=20., peak_traction_pa=30.,
        cohesive_stiffness_pa_m=100., compression_stiffness_pa_m=200.)
    seed = audit.cohesive_prestress_seed(law, [[-4., 3., 2.], [2., -4., 5.]])
    np.testing.assert_allclose(seed["reference_offset_m"], [[-.02, .03, .02], [.02, -.04, .05]])
    np.testing.assert_allclose(seed["auxiliary_fracture_energy_j_m2"],
        20+seed["reference_cohesive_energy_j_m2"])
    assert np.all(seed["final_separation_m"] > 2*20/30)
    assert seed["physical_release_admitted"] is False


def test_reference_Gc_seed_refuses_overstrength_without_damage_invention():
    law = dict(fracture_energy_j_m2=20., peak_traction_pa=30.,
        cohesive_stiffness_pa_m=100., compression_stiffness_pa_m=200.)
    with pytest.raises(ValueError, match="overstrength"):
        audit.cohesive_prestress_seed(law, [[31., 0., 0.]])


def test_resealed_original_material_q_substitution_is_rejected_numerically():
    parent = manufactured_parent(); s = surface(parent); raw = surface_packet(parent, s)
    audit.audit_embedded_surface(raw, parent, normal_xyz=[1., 0., 0.], offset_m=.1, face_indices=[0])
    # The same normalized plane does not authorize changing its literal
    # constructor inputs; reconstruction must retain their exact provenance.
    rescaled = copy.deepcopy(raw)
    rescaled["declared_normal_xyz"] = [2., 0., 0.]; rescaled["declared_offset_m"] = .2
    rescaled["embedded_surface_sha256"] = audit.digest({k: v for k, v in rescaled.items() if k != "embedded_surface_sha256"})
    with pytest.raises(ValueError, match="declared_normal_xyz differs"):
        audit.audit_embedded_surface(rescaled, parent, normal_xyz=[1., 0., 0.], offset_m=.1, face_indices=[0])
    raw["mass_coordinate_q"][0] += .01
    raw["embedded_surface_sha256"] = audit.digest({k: v for k, v in raw.items() if k != "embedded_surface_sha256"})
    with pytest.raises(ValueError, match="mass_coordinate_q differs"):
        audit.audit_embedded_surface(raw, parent, normal_xyz=[1., 0., 0.], offset_m=.1, face_indices=[0])


def test_resealed_full_current_pose_replaced_with_radial_top_copy_is_rejected():
    parent = manufactured_parent(); s = surface(parent); observer = {"binding_sha256": "fixture-binding"}
    raw = sample_packet(parent, s, observer)
    audit.audit_samples(raw, parent, observer)
    X = np.asarray(parent["runtime_state"]["geometry"]["vertices_m"])
    A = np.asarray(parent["runtime_state"]["displacement_m"]).reshape(3, 6)[:, :3]
    raw["current_position_m"] = (s["radial_scale"][:, None]*(s["barycentric"]@(X+A))).tolist()
    raw["sample_sha256"] = audit.digest({k: v for k, v in raw.items() if k != "sample_sha256"})
    with pytest.raises(ValueError, match="current_position_m differs"):
        audit.audit_samples(raw, parent, observer)


def test_resealed_pointwise_saved_stress_cannot_be_replaced_by_face_average():
    parent = manufactured_parent(); s = surface(parent); observer = {"binding_sha256": "fixture-binding"}
    raw = sample_packet(parent, s, observer)
    mean = np.mean(raw["stress_world_tensor"], axis=0)
    raw["stress_world_tensor"] = np.broadcast_to(mean, np.asarray(raw["stress_world_tensor"]).shape).tolist()
    raw["sample_sha256"] = audit.digest({k: v for k, v in raw.items() if k != "sample_sha256"})
    with pytest.raises(ValueError, match="stress_world_tensor differs"):
        audit.audit_samples(raw, parent, observer)


@pytest.mark.parametrize("order", (True, 1, 65, 2.5))
def test_bad_quadrature_orders_refused(order):
    with pytest.raises(ValueError): surface(order=order)


@pytest.mark.parametrize("normal", ([0., 0., 0.], [np.nan, 1., 0.], [True, 0., 0.]))
def test_nonphysical_plane_inputs_refused(normal):
    with pytest.raises(ValueError): surface(normal=normal)


def test_large_A_force_cannot_hide_resealed_P_force_erasure():
    expected = np.tile([1e15, 2e15, -3e15, 1e4, -2e4, 3e4], (2, 3))
    forged = expected.reshape(2, 3, 6).copy(); forged[:, :, 3:] = 0
    with pytest.raises(ValueError, match="P differs"):
        audit.check_force_modes(forged, expected, "paired force")
