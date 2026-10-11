"""Independent pointwise material memory and world-plane geometry identities."""
from __future__ import annotations

import ast
import copy
from pathlib import Path

import numpy as np
import pytest

from analysis.genesis_volume_location_validation import independent_audit as audit


def material_parent():
    X = np.array(((1., 1., 1.), (1., -1., -1.), (-1., 1., -1.)))
    density, coefficient = 2., .9
    mass = coefficient*density*np.linalg.det(X.T)/6
    qhi = (1-.7**3)/coefficient
    pose = np.array(((.03, -.02, .01, -.01, .04, .02),
                     (-.01, .03, .02, .03, -.02, .01),
                     (.02, .01, -.03, -.02, .01, -.04)))
    e0vertices = np.array(((.01, .02, -.01, .03, -.02, .04),
                          (.02, -.01, .03, .02, .01, -.03),
                          (-.01, .03, .02, -.02, .04, .01)))
    e1vertices = e0vertices*.2+np.arange(18).reshape(3, 6)*.0001
    c = dict(id=5, face_index=0, q_lo=.05, live_q_hi=qhi, inner_scale=.7,
             outer_scale=(1-coefficient*.05)**(1/3), owner_id="test-body", material_id="test-material")
    shell = dict(young_modulus_pa=71., poisson_ratio=.23, residual_stiffness=.02)
    return dict(checkpoint_sha256="test-only", source_checkpoint_sha256="test-formation",
        source_packet=dict(current_geometry=dict(face_frames_xyz=np.eye(3)[:, :2][None].tolist()),
                           source=dict(parameters=dict(shell=shell))),
        runtime_state=dict(displacement_m=pose.ravel().tolist(), fault_history=dict(damage=[.17]),
            geometry=dict(vertices_m=X.tolist(), faces=[[0, 1, 2]], column_mass_kg=[mass],
                          density_kg_m3=density, cohorts=[c]),
            mechanics=dict(state=dict(cohorts=[dict(id=5,
                strain0=(audit.SOURCE_ANGULAR_POINTS@e0vertices).tolist(),
                strain1=(audit.SOURCE_ANGULAR_POINTS@e1vertices).tolist())]))))


def sample(parent=None, N=(.2, .3, .5), q=.4, normal=None):
    return audit.material_samples(material_parent() if parent is None else parent,
        [5], [N], [q], normal_world=None if normal is None else [normal])


def test_full_current_position_retains_both_A_and_P_amplitudes():
    parent = material_parent(); values = sample(parent)
    X = np.asarray(parent["runtime_state"]["geometry"]["vertices_m"])
    u = np.asarray(parent["runtime_state"]["displacement_m"]).reshape(3, 6)
    N, lam = values["angular_barycentric"][0], values["radial_scale"][0]
    expected = lam*(N@X)+N@u[:, :3]+(lam-1)*(N@u[:, 3:])
    np.testing.assert_allclose(values["current_position_m"][0], expected, rtol=2e-15, atol=2e-16)
    assert np.linalg.norm(expected-lam*(N@(X+u[:, :3]))) > .002


def test_local_memory_is_affine_angular_and_inverse_radial_not_face_mean():
    parent = material_parent(); values = sample(parent)
    memory = parent["runtime_state"]["mechanics"]["state"]["cohorts"][0]
    inverse = np.linalg.inv(audit.SOURCE_ANGULAR_POINTS)
    N, lam = values["angular_barycentric"][0], values["radial_scale"][0]
    expected = N@inverse@(np.asarray(memory["strain0"])+np.asarray(memory["strain1"])/lam)
    np.testing.assert_allclose(values["elastic_strain_local6"][0], expected, rtol=2e-15, atol=2e-17)
    wrong = np.mean(memory["strain0"], axis=0)+np.mean(memory["strain1"], axis=0)/lam
    assert np.linalg.norm(expected-wrong) > .003


def test_engineering_shear_half_and_stress_shear_full_preserve_energy():
    values = sample()
    et, st = values["elastic_strain_local_tensor"][0], values["stress_local_tensor"][0]
    assert et[0, 1] == values["elastic_strain_local6"][0, 3]/2
    assert st[0, 1] == values["stress_local6"][0, 3]
    assert .5*np.sum(et*st) == pytest.approx(values["elastic_energy_density_j_m3"][0], rel=3e-16)


def test_stress_tensor_and_traction_rotate_with_inherited_source_frame():
    parent = material_parent(); angle = .63
    Q = np.array(((np.cos(angle), 0., np.sin(angle)),
                  (0., 1., 0.), (-np.sin(angle), 0., np.cos(angle))))
    parent["source_packet"]["current_geometry"]["face_frames_xyz"] = Q[:, :2][None].tolist()
    values = sample(parent, normal=(0., 0., 1.))
    expected = Q@values["stress_local_tensor"][0]@Q.T
    np.testing.assert_allclose(values["stress_world_tensor"][0], expected, rtol=3e-16)
    np.testing.assert_array_equal(values["traction_world_pa"][0], expected[:, 2])


def test_damage_uses_current_accepted_material_not_original_birth_value():
    parent = material_parent(); initial = sample(parent)
    parent["runtime_state"]["fault_history"]["damage"][0] = .61
    after = sample(parent)
    factor = lambda d: .02+.98*(1-d)**2
    np.testing.assert_allclose(after["stress_local6"], initial["stress_local6"]*factor(.61)/factor(.17), rtol=3e-16)
    np.testing.assert_array_equal(after["elastic_strain_local6"], initial["elastic_strain_local6"])


@pytest.mark.parametrize("ids,N,q", (([True], [[.2, .3, .5]], [.4]),
    ([6], [[.2, .3, .5]], [.4]), ([5], [[-.1, .4, .7]], [.4]),
    ([5], [[.1, .2, .3]], [.4]), ([5], [[.2, .3, .5]], [.01]),
    ([5], [[.2, .3, .5]], [1.]), ([5], [[.2, .3, .5]], [np.nan])))
def test_material_probe_refuses_nonmaterial_or_outside_samples(ids, N, q):
    with pytest.raises(ValueError): audit.material_samples(material_parent(), ids, N, q)


def test_traction_refuses_nonunit_normal_instead_of_changing_magnitude():
    with pytest.raises(ValueError, match="unit"):
        sample(normal=(0., 0., 2.))


def test_duplicate_probes_keep_original_material_identifiers():
    values = audit.material_samples(material_parent(), [5, 5], [[1., 0., 0.], [0., 1., 0.]], [.4, .5])
    assert values["owner_ids"] == ["test-body", "test-body"]
    assert values["material_ids"] == ["test-material", "test-material"]
    np.testing.assert_array_equal(values["cohort_ids"], [5, 5])


def vertical_section(A=None, P=None, offset=0.):
    X = np.array(((1., 0., 1.), (-1., 0., 1.), (0., 2., 1.)))
    return audit.section_coefficients(X, np.zeros((3, 3)) if A is None else A,
        np.zeros((3, 3)) if P is None else P, (1., 0., 0.), offset)


def test_world_plane_cut_is_barycentric_line_at_each_material_depth():
    c = vertical_section(); cut = audit.section_at_scale(c, .8)
    assert cut["kind"] == "segment"
    np.testing.assert_allclose(cut["current_points_m"][:, 0], 0., atol=1e-16)
    assert any(np.array_equal(N, [0., 0., 1.]) for N in cut["barycentric"])
    assert any(np.array_equal(N, [.5, .5, 0.]) for N in cut["barycentric"])


def test_full_depth_P_changes_barycentric_cut_without_changing_world_plane():
    P = np.array(((.7, .1, .3), (-.2, -.1, .4), (.3, .2, -.1)))
    c = vertical_section(P=P, offset=.1)
    a, b = (audit.section_at_scale(c, lam) for lam in (.7, .95))
    np.testing.assert_allclose(a["current_points_m"][:, 0], .1, atol=8e-17)
    np.testing.assert_allclose(b["current_points_m"][:, 0], .1, atol=8e-17)
    assert np.linalg.norm(a["barycentric"]-b["barycentric"]) > .02


def test_vertex_crossings_partition_every_change_of_cut_edge_pair():
    c = vertical_section(offset=.8)
    np.testing.assert_array_equal(audit.section_breaks(c, .5, 1.), [.5, .8, 1.])
    assert audit.section_at_scale(c, .7)["kind"] == "empty"
    assert audit.section_at_scale(c, .8)["kind"] == "touch"
    assert audit.section_at_scale(c, .9)["kind"] == "segment"


def test_horizontal_section_has_area_without_radial_interval_measure():
    X = np.array(((1., 0., 1.), (-1., 0., 1.), (0., 2., 1.)))
    c = audit.section_coefficients(X, np.zeros((3, 3)), np.zeros((3, 3)), (0., 0., 1.), .8)
    assert audit.section_at_scale(c, .8)["kind"] == "full_triangle"
    with pytest.raises(ValueError, match="explicit barycentric"):
        audit.section_point(c, .8)
    p = audit.section_point(c, .8, barycentric=(.2, .3, .5))
    assert p["current_position_m"][2] == pytest.approx(.8)
    measure = audit.section_measure(c, .5, 1.)
    assert measure["area_m2"] == pytest.approx(2*.8**2, rel=3e-16)
    np.testing.assert_allclose(measure["first_moment_m3"], 2*.8**2*.8*X.mean(axis=0), atol=3e-16)


def test_independent_surface_area_matches_exact_vertical_trapezoid():
    measure = audit.section_measure(vertical_section(), .5, 1.)
    assert measure["area_m2"] == pytest.approx(1-.5**2, rel=8e-16)
    # Integral on x=0, .5<=z<=1, 0<=y<=2*z.
    np.testing.assert_allclose(measure["first_moment_m3"], [0., 2*(1-.5**3)/3, 2*(1-.5**3)/3], rtol=2e-15, atol=2e-17)


def test_section_quadrature_adds_material_depth_partitions_without_double_area():
    c = vertical_section(P=np.array(((.2, .1, .3), (-.1, -.2, -.1), (.1, .2, .1))), offset=.1)
    whole = audit.section_measure(c, .6, 1., order=24)
    a, b = audit.section_measure(c, .6, .83, order=24), audit.section_measure(c, .83, 1., order=24)
    assert whole["area_m2"] == pytest.approx(a["area_m2"]+b["area_m2"], rel=2e-14)
    np.testing.assert_allclose(whole["first_moment_m3"], a["first_moment_m3"]+b["first_moment_m3"], rtol=2e-14)


def test_horizontal_partition_requires_explicit_single_sided_boundary_ownership():
    X = np.array(((1., 0., 1.), (-1., 0., 1.), (0., 2., 1.)))
    c = audit.section_coefficients(X, np.zeros((3, 3)), np.zeros((3, 3)), (0., 0., 1.), .8)
    whole = audit.section_measure(c, .5, 1.)
    closed_a, closed_b = audit.section_measure(c, .5, .8), audit.section_measure(c, .8, 1.)
    assert closed_a["area_m2"]+closed_b["area_m2"] == pytest.approx(2*whole["area_m2"])
    a = audit.section_measure(c, .5, .8, include_outer_boundary=False)
    b = audit.section_measure(c, .8, 1., include_inner_boundary=True)
    assert a["area_m2"]+b["area_m2"] == pytest.approx(whole["area_m2"], rel=3e-16)


def sample_dto(parent=None):
    parent = material_parent() if parent is None else parent
    values = sample(parent, normal=(0., 0., 1.))
    parent["runtime_state"]["geometry"].update(source_identity="test-identity", time_myr=.7)
    body = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in values.items()}
    body.update(format="genesis-volume-material-stress-samples-0.1",
        parent_checkpoint_sha256=parent["checkpoint_sha256"],
        formation_checkpoint_sha256=parent["source_checkpoint_sha256"],
        source_identity="test-identity", time_myr=.7, observer_binding_sha256="test-observer",
        supplied_normal_is_fracture_orientation=False, **audit.stress_claims())
    return parent, {**body, "sample_sha256": audit.digest(body)}, {"binding_sha256": "test-observer"}


def reseal_sample(raw):
    raw["sample_sha256"] = audit.digest({k: v for k, v in raw.items() if k != "sample_sha256"})


def test_independent_sample_audit_recomputes_full_source_not_only_seal():
    parent, raw, observer = sample_dto()
    report = audit.audit_samples(raw, parent, observer)
    assert report["source_local_six_strain_memory_checked"] is True
    assert report["full_current_A_and_P_placement_checked"] is True


def test_resealed_radial_top_only_A_pose_cannot_replace_full_current_material_position():
    parent, raw, observer = sample_dto()
    X = np.asarray(parent["runtime_state"]["geometry"]["vertices_m"])
    A = np.asarray(parent["runtime_state"]["displacement_m"]).reshape(3, 6)[:, :3]
    raw["current_position_m"] = [raw["radial_scale"][0]*(np.asarray(raw["angular_barycentric"][0])@(X+A))]
    raw["current_position_m"] = np.asarray(raw["current_position_m"]).tolist()
    reseal_sample(raw)
    with pytest.raises(ValueError, match="current_position_m"):
        audit.audit_samples(raw, parent, observer)


def test_resealed_face_mean_stress_cannot_replace_pointwise_stored_memory():
    parent, raw, observer = sample_dto()
    memory = parent["runtime_state"]["mechanics"]["state"]["cohorts"][0]
    lam = raw["radial_scale"][0]
    wrong_strain = np.mean(memory["strain0"], axis=0)+np.mean(memory["strain1"], axis=0)/lam
    shell = parent["source_packet"]["source"]["parameters"]["shell"]
    C = audit.isotropic_matrix(shell["young_modulus_pa"], shell["poisson_ratio"], .17, shell["residual_stiffness"])
    wrong_stress = C@wrong_strain; wrong_tensor = audit.tensor(wrong_stress)
    raw["stress_local6"] = [wrong_stress.tolist()]
    raw["stress_local_tensor"] = raw["stress_world_tensor"] = [wrong_tensor.tolist()]
    raw["traction_world_pa"] = [wrong_tensor[:, 2].tolist()]
    reseal_sample(raw)
    with pytest.raises(ValueError, match="stress"):
        audit.audit_samples(raw, parent, observer)


def test_resealed_physical_surface_claim_cannot_be_added_to_material_probe():
    parent, raw, observer = sample_dto()
    raw["selected_sharp_material_surface"] = {"offset_m": .2}
    reseal_sample(raw)
    with pytest.raises(ValueError, match="selected_sharp_material_surface"):
        audit.audit_samples(raw, parent, observer)


def test_resealed_extra_unchecked_sample_field_is_refused():
    parent, raw, observer = sample_dto()
    raw["new_birth_energy_credit_j"] = 1.
    reseal_sample(raw)
    with pytest.raises(ValueError, match="schema"):
        audit.audit_samples(raw, parent, observer)


def test_plane_positive_scaling_preserves_oriented_section():
    c = vertical_section(offset=.2)
    d = audit.section_coefficients(c["Y"], c["Z"], np.zeros((3, 3)), (3., 0., 0.), .6)
    np.testing.assert_allclose(audit.section_at_scale(c, .9)["current_points_m"],
                               audit.section_at_scale(d, .9)["current_points_m"], rtol=3e-16)


def test_field_variance_exposes_all_saved_angular_and_radial_coefficients():
    result = audit.field_variance(material_parent())
    assert result["cohorts_examined"] == 1 and result["samples_examined"] == 6
    assert result["maximum_radial_stress_difference_pa"] > .05
    assert result["maximum_angular_stress_difference_pa"] > .5
    assert result["radial_coefficient_memory_nonzero"] is True


def test_observation_never_mutates_parent_memory_pose_or_clock():
    parent = material_parent(); before = copy.deepcopy(parent)
    sample(parent); audit.field_variance(parent)
    assert parent == before


def test_independent_module_imports_no_new_or_old_tectonics_producer():
    tree = ast.parse(Path(audit.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert not any(alias.name.startswith("tectonics") for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("tectonics")
            assert "genesis_volume_material" not in (node.module or "")
