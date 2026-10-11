"""Independent mathematical checks of Γ measure and prospective work.

These examples contain explicit mechanical geometry and law parameters; they
are neither new material history nor a claim of authentic Genesis activation.
"""
from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from analysis.genesis_volume_creation_validation import independent_audit as audit


LAW = dict(fracture_energy_j_m2=23., peak_traction_pa=7.,
           cohesive_stiffness_pa_m=19., compression_stiffness_pa_m=29.)


def geometry_fixture():
    vertices = np.asarray(((1., 1., 1.), (1., -1., -1.),
                           (-1., 1., -1.), (-1., -1., 1.)))
    faces = np.asarray(((0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2)))
    coefficient = np.asarray((.9, .8, .95, .91))
    density = 2.
    masses = [c*density*np.linalg.det(vertices[row].T)/6 for c, row in zip(coefficient, faces)]
    depths = ((.7, .85, 1.), (.8, .9, 1.), (.75, .86, 1.), (.65, .82, 1.))
    cohorts = []
    for face, values in enumerate(depths):
        for lo, hi in zip(values[:-1], values[1:]):
            cohorts.append(dict(id=100+len(cohorts), face_index=face,
                q_lo=(1-hi**3)/coefficient[face], live_q_hi=(1-lo**3)/coefficient[face],
                inner_scale=lo, outer_scale=hi, owner_id=f"test-owner-{face}",
                material_id=f"test-material-{len(cohorts)}"))
    return dict(vertices_m=vertices.tolist(), faces=faces.tolist(),
                column_mass_kg=masses, density_kg_m3=density, cohorts=cohorts)


def surface_fixture():
    return audit.surface_oracle(geometry_fixture(), ((0, 1), (1, 2), (0, 2)))


def test_radial_surface_area_matches_exact_ruled_edge_integral():
    surface = surface_fixture(); vertices = np.asarray(geometry_fixture()["vertices_m"])
    expected_areas = []
    for edge, faces in zip(surface["cut_edges"], surface["side_face_indices"]):
        depths = (.7, .8, .75, .65)
        inner = max(depths[faces[0]], depths[faces[1]])
        factor = np.linalg.norm(np.cross(*vertices[edge]))
        expected_areas.append(factor*(1-inner**2)/2)
    np.testing.assert_allclose(surface["edge_area_m2"], expected_areas, rtol=3e-15)
    assert sum(surface["patch_area_m2"]) == pytest.approx(sum(expected_areas), rel=3e-15)
    np.testing.assert_allclose(surface["area_weights_m2"].sum(axis=1), surface["patch_area_m2"], rtol=3e-15)


def test_distinct_mass_charts_are_paired_at_physical_depth_not_equal_q():
    surface = surface_fixture(); geometry = geometry_fixture()
    assert np.any(abs(surface["side_q_bounds"][:, 0]-surface["side_q_bounds"][:, 1]) > .01)
    for pair, bounds in zip(surface["side_cohort_indices"], surface["lambda_bounds"]):
        a, b = [geometry["cohorts"][i] for i in pair]
        assert a["face_index"] != b["face_index"]
        assert bounds[0] >= max(a["inner_scale"], b["inner_scale"])
        assert bounds[1] <= min(a["outer_scale"], b["outer_scale"])
    assert len(surface["unmatched_edge_indices"]) > 0
    # Solid-only area is disjoint from the paired Γ charge.
    assert surface["unmatched_area_m2"].sum() > 0


def test_midpoint_birth_boundaries_preserve_both_source_ids_and_owners():
    geometry = geometry_fixture(); surface = surface_fixture()
    for pair, ids, owners, materials in zip(surface["side_cohort_indices"], surface["side_cohort_ids"],
                                           surface["side_owner_ids"], surface["side_material_ids"]):
        assert ids.tolist() == [geometry["cohorts"][i]["id"] for i in pair]
        assert owners == [geometry["cohorts"][i]["owner_id"] for i in pair]
        assert materials == [geometry["cohorts"][i]["material_id"] for i in pair]


def test_surface_frame_is_oriented_from_bank0_into_bank1():
    surface = surface_fixture(); geometry = geometry_fixture()
    vertices, faces = np.asarray(geometry["vertices_m"]), np.asarray(geometry["faces"])
    for edge, owners, normal, t1, t2 in zip(surface["cut_edges"], surface["side_face_indices"],
                                         surface["edge_normals_xyz"], surface["edge_tangent1_xyz"],
                                         surface["edge_tangent2_xyz"]):
        frame = np.column_stack((normal, t1, t2))
        np.testing.assert_allclose(frame.T@frame, np.eye(3), atol=4e-16)
        assert np.linalg.det(frame) == pytest.approx(1., rel=2e-15)
        thirds = [next(v for v in faces[face] if v not in edge) for face in owners]
        assert normal@vertices[thirds[0]] < 0 < normal@vertices[thirds[1]]


def test_six_dof_trace_matches_cartesian_finite_difference_and_direct_field():
    surface = surface_fixture(); rng = np.random.default_rng(823)
    motion = rng.normal(size=surface["bank_ndof"])
    direct = audit.surface_jump(surface, motion)
    traced = np.einsum("pqai,pi->pqa", surface["jump_trace"], motion[surface["trace_dofs"]])
    np.testing.assert_allclose(direct, traced, rtol=3e-15, atol=5e-16)
    for dof in range(surface["bank_ndof"]):
        perturb = np.zeros(surface["bank_ndof"]); perturb[dof] = 1e-6
        difference = (audit.surface_jump(surface, motion+perturb)-audit.surface_jump(surface, motion-perturb))/(2e-6)
        exact = audit.surface_jump(surface, perturb)/1e-6
        np.testing.assert_allclose(difference, exact, rtol=2e-8, atol=4e-10)


def test_full_A_and_P_traction_forces_are_work_dual():
    surface = surface_fixture(); rng = np.random.default_rng(9991)
    motion = rng.normal(size=surface["bank_ndof"])
    traction = rng.normal(size=(len(surface["patch_edge_indices"]), 4, 3))
    force = audit.surface_force(surface, traction)
    work = np.einsum("pqa,pqa,pq->", traction, audit.surface_jump(surface, motion), surface["area_weights_m2"])
    assert force@motion == pytest.approx(work, rel=4e-15)
    assert np.linalg.norm(force.reshape(-1, 6)[:, 3:]) > 1e-8
    probe_P = np.zeros_like(motion).reshape(-1, 6)
    probe_P[:, 3:] = motion.reshape(-1, 6)[:, 3:]
    work_P = np.einsum("pqa,pqa,pq->", traction, audit.surface_jump(surface, probe_P.ravel()),
                       surface["area_weights_m2"])
    assert force@probe_P.ravel() == pytest.approx(work_P, rel=4e-15)


def test_rigid_translation_and_reference_rotation_have_zero_tied_jump():
    surface = surface_fixture(); geometry = geometry_fixture()
    vertices = np.asarray(geometry["vertices_m"])[surface["topology"]["parent_vertex"]]
    shift = np.repeat([[.2, -.3, .4]], len(vertices), axis=0)
    motion = np.column_stack((shift, np.zeros_like(shift))).ravel()
    assert not np.any(audit.surface_jump(surface, motion))
    rotation = np.cross(np.repeat([[.1, .2, .3]], len(vertices), axis=0), vertices)
    motion = np.column_stack((rotation, rotation)).ravel()
    assert not np.any(audit.surface_jump(surface, motion))
    # Applying a translation to only the isolated face is a real vector jump.
    motion = np.zeros(surface["bank_ndof"]).reshape(-1, 6)
    motion[:3, :3] = shift[:3]
    assert np.linalg.norm(audit.surface_jump(surface, motion.ravel())) > .1


def test_pure_P_difference_is_preserved_below_top_surface():
    surface = surface_fixture(); motion = np.zeros((surface["bank_ndof"]//6, 6))
    motion[:3, 3:] = (.2, -.4, .7)
    jump = audit.surface_jump(surface, motion.ravel())
    assert np.linalg.norm(jump) > .01
    assert np.all(np.linalg.norm(jump, axis=-1) > 0)


def test_empty_surface_keeps_no_contacts_or_creation_charge():
    surface = audit.surface_oracle(geometry_fixture(), [])
    assert surface["jump_trace"].shape == (0, 4, 3, 24)
    assert surface["area_weights_m2"].shape == (0, 4)
    assert not len(surface["patch_edge_indices"])
    assert not np.any(audit.surface_force(surface, np.zeros((0, 4, 3))))


@pytest.mark.parametrize("mutation", ("overlap", "wrong_scale", "negative_mass", "textual_index"))
def test_independent_surface_refuses_unphysical_parent_mutations(mutation):
    geometry = geometry_fixture()
    if mutation == "overlap":
        geometry["cohorts"].append(copy.deepcopy(geometry["cohorts"][0]))
    elif mutation == "wrong_scale":
        geometry["cohorts"][0]["inner_scale"] += .01
    elif mutation == "negative_mass":
        geometry["column_mass_kg"][0] = -1.
    else:
        geometry["faces"][0][0] = "0"
    with pytest.raises(ValueError):
        audit.surface_oracle(geometry, ((0, 1),))


@pytest.mark.parametrize("new", ((.2, .1, -.1), (1.1, .7, -.2), (7., 0., 0.),
                                 (-.9, 1., -2.), (1., 3., 2.)))
def test_direct_traction_integral_balances_creation_and_recoverable_energy(new):
    result = audit.creation_segment(LAW, np.zeros((1, 3)), np.asarray([new]), np.zeros(1))
    assert result["mechanical_work_j_m2"][0] == pytest.approx(
        (result["stored_energy_j_m2"]+result["fracture_dissipation_j_m2"])[0], rel=3e-11)
    assert 0 <= result["fracture_dissipation_j_m2"][0] <= LAW["fracture_energy_j_m2"]


def test_complete_separation_cost_is_Gc_once_not_two_bank_areas():
    failure = 2*LAW["fracture_energy_j_m2"]/LAW["peak_traction_pa"]
    result = audit.creation_segment(LAW, np.zeros((1, 3)), np.array([[2*failure, 0., 0.]]), np.zeros(1))
    assert result["fracture_dissipation_j_m2"][0] == LAW["fracture_energy_j_m2"]
    assert result["stored_energy_j_m2"][0] == 0
    assert result["mechanical_work_j_m2"][0] == pytest.approx(LAW["fracture_energy_j_m2"], rel=2e-13)
    surface = surface_fixture(); area = surface["area_weights_m2"].sum()
    assert area*result["mechanical_work_j_m2"][0] == pytest.approx(area*LAW["fracture_energy_j_m2"], rel=2e-13)


def test_unload_reload_preserves_maximum_and_no_double_fracture_charge():
    zero = np.zeros((1, 3)); peak = np.array([[1., 2., .3]])
    loaded = audit.creation_segment(LAW, zero, peak, np.zeros(1))
    unload = audit.creation_segment(LAW, peak, zero, loaded["maximum_separation_m"])
    reload = audit.creation_segment(LAW, zero, peak, unload["maximum_separation_m"])
    assert unload["fracture_dissipation_increment_j_m2"][0] == 0
    assert reload["fracture_dissipation_increment_j_m2"][0] == 0
    assert unload["maximum_separation_m"][0] == loaded["maximum_separation_m"][0]
    assert unload["mechanical_work_j_m2"][0] < 0
    assert (unload["mechanical_work_j_m2"]+reload["mechanical_work_j_m2"])[0] == pytest.approx(0., abs=1e-12)


def test_two_shear_rotations_preserve_work_damage_and_vector_traction():
    theta = .711
    rotation = np.array(((1., 0., 0.), (0., np.cos(theta), -np.sin(theta)),
                         (0., np.sin(theta), np.cos(theta))))
    old, new = np.array([[.7, .8, -.3]]), np.array([[-.4, 2., .9]])
    history = np.array([np.linalg.norm(old)])
    a = audit.creation_segment(LAW, old, new, history)
    b = audit.creation_segment(LAW, old@rotation.T, new@rotation.T, history)
    for field in ("mechanical_work_j_m2", "stored_energy_j_m2", "fracture_dissipation_j_m2", "damage"):
        np.testing.assert_allclose(a[field], b[field], rtol=3e-14)
    np.testing.assert_allclose(a["traction_pa"]@rotation.T, b["traction_pa"], rtol=3e-14)
    np.testing.assert_allclose(rotation@a["tangent_pa_m"][0]@rotation.T, b["tangent_pa_m"][0], atol=5e-15)


def test_compression_crossing_does_not_create_normal_fracture_energy():
    result = audit.creation_segment(LAW, np.array([[-3., 0., 0.]]),
                                    np.array([[.2, 0., 0.]]), np.zeros(1))
    assert result["fracture_dissipation_increment_j_m2"][0] == 0
    expected = .5*LAW["cohesive_stiffness_pa_m"]*.2**2-.5*LAW["compression_stiffness_pa_m"]*3**2
    assert result["mechanical_work_j_m2"][0] == pytest.approx(expected, rel=2e-14)


@pytest.mark.parametrize("loading,gap", ((True, (.9, .8, -.7)), (True, (-.6, .8, -.7)),
                                          (False, (.9, .8, -.7)), (False, (-.6, .8, -.7))))
def test_full_three_mode_tangent_matches_force_finite_difference(loading, gap):
    gap = np.asarray([gap]); history = np.array([0. if loading else 3.])
    response = audit.law_point(LAW, gap, history)
    h = 1e-6
    finite = np.column_stack([(audit.law_point(LAW, gap+h*axis, history)["traction_pa"][0]
                               -audit.law_point(LAW, gap-h*axis, history)["traction_pa"][0])/(2*h)
                              for axis in np.eye(3)])
    np.testing.assert_allclose(response["tangent_pa_m"][0], finite, rtol=2e-8, atol=4e-9)


@pytest.mark.parametrize("bad", (True, -1., 0., float("nan"), "23"))
def test_unknown_or_invalid_creation_energy_is_not_replaced_with_default(bad):
    law = dict(LAW, fracture_energy_j_m2=bad)
    with pytest.raises(ValueError):
        audit.validate_law(law)


def test_historical_losses_are_not_law_parameters_or_new_work_credit():
    for key in ("friction_work_j", "viscous_work_j", "damage_energy_change_j", "legacy_default_Gc"):
        with pytest.raises(ValueError):
            audit.validate_law({**LAW, key: 1e30})
    with pytest.raises(ValueError):
        audit.validate_law({key: value for key, value in LAW.items() if key != "fracture_energy_j_m2"})


def test_direct_integral_refuses_inconsistent_old_irreversible_history():
    with pytest.raises(ValueError, match="inconsistent old history"):
        audit.integrate_traction_work(LAW, np.ones((1, 3)), np.zeros((1, 3)), np.zeros(1))


def test_independent_audit_does_not_import_producer_modules():
    path = Path(audit.__file__)
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(not alias.name.startswith("tectonics") for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("tectonics")


def grown_geometry_fixture():
    geometry = geometry_fixture()
    # Two newly solid cohorts extend the existing fronts at different depths.
    # Their equal-age labels never imply equal q or equal physical support.
    for identity, face, lo, hi, coefficient in ((108, 0, .6, .7, .9), (109, 1, .65, .8, .8)):
        geometry["cohorts"].append(dict(id=identity, face_index=face,
            q_lo=(1-hi**3)/coefficient, live_q_hi=(1-lo**3)/coefficient,
            inner_scale=lo, outer_scale=hi, owner_id=f"test-owner-{face}",
            material_id=f"test-material-{identity}"))
    return geometry


def test_calculated_birth_growth_adds_paired_and_unmatched_physical_support():
    old, new = geometry_fixture(), grown_geometry_fixture()
    assert audit.birth_identity_delta(old, new) == [108, 109]
    surface = audit.surface_oracle(new, ((0, 1), (1, 2), (0, 2)))
    pairs = surface["side_cohort_ids"]
    assert any(set(pair) == {108, 109} for pair in pairs)
    assert any(109 in pair and 108 not in pair for pair in pairs)
    assert 108 in surface["unmatched_cohort_ids"]
    assert surface["patch_area_m2"].sum() > surface_fixture()["patch_area_m2"].sum()


@pytest.mark.parametrize("mutation", ("duplicate", "replacement", "lost_ancestor"))
def test_birth_growth_refuses_replaced_or_reused_material_history(mutation):
    old, new = geometry_fixture(), grown_geometry_fixture()
    if mutation == "duplicate":
        new["cohorts"][-1]["id"] = old["cohorts"][0]["id"]
    elif mutation == "replacement":
        new["cohorts"][0]["material_id"] = "replacement-history"
    else:
        new["cohorts"].pop(0)
    with pytest.raises(ValueError):
        audit.birth_identity_delta(old, new)


@pytest.fixture(scope="module")
def actual_parent_packet():
    path = Path(__file__).resolve().parents[1]/(
        "analysis/genesis_volume_slip_validation/evidence/actual_control/direct_after_second.json")
    return json.loads(path.read_bytes())


def minimal_claims(parent):
    return dict(format=audit.FORMAT,
        scope="source_bound_prospective_surface_and_creation_kernel_not_physical_genesis_fracture",
        parent_raw_sha256=audit.ACTUAL_PARENT_RAW_SHA256, parent_checkpoint=parent,
        no_physical_fracture_or_contact_admission=True,
        bank_displacement_semantics="from_zero_computational_fixture_origin_not_absolute_accepted_Genesis_pose",
        cohesive_energy_measure="one_original_paired_interface_area_counts_both_created_banks",
        no_prestress_replacement_or_release_equilibrium_claim=True, fixture_cases=[])


def test_empty_computational_case_list_cannot_be_resealed_as_success(actual_parent_packet):
    with pytest.raises(ValueError, match="nonempty fixture cases"):
        audit.audit_fixture(minimal_claims(actual_parent_packet), actual_parent_packet)
    with pytest.raises(ValueError, match="nonempty computational paths"):
        audit.audit_case(dict(computational_fixture=True, paths=[]), actual_parent_packet)


def test_absolute_pose_double_area_and_release_claims_cannot_be_resealed(actual_parent_packet):
    for field, value in (("bank_displacement_semantics", "absolute_accepted_Genesis_pose"),
                         ("cohesive_energy_measure", "two_separately_charged_banks"),
                         ("no_prestress_replacement_or_release_equilibrium_claim", False)):
        claims = minimal_claims(actual_parent_packet); claims[field] = value
        with pytest.raises(ValueError, match="displacement, area or release semantics"):
            audit.audit_fixture(claims, actual_parent_packet)
