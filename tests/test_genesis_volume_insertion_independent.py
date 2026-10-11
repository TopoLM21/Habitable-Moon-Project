"""Independent physical identities for prospective prestress insertion."""
from __future__ import annotations

import ast
import copy
from pathlib import Path

import numpy as np
import pytest

from analysis.genesis_volume_insertion_validation import independent_audit as audit


LAW = dict(fracture_energy_j_m2=23., peak_traction_pa=7.,
           cohesive_stiffness_pa_m=19., compression_stiffness_pa_m=29.)


def reference_surface():
    vertices = np.asarray(((1., 1., 1.), (1., -1., -1.),
                           (-1., 1., -1.), (-1., -1., 1.)))
    faces = np.asarray(((0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2)))
    rho, coefficient = 2., .9
    masses = [coefficient*rho*np.linalg.det(vertices[row].T)/6 for row in faces]
    cohorts = [dict(id=i+1, face_index=i, q_lo=0., live_q_hi=(1-.7**3)/coefficient,
                   inner_scale=.7, outer_scale=1., owner_id=f"test-owner-{i}", material_id=f"test-material-{i}")
               for i in range(4)]
    geometry = dict(vertices_m=vertices.tolist(), faces=faces.tolist(), density_kg_m3=rho,
                    column_mass_kg=masses, cohorts=cohorts)
    return audit.surface_oracle(geometry, ((0, 1), (1, 2), (0, 2)))


def test_constraint_reaction_preserves_nonzero_parent_residual():
    J = np.column_stack((-np.eye(6), np.eye(6)))
    residual = np.asarray((.2, -.3, .4, .7, -.2, .9, -.5, .6, .1, -.4, .8, .3))
    result = audit.interval_reaction(J, residual)
    R = np.vstack((np.eye(6), np.eye(6)))
    np.testing.assert_allclose(J@result["remaining_residual_n"], 0., atol=2e-16)
    np.testing.assert_allclose(R.T@result["remaining_residual_n"], R.T@residual, atol=2e-16)
    assert np.linalg.norm(R.T@residual) > .1
    tied_motion = R@np.arange(6.)
    assert result["interface_force_n"]@tied_motion == pytest.approx(0., abs=2e-16)


def test_reaction_has_correct_dual_sign_for_an_independent_jump():
    J = np.column_stack((-np.eye(6), np.eye(6)))
    result = audit.interval_reaction(J, np.arange(12.)*.2)
    probe = np.sin(np.arange(12.)*.37)
    assert result["interface_force_n"]@probe == pytest.approx(result["canonical_reaction_n"]@(J@probe), rel=3e-15)
    assert np.linalg.norm(J@result["remaining_residual_n"]) < 1e-15


def test_empty_constraint_family_leaves_the_parent_response_intact():
    residual = np.asarray((1., 2., -3., 4., 5., -6.))
    result = audit.interval_reaction(np.zeros((0, 6)), residual)
    np.testing.assert_array_equal(result["remaining_residual_n"], residual)
    assert not np.any(result["interface_force_n"])


def test_redundant_constraints_do_not_define_an_independent_reaction():
    J = np.asarray(((-1., 1.), (-2., 2.)))
    with pytest.raises(ValueError, match="independent"):
        audit.interval_reaction(J, np.ones(2))


def test_area_L2_traction_reproduces_all_A_P_reaction_and_work():
    surface = reference_surface(); rng = np.random.default_rng(9113)
    eta = rng.normal(size=len(surface["topology"]["jump_operator"]))
    eta[np.arange(len(eta)) % 6 >= 3] *= .3
    representative = audit.minimum_area_traction(surface, eta)
    force = audit.surface_force(surface, representative["traction_world_pa"])
    np.testing.assert_allclose(force, surface["topology"]["jump_operator"].T@eta, rtol=2e-13, atol=2e-13)
    motion = rng.normal(size=surface["bank_ndof"])
    world_jump = audit.surface_jump(surface, motion)
    work = np.einsum("pqa,pqa,pq->", representative["traction_world_pa"], world_jump, surface["area_weights_m2"])
    assert force@motion == pytest.approx(work, rel=2e-13)
    assert representative["rank"] == 18
    assert len(representative["singular_values"]) == 18


def test_minimum_area_norm_is_orthogonal_to_every_zero_force_traction():
    surface = reference_surface(); rng = np.random.default_rng(126)
    eta = rng.normal(size=len(surface["topology"]["jump_operator"]))
    original = audit.minimum_area_traction(surface, eta)["traction_world_pa"]
    arbitrary = rng.normal(size=original.shape)
    H = audit.minimum_area_traction(surface, np.zeros_like(eta))["H"]
    weights = np.repeat(surface["area_weights_m2"].ravel(), 3)
    arbitrary_eta = H.T@(weights*arbitrary.ravel())
    null = arbitrary-audit.minimum_area_traction(surface, arbitrary_eta)["traction_world_pa"]
    assert np.linalg.norm(audit.surface_force(surface, null)) < 1e-12
    inner = np.einsum("pqa,pqa,pq->", original, null, surface["area_weights_m2"])
    assert abs(inner) < 1e-12
    norm = lambda t: np.einsum("pqa,pqa,pq->", t, t, surface["area_weights_m2"])
    assert norm(original+null) == pytest.approx(norm(original)+norm(null), rel=2e-13)


def test_missing_radial_trace_cannot_replace_full_18_jump_modes():
    surface = reference_surface()
    surface["jump_trace"] = surface["jump_trace"].copy()
    surface["jump_trace"][..., np.arange(24) % 6 >= 3] = 0.
    with pytest.raises(ValueError, match="span all canonical jump modes"):
        audit.minimum_area_traction(surface, np.ones(18))


@pytest.mark.parametrize("initial", ((3., 2., -1.), (-4., 2., -1.), (0., 0., 0.)))
def test_zero_gap_insertion_keeps_traction_with_zero_new_energy_or_work(initial):
    initial = np.asarray([initial]); seed = audit.prestress_seed(LAW, initial)
    result = audit.prestress_segment(LAW, initial, np.zeros((1, 3)), np.zeros((1, 3)),
                                    seed["initial_maximum_separation_m"])
    np.testing.assert_allclose(result["traction_pa"], initial, rtol=2e-15, atol=1e-15)
    np.testing.assert_allclose(result["stored_energy_j_m2"], 0., atol=3e-16)
    assert not np.any(result["fracture_dissipation_j_m2"])
    assert not np.any(result["mechanical_work_j_m2"])


@pytest.mark.parametrize("new", ((.1, .2, -.1), (1., .5, -.3), (8., 0., 0.),
                                 (-.8, .3, .9), (-.3, -2., 3.)))
def test_shifted_traction_path_balances_relative_energy_and_true_creation(new):
    initial = np.asarray([[3., 2., -1.]])
    seed = audit.prestress_seed(LAW, initial)
    result = audit.prestress_segment(LAW, initial, np.zeros((1, 3)), np.asarray([new]),
                                    seed["initial_maximum_separation_m"])
    assert result["mechanical_work_j_m2"][0] == pytest.approx(
        result["stored_energy_j_m2"][0]+result["fracture_dissipation_j_m2"][0], rel=3e-10, abs=2e-11)
    assert 0 <= result["fracture_dissipation_j_m2"][0] <= LAW["fracture_energy_j_m2"]


def test_complete_failure_cost_is_source_Gc_despite_inserted_prestress():
    initial = np.asarray([[3., 2., -1.]])
    seed = audit.prestress_seed(LAW, initial)
    final_h = np.array([[2*seed["final_separation_m"][0], 0., 0.]])
    final_gap = final_h-seed["reference_offset_m"]
    result = audit.prestress_segment(LAW, initial, np.zeros((1, 3)), final_gap,
                                    seed["initial_maximum_separation_m"])
    assert result["fracture_dissipation_j_m2"][0] == pytest.approx(LAW["fracture_energy_j_m2"], rel=2e-15)
    assert result["stored_energy_j_m2"][0] == pytest.approx(0., abs=1e-15)
    assert result["mechanical_work_j_m2"][0] == pytest.approx(LAW["fracture_energy_j_m2"], rel=2e-13)
    naive_cost = LAW["fracture_energy_j_m2"]-seed["reference_cohesive_energy_j_m2"][0]
    assert abs(result["fracture_dissipation_j_m2"][0]-naive_cost) > .1


def test_prestress_compression_unloads_without_becoming_a_fracture_credit():
    initial = np.asarray([[-4., 0., 0.]])
    seed = audit.prestress_seed(LAW, initial)
    gap = -seed["reference_offset_m"]
    result = audit.prestress_segment(LAW, initial, np.zeros((1, 3)), gap,
                                    seed["initial_maximum_separation_m"])
    assert result["fracture_dissipation_j_m2"][0] == 0
    assert result["mechanical_work_j_m2"][0] == pytest.approx(-seed["reference_compression_energy_j_m2"][0], rel=2e-14)


def test_restart_maximum_prevents_new_fracture_charge_on_reload():
    initial = np.asarray([[3., 2., -1.]])
    seed = audit.prestress_seed(LAW, initial)
    zero, peak = np.zeros((1, 3)), np.asarray([[1., .3, -.2]])
    loading = audit.prestress_segment(LAW, initial, zero, peak, seed["initial_maximum_separation_m"])
    unloading = audit.prestress_segment(LAW, initial, peak, zero, loading["maximum_separation_m"])
    reloading = audit.prestress_segment(LAW, initial, zero, peak, unloading["maximum_separation_m"])
    assert unloading["fracture_dissipation_increment_j_m2"][0] == 0
    assert reloading["fracture_dissipation_increment_j_m2"][0] == 0
    assert (unloading["mechanical_work_j_m2"]+reloading["mechanical_work_j_m2"])[0] == pytest.approx(0., abs=2e-12)


def test_two_tangent_rotation_preserves_seed_and_shifted_work():
    angle = .721
    R = np.array(((1., 0., 0.), (0., np.cos(angle), -np.sin(angle)), (0., np.sin(angle), np.cos(angle))))
    initial, gap = np.asarray([[3., 2., -1.]]), np.asarray([[-.8, 1.7, .4]])
    seed = audit.prestress_seed(LAW, initial)
    a = audit.prestress_segment(LAW, initial, np.zeros((1, 3)), gap, seed["initial_maximum_separation_m"])
    b = audit.prestress_segment(LAW, initial@R.T, np.zeros((1, 3)), gap@R.T, seed["initial_maximum_separation_m"])
    for field in ("mechanical_work_j_m2", "stored_energy_j_m2", "fracture_dissipation_j_m2", "damage"):
        np.testing.assert_allclose(a[field], b[field], rtol=3e-14)
    np.testing.assert_allclose(a["traction_pa"]@R.T, b["traction_pa"], rtol=3e-14)


def test_seed_above_declared_peak_is_not_hidden_by_damage_or_offset():
    with pytest.raises(ValueError, match="exceeds the declared peak"):
        audit.prestress_seed(LAW, np.asarray([[8., 0., 0.]]))


def test_old_history_cannot_erase_the_inserted_prestress_maximum():
    with pytest.raises(ValueError, match="below the inserted reference"):
        audit.prestress_segment(LAW, np.asarray([[3., 2., -1.]]), np.zeros((1, 3)), np.ones((1, 3)), np.zeros(1))


@pytest.mark.parametrize("excess", (0., 2., 8.))
def test_machine_peak_guard_is_recorded_and_cannot_create_initial_damage(excess):
    initial = np.asarray([[LAW["peak_traction_pa"]*(1+excess*np.finfo(float).eps), 0., 0.]])
    seed = audit.prestress_seed(LAW, initial)
    assert seed["initial_maximum_separation_m"][0] <= LAW["peak_traction_pa"]/LAW["cohesive_stiffness_pa_m"]
    np.testing.assert_array_equal(seed["initial_traction_pa"], initial)
    assert seed["reference_peak_snap_factor"][0] <= 1
    np.testing.assert_allclose(seed["reference_traction_pa"], initial, rtol=32*np.finfo(float).eps)
    result = audit.prestress_segment(LAW, initial, np.zeros((1, 3)), np.zeros((1, 3)),
                                    seed["initial_maximum_separation_m"])
    assert result["damage"][0] == 0
    assert result["fracture_dissipation_j_m2"][0] == 0
    assert result["mechanical_work_j_m2"][0] == 0


def mechanical_case():
    """A small unphysical DTO used solely to test adversarial work claims."""
    surface = reference_surface(); shape = surface["area_weights_m2"].shape
    initial = np.broadcast_to(np.asarray((1., .2, -.1)), shape+(3,)).copy()
    seed = audit.prestress_seed(LAW, initial)
    plain = lambda value: value.tolist() if isinstance(value, np.ndarray) else value
    seed_dto = {k: plain(v) for k, v in seed.items()}
    seed_dto.update(law=LAW, kernel_only=True, physical_release_admitted=False, normal_contact_admitted=False)
    seed_dto["seed_sha256"] = audit.digest(seed_dto)
    weights = surface["area_weights_m2"]
    axes = np.stack((surface["edge_normals_xyz"], surface["edge_tangent1_xyz"],
        surface["edge_tangent2_xyz"]), axis=-1)[surface["patch_edge_indices"]]
    old_u, old_local = np.zeros(surface["bank_ndof"]), np.zeros(shape+(3,))
    old_kappa = seed["initial_maximum_separation_m"]
    large = np.zeros((surface["bank_ndof"]//6, 6)); large[4:, :3] = (200., 120., 90.)
    paths = []; total_work = 0.
    for name, u in (("zero_birth", old_u.copy()), ("complete_failure", large.ravel())):
        world = audit.surface_jump(surface, u)
        local = np.einsum("pqa,pab->pqb", world, axes)
        response = audit.prestress_segment(LAW, initial, old_local, local, old_kappa)
        world_t = np.einsum("pqb,pab->pqa", response["traction_pa"], axes)
        path = dict(name=name, old_bank_displacement_from_fixture_origin_m=old_u.tolist(),
            bank_displacement_from_fixture_origin_m=u.tolist(), old_kappa_m=old_kappa.tolist(),
            world_jump_m=world.tolist(), local_jump_m=local.tolist(),
            traction_world_pa=world_t.tolist(), bank_force_n=audit.surface_force(surface, world_t).tolist())
        for reported, key in (("kappa_m", "maximum_separation_m"), ("damage", "damage"),
                ("traction_local_pa", "traction_pa"), ("tangent_local_pa_m", "tangent_pa_m"),
                ("stored_energy_density_j_m2", "stored_energy_j_m2"),
                ("cumulative_dissipation_density_j_m2", "fracture_dissipation_j_m2"),
                ("dissipation_increment_density_j_m2", "fracture_dissipation_increment_j_m2"),
                ("mechanical_work_increment_density_j_m2", "mechanical_work_j_m2")):
            path[reported] = response[key].tolist()
        for reported, key in (("stored_energy_j", "stored_energy_j_m2"),
                ("cumulative_dissipation_j", "fracture_dissipation_j_m2"),
                ("dissipation_increment_j", "fracture_dissipation_increment_j_m2"),
                ("mechanical_work_increment_j", "mechanical_work_j_m2")):
            path[reported] = float(np.sum(weights*response[key]))
        paths.append(path); total_work += path["mechanical_work_increment_j"]
        old_u, old_local, old_kappa = u, local, response["maximum_separation_m"]
    terminal = dict(format="genesis-computational-prestress-insertion-history-0.1", scope=audit.SCOPE,
        parent_checkpoint_sha256="mechanical_fixture_only", surface_binding_sha256="mechanical_surface_only",
        seed_sha256=seed_dto["seed_sha256"], bank_displacement_from_fixture_origin_m=old_u.tolist(),
        local_gap_m=old_local.tolist(), maximum_separation_m=old_kappa.tolist(),
        stored_interface_energy_j=path["stored_energy_j"], cumulative_fracture_work_j=path["cumulative_dissipation_j"],
        cumulative_cohesive_work_j=total_work,
        stored_energy_semantics="signed_change_from_prestressed_reference_not_new_historical_energy",
        physical_creation_admitted=False, physical_release_admitted=False, normal_contact_admitted=False,
        accepted_path_segments=[p["bank_displacement_from_fixture_origin_m"] for p in paths])
    terminal["state_sha256"] = audit.digest(terminal)
    case = dict(law=LAW, seed=seed_dto, paths=paths, terminal_state=terminal,
        geometry={"binding_sha256":"mechanical_surface_only"}, direct_resumed_computational_state_byte_equal=True)
    return case, surface, initial


def test_whole_computational_work_packet_checks_true_cost_and_terminal_history():
    case, surface, initial = mechanical_case()
    result = audit.audit_seeded_paths(case, surface, initial, "mechanical_fixture_only")
    assert result["terminal_whole_schema_seal_and_path_catalogue_checked"]
    assert result["paired_interface_area_counted_once"]


@pytest.mark.parametrize("forgery", ("retuning", "naive_cost", "erased_path"))
def test_resealed_false_creation_accounting_is_rejected(forgery):
    case, surface, initial = mechanical_case(); case = copy.deepcopy(case)
    if forgery == "retuning":
        case["seed"]["auxiliary_fracture_energy_j_m2"] = np.full(surface["area_weights_m2"].shape,
            LAW["fracture_energy_j_m2"]).tolist()
        case["seed"]["seed_sha256"] = audit.digest({k:v for k,v in case["seed"].items() if k != "seed_sha256"})
        case["terminal_state"]["seed_sha256"] = case["seed"]["seed_sha256"]
        marker = "auxiliary_fracture_energy_j_m2 differs"
    elif forgery == "naive_cost":
        case["paths"][-1]["cumulative_dissipation_density_j_m2"] = (
            np.asarray(case["paths"][-1]["cumulative_dissipation_density_j_m2"])
            -np.asarray(case["seed"]["reference_cohesive_energy_j_m2"])).tolist()
        marker = "cumulative_dissipation_density_j_m2 differs"
    else:
        case["terminal_state"]["accepted_path_segments"] = []
        marker = "accepted_path_segments differs"
    case["terminal_state"]["state_sha256"] = audit.digest({k:v for k,v in case["terminal_state"].items()
        if k != "state_sha256"})
    with pytest.raises(ValueError, match=marker):
        audit.audit_seeded_paths(case, surface, initial, "mechanical_fixture_only")


def test_force_consistent_double_insertion_is_rejected_even_if_its_balance_is_resealed():
    surface = reference_surface()
    initial = np.broadcast_to((2., -.3, .7), surface["area_weights_m2"].shape+(3,)).copy()
    force = audit.surface_force(surface, initial)
    residual, bulk = np.sin(np.arange(surface["bank_ndof"])), np.cos(np.arange(surface["bank_ndof"]))
    reaction = dict(constraint_force_n=force, remaining_residual_n=residual+force,
        bank_residual_n=residual, bank_average_internal_force_n=bulk)
    wrong_force = 2*force
    insertion = dict(bank_force_n=wrong_force, old_tied_residual_n=residual+force,
        new_seeded_residual_n=residual+wrong_force, bulk_plus_seeded_internal_force_n=bulk+wrong_force,
        replaces_constraint_force_once=True, retains_parent_residual=True,
        relative_stored_interface_energy_j=0., fracture_work_j=0., mechanical_work_j=0.,
        existing_Genesis_memory_or_energy_reinitialized=False,
        physical_creation_admitted=False, physical_release_admitted=False,
        force_replacement_relative_errors_A_P=[0.,0.], retained_residual_relative_errors_A_P=[0.,0.])
    with pytest.raises(ValueError, match="complete single seeded replacement A force differs"):
        audit.audit_interface_replacement(insertion, reaction, surface, initial)


def test_oracle_has_no_producer_imports():
    tree = ast.parse(Path(audit.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(not alias.name.startswith("tectonics") for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("tectonics")
