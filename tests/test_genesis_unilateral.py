"""Independent unilateral-contact, virtual-extension and load-path controls."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from analysis.genesis_shell_release_fixture import build_shell_release_fixture
from tectonics.genesis_seams import rebuild_seam_mesh
from tectonics.genesis_shell_release import FrozenShell
from tectonics.genesis_unilateral import (
    ContactSolveParameters, LoadPathParameters, UnilateralShell,
)


@pytest.fixture(scope="module")
def fixture():
    return build_shell_release_fixture(2)


def _shell(fixture, amplitude=1., strain=None, **kwargs):
    return FrozenShell.from_uniform(
        fixture.mesh, fixture.radius_m, fixture.depth_m, 6e10, .25,
        fixture.traction(fixture.mesh.vertices, amplitude),
        elastic_strain=strain, **kwargs,
    )


def _energy_roundoff(model, result):
    return 200*np.finfo(float).eps*max(
        abs(result.before.reduced_potential_j),
        abs(result.after.reduced_potential_j), model.shell.initial_energy_j,
        np.finfo(float).tiny,
    )


def _world(state, vector):
    return np.einsum("vij,vj->vi", state.membrane.vertex_basis,
                     vector[:-1].reshape(-1, 2))


def test_tensile_contact_equals_free_bank_solution(fixture):
    shell = _shell(fixture)
    free = shell.compare_extension(fixture.seed_cuts, fixture.trial_cuts)
    contact = UnilateralShell(shell).compare_extension(fixture.seed_cuts, fixture.trial_cuts)
    assert contact.release_j == pytest.approx(free.release_j, rel=1e-11)
    for previous, current in ((free.before, contact.before), (free.after, contact.after)):
        np.testing.assert_allclose(current.displacement_m, previous.displacement_m,
                                   rtol=1e-10, atol=1e-12)
        np.testing.assert_array_equal(current.normal_reaction_n, 0.)
        assert current.active_contact_count == 0
        assert current.contact_work_j == 0
        assert current.admissible_contact


def test_compression_closes_banks_without_false_positive_release(fixture):
    shell = _shell(fixture, amplitude=-1.)
    free = shell.compare_extension(fixture.seed_cuts, fixture.trial_cuts)
    assert free.release_j > 1e6
    assert free.after.min_gap_m < -1e-6
    model = UnilateralShell(shell)
    result = model.compare_extension(fixture.seed_cuts, fixture.trial_cuts)
    assert result.admissible_open_crack
    assert abs(result.release_j) <= _energy_roundoff(model, result)
    intact = shell.solve([])
    for state in (result.before, result.after):
        assert state.active_contact_count > 0
        assert np.all(state.normal_reaction_n >= 0)
        assert state.min_gap_m >= -shell.penetration_tolerance_m
        assert state.complementarity_relative_error < 1e-12
        # For this symmetric, purely normal loading the banks recover the
        # intact deformation. Generic compression can still release shear.
        np.testing.assert_allclose(state.displacement_m[state.membrane.dofs],
                                   intact.displacement_m[intact.membrane.dofs],
                                   rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("edge_count", [0, 1, 2, 3])
def test_unloaded_contact_has_no_energy_or_reaction(fixture, edge_count):
    state = UnilateralShell(_shell(fixture, amplitude=0.)).solve(fixture.cuts(edge_count))
    np.testing.assert_array_equal(state.displacement_m, 0.)
    np.testing.assert_array_equal(state.normal_gap_m, 0.)
    np.testing.assert_array_equal(state.normal_reaction_n, 0.)
    assert state.stored_energy_j == state.reduced_potential_j == 0
    assert state.contact_work_j == state.complementarity_relative_error == 0
    assert state.active_contact_count == 0
    assert state.admissible_contact


@pytest.mark.parametrize("amplitude", [-1., 0., 1.])
def test_same_topology_never_releases_energy(fixture, amplitude):
    result = UnilateralShell(_shell(fixture, amplitude)).compare_extension(
        fixture.seed_cuts, fixture.seed_cuts)
    assert result.release_j == result.added_area_m2 == result.mean_release_j_m2 == 0
    np.testing.assert_array_equal(result.before.displacement_m, result.after.displacement_m)


def test_shared_tips_and_duplicate_normals_do_not_make_contact_singular(fixture):
    model = UnilateralShell(_shell(fixture, amplitude=-1.))
    single = model.compare_extension(fixture.cuts(0), fixture.cuts(1))
    assert single.after.normal_operator.nnz == 0
    assert single.after.dual_rank == 0
    assert single.release_j == 0
    state = model.solve(fixture.trial_cuts)
    operator = state.normal_operator.toarray()
    nonzero = operator[np.linalg.norm(operator, axis=1) > 0]
    assert len(nonzero) > state.dual_rank > 0
    assert np.linalg.matrix_rank(nonzero) == state.dual_rank
    assert state.equilibrium_residual < 1e-11
    reverse = model.solve(fixture.trial_cuts[::-1, ::-1])
    np.testing.assert_array_equal(reverse.displacement_m, state.displacement_m)
    np.testing.assert_array_equal(reverse.normal_reaction_n, state.normal_reaction_n)


def test_nearly_parallel_bank_normals_retain_real_contact_directions(fixture):
    # A tiny resolved kink makes formerly duplicate endpoint rows distinct.
    # Dropping their weak compliance eigenmodes at 1e-12 loses real inequality
    # directions and used to reject a feasible, small-strain loaded shell.
    vertices = fixture.mesh.vertices.copy()
    vertices[fixture.vertex_path[1]] += 1e-7*fixture.normal_xyz
    vertices /= np.linalg.norm(vertices, axis=1)[:, None]
    mesh = rebuild_seam_mesh(fixture.mesh, vertices)
    strain = np.random.default_rng(1).normal(size=(mesh.cell_count, 3))*1e-7
    shell = FrozenShell.from_uniform(mesh, fixture.radius_m, fixture.depth_m,
        6e10, .25, np.zeros_like(vertices), elastic_strain=strain)
    state = UnilateralShell(shell).solve(fixture.trial_cuts)
    assert state.dual_rank == 4
    assert state.active_contact_count > 0
    assert state.min_gap_m >= -shell.penetration_tolerance_m
    assert state.equilibrium_residual < 1e-11
    assert state.complementarity_relative_error < 1e-11
    assert state.admissible_contact


def test_gap_sign_and_contact_forces_obey_virtual_work(fixture):
    model = UnilateralShell(_shell(fixture, amplitude=-1.))
    state = model.solve(fixture.trial_cuts)
    variation = np.random.default_rng(91).normal(size=len(state.displacement_m))
    measured_gap, _ = model.shell._gaps(state.topology, state.membrane, variation)
    np.testing.assert_allclose(state.normal_operator@variation, measured_gap,
                               rtol=1e-13, atol=1e-14)
    reaction_force = state.normal_operator.T@state.normal_reaction_n
    assert variation@reaction_force == pytest.approx(
        measured_gap@state.normal_reaction_n, rel=1e-13)
    np.testing.assert_allclose(state.bulk_matrix@state.displacement_m
        +state.initial_bulk_force-state.external_force, reaction_force,
        rtol=1e-10, atol=.01)
    # Equal and opposite tractions at identical reference positions cannot
    # supply a spurious net force or rigid-rotation moment.
    force_xyz = _world(state, reaction_force)
    np.testing.assert_allclose(force_xyz.sum(axis=0), 0., atol=.001)
    torque = np.cross(state.topology.mesh.vertices, force_xyz).sum(axis=0)
    np.testing.assert_allclose(torque, 0., atol=.001)


def test_kinked_extension_requires_contact_reaction_energy(fixture):
    path = np.array([0, 59, 52, 44, 51, 56, 16])
    cuts = np.column_stack((path[:-1], path[1:]))
    strain = np.random.default_rng(7).normal(size=(fixture.mesh.cell_count, 3))*1e-7
    model = UnilateralShell(_shell(fixture, amplitude=0., strain=strain))
    result = model.compare_extension(cuts[:4], cuts)
    assert result.release_j > 1e12
    assert result.contact_release_term_j > .1*result.release_j
    assert result.release_j == pytest.approx(
        result.relaxation_energy_j+result.contact_release_term_j, rel=1e-10)
    assert result.release_j == pytest.approx(result.potential_difference_j, rel=1e-10)
    assert result.lifted_min_gap_m >= -model.shell.penetration_tolerance_m
    assert result.material_area_relative_error == result.material_volume_relative_error == 0
    assert result.after.active_contact_count > 0
    assert np.max(np.abs(result.after.tangential_jump_m)) > .01
    lifted = model.shell._prolongation(result.before, result.after)@result.before.displacement_m
    assert result.contact_release_term_j == pytest.approx(
        result.after.normal_reaction_n@(result.after.normal_operator@lifted), rel=1e-13)
    assert abs(result.after.contact_work_j) < result.release_j*1e-12


@pytest.mark.parametrize("factor", [.125, 8.])
def test_compressive_displacement_and_reaction_scale_with_load(fixture, factor):
    model = UnilateralShell(_shell(fixture, amplitude=-1.))
    reference = model.solve(fixture.trial_cuts)
    scaled = model.solve(fixture.trial_cuts, load_factor=factor)
    np.testing.assert_allclose(scaled.displacement_m, reference.displacement_m*factor,
                               rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(scaled.normal_operator.T@scaled.normal_reaction_n,
        (reference.normal_operator.T@reference.normal_reaction_n)*factor,
        rtol=1e-10, atol=.01)
    assert scaled.reduced_potential_j == pytest.approx(
        reference.reduced_potential_j*factor**2, rel=1e-11)


def test_uniform_prestress_relaxation_survives_rank_deficient_contact(fixture):
    strain = np.tile([2e-5, 2e-5, 0.], (fixture.mesh.cell_count, 1))
    model = UnilateralShell(_shell(fixture, amplitude=0., strain=strain))
    result = model.compare_extension(fixture.seed_cuts, fixture.trial_cuts)
    for state in (result.before, result.after):
        assert state.displacement_m[-1] == pytest.approx(-2e-5*fixture.radius_m, rel=1e-12)
        assert np.max(np.abs(state.displacement_m[:-1])) < 1e-9
        assert state.stored_energy_j < model.shell.initial_energy_j*1e-24
    assert abs(result.release_j) <= _energy_roundoff(model, result)


def test_loading_zero_keeps_frozen_prestress(fixture):
    strain = np.random.default_rng(4).normal(size=(fixture.mesh.cell_count, 3))*1e-7
    loaded = UnilateralShell(_shell(fixture, amplitude=5., strain=strain))
    no_traction = UnilateralShell(_shell(fixture, amplitude=0., strain=strain))
    start = loaded.solve(fixture.trial_cuts, load_factor=0.)
    reference = no_traction.solve(fixture.trial_cuts)
    assert np.linalg.norm(start.displacement_m) > .01
    np.testing.assert_array_equal(start.external_force, 0.)
    np.testing.assert_allclose(start.displacement_m, reference.displacement_m,
                               rtol=1e-10, atol=1e-11)
    np.testing.assert_array_equal(start.initial_bulk_force, reference.initial_bulk_force)


def test_continuation_subdivides_load_without_changing_final_solution(fixture):
    model = UnilateralShell(_shell(fixture, amplitude=-1.))
    limits = LoadPathParameters(max_load_increment=1., min_load_increment=1e-5,
        max_incremental_strain=1e-10, max_incremental_motion_edge_fraction=1e-9)
    result = model.continue_loading(fixture.trial_cuts, parameters=limits)
    assert result.reached_target
    assert result.stop_reason == "target_reached"
    assert any(not attempt.accepted and attempt.reason == "increment_limit"
               for attempt in result.attempts)
    accepted = [attempt for attempt in result.attempts if attempt.accepted]
    assert len(accepted) > 1
    assert all(attempt.increment_strain <= limits.max_incremental_strain for attempt in accepted)
    assert all(attempt.increment_motion_edge_fraction <= limits.max_incremental_motion_edge_fraction
               for attempt in accepted)
    direct = model.solve(fixture.trial_cuts)
    np.testing.assert_array_equal(result.last_accepted.displacement_m, direct.displacement_m)
    np.testing.assert_array_equal(result.last_accepted.normal_reaction_n, direct.normal_reaction_n)
    assert result.last_accepted.load_factor == 1.


@pytest.mark.parametrize("increment", [.1, .025])
def test_small_increments_cannot_hide_excessive_total_deformation(fixture, increment):
    model = UnilateralShell(_shell(fixture, amplitude=4e6))
    direct = model.solve(fixture.trial_cuts)
    assert not direct.admissible_contact
    result = model.continue_loading(fixture.trial_cuts, parameters=LoadPathParameters(
        max_load_increment=increment, max_incremental_strain=.002,
        max_incremental_motion_edge_fraction=.005))
    assert not result.reached_target
    assert result.stop_reason == "total_reference_limit"
    assert 0 < result.last_accepted.load_factor < 1.
    assert result.last_accepted.admissible_contact
    assert result.attempts[-1].total_strain > model.shell.max_strain
    assert result.attempts[-1].increment_strain < .002
    repeated = model.solve(fixture.trial_cuts)
    np.testing.assert_array_equal(repeated.displacement_m, direct.displacement_m)


def test_invalid_initial_prestress_is_not_erased_to_start_loading(fixture):
    strain = np.tile([.006, .006, 0.], (fixture.mesh.cell_count, 1))
    model = UnilateralShell(_shell(fixture, amplitude=1., strain=strain))
    result = model.continue_loading(fixture.trial_cuts)
    assert not result.reached_target
    assert result.stop_reason == "initial_reference_limit"
    assert result.last_accepted is None
    assert not result.attempts
    assert not result.initial.admissible_contact
    assert result.initial.load_factor == 0


def test_zero_target_requires_no_fictitious_step(fixture):
    result = UnilateralShell(_shell(fixture)).continue_loading(fixture.trial_cuts, target_load_factor=0.)
    assert result.reached_target
    assert not result.attempts
    assert result.initial is result.last_accepted


def test_continuation_budgets_fail_without_advancing_rejected_state(fixture):
    model = UnilateralShell(_shell(fixture))
    too_small = LoadPathParameters(max_load_increment=.1, min_load_increment=.05,
                                  max_incremental_strain=1e-20)
    result = model.continue_loading(fixture.trial_cuts, parameters=too_small)
    assert not result.reached_target
    assert result.stop_reason == "minimum_load_increment"
    assert result.last_accepted is result.initial
    assert all(not attempt.accepted for attempt in result.attempts)
    limited = model.continue_loading(fixture.trial_cuts, parameters=LoadPathParameters(
        max_load_increment=.1, max_attempts=1))
    assert not limited.reached_target
    assert limited.stop_reason == "attempt_budget"
    assert limited.last_accepted.load_factor == .1


def test_rotation_preserves_compressive_contact_force_and_motion(fixture):
    rotation = Rotation.from_rotvec([.17, -.31, .52]).as_matrix()
    rotated_fixture = build_shell_release_fixture(2, rotation=rotation)
    original = UnilateralShell(_shell(fixture, amplitude=-1.)).solve(fixture.trial_cuts)
    rotated = UnilateralShell(_shell(rotated_fixture, amplitude=-1.)).solve(rotated_fixture.trial_cuts)
    assert rotated.reduced_potential_j == pytest.approx(original.reduced_potential_j, rel=1e-11)
    np.testing.assert_allclose(_world(rotated, rotated.displacement_m),
        _world(original, original.displacement_m)@rotation.T, rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(_world(rotated, rotated.normal_operator.T@rotated.normal_reaction_n),
        _world(original, original.normal_operator.T@original.normal_reaction_n)@rotation.T,
        rtol=1e-10, atol=.01)


def test_healing_and_diagnostic_row_overflow_are_rejected(fixture):
    model = UnilateralShell(_shell(fixture))
    with pytest.raises(ValueError, match="contain every existing cut"):
        model.compare_extension(fixture.trial_cuts, fixture.seed_cuts)
    with pytest.raises(ValueError, match="row budget"):
        UnilateralShell(_shell(fixture), ContactSolveParameters(max_contact_rows=1)).solve(fixture.trial_cuts)


@pytest.mark.parametrize("load", [-1., np.nan, np.inf, True])
def test_invalid_load_factors_are_rejected(fixture, load):
    model = UnilateralShell(_shell(fixture))
    with pytest.raises(ValueError, match="load_factor"):
        model.solve(fixture.seed_cuts, load_factor=load)


@pytest.mark.parametrize("edge_count", [0, 2])
def test_load_overflow_cannot_be_reported_as_admissible_contact(fixture, edge_count):
    model = UnilateralShell(_shell(fixture))
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError):
            model.solve(fixture.cuts(edge_count), load_factor=1e308)
