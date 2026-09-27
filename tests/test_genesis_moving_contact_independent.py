"""Independent physical invariants of the moving local-contact chart.

These checks use world vectors, finite rigid rotations, and virtual work
rather than repeating the implementation's component transport formulas.
"""
from dataclasses import fields

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tectonics.genesis_crack_path import ReferenceCrackPath
from tectonics.genesis_material import rebuild_material_mesh
from tectonics.genesis_moving_path_diagnostics import observe_material_path
from tectonics.genesis_moving_support import MaterialPathSupport
from tectonics.genesis_moving_tied import MovingTiedLoading, MovingTiedMechanics
from tectonics.genesis_path_activation import activate_tied_onset
from tectonics.genesis_path_mesh import insert_crack_path
from tectonics.genesis_shell import Membrane
from tectonics.mesh import build_icosphere


@pytest.fixture(scope="module")
def support():
    mesh = build_icosphere(1)
    points = np.array([[.6, .2, .2], [.2, .5, .3], [.25, .2, .55]])
    points = points @ mesh.vertices[mesh.faces[0]]
    points /= np.linalg.norm(points, axis=1)[:, None]
    path = ReferenceCrackPath(points, 5300.)
    insertion = insert_crack_path(mesh, path)
    return MaterialPathSupport(mesh, insertion, 5.3e6, .25)


def _world_relative(basis, components):
    """Enrichment is the signed difference between two colocated banks."""
    mesh = basis.subdivision.mesh
    frames = Membrane(mesh, .25).vertex_basis[basis.enrichment_vertices]
    return np.einsum("vij,vj->vi", frames, np.asarray(components).reshape(-1, 2))


@pytest.mark.parametrize("stationary_vertex_spin", [False, True])
def test_enrichment_transport_follows_a_finite_rigid_world_rotation(support, stationary_vertex_spin):
    from tectonics.genesis_moving_contact import transport_enrichment

    old = support.basis_at(support.parent_mesh, support.radius_m)
    vector = np.random.default_rng(712).normal(size=old.ndof-old.nparent) * .3
    if stationary_vertex_spin:
        # Rotation about this vertex leaves its position unchanged. A minimal
        # old-position/new-position sphere rotation would incorrectly lose
        # the physical spin of its separation vector.
        axis = old.subdivision.mesh.vertices[old.enrichment_vertices[0]]
        rotation = Rotation.from_rotvec(.73 * axis).as_matrix()
    else:
        rotation = Rotation.from_rotvec([.43, -.57, .22]).as_matrix()
    rotated = rebuild_material_mesh(support.parent_mesh,
        support.parent_mesh.vertices @ rotation.T)
    current = support.basis_at(rotated, support.radius_m)
    carried = transport_enrichment(old, current, vector,
        support.radius_m, support.radius_m)
    np.testing.assert_allclose(_world_relative(current, carried),
        _world_relative(old, vector) @ rotation.T, rtol=2e-12, atol=2e-13)
    np.testing.assert_allclose(np.linalg.norm(carried.reshape(-1, 2), axis=1),
        np.linalg.norm(vector.reshape(-1, 2), axis=1), rtol=2e-12, atol=2e-13)
    np.testing.assert_array_equal(vector,
        np.random.default_rng(712).normal(size=old.ndof-old.nparent) * .3)


def test_enrichment_transport_keeps_a_physical_separation_under_radius_change(support):
    from tectonics.genesis_moving_contact import transport_enrichment

    old = support.basis_at(support.parent_mesh, support.radius_m)
    current = support.basis_at(support.parent_mesh, .98 * support.radius_m)
    vector = np.random.default_rng(43).normal(size=old.ndof-old.nparent)
    carried = transport_enrichment(old, current, vector,
        support.radius_m, .98 * support.radius_m)
    # This owner's enrichment components represent metres of independent
    # separation, not an angle to be rescaled by the changing planet radius.
    np.testing.assert_allclose(_world_relative(current, carried),
        _world_relative(old, vector), rtol=2e-12, atol=2e-13)


def _birth_case(support):
    """A controlled prestress balanced by parent forces, with exact zero motion.

    This calibrates an admissible traction-preserving birth with real solves.
    It is a mechanical test, not an alternative thermal genesis trajectory.
    """
    from tectonics.genesis_moving_contact import MovingContactMechanics

    mesh, radius = support.parent_mesh, support.radius_m
    tied = MovingTiedMechanics(mesh, radius)
    volume = mesh.areas_unit_sphere * radius**2 * 1e4
    young = np.full(mesh.cell_count, 60e9)

    def solve(amplitude):
        elastic = np.tile([amplitude, amplitude, 0.], (mesh.cell_count, 1))

        def force(current, current_radius, depth, membrane):
            result = np.zeros(membrane.ndof)
            if current.cell_count == mesh.cell_count:
                # Direct quadrature of a held reference prestress load.
                # Fine relative load is zero; parent quadrature is retained
                # by the assumed-strain research discretization.
                stress = (elastic @ membrane.d.T) * young[:, None]
                for face in range(current.cell_count):
                    result[membrane.dofs[face]] += (
                        membrane.b[face].T @ stress[face]) * volume[face] / current_radius
            return result

        state = tied.trial(tied.initial(elastic), MovingTiedLoading(1.,
            volume, young, elastic, np.ones(len(volume)), force))
        observation = observe_material_path(tied, state, support,
            np.zeros(mesh.cell_count), force)
        return state, observation, force

    _, calibration, _ = solve(1e-5)
    state, observed, force = solve(1e-5 * (1-5e-7)
        / calibration.onset.maximum_observed_ratio)
    birth = activate_tied_onset(observed.model, observed.state,
        observed.stress_pa, observed.water_per_trace)
    model = MovingContactMechanics(tied, support, birth.model, birth.state, state)
    return model, force, support.extensive(volume), young[support.parent_face]


def _open_once(case, years=.1):
    from tectonics.genesis_moving_contact import MovingContactLoading

    model, force, volume, young = case
    before = model.initial_state
    factor = 1.01

    def increased_force(*arguments):
        return factor * force(*arguments)

    return model.trial(before, MovingContactLoading(years, volume, young,
        factor * before.elastic_strain, np.ones(len(volume)),
        np.zeros(model.history_model.ntraces), increased_force))


@pytest.mark.parametrize("growth", [1., 1.01])
def test_balanced_birth_does_not_move_or_damage_when_fresh_solid_is_stress_free(support, growth):
    from tectonics.genesis_moving_contact import MovingContactLoading

    model, force, volume, young = _birth_case(support)
    before = model.initial_state
    loading = MovingContactLoading(1., growth * volume, young,
        before.elastic_strain/growth, np.ones(len(volume)),
        np.zeros(model.history_model.ntraces), force)
    after = model.trial(before, loading)
    # Old stress times old solid volume is held fixed. New material carries
    # no initial stress or interface traction, so this is an exact equilibrium
    # regardless of how the material amount is partitioned into cohorts.
    np.testing.assert_array_equal(after.vertices, before.vertices)
    assert after.radius_m == before.radius_m
    np.testing.assert_array_equal(after.enrichment_m, before.enrichment_m)
    np.testing.assert_allclose(after.elastic_strain, before.elastic_strain/growth,
        rtol=2e-14, atol=1e-18)
    for field in fields(before.history.initial):
        np.testing.assert_array_equal(getattr(after.history.initial, field.name),
            getattr(before.history.initial, field.name), err_msg=field.name)
    assert after.external_work_j == before.external_work_j
    assert after.drag_work_j == before.drag_work_j
    assert after.mechanical_remainder_j == before.mechanical_remainder_j
    assert model.force_diagnostics(after)["raw_relative_residual"] < 1e-12
    if growth == 1:
        assert not len(after.history.added.trace_index)
    else:
        added = after.history.added
        assert len(added.trace_index) == model.history_model.ntraces
        np.testing.assert_array_equal(added.traction_pa, 0.)
        np.testing.assert_array_equal(added.fracture_work_j, 0.)
        np.testing.assert_array_equal(added.damage, 0.)
        np.testing.assert_allclose(added.area_ref_m2,
            (growth-1) * before.history.initial.area_ref_m2,
            rtol=2e-13, atol=1e-6)
        np.testing.assert_array_equal(added.bonded, True)
    model._validate_state(after)


@pytest.mark.parametrize("opened", [False, True])
def test_assembled_contact_force_matches_independent_world_bank_virtual_work(support, opened):
    case = _birth_case(support)
    model = case[0]
    state = _open_once(case) if opened else model.initial_state
    basis = model.basis_for(state)
    geometry, _ = model.geometry_for(state)
    virtual = np.random.default_rng(552).normal(size=basis.ndof)
    split_virtual = basis.displacement_operator @ virtual
    world = np.einsum("vij,vj->vi", basis.membrane.vertex_basis,
        split_virtual[:-1].reshape(-1, 2))
    banks = world[basis.topology.bank_vertices]
    relative = (banks[:, 1]-banks[:, 0]).reshape(-1, 3)
    force = np.zeros_like(relative)
    for cohorts in (state.history.initial, state.history.added):
        i = cohorts.trace_index
        world_force = (cohorts.traction_pa[:, :1] * geometry.interface_normal[i]
            + cohorts.traction_pa[:, 1:] * geometry.interface_tangent[i])
        np.add.at(force, i, cohorts.area_ref_m2[:, None] * world_force)
    direct = float(np.sum(force * relative))
    generalized = float(model.force_diagnostics(state)["contact_force_n"] @ virtual)
    assert generalized == pytest.approx(direct, rel=2e-13, abs=1.)
    assert abs(direct) > 1e10


def test_loaded_opening_is_covariant_under_finite_rotation_of_entire_planet(support):
    rotation = Rotation.from_rotvec([.21, -.39, .57]).as_matrix()
    rotated_mesh = rebuild_material_mesh(support.parent_mesh,
        support.parent_mesh.vertices @ rotation.T)
    rotated_insertion = support.basis_at(rotated_mesh, support.radius_m).insertion
    rotated_support = MaterialPathSupport(rotated_mesh, rotated_insertion,
        support.radius_m, support.poisson_ratio)
    first_case, second_case = _birth_case(support), _birth_case(rotated_support)
    first, second = _open_once(first_case, 10.), _open_once(second_case, 10.)
    first_model, second_model = first_case[0], second_case[0]
    first_basis, second_basis = first_model.basis_for(first), second_model.basis_for(second)
    assert first_model.jump(first)[:, 0].max() > 1e-5
    assert first.history.initial.fracture_work_j.sum() > 0
    np.testing.assert_allclose(second.vertices, first.vertices @ rotation.T,
        rtol=0, atol=2e-11)
    assert second.radius_m == pytest.approx(first.radius_m, rel=2e-12)
    np.testing.assert_allclose(_world_relative(second_basis, second.enrichment_m),
        _world_relative(first_basis, first.enrichment_m) @ rotation.T,
        rtol=2e-6, atol=1e-9)
    np.testing.assert_allclose(second.elastic_strain, first.elastic_strain,
        rtol=2e-6, atol=1e-11)
    np.testing.assert_allclose(second_model.jump(second), first_model.jump(first),
        rtol=2e-6, atol=1e-9)
    np.testing.assert_allclose(second.history.initial.damage,
        first.history.initial.damage, rtol=2e-6, atol=1e-9)
    assert second.drag_work_j == pytest.approx(first.drag_work_j, rel=2e-5, abs=1e6)
    radial = first_model.nparent-1
    force = first.last_external_force_n
    external_roundoff = (2*np.spacing(first.radius_m)*abs(force[radial])
        +8*np.finfo(float).eps*first.radius_m*np.linalg.norm(force[:radial]))
    # The controlled load has a radial force of planetary magnitude. One ULP
    # in a micrometre radial increment is a resolvable absolute work floor.
    assert second.external_work_j == pytest.approx(first.external_work_j,
        rel=2e-5, abs=external_roundoff)
    # Near-zero parent motions subtract planetary-scale positions before
    # contracting strain against the entire prestressed volume. Their work
    # has an absolute floating-point floor, even with covariant local gaps.
    # Bound that floor by material stress-volume, not by the tiny net work.
    work_roundoff = 32*np.finfo(float).eps * np.sum(first.last_volume_m3
        * np.linalg.norm(first_model.stress(first), axis=1))
    assert second.bulk_work_j == pytest.approx(first.bulk_work_j,
        rel=2e-5, abs=work_roundoff)
