"""Bulk-equilibrium controls independent of the contact-birth implementation."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from tectonics.genesis_crack_path import ReferenceCrackPath
from tectonics.genesis_extrinsic_contact_law import extrinsic_damage, extrinsic_dissipation
from tectonics.genesis_path_activation import activate_tied_onset, load_born_path
from tectonics.genesis_path_basis import EmbeddedPathBasis
from tectonics.genesis_path_birth import classify_tied_onset, recover_tied_tractions
from tectonics.genesis_path_dynamics import PathLoading, PathMechanics
from tectonics.genesis_path_mesh import insert_crack_path
from tectonics.mesh import build_icosphere
from test_genesis_path_dynamics import _model, _material, _same


def _long_model():
    mesh, radius = build_icosphere(1), 5.3e6
    weights = np.array([[.65, .15, .2], [.43, .30, .27], [.2, .5, .3],
                        [.23, .35, .42], [.25, .2, .55]])
    points = weights @ mesh.vertices[mesh.faces[0]]
    points /= np.linalg.norm(points, axis=1)[:, None]
    path = ReferenceCrackPath(points, radius/1000.)
    basis = EmbeddedPathBasis(mesh, insert_crack_path(mesh, path), radius, .25)
    return PathMechanics(basis, np.full(basis.subdivision.mesh.cell_count, 1e4))


def _physical_birth(*, shear=False, long=False):
    """Parent forces hold bulk prestress; reactions come from the actual solve."""
    model = _long_model() if long else _model()
    volume, elasticity = _material(model)
    count = len(volume)
    seed = np.array([1e-5, 1e-5, 6e-5 if shear else 0.])
    water = np.full(count, float(shear))
    trace_water = np.full(len(model.trace_depth_m), float(shear))

    def equilibrium(scale):
        elastic = np.tile(seed*scale, (count, 1))
        _, force = model.basis.bulk(volume, elasticity, elastic)
        force[model.basis.nparent:] = 0.
        loading = PathLoading(1., volume, elasticity, elastic, np.ones(count), force, water)
        state = model.trial(model.initial(elastic), loading)
        stress = np.einsum("fij,fj->fi", elasticity, state.elastic_strain)
        return state, stress, loading

    state, stress, _ = equilibrium(1.)
    recovery = recover_tied_tractions(model, state, stress)
    score = classify_tied_onset(recovery, trace_water).maximum_observed_ratio
    state, stress, loading = equilibrium((1-2e-7)/score)
    before = deepcopy(state)
    transition = activate_tied_onset(model, state, stress, trace_water)
    _same(state, before)
    return model, state, loading, transition


@pytest.mark.parametrize("shear", [False, True])
def test_birth_preserves_real_bulk_equilibrium_under_frozen_loading(shear):
    tied, source, loading, birth = _physical_birth(shear=shear)
    born, state = birth.model, birth.state
    assert birth.governing_mode == ("shear" if shear else "tensile")
    np.testing.assert_array_equal(state.displacement_m, source.displacement_m)
    np.testing.assert_array_equal(state.elastic_strain, source.elastic_strain)
    np.testing.assert_array_equal(born.jump_operator @ state.displacement_m, 0.)
    assert born._contact_energy(state.cohorts, np.zeros(len(born.trace_depth_m)),
                               np.zeros(len(born.trace_depth_m))) == 0.
    for dt in (.001, 20.):
        after = born.trial(state, replace(loading, dt_years=dt, memory=state.elastic_strain))
        np.testing.assert_array_equal(after.displacement_m, state.displacement_m)
        np.testing.assert_array_equal(after.elastic_strain, state.elastic_strain)
        np.testing.assert_array_equal(after.cohorts.plastic_slip_m, 0.)
        np.testing.assert_array_equal(after.cohorts.fracture_work_j, 0.)
        assert after.external_work_j == state.external_work_j
        assert after.drag_work_j == state.drag_work_j
        assert after.mechanical_remainder_j == state.mechanical_remainder_j


def test_long_support_releases_only_one_vertex_and_preserves_virtual_work():
    tied, source, _, birth = _physical_birth(long=True)
    born = birth.model
    assert tied.basis.ndof-tied.basis.nparent >= 6
    free = born.basis.free_dofs(birth.state.active_interval)
    released = free[free >= tied.basis.nparent]
    assert len(released) == 2
    assert birth.state.active_interval.right_m-birth.state.active_interval.left_m < tied.basis.insertion.path.length_m
    rng = np.random.default_rng(935)
    virtual = np.zeros(tied.basis.ndof)
    virtual[released] = rng.normal(size=2)
    jump = (tied.jump_operator @ virtual).reshape(-1, 2)
    work = np.sum(tied.geometry.interface_area_m2[:, None]*born.birth_traction_pa*jump)
    assert work == pytest.approx(source.constraint_reaction_n @ virtual, rel=5e-14)
    untouched = np.asarray(tied.jump_operator[:, released].power(2).sum(axis=1)).ravel().reshape(-1, 2).sum(axis=1) == 0
    assert np.count_nonzero(~untouched) == 2
    np.testing.assert_array_equal(born.birth_traction_pa[untouched], 0.)
    np.testing.assert_array_equal(birth.state.constraint_reaction_n[free], 0.)


def test_changed_load_continues_identically_after_self_describing_restart(tmp_path):
    tied, _, loading, birth = _physical_birth(shear=True)
    model, initial = birth.model, birth.state
    first_loading = replace(loading, dt_years=.01,
        memory=initial.elastic_strain, external_force=loading.external_force*1.002)
    first = model.trial(initial, first_loading)
    assert np.linalg.norm(first.displacement_m-initial.displacement_m) > 0.
    path = tmp_path/"born.npz"
    model.save_state(path, first)
    restored_model, restored = load_born_path(tied, path)
    _same(first, restored)
    next_loading = replace(first_loading, memory=first.elastic_strain,
        external_force=loading.external_force*1.004)
    expected = model.trial(first, next_loading)
    actual = restored_model.trial(restored, next_loading)
    _same(expected, actual)


def test_never_released_trace_cannot_acquire_forged_fracture_history():
    _, _, _, birth = _physical_birth(long=True)
    model, forged = birth.model, deepcopy(birth.state)
    touched = np.any(model.birth_traction_pa != 0., axis=1)
    index = int(np.flatnonzero(~touched)[0])
    # Internally consistent local constitutive state, impossible on a tied tip.
    forged.cohorts.max_opening_m[index] = model.law_parameters.failure_opening_m*2
    forged.cohorts.damage[:] = extrinsic_damage(forged.cohorts.max_opening_m,
        model.birth_traction_pa, model.law_parameters)
    forged.cohorts.fracture_work_j[:] = forged.cohorts.area_ref_m2*extrinsic_dissipation(
        forged.cohorts.max_opening_m, model.birth_traction_pa, model.law_parameters)
    forged.elapsed_years += .1
    with pytest.raises(ValueError):
        model._validate_state(forged)


def test_birth_clock_cannot_contain_postbirth_plastic_history():
    _, _, _, birth = _physical_birth(shear=True)
    model, forged = birth.model, deepcopy(birth.state)
    index = int(np.argmax(np.linalg.norm(model.birth_traction_pa, axis=1)))
    forged.cohorts.cumulative_slip_m[index] = .1
    # Net signed slip and current traction remain unchanged; this could only
    # describe a finite excursion and reversal after contact activation.
    with pytest.raises(ValueError):
        model._validate_state(forged)
