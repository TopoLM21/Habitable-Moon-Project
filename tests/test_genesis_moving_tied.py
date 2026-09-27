"""Rollback, physical-time and persistence contracts of moving tied mechanics."""
from copy import deepcopy
from dataclasses import fields, replace
import json

import numpy as np
import pytest
from scipy import sparse
from scipy.sparse.linalg import spsolve

from tectonics.genesis_contact import ContactParameters, SECONDS_PER_YEAR
from tectonics.genesis_material import face_deformation, polar_increment
from tectonics.genesis_moving_tied import (MovingTiedLoading, MovingTiedMechanics,
    MovingTiedRetry, _current_increment, _nodal_area)
from tectonics.genesis_shell import Membrane
from tectonics.mesh import build_icosphere


@pytest.fixture
def model():
    return MovingTiedMechanics(build_icosphere(1), 5e6)


def zero_force(mesh, radius, depth, membrane):
    return np.zeros(membrane.ndof)


def load(model, memory=None, *, dt=10., beta=1., external=zero_force):
    count = model.template_mesh.cell_count
    memory = np.zeros((count, 3)) if memory is None else memory
    return MovingTiedLoading(dt, model.template_mesh.areas_unit_sphere*model.initial_radius_m**2*1e4,
        np.full(count, 6e10), memory, np.full(count, beta), external)


def body_force(amplitude):
    def result(mesh, radius, depth, membrane):
        # A fixed ambient poloidal field, projected on actual tangent frames.
        direction = np.array([.3, -.2, 1.])
        traction = amplitude*(direction-mesh.vertices*(mesh.vertices@direction)[:, None])
        area = _nodal_area(mesh, radius)
        return np.r_[(np.einsum("vij,vi->vj", membrane.vertex_basis, traction)*area[:, None]).ravel(), 0.]
    return result


def exact(first, second):
    for field in fields(first):
        left, right = getattr(first, field.name), getattr(second, field.name)
        if isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        else:
            assert left == right, field.name


def test_unloaded_hold_preserves_geometry_and_has_no_work(model):
    initial = model.initial(np.zeros((model.template_mesh.cell_count, 3)))
    after = model.trial(initial, load(model))
    np.testing.assert_array_equal(initial.vertices, after.vertices)
    np.testing.assert_array_equal(after.elastic_strain, 0.)
    assert after.radius_m == initial.radius_m
    for name in ("external_work_j", "drag_work_j", "bulk_work_j", "bulk_loading_correction_j",
                 "mechanical_remainder_j"):
        assert getattr(after, name) == 0.
    assert after.elapsed_years == 10 and after.accepted_steps == 1
    assert model.force_diagnostics(after)["relative_residual"] == 0


@pytest.mark.parametrize("memory_value,beta", [(2e-4, 1.), (2e-4, .3), (-2e-4, .7)])
def test_uniform_thermal_memory_has_analytic_free_radius(model, memory_value, beta):
    memory = np.zeros((model.template_mesh.cell_count, 3))
    memory[:, :2] = memory_value
    initial = model.initial(np.zeros_like(memory))
    after = model.trial(initial, load(model, memory, beta=beta))
    assert after.radius_m == pytest.approx(initial.radius_m*np.exp(-memory_value/beta), rel=4e-14)
    assert np.max(np.abs(after.elastic_strain)) < 2e-14
    np.testing.assert_allclose(after.vertices, initial.vertices, atol=2e-14, rtol=0)
    assert after.equilibrium_residual <= model.parameters.equilibrium_tolerance
    initial_energy = .5*np.sum(np.einsum("fi,ij,fj->f", memory,
        Membrane(model.template_mesh, model.poisson_ratio).d, memory)*load(model).volume_m3*6e10)
    # Trapezoid work accounts explicitly for the supplied beta factor.
    assert after.bulk_work_j == pytest.approx(-initial_energy/beta, rel=2e-10)
    assert after.bulk_loading_correction_j == pytest.approx(initial_energy*(1-1/beta), abs=100., rel=2e-10)


def test_small_loaded_increment_matches_existing_linear_physical_time_system(model):
    initial = model.initial(np.zeros((model.template_mesh.cell_count, 3)))
    # Centimetre motion is still in the linear regime and is well resolved
    # against sub-nanometre unit-vector storage roundoff at planetary radius.
    loading = load(model, dt=20., external=body_force(5000.))
    membrane = Membrane(model.template_mesh, model.poisson_ratio)
    volume, young = loading.volume_m3, loading.young_modulus_pa
    local = membrane.ki*(young*volume/model.initial_radius_m**2)[:, None, None]
    stiffness = sparse.coo_matrix((local.ravel(), (membrane.rr, membrane.cc)),
        shape=(model.ndof, model.ndof)).tocsr()
    drag = np.r_[np.repeat(_nodal_area(model.template_mesh, model.initial_radius_m), 2), 0.]
    drag *= model.parameters.basal_drag_pa_s_m/(loading.dt_years*SECONDS_PER_YEAR)
    external = loading.external_force(model.template_mesh, model.initial_radius_m,
        np.full(model.template_mesh.cell_count, 1e4), membrane)
    expected = spsolve(stiffness+sparse.diags(drag), external)
    actual = model.trial(initial, loading)
    assert np.linalg.norm(actual.last_increment_current_m-expected)/np.linalg.norm(expected) < 5e-5
    assert actual.drag_work_j > 0
    audit = model.force_diagnostics(actual)
    assert np.linalg.norm(audit["residual_force_n"])/np.linalg.norm(actual.last_external_force_n) < 1e-7


def test_callback_sees_changed_geometry_and_current_material_depth(model):
    initial = model.initial(np.zeros((model.template_mesh.cell_count, 3)))
    memory = initial.elastic_strain.copy()
    memory[:, :2] = 2e-4
    calls = []
    volume = load(model).volume_m3
    def external(mesh, radius, depth, membrane):
        np.testing.assert_allclose(depth*mesh.areas_unit_sphere*radius**2, volume, rtol=3e-16)
        assert membrane.mesh is mesh
        calls.append(radius)
        return np.zeros(model.ndof)
    model.trial(initial, load(model, memory, external=external))
    assert min(calls) < initial.radius_m and len(calls) > 1


def test_trial_does_not_mutate_accepted_history_or_loading(model):
    initial = model.initial(np.zeros((model.template_mesh.cell_count, 3)))
    loading = load(model, external=body_force(5e4))
    saved_state, saved_loading = deepcopy(initial), deepcopy(loading)
    after = model.trial(initial, loading)
    exact(initial, saved_state)
    for name in ("volume_m3", "young_modulus_pa", "memory", "effective_b"):
        np.testing.assert_array_equal(getattr(loading, name), getattr(saved_loading, name))
    assert not np.shares_memory(after.vertices, initial.vertices)
    assert not np.shares_memory(after.last_volume_m3, loading.volume_m3)


def test_rejected_large_physical_step_does_not_leak_state(model):
    initial = model.initial(np.zeros((model.template_mesh.cell_count, 3)))
    saved = deepcopy(initial)
    memory = initial.elastic_strain.copy()
    memory[:, :2] = .008
    with pytest.raises(MovingTiedRetry, match="increment_limit"):
        model.trial(initial, load(model, memory))
    exact(initial, saved)


def test_memory_is_applied_once_despite_multiple_newton_evaluations(model):
    initial = model.initial(np.zeros((model.template_mesh.cell_count, 3)))
    memory = initial.elastic_strain.copy()
    memory[:, :2] = .001
    calls = []
    def external(mesh, radius, depth, membrane):
        calls.append(radius)
        return np.zeros(model.ndof)
    result = model.trial(initial, load(model, memory, beta=.4, external=external))
    assert len(calls) >= 3
    assert result.radius_m == pytest.approx(initial.radius_m*np.exp(-.001/.4), rel=2e-14)


def test_checkpoint_same_future_loading_is_bitwise_exact(model, tmp_path):
    initial = model.initial(np.zeros((model.template_mesh.cell_count, 3)))
    first = model.trial(initial, load(model, external=body_force(3e4)))
    path = tmp_path/"moving.npz"
    model.save_state(path, first)
    restored = model.load_state(path)
    exact(first, restored)
    direct = model.trial(first, load(model, first.elastic_strain*.99, dt=17., beta=.995,
                                   external=body_force(3e4)))
    resumed = model.trial(restored, load(model, restored.elastic_strain*.99, dt=17., beta=.995,
                                        external=body_force(3e4)))
    exact(direct, resumed)
    restored_model, restored_state = MovingTiedMechanics.from_checkpoint(path)
    assert restored_model.fingerprint == model.fingerprint
    exact(restored_state, first)
    reconstructed = restored_model.trial(restored_state,
        load(restored_model, restored_state.elastic_strain*.99, dt=17., beta=.995,
             external=body_force(3e4)))
    exact(direct, reconstructed)


@pytest.mark.parametrize("bad", ["extra", "missing", "vertices", "force", "drag", "clock", "count", "nan", "modulus"])
def test_checkpoint_rejects_incomplete_or_inconsistent_history(model, tmp_path, bad):
    initial = model.initial(np.zeros((model.template_mesh.cell_count, 3)))
    first = model.trial(initial, load(model, external=body_force(3e4)))
    path = tmp_path/"moving.npz"
    model.save_state(path, first)
    with np.load(path, allow_pickle=False) as source:
        data = {name: source[name].copy() for name in source.files}
    meta = json.loads(str(data["metadata"]))
    if bad == "extra":
        data["unexpected"] = np.array(1.)
    elif bad == "missing":
        data.pop("elastic_strain")
    elif bad == "vertices":
        data["vertices"][0] *= 1.001
    elif bad == "force":
        data["last_external_force_n"][0] += 1e13
    elif bad == "drag":
        data["last_drag_force_n"][0] += 1e13
    elif bad == "clock":
        meta["last_step_years"] = 11.
    elif bad == "count":
        meta["accepted_steps"] = 0
    elif bad == "nan":
        data["elastic_strain"][0, 0] = np.nan
    elif bad == "modulus":
        data["last_young_modulus_pa"][0] = 0.
    data["metadata"] = np.array(json.dumps(meta))
    np.savez_compressed(path, **data)
    with pytest.raises(ValueError):
        model.load_state(path)
    with pytest.raises(ValueError):
        MovingTiedMechanics.from_checkpoint(path)


def test_checkpoint_rejects_different_policy_or_geometry(model, tmp_path):
    initial = model.initial(np.zeros((model.template_mesh.cell_count, 3)))
    path = tmp_path/"moving.npz"
    model.save_state(path, initial)
    other = MovingTiedMechanics(model.template_mesh, model.initial_radius_m,
        parameters=replace(model.parameters, basal_drag_pa_s_m=2e14))
    with pytest.raises(ValueError, match="different geometry or parameters"):
        other.load_state(path)


@pytest.mark.parametrize("field,value", [("dt_years", 0.), ("dt_years", True),
    ("volume_m3", -1.), ("young_modulus_pa", 0.), ("effective_b", 1.1),
    ("effective_b", 0.), ("memory", np.nan), ("external_force", None)])
def test_invalid_loading_is_rejected(model, field, value):
    initial = model.initial(np.zeros((model.template_mesh.cell_count, 3)))
    loading = load(model)
    if isinstance(getattr(loading, field), np.ndarray):
        value = np.full_like(getattr(loading, field), value)
    with pytest.raises(ValueError):
        model.trial(initial, replace(loading, **{field: value}))


def test_rotation_direction_drag_uses_arrival_tangent_and_positive_work(model):
    from scipy.spatial.transform import Rotation
    from tectonics.genesis_material import rebuild_material_mesh
    old = model.template_mesh
    matrix = Rotation.from_rotvec([.002, -.001, .0015]).as_matrix()
    new = rebuild_material_mesh(old, old.vertices@matrix.T)
    membrane = Membrane(new, model.poisson_ratio)
    motion, angle = _current_increment(old, new, model.initial_radius_m, model.initial_radius_m, membrane)
    world = np.einsum("vij,vj->vi", membrane.vertex_basis,
        motion[:-1].reshape(-1, 2)/model.initial_radius_m)
    recovered = np.cos(angle)[:, None]*new.vertices-np.sinc(angle/np.pi)[:, None]*world
    np.testing.assert_allclose(recovered, old.vertices, atol=4e-16, rtol=0)
    assert np.all(np.einsum("vi,vi->v", world, new.vertices-old.vertices) >= 0)
    _, strain = polar_increment(face_deformation(old, new, 5000., 5000.))
    assert np.max(np.abs(strain)) < 2e-15


def test_many_small_steps_can_exceed_old_total_strain_limit(model):
    state = model.initial(np.zeros((model.template_mesh.cell_count, 3)))
    # Repeated uniform physical contractions have zero elastic stress but
    # accumulate beyond .005, which was the obsolete reference limit.
    for _ in range(7):
        memory = state.elastic_strain.copy()
        memory[:, :2] += .001
        state = model.trial(state, load(model, memory))
    assert state.radius_m/model.initial_radius_m == pytest.approx(np.exp(-.007), rel=2e-13)
    assert np.max(np.abs(state.elastic_strain)) < 3e-13
    assert state.accepted_steps == 7
