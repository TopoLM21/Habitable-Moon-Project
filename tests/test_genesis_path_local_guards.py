"""Integration controls for the optional fixed-reference local geometry policy.

The prescribed motions below probe numerical acceptance, not a finite-motion
constitutive solution or a physical crack-growth criterion.
"""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from tectonics.genesis_crack_path import CrackInterval
from tectonics.genesis_path_dynamics import PathMechanics, _PathRetry
from tectonics.genesis_path_geometry import PathGeometryParameters
from test_genesis_path_dynamics import _isothermal, _model, _same


@pytest.fixture(scope="module")
def legacy():
    return _model()


def _local(legacy, parameters=None):
    return PathMechanics(legacy.basis, legacy.depth_m,
        parameters=legacy.parameters, law_parameters=legacy.law_parameters,
        geometry_parameters=parameters or PathGeometryParameters())


def _initial(model):
    return model.initial(np.zeros((model.basis.topology.mesh.cell_count, 3)))


def _spin(model, angle):
    state = _initial(model)
    basis = model.basis
    world = basis.radius_m*np.cross([0., 0., angle], basis.subdivision.parent_mesh.vertices)
    state.displacement_m[:basis.nparent-1] = np.einsum("vij,vi->vj",
        basis.parent_membrane.vertex_basis, world).ravel()
    return state


def _loading_for_tied_motion(model, before, displacement, years=1.):
    """Manufactured smooth rotation force; all prospective bank jumps stay tied."""
    from tectonics.genesis_contact import SECONDS_PER_YEAR
    loading = _isothermal(model, before, years)
    matrix, force = model.basis.bulk(loading.volume_m3,
        loading.elasticity*loading.effective_b[:, None, None], loading.memory)
    drag = model.basis.drag_area_m2*model.parameters.basal_drag_pa_s_m/(years*SECONDS_PER_YEAR)
    external = force+matrix@displacement+drag*displacement
    return replace(loading, external_force=external)


def test_default_keeps_legacy_fingerprint_and_checkpoint_layout(legacy, tmp_path):
    # Fixed small discretization fingerprint from the pre-policy hash format:
    # adding an optional policy must not invalidate ordinary 0.1 snapshots.
    assert legacy.fingerprint == "f48e41262cc71af229e645914ca856f0b7ad4d234427528d6e3750c50026d37c"
    explicit_none = PathMechanics(legacy.basis, legacy.depth_m,
        geometry_parameters=None)
    assert explicit_none.fingerprint == legacy.fingerprint
    state = _initial(legacy)
    state = legacy.trial(state, _isothermal(legacy, state, 1.))
    path = tmp_path/"legacy-mechanics.npz"
    legacy.save_state(path, state)
    _same(state, explicit_none.load_state(path))
    assert "gradient_norm" not in legacy.geometry_metrics(state)


def test_local_policy_accepts_smooth_motion_rejected_by_global_edge_proxy(legacy):
    local = _local(legacy)
    before = _initial(local)
    untouched = deepcopy(before)
    desired = _spin(local, .005).displacement_m
    loading = _loading_for_tied_motion(local, before, desired)
    with pytest.raises(_PathRetry, match="^path_reference_motion_limit$"):
        legacy.trial(before, loading)
    after = local.trial(before, loading)
    _same(before, untouched)
    np.testing.assert_allclose(after.displacement_m, desired, rtol=3e-12, atol=3e-9)
    metrics = local.geometry_metrics(after)
    assert metrics["motion_edge_fraction"] > legacy.parameters.max_motion_edge_fraction
    assert metrics["gradient_norm"] < local.geometry_parameters.max_displacement_gradient
    assert metrics["linear_strain_error"] < local.geometry_parameters.max_linear_strain_error
    assert after.active_interval is None
    np.testing.assert_array_equal(after.cohorts.damage, 0.)
    np.testing.assert_array_equal(after.cohorts.traction_pa, 0.)
    assert after.elapsed_years == 1.


@pytest.mark.parametrize(("field", "bound", "reason"), [
    ("max_displacement_gradient", .004, "gradient"),
    ("max_material_rotation_rad", .004, "rotation"),
    ("max_tangent_motion_radius_fraction", .004, "tangent_motion"),
    ("max_linear_strain_error", 1e-6, "linearization"),
])
def test_local_policy_rejects_independently_resolved_kinematic_limits(legacy, field, bound, reason):
    parameters = replace(PathGeometryParameters(), **{field: bound})
    model = _local(legacy, parameters)
    state = _spin(model, .005)
    before = deepcopy(state)
    # All of these motions have almost zero compatible infinitesimal strain;
    # this cannot justify ignoring frame rotation or finite geometry effects.
    assert model.geometry_metrics(state)["geometric_added_strain"] < 1e-14
    with pytest.raises(_PathRetry, match="^path_local_"+reason+"_limit$"):
        model._check_geometry(state, state)
    _same(before, state)


def test_rotation_with_small_bank_slip_is_rejected_by_contact_frame_error(legacy):
    policy = replace(PathGeometryParameters(), max_relative_contact_jump_error=1e-4)
    model = _local(legacy, policy)
    state = _spin(model, .005)
    state.displacement_m[model.basis.nparent:] = np.random.default_rng(100).normal(
        size=model.basis.ndof-model.basis.nparent)*.1
    metrics = model.geometry_metrics(state)
    assert metrics["linear_strain_error"] < policy.max_linear_strain_error
    assert metrics["jump_edge_fraction"] < model.parameters.max_jump_edge_fraction
    assert metrics["penetration_m"] < model.parameters.max_penetration_m
    assert metrics["relative_contact_jump_error"] > policy.max_relative_contact_jump_error
    with pytest.raises(_PathRetry, match="^path_local_contact_frame_limit$"):
        model._check_geometry(state, state)


def test_local_policy_still_enforces_finite_strain_even_for_exact_radial_mode(legacy):
    model = _local(legacy)
    state = _initial(model)
    state.displacement_m[model.basis.nparent-1] = (
        model.parameters.max_added_strain-1e-6)*model.basis.radius_m
    metrics = model.geometry_metrics(state)
    assert metrics["constitutive_added_strain"] < model.parameters.max_added_strain
    assert metrics["finite_green_strain"] > model.parameters.max_added_strain
    with pytest.raises(_PathRetry, match="^path_local_finite_strain_limit$"):
        model._check_geometry(state, state)


@pytest.mark.parametrize("change", [None, PathGeometryParameters(max_material_rotation_rad=.01)])
def test_checkpoint_cannot_silently_change_geometry_policy(legacy, tmp_path, change):
    model = _local(legacy)
    state = _initial(model)
    state = model.trial(state, _isothermal(model, state, 2.))
    path = tmp_path/"local-mechanics.npz"
    model.save_state(path, state)
    _same(state, _local(legacy).load_state(path))
    other = PathMechanics(legacy.basis, legacy.depth_m, geometry_parameters=change)
    assert other.fingerprint != model.fingerprint
    with pytest.raises(ValueError, match="different geometry or parameters"):
        other.load_state(path)


def test_local_policy_cannot_release_or_advance_past_stopped_reference(legacy, tmp_path):
    model = _local(legacy)
    state = _initial(model)
    state = model.trial(state, _isothermal(model, state, 2.))
    stopped = replace(state, stopped_reason="path_local_linearization_limit")
    path = tmp_path/"stopped.npz"
    model.save_state(path, stopped)
    restored = model.load_state(path)
    with pytest.raises(ValueError, match="stopped"):
        model.release(restored, CrackInterval(0., model.basis.insertion.path.length_m))
    with pytest.raises(ValueError, match="stopped"):
        model.trial(restored, _isothermal(model, restored, 1.))
    after = model.advance(restored, 10., lambda *args: pytest.fail("Stopped state requested fresh loading"))
    _same(restored, after)
    np.testing.assert_array_equal(after.displacement_m, stopped.displacement_m)


def test_local_policy_does_not_relax_existing_reference_strain_guard(legacy):
    model = _local(legacy)
    state = _initial(model)
    state.displacement_m[model.basis.nparent-1] = 1.01*model.parameters.max_added_strain*model.basis.radius_m
    for selected in (legacy, model):
        with pytest.raises(_PathRetry, match="^path_reference_strain_limit$"):
            selected._check_geometry(state, state)


def test_invalid_opt_in_policy_is_rejected_before_solver_setup(legacy):
    with pytest.raises(ValueError, match="PathGeometryParameters"):
        PathMechanics(legacy.basis, legacy.depth_m, geometry_parameters={})
