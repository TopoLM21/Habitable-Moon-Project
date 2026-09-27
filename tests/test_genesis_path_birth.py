"""Independent reaction, work, covariance and strength controls for birth diagnostics."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from scipy.optimize import linprog
from scipy import sparse

from tectonics.genesis_contact_law import ContactLawParameters
from tectonics.genesis_crack_path import CrackInterval, ReferenceCrackPath
from tectonics.genesis_material import rebuild_material_mesh
from tectonics.genesis_path_basis import EmbeddedPathBasis
from tectonics.genesis_path_birth import recover_tied_tractions, classify_tied_onset, tied_reaction_capacity
from tectonics.genesis_path_dynamics import PathMechanics
from test_genesis_path_dynamics import _model, _isothermal, _same


def _accepted(model):
    initial = model.initial(np.zeros((model.basis.subdivision.mesh.cell_count, 3)))
    return model.trial(initial, _isothermal(model, initial, 1.))


def _reaction(model, state, traction):
    reaction = np.asarray(model.jump_operator.T@(
        np.repeat(model.geometry.interface_area_m2, 2)*traction.ravel())).ravel()
    return replace(state, constraint_reaction_n=reaction)


def _stress(model):
    return np.random.default_rng(201).normal(size=(model.basis.subdivision.mesh.cell_count, 3))*1e5


def test_recovered_traction_balances_reaction_and_arbitrary_virtual_work():
    model = _model()
    state = _accepted(model)
    rng = np.random.default_rng(37)
    source_traction = rng.normal(size=(len(model.trace_depth_m), 2))*1e6
    state = _reaction(model, state, source_traction)
    before, stress = deepcopy(state), _stress(model)
    result = recover_tied_tractions(model, state, stress)
    np.testing.assert_allclose(result.recovered_reaction_n, result.target_reaction_n, rtol=5e-15, atol=1.)
    virtual = rng.normal(size=model.basis.ndof)
    interface_work = np.sum(model.geometry.interface_area_m2[:, None]*result.traction_pa
                            *(model.jump_operator@virtual).reshape(-1, 2))
    assert interface_work == pytest.approx(state.constraint_reaction_n@virtual, rel=5e-15)
    assert result.relative_residual < 1e-14
    assert result.reaction_rank == model.basis.ndof-model.basis.nparent
    assert result.traction_nullity > 0
    assert np.count_nonzero(~result.trace_observed) == 2
    _same(before, state)
    for array in (result.traction_pa, result.stress_prior_pa, result.correction_pa,
                  result.trace_observed, result.target_reaction_n, result.recovered_reaction_n):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.setflags(write=True)


def test_balanced_prior_is_unchanged_and_unobservable_tips_keep_prior():
    model = _model()
    state, stress = _accepted(model), _stress(model)
    first = recover_tied_tractions(model, state, stress)
    balanced = _reaction(model, state, first.stress_prior_pa)
    result = recover_tied_tractions(model, balanced, stress)
    np.testing.assert_allclose(result.correction_pa, 0., rtol=0, atol=2e-10)
    np.testing.assert_array_equal(result.traction_pa[~result.trace_observed],
                                  result.stress_prior_pa[~result.trace_observed])


def test_reconstruction_is_minimum_area_weighted_change_not_unique_traction():
    model = _model()
    state = _accepted(model)
    result = recover_tied_tractions(model, state, _stress(model))
    j = model.jump_operator[:, model.basis.nparent:].toarray()
    area = np.repeat(model.geometry.interface_area_m2, 2)
    rng = np.random.default_rng(985)
    value = rng.normal(size=j.shape[0])*1e5
    null = value-j@np.linalg.solve((j.T*area)@j, (j.T*area)@value)
    np.testing.assert_allclose((j.T*area)@null, 0., atol=2.)
    correction = result.correction_pa.ravel()
    baseline = np.sum(area*correction**2)
    assert np.sum(area*(correction+null)**2) > baseline
    assert abs(np.sum(area*correction*null)) < 1e-14*max(baseline, 1.)


def test_world_rotation_changes_neither_local_traction_nor_strength():
    model = _model()
    b = model.basis
    rotation = Rotation.from_rotvec([.7, -.8, .3]).as_matrix()
    parent = rebuild_material_mesh(b.subdivision.parent_mesh,
        b.subdivision.parent_mesh.vertices@rotation.T)
    child = rebuild_material_mesh(b.subdivision.mesh, b.subdivision.mesh.vertices@rotation.T)
    path = ReferenceCrackPath(b.insertion.path.points_xyz@rotation.T, b.radius_m/1000.)
    arclength = b.insertion.path_arclength_m.copy()
    arclength[-1] = path.length_m
    insertion = replace(b.insertion, mesh=child, path=path, path_arclength_m=arclength)
    rotated = PathMechanics(EmbeddedPathBasis(parent, insertion, b.radius_m, .25), model.depth_m)
    traction = np.random.default_rng(65).normal(size=(len(model.trace_depth_m), 2))*1e6
    stress = _stress(model)
    left = recover_tied_tractions(model, _reaction(model, _accepted(model), traction), stress)
    right = recover_tied_tractions(rotated, _reaction(rotated, _accepted(rotated), traction), stress)
    np.testing.assert_allclose(left.stress_prior_pa, right.stress_prior_pa, rtol=1e-12, atol=1e-8)
    np.testing.assert_allclose(left.traction_pa, right.traction_pa, rtol=1e-12, atol=1e-7)


@pytest.mark.parametrize("factor,status", [(0., "below_strength"), (.5, "below_strength"),
    (1., "on_strength_surface"), (1.5, "strength_exceeded")])
def test_tensile_strength_crossing_diagnostic_never_authorizes_birth(factor, status):
    model = _model()
    result = recover_tied_tractions(model, _accepted(model), _stress(model))
    traction = np.zeros_like(result.traction_pa)
    traction[:, 0] = factor*model.law_parameters.tensile_strength_pa
    # Unobservable tips cannot decide the observed event.
    traction[~result.trace_observed, 0] = 100*model.law_parameters.tensile_strength_pa
    result = replace(result, traction_pa=traction)
    diagnosed = classify_tied_onset(result, np.zeros(len(traction)), model.law_parameters)
    assert diagnosed.status == status
    assert diagnosed.maximum_observed_ratio == pytest.approx(factor)
    assert not diagnosed.physical_birth_ready
    assert not diagnosed.tensile_exceeded[~result.trace_observed].any()
    assert not diagnosed.normal_ratio.flags.writeable


def test_wet_coulomb_strength_includes_compression_and_does_not_change_tension():
    model = _model()
    result = recover_tied_tractions(model, _accepted(model), _stress(model))
    traction = np.tile([-3e6, 1.7e6], (len(result.traction_pa), 1))
    result = replace(result, traction_pa=traction)
    dry = classify_tied_onset(result, np.zeros(len(traction)))
    wet = classify_tied_onset(result, np.ones(len(traction)))
    np.testing.assert_allclose(dry.shear_strength_pa, 3.8e6)
    np.testing.assert_allclose(wet.shear_strength_pa, 1.6e6)
    assert dry.status == "below_strength"
    assert wet.status == "strength_exceeded"
    np.testing.assert_array_equal(wet.normal_ratio, 0.)


def test_zero_shear_strength_reports_zero_or_infinite_utilization():
    model = _model()
    result = recover_tied_tractions(model, _accepted(model), np.zeros_like(_stress(model)))
    law = ContactLawParameters(cohesion_pa=0, friction_dry=0, friction_wet=0)
    zero = classify_tied_onset(result, np.zeros(len(result.traction_pa)), law)
    assert zero.maximum_observed_ratio == 0.
    loaded = replace(result, traction_pa=np.tile([0., 1.], (len(result.traction_pa), 1)))
    assert classify_tied_onset(loaded, np.zeros(len(result.traction_pa)), law).maximum_observed_ratio == np.inf


def test_rejects_unaccepted_released_stopped_or_rank_deficient_state():
    model = _model()
    stress = _stress(model)
    with pytest.raises(ValueError, match="accepted reaction"):
        recover_tied_tractions(model, model.initial(np.zeros_like(stress)), stress)
    accepted = _accepted(model)
    released = model.release(accepted, CrackInterval(0., model.basis.insertion.path.length_m))
    with pytest.raises(ValueError, match="all support banks tied"):
        recover_tied_tractions(model, released, stress)
    with pytest.raises(ValueError, match="unstopped"):
        recover_tied_tractions(model, replace(accepted, stopped_reason="test"), stress)
    model.jump_operator = model.jump_operator.tolil()
    model.jump_operator[:, model.basis.nparent:] = 0.
    model.jump_operator = model.jump_operator.tocsr()
    with pytest.raises(ValueError, match="rank deficient"):
        recover_tied_tractions(model, accepted, stress)


@pytest.mark.parametrize("value", [np.nan, np.inf, 1j, True])
def test_rejects_invalid_stress(value):
    model = _model()
    stress = np.full(_stress(model).shape, value)
    with pytest.raises(ValueError, match="stress_pa"):
        recover_tied_tractions(model, _accepted(model), stress)


@pytest.mark.parametrize("value", [-.1, 1.1, np.nan, 1j, True])
def test_rejects_invalid_water(value):
    model = _model()
    recovery = recover_tied_tractions(model, _accepted(model), _stress(model))
    with pytest.raises(ValueError, match="water_per_trace"):
        classify_tied_onset(recovery, np.full(len(recovery.traction_pa), value))


@pytest.mark.parametrize("value", [-.1, 1., np.inf, True])
def test_rejects_invalid_onset_tolerance(value):
    model = _model()
    recovery = recover_tied_tractions(model, _accepted(model), _stress(model))
    with pytest.raises(ValueError, match="relative_tolerance"):
        classify_tied_onset(recovery, np.zeros(len(recovery.traction_pa)), relative_tolerance=value)


def _capacity_control(traction=None):
    """Two traces carrying one vertex, with an exact Cartesian jump map."""
    model = _model()
    count = len(model.trace_depth_m)
    relative = np.zeros((2*count, model.basis.ndof-model.basis.nparent))
    assert relative.shape[1] == 2
    relative[:2] = np.eye(2)
    relative[2:4] = np.eye(2)
    model.jump_operator = sparse.hstack((sparse.csr_matrix((2*count, model.basis.nparent)),
        sparse.csr_matrix(relative)), format="csr")
    state = _accepted(model)
    if traction is not None:
        state = _reaction(model, state, np.broadcast_to(traction, (count, 2)))
    return model, state


def test_capacity_certificate_excludes_every_traction_nullspace_choice():
    model, state = _capacity_control([6e6, .3e6])
    water = np.zeros(len(model.trace_depth_m))
    certificate = tied_reaction_capacity(model, state, water)
    assert certificate.finite_capacity.all()
    assert certificate.violations.all()
    assert certificate.ratio[0] > 2.
    # Different stress priors change the recovered endpoint traction field,
    # but leave the necessary capacity certificate exactly unchanged.
    first = recover_tied_tractions(model, state, _stress(model))
    second = recover_tied_tractions(model, state, -3*_stress(model))
    assert np.linalg.norm(first.traction_pa-second.traction_pa) > 0
    again = tied_reaction_capacity(model, state, water)
    for name in certificate.__dataclass_fields__:
        np.testing.assert_array_equal(getattr(certificate, name), getattr(again, name))
        assert not getattr(certificate, name).flags.writeable


def test_finite_capacity_matches_independent_piecewise_linear_programs_with_mixed_water():
    model = _model()
    state = _accepted(model)
    water = np.linspace(.1, .9, len(model.trace_depth_m))
    traction = np.tile([6e6, .1e6], (len(water), 1))
    state = _reaction(model, state, traction)
    certificate = tied_reaction_capacity(model, state, water)
    assert certificate.finite_capacity.all()
    jump = model.jump_operator[:, model.basis.nparent:].toarray()
    law = model.law_parameters
    for vertex, direction in enumerate(certificate.direction):
        capacity = 0.
        for trace, area in enumerate(model.geometry.interface_area_m2):
            coefficients = jump[2*trace:2*trace+2, 2*vertex:2*vertex+2]@direction
            if not np.any(coefficients):
                continue
            mu = law.friction_dry+(law.friction_wet-law.friction_dry)*water[trace]
            cohesion = law.cohesion_pa*(1-(1-law.wet_cohesion_fraction)*water[trace])
            tensile = linprog(-coefficients, A_ub=[[0., 1.], [0., -1.]],
                b_ub=[cohesion, cohesion], bounds=[(0., law.tensile_strength_pa), (None, None)], method="highs")
            compressive = linprog(-coefficients, A_ub=[[mu, 1.], [mu, -1.]],
                b_ub=[cohesion, cohesion], bounds=[(None, 0.), (None, None)], method="highs")
            assert tensile.success and compressive.success
            capacity += area*max(-tensile.fun, -compressive.fun)
        assert certificate.capacity_n[vertex] == pytest.approx(capacity, rel=2e-14)


def test_compression_gives_unbounded_capacity_and_no_false_certificate():
    model, state = _capacity_control([-1e6, 0.])
    certificate = tied_reaction_capacity(model, state, np.ones(len(model.trace_depth_m)))
    assert certificate.unbounded_capacity.all()
    assert not certificate.finite_capacity.any()
    assert not certificate.violations.any()
    np.testing.assert_array_equal(certificate.capacity_n, np.inf)
    np.testing.assert_array_equal(certificate.ratio, 0.)


def test_zero_reaction_and_zero_shear_capacity_are_handled_explicitly():
    model, zero = _capacity_control()
    water = np.zeros(len(model.trace_depth_m))
    idle = tied_reaction_capacity(model, zero, water)
    np.testing.assert_array_equal(idle.required_n, 0.)
    np.testing.assert_array_equal(idle.capacity_n, 0.)
    np.testing.assert_array_equal(idle.ratio, 0.)
    assert idle.finite_capacity.all() and not idle.violations.any()
    model, shear = _capacity_control([0., 1.])
    law = ContactLawParameters(cohesion_pa=0., friction_dry=0., friction_wet=0.)
    result = tied_reaction_capacity(model, shear, water, law)
    np.testing.assert_array_equal(result.capacity_n, 0.)
    np.testing.assert_array_equal(result.ratio, np.inf)
    assert result.finite_capacity.all() and result.violations.all()


def test_near_zero_compressive_support_margin_is_uncertified_not_clamped():
    # d_n - mu*|d_s| is a tiny negative number, so finite support cannot
    # conservatively be claimed by rounding the coefficient to zero.
    model, state = _capacity_control([.6-1e-14, 1.])
    result = tied_reaction_capacity(model, state, np.zeros(len(model.trace_depth_m)))
    assert result.indeterminate_capacity.all()
    assert not result.finite_capacity.any()
    assert not result.violations.any()


def test_capacity_ratio_below_one_is_only_a_necessary_test():
    model, state = _capacity_control([.2e6, 0.])
    result = tied_reaction_capacity(model, state, np.zeros(len(model.trace_depth_m)))
    assert np.all(result.ratio < 1.)
    assert not result.violations.any()
    assert not hasattr(result, "physical_birth_ready")


@pytest.mark.parametrize("water", [-.1, 1.1, np.nan, True])
def test_capacity_rejects_invalid_water(water):
    model, state = _capacity_control()
    with pytest.raises(ValueError, match="water_per_trace"):
        tied_reaction_capacity(model, state, np.full(len(model.trace_depth_m), water))


@pytest.mark.parametrize("name,value", [("coefficient_tolerance", -1.),
    ("coefficient_tolerance", True), ("relative_tolerance", np.inf)])
def test_capacity_rejects_invalid_tolerance(name, value):
    model, state = _capacity_control()
    with pytest.raises(ValueError, match=name):
        tied_reaction_capacity(model, state, np.zeros(len(model.trace_depth_m)), **{name: value})
