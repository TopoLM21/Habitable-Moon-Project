"""Complete interface kernel: scaled numerical fixtures and literal Genesis geometry."""
from dataclasses import FrozenInstanceError, replace
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tectonics.genesis_volume_release_modes import (
    CompactReleaseModesRefusal, compact_release_modes_from_dict,
    compact_release_modes_to_dict, resolve_compact_release_modes,
    validate_compact_release_modes, validate_release_increment,
)

PIN = "a"*64


def _trace(matrix):
    matrix = np.asarray(matrix, float)
    return np.pad(matrix, ((0, (-len(matrix)) % 3), (0, 0))).reshape(-1, 3, 18)


def _resolve(matrix, **kwargs):
    trace = _trace(matrix)
    return resolve_compact_release_modes(trace, np.ones(len(trace)), PIN, **kwargs)


def _assert_complete(value):
    z, b = value.tied_null_modes, value.release_basis
    assert z.shape == (18, value.nullity)
    assert b.shape == (18, value.rank)
    full = np.column_stack((z, b))
    np.testing.assert_allclose(full.T@full, np.eye(18), rtol=0, atol=3e-15)
    np.testing.assert_allclose(full@full.T, np.eye(18), rtol=0, atol=3e-15)
    assert value.physical_release_admitted is value.funding_admitted is False


def test_full_rank_compact_trace_needs_no_tied_kernel_and_does_not_claim_release():
    value = _resolve(np.eye(18))
    assert value.rank == 18 and value.nullity == 0
    assert value.retained_rank_margin > 10
    assert value.rejected_rank_margin is None
    np.testing.assert_array_equal(value.normalized_singular_values, np.ones(18))
    validate_release_increment(value, np.arange(18.), np.ones(18))
    _assert_complete(value)


def test_complete_kernel_ties_unobserved_directions_beyond_three_supplied_witnesses():
    matrix = np.zeros((9, 18)); matrix[:, :9] = np.eye(9)
    witnesses = np.eye(18)[:, 9:12]
    value = _resolve(matrix, analytic_null_modes=witnesses)
    assert value.rank == value.nullity == 9
    _assert_complete(value)
    escaped = np.eye(18)[:, 17]
    assert np.linalg.norm(witnesses.T@escaped) == 0.
    with pytest.raises(CompactReleaseModesRefusal, match="complete jump-kernel"):
        validate_release_increment(value, escaped)
    validate_release_increment(value, value.release_basis@np.arange(9.))


def test_column_normalization_is_back_transformed_before_Euclidean_orthogonal_completion():
    matrix = np.zeros((9, 18)); matrix[:, :9] = np.eye(9)
    scales = np.logspace(-5, 5, 9)
    matrix[:, 9:] = np.diag(scales)
    value = _resolve(matrix)
    expected_null = np.vstack((-np.diag(scales), np.eye(9)))
    expected_null /= np.linalg.norm(expected_null, axis=0)
    np.testing.assert_allclose(value.tied_null_modes@value.tied_null_modes.T,
        expected_null@expected_null.T, rtol=0, atol=4e-15)
    np.testing.assert_allclose(matrix@value.tied_null_modes, 0., rtol=0, atol=2e-11)
    assert value.rank == 9
    _assert_complete(value)


@pytest.mark.parametrize("trace_scale,weight_scale", [(1e-140, 1e-100), (1e140, 1e100)])
def test_uniform_extreme_units_preserve_rank_and_kernel_without_norm_overflow(trace_scale, weight_scale):
    trace = _trace(np.vstack((np.eye(9), np.zeros((9, 9)))).T)
    reference = resolve_compact_release_modes(trace, np.ones(len(trace)), PIN)
    value = resolve_compact_release_modes(trace*trace_scale,
        np.ones(len(trace))*weight_scale, PIN)
    assert value.rank == reference.rank
    np.testing.assert_allclose(value.tied_null_modes@value.tied_null_modes.T,
        reference.tied_null_modes@reference.tied_null_modes.T, rtol=0, atol=2e-15)
    assert np.isfinite(value.column_scales_m).all()
    _assert_complete(value)


def test_zero_weight_rows_are_excluded_without_adding_a_positive_weight_floor():
    trace = _trace(np.eye(18))
    weights = np.ones(len(trace)); weights[0] = 0.
    trace[0] *= 1e300
    value = resolve_compact_release_modes(trace, weights, PIN)
    assert value.rank == 15 and value.nullity == 3
    np.testing.assert_array_equal(value.column_scales_m[:3], np.zeros(3))
    with pytest.raises(CompactReleaseModesRefusal, match="tied complete"):
        validate_release_increment(value, np.eye(18)[:, 0])
    _assert_complete(value)


def test_zero_trace_ties_all_modes_without_inventing_a_rank_scale():
    value = resolve_compact_release_modes(np.zeros((2, 3, 18)), [0., 2.], PIN)
    assert value.rank == 0 and value.nullity == 18 and value.rank_threshold == 0.
    np.testing.assert_array_equal(value.column_scales_m, np.zeros(18))
    validate_release_increment(value, np.zeros(18), np.zeros(18))
    with pytest.raises(CompactReleaseModesRefusal, match="tied complete"):
        validate_release_increment(value, np.eye(18)[:, 0])
    _assert_complete(value)


def test_fewer_rows_than_coefficients_still_resolves_all_right_kernel_directions():
    matrix = np.zeros((3, 18)); matrix[:, :3] = np.eye(3)
    matrix[:, 3:6] = np.diag((2., 3., 4.))
    value = _resolve(matrix)
    assert value.rank == 3 and value.nullity == 15
    _assert_complete(value)


@pytest.mark.parametrize("delta", [1e-10, 5e-11, 5e-10])
def test_near_threshold_rank_is_refused_instead_of_silently_releasing_a_mode(delta):
    matrix = np.eye(18); matrix[:, 1] = matrix[:, 0]; matrix[1, 1] = delta
    with pytest.raises(CompactReleaseModesRefusal, match="rank is ambiguous"):
        _resolve(matrix)


def test_sub_threshold_numerical_kernel_keeps_measured_residual_and_margin():
    matrix = np.eye(18); matrix[:, 1] = matrix[:, 0]; matrix[1, 1] = 1e-13
    value = _resolve(matrix)
    assert value.rank == 17 and value.nullity == 1
    assert 0. < value.rejected_rank_margin < .1
    assert 0. < value.weighted_kernel_relative_residual < 1e-11
    with pytest.raises(CompactReleaseModesRefusal, match="tied complete"):
        validate_release_increment(value, value.tied_null_modes[:, 0])


def test_velocity_cannot_escape_tied_kernel_when_displacement_increment_is_valid():
    matrix = np.zeros((3, 18)); matrix[:, :3] = np.eye(3)
    value = _resolve(matrix)
    before = np.zeros(18)
    with pytest.raises(CompactReleaseModesRefusal, match="Release velocity"):
        validate_release_increment(value, before, np.eye(18)[:, 17])
    np.testing.assert_array_equal(before, np.zeros(18))


@pytest.mark.parametrize("scale", [1e-120, 1e120])
def test_vector_tied_component_test_scales_with_vector_norm_without_an_absolute_floor(scale):
    matrix = np.zeros((3, 18)); matrix[:, :3] = np.eye(3)
    value = _resolve(matrix)
    vector = value.release_basis[:, 0]*scale
    validate_release_increment(value, vector)
    vector += value.tied_null_modes[:, 0]*(1e-9*scale)
    with pytest.raises(CompactReleaseModesRefusal, match="tied complete"):
        validate_release_increment(value, vector)


@pytest.mark.parametrize("mutation", ["bool_trace", "mixed_bool_trace", "nan_trace", "bad_trace_shape",
    "bool_weights", "mixed_bool_weights", "negative_weights", "zero_weights", "inf_weights",
    "bad_weights_shape", "bad_source", "bool_tolerance", "zero_tolerance", "sub_arithmetic_tolerance", "bad_separation"])
def test_malformed_source_inputs_are_refused(mutation):
    trace = _trace(np.eye(18)); weights = np.ones(len(trace)); pin = PIN; kwargs = {}
    if mutation == "bool_trace": trace = trace.astype(bool)
    if mutation == "mixed_bool_trace": trace = trace.tolist(); trace[0][0][0] = True
    if mutation == "nan_trace": trace[0, 0, 0] = np.nan
    if mutation == "bad_trace_shape": trace = trace[:, :, :17]
    if mutation == "bool_weights": weights = weights.astype(bool)
    if mutation == "mixed_bool_weights": weights = weights.tolist(); weights[0] = True
    if mutation == "negative_weights": weights[0] = -1
    if mutation == "zero_weights": weights[:] = 0
    if mutation == "inf_weights": weights[0] = np.inf
    if mutation == "bad_weights_shape": weights = weights[:, None]
    if mutation == "bad_source": pin = "A"*64
    if mutation == "bool_tolerance": kwargs["relative_rank_tolerance"] = True
    if mutation == "zero_tolerance": kwargs["relative_rank_tolerance"] = 0.
    if mutation == "sub_arithmetic_tolerance": kwargs["relative_rank_tolerance"] = 1e-20
    if mutation == "bad_separation": kwargs["rank_separation_factor"] = 1.
    with pytest.raises(CompactReleaseModesRefusal):
        resolve_compact_release_modes(trace, weights, pin, **kwargs)


@pytest.mark.parametrize("witnesses", [np.zeros((18, 1)), np.eye(18)[:, :1],
    np.column_stack((np.eye(18)[:, 17], np.eye(18)[:, 17]))])
def test_analytic_witnesses_must_be_independent_and_lie_in_complete_kernel(witnesses):
    matrix = np.zeros((3, 18)); matrix[:, :3] = np.eye(3)
    with pytest.raises(CompactReleaseModesRefusal, match="Analytic null"):
        _resolve(matrix, analytic_null_modes=witnesses)


def test_increment_rejects_boolean_or_nonfinite_vectors_and_missing_required_increment():
    value = _resolve(np.eye(18))
    for vector in (None, np.ones(18, dtype=bool), [True]+[0.]*17, [np.inf]+[0.]*17, np.zeros(17)):
        with pytest.raises(CompactReleaseModesRefusal):
            validate_release_increment(value, vector)


def test_arrays_and_descriptor_are_immutable_and_replay_requires_literal_trusted_source():
    value = _resolve(np.eye(18))
    with pytest.raises(FrozenInstanceError): value.rank = 17
    for array in (value.trace_xyz, value.reference_area_weights_m2,
            value.tied_null_modes, value.release_basis, value.normalized_singular_values):
        with pytest.raises(ValueError): array.setflags(write=True)
    raw = compact_release_modes_to_dict(value)
    restored = compact_release_modes_from_dict(raw, trace_xyz=value.trace_xyz,
        reference_area_weights_m2=value.reference_area_weights_m2, source_binding_sha256=PIN)
    assert restored.binding_sha256 == value.binding_sha256
    with pytest.raises(CompactReleaseModesRefusal, match="trusted source replay"):
        compact_release_modes_from_dict(raw, trace_xyz=value.trace_xyz,
            reference_area_weights_m2=value.reference_area_weights_m2, source_binding_sha256="b"*64)
    forged = copy.deepcopy(raw); forged["physical_release_admitted"] = True
    with pytest.raises(CompactReleaseModesRefusal, match="trusted source replay"):
        compact_release_modes_from_dict(forged, trace_xyz=value.trace_xyz,
            reference_area_weights_m2=value.reference_area_weights_m2, source_binding_sha256=PIN)
    with pytest.raises(CompactReleaseModesRefusal, match="immutable binding"):
        replace(value, rank=17)
    validate_compact_release_modes(value)


SOURCE_PINS = {
    0: "1078ea2101b8b2e31b0308d19eaafcc5bf28afb475fc8448ba64a3b20b992b57",
    40: "2debb0ebdb4b5c1c86bebb0a14118c8648c4a4fe61ec4042ca6fd23cbbedb0bf",
    79: "3cc2cbb4ecbc25de5d8d7eafa039db80359b8746132ac166222d8198f5778d18",
}


@pytest.mark.parametrize("face", [0, 40, 79])
def test_literal_genuine_Genesis_material_sections_recover_rank15_and_all_three_kernel_modes(face):
    # Read a 58KB pinned ORIGINAL section, not the large whole-seed fixture;
    # reconstruct only geometry/quadrature.  No runtime restoration occurs.
    from tectonics.genesis_volume_geometry import ConicalCohortGeometry
    from tectonics.genesis_volume_material_sections import prepare_cohort_material_section
    from tectonics.genesis_volume_finite_seed import prepare_cohort_finite_seed
    root = Path(__file__).resolve().parents[1]
    path = root/f"analysis/genesis_volume_location_validation/evidence/actual_control/supplied_current_plane_face_{face}.json"
    original = path.read_bytes()
    assert hashlib.sha256(original).hexdigest() == SOURCE_PINS[face]
    packet = json.loads(original)
    sections = packet["sections"]
    old = min(sections["cohorts"], key=lambda entry: entry["cohort_id"])
    lower, upper = old["lambda_bounds"]
    cohort = ConicalCohortGeometry(old["cohort_id"], face, 0,
        *old["q_bounds"], old["mass_kg"], old["volume_m3"],
        tuple(map(tuple, old["triangle_m"])), upper, lower, (), old["owner_id"], old["material_id"])
    pose = np.column_stack((old["displacement_A_m"], old["slope_P_m"]))
    X = np.asarray(old["triangle_m"])
    normal = np.asarray(old["normal_xyz"])
    N, lam = np.full(3, 1/3.), .5*(lower+upper)
    center = N@(lam*(X+pose[:, 3:])+pose[:, :3]-pose[:, 3:])
    radius = .12*(upper-lower)*np.linalg.norm(N@(X+pose[:, 3:]))
    section = prepare_cohort_material_section(cohort, pose, normal, float(normal@center),
        conical_mass_coefficient=old["conical_mass_coefficient"], cohort_index=old["cohort_index"])
    seed = prepare_cohort_finite_seed(section, center, radius, quadrature_order=8)
    surface = seed.surface
    t = (surface.radial_scale-lower)/(upper-lower)
    r2 = np.sum((surface.current_position_m-center)**2, axis=1)
    bubble = np.maximum(1-r2/radius**2, 0.)**2*(27*np.prod(surface.barycentric, axis=1))*(4*t*(1-t))
    trace = bubble[:, None, None]*surface.parent_trace
    analytic = np.zeros((18, 3))
    scalar_A = (X+pose[:, :3])@normal-normal@center
    scalar_P = (X+pose[:, 3:])@normal
    for vertex in range(3):
        analytic[6*vertex+np.arange(3), np.arange(3)] = scalar_A[vertex]
        analytic[6*vertex+3+np.arange(3), np.arange(3)] = scalar_P[vertex]
    analytic /= np.linalg.norm(analytic, axis=0)
    value = resolve_compact_release_modes(trace, surface.reference_area_weights_m2,
        SOURCE_PINS[face], analytic_null_modes=analytic)
    assert value.rank == 15 and value.nullity == 3
    _assert_complete(value)
    assert value.analytic_null_relative_residual < 1e-11
    assert value.analytic_null_subspace_error <= value.analytic_null_subspace_error_bound
    assert value.minimum_retained_weighted_singular_value_m > 0.
    assert value.retained_weighted_condition_number > 1e12
    # Do not replace this conditioning-derived diagnostic by a loose angle
    # tolerance.  Every known continuous source refinement is exactly tied.
    np.testing.assert_allclose(value.release_basis.T@analytic, 0., rtol=0, atol=2e-15)
    np.testing.assert_allclose(value.tied_null_modes@value.tied_null_modes.T,
        analytic@analytic.T, rtol=0, atol=2e-15)
    vector = value.release_basis@np.sin(np.arange(15)+.4)*1e-5
    validate_release_increment(value, vector)
    with pytest.raises(CompactReleaseModesRefusal, match="tied complete"):
        validate_release_increment(value, vector+1e-8*value.tied_null_modes[:, 0])
    assert path.read_bytes() == original
