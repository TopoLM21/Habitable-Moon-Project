"""Independent release mathematics and deliberately resealed forgeries."""
import ast
from pathlib import Path

import numpy as np
import pytest

from analysis.genesis_volume_release_validation import independent_audit as oracle


def plain(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {k: plain(v) for k, v in value.items()}
    return value


def reseal(raw, key):
    body = {k: v for k, v in raw.items() if k != key}
    return {**body, key: oracle.digest(body)}


def mode_packet(trace, weights):
    result = oracle.independent_jump_modes(trace, weights)
    body = dict(trace_xyz=plain(trace), reference_area_weights_m2=plain(weights),
        relative_rank_tolerance=1e-10, rank_separation_factor=10.,
        rank=result["rank"], nullity=result["nullity"],
        column_scales_m=plain(result["column_scales_m"]),
        normalized_singular_values=plain(result["normalized_singular_values"]),
        tied_null_modes=plain(result["tied_null_modes"]), release_basis=plain(result["release_basis"]),
        complete_kernel_remains_tied=True, physical_release_admitted=False, funding_admitted=False)
    return reseal(body, "release_modes_sha256")


@pytest.fixture
def trace():
    matrix = np.zeros((18, 18))
    matrix[:15, :15] = np.diag(np.geomspace(1, 1e-7, 15))
    return matrix.reshape(6, 3, 18)


@pytest.fixture
def law():
    return dict(fracture_energy_j_m2=1e6, peak_traction_pa=4e6,
                cohesive_stiffness_pa_m=1e9, compression_stiffness_pa_m=2e9)


def cohesive_example(law, initial, old, new):
    trace = np.zeros((1, 3, 18)); trace[0, :, :3] = np.eye(3)
    old_a, new_a = np.zeros(18), np.zeros(18)
    old_a[:3], new_a[:3] = old, new
    initial = np.asarray([initial], float)
    seed = oracle.insertion.prestress_seed(law, initial)
    h = seed["reference_offset_m"][0]+old_a[:3]
    maximum = np.asarray([max(seed["initial_maximum_separation_m"][0],
                              np.linalg.norm(np.r_[max(h[0], 0), h[1:]]))])
    result = oracle.independent_cohesive_path(law, initial, np.asarray([127.]),
        trace, np.eye(3)[None], old_a, new_a, maximum)
    body = plain({k: v for k, v in result.items() if k not in (
        "force_absolute_integral_n", "work_integral_roundoff_bound_j", "integrated_absolute_work_j")})
    body.update(kernel_only=True, physical_release_admitted=False, normal_contact_admitted=False,
                funding_admitted=False, numerically_integrated_work_j=result["mechanical_work_j"])
    return result, reseal(body, "cohesive_response_sha256")


def test_complete_kernel_retains_all_three_bulk_only_modes(trace):
    result = oracle.audit_jump_modes(mode_packet(trace, np.ones(6)), trace, np.ones(6))
    assert result["rank"] == 15 and result["nullity"] == 3
    np.testing.assert_allclose(result["null_projector"], np.diag([0]*15+[1]*3))


def test_kernel_signs_and_degenerate_rotations_do_not_change_admission(trace):
    packet = mode_packet(trace, np.ones(6))
    packet["tied_null_modes"] = (np.asarray(packet["tied_null_modes"])@np.asarray([[0, 1, 0], [-1, 0, 0], [0, 0, -1]])).tolist()
    packet["release_basis"] = (-np.asarray(packet["release_basis"])).tolist()
    oracle.audit_jump_modes(reseal(packet, "release_modes_sha256"), trace, np.ones(6))


def test_resealed_kernel_release_is_refused_even_at_zero_increment(trace):
    packet = mode_packet(trace, np.ones(6))
    packet["rank"] = 18; packet["nullity"] = 0
    packet["tied_null_modes"] = [[] for _ in range(18)]
    packet["release_basis"] = np.eye(18).tolist()
    with pytest.raises(ValueError, match="kernel dimension"):
        oracle.audit_jump_modes(reseal(packet, "release_modes_sha256"), trace, np.ones(6))


@pytest.mark.parametrize("name", ["increment", "velocity", "memory"])
def test_tied_kernel_component_is_refused(trace, name):
    modes = oracle.audit_jump_modes(mode_packet(trace, np.ones(6)), trace, np.ones(6))
    value = np.zeros(18); value[-1] = 1e-19
    with pytest.raises(ValueError, match="complete jump-kernel"):
        oracle.require_tied_kernel(modes, value, name)


def test_zero_jump_trace_keeps_every_direction_tied():
    result = oracle.independent_jump_modes(np.zeros((1, 3, 18)), np.ones(1))
    assert result["rank"] == 0 and result["nullity"] == 18
    np.testing.assert_array_equal(result["null_projector"], np.eye(18))


def test_rank_ambiguity_is_refused_without_floor():
    matrix = np.eye(18)
    matrix[:, 1] = matrix[:, 0]+2e-10*matrix[:, 1]
    with pytest.raises(ValueError, match="unresolved singular"):
        oracle.independent_jump_modes(matrix.reshape(6, 3, 18), np.ones(6))


@pytest.mark.parametrize("initial,old,new", [
    ([1e6, 1.2e6, -.8e6], [0, 0, 0], [.03, .05, -.04]),
    ([-8e6, 1e6, -2e6], [0, 0, 0], [.03, .05, -.04]),
    ([1e6, 1.2e6, -.8e6], [.1, .1, 0], [-.02, -.03, .02]),
    ([0, 0, 0], [0, 0, 0], [.8, .4, .3]),
])
def test_direct_vector_path_includes_softening_unloading_and_compression(law, initial, old, new):
    result, packet = cohesive_example(law, initial, old, new)
    oracle.audit_cohesive_path(packet, result)
    assert result["fracture_dissipation_increment_j"] >= 0
    assert result["force_work_j"] == pytest.approx(result["mechanical_work_j"], rel=5e-10, abs=1e-6)


def test_complete_failure_costs_gc_reference_area_once(law):
    result, _ = cohesive_example(law, [0, 0, 0], [0, 0, 0], [.8, .4, .3])
    assert result["fracture_dissipation_j"] == pytest.approx(127*law["fracture_energy_j_m2"])
    assert result["stored_energy_j"] == 0


def test_resealed_endpoint_traction_cannot_replace_path_mean(law):
    result, packet = cohesive_example(law, [1e6, 0, 0], [0, 0, 0], [.03, .02, -.01])
    packet["mean_generalized_force_n"] = packet["endpoint_generalized_force_n"]
    with pytest.raises(ValueError, match="direct cohesive path"):
        oracle.audit_cohesive_path(reseal(packet, "cohesive_response_sha256"), result)


@pytest.mark.parametrize("credit", ["historical_dissipation_available_for_creation_j",
    "historical_dissipation_credit_j", "arithmetic_roundoff_credit_j", "creation_work_debited_from_Genesis_j"])
def test_history_and_4096_j_roundoff_cannot_pay_for_creation(credit):
    with pytest.raises(ValueError, match="cannot fund"):
        oracle.scalar_zero_credit({credit: 4096.})
    oracle.scalar_zero_credit({credit: 0.})


def test_weak_inertia_uses_own_scale():
    with pytest.raises(ValueError, match="inertia"):
        oracle.mode_close(np.zeros(18), np.full(18, 1e-19), "weak inertia")


def test_midpoint_budget_equals_full_residual_dual_work():
    rng = np.random.default_rng(19)
    n = 24
    matrix = rng.normal(size=(n, n)); M = matrix.T@matrix+np.eye(n)
    D = np.diag(np.linspace(0, 1, n))
    d, v0, F, bulk = [rng.normal(size=n) for _ in range(4)]
    gamma = rng.normal(size=18)
    result = oracle.discrete_path_budget(M, D, F, d, v0, 3., bulk, gamma,
        np.asarray([bulk@d, 0., 0.]), float(gamma@d[-18:]))
    assert result["work_balance_residual_j"] == pytest.approx(result["residual_dual_work_j"], rel=3e-14)
    v1 = result["after_velocity_m_s"]
    assert result["kinetic_energy_change_j"] == pytest.approx(.5*(v1-v0)@M@(v1+v0), rel=3e-14)


def test_restriction_precedes_gram_and_preserves_weak_source_direction():
    G = np.zeros((2, 6, 18)); G[:, 0, 0] = 1.; G[:, 0, 1] = 1.
    G[1, 0, 1] += 2e-10
    E = G[:, :3].copy(); B = np.zeros_like(G); T = np.zeros_like(E)
    W = np.zeros((18, 1)); W[:2, 0] = [1/np.sqrt(2), -1/np.sqrt(2)]
    result = oracle.restrict_point_operators(G, E, B, T, np.ones(2), np.eye(6), 3000., W)
    rounded_gram = np.einsum("pai,paj->ij", G, G)
    rounded_restriction = float(W[:, 0]@rounded_gram@W[:, 0])
    true_restricted = float(result["stiffness_enriched"][0, 0])
    assert true_restricted > 0.
    assert rounded_restriction <= 0. or abs(rounded_restriction-true_restricted) > .5*true_restricted
    assert true_restricted == pytest.approx(float(np.sum(result["G_w"]**2)), rel=1e-15)
    assert result["mass_enriched"][0, 0] == pytest.approx(3000.*true_restricted, rel=1e-15)


def test_restricted_signed_cross_and_self_keep_their_own_scale():
    G = np.zeros((1, 6, 18)); G[0, 0, 0] = 1e-13
    E = np.zeros((1, 3, 18)); B = np.zeros_like(G); T = np.zeros_like(E)
    W = np.eye(18)[:, :1]
    quad = oracle.restrict_point_operators(G, E, B, T, [127.], np.eye(6), 3000., W)
    old = np.ones((1, 6)); added = quad["G_w"][:, :, 0]
    parts = oracle.restricted_inner_parts(quad, old, added, old, -added)
    assert parts[0] == 0.
    assert parts[1] < 0.
    assert float(parts[1]) == pytest.approx(-127e-26, rel=1e-15)
    with pytest.raises(ValueError, match="tiny represented self"):
        oracle.close(0., parts[1], "tiny represented self", rtol=3e-6, atol=1e-35)


def test_rounded_legacy_gram_is_kept_separate_from_source_point_work():
    G = np.zeros((2, 6, 18)); G[:, 0, :2] = 1.; G[1, 0, 1] += 2e-10
    Kaa = np.einsum("pai,paj->ij", G, G)
    da = np.zeros(18); da[:2] = [1e10, -1e10]
    zero = np.zeros((1, 3, 6))
    local = dict(barycentric=np.repeat(oracle.seed.prior.SOURCE_ANGULAR_POINTS[:1], 2, axis=0),
        radial_scale=[1., 1.], reference_volume_weights_m3=[1., 1.],
        enrichment_strain_operator=G, elasticity_pa=np.eye(6),
        stiffness_cross=np.zeros((18, 18)), stiffness_enriched=Kaa)
    context = dict(selected_index=0, parent_ndof=6)
    previous = dict(e0=zero, e1=zero, z=np.zeros(18))
    bulk = dict(strain0=zero, strain1=zero, coefficients_m=da,
        mean0=zero, mean1=zero, mean_coefficients_m=.5*da)
    witness = oracle.archived_legacy_small_response(context, dict(embedded_partition=dict(local=local)),
        previous, bulk, zero, zero, np.r_[np.zeros(6), da], [0.])
    represented = np.einsum("pai,i->pa", np.asarray(G, np.longdouble), np.asarray(da, np.longdouble))
    physical_computational_energy = .5*float(np.sum(represented**2))
    assert physical_computational_energy > 1.
    assert witness["after"][1] != pytest.approx(physical_computational_energy, rel=1e-6)
    assert witness["after"][1] == .5*float(da@Kaa@da)
    oracle.scalar_zero_credit(dict(arithmetic_roundoff_credit_j=0.))


def test_restricted_field_lift_cannot_drop_a_source_component():
    W = np.eye(18)[:, :15]
    full = np.zeros((1, 3, 18)); full[0, 0, 0] = 2.; full[0, 1, 17] = 1e-15
    a = np.zeros(18); a[0] = .003
    restricted = np.asarray(full, np.longdouble)@np.asarray(W, np.longdouble)
    report, passed = oracle.source_expansion_witnesses(dict(displacement=full),
        dict(displacement=restricted), W, dict(increment=a))
    assert passed and report["increment_displacement"]["maximum_absolute_difference"] == 0.
    a[-1] = 1.
    report, passed = oracle.source_expansion_witnesses(dict(displacement=full),
        dict(displacement=restricted), W, dict(increment=a))
    assert not passed
    assert report["increment_displacement"]["maximum_absolute_difference"] == 1e-15


def test_reduction_allowance_uses_actual_tiny_operands_without_floor():
    tiny = np.asarray([0., 1e-29], np.longdouble)
    bound = oracle.reduction_roundoff(tiny, 100)
    assert bound[0] == 0.
    assert 0 < bound[1] < 1e-42
    with pytest.raises(ValueError, match="unresolved"):
        oracle.reduction_roundoff(tiny, 10**30)


def test_cohesive_path_oracle_accepts_restricted_fifteen_coordinate_chart(law):
    trace = np.zeros((1, 3, 15)); trace[0, :, :3] = np.eye(3)
    old, new = np.zeros(15), np.zeros(15); new[:3] = [.03, -.02, .01]
    result = oracle.independent_cohesive_path(law, np.asarray([[1e6, 0., 0.]]),
        np.asarray([127.]), trace, np.eye(3)[None], old, new, np.asarray([.001]))
    body = plain({k: v for k, v in result.items() if k not in (
        "force_absolute_integral_n", "work_integral_roundoff_bound_j", "integrated_absolute_work_j")})
    body.update(kernel_only=True, physical_release_admitted=False, normal_contact_admitted=False,
        funding_admitted=False, numerically_integrated_work_j=result["mechanical_work_j"])
    oracle.audit_cohesive_path(reseal(body, "cohesive_response_sha256"), result)


@pytest.mark.parametrize("bad", [[True, 2.], [float("nan"), 1.], [1+2j], ["1"]])
def test_nonliteral_numeric_inputs_refused(bad):
    with pytest.raises(ValueError, match="finite real"):
        oracle.array(bad)


def test_static_dependency_closure_contains_no_producer_or_planetary_loader():
    root = Path(__file__).resolve().parents[1]
    pending = [root/"analysis/genesis_volume_release_validation/independent_audit.py"]
    seen = set()
    while pending:
        path = pending.pop()
        if path in seen: continue
        seen.add(path)
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                assert not module.startswith("tectonics")
                if module.startswith("analysis."):
                    base = root.joinpath(*module.split("."))
                    if base.with_suffix(".py").exists(): pending.append(base.with_suffix(".py"))
                    else:
                        for alias in node.names:
                            if (base/(alias.name+".py")).exists(): pending.append(base/(alias.name+".py"))
            if isinstance(node, ast.Import):
                assert all(not alias.name.startswith("tectonics") for alias in node.names)
            if isinstance(node, ast.Name):
                assert node.id != "load_volume_slip_checkpoint"
    assert len(seen) >= 7
