"""Independent scalar and full-field Maxwell audit counterexamples.

Manufactured arrays below are local mathematical quadrature probes. They do
not initialize, replace or continue any planetary runtime state.
"""
from __future__ import annotations

import ast
import copy
from pathlib import Path

import numpy as np
import pytest

from analysis.genesis_volume_memory_validation import independent_audit as audit


def mathematical_context():
    g, w = np.polynomial.legendre.leggauss(12)
    lam, wr = .7+.15*g, w/2
    # Fixed local Cartesian engineering strain matrices; not a mature planet.
    B0 = np.zeros((1, 3, 6, 18)); B1 = np.zeros_like(B0)
    for k in range(3):
        B0[0, k, :, 6*k:6*k+6] = np.eye(6)
    C = np.diag([7., 9., 11., 3., 4., 5.])
    N = np.asarray([[.2, .3, .5], [.5, .2, .3]])
    G = np.zeros((2, 6, 18))
    G[0, :, :6] = .7*np.eye(6); G[1, :, :6] = -.3*np.eye(6)
    B = np.zeros_like(G)
    for k in range(3):
        B[:, :, 6*k:6*k+6] = N[:, k, None, None]*np.eye(6)
    local = dict(radial_scale=np.asarray([.68, .74]), enrichment_strain_operator=G,
        parent_strain_operator=B, elasticity_pa=C, reference_volume_weights_m3=np.asarray([.15, .12]))
    return dict(cohort_ids=[1], selected_index=0, parent_ndof=18, radial_scale=lam[None],
        radial_weights=wr[None], volume_m3=np.asarray([1.]), elasticity_pa=C[None],
        B0=B0, B1=B1, dofs=np.arange(18)[None], selected_parent_dofs=np.arange(18),
        interpolation=N@np.linalg.inv(audit.seed.prior.SOURCE_ANGULAR_POINTS), local=local)


def mathematical_step(context, *, ratio=.8, old_z=None, increment_a=None):
    old0 = np.asarray([[[.12, -.05, .03, .02, .01, -.02]]*3])
    old1 = np.asarray([[[.004, -.002, .001, .002, .001, .0003]]*3])
    inc0 = .17*old0; inc1 = -.25*old1
    za = np.zeros(18) if old_z is None else old_z
    da = np.r_[[-.06, .03, -.02, .01, .004, .005], np.zeros(12)] if increment_a is None else increment_a
    result = audit.maxwell_step(context, old0, old1, za, inc0, inc1, da, [ratio])
    return result, (old0, old1, za, inc0, inc1, da)


def test_oracle_has_no_producer_imports_or_planetary_replay():
    tree = ast.parse(Path(audit.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(not name.name.startswith("tectonics") for name in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("tectonics")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in {"__import__", "eval", "exec", "load_volume_slip_checkpoint"}


@pytest.mark.parametrize("ratio", [0., 1e-9, .02, .8, 7., 1000.])
def test_time_quadrature_closes_each_original_cross_self_ledger(ratio):
    context = mathematical_context()
    raw, _ = mathematical_step(context, ratio=ratio)
    report = audit.audit_response(audit.plain(raw), raw)
    assert report["original_cross_self_terms_checked_separately"]
    assert raw["viscous_dissipation_j"] >= 0
    expected = 1. if ratio == 0 else -np.expm1(-ratio)/ratio
    np.testing.assert_allclose(raw["coefficients_m"][:6],
        expected*np.asarray([-.06, .03, -.02, .01, .004, .005]), rtol=2e-14, atol=1e-16)


def test_independent_enriched_work_matches_direct_point_time_integration():
    context = mathematical_context()
    response, (old0, old1, za, inc0, inc1, da) = mathematical_step(context, old_z=np.r_[np.full(6, .007), np.zeros(12)])
    x, w = np.polynomial.legendre.leggauss(100)
    time, weights = (x+1)/2, w/2
    ratio = .8
    direct_work, direct_loss = np.zeros(3), np.zeros(3)
    for t, weight in zip(time, weights):
        r, b = np.exp(-ratio*t), -np.expm1(-ratio*t)/ratio
        e0, e1, z = r*old0+b*inc0, r*old1+b*inc1, r*za+b*da
        direct_work += weight*audit.field_inner_decomposition(context, inc0, inc1, da, e0, e1, z)
        direct_loss += weight*ratio*audit.field_inner_decomposition(context, e0, e1, z, e0, e1, z)
    np.testing.assert_allclose(response["work_decomposition_j"], direct_work, rtol=3e-14, atol=1e-15)
    np.testing.assert_allclose(response["dissipation_decomposition_j"], direct_loss, rtol=3e-14, atol=1e-15)


def test_negative_cross_loss_is_retained_but_full_loss_is_nonnegative():
    response, _ = mathematical_step(mathematical_context())
    assert response["dissipation_decomposition_j"][1] < 0
    assert response["viscous_dissipation_j"] > 0


@pytest.mark.parametrize("key", ["coefficients_m", "mean_coefficients_m"])
def test_fabricated_zero_enriched_memory_or_average_is_rejected(key):
    expected, _ = mathematical_step(mathematical_context())
    forged = audit.plain(expected)
    forged[key] = [0.]*18
    with pytest.raises(ValueError, match=key):
        audit.audit_response(forged, expected)


@pytest.mark.parametrize("key,index", [("energy_decomposition_j", 1), ("energy_decomposition_j", 2),
    ("work_decomposition_j", 1), ("dissipation_decomposition_j", 1), ("dissipation_decomposition_j", 2)])
def test_small_cross_and_self_terms_cannot_be_erased_under_old_energy(key, index):
    expected, _ = mathematical_step(mathematical_context())
    forged = audit.plain(expected)
    forged[key][index] = 0.
    with pytest.raises(ValueError, match=key):
        audit.audit_response(forged, expected)


def test_wrong_endpoint_substituted_for_average_field_is_rejected():
    expected, _ = mathematical_step(mathematical_context())
    forged = audit.plain(expected)
    forged["mean0"] = forged["strain0"]
    with pytest.raises(ValueError, match="mean0"):
        audit.audit_response(forged, expected)


@pytest.mark.parametrize("key", ["thermal_work_j", "damage_energy_change_j", "remelt_energy_removed_j"])
def test_frozen_private_scope_rejects_unsupported_driver(key):
    expected, _ = mathematical_step(mathematical_context())
    forged = audit.plain(expected); forged[key] = 1.
    with pytest.raises(ValueError, match="frozen private response"):
        audit.audit_response(forged, expected)


def test_old_field_is_not_reconstructed_by_adding_absolute_displacement():
    context = mathematical_context()
    response, args = mathematical_step(context)
    old0, old1, za, inc0, inc1, da = args
    # An unrelated total displacement history does not occur in the ODE inputs.
    assert "absolute_displacement_m" not in context
    another = audit.maxwell_step(context, old0, old1, za, inc0, inc1, da, [.8])
    np.testing.assert_array_equal(response["strain0"], another["strain0"])


def test_zero_driving_retains_relaxed_nonzero_z_instead_of_erasing_history():
    context = mathematical_context()
    prior = np.r_[np.full(6, .007), np.zeros(12)]
    response, _ = mathematical_step(context, old_z=prior, increment_a=np.zeros(18))
    np.testing.assert_allclose(response["coefficients_m"], np.exp(-.8)*prior, rtol=2e-15, atol=0.)
    assert np.linalg.norm(response["coefficients_m"]) > 0


def sealed(body, key):
    return {**body, key: audit.digest(body)}


def mathematical_birth_archive():
    context = mathematical_context()
    _, (old0, old1, _, _, _, _) = mathematical_step(context)
    energy = float(.5*audit.field_inner_decomposition(context, old0, old1, np.zeros(18),
        old0, old1, np.zeros(18)).sum())
    ledgers = dict(initial_energy_j=energy, elastic_energy_j=energy, mechanical_work_j=0.,
        viscous_dissipation_j=0., thermal_work_j=0., damage_energy_change_j=0.,
        remelt_energy_removed_j=0., balance_residual_j=0.)
    state = dict(source_identity="a"*64, time_myr=.3, accepted_steps=4,
        cohorts=[dict(id=1, temperature_k=900., strain0=old0[0].tolist(), strain1=old1[0].tolist())], **ledgers)
    memory = sealed(dict(format="genesis-volume-mechanics-0.1",
        energy_scope="prospective_full_isotropic_conical_elastic_memory",
        voigt_order=["xx", "yy", "zz", "xy", "xz", "yz"], state=state), "packet_sha256")
    context["parent"] = dict(runtime_state=dict(mechanics=memory, geometry=dict(time_myr=.3)))
    context["birth_operator_energy_roundoff_j"] = 0.
    raw = sealed(dict(format="genesis-volume-enriched-memory-0.1", scope=audit.MEMORY_SCOPE,
        physical_release_admitted=False, funding_admitted=False, new_Genesis_time_myr=0.,
        new_material_birth_count=0, state=dict(parent_mechanics=copy.deepcopy(memory),
            partition_binding_sha256="c"*64, birth_parent_memory_sha256=memory["packet_sha256"],
            birth_time_myr=.3, displacement_m=[0.]*18, coefficients_m=[0.]*18,
            birth_operator_energy_roundoff_j=0., journal=[], **ledgers)), "packet_sha256")
    kwargs = dict(expected0=old0, expected1=old1, expected_z=np.zeros(18), expected_a=np.zeros(18),
        expected_time=.3, expected_steps=0, expected_journal=[], birth=True)
    return context, raw, kwargs


def test_exact_birth_archive_keeps_old_memory_and_creates_only_zero_new_field():
    context, raw, kwargs = mathematical_birth_archive()
    report = audit.audit_state_archive(raw, context, "c"*64, **kwargs)
    assert report["zero_amplitude_birth_exact"] is True


@pytest.mark.parametrize("forgery", ["old_memory", "new_coefficients", "temperature"])
def test_resealed_birth_archive_cannot_replace_history(forgery):
    context, raw, kwargs = mathematical_birth_archive()
    if forgery == "old_memory":
        raw["state"]["parent_mechanics"]["state"]["cohorts"][0]["strain0"][0][0] += .01
    elif forgery == "new_coefficients":
        raw["state"]["coefficients_m"][0] = .01
    else:
        raw["state"]["parent_mechanics"]["state"]["cohorts"][0]["temperature_k"] += 10.
    memory = raw["state"]["parent_mechanics"]
    memory["packet_sha256"] = audit.digest({k: v for k, v in memory.items() if k != "packet_sha256"})
    raw["packet_sha256"] = audit.digest({k: v for k, v in raw.items() if k != "packet_sha256"})
    with pytest.raises(ValueError, match="memory|history|temperature"):
        audit.audit_state_archive(raw, context, "c"*64, **kwargs)


def mathematical_reaction():
    n = 18
    internal = np.tile([2e6, -3e6, 1e6, .03, -.04, .01], 3)
    inertia = np.tile([1e-22, -2e-22, 3e-22, 4e-28, -5e-28, 6e-28], 3)
    residual = internal+inertia
    report = dict(before_time_myr=.2, after_time_myr=.3, momentum_equation_relative_residual=0.)
    runtime = dict(geometry=dict(source_identity="a"*64, time_myr=.3),
        last_report=report, accepted_step_sizes_myr=[.1], mechanics={}, history={}, fault_history={})
    parent = dict(checkpoint_sha256="b"*64, source_checkpoint_sha256="d"*64, runtime_state=runtime)
    context = dict(parent=parent, parent_ndof=n, cohort_ids=[1], selected_index=0)
    partition = dict(embedded_partition_sha256="c"*64, local=dict(quadrature_order=12))
    fields = dict(dt_s=1., parent_increment_m=np.zeros(n), parent_before_velocity_m_s=np.zeros(n),
        interval_average_elastic_constant=np.zeros((1, 3, 6)), interval_average_elastic_inverse_scale=np.zeros((1, 3, 6)),
        parent_inertia_force_n=np.zeros(n), parent_drag_force_n=np.zeros(n),
        parent_average_internal_force_n=internal.copy(), parent_external_force_n=internal.copy(),
        parent_residual_n=np.zeros(n), added_inertia_force_n=inertia.copy(), added_drag_force_n=np.zeros(18),
        added_average_internal_force_n=internal.copy(), added_external_force_n=np.zeros(18), added_residual_n=residual,
        endpoint_static_constraint_reaction_n=-2*internal, constraint_multipliers_n=-residual,
        constraint_force_n=np.r_[np.zeros(n), -residual], remaining_residual_n=np.zeros(n+18),
        enrichment_increment_m=np.zeros(18), enrichment_before_velocity_m_s=np.zeros(18), enrichment_elastic_memory_m=np.zeros(18))
    body = dict(format="genesis-accepted-interval-tied-compact-seed-reaction-0.1", scope=audit.REACTION_SCOPE,
        accepted_parent_checkpoint_sha256=parent["checkpoint_sha256"], formation_checkpoint_sha256=parent["source_checkpoint_sha256"],
        source_identity="a"*64, time_myr=.3, before_time_myr=.2, after_time_myr=.3, accepted_step_sizes_myr=[.1],
        accepted_final_report_sha256=audit.digest(report), full_memory_sha256=audit.digest({}),
        birth_history_sha256=audit.digest({}), fault_history_sha256=audit.digest({}), endpoint_geometry_sha256=audit.digest(runtime["geometry"]),
        finite_seed_binding_sha256="f"*64, embedded_partition_binding_sha256="c"*64, partition_quadrature_order=12,
        selected_cohort_id=1, parent_ndof=n, recomputed_parent_momentum_relative_residual=0., parent_momentum_relative_residual=0.,
        parent_operator_blocks_preserved_exactly=True, parent_residual_preserved_exactly=True,
        constraint_force_parent_pullback_exact_zero=True, added_residual_after_constraints_exact_zero=True,
        tied_constraint_work_j=0., exact_canonical_jump_work_j=0., added_mode_basal_drag_and_external_force_exact_zero=True,
        mass_convention="after_mass_with_accepted_start_comoving_velocity",
        average_response_scope="saved_last_accepted_interval_not_endpoint_static_stress",
        basis_scope="accepted_endpoint_frozen_computational_basis_hypothetically_extended_over_saved_interval",
        added_memory_scope="new_zero_compact_coefficients_old_calculated_memory_unchanged",
        reaction_semantics="negative_added_inertia_plus_accepted_average_internal_residual",
        surface_force_is_dynamic_reaction=False, surface_traction_replacement=None,
        new_accepted_intervals=0, new_material_births=0, released_physical_dofs=0,
        transition_time_increment_myr=0., creation_work_debited_from_Genesis_j=0.)
    body.update({key: False for key in ("physical_creation_admitted", "physical_release_admitted", "funding_admitted",
        "historical_seed_claimed", "source_memory_reinitialized", "surface_traction_replacement_admitted", "activation_accepted")})
    raw = sealed({**body, "reaction_context_sha256": audit.digest(body), **audit.plain(fields)}, "seed_reaction_sha256")
    return context, partition, fields, raw


def test_full_tied_reaction_audits_tiny_inertia_separately(monkeypatch):
    context, partition, expected, raw = mathematical_reaction()
    monkeypatch.setattr(audit, "dynamic_reaction_expected", lambda context: expected)
    report = audit.audit_dynamic_reaction(raw, context, partition, "f"*64)
    assert report["added_inertia_checked_separately_even_below_total_force_roundoff"]


@pytest.mark.parametrize("forgery", ["omitted_inertia", "endpoint_stress", "funding"])
def test_resealed_dynamic_reaction_rejects_missing_inertia_or_wrong_stress(monkeypatch, forgery):
    context, partition, expected, raw = mathematical_reaction()
    monkeypatch.setattr(audit, "dynamic_reaction_expected", lambda context: expected)
    if forgery == "omitted_inertia":
        raw["added_inertia_force_n"] = [0.]*18
    elif forgery == "endpoint_stress":
        raw["added_average_internal_force_n"] = (-np.asarray(raw["endpoint_static_constraint_reaction_n"])).tolist()
    else:
        raw["funding_admitted"] = True
        body = {k: v for k, v in raw.items() if k not in set(expected)|{"reaction_context_sha256", "seed_reaction_sha256"}}
        raw["reaction_context_sha256"] = audit.digest(body)
    raw["seed_reaction_sha256"] = audit.digest({k: v for k, v in raw.items() if k != "seed_reaction_sha256"})
    with pytest.raises(ValueError, match="inertia|average_internal|funding"):
        audit.audit_dynamic_reaction(raw, context, partition, "f"*64)


def force_witness():
    context = mathematical_context()
    expected, _ = mathematical_step(context)
    raw = audit.plain(expected)
    raw["parent_mean_force_n"] = audit.full_generalized_force(context, expected["mean0"], expected["mean1"], np.zeros(18))[:18].tolist()
    raw["enriched_parent_force_n"] = audit.enriched_parent_force(context, expected["mean_coefficients_m"]).tolist()
    return context, expected, raw


def test_parent_coupling_force_is_checked_separately_from_original_force():
    context, expected, raw = force_witness()
    assert audit.audit_force_decomposition(raw, context, expected)["enriched_parent_coupling_checked_separately"]


@pytest.mark.parametrize("forgery", ["zero_coupling", "wrong_sum"])
def test_parent_force_decomposition_rejects_erased_coupling(forgery):
    context, expected, raw = force_witness()
    if forgery == "zero_coupling": raw["enriched_parent_force_n"] = [0.]*18
    else: raw["mean_generalized_force_n"][0] += .01
    with pytest.raises(ValueError, match="coupling|decomposition"):
        audit.audit_force_decomposition(raw, context, expected)


def convergence_witness():
    vector = np.arange(1., 19.)
    raw = dict(orders=[8, 12], observed_relative_tolerance=2e-6,
        added_inertia_force_n_order8=(1+1e-8)*vector, added_average_internal_force_n_order8=(1-2e-8)*vector)
    reaction = dict(added_inertia_force_n=vector, added_average_internal_force_n=vector)
    for source, key in (("added_inertia_force_n", "added_inertia_errors_A_P"),
        ("added_average_internal_force_n", "added_average_force_errors_A_P")):
        before, after = raw[source+"_order8"].reshape(-1, 6), reaction[source].reshape(-1, 6)
        raw[key] = [float(np.linalg.norm(before[:, i:i+3]-after[:, i:i+3])/np.linalg.norm(after[:, i:i+3])) for i in (0, 3)]
    return audit.plain(raw), audit.plain(reaction)


def test_observed_convergence_recomputes_literal_force_family_errors():
    raw, reaction = convergence_witness()
    assert audit.audit_reaction_convergence(raw, reaction)["observed_order_eight_twelve_agreement"]


def test_reported_fake_zero_convergence_error_is_rejected():
    raw, reaction = convergence_witness(); raw["added_inertia_errors_A_P"] = [0., 0.]
    with pytest.raises(ValueError, match="convergence"):
        audit.audit_reaction_convergence(raw, reaction)
