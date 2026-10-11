"""Native journal math unit fixtures; no planetary initialization or restore."""
import ast
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from analysis.genesis_volume_release_validation import independent_native_audit as oracle


def sealed(body, key='native_release_state_sha256'):
    body = {k: v for k, v in body.items() if k != key}
    return {**body, key: oracle.digest(body)}


@pytest.fixture
def lineage(monkeypatch):
    # Minimal DTO protocol fixture.  Authentic planetary source authentication
    # belongs to audit_case; this fixture tests only journal replay arithmetic.
    W = np.eye(18)[:, :3]
    law = dict(fracture_energy_j_m2=1e6, peak_traction_pa=4e6,
               cohesive_stiffness_pa_m=1e9, compression_stiffness_pa_m=2e9)
    source = dict(trace=np.eye(3)[None], weights=np.ones(1), basis=np.eye(3)[None],
                  initial_traction_world=np.zeros((1, 3)))
    initial = dict(journal=[], displacement_m=[0.]*18, coefficients_m=[0.]*18)
    prior = dict(birth_memory=dict(state=deepcopy(initial)), source_memory_birth=dict(pin='unit-source-birth'),
                 embedded_partition=dict(embedded_partition_sha256='unit-literal-points'))
    parent = dict(runtime_state=dict(geometry=dict(time_myr=1.)))
    checked = dict(numerical_context=dict(parent_ndof=1, selected_index=0),
        operative_source=source, law=law, initial_maximum=np.zeros(1),
        modes=dict(rank=3, observable_force_basis=W), context_binding_sha256='unit-context',
        restricted_quadrature=dict(density_kg_m3=3000., array_sha256={}))
    monkeypatch.setattr(oracle.memory, 'source_maxwell_ratios', lambda p, old, new: np.array([(new-old)*1000]))
    return parent, prior, checked, dict(state=initial)


def native_packet(lineage, full, increments=()):
    parent, prior, checked, _ = lineage
    W = np.asarray(checked['modes']['observable_force_basis'], np.longdouble)
    c, z, v, time, journal = np.zeros(3), np.zeros(3), np.zeros(3), 1., []
    for dq, dw, after in increments:
        dt = (after-time)*oracle.memory.SECONDS_PER_MYR
        x = float((after-time)*1000)
        alpha, beta = float(np.exp(-x)), float(-np.expm1(-x)/x) if x else 1.
        dw = np.asarray(dw, float)
        expanded = np.asarray(W@np.asarray(dw, np.longdouble), float)
        journal.append(dict(after_time_myr=after, dt_s=dt, parent_increment_m=[dq],
            increment_w_m=dw.tolist(), source_expanded_increment_full18_m=expanded.tolist(),
            selected_maxwell_ratio=x, selected_maxwell_alpha=alpha, selected_maxwell_beta=beta))
        c, z, v = c+dw, alpha*z+beta*dw, 2*dw/dt-v
        time = after
    body = dict(format=oracle.NATIVE_FORMAT, scope=oracle.NATIVE_SCOPE,
        context_binding_sha256=checked['context_binding_sha256'],
        source_memory_birth_binding_sha256=oracle.digest(prior['source_memory_birth']),
        source_initial_full18_memory_sha256=oracle.digest(prior['birth_memory']['state']),
        source_quadrature_binding_sha256=oracle.quadrature_binding(checked, prior),
        full18_memory_sha256=oracle.digest(full['state']), native_rank=3,
        accepted_source_birth_time_myr=1., time_myr=time, displacement_w_m=c.tolist(),
        coefficients_w_m=z.tolist(), velocity_w_m_s=v.tolist(), journal=journal,
        native_birth_from_authentic_zero_enrichment=True,
        mature_full18_projection_used_as_native_initialization=False,
        full18_fields_scope='expanded_float64_arithmetic_witness_only_not_authoritative_coordinates',
        cohort_parent_memory_scope='unchanged_original_source_Maxwell_parent_archive',
        accepted_planet_memory_changed=False, physical_release_admitted=False, funding_admitted=False,
        new_accepted_Genesis_time_myr=0., creation_work_debited_from_Genesis_j=0.,
        historical_dissipation_available_for_creation_j=0.)
    return sealed(body)


def full_witness(lineage, increments):
    _, _, checked, original = lineage
    full = deepcopy(original)
    W, time = np.asarray(checked['modes']['observable_force_basis'], np.longdouble), 1.
    for dq, dw, after in increments:
        full['state']['journal'].append(dict(after_time_myr=after, parent_increment_m=[dq],
            enrichment_increment_m=np.asarray(W@np.asarray(dw, np.longdouble), float).tolist(),
            ratio=[(after-time)*1000]))
        time = after
    return full


def test_exact_zero_birth_preserves_source_parent_identity(lineage):
    parent, prior, checked, full = lineage
    raw = native_packet(lineage, full)
    result = oracle.replay_native_history(raw, full, checked, parent, prior)
    np.testing.assert_array_equal(result['c'], np.zeros(3))
    np.testing.assert_array_equal(result['z'], np.zeros(3))
    np.testing.assert_array_equal(result['velocity'], np.zeros(3))
    assert result['report']['native_zero_birth_replayed']
    assert not result['report']['mature_full18_projection_used']


def test_linked_history_replays_native_memory_velocity_and_restart(lineage):
    parent, prior, checked, _ = lineage
    increments = [(2e-6, [1e-4, -2e-4, 3e-4], 1.0001), (-1e-6, [-.5e-4, 1e-4, -1.5e-4], 1.00017)]
    full = full_witness(lineage, increments)
    raw = native_packet(lineage, full, increments)
    first = oracle.replay_native_history(raw, full, checked, parent, prior)
    second = oracle.replay_native_history(deepcopy(raw), deepcopy(full), checked, parent, prior)
    np.testing.assert_array_equal(first['c'], [.5e-4, -1e-4, 1.5e-4])
    np.testing.assert_array_equal(first['z'], second['z'])
    np.testing.assert_array_equal(first['velocity'], second['velocity'])
    assert first['report']['native_journal_steps'] == 2
    assert first['binding_sha256'] == second['binding_sha256']
    assert np.all(first['maximum'] >= 0)


@pytest.mark.parametrize('field', ['displacement_w_m', 'coefficients_w_m', 'velocity_w_m_s'])
def test_resealed_native_coordinate_rewrite_is_refused(lineage, field):
    parent, prior, checked, _ = lineage
    increments = [(0., [1e-4, -2e-4, 3e-4], 1.0001)]
    full = full_witness(lineage, increments)
    raw = native_packet(lineage, full, increments)
    raw[field][1] += 1e-12
    with pytest.raises(ValueError, match='authentic zero-birth journal'):
        oracle.replay_native_history(sealed(raw), full, checked, parent, prior)


@pytest.mark.parametrize('field', ['selected_maxwell_ratio', 'selected_maxwell_alpha', 'selected_maxwell_beta', 'dt_s'])
def test_resealed_source_factor_or_clock_rewrite_is_refused(lineage, field):
    parent, prior, checked, _ = lineage
    increments = [(0., [1e-4, -2e-4, 3e-4], 1.0001)]
    full = full_witness(lineage, increments)
    raw = native_packet(lineage, full, increments)
    raw['journal'][0][field] *= 1.000001
    with pytest.raises(ValueError, match='original source Maxwell'):
        oracle.replay_native_history(sealed(raw), full, checked, parent, prior)


@pytest.mark.parametrize('field', ['parent_increment_m', 'enrichment_increment_m'])
def test_native_history_must_match_original_parent_and_expanded_journal(lineage, field):
    parent, prior, checked, _ = lineage
    increments = [(1e-6, [1e-4, -2e-4, 3e-4], 1.0001)]
    full = full_witness(lineage, increments)
    raw = native_packet(lineage, full, increments)
    full['state']['journal'][0][field][0] += 1e-14
    raw['full18_memory_sha256'] = oracle.digest(full['state'])
    with pytest.raises(ValueError, match='increment link|witness lift'):
        oracle.replay_native_history(sealed(raw), full, checked, parent, prior)


def test_zero_native_history_cannot_bind_mature_full18_projection(lineage):
    parent, prior, checked, full = lineage
    full['state']['displacement_m'][0] = 1.
    raw = native_packet(lineage, full)
    with pytest.raises(ValueError, match='cannot bind or project a mature'):
        oracle.replay_native_history(raw, full, checked, parent, prior)


@pytest.mark.parametrize('field,value', [
    ('mature_full18_projection_used_as_native_initialization', True),
    ('native_birth_from_authentic_zero_enrichment', False),
    ('historical_dissipation_available_for_creation_j', 4096.),
    ('source_memory_birth_binding_sha256', 'made-up-source'),
    ('context_binding_sha256', 'different-context'),
    ('source_quadrature_binding_sha256', 'different-points')])
def test_native_packet_cannot_reseal_changed_source_or_funding_scope(lineage, field, value):
    parent, prior, checked, full = lineage
    raw = native_packet(lineage, full)
    raw[field] = value
    with pytest.raises(ValueError, match='authentic zero-birth journal'):
        oracle.replay_native_history(sealed(raw), full, checked, parent, prior)


def test_native_math_view_does_not_project_state_or_change_source_tensors():
    q = dict(basis_w=np.ones((18, 15)), G_w=np.arange(90).reshape(1, 6, 15), E_w=np.arange(45).reshape(1, 3, 15))
    view = oracle.native_coordinate_math_view(q)
    np.testing.assert_array_equal(view['basis_w'], np.eye(15))
    assert view['G_w'] is q['G_w'] and view['E_w'] is q['E_w']
    assert q['basis_w'].shape == (18, 15)


def test_native_history_binding_and_selfseal_are_distinct_domains(lineage):
    parent, prior, checked, full = lineage
    raw = native_packet(lineage, full)
    result = oracle.replay_native_history(raw, full, checked, parent, prior)
    assert result['binding_sha256'] == oracle.digest(raw)
    assert result['binding_sha256'] != raw['native_release_state_sha256']


def test_source_point_witness_compares_saved_native_coefficients_directly():
    # Strong/weak mixing during a float64 lift loses the tiny first native
    # amplitude.  The authoritative field must still use its saved amplitude.
    W = np.eye(18)[:, :3]
    W[:2, :2] = np.array([[1., 1.], [-1., 1.]])/np.sqrt(2)
    native = np.array([1e-8, 2e8, 0.])
    full = np.asarray(np.asarray(W, np.longdouble)@native.astype(np.longdouble), float)
    recovered = np.asarray(np.asarray(W, np.longdouble).T@full.astype(np.longdouble), float)
    assert recovered[0] != native[0]
    G = np.zeros((1, 3, 18)); G[0, :, :3] = np.eye(3)
    restricted = np.einsum('pai,ij->paj', G.astype(np.longdouble), W.astype(np.longdouble))
    reports, passed = oracle.source_expansion_witnesses(
        {'strain': G}, {'strain': restricted}, W, {'saved': full}, {'saved': native})
    assert passed and reports['saved_strain']['passed']
    assert reports['saved_strain']['maximum_absolute_difference'] > 0
    corrupt = native.copy(); corrupt[1] += 1.
    _, passed = oracle.source_expansion_witnesses(
        {'strain': G}, {'strain': restricted}, W, {'saved': full}, {'saved': corrupt})
    assert not passed


def test_canceling_source_lifts_report_cumulative_kernel_rounding_without_projection():
    # Every individual expansion belongs to the same 15-dimensional range.
    # Float64 accumulation can leave a residual outside it after cancellation;
    # the strict literal18 criterion must honestly report failure.
    Q, _ = np.linalg.qr(np.random.default_rng(190).normal(size=(18, 18)))
    W, N = Q[:, :15], Q[:, 15:]
    first = np.zeros(15); first[:3] = [1.0100001e8, -2.77777e8, 4.8981e8]
    second = -first; second[:3] += [5e-8, -8e-8, 0.]
    expanded = lambda value: np.asarray(W.astype(np.longdouble)@value.astype(np.longdouble), float)
    a, b = expanded(first), expanded(second)
    cumulative, native = a+b, first+second
    saved = cumulative.copy()
    reports, all_passed = oracle.legacy_kernel_witnesses(N,
        dict(first=a, second=b, cumulative=cumulative))
    assert reports['first']['literal18_kernel_tolerance_passed']
    assert reports['second']['literal18_kernel_tolerance_passed']
    assert not reports['cumulative']['literal18_kernel_tolerance_passed'] and not all_passed
    np.testing.assert_array_equal(cumulative, saved)
    assert np.linalg.norm(N.T@cumulative) > 512*oracle.EPS*np.linalg.norm(cumulative)
    with pytest.raises(ValueError, match='complete jump-kernel'):
        oracle.literal.require_tied_kernel({'declared_tied_null_modes': N}, cumulative, 'strict literal18')
    # Native coordinates retain their own source-born sum, not a projection of
    # the damaged arithmetic witness.  Their dimensionality has no null slot.
    assert native.shape == (15,)
    assert not np.array_equal(native, W.T@cumulative)


def test_canceling_lifts_leave_point_discrepancy_as_honest_nonadmitting_witness():
    Q, _ = np.linalg.qr(np.random.default_rng(190).normal(size=(18, 18)))
    W = Q[:, :15]
    first = np.zeros(15); first[:3] = [1.0100001e8, -2.77777e8, 4.8981e8]
    second = -first; second[:3] += [5e-8, -8e-8, 0.]
    lift = lambda value: np.asarray(W.astype(np.longdouble)@value.astype(np.longdouble), float)
    full, native = lift(first)+lift(second), first+second
    G = np.eye(18).reshape(6, 3, 18)
    G_w = np.einsum('pai,ij->paj', G.astype(np.longdouble), W.astype(np.longdouble))
    reports, passed = oracle.source_expansion_witnesses(
        {'strain': G}, {'strain': G_w}, W, {'cumulative': full}, {'cumulative': native})
    assert not passed and not reports['cumulative_strain']['passed']
    assert reports['cumulative_strain']['maximum_difference_to_arithmetic_bound_ratio'] > 1.
    np.testing.assert_array_equal(native, first+second)


def test_endpoint_addition_exposes_lost_tiny_increment_without_energy_credit():
    old = np.array([1e16, 0., 0.])
    increment = np.array([1., 0., 0.])
    witness = oracle.endpoint_addition_witness(np.r_[old, np.zeros(15)], np.r_[old+increment, np.zeros(15)],
        np.r_[increment, np.zeros(15)], old, old+increment, increment)
    assert witness['difference'][0] == -1.
    assert witness['passed']  # Addition arithmetic is honest, not a work budget.
    # Mean force contracted with supplied increment still differs from actual
    # zero-length endpoint path by 1 J; a huge old ledger cannot pay this work.
    actual_path_work, supplied_force_work = 0., float(np.ones(3)@increment)
    current_bound = 3e-9*(abs(actual_path_work)+abs(supplied_force_work))+128*oracle.EPS
    assert abs(supplied_force_work-actual_path_work) > current_bound


def test_native_oracle_and_reused_literal_oracle_have_no_producer_imports():
    for module in (oracle, oracle.literal):
        source = Path(module.__file__).read_text()
        imports = [n for n in ast.walk(ast.parse(source)) if isinstance(n, (ast.Import, ast.ImportFrom))]
        names = [a.name for n in imports if isinstance(n, ast.Import) for a in n.names]
        names += [n.module or '' for n in imports if isinstance(n, ast.ImportFrom)]
        assert not any(name.startswith('tectonics') for name in names)
