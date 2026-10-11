"""Fixed-material constitutive probes; none initialize or advance a planet."""
from dataclasses import replace
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tectonics.genesis_volume_basal import assemble_genesis_volume_basal
from tectonics.genesis_volume_embedded_partition import prepare_embedded_partition
from tectonics.genesis_volume_enriched_memory import (
    EnrichedMemoryRefusal, initial_enriched_memory, prepare_enriched_loading,
    evaluate_enriched_loading, accept_enriched_loading, enriched_memory_to_dict,
    enriched_memory_from_dict, validate_enriched_memory,
)
from tectonics.genesis_volume_finite_seed import prepare_volume_finite_seed
from tectonics.genesis_volume_loading import (
    RadialMoments, VolumeCohortMemory, VolumeMechanicsState, VolumeLoading,
    coefficient_inner_product, evaluate_volume_loading,
)
from tectonics.genesis_volume_operator import assemble_genesis_volume_operator


@pytest.fixture(scope="module")
def fixture():
    # Unchanged formation export, used only as a tensor/material test fixture.
    path = Path(__file__).resolve().parents[1]/"analysis/genesis_cohort_mechanics_validation/evidence/control/material_export.json"
    assembly = assemble_genesis_volume_operator(json.loads(path.read_text()))
    cohort = assembly.geometry.cohorts[0]
    center = .5*(cohort.inner_scale+cohort.outer_scale)*np.mean(cohort.top_triangle_m, axis=0)
    normal = assembly.basis_xyz[0]@np.array((.7, .3, .2))
    normal /= np.linalg.norm(normal)
    seed = prepare_volume_finite_seed(assembly.geometry, np.zeros(assembly.ndof), normal,
        float(normal@center), center, 100., cohort_id=cohort.id, quadrature_order=4)
    p = prepare_embedded_partition(assembly, seed, assembly.memory0, assembly.memory1,
        basal=assemble_genesis_volume_basal(assembly), quadrature_order=4)
    records = tuple(VolumeCohortMemory(int(i), 1400., p.memory0[j], p.memory1[j])
                    for j, i in enumerate(assembly.cohort_ids))
    parent = VolumeMechanicsState(p.source_identity, p.time_myr, 0, records,
        p.initial_elastic_energy_j, p.initial_elastic_energy_j)
    return p, parent


def loading(p, state, after_time=None, rate=.6):
    """Explicit constant-temperature/fixed-material Maxwell factors for probes."""
    parent = state.parent_state
    after_time = parent.time_myr+.001 if after_time is None else after_time
    n = len(parent.cohorts)
    old0 = np.array([c.strain0 for c in parent.cohorts])
    old1 = np.array([c.strain1 for c in parent.cohorts])
    x = np.full(n, rate*(after_time-parent.time_myr)/.001)
    beta = np.divide(-np.expm1(-x), x, out=np.ones_like(x), where=x != 0)
    r = np.exp(-x)[:, None, None]
    moments = RadialMoments(tuple(p.parent_assembly.volume_m3),
        tuple(p.parent_assembly.inverse_scale_mean), tuple(p.parent_assembly.inverse_scale_variance))
    return VolumeLoading(parent, tuple(p.parent_assembly.cohort_ids), after_time,
        np.array([c.temperature_k for c in parent.cohorts]), moments,
        old0, old1, np.zeros_like(old0), r*old0, r*old1, x, beta,
        p.parent_assembly.C, 0., 0.)


def probe(p, state, rate=.6, scale=1.):
    rng = np.random.default_rng(7061)
    q = rng.normal(size=p.parent_ndof)*.01*scale
    a = rng.normal(size=18)*.002*scale
    trial = prepare_enriched_loading(p, state, loading(p, state, rate=rate))
    return trial, evaluate_enriched_loading(p, trial, q, a)


def reseal(raw):
    raw["packet_sha256"] = hashlib.sha256(json.dumps(
        {k: v for k, v in raw.items() if k != "packet_sha256"},
        sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return raw


def test_birth_preserves_entire_original_memory_and_history(fixture):
    p, parent = fixture
    state = initial_enriched_memory(p, parent)
    assert state.parent_state is parent
    assert state.mechanics is parent
    np.testing.assert_array_equal(state.coefficients_m, np.zeros(18))
    np.testing.assert_array_equal(state.displacement_m, np.zeros(18))
    assert state.elastic_energy_j == parent.elastic_energy_j
    assert state.birth_time_myr == p.time_myr
    assert not state.physical_release_admitted and not state.funding_admitted
    assert state.private_steps == 0
    validate_enriched_memory(p, state)
    for array in (state.coefficients_m, state.displacement_m):
        with pytest.raises(ValueError): array.setflags(write=True)


def test_saved_birth_energy_roundoff_is_recorded_without_rewriting_history(fixture):
    p, parent = fixture
    # Frozen genuine current-parent witness: the producer/operator reduction
    # differs by exactly one float64 ULP, without a different strain field.
    assert 3.4441607157336404e19-3.44416071573364e19 == 4096.
    saved = p.initial_elastic_energy_j-4096.
    literal = replace(parent, initial_energy_j=saved, elastic_energy_j=saved)
    state = initial_enriched_memory(p, literal)
    assert state.parent_state is literal
    assert state.initial_energy_j == saved and state.elastic_energy_j == saved
    assert state.birth_operator_energy_roundoff_j == p.initial_elastic_energy_j-saved
    assert state.mechanical_work_j == 0 and state.viscous_dissipation_j == 0
    validate_enriched_memory(p, state)
    restored = enriched_memory_from_dict(enriched_memory_to_dict(state), p, literal)
    assert restored.elastic_energy_j == saved
    large = replace(parent, initial_energy_j=parent.initial_energy_j+1e10,
                    elastic_energy_j=parent.elastic_energy_j+1e10)
    with pytest.raises(EnrichedMemoryRefusal, match="Birth must preserve"):
        initial_enriched_memory(p, large)


def test_birth_rejects_replacement_strain_even_with_a_consistent_parent_ledger(fixture):
    p, parent = fixture
    records = list(parent.cohorts)
    changed = np.array(records[0].strain0); changed[0, 0] += 1e-9
    records[0] = replace(records[0], strain0=changed)
    changed_parent = replace(parent, cohorts=tuple(records))
    with pytest.raises(EnrichedMemoryRefusal, match="Birth must preserve"):
        initial_enriched_memory(p, changed_parent)


@pytest.mark.parametrize("rate", [0., 1e-8, .2, 2., 1000.])
def test_maxwell_endpoint_mean_and_full_field_balance(fixture, rate):
    p, parent = fixture
    state = initial_enriched_memory(p, parent)
    trial, response = probe(p, state, rate=rate)
    x = trial.parent_loading.ratio[p.selected_cohort_index]
    beta = 1. if x == 0 else -np.expm1(-x)/x
    theta = .5 if x == 0 else (1-beta)/x
    if x < 1e-3:
        theta = .5-x/6+x*x/24-x**3/120
    np.testing.assert_allclose(response.coefficients_m, beta*response.enrichment_increment_m, rtol=2e-15)
    np.testing.assert_allclose(response.mean_coefficients_m, theta*response.enrichment_increment_m, rtol=2e-14)
    np.testing.assert_array_equal(response.displacement_m, response.enrichment_increment_m)
    scale = max(response.elastic_energy_j, state.elastic_energy_j, abs(response.mechanical_work_j), 1.)
    assert abs(response.balance_residual_j) < 5e-12*scale
    assert response.viscous_dissipation_j >= 0
    assert response.elastic_energy_j == sum(response.energy_decomposition_j)
    assert response.mechanical_work_j == sum(response.work_decomposition_j)
    assert response.viscous_dissipation_j == sum(response.dissipation_decomposition_j)
    if rate == 0:
        assert response.viscous_dissipation_j == 0


def test_zero_enrichment_path_exactly_retains_original_loading_response(fixture):
    p, parent = fixture
    state = initial_enriched_memory(p, parent)
    q = np.random.default_rng(446).normal(size=p.parent_ndof)*.01
    trial = prepare_enriched_loading(p, state, loading(p, state))
    actual = evaluate_enriched_loading(p, trial, q)
    local = q[p.parent_assembly.dofs]
    inc0 = np.einsum("cqai,ci->cqa", p.parent_assembly.B0, local)
    inc1 = np.einsum("cqai,ci->cqa", p.parent_assembly.B1, local)
    old = evaluate_volume_loading(trial.parent_loading, inc0, inc1)
    for name in ("elastic_energy_j", "mechanical_work_j", "viscous_dissipation_j"):
        assert getattr(actual, name) == getattr(old, name)
    np.testing.assert_array_equal(actual.strain0, old.strain0)
    np.testing.assert_array_equal(actual.strain1, old.strain1)


def test_nonzero_memory_relaxes_without_adding_displacement_again(fixture):
    p, parent = fixture
    birth = initial_enriched_memory(p, parent)
    trial, response = probe(p, birth)
    state = accept_enriched_loading(p, trial, response)
    next_trial = prepare_enriched_loading(p, state, loading(p, state, rate=2.))
    second = evaluate_enriched_loading(p, next_trial, np.zeros(p.parent_ndof))
    x = next_trial.parent_loading.ratio[p.selected_cohort_index]
    np.testing.assert_array_equal(second.coefficients_m, np.exp(-x)*state.coefficients_m)
    np.testing.assert_array_equal(second.displacement_m, state.displacement_m)
    assert second.mechanical_work_j == 0
    assert second.elastic_energy_j < state.elastic_energy_j
    assert second.viscous_dissipation_j > 0
    validate_enriched_memory(p, accept_enriched_loading(p, next_trial, second))


def test_signed_cross_loss_is_retained_without_becoming_a_funding_ledger(fixture):
    p, parent = fixture
    birth = initial_enriched_memory(p, parent)
    trial, positive = probe(p, birth)
    opposite = evaluate_enriched_loading(p, trial, positive.parent_increment_m,
                                         -positive.enrichment_increment_m)
    # Cross terms change sign; enriched self term does not.  Their sign does
    # not define a separately admissible dissipation or source of crack work.
    assert positive.dissipation_decomposition_j[1] == pytest.approx(
        -opposite.dissipation_decomposition_j[1], rel=2e-15)
    assert positive.dissipation_decomposition_j[2] == opposite.dissipation_decomposition_j[2]
    assert min(positive.dissipation_decomposition_j[1], opposite.dissipation_decomposition_j[1]) < 0
    assert min(positive.viscous_dissipation_j, opposite.viscous_dissipation_j) > 0
    assert not birth.funding_admitted


def test_nonzero_restart_replays_from_trusted_source_birth(fixture):
    p, parent = fixture
    state = initial_enriched_memory(p, parent)
    for scale in (1., -.3):
        trial, response = probe(p, state, scale=scale)
        state = accept_enriched_loading(p, trial, response)
    raw = json.loads(json.dumps(enriched_memory_to_dict(state)))
    restored = enriched_memory_from_dict(raw, p, parent,
        lambda s, t: loading(p, s, t))
    assert enriched_memory_to_dict(restored) == raw
    assert restored.private_steps == 2
    np.testing.assert_array_equal(restored.coefficients_m, state.coefficients_m)
    np.testing.assert_array_equal(restored.displacement_m, state.displacement_m)
    with pytest.raises(EnrichedMemoryRefusal, match="source-bound loading replay"):
        enriched_memory_from_dict(raw, p, parent)


@pytest.mark.parametrize("field", ["coefficients_m", "displacement_m", "parent_mechanics", "ratio"])
def test_correctly_resealed_restart_replacement_is_rejected(fixture, field):
    p, parent = fixture
    trial, response = probe(p, initial_enriched_memory(p, parent))
    state = accept_enriched_loading(p, trial, response)
    raw = copy.deepcopy(enriched_memory_to_dict(state))
    if field == "ratio": raw["state"]["journal"][0]["ratio"][0] += .01
    elif field == "parent_mechanics":
        raw["state"][field]["state"]["cohorts"][0]["strain0"][0][0] += 1e-6
    else: raw["state"][field][0] += .001
    with pytest.raises(EnrichedMemoryRefusal):
        enriched_memory_from_dict(reseal(raw), p, parent, lambda s, t: loading(p, s, t))


@pytest.mark.parametrize("change", ["thermal", "thermal_shape", "temperature", "elasticity", "material", "boundary", "beta", "memory"])
def test_unsupported_material_changes_and_factor_tampering_are_rejected(fixture, change):
    p, parent = fixture
    state = initial_enriched_memory(p, parent)
    good = loading(p, state)
    if change == "thermal":
        value = good.thermal0.copy(); value[0, 0, 0] = 1e-9
        bad = replace(good, thermal0=value)
    elif change == "thermal_shape": bad = replace(good, thermal0=np.zeros((1, 1, 6)))
    elif change == "temperature": bad = replace(good, temperatures_k=good.temperatures_k+1)
    elif change == "elasticity": bad = replace(good, elasticity_pa=good.elasticity_pa*1.001)
    elif change == "material": bad = replace(good, after_ids=good.after_ids[1:])
    elif change == "boundary":
        moments = good.radial_moments
        bad = replace(good, radial_moments=replace(moments, volume_m3=tuple(np.array(moments.volume_m3)*.999)))
    elif change == "beta": bad = replace(good, beta=good.beta*.999)
    else: bad = replace(good, memory0=good.memory0*1.001)
    with pytest.raises(EnrichedMemoryRefusal): prepare_enriched_loading(p, state, bad)


def test_modified_response_cannot_be_accepted(fixture):
    p, parent = fixture
    trial, response = probe(p, initial_enriched_memory(p, parent))
    fake = replace(response, coefficients_m=response.coefficients_m+.001)
    with pytest.raises(EnrichedMemoryRefusal, match="modified"):
        accept_enriched_loading(p, trial, fake)
