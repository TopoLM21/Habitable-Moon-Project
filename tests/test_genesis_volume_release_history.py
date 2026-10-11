"""Native recurrence parser arithmetic; synthetic fixtures are not planets.

Authentic factory and full source replay are covered by release_work tests.
These small cases isolate zero-birth recurrence and schema forgery rejection.
"""
from dataclasses import replace
import copy
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from tectonics.genesis_volume_checkpoint import _canonical
from tectonics.genesis_volume_release_history import (
    NativeReleaseHistoryRefusal, _make, _advance_native_release_state,
    _rebuild_native_release_state, native_release_state_to_dict,
    expand_native_release_increment,
)
from test_genesis_volume_release_quadrature import weak_fields


def synthetic_chain():
    context, _ = weak_fields()
    context.partition.parent_ndof = 2
    context.partition.selected_cohort_index = 0
    context.partition.time_myr = 4.
    context.modes = SimpleNamespace(rank=1)
    context.binding_sha256 = 'c'*64
    birth = SimpleNamespace(binding_sha256='d'*64, time_myr=4., journal=(),
        displacement_m=np.zeros(18), coefficients_m=np.zeros(18))
    context.birth = SimpleNamespace(binding_sha256='e'*64, initial_memory=birth)
    native = _make(context, birth.binding_sha256, np.zeros(1), np.zeros(1), np.zeros(1), 4., [])
    dq, dw = np.array((.125, -.25)), np.array((3e5,))
    d = np.r_[dq, dw]
    expanded = expand_native_release_increment(context, d)[2:]
    x = .25
    loading = SimpleNamespace(ratio=np.array((x,)), beta=np.array((-np.expm1(-x)/x,)))
    after = SimpleNamespace(time_myr=4.+1e-6, binding_sha256='f'*64,
        journal=(SimpleNamespace(after_time_myr=4.+1e-6,
            parent_increment_m=dq, enrichment_increment_m=expanded, ratio=(x,)),))
    result = _advance_native_release_state(context, native, after, d, loading)
    return context, native, after, d, loading, result


def test_native_large_weak_coordinate_and_memory_follow_original_scalar_Maxwell():
    c, before, after, d, loading, value = synthetic_chain()
    raw = native_release_state_to_dict(value)
    np.testing.assert_array_equal(value.displacement_w_m, [3e5])
    np.testing.assert_array_equal(value.coefficients_w_m, loading.beta*d[2:])
    np.testing.assert_array_equal(value.velocity_w_m_s,
        2*d[2:]/((after.time_myr-before.time_myr)*1e6*365.25*86400))
    assert raw['native_birth_from_authentic_zero_enrichment']
    assert not raw['mature_full18_projection_used_as_native_initialization']
    assert not raw['physical_release_admitted'] and not raw['funding_admitted']
    assert _rebuild_native_release_state(c, raw, after).binding_sha256 == value.binding_sha256


def test_native_live_arrays_and_decoded_journal_cannot_mutate_the_history():
    *_, value = synthetic_chain()
    with pytest.raises(ValueError): value.coefficients_w_m.setflags(write=True)
    journal = value.journal
    journal[0]['increment_w_m'][0] = 0.
    assert value.journal[0]['increment_w_m'] == [3e5]
    with pytest.raises(NativeReleaseHistoryRefusal, match='changed'):
        native_release_state_to_dict(replace(value, velocity_w_m_s=np.zeros(1)))


@pytest.mark.parametrize('field', [
    'displacement_w_m', 'coefficients_w_m', 'velocity_w_m_s', 'full18_memory_sha256',
    'source_memory_birth_binding_sha256', 'mature_full18_projection_used_as_native_initialization',
    'historical_dissipation_available_for_creation_j',
])
def test_resealed_native_state_cannot_reinitialize_or_credit_history(field):
    c, _, after, _, _, value = synthetic_chain()
    bad = native_release_state_to_dict(value)
    if isinstance(bad[field], list): bad[field][0] += 1.
    elif isinstance(bad[field], str): bad[field] = '0'*64
    elif isinstance(bad[field], bool): bad[field] = not bad[field]
    else: bad[field] = 4096.
    bad['native_release_state_sha256'] = hashlib.sha256(_canonical({k: v for k, v in bad.items()
        if k != 'native_release_state_sha256'})).hexdigest()
    with pytest.raises(NativeReleaseHistoryRefusal, match='zero-birth journal'):
        _rebuild_native_release_state(c, bad, after)


@pytest.mark.parametrize('field', ['selected_maxwell_ratio', 'selected_maxwell_alpha',
    'selected_maxwell_beta', 'source_expanded_increment_full18_m', 'parent_increment_m', 'dt_s'])
def test_native_journal_cannot_replace_source_ratios_or_full18_expansion(field):
    c, _, after, _, _, value = synthetic_chain()
    bad = native_release_state_to_dict(value)
    if isinstance(bad['journal'][0][field], list): bad['journal'][0][field][0] += 1.
    else: bad['journal'][0][field] += 1.
    with pytest.raises(NativeReleaseHistoryRefusal, match='original source Maxwell'):
        _rebuild_native_release_state(c, bad, after)


def test_projecting_a_mature_archive_cannot_replace_the_required_native_chain():
    c, born, after, *_ = synthetic_chain()
    bad = native_release_state_to_dict(born)
    bad['time_myr'] = after.time_myr
    bad['full18_memory_sha256'] = after.binding_sha256
    with pytest.raises(NativeReleaseHistoryRefusal, match='journal lengths'):
        _rebuild_native_release_state(c, bad, after)
