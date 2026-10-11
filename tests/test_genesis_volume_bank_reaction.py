"""Accepted-source dynamic tie reactions and complete weak traction replacement.

The physical parent is an actual one-year continuation from the immutable
formation export. Supplied edge supports and deliberately restricted trace
operators test computation; no fault activation or new source history is seeded.
"""
from dataclasses import FrozenInstanceError, replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tectonics.genesis_volume_bank_reaction import (
    VolumeReactionRefusal, VolumeReactionRangeRefusal, _digest,
    prepare_volume_bank_reaction, prepare_volume_reaction_traction,
    validate_volume_bank_reaction,
    volume_bank_reaction_to_dict, volume_reaction_traction_to_dict,
    validate_volume_bank_reaction_payload, validate_volume_reaction_traction_payload)
from tectonics.genesis_volume_banks import prepare_volume_banks, evaluate_volume_banks
from tectonics.genesis_volume_checkpoint import _canonical
from tectonics.genesis_volume_fracture_surface import (
    prepare_volume_fracture_surface, evaluate_surface_jump, integrate_surface_traction)
from tectonics.genesis_volume_slip_checkpoint import volume_slip_checkpoint_to_dict
from tectonics.genesis_volume_slip_dynamics import VolumeSlipModel


@pytest.fixture(scope="module")
def actual():
    path = Path(__file__).resolve().parents[1]/(
        "analysis/genesis_cohort_mechanics_validation/evidence/control/material_export.json")
    source_bytes = path.read_bytes()
    model = VolumeSlipModel(json.loads(source_bytes))
    initial = model.initial_state()
    state = model.step(initial, 1e-6)
    before = _canonical(volume_slip_checkpoint_to_dict(model, state))
    parent, basal = model._assembly(state.geometry, state.fault_history.damage)
    a, b, c = state.geometry.faces[0]
    banks = prepare_volume_banks(parent, ((a, b), (b, c), (c, a)), basal=basal)
    surface = prepare_volume_fracture_surface(banks)
    reaction = prepare_volume_bank_reaction(model, state, banks)
    traction = prepare_volume_reaction_traction(reaction, surface)
    yield model, initial, state, parent, basal, banks, surface, reaction, traction
    assert path.read_bytes() == source_bytes
    assert _canonical(volume_slip_checkpoint_to_dict(model, state)) == before


def assert_six_force_close(actual, expected, rtol=2e-10):
    for sl in (slice(0, 3), slice(3, 6)):
        a, b = np.asarray(actual).reshape(-1, 6)[:, sl], np.asarray(expected).reshape(-1, 6)[:, sl]
        assert np.linalg.norm(a-b) <= rtol*max(np.linalg.norm(b), 1e-10)


def test_requires_a_real_accepted_interval_not_an_initialization(actual):
    model, initial, _, _, _, banks, *_ = actual
    with pytest.raises(VolumeReactionRefusal, match="accepted interval"):
        prepare_volume_bank_reaction(model, initial, banks)


def test_source_ancestry_clock_and_average_force_binding(actual):
    model, _, state, _, _, banks, _, reaction, _ = actual
    raw = volume_bank_reaction_to_dict(reaction)
    packet = volume_slip_checkpoint_to_dict(model, state)
    assert raw["accepted_parent_checkpoint_sha256"] == packet["checkpoint_sha256"]
    assert raw["formation_checkpoint_sha256"] == packet["source_checkpoint_sha256"]
    assert raw["time_myr"] == state.time_myr
    assert raw["bank_binding_sha256"] == banks.binding_sha256
    assert raw["transition_time_increment_myr"] == raw["creation_work_debited_from_Genesis_j"] == 0
    assert raw["source_calculated_creation_law"] is None
    assert not raw["physical_creation_admitted"] and not raw["physical_release_admitted"]
    np.testing.assert_array_equal(reaction.interval_average_elastic_constant,
        state.last_report["weak_plane_slip"]["average_strain0"])
    np.testing.assert_array_equal(reaction.interval_average_elastic_inverse_scale,
        state.last_report["weak_plane_slip"]["average_strain1"])
    assert not state.fault_history.fault_active.any()


def test_average_force_is_not_endpoint_prestress(actual):
    *_, banks, _, reaction, _ = actual
    state = actual[2]
    e0 = np.array([c.strain0 for c in state.mechanics.cohorts])
    e1 = np.array([c.strain1 for c in state.mechanics.cohorts])
    endpoint = evaluate_volume_banks(banks, np.zeros(banks.ndof),
        elastic_constant=e0, elastic_inverse_scale=e1).force
    assert np.linalg.norm(endpoint-reaction.bank_average_internal_force_n) > 1e-10*np.linalg.norm(endpoint)
    # Neither accepted Δu nor absolute q is added again to the average memory.
    wrong = evaluate_volume_banks(banks, reaction.bank_increment_m,
        elastic_constant=reaction.interval_average_elastic_constant,
        elastic_inverse_scale=reaction.interval_average_elastic_inverse_scale).force
    assert np.linalg.norm(wrong-reaction.bank_average_internal_force_n) > 1e-10*np.linalg.norm(wrong)


def test_tie_reaction_balances_banks_and_preserves_parent_residual(actual):
    banks, reaction = actual[5], actual[7]
    J, R = banks.J.toarray(), banks.R.toarray()
    assert np.linalg.norm(reaction.constraint_force_n) > 0
    np.testing.assert_allclose(J@reaction.remaining_residual_n, 0,
        atol=5e-15*np.linalg.norm(reaction.bank_residual_n))
    parent_remainder = R@np.linalg.solve(R.T@R, reaction.parent_residual_n)
    for sl in (slice(0, 3), slice(3, 6)):
        error = np.linalg.norm((reaction.remaining_residual_n-parent_remainder).reshape(-1, 6)[:, sl])
        scale = np.linalg.norm(reaction.bank_residual_n.reshape(-1, 6)[:, sl])
        assert error < 2e-13*scale
    assert np.linalg.norm(reaction.parent_residual_n) > 0
    assert np.linalg.norm(reaction.remaining_residual_n) > 0
    np.testing.assert_array_equal(J@reaction.bank_increment_m, np.zeros(len(J)))
    np.testing.assert_array_equal(J@reaction.bank_before_velocity_m_s, np.zeros(len(J)))
    translation = R@np.tile((.8, -.3, .7, .2, -.1, .5), R.shape[1]//6)
    assert abs(reaction.constraint_force_n@translation) < 2e-14*np.linalg.norm(reaction.constraint_force_n)*np.linalg.norm(translation)


def test_inertia_and_drag_are_present_and_reactions_do_no_tied_work(actual):
    reaction = actual[7]
    assert np.linalg.norm(reaction.bank_drag_force_n) > 0
    assert np.linalg.norm(reaction.bank_inertia_force_n) > 0
    residual = (reaction.bank_inertia_force_n+reaction.bank_drag_force_n
        +reaction.bank_average_internal_force_n-reaction.bank_external_force_n)
    np.testing.assert_array_equal(reaction.bank_residual_n, residual)
    scale = max(np.linalg.norm(reaction.constraint_force_n)*np.linalg.norm(reaction.bank_increment_m), 1.)
    assert abs(reaction.constraint_force_n@reaction.bank_increment_m) < 2e-14*scale


def test_gamma_force_replaces_complete_full_depth_reaction(actual):
    surface, reaction, traction = actual[6:9]
    assert traction.rank == len(reaction.constraint_multipliers_n)
    assert_six_force_close(traction.replacement_force_n, reaction.constraint_force_n)
    assert np.max(traction.modal_scale) > 1e4
    assert np.linalg.norm(traction.world_traction_pa) > 0
    weights = surface.area_weights_m2
    assert traction.area_l2_traction_squared_pa2_m2 == pytest.approx(
        np.sum(weights*np.sum(traction.world_traction_pa**2, axis=-1)), rel=3e-15)
    # All three local traction components contribute on the authentic support.
    edges = surface.patch_edge_indices
    frame = np.stack((surface.edge_normals_xyz[edges], surface.edge_tangent1_xyz[edges],
                      surface.edge_tangent2_xyz[edges]), axis=-1)
    local = np.einsum("pji,pqj->pqi", frame, traction.world_traction_pa)
    assert np.all(np.linalg.norm(local, axis=(0, 1)) > 1e-12*np.linalg.norm(local))


def test_replacement_virtual_work_and_area_l2_minimum(actual):
    banks, surface, reaction, traction = actual[5:9]
    probe = np.sin(.13*np.arange(banks.ndof)+.3)
    trace = evaluate_surface_jump(surface, probe)
    rhs = np.sum(surface.area_weights_m2*np.sum(traction.world_traction_pa*trace, axis=-1))
    assert traction.replacement_force_n@probe == pytest.approx(rhs, rel=2e-13, abs=1e-3)
    assert reaction.constraint_force_n@probe == pytest.approx(rhs, rel=2e-10, abs=1e-3)
    root_w = np.sqrt(np.repeat(surface.area_weights_m2.ravel(), 3))
    L = traction.modal_scale[:, None]*traction.canonical_trace.T*root_w[None, :]
    z = root_w*traction.world_traction_pa.ravel()
    random = np.cos(.71*np.arange(len(z)))
    null = random-np.linalg.lstsq(L, L@random, rcond=traction.svd_relative_cutoff)[0]
    assert np.linalg.norm(L@null) < 2e-12*np.linalg.norm(L)*np.linalg.norm(random)
    assert abs(z@null) < 2e-12*np.linalg.norm(z)*np.linalg.norm(null)
    assert np.dot(z+np.linalg.norm(z)*null, z+np.linalg.norm(z)*null) > z@z


@pytest.mark.parametrize("support", ("empty", "single_edge_tip"))
def test_empty_and_shared_tip_supports_have_no_new_reaction(actual, support):
    model, _, state, parent, basal, *_ = actual
    a, b, _ = state.geometry.faces[0]
    banks = prepare_volume_banks(parent, [] if support == "empty" else [(a, b)], basal=basal)
    surface = prepare_volume_fracture_surface(banks)
    reaction = prepare_volume_bank_reaction(model, state, banks)
    traction = prepare_volume_reaction_traction(reaction, surface)
    assert len(reaction.constraint_multipliers_n) == traction.rank == 0
    np.testing.assert_array_equal(reaction.constraint_force_n, np.zeros(banks.ndof))
    np.testing.assert_array_equal(traction.world_traction_pa, np.zeros((surface.patch_count, 4, 3)))
    assert traction.area_l2_traction_squared_pa2_m2 == 0


@pytest.mark.parametrize("missing", ("all", "depth_P"))
def test_missing_trace_modes_refuse_instead_of_projecting_away_reaction(actual, missing):
    # Explicit computational operator restriction, never an actual source Γ.
    surface, reaction = actual[6:8]
    trace = np.array(surface.jump_trace)
    if missing == "all":
        trace[:] = 0
    else:
        trace[..., np.array((3, 4, 5, 9, 10, 11, 15, 16, 17, 21, 22, 23))] = 0
    restricted = replace(surface, jump_trace=trace, binding_sha256="")
    restricted = replace(restricted, binding_sha256=_digest(restricted))
    with pytest.raises(VolumeReactionRangeRefusal):
        prepare_volume_reaction_traction(reaction, restricted)


@pytest.mark.parametrize("field", ("physical_release_admitted", "instantaneous_birth_traction_claimed",
                                    "replacement_force_n", "accepted_parent_checkpoint_sha256"))
def test_resealed_traction_dto_cannot_supply_physical_claim_or_changed_force(actual, field):
    traction = actual[8]
    raw = volume_reaction_traction_to_dict(traction)
    if field == "replacement_force_n":
        raw[field][0] += 1e8
    elif field.endswith("sha256"):
        raw[field] = "0"*64
    else:
        raw[field] = True
    raw["traction_sha256"] = hashlib.sha256(_canonical({k: v for k, v in raw.items() if k != "traction_sha256"})).hexdigest()
    with pytest.raises(VolumeReactionRefusal, match="context"):
        validate_volume_reaction_traction_payload(raw, expected_traction=traction)


def test_full_contextual_serializers_roundtrip_and_refuse_changed_reaction(actual):
    reaction, traction = actual[7:9]
    raw = volume_bank_reaction_to_dict(reaction)
    assert validate_volume_bank_reaction_payload(json.loads(_canonical(raw)), expected_reaction=reaction) == raw
    tr = volume_reaction_traction_to_dict(traction)
    assert validate_volume_reaction_traction_payload(json.loads(_canonical(tr)), expected_traction=traction) == tr
    raw["constraint_multipliers_n"][0] *= 1.01
    raw["reaction_sha256"] = hashlib.sha256(_canonical({k: v for k, v in raw.items() if k != "reaction_sha256"})).hexdigest()
    with pytest.raises(VolumeReactionRefusal, match="context"):
        validate_volume_bank_reaction_payload(raw, expected_reaction=reaction)


def test_prepared_arrays_are_immutable_and_mismatched_support_is_refused(actual):
    model, _, state, parent, basal, banks, surface, reaction, traction = actual
    for value in (reaction.bank_residual_n, reaction.constraint_force_n,
                  reaction.interval_average_elastic_constant, traction.world_traction_pa,
                  traction.modal_scale, traction.canonical_trace):
        with pytest.raises(ValueError):
            value.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        reaction.dt_s = 1.
    with pytest.raises(VolumeReactionRefusal, match="binding"):
        validate_volume_bank_reaction(replace(reaction, dt_s=1.))
    other = prepare_volume_banks(parent, [], basal=basal)
    with pytest.raises(ValueError, match="different"):
        prepare_volume_reaction_traction(reaction, prepare_volume_fracture_surface(other))


def test_changed_accepted_mean_cannot_be_used_as_a_balanced_interval(actual):
    model, _, state, _, _, banks, *_ = actual
    report = json.loads(json.dumps(state.last_report))
    report["weak_plane_slip"]["average_strain0"][0][0][0] += .001
    candidate = replace(state, last_report=report)
    with pytest.raises(VolumeReactionRefusal, match="parent momentum"):
        prepare_volume_bank_reaction(model, candidate, banks)
