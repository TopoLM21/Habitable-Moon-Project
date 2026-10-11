"""Release diagnostics on an archived formation tensor branch, never a planet reset."""
from dataclasses import FrozenInstanceError, replace
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tectonics.genesis_volume_checkpoint import _canonical
from tectonics.genesis_volume_creation_law import ProspectiveCreationLaw
from tectonics.genesis_volume_dynamics import coefficient_arrays
from tectonics.genesis_volume_embedded_partition import prepare_embedded_partition
from tectonics.genesis_volume_embedded_prestress import pull_back_embedded_prestress
from tectonics.genesis_volume_finite_seed import prepare_volume_finite_seed
from tectonics.genesis_volume_material_stress import bind_volume_material_stress, evaluate_volume_material_stress
from tectonics.genesis_volume_nucleation import bind_volume_nucleation_proposal, VolumeNucleationProposal
from tectonics.genesis_volume_release_work import (
    VolumeReleaseWorkRefusal, prepare_volume_release_work, validate_volume_release_work,
    volume_release_work_to_dict, volume_release_work_from_dict,
    evaluate_release_work, volume_release_work_response_to_dict,
    replay_volume_release_work_response, _summation_roundoff,
    evaluate_native_release_work, replay_native_release_work_response, _full18_kernel_witnesses,
)
from tectonics.genesis_volume_release_history import (
    make_native_release_birth, native_release_state_to_dict, restart_native_release_state,
    expand_native_release_increment,
)
from tectonics.genesis_volume_release_modes import validate_release_increment
from tectonics.genesis_volume_seed_memory import prepare_source_seed_memory
from tectonics.genesis_volume_slip_checkpoint import volume_slip_checkpoint_to_dict
from tectonics.genesis_volume_slip_dynamics import VolumeSlipModel


def make_archived_tensor_context(law=None):
    """Independent numerical fixture; this is not the accepted current planet."""
    path = Path(__file__).resolve().parents[1]/(
        "analysis/genesis_cohort_mechanics_validation/evidence/control/material_export.json")
    original = path.read_bytes()
    model = VolumeSlipModel(json.loads(original))
    state = model.step(model.initial_state(), 1e-6)
    parent, basal = model._assembly(state.geometry, state.fault_history.damage)
    c = state.geometry.cohorts[0]
    pose = state.displacement_m[parent.dofs[0]].reshape(3, 6)
    triangle = np.asarray(c.top_triangle_m)
    n, lam = np.full(3, 1/3.), .5*(c.inner_scale+c.outer_scale)
    center = n@(lam*(triangle+pose[:, 3:])+(pose[:, :3]-pose[:, 3:]))
    normal = np.array((1., 2., 3.)); normal /= np.linalg.norm(normal)
    # This deliberately clipped 100 m tensor probe resolves the weak surface
    # directions without a rank tolerance change.  It is not a source-selected
    # current-planet seed or a certified clipped-bulk quadrature.
    radius = 100.
    seed = prepare_volume_finite_seed(state.geometry, state.displacement_m, normal,
        float(normal@center), center, radius, cohort_id=c.id, quadrature_order=8)
    e0, e1 = coefficient_arrays(state.mechanics)
    p = prepare_embedded_partition(parent, seed, e0, e1, basal=basal, quadrature_order=4)
    birth = prepare_source_seed_memory(model, state, seed, partition=p, quadrature_order=4)
    observer = bind_volume_material_stress(model, state)
    samples = evaluate_volume_material_stress(observer, seed.surface.cohort_ids,
        seed.surface.barycentric, seed.surface.mass_coordinate_q)
    pullback = pull_back_embedded_prestress(seed.surface, samples.stress_world_tensor)
    strength = max(10.*float(np.max(np.linalg.norm(pullback.reference_traction_local_pa, axis=1))), 1e7)
    law = ProspectiveCreationLaw(strength, strength, 100.*strength, 100.*strength) if law is None else law
    proposal = bind_volume_nucleation_proposal(model, state)
    context = prepare_volume_release_work(birth, seed, law,
        source_nucleation_proposal=proposal, parent_velocity_m_s=state.velocity_m_s)
    assert path.read_bytes() == original
    return context, model, state, proposal, law


@pytest.fixture(scope="module")
def tensor():
    result = make_archived_tensor_context()
    context, model, state, *_ = result
    before = _canonical(volume_slip_checkpoint_to_dict(model, state))
    yield result
    assert _canonical(volume_slip_checkpoint_to_dict(model, state)) == before
    validate_volume_release_work(context)


@pytest.fixture(scope="module")
def zero_path(tensor):
    c, *_ = tensor
    return evaluate_release_work(c, c.birth.initial_memory,
        c.cohesive.seed.initial_maximum_separation_m, np.zeros(c.partition.ndof),
        c.partition.time_myr+1e-6)


@pytest.fixture(scope="module")
def added_path(tensor):
    c, *_ = tensor
    d = np.zeros(c.partition.ndof)
    d[c.partition.parent_ndof:] = c.modes.release_basis@np.sin(np.arange(c.modes.rank)+.2)*2e-7
    return evaluate_release_work(c, c.birth.initial_memory,
        c.cohesive.seed.initial_maximum_separation_m, d, c.partition.time_myr+1e-6)


def test_source_context_preserves_genuine_memory_and_velocities(tensor):
    c, _, state, proposal, law = tensor
    raw = volume_release_work_to_dict(c)
    assert c.birth.initial_memory.parent_state is state.mechanics
    np.testing.assert_array_equal(c.initial_velocity_m_s[:c.partition.parent_ndof], state.velocity_m_s)
    np.testing.assert_array_equal(c.initial_velocity_m_s[c.partition.parent_ndof:], np.zeros(18))
    assert raw["prospective_creation_law"]["fracture_energy_j_m2"] == law.fracture_energy_j_m2
    assert raw["source_nucleation_proposal"] == json.loads(proposal.payload_json)
    assert raw["complete_jump_kernel_remains_tied"]
    assert c.modes.nullity >= 3
    assert raw["historical_dissipation_available_for_creation_j"] == 0.
    assert raw["creation_work_debited_from_Genesis_j"] == raw["new_accepted_Genesis_time_myr"] == 0.
    assert not c.physical_release_admitted and not c.funding_admitted
    assert "accepted_Genesis_has_no_new_source_activation" in c.physical_refusals


def test_saved_reaction_is_retained_as_a_different_scope_diagnostic(tensor):
    c, *_ = tensor
    np.testing.assert_array_equal(c.archived_constraint_multipliers_n,
        c.birth.reaction.constraint_multipliers_n)
    np.testing.assert_array_equal(c.archived_reaction_minus_initial_surface_force_n,
        c.birth.reaction.constraint_multipliers_n-c.initial_surface_force_n)
    assert np.linalg.norm(c.archived_reaction_minus_initial_surface_force_n) > 0.
    assert volume_release_work_to_dict(c)["dynamic_bank_reaction_replaced"] is False


def test_scalar_work_and_kinematics_do_not_certify_unbalanced_zero_path(zero_path):
    raw = volume_release_work_response_to_dict(zero_path)
    assert raw["kinematic_identity_passed"] and raw["work_identity_passed"]
    assert raw["discrete_work_balance_passed"]
    assert not raw["projected_momentum_passed"] and not raw["balanced_trajectory"]
    assert raw["external_work_j"] == raw["budget_required_work_j"] == 0.
    assert raw["kinetic_change_stable_j"] == 0.
    assert raw["bulk_dissipation_decomposition_j"][0] > 0.
    assert raw["historical_dissipation_available_for_creation_j"] == 0.


def test_actual_added_path_keeps_all_force_and_work_terms(added_path, tensor):
    c, *_ = tensor
    raw = volume_release_work_response_to_dict(added_path)
    d = np.asarray(raw["increment_full_m"])
    forces = [np.asarray(raw[k]) for k in ("inertia_force_n", "basal_drag_force_n",
        "bulk_mean_force_n", "cohesive_mean_force_n")]
    residual = sum(forces)-c.partition.external_force_n
    np.testing.assert_array_equal(raw["residual_n"], residual)
    assert raw["external_work_j"] == c.partition.external_force_n@d
    assert raw["kinetic_change_stable_j"] == pytest.approx(
        np.asarray(raw["operative_increment_m"])@np.asarray(raw["operative_inertia_force_n"]), rel=3e-14)
    assert np.any(np.asarray(raw["inertia_force_n"])[c.partition.parent_ndof:] != 0.)
    assert raw["bulk_force_work_duality_passed"]
    assert raw["bulk_small_component_balance_passed"]
    assert raw["work_identity_passed"]
    np.testing.assert_allclose(raw["bulk_force_work_decomposition_j"],
        raw["bulk_work_decomposition_j"], rtol=3e-9, atol=0.)
    assert raw["cohesive_work_j"] == pytest.approx(added_path.cohesive_response.force_work_j, rel=3e-13)
    assert raw["cohesive_endpoint_work_agrees_at_integral_precision"]
    np.testing.assert_allclose(sum(raw["kinetic_change_decomposition_j"]),
        raw["kinetic_change_stable_j"], rtol=3e-14, atol=0.)
    assert not raw["funding_admitted"] and not raw["physical_release_admitted"]
    np.testing.assert_allclose(c.observable_force_basis.T@c.observable_force_basis,
        np.eye(c.modes.rank), rtol=0., atol=2e-14)
    observed = np.asarray(raw["observable_force_residual_n"])
    np.testing.assert_array_equal(observed,
        np.asarray(raw["operative_projected_residual_n"])[c.partition.parent_ndof:])
    assert c.observable_force_singular_values_m[-1] > 0.
    assert raw["source_point_expansion_passed"]
    assert raw["legacy_witness_differences_available_for_funding"] is False
    assert raw["weak_geometry_accuracy_certified"] is False
    np.testing.assert_array_equal(raw["legacy_rounded_bulk_work_decomposition_j"],
        added_path.bulk_response.work_decomposition_j)


def test_cancelled_point_force_has_its_own_operand_roundoff_bound():
    # The exact sum of these represented operands is one.  Float64 sequential
    # accumulation loses it, although a bound based on the net result is zero.
    operands = np.array((1e16, 1., -1e16))
    rounded = float((operands[0]+operands[1])+operands[2])
    exact = np.sum(np.asarray(operands, np.longdouble))
    assert rounded == 0. and exact == 1.
    own = _summation_roundoff(np.sum(np.abs(operands)), 3, epsilon=np.finfo(float).eps)
    assert abs(rounded-exact) <= own
    assert 256*np.finfo(float).eps*abs(rounded) < abs(rounded-exact)


def test_exact_zero_operand_roundoff_has_no_floor():
    np.testing.assert_array_equal(_summation_roundoff(np.zeros(4), 100000), np.zeros(4))
    tiny = np.longdouble("1e-200")
    assert 0. < _summation_roundoff(tiny, 10) < tiny


def test_primary_momentum_and_duality_report_own_point_bounds(added_path):
    raw = volume_release_work_response_to_dict(added_path)
    names = ("inertia", "basal_drag", "bulk", "cohesive", "external")
    parts = [np.asarray(raw["operative_"+name+"_component_roundoff_bound_n"]) for name in names]
    assert all(np.all(part >= 0.) for part in parts)
    assert np.any(parts[2] > 0.) and np.any(parts[3] > 0.)
    scale = np.asarray(raw["projected_residual_scale_n"])
    bound = np.asarray(raw["projected_momentum_component_bound_n"])
    np.testing.assert_allclose(bound, raw["momentum_relative_tolerance"]*scale
        +sum(parts)+256*np.finfo(float).eps*scale, rtol=3e-15, atol=0.)
    work = np.asarray(raw["bulk_work_decomposition_j"])
    force_work = np.asarray(raw["bulk_force_work_decomposition_j"])
    own = np.asarray(raw["bulk_force_work_duality_bound_decomposition_j"])
    assert np.all(np.abs(work-force_work) <= own)
    assert raw["legacy_witness_differences_available_for_funding"] is False


@pytest.mark.parametrize("kind", ["increment", "velocity"])
def test_complete_jump_kernel_is_never_released(tensor, kind):
    c, *_ = tensor
    d, velocity = np.zeros(c.partition.ndof), None
    if kind == "increment":
        d[c.partition.parent_ndof:] = c.modes.tied_null_modes[:, 0]*1e-6
    else:
        velocity = np.array(c.initial_velocity_m_s)
        velocity[c.partition.parent_ndof:] = c.modes.tied_null_modes[:, 0]*1e-8
    with pytest.raises(ValueError, match="kernel|velocity"):
        evaluate_release_work(c, c.birth.initial_memory,
            c.cohesive.seed.initial_maximum_separation_m, d, c.partition.time_myr+1e-6,
            before_velocity_m_s=velocity)


@pytest.mark.parametrize("field", ["available_work_j", "historical_dissipation_j", "external_force_n"])
def test_funding_credits_and_replacement_forcing_are_not_inputs(tensor, field):
    c, *_ = tensor
    with pytest.raises(TypeError):
        evaluate_release_work(c, c.birth.initial_memory,
            c.cohesive.seed.initial_maximum_separation_m, np.zeros(c.partition.ndof),
            c.partition.time_myr+1e-6, **{field: 1e30})


def test_initial_cohesive_history_cannot_be_invented(tensor):
    c, *_ = tensor
    false_history = np.array(c.cohesive.seed.initial_maximum_separation_m)+c.cohesive.seed.law.peak_separation_m
    with pytest.raises(VolumeReleaseWorkRefusal, match="cohesive history"):
        evaluate_release_work(c, c.birth.initial_memory, false_history, np.zeros(c.partition.ndof),
            c.partition.time_myr+1e-6)


@pytest.mark.parametrize("field", ["parent_checkpoint_sha256", "formation_checkpoint_sha256",
                                  "source_identity", "time_myr", "accepted_step_sizes_myr"])
def test_resealed_proposal_cannot_replace_source_history(tensor, field):
    c, _, _, proposal, law = tensor
    raw = json.loads(proposal.payload_json)
    raw[field] = "substitute" if field not in ("time_myr", "accepted_step_sizes_myr") else 3.
    raw["nucleation_sha256"] = hashlib.sha256(_canonical({k: v for k, v in raw.items()
        if k != "nucleation_sha256"})).hexdigest()
    payload = _canonical(raw).decode()
    forged = VolumeNucleationProposal(payload, hashlib.sha256(payload.encode()).hexdigest())
    with pytest.raises(VolumeReleaseWorkRefusal, match="proposal"):
        prepare_volume_release_work(c.birth, c.finite_seed, law, source_nucleation_proposal=forged)


@pytest.mark.parametrize("field", ["accepted_fault_history", "prepared_fault_history",
    "newly_activating_faces", "source_fault_parameters", "parent_checkpoint_sha256"])
def test_nested_resealed_onset_cannot_change_calculated_source(tensor, field):
    c, _, _, proposal, law = tensor
    raw = json.loads(proposal.payload_json)
    onset = raw["source_onset"]
    if field == "accepted_fault_history": onset[field]["damage"][0] += .1
    elif field == "prepared_fault_history": onset[field]["damage"][0] += .1
    elif field == "newly_activating_faces": onset[field] = [0]; raw[field] = [0]
    elif field == "source_fault_parameters": onset[field]["activation_damage"] -= .1
    else: onset[field] = "0"*64
    onset["onset_sha256"] = hashlib.sha256(_canonical({k: v for k, v in onset.items()
        if k != "onset_sha256"})).hexdigest()
    raw["nucleation_sha256"] = hashlib.sha256(_canonical({k: v for k, v in raw.items()
        if k != "nucleation_sha256"})).hexdigest()
    payload = _canonical(raw).decode()
    forged = VolumeNucleationProposal(payload, hashlib.sha256(payload.encode()).hexdigest())
    with pytest.raises(VolumeReleaseWorkRefusal, match="history|activation flags|parameters|source context"):
        prepare_volume_release_work(c.birth, c.finite_seed, law, source_nucleation_proposal=forged)


def test_context_and_arrays_are_immutable_and_live_mutations_are_guarded(tensor):
    c, *_ = tensor
    with pytest.raises(FrozenInstanceError): c.payload_json = "replacement"
    with pytest.raises(ValueError): c.initial_velocity_m_s.setflags(write=True)
    changed = replace(c, initial_velocity_m_s=c.initial_velocity_m_s+1.)
    with pytest.raises(VolumeReleaseWorkRefusal, match="context changed"):
        validate_volume_release_work(changed)
    f = np.array(c.partition.external_force_n); f[0] += max(1., abs(f[0]))
    changed_p = replace(c.partition, external_force_n=f)
    changed_reaction = replace(c.birth.reaction, partition=changed_p)
    changed_birth = replace(c.birth, reaction=changed_reaction)
    changed = replace(c, birth=changed_birth)
    with pytest.raises(VolumeReleaseWorkRefusal, match="context changed"):
        validate_volume_release_work(changed)


@pytest.mark.parametrize("name,value", [("after_time_myr", True), ("after_time_myr", float("nan")),
    ("momentum_relative_tolerance", False), ("work_relative_tolerance", float("inf"))])
def test_non_numeric_or_non_finite_path_inputs_are_refused(tensor, name, value):
    c, *_ = tensor
    args = dict(after_time_myr=c.partition.time_myr+1e-6)
    args[name] = value
    with pytest.raises(ValueError):
        evaluate_release_work(c, c.birth.initial_memory,
            c.cohesive.seed.initial_maximum_separation_m, np.zeros(c.partition.ndof), **args)


def test_context_packet_replay_requires_independent_trusted_source(tensor):
    c, _, _, proposal, law = tensor
    raw = volume_release_work_to_dict(c)
    restored = volume_release_work_from_dict(raw, c.birth, c.finite_seed, law,
        source_nucleation_proposal=proposal)
    assert volume_release_work_to_dict(restored) == raw
    bad = copy.deepcopy(raw)
    bad["historical_dissipation_available_for_creation_j"] = 4096.
    bad["release_work_context_sha256"] = hashlib.sha256(_canonical({k: v for k, v in bad.items()
        if k != "release_work_context_sha256"})).hexdigest()
    with pytest.raises(VolumeReleaseWorkRefusal, match="trusted source"):
        volume_release_work_from_dict(bad, c.birth, c.finite_seed, law,
            source_nucleation_proposal=proposal)


def test_correctly_resealed_response_cannot_erase_small_cross_work(added_path, tensor):
    c, *_ = tensor
    bad = volume_release_work_response_to_dict(added_path)
    assert bad["bulk_work_decomposition_j"][1] != 0.
    bad["bulk_work_decomposition_j"][1] = 0.
    bad["release_work_response_sha256"] = hashlib.sha256(_canonical({k: v for k, v in bad.items()
        if k != "release_work_response_sha256"})).hexdigest()
    with pytest.raises(VolumeReleaseWorkRefusal, match="trajectory replay"):
        replay_volume_release_work_response(c, bad, c.birth.initial_memory,
            c.cohesive.seed.initial_maximum_separation_m)


def test_private_nonzero_memory_continuation_preserves_velocity_journal(added_path, tensor):
    c, *_ = tensor
    first = volume_release_work_response_to_dict(added_path)
    second = evaluate_release_work(c, added_path.after_memory,
        added_path.cohesive_response.new_maximum_separation_m, np.zeros(c.partition.ndof),
        added_path.after_memory.time_myr+1e-6, before_velocity_m_s=first["after_velocity_m_s"])
    raw = volume_release_work_response_to_dict(second)
    np.testing.assert_array_equal(raw["before_velocity_m_s"], first["after_velocity_m_s"])
    assert second.after_memory.private_steps == 2
    assert second.after_memory.birth_parent_memory_sha256 == c.birth.initial_memory.birth_parent_memory_sha256
    assert not second.physical_release_admitted and not second.funding_admitted


@pytest.fixture(scope="module")
def native_path(tensor):
    c, *_ = tensor
    born = make_native_release_birth(c)
    d = np.r_[np.zeros(c.partition.parent_ndof), np.sin(np.arange(c.modes.rank)+.2)*2e-7]
    result = evaluate_native_release_work(c, c.birth.initial_memory, born,
        c.operative_cohesive.seed.initial_maximum_separation_m, d, c.partition.time_myr+1e-6)
    return born, d, result


def test_native_factory_inherits_parent_without_reinitializing_its_history(tensor, native_path):
    c, *_ = tensor
    born, _, _ = native_path
    raw = native_release_state_to_dict(born)
    assert born.full18_memory_sha256 == c.birth.initial_memory.binding_sha256
    assert raw['source_memory_birth_binding_sha256'] == c.birth.binding_sha256
    for a in (born.displacement_w_m, born.coefficients_w_m, born.velocity_w_m_s):
        np.testing.assert_array_equal(a, np.zeros(c.modes.rank))
    assert not born.journal


def test_native_path_preserves_source_archive_and_its_own_memory(native_path, tensor):
    c, *_ = tensor
    born, d, value = native_path
    raw = volume_release_work_response_to_dict(value)
    assert raw['native_coordinate_history_used']
    assert not raw['full18_roundtrip_used_for_operative_coordinates']
    assert raw['source_point_expansion_is_native_admission_requirement'] is False
    assert raw['native_active_coordinate_count'] == c.modes.rank
    assert raw['native_tied_kernel_coordinate_count'] == 0
    assert raw['legacy_full18_kernel_witness_scope'] == 'rounded_full18_cumulative_source_journal_arithmetic_witness_only_not_native_admission'
    assert set(raw['legacy_full18_kernel_witnesses']) == {
        'old_displacement','new_displacement','displacement_increment','before_velocity',
        'after_velocity','old_elastic_memory','new_elastic_memory','mean_elastic_memory'}
    assert raw['cohesive_force_work_duality_passed'] and raw['operative_endpoint_addition_passed']
    np.testing.assert_array_equal(value.after_native.displacement_w_m, d[c.partition.parent_ndof:])
    np.testing.assert_array_equal(value.after_memory.journal[0].parent_increment_m, d[:c.partition.parent_ndof])
    restarted = restart_native_release_state(c, native_release_state_to_dict(value.after_native), value.after_memory)
    assert restarted.binding_sha256 == value.after_native.binding_sha256
    assert not value.physical_release_admitted and not value.funding_admitted


def test_cancelling_source_lifts_leave_a_reported_kernel_shadow_without_native_kernel_axes(tensor):
    """Pure coefficient arithmetic, not an accepted material trajectory."""
    c, *_ = tensor
    n = c.partition.parent_ndof
    first = np.r_[np.zeros(n), np.sin(np.arange(c.modes.rank)+.2)*1e8]
    second = -first
    second[n:] += np.cos(np.arange(c.modes.rank)+.3)*1e-6
    lift1, lift2 = (expand_native_release_increment(c, d)[n:] for d in (first, second))
    # Each actual source expansion satisfies the original tied increment
    # guard.  Float64 accumulation retains an independently observable
    # kernel component after the large native coordinates nearly cancel.
    validate_release_increment(c.modes, lift1)
    validate_release_increment(c.modes, lift2)
    shadow = lift1+lift2
    original = shadow.copy()
    with pytest.raises(ValueError, match='tied complete jump-kernel'):
        validate_release_increment(c.modes, shadow)
    reports, passed = _full18_kernel_witnesses(c, dict(new_displacement=shadow))
    report = reports['new_displacement']
    assert not passed and not report['literal18_kernel_tolerance_passed']
    assert report['relative_null_component'] > 1e-6
    np.testing.assert_array_equal(shadow, original)
    native = first[n:]+second[n:]
    assert native.shape == (c.modes.rank,)
    assert np.linalg.norm(native) < 1e-5
    assert np.linalg.norm(c.modes.tied_null_modes.T@c.observable_force_basis@native) < 1e-20


def test_native_linked_step_replays_native_velocity_and_cohesive_history(native_path, tensor):
    c, *_ = tensor
    _, d, first = native_path
    raw_first = volume_release_work_response_to_dict(first)
    second = evaluate_native_release_work(c, first.after_memory, first.after_native,
        first.cohesive_response.new_maximum_separation_m, -.25*d, first.after_memory.time_myr+1e-6,
        before_velocity_m_s=raw_first['operative_after_velocity_m_s'])
    raw = volume_release_work_response_to_dict(second)
    np.testing.assert_array_equal(raw['operative_before_velocity_m_s'], raw_first['operative_after_velocity_m_s'])
    np.testing.assert_array_equal(second.after_native.displacement_w_m, .75*d[c.partition.parent_ndof:])
    assert len(second.after_native.journal) == second.after_memory.private_steps == 2
    assert raw['cohesive_force_work_duality_passed'] and raw['cohesive_history_replayed_from_source_birth']
    assert not raw['accepted_planet_memory_changed']


def test_native_work_receipt_replays_from_both_trusted_source_histories(native_path, tensor):
    c, *_ = tensor
    born, _, first = native_path
    raw = volume_release_work_response_to_dict(first)
    replayed = replay_native_release_work_response(c, raw, c.birth.initial_memory, born,
        c.operative_cohesive.seed.initial_maximum_separation_m)
    assert volume_release_work_response_to_dict(replayed) == raw
    forged = copy.deepcopy(raw)
    forged['historical_dissipation_available_for_creation_j'] = 4096.
    with pytest.raises(VolumeReleaseWorkRefusal, match='source-born journal replay'):
        replay_native_release_work_response(c, forged, c.birth.initial_memory, born,
            c.operative_cohesive.seed.initial_maximum_separation_m)


def test_native_inputs_cannot_supply_wrong_source_velocity_or_old_cohesive_history(native_path, tensor):
    c, *_ = tensor
    born, d, first = native_path
    v = volume_release_work_response_to_dict(first)['operative_after_velocity_m_s']
    with pytest.raises(VolumeReleaseWorkRefusal, match='Native before velocity'):
        evaluate_native_release_work(c, first.after_memory, first.after_native,
            first.cohesive_response.new_maximum_separation_m, -.25*d, first.after_memory.time_myr+1e-6,
            before_velocity_m_s=np.asarray(v)+1.)
    with pytest.raises(VolumeReleaseWorkRefusal, match='Old cohesive history'):
        evaluate_native_release_work(c, first.after_memory, first.after_native,
            first.cohesive_response.new_maximum_separation_m+1e-6, -.25*d, first.after_memory.time_myr+1e-6)
