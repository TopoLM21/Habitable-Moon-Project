"""A true formation continuation supplies the compact tied average reaction."""
from dataclasses import FrozenInstanceError, replace
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tectonics.genesis_volume_checkpoint import _canonical
from tectonics.genesis_volume_dynamics import coefficient_arrays
from tectonics.genesis_volume_embedded_partition import (
    _digest, _partition_body, prepare_embedded_partition)
from tectonics.genesis_volume_finite_seed import prepare_volume_finite_seed
from tectonics.genesis_volume_operator import ANGULAR_BARYCENTRIC, evaluate_genesis_volume_operator
from tectonics.genesis_volume_seed_reaction import (
    GenesisVolumeSeedReaction, VolumeSeedReactionRefusal,
    prepare_volume_seed_reaction, validate_volume_seed_reaction,
    volume_seed_reaction_to_dict, volume_seed_reaction_from_dict)
from tectonics.genesis_volume_slip_checkpoint import volume_slip_checkpoint_to_dict
from tectonics.genesis_volume_slip_dynamics import VolumeSlipModel


@pytest.fixture(scope="module")
def actual():
    path = Path(__file__).resolve().parents[1]/(
        "analysis/genesis_cohort_mechanics_validation/evidence/control/material_export.json")
    original = path.read_bytes()
    model = VolumeSlipModel(json.loads(original))
    initial = model.initial_state()
    state = model.step(initial, 1e-6)
    before = _canonical(volume_slip_checkpoint_to_dict(model, state))
    parent, basal = model._assembly(state.geometry, state.fault_history.damage)
    c = state.geometry.cohorts[0]
    dofs = parent.dofs[0]
    X = np.asarray(c.top_triangle_m)
    pose = state.displacement_m[dofs].reshape(3, 6)
    N, lam = np.full(3, 1/3.), .5*(c.inner_scale+c.outer_scale)
    center = N@(lam*(X+pose[:, 3:])+(pose[:, :3]-pose[:, 3:]))
    normal = np.array((1., 2., 3.)); normal /= np.linalg.norm(normal)
    radius = .12*(c.outer_scale-c.inner_scale)*np.linalg.norm(N@(X+pose[:, 3:]))
    seed = prepare_volume_finite_seed(state.geometry, state.displacement_m,
        normal, float(normal@center), center, radius, cohort_id=c.id, quadrature_order=4)
    e0, e1 = coefficient_arrays(state.mechanics)
    partition = prepare_embedded_partition(parent, seed, e0, e1, basal=basal, quadrature_order=4)
    reaction = prepare_volume_seed_reaction(model, state, seed,
        partition=partition, quadrature_order=4)
    yield model, initial, state, parent, basal, seed, partition, reaction
    assert path.read_bytes() == original
    assert _canonical(volume_slip_checkpoint_to_dict(model, state)) == before


def test_initial_state_has_no_accepted_interval_reaction(actual):
    model, initial, _, _, _, seed, *_ = actual
    with pytest.raises(VolumeSeedReactionRefusal, match="actual accepted interval"):
        prepare_volume_seed_reaction(model, initial, seed, quadrature_order=4)


def test_full_saved_average_and_history_binding(actual):
    model, _, state, _, _, seed, partition, reaction = actual
    raw = volume_seed_reaction_to_dict(reaction)
    checkpoint = volume_slip_checkpoint_to_dict(model, state)
    assert raw["accepted_parent_checkpoint_sha256"] == checkpoint["checkpoint_sha256"]
    for field, runtime in (("accepted_final_report_sha256", "last_report"),
        ("full_memory_sha256", "mechanics"), ("birth_history_sha256", "history"),
        ("fault_history_sha256", "fault_history"), ("endpoint_geometry_sha256", "geometry")):
        assert raw[field] == hashlib.sha256(_canonical(checkpoint["runtime_state"][runtime])).hexdigest()
    assert raw["finite_seed_binding_sha256"] == seed.binding_sha256
    assert raw["embedded_partition_binding_sha256"] == partition.binding_sha256
    assert raw["historical_seed_claimed"] is False
    assert raw["surface_traction_replacement_admitted"] is False
    assert raw["physical_creation_admitted"] is raw["physical_release_admitted"] is False
    assert raw["funding_admitted"] is False
    assert raw["new_accepted_intervals"] == raw["new_material_births"] == raw["released_physical_dofs"] == 0
    assert raw["transition_time_increment_myr"] == raw["creation_work_debited_from_Genesis_j"] == 0
    np.testing.assert_array_equal(reaction.interval_average_elastic_constant,
        state.last_report["weak_plane_slip"]["average_strain0"])
    np.testing.assert_array_equal(reaction.interval_average_elastic_inverse_scale,
        state.last_report["weak_plane_slip"]["average_strain1"])


def test_average_internal_force_matches_frozen_regular_bulk_virtual_work(actual):
    reaction, p = actual[-1], actual[-2]
    local, ci = p.local, p.selected_cohort_index
    angular = local.barycentric@np.linalg.inv(ANGULAR_BARYCENTRIC)
    mean = (angular@reaction.interval_average_elastic_constant[ci]
        +(angular@reaction.interval_average_elastic_inverse_scale[ci])/local.radial_scale[:, None])
    stress = mean@local.elasticity_pa.T
    # Independent scalar contraction and both original mode families remain visible.
    for columns in (np.array((0,1,2,6,7,8,12,13,14)), np.array((3,4,5,9,10,11,15,16,17))):
        probe = np.zeros(18); probe[columns] = 1e-6*np.sin(.3+np.arange(9))
        work = np.sum(local.reference_volume_weights_m3*np.sum(
            stress*np.einsum("pai,i->pa", local.enrichment_strain_operator, probe), axis=1), dtype=np.longdouble)
        assert reaction.added_average_internal_force_n@probe == pytest.approx(float(work), rel=2e-10, abs=1e-5)


def test_average_is_not_endpoint_static_reaction_or_extra_loading_increment(actual):
    reaction, p = actual[-1], actual[-2]
    assert np.linalg.norm(reaction.added_average_internal_force_n-p.local.enrichment_force_n) > 1e-8*np.linalg.norm(p.local.enrichment_force_n)
    doubled = reaction.added_average_internal_force_n+p.local.stiffness_cross.T@reaction.parent_increment_m[p.selected_parent_dofs]
    assert np.linalg.norm(doubled-reaction.added_average_internal_force_n) > 1e-8*np.linalg.norm(reaction.added_average_internal_force_n)
    np.testing.assert_array_equal(reaction.endpoint_static_constraint_reaction_n, -p.local.enrichment_force_n)


def test_inertia_is_nonzero_and_drag_and_external_added_modes_are_exact_zero(actual):
    reaction, p = actual[-1], actual[-2]
    n, dt = p.parent_ndof, reaction.dt_s
    assert np.linalg.norm(reaction.added_inertia_force_n) > 0
    expected = 2*(p.M[n:, :n]@(reaction.parent_increment_m-dt*reaction.parent_before_velocity_m_s))/(dt*dt)
    np.testing.assert_array_equal(reaction.added_inertia_force_n, expected)
    np.testing.assert_array_equal(reaction.added_drag_force_n, np.zeros(18))
    np.testing.assert_array_equal(reaction.added_external_force_n, np.zeros(18))
    np.testing.assert_array_equal(reaction.added_residual_n,
        reaction.added_inertia_force_n+reaction.added_average_internal_force_n)
    np.testing.assert_array_equal(reaction.constraint_multipliers_n, -reaction.added_residual_n)


def test_parent_residual_preserved_exactly_and_added_reaction_has_zero_tied_work(actual):
    _, _, _, parent, basal, _, p, reaction = actual
    dt, inc, v0 = reaction.dt_s, reaction.parent_increment_m, reaction.parent_before_velocity_m_s
    mean = evaluate_genesis_volume_operator(parent, np.zeros(parent.ndof),
        elastic_constant=reaction.interval_average_elastic_constant,
        elastic_inverse_scale=reaction.interval_average_elastic_inverse_scale).force
    expected = 2*(basal.mass_matrix@(inc-dt*v0))/(dt*dt)+basal.basal_drag_matrix@inc/dt+mean-basal.external_force_n
    np.testing.assert_array_equal(reaction.parent_residual_n, expected)
    assert np.linalg.norm(expected) > 0
    np.testing.assert_array_equal(p.R.T@reaction.remaining_residual_n, expected)
    np.testing.assert_array_equal(p.J@reaction.remaining_residual_n, np.zeros(18))
    np.testing.assert_array_equal(p.R.T@reaction.constraint_force_n, np.zeros(parent.ndof))
    assert reaction.constraint_force_n@(p.R@inc) == 0
    assert reaction.constraint_multipliers_n@(p.J@(p.R@inc)) == 0
    for field in (reaction.enrichment_increment_m, reaction.enrichment_before_velocity_m_s,
                  reaction.enrichment_elastic_memory_m):
        np.testing.assert_array_equal(field, np.zeros(18))


def test_source_bound_descriptor_roundtrip(actual):
    model, _, state, _, _, seed, p, reaction = actual
    restored = volume_seed_reaction_from_dict(volume_seed_reaction_to_dict(reaction),
        model=model, state=state, finite_seed=seed, partition=p, quadrature_order=4)
    assert restored.binding_sha256 == reaction.binding_sha256


@pytest.mark.parametrize("field,value", [
    ("added_inertia_force_n", [0.]*18), ("added_average_internal_force_n", [0.]*18),
    ("constraint_multipliers_n", [0.]*18), ("accepted_final_report_sha256", "0"*64),
    ("physical_release_admitted", True), ("historical_seed_claimed", True),
    ("creation_work_debited_from_Genesis_j", 1.), ("dt_s", 1.),
])
def test_resealed_descriptor_cannot_change_history_or_mechanical_residual(actual, field, value):
    model, _, state, _, _, seed, p, reaction = actual
    raw = volume_seed_reaction_to_dict(reaction)
    assert raw[field] != value
    raw[field] = value
    raw["seed_reaction_sha256"] = hashlib.sha256(_canonical({k: v for k, v in raw.items() if k != "seed_reaction_sha256"})).hexdigest()
    with pytest.raises(VolumeSeedReactionRefusal, match="complete accepted-source recomputation"):
        volume_seed_reaction_from_dict(raw, model=model, state=state,
            finite_seed=seed, partition=p, quadrature_order=4)


@pytest.mark.parametrize("change", ["missing_mean", "duration", "before_clock", "requested_duration", "increment", "before_velocity", "initial_displacement", "average"])
def test_missing_or_changed_saved_interval_is_refused(actual, change):
    model, _, state, _, _, seed, p, _ = actual
    report = copy.deepcopy(state.last_report)
    if change == "missing_mean":
        del report["weak_plane_slip"]["average_strain0"]
    elif change == "duration":
        report["dt_myr"] *= 2
    elif change == "before_clock":
        report["before_time_myr"] += 1e-5
    elif change == "requested_duration":
        report["requested_dt_myr"] *= 2
    elif change == "increment":
        report["audit_input"]["increment_m"][0] += 1.
    elif change == "before_velocity":
        report["audit_input"]["before_velocity_m_s"][0] += 1.
    elif change == "initial_displacement":
        report["audit_input"]["before_displacement_m"][0] += 1.
    elif change == "average":
        report["weak_plane_slip"]["average_strain0"][0][0][0] += 1.
    with pytest.raises(VolumeSeedReactionRefusal):
        prepare_volume_seed_reaction(model, replace(state, last_report=report), seed,
            partition=p, quadrature_order=4)


@pytest.mark.parametrize("change", ["memory0", "selected_memory1", "mass", "drag", "external", "radius", "center"])
def test_resealed_typed_partition_cannot_change_source_chart_memory_or_old_operators(actual, change):
    model, _, state, _, _, seed, p, _ = actual
    if change == "memory0":
        values = np.array(p.memory0); values[-1, 0, 0] += 1.
        changed = replace(p, memory0=values)
    elif change == "selected_memory1":
        values = np.array(p.local.memory1); values[0, 0] += 1.
        changed = replace(p, local=replace(p.local, memory1=values))
    elif change in ("mass", "drag"):
        original = p.M if change == "mass" else p.D
        matrix = original.copy(); matrix.data[0] *= 2.
        assert not np.array_equal(matrix.data, original.data)
        changed = replace(p, **{"mass_matrix" if change == "mass" else "basal_drag_matrix": matrix})
    elif change == "external":
        values = np.array(p.external_force_n); values[0] += 1.
        changed = replace(p, external_force_n=values)
    elif change == "radius":
        changed = replace(p, local=replace(p.local, seed_radius_m=2*p.local.seed_radius_m))
    else:
        changed = replace(p, local=replace(p.local, seed_center_world_m=p.local.seed_center_world_m+1.))
    changed = replace(changed, binding_sha256=_digest(_partition_body(changed)))
    with pytest.raises(VolumeSeedReactionRefusal):
        prepare_volume_seed_reaction(model, state, seed, partition=changed, quadrature_order=4)


def test_reaction_arrays_and_wrapper_are_immutable(actual):
    reaction = actual[-1]
    assert type(reaction) is GenesisVolumeSeedReaction
    with pytest.raises(ValueError):
        reaction.added_residual_n[0] = 0
    with pytest.raises(FrozenInstanceError):
        reaction.dt_s = 1.
    with pytest.raises(VolumeSeedReactionRefusal, match="immutable accepted interval binding"):
        validate_volume_seed_reaction(replace(reaction, binding_sha256="0"*64))


@pytest.fixture(scope="module")
def source_birth(actual):
    from tectonics.genesis_volume_seed_memory import prepare_source_seed_memory
    model, _, state, _, _, seed, p, reaction = actual
    return prepare_source_seed_memory(model, state, seed, partition=p,
        reaction=reaction, quadrature_order=4)


def test_source_memory_birth_keeps_complete_Genesis_mechanics_and_only_new_coefficients_zero(actual, source_birth):
    from tectonics.genesis_volume_loading import volume_mechanics_to_dict
    from tectonics.genesis_volume_seed_memory import source_seed_memory_to_dict
    state = actual[2]
    initial = source_birth.initial_memory
    assert volume_mechanics_to_dict(initial.parent_state) == volume_mechanics_to_dict(state.mechanics)
    for field in ("initial_energy_j", "elastic_energy_j", "mechanical_work_j", "viscous_dissipation_j",
                  "thermal_work_j", "damage_energy_change_j", "remelt_energy_removed_j", "balance_residual_j"):
        assert getattr(initial, field) == getattr(state.mechanics, field)
    assert initial.journal == () and initial.birth_time_myr == state.time_myr
    np.testing.assert_array_equal(initial.displacement_m, np.zeros(18))
    np.testing.assert_array_equal(initial.coefficients_m, np.zeros(18))
    raw = source_seed_memory_to_dict(source_birth)
    assert raw["accepted_planet_memory_changed"] is False
    assert raw["saved_parent_elastic_energy_j"] == state.mechanics.elastic_energy_j
    assert raw["inherited_energy_reduction_difference_j"] == source_birth.partition.initial_elastic_energy_j-state.mechanics.elastic_energy_j
    assert raw["new_Genesis_time_myr"] == raw["creation_work_debited_from_Genesis_j"] == 0.


def test_source_loading_uses_literal_Maxwell_parameters_and_fixed_original_material(actual, source_birth):
    from tectonics.genesis import SECONDS_PER_MYR
    from tectonics.genesis_volume_dynamics import radial_moments
    from tectonics.genesis_volume_seed_memory import source_seed_memory_loading
    model = actual[0]
    s = source_birth.initial_memory
    after = s.time_myr+1e-6
    loading = source_seed_memory_loading(source_birth, s, after)
    temp = np.asarray([c.temperature_k for c in s.parent_state.cohorts])
    parameters = model.shell_parameters
    eta = np.clip(parameters.viscosity_reference_pa_s*np.exp(np.clip(
        parameters.activation_energy_j_mol/8.314462618*(1/temp-1/parameters.viscosity_reference_temperature_k), -60, 60)),
        parameters.viscosity_min_pa_s, parameters.viscosity_max_pa_s)
    expected = (after-s.time_myr)*SECONDS_PER_MYR*parameters.young_modulus_pa/eta
    np.testing.assert_array_equal(loading.ratio, expected)
    np.testing.assert_array_equal(loading.thermal0, np.zeros_like(loading.thermal0))
    np.testing.assert_array_equal(loading.temperatures_k, temp)
    np.testing.assert_array_equal(loading.elasticity_pa, source_birth.partition.parent_assembly.C)
    assert loading.radial_moments == radial_moments(source_birth.partition.parent_assembly)
    assert loading.damage_energy_change_j == loading.remelt_energy_removed_j == 0.


def _private_source_step(birth, before, phase):
    from tectonics.genesis_volume_enriched_memory import (
        prepare_enriched_loading, evaluate_enriched_loading, accept_enriched_loading)
    from tectonics.genesis_volume_seed_memory import source_seed_memory_loading
    p = birth.partition
    loading = source_seed_memory_loading(birth, before, before.time_myr+1e-6)
    trial = prepare_enriched_loading(p, before, loading)
    q = 1e-8*np.sin(phase+.17*np.arange(p.parent_ndof))
    a = 1e-5*np.cos(phase+.31*np.arange(18))
    response = evaluate_enriched_loading(p, trial, q, a)
    return accept_enriched_loading(p, trial, response)


@pytest.fixture(scope="module")
def source_private_two_steps(source_birth):
    first = _private_source_step(source_birth, source_birth.initial_memory, .2)
    return _private_source_step(source_birth, first, .7)


def test_source_private_nonzero_memory_restart_and_later_continuation_are_exact(source_birth, source_private_two_steps):
    from tectonics.genesis_volume_enriched_memory import enriched_memory_to_dict
    from tectonics.genesis_volume_seed_memory import restart_source_seed_memory
    state = source_private_two_steps
    assert len(state.journal) == 2
    assert np.linalg.norm(state.coefficients_m) > 0
    assert not np.array_equal(state.coefficients_m, state.displacement_m)
    raw = enriched_memory_to_dict(state)
    restored = restart_source_seed_memory(source_birth, raw)
    assert enriched_memory_to_dict(restored) == raw
    direct = _private_source_step(source_birth, state, 1.3)
    resumed = _private_source_step(source_birth, restored, 1.3)
    assert enriched_memory_to_dict(direct) == enriched_memory_to_dict(resumed)


@pytest.mark.parametrize("change", ["constitutive_parameters", "accepted_birth_clock"])
def test_resealed_source_birth_cannot_change_Genesis_parameters_or_clock(actual, source_birth, change):
    from tectonics.genesis_volume_seed_memory import (
        SourceSeedMemoryRefusal, source_seed_memory_to_dict, source_seed_memory_from_dict)
    model, _, state, _, _, seed, p, reaction = actual
    raw = source_seed_memory_to_dict(source_birth)
    if change == "constitutive_parameters":
        raw["shell_parameters"]["viscosity_reference_pa_s"] *= 2.
    else:
        raw["accepted_time_myr"] += 1e-6
    raw["source_memory_birth_sha256"] = hashlib.sha256(_canonical({k: v for k, v in raw.items()
        if k != "source_memory_birth_sha256"})).hexdigest()
    with pytest.raises(SourceSeedMemoryRefusal, match="accepted Genesis reconstruction"):
        source_seed_memory_from_dict(raw, model=model, state=state, finite_seed=seed,
            partition=p, reaction=reaction, quadrature_order=4)


def test_resealed_private_final_memory_cannot_replace_replayed_source_history(source_birth, source_private_two_steps):
    from tectonics.genesis_volume_enriched_memory import EnrichedMemoryRefusal, enriched_memory_to_dict
    from tectonics.genesis_volume_seed_memory import restart_source_seed_memory
    raw = enriched_memory_to_dict(source_private_two_steps)
    raw["state"]["coefficients_m"][0] += 1e-5
    raw["packet_sha256"] = hashlib.sha256(_canonical({k: v for k, v in raw.items() if k != "packet_sha256"})).hexdigest()
    with pytest.raises(EnrichedMemoryRefusal, match="birth-journal replay"):
        restart_source_seed_memory(source_birth, raw)


def test_resealed_typed_source_factory_cannot_change_authentic_viscosity(source_birth):
    from tectonics.genesis_volume_seed_memory import (
        SourceSeedMemoryRefusal, source_seed_memory_to_dict, source_seed_memory_loading)
    raw = source_seed_memory_to_dict(source_birth)
    raw["shell_parameters"]["viscosity_reference_pa_s"] *= 2.
    raw["source_memory_birth_sha256"] = hashlib.sha256(_canonical({k: v for k, v in raw.items()
        if k != "source_memory_birth_sha256"})).hexdigest()
    payload = _canonical(raw).decode()
    forged = replace(source_birth, payload_json=payload,
        binding_sha256=hashlib.sha256(payload.encode()).hexdigest())
    with pytest.raises(SourceSeedMemoryRefusal, match="authentic source"):
        source_seed_memory_loading(forged, forged.initial_memory, forged.initial_memory.time_myr+1e-6)


def test_resealed_typed_private_state_cannot_change_fixed_birth_temperature(source_birth, source_private_two_steps):
    from dataclasses import fields
    from tectonics.genesis_volume_enriched_memory import _new_state, validate_enriched_memory
    from tectonics.genesis_volume_seed_memory import SourceSeedMemoryRefusal, source_seed_memory_loading
    original = source_private_two_steps
    records = list(original.parent_state.cohorts)
    records[0] = replace(records[0], temperature_k=records[0].temperature_k+25.)
    values = {f.name: getattr(original, f.name) for f in fields(original)
              if f.name != "binding_sha256"}
    values["parent_state"] = replace(original.parent_state, cohorts=tuple(records))
    forged = _new_state(**values)
    # A valid self-seal and unchanged elastic field do not authorize a new
    # temperature history or a different Arrhenius viscosity in this branch.
    validate_enriched_memory(source_birth.partition, forged)
    assert forged.birth_parent_memory_sha256 == original.birth_parent_memory_sha256
    with pytest.raises(SourceSeedMemoryRefusal, match="temperatures differ from the authentic birth"):
        source_seed_memory_loading(source_birth, forged, forged.time_myr+1e-6)
    assert original.parent_state.cohorts[0].temperature_k == source_birth.initial_memory.parent_state.cohorts[0].temperature_k
