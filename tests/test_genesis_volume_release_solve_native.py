"""Native source-born trajectories on an archived tensor branch, not a planet reset."""
from dataclasses import replace
import json

import numpy as np
import pytest

from tectonics.genesis_volume_enriched_memory import enriched_memory_to_dict
from tectonics.genesis_volume_release_history import (
    make_native_release_birth, native_release_state_to_dict, restart_native_release_state,
)
from tectonics.genesis_volume_release_solve import VolumeReleaseSolveRefusal, _canonical, _digest
from tectonics.genesis_volume_release_solve_native import (
    solve_native_release_work, volume_native_release_solve_to_dict,
)
from tectonics.genesis_volume_release_work import (
    volume_release_work_response_to_dict, replay_native_release_work_response,
)


@pytest.fixture(scope="module")
def tensor():
    from test_genesis_volume_release_work import make_archived_tensor_context
    context, *rest = make_archived_tensor_context()
    original = enriched_memory_to_dict(context.birth.initial_memory)
    yield context, *rest
    assert enriched_memory_to_dict(context.birth.initial_memory) == original


@pytest.fixture(scope="module")
def paths(tensor):
    context, *_ = tensor
    before = context.birth.initial_memory
    native = make_native_release_birth(context)
    old = context.cohesive.seed.initial_maximum_separation_m
    first = solve_native_release_work(context, before, native, old, before.time_myr+1e-7)
    second = solve_native_release_work(context, first.response.after_memory, first.response.after_native,
        first.response.cohesive_response.new_maximum_separation_m, first.response.after_memory.time_myr+7e-8)
    return native, first, second


def test_native_source_birth_has_exact_zero_extra_state_and_preserves_original_memory(tensor):
    context, *_ = tensor
    before = enriched_memory_to_dict(context.birth.initial_memory)
    native = make_native_release_birth(context)
    for value in (native.displacement_w_m, native.coefficients_w_m, native.velocity_w_m_s):
        np.testing.assert_array_equal(value, np.zeros(context.modes.rank))
        assert not value.flags.writeable
    assert native.journal == ()
    assert native.time_myr == context.birth.initial_memory.time_myr
    assert native.full18_memory_sha256 == context.birth.initial_memory.binding_sha256
    assert enriched_memory_to_dict(context.birth.initial_memory) == before


def test_two_source_forced_native_paths_balance_without_a_float18_coordinate_roundtrip(paths,tensor):
    context, *_ = tensor
    birth, first, second = paths
    for result in (first,second):
        solve = volume_native_release_solve_to_dict(result)
        raw = volume_release_work_response_to_dict(result.response)
        assert raw["balanced_trajectory"] and raw["projected_momentum_passed"]
        assert raw["work_identity_passed"] and raw["discrete_work_balance_passed"]
        assert raw["bulk_small_component_balance_passed"] and raw["bulk_force_work_duality_passed"]
        assert solve["native_displacement_memory_velocity_authoritative"]
        assert solve["no_full18_roundtrip_used_for_operative_path"]
        assert solve["no_ULP_jitter_or_tied_kernel_freedom_used"]
        assert solve["compact_jump_rank"] == context.modes.rank == 15
        assert solve["tied_kernel_dimension"] == 3
        assert solve["complete_jump_kernel_remains_tied"]
        assert solve["maximum_new_effective_separation_m"] <= solve["peak_separation_m"]
        assert raw["fracture_dissipation_increment_j"] == 0.
        assert raw["physical_release_admitted"] is raw["funding_admitted"] is False
        assert raw["new_accepted_Genesis_time_myr"] == raw["creation_work_debited_from_Genesis_j"] == 0.
    assert len(first.response.after_native.journal) == 1
    assert len(second.response.after_native.journal) == 2
    assert birth.binding_sha256 == volume_native_release_solve_to_dict(first)["before_native_history_binding_sha256"]
    assert first.response.after_native.binding_sha256 == volume_native_release_solve_to_dict(second)["before_native_history_binding_sha256"]


def test_native_history_and_work_response_replay_exactly_from_same_source_birth(paths,tensor):
    context, *_ = tensor
    birth, first, second = paths
    for result in (first,second):
        original=native_release_state_to_dict(result.response.after_native)
        replayed=restart_native_release_state(context,original,result.response.after_memory)
        assert native_release_state_to_dict(replayed)==original
    raw=volume_release_work_response_to_dict(first.response)
    replayed=replay_native_release_work_response(context,raw,context.birth.initial_memory,birth,
        context.cohesive.seed.initial_maximum_separation_m)
    assert volume_release_work_response_to_dict(replayed)==raw


def test_native_solver_refuses_a_velocity_reset_even_when_coordinate_kernel_is_tied(tensor):
    context, *_ = tensor
    native=make_native_release_birth(context)
    changed=np.r_[context.initial_velocity_m_s[:context.partition.parent_ndof],native.velocity_w_m_s]
    changed[-1]=1e-8
    with pytest.raises(VolumeReleaseSolveRefusal,match="trusted source journal"):
        solve_native_release_work(context,context.birth.initial_memory,native,
            context.cohesive.seed.initial_maximum_separation_m,context.partition.time_myr+1e-7,
            before_velocity_m_s=changed)


def test_native_solver_refuses_preexisting_softening_without_a_new_initialization(tensor):
    context, *_ = tensor
    native=make_native_release_birth(context)
    old=np.array(context.cohesive.seed.initial_maximum_separation_m)
    old[0]=np.nextafter(context.cohesive.seed.peak_separation_m,np.inf)
    with pytest.raises(VolumeReleaseSolveRefusal,match="damaged history or softening"):
        solve_native_release_work(context,context.birth.initial_memory,native,old,context.partition.time_myr+1e-7)


def test_resealed_native_solver_cannot_claim_physical_admission(paths):
    import hashlib
    _,first,_=paths
    raw=json.loads(first.payload_json)
    raw['physical_release_admitted']=True
    raw['native_release_solve_sha256']=_digest({k:v for k,v in raw.items() if k!='native_release_solve_sha256'})
    payload=_canonical(raw).decode()
    forged=replace(first,payload_json=payload,binding_sha256=hashlib.sha256(payload.encode()).hexdigest())
    with pytest.raises(VolumeReleaseSolveRefusal,match="physical claim"):
        volume_native_release_solve_to_dict(forged)
