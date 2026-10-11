"""Original 2D material must be archived once through true contact recrossing."""
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import pytest

from tectonics.fractional_surface import EXTENSIVE_FIELDS
from tectonics.geometric_native_donor_sweep import certify_native_donor_patch_sweep
from tectonics.geometric_native_material_history import (
    make_native_material_memory_reference, initialize_native_material_history,
    evaluate_native_material_history, native_material_history_to_dict,
    native_material_history_from_dict, native_material_history_snapshot_to_dict,
    evaluate_native_material_point_history, native_material_point_history_to_dict,
    native_material_point_history_from_dict,
)
from tectonics.geometric_native_pose import FractionInterval as I
from tectonics.geometric_native_surface import native_surface_from_dict

ROOT=Path(__file__).resolve().parents[1]
EVIDENCE=ROOT/"analysis/geometric_native_contact_event_validation/time_support_release_01"


def encoded(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()


@pytest.fixture(scope="module")
def actual():
    prepared=json.loads((EVIDENCE/"initial.json").read_bytes())
    candidate=json.loads((EVIDENCE/"force_candidate.json").read_bytes())
    support=json.loads((EVIDENCE/"constitutive_time_support.json").read_bytes())
    surface=native_surface_from_dict(prepared["current_surface"])
    parent=next(f for f in surface.fragments if f.fragment_id=="octant:6")
    receiver=next(f for f in surface.fragments if f.fragment_id=="octant:2")
    omega=dict(candidate["omega_rad_per_myr"])
    sweep=certify_native_donor_patch_sweep(parent.geometry,receiver.geometry,
        tuple(omega[parent.parcel.plate]),tuple(omega[receiver.parcel.plate]),
        candidate["admission"]["dt_myr"])
    before=prepared["native_underthrust_motion"]["snapshot"]
    after=dict(candidate=deepcopy(candidate),whole_time_support=deepcopy(support),adopted=False)
    reference=make_native_material_memory_reference(before,after,
        constitutive_path_sha256=support["path_sha256"])
    history=initialize_native_material_history(parent,receiver,sweep,
        material_epoch_myr=surface.time_myr,memory_reference=reference)
    return parent,receiver,sweep,reference,history,before,after


def test_true_original_material_moves_to_one_cohort_across_all_three_events(actual):
    parent,receiver,sweep,reference,history,*_=actual
    assert history.parent==parent and history.receiver_parent==receiver
    assert tuple(e.kind for e in history.events)==("entry","exit","entry")
    assert tuple(e.operation for e in history.events)==(
        "live_to_passive_archive_once","contact_exit_keep_archive","contact_reentry_reuse_archive")
    assert {e.cohort_id for e in history.events}=={history.cohort_id}
    snapshots=[evaluate_native_material_history(history,t) for t in sweep.witness_times]
    assert [s.contact_occupancy for s in snapshots]==["outside","inside","outside","inside","inside"]
    assert [s.material_allocation for s in snapshots]==["live","archived","archived","archived","archived"]
    assert all(s.memory_reference==reference and not s.admission_authorized for s in snapshots)
    for old,new in zip(snapshots[1].allocations,snapshots[-1].allocations):
        assert old.archived_amount_bounds==new.archived_amount_bounds
        assert old.live_amount_bounds==new.live_amount_bounds==I(0)
        assert old.conserved_total==new.conserved_total


def test_four_positive_reservoir_shares_keep_exact_native_measure_provenance(actual):
    parent,_,sweep,_,history,*_=actual
    assert tuple(s.reservoir for s in history.shares)==EXTENSIVE_FIELDS
    assert sweep.patch_area_bounds.lo>0
    for share in history.shares:
        assert share.parent_amount==Fraction.from_float(getattr(parent.parcel,share.reservoir))
        assert 0<share.native_area_ratio_bounds.lo<=share.native_area_ratio_bounds.hi<1
        assert share.amount_bounds==share.native_area_ratio_bounds*share.parent_amount
        assert share.amount_bounds.lo>0
        assert share.numerator_geometry_sha256==history.patch_geometry_sha256
        assert "native_area(patch) / native_area(parent)" in share.expression
    assert history.parent.parcel.material_fields==parent.parcel.material_fields
    assert history.parent.parcel.specific_properties==parent.parcel.specific_properties


def test_unknown_fraction_of_first_entry_keeps_conserved_implicit_material_partition(actual):
    history=actual[4]
    entry=history.events[0].time_fraction
    snapshot=evaluate_native_material_history(history,(entry.lo+entry.hi)/2)
    assert snapshot.contact_occupancy=="mixed_entry_window"
    assert snapshot.material_allocation=="implicit_first_entry_partition"
    for allocation in snapshot.allocations:
        assert allocation.live_amount_bounds==allocation.archived_amount_bounds==I(0,allocation.conserved_total.amount_bounds.hi)
        assert "first_entry(q) > query_time" in allocation.live_expression
        assert "first_entry(q) <= query_time" in allocation.archived_expression
        assert allocation.partition_identity=="live_subset disjoint_union archived_subset = original_patch"


@pytest.mark.parametrize("event_index",[1,2])
def test_contact_departure_or_return_does_not_duplicate_archived_reservoirs(actual,event_index):
    history=actual[4]
    event=history.events[event_index]
    snapshot=evaluate_native_material_history(history,(event.time_fraction.lo+event.time_fraction.hi)/2)
    assert snapshot.material_allocation=="archived"
    expected=evaluate_native_material_history(history,1)
    assert snapshot.allocations==expected.allocations
    raw=native_material_history_snapshot_to_dict(snapshot)
    assert raw["archive_cohort_count_bounds"]==[1,1]
    assert raw["contact_exit_or_reentry_creates_new_cohort"] is False


def test_true_event_age_increment_survives_below_binary64_parent_age_ulp(actual):
    parent,_,sweep,_,history,*_=actual
    time=sweep.witness_times[2]
    snapshot=evaluate_native_material_history(history,time)
    increment=time*Fraction.from_float(sweep.dt_myr)
    assert 0<increment<Fraction(1,10**17)
    assert snapshot.material_age_myr==Fraction.from_float(parent.parcel.age_myr)+increment
    assert snapshot.material_age_myr>Fraction.from_float(parent.parcel.age_myr)
    assert float(snapshot.material_age_myr)==parent.parcel.age_myr


def test_material_checkpoint_full_replay_and_exact_time_queries_are_byte_identical(actual):
    _,_,sweep,reference,history,*_=actual
    raw=native_material_history_to_dict(history)
    loaded=native_material_history_from_dict(deepcopy(raw),expected_memory_reference=reference)
    assert loaded==history
    assert encoded(raw)==encoded(native_material_history_to_dict(loaded))
    for t in sweep.witness_times:
        a=native_material_history_snapshot_to_dict(evaluate_native_material_history(history,t))
        b=native_material_history_snapshot_to_dict(evaluate_native_material_history(loaded,t))
        assert encoded(a)==encoded(b)
    assert not raw["admission_authorized"] and not raw["global_inventory_updated"]
    assert all(p["conservation_identity"]=="patch_share + unmodelled_parent_complement = parent_amount"
               for p in raw["parent_reservoir_partitions"])


def test_mechanical_evidence_is_frozen_without_adopting_or_mutating_candidate(actual):
    _,_,_,reference,history,before,after=actual
    old_before,old_after=encoded(before),encoded(after)
    for t in [0,history.events[1].time_fraction.hi,1]:
        snapshot=evaluate_native_material_history(history,t)
        assert snapshot.memory_reference.before_payload_json==old_before
        assert snapshot.memory_reference.candidate_after_payload_json==old_after
    assert encoded(before)==old_before and encoded(after)==old_after
    mutable_before=deepcopy(before)
    frozen=make_native_material_memory_reference(mutable_before,deepcopy(after),
        constitutive_path_sha256=reference.constitutive_path_sha256)
    mutable_before["elapsed_time_s"]=1000000
    assert frozen.before_payload_json==reference.before_payload_json


@pytest.mark.parametrize("damage",["double_archive","zero_share","other_cohort","work_reset",
    "wrong_parent","false_area","false_global_admission","extra_key","boolean_epoch"])
def test_resealed_derived_material_history_cannot_override_proof_or_memory(actual,damage):
    history=actual[4]
    raw=deepcopy(native_material_history_to_dict(history))
    if damage=="double_archive":raw["events"][2]["operation"]="live_to_passive_archive_once"
    elif damage=="zero_share":raw["shares"][0]["amount_bounds"]={"lo":[0,1],"hi":[0,1]}
    elif damage=="other_cohort":raw["events"][1]["cohort_id"]="forged-second-cohort"
    elif damage=="work_reset":raw["memory_reference"]["before_payload"]["cumulative_source_work_j"]=123
    elif damage=="wrong_parent":raw["parent"]["parcel"]["material_id"]="new-unnagged-material"
    elif damage=="false_area":raw["parent_reservoir_partitions"][0]["conservation_identity"]="zero complement"
    elif damage=="false_global_admission":raw["admission_authorized"]=True
    elif damage=="extra_key":raw["resurrect_archived_material"]=True
    else:raw["material_epoch_myr"]=[False,1]
    body={k:v for k,v in raw.items() if k!="history_sha256"}
    raw["history_sha256"]=hashlib.sha256(encoded(body)).hexdigest()
    with pytest.raises(ValueError):native_material_history_from_dict(raw,expected_memory_reference=actual[3])


def test_external_mechanical_reference_mismatch_is_rejected_even_with_valid_local_hash(actual):
    reference,history=actual[3:5]
    different=make_native_material_memory_reference({"preload":"different"},
        json.loads(reference.candidate_after_payload_json),
        constitutive_path_sha256=reference.constitutive_path_sha256)
    with pytest.raises(ValueError,match="external physical evidence"):
        native_material_history_from_dict(native_material_history_to_dict(history),expected_memory_reference=different)


def test_forged_frozen_history_is_rejected_before_material_time_query(actual):
    history=actual[4]
    damaged=replace(history,events=(history.events[0],replace(history.events[1],operation="unbury"),history.events[2]))
    with pytest.raises(ValueError):evaluate_native_material_history(damaged,1)


@pytest.mark.parametrize("time",[Fraction(-1,10),Fraction(11,10),True,float("nan"),float("inf")])
def test_invalid_material_time_cannot_select_allocation(actual,time):
    with pytest.raises(ValueError):evaluate_native_material_history(actual[4],time)


def test_swapped_physical_banks_cannot_reuse_donor_history(actual):
    parent,receiver,sweep,reference,*_=actual
    with pytest.raises(ValueError):
        initialize_native_material_history(receiver,parent,sweep,memory_reference=reference)


def test_same_owner_pair_cannot_claim_interplate_material_transfer(actual):
    parent,receiver,sweep,reference,*_=actual
    receiver=replace(receiver,parcel=replace(receiver.parcel,plate=parent.parcel.plate))
    with pytest.raises(ValueError):initialize_native_material_history(parent,receiver,sweep,memory_reference=reference)


def test_local_snapshot_cannot_claim_force_permission(actual):
    snapshot=evaluate_native_material_history(actual[4],1)
    with pytest.raises(ValueError):native_material_history_snapshot_to_dict(replace(snapshot,admission_authorized=True))


def test_two_material_points_at_the_same_time_keep_different_first_entry_histories(actual):
    sweep,history=actual[2],actual[4]
    early=(sweep.patch_x_bounds.lo,sweep.patch_y_bounds.hi)
    late=(sweep.patch_x_bounds.hi,sweep.patch_y_bounds.lo)
    a=evaluate_native_material_point_history(history,*early,1)
    b=evaluate_native_material_point_history(history,*late,1)
    assert a.events[0].time_fraction.hi<b.events[0].time_fraction.lo
    time=(a.events[0].time_fraction.hi+b.events[0].time_fraction.lo)/2
    first=evaluate_native_material_point_history(history,*early,time)
    second=evaluate_native_material_point_history(history,*late,time)
    assert (first.contact_occupancy,first.material_allocation)==("inside","archived")
    assert (second.contact_occupancy,second.material_allocation)==("outside","live")
    assert first.archive_cohort_id==second.archive_cohort_id==history.cohort_id
    assert first.memory_reference==second.memory_reference==history.memory_reference


@pytest.mark.parametrize("sequence",[1,2,3])
def test_exact_material_point_root_enclosure_keeps_event_time_unresolved(actual,sequence):
    sweep,history=actual[2],actual[4]
    x,y=(sweep.patch_x_bounds.lo+sweep.patch_x_bounds.hi)/2,(sweep.patch_y_bounds.lo+sweep.patch_y_bounds.hi)/2
    events=evaluate_native_material_point_history(history,x,y,1).events
    event=events[sequence-1]
    time=(event.time_fraction.lo+event.time_fraction.hi)/2
    snapshot=evaluate_native_material_point_history(history,x,y,time)
    assert snapshot.contact_occupancy=="event_window" and snapshot.unresolved_event_sequence==sequence
    assert snapshot.material_allocation==("first_entry_window" if sequence==1 else "archived")
    assert snapshot.material_age_myr==Fraction.from_float(history.parent.parcel.age_myr)+time*Fraction.from_float(sweep.dt_myr)


def test_point_query_checkpoint_replays_the_executable_material_time_function(actual):
    sweep,history=actual[2],actual[4]
    x,y=sweep.patch_x_bounds.lo,sweep.patch_y_bounds.hi
    snapshot=evaluate_native_material_point_history(history,x,y,sweep.sample_fractions[2])
    raw=native_material_point_history_to_dict(snapshot)
    loaded=native_material_point_history_from_dict(deepcopy(raw),history=history)
    assert loaded==snapshot
    assert encoded(raw)==encoded(native_material_point_history_to_dict(loaded))
    assert (raw["contact_occupancy"],raw["material_allocation"])==("outside","archived")
    assert raw["finite_material_reservoir_assigned_to_point"] is False
    damaged=deepcopy(raw);damaged["material_allocation"]="live"
    with pytest.raises(ValueError):native_material_point_history_from_dict(damaged,history=history)
    with pytest.raises(ValueError):native_material_point_history_to_dict(replace(snapshot,material_allocation="live"))


@pytest.mark.parametrize("damage",["float_epoch","float_share","float_sequence"])
def test_equal_numeric_values_cannot_bypass_exact_types_in_cached_history(actual,damage):
    history=actual[4]
    # Warm validation first: dataclass numeric equality alone is insufficient.
    evaluate_native_material_history(history,0)
    if damage=="float_epoch":
        damaged=replace(history,material_epoch_myr=float(history.material_epoch_myr))
    elif damage=="float_share":
        damaged=replace(history,shares=(replace(history.shares[0],parent_amount=float(history.shares[0].parent_amount)),*history.shares[1:]))
    else:
        damaged=replace(history,events=(replace(history.events[0],sequence=1.),*history.events[1:]))
    with pytest.raises(ValueError):evaluate_native_material_history(damaged,history.sweep_certificate.sample_fractions[2])


@pytest.mark.parametrize("damage",["phase","float_time","reservoirs"])
def test_aggregate_material_snapshot_is_replayed_before_serializing(actual,damage):
    history=actual[4]
    snapshot=evaluate_native_material_history(history,0)
    if damage=="phase":
        damaged=replace(snapshot,material_allocation="archived",contact_occupancy="inside")
    elif damage=="float_time":
        damaged=replace(snapshot,time_fraction=0.)
    else:
        damaged=replace(snapshot,allocations=tuple(replace(a,live_amount_bounds=I(0),archived_amount_bounds=I(0)) for a in snapshot.allocations))
    with pytest.raises(ValueError):native_material_history_snapshot_to_dict(damaged)
