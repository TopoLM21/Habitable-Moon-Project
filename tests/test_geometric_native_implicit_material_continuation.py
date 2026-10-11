"""Actual fresh force masks, all material reservoirs, and anchored restart."""
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
from functools import lru_cache

import pytest

from tectonics.geometric_native_first_entry_material import _typed_fingerprint
from tectonics.geometric_native_implicit_material_continuation import (
    advance_native_implicit_material_epoch,native_implicit_material_continuation_from_dict,
    native_implicit_material_continuation_to_dict,validate_native_implicit_material_continuation,
)
from tectonics.geometric_native_implicit_material_epoch import (
    native_implicit_material_epoch_from_dict,native_implicit_material_epoch_to_dict,
    transport_native_implicit_material_epoch,
)
from tectonics.geometric_native_implicit_mantle_loading import (
    assemble_native_implicit_physical_spatial_mantle_load,native_implicit_load_force_error_bounds,
)
from tectonics.geometric_native_pose import FractionInterval as I
from tests.test_geometric_native_implicit_underthrust_motion import get_implicit_refusal_case


@pytest.fixture(scope="module")
def actual_continuation():
    return get_implicit_refusal_case()


@lru_cache(maxsize=1)
def get_material_continuation_restart_case():
    case=get_implicit_refusal_case()
    epoch=case["material_candidate"]
    rested=advance_native_implicit_material_epoch(epoch,{0:(0.,0.,0.),1:(0.,0.,0.)},epoch.dt_myr,
        provenance="Common SO(3) identity continuation verifies material history and recursive restart only")
    return case,rested


def test_actual_new_full_euler_vector_drives_new_masks_without_force_adoption(actual_continuation):
    case=actual_continuation
    epoch=case["material_candidate"]
    assert _typed_fingerprint(epoch.previous_epoch)==_typed_fingerprint(case["before"].current_material_epoch)
    assert epoch.omega_rad_per_myr==case["candidate"].omega_rad_per_myr
    assert all(all(value!=0. for value in omega) for _,omega in epoch.omega_rad_per_myr)
    assert epoch.force_admission_authorized is False
    assert len(case["before"].receipts)==1
    assert len(epoch.components)>len(epoch.previous_epoch.components)
    assert set(row.component_id for row in epoch.previous_epoch.components)<=set(row.component_id for row in epoch.components)


def test_each_original_and_newborn_parent_keeps_its_four_complementary_reservoirs(actual_continuation):
    epoch=actual_continuation["material_candidate"]
    snapshots=epoch.current_material_parent_reservoirs
    assert len(snapshots)==len(epoch.components)
    for _,_,_,_,reservoirs in snapshots:
        assert len(reservoirs)==4
        for row in reservoirs:
            total=row.archive_amount_bounds+row.live_amount_bounds
            assert total.lo<=row.parent_amount_bounds.hi and total.hi>=row.parent_amount_bounds.lo
            assert row.archive_amount_bounds.lo>=0 and row.live_amount_bounds.lo>=0
    original=dict(epoch.current_original_parent_reservoirs)
    source=epoch.source_material_state
    assert len(original)==8
    for parent in source.parent_allocations:
        cohorts=tuple(row for row in epoch.new_archives if row.material_id==parent.material_id)
        for old,total in zip(parent.reservoirs,original[parent.fragment_id]):
            increment=sum((next(row.archive_amount_bounds for row in cohort.reservoirs
                if row.reservoir==old.reservoir) for cohort in cohorts),I(0))
            accumulated=old.archived_amount_bounds+increment
            assert accumulated.lo<=total.archive_amount_bounds.hi
            assert accumulated.hi>=total.archive_amount_bounds.lo
            assert total.parent_amount_bounds==I(old.parent_amount)


def test_positive_tiny_old_and_new_hot_cohorts_are_retained(actual_continuation):
    epoch=actual_continuation["material_candidate"]
    old_five=next(row for row in epoch.components if row.material_id=="octant-material:5")
    original_five=next(row for row in epoch.source_material_state.parent_allocations if row.fragment_id=="octant:5")
    assert old_five.archived_area_bounds_sr.lo>0
    assert old_five.archived_area_bounds_sr.hi>=original_five.archived_area_bounds_sr.lo>0
    hot=tuple(row for row in epoch.new_archives if row.material_id.startswith("native-first-entry-hot-birth:")
        and row.positive_measure_proved)
    assert len(hot)>=2
    assert all(row.newly_archived_area_bounds_sr.lo>0 for row in hot)
    assert all(not row.newborn_physical_pair_admission_authorized for row in hot)
    assert any(float(row.newly_archived_area_bounds_sr.hi)<1e-40 for row in hot)


def test_current_and_candidate_source_operator_errors_are_explicitly_bounded(actual_continuation):
    case=actual_continuation
    actual_omega=dict(case["candidate"].omega_rad_per_myr)
    current=native_implicit_load_force_error_bounds(case["load"],actual_omega)
    assert float(current["relative_residual_norm_upper"])<2e-10
    candidate_load=assemble_native_implicit_physical_spatial_mantle_load(case["first"]["original"]["mesh"],
        case["material_candidate"],case["load"].source)
    bounded=native_implicit_load_force_error_bounds(candidate_load,actual_omega)
    assert float(bounded["relative_residual_norm_upper"])<2e-10
    assert not bounded["material_midpoint_debit"]
    hot_ids={row.component_id for row in case["material_candidate"].components if row.material_kind=="hot_birth"}
    loaded_ids={row.component_id for row in candidate_load.material_cell_measures}
    assert hot_ids<=loaded_ids


def test_two_continuation_restart_replays_nested_source_and_all_ages(actual_continuation):
    case,rested=get_material_continuation_restart_case()
    raw=native_implicit_material_epoch_to_dict(rested)
    assert "source_material_state" in raw["previous_epoch"]["previous_epoch"]
    restored=native_implicit_material_epoch_from_dict(raw,
        expected_source_material_state=case["material_candidate"].source_material_state)
    assert restored==rested
    assert not rested.new_archives
    assert tuple(row.component_id for row in rested.components)==tuple(row.component_id for row in rested.previous_epoch.components)
    assert all(row.age_myr==rested.time_myr-row.birth_time_myr for row in rested.components)


def test_embedded_valid_other_prior_epoch_cannot_replace_independently_trusted_prior(actual_continuation):
    case,rested=get_material_continuation_restart_case()
    raw=native_implicit_material_continuation_to_dict(rested)
    with pytest.raises(ValueError,match="independently supplied prior"):
        native_implicit_material_continuation_from_dict(raw,
            expected_source_material_state=rested.source_material_state,
            expected_previous_epoch=case["before"].current_material_epoch)
    compact=native_implicit_material_continuation_to_dict(rested,include_previous=False)
    assert native_implicit_material_continuation_from_dict(compact,
        expected_source_material_state=rested.source_material_state,
        expected_previous_epoch=rested.previous_epoch)==rested
    forged=deepcopy(raw)
    forged["components"][0]["owner"]=False
    with pytest.raises(ValueError):
        native_implicit_material_continuation_from_dict(forged,
            expected_source_material_state=rested.source_material_state,
            expected_previous_epoch=rested.previous_epoch)


def test_free_transport_cannot_reset_new_archives_or_births_to_the_initial_epoch(actual_continuation):
    with pytest.raises(ValueError,match="preserve every new archive and birth"):
        transport_native_implicit_material_epoch(actual_continuation["material_candidate"],
            {0:(0.,0.,0.),1:(0.,0.,0.)},.001)


def test_zero_age_boolean_alias_cannot_hit_authenticated_material_replay_cache(actual_continuation):
    epoch=actual_continuation["material_candidate"]
    index=next(index for index,row in enumerate(epoch.components) if row.age_myr==Fraction(0))
    rows=list(epoch.components)
    rows[index]=replace(rows[index],age_myr=False)
    forged=replace(epoch,components=tuple(rows))
    # bool.numerator/denominator produce the same saved [0,1] as Fraction(0).
    # The independent strict raw-field fingerprint must still reject it.
    with pytest.raises(ValueError,match="complete prior mask and new SO"):
        validate_native_implicit_material_continuation(forged)
