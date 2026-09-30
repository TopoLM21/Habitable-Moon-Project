"""Physical parcel budgets and histories, independent of overlap policy."""
from dataclasses import FrozenInstanceError, replace
import json
import math

import pytest

from tectonics.fractional_surface import (
    EXTENSIVE_FIELDS, FractionalSurfaceState, IncomingPiece, SurfaceParcel,
    TransferPiece, commit_surface, remap_surface, split_incoming, split_parcel,
    state_from_json, state_to_json, surface_totals,
)


def parcel(cell, plate, identity, area, age=7., **kwargs):
    values = dict(cell=cell, plate=plate, material_id=identity, area_km2=area,
        oceanic_volume_km3=7.*area, cold_mantle_volume_km3=20.*area,
        density_excess_mass_kg=20.*area*60e9, age_myr=age,
        material_fields=(("damage", .25), ("water_access", .75), ("strength_pa", 4e6)))
    values.update(kwargs)
    return SurfaceParcel(**values)


def state():
    return FractionalSurfaceState(10., (2., 3.),
        (parcel(0, 0, "a", 2.), parcel(1, 1, "b", 3., 2.,
         material_fields=(("damage", .75), ("water_access", .1), ("strength_pa", 9e6)))))


def test_fractional_map_closes_unequal_cell_areas_and_keeps_distinct_histories():
    old = state()
    # Half of A enters cell1, one third of B enters cell0; both capacities close.
    mapped = remap_surface(old, (TransferPiece(0, 0, .5), TransferPiece(0, 1, .5),
        TransferPiece(1, 0, 1./3.), TransferPiece(1, 1, 2./3.)), 2.)
    result = commit_surface(old, mapped, (), (), 2.)
    assert result.time_myr == 12.
    assert len(result.parcels) == 4
    assert surface_totals(result) == surface_totals(old)
    for component in result.parcels:
        source = next(p for p in old.parcels if p.material_id == component.material_id)
        assert component.plate == source.plate
        assert component.age_myr == source.age_myr+2.
        assert component.material_fields == source.material_fields
        assert component.mantle_thickness_km == source.mantle_thickness_km
        assert component.density_anomaly_kg_m3 == source.density_anomaly_kg_m3
    # A display winner cannot erase the other plate, age or damaged fraction.
    assert {p.plate for p in result.parcels if p.cell == 0} == {0, 1}
    assert {p.age_myr for p in result.parcels if p.cell == 0} == {9., 4.}


def test_ocean_loss_and_ridge_creation_close_the_same_single_donor_transaction():
    old = state()
    incoming = remap_surface(old, (TransferPiece(0, 1, 1.), TransferPiece(1, 1, 1.)), 1.)
    retained = (incoming[0], split_incoming(incoming[1], 1./3.))
    losses = (split_incoming(incoming[1], 2./3.),)
    newborn = (parcel(0, 0, "ridge:11:0", 2., 0., cold_mantle_volume_km3=0.,
                      density_excess_mass_kg=0., material_fields=(("damage", 0.),)),)
    result = commit_surface(old, retained, losses, newborn, 1.)
    before, after = surface_totals(old), surface_totals(result)
    for name in EXTENSIVE_FIELDS:
        assert after[name]+math.fsum(getattr(piece.parcel, name) for piece in losses) == pytest.approx(
            before[name]+math.fsum(getattr(p, name) for p in newborn), rel=2e-15)
    assert losses[0].source_index == 1
    assert losses[0].fraction == 2./3.
    assert losses[0].parcel.age_myr == 3.
    assert losses[0].parcel.material_fields == old.parcels[1].material_fields
    assert surface_totals(old) == before


def test_multiple_losses_from_one_donor_keep_distinct_receiver_provenance():
    old = FractionalSurfaceState(10., (1., 1., 1.), tuple(parcel(i, i, str(i), 1.) for i in range(3)))
    incoming = remap_surface(old, (TransferPiece(0, 1, .25), TransferPiece(0, 2, .75),
        TransferPiece(1, 1, 1.), TransferPiece(2, 2, 1.)), 1.)
    losses, retained = incoming[:2], incoming[2:]
    newborn = (parcel(0, 1, "ridge", 1., 0.),)
    result = commit_surface(old, retained, losses, newborn, 1.)
    assert [piece.parcel.cell for piece in losses] == [1, 2]
    assert math.fsum(piece.fraction for piece in losses) == 1.
    assert surface_totals(result) == surface_totals(old)


@pytest.mark.parametrize('transfers', [
    (TransferPiece(0, 0, .9), TransferPiece(1, 1, 1.)),
    (TransferPiece(0, 0, .5), TransferPiece(0, 1, .6), TransferPiece(1, 1, 1.)),
    (TransferPiece(0, 0, 1.),),
    (TransferPiece(0, 0, 1.), TransferPiece(2, 1, 1.)),
    (TransferPiece(0, 0, 1.), TransferPiece(1, 2, 1.)),
])
def test_invalid_geometric_map_rejected_without_partial_mutation(transfers):
    old = state()
    snapshot = state_to_json(old)
    with pytest.raises(ValueError):
        remap_surface(old, transfers, 1.)
    assert state_to_json(old) == snapshot


@pytest.mark.parametrize('tamper', ['mass', 'area', 'age', 'damage', 'plate', 'origin', 'duplicate', 'missing'])
def test_commit_checks_material_lineage_and_each_donor_budget_before_publish(tamper):
    old = state()
    incoming = list(remap_surface(old, (TransferPiece(0, 0, 1.), TransferPiece(1, 1, 1.)), 1.))
    p = incoming[0].parcel
    if tamper == 'mass':
        incoming[0] = replace(incoming[0], parcel=replace(p, oceanic_volume_km3=p.oceanic_volume_km3*.99,
                                                        specific_properties=()))
    elif tamper == 'area':
        incoming[0] = replace(incoming[0], parcel=replace(p, area_km2=p.area_km2*.99,
                                                        specific_properties=()))
    elif tamper == 'age':
        incoming[0] = replace(incoming[0], parcel=replace(p, age_myr=p.age_myr+1.))
    elif tamper == 'damage':
        incoming[0] = replace(incoming[0], parcel=replace(p, material_fields=(("damage", 0.),)))
    elif tamper == 'plate':
        incoming[0] = replace(incoming[0], parcel=replace(p, plate=9))
    elif tamper == 'origin':
        incoming[0] = replace(incoming[0], parcel=replace(p, material_id="fabricated"))
    elif tamper == 'duplicate':
        incoming.append(incoming[0])
    else:
        incoming.pop()
    snapshot = state_to_json(old)
    with pytest.raises(ValueError):
        commit_surface(old, incoming, (), (), 1.)
    assert state_to_json(old) == snapshot


def test_source_conservation_alone_does_not_excuse_target_overfill_or_gap():
    old = state()
    incoming = remap_surface(old, (TransferPiece(0, 1, 1.), TransferPiece(1, 1, 1.)), 1.)
    with pytest.raises(ValueError, match='capacity'):
        commit_surface(old, incoming, (), (), 1.)
    with pytest.raises(ValueError, match='capacity'):
        commit_surface(old, incoming[1:], incoming[:1], (), 1.)


def test_newborn_identity_and_clock_are_explicit_and_original_material_cannot_be_reintroduced():
    old = state()
    incoming = remap_surface(old, (TransferPiece(0, 0, 1.), TransferPiece(1, 1, 1.)), 1.)
    for birth in (parcel(0, 0, "a", 2., 0.), parcel(0, 0, "fresh", 2., 1.)):
        with pytest.raises(ValueError, match='Newborn'):
            commit_surface(old, incoming[1:], incoming[:1], (birth,), 1.)


def test_only_identical_histories_merge_no_mean_age_damage_or_thickness():
    old = FractionalSurfaceState(10., (4.,), (
        parcel(0, 0, "same", 1., 5.), parcel(0, 0, "same", 1., 5.),
        parcel(0, 0, "same", 1., 2.), parcel(0, 0, "same", 1., 5.,
            material_fields=(("damage", .9),))))
    incoming = remap_surface(old, tuple(TransferPiece(i, 0, 1.) for i in range(4)), 1.)
    result = commit_surface(old, incoming, (), (), 1.)
    assert len(result.parcels) == 3
    assert sorted((p.age_myr, p.area_km2) for p in result.parcels) == [(3., 1.), (6., 1.), (6., 2.)]
    assert surface_totals(result) == surface_totals(old)


def test_nested_split_equals_direct_split_and_advances_age_once():
    old = state()
    incoming = remap_surface(old, (TransferPiece(0, 0, 1.), TransferPiece(1, 1, 1.)), 2.)
    a = split_incoming(split_incoming(incoming[0], .5), .5)
    b = split_parcel(old.parcels[0], .25, target_cell=0, age_increment_myr=2.)
    assert a == IncomingPiece(0, .25, b)
    assert a.parcel.age_myr == 9.


@pytest.mark.parametrize('changes', [
    {'area_km2': 0.}, {'oceanic_volume_km3': -1.}, {'cold_mantle_volume_km3': 0.},
    {'density_excess_mass_kg': float('nan')}, {'cell': -1}, {'plate': .5},
    {'material_id': ''}, {'material_fields': (("x", 1.), ("x", 2.))},
    {'material_fields': (("x", float('inf')),)},
])
def test_parcel_rejects_invalid_physical_or_identity_fields(changes):
    with pytest.raises(ValueError):
        replace(parcel(0, 0, "a", 1.), **changes)


def test_immutable_json_snapshot_roundtrip_preserves_every_sparse_history():
    old = state()
    with pytest.raises(FrozenInstanceError):
        old.time_myr = 100.
    with pytest.raises(FrozenInstanceError):
        old.parcels[0].age_myr = 100.
    restored = state_from_json(json.loads(json.dumps(state_to_json(old))))
    assert restored == old
    assert isinstance(restored.parcels, tuple)
    assert isinstance(restored.parcels[0].material_fields, tuple)
    invalid = state_to_json(old)
    invalid['material_model'] = 'mixed_continental'
    with pytest.raises(ValueError, match='oceanic only'):
        state_from_json(invalid)
    invalid = state_to_json(old)
    invalid['parcels'][0]['area_km2'] *= .99
    with pytest.raises(ValueError, match='specific properties|capacity'):
        state_from_json(invalid)


def test_lost_origin_id_cannot_be_reused_after_checkpoint_roundtrip():
    old = state()
    incoming = remap_surface(old, (TransferPiece(0, 0, 1.), TransferPiece(1, 1, 1.)), 1.)
    newer = commit_surface(old, incoming[1:], incoming[:1], (parcel(0, 0, "birth1", 2., 0.),), 1.)
    assert "a" not in {p.material_id for p in newer.parcels}
    assert "a" in newer.known_material_ids
    restored = state_from_json(json.loads(json.dumps(state_to_json(newer))))
    missing_history = state_to_json(newer)
    del missing_history['known_material_ids']
    with pytest.raises(ValueError, match='historical material ID registry'):
        state_from_json(missing_history)
    target = next(i for i, p in enumerate(restored.parcels) if p.cell == 0)
    mapped = remap_surface(restored,
        tuple(TransferPiece(i, p.cell, 1.) for i, p in enumerate(restored.parcels)), 1.)
    with pytest.raises(ValueError, match='fresh origin'):
        commit_surface(restored, [p for p in mapped if p.source_index != target],
            [p for p in mapped if p.source_index == target], (parcel(0, 0, "a", 2., 0.),), 1.)


def test_repeated_uneven_splits_recombine_without_roundoff_cohort_proliferation():
    original = FractionalSurfaceState(10., (13.7,), (parcel(0, 0, "origin", 13.7,
        oceanic_volume_km3=31.9, cold_mantle_volume_km3=211.3,
        density_excess_mass_kg=1.2357e15),))
    current = original
    for _ in range(50):
        mapped = remap_surface(current, (TransferPiece(0, 0, .123),
            TransferPiece(0, 0, .287), TransferPiece(0, 0, .59)), .125)
        current = commit_surface(current, mapped, (), (), .125)
        assert len(current.parcels) == 1
        assert current.parcels[0].specific_properties == original.parcels[0].specific_properties
    assert current.parcels[0].age_myr == original.parcels[0].age_myr+6.25
    for name in EXTENSIVE_FIELDS:
        assert getattr(current.parcels[0], name) == pytest.approx(getattr(original.parcels[0], name), rel=2e-14)


@pytest.mark.parametrize('time,dt', [(10., 0.), (1e20, 1.), (1e308, 1e308)])
def test_commit_rejects_nonadvancing_or_nonfinite_clock_without_touching_input(time, dt):
    old = replace(state(), time_myr=time)
    identity = (TransferPiece(0, 0, 1.), TransferPiece(1, 1, 1.))
    mapped = remap_surface(old, identity, dt)
    before = state_to_json(old)
    with pytest.raises(ValueError, match='timestep|clock'):
        commit_surface(old, mapped, (), (), dt)
    assert state_to_json(old) == before


def test_small_zero_reservoir_cannot_gain_mass_under_an_absolute_tolerance_floor():
    p = parcel(0, 0, 'zero-mass', 1e-20, density_excess_mass_kg=0.)
    with pytest.raises(ValueError, match='specific properties'):
        replace(p, density_excess_mass_kg=1e-16)
    old = FractionalSurfaceState(1., (1e-20,), (p,))
    (incoming,) = remap_surface(old, (TransferPiece(0, 0, 1.),), 1.)
    forged = replace(incoming, parcel=replace(incoming.parcel,
        density_excess_mass_kg=1e-16, specific_properties=()))
    with pytest.raises(ValueError, match='history'):
        commit_surface(old, (forged,), (), (), 1.)
    with pytest.raises(ValueError, match='capacity'):
        FractionalSurfaceState(1., (1e-20,), ())
    with pytest.raises(ValueError, match='capacity'):
        FractionalSurfaceState(1., (1e-20,), (split_parcel(p, .5),))


def test_inherited_signature_rejects_changes_beyond_arithmetic_roundoff():
    p = parcel(0, 0, 'origin', 1.)
    with pytest.raises(ValueError, match='specific properties'):
        replace(p, oceanic_volume_km3=p.oceanic_volume_km3*(1.+1e-13))
    almost = replace(p, oceanic_volume_km3=p.oceanic_volume_km3*(1.+2e-15))
    assert almost.specific_properties == p.specific_properties
