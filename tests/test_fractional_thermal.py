"""Age-resolved cooling sources must commute with conservative parcel splits."""
from dataclasses import replace
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from tectonics.fractional_surface import (FractionalSurfaceState, SurfaceParcel,
    _merge_identical_histories, split_parcel, state_from_json, state_to_json,
    surface_totals)
from tectonics.fractional_thermal import refresh_fractional_mechanics
from tectonics.genesis import GenesisParameters
from tectonics.genesis_local_mechanics import refresh_young_material_mechanics
from tectonics.genesis_shell import ShellParameters, rock_enthalpy


@pytest.fixture
def thermal():
    model = SimpleNamespace(thermal=GenesisParameters(), shell=ShellParameters())
    ts, tm = 282., 1575.
    depth = model.shell.column_depth_km
    z = (np.arange(model.shell.column_layers)+.5)*depth/model.shell.column_layers
    sample = SimpleNamespace(time_myr=405.75,
        thermal={"mantle_temperature_k": tm, "surface_temperature_k": ts},
        lid_thickness_km=depth*(model.thermal.solidus_k-ts)/(tm-ts),
        mean_lid_temperature_k=.5*(ts+model.thermal.solidus_k),
        column_depth_limit_reached=False,
        state=SimpleNamespace(column_enthalpy=rock_enthalpy(ts+(tm-ts)*z/depth, model.thermal)))
    return model, sample, .75


def parcel(area=1., age=3., identity="source:0", cell=0, crust=1.):
    return SurfaceParcel(cell=cell, plate=2, material_id=identity,
        area_km2=area, oceanic_volume_km3=area*crust,
        cold_mantle_volume_km3=area*2., density_excess_mass_kg=area*2.*10e9,
        age_myr=age, material_fields=(("damage", .38), ("strength_pa", 4.2e6)),
        specific_properties=(crust, 2., 2.*10e9))


def refresh(state, thermal):
    model, sample, origin = thermal
    return refresh_fractional_mechanics(state, sample, model, origin_time_myr=origin)


def test_unit_cell_parity_with_existing_young_05_closure_and_primordial_column(thermal):
    model, sample, origin = thermal
    ages = [0., .001, .1, .4, 3., 300., 405.]
    old = FractionalSurfaceState(sample.time_myr, tuple(1. for _ in ages),
        tuple(parcel(age=age, cell=i, identity=f"source:{i}") for i, age in enumerate(ages)))
    column_before = sample.state.column_enthalpy.copy()
    view = SimpleNamespace(time_myr=old.time_myr, crust_age_myr=np.asarray(ages),
        crust_thickness_km=np.ones(len(ages)))
    expected = refresh_young_material_mechanics(view, sample, model, origin_time_myr=origin)
    new, diagnostic = refresh(old, thermal)
    np.testing.assert_array_equal([p.cold_mantle_volume_km3 for p in new.parcels],
        view.mantle_lithosphere_thickness_km)
    np.testing.assert_array_equal([p.density_excess_mass_kg for p in new.parcels],
        view.mantle_lithosphere_thickness_km*view.mantle_lithosphere_density_anomaly_kg_m3*1e9)
    np.testing.assert_array_equal(sample.state.column_enthalpy, column_before)
    assert new.parcels[0].cold_mantle_volume_km3 == new.parcels[0].density_excess_mass_kg == 0.
    assert new.parcels[1].cold_mantle_volume_km3 == 0.  # Cooling remains inside chemical crust.
    assert new.parcels[-1].mantle_thickness_km == sample.lid_thickness_km-1.
    assert diagnostic["primordial_parcel_count"] == expected["primordial_cell_count"] == 1
    assert diagnostic["rejuvenated_parcel_count"] == 6
    assert diagnostic["local_total_lid_thickness_km"] == expected["local_total_lid_thickness_km"].tolist()
    assert 0. < diagnostic["local_total_lid_thickness_km"][1] < 1.


def test_mixed_ages_and_intrinsic_crust_are_not_replaced_by_a_mean(thermal):
    _, sample, _ = thermal
    old = FractionalSurfaceState(sample.time_myr, (4.,),
        (parcel(area=1., age=.1, identity="young", crust=.5),
         parcel(area=3., age=3., identity="older", crust=2.)))
    new, _ = refresh(old, thermal)
    for original, actual in zip(old.parcels, new.parcels):
        separate = FractionalSurfaceState(old.time_myr, (original.area_km2,), (original,))
        independent, _ = refresh(separate, thermal)
        assert actual == independent.parcels[0]
    mean = FractionalSurfaceState(old.time_myr, (4.,),
        (parcel(area=4., age=(.1+3.*3.)/4., crust=(.5+3.*2.)/4.),))
    averaged, _ = refresh(mean, thermal)
    assert surface_totals(new)["cold_mantle_volume_km3"] != pytest.approx(
        surface_totals(averaged)["cold_mantle_volume_km3"], rel=1e-3)
    assert surface_totals(new)["density_excess_mass_kg"] != pytest.approx(
        surface_totals(averaged)["density_excess_mass_kg"], rel=1e-3)


def test_refresh_preserves_chemical_inventory_lineage_damage_and_input(thermal):
    _, sample, _ = thermal
    old = FractionalSurfaceState(sample.time_myr, (3.,), (parcel(area=3.),),
        known_material_ids=("source:0", "already-lost"))
    snapshot = state_to_json(old)
    new, diagnostic = refresh(old, thermal)
    assert state_to_json(old) == snapshot
    assert new.time_myr == old.time_myr
    assert new.cell_areas_km2 == old.cell_areas_km2
    assert new.known_material_ids == old.known_material_ids
    a, b = old.parcels[0], new.parcels[0]
    for name in ("area_km2", "oceanic_volume_km3", "age_myr", "material_id", "cell", "plate", "material_fields"):
        assert getattr(a, name) == getattr(b, name)
    assert a.specific_properties[0] == b.specific_properties[0]
    assert a.specific_properties[1:] != b.specific_properties[1:]
    for name, delta in diagnostic["thermal_source_delta"].items():
        assert diagnostic["after"][name] == pytest.approx(diagnostic["before"][name]+delta, rel=2e-15)
    assert diagnostic["additional_global_heat_sink_j"] == 0.
    json.dumps(diagnostic, allow_nan=False)


def test_split_merge_and_permutation_invariance(thermal):
    _, sample, _ = thermal
    p = parcel(area=7.3, crust=1.123456789)
    whole = FractionalSurfaceState(sample.time_myr, (p.area_km2,), (p,))
    split = replace(whole, parcels=tuple(split_parcel(p, fraction)
        for fraction in (.1, .2, .3, .4)))
    cooled, dc = refresh(whole, thermal)
    pieces, dp = refresh(split, thermal)
    reversed_pieces, dr = refresh(replace(split, parcels=split.parcels[::-1]), thermal)
    assert reversed_pieces.parcels == pieces.parcels[::-1]
    assert dr == dp
    assert all(piece.specific_properties == cooled.parcels[0].specific_properties for piece in pieces.parcels)
    merged = _merge_identical_histories(pieces.parcels)
    assert len(merged) == 1
    for name in surface_totals(cooled):
        assert surface_totals(pieces)[name] == pytest.approx(surface_totals(cooled)[name], rel=2e-15)
        assert getattr(merged[0], name) == pytest.approx(getattr(cooled.parcels[0], name), rel=2e-15)
    for name in dc["thermal_source_delta"]:
        assert dp["thermal_source_delta"][name] == pytest.approx(dc["thermal_source_delta"][name], rel=2e-15)


def test_total_lid_diagnostics_follow_exact_returned_parcel_order(thermal):
    _, sample, _ = thermal
    old = FractionalSurfaceState(sample.time_myr, (3.,),
        tuple(parcel(age=age, identity=str(age)) for age in (.001, 3., .1)))
    forward, fd = refresh(old, thermal)
    backward, bd = refresh(replace(old, parcels=old.parcels[::-1]), thermal)
    assert backward.parcels == forward.parcels[::-1]
    assert bd["local_total_lid_thickness_km"] == fd["local_total_lid_thickness_km"][::-1]
    assert fd["local_total_lid_thickness_km"][0] < fd["local_total_lid_thickness_km"][2]
    assert fd["local_total_lid_thickness_km"][2] < fd["local_total_lid_thickness_km"][1]


def test_repeated_refresh_and_checkpoint_have_no_hidden_cooling_clock(thermal):
    _, sample, _ = thermal
    old = FractionalSurfaceState(sample.time_myr, (1.,), (parcel(),))
    first, _ = refresh(old, thermal)
    second, diagnostic = refresh(first, thermal)
    restored = state_from_json(json.loads(json.dumps(state_to_json(first))))
    resumed, resumed_diagnostic = refresh(restored, thermal)
    assert second == resumed == first
    assert resumed_diagnostic == diagnostic
    assert all(delta == 0. for delta in diagnostic["thermal_source_delta"].values())


def test_primordial_initial_handoff_uses_reference_column(thermal):
    model, sample, origin = thermal
    sample.time_myr = origin
    old = FractionalSurfaceState(origin, (1.,), (parcel(age=0.),))
    new, diagnostic = refresh(old, thermal)
    assert diagnostic["primordial_parcel_count"] == 1
    assert new.parcels[0].mantle_thickness_km == sample.lid_thickness_km-1.


@pytest.mark.parametrize("change", ["sample_clock", "exhaustion_flag", "exhausted_depth", "nan_temperature", "column_shape"])
def test_inconsistent_or_unsupported_thermal_context_rejected_atomically(thermal, change):
    model, sample, _ = thermal
    old = FractionalSurfaceState(sample.time_myr, (1.,), (parcel(),))
    snapshot = state_to_json(old)
    if change == "sample_clock":
        sample.time_myr += .01
    elif change == "exhaustion_flag":
        sample.column_depth_limit_reached = True
    elif change == "exhausted_depth":
        sample.lid_thickness_km = model.shell.column_depth_km
    elif change == "nan_temperature":
        sample.thermal["mantle_temperature_k"] = math.nan
    else:
        sample.state.column_enthalpy = sample.state.column_enthalpy[:-1]
    with pytest.raises(ValueError):
        refresh(old, thermal)
    assert state_to_json(old) == snapshot


@pytest.mark.parametrize("options", [{"origin_time_myr": -1.}, {"origin_time_myr": 1000.},
    {"origin_time_myr": .75, "thermal_diffusivity_m2_s": 0.}])
def test_invalid_origin_and_cooling_diffusivity_are_rejected(thermal, options):
    model, sample, _ = thermal
    old = FractionalSurfaceState(sample.time_myr, (1.,), (parcel(),))
    with pytest.raises(ValueError, match="origin, clock, and diffusivity"):
        refresh_fractional_mechanics(old, sample, model, **options)
