"""Event material can cool at its own clock without artificial cell capacity."""
from copy import deepcopy
from dataclasses import replace
import math
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import erfinv

from tectonics.fractional_surface import FractionalSurfaceState, SurfaceParcel, split_parcel
from tectonics.fractional_thermal import refresh_fractional_mechanics, refresh_parcel_mechanics
from tectonics.genesis import GenesisParameters, SECONDS_PER_MYR
from tectonics.genesis_shell import ShellParameters, rock_enthalpy


@pytest.fixture
def context():
    model = SimpleNamespace(thermal=GenesisParameters(), shell=ShellParameters())
    ts, tm = 282., 1575.
    depth = model.shell.column_depth_km
    z = (np.arange(model.shell.column_layers)+.5)*depth/model.shell.column_layers
    sample = SimpleNamespace(time_myr=52.5,
        thermal={"mantle_temperature_k": tm, "surface_temperature_k": ts},
        lid_thickness_km=depth*(model.thermal.solidus_k-ts)/(tm-ts),
        mean_lid_temperature_k=.5*(ts+model.thermal.solidus_k),
        column_depth_limit_reached=False,
        state=SimpleNamespace(column_enthalpy=rock_enthalpy(ts+(tm-ts)*z/depth, model.thermal)))
    return model, sample


def piece(age=.4, area=.37, identity="removed:1", cell=731):
    return SurfaceParcel(cell=cell, plate=7, material_id=identity,
        area_km2=area, oceanic_volume_km3=area,
        cold_mantle_volume_km3=2.*area, density_excess_mass_kg=20e9*area,
        age_myr=age, material_fields=(("fracture_damage", .375),),
        specific_properties=(1., 2., 20e9))


def cool(parcels, context):
    model, sample = context
    return refresh_parcel_mechanics(parcels, sample.time_myr, sample, model,
        origin_time_myr=.5)


def test_sparse_event_material_uses_its_age_and_preserves_inventory(context):
    model, sample = context
    original = (piece(age=.05), piece(age=2., identity="removed:2", cell=918, area=.91))
    before_column = sample.state.column_enthalpy.copy()
    cooled, diag = cool(iter(original), context)
    assert len(cooled) == 2
    ts, tm = sample.thermal["surface_temperature_k"], sample.thermal["mantle_temperature_k"]
    coefficient = 2.*math.sqrt(1e-6*SECONDS_PER_MYR)/1000.*erfinv((model.thermal.solidus_k-ts)/(tm-ts))
    for old, new in zip(original, cooled):
        expected_h = max(min(sample.lid_thickness_km, coefficient*math.sqrt(old.age_myr))-1., 0.)
        assert new.cold_mantle_volume_km3 == pytest.approx(old.area_km2*expected_h, rel=2e-15)
        for field in ("cell", "plate", "material_id", "area_km2", "oceanic_volume_km3",
                      "age_myr", "material_fields"):
            assert getattr(new, field) == getattr(old, field)
    np.testing.assert_array_equal(sample.state.column_enthalpy, before_column)
    for field, delta in diag["thermal_source_delta"].items():
        assert math.fsum(getattr(p, field) for p in cooled) == pytest.approx(
            math.fsum(getattr(p, field) for p in original)+delta, rel=2e-15)
    assert diag["additional_global_heat_sink_j"] == 0.


def test_arbitrary_helper_preserves_existing_surface_result_exactly(context):
    model, sample = context
    old = FractionalSurfaceState(sample.time_myr, (3.,),
        (piece(age=.2, area=1., cell=0), piece(age=4., area=2., cell=0, identity="second")))
    surface, surface_diag = refresh_fractional_mechanics(old, sample, model, origin_time_myr=.5)
    events, events_diag = cool(old.parcels, context)
    assert events == surface.parcels
    assert events_diag == surface_diag


def test_event_capture_cooling_commutes_with_area_splitting(context):
    original = piece(age=1.25)
    whole, whole_diag = cool((original,), context)
    fractions = (.21132486540518713, .7886751345948129)
    split, split_diag = cool(tuple(split_parcel(original, f) for f in fractions), context)
    for p in split:
        assert p.specific_properties == whole[0].specific_properties
    for field in ("cold_mantle_volume_km3", "density_excess_mass_kg"):
        assert math.fsum(getattr(p, field) for p in split) == pytest.approx(getattr(whole[0], field), rel=2e-15)
        assert split_diag["thermal_source_delta"][field] == pytest.approx(
            whole_diag["thermal_source_delta"][field], rel=2e-15)


def test_event_cooling_never_uses_an_averaged_age(context):
    young, old = piece(age=.05, area=1.), piece(age=2., area=1., identity="older")
    actual, _ = cool((young, old), context)
    for before, after in zip((young, old), actual):
        independently, _ = cool((before,), context)
        assert independently[0] == after
    averaged, _ = cool((piece(age=1.025, area=2.),), context)
    assert sum(p.cold_mantle_volume_km3 for p in actual) != pytest.approx(
        averaged[0].cold_mantle_volume_km3, rel=1e-3)


def test_event_refresh_is_idempotent_and_does_not_advance_its_age(context):
    first, _ = cool((piece(),), context)
    repeated, diag = cool(first, context)
    assert repeated == first
    assert repeated[0].age_myr == .4
    assert all(value == 0. for value in diag["thermal_source_delta"].values())


@pytest.mark.parametrize("bad", [(), (object(),)])
def test_empty_or_nonparcel_event_groups_are_explicitly_rejected(context, bad):
    with pytest.raises(ValueError, match="nonempty sequence of SurfaceParcel"):
        cool(bad, context)


@pytest.mark.parametrize("clock", [math.nan, math.inf, 52.6])
def test_event_clock_must_match_the_sample(context, clock):
    model, sample = context
    with pytest.raises(ValueError, match="clocks disagree"):
        refresh_parcel_mechanics((piece(),), clock, sample, model, origin_time_myr=.5)


@pytest.mark.parametrize("fault", ["exhausted", "shape", "nan", "negative_column"])
def test_event_thermal_validation_is_identical_to_surface_validation(context, fault):
    model, sample = context
    if fault == "exhausted":
        sample.column_depth_limit_reached = True
    elif fault == "shape":
        sample.state.column_enthalpy = sample.state.column_enthalpy[:-1]
    elif fault == "nan":
        sample.thermal["surface_temperature_k"] = math.nan
    else:
        sample.state.column_enthalpy[0] = -1.
    original = (piece(),)
    saved = deepcopy(original)
    with pytest.raises(ValueError):
        cool(original, context)
    assert original == saved
