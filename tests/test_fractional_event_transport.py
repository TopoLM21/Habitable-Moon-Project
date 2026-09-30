"""Event histories across CFL boundaries and explicit saved-scheme compatibility."""
from dataclasses import replace
import math

import pytest

from tectonics.fractional_transport import advance_fractional_transport
from tectonics.fractional_coupling import (
    EVENT_QUADRATURE, LEGACY_VERSION, VERSION, load_coupled_checkpoint, save_coupled_checkpoint,
)
from test_fractional_time_coupling import transport_fixture
from test_fractional_transport import assert_material_balance
from test_fractional_coupling import source


def test_multiple_cfl_intervals_preserve_birth_and_removal_physical_ages():
    mesh, radius, state, birth, omega, created_at = transport_fixture()
    result = advance_fractional_transport(mesh, state, omega*6., radius, 5.,
        birth_factory=birth, event_time_quadrature=EVENT_QUADRATURE)
    assert result.diagnostics["substeps"] > 1
    assert result.diagnostics["max_outgoing_fraction"] <= 1.
    assert len(created_at) == len(result.births)
    assert_material_balance(state, result)
    initial = {p.material_id: p for p in state.parcels}
    for parcel in result.state.parcels:
        birth_time = created_at.get(parcel.material_id, state.time_myr-50.)
        assert parcel.age_myr == pytest.approx(result.state.time_myr-birth_time, abs=1e-12)
    for record in result.losses:
        birth_time = created_at.get(record.parcel.material_id, state.time_myr-50.)
        assert record.time_myr > birth_time
        assert record.parcel.age_myr == pytest.approx(record.time_myr-birth_time, abs=1e-12)
        if record.parcel.material_id in initial:
            assert record.parcel.material_fields == initial[record.parcel.material_id].material_fields


@pytest.mark.parametrize("rule", [(), ((.5, .9),), ((.25, -.1), (.75, 1.1)),
    ((.75, .5), (.25, .5)), ((.5, .5), (.5, .5)), ((math.nan, 1.),), ((1.1, 1.),)])
def test_invalid_time_rule_is_rejected_before_any_material_birth(rule):
    mesh, radius, state, birth, omega, created_at = transport_fixture()
    with pytest.raises(ValueError, match="Event time quadrature"):
        advance_fractional_transport(mesh, state, omega, radius, 1.,
            birth_factory=birth, event_time_quadrature=rule)
    assert not created_at


def test_old_saved_scheme_requires_explicit_legacy_resume(source, tmp_path):
    model, _, _, state = source
    legacy = replace(state, provenance=dict(state.provenance, coupling_version=LEGACY_VERSION))
    path = tmp_path/"old_scheme.json"
    save_coupled_checkpoint(path, model.mesh, legacy)
    loaded = load_coupled_checkpoint(path, model.mesh, model, legacy.provenance)
    assert loaded.surface == legacy.surface
    new_provenance = dict(legacy.provenance, coupling_version=VERSION, time_scheme="gauss2_v2")
    with pytest.raises(ValueError, match="another event time scheme"):
        load_coupled_checkpoint(path, model.mesh, model, new_provenance)
