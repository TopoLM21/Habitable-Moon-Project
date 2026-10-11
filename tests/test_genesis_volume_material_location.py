"""Location cannot be invented from a face orientation or resealed claims."""
from dataclasses import FrozenInstanceError, replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tectonics.genesis_volume_checkpoint import _canonical
from tectonics.genesis_volume_material_location import (
    _cohort_offset_family, bind_volume_material_location,
    volume_material_location_to_dict, volume_material_location_from_dict)
from tectonics.genesis_volume_slip_checkpoint import volume_slip_checkpoint_to_dict
from tectonics.genesis_volume_slip_dynamics import VolumeSlipModel


def cohort():
    return SimpleNamespace(id=7, face_index=3, layer_index=2, q_lo=.1,
        live_q_hi=.4, inner_scale=.75, outer_scale=.9,
        material_id="inherited-material", owner_id="inherited-owner")


def test_full_current_offset_range_includes_depth_amplitudes():
    X = np.array(((1., 0., 4.), (0., 2., 4.), (-1., 0., 4.)))
    U = np.array(((.4, .2, .1, .1, .2, 2.),
                  (-.1, .2, .2, 1., -.2, 4.),
                  (.3, -.2, .1, -.5, .2, -3.)))
    n = np.array((1., 2., 3.))/np.sqrt(14)
    raw = _cohort_offset_family(cohort(), X, U, n)
    corners = []
    for lam in (.75, .9):
        for i in range(3):
            # Independent evaluation of the literal displacement law.
            corners.append(n@(lam*X[i]+U[i, :3]+(lam-1)*U[i, 3:]))
    assert raw["offset_bounds_m"] == pytest.approx((min(corners), max(corners)))
    assert np.allclose(np.ravel(raw["corner_offsets_m"]), corners, rtol=2e-15, atol=1e-15)
    rng = np.random.default_rng(37)
    for N, lam in zip(rng.dirichlet(np.ones(3), 100), rng.uniform(.75, .9, 100)):
        value = n@(lam*(N@X)+N@U[:, :3]+(lam-1)*(N@U[:, 3:]))
        assert min(corners) <= value <= max(corners)
    assert raw["cohort_id"] == 7 and raw["material_id"] == "inherited-material"
    assert raw["mass_coordinate_q_bounds"] == [.1, .4]


def test_radial_copy_of_current_top_misses_saved_depth_kinematics():
    X = np.array(((0., 0., 4.), (1., 0., 4.), (0., 1., 4.)))
    U = np.tile((0., 0., 2., 0., 0., 4.), (3, 1))
    raw = _cohort_offset_family(cohort(), X, U, (0., 0., 1.))
    assert raw["offset_bounds_m"] == pytest.approx((4., 5.2))
    naive = (6*.75, 6*.9)
    assert not np.allclose(raw["offset_bounds_m"], naive)


@pytest.mark.parametrize("normal", [(0., 0., 0.), (0., 0., 2.), (0., float('nan'), 1.)])
def test_offset_family_refuses_invalid_orientation(normal):
    with pytest.raises(ValueError, match="finite six-DOF"):
        _cohort_offset_family(cohort(), np.eye(3), np.zeros((3, 6)), normal)


def test_no_material_interval_can_be_extended_to_the_planet_centre():
    c = cohort()
    c.inner_scale = 0.
    with pytest.raises(ValueError, match="positive live"):
        _cohort_offset_family(c, np.eye(3), np.zeros((3, 6)), (0., 0., 1.))


@pytest.fixture(scope="module")
def accepted():
    path = Path(__file__).resolve().parents[1]/(
        "analysis/genesis_cohort_mechanics_validation/evidence/control/material_export.json")
    source = path.read_bytes()
    model = VolumeSlipModel(json.loads(source))
    state = model.initial_state()
    checkpoint = _canonical(volume_slip_checkpoint_to_dict(model, state))
    location = bind_volume_material_location(model, state)
    yield model, state, location
    assert path.read_bytes() == source
    assert _canonical(volume_slip_checkpoint_to_dict(model, state)) == checkpoint


def test_actual_no_event_does_not_make_a_centroid_or_edge_fracture(accepted):
    model, state, location = accepted
    raw = volume_material_location_to_dict(location)
    assert raw["orientation_offset_families"] == []
    assert raw["newly_activating_faces"] == raw["existing_active_faces"] == []
    assert raw["source_new_activation_event"] is None
    assert raw["status"] == "no_calculated_activation_event"
    assert raw["selected_sharp_material_surface"] is None
    assert raw["source_nucleation_location_law"] is None
    assert raw["source_physical_dip_law"] is None
    assert raw["source_surface_creation_energy_law"] is None
    assert raw["time_myr"] == state.time_myr
    assert raw["parent_checkpoint_sha256"] == volume_slip_checkpoint_to_dict(model, state)["checkpoint_sha256"]
    assert all(raw[key] is False for key in ("physical_creation_admitted", "physical_release_admitted",
        "physical_fault_contact_admission_claimed", "history_reinitialized", "activation_accepted", "clock_advanced"))
    assert raw["creation_work_debited_from_Genesis_j"] == 0.


def test_full_source_bound_descriptor_restore_and_immutable_binding(accepted):
    model, state, location = accepted
    raw = volume_material_location_to_dict(location)
    assert volume_material_location_from_dict(raw, model=model, state=state) == location
    with pytest.raises(FrozenInstanceError):
        location.payload_json = "{}"
    with pytest.raises(ValueError, match="immutable"):
        volume_material_location_to_dict(replace(location, payload_json="{}"))


@pytest.mark.parametrize("key,value", [
    ("time_myr", 0.), ("parent_checkpoint_sha256", "0"*64),
    ("orientation_offset_families", [{"face_index": 0, "selected_offset_m": 0.}]),
    ("selected_sharp_material_surface", {"centroid_of_face": 0}),
    ("source_nucleation_location_law", {"nearest_edge": True}),
    ("source_physical_dip_law", {"dip_degrees": 45.}),
    ("source_surface_creation_energy_law", {"Gc": 1e8}),
    ("physical_creation_admitted", True), ("physical_release_admitted", True),
    ("creation_work_debited_from_Genesis_j", 1.),
])
def test_resealed_localization_and_history_substitutions_refused(accepted, key, value):
    model, state, location = accepted
    raw = volume_material_location_to_dict(location)
    raw[key] = value
    raw["location_sha256"] = hashlib.sha256(_canonical({
        k: v for k, v in raw.items() if k != "location_sha256"})).hexdigest()
    with pytest.raises(ValueError, match="complete accepted-source"):
        volume_material_location_from_dict(raw, model=model, state=state)


@pytest.mark.parametrize("model,state", [(None, None), (object(), object())])
def test_location_requires_typed_accepted_runtime(model, state):
    with pytest.raises(ValueError, match="accepted Genesis"):
        bind_volume_material_location(model, state)
