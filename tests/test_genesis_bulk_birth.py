"""Conserved sub-layer material birth from accepted connected-front history."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from tectonics.genesis_bulk_birth import (
    BulkBirthHistory, advance_bulk_birth, bulk_birth_from_dict, bulk_birth_to_dict,
    initial_bulk_birth, validate_bulk_birth, validate_bulk_birth_continuation,
)
from tectonics.mesh import build_icosphere


def origin():
    mesh = build_icosphere(0)
    # Unequal thermal-layer inventories exercise a mass coordinate rather
    # than accidentally relying on uniform layers or geometric thickness.
    mass = np.tile([1., 3., 6.], (mesh.cell_count, 1))
    history = initial_bulk_birth(mesh, mass, 10., {"thermal": {"initial_temperature_k": 2300.},
        "shell": {"seed": 3, "column_layers": 3}, "purpose": "analytic front control"}, 1400.)
    return mesh, mass, history


def advance(history, mesh, end, front, *, radius=10.):
    after = np.full(len(history.front_fraction), front) if np.ndim(front) == 0 else front
    return advance_bulk_birth(history, before_front_fraction=history.front_fraction,
        after_front_fraction=after, before_time_myr=history.time_myr,
        after_time_myr=end, mesh=mesh, radius_km=radius, solidus_k=1400.)


def reseal(raw):
    body = {key: value for key, value in raw.items() if key != "sha256"}
    raw["sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return raw


def test_molten_origin_retains_inventory_without_solid_material_or_tensors():
    mesh, mass, history = origin()
    assert history.time_myr == history.accepted_intervals == 0
    assert history.cohorts == history.intervals == ()
    assert history.live_mass_kg == history.formed_mass_kg == history.remelted_mass_kg == 0
    assert np.array_equal(history.layer_mass_kg, mass)
    assert deepcopy(history) is history
    with pytest.raises(FrozenInstanceError):
        history.time_myr = 1.
    later = advance(history, mesh, .25, 0.)
    assert later.accepted_intervals == 1 and later.cohorts == ()
    assert "strain" not in json.dumps(bulk_birth_to_dict(later))
    assert "prestress" not in json.dumps(bulk_birth_to_dict(later))


def test_growth_partitions_existing_nonuniform_layer_mass_and_brackets_birth():
    mesh, mass, history = origin()
    frozen = advance(history, mesh, .8, .55)
    first = [c for c in frozen.cohorts if c.face_index == 0]
    assert [(c.layer_index, c.q_lo, c.q_hi) for c in first] == [(0, 0., .1), (1, .1, .4), (2, .4, .55)]
    assert [c.id for c in frozen.cohorts] == list(range(1, 3*mesh.cell_count+1))
    assert all(c.birth_before_time_myr == 0. and c.birth_after_time_myr == .8 for c in frozen.cohorts)
    assert all(c.birth_temperature_k == 1400. for c in frozen.cohorts)
    assert frozen.live_mass_per_face_kg == pytest.approx([5.5]*mesh.cell_count)
    assert frozen.live_mass_kg == pytest.approx(5.5*mesh.cell_count)
    assert frozen.formed_mass_kg == frozen.live_mass_kg
    assert history.cohorts == () and np.array_equal(history.layer_mass_kg, mass)
    validate_bulk_birth(frozen, layer_mass_kg=mass, time_myr=.8, front_fraction=[.55]*mesh.cell_count)


def test_remelt_retires_deepest_overlap_and_refreeze_creates_new_identities():
    mesh, mass, history = origin()
    frozen = advance(history, mesh, 1., .6)
    melted = advance(frozen, mesh, 2., .25)
    first = [c for c in melted.cohorts if c.face_index == 0]
    assert [c.live_q_hi for c in first] == [.1, .25, .4]
    assert melted.live_mass_per_face_kg == pytest.approx([2.5]*mesh.cell_count)
    assert melted.remelted_mass_per_face_kg == pytest.approx([3.5]*mesh.cell_count)
    refrozen = advance(melted, mesh, 3., .7)
    inherited = refrozen.cohorts[:len(melted.cohorts)]
    assert inherited == melted.cohorts
    new = [c for c in refrozen.cohorts[len(inherited):] if c.face_index == 0]
    assert [(c.layer_index, c.q_lo, c.q_hi) for c in new] == [(1, .25, .4), (2, .4, .7)]
    assert all(c.id >= melted.next_id for c in new)
    assert refrozen.formed_mass_per_face_kg == pytest.approx([10.5]*mesh.cell_count)
    assert refrozen.live_mass_per_face_kg == pytest.approx([7.]*mesh.cell_count)
    assert refrozen.remelted_mass_per_face_kg == pytest.approx([3.5]*mesh.cell_count)
    molten = advance(refrozen, mesh, 4., 0.)
    assert molten.live_mass_kg == 0
    assert molten.cohorts and all(c.live_q_hi == c.q_lo for c in molten.cohorts)
    assert molten.formed_mass_kg == molten.remelted_mass_kg
    assert bulk_birth_from_dict(bulk_birth_to_dict(molten)) == molten
    validate_bulk_birth_continuation(refrozen, frozen)


def test_birth_geometry_is_literal_old_face_metric_frame_and_mass_is_independent():
    mesh, _, history = origin()
    # A quarter-turn is an independent rigid-rotation check of the world basis.
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    rotated = SimpleNamespace(vertices=mesh.vertices@rotation.T, faces=mesh.faces.copy())
    first = advance(history, mesh, 1., .1)
    turned = advance(history, rotated, 1., .1)
    for a, b in zip(first.cohorts, turned.cohorts):
        assert np.asarray(a.birth_metric_m2) == pytest.approx(np.asarray(b.birth_metric_m2))
        assert np.asarray(b.birth_frame_xyz) == pytest.approx(rotation@np.asarray(a.birth_frame_xyz))
    expanded = advance(first, mesh, 2., .2, radius=20.)
    a = first.cohorts[0]
    b = expanded.cohorts[len(first.cohorts)]
    assert np.asarray(b.birth_metric_m2) == pytest.approx(4*np.asarray(a.birth_metric_m2))
    assert b.birth_radius_km == 20.
    assert expanded.live_mass_per_face_kg == pytest.approx([2.]*mesh.cell_count)
    assert expanded.cohorts[:len(first.cohorts)] == first.cohorts


def test_faces_freeze_and_remelt_independently_without_material_exchange():
    mesh, _, history = origin()
    front = np.zeros(mesh.cell_count)
    front[0], front[1] = .45, .2
    frozen = advance(history, mesh, 1., front)
    end = front.copy()
    end[0], end[1], end[2] = .15, .4, .3
    next_state = advance(frozen, mesh, 2., end)
    expected = np.zeros(mesh.cell_count)
    expected[:3] = [1.5, 4., 3.]
    assert next_state.live_mass_per_face_kg == pytest.approx(expected)
    assert next_state.remelted_mass_per_face_kg[0] == pytest.approx(3.)
    assert next_state.remelted_mass_per_face_kg[1:] == pytest.approx([0.]*(mesh.cell_count-1))
    assert bulk_birth_from_dict(bulk_birth_to_dict(next_state)) == next_state


def test_archive_is_exact_independent_and_replay_restores_continuation():
    mesh, _, history = origin()
    partial = advance(history, mesh, .3, .35)
    raw = bulk_birth_to_dict(partial)
    restored = bulk_birth_from_dict(raw)
    assert restored == partial and restored is not partial
    assert advance(restored, mesh, .6, .8) == advance(partial, mesh, .6, .8)
    raw["cohorts"][0]["q_hi"] = .099
    assert partial.cohorts[0].q_hi == .1
    with pytest.raises(ValueError, match="hash"):
        bulk_birth_from_dict(raw)


@pytest.mark.parametrize("mutation", [
    lambda raw: raw["cohorts"][0].update(birth_before_time_myr=.01),
    lambda raw: raw["cohorts"][0].update(id=1000),
    lambda raw: raw["cohorts"][0].update(live_q_hi=.05),
    lambda raw: raw["cohorts"][0].update(birth_temperature_k=1399.),
    lambda raw: raw["intervals"][1].update(before_time_myr=.4),
    lambda raw: raw["intervals"][0]["after_front_fraction"].__setitem__(0, .15),
    lambda raw: raw.update(time_myr=3.),
    lambda raw: raw["layer_mass_kg"][0].__setitem__(0, 2.),
    lambda raw: raw["cohorts"][0].update(birth_frame_xyz=[[0., 0., 0.]]*3),
    lambda raw: raw["cohorts"][0].update(birth_metric_m2=[[1., 2.], [2., 1.]]),
    lambda raw: raw["cohorts"][0].update(invented_total_strain=[0.]*6),
])
def test_resealed_archive_cannot_replace_source_birth_or_retirement(mutation):
    mesh, _, history = origin()
    history = advance(advance(history, mesh, 1., .3), mesh, 2., .5)
    raw = bulk_birth_to_dict(history)
    mutation(raw)
    reseal(raw)
    with pytest.raises(ValueError):
        bulk_birth_from_dict(raw)


def test_new_constructed_history_cannot_bypass_validation_via_cache():
    mesh, mass, history = origin()
    history = advance(history, mesh, 1., .3)
    changed = replace(history, cohorts=(replace(history.cohorts[0], live_q_hi=.05),)+history.cohorts[1:])
    with pytest.raises(ValueError, match="retirement"):
        validate_bulk_birth(changed, layer_mass_kg=mass, time_myr=1., front_fraction=history.front_fraction)


@pytest.mark.parametrize("bad", ["clock", "front", "mass", "identity"])
def test_external_owner_must_bind_inventory_clock_front_and_origin(bad):
    mesh, mass, history = origin()
    history = advance(history, mesh, 1., .3)
    args = dict(layer_mass_kg=mass, time_myr=1., front_fraction=history.front_fraction,
                source_identity=history.source_identity)
    if bad == "clock":
        args["time_myr"] = 1.1
    elif bad == "front":
        args["front_fraction"] = [.31]*mesh.cell_count
    elif bad == "mass":
        args["layer_mass_kg"] = mass*2
    else:
        args["source_identity"] = "0"*64
    with pytest.raises(ValueError):
        validate_bulk_birth(history, **args)


def test_inherited_continuation_preserves_source_prefix_and_origin():
    mesh, _, history = origin()
    source = advance(history, mesh, 1., .3)
    later = advance(source, mesh, 2., .5)
    assert validate_bulk_birth_continuation(later, source) is later
    alternate = advance(advance(history, mesh, .5, .3), mesh, 2., .5)
    with pytest.raises(ValueError, match="accepted history"):
        validate_bulk_birth_continuation(alternate, source)
    with pytest.raises(ValueError, match="accepted history"):
        validate_bulk_birth_continuation(source, later)
    raw = bulk_birth_to_dict(later)
    raw["cohorts"][0]["birth_radius_km"] = 11.
    modified = bulk_birth_from_dict(reseal(raw))
    with pytest.raises(ValueError, match="source cohort"):
        validate_bulk_birth_continuation(modified, source)


def test_invalid_transition_never_mutates_accepted_history():
    mesh, _, history = origin()
    frozen = advance(history, mesh, 1., .3)
    before = bulk_birth_to_dict(frozen)
    with pytest.raises(ValueError):
        advance(frozen, mesh, 1., .4)
    invalid = SimpleNamespace(vertices=mesh.vertices.copy(), faces=mesh.faces.copy())
    invalid.vertices[0] *= .5
    with pytest.raises(ValueError, match="unit vertices"):
        advance(frozen, invalid, 2., .4)
    assert bulk_birth_to_dict(frozen) == before


def test_missing_or_old_archive_requires_replay_instead_of_filling_birth():
    with pytest.raises(ValueError, match="recalculate"):
        bulk_birth_from_dict(None)
    with pytest.raises(ValueError, match="Missing"):
        validate_bulk_birth(None, layer_mass_kg=[], time_myr=0., front_fraction=[])
    _, _, history = origin()
    raw = bulk_birth_to_dict(history)
    raw["format"] = "genesis-bulk-birth-0.0"
    with pytest.raises(ValueError, match="model"):
        bulk_birth_from_dict(reseal(raw))
