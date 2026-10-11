"""Explicit material points preserve calculated Genesis memory and its source.

The runtime fixture is the unchanged genuine formation export. Manufactured
tensor fields below test the already declared spatial representation only;
they do not create a planetary history or a fracture-location law.
"""
from dataclasses import FrozenInstanceError, replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tectonics.genesis_volume_checkpoint import _canonical
from tectonics.genesis_volume_material_stress import (
    VolumeMaterialStressRefusal, _pointwise_fields,
    bind_volume_material_stress, evaluate_volume_material_stress,
    volume_material_stress_observer_to_dict, validate_volume_material_stress_observer_payload,
    volume_material_stress_to_dict, volume_material_stress_from_dict)
from tectonics.genesis_volume_operator import ANGULAR_BARYCENTRIC
from tectonics.genesis_volume_slip_checkpoint import volume_slip_checkpoint_to_dict
from tectonics.genesis_volume_slip_dynamics import VolumeSlipModel


@pytest.fixture(scope="module")
def accepted():
    path = Path(__file__).resolve().parents[1]/(
        "analysis/genesis_cohort_mechanics_validation/evidence/control/material_export.json")
    original = path.read_bytes()
    model = VolumeSlipModel(json.loads(original))
    state = model.initial_state()
    before = _canonical(volume_slip_checkpoint_to_dict(model, state))
    observer = bind_volume_material_stress(model, state)
    yield model, state, observer
    assert path.read_bytes() == original
    assert _canonical(volume_slip_checkpoint_to_dict(model, state)) == before


def _sample(accepted):
    observer = accepted[2]
    c = observer.geometry.cohorts[0]
    return evaluate_volume_material_stress(observer, c.id, (.2, .3, .5),
        .5*(c.q_lo+c.live_q_hi), normal_world=(0., 0., 1.))


def test_parent_identity_and_entire_runtime_are_bound_without_reinitialization(accepted):
    model, state, observer = accepted
    parent = volume_slip_checkpoint_to_dict(model, state)
    raw = volume_material_stress_observer_to_dict(observer)
    assert observer.parent_checkpoint_sha256 == parent["checkpoint_sha256"]
    assert observer.formation_checkpoint_sha256 == parent["source_checkpoint_sha256"]
    assert observer.time_myr == state.time_myr
    assert observer.accepted_steps == state.mechanics.accepted_steps
    assert observer.accepted_step_sizes_myr == tuple(state.accepted_step_sizes_myr)
    assert "source_packet" not in raw and "source_packet_json" not in raw["geometry"]
    assert raw["source_creation_energy_law"] is raw["selected_sharp_material_surface"] is None
    for flag in ("history_reinitialized", "clock_advanced", "activation_accepted",
            "physical_creation_admitted", "physical_release_admitted",
            "physical_fault_contact_admission_claimed", "historical_three_dimensional_memory_recovered",
            "instantaneous_future_birth_traction_claimed"):
        assert raw[flag] is False
    assert raw["creation_work_debited_from_Genesis_j"] == 0


def test_all_source_cohort_memories_and_coefficients_are_retained(accepted):
    model, state, observer = accepted
    assembly, _ = model._assembly(state.geometry, state.fault_history.damage)
    np.testing.assert_array_equal(observer.e0, [c.strain0 for c in state.mechanics.cohorts])
    np.testing.assert_array_equal(observer.e1, [c.strain1 for c in state.mechanics.cohorts])
    np.testing.assert_array_equal(observer.cohort_ids, assembly.cohort_ids)
    np.testing.assert_array_equal(observer.face_indices, assembly.face_indices)
    np.testing.assert_array_equal(observer.basis_xyz, assembly.basis_xyz)
    np.testing.assert_array_equal(observer.elasticity_pa, assembly.elasticity_pa)
    np.testing.assert_array_equal(observer.displacement_m, state.displacement_m)
    assert np.linalg.norm(observer.e0) > 0  # formed material is not reset stress-free


def test_initial_formation_constant_depth_memory_reconstructs_at_every_explicit_corner(accepted):
    observer = accepted[2]
    cohorts = observer.geometry.cohorts
    ids = np.repeat(observer.cohort_ids, 6)
    N = np.tile(np.tile(np.eye(3), (2, 1)), (len(cohorts), 1))
    q = np.concatenate([np.repeat((c.q_lo, c.live_q_hi), 3) for c in cohorts])
    samples = evaluate_volume_material_stress(observer, ids, N, q)
    # The unchanged formation adapter explicitly imports a depth-constant
    # plane-stress elastic field. All later nonconstant memory remains retained.
    np.testing.assert_array_equal(observer.e1, np.zeros_like(observer.e1))
    expected = np.repeat(observer.e0[:, 0], 6, axis=0)
    np.testing.assert_allclose(samples.elastic_strain_local6, expected, rtol=2e-15, atol=1e-19)
    expected_stress = np.einsum("nij,nj->ni", np.repeat(observer.elasticity_pa, 6, axis=0), expected)
    np.testing.assert_allclose(samples.stress_local6, expected_stress, rtol=3e-14, atol=1e-7)
    np.testing.assert_allclose(samples.current_position_m, samples.reference_position_m, atol=1e-9)
    np.testing.assert_array_equal(samples.radial_scale,
        np.concatenate([np.repeat((c.outer_scale, c.inner_scale), 3) for c in cohorts]))


def test_material_chart_is_mass_coordinate_not_equal_depth(accepted):
    observer = accepted[2]
    c = observer.geometry.cohorts[0]
    q = np.linspace(c.q_lo, c.live_q_hi, 5)
    N = np.repeat([[.2, .3, .5]], len(q), axis=0)
    sample = evaluate_volume_material_stress(observer, c.id, N, q)
    X = np.asarray(c.top_triangle_m)
    coefficient = 6*observer.geometry.column_mass_kg[c.face_index]/(
        observer.geometry.density_kg_m3*np.linalg.det(X))
    np.testing.assert_allclose(sample.radial_scale**3, 1-coefficient*q, atol=1e-15)
    np.testing.assert_allclose(sample.reference_position_m,
        sample.radial_scale[:, None]*(N@X), rtol=1e-15)
    assert sample.material_ids == (c.material_id,)*5
    assert sample.owner_ids == (c.owner_id,)*5
    np.testing.assert_array_equal(sample.cohort_ids, np.repeat(c.id, 5))


def test_source_pointwise_stress_rotation_and_probe_traction(accepted):
    sample = _sample(accepted)
    assert sample.stress_world_tensor.shape == (1, 3, 3)
    np.testing.assert_allclose(sample.stress_world_tensor,
        sample.stress_world_tensor.swapaxes(-1, -2), rtol=2e-15, atol=1e-8)
    np.testing.assert_array_equal(sample.traction_world_pa, sample.stress_world_tensor[:, :, 2])
    assert sample.elastic_energy_density_j_m3[0] == pytest.approx(
        .5*sample.elastic_strain_local6[0]@sample.stress_local6[0], rel=1e-15)
    raw = volume_material_stress_to_dict(sample)
    assert raw["supplied_normal_is_fracture_orientation"] is False
    assert not raw["instantaneous_future_birth_traction_claimed"]


def test_probe_normal_sign_and_vectorization_are_preserved(accepted):
    observer = accepted[2]
    c = observer.geometry.cohorts[0]
    sample = evaluate_volume_material_stress(observer, c.id,
        np.repeat([[.2, .3, .5]], 2, axis=0), c.q_lo,
        normal_world=((0., 1., 0.), (0., -1., 0.)))
    np.testing.assert_array_equal(sample.traction_world_pa[0], -sample.traction_world_pa[1])
    without = evaluate_volume_material_stress(observer, c.id, (.2, .3, .5), c.q_lo)
    assert without.normal_world is without.traction_world_pa is None


def test_manufactured_affine_angular_inverse_radius_field_and_engineering_shear():
    # This is an analytical interpolation fixture, independent of any planet.
    N = np.array(((.15, .25, .6), (.7, .2, .1), (.05, .55, .4)))
    radial = np.array((.91, .99, .97))
    def constant_part(b):
        return 1e-4*np.column_stack((b[:, 0]+2*b[:, 1]-b[:, 2],
            b[:, 2], -.2+b[:, 0], .4+b[:, 1], -.3*b[:, 0], .2*b[:, 2]))
    def inverse_part(b):
        return 1e-4*np.column_stack((.3+.2*b[:, 2], -.1*b[:, 1],
            .4*b[:, 0], .1*b[:, 0], .3*b[:, 2], -.2*b[:, 1]))
    e0 = np.repeat(constant_part(ANGULAR_BARYCENTRIC)[None], len(N), axis=0)
    e1 = np.repeat(inverse_part(ANGULAR_BARYCENTRIC)[None], len(N), axis=0)
    young, nu = 70e9, .25
    mu, lame = young/(2*(1+nu)), young*nu/((1+nu)*(1-2*nu))
    C = np.eye(6)*mu
    C[:3, :3] = lame
    C[np.arange(3), np.arange(3)] += 2*mu
    angle = np.pi/6
    Q = np.array(((np.cos(angle), -np.sin(angle), 0.),
                  (np.sin(angle), np.cos(angle), 0.), (0., 0., 1.)))
    result = _pointwise_fields(e0, e1, np.repeat(C[None], len(N), axis=0),
        np.repeat(Q[None], len(N), axis=0), N, radial)
    expected_e = constant_part(N)+inverse_part(N)/radial[:, None]
    expected_s = expected_e*mu
    expected_s[:, :3] = 2*mu*expected_e[:, :3]+lame*expected_e[:, :3].sum(axis=1)[:, None]
    np.testing.assert_allclose(result[0], expected_e, rtol=3e-15, atol=1e-19)
    np.testing.assert_allclose(result[3], expected_s, rtol=3e-15, atol=1e-8)
    np.testing.assert_allclose(result[1][:, 0, 1], .5*expected_e[:, 3], rtol=3e-15)
    np.testing.assert_allclose(result[4][:, 0, 1], expected_s[:, 3], rtol=3e-15)
    np.testing.assert_allclose(result[5], np.array([Q@s@Q.T for s in result[4]]), rtol=3e-15, atol=1e-8)
    np.testing.assert_allclose(np.einsum("nij,nij->n", result[2], result[5]),
        np.einsum("ni,ni->n", expected_e, expected_s), rtol=3e-15)


def test_owned_bytes_and_frozen_dataclasses_prevent_snapshot_mutation(accepted):
    observer = accepted[2]
    sample = _sample(accepted)
    for a in (observer.e0, observer.e1, observer.displacement_m, observer.elasticity_pa,
              sample.current_position_m, sample.stress_world_tensor, sample.normal_world):
        with pytest.raises(ValueError):
            a.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        observer.time_myr = 0
    with pytest.raises(FrozenInstanceError):
        sample.sample_sha256 = "0"*64


def test_parent_array_changes_do_not_mutate_owned_observer_snapshot(accepted):
    # Only a temporary copy of the parent's mutable array is passed to the
    # dataclass copy; the genuine state remains untouched.
    observer = accepted[2]
    mutable = observer.displacement_m.copy()
    copied = replace(observer, displacement_m=mutable)
    mutable[:] = 7
    np.testing.assert_array_equal(copied.displacement_m, observer.displacement_m)
    assert not np.any(copied.displacement_m == 7)


def test_contextual_observer_and_sample_json_roundtrip(accepted):
    observer = accepted[2]
    raw = json.loads(_canonical(volume_material_stress_observer_to_dict(observer)))
    assert _canonical(validate_volume_material_stress_observer_payload(raw,
        expected_observer=observer)) == _canonical(raw)
    sample = _sample(accepted)
    raw_sample = json.loads(_canonical(volume_material_stress_to_dict(sample)))
    restored = volume_material_stress_from_dict(raw_sample, observer=observer)
    assert _canonical(volume_material_stress_to_dict(restored)) == _canonical(raw_sample)


@pytest.mark.parametrize("field", ("stress_world_tensor", "current_position_m",
    "elastic_strain_local6", "radial_scale", "traction_world_pa", "cohort_ids",
    "parent_checkpoint_sha256", "physical_release_admitted", "source_creation_energy_law",
    "selected_sharp_material_surface", "creation_work_debited_from_Genesis_j", "unexpected_field"))
def test_resealed_sample_substitutions_do_not_replace_the_source(accepted, field):
    observer = accepted[2]
    raw = volume_material_stress_to_dict(_sample(accepted))
    if field in ("stress_world_tensor", "current_position_m", "elastic_strain_local6", "traction_world_pa"):
        raw[field] = np.asarray(raw[field])+1
        raw[field] = raw[field].tolist()
    elif field == "radial_scale":
        raw[field][0] *= .9
    elif field == "cohort_ids":
        raw[field][0] += 1
    elif field == "parent_checkpoint_sha256":
        raw[field] = "0"*64
    elif field in ("source_creation_energy_law", "selected_sharp_material_surface"):
        raw[field] = {"fabricated": True}
    else:
        raw[field] = 1
    raw["sample_sha256"] = hashlib.sha256(_canonical({
        k: v for k, v in raw.items() if k != "sample_sha256"})).hexdigest()
    with pytest.raises(VolumeMaterialStressRefusal, match="full accepted-source"):
        volume_material_stress_from_dict(raw, observer=observer)


@pytest.mark.parametrize("field", ("elastic_constant", "displacement_m", "face_damage",
    "parent_checkpoint_sha256", "physical_creation_admitted", "selected_sharp_material_surface"))
def test_resealed_observer_payload_requires_recomputed_context(accepted, field):
    observer = accepted[2]
    raw = volume_material_stress_observer_to_dict(observer)
    if field in ("elastic_constant", "displacement_m", "face_damage"):
        raw[field] = (np.asarray(raw[field])+1).tolist()
    elif field == "parent_checkpoint_sha256":
        raw[field] = "0"*64
    else:
        raw[field] = True
    raw["binding_sha256"] = hashlib.sha256(_canonical({
        k: v for k, v in raw.items() if k != "binding_sha256"})).hexdigest()
    with pytest.raises(VolumeMaterialStressRefusal, match="full accepted-source"):
        validate_volume_material_stress_observer_payload(raw, expected_observer=observer)


@pytest.mark.parametrize("N", ((-1e-8, .5, .50000001), (1.1, 0., -.1),
    (.2, .3, .4), (True, 0., 0.), (np.nan, 0., 1.), (), ((1., 0.),)))
def test_invalid_material_angular_coordinates_refuse(accepted, N):
    observer = accepted[2]
    c = observer.geometry.cohorts[0]
    with pytest.raises(VolumeMaterialStressRefusal):
        evaluate_volume_material_stress(observer, c.id, N, c.q_lo)


@pytest.mark.parametrize("which", ("below", "above", "nan", "boolean", "unknown_id",
    "float_id", "boolean_id", "mismatched_count", "nonunit_normal", "bad_normal_shape"))
def test_outside_material_or_invalid_probe_refuses(accepted, which):
    observer = accepted[2]
    c = observer.geometry.cohorts[0]
    ids, q, normal = c.id, c.q_lo, None
    if which == "below": q = np.nextafter(c.q_lo, -np.inf)
    elif which == "above": q = np.nextafter(c.live_q_hi, np.inf)
    elif which == "nan": q = np.nan
    elif which == "boolean": q = False
    elif which == "unknown_id": ids = int(observer.cohort_ids[-1])+1
    elif which == "float_id": ids = float(c.id)
    elif which == "boolean_id": ids = True
    elif which == "mismatched_count": q = (c.q_lo, c.q_lo)
    elif which == "nonunit_normal": normal = (0., 0., 2.)
    elif which == "bad_normal_shape": normal = (1., 0.)
    with pytest.raises(VolumeMaterialStressRefusal):
        evaluate_volume_material_stress(observer, ids, (.2, .3, .5), q, normal_world=normal)


def test_changed_runtime_snapshot_and_sample_integrity_are_refused(accepted):
    observer = accepted[2]
    c = observer.geometry.cohorts[0]
    altered = replace(observer, elastic_constant=observer.e0+1e-3)
    with pytest.raises(VolumeMaterialStressRefusal, match="immutable binding"):
        evaluate_volume_material_stress(altered, c.id, (.2, .3, .5), c.q_lo)
    sample = _sample(accepted)
    with pytest.raises(VolumeMaterialStressRefusal, match="immutable binding"):
        volume_material_stress_to_dict(replace(sample, stress_world_tensor=sample.stress_world_tensor+1))


@pytest.mark.parametrize("model,state", ((None, None), (object(), object())))
def test_runtime_requires_typed_accepted_source(model, state):
    with pytest.raises(VolumeMaterialStressRefusal, match="accepted Genesis"):
        bind_volume_material_stress(model, state)


def test_seeded_initial_damage_cannot_be_bound_as_authentic_formation(accepted):
    model, state, _ = accepted
    damage = state.fault_history.damage.copy()
    damage[0] = .7
    altered = replace(state, fault_history=replace(state.fault_history, damage=damage))
    with pytest.raises(ValueError, match="literal Genesis"):
        bind_volume_material_stress(model, altered)
