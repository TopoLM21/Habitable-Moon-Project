"""Actual formation memory bound to an explicitly computational finite seed."""
from dataclasses import replace
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tectonics.genesis_volume_checkpoint import _canonical
from tectonics.genesis_volume_creation_law import ProspectiveCreationLaw
from tectonics.genesis_volume_finite_seed import prepare_volume_finite_seed, prepare_cohort_finite_seed
from tectonics.genesis_volume_geometry import ConicalCohortGeometry
from tectonics.genesis_volume_material_sections import prepare_cohort_material_section
from tectonics.genesis_volume_material_stress import bind_volume_material_stress
from tectonics.genesis_volume_nucleation import ProspectiveVolumeNucleationLaw
from tectonics.genesis_volume_seed_insertion import (
    BoundVolumeSeedInsertion, VolumeSeedInsertionRefusal,
    prepare_volume_seed_insertion, bind_proposed_volume_seed_insertion,
    volume_seed_insertion_to_dict, volume_seed_insertion_from_dict,
    proposed_volume_seed_insertion_from_dict, _validate_material_positions,
)
from tectonics.genesis_volume_slip_checkpoint import volume_slip_checkpoint_to_dict
from tectonics.genesis_volume_slip_dynamics import VolumeSlipModel


def _seal(raw):
    raw = copy.deepcopy(raw)
    raw["seed_insertion_sha256"] = hashlib.sha256(_canonical({k: v for k, v in raw.items()
        if k != "seed_insertion_sha256"})).hexdigest()
    return raw


@pytest.fixture(scope="module")
def accepted():
    path = Path(__file__).resolve().parents[1]/(
        "analysis/genesis_cohort_mechanics_validation/evidence/control/material_export.json")
    original = path.read_bytes()
    model = VolumeSlipModel(json.loads(original))
    state = model.initial_state()
    before = _canonical(volume_slip_checkpoint_to_dict(model, state))
    c = state.geometry.cohorts[0]
    N = np.full(3, 1./3.)
    lam = (c.outer_scale+c.inner_scale)/2.
    X = np.asarray(c.top_triangle_m)
    dofs = (6*np.asarray(state.geometry.faces[c.face_index])[:, None]+np.arange(6)).ravel()
    U = state.displacement_m[dofs].reshape(3, 6)
    center = N@(lam*X+U[:, :3]+(lam-1)*U[:, 3:])
    normal = np.array((1., 2., 3.)); normal /= np.linalg.norm(normal)
    # A genuinely resolved small ball wholly within this manufactured probe's
    # original material band. This is not the source-selected fracture radius.
    radius = .12*(c.outer_scale-c.inner_scale)*np.linalg.norm(N@X)
    seed = prepare_volume_finite_seed(state.geometry, state.displacement_m, normal,
        normal@center, center, radius, cohort_id=c.id, quadrature_order=4)
    law = ProspectiveCreationLaw(1e6, 1e8, 1e10, 1e10)
    observer = bind_volume_material_stress(model, state)
    insertion = prepare_volume_seed_insertion(model, state, seed, observer=observer,
        creation_law=law, partition_quadrature_order=4)
    yield model, state, seed, observer, law, insertion
    assert path.read_bytes() == original
    assert _canonical(volume_slip_checkpoint_to_dict(model, state)) == before


def test_tied_insertion_preserves_formation_memory_ledgers_and_physical_time(accepted):
    model, state, seed, observer, law, insertion = accepted
    raw = volume_seed_insertion_to_dict(insertion)
    parent = volume_slip_checkpoint_to_dict(model, state)
    assert raw["parent_checkpoint_sha256"] == parent["checkpoint_sha256"]
    assert raw["observer_binding_sha256"] == observer.binding_sha256
    assert raw["inherited_runtime_ledgers"] == {k: v for k, v in parent["runtime_state"].items()
                                              if k.endswith("_j")}
    assert raw["finite_positive_surface_area_prepared"] is True
    assert raw["constrained_bulk_modes_prepared"] is True
    assert raw["released_physical_dofs"] == raw["creation_work_debited_from_Genesis_j"] == 0
    assert raw["material_memory_changed"] is raw["clock_advanced"] is False
    assert raw["additional_memory_needed_before_nonzero_enrichment"] is True
    assert raw["surface_force_is_discrete_bulk_reaction"] is False


def test_surface_virtual_force_is_dual_to_compact_jump_in_both_original_mode_families(accepted):
    raw = volume_seed_insertion_to_dict(accepted[-1])
    trace = np.asarray(raw["surface_enrichment_trace"])
    force = np.asarray(raw["surface_enrichment_force_n"])
    traction = np.asarray(raw["prestress_pullback"]["reference_traction_world_pa"])
    weights = accepted[2].surface.reference_area_weights_m2
    for family in (slice(0, 3), slice(3, 6)):
        a = np.zeros((3, 6)); a[:, family] = np.sin(np.arange(9).reshape(3, 3)+.4)*1e-6
        jump = np.einsum("pai,i->pa", trace, a.ravel())
        work = float(np.einsum("pa,pa,p->", traction, jump, weights))
        assert np.linalg.norm(jump) > 0
        assert float(force@a.ravel()) == pytest.approx(work, rel=2e-13, abs=1e-7)
    # A local enrichment has one compact trace, rather than independent whole
    # cohort fields with an unintended discontinuity on the intact ligament.
    assert trace.shape[2] == 18
    assert np.all(np.asarray(raw["surface_enrichment_bubble"]) > 0)


def test_complete_source_reconstruction_accepts_untampered_restart_descriptor(accepted):
    model, state, seed, observer, law, insertion = accepted
    restored = volume_seed_insertion_from_dict(volume_seed_insertion_to_dict(insertion),
        model=model, state=state, finite_seed=seed, observer=observer,
        creation_law=law, partition_quadrature_order=4)
    assert restored.payload_json == insertion.payload_json


def test_bulk_enrichment_null_modes_cannot_be_mistaken_for_funded_crack_opening(accepted):
    raw = volume_seed_insertion_to_dict(accepted[-1])
    modes = np.asarray(raw["analytic_surface_null_modes"])
    trace = np.asarray(raw["surface_enrichment_trace"])
    weights = accepted[2].surface.reference_area_weights_m2
    weighted = (np.sqrt(weights)[:, None, None]*trace).reshape(-1, 18)
    assert modes.shape == (18, 3)
    np.testing.assert_allclose(modes.T@modes, np.eye(3), rtol=0, atol=3e-15)
    # Three continuous refinements have no interface opening, but have a
    # nonzero regular bulk strain and could release elastic energy if freed.
    assert np.linalg.norm(weighted@modes) < 1e-11*np.linalg.norm(weighted)
    B = np.asarray(raw["embedded_partition"]["local"]["enrichment_strain_operator"])
    assert np.linalg.norm(np.einsum("pai,ij->paj", B, modes)) > 0
    assert raw["surface_jump_trace_rank"] < 18
    assert raw["surface_null_modes_remain_constrained"] is True
    assert raw["null_mode_release_needs_separate_work_admission"] is True
    assert raw["volume_rank_does_not_certify_surface_opening"] is True


@pytest.mark.parametrize("key,value", [
    ("surface_enrichment_force_n", [0.]*18),
    ("observer_binding_sha256", "0"*64),
    ("creation_work_debited_from_Genesis_j", 1.),
    ("time_myr", 99.),
    ("additional_memory_needed_before_nonzero_enrichment", False),
])
def test_resealed_descriptor_cannot_replace_math_memory_or_creation_funding(accepted, key, value):
    model, state, seed, observer, law, insertion = accepted
    raw = volume_seed_insertion_to_dict(insertion)
    assert raw[key] != value
    raw[key] = value
    with pytest.raises(VolumeSeedInsertionRefusal, match="complete accepted-source recomputation"):
        volume_seed_insertion_from_dict(_seal(raw), model=model, state=state, finite_seed=seed,
            observer=observer, creation_law=law, partition_quadrature_order=4)


def test_geometry_cannot_be_transplanted_from_a_different_ball_even_with_same_plane(accepted):
    model, state, seed, observer, law, insertion = accepted
    s = seed.surface
    other = prepare_volume_finite_seed(state.geometry, state.displacement_m,
        s.declared_normal_xyz, s.declared_offset_m, seed.seed_center_world_m,
        seed.seed_radius_m*.8, cohort_id=seed.selected_cohort_id, quadrature_order=4)
    with pytest.raises(VolumeSeedInsertionRefusal, match="complete accepted-source recomputation"):
        volume_seed_insertion_from_dict(volume_seed_insertion_to_dict(insertion),
            model=model, state=state, finite_seed=other, observer=observer,
            creation_law=law, partition_quadrature_order=4)


def test_saved_observer_is_checked_against_full_accepted_memory(accepted):
    model, state, seed, observer, law, insertion = accepted
    changed = replace(observer, elastic_inverse_scale=observer.elastic_inverse_scale+1e-6)
    with pytest.raises(ValueError):
        prepare_volume_seed_insertion(model, state, seed, observer=changed,
            creation_law=law, partition_quadrature_order=4)


@pytest.mark.parametrize("configured", [False, True])
def test_proposed_entry_retains_actual_genesis_gate_without_fabricated_seed(accepted, configured):
    model, state, seed, observer, creation, insertion = accepted
    law = ProspectiveVolumeNucleationLaw(12e6, 4e6, .6, 100., creation) if configured else None
    bound = bind_proposed_volume_seed_insertion(model, state, law,
        quadrature_order=4, partition_quadrature_order=4)
    raw = volume_seed_insertion_to_dict(bound)
    assert raw["status"] == "no_calculated_activation_event"
    assert raw["insertion"] is None
    assert raw["nucleation_proposal"]["candidate"] is None
    assert raw["finite_positive_surface_area_prepared"] is False
    assert raw["constrained_bulk_modes_prepared"] is False
    raw["insertion"] = volume_seed_insertion_to_dict(insertion)
    with pytest.raises(VolumeSeedInsertionRefusal, match="accepted source birth gate"):
        proposed_volume_seed_insertion_from_dict(_seal(raw), model=model, state=state,
            law=law, quadrature_order=4, partition_quadrature_order=4)


def test_immutable_wrapper_checks_seal_and_denies_unfunded_release(accepted):
    insertion = accepted[-1]
    with pytest.raises(VolumeSeedInsertionRefusal, match="immutable source"):
        volume_seed_insertion_to_dict(replace(insertion, binding_sha256="0"*64))
    raw = volume_seed_insertion_to_dict(insertion); raw["physical_release_admitted"] = True
    payload = _canonical(_seal(raw)).decode()
    forged = BoundVolumeSeedInsertion(payload, hashlib.sha256(payload.encode()).hexdigest())
    with pytest.raises(VolumeSeedInsertionRefusal, match="cannot admit release"):
        volume_seed_insertion_to_dict(forged)


@pytest.fixture(scope="module")
def accepted_cancellation_positions():
    """Only evaluate literal accepted material coordinates, without replay.

    Face 79 has a small z coordinate produced by cancelling million-metre
    operands. The section and the stress observer use different sum orders.
    This fixture creates no runtime state or new elastic memory.
    """
    root = Path(__file__).resolve().parents[1]
    path = root/"analysis/genesis_volume_creation_validation/evidence/actual_continuation/direct_after_second.json"
    original = path.read_bytes()
    assert hashlib.sha256(original).hexdigest() == "62579a0bad128830a66e2f44579cf2b1f5d3835b691225b8d431102f3e4486d7"
    parent = json.loads(original)
    geometry = parent["runtime_state"]["geometry"]
    raw = min((c for c in geometry["cohorts"] if c["face_index"] == 79), key=lambda c: c["id"])
    cohort = ConicalCohortGeometry(**{**raw,
        "top_triangle_m": tuple(map(tuple, raw["top_triangle_m"])),
        "tetrahedra_m": tuple(tuple(tuple(v) for v in tetra) for tetra in raw["tetrahedra_m"])})
    vertices = np.asarray(geometry["faces"][79], int)
    X = np.asarray(raw["top_triangle_m"])
    pose = np.asarray(parent["runtime_state"]["displacement_m"]).reshape(-1, 6)[vertices]
    A, P = pose[:, :3], pose[:, 3:]
    N, lam = np.full(3, 1/3.), .5*(raw["inner_scale"]+raw["outer_scale"])
    center = N@(lam*(X+P)+(A-P))
    normal = np.asarray(json.loads((root/"analysis/genesis_volume_location_validation/evidence/actual_control/supplied_current_plane_face_79.json").read_bytes())["normal_xyz"])
    radius = .12*(raw["outer_scale"]-raw["inner_scale"])*np.linalg.norm(N@(X+P))
    coefficient = 6*geometry["column_mass_kg"][79]/(geometry["density_kg_m3"]*np.linalg.det(X))
    section = prepare_cohort_material_section(cohort, pose, normal, float(normal@center),
        conical_mass_coefficient=coefficient, cohort_index=79)
    surface = prepare_cohort_finite_seed(section, center, radius, quadrature_order=8).surface
    # Reconstruct the observer's declared q-chart placement independently of
    # the comparison guard; no stress samples or accepted history are invented.
    count = surface.point_count
    radial = np.cbrt(1-coefficient*surface.mass_coordinate_q)
    triangle = np.repeat(X[None], count, axis=0)
    nodes = np.repeat(pose[None], count, axis=0)
    angular = surface.barycentric
    samples = SimpleNamespace(radial_scale=radial,
        reference_position_m=radial[:, None]*np.einsum("ni,nij->nj", angular, triangle),
        current_position_m=radial[:, None]*np.einsum("ni,nij->nj", angular, triangle+nodes[..., 3:])
            +np.einsum("ni,nij->nj", angular, nodes[..., :3]-nodes[..., 3:]))
    yield surface, samples
    assert path.read_bytes() == original


def test_genuine_material_placement_accepts_cartesian_cancellation_roundoff(accepted_cancellation_positions):
    surface, samples = accepted_cancellation_positions
    np.testing.assert_array_equal(samples.radial_scale, surface.radial_scale)
    old_bound = 128*np.finfo(float).eps*np.maximum(np.abs(surface.current_position_m), 1.)
    assert np.any(np.abs(samples.current_position_m-surface.current_position_m) > old_bound)
    assert np.max(np.abs(samples.current_position_m-surface.current_position_m)) < 2e-9
    _validate_material_positions(surface, samples)


@pytest.mark.parametrize("field", ["current_position_m", "reference_position_m"])
def test_roundoff_guard_rejects_changed_material_position(accepted_cancellation_positions, field):
    surface, samples = accepted_cancellation_positions
    changed = copy.deepcopy(samples)
    getattr(changed, field)[0, 2] += 1e-6
    with pytest.raises(VolumeSeedInsertionRefusal, match="material positions differ"):
        _validate_material_positions(surface, changed)


def test_roundoff_guard_rejects_changed_material_radial_coordinate(accepted_cancellation_positions):
    surface, samples = accepted_cancellation_positions
    changed = copy.deepcopy(samples)
    changed.radial_scale[0] += 1e-8
    with pytest.raises(VolumeSeedInsertionRefusal, match="material radial coordinates differ"):
        _validate_material_positions(surface, changed)
