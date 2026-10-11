"""Reference metric pullback tests; all declared planes are computational."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tectonics.genesis_volume_checkpoint import _canonical
from tectonics.genesis_volume_creation_law import ProspectiveCreationLaw
from tectonics.genesis_volume_embedded_prestress import (
    EmbeddedPrestressRefusal, embedded_prestress_pullback_to_dict,
    prepare_volume_embedded_prestress, pull_back_embedded_prestress,
    volume_embedded_prestress_to_dict, volume_embedded_prestress_from_dict,
)
from tectonics.genesis_volume_embedded_surface import (
    integrate_embedded_surface_traction, integrate_embedded_surface_virtual_work,
    prepare_cohort_embedded_surface, prepare_volume_embedded_surface,
)
from tectonics.genesis_volume_geometry import ConicalCohortGeometry
from tectonics.genesis_volume_material_sections import prepare_cohort_material_section
from tectonics.genesis_volume_material_stress import bind_volume_material_stress
from tectonics.genesis_volume_prestress_creation import (
    PrestressCreationRefusal, evaluate_prestress_creation,
)
from tectonics.genesis_volume_slip_checkpoint import volume_slip_checkpoint_to_dict
from tectonics.genesis_volume_slip_dynamics import VolumeSlipModel


def _surface(*, deformed=True):
    X = np.array(((0., 0., 10.), (2., 0., 10.), (0., 3., 10.)))
    qlo, qhi, coefficient = .1, .8, .5
    a, b = (1-coefficient*np.array((qlo, qhi)))**(1/3)
    volume = np.linalg.det(X)*(a**3-b**3)/6
    c = ConicalCohortGeometry(17, 0, 2, qlo, qhi, 2*volume, volume,
        tuple(map(tuple, X)), a, b, (), "source-owner", "original-material")
    U = np.zeros((3, 6))
    if deformed:
        U = np.array(((.1, .2, .3, .8, .1, -.2),
                      (-.1, .05, .04, -.4, .4, .6),
                      (.2, -.3, .15, .1, -.2, -.5)))
    n = np.array((1., .7, .2)); n /= np.linalg.norm(n)
    target = np.full(3, 1/3)@((a+b)/2*X+U[:, :3]+((a+b)/2-1)*U[:, 3:])
    section = prepare_cohort_material_section(c, U, n, n@target,
        conical_mass_coefficient=coefficient)
    return prepare_cohort_embedded_surface(section)


def _stress(surface):
    sigma = np.array(((4e6, 2e5, -1e5), (2e5, 2e6, 3e5), (-1e5, 3e5, 1e6)))
    return np.broadcast_to(sigma, (surface.point_count, 3, 3))


def _law():
    return ProspectiveCreationLaw(1e6, 1e8, 1e10, 2e10)


def test_full_deformed_reference_metric_pullback_preserves_material_boundary_force():
    surface = _surface()
    response = pull_back_embedded_prestress(surface, _stress(surface))
    expected = np.einsum("pij,pj->pi", _stress(surface), surface.reference_normal_xyz)
    np.testing.assert_allclose(response.reference_traction_world_pa, expected)
    np.testing.assert_allclose(response.birth_traction_world_pa*surface.current_area_weights_m2[:, None],
        expected*surface.reference_area_weights_m2[:, None], rtol=2e-15, atol=1e-10)
    actual_force = integrate_embedded_surface_traction(surface, response.birth_traction_world_pa,
        area_convention="current")
    np.testing.assert_allclose(actual_force, response.paired_parent_force_n, rtol=3e-15, atol=1e-9)
    # The erroneous current-area formula is detectably different on full P.
    naive = np.einsum("pij,pj->pi", _stress(surface), surface.current_normal_xyz)
    wrong_force = integrate_embedded_surface_traction(surface, naive, area_convention="current")
    assert np.linalg.norm(wrong_force-actual_force) > 1e4


def test_reference_and_birth_area_virtual_work_match_original_modes():
    surface = _surface()
    result = pull_back_embedded_prestress(surface, _stress(surface))
    paired = np.stack((np.sin(np.arange(18)), np.cos(np.arange(18))))*.001
    reference = integrate_embedded_surface_virtual_work(surface, result.reference_traction_world_pa,
        paired, area_convention="reference")
    birth = integrate_embedded_surface_virtual_work(surface, result.birth_traction_world_pa,
        paired, area_convention="current")
    force_work = float(np.sum(result.paired_parent_force_n*paired))
    assert birth == pytest.approx(reference, rel=3e-14, abs=1e-9)
    assert force_work == pytest.approx(reference, rel=3e-14, abs=1e-9)


def test_undeformed_measure_reduces_to_equal_reference_and_birth_traction():
    surface = _surface(deformed=False)
    result = pull_back_embedded_prestress(surface, _stress(surface))
    np.testing.assert_allclose(result.birth_area_ratio, 1., rtol=2e-15)
    np.testing.assert_allclose(result.birth_traction_world_pa, result.reference_traction_world_pa, rtol=2e-15)


def test_birth_local_basis_preserves_work_and_virgin_seed_traction_without_funding():
    surface = _surface()
    result = pull_back_embedded_prestress(surface, _stress(surface), creation_law=_law())
    reconstructed = np.einsum("pij,pj->pi", result.birth_basis_xyz, result.birth_traction_local_pa)
    np.testing.assert_allclose(reconstructed, result.birth_traction_world_pa, rtol=3e-15, atol=1e-9)
    response = evaluate_prestress_creation(result.seed, np.zeros((surface.point_count, 3)),
        np.zeros((surface.point_count, 3)), result.seed.initial_maximum_separation_m)
    np.testing.assert_allclose(response.traction_pa, result.reference_traction_local_pa, rtol=3e-15, atol=1e-9)
    np.testing.assert_array_equal(response.stored_energy_j_m2, np.zeros(surface.point_count))
    np.testing.assert_array_equal(response.fracture_dissipation_j_m2, np.zeros(surface.point_count))
    raw = embedded_prestress_pullback_to_dict(result)
    assert raw["full_separation_cost_on_explicit_surface_j"] == pytest.approx(1e6*surface.total_reference_area_m2)
    assert raw["funding_admitted"] is raw["bulk_reaction_replaced"] is raw["physical_surface_selected"] is False
    assert raw["creation_work_debited_from_Genesis_j"] == 0.


def test_overstrength_seed_is_refused_instead_of_inventing_initial_damage():
    surface = _surface()
    weak = ProspectiveCreationLaw(1e6, 1e4, 1e10, 1e10)
    with pytest.raises(PrestressCreationRefusal, match="peak strength"):
        pull_back_embedded_prestress(surface, _stress(surface), creation_law=weak)


@pytest.mark.parametrize("bad", [True, [[True, 1]], "stress", np.nan, np.inf, [1., 2.]])
def test_invalid_stress_inputs_are_typed_refusals(bad):
    with pytest.raises(EmbeddedPrestressRefusal):
        pull_back_embedded_prestress(_surface(), bad)


def test_nonsymmetric_stress_cannot_silently_change_energy_conjugacy():
    surface = _surface()
    bad = _stress(surface).copy(); bad[:, 0, 1] += 1e5
    with pytest.raises(EmbeddedPrestressRefusal, match="symmetric"):
        pull_back_embedded_prestress(surface, bad)


def test_pullback_snapshot_owns_data_and_detects_changed_force():
    surface = _surface()
    sigma = _stress(surface).copy()
    result = pull_back_embedded_prestress(surface, sigma)
    old = result.stress_world_pa.copy(); sigma[:] = 0
    np.testing.assert_array_equal(result.stress_world_pa, old)
    with pytest.raises(ValueError):
        result.birth_traction_world_pa[0, 0] = 1
    with pytest.raises(EmbeddedPrestressRefusal, match="binding"):
        embedded_prestress_pullback_to_dict(replace(result, paired_parent_force_n=result.paired_parent_force_n+1))


@pytest.fixture(scope="module")
def accepted():
    path = Path(__file__).resolve().parents[1]/(
        "analysis/genesis_cohort_mechanics_validation/evidence/control/material_export.json")
    original = path.read_bytes()
    model = VolumeSlipModel(json.loads(original)); state = model.initial_state()
    before = _canonical(volume_slip_checkpoint_to_dict(model, state))
    observer = bind_volume_material_stress(model, state)
    c = state.geometry.cohorts[0]
    n = np.array((1., 2., 3.)); n /= np.linalg.norm(n)
    target = (c.outer_scale+c.inner_scale)/2*np.asarray(c.top_triangle_m).mean(axis=0)
    surface = prepare_volume_embedded_surface(state.geometry, state.displacement_m, n, n@target,
        face_indices=[c.face_index], quadrature_order=4)
    value = prepare_volume_embedded_prestress(model, state, surface, observer=observer, creation_law=_law())
    yield model, state, observer, surface, value
    assert path.read_bytes() == original
    assert _canonical(volume_slip_checkpoint_to_dict(model, state)) == before


def test_source_binding_carries_actual_calculated_memory_and_full_unchanged_ledgers(accepted):
    model, state, observer, surface, value = accepted
    raw = volume_embedded_prestress_to_dict(value)
    parent = volume_slip_checkpoint_to_dict(model, state)
    assert raw["parent_checkpoint_sha256"] == parent["checkpoint_sha256"]
    assert raw["formation_checkpoint_sha256"] == parent["source_checkpoint_sha256"]
    assert raw["observer_binding_sha256"] == observer.binding_sha256
    assert raw["source_stress_samples"]["cohort_ids"] == surface.cohort_ids.tolist()
    assert raw["inherited_runtime_ledgers"] == {k: v for k, v in parent["runtime_state"].items() if k.endswith("_j")}
    assert raw["parameter_provenance"] == "explicit_new_constitutive_input_not_historical_toughness"
    assert raw["source_historical_surface_creation_energy_law"] is raw["selected_sharp_material_surface"] is None
    assert raw["funding_admitted"] is raw["material_memory_changed"] is raw["clock_advanced"] is False


def test_source_binding_canonical_roundtrip(accepted):
    model, state, observer, surface, value = accepted
    restored = volume_embedded_prestress_from_dict(volume_embedded_prestress_to_dict(value),
        model=model, state=state, surface=surface, observer=observer, creation_law=_law())
    assert volume_embedded_prestress_to_dict(restored) == volume_embedded_prestress_to_dict(value)


def test_source_binding_reuses_literal_plane_coefficients_after_nonidempotent_normalization(accepted):
    model, state, observer, *_ = accepted
    # Actual face79 coefficients: normalization moves n and d by one ULP;
    # repeating it on an already normalized section moves d a second time.
    n = np.array((0.38018781261549783, -0.7060630805716389, 0.5974379912529252))
    c = state.geometry.cohorts[0]
    target = (c.outer_scale+c.inner_scale)/2*np.asarray(c.top_triangle_m).mean(axis=0)
    d = float(n@target)
    surface = prepare_volume_embedded_surface(state.geometry, state.displacement_m, n, d,
        face_indices=[c.face_index], quadrature_order=4)
    np.testing.assert_array_equal(surface.declared_normal_xyz, n)
    assert surface.declared_offset_m == d
    value = prepare_volume_embedded_prestress(model, state, surface, observer=observer)
    raw = volume_embedded_prestress_to_dict(value)
    restored = volume_embedded_prestress_from_dict(raw, model=model, state=state,
        surface=surface, observer=observer)
    assert volume_embedded_prestress_to_dict(restored) == raw


@pytest.mark.parametrize("key,value", [("funding_admitted", True),
    ("source_historical_surface_creation_energy_law", {"Gc": 1e6}),
    ("time_myr", 1.), ("observer_binding_sha256", "0"*64)])
def test_resealed_source_funding_history_and_context_forgeries_are_refused(accepted, key, value):
    model, state, observer, surface, bound = accepted
    raw = volume_embedded_prestress_to_dict(bound); raw[key] = value
    raw["embedded_prestress_sha256"] = hashlib.sha256(_canonical({k: v for k, v in raw.items()
        if k != "embedded_prestress_sha256"})).hexdigest()
    with pytest.raises(EmbeddedPrestressRefusal, match="recomputation"):
        volume_embedded_prestress_from_dict(raw, model=model, state=state,
            surface=surface, observer=observer, creation_law=_law())


def test_standalone_manufactured_surface_cannot_claim_accepted_planet_binding(accepted):
    model, state, *_ = accepted
    with pytest.raises(EmbeddedPrestressRefusal, match="original source"):
        prepare_volume_embedded_prestress(model, state, _surface())
