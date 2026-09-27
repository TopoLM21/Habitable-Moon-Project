"""Reject deforming fragments and preserve contact observations during audit."""
from copy import deepcopy
from dataclasses import fields, replace
import json

import numpy as np
import pytest

from tectonics.genesis_contact import ContactModel, save_contact_checkpoint
from tectonics.genesis_handoff import (HandoffScreenParameters, assess_contact_handoff,
                                      component_labels, fit_rigid_regions)
from tectonics.mesh import build_icosphere
from tectonics.genesis_seams import split_mesh
from test_genesis_contact import _source_bytes


def _hemispheres():
    mesh = build_icosphere(1)
    labels = (mesh.centroids[:, 2] > 0).astype(np.int64)
    return mesh, labels


def test_recovers_independent_euler_vectors_with_physical_units():
    mesh, labels = _hemispheres()
    radius = 5300.
    omega = np.asarray([[.01, -.02, .03], [-.03, .01, .02]])
    velocity = np.cross(omega[labels], mesh.centroids)*radius
    result = fit_rigid_regions(mesh, labels, velocity, radius)
    np.testing.assert_allclose(result["omega_rad_per_myr"], omega, atol=1e-15)
    np.testing.assert_allclose(result["residual_velocity_km_myr"], 0., atol=1e-12)
    assert all(r["fit_rank"] == 3 for r in result["regions"])


def test_large_common_rotation_does_not_hide_deformation():
    mesh, labels = _hemispheres()
    radius = 5000.
    r = mesh.centroids
    base = np.cross(np.array([[2., 0., 0.], [-1., 1., 0.]])[labels], r)
    deformation = np.cross(r, np.array([1., 2., -1.]))*r[:, :1]
    first = fit_rigid_regions(mesh, labels, base+deformation, radius)
    common = np.array([3000., -1000., 4000.])
    rotated = fit_rigid_regions(mesh, labels, base+deformation+np.cross(common, r), radius)
    np.testing.assert_allclose([x["rigid_residual_fraction"] for x in first["regions"]],
                               [x["rigid_residual_fraction"] for x in rotated["regions"]], rtol=2e-11)
    np.testing.assert_allclose(rotated["omega_rad_per_myr"], first["omega_rad_per_myr"]+common/radius,
                               rtol=1e-12, atol=1e-12)


def test_one_cell_rotation_fit_is_underdetermined_and_cannot_be_a_plate():
    mesh, _ = _hemispheres()
    labels = np.ones(mesh.cell_count, dtype=np.int64)
    labels[0] = 0
    result = fit_rigid_regions(mesh, labels, np.cross([2., 3., 4.], mesh.centroids), 5000.)
    assert result["regions"][0]["fit_rank"] == 2


def test_static_shell_is_finite_but_not_evidence_of_relative_motion():
    mesh, labels = _hemispheres()
    result = fit_rigid_regions(mesh, labels, np.zeros((mesh.cell_count, 3)), 5000.)
    assert all(r["relative_rms_speed_km_myr"] == 0 for r in result["regions"])
    assert all(r["rigid_residual_fraction"] == 0 for r in result["regions"])
    json.dumps(result["regions"], allow_nan=False)


@pytest.mark.parametrize("invalid", ["negative_labels", "float_labels", "sparse_labels", "radial", "nan", "radius"])
def test_invalid_fits_are_rejected(invalid):
    mesh, labels = _hemispheres()
    velocity = np.zeros((mesh.cell_count, 3))
    radius = 5000.
    if invalid == "negative_labels": labels -= 1
    if invalid == "float_labels": labels = labels.astype(float)
    if invalid == "sparse_labels": labels *= 2
    if invalid == "radial": velocity = mesh.centroids.copy()
    if invalid == "nan": velocity[0, 0] = np.nan
    if invalid == "radius": radius = 0.
    with pytest.raises(ValueError):
        fit_rigid_regions(mesh, labels, velocity, radius)


def test_components_follow_actual_cut_network():
    mesh, original = _hemispheres()
    cuts = np.array([[u, v] for a, b, u, v in mesh.shared_edges if original[a] != original[b]])
    separated = split_mesh(mesh, cuts)
    labels = component_labels(separated.mesh)
    assert len(np.unique(labels)) == 2
    assert np.all(labels[original == 0] == labels[np.flatnonzero(original == 0)[0]])
    assert np.all(labels[original == 1] == labels[np.flatnonzero(original == 1)[0]])


@pytest.fixture(scope="module")
def observed(tmp_path_factory):
    source = _source_bytes(tmp_path_factory.mktemp("handoff_source"))
    model = ContactModel(source[0])
    start = model.step(model.initial(), 10.)
    middle = model.step(start, 20.)
    end = model.step(middle, 30.)
    return model, [start, middle, end]


def test_contact_report_cannot_enable_handoff_and_keeps_source_budgets(observed):
    model, states = observed
    before = deepcopy(states)
    report, arrays = assess_contact_handoff(model, states)
    assert report["handoff_ready"] is False
    assert not report["checks"]["geological_persistence"]
    blockers = {row["code"] for row in report["blockers"]}
    assert {"frozen_thermal_orbit", "crust_inventory_missing", "conservative_remap_missing"} <= blockers
    assert "crack_path_localization_missing" in blockers
    assert 1 <= report["cohesive_component_count"] <= report["cut_component_count"]
    assert report["source_time_myr"] == model.source_time_myr
    assert report["observation_years"] == 20.
    np.testing.assert_array_equal(arrays["layer_mass_kg"], model.layer_mass_kg)
    np.testing.assert_array_equal(arrays["column_enthalpy_j_kg"], model.column_enthalpy)
    json.dumps(report, allow_nan=False)
    for old, new in zip(before, states):
        for f in fields(old):
            if isinstance(getattr(old, f.name), np.ndarray):
                np.testing.assert_array_equal(getattr(old, f.name), getattr(new, f.name))
            else:
                assert getattr(old, f.name) == getattr(new, f.name)


def test_one_interval_does_not_prove_time_stability(observed):
    report, _ = assess_contact_handoff(observed[0], observed[1][:2])
    assert not report["checks"]["short_window_stability"]


def test_observation_order_and_corrupt_checkpoint_state_are_rejected(observed):
    model, states = observed
    with pytest.raises(ValueError, match="increase"):
        assess_contact_handoff(model, states[::-1])
    with pytest.raises(ValueError, match="two"):
        assess_contact_handoff(model, states[:1])
    bad = replace(states[-1], elapsed_years=float("nan"))
    with pytest.raises(ValueError):
        assess_contact_handoff(model, [states[0], bad])


@pytest.mark.parametrize("kwargs", [{"min_region_cells": True}, {"min_region_cells": 1},
    {"max_rigid_residual_fraction": 2.}, {"min_observation_years": float("inf")}])
def test_screen_controls_are_explicit_and_validated(kwargs):
    with pytest.raises(ValueError):
        HandoffScreenParameters(**kwargs).validate()


def test_cli_records_not_ready_without_overwriting_source_or_results(tmp_path, observed):
    from run_genesis_handoff import main
    model, states = observed
    source = tmp_path/"contact.npz"
    save_contact_checkpoint(source, model, states[-1])
    initial_bytes = source.read_bytes()
    output = tmp_path/"report"
    assert main(["--checkpoint", str(source), "--output", str(output), "--probe-years", "1", "--intervals", "2"]) == 0
    report = json.loads((output/"handoff_report.json").read_text(encoding="utf-8"))
    assert report["handoff_ready"] is False
    assert (output/"handoff_assessment.png").is_file()
    assert not (output/"mature_checkpoint.npz").exists()
    assert source.read_bytes() == initial_bytes
    protected = (output/"handoff_report.json").read_bytes()
    assert main(["--checkpoint", str(source), "--output", str(output)]) == 1
    assert (output/"handoff_report.json").read_bytes() == protected
