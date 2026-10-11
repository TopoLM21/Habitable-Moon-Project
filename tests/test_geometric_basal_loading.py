"""Physical-source reuse, compact ownership and immutable load checkpoints."""
from dataclasses import FrozenInstanceError, replace
import hashlib
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tectonics.basal_coupling import PrescribedBasalParameters
from tectonics.fractional_dynamics import solve_fractional_basal_dynamics
from tectonics.fractional_surface import FractionalSurfaceState, SurfaceParcel
from tectonics.geometric_basal_loading import (
    assemble_geometric_basal_load, basal_load_to_dict, basal_load_from_dict,
    validate_geometric_basal_load,
)
from tectonics.geometric_boundary import surface_digest
from tectonics.geometric_surface import from_fractional_surface, split_geometry
from tectonics.mesh import build_icosphere


PARAMETERS = PrescribedBasalParameters(17, 80000., 1e14, 2.)
PROVENANCE = "Synthetic prescribed Genesis source; independent beta; explicit thermal lid"


def world(*, owner_ids=(0, 1)):
    mesh = build_icosphere(0)
    radius = 1000.
    areas = mesh.physical_cell_areas_km2(radius)
    parcels = tuple(SurfaceParcel(i, owner_ids[int(mesh.centroids[i, 0] > 0)], f"origin:{i}",
        float(a), float(a*2.), float(a*3.), float(a*3e9), 10.) for i, a in enumerate(areas))
    fractional = FractionalSurfaceState(10., tuple(areas), parcels)
    surface = from_fractional_surface(mesh, fractional, radius)
    thickness = .1+8*(mesh.centroids[:, 2]+1)/2
    lid = {fragment.fragment_id: float(thickness[fragment.parcel.cell]) for fragment in surface.fragments}
    return mesh, surface, fractional, lid, thickness


def load(mesh, surface, lid, parameters=PARAMETERS):
    return assemble_geometric_basal_load(mesh, surface, parameters, lid, PROVENANCE)


def test_pure_cell_projection_exactly_reuses_fractional_basal_operators():
    mesh, surface, fractional, lid, thickness = world()
    before = surface_digest(surface)
    result = load(mesh, surface, lid)
    oracle = solve_fractional_basal_dynamics(mesh, fractional, surface.radius_km,
                                           PARAMETERS, thickness, plate_count=2)
    drag = np.asarray(result.drag_nm_s)
    for index in range(2):
        np.testing.assert_array_equal(drag[3*index:3*index+3, 3*index:3*index+3],
                                      oracle.basal_drag_tensor_nm_s[index])
    np.testing.assert_array_equal(np.asarray(result.torque_nm).reshape(2, 3),
                                  oracle.basal_driving_torque_nm)
    assert surface_digest(surface) == before
    assert result.plate_ids == (0, 1)
    assert dict(result.projection_diagnostics)["max_relative_material_residual"] == 0.


def test_noncontiguous_huge_owner_ids_allocate_only_actual_three_dof_blocks():
    mesh, surface, _, lid, _ = world(owner_ids=(7, 10**12))
    result = load(mesh, surface, lid)
    assert result.plate_ids == (7, 10**12)
    assert np.asarray(result.drag_nm_s).shape == (6, 6)
    assert len(result.torque_nm) == 6
    with pytest.raises(FrozenInstanceError):
        result.provenance = "changed"
    assert isinstance(result.drag_nm_s, tuple)
    assert all(isinstance(row, tuple) for row in result.drag_nm_s)


def test_splitting_identical_material_changes_no_integrated_source_or_drag():
    mesh, surface, _, lid, _ = world()
    original = load(mesh, surface, lid)
    fragment = surface.fragments[0]
    a, b, c = np.asarray(fragment.polygon)
    middle = b+c
    middle /= np.linalg.norm(middle)
    children = split_geometry(fragment, (np.array([a, b, middle]), np.array([a, middle, c])))
    split = replace(surface, fragments=children+surface.fragments[1:])
    split_lid = {**lid, children[0].fragment_id: lid[fragment.fragment_id],
                 children[1].fragment_id: lid[fragment.fragment_id]}
    del split_lid[fragment.fragment_id]
    result = load(mesh, split, split_lid)
    np.testing.assert_allclose(result.drag_nm_s, original.drag_nm_s, rtol=3e-15, atol=0.)
    np.testing.assert_allclose(result.torque_nm, original.torque_nm, rtol=3e-15,
                               atol=np.linalg.norm(original.torque_nm)*3e-15)


def test_beta_drag_remains_independent_of_thermal_transmission():
    mesh, surface, _, lid, _ = world()
    thin = load(mesh, surface, {key: 0. for key in lid})
    thick = load(mesh, surface, {key: 100. for key in lid})
    assert thin.drag_nm_s == thick.drag_nm_s
    assert all(value == 0. for value in thin.torque_nm)
    assert np.linalg.norm(thick.torque_nm) > 0.


@pytest.mark.parametrize("what", ["missing", "extra", "none", "negative", "bool", "nan"])
def test_no_thermal_thickness_fallback_or_invalid_lid(what):
    mesh, surface, _, lid, _ = world()
    if what == "missing":
        del lid[surface.fragments[0].fragment_id]
    elif what == "extra":
        lid["absent"] = 1.
    elif what == "none":
        lid = None
    else:
        lid[surface.fragments[0].fragment_id] = {"negative": -1., "bool": True, "nan": float("nan")}[what]
    with pytest.raises(ValueError):
        load(mesh, surface, lid)


def test_degenerate_single_cell_plate_quadrature_is_rejected():
    mesh, surface, _, lid, _ = world()
    first = surface.fragments[0]
    singular = replace(surface, fragments=(replace(first, parcel=replace(first.parcel, plate=123)),
                                          *surface.fragments[1:]))
    with pytest.raises(ValueError, match="Degenerate"):
        load(mesh, singular, lid)


def test_json_restart_and_trusted_reassembly_are_exact():
    mesh, surface, _, lid, _ = world()
    original = load(mesh, surface, lid)
    raw = json.loads(json.dumps(basal_load_to_dict(surface, original), sort_keys=True, allow_nan=False))
    assert basal_load_from_dict(raw, surface) == original
    assert basal_load_from_dict(raw, surface, mesh=mesh, parameters=PARAMETERS,
        thermal_lid_by_fragment=lid, provenance=PROVENANCE) == original


def test_fragment_tuple_reorder_preserves_exact_load_and_trusted_restart():
    mesh, surface, _, lid, _ = world()
    original = load(mesh, surface, lid)
    reordered = replace(surface, fragments=tuple(reversed(surface.fragments)))
    assert surface_digest(surface) == surface_digest(reordered)
    assert load(mesh, reordered, lid) == original
    raw = json.loads(json.dumps(basal_load_to_dict(surface, original)))
    assert basal_load_from_dict(raw, reordered, mesh=mesh, parameters=PARAMETERS,
        thermal_lid_by_fragment=lid, provenance=PROVENANCE) == original


@pytest.mark.parametrize("what", ["surface", "mesh", "source", "lid", "provenance"])
def test_restart_is_bound_to_exact_physical_inputs(what):
    mesh, surface, _, lid, _ = world()
    original = load(mesh, surface, lid)
    args = dict(mesh=mesh, parameters=PARAMETERS, thermal_lid_by_fragment=lid, provenance=PROVENANCE)
    if what == "surface":
        surface = replace(surface, time_myr=11.)
    elif what == "mesh":
        args["mesh"] = build_icosphere(1)
    elif what == "source":
        args["parameters"] = replace(PARAMETERS, seed=19)
    elif what == "lid":
        args["thermal_lid_by_fragment"] = {key: value+1 for key, value in lid.items()}
    else:
        args["provenance"] = "changed"
    with pytest.raises(ValueError):
        validate_geometric_basal_load(surface, original, **args)


@pytest.mark.parametrize("what", ["torque", "drag", "thermal", "missing_parameter", "source_metadata", "unknown"])
def test_corrupted_checkpoint_is_rejected(what):
    mesh, surface, _, lid, _ = world()
    raw = json.loads(json.dumps(basal_load_to_dict(surface, load(mesh, surface, lid))))
    if what == "torque":
        raw["torque_nm"][0] *= 2.
    elif what == "drag":
        raw["drag_nm_s"][0][0] *= 2.
    elif what == "thermal":
        raw["thermal_lid_by_fragment"][0][1] += 1.
    elif what == "missing_parameter":
        raw["source_parameters"].pop()
    elif what == "source_metadata":
        raw["source_metadata"]["frame"] = "moving"
    else:
        raw["unknown"] = 1
    with pytest.raises(ValueError):
        basal_load_from_dict(raw, surface)


def test_trusted_reassembly_rejects_fabricated_torque_even_with_updated_checksum():
    mesh, surface, _, lid, _ = world()
    raw = json.loads(json.dumps(basal_load_to_dict(surface, load(mesh, surface, lid))))
    raw["torque_nm"][0] += np.linalg.norm(raw["torque_nm"])
    body = {key: value for key, value in raw.items() if key != "load_sha256"}
    raw["load_sha256"] = hashlib.sha256(json.dumps(body,sort_keys=True,separators=(",", ":"),
        ensure_ascii=False,allow_nan=False).encode()).hexdigest()
    with pytest.raises(ValueError, match="trusted source reassembly"):
        basal_load_from_dict(raw, surface, mesh=mesh, parameters=PARAMETERS,
            thermal_lid_by_fragment=lid, provenance=PROVENANCE)


def test_passive_coordinate_change_rotates_physical_operators_and_preserves_power():
    mesh, surface, _, lid, _ = world()
    original = load(mesh, surface, lid)
    # Rotate the actual physical operators. Regenerating the same seed in new
    # axes holds its quadratic matrix fixed and is a different loading case.
    rotation = Rotation.from_rotvec([.3, -.2, .1]).as_matrix()
    transform = np.kron(np.eye(2), rotation)
    drag, torque = np.asarray(original.drag_nm_s), np.asarray(original.torque_nm)
    velocity = np.array([1., 2., -3., -2., .5, 1.])*1e-14
    rotated_drag = transform@drag@transform.T
    rotated_torque = transform@torque
    rotated_velocity = transform@velocity
    assert rotated_velocity@rotated_drag@rotated_velocity == pytest.approx(velocity@drag@velocity, rel=2e-15)
    assert rotated_torque@rotated_velocity == pytest.approx(torque@velocity, rel=2e-15)
