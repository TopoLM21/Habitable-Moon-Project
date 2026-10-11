"""Native source quadrature, physical work, and trusted loading controls."""
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
from functools import lru_cache
import json
import math

import numpy as np
import pytest

from tectonics.geometric_native_mantle_loading import (
    FORMAT, QUADRATURE, NativePhysicalSpatialMantleLoad, _body, _hash, _traction,
    assemble_native_physical_spatial_mantle_load,
    native_spatial_mantle_load_from_dict, native_spatial_mantle_load_to_dict,
    validate_native_physical_spatial_mantle_load,
)
from tectonics.geometric_native_polygons import NativePolygon
from tectonics.geometric_native_surface import (
    NativeSurfaceState, enroll_native_surface,
)
from tectonics.geometric_rigid_mantle_loading import _integrate
from tectonics.geometric_spatial_mantle_loading import _source_traction
from tests.test_geometric_precision import thin_partition
from tests.test_geometric_spatial_mantle_loading import SOURCE, fixture as legacy_fixture


@lru_cache(maxsize=1)
def fixture():
    mesh, old, _, legacy = legacy_fixture()
    native = enroll_native_surface(old)
    load = assemble_native_physical_spatial_mantle_load(mesh, native, SOURCE)
    return mesh, native, load, legacy


def _seal(load):
    return replace(load, load_sha256=_hash(_body(load)))


def test_native_footprints_reproduce_the_unchanged_independent_physical_source():
    mesh, surface, load, legacy = fixture()
    assert type(load) is NativePhysicalSpatialMantleLoad
    assert load.source == SOURCE == legacy.source
    assert len(load.samples) == len(legacy.samples) == 32
    np.testing.assert_allclose(load.drag_nm_s, legacy.drag_nm_s, rtol=8e-16, atol=0.)
    # Different valid positive footprint measures can move a cancelled
    # transverse sum by a few accumulation ulps. Compare the complete physical
    # operator rather than require a rounded zero transverse component.
    assert np.linalg.norm(np.asarray(load.torque_nm)-legacy.torque_nm) < 8e-16*np.linalg.norm(legacy.torque_nm)
    velocity = np.linalg.solve(load.drag_nm_s, load.torque_nm).reshape((-1, 3))
    assert velocity[0, 2] > SOURCE.angular_velocity_rad_s[2] > velocity[1, 2]
    assert velocity[0, 2]-velocity[1, 2] > .2*SOURCE.angular_velocity_rad_s[2]
    per_fragment, per_cell = {}, {}
    for sample in load.samples:
        assert sample.position == tuple(map(float, mesh.centroids[sample.cell]))
        assert sample.source_traction_pa == _source_traction(SOURCE, surface.radius_km*1000., sample.position)
        per_fragment.setdefault(sample.source_fragment_id, []).append(sample.area_m2)
        per_cell.setdefault(sample.cell, []).append(sample.area_m2)
    for fragment in surface.fragments:
        assert math.fsum(per_fragment[fragment.fragment_id]) == pytest.approx(
            fragment.parcel.area_km2*1e6, rel=4*np.finfo(float).eps, abs=0.)
    for cell, areas in per_cell.items():
        assert math.fsum(areas) == pytest.approx(mesh.areas_unit_sphere[cell]*(surface.radius_km*1000.)**2,
                                              rel=5e-15, abs=0.)


def test_native_assembly_never_queries_a_projected_world_polygon(monkeypatch):
    import tectonics.geometric_surface as old_surface
    mesh, surface, expected, _ = fixture()
    def forbidden(*args, **kwargs):
        raise AssertionError("A world polygon is a display projection")
    monkeypatch.setattr(NativePolygon, "world_polygon", forbidden)
    monkeypatch.setattr(old_surface, "project_to_mesh", forbidden)
    assert assemble_native_physical_spatial_mantle_load(mesh, surface, SOURCE) == expected


def test_native_source_and_drag_virtual_work_match_independent_sample_power():
    _, surface, load, _ = fixture()
    trial = ((-7e-12, 2e-12, 4e-12), (3e-12, -5e-12, -1e-12))
    velocities = {owner: np.asarray(value, dtype=np.longdouble)
                  for owner, value in zip(load.plate_ids, trial)}
    source_power, drag_power = np.longdouble(0.), np.longdouble(0.)
    for sample in load.samples:
        velocity = np.longdouble(surface.radius_km*1000.)*np.cross(
            velocities[sample.plate_id], np.asarray(sample.position, dtype=np.longdouble))
        source_power += np.longdouble(sample.area_m2)*np.asarray(sample.source_traction_pa,
                                                               dtype=np.longdouble)@velocity
        drag_power += np.longdouble(sample.area_m2)*np.longdouble(SOURCE.basal_drag_pa_s_m)*(velocity@velocity)
    compact = np.asarray(trial).reshape(-1)
    assert float(source_power) == pytest.approx(np.asarray(load.torque_nm)@compact, rel=3e-15)
    assert float(drag_power) == pytest.approx(compact@np.asarray(load.drag_nm_s)@compact, rel=3e-15)


def test_full_prescribed_source_vector_survives_native_assembly_and_mesh_restart():
    mesh, surface, _, _ = fixture()
    source = replace(SOURCE, angular_velocity_rad_s=(3e-16, -4e-16, 1e-12),
                     amplitude_gradient=(.1, -.12, .08))
    load = assemble_native_physical_spatial_mantle_load(mesh, surface, source)
    raw = native_spatial_mantle_load_to_dict(surface, load)
    assert raw["source"]["angular_velocity_rad_s"] == [3e-16, -4e-16, 1e-12]
    assert native_spatial_mantle_load_from_dict(raw, surface, mesh=mesh, source=source) == load


def test_native_mesh_restart_reassembles_samples_and_exact_json_payload():
    mesh, surface, load, _ = fixture()
    raw = native_spatial_mantle_load_to_dict(surface, load)
    assert raw["format"] == FORMAT and raw["quadrature"] == QUADRATURE
    assert isinstance(raw["samples"], list) and isinstance(raw["source"]["angular_velocity_rad_s"], list)
    restored = native_spatial_mantle_load_from_dict(raw, surface, mesh=mesh,
                                                  source=SOURCE, provenance=SOURCE.provenance)
    assert restored == load
    assert json.dumps(native_spatial_mantle_load_to_dict(surface, restored), sort_keys=True) == json.dumps(raw, sort_keys=True)
    with pytest.raises(ValueError, match="trusted mesh"):
        native_spatial_mantle_load_from_dict(raw, surface)
    with pytest.raises(AttributeError):
        load.torque_nm = ()


@pytest.mark.parametrize("kind", ("traction", "torque", "drag", "owner", "area", "cell", "identity"))
def test_rehashed_tampering_cannot_replace_the_native_material_or_source(kind):
    _, surface, load, _ = fixture()
    sample = load.samples[0]
    if kind == "traction":
        changed = replace(load, samples=(replace(sample,
            source_traction_pa=tuple(2.*value for value in sample.source_traction_pa)), *load.samples[1:]))
    elif kind == "torque":
        changed = replace(load, torque_nm=tuple(2.*value for value in load.torque_nm))
    elif kind == "drag":
        changed = replace(load, drag_nm_s=tuple(tuple(2.*value for value in row) for row in load.drag_nm_s))
    elif kind == "owner":
        changed = replace(load, samples=(replace(sample, plate_id=1-sample.plate_id), *load.samples[1:]))
    elif kind == "area":
        changed = replace(load, samples=(replace(sample, area_m2=2.*sample.area_m2), *load.samples[1:]))
    elif kind == "cell":
        changed = replace(load, samples=(replace(sample, cell=sample.cell+1), *load.samples[1:]))
    else:
        changed = replace(load, samples=(replace(sample, source_fragment_id="foreign"), *load.samples[1:]))
    with pytest.raises(ValueError):
        validate_native_physical_spatial_mantle_load(surface, _seal(changed))


@pytest.mark.parametrize("path", (("transmission",), ("plate_ids", 0),
    ("source", "angular_velocity_rad_s", 0), ("source", "basal_drag_pa_s_m"),
    ("samples", 0, "cell"), ("samples", 0, "area_m2"),
    ("samples", 0, "position", 0), ("drag_nm_s", 0, 0), ("torque_nm", 0)))
def test_native_loader_rejects_boolean_physical_values_even_with_a_fresh_hash(path):
    mesh, surface, load, _ = fixture()
    raw = native_spatial_mantle_load_to_dict(surface, load)
    node = raw
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = True
    body = dict(raw)
    body.pop("load_sha256")
    raw["load_sha256"] = _hash(body)
    with pytest.raises(ValueError):
        native_spatial_mantle_load_from_dict(raw, surface, mesh=mesh)


def test_native_loader_rejects_extra_keys_and_non_json_vectors():
    mesh, surface, load, _ = fixture()
    original = native_spatial_mantle_load_to_dict(surface, load)
    for path in ((), ("source",), ("samples", 0)):
        raw = deepcopy(original)
        node = raw
        for key in path:
            node = node[key]
        node["extra"] = 1
        with pytest.raises(ValueError):
            native_spatial_mantle_load_from_dict(raw, surface, mesh=mesh)
    raw = deepcopy(original)
    raw["source"]["angular_velocity_rad_s"] = tuple(raw["source"]["angular_velocity_rad_s"])
    with pytest.raises(ValueError, match="JSON lists"):
        native_spatial_mantle_load_from_dict(raw, surface, mesh=mesh)


def test_mesh_reassembly_rejects_self_consistent_rehashed_foreign_quadrature():
    mesh, surface, load, _ = fixture()
    altered = []
    for sample in load.samples:
        position = tuple(map(float, mesh.centroids[(sample.cell+1)%mesh.cell_count]))
        altered.append(replace(sample, position=position,
            source_traction_pa=_source_traction(load.source, surface.radius_km*1000., position)))
    drag, torque = _integrate(altered, load.plate_ids, load.source, surface.radius_km*1000.)
    forged = _seal(replace(load, samples=tuple(altered), drag_nm_s=drag, torque_nm=torque))
    assert validate_native_physical_spatial_mantle_load(surface, forged) == forged
    with pytest.raises(ValueError, match="trusted native footprint reassembly"):
        validate_native_physical_spatial_mantle_load(surface, forged, mesh=mesh)
    raw = native_spatial_mantle_load_to_dict(surface, forged)
    with pytest.raises(ValueError, match="trusted native footprint reassembly"):
        native_spatial_mantle_load_from_dict(raw, surface, mesh=mesh)


def test_native_load_binding_includes_time_material_fields_and_support_pose():
    _, surface, load, _ = fixture()
    changed = NativeSurfaceState(surface.time_myr+1., surface.radius_km, surface.fragments,
                                 surface.known_material_ids)
    with pytest.raises(ValueError, match="another exact material surface"):
        validate_native_physical_spatial_mantle_load(changed, load)
    fragments = tuple(replace(fragment, parcel=replace(fragment.parcel,
        material_fields=(("damage", .123),))) for fragment in surface.fragments)
    changed = NativeSurfaceState(surface.time_myr, surface.radius_km, fragments,
                                 surface.known_material_ids)
    with pytest.raises(ValueError, match="another exact material surface"):
        validate_native_physical_spatial_mantle_load(changed, load)


def test_tiny_native_source_sample_survives_a_collapsed_world_projection():
    mesh, _, _, _ = fixture()
    old = thin_partition()
    old = replace(old, fragments=tuple(replace(fragment,
        parcel=replace(fragment.parcel, plate=0)) for fragment in old.fragments))
    native = enroll_native_surface(old)
    fragments = tuple(replace(fragment,
        geometry=fragment.geometry.rotated((1., 2., 3.), .03)) for fragment in native.fragments)
    rotated = NativeSurfaceState(.03, native.radius_km, fragments, native.known_material_ids)
    thin = next(fragment for fragment in rotated.fragments if fragment.fragment_id == "thin")
    assert len(set(thin.geometry.world_polygon())) < 3
    load = assemble_native_physical_spatial_mantle_load(mesh, rotated, SOURCE)
    tiny = [sample for sample in load.samples if sample.source_fragment_id == "thin"]
    assert tiny and all(0. < sample.area_m2 < 1e-28 for sample in tiny)
    assert math.fsum(sample.area_m2 for sample in tiny) == thin.parcel.area_km2*1e6


@pytest.mark.parametrize("kind", ("negative_area", "changed_area", "opposite_centroid", "scaled_vertex"))
def test_assembly_and_restart_reject_malformed_physical_integration_mesh(kind):
    mesh, surface, load, _ = fixture()
    if kind == "negative_area":
        changed = replace(mesh, areas_unit_sphere=-mesh.areas_unit_sphere)
    elif kind == "changed_area":
        changed = replace(mesh, areas_unit_sphere=2.*mesh.areas_unit_sphere)
    elif kind == "opposite_centroid":
        changed = replace(mesh, centroids=-mesh.centroids)
    else:
        changed = replace(mesh, vertices=2.*mesh.vertices)
    with pytest.raises(ValueError):
        assemble_native_physical_spatial_mantle_load(changed, surface, SOURCE)
    with pytest.raises(ValueError):
        native_spatial_mantle_load_from_dict(native_spatial_mantle_load_to_dict(surface, load),
                                           surface, mesh=changed)


def test_unresolved_subnormal_source_cross_product_refuses_instead_of_losing_components():
    mesh, surface, _, _ = fixture()
    source = replace(SOURCE, angular_velocity_rad_s=(5e-324, 0., 0.))
    with pytest.raises(ValueError, match="full prescribed source vector"):
        assemble_native_physical_spatial_mantle_load(mesh, surface, source)


@pytest.mark.parametrize("beta", (1e250, 1e-250))
def test_finite_well_conditioned_physical_drag_handles_extreme_declared_scaling(beta):
    mesh, surface, ordinary, _ = fixture()
    source = replace(SOURCE, basal_drag_pa_s_m=beta)
    load = assemble_native_physical_spatial_mantle_load(mesh, surface, source)
    reference = np.asarray(ordinary.drag_nm_s)/SOURCE.basal_drag_pa_s_m
    actual = np.asarray(load.drag_nm_s)/beta
    assert np.linalg.norm(actual-reference) < 8e-16*np.linalg.norm(reference)
    assert validate_native_physical_spatial_mantle_load(surface, load, mesh=mesh) == load


def test_complete_prescribed_product_refuses_intermediate_scale_underflow():
    mesh, surface, _, _ = fixture()
    source = replace(SOURCE, angular_velocity_rad_s=(0., 0., 1e300),
        basal_drag_pa_s_m=1e-320, amplitude_constant=1e-20,
        amplitude_gradient=(0., 0., 0.))
    position, radius_m = (1., 0., 0.), 1e5
    # The mathematical physical source is representable even though the
    # inherited beta*R*amplitude intermediate becomes exactly zero.
    exact = (Fraction.from_float(source.basal_drag_pa_s_m)*Fraction.from_float(radius_m)
        *Fraction.from_float(source.amplitude_constant)
        *Fraction.from_float(source.angular_velocity_rad_s[2]))
    assert 0. < float(exact) < 1e-35
    assert _source_traction(source, radius_m, position) == (0., 0., 0.)
    with pytest.raises(ValueError, match="full prescribed traction product"):
        _traction(source, radius_m, position)
    with pytest.raises(ValueError, match="full prescribed traction product"):
        assemble_native_physical_spatial_mantle_load(mesh, surface, source)
