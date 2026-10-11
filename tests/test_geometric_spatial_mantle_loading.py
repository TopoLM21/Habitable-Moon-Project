"""Independent spatial source controls, physical work, and trusted restart."""
from dataclasses import replace
from functools import lru_cache
import json
import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tests.test_geometric_distributed_contact_invariants import (
    PARAMETERS, PROVENANCE, split_material_cuts, world,
)
from tectonics.geometric_basal_loading import assemble_geometric_basal_load, basal_load_to_dict
from tectonics.geometric_basal_sources import (
    basal_source_from_dict, basal_source_operators, basal_source_to_dict, validate_basal_source,
)
from tectonics.geometric_rigid_mantle_loading import (
    PrescribedRigidMantleSource, assemble_physical_rigid_mantle_load, rigid_mantle_load_to_dict,
)
from tectonics.geometric_spatial_mantle_loading import (
    FORMAT, MODEL, PhysicalSpatialMantleLoad, PrescribedSpatialMantleSource,
    _body, _hash, _integrate, _source_traction, assemble_physical_spatial_mantle_load,
    spatial_mantle_load_from_dict, spatial_mantle_load_to_dict,
    validate_physical_spatial_mantle_load,
)


SOURCE = PrescribedSpatialMantleSource((0., 0., 1e-12), 1e14, 1., (.2, 0., 0.),
    "Explicit globally positive spatial mantle flow fixed before any plate solve")


@lru_cache(maxsize=1)
def fixture():
    mesh, original, lid = world()
    # Actual x hemispheres: source depends on world position, never on owners.
    surface = replace(original, fragments=tuple(replace(fragment,
        parcel=replace(fragment.parcel, plate=int(np.mean(np.asarray(fragment.polygon)[:, 0]) < 0.)))
        for fragment in original.fragments))
    return mesh, surface, lid, assemble_physical_spatial_mantle_load(mesh, surface, SOURCE)


def _seal(load):
    return replace(load, load_sha256=_hash(_body(load)))


def test_independently_prescribed_positive_spatial_field_drives_differential_basal_motion():
    mesh, surface, _, load = fixture()
    assert isinstance(load, PhysicalSpatialMantleLoad)
    free_velocity = np.linalg.solve(load.drag_nm_s, load.torque_nm).reshape((-1, 3))
    # A source defined before the solve yields two different material speeds.
    assert free_velocity[0, 2] > SOURCE.angular_velocity_rad_s[2] > free_velocity[1, 2]
    assert free_velocity[0, 2]-free_velocity[1, 2] > .2*SOURCE.angular_velocity_rad_s[2]
    np.testing.assert_allclose(free_velocity[:, :2], 0., atol=2e-29, rtol=0.)
    q = np.tile(SOURCE.angular_velocity_rad_s, len(load.plate_ids))
    assert np.linalg.norm(np.asarray(load.torque_nm)-np.asarray(load.drag_nm_s)@q) > .09*np.linalg.norm(load.torque_nm)
    for sample in load.samples:
        x = mesh.centroids[sample.cell]
        amplitude = math.fsum((SOURCE.amplitude_constant, *(a*float(y) for a, y in zip(SOURCE.amplitude_gradient, x))))
        assert 1.-.2 <= amplitude <= 1.+.2
        expected = SOURCE.basal_drag_pa_s_m*surface.radius_km*1000.*amplitude*np.cross(SOURCE.angular_velocity_rad_s, x)
        np.testing.assert_array_equal(sample.source_traction_pa, expected)
    assert validate_physical_spatial_mantle_load(surface, load, mesh=mesh, source=SOURCE) == load


def test_zero_gradient_matches_rigid_source_operators_exactly_but_keeps_distinct_source_kind():
    mesh, surface, _, spatial = fixture()
    constant = assemble_physical_spatial_mantle_load(mesh, surface,
        replace(SOURCE, amplitude_gradient=(0., 0., 0.)))
    rigid = assemble_physical_rigid_mantle_load(mesh, surface,
        PrescribedRigidMantleSource(SOURCE.angular_velocity_rad_s, SOURCE.basal_drag_pa_s_m, SOURCE.provenance))
    assert constant.drag_nm_s == rigid.drag_nm_s == spatial.drag_nm_s
    assert constant.torque_nm == rigid.torque_nm
    assert constant.load_sha256 != rigid.load_sha256
    assert basal_source_to_dict(surface, constant)["format"] != basal_source_to_dict(surface, rigid)["format"]
    assert constant.samples[0].source_traction_pa == rigid.samples[0].source_traction_pa


def test_spatial_source_and_drag_virtual_work_match_independent_physical_sample_power():
    _, surface, _, load = fixture()
    trial = ((-7e-12, 2e-12, 4e-12), (3e-12, -5e-12, -1e-12))
    velocities = {owner: np.asarray(value, dtype=np.longdouble) for owner, value in zip(load.plate_ids, trial)}
    source_power, drag_power = np.longdouble(0.), np.longdouble(0.)
    for sample in load.samples:
        v = np.longdouble(surface.radius_km*1000.)*np.cross(
            velocities[sample.plate_id], np.asarray(sample.position, dtype=np.longdouble))
        source_power += np.longdouble(sample.area_m2)*np.asarray(sample.source_traction_pa, dtype=np.longdouble)@v
        drag_power += np.longdouble(sample.area_m2)*np.longdouble(SOURCE.basal_drag_pa_s_m)*(v@v)
    compact = np.asarray(trial).reshape(-1)
    assert float(source_power) == pytest.approx(np.asarray(load.torque_nm)@compact, rel=3e-15)
    assert float(drag_power) == pytest.approx(compact@np.asarray(load.drag_nm_s)@compact, rel=3e-15)


def test_sparse_renumbering_changes_only_owner_coordinates_and_preserves_world_field():
    mesh, surface, _, load = fixture()
    rename = {0: 10**12, 1: 7}
    changed_surface = replace(surface, fragments=tuple(replace(fragment,
        parcel=replace(fragment.parcel, plate=rename[fragment.parcel.plate])) for fragment in surface.fragments))
    changed = assemble_physical_spatial_mantle_load(mesh, changed_surface, SOURCE)
    assert changed.plate_ids == (7, 10**12)
    assert changed.source == load.source
    assert len(changed.torque_nm) == 6
    original_samples = {sample.fragment_id: sample for sample in load.samples}
    for sample in changed.samples:
        assert sample == replace(original_samples[sample.fragment_id], plate_id=sample.plate_id)
    permutation = [3, 4, 5, 0, 1, 2]
    np.testing.assert_array_equal(changed.torque_nm, np.asarray(load.torque_nm)[permutation])
    np.testing.assert_array_equal(changed.drag_nm_s,
        np.asarray(load.drag_nm_s)[np.ix_(permutation, permutation)])


def test_material_subdivision_preserves_the_physical_spatial_source_operators():
    mesh, surface, lid, load = fixture()
    divided, _ = split_material_cuts(surface, lid)
    changed = assemble_physical_spatial_mantle_load(mesh, divided, SOURCE)
    assert np.linalg.norm(np.asarray(changed.drag_nm_s)-load.drag_nm_s) <= 2e-14*np.linalg.norm(load.drag_nm_s)
    assert np.linalg.norm(np.asarray(changed.torque_nm)-load.torque_nm) <= 2e-14*np.linalg.norm(load.torque_nm)
    assert changed.source == load.source
    assert changed.surface_sha256 != load.surface_sha256


def test_json_restart_reassembles_the_explicit_spatial_field_and_physical_samples_exactly():
    mesh, surface, _, load = fixture()
    raw = json.loads(json.dumps(basal_source_to_dict(surface, load)))
    assert raw["format"] == FORMAT and raw["model"] == MODEL and raw["transmission"] == 1.
    restored = basal_source_from_dict(raw, surface, mesh=mesh, source=SOURCE, provenance=SOURCE.provenance)
    assert restored == load
    assert json.dumps(basal_source_to_dict(surface, restored), sort_keys=True) == json.dumps(raw, sort_keys=True)
    assert basal_source_from_dict(raw, surface, mesh=mesh, parameters=SOURCE) == load
    assert basal_source_operators(surface, restored)[0] == (0, 1)


@pytest.mark.parametrize("kind", ("traction", "torque", "drag", "owner", "area"))
def test_rehashed_sample_and_operator_tampering_cannot_replace_the_physical_source(kind):
    _, surface, _, load = fixture()
    first = load.samples[0]
    if kind == "traction":
        changed = replace(load, samples=(replace(first,
            source_traction_pa=tuple(2.*x for x in first.source_traction_pa)), *load.samples[1:]))
    elif kind == "torque":
        changed = replace(load, torque_nm=tuple(2.*x for x in load.torque_nm))
    elif kind == "drag":
        changed = replace(load, drag_nm_s=tuple(tuple(2.*x for x in row) for row in load.drag_nm_s))
    elif kind == "owner":
        changed = replace(load, samples=(replace(first, plate_id=1-first.plate_id), *load.samples[1:]))
    else:
        changed = replace(load, samples=(replace(first, area_m2=2.*first.area_m2), *load.samples[1:]))
    with pytest.raises(ValueError):
        validate_physical_spatial_mantle_load(surface, _seal(changed))


def test_trusted_mesh_reassembly_rejects_rehashed_self_consistent_foreign_quadrature():
    mesh, surface, _, load = fixture()
    rotation = Rotation.from_rotvec((.2, -.1, .3)).as_matrix()
    samples = tuple(replace(sample, position=tuple(map(float, rotation@sample.position)),
        source_traction_pa=_source_traction(SOURCE, surface.radius_km*1000., rotation@sample.position))
        for sample in load.samples)
    drag, torque = _integrate(samples, load.plate_ids, SOURCE, surface.radius_km*1000.)
    forged = _seal(replace(load, samples=samples, drag_nm_s=drag, torque_nm=torque))
    assert validate_physical_spatial_mantle_load(surface, forged) == forged
    with pytest.raises(ValueError, match="trusted geometric source reassembly"):
        validate_physical_spatial_mantle_load(surface, forged, mesh=mesh)


def test_saved_source_can_change_only_if_the_explicit_trusted_input_also_changes():
    mesh, surface, _, load = fixture()
    changed = assemble_physical_spatial_mantle_load(mesh, surface,
        replace(SOURCE, amplitude_gradient=(.3, -.1, .05)))
    assert changed.drag_nm_s == load.drag_nm_s and changed.torque_nm != load.torque_nm
    with pytest.raises(ValueError, match="physical source changed"):
        validate_physical_spatial_mantle_load(surface, changed, mesh=mesh, source=SOURCE)


def test_existing_quadratic_and_rigid_codecs_remain_identical_under_union_dispatch():
    mesh, surface, lid, _ = fixture()
    default = assemble_geometric_basal_load(mesh, surface, PARAMETERS, lid, PROVENANCE)
    rigid = assemble_physical_rigid_mantle_load(mesh, surface,
        PrescribedRigidMantleSource(SOURCE.angular_velocity_rad_s, SOURCE.basal_drag_pa_s_m, SOURCE.provenance))
    assert basal_source_to_dict(surface, default) == basal_load_to_dict(surface, default)
    assert basal_source_to_dict(surface, rigid) == rigid_mantle_load_to_dict(surface, rigid)
    for saved in (default, rigid):
        raw = json.loads(json.dumps(basal_source_to_dict(surface, saved)))
        assert basal_source_from_dict(raw, surface) == saved


@pytest.mark.parametrize("changes", (
    {"angular_velocity_rad_s": [1e-12, 0., 0.]},
    {"angular_velocity_rad_s": (True, 0., 0.)},
    {"angular_velocity_rad_s": (float("nan"), 0., 0.)},
    {"basal_drag_pa_s_m": 0.},
    {"amplitude_constant": True},
    {"amplitude_constant": 0.},
    {"amplitude_constant": .2},
    {"amplitude_gradient": [0., 0., 0.]},
    {"amplitude_gradient": (False, 0., 0.)},
    {"amplitude_gradient": (float("inf"), 0., 0.)},
    {"amplitude_gradient": (1., 1., 1.)},
    {"provenance": ""},
))
def test_source_requires_explicit_immutable_finite_globally_positive_inputs(changes):
    with pytest.raises(ValueError):
        replace(SOURCE, **changes)


def test_strict_global_positivity_accepts_a_finite_constant_just_above_gradient_norm():
    assert replace(SOURCE, amplitude_constant=math.nextafter(.2, math.inf)).amplitude_constant > .2


@pytest.mark.parametrize("changes", (
    {"format": "posthoc_fitted_owner_torques"},
    {"model": "fractional_prescribed_basal_v1"},
    {"source_parameters": {}},
    {"transmission": True},
))
def test_distinct_explicit_source_schema_cannot_be_silently_migrated(changes):
    _, surface, _, load = fixture()
    raw = json.loads(json.dumps(spatial_mantle_load_to_dict(surface, load)))
    with pytest.raises(ValueError):
        basal_source_from_dict(dict(raw, **changes), surface)


def test_source_dispatch_rejects_thermal_metadata_other_source_parameters_and_owner_target_fields():
    _, surface, _, load = fixture()
    with pytest.raises(ValueError):
        validate_basal_source(surface, load, parameters=PARAMETERS)
    with pytest.raises(ValueError):
        validate_basal_source(surface, load, thermal_lid_by_fragment={})
    raw = json.loads(json.dumps(spatial_mantle_load_to_dict(surface, load)))
    raw["source"]["target_owner_torques"] = {"0": [0., 0., 1.]}
    with pytest.raises(ValueError):
        spatial_mantle_load_from_dict(raw, surface)
