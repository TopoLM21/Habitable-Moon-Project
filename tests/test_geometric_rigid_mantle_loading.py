"""An independently prescribed physical rigid mantle source with honest metadata."""
from dataclasses import replace
from functools import lru_cache
import hashlib
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tests.test_geometric_distributed_contact_invariants import (
    PARAMETERS, PROVENANCE, split_material_cuts, world,
)
from tectonics.geometric_basal_loading import (
    assemble_geometric_basal_load, basal_load_to_dict,
)
from tectonics.geometric_basal_sources import (
    basal_source_from_dict, basal_source_operators, basal_source_to_dict, validate_basal_source,
)
from tectonics.geometric_rigid_mantle_loading import (
    FORMAT, MODEL, PhysicalRigidMantleLoad, PrescribedRigidMantleSource,
    _body, _hash, _integrate, _source_traction, assemble_physical_rigid_mantle_load,
    rigid_mantle_load_from_dict, rigid_mantle_load_to_dict, validate_physical_rigid_mantle_load,
)
from tectonics.geometric_surface import totals


SOURCE = PrescribedRigidMantleSource((2e-12, -1e-12, 3e-12), 1e14,
    "Manufactured rigid mantle flow fixed before any plate solve; full transmitted traction")


@lru_cache(maxsize=None)
def fixture():
    mesh, surface, lid = world()
    load = assemble_physical_rigid_mantle_load(mesh, surface, SOURCE)
    return mesh, surface, lid, load


def _seal(load):
    return replace(load, load_sha256=_hash(_body(load)))


def test_source_is_fixed_before_solve_and_sample_tractions_independently_generate_common_velocity():
    mesh, surface, _, load = fixture()
    assert isinstance(load, PhysicalRigidMantleLoad)
    assert load.source == SOURCE
    assert load.provenance == SOURCE.provenance
    q = np.tile(SOURCE.angular_velocity_rad_s, len(load.plate_ids))
    # Driving torque was summed from the traction field, not assigned D*omega.
    residual = np.asarray(load.torque_nm)-np.asarray(load.drag_nm_s)@q
    assert np.linalg.norm(residual)/np.linalg.norm(load.torque_nm) < 8*np.finfo(float).eps
    for sample in load.samples:
        expected = SOURCE.basal_drag_pa_s_m*surface.radius_km*1000.*np.cross(
            SOURCE.angular_velocity_rad_s, mesh.centroids[sample.cell])
        np.testing.assert_array_equal(sample.source_traction_pa, expected)
    assert validate_physical_rigid_mantle_load(surface, load, mesh=mesh, source=SOURCE) == load


def test_independent_sample_power_matches_compact_torque_and_drag_virtual_work():
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


def test_zero_independently_prescribed_mantle_source_keeps_drag_without_driving_torque():
    mesh, surface, _, load = fixture()
    quiet = assemble_physical_rigid_mantle_load(mesh, surface, replace(SOURCE, angular_velocity_rad_s=(0., 0., 0.)))
    assert quiet.torque_nm == (0.,)*len(load.torque_nm)
    assert quiet.drag_nm_s == load.drag_nm_s
    assert all(sample.source_traction_pa == (0., 0., 0.) for sample in quiet.samples)


def test_material_cuts_preserve_physical_rigid_source_operators_and_original_material():
    mesh, surface, lid, load = fixture()
    divided, _ = split_material_cuts(surface, lid)
    changed = assemble_physical_rigid_mantle_load(mesh, divided, SOURCE)
    assert np.linalg.norm(np.asarray(changed.drag_nm_s)-load.drag_nm_s) <= 2e-14*np.linalg.norm(load.drag_nm_s)
    assert np.linalg.norm(np.asarray(changed.torque_nm)-load.torque_nm) <= 2e-14*np.linalg.norm(load.torque_nm)
    for name, value in totals(surface.fragments).items():
        assert totals(divided.fragments)[name] == pytest.approx(value, rel=64*np.finfo(float).eps)
    assert changed.source == load.source
    assert changed.surface_sha256 != load.surface_sha256


def test_rigid_source_metadata_cannot_masquerade_as_quadratic_thermal_source():
    _, surface, _, load = fixture()
    raw = json.loads(json.dumps(rigid_mantle_load_to_dict(surface, load)))
    assert raw["format"] == FORMAT and raw["model"] == MODEL and raw["transmission"] == 1.
    assert "source_parameters" not in raw and "thermal_lid_by_fragment" not in raw
    with pytest.raises(ValueError):
        rigid_mantle_load_from_dict(dict(raw, model="fractional_prescribed_basal_v1"), surface)
    with pytest.raises(ValueError):
        validate_basal_source(surface, load, parameters=PARAMETERS)
    with pytest.raises(ValueError):
        validate_basal_source(surface, load, thermal_lid_by_fragment={})


def test_exact_json_restart_reassembles_trusted_physical_samples_and_source():
    mesh, surface, _, load = fixture()
    raw = json.loads(json.dumps(basal_source_to_dict(surface, load)))
    restored = basal_source_from_dict(raw, surface, mesh=mesh, source=SOURCE, provenance=SOURCE.provenance)
    assert restored == load
    assert json.dumps(basal_source_to_dict(surface, restored), sort_keys=True) == json.dumps(raw, sort_keys=True)
    assert basal_source_from_dict(raw, surface, mesh=mesh, parameters=SOURCE) == load


@pytest.mark.parametrize("kind", ("traction", "torque", "drag", "owner", "area"))
def test_rehashed_physical_sample_and_operator_edits_are_rejected(kind):
    _, surface, _, load = fixture()
    if kind == "traction":
        sample = load.samples[0]
        changed = replace(load, samples=(replace(sample, source_traction_pa=tuple(2.*x for x in sample.source_traction_pa)), *load.samples[1:]))
    elif kind == "torque":
        changed = replace(load, torque_nm=tuple(2.*x for x in load.torque_nm))
    elif kind == "drag":
        changed = replace(load, drag_nm_s=tuple(tuple(2.*x for x in row) for row in load.drag_nm_s))
    elif kind == "owner":
        sample = load.samples[0]
        changed = replace(load, samples=(replace(sample, plate_id=1-sample.plate_id), *load.samples[1:]))
    else:
        sample = load.samples[0]
        changed = replace(load, samples=(replace(sample, area_m2=2.*sample.area_m2), *load.samples[1:]))
    with pytest.raises(ValueError):
        validate_physical_rigid_mantle_load(surface, _seal(changed))


def test_trusted_mesh_reassembly_rejects_rehashed_foreign_sample_geometry():
    mesh, surface, _, load = fixture()
    rotation = Rotation.from_rotvec((.2, -.1, .3)).as_matrix()
    samples = tuple(replace(sample, position=tuple(map(float, rotation@sample.position)),
        source_traction_pa=_source_traction(SOURCE, surface.radius_km*1000., rotation@sample.position))
        for sample in load.samples)
    drag, torque = _integrate(samples, load.plate_ids, SOURCE, surface.radius_km*1000.)
    forged = _seal(replace(load, samples=samples, drag_nm_s=drag, torque_nm=torque))
    # Self-consistent physics does not authenticate a changed geometry sampling.
    assert validate_physical_rigid_mantle_load(surface, forged) == forged
    with pytest.raises(ValueError, match="trusted geometric source reassembly"):
        validate_physical_rigid_mantle_load(surface, forged, mesh=mesh)


def test_default_source_union_dispatch_preserves_old_codec_and_signature_exactly():
    mesh, surface, lid, _ = fixture()
    previous = assemble_geometric_basal_load(mesh, surface, PARAMETERS, lid, PROVENANCE)
    expected = basal_load_to_dict(surface, previous)
    assert basal_source_to_dict(surface, previous) == expected
    ids, drag, torque, signature = basal_source_operators(surface, previous)
    assert ids == previous.plate_ids
    np.testing.assert_array_equal(drag, previous.drag_nm_s)
    np.testing.assert_array_equal(torque, previous.torque_nm)
    encoded = json.dumps(expected, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    assert signature == hashlib.sha256(encoded).hexdigest()
    assert basal_source_from_dict(json.loads(json.dumps(expected)), surface, mesh=mesh,
        parameters=PARAMETERS, thermal_lid_by_fragment=lid, provenance=PROVENANCE) == previous


@pytest.mark.parametrize("source_args", (
    ([1e-12, 0., 0.], 1e14, "Nonimmutable vector"),
    ((True, 0., 0.), 1e14, "Boolean physical velocity"),
    ((float("nan"), 0., 0.), 1e14, "Nonfinite source"),
    ((1e-12, 0., 0.), 0., "Zero drag"),
    ((1e-12, 0., 0.), 1e14, ""),
))
def test_physical_source_requires_explicit_finite_immutable_inputs(source_args):
    with pytest.raises(ValueError):
        PrescribedRigidMantleSource(*source_args)


def test_unknown_source_format_and_extra_schema_fields_are_rejected():
    _, surface, _, load = fixture()
    raw = json.loads(json.dumps(rigid_mantle_load_to_dict(surface, load)))
    with pytest.raises(ValueError):
        basal_source_from_dict(dict(raw, format="implicitly_fitted_torque"), surface)
    with pytest.raises(ValueError):
        basal_source_from_dict(dict(raw, source_parameters={}), surface)
