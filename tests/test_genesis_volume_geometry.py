"""Independent mass, kinematic and actual-source checks for volume geometry."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tectonics.genesis_volume_geometry import (
    GEOMETRY_CLOSURE, REFERENCE_SCOPE, TETRAHEDRON_SCOPE,
    build_genesis_volume_geometry, build_updated_genesis_volume_geometry, conical_cohort_tetrahedra,
    conical_jacobian, conical_point, validate_genesis_volume_geometry,
)
from tectonics.mesh import build_icosphere


def _triangle():
    mesh = build_icosphere(1)
    return mesh.vertices[mesh.faces[0]]*1e6


def _volumes(tetrahedra):
    # Independent signed scalar triple product, rather than the producer's
    # batched determinant or its analytic frustum formula.
    return np.array([np.dot(np.cross(t[1]-t[0], t[2]-t[0]), t[3]-t[0])/6
                     for t in np.asarray(tetrahedra)])


def _coordinate_roundoff_volume_bound(triangle):
    triangle = np.asarray(triangle)
    edge = max(np.linalg.norm(triangle[1]-triangle[0]), np.linalg.norm(triangle[2]-triangle[0]))
    return 16*np.finfo(float).eps*np.linalg.norm(triangle[0])*edge**2


@pytest.mark.parametrize("uv", [(0., 0.), (1., 0.), (0., 1.), (.23, .31)])
@pytest.mark.parametrize("q", [0., .4, 1.])
def test_material_jacobian_integrates_exact_mass_at_every_coordinate(uv, q):
    triangle, mass, rho = _triangle(), 3e16, 3000.
    jacobian = conical_jacobian(triangle, mass, rho, uv, q)
    assert np.linalg.det(jacobian) == pytest.approx(-2*mass/rho, rel=2e-14)
    if q == 0:
        expected = ((1-uv[0]-uv[1])*triangle[0]+uv[0]*triangle[1]+uv[1]*triangle[2])
        np.testing.assert_allclose(conical_point(triangle, mass, rho, uv, q), expected,
                                   rtol=2e-15, atol=1e-10)


def test_jacobian_is_derivative_of_the_actual_placement():
    triangle, mass, rho = _triangle(), 3e17, 3000.
    coordinates = np.array([.23, .31, .47])
    analytic = conical_jacobian(triangle, mass, rho, coordinates[:2], coordinates[2])
    columns = []
    for i in range(3):
        step = np.zeros(3)
        step[i] = 1e-4
        plus, minus = coordinates+step, coordinates-step
        columns.append((conical_point(triangle, mass, rho, plus[:2], plus[2])
                        -conical_point(triangle, mass, rho, minus[:2], minus[2]))/(2*step[i]))
    np.testing.assert_allclose(analytic, np.column_stack(columns), rtol=2e-8, atol=1e-6)


def test_fixed_material_coordinates_conserve_density_under_nonuniform_face_deformation():
    triangle, mass, rho = _triangle(), 3e17, 3000.
    changed = triangle.copy()
    changed[1] += np.array([9000., -4000., 2000.])
    changed /= np.linalg.norm(changed, axis=1)[:, None]
    changed *= 1.01e6
    for uv, q in [((.13, .22), .1), ((.41, .2), .6), ((.2, .7), .9)]:
        old = conical_jacobian(triangle, mass, rho, uv, q)
        new = conical_jacobian(changed, mass, rho, uv, q)
        deformation = new@np.linalg.inv(old)
        assert np.linalg.det(deformation) == pytest.approx(1., rel=3e-14)
        assert not np.allclose(deformation, np.eye(3))


def test_rigid_rotation_transports_placement_and_jacobian_objectively():
    triangle, mass, rho = _triangle(), 3e17, 3000.
    rotation = Rotation.from_rotvec([.2, -.4, .3]).as_matrix()
    moved = triangle@rotation.T
    point = conical_point(triangle, mass, rho, (.2, .3), .4)
    jacobian = conical_jacobian(triangle, mass, rho, (.2, .3), .4)
    np.testing.assert_allclose(conical_point(moved, mass, rho, (.2, .3), .4),
                               rotation@point, rtol=3e-15, atol=2e-10)
    np.testing.assert_allclose(conical_jacobian(moved, mass, rho, (.2, .3), .4),
                               rotation@jacobian, rtol=3e-15, atol=1e-10)


@pytest.mark.parametrize("lo,hi", [(0., 1.), (.01, .02), (.64, .64001)])
def test_geometric_tetrahedra_partition_the_exact_live_mass_volume(lo, hi):
    triangle, mass, rho = _triangle(), 3e17, 3000.
    tetrahedra = conical_cohort_tetrahedra(triangle, mass, rho, lo, hi, (5, 2, 7))
    volumes = _volumes(tetrahedra)
    assert tetrahedra.shape == (3, 4, 3)
    assert np.all(volumes > 0)
    assert abs(volumes.sum()-mass*(hi-lo)/rho) <= _coordinate_roundoff_volume_bound(triangle)


def test_global_vertex_sorting_makes_tetrahedral_split_independent_of_face_cyclic_order():
    triangle, mass, rho = _triangle(), 3e17, 3000.
    ids = np.array([18, 4, 11])
    original = conical_cohort_tetrahedra(triangle, mass, rho, .1, .6, ids)
    for shift in (1, 2):
        moved = conical_cohort_tetrahedra(np.roll(triangle, shift, axis=0), mass, rho,
                                        .1, .6, np.roll(ids, shift))
        np.testing.assert_allclose(moved, original, rtol=0, atol=3e-10)


def test_geometry_tetrahedra_are_not_claimed_as_fixed_incompressible_material_parcels():
    # Both complete frusta have exactly the same material volume. Their three
    # geometric partitions have different fractions after a radius change.
    triangle, mass, rho = _triangle(), 3e17, 3000.
    old = _volumes(conical_cohort_tetrahedra(triangle, mass, rho, 0., 1., (0, 1, 2)))
    new = _volumes(conical_cohort_tetrahedra(1.1*triangle, mass, rho, 0., 1., (0, 1, 2)))
    assert new.sum() == pytest.approx(old.sum(), rel=1e-12)
    assert np.max(np.abs(new/old-1)) > 1e-4


@pytest.mark.parametrize("bad", [-.1, 1.1, float("nan"), True])
def test_invalid_mass_coordinate_is_rejected(bad):
    with pytest.raises(ValueError):
        conical_point(_triangle(), 3e17, 3000., (.2, .3), bad)


@pytest.mark.parametrize("bad", [(-.1, .2), (.8, .8), (float("nan"), 0), (0.,)])
def test_invalid_barycentric_coordinate_is_rejected(bad):
    with pytest.raises(ValueError):
        conical_jacobian(_triangle(), 3e17, 3000., bad, .5)


@pytest.mark.parametrize("mutation", ["orientation", "collapse", "radius", "centre", "mass", "density"])
def test_invalid_physical_chart_is_rejected(mutation):
    triangle, mass, rho = _triangle(), 3e17, 3000.
    if mutation == "orientation":
        triangle[[1, 2]] = triangle[[2, 1]]
    elif mutation == "collapse":
        triangle[1] = triangle[0]
    elif mutation == "radius":
        triangle[1] *= 1.01
    elif mutation == "centre":
        mass = 1e30
    elif mutation == "mass":
        mass = float("inf")
    else:
        rho = 0.
    with pytest.raises(ValueError):
        conical_cohort_tetrahedra(triangle, mass, rho, .1, .6, (0, 1, 2))


@pytest.fixture(scope="module")
def actual_packet():
    # Frozen evidence is the completed canonical formation from actual
    # molten birth. It remains byte-for-byte untouched by this new closure.
    path = Path(__file__).resolve().parents[1]/"analysis/genesis_cohort_mechanics_validation/evidence/control/material_export.json"
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def actual_geometry(actual_packet):
    return build_genesis_volume_geometry(actual_packet,
        expected_source_checkpoint_sha256=actual_packet["source"]["checkpoint_sha256"])


def test_actual_birth_source_retains_ids_mass_ownership_and_unavailable_history(actual_packet,
                                                                             actual_geometry):
    geometry = actual_geometry
    assert geometry.closure == GEOMETRY_CLOSURE
    assert geometry.reference_scope == REFERENCE_SCOPE
    assert geometry.tetrahedron_scope == TETRAHEDRON_SCOPE
    assert geometry.source_checkpoint_sha256 == actual_packet["source"]["checkpoint_sha256"]
    assert geometry.source_identity == actual_packet["birth_source_identity"]
    assert geometry.time_myr == actual_packet["time_myr"]
    assert len(geometry.cohorts) == 480
    assert len(geometry.owner_components) == 1
    assert geometry.owner_components[0] == tuple(range(80))
    assert len(set(geometry.owner_ids_per_face)) == 1
    assert not geometry.full_historical_three_dimensional_memory_present
    assert not geometry.stress_free_reinitialization
    assert not geometry.continental_plate_ownership_claimed
    assert not geometry.physical_force_work_admission
    assert all(v is None for v in actual_packet["unknown_fields"].values())
    history = {c["id"]: c for c in actual_packet["bulk_birth_history"]["cohorts"]}
    live_catalogue = {c["source_cohort_id"]: c for c in actual_packet["solid_birth_cohort_material_catalogue"]
                      if c["live_mass_kg"] > 0}
    assert tuple(c.id for c in geometry.cohorts) == tuple(live_catalogue)
    for cohort in geometry.cohorts:
        birth = history[cohort.id]
        assert (cohort.face_index, cohort.layer_index, cohort.q_lo, cohort.live_q_hi) == (
            birth["face_index"], birth["layer_index"], birth["q_lo"], birth["live_q_hi"])
        assert cohort.mass_kg == live_catalogue[cohort.id]["live_mass_kg"]
        assert cohort.material_id == live_catalogue[cohort.id]["material_id"]
        assert cohort.volume_m3 == pytest.approx(cohort.mass_kg/geometry.density_kg_m3, rel=2e-15)
        assert abs(_volumes(cohort.tetrahedra_m).sum()-cohort.volume_m3) <= (
            _coordinate_roundoff_volume_bound(cohort.top_triangle_m))
        assert cohort.owner_id == geometry.owner_ids_per_face[cohort.face_index]
    source_volume = sum(c["live_mass_kg"] for c in live_catalogue.values())/geometry.density_kg_m3
    assert sum(c.volume_m3 for c in geometry.cohorts) == pytest.approx(source_volume, rel=3e-15)
    assert deepcopy(geometry) is geometry
    with pytest.raises(FrozenInstanceError):
        geometry.time_myr = 1.


def test_actual_geometry_readback_rebuilds_every_cached_claim(actual_packet, actual_geometry):
    assert validate_genesis_volume_geometry(actual_geometry, actual_packet) is actual_geometry
    bad = replace(actual_geometry, cohorts=(replace(actual_geometry.cohorts[0], volume_m3=1.),)
                  +actual_geometry.cohorts[1:])
    with pytest.raises(ValueError, match="full source geometry replay"):
        validate_genesis_volume_geometry(bad, actual_packet)


def test_resealed_geometry_input_cannot_replace_the_actual_source(actual_packet):
    packet = deepcopy(actual_packet)
    packet["current_geometry"]["radius_km"] *= 1.001
    body = {k: v for k, v in packet.items() if k != "packet_sha256"}
    packet["packet_sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    with pytest.raises(ValueError, match="full source replay"):
        build_genesis_volume_geometry(packet)


def test_wrong_independently_bound_source_hash_is_rejected(actual_packet):
    with pytest.raises(ValueError, match="independently supplied source"):
        build_genesis_volume_geometry(actual_packet, expected_source_checkpoint_sha256="0"*64)


def test_precursor_current_geometry_keeps_raft_and_activation_history_unavailable():
    from tectonics.genesis_bulk_export import export_genesis_bulk_material

    path = Path(__file__).resolve().parents[1]/"analysis/genesis_cohort_mechanics_validation/evidence/control/precursor_fault_checkpoint.npz"
    packet = export_genesis_bulk_material(path.read_bytes(),
        {"purpose": "current geometry of genuine pre-gate solid precursor"})
    geometry = build_genesis_volume_geometry(packet)
    assert len(geometry.cohorts) == 80
    assert not packet["current_shell_memory"]["membrane_established"]
    assert packet["instantaneous_cohort_plane_stress_3d"] is None
    assert all(c["elastic_strain"] is None and c["activation_time_myr"] is None
               for c in json.loads(geometry.source_packet_json)["cohort_mechanics"]["state"]["cohorts"])
    assert not geometry.full_historical_three_dimensional_memory_present
    assert not geometry.stress_free_reinitialization


def test_actual_molten_source_has_no_solid_geometry_or_material_owner(tmp_path):
    from tectonics.genesis import GenesisParameters
    from tectonics.genesis_bulk_export import export_genesis_bulk_material
    from tectonics.genesis_faults import FaultModel, save_fault_checkpoint
    from tectonics.genesis_onset import OnsetParameters
    from tectonics.genesis_shell import ShellParameters
    from tectonics.genesis_tides import TidalParameters

    model = FaultModel(ShellParameters(subdivisions=1), GenesisParameters(),
                       OnsetParameters(), TidalParameters(enabled=False))
    state, thermal, orbit = model.initial()
    path = tmp_path/"actual_molten_fault.npz"
    save_fault_checkpoint(path, model, state, thermal, orbit, {},
                          {"purpose": "actual molten geometry control"})
    packet = export_genesis_bulk_material(path.read_bytes(), {"purpose": "no invented solid"})
    geometry = build_genesis_volume_geometry(packet)
    assert geometry.time_myr == 0.
    assert geometry.cohorts == () and geometry.owner_components == ()
    assert geometry.owner_ids_per_face == (None,)*80
    assert json.loads(geometry.source_packet_json)["cohort_mechanics"]["state"]["cohorts"] == []


@pytest.fixture(scope="module")
def actual_continued_history():
    from tectonics.genesis_bulk_birth import bulk_birth_from_dict

    path = Path(__file__).resolve().parents[1]/"analysis/genesis_cohort_mechanics_validation/evidence/control/fault_direct_continued.npz"
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
    return bulk_birth_from_dict(metadata["state"]["bulk_birth_history"])


def test_updated_live_material_is_separate_from_the_original_checkpoint(actual_geometry,
                                                                      actual_continued_history):
    updated = build_updated_genesis_volume_geometry(actual_geometry, actual_continued_history)
    assert updated.time_myr == actual_continued_history.time_myr > actual_geometry.time_myr
    assert len(updated.cohorts) == 640 and len(actual_geometry.cohorts) == 480
    assert updated.vertices_m == actual_geometry.vertices_m
    assert updated.source_packet_json == actual_geometry.source_packet_json
    assert updated.source_checkpoint_sha256 == actual_geometry.source_checkpoint_sha256
    assert actual_geometry.runtime_birth_history_json is None
    assert updated.runtime_birth_history_json is not None
    assert json.loads(updated.source_packet_json)["time_myr"] == actual_geometry.time_myr
    assert sum(c.mass_kg for c in updated.cohorts) == pytest.approx(
        actual_continued_history.live_mass_kg, rel=3e-15)
    assert not updated.stress_free_reinitialization
    assert not updated.full_historical_three_dimensional_memory_present
    assert validate_genesis_volume_geometry(updated) is updated


def test_updated_material_cannot_reset_its_accepted_source_prefix(actual_geometry,
                                                                actual_continued_history):
    from tectonics.genesis_bulk_birth import bulk_birth_from_dict

    updated = build_updated_genesis_volume_geometry(actual_geometry, actual_continued_history)
    original = bulk_birth_from_dict(json.loads(actual_geometry.source_packet_json)["bulk_birth_history"])
    with pytest.raises(ValueError, match="substituted its origin or accepted history"):
        build_updated_genesis_volume_geometry(updated, original)
