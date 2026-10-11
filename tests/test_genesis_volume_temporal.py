"""Adversarial full-time geometry checks; no force acceptance inferred."""
from dataclasses import replace
from fractions import Fraction as F
from itertools import product
import hashlib
import json

import numpy as np
import pytest

from tectonics.genesis_volume_temporal import (
    ConeHomotheticCertificate, TetrahedronPath, UnresolvedGenesisVolumeTemporal,
    certify_cone_homothetic_path, certify_tetrahedron_paths,
    cone_homothetic_from_dict, cone_homothetic_to_dict, tetrahedron_paths_to_dict,
    tetrahedron_paths_from_dict,
)


SOURCE = "a"*64
VERTICES = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
FACES = tuple((a, b, c) if np.linalg.det(np.array((VERTICES[a], VERTICES[b], VERTICES[c]))) > 0
              else (a, c, b) for a, b, c in product((0, 1), (2, 3), (4, 5)))
ZERO = ((0, 0, 0),)*6
TETRA = ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))


def cone(A=ZERO, P=ZERO, **kwargs):
    return certify_cone_homothetic_path(VERTICES, FACES, A, P,
        lambda_min=F(1, 2), source_identity=SOURCE, **kwargs)


def scaled_vertices(scale):
    return tuple(tuple(F(x)*scale for x in point) for point in VERTICES)


def cell(name, before=TETRA, after=None, owner=0, vertex_ids=None):
    return TetrahedronPath(name, "material:"+name, owner,
        tuple(f"{name}:{i}" for i in range(4)) if vertex_ids is None else vertex_ids,
        before, before if after is None else after)


def shifted(points, delta):
    return tuple(tuple(F(x)+F(d) for x, d in zip(point, delta)) for point in points)


def tetra_certificate(*cells, **kwargs):
    return certify_tetrahedron_paths(cells, source_identity=SOURCE, **kwargs)


def reseal(packet):
    body = {key: value for key, value in packet.items() if key != "payload_sha256"}
    packet["payload_sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return packet


def test_identity_certifies_whole_clock_all_pairs_and_positive_jacobian():
    result = cone()
    assert type(result) is ConeHomotheticCertificate
    assert result.max_gradient_frobenius_squared == 0
    assert result.positive_jacobian_whole_interval_certified
    assert result.whole_clock_all_pair_noninterpenetration_certified
    assert not result.physical_force_work_admission


def test_uniform_affine_strain_independent_exact_gradient():
    scale = F(1, 10)
    result = cone(scaled_vertices(scale), scaled_vertices(scale))
    # u=s*x on every cone and also in the inner extension.
    assert result.max_gradient_frobenius_squared == 3*scale*scale
    assert all(row == (0, 0, 3*scale*scale, 3*scale*scale)
               for row in result.face_gradient_bounds_squared)


def test_before_after_total_fields_not_reinterpreted_as_increment():
    old, new = F(1, 5), F(-1, 10)
    result = cone(scaled_vertices(new), scaled_vertices(new),
        before_displacement_A_xyz=scaled_vertices(old), before_slope_P_xyz=scaled_vertices(old),
        before_time_myr=F(868, 1000), after_time_myr=F(868001, 1000000))
    assert result.max_gradient_frobenius_squared == 3*old*old
    assert result.before_time_myr == F(868, 1000)
    assert result.after_time_myr == F(868001, 1000000)


def test_unsafe_before_state_is_not_hidden_by_small_new_increment():
    with pytest.raises(UnresolvedGenesisVolumeTemporal, match="gradient"):
        cone(before_displacement_A_xyz=scaled_vertices(F(1)), before_slope_P_xyz=scaled_vertices(F(1)))


def test_gradient_bound_matches_independent_finite_difference_at_interior_points():
    A = tuple((F(i-2, 1000), F(i%3-1, 1200), F(1-i, 1300)) for i in range(6))
    P = tuple((F(3-i, 900), F(i-1, 1100), F(i%2, 1400)) for i in range(6))
    result = cone(A, P)
    for face, bounds in zip(FACES, result.face_gradient_bounds_squared):
        v = np.asarray([VERTICES[i] for i in face], float).T
        a = np.asarray([A[i] for i in face], float).T
        p = np.asarray([P[i] for i in face], float).T

        def displacement(x):
            raw = np.linalg.solve(v, x)
            lam = raw.sum()
            N = raw/lam
            if lam >= .5:
                return (a+(lam-1)*p)@N
            return lam/.5*(a-.5*p)@N

        for lam in (.2, .6, .9):
            x = lam*v@np.array((.2, .3, .5))
            h = 1e-6
            derivative = np.column_stack([(displacement(x+h*np.eye(3)[j])-
                                           displacement(x-h*np.eye(3)[j]))/(2*h) for j in range(3)])
            assert np.sum(derivative**2) <= float(bounds[3] if lam < .5 else bounds[2])*(1+1e-7)


def test_shared_cone_boundaries_are_one_field_not_independent_face_values():
    # One global value per vertex is mandatory; face-local arrays cannot pass.
    with pytest.raises(ValueError):
        certify_cone_homothetic_path(VERTICES, FACES, [[ZERO[i] for i in f] for f in FACES], ZERO,
            lambda_min=.5, source_identity=SOURCE)


@pytest.mark.parametrize("lambda_min", [0, -1, 2])
def test_global_radial_domain_requires_positive_bounded_threshold(lambda_min):
    with pytest.raises(ValueError, match="threshold"):
        certify_cone_homothetic_path(VERTICES, FACES, ZERO, ZERO,
            lambda_min=lambda_min, source_identity=SOURCE)


@pytest.mark.parametrize("faces", [FACES[:-1], FACES+(FACES[0],),
    ((FACES[0][0], FACES[0][2], FACES[0][1]),)+FACES[1:]])
def test_open_duplicate_or_reversed_boundary_refused(faces):
    with pytest.raises(ValueError):
        certify_cone_homothetic_path(VERTICES, faces, ZERO, ZERO,
            lambda_min=.5, source_identity=SOURCE)


def test_nonconvex_reference_domain_refused():
    vertices = list(VERTICES)
    vertices[0] = (.01, 0, 0)
    # Moving this octahedral vertex inward still makes a convex octahedron;
    # append an outward vertex not represented by its supporting triangles.
    vertices.append((2, 0, 0))
    with pytest.raises(ValueError, match="convex"):
        certify_cone_homothetic_path(vertices, FACES, ZERO+((0, 0, 0),), ZERO+((0, 0, 0),),
            lambda_min=.5, source_identity=SOURCE)


def test_gradient_at_exact_threshold_one_is_unresolved_without_epsilon():
    # Rank-one affine u=(x,0,0) has squared Frobenius exactly one.
    field = tuple((x, 0, 0) for x, _, _ in VERTICES)
    with pytest.raises(UnresolvedGenesisVolumeTemporal):
        cone(field, field)


def test_valid_coplanar_convex_hull_triangulation_is_accepted():
    vertices = tuple(product((-1, 1), repeat=3))
    faces = []
    for fixed_axis, side in product(range(3), (-1, 1)):
        other = tuple(i for i in range(3) if i != fixed_axis)
        square = []
        for first, second in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            point = [side if i == fixed_axis else 0 for i in range(3)]
            point[other[0]], point[other[1]] = first, second
            square.append(vertices.index(tuple(point)))
        for face in ((square[0], square[1], square[2]), (square[0], square[2], square[3])):
            if np.linalg.det(np.asarray([vertices[i] for i in face], float)) < 0:
                face = (face[0], face[2], face[1])
            faces.append(face)
    result = certify_cone_homothetic_path(vertices, faces, ((0, 0, 0),)*8, ((0, 0, 0),)*8,
        lambda_min=.5, source_identity=SOURCE)
    assert result.whole_clock_all_pair_noninterpenetration_certified
    # A source node touching another facet triangle without subdivision is
    # forbidden even if its topology IDs are all different.
    with pytest.raises(ValueError, match="T-junction"):
        certify_cone_homothetic_path(vertices+((1, 0, 0),), faces,
            ((0, 0, 0),)*9, ((0, 0, 0),)*9, lambda_min=.5, source_identity=SOURCE)


def test_cone_save_load_replays_full_geometry_and_rejects_resealed_false_claim():
    result = cone(scaled_vertices(F(1, 100)), scaled_vertices(F(1, 100)))
    packet = json.loads(json.dumps(cone_homothetic_to_dict(result)))
    assert cone_homothetic_from_dict(packet) == result
    packet["max_gradient_frobenius_squared"] = [0, 1]
    with pytest.raises(ValueError, match="replay"):
        cone_homothetic_from_dict(reseal(packet))


def test_extra_resealed_certificate_fields_refused():
    packet = cone_homothetic_to_dict(cone())
    packet["physical_force_work_admission"] = True
    with pytest.raises(ValueError, match="replay"):
        cone_homothetic_from_dict(reseal(packet))


def test_in_memory_certificate_tampering_cannot_be_serialized():
    with pytest.raises(ValueError, match="replay"):
        cone_homothetic_to_dict(replace(cone(), max_gradient_frobenius_squared=F(1, 2)))


def test_all_unordered_pairs_including_same_owner_have_full_time_proofs():
    cells = tuple(cell(str(i), shifted(TETRA, (3*i, 0, 0)),
                       shifted(TETRA, (3*i+F(1, 3), 0, 0))) for i in range(3))
    result = tetra_certificate(*cells)
    assert len(result.pair_separation_leaves) == 3
    assert all(leaves[0][:2] == (0, 1) for _, _, leaves in result.pair_separation_leaves)
    assert not result.physical_force_work_admission


def test_coincident_tetrahedra_not_accepted_by_identically_zero_edge_cross_axis():
    with pytest.raises(UnresolvedGenesisVolumeTemporal, match="overlap"):
        tetra_certificate(cell("a"), cell("b"))


def test_shared_node_does_not_skip_overlapping_pair():
    inside = ((0, 0, 0), (F(1, 2), 0, 0), (0, F(1, 2), 0), (0, 0, F(1, 2)))
    with pytest.raises(UnresolvedGenesisVolumeTemporal, match="overlap"):
        tetra_certificate(cell("a", vertex_ids=("shared", "a1", "a2", "a3")),
            cell("b", inside, vertex_ids=("shared", "b1", "b2", "b3")))


def test_shared_edge_does_not_skip_overlapping_pair():
    inside = ((0, 0, 0), (1, 0, 0), (0, F(1, 2), 0), (0, 0, F(1, 2)))
    with pytest.raises(UnresolvedGenesisVolumeTemporal, match="overlap"):
        tetra_certificate(cell("a", vertex_ids=("s0", "s1", "a2", "a3")),
            cell("b", inside, vertex_ids=("s0", "s1", "b2", "b3")))


def test_shared_face_does_not_skip_same_side_overlapping_pair():
    inside = TETRA[:3]+((0, 0, F(1, 2)),)
    with pytest.raises(UnresolvedGenesisVolumeTemporal, match="overlap"):
        tetra_certificate(cell("a", vertex_ids=("s0", "s1", "s2", "a3")),
            cell("b", inside, vertex_ids=("s0", "s1", "s2", "b3")))


def test_exact_shared_face_opposite_sides_accepts_boundary_touching_only():
    below = (TETRA[0], TETRA[2], TETRA[1], (0, 0, -1))
    result = tetra_certificate(cell("a", vertex_ids=("s0", "s1", "s2", "a3")),
        cell("b", below, owner=1, vertex_ids=("s0", "s2", "s1", "b3")))
    assert result.pair_separation_leaves[0][2][0][-1] == 0
    assert result.whole_clock_all_pair_noninterpenetration_certified
    assert not result.physical_force_work_admission


def test_shared_node_actual_path_disagreement_is_not_tolerated():
    with pytest.raises(ValueError, match="inconsistent"):
        tetra_certificate(cell("a", vertex_ids=("shared", "a1", "a2", "a3")),
            cell("b", shifted(TETRA, (2, 0, 0)), vertex_ids=("shared", "b1", "b2", "b3")))


def test_positive_endpoint_volumes_with_interior_inversion_are_refused():
    after = ((0, 0, 0), (-2, 0, 0), (0, F(-1, 2), 0), (0, 0, 1))
    assert np.linalg.det(np.asarray(after[1:], float)) > 0
    with pytest.raises(UnresolvedGenesisVolumeTemporal, match="nonpositive"):
        tetra_certificate(cell("inverted", after=after))


def test_separated_endpoints_with_interior_collision_are_refused():
    with pytest.raises(UnresolvedGenesisVolumeTemporal, match="overlap"):
        tetra_certificate(cell("left", shifted(TETRA, (-2, 0, 0)), shifted(TETRA, (2, 0, 0))),
            cell("right", shifted(TETRA, (2, 0, 0)), shifted(TETRA, (-2, 0, 0))))


def test_zero_thickness_birth_requires_distinct_declared_live_domain():
    born = TETRA[:3]+((0, 0, 0),)
    with pytest.raises(UnresolvedGenesisVolumeTemporal, match="nonpositive"):
        tetra_certificate(cell("born", before=born, after=TETRA))


def test_reverse_tangential_motion_control_keeps_geometry_and_external_law_separate():
    below = (TETRA[0], TETRA[2], TETRA[1], (0, 0, -1))
    # This is a small explicit geometry control, not the old native C5->C1
    # fixture. The latter requires its actual placements and archived memory.
    result = tetra_certificate(cell("upper", TETRA, owner=1),
        cell("lower", below, shifted(below, (F(-1, 1000), 0, 0)), owner=5))
    assert result.whole_clock_all_pair_noninterpenetration_certified
    assert not result.physical_force_work_admission


def test_resource_exhaustion_is_unresolved_not_success():
    with pytest.raises(UnresolvedGenesisVolumeTemporal, match="budget"):
        tetra_certificate(cell("a"), cell("b", shifted(TETRA, (2, 0, 0))), max_time_leaves=1)


def test_affine_tetrahedron_packet_declares_its_narrow_path_scope():
    packet = tetrahedron_paths_to_dict(tetra_certificate(cell("a")))
    assert packet["physical_path"] == "literal_affine_vertex_paths_only_not_bilinear_frustum_paths"
    assert not packet["physical_force_work_admission"]


def test_affine_tetrahedron_restart_replays_all_pairs_and_rejects_resealed_changes():
    result = tetra_certificate(cell("a"), cell("b", shifted(TETRA, (2, 0, 0))))
    packet = json.loads(json.dumps(tetrahedron_paths_to_dict(result)))
    assert tetrahedron_paths_from_dict(packet) == result
    packet["pair_separation_leaves"] = []
    with pytest.raises(ValueError, match="replay"):
        tetrahedron_paths_from_dict(reseal(packet))
