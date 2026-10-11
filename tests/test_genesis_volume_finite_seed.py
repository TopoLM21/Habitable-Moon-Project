"""Independent analytic/manufactured checks of finite original-material seeds."""
from dataclasses import FrozenInstanceError, replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tectonics.genesis_volume_checkpoint import _canonical
from tectonics.genesis_volume_geometry import ConicalCohortGeometry, build_genesis_volume_geometry
from tectonics.genesis_volume_material_sections import prepare_cohort_material_section
from tectonics.genesis_volume_finite_seed import (
    FiniteSeedRefusal, prepare_cohort_finite_seed, prepare_volume_finite_seed,
    validate_volume_finite_seed, volume_finite_seed_to_dict, volume_finite_seed_from_dict,
)
from tectonics.genesis_volume_embedded_surface import (
    prepare_cohort_embedded_surface, evaluate_embedded_surface_positions,
    integrate_embedded_surface_traction, integrate_embedded_surface_virtual_work,
)


X = np.array(((0., 0., 10.), (2., 0., 10.), (0., 3., 10.)))
WARP = np.array(((.8781137771339679, .543831412520262, .6754167910986415,
                  -.757524872286168, .6069121506802474, -.11771253139559015),
                 (-.13276268538817584, .7843965189250424, -.30574534461487113,
                  -.8957517537261137, -.3286498419843098, -.8377790136550155),
                 (-1.2878999996781384, -.10188987889767726, .27879528124588066,
                  -1.5134428923312426, .010134114448797306, .17863868099772717)))


def _cohort():
    coefficient = .5
    qlo, qhi = .1, .8
    a, b = (1-coefficient*np.array((qlo, qhi)))**(1/3)
    volume = np.linalg.det(X)*(a**3-b**3)/6
    return ConicalCohortGeometry(17, 0, 2, qlo, qhi, 2*volume, volume,
        tuple(map(tuple, X)), a, b, (), "source-owner", "original-material"), coefficient


def _section(normal=(1., 0., 0.), offset=.7, U=None):
    cohort, coefficient = _cohort()
    return prepare_cohort_material_section(cohort, np.zeros((3, 6)) if U is None else U,
        normal, offset, conical_mass_coefficient=coefficient, cohort_index=8)


def _middle():
    cohort, _ = _cohort()
    return .5*(cohort.outer_scale+cohort.inner_scale)


@pytest.mark.parametrize("radius", [.05, .000013])
def test_tiny_inside_disk_is_clipped_even_when_both_endpoints_and_old_gauss_nodes_are_outside(radius):
    section = _section()
    center = np.array((.7, .51, 10*_middle()))
    old = prepare_cohort_embedded_surface(section, quadrature_order=2)
    assert np.min(np.linalg.norm(old.current_position_m-center, axis=1)) > radius
    seed = prepare_cohort_finite_seed(section, center, radius, quadrature_order=16)
    surface = seed.surface
    assert surface.total_current_area_m2 == pytest.approx(np.pi*radius**2, rel=8e-9)
    np.testing.assert_allclose(surface.total_reference_area_m2, surface.total_current_area_m2, rtol=3e-15)
    assert np.all(np.linalg.norm(surface.current_position_m-center, axis=1) <= radius*(1+1e-10))
    assert np.all(surface.barycentric > 0)
    assert surface.patch_count >= 1


def test_corner_seed_has_finite_quarter_disk_area_and_remains_one_original_cohort():
    cohort, _ = _cohort()
    center = np.array((0., 0., 10*cohort.outer_scale))
    normal = np.array((1., 0., .1))
    section = _section(normal, normal@center)
    seed = prepare_cohort_finite_seed(section, center, .03, quadrature_order=18)
    assert seed.surface.total_current_area_m2 == pytest.approx(np.pi*.03**2/4, rel=3e-12)
    assert seed.surface.selected_cohort_ids == (17,)
    assert set(seed.surface.cohort_ids) == {17}
    assert set(seed.surface.owner_ids) == {"source-owner"}
    assert not seed.partitions_cohort_into_disconnected_banks


@pytest.mark.parametrize("radius", [.02, .04, .08])
def test_corner_area_scales_as_radius_squared(radius):
    cohort, _ = _cohort()
    center = np.array((0., 0., 10*cohort.outer_scale))
    section = _section((1., 0., .1), cohort.outer_scale)
    seed = prepare_cohort_finite_seed(section, center, radius, quadrature_order=18)
    assert seed.surface.total_current_area_m2/radius**2 == pytest.approx(np.pi/4, rel=3e-12)


def test_large_sphere_recovers_whole_section_and_finite_subset_leaves_ligament():
    section = _section()
    center = np.array((.7, .51, 10*_middle()))
    whole = prepare_cohort_embedded_surface(section, quadrature_order=16)
    large = prepare_cohort_finite_seed(section, center, 100., quadrature_order=16)
    small = prepare_cohort_finite_seed(section, center, .04, quadrature_order=16)
    assert large.surface.total_current_area_m2 == pytest.approx(whole.total_current_area_m2, rel=5e-15)
    assert small.surface.total_current_area_m2 < whole.total_current_area_m2/100
    packet = volume_finite_seed_to_dict(small)
    assert packet["finite_front_and_surrounding_material_not_released"]
    assert not packet["cohort_disconnection_certified"]
    assert not packet["partitions_cohort_into_disconnected_banks"]
    assert not packet["physical_creation_admitted"]
    assert not packet["physical_release_admitted"]


@pytest.mark.parametrize("kind", ["angular_boundary", "outer_boundary", "sphere_miss", "sphere_tangent"])
def test_boundary_only_miss_and_tangent_have_no_positive_interior_area(kind):
    cohort, _ = _cohort()
    section, center = {
        "angular_boundary": (_section((1., 0., 0.), 0.), (0., .5, 10*_middle())),
        "outer_boundary": (_section((0., 0., 1.), 10*cohort.outer_scale), (.5, .5, 10*cohort.outer_scale)),
        "sphere_miss": (_section(), (.7, 100., 10*_middle())),
        "sphere_tangent": (_section(), (.7, -.05, 10*_middle())),
    }[kind]
    with pytest.raises(FiniteSeedRefusal, match="positive relative-interior"):
        prepare_cohort_finite_seed(section, center, .05)


def test_horizontal_triangle_disk_clipping_has_exact_area_and_cartesian_moments():
    lam = _middle()
    center = np.array((.5, .5, 10*lam))
    radius = .15
    seed = prepare_cohort_finite_seed(_section((0., 0., 1.), 10*lam), center, radius,
        quadrature_order=12)
    surface = seed.surface
    area = np.pi*radius**2
    assert set(surface.patch_kinds) == {"sphere_clipped_constant_lambda_triangle"}
    assert surface.total_current_area_m2 == pytest.approx(area, rel=4e-15)
    delta = surface.current_position_m-center
    np.testing.assert_allclose(np.einsum("pi,p->i", delta, surface.current_area_weights_m2), 0., atol=2e-16)
    moment = np.einsum("pi,pj,p->ij", delta, delta, surface.current_area_weights_m2)
    np.testing.assert_allclose(moment, np.diag((area*radius**2/4, area*radius**2/4, 0)), atol=4e-17)


def test_horizontal_disk_keeps_arithmetic_center_plane_residual_in_its_sphere_radius():
    # Public constructor permits only a roundoff-size normal residual.  The
    # exact clipping helper must still use R²-delta², even near resolution.
    from tectonics.genesis_volume_finite_seed import _triangle_polar
    lam = _middle()
    section = _section((0., 0., 1.), 10*lam)
    center = np.array((.5, .5, section.offset_m+2e-13))
    radius = 2.5e-13
    delta = np.longdouble(section.offset_m)-np.longdouble(center[2])
    expected = float(np.sqrt(np.longdouble(radius)**2-delta**2))
    angles, clip, _, _ = _triangle_polar(section, lam, center, radius)
    for theta in (0., .4, 1.2, 2.1, 4.3, 5.7):
        assert clip(theta)[2] == expected
    assert expected < .8*radius


def test_horizontal_corner_disk_has_quarter_area():
    lam = _middle()
    center = np.array((0., 0., 10*lam))
    seed = prepare_cohort_finite_seed(_section((0., 0., 1.), 10*lam), center, .1,
        quadrature_order=12)
    assert seed.surface.total_current_area_m2 == pytest.approx(np.pi*.1**2/4, rel=4e-15)


def test_horizontal_seed_center_outside_triangle_is_actually_clipped_to_circular_segment():
    lam = _middle()
    radius, distance = .1, .04
    center = np.array((.5, -distance, 10*lam))
    seed = prepare_cohort_finite_seed(_section((0., 0., 1.), 10*lam), center, radius,
        quadrature_order=24)
    expected = radius**2*np.arccos(distance/radius)-distance*np.sqrt(radius**2-distance**2)
    assert seed.surface.total_current_area_m2 == pytest.approx(expected, rel=2e-14)
    assert np.min(seed.surface.current_position_m[:, 1]) > 0


def test_affine_dense_A_P_finite_disk_has_independent_reference_area_transform():
    F = np.array(((1.15, .25, .07), (.1, .8, .03), (.05, -.06, 1.2)))
    shift = np.array((.25, -.13, .21))
    P = X@(F-np.eye(3)).T
    U = np.concatenate((P+shift, P), axis=1)
    N = np.array((.3, .3, .4))
    center = (_middle()*(N@X))@F.T+shift
    normal = np.array((1., .7, .2)); normal /= np.linalg.norm(normal)
    seed = prepare_cohort_finite_seed(_section(normal, normal@center, U), center, .025,
        quadrature_order=18)
    surface = seed.surface
    assert surface.total_current_area_m2 == pytest.approx(np.pi*.025**2, rel=8e-13)
    ratio = np.linalg.norm(F.T@normal)/np.linalg.det(F)
    assert surface.total_reference_area_m2 == pytest.approx(surface.total_current_area_m2*ratio, rel=5e-14)
    np.testing.assert_allclose(surface.current_to_reference_volume_ratio, np.linalg.det(F), rtol=5e-15)


def test_warped_finite_sheet_has_full_chart_metrics_and_convergent_reference_area():
    normal = np.array((1., .7, .2)); normal /= np.linalg.norm(normal)
    N, lam = np.array((.2, .3, .5)), _middle()
    center = N@(lam*(X+WARP[:, 3:])+WARP[:, :3]-WARP[:, 3:])
    section = _section(normal, normal@center, WARP)
    seeds = [prepare_cohort_finite_seed(section, center, .08, quadrature_order=o) for o in (4, 8, 12, 20)]
    areas = np.array([s.surface.total_reference_area_m2 for s in seeds])
    assert abs(areas[0]-areas[-1]) > 1e-7
    assert abs(areas[1]-areas[-1]) < abs(areas[0]-areas[-1])/100
    assert abs(areas[2]-areas[-1]) < 1e-12
    surface = seeds[-1].surface
    assert surface.total_current_area_m2 == pytest.approx(np.pi*.08**2, rel=3e-12)
    assert abs(surface.total_current_area_m2-surface.total_reference_area_m2) > 1e-3
    assert np.all(surface.mass_coordinate_q > section.q_bounds[0])
    assert np.all(surface.mass_coordinate_q < section.q_bounds[1])


@pytest.mark.parametrize("area", ["current", "reference"])
def test_finite_surface_retains_full_original_modes_and_discrete_force_work_dual(area):
    center = np.array((.7, .51, 10*_middle()))
    seed = prepare_cohort_finite_seed(_section(), center, .1, quadrature_order=8)
    surface = seed.surface
    rng = np.random.default_rng(281)
    fields = rng.normal(size=(2, 18))*.02
    traction = rng.normal(size=(surface.point_count, 3))*1e7
    force = integrate_embedded_surface_traction(surface, traction, area_convention=area)
    work = integrate_embedded_surface_virtual_work(surface, traction, fields, area_convention=area)
    assert float(np.sum(force*fields)) == pytest.approx(work, rel=1e-14, abs=1e-9)
    assert np.linalg.norm(force[:, 3::6]) > 0
    np.testing.assert_array_equal(surface.paired_jump_trace@np.concatenate((np.eye(18), np.eye(18))),
        np.zeros((surface.point_count, 3, 18)))


def test_frozen_finite_seed_follows_original_material_even_after_leaving_birth_sphere():
    center = np.array((.7, .51, 10*_middle()))
    seed = prepare_cohort_finite_seed(_section(), center, .04)
    before = _canonical(volume_finite_seed_to_dict(seed))
    U = np.zeros((3, 6)); U[:, :3] = (1., .2, -.3); U[:, 3:] = (.4, -.7, .8)
    position = evaluate_embedded_surface_positions(seed.surface, U.ravel())
    expected = seed.surface.reference_position_m+seed.surface.barycentric@U[:, :3]
    expected += (seed.surface.radial_scale[:, None]-1)*(seed.surface.barycentric@U[:, 3:])
    np.testing.assert_allclose(position, expected, atol=2e-15)
    assert np.min(np.linalg.norm(position-center, axis=1)) > seed.seed_radius_m
    assert _canonical(volume_finite_seed_to_dict(seed)) == before


def test_normal_reversal_preserves_area_and_flips_both_oriented_normals():
    center = np.array((.7, .51, 10*_middle()))
    a = prepare_cohort_finite_seed(_section(), center, .04)
    b = prepare_cohort_finite_seed(_section((-1., 0., 0.), -.7), center, .04)
    assert a.surface.total_current_area_m2 == b.surface.total_current_area_m2
    np.testing.assert_allclose(a.surface.current_normal_xyz, -b.surface.current_normal_xyz)
    np.testing.assert_allclose(a.surface.reference_normal_xyz, -b.surface.reference_normal_xyz)


@pytest.mark.parametrize("radius", [True, 0., -1., float("nan"), float("inf"), "1", 1e-20])
def test_invalid_or_unresolved_radius_refused(radius):
    with pytest.raises(FiniteSeedRefusal, match="radius"):
        prepare_cohort_finite_seed(_section(), (.7, .5, 10*_middle()), radius)


@pytest.mark.parametrize("center", [(1., .5, 10.), (True, .5, 10.), (float("nan"), .5, 10.), (.7, .5)])
def test_invalid_or_off_plane_center_refused(center):
    with pytest.raises(FiniteSeedRefusal):
        prepare_cohort_finite_seed(_section(), center, .1)


@pytest.mark.parametrize("order", [True, 1, 65, 4., "8"])
def test_invalid_quadrature_order_is_typed_refusal(order):
    with pytest.raises(FiniteSeedRefusal, match="quadrature order"):
        prepare_cohort_finite_seed(_section(), (.7, .5, 10*_middle()), .1,
            quadrature_order=order)


def test_seed_and_surface_are_immutable_and_tampering_refused():
    seed = prepare_cohort_finite_seed(_section(), (.7, .5, 10*_middle()), .1)
    with pytest.raises(ValueError):
        seed.seed_center_world_m[0] = 0
    with pytest.raises(FrozenInstanceError):
        seed.seed_radius_m = 1
    with pytest.raises(FiniteSeedRefusal, match="binding"):
        validate_volume_finite_seed(replace(seed, seed_radius_m=.2))


def test_warped_corner_can_support_locally_even_when_whole_cohort_plane_cuts_interior():
    # Independent review counterexample: det current J=lambda>0 everywhere,
    # but all plane-interior points near this corner require lambda<.9.
    # Whole-cohort min/max therefore cannot certify a finite corner seed.
    reference = np.array(((-.5, -.5, 1.), (.5, -.5, 1.), (-.5, .5, 1.)))
    V = np.array(((-.1, 0., 1.), (.9, -1., 1.), (-.1, 0., 1.)))
    W = np.array(((.1, 0., 0.), (-.8, 0., 0.), (1.1, 0., 0.)))
    coefficient = .6
    qlo, qhi = 0., (1-.8**3)/coefficient
    volume = np.linalg.det(reference)*(1-.8**3)/6
    cohort = ConicalCohortGeometry(18, 0, 2, qlo, qhi, 2*volume, volume,
        tuple(map(tuple, reference)), 1., .8, (), "source-owner", "original-material")
    U = np.concatenate((V+W-reference, V-reference), axis=1)
    section = prepare_cohort_material_section(cohort, U, (1., 0., 0.), 0.,
        conical_mass_coefficient=coefficient)
    assert section.interior_cut
    assert prepare_cohort_embedded_surface(section).total_current_area_m2 > 0
    with pytest.raises(FiniteSeedRefusal, match="positive relative-interior"):
        prepare_cohort_finite_seed(section, (0., 0., 1.), .05)


@pytest.fixture(scope="module")
def source_geometry():
    path = Path(__file__).resolve().parents[1]/"analysis/genesis_cohort_mechanics_validation/evidence/control/material_export.json"
    original = path.read_bytes()
    geometry = build_genesis_volume_geometry(json.loads(original))
    yield geometry
    assert path.read_bytes() == original


def test_source_geometry_uses_literal_plane_and_selected_original_cohort_roundtrip(source_geometry):
    geometry = source_geometry
    cohort = geometry.cohorts[0]
    face = np.asarray(geometry.faces[cohort.face_index], int)
    lam = .5*(cohort.outer_scale+cohort.inner_scale)
    center = lam*np.asarray(geometry.vertices_m)[face].mean(axis=0)
    normal = np.array((.38018781261549783, -.7060630805716389, .5974379912529252))
    offset = float(normal@center)
    U = np.zeros(6*len(geometry.vertices_m))
    seed = prepare_volume_finite_seed(geometry, U, normal, offset, center, 100., cohort_id=cohort.id)
    packet = volume_finite_seed_to_dict(seed)
    restored = volume_finite_seed_from_dict(packet, geometry=geometry, displacement_m=U,
        normal_xyz=normal, offset_m=offset, seed_center_world_m=center, seed_radius_m=100.,
        cohort_id=cohort.id)
    assert _canonical(volume_finite_seed_to_dict(restored)) == _canonical(packet)
    np.testing.assert_array_equal(seed.surface.declared_normal_xyz, normal)
    assert seed.surface.declared_offset_m == offset
    assert seed.surface.selected_cohort_ids == (cohort.id,)
    assert seed.surface.parent_ndof == len(U)
    assert seed.surface.source_geometry_bound


def test_source_resealed_finite_seed_tampering_is_rejected_by_full_reconstruction(source_geometry):
    geometry = source_geometry
    cohort = geometry.cohorts[0]
    face = np.asarray(geometry.faces[cohort.face_index], int)
    center = .5*(cohort.outer_scale+cohort.inner_scale)*np.asarray(geometry.vertices_m)[face].mean(axis=0)
    normal = np.array((1., .3, .4))
    offset = float(normal@center)
    U = np.zeros(6*len(geometry.vertices_m))
    seed = prepare_volume_finite_seed(geometry, U, normal, offset, center, 100., cohort_id=cohort.id)
    packet = volume_finite_seed_to_dict(seed)
    packet["seed_radius_m"] *= 2
    body = {k:v for k,v in packet.items() if k != "finite_seed_sha256"}
    packet["finite_seed_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
    with pytest.raises(FiniteSeedRefusal, match="complete original material"):
        volume_finite_seed_from_dict(packet, geometry=geometry, displacement_m=U, normal_xyz=normal,
            offset_m=offset, seed_center_world_m=center, seed_radius_m=100., cohort_id=cohort.id)


@pytest.mark.parametrize("cohort_id", [True, -1, 999999999, "1"])
def test_source_unknown_or_invalid_original_cohort_id_refused(source_geometry, cohort_id):
    with pytest.raises(FiniteSeedRefusal, match="cohort ID"):
        prepare_volume_finite_seed(source_geometry, np.zeros(6*len(source_geometry.vertices_m)),
            (1., 0., 0.), 1., (1., 0., 0.), 100., cohort_id=cohort_id)
