"""Analytic geometry, full-mode work, material advection and source refusals.

The small conical fixtures are explicitly manufactured geometry.  The source
fixture is the unchanged real formation export; no test creates a planetary
fracture or replaces the history by a mature initial condition.
"""
from dataclasses import FrozenInstanceError, replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tectonics.genesis_volume_checkpoint import _canonical
from tectonics.genesis_volume_geometry import ConicalCohortGeometry, build_genesis_volume_geometry
from tectonics.genesis_volume_material_sections import prepare_cohort_material_section
from tectonics.genesis_volume_embedded_surface import (
    EmbeddedSurfaceRefusal, prepare_cohort_embedded_surface, prepare_volume_embedded_surface,
    validate_volume_embedded_surface, evaluate_embedded_surface_trace,
    evaluate_embedded_surface_positions, evaluate_embedded_surface_jump,
    integrate_embedded_surface_traction, integrate_embedded_surface_virtual_work,
    volume_embedded_surface_to_dict, volume_embedded_surface_from_dict,
)


X = np.array(((0., 0., 10.), (2., 0., 10.), (0., 3., 10.)))
WARP = np.array(((.8781137771339679, .543831412520262, .6754167910986415,
                  -.757524872286168, .6069121506802474, -.11771253139559015),
                 (-.13276268538817584, .7843965189250424, -.30574534461487113,
                  -.8957517537261137, -.3286498419843098, -.8377790136550155),
                 (-1.2878999996781384, -.10188987889767726, .27879528124588066,
                  -1.5134428923312426, .010134114448797306, .17863868099772717)))


def _cohort(*, deep=False):
    coefficient = .95 if deep else .5
    qlo, qhi = (.01, .99) if deep else (.1, .8)
    a, b = (1-coefficient*np.array((qlo, qhi)))**(1/3)
    volume = np.linalg.det(X)*(a**3-b**3)/6
    cohort = ConicalCohortGeometry(17, 0, 2, qlo, qhi, 2*volume, volume,
        tuple(map(tuple, X)), a, b, (), "source-owner", "original-material")
    return cohort, coefficient


def _section(normal=(1., 0., 0.), offset=.7, U=None, *, deep=False):
    cohort, coefficient = _cohort(deep=deep)
    return prepare_cohort_material_section(cohort, np.zeros((3, 6)) if U is None else U,
        normal, offset, conical_mass_coefficient=coefficient, cohort_index=8)


@pytest.fixture(scope="module")
def warped():
    normal = np.array((1., .7, .2))
    normal /= np.linalg.norm(normal)
    return _section(normal, 2.199963621561037, WARP, deep=True)


def test_vertical_nonedge_surface_area_matches_independent_cartesian_polygon_integral():
    section = _section()
    surface = prepare_cohort_embedded_surface(section, quadrature_order=2)
    b, a = section.lambda_bounds
    # On x=.7: z=10lambda, 0<=y<=3lambda-1.5*.7.
    expected = 15*(a*a-b*b)-15*.7*(a-b)
    assert surface.total_current_area_m2 == pytest.approx(expected, rel=2e-15)
    assert surface.total_reference_area_m2 == pytest.approx(expected, rel=2e-15)
    np.testing.assert_allclose(surface.current_position_m[:, 0], .7, atol=2e-15)
    np.testing.assert_allclose(surface.current_normal_xyz, np.tile((1., 0., 0.), (4, 1)), atol=3e-15)
    assert surface.patch_count == 1


def test_horizontal_full_triangle_area_and_quadratic_moments_are_analytic():
    cohort, _ = _cohort()
    lam = .5*(cohort.outer_scale+cohort.inner_scale)
    surface = prepare_cohort_embedded_surface(_section((0., 0., 1.), 10*lam), quadrature_order=2)
    expected = 3*lam*lam
    assert surface.patch_kinds == ("constant_lambda_triangle",)
    assert surface.total_current_area_m2 == pytest.approx(expected, rel=2e-15)
    N, weights = surface.barycentric, surface.current_area_weights_m2
    np.testing.assert_allclose(np.einsum("pi,p->i", N, weights), expected/3, rtol=3e-15)
    matrix = np.einsum("pi,pj,p->ij", N, N, weights)
    np.testing.assert_allclose(matrix, expected*(np.ones((3, 3))+np.eye(3))/12, rtol=4e-15)


def test_affine_current_metrics_follow_independent_nanson_identity():
    F = np.array(((1.15, .25, .07), (.1, .8, .03), (0., 0., 1.2)))
    U = np.concatenate((X@(F-np.eye(3)).T, X@(F-np.eye(3)).T), axis=1)
    cohort, _ = _cohort()
    lam = .5*(cohort.outer_scale+cohort.inner_scale)
    normal = np.linalg.solve(F.T, (0., 0., 1.))
    normal_magnitude = np.linalg.norm(normal)
    normal /= normal_magnitude
    section = _section(normal, 10*lam/normal_magnitude, U)
    surface = prepare_cohort_embedded_surface(section, quadrature_order=3)
    area = .5*lam**2*np.linalg.norm(np.cross(F@(X[1]-X[0]), F@(X[2]-X[0])))
    assert surface.total_current_area_m2 == pytest.approx(area, rel=3e-15)
    assert surface.total_reference_area_m2 == pytest.approx(3*lam**2, rel=4e-15)
    np.testing.assert_allclose(surface.current_to_reference_volume_ratio, np.linalg.det(F), rtol=4e-15)
    reference = surface.reference_normal_xyz*surface.reference_area_weights_m2[:, None]
    nanson = np.linalg.det(F)*np.linalg.solve(F.T, reference.T).T
    np.testing.assert_allclose(surface.current_normal_xyz*surface.current_area_weights_m2[:, None],
                               nanson, rtol=6e-15, atol=2e-15)


def test_dense_affine_oblique_current_plane_has_exact_reference_metric_transform():
    F = np.array(((1.15, .25, .07), (.1, .8, .03), (.05, -.06, 1.2)))
    shift = np.array((.25, -.13, .21))
    P = X@(F-np.eye(3)).T
    U = np.concatenate((P+shift, P), axis=1)
    cohort, _ = _cohort()
    lam = .5*(cohort.outer_scale+cohort.inner_scale)
    normal = np.array((1., .7, .2)); normal /= np.linalg.norm(normal)
    offset = float(normal@(lam*(np.array((.2, .3, .5))@X)@F.T+shift))
    surface = prepare_cohort_embedded_surface(_section(normal, offset, U), quadrature_order=4)
    np.testing.assert_allclose(surface.current_position_m,
        surface.reference_position_m@F.T+shift, rtol=2e-15, atol=3e-15)
    transformed = F.T@normal
    reference_normal = transformed/np.linalg.norm(transformed)
    np.testing.assert_allclose(surface.reference_normal_xyz,
        np.tile(reference_normal, (surface.point_count, 1)), rtol=3e-15, atol=3e-15)
    ratio = np.linalg.norm(transformed)/np.linalg.det(F)
    np.testing.assert_allclose(surface.reference_area_weights_m2/surface.current_area_weights_m2,
                               ratio, rtol=3e-15)


def _independent_section_point(section, lam, fraction, edges):
    # Direct plane intersections in Cartesian space; no producer derivatives.
    Y = lam*np.array(section.triangle_m)+section.displacement_A_m+(lam-1)*section.slope_P_m
    values = Y@section.normal_xyz-section.offset_m
    angular = []
    for i, j in edges:
        N = np.zeros(3)
        if i == j:
            N[i] = 1.
        else:
            t = -values[i]/(values[j]-values[i])
            N[i], N[j] = 1-t, t
        angular.append(N)
    N = (1-fraction)*angular[0]+fraction*angular[1]
    return N@Y, lam*(N@np.array(section.triangle_m)), N


def test_full_P_surface_derivatives_include_moving_angular_intersection(warped):
    surface = prepare_cohort_embedded_surface(warped, quadrature_order=3)
    index = 4
    lam = surface.radial_scale[index]
    patch = surface.point_patch_indices[index]
    interval = warped.intervals[patch]
    # With order 3, point 4 uses middle lambda and middle segment fraction.
    assert surface.barycentric[index].sum() == pytest.approx(1.)
    step = 1e-6
    plus = _independent_section_point(warped, lam+step, .5, interval.edge_pairs)
    minus = _independent_section_point(warped, lam-step, .5, interval.edge_pairs)
    np.testing.assert_allclose(surface.current_differentials_m[index, 1],
                               (plus[0]-minus[0])/(2*step), rtol=3e-10, atol=2e-9)
    np.testing.assert_allclose(surface.reference_differentials_m[index, 1],
                               (plus[1]-minus[1])/(2*step), rtol=3e-10, atol=2e-9)
    omitted = surface.barycentric[index]@(X+WARP[:, 3:])
    assert np.linalg.norm(omitted-surface.current_differentials_m[index, 1]) > .1
    assert abs(surface.current_differentials_m[index, 1]@warped.normal_xyz) < 2e-14


def test_general_full_P_area_converges_with_radial_quadrature(warped):
    surfaces = [prepare_cohort_embedded_surface(warped, quadrature_order=o) for o in (4, 8, 12, 24)]
    reference = surfaces[-1].total_current_area_m2
    errors = [abs(s.total_current_area_m2-reference) for s in surfaces[:-1]]
    assert errors[0] > 1e-6  # The generic integral is not declared exact by finite Gauss.
    assert errors[1] < 1e-10 and errors[1] < errors[0]/10000
    assert errors[2] < 1e-12
    assert abs(surfaces[-1].total_current_area_m2-surfaces[-1].total_reference_area_m2) > .1


@pytest.mark.parametrize("area", ["current", "reference"])
def test_full_A_P_paired_force_is_discrete_virtual_work_dual(warped, area):
    surface = prepare_cohort_embedded_surface(warped, quadrature_order=4)
    rng = np.random.default_rng(192)
    paired = rng.normal(size=(2, 18))*.02
    traction = rng.normal(size=(surface.point_count, 3))*1e7
    force = integrate_embedded_surface_traction(surface, traction, area_convention=area)
    work = integrate_embedded_surface_virtual_work(surface, traction, paired, area_convention=area)
    assert float(np.sum(force*paired)) == pytest.approx(work, rel=3e-15, abs=1e-9)
    np.testing.assert_array_equal(force[0], -force[1])
    assert np.linalg.norm(force[:, 3::6]) > 0  # The depth slopes retain conjugate force.


def test_identical_virtual_banks_remain_zero_jump_and_introduce_no_mesh_dofs(warped):
    surface = prepare_cohort_embedded_surface(warped)
    parent = np.linspace(-.2, .3, 18)
    np.testing.assert_array_equal(evaluate_embedded_surface_jump(surface, np.stack((parent, parent))),
                                  np.zeros((surface.point_count, 3)))
    local_tied = np.concatenate((np.eye(18), np.eye(18)), axis=0)
    np.testing.assert_array_equal(surface.paired_jump_trace@local_tied,
                                  np.zeros((surface.point_count, 3, 18)))
    assert not surface.physical_release_admitted and not surface.dofs_created_or_released


def test_frozen_material_support_advects_full_modes_without_reintersecting_plane(warped):
    surface = prepare_cohort_embedded_surface(warped)
    np.testing.assert_allclose(evaluate_embedded_surface_positions(surface, WARP.ravel()),
                               surface.current_position_m, rtol=2e-15, atol=3e-15)
    before = volume_embedded_surface_to_dict(surface)
    new = WARP.copy()
    new[:, :3] += (.4, .1, -.2)
    new[:, 3:] += (.3, -.2, .6)
    expected = surface.reference_position_m+surface.barycentric@new[:, :3]
    expected += (surface.radial_scale[:, None]-1)*(surface.barycentric@new[:, 3:])
    positions = evaluate_embedded_surface_positions(surface, new.ravel())
    np.testing.assert_allclose(positions, expected, rtol=3e-15, atol=3e-15)
    assert np.max(np.abs(positions@warped.normal_xyz-warped.offset_m)) > .01
    assert volume_embedded_surface_to_dict(surface) == before


@pytest.mark.parametrize("kind", ["horizontal_inner", "horizontal_outer", "angular_edge", "vertex", "miss"])
def test_boundary_only_supports_do_not_gain_or_duplicate_interior_area(kind):
    cohort, _ = _cohort()
    n, d = {
        "horizontal_inner": ((0., 0., 1.), 10*cohort.inner_scale),
        "horizontal_outer": ((0., 0., 1.), 10*cohort.outer_scale),
        "angular_edge": ((1., 0., 0.), 0.),
        "vertex": ((1., 1., 1.), 10*cohort.inner_scale),
        "miss": ((1., 0., 0.), 100.),
    }[kind]
    surface = prepare_cohort_embedded_surface(_section(n, d))
    assert surface.point_count == surface.patch_count == 0
    assert surface.total_current_area_m2 == surface.total_reference_area_m2 == 0
    assert surface.parent_trace.shape == (0, 3, 18)
    assert integrate_embedded_surface_traction(surface, np.empty((0, 3)),
        area_convention="current").shape == (2, 18)


def test_inverted_current_volume_chart_is_refused_despite_positive_surface_area():
    F = np.diag((-1., 1., 1.))
    U = np.concatenate((X@(F-np.eye(3)).T, X@(F-np.eye(3)).T), axis=1)
    section = _section((1., 0., 0.), -.7, U)
    with pytest.raises(EmbeddedSurfaceRefusal, match="inverted"):
        prepare_cohort_embedded_surface(section)


def test_internal_radial_volume_fold_is_refused_even_when_endpoints_are_positive():
    V = np.array(((0., 0., 10.), (1., 0., 10.), (0., 1., 10.)))
    W = np.array(((0., 0., 0.), (-.55, 0., 0.), (0., -.75, 0.)))
    P = V-X
    U = np.concatenate((W+P, P), axis=1)
    section = _section((1., .7, .2), 1.2, U, deep=True)
    b, a = section.lambda_bounds
    assert (b-.55)*(b-.75) > 0 and (a-.55)*(a-.75) > 0
    with pytest.raises(EmbeddedSurfaceRefusal, match="inverted"):
        prepare_cohort_embedded_surface(section)


def test_arrays_and_full_binding_are_immutable(warped):
    surface = prepare_cohort_embedded_surface(warped)
    with pytest.raises(FrozenInstanceError):
        surface.parent_ndof = 36
    for array in (surface.current_position_m, surface.current_area_weights_m2,
                  surface.paired_jump_trace, surface.trace_dofs):
        with pytest.raises(ValueError):
            array.setflags(write=True)
    with pytest.raises(EmbeddedSurfaceRefusal, match="immutable"):
        validate_volume_embedded_surface(replace(surface, parent_ndof=36))


@pytest.mark.parametrize("order", [True, 1, 65, 4., "8"])
def test_invalid_quadrature_orders_are_refused(order):
    with pytest.raises(EmbeddedSurfaceRefusal, match="quadrature"):
        prepare_cohort_embedded_surface(_section(), quadrature_order=order)


@pytest.mark.parametrize("bad", [True, np.full(18, np.nan), np.ones(17), np.ones((3, 6)),
                                  [True]+[0.]*17, np.ones(18, dtype=complex)])
def test_invalid_parent_trace_fields_are_refused(bad):
    surface = prepare_cohort_embedded_surface(_section(), quadrature_order=2)
    with pytest.raises(EmbeddedSurfaceRefusal):
        evaluate_embedded_surface_trace(surface, bad)


@pytest.mark.parametrize("bad", [None, "born", True, ("current",), np.array(("current",))])
def test_area_convention_is_explicit_and_validated(bad):
    surface = prepare_cohort_embedded_surface(_section(), quadrature_order=2)
    with pytest.raises(EmbeddedSurfaceRefusal, match="area convention"):
        integrate_embedded_surface_traction(surface, np.ones((surface.point_count, 3)), area_convention=bad)


def test_nonfinite_traction_force_and_virtual_work_are_refused():
    surface = prepare_cohort_embedded_surface(_section(), quadrature_order=2)
    with pytest.raises(EmbeddedSurfaceRefusal):
        integrate_embedded_surface_traction(surface, np.full((4, 3), np.inf), area_convention="current")
    with pytest.raises(EmbeddedSurfaceRefusal):
        integrate_embedded_surface_virtual_work(surface, np.full((4, 3), 1e308),
            np.stack((np.zeros(18), np.full(18, 1e308))), area_convention="current")


@pytest.fixture(scope="module")
def authentic():
    path = Path(__file__).resolve().parents[1]/(
        "analysis/genesis_cohort_mechanics_validation/evidence/control/material_export.json")
    original = path.read_bytes()
    geometry = build_genesis_volume_geometry(json.loads(original))
    U = np.zeros(6*len(geometry.vertices_m))
    face = 0
    cohort = next(c for c in geometry.cohorts if c.face_index == face)
    N = np.full(3, 1/3)
    normal = np.array((1., 2., 3.)); normal /= np.linalg.norm(normal)
    lam = .5*(cohort.outer_scale+cohort.inner_scale)
    offset = float(normal@(lam*(N@np.array(cohort.top_triangle_m))))
    surface = prepare_volume_embedded_surface(geometry, U, normal, offset,
        face_indices=[face], cohort_ids=[cohort.id], quadrature_order=4)
    yield geometry, U, normal, offset, cohort.id, surface
    assert path.read_bytes() == original


def test_authentic_source_and_original_global_trace_dofs_remain_bound(authentic):
    geometry, U, normal, offset, cohort_id, surface = authentic
    assert surface.source_geometry_bound
    assert surface.source_checkpoint_sha256 == geometry.source_checkpoint_sha256
    assert surface.parent_ndof == 6*len(geometry.vertices_m)
    assert surface.selected_cohort_ids == (cohort_id,)
    np.testing.assert_array_equal(surface.cohort_ids, np.full(surface.point_count, cohort_id))
    expected = (6*np.array(geometry.faces[0])[:, None]+np.arange(6)).reshape(18)
    np.testing.assert_array_equal(surface.trace_dofs, np.tile(expected, (surface.point_count, 1)))
    np.testing.assert_allclose(surface.current_position_m@normal, offset, atol=2e-9)
    assert surface.point_count > 0
    payload = volume_embedded_surface_to_dict(surface)
    assert payload["bulk_energy_force_identity_claimed"] is False
    assert payload["finite_stress_pushforward_claimed"] is False
    assert payload["creation_work_debited_from_Genesis_j"] == 0


def test_authentic_surface_canonical_roundtrip_reconstructs_all_metrics(authentic):
    geometry, U, normal, offset, cohort_id, surface = authentic
    raw = volume_embedded_surface_to_dict(surface)
    restored = volume_embedded_surface_from_dict(raw, geometry=geometry, displacement_m=U,
        normal_xyz=normal, offset_m=offset, face_indices=[0], cohort_ids=[cohort_id], quadrature_order=4)
    assert restored.binding_sha256 == surface.binding_sha256
    assert volume_embedded_surface_to_dict(restored) == raw


@pytest.mark.parametrize("field", ["current_area_weights_m2", "reference_area_weights_m2",
                                  "mass_coordinate_q", "parent_trace", "physical_release_admitted"])
def test_resealed_source_archives_cannot_replace_metric_memory_coordinates_or_claims(authentic, field):
    geometry, U, normal, offset, cohort_id, surface = authentic
    raw = volume_embedded_surface_to_dict(surface)
    if field == "physical_release_admitted":
        raw[field] = True
    elif field == "parent_trace":
        raw[field][0][0][3] += .01
    else:
        raw[field][0] += .01
    body = {k: v for k, v in raw.items() if k != "embedded_surface_sha256"}
    raw["embedded_surface_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
    with pytest.raises(EmbeddedSurfaceRefusal, match="reconstruction"):
        volume_embedded_surface_from_dict(raw, geometry=geometry, displacement_m=U,
            normal_xyz=normal, offset_m=offset, face_indices=[0], cohort_ids=[cohort_id], quadrature_order=4)


def test_empty_original_material_subset_retains_authentic_binding(authentic):
    geometry, U, normal, offset, _, _ = authentic
    surface = prepare_volume_embedded_surface(geometry, U, normal, offset,
        face_indices=[0], cohort_ids=[], quadrature_order=2)
    assert surface.source_geometry_bound and surface.point_count == 0
    assert surface.selected_cohort_ids == ()
    assert surface.material_sections.cohorts


def test_archived_third_plane_literal_inputs_rebuild_source_binding_without_renormalization(authentic):
    geometry, U, _, _, _, _ = authentic
    # These are the previous accepted computational face79 plane inputs.  The
    # existing normalizer changes offset by one ULP on each of its first two
    # calls, so normalized output is not interchangeable with constructor input.
    normal = np.array((.38018781261549783, -.7060630805716389, .5974379912529252))
    offset = 509450.6672730301
    surface = prepare_volume_embedded_surface(geometry, U, normal, offset,
        face_indices=[79], quadrature_order=2)
    np.testing.assert_array_equal(surface.declared_normal_xyz, normal)
    assert surface.declared_offset_m == offset
    with pytest.raises(ValueError):
        surface.declared_normal_xyz.setflags(write=True)
    rebuilt = prepare_volume_embedded_surface(geometry, U,
        surface.declared_normal_xyz, surface.declared_offset_m,
        face_indices=surface.material_sections.face_indices,
        cohort_ids=surface.selected_cohort_ids, quadrature_order=surface.quadrature_order)
    assert _canonical(volume_embedded_surface_to_dict(rebuilt)) == _canonical(
        volume_embedded_surface_to_dict(surface))
    normalized_rebuild = prepare_volume_embedded_surface(geometry, U,
        surface.material_sections.normal_xyz, surface.material_sections.offset_m,
        face_indices=[79], quadrature_order=2)
    assert normalized_rebuild.material_sections.binding_sha256 != surface.material_sections.binding_sha256


@pytest.mark.parametrize("ids", [[-1], [True], [0, 0], [[0]], [np.iinfo(np.uint64).max]])
def test_material_subset_cannot_invent_cohorts(authentic, ids):
    geometry, U, normal, offset, _, _ = authentic
    with pytest.raises(EmbeddedSurfaceRefusal):
        prepare_volume_embedded_surface(geometry, U, normal, offset, face_indices=[0],
            cohort_ids=ids, quadrature_order=2)
