"""Independent low-cost geometry checks for declared current-plane supports."""
from dataclasses import FrozenInstanceError, replace
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from tectonics.genesis_volume_checkpoint import _canonical
from tectonics.genesis_volume_geometry import ConicalCohortGeometry, build_genesis_volume_geometry
from tectonics.genesis_volume_material_sections import (
    prepare_cohort_material_section, prepare_volume_material_sections,
    evaluate_cohort_material_section, volume_material_sections_to_dict,
    volume_material_sections_from_dict,
)


def _cohort():
    triangle = np.array(((0., 0., 10.), (2., 0., 10.), (0., 3., 10.)))
    qlo, qhi, coefficient = .1, .8, .5
    a, b = (1-coefficient*np.array((qlo, qhi)))**(1/3)
    volume = np.linalg.det(triangle)*(a**3-b**3)/6
    return ConicalCohortGeometry(17, 0, 2, qlo, qhi, 2*volume, volume,
        tuple(map(tuple, triangle)), a, b, (), "source-owner", "original-material"), coefficient


def _section(normal=(1., 0., 0.), offset=.7, amplitudes=None):
    cohort, coefficient = _cohort()
    if amplitudes is None:
        amplitudes = np.zeros((3, 6))
    return prepare_cohort_material_section(cohort, amplitudes, normal, offset,
        conical_mass_coefficient=coefficient, cohort_index=8)


def _independent_line_points(section, lam):
    # Direct Cartesian plane intersections, rather than the producer's
    # projected coefficient or ratio implementation.
    triangle = (lam*section.triangle_m+section.displacement_A_m
                +(lam-1)*section.slope_P_m)
    values = triangle@section.normal_xyz-section.offset_m
    points = []
    for i, j in ((0, 1), (0, 2), (1, 2)):
        if (values[i] < 0 < values[j]) or (values[j] < 0 < values[i]):
            weight = -values[i]/(values[j]-values[i])
            points.append((1-weight)*triangle[i]+weight*triangle[j])
    return points


def test_nonedge_radial_plane_sections_use_analytic_barycentric_intersections():
    section = _section()
    assert section.status == "interior_cut"
    assert section.interior_cut
    assert section.cohort_id == 17
    assert section.cohort_index == 8
    assert section.owner_id == "source-owner"
    assert section.material_id == "original-material"
    for interval in section.intervals:
        lam = sum(interval.lambda_bounds)/2
        for fraction in (0., .23, 1.):
            point = evaluate_cohort_material_section(section, lam, fraction=fraction)
            assert point.xyz_m[0] == pytest.approx(.7, abs=2e-15)
            assert point.barycentric[1] == pytest.approx(.7/(2*lam), abs=2e-15)
            assert point.q == pytest.approx((1-lam**3)/.5, abs=2e-15)
            assert abs(point.plane_residual_m) < 2e-15
        assert interval.slice_kind == "segment"
        assert len(interval.edge_pairs) == 2


def test_full_P_warp_is_pulled_back_instead_of_radially_copying_top_trace():
    amplitudes = np.array(((.1, .2, .3, .8, .1, -.2),
                           (-.1, .05, .04, -.4, .4, .6),
                           (.2, -.3, .15, .1, -.2, -.5)))
    cohort, _ = _cohort()
    lam0 = (cohort.inner_scale+cohort.outer_scale)/2
    normal = np.array((1., .7, .2))
    normal /= np.linalg.norm(normal)
    target = np.full(3, 1/3)@(lam0*np.array(cohort.top_triangle_m)+amplitudes[:, :3]
                            +(lam0-1)*amplitudes[:, 3:])
    section = _section(normal, normal@target, amplitudes)
    maximum_radial_copy_error = 0.
    for interval in section.intervals:
        lam = sum(interval.lambda_bounds)/2
        expected_points = _independent_line_points(section, lam)
        assert len(expected_points) == 2
        actual_points = [evaluate_cohort_material_section(section, lam, fraction=f).xyz_m
                         for f in (0., 1.)]
        for point in actual_points:
            assert min(np.linalg.norm(point-expected) for expected in expected_points) < 2e-14
        for f in (.11, .77):
            point = evaluate_cohort_material_section(section, lam, fraction=f)
            full = point.barycentric@(lam*np.array(cohort.top_triangle_m)
                +amplitudes[:, :3]+(lam-1)*amplitudes[:, 3:])
            np.testing.assert_allclose(point.xyz_m, full, rtol=2e-15, atol=2e-14)
            naive = lam*(point.barycentric@(np.array(cohort.top_triangle_m)+amplitudes[:, :3]))
            maximum_radial_copy_error = max(maximum_radial_copy_error, np.linalg.norm(full-naive))
            assert abs(normal@point.xyz_m-section.offset_m) < 2e-14
    assert maximum_radial_copy_error > .01


def test_oblique_declared_plane_changes_angular_location_with_depth():
    section = _section((1., 0., .2), 2.5)
    assert section.intervals
    locations = []
    for lam in np.linspace(*section.lambda_bounds, 7):
        point = evaluate_cohort_material_section(section, lam, fraction=.5)
        locations.append(point.barycentric)
        assert section.normal_xyz@point.xyz_m == pytest.approx(section.offset_m, abs=5e-15)
    assert np.linalg.norm(locations[-1]-locations[0]) > .01


def test_vertex_roots_partition_lambda_without_selecting_an_edge():
    section = _section((1., 0., 0.), 1.8)
    assert .9 in section.lambda_breakpoints
    assert section.lambda_breakpoints == (section.lambda_bounds[0], .9, section.lambda_bounds[1])
    assert section.intervals[0].lambda_bounds[0] == .9
    point = evaluate_cohort_material_section(section, .9)
    np.testing.assert_array_equal(point.barycentric, (0., 1., 0.))
    with pytest.raises(ValueError, match="no section"):
        evaluate_cohort_material_section(section, (section.lambda_bounds[0]+.9)/2)


def test_horizontal_internal_plane_retains_full_constant_lambda_triangle():
    cohort, _ = _cohort()
    lam = (cohort.inner_scale+cohort.outer_scale)/2
    section = _section((0., 0., 1.), 10*lam)
    assert section.status == "interior_cut"
    assert section.intervals == ()
    assert len(section.constant_lambda_sections) == 1
    support = section.constant_lambda_sections[0]
    assert support.lambda_value == pytest.approx(lam, abs=2e-16)
    assert support.location == "interior"
    with pytest.raises(ValueError, match="explicit barycentric"):
        evaluate_cohort_material_section(section, support.lambda_value)
    point = evaluate_cohort_material_section(section, support.lambda_value,
                                           barycentric=(.2, .3, .5))
    np.testing.assert_allclose(point.xyz_m, lam*np.array((.6, 1.5, 10.)), rtol=2e-15)
    assert point.q == pytest.approx((1-lam**3)/.5, abs=2e-15)


@pytest.mark.parametrize("boundary", ["inner", "outer"])
def test_horizontal_boundary_plane_never_admits_an_interior_cut(boundary):
    cohort, _ = _cohort()
    lam = cohort.inner_scale if boundary == "inner" else cohort.outer_scale
    section = _section((0., 0., 1.), 10*lam)
    assert not section.interior_cut
    assert section.status == "boundary_support"
    assert section.lambda_breakpoints == (cohort.inner_scale, cohort.outer_scale)
    assert section.constant_lambda_sections[0].location == f"{boundary}_boundary"
    point = evaluate_cohort_material_section(section, lam, barycentric=(1., 0., 0.))
    assert point.q == (cohort.live_q_hi if boundary == "inner" else cohort.q_lo)


def test_resolved_distinct_near_boundary_plane_keeps_its_interior_root_and_literal_end_q():
    cohort, _ = _cohort()
    requested = float(cohort.inner_scale)+1024*math.ulp(float(cohort.inner_scale))
    section = _section((0., 0., 1.), 10*requested)
    root = section.constant_lambda_sections[0].lambda_value
    assert root > cohort.inner_scale
    assert abs(root-cohort.inner_scale) > 4*math.ulp(float(cohort.inner_scale))
    assert abs(root-requested) <= math.ulp(requested)
    assert section.lambda_breakpoints == (cohort.inner_scale, root, cohort.outer_scale)
    assert section.constant_lambda_sections[0].location == "interior"
    assert section.interior_cut
    assert section.status == "interior_cut"
    point = evaluate_cohort_material_section(section, root, barycentric=(.2, .3, .5))
    assert cohort.q_lo < point.q < cohort.live_q_hi
    assert section.q_bounds == (cohort.q_lo, cohort.live_q_hi)


def test_angular_edge_boundary_has_no_interior_material_split():
    section = _section((1., 0., 0.), 0.)
    assert section.status == "boundary_support"
    assert not section.interior_cut
    assert all(interval.edge_pairs == ((0, 0), (2, 2)) for interval in section.intervals)
    point = evaluate_cohort_material_section(section, sum(section.lambda_bounds)/2, fraction=.25)
    np.testing.assert_array_equal(point.barycentric, (.75, 0., .25))


def test_isolated_vertex_touch_and_empty_plane_do_not_gain_surface_support():
    cohort, _ = _cohort()
    touch = _section((1., 1., 1.), 10*cohort.inner_scale)
    assert touch.status == "boundary_touch"
    assert not touch.interior_cut
    assert touch.intervals == touch.constant_lambda_sections == ()
    point = evaluate_cohort_material_section(touch, cohort.inner_scale)
    np.testing.assert_array_equal(point.barycentric, (1., 0., 0.))
    empty = _section((1., 0., 0.), 100.)
    assert empty.status == "no_intersection"
    assert empty.intervals == empty.constant_lambda_sections == ()
    assert not empty.interior_cut
    with pytest.raises(ValueError, match="no section"):
        evaluate_cohort_material_section(empty, sum(empty.lambda_bounds)/2)


def test_touching_radial_vertex_curve_has_only_point_slices():
    section = _section((1., 1., 0.), 0.)
    assert not section.interior_cut
    assert all(interval.slice_kind == "point" for interval in section.intervals)
    point = evaluate_cohort_material_section(section, sum(section.lambda_bounds)/2)
    np.testing.assert_array_equal(point.barycentric, (1., 0., 0.))


def test_coplanar_current_material_volume_is_refused():
    cohort, _ = _cohort()
    amplitudes = np.zeros((3, 6))
    amplitudes[:, 2] = amplitudes[:, 5] = -np.array(cohort.top_triangle_m)[:, 2]
    with pytest.raises(ValueError, match="coplanar"):
        _section((0., 0., 1.), 0., amplitudes)


def test_plane_scaling_and_orientation_reversal_preserve_geometric_support():
    original = _section((1., 2., .1), 2.)
    scaled = _section((100., 200., 10.), 200.)
    reversed_plane = _section((-1., -2., -.1), -2.)
    assert original.status == scaled.status == reversed_plane.status
    for interval in original.intervals:
        lam = sum(interval.lambda_bounds)/2
        expected = evaluate_cohort_material_section(original, lam, fraction=.31)
        for other in (scaled, reversed_plane):
            actual = evaluate_cohort_material_section(other, lam, fraction=.31)
            np.testing.assert_allclose(actual.xyz_m, expected.xyz_m, rtol=3e-15, atol=3e-14)
            np.testing.assert_allclose(actual.barycentric, expected.barycentric, rtol=3e-15, atol=3e-15)
            assert actual.q == expected.q


def test_section_fields_are_frozen_and_array_owners_cannot_be_made_writeable():
    section = _section()
    with pytest.raises(FrozenInstanceError):
        section.offset_m = 0.
    for field in (section.triangle_m, section.normal_xyz, section.projection_slopes,
                  section.displacement_A_m, section.slope_P_m):
        with pytest.raises(ValueError):
            field.setflags(write=True)
    point = evaluate_cohort_material_section(section, sum(section.lambda_bounds)/2)
    with pytest.raises(ValueError):
        point.xyz_m[0] = 0.
    with pytest.raises(ValueError, match="immutable"):
        evaluate_cohort_material_section(replace(section, offset_m=section.offset_m+1),
                                        sum(section.lambda_bounds)/2)


@pytest.mark.parametrize("normal", [
    (0., 0., 0.), (True, False, True), (1., float("nan"), 0.),
    (1., 0., float("inf")), (1., 0.), (1+1j, 0, 0), ("1", "0", "0"),
])
def test_invalid_plane_normals_are_refused(normal):
    with pytest.raises(ValueError):
        _section(normal)


@pytest.mark.parametrize("offset", [True, float("nan"), float("inf"), 1j, "1"])
def test_invalid_plane_offsets_are_refused(offset):
    with pytest.raises(ValueError):
        _section(offset=offset)


@pytest.mark.parametrize("amplitudes", [
    np.zeros((3, 5)), np.full((3, 6), np.nan), np.zeros((3, 6), dtype=bool),
    np.zeros((3, 6), dtype=complex), np.full((3, 6), "0"),
])
def test_invalid_current_amplitudes_are_refused(amplitudes):
    with pytest.raises(ValueError):
        _section(amplitudes=amplitudes)


@pytest.mark.parametrize("coefficient", [0., -1., True, float("nan"), 2.])
def test_invalid_or_mismatched_cohort_q_charts_are_refused(coefficient):
    cohort, _ = _cohort()
    with pytest.raises(ValueError):
        prepare_cohort_material_section(cohort, np.zeros((3, 6)), (1., 0., 0.), .7,
                                        conical_mass_coefficient=coefficient)


@pytest.mark.parametrize("fraction", [-.1, 1.1, True, float("nan")])
def test_invalid_section_fractions_are_refused(fraction):
    section = _section()
    with pytest.raises(ValueError):
        evaluate_cohort_material_section(section, sum(section.lambda_bounds)/2, fraction=fraction)


@pytest.mark.parametrize("barycentric", [
    (-.1, .2, .9), (.1, .2, .3), (1., 0., 0.), (True, False, False),
    (.2, .3, float("nan")),
])
def test_invalid_or_off_plane_barycentric_points_are_refused(barycentric):
    section = _section()
    with pytest.raises(ValueError):
        evaluate_cohort_material_section(section, sum(section.lambda_bounds)/2,
                                        barycentric=barycentric)


@pytest.fixture(scope="module")
def source_geometry():
    path = Path(__file__).resolve().parents[1]/(
        "analysis/genesis_cohort_mechanics_validation/evidence/control/material_export.json")
    original = path.read_bytes()
    geometry = build_genesis_volume_geometry(json.loads(original))
    yield geometry
    assert path.read_bytes() == original


def test_declared_source_support_retains_all_original_cohorts_without_history_updates(source_geometry):
    geometry = source_geometry
    amplitudes = np.zeros(6*len(geometry.vertices_m))
    face = np.array(geometry.faces[0])
    triangle = np.array(geometry.vertices_m)[face]
    normal = triangle[1]-triangle[0]
    normal /= np.linalg.norm(normal)
    cohort = next(c for c in geometry.cohorts if c.face_index == 0)
    point = ((cohort.inner_scale+cohort.outer_scale)/2)*triangle.mean(axis=0)
    sections = prepare_volume_material_sections(geometry, amplitudes, normal, normal@point,
                                               face_indices=[0])
    raw = volume_material_sections_to_dict(sections)
    assert raw["source_checkpoint_sha256"] == geometry.source_checkpoint_sha256
    assert raw["source_identity"] == geometry.source_identity
    assert raw["time_myr"] == geometry.time_myr
    selected = [c for c in geometry.cohorts if c.face_index == 0]
    assert len(sections.cohorts) == len(selected)
    for record, original in zip(sections.cohorts, selected):
        assert record.cohort_id == original.id
        assert record.q_bounds == (original.q_lo, original.live_q_hi)
        assert record.mass_kg == original.mass_kg
        assert record.volume_m3 == original.volume_m3
        assert record.owner_id == original.owner_id
        assert record.material_id == original.material_id
    for key in ("calculated_location_selected", "physical_creation_admitted",
                "physical_release_admitted", "physical_contact_admitted",
                "material_memory_changed", "dofs_created_or_released", "clock_advanced"):
        assert raw[key] is False
    assert raw["source_creation_energy_law"] is None
    assert raw["creation_work_debited_from_Genesis_j"] == 0
    assert raw["integration_scope"] == "analytic_slice_geometry_no_area_or_volume_quadrature_claim"
    restored = volume_material_sections_from_dict(raw, geometry=geometry, displacement_m=amplitudes,
        normal_xyz=normal, offset_m=normal@point, face_indices=[0])
    assert restored.binding_sha256 == sections.binding_sha256
    assert volume_material_sections_to_dict(restored) == raw
    raw["cohorts"][0]["q_bounds"][0] += .001
    raw["sections_sha256"] = hashlib.sha256(_canonical({k: v for k, v in raw.items()
        if k != "sections_sha256"})).hexdigest()
    with pytest.raises(ValueError, match="full material/support reconstruction"):
        volume_material_sections_from_dict(raw, geometry=geometry, displacement_m=amplitudes,
            normal_xyz=normal, offset_m=normal@point, face_indices=[0])


def test_empty_face_selection_is_an_empty_support_without_manufactured_material(source_geometry):
    sections = prepare_volume_material_sections(source_geometry,
        np.zeros(6*len(source_geometry.vertices_m)), (1., 0., 0.), 0., face_indices=[])
    assert sections.face_indices == sections.cohorts == ()


def test_archive_boolean_type_substitution_is_refused_even_with_old_equal_python_values(source_geometry):
    amplitudes = np.zeros(6*len(source_geometry.vertices_m))
    support = prepare_volume_material_sections(source_geometry, amplitudes,
        (1., 0., 0.), 0., face_indices=[])
    raw = volume_material_sections_to_dict(support)
    raw["physical_creation_admitted"] = 0
    assert raw["physical_creation_admitted"] == False
    with pytest.raises(ValueError, match="full material/support reconstruction"):
        volume_material_sections_from_dict(raw, geometry=source_geometry,
            displacement_m=amplitudes, normal_xyz=(1., 0., 0.), offset_m=0., face_indices=[])


@pytest.mark.parametrize("faces", [[True], [0., 1.], [0, 0], [-1], [80], [[0]]])
def test_invalid_selected_faces_are_refused(source_geometry, faces):
    with pytest.raises(ValueError):
        prepare_volume_material_sections(source_geometry,
            np.zeros(6*len(source_geometry.vertices_m)), (1., 0., 0.), 0., face_indices=faces)
