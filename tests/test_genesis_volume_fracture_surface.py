"""Finite source-side surfaces, six-DOF jump traces, and exact virtual work."""
from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tectonics.genesis_volume_banks import prepare_volume_banks
from tectonics.genesis_volume_fracture_surface import (
    _interface_intervals, evaluate_surface_jump, integrate_surface_traction,
    prepare_volume_fracture_surface, validate_volume_fracture_surface,
)
from tectonics.genesis_volume_operator import assemble_genesis_volume_operator


@pytest.fixture(scope="module")
def parent():
    path = Path(__file__).resolve().parents[1]/"analysis/genesis_cohort_mechanics_validation/evidence/control/material_export.json"
    return assemble_genesis_volume_operator(json.loads(path.read_text()))


@pytest.fixture(scope="module")
def banks(parent):
    a, b, c = parent.geometry.faces[0]
    # An explicitly supplied computational closed loop, not a Genesis fault.
    return prepare_volume_banks(parent, ((a, b), (b, c), (c, a)))


@pytest.fixture(scope="module")
def surface(banks):
    return prepare_volume_fracture_surface(banks)


def test_surface_remains_source_bound_and_prospective(parent, banks, surface):
    assert surface.banks is banks
    assert surface.bank_binding_sha256 == banks.binding_sha256
    assert surface.source_checkpoint_sha256 == parent.source_checkpoint_sha256
    assert surface.time_myr == parent.time_myr
    assert surface.patch_count > len(surface.cut_edges)
    assert not surface.calculated_fracture_surface
    assert not surface.physical_creation_admitted
    assert not surface.physical_contact_release_admitted
    assert not surface.physical_force_work_admission
    assert validate_volume_fracture_surface(surface, banks=banks) is surface
    np.testing.assert_array_equal(surface.cut_edges, banks.topology.cut_edges)


def test_material_side_q_charts_are_not_equated(parent, surface):
    assert np.max(np.abs(surface.side_q_bounds[:, 0]-surface.side_q_bounds[:, 1])) > 1e-6
    for patch, ((lo, hi), indices, ids, q_bounds) in enumerate(zip(
            surface.lambda_bounds, surface.side_cohort_indices,
            surface.side_cohort_ids, surface.side_q_bounds)):
        for side, (index, identity, (q_lo, q_hi)) in enumerate(zip(indices, ids, q_bounds)):
            cohort = parent.geometry.cohorts[index]
            assert identity == cohort.id
            assert cohort.face_index == surface.side_face_indices[surface.patch_edge_indices[patch], side]
            assert surface.side_owner_ids[patch][side] == cohort.owner_id
            assert surface.side_material_ids[patch][side] == cohort.material_id
            assert cohort.inner_scale <= lo < hi <= cohort.outer_scale
            assert cohort.q_lo <= q_lo < q_hi <= cohort.live_q_hi
            triangle = np.asarray(cohort.top_triangle_m)
            # Independent q→lambda reconstruction in the physical volume chart.
            coefficient = 6*parent.geometry.column_mass_kg[cohort.face_index]/(
                parent.geometry.density_kg_m3*np.linalg.det(triangle))
            reconstructed = np.cbrt(1-coefficient*np.array((q_hi, q_lo)))
            np.testing.assert_allclose(reconstructed, (lo, hi), rtol=0, atol=2e-16)


def test_positive_surface_area_counts_one_paired_interface_and_unmatched_depth(parent, surface):
    vertices = np.asarray(parent.geometry.vertices_m)
    assert np.all(surface.area_weights_m2 > 0)
    assert np.all(surface.patch_area_m2 > 0)
    np.testing.assert_allclose(surface.area_weights_m2.sum(axis=1), surface.patch_area_m2,
                               rtol=4e-16, atol=0)
    assert len(surface.unmatched_edge_indices) > 0
    for edge, ((u, v), face_ids) in enumerate(zip(surface.cut_edges, surface.side_face_indices)):
        # Integrate the whole radial edge side analytically, independent of
        # the producer's cohort subdivision and patch summation.
        jacobian = np.linalg.norm(np.cross(vertices[u], vertices[v]))
        bottom = [min(c.inner_scale for c in parent.geometry.cohorts if c.face_index == face)
                  for face in face_ids]
        overlap_bottom = max(bottom)
        expected = .5*jacobian*(1-overlap_bottom)*(1+overlap_bottom)
        assert surface.edge_area_m2[edge] == pytest.approx(expected, rel=2e-15)
        for side in (0, 1):
            extra = surface.unmatched_area_m2[(surface.unmatched_edge_indices == edge)
                                             & (surface.unmatched_side_indices == side)].sum()
            whole = .5*jacobian*(1-bottom[side])*(1+bottom[side])
            assert surface.edge_area_m2[edge]+extra == pytest.approx(whole, rel=2e-15)
    assert surface.total_area_m2 == surface.edge_area_m2.sum()


def test_normals_and_tangents_are_oriented_by_original_material_sides(parent, surface):
    vertices, faces = np.asarray(parent.geometry.vertices_m), np.asarray(parent.geometry.faces)
    for edge, (u, v) in enumerate(surface.cut_edges):
        n, t1, t2 = (values[edge] for values in
                    (surface.edge_normals_xyz, surface.edge_tangent1_xyz, surface.edge_tangent2_xyz))
        frame = np.column_stack((n, t1, t2))
        np.testing.assert_allclose(frame.T@frame, np.eye(3), rtol=0, atol=4e-16)
        assert np.linalg.det(frame) == pytest.approx(1., rel=5e-16)
        assert t1@(vertices[v]-vertices[u]) > 0
        for side, sign in ((0, -1), (1, 1)):
            third = next(x for x in faces[surface.side_face_indices[edge, side]] if x not in (u, v))
            assert sign*n@vertices[third] > 0
    for patch, edge in enumerate(surface.patch_edge_indices):
        u, v = surface.cut_edges[edge]
        x = surface.quadrature_lambda[patch, :, None]*(
            (1-surface.quadrature_s[patch, :, None])*vertices[u]
            +surface.quadrature_s[patch, :, None]*vertices[v])
        np.testing.assert_array_equal(surface.quadrature_xyz_m[patch], x)


def test_all_six_tied_components_have_zero_physical_surface_jump(banks, surface):
    q = np.random.default_rng(29133).normal(size=banks.parent_assembly.ndof)
    jump = evaluate_surface_jump(surface, banks.R@q)
    assert np.max(np.abs(jump)) < 3e-16
    traction = np.random.default_rng(289).normal(size=jump.shape)*3e6
    force = integrate_surface_traction(surface, traction)
    assert np.linalg.norm(banks.R.T@force) < 2e-15*np.linalg.norm(force)


@pytest.mark.parametrize("component", range(6))
def test_each_independent_A_and_P_component_has_actual_surface_trace(banks, surface, component):
    q = np.zeros(banks.ndof)
    q[banks.parent_assembly.ndof+component] = .017
    actual = evaluate_surface_jump(surface, q)
    expected = np.zeros_like(actual)
    for patch, edge in enumerate(surface.patch_edge_indices):
        bank_vertices = banks.topology.bank_vertices[edge]
        for bank, sign in ((0, -1.), (1, 1.)):
            for endpoint, vertex in enumerate(bank_vertices[bank]):
                if 6*vertex+component == banks.parent_assembly.ndof+component:
                    shape = surface.quadrature_s[patch] if endpoint else 1-surface.quadrature_s[patch]
                    radial = surface.quadrature_lambda[patch]-1 if component >= 3 else 1.
                    expected[patch, :, component % 3] += sign*.017*shape*radial
    np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=1e-22)
    assert np.max(np.abs(actual)) > 0


def test_virtual_work_action_reaction_and_traction_gradient(banks, surface):
    rng = np.random.default_rng(81732)
    q = rng.normal(size=banks.ndof)*.007
    dq = rng.normal(size=banks.ndof)*.01
    # World traction varies over both angular and radial coordinates.
    traction = rng.normal(size=(surface.patch_count, 4, 3))*5e6
    force = integrate_surface_traction(surface, traction)
    jump = evaluate_surface_jump(surface, q)
    expected = np.einsum("pq,pqa,pqa->", surface.area_weights_m2, traction, jump)
    assert q@force == pytest.approx(expected, rel=6e-15)
    # Every rigid translation has zero virtual work: opposite banks carry
    # equal and opposite resultant, including shared crack-tip accumulation.
    for axis in np.eye(3):
        rigid = np.zeros((banks.ndof//6, 6))
        rigid[:, :3] = axis
        assert abs(rigid.ravel()@force) < 3e-15*np.linalg.norm(force)
    # The field is exactly linear; a finite displacement probe avoids loss
    # from subtracting nearly equal O(1e12 J) work values.
    h = .1
    plus = np.einsum("pq,pqa,pqa->", surface.area_weights_m2, traction,
                      evaluate_surface_jump(surface, q+h*dq))
    minus = np.einsum("pq,pqa,pqa->", surface.area_weights_m2, traction,
                       evaluate_surface_jump(surface, q-h*dq))
    assert (plus-minus)/(2*h) == pytest.approx(dq@force, rel=3e-13)


def test_quadratic_jump_work_matches_higher_order_independent_surface_quadrature(banks, surface):
    q = np.random.default_rng(391).normal(size=banks.ndof)*.01
    jump = evaluate_surface_jump(surface, q)
    producer = .5*np.einsum("pq,pqa,pqa->", surface.area_weights_m2, jump, jump)
    nodes, gauss_weights = np.polynomial.legendre.leggauss(6)
    ss, sw = .5*(1+nodes), .5*gauss_weights
    vertices = np.asarray(banks.parent_assembly.geometry.vertices_m)
    total = 0.
    for edge, (u, v) in enumerate(surface.cut_edges):
        jacobian = np.linalg.norm(np.cross(vertices[u], vertices[v]))
        selected = surface.lambda_bounds[surface.patch_edge_indices == edge]
        lo, hi = selected[0, 0], selected[-1, 1]
        lam, lw = lo+.5*(hi-lo)*(1+nodes), .5*(hi-lo)*gauss_weights
        nodal = q.reshape(-1, 6)[banks.topology.bank_vertices[edge]]
        for s, ws in zip(ss, sw):
            amplitudes = (1-s)*nodal[:, 0]+s*nodal[:, 1]
            for radial, wr in zip(lam, lw):
                field = amplitudes[:, :3]+(radial-1)*amplitudes[:, 3:]
                delta = field[1]-field[0]
                total += .5*ws*wr*radial*jacobian*(delta@delta)
    assert producer == pytest.approx(total, rel=5e-15)


def test_isolated_tips_cancel_trace_and_empty_support_stays_empty(parent):
    a, b, _ = parent.geometry.faces[0]
    single = prepare_volume_fracture_surface(prepare_volume_banks(parent, ((a, b),)))
    q = np.random.default_rng(228).normal(size=single.banks.ndof)
    assert np.max(np.abs(evaluate_surface_jump(single, q))) < 3e-16
    force = integrate_surface_traction(single, np.full((single.patch_count, 4, 3), 8e6))
    assert np.max(np.abs(force)) == 0
    empty = prepare_volume_fracture_surface(prepare_volume_banks(parent, []))
    assert empty.patch_count == 0 and empty.total_area_m2 == 0
    assert empty.jump_trace.shape == (0, 4, 3, 24)
    assert evaluate_surface_jump(empty, np.zeros(empty.banks.ndof)).shape == (0, 4, 3)
    np.testing.assert_array_equal(integrate_surface_traction(empty, np.empty((0, 4, 3))),
                                  np.zeros(empty.banks.ndof))


@pytest.mark.parametrize("seed", (418, 419, 420))
def test_random_unequal_fronts_partition_one_side_only_and_paired_coverage(seed):
    # Geometry-only probe, never an invented accepted Genesis state.
    rng = np.random.default_rng(seed)
    sides = []
    for side in (0, 1):
        bottom = rng.uniform(.3, .8)
        levels = np.r_[bottom, np.sort(rng.uniform(bottom, 1., size=13+side)), 1.]
        sides.append([(100*side+i, SimpleNamespace(inner_scale=lo, outer_scale=hi))
                      for i, (lo, hi) in enumerate(zip(levels, levels[1:]))])
    intervals = _interface_intervals(sides)
    area = np.zeros(2)
    paired = 0.
    for lo, hi, owners in intervals:
        weight = .5*(hi-lo)*(hi+lo)
        if None not in owners:
            paired += weight
        for side, identity in enumerate(owners):
            if identity is not None:
                area[side] += weight
                cohort = dict(sides[side])[identity]
                assert cohort.inner_scale <= lo < hi <= cohort.outer_scale
    bottom = [current[0][1].inner_scale for current in sides]
    np.testing.assert_allclose(area, .5*(1-np.square(bottom)), rtol=2e-15)
    assert paired == pytest.approx(.5*(1-max(bottom)**2), rel=3e-15)


def test_overlapping_cohort_ownership_is_rejected():
    bad = [[(0, SimpleNamespace(inner_scale=.4, outer_scale=1.)),
            (1, SimpleNamespace(inner_scale=.5, outer_scale=.8))], []]
    with pytest.raises(ValueError, match="overlapping"):
        _interface_intervals(bad)


@pytest.mark.parametrize("field", ("side_cohort_ids", "side_q_bounds", "lambda_bounds",
                                   "area_weights_m2", "jump_trace", "trace_dofs", "edge_normals_xyz"))
def test_changed_surface_field_is_rejected_by_original_binding(surface, field):
    changed = getattr(surface, field).copy()
    changed.flat[0] += 1
    with pytest.raises(ValueError, match="binding"):
        validate_volume_fracture_surface(replace(surface, **{field: changed}))


def test_surface_is_immutable_and_bound_to_exact_banks(parent, surface):
    with pytest.raises(FrozenInstanceError):
        surface.time_myr += 1
    with pytest.raises(ValueError):
        surface.jump_trace.setflags(write=True)
    other = prepare_volume_banks(parent, [])
    with pytest.raises(ValueError, match="different"):
        validate_volume_fracture_surface(surface, banks=other)


@pytest.mark.parametrize("bad", (True, 1j, float("nan"), []))
def test_invalid_jump_and_traction_fields_are_rejected(surface, bad):
    with pytest.raises(ValueError):
        evaluate_surface_jump(surface, np.full(surface.banks.ndof, bad) if bad != [] else [])
    with pytest.raises(ValueError):
        integrate_surface_traction(surface,
            np.full((surface.patch_count, 4, 3), bad) if bad != [] else [])
