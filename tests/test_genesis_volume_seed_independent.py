"""Independent finite-seed geometry and compact-basis mathematical probes."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from analysis.genesis_volume_seed_validation import independent_audit as audit

@pytest.fixture
def authentic_reaction_difference():
    """Frozen cohort-1 quadrature witnesses from the actual 800-cohort parent.

    Parent raw SHA: 62579a0bad128830a66e2f44579cf2b1f5d3835b691225b8d431102f3e4486d7.
    Diagnostic case raw SHA: e9de3c772fad1230ac62e4b2fa3b73fa1f7df79c6f34d6e0a34e278a6b8185e8.
    Both original force comparisons pass, while the old residual check fails.
    These numerical witnesses create no material or planetary runtime state.
    """
    reaction = np.array([17174312.076537423,-6474715.15217528,48108350.09706801,-57.70245249737093,21.75382215088789,-161.63499223271145,17174312.076539855,-6474715.152175511,48108350.09706772,-57.70245290048404,21.753822302859216,-161.63499336188025,17174312.076551326,-6474715.152183383,48108350.09708097,-57.70245751739019,21.75382404344462,-161.6350062946048])
    surface = np.array([17171316.57944442,-6473585.849861643,48099959.16866292,-57.69238856814959,21.75002839187199,-161.60680001762955,17171316.57944591,-6473585.849860676,48099959.16866322,-57.69238897542492,21.750028545409645,-161.60680115846836,17171316.579452112,-6473585.849861821,48099959.16867976,-57.69239364002075,21.750030303958134,-161.606814224841])
    difference = np.array([2995.497093003243,-1129.3023136369884,8390.928405091166,-0.010063929221338697,0.0037937590158989565,-0.028192215081901395,2995.4970939457417,-1129.3023148356006,8390.92840450257,-0.010063925059121459,0.0037937574495714443,-0.02819220341189066,2995.497099213302,-1129.3023215616122,8390.928401209414,-0.010063877369439922,0.003793739486486203,-0.028192069763804284])
    independent_reaction = np.array([17174312.076537155,-6474715.152176099,48108350.097069465,-57.70245249737051,21.753822150890855,-161.6349922327178,17174312.076539587,-6474715.15217633,48108350.097069174,-57.70245290048362,21.75382230286218,-161.6349933618866,17174312.076551057,-6474715.152184201,48108350.09708242,-57.702457517389774,21.753824043447583,-161.63500629461114])
    independent_surface = np.array([17171316.581723984,-6473585.850721041,48099959.17504838,-57.692388575719036,21.75002839472569,-161.60680003883294,17171316.58172547,-6473585.850720073,48099959.17504868,-57.69238898299435,21.75002854826332,-161.6068011796717,17171316.581731685,-6473585.850721221,48099959.17506524,-57.692393647590215,21.75003030681182,-161.6068142460444])
    return difference, reaction, surface, -reaction, independent_reaction, independent_surface


def test_authentic_small_reaction_difference_propagates_operand_error(authentic_reaction_difference):
    d, r, f, g, ri, fi = authentic_reaction_difference
    audit.mode_errors(r, ri, "original independent reaction", rtol=5e-7, atol=1e-5)
    audit.mode_errors(f, fi, "original independent surface force", rtol=3e-7, atol=1e-5)
    with pytest.raises(ValueError, match="A differs"):
        audit.mode_errors(d, ri-fi, "old ill-conditioned reaction difference", rtol=5e-7, atol=1e-5)
    report = audit.audit_reaction_difference(d, r, f, g, ri, fi, "authentic difference")
    assert report["literal_force_difference_exact"] is True
    assert report["literal_negative_bulk_reaction_exact"] is True
    assert report["independent_operand_tolerances_unchanged"] is True
    assert report["small_difference_relative_accuracy_certified"] is False


@pytest.mark.parametrize("forgery", ["zero_difference", "surface_as_bulk_reaction"])
def test_exact_force_identities_reject_fabricated_zero_reaction_difference(authentic_reaction_difference, forgery):
    d, r, f, g, ri, fi = authentic_reaction_difference
    if forgery == "surface_as_bulk_reaction":
        r = f.copy()
    with pytest.raises(ValueError, match="literal force difference|negative archived bulk gradient"):
        audit.audit_reaction_difference(np.zeros_like(d), r, f, g, ri, fi, "forged difference")


def test_consistent_difference_cannot_hide_incorrect_independent_surface_force(authentic_reaction_difference):
    d, r, f, g, ri, fi = authentic_reaction_difference
    f[0] += 1000.
    with pytest.raises(ValueError, match="independent surface force A differs"):
        audit.audit_reaction_difference(r-f, r, f, g, ri, fi, "changed force")


@pytest.fixture
def authentic_represented_plane():
    # Frozen face79 witness; the two equivalent normalization orders differ
    # by one ulp in their coefficients, with an independently checked plane.
    return (2.0318680071795825e-10,
        np.array([0.3801878126154979, -0.7060630805716389, 0.5974379912529252]),
        509455.74179683614,
        np.array([4613545.570985762, 1762448.4079764485, -264.97302823846616]),
        np.array([0.3801878126154979, -0.706063080571639, 0.5974379912529253]),
        509455.7417968362)


def test_authentic_plane_residual_keeps_literal_represented_coefficients(authentic_represented_plane):
    saved, n, d, x, ni, di = authentic_represented_plane
    independent = float(np.asarray(ni, np.longdouble)@np.asarray(x, np.longdouble)-np.longdouble(di))
    assert abs(saved-independent) > 1e-12
    report = audit.audit_plane_center_residual(saved, n, d, x, ni, di)
    assert report["literal_archived_center_residual_exact"] is True


@pytest.mark.parametrize("forgery", ["zero_residual", "translated_plane"])
def test_plane_residual_guard_rejects_fabricated_zero_or_changed_plane(authentic_represented_plane, forgery):
    saved, n, d, x, ni, di = authentic_represented_plane
    if forgery == "zero_residual":
        saved = 0.
    else:
        d += 1e-6
        saved = float(np.asarray(n, np.longdouble)@np.asarray(x, np.longdouble)-np.longdouble(d))
    with pytest.raises(ValueError, match="literal plane center residual|represented plane offset"):
        audit.audit_plane_center_residual(saved, n, d, x, ni, di)


def parent(*, pose=False):
    X = np.array(((-1., 0., 1.), (1., 0., 1.), (0., 2., 1.)))
    u = np.array(((.03, -.02, .01, -.01, .04, .02),
        (-.01, .03, .02, .03, -.02, .01), (.02, .01, -.03, -.02, .01, -.04))) if pose else np.zeros((3, 6))
    e0 = np.array(((.01, .02, -.01, .03, -.02, .04),
        (.02, -.01, .03, .02, .01, -.03), (-.01, .03, .02, -.02, .04, .01)))
    e1 = .2*e0+np.arange(18).reshape(3, 6)*.0001
    coefficient, density = .9, 2.
    cohort = dict(id=5, face_index=0, q_lo=0., live_q_hi=(1-.5**3)/coefficient,
        inner_scale=.5, outer_scale=1., owner_id="manufactured-body", material_id="manufactured-material")
    geometry = dict(vertices_m=X.tolist(), faces=[[0, 1, 2]],
        column_mass_kg=[coefficient*density*np.linalg.det(X.T)/6], density_kg_m3=density,
        cohorts=[cohort], source_identity="manufactured_only_not_planet", time_myr=.1)
    return dict(checkpoint_sha256="manufactured_only", source_checkpoint_sha256=audit.FORMATION_SHA256,
        source_packet=dict(current_geometry=dict(face_frames_xyz=np.eye(3)[:, :2][None].tolist()),
            source=dict(parameters=dict(shell=dict(young_modulus_pa=71., poisson_ratio=.23, residual_stiffness=.02)))),
        runtime_state=dict(displacement_m=u.ravel().tolist(), velocity_m_s=np.zeros(18).tolist(),
            fault_history=dict(damage=[.17]), geometry=geometry, accepted_step_sizes_myr=[.0001]*4,
            mechanics=dict(state=dict(accepted_steps=4, cohorts=[dict(id=5,
                strain0=(audit.prior.SOURCE_ANGULAR_POINTS@e0).tolist(),
                strain1=(audit.prior.SOURCE_ANGULAR_POINTS@e1).tolist())]))))


def chart(p=None, *, normal=(1., 0., 0.), offset=0.):
    return audit.cohort_chart(parent() if p is None else p, 5, normal, offset)


def finite(p=None, *, normal=(1., 0., 0.), center=(0., .5, .75), radius=.1, order=12):
    p = parent() if p is None else p
    return audit.finite_quadrature(p, normal, np.asarray(normal)@center, center, radius,
        cohort_id=5, order=order)


def test_no_producer_imports_dynamic_loading_or_public_replay():
    tree = ast.parse(Path(audit.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(not name.name.startswith("tectonics") for name in node.names)
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("tectonics")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in {"__import__", "eval", "exec", "load_volume_slip_checkpoint"}


def test_embedded_full_disk_area_and_sine_mapping_are_independent_integrals():
    s = finite()
    assert s["current_area_weights_m2"].sum() == pytest.approx(np.pi*.1**2, rel=2e-10)
    np.testing.assert_array_equal(s["current_area_weights_m2"], s["reference_area_weights_m2"])
    assert s["patch_kinds"] == ["sphere_clipped_lambda_segment"]


def test_seed_smaller_than_parent_angular_and_radial_nodes_is_not_missed():
    s = finite(radius=1e-5, order=12)
    assert s["current_area_weights_m2"].sum() == pytest.approx(np.pi*1e-10, rel=2e-6)
    assert len(s["cohort_ids"]) == 144


def test_empty_sphere_relative_interior_is_refused():
    with pytest.raises(ValueError, match="no positive relative-interior support"):
        finite(center=(0., 30., .75))


def test_radius_zero_and_nan_are_refused():
    for radius in (0., -1., float("nan")):
        with pytest.raises(ValueError, match="radius must be positive"):
            finite(radius=radius)


def test_positive_surface_nodes_lie_in_sphere_and_original_material():
    s = finite(radius=.35)
    assert np.all(np.linalg.norm(s["current_position_m"]-[0., .5, .75], axis=1) < .35)
    assert np.all(s["barycentric"] > 0)
    assert np.all(s["radial_scale"] > .5) and np.all(s["radial_scale"] < 1.)
    assert np.all(s["reference_area_weights_m2"] > 0)


def test_cohort_radial_boundary_clips_disk_without_adding_material():
    s = finite(center=(0., .5, .5), radius=.1)
    assert s["current_area_weights_m2"].sum() == pytest.approx(np.pi*.1**2/2, rel=2e-10)
    assert s["selected_cohort_ids"] == [5]


def test_horizontal_disk_uses_polar_chart_and_exact_area():
    s = finite(normal=(0., 0., 1.), center=(0., .5, .75), radius=.1)
    assert all(kind == "sphere_clipped_constant_lambda_triangle" for kind in s["patch_kinds"])
    assert s["current_area_weights_m2"].sum() == pytest.approx(np.pi*.1**2, rel=2e-14)


def test_plane_scaling_preserves_geometry_but_literal_context_stays_available():
    first = finite(normal=(1., 0., 0.))
    second = finite(normal=(2., 0., 0.))
    np.testing.assert_array_equal(first["current_position_m"], second["current_position_m"])
    np.testing.assert_array_equal(first["parent_trace"], second["parent_trace"])


def test_surface_reference_position_and_full_P_reproduce_accepted_pose():
    p = parent(pose=True); s = finite(p)
    u = np.asarray(p["runtime_state"]["displacement_m"])
    positions = s["reference_position_m"]+np.einsum("pai,i->pa", s["parent_trace"], u)
    np.testing.assert_allclose(positions, s["current_position_m"], rtol=2e-15, atol=3e-16)
    assert np.linalg.norm(s["parent_trace"][:, :, 3:6]) > .01
    assert np.max(abs(s["current_to_reference_volume_ratio"]-1)) > .01


def test_compact_weight_vanishes_on_boundaries_with_finite_one_sided_gradient():
    c = chart(); N = np.array([[0., .4, .6], [.4, 0., .6], [.4, .6, 0.], [.3, .3, .4], [.3, .3, .4]])
    lam = np.array([.75, .75, .75, .5, 1.])
    fields = audit.compact_fields(c, N, lam, [0., .5, .75], 10.)
    np.testing.assert_array_equal(fields["support_weight"], np.zeros(5))
    # Material-boundary w is zero; its normal derivative can be nonzero.
    assert np.isfinite(fields["support_gradient_reference_m_inv"]).all()


def test_sphere_rim_squared_weight_has_zero_gradient():
    c = chart(); N = np.array([[.3, .3, .4]]); lam = np.array([.75])
    position = N[0]@(lam[0]*c["Y"]+c["Z"])
    fields = audit.compact_fields(c, N, lam, position+[.1, 0., 0.], .1)
    np.testing.assert_allclose(fields["support_weight"], 0., atol=1e-29)
    np.testing.assert_allclose(fields["support_gradient_reference_m_inv"], 0., atol=1e-12)


def test_compact_reference_gradient_includes_birth_deformation_chain_rule():
    c = chart(parent(pose=True)); N = np.array([[.3, .3, .4]]); lam = np.array([.73])
    position = N[0]@(lam[0]*c["Y"]+c["Z"])
    center = position+[.03, -.02, .01]
    fields = audit.compact_fields(c, N, lam, center, .3)
    grad_parameter = []
    step = 2e-6
    for i in range(3):
        leftN, rightN, leftL, rightL = N.copy(), N.copy(), lam.copy(), lam.copy()
        if i < 2:
            leftN[0, 0] += step; leftN[0, i+1] -= step
            rightN[0, 0] -= step; rightN[0, i+1] += step
        else:
            leftL -= step; rightL += step
        left = audit.compact_fields(c, leftN, leftL, center, .3)["support_weight"][0]
        right = audit.compact_fields(c, rightN, rightL, center, .3)["support_weight"][0]
        grad_parameter.append((right-left)/(2*step))
    expected = np.linalg.solve(fields["reference_volume_jacobian_m"][0].T, grad_parameter)
    np.testing.assert_allclose(fields["support_gradient_reference_m_inv"][0], expected, rtol=2e-9, atol=1e-9)


def test_full_enriched_displacement_gradient_matches_material_finite_difference():
    c = chart(parent(pose=True), offset=-.5); N = np.array([[.3, .3, .4]]); lam = np.array([.73])
    center = N[0]@(lam[0]*c["Y"]+c["Z"])+[.03, -.02, .01]
    fields = audit.compact_fields(c, N, lam, center, .3)
    rng = np.random.default_rng(9); a = rng.normal(size=18)
    parameter_gradient = np.zeros((3, 3)); step = 1e-6
    for i in range(3):
        leftN, rightN, leftL, rightL = N.copy(), N.copy(), lam.copy(), lam.copy()
        if i < 2:
            leftN[0, 0] += step; leftN[0, i+1] -= step
            rightN[0, 0] -= step; rightN[0, i+1] += step
        else:
            leftL -= step; rightL += step
        left = audit.compact_fields(c, leftN, leftL, center, .3)["enrichment_trace"][0]@a
        right = audit.compact_fields(c, rightN, rightL, center, .3)["enrichment_trace"][0]@a
        parameter_gradient[:, i] = (right-left)/(2*step)
    expected = parameter_gradient@np.linalg.inv(fields["reference_volume_jacobian_m"][0])
    actual = np.einsum("i,ijk->jk", a, fields["enrichment_displacement_gradient"][0])
    np.testing.assert_allclose(actual, expected, rtol=5e-9, atol=3e-9)


def test_H_negative_bulk_is_minus_half_with_surface_jump_wT():
    c = chart(offset=.5); N = np.array([[.3, .3, .4]]); lam = np.array([.73])
    center = N[0]@(lam[0]*c["Y"]+c["Z"])
    fields = audit.compact_fields(c, N, lam, center, .3)
    assert fields["heaviside"][0] == -.5
    np.testing.assert_array_equal(fields["enrichment_trace"], -.5*fields["surface_enrichment_trace"])
    assert np.linalg.norm(fields["surface_enrichment_trace"]) > 0


def test_bulk_enrichment_does_not_add_delta_H_singular_term():
    c = chart(); N = np.array([[.3, .3, .4]]); lam = np.array([.73])
    center = N[0]@(lam[0]*c["Y"]+c["Z"])
    fields = audit.compact_fields(c, N, lam, center, .3)
    assert np.isfinite(fields["enrichment_displacement_gradient"]).all()
    assert fields["enrichment_displacement_gradient"].shape == (1, 18, 3, 3)


def test_stress_uses_saved_six_component_memory_and_reference_normal():
    p = parent(pose=True); s = finite(p)
    fields = audit.prior.prestress_fields(p, s)
    expected = np.einsum("pij,pj->pi", fields["samples"]["stress_world_tensor"], s["reference_normal_xyz"])
    np.testing.assert_array_equal(fields["reference_traction_world_pa"], expected)
    wrong = np.einsum("pij,pj->pi", fields["samples"]["stress_world_tensor"], s["current_normal_xyz"])
    assert np.linalg.norm(expected-wrong) > .001


def test_ball_local_bulk_rule_recovers_positive_volume_on_both_sides():
    points = audit.compact_ball_quadrature(chart(), [0., .5, .75], .1, order=4)
    assert len(points["barycentric"]) == 4*4**3
    assert set(points["side"]) == {-.5, .5}
    assert points["birth_volume_weights_m3"].sum() == pytest.approx(4*np.pi*.1**3/3, rel=3e-15)
    np.testing.assert_allclose(points["reference_volume_weights_m3"], points["birth_volume_weights_m3"], rtol=2e-15)


def test_new_compact_inertia_and_elasticity_are_nonzero_on_tiny_seed():
    p = parent(); c = chart(); center = [0., .5, .75]
    bulk = audit.compact_bulk_terms(p, [1., 0., 0.], 0., center, 1e-5, cohort_id=5, order=4)
    assert np.trace(bulk["mass_enriched"]) > 0 and np.trace(bulk["stiffness_enriched"]) > 0
    assert len(bulk["barycentric"]) == 256


def test_compact_bulk_energy_derivative_and_mass_reference_measure():
    p = parent(pose=True); bulk = audit.compact_bulk_terms(p, [1., 0., 0.], 0., [0., .5, .75], .1, cohort_id=5, order=4)
    rng = np.random.default_rng(30); q, a, perturbation = rng.normal(size=(3, 18))*.003
    def energy(amplitude):
        return float(amplitude@bulk["enrichment_force_n"]+amplitude@bulk["stiffness_cross"].T@q+
            .5*amplitude@bulk["stiffness_enriched"]@amplitude)
    force = bulk["enrichment_force_n"]+bulk["stiffness_cross"].T@q+bulk["stiffness_enriched"]@a
    step = 1e-4
    numerical = (energy(a+step*perturbation)-energy(a-step*perturbation))/(2*step)
    assert numerical == pytest.approx(force@perturbation, rel=3e-12)
    velocity = np.einsum("pai,i->pa", bulk["enrichment_trace"], a)
    direct = p["runtime_state"]["geometry"]["density_kg_m3"]*np.einsum("p,pa,pa->", bulk["reference_volume_weights_m3"], velocity, velocity)
    assert a@bulk["mass_enriched"]@a == pytest.approx(direct, rel=5e-15)
    assert np.max(abs(bulk["reference_volume_weights_m3"]/bulk["birth_volume_weights_m3"]-1)) > .001


def test_weak_signed_mass_coupling_survives_planetary_world_translation():
    p = parent(); translated = copy.deepcopy(p)
    shift = np.array([2.**22, -2.**23, 3.*2.**22])
    pose = np.zeros((3, 6)); pose[:, :3] = shift
    translated["runtime_state"]["displacement_m"] = pose.ravel().tolist()
    center, radius = np.array([0., .5, .75]), .002
    reference = audit.compact_bulk_terms(p, [1., 0., 0.], 0., center, radius, cohort_id=5, order=8)
    moved = audit.compact_bulk_terms(translated, [1., 0., 0.], shift[0],
        center+shift, radius, cohort_id=5, order=8)
    A = np.array([0, 1, 2, 6, 7, 8, 12, 13, 14])
    weak = np.linalg.norm(reference["mass_cross"][np.ix_(A, A)])
    support_mass = np.linalg.norm(reference["mass_enriched"][np.ix_(A, A)])
    assert 0 < weak < .01*support_mass
    # Normalise each genuine A/P block by itself: no large parent/AA floor.
    audit.matrix_mode_errors(moved["mass_cross"], reference["mass_cross"],
        "translated weak signed mass", rtol=2e-10, atol=0.)
    audit.matrix_mode_errors(moved["mass_enriched"], reference["mass_enriched"],
        "translated compact mass", rtol=2e-12, atol=0.)
    np.testing.assert_allclose(moved["barycentric"], reference["barycentric"], rtol=0, atol=2e-15)
    np.testing.assert_array_equal(moved["memory0"], reference["memory0"])
    np.testing.assert_array_equal(moved["memory1"], reference["memory1"])
    assert all(v.dtype == np.dtype(float) for v in moved.values() if isinstance(v, np.ndarray))


def test_genuine_genesis_cohort_weak_mass_converges_without_cancellation_floor():
    path = audit.ROOT/"analysis/genesis_volume_creation_validation/evidence/actual_continuation/direct_after_second.json"
    payload = path.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == audit.PARENT_RAW_SHA256
    p = json.loads(payload); geometry = p["runtime_state"]["geometry"]
    cohort = min((c for c in geometry["cohorts"] if c["face_index"] == 0), key=lambda c: c["id"])
    vertices = np.asarray(geometry["faces"][0], int)
    X = np.asarray(geometry["vertices_m"])[vertices]
    pose = np.asarray(p["runtime_state"]["displacement_m"]).reshape(-1, 6)[vertices]
    normal = json.loads((audit.ROOT/"analysis/genesis_volume_location_validation/evidence/actual_control/supplied_current_plane_face_0.json").read_bytes())["normal_xyz"]
    N, lam = np.full(3, 1/3.), .5*(cohort["outer_scale"]+cohort["inner_scale"])
    center = N@(lam*(X+pose[:, 3:])+(pose[:, :3]-pose[:, 3:]))
    radius = .12*(cohort["outer_scale"]-cohort["inner_scale"])*np.linalg.norm(N@(X+pose[:, 3:]))
    offset = float(np.asarray(normal)@center)
    records = [audit.compact_bulk_terms(p, normal, offset, center, radius,
        cohort_id=cohort["id"], order=order) for order in (8, 12)]
    for key in ("stiffness_cross", "stiffness_enriched", "mass_cross", "mass_enriched"):
        audit.matrix_mode_errors(records[0][key], records[1][key],
            "genuine Genesis order convergence "+key, rtol=2e-6, atol=0.)
    A = np.array([0, 1, 2, 6, 7, 8, 12, 13, 14])
    weak = np.linalg.norm(records[1]["mass_cross"][np.ix_(A, A)])
    support_mass = np.linalg.norm(records[1]["mass_enriched"][np.ix_(A, A)])
    assert 0 < weak < 1e-5*support_mass
    assert json.loads(payload) == p and path.read_bytes() == payload


def csr(matrix):
    a = np.asarray(matrix); data, indices, indptr = [], [], [0]
    for row in a:
        selected = np.flatnonzero(row != 0)
        data.extend(row[selected].tolist()); indices.extend(selected.tolist()); indptr.append(len(data))
    return dict(shape=list(a.shape), data=data, indices=indices, indptr=indptr)


def sealed(body, key):
    result = copy.deepcopy(body); result[key] = audit.digest({k: v for k, v in result.items() if k != key})
    return result


@pytest.fixture(scope="module")
def compact_packet():
    p = parent(pose=True); normal = [1., 0., 0.]; center = [0., .5, .75]; radius = .1
    definition = dict(original_cohort_id=5, normal_xyz=normal, offset_m=0., seed_center_world_m=center, seed_radius_m=radius)
    c = audit.cohort_chart(p, 5, normal, 0.)
    local = audit.compact_bulk_terms(p, normal, 0., center, radius, cohort_id=5, order=4)
    original = audit.cohort_original_terms(p, 5); n = 18
    terms = dict(parent_ndof=n, stiffness=original["stiffness"], mass=original["mass"], basal_drag=np.zeros((n, n)),
        external_force_n=np.zeros(n), saved_endpoint_elastic_force_n=original["force_n"],
        saved_endpoint_elastic_energy_j=original["energy_j"], cohort_terms={5: original},
        archived_parent_matrices=dict(stiffness=original["stiffness"], mass=original["mass"], basal_drag=np.zeros((n, n))),
        archived_parent_force_n=original["force_n"], archived_initial_energy_j=original["energy_j"])
    c0 = c["cohort"]
    section = dict(cohort_index=0, cohort_id=5, face_index=0, owner_id=c0["owner_id"], material_id=c0["material_id"],
        q_bounds=[c0["q_lo"], c0["live_q_hi"]], lambda_bounds=[c0["inner_scale"], c0["outer_scale"]], interior_cut=True,
        triangle_m=c["X"], displacement_A_m=c["A"], slope_P_m=c["P"], normal_xyz=c["n"], offset_m=c["d"],
        projection_slopes=c["y"], projection_offsets=c["z"], conical_mass_coefficient=c["coefficient"],
        mass_kg=original["mass_kg"], volume_m3=original["volume_m3"], binding_sha256="0"*64)
    E = local["enrichment_trace"]
    stacked = (np.sqrt(local["reference_volume_weights_m3"])[:, None, None]*E).reshape(-1, 18)
    normalized = stacked/np.linalg.norm(stacked, axis=0)
    A = np.array([0, 1, 2, 6, 7, 8, 12, 13, 14]); P = A+3
    singular = np.linalg.svd(normalized, compute_uv=False)
    K = np.block([[terms["stiffness"], local["stiffness_cross"]], [local["stiffness_cross"].T, local["stiffness_enriched"]]])
    M = np.block([[terms["mass"], local["mass_cross"]], [local["mass_cross"].T, local["mass_enriched"]]])
    diag = np.sqrt(np.diag(K))
    metadata = dict(section=section, numerical_enrichment_trace_rank=int(np.linalg.matrix_rank(normalized)),
        numerical_enrichment_A_trace_rank=int(np.linalg.matrix_rank(normalized[:, A])),
        numerical_enrichment_P_trace_rank=int(np.linalg.matrix_rank(normalized[:, P])),
        normalized_enrichment_trace_condition_number=float(singular[0]/singular[-1]),
        center_plane_residual_m=0., scaled_augmented_stiffness_minimum_eigenvalue=float(np.linalg.eigvalsh(K/diag[:, None]/diag[None, :]).min()))
    raw = dict(format="genesis-tied-finite-seed-compact-bulk-enrichment-0.1",
        scope="prospective_compact_virtual_enrichment_original_memory_no_release_or_funding",
        source_checkpoint_sha256=p["source_checkpoint_sha256"], source_identity=p["runtime_state"]["geometry"]["source_identity"],
        time_myr=.1, parent_ndof=n, ndof=n+18, selected_cohort_id=5, selected_cohort_index=0,
        selected_parent_dofs=list(range(18)), memory0=[original["memory0"]], memory1=[original["memory1"]],
        bubble_definition="H_plus_minus_half_times_sphere_squared_times_27_N0_N1_N2_times_4_t_1_minus_t",
        bulk_gradient="regular_one_sided_reference_gradient_distributional_interface_delta_excluded",
        quadrature_domain="current_seed_ball_two_grid_hemispheres_unique_frozen_birth_chart_inverse",
        material_side_convention="sign_of_literal_section_normal_dot_birth_position_minus_section_offset",
        sphere_center_plane_residual_semantics="represented_plane_residual_no_physical_threshold_or_plane_translation",
        normalized_rank_semantics="formal_18_columns_retained_resolved_rank_reported_no_inversion_or_release_solver",
        quadrature_accuracy="approximate_cohort_clipping_requires_order_convergence_no_exact_integral_claim",
        reaction_semantics="negative_static_endpoint_bulk_elastic_enrichment_gradient_not_accepted_dynamic_birth_reaction",
        added_amplitudes="18_original_A_xyz_P_xyz_modes_per_selected_cohort_all_constrained_zero",
        enriched_memory_representable_by_old_affine_coefficients=False, physical_creation_admitted=False,
        physical_release_admitted=False, funding_admitted=False, released_dof_count=0, new_material_birth_count=0,
        creation_work_debited_from_Genesis_j=0., parent_response_binding_sha256="0"*64,
        local={**local, **metadata}, stiffness=csr(K), mass_matrix=csr(M), basal_drag_matrix=csr(np.zeros((36, 36))),
        external_force_n=np.zeros(36), initial_force_n=np.r_[terms["saved_endpoint_elastic_force_n"], local["enrichment_force_n"]],
        initial_elastic_energy_j=terms["saved_endpoint_elastic_energy_j"],
        prolongation=csr(np.vstack((np.eye(18), np.zeros((18, 18))))), tie_operator=csr(np.hstack((np.zeros((18, 18)), np.eye(18)))))
    return p, terms, definition, sealed(audit.plain(raw), "embedded_partition_sha256")


def test_manufactured_full_constrained_packet_positive_audit(compact_packet):
    p, terms, definition, raw = compact_packet
    _, report = audit.audit_partition(raw, p, terms, definition, order=4)
    assert report["reference_gradient_and_full_original_A_P_basis_checked"] is True


@pytest.mark.parametrize("mutation,reason", [
    ("bulk_gradient", "enrichment_strain_operator differs"),
    ("added_mass_P", "mass_matrix enriched PP differs"),
    ("R_epsilon", "exact constrained compact prolongation differs"),
    ("rank_metadata", "numerical rank metadata"),
    ("conditioning_metadata", "conditioning metadata differs"),
    ("planetary_position", "birth_position_m differs"),
    ("invented_drag", "invented enriched drag"),
])
def test_resealed_compact_packet_mathematical_forgeries_are_refused(compact_packet, mutation, reason):
    p, terms, definition, original = compact_packet; raw = copy.deepcopy(original)
    if mutation == "bulk_gradient": raw["local"]["enrichment_strain_operator"][0][0][0] += .01
    if mutation == "added_mass_P":
        M = audit.dense_csr(raw["mass_matrix"], "test mass"); P = 18+np.array([3, 4, 5, 9, 10, 11, 15, 16, 17])
        M[np.ix_(P, P)] = 0.; raw["mass_matrix"] = csr(M)
    if mutation == "R_epsilon":
        R = audit.dense_csr(raw["prolongation"], "test R"); R[0, 0] += 1e-8; raw["prolongation"] = csr(R)
    if mutation == "rank_metadata": raw["local"]["numerical_enrichment_trace_rank"] = 3
    if mutation == "conditioning_metadata": raw["local"]["normalized_enrichment_trace_condition_number"] = 1.
    if mutation == "planetary_position": raw["local"]["birth_position_m"][0][0] += .01
    if mutation == "invented_drag":
        D = np.zeros((36, 36)); D[35, 35] = 1e-10; raw["basal_drag_matrix"] = csr(D)
    raw = sealed(raw, "embedded_partition_sha256")
    with pytest.raises(ValueError, match=reason): audit.audit_partition(raw, p, terms, definition, order=4)


def test_tiny_PP_matrix_cannot_be_erased_behind_huge_AA_scale():
    expected = np.zeros((6, 6)); expected[:3, :3] = np.eye(3)*1e18; expected[3:, 3:] = np.eye(3)*1e-14
    forged = expected.copy(); forged[3:, 3:] = 0.
    with pytest.raises(ValueError, match="PP differs"):
        audit.matrix_mode_errors(forged, expected, "physical mass", atol=1e-20)


def test_material_plane_has_jump_null_modes_that_are_not_bulk_null_modes():
    p = parent(pose=True); c = chart(p); s = finite(p)
    amplitude = np.zeros((3, 6))
    amplitude[:, 0] = (c["X"]+c["A"])@c["n"]-c["d"]
    amplitude[:, 3] = (c["X"]+c["P"])@c["n"]
    surface_fields = audit.compact_fields(c, s["barycentric"], s["radial_scale"], [0., .5, .75], .1)
    jump = np.einsum("pai,i->pa", surface_fields["surface_enrichment_trace"], amplitude.ravel())
    np.testing.assert_allclose(jump, 0., atol=2e-16)
    bulk = audit.compact_bulk_terms(p, [1., 0., 0.], 0., [0., .5, .75], .1, cohort_id=5, order=4)
    assert amplitude.ravel()@bulk["stiffness_enriched"]@amplitude.ravel() > 1e-8


def test_reference_surface_force_does_not_replace_discrete_bulk_gradient():
    p = parent(pose=True); s = finite(p, order=8)
    c = chart(p)
    surface_fields = audit.compact_fields(c, s["barycentric"], s["radial_scale"], [0., .5, .75], .1)
    prestress = audit.prior.prestress_fields(p, s)
    surface_force = np.einsum("pai,pa,p->i", surface_fields["surface_enrichment_trace"],
        prestress["reference_traction_world_pa"], s["reference_area_weights_m2"])
    bulk = audit.compact_bulk_terms(p, [1., 0., 0.], 0., [0., .5, .75], .1, cohort_id=5, order=8)
    assert np.linalg.norm(-bulk["enrichment_force_n"]-surface_force)/np.linalg.norm(surface_force) > 1e-3


