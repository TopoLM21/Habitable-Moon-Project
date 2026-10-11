"""Independent mathematical checks; no bank producer is imported here."""
from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from analysis.genesis_volume_bank_validation import independent_audit as audit


def tetrahedron():
    vertices = np.asarray(((1., 1., 1.), (1., -1., -1.),
                           (-1., 1., -1.), (-1., -1., 1.)))
    faces = np.asarray(((0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2)))
    return vertices, faces


def material_case():
    X = np.asarray(((0., 0., 3.), (2., 0., 3.), (0., 2., 3.)))
    frame = np.eye(3)
    cohort = {"id": 1, "face_index": 0, "q_lo": .1, "live_q_hi": .8}
    C = audit.isotropic_matrix(17., .23, .19, .07)
    return X, frame, .8, 2., cohort, C


def test_graph_split_closed_face_cycle_has_known_components():
    X, faces = tetrahedron()
    topo = audit.reference_topology(X, faces, ((0, 1), (1, 2), (2, 0)))
    assert topo["parent_vertex"].tolist() == [0, 1, 2, 3, 0, 1, 2]
    assert topo["faces"].tolist() == [[0, 1, 2], [4, 3, 5], [4, 6, 3], [5, 3, 6]]
    assert np.array_equal(topo["parent_vertex"][topo["faces"]], faces)
    R, J = topo["prolongation"], topo["jump_operator"]
    assert R.shape == (42, 24) and J.shape == (18, 42)
    assert np.array_equal(J@R, np.zeros((18, 24)))
    assert np.linalg.matrix_rank(J) == 18
    # Both Cartesian top motion and depth slope have independent jumps.
    for component in range(6):
        displacement = np.zeros(42); displacement[6*4+component] = component+1.
        expected = np.zeros(18); expected[component] = component+1.
        assert np.array_equal(J@displacement, expected)


def test_isolated_edge_retains_both_tip_vertices():
    X, faces = tetrahedron()
    topo = audit.reference_topology(X, faces, ((0, 1),))
    assert np.array_equal(topo["parent_vertex"], np.arange(4))
    assert np.array_equal(topo["faces"], faces)
    assert topo["jump_operator"].shape == (0, 24)
    assert topo["bank_vertices"].tolist() == [[[0, 1], [0, 1]]]


def test_all_edge_cuts_separate_face_corners_and_keep_material_order():
    X, faces = tetrahedron()
    cuts = sorted({tuple(sorted((int(row[a]), int(row[b]))))
                   for row in faces for a, b in ((0, 1), (1, 2), (2, 0))})
    topo = audit.reference_topology(X, faces, cuts)
    assert len(topo["parent_vertex"]) == 12
    assert len(set(topo["faces"].ravel())) == 12
    assert np.array_equal(topo["parent_vertex"][topo["faces"]], faces)
    assert np.array_equal(topo["jump_operator"]@topo["prolongation"], np.zeros((48, 24)))


@pytest.mark.parametrize("mutation", ("duplicate_cut", "nonedge", "float_cut", "reversed_face", "open"))
def test_independent_topology_refuses_invalid_reference(mutation):
    X, faces = tetrahedron(); cuts = [(0, 1)]
    if mutation == "duplicate_cut": cuts = [(0, 1), (1, 0)]
    elif mutation == "nonedge": cuts = [(0, 9)]
    elif mutation == "float_cut": cuts = [(0., 1.)]
    elif mutation == "reversed_face": faces = faces.copy(); faces[1] = faces[1, ::-1]
    elif mutation == "open": faces = faces[:3]
    with pytest.raises(ValueError): audit.reference_topology(X, faces, cuts)


def test_inverse_chart_strain_matches_cartesian_finite_difference():
    X, frame, mass, rho, cohort, C = material_case()
    rng = np.random.default_rng(931)
    motion = rng.normal(size=(3, 6))*.02
    N = np.asarray(((.2, .3, .5), (.5, .4, .1)))
    B0, B1 = audit.independent_strain_operators(X, frame, N)
    inverse = np.linalg.inv(X.T)
    for point, lam in enumerate((.83, .97)):
        position = lam*(N[point]@X)
        def displacement(x):
            cone = inverse@x
            radial = cone.sum()
            barycentric = cone/radial
            return barycentric@(motion[:, :3]+(radial-1)*motion[:, 3:])
        h = 1e-5
        gradient = np.column_stack([(displacement(position+h*axis)-displacement(position-h*axis))/(2*h)
                                    for axis in np.eye(3)])
        direct = np.asarray((gradient[0, 0], gradient[1, 1], gradient[2, 2],
                             gradient[0, 1]+gradient[1, 0], gradient[0, 2]+gradient[2, 0],
                             gradient[1, 2]+gradient[2, 1]))
        computed = (B0[point]+B1[point]/lam)@motion.ravel()
        np.testing.assert_allclose(computed, direct, rtol=2e-9, atol=5e-12)


def test_affine_world_motion_gives_uniform_six_strain_and_rigid_null_modes():
    X, frame, mass, rho, cohort, C = material_case()
    A = np.asarray(((.01, .007, -.002), (-.003, -.006, .008), (.005, -.002, .004)))
    u = X@A.T
    displacement = np.column_stack((u, u)).ravel()
    zero = np.zeros((3, 6))
    result = audit.cohort_quadrature(X, frame, mass, rho, cohort, zero, zero, C, displacement)
    engineering = np.asarray((A[0, 0], A[1, 1], A[2, 2], A[0, 1]+A[1, 0],
                              A[0, 2]+A[2, 0], A[1, 2]+A[2, 1]))
    expected = .5*result["volume_m3"]*engineering@C@engineering
    assert result["elastic_energy_j"] == pytest.approx(expected, rel=2e-14)
    np.testing.assert_allclose(result["mean_elastic_strain"], engineering, rtol=2e-14, atol=1e-15)
    rigid = np.column_stack((np.repeat([[.3, -.2, .4]], 3, axis=0), np.zeros((3, 3)))).ravel()
    assert abs(rigid@result["stiffness"]@rigid) < 1e-15
    omega = np.asarray((.3, -.1, .2))
    rotation = np.cross(np.repeat(omega[None], 3, axis=0), X)
    rotation = np.column_stack((rotation, rotation)).ravel()
    assert abs(rotation@result["stiffness"]@rotation) < 3e-14


def test_angular_and_radial_memory_variance_matches_separate_exact_integrals():
    X, frame, mass, rho, cohort, C = material_case()
    # Vertex affine angular values are independent data; producer-point
    # samples are their barycentric evaluation.
    vertex0 = np.asarray(((.2, -.1, .07, .09, .06, -.03),
                          (.03, .11, -.02, -.06, .04, .08),
                          (-.04, .02, .13, .04, -.05, .01)))
    vertex1 = np.asarray(((.03, .07, -.01, .02, -.03, .08),
                          (-.08, .01, .03, .06, .02, -.04),
                          (.05, -.09, .01, -.03, .07, .02)))
    old0, old1 = audit.SOURCE_ANGULAR_POINTS@vertex0, audit.SOURCE_ANGULAR_POINTS@vertex1
    result = audit.cohort_quadrature(X, frame, mass, rho, cohort, old0, old1, C)
    coefficient = 6*mass/(rho*np.linalg.det(X.T))
    outer = (1-coefficient*cohort["q_lo"])**(1/3)
    inner = (1-coefficient*cohort["live_q_hi"])**(1/3)
    radial1 = 1.5*(outer**2-inner**2)/(outer**3-inner**3)
    radial2 = 3*(outer-inner)/(outer**3-inner**3)
    gram = (np.ones((3, 3))+np.eye(3))/12
    inner_product = lambda a, b: np.einsum("ij,ia,ab,jb->", gram, a, C, b)
    expected = .5*result["volume_m3"]*(inner_product(vertex0, vertex0)
        +2*radial1*inner_product(vertex0, vertex1)+radial2*inner_product(vertex1, vertex1))
    assert result["elastic_energy_j"] == pytest.approx(expected, rel=2e-14)
    flattened = audit.cohort_quadrature(X, frame, mass, rho, cohort,
        np.repeat(old0.mean(axis=0)[None], 3, axis=0),
        np.repeat(old1.mean(axis=0)[None], 3, axis=0), C)
    assert abs(result["elastic_energy_j"]-flattened["elastic_energy_j"]) > .005*expected


def test_endpoint_prestress_and_new_work_do_not_add_absolute_saved_motion():
    X, frame, mass, rho, cohort, C = material_case()
    old0 = np.ones((3, 6))*.001
    old1 = np.asarray(((.001, .002, .003, .004, .005, .006),)*3)
    initial = audit.cohort_quadrature(X, frame, mass, rho, cohort, old0, old1, C)
    rng = np.random.default_rng(27); du = rng.normal(size=18)*.01
    endpoint = audit.cohort_quadrature(X, frame, mass, rho, cohort, old0, old1, C, du)
    work = initial["force_n"]@du+.5*du@initial["stiffness"]@du
    assert endpoint["elastic_energy_j"]-initial["elastic_energy_j"] == pytest.approx(work, rel=2e-14)
    assert not np.allclose(initial["force_n"], endpoint["force_n"])
    np.testing.assert_array_equal(initial["constant_strain"], old0)
    np.testing.assert_array_equal(initial["inverse_scale_strain"], old1)


def test_cartesian_frame_rotation_preserves_all_six_dof_force_and_memory():
    X, frame, mass, rho, cohort, C = material_case()
    rng = np.random.default_rng(832)
    old0, old1 = rng.normal(size=(3, 6))*.003, rng.normal(size=(3, 6))*.002
    du = rng.normal(size=(3, 6))*.01
    angle = .83
    O = np.asarray(((np.cos(angle), -np.sin(angle), 0.),
                    (np.sin(angle), np.cos(angle), 0.), (0., 0., 1.)))
    original = audit.cohort_quadrature(X, frame, mass, rho, cohort, old0, old1, C, du.ravel())
    rotated_du = np.column_stack((du[:, :3]@O.T, du[:, 3:]@O.T))
    rotated = audit.cohort_quadrature(X@O.T, O@frame, mass, rho, cohort,
                                    old0, old1, C, rotated_du.ravel())
    assert original["elastic_energy_j"] == pytest.approx(rotated["elastic_energy_j"], rel=3e-15)
    force = original["force_n"].reshape(3, 6)
    expected = np.column_stack((force[:, :3]@O.T, force[:, 3:]@O.T))
    np.testing.assert_allclose(rotated["force_n"], expected.ravel(), rtol=8e-14, atol=1e-16)
    np.testing.assert_allclose(rotated["constant_strain"], original["constant_strain"], rtol=3e-14, atol=1e-16)
    np.testing.assert_allclose(rotated["inverse_scale_strain"], original["inverse_scale_strain"], rtol=3e-14, atol=1e-16)


def test_each_cohort_identity_survives_operator_assembly():
    X, frame, mass, rho, cohort, C = material_case()
    faces = np.asarray(((0, 1, 2),))
    memory = [{"id": 2, "strain0": np.zeros((3, 6)), "strain1": np.zeros((3, 6))}]
    with pytest.raises(ValueError, match="cohort identity"):
        audit.integrated_operator(X, faces, faces, [mass], rho, [cohort], memory, [frame], [C])


@pytest.mark.parametrize("small_block", ("mass", "drag"))
def test_thin_depth_block_corruption_is_not_hidden_by_top_cartesian_scale(small_block):
    # A single global relative tolerance would accept this 1% P-P mutation:
    # its absolute error is 1e-16 of the leading A-A block.
    matrix = np.kron(np.asarray(((1., -1e-7), (-1e-7, 1.2e-14))), np.eye(3))
    matrix *= 7e18 if small_block == "mass" else 4e26
    corrupted = matrix.copy(); corrupted[3:, 3:] *= 1.01
    assert audit._close(corrupted, matrix, "weak global check", atol=0) < 1e-11
    with pytest.raises(ValueError, match="P-P block"):
        audit._six_dof_matrix_close(corrupted, matrix, small_block)


def test_split_mass_linear_momentum_is_exact_physical_mass_integral():
    X, faces = tetrahedron()
    topology = audit.reference_topology(X, faces, ((0, 1), (1, 2), (0, 2)))
    masses, rho = np.asarray((.15, .21, .18, .23)), 1.7
    cohorts = [{"id": i+1, "face_index": i, "q_lo": .1, "live_q_hi": .7} for i in range(4)]
    M = audit.mass_matrix(X, faces, masses, rho, cohorts)
    split_X = X[topology["parent_vertex"]]
    MB = audit.mass_matrix(split_X, topology["faces"], masses, rho, cohorts)
    R = topology["prolongation"]
    np.testing.assert_allclose(R.T@MB@R, M, rtol=3e-15, atol=1e-17)
    constant = np.asarray((.17, -.31, .23))
    velocity = np.column_stack((np.repeat(constant[None], len(X), axis=0), np.zeros_like(X))).ravel()
    expected_mass = masses.sum()*.6
    before = audit.momentum(M, velocity, X)
    after = audit.momentum(MB, R@velocity, split_X)
    np.testing.assert_allclose(before["linear_momentum_kg_m_s"], expected_mass*constant, rtol=2e-15)
    assert before["kinetic_energy_j"] == pytest.approx(.5*expected_mass*(constant@constant), rel=2e-15)
    for key in before:
        np.testing.assert_allclose(before[key], after[key], rtol=5e-15, atol=1e-17)


def test_basal_tangent_and_nodal_force_use_all_six_velocity_shapes():
    X, faces = tetrahedron()
    masses, rho = np.asarray((.15, .21, .18, .23)), 1.7
    cohorts = [{"id": i+1, "face_index": i, "q_lo": 0., "live_q_hi": .7} for i in range(4)]
    force = np.asarray(((.3, .1, -.2), (-.2, .4, .5), (.1, -.3, .2), (.4, .2, -.1)))
    D, F = audit.basal_drag_and_force(X, faces, masses, rho, cohorts, 2.3, force)
    constant = np.asarray((.2, -.3, .1))
    velocity = np.column_stack((np.repeat(constant[None], len(X), axis=0), np.zeros_like(X))).ravel()
    expected_work, expected_force_work = 0., 0.
    for face, row in enumerate(faces):
        triangle = X[row]
        normal = np.cross(triangle[1]-triangle[0], triangle[2]-triangle[0])
        top_area = np.linalg.norm(normal)/2; normal /= np.linalg.norm(normal)
        lam = (1-6*masses[face]*.7/(rho*np.linalg.det(triangle.T)))**(1/3)
        area = top_area*lam*lam
        tangent = constant-normal*np.dot(normal, constant)
        expected_work += area*2.3*np.dot(tangent, tangent)
        expected_force_work += area*np.dot(constant, force[face])
    assert velocity@D@velocity == pytest.approx(expected_work, rel=3e-15)
    assert velocity@F == pytest.approx(expected_force_work, rel=3e-15)
    assert np.any(F.reshape(-1, 6)[:, 3:] != 0)
    assert np.any(D[3:6] != 0)


def test_actual_parent_is_literal_preserved_cohort_memory():
    path = Path(__file__).resolve().parents[1]/"analysis/genesis_volume_slip_validation/evidence/actual_control/direct_after_second.json"
    parent = audit._actual_parent(path)
    fields = audit._source_fields(parent)
    assert len(fields[3]) == 640 and len(fields[4]) == 640
    assert any(np.any(row["strain1"]) for row in fields[4])
    assert fields[2]["time_myr"] == pytest.approx(.8682, abs=1e-14)
    fixture = {"format": audit.FORMAT, "scope": audit.SCOPE,
               "parent_raw_sha256": audit.ACTUAL_PARENT_RAW_SHA256,
               "parent_checkpoint": copy.deepcopy(parent)}
    fixture["parent_checkpoint"]["runtime_state"]["mechanics"]["state"]["cohorts"][0]["strain1"][0][0] += 1e-7
    with pytest.raises(ValueError, match="entire actual parent checkpoint"):
        audit.audit_fixture(fixture, parent)


@pytest.mark.parametrize("false_claim", ("physical_release_admitted", "physical_creation_work_assigned",
                                        "endpoint_force_is_release_traction", "history_reinitialized"))
def test_actual_descriptor_cannot_authorize_a_resealed_false_claim(false_claim):
    path = Path(__file__).resolve().parents[1]/"analysis/genesis_volume_slip_validation/evidence/actual_control/direct_after_second.json"
    parent = audit._actual_parent(path)
    # The independently accepted source has no eligible fault and no new
    # transition.  A purported seal on a release claim carries no authority.
    body = {"format": audit.TRANSFER_FORMAT, "scope": audit.TRANSFER_SCOPE,
            false_claim: True, "parent_checkpoint_sha256": parent["checkpoint_sha256"]}
    forged = audit._seal(body, "transfer_sha256")
    with pytest.raises(ValueError, match="source-bound transfer descriptor"):
        audit.audit_transfer_descriptor(forged, parent)


@pytest.mark.parametrize("change", ({"format": "mechanical-test"}, {"scope": "admitted_physical_fault"},
                                   {"parent_raw_sha256": "0"*64}))
def test_audit_rejects_unbound_scope_before_physics(change):
    fixture = {"format": audit.FORMAT, "scope": audit.SCOPE,
               "parent_raw_sha256": audit.ACTUAL_PARENT_RAW_SHA256}
    fixture.update(change)
    with pytest.raises(ValueError): audit.audit_fixture(fixture, {})


@pytest.mark.parametrize("raw", ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}'))
def test_literal_json_refuses_duplicate_and_nonfinite(tmp_path, raw):
    path = tmp_path/"untrusted.json"; path.write_text(raw)
    with pytest.raises(ValueError): audit._literal_json(path)


def test_oracle_and_reused_quadrature_have_no_tectonics_imports():
    sources = (Path(audit.__file__), Path(__file__).resolve().parents[1]/
               "analysis/genesis_volume_continuation_validation/independent_audit.py")
    for source in sources:
        for node in ast.walk(ast.parse(source.read_text())):
            if isinstance(node, ast.Import):
                assert all(not alias.name.startswith("tectonics") for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("tectonics")
