"""Producer-free audit of a tied computational-bank transfer.

Corner connectivity is reconstructed by graph traversal.  Six Cartesian
DOFs, conical strains, inherited cohort memory, mass and basal forces are
integrated independently.  The selected reference edges are a declared
computational fixture: this module does not infer a physical fault surface,
release a tie, or certify contact or material admission.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

# These are producer-free numerical kernels.  In particular the Cartesian
# inverse-chart strain derivation and Duffy/radial Gauss rules do not call the
# producer's local-plane operators or its analytic radial moments.
from analysis.genesis_volume_continuation_validation.independent_audit import (
    SOURCE_ANGULAR_POINTS, angular_rule, basal_drag_and_force, conical_scales,
    independent_strain_operators, isotropic_matrix, mass_matrix,
    prescribed_traction, radial_rule,
)


FORMAT = "genesis-volume-bank-control-0.1"
SCOPE = "tied_computational_bank_transfer_from_verified_actual_genesis_volume_state"
ACTUAL_PARENT_RAW_SHA256 = "34cc53c215b2e9b8300a8f47b33801d49182b611a3ad985217a9c4cef25f2aed"
ACTUAL_PARENT_CHECKPOINT_SHA256 = "029b83562f85fe554de0d799ed64bec9003faa558a95556716be8285e89c7f2e"
TRANSFER_FORMAT = "genesis-volume-bank-transfer-0.1"
TRANSFER_SCOPE = "zero_time_accepted_genesis_memory_on_tied_six_dof_support"
UNRESOLVED_PHYSICAL_INPUTS = [
    "calculated_material_fracture_surface_and_through_depth_geometry",
    "paired_physical_depth_intervals_with_cohort_side_ownership",
    "source_calculated_surface_creation_law_and_creation_work",
    "accepted_interval_traction_and_release_balance",
    "both_tangential_contact_modes_and_spatial_underthrust_admission",
]


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode()


def _seal(value, key="sha256"):
    result = dict(value)
    result[key] = hashlib.sha256(canonical({k: v for k, v in result.items()
                                         if k != key})).hexdigest()
    return result


def _array(value, shape=None, integer=False):
    original = np.asarray(value)
    if integer:
        if original.size and original.dtype.kind not in "iu":
            raise ValueError("Independent bank audit requires literal integer indices")
        result = original.astype(np.int64)
    else:
        if original.size and original.dtype.kind not in "iuf":
            raise ValueError("Independent bank audit refuses logical, complex or textual numerical arrays")
        result = np.asarray(value, float)
    if (shape is not None and result.shape != shape) or not np.isfinite(result).all():
        raise ValueError("Independent bank audit received an invalid finite array")
    return result


def _close(actual, expected, name, rtol=3e-11, atol=1e-11):
    a, b = _array(actual), _array(expected)
    if a.shape != b.shape:
        raise ValueError(f"Independent bank audit: {name} shape differs")
    # A global scale is appropriate for assembled matrices with exact zero
    # blocks and cancellation in individual force components.
    scale = max(float(np.max(np.abs(b), initial=0.)), 1e-300)
    error = float(np.max(np.abs(a-b), initial=0.))
    if error > atol+rtol*scale:
        raise ValueError(f"Independent bank audit: {name} differs")
    return error/max(scale, atol, 1e-300)


def _exact(actual, expected, name):
    if canonical(actual) != canonical(expected):
        raise ValueError(f"Independent bank audit: {name} differs literally")


def _six_dof_matrix_close(actual, expected, name):
    """Scale Cartesian A/P blocks separately, including thin depth inertia."""
    a, b = _array(actual), _array(expected)
    if a.shape != b.shape or b.ndim != 2 or b.shape[0] != b.shape[1] or b.shape[0] % 6:
        raise ValueError(f"Independent bank audit: {name} six-DOF matrix shape differs")
    A, P = np.flatnonzero(np.arange(len(b)) % 6 < 3), np.flatnonzero(np.arange(len(b)) % 6 >= 3)
    errors = []
    for rname, rows in (("A", A), ("P", P)):
        for cname, cols in (("A", A), ("P", P)):
            errors.append(_close(a[np.ix_(rows, cols)], b[np.ix_(rows, cols)],
                                 f"{name} {rname}-{cname} block", atol=0))
    return max(errors)


def _six_dof_force_close(actual, expected, name):
    a, b = _array(actual), _array(expected)
    if a.shape != b.shape or b.ndim != 1 or len(b) % 6:
        raise ValueError(f"Independent bank audit: {name} six-DOF force shape differs")
    errors = [_close(a.reshape(-1, 6)[:, start:start+3], b.reshape(-1, 6)[:, start:start+3],
                     f"{name} {'A' if start == 0 else 'P'} components", atol=1e-7)
              for start in (0, 3)]
    return max(errors)


def _keys(value, expected, name):
    if type(value) is not dict or set(value) != set(expected):
        raise ValueError(f"Independent bank audit: {name} fields differ")


def _array_digest(value):
    """Recreate the published literal field byte binding, without producer code."""
    array = np.asarray(value, float)
    digest = hashlib.sha256()
    digest.update(str((array.shape, array.dtype.str)).encode())
    digest.update(array.tobytes(order="C"))
    digest.update(b"\x00")
    return digest.hexdigest()


def audit_transfer_descriptor(descriptor, parent, *, topology=None):
    """Bind an eligibility descriptor to raw accepted history and derived maps.

    The producer's in-memory support seal is opaque here.  Independently
    derived topology and every operator are checked separately, so this hash
    can identify that object but cannot authorize a physical release.
    """
    state = parent["runtime_state"]
    geometry = state["geometry"]
    memory = state["mechanics"]["state"]["cohorts"]
    ids = [int(c["id"]) for c in geometry["cohorts"]]
    if [int(c["id"]) for c in memory] != ids:
        raise ValueError("Independent descriptor lost ordered full-cohort memory")
    e0, e1 = np.asarray([c["strain0"] for c in memory]), np.asarray([c["strain1"] for c in memory])
    q, v = _array(state["displacement_m"]), _array(state["velocity_m_s"])
    if topology is None:
        support_kind, cuts, bank_pin = "original_connected_domain", [], None
        ndof, parent_vertex = len(q), list(range(len(q)//6))
    else:
        support_kind = "explicit_computational_edge_split"
        cuts = topology["cut_edges"].tolist()
        parent_vertex = topology["parent_vertex"].tolist()
        R = topology["prolongation"]
        q, v = R@q, R@v
        ndof = len(q)
        bank_pin = descriptor.get("support_binding_sha256")
        if (type(bank_pin) is not str or len(bank_pin) != 64
                or any(c not in "0123456789abcdef" for c in bank_pin)):
            raise ValueError("Independent descriptor support identifier is malformed")
    excluded = {"geometry", "thermal", "history", "mechanics", "fault_history",
                "displacement_m", "velocity_m_s", "last_report", "last_certificate",
                "plane_normal_world", "slip_direction_world"}
    digest = lambda x: hashlib.sha256(canonical(x)).hexdigest()
    active = np.flatnonzero(state["fault_history"]["fault_active"]).tolist()
    expected = dict(
        format=TRANSFER_FORMAT, scope=TRANSFER_SCOPE,
        parent_checkpoint_sha256=parent["checkpoint_sha256"],
        formation_checkpoint_sha256=parent["source_checkpoint_sha256"],
        source_identity=geometry["source_identity"], time_myr=geometry["time_myr"],
        accepted_step_sizes_myr=state["accepted_step_sizes_myr"],
        accepted_steps=state["mechanics"]["state"]["accepted_steps"], cohort_ids=ids,
        material_geometry_sha256=digest(state["geometry"]),
        birth_history_sha256=digest(state["history"]),
        full_cohort_memory_sha256=digest(state["mechanics"]),
        fault_history_sha256=digest(state["fault_history"]),
        thermal_history_sha256=digest(state["thermal"]),
        inherited_runtime_ledgers_and_closures={k: val for k, val in state.items() if k not in excluded},
        support_kind=support_kind, supplied_computational_cut_edges=cuts,
        support_binding_sha256=bank_pin, ndof=ndof, parent_vertex=parent_vertex,
        field_sha256={"displacement_m": _array_digest(q), "velocity_m_s": _array_digest(v),
                      "elastic_constant": _array_digest(e0), "elastic_inverse_scale": _array_digest(e1)},
        active_weak_plane_faces=active,
        physical_cut_status="no_eligible_cut" if not active else "unresolved_physical_cut",
        unresolved_physical_inputs=UNRESOLVED_PHYSICAL_INPUTS,
        sharp_physical_banks_created=False, physical_release_admitted=False,
        spatial_underthrust_admitted=False, physical_creation_work_assigned=False,
        endpoint_force_is_release_traction=False,
        transition_time_increment_myr=0., transition_work_j=0., transition_impulse_kg_m_s=0.,
        memory_update_increment_m=0., history_reinitialized=False,
        connected_material_owner_ids_changed=False,
    )
    expected["transfer_sha256"] = digest(expected)
    _exact(descriptor, expected, "independently source-bound transfer descriptor")
    return {"transfer_sha256": expected["transfer_sha256"],
            "physical_cut_status": expected["physical_cut_status"],
            "all_history_and_field_bindings_checked": True,
            "support_binding_sha256_is_opaque_identifier": topology is not None}


def reference_topology(vertices_m, faces, cut_edges):
    """Find corner-star components without the producer's union-find.

    A component may cross a reference edge only when that edge is uncut.
    First corner encounter retains the parent vertex index.  The two
    endpoints of a finite isolated cut can therefore remain shared, exactly
    as dictated by their remaining incident-face paths.
    """
    vertices = _array(vertices_m)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 4:
        raise ValueError("Independent bank reference needs finite 3-D vertices")
    faces = _array(faces, integer=True)
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) < 4:
        raise ValueError("Independent bank reference needs triangular faces")
    if np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise ValueError("Independent bank reference face index is out of range")
    if len({tuple(sorted(row)) for row in faces}) != len(faces):
        raise ValueError("Independent bank reference has duplicate faces")
    owners = {}
    used = set()
    links = [set() for _ in range(3*len(faces))]
    vertex_faces = [[] for _ in vertices]
    for f, row in enumerate(faces):
        if len(set(row)) != 3:
            raise ValueError("Independent bank reference has a collapsed face")
        triangle = vertices[row]
        normal = np.cross(triangle[1]-triangle[0], triangle[2]-triangle[0])
        if np.dot(normal, triangle[0]) <= 0:
            raise ValueError("Independent bank reference has a nonoutward face")
        for corner, v in enumerate(row):
            used.add(int(v)); vertex_faces[v].append(3*f+corner)
        for a, b in ((0, 1), (1, 2), (2, 0)):
            va, vb = int(row[a]), int(row[b])
            edge = tuple(sorted((va, vb)))
            low, high = (a, b) if va < vb else (b, a)
            owners.setdefault(edge, []).append((f, low, high, va, vb))
    if used != set(range(len(vertices))):
        raise ValueError("Independent bank reference has an unused parent vertex")
    for edge, pair in owners.items():
        if len(pair) != 2 or pair[0][3:] != pair[1][3:][::-1]:
            raise ValueError("Independent bank reference is not closed and oriented")
    cuts = np.asarray(cut_edges)
    if cuts.size == 0:
        if cuts.shape not in ((0,), (0, 2)):
            raise ValueError("Independent cut edges have invalid empty shape")
        keys = []
    else:
        cuts = _array(cuts, integer=True)
        if cuts.ndim != 2 or cuts.shape[1] != 2:
            raise ValueError("Independent cut edges must have two endpoints")
        keys = [tuple(sorted(map(int, row))) for row in cuts]
        if len(set(keys)) != len(keys) or any(key not in owners for key in keys):
            raise ValueError("Independent cut edges must be unique reference edges")
        keys.sort()
    cut_set = set(keys)
    complete_links = [set() for _ in links]
    for edge, pair in owners.items():
        (fa, au, av, _, _), (fb, bu, bv, _, _) = pair
        for ca, cb in ((3*fa+au, 3*fb+bu), (3*fa+av, 3*fb+bv)):
            complete_links[ca].add(cb); complete_links[cb].add(ca)
            if edge not in cut_set:
                links[ca].add(cb); links[cb].add(ca)
    # Closed edge incidence alone does not exclude two disconnected stars at
    # a nonmanifold parent vertex.  Refuse such a parent before cutting it.
    for corners in vertex_faces:
        pending, reached = [corners[0]], set()
        while pending:
            c = pending.pop()
            if c not in reached:
                reached.add(c); pending.extend(complete_links[c]-reached)
        if reached != set(corners) or any(len(complete_links[c]) != 2 for c in corners):
            raise ValueError("Independent bank reference has a nonmanifold vertex star")
    component = np.full(3*len(faces), -1, int)
    for seed in range(len(component)):
        if component[seed] >= 0:
            continue
        pending = [seed]
        while pending:
            c = pending.pop()
            if component[c] < 0:
                component[c] = seed; pending.extend(links[c])
    parent = list(range(len(vertices)))
    seen = [{} for _ in vertices]
    split = np.empty_like(faces)
    for c, v in enumerate(faces.ravel()):
        comp = int(component[c])
        if comp not in seen[v]:
            seen[v][comp] = int(v) if not seen[v] else len(parent)
            if seen[v][comp] != v:
                parent.append(int(v))
        split.flat[c] = seen[v][comp]
    parent = np.asarray(parent, int)
    seam_faces, bank_vertices = [], []
    for edge in keys:
        pair = sorted(owners[edge])
        seam_faces.append([p[0] for p in pair])
        bank_vertices.append([[int(split[f, low]), int(split[f, high])]
                              for f, low, high, _, _ in pair])
    R = np.zeros((6*len(parent), 6*len(vertices)))
    R[np.arange(len(R)), (6*parent[:, None]+np.arange(6)).ravel()] = 1.
    J = np.zeros((6*(len(parent)-len(vertices)), len(R)))
    for row, v in enumerate(range(len(vertices), len(parent))):
        for c in range(6):
            J[6*row+c, 6*v+c] = 1.
            J[6*row+c, 6*parent[v]+c] = -1.
    return {"parent_vertex": parent, "faces": split,
            "cut_edges": np.asarray(keys, int).reshape(-1, 2),
            "seam_faces": np.asarray(seam_faces, int).reshape(-1, 2),
            "bank_vertices": np.asarray(bank_vertices, int).reshape(-1, 2, 2),
            "prolongation": R, "jump_operator": J}


def cohort_quadrature(triangle, frame, column_mass, density, cohort,
                      old0, old1, C, increment18=None, factor=1.):
    """Independent angular/radial integration of one inherited cohort.

    The only displacement here is a NEW probe increment.  Absolute saved
    displacement must not be added to already calculated elastic memory.
    """
    N, wa = angular_rule(4)
    lo, hi = float(cohort["q_lo"]), float(cohort.get("live_q_hi", cohort.get("q_hi")))
    q, wr = radial_rule(lo, hi, 32)
    lam, _ = conical_scales(triangle, column_mass, density, q)
    s1, s2 = float(wr@(1/lam)), float(wr@(1/lam**2))
    B0, B1 = independent_strain_operators(triangle, frame, N)
    interpolation = N@np.linalg.inv(SOURCE_ANGULAR_POINTS)
    e0, e1 = interpolation@_array(old0, (3, 6)), interpolation@_array(old1, (3, 6))
    C = _array(C, (6, 6))
    if not np.isfinite(factor) or factor < 0:
        raise ValueError("Independent cohort probe factor must be nonnegative finite")
    du = np.zeros(18) if increment18 is None else _array(increment18, (18,))
    e0 += factor*(B0@du); e1 += factor*(B1@du)
    volume = float(column_mass)*(hi-lo)/float(density)
    mean = e0+s1*e1
    stress, stress1 = mean@C.T, e1@C.T
    variance = s2-s1*s1
    # Direct radial quadrature variance avoids negative roundoff when the
    # two stable radial moments nearly cancel in very thin newborn cells.
    variance = float(wr@((1/lam-s1)**2))
    energy = .5*volume*np.einsum("p,pa,pa->", wa, mean, stress)
    energy += .5*volume*variance*np.einsum("p,pa,pa->", wa, e1, stress1)
    Bmean = B0+s1*B1
    force = volume*(np.einsum("p,pai,pa->i", wa, Bmean, stress)
                    +variance*np.einsum("p,pai,pa->i", wa, B1, stress1))
    K = volume*(np.einsum("p,pai,ab,pbj->ij", wa, Bmean, C, Bmean, optimize=True)
                +variance*np.einsum("p,pai,ab,pbj->ij", wa, B1, C, B1, optimize=True))
    source_B0, source_B1 = independent_strain_operators(triangle, frame, SOURCE_ANGULAR_POINTS)
    return {"stiffness": K, "force_n": force, "elastic_energy_j": float(energy),
            "constant_strain": _array(old0)+factor*(source_B0@du),
            "inverse_scale_strain": _array(old1)+factor*(source_B1@du),
            "mean_elastic_strain": np.einsum("p,pa->a", wa, mean),
            "mean_stress_pa": np.einsum("p,pa->a", wa, stress),
            "volume_m3": volume}


def integrated_operator(vertices_m, parent_faces, bank_faces, column_mass,
                        density, cohorts, memory, frames, matrices,
                        increment_m=None, strain_factors=None):
    """Assemble independently on split connectivity, retaining parent cones."""
    vertices, parent_faces, bank_faces = (_array(vertices_m),
        _array(parent_faces, integer=True), _array(bank_faces, integer=True))
    # ndof is determined by actual split faces; no unused bank is accepted.
    if bank_faces.shape != parent_faces.shape or bank_faces.min() < 0:
        raise ValueError("Independent bank faces changed the material face catalogue")
    ndof = 6*(int(bank_faces.max())+1)
    if set(bank_faces.ravel()) != set(range(ndof//6)):
        raise ValueError("Independent bank operator has an unused bank vertex")
    du = np.zeros(ndof) if increment_m is None else _array(increment_m, (ndof,))
    factors = np.ones(len(cohorts)) if strain_factors is None else _array(strain_factors, (len(cohorts),))
    if len(memory) != len(cohorts) or len(frames) != len(cohorts) or len(matrices) != len(cohorts):
        raise ValueError("Independent bank material arrays lost cohort rows")
    K, force = np.zeros((ndof, ndof)), np.zeros(ndof)
    energy, values0, values1, means, stresses = 0., [], [], [], []
    ids = []
    for index, cohort in enumerate(cohorts):
        if int(memory[index]["id"]) != int(cohort["id"]):
            raise ValueError("Independent bank memory row changed cohort identity")
        ids.append(int(cohort["id"]))
        face = int(cohort["face_index"])
        dofs = (6*bank_faces[face, :, None]+np.arange(6)).ravel()
        result = cohort_quadrature(vertices[parent_faces[face]], frames[index],
            column_mass[face], density, cohort, memory[index]["strain0"],
            memory[index]["strain1"], matrices[index], du[dofs], factors[index])
        K[np.ix_(dofs, dofs)] += factors[index]*result["stiffness"]
        np.add.at(force, dofs, result["force_n"])
        energy += result["elastic_energy_j"]
        values0.append(result["constant_strain"]); values1.append(result["inverse_scale_strain"])
        means.append(result["mean_elastic_strain"]); stresses.append(result["mean_stress_pa"])
    if len(set(ids)) != len(ids):
        raise ValueError("Independent bank operator duplicated a material cohort")
    return {"stiffness": K, "force_n": force, "elastic_energy_j": float(energy),
            "constant_strain": np.asarray(values0), "inverse_scale_strain": np.asarray(values1),
            "mean_elastic_strain": np.asarray(means), "mean_stress_pa": np.asarray(stresses)}


def momentum(mass, velocity, vertices_m):
    """Integrals of physical Cartesian momentum from exact velocity shapes."""
    X, v = _array(vertices_m), _array(velocity)
    M = _array(mass, (6*len(X), 6*len(X)))
    if v.shape != (len(M),):
        raise ValueError("Independent physical momentum velocity shape differs")
    translation, rotation = np.zeros((len(M), 3)), np.zeros((len(M), 3))
    for vertex, x in enumerate(X):
        translation[6*vertex:6*vertex+3] = np.eye(3)
        for axis, unit in enumerate(np.eye(3)):
            rotation[6*vertex:6*vertex+3, axis] = np.cross(unit, x)
            rotation[6*vertex+3:6*vertex+6, axis] = np.cross(unit, x)
    mv = M@v
    return {"kinetic_energy_j": float(.5*v@mv),
            "linear_momentum_kg_m_s": translation.T@mv,
            "angular_momentum_kg_m2_s": rotation.T@mv}


def _literal_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Independent bank audit refuses duplicate JSON keys")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON constant")))


def _actual_parent(path):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != ACTUAL_PARENT_RAW_SHA256:
        raise ValueError("Independent bank parent is not the sealed actual Genesis endpoint")
    packet = _literal_json(path)
    if packet["checkpoint_sha256"] != ACTUAL_PARENT_CHECKPOINT_SHA256:
        raise ValueError("Independent bank parent checkpoint binding differs")
    if hashlib.sha256(canonical({k: v for k, v in packet.items()
                              if k != "checkpoint_sha256"})).hexdigest() != packet["checkpoint_sha256"]:
        raise ValueError("Independent bank parent checkpoint seal differs")
    return packet


def _source_fields(parent):
    state, source = parent["runtime_state"], parent["source_packet"]
    geometry = state["geometry"]
    cohorts, memory = geometry["cohorts"], state["mechanics"]["state"]["cohorts"]
    if len(cohorts) != 640 or not np.isclose(geometry["time_myr"], .8682, rtol=0, atol=1e-14):
        raise ValueError("Independent bank actual endpoint clock/catalogue differs")
    if any(state["fault_history"]["fault_active"]):
        raise ValueError("Independent bank control changed the actual zero-active-fault source")
    shell = source["source"]["parameters"]["shell"]
    raw_frames = _array(source["current_geometry"]["face_frames_xyz"])
    face_frames = np.concatenate((raw_frames, np.cross(raw_frames[:, :, 0],
                                                      raw_frames[:, :, 1])[:, :, None]), axis=2)
    faces, vertices = _array(geometry["faces"], integer=True), _array(geometry["vertices_m"])
    source_vertices = (_array(source["current_geometry"]["vertices_unit_xyz"])
                       *(1000*float(source["current_geometry"]["radius_km"])))
    _close(vertices, source_vertices, "original reference vertices", rtol=0, atol=0)
    _exact(geometry["faces"], source["current_geometry"]["faces"], "original reference faces")
    # The inherited frame is also derived from the raw Cartesian triangle.
    for index, row in enumerate(faces):
        triangle = vertices[row]
        first = triangle[1]-triangle[0]; first /= np.linalg.norm(first)
        normal = np.cross(triangle[1]-triangle[0], triangle[2]-triangle[0]); normal /= np.linalg.norm(normal)
        _close(face_frames[index], np.column_stack((first, np.cross(normal, first), normal)),
               "literal source face frame", rtol=3e-12, atol=3e-12)
    frames, matrices = [], []
    for cohort in cohorts:
        face = int(cohort["face_index"])
        frames.append(face_frames[face])
        matrices.append(isotropic_matrix(shell["young_modulus_pa"], shell["poisson_ratio"],
            state["fault_history"]["damage"][face], shell["residual_stiffness"]))
    return state, source, geometry, cohorts, memory, np.asarray(frames), np.asarray(matrices)


def audit_fixture(fixture, parent):
    """Audit a complete source-bound control DTO against external raw source."""
    if fixture.get("format") != FORMAT or fixture.get("scope") != SCOPE:
        raise ValueError("Independent bank audit refuses a different fixture scope")
    if fixture.get("parent_raw_sha256") != ACTUAL_PARENT_RAW_SHA256:
        raise ValueError("Independent bank fixture raw parent pin differs")
    if "fixture_sha256" in fixture:
        expected = hashlib.sha256(canonical({k: v for k, v in fixture.items()
                                            if k != "fixture_sha256"})).hexdigest()
        if fixture["fixture_sha256"] != expected:
            raise ValueError("Independent bank fixture self-seal differs")
    _exact(fixture["parent_checkpoint"], parent, "entire actual parent checkpoint")
    _keys(fixture, ("format", "scope", "parent_raw_sha256", "parent_path", "parent_checkpoint",
                   "prescribed_basal", "fixture_cases", "angular_momentum_position",
                   "increment_semantics", "actual_no_cut_transfer", "fixture_sha256"), "fixture")
    _exact(fixture["angular_momentum_position"], "unchanged_reference_chart_r_equals_lambda_X",
           "angular momentum position scope")
    _exact(fixture["increment_semantics"], "new_increment_only_absolute_q_is_not_reapplied_to_accepted_memory",
           "accepted memory increment semantics")
    actual_descriptor_report = audit_transfer_descriptor(fixture["actual_no_cut_transfer"], parent)
    state, source, geometry, cohorts, memory, frames, matrices = _source_fields(parent)
    vertices, faces = _array(geometry["vertices_m"]), _array(geometry["faces"], integer=True)
    masses, rho = _array(geometry["column_mass_kg"]), float(geometry["density_kg_m3"])
    shell = source["source"]["parameters"]["shell"]
    contact = json.loads(state["contact_parameters_json"])
    traction = prescribed_traction(vertices, faces, masses, rho, cohorts,
                                   source["current_geometry"]["radius_km"], shell)[faces]
    drag = float(contact["basal_drag_pa_s_m"])
    _keys(fixture["prescribed_basal"], ("traction_face_xyz_pa", "drag_pa_s_per_m"), "prescribed basal")
    _close(fixture["prescribed_basal"]["traction_face_xyz_pa"], traction, "source basal traction", atol=1e-8)
    _close(fixture["prescribed_basal"]["drag_pa_s_per_m"], drag, "source basal drag", rtol=0, atol=0)
    parent_operator = integrated_operator(vertices, faces, faces, masses, rho, cohorts, memory, frames, matrices)
    parent_mass = mass_matrix(vertices, faces, masses, rho, cohorts)
    parent_D, parent_F = basal_drag_and_force(vertices, faces, masses, rho, cohorts, drag, traction)
    _close(parent_operator["elastic_energy_j"], state["mechanics"]["state"]["elastic_energy_j"],
           "actual inherited endpoint energy", rtol=3e-11, atol=1e4)
    q, v = _array(state["displacement_m"]), _array(state["velocity_m_s"])
    parent_momentum = momentum(parent_mass, v, vertices)
    _close(parent_momentum["kinetic_energy_j"], state["kinetic_energy_j"], "actual endpoint kinetic energy", atol=1e-7)
    reports, names = [], set()
    for case in fixture["fixture_cases"]:
        _keys(case, ("name", "synthetic_computational_cut", "cut_edges", "topology", "maps",
                     "operator", "tied", "probes", "transfer_descriptor"), "fixture case")
        name = case["name"]
        if name in names:
            raise ValueError("Independent bank fixture has duplicate case names")
        names.add(name)
        if case.get("synthetic_computational_cut") is not True:
            raise ValueError("Independent bank selected edge fixture must declare its synthetic scope")
        topo = reference_topology(vertices, faces, case["cut_edges"])
        _keys(case["topology"], ("parent_vertex", "faces", "seam_faces", "bank_vertices"), "topology")
        _keys(case["maps"], ("prolongation", "jump_operator", "prolongation_shape", "jump_operator_shape"), "maps")
        case_descriptor_report = audit_transfer_descriptor(case["transfer_descriptor"], parent, topology=topo)
        for key in ("parent_vertex", "faces", "seam_faces", "bank_vertices"):
            a = _array(case["topology"][key], integer=True)
            if a.size == 0 and topo[key].size == 0:
                a = a.reshape(topo[key].shape)
            if not np.array_equal(a, topo[key]):
                raise ValueError(f"Independent bank topology {key} differs")
        R, J = topo["prolongation"], topo["jump_operator"]
        _close(case["maps"]["prolongation"], R, "six-DOF prolongation", rtol=0, atol=0)
        declared_J = _array(case["maps"]["jump_operator"])
        if declared_J.size == 0 and J.size == 0:
            declared_J = declared_J.reshape(J.shape)
        _exact(case["maps"]["jump_operator_shape"], list(J.shape), "declared jump operator shape")
        _exact(case["maps"]["prolongation_shape"], list(R.shape), "declared prolongation shape")
        _close(declared_J, J, "six-DOF jump operator", rtol=0, atol=0)
        split_X = vertices[topo["parent_vertex"]]
        bank_op = integrated_operator(vertices, faces, topo["faces"], masses, rho, cohorts, memory, frames, matrices)
        bank_mass = mass_matrix(split_X, topo["faces"], masses, rho, cohorts)
        bank_D, bank_F = basal_drag_and_force(split_X, topo["faces"], masses, rho, cohorts, drag, traction)
        declared = case["operator"]
        _keys(declared, ("stiffness", "mass_matrix", "basal_drag_matrix", "external_force_n",
                         "prestress_force_n", "elastic_energy_j"), "operator")
        errors = {}
        for key, expected in (("stiffness", bank_op["stiffness"]), ("mass_matrix", bank_mass),
            ("basal_drag_matrix", bank_D), ("external_force_n", bank_F),
            ("prestress_force_n", bank_op["force_n"]), ("elastic_energy_j", bank_op["elastic_energy_j"])):
            if key in ("stiffness", "mass_matrix", "basal_drag_matrix"):
                errors[key] = _six_dof_matrix_close(declared[key], expected, f"independent bank {key}")
            elif key in ("external_force_n", "prestress_force_n"):
                errors[key] = _six_dof_force_close(declared[key], expected, f"independent bank {key}")
            else:
                errors[key] = _close(declared[key], expected, f"independent bank {key}", atol=1e-7)
        for key, bank, old in (("stiffness", bank_op["stiffness"], parent_operator["stiffness"]),
                              ("mass", bank_mass, parent_mass), ("basal_drag", bank_D, parent_D)):
            errors[f"tied_{key}"] = _six_dof_matrix_close(R.T@bank@R, old, f"tied {key} pullback")
        _six_dof_force_close(R.T@bank_op["force_n"], parent_operator["force_n"], "tied endpoint prestress force")
        _six_dof_force_close(R.T@bank_F, parent_F, "tied basal force")
        tied = case["tied"]
        _keys(tied, ("displacement_m", "velocity_m_s", "kinetic_energy_j",
                    "linear_momentum_kg_m_s", "angular_momentum_kg_m2_s"), "tied state")
        _close(tied["displacement_m"], R@q, "literal copied displacement", rtol=0, atol=0)
        _close(tied["velocity_m_s"], R@v, "literal copied velocity", rtol=0, atol=0)
        _close(J@_array(tied["displacement_m"]), np.zeros(len(J)), "all-depth displacement tie", rtol=0, atol=0)
        _close(J@_array(tied["velocity_m_s"]), np.zeros(len(J)), "all-depth velocity tie", rtol=0, atol=0)
        new_momentum = momentum(bank_mass, R@v, split_X)
        for key, expected in new_momentum.items():
            _close(tied[key], expected, f"bank physical {key}", atol=1e-7)
            _close(expected, parent_momentum[key], f"tied physical {key}", atol=1e-7)
        probe_reports = []
        for probe in case["probes"]:
            _keys(probe, ("name", "parent_increment_m", "bank_increment_m", "synthetic_nonzero_jump",
                         "elastic_energy_j", "force_n", "mechanical_work_j", "jump_coefficients_m",
                         "constant_strain", "inverse_scale_strain"), "probe")
            inc = _array(probe["bank_increment_m"], (len(R),))
            result = integrated_operator(vertices, faces, topo["faces"], masses, rho,
                                         cohorts, memory, frames, matrices, inc)
            work = float(bank_op["force_n"]@inc+.5*inc@bank_op["stiffness"]@inc)
            _close(result["elastic_energy_j"]-bank_op["elastic_energy_j"], work,
                   "new incremental elastic work identity", rtol=8e-11, atol=2e4)
            for key in ("elastic_energy_j", "force_n", "constant_strain", "inverse_scale_strain"):
                atol = 2e-12 if key in ("constant_strain", "inverse_scale_strain") else 1e-7
                if key == "force_n":
                    _six_dof_force_close(probe[key], result[key], "independent probe force")
                else:
                    _close(probe[key], result[key], f"independent probe {key}", atol=atol)
            _close(probe["mechanical_work_j"], work, "independent probe work", atol=2e4)
            _close(probe["jump_coefficients_m"], J@inc, "full six-DOF probe jump", rtol=0, atol=0)
            if probe["synthetic_nonzero_jump"] is False:
                parent_inc = _array(probe["parent_increment_m"], (len(q),))
                _close(inc, R@parent_inc, "tied probe prolongation", rtol=0, atol=0)
                original = integrated_operator(vertices, faces, faces, masses, rho,
                                               cohorts, memory, frames, matrices, parent_inc)
                _close(result["elastic_energy_j"], original["elastic_energy_j"], "tied probe energy", atol=1e4)
                _six_dof_force_close(R.T@result["force_n"], original["force_n"], "tied probe physical force")
                _close(J@inc, np.zeros(len(J)), "tied probe zero jump", rtol=0, atol=0)
            elif probe["synthetic_nonzero_jump"] is True:
                if probe["parent_increment_m"] is not None or not np.any(J@inc):
                    raise ValueError("Independent nonzero jump fixture is not explicitly synthetic")
            else:
                raise ValueError("Independent probe requires a literal synthetic flag")
            probe_reports.append({"name": probe["name"], "synthetic_nonzero_jump": probe["synthetic_nonzero_jump"],
                                  "elastic_energy_j": result["elastic_energy_j"], "mechanical_work_j": work,
                                  "maximum_jump_coefficient_m": float(np.max(np.abs(J@inc), initial=0.))})
        reports.append({"name": name, "parent_vertices": len(vertices), "bank_vertices": len(split_X),
                        "independent_jump_rows": len(J), "relative_errors": errors, "probes": probe_reports})
        reports[-1]["transfer_descriptor"] = case_descriptor_report
    if not reports:
        raise ValueError("Independent bank control has no declared computational fixture")
    return _seal({"format": "independent-genesis-volume-bank-audit-0.1", "scope": SCOPE,
                  "all_checks_pass": True, "producer_imports": False,
                  "parent_raw_sha256": ACTUAL_PARENT_RAW_SHA256,
                  "parent_checkpoint_sha256": ACTUAL_PARENT_CHECKPOINT_SHA256,
                  "fixture_canonical_sha256": hashlib.sha256(canonical(fixture)).hexdigest(),
                  "actual_time_myr": geometry["time_myr"], "actual_cohorts": len(cohorts),
                  "actual_active_faults": 0, "physical_fault_surface_inferred": False,
                  "actual_no_cut_transfer": actual_descriptor_report,
                  "angular_momentum_position": fixture["angular_momentum_position"],
                  "six_dof_matrix_blocks_compared_separately": True,
                  "ties_released": False, "physical_banks_contact_admitted": False,
                  "cases": reports})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture")
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit_fixture(_literal_json(args.fixture), _actual_parent(args.parent_checkpoint))
    rendered = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+"\n"
    output = Path(args.output)
    with output.open("x") as stream:
        stream.write(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
