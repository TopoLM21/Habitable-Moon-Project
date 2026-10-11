"""Independent NumPy quadrature for the declared volume midpoint step.

No tectonics/producer imports are used. Angular integration uses a Duffy
tensor rule; radial integration uses the material mass fraction, and Maxwell
work uses a separate time rule rather than producer beta/theta formulas.

Endpoint velocities are the discrete midpoint momentum variables. The work
path is affine displacement. This audit never equates both interpolations
to one continuously resolved physical trajectory.
"""
from __future__ import annotations

from fractions import Fraction
import argparse
import base64
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np


SOURCE_ANGULAR_POINTS = np.array(((2/3, 1/6, 1/6), (1/6, 2/3, 1/6),
                                (1/6, 1/6, 2/3)))


def _array(value, shape=None):
    result = np.asarray(value, float)
    if (shape is not None and result.shape != shape) or not np.isfinite(result).all():
        raise ValueError("Independent audit received an invalid finite array")
    return result


def _high(cohort):
    return float(cohort["live_q_hi"] if "live_q_hi" in cohort else cohort["q_hi"])


def angular_rule(order=4):
    """Normalized quadrature on N_i>=0, sum(N)=1; independent degree>=2 rule."""
    x, w = np.polynomial.legendre.leggauss(order)
    x, w = (x+1)/2, w/2
    s, t = np.meshgrid(x, x, indexing="ij")
    ws, wt = np.meshgrid(w, w, indexing="ij")
    N = np.stack(((1-s)*(1-t), s, (1-s)*t), axis=-1).reshape(-1, 3)
    weights = (2*ws*wt*(1-s)).ravel()
    return N, weights


def radial_rule(q_lo, q_hi, order=32):
    if not 0 <= q_lo < q_hi <= 1:
        raise ValueError("Independent audit needs a positive live material interval")
    x, w = np.polynomial.legendre.leggauss(order)
    return q_lo+(q_hi-q_lo)*(x+1)/2, w/2


def conical_scales(triangle, column_mass_kg, density_kg_m3, mass_fraction):
    X = _array(triangle, (3, 3))
    determinant = np.linalg.det(X.T)
    if determinant <= 0 or column_mass_kg <= 0 or density_kg_m3 <= 0:
        raise ValueError("Independent conical reference must have positive mass/orientation")
    coefficient = 6*column_mass_kg/(density_kg_m3*determinant)
    z = -coefficient*np.asarray(mass_fraction, float)
    if np.any(z <= -1):
        raise ValueError("Independent conical material reaches the centre")
    minus_one = np.expm1(np.log1p(z)/3)
    return 1+minus_one, minus_one


def mass_matrix(vertices_m, faces, column_mass_kg, density_kg_m3, cohorts):
    """Integrate physical reference mass and the exact conical velocity shapes."""
    X, faces = _array(vertices_m), np.asarray(faces, int)
    masses = _array(column_mass_kg, (len(faces),))
    N, wa = angular_rule()
    angular_gram = np.einsum("p,pi,pj->ij", wa, N, N)
    matrix = np.zeros((6*len(X), 6*len(X)))
    for cohort in cohorts:
        face = int(cohort["face_index"])
        lo, hi = float(cohort["q_lo"]), _high(cohort)
        q, wr = radial_rule(lo, hi)
        _, lm1 = conical_scales(X[faces[face]], masses[face], density_kg_m3, q)
        radial = np.array(((1., np.dot(wr, lm1)),
                           (np.dot(wr, lm1), np.dot(wr, lm1*lm1))))
        local = masses[face]*(hi-lo)*np.kron(angular_gram, np.kron(radial, np.eye(3)))
        dofs = (6*faces[face, :, None]+np.arange(6)).ravel()
        matrix[np.ix_(dofs, dofs)] += local
    return matrix


def basal_drag_and_force(vertices_m, faces, column_mass_kg, density_kg_m3,
                         cohorts, drag_pa_s_per_m, traction_face_xyz_pa):
    """Independent reference basal tangent drag/traction angular integration."""
    X, faces = _array(vertices_m), np.asarray(faces, int)
    masses = _array(column_mass_kg, (len(faces),))
    drag = np.broadcast_to(_array(drag_pa_s_per_m), (len(faces),))
    traction = _array(traction_face_xyz_pa)
    if traction.shape == (len(faces), 3):
        traction = np.repeat(traction[:, None, :], 3, axis=1)
    if traction.shape != (len(faces), 3, 3):
        raise ValueError("Independent basal traction needs face constants or three nodal vectors")
    if np.any(drag < 0):
        raise ValueError("Independent basal drag cannot be negative")
    fronts = np.zeros(len(faces))
    for cohort in cohorts:
        fronts[int(cohort["face_index"])] = max(fronts[int(cohort["face_index"])], _high(cohort))
    N, wa = angular_rule()
    matrix, force = np.zeros((6*len(X), 6*len(X))), np.zeros(6*len(X))
    for face, front in enumerate(fronts):
        if front == 0:
            continue
        triangle = X[faces[face]]
        lam, lm1 = conical_scales(triangle, masses[face], density_kg_m3, front)
        normal = np.cross(triangle[1]-triangle[0], triangle[2]-triangle[0])
        top_area = np.linalg.norm(normal)/2
        normal /= np.linalg.norm(normal)
        projector = np.eye(3)-np.outer(normal, normal)
        shape = np.zeros((len(N), 3, 18))
        for vertex in range(3):
            shape[:, :, 6*vertex:6*vertex+3] = N[:, vertex, None, None]*np.eye(3)
            shape[:, :, 6*vertex+3:6*vertex+6] = lm1*N[:, vertex, None, None]*np.eye(3)
        area = top_area*lam*lam
        local_D = area*drag[face]*np.einsum("p,pai,ab,pbj->ij", wa, shape, projector, shape)
        local_F = area*np.einsum("p,pai,pa->i", wa, shape, N@traction[face])
        dofs = (6*faces[face, :, None]+np.arange(6)).ravel()
        matrix[np.ix_(dofs, dofs)] += local_D
        np.add.at(force, dofs, local_F)
    return matrix, force


def isotropic_matrix(young_modulus_pa, poisson_ratio, damage, residual_stiffness):
    E, nu = float(young_modulus_pa), float(poisson_ratio)
    shear = E/(2*(1+nu))
    lame = E*nu/((1+nu)*(1-2*nu))
    C = np.zeros((6, 6))
    C[:3, :3] = lame
    C[np.arange(3), np.arange(3)] += 2*shear
    C[3:, 3:] = np.eye(3)*shear
    return (residual_stiffness+(1-residual_stiffness)*(1-float(damage))**2)*C


def engineering_gradient(gradient, frame):
    local = np.einsum("ia,...ij,jb->...ab", frame, gradient, frame)
    return np.stack((local[..., 0, 0], local[..., 1, 1], local[..., 2, 2],
        local[..., 0, 1]+local[..., 1, 0], local[..., 0, 2]+local[..., 2, 0],
        local[..., 1, 2]+local[..., 2, 1]), axis=-1)


def independent_strain_operators(triangle, frame, N):
    """Cartesian derivation independent of the producer's local-plane B code."""
    inverse = np.linalg.inv(np.asarray(triangle, float).T)
    qplane = inverse.sum(axis=0)
    b0, b1 = np.zeros((len(N), 6, 18)), np.zeros((len(N), 6, 18))
    for vertex in range(3):
        changing = inverse[vertex][None, :]-N[:, vertex, None]*qplane
        for axis in range(3):
            vector = np.eye(3)[axis]
            slope = engineering_gradient(vector[None, :, None]*changing[:, None, :], frame)
            direct = engineering_gradient(np.outer(vector, inverse[vertex]), frame)
            b1[:, :, 6*vertex+axis] = slope
            b0[:, :, 6*vertex+3+axis] = direct
            b1[:, :, 6*vertex+3+axis] = -slope
    return b0, b1


def temporal_factors(ratio):
    """Numerical integrals of the exact ODE fundamental solution in time."""
    ratio = float(ratio)
    if not math.isfinite(ratio) or ratio < 0:
        raise ValueError("Independent Maxwell ratio must be nonnegative finite")
    if ratio:
        boundaries = sorted(set((0., 1.)+tuple(min(1., c/ratio) for c in (1., 4., 16., 64.))))
    else:
        boundaries = (0., 1.)
    x, w = np.polynomial.legendre.leggauss(32)
    integrals = np.zeros(5)
    for low, high in zip(boundaries, boundaries[1:]):
        time, weight = low+(high-low)*(x+1)/2, (high-low)*w/2
        relaxation = np.exp(-ratio*time)
        driving = time if ratio == 0 else -np.expm1(-ratio*time)/ratio
        integrals += np.array((np.dot(weight, relaxation), np.dot(weight, driving),
            np.dot(weight, ratio*relaxation**2), np.dot(weight, 2*ratio*relaxation*driving),
            np.dot(weight, ratio*driving**2)))
    return integrals


def quadrature_cohort(triangle, frame, column_mass_kg, density_kg_m3, cohort,
                      displacement_increment18_m, old0, old1, thermal0, ratio, constitutive_matrix):
    """Integrate Maxwell energy/work/dissipation and average force independently."""
    lo, hi = float(cohort["q_lo"]), _high(cohort)
    N, wa = angular_rule()
    q, wr = radial_rule(lo, hi)
    lam, _ = conical_scales(triangle, column_mass_kg, density_kg_m3, q)
    weights = (wa[:, None]*wr[None, :]).ravel()
    volume = column_mass_kg*(hi-lo)/density_kg_m3
    b0, b1 = independent_strain_operators(triangle, frame, N)
    B = (b0[:, None, :, :]+b1[:, None, :, :]/lam[None, :, None, None]).reshape(-1, 6, 18)
    increment = np.einsum("pai,i->pa", B, displacement_increment18_m)
    interpolation = N@np.linalg.inv(SOURCE_ANGULAR_POINTS)
    e0, e1 = interpolation@_array(old0, (3, 6)), interpolation@_array(old1, (3, 6))
    old_elastic = (e0[:, None, :]+e1[:, None, :]/lam[None, :, None]).reshape(-1, 6)
    thermal = np.repeat(interpolation@_array(thermal0, (3, 6)), len(lam), axis=0)
    driver = increment-thermal
    C = _array(constitutive_matrix, (6, 6))
    a, b, aa, ab, bb = temporal_factors(ratio)
    average_stress = (a*old_elastic+b*driver)@C.T
    inner = lambda first, second: np.einsum("pi,ij,pj->p", first, C, second)
    work = volume*np.dot(weights, np.einsum("pi,pi->p", increment, average_stress))
    thermal_work = -volume*np.dot(weights, np.einsum("pi,pi->p", thermal, average_stress))
    loss = volume*np.dot(weights, aa*inner(old_elastic, old_elastic)
        +ab*inner(old_elastic, driver)+bb*inner(driver, driver))
    endpoint_driving = 1. if ratio == 0 else -math.expm1(-ratio)/ratio
    endpoint = math.exp(-ratio)*old_elastic+endpoint_driving*driver
    before_energy = .5*volume*np.dot(weights, inner(old_elastic, old_elastic))
    after_energy = .5*volume*np.dot(weights, inner(endpoint, endpoint))
    force = volume*np.einsum("p,pai,pa->i", weights, B, average_stress)
    scale = max(abs(work), abs(thermal_work), abs(loss), before_energy, after_energy, 1.)
    residual = after_energy-before_energy-(work+thermal_work-loss)
    if abs(residual) > 2e-11*scale or loss < -2e-13*scale:
        raise ValueError("Independent numerical Maxwell work identity failed")
    return {"mechanical_work_j": float(work), "thermal_work_j": float(thermal_work),
        "viscous_dissipation_j": float(loss), "before_after_damage_energy_j": float(before_energy),
        "elastic_energy_j": float(after_energy), "balance_residual_j": float(residual),
        "relative_balance_residual": float(abs(residual)/scale), "average_force_n": force}


def coefficient_energy(triangle, column_mass_kg, density_kg_m3, cohort, strain0, strain1, C):
    N, wa = angular_rule()
    q, wr = radial_rule(float(cohort["q_lo"]), _high(cohort))
    lam, _ = conical_scales(triangle, column_mass_kg, density_kg_m3, q)
    interpolation = N@np.linalg.inv(SOURCE_ANGULAR_POINTS)
    e0, e1 = interpolation@_array(strain0, (3, 6)), interpolation@_array(strain1, (3, 6))
    elastic = e0[:, None, :]+e1[:, None, :]/lam[None, :, None]
    density = .5*np.einsum("pri,ij,prj->pr", elastic, C, elastic)
    volume = column_mass_kg*(_high(cohort)-float(cohort["q_lo"]))/density_kg_m3
    return float(volume*np.einsum("p,r,pr->", wa, wr, density))


def dynamic_balance(mass_before, mass_after, mass_born, mass_removed, drag,
                    external_force, average_elastic_force, q_before, q_after,
                    velocity_before, velocity_after, dt_s, mechanical_work_j):
    """Audit the discrete midpoint equation and explicit co-moving phase flux."""
    q0, q1, v0, v1 = map(_array, (q_before, q_after, velocity_before, velocity_after))
    n = len(q0)
    M0, M1, Min, Mout, D = (_array(a, (n, n)) for a in
        (mass_before, mass_after, mass_born, mass_removed, drag))
    for name, matrix in (("before mass", M0), ("after mass", M1),
                         ("birth mass", Min), ("removed mass", Mout), ("drag", D)):
        matrix_scale = max(np.max(np.abs(matrix)), 1e-100)
        if (np.max(np.abs(matrix-matrix.T)) > 2e-12*matrix_scale
                or np.linalg.eigvalsh((matrix+matrix.T)/2).min() < -2e-12*matrix_scale):
            raise ValueError(f"Independent {name} matrix is not symmetric positive semidefinite")
    Fext, Fav = _array(external_force, (n,)), _array(average_elastic_force, (n,))
    dt = float(dt_s)
    if dt <= 0 or not math.isfinite(dt):
        raise ValueError("Independent midpoint dt must be positive")
    dq = q1-q0
    vmid = dq/dt
    midpoint_residual = v1+v0-2*vmid
    momentum_residual = M1@(v1-v0)/dt+D@vmid+Fav-Fext
    force_scale = max(np.max(np.abs(M1@(v1-v0)/dt)), np.max(np.abs(D@vmid)),
                      np.max(np.abs(Fav)), np.max(np.abs(Fext)), 1.)
    partition_error = np.max(np.abs(M1-M0-Min+Mout))/max(np.max(np.abs(M0)), np.max(np.abs(M1)), 1.)
    kinetic0, kinetic1 = float(.5*v0@M0@v0), float(.5*v1@M1@v1)
    born, removed = float(.5*v0@Min@v0), float(.5*v0@Mout@v0)
    drag_work, external_work = float(dt*vmid@D@vmid), float(Fext@dq)
    force_work = float(Fav@dq)
    global_residual = external_work+born-removed-(kinetic1-kinetic0+drag_work+mechanical_work_j)
    global_scale = max(abs(external_work), abs(born), abs(removed), kinetic0, kinetic1,
                       abs(drag_work), abs(mechanical_work_j), 1.)
    if partition_error > 3e-11:
        raise ValueError("Independent phase mass-matrix partition failed")
    if abs(force_work-mechanical_work_j) > 2e-10*max(abs(force_work), abs(mechanical_work_j), 1.):
        raise ValueError("Average elastic force is not conjugate to the Maxwell work path")
    if np.max(np.abs(midpoint_residual)) > 2e-11*max(np.max(np.abs(v0)), np.max(np.abs(v1)), 1e-25):
        raise ValueError("Discrete endpoint momentum velocities disagree with midpoint kinematics")
    if np.max(np.abs(momentum_residual)) > 2e-8*force_scale:
        raise ValueError("Independent full-coordinate momentum equation failed")
    if abs(global_residual) > 2e-9*global_scale:
        raise ValueError("Independent kinetic/phase/drag/Maxwell global work balance failed")
    return {"kinetic_before_j": kinetic0, "kinetic_after_j": kinetic1,
        "kinetic_birth_inflow_j": born, "kinetic_remelt_outflow_j": removed,
        "drag_dissipation_j": drag_work, "external_work_j": external_work,
        "mechanical_work_j": float(mechanical_work_j), "average_force_work_j": force_work,
        "global_balance_residual_j": float(global_residual),
        "relative_global_balance_residual": float(abs(global_residual)/global_scale),
        "relative_max_coordinate_momentum_residual": float(np.max(np.abs(momentum_residual))/force_scale),
        "relative_mass_matrix_partition_residual": float(partition_error),
        "scope": "discrete_midpoint_momentum_with_affine_Maxwell_work_path",
        "continuous_velocity_history_certified": False}


def decode_exact_array(value):
    return np.array([[float(Fraction(*item)) for item in row] for row in value])


def exact_reference_and_gradient(certificate):
    """Separate rational simplicial-hull and global derivative proof.

    Actual formation controls have strictly simplicial convex hull facets.
    This bounded independent checker refuses coplanar non-simplicial cases;
    the producer's more general coplanar triangulation proof is not reused.
    """
    vertices = tuple(tuple(Fraction(*x) for x in row) for row in certificate["vertices_xyz"])
    faces = tuple(tuple(row) for row in certificate["faces"])
    fields = tuple(tuple(tuple(Fraction(*x) for x in row) for row in certificate[key]) for key in
        ("before_displacement_A_xyz", "before_slope_P_xyz", "displacement_A_xyz", "slope_P_xyz"))
    lam_min = Fraction(*certificate["lambda_min"])
    if not 0 < lam_min <= 1 or len(set(vertices)) != len(vertices):
        raise ValueError("Independent exact reference coordinates/domain are invalid")
    dot = lambda a, b: sum((x*y for x, y in zip(a, b)), Fraction(0))
    sub = lambda a, b: tuple(x-y for x, y in zip(a, b))
    cross = lambda a, b: (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])
    edges, links, used = {}, {i: {} for i in range(len(vertices))}, set()
    if len({tuple(sorted(face)) for face in faces}) != len(faces):
        raise ValueError("Independent exact reference has duplicate triangles")
    maximum = Fraction(0)
    for face in faces:
        if len(face) != 3 or len(set(face)) != 3 or any(type(i) is not int or not 0 <= i < len(vertices) for i in face):
            raise ValueError("Independent exact reference face is invalid")
        used.update(face)
        v = tuple(vertices[i] for i in face)
        normal = cross(sub(v[1], v[0]), sub(v[2], v[0]))
        offset = dot(normal, v[0])
        if offset <= 0 or any(dot(normal, point) >= offset for i, point in enumerate(vertices) if i not in face):
            raise ValueError("Independent exact reference is not a strict simplicial convex hull")
        for i, a in enumerate(face):
            b, c = face[(i+1)%3], face[(i+2)%3]
            edges.setdefault(tuple(sorted((a, b))), []).append((a, b))
            links[a].setdefault(b, []).append(c)
            links[a].setdefault(c, []).append(b)
        determinant = dot(v[0], cross(v[1], v[2]))
        inverse = tuple(tuple(x/determinant for x in row) for row in
            (cross(v[1], v[2]), cross(v[2], v[0]), cross(v[0], v[1])))
        qplane = tuple(sum((row[j] for row in inverse), Fraction(0)) for j in range(3))
        for A, P in ((fields[0], fields[1]), (fields[2], fields[3])):
            a, p = tuple(A[i] for i in face), tuple(P[i] for i in face)
            H = tuple(tuple(sum((p[k][i]*inverse[k][j] for k in range(3)), Fraction(0)) for j in range(3)) for i in range(3))
            K = tuple(tuple(sum(((a[k][i]-p[k][i])*inverse[k][j] for k in range(3)), Fraction(0)) for j in range(3)) for i in range(3))
            for corner in range(3):
                for lam in (lam_min, Fraction(1)):
                    squared = sum(((H[i][j]+(K[i][j]-(a[corner][i]-p[corner][i])*qplane[j])/lam)**2
                                   for i in range(3) for j in range(3)), Fraction(0))
                    maximum = max(maximum, squared)
            inner = tuple(tuple(sum(((a[k][i]+(lam_min-1)*p[k][i])*inverse[k][j]/lam_min
                                      for k in range(3)), Fraction(0)) for j in range(3)) for i in range(3))
            maximum = max(maximum, sum((x*x for row in inner for x in row), Fraction(0)))
    if (used != set(range(len(vertices))) or len(vertices)-len(edges)+len(faces) != 2
            or any(len(rows) != 2 or rows[0] != rows[1][::-1] for rows in edges.values())):
        raise ValueError("Independent exact reference is not a closed oriented spherical boundary")
    for link in links.values():
        if any(len(rows) != 2 or len(set(rows)) != 2 for rows in link.values()):
            raise ValueError("Independent exact reference vertex is not manifold")
        reached, pending = set(), [next(iter(link))]
        while pending:
            node = pending.pop()
            if node not in reached:
                reached.add(node); pending.extend(set(link[node])-reached)
        if reached != set(link):
            raise ValueError("Independent exact reference vertex link is disconnected")
    graph = {i: set() for i in range(len(vertices))}
    for a, b in edges:
        graph[a].add(b); graph[b].add(a)
    reached, pending = set(), [0]
    while pending:
        node = pending.pop()
        if node not in reached:
            reached.add(node); pending.extend(graph[node]-reached)
    if len(reached) != len(vertices):
        raise ValueError("Independent exact reference boundary is disconnected")
    claimed = Fraction(*certificate["max_gradient_frobenius_squared"])
    if maximum != claimed or maximum >= 1:
        raise ValueError("Independent exact all-domain gradient proof differs or is unresolved")
    return maximum


def geometry_binding(certificate, vertices_m, faces, q_before, q_after,
                     cohorts, column_mass_kg, density_kg_m3, before_time_myr, after_time_myr):
    """Independently bind and recompute the cone proof used by this interval."""
    body = {key: value for key, value in certificate.items() if key != "payload_sha256"}
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if hashlib.sha256(encoded).hexdigest() != certificate["payload_sha256"]:
        raise ValueError("Independent cone certificate seal mismatch")
    exact_bound = exact_reference_and_gradient(certificate)
    vertices, faces = _array(vertices_m), np.asarray(faces, int)
    q0, q1 = _array(q_before).reshape(-1, 6), _array(q_after).reshape(-1, 6)
    fields = (decode_exact_array(certificate["before_displacement_A_xyz"]),
        decode_exact_array(certificate["before_slope_P_xyz"]),
        decode_exact_array(certificate["displacement_A_xyz"]), decode_exact_array(certificate["slope_P_xyz"]))
    expected = (q0[:, :3], q0[:, 3:], q1[:, :3], q1[:, 3:])
    if (not np.array_equal(decode_exact_array(certificate["vertices_xyz"]), vertices)
            or not np.array_equal(np.asarray(certificate["faces"], int), faces)
            or any(not np.array_equal(actual, wanted) for actual, wanted in zip(fields, expected))):
        raise ValueError("Cone proof is not bound to actual reference and before/after total fields")
    if (float(Fraction(*certificate["before_time_myr"])) != before_time_myr
            or float(Fraction(*certificate["after_time_myr"])) != after_time_myr):
        raise ValueError("Cone proof physical interval differs from the actual shared clock")
    exact_lambda_min = Fraction(*certificate["lambda_min"])
    lam_min = float(exact_lambda_min)
    smallest_live_scale = 1.
    minimum_exact_radial_margin = None
    for c in cohorts:
        face = int(c["face_index"])
        inner, _ = conical_scales(vertices[faces[face]], column_mass_kg[face], density_kg_m3, _high(c))
        smallest_live_scale = min(smallest_live_scale, float(inner))
        v = [[Fraction.from_float(float(x)) for x in row] for row in vertices[faces[face]]]
        determinant = (v[0][0]*(v[1][1]*v[2][2]-v[1][2]*v[2][1])
            -v[0][1]*(v[1][0]*v[2][2]-v[1][2]*v[2][0])
            +v[0][2]*(v[1][0]*v[2][1]-v[1][1]*v[2][0]))
        exact_live_cube = 1-6*Fraction.from_float(float(column_mass_kg[face]))*Fraction.from_float(_high(c))/(
            Fraction.from_float(float(density_kg_m3))*determinant)
        margin = exact_live_cube-exact_lambda_min**3
        if margin < 0:
            raise ValueError("Actual live material lies outside the exact cone proof radial domain")
        minimum_exact_radial_margin = margin if minimum_exact_radial_margin is None else min(minimum_exact_radial_margin, margin)
    bound = 0.
    for face in faces:
        V = vertices[face].T
        inverse = np.linalg.inv(V)
        qplane = inverse.sum(axis=0)
        for A, P in ((fields[0], fields[1]), (fields[2], fields[3])):
            a, p = A[face].T, P[face].T
            H, K = p@inverse, (a-p)@inverse
            for vertex in range(3):
                for lam in (lam_min, 1.):
                    G = H+(K-np.outer((a-p)[:, vertex], qplane))/lam
                    bound = max(bound, float(np.sum(G*G)))
            inner_gradient = (a+(lam_min-1)*p)@inverse/lam_min
            bound = max(bound, float(np.sum(inner_gradient*inner_gradient)))
    claimed = float(Fraction(*certificate["max_gradient_frobenius_squared"]))
    if bound >= 1 or not math.isclose(bound, claimed, rel_tol=2e-10, abs_tol=1e-25):
        raise ValueError("Independent cone gradient norm does not agree with the exact certificate")
    return {"independent_max_gradient_frobenius_squared": bound,
        "exact_certificate_max_gradient_frobenius_squared": claimed,
        "independently_recomputed_exact_bound": [exact_bound.numerator, exact_bound.denominator],
        "exact_simplicial_convex_reference_certified": True,
        "smallest_live_scale": smallest_live_scale, "lambda_min": lam_min,
        "minimum_exact_massq_radial_cube_margin": [minimum_exact_radial_margin.numerator, minimum_exact_radial_margin.denominator],
        "actual_before_after_fields_bound": True, "all_live_material_within_domain": True,
        "whole_clock_connected_body_noninterpenetration": True,
        "old_native_C5_to_C1_admission_claimed": False}


def global_temperatures(global_state, p):
    """Bisection inversion of the declared enthalpy/IAPWS reservoir law."""
    area = 4*math.pi*(1000*p["radius_km"])**2
    water = p["water_volume_km3"]*1e9*p["water_density_kg_m3"]/area
    surface = p["surface_layer_depth_m"]*p["surface_layer_density_kg_m3"]
    mantle = 5.9722e24*p["mass_earth"]*p["mantle_mass_fraction"]/area-surface

    def rock(t):
        fraction = min(1., max(0., (t-p["solidus_k"])/(p["liquidus_k"]-p["solidus_k"])))
        return p["silicate_heat_capacity_j_kg_k"]*t+p["silicate_latent_heat_j_kg"]*fraction

    def vapor(t):
        if t >= 647.096:
            return water
        if t < 273.16:
            pressure = 611.657*math.exp(5420.*(1/273.16-1/max(t, 1.)))
        else:
            theta = 1-t/647.096
            pressure = 22.064e6*math.exp(647.096/t*sum(a*theta**b for a, b in
                ((-7.85951783, 1), (1.84408259, 1.5), (-11.7866497, 3),
                 (22.6807411, 3.5), (-15.9618719, 4), (1.80122502, 7.5))))
        amount = min(water, pressure/p["surface_gravity_m_s2"])
        if water*p["surface_gravity_m_s2"] > 22.064e6:
            x = min(1., max(0., (647.096-t)/p["critical_transition_width_k"]))
            amount = water+x*x*(3-2*x)*(amount-water)
        return amount

    def invert(target, law):
        low, high = 0., max(4000., 2*p["initial_temperature_k"])
        if not law(low) <= target < law(high):
            raise ValueError("Independent thermal energy lies outside its inversion bracket")
        for _ in range(90):
            middle = (low+high)/2
            if law(middle) < target:
                low = middle
            else:
                high = middle
        return (low+high)/2

    energy = global_state["energy"]
    return (invert(energy[0]*1e12, lambda t: mantle*rock(t)),
        invert(energy[1]*1e12, lambda t: surface*rock(t)
            +water*p["water_heat_capacity_j_kg_k"]*t+p["water_latent_heat_j_kg"]*vapor(t)))


def conductive_temperatures(thermal, parameters):
    h = _array(thermal["column_enthalpy"])
    cp, latent = parameters["silicate_heat_capacity_j_kg_k"], parameters["silicate_latent_heat_j_kg"]
    solidus, liquidus = parameters["solidus_k"], parameters["liquidus_k"]
    width = liquidus-solidus
    return np.where(h <= cp*solidus, h/cp, np.where(h >= cp*liquidus+latent,
        (h-latent)/cp, (h+latent*solidus/width)/(cp+latent/width)))


def profile_mean(layer_mass, layer_temperature, surface, mantle, lo, hi):
    boundary = np.r_[0., np.cumsum(layer_mass)/np.sum(layer_mass)]
    nodes = np.r_[0., .5*(boundary[:-1]+boundary[1:]), 1.]
    values = np.r_[surface, layer_temperature, mantle]
    integral = 0.
    for a, b, va, vb in zip(nodes[:-1], nodes[1:], values[:-1], values[1:]):
        left, right = max(a, lo), min(b, hi)
        if right > left:
            integral += (right-left)*(va+(vb-va)*((left+right)/2-a)/(b-a))
    return integral/(hi-lo)


def prescribed_traction(vertices_m, faces, column_mass, density, cohorts, radius_km, shell):
    """Recompute the source's prescribed pattern and conical depth transmission."""
    vertices = _array(vertices_m)/(1000*radius_km)
    faces = np.asarray(faces, int)
    triangle = vertices[faces]
    a, b, c = triangle.transpose(1, 0, 2)
    determinant = np.einsum("fi,fi->f", a, np.cross(b, c))
    spherical_area = 2*np.arctan2(determinant, 1+np.einsum("fi,fi->f", a, b)
        +np.einsum("fi,fi->f", b, c)+np.einsum("fi,fi->f", c, a))
    fronts = np.zeros(len(faces))
    for cohort in cohorts:
        face = int(cohort["face_index"])
        fronts[face] = max(fronts[face], _high(cohort))
    normal = np.cross(b-a, c-a)
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    height = 1000*radius_km*np.einsum("fi,fi->f", a, normal)
    bottom = np.array([conical_scales(vertices_m[faces[i]], column_mass[i], density, q)[0]
                       for i, q in enumerate(fronts)])
    depth_km = (1-bottom)*height/1000
    coupling = -np.expm1(-depth_km/shell["traction_coupling_depth_km"])
    node_coupling, node_area = np.zeros(len(vertices)), np.zeros(len(vertices))
    for face, nodes in enumerate(faces):
        node_coupling[nodes] += spherical_area[face]*coupling[face]
        node_area[nodes] += spherical_area[face]
    node_coupling /= node_area
    rng = np.random.default_rng(int(shell["seed"]))
    matrix = rng.normal(size=(3, 3))
    matrix = (matrix+matrix.T)/2
    matrix -= np.eye(3)*np.trace(matrix)/3
    gradient = 2*vertices@matrix.T
    gradient -= vertices*np.sum(gradient*vertices, axis=1)[:, None]
    gradient /= max(2*np.linalg.norm(matrix, 2), 1e-12)
    return shell["convective_traction_pa"]*node_coupling[:, None]*gradient


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def _assert_close(actual, expected, name, rtol=3e-10, atol=1e-12):
    actual, expected = _array(actual), _array(expected)
    if actual.shape != expected.shape or not np.allclose(actual, expected, rtol=rtol, atol=atol):
        scale = max(float(np.max(np.abs(actual))), float(np.max(np.abs(expected))), 1e-100)
        raise ValueError(f"Independent {name} differs; max scaled error={np.max(np.abs(actual-expected))/scale:.6g}")


def endpoint_clock_binding(state, expected_time_myr):
    clocks = (state["geometry"]["time_myr"], state["thermal"]["time_myr"],
        state["history"]["time_myr"], state["mechanics"]["state"]["time_myr"],
        state["thermal"]["global_state"]["time_myr"], state["thermal"]["orbit"]["time_myr"])
    if not math.isfinite(expected_time_myr) or any(t != expected_time_myr for t in clocks):
        raise ValueError("Independent endpoint does not have one actual shared clock")


def source_dto_binding(packet):
    """Decode literal embedded NPZ arrays/metadata; this is not a Genesis rerun."""
    source = packet["source_packet"]
    encoded = base64.b64decode(source["source"]["checkpoint_base64"], validate=True)
    digest = hashlib.sha256(encoded).hexdigest()
    if digest != packet["source_checkpoint_sha256"] or digest != source["source"]["checkpoint_sha256"]:
        raise ValueError("Independent literal embedded formation bytes differ from source identity")
    with np.load(io.BytesIO(encoded), allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        if (_canonical(metadata["parameters"]) != _canonical(source["source"]["parameters"])
                or metadata["parameter_hash"] != source["source"]["parameter_hash"]):
            raise ValueError("Independent source DTO changed embedded source parameters")
        for field in ("bulk_birth_history", "cohort_mechanics"):
            if _canonical(metadata["state"][field]) != _canonical(source[field]):
                raise ValueError("Independent source DTO changed its embedded formation history: "+field)
        for actual, name in ((source["current_geometry"]["vertices_unit_xyz"], "vertices"),
            (source["current_columns"]["layer_mass_kg"], "layer_mass_kg"),
            (source["current_columns"]["specific_enthalpy_j_kg"], "column_enthalpy"),
            (source["current_shell_memory"]["damage"], "damage"),
            (source["current_shell_memory"]["elastic_strain_xx_yy_gamma_xy"], "elastic_strain"),
            (source["current_shell_memory"]["velocity_km_myr_world"], "velocity_km_myr")):
            _assert_close(actual, data[name], "literal source DTO "+name, rtol=0, atol=0)
        if (source["time_myr"] != metadata["state"]["time_myr"]
                or source["current_geometry"]["radius_km"] != metadata["state"]["radius_km"]
                or source["birth_source_identity"] != metadata["state"]["bulk_birth_history"]["source_identity"]
                or _canonical(source["current_geometry"]["faces"]) != _canonical(metadata["state"]["bulk_birth_history"]["initial_faces"])):
            raise ValueError("Independent source DTO changed embedded clock/reference context")
    return {"embedded_npz_arrays_and_histories_bound": True,
        "complete_Genesis_reexecution_claimed": False, "checkpoint_sha256": digest}


def audit_initial(packet):
    source, initial = packet["source_packet"], packet["runtime_state"]
    endpoint_clock_binding(initial, source["time_myr"])
    geometry = initial["geometry"]
    vertices, faces = _array(geometry["vertices_m"]), np.asarray(geometry["faces"], int)
    shell = source["source"]["parameters"]["shell"]
    expected_vertices = _array(source["current_geometry"]["vertices_unit_xyz"])*(1000*source["current_geometry"]["radius_km"])
    _assert_close(vertices, expected_vertices, "literal source reference", rtol=0, atol=0)
    _assert_close(faces, source["current_geometry"]["faces"], "literal source material connectivity", rtol=0, atol=0)
    _assert_close(geometry["column_mass_kg"], _array(source["current_columns"]["layer_mass_kg"]).sum(axis=1),
                  "conserved source column mass", rtol=0, atol=0)
    if (geometry["density_kg_m3"] != shell["density_kg_m3"]
            or geometry["source_identity"] != source["birth_source_identity"]
            or geometry["source_checkpoint_sha256"] != source["source"]["checkpoint_sha256"]):
        raise ValueError("Independent initial geometry changed its physical source context")
    if np.any(_array(initial["displacement_m"]) != 0):
        raise ValueError("Independent initial reference has a replacement material displacement")
    old_cohorts = {int(c["id"]): c for c in source["cohort_mechanics"]["state"]["cohorts"]}
    memories = {int(c["id"]): c for c in initial["mechanics"]["state"]["cohorts"]}
    if set(old_cohorts) != set(memories):
        raise ValueError("Independent initial source cohort IDs changed")
    energy = 0.
    for c in geometry["cohorts"]:
        cid, face = int(c["id"]), int(c["face_index"])
        xx, yy, gamma = old_cohorts[cid]["elastic_strain"]
        zz = -shell["poisson_ratio"]/(1-shell["poisson_ratio"])*(xx+yy)
        expected0 = np.tile((xx, yy, zz, gamma, 0., 0.), (3, 1))
        _assert_close(memories[cid]["strain0"], expected0, "initial inherited cohort tensor", rtol=2e-12, atol=2e-14)
        _assert_close(memories[cid]["strain1"], np.zeros((3, 6)), "computed constant source radial coefficient", rtol=0, atol=2e-14)
        C = isotropic_matrix(shell["young_modulus_pa"], shell["poisson_ratio"],
            initial["thermal"]["damage"][face], shell["residual_stiffness"])
        energy += coefficient_energy(vertices[faces[face]], geometry["column_mass_kg"][face],
            geometry["density_kg_m3"], c, memories[cid]["strain0"], memories[cid]["strain1"], C)
    source_energy = source["instantaneous_cohort_plane_stress_3d"]["elastic_energy_j"]
    _assert_close(energy, source_energy, "full imported cohort energy", rtol=2e-12, atol=1e3)
    # Reconstruct the old spherical triangle using world edge constraints,
    # independently of the producer's local (x,y,h) reconstruction routine.
    interval = source["bulk_birth_history"]["intervals"][-1]
    records = [c for c in source["bulk_birth_history"]["cohorts"]
        if c["birth_before_time_myr"] == interval["before_time_myr"]
        and c["birth_after_time_myr"] == interval["after_time_myr"]]
    if {int(c["face_index"]) for c in records} != set(range(len(faces))):
        raise ValueError("Independent source last accepted charts lack complete nodal coverage")
    previous = np.full_like(vertices, np.nan)
    chart_resolution = 0.
    for c in records:
        G, Q = _array(c["birth_metric_m2"]), _array(c["birth_frame_xyz"])
        first = math.sqrt(G[0, 0])
        upper = np.linalg.cholesky(G).T
        edges = Q[:, :2]@upper
        projection = edges@np.linalg.solve(G, -.5*np.diag(G))
        radius = 1000*c["birth_radius_km"]
        vertex0 = projection+math.sqrt(radius*radius-projection@projection)*Q[:, 2]
        triangle = np.array((vertex0, vertex0+edges[:, 0], vertex0+edges[:, 1]))
        tolerance = 256*np.finfo(float).eps*radius*max(1., first/upper[1, 1])
        chart_resolution = max(chart_resolution, tolerance)
        for node, point in zip(faces[int(c["face_index"])], triangle):
            if np.isfinite(previous[node]).all():
                _assert_close(point, previous[node], "shared recovered source nodal position", rtol=0, atol=tolerance)
            else:
                previous[node] = point
    elapsed = (interval["after_time_myr"]-interval["before_time_myr"])*365.25*86400*1e6
    rate = (vertices-previous)/elapsed
    actual_velocity = _array(initial["velocity_m_s"]).reshape(-1, 6)
    _assert_close(actual_velocity[:, :3], rate, "last accepted source nodal rate", rtol=2e-10,
                  atol=chart_resolution/elapsed)
    _assert_close(actual_velocity[:, 3:], actual_velocity[:, :3], "declared current co-moving radial rate", rtol=0, atol=0)
    M = mass_matrix(vertices, faces, geometry["column_mass_kg"], geometry["density_kg_m3"], geometry["cohorts"])
    kinetic = float(.5*actual_velocity.ravel()@M@actual_velocity.ravel())
    _assert_close(kinetic, initial["initial_kinetic_energy_j"], "initial current kinetic energy", rtol=3e-10, atol=1e-6)
    return {"full_inherited_cohort_energy_j": energy, "source_energy_j": source_energy,
        "face_mean_energy_j": source["instantaneous_plane_stress_3d"]["elastic_energy_of_face_mean_j"],
        "source_cohort_count": len(memories), "initial_current_kinetic_energy_j": kinetic,
        "source_last_interval_myr": [interval["before_time_myr"], interval["after_time_myr"]],
        "recovered_nodal_rate_max_difference_m_s": float(np.max(np.abs(actual_velocity[:, :3]-rate))),
        "recovered_chart_resolution_m": chart_resolution,
        "current_velocity_extension": "co_moving_radial_P_equals_A",
        "full_historical_velocity_or_natural_metric_recovered": False}


def audit_interval(before_packet, after_packet):
    before, after = before_packet["runtime_state"], after_packet["runtime_state"]
    source = after_packet["source_packet"]
    if _canonical(before_packet["source_packet"]) != _canonical(source):
        raise ValueError("Independent volume interval changed its complete formation source")
    geometry = after["geometry"]
    for name in ("vertices_m", "faces", "column_mass_kg", "density_kg_m3",
                 "source_identity", "source_checkpoint_sha256"):
        if _canonical(before["geometry"][name]) != _canonical(geometry[name]):
            raise ValueError("Independent interval silently changed its immutable source reference: "+name)
    vertices, faces = _array(geometry["vertices_m"]), np.asarray(geometry["faces"], int)
    mass, rho = _array(geometry["column_mass_kg"]), float(geometry["density_kg_m3"])
    shell, thermal_parameters = (source["source"]["parameters"][key] for key in ("shell", "thermal"))
    bcohorts, acohorts = before["geometry"]["cohorts"], geometry["cohorts"]
    bmap, amap = ({int(c["id"]): c for c in rows} for rows in (bcohorts, acohorts))
    born = [c for c in acohorts if int(c["id"]) not in bmap]
    removed = []
    for c in bcohorts:
        survivor = amap.get(int(c["id"]))
        lower = c["q_lo"] if survivor is None else _high(survivor)
        if lower < _high(c):
            removed.append({**c, "q_lo": lower})
    M0, M1, Min, Mout = (mass_matrix(vertices, faces, mass, rho, rows)
                         for rows in (bcohorts, acohorts, born, removed))
    report, inputs = after["last_report"], after["last_report"]["audit_input"]
    if (_canonical(inputs["before_cohorts"]) != _canonical(bcohorts)
            or _canonical(inputs["after_cohorts"]) != _canonical(acohorts)):
        raise ValueError("Independent loading catalogue is not bound to the actual endpoint snapshots")
    q0, q1, v0, v1 = (_array(state[name]) for state, name in
        ((before, "displacement_m"), (after, "displacement_m"),
         (before, "velocity_m_s"), (after, "velocity_m_s")))
    dq = q1-q0
    _assert_close(inputs["before_displacement_m"], q0, "old displacement", rtol=0, atol=0)
    _assert_close(inputs["before_velocity_m_s"], v0, "old velocity", rtol=0, atol=0)
    _assert_close(inputs["increment_m"], dq, "accepted displacement increment", rtol=2e-12, atol=1e-15)
    traction = prescribed_traction(vertices, faces, mass, rho, acohorts,
        source["current_geometry"]["radius_km"], shell)
    _assert_close(traction, inputs["basal_traction_at_vertices_pa"], "prescribed basal traction", rtol=2e-10, atol=1e-8)
    D, Fext = basal_drag_and_force(vertices, faces, mass, rho, acohorts,
        inputs["basal_drag_pa_s_m"], traction[faces])
    dt = (report["after_time_myr"]-report["before_time_myr"])*365.25*86400*1e6
    endpoint_clock_binding(before, report["before_time_myr"])
    endpoint_clock_binding(after, report["after_time_myr"])
    if dt <= 0:
        raise ValueError("Independent accepted interval must advance the physical clock")
    tm0, ts0 = global_temperatures(before["thermal"]["global_state"], thermal_parameters)
    tm1, ts1 = global_temperatures(after["thermal"]["global_state"], thermal_parameters)
    column0 = conductive_temperatures(before["thermal"], thermal_parameters)
    column1 = conductive_temperatures(after["thermal"], thermal_parameters)
    layer_mass = _array(after["thermal"]["layer_mass_kg"])
    bm = {int(c["id"]): c for c in before["mechanics"]["state"]["cohorts"]}
    am = {int(c["id"]): c for c in after["mechanics"]["state"]["cohorts"]}
    average_force = np.zeros(len(q0))
    totals = {key: 0. for key in ("mechanical_work_j", "thermal_work_j", "viscous_dissipation_j", "elastic_energy_j")}
    worst = 0.
    damage_energy, before_energy, removed_energy = 0., 0., 0.
    for c in bcohorts:
        cid, face = int(c["id"]), int(c["face_index"])
        C = isotropic_matrix(shell["young_modulus_pa"], shell["poisson_ratio"],
            before["thermal"]["damage"][face], shell["residual_stiffness"])
        before_energy += coefficient_energy(vertices[faces[face]], mass[face], rho, c,
                                            bm[cid]["strain0"], bm[cid]["strain1"], C)
    for c in removed:
        cid, face = int(c["id"]), int(c["face_index"])
        C = isotropic_matrix(shell["young_modulus_pa"], shell["poisson_ratio"],
            before["thermal"]["damage"][face], shell["residual_stiffness"])
        removed_energy += coefficient_energy(vertices[faces[face]], mass[face], rho, c,
                                            bm[cid]["strain0"], bm[cid]["strain1"], C)
    _assert_close(before_energy, before["mechanics"]["state"]["elastic_energy_j"],
                  "independent prior accepted energy", rtol=2e-12, atol=1e3)
    for index, cohort in enumerate(acohorts):
        cid, face = int(cohort["id"]), int(cohort["face_index"])
        old = bm.get(cid)
        old0 = np.zeros((3, 6)) if old is None else _array(old["strain0"])
        old1 = np.zeros((3, 6)) if old is None else _array(old["strain1"])
        _assert_close(inputs["old0"][index], old0, "old constant strain coefficient", rtol=0, atol=0)
        _assert_close(inputs["old1"][index], old1, "old radial strain coefficient", rtol=0, atol=0)
        lo, hi = float(cohort["q_lo"]), _high(cohort)
        previous_T = (thermal_parameters["solidus_k"] if old is None else
            profile_mean(layer_mass[face], column0[face], ts0, tm0, lo, hi))
        temperature = profile_mean(layer_mass[face], column1[face], ts1, tm1, lo, hi)
        _assert_close(temperature, am[cid]["temperature_k"], "accepted cohort temperature", rtol=2e-12, atol=1e-8)
        thermal = np.zeros((3, 6)); thermal[:, :3] = shell["linear_expansion_per_k"]*(temperature-previous_T)
        _assert_close(inputs["thermal0"][index], thermal, "independent thermal driver", rtol=2e-7, atol=2e-13)
        viscosity = np.clip(shell["viscosity_reference_pa_s"]*np.exp(np.clip(
            shell["activation_energy_j_mol"]/8.314462618*(1/temperature-1/shell["viscosity_reference_temperature_k"]), -60, 60)),
            shell["viscosity_min_pa_s"], shell["viscosity_max_pa_s"])
        ratio = 0. if old is None else dt*shell["young_modulus_pa"]/viscosity
        _assert_close(ratio, inputs["maxwell_ratio"][index], "independent Maxwell rate", rtol=2e-10, atol=1e-15)
        damage = after["thermal"]["damage"][face]
        C = isotropic_matrix(shell["young_modulus_pa"], shell["poisson_ratio"], damage, shell["residual_stiffness"])
        if old is not None:
            old_C = isotropic_matrix(shell["young_modulus_pa"], shell["poisson_ratio"],
                before["thermal"]["damage"][face], shell["residual_stiffness"])
            damage_energy += coefficient_energy(vertices[faces[face]], mass[face], rho, cohort,
                                                old0, old1, C-old_C)
        _assert_close(C, inputs["elasticity_pa"][index], "constitutive tensor", rtol=2e-14, atol=1e-6)
        triangle = vertices[faces[face]]
        e1 = triangle[1]-triangle[0]; e1 /= np.linalg.norm(e1)
        normal = np.cross(triangle[1]-triangle[0], triangle[2]-triangle[0]); normal /= np.linalg.norm(normal)
        frame = np.column_stack((e1, np.cross(normal, e1), normal))
        _assert_close(frame, inputs["basis_xyz"][index], "cohort Cartesian frame", rtol=2e-12, atol=2e-12)
        dofs = (6*faces[face, :, None]+np.arange(6)).ravel()
        result = quadrature_cohort(triangle, frame, mass[face], rho, cohort, dq[dofs], old0, old1, thermal, ratio, C)
        np.add.at(average_force, dofs, result["average_force_n"])
        for key in totals:
            totals[key] += result[key]
        worst = max(worst, result["relative_balance_residual"])
        # Independently verify the accepted endpoint *coefficient* memory.
        N = SOURCE_ANGULAR_POINTS
        B0, B1 = independent_strain_operators(triangle, frame, N)
        inc0, inc1 = B0@dq[dofs], B1@dq[dofs]
        beta = 1. if ratio == 0 else -math.expm1(-ratio)/ratio
        expected0 = math.exp(-ratio)*old0+beta*(inc0-thermal)
        expected1 = math.exp(-ratio)*old1+beta*inc1
        _assert_close(am[cid]["strain0"], expected0, "accepted constant coefficient", rtol=2e-9, atol=2e-12)
        _assert_close(am[cid]["strain1"], expected1, "accepted radial coefficient", rtol=2e-9, atol=2e-12)
    for key in ("mechanical_work_j", "thermal_work_j", "viscous_dissipation_j"):
        _assert_close(totals[key], report[key], "quadrature "+key, rtol=2e-7, atol=1e3)
    _assert_close(totals["elastic_energy_j"], after["mechanics"]["state"]["elastic_energy_j"], "quadrature endpoint energy", rtol=2e-9, atol=1e3)
    elastic_residual = (totals["elastic_energy_j"]-before_energy-(totals["mechanical_work_j"]
        +totals["thermal_work_j"]-totals["viscous_dissipation_j"]+damage_energy-removed_energy))
    if abs(elastic_residual) > 2e-9*max(totals["elastic_energy_j"], before_energy, 1.):
        raise ValueError("Independent full elastic/thermal/viscous/damage/remelt interval balance failed")
    totals.update(before_elastic_energy_j=before_energy, damage_energy_change_j=damage_energy,
                  remelt_energy_removed_j=removed_energy, full_elastic_balance_residual_j=elastic_residual)
    balance = dynamic_balance(M0, M1, Min, Mout, D, Fext, average_force, q0, q1, v0, v1, dt, totals["mechanical_work_j"])
    for independent_name, reported_name in (("kinetic_birth_inflow_j", "birth_kinetic_energy_j"),
        ("kinetic_remelt_outflow_j", "remelt_kinetic_energy_j"), ("drag_dissipation_j", "basal_dissipation_j"),
        ("external_work_j", "external_work_j")):
        _assert_close(balance[independent_name], report[reported_name], independent_name, rtol=3e-8, atol=1e3)
    geometry_audit = geometry_binding(after["last_certificate"], vertices, faces, q0, q1,
        bcohorts+acohorts, mass, rho, report["before_time_myr"], report["after_time_myr"])
    return {"before_time_myr": report["before_time_myr"], "after_time_myr": report["after_time_myr"],
        "live_cohort_count": len(acohorts), "newborn_count": len(born), "removed_band_count": len(removed),
        "quadrature": totals, "maximum_cohort_relative_work_residual": worst,
        "discrete_dynamic_balance": balance, "geometry": geometry_audit,
        "continuous_trajectory_certified": False, "shared_heat_mechanical_closure_claimed": False,
        "old_native_C5_to_C1_admission_claimed": False, "point_1_completed": False}


def audit_directory(directory):
    directory = Path(directory)
    snapshots = []
    source_bindings = []
    for path in sorted(directory.glob("*.json")):
        raw = json.loads(path.read_text())
        if raw.get("format") == "genesis-volume-continuation-checkpoint-0.1":
            body = {key: value for key, value in raw.items() if key != "checkpoint_sha256"}
            if hashlib.sha256(_canonical(body)).hexdigest() != raw["checkpoint_sha256"]:
                raise ValueError("Independent volume checkpoint seal mismatch")
            source = raw["source_packet"]
            source_bytes = base64.b64decode(source["source"]["checkpoint_base64"], validate=True)
            if hashlib.sha256(source_bytes).hexdigest() != raw["source_checkpoint_sha256"]:
                raise ValueError("Independent formation source bytes changed")
            source_bindings.append({"name": path.name, **source_dto_binding(raw)})
            count = raw["runtime_state"]["mechanics"]["state"]["accepted_steps"]
            snapshots.append((count, path, raw))
    if not snapshots:
        raise ValueError("No actual volume checkpoints found in the control directory")
    grouped = {}
    for count, path, raw in snapshots:
        grouped.setdefault(count, []).append((path, raw))
    if sorted(grouped) != list(range(max(grouped)+1)):
        raise ValueError("Independent audit needs every accepted endpoint including initial")
    restart_pairs = []
    for count, rows in grouped.items():
        for path, raw in rows[1:]:
            if _canonical(rows[0][1]) != _canonical(raw):
                raise ValueError("Direct/restart actual volume checkpoints differ")
            restart_pairs.append({"accepted_steps": count, "first": rows[0][0].name,
                                  "second": path.name, "complete_json_equal": True})
    intervals = [audit_interval(grouped[count-1][0][1], grouped[count][0][1])
                 for count in range(1, max(grouped)+1)]
    if not intervals:
        raise ValueError("Independent audit requires a genuinely advanced detailed volume interval")
    report = {"format": "genesis-volume-continuation-independent-audit-0.1", "status": "PASS",
        "producer_imports": False, "quadrature": "Duffy4x4_angular_Gauss32_massq_Gauss32_adaptive_time",
        "input_files": [{"name": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                        for _, path, _ in snapshots],
        "literal_formation_source_bindings": source_bindings,
        "initial_source_audit": audit_initial(grouped[0][0][1]),
        "intervals": intervals, "restart_pairs": restart_pairs,
        "original_point_1_completed": False, "old_C5_to_C1_admission_claimed": False}
    return {**report, "report_sha256": hashlib.sha256(_canonical(report)).hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = audit_directory(arguments.directory)
    encoded = json.dumps(report, sort_keys=True, indent=2, allow_nan=False)+"\n"
    if arguments.output is not None:
        arguments.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
