"""Producer-free finite material seed and compact embedded-basis oracle.

The preserved Genesis packet is the sole source of material geometry, pose,
elastic memory and constitutive parameters.  New geometry and enrichment are
derived with NumPy; immutable older independent oracles provide only already
audited material-field and original-volume mathematics.  No planetary replay
or mechanics producer is imported or called here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from numpy.polynomial import Polynomial

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis.genesis_volume_nucleation_validation import independent_audit as prior
from analysis.genesis_volume_continuation_validation import independent_audit as volume

PARENT_RAW_SHA256 = prior.PARENT_RAW_SHA256
PARENT_CHECKPOINT_SHA256 = prior.PARENT_CHECKPOINT_SHA256
FORMATION_SHA256 = prior.FORMATION_SHA256


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError("Independent finite seed audit: "+message)


def array(value, shape=None):
    raw = np.asarray(value)
    require(raw.dtype.kind in "fiu" and (shape is None or raw.shape == shape)
            and np.isfinite(raw).all(), "invalid finite numeric array")
    return np.asarray(raw, float)


def close(actual, expected, name, *, rtol=3e-9, atol=1e-10):
    a, b = array(actual), array(expected)
    require(a.shape == b.shape, name+" dimensions differ")
    scale = max(float(np.max(abs(b), initial=0.)), np.finfo(float).tiny)
    error = float(np.max(abs(a-b), initial=0.))
    require(error <= atol+rtol*scale, name+" differs")
    return error/max(scale, atol, np.finfo(float).tiny)


def fixed(packet, expected, name):
    for key, value in expected.items():
        require(key in packet and canonical(packet[key]) == canonical(value),
                name+" "+key+" differs")


def seal(packet, key):
    require(type(packet) is dict and packet.get(key) == digest(
        {k: v for k, v in packet.items() if k != key}), key+" seal differs")


def plain(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [plain(v) for v in value]
    return value


def cohort_chart(parent, cohort_id, normal_xyz, offset_m):
    geometry = parent["runtime_state"]["geometry"]
    matches = [(i, c) for i, c in enumerate(geometry["cohorts"]) if c["id"] == cohort_id]
    require(type(cohort_id) is int and len(matches) == 1,
            "seed lost single original live cohort identity")
    index, cohort = matches[0]
    face = int(cohort["face_index"])
    vertices = np.asarray(geometry["faces"][face], int)
    X = array(geometry["vertices_m"])[vertices]
    pose = array(parent["runtime_state"]["displacement_m"]).reshape(-1, 6)[vertices]
    chart = prior.plane_coefficients(X, pose[:, :3], pose[:, 3:], normal_xyz, offset_m)
    chart.update(cohort=cohort, cohort_index=index, face=face,
        trace_dofs=(6*vertices[:, None]+np.arange(6)).ravel(),
        parent_ndof=6*len(geometry["vertices_m"]),
        coefficient=6*geometry["column_mass_kg"][face]/
            (geometry["density_kg_m3"]*np.linalg.det(X.T)))
    prior.whole_cohort_chart_positive(chart, cohort["inner_scale"], cohort["outer_scale"])
    return chart


def real_event_roots(coefficients):
    """Companion eigenvalues, independent of producer derivative isolation."""
    c = np.asarray(coefficients, float)
    while len(c) > 1 and c[-1] == 0: c = c[:-1]
    if not np.any(c) or len(c) == 1: return []
    c = c/np.max(abs(c))
    roots = Polynomial(c).roots()
    result = []
    for root in roots:
        if abs(root.imag) <= 1e-7*max(1., abs(root.real)) and -1 < root.real < 1:
            x = float(root.real)
            # Polish simple roots in scalar arithmetic. Multiple roots are
            # retained if the normalized polynomial residual is negligible.
            p, dp = Polynomial(c), Polynomial(c).deriv()
            for _ in range(8):
                derivative = dp(x)
                if derivative == 0: break
                step = p(x)/derivative
                if abs(step) > .1 or not -1 < x-step < 1: break
                x -= step
            require(abs(p(x)) <= 2e-7*np.sum(abs(c)), "sphere event root unresolved")
            result.append(x)
    return sorted(result)


def event_breaks(chart, lo, hi, edges, center, radius):
    """Rational endpoints and plane-line distance identify all sphere events."""
    mid, half = lo+(hi-lo)/2, (hi-lo)/2
    V = chart["Y"].astype(np.longdouble)
    W = chart["Z"].astype(np.longdouble)-np.asarray(center, np.longdouble)
    Y = [[Polynomial([mid*V[i, k]+W[i, k], half*V[i, k]])
          for k in range(3)] for i in range(3)]
    n = chart["n"]
    dot = lambda a, b: sum((x*y for x, y in zip(a, b)), Polynomial([0.]))
    delta = np.longdouble(chart["d"])-np.asarray(center, np.longdouble)@np.asarray(n, np.longdouble)
    scalar = [sum((Y[i][k]*n[k] for k in range(3)), Polynomial([0.]))-delta for i in range(3)]
    events = []
    for i, j in edges:
        if i == j:
            B, D = Y[i], Polynomial([1.])
        else:
            D = scalar[i]-scalar[j]
            B = [scalar[i]*Y[j][k]-scalar[j]*Y[i][k] for k in range(3)]
        events.append(dot(B, B)-radius**2*D*D)
    E = [[Y[i][k]-Y[0][k] for k in range(3)] for i in (1, 2)]
    normal = [E[0][1]*E[1][2]-E[0][2]*E[1][1],
              E[0][2]*E[1][0]-E[0][0]*E[1][2],
              E[0][0]*E[1][1]-E[0][1]*E[1][0]]
    normal_component = sum((normal[k]*n[k] for k in range(3)), Polynomial([0.]))
    g = dot(normal, Y[0])-delta*normal_component
    tangent = [normal[k]-normal_component*n[k] for k in range(3)]
    events.append(g*g-(radius**2-delta*delta)*dot(tangent, tangent))
    roots = sorted(lo+half*(z+1) for p in events for z in real_event_roots(p.coef))
    # Distinct companion roots can land within one ULP of the same event.
    unique = []
    for root in roots:
        if lo < root < hi and (not unique or root-unique[-1] >
                16*max(abs(np.spacing(root)), abs(np.spacing(unique[-1])))):
            unique.append(root)
    return [lo]+unique+[hi]


def clipped_segment(chart, edges, lam, center, radius):
    N0, D0 = prior.edge_barycentric(chart, edges[0], lam)
    N1, D1 = prior.edge_barycentric(chart, edges[1], lam)
    Y = lam*chart["Y"]+chart["Z"]
    origin = np.asarray(N0@Y-center, np.longdouble)
    direction = np.asarray((N1-N0)@Y, np.longdouble)
    d2 = direction@direction
    require(d2 > 0, "sphere section segment collapsed")
    foot = -(origin@direction)/d2
    perpendicular = origin+foot*direction
    margin = np.longdouble(radius)**2-perpendicular@perpendicular
    if margin <= 0: return None
    extent = np.sqrt(margin/d2)
    lower, upper = max(0., foot-extent), min(1., foot+extent)
    if upper <= lower: return None
    return N0, N1, D0, D1, float(lower), float(upper)


def polar_triangle(chart, lam, center, radius):
    normal = chart["n"]
    axis = np.eye(3)[np.argmin(abs(normal))]
    first = np.cross(normal, axis); first /= np.linalg.norm(first)
    second = np.cross(normal, first)
    Y = lam*chart["Y"]+chart["Z"]
    Q = np.column_stack(((Y-center)@first, (Y-center)@second))
    inv = np.linalg.inv(np.column_stack((Q[1]-Q[0], Q[2]-Q[0])))
    ccw = Q if np.linalg.det(np.column_stack((Q[1]-Q[0], Q[2]-Q[0]))) > 0 else Q[[0, 2, 1]]
    angles = [0., 2*math.pi]
    cross2 = lambda a, b: a[0]*b[1]-a[1]*b[0]
    constraints = []
    for point, next_point in zip(ccw, np.roll(ccw, -1, axis=0)):
        edge = next_point-point
        constraints.append((edge, cross2(edge, -point)))
        for v in (point, edge, -edge):
            if np.linalg.norm(v): angles.append(math.atan2(v[1], v[0])%(2*math.pi))
        polynomial = [point@point-radius**2, 2*(point@edge), edge@edge]
        for root in Polynomial(polynomial).roots():
            if abs(root.imag) < 1e-12 and 0 <= root.real <= 1:
                v = point+root.real*edge
                if np.linalg.norm(v): angles.append(math.atan2(v[1], v[0])%(2*math.pi))

    def clip(theta):
        direction = np.array([math.cos(theta), math.sin(theta)])
        lower, upper = 0., radius
        for edge, constant in constraints:
            coefficient = cross2(edge, direction)
            if coefficient > 0: lower = max(lower, -constant/coefficient)
            elif coefficient < 0: upper = min(upper, -constant/coefficient)
            elif constant < 0: return None
        return (direction, lower, upper) if upper > lower else None

    return sorted(set(angles)), clip, Q, inv


def finite_quadrature(parent, normal_xyz, offset_m, center, radius, *, cohort_id, order=8):
    chart = cohort_chart(parent, cohort_id, normal_xyz, offset_m)
    center, radius = array(center, (3,)), float(radius)
    require(np.isfinite(radius) and radius > 0, "finite seed radius must be positive")
    require(abs(chart["n"]@center-chart["d"]) <= 64*np.finfo(float).eps*
        max(np.sum(abs(chart["n"]*center))+abs(chart["d"]), 1.), "seed center leaves plane")
    c = chart["cohort"]
    segments, triangles = prior.section_partition(chart, c["inner_scale"], c["outer_scale"])
    g, w = np.polynomial.legendre.leggauss(order); g, w = (g+1)/2, w/2
    result, records, kinds, bounds = [], [], [], []

    def add(N, lam, dNs, dNl, weight, patch, *, triangle=False):
        Y = lam*chart["Y"]+chart["Z"]
        if triangle:
            dc, dr = Y[1:]-Y[0], lam*(chart["X"][1:]-chart["X"][0])
        else:
            dc = np.array([dNs@Y, dNl@Y+N@chart["Y"]])
            dr = np.array([lam*(dNs@chart["X"]), N@chart["X"]+lam*(dNl@chart["X"])])
        cross_current, cross_reference = np.cross(*dc), np.cross(*dr)
        jc, jr = np.linalg.norm(cross_current), np.linalg.norm(cross_reference)
        require(jc > 0 and jr > 0, "finite seed metric collapsed")
        orientation = 1 if cross_current@chart["n"] > 0 else -1
        Jc, Jr = prior.current_chart_jacobians(chart, N, lam)
        T = prior.parent_trace(N, lam)
        result.append(dict(point_patch_indices=patch, cohort_indices=chart["cohort_index"],
            cohort_ids=cohort_id, face_indices=chart["face"], barycentric=N, radial_scale=lam,
            mass_coordinate_q=-np.expm1(3*np.log(lam))/chart["coefficient"],
            current_position_m=N@Y, reference_position_m=lam*(N@chart["X"]),
            current_differentials_m=dc, reference_differentials_m=dr,
            current_volume_jacobian_m=Jc, reference_volume_jacobian_m=Jr,
            current_volume_determinant_m3=np.linalg.det(Jc), reference_volume_determinant_m3=np.linalg.det(Jr),
            current_to_reference_volume_ratio=np.linalg.det(Jc)/np.linalg.det(Jr),
            current_normal_xyz=orientation*cross_current/jc,
            reference_normal_xyz=orientation*cross_reference/jr,
            current_area_weights_m2=weight*jc, reference_area_weights_m2=weight*jr,
            parent_trace=T, paired_jump_trace=np.concatenate((-T, T), axis=1),
            trace_dofs=chart["trace_dofs"]))

    for lo, hi, edges in segments:
        breaks = event_breaks(chart, lo, hi, edges, center, radius)
        for a, b in zip(breaks, breaks[1:]):
            if clipped_segment(chart, edges, (a+b)/2, center, radius) is None: continue
            patch = len(records); records.append(["lambda_segment", plain(edges), [a, b]])
            bounds.append((a, b)); kinds.append("sphere_clipped_lambda_segment")
            for t, wt in zip(g, w):
                lam = a+(b-a)*(1-math.cos(math.pi*t))/2
                derivative = (b-a)*math.pi*math.sin(math.pi*t)/2
                clipped = clipped_segment(chart, edges, lam, center, radius)
                require(clipped is not None, "finite seed analytic support is unresolved")
                N0, N1, D0, D1, lower, upper = clipped
                for s, ws in zip(g, w):
                    v = lower+(upper-lower)*s
                    add((1-v)*N0+v*N1, lam, N1-N0, (1-v)*D0+v*D1,
                        derivative*wt*(upper-lower)*ws, patch)
    for lam in triangles:
        angles, clip, Q, inv = polar_triangle(chart, lam, center, radius)
        for a, b in zip(angles, angles[1:]):
            if clip((a+b)/2) is None: continue
            patch = len(records); records.append(["constant_lambda_triangle", lam, [a, b]])
            bounds.append((lam, lam)); kinds.append("sphere_clipped_constant_lambda_triangle")
            for t, wt in zip(g, w):
                direction, lower, upper = clip(a+(b-a)*t)
                for s, ws in zip(g, w):
                    r = lower+(upper-lower)*s
                    uv = inv@(r*direction-Q[0]); N = np.r_[1-uv.sum(), uv]
                    dc = (lam*chart["Y"]+chart["Z"])[1:]-(lam*chart["Y"]+chart["Z"])[0]
                    jc = np.linalg.norm(np.cross(*dc))
                    add(N, lam, None, None, (b-a)*wt*(upper-lower)*ws*r/jc, patch, triangle=True)
    require(result, "finite seed has no positive relative-interior support")
    data = {name: np.asarray([row[name] for row in result]) for name in result[0]}
    data.update(clipping_records=records, patch_kinds=kinds, patch_lambda_bounds=np.asarray(bounds),
        parent_ndof=chart["parent_ndof"], owner_ids=[c["owner_id"]]*len(result),
        material_ids=[c["material_id"]]*len(result), selected_cohort_ids=[cohort_id],
        selected_cohort_indices=[chart["cohort_index"]], patch_cohort_ids=[cohort_id]*len(records),
        quadrature_order=order)
    return data


def inverse_three(matrix):
    """Explicit adjugate, including extended precision unsupported by linalg."""
    a = np.asarray(matrix)
    cofactors = np.stack((np.cross(a[..., :, 1], a[..., :, 2]),
        np.cross(a[..., :, 2], a[..., :, 0]), np.cross(a[..., :, 0], a[..., :, 1])), axis=-2)
    determinant = np.sum(a[..., :, 0]*cofactors[..., 0, :], axis=-1)
    require(np.isfinite(determinant).all() and np.all(determinant != 0), "singular compact material chart")
    return cofactors/determinant[..., None, None], determinant


def compact_chart_jacobians(chart, barycentric, radial_scale):
    dtype = np.asarray(barycentric).dtype
    X, Y, Z = (np.asarray(chart[k], dtype) for k in ("X", "Y", "Z"))
    N, lam = barycentric, radial_scale
    return (np.column_stack((lam*(Y[1]-Y[0])+(Z[1]-Z[0]),
        lam*(Y[2]-Y[0])+(Z[2]-Z[0]), N@Y)),
        np.column_stack((lam*(X[1]-X[0]), lam*(X[2]-X[0]), N@X)))


def compact_fields(chart, barycentric, radial_scale, center, radius, *, birth_displacement=None):
    """Full reference derivative of the frozen compact A/P enrichment."""
    dtype = np.longdouble if np.asarray(barycentric).dtype == np.dtype(np.longdouble) else float
    N, lam = (np.asarray(array(v), dtype) if dtype is float else np.asarray(v, dtype)
        for v in (barycentric, radial_scale))
    require(np.isfinite(N).all() and np.isfinite(lam).all(), "invalid compact material coordinates")
    require(N.shape == (len(lam), 3), "compact material coordinates differ")
    center = np.asarray(array(center, (3,)), dtype); radius = dtype(radius)
    inner, outer = (dtype(chart["cohort"][k]) for k in ("inner_scale", "outer_scale"))
    depth = outer-inner
    t = (lam-inner)/depth
    dN = np.array([[-1., -1., 0.], [1., 0., 0.], [0., 1., 0.]], dtype=dtype)
    Y, Z = np.asarray(chart["Y"], dtype), np.asarray(chart["Z"], dtype)
    # Keep metre-scale birth offsets separate from planetary world coordinates.
    # Input chart coefficients are still the original represented float values.
    distance = (np.einsum("pi,pij->pj", N, lam[:, None, None]*Y+(Z-center))
        if birth_displacement is None else np.asarray(birth_displacement, dtype))
    positions = center+distance
    h = np.maximum(1-np.einsum("pi,pi->p", distance, distance)/radius**2, 0.)
    angular = 27*np.prod(N, axis=1)
    radial = 4*t*(1-t)
    weight = h*h*angular*radial
    Jc = np.array([compact_chart_jacobians(chart, n, l)[0] for n, l in zip(N, lam)])
    Jr = np.array([compact_chart_jacobians(chart, n, l)[1] for n, l in zip(N, lam)])
    grad_h_chart = -2*np.einsum("pi,pij->pj", distance, Jc)/radius**2
    grad_h_chart[h == 0] = 0
    grad_angular = np.array([27*(n[1]*n[2]*dN[0]+n[0]*n[2]*dN[1]+n[0]*n[1]*dN[2]) for n in N])
    grad_radial = np.zeros_like(N); grad_radial[:, 2] = 4*(1-2*t)/depth
    grad_weight_chart = ((2*h*angular*radial)[:, None]*grad_h_chart+
        (h*h*radial)[:, None]*grad_angular+(h*h*angular)[:, None]*grad_radial)
    inverses, _ = inverse_three(Jr)
    grad_weight_ref = np.einsum("pi,pij->pj", grad_weight_chart, inverses)
    phi = distance@np.asarray(chart["n"], dtype)+(center@np.asarray(chart["n"], dtype)-dtype(chart["d"]))
    H = np.where(phi >= 0, .5, -.5)
    trace = np.zeros((len(N), 3, 18), dtype=dtype)
    for vertex in range(3):
        trace[:, :, 6*vertex:6*vertex+3] = N[:, vertex, None, None]*np.eye(3, dtype=dtype)
        trace[:, :, 6*vertex+3:6*vertex+6] = ((lam-1)*N[:, vertex])[:, None, None]*np.eye(3, dtype=dtype)
    gradients = np.zeros((len(N), 18, 3, 3), dtype=dtype)
    for p, (n, l) in enumerate(zip(N, lam)):
        inverse = inverses[p]
        for vertex in range(3):
            grad_N = dN[vertex]@inverse
            grad_P = ((l-1)*dN[vertex]+[0., 0., n[vertex]])@inverse
            for axis in range(3):
                gradients[p, 6*vertex+axis, axis] = grad_N
                gradients[p, 6*vertex+3+axis, axis] = grad_P
    enriched_gradient = H[:, None, None, None]*(weight[:, None, None, None]*gradients+
        np.einsum("pai,pj->piaj", trace, grad_weight_ref))
    return dict(position_m=positions, support_weight=weight, support_gradient_reference_m_inv=grad_weight_ref,
        heaviside=H, parent_trace=trace, enrichment_trace=H[:, None, None]*weight[:, None, None]*trace,
        surface_enrichment_trace=weight[:, None, None]*trace,
        parent_displacement_gradient=gradients, enrichment_displacement_gradient=enriched_gradient,
        current_volume_jacobian_m=Jc, reference_volume_jacobian_m=Jr)


def birth_inverse(chart, position):
    """Newton inversion followed by a separate cubic uniqueness witness."""
    X, Y, Z = chart["X"], chart["Y"], chart["Z"]
    homogeneous = np.linalg.solve(X.T, position-Z.mean(axis=0))
    lam = float(homogeneous.sum()); N = homogeneous/lam
    parameter = np.r_[N[1:], lam]
    for _ in range(12):
        N = np.r_[1-parameter[:2].sum(), parameter[:2]]; lam = parameter[2]
        current = N@(lam*Y+Z)
        J, _ = prior.current_chart_jacobians(chart, N, lam)
        correction = np.linalg.solve(J, current-position)
        parameter -= correction
        if np.linalg.norm(correction) <= 4e-15*max(np.linalg.norm(parameter), 1.): break
    N = np.r_[1-parameter[:2].sum(), parameter[:2]]; lam = float(parameter[2])
    c = chart["cohort"]
    tolerance = 256*np.finfo(float).eps
    if np.min(N) < -tolerance or np.max(N) > 1+tolerance or not (
            c["inner_scale"]-tolerance <= lam <= c["outer_scale"]+tolerance): return None
    require(np.linalg.norm(N@(lam*Y+Z)-position) <= 1024*np.finfo(float).eps*
        max(np.linalg.norm(position), np.linalg.norm(Y), 1.), "frozen birth inverse residual differs")
    # Uniqueness is checked by all roots of the radial slice plane. Newton
    # does not silently select one inverse image from an overlapping chart.
    eY, eZ = Y[1:]-Y[0], Z[1:]-Z[0]
    cross0 = np.cross(*eZ)
    cross1 = np.cross(eZ[0], eY[1])+np.cross(eY[0], eZ[1])
    cross2 = np.cross(*eY)
    delta = position-Z[0]
    polynomial = [cross0@delta, cross1@delta-cross0@Y[0],
                  cross2@delta-cross1@Y[0], -cross2@Y[0]]
    roots = Polynomial(np.asarray(polynomial)/max(abs(np.asarray(polynomial)))).roots()
    candidates = []
    for root in roots:
        if abs(root.imag) > 2e-12 or not c["inner_scale"]-tolerance <= root.real <= c["outer_scale"]+tolerance:
            continue
        depth = float(root.real); vertices = depth*Y+Z
        E = (vertices[1:]-vertices[0]).T
        uv = np.linalg.solve(E.T@E, E.T@(position-vertices[0]))
        barycentric = np.r_[1-uv.sum(), uv]
        if np.min(barycentric) >= -2e-12 and np.max(barycentric) <= 1+2e-12:
            candidates.append(depth)
    require(len(candidates) == 1, "frozen birth chart lacks a unique in-cohort inverse")
    return np.clip(N, 0., 1.)/np.clip(N, 0., 1.).sum(), min(max(lam, c["inner_scale"]), c["outer_scale"])


def precise_birth_inverse(chart, center, displacement):
    """Scalar radial-plane refinement after the independent cubic witness.

    The producer refines three chart parameters together.  Here the radial
    plane equation is solved first, then its two barycentric coordinates are
    obtained from an explicit Gram solve.  No material coefficients change.
    """
    ld = np.longdouble
    center, displacement = np.asarray(center, ld), np.asarray(displacement, ld)
    witness = birth_inverse(chart, np.asarray(center+displacement, float))
    if witness is None: return None
    Y, sourceZ = np.asarray(chart["Y"], ld), np.asarray(chart["Z"], ld)
    Z = sourceZ-center
    edgesY, edgesZ = Y[1:]-Y[0], sourceZ[1:]-sourceZ[0]
    lam = ld(witness[1])
    for _ in range(16):
        edges = lam*edgesY+edgesZ
        normal = np.cross(edges[0], edges[1])
        normal_derivative = np.cross(edgesY[0], edges[1])+np.cross(edges[0], edgesY[1])
        relative = displacement-(lam*Y[0]+Z[0])
        derivative = normal_derivative@relative-normal@Y[0]
        require(derivative != 0, "compact radial inverse has singular plane derivative")
        correction = (normal@relative)/derivative
        lam -= correction
        if abs(correction) <= 8*np.finfo(ld).eps*max(abs(lam), ld(1)): break
    edges = lam*edgesY+edgesZ
    relative = displacement-(lam*Y[0]+Z[0])
    g00, g01, g11 = edges[0]@edges[0], edges[0]@edges[1], edges[1]@edges[1]
    determinant = g00*g11-g01*g01
    require(determinant > 0, "compact radial inverse has degenerate triangle")
    r0, r1 = edges[0]@relative, edges[1]@relative
    uv = np.array(((g11*r0-g01*r1)/determinant, (g00*r1-g01*r0)/determinant), dtype=ld)
    N = np.r_[ld(1)-uv.sum(), uv]
    inner, outer = (ld(chart["cohort"][k]) for k in ("inner_scale", "outer_scale"))
    tolerance = ld(256*np.finfo(float).eps)
    if N.min() < -tolerance or N.max() > 1+tolerance or not inner-tolerance <= lam <= outer+tolerance:
        return None
    N = np.clip(N, 0, 1); N /= N.sum(); lam = min(max(lam, inner), outer)
    material_displacement = N@(lam*Y+Z)
    require(np.sqrt(np.sum((material_displacement-displacement)**2)) <=
        1024*np.finfo(ld).eps*max(np.max(abs(Y)), np.max(abs(center)), ld(1)),
        "precise frozen birth inverse residual differs")
    return N, lam, material_displacement


def compact_ball_quadrature(chart, center, radius, *, order=8):
    """Direct small-ball integration; never a mask on parent volume nodes."""
    ld = np.longdouble
    center, radius = np.asarray(array(center, (3,)), ld), ld(radius)
    normal = np.asarray(chart["n"], ld)
    axis = np.eye(3, dtype=ld)[np.argmin(abs(normal))]
    first = axis-normal*(axis@normal); first /= np.sqrt(first@first)
    second = np.cross(normal, first)
    g, w = np.polynomial.legendre.leggauss(order); g, w = (g+1)/2, w/2
    angles, angleweights = np.polynomial.legendre.leggauss(2*order)
    angles, angleweights = math.pi*(angles+1), math.pi*angleweights
    records = []
    for sign in (-1., 1.):
        for zt, wz in zip(g, w):
            z = sign*radius*zt; disk_squared = radius**2-z**2
            for s, ws in zip(g, w):
                rho = np.sqrt(disk_squared)*s
                for theta, wt in zip(angles, angleweights):
                    displacement = z*normal+rho*(ld(math.cos(theta))*first+ld(math.sin(theta))*second)
                    inverse = precise_birth_inverse(chart, center, displacement)
                    if inverse is None: continue
                    N, lam, material_displacement = inverse
                    Jb, Jr = compact_chart_jacobians(chart, N, lam)
                    inner, outer = (ld(chart["cohort"][k]) for k in ("inner_scale", "outer_scale"))
                    t = (lam-inner)/(outer-inner)
                    if np.prod(N) <= 0 or not 0 < t < 1: continue
                    born = radius*wz*disk_squared*s*ws*wt
                    reference = born*inverse_three(Jr)[1]/inverse_three(Jb)[1]
                    phi = material_displacement@normal+(center@normal-ld(chart["d"]))
                    records.append((N, lam, reference, born, .5 if phi >= 0 else -.5, material_displacement))
    require(records and {r[4] for r in records} == {-.5, .5},
            "compact seed lacks both positive-volume sides")
    return dict(barycentric=np.array([r[0] for r in records]), radial_scale=np.array([r[1] for r in records]),
        reference_volume_weights_m3=np.array([r[2] for r in records]),
        birth_volume_weights_m3=np.array([r[3] for r in records]), side=np.array([r[4] for r in records]),
        _birth_displacement_m=np.array([r[5] for r in records]))


def cohort_original_terms(parent, cohort_id):
    geometry = parent["runtime_state"]["geometry"]
    c = next(c for c in geometry["cohorts"] if c["id"] == cohort_id)
    f = int(c["face_index"]); vertices = np.asarray(geometry["faces"][f], int)
    X = array(geometry["vertices_m"])[vertices]
    Q = prior.source_frames(parent)[f]
    shell = parent["source_packet"]["source"]["parameters"]["shell"]
    C = prior.isotropic_matrix(shell["young_modulus_pa"], shell["poisson_ratio"],
        parent["runtime_state"]["fault_history"]["damage"][f], shell["residual_stiffness"])
    memory = next(m for m in parent["runtime_state"]["mechanics"]["state"]["cohorts"] if m["id"] == cohort_id)
    e0, e1 = array(memory["strain0"]), array(memory["strain1"])
    q, wr = volume.radial_rule(c["q_lo"], c["live_q_hi"])
    lam = prior.conical_scales(X, geometry["column_mass_kg"][f], geometry["density_kg_m3"], q)[0]
    N = prior.SOURCE_ANGULAR_POINTS
    b0, b1 = volume.independent_strain_operators(X, Q, N)
    B = b0[:, None]+b1[:, None]/lam[None, :, None, None]
    e = e0[:, None]+e1[:, None]/lam[None, :, None]
    stress = e@C.T
    mass = geometry["column_mass_kg"][f]*(c["live_q_hi"]-c["q_lo"])
    size = mass/geometry["density_kg_m3"]
    weights = wr[None, :]/3
    K = size*np.einsum("ar,arbi,bc,arcj->ij", np.broadcast_to(weights, B.shape[:2]), B, C, B)
    F = size*np.einsum("ar,arbi,arb->i", np.broadcast_to(weights, B.shape[:2]), B, stress)
    energy = .5*size*np.einsum("ar,arb,arb->", np.broadcast_to(weights, B.shape[:2]), e, stress)
    Mglobal = volume.mass_matrix(geometry["vertices_m"], geometry["faces"], geometry["column_mass_kg"],
        geometry["density_kg_m3"], [c])
    dofs = (6*vertices[:, None]+np.arange(6)).ravel()
    return dict(stiffness=K, mass=Mglobal[np.ix_(dofs, dofs)], force_n=F, energy_j=float(energy),
        dofs=dofs, basis_xyz=Q, elasticity_pa=C, memory0=e0, memory1=e1,
        volume_m3=size, mass_kg=mass)


def original_parent_terms(parent):
    geometry = parent["runtime_state"]["geometry"]
    n = 6*len(geometry["vertices_m"])
    K, F, energy = np.zeros((n, n)), np.zeros(n), 0.
    terms = {}
    for c in geometry["cohorts"]:
        local = cohort_original_terms(parent, c["id"]); dofs = local["dofs"]
        K[np.ix_(dofs, dofs)] += local["stiffness"]
        np.add.at(F, dofs, local["force_n"]); energy += local["energy_j"]
        terms[c["id"]] = local
    M = volume.mass_matrix(geometry["vertices_m"], geometry["faces"], geometry["column_mass_kg"],
        geometry["density_kg_m3"], geometry["cohorts"])
    runtime = parent["runtime_state"]
    inputs = runtime["last_report"]["audit_input"]
    traction = array(inputs["basal_traction_at_vertices_pa"])[np.asarray(geometry["faces"], int)]
    D, external = volume.basal_drag_and_force(geometry["vertices_m"], geometry["faces"],
        geometry["column_mass_kg"], geometry["density_kg_m3"], geometry["cohorts"],
        inputs["basal_drag_pa_s_m"], traction)
    return dict(stiffness=K, mass=M, basal_drag=D, external_force_n=external,
        saved_endpoint_elastic_force_n=F, saved_endpoint_elastic_energy_j=energy,
        cohort_terms=terms, parent_ndof=n)


def compact_bulk_terms(parent, normal_xyz, offset_m, center, radius, *, cohort_id, order=8):
    chart = cohort_chart(parent, cohort_id, normal_xyz, offset_m)
    local = cohort_original_terms(parent, cohort_id)
    points = compact_ball_quadrature(chart, center, radius, order=order)
    N, lam, weights = points["barycentric"], points["radial_scale"], points["reference_volume_weights_m3"]
    fields = compact_fields(chart, N, lam, center, radius,
        birth_displacement=points.pop("_birth_displacement_m"))
    # Material side belongs to the literal birth plane; grid hemispheres
    # are only integration subdivisions.
    H = points["side"]
    gradients = fields["enrichment_displacement_gradient"]*(H/fields["heaviside"])[:, None, None, None]
    Q, C = local["basis_xyz"], local["elasticity_pa"]
    B = np.swapaxes(volume.engineering_gradient(fields["parent_displacement_gradient"], Q), 1, 2)
    G = np.swapaxes(volume.engineering_gradient(gradients, Q), 1, 2)
    T = fields["parent_trace"]; E = H[:, None, None]*fields["surface_enrichment_trace"]
    interpolation = N@np.linalg.inv(prior.SOURCE_ANGULAR_POINTS)
    elastic = interpolation@local["memory0"]+(interpolation@local["memory1"])/lam[:, None]
    stress = elastic@C.T
    Kqa = np.einsum("pai,ab,pbj,p->ij", B, C, G, weights)
    Kaa = np.einsum("pai,ab,pbj,p->ij", G, C, G, weights)
    rho = parent["runtime_state"]["geometry"]["density_kg_m3"]
    Mqa = rho*np.einsum("pai,paj,p->ij", T, E, weights)
    Maa = rho*np.einsum("pai,paj,p->ij", E, E, weights)
    force = np.einsum("pai,pa,p->i", G, stress, weights)
    witness = np.einsum("pai,ab,pbj,p->ij", B, C, B, weights)
    require(np.trace(Kaa) > 0 and np.trace(Maa) > 0, "compact enriched operator is empty")
    result = dict(**points, reference_position_m=lam[:, None]*(N@chart["X"]),
        birth_position_m=fields["position_m"], bubble=fields["support_weight"],
        bubble_gradient_reference_m_inverse=fields["support_gradient_reference_m_inv"],
        parent_trace=T, enrichment_trace=E, parent_strain_operator=B, enrichment_strain_operator=G,
        point_elastic_strain=elastic, point_stress_pa=stress,
        parent_stiffness=local["stiffness"], parent_mass=local["mass"], parent_force_n=local["force_n"],
        parent_energy_j=local["energy_j"], stiffness_cross=Kqa, stiffness_enriched=Kaa,
        mass_cross=Mqa, mass_enriched=Maa, enrichment_force_n=force,
        support_parent_stiffness_witness=witness, basis_xyz=Q, elasticity_pa=C,
        memory0=local["memory0"], memory1=local["memory1"], quadrature_order=order,
        seed_center_world_m=np.asarray(center), seed_radius_m=radius)
    # Archive-facing arrays stay float64; extended precision is needed only
    # before signed integration of the weak, nonzero odd mass coupling.
    return {key: np.asarray(value, float) if isinstance(value, np.ndarray) else value
        for key, value in result.items()}


def dense_csr(raw, name):
    require(type(raw) is dict and set(raw) == {"shape", "indptr", "indices", "data"}, name+" CSR schema differs")
    shape = raw["shape"]
    require(type(shape) is list and len(shape) == 2 and all(type(n) is int and n >= 0 for n in shape),
            name+" CSR shape differs")
    require(type(raw["indptr"]) is list and type(raw["indices"]) is list
        and all(type(i) is int for i in raw["indptr"]+raw["indices"]), name+" CSR integer indices differ")
    indptr, indices, values = np.asarray(raw["indptr"], int), np.asarray(raw["indices"], int), array(raw["data"])
    require(indptr.dtype.kind in "iu" and indices.dtype.kind in "iu" and indptr.shape == (shape[0]+1,)
        and indptr[0] == 0 and np.all(np.diff(indptr) >= 0) and indptr[-1] == len(values) == len(indices)
        and np.all(indices >= 0) and np.all(indices < shape[1]), name+" CSR structure differs")
    result = np.zeros(shape)
    for row in range(shape[0]):
        np.add.at(result[row], indices[indptr[row]:indptr[row+1]], values[indptr[row]:indptr[row+1]])
    return result


def mode_errors(actual, expected, name, *, rtol=3e-8, atol=1e-5):
    a, b = array(actual), array(expected)
    require(a.shape == b.shape and a.shape[-1] % 6 == 0, name+" original mode dimensions differ")
    a, b = a.reshape(-1, a.shape[-1]//6, 6), b.reshape(-1, b.shape[-1]//6, 6)
    return {mode: close(a[:, :, start:start+3], b[:, :, start:start+3], name+" "+mode,
        rtol=rtol, atol=atol) for start, mode in ((0, "A"), (3, "P"))}


def audit_reaction_difference(difference, reaction, surface_force, bulk_gradient,
        independent_reaction, independent_surface_force, name):
    """Keep literal force identities and propagate independently checked errors.

    A small R-Fs cannot inherit the relative accuracy of its large operands.
    Its stored value must nevertheless be their exact arithmetic difference;
    neither a fabricated zero nor a substituted surface reaction is allowed.
    """
    d, r, f, g, ri, fi = map(array, (difference, reaction, surface_force,
        bulk_gradient, independent_reaction, independent_surface_force))
    require(d.shape == r.shape == f.shape == g.shape == ri.shape == fi.shape
        and d.ndim == 1 and len(d) % 6 == 0, name+" force dimensions differ")
    require(np.array_equal(r, -g), name+" constraint reaction differs from negative archived bulk gradient")
    require(np.array_equal(d, r-f), name+" stored reaction difference is not the literal force difference")
    reaction_errors = mode_errors(r, ri, name+" independent constraint reaction", rtol=5e-7, atol=1e-5)
    surface_errors = mode_errors(f, fi, name+" independent surface force", rtol=3e-7, atol=1e-5)
    expected = ri-fi
    roundoff = 128*np.finfo(float).eps*(np.abs(r)+np.abs(f)+np.abs(ri)+np.abs(fi))
    propagated = np.abs(r-ri)+np.abs(f-fi)+roundoff
    error = np.abs(d-expected)
    require(np.all(error <= propagated), name+" independently reconstructed reaction difference exceeds operand errors")
    modes = {}
    for start, mode in ((0, "A"), (3, "P")):
        selected = np.arange(len(d)).reshape(-1, 6)[:, start:start+3].ravel()
        modes[mode] = dict(maximum_difference_error_n=float(np.max(error[selected], initial=0.)),
            maximum_propagated_operand_error_bound_n=float(np.max(propagated[selected], initial=0.)))
    return dict(literal_negative_bulk_reaction_exact=True, literal_force_difference_exact=True,
        independent_operand_tolerances_unchanged=True, independent_reaction_errors=reaction_errors,
        independent_surface_force_errors=surface_errors, propagated_difference_errors=modes,
        small_difference_relative_accuracy_certified=False)


def audit_plane_center_residual(saved_residual, normal, offset, center,
        independent_normal, independent_offset):
    """Verify literal metadata and independent normalization at roundoff scale."""
    n, ni, x = (array(v, (3,)) for v in (normal, independent_normal, center))
    d, di, saved = (float(array(v, ())) for v in (offset, independent_offset, saved_residual))
    eps = np.finfo(float).eps
    close(n, ni, "represented plane normal", rtol=64*eps, atol=0.)
    close(d, di, "represented plane offset", rtol=64*eps, atol=64*eps)
    nl, nil, xl = (np.asarray(v, np.longdouble) for v in (n, ni, x))
    dl, dil = np.longdouble(d), np.longdouble(di)
    literal = float(np.dot(nl, xl)-dl)
    require(saved == literal, "literal plane center residual does not match archived coefficients")
    independent = float(np.dot(nil, xl)-dil)
    bound = float(np.abs(xl)@np.abs(nl-nil)+abs(dl-dil)
        +32*np.finfo(np.longdouble).eps*(np.abs(xl)@(abs(nl)+abs(nil))+abs(dl)+abs(dil)))
    require(abs(saved-independent) <= bound, "plane center residual exceeds represented coefficient error")
    return dict(literal_archived_center_residual_exact=True,
        independent_plane_coefficients_match_at_roundoff_resolution=True,
        independent_residual_difference_m=abs(saved-independent),
        propagated_coefficient_error_bound_m=bound)


def matrix_mode_errors(actual, expected, name, *, rtol=5e-8, atol=1e-7):
    a, b = array(actual), array(expected)
    require(a.shape == b.shape and a.ndim == 2 and a.shape[0] % 6 == a.shape[1] % 6 == 0,
            name+" matrix mode dimensions differ")
    errors = {}
    for rowstart, rowmode in ((0, "A"), (3, "P")):
        rows = np.array([6*i+j for i in range(a.shape[0]//6) for j in range(rowstart, rowstart+3)])
        for colstart, colmode in ((0, "A"), (3, "P")):
            cols = np.array([6*i+j for i in range(a.shape[1]//6) for j in range(colstart, colstart+3)])
            errors[rowmode+colmode] = close(a[np.ix_(rows, cols)], b[np.ix_(rows, cols)],
                name+" "+rowmode+colmode, rtol=rtol, atol=atol)
    return errors


def audit_finite_seed(raw, parent, *, normal_xyz, offset_m, center, radius, cohort_id, order=8):
    seal(raw, "finite_seed_sha256")
    fixed(raw, dict(format="genesis-original-material-finite-sphere-seed-0.1",
        scope="prospective_finite_material_seed_no_physical_insertion_or_funding",
        seed_center_world_m=plain(np.asarray(center)), seed_radius_m=float(radius), selected_cohort_id=cohort_id,
        clipping_semantics="analytic_sphere_segment_roots_radial_event_polynomials_and_disk_triangle_rays",
        material_support_semantics="frozen_original_N_q_not_world_sphere_reintersection",
        original_cohort_count=1, finite_front_and_surrounding_material_not_released=True,
        cohort_disconnection_certified=False, partitions_cohort_into_disconnected_banks=False,
        geometry_integrals_exact=False, physical_creation_admitted=False, physical_release_admitted=False,
        dofs_created_or_released=False, material_memory_changed=False, material_history_reinitialized=False,
        clock_advanced=False, creation_work_debited_from_Genesis_j=0.), "finite seed context")
    expected = finite_quadrature(parent, normal_xyz, offset_m, center, radius, cohort_id=cohort_id, order=order)
    surface = raw["surface"]; seal(surface, "embedded_surface_sha256")
    geometry = parent["runtime_state"]["geometry"]
    fixed_fields = dict(format="genesis-original-material-embedded-surface-trace-0.1",
        scope="explicit_current_plane_material_quadrature_paired_virtual_trace_no_physical_insertion",
        source_checkpoint_sha256=parent["source_checkpoint_sha256"], source_identity=geometry["source_identity"],
        time_myr=geometry["time_myr"], source_geometry_bound=True,
        declared_normal_xyz=plain(np.asarray(normal_xyz)), declared_offset_m=float(offset_m),
        declared_plane_input_semantics="literal_numeric_constructor_coefficients_before_normalization",
        quadrature_rule="tensor_Gauss_on_analytic_lambda_pieces_and_Duffy_triangle",
        original_parent_modes="A_xyz_and_P_xyz_metres_per_original_vertex",
        paired_trace_scope="two_virtual_parent_fields_not_created_or_released_mesh_DOFs",
        material_support_semantics="birth_pullback_N_q_frozen_not_world_plane_reintersection",
        area_conventions="current_and_reference_separate_interface_counted_once",
        support_scope="relative_interior_cohort_sheets_boundary_only_support_has_zero_area",
        numerical_integration_scope="approximate_geometry_integrals_exact_discrete_trace_work_at_declared_order",
        geometry_integrals_exact=False,
        recorded_metrics_semantics="fixed_declared_plane_pullback_snapshot_not_transported_after_follow",
        creation_work_debited_from_Genesis_j=0., **{k: False for k in prior.SURFACE_FALSE})
    for key in ("owner_ids", "material_ids", "selected_cohort_ids", "selected_cohort_indices", "patch_cohort_ids",
                "patch_kinds", "parent_ndof", "quadrature_order"):
        fixed_fields[key] = plain(expected[key])
    fixed(surface, fixed_fields, "finite material surface context")
    errors = {}
    for key in set(expected)-set(fixed_fields)-{"clipping_records"}:
        errors[key] = close(surface[key], expected[key], "finite material surface "+key,
            rtol=0 if "position" in key or key in {"barycentric", "radial_scale", "mass_coordinate_q"}
            else 2e-7 if "area" in key else 3e-8,
            atol=2e-7 if "position" in key else 1e-7 if "area" in key else
                3e-13 if key in {"barycentric", "radial_scale", "mass_coordinate_q"} else 3e-11)
    mode_errors(surface["parent_trace"], expected["parent_trace"], "finite surface parent trace", atol=1e-12)
    records = raw["clipping_records"]
    require(len(records) == len(expected["clipping_records"]), "finite seed clipping domain catalogue differs")
    for actual, wanted in zip(records, expected["clipping_records"]):
        require(actual[0] == wanted[0], "finite seed clipping domain kind differs")
        close(actual[1], wanted[1], "finite seed clipping edges or depth", atol=3e-11)
        close(actual[2], wanted[2], "finite seed clipping analytic bounds", atol=3e-11)
    close(surface["total_reference_area_m2"], expected["reference_area_weights_m2"].sum(),
        "finite total reference area", rtol=2e-7, atol=1e-7)
    close(surface["total_current_area_m2"], expected["current_area_weights_m2"].sum(),
        "finite total birth area", rtol=2e-7, atol=1e-7)
    # Direct support checks are independent of numerical integral comparisons.
    positions = array(surface["current_position_m"])
    chart = cohort_chart(parent, cohort_id, normal_xyz, offset_m)
    rawN, rawlam = array(surface["barycentric"]), array(surface["radial_scale"])
    rawcurrent = np.einsum("pi,pij->pj", rawN, rawlam[:, None, None]*chart["Y"]+chart["Z"])
    rawreference = rawlam[:, None]*(rawN@chart["X"])
    close(positions, rawcurrent, "finite archived current material coordinate consistency", rtol=0, atol=2e-7)
    close(surface["reference_position_m"], rawreference,
        "finite archived reference material coordinate consistency", rtol=0, atol=2e-7)
    require(np.all(np.linalg.norm(positions-np.asarray(center), axis=1) <= radius*(1+1e-10)),
            "finite seed archived nodes leave declared sphere support")
    require(np.all(array(surface["reference_area_weights_m2"]) > 0)
        and np.all(array(surface["current_area_weights_m2"]) > 0), "finite seed area is nonpositive")
    allowed = set(fixed_fields)|(set(expected)-{"clipping_records"})|{"material_sections_binding_sha256",
        "total_current_area_m2", "total_reference_area_m2", "embedded_surface_sha256"}
    pin = surface.get("material_sections_binding_sha256")
    require(type(pin) is str and len(pin) == 64 and all(c in "0123456789abcdef" for c in pin),
        "finite original section binding lacks integrity link")
    require(set(surface) == allowed, "whole finite material surface schema differs")
    require(set(raw) == {"format", "scope", "surface", "seed_center_world_m", "seed_radius_m",
        "selected_cohort_id", "clipping_records", "clipping_semantics", "material_support_semantics",
        "original_cohort_count", "finite_front_and_surrounding_material_not_released",
        "cohort_disconnection_certified", "partitions_cohort_into_disconnected_banks", "geometry_integrals_exact",
        "physical_creation_admitted", "physical_release_admitted", "dofs_created_or_released",
        "material_memory_changed", "material_history_reinitialized", "clock_advanced",
        "creation_work_debited_from_Genesis_j", "finite_seed_sha256"}, "whole finite seed schema differs")
    return expected, dict(points_examined=len(expected["cohort_ids"]), original_cohort_id=cohort_id,
        positive_relative_interior_area_checked=True, analytic_sphere_clipping_reconstructed=True,
        full_original_A_P_trace_checked=True, numerical_relative_errors=errors,
        opaque_section_binding_scope="integrity_link_only_not_independently_reconstructed")


def audit_pullback(raw, parent, surface_raw, surface, *, creation_law):
    seal(raw, "pullback_sha256")
    calculated = prior.prestress_fields(parent, surface)
    basis = prior.birth_basis(surface["current_normal_xyz"])
    reference = calculated["reference_traction_world_pa"]
    equivalent = calculated["born_current_equivalent_traction_world_pa"]
    fields = dict(stress_world_pa=calculated["samples"]["stress_world_tensor"],
        reference_traction_world_pa=reference,
        birth_area_ratio=surface["reference_area_weights_m2"]/surface["current_area_weights_m2"],
        birth_traction_world_pa=equivalent, birth_basis_xyz=basis,
        birth_traction_local_pa=np.einsum("pji,pj->pi", basis, equivalent),
        reference_traction_local_pa=np.einsum("pji,pj->pi", basis, reference),
        paired_parent_force_n=calculated["paired_force_n"])
    claims = dict(surface_sha256=surface_raw["embedded_surface_sha256"],
        traction_measure="fixed_inherited_small_strain_sigma_times_reference_normal_per_reference_area",
        equivalent_birth_measure="reference_traction_times_reference_area_over_fixed_birth_area",
        creation_area_measure="original_reference_material_area_Gc_counts_both_banks_once",
        seed_traction_measure="reference_area_traction_in_fixed_birth_normal_tangent_basis",
        work_scope="algebraic_duality_under_the_same_declared_quadrature", bulk_reaction_replaced=False,
        finite_stress_pushforward_claimed=False, physical_surface_selected=False,
        physical_creation_admitted=False, physical_release_admitted=False,
        funding_admitted=False, creation_work_debited_from_Genesis_j=0.)
    fixed(raw, claims, "reference-work finite prestress context")
    errors = {key: close(raw[key], value, "reference-work finite prestress "+key,
        rtol=3e-7 if "force" in key else 5e-8,
        atol=1e-4 if "force" in key else 1e-3 if "stress" in key or "traction" in key else 3e-11)
        for key, value in fields.items()}
    mode_errors(raw["paired_parent_force_n"], fields["paired_parent_force_n"],
        "reference-work finite force", rtol=3e-7, atol=1e-5)
    expected_seed = prior.cohesive_prestress_seed(creation_law, fields["reference_traction_local_pa"])
    require(set(raw["seed"]) == set(expected_seed), "whole finite cohesive seed schema differs")
    for key, value in expected_seed.items():
        if key in {"law", "kernel_only", "physical_release_admitted", "normal_contact_admitted"}:
            fixed(raw["seed"], {key: value}, "virgin cohesive seed")
        else:
            close(raw["seed"][key], value, "finite cohesive seed "+key, rtol=5e-8, atol=1e-9)
    close(raw["full_separation_cost_on_explicit_surface_j"], creation_law["fracture_energy_j_m2"]*
        surface["reference_area_weights_m2"].sum(), "finite surface price is not available funding", rtol=3e-7, atol=1e-3)
    require(set(raw) == set(claims)|set(fields)|{"seed", "full_separation_cost_on_explicit_surface_j", "pullback_sha256"},
            "whole finite reference-work prestress schema differs")
    return calculated, dict(saved_sigma_reference_normal_and_area_checked=True,
        virgin_cohesive_seed_has_no_new_damage=True, explicit_price_is_not_available_funding=True,
        numerical_relative_errors=errors)


def audit_parent_operator(raw, parent):
    expected = original_parent_terms(parent)
    expected["archived_parent_matrices"] = {key: dense_csr(raw[key], "literal source parent "+key)
        for key in ("stiffness", "mass", "basal_drag")}
    expected["archived_parent_force_n"] = array(raw["saved_endpoint_elastic_force_n"])
    expected["archived_initial_energy_j"] = float(raw["saved_endpoint_elastic_energy_j"])
    fixed(raw, dict(parent_ndof=expected["parent_ndof"],
        cohort_ids=[c["id"] for c in parent["runtime_state"]["geometry"]["cohorts"]],
        endpoint_force_scope="saved_endpoint_memory_with_zero_increment_not_absolute_pose_added_again"),
        "whole original volume operator context")
    errors = {}
    for key in ("stiffness", "mass", "basal_drag"):
        errors[key] = matrix_mode_errors(dense_csr(raw[key], "parent "+key), expected[key],
            "original parent "+key, rtol=3e-10, atol=1e-2 if key == "mass" else 1e-4)
    for key in ("external_force_n", "saved_endpoint_elastic_force_n"):
        errors[key] = mode_errors(raw[key], expected[key], "original parent "+key, rtol=3e-10, atol=1e-2)
    errors["energy"] = close(raw["saved_endpoint_elastic_energy_j"], expected["saved_endpoint_elastic_energy_j"],
        "original saved endpoint elastic energy", rtol=5e-12, atol=1.)
    require(set(raw) == {"parent_ndof", "cohort_ids", "stiffness", "mass", "basal_drag", "external_force_n",
        "saved_endpoint_elastic_force_n", "saved_endpoint_elastic_energy_j", "endpoint_force_scope"},
        "whole original parent operator schema differs")
    return expected, dict(original_K_M_D_external_force_and_saved_endpoint_memory_force_checked=True,
        all_original_A_P_matrix_blocks_checked_separately=True, numerical_relative_errors=errors)


def audit_partition(raw, parent, parent_terms, definition, *, order=8):
    seal(raw, "embedded_partition_sha256")
    geometry = parent["runtime_state"]["geometry"]
    cid = definition["original_cohort_id"]
    chart = cohort_chart(parent, cid, definition["normal_xyz"], definition["offset_m"])
    expected = compact_bulk_terms(parent, definition["normal_xyz"], definition["offset_m"],
        definition["seed_center_world_m"], definition["seed_radius_m"], cohort_id=cid, order=order)
    memories = {c["id"]: c for c in parent["runtime_state"]["mechanics"]["state"]["cohorts"]}
    n = parent_terms["parent_ndof"]
    context = dict(format="genesis-tied-finite-seed-compact-bulk-enrichment-0.1",
        scope="prospective_compact_virtual_enrichment_original_memory_no_release_or_funding",
        source_checkpoint_sha256=parent["source_checkpoint_sha256"], source_identity=geometry["source_identity"],
        time_myr=geometry["time_myr"], parent_ndof=n, ndof=n+18, selected_cohort_id=cid,
        selected_cohort_index=chart["cohort_index"], selected_parent_dofs=chart["trace_dofs"].tolist(),
        memory0=[memories[c["id"]]["strain0"] for c in geometry["cohorts"]],
        memory1=[memories[c["id"]]["strain1"] for c in geometry["cohorts"]],
        bubble_definition="H_plus_minus_half_times_sphere_squared_times_27_N0_N1_N2_times_4_t_1_minus_t",
        bulk_gradient="regular_one_sided_reference_gradient_distributional_interface_delta_excluded",
        quadrature_domain="current_seed_ball_two_grid_hemispheres_unique_frozen_birth_chart_inverse",
        material_side_convention="sign_of_literal_section_normal_dot_birth_position_minus_section_offset",
        sphere_center_plane_residual_semantics="represented_plane_residual_no_physical_threshold_or_plane_translation",
        normalized_rank_semantics="formal_18_columns_retained_resolved_rank_reported_no_inversion_or_release_solver",
        quadrature_accuracy="approximate_cohort_clipping_requires_order_convergence_no_exact_integral_claim",
        reaction_semantics="negative_static_endpoint_bulk_elastic_enrichment_gradient_not_accepted_dynamic_birth_reaction",
        added_amplitudes="18_original_A_xyz_P_xyz_modes_per_selected_cohort_all_constrained_zero",
        enriched_memory_representable_by_old_affine_coefficients=False,
        physical_creation_admitted=False, physical_release_admitted=False, funding_admitted=False,
        released_dof_count=0, new_material_birth_count=0, creation_work_debited_from_Genesis_j=0.)
    fixed(raw, context, "constrained compact original volume context")
    pin = raw.get("parent_response_binding_sha256")
    require(type(pin) is str and len(pin) == 64 and all(c in "0123456789abcdef" for c in pin),
        "compact original parent response lacks integrity link")
    local = raw["local"]
    errors = {}
    for key, value in expected.items():
        errors[key] = close(local[key], value, "compact bulk "+key, rtol=0 if "position" in key or key in {"barycentric", "radial_scale"}
            else 8e-7 if "gradient" in key or "strain_operator" in key
            else 3e-7 if "force" in key else 5e-8,
            atol=3e-7 if "position" in key else 3e-13 if key in {"barycentric", "radial_scale"}
                else 5e-6 if "force" in key else 1e-8)
    mode_errors(local["parent_trace"], expected["parent_trace"], "compact bulk parent trace", rtol=1e-9, atol=1e-12)
    mode_errors(local["enrichment_trace"], expected["enrichment_trace"], "compact bulk enrichment trace", rtol=5e-8, atol=1e-12)
    mode_errors(local["parent_strain_operator"], expected["parent_strain_operator"],
        "compact bulk parent strain operator", rtol=8e-7, atol=1e-12)
    mode_errors(local["enrichment_strain_operator"], expected["enrichment_strain_operator"],
        "compact bulk enriched strain operator", rtol=8e-7, atol=1e-12)
    mode_errors(local["enrichment_force_n"], expected["enrichment_force_n"],
        "compact bulk elastic enrichment gradient", rtol=5e-7, atol=1e-5)
    for key in ("stiffness_cross", "stiffness_enriched", "mass_cross", "mass_enriched", "support_parent_stiffness_witness"):
        matrix_mode_errors(local[key], expected[key], "compact "+key, rtol=3e-6, atol=1e-20)
    section = local["section"]; c = chart["cohort"]
    pin = section.get("binding_sha256")
    require(type(pin) is str and len(pin) == 64 and all(x in "0123456789abcdef" for x in pin),
        "compact original cohort section lacks integrity link")
    fixed(section, dict(cohort_index=chart["cohort_index"], cohort_id=cid, face_index=chart["face"],
        owner_id=c["owner_id"], material_id=c["material_id"],
        q_bounds=[c["q_lo"], c["live_q_hi"]], lambda_bounds=[c["inner_scale"], c["outer_scale"]],
        interior_cut=True), "original cohort section identity")
    for key, value in dict(triangle_m=chart["X"], displacement_A_m=chart["A"], slope_P_m=chart["P"],
        normal_xyz=chart["n"], offset_m=chart["d"], projection_slopes=chart["y"], projection_offsets=chart["z"],
        conical_mass_coefficient=chart["coefficient"], mass_kg=parent_terms["cohort_terms"][cid]["mass_kg"],
        volume_m3=parent_terms["cohort_terms"][cid]["volume_m3"]).items():
        close(section[key], value, "original selected cohort section "+key, rtol=3e-12, atol=1e-10)
    rawN, rawlam = array(local["barycentric"]), array(local["radial_scale"])
    close(local["birth_position_m"], np.einsum("pi,pij->pj", rawN, rawlam[:, None, None]*chart["Y"]+chart["Z"]),
        "compact archived birth coordinate consistency", rtol=0, atol=3e-7)
    close(local["reference_position_m"], rawlam[:, None]*(rawN@chart["X"]),
        "compact archived reference coordinate consistency", rtol=0, atol=3e-7)
    augmented = np.block([[expected["parent_stiffness"], expected["stiffness_cross"]],
        [expected["stiffness_cross"].T, expected["stiffness_enriched"]]])
    diagonal = np.sqrt(np.maximum(np.diag(augmented), np.finfo(float).tiny))
    minimum = float(np.linalg.eigvalsh((augmented+augmented.T)/2/diagonal[:, None]/diagonal[None, :]).min())
    require(minimum >= -1e-9 and -1e-9 <= float(local["scaled_augmented_stiffness_minimum_eigenvalue"]) <= 1e-5,
            "compact augmented stiffness is not nonnegative at declared order")
    E = expected["enrichment_trace"]
    stacked = (np.sqrt(expected["reference_volume_weights_m3"])[:, None, None]*E).reshape(-1, 18)
    normalized = stacked/np.maximum(np.linalg.norm(stacked, axis=0), np.finfo(float).tiny)
    rank = int(np.linalg.matrix_rank(normalized))
    A = np.array([0, 1, 2, 6, 7, 8, 12, 13, 14]); P = A+3
    Arank, Prank = (int(np.linalg.matrix_rank(normalized[:, columns])) for columns in (A, P))
    require(type(local["numerical_enrichment_trace_rank"]) is int and 3 <= local["numerical_enrichment_trace_rank"] <= 18
        and type(local["numerical_enrichment_A_trace_rank"]) is int and 3 <= local["numerical_enrichment_A_trace_rank"] <= 9
        and type(local["numerical_enrichment_P_trace_rank"]) is int and 3 <= local["numerical_enrichment_P_trace_rank"] <= 9,
        "compact resolved A/P trace ranks are invalid")
    require(np.isfinite(local["normalized_enrichment_trace_condition_number"])
        and local["normalized_enrichment_trace_condition_number"] >= 1., "compact conditioning report is invalid")
    archived_E = array(local["enrichment_trace"])
    archived_stacked = (np.sqrt(array(local["reference_volume_weights_m3"]))[:, None, None]*archived_E).reshape(-1, 18)
    archived_normalized = archived_stacked/np.maximum(np.linalg.norm(archived_stacked, axis=0), np.finfo(float).tiny)
    archived_singular = np.linalg.svd(archived_normalized, compute_uv=False)
    fixed(local, dict(numerical_enrichment_trace_rank=int(np.linalg.matrix_rank(archived_normalized)),
        numerical_enrichment_A_trace_rank=int(np.linalg.matrix_rank(archived_normalized[:, A])),
        numerical_enrichment_P_trace_rank=int(np.linalg.matrix_rank(archived_normalized[:, P]))),
        "literal archived compact trace numerical rank metadata")
    close(local["normalized_enrichment_trace_condition_number"],
        archived_singular[0]/max(archived_singular[-1], np.finfo(float).tiny),
        "literal archived compact trace conditioning metadata", rtol=1e-12, atol=0.)
    center_report = audit_plane_center_residual(local["center_plane_residual_m"],
        section["normal_xyz"], section["offset_m"], definition["seed_center_world_m"], chart["n"], chart["d"])
    Kqa = np.zeros((n, 18)); Mqa = np.zeros((n, 18))
    Kqa[chart["trace_dofs"]] = expected["stiffness_cross"]
    Mqa[chart["trace_dofs"]] = expected["mass_cross"]
    K = np.block([[parent_terms["stiffness"], Kqa], [Kqa.T, expected["stiffness_enriched"]]])
    M = np.block([[parent_terms["mass"], Mqa], [Mqa.T, expected["mass_enriched"]]])
    D = np.zeros((n+18, n+18)); D[:n, :n] = parent_terms["basal_drag"]
    F = np.r_[parent_terms["saved_endpoint_elastic_force_n"], expected["enrichment_force_n"]]
    R = np.vstack((np.eye(n), np.zeros((18, n)))); J = np.hstack((np.zeros((18, n)), np.eye(18)))
    for key, value, parentkey in (("stiffness", K, "stiffness"), ("mass_matrix", M, "mass")):
        archived = dense_csr(raw[key], "partition "+key)
        require(np.array_equal(archived[:n, :n], parent_terms["archived_parent_matrices"][parentkey]),
            "compact tied literal original parent "+key+" changed")
        matrix_mode_errors(archived[:n, n:], value[:n, n:], "whole compact "+key+" cross", rtol=3e-6, atol=1e-20)
        matrix_mode_errors(archived[n:, :n], value[n:, :n], "whole compact "+key+" transposed cross", rtol=3e-6, atol=1e-20)
        matrix_mode_errors(archived[n:, n:], value[n:, n:], "whole compact "+key+" enriched", rtol=3e-6, atol=1e-20)
    archived_D = dense_csr(raw["basal_drag_matrix"], "partition basal drag")
    require(np.array_equal(archived_D[:n, :n], parent_terms["archived_parent_matrices"]["basal_drag"])
        and not np.any(archived_D[n:]) and not np.any(archived_D[:, n:]),
        "compact basal resistance changed original parent or invented enriched drag")
    for key, value in (("prolongation", R), ("tie_operator", J)):
        require(np.array_equal(dense_csr(raw[key], key), value), "exact constrained compact "+key+" differs")
    require(not np.any(array(raw["external_force_n"])[n:]), "compact added external force is not exactly zero")
    require(np.array_equal(array(raw["initial_force_n"][:n]), parent_terms["archived_parent_force_n"]),
        "compact tied literal original parent endpoint force changed")
    mode_errors(raw["initial_force_n"][:n], F[:n], "tied source initial force", rtol=3e-10, atol=1e-2)
    mode_errors(raw["initial_force_n"][n:], F[n:], "compact source added initial force", rtol=5e-7, atol=1e-5)
    close(raw["external_force_n"], np.r_[parent_terms["external_force_n"], np.zeros(18)],
        "compact external force", rtol=3e-10, atol=1e-2)
    close(raw["initial_elastic_energy_j"], parent_terms["saved_endpoint_elastic_energy_j"],
        "tied complete original energy", rtol=5e-12, atol=1.)
    require(set(local) == set(expected)|{"section", "numerical_enrichment_trace_rank",
        "numerical_enrichment_A_trace_rank", "numerical_enrichment_P_trace_rank",
        "normalized_enrichment_trace_condition_number", "center_plane_residual_m",
        "scaled_augmented_stiffness_minimum_eigenvalue"}, "whole compact local bulk schema differs")
    require(set(raw) == set(context)|{"parent_response_binding_sha256", "local", "stiffness", "mass_matrix",
        "basal_drag_matrix", "external_force_n", "initial_force_n", "initial_elastic_energy_j",
        "prolongation", "tie_operator", "embedded_partition_sha256"}, "whole constrained compact partition schema differs")
    expected.update(whole_K=K, whole_M=M, whole_D=D, whole_F=F, whole_R=R, whole_J=J)
    return expected, dict(reference_gradient_and_full_original_A_P_basis_checked=True,
        distributional_interface_gradient_excluded=True, both_material_sides_positive_volume=True,
        entire_saved_six_component_memory_preserved=True, all_amplitudes_remain_constrained_zero=True,
        sphere_local_quadrature_not_parent_mask=True, original_parent_K_M_D_force_energy_checked=True,
        independent_resolved_trace_rank=rank, independent_resolved_A_rank=Arank, independent_resolved_P_rank=Prank,
        numerical_rank_not_full_independent_release_solver=True, scaled_stiffness_minimum_eigenvalue=minimum,
        interface_traction_range_or_physical_release_solver_certified=False,
        material_plane_jump_null_modes_not_released=True,
        opaque_parent_response_binding_scope="integrity_link_only_not_independently_reconstructed",
        represented_plane_center_residual=center_report,
        numerical_relative_errors=errors)


INSERTION_FALSE = ("physical_creation_admitted", "physical_release_admitted", "funding_admitted",
    "history_reinitialized", "material_memory_changed", "activation_accepted", "clock_advanced",
    "dynamic_bank_reaction_replaced", "historical_three_dimensional_memory_recovered")


def insertion_context(parent, observer):
    runtime = parent["runtime_state"]; geometry = runtime["geometry"]
    return dict(format="genesis-source-bound-finite-seed-constrained-insertion-0.1",
        scope="calculated_memory_finite_material_seed_and_constrained_compact_bulk_modes",
        parent_checkpoint_sha256=parent["checkpoint_sha256"], formation_checkpoint_sha256=parent["source_checkpoint_sha256"],
        source_identity=geometry["source_identity"], time_myr=geometry["time_myr"],
        accepted_steps=runtime["mechanics"]["state"]["accepted_steps"],
        accepted_step_sizes_myr=runtime["accepted_step_sizes_myr"], observer_binding_sha256=observer["binding_sha256"],
        inherited_runtime_ledgers={k: v for k, v in runtime.items() if k.endswith("_j")},
        source_historical_surface_creation_energy_law=None, selected_sharp_material_surface=None,
        historical_dissipation_available_for_creation_j=0., creation_work_debited_from_Genesis_j=0.,
        new_accepted_intervals=0, new_material_births=0, released_physical_dofs=0,
        **{k: False for k in INSERTION_FALSE})


def audit_gated_insertion(raw, parent, observer, nucleation):
    seal(raw, "seed_insertion_sha256")
    context = insertion_context(parent, observer)
    expected = {**context, "status": "no_calculated_activation_event",
        "geometry_provenance": "prospective_source_gated_seed_not_physical_admission",
        "nucleation_proposal": nucleation, "insertion": None,
        "finite_positive_surface_area_prepared": False, "constrained_bulk_modes_prepared": False,
        "additional_memory_needed_before_nonzero_enrichment": True}
    fixed(raw, expected, "authentic source-gated finite insertion")
    require(set(raw) == set(expected)|{"seed_insertion_sha256"}, "whole source-gated finite insertion schema differs")
    return dict(actual_noevent_gate_retained=True, no_finite_surface_or_modes_invented=True,
        no_time_memory_dofs_or_funding_changed=True)


def audit_bound_insertion(raw, parent, observer, parent_terms, definition, seed_raw, surface):
    seal(raw, "seed_insertion_sha256")
    context = {**insertion_context(parent, observer),
        "status": "constrained_computational_material_seed_no_physical_birth",
        "geometry_provenance": "explicit_computational_seed_not_source_activation",
        "finite_seed": seed_raw, "partition_quadrature_order": 8,
        "reaction_difference_semantics": "retained_endpoint_bulk_divergence_and_quadrature_terms_not_zeroed",
        "surface_force_semantics": "positive_normal_reference_traction_dual_to_compact_jump",
        "constraint_reaction_semantics": "endpoint_elastic_added_mode_reaction_not_full_dynamic_reaction",
        "surface_force_is_discrete_bulk_reaction": False, "finite_positive_surface_area_prepared": True,
        "constrained_bulk_modes_prepared": True, "additional_memory_needed_before_nonzero_enrichment": True,
        "surface_null_modes_remain_constrained": True, "volume_rank_does_not_certify_surface_opening": True,
        "null_mode_release_needs_separate_work_admission": True,
        "parameter_provenance": "explicit_new_constitutive_input_not_historical_toughness"}
    fixed(raw, context, "source-bound constrained finite insertion")
    sample_report = prior.audit_samples(raw["source_stress_samples"], parent, observer)
    close(raw["source_stress_samples"]["angular_barycentric"], surface["barycentric"], "finite prestress sample coordinates")
    close(raw["source_stress_samples"]["mass_coordinate_q"], surface["mass_coordinate_q"], "finite prestress sample mass", atol=3e-11)
    stress, prestress_report = audit_pullback(raw["prestress_pullback"], parent, seed_raw["surface"], surface,
        creation_law=prior.DECLARED_LAW["creation_law"])
    partition, partition_report = audit_partition(raw["embedded_partition"], parent, parent_terms, definition)
    chart = cohort_chart(parent, definition["original_cohort_id"], definition["normal_xyz"], definition["offset_m"])
    fields = compact_fields(chart, surface["barycentric"], surface["radial_scale"],
        definition["seed_center_world_m"], definition["seed_radius_m"])
    bubble, trace = fields["support_weight"], fields["surface_enrichment_trace"]
    force = np.einsum("pai,pa,p->i", trace, stress["reference_traction_world_pa"], surface["reference_area_weights_m2"])
    reaction = -partition["enrichment_force_n"]
    close(raw["surface_enrichment_bubble"], bubble, "finite surface compact bubble", rtol=5e-8, atol=2e-10)
    mode_errors(raw["surface_enrichment_trace"], trace, "finite surface enrichment trace", rtol=5e-8, atol=1e-12)
    alpha = np.zeros((18, 3))
    scalar_A = (chart["X"]+chart["A"])@chart["n"]-chart["d"]
    scalar_P = (chart["X"]+chart["P"])@chart["n"]
    for vertex in range(3):
        for axis in range(3):
            alpha[6*vertex+axis, axis] = scalar_A[vertex]
            alpha[6*vertex+3+axis, axis] = scalar_P[vertex]
    alpha /= np.linalg.norm(alpha, axis=0)
    close(raw["analytic_surface_null_modes"], alpha, "analytic original material plane jump null modes", rtol=3e-12, atol=1e-15)
    archived_trace = array(raw["surface_enrichment_trace"])
    archived_area = array(seed_raw["surface"]["reference_area_weights_m2"])
    weighted = (np.sqrt(archived_area)[:, None, None]*archived_trace).reshape(-1, 18)
    column_scale = np.maximum(np.linalg.norm(weighted, axis=0), np.finfo(float).tiny)
    jump_rank = int(np.linalg.matrix_rank(weighted/column_scale))
    fixed(raw, dict(surface_jump_trace_rank=jump_rank), "literal archived numerical surface jump rank")
    null_residual = np.linalg.norm(weighted@array(raw["analytic_surface_null_modes"]), axis=0)
    close(raw["surface_null_jump_weighted_norm_m"], null_residual,
        "literal archived analytic jump null residual", rtol=1e-12, atol=1e-20)
    require(np.max(null_residual) <= 2048*np.finfo(float).eps*max(np.linalg.norm(weighted), 1.),
        "analytic material plane null modes have nonzero resolved surface jump")
    mode_errors(raw["surface_enrichment_force_n"], force, "finite compact surface force", rtol=3e-7, atol=1e-5)
    mode_errors(raw["endpoint_elastic_constraint_reaction_n"], reaction,
        "endpoint elastic constraint reaction is negative bulk gradient", rtol=5e-7, atol=1e-5)
    difference_report = audit_reaction_difference(raw["constraint_minus_surface_force_n"],
        raw["endpoint_elastic_constraint_reaction_n"], raw["surface_enrichment_force_n"],
        raw["embedded_partition"]["local"]["enrichment_force_n"], reaction, force,
        "retained endpoint bulk reaction minus surface force")
    require(set(raw) == set(context)|{"source_stress_samples", "prestress_pullback", "embedded_partition",
        "surface_enrichment_bubble", "surface_enrichment_trace", "surface_enrichment_force_n",
        "analytic_surface_null_modes", "surface_null_jump_weighted_norm_m", "surface_jump_trace_rank",
        "endpoint_elastic_constraint_reaction_n", "constraint_minus_surface_force_n", "seed_insertion_sha256"},
        "whole source-bound constrained insertion schema differs")
    return dict(partition=partition, stress=stress, surface_trace=trace, surface_force=force), dict(
        source_stress=sample_report, prestress=prestress_report, compact_partition=partition_report,
        surface_jump_is_same_compact_bulk_basis=True, endpoint_bulk_reaction_rebuilt_from_saved_memory=True,
        reference_surface_sigma_normal_not_substituted_for_discrete_bulk_gradient=True,
        analytic_material_plane_jump_null_modes_checked=True,
        archived_numerical_surface_jump_rank=jump_rank,
        retained_reaction_difference=difference_report,
        surface_null_modes_not_admitted_as_fracture_funding=True,
        no_actual_dynamic_reaction_or_funding_admitted=True)


def audit_seed_definition(raw, parent, face):
    geometry = parent["runtime_state"]["geometry"]
    c = min((c for c in geometry["cohorts"] if c["face_index"] == face), key=lambda c: c["id"])
    vertices = np.asarray(geometry["faces"][face], int)
    X = array(geometry["vertices_m"])[vertices]
    pose = array(parent["runtime_state"]["displacement_m"]).reshape(-1, 6)[vertices]
    N = np.full(3, 1/3.); lam = .5*(c["inner_scale"]+c["outer_scale"])
    center = N@(lam*(X+pose[:, 3:])+(pose[:, :3]-pose[:, 3:]))
    thickness = (c["outer_scale"]-c["inner_scale"])*np.linalg.norm(N@(X+pose[:, 3:]))
    path = ROOT/"analysis/genesis_volume_location_validation/evidence/actual_control"/f"supplied_current_plane_face_{face}.json"
    original = path.read_bytes(); old = json.loads(original); normal = array(old["normal_xyz"])
    q = (c["q_lo"]+(c["live_q_hi"]-c["q_lo"])*(c["outer_scale"]**3-lam**3)/
        (c["outer_scale"]**3-c["inner_scale"]**3))
    fixed(raw, dict(original_face_index=face, original_cohort_id=c["id"], original_layer_index=c["layer_index"],
        original_owner_id=c["owner_id"], original_material_id=c["material_id"], normal_xyz=normal.tolist(),
        previous_normal_source_pin=dict(path=str(path.relative_to(ROOT)), raw_sha256=hashlib.sha256(original).hexdigest()),
        center_original_barycentric=N.tolist(),
        plane_provenance="prior_explicit_normal_with_new_declared_current_cohort_midpoint_offset",
        radius_provenance="explicit_computational_0p12_fraction_of_current_centroid_band_thickness_not_material_calibration",
        computational_fixture=True, actual_nucleation_candidate=None, material_birth_history_replaced=False,
        physical_creation_admitted=False), "explicit source material seed definition")
    for key, value in dict(seed_center_world_m=center, offset_m=float(normal@center), seed_radius_m=.12*thickness,
        current_centroid_radial_band_thickness_m=thickness, center_original_radial_scale=lam,
        center_original_mass_coordinate_q=q).items():
        close(raw[key], value, "source material finite seed definition "+key, rtol=1e-13, atol=1e-10)
    return raw


def audit_surface_virtual(raw, parent, surface, stress):
    n = surface["parent_ndof"]; j = np.arange(n, dtype=float)
    increments = np.stack((1e-5*np.sin(.173*(j+1)), 1e-5*np.cos(.119*(j+2))))
    close(raw["paired_original_parent_virtual_increments_m"], increments, "declared surface virtual increments", rtol=0, atol=1e-20)
    jump = prior.point_trace(surface, increments[1]-increments[0])
    close(raw["virtual_jump_m"], jump, "finite surface virtual jump", atol=1e-14)
    force = stress["paired_force_n"]
    mode_errors(raw["reference_paired_parent_force_n"], force, "finite surface virtual reference force", rtol=3e-7)
    mode_errors(raw["equivalent_birth_paired_parent_force_n"], force, "finite surface virtual born force", rtol=3e-7)
    work = prior.paired_virtual_work(surface, stress["reference_traction_world_pa"], increments)
    for key in ("reference_virtual_work_j", "equivalent_birth_virtual_work_j", "nodal_virtual_work_j"):
        close(raw[key], work, "finite surface virtual "+key, rtol=3e-7, atol=1e-5)
    fixed(raw, dict(equal_parent_fields_zero_jump_exact=True,
        work_scope="discrete_trace_duality_not_funded_physical_release"), "finite surface work scope")
    require(max(raw["force_relative_errors_A_P"]) < 3e-12 and max(raw["virtual_work_relative_errors"]) < 3e-12,
            "finite surface recorded force or work identity failed")
    return dict(reference_birth_and_nodal_work_checked=True, full_A_P_virtual_trace_duality_checked=True)


def audit_bulk_virtual(raw, parent, parent_terms, definition, inserted):
    p = inserted["partition"]; n = parent_terms["parent_ndof"]
    q = 1e-5*np.sin(.217*(np.arange(n)+1.)); a = 2e-5*np.cos(.173*(np.arange(18)+3.))
    close(raw["parent_virtual_increment_m"], q, "declared compact parent virtual increment", rtol=0, atol=1e-20)
    close(raw["enrichment_virtual_amplitude_m"], a, "declared compact enriched virtual amplitude", rtol=0, atol=1e-20)
    delta = np.r_[q, a]; force = p["whole_F"]+p["whole_K"]@delta
    midpoint = p["whole_F"]+.5*p["whole_K"]@delta
    work = float(midpoint@delta)
    response = raw["virtual_response"]
    mode_errors(response["generalized_force_n"][:n], force[:n], "virtual compact parent force", rtol=5e-9, atol=1e-2)
    mode_errors(response["generalized_force_n"][n:], force[n:], "virtual compact added force", rtol=5e-7, atol=1e-5)
    mode_errors(response["static_tie_reaction_n"], -force[n:], "virtual endpoint elastic tie reaction", rtol=5e-7, atol=1e-5)
    mode_errors(raw["midpoint_generalized_force_n"], midpoint, "virtual midpoint compact force", rtol=5e-9, atol=1e-2)
    for key in ("midpoint_virtual_work_j", "quadratic_bulk_virtual_work_j"):
        close(raw[key], work, "compact force quadratic work "+key, rtol=5e-10, atol=1e-3)
    close(response["finite_increment_bulk_work_j"], work, "virtual response force work", rtol=5e-10, atol=1e-3)
    close(response["elastic_energy_j"], parent_terms["saved_endpoint_elastic_energy_j"]+work,
        "virtual compact complete elastic energy", rtol=5e-12, atol=1.)
    cid = definition["original_cohort_id"]; dofs = parent_terms["cohort_terms"][cid]["dofs"]
    pointstrain = (p["point_elastic_strain"]+np.einsum("pai,i->pa", p["parent_strain_operator"], q[dofs])+
        np.einsum("pai,i->pa", p["enrichment_strain_operator"], a))
    close(response["point_elastic_strain"], pointstrain, "virtual regular enriched point memory", rtol=5e-7, atol=1e-10)
    geometry = parent["runtime_state"]["geometry"]
    memory0, memory1 = [], []
    memories = {c["id"]: c for c in parent["runtime_state"]["mechanics"]["state"]["cohorts"]}
    frames = prior.source_frames(parent)
    for cohort in geometry["cohorts"]:
        f = cohort["face_index"]; vertices = np.asarray(geometry["faces"][f], int)
        b0, b1 = volume.independent_strain_operators(array(geometry["vertices_m"])[vertices], frames[f], prior.SOURCE_ANGULAR_POINTS)
        localq = q[(6*vertices[:, None]+np.arange(6)).ravel()]
        memory0.append(array(memories[cohort["id"]]["strain0"])+b0@localq)
        memory1.append(array(memories[cohort["id"]]["strain1"])+b1@localq)
    close(response["parent_memory0"], memory0, "virtual original affine memory update0", rtol=5e-10, atol=2e-12)
    close(response["parent_memory1"], memory1, "virtual original inverse-scale memory update1", rtol=5e-10, atol=2e-12)
    fixed(response, dict(computational_enrichment_probe=True), "virtual enriched response provenance")
    identities = raw["tied_parent_identities_exact"]
    require(set(identities) == {"stiffness", "mass", "basal_drag", "external_force", "endpoint_elastic_force",
        "endpoint_elastic_energy", "tie_times_prolongation", "memory0", "memory1"}
        and all(v is True for v in identities.values()), "tied original parent equality failed")
    fixed(raw, dict(tied_original_response_exact=True, all_enrichment_amplitudes_in_actual_state_zero=True,
        enriched_probe_is_computational_not_accepted_material_history=True,
        surface_force_is_discrete_bulk_reaction=False,
        static_endpoint_reaction_is_accepted_dynamic_birth_reaction=False,
        added_modes_released=False, funding_admitted=False), "compact virtual work scope")
    for key, value in dict(static_endpoint_bulk_enrichment_gradient_n=p["enrichment_force_n"],
        static_endpoint_elastic_constraint_reaction_n=-p["enrichment_force_n"],
        surface_jump_generalized_force_n=inserted["surface_force"]).items():
        mode_errors(raw[key], value, "virtual compact "+key, rtol=5e-7, atol=1e-5)
    audit_reaction_difference(raw["constraint_minus_surface_force_n"],
        raw["static_endpoint_elastic_constraint_reaction_n"], raw["surface_jump_generalized_force_n"],
        raw["static_endpoint_bulk_enrichment_gradient_n"], -p["enrichment_force_n"],
        inserted["surface_force"], "virtual compact retained reaction difference")
    chart = cohort_chart(parent, cid, definition["normal_xyz"], definition["offset_m"])
    probes = raw["original_material_support_zero_trace_probes"]
    c = chart["cohort"]; mid = .5*(c["inner_scale"]+c["outer_scale"])
    catalogue = [("inner_original_cohort_boundary", [1/3]*3, c["inner_scale"]),
        ("outer_original_cohort_boundary", [1/3]*3, c["outer_scale"]),
        ("original_angular_boundary", [0., .5, .5], mid),
        ("outside_birth_ball_inside_original_cohort", [.34, .33, .33], mid)]
    require(type(probes) is list and len(probes) == 4, "compact zero-support probe catalogue differs")
    for probe, (name, barycentric, radial_scale) in zip(probes, catalogue):
        fixed(probe, dict(name=name, barycentric=barycentric, radial_scale=radial_scale), "compact original support probe definition")
        fields = compact_fields(chart, np.asarray(probe["barycentric"])[None], np.array([probe["radial_scale"]]),
            definition["seed_center_world_m"], definition["seed_radius_m"])
        close(probe["birth_position_m"], fields["position_m"][0], "compact zero-support original position", rtol=0, atol=3e-7)
        require(probe["bubble"] == 0 and not np.any(probe["enrichment_trace"]) and not np.any(probe["jump_trace"])
            and fields["support_weight"][0] == 0, "compact basis leaked beyond its material support")
    require(raw["force_work_relative_error"] < 3e-12
        and raw["complete_parent_energy_subtraction_error_j"] <= raw["complete_parent_energy_rounding_allowance_j"],
        "compact virtual work or complete parent energy subtraction failed")
    archived_midpoint = array(raw["midpoint_generalized_force_n"])
    archived_work = float(raw["quadratic_bulk_virtual_work_j"])
    work_scale = max(abs(raw["midpoint_virtual_work_j"]), abs(archived_work),
        float(np.sum(abs(archived_midpoint*delta))), 1.)
    work_error = abs(raw["midpoint_virtual_work_j"]-archived_work)/work_scale
    close(raw["force_work_relative_error"], work_error, "literal archived compact force-work error", rtol=1e-12, atol=1e-18)
    initial_energy = float(parent_terms["archived_initial_energy_j"])
    change = float(response["elastic_energy_j"])-initial_energy
    error = abs(change-archived_work)
    allowance = 256*np.finfo(float).eps*max(abs(float(response["elastic_energy_j"])), abs(initial_energy), abs(archived_work), 1.)
    close(raw["virtual_complete_parent_energy_change_j"], change, "literal archived complete parent energy change", rtol=0, atol=0)
    close(raw["complete_parent_energy_subtraction_error_j"], error, "literal archived complete parent energy subtraction error", rtol=0, atol=0)
    close(raw["complete_parent_energy_rounding_allowance_j"], allowance, "literal archived complete parent energy rounding allowance", rtol=1e-14, atol=0)
    return dict(tied_original_parent_response_checked=True, added_bulk_gradient_and_midpoint_work_checked=True,
        original_memory_update_checked_without_reinitialization=True,
        nonzero_enrichment_only_virtual_not_accepted_or_persisted=True,
        separate_A_and_P_forces_checked=True, zero_support_probes_checked=True)


def audit_follow(raw, parent, surface, definition):
    translation = np.array([1000., -600., 400.])
    increment = np.zeros(surface["parent_ndof"]); increment.reshape(-1, 6)[:, :3] = translation
    expected = surface["reference_position_m"]+prior.point_trace(surface,
        array(parent["runtime_state"]["displacement_m"])+increment)
    close(raw["parent_pose_increment_m"], increment, "frozen finite material pose increment", rtol=0, atol=0)
    close(raw["followed_material_position_m"], expected, "frozen finite material followed positions", rtol=0, atol=3e-7)
    close(raw["translation_world_m"], translation, "frozen finite material translation", rtol=0, atol=0)
    distances = np.linalg.norm(expected-definition["seed_center_world_m"], axis=1)
    require(np.all(distances > definition["seed_radius_m"]), "frozen finite material follow remained in birth sphere")
    close(raw["minimum_distance_to_fixed_birth_center_m"], distances.min(), "frozen follow minimum sphere distance", rtol=1e-9, atol=1e-6)
    fixed(raw, dict(computational_pose_probe=True, runtime_state_changed=False,
        all_followed_points_outside_fixed_birth_sphere=True, frozen_barycentric_and_q_retained=True,
        frozen_birth_enrichment_retained=True, world_sphere_or_plane_reintersection_performed=False),
        "frozen finite material follow semantics")
    return dict(frozen_N_q_trace_follow_checked=True, world_sphere_reintersection_not_performed=True)


def audit_convergence(records, comparisons, parent, definition, inserted, surface):
    require(type(records) is list and [r.get("quadrature_order") for r in records] == [4, 8, 12],
            "finite seed complete quadrature convergence orders differ")
    numerical = []
    for record in records:
        order = record["quadrature_order"]
        surf = surface if order == 8 else finite_quadrature(parent, definition["normal_xyz"], definition["offset_m"],
            definition["seed_center_world_m"], definition["seed_radius_m"], cohort_id=definition["original_cohort_id"], order=order)
        bulk = inserted["partition"] if order == 8 else compact_bulk_terms(parent, definition["normal_xyz"],
            definition["offset_m"], definition["seed_center_world_m"], definition["seed_radius_m"],
            cohort_id=definition["original_cohort_id"], order=order)
        expected = dict(quadrature_order=order, surface_point_count=len(surf["cohort_ids"]),
            volume_point_count=len(bulk["barycentric"]),
            side_point_counts=[int(np.sum(bulk["side"] == side)) for side in (-.5, .5)],
            total_reference_area_m2=float(surf["reference_area_weights_m2"].sum()),
            total_current_area_m2=float(surf["current_area_weights_m2"].sum()),
            sampled_reference_support_volume_m3=float(bulk["reference_volume_weights_m3"].sum()),
            sampled_birth_support_volume_m3=float(bulk["birth_volume_weights_m3"].sum()),
            static_endpoint_bulk_enrichment_gradient_n=bulk["enrichment_force_n"])
        for key in ("stiffness_cross", "stiffness_enriched", "mass_cross", "mass_enriched"):
            expected[key] = bulk[key]
        for key, value in expected.items():
            if key in {"quadrature_order", "surface_point_count", "volume_point_count", "side_point_counts"}:
                fixed(record, {key: value}, "finite convergence counts")
            else:
                close(record[key], value, "finite quadrature convergence "+key, rtol=5e-7, atol=1e-6)
        mode_errors(record["static_endpoint_bulk_enrichment_gradient_n"], bulk["enrichment_force_n"],
            "convergence original A/P added force", rtol=5e-7, atol=1e-5)
        fixed(record, dict(geometrical_integrals_claimed_exact=False, physical_release_admitted=False), "finite quadrature claims")
        require(len(bulk["barycentric"]) == 4*order**3, "contained seed ball was undersampled or support lost")
        numerical.append(expected)
    require(type(comparisons) is list and len(comparisons) == 2, "finite convergence comparison catalogue differs")
    final = records[-1]
    for index, comparison in enumerate(comparisons):
        row = records[index]
        fixed(comparison, dict(quadrature_order=row["quadrature_order"], comparison_order=12,
            finite_order_comparison_only=True, certified_continuum_error_bound=False), "finite convergence comparison scope")
        area = [abs(row[k]-final[k])/max(abs(final[k]), 1.) for k in ("total_reference_area_m2", "total_current_area_m2")]
        matrix = {key: float(np.linalg.norm(array(row[key])-array(final[key]))/max(np.linalg.norm(array(final[key])), 1.))
            for key in ("stiffness_cross", "stiffness_enriched", "mass_cross", "mass_enriched")}
        force = [float(np.linalg.norm(array(row["static_endpoint_bulk_enrichment_gradient_n"]).reshape(3, 6)[:, s:s+3]-
            array(final["static_endpoint_bulk_enrichment_gradient_n"]).reshape(3, 6)[:, s:s+3])/
            max(np.linalg.norm(array(final["static_endpoint_bulk_enrichment_gradient_n"]).reshape(3, 6)[:, s:s+3]), 1.))
            for s in (0, 3)]
        close(comparison["area_relative_errors_reference_current"], area, "saved finite area convergence", rtol=1e-12, atol=1e-15)
        close(comparison["endpoint_enrichment_force_relative_errors_A_P"], force, "saved finite A/P force convergence", rtol=1e-12, atol=1e-15)
        for key, value in matrix.items():
            close(comparison["bulk_matrix_relative_errors"][key], value, "saved finite matrix convergence "+key, rtol=1e-12, atol=1e-15)
            family = {}
            A = np.array([0, 1, 2, 6, 7, 8, 12, 13, 14]); P = A+3
            for rows, rname in ((A, "A"), (P, "P")):
                for columns, cname in ((A, "A"), (P, "P")):
                    first, last = array(row[key])[np.ix_(rows, columns)], array(final[key])[np.ix_(rows, columns)]
                    family[rname+cname] = float(np.linalg.norm(first-last)/max(np.linalg.norm(last), np.finfo(float).tiny))
                    close(comparison["bulk_matrix_relative_errors_AA_AP_PA_PP"][key][rname+cname], family[rname+cname],
                        "saved separate A/P matrix convergence "+key+rname+cname, rtol=1e-12, atol=1e-15)
            if index == 1: require(max(family.values()) <= 2e-6, "order-eight/twelve A/P bulk matrix convergence failed")
        if index == 1:
            require(max(area+force+list(matrix.values())) <= 2e-6, "order-eight/twelve finite work convergence failed")
    return dict(orders_checked=[4, 8, 12], positive_sphere_local_volume_support_at_every_order=True,
        every_AA_AP_PA_PP_matrix_family_checked=True, order_eight_twelve_agreement_checked=True,
        finite_order_comparison_is_not_certified_continuum_error=True)


def audit_large_probe(raw, parent, definition):
    surface, report = audit_finite_seed(raw["finite_seed"], parent, normal_xyz=definition["normal_xyz"],
        offset_m=definition["offset_m"], center=definition["seed_center_world_m"], radius=100.,
        cohort_id=definition["original_cohort_id"], order=8)
    fixed(raw, dict(name="separate_100_m_original_thin_cohort_surface_only_probe", computational_fixture=True,
        seed_radius_m=100., current_disk_clipped_by_original_cohort=True,
        volume_enrichment_or_operator_built=False, bulk_quadrature_convergence_claimed=False,
        physical_activation_admitted=False, physical_creation_admitted=False, physical_release_admitted=False,
        funding_admitted=False, creation_work_debited_from_Genesis_j=0.), "separate clipped 100m surface scope")
    rows = raw["quadrature_order_records"]
    require([r["quadrature_order"] for r in rows] == [4, 8, 12], "100m clipped surface quadrature orders differ")
    for record in rows:
        order = record["quadrature_order"]
        data = surface if order == 8 else finite_quadrature(parent, definition["normal_xyz"], definition["offset_m"],
            definition["seed_center_world_m"], 100., cohort_id=definition["original_cohort_id"], order=order)
        fixed(record, dict(surface_point_count=len(data["cohort_ids"])), "100m clipped surface counts")
        for key in ("reference", "current"):
            close(record["total_"+key+"_area_m2"], data[key+"_area_weights_m2"].sum(),
                "100m independently clipped "+key+" area", rtol=3e-7, atol=1e-5)
    errors = [abs(rows[1][k]-rows[2][k])/max(abs(rows[2][k]), 1.) for k in
        ("total_reference_area_m2", "total_current_area_m2")]
    close(raw["order_eight_twelve_area_relative_errors"], errors, "100m clipped finite area comparison", rtol=1e-12, atol=1e-15)
    require(max(errors) <= 2e-6 and surface["current_area_weights_m2"].sum() < .95*np.pi*100.**2,
            "100m disk was not clipped by original thin cohort")
    close(raw["unclipped_current_disk_area_m2"], np.pi*100.**2, "unclipped 100m disk reference", rtol=1e-15)
    return report


def audit_case(case, parent, observer, parent_terms, face):
    definition = audit_seed_definition(case["seed_definition"], parent, face)
    cid = definition["original_cohort_id"]
    fixed(case, dict(name=f"finite_computational_seed_cohort_{cid}_face_{face}", computational_fixture=True,
        order_eight_twelve_observed_relative_agreement_tolerance=2e-6,
        order_eight_twelve_observed_agreement_pass=True, finite_positive_material_surface_area_prepared=True,
        surrounding_ligament_retained=True, whole_cohort_disconnected_banks_created=False, source_activation_event=None,
        physical_creation_admitted=False, physical_release_admitted=False, physical_contact_admitted=False,
        funding_admitted=False, original_material_memory_changed=False, actual_dynamic_reaction_replaced=False,
        actual_creation_work_debited_from_Genesis_j=0.), "finite computational actual case claims")
    surface, surface_report = audit_finite_seed(case["finite_seed"], parent, normal_xyz=definition["normal_xyz"],
        offset_m=definition["offset_m"], center=definition["seed_center_world_m"], radius=definition["seed_radius_m"],
        cohort_id=cid)
    inserted, insertion_report = audit_bound_insertion(case["source_bound_insertion"], parent, observer,
        parent_terms, definition, case["finite_seed"], surface)
    virtual = case["virtual_work"]
    surface_work = audit_surface_virtual(virtual["surface"], parent, surface, inserted["stress"])
    bulk_work = audit_bulk_virtual(virtual["compact_bulk"], parent, parent_terms, definition, inserted)
    followed = audit_follow(virtual["frozen_material_follow"], parent, surface, definition)
    convergence = audit_convergence(case["quadrature_convergence_records"], case["quadrature_convergence_comparisons"],
        parent, definition, inserted, surface)
    large = audit_large_probe(case["separate_100_m_clipped_surface_only_probe"], parent, definition)
    require(case["finite_seed"]["surface"]["material_sections_binding_sha256"] ==
        case["separate_100_m_clipped_surface_only_probe"]["finite_seed"]["surface"]["material_sections_binding_sha256"],
        "same original plane sections lost radius-independent integrity link")
    return dict(name=case["name"], original_cohort_id=cid, finite_surface=surface_report,
        constrained_insertion=insertion_report, surface_virtual_work=surface_work, bulk_virtual_work=bulk_work,
        frozen_material_follow=followed, convergence=convergence, separate_100m_clipped_surface=large)


FORMAT = "genesis-volume-finite-seed-tied-enrichment-control-0.1"
SCOPE = "source_bound_finite_material_seed_and_tied_bulk_enrichment_not_funded_physical_release"


def audit_fixture(fixture, parent, control_directory=None):
    seal(fixture, "fixture_sha256")
    require(parent.get("checkpoint_sha256") == PARENT_CHECKPOINT_SHA256 and digest(
        {k: v for k, v in parent.items() if k != "checkpoint_sha256"}) == PARENT_CHECKPOINT_SHA256,
        "complete authentic parent pin differs")
    require(parent.get("source_checkpoint_sha256") == FORMATION_SHA256, "authentic molten formation ancestor differs")
    context = dict(format=FORMAT, scope=SCOPE, parent_raw_sha256=PARENT_RAW_SHA256,
        parent_checkpoint_sha256=PARENT_CHECKPOINT_SHA256,
        original_parent_ndof=252, original_cohort_count=800,
        stress_scope="current_accepted_endpoint_full_cohort_elastic_memory_not_last_interval_average",
        stress_representation="inherited_fixed_frame_small_strain_reference_energy_stress_not_finite_Cauchy_pushforward",
        material_chart="x(N,lambda)=N@(lambda*(X+P)+(A-P))",
        constrained_compact_enrichment_modes_per_probe=18, computational_finite_seed_clipping_performed=True,
        physical_Gamma_selected_on_actual_source=False, actual_activation_accepted=False,
        new_Genesis_time_steps=0, new_material_cohorts=0, creation_work_debited_from_Genesis_j=0.,
        physical_creation_admitted=False, physical_release_admitted=False, physical_contact_admitted=False,
        funding_admitted=False, actual_dynamic_reaction_replaced=False, actual_birth_energy_balance_accepted=False,
        material_memory_changed=False, physical_dofs_created_or_released=False, history_reinitialized=False,
        point_one_complete=False)
    fixed(fixture, context, "whole authentic finite seed control context")
    geometry = parent["runtime_state"]["geometry"]
    require(len(geometry["cohorts"]) == 800 and abs(geometry["time_myr"]-.8684) < 1e-14
        and parent["runtime_state"]["accepted_step_sizes_myr"] == [.0001]*4
        and not any(parent["runtime_state"]["fault_history"]["fault_active"]),
        "authentic catalogue, clock, accepted intervals or inactive fault history differ")
    observer, onset = fixture["material_stress_observer"], fixture["actual_onset_prerequisite"]
    observer_report = prior.audit_observer(observer, parent)
    onset_report = prior.audit_onset(onset, parent)
    nucleation = {}
    gates = {}
    for name, law in (("unconfigured", None), ("explicit_computational_law", prior.DECLARED_LAW)):
        packet = fixture["nucleation_"+name]
        nucleation[name] = prior.audit_nucleation(packet, parent, observer, onset, law)
        gates[name] = audit_gated_insertion(fixture["proposed_insertion_"+name], parent, observer, packet)
    parent_terms, parent_report = audit_parent_operator(fixture["parent_operator_reference"], parent)
    cases = fixture["finite_seed_cases"]
    require(type(cases) is list and len(cases) == 3
        and [c["seed_definition"]["original_face_index"] for c in cases] == [0, 40, 79],
        "three original finite seed actual cases differ")
    reports = [audit_case(case, parent, observer, parent_terms, face) for case, face in zip(cases, (0, 40, 79))]
    require(len({case["source_bound_insertion"]["embedded_partition"]["parent_response_binding_sha256"]
        for case in cases}) == 1, "three probes lost same original parent response integrity link")
    require(set(fixture) == set(context)|{"parent_path", "actual_onset_prerequisite", "material_stress_observer",
        "nucleation_unconfigured", "nucleation_explicit_computational_law", "proposed_insertion_unconfigured",
        "proposed_insertion_explicit_computational_law", "parent_operator_reference", "finite_seed_cases", "fixture_sha256"},
        "whole authentic finite seed fixture schema differs")
    if control_directory is not None:
        directory = Path(control_directory)
        before, after = (directory/name for name in ("parent_before.json", "parent_after.json"))
        require(before.read_bytes() == after.read_bytes() and hashlib.sha256(before.read_bytes()).hexdigest() == PARENT_RAW_SHA256,
            "entire accepted planet changed during actual finite seed control")
        copies = {"control_fixture.json": fixture, "material_stress_observer.json": observer,
            "actual_onset_prerequisite.json": onset, "parent_operator_reference.json": fixture["parent_operator_reference"]}
        for name in ("unconfigured", "explicit_computational_law"):
            copies["nucleation_"+name+".json"] = fixture["nucleation_"+name]
            copies["proposed_insertion_"+name+".json"] = fixture["proposed_insertion_"+name]
        copies.update({case["name"]+".json": case for case in cases})
        for name, packet in copies.items():
            require(canonical(json.loads((directory/name).read_bytes())) == canonical(packet), "archived source-bound finite artifact differs: "+name)
    report = dict(format="independent-genesis-volume-finite-seed-compact-audit-0.1", status="PASS", scope=SCOPE,
        producer_imports=False, producer_functions_called=False, public_planetary_replay_performed=False,
        parent_raw_sha256=PARENT_RAW_SHA256, parent_checkpoint_sha256=PARENT_CHECKPOINT_SHA256,
        formation_checkpoint_sha256=FORMATION_SHA256, authentic_parent_full_fields_and_history_checked=True,
        entire_before_after_parent_byte_equality_checked=control_directory is not None,
        time_myr=geometry["time_myr"], cohort_count=800, original_parent_ndof=252, new_Genesis_time_steps=0,
        material_stress_observer=observer_report, inherited_onset=onset_report, actual_nucleation=nucleation,
        actual_source_gated_insertion=gates, original_parent_operator=parent_report, finite_seed_cases=reports,
        new_material_cohorts=0, physical_Gamma_selected=False, physical_creation_admitted=False,
        physical_release_admitted=False, funding_admitted=False, actual_dynamic_reaction_replaced=False,
        actual_birth_energy_balance_accepted=False, point_one_complete=False, fixture_sha256=fixture["fixture_sha256"])
    return {**report, "report_sha256": digest(report)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, default=ROOT/
        "analysis/genesis_volume_creation_validation/evidence/actual_continuation/direct_after_second.json")
    parser.add_argument("--control-directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    parent_bytes = args.parent_checkpoint.read_bytes()
    require(hashlib.sha256(parent_bytes).hexdigest() == PARENT_RAW_SHA256, "immutable genuine parent raw pin differs")
    report = audit_fixture(json.loads(args.fixture.read_bytes()), json.loads(parent_bytes), args.control_directory)
    require(not args.output.exists(), "independent output destination already exists")
    args.output.write_bytes(canonical(report)+b"\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()

