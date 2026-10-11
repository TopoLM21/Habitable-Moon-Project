"""Exact geometric admission of declared three-dimensional material paths.

The cone certificate proves injectivity of the before/after displacement
homotopy on the *whole*
convex reference polyhedral ball, including every material pair and every
time.  A continuous, piecewise differentiable displacement with gradient
norm below one is Lipschitz with constant below one on this convex domain.
Thus two distinct points cannot acquire the same position.  The artificial
inner extension is only a proof device; no inner solid material is created.

The separate tetrahedron certificate concerns literal affine vertex paths.
It must not be applied to a bilinear cone/frustum deformation as though its
tetrahedral endpoint interpolant were the same physical path.  Neither
certificate supplies a contact constitutive law, work, or force acceptance.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as F
from itertools import combinations
from math import comb, isfinite
from numbers import Integral, Real
import hashlib
import json


CONE_FORMAT = "genesis-cone-homothetic-whole-clock-injectivity-0.1"
TETRA_FORMAT = "genesis-affine-tetrahedron-whole-clock-separation-0.1"


class UnresolvedGenesisVolumeTemporal(ValueError):
    """The declared full-time geometric admission has not been proved."""


def _f(value):
    if isinstance(value, bool):
        raise ValueError("Temporal coordinates require finite literal real numbers")
    if type(value) is F:
        return value
    if isinstance(value, Integral):
        return F(int(value))
    if isinstance(value, Real) and isfinite(float(value)):
        return F.from_float(float(value))
    raise ValueError("Temporal coordinates require finite literal real numbers")


def _rows(value, width, name):
    try:
        rows = tuple(tuple(_f(x) for x in row) for row in value)
    except TypeError as exc:
        raise ValueError(f"Invalid {name}") from exc
    if not rows or any(len(row) != width for row in rows):
        raise ValueError(f"Invalid {name} dimensions")
    return rows


def _pin(value):
    if (type(value) is not str or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise ValueError("Temporal source identity must be a literal SHA256")
    return value


def _times(before, after):
    before, after = _f(before), _f(after)
    if before < 0 or after <= before:
        raise ValueError("Temporal path requires its positive physical time interval")
    return before, after


def _dot(a, b):
    return sum((x*y for x, y in zip(a, b)), F(0))


def _cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2],
            a[0]*b[1]-a[1]*b[0])


def _sub(a, b):
    return tuple(x-y for x, y in zip(a, b))


def _inverse_columns(v):
    determinant = _dot(v[0], _cross(v[1], v[2]))
    if determinant <= 0:
        raise ValueError("Reference cone face must have strictly positive orientation")
    return tuple(tuple(x/determinant for x in row) for row in
                 (_cross(v[1], v[2]), _cross(v[2], v[0]), _cross(v[0], v[1])))


def _nodal_matrix_product(nodal_columns, inverse):
    return tuple(tuple(sum((nodal_columns[k][i]*inverse[k][j] for k in range(3)), F(0))
                       for j in range(3)) for i in range(3))


def _squared(matrix):
    return sum((x*x for row in matrix for x in row), F(0))


def _triangle_projected(vertices, normal):
    drop = next(i for i, value in enumerate(normal) if value)
    return tuple(tuple(point[i] for i in range(3) if i != drop) for point in vertices), drop


def _cross2(first, second):
    return first[0]*second[1]-first[1]*second[0]


def _inside_triangle(point, triangle):
    signs = tuple(_cross2(_sub(b, a), _sub(point, a))
                  for a, b in zip(triangle, triangle[1:]+triangle[:1]))
    return all(value >= 0 for value in signs) or all(value <= 0 for value in signs)


def _triangle_interiors_overlap(first, second):
    for triangle, other in ((first, second), (second, first)):
        orientation = _cross2(_sub(triangle[1], triangle[0]), _sub(triangle[2], triangle[0]))
        for a, b in zip(triangle, triangle[1:]+triangle[:1]):
            signs = tuple(_cross2(_sub(b, a), _sub(point, a))*orientation for point in other)
            if all(value <= 0 for value in signs):
                return False
    return True


def _convex_closed_faces(vertices, faces):
    if len(vertices) < 4 or len(set(vertices)) != len(vertices):
        raise ValueError("Reference polyhedral ball needs distinct nondegenerate vertices")
    canonical, edges, facet_groups = set(), {}, {}
    vertex_links = {i: [] for i in range(len(vertices))}
    used = set()
    for face in faces:
        if (len(face) != 3 or any(type(i) is not int or not 0 <= i < len(vertices) for i in face)
                or len(set(face)) != 3 or tuple(sorted(face)) in canonical):
            raise ValueError("Reference faces must be unique oriented triangles")
        canonical.add(tuple(sorted(face)))
        used.update(face)
        v = tuple(vertices[i] for i in face)
        _inverse_columns(v)
        normal = _cross(_sub(v[1], v[0]), _sub(v[2], v[0]))
        offset = _dot(normal, v[0])
        if any(_dot(normal, point) > offset for point in vertices):
            raise ValueError("Reference polyhedral ball is not convex")
        triangle, drop = _triangle_projected(v, normal)
        for index, point in enumerate(vertices):
            if (index not in face and _dot(normal, point) == offset
                    and _inside_triangle(tuple(point[j] for j in range(3) if j != drop), triangle)):
                raise ValueError("Reference boundary contains a geometric T-junction or overlapping facet")
        scale = next(value for value in normal if value)
        plane_key = tuple(value/scale for value in normal)+(offset/scale,)
        facet_groups.setdefault(plane_key, []).append(triangle)
        for i, a in enumerate(face):
            vertex_links[a].append((face[(i+1)%3], face[(i+2)%3]))
        for a, b in zip(face, face[1:]+face[:1]):
            edges.setdefault(tuple(sorted((a, b))), []).append((a, b))
    if used != set(range(len(vertices))):
        raise ValueError("Reference polyhedral ball has unused vertices")
    if (any(len(rows) != 2 or rows[0] != rows[1][::-1] for rows in edges.values())
            or len(vertices)-len(edges)+len(faces) != 2):
        raise ValueError("Reference boundary is not a closed oriented sphere")
    adjacency = {i: set() for i in range(len(vertices))}
    for a, b in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)
    reached, pending = set(), [0]
    while pending:
        current = pending.pop()
        if current not in reached:
            reached.add(current)
            pending.extend(adjacency[current]-reached)
    if len(reached) != len(vertices):
        raise ValueError("Reference polyhedral boundary is disconnected")
    for link_edges in vertex_links.values():
        link = {}
        for a, b in link_edges:
            link.setdefault(a, []).append(b)
            link.setdefault(b, []).append(a)
        if any(len(adjacent) != 2 or len(set(adjacent)) != 2 for adjacent in link.values()):
            raise ValueError("Reference vertex link is not a simple manifold cycle")
        start = next(iter(link))
        reached, pending = set(), [start]
        while pending:
            current = pending.pop()
            if current not in reached:
                reached.add(current)
                pending.extend(set(link[current])-reached)
        if len(reached) != len(link):
            raise ValueError("Reference vertex link has disconnected cycles")
    for triangles in facet_groups.values():
        if any(_triangle_interiors_overlap(first, second) for first, second in combinations(triangles, 2)):
            raise ValueError("Reference coplanar triangle interiors overlap")


@dataclass(frozen=True, slots=True)
class ConeHomotheticCertificate:
    source_identity: str
    before_time_myr: F
    after_time_myr: F
    vertices_xyz: tuple
    faces: tuple
    displacement_A_xyz: tuple
    slope_P_xyz: tuple
    before_displacement_A_xyz: tuple
    before_slope_P_xyz: tuple
    lambda_min: F
    face_gradient_bounds_squared: tuple
    max_gradient_frobenius_squared: F

    @property
    def positive_jacobian_whole_interval_certified(self):
        return True

    @property
    def whole_clock_all_pair_noninterpenetration_certified(self):
        return True

    @property
    def physical_force_work_admission(self):
        return False


def certify_cone_homothetic_path(vertices_xyz, faces, displacement_A_xyz,
        slope_P_xyz, *, lambda_min, source_identity,
        before_time_myr=0., after_time_myr=1.,
        before_displacement_A_xyz=None, before_slope_P_xyz=None):
    """Prove the literal homothetic cone field injective at all times.

    On cone ``x=lambda*sum(N_i*v_i)``, ``sum(N)=1``, the material displacement
    is ``sum(N_i*(A_i+(lambda-1)*P_i))`` for ``lambda>=lambda_min``.  The
    physical path interpolates the before/after *total* displacements relative
    to this same reference. Missing before fields mean a literal zero field.
    Below
    that globally common threshold extend linearly to zero at the origin.
    Shared A/P node values make both fields continuous across all cone faces.
    Every actual solid material point must lie in the declared outer domain;
    assigning material, mass, or ownership is the caller's separate contract.
    """
    source_identity = _pin(source_identity)
    before, after = _times(before_time_myr, after_time_myr)
    vertices = _rows(vertices_xyz, 3, "reference vertices")
    A, P = _rows(displacement_A_xyz, 3, "nodal A"), _rows(slope_P_xyz, 3, "nodal P")
    zeros = ((F(0), F(0), F(0)),)*len(vertices)
    before_A = zeros if before_displacement_A_xyz is None else _rows(
        before_displacement_A_xyz, 3, "before nodal A")
    before_P = zeros if before_slope_P_xyz is None else _rows(before_slope_P_xyz, 3, "before nodal P")
    if len(A) != len(vertices) or len(P) != len(vertices):
        raise ValueError("Continuous cone field requires one shared A/P value per source vertex")
    if len(before_A) != len(vertices) or len(before_P) != len(vertices):
        raise ValueError("Before field requires one shared A/P value per source vertex")
    try:
        faces = tuple(tuple(int(i) if isinstance(i, Integral) and not isinstance(i, bool) else i
                            for i in face) for face in faces)
    except TypeError as exc:
        raise ValueError("Invalid reference faces") from exc
    if not faces:
        raise ValueError("Reference polyhedral ball needs its whole boundary")
    lambda_min = _f(lambda_min)
    if not 0 < lambda_min <= 1:
        raise ValueError("Global inner extension threshold must be in (0,1]")
    _convex_closed_faces(vertices, faces)
    bounds = []
    for face in faces:
        v = tuple(vertices[i] for i in face)
        inverse = _inverse_columns(v)
        q = tuple(sum((row[j] for row in inverse), F(0)) for j in range(3))
        endpoint_bounds = []
        for nodes_A, nodes_P in ((before_A, before_P), (A, P)):
            a, p = (tuple(rows[i] for i in face) for rows in (nodes_A, nodes_P))
            difference = tuple(_sub(ai, pi) for ai, pi in zip(a, p))
            H, K = _nodal_matrix_product(p, inverse), _nodal_matrix_product(difference, inverse)
            outer = []
            for corner in difference:
                for lam in (lambda_min, F(1)):
                    gradient = tuple(tuple(H[i][j]+(K[i][j]-corner[i]*q[j])/lam
                                           for j in range(3)) for i in range(3))
                    outer.append(_squared(gradient))
            inner_columns = tuple(tuple((ai+(lambda_min-1)*pi)/lambda_min
                                        for ai, pi in zip(av, pv)) for av, pv in zip(a, p))
            inner = _squared(_nodal_matrix_product(inner_columns, inverse))
            endpoint_bounds.extend((max(outer), inner))
        bounds.append(tuple(endpoint_bounds))
    maximum = max(value for row in bounds for value in row)
    if maximum >= 1:
        raise UnresolvedGenesisVolumeTemporal(
            "Whole-domain displacement gradient norm below one was not proved")
    return ConeHomotheticCertificate(source_identity, before, after, vertices, faces,
                                    A, P, before_A, before_P, lambda_min, tuple(bounds), maximum)


def _r(value):
    return [value.numerator, value.denominator]


def _cone_body(c):
    return dict(format=CONE_FORMAT, source_identity=c.source_identity,
        before_time_myr=_r(c.before_time_myr), after_time_myr=_r(c.after_time_myr),
        vertices_xyz=[list(map(_r, row)) for row in c.vertices_xyz],
        faces=[list(row) for row in c.faces],
        displacement_A_xyz=[list(map(_r, row)) for row in c.displacement_A_xyz],
        slope_P_xyz=[list(map(_r, row)) for row in c.slope_P_xyz],
        before_displacement_A_xyz=[list(map(_r, row)) for row in c.before_displacement_A_xyz],
        before_slope_P_xyz=[list(map(_r, row)) for row in c.before_slope_P_xyz],
        lambda_min=_r(c.lambda_min),
        face_gradient_bounds_squared=[list(map(_r, row)) for row in c.face_gradient_bounds_squared],
        max_gradient_frobenius_squared=_r(c.max_gradient_frobenius_squared),
        reference_domain="whole_convex_origin_star_polyhedral_ball",
        physical_path="X+(1-t)*u_before(X)+t*u_after(X), 0<=t<=1; each outer u=sum(N*(A+(lambda-1)*P))",
        inner_extension="lambda/lambda_min*sum(N*(A+(lambda_min-1)*P))",
        proof="continuous_field_on_convex_domain_Lipschitz_constant_below_one",
        gradient_bound="exact_Frobenius_squared_at_all_N_corners_radial_endpoints_and_time_endpoints",
        positive_jacobian_whole_interval_certified=True,
        whole_clock_all_pair_noninterpenetration_certified=True,
        same_owner_and_adjacent_material_pairs_included=True,
        artificial_inner_extension_creates_material=False,
        physical_force_work_admission=False)


def _encoded(body):
    return json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def cone_homothetic_to_dict(certificate):
    if type(certificate) is not ConeHomotheticCertificate:
        raise ValueError("Expected a complete cone temporal certificate")
    proved = certify_cone_homothetic_path(certificate.vertices_xyz, certificate.faces,
        certificate.displacement_A_xyz, certificate.slope_P_xyz,
        lambda_min=certificate.lambda_min, source_identity=certificate.source_identity,
        before_time_myr=certificate.before_time_myr, after_time_myr=certificate.after_time_myr,
        before_displacement_A_xyz=certificate.before_displacement_A_xyz,
        before_slope_P_xyz=certificate.before_slope_P_xyz)
    if proved != certificate:
        raise ValueError("Cone temporal certificate differs from complete exact replay")
    body = _cone_body(proved)
    return dict(body, payload_sha256=hashlib.sha256(_encoded(body)).hexdigest())


def _decode_fraction(value):
    if (type(value) is not list or len(value) != 2 or any(type(x) is not int for x in value)
            or value[1] <= 0):
        raise ValueError("Exact temporal fraction requires its canonical integer pair")
    result = F(*value)
    if _r(result) != value:
        raise ValueError("Temporal fraction is not canonical")
    return result


def cone_homothetic_from_dict(raw):
    if type(raw) is not dict or raw.get("format") != CONE_FORMAT:
        raise ValueError("Invalid cone temporal certificate schema")
    body = {key: value for key, value in raw.items() if key != "payload_sha256"}
    if raw.get("payload_sha256") != hashlib.sha256(_encoded(body)).hexdigest():
        raise ValueError("Cone temporal certificate seal differs")
    try:
        result = certify_cone_homothetic_path(
            tuple(tuple(map(_decode_fraction, row)) for row in raw["vertices_xyz"]), raw["faces"],
            tuple(tuple(map(_decode_fraction, row)) for row in raw["displacement_A_xyz"]),
            tuple(tuple(map(_decode_fraction, row)) for row in raw["slope_P_xyz"]),
            lambda_min=_decode_fraction(raw["lambda_min"]), source_identity=raw["source_identity"],
            before_time_myr=_decode_fraction(raw["before_time_myr"]),
            after_time_myr=_decode_fraction(raw["after_time_myr"]),
            before_displacement_A_xyz=tuple(tuple(map(_decode_fraction, row))
                                          for row in raw["before_displacement_A_xyz"]),
            before_slope_P_xyz=tuple(tuple(map(_decode_fraction, row))
                                   for row in raw["before_slope_P_xyz"]))
    except (KeyError, TypeError) as exc:
        raise ValueError("Invalid cone temporal certificate fields") from exc
    if _encoded(_cone_body(result)) != _encoded(body):
        raise ValueError("Cone temporal certificate differs from full geometric replay")
    return result


def _trim(poly):
    poly = tuple(poly)
    while len(poly) > 1 and poly[-1] == 0:
        poly = poly[:-1]
    return poly


def _padd(a, b):
    return _trim(tuple((a[i] if i < len(a) else F(0))+(b[i] if i < len(b) else F(0))
                       for i in range(max(len(a), len(b)))))


def _pneg(a):
    return tuple(-x for x in a)


def _psub(a, b):
    return _padd(a, _pneg(b))


def _pmul(a, b):
    result = [F(0)]*(len(a)+len(b)-1)
    for i, av in enumerate(a):
        for j, bv in enumerate(b):
            result[i+j] += av*bv
    return _trim(result)


def _pdot(a, b):
    result = (F(0),)
    for av, bv in zip(a, b):
        result = _padd(result, _pmul(av, bv))
    return result


def _pcross(a, b):
    return (_psub(_pmul(a[1], b[2]), _pmul(a[2], b[1])),
            _psub(_pmul(a[2], b[0]), _pmul(a[0], b[2])),
            _psub(_pmul(a[0], b[1]), _pmul(a[1], b[0])))


def _peval(poly, t):
    value = F(0)
    for coefficient in reversed(poly):
        value = value*t+coefficient
    return value


def _bernstein_bounds(poly, low, high):
    n, width = len(poly)-1, high-low
    power = tuple(sum((poly[k]*comb(k, j)*low**(k-j)*width**j
                       for k in range(j, n+1)), F(0)) for j in range(n+1))
    bernstein = tuple(sum((power[k]*F(comb(i, k), comb(n, k)) for k in range(i+1)), F(0))
                      for i in range(n+1))
    return min(bernstein), max(bernstein)


@dataclass(frozen=True, slots=True)
class TetrahedronPath:
    cell_id: str
    material_id: str
    owner_id: int
    vertex_ids: tuple
    vertices_before_xyz: tuple
    vertices_after_xyz: tuple

    def __post_init__(self):
        if (type(self.cell_id) is not str or not self.cell_id or type(self.material_id) is not str
                or not self.material_id or type(self.owner_id) is not int or self.owner_id < 0):
            raise ValueError("Tetrahedron path needs literal material/cell/owner identities")
        vertex_ids = tuple(self.vertex_ids)
        if (len(vertex_ids) != 4 or len(set(vertex_ids)) != 4
                or any(type(x) is not str or not x for x in vertex_ids)):
            raise ValueError("Tetrahedron path needs four distinct literal node identities")
        before = _rows(self.vertices_before_xyz, 3, "tetrahedron before vertices")
        after = _rows(self.vertices_after_xyz, 3, "tetrahedron after vertices")
        if len(before) != 4 or len(after) != 4:
            raise ValueError("Tetrahedron path needs four vertices at both endpoints")
        object.__setattr__(self, "vertex_ids", vertex_ids)
        object.__setattr__(self, "vertices_before_xyz", before)
        object.__setattr__(self, "vertices_after_xyz", after)


@dataclass(frozen=True, slots=True)
class TetrahedralPathCertificate:
    source_identity: str
    before_time_myr: F
    after_time_myr: F
    cells: tuple
    positive_volume_leaves: tuple
    pair_separation_leaves: tuple
    max_depth: int
    max_time_leaves: int

    @property
    def positive_jacobian_whole_interval_certified(self):
        return True

    @property
    def whole_clock_all_pair_noninterpenetration_certified(self):
        return True

    @property
    def physical_force_work_admission(self):
        return False


def _vertex_polynomials(cell):
    return tuple(tuple(_trim((a, b-a)) for a, b in zip(before, after))
                 for before, after in zip(cell.vertices_before_xyz, cell.vertices_after_xyz))


def _pvector_sub(a, b):
    return tuple(_psub(x, y) for x, y in zip(a, b))


def _tetra_axes(first, second):
    axes = [tuple((F(int(i == j)),) for i in range(3)) for j in range(3)]
    for points in (first, second):
        for face in combinations(range(4), 3):
            axes.append(_pcross(_pvector_sub(points[face[1]], points[face[0]]),
                                _pvector_sub(points[face[2]], points[face[0]])))
    first_edges = tuple(_pvector_sub(first[b], first[a]) for a, b in combinations(range(4), 2))
    second_edges = tuple(_pvector_sub(second[b], second[a]) for a, b in combinations(range(4), 2))
    axes.extend(_pcross(a, b) for a in first_edges for b in second_edges)
    # A polynomial axis can vanish at isolated times: continuity and positive
    # volumes extend separation there. An identically zero axis proves nothing.
    return tuple(axis for axis in axes if any(any(coefficient != 0 for coefficient in p) for p in axis))


def certify_tetrahedron_paths(cells, *, source_identity,
        before_time_myr=0., after_time_myr=1., max_depth=14, max_time_leaves=200000):
    """Certify all unordered cell pairs on their actual straight vertex paths.

    Exact Bernstein convex-hull bounds prove every separation sign on each
    closed time leaf. Shared nodes never skip a pair. Reverse motion has no
    special exclusion: geometric nonpenetration, rather than a slip sign,
    decides this certificate. Constitutive/friction/work admission is external.
    """
    source_identity = _pin(source_identity)
    before, after = _times(before_time_myr, after_time_myr)
    cells = tuple(cells)
    if not cells or any(type(cell) is not TetrahedronPath for cell in cells):
        raise ValueError("Temporal admission requires explicit live tetrahedron paths")
    if len({cell.cell_id for cell in cells}) != len(cells):
        raise ValueError("Tetrahedron cell IDs are not unique")
    if (type(max_depth) is not int or not 0 <= max_depth <= 30
            or type(max_time_leaves) is not int or not 1 <= max_time_leaves <= 1000000):
        raise ValueError("Temporal subdivision needs explicit bounded resources")
    known_nodes = {}
    for cell in cells:
        for node, old, new in zip(cell.vertex_ids, cell.vertices_before_xyz, cell.vertices_after_xyz):
            if node in known_nodes and known_nodes[node] != (old, new):
                raise ValueError("A shared node has inconsistent actual vertex paths")
            known_nodes[node] = (old, new)
    vertices = tuple(_vertex_polynomials(cell) for cell in cells)
    volume_proofs, pair_proofs = [], []
    leaves_used = 0

    def budget(depth):
        nonlocal leaves_used
        leaves_used += 1
        if depth > max_depth or leaves_used > max_time_leaves:
            raise UnresolvedGenesisVolumeTemporal("Exact temporal subdivision budget exhausted")

    for cell, points in zip(cells, vertices):
        edges = tuple(_pvector_sub(points[i], points[0]) for i in (1, 2, 3))
        determinant = _pdot(edges[0], _pcross(edges[1], edges[2]))
        pending, proved = [(F(0), F(1), 0)], []
        while pending:
            low, high, depth = pending.pop()
            budget(depth)
            lower, upper = _bernstein_bounds(determinant, low, high)
            if lower > 0:
                proved.append((low, high, lower, upper))
            elif any(_peval(determinant, t) <= 0 for t in (low, (low+high)/2, high)):
                raise UnresolvedGenesisVolumeTemporal(f"Cell {cell.cell_id} has a nonpositive volume witness")
            else:
                middle = (low+high)/2
                pending.extend(((middle, high, depth+1), (low, middle, depth+1)))
        volume_proofs.append((cell.cell_id, tuple(proved)))
    for i, j in combinations(range(len(cells)), 2):
        first, second = vertices[i], vertices[j]
        aabb_separator = None
        for axis in range(3):
            a = tuple(point[axis] for points in (cells[i].vertices_before_xyz,
                       cells[i].vertices_after_xyz) for point in points)
            b = tuple(point[axis] for points in (cells[j].vertices_before_xyz,
                       cells[j].vertices_after_xyz) for point in points)
            if max(a) <= min(b):
                aabb_separator = (F(0), F(1), axis, 1, min(b)-max(a))
                break
            if max(b) <= min(a):
                aabb_separator = (F(0), F(1), axis, -1, min(a)-max(b))
                break
        if aabb_separator is not None:
            budget(0)
            pair_proofs.append((cells[i].cell_id, cells[j].cell_id, (aabb_separator,)))
            continue
        axes = _tetra_axes(first, second)
        gaps = tuple(tuple(_pdot(axis, _pvector_sub(b, a)) for a in first for b in second)
                     for axis in axes)
        pending, proved = [(F(0), F(1), 0)], []
        while pending:
            low, high, depth = pending.pop()
            budget(depth)
            separator = None
            for axis_index, rows in enumerate(gaps):
                bounds = tuple(_bernstein_bounds(row, low, high) for row in rows)
                if all(lower >= 0 for lower, _ in bounds):
                    separator = (axis_index, 1, min(lower for lower, _ in bounds))
                    break
                if all(upper <= 0 for _, upper in bounds):
                    separator = (axis_index, -1, min(-upper for _, upper in bounds))
                    break
            if separator is not None:
                proved.append((low, high, *separator))
                continue
            middle = (low+high)/2
            point_separated = False
            for axis, rows in zip(axes, gaps):
                if not any(_peval(component, middle) for component in axis):
                    continue
                values = tuple(_peval(row, middle) for row in rows)
                if min(values) >= 0 or max(values) <= 0:
                    point_separated = True
                    break
            if not point_separated:
                raise UnresolvedGenesisVolumeTemporal(
                    f"Cells {cells[i].cell_id}/{cells[j].cell_id} have an interior overlap witness")
            pending.extend(((middle, high, depth+1), (low, middle, depth+1)))
        pair_proofs.append((cells[i].cell_id, cells[j].cell_id, tuple(proved)))
    return TetrahedralPathCertificate(source_identity, before, after, cells,
        tuple(volume_proofs), tuple(pair_proofs), max_depth, max_time_leaves)


def tetrahedron_paths_to_dict(certificate):
    if type(certificate) is not TetrahedralPathCertificate:
        raise ValueError("Expected a complete affine tetrahedron temporal certificate")
    replay = certify_tetrahedron_paths(certificate.cells, source_identity=certificate.source_identity,
        before_time_myr=certificate.before_time_myr, after_time_myr=certificate.after_time_myr,
        max_depth=certificate.max_depth, max_time_leaves=certificate.max_time_leaves)
    if replay != certificate:
        raise ValueError("Tetrahedron temporal certificate differs from complete exact replay")
    body = dict(format=TETRA_FORMAT, source_identity=certificate.source_identity,
        before_time_myr=_r(certificate.before_time_myr), after_time_myr=_r(certificate.after_time_myr),
        cells=[dict(cell_id=cell.cell_id, material_id=cell.material_id, owner_id=cell.owner_id,
            vertex_ids=list(cell.vertex_ids),
            vertices_before_xyz=[list(map(_r, row)) for row in cell.vertices_before_xyz],
            vertices_after_xyz=[list(map(_r, row)) for row in cell.vertices_after_xyz])
            for cell in certificate.cells],
        positive_volume_leaves=[dict(cell_id=cell, leaves=[list(map(_r, row)) for row in leaves])
                                for cell, leaves in certificate.positive_volume_leaves],
        pair_separation_leaves=[dict(first_cell_id=a, second_cell_id=b,
            leaves=[dict(time_lo=_r(low), time_hi=_r(high), axis_index=axis, orientation=sign,
                         lower_projection_gap=_r(gap)) for low, high, axis, sign, gap in leaves])
            for a, b, leaves in certificate.pair_separation_leaves],
        max_depth=certificate.max_depth, max_time_leaves=certificate.max_time_leaves,
        physical_path="literal_affine_vertex_paths_only_not_bilinear_frustum_paths",
        positive_jacobian_whole_interval_certified=True,
        whole_clock_all_pair_noninterpenetration_certified=True,
        same_owner_and_shared_node_pairs_included=True,
        physical_force_work_admission=False)
    return dict(body, payload_sha256=hashlib.sha256(_encoded(body)).hexdigest())


def tetrahedron_paths_from_dict(raw):
    if type(raw) is not dict or raw.get("format") != TETRA_FORMAT:
        raise ValueError("Invalid affine tetrahedron temporal certificate schema")
    body = {key: value for key, value in raw.items() if key != "payload_sha256"}
    if raw.get("payload_sha256") != hashlib.sha256(_encoded(body)).hexdigest():
        raise ValueError("Affine tetrahedron temporal certificate seal differs")
    try:
        cells = tuple(TetrahedronPath(row["cell_id"], row["material_id"], row["owner_id"],
            tuple(row["vertex_ids"]),
            tuple(tuple(map(_decode_fraction, point)) for point in row["vertices_before_xyz"]),
            tuple(tuple(map(_decode_fraction, point)) for point in row["vertices_after_xyz"]))
            for row in raw["cells"])
        result = certify_tetrahedron_paths(cells, source_identity=raw["source_identity"],
            before_time_myr=_decode_fraction(raw["before_time_myr"]),
            after_time_myr=_decode_fraction(raw["after_time_myr"]),
            max_depth=raw["max_depth"], max_time_leaves=raw["max_time_leaves"])
    except (KeyError, TypeError) as exc:
        raise ValueError("Invalid affine tetrahedron temporal certificate fields") from exc
    if _encoded(tetrahedron_paths_to_dict(result)) != _encoded(raw):
        raise ValueError("Affine tetrahedron temporal certificate differs from full geometric replay")
    return result
