"""Finite prospective plane/sphere seeds in one original material cohort.

This computes an actual clipped integration domain, not a mask on a previously
sampled infinite sheet.  Material N/q and the original six A/P modes are frozen
at preparation.  The finite crack front leaves the surrounding volume intact;
the packet makes no claim that it divides the cohort into independent banks.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
import hashlib
import math
from numbers import Integral

import numpy as np

from .genesis_volume_checkpoint import _canonical
from .genesis_volume_material_sections import (
    CohortMaterialSection, _q_at_lambda, _validate_cohort_section,
    prepare_volume_material_sections,
)
from .genesis_volume_embedded_surface import (
    GenesisVolumeEmbeddedSurface, EmbeddedSurfaceRefusal, _binding, _numeric,
    _order, _edge_endpoint, _metrics, _trace, _current_chart_orientation,
    validate_volume_embedded_surface, volume_embedded_surface_to_dict,
)
from .genesis_volume_operator import _immutable


FORMAT = "genesis-original-material-finite-sphere-seed-0.1"
SCOPE = "prospective_finite_material_seed_no_physical_insertion_or_funding"


class FiniteSeedRefusal(ValueError):
    """The declared finite relative-interior seed cannot be resolved."""


@dataclass(frozen=True, slots=True)
class GenesisVolumeFiniteSeed:
    surface: GenesisVolumeEmbeddedSurface
    seed_center_world_m: np.ndarray
    seed_radius_m: float
    selected_cohort_id: int
    clipping_records: tuple
    binding_sha256: str
    format: str = FORMAT
    scope: str = SCOPE

    def __post_init__(self):
        object.__setattr__(self, "seed_center_world_m", _immutable(self.seed_center_world_m))
        object.__setattr__(self, "clipping_records", tuple(self.clipping_records))

    @property
    def physical_creation_admitted(self):
        return False

    @property
    def physical_release_admitted(self):
        return False

    @property
    def partitions_cohort_into_disconnected_banks(self):
        return False


def _inputs(center, radius, section):
    try:
        center = _numeric(center, "Finite seed center", (3,))
        if isinstance(radius, (bool, np.bool_)) or not isinstance(radius, (int, float, np.integer, np.floating)):
            raise ValueError("Finite seed radius must be a positive finite real")
        radius = float(radius)
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError("Finite seed radius must be a positive finite real")
        _validate_cohort_section(section)
        _current_chart_orientation(section)
    except (ValueError, TypeError, OverflowError) as error:
        raise FiniteSeedRefusal(str(error)) from error
    plane_error = float(section.normal_xyz@center-section.offset_m)
    scale = float(np.sum(np.abs(section.normal_xyz*center))+abs(section.offset_m))
    if abs(plane_error) > 64*np.finfo(float).eps*max(scale, 1.):
        raise FiniteSeedRefusal("Finite seed center must lie on its declared plane")
    # The represented sphere must be larger than arithmetic position resolution.
    if radius <= 64*np.finfo(float).eps*max(float(np.max(np.abs(center))), 1.):
        raise FiniteSeedRefusal("Finite seed radius is below current position resolution")
    if not section.interior_cut:
        raise FiniteSeedRefusal("Finite seed has no positive relative-interior material area")
    return center, radius


# Polynomial coefficients are in a local radial parameter z in [-1,1], using
# long-double arithmetic.  Recursive derivative isolation finds simple roots
# and stationary even roots; no fixed radial sampling can miss a small seed.
def _trim(c):
    c = np.asarray(c, np.longdouble)
    while len(c) > 1 and c[-1] == 0:
        c = c[:-1]
    return c


def _padd(a, b):
    result = np.zeros(max(len(a), len(b)), np.longdouble)
    result[:len(a)] += a
    result[:len(b)] += b
    return _trim(result)


def _pmul(a, b):
    return _trim(np.convolve(np.asarray(a, np.longdouble), np.asarray(b, np.longdouble)))


def _pdot(a, b):
    result = np.zeros(1, np.longdouble)
    for i in range(3):
        result = _padd(result, _pmul(a[:, i], b[:, i]))
    return result


def _pcross(a, b):
    result = np.zeros((len(a)+len(b)-1, 3), np.longdouble)
    for i, j, k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        result[:, i] = np.convolve(a[:, j], b[:, k])-np.convolve(a[:, k], b[:, j])
    return result


def _peval(c, x):
    return np.polynomial.polynomial.polyval(np.longdouble(x), c)


def _roots(c, lo=-1., hi=1.):
    c = _trim(c)
    norm = np.max(np.abs(c))
    if norm == 0:
        return ()  # identically-zero event polynomial needs no domain split
    c = c/norm
    if len(c) == 1:
        return ()
    if len(c) == 2:
        root = -c[0]/c[1]
        return (float(root),) if lo < root < hi else ()
    derivative = np.arange(1, len(c), dtype=np.longdouble)*c[1:]
    stationary = _roots(derivative, lo, hi)
    boundaries = (lo,)+stationary+(hi,)
    roots = []
    tolerance = 128*np.finfo(np.longdouble).eps*float(np.sum(np.abs(c)))
    for x in stationary:
        if abs(_peval(c, x)) <= tolerance:
            roots.append(float(x))
    for a, b in zip(boundaries, boundaries[1:]):
        fa, fb = _peval(c, a), _peval(c, b)
        if fa == 0 or fb == 0 or np.signbit(fa) == np.signbit(fb):
            continue
        left, right = np.longdouble(a), np.longdouble(b)
        for _ in range(90):
            mid = (left+right)/2
            if mid == left or mid == right:
                break
            fm = _peval(c, mid)
            if fm == 0:
                left = right = mid
                break
            if np.signbit(fm) == np.signbit(fa):
                left, fa = mid, fm
            else:
                right = mid
        root = float((left+right)/2)
        if lo < root < hi:
            roots.append(root)
    return tuple(sorted(set(roots)))


def _event_breaks(section, interval, center, radius):
    """All sphere endpoint crossings and supporting-line tangencies."""
    lo, hi = interval.lambda_bounds
    mid, half = np.longdouble(lo)+np.longdouble(hi-lo)/2, np.longdouble(hi-lo)/2
    V = np.asarray(section.triangle_m+section.slope_P_m, np.longdouble)
    W = np.asarray(section.displacement_A_m-section.slope_P_m, np.longdouble)
    C = np.asarray(center, np.longdouble)
    Y = np.stack((mid*V+W-C, half*V))  # polynomial, vertex, xyz
    n = np.asarray(section.normal_xyz, np.longdouble)
    polynomials = []
    for i, j in interval.edge_pairs:
        if i == j:
            B = Y[:, i]
            D = np.ones(1, np.longdouble)
        else:
            # Use the literal normalized plane offset, rather than replacing
            # it by n.center (which can differ by a last arithmetic ULP).
            si, sj = Y[:, i]@n, Y[:, j]@n
            shift = n@C-np.longdouble(section.offset_m)
            si[0] += shift
            sj[0] += shift
            D = si-sj
            B = np.zeros((3, 3), np.longdouble)
            for k in range(3):
                B[:, k] = np.convolve(si, Y[:, j, k])-np.convolve(sj, Y[:, i, k])
        polynomials.append(_padd(_pdot(B, B), -radius*radius*_pmul(D, D)))
    area_normal = _pcross(Y[:, 1]-Y[:, 0], Y[:, 2]-Y[:, 0])
    g = _pdot(area_normal, Y[:, 0])
    projection = area_normal@n
    # Account for the arithmetic difference between the center and literal
    # normalized plane.  The sphere/plane disk is centered at C+delta*n.
    delta = np.longdouble(section.offset_m)-n@C
    g = _padd(g, -delta*projection)
    h = area_normal-projection[:, None]*n
    disk_radius_squared = np.longdouble(radius)**2-delta**2
    polynomials.append(_padd(_pmul(g, g), -disk_radius_squared*_pdot(h, h)))
    roots = set()
    for polynomial in polynomials:
        for z in _roots(polynomial):
            lam = float(mid+half*np.longdouble(z))
            if lo < lam < hi:
                roots.add(lam)
    return (lo,)+tuple(sorted(roots))+(hi,)


def _clip_segment(section, interval, lam, center, radius):
    N0, D0 = _edge_endpoint(section, interval.edge_pairs[0], lam)
    N1, D1 = _edge_endpoint(section, interval.edge_pairs[1], lam)
    Y = lam*(section.triangle_m+section.slope_P_m)+(section.displacement_A_m-section.slope_P_m)
    start = np.asarray(N0@Y-center, np.longdouble)
    delta = np.asarray((N1-N0)@Y, np.longdouble)
    squared = delta@delta
    if squared <= 0 or not np.isfinite(squared):
        raise FiniteSeedRefusal("Finite seed segment chart is collapsed")
    nearest = -(start@delta)/squared
    perpendicular = start+nearest*delta
    margin = np.longdouble(radius)**2-perpendicular@perpendicular
    if margin <= 0:
        return None
    half = np.sqrt(margin/squared)
    lower, upper = max(np.longdouble(0), nearest-half), min(np.longdouble(1), nearest+half)
    if upper <= lower:
        return None
    return N0, N1, D0, D1, float(lower), float(upper)


def _basis(normal):
    axis = np.eye(3)[int(np.argmin(np.abs(normal)))]
    e1 = np.cross(normal, axis)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(normal, e1)


def _cross2(a, b):
    return a[0]*b[1]-a[1]*b[0]


def _triangle_polar(section, lam, center, radius):
    """Exact ray/triangle/disk clipping with all angular topology events."""
    Y = lam*(section.triangle_m+section.slope_P_m)+(section.displacement_A_m-section.slope_P_m)
    # Even a center accepted as arithmetically on-plane has a represented
    # normal residual.  Clip the actual sphere/plane disk, not its full-radius
    # projection (the difference matters near the coordinate resolution).
    n = np.asarray(section.normal_xyz, np.longdouble)
    delta = np.longdouble(section.offset_m)-n@np.asarray(center, np.longdouble)
    squared = np.longdouble(radius)**2-delta**2
    if squared <= 0:
        raise FiniteSeedRefusal("Finite seed has no positive sphere-plane disk area")
    radius = float(np.sqrt(squared))
    e1, e2 = _basis(section.normal_xyz)
    Q = np.column_stack(((Y-center)@e1, (Y-center)@e2))
    permutation = np.arange(3)
    if _cross2(Q[1]-Q[0], Q[2]-Q[0]) < 0:
        permutation = np.array((0, 2, 1))
    polygon = Q[permutation]
    edges = np.roll(polygon, -1, axis=0)-polygon
    angles = {0., 2*math.pi}

    def angle(point):
        if np.linalg.norm(point) > 0:
            angles.add(float(math.atan2(point[1], point[0])%(2*math.pi)))

    for point, edge in zip(polygon, edges):
        angle(point)
        angle(edge)
        angle(-edge)
        a, b, c = edge@edge, 2*(point@edge), point@point-radius*radius
        discriminant = b*b-4*a*c
        if discriminant >= 0:
            for t in ((-b-math.sqrt(discriminant))/(2*a), (-b+math.sqrt(discriminant))/(2*a)):
                if 0 <= t <= 1:
                    angle(point+t*edge)
    matrix = np.column_stack((Q[1]-Q[0], Q[2]-Q[0]))
    inverse = np.linalg.inv(matrix)
    constants = np.array([_cross2(edge, -point) for point, edge in zip(polygon, edges)])

    def clip(theta):
        direction = np.array((math.cos(theta), math.sin(theta)))
        lower, upper = 0., radius
        for edge, c in zip(edges, constants):
            a = _cross2(edge, direction)
            if a > 0:
                lower = max(lower, -c/a)
            elif a < 0:
                upper = min(upper, -c/a)
            elif c < 0:
                return None
        if upper <= lower:
            return None
        return direction, lower, upper

    return tuple(sorted(angles)), clip, Q, inverse


def _prepare(section, center, radius, *, sections, parent_ndof, trace_dofs,
             order, declared_normal, declared_offset):
    center, radius = _inputs(center, radius, section)
    try:
        order = _order(order)
    except ValueError as error:
        raise FiniteSeedRefusal(str(error)) from error
    nodes, weights = np.polynomial.legendre.leggauss(order)
    g, gw = .5*(nodes+1), .5*weights
    names = ("point_patch_indices", "cohort_indices", "cohort_ids", "face_indices",
        "barycentric", "radial_scale", "mass_coordinate_q", "current_position_m",
        "reference_position_m", "current_differentials_m", "reference_differentials_m",
        "current_volume_jacobian_m", "reference_volume_jacobian_m",
        "current_volume_determinant_m3", "reference_volume_determinant_m3",
        "current_to_reference_volume_ratio", "current_normal_xyz", "reference_normal_xyz",
        "current_area_weights_m2", "reference_area_weights_m2", "parent_trace",
        "paired_jump_trace", "trace_dofs")
    result = {name: [] for name in names}
    patch_bounds, patch_kinds, records = [], [], []

    def add(patch, N, lam, derivatives, parameter_weight):
        values = _metrics(section, N, lam, derivatives)
        current, reference, dc, dr, nc, nr, jc, jr, Jc, Jr, detc, detr, ratio = values
        T = _trace(N, lam)
        row = dict(point_patch_indices=patch, cohort_indices=section.cohort_index,
            cohort_ids=section.cohort_id, face_indices=section.face_index,
            barycentric=N, radial_scale=lam, mass_coordinate_q=_q_at_lambda(section, lam),
            current_position_m=current, reference_position_m=reference,
            current_differentials_m=dc, reference_differentials_m=dr,
            current_volume_jacobian_m=Jc, reference_volume_jacobian_m=Jr,
            current_volume_determinant_m3=detc, reference_volume_determinant_m3=detr,
            current_to_reference_volume_ratio=ratio, current_normal_xyz=nc,
            reference_normal_xyz=nr, current_area_weights_m2=parameter_weight*jc,
            reference_area_weights_m2=parameter_weight*jr, parent_trace=T,
            paired_jump_trace=np.concatenate((-T, T), axis=1), trace_dofs=trace_dofs)
        if not (row["current_area_weights_m2"] > 0 and row["reference_area_weights_m2"] > 0
                and np.isfinite((row["current_area_weights_m2"], row["reference_area_weights_m2"])).all()):
            raise FiniteSeedRefusal("Finite seed quadrature area is unresolved")
        if np.linalg.norm(current-center) > radius*(1+2e-12):
            raise FiniteSeedRefusal("Finite seed clipped quadrature point lies outside its sphere")
        for name in names:
            result[name].append(row[name])

    for interval in section.intervals:
        if interval.slice_kind != "segment":
            continue
        breaks = _event_breaks(section, interval, center, radius)
        for lo, hi in zip(breaks, breaks[1:]):
            if _clip_segment(section, interval, lo+.5*(hi-lo), center, radius) is None:
                continue
            patch = len(patch_bounds)
            patch_bounds.append((lo, hi))
            patch_kinds.append("sphere_clipped_lambda_segment")
            records.append(("lambda_segment", interval.edge_pairs, (lo, hi)))
            # Sine mapping removes square-root endpoint behavior at disk tangency.
            for t, wt in zip(g, gw):
                lam = lo+(hi-lo)*.5*(1-math.cos(math.pi*t))
                dlam = (hi-lo)*.5*math.pi*math.sin(math.pi*t)
                clipped = _clip_segment(section, interval, lam, center, radius)
                if clipped is None:
                    raise FiniteSeedRefusal("Finite seed support topology is unresolved inside an analytic radial piece")
                N0, N1, D0, D1, s0, s1 = clipped
                for t_s, wt_s in zip(g, gw):
                    s = s0+(s1-s0)*t_s
                    N = (1-s)*N0+s*N1
                    add(patch, N, lam, (N1-N0, (1-s)*D0+s*D1, "segment"),
                        float(dlam*wt*(s1-s0)*wt_s))
    for constant in section.constant_lambda_sections:
        if constant.location != "interior":
            continue
        lam = constant.lambda_value
        angles, clip, Q, inverse = _triangle_polar(section, lam, center, radius)
        for lo, hi in zip(angles, angles[1:]):
            if clip(lo+.5*(hi-lo)) is None:
                continue
            patch = len(patch_bounds)
            patch_bounds.append((lam, lam))
            patch_kinds.append("sphere_clipped_constant_lambda_triangle")
            records.append(("constant_lambda_triangle", lam, (lo, hi)))
            for t, wt in zip(g, gw):
                theta = lo+(hi-lo)*t
                clipped = clip(theta)
                if clipped is None:
                    raise FiniteSeedRefusal("Finite seed angular support topology is unresolved")
                direction, r0, r1 = clipped
                for t_r, wt_r in zip(g, gw):
                    r = r0+(r1-r0)*t_r
                    uv = inverse@(r*direction-Q[0])
                    N = np.array((1-uv.sum(), uv[0], uv[1]))
                    metric = _metrics(section, N, lam, (None, None, "triangle"))
                    polar_weight = float((hi-lo)*wt*(r1-r0)*wt_r*r)
                    add(patch, N, lam, (None, None, "triangle"), polar_weight/metric[6])
    count = len(result["cohort_ids"])
    if not count:
        raise FiniteSeedRefusal("Finite seed has no positive relative-interior material area")
    shapes = {"barycentric": (3,), "current_position_m": (3,), "reference_position_m": (3,),
        "current_differentials_m": (2, 3), "reference_differentials_m": (2, 3),
        "current_volume_jacobian_m": (3, 3), "reference_volume_jacobian_m": (3, 3),
        "current_normal_xyz": (3,), "reference_normal_xyz": (3,),
        "parent_trace": (3, 18), "paired_jump_trace": (3, 36), "trace_dofs": (18,)}
    integers = {"point_patch_indices", "cohort_indices", "cohort_ids", "face_indices", "trace_dofs"}
    arrays = {name: np.asarray(value, np.int64 if name in integers else float).reshape(
        (count,)+shapes.get(name, ())) for name, value in result.items()}
    surface = GenesisVolumeEmbeddedSurface(material_sections=sections, section_records=(section,),
        material_sections_binding_sha256=sections.binding_sha256 if sections is not None else _binding((section,)),
        source_checkpoint_sha256=sections.source_checkpoint_sha256 if sections is not None else "",
        source_identity=sections.source_identity if sections is not None else "computational_cohort",
        time_myr=sections.time_myr if sections is not None else 0.,
        source_geometry_bound=sections is not None, declared_normal_xyz=np.asarray(declared_normal),
        declared_offset_m=float(declared_offset), parent_ndof=parent_ndof, quadrature_order=order,
        selected_cohort_ids=(section.cohort_id,), selected_cohort_indices=(section.cohort_index,),
        patch_cohort_ids=np.full(len(patch_bounds), section.cohort_id, np.int64),
        patch_kinds=tuple(patch_kinds), patch_lambda_bounds=np.asarray(patch_bounds),
        owner_ids=(section.owner_id,)*count, material_ids=(section.material_id,)*count,
        binding_sha256="", **arrays)
    surface = replace(surface, binding_sha256=_binding(surface))
    value = GenesisVolumeFiniteSeed(surface, center, radius, section.cohort_id, tuple(records), "")
    return replace(value, binding_sha256=_binding(value))


def prepare_cohort_finite_seed(section, seed_center_world_m, seed_radius_m, *, quadrature_order=8):
    """Manufactured one-cohort fixture; no accepted planetary source binding."""
    if type(section) is not CohortMaterialSection:
        raise FiniteSeedRefusal("A prepared original cohort material section is required")
    try:
        return _prepare(section, seed_center_world_m, seed_radius_m, sections=None,
            parent_ndof=18, trace_dofs=np.arange(18), order=quadrature_order,
            declared_normal=section.normal_xyz, declared_offset=section.offset_m)
    except EmbeddedSurfaceRefusal as error:
        raise FiniteSeedRefusal(str(error)) from error


def prepare_volume_finite_seed(geometry, displacement_m, normal_xyz, offset_m,
                              seed_center_world_m, seed_radius_m, *, cohort_id, quadrature_order=8):
    """Clip one explicitly selected ORIGINAL live cohort; never add neighbours."""
    if isinstance(cohort_id, (bool, np.bool_)) or not isinstance(cohort_id, Integral):
        raise FiniteSeedRefusal("Finite seed cohort ID must be an original integer ID")
    matches = [c for c in geometry.cohorts if c.id == cohort_id]
    if len(matches) != 1:
        raise FiniteSeedRefusal("Finite seed cohort ID is outside the original material")
    try:
        declared_normal = _numeric(normal_xyz, "Finite seed declared plane normal", (3,))
        sections = prepare_volume_material_sections(geometry, displacement_m, normal_xyz,
            offset_m, face_indices=(matches[0].face_index,))
        section = next(c for c in sections.cohorts if c.cohort_id == cohort_id)
        face = np.asarray(geometry.faces[section.face_index], np.int64)
        dofs = (6*face[:, None]+np.arange(6)).reshape(18)
        return _prepare(section, seed_center_world_m, seed_radius_m, sections=sections,
            parent_ndof=6*len(geometry.vertices_m), trace_dofs=dofs, order=quadrature_order,
            declared_normal=declared_normal, declared_offset=offset_m)
    except (ValueError, TypeError, OverflowError) as error:
        raise FiniteSeedRefusal(str(error)) from error


def validate_volume_finite_seed(seed):
    if (type(seed) is not GenesisVolumeFiniteSeed or seed.format != FORMAT or seed.scope != SCOPE
            or seed.binding_sha256 != _binding(seed)):
        raise FiniteSeedRefusal("Finite seed differs from its immutable support binding")
    validate_volume_embedded_surface(seed.surface)
    if seed.surface.selected_cohort_ids != (seed.selected_cohort_id,):
        raise FiniteSeedRefusal("Finite seed original material support differs")
    return seed


def volume_finite_seed_to_dict(seed):
    validate_volume_finite_seed(seed)
    body = dict(format=FORMAT, scope=SCOPE, surface=volume_embedded_surface_to_dict(seed.surface),
        seed_center_world_m=seed.seed_center_world_m.tolist(), seed_radius_m=seed.seed_radius_m,
        selected_cohort_id=seed.selected_cohort_id, clipping_records=seed.clipping_records,
        clipping_semantics="analytic_sphere_segment_roots_radial_event_polynomials_and_disk_triangle_rays",
        material_support_semantics="frozen_original_N_q_not_world_sphere_reintersection",
        original_cohort_count=1, finite_front_and_surrounding_material_not_released=True,
        cohort_disconnection_certified=False,
        partitions_cohort_into_disconnected_banks=False, geometry_integrals_exact=False,
        physical_creation_admitted=False, physical_release_admitted=False,
        dofs_created_or_released=False, material_memory_changed=False,
        material_history_reinitialized=False, clock_advanced=False,
        creation_work_debited_from_Genesis_j=0.)
    return {**body, "finite_seed_sha256": hashlib.sha256(_canonical(body)).hexdigest()}


def volume_finite_seed_from_dict(raw, *, geometry, displacement_m, normal_xyz, offset_m,
                          seed_center_world_m, seed_radius_m, cohort_id, quadrature_order=8):
    expected = prepare_volume_finite_seed(geometry, displacement_m, normal_xyz, offset_m,
        seed_center_world_m, seed_radius_m, cohort_id=cohort_id, quadrature_order=quadrature_order)
    try:
        equal = _canonical(raw) == _canonical(volume_finite_seed_to_dict(expected))
    except (ValueError, TypeError, OverflowError) as error:
        raise FiniteSeedRefusal("Finite seed archive is not a finite canonical packet") from error
    if not equal:
        raise FiniteSeedRefusal("Finite seed differs from complete original material clipping reconstruction")
    return expected


# Short aliases retain the identical complete source/geometry reconstruction.
finite_seed_to_dict = volume_finite_seed_to_dict
finite_seed_from_dict = volume_finite_seed_from_dict
