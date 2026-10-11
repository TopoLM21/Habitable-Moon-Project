"""Producer-free radial-line checks of sphere/original-cohort bulk domains.

The integration route holds material N fixed.  A current birth line is affine
in lambda, so its sphere and literal Gamma intersections are solved before
any quadrature.  This deliberately does not use the producer's lambda slices,
disk polygons, event roots, inverse chart, or archived point fields.

Angular refinement uses conservative convex-cell bounds to discover positive
branches, including branches missed by every initial quadrature point.  Its
embedded quadrature errors are numerical convergence observations, not rigorous
contraction error bounds.  General physical accuracy is therefore never
certified by this module.  Failure to resolve a branch is an explicit refusal.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Callable

import numpy as np


class IndependentGeometryRefusal(ValueError):
    """The independent domain or requested numerical convergence is unresolved."""


@dataclass(frozen=True)
class Source:
    X: np.ndarray
    A: np.ndarray
    P: np.ndarray
    C: np.ndarray
    D: np.ndarray
    bounds: tuple[float, float]
    center: np.ndarray
    radius: float
    normal: np.ndarray
    offset: float
    det_X: float


def _array(value, shape, name):
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as error:
        raise IndependentGeometryRefusal(name + " is not numeric") from error
    if result.shape != shape or not np.isfinite(result).all():
        raise IndependentGeometryRefusal(name + " has invalid shape or nonfinite values")
    return result.copy()


def read_source(raw: dict) -> Source:
    """Read literal float64 source sums; never mutate or reinitialize a source."""
    geometry = raw.get("geometry_source", raw)
    X = _array(geometry["triangle_m"], (3, 3), "source triangle")
    A = _array(geometry["displacement_A_m"], (3, 3), "source A")
    P = _array(geometry["slope_P_m"], (3, 3), "source P")
    bounds = _array(geometry["lambda_bounds"], (2,), "original lambda bounds")
    normal = _array(geometry["normal_xyz"], (3,), "literal Gamma normal")
    center = _array(raw["seed_center_world_m"], (3,), "sphere center")
    radius, offset = float(raw["seed_radius_m"]), float(geometry["offset_m"])
    if not (0 < bounds[0] < bounds[1]) or not math.isfinite(radius) or radius <= 0:
        raise IndependentGeometryRefusal("Original radial interval or sphere radius is invalid")
    if not math.isfinite(offset) or abs(float(normal @ normal)-1) > 128*np.finfo(float).eps:
        raise IndependentGeometryRefusal("Literal Gamma normal/offset is invalid")
    det_X = float(np.linalg.det(X))
    if not det_X > 0:
        raise IndependentGeometryRefusal("Original reference chart has nonpositive orientation")
    # These sums are authenticated float64 source operands; extended precision
    # is used only after the identical source arithmetic has been performed.
    C, D = np.asarray(X+P, np.longdouble), np.asarray(A-P, np.longdouble)
    return Source(np.asarray(X, np.longdouble), np.asarray(A, np.longdouble),
                  np.asarray(P, np.longdouble), C, D, tuple(map(float, bounds)),
                  np.asarray(center, np.longdouble), radius,
                  np.asarray(normal, np.longdouble), offset, det_X)


def _line_sphere(source: Source, N, radius=None):
    """Closest-point quadratic solution avoids subtracting a discriminant."""
    N = np.asarray(N, np.longdouble)
    direction, origin = N @ source.C, N @ source.D-source.center
    squared = direction @ direction
    if squared <= 0 or not np.isfinite(squared):
        raise IndependentGeometryRefusal("A birth radial line is collapsed")
    midpoint = -(origin @ direction)/squared
    nearest = origin+midpoint*direction
    r = np.longdouble(source.radius if radius is None else radius)
    if r <= 0:
        return None
    margin = r*r-nearest @ nearest
    if margin <= 0:
        return None
    half = np.sqrt(margin/squared)
    lo, hi = max(np.longdouble(source.bounds[0]), midpoint-half), min(np.longdouble(source.bounds[1]), midpoint+half)
    return (lo, hi) if hi > lo else None


def _side_interval(source: Source, N, interval, side, margin=0.):
    if interval is None:
        return None
    lo, hi = interval
    slope = np.longdouble(side)*((np.asarray(N, np.longdouble) @ source.C) @ source.normal)
    intercept = np.longdouble(side)*((np.asarray(N, np.longdouble) @ source.D) @ source.normal-source.offset)-margin
    if slope > 0:
        lo = max(lo, -intercept/slope)
    elif slope < 0:
        hi = min(hi, -intercept/slope)
    elif intercept < 0:
        return None
    return (lo, hi) if hi > lo else None


def radial_intervals(raw_or_source, barycentric):
    """Return disjoint positive-length intervals, with literal sides -/+1."""
    source = raw_or_source if isinstance(raw_or_source, Source) else read_source(raw_or_source)
    N = _array(barycentric, (3,), "material N")
    if np.min(N) < 0 or abs(N.sum()-1) > 64*np.finfo(float).eps:
        raise IndependentGeometryRefusal("Material N is outside its original triangle")
    interval = _line_sphere(source, N)
    return {side: _side_interval(source, N, interval, side) for side in (-1, 1)}


def _triangle_area(vertices):
    uv = np.asarray(vertices, np.longdouble)[:, 1:]
    a, b = uv[1]-uv[0], uv[2]-uv[0]
    return abs(a[0]*b[1]-a[1]*b[0])/2


def _children(vertices):
    a, b, c = vertices
    ab, bc, ca = (a+b)/2, (b+c)/2, (c+a)/2
    return (np.array((a, ab, ca)), np.array((ab, b, bc)),
            np.array((ca, bc, c)), np.array((ab, bc, ca)))


def _angular_support(source):
    """Conservative barycentric support polygon from an independent adjugate.

    No birth inverse or producer clipping events are used.  Over the original
    lambda interval each matrix Y(lambda) has polynomial cofactors.  Interval
    bounds on center dot cofactor plus R times its norm bound every sphere
    point's N.  A constant translation admits the tighter affine cone bound.
    If interval arithmetic cannot separate a determinant from zero, retain
    the complete original triangle rather than manufacture a small support.
    """
    radius = np.longdouble(source.radius)
    eps = np.finfo(np.longdouble).eps
    lower, upper = np.zeros(3, np.longdouble), np.ones(3, np.longdouble)
    if np.array_equal(source.D[0], source.D[1]) and np.array_equal(source.D[0], source.D[2]):
        determinant = _det(source.C)
        if determinant != 0:
            columns = np.column_stack((np.cross(source.C[1], source.C[2]),
                                       np.cross(source.C[2], source.C[0]),
                                       np.cross(source.C[0], source.C[1])))/determinant
            point = source.center-source.D[0]
            center_v = point @ columns
            center_lambda = center_v.sum()
            lambda_width = radius*np.sqrt(np.sum(columns.sum(axis=1)**2))
            resolution = 512*eps*(np.sum(np.abs(point[:, None]*columns))+abs(center_lambda))
            denlo = max(np.longdouble(source.bounds[0]), center_lambda-lambda_width-resolution)
            denhi = min(np.longdouble(source.bounds[1]), center_lambda+lambda_width+resolution)
            if denhi < denlo:
                return ()
            if denlo > 0:
                width = radius*np.sqrt(np.sum(columns*columns, axis=0))+resolution
                lower = np.maximum(lower, (center_v-width)/denhi)
                upper = np.minimum(upper, (center_v+width)/denlo)
    else:
        mid = np.longdouble(source.bounds[0])+np.longdouble(source.bounds[1]-source.bounds[0])/2
        half = np.longdouble(source.bounds[1]-source.bounds[0])/2
        Y = mid*source.C+source.D
        cofactors = []
        for i, j in ((1, 2), (2, 0), (0, 1)):
            cofactors.append(np.stack((np.cross(Y[i], Y[j]),
                np.cross(source.C[i], Y[j])+np.cross(Y[i], source.C[j]),
                np.cross(source.C[i], source.C[j]))))
        dc = np.zeros(4, np.longdouble)
        for power in range(3):
            dc[power] += Y[0] @ cofactors[0][power]
            dc[power+1] += source.C[0] @ cofactors[0][power]
        powers = half**np.arange(4)
        variation = np.sum(np.abs(dc[1:])*powers[1:])
        determinant_resolution = 1024*eps*np.sum(np.abs(dc)*powers)
        denlo, denhi = dc[0]-variation-determinant_resolution, dc[0]+variation+determinant_resolution
        if denlo > 0:
            for i, cofactor in enumerate(cofactors):
                numerator = cofactor @ source.center
                center = numerator[0]
                spread = np.sum(np.abs(numerator[1:])*powers[1:3])
                cofactor_bound = np.sum(np.abs(cofactor)*powers[:3, None], axis=0)
                spread += radius*np.sqrt(cofactor_bound @ cofactor_bound)
                spread += 1024*eps*np.sum(np.abs(cofactor*source.center)*powers[:3, None])
                lower[i] = max(lower[i], (center-spread)/denhi)
                upper[i] = min(upper[i], (center+spread)/denlo)
    polygon = [row.copy() for row in np.eye(3, dtype=np.longdouble)]
    for coordinate in range(3):
        for threshold, sense in ((lower[coordinate], 1), (upper[coordinate], -1)):
            if not polygon:
                return ()
            output = []
            for a, b in zip(polygon, polygon[1:]+polygon[:1]):
                va, vb = sense*(a[coordinate]-threshold), sense*(b[coordinate]-threshold)
                if va >= 0:
                    output.append(a)
                if (va < 0) != (vb < 0):
                    output.append(a+(b-a)*va/(va-vb))
            polygon = output
    if len(polygon) < 3:
        return ()
    return tuple(np.array((polygon[0], polygon[i], polygon[i+1])) for i in range(1, len(polygon)-1))


def _cell_bounds(source, vertices):
    """Conservative volume envelope and branch test from convex corner bounds.

    Bilinear birth coordinates over a triangle times lambda interval belong
    to the convex hull of its six corners.  Axis-aligned distance is a lower
    bound on distance to that hull, so an outside cell can safely be discarded.
    A representative radial line with radius R+/-epsilon gives outer/inner
    reference-volume intervals common to the entire angular cell.
    """
    vertices = np.asarray(vertices, np.longdouble)
    lo, hi = map(np.longdouble, source.bounds)
    corners = np.vstack((vertices @ (lo*source.C+source.D), vertices @ (hi*source.C+source.D)))
    relative = corners-source.center
    low, high = relative.min(axis=0), relative.max(axis=0)
    distance = np.maximum(np.maximum(low, -high), 0)
    # A longdouble summation allowance protects the strict outside decision.
    scale = np.sum(np.abs(corners), axis=1).max()+np.sum(np.abs(source.center))
    position_error = 64*np.finfo(np.longdouble).eps*scale
    if np.sqrt(distance @ distance) > source.radius+position_error:
        return {side: (0., 0.) for side in (-1, 1)}
    middle = vertices.mean(axis=0)
    middle_corners = np.vstack((middle @ (lo*source.C+source.D), middle @ (hi*source.C+source.D)))
    deviation = corners-np.vstack((np.tile(middle_corners[0], (3, 1)), np.tile(middle_corners[1], (3, 1))))
    epsilon = np.sqrt(np.sum(deviation*deviation, axis=1)).max()+position_error
    plane_error = np.max(np.abs(deviation @ source.normal))+position_error
    outer = _line_sphere(source, middle, source.radius+epsilon)
    inner = _line_sphere(source, middle, source.radius-epsilon)
    area_det = _triangle_area(vertices)*source.det_X
    result = {}
    for side in (-1, 1):
        outer_side = _side_interval(source, middle, outer, side, -plane_error)
        inner_side = _side_interval(source, middle, inner, side, plane_error)
        def volume(interval):
            if interval is None:
                return 0.
            a, b = interval
            # Difference of cubes is evaluated without large nearly-equal terms.
            return float(area_det*(b-a)*(b*b+b*a+a*a)/3)
        result[side] = (volume(inner_side), volume(outer_side))
    return result


MOMENT_NAMES = ("reference_volume_m3", "birth_volume_m3", "birth_u0", "birth_u1", "birth_u2",
                "birth_u00", "birth_u11", "birth_u22", "birth_u01", "birth_u02", "birth_u12")


def _det(matrix):
    return matrix[:, 0] @ np.cross(matrix[:, 1], matrix[:, 2])


def _point_moments(source, N, lam):
    Y = lam*source.C+source.D
    Jr = np.column_stack((lam*(source.X[1]-source.X[0]), lam*(source.X[2]-source.X[0]), N @ source.X))
    Jb = np.column_stack((Y[1]-Y[0], Y[2]-Y[0], N @ source.C))
    dr, db = _det(Jr), _det(Jb)
    if dr <= 0 or db <= 0:
        raise IndependentGeometryRefusal("Independent point chart orientation is nonpositive")
    relative = (N @ Y-source.center)/source.radius
    products = relative[[0, 1, 2, 0, 0, 1]]*relative[[0, 1, 2, 1, 2, 2]]
    return np.r_[dr, db, db*relative, db*products], dr


def _rule(source, vertices, angular_order, radial_order, evaluator, n_extra):
    nodes, weights = np.polynomial.legendre.leggauss(angular_order)
    g, gw = np.asarray((nodes+1)/2, np.longdouble), np.asarray(weights/2, np.longdouble)
    ln, lw = np.polynomial.legendre.leggauss(radial_order)
    ln, lw = np.asarray(ln, np.longdouble), np.asarray(lw, np.longdouble)
    jac = 2*_triangle_area(vertices)
    values = np.zeros((2, len(MOMENT_NAMES)+n_extra), np.longdouble)
    absolute = np.zeros_like(values)
    for s, sw in zip(g, gw):
        for t, tw in zip(g, gw):
            N = (1-s)*vertices[0]+s*(1-t)*vertices[1]+s*t*vertices[2]
            angular_weight = jac*s*sw*tw
            whole = _line_sphere(source, N)
            for j, side in enumerate((-1, 1)):
                interval = _side_interval(source, N, whole, side)
                if interval is None:
                    continue
                a, b = interval
                midpoint, half = (a+b)/2, (b-a)/2
                if evaluator is None:
                    lambdas = midpoint+half*ln
                    Ys = lambdas[:, None, None]*source.C+source.D
                    drs = lambdas*lambdas*source.det_X
                    dbs = np.cross(Ys[:, 1]-Ys[:, 0], Ys[:, 2]-Ys[:, 0]) @ (N @ source.C)
                    if np.any(dbs <= 0):
                        raise IndependentGeometryRefusal("Independent point chart orientation is nonpositive")
                    relative = (np.einsum("i,pij->pj", N, Ys)-source.center)/source.radius
                    products = relative[:, [0, 1, 2, 0, 0, 1]]*relative[:, [0, 1, 2, 1, 2, 2]]
                    moments = np.column_stack((drs, dbs, dbs[:, None]*relative, dbs[:, None]*products))
                    contributions = (angular_weight*half*lw)[:, None]*moments
                    values[j] += np.sum(contributions, axis=0)
                    absolute[j] += np.sum(np.abs(contributions), axis=0)
                    continue
                for node, weight in zip(ln, lw):
                    lam = midpoint+half*node
                    moment, dr = _point_moments(source, N, lam)
                    if evaluator is not None:
                        extra = np.asarray(evaluator(source, N, lam, side), np.longdouble)
                        if extra.shape != (n_extra,) or not np.isfinite(extra).all():
                            raise IndependentGeometryRefusal("Independent contraction evaluator is invalid")
                        moment = np.r_[moment, dr*extra]
                    contribution = angular_weight*half*weight*moment
                    values[j] += contribution
                    absolute[j] += np.abs(contribution)
    return values, absolute


def integrate_radial_lines(raw, *, relative_tolerance=2e-6, max_cells=20000,
                           max_depth=24, angular_orders=(3, 5), radial_order=4,
                           evaluator: Callable | None=None, extra_names=()):
    """Independent positive-branch discovery and own-scale numerical refinement.

    Each component has its own integral-of-absolute-operands scale.  Zero or
    tiny signed cross terms never borrow a bulk energy or unrelated entry.
    Returns explicit rigorous volume envelopes separately from the numerical
    quadrature estimates.  A general contraction certification remains false.
    """
    source = read_source(raw)
    if not 0 < relative_tolerance < .1 or max_cells < 1 or max_depth < 1:
        raise IndependentGeometryRefusal("Invalid independent refinement request")
    low_order, high_order = angular_orders
    if not 1 <= low_order < high_order or radial_order < 1:
        raise IndependentGeometryRefusal("Invalid independent quadrature orders")
    n_extra = len(extra_names)
    shape = (2, len(MOMENT_NAMES)+n_extra)
    total = np.zeros(shape, np.longdouble)
    absolute = np.zeros(shape, np.longdouble)
    error = np.zeros(shape, np.longdouble)
    lower = np.zeros(2, np.longdouble)
    upper = np.zeros(2, np.longdouble)
    heap, sequence, evaluated = [], 0, 0
    # Zero sampled volume in a conservative positive branch is an unresolved
    # branch, rather than an estimate of zero integration error.
    def add(vertices, depth):
        nonlocal sequence, evaluated, total, absolute, error, lower, upper
        bounds = _cell_bounds(source, vertices)
        lo = np.array([bounds[s][0] for s in (-1, 1)], np.longdouble)
        hi = np.array([bounds[s][1] for s in (-1, 1)], np.longdouble)
        if not np.any(hi):
            return
        low_radial = radial_order if evaluator is not None else 3
        high_radial = radial_order+2 if evaluator is not None else 3
        coarse, _ = _rule(source, vertices, low_order, low_radial, evaluator, n_extra)
        fine, own = _rule(source, vertices, high_order, high_radial, evaluator, n_extra)
        observed = 8*np.abs(fine-coarse)
        unresolved = np.where((fine[:, 0] == 0) & (hi > 0), hi, 0)
        observed[:, 0] = np.maximum(observed[:, 0], unresolved)
        total += fine; absolute += own; error += observed; lower += lo; upper += hi
        volume_scale = np.sum(own[:, 0])
        normalized = np.divide(observed, own, out=np.zeros_like(observed), where=own != 0)
        priority = float(np.max(unresolved)+volume_scale*np.max(normalized))
        heapq.heappush(heap, (-priority, sequence, vertices, depth, fine, own, observed, lo, hi))
        sequence += 1; evaluated += 1
    for support_triangle in _angular_support(source):
        add(support_triangle, 0)
    while heap:
        roundoff = 512*np.finfo(float).eps*absolute
        threshold = relative_tolerance*absolute+roundoff
        found_both = np.all(total[:, 0] > 0)
        if found_both and np.all(error <= threshold):
            break
        if evaluated+4 > max_cells:
            ratio = np.divide(error, threshold, out=np.full_like(error, np.inf), where=threshold != 0)
            where = tuple(map(int, np.unravel_index(np.argmax(ratio), ratio.shape)))
            raise IndependentGeometryRefusal("Independent angular branch/contraction refinement exceeded max_cells; worst own-scale error ratio="+str(float(np.max(ratio)))+" component="+str(where)+" values="+str(np.asarray(total[:, :2], float).tolist()))
        _, _, vertices, depth, fine, own, observed, lo, hi = heapq.heappop(heap)
        if depth >= max_depth:
            raise IndependentGeometryRefusal("Independent positive branch or contraction remains unresolved at max_depth")
        total -= fine; absolute -= own; error -= observed; lower -= lo; upper -= hi
        for child in _children(vertices):
            add(child, depth+1)
    if not np.all(total[:, 0] > 0):
        raise IndependentGeometryRefusal("Independent domain has no resolved positive volume on both literal Gamma sides")
    return {
        "format": "independent-original-cohort-radial-line-domain-audit-0.1",
        "moment_names": list(MOMENT_NAMES)+list(extra_names),
        "side_order": [-.5, .5], "values": np.asarray(total, float).tolist(),
        "own_absolute_operand_scales": np.asarray(absolute, float).tolist(),
        "observed_embedded_refinement_errors": np.asarray(np.maximum(error, 0), float).tolist(),
        "reference_volume_envelope_lower_m3": np.asarray(lower, float).tolist(),
        "reference_volume_envelope_upper_m3": np.asarray(upper, float).tolist(),
        "evaluated_angular_cells": evaluated,
        "relative_tolerance": float(relative_tolerance),
        "positive_branches_conservatively_discovered": True,
        "independent_numerical_convergence_verified": True,
        "general_contraction_accuracy_certified": False,
        "physical_creation_admitted": False, "physical_release_admitted": False,
        "conditioning_or_roundoff_available_for_funding": False,
    }


def audit_archived_geometry(raw, independent_report=None, *, relative_tolerance=None):
    """Check literal source-node-domain consistency and independent moments.

    Matching independent numerical estimates does not upgrade the general
    integral error envelope or admit a physical event.
    """
    source = read_source(raw)
    N = np.asarray(raw["barycentric"], np.longdouble)
    lam = np.asarray(raw["radial_scale"], np.longdouble)
    side = np.asarray(raw["side"], float)
    rw = np.asarray(raw["reference_volume_weights_m3"], np.longdouble)
    bw = np.asarray(raw["birth_volume_weights_m3"], np.longdouble)
    centered = np.asarray(raw["centered_birth_offsets_m"], np.longdouble)
    count = len(lam)
    if N.shape != (count, 3) or centered.shape != (count, 3) or any(x.shape != (count,) for x in (side, rw, bw)):
        raise IndependentGeometryRefusal("Archived geometry array shape is inconsistent")
    if count == 0 or not all(np.isfinite(x).all() for x in (N, lam, side, rw, bw, centered)):
        raise IndependentGeometryRefusal("Archived geometry is empty or nonfinite")
    if np.any(N < 0) or np.any(np.abs(N.sum(axis=1)-1) > 64*np.finfo(float).eps):
        raise IndependentGeometryRefusal("Archived barycentric points leave original cohort")
    if np.any(lam < source.bounds[0]) or np.any(lam > source.bounds[1]) or np.any(rw <= 0) or np.any(bw <= 0):
        raise IndependentGeometryRefusal("Archived radial point or volume weight is invalid")
    values = np.zeros((2, len(MOMENT_NAMES)), np.longdouble)
    own = np.zeros_like(values)
    for p in range(count):
        Y = lam[p]*source.C+source.D
        relative = N[p] @ Y-source.center
        # Large world coordinates cannot supply a tiny-seed absolute floor.
        source_arithmetic = 512*np.finfo(np.longdouble).eps*(np.sum(np.abs(N[p, :, None]*Y))+np.sum(np.abs(source.center)))
        # Public float64 N/lambda can be rounded from private integration
        # coordinates.  Propagate those individual serialized operand ULPs;
        # this is a position consistency witness, never a work/error credit.
        dto_arithmetic = 16*np.finfo(float).eps*(np.sum(np.abs(N[p, :, None]*Y))+
            abs(lam[p])*np.sum(np.abs(N[p, :, None]*source.C))+np.sum(np.abs(centered[p])))
        if np.max(np.abs(relative-centered[p])) > source_arithmetic+dto_arithmetic:
            raise IndependentGeometryRefusal("Archived centered birth point differs from raw original source chart")
        if centered[p] @ centered[p] > source.radius**2*(1+512*np.finfo(float).eps):
            raise IndependentGeometryRefusal("Archived quadrature point leaves the actual sphere")
        phi = source.normal @ (source.center+centered[p])-source.offset
        expected = .5 if phi >= 0 else -.5
        if side[p] != expected:
            raise IndependentGeometryRefusal("Archived side differs from the literal source Gamma plane")
        _, dr = _point_moments(source, N[p], lam[p])
        Jb = np.column_stack((Y[1]-Y[0], Y[2]-Y[0], N[p] @ source.C))
        db = _det(Jb)
        if abs(rw[p]*db-bw[p]*dr) > 512*np.finfo(float).eps*(abs(rw[p]*db)+abs(bw[p]*dr)):
            raise IndependentGeometryRefusal("Archived birth/reference volume Jacobian ratio is inconsistent")
        u = centered[p]/source.radius
        products = u[[0, 1, 2, 0, 0, 1]]*u[[0, 1, 2, 1, 2, 2]]
        row = np.r_[rw[p], bw[p], bw[p]*u, bw[p]*products]
        j = 0 if side[p] == -.5 else 1
        values[j] += row; own[j] += np.abs(row)
    if independent_report is None:
        return {"source_node_domain_consistency_verified": True,
                "independent_geometric_moment_agreement_verified": False,
                "moment_names": list(MOMENT_NAMES),
                "archived_moments": np.asarray(values, float).tolist(),
                "own_absolute_operand_scales": np.asarray(own, float).tolist(),
                "general_contraction_accuracy_certified": False,
                "physical_creation_admitted": False, "physical_release_admitted": False}
    report_names = independent_report["moment_names"]
    if report_names[:len(MOMENT_NAMES)] != list(MOMENT_NAMES):
        raise IndependentGeometryRefusal("Independent moment convention differs")
    truth = np.asarray(independent_report["values"], np.longdouble)[:, :len(MOMENT_NAMES)]
    scales = np.asarray(independent_report["own_absolute_operand_scales"], np.longdouble)[:, :len(MOMENT_NAMES)]
    observed = np.asarray(independent_report["observed_embedded_refinement_errors"], np.longdouble)[:, :len(MOMENT_NAMES)]
    tolerance = independent_report["relative_tolerance"] if relative_tolerance is None else relative_tolerance
    budget = observed+tolerance*(own+scales)+512*np.finfo(float).eps*(own+scales)
    if np.any(np.abs(values-truth) > budget):
        raise IndependentGeometryRefusal("Archived geometric moments disagree with independent radial-line integration on their own operand scales")
    return {"source_node_domain_consistency_verified": True,
            "independent_geometric_moment_agreement_verified": True,
            "general_contraction_accuracy_certified": False,
            "physical_creation_admitted": False, "physical_release_admitted": False}


def analytic_ball_moments(radius):
    """Birth-coordinate golden values use centered dimensional moments."""
    r = float(radius)
    if not math.isfinite(r) or r <= 0:
        raise IndependentGeometryRefusal("Analytic ball radius is invalid")
    return {"volume_m3": 4*math.pi*r**3/3,
            "first_m4": [0., 0., 0.],
            "second_m5": (np.eye(3)*(4*math.pi*r**5/15)).tolist()}


def analytic_cap_volume(radius, cap_height):
    """Stable even for a positive cap missed by all coarse sampled nodes."""
    r, h = float(radius), float(cap_height)
    if not math.isfinite(r) or not math.isfinite(h) or r <= 0 or not 0 <= h <= 2*r:
        raise IndependentGeometryRefusal("Analytic cap radius or height is invalid")
    return math.pi*h*h*(r-h/3)


CONTRACTION_NAMES = ("K_cross", "K_self_half", "M_cross", "M_self_half", "prestress_work",
                     "maxwell_work_old_support", "maxwell_work_cross", "maxwell_work_self",
                     "maxwell_energy_change_cross", "maxwell_energy_change_self",
                     "maxwell_dissipation_cross", "maxwell_dissipation_self")


def _inverse_transpose(J, rhs):
    determinant = _det(J)
    if determinant <= 0:
        raise IndependentGeometryRefusal("Contraction chart is not positively oriented")
    # J^-T columns are the cofactor columns; no producer Jacobian solver.
    cofactors = np.column_stack((np.cross(J[:, 1], J[:, 2]),
                                np.cross(J[:, 2], J[:, 0]),
                                np.cross(J[:, 0], J[:, 1])))
    return cofactors @ rhs/determinant


def _engineering(gradient):
    return np.array((gradient[0, 0], gradient[1, 1], gradient[2, 2],
                     gradient[0, 1]+gradient[1, 0],
                     gradient[0, 2]+gradient[2, 0],
                     gradient[1, 2]+gradient[2, 1]), np.longdouble)


def _probe_field(source, N, lam, side, Q, amplitude, *, enriched):
    """Differentiate the literal nodal field directly in reference Cartesian x."""
    amplitude = np.asarray(amplitude, np.longdouble).reshape(3, 6)
    A, P = amplitude[:, :3], amplitude[:, 3:]
    value = N @ (A+(lam-1)*P)
    parameter_gradient = np.column_stack((A[1]-A[0]+(lam-1)*(P[1]-P[0]),
                                         A[2]-A[0]+(lam-1)*(P[2]-P[0]), N @ P))
    Jr = np.column_stack((lam*(source.X[1]-source.X[0]), lam*(source.X[2]-source.X[0]), N @ source.X))
    cartesian_gradient = np.column_stack([_inverse_transpose(Jr, row) for row in parameter_gradient]).T
    if enriched:
        b, a = source.bounds
        t = (lam-b)/(a-b)
        Y = lam*source.C+source.D
        Jb = np.column_stack((Y[1]-Y[0], Y[2]-Y[0], N @ source.C))
        relative = N @ Y-source.center
        rho2 = (relative @ relative)/source.radius**2
        angular, radial = 27*np.prod(N), 4*t*(1-t)
        sphere = (1-rho2)**2
        bubble = sphere*angular*radial
        angular_derivative = np.array((27*N[2]*(N[0]-N[1]), 27*N[1]*(N[0]-N[2]), 0), np.longdouble)
        radial_derivative = np.array((0, 0, 4*(1-2*t)/(a-b)), np.longdouble)
        sphere_derivative = -4*(1-rho2)*(Jb.T @ relative)/source.radius**2
        derivative = sphere*radial*angular_derivative+angular*radial*sphere_derivative+angular*sphere*radial_derivative
        gradient = _inverse_transpose(Jr, derivative)
        H = np.longdouble(side)/2
        cartesian_gradient = H*(bubble*cartesian_gradient+np.outer(value, gradient))
        value = H*bubble*value
    return value, _engineering(Q.T @ cartesian_gradient @ Q)


def make_contraction_evaluator(*, elasticity_pa, basis_xyz, strain0, strain1,
                               density_kg_m3, parent_probe_m, enrichment_probe_m,
                               before_enrichment_memory_m=None, maxwell_ratio=0.):
    """Independent directional K/M/prestress and prospective Maxwell contractions.

    Maxwell time integration is an independent 20-point temporal rule.  Spatial
    old terms cover only this selected support, and never replace the analytic
    full-parent memory.  Cross/self energy changes are formed from small field
    increments rather than subtraction of two huge historical energies.
    """
    C = _array(elasticity_pa, (6, 6), "contraction elasticity")
    Q = _array(basis_xyz, (3, 3), "contraction basis")
    if not np.allclose(C, C.T, rtol=0, atol=64*np.finfo(float).eps*np.max(np.abs(C))) or np.linalg.eigvalsh(C).min() <= 0:
        raise IndependentGeometryRefusal("Contraction elasticity is not symmetric positive definite")
    if np.max(np.abs(Q.T @ Q-np.eye(3))) > 256*np.finfo(float).eps or np.linalg.det(Q) <= 0:
        raise IndependentGeometryRefusal("Contraction basis is not proper orthogonal")
    e0 = _array(strain0, (3, 6), "retained angular memory0")
    e1 = _array(strain1, (3, 6), "retained angular memory1")
    q = _array(parent_probe_m, (18,), "parent contraction probe")
    a = _array(enrichment_probe_m, (18,), "enrichment contraction probe")
    z = np.zeros(18) if before_enrichment_memory_m is None else _array(before_enrichment_memory_m, (18,), "native expanded memory witness")
    density, ratio = float(density_kg_m3), float(maxwell_ratio)
    if not math.isfinite(density) or density <= 0 or not math.isfinite(ratio) or ratio < 0 or ratio > 16:
        raise IndependentGeometryRefusal("Independent density or bounded temporal ratio is invalid")
    C, Q, e0, e1, q, a, z = (np.asarray(v, np.longdouble) for v in (C, Q, e0, e1, q, a, z))
    # Literal float64 sampling points and inversion precede precision promotion.
    angular = np.array(((2/3, 1/6, 1/6), (1/6, 2/3, 1/6), (1/6, 1/6, 2/3)))
    interpolation_inverse = np.asarray(np.linalg.inv(angular), np.longdouble)
    tn, tw = np.polynomial.legendre.leggauss(20)
    times, weights = np.asarray((tn+1)/2, np.longdouble), np.asarray(tw/2, np.longdouble)
    decay = np.exp(-np.longdouble(ratio)*times)
    loading = times if ratio == 0 else -np.expm1(-np.longdouble(ratio)*times)/ratio
    alpha = np.exp(-np.longdouble(ratio))
    decay_change = np.expm1(-np.longdouble(ratio))
    beta = np.longdouble(1) if ratio == 0 else -np.expm1(-np.longdouble(ratio))/ratio
    def evaluate(source, N, lam, side):
        uq, bq = _probe_field(source, N, lam, side, Q, q, enriched=False)
        ua, ga = _probe_field(source, N, lam, side, Q, a, enriched=True)
        _, gz = _probe_field(source, N, lam, side, Q, z, enriched=True)
        interpolation = N @ interpolation_inverse
        old = interpolation @ e0+(interpolation @ e1)/lam
        base = decay[:, None]*old+loading[:, None]*bq
        extra = decay[:, None]*gz+loading[:, None]*ga
        meanbase = weights @ base
        meanextra = weights @ extra
        endbase, endextra = alpha*old+beta*bq, alpha*gz+beta*ga
        delta_base, delta_extra = decay_change*old+beta*bq, decay_change*gz+beta*ga
        def dot(left, right):
            return left @ C @ right
        return np.array((dot(bq, ga), dot(ga, ga)/2,
                         density*(uq @ ua), density*(ua @ ua)/2, dot(ga, old),
                         dot(bq, meanbase), dot(bq, meanextra)+dot(ga, meanbase), dot(ga, meanextra),
                         dot(delta_base, endextra)+dot(old, delta_extra),
                         dot(delta_extra, endextra+gz)/2,
                         2*ratio*np.sum(weights*np.einsum("pa,ab,pb->p", base, C, extra)),
                         ratio*np.sum(weights*np.einsum("pa,ab,pb->p", extra, C, extra))), np.longdouble)
    return evaluate, CONTRACTION_NAMES
