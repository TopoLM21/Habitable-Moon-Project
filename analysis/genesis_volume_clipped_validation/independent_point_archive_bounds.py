"""Local coordinate-conversion allowances for literal point archives.

An archived float64 N, lambda or centered offset represents a live coordinate
within EPS64*abs(value).  Error-radius propagation retains thin-cohort t and
radial-gradient sensitivities.  These allowances describe archive arithmetic
only: they neither bound domain integration error nor supply work/funding.
No producer import, coordinate perturbation sweep or positive scale floor is
used.  Frozen source X/A/P/C/Q/e0/e1 are exact inputs to this conversion route.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LD = np.longdouble
EPS = LD(np.finfo(float).eps)


class PointArchiveBoundRefusal(ValueError):
    pass


@dataclass(frozen=True)
class _Radius:
    center: np.ndarray
    radius: np.ndarray

    @classmethod
    def fixed(cls, value):
        a = np.asarray(value, LD)
        return cls(a, np.zeros_like(a))

    @classmethod
    def coordinate(cls, value):
        a = np.asarray(value, LD)
        return cls(a, EPS*np.abs(a))

    def __getitem__(self, key):
        return _Radius(self.center[key], self.radius[key])

    def __add__(self, other):
        other = _as(other)
        return _Radius(self.center+other.center, self.radius+other.radius)

    __radd__ = __add__

    def __neg__(self):
        return _Radius(-self.center, self.radius)

    def __sub__(self, other):
        return self+-_as(other)

    def __rsub__(self, other):
        return _as(other)+-self

    def __mul__(self, other):
        other = _as(other)
        return _Radius(self.center*other.center,
            np.abs(self.center)*other.radius+np.abs(other.center)*self.radius+self.radius*other.radius)

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = _as(other)
        lower = np.abs(other.center)-other.radius
        if np.any(lower <= 0) or not np.isfinite(lower).all():
            raise PointArchiveBoundRefusal("Coordinate-conversion divisor interval contains zero")
        center = self.center/other.center
        radius = (self.radius+np.abs(center)*other.radius)/lower
        return _Radius(center, radius)

    def sum(self, axis):
        return _Radius(self.center.sum(axis=axis), self.radius.sum(axis=axis))

    def transform(self, matrix):
        matrix = np.asarray(matrix, LD)
        return _Radius(self.center@matrix, self.radius@np.abs(matrix))


def _as(value):
    return value if type(value) is _Radius else _Radius.fixed(value)


def _stack(values, axis):
    return _Radius(np.stack([v.center for v in values], axis=axis),
                   np.stack([v.radius for v in values], axis=axis))


def _sym(v, g):
    return _stack((v[:, 0]*g[:, 0], v[:, 1]*g[:, 1], v[:, 2]*g[:, 2],
        v[:, 0]*g[:, 1]+v[:, 1]*g[:, 0], v[:, 0]*g[:, 2]+v[:, 2]*g[:, 0],
        v[:, 1]*g[:, 2]+v[:, 2]*g[:, 1]), 1)


def _source_array(value, shape, name):
    a = np.asarray(value, LD)
    if a.shape != shape or not np.isfinite(a).all():
        raise PointArchiveBoundRefusal(name+" shape/nonfinite source differs")
    return a


def _calculate(domain, elasticity_pa, basis_xyz, memory0, memory1):
    source = domain["geometry_source"]
    X, A, P = (_source_array(source[k], (3, 3), k) for k in
               ("triangle_m", "displacement_A_m", "slope_P_m"))
    Q = _source_array(basis_xyz, (3, 3), "Source frame")
    _source_array(elasticity_pa, (6, 6), "Source elasticity")
    e0, e1 = (_source_array(v, (3, 6), "Complete source memory") for v in (memory0, memory1))
    N = _Radius.coordinate(domain["barycentric"])
    lam = _Radius.coordinate(domain["radial_scale"])
    relative = _Radius.coordinate(domain["centered_birth_offsets_m"])
    count = len(lam.center)
    if N.center.shape != (count, 3) or relative.center.shape != (count, 3) or count == 0:
        raise PointArchiveBoundRefusal("Point coordinate shapes differ")
    b, a = map(LD, source["lambda_bounds"])
    R = LD(domain["seed_radius_m"])
    if not 0 < b < a or not np.isfinite(R) or R <= 0:
        raise PointArchiveBoundRefusal("Original thickness/radius differs")
    determinant = X[0]@np.cross(X[1], X[2])
    if not np.isfinite(determinant) or determinant <= 0:
        raise PointArchiveBoundRefusal("Original reference orientation differs")
    inverse = (np.column_stack((np.cross(X[1], X[2]), np.cross(X[2], X[0]),
                                np.cross(X[0], X[1])))/determinant).T
    grad_lambda = inverse.sum(axis=0)
    gradients = (_as(inverse[None])-N[:, :, None]*grad_lambda[None, None])/lam[:, None, None]
    pgrad = (lam-1)[:, None, None]*gradients+N[:, :, None]*grad_lambda[None, None]
    gcenter, gradius = np.empty((count, 3, 18), LD), np.empty((count, 3, 18), LD)
    Tcenter, Tradius = np.zeros((count, 3, 18), LD), np.zeros((count, 3, 18), LD)
    v = np.zeros((3, 18), LD)
    for vertex in range(3):
        for axis in range(3):
            i = 6*vertex+axis
            gcenter[:, :, i], gradius[:, :, i] = gradients.center[:, vertex], gradients.radius[:, vertex]
            gcenter[:, :, i+3], gradius[:, :, i+3] = pgrad.center[:, vertex], pgrad.radius[:, vertex]
            Tcenter[:, axis, i], Tradius[:, axis, i] = N.center[:, vertex], N.radius[:, vertex]
            slope = (lam-1)*N[:, vertex]
            Tcenter[:, axis, i+3], Tradius[:, axis, i+3] = slope.center, slope.radius
            v[axis, i] = v[axis, i+3] = 1
    g = _Radius(gcenter, gradius)
    localg = _Radius(np.einsum("ij,pjm->pim", Q.T, g.center),
                     np.einsum("ij,pjm->pim", np.abs(Q.T), g.radius))
    localv = _Radius.fixed(np.broadcast_to((Q.T@v)[None], (count, 3, 18)))
    B = _sym(localv, localg)
    T = _Radius(Tcenter, Tradius)
    # Original source sums remain literal float64, as in the archive.
    birthC = np.asarray(np.asarray(X, float)+np.asarray(P, float), LD)
    birthD = np.asarray(np.asarray(A, float)-np.asarray(P, float), LD)
    t = (lam-b)/(a-b)
    r2 = (relative*relative).sum(1)/(R*R)
    sphere = (1-r2)*(1-r2)
    angular = 27*N[:, 0]*N[:, 1]*N[:, 2]
    radial = 4*t*(1-t)
    w = sphere*angular*radial
    d1 = lam[:, None]*(birthC[1]-birthC[0])[None]+(birthD[1]-birthD[0])[None]
    d2 = lam[:, None]*(birthC[2]-birthC[0])[None]+(birthD[2]-birthD[0])[None]
    dl = N.transform(birthC)
    products = _stack(((d1*relative).sum(1), (d2*relative).sum(1), (dl*relative).sum(1)), 1)
    spheregrad = (-4*(1-r2)/(R*R))[:, None]*products
    angulargrad = _stack((27*N[:, 2]*(N[:, 0]-N[:, 1]),
                          27*N[:, 1]*(N[:, 0]-N[:, 2]), _Radius.fixed(np.zeros(count, LD))), 1)
    radialgrad = _stack((_Radius.fixed(np.zeros(count, LD)), _Radius.fixed(np.zeros(count, LD)),
                         4*(1-2*t)/(a-b)), 1)
    parameter = sphere[:, None]*radial[:, None]*angulargrad
    parameter += angular[:, None]*radial[:, None]*spheregrad
    parameter += angular[:, None]*sphere[:, None]*radialgrad
    gradient = parameter[:, 0, None]*gradients[:, 1]+parameter[:, 1, None]*gradients[:, 2]
    gradient += parameter[:, 2, None]*grad_lambda[None]
    localT = _Radius(np.einsum("ij,pjm->pim", Q.T, T.center),
                     np.einsum("ij,pjm->pim", np.abs(Q.T), T.radius))
    localgrad = gradient.transform(Q)
    extra = _sym(localT, localgrad[:, :, None])
    side = _Radius.fixed(np.asarray(domain["side"], LD))
    E = side[:, None, None]*w[:, None, None]*T
    G = side[:, None, None]*(w[:, None, None]*B+extra)
    samples = np.array(((2/3, 1/6, 1/6), (1/6, 2/3, 1/6), (1/6, 1/6, 2/3)))
    angularcoeff = N.transform(np.asarray(np.linalg.inv(samples), LD))
    elastic = angularcoeff.transform(e0)+angularcoeff.transform(e1)/lam[:, None]
    values = dict(parent_strain_operator=B, enrichment_strain_operator=G,
        parent_trace=T, enrichment_trace=E, point_elastic_strain=elastic,
        bubble=w, bubble_gradient_reference_m_inverse=gradient)
    result = {name:value.radius for name, value in values.items()}
    if any(not np.isfinite(v).all() or np.any(v < 0) for v in result.values()):
        raise PointArchiveBoundRefusal("Coordinate-conversion allowance overflowed")
    return result


def point_archive_conversion_bounds(domain, elasticity_pa, basis_xyz, memory0, memory1):
    """Return local allowances with at most 2048 active interval points.

    Output radii have the same field shapes, but intermediate center/radius
    tables are bounded independently of the full archived quadrature size.
    """
    count = len(domain["radial_scale"])
    if not 1 <= count <= 250000:
        raise PointArchiveBoundRefusal("Point archive bounds require 1..250000 points")
    output = None
    for start in range(0, count, 2048):
        stop = min(start+2048, count)
        local = dict(domain)
        for name in ("barycentric", "radial_scale", "centered_birth_offsets_m", "side"):
            local[name] = domain[name][start:stop]
        result = _calculate(local, elasticity_pa, basis_xyz, memory0, memory1)
        if output is None:
            output = {name:np.empty((count,)+value.shape[1:], LD) for name, value in result.items()}
        for name, value in result.items():
            output[name][start:stop] = value
    return output
