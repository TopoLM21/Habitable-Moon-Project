"""Exact source polynomial preparation before any native projection rounding.

The frozen binary64 birth geometry, vertex operators and angular interpolation
are inputs, not recomputed physical assumptions. This runtime implementation
does not import the independent coefficient/contraction oracle.
"""
from fractions import Fraction

import numpy as np

from .genesis_volume_operand_bounds import DirectedInterval, rational_enclosure


def fraction(value):
    if isinstance(value, (float, np.floating)):
        return Fraction(*value.as_integer_ratio())
    return Fraction(value)


def rational(value):
    value = np.asarray(value)
    return np.array([fraction(v) for v in value.flat], object).reshape(value.shape)


def constant(value):
    return np.asarray(value, object)[None]


def _broadcast(value, shape):
    return np.broadcast_to(value.reshape((len(value),)+(1,)*(len(shape)-value.ndim+1)
                                        + value.shape[1:]), (len(value),)+shape)


def add(a, b):
    shape = np.broadcast_shapes(a.shape[1:], b.shape[1:])
    result = np.full((max(len(a), len(b)),)+shape, Fraction(), object)
    result[:len(a)] += _broadcast(a, shape)
    result[:len(b)] += _broadcast(b, shape)
    return result


def product(a, b):
    shape = np.broadcast_shapes(a.shape[1:], b.shape[1:])
    a, b = _broadcast(a, shape), _broadcast(b, shape)
    result = np.full((len(a)+len(b)-1,)+shape, Fraction(), object)
    for i, left in enumerate(a):
        for j, right in enumerate(b):
            result[i+j] += left*right
    return result


def project(polynomial, basis):
    if (basis.shape[0] == basis.shape[1] and
            np.array_equal(basis, np.eye(basis.shape[0]))):
        return polynomial
    return polynomial @ basis


def enclose(value):
    value = np.asarray(value, object)
    lower, upper = np.empty(value.shape, np.longdouble), np.empty(value.shape, np.longdouble)
    for index in np.ndindex(value.shape):
        bound = rational_enclosure(value[index])
        lower[index], upper[index] = bound.lower, bound.upper
    return DirectedInterval(lower, upper)


def _inverse(matrix):
    # Exact Gauss-Jordan elimination, separate from the oracle's cofactors.
    work = np.column_stack((matrix.copy(), rational(np.eye(3))))
    for col in range(3):
        pivot = next((i for i in range(col, 3) if work[i, col]), None)
        if pivot is None:
            raise ValueError('Singular literal source chart')
        work[[col, pivot]] = work[[pivot, col]]
        work[col] /= work[col, col]
        for row in range(3):
            if row != col:
                work[row] -= work[row, col]*work[col]
    return work[:, 3:]


class RationalFields:
    """Reuse exact source constants; evaluate the unchanged polynomial fields."""

    def __init__(self, source):
        self.source = source
        self.C = rational(source.geometry.birth_C_m)
        self.D = rational(source.geometry.birth_D_m)
        self.center = rational(source.geometry.seed_center_world_m)
        self.radius2 = fraction(source.geometry.seed_radius_m)**2
        self.lower, self.upper = map(fraction, source.section.lambda_bounds)
        self.Q = rational(source.basis_xyz)
        X = rational(source.section.triangle_m)
        self.inverse = _inverse(np.column_stack((X[1]-X[0], X[2]-X[0], X[0])))
        self.B0 = rational(source.literal_vertex_B0)
        self.B1 = rational(source.literal_vertex_B1)
        self.angular_inverse = rational(source.angular_inverse)
        self.memory0, self.memory1 = rational(source.memory0), rational(source.memory1)
        self.determinant = sum((X[0,j]*(X[1,(j+1)%3]*X[2,(j+2)%3]
            - X[1,(j+2)%3]*X[2,(j+1)%3]) for j in range(3)), Fraction())
        self.mass = self.determinant*fraction(source.section.mass_kg)/fraction(source.section.volume_m3)

    def memory(self, N, lam, e0, e1):
        weights = N @ self.angular_inverse
        return add(product(lam, constant(weights @ e0)), constant(weights @ e1))

    def evaluate(self, interval):
        if interval.source is not self.source.geometry:
            raise ValueError('Foreign source interval')
        N = rational(interval.barycentric)
        mid, half = fraction(interval.lambda_mid), fraction(interval.lambda_halfwidth)
        lam = np.array([mid, half], object)
        u, d = N @ self.C, N @ self.D
        offset = np.stack((mid*u+d-self.center, half*u))
        S = add(constant(Fraction(1)), -product(offset, offset).sum(axis=1)/self.radius2)
        sphere = product(S, S)
        width = self.upper-self.lower
        t = np.array([mid-self.lower, half], object)/width
        tbar = np.array([self.upper-mid, -half], object)/width
        radial = 4*product(t, tbar)
        angular = 27*N[0]*N[1]*N[2]
        w = angular*product(radial, sphere)
        derivatives = []
        for i, ag in ((1, 27*N[2]*(N[0]-N[1])), (2, 27*N[1]*(N[0]-N[2]))):
            edge = add(product(lam, constant(self.C[i]-self.C[0])), constant(self.D[i]-self.D[0]))
            ds = -4*product(S, product(edge, offset).sum(axis=1))/self.radius2
            derivatives.append(add(ag*product(radial, sphere), angular*product(radial, ds)))
        du, dv = derivatives
        dl_sphere = -4*product(S, product(constant(u), offset).sum(axis=1))/self.radius2
        dl = angular*add(product(radial, dl_sphere), product(sphere, 4*add(tbar, -t)/width))
        radial_gradient = add(product(lam, dl), -add(N[1]*du, N[2]*dv))
        L = add(add(product(du, constant(self.inverse[0])), product(dv, constant(self.inverse[1]))),
                product(radial_gradient, constant(self.inverse[2])))
        Bhat = add(product(lam, constant(sum(N[i]*self.B0[i] for i in range(3)))),
                   constant(sum(N[i]*self.B1[i] for i in range(3))))
        T = np.full((2, 3, 18), Fraction(), object)
        for i in range(3):
            for j in range(3):
                T[0,j,6*i+j] = N[i]
                T[:,j,6*i+3+j] = np.array([mid-1, half], object)*N[i]
        local_t, local_g = self.Q.T @ T, L @ self.Q
        extra = []
        for i, j in ((0,0),(1,1),(2,2),(0,1),(0,2),(1,2)):
            term = product(local_t[:,i,:], local_g[:,j])
            if i != j:
                term = add(term, product(local_t[:,j,:], local_g[:,i]))
            extra.append(term)
        G = add(product(w, Bhat), np.stack(extra, axis=1))
        return dict(lam=lam, w=w, L=L, Bhat=Bhat, Ghat_without_H=G,
                    T=T, E_without_H=product(w,T), memoryhat=self.memory(N,lam,self.memory0,self.memory1))
