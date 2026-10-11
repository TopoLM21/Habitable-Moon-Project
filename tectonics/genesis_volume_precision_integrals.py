"""Runtime radial values with exact source preparation and bounded mass repair.

Each value is the represented center of a fresh source enclosure. Its outward
distance to that enclosure remains an error, charged to the original allowance.
Unresolved components remain explicit; this module never grants angular or
physical admission. Frozen historical producers and proofs are unchanged.
"""
from dataclasses import dataclass
from fractions import Fraction
import re

import numpy as np

from .genesis_volume_rational_fields import (
    RationalFields, rational, fraction, product, project, enclose,
)
from .genesis_volume_integral_bounds import (
    prepare_integral_bounds, _weighted_moments, _bilinear_integral,
    polynomial_integral, multiply, add, PROBE_SCALARS, VOLUMES,
)
from .genesis_volume_operand_bounds import DirectedInterval, point, rational_enclosure
from .genesis_volume_exact_lambda import prepare_fixed_N_interval

LD = np.longdouble
# Internal trigger for spending exact-arithmetic work, NOT an acceptance
# tolerance. Pointwise 2e-6 errors can exhaust a signed angular sum's budget.
# The angular control still measures the actual summed error against 2e-6.
MASS_REPAIR_TRIGGER = Fraction(1, 2**40)


@dataclass(frozen=True)
class PrecisionLine:
    names: tuple
    values: np.ndarray
    errors: np.ndarray
    enclosure: DirectedInterval
    fallback_names: tuple
    unresolved_names: tuple
    source_binding_sha256: str
    geometry_operand_sha256: str
    interval: object
    angular_accuracy_certified: bool = False
    physical_creation_admitted: bool = False
    available_for_creation_funding: bool = False


def _center(enclosure):
    # Half-sums avoid overflow for finite same-sign endpoints.
    value = enclosure.lower/LD(2)+enclosure.upper/LD(2)
    value = np.where(enclosure.lower == enclosure.upper, enclosure.lower, value)
    return value, enclosure.absolute_error_for(value)


class PreparedPrecisionIntegrals:
    def __init__(self, source, raw, *, added_basis=None, parent_basis=None, probes=(),
                 relative_tolerance=2e-6, max_mass_fallback=64):
        if relative_tolerance != 2e-6:
            raise ValueError('This precision lane retains the original 2e-6 tolerance')
        if type(max_mass_fallback) is not int or not 0 <= max_mass_fallback <= 324:
            raise ValueError('Mass fallback work cap must be between 0 and 324')
        self.base = prepare_integral_bounds(source, raw, added_basis=added_basis,
                                           parent_basis=parent_basis, probes=probes)
        self.source, self.names = source, self.base.names
        self.fields = RationalFields(source)
        self.V = rational(self.base.parent_basis.lower)
        self.W = None if self.base.added_basis is None else rational(self.base.added_basis.lower)
        self.I18 = rational(np.eye(18))
        self.I3 = point(np.eye(3))
        self.det, self.mass = rational_enclosure(self.fields.determinant), rational_enclosure(self.fields.mass)
        self.tolerance = fraction(relative_tolerance)
        self.max_mass_fallback = max_mass_fallback
        self.probes = tuple(dict(name=p['name'], **{k:rational(p[k].lower if k in ('q','a','z')
                    else p[k]) for k in ('q','a','z','e0','e1')}) for p in self.base.probes)

    def evaluate(self, N):
        interval = prepare_fixed_N_interval(self.source.geometry, N)
        geometry = self.base.geometry.evaluate(N, interval)
        if interval is None:
            enclosure = point(np.zeros(len(self.names), LD))
            return self._result(enclosure, (), geometry, interval)
        f = self.fields.evaluate(interval)
        bounded = {name:enclose(value) for name,value in f.items()}
        lam2 = product(f['lam'],f['lam'])
        moments = {s:geometry.moments(16,s) for s in (0,1,2)}
        mass_moments = {s:_weighted_moments(moments[s],enclose(lam2)) for s in (1,2)}
        projected = {}
        def projected_field(name, basis):
            key = (name, id(basis))
            if key not in projected:
                exact = project(f[name],basis)
                projected[key] = exact, enclose(exact)
            return projected[key][1]
        def integrate(left,middle,right,side,mass=False):
            return _bilinear_integral(left,middle,right,mass_moments[side] if mass else moments[side],
                                      self.mass if mass else self.det)
        C = self.base.C
        lowers, uppers = [], []
        def append(values):
            for value in values.values():
                lowers.append(value.lower.reshape(-1)); uppers.append(value.upper.reshape(-1))
        for sign in (-1,1):
            append({'volume':multiply(polynomial_integral(enclose(lam2),geometry.bank_moments(2,sign)),self.det)})
        def blocks(V,W):
            B,G,T,E = (projected_field(name,basis) for name,basis in
                       (('Bhat',V),('Ghat_without_H',W),('T',V),('E_without_H',W)))
            memory = bounded['memoryhat'].reshape((2,6,1))
            return dict(stiffness_cross=integrate(B,C,G,1),stiffness_enriched=integrate(G,C,G,2),
                mass_cross=integrate(T,self.I3,E,1,True),mass_enriched=integrate(E,self.I3,E,2,True),
                enrichment_force_n=integrate(G,C,memory,1)[:,0])
        append(blocks(self.I18,self.I18))
        if self.W is not None:
            append(blocks(self.V,self.W))
        for probe in self.probes:
            Bq,Ga,Gz,Tq,Ea = (projected_field(name,basis) for name,basis in
                (('Bhat',probe['q']),('Ghat_without_H',probe['a']),('Ghat_without_H',probe['z']),
                 ('T',probe['q']),('E_without_H',probe['a'])))
            mean = enclose(self.fields.memory(rational(N),f['lam'],probe['e0'],probe['e1'])).reshape((2,6,1))
            values = dict(cross_parent_work_j=integrate(Bq,C,Gz,1)[0,0],cross_added_work_j=integrate(Ga,C,mean,1)[0,0],
                self_work_j=integrate(Ga,C,Gz,2)[0,0],stiffness_cross_bilinear_j=integrate(Bq,C,Ga,1)[0,0],
                stiffness_self_bilinear_j=integrate(Ga,C,Ga,2)[0,0],mass_cross_bilinear_kg_m2=integrate(Tq,self.I3,Ea,1,True)[0,0],
                mass_self_bilinear_kg_m2=integrate(Ea,self.I3,Ea,2,True)[0,0],
                added_parent_force_n=integrate(bounded['Bhat'],C,Gz,1)[:,0],
                enrichment_force_n=add(integrate(bounded['Ghat_without_H'],C,mean,1)[:,0],integrate(bounded['Ghat_without_H'],C,Gz,2)[:,0]))
            append(values)
        lower, upper = np.concatenate(lowers),np.concatenate(uppers)
        enclosure = DirectedInterval(lower,upper)
        center,error = _center(enclosure)
        failures = self._failures(center,error,tolerance=MASS_REPAIR_TRIGGER)
        fallback = []
        for index in failures:
            match = re.fullmatch(r'(full|projected)\.mass_(cross|enriched)\[(\d+),(\d+)\]',self.names[index])
            if match is None or len(fallback) == self.max_mass_fallback:
                continue
            prefix,kind,i,j = match.groups(); i,j = int(i),int(j)
            V,W = (self.I18,self.I18) if prefix=='full' else (self.V,self.W)
            left = projected[('T' if kind=='cross' else 'E_without_H',id(V if kind=='cross' else W))][0][:,:,i]
            right = projected[('E_without_H',id(W))][0][:,:,j]
            polynomial = product(lam2,product(left,right).sum(axis=1))*self.fields.mass
            exact_moments = geometry.rational_moment_enclosures[1 if kind=='cross' else 2]
            lo=hi=Fraction()
            for coefficient,(a,b) in zip(polynomial,exact_moments):
                ends = coefficient*a,coefficient*b
                lo += min(ends); hi += max(ends)
            lower[index],upper[index] = rational_enclosure(lo).lower,rational_enclosure(hi).upper
            fallback.append(self.names[index])
        return self._result(DirectedInterval(lower,upper),tuple(fallback),geometry,interval)

    def _failures(self,values,errors,*,tolerance=None):
        # Exact comparison: the decision cannot gain a rounded-up allowance.
        tolerance = self.tolerance if tolerance is None else tolerance
        return [i for i,(v,e) in enumerate(zip(values,errors))
                if fraction(e)>tolerance*abs(fraction(v))]

    def _result(self,enclosure,fallback,geometry,interval):
        values,errors = _center(enclosure)
        values.setflags(write=False)
        return PrecisionLine(self.names,values,errors,enclosure,fallback,
            tuple(self.names[i] for i in self._failures(values,errors)),self.source.binding_sha256,
            geometry.line_operand_sha256,interval)


def prepare_precision_integrals(source, raw, **options):
    return PreparedPrecisionIntegrals(source, raw, **options)
