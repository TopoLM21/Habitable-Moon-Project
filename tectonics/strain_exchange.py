"""Experimental strain-gated exchange with a finite compliant pressure store.

Only existing basalt footprints exchange rock-equivalent material. This is
instantaneous passive hydraulic equilibration within a geometric volume quota,
not melting, a conduit-flow solution, subduction, or a fracture mechanism.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import brentq

from .conforming_surface import column_potential, normalize, spherical_measures
from .deformation_exchange import DeformationExchange
from .thermal_budget import _add


@dataclass(frozen=True)
class PressureParameters:
    compliance_m3_pa: float = 1e9
    reference_volume_m3: float = 1e20
    initial_pressure_pa: float = 0.
    max_compliance_strain: float = .05
    exchange_fraction: float = 1.

    def __post_init__(self):
        if any(not math.isfinite(x) for x in asdict(self).values()):
            raise ValueError('Nonfinite pressure parameter')
        if min(self.compliance_m3_pa, self.reference_volume_m3) <= 0:
            raise ValueError('Pressure store requires positive volume and compliance')
        if self.initial_pressure_pa < 0 or not 0 < self.max_compliance_strain <= .1:
            raise ValueError('Outside the small-strain compliant-store domain')
        if not 0 <= self.exchange_fraction <= 1:
            raise ValueError('Invalid strain exchange fraction')
        if self.initial_pressure_pa*self.compliance_m3_pa/self.reference_volume_m3 > self.max_compliance_strain:
            raise ValueError('Initial pressure exceeds compliance domain')


def boundary_zone_fraction(surface, width_m, refinement=2):
    """Initial material band, union of compact kernels about actual shared arcs.

    The material band is advected with its faces. Width is an initial physical
    length, not a number of grid cells. No area/stock is added at junctions.
    Refinement changes integration only, not mechanical/material inventories.
    """
    if not math.isfinite(width_m) or not 0 < width_m < math.pi*surface.params.radius_m/2:
        raise ValueError('Invalid boundary-zone width')
    if isinstance(refinement, bool) or int(refinement) != refinement or not 0 <= refinement <= 5:
        raise ValueError('Invalid boundary-zone quadrature')
    owners = surface.arrays['owner']
    arcs = [(u, v) for a, b, u, v in surface.edges if owners[a] != owners[b]]
    if not arcs:
        return np.zeros(len(surface.faces))
    ends = surface.vertices[np.asarray(arcs)]
    a, b = ends[:, 0], ends[:, 1]
    n = normalize(np.cross(a, b))
    length = np.arctan2(np.linalg.norm(np.cross(a, b), axis=1), np.sum(a*b, axis=1))
    tangent = np.cross(n, a)
    tri = surface.vertices[surface.faces][:, None, :, :]
    for _ in range(refinement):
        x, y, z = np.moveaxis(tri, -2, 0)
        xy, yz, zx = normalize(x+y), normalize(y+z), normalize(z+x)
        tri = np.stack([np.stack(t, axis=-2) for t in
            ((x, xy, zx), (xy, y, yz), (zx, yz, z), (xy, yz, zx))], axis=2).reshape(len(surface.faces), -1, 3, 3)
    x, y, z = np.moveaxis(tri, -2, 0)
    weights = 2*np.arctan2(np.sum(x*np.cross(y, z), axis=-1),
                         1+np.sum(x*y+y*z+z*x, axis=-1))
    points = normalize(tri.sum(axis=-2)).reshape(-1, 3)
    values = np.empty(len(points)); angular_width = width_m/surface.params.radius_m
    for first in range(0, len(points), 256):
        p = points[first:first+256]
        pa, pb, pn, pt = p@a.T, p@b.T, p@n.T, p@tangent.T
        angle = np.arctan2(pt, pa)
        endpoint = np.arccos(np.clip(np.maximum(pa, pb), -1, 1))
        distance = np.where((angle >= 0) & (angle <= length),
                            np.arcsin(np.clip(abs(pn), 0, 1)), endpoint)
        d = np.min(distance, axis=1)/angular_width
        values[first:first+len(p)] = np.maximum(1-d*d, 0)**2
    values = values.reshape(weights.shape)
    return np.sum(weights*values, axis=1)/np.sum(weights, axis=1)


def passive_projection(area, volume, lower, upper, pressure, compliance, rho, gravity, radius):
    """Convex hydraulic equilibration with fixed-area directional volume bounds.

    Minimize U(V)+C*P²/2, P=P_before-sum(V-V_before)/C. The unique pressure
    root gives the KKT solution, rather than spending a guessed amount of heat.
    Work and irreversible loss are independently integrated along the transfer.
    """
    area, volume, lower, upper = (np.asarray(x, float) for x in (area, volume, lower, upper))
    if not (area.shape == volume.shape == lower.shape == upper.shape) or area.ndim != 1:
        raise ValueError('Invalid exchange arrays')
    if any(np.any(~np.isfinite(x)) for x in (area, volume, lower, upper)):
        raise ValueError('Nonfinite exchange bounds')
    if np.any(lower < 0) or np.any(lower > volume) or np.any(upper < volume):
        raise ValueError('Invalid directional volume bounds')
    if np.any((area <= 0) & (upper > 0)) or np.any(area < 0):
        raise ValueError('Missing material footprint')
    if any(not math.isfinite(x) for x in (pressure, compliance, rho, gravity, radius)) or pressure < 0 or min(compliance, rho, gravity, radius) <= 0:
        raise ValueError('Invalid hydraulic parameters')
    def volumes(p):
        h = p/(rho*gravity)
        v = area*(h+h*h/radius+h**3/(3*radius**2))
        return np.clip(v, lower, upper)
    def balance(p):
        return math.fsum([p, -pressure, math.fsum(volumes(p)-volume)/compliance])
    # Check the exact active-set condition at the *stored* state before
    # inverting h(V). Long-double arithmetic avoids an inverse/forward ULP
    # discrepancy spuriously pumping material at an already blocked equilibrium.
    ld = np.longdouble
    depth = np.divide(volume.astype(ld), area.astype(ld), out=np.zeros_like(volume, dtype=ld), where=area > 0)
    old_mu = ld(rho)*ld(gravity)*ld(radius)*np.expm1(np.log1p(3*depth/ld(radius))/3)
    stationary = np.all(((upper == volume) | (old_mu >= ld(pressure))) &
                        ((lower == volume) | (old_mu <= ld(pressure))))
    high = pressure+math.fsum(volume-lower)/compliance
    if stationary:
        end_pressure = pressure
    elif high == 0:
        end_pressure = 0.
    else:
        # Outward rounding of the mathematical bracket, not a wider physical
        # pressure range. At a fully quota-limited return the root is its end.
        high = math.nextafter(high, math.inf)
        end_pressure = brentq(balance, 0., high, xtol=max(1e-8, math.ulp(high)),
                              rtol=4*np.finfo(float).eps)
    target = volume.copy() if stationary else volumes(end_pressure)
    delta = target-volume
    net = math.fsum(delta)
    actual_pressure = pressure-net/compliance
    scale = max(pressure, high, 1.)
    if abs(actual_pressure-end_pressure) > 64*math.ulp(scale) or actual_pressure < 0:
        raise ValueError('Pressure-volume solve unresolved')
    # Integration is independent of subtracting two large stored potentials.
    source_work = (pressure-.5*net/compliance)*net
    def integrals(order):
        nodes, weights = leggauss(order); work = []; loss = []
        for t, w in zip((nodes+1)/2, weights/2):
            h = column_potential(area, volume+t*delta, rho, gravity, radius)[2]
            mu = rho*gravity*h
            work.append(float(w)*math.fsum(mu*delta))
            loss.append(float(w)*math.fsum(((pressure-t*net/compliance)-mu)*delta))
        return math.fsum(work), math.fsum(loss)
    low = integrals(8); placed, heat = integrals(16)
    work_scale = max(abs(source_work), abs(placed), abs(heat), 1.)
    if max(abs(placed-low[0]), abs(heat-low[1])) > 1e-11*work_scale or heat < 0:
        raise ValueError('Hydraulic work/loss is unresolved or nonpassive')
    initial_u = column_potential(area, volume, rho, gravity, radius)[0]
    final_u = column_potential(area, target, rho, gravity, radius)[0]
    defect = math.fsum([*final_u, *(-initial_u), heat, -source_work])
    return target, dict(pressure_pa=actual_pressure, net_volume_m3=net,
        source_work_j=source_work, column_work_j=placed, heat_j=heat,
        work_identity_residual_j=math.fsum([source_work, -placed, -heat]),
        energy_defect_j=defect)


class StrainExchange:
    VERSION = 1

    def __init__(self, coupled, zone_fraction, params=PressureParameters(), provenance=None):
        transfers = (math.fsum(coupled.material.data[k]) for k in ('created_kg', 'recycled_kg'))
        if coupled.thermal.time_myr != 0 or math.fsum(coupled.surface.data['external_work_j']) != 0 or any(transfers):
            raise ValueError('Pressure store requires an explicit zero-time initialization')
        self.coupled = deepcopy(coupled); self.params = params
        self.zone_fraction = np.asarray(zone_fraction, float).copy()
        rho = coupled.surface.params.basalt_density_kg_m3
        if params.reference_volume_m3*rho >= coupled.material.pool_masses()[0]:
            raise ValueError('Pressure-store reference volume exceeds existing mantle inventory')
        energy = .5*params.compliance_m3_pa*params.initial_pressure_pa**2
        self.data = dict(version=self.VERSION, net_volume_m3=[0., 0.],
            initial_energy_j=energy, reservoir_energy_j=[energy, 0.], source_work_j=[0., 0.],
            hydraulic_heat_j=[0., 0.], history=[],
            zone_identity=hashlib.sha256(self.zone_fraction.tobytes()).hexdigest(),
            provenance=deepcopy(provenance or {}),
            initialization='new finite compliant-store energy; no extra mass/heat and no recovered pressure history')
        self._validate()

    def pressure(self):
        return math.fsum([self.params.initial_pressure_pa,
            *(-x/self.params.compliance_m3_pa for x in self.data['net_volume_m3'])])

    def _validate(self):
        self.coupled._validate()
        if self.zone_fraction.shape != (len(self.coupled.surface.faces),) or np.any(~np.isfinite(self.zone_fraction)) or np.any((self.zone_fraction < 0) | (self.zone_fraction > 1)):
            raise ValueError('Invalid material boundary-zone weights')
        if hashlib.sha256(self.zone_fraction.tobytes()).hexdigest() != self.data['zone_identity']:
            raise ValueError('Changed material boundary-zone identity')
        p = self.pressure(); par = self.params
        if p < 0 or p*par.compliance_m3_pa/par.reference_volume_m3 > par.max_compliance_strain:
            raise ValueError('Pressure store outside small-compliance domain')
        net = math.fsum(self.data['net_volume_m3'])
        born = self.coupled.surface.data['material_created_m3'][1]
        returned = self.coupled.surface.data['material_returned_m3'][1]
        net_error = math.fsum([*self.data['net_volume_m3'], *(-x for x in born), *returned])
        if abs(net_error) > 128*math.ulp(max(abs(math.fsum(born)), abs(math.fsum(returned)), 1.)):
            raise ValueError('Pressure withdrawal disagrees with material transfer history')
        if net >= par.reference_volume_m3:
            raise ValueError('Finite pressure-store volume exhausted')
        energy = .5*par.compliance_m3_pa*p*p
        stored = math.fsum(self.data['reservoir_energy_j'])
        tol = 128*math.ulp(max(energy, self.data['initial_energy_j'], 1.))
        if abs(stored-energy) > tol or stored < 0:
            raise ValueError('Pressure store energy disagrees with EOS')
        surface_work = self.coupled.surface.data['external_work_j']
        if math.fsum([*surface_work, *(-x for x in self.data['source_work_j'])]) != 0:
            raise ValueError('Undeclared external mechanical work')

    def _exchange(self, area_before):
        c = self.coupled; s = c.surface; p = s.params; par = self.params
        area = s.geometry(s.vertices)[0]
        old = s.arrays['basalt_volume_m3'].copy()
        # Finite actual area changes, never a jump between imposed Euler poles.
        strain = (area-area_before)/area_before
        quota = old*par.exchange_fraction*self.zone_fraction*strain
        lower = old+np.minimum(quota, 0); upper = old+np.maximum(quota, 0)
        footprint = area*(1-s.arrays['felsic_fraction'])
        target, event = passive_projection(footprint, old, lower, upper,
            self.pressure(), par.compliance_m3_pa, p.basalt_density_kg_m3,
            p.gravity_m_s2, p.radius_m)
        born = np.maximum(target-old, 0); returned = np.maximum(old-target, 0)
        bv = math.fsum(born); rv = math.fsum(returned)
        old_totals = c.material.data['surface_volumes_km3']
        after_return = [old_totals[0], math.fsum(old-returned)/1e9]
        final = [old_totals[0], math.fsum(target)/1e9]
        # The inherited single thermal/material owner carries donor enthalpy and
        # pool-averaged provenance. It rejects exhaustion instead of inventing stock.
        c.material.exchange([0., 0.], [0., rv/1e9], after_return, c.thermal_params)
        c.material.exchange([0., bv/1e9], [0., 0.], final, c.thermal_params)
        keep = np.divide(old-returned, old, out=np.ones_like(old), where=old > 0)
        s.arrays['basalt_age_moment_m3_myr'] *= keep
        s.arrays['basalt_volume_m3'] = target
        _add(s.data['material_created_m3'][1], bv)
        _add(s.data['material_returned_m3'][1], rv)
        _add(s.data['heat_generated_j'], event['heat_j'])
        _add(s.data['external_work_j'], event['source_work_j'])  # internal port to surface
        _add(s.data['energy_defect_j'], event['energy_defect_j'])
        c._deposit_heat(event['heat_j'])
        _add(self.data['net_volume_m3'], event['net_volume_m3'])
        _add(self.data['reservoir_energy_j'], -event['source_work_j'])
        _add(self.data['source_work_j'], event['source_work_j'])
        _add(self.data['hydraulic_heat_j'], event['heat_j'])
        c.thermal = replace(c.thermal, mantle_temperature_k=float(c.material.temperatures(c.thermal_params)[0]))
        s.evaluate(s.vertices)
        event.update(time_myr=c.thermal.time_myr, born_m3=bv, returned_m3=rv,
            positive_quota_m3=math.fsum(np.maximum(quota, 0)),
            negative_quota_m3=-math.fsum(np.minimum(quota, 0)),
            max_area_strain=float(np.max(abs(strain))))
        self.data['history'].append(event)

    def step(self, dt):
        work = deepcopy(self)
        area_before = work.coupled.surface.geometry(work.coupled.surface.vertices)[0]
        work.coupled.step(dt)
        work._exchange(area_before)
        work._validate()
        self.__dict__.update(work.__dict__)

    def advance(self, target):
        if not math.isfinite(target) or target < self.coupled.thermal.time_myr:
            raise ValueError('Invalid strain-exchange target')
        while self.coupled.thermal.time_myr < target:
            self.step(min(self.coupled.options.step_myr, target-self.coupled.thermal.time_myr))

    def diagnostics(self):
        d = self.coupled.diagnostics()
        residual = math.fsum([d['combined_energy_residual_j'],
            *self.data['source_work_j'], *self.data['reservoir_energy_j'], -self.data['initial_energy_j']])
        d.update(combined_energy_residual_j=residual, external_work_j=0.,
            surface_pressure_work_j=math.fsum(self.data['source_work_j']),
            pressure_pa=self.pressure(), reservoir_energy_j=math.fsum(self.data['reservoir_energy_j']),
            initial_reservoir_energy_j=self.data['initial_energy_j'],
            reservoir_net_withdrawal_m3=math.fsum(self.data['net_volume_m3']),
            hydraulic_heat_j=math.fsum(self.data['hydraulic_heat_j']),
            basalt_created_m3=math.fsum(self.coupled.surface.data['material_created_m3'][1]),
            basalt_returned_m3=math.fsum(self.coupled.surface.data['material_returned_m3'][1]),
            scope='strain-gated basalt-equivalent exchange, finite pressure, sensible heat; no melting, slip or fracture')
        return d

    def checkpoint(self):
        return dict(data=deepcopy(self.data), parameters=asdict(self.params),
            zone_fraction=self.zone_fraction.tolist(), coupled=self.coupled.checkpoint())

    @classmethod
    def restore(cls, cp):
        d = deepcopy(cp); obj = cls.__new__(cls)
        if d['data']['version'] != cls.VERSION:
            raise ValueError('Unknown strain-exchange checkpoint')
        obj.data = d['data']; obj.params = PressureParameters(**d['parameters'])
        obj.zone_fraction = np.asarray(d['zone_fraction'])
        obj.coupled = DeformationExchange.restore(d['coupled']); obj._validate()
        return obj

    def save(self, path):
        path = Path(path); path.mkdir(parents=True, exist_ok=False)
        cp = self.checkpoint(); arrays = cp['coupled']['surface'].pop('arrays')
        np.savez_compressed(path/'surface.npz', **{k: np.asarray(v) for k, v in arrays.items()})
        with (path/'meta.json').open('x') as f:
            json.dump(cp, f, indent=2, allow_nan=False)

    @classmethod
    def load(cls, path):
        path = Path(path); cp = json.loads((path/'meta.json').read_text())
        with np.load(path/'surface.npz') as z:
            cp['coupled']['surface']['arrays'] = {k: z[k].tolist() for k in z.files}
        return cls.restore(cp)
