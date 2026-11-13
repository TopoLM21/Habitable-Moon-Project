"""Opt-in finite pseudocomponent melting inventory for the coarse runtime.

This is a fixed phase envelope, not a deep-mantle EOS or a petrological melt
calculation. Only an explicitly tagged part of the existing mantle is fusible.
Both boxes own full enthalpy; selective liquid extraction retains component
depletion. No heat is silently converted to placement work.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import math

import numpy as np
from scipy.optimize import brentq

from .material_exchange import MaterialExchangeBudget
from .deformation_exchange import DeformationExchange
from .strain_exchange import StrainExchange
from .thermal import radiogenic_power_w, tidal_heating_power_w, SECONDS_PER_MYR
from .thermal_budget import _add


@dataclass(frozen=True)
class PhaseParameters:
    # Mandatory explicit synthetic initialization; no inferred lunar solidus.
    melting_temperatures_k: tuple
    mantle_fusible_fraction: float
    latent_heat_j_kg: float = 4e5
    retained_liquid_fraction: float = .02
    phase_atol_fraction: float = 1e-3

    def __post_init__(self):
        t = tuple(float(x) for x in self.melting_temperatures_k)
        object.__setattr__(self, 'melting_temperatures_k', t)
        if not t or any(not math.isfinite(x) or x <= 0 for x in t) or any(b <= a for a, b in zip(t, t[1:])):
            raise ValueError('Melting temperatures must be finite, positive and ordered')
        vals = (self.mantle_fusible_fraction, self.latent_heat_j_kg,
                self.retained_liquid_fraction, self.phase_atol_fraction)
        if not all(math.isfinite(x) for x in vals):
            raise ValueError('Nonfinite phase parameter')
        if not 0 <= self.mantle_fusible_fraction < 1 or self.latent_heat_j_kg <= 0:
            raise ValueError('Invalid finite phase inventory')
        if not 0 <= self.retained_liquid_fraction < 1 or not 0 < self.phase_atol_fraction < 1:
            raise ValueError('Invalid retention or phase accuracy')


def caloric_state(energy, capacity, components, melting, latent):
    """Invert H=C*T+L*sum(liquid masses), including Stefan plateaus.

    All components have the same sensible heat and latent heat, and their own
    fixed melting point. They are effective immiscible constituents; there is
    no assumed equilibrium chemistry or pressure-dependent latent work.
    """
    e = np.longdouble(energy); c = np.longdouble(capacity)
    m = np.asarray(components, dtype=np.longdouble)
    if c <= 0 or e < 0 or np.any(m < 0) or not np.isfinite(e+c+np.sum(m)):
        raise ValueError('Invalid caloric inventory')
    liquid = np.zeros(len(m), dtype=np.longdouble); used = np.longdouble(0.)
    for i, tm in enumerate(melting):
        start = c*tm + latent*used
        if e < start:
            return float((e-latent*used)/c), liquid
        if m[i] and e <= start+latent*m[i]:
            liquid[i] = (e-start)/latent
            return float(tm), liquid
        liquid[i] = m[i]; used += m[i]
    return float((e-latent*used)/c), liquid


def implicit_phase_exchange(energies, capacities, components, phase, conductances,
                            powers, surface_k, dt_myr):
    """Monotone implicit enthalpy exchange with one shared signed basal flux.

    Frozen positive conductances; first order, with step doubling in the owner.
    The scalar root treats partial melting without dividing by a fictitious cp.
    No latent heat correction is subsequently added to either energy ledger.
    """
    h = np.longdouble(dt_myr)*SECONDS_PER_MYR
    if h <= 0 or not np.isfinite(h): raise ValueError('Invalid phase heat step')
    e = np.asarray(energies, dtype=np.longdouble); c = np.asarray(capacities, dtype=np.longdouble)
    gm, gs = np.asarray(conductances, dtype=np.longdouble)
    rm, rc, tide = np.asarray(powers, dtype=np.longdouble)
    if min(gm, gs, rm, rc, tide) < 0: raise ValueError('Negative conductance or source')
    tm = phase.melting_temperatures_k; L = phase.latent_heat_j_kg
    if surface_k >= min(tm): raise ValueError('Surface must lie below the phase envelope')
    em = e[0]+h*(rm+tide); ec = e[1]+h*rc
    lower = c[1]*surface_k-ec; upper = em-c[0]*surface_k
    if lower > upper: raise ValueError('Heat below the finite surface-temperature domain')
    if gm == 0:
        qb = np.longdouble(0.)
    else:
        initial_t = [caloric_state(e[i], c[i], components[i], tm, L)[0] for i in range(2)]
        # Root tolerance belongs to this exchange, not the enormous total stock.
        scale = max(float(h*gm*abs(initial_t[0]-initial_t[1])),
                    float(h*gs*abs(initial_t[1]-surface_k)), float(h*(rm+rc+tide)), 1.)
        def residual(x):
            q = np.longdouble(x)*scale
            mt = caloric_state(em-q, c[0], components[0], tm, L)[0]
            ct = caloric_state(ec+q+h*gs*surface_k, c[1]+h*gs, components[1], tm, L)[0]
            return float((q-h*gm*(mt-ct))/scale)
        qb = np.longdouble(brentq(residual, float(lower/scale), float(upper/scale),
                                  xtol=5e-16, rtol=1e-15))*scale
    ct = caloric_state(ec+qb+h*gs*surface_k, c[1]+h*gs, components[1], tm, L)[0]
    qs = h*gs*(ct-surface_k)
    q = np.array([h*(rm+rc), h*tide, qs, qb], dtype=np.longdouble)
    rad = h*np.array([rm, rc], dtype=np.longdouble)
    delta = np.array([rad[0]+q[1]-qb, rad[1]+qb-qs], dtype=np.longdouble)
    return e+delta, q, rad


class PhaseMaterialBudget(MaterialExchangeBudget):
    VERSION = 'finite_phase_inventory_1'

    @classmethod
    def from_fresh(cls, base, params, phase):
        d = base.checkpoint()
        if (d['version'] != MaterialExchangeBudget.VERSION or d['time_myr'] != 0 or
            d['accepted_steps'] or math.fsum(d['created_kg']) or math.fsum(d['recycled_kg'])):
            raise ValueError('Phase initialization requires a fresh unexchanged zero-time inventory')
        obj = cls.__new__(cls); obj.options = base.options; obj.phase = phase; obj.data = d
        n = len(phase.melting_temperatures_k)
        origins = np.array(d['origin_hi'])+np.array(d['origin_lo'])
        tags = np.zeros((2, n, 3))
        tags[0] = origins[0]*phase.mantle_fusible_fraction/n
        tags[1] = origins[2]/n
        # Put division remainder back into a recorded component, not new mass.
        for i, total in enumerate((origins[0]*phase.mantle_fusible_fraction, origins[2])):
            for j in range(3): tags[i, -1, j] = math.fsum([float(total[j]), *(-tags[i, :-1, j])])
        d.update(version=cls.VERSION, phase_parameters=dict(asdict(phase), melting_temperatures_k=list(phase.melting_temperatures_k)),
                 phase_origin_hi=tags.tolist(), phase_origin_lo=np.zeros_like(tags).tolist(),
                 initial_phase_origin_kg=tags.sum(axis=0).tolist(),
                 phase_initialization=dict(old_temperature_k=base.temperatures(params).tolist(),
                    energy_added_j=0., mass_added_kg=0.,
                    description='new tagged pseudocomponents at unchanged total enthalpy; not recovered melting history'))
        d['phase_initialization']['new_temperature_k'] = obj.temperatures(params).tolist()
        obj._validate_phase(); return obj

    def _phase_origins(self):
        return np.asarray(self.data['phase_origin_hi'], np.longdouble)+np.asarray(self.data['phase_origin_lo'], np.longdouble)

    def _validate_phase(self):
        tags = self._phase_origins(); n = len(self.phase.melting_temperatures_k)
        if tags.shape != (2, n, 3) or np.any(~np.isfinite(tags)) or np.any(tags < 0):
            raise ValueError('Invalid saved phase components')
        origins = np.asarray(self.data['origin_hi'], np.longdouble)+np.asarray(self.data['origin_lo'], np.longdouble)
        initial = np.asarray(self.data['initial_phase_origin_kg'], np.longdouble)
        if initial.shape != (n, 3) or np.any(~np.isfinite(initial)) or np.any(initial < 0):
            raise ValueError('Invalid phase origin reference')
        bound = 128*np.finfo(float).eps*np.maximum(initial, 1.)
        if np.any(abs(tags.sum(axis=0)-initial) > bound):
            raise ValueError('Phase components or their origins were created/lost')
        bound = 128*np.finfo(float).eps*np.maximum(abs(origins[[0, 2]]), 1.)
        if np.any(tags[0].sum(axis=0)-origins[0] > bound[0]) or np.any(abs(tags[1].sum(axis=0)-origins[2]) > bound[1]):
            raise ValueError('Phase inventory disagrees with material pools')

    @classmethod
    def restore(cls, data, options):
        if data.get('version') != cls.VERSION: raise ValueError('Unsupported phase checkpoint')
        common = deepcopy(data); common['version'] = MaterialExchangeBudget.VERSION
        MaterialExchangeBudget.restore(common, options)
        obj = cls.__new__(cls); obj.options = options; obj.data = deepcopy(data)
        obj.phase = PhaseParameters(**data['phase_parameters'])
        obj._validate_phase(); return obj

    def phase_state(self, params):
        m = self.pool_masses(); cp = params.mantle_heat_capacity_j_kg_k
        components = self._phase_origins().sum(axis=2)
        out = []
        for i, key in enumerate(('energy_m_j', 'energy_c_j')):
            e = sum(map(np.longdouble, self.data[key]))
            mass = m[0] if i == 0 else math.fsum(m[1:])
            out.append(caloric_state(e, cp*mass, components[i],
                         self.phase.melting_temperatures_k, self.phase.latent_heat_j_kg))
        return out

    def temperatures(self, params):
        return np.array([x[0] for x in self.phase_state(params)])

    def available_mobile_kg(self, params):
        liquid = np.sum(self.phase_state(params)[0][1])
        eligible = np.sum(self._phase_origins()[0])
        r = self.phase.retained_liquid_fraction
        value = max(np.longdouble(0.), (liquid-r*eligible)/(1-r))
        result = float(value)
        return np.nextafter(result, 0.) if np.longdouble(result) > value else result

    def entropy_j_k(self, params):
        mass = self.pool_masses(); cp = params.mantle_heat_capacity_j_kg_k
        return math.fsum(cp*m*math.log(t)+float(np.sum(self.phase.latent_heat_j_kg*l/
            np.asarray(self.phase.melting_temperatures_k)))
            for m, (t, l) in zip((mass[0], math.fsum(mass[1:])), self.phase_state(params)))

    def exchange(self, born_km3, returned_km3, target_volumes_km3, params):
        """Birth first from liquid tags, then well-mixed basalt return, atomically.

        No felsic melting/recycling law is supplied in this bounded mode.
        Returning basalt retains its actual component mix and full enthalpy.
        """
        born, returned, target = (np.asarray(v, float) for v in (born_km3, returned_km3, target_volumes_km3))
        if any(v.shape != (2,) or np.any(~np.isfinite(v)) or np.any(v < 0) for v in (born, returned, target)):
            raise ValueError('Invalid phase transfer')
        if born[0] or returned[0]: raise ValueError('No felsic phase-transfer law')
        w = deepcopy(self); d = w.data; rho = self.options.basalt_density_kg_m3
        b, r = float(born[1]*1e9*rho), float(returned[1]*1e9*rho)
        available = w.available_mobile_kg(params)
        if b > available+32*math.ulp(max(available, b, 1.)):
            raise ValueError('Request exceeds finite mobile liquid inventory')
        if r >= w.pool_masses()[2]+b or target[1] == 0:
            raise ValueError('Phase mode requires a nonempty basalt pool')

        def transfer(amount, source):
            if amount == 0: return
            tags = w._phase_origins(); components = tags[source].sum(axis=1)
            t, liquid = w.phase_state(params)[source]
            weights = liquid if source == 0 else components
            total = np.sum(weights)
            if total < amount: raise ValueError('Phase donor exhausted')
            moved = np.divide(tags[source], components[:, None],
                out=np.zeros_like(tags[source]), where=components[:, None] > 0)*weights[:, None]*(amount/total)
            i, j = np.unravel_index(np.argmax(moved), moved.shape)
            moved[i, j] += np.longdouble(amount)-np.sum(moved)
            # One set of actual transfers updates both provenance ledgers.
            for k in range(len(components)):
                for j in range(3):
                    part = float(moved[k, j])
                    for row, sign in ((source, -1.), (1-source, 1.)):
                        pair = [d['phase_origin_hi'][row][k][j], d['phase_origin_lo'][row][k][j]]
                        _add(pair, sign*part)
                        d['phase_origin_hi'][row][k][j], d['phase_origin_lo'][row][k][j] = pair
                        pool = 0 if row == 0 else 2
                        pair = [d['origin_hi'][pool][j], d['origin_lo'][pool][j]]
                        _add(pair, sign*part); d['origin_hi'][pool][j], d['origin_lo'][pool][j] = pair
            f = 1. if source == 0 else float(np.sum(liquid)/np.sum(components))
            q = amount*(params.mantle_heat_capacity_j_kg_k*t+self.phase.latent_heat_j_kg*f)
            _add(d['energy_m_j'], (-1 if source == 0 else 1)*q)
            _add(d['energy_c_j'], (1 if source == 0 else -1)*q)
            _add(d['advected_out_j' if source == 0 else 'advected_return_j'], q)
        transfer(b, 0); transfer(r, 1)
        _add(d['created_kg'], b); _add(d['recycled_kg'], r)
        expected = target*1e9*np.array([self.options.continental_rock_density_kg_m3, rho])
        if any(abs(a-z) > 64*math.ulp(max(a, z, 1.)) for a, z in zip(w.pool_masses()[1:], expected)):
            raise ValueError('Phase transfer disagrees with geometric inventory')
        d['surface_volumes_km3'] = target.tolist()
        w._validate_phase(); self.data = d

    def advance(self, state, dt_myr, mass_earth, radius_km, gravity, period_hours,
                primary_mass_jupiter, eccentricity, params):
        if dt_myr <= 0 or not math.isfinite(dt_myr) or state.time_myr != self.data['time_myr']:
            raise ValueError('Invalid phase thermal increment')
        signature = dict(params=asdict(params), mass_earth=mass_earth, radius_km=radius_km,
            gravity=gravity, period_hours=period_hours, primary_mass_jupiter=primary_mass_jupiter,
            eccentricity_times=np.asarray(eccentricity.times_myr).tolist(),
            eccentricity_values=np.asarray(eccentricity.eccentricity).tolist())
        if self.data['physical_signature'] not in (None, signature):
            raise ValueError('Phase thermal configuration changed')
        w = deepcopy(self); d = w.data; d['physical_signature'] = signature
        current_t = w.temperatures(params)[0]
        if abs(current_t-state.mantle_temperature_k) > 4*math.ulp(current_t):
            raise ValueError('Thermal state disagrees with phase enthalpy')
        time = state.time_myr; target = time+dt_myr
        if not math.isfinite(target) or target <= time: raise ValueError('Unrepresentable phase target')
        mass = w.pool_masses(); cap = params.mantle_heat_capacity_j_kg_k*np.array([mass[0], math.fsum(mass[1:])])
        comps = w._phase_origins().sum(axis=2); totals = comps.sum(axis=1)
        volume = math.fsum(d['surface_volumes_km3'])*1e9
        melting = self.phase.melting_temperatures_k; L = self.phase.latent_heat_j_kg
        def state_of(e):
            states = [caloric_state(e[i], cap[i], comps[i], melting, L) for i in range(2)]
            temp = np.array([s[0] for s in states])
            phi = np.array([float(np.sum(s[1])/total) if total else 0. for s, total in zip(states, totals)])
            return temp, phi
        def trial(t, e, h):
            temp, _ = state_of(e)
            if temp[0] < params.surface_temperature_k+1 or temp[1] < params.surface_temperature_k:
                raise ValueError('Phase temperature outside thermal domain')
            g = self._conductances(temp[0], volume, d['area_m2'], radius_km, gravity, params)
            age = d['origin_system_age_myr']+t+h/2-d['origin_time_myr']
            powers = [radiogenic_power_w(age, mass[0], params),
                radiogenic_power_w(age, math.fsum(mass[1:]), params),
                tidal_heating_power_w(eccentricity.at(t+h/2), radius_km, period_hours, primary_mass_jupiter, params)]
            return implicit_phase_exchange(e, cap, comps, self.phase, g, powers, params.surface_temperature_k, h)
        for count in range(self.options.max_trials):
            if time == target: break
            future = np.asarray(eccentricity.times_myr); future = future[(future > time) & (future < target)]
            endpoint = float(future[0]) if len(future) else target
            h = min(d['next_step_myr'], self.options.max_step_myr, endpoint-time)
            if time+h <= time: raise RuntimeError('Phase time step underflow')
            e = np.array([sum(map(np.longdouble, d[k])) for k in ('energy_m_j', 'energy_c_j')])
            coarse, cq, _ = trial(time, e, h)
            middle, q1, r1 = trial(time, e, h/2)
            fine, q2, r2 = trial(time+h/2, middle, h/2)
            q = q1+q2; rad = r1+r2
            ft, ff = state_of(fine); ct, cf = state_of(coarse)
            error = max(float(np.max(abs(ft-ct)/(self.options.temperature_atol_k+self.options.temperature_rtol*abs(ft)))),
                float(np.max(abs(q-cq)/(self.options.exchange_atol_j_per_myr*h+self.options.exchange_rtol*abs(q)))),
                float(np.max(abs(ff-cf)/self.phase.phase_atol_fraction)))
            if not math.isfinite(error) or error > 1:
                d['rejected_steps'] += 1
                d['next_step_myr'] = h*(.5 if not math.isfinite(error) else max(.1, .8*error**(-.5)))
                continue
            for value in (rad[0], q[1], -q[3]): _add(d['energy_m_j'], float(value))
            for value in (rad[1], q[3], -q[2]): _add(d['energy_c_j'], float(value))
            for value in rad: _add(d['radiogenic_input_j'], float(value))
            for key, value in zip(('orbital_input_j', 'surface_output_j', 'basal_exchange_j'), q[1:]): _add(d[key], float(value))
            time = endpoint if h == endpoint-time else time+h
            d['accepted_steps'] += 1
            d['next_step_myr'] = min(self.options.max_step_myr, h*min(2., .9*max(error, 1e-12)**(-.5)))
        else:
            if time != target: raise RuntimeError('Phase thermal trial budget exhausted')
        d['time_myr'] = target
        new = replace(state, time_myr=target, system_age_myr=d['origin_system_age_myr']+target-d['origin_time_myr'],
                      mantle_temperature_k=float(w.temperatures(params)[0]))
        result = w.thermal_state(new, mass_earth, radius_km, gravity, period_hours, primary_mass_jupiter, eccentricity, params)
        self.data = d
        return result

    def diagnostics(self, params):
        d = super().diagnostics(params)
        states = self.phase_state(params); tags = self._phase_origins()
        error = tags.sum(axis=0)-np.asarray(self.data['initial_phase_origin_kg'], np.longdouble)
        d.update(mobile_liquid_kg=self.available_mobile_kg(params),
            liquid_masses_kg=[float(np.sum(s[1])) for s in states],
            eligible_mantle_mass_kg=float(np.sum(tags[0])),
            latent_energy_j=float(sum(np.sum(s[1])*self.phase.latent_heat_j_kg for s in states)),
            phase_origin_max_residual_kg=float(np.max(abs(error))), phase_enthalpy_includes_latent=True)
        return d


class PhaseDeformationExchange(DeformationExchange):
    VERSION = 'phase_deformation_1'
    MATERIAL_CLASS = PhaseMaterialBudget

    def _validate(self):
        super()._validate()
        self.material._validate_phase()
        self._require_solid()

    def _require_solid(self):
        if np.any(self.material.phase_state(self.thermal_params)[1][1] > 0):
            raise ValueError('Welded surface mechanics requires solid crust; molten crust is outside this mode')

    def _thermal_step(self, dt):
        super()._thermal_step(dt)
        self._require_solid()

    def _deposit_heat(self, q):
        super()._deposit_heat(q)
        self._require_solid()

    def diagnostics(self):
        d = super().diagnostics(); d['latent_heat_included'] = True
        d['scope'] = 'finite pseudocomponent enthalpy and welded deformation; no fracture'
        return d


class PhaseStrainExchange(StrainExchange):
    VERSION = 'phase_strain_1'
    COUPLED_CLASS = PhaseDeformationExchange

    @classmethod
    def from_fresh(cls, base, phase):
        c = base.coupled
        if c.thermal.time_myr != 0 or base.data['history']:
            raise ValueError('Phase wrapper requires a fresh zero-time state')
        b = PhaseMaterialBudget.from_fresh(c.material, c.thermal_params, phase)
        thermal = replace(c.thermal, mantle_temperature_k=float(b.temperatures(c.thermal_params)[0]))
        coupled = PhaseDeformationExchange(c.surface, b, thermal, c.thermal_params, c.eccentricity, c.options)
        return cls(coupled, base.zone_fraction, base.params, dict(base.data['provenance'],
            phase_initialization='new fixed phase envelope at unchanged enthalpy; no inferred deep melt'))

    def _supply_quota(self, quota):
        c = self.coupled; pos = np.maximum(quota, 0)
        available = np.longdouble(c.material.available_mobile_kg(c.thermal_params))/c.surface.params.basalt_density_kg_m3
        requested = np.sum(pos, dtype=np.longdouble)
        factor = min(np.longdouble(1.), available/requested) if requested else np.longdouble(1.)
        old = c.surface.arrays['basalt_volume_m3']
        desired = old.astype(np.longdouble)+pos.astype(np.longdouble)*factor
        upper = desired.astype(float)
        rounded_up = upper.astype(np.longdouble) > desired
        upper[rounded_up] = np.nextafter(upper[rounded_up], old[rounded_up])
        # Rounding cannot promise more liquid than exists. Sub-ULP requests stay
        # in the donor, rather than turning an inventory residual into heat/mass.
        return np.minimum(quota, 0)+(upper-old)

    def _transfer(self, old, target, born, returned):
        c = self.coupled; f = c.material.data['surface_volumes_km3'][0]
        # No same-step cold return is silently counted as an immediately liquid source.
        c.material.exchange([0., math.fsum(born)/1e9], [0., math.fsum(returned)/1e9],
                            [f, math.fsum(target)/1e9], c.thermal_params)

    def diagnostics(self):
        d = super().diagnostics()
        d['scope'] = 'finite phase-gated strain exchange; fixed synthetic phase envelope, no fracture or convection'
        return d
