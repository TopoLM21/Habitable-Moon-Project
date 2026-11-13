from copy import deepcopy
from dataclasses import replace
from decimal import Decimal, localcontext
import math

import numpy as np
import pytest

from tectonics.phase_exchange import (PhaseParameters, PhaseMaterialBudget,
    PhaseStrainExchange, caloric_state, implicit_phase_exchange)
from tectonics.strain_exchange import StrainExchange
from tectonics.material_exchange import MaterialExchangeBudget
from tectonics.thermal import SECONDS_PER_MYR
from test_strain_exchange import initialized


def setup(phase=None, step=1., pressure=0.):
    p = phase or PhaseParameters((1700., 1800., 1900., 2000.), 1e-4)
    return PhaseStrainExchange.from_fresh(initialized(pressure=pressure, step=step), p)


def total_energy(b):
    return sum(map(Decimal, b.data['energy_m_j']+b.data['energy_c_j']))


def exchange_mass(b, params, born=0., returned=0.):
    rho = b.options.basalt_density_kg_m3*1e9
    v = b.data['surface_volumes_km3'].copy()
    v[1] += (born-returned)/rho
    b.exchange([0., born/rho], [0., returned/rho], v, params)


def test_stefan_plateau_and_reversible_enthalpy_are_analytic():
    c = 1200.; m = [1.]; tm = [1600.]; L = 4e5
    for t in (300., 1599., 1601., 2200.):
        liquid = float(t > tm[0]); e = c*t+L*liquid
        out, f = caloric_state(e, c, m, tm, L)
        assert out == pytest.approx(t, abs=1e-12)
        assert f[0] == liquid
    for f in np.linspace(0, 1, 17):
        out, liquid = caloric_state(c*tm[0]+L*f, c, m, tm, L)
        assert out == tm[0] and liquid[0] == pytest.approx(f, abs=1e-15)
        assert c*out+L*liquid[0] == pytest.approx(c*tm[0]+L*f, rel=1e-15)


def test_explicit_phase_initialization_adds_neither_mass_nor_enthalpy():
    old = initialized(); cp = old.checkpoint(); x = PhaseStrainExchange.from_fresh(
        old, PhaseParameters((1700., 1800., 1900., 2000.), 1e-4))
    b = x.coupled.material; p = x.coupled.thermal_params
    assert old.checkpoint() == cp
    assert b.data['energy_m_j'] == old.coupled.material.data['energy_m_j']
    assert b.data['energy_c_j'] == old.coupled.material.data['energy_c_j']
    assert b.pool_masses().tolist() == old.coupled.material.pool_masses().tolist()
    assert b.temperatures(p)[0] < old.coupled.material.temperatures(p)[0]
    assert 0 < b.available_mobile_kg(p) < b.pool_masses()[0]*1e-4
    assert x.data['initial_energy_j'] == old.data['initial_energy_j']
    old.advance(.25)
    with pytest.raises(ValueError, match='fresh'): PhaseStrainExchange.from_fresh(old, b.phase)
    with pytest.raises(ValueError): MaterialExchangeBudget.restore(b.checkpoint(), b.options)


def test_hot_is_not_molten_and_pressure_cannot_manufacture_feedstock():
    x = setup(PhaseParameters((2100., 2300.), 1e-3), pressure=5e8)
    b = x.coupled.material; p = x.coupled.thermal_params
    assert b.temperatures(p)[0] > 1800 and b.available_mobile_kg(p) == 0
    area = x.coupled.surface.geometry(x.coupled.surface.vertices)[0]
    x._exchange(area*.999)  # imposed dilatation tests the exchange operator only
    assert x.diagnostics()['basalt_created_m3'] == 0
    assert x.diagnostics()['surface_pressure_work_j'] == 0
    before = b.checkpoint()
    with pytest.raises(ValueError, match='mobile'): exchange_mass(b, p, born=1e17)
    assert b.checkpoint() == before


def test_selective_extraction_depletes_components_and_carries_full_enthalpy():
    x = setup(); b = x.coupled.material; p = x.coupled.thermal_params
    tags = b._phase_origins().copy(); t = b.temperatures(p)[0]
    amount = b.available_mobile_kg(p)*.5
    old = deepcopy(b.data); s0 = b.entropy_j_k(p)
    with localcontext() as ctx:
        ctx.prec = 65
        energy = total_energy(b)
        exchange_mass(b, p, born=amount)
        assert abs(total_energy(b)-energy) < Decimal('1e7')
        removed = sum(map(Decimal, old['energy_m_j']))-sum(map(Decimal, b.data['energy_m_j']))
        assert float(removed) == pytest.approx(amount*(1200*t+4e5), rel=3e-15)
    assert b.entropy_j_k(p) >= s0
    assert np.array_equal(b._phase_origins()[0, 2:], tags[0, 2:])
    assert np.allclose(b._phase_origins().sum(axis=0), tags.sum(axis=0), rtol=3e-15, atol=0)
    assert b.available_mobile_kg(p) == pytest.approx(amount, rel=1e-11)
    assert b.temperatures(p)[0] == pytest.approx(t, abs=5e-12)


def test_exhaustion_does_not_regenerate_liquid_on_repeated_requests():
    x = setup(); b = x.coupled.material; p = x.coupled.thermal_params
    available = b.available_mobile_kg(p)
    exchange_mass(b, p, born=available*(1-1e-10))
    left = b.available_mobile_kg(p)
    assert left < available*2e-10
    before = b.checkpoint()
    for _ in range(4):
        with pytest.raises(ValueError, match='mobile'): exchange_mass(b, p, born=available*.01)
        assert b.checkpoint() == before


def test_cold_return_has_no_latent_heat_and_preserves_fusibility_history():
    x = setup(); b = x.coupled.material; p = x.coupled.thermal_params
    old = deepcopy(b.data); t = b.temperatures(p)[1]; tags = b._phase_origins().sum(axis=0)
    amount = b.pool_masses()[2]*.001
    exchange_mass(b, p, returned=amount)
    assert math.fsum(b.data['advected_return_j']) == pytest.approx(amount*1200*t, rel=2e-15)
    assert np.allclose(tags, b._phase_origins().sum(axis=0), rtol=2e-15)
    assert b.temperatures(p)[0] < old['phase_initialization']['new_temperature_k'][0]
    assert np.sum(b._phase_origins()[0]) > b.pool_masses()[0]*b.phase.mantle_fusible_fraction
    assert b.diagnostics(p)['phase_origin_max_residual_kg'] < 1e6


def test_implicit_two_box_matches_linear_backward_euler_without_phase_mass():
    phase = PhaseParameters((1700.,), 0.)
    cap = np.array([1e8, 3e7]); temp = np.array([1800., 700.])
    g = np.array([10., 5.]); h = 1e5; powers = np.array([20., 3., 2.]); ts = 288.
    A = np.array([[-g[0]/cap[0], g[0]/cap[0]], [g[0]/cap[1], -(g[0]+g[1])/cap[1]]])
    f = np.array([(powers[0]+powers[2])/cap[0], (powers[1]+g[1]*ts)/cap[1]])
    expected = np.linalg.solve(np.eye(2)-h*A, temp+h*f)
    e, q, rad = implicit_phase_exchange(cap*temp, cap, np.zeros((2, 1)), phase, g, powers, ts, h/SECONDS_PER_MYR)
    assert e/cap == pytest.approx(expected, rel=2e-14)
    assert float(np.sum(e-cap*temp)-np.sum(rad)-q[1]+q[2]) == pytest.approx(0., abs=.0001)


def test_latent_heating_and_isolated_exchange_have_no_free_heat():
    phase = PhaseParameters((1600.,), .1)
    cap = [1200., 1200.]; m = np.ones((2, 1)); energy = [1200*1600., 1200*300.]
    e, q, rad = implicit_phase_exchange(energy, cap, m, phase, [0., 0.], [2e5, 0., 0.], 288., 1/SECONDS_PER_MYR)
    t, liquid = caloric_state(e[0], cap[0], m[0], phase.melting_temperatures_k, phase.latent_heat_j_kg)
    assert t == 1600 and liquid[0] == pytest.approx(.5)
    assert float(np.sum(e)-sum(energy)) == pytest.approx(2e5)
    e0 = np.array([1200*1800.+4e5, 1200*1400.])
    def entropy(e):
        return sum(1200*math.log(t)+float(4e5*np.sum(l)/1600)
            for t, l in [caloric_state(e[i], cap[i], m[i], (1600.,), 4e5) for i in range(2)])
    e, q, _ = implicit_phase_exchange(e0, cap, m, phase, [1e3, 0.], [0., 0., 0.], 288., 1/SECONDS_PER_MYR)
    assert float(np.sum(e)-np.sum(e0)) == pytest.approx(0., abs=1e-8)
    assert entropy(e) >= entropy(e0)


def test_small_exchange_accuracy_is_not_scaled_by_total_thermal_stock():
    phase = PhaseParameters((2000.,), 0.)
    cap = np.array([1e27, 1e25]); temp = np.array([1800., 500.])
    h = 7200.; g = [1e12, 2e12]
    e, q, _ = implicit_phase_exchange(cap*temp, cap, np.zeros((2, 1)), phase,
                                    g, [0., 0., 0.], 288., h/SECONDS_PER_MYR)
    with localcontext() as ctx:
        ctx.prec = 65
        cm, cc = map(Decimal, cap); gm, gs = map(Decimal, g); dt = Decimal(h)
        mt, ct = [Decimal(float(a))/c for a, c in zip(cap*temp, (cm, cc))]
        a, b = 1+dt*gm/cm, -dt*gm/cm
        c, d = -dt*gm/cc, 1+dt*(gm+gs)/cc
        rhs = ct+dt*gs*288/cc; det = a*d-b*c
        mt1, ct1 = (mt*d-b*rhs)/det, (a*rhs-c*mt)/det
        assert float(q[3]) == pytest.approx(float(dt*gm*(mt1-ct1)), rel=2e-14)
        assert float(q[2]) == pytest.approx(float(dt*gs*(ct1-288)), rel=2e-14)


def test_liquid_crust_is_not_silently_advanced_as_welded_solid():
    with pytest.raises(ValueError, match='solid crust'):
        setup(PhaseParameters((300.,), 1e-4))


def test_representable_geometric_quota_cannot_overdraw_near_empty_stock():
    x = setup(PhaseParameters((1700., 1800., 1900., 2000.), 1e-9), pressure=5e8)
    s = x.coupled.surface; b = x.coupled.material; p = x.coupled.thermal_params
    old = s.arrays['basalt_volume_m3']; request = old*.001
    capped = x._supply_quota(request)
    actual = (old+capped)-old
    used = np.sum(actual, dtype=np.longdouble)*s.params.basalt_density_kg_m3
    assert used <= b.available_mobile_kg(p)
    assert np.all(actual >= 0) and np.all(actual <= request)
    area = s.geometry(s.vertices)[0]
    x._exchange(area*.999)
    assert x.diagnostics()['basalt_created_m3']*s.params.basalt_density_kg_m3 <= float(used)+32*math.ulp(float(used))


def test_phase_discretization_approaches_independent_continuous_envelope():
    cp = 1200.; m = 1.; eligible = .01; L = 4e5; lo = 1700.; hi = 2200.
    errors = []
    for n in (4, 16, 64):
        melting = lo+(np.arange(n)+.5)*(hi-lo)/n; err = []
        for old_t in np.linspace(1740, 2150, 47):
            expected = (cp*old_t+L*eligible*lo/(hi-lo))/(cp+L*eligible/(hi-lo))
            t, liquid = caloric_state(cp*old_t, cp*m, np.full(n, eligible/n), melting, L)
            err.append(abs(t-expected))
            assert abs(float(np.sum(liquid)/eligible)-(expected-lo)/(hi-lo)) <= 1/n
        errors.append(max(err))
    assert errors[2] < errors[1]/3 < errors[0]/9


def test_linked_heat_mass_restart_and_tamper_rejection(tmp_path):
    x = setup(); x.advance(1.); x.save(tmp_path/'phase')
    y = PhaseStrainExchange.load(tmp_path/'phase')
    assert x.checkpoint() == y.checkpoint()
    x.advance(2.); y.advance(2.)
    assert x.checkpoint() == y.checkpoint()
    d = x.diagnostics()
    assert abs(d['thermal_energy_residual_j']) < 1e-10*d['heat_generated_j']
    assert abs(d['combined_energy_residual_j']) < 1e-4*d['heat_generated_j']
    assert abs(d['mass_residual_kg']) < 2e9
    assert d['phase_origin_max_residual_kg'] < 2e7
    assert d['external_work_j'] == 0 and d['latent_heat_included']
    with pytest.raises(ValueError): StrainExchange.restore(x.checkpoint())
    bad = x.checkpoint(); bad['coupled']['material']['phase_origin_hi'][0][0][0] *= 1.01
    with pytest.raises(ValueError, match='Phase'): PhaseStrainExchange.restore(bad)


def test_linked_time_refinement_and_controller_preservation():
    ds = []
    for dt in (1., .5, .25):
        x = setup(step=dt); x.advance(2.); ds.append(x.diagnostics())
    for key in ('basalt_returned_m3', 'surface_output_j'):
        a = abs(ds[0][key]-ds[1][key]); b = abs(ds[1][key]-ds[2][key])
        assert b < .8*a
    x = setup(); x.advance(.1); before = x.checkpoint()
    with pytest.raises(ValueError): x.step(-1.)
    assert before == x.checkpoint()
