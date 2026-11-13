from copy import deepcopy
from dataclasses import replace
from decimal import Decimal, localcontext
import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tectonics.conforming_surface import column_potential
from tectonics.strain_exchange import (PressureParameters, StrainExchange,
    boundary_zone_fraction, passive_projection)
from test_deformation_exchange import coupled
from test_regional_deformation import deforming


def project(volume, lower, upper, pressure, compliance=1e9, area=None, radius=5.287e6):
    volume = np.asarray(volume, float)
    if area is None: area = np.full(len(volume), 1e12)
    return passive_projection(np.asarray(area), volume, np.asarray(lower), np.asarray(upper),
        pressure, compliance, 2900., 7.12, radius)


def initialized(pressure=0., step=1.):
    c = coupled(step=step)
    return StrainExchange(c, boundary_zone_fraction(c.surface, 1e6, 2),
        PressureParameters(initial_pressure_pa=pressure), {'test': 'explicit pressure initialization'})


def test_single_column_thin_limit_matches_independent_analytic_pressure():
    area = 1e12; v = 1e15; C = 1e9; p = 1e8; k = 2900*7.12/area
    target, event = project([v], [v], [1e17], p, C, radius=1e30)
    q = (p-k*v)/(k+1/C)
    assert target[0]-v == pytest.approx(q, rel=3e-14)
    assert event['pressure_pa'] == pytest.approx(p-q/C, rel=3e-15)
    assert event['column_work_j'] == pytest.approx(k*(v*q+.5*q*q), rel=3e-14)
    assert event['heat_j'] == pytest.approx(.5*(k+1/C)*q*q, rel=4e-14)


def test_unpressurized_store_cannot_pay_birth_and_can_be_charged_by_return():
    v = np.array([1e16])
    target, e = project(v, v, v*1.1, 0.)
    assert np.array_equal(target, v) and e['source_work_j'] == e['heat_j'] == 0
    target, e = project(v, v*.5, v, 0.)
    assert target[0] < v[0] and e['pressure_pa'] > 0 and e['source_work_j'] < 0
    assert e['heat_j'] > 0
    assert -.5*1e9*e['pressure_pa']**2 == pytest.approx(e['source_work_j'], rel=3e-15)


def test_finite_pressure_stops_at_cost_without_unlimited_replenishment():
    v = np.array([1e15]); pressure = 1e8; initial = .5*1e9*pressure**2
    work = 0.; heat = 0.
    for _ in range(25):
        v, e = project(v, v, v*1.01, pressure)
        pressure = e['pressure_pa']; work += e['source_work_j']; heat += e['heat_j']
    assert pressure < 1e8 and 0 < work < initial and heat > 0
    assert .5*1e9*pressure**2+work == pytest.approx(initial, rel=3e-15)
    v, e = project(v, v, v*100, pressure)
    next_v, next_e = project(v, v, v*100, e['pressure_pa'])
    assert next_v == pytest.approx(v, rel=3e-15)
    assert abs(next_e['source_work_j']) < initial*1e-14


def test_simultaneous_geometric_gates_are_passive_and_satisfy_kkt():
    v = np.array([4e16, 1e16, 2e16]); lo = v*np.array([.7, 1., 1.]); hi = v*np.array([1., 1.5, 1.])
    target, e = project(v, lo, hi, 3e8)
    h = column_potential(np.full(3, 1e12), target, 2900., 7.12, 5.287e6)[2]
    gradient = 2900*7.12*h-e['pressure_pa']
    assert np.all(target >= lo) and np.all(target <= hi)
    for i in range(2):
        if lo[i] < target[i] < hi[i]: assert abs(gradient[i]) < 1e-6
        elif target[i] == lo[i]: assert gradient[i] >= -1e-6
        else: assert gradient[i] <= 1e-6
    assert target[2] == v[2] and e['heat_j'] >= 0
    assert abs(e['work_identity_residual_j']) < 2e-14*max(abs(e['column_work_j']), 1.)


def test_pressure_projection_preserves_small_work_against_decimal_columns():
    area = np.array([1e12, 2e12]); v = np.array([1e16, 3e16])
    target, e = project(v, v*np.array([1., 1-1e-7]), v*np.array([1+1e-7, 1.]), 3e8, area=area)
    with localcontext() as ctx:
        ctx.prec = 65
        def energy(vol):
            total = Decimal(0); R = Decimal('5287000'); rho_g = Decimal(2900)*Decimal('7.12')
            for a, x in zip(area, vol):
                A = Decimal(float(a)); V = Decimal(float(x))
                h = R*((1+3*V/(A*R))**(Decimal(1)/3)-1)
                total += rho_g*A*h*h*(Decimal('.5')+h/R*(Decimal(2)/3+h/R/4))
            return total
        delta = float(energy(target)-energy(v))
    assert e['column_work_j'] == pytest.approx(delta, rel=5e-14)


def test_material_band_rotation_no_junction_double_count_and_quadrature_refinement():
    s = deforming(2); w = boundary_zone_fraction(s, 1e6, 2)
    other = deepcopy(s); other.vertices = Rotation.from_rotvec([.4, -.2, .7]).apply(s.vertices)
    assert boundary_zone_fraction(other, 1e6, 2) == pytest.approx(w, abs=2e-14)
    assert np.min(w) >= 0 and np.max(w) <= 1
    areas = s.geometry(s.vertices)[0]
    integral = [np.dot(areas, boundary_zone_fraction(s, 1e6, r)) for r in (1, 2, 3)]
    assert abs(integral[2]-integral[1]) < abs(integral[1]-integral[0])
    other.arrays['owner'][:] = 0
    assert not np.any(boundary_zone_fraction(other, 1e6))


def test_zero_gate_recovers_parent_coupling_without_material_transfer():
    base = coupled(); o = StrainExchange(base, np.zeros(len(base.surface.faces)))
    base.advance(2.); o.advance(2.)
    # Histories differ by explicit zero exchanges only; the physical state agrees.
    assert np.array_equal(o.coupled.surface.vertices, base.surface.vertices)
    assert o.coupled.material.checkpoint() == base.material.checkpoint()
    assert o.diagnostics()['basalt_created_m3'] == o.diagnostics()['basalt_returned_m3'] == 0
    assert o.diagnostics()['reservoir_energy_j'] == 0


def test_linked_exchange_conserves_mass_heat_provenance_and_felsic_history():
    o = initialized(); initial = o.checkpoint(); o.advance(4.); d = o.diagnostics()
    assert d['external_work_j'] == 0 and d['hydraulic_heat_j'] > 0
    assert d['basalt_created_m3'] > 0 or d['basalt_returned_m3'] > 0
    assert abs(d['combined_energy_residual_j'])/d['heat_generated_j'] < 1e-4
    assert abs(d['thermal_energy_residual_j'])/d['heat_generated_j'] < 1e-11
    assert np.max(np.abs(d['origin_mass_residual_kg'])) < 2e9
    assert abs(d['mass_residual_kg']) < 2e9
    assert o.coupled.surface.arrays['felsic_volume_m3'].tolist() == initial['coupled']['surface']['arrays']['felsic_volume_m3']
    assert o.coupled.surface.arrays['owner'].tolist() == initial['coupled']['surface']['arrays']['owner']
    assert o.data['initial_energy_j'] == 0
    assert d['reservoir_energy_j'] > 0  # gravity-driven return charges the store


def test_restart_through_exchange_and_failure_atomicity(tmp_path):
    o = initialized(); o.advance(2.); o.save(tmp_path/'first')
    restored = StrainExchange.load(tmp_path/'first'); assert o.checkpoint() == restored.checkpoint()
    o.advance(4.); restored.advance(4.); assert o.checkpoint() == restored.checkpoint()
    before = o.checkpoint()
    with pytest.raises(ValueError): o.step(-1.)
    assert before == o.checkpoint()
    bad = deepcopy(before); bad['zone_fraction'][0] = .123
    with pytest.raises(ValueError, match='identity'): StrainExchange.restore(bad)
    bad = deepcopy(before); bad['data']['reservoir_energy_j'][0] *= 2
    with pytest.raises(ValueError, match='EOS'): StrainExchange.restore(bad)


def test_strain_exchange_temporal_error_decreases_without_changing_physics():
    states = []
    for dt in (1., .5, .25):
        o = initialized(step=dt); o.advance(4.); states.append(o.diagnostics())
    errors = [abs(states[i]['basalt_returned_m3']-states[i+1]['basalt_returned_m3']) for i in (0, 1)]
    assert errors[1] < .7*errors[0]
    assert all(abs(d['combined_energy_residual_j'])/d['heat_generated_j'] < 1e-4 for d in states)


def test_compression_into_high_pressure_store_is_blocked_without_forcing_birth():
    # In this hemispheric fixture basalt only compresses. A 300 MPa store
    # exceeds its ~145 MPa column cost: neither direction is admissible.
    o = initialized(pressure=3e8); o.advance(4.); d = o.diagnostics()
    assert d['basalt_created_m3'] == d['basalt_returned_m3'] == 0
    assert d['hydraulic_heat_j'] == 0 and d['pressure_pa'] == 3e8


def test_no_implicit_pressure_initialization_from_evolved_state_or_new_mass():
    c = coupled(); c.advance(1.)
    with pytest.raises(ValueError, match='zero-time'): StrainExchange(c, np.zeros(len(c.surface.faces)))
    c = coupled()
    with pytest.raises(ValueError, match='inventory'):
        StrainExchange(c, np.ones(len(c.surface.faces)), PressureParameters(reference_volume_m3=1e25))
    with pytest.raises(ValueError, match='compliance domain'):
        PressureParameters(initial_pressure_pa=1e15)
