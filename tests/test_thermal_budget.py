from dataclasses import asdict, replace
from decimal import Decimal, localcontext
import json
import math

import numpy as np
import pytest

from tectonics.thermal import ThermalParameters, initialize_thermal_state, _geometry, SECONDS_PER_MYR
from tectonics.thermal_budget import ThermalBudgetIntegrator, ThermalIntegrationParameters
from tectonics.tides import constant_eccentricity, EccentricityHistory


def initial(p):
    return initialize_thermal_state(0.5, 5287., 7.12, p)


def advance(integrator, state, dt, p, eccentricity=None):
    return integrator.advance(state, dt, 0.5, 5287., 7.12, 47., 5.,
                              constant_eccentricity(0.) if eccentricity is None else eccentricity, p)[0]


def capacity(p):
    return _geometry(0.5, 5287., p)[1] * p.mantle_heat_capacity_j_kg_k


def no_sources(**kw):
    return ThermalParameters(radiogenic_specific_power_at_formation_w_kg=0., tidal_k2_over_q=0., **kw)


def test_no_exchange_exact_equilibrium():
    p = no_sources(thermal_conductivity_w_m_k=0.)
    s = initial(p); ctl = ThermalBudgetIntegrator(s)
    n = advance(ctl, s, 100., p)
    assert n.mantle_temperature_k == s.mantle_temperature_k
    assert n.time_myr == 100.
    assert all(ctl.data[k] == [0., 0.] for k in
               ('energy_change_j', 'radiogenic_input_j', 'orbital_input_j', 'surface_output_j'))


def test_conductive_cooling_second_order_without_temperature_floor():
    # Nu=1 yields d(T-Ts)/dt = -k*A/(D*C)*(T-Ts), independently soluble.
    p = no_sources(critical_rayleigh=1e100, thermal_conductivity_w_m_k=4000.)
    s = initial(p)
    _, _, area, depth = _geometry(.5, 5287., p)
    rate = p.thermal_conductivity_w_m_k * area / (depth * capacity(p)) * SECONDS_PER_MYR
    exact = p.surface_temperature_k + (s.mantle_temperature_k-p.surface_temperature_k)*math.exp(-rate*100)
    errors = []
    for h in (8., 4., 2.):
        opts = ThermalIntegrationParameters(initial_step_myr=h, max_step_myr=h,
                                           temperature_atol_k=1e9, exchange_rtol=1e9)
        ctl = ThermalBudgetIntegrator(s, opts)
        n = advance(ctl, s, 100., p)
        errors.append(abs(n.mantle_temperature_k - exact))
        assert abs(ctl.diagnostics(n, capacity(p))['balance_residual_j']) < 1e-14*abs(ctl.data['energy_change_j'][0])
    assert errors[0]/errors[1] > 3.5
    assert errors[1]/errors[2] > 3.5


def test_tolerance_sequence_against_analytic_solution():
    p = no_sources(critical_rayleigh=1e100, thermal_conductivity_w_m_k=4000.)
    s = initial(p)
    _, _, area, depth = _geometry(.5, 5287., p)
    exact = p.surface_temperature_k + (s.mantle_temperature_k-p.surface_temperature_k)*math.exp(
        -p.thermal_conductivity_w_m_k*area/(depth*capacity(p))*SECONDS_PER_MYR*100)
    errors = []
    for tol in (1e-2, 1e-4, 1e-6):
        ctl = ThermalBudgetIntegrator(s, ThermalIntegrationParameters(
            initial_step_myr=100., max_step_myr=100., temperature_atol_k=tol,
            temperature_rtol=tol/2000, exchange_rtol=tol))
        n = advance(ctl, s, 100., p)
        errors.append(abs(n.mantle_temperature_k-exact))
        assert ctl.data['rejected_steps'] > 0
        assert ctl.data['max_accepted_error_ratio'] <= 1
    assert errors[2] < errors[1] < errors[0]
    assert errors[2] < .001


def test_radiogenic_source_matches_independent_integral():
    p = ThermalParameters(thermal_conductivity_w_m_k=0., tidal_k2_over_q=0.)
    s = initial(p)
    ctl = ThermalBudgetIntegrator(s, ThermalIntegrationParameters(exchange_rtol=1e-8))
    n = advance(ctl, s, 100., p)
    lam = math.log(2)/p.radiogenic_effective_half_life_myr
    mass = _geometry(.5, 5287., p)[1]
    exact = mass*p.radiogenic_specific_power_at_formation_w_kg*math.exp(-lam*s.system_age_myr)*(-math.expm1(-lam*100))/lam*SECONDS_PER_MYR
    assert math.fsum(ctl.data['radiogenic_input_j']) == pytest.approx(exact, rel=1e-7)
    assert n.mantle_temperature_k == pytest.approx(s.mantle_temperature_k+exact/capacity(p), rel=1e-8)


def test_prescribed_short_orbital_pulse_is_not_skipped():
    p = ThermalParameters(thermal_conductivity_w_m_k=0., radiogenic_specific_power_at_formation_w_kg=0.)
    s = initial(p)
    history = EccentricityHistory(np.array([0., 4.999, 5., 5.001, 10.]),
                                 np.array([0., 0., .001, 0., 0.]))
    ctl = ThermalBudgetIntegrator(s)
    advance(ctl, s, 10., p, history)
    from tectonics.thermal import tidal_heating_power_w
    peak = tidal_heating_power_w(.001, 5287., 47., 5., p)
    exact = peak*(.002/3)*SECONDS_PER_MYR  # integral of a squared triangular pulse
    assert math.fsum(ctl.data['orbital_input_j']) == pytest.approx(exact, rel=.002)
    assert ctl.data['rejected_steps'] > 0


def test_two_hour_terminal_interval_has_exact_time_and_resolved_exchange():
    p = ThermalParameters(thermal_conductivity_w_m_k=0., tidal_k2_over_q=0.,
                          radiogenic_effective_half_life_myr=1e100)
    s = replace(initial(p), time_myr=1.03, system_age_myr=501.03)
    ctl = ThermalBudgetIntegrator(s)
    dt = 2/(365.25*24*1e6)
    target = s.time_myr+dt
    n = advance(ctl, s, dt, p)
    with localcontext() as context:
        context.prec = 70
        mass = _geometry(.5, 5287., p)[1]
        exact = Decimal(mass)*Decimal(p.radiogenic_specific_power_at_formation_w_kg)*Decimal(target-s.time_myr)*Decimal(SECONDS_PER_MYR)
        got = sum(Decimal(v) for v in ctl.data['energy_change_j'])
        assert abs((got-exact)/exact) < Decimal('5e-16')
    assert n.time_myr == target
    assert ctl.diagnostics(n, capacity(p))['balance_residual_j'] == 0.


def test_restart_keeps_controller_and_compensations_bitwise():
    p = ThermalParameters()
    s = initial(p); a = ThermalBudgetIntegrator(s)
    middle = advance(a, s, 4., p)
    b = ThermalBudgetIntegrator.restore(json.loads(json.dumps(a.checkpoint())))
    end_a = advance(a, middle, 4., p)
    end_b = advance(b, middle, 4., p)
    assert asdict(end_a) == asdict(end_b)
    assert a.checkpoint() == b.checkpoint()


def test_input_change_or_lost_memory_rejected():
    p = ThermalParameters(); s = initial(p); ctl = ThermalBudgetIntegrator(s)
    middle = advance(ctl, s, 1., p)
    snapshot = ctl.checkpoint()
    with pytest.raises(ValueError, match='Physical configuration'):
        advance(ctl, middle, 1., replace(p, mantle_mass_fraction=.5))
    with pytest.raises(ValueError, match='energy inventory'):
        advance(ctl, replace(middle, mantle_temperature_k=middle.mantle_temperature_k+1), 1., p)
    with pytest.raises(ValueError, match='options changed'):
        ThermalBudgetIntegrator.restore(snapshot, ThermalIntegrationParameters(max_step_myr=8.))
    assert ctl.checkpoint() == snapshot


def test_failed_step_does_not_commit_partial_history_or_floor_heat():
    p = no_sources(critical_rayleigh=1e100, thermal_conductivity_w_m_k=4e9)
    s = initial(p)
    ctl = ThermalBudgetIntegrator(s, ThermalIntegrationParameters(max_trials=3))
    snapshot = ctl.checkpoint()
    with pytest.raises(RuntimeError, match='budget exhausted'):
        advance(ctl, s, 1., p)
    assert ctl.checkpoint() == snapshot


@pytest.mark.parametrize('dt', [0., -1., math.nan, math.inf])
def test_invalid_increment(dt):
    p = ThermalParameters(); s = initial(p)
    with pytest.raises(ValueError):
        advance(ThermalBudgetIntegrator(s), s, dt, p)


def test_old_state_starts_new_explicit_budget_origin():
    p = ThermalParameters(); s = replace(initial(p), time_myr=27., system_age_myr=527.)
    ctl = ThermalBudgetIntegrator(s)
    assert ctl.data['origin_time_myr'] == 27.
    assert ctl.data['energy_change_j'] == [0., 0.]
    assert ctl.data['next_step_myr'] > 0.
    assert advance(ctl, s, .1, p).time_myr == 27.1


def test_incomplete_saved_controller_is_not_silently_reinitialized():
    p = ThermalParameters(); ctl = ThermalBudgetIntegrator(initial(p))
    data = ctl.checkpoint(); del data['next_step_myr']
    with pytest.raises(ValueError, match='Incomplete thermal budget checkpoint'):
        ThermalBudgetIntegrator.restore(data)
    data = ctl.checkpoint(); data['energy_change_j'][1] = math.nan
    with pytest.raises(ValueError, match='Invalid compensated energy'):
        ThermalBudgetIntegrator.restore(data)


def test_native_checkpoint_roundtrip_retains_new_controller(tmp_path):
    from tectonics.checkpoint import RunCheckpoint, load_checkpoint, save_checkpoint
    from tectonics.mesh import build_icosphere
    from tectonics.plates import random_plate_system, PlateSystem
    from tectonics.lithosphere import initialize_lithosphere, boundary_records_for_state
    from tectonics.continental import initialize_continental_cycle
    from tectonics.topography import initialize_topography, TopographyParameters
    from tectonics.topology import PlateTopologyManager, PlateTopologyParameters
    mesh = build_icosphere(0)
    plates = random_plate_system(mesh, 3, 321, .2, .1, .4)
    lith = initialize_lithosphere(mesh, plates, .25, 2)
    system = PlateSystem(lith.cell_plate.copy(), plates.plates)
    boundaries = boundary_records_for_state(mesh, lith, system, 5287., 4., 1.)
    topo = initialize_topography(mesh, lith, boundaries, TopographyParameters())
    p = ThermalParameters(); s = initial(p); ctl = ThermalBudgetIntegrator(s)
    s = advance(ctl, s, 4., p)
    cp = RunCheckpoint(lith, initialize_continental_cycle(mesh), s, topo, system, system,
                       PlateTopologyManager(PlateTopologyParameters()), .25, 123., [], [], [], [], [], [])
    cp.thermal_integration = ctl.checkpoint()
    save_checkpoint(tmp_path/'cp', cp)
    got = load_checkpoint(tmp_path/'cp', PlateTopologyManager(PlateTopologyParameters()))
    assert got.thermal_integration == cp.thermal_integration
    resumed = ThermalBudgetIntegrator.restore(got.thermal_integration)
    assert asdict(advance(resumed, got.thermal, 4., p)) == asdict(advance(ctl, s, 4., p))
    meta_path = tmp_path/'cp'/'meta.json'
    meta = json.loads(meta_path.read_text()); del meta['thermal_integration']
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='missing its thermal controller'):
        load_checkpoint(tmp_path/'cp', PlateTopologyManager(PlateTopologyParameters()))
