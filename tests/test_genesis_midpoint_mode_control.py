"""Production inertial solve regression, no Genesis evolution or model replay."""
from unittest.mock import patch
import numpy as np
import pytest
from analysis.genesis_midpoint_mode_control.control import DragFixture,run_mode,production


def test_real_production_solve_and_update_are_not_reimplemented():
    assert DragFixture._mechanical_trial is production.VolumeModel._mechanical_trial
    assert DragFixture._advance_trial is production.VolumeModel._advance_trial
    with patch.object(production.linalg,'solve',wraps=production.linalg.solve) as solve:
        result=run_mode(1.,1.)
    assert solve.call_count==4
    assert result['rows'][0]['endpoint_velocity_over_initial']==pytest.approx(1/3)
    assert result['rows'][0]['mean_velocity_over_initial']==pytest.approx(2/3)


@pytest.mark.parametrize('z',[.01,1.,2.,10.,1e4,3e10])
def test_production_amplification_and_discrete_work(z):
    r=(2-z)/(2+z);result=run_mode(1.,z)
    for row in result['rows']:
        assert row['endpoint_velocity_over_initial']==pytest.approx(r**row['step'],abs=2e-14)
        assert abs(row['update_error_over_initial'])<2e-14
        assert abs(row['normalized_balance_residual'])<2e-14
        assert row['cumulative_drag_over_initial']>=0
        assert row['kinetic_over_initial']+row['cumulative_drag_over_initial']==pytest.approx(1.,abs=2e-14)


def test_resolved_solution_converges_to_analytic_decay_at_fixed_time():
    errors=[]
    for n in [1,2,4]:
        row=run_mode(1.,.1/n,count=n)['rows'][-1]
        errors.append(abs(row['endpoint_velocity_over_initial']-np.exp(-.1)))
        # Fixed mass gives the same displacement error through q=(v0-v)/gamma.
        assert row['displacement_m']/1e-8==pytest.approx(1-row['endpoint_velocity_over_initial'],abs=2e-14)
    assert 3.9<errors[0]/errors[1]<4.1
    assert 3.9<errors[1]/errors[2]<4.1


def test_energy_pass_does_not_hide_wrong_direction_and_missing_damping():
    row=run_mode(1.,1e4,count=1)['rows'][0]
    assert row['analytical_velocity_over_initial']==0 # numerical underflow, analytic value positive
    assert row['endpoint_velocity_over_initial']<-.999
    assert row['mean_velocity_over_initial']>0
    assert row['kinetic_over_initial']>.999
    assert abs(row['normalized_balance_residual'])<2e-14
    assert row['displacement_m']>1.99*row['exact_displacement_m']
