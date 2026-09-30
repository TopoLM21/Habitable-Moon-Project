"""Analytic checks for the offline, non-production strength sensitivity probe."""
import numpy as np
import pytest
from physics_strength_probe import extensional_yield_pa, extension_response


def test_pressure_integrated_mohr_coulomb_matches_linear_analytic_integral():
    h,crust,rho,g,mu,c=32000.,1000.,3000.,7.12,.2,1.4e6
    nodes,weights=np.polynomial.legendre.leggauss(16)
    z=crust+.5*h*(nodes+1.)
    y=extensional_yield_pa(rho*g*z,4.2e6,c,mu)
    phi=np.arctan(mu)
    mean_expected=2.*(c*np.cos(phi)+rho*g*(crust+.5*h)*np.sin(phi))/(1.+np.sin(phi))
    assert .5*np.dot(weights,y)==pytest.approx(mean_expected,rel=2e-15)


def test_newtonian_plane_strain_rate_matches_constant_viscosity_analytic_solution():
    eta=2e23;stress=3e6;width=10000.;h=20000.
    result=extension_response(stress*width*h,width,np.array([h]),np.array([eta]),np.array([50e6]))
    assert result['strain_rate_s']==pytest.approx(stress/(4.*eta),rel=2e-15)
    assert result['force_relative_residual']<2e-15


def test_fully_yielded_section_has_no_unique_steady_rate():
    result=extension_response(1e15,1000.,np.array([10000.]),np.array([1e22]),np.array([1e7]))
    assert result['status']=='at_or_above_idealized_plastic_capacity'
    assert result['strain_rate_s'] is None
    assert result['strain_time_myr'] is None


def test_compression_is_not_misreported_as_tensile_necking():
    result=extension_response(-1e15,1000.,np.array([10000.]),np.array([1e22]),np.array([1e7]))
    assert result['status']=='no_tensile_extension'
    assert result['strain_rate_s']==0.
