import numpy as np
import pytest
from analysis.genesis_retirement_diagnosis.run import integrate_linear,connected_solid_fraction

ARGS=(1200.,400000.,1400.,2000.)


def test_exact_enthalpy_and_melt_integral_across_both_phase_boundaries():
    h,phi=integrate_linear([0.,1.],[1300.,2100.],0.,1.,*ARGS)
    assert phi==pytest.approx(.5)
    assert h==pytest.approx(1200*1700+400000*.5)


def test_solidus_crossing_does_not_liquefy_whole_interval():
    h,phi=integrate_linear([0.,1.],[1400.,1400.12],0.,1.,*ARGS)
    assert phi==pytest.approx(.0001)
    assert 1-phi==pytest.approx(.9999)


def test_hot_cap_disconnects_but_does_not_melt_lower_solid():
    q=[0.,.25,.5,1.];before=[1300.]*4;after=[1500.,1300.,1300.,1300.]
    assert connected_solid_fraction(q,before,1400.)==1
    assert connected_solid_fraction(q,after,1400.)==0
    h0,phi0=integrate_linear(q,before,.25,1.,*ARGS)
    h1,phi1=integrate_linear(q,after,.25,1.,*ARGS)
    assert h1==h0 and phi0==phi1==0


def test_deep_boundary_change_cannot_move_first_top_crossing():
    q=[0.,.25,.75,1.]
    assert connected_solid_fraction(q,[1378.,1577.,1900.,1654.],1400.)==connected_solid_fraction(q,[1378.,1577.,1900.,1645.],1400.)


def test_centred_temperature_interpolation_does_not_preserve_cell_enthalpy():
    q=[0.,.25,.75,1.];T=[1300.,1500.,1700.,1700.]
    h,_=integrate_linear(q,T,0.,.5,*ARGS)
    FV=.5*(1200*1500+400000*(1500-1400)/600)
    assert not np.isclose(h,FV,rtol=1e-6)
