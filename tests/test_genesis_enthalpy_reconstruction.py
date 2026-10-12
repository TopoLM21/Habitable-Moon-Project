import math
import pytest
from tectonics.genesis import GenesisParameters,silicate_specific_enthalpy
from tectonics.genesis_enthalpy_reconstruction import EnthalpyColumn,require_resolved_mechanical_input


def test_unequal_volume_mass_weights_and_arbitrary_partition():
    c=EnthalpyColumn((2.,8.),(20.,240.),2.)
    assert c.bounds==(0.,.2,1.)
    mid=c.integrate(.1,.6)
    assert mid['mass_kg']==pytest.approx(5.)
    assert mid['volume_m3']==pytest.approx(2.5)
    assert mid['enthalpy_j']==pytest.approx(130.)
    parts=[c.integrate(a,b) for a,b in zip([0.,.1,.2,.6],[.1,.2,.6,1.])]
    assert math.fsum(x['enthalpy_j'] for x in parts)==pytest.approx(260.)


@pytest.mark.parametrize('before,after',[(1600.,1500.),(1500.,1600.),(1390.,1410.),(2010.,1990.)])
def test_heating_cooling_and_two_phase_mean_conservation(before,after):
    p=GenesisParameters();columns=[EnthalpyColumn((3.,),(3*silicate_specific_enthalpy(t,p),),3000.) for t in (before,after)]
    phases=[c.representative_phase(p) for c in columns]
    assert phases[0]['temperature_k'][0]==pytest.approx(before)
    assert phases[1]['temperature_k'][0]==pytest.approx(after)
    for c in columns:assert c.integrate()['enthalpy_j']==c.energy_j[0]
    assert (phases[1]['melt_fraction'][0]-phases[0]['melt_fraction'][0])*(after-before)>=0


def test_no_overshoot_positive_local_maximum_principle():
    c=EnthalpyColumn((1.,2.,1.),(10.,60.,20.),3.)
    for lo,hi,expected in [(0.,.25,10.),(.25,.75,30.),(.75,1.,20.)]:
        v=c.integrate(lo,hi)
        assert v['enthalpy_j']/v['mass_kg']==expected
    assert min(c.specific_h)>=0 and max(c.specific_h)==30.


def test_surface_value_cannot_replace_enthalpy_or_prove_retirement():
    p=GenesisParameters();c=EnthalpyColumn((1.,1.),(silicate_specific_enthalpy(1577.,p),silicate_specific_enthalpy(1800.,p)),3000.)
    assert c.connected_front_bracket(1377.,p)==c.connected_front_bracket(1378.,p)
    bracket=c.connected_front_bracket(1378.,p)
    assert bracket['q_min']<.02<bracket['q_max']
    assert c.integrate()['enthalpy_j']==sum(c.energy_j)


def test_hot_surface_only_certifies_connectivity_not_melt():
    p=GenesisParameters();c=EnthalpyColumn((1.,),(silicate_specific_enthalpy(1300.,p),),3000.)
    assert c.connected_front_bracket(1500.,p)['q_max']==0
    assert c.representative_phase(p)['melt_fraction']==[0.]


def test_below_solidus_cell_mean_does_not_prove_subcell_topology():
    p=GenesisParameters();c=EnthalpyColumn((1.,),(silicate_specific_enthalpy(1390.,p),),3000.)
    assert c.connected_front_bracket(1300.,p)['q_min']==0
    assert c.connected_front_bracket(1300.,p)['q_max']==1


def test_invalid_data_and_unresolved_guard():
    with pytest.raises(ValueError):EnthalpyColumn((1.,),(-1.,),3000.)
    with pytest.raises(ValueError):EnthalpyColumn((0.,),(1.,),3000.)
    with pytest.raises(ValueError,match='STOP'):require_resolved_mechanical_input({'mechanical_input_ready':True})
