from copy import deepcopy
import pytest
from tectonics.genesis import GenesisParameters,ENERGY_SCALE
from analysis.genesis_energy_inventory.audit import thermal_budget,mechanical_budget,read_pinned


def heat_pair():
    params=dict(radiogenic_specific_power_w_kg=0.,stellar_flux_w_m2=0.)
    factor=GenesisParameters(**params).area_m2*ENERGY_SCALE
    before=dict(time_myr=0.,global_state={'energy':[0.,0.,0.,0.]},orbit={'dissipated_energy_j':0.},tidal_heat_received_j=0.,layer_mass_kg=[[2.]],column_enthalpy=[[10.]],boundary_energy_j=0.)
    after=deepcopy(before);after.update(time_myr=.00005,global_state={'energy':[100/factor,0.,100/factor,0.]},orbit={'dissipated_energy_j':100.},tidal_heat_received_j=100.,column_enthalpy=[[12.]],boundary_energy_j=4.)
    return before,after,params


def test_separate_heat_ledgers_and_unrecorded_terms():
    r=thermal_budget(*heat_pair())
    assert all(c['status']=='PASS' for c in r['checks'].values())
    assert r['column_delta_j']==4.
    assert r['global_input_j']==pytest.approx(100.)
    assert r['column_top_boundary_in_j'] is None
    assert r['mantle_to_surface_independently_recorded_j'] is None


def test_double_counted_tidal_input_detected_even_if_global_balance_closes():
    b,a,p=heat_pair();a['global_state']['energy']=[2*x for x in a['global_state']['energy']]
    r=thermal_budget(b,a,p)
    assert r['checks']['global_thermal']['status']=='PASS'
    assert r['checks']['source_decomposition']['status']=='FAIL'


def test_unrecorded_heat_injection_detected():
    b,a,p=heat_pair();a['column_enthalpy'][0][0]+=3.
    assert thermal_budget(b,a,p)['checks']['columns']['status']=='FAIL'


def test_unknown_mass_transfer_refused():
    b,a,p=heat_pair();a['layer_mass_kg'][0][0]=3.
    with pytest.raises(ValueError,match='mass'):thermal_budget(b,a,p)


def mechanical():
    return dict(kinetic_before_j=0.,kinetic_after_j=2.,elastic_before_j=20.,elastic_after_j=10.,external_work_j=4.,thermal_work_j=3.,birth_kinetic_inflow_j=2.,damage_energy_change_j=-1.,physical_drag_j=5.,physical_maxwell_j=7.,numerical_kinetic_j=1.,numerical_memory_j=3.,remelt_kinetic_outflow_j=0.,remelt_elastic_outflow_j=0.)


def test_mechanical_signs_and_unknown_counterentry():
    r=mechanical_budget(mechanical());assert r['check']['status']=='PASS'
    assert r['thermal_counterentry_j'] is None
    assert r['numerical_heat_conversion']=='forbidden'


def test_wrong_damage_sign_detected():
    l=mechanical();l['damage_energy_change_j']=1.
    assert mechanical_budget(l)['check']['status']=='FAIL'


@pytest.mark.parametrize('key',['physical_maxwell_j','numerical_memory_j','birth_kinetic_inflow_j'])
def test_missing_term_is_not_zero(key):
    l=mechanical();del l[key]
    with pytest.raises(KeyError):mechanical_budget(l)


def test_corrupt_saved_input_refused(tmp_path):
    p=tmp_path/'x';p.write_text('{}')
    with pytest.raises(ValueError,match='hash'):read_pinned(p,'0'*64)
