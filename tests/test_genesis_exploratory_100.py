import copy,inspect,json
from pathlib import Path
import numpy as np
from tectonics.genesis import GenesisParameters
from analysis.genesis_exploratory_100.run import configuration
from analysis.genesis_exploratory_100.checks import thermal_check,assess
from analysis.genesis_coupled_split_control.mechanical import advance


def test_isolated_config_preserves_work_and_gate_requirements():
    c=configuration();v=json.loads(Path('analysis/genesis_adaptive_control/config.json').read_text())
    assert v['max_step_years']==50 and v['temperature_atol_k']==.05
    assert c['max_step_years']==100 and c['temperature_atol_k']==.5
    for k in ('global_work_atol_j_per_year','global_work_rtol_stored','local_work_atol_j_per_year','local_work_rtol_stored','damage_gate_margin','damage_error_fraction_of_margin'):assert c[k]==v[k]
    assert inspect.signature(advance).parameters['diagnostic_max_step_years'].default==50


def case():
    gp=GenesisParameters();T=np.array([1399.9,2000.,1000.]);m=np.ones(3)*1000
    h=np.array([1200*T[0],1200*T[1]+400000,4200*T[2]+2260000])
    g=dict(vertices_m=[[0,0,10],[1,0,10],[0,1,10]],faces=[[0,1,2]])
    s=dict(geometry=g,displacement_m=[0.]*18,fault_history=dict(damage=[.04],fault_active=[False],fault_candidate_age_myr=[0.]))
    p=dict(state=s,thermal=dict(mass_kg=m,energy_j=m*h))
    cuts=[dict(face=0,layer=0,material_id=1)]
    root=dict(source_packet=dict(source=dict(parameters=dict(shell=dict(density_kg_m3=3000.,conductivity_w_m_k=3.6),faults=dict(activation_damage=.65)))))
    return p,cuts,root,gp


def test_warm_neighbors_and_endpoint_signs_do_not_certify_path():
    p,cuts,root,gp=case()
    # Include a hot, unborn cut; two cuts plus the same reservoirs.
    p['thermal']['mass_kg']=np.ones(4)*1000
    p['thermal']['energy_j']=1000*np.array([1200*1399.9,1200*1700+400000*.5,1200*2000+400000,4200*1000+2260000])
    cuts.append(dict(face=0,layer=1,material_id=None))
    r=thermal_check(p,cuts,root,gp)
    assert r['unsupported']==[] and not r['full_path_certified']
    assert r['old_boundary_flux_upper_w']>0 and not r['old_one_sided_conduction_bound']
    p['thermal']['energy_j'][0]=1000*(1200*1401+400000/600)
    assert 'OLD_MATERIAL_AT_OR_ABOVE_SOLIDUS' in thermal_check(p,cuts,root,gp)['unsupported']


def test_displacement_is_a_new_explicit_output_budget():
    c=configuration()
    o=dict(ids=[1],stress_pa=np.zeros((1,9)),velocity_m_s=np.zeros((1,6,3)),damage=np.array([.04]),temperature_k=np.array([1000.]),enthalpy_j_kg=np.array([1.2e6]),local_work_j=np.zeros(1),local_stored_j=np.ones(1),physical_heat_j=np.zeros(1),numerical_kinetic_j=0.,global_stored_j=1.,physical_drag_j=0.,activation_damage=.65,gate_flags=False,displacement_increment_m=np.zeros((1,6,3)))
    fine=copy.deepcopy(o);fine['displacement_increment_m'][0,0,0]=.1
    r=assess(o,o,fine,100,c)
    assert r['checks']['displacement_m']['local_ratio']>1
