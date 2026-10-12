from dataclasses import asdict,replace
import hashlib,json,math
import pytest
from tectonics.genesis import GenesisParameters
from tectonics.genesis_reservoir_thermal import Reservoir,CoupledThermalState,specific_h
from tectonics.genesis_reservoir_conduction import column_network,step
from analysis.genesis_reservoir_step_validation.run import independent


def fixture():
    p=GenesisParameters();parameters={'thermal':asdict(p),'shell':{'density_kg_m3':3000.,'conductivity_w_m_k':3.6}}
    root={'source_packet':{'source':{'parameters':parameters}},'runtime_state':{'geometry':{'vertices_m':[[0.,0.,10.],[2.,0.,10.],[0.,2.,10.]],'faces':[[0,1,2]]},'displacement_m':[0.]*18,'thermal':{'layer_mass_kg':[[6000.,6000.]],'time_myr':0.}}}
    raw=json.dumps(root).encode();cells=(Reservoir('deep_mantle','silicate',10000.,10000*specific_h(1650.,'silicate',p)),Reservoir('column:0:0','silicate',6000.,6000*specific_h(1500.,'silicate',p)),Reservoir('column:0:1','silicate',6000.,6000*specific_h(1700.,'silicate',p)),Reservoir('surface_water','water',1000.,1000*specific_h(700.,'water',p)))
    origin={'legacy_raw_sha256':hashlib.sha256(raw).hexdigest(),'legacy_mantle_temperature_k':1660.}
    return CoupledThermalState(0.,cells,json.dumps(asdict(p)),json.dumps(origin),math.fsum(c.energy_j for c in cells)),raw


def test_exact_inherited_stencil_and_simultaneous_conservation():
    s,raw=fixture();edges,g=column_network(s,raw)
    assert g['area_m2']==pytest.approx([2.])
    assert [e[2] for e in edges]==pytest.approx([14.4,7.2,14.4])
    after,j=step(s,raw,1.)
    audit=independent(s,after,raw,1.,j)
    assert audit['status']=='PASS'
    assert s.cells[0].energy_j<after.cells[0].energy_j


def test_independent_audit_detects_lost_boundary_counterentry():
    s,raw=fixture();after,j=step(s,raw,1.)
    cells=list(after.cells);cells[0]=s.cells[0]
    with pytest.raises(ValueError,match='audit'):independent(s,replace(after,cells=tuple(cells)),raw,1.,j)


def test_legacy_authentication():
    s,raw=fixture()
    with pytest.raises(ValueError,match='SHA'):column_network(s,raw+b' ')


def test_no_extra_global_or_cohort_thermal_reservoir():
    s,raw=fixture();extra=Reservoir('global_surface_duplicated','silicate',1.,10.)
    c=replace(s.cells[0],energy_j=s.cells[0].energy_j-10.)
    s=replace(s,cells=(c,*s.cells[1:],extra))
    with pytest.raises(ValueError,match='ownership'):column_network(s,raw)


def test_no_silent_mechanical_attachment():
    s,raw=fixture();s=replace(s,cells=(*s.cells,Reservoir('elastic:column:0:0','elastic',0.,0.)))
    with pytest.raises(ValueError,match='driver'):step(s,raw,1.)


def test_changed_column_mass_is_rejected():
    s,raw=fixture();cells=list(s.cells);cells[1]=replace(cells[1],mass_kg=6001.)
    with pytest.raises(ValueError,match='mass'):column_network(replace(s,cells=tuple(cells)),raw)
