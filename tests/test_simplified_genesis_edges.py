from copy import deepcopy
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from analysis.simplified_genesis_validation.synthetic import deformable_edge_synthetic
from tectonics.simplified_genesis import SimplifiedGenesis
from tectonics.simplified_genesis_edges import enable,swept_mesh_certificate,_assembly
from tectonics.genesis_material import rebuild_material_mesh
from tectonics.genesis_transfer_bridge import encoded


def born():
    m=deformable_edge_synthetic()
    assert len(m.state['omega'])==1 and max(m.state['damage'])==0
    m.advance(25.);m.advance(25.)
    assert len(m.state['omega'])==3 and len(m.state['events'])==2
    return m


def test_deformed_three_domains_energy_momentum_mass_and_restart(tmp_path):
    m=born();enable(m,1);before=np.asarray(m.state['edge_material']['vertices']);oldmemory=deepcopy(m.state['memory'])
    ledger=m.advance(.01)
    assert ledger['relative_core_omega_rad_s']>1e-15 and ledger['max_strain']>1e-8
    assert ledger['max_core_geometric_strain']<1e-10
    assert ledger['angular_momentum_relative']<m.config['edge_strips']['relative_momentum_tolerance']
    assert abs(ledger['whole_balance_j'])<ledger['whole_allowance_j']
    assert abs(ledger['linear_balance_j'])<1e-5*ledger['work_scale_j']
    assert ledger['maxwell_heat_j']>0 and ledger['drag_heat_j']>0
    assert ledger['total_mass_kg']==sum(m.config['face_mass_kg'])
    assert ledger['minimum_thickness_m']>0 and ledger['whole_interval']['all_times']
    assert not ledger['whole_interval']['endpoint_sampling']
    assert oldmemory!=m.state['memory']
    after=np.asarray(m.state['edge_material']['vertices']);assert np.max(abs(after-before))>1e-9
    total=np.zeros(3)
    for row in ledger['normal_impulses']:
        assert row['normal_impulse_n_s']>=0
        normal=np.asarray(row['normal']);point=np.asarray(row['point_unit'])
        assert abs(normal@point)<1e-12
        impulse=normal*row['normal_impulse_n_s'];total-=impulse;total+=impulse
    np.testing.assert_array_equal(total,0)
    m.save(tmp_path/'edge.json');copy=SimplifiedGenesis.load(tmp_path/'edge.json')
    m.advance(.01);copy.advance(.01);assert encoded(m.state)==encoded(copy.state)


def test_boundary_width_sensitivity_is_exposed():
    first=born();second=SimplifiedGenesis(first.config,first.state)
    enable(first,1);enable(second,2)
    a=first.advance(.01);b=second.advance(.01)
    assert a['relative_core_omega_rad_s']>1e-15 and b['relative_core_omega_rad_s']>1e-15
    assert not np.isclose(a['relative_core_omega_rad_s'],b['relative_core_omega_rad_s'],rtol=1e-3,atol=0)
    assert sum(first.config['edge_strips']['strip_faces'])<sum(second.config['edge_strips']['strip_faces'])


def test_independent_traces_and_no_tangential_reaction():
    m=born();enable(m,1);mesh=rebuild_material_mesh(m.mesh,np.asarray(m.state['edge_material']['vertices']))
    _,_,_,_,maps,rows=_assembly(m,mesh)
    for row in rows:
        a,b=row['owners'];v=row['vertex'];left=maps[a,v][0];right=maps[b,v][0]
        assert not set(left)&set(right)
        n=np.array(row['normal']);t=np.cross(mesh.vertices[v],n)
        assert abs(n@t)<1e-14


def test_finite_path_rejects_safe_endpoints_with_internal_collapse():
    m=deformable_edge_synthetic();R=Rotation.from_rotvec([0,0,np.pi]).as_matrix()
    end=m.mesh.vertices@R.T;rebuild_material_mesh(m.mesh,end)
    with pytest.raises(ValueError,match='SWEPT_FACE'):swept_mesh_certificate(m.mesh,end-m.mesh.vertices)


def test_tensile_contact_refusal_is_transactional():
    m=born();m.config['edge_prescribed_compressive_pressure_pa']=0.;enable(m,1);before=encoded(m.state)
    with pytest.raises(ValueError,match='OPENING_OR_CONTACT_CHANGE'):m.advance(.01)
    assert encoded(m.state)==before
