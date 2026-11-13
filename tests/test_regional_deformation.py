import math
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tectonics.conforming_surface import ConformingSurface, normalize, spherical_measures
from tectonics.regional_deformation import DeformingSurface, DilationParameters
from test_conforming_surface import surface


def deforming(sub=1, amplitude=.1, length=1e6):
    return DeformingSurface.from_surface(surface(sub, amplitude), DilationParameters(length))


def test_zero_dilation_recovers_old_force_law_without_initial_energy_change():
    old=surface(2);new=DeformingSurface.from_surface(old,DilationParameters(0.))
    before=old.checkpoint();a=old.evaluate(old.vertices);b=new.evaluate(new.vertices)
    assert new.budget()==old.budget()
    assert b['velocity_per_s']==pytest.approx(a['velocity_per_s'],rel=2e-14,abs=1e-33)
    assert b['dissipation_w']==pytest.approx(a['dissipation_w'],rel=2e-15)
    assert old.checkpoint()==before


def test_area_rate_and_virtual_power_use_the_same_moving_geometry():
    obj=deforming(2,amplitude=.25);e=obj.evaluate(obj.vertices)
    direction=e['velocity_per_s']/np.max(np.linalg.norm(e['velocity_per_s'],axis=1))
    eps=1e-6
    plus=obj.evaluate(normalize(obj.vertices+eps*direction))
    minus=obj.evaluate(normalize(obj.vertices-eps*direction))
    numeric_area=(plus['area_m2']-minus['area_m2'])/(2*eps)
    expected=e['area_rate_m2_s']/np.max(np.linalg.norm(e['velocity_per_s'],axis=1))
    assert np.linalg.norm(numeric_area-expected)/np.linalg.norm(expected)<2e-9
    numeric_work=(plus['potential_j']-minus['potential_j'])/(2*eps)
    expected_work=-e['force_power_w']/np.max(np.linalg.norm(e['velocity_per_s'],axis=1))
    assert numeric_work==pytest.approx(expected_work,rel=2e-8)
    assert e['basal_dissipation_w']>0 and e['dilation_dissipation_w']>0
    assert abs(e['power_residual_w'])/e['dissipation_w']<2e-13
    assert abs(math.fsum(e['area_rate_m2_s']))<1e-13*sum(abs(e['area_rate_m2_s']))


def test_dilation_rotation_covariance_and_uniform_equilibrium():
    obj=deforming(2);e=obj.evaluate(obj.vertices);rot=Rotation.from_rotvec([.4,-.2,.7])
    other=obj.evaluate(rot.apply(obj.vertices))
    verror=np.linalg.norm(other['velocity_per_s']-rot.apply(e['velocity_per_s']))
    assert verror/np.linalg.norm(e['velocity_per_s'])<2e-12
    assert other['dissipation_w']==pytest.approx(e['dissipation_w'],rel=2e-13)
    quiet=deforming(2,amplitude=0);before=quiet.vertices.copy();quiet.advance(1.)
    assert np.max(abs(quiet.vertices-before))<1e-14
    assert quiet.budget()['heat_generated_j']/quiet.data['initial_potential_j']<1e-24


@pytest.mark.parametrize('length',[5e5,2e6])
def test_linear_spherical_harmonic_rate_has_physical_length_not_grid_length(length):
    errors=[]
    for sub in (1,2,3):
        obj=deforming(sub,amplitude=1e-4,length=length);p=obj.params;e=obj.evaluate(obj.vertices)
        h=e['height_m'][:,1];area=e['area_m2'];y=(3*obj.mesh().centroids[:,2]**2-1)/2
        J=h+h*h/p.radius_m+h**3/(3*p.radius_m**2)
        hdot=-J/(1+h/p.radius_m)**2*e['area_rate_m2_s']/area
        measured=-np.sum(area*y*hdot)/np.sum(area*y*(h-20000))
        J0=20000+20000**2/p.radius_m+20000**3/(3*p.radius_m**2)
        expected=p.basalt_density_kg_m3*p.gravity_m_s2*J0**2*6/(p.basal_drag_pa_s_m*(1+20000/p.radius_m)**2*p.radius_m**2)
        expected/=1+6*(length/p.radius_m)**2
        errors.append(abs(measured/expected-1))
    assert errors[-1]<errors[-2]<errors[0]
    assert errors[-1]<.012


def test_finite_work_convergence_and_material_memory_restart(tmp_path):
    errors=[]
    for dt in (2.,1.,.5):
        obj=deforming();before=obj.checkpoint()
        for _ in range(round(10/dt)):obj.step_fixed(dt)
        errors.append(abs(obj.budget()['energy_residual_j']))
        assert obj.budget()['external_work_j']==0
        assert obj.budget()['volume_m3']==obj.data['initial_volume_m3']
        assert obj.arrays['owner'].tolist()==before['arrays']['owner']
        assert abs(obj.budget()['area_relative_error'])<1e-15
    assert errors[0]>3*errors[1] and errors[1]>3*errors[2]
    obj=deforming();obj.advance(2.);obj.save(tmp_path/'first')
    restored=DeformingSurface.load(tmp_path/'first')
    obj.advance(4.);restored.advance(4.)
    assert obj.checkpoint()==restored.checkpoint()
    with pytest.raises(ValueError,match='version'):ConformingSurface.load(tmp_path/'first')


def test_rigid_rotation_has_no_dilation_dissipation():
    obj=deforming(2);area,basis,J=obj.rate_operator(obj.vertices)
    v=np.cross([.1,.2,-.3],obj.vertices)
    tangent=np.einsum('vjd,vj->vd',basis,v).ravel()
    assert np.linalg.norm(J@tangent)<1e-14
