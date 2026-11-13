from dataclasses import replace
import json,math
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.spatial.transform import Rotation
from tectonics.conforming_surface import (ConformingSurface,SurfaceParameters,SurfaceControl,
    spherical_measures,column_potential,normalize)
from tectonics.mesh import build_icosphere
from tectonics.thermal import SECONDS_PER_MYR


def surface(sub=1,amplitude=.1,params=SurfaceParameters(),control=SurfaceControl()):
    mesh=build_icosphere(sub);area=mesh.areas_unit_sphere*params.radius_m**2
    h=20000*(1+amplitude*(3*mesh.centroids[:,2]**2-1)/2)
    volume=area*(h+h*h/params.radius_m+h**3/(3*params.radius_m**2))
    n=len(area);zero=np.zeros(n)
    arrays=dict(owner=(mesh.centroids[:,0]<0).astype(int),felsic_fraction=zero,
        felsic_volume_m3=zero,basalt_volume_m3=volume,felsic_age_moment_m3_myr=zero,
        basalt_age_moment_m3_myr=volume*40,potential=np.linspace(0,.2,n))
    return ConformingSurface(mesh,arrays,params,control)


def test_finite_column_potential_against_independent_radial_integral_and_force():
    area=np.array([1e12]);radius=5.287e6;rho=2900.;g=7.12;h=40000.
    volume=area*quad(lambda z:(1+z/radius)**2,0,h)[0]
    U,pressure,height=column_potential(area,volume,rho,g,radius)
    expected=rho*g*area*quad(lambda z:z*(1+z/radius)**2,0,h)[0]
    assert U==pytest.approx(expected,rel=2e-15);assert height==pytest.approx([h],rel=2e-15)
    epsilon=1e-5
    plus=column_potential(area*(1+epsilon),volume,rho,g,radius)[0]
    minus=column_potential(area*(1-epsilon),volume,rho,g,radius)[0]
    assert (plus-minus)/(2*epsilon*area)==pytest.approx(pressure,rel=3e-10)


def test_spherical_area_and_virtual_work_are_independently_differentiable():
    obj=surface(2);x=obj.vertices;faces=obj.faces
    d=np.random.default_rng(7).normal(size=x.shape);d-=x*np.sum(x*d,axis=1)[:,None]
    ev=obj.evaluate(x);eps=1e-6
    plus=obj.evaluate(normalize(x+eps*d));minus=obj.evaluate(normalize(x-eps*d))
    derivative=(plus['potential_j']-minus['potential_j'])/(2*eps)
    work=math.fsum((ev['gradient_j']*d).ravel())
    assert derivative==pytest.approx(work,rel=3e-8)
    area,grad=spherical_measures(x,faces)
    da=np.sum(grad*d[faces],axis=(1,2))
    difference=(spherical_measures(normalize(x+eps*d),faces)[0]-spherical_measures(normalize(x-eps*d),faces)[0])/(2*eps)
    assert difference==pytest.approx(da,rel=2e-8,abs=2e-10)
    assert math.fsum(area)==pytest.approx(4*math.pi,rel=2e-15)
    # Rotational invariance: total internal torque is zero without removing it.
    torques=np.sum(np.cross(x,ev['gradient_j']),axis=0)
    assert np.linalg.norm(torques)/np.sum(np.linalg.norm(ev['gradient_j'],axis=1))<1e-13


def test_uniform_state_is_quiet_and_common_rotation_does_no_work():
    quiet=surface(2,amplitude=0);initial=quiet.checkpoint();quiet.advance(10.)
    assert np.max(np.linalg.norm(quiet.vertices-np.array(initial['arrays']['vertices']),axis=1))<2e-14
    assert quiet.budget()['heat_generated_j']/quiet.data['initial_potential_j']<1e-24
    obj=surface(2);rot=Rotation.from_rotvec([.3,-.7,.6]);start=obj.evaluate(obj.vertices)
    moved=obj.evaluate(rot.apply(obj.vertices))
    assert moved['potential_j']==pytest.approx(start['potential_j'],rel=2e-15)
    force_error=np.max(np.linalg.norm(moved['gradient_j']-rot.apply(start['gradient_j']),axis=1))
    assert force_error/np.max(np.linalg.norm(start['gradient_j'],axis=1))<2e-12


def test_positive_drag_power_and_finite_step_energy_error_converge():
    errors=[]
    for dt in (2.,1.,.5):
        obj=surface(1);v0={k:v.copy() for k,v in obj.arrays.items()};start=obj.budget()
        for _ in range(round(10/dt)):obj.step_fixed(dt)
        end=obj.budget();errors.append(abs(end['energy_residual_j']))
        assert end['potential_j']<start['potential_j'];assert end['heat_generated_j']>0
        assert end['volume_m3']==start['volume_m3'];assert end['mass_kg']==start['mass_kg']
        assert abs(end['area_relative_error'])<3e-15
        assert obj.arrays['owner'].tolist()==v0['owner'].tolist()
        assert obj.arrays['potential'].tolist()==v0['potential'].tolist()
        assert obj.arrays['basalt_age_moment_m3_myr']==pytest.approx(v0['basalt_age_moment_m3_myr']+v0['basalt_volume_m3']*10,rel=2e-15)
    assert errors[0]/errors[1]>3 and errors[1]/errors[2]>3


def test_small_amplitude_spherical_diffusion_limit_converges_with_mesh():
    errors=[]
    for sub in (1,2,3):
        obj=surface(sub,amplitude=1e-4);p=obj.params;ev=obj.evaluate(obj.vertices)
        area,gradient=obj.geometry(obj.vertices)
        area_rate=np.sum(gradient*ev['velocity_per_s'][obj.faces],axis=(1,2))
        h=ev['height_m'][:,1];J=h+h*h/p.radius_m+h**3/(3*p.radius_m**2)
        hdot=-J/(1+h/p.radius_m)**2*area_rate/area
        y=(3*obj.mesh().centroids[:,2]**2-1)/2
        measured=-np.sum(area*y*hdot)/np.sum(area*y*(h-20000))
        J0=20000+20000**2/p.radius_m+20000**3/(3*p.radius_m**2)
        expected=p.basalt_density_kg_m3*p.gravity_m_s2*J0**2/(p.basal_drag_pa_s_m*(1+20000/p.radius_m)**2)*6/p.radius_m**2
        errors.append(abs(measured/expected-1))
    assert errors[2]<errors[1]<errors[0]
    assert errors[2]<.025


def test_restart_retains_geometry_materials_controller_and_work(tmp_path):
    obj=surface(1);obj.advance(5.);obj.save(tmp_path/'half')
    restart=ConformingSurface.load(tmp_path/'half')
    assert restart.checkpoint()==obj.checkpoint()
    obj.advance(10.);restart.advance(10.)
    assert restart.checkpoint()==obj.checkpoint()
    with pytest.raises(FileExistsError):restart.save(tmp_path/'half')
    bad=json.loads(json.dumps(obj.checkpoint()));bad['arrays']['basalt_volume_m3'][0]*=1.01
    with pytest.raises(ValueError,match='volume'):ConformingSurface.restore(bad)


def test_invalid_geometry_and_new_initialization_cannot_silently_repair_old_gaps():
    obj=surface(1);x=obj.vertices.copy();x[obj.faces[0,[0,1]]]=x[obj.faces[0,[1,0]]]
    with pytest.raises(ValueError,match='Inverted'):obj.evaluate(x)
    from test_regional_area import inventory
    inv=inventory();inv.coarse.time_myr=1.
    with pytest.raises(ValueError,match='zero-time'):ConformingSurface.from_area_inventory(inv)


def test_basal_parameter_changes_time_scale_not_available_energy():
    a=surface(1);b=surface(1,params=replace(a.params,basal_drag_pa_s_m=2e17))
    ea=a.evaluate(a.vertices);eb=b.evaluate(b.vertices)
    assert ea['potential_j']==eb['potential_j']
    assert ea['velocity_per_s']==pytest.approx(2*eb['velocity_per_s'],rel=2e-15)
    assert ea['dissipation_w']==pytest.approx(2*eb['dissipation_w'],rel=2e-15)


def test_tiny_terminal_interval_is_integrated_not_skipped():
    obj=surface(1);obj.advance(1.);target=np.nextafter(1.,2.)
    before=obj.data['accepted_steps'];obj.advance(float(target))
    assert obj.data['time_myr']==target and obj.data['accepted_steps']==before+1
    assert obj.data['history'][-1]['dt_myr']==target-1.


def test_prescribed_common_boundary_motion_accounts_external_work_and_history():
    obj=surface(1);before=obj.budget();volume=obj.arrays['basalt_volume_m3'].copy()
    obj.prescribed_step(np.array([[0.,0.,.02],[.01,.01,-.02]]),1.)
    b=obj.budget();assert b['external_work_j']!=0 and b['heat_generated_j']==0
    assert abs(b['energy_residual_j'])/abs(b['external_work_j'])<1e-9
    assert abs(b['area_relative_error'])<2e-15
    assert np.array_equal(volume,obj.arrays['basalt_volume_m3'])
    assert ConformingSurface.restore(obj.checkpoint()).checkpoint()==obj.checkpoint()
    assert obj.data['history'][-1]['mode']=='prescribed_welded_projection'


def test_uniform_prescribed_rotation_keeps_all_areas_and_energy():
    obj=surface(1);area=obj.evaluate(obj.vertices)['area_m2'].copy();initial=obj.budget()['potential_j']
    obj.prescribed_step(np.array([[.3,-.2,.1]]*2),1.)
    assert obj.evaluate(obj.vertices)['area_m2']==pytest.approx(area,rel=2e-15)
    assert abs(obj.budget()['external_work_j'])/initial<1e-15
