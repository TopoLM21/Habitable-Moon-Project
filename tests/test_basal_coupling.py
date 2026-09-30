"""The prescribed source and its current interaction have distinct meanings."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from tectonics.basal_coupling import (PRESCRIBED_BASAL_LAW, PrescribedBasalParameters,
    basal_coupling_fraction, basal_interaction_from_traction, prescribed_cell_basal_state,
    prescribed_vertex_traction, velocity_to_local_omega)
from tectonics.genesis import SECONDS_PER_MYR
from tectonics.genesis_shell import ShellParameters, mantle_traction
from tectonics.genesis_starter_material import (independent_mantle_omega,
    independent_mantle_source_omega, mantle_source_parameters)
from tectonics.mesh import build_icosphere


@pytest.fixture(scope="module")
def mesh():
    return build_icosphere(2)


@pytest.fixture
def parameters():
    return PrescribedBasalParameters(seed=17, convective_traction_pa=80000.,
        basal_drag_pa_s_m=1e14, traction_coupling_depth_km=2.)


def model_for(mesh, parameters):
    return SimpleNamespace(mesh=mesh,
        shell=ShellParameters(subdivisions=2, seed=99,
            convective_traction_pa=parameters.convective_traction_pa,
            traction_coupling_depth_km=parameters.traction_coupling_depth_km),
        parameters=SimpleNamespace(seed=parameters.seed, basal_drag_pa_s_m=parameters.basal_drag_pa_s_m),
        thermal=SimpleNamespace(radius_km=5290.), loading=SimpleNamespace(sample=lambda x: x))


def test_source_does_not_consult_partition_epoch_or_thermal_sample(mesh, parameters):
    model = model_for(mesh, parameters)
    first = independent_mantle_source_omega(model)
    early = SimpleNamespace(thermal_context=SimpleNamespace(lid_thickness_km=.216162257))
    late = SimpleNamespace(thermal_context=SimpleNamespace(lid_thickness_km=34.595806207))
    # The transmitted fields depend on current H. The new source does not.
    assert np.linalg.norm(independent_mantle_omega(model, late)) > 9*np.linalg.norm(
        independent_mantle_omega(model, early))
    model.loading.sample = lambda _: (_ for _ in ()).throw(AssertionError("sample accessed"))
    np.testing.assert_array_equal(independent_mantle_source_omega(model), first)
    assert "time" not in str(mantle_source_parameters(model).metadata())


def test_initial_transmitted_velocity_matches_existing_starter(mesh, parameters):
    model = model_for(mesh, parameters)
    h = .2161622569084551
    original = independent_mantle_omega(model,
        SimpleNamespace(thermal_context=SimpleNamespace(lid_thickness_km=h)))
    current = prescribed_cell_basal_state(mesh, parameters, h)
    np.testing.assert_allclose(velocity_to_local_omega(mesh.centroids,
        current.equilibrium_velocity_m_s, model.thermal.radius_km), original, rtol=3e-15, atol=1e-20)
    shell = replace(model.shell, seed=parameters.seed)
    np.testing.assert_array_equal(mantle_traction(mesh, shell, np.full(mesh.cell_count,h)),
        prescribed_vertex_traction(mesh, seed=parameters.seed,
            convective_traction_pa=parameters.convective_traction_pa, thickness_km=h,
            traction_coupling_depth_km=parameters.traction_coupling_depth_km))


def test_source_coupling_is_not_also_a_drag_multiplier(mesh, parameters):
    source = prescribed_cell_basal_state(mesh, parameters, 100.)
    moving = source.equilibrium_velocity_m_s
    zero_lid = prescribed_cell_basal_state(mesh, parameters, 0., moving)
    np.testing.assert_array_equal(zero_lid.transmitted_traction_pa, 0.)
    np.testing.assert_allclose(zero_lid.slip_traction_pa,
        -parameters.basal_drag_pa_s_m*moving, rtol=2e-15, atol=0)
    assert np.all(zero_lid.drag_dissipation_w_m2 >= 0.)
    # The traction-driven choice is explicit: damping a preexisting motion is
    # distinct from transmitting mantle driving. It is not beta*c*(um-up).


def test_zero_equilibrium_slip_has_zero_basal_force(mesh, parameters):
    frozen = prescribed_cell_basal_state(mesh, parameters, .6)
    comoving = prescribed_cell_basal_state(mesh, parameters, .6,
                                          frozen.equilibrium_velocity_m_s)
    np.testing.assert_array_equal(comoving.slip_traction_pa, 0.)
    np.testing.assert_allclose(comoving.source_power_w_m2,
        comoving.drag_dissipation_w_m2, rtol=4e-15, atol=1e-26)


def test_power_budget_identifies_external_source_and_nonnegative_dissipation(mesh, parameters):
    rng = np.random.default_rng(12)
    velocity = rng.normal(size=(mesh.cell_count,3))*1e-9
    velocity -= mesh.centroids*np.sum(velocity*mesh.centroids, axis=1)[:,None]
    result = prescribed_cell_basal_state(mesh, parameters, 2., velocity)
    np.testing.assert_allclose(result.net_plate_power_w_m2,
        result.source_power_w_m2-result.drag_dissipation_w_m2, rtol=2e-14, atol=1e-26)
    assert np.all(result.drag_dissipation_w_m2 >= 0.)


def test_nonuniform_coupling_precedes_vector_average(mesh, parameters):
    h = np.where(mesh.centroids[:,0] > .2, 12., .02)
    result = prescribed_cell_basal_state(mesh, parameters, h)
    vertex = prescribed_vertex_traction(mesh, seed=parameters.seed,
        convective_traction_pa=parameters.convective_traction_pa, thickness_km=h,
        traction_coupling_depth_km=parameters.traction_coupling_depth_km)
    expected = vertex[mesh.faces].sum(axis=1)/3.
    expected -= mesh.centroids*np.sum(expected*mesh.centroids,axis=1)[:,None]
    np.testing.assert_allclose(result.transmitted_traction_pa, expected, rtol=3e-15, atol=1e-11)
    naive = result.source_traction_pa*basal_coupling_fraction(h,2.)[:,None]
    assert np.max(np.linalg.norm(result.transmitted_traction_pa-naive,axis=1)) > 1000.
    np.testing.assert_allclose(np.sum(result.transmitted_traction_pa*mesh.centroids,axis=1),
        0., atol=2*np.finfo(float).eps*parameters.convective_traction_pa)


def test_thin_and_saturated_source_transmission_limits(mesh, parameters):
    zero = prescribed_cell_basal_state(mesh, parameters, 0.)
    np.testing.assert_array_equal(zero.equilibrium_velocity_m_s,0.)
    saturated = prescribed_cell_basal_state(mesh, parameters, 200.)
    np.testing.assert_array_equal(saturated.transmitted_traction_pa, saturated.source_traction_pa)
    h = 1e-9
    assert basal_coupling_fraction(h,2.) == pytest.approx(h/2., rel=1e-9)


def test_si_velocity_and_omega_units_roundtrip():
    x = np.array([[0.,0.,1.],[1.,0.,0.]])
    velocity = np.array([[2e-9,-3e-9,0.],[0.,-1e-9,4e-9]])
    radius = 5000.
    omega = velocity_to_local_omega(x, velocity, radius)
    recovered_m_s = np.cross(omega,x)*radius*1000./SECONDS_PER_MYR
    np.testing.assert_allclose(recovered_m_s,velocity,rtol=2e-15,atol=0)


def test_source_scale_and_drag_units(mesh, parameters):
    a = prescribed_cell_basal_state(mesh,parameters,3.)
    b = prescribed_cell_basal_state(mesh,replace(parameters,convective_traction_pa=160000.),3.)
    c = prescribed_cell_basal_state(mesh,replace(parameters,basal_drag_pa_s_m=2e14),3.)
    np.testing.assert_allclose(b.equilibrium_velocity_m_s,2*a.equilibrium_velocity_m_s,rtol=2e-15)
    np.testing.assert_allclose(c.equilibrium_velocity_m_s,.5*a.equilibrium_velocity_m_s,rtol=2e-15)
    np.testing.assert_array_equal(c.source_traction_pa,a.source_traction_pa)
    assert parameters.metadata()["law"] == PRESCRIBED_BASAL_LAW


@pytest.mark.parametrize("h,depth", [(-1,2),(np.nan,2),(1,0),(1,np.inf)])
def test_invalid_coupling_rejected(h,depth):
    with pytest.raises(ValueError):
        basal_coupling_fraction(h,depth)


@pytest.mark.parametrize("beta", [0,-1,np.nan,np.inf])
def test_invalid_drag_rejected(beta):
    with pytest.raises(ValueError):
        basal_interaction_from_traction(np.zeros((2,3)),np.zeros((2,3)),beta)
