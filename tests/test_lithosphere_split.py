import numpy as np
from tectonics.lithosphere import LithosphereState, CrustType, oceanic_thermal_lithosphere_total_thickness_km, refresh_mechanical_lithosphere, mantle_lithosphere_negative_buoyancy_proxy


def _state(ages, types=None):
    ages=np.asarray(ages,dtype=float); n=len(ages)
    if types is None: types=np.zeros(n,dtype=np.int8)
    frac=(np.asarray(types)==int(CrustType.CONTINENTAL)).astype(float)
    thick=np.where(frac>0,35.0,7.0)
    return LithosphereState(0.,np.zeros(n,dtype=np.int32),np.asarray(types,dtype=np.int8),ages,thick,np.zeros(n),continental_fraction=frac,continental_volume_km3=frac*thick)


def test_oceanic_thermal_lithosphere_grows_sqrt_age():
    h=oceanic_thermal_lithosphere_total_thickness_km(np.array([0.,25.,100.,160.]))
    assert h[0] == 0.0
    assert np.all(np.diff(h)>0)
    assert 105.0 < h[2] < 118.0
    assert h[-1] < 155.1


def test_crust_is_not_mechanical_lithosphere():
    st=_state([100.])
    refresh_mechanical_lithosphere(st,0.0)
    assert st.crust_thickness_km[0] == 7.0
    assert st.mantle_lithosphere_thickness_km[0] > 95.0
    assert st.mantle_lithosphere_thickness_km[0] != st.crust_thickness_km[0]


def test_old_ocean_has_greater_integrated_negative_buoyancy():
    st=_state([10.,120.])
    refresh_mechanical_lithosphere(st,0.0)
    b=mantle_lithosphere_negative_buoyancy_proxy(st)
    assert b[1] > b[0] > 0.0


def test_continental_rifting_thins_mantle_root_without_changing_crust_directly():
    st=_state([500.,500.], np.ones(2,dtype=np.int8))
    st.rift_extension=np.array([0.,1.])
    refresh_mechanical_lithosphere(st,0.0)
    assert st.mantle_lithosphere_thickness_km[1] < st.mantle_lithosphere_thickness_km[0]
    assert np.all(st.crust_thickness_km == 35.0)
    assert st.mantle_lithosphere_density_anomaly_kg_m3[0] < 0.0


def test_ocean_ocean_subduction_prefers_more_negatively_buoyant_mantle_lithosphere():
    from tectonics.dynamics import _choose_subducting_side
    from tectonics.kinematics import BoundaryRecord, BoundaryType
    st=_state([80.,80.])
    st.cell_plate=np.array([0,1],dtype=np.int32)
    st.mantle_lithosphere_thickness_km=np.array([40.,120.])
    st.mantle_lithosphere_density_anomaly_kg_m3=np.array([60.,60.])
    b=BoundaryRecord(0,1,0,1,0,1,np.array([1.,0.,0.]),-10.,0.,10.,BoundaryType.CONVERGENT)
    assert _choose_subducting_side(st,b)==1


def test_new_mantle_lithosphere_layer_does_not_double_count_bathymetry():
    from tectonics.mesh import build_icosphere
    from tectonics.plates import random_plate_system
    from tectonics.lithosphere import initialize_lithosphere
    from tectonics.topography import equilibrium_elevation,TopographyParameters
    mesh=build_icosphere(1); plates=random_plate_system(mesh,4,99,0.2,0.1,0.4)
    st=initialize_lithosphere(mesh,plates,0.25,2,radius_km=5287.0)
    before,_=equilibrium_elevation(mesh,st,[],TopographyParameters(),5287.0)
    st.mantle_lithosphere_thickness_km=np.linspace(0.,150.,mesh.cell_count)
    st.mantle_lithosphere_density_anomaly_kg_m3=np.linspace(-10.,65.,mesh.cell_count)
    after,_=equilibrium_elevation(mesh,st,[],TopographyParameters(),5287.0)
    assert np.array_equal(before,after)
