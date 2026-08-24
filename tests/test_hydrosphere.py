import numpy as np

from tectonics.hydrosphere import (
    HydrosphereParameters,
    initialize_hydrosphere,
    diagnose_hydrosphere,
    solve_sea_level_m,
    water_volume_at_sea_level_km3,
)
from tectonics.lithosphere import initialize_lithosphere
from tectonics.mesh import build_icosphere
from tectonics.plates import random_plate_system
from tectonics.topography import TopographyState


def _fixture(sub=2):
    mesh=build_icosphere(sub)
    plates=random_plate_system(mesh,plate_count=6,seed=7,boundary_roughness=.25,min_speed_deg_per_myr=.1,max_speed_deg_per_myr=.2)
    state=initialize_lithosphere(mesh,plates,0.28,4,7.0,35.0,500.0,radius_km=5287.0)
    return mesh,state


def test_initial_calibration_returns_reference_sea_level():
    mesh,state=_fixture()
    # Deliberately simple mixed relief.
    elev=np.linspace(-5000.0,1200.0,mesh.cell_count)
    topo=TopographyState(0.0,elev)
    p=HydrosphereParameters(reference_sea_level_m=123.0)
    hydro=initialize_hydrosphere(mesh,topo,5287.0,p)
    d=diagnose_hydrosphere(mesh,state,topo,hydro,5287.0,p)
    assert abs(d.sea_level_m-123.0)<1e-3
    assert abs(d.relative_volume_error)<1e-8


def test_fixed_inventory_moves_sea_level_with_basin_geometry():
    mesh,state=_fixture()
    elev=np.linspace(-6000.0,1000.0,mesh.cell_count)
    topo=TopographyState(0.0,elev)
    p=HydrosphereParameters(reference_sea_level_m=0.0)
    hydro=initialize_hydrosphere(mesh,topo,5287.0,p)
    # Raise all terrain by 250 m. With exactly conserved water and identical
    # geometry shifted radially, sea level should rise almost the same amount.
    topo2=TopographyState(4.0,elev+250.0)
    d2=diagnose_hydrosphere(mesh,state,topo2,hydro,5287.0,p)
    assert 249.0 < d2.sea_level_m < 251.0
    assert abs(d2.relative_volume_error)<1e-8


def test_sea_level_solver_reproduces_requested_volume():
    mesh,_=_fixture()
    elev=np.sin(np.arange(mesh.cell_count))*1800.0-2200.0
    p=HydrosphereParameters()
    wanted=water_volume_at_sea_level_km3(mesh,elev,5287.0,345.0)
    solved=solve_sea_level_m(mesh,elev,5287.0,wanted,p)
    assert abs(solved-345.0)<1e-3


def test_land_ocean_fractions_close_budget():
    mesh,state=_fixture()
    elev=np.linspace(-4500.0,1500.0,mesh.cell_count)
    topo=TopographyState(0.0,elev)
    p=HydrosphereParameters()
    h=initialize_hydrosphere(mesh,topo,5287.0,p)
    d=diagnose_hydrosphere(mesh,state,topo,h,5287.0,p)
    assert abs(d.land_area_fraction+d.ocean_area_fraction-1.0)<1e-12
    assert 0.0 <= d.shallow_sea_area_fraction <= d.ocean_area_fraction
    assert 0.0 <= d.deep_ocean_area_fraction <= d.ocean_area_fraction

from tectonics.topography import TopographyParameters, equilibrium_elevation


def test_subgrid_hypsometry_keeps_continental_patch_exposed_inside_wet_mean_cell():
    mesh,state=_fixture(sub=1)
    radius=5287.0
    areas=mesh.physical_cell_areas_km2(radius)
    ptop=TopographyParameters()
    ph=HydrosphereParameters(reference_sea_level_m=0.0, subgrid_material_hypsometry=True)
    cell=int(np.flatnonzero(state.crust_type==0)[0])  # oceanic legacy cell
    state.crust_age_myr[cell]=80.0
    state.continental_fraction[cell]=0.8
    state.continental_volume_km3[cell]=areas[cell]*0.8*35.0
    # Build the material-aware scalar mean.  It is still below zero because the
    # unresolved oceanic 20% is several km deep.
    elev,_=equilibrium_elevation(mesh,state,[],ptop,radius)
    topo=TopographyState(0.0,elev)
    assert topo.elevation_m[cell] < 0.0

    # Use an explicit inventory calibrated at zero with subgrid patches.
    hydro=initialize_hydrosphere(mesh,topo,radius,ph,state,ptop)
    d=diagnose_hydrosphere(mesh,state,topo,hydro,radius,ph,ptop)
    assert abs(d.sea_level_m) < 1e-3
    # The continental patch in the mixed cell is exposed even though the scalar
    # area-mean height of that whole cell is submerged.
    assert d.exposed_continental_material_area_fraction > 0.0
    assert d.submerged_continental_material_area_fraction < float(np.sum(areas*state.continental_fraction)/np.sum(areas))


def test_subgrid_material_water_solver_reproduces_reference_level():
    mesh,state=_fixture(sub=2)
    radius=5287.0
    areas=mesh.physical_cell_areas_km2(radius)
    ptop=TopographyParameters()
    # Make several material-fringe cells so the subgrid solver is genuinely used.
    ocean=np.flatnonzero(state.crust_type==0)[:5]
    for j,cell in enumerate(ocean):
        f=0.15+0.15*j
        state.continental_fraction[cell]=f
        state.continental_volume_km3[cell]=areas[cell]*f*(32.0+j)
        state.crust_age_myr[cell]=40.0+10*j
    elev,_=equilibrium_elevation(mesh,state,[],ptop,radius)
    topo=TopographyState(0.0,elev)
    ph=HydrosphereParameters(reference_sea_level_m=87.0,subgrid_material_hypsometry=True)
    hydro=initialize_hydrosphere(mesh,topo,radius,ph,state,ptop)
    d=diagnose_hydrosphere(mesh,state,topo,hydro,radius,ph,ptop)
    assert abs(d.sea_level_m-87.0)<1e-3
    assert abs(d.relative_volume_error)<1e-8
    assert abs(d.land_area_fraction+d.ocean_area_fraction-1.0)<1e-12
