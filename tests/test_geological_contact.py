from types import SimpleNamespace
import numpy as np
import pytest
from tectonics.geological_contact import adjacent_gap_owners
from tectonics.mesh import build_icosphere,connected_components
from analysis.geological_contact.sea_level import exact_sea
from tectonics.hydrosphere import solve_sea_level_material_m,HydrosphereParameters


def test_orphan_newborn_attaches_without_recoloring_surviving_material():
    # Geometric proximity can cross another plate; adjacency may not.
    mesh=SimpleNamespace(cell_count=4,neighbors=((1,),(0,2),(1,3),(2,)),
        centroids=np.array([[1.,0,0],[0,1.,0],[-1.,0,0],[1.,0,0]]))
    owner=np.array([0,1,1,0]);gap=np.array([False,False,False,True])
    out=adjacent_gap_owners(mesh,owner,gap)
    np.testing.assert_array_equal(out,[0,1,1,1])
    np.testing.assert_array_equal(out[~gap],owner[~gap])


def test_valid_newborn_attachment_is_byte_identical():
    mesh=build_icosphere(1);owner=np.zeros(mesh.cell_count,dtype=int)
    owner[mesh.centroids[:,0]<0]=1
    gap=np.zeros(mesh.cell_count,dtype=bool);gap[0]=True
    # Both hemisphere domains have surviving cells, so nothing is changed.
    np.testing.assert_array_equal(adjacent_gap_owners(mesh,owner,gap),owner)


def test_independent_wet_set_cubic_matches_bisection_and_water_volume():
    mesh=build_icosphere(1);f=np.linspace(0,1,mesh.cell_count)
    ocean=np.linspace(-6000,-1000,mesh.cell_count);cont=np.linspace(-1500,2000,mesh.cell_count)
    volume=9e8;r=5287
    result=exact_sea(mesh,f,ocean,cont,r,volume)
    native=solve_sea_level_material_m(mesh,f,ocean,cont,r,volume,HydrosphereParameters())
    assert abs(result['sea_m']-native)<1e-4
    assert abs(result['water_relative_error'])<1e-12


def test_continental_displacement_requires_explicit_scenario_choice():
    from tectonics.unified_geological_genesis import UnifiedGeologicalGenesis as U
    e=object.__new__(U);e.record={}
    with pytest.raises(TypeError):e.enable_mature_cycle()
    e.enable_mature_cycle(continental_displacement=None)
    assert e.record['continental_displacement'] is None
    e.enable_mature_cycle(continental_displacement='retire_displaced_basalt')
    assert e.record['continental_displacement']=='retire_displaced_basalt'
    with pytest.raises(ValueError):e.enable_mature_cycle(continental_displacement='silent')


def test_contact_audit_rejects_duplicate_material_even_when_owners_match():
    from tectonics.geological_contact import contact_inventory
    mesh=build_icosphere(0);owner=np.zeros(mesh.cell_count,dtype=int)
    state=SimpleNamespace(cell_plate=owner)
    source=np.arange(mesh.cell_count)
    trace={'rigid_transport':{0:{'source':source}},'winner_source':source.copy()}
    assert contact_inventory(mesh,state,state,trace,5287)['winner_duplicates']==0
    trace['winner_source'][1]=0
    with pytest.raises(ValueError,match='provenance'):
        contact_inventory(mesh,state,state,trace,5287)


def test_area_aware_displacement_does_not_retire_rigidly_remapped_basalt():
    from tectonics.lithosphere import continental_basalt_retention as retained
    area=np.array([2.,1.]);old=np.array([.2,0.]);source=np.array([1,0]);volume=np.array([1.,7.])
    new=np.array([0.,.4])  # same extensive continental area .4 on smaller target
    legacy=retained(area,old,source,new,volume,'retire_displaced_basalt')
    fixed=retained(area,old,source,new,volume,'retire_added_continental_area')
    assert legacy[1]==pytest.approx(.75)
    np.testing.assert_array_equal(fixed,[1.,1.])
    new[1]=.7  # additional continental area .3 fills half the remaining .6
    assert retained(area,old,source,new,volume,'retire_added_continental_area')[1]==pytest.approx(.5)


def test_area_mode_refuses_impossible_inherited_column_instead_of_mass_sink():
    from tectonics.lithosphere import continental_basalt_retention
    with pytest.raises(ValueError,match='raster packing'):
        continental_basalt_retention(np.array([2.,1.]),np.array([.8,0.]),np.array([1,0]),
            np.array([0.,1.]),np.array([1.,7.]),'retire_added_continental_area')
