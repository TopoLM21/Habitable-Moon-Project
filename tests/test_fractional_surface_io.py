"""Independent import, minority-history, snapshot and refinement contracts."""
from collections import defaultdict
from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from tectonics.fractional_surface import FractionalSurfaceState, state_to_json, surface_totals
from tectonics.fractional_surface_io import (surface_from_lithosphere, save_fractional_checkpoint,
    load_fractional_checkpoint, refine_fractional_surface)
from tectonics.lithosphere import initialize_lithosphere
from tectonics.mesh import build_icosphere
from tectonics.plates import random_plate_system


EXTENSIVE=("area_km2","oceanic_volume_km3","cold_mantle_volume_km3","density_excess_mass_kg")


@pytest.fixture
def raster():
    mesh=build_icosphere(1)
    system=random_plate_system(mesh,3,413,0.,.1,.2)
    state=initialize_lithosphere(mesh,system,continental_fraction=0.,continental_nuclei=0)
    areas=mesh.physical_cell_areas_km2(1000.)
    n=mesh.cell_count
    state.time_myr=23.5
    state.oceanic_volume_km3=areas*np.linspace(1.1,2.2,n)
    state.mantle_lithosphere_thickness_km=np.linspace(0.,21.,n)
    state.mantle_lithosphere_density_anomaly_kg_m3=np.linspace(0.,71.,n)
    state.crust_age_myr=np.linspace(0.,80.,n)
    state.tidal_damage=np.linspace(.05,.65,n)
    memory=SimpleNamespace(damage=state.tidal_damage.copy(),cooling_stress_pa=np.linspace(-1e6,2e6,n),
        water_access=np.linspace(0.,1.,n),yield_ratio=np.linspace(0.,2.,n),strength_pa=np.linspace(1e6,5e6,n),
        eligible=np.arange(n)%2==0,consumed_band=np.arange(n)%3==0)
    return mesh,state,memory


def mixed_surface(raster):
    mesh,state,memory=raster
    surface=surface_from_lithosphere(mesh,state,1000.,fracture_memory=memory)
    original=surface.parcels[0]
    pieces=[]
    for fraction,age,material_id,plate,damage in ((.25,5.,"minority",7,.9),(.75,70.,"majority",1,.1)):
        pieces.append(replace(original,plate=plate,material_id=material_id,age_myr=age,
            material_fields=(("fracture_damage",damage),("fracture_cooling_stress_pa",damage*1e6)),
            **{name:fraction*getattr(original,name) for name in EXTENSIVE}))
    return replace(surface,parcels=tuple(pieces)+surface.parcels[1:],
        known_material_ids=tuple(set(surface.known_material_ids)|{"minority","majority","spent-material"}))


def by_material(state):
    result=defaultdict(lambda:np.zeros(4))
    for parcel in state.parcels:
        result[parcel.material_id]+=np.array([getattr(parcel,name) for name in EXTENSIVE])
    return result


def test_import_preserves_actual_extensive_material_age_and_fracture_fields_without_aliases(raster):
    mesh,state,memory=raster
    before=deepcopy(state)
    surface=surface_from_lithosphere(mesh,state,1000.,fracture_memory=memory)
    totals=surface_totals(surface)
    area=mesh.physical_cell_areas_km2(1000.)
    assert surface.time_myr==23.5
    assert totals["oceanic_volume_km3"]==pytest.approx(state.oceanic_volume_km3.sum(),rel=2e-14)
    assert totals["cold_mantle_volume_km3"]==pytest.approx(area@state.mantle_lithosphere_thickness_km,rel=2e-14)
    assert totals["density_excess_mass_kg"]==pytest.approx(np.sum(area*state.mantle_lithosphere_thickness_km
        *state.mantle_lithosphere_density_anomaly_kg_m3)*1e9,rel=2e-14)
    for parcel in surface.parcels:
        i=parcel.cell
        assert parcel.plate==state.cell_plate[i]
        assert parcel.age_myr==state.crust_age_myr[i]
        assert dict(parcel.material_fields)["fracture_cooling_stress_pa"]==memory.cooling_stress_pa[i]
        assert dict(parcel.material_fields)["fracture_eligible"]==float(memory.eligible[i])
    np.testing.assert_array_equal(state.oceanic_volume_km3,before.oceanic_volume_km3)
    frozen=state_to_json(surface)
    state.oceanic_volume_km3[:]=0.
    memory.cooling_stress_pa[:]=0.
    assert state_to_json(surface)==frozen


@pytest.mark.parametrize("field",["continental_fraction","continental_volume_km3","sediment_volume_km3"])
def test_import_refuses_invisible_non_oceanic_material_instead_of_discarding_it(raster,field):
    mesh,state,memory=raster
    values=np.zeros(mesh.cell_count)
    values[3]=1e-12
    setattr(state,field,values)
    with pytest.raises(ValueError,match=field):
        surface_from_lithosphere(mesh,state,1000.,fracture_memory=memory)


def test_import_refuses_inconsistent_fracture_damage(raster):
    mesh,state,memory=raster
    memory.damage[0]+=.01
    with pytest.raises(ValueError,match="damage disagree"):
        surface_from_lithosphere(mesh,state,1000.,fracture_memory=memory)


def test_independent_snapshot_preserves_minority_ownership_distinct_ages_and_histories(raster,tmp_path):
    mesh,_,_=raster
    state=mixed_surface(raster)
    path=tmp_path/"fractional.json"
    before=state_to_json(state)
    provenance=dict(source="unchanged-source",source_sha256="example",experiment="transport-only")
    save_fractional_checkpoint(path,mesh,state,1000.,provenance=provenance)
    loaded,actual_provenance=load_fractional_checkpoint(path,mesh,1000.)
    assert state_to_json(loaded)==before
    assert actual_provenance==provenance
    pieces=[p for p in loaded.parcels if p.cell==0]
    assert {(p.plate,p.age_myr) for p in pieces}=={(7,5.),(1,70.)}
    assert "spent-material" in loaded.known_material_ids
    assert state_to_json(state)==before
    saved=path.read_bytes()
    with pytest.raises(FileExistsError):
        save_fractional_checkpoint(path,mesh,state,1000.)
    assert path.read_bytes()==saved


@pytest.mark.parametrize("corruption",["format","radius","mesh","negative_material","valid_age_edit"])
def test_corrupt_snapshot_fails_closed(raster,tmp_path,corruption):
    mesh,_,_=raster
    state=mixed_surface(raster)
    path=save_fractional_checkpoint(tmp_path/"bad.json",mesh,state,1000.)
    data=json.loads(path.read_text(encoding="utf-8"))
    if corruption=="format":
        data["format"]="genesis-starter-continuation-0.2"
    elif corruption=="radius":
        data["radius_km"]*=2.
    elif corruption=="mesh":
        data["mesh_sha256"]="0"*64
    elif corruption=="negative_material":
        data["state"]["parcels"][0]["oceanic_volume_km3"]=-1.
    else:
        data["state"]["parcels"][0]["age_myr"]+=1.
    path.write_text(json.dumps(data),encoding="utf-8")
    with pytest.raises(ValueError):
        load_fractional_checkpoint(path,mesh,1000.)


def test_refinement_preserves_each_material_budget_and_unaveraged_histories(raster):
    mesh,_,_=raster
    state=mixed_surface(raster)
    snapshot=state_to_json(state)
    refined=refine_fractional_surface(state,mesh,build_icosphere(2),1000.)
    assert refined.known_material_ids==state.known_material_ids
    coarse_by_id,fine_by_id=by_material(state),by_material(refined)
    assert coarse_by_id.keys()==fine_by_id.keys()
    for material_id,totals in coarse_by_id.items():
        np.testing.assert_allclose(fine_by_id[material_id],totals,rtol=3e-14,atol=0.)
    for name in ("minority","majority"):
        old=next(p for p in state.parcels if p.material_id==name)
        children=[p for p in refined.parcels if p.material_id==name]
        assert len(children)==4
        assert {p.cell for p in children}=={0,1,2,3}
        assert all((p.plate,p.age_myr,p.material_fields)==(old.plate,old.age_myr,old.material_fields) for p in children)
    for cell,area in enumerate(refined.cell_areas_km2):
        assert sum(p.area_km2 for p in refined.parcels if p.cell==cell)==pytest.approx(area,rel=3e-14)
    assert state_to_json(state)==snapshot


def test_same_mesh_refinement_is_exact_and_coarsening_is_rejected(raster):
    mesh,_,_=raster
    state=mixed_surface(raster)
    assert state_to_json(refine_fractional_surface(state,mesh,mesh,1000.))==state_to_json(state)
    with pytest.raises(ValueError,match="coarsening"):
        refine_fractional_surface(state,mesh,build_icosphere(0),1000.)
