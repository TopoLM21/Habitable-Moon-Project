"""Analytic prescribed collision checks for the opted-in material interface."""
from types import SimpleNamespace
import math
import numpy as np
import pytest
from analysis.geological_area_transport.experiment import fixture
from tectonics.geological_fractional import project
from tectonics.lithosphere import LithosphereState
from tectonics.plates import Plate,PlateSystem
from tectonics.geological_material import CONVERSION

def engine():
    mesh,r,patches,omega=fixture(0);owner=(mesh.centroids[:,0]<0).astype(int);n=mesh.cell_count
    system=PlateSystem(owner,tuple(Plate(i,int(np.flatnonzero(owner==i)[0]),np.array([0.,0.,1.]),omega[i][2]) for i in range(2)))
    state=LithosphereState(0.,owner,np.zeros(n,dtype=np.int8),np.zeros(n),np.ones(n)*7,np.zeros(n),
        oceanic_volume_km3=mesh.physical_cell_areas_km2(r)*7,continental_volume_km3=np.zeros(n))
    rows=[];parts={}
    for p in patches:
        cell=int(p.identity.split('-')[1]);rows.append(dict(id=p.identity,owner=p.owner,cell=cell,vertices=p.reference_vertices.tolist()))
        parts[p.identity]=[dict(id=p.identity,surface_patch=p.identity,mass_kg=p.mass_kg,energy_j=p.enthalpy_j,material_phase='mafic',composition='mafic')]
    record=dict(radius_km=r,fractional_surface=rows,fractional_generation=0,material_parcels=dict(node_parts=parts),
        thermal_inventory=dict(energy_j=[1e30,0],mass_kg=[1e24,1]))
    return SimpleNamespace(mesh=mesh,record=record,checkpoint=SimpleNamespace(state=state)),system,r

def test_actual_ocean_collision_retires_lune_mass_H_and_provenance():
    e,system,r=engine();before={p['id']:p.copy() for parts in e.record['material_parcels']['node_parts'].values() for p in parts}
    pre,trace=project(e,system,1.)
    d=trace['fractional_geometry'];expected=4*.01*r*r
    assert d['retired_area_km2']==pytest.approx(expected,rel=1e-11)
    assert d['created_area_km2']==pytest.approx(expected,rel=1e-11)
    assert trace['fractional_retired_volume']==pytest.approx(expected*7,rel=1e-11)
    assert d['max_identity_relative_error']<1e-12 and d['max_identity_H_relative_error']<1e-12
    assert d['mixed_owner_cells']>0
    active=[p for profile in trace['fractional_profiles'] for p in profile[0]]
    for identity,p in before.items():
        allparts=[q for q in active+trace['fractional_retired'] if q['id']==identity]
        assert math.fsum(q['mass_kg'] for q in allparts)==pytest.approx(p['mass_kg'],rel=1e-12)
        assert math.fsum(q['energy_j'] for q in allparts)==pytest.approx(p['energy_j'],rel=1e-12)
        assert all(q['composition']==p['composition'] for q in allparts)
    assert pre.oceanic_volume_km3.sum()==pytest.approx(4*np.pi*r*r*7,rel=1e-12)

def test_felsic_unsupported_state_is_refused_before_mutation():
    from copy import deepcopy
    e,system,r=engine();e.checkpoint.state.continental_volume_km3[0]=1.
    rows=deepcopy(e.record['fractional_surface'])
    with pytest.raises(ValueError,match='FRACTIONAL_FELSIC_INTERFACE_REQUIRED'):project(e,system,1.)
    assert e.record['fractional_surface']==rows

def test_real_skinny_patch_fails_closed_instead_of_renormalizing():
    import json
    from pathlib import Path
    from tectonics.geological_area_transport import MaterialPatch,transfer_rigid_patches
    from tectonics.mesh import build_icosphere
    d=json.loads((Path(__file__).parents[1]/'analysis/geological_fractional/precision_failure.json').read_text())
    p=MaterialPatch(d['identity'],0,np.array(d['vertices']),1e10,1e16,'mafic')
    with pytest.raises(ValueError,match='Geometric source area failed'):
        transfer_rigid_patches(build_icosphere(2),[p],{0:d['omega']},d['dt_myr'],d['radius_km'])
