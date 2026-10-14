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
        oceanic_volume_km3=mesh.physical_cell_areas_km2(r)*7,continental_volume_km3=np.zeros(n),continental_fraction=np.zeros(n))
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

def test_continental_contact_keeps_felsic_mass_and_H():
    e,system,r=engine();state=e.checkpoint.state
    state.continental_fraction[:]=.6;state.crust_type[:]=1
    state.continental_volume_km3[:]=e.mesh.physical_cell_areas_km2(r)*.6*35
    state.oceanic_volume_km3*=.4
    original=[]
    for group in e.record['material_parcels']['node_parts'].values():
        p=group[0];q=dict(p,id=p['id']+'-felsic',material_phase='felsic',composition='felsic',mass_kg=p['mass_kg']*3,energy_j=p['energy_j']*3)
        p['mass_kg']*=.4;p['energy_j']*=.4;group.append(q);original.append(q)
    pre,t=project(e,system,1.)
    assert not any(p['material_phase']=='felsic' for p in t['fractional_retired'])
    felsic=[p for profile in t['fractional_profiles'] for p in profile[1]]
    assert math.fsum(p['mass_kg'] for p in felsic)==pytest.approx(math.fsum(p['mass_kg'] for p in original),rel=1e-13)
    assert math.fsum(p['energy_j'] for p in felsic)==pytest.approx(math.fsum(p['energy_j'] for p in original),rel=1e-13)
    assert max(pre.continental_fraction)<=1


def test_real_skinny_patch_fails_closed_instead_of_renormalizing():
    import json
    from pathlib import Path
    from tectonics.geological_area_transport import MaterialPatch,transfer_rigid_patches
    from tectonics.mesh import build_icosphere
    d=json.loads((Path(__file__).parents[1]/'analysis/geological_fractional/precision_failure.json').read_text())
    p=MaterialPatch(d['identity'],0,np.array(d['vertices']),1e10,1e16,'mafic')
    with pytest.raises(ValueError,match='Geometric source area failed'):
        transfer_rigid_patches(build_icosphere(2),[p],{0:d['omega']},d['dt_myr'],d['radius_km'])


def test_thin_patch_bounded_correction_conserves_material():
    import json
    from pathlib import Path
    from tectonics.geological_area_transport import MaterialPatch,transfer_rigid_patches
    from tectonics.mesh import build_icosphere
    d=json.loads((Path(__file__).parents[1]/'analysis/geological_fractional/precision_failure.json').read_text())
    p=MaterialPatch(d['identity'],0,np.array(d['vertices']),1e10,1e16,'mafic')
    out=transfer_rigid_patches(build_icosphere(2),[p],{0:d['omega']},d['dt_myr'],d['radius_km'],roundoff_correction=True)
    assert math.fsum(x.mass_kg for x in out.pieces)==pytest.approx(p.mass_kg,rel=2e-16)
    assert math.fsum(x.enthalpy_j for x in out.pieces)==pytest.approx(p.enthalpy_j,rel=2e-16)
    assert abs(out.diagnostics['total_area_residual_km2'])<out.diagnostics['sources'][0]['roundoff_budget_km2']

def test_roundoff_budget_rejects_missing_support_and_real_area_change():
    from tectonics.geological_roundoff import conservative_weights,convex_roundoff
    from tectonics.spherical_polygons import normalize_polygon
    with pytest.raises(ValueError,match='No positive'):conservative_weights([],1.,1e-10)
    with pytest.raises(ValueError,match='exceeds boundary'):conservative_weights([.9],1.,1e-10)
    # A real concavity may not be concealed by the convex repair.
    v=np.array([[0.,0.,1.],[.1,0,1.],[.03,.03,1.],[0,.1,1.]])
    v/=np.linalg.norm(v,axis=1)[:,None]
    with pytest.raises(ValueError,match='exceeds roundoff'):convex_roundoff(v,5287.)
    with pytest.raises(ValueError):normalize_polygon([[1.,0,0],[1.,0,0],[1.,0,0]])


def test_fractional_macro_failure_rolls_back_geometry_and_material(source,monkeypatch):
    from tectonics.unified_geological_genesis import UnifiedGeologicalGenesis as U
    from tectonics.genesis_transfer_bridge import encoded
    import tectonics.geological_fractional as module
    e=U.from_checkpoint(source);e.enable_fractional_transport();before=encoded(e.record);cp=e.checkpoint
    def fail(trial,system,dt):
        trial.record['fractional_surface'][0]['owner']=999
        trial.record['material_parcels']['node_parts'].clear()
        raise ValueError('injected post-mutation failure')
    monkeypatch.setattr(module,'project',fail)
    with pytest.raises(ValueError,match='injected'):e.step(.1)
    assert e.checkpoint is cp and encoded(e.record)==before

# Shared realistic young-shell fixture; no external checkpoint required.
from test_unified_geological_genesis import source

def test_gnomonic_difference_of_thin_strip_stays_in_receiving_cell():
    from tectonics.geological_planar import intersection,difference
    from tectonics.spherical_polygons import polygon_area
    # Nearly coincident edges, including a duplicated endpoint after rounding.
    def rays(xy):
        a=np.c_[np.array(xy),np.ones(len(xy))];return a/np.linalg.norm(a,axis=1)[:,None]
    a=rays([[-.1,-.1],[.1,-.1],[.1,.1],[-.1,.1]])
    b=rays([[-.1+5e-9,-.1],[.1,-.1],[.1,.1],[.1,.1],[-.1+5e-9,.1]])
    overlap=intersection(a,b,[0,0,1]);rest=difference(a,b,[0,0,1])
    assert polygon_area(overlap)+math.fsum(polygon_area(p) for p in rest)==pytest.approx(polygon_area(a),rel=1e-14)
    for p in rest:
        assert np.max(abs(p[:,:2]/p[:,2,None]))<=.100000000000001
    assert len(rest)==1 and polygon_area(rest[0])>0
