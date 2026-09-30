"""Independent geometry, material, symmetry and temporal checks for fractional FV."""
from dataclasses import replace
import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.linalg import expm

from tectonics.fractional_surface import (FractionalSurfaceState, SurfaceParcel,
    TransferPiece, commit_surface, remap_surface)
from tectonics.fractional_transport import (_resolve, advance_fractional_transport,
    rigid_edge_area_fluxes)
from tectonics.fractional_surface_io import refine_fractional_surface
from tectonics.mesh import build_icosphere


def make_state(mesh, radius=100., owners=None):
    areas=mesh.physical_cell_areas_km2(radius)
    if owners is None:
        owners=(mesh.centroids[:,0]>0.).astype(int)
    parcels=tuple(SurfaceParcel(i,int(owners[i]),f'source:{i}',float(a),2.*a,10.*a,1e12*a,
        5.+i/10.,(('damage',i/len(areas)),('water',.2))) for i,a in enumerate(areas))
    return FractionalSurfaceState(0.,tuple(areas),parcels)


def birth(cell,plate,area,time,serial):
    return SurfaceParcel(cell,plate,f'birth:{time.hex()}:{serial}',area,2.*area,0.,0.,0.,(('damage',0.),('water',0.)))


def per_origin(parcels,field):
    groups={}
    for p in parcels:
        groups.setdefault(p.material_id,[]).append(getattr(p,field))
    return {key:math.fsum(value) for key,value in groups.items()}


def assert_material_balance(initial,result):
    for field in ('area_km2','oceanic_volume_km3','cold_mantle_volume_km3','density_excess_mass_kg'):
        old=per_origin(initial.parcels,field);new=per_origin(result.state.parcels,field)
        born=per_origin(result.births,field);lost=per_origin((x.parcel for x in result.losses),field)
        for identity in old.keys()|new.keys()|born.keys()|lost.keys():
            assert new.get(identity,0.)+lost.get(identity,0.)==pytest.approx(
                old.get(identity,0.)+born.get(identity,0.),rel=3e-13,abs=1e-10)


def test_edge_flux_matches_independent_geodesic_velocity_integral():
    mesh=build_icosphere(1);omega=np.array([[.013,-.029,.057]])
    flux=rigid_edge_area_fluxes(mesh,omega,300.)[0]
    for edge in (0,13,54,80):
        a,b,iu,iv=mesh.shared_edges[edge];u,v=mesh.vertices[[iu,iv]]
        angle=np.arccos(u@v); tangent=(v-(u@v)*u)/np.sin(angle)
        n=np.cross(u,v);n/=np.linalg.norm(n)
        n*=np.sign(n@(mesh.centroids[b]-mesh.centroids[a]))
        expected=300.**2*quad(lambda s:np.cross(omega[0],np.cos(s)*u+np.sin(s)*tangent)@n,
                              0.,angle,epsabs=1e-12)[0]
        assert flux[edge]==pytest.approx(expected,rel=2e-14,abs=2e-12)


@pytest.mark.parametrize('subdivisions',[0,1,2])
def test_rigid_flux_telescope_divergence_on_every_cell(subdivisions):
    mesh=build_icosphere(subdivisions);omega=np.random.default_rng(7).normal(size=(3,3))
    flux=rigid_edge_area_fluxes(mesh,omega,4150.)
    edges=np.asarray(mesh.shared_edges)
    for row in flux:
        divergence=np.bincount(edges[:,0],weights=row,minlength=mesh.cell_count)-np.bincount(edges[:,1],weights=row,minlength=mesh.cell_count)
        assert np.max(np.abs(divergence))/np.max(np.abs(row))<1e-14


def test_refined_uniform_rotation_has_no_quadrature_induced_self_subduction():
    # The actual saved50 sub4→sub5 failure arose because parent extensive area
    # was conserved while independently rounded child geometry defined a
    # different capacity. Refinement must persist one authoritative area.
    coarse,fine=build_icosphere(4),build_icosphere(5)
    initial=make_state(coarse,radius=4150.,owners=np.zeros(coarse.cell_count,dtype=int))
    refined=refine_fractional_surface(initial,coarse,fine,4150.)
    occupied=np.bincount([p.cell for p in refined.parcels],weights=[p.area_km2 for p in refined.parcels],minlength=fine.cell_count)
    assert np.max(np.abs(occupied/np.asarray(refined.cell_areas_km2)-1.))<8*np.finfo(float).eps
    result=advance_fractional_transport(fine,refined,np.array([[.0007,-.0004,.0002]]),4150.,1.,birth_factory=birth)
    assert result.losses==() and result.births==()
    assert_material_balance(initial,result)


def test_common_rotation_preserves_full_coverage_without_birth_or_subduction():
    mesh=build_icosphere(1);state=make_state(mesh)
    state=replace(state,parcels=tuple(replace(p,
        cold_mantle_volume_km3=p.cold_mantle_volume_km3*(1.+p.cell/100.),
        density_excess_mass_kg=p.density_excess_mass_kg*(1.+p.cell/50.),
        specific_properties=()) for p in state.parcels))
    result=advance_fractional_transport(mesh,state,np.array([[.01,.02,.04]]*2),100.,1.,birth_factory=birth)
    assert result.losses==() and result.births==()
    assert_material_balance(state,result)
    originals={p.material_id:p for p in state.parcels}
    for parcel in result.state.parcels:
        source=originals[parcel.material_id]
        assert parcel.material_fields==source.material_fields
        assert parcel.plate==source.plate
        assert parcel.age_myr==source.age_myr+1.
    assert any(len({p.plate for p in result.state.parcels if p.cell==cell})>1 for cell in range(mesh.cell_count))


def test_each_mixed_cell_component_follows_its_own_plate_velocity():
    mesh=build_icosphere(0);base=make_state(mesh)
    parcels=tuple(replace(p,plate=plate,material_id=f'{p.material_id}:{plate}',
        **{field:getattr(p,field)*.5 for field in ('area_km2','oceanic_volume_km3',
            'cold_mantle_volume_km3','density_excess_mass_kg')}) for p in base.parcels for plate in (0,1))
    state=replace(base,parcels=parcels,known_material_ids=())
    result=advance_fractional_transport(mesh,state,np.array([[0.,0.,.1],[0.,0.,0.]]),100.,1.,birth_factory=birth)
    assert not result.losses and not result.births
    originals={p.material_id:p for p in parcels}
    stationary=[p for p in result.state.parcels if p.plate==1]
    assert len(stationary)==mesh.cell_count
    assert all(p.cell==originals[p.material_id].cell for p in stationary)
    assert any(p.cell!=originals[p.material_id].cell for p in result.state.parcels if p.plate==0)
    assert_material_balance(state,result)


def test_positive_z_rotation_moves_tracer_centroid_toward_positive_y():
    mesh=build_icosphere(1);state=make_state(mesh,owners=np.zeros(mesh.cell_count,dtype=int))
    state=replace(state,parcels=tuple(replace(p,material_fields=(('tracer',1.+.2*mesh.centroids[p.cell,0]),)) for p in state.parcels))
    result=advance_fractional_transport(mesh,state,np.array([[0.,0.,.1]]),100.,.1,birth_factory=birth)
    def centroid(parcels):
        return sum((p.area_km2*dict(p.material_fields)['tracer']*mesh.centroids[p.cell] for p in parcels),np.zeros(3))
    assert centroid(result.state.parcels)[1]>centroid(state.parcels)[1]+1.


def test_zero_rotation_ages_material_once_and_has_no_new_material():
    mesh=build_icosphere(0);state=make_state(mesh)
    result=advance_fractional_transport(mesh,state,np.zeros((2,3)),100.,2.,birth_factory=birth)
    assert result.losses==() and result.births==()
    assert len(result.state.parcels)==len(state.parcels)
    for old,new in zip(state.parcels,result.state.parcels):
        assert new==replace(old,age_myr=old.age_myr+2.)


def test_convergence_and_divergence_close_every_original_material_budget():
    mesh=build_icosphere(1);state=make_state(mesh)
    result=advance_fractional_transport(mesh,state,np.array([[0.,0.,.05],[0.,0.,-.05]]),100.,1.,birth_factory=birth)
    assert result.losses and result.births
    assert_material_balance(state,result)
    assert result.diagnostics['created_area_km2']==pytest.approx(result.diagnostics['subducted_area_km2'],rel=2e-13)
    for lost in result.losses:
        assert math.fsum(weight for _,weight in lost.receiver_plate_fractions)==pytest.approx(1.)
        assert all(plate!=lost.parcel.plate for plate,_ in lost.receiver_plate_fractions)
    assert all(p.age_myr==0. and p.cold_mantle_volume_km3==0. and p.density_excess_mass_kg==0. for p in result.births)


def test_subcell_acceptance_is_continuous_before_any_raster_jump():
    mesh=build_icosphere(0);state=make_state(mesh)
    omega=np.array([[0.,0.,.05],[0.,0.,-.05]])
    rates=[]
    for dt in (.01,.005,.0025):
        result=advance_fractional_transport(mesh,state,omega,100.,dt,birth_factory=birth)
        assert result.diagnostics['substeps']==1
        assert result.diagnostics['subducted_area_km2']>0.
        assert max(p.parcel.area_km2 for p in result.losses)<min(state.cell_areas_km2)*.01
        rates.append(result.diagnostics['subducted_area_km2']/dt)
    np.testing.assert_allclose(rates,rates[0],rtol=2e-11)


def test_cfl_substeps_never_overspend_and_keep_birth_history():
    mesh=build_icosphere(0);state=make_state(mesh)
    result=advance_fractional_transport(mesh,state,np.array([[0.,0.,.3],[0.,0.,-.3]]),100.,5.,birth_factory=birth)
    assert result.diagnostics['substeps']>1
    assert result.diagnostics['max_outgoing_fraction']<=1.
    assert_material_balance(state,result)
    assert all(p.area_km2>0. for p in result.state.parcels)
    assert result.state.time_myr==pytest.approx(5.)
    assert len({p.material_id for p in result.births})==len(result.births)


def overlap_case(equal=False):
    state=FractionalSurfaceState(0.,(100.,100.),(
        SurfaceParcel(0,0,'a',100.,200.,1000.,1e14,10.,(('damage',.2),)),
        SurfaceParcel(1,1,'b',100.,200.,1000.,1e14 if equal else 1e13,10.,(('damage',.3),))))
    incoming=remap_surface(state,(TransferPiece(0,0,.5),TransferPiece(0,1,.5),TransferPiece(1,0,1.)),1.)
    retained,lost,born,records,_,_=_resolve(state,incoming,1.,birth,0)
    final=commit_surface(state,retained,lost,born,1.)
    return state,final,records,born


def test_overlap_rejects_actual_heavier_ocean_instead_of_plate_number():
    _,_,losses,_=overlap_case()
    assert len(losses)==1
    assert losses[0].parcel.material_id=='a'
    assert losses[0].parcel.area_km2==50.
    assert losses[0].parcel.material_fields==(('damage',.2),)


def test_identical_material_polarity_tie_splits_rejection_symmetrically():
    _,_,losses,_=overlap_case(equal=True)
    areas={x.parcel.material_id:x.parcel.area_km2 for x in losses}
    assert areas==pytest.approx({'a':50./3.,'b':100./3.})


def test_relabeling_plate_ids_changes_only_ids():
    mesh=build_icosphere(0);state=make_state(mesh);omega=np.array([[0.,0.,.04],[0.,0.,-.05]])
    first=advance_fractional_transport(mesh,state,omega,100.,1.,birth_factory=birth)
    changed=replace(state,parcels=tuple(replace(p,plate=1-p.plate) for p in state.parcels))
    second=advance_fractional_transport(mesh,changed,omega[::-1],100.,1.,birth_factory=birth)
    def summary(result):
        return {(p.cell,p.material_id if p.material_id.startswith('source') else 'born',p.area_km2,p.oceanic_volume_km3,p.cold_mantle_volume_km3) for p in result.state.parcels}
    assert summary(first)==summary(second)
    assert per_origin((x.parcel for x in first.losses),'area_km2')==per_origin((x.parcel for x in second.losses),'area_km2')


def test_fixed_grid_timestep_refinement_converges_to_semidiscrete_advection():
    mesh=build_icosphere(0);state=make_state(mesh,owners=np.zeros(mesh.cell_count,dtype=int))
    omega=np.array([[.013,.021,.05]]);flux=rigid_edge_area_fluxes(mesh,omega,100.)[0]
    areas=np.asarray(state.cell_areas_km2);generator=np.zeros((len(areas),len(areas)))
    for (a,b,_,_),value in zip(mesh.shared_edges,flux):
        source,target=(a,b) if value>0 else (b,a)
        rate=abs(value)/areas[source]
        generator[target,source]+=rate;generator[source,source]-=rate
    initial=areas*np.array([dict(p.material_fields)['damage'] for p in state.parcels])
    exact=expm(generator*2.)@initial
    errors=[]
    for dt in (1.,.5,.25):
        current=state
        for _ in range(round(2./dt)):
            result=advance_fractional_transport(mesh,current,omega,100.,dt,birth_factory=birth)
            assert not result.losses and not result.births
            current=result.state
        actual=np.zeros(len(areas))
        for parcel in current.parcels:
            actual[parcel.cell]+=parcel.area_km2*dict(parcel.material_fields)['damage']
        errors.append(np.linalg.norm(actual-exact))
    assert errors[1]<.55*errors[0] and errors[2]<.55*errors[1]


@pytest.mark.parametrize('dt',[0.,-1.,float('nan')])
def test_invalid_timestep_fails_without_mutating_input(dt):
    mesh=build_icosphere(0);state=make_state(mesh)
    with pytest.raises(ValueError):
        advance_fractional_transport(mesh,state,np.zeros((2,3)),100.,dt,birth_factory=birth)
    assert state.time_myr==0.
