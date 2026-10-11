"""Conservative differential native material transactions and restart controls."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import math

import pytest

from tectonics.fractional_surface import EXTENSIVE_FIELDS, SurfaceParcel
from tectonics.geometric_intrinsic_surface import enroll_intrinsic_surface
from tectonics.geometric_native_surface import (
    NativeSurfaceState, advance_native_surface, enroll_native_surface,
    load_native_checkpoint, native_surface_from_dict, native_surface_to_dict,
    save_native_checkpoint,
)
from tectonics.geometric_polarity import PolarityResolution
from tectonics.geometric_surface import save_geometric_checkpoint
from tectonics.geometric_transport import UnresolvedPolarityError
from test_geometric_precision import NEXT_DT_MYR, NEXT_OMEGA, thin_partition


def _source(*, older_thin=True, sparse_owners=False):
    state = thin_partition()
    return replace(state, fragments=tuple(replace(fragment, parcel=replace(fragment.parcel,
        age_myr=10. if older_thin and fragment.parcel.plate==1 else 5.,
        plate=(101 if fragment.parcel.plate==1 else 7) if sparse_owners
              else fragment.parcel.plate,
        material_fields=(('damage',(index+1)/16.),)))
        for index,fragment in enumerate(state.fragments)))


@pytest.fixture(scope='module')
def initial():
    return enroll_native_surface(_source())


def _snapshot(state):
    return json.dumps(native_surface_to_dict(state), sort_keys=True,
                      separators=(',', ':'), allow_nan=False)


def _birth(cell, plate, area, time, serial):
    return SurfaceParcel(cell, plate, f'native-control-birth:{time.hex()}:{serial}',
                         area, 2.*area, 0., 0., 0., (('damage',0.),))


def _declared_geometric_ordering(left, right, overlap):
    # This is an explicit geometric endpoint control. It supplies no force or
    # constitutive law for physical material pairs and chooses no side by ID.
    assert left.parcel.plate != right.parcel.plate
    return PolarityResolution(-1 if left.parcel.plate==1 else 1,
                              'declared_geometric_control_owner_1_remains_surface')


def _ledger(fragments):
    result = {}
    for fragment in fragments:
        for field in EXTENSIVE_FIELDS:
            result.setdefault((fragment.parcel.material_id,field),[]).append(
                getattr(fragment.parcel,field))
    return {key: math.fsum(values) for key,values in result.items()}


def _assert_per_material_budgets(before, result):
    original, retained = _ledger(before.fragments), _ledger(result.state.fragments)
    removed, born = _ledger(loss.fragment for loss in result.losses), _ledger(result.births)
    for key in original.keys()|retained.keys()|removed.keys()|born.keys():
        assert retained.get(key,0.)+removed.get(key,0.) == pytest.approx(
            original.get(key,0.)+born.get(key,0.), rel=5e-12, abs=0.)


def _step(state, *, full_source=False):
    owners = {fragment.parcel.plate for fragment in state.fragments}
    omega = ({owner:NEXT_OMEGA[0][owner] for owner in owners} if full_source else
             {owner:(0.,0.,.01 if owner==1 else 0.) for owner in owners})
    dt = NEXT_DT_MYR if full_source else 1e-5
    return advance_native_surface(state,omega,dt,birth_factory=_birth,
        polarity_resolver=None if full_source else _declared_geometric_ordering)


def test_enrollment_preserves_all_positive_shapes_and_material_histories(initial):
    source = _source()
    assert len(initial.fragments)==len(source.fragments)==9
    assert _ledger(initial.fragments)==_ledger(source.fragments)
    thin = next(fragment for fragment in initial.fragments if fragment.fragment_id=='thin')
    assert 0<thin.parcel.area_km2<1e-30
    assert thin.geometry.area_sr*initial.radius_km**2 == pytest.approx(
        thin.parcel.area_km2,rel=2e-12,abs=0.)
    for old,current in zip(source.fragments,initial.fragments):
        assert current.fragment_id==old.fragment_id
        assert current.parcel==old.parcel
        assert current.parent_fragment_id==old.parent_fragment_id
    assert enroll_native_surface(enroll_intrinsic_surface(source))==initial
    with pytest.raises(AttributeError):
        initial.time_myr=1.


def test_complete_actual_source_rotation_has_real_loss_birth_and_four_independent_ledgers(initial):
    before = _snapshot(initial)
    result = _step(initial,full_source=True)
    assert result.losses and result.births
    assert result.state.time_myr==initial.time_myr+NEXT_DT_MYR
    _assert_per_material_budgets(initial,result)
    assert _snapshot(initial)==before
    for loss in result.losses:
        assert loss.fragment.parcel.area_km2>0
        assert loss.receiver_plate != loss.fragment.parcel.plate
        assert loss.time_myr==result.state.time_myr
        assert loss.polarity_basis=='material_contrast'
        # Native geometric adjacency does not silently manufacture mechanical
        # attachment or a new physical fault pair.
        assert loss.attachment_status=='unresolved_native_support'
    originals = {fragment.fragment_id:fragment for fragment in initial.fragments}
    for newborn in result.births:
        assert newborn.parent_fragment_id in originals
        assert newborn.parcel.plate==originals[newborn.parent_fragment_id].parcel.plate
        assert newborn.parcel.age_myr==newborn.parcel.cold_mantle_volume_km3==0.
        assert newborn.parcel.density_excess_mass_kg==0.
        assert newborn.parcel.material_id not in initial.known_material_ids


def test_four_differential_native_cut_steps_preserve_every_ledger_and_resume_exactly(initial,tmp_path):
    state = initial
    all_losses,all_births = [],[]
    first_source = _snapshot(initial)
    halfway = None
    for index in range(4):
        result = _step(state)
        _assert_per_material_budgets(state,result)
        assert result.losses and result.births
        newborn_ids={fragment.fragment_id for fragment in result.births}
        retained_parents={fragment.parent_fragment_id for fragment in result.state.fragments
                          if fragment.fragment_id not in newborn_ids}
        assert retained_parents & {loss.source_fragment_id for loss in result.losses}
        assert all(fragment.parcel.area_km2>0 for fragment in result.state.fragments)
        sources={fragment.fragment_id:fragment for fragment in state.fragments}
        for loss in result.losses:
            source=sources[loss.source_fragment_id]
            assert loss.fragment.parcel.material_id==source.parcel.material_id
            assert loss.fragment.parcel.material_fields==source.parcel.material_fields
            assert loss.fragment.parcel.specific_properties==source.parcel.specific_properties
            assert loss.fragment.parcel.age_myr==source.parcel.age_myr+1e-5
        assert set(state.known_material_ids)<=set(result.state.known_material_ids)
        all_losses.extend(result.losses)
        all_births.extend(result.births)
        state = result.state
        if index==1:
            halfway=state
    assert halfway is not None
    assert _snapshot(initial)==first_source
    original,final = _ledger(initial.fragments),_ledger(state.fragments)
    lost,born = _ledger(loss.fragment for loss in all_losses),_ledger(all_births)
    for key in original.keys()|final.keys()|lost.keys()|born.keys():
        assert final.get(key,0.)+lost.get(key,0.) == pytest.approx(
            original.get(key,0.)+born.get(key,0.),rel=5e-12,abs=0.)
    provenance = dict(control='four differential native geometric steps',loaded_underthrust_steps=0)
    partial_path = save_native_checkpoint(tmp_path/'partial.json',halfway,provenance=provenance)
    resumed,metadata = load_native_checkpoint(partial_path)
    partial_resaved = save_native_checkpoint(tmp_path/'partial_resaved.json',resumed,provenance=metadata)
    assert partial_path.read_bytes()==partial_resaved.read_bytes()
    for _ in range(2):
        resumed=_step(resumed).state
    assert _snapshot(resumed)==_snapshot(state)
    direct_path = save_native_checkpoint(tmp_path/'direct_final.json',state,provenance=provenance)
    resumed_path = save_native_checkpoint(tmp_path/'resumed_final.json',resumed,provenance=metadata)
    assert direct_path.read_bytes()==resumed_path.read_bytes()
    final_restored,metadata = load_native_checkpoint(direct_path)
    final_resaved = save_native_checkpoint(tmp_path/'final_resaved.json',final_restored,provenance=metadata)
    assert direct_path.read_bytes()==final_resaved.read_bytes()


def test_real_polarity_ties_refuse_atomically_before_any_birth_call(tmp_path):
    tied = enroll_native_surface(_source(older_thin=False))
    before = _snapshot(tied)
    path = save_native_checkpoint(tmp_path/'accepted.json',tied)
    saved = path.read_bytes()
    calls=[]
    def forbidden_birth(*args):
        calls.append(args)
        raise AssertionError('Unresolved physical polarity cannot create newborn material')
    with pytest.raises(UnresolvedPolarityError) as failure:
        advance_native_surface(tied,NEXT_OMEGA[0],NEXT_DT_MYR,birth_factory=forbidden_birth)
    assert failure.value.overlaps and all(overlap.area_km2>0 for overlap in failure.value.overlaps)
    assert calls==[]
    assert _snapshot(tied)==before and path.read_bytes()==saved


@pytest.mark.parametrize('corruption',['material_area','duplicate_geometry','extra_geometry_receipt',
                                     'clock_bool','material_field_bool','extra_surface_field'])
def test_rehashed_checkpoint_still_validates_material_native_support_and_topology(initial,tmp_path,
        corruption):
    path = save_native_checkpoint(tmp_path/'accepted.json',initial)
    saved = path.read_bytes()
    payload = json.loads(saved)
    raw = payload['state']
    if corruption=='material_area':
        for field in EXTENSIVE_FIELDS:
            raw['fragments'][0]['parcel'][field]*=2.
    elif corruption=='duplicate_geometry':
        # Equal-area octants keep the global material sum unchanged. Only a
        # genuine geometry/topology audit can catch this duplicated footprint.
        raw['fragments'][1]['geometry']=deepcopy(raw['fragments'][0]['geometry'])
    elif corruption=='extra_geometry_receipt':
        raw['fragments'][0]['geometry']['trust_area_instead_of_support']=1.
    elif corruption=='clock_bool':
        raw['time_myr']=True
    elif corruption=='material_field_bool':
        raw['fragments'][0]['parcel']['material_fields']=[['damage',True]]
    else:
        raw['accept_without_topology']=True
    payload.pop('payload_sha256')
    encoded=json.dumps(payload,sort_keys=True,separators=(',',':'),ensure_ascii=False,
                       allow_nan=False).encode()
    payload['payload_sha256']=hashlib.sha256(encoded).hexdigest()
    corrupt=tmp_path/'rehashed_invalid.json'
    corrupt.write_text(json.dumps(payload,ensure_ascii=False,allow_nan=False)+'\n')
    with pytest.raises(ValueError):
        load_native_checkpoint(corrupt)
    assert path.read_bytes()==saved


def test_invalid_native_state_and_legacy_writer_refuse_before_creating_destination(initial,tmp_path):
    raw=native_surface_to_dict(initial)
    raw['fragments'][1]['geometry']=deepcopy(raw['fragments'][0]['geometry'])
    with pytest.raises(ValueError,match='overlap'):
        native_surface_from_dict(raw)
    forged=object.__new__(NativeSurfaceState)
    for name in NativeSurfaceState.__slots__:
        object.__setattr__(forged,name,getattr(initial,name))
    object.__setattr__(forged,'time_myr',True)
    invalid_path=tmp_path/'invalid_parent'/'checkpoint.json'
    with pytest.raises(ValueError):
        save_native_checkpoint(invalid_path,forged)
    assert not invalid_path.parent.exists()
    legacy_path=tmp_path/'legacy_parent'/'checkpoint.json'
    with pytest.raises((ValueError,TypeError)):
        save_geometric_checkpoint(legacy_path,initial)
    assert not legacy_path.parent.exists()


@pytest.mark.parametrize('omega,dt',[
    ({0:(0.,0.,0.)},NEXT_DT_MYR),
    ({0:(0.,0.,0.),1:(0.,0.,.01),2:(0.,0.,0.)},NEXT_DT_MYR),
    ({0:(0.,0.,0.),1:(True,0.,.01)},NEXT_DT_MYR),
    ({0:(0.,0.,0.),1:('0',0.,.01)},NEXT_DT_MYR),
    (NEXT_OMEGA[0],True),
    (NEXT_OMEGA[0],'0.01'),
    ({0:(0.,0.,0.),1:(0.,0.,1.)},.3),
])
def test_invalid_owner_mapping_clock_or_pose_domain_is_atomic(initial,omega,dt):
    before=_snapshot(initial)
    calls=[]
    def forbidden_birth(*args):
        calls.append(args)
        raise AssertionError('Invalid native motion cannot reach a material birth')
    with pytest.raises(ValueError):
        advance_native_surface(initial,omega,dt,birth_factory=forbidden_birth)
    assert calls==[] and _snapshot(initial)==before


def test_sparse_owner_labels_have_exact_mapping_coverage_without_dense_owner_array():
    state=enroll_native_surface(_source(sparse_owners=True))
    source_ledger=_ledger(state.fragments)
    vector=(.023,-.037,.049)
    result=advance_native_surface(state,{7:vector,101:vector},.71,birth_factory=_birth)
    assert result.losses==result.births==()
    assert _ledger(result.state.fragments)==source_ledger
    assert result.state.time_myr==.71
    assert {fragment.parcel.plate for fragment in result.state.fragments}=={7,101}


def test_exclusive_native_checkpoint_preserves_existing_file_and_roundtrip_is_exact(initial,tmp_path):
    path=save_native_checkpoint(tmp_path/'accepted.json',initial,provenance={'scope':'native geometry'})
    before=path.read_bytes()
    with pytest.raises(FileExistsError):
        save_native_checkpoint(path,initial)
    assert path.read_bytes()==before
    restored,metadata=load_native_checkpoint(path)
    resaved=save_native_checkpoint(tmp_path/'resaved.json',restored,provenance=metadata)
    assert path.read_bytes()==resaved.read_bytes()


@pytest.mark.parametrize('entry',['constructor','enrollment','dictionary','checkpoint'])
def test_positive_thin_partition_gap_is_rejected_despite_unchanged_global_float_area(initial,
        tmp_path,entry):
    thin=next(fragment for fragment in initial.fragments if fragment.fragment_id=='thin')
    remaining=tuple(fragment for fragment in initial.fragments if fragment.fragment_id!='thin')
    assert thin.geometry.area_sr>0 and thin.parcel.area_km2>0
    assert math.fsum(fragment.parcel.area_km2 for fragment in remaining)==math.fsum(
        fragment.parcel.area_km2 for fragment in initial.fragments)
    if entry=='constructor':
        with pytest.raises(ValueError):
            NativeSurfaceState(initial.time_myr,initial.radius_km,remaining,
                               initial.known_material_ids)
        return
    if entry=='enrollment':
        old=_source()
        old=replace(old,fragments=tuple(fragment for fragment in old.fragments
                                        if fragment.fragment_id!='thin'))
        with pytest.raises(ValueError):
            enroll_native_surface(old)
        return
    if entry=='dictionary':
        raw=native_surface_to_dict(initial)
        raw['fragments']=[row for row in raw['fragments'] if row['fragment_id']!='thin']
        with pytest.raises(ValueError):
            native_surface_from_dict(raw)
        return
    path=save_native_checkpoint(tmp_path/'accepted_complete.json',initial)
    before=path.read_bytes()
    payload=json.loads(before)
    payload['state']['fragments']=[row for row in payload['state']['fragments']
                                    if row['fragment_id']!='thin']
    payload.pop('payload_sha256')
    canonical=json.dumps(payload,sort_keys=True,separators=(',',':'),ensure_ascii=False,
                         allow_nan=False).encode()
    payload['payload_sha256']=hashlib.sha256(canonical).hexdigest()
    invalid=tmp_path/'rehashed_missing_thin.json'
    invalid.write_text(json.dumps(payload,sort_keys=True,separators=(',',':'),
                                  ensure_ascii=False,allow_nan=False)+'\n')
    with pytest.raises(ValueError):
        load_native_checkpoint(invalid)
    assert path.read_bytes()==before
