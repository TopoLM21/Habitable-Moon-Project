"""Actual native burial, donor history, whole material domains and live clamps."""
from copy import deepcopy
from dataclasses import replace
from functools import lru_cache
import json
import math

import pytest

from tests.underthrust_case import physical_inputs
from tectonics.fractional_surface import EXTENSIVE_FIELDS, SurfaceParcel
from tectonics.geometric_fault_motion_support import UnresolvedFaultMotionSupport
from tectonics.geometric_intrinsic_geometry import RotationIncrement
from tectonics.geometric_native_buried_support import (
    FORMAT, _hash, _record_body, _seal_record, _seal_registry, _seal_transaction,
    _transaction_body, advance_native_buried_registry, initial_native_buried_registry,
    inverse_native_history, native_buried_registry_from_dict,
    native_buried_registry_to_dict, native_pullback, native_surface_digest,
    validate_native_bending_bank_support, validate_native_buried_registry,
    validate_native_original_material_support,
)
from tectonics.geometric_native_surface import (
    advance_native_surface, enroll_native_surface, native_surface_to_dict,
)
from tectonics.geometric_polarity import PolarityResolution


def _birth(cell, owner, area, time, serial):
    return SurfaceParcel(cell, owner, f'native-bank-test:{time.hex()}:{serial}',
                         area, 2.*area, 0., 0., 0., (('damage',0.),))


def _ordering(left, right, overlap):
    # Declared geometric test control. No loaded force interval is inferred.
    return PolarityResolution(-1 if left.parcel.plate==0 else 1,
                              'declared_geometric_control_owner_0_remains_surface')


@lru_cache(maxsize=None)
def _initial():
    _, legacy, _, seed, _ = physical_inputs()
    origin = enroll_native_surface(legacy)
    return legacy, seed, origin, initial_native_buried_registry(origin)


@lru_cache(maxsize=None)
def _accepted(count=1, *, large=False, common=False):
    legacy, seed, origin, registry = _initial()
    current = origin
    histories = dict(registry.owner_histories)
    results = []
    for index in range(count):
        if common:
            vector = (.1,.2,-.3) if index%2==0 else (-.3,.1,.2)
            omega, dt = {0:vector,1:vector}, .01
        else:
            omega = {0:(0.,0.,0.),1:(0.,0.,.1 if large else .01)}
            dt = 1. if large else 1e-5
        result = advance_native_surface(current,omega,dt,birth_factory=_birth,
                                        polarity_resolver=_ordering)
        histories = {owner:history+(RotationIncrement(omega[owner],dt),)
                     for owner,history in histories.items()}
        registry = advance_native_buried_registry(origin,registry,current,result,histories)
        current = result.state
        results.append(result)
    return legacy, seed, origin, current, registry, tuple(results)


def _rebind_records(registry, changed):
    """Resign altered payloads completely; downstream physical proofs must refuse."""
    mapping, records = {}, []
    for old, record in zip(registry.records, changed):
        digest = _hash(_record_body(record)['loss'])
        record = _seal_record(replace(record,payload_sha256=digest,
                                     event_id='native-accepted-loss:'+digest))
        mapping[old.event_id] = record.event_id
        records.append(record)
    transactions=[]
    for transaction in registry.transactions:
        transaction=replace(transaction,event_ids=tuple(sorted(mapping[identity]
                                                for identity in transaction.event_ids)))
        body=_transaction_body(transaction)
        body.pop('transaction_id')
        transactions.append(_seal_transaction(replace(transaction,
            transaction_id='native-burial-transaction:'+_hash(body))))
    return _seal_registry(replace(registry,records=tuple(records),transactions=tuple(transactions)))


def test_initial_native_support_certifies_all_material_reservoirs_and_eight_whole_clamps():
    legacy,seed,origin,registry=_initial()
    receipt=validate_native_bending_bank_support(origin,seed,origin,registry,legacy_origin=legacy)
    assert len(receipt.live_clamps)==8==2*len(seed.fault_traces.traces)
    assert len({(clamp.trace_id,clamp.bank) for clamp in receipt.live_clamps})==8
    for clamp in receipt.live_clamps:
        assert clamp.enclosing_geometry.area_sr>0
        assert clamp.live_original_fragment_ids
        assert clamp.exact_width_radians[0]>0
        assert 'w<tan(w)<4w' in clamp.analytic_enclosure
    material=validate_native_original_material_support(origin,origin,registry)
    assert len(material.original_material_domains)==8
    for domain in material.original_material_domains:
        assert set(dict(domain.conserved_reservoirs))==set(EXTENSIVE_FIELDS)
        assert domain.live_fragment_ids and domain.buried_event_ids==()
    assert 'separate_live_continuous_clamp' in material.live_clamp_support


def test_two_actual_native_cuts_keep_full_loss_payloads_donor_histories_and_live_clamps():
    legacy,seed,origin,current,registry,results=_accepted(2)
    assert all(result.losses and result.births for result in results)
    assert len(registry.records)==sum(len(result.losses) for result in results)
    assert len(registry.transactions)==2
    records={record.payload_sha256:record for record in registry.records}
    for result in results:
        for loss in result.losses:
            record=records[_hash(_record_body(next(record for record in registry.records
                                                 if record.loss==loss))['loss'])]
            assert record.loss==loss
            assert record.donor_history[-1].omega_rad_per_myr==(0.,0.,.01)
            assert record.original_geometry==native_pullback(loss.fragment.geometry,record.donor_history)
            assert loss.fragment.parcel.area_km2>0
    receipt=validate_native_bending_bank_support(origin,seed,current,registry,legacy_origin=legacy)
    assert len(receipt.live_clamps)==8
    material=validate_native_original_material_support(origin,current,registry)
    assert any(domain.buried_event_ids for domain in material.original_material_domains)
    for domain in material.original_material_domains:
        for field,total in domain.conserved_reservoirs:
            live=math.fsum(getattr(fragment.parcel,field) for fragment in current.fragments
                          if fragment.parcel.material_id==domain.material_id)
            buried=math.fsum(getattr(record.loss.fragment.parcel,field) for record in registry.records
                            if record.loss.fragment.parcel.material_id==domain.material_id)
            assert live+buried==pytest.approx(total,rel=2e-13,abs=0.)


def test_noncommuting_common_motion_preserves_full_history_but_inverse_reference_is_exact():
    legacy,seed,origin,current,registry,results=_accepted(2,common=True)
    assert registry.records==() and all(result.losses==result.births==() for result in results)
    for owner,history in registry.owner_histories:
        assert len(history)==2 and history[0].omega_rad_per_myr!=history[1].omega_rad_per_myr
        inverse=inverse_native_history(history)
        assert tuple(step.dt_myr for step in inverse)==tuple(step.dt_myr for step in reversed(history))
        assert inverse[0].omega_rad_per_myr==tuple(-value for value in history[-1].omega_rad_per_myr)
    receipt=validate_native_bending_bank_support(origin,seed,current,registry,legacy_origin=legacy)
    assert len(receipt.live_clamps)==8
    assert tuple(len(history) for _,history in receipt.owner_histories)==(2,2)


def test_native_registry_json_roundtrip_is_exact_and_resumed_acceptance_matches_direct():
    legacy,seed,origin,current,registry,results=_accepted(2)
    raw=native_buried_registry_to_dict(registry)
    assert raw['format']==FORMAT
    encoded=json.dumps(raw,sort_keys=True,separators=(',',':'),allow_nan=False)
    restored=native_buried_registry_from_dict(json.loads(encoded),origin,current)
    assert restored==registry
    assert json.dumps(native_buried_registry_to_dict(restored),sort_keys=True,
                      separators=(',',':'),allow_nan=False)==encoded
    _,_,_,halfway,partial,_=_accepted(1)
    partial=native_buried_registry_from_dict(json.loads(json.dumps(native_buried_registry_to_dict(partial))),
                                           origin,halfway)
    omega={0:(0.,0.,0.),1:(0.,0.,.01)};dt=1e-5
    result=advance_native_surface(halfway,omega,dt,birth_factory=_birth,polarity_resolver=_ordering)
    history={owner:history+(RotationIncrement(omega[owner],dt),)
             for owner,history in partial.owner_histories}
    resumed=advance_native_buried_registry(origin,partial,halfway,result,history)
    assert resumed==registry


def test_buried_and_newborn_material_cannot_replace_live_original_clamp_support():
    legacy,seed,origin,current,registry,_=_accepted(large=True)
    material=validate_native_original_material_support(origin,current,registry)
    assert any(domain.buried_event_ids for domain in material.original_material_domains)
    assert registry.records and current.known_material_ids!=origin.known_material_ids
    with pytest.raises(UnresolvedFaultMotionSupport,match='LIVE original'):
        validate_native_bending_bank_support(origin,seed,current,registry,legacy_origin=legacy)


@pytest.mark.parametrize('corruption',['donor_history','original_geometry','material_intensive','loss_time'])
def test_fully_resigned_burial_payload_still_refuses_forged_material_ancestry(corruption):
    _,_,origin,current,registry,_=_accepted()
    changed=list(registry.records)
    record=changed[0]
    if corruption=='donor_history':
        record=replace(record,donor_history=(RotationIncrement((1e-40,0.,.01),1e-5),))
    elif corruption=='original_geometry':
        record=replace(record,original_geometry=origin.fragments[0].geometry)
    elif corruption=='material_intensive':
        parcel=record.loss.fragment.parcel
        parcel=replace(parcel,cold_mantle_volume_km3=2.*parcel.cold_mantle_volume_km3,
                       density_excess_mass_kg=2.*parcel.density_excess_mass_kg,
                       specific_properties=(parcel.specific_properties[0],
                                            2.*parcel.specific_properties[1],2.*parcel.specific_properties[2]))
        record=replace(record,loss=replace(record.loss,fragment=replace(record.loss.fragment,parcel=parcel)))
    else:
        record=replace(record,loss=replace(record.loss,time_myr=record.loss.time_myr+.01))
    changed[0]=record
    forged=_rebind_records(registry,changed)
    with pytest.raises(ValueError):
        validate_native_buried_registry(origin,forged,current)
    with pytest.raises(ValueError):
        native_buried_registry_from_dict(native_buried_registry_to_dict(forged),origin,current)


def test_fully_resigned_transaction_identity_is_authenticated():
    _,_,origin,current,registry,_=_accepted()
    first=_seal_transaction(replace(registry.transactions[0],transaction_id='invented-transaction'))
    forged=_seal_registry(replace(registry,transactions=(first,)))
    with pytest.raises(ValueError,match='identity'):
        validate_native_buried_registry(origin,forged,current)


def test_registry_refuses_missing_loss_even_with_resigned_transaction_and_registry():
    _,_,origin,current,registry,_=_accepted()
    omitted=registry.records[0].event_id
    transaction=replace(registry.transactions[0],event_ids=tuple(identity for identity in
                                                               registry.transactions[0].event_ids if identity!=omitted))
    body=_transaction_body(transaction);body.pop('transaction_id')
    transaction=_seal_transaction(replace(transaction,transaction_id='native-burial-transaction:'+_hash(body)))
    forged=_seal_registry(replace(registry,records=registry.records[1:],transactions=(transaction,)))
    with pytest.raises(ValueError,match='missing positive material support'):
        validate_native_buried_registry(origin,forged,current)


def test_clamp_proof_authenticates_initial_seed_origin_and_exact_width_domain():
    legacy,seed,origin,registry=_initial()
    with pytest.raises(ValueError,match='legacy-origin'):
        validate_native_bending_bank_support(origin,seed,origin,registry)
    with pytest.raises(ValueError,match='mechanical configuration'):
        validate_native_bending_bank_support(origin,replace(seed,surface_sha256='0'*64),origin,
                                            registry,legacy_origin=legacy)
    with pytest.raises(ValueError,match='physical widths'):
        parameter=seed.bending_parameters[0]
        altered=replace(parameter,sub=replace(parameter.sub,width_m=parameter.sub.width_m*1.1))
        validate_native_bending_bank_support(origin,replace(seed,bending_parameters=
            (altered,)+seed.bending_parameters[1:]),origin,registry,legacy_origin=legacy)


@pytest.mark.parametrize('bad_owner',[True,'0',-.5])
def test_donor_history_map_rejects_coerced_owner_types(bad_owner):
    _,_,origin,registry=_initial()
    result=advance_native_surface(origin,{0:(0.,0.,0.),1:(0.,0.,.01)},1e-5,
                                  birth_factory=_birth,polarity_resolver=_ordering)
    with pytest.raises(ValueError):
        advance_native_buried_registry(origin,registry,origin,result,
            {bad_owner:(RotationIncrement((0.,0.,0.),1e-5),),1:(RotationIncrement((0.,0.,.01),1e-5),)})


def test_invalid_acceptance_is_atomic_for_registry_and_surface():
    _,_,origin,registry=_initial()
    before=json.dumps(native_buried_registry_to_dict(registry),sort_keys=True)
    surface=json.dumps(native_surface_to_dict(origin),sort_keys=True)
    result=advance_native_surface(origin,{0:(0.,0.,0.),1:(0.,0.,.01)},1e-5,
                                  birth_factory=_birth,polarity_resolver=_ordering)
    with pytest.raises(ValueError,match='single physical clock'):
        advance_native_buried_registry(origin,registry,origin,result,
            {0:(RotationIncrement((0.,0.,0.),2e-5),),1:(RotationIncrement((0.,0.,.01),2e-5),)})
    assert json.dumps(native_buried_registry_to_dict(registry),sort_keys=True)==before
    assert json.dumps(native_surface_to_dict(origin),sort_keys=True)==surface


@pytest.mark.parametrize('unsupported',['not_quarter_frame','width_domain'])
def test_quarter_clamp_certificate_explicitly_refuses_unsupported_physical_domain(unsupported):
    from tectonics.geometric_native_buried_support import _quarter_clamp_envelope
    legacy,seed,origin,registry=_initial()
    trace=seed.fault_traces.traces[0]
    width=seed.bending_parameters[0].sub.width_m
    if unsupported=='not_quarter_frame':
        # Both saved unit endpoints are valid physical rays; their arc is
        # shorter than a quarter and lies outside this explicit native proof.
        trace=replace(trace,end=tuple((a+b)/math.sqrt(2.) for a,b in zip(trace.start,trace.end)))
    else:
        width=origin.radius_km*1000.*.26
    with pytest.raises(UnresolvedFaultMotionSupport):
        _quarter_clamp_envelope(trace,width,origin.radius_km,-1,'sub')
