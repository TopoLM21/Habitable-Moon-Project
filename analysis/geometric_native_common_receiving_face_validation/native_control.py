"""New complete curved-face endpoint controls; old physical mains are not run."""
from fractions import Fraction as F
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import time

from tectonics import geometric_native_common_receiving_face as api
from tectonics.geometric_native_common_reference import bind_native_common_reference
from tectonics.geometric_native_common_contact_cuts import bind_common_contact_cuts
from analysis.geometric_native_common_cold_continuum_energy_validation.native_control import cold_carrier

ROOT=Path(__file__).resolve().parents[2]
PROVENANCE=('NEW C6/HOT C1 virgin permanently-unbonded frictionless bare interface; '
    'kn=1e7 Pa/m from existing HOT parameter category; fixed genuine common face measure, '
    'restgap0 at coincident common source reference; actual prepared gap retained; '
    'no inherited -0.01 preload/history or additional elastic columns')


@lru_cache(maxsize=1)
def context():
    path=ROOT/'analysis/geometric_native_common_hot_mean_validation/release_validation_02/native_control/source_packet.json'
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!='0f8a0ee8ae082e3434b136bab7515944354eaf059db95965d0e87259f6791035':
        raise ValueError('The complete authentic HOT source changed')
    hot=json.loads(raw);reference=bind_native_common_reference()
    cuts=bind_common_contact_cuts(reference)
    cut=next(c for c in cuts.cuts if c.cut_id=='retained-contact:octant:5:hot-curved-extra')
    declaration=api.declare_native_receiving_face(reference,cut,provenance=PROVENANCE)
    return reference,cut,declaration,cold_carrier(),hot


def run(output):
    started=time.perf_counter();ref,cut,law,cold,hot=context();rows=[];integrals=[]
    cubes=api.I(ref.radius_m**3-3*ref.radius_m**2*F(1000))
    for query in hot['evaluation_queries']:
        for label,lam in (('start',api.I(0)),('interior',api.I(F(1,3))),('end',api.I(1))):
            point=api.evaluate_native_receiving_face(ref,cut,law,cold,hot,query['label'],lam,cubes,
                cold_coordinates=(F(0),)*56)
            rows.append(dict(label=label,response=api.native_receiving_face_point_to_dict(point)))
        for cold_label,coordinates in (('prepared_zero',(F(0),)*56),('saved_full',cold.bound_coordinate_increments)):
            operator=api.integrate_native_receiving_face(ref,cut,law,cold,hot,query['label'],cold_coordinates=coordinates)
            integrals.append(dict(cold_label=cold_label,hot_query_label=query['label'],operator=operator))
    raw=rows[-1]['response']
    restored=api.native_receiving_face_point_from_dict(raw,reference=ref,cut=cut,declaration=law,cold_system=cold,hot_packet=hot)
    assert api.encoded(api.native_receiving_face_point_to_dict(restored))==api.encoded(raw)
    rawlaw=api.native_receiving_face_declaration_to_dict(law)
    assert api.native_receiving_face_declaration_to_dict(api.native_receiving_face_declaration_from_dict(rawlaw,reference=ref,cut=cut))==rawlaw
    packet=dict(format='native-complete-curved-receiving-face-control-1',scope=api.SCOPE,
        declaration=rawlaw,point_queries=rows,whole_face_operators=integrals,
        complete_material_operator_restart_verified=True,point_query_count=len(rows),whole_face_operator_count=len(integrals),
        independent_temporal_path_work_certified=False,physical_source_work_handover_certified=False,
        preparation_or_activation_work_paid=False,physical_material_admission_authorized=False,
        new_mixed_equilibrium_certified=False,clocks_advanced=False,accepted_old_native_intervals=1,
        accepted_new_native_intervals=0,original_point_1_completed=False,original_point_1_remaining_work_blocks=1,
        remaining_physical_conditions=hot['remaining_physical_conditions'],seconds=time.perf_counter()-started)
    packet['packet_sha256']=api._sha(packet)
    with Path(output).open('xb') as f:f.write(api.encoded(packet)+b'\n')
    print(json.dumps(dict(status='PASS_NEW_COMPLETE_CURVED_RECEIVING_FACE_CONTROL',
        point_query_count=len(rows),whole_face_operator_count=len(integrals),seconds=packet['seconds'],
        packet_sha256=packet['packet_sha256']),sort_keys=True))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    run(parser.parse_args().output)
