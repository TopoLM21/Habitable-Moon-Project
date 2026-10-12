"""Bind an audited solid cut-cell thermal endpoint to existing loading preparation."""
import hashlib,json,math
import numpy as np
from .genesis_volume_loading import volume_mechanics_from_dict,prepare_volume_loading,RadialMoments
from .genesis_volume_operator import conical_radial_moments
from .genesis_shell import ShellParameters


def prepare_saved_loading(legacy_raw,packet):
    body={k:v for k,v in packet.items() if k!='sha256'}
    digest=lambda x:hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    if digest(body)!=packet['sha256'] or hashlib.sha256(legacy_raw).hexdigest()!=packet['parent_raw_sha256']:raise ValueError('Thermal/parent seal')
    root=json.loads(legacy_raw);r=root['runtime_state'];parent=volume_mechanics_from_dict(r['mechanics'])
    if digest(r['mechanics'])!=packet['parent_memory_sha256']:raise ValueError('Parent memory mismatch')
    request=packet['thermal_input_for_old_material'];ids=[x['id'] for x in request]
    if len(ids)!=len(set(ids)) or set(ids)!={c.id for c in parent.cohorts}:raise ValueError('Material ID coverage')
    if any(x['liquid_fraction']!=0 for x in request):raise ValueError('Mixed old material not admitted')
    byid={x['id']:x for x in request};g=r['geometry']['cohorts'];ordered=[c['id'] for c in g]
    if ordered!=[c.id for c in parent.cohorts]:raise ValueError('Canonical geometry order')
    for c in parent.cohorts:
        if not math.isclose(c.temperature_k,byid[c.id]['before_temperature_k'],rel_tol=0,abs_tol=1e-8):raise ValueError('Old temperature reset')
    radial=[conical_radial_moments(c['outer_scale'],c['inner_scale']) for c in g]
    moments=RadialMoments(tuple(c['volume_m3'] for c in g),tuple(x[0] for x in radial),tuple(x[1] for x in radial))
    removed=RadialMoments((0.,)*len(g),(0.,)*len(g),(0.,)*len(g))
    damage=np.asarray(r['fault_history']['damage'])[np.array([c['face_index'] for c in g])]
    return prepare_volume_loading(parent,after_ids=ordered,after_temperature_k=[byid[i]['after_temperature_k'] for i in ordered],
      before_temperature_k=[byid[i]['before_temperature_k'] for i in ordered],birth_temperature_k=root['source_packet']['source']['parameters']['thermal']['solidus_k'],
      radial_moments=moments,before_radial_moments=moments,removed_radial_moments=removed,
      before_damage=damage,after_damage=damage,parameters=ShellParameters(**root['source_packet']['source']['parameters']['shell']),after_time_myr=packet['time_myr'])
