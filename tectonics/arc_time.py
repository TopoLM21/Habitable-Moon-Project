"""Localize source gates on the existing frozen-boundary slab trajectory.

This changes time integration, not slab growth or arc thresholds. Sources use
midpoint states on intervals split at exact age/depth gates. A newly detached
slab requires its own event localization and is rejected rather than guessed.
"""
from copy import deepcopy
import math
import numpy as np
from scipy.optimize import brentq
from .subduction_memory import SlabZone


def arc_time_slices(before,after,dt,slab,arc):
    if before is None or after is None or slab is None or not math.isfinite(dt) or dt<=0:
        raise ValueError('Localized source needs both slab states and a positive interval')
    if not 0<slab.initial_dip_deg<=slab.mature_dip_deg<90 or slab.dip_maturation_myr<=0:
        raise ValueError('Localized source requires the existing monotone slab-dip law')
    starts={};cuts={0.,float(dt)}
    for key,z in after.zones.items():
        old=before.zones.get(key,SlabZone(*key));starts[key]=old
        if z.broken_off and not old.broken_off:
            raise ValueError('New slab breakoff needs localized detachment before source integration')
        if z.broken_off:
            t=arc.post_breakoff_duration_myr-old.post_breakoff_age_myr
            if 0<t<dt:cuts.add(float(t))
        elif z.active:
            if abs(z.active_age_myr-old.active_age_myr-dt)>64*math.ulp(max(z.active_age_myr,dt,1.)):
                raise ValueError('Saved slab age does not span the requested source interval')
            def depth(t):return _trajectory(old,z,t,slab)[3]
            if not np.isclose(depth(dt),z.slab_depth_km,rtol=2e-13,atol=1e-11):
                raise ValueError('Slab endpoint disagrees with its frozen growth law')
            age=max(0.,arc.min_active_age_myr-old.active_age_myr)
            if 0<age<dt:cuts.add(float(age))
            if depth(0)<arc.min_active_slab_depth_km<depth(dt):
                t=brentq(lambda t:depth(t)-arc.min_active_slab_depth_km,0.,dt,xtol=2e-13,rtol=8*np.finfo(float).eps)
                if 0<t<dt:cuts.add(float(t))
    cuts=sorted(cuts);result=[]
    for lo,hi in zip(cuts[:-1],cuts[1:]):
        t=(lo+hi)/2;sample=deepcopy(after)
        for key,z in sample.zones.items():
            old=starts[key]
            if z.broken_off:z.post_breakoff_age_myr=old.post_breakoff_age_myr+t
            elif z.active:
                z.active_age_myr,z.slab_length_km,z.dip_deg,z.slab_depth_km=_trajectory(old,z,t,slab)
        result.append((lo,hi,sample))
    return result


def _trajectory(old,end,t,params):
    age=old.active_age_myr+t
    length=min(params.slab_length_cap_km,old.slab_length_km+max(end.convergence_rate_km_per_myr,0.)*params.slab_length_growth_efficiency*t)
    dip=params.initial_dip_deg+(params.mature_dip_deg-params.initial_dip_deg)*(-math.expm1(-age/params.dip_maturation_myr))
    depth=min(params.slab_depth_cap_km,length*math.sin(math.radians(dip)))
    return age,length,dip,depth
