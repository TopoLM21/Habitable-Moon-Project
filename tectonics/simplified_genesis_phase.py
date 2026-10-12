"""Conservative phase accounting on fixed thermal supports.

Uses the existing linear enthalpy phase law. Whole-cell load-bearing admission
at the solidus is distinct from progressive crystal mass production in mush.
"""
import math
import numpy as np
from .genesis_implicit_enthalpy import invert,phase
from .genesis_cutcell_initialization import partition_cell


def solid_inventory(rows,solidus,liquidus):
    """Partition invariant for subdivisions preserving each material's h."""
    total=[]
    for c in rows:
        T=invert(c['energy_j']/c['mass_kg'],c['cp'],c['latent'],solidus,liquidus,0.)
        total.append(c['mass_kg']*(1-float(phase(T,solidus,liquidus))))
    return math.fsum(total)


def subdivide_uniform_cell(cell,fraction):
    """Existing exact mass/enthalpy partition, no physical birth or new heat."""
    if not 0<fraction<1:raise ValueError('SPLIT_FRACTION')
    lo=cell['q_lo'];hi=cell['q_hi'];mid=lo+fraction*(hi-lo);h=cell['energy_j']/cell['mass_kg']
    anchors=[dict(id=cell['id'],q_lo=a,q_hi=b,specific_h=h) for a,b in [(lo,mid),(mid,hi)]]
    rows=partition_cell(face=cell['face'],layer=cell['layer'],q_lo=lo,q_hi=hi,mass_kg=cell['mass_kg'],energy_j=cell['energy_j'],anchors=anchors,lower_h=h,upper_h=h)
    return [dict(c,id=cell['id'],cp=cell['cp'],latent=cell['latent']) for c in rows]


def promote(config,before,after,T0,T1,energy,inertia,omega):
    """Endpoint admission, zero stress, explicit rest-melt angular mass loading.

    No total mass/enthalpy is added: thermal cells already own their inventories.
    Only the allocation to load-bearing mechanical mass changes.
    """
    cells=config.get('phase_cells',[]);entries=after.get('born_material',[]);admitted={x['cell_id'] for x in entries}
    ts=config['solidus'];tl=config['liquidus'];formed=[];melted=[];promotions=[]
    volume=np.zeros(len(inertia));newI=np.array(inertia,copy=True)
    for c in sorted(cells,key=lambda x:(x['face'],x['q_lo'])):
        i=c['node'];f=c['face'];m=c['mass_kg'];oldsolid=1-float(phase(T0[i],ts,tl));solid=1-float(phase(T1[i],ts,tl))
        change=m*(solid-oldsolid);formed.append(max(change,0.));melted.append(max(-change,0.))
        if c['id'] in admitted:
            if T1[i]>ts:raise ValueError('LOAD_BEARING_REMELT_UNSUPPORTED')
            continue
        if T1[i]>ts:continue
        if before['consumed'][f] or after['consumed'][f]:raise ValueError('SOLIDIFICATION_ACROSS_FRACTURE_UNSUPPORTED')
        if any(x['face']==f and x['q_hi']<=c['q_lo'] and x['id'] not in admitted for x in cells):raise ValueError('DETACHED_SOLID_BODY_UNSUPPORTED')
        entry=dict(cell_id=c['id'],node=i,face=f,mass_kg=m,volume_m3=c['volume_m3'],inertia=c['inertia'],memory=[0.]*6,birth_time_myr=after['time_myr'],birth_enthalpy_j=float(energy[i]),specific_enthalpy_j_kg=float(energy[i]/m))
        entries.append(entry);admitted.add(c['id']);promotions.append(entry);newI[f]+=np.array(c['inertia']);volume[f]+=c['volume_m3']
    if promotions and len(after['events'])!=len(before['events']):raise ValueError('SIMULTANEOUS_BIRTH_SPLIT_UNSUPPORTED')
    capture=0.;momentum_error=0.;owner=np.array(after['owner']);newomega=np.array(omega,copy=True)
    for k in range(len(omega)):
        ids=owner==k;old=inertia[ids].sum(axis=0);new=newI[ids].sum(axis=0);w=np.array(omega[k]);L=old@w
        if not any(owner[x['face']]==k for x in promotions):continue
        v=np.linalg.solve(new,L)
        delta=float(.5*w@old@w-.5*v@new@v)
        if delta < -1e-12*max(abs(.5*w@old@w),1.):raise ValueError('MASS_CAPTURE_ENERGY')
        delta=max(delta,0.);capture+=delta;newomega[k]=v
        momentum_error=max(momentum_error,float(np.linalg.norm(new@v-L)/max(np.linalg.norm(L),1.)))
        group=[x for x in promotions if owner[x['face']]==k];mass=math.fsum(x['mass_kg'] for x in group)
        for x in group:energy[x['node']]+=delta*x['mass_kg']/mass
    after['born_material']=entries
    return newomega,dict(new_phase_solid_kg=math.fsum(formed),mush_remelted_kg=math.fsum(melted),new_load_bearing_kg=math.fsum(x['mass_kg'] for x in promotions),new_load_bearing_cells=len(promotions),transferred_enthalpy_j=math.fsum(x['birth_enthalpy_j'] for x in promotions),added_latent_heat_j=0.,capture_heat_j=capture,capture_angular_momentum_relative=momentum_error),newI
