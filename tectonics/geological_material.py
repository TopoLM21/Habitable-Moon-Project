"""Trace native mafic/felsic transfers without inventing a second inventory.

Same reference density and silicate EOS for both effective composition labels.
Native arc production draws hot material from a finite well-mixed interior;
retired material returns its actual H there. This is not elemental petrology.
"""
from copy import deepcopy
import math
import numpy as np
from .experimental_geological_parcels import remap

CONVERSION=3000.*1e9

def coalesce_node(parts):
    """Merge duplicate fragments only inside one already mixed thermal node.

    Source identity, composition, node mass and H are preserved; no mixing
    between bins, birth IDs or different compositions is introduced.
    """
    groups={}
    for p in parts:groups.setdefault((p['id'],p.get('material_phase')),[]).append(p)
    return [dict(rows[0],mass_kg=math.fsum(p['mass_kg'] for p in rows),energy_j=math.fsum(p['energy_j'] for p in rows),
        **({'mass_coordinate_point_assignment':True} if any(p.get('mass_coordinate_point_assignment') for p in rows) else {})) for rows in groups.values()]

def transport_cycle(parcel,mass,energy,old,lith,final,trace,cycle,time):
    nodes=parcel['nodes'];n=len(nodes);before=parcel['node_parts']
    profiles=[[[],[]] for _ in range(n)]
    generated=[];retired=[];processes={};donor_h=float(energy[-2]/mass[-2])
    oldM=float(mass[-2]);oldH=float(energy[-2]);parcel['generation']+=1;generation=parcel['generation']
    original=[]
    for f,row in enumerate(nodes):
        parts=[dict(p) for i in row for p in before[str(i)]]
        for p in parts:p.setdefault('material_phase','mafic')
        groups=[[p for p in parts if p['material_phase']==phase] for phase in ('mafic','felsic')]
        for phase,vol in enumerate((old.oceanic_volume_km3[f],old.continental_volume_km3[f])):
            if not math.isclose(math.fsum(p['mass_kg'] for p in groups[phase]),vol*CONVERSION,rel_tol=3e-13,abs_tol=1.):
                raise ValueError('Composition provenance differs from native inventory')
        original.append(groups)
    winner=trace['winner_source'];survived=set(map(int,winner[winner>=0]))
    for f,src in enumerate(winner):
        if src>=0:profiles[f][0]=list(original[src][0])
    for src in range(n):
        if src not in survived:retired.extend(original[src][0])
    processes['oceanic_subduction']=math.fsum(p['mass_kg'] for p in retired)
    for pid,row in trace['rigid_transport'].items():
        for src,dst in zip(row['source'],row['target']):profiles[int(dst)][1].extend(original[int(src)][1])

    def remove(f,k,amount):
        parts=profiles[f][k];total=math.fsum(p['mass_kg'] for p in parts)
        if amount>total*(1+3e-13)+1.:raise ValueError('Material transfer exceeds donor')
        if total==0:return []
        fraction=min(amount/total,1.);out=[];kept=[]
        for p in parts:
            out.append(dict(p,mass_kg=p['mass_kg']*fraction,energy_j=p['energy_j']*fraction))
            if fraction<1:kept.append(dict(p,mass_kg=p['mass_kg']*(1-fraction),energy_j=p['energy_j']*(1-fraction)))
        profiles[f][k]=kept;return out

    def move(transfers):
        for src,dst,volume in transfers:profiles[dst][1].extend(remove(src,1,volume*CONVERSION))

    def adjust(name,basalt,continental):
        for k,volumes in enumerate((basalt,continental)):
            for f,vol in enumerate(volumes):
                wanted=float(vol)*CONVERSION;present=math.fsum(p['mass_kg'] for p in profiles[f][k]);delta=wanted-present
                if abs(delta)<=64*np.finfo(float).eps*max(wanted,present,1.):continue
                phase=('mafic','felsic')[k]
                if delta>0:
                    p=dict(id=f'{name}-{generation}-{f}-{phase}',origin_face=f,birth_cell_id=None,
                        composition='native_effective_arc_felsic' if k else 'native_reference_mafic',material_phase=phase,
                        origin_reservoir='well_mixed_interior',process=name,time_myr=time,mass_kg=delta,energy_j=delta*donor_h)
                    profiles[f][k].append(p);generated.append(p)
                else:retired.extend(remove(f,k,-delta))
                processes[name]=processes.get(name,0.)+delta

    move(trace.get('continental_spreading',[]))
    present_cont=np.array([math.fsum(p['mass_kg'] for p in x[1])/CONVERSION for x in profiles])
    if 'basalt_before_displacement' in trace:
        adjust('ridge_creation',trace['basalt_before_displacement'],present_cont)
        adjust('continental_displacement',trace['basalt_after_displacement'],present_cont)
    # A rift explicitly retires its old basalt before inserting new basalt,
    # even where old/new volumes happen to match.
    for f in trace.get('rift_replaced_faces',[]):retired.extend(remove(f,0,math.fsum(p['mass_kg'] for p in profiles[f][0])))
    adjust('lithosphere_birth_and_rift',lith.oceanic_volume_km3,lith.continental_volume_km3)
    for name,b,c in cycle['stages']:
        if name=='collapse':move(cycle.get('collapse_transfers',[]))
        adjust(name,b,c)
    createdM=math.fsum(p['mass_kg'] for p in generated);createdH=math.fsum(p['energy_j'] for p in generated)
    retiredM=math.fsum(p['mass_kg'] for p in retired);retiredH=math.fsum(p['energy_j'] for p in retired)
    if createdM>=oldM:raise ValueError('Interior donor exhausted')
    mass[-2]=oldM-createdM+retiredM;energy[-2]=oldH-createdH+retiredH
    for f,row in enumerate(nodes):
        parts=profiles[f][0]+profiles[f][1]
        mapped=remap(parts,parcel['weights'][f])
        for i,pieces in zip(row,mapped):
            pieces=coalesce_node(pieces)
            parcel['node_parts'][str(i)]=pieces;mass[i]=math.fsum(p['mass_kg'] for p in pieces);energy[i]=math.fsum(p['energy_j'] for p in pieces)
    # Archive provenance, not a duplicate active slab mass reservoir.
    parcel['recycled_history'].append(dict(time_myr=time,parts=retired,destination='well_mixed_deep_silicate',process_mass_kg=processes))
    return dict(created_mass=createdM,created_H=createdH,recycled_mass=retiredM,recycled_H=retiredH,donor_h=donor_h)
