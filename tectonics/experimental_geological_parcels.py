"""Conservative ragged solid profiles on the inherited fixed thermal graph.

Profiles move by material_source_index; receiving node bins remain Eulerian.
Layer fragments preserve source identity and undifferentiated composition.
Mixing inside a receiving thermal node is explicit, not geometric remeshing.
"""
from copy import deepcopy
import math
import numpy as np

COMPOSITION='undifferentiated_silicate_unknown_elemental_fractions'


def layout(model):
    c,s=model.config,model.state;nodes=[list(map(int,row)) for row in c['solid_nodes']]
    catalog={x['id']:x for x in c.get('phase_cells',[])};seen=set();used={i for row in nodes for i in row}
    for x in sorted(s.get('born_material',[]),key=lambda x:(x['face'],catalog.get(x['cell_id'],{}).get('q_lo',0))):
        if x['cell_id'] in seen or x['cell_id'] not in catalog:raise ValueError('PARCEL_BIRTH_ID')
        row=catalog[x['cell_id']];i=x['node'];f=x['face']
        if (i in used or not 0<=i<len(c['mass'])-2 or not 0<=f<len(nodes)
            or any(x[k]!=row[k] for k in ('node','face','mass_kg','volume_m3'))
            or x['mass_kg']!=c['mass'][i]):raise ValueError('PARCEL_BIRTH_INVENTORY')
        if model.temperature(s['energy'])[i]>c['solidus']:raise ValueError('PARCEL_BIRTH_NOT_SOLID')
        nodes[f].append(i);used.add(i);seen.add(x['cell_id'])
    flat=[i for row in nodes for i in row]
    if len(set(flat))!=len(flat) or any(i<0 or i>=len(c['mass'])-2 for i in flat):raise ValueError('PARCEL_OVERLAPPING_NODES')
    return nodes


def initialize(model,nodes):
    c,s=model.config,model.state;birth={x['node']:x for x in s.get('born_material',[])};parts={}
    for f,row in enumerate(nodes):
        for i in row:
            parts[str(i)]=[dict(id='source-node-'+str(i),origin_face=f,birth_cell_id=birth.get(i,{}).get('cell_id'),
                composition=COMPOSITION,mass_kg=c['mass'][i],energy_j=s['energy'][i])]
    return dict(nodes=nodes,weights=[(np.array(c['mass'])[row]/sum(np.array(c['mass'])[row])).tolist() for row in nodes],
        node_parts=parts,recycled_history=[],generation=0,
        approximation='cumulative mass-coordinate remap into fixed receiving thermal bins; fragments mix thermally within bins')


def remap(parts,weights):
    """Split identities by overlapping cumulative mass intervals, preserving m/H."""
    total=math.fsum(x['mass_kg'] for x in parts);w=np.array(weights,dtype=float)
    if total<=0 or np.any(w<=0) or not np.isfinite(w).all():raise ValueError('PARCEL_POSITIVE_MASS')
    ends=np.r_[0.,np.cumsum(w/w.sum())*total];ends[-1]=total;out=[[] for _ in w];start=0.
    for index,p in enumerate(parts):
        stop=total if index==len(parts)-1 else start+p['mass_kg']
        overlaps=np.maximum(0,np.minimum(stop,ends[1:])-np.maximum(start,ends[:-1]));ids=np.flatnonzero(overlaps>0)
        if not len(ids):raise ValueError('PARCEL_UNRESOLVED_MASS_INTERVAL')
        fractions=overlaps[ids]/overlaps[ids].sum();usedM=[];usedH=[]
        for j,fraction in zip(ids,fractions):
            last=j==ids[-1];mass=p['mass_kg']-math.fsum(usedM) if last else p['mass_kg']*fraction
            energy=p['energy_j']-math.fsum(usedH) if last else p['energy_j']*fraction
            out[j].append(dict(p,mass_kg=mass,energy_j=energy));usedM.append(mass);usedH.append(energy)
        start=stop
    for identity in {p['id'] for p in parts}:
        before=[p for p in parts if p['id']==identity];after=[p for row in out for p in row if p['id']==identity]
        for key in ('mass_kg','energy_j'):
            if not math.isclose(math.fsum(p[key] for p in before),math.fsum(p[key] for p in after),rel_tol=2e-14):
                raise ValueError('PARCEL_IDENTITY_CONSERVATION')
    return out


def transport(parcel,mass,energy,source_index,generated_face_mass,time):
    old=deepcopy(parcel);nodes=old['nodes'];oldM=mass.copy();oldH=energy.copy();source_index=np.asarray(source_index)
    sources=source_index[source_index>=0]
    if len(np.unique(sources))!=len(sources) or np.any(sources>=len(nodes)):raise ValueError('Thermal parcel duplication')
    # Validate every active node before any mapping. Provenance is not a second inventory.
    for row in nodes:
        for i in row:
            p=old['node_parts'][str(i)]
            if not math.isclose(math.fsum(x['mass_kg'] for x in p),oldM[i],rel_tol=2e-14):raise ValueError('PARCEL_NODE_MASS')
            if not math.isclose(math.fsum(x['energy_j'] for x in p),oldH[i],rel_tol=2e-14):raise ValueError('PARCEL_NODE_ENTHALPY')
    lost=[i for f,row in enumerate(nodes) if f not in set(sources) for i in row]
    recycledM=math.fsum(oldM[lost]);recycledH=math.fsum(oldH[lost]);createdM=math.fsum(generated_face_mass[source_index<0]);donor_h=oldH[-2]/oldM[-2]
    if createdM>=oldM[-2]:raise ValueError('Deep donor exhausted before recycling')
    recycled_parts=[p for i in lost for p in old['node_parts'][str(i)]]
    parcel['generation']+=1
    for f,row in enumerate(nodes):
        src=source_index[f]
        if src>=0:parts=[p for i in nodes[src] for p in old['node_parts'][str(i)]]
        else:parts=[dict(id=f"native-{parcel['generation']}-{f}",origin_face=f,birth_cell_id=None,
            composition=COMPOSITION,mass_kg=float(generated_face_mass[f]),energy_j=float(generated_face_mass[f]*donor_h))]
        mapped=[deepcopy(old['node_parts'][str(i)]) for i in row] if src==f else remap(parts,old['weights'][f])
        for i,segments in zip(row,mapped):
            parcel['node_parts'][str(i)]=segments;mass[i]=math.fsum(p['mass_kg'] for p in segments);energy[i]=math.fsum(p['energy_j'] for p in segments)
    mass[-2]=oldM[-2]-createdM+recycledM;energy[-2]=oldH[-2]-createdM*donor_h+recycledH
    parcel['recycled_history'].append(dict(time_myr=time,parts=recycled_parts,destination='deep aggregate silicate reservoir; source identities archived, not additional active inventory'))
    return dict(recycled_mass=recycledM,recycled_H=recycledH,created_mass=createdM,created_H=createdM*donor_h,donor_h=donor_h,
        moved_profiles=int(sum((source_index>=0)&(source_index!=np.arange(len(nodes))))),recycled_parts=len(recycled_parts))


def sync_thermal(parcel,mass,energy):
    """Homogenize fragment H in each Eulerian bin after the active thermal solve."""
    for row in parcel['nodes']:
        for i in row:
            pieces=parcel['node_parts'][str(i)];used=[]
            for k,p in enumerate(pieces):
                p['energy_j']=float(energy[i]-math.fsum(used)) if k==len(pieces)-1 else float(energy[i]*p['mass_kg']/mass[i]);used.append(p['energy_j'])
