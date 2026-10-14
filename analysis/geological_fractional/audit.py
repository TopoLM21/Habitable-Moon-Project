"""Independent readback of saved polygons, not of native owner assignments."""
import math
from collections import defaultdict
import numpy as np
from tectonics.spherical_polygons import polygon_area, intersect_convex
from tectonics.mesh import connected_components

def shared_edge(a,b,tol=1e-9):
    """Positive shared arc, at stated angular tolerance, not centroid adjacency."""
    def angle(x,y):return float(np.arctan2(np.linalg.norm(np.cross(x,y)),np.dot(x,y)))
    def onarc(x,u,v):return abs(angle(u,x)+angle(x,v)-angle(u,v))<tol
    for u,v in zip(a,np.roll(a,-1,axis=0)):
        n=np.cross(u,v);norm=np.linalg.norm(n)
        if norm<1e-14:continue
        n/=norm
        for w,z in zip(b,np.roll(b,-1,axis=0)):
            m=np.cross(w,z);norm=np.linalg.norm(m)
            if norm<1e-14:continue
            m/=norm
            if np.linalg.norm(np.cross(n,m))>tol:continue
            points=[x for x in (u,v,w,z) if onarc(x,u,v) and onarc(x,w,z)]
            if any(angle(x,y)>tol for i,x in enumerate(points) for y in points[i+1:]):return True
    return False


def polygon_components(rows,mesh):
    """Build owner graph using polygon edges within a cell or neighboring cells."""
    groups=defaultdict(list)
    for i,row in enumerate(rows):groups[row['cell']].append(i)
    parent=list(range(len(rows)))
    def root(i):
        while parent[i]!=i:i=parent[i]
        return i
    polys=[np.asarray(row['vertices']) for row in rows]
    for i,row in enumerate(rows):
        for cell in [row['cell'],*mesh.neighbors[row['cell']]]:
            for j in groups[int(cell)]:
                if j<=i or rows[j]['owner']!=row['owner']:continue
                if root(i)!=root(j) and shared_edge(polys[i],polys[j]):parent[root(j)]=root(i)
    result={}
    for owner in {x['owner'] for x in rows}:
        areas=defaultdict(list)
        for i,row in enumerate(rows):
            if row['owner']==owner:areas[root(i)].append(polygon_area(polys[i]))
        result[str(owner)]=dict(count=len(areas),component_areas_steradians=sorted([math.fsum(v) for v in areas.values()],reverse=True))
    return result


def readback(engine):
    r=engine.record;mesh=engine.mesh;radius=r['radius_km'];rows=r['fractional_surface']
    groups=defaultdict(list)
    for row in rows:groups[row['cell']].append(row)
    overlap=[];areas=[];mixed=0
    for cell,group in groups.items():
        areas.append(abs(math.fsum(polygon_area(np.asarray(x['vertices']))*radius**2 for x in group)-mesh.physical_cell_areas_km2(radius)[cell]))
        mixed+=len({x['owner'] for x in group})>1
        for i,a in enumerate(group):
            for b in group[i+1:]:
                poly=intersect_convex(np.asarray(a['vertices']),np.asarray(b['vertices']))
                overlap.append(polygon_area(poly)*radius**2 if len(poly) else 0.)
    # Necessary coarse connectivity check; explicitly not a polygon-edge proof.
    occupancy=defaultdict(set)
    for row in rows:occupancy[row['owner']].add(row['cell'])
    components={str(owner):len(connected_components(np.array(sorted(cells)),mesh.neighbors)) for owner,cells in occupancy.items()}
    current={}
    for parts in r['material_parcels']['node_parts'].values():
        for part in parts:current.setdefault(part['id'],[]).append(part['mass_kg'])
    for event in r['material_parcels']['recycled_history']:
        for part in event['parts']:current.setdefault(part['id'],[]).append(part['mass_kg'])
    return dict(max_cell_area_error_km2=max(areas),pairwise_intrusion_sum_km2=math.fsum(overlap),
        max_pairwise_intrusion_km2=max(overlap,default=0),mixed_owner_cells=mixed,
        occupied_cell_components_by_owner=components,polygon_edge_components_by_owner=polygon_components(rows,mesh),polygon_edge_tolerance_rad=1e-9,
        identity_mass_kg={k:math.fsum(v) for k,v in current.items()})

if __name__=='__main__':
    import sys,json
    from pathlib import Path
    from tectonics.unified_geological_genesis import UnifiedGeologicalGenesis as U
    e=U.load(sys.argv[1]);result=readback(e)
    initial=U.load(sys.argv[2]);before={}
    for parts in initial.record['material_parcels']['node_parts'].values():
        for p in parts:before.setdefault(p['id'],[]).append(p['mass_kg'])
    result['initial_identity_count']=len(before)
    result['max_original_identity_mass_relative_error']=max(abs(result['identity_mass_kg'].get(k,0)-math.fsum(v))/math.fsum(v) for k,v in before.items())
    del result['identity_mass_kg']
    Path(sys.argv[3]).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
