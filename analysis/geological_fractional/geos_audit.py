"""Independent GEOS overlay of saved rays in per-cell gnomonic charts."""
import json,math,sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import shapely
from shapely.geometry import Polygon
from shapely.ops import unary_union
from tectonics.unified_geological_genesis import UnifiedGeologicalGenesis as U
from tectonics.spherical_polygons import polygon_area
from .audit import polygon_components

def run(saved,initial,output):
    e=U.load(saved);start=U.load(initial);r=e.record['radius_km'];groups=defaultdict(list)
    for row in e.record['fractional_surface']:groups[row['cell']].append(row)
    overlaps=[];gaps=[];outside=[];invalid=[];raw_invalid=[];signed=[]
    for cell,rows in groups.items():
        c=e.mesh.centroids[cell];u=np.cross(c,np.eye(3)[np.argmin(abs(c))]);u/=np.linalg.norm(u);v=np.cross(c,u)
        def project(vertices):
            x=np.asarray(vertices);return np.c_[x@u,x@v]/(x@c)[:,None]
        def area(geometry):
            if geometry.is_empty:return 0.
            if geometry.geom_type!='Polygon':return math.fsum(area(g) for g in getattr(geometry,'geoms',[]))
            def ring(coords):
                xy=np.array(coords)[:-1];rays=c+xy[:,0,None]*u+xy[:,1,None]*v
                rays/=np.linalg.norm(rays,axis=1)[:,None];return polygon_area(rays)*r*r
            return ring(geometry.exterior.coords)-math.fsum(ring(h.coords) for h in geometry.interiors)
        domain=shapely.set_precision(Polygon(project(e.mesh.vertices[e.mesh.faces[cell]])),1e-12)
        raw=[Polygon(project(row['vertices'])) for row in rows]
        raw_invalid.extend(dict(cell=cell,patch=rows[i]['id']) for i,p in enumerate(raw) if not p.is_valid)
        polys=[shapely.set_precision(p,1e-12) for p in raw]
        for i,p in enumerate(polys):
            if not p.is_valid:invalid.append(dict(cell=cell,patch=rows[i]['id']));continue
            outside.append(area(p.difference(domain)))
            for q in polys[:i]:
                if q.is_valid:overlaps.append(area(p.intersection(q)))
        if any(not p.is_valid for p in polys):continue
        union=unary_union(polys);gaps.append(area(domain.difference(union)))
        signed.append(math.fsum(polygon_area(row['vertices'])*r*r for row in rows)-e.mesh.physical_cell_areas_km2(r)[cell])
    def inventory(engine):
        inv=defaultdict(list)
        groups=list(engine.record['material_parcels']['node_parts'].values())+[x['parts'] for x in engine.record['material_parcels']['recycled_history']]
        for group in groups:
            for p in group:inv[p['id']].append(p['mass_kg'])
        return {k:math.fsum(v) for k,v in inv.items()}
    before=inventory(start);after=inventory(e)
    result=dict(audit_grid_size_rad=1e-12,engine='GEOS saved-coordinate readback '+shapely.geos_version_string,shapely_version=shapely.__version__,invalid_polygons=invalid,raw_invalid_polygons=raw_invalid,
        pairwise_intrusion_sum_km2=math.fsum(overlaps),max_pairwise_intrusion_km2=max(overlaps,default=0),
        gap_sum_km2=math.fsum(gaps),outside_cell_sum_km2=math.fsum(outside),max_cell_area_error_km2=max(map(abs,signed),default=0),
        initial_identity_count=len(before),max_identity_mass_relative_error=max(abs(after.get(k,0)-m)/m for k,m in before.items() if m),
        polygon_edge_components_by_owner=polygon_components(e.record['fractional_surface'],e.mesh),polygon_edge_tolerance_rad=1e-9)
    Path(output).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));return result
if __name__=='__main__':run(*sys.argv[1:])
