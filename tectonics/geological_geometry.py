"""Common diagnostics for rigid motion, rasterization and actual split borders."""
import numpy as np
from .mesh import connected_components

def transport_geometry(mesh,old,new,trace,radius):
    rows=trace['rigid_transport'];raw=sum(max(x['components']-1,0) for x in rows.values())
    post=sum(max(len(connected_components(np.flatnonzero(new.cell_plate==p),mesh.neighbors))-1,0) for p in range(len(new.plates)))
    edges=np.array(mesh.shared_edges)[:,:2].astype(int);source=trace['winner_source'];valid=source>=0
    # A rigid transform preserves all source pair distances exactly. Raster
    # assignment need not; report its actual point error instead of hiding it.
    return dict(raw_footprint_extra_components=raw,post_overlap_extra_components=post,
        max_assignment_error_km=max(x['max_assignment_error_rad'] for x in rows.values())*radius,
        committed_plates=sum(x['committed'] for x in rows.values()),
        inherited_material_owner_mismatches=int(np.count_nonzero(old.cell_plate[source[valid]]!=new.cell_plate[valid])),
        boundary_edges=int(np.count_nonzero(new.cell_plate[edges[:,0]]!=new.cell_plate[edges[:,1]])))

def split_geometry(mesh,before,after,damage_region,radius):
    edges=np.array(mesh.shared_edges)[:,:2].astype(int)
    fresh=(before.cell_plate[edges[:,0]]==before.cell_plate[edges[:,1]])&(after.cell_plate[edges[:,0]]!=after.cell_plate[edges[:,1]])
    border=edges[fresh];region=set(map(int,damage_region));a=mesh.physical_cell_areas_km2(radius)
    return dict(damage_region_cells=len(region),damage_region_area_km2=float(a[list(region)].sum()),
        new_border_edges=border.tolist(),border_cells=sorted(set(border.ravel().tolist())),
        edges_touching_damage=int(sum(int(i) in region or int(j) in region for i,j in border)),
        child_components=[len(connected_components(np.flatnonzero(after.cell_plate==p),mesh.neighbors)) for p in range(len(after.plates))])
