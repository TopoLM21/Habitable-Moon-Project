"""Local newborn attachment and audits; never relabel surviving material."""
import heapq
import numpy as np
from .mesh import connected_components
from .dynamics import angular_velocity_vectors


def adjacent_gap_owners(mesh, owner, gap):
    """Fill only newborn gaps along adjacent paths from surviving domains.

    Every repaired cell has an already attached neighbor. No path crosses a survivor
    of another plate; existing material owners are immutable. This is a raster
    attachment rule, not a model of deformation or subduction.
    """
    out = np.array(owner, dtype=np.int32, copy=True)
    # Preserve the historical nearest assignment whenever its newborn domains
    # already touch surviving material of that owner. Repair orphan newborns
    # only; globally rerouting valid gaps changes contact trajectories needlessly.
    orphan = np.zeros(mesh.cell_count,dtype=bool)
    for pid in np.unique(out):
        for component in connected_components(np.flatnonzero(out == pid),mesh.neighbors):
            if np.all(gap[component]):
                orphan[component] = True
    if not np.any(orphan):
        return out
    out[orphan] = -1
    queue = []
    for cell in np.flatnonzero(~orphan):
        for nb in mesh.neighbors[cell]:
            if orphan[nb]:
                distance = float(np.arccos(np.clip(mesh.centroids[cell] @ mesh.centroids[nb], -1, 1)))
                heapq.heappush(queue, (distance, int(out[cell]), int(nb)))
    while queue:
        distance, pid, cell = heapq.heappop(queue)
        if out[cell] >= 0:
            continue
        out[cell] = pid
        for nb in mesh.neighbors[cell]:
            if out[nb] < 0:
                step = float(np.arccos(np.clip(mesh.centroids[cell] @ mesh.centroids[nb], -1, 1)))
                heapq.heappush(queue, (distance + step, pid, int(nb)))
    if np.any(out < 0):
        raise ValueError('Newborn attachment has no surviving adjacent domain')
    return out


def contact_inventory(mesh, old, new, trace, radius):
    """Audit raster ownership separately from conservative material volume."""
    areas = mesh.physical_cell_areas_km2(radius)
    sources = np.concatenate([row['source'] for row in trace['rigid_transport'].values()])
    winner = trace['winner_source']
    retained = winner[winner >= 0]
    gap = winner < 0
    orphan = []
    for pid in np.unique(new.cell_plate):
        for component in connected_components(np.flatnonzero(new.cell_plate == pid), mesh.neighbors):
            if np.all(gap[component]):
                orphan.extend(component)
    duplicate_sources = len(sources) - len(np.unique(sources))
    duplicate_winners = len(retained) - len(np.unique(retained))
    absent_sources = len(set(range(mesh.cell_count)) - set(map(int, sources)))
    mismatches = np.count_nonzero(old.cell_plate[retained] != new.cell_plate[~gap])
    if duplicate_sources or duplicate_winners or absent_sources or mismatches:
        raise ValueError('Invalid contact material provenance or ownership')
    return dict(source_duplicates=duplicate_sources,missing_sources=absent_sources,
        winner_duplicates=duplicate_winners,inherited_owner_mismatches=int(mismatches),
        orphan_newborn_cells=list(map(int,orphan)),newborn_area_km2=float(areas[gap].sum()),
        surface_partition_area_km2=float(areas.sum()),
        # Rigid parcel count is conservative, but mesh cells are unequal-area.
        # This explicit raster footprint error is NOT a mass error or hidden sink.
        retained_raster_area_minus_source_area_km2=float(areas[~gap].sum()-areas[retained].sum()))


def repair_audit(mesh, before, after, repair, radius, threshold):
    """Describe actual detached areas and velocity changes without threshold tuning."""
    areas = mesh.physical_cell_areas_km2(radius)
    old_w, new_w = angular_velocity_vectors(before), angular_velocity_vectors(after)
    inherited = np.asarray(repair.parent_by_plate)[after.cell_plate]
    changed = inherited != before.cell_plate
    components = []
    for pid in np.unique(before.cell_plate):
        groups = sorted(connected_components(np.flatnonzero(before.cell_plate == pid), mesh.neighbors),
                        key=lambda x: (-float(areas[x].sum()), min(x)))
        for cells in groups[1:]:
            area = float(areas[cells].sum())
            components.append(dict(parent=int(pid),cells=list(map(int,cells)),area_km2=area,
                area_over_threshold=area/threshold if threshold else None,
                promoted=bool(np.all(inherited[cells] == pid))))
    def momentum(owner,w):
        v=w[owner];n=mesh.centroids
        return np.sum(areas[:,None]*(v-n*np.sum(n*v,axis=1)[:,None]),axis=0)
    L0=momentum(before.cell_plate,old_w);L1=momentum(after.cell_plate,new_w)
    pure_errors=[]
    for pid in np.unique(after.cell_plate):
        cells=np.flatnonzero(after.cell_plate==pid);parents=np.unique(before.cell_plate[cells])
        if len(parents)==1:pure_errors.append(np.linalg.norm(new_w[pid]-old_w[parents[0]]))
    transfers=[]
    for parent in np.unique(before.cell_plate[changed]):
        for child in np.unique(after.cell_plate[changed & (before.cell_plate==parent)]):
            selected=np.flatnonzero(changed & (before.cell_plate==parent) & (after.cell_plate==child))
            for cells in connected_components(selected,mesh.neighbors):
                cells=np.array(cells,dtype=int)
                v0=np.cross(old_w[parent],radius*mesh.centroids[cells])
                v1=np.cross(new_w[child],radius*mesh.centroids[cells])
                dv=v1-v0
                transfers.append(dict(kind='contact_fragment_transfer',parent=int(parent),child=int(child),
                    cells=cells.tolist(),area_km2=float(areas[cells].sum()),
                    omega_before_rad_per_myr=old_w[parent].tolist(),omega_after_rad_per_myr=new_w[child].tolist(),
                    velocity_change_km_per_myr=dv.tolist(),max_speed_change_km_per_myr=float(np.max(np.linalg.norm(dv,axis=1))),
                    area_weighted_kinetic_proxy_change_km4_per_myr2=float(.5*np.sum(areas[cells]*(np.sum(v1*v1,axis=1)-np.sum(v0*v0,axis=1)))),
                    mechanical_work_closed=False,material_fields_changed=False,
                    rule='existing small-fragment threshold and thin-shell momentum pickup; diagnostic event, not a new capture law'))
    return dict(fragment_transfers=transfers,detached_components=components,reassigned_cells=int(changed.sum()),
        reassigned_area_km2=float(areas[changed].sum()),promoted=repair.promoted_components,
        thin_shell_momentum_relative_residual=float(np.linalg.norm(L1-L0)/max(np.linalg.norm(L0),1e-30)),
        max_pure_child_omega_error=float(max(pure_errors,default=0)),
        connected_domains=all(len(connected_components(np.flatnonzero(after.cell_plate==p),mesh.neighbors))==1 for p in np.unique(after.cell_plate)),
        max_ownership_velocity_change_km_per_myr=float(np.max(np.linalg.norm(np.cross(new_w[after.cell_plate]-old_w[before.cell_plate],radius*mesh.centroids),axis=1))))
