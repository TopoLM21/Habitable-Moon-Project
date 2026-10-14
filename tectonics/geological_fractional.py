"""Fractional rigid surface transport connected to the geological thermal ledger.

An explicit experimental mode: youngest ocean survives overlapping ocean;
removed material returns its actual mass/H and provenance to the deep reservoir.
Geometric polygons, not raster owners, determine the next material motion.
"""
from copy import deepcopy
from dataclasses import fields
import math
import numpy as np
from scipy.spatial import ConvexHull
from .geological_area_transport import MaterialPatch,transfer_rigid_patches
from .spherical_polygons import normalize_polygon,polygon_area,intersect_convex,subtract_convex
from .dynamics import angular_velocity_vectors
from .geological_material import CONVERSION,coalesce_node


def enable(engine):
    if engine.record.get('fractional_surface') is not None:
        raise ValueError('Fractional surface already initialized; preserve its saved geometry')
    engine.enable_mature_cycle(continental_displacement='retire_added_continental_area')
    rows=[];state=engine.checkpoint.state
    for face,vertices in enumerate(engine.mesh.faces):
        identity=f'patch-0-{face}'
        rows.append(dict(id=identity,owner=int(state.cell_plate[face]),cell=face,vertices=engine.mesh.vertices[vertices].tolist()))
        for node in engine.record['material_parcels']['nodes'][face]:
            for part in engine.record['material_parcels']['node_parts'][str(node)]:part['surface_patch']=identity
    engine.record.update(fractional_surface=rows,fractional_generation=0,contact_diagnostics=True,
        fractional_rule='spherical overlap; native youngest-ocean priority; explicit material/H retirement; native ridge gaps',
        fractional_approval='user voice approval 01a13b7d-7800-7114-9878-5fdfff305aba',
        fractional_approximations=['scalar native force/topology raster is dominant physical area; it does not advect minority material',
            'thermal mixing within receiving fixed bins; patch identity survives',
            'same-owner same-cell convex geometry consolidation; material profiles mix within those bins'])


def scale_parts(parts,fraction,patch_id):
    return [dict(p,mass_kg=p['mass_kg']*fraction,energy_j=p['energy_j']*fraction,surface_patch=patch_id) for p in parts]


def _merge_convex(rows,radius):
    """Merge only a gap-free convex union within one owner and receiving cell."""
    groups={}
    for row in rows:groups.setdefault((row['owner'],row['cell']),[]).append(row)
    merged=[];count=0
    for group in groups.values():
        if len(group)==1:merged.extend(group);continue
        verts=np.concatenate([x['poly'] for x in group]);centre=verts.sum(0);centre/=np.linalg.norm(centre)
        axis=np.eye(3)[np.argmin(abs(centre))];u=np.cross(centre,axis);u/=np.linalg.norm(u);v=np.cross(centre,u)
        denom=verts@centre
        if np.any(denom<=0):merged.extend(group);continue
        xy=np.c_[verts@u,verts@v]/denom[:,None]
        try:
            hull=ConvexHull(xy);poly=normalize_polygon(verts[hull.vertices])
        except (ValueError,RuntimeError):merged.extend(group);continue
        expected=math.fsum(polygon_area(x['poly']) for x in group);actual=polygon_area(poly)
        if abs(actual-expected)>2e-13*expected:merged.extend(group);continue
        row=dict(group[0]);row['poly']=poly;row['parts']=[p for x in group for p in x['parts']]
        row['source_weights']={}
        for x in group:
            for src,a in x['source_weights'].items():row['source_weights'][src]=row['source_weights'].get(src,0)+a
        for p in row['parts']:p['surface_patch']=row['id']
        merged.append(row);count+=len(group)-1
    return merged,count


def project(engine,system,dt):
    mesh=engine.mesh;r=engine.record['radius_km'];old=engine.checkpoint.state
    if np.any(old.continental_volume_km3>0):
        raise ValueError('FRACTIONAL_FELSIC_INTERFACE_REQUIRED: preserve checkpoint; continental accretion/spreading not yet mapped')
    old_rows=engine.record['fractional_surface'];by_id={x['id']:x for x in old_rows};part_by_id={x['id']:[] for x in old_rows}
    for parts in engine.record['material_parcels']['node_parts'].values():
        for p in parts:part_by_id[p['surface_patch']].append(p)
    patches=[MaterialPatch(x['id'],x['owner'],np.asarray(x['vertices']),
        math.fsum(p['mass_kg'] for p in part_by_id[x['id']]),math.fsum(p['energy_j'] for p in part_by_id[x['id']]),'mafic') for x in old_rows]
    omega=angular_velocity_vectors(system);out=transfer_rigid_patches(mesh,patches,dict(enumerate(omega)),dt,r)
    buckets=[[] for _ in range(mesh.cell_count)]
    for piece in out.pieces:buckets[piece.target].append(piece)
    generation=engine.record['fractional_generation']+1;counter=0;rows=[];retired=[];generated=[]
    coverage=np.zeros((len(system.plates),mesh.cell_count));source_matrix=np.zeros((mesh.cell_count,mesh.cell_count))
    contact_area=0.;created_area=0.;retired_area=0.;total_before=math.fsum(polygon_area(np.asarray(x['vertices']))*r*r for x in old_rows)
    donor=engine.record['thermal_inventory']['energy_j'][-2]/engine.record['thermal_inventory']['mass_kg'][-2]
    def append(poly,owner,target,parts,source_weights):
        nonlocal counter
        identity=f'patch-{generation}-{counter}';counter+=1
        for p in parts:p['surface_patch']=identity
        row=dict(id=identity,owner=int(owner),cell=int(target),poly=poly,parts=parts,source_weights=source_weights)
        rows.append(row);return row
    for target,incoming in enumerate(buckets):
        # Same rule as native ocean-ocean overlap; deterministic source tie only
        # after plate priority, so a change in polygon enumeration has no effect.
        ordered=sorted(incoming,key=lambda p:(old.crust_age_myr[by_id[p.identity]['cell']],p.owner,p.identity))
        occupied=[]
        for piece in ordered:
            origin=by_id[piece.identity];source=origin['cell'];source_area=polygon_area(np.asarray(origin['vertices']))*r*r
            survivors=[piece.vertices]
            for existing in occupied:
                if existing['owner']==piece.owner:continue
                next_survivors=[]
                for poly in survivors:
                    overlap=intersect_convex(poly,existing['poly'])
                    if len(overlap):
                        a=polygon_area(overlap)*r*r;contact_area+=a;retired_area+=a
                        retired.extend(scale_parts(part_by_id[piece.identity],a/source_area,'retired'))
                    next_survivors.extend(subtract_convex(poly,existing['poly']))
                survivors=next_survivors
                if not survivors:break
            for poly in survivors:
                a=polygon_area(poly)*r*r;parts=scale_parts(part_by_id[piece.identity],a/source_area,'pending')
                row=append(poly,piece.owner,target,parts,{source:a});occupied.append(row)
                coverage[piece.owner,target]+=a;source_matrix[target,source]+=a
        vacant=[normalize_polygon(mesh.vertices[mesh.faces[target]])]
        for existing in occupied:
            vacant=[part for poly in vacant for part in subtract_convex(poly,existing['poly'])]
        # Newborn area attaches to the largest already present adjacent material;
        # if the whole target is vacant use the old local native owner. No
        # surviving material is reassigned to that owner by this operation.
        owner=int(np.argmax(coverage[:,target])) if occupied else int(old.cell_plate[target])
        for poly in vacant:
            a=polygon_area(poly)*r*r;created_area+=a;mass=a*7.*CONVERSION
            if mass<=0:raise ValueError('Nonpositive ridge piece')
            part=dict(id=f'fractional-ridge-{generation}-{counter}',mass_kg=mass,energy_j=mass*donor,
                composition='native_reference_mafic',material_phase='mafic',process='fractional_ridge',origin_reservoir='deep',
                time_myr=old.time_myr+dt,origin_face=target,birth_cell_id=None)
            row=append(poly,owner,target,[part],{});generated.append(dict(row['parts'][0]));coverage[owner,target]+=a
    rows,merged=_merge_convex(rows,r)
    areas=mesh.physical_cell_areas_km2(r);covered=np.zeros(mesh.cell_count)
    for row in rows:covered[row['cell']]+=polygon_area(row['poly'])*r*r
    max_area_error=float(np.max(abs(covered-areas)/areas))
    if max_area_error>1e-10:raise ValueError(f'Fractional surface partition failed: {max_area_error}')
    # Converting only the native mechanical diagnostic raster; minority patches
    # retain their own owners and angular velocities in the geometric inventory.
    owner=np.argmax(coverage,axis=0).astype(np.int32)
    pre=deepcopy(old);pre.cell_plate=owner
    profiles=[[[],[]] for _ in areas]
    for row in rows:profiles[row['cell']][0].extend(row['parts'])
    for field in fields(pre):
        name=field.name;value=getattr(old,name)
        if not isinstance(value,np.ndarray) or value.shape!=(mesh.cell_count,) or name in ('cell_plate','crust_type','oceanic_volume_km3','continental_fraction','continental_volume_km3','sediment_volume_km3'):continue
        weight=areas;mapped=(source_matrix@value)/weight
        setattr(pre,name,mapped)
    pre.oceanic_volume_km3=np.array([math.fsum(p['mass_kg'] for p in x[0])/CONVERSION for x in profiles])
    pre.crust_thickness_km=pre.oceanic_volume_km3/areas
    pre.continental_fraction=np.zeros(len(areas));pre.continental_volume_km3=np.zeros(len(areas))
    pre.crust_type=np.zeros_like(pre.crust_type)
    source_weights=source_matrix/areas[:,None]
    engine.record['fractional_surface']=[dict(id=x['id'],owner=x['owner'],cell=x['cell'],vertices=np.asarray(x['poly']).tolist()) for x in rows]
    engine.record['fractional_generation']=generation
    createdM=math.fsum(x['mass_kg'] for x in generated);retiredM=math.fsum(x['mass_kg'] for x in retired)
    # Independent source identity readback (including retired parts) precedes heat.
    before={};after={};before_h={};after_h={}
    for group in part_by_id.values():
        for p in group:
            before.setdefault(p['id'],[]).append(p['mass_kg']);before_h.setdefault(p['id'],[]).append(p['energy_j'])
    for profile in profiles:
        for p in profile[0]:
            after.setdefault(p['id'],[]).append(p['mass_kg']);after_h.setdefault(p['id'],[]).append(p['energy_j'])
    for p in retired:
        after.setdefault(p['id'],[]).append(p['mass_kg']);after_h.setdefault(p['id'],[]).append(p['energy_j'])
    id_error=max((abs(math.fsum(after.get(k,[]))-math.fsum(v))/math.fsum(v) for k,v in before.items()),default=0.)
    id_h_error=max((abs(math.fsum(after_h.get(k,[]))-math.fsum(v))/max(abs(math.fsum(v)),1.) for k,v in before_h.items()),default=0.)
    if max(id_error,id_h_error)>1e-10:raise ValueError(f'Fractional identity budget failed: {id_error}')
    diagnostic=dict(raw_footprint_extra_components=None,post_overlap_extra_components=None,max_assignment_error_km=None,
        committed_plates=len(system.plates),geometric_projection='true spherical pieces; raster assignment diagnostics not applicable; connectivity requires independent audit',
        max_cell_area_relative_error=max_area_error,max_identity_relative_error=id_error,max_identity_H_relative_error=id_h_error,
        area_balance_km2=float(covered.sum()-total_before+retired_area-created_area),
        retired_area_km2=retired_area,created_area_km2=created_area,contact_area_km2=contact_area,
        active_patches=len(rows),merged_convex_pieces=merged,mixed_owner_cells=int(np.count_nonzero((coverage>0).sum(0)>1)),
        source_area_relative_error=out.diagnostics['max_source_area_relative_residual'])
    return pre,dict(fractional_profiles=profiles,fractional_retired=retired,fractional_generated=generated,
        fractional_geometry=diagnostic,fractional_source_weights=source_weights,
        fractional_created_volume=createdM/CONVERSION,fractional_retired_volume=retiredM/CONVERSION)


def remap_owners(engine,old_system,new_system):
    """Apply existing topology lineage; minority material is not silently welded."""
    old=old_system.cell_plate;new=new_system.cell_plate;fallback={}
    for pid in range(len(old_system.plates)):
        cells=np.flatnonzero(old==pid)
        if len(cells):fallback[pid]=int(np.bincount(new[cells]).argmax())
    for row in engine.record['fractional_surface']:
        p=row['owner'];cell=row['cell']
        if old[cell]==p:row['owner']=int(new[cell])
        elif p in fallback:row['owner']=fallback[p]
        else:raise ValueError('Fractional minority plate lost its native domain; explicit topology resolution required')


def bind_new_parts(engine):
    """Attach native cycle births to the receiving cell's real surface pieces."""
    groups={}
    for row in engine.record['fractional_surface']:groups.setdefault(row['cell'],[]).append(row)
    parcel=engine.record['material_parcels']
    for cell,nodes in enumerate(parcel['nodes']):
        local=groups[cell];valid={p['id'] for p in local};a=np.array([polygon_area(np.asarray(p['vertices'])) for p in local]);a/=a.sum()
        for node in nodes:
            out=[]
            for part in parcel['node_parts'][str(node)]:
                if part.get('surface_patch') in valid:out.append(part)
                else:
                    usedM=[];usedH=[]
                    for i,(row,w) in enumerate(zip(local,a)):
                        m=part['mass_kg']-math.fsum(usedM) if i==len(local)-1 else part['mass_kg']*w
                        h=part['energy_j']-math.fsum(usedH) if i==len(local)-1 else part['energy_j']*w
                        out.append(dict(part,mass_kg=m,energy_j=h,surface_patch=row['id']));usedM.append(m);usedH.append(h)
            parcel['node_parts'][str(node)]=coalesce_node(out)
