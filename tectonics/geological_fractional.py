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
from .geological_planar import intersection,difference,vacancy
from .geological_roundoff import area_budget,conservative_weights,convex_roundoff
from .dynamics import angular_velocity_vectors
from .geological_material import CONVERSION,coalesce_node


def enable(engine):
    if engine.record.get('fractional_surface') is not None:
        raise ValueError('Fractional surface already initialized; preserve its saved geometry')
    if not engine.record.get('mature_cycle'):engine.enable_mature_cycle(continental_displacement='retire_added_continental_area')
    else:engine.record['continental_displacement']='retire_added_continental_area'
    rows=[];state=engine.checkpoint.state
    for face,vertices in enumerate(engine.mesh.faces):
        identity=f'patch-0-{face}'
        rows.append(dict(id=identity,owner=int(state.cell_plate[face]),cell=face,vertices=engine.mesh.vertices[vertices].tolist()))
        for node in engine.record['material_parcels']['nodes'][face]:
            for part in engine.record['material_parcels']['node_parts'][str(node)]:part['surface_patch']=identity
    engine.record.update(fractional_surface=rows,fractional_generation=0,contact_diagnostics=True,
        fractional_rule='spherical overlap; native ocean priority and continental accretion/spreading; explicit mafic/H retirement; native ridge gaps',
        fractional_approval='user voice approval 01a13b7d-7800-7114-9878-5fdfff305aba',
        fractional_approximations=['scalar native force/topology raster is dominant physical area; it does not advect minority material',
            'thermal mixing within receiving fixed bins; patch identity survives',
            'same-owner same-cell convex geometry consolidation; material profiles mix within those bins',
            'binary64 boundary-tube roundoff budget; explicit conservative material correction',
            'continental columns accrete and use native footprint spreading; no oceanic sinking of felsic contact material'])


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


def _clean_roundoff(engine):
    rows=engine.record['fractional_surface'];r=engine.record['radius_km'];removed={};area_removed=0.;mass_moved=0.;mass_distance=0.;max_distance=0.
    areas={row['id']:polygon_area(np.asarray(row['vertices']))*r*r for row in rows}
    resolved=[row for row in rows if areas[row['id']]>area_budget(row['vertices'],r,stages=32)]
    for row in rows:
        if row in resolved:continue
        candidates=[x for x in resolved if x['owner']==row['owner']]
        if not candidates:raise ValueError('Numerically unresolved whole plate needs explicit topology resolution')
        centre=np.mean(row['vertices'],axis=0);centre/=np.linalg.norm(centre)
        def distance(x):
            c=np.mean(x['vertices'],axis=0);c/=np.linalg.norm(c)
            return float(np.arctan2(np.linalg.norm(np.cross(centre,c)),np.dot(centre,c))*r)
        target=min(candidates,key=lambda x:(x['cell']!=row['cell'],distance(x)))
        dist=distance(target);removed[row['id']]=(target['id'],dist);area_removed+=areas[row['id']];max_distance=max(max_distance,dist)
    for parts in engine.record['material_parcels']['node_parts'].values():
        for part in parts:
            if part['surface_patch'] in removed:
                target,dist=removed[part['surface_patch']];part['surface_patch']=target;mass_moved+=part['mass_kg'];mass_distance+=part['mass_kg']*dist
    engine.record['fractional_surface']=[row for row in rows if row['id'] not in removed]
    return dict(removed_polygons=len(removed),removed_area_km2=area_removed,conservatively_reassigned_mass_kg=mass_moved,
        max_reassignment_distance_km=max_distance,mass_distance_kg_km=mass_distance)


def project(engine,system,dt):
    mesh=engine.mesh;r=engine.record['radius_km'];old=engine.checkpoint.state
    engine.record['fractional_geometry_algorithm']='gnomonic-bounded-closure-geos-complement-v1'
    cleaning=_clean_roundoff(engine)
    old_rows=engine.record['fractional_surface'];by_id={x['id']:x for x in old_rows};part_by_id={x['id']:[] for x in old_rows}
    for parts in engine.record['material_parcels']['node_parts'].values():
        for p in parts:part_by_id[p['surface_patch']].append(p)
    patch_cont_area={}
    cell_areas=mesh.physical_cell_areas_km2(r)
    for row in old_rows:
        cell=row['cell'];volume=old.continental_volume_km3[cell]
        felsic=math.fsum(p['mass_kg'] for p in part_by_id[row['id']] if p.get('material_phase')=='felsic')/CONVERSION
        patch_cont_area[row['id']]=cell_areas[cell]*old.continental_fraction[cell]*felsic/volume if volume>0 else 0.
    continental_area=np.zeros(mesh.cell_count);contact_correction=0.;contact_budget=0.
    patches=[MaterialPatch(x['id'],x['owner'],np.asarray(x['vertices']),
        math.fsum(p['mass_kg'] for p in part_by_id[x['id']]),math.fsum(p['energy_j'] for p in part_by_id[x['id']]),'mixed_native' if any(p.get('material_phase')=='felsic' for p in part_by_id[x['id']]) else 'mafic') for x in old_rows]
    omega=angular_velocity_vectors(system);out=transfer_rigid_patches(mesh,patches,dict(enumerate(omega)),dt,r,roundoff_correction=True)
    buckets=[[] for _ in range(mesh.cell_count)]
    for piece in out.pieces:buckets[piece.target].append(piece)
    generation=engine.record['fractional_generation']+1;counter=0;rows=[];retired=[];generated=[]
    coverage=np.zeros((len(system.plates),mesh.cell_count));source_matrix=np.zeros((mesh.cell_count,mesh.cell_count))
    unresolved_gap_area=0.;boundary_repair_area=0.;maximum_gap_grid=0.;gap_area_budget=0.
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
        def priority(p):
            src=by_id[p.identity]['cell'];continental=old.crust_type[src]==1
            return ((0,old.tidal_damage[src],-old.crust_thickness_km[src],p.owner,p.identity) if continental else
                    (1,old.crust_age_myr[src],0,p.owner,p.identity))
        ordered=sorted(incoming,key=priority)
        occupied=[]
        for piece in ordered:
            origin=by_id[piece.identity];source=origin['cell'];source_area=polygon_area(np.asarray(origin['vertices']))*r*r
            survivors=[piece.vertices];allocations=[]
            for existing in occupied:
                if existing['owner']==piece.owner:continue
                next_survivors=[]
                for poly in survivors:
                    overlap=intersection(poly,existing['poly'],mesh.centroids[target])
                    if len(overlap):
                        a=polygon_area(overlap)*r*r;contact_area+=a;retired_area+=a
                        allocations.append((overlap,existing))
                    next_survivors.extend(part for part in difference(poly,existing['poly'],mesh.centroids[target])
                        if polygon_area(part)*r*r>area_budget(part,r,stages=2))
                survivors=next_survivors
                if not survivors:break
            all_polys=[p for p,_ in allocations]+survivors
            budget=area_budget(piece.vertices,r,stages=2+sum(len(x['poly']) for x in occupied));contact_budget+=budget
            weights,error=conservative_weights([polygon_area(p)*r*r for p in all_polys],piece.area_km2,budget)
            piece_mass=math.fsum(p['mass_kg'] for p in part_by_id[piece.identity])
            piece_fraction=piece.mass_kg/piece_mass if piece_mass else piece.area_km2/source_area
            contact_correction+=abs(error)
            continental_area[target]+=patch_cont_area[piece.identity]*piece_fraction
            for (poly,existing),w in zip(allocations,weights):
                transferred=scale_parts(part_by_id[piece.identity],piece_fraction*w,existing['id'])
                # Native continental columns accrete, they do not enter the
                # oceanic subduction sink. Preserve every felsic ID and H.
                existing['parts'].extend(p for p in transferred if p.get('material_phase')=='felsic')
                retired.extend(p for p in transferred if p.get('material_phase')!='felsic')
            for poly,w in zip(survivors,weights[len(allocations):]):
                a=polygon_area(poly)*r*r;parts=scale_parts(part_by_id[piece.identity],piece_fraction*w,'pending')
                row=append(poly,piece.owner,target,parts,{source:a});occupied.append(row)
                coverage[piece.owner,target]+=a;source_matrix[target,source]+=a
        for row in occupied:
            try:normalize_polygon(row['poly'])
            except ValueError:row['poly']=convex_roundoff(row['poly'],r)
        vacant,grid,stages=vacancy(mesh.vertices[mesh.faces[target]],[x['poly'] for x in occupied],mesh.centroids[target])
        gap_budget=area_budget(mesh.vertices[mesh.faces[target]],r,stages=stages)
        maximum_gap_grid=max(maximum_gap_grid,grid);gap_area_budget+=gap_budget
        # Newborn area attaches to the largest already present adjacent material;
        # if the whole target is vacant use the old local native owner. No
        # surviving material is reassigned to that owner by this operation.
        owner=int(np.argmax(coverage[:,target])) if occupied else int(old.cell_plate[target])
        for poly in vacant:
            a=polygon_area(poly)*r*r
            if a<=area_budget(poly,r,stages=stages):unresolved_gap_area+=a;continue
            created_area+=a;mass=a*7.*CONVERSION
            if mass<=0:raise ValueError('Nonpositive ridge piece')
            part=dict(id=f'fractional-ridge-{generation}-{counter}',mass_kg=mass,energy_j=mass*donor,
                composition='native_reference_mafic',material_phase='mafic',process='fractional_ridge',origin_reservoir='deep',
                time_myr=old.time_myr+dt,origin_face=target,birth_cell_id=None)
            row=append(poly,owner,target,[part],{});generated.append(dict(row['parts'][0]));coverage[owner,target]+=a
    # Consolidate unresolved boundary scraps inside the same owner/thermal cell
    # before serialization. This changes neither a thermal bin nor its m/H.
    cell_clean_area=0.;cell_clean_mass=0.;cell_clean_count=0;groups={}
    for row in rows:groups.setdefault((row['owner'],row['cell']),[]).append(row)
    retained=[]
    for group in groups.values():
        target=max(group,key=lambda row:polygon_area(row['poly']))
        for row in group:
            area=polygon_area(row['poly'])*r*r
            if row is not target and area<=area_budget(row['poly'],r,stages=32):
                cell_clean_area+=area;cell_clean_mass+=math.fsum(p['mass_kg'] for p in row['parts']);cell_clean_count+=1
                for p in row['parts']:p['surface_patch']=target['id']
                target['parts'].extend(row['parts'])
                for src,a in row['source_weights'].items():target['source_weights'][src]=target['source_weights'].get(src,0)+a
            else:retained.append(row)
    rows,merged=_merge_convex(retained,r)
    areas=mesh.physical_cell_areas_km2(r);covered=np.zeros(mesh.cell_count)
    for row in rows:covered[row['cell']]+=polygon_area(row['poly'])*r*r
    max_area_error=float(np.max(abs(covered-areas)/areas))
    if max_area_error>1e-10:
        cell=int(np.argmax(abs(covered-areas)/areas))
        error=ValueError(f'Fractional surface partition failed: {max_area_error}; cell={cell}; area error={covered[cell]-areas[cell]} km2')
        error.geometry_failure=dict(cell=cell,expected_area_km2=float(areas[cell]),rows=[dict(owner=x['owner'],vertices=np.asarray(x['poly']).tolist()) for x in rows if x['cell']==cell])
        raise error
    # Converting only the native mechanical diagnostic raster; minority patches
    # retain their own owners and angular velocities in the geometric inventory.
    owner=np.argmax(coverage,axis=0).astype(np.int32)
    pre=deepcopy(old);pre.cell_plate=owner
    profiles=[[[],[]] for _ in areas]
    for row in rows:
        for part in row['parts']:profiles[row['cell']][int(part.get('material_phase')=='felsic')].append(part)
    for field in fields(pre):
        name=field.name;value=getattr(old,name)
        if not isinstance(value,np.ndarray) or value.shape!=(mesh.cell_count,) or name in ('cell_plate','crust_type','oceanic_volume_km3','continental_fraction','continental_volume_km3','sediment_volume_km3'):continue
        weight=areas;mapped=(source_matrix@value)/weight
        setattr(pre,name,mapped)
    pre.oceanic_volume_km3=np.array([math.fsum(p['mass_kg'] for p in x[0])/CONVERSION for x in profiles])
    pre.continental_fraction=continental_area/areas
    pre.continental_volume_km3=np.array([math.fsum(p['mass_kg'] for p in x[1])/CONVERSION for x in profiles])
    # Resolve the same native footprint overflow before entering the identity
    # transport adapter (native field readers clip fractions to [0,1]).
    from .lithosphere import _redistribute_continental_footprint_overflow,continental_basalt_retention,effective_continental_thickness_km,effective_oceanic_thickness_km
    spreading=[]
    _redistribute_continental_footprint_overflow(mesh,areas,pre.continental_fraction,pre.continental_volume_km3,
        np.flatnonzero(pre.continental_fraction>1),transfers=spreading)
    for src,dst,volume in spreading:
        total=math.fsum(p['mass_kg'] for p in profiles[src][1]);fraction=volume*CONVERSION/total
        if fraction>1+1e-12:raise ValueError('Continental spreading exceeds donor')
        fraction=min(fraction,1.)
        profiles[dst][1].extend(scale_parts(profiles[src][1],fraction,'rebind'))
        profiles[src][1]=scale_parts(profiles[src][1],1-fraction,'rebind') if fraction<1 else []
    prior_cont=np.clip((source_matrix@old.continental_fraction)/areas,0.,1.)
    keep=continental_basalt_retention(areas,prior_cont,np.arange(len(areas)),pre.continental_fraction,
        pre.oceanic_volume_km3,'retire_added_continental_area')
    for cell,fraction in enumerate(keep):
        if fraction<1:
            retired.extend(scale_parts(profiles[cell][0],1-fraction,'retired'))
            profiles[cell][0]=[dict(p,mass_kg=p['mass_kg']*fraction,energy_j=p['energy_j']*fraction) for p in profiles[cell][0]] if fraction>0 else []
    pre.oceanic_volume_km3*=keep
    pre.crust_type=(pre.continental_fraction>=.5).astype(old.crust_type.dtype)
    ch=effective_continental_thickness_km(pre.continental_fraction,pre.continental_volume_km3,areas)
    oh=effective_oceanic_thickness_km(pre.continental_fraction,pre.oceanic_volume_km3,areas)
    pre.crust_thickness_km=np.where(pre.crust_type==1,ch,oh)
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
        for p in profile[0]+profile[1]:
            after.setdefault(p['id'],[]).append(p['mass_kg']);after_h.setdefault(p['id'],[]).append(p['energy_j'])
    for p in retired:
        after.setdefault(p['id'],[]).append(p['mass_kg']);after_h.setdefault(p['id'],[]).append(p['energy_j'])
    id_error=max((abs(math.fsum(after.get(k,[]))-math.fsum(v))/math.fsum(v) for k,v in before.items()),default=0.)
    id_h_error=max((abs(math.fsum(after_h.get(k,[]))-math.fsum(v))/max(abs(math.fsum(v)),1.) for k,v in before_h.items()),default=0.)
    if max(id_error,id_h_error)>1e-10:raise ValueError(f'Fractional identity budget failed: {id_error}')
    diagnostic=dict(end_cell_roundoff_cleanup=dict(polygons=cell_clean_count,area_km2=cell_clean_area,mass_kg=cell_clean_mass),gap_overlay_max_grid=maximum_gap_grid,gap_area_roundoff_budget_km2=gap_area_budget,unresolved_gap_area_km2=unresolved_gap_area,boundary_repair_area_km2=boundary_repair_area,roundoff_cleaning=cleaning,raw_footprint_extra_components=None,post_overlap_extra_components=None,max_assignment_error_km=None,
        committed_plates=len(system.plates),geometric_projection='true spherical pieces; raster assignment diagnostics not applicable; connectivity requires independent audit',
        max_cell_area_relative_error=max_area_error,max_identity_relative_error=id_error,max_identity_H_relative_error=id_h_error,
        area_balance_km2=float(covered.sum()-total_before+retired_area-created_area),
        retired_area_km2=retired_area,removed_surface_footprint_area_km2=retired_area,created_area_km2=created_area,contact_area_km2=contact_area,
        recycled_mafic_mass_kg=retiredM,contact_felsic_subduction_mass_kg=0.,
        active_patches=len(rows),merged_convex_pieces=merged,mixed_owner_cells=int(np.count_nonzero((coverage>0).sum(0)>1)),
        source_area_relative_error=out.diagnostics['max_source_area_relative_residual'],
        contact_absolute_area_correction_km2=contact_correction,contact_area_correction_budget_km2=contact_budget,
        source_area_correction_budget_km2=out.diagnostics['total_roundoff_area_budget_km2'],source_absolute_mass_correction_kg=out.diagnostics['total_absolute_mass_correction_kg'],continental_spreading_count=len(spreading),
        source_absolute_area_correction_km2=out.diagnostics['total_absolute_area_correction_km2'])
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
