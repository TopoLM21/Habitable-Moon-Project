"""Opt-in geometric transfer of rigid material patches, before contact physics.

Source polygons remain authoritative. A receiving raster cell may contain
several pieces/owners; it does not acquire or retire material by winning an ID.
No basalt displacement, subduction, deformation or thermal solver is invoked.
"""
from dataclasses import dataclass
import math
import numpy as np
from .spherical_polygons import (
    GeometryDiagnostics, normalize_polygon, rotate_polygon, polygon_area, intersect_convex,
)


@dataclass(frozen=True)
class MaterialPatch:
    identity: str
    owner: int
    reference_vertices: np.ndarray
    mass_kg: float
    enthalpy_j: float
    composition: str


@dataclass(frozen=True)
class AreaPiece:
    identity: str
    owner: int
    target: int
    vertices: np.ndarray
    area_km2: float
    mass_kg: float
    enthalpy_j: float
    composition: str


@dataclass
class AreaTransfer:
    pieces: tuple[AreaPiece, ...]
    diagnostics: dict


class UnresolvedContact(ValueError):
    """Several physical material owners cannot be replaced by one raster ID."""


def _cap(vertices):
    centre = np.sum(vertices,axis=0)
    centre /= np.linalg.norm(centre)
    radius = float(np.max(np.arccos(np.clip(vertices @ centre,-1,1))))
    if radius >= math.pi/2:
        raise ValueError('Area transfer supports small convex hemisphere patches')
    return centre,radius


def transfer_rigid_patches(mesh, patches, omega_by_owner, elapsed_myr, radius_km):
    """Intersect prescribed rigid poses with mesh; preserve every source identity.

    elapsed_myr refers to each patch's fixed reference pose. Do not replace
    reference patches with raster averages between steps (that diffuses material).
    No column/row renormalization or positive-area cutoff is used. The routine
    refuses a measured conservation failure instead of hiding it with a scale.
    """
    if not np.isfinite(elapsed_myr) or elapsed_myr < 0 or not np.isfinite(radius_km) or radius_km <= 0:
        raise ValueError('Finite nonnegative elapsed time and positive radius required')
    patches=tuple(patches)
    if len({p.identity for p in patches})!=len(patches):
        raise ValueError('Duplicate material source identity')
    targets=[normalize_polygon(mesh.vertices[face]) for face in mesh.faces]
    caps=[_cap(p) for p in targets]
    centres=np.array([c for c,r in caps]);radii=np.array([r for c,r in caps])
    diagnostics=GeometryDiagnostics();pieces=[];source_rows=[];calls=0
    for patch in sorted(patches,key=lambda p:p.identity):
        if patch.owner not in omega_by_owner or not np.isfinite(patch.mass_kg) or patch.mass_kg<0 or not np.isfinite(patch.enthalpy_j):
            raise ValueError('Invalid material inventory or missing prescribed motion')
        ref=normalize_polygon(patch.reference_vertices)
        area=polygon_area(ref)*radius_km**2
        moved=rotate_polygon(ref,omega_by_owner[patch.owner],elapsed_myr)
        centre,radius=_cap(moved)
        # Convex spherical caps contain their minor arcs. The margin protects
        # candidate selection at floating-point equality; it does not trim area.
        candidate=np.flatnonzero(centres@centre>=np.cos(np.minimum(radius+radii,math.pi))-64*np.finfo(float).eps)
        own=[]
        for target in candidate:
            calls+=1
            clipped=intersect_convex(moved,targets[target],diagnostics=diagnostics)
            if not len(clipped):
                continue
            a=polygon_area(clipped)*radius_km**2
            own.append(AreaPiece(patch.identity,patch.owner,int(target),clipped,a,
                patch.mass_kg*(a/area),patch.enthalpy_j*(a/area),patch.composition))
        total=math.fsum(p.area_km2 for p in own)
        error=(total-area)/area
        if abs(error)>1e-10:
            raise ValueError(f'Geometric source area failed for {patch.identity}: {error}')
        source_rows.append(dict(identity=patch.identity,owner=patch.owner,area_km2=area,
            area_residual_km2=total-area,area_relative_residual=error,
            mass_residual_kg=math.fsum(p.mass_kg for p in own)-patch.mass_kg,
            H_residual_j=math.fsum(p.enthalpy_j for p in own)-patch.enthalpy_j))
        pieces.extend(own)
    return AreaTransfer(tuple(pieces),dict(sources=source_rows,intersection_calls=calls,
        ambiguous_plane_predicates=diagnostics.ambiguous_plane_predicates,
        lower_dimensional_results=diagnostics.lower_dimensional_results,
        max_source_area_relative_residual=max((abs(r['area_relative_residual']) for r in source_rows),default=0),
        total_area_residual_km2=math.fsum(r['area_residual_km2'] for r in source_rows),
        total_mass_residual_kg=math.fsum(r['mass_residual_kg'] for r in source_rows),
        total_H_residual_j=math.fsum(r['H_residual_j'] for r in source_rows)))


def inspect_two_owner_contact(mesh, transfer, radius_km):
    """Measure true overlaps separately from mere coexistence within one cell.

    Union = sum of material areas - pair intersections for two disjoint rigid
    plates. More than two owners is refused here: it needs higher intersections.
    Physical collisions remain pending, with both identities and both owners.
    """
    if len({p.owner for p in transfer.pieces})>2:
        raise ValueError('Local contact inspector supports at most two rigid owners')
    buckets=[[] for _ in range(mesh.cell_count)]
    for piece in transfer.pieces:buckets[piece.target].append(piece)
    rows=[];contacts=[]
    areas=mesh.physical_cell_areas_km2(radius_km)
    for target,pieces in enumerate(buckets):
        overlap=[]
        for i,a in enumerate(pieces):
            for b in pieces[i+1:]:
                if a.owner==b.owner:continue
                clipped=intersect_convex(a.vertices,b.vertices)
                if len(clipped):
                    area=polygon_area(clipped)*radius_km**2
                    overlap.append(area)
                    contacts.append(dict(target=target,identities=[a.identity,b.identity],owners=[a.owner,b.owner],
                        area_km2=area,status='pending_contact_law_no_retirement_or_owner_change'))
        material=math.fsum(p.area_km2 for p in pieces);collision=math.fsum(overlap)
        gap=float(areas[target])-material+collision
        if gap < -1e-10*areas[target]:
            raise ValueError('Contact union overfills target: invalid source partition')
        owners=sorted({p.owner for p in pieces})
        rows.append(dict(target=target,owners=owners,material_area_km2=material,
            collision_area_km2=collision,gap_area_km2=gap,
            exclusive_owner=owners[0] if len(owners)==1 else None))
    return dict(cells=rows,contacts=contacts,collision_area_km2=math.fsum(c['area_km2'] for c in contacts),
        gap_area_km2=math.fsum(c['gap_area_km2'] for c in rows),
        mixed_cells=sum(len(c['owners'])>1 for c in rows),
        unresolved_collision_cells=sum(c['collision_area_km2']>0 for c in rows))


def require_single_owner(contact):
    """Require exact full ownership; even an unresolved roundoff gap is refused.

    This guard does not snap a partial footprint into a whole-cell assignment.
    """
    if any(len(row['owners'])!=1 or row['gap_area_km2']!=0. for row in contact['cells']) or contact['contacts']:
        raise UnresolvedContact('Mixed/empty/contact cells need an explicit interface; no owner winner or material sink selected')
    return np.array([row['exclusive_owner'] for row in contact['cells']],dtype=np.int32)
