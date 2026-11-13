"""Coarse basalt exchange on physical boundary lengths, V'=S-kV, k=v*L/A.

Frozen coefficients give a positive finite-donor update with a conserved age
moment. This is a mixed-column finite-volume approximation, not slab geometry.
Raster overlap itself is not an additional recycling operation.
"""
import math
import numpy as np
from .kinematics import BoundaryType


def frozen_exchange(volume,age_moment,source_rate,drain_rate,dt):
    v=np.asarray(volume,float);moment=np.asarray(age_moment,float)
    s=np.asarray(source_rate,float);k=np.asarray(drain_rate,float)
    if dt<=0 or not math.isfinite(dt) or any(x.shape!=v.shape or np.any(~np.isfinite(x)) or np.any(x<0) for x in (v,moment,s,k)):
        raise ValueError('Invalid finite-volume boundary exchange')
    x=k*dt;e=np.exp(-x);small=x<1e-4
    phi1=np.empty_like(x);phi2=np.empty_like(x);removed_source_fraction=np.empty_like(x)
    xs=x[small]
    phi1[small]=1-xs/2+xs**2/6-xs**3/24+xs**4/120
    phi2[small]=.5-xs/3+xs**2/8-xs**3/30+xs**4/144
    removed_source_fraction[small]=xs/2-xs**2/6+xs**3/24-xs**4/120
    xl=x[~small];el=e[~small]
    phi1[~small]=-np.expm1(-xl)/xl
    phi2[~small]=(-np.expm1(-xl)-xl*el)/(xl*xl)
    removed_source_fraction[~small]=1-phi1[~small]
    born=s*dt;returned=-np.expm1(-x)*v+born*removed_source_fraction
    final=e*v+born*phi1
    aged=e*(moment+v*dt)+s*dt*dt*phi2
    return final,aged,born,returned


def exchange_on_boundaries(mesh,state,boundaries,dt,radius_km,reference_thickness_km):
    from .lithosphere import continental_material_fields
    from .subduction_memory import choose_subducting_side
    areas=mesh.physical_cell_areas_km2(radius_km);frac,_=continental_material_fields(state,areas)
    ocean=1-np.clip(frac,0,1);source=np.zeros(len(areas));drain=np.zeros(len(areas))
    for b in boundaries:
        a,c=b.face_a,b.face_b
        u,w=mesh.vertices[b.vertex_u],mesh.vertices[b.vertex_v]
        length=radius_km*math.atan2(float(np.linalg.norm(np.cross(u,w))),float(u@w))
        if b.boundary_type==BoundaryType.DIVERGENT:
            for cell in (a,c):
                source[cell]+=.5*max(b.normal_rate_km_per_myr,0.)*length*reference_thickness_km*ocean[cell]
        elif b.boundary_type==BoundaryType.CONVERGENT:
            sub=choose_subducting_side(state,b)
            if sub is None:continue
            cell=a if sub==b.plate_a else c
            if ocean[cell]>0:drain[cell]+=max(-b.normal_rate_km_per_myr,0.)*length/areas[cell]
    return frozen_exchange(state.oceanic_volume_km3,state.oceanic_age_moment_km3_myr,source,drain,dt)
