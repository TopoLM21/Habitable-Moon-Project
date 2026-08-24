"""v0.22 effective elastic-plate flexure on the spherical surface mesh.

The solver uses a finite-volume Laplace operator on triangular face cells and
solves the variable-rigidity floating-plate equation

    w + (1/k) L[D L[w]] = h_local,

where k = Delta_rho*g and h_local is the local-isostatic/tectonic target anomaly.
In weak form the system is symmetric positive definite:

    (M + K diag(D/A) K / k) w = M h_local.

This preserves the area-weighted mean anomaly, gives the oscillatory flexural
Green response (foreland/forearc basin and outer-rise side lobes), and remains
mesh-physical because all distances and cell areas are in SI units.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import LinearOperator, cg

from .mesh import SphereMesh
from .lithosphere import LithosphereState

Array=np.ndarray

@dataclass(slots=True)
class FlexureParameters:
    enabled: bool = True
    young_modulus_pa: float = 70.0e9
    poisson_ratio: float = 0.25
    restoring_density_contrast_kg_m3: float = 2500.0
    elastic_fraction_of_mechanical_lithosphere: float = 0.25
    min_elastic_thickness_km: float = 4.0
    max_elastic_thickness_km: float = 50.0
    rift_damage_gain: float = 1.15
    seam_damage_gain: float = 0.75
    tidal_damage_gain: float = 0.35
    min_damage_factor: float = 0.28
    cg_rtol: float = 2.0e-7
    cg_maxiter: int = 220

@dataclass(slots=True)
class FlexureDiagnostics:
    mean_elastic_thickness_km: float = 0.0
    min_elastic_thickness_km: float = 0.0
    max_elastic_thickness_km: float = 0.0
    mean_flexural_parameter_km: float = 0.0
    max_abs_flexural_correction_m: float = 0.0
    rms_flexural_correction_m: float = 0.0
    cg_iterations: int = 0
    cg_converged: bool = True
    area_mean_source_m: float = 0.0
    area_mean_response_m: float = 0.0

# Process-local cache. Mesh geometry and radius do not change during a run.
_GEOM_CACHE: dict[tuple[int,float], tuple[csr_matrix,Array]] = {}

def _geometry_operators(mesh: SphereMesh, radius_km: float) -> tuple[csr_matrix,Array]:
    key=(int(mesh.cell_count),round(float(radius_km),6))
    got=_GEOM_CACHE.get(key)
    if got is not None:
        return got
    n=mesh.cell_count
    R=float(radius_km)*1000.0
    areas=np.asarray(mesh.areas_unit_sphere,dtype=np.float64)*(R*R)
    rows=[]; cols=[]; vals=[]
    diag=np.zeros(n,dtype=np.float64)
    for fa,fb,u,v in mesh.shared_edges:
        fa=int(fa);fb=int(fb);u=int(u);v=int(v)
        edge_ang=float(np.arccos(np.clip(np.dot(mesh.vertices[u],mesh.vertices[v]),-1.0,1.0)))
        ctr_ang=float(np.arccos(np.clip(np.dot(mesh.centroids[fa],mesh.centroids[fb]),-1.0,1.0)))
        edge=max(edge_ang*R,1e-9); dist=max(ctr_ang*R,1e-9)
        c=edge/dist
        diag[fa]+=c;diag[fb]+=c
        rows.extend((fa,fb));cols.extend((fb,fa));vals.extend((-c,-c))
    rows.extend(range(n));cols.extend(range(n));vals.extend(diag.tolist())
    K=csr_matrix((np.asarray(vals), (np.asarray(rows),np.asarray(cols))),shape=(n,n))
    _GEOM_CACHE[key]=(K,areas)
    return K,areas

def effective_elastic_thickness_km(state: LithosphereState, params: FlexureParameters) -> Array:
    n=len(state.crust_thickness_km)
    crust=np.maximum(np.asarray(state.crust_thickness_km,dtype=np.float64),0.0)
    if state.mantle_lithosphere_thickness_km is None:
        mech=crust+80.0
    else:
        mech=crust+np.maximum(np.asarray(state.mantle_lithosphere_thickness_km,dtype=np.float64),0.0)
    Te=float(params.elastic_fraction_of_mechanical_lithosphere)*mech
    rift=np.zeros(n) if state.rift_extension is None else np.maximum(np.asarray(state.rift_extension,dtype=np.float64),0.0)
    seam=np.zeros(n) if state.collision_seam_weakness is None else np.clip(np.asarray(state.collision_seam_weakness,dtype=np.float64),0.0,1.0)
    tidal=np.clip(np.asarray(state.tidal_damage,dtype=np.float64),0.0,1.0)
    damage=np.exp(-float(params.rift_damage_gain)*rift)
    damage*=np.clip(1.0-float(params.seam_damage_gain)*seam,float(params.min_damage_factor),1.0)
    damage*=np.clip(1.0-float(params.tidal_damage_gain)*tidal,float(params.min_damage_factor),1.0)
    damage=np.clip(damage,float(params.min_damage_factor),1.0)
    Te*=damage
    return np.clip(Te,float(params.min_elastic_thickness_km),float(params.max_elastic_thickness_km))

def flexural_rigidity_nm(Te_km: Array, params: FlexureParameters) -> Array:
    Te=np.asarray(Te_km,dtype=np.float64)*1000.0
    E=float(params.young_modulus_pa); nu=float(params.poisson_ratio)
    return E*Te**3/(12.0*max(1.0-nu*nu,1e-12))

def flexural_parameter_km(Te_km: Array, gravity_m_s2: float, params: FlexureParameters) -> Array:
    D=flexural_rigidity_nm(Te_km,params)
    k=max(float(params.restoring_density_contrast_kg_m3)*float(gravity_m_s2),1e-12)
    return (4.0*D/k)**0.25/1000.0

def solve_flexural_response(
    mesh: SphereMesh,
    state: LithosphereState,
    local_target_m: Array,
    radius_km: float,
    gravity_m_s2: float,
    params: FlexureParameters,
) -> tuple[Array,FlexureDiagnostics,Array,Array]:
    h=np.asarray(local_target_m,dtype=np.float64)
    if h.shape!=(mesh.cell_count,): raise ValueError('local_target_m must match mesh cell count')
    Te=effective_elastic_thickness_km(state,params)
    alpha=flexural_parameter_km(Te,gravity_m_s2,params)
    if not bool(params.enabled):
        d=FlexureDiagnostics(mean_elastic_thickness_km=float(np.mean(Te)),min_elastic_thickness_km=float(np.min(Te)),max_elastic_thickness_km=float(np.max(Te)),mean_flexural_parameter_km=float(np.mean(alpha)),area_mean_source_m=float(np.mean(h)),area_mean_response_m=float(np.mean(h)))
        return h.copy(),d,Te,alpha
    K,A=_geometry_operators(mesh,radius_km)
    D=flexural_rigidity_nm(Te,params)
    q=D/np.maximum(A,1e-30)
    k=max(float(params.restoring_density_contrast_kg_m3)*float(gravity_m_s2),1e-12)
    def matvec(x):
        x=np.asarray(x,dtype=np.float64)
        return A*x + (K @ (q*(K @ x)))/k
    diagB=np.asarray(K.multiply(K) @ q).ravel()
    diagA=A+diagB/k
    op=LinearOperator((len(A),len(A)),matvec=matvec,dtype=np.float64)
    pre=LinearOperator((len(A),len(A)),matvec=lambda x: np.asarray(x)/np.maximum(diagA,1e-30),dtype=np.float64)
    rhs=A*h
    iterations=[0]
    def cb(_): iterations[0]+=1
    w,info=cg(op,rhs,x0=h,rtol=float(params.cg_rtol),atol=0.0,maxiter=int(params.cg_maxiter),M=pre,callback=cb)
    if info<0: raise RuntimeError(f'flexure CG failed with illegal input/breakdown info={info}')
    correction=w-h
    mean_src=float(np.sum(A*h)/np.sum(A)); mean_rsp=float(np.sum(A*w)/np.sum(A))
    diag=FlexureDiagnostics(
        mean_elastic_thickness_km=float(np.mean(Te)),min_elastic_thickness_km=float(np.min(Te)),max_elastic_thickness_km=float(np.max(Te)),mean_flexural_parameter_km=float(np.mean(alpha)),
        max_abs_flexural_correction_m=float(np.max(np.abs(correction))),rms_flexural_correction_m=float(np.sqrt(np.mean(correction*correction))),cg_iterations=int(iterations[0]),cg_converged=bool(info==0),area_mean_source_m=mean_src,area_mean_response_m=mean_rsp)
    return np.asarray(w,dtype=np.float64),diag,Te,alpha

__all__=['FlexureParameters','FlexureDiagnostics','effective_elastic_thickness_km','flexural_rigidity_nm','flexural_parameter_km','solve_flexural_response']
