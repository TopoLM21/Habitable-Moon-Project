"""Static free-edge membrane diagnostic for disconnected rigid plate domains.

This helper is deliberately not installed in live fracture evolution. It has
no Maxwell history, plasticity, contact law, or transported stress memory.
Boundary vertices are duplicated per connected load-bearing domain. Tangent
displacements are constrained to the sphere and natural edge traction is zero.
Three domain rotations are gauges; the unbalanced rigid-load component is
reported separately because plate dynamics owns its torque.

Input basal traction is specified at each face's three vertices in Cartesian
Pa, allowing different tractions on the two sides of a plate boundary. Output
stress is [sigma_11, sigma_22, sigma_12] in the same constant face frame as
genesis_shell.Membrane: e1 follows vertex0→vertex1, e2=n_face×e1. Strain's third
component is engineering shear gamma_12. Returned displacement is Cartesian m.
"""
from __future__ import annotations

from dataclasses import dataclass
import warnings
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve, MatrixRankWarning

from .genesis_shell import Membrane, maximum_total_strain
from .mesh import connected_components


@dataclass(frozen=True)
class PlateMembraneDiagnostic:
    stress_pa: np.ndarray
    strain: np.ndarray
    face_vertex_displacement_m: np.ndarray
    components: tuple[dict, ...]
    elastic_energy_j: float
    balanced_external_work_j: float
    max_principal_strain: float
    within_small_strain_limit: bool


def solve_plate_basal_membrane(mesh, cell_plate, face_vertex_traction_pa,
        thickness_km, *, radius_km, young_modulus_pa=6e10, poisson_ratio=.25,
        stiffness_fraction=None, small_strain_limit=.05):
    """Solve a static basal-load diagnostic, without modifying any input.

    The lumped nodal area metric projects only the net rotational load. This
    satisfies the Fredholm solvability condition of each free domain and makes
    an added rigid-slip traction contribute only to the reported removed torque.
    The projected load is not a new force, speed, or damage source.
    """
    owner = np.asarray(cell_plate)
    traction = np.asarray(face_vertex_traction_pa, dtype=float)
    depth = np.asarray(thickness_km, dtype=float)
    n = mesh.cell_count
    if owner.shape != (n,) or owner.dtype.kind not in "iu" or np.any(owner < 0):
        raise ValueError("Plate ownership must be nonnegative cell IDs")
    if traction.shape != (n,3,3) or not np.isfinite(traction).all():
        raise ValueError("Basal traction must have shape (faces, 3 vertices, 3 Cartesian components)")
    if depth.shape != (n,) or not np.isfinite(depth).all() or np.any(depth < 0):
        raise ValueError("Load-bearing thickness must be finite and nonnegative")
    modulus, radius = float(young_modulus_pa), float(radius_km)*1000.
    if not (np.isfinite(modulus) and modulus > 0 and np.isfinite(radius) and radius > 0
            and 0 < poisson_ratio < .49 and 0 < small_strain_limit <= .1):
        raise ValueError("Invalid membrane material, radius, or small-strain limit")
    fraction = np.ones(n) if stiffness_fraction is None else np.asarray(stiffness_fraction,dtype=float)
    if fraction.shape != (n,) or not np.isfinite(fraction).all() or np.any((fraction <= 0) | (fraction > 1)):
        raise ValueError("Membrane stiffness fractions must be finite in (0,1]")
    active = depth > 0
    if np.any(traction[~active] != 0.):
        raise ValueError("A zero-thickness face cannot support prescribed basal traction")
    geometry = Membrane(mesh,poisson_ratio)
    stress = np.zeros((n,3)); strain = np.zeros((n,3))
    displacement = np.zeros((n,3,3)); rows=[]
    total_energy=0.; total_work=0.
    for pid in np.unique(owner[active]):
        faces = np.flatnonzero(active & (owner == pid))
        for component in connected_components(faces,mesh.neighbors):
            cells=np.asarray(sorted(component),dtype=int)
            vertices, inverse=np.unique(mesh.faces[cells],return_inverse=True)
            local_faces=inverse.reshape(-1,3)
            ndof=2*len(vertices)
            dofs=(2*local_faces[:,:,None]+np.arange(2)).reshape(-1,6)
            b=geometry.b[cells,:,:6]
            area=mesh.areas_unit_sphere[cells]
            h=depth[cells]*1000.
            local_k=np.einsum('fai,ab,fbj,f->fij',b,geometry.d,b,area*h*fraction[cells])
            rr=np.repeat(dofs,6,axis=1).ravel();cc=np.tile(dofs,(1,6)).ravel()
            k=sparse.coo_matrix((local_k.ravel(),(rr,cc)),shape=(ndof,ndof)).tocsr()
            # With u=R*d, divide the physical stiffness and load by E*R^2:
            # K'=sum(A_unit*H*B^T D B) [m], f'=R/E*sum(A_unit*N*tau) [m].
            local_load=np.einsum('fvij,fvi->fvj',geometry.vertex_basis[mesh.faces[cells]],traction[cells])
            local_load*=area[:,None,None]*radius/(3.*modulus)
            rhs=np.zeros(ndof);np.add.at(rhs,dofs.ravel(),local_load.reshape(-1))
            lumped=np.zeros(len(vertices));np.add.at(lumped,local_faces.ravel(),np.repeat(area/3.,3))
            weight=np.repeat(lumped,2)
            basis=geometry.vertex_basis[vertices]
            modes=np.column_stack([np.einsum('vij,vi->vj',basis,np.cross(axis,mesh.vertices[vertices])).ravel()
                for axis in np.eye(3)])
            gram=modes.T@(weight[:,None]*modes)
            if np.linalg.matrix_rank(gram) != 3:
                raise ValueError("Load-bearing domain has unresolved rotational gauges")
            raw_torque=modes.T@rhs
            removed=weight[:,None]*modes@np.linalg.solve(gram,raw_torque)
            balanced=rhs-removed
            constraints=(modes.T*weight)
            constraints/=np.linalg.norm(constraints,axis=1)[:,None]
            constraints=sparse.csr_matrix(constraints)
            scale=float(np.average(h,weights=area))
            system=sparse.bmat([[k/scale,constraints.T],[constraints,None]],format='csc')
            target=np.r_[balanced/scale,np.zeros(3)]
            with warnings.catch_warnings():
                warnings.simplefilter('error',MatrixRankWarning)
                answer=spsolve(system,target)
            if not np.isfinite(answer).all():
                raise RuntimeError("Plate membrane equilibrium did not converge")
            d=answer[:ndof]
            strain[cells]=np.einsum('fai,fi->fa',b,d[dofs])
            stress[cells]=np.einsum('ab,fb->fa',geometry.d,strain[cells])*(modulus*fraction[cells,None])
            cartesian=np.einsum('vij,vj->vi',basis,d.reshape(-1,2))*radius
            displacement[cells]=cartesian[local_faces]
            energy=.5*float(np.sum(area*h*np.sum(strain[cells]*stress[cells],axis=1)))*radius**2
            work=float(d@balanced)*modulus*radius**2
            total_energy+=energy;total_work+=work
            residual=float(np.linalg.norm(system@answer-target)/max(np.linalg.norm(target),1e-30))
            relative_load=float(np.linalg.norm(balanced/np.sqrt(weight))/max(np.linalg.norm(rhs/np.sqrt(weight)),1e-30))
            rows.append(dict(plate_id=int(pid),first_face=int(cells[0]),face_count=len(cells),vertex_count=len(vertices),
                raw_basal_torque_nm=(raw_torque*modulus*radius**2).tolist(),
                removed_rigid_torque_nm=((modes.T@removed)*modulus*radius**2).tolist(),
                balanced_torque_nm=((modes.T@balanced)*modulus*radius**2).tolist(),
                retained_load_relative_norm=relative_load,equilibrium_relative_residual=residual,
                gauge_residual=float(np.linalg.norm(constraints@d)),
                elastic_energy_j=energy,balanced_external_work_j=work))
    maximum=maximum_total_strain(strain)
    return PlateMembraneDiagnostic(stress,strain,displacement,tuple(rows),total_energy,total_work,
        maximum,bool(maximum <= small_strain_limit))
