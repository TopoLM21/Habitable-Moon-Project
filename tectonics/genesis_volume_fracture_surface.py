"""Source-bound radial side surfaces on prospective six-DOF bank supports.

An explicitly supplied reference mesh edge is extruded through the actual
overlapping live material on its two incident faces.  The faces have different
mass-q charts: their cohorts are intersected at common *physical lambda*, not
at equal q.  Every positive patch retains both original cohort identities.

This is a mesh-conforming computational surface closure, not recovery of a
located fracture from the diffuse weak-plane field.  It neither selects a cut
nor admits physical creation, release, contact, or finite bank trajectories.
Surface area counts the paired interface once; a fracture energy per unit
interface area already includes both newly exposed banks.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import math

import numpy as np

from .genesis_volume_banks import GenesisVolumeBanks, _hash_value, validate_volume_banks
from .genesis_volume_operator import _immutable


FORMAT = "genesis-source-bound-prospective-volume-fracture-surface-0.1"
SCOPE = "mesh_conforming_reference_radial_solid_side_support_not_physical_fracture"


@dataclass(frozen=True, slots=True)
class SourceBoundVolumeFractureSurface:
    banks: GenesisVolumeBanks = field(repr=False)
    bank_binding_sha256: str
    source_checkpoint_sha256: str
    source_identity: str
    time_myr: float
    cut_edges: np.ndarray
    side_face_indices: np.ndarray
    edge_normals_xyz: np.ndarray
    edge_tangent1_xyz: np.ndarray
    edge_tangent2_xyz: np.ndarray
    patch_edge_indices: np.ndarray
    side_cohort_indices: np.ndarray
    side_cohort_ids: np.ndarray
    side_owner_ids: tuple
    side_material_ids: tuple
    side_q_bounds: np.ndarray
    lambda_bounds: np.ndarray
    quadrature_s: np.ndarray
    quadrature_lambda: np.ndarray
    quadrature_xyz_m: np.ndarray
    area_weights_m2: np.ndarray
    jump_trace: np.ndarray
    trace_dofs: np.ndarray
    patch_area_m2: np.ndarray
    edge_area_m2: np.ndarray
    unmatched_edge_indices: np.ndarray
    unmatched_side_indices: np.ndarray
    unmatched_cohort_indices: np.ndarray
    unmatched_cohort_ids: np.ndarray
    unmatched_lambda_bounds: np.ndarray
    unmatched_area_m2: np.ndarray
    binding_sha256: str
    format: str = FORMAT
    scope: str = SCOPE

    @property
    def patch_count(self):
        return len(self.patch_edge_indices)

    @property
    def total_area_m2(self):
        return float(self.edge_area_m2.sum())

    @property
    def calculated_fracture_surface(self):
        return False

    @property
    def physical_creation_admitted(self):
        return False

    @property
    def physical_contact_release_admitted(self):
        return False

    @property
    def physical_force_work_admission(self):
        return False


def _binding(surface):
    digest = hashlib.sha256()
    _hash_value(digest, surface)
    return digest.hexdigest()


def _array(values, shape, dtype=float):
    return _immutable(np.asarray(values, dtype=dtype).reshape(shape), dtype)


def _q_at_lambda(cohort, value, coefficient):
    # Preserve literal material endpoints instead of converting them through
    # a rounded lambda.  Other face boundaries require a separate chart.
    if value == cohort.outer_scale:
        return cohort.q_lo
    if value == cohort.inner_scale:
        return cohort.live_q_hi
    result = -math.expm1(3*math.log(value))/coefficient
    resolution = 32*np.finfo(float).eps/coefficient
    if not cohort.q_lo-resolution <= result <= cohort.live_q_hi+resolution:
        raise ValueError("Interface physical depth falls outside its side cohort chart")
    return min(max(result, cohort.q_lo), cohort.live_q_hi)


def _owners_at_interval(side_cohorts, lo, hi):
    result = [index for index, cohort in side_cohorts
              if cohort.inner_scale <= lo and hi <= cohort.outer_scale]
    if len(result) > 1:
        raise ValueError("Interface interval has overlapping material cohorts on one side")
    return result[0] if result else None


def _interface_intervals(side_cohorts):
    """Partition different side fronts without equating their material charts."""
    endpoints = sorted({value for current in side_cohorts for _, cohort in current
                        for value in (cohort.inner_scale, cohort.outer_scale)})
    result = []
    for lo, hi in zip(endpoints, endpoints[1:]):
        if not 0 < lo < hi <= 1:
            raise ValueError("Interface partition has a nonpositive physical radial interval")
        indices = [_owners_at_interval(current, lo, hi) for current in side_cohorts]
        if indices != [None, None]:
            result.append((lo, hi, indices))
    return result


def prepare_volume_fracture_surface(banks):
    """Integrate all supplied cut supports, retaining exact source-side IDs.

    The reference parameterization is x(s,lambda)=lambda*((1-s)Xu+sXv).
    Each patch uses two Gauss points in each coordinate; dA is
    lambda*|Xu cross Xv| ds dlambda.  This integrates quadratic six-DOF
    jump energy and its virtual work exactly.  Unmatched solid-only depths
    are recorded separately and receive no paired jump or creation charge.
    """
    validate_volume_banks(banks)
    parent = banks.parent_assembly
    geometry = parent.geometry
    vertices = np.asarray(geometry.vertices_m, float)
    faces = np.asarray(geometry.faces, int)
    topology = banks.topology
    cohorts = geometry.cohorts
    face_cohorts = [[] for _ in faces]
    for index, cohort in enumerate(cohorts):
        face_cohorts[cohort.face_index].append((index, cohort))
    coefficients = []
    for face, mass in zip(faces, geometry.column_mass_kg):
        triangle = vertices[face]
        r = float(np.linalg.norm(triangle[0]))
        determinant = float(np.linalg.det(triangle/r))
        coefficient = 6*(mass/geometry.density_kg_m3/r/r/r)/determinant
        if not math.isfinite(coefficient) or coefficient <= 0:
            raise ValueError("Interface side has an invalid original conical material chart")
        coefficients.append(coefficient)

    norms, tangent1, tangent2 = [], [], []
    pe, ci, ids, owners, materials, qb, lb = [], [], [], [], [], [], []
    qs, ql, points, weights, traces, dofs, areas = [], [], [], [], [], [], []
    ue, us, ui, uids, ulb, ua = [], [], [], [], [], []
    edge_area = np.zeros(len(topology.cut_edges))
    gauss = .5*(1+np.array((-1., 1.))/math.sqrt(3))

    for edge_index, ((u, v), side_faces, bank_vertices) in enumerate(zip(
            topology.cut_edges, topology.seam_faces, topology.bank_vertices)):
        x0, x1 = vertices[u], vertices[v]
        cross = np.cross(x0, x1)
        jacobian = float(np.linalg.norm(cross))
        direction = x1-x0
        length = float(np.linalg.norm(direction))
        if not math.isfinite(jacobian) or jacobian <= 0 or not math.isfinite(length) or length <= 0:
            raise ValueError("Interface edge is physically collapsed or nonfinite")
        normal = cross/jacobian
        third = next(vertex for vertex in faces[side_faces[0]] if vertex not in (u, v))
        orientation = float(normal@vertices[third])
        if not math.isfinite(orientation) or orientation == 0:
            raise ValueError("Interface cannot orient its two original material sides")
        normal *= -math.copysign(1., orientation)
        opposite = next(vertex for vertex in faces[side_faces[1]] if vertex not in (u, v))
        if normal@vertices[opposite] <= 0:
            raise ValueError("Interface side normals disagree with source face orientation")
        t1 = direction/length
        t2 = np.cross(normal, t1)
        norms.append(normal)
        tangent1.append(t1)
        tangent2.append(t2)

        side = [face_cohorts[int(face)] for face in side_faces]
        edge_dofs = (6*bank_vertices[:, :, None]+np.arange(6)).reshape(24)
        for lo, hi, index in _interface_intervals(side):
            patch_area = .5*jacobian*(hi-lo)*(hi+lo)
            if not math.isfinite(patch_area) or patch_area <= 0:
                raise ValueError("Interface patch has nonpositive physical area")
            if None in index:
                occupied = 0 if index[0] is not None else 1
                cohort_index = index[occupied]
                ue.append(edge_index)
                us.append(occupied)
                ui.append(cohort_index)
                uids.append(cohorts[cohort_index].id)
                ulb.append((lo, hi))
                ua.append(patch_area)
                continue
            selected = [cohorts[i] for i in index]
            pe.append(edge_index)
            ci.append(index)
            ids.append([cohort.id for cohort in selected])
            owners.append(tuple(cohort.owner_id for cohort in selected))
            materials.append(tuple(cohort.material_id for cohort in selected))
            side_q = [(_q_at_lambda(cohort, hi, coefficients[cohort.face_index]),
                       _q_at_lambda(cohort, lo, coefficients[cohort.face_index]))
                      for cohort in selected]
            if any(q_lo >= q_hi for q_lo, q_hi in side_q):
                raise ValueError("Interface patch is below resolvable positive side material depth")
            qb.append(side_q)
            lb.append((lo, hi))
            patch_s, patch_lambda, patch_points, patch_weights, patch_trace = [], [], [], [], []
            for s in gauss:
                for radial in gauss:
                    lam = lo+(hi-lo)*radial
                    patch_s.append(s)
                    patch_lambda.append(lam)
                    patch_points.append(lam*((1-s)*x0+s*x1))
                    patch_weights.append(.25*(hi-lo)*lam*jacobian)
                    trace = np.zeros((3, 24))
                    for bank, sign in ((0, -1.), (1, 1.)):
                        for endpoint, shape in enumerate((1-s, s)):
                            start = 12*bank+6*endpoint
                            trace[:, start:start+3] = sign*shape*np.eye(3)
                            trace[:, start+3:start+6] = sign*shape*(lam-1)*np.eye(3)
                    patch_trace.append(trace)
            qs.append(patch_s)
            ql.append(patch_lambda)
            points.append(patch_points)
            weights.append(patch_weights)
            traces.append(patch_trace)
            dofs.append(edge_dofs)
            areas.append(patch_area)
            edge_area[edge_index] += patch_area

    nedge, npatch, nunmatched = len(topology.cut_edges), len(pe), len(ue)
    surface = SourceBoundVolumeFractureSurface(banks, banks.binding_sha256,
        parent.source_checkpoint_sha256, parent.source_identity, parent.time_myr,
        topology.cut_edges, topology.seam_faces,
        _array(norms, (nedge, 3)), _array(tangent1, (nedge, 3)), _array(tangent2, (nedge, 3)),
        _array(pe, (npatch,), int), _array(ci, (npatch, 2), int), _array(ids, (npatch, 2), int),
        tuple(owners), tuple(materials), _array(qb, (npatch, 2, 2)), _array(lb, (npatch, 2)),
        _array(qs, (npatch, 4)), _array(ql, (npatch, 4)), _array(points, (npatch, 4, 3)),
        _array(weights, (npatch, 4)), _array(traces, (npatch, 4, 3, 24)), _array(dofs, (npatch, 24), int),
        _array(areas, (npatch,)), _immutable(edge_area),
        _array(ue, (nunmatched,), int), _array(us, (nunmatched,), int),
        _array(ui, (nunmatched,), int), _array(uids, (nunmatched,), int),
        _array(ulb, (nunmatched, 2)), _array(ua, (nunmatched,)), "")
    return replace(surface, binding_sha256=_binding(surface))


def validate_volume_fracture_surface(surface, *, banks=None):
    """Validate a prepared in-memory surface, not an untrusted restart DTO."""
    if (type(surface) is not SourceBoundVolumeFractureSurface
            or surface.format != FORMAT or surface.scope != SCOPE):
        raise ValueError("Volume fracture surface has an unsupported computational scope")
    validate_volume_banks(surface.banks)
    if banks is not None:
        validate_volume_banks(banks)
        if banks.binding_sha256 != surface.bank_binding_sha256:
            raise ValueError("Volume fracture surface belongs to different source-bound banks")
    if (surface.bank_binding_sha256 != surface.banks.binding_sha256
            or surface.binding_sha256 != _binding(surface)):
        raise ValueError("Volume fracture surface differs from its prepared source binding")
    return surface


def _real(value, name, shape):
    raw = np.asarray(value)
    if raw.dtype.kind not in "ifu":
        raise ValueError(f"{name} requires finite real numeric values")
    result = np.asarray(raw, float)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} has invalid shape or nonfinite values")
    return result


def evaluate_surface_jump(surface, bank_displacement_increment_m):
    """Return world Cartesian bank1-minus-bank0 jumps at area quadrature."""
    validate_volume_fracture_surface(surface)
    u = _real(bank_displacement_increment_m, "Surface bank increment", (surface.banks.ndof,))
    return _immutable(np.einsum("pqai,pi->pqa", surface.jump_trace, u[surface.trace_dofs]))


def integrate_surface_traction(surface, traction_pa):
    """Return the generalized force exactly dual to the jump trace.

    Traction is world Cartesian and conjugate to bank1-minus-bank0 jump;
    force on bank1 is positive, with the opposite force on bank0.  This
    evaluates a supplied constitutive field and makes no release admission.
    """
    validate_volume_fracture_surface(surface)
    traction = _real(traction_pa, "Surface traction", (surface.patch_count, 4, 3))
    local = np.einsum("pqai,pqa,pq->pi", surface.jump_trace, traction, surface.area_weights_m2)
    result = np.zeros(surface.banks.ndof)
    np.add.at(result, surface.trace_dofs.ravel(), local.ravel())
    return _immutable(result)
