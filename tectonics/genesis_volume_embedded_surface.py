"""Material surface quadrature and paired virtual traces inside Genesis cohorts.

A declared current world plane is pulled back once to the original material
coordinates.  Its reference and current area metrics are kept separately.
All traces use the original Cartesian A/P interpolation; pairing two traces
does not split a mesh, create DOFs, release a tie, or fund a fracture.

The force integration below is the exact discrete dual of the declared trace
and chosen area weights.  It does not equate the saved small-strain stress
times a current normal to the parent's reference-energy force.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
import hashlib
import math
from numbers import Integral

import numpy as np

from .genesis_volume_banks import _hash_value
from .genesis_volume_checkpoint import _canonical
from .genesis_volume_material_sections import (
    CohortMaterialSection, _number, _q_at_lambda, _validate_cohort_section,
    prepare_volume_material_sections, volume_material_sections_to_dict,
)
from .genesis_volume_operator import _immutable


FORMAT = "genesis-original-material-embedded-surface-trace-0.1"
SCOPE = "explicit_current_plane_material_quadrature_paired_virtual_trace_no_physical_insertion"


class EmbeddedSurfaceRefusal(ValueError):
    """A declared surface cannot be resolved in its original material chart."""


def _numeric(value, name, shape=None, *, integer=False):
    def boolean(v):
        return isinstance(v, (bool, np.bool_)) or (
            isinstance(v, (list, tuple)) and any(boolean(x) for x in v))
    if boolean(value):
        raise EmbeddedSurfaceRefusal(f"{name} requires real numeric values")
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise EmbeddedSurfaceRefusal(f"{name} requires a numeric array") from error
    if raw.dtype.kind not in ("iu" if integer else "ifu") and not (
            integer and raw.size == 0 and raw.dtype.kind == "f"):
        raise EmbeddedSurfaceRefusal(f"{name} requires real numeric values")
    if integer and raw.dtype.kind == "u" and np.any(raw > np.iinfo(np.int64).max):
        raise EmbeddedSurfaceRefusal(f"{name} is outside the integer range")
    result = np.asarray(raw, np.int64 if integer else float)
    if (shape is not None and result.shape != shape) or not np.isfinite(result).all():
        raise EmbeddedSurfaceRefusal(f"{name} has invalid shape or nonfinite values")
    return result


def _order(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or not 2 <= value <= 64:
        raise EmbeddedSurfaceRefusal("Surface quadrature order must be an integer from 2 through 64")
    return int(value)


def _binding(value):
    digest = hashlib.sha256()
    _hash_value(digest, value)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class GenesisVolumeEmbeddedSurface:
    material_sections: object | None
    section_records: tuple
    material_sections_binding_sha256: str
    source_checkpoint_sha256: str
    source_identity: str
    time_myr: float
    source_geometry_bound: bool
    declared_normal_xyz: np.ndarray
    declared_offset_m: float
    parent_ndof: int
    quadrature_order: int
    selected_cohort_ids: tuple
    selected_cohort_indices: tuple
    patch_cohort_ids: np.ndarray
    patch_kinds: tuple
    patch_lambda_bounds: np.ndarray
    point_patch_indices: np.ndarray
    cohort_indices: np.ndarray
    cohort_ids: np.ndarray
    face_indices: np.ndarray
    owner_ids: tuple
    material_ids: tuple
    barycentric: np.ndarray
    radial_scale: np.ndarray
    mass_coordinate_q: np.ndarray
    current_position_m: np.ndarray
    reference_position_m: np.ndarray
    current_differentials_m: np.ndarray
    reference_differentials_m: np.ndarray
    current_volume_jacobian_m: np.ndarray
    reference_volume_jacobian_m: np.ndarray
    current_volume_determinant_m3: np.ndarray
    reference_volume_determinant_m3: np.ndarray
    current_to_reference_volume_ratio: np.ndarray
    current_normal_xyz: np.ndarray
    reference_normal_xyz: np.ndarray
    current_area_weights_m2: np.ndarray
    reference_area_weights_m2: np.ndarray
    parent_trace: np.ndarray
    paired_jump_trace: np.ndarray
    trace_dofs: np.ndarray
    binding_sha256: str
    format: str = FORMAT
    scope: str = SCOPE

    def __post_init__(self):
        integers = {"patch_cohort_ids", "point_patch_indices", "cohort_indices",
                    "cohort_ids", "face_indices", "trace_dofs"}
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, np.ndarray):
                object.__setattr__(self, field.name,
                    _immutable(value, np.int64 if field.name in integers else float))
        for name in ("section_records", "selected_cohort_ids", "selected_cohort_indices",
                     "patch_kinds", "owner_ids", "material_ids"):
            object.__setattr__(self, name, tuple(getattr(self, name)))

    @property
    def point_count(self):
        return len(self.cohort_ids)

    @property
    def patch_count(self):
        return len(self.patch_cohort_ids)

    @property
    def total_current_area_m2(self):
        return float(self.current_area_weights_m2.sum())

    @property
    def total_reference_area_m2(self):
        return float(self.reference_area_weights_m2.sum())

    @property
    def physical_creation_admitted(self):
        return False

    @property
    def physical_release_admitted(self):
        return False

    @property
    def dofs_created_or_released(self):
        return False


def _current_chart_orientation(section):
    """Check the whole cohort chart, including quadratic radial extrema.

    The determinant is affine in barycentric N and quadratic in lambda.
    Checking the three vertex quadratics over the complete interval therefore
    rules out an inverted/collapsed chart without relying on area samples.
    This local condition does not admit global contact or self intersection.
    """
    X, P, A = section.triangle_m, section.slope_P_m, section.displacement_A_m
    scale = max(float(np.max(np.abs(X))), float(np.max(np.abs(P))),
                float(np.max(np.abs(A))), 1.)
    V = X/scale+P/scale
    W = A/scale-P/scale
    dv1, dv2 = V[1]-V[0], V[2]-V[0]
    dw1, dw2 = W[1]-W[0], W[2]-W[0]
    c2 = np.cross(dv1, dv2)@V.T
    c1 = (np.cross(dv1, dw2)+np.cross(dw1, dv2))@V.T
    c0 = np.cross(dw1, dw2)@V.T
    lo, hi = section.lambda_bounds
    for i in range(3):
        checks = [lo, hi]
        if c2[i] != 0:
            stationary = -c1[i]/(2*c2[i])
            if lo < stationary < hi:
                checks.append(float(stationary))
        for lam in checks:
            value = (c2[i]*lam+c1[i])*lam+c0[i]
            bound = 64*np.finfo(float).eps*(abs(c2[i]*lam*lam)+abs(c1[i]*lam)+abs(c0[i]))
            if not math.isfinite(float(value)) or value <= bound:
                raise EmbeddedSurfaceRefusal("Current material chart is inverted, collapsed, or unresolved")


def _edge_endpoint(section, edge, lam):
    i, j = edge
    N = np.zeros(3)
    derivative = np.zeros(3)
    if i == j:
        N[i] = 1.
        return N, derivative
    slopes, offsets, offset = section.projection_slopes, section.projection_offsets, section.offset_m
    scale = max(float(np.max(np.abs(lam*slopes))),
                float(np.max(np.abs(offsets))), abs(offset), 1.)
    signed = lam*(slopes/scale)+(offsets/scale-offset/scale)
    si, sj = signed[i], signed[j]
    denominator = si-sj
    if denominator == 0:
        raise EmbeddedSurfaceRefusal("Surface edge intersection derivative is unresolved")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        t = si/denominator
        dt = ((si*(slopes[j]/scale)-sj*(slopes[i]/scale))/denominator)/denominator
    if not np.isfinite((t, dt)).all() or not -64*np.finfo(float).eps <= t <= 1+64*np.finfo(float).eps:
        raise EmbeddedSurfaceRefusal("Surface edge intersection derivative is unresolved")
    t = min(max(float(t), 0.), 1.)
    N[i], N[j] = 1-t, t
    derivative[i], derivative[j] = -dt, dt
    return N, derivative


def _metrics(section, N, lam, derivatives):
    """Metric of the frozen material pullback, with oriented area normals."""
    X, P, A = section.triangle_m, section.slope_P_m, section.displacement_A_m
    with np.errstate(over="ignore", invalid="ignore"):
        V = X+P
        Y = lam*V+(A-P)
        current = N@Y
        reference = lam*(N@X)
        if derivatives[2] == "segment":
            dNs, dNl = derivatives[:2]
            dc = np.array((dNs@Y, dNl@Y+N@V))
            dr = np.array((lam*(dNs@X), N@X+lam*(dNl@X)))
        else:
            # Derivatives are wrt the standard triangle's angular u,v.
            dc = np.array((Y[1]-Y[0], Y[2]-Y[0]))
            dr = lam*np.array((X[1]-X[0], X[2]-X[0]))
        cc, cr = np.cross(dc[0], dc[1]), np.cross(dr[0], dr[1])
        jc, jr = np.linalg.norm(cc), np.linalg.norm(cr)
        Jc = np.column_stack((Y[1]-Y[0], Y[2]-Y[0], N@V))
        Jr = np.column_stack((lam*(X[1]-X[0]), lam*(X[2]-X[0]), N@X))
        detc, detr = np.linalg.det(Jc), np.linalg.det(Jr)
    if not (np.isfinite(current).all() and np.isfinite(reference).all()
            and np.isfinite(dc).all() and np.isfinite(dr).all()
            and np.isfinite((jc, jr)).all() and jc > 0 and jr > 0):
        raise EmbeddedSurfaceRefusal("Embedded surface has nonfinite or collapsed area metric")
    nc, nr = cc/jc, cr/jr
    alignment = float(nc@section.normal_xyz)
    if abs(abs(alignment)-1.) > 2e-10:
        raise EmbeddedSurfaceRefusal("Differential surface normal disagrees with the declared current plane")
    if not (np.isfinite(Jc).all() and np.isfinite(Jr).all()
            and np.isfinite((detc, detr)).all() and detc > 0 and detr > 0):
        raise EmbeddedSurfaceRefusal("Embedded surface volume chart is inverted, collapsed, or nonfinite")
    ratio = float(detc/detr)
    if not math.isfinite(ratio) or ratio <= 0:
        raise EmbeddedSurfaceRefusal("Embedded current/reference volume ratio is unresolved")
    orientation = 1. if alignment > 0 else -1.
    return (current, reference, dc, dr, orientation*nc, orientation*nr, float(jc), float(jr),
            Jc, Jr, float(detc), float(detr), ratio)


def _trace(N, lam):
    result = np.zeros((3, 18))
    for i in range(3):
        result[:, 6*i:6*i+3] = N[i]*np.eye(3)
        result[:, 6*i+3:6*i+6] = (lam-1)*N[i]*np.eye(3)
    return result


def _prepare(records, *, sections, parent_ndof, face_dofs, order, source_geometry_bound,
             declared_normal_xyz, declared_offset_m):
    order = _order(order)
    nodes, weights = np.polynomial.legendre.leggauss(order)
    gauss, gauss_weights = .5*(nodes+1), .5*weights
    names = ("point_patch_indices", "cohort_indices", "cohort_ids", "face_indices",
             "barycentric", "radial_scale", "mass_coordinate_q", "current_position_m",
             "reference_position_m", "current_differentials_m", "reference_differentials_m",
             "current_volume_jacobian_m", "reference_volume_jacobian_m",
             "current_volume_determinant_m3", "reference_volume_determinant_m3",
             "current_to_reference_volume_ratio",
             "current_normal_xyz", "reference_normal_xyz", "current_area_weights_m2",
             "reference_area_weights_m2", "parent_trace", "paired_jump_trace", "trace_dofs")
    result = {name: [] for name in names}
    owners, materials, patch_ids, patch_kinds, patch_bounds = [], [], [], [], []

    def add(section, patch, N, lam, derivatives, parameter_weight):
        current, reference, dc, dr, nc, nr, jc, jr, Jc, Jr, detc, detr, ratio = _metrics(
            section, N, lam, derivatives)
        T = _trace(N, lam)
        values = dict(point_patch_indices=patch, cohort_indices=section.cohort_index,
            cohort_ids=section.cohort_id, face_indices=section.face_index,
            barycentric=N, radial_scale=lam, mass_coordinate_q=_q_at_lambda(section, lam),
            current_position_m=current, reference_position_m=reference,
            current_differentials_m=dc, reference_differentials_m=dr,
            current_volume_jacobian_m=Jc, reference_volume_jacobian_m=Jr,
            current_volume_determinant_m3=detc, reference_volume_determinant_m3=detr,
            current_to_reference_volume_ratio=ratio,
            current_normal_xyz=nc, reference_normal_xyz=nr,
            current_area_weights_m2=parameter_weight*jc,
            reference_area_weights_m2=parameter_weight*jr,
            parent_trace=T, paired_jump_trace=np.concatenate((-T, T), axis=1),
            trace_dofs=face_dofs[section.face_index])
        if not (math.isfinite(values["current_area_weights_m2"])
                and math.isfinite(values["reference_area_weights_m2"])
                and values["current_area_weights_m2"] > 0
                and values["reference_area_weights_m2"] > 0):
            raise EmbeddedSurfaceRefusal("Embedded surface quadrature weights are nonpositive or nonfinite")
        for name in names:
            result[name].append(values[name])
        owners.append(section.owner_id)
        materials.append(section.material_id)

    for section in records:
        _validate_cohort_section(section)
        _current_chart_orientation(section)
        if not section.interior_cut:
            # This module represents relative-interior in-cohort sheets. A
            # shared cohort/face boundary needs a different paired support;
            # summing closed-domain boundary sheets would count it twice.
            continue
        for interval in section.intervals:
            if interval.slice_kind != "segment":
                continue  # A vertex curve has no positive area support.
            lo, hi = interval.lambda_bounds
            patch = len(patch_ids)
            patch_ids.append(section.cohort_id)
            patch_kinds.append("lambda_segment")
            patch_bounds.append((lo, hi))
            for radial, rw in zip(gauss, gauss_weights):
                lam = lo+(hi-lo)*radial
                N0, D0 = _edge_endpoint(section, interval.edge_pairs[0], lam)
                N1, D1 = _edge_endpoint(section, interval.edge_pairs[1], lam)
                for s, sw in zip(gauss, gauss_weights):
                    N = (1-s)*N0+s*N1
                    dNl = (1-s)*D0+s*D1
                    add(section, patch, N, lam, (N1-N0, dNl, "segment"),
                        float((hi-lo)*rw*sw))
        for constant in section.constant_lambda_sections:
            if constant.location != "interior":
                continue
            patch = len(patch_ids)
            patch_ids.append(section.cohort_id)
            patch_kinds.append("constant_lambda_triangle")
            lam = constant.lambda_value
            patch_bounds.append((lam, lam))
            # Duffy's square-to-triangle map: u=r,v=(1-r)t, det=1-r.
            for r, rw in zip(gauss, gauss_weights):
                for t, tw in zip(gauss, gauss_weights):
                    u, v = r, (1-r)*t
                    N = np.array((1-u-v, u, v))
                    add(section, patch, N, lam, (None, None, "triangle"),
                        float(rw*tw*(1-r)))

    shapes = {"barycentric": (3,), "current_position_m": (3,), "reference_position_m": (3,),
        "current_differentials_m": (2, 3), "reference_differentials_m": (2, 3),
        "current_volume_jacobian_m": (3, 3), "reference_volume_jacobian_m": (3, 3),
        "current_normal_xyz": (3,), "reference_normal_xyz": (3,),
        "parent_trace": (3, 18), "paired_jump_trace": (3, 36), "trace_dofs": (18,)}
    integers = {"point_patch_indices", "cohort_indices", "cohort_ids", "face_indices", "trace_dofs"}
    count = len(owners)
    arrays = {name: np.asarray(value, np.int64 if name in integers else float).reshape(
        (count,)+shapes.get(name, ())) for name, value in result.items()}
    value = GenesisVolumeEmbeddedSurface(material_sections=sections, section_records=tuple(records),
        material_sections_binding_sha256=(sections.binding_sha256 if sections is not None
            else _binding(tuple(records))),
        source_checkpoint_sha256=sections.source_checkpoint_sha256 if sections is not None else "",
        source_identity=sections.source_identity if sections is not None else "computational_cohort",
        time_myr=sections.time_myr if sections is not None else 0.,
        source_geometry_bound=source_geometry_bound,
        declared_normal_xyz=np.asarray(declared_normal_xyz, float),
        declared_offset_m=float(declared_offset_m), parent_ndof=parent_ndof,
        quadrature_order=order, selected_cohort_ids=tuple(c.cohort_id for c in records),
        selected_cohort_indices=tuple(c.cohort_index for c in records),
        patch_cohort_ids=np.asarray(patch_ids, np.int64), patch_kinds=tuple(patch_kinds),
        patch_lambda_bounds=np.asarray(patch_bounds, float).reshape(-1, 2),
        owner_ids=tuple(owners), material_ids=tuple(materials), binding_sha256="", **arrays)
    return replace(value, binding_sha256=_binding(value))


def prepare_volume_embedded_surface(geometry, displacement_m, normal_xyz, offset_m, *,
                                    face_indices=None, cohort_ids=None, quadrature_order=8):
    """Quadrature of an explicit plane with source geometry and full A/P.

    Plane, kinematics and optional material subset remain computational input.
    ``cohort_ids`` selects original material; it never creates a new cohort or
    owner and never changes the source geometry/history.  Empty subsets are
    supported and produce a typed empty surface with the full source binding.
    """
    _numeric(displacement_m, "Embedded parent displacement")
    declared_normal = _numeric(normal_xyz, "Embedded plane normal", (3,))
    if face_indices is not None:
        _numeric(face_indices, "Embedded face indices", integer=True)
    try:
        declared_offset = _number(offset_m, "Embedded plane offset")
        sections = prepare_volume_material_sections(geometry, displacement_m, normal_xyz,
            offset_m, face_indices=face_indices)
    except (TypeError, ValueError, OverflowError) as error:
        raise EmbeddedSurfaceRefusal(str(error)) from error
    records = sections.cohorts
    if cohort_ids is not None:
        ids = _numeric(cohort_ids, "Embedded surface cohort IDs", integer=True)
        if ids.ndim != 1 or len(np.unique(ids)) != len(ids):
            raise EmbeddedSurfaceRefusal("Embedded surface cohort IDs must be a unique one-dimensional sequence")
        available = {c.cohort_id for c in records}
        if any(int(i) not in available for i in ids):
            raise EmbeddedSurfaceRefusal("Embedded surface cohort ID is outside the selected original material")
        wanted = set(map(int, ids))
        records = tuple(c for c in records if c.cohort_id in wanted)
    faces = np.asarray(geometry.faces, np.int64)
    face_dofs = {i: (6*face[:, None]+np.arange(6)).reshape(18)
                 for i, face in enumerate(faces)}
    return _prepare(records, sections=sections, parent_ndof=6*len(geometry.vertices_m),
        face_dofs=face_dofs, order=quadrature_order, source_geometry_bound=True,
        declared_normal_xyz=declared_normal, declared_offset_m=declared_offset)


def prepare_cohort_embedded_surface(section, *, quadrature_order=8):
    """Standalone computational differential/trace fixture, no planet binding."""
    if type(section) is not CohortMaterialSection:
        raise EmbeddedSurfaceRefusal("A prepared material cohort section is required")
    return _prepare((section,), sections=None, parent_ndof=18,
        face_dofs={section.face_index: np.arange(18)}, order=quadrature_order,
        source_geometry_bound=False, declared_normal_xyz=section.normal_xyz,
        declared_offset_m=section.offset_m)


def validate_volume_embedded_surface(surface):
    if (type(surface) is not GenesisVolumeEmbeddedSurface or surface.format != FORMAT
            or surface.scope != SCOPE or surface.binding_sha256 != _binding(surface)):
        raise EmbeddedSurfaceRefusal("Embedded surface differs from its immutable material/support binding")
    if surface.material_sections is not None:
        volume_material_sections_to_dict(surface.material_sections)
        if surface.material_sections.binding_sha256 != surface.material_sections_binding_sha256:
            raise EmbeddedSurfaceRefusal("Embedded surface material section binding differs")
    for section in surface.section_records:
        _validate_cohort_section(section)
    return surface


def evaluate_embedded_surface_trace(surface, parent_increment_m):
    """World trace of a NEW parent increment, not a second strain-memory update."""
    validate_volume_embedded_surface(surface)
    u = _numeric(parent_increment_m, "Embedded parent increment", (surface.parent_ndof,))
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.einsum("pai,pi->pa", surface.parent_trace, u[surface.trace_dofs])
    if not np.isfinite(result).all():
        raise EmbeddedSurfaceRefusal("Embedded parent trace overflowed")
    return _immutable(result)


def evaluate_embedded_surface_positions(surface, parent_displacement_m):
    """Advect the frozen N/q support with supplied absolute parent A/P.

    The born/current normals and both recorded area metrics remain the birth
    snapshot.  This function only calculates positions; it never intersects
    a world plane again or transports stress, normals, or integration area.
    """
    trace = evaluate_embedded_surface_trace(surface, parent_displacement_m)
    with np.errstate(over="ignore", invalid="ignore"):
        result = surface.reference_position_m+trace
    if not np.isfinite(result).all():
        raise EmbeddedSurfaceRefusal("Embedded material support advection overflowed")
    return _immutable(result)


def evaluate_embedded_surface_jump(surface, paired_parent_increments_m):
    """Virtual side1-minus-side0 trace; paired fields are not released DOFs."""
    validate_volume_embedded_surface(surface)
    u = _numeric(paired_parent_increments_m, "Paired embedded parent increments",
                 (2, surface.parent_ndof))
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.einsum("pai,pi->pa", surface.parent_trace,
            u[1, surface.trace_dofs]-u[0, surface.trace_dofs])
    if not np.isfinite(result).all():
        raise EmbeddedSurfaceRefusal("Embedded virtual jump overflowed")
    return _immutable(result)


def _area(surface, convention):
    if type(convention) is not str:
        raise EmbeddedSurfaceRefusal("Embedded area convention must be explicitly current or reference")
    if convention == "current":
        return surface.current_area_weights_m2
    if convention == "reference":
        return surface.reference_area_weights_m2
    raise EmbeddedSurfaceRefusal("Embedded area convention must be explicitly current or reference")


def integrate_embedded_surface_traction(surface, traction_pa, *, area_convention):
    """Discrete paired force dual to the trace under the specified metric.

    The supplied traction is conjugate to world virtual jump and the selected
    current/reference area.  The function does not infer finite stress,
    prestress replacement, material funding, or a physical bank trajectory.
    """
    validate_volume_embedded_surface(surface)
    traction = _numeric(traction_pa, "Embedded surface traction", (surface.point_count, 3))
    weights = _area(surface, area_convention)
    with np.errstate(over="ignore", invalid="ignore"):
        local = np.einsum("pai,pa,p->pi", surface.parent_trace, traction, weights)
        force = np.zeros(surface.parent_ndof)
        np.add.at(force, surface.trace_dofs.ravel(), local.ravel())
    if not np.isfinite(force).all():
        raise EmbeddedSurfaceRefusal("Embedded surface integrated force overflowed")
    return _immutable(np.stack((-force, force)))


def integrate_embedded_surface_virtual_work(surface, traction_pa, paired_parent_increments_m, *,
                                             area_convention):
    """Quadrature work of a supplied traction and paired virtual increment."""
    jump = evaluate_embedded_surface_jump(surface, paired_parent_increments_m)
    traction = _numeric(traction_pa, "Embedded surface traction", (surface.point_count, 3))
    with np.errstate(over="ignore", invalid="ignore"):
        work = float(np.einsum("pa,pa,p->", traction, jump, _area(surface, area_convention)))
    if not math.isfinite(work):
        raise EmbeddedSurfaceRefusal("Embedded surface virtual work overflowed")
    return work


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return [_plain(x) for x in value]
    return value


def volume_embedded_surface_to_dict(surface):
    validate_volume_embedded_surface(surface)
    body = {field.name: _plain(getattr(surface, field.name)) for field in fields(surface)
            if field.name not in ("material_sections", "section_records", "binding_sha256")}
    body.update(quadrature_rule="tensor_Gauss_on_analytic_lambda_pieces_and_Duffy_triangle",
        original_parent_modes="A_xyz_and_P_xyz_metres_per_original_vertex",
        paired_trace_scope="two_virtual_parent_fields_not_created_or_released_mesh_DOFs",
        material_support_semantics="birth_pullback_N_q_frozen_not_world_plane_reintersection",
        declared_plane_input_semantics="literal_numeric_constructor_coefficients_before_normalization",
        area_conventions="current_and_reference_separate_interface_counted_once",
        support_scope="relative_interior_cohort_sheets_boundary_only_support_has_zero_area",
        numerical_integration_scope="approximate_geometry_integrals_exact_discrete_trace_work_at_declared_order",
        geometry_integrals_exact=False,
        recorded_metrics_semantics="fixed_declared_plane_pullback_snapshot_not_transported_after_follow",
        total_current_area_m2=surface.total_current_area_m2,
        total_reference_area_m2=surface.total_reference_area_m2,
        bulk_energy_force_identity_claimed=False, finite_stress_pushforward_claimed=False,
        physical_creation_admitted=False, physical_release_admitted=False,
        physical_contact_admitted=False, dofs_created_or_released=False,
        material_memory_changed=False, material_history_reinitialized=False,
        clock_advanced=False, creation_work_debited_from_Genesis_j=0.)
    return {**body, "embedded_surface_sha256": hashlib.sha256(_canonical(body)).hexdigest()}


def volume_embedded_surface_from_dict(raw, *, geometry, displacement_m, normal_xyz, offset_m,
                                      face_indices=None, cohort_ids=None, quadrature_order=8):
    """Rebuild every metric and trace against independently supplied inputs."""
    expected = prepare_volume_embedded_surface(geometry, displacement_m, normal_xyz, offset_m,
        face_indices=face_indices, cohort_ids=cohort_ids, quadrature_order=quadrature_order)
    try:
        equal = _canonical(raw) == _canonical(volume_embedded_surface_to_dict(expected))
    except (TypeError, ValueError, OverflowError) as error:
        raise EmbeddedSurfaceRefusal("Embedded surface archive is not a finite canonical packet") from error
    if not equal:
        raise EmbeddedSurfaceRefusal("Embedded surface differs from complete material/metric/trace reconstruction")
    return expected
