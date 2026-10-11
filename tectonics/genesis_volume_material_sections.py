"""Declared world-plane sections of the full current Genesis material chart.

The chart is x(N, lambda) = N @ [lambda*(X+P)+(A-P)].  A supplied plane is
pulled back through this chart, including all six Cartesian nodal amplitudes.
It is not replaced by a radial copy of its top trace or by a mesh edge.

This geometric support does not choose a plane offset, activate a fault, split
memory, introduce/release DOFs, or debit work.  Source material is validated;
the supplied kinematics and plane remain explicit computational inputs.  An
accepted-state caller must separately bind those inputs to its checkpoint.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import math
from numbers import Real

import numpy as np

from .genesis_volume_checkpoint import _canonical
from .genesis_volume_geometry import (
    ConicalCohortGeometry, _chart, validate_genesis_volume_geometry,
)
from .genesis_volume_operator import _immutable
from .genesis_volume_banks import _hash_value


FORMAT = "genesis-declared-current-volume-material-sections-0.1"
SCOPE = "explicit_world_plane_support_no_localization_or_physical_admission"


def _real_array(value, shape, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in "ifu":
        raise ValueError(f"{name} requires real numeric values")
    result = np.asarray(raw, float)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} has invalid shape or nonfinite values")
    return result


def _number(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _plane(normal, offset):
    """Normalize both coefficients, retaining the supplied oriented plane."""
    normal = _real_array(normal, (3,), "Plane normal")
    offset = _number(offset, "Plane offset")
    scale = float(np.max(np.abs(normal)))
    if scale == 0:
        raise ValueError("Plane normal must be nonzero")
    scaled = normal/scale
    magnitude = float(np.linalg.norm(scaled))
    result_offset = (offset/scale)/magnitude
    if not math.isfinite(result_offset):
        raise ValueError("Normalized plane offset is not finite")
    return _immutable(scaled/magnitude), result_offset


@dataclass(frozen=True, slots=True)
class MaterialSectionInterval:
    lambda_bounds: tuple
    edge_pairs: tuple
    slice_kind: str


@dataclass(frozen=True, slots=True)
class ConstantLambdaMaterialSection:
    lambda_value: float
    location: str


@dataclass(frozen=True, slots=True)
class CohortMaterialSection:
    cohort_index: int
    cohort_id: int
    face_index: int
    owner_id: str
    material_id: str
    mass_kg: float
    volume_m3: float
    q_bounds: tuple
    lambda_bounds: tuple
    conical_mass_coefficient: float
    triangle_m: np.ndarray
    displacement_A_m: np.ndarray
    slope_P_m: np.ndarray
    normal_xyz: np.ndarray
    offset_m: float
    projection_slopes: np.ndarray
    projection_offsets: np.ndarray
    lambda_breakpoints: tuple
    intervals: tuple
    constant_lambda_sections: tuple
    interior_cut: bool
    status: str
    binding_sha256: str


@dataclass(frozen=True, slots=True)
class MaterialSectionPoint:
    barycentric: np.ndarray
    lambda_value: float
    q: float
    xyz_m: np.ndarray
    plane_residual_m: float


@dataclass(frozen=True, slots=True)
class VolumeMaterialSections:
    geometry_binding_sha256: str
    source_checkpoint_sha256: str
    source_identity: str
    time_myr: float
    displacement_sha256: str
    normal_xyz: np.ndarray
    offset_m: float
    face_indices: tuple
    cohorts: tuple
    binding_sha256: str
    format: str = FORMAT
    scope: str = SCOPE

    @property
    def calculated_location_selected(self):
        return False

    @property
    def physical_creation_admitted(self):
        return False

    @property
    def physical_release_admitted(self):
        return False


def _binding(value):
    digest = hashlib.sha256()
    _hash_value(digest, value)
    return digest.hexdigest()


def _scaled_signed(section, lam):
    radial = lam*section.projection_slopes
    scale = max(float(np.max(np.abs(radial))),
                float(np.max(np.abs(section.projection_offsets))), abs(section.offset_m), 1.)
    terms = (radial/scale, section.projection_offsets/scale, section.offset_m/scale)
    raw = terms[0]+terms[1]-terms[2]
    # Only arithmetic resolution is used to identify a represented endpoint
    # or coincident edge.  This is not a physical contact/localization width.
    bound = 16*np.finfo(float).eps*np.maximum(
        np.abs(terms[0])+np.abs(terms[1])+abs(terms[2]), 1/scale)
    return raw, bound


def _signed(section, lam):
    # A COMMON scale preserves edge-root ratios and avoids overflow both in
    # signed subtraction and in its roundoff bound for finite large inputs.
    raw, bound = _scaled_signed(section, lam)
    return np.where(np.abs(raw) <= bound, 0., raw)


def _slice(section, lam):
    """Barycentric intersection vertices, in canonical vertex/edge order."""
    values = _signed(section, lam)
    if np.all(values == 0):
        return np.eye(3), ((0, 0), (1, 1), (2, 2)), "triangle"
    points, edges = [], []
    for vertex in np.flatnonzero(values == 0):
        point = np.zeros(3)
        point[vertex] = 1.
        points.append(point)
        edges.append((int(vertex), int(vertex)))
    for i, j in ((0, 1), (0, 2), (1, 2)):
        if (values[i] < 0 < values[j]) or (values[j] < 0 < values[i]):
            # A convex ratio avoids overflow in subtracting large opposite
            # signed values; the formula remains the analytic edge root.
            scale = max(abs(values[i]), abs(values[j]))
            t = abs(values[i])/scale/(abs(values[i])/scale+abs(values[j])/scale)
            point = np.zeros(3)
            point[i], point[j] = 1-t, t
            points.append(point)
            edges.append((i, j))
    if not points:
        return np.empty((0, 3)), (), "empty"
    if len(points) > 2:
        raise ValueError("Material plane slice has unresolved intersection multiplicity")
    return np.asarray(points), tuple(edges), "point" if len(points) == 1 else "segment"


def _q_at_lambda(section, lam):
    lo, hi = section.lambda_bounds
    if lam == lo:
        return section.q_bounds[1]
    if lam == hi:
        return section.q_bounds[0]
    result = -math.expm1(3*math.log(lam))/section.conical_mass_coefficient
    qlo, qhi = section.q_bounds
    resolution = 64*np.finfo(float).eps/section.conical_mass_coefficient
    if not qlo-resolution <= result <= qhi+resolution:
        raise ValueError("Material section depth differs from its cohort q chart")
    return min(max(result, qlo), qhi)


def prepare_cohort_material_section(cohort, nodal_displacement_m, normal_xyz, offset_m,
                                    *, conical_mass_coefficient, cohort_index=0):
    """Prepare one explicitly declared geometric cohort section.

    This low-level entry point also supports isolated computational geometry
    tests.  It does not validate the cohort against a planetary source; use
    ``prepare_volume_material_sections`` for that material binding.
    """
    from dataclasses import replace
    if type(cohort) is not ConicalCohortGeometry:
        raise ValueError("Material section requires a conical cohort geometry")
    if isinstance(cohort_index, (bool, np.bool_)) or not isinstance(cohort_index, (int, np.integer)) or cohort_index < 0:
        raise ValueError("Material cohort index must be a nonnegative integer")
    X = _real_array(cohort.top_triangle_m, (3, 3), "Cohort reference triangle")
    displacement = _real_array(nodal_displacement_m, (3, 6), "Cohort nodal amplitudes")
    A, P = displacement[:, :3], displacement[:, 3:]
    normal, offset = _plane(normal_xyz, offset_m)
    coefficient = _number(conical_mass_coefficient, "Conical mass coefficient")
    a, b = (_number(cohort.outer_scale, "Cohort outer scale"),
            _number(cohort.inner_scale, "Cohort inner scale"))
    qlo, qhi = (_number(cohort.q_lo, "Cohort lower q"),
                _number(cohort.live_q_hi, "Cohort upper q"))
    if not (coefficient > 0 and 0 < b < a <= 1 and 0 <= qlo < qhi <= 1):
        raise ValueError("Material section requires positive live conical support")
    if coefficient*qhi >= 1:
        raise ValueError("Cohort q chart reaches or crosses the conical centre")
    expected = np.exp(np.log1p(-coefficient*np.array((qlo, qhi)))/3)
    if not np.isfinite(expected).all() or not np.allclose((a, b), expected, rtol=4e-14, atol=0):
        raise ValueError("Cohort lambda boundaries differ from its mass-q chart")
    if (not isinstance(cohort.owner_id, str) or not cohort.owner_id
            or not isinstance(cohort.material_id, str) or not cohort.material_id):
        raise ValueError("Material section requires original owner and material IDs")
    with np.errstate(over="ignore", invalid="ignore"):
        slopes, offsets = (X+P)@normal, (A-P)@normal
    if not np.isfinite(slopes).all() or not np.isfinite(offsets).all():
        raise ValueError("Material plane projections are not finite")
    with np.errstate(over="ignore", invalid="ignore"):
        endpoints = np.concatenate((b*slopes+offsets, a*slopes+offsets))
    if not np.isfinite(endpoints).all():
        raise ValueError("Material plane endpoint projections are not finite")
    section = CohortMaterialSection(int(cohort_index), int(cohort.id), int(cohort.face_index),
        cohort.owner_id, cohort.material_id, _number(cohort.mass_kg, "Cohort mass"),
        _number(cohort.volume_m3, "Cohort volume"), (qlo, qhi), (b, a), coefficient,
        *map(_immutable, (X, A, P)), normal, offset,
        _immutable(slopes), _immutable(offsets), (), (), (),
        bool(float(endpoints.min()) < offset < float(endpoints.max())), "", "")
    if section.mass_kg <= 0 or section.volume_m3 <= 0:
        raise ValueError("Material section requires positive cohort mass and volume")
    breaks = {b, a}
    endpoint_signs = {b: _signed(section, b), a: _signed(section, a)}
    for vertex, (slope, intercept) in enumerate(zip(slopes, offsets)):
        if slope != 0:
            root = (offset-float(intercept))/float(slope)
            if math.isfinite(root):
                # A literal boundary and its algebraically reconstructed root
                # can differ by one ULP.  Canonicalize only their arithmetic
                # equivalence, checked against this same vertex's represented
                # endpoint zero; this is not a physical depth tolerance.
                for endpoint in (b, a):
                    ulp_bound = 4*max(math.ulp(float(root)), math.ulp(float(endpoint)))
                    if endpoint_signs[endpoint][vertex] == 0 and abs(root-endpoint) <= ulp_bound:
                        root = endpoint
                        break
                if b < root < a:
                    breaks.add(float(root))
    breaks = tuple(sorted(breaks))
    intervals = []
    for lower, upper in zip(breaks, breaks[1:]):
        midpoint = lower+.5*(upper-lower)
        _, edges, kind = _slice(section, midpoint)
        if kind == "triangle":
            if np.all(slopes == 0) and np.all(offsets == offset):
                raise ValueError("Declared current material volume is coplanar; section is unresolved")
            raise ValueError("Material plane interval is below arithmetic resolution; section is unresolved")
        if kind in ("point", "segment"):
            intervals.append(MaterialSectionInterval((lower, upper), edges, kind))
    constant = []
    for lam in breaks:
        _, _, kind = _slice(section, lam)
        if kind == "triangle":
            location = "interior" if b < lam < a else "inner_boundary" if lam == b else "outer_boundary"
            constant.append(ConstantLambdaMaterialSection(lam, location))
    boundary_kind = [_slice(section, lam)[2] for lam in breaks]
    if intervals or constant:
        status = "interior_cut" if section.interior_cut else "boundary_support"
    elif "point" in boundary_kind or "segment" in boundary_kind:
        status = "boundary_touch"
    else:
        status = "no_intersection"
    section = replace(section, lambda_breakpoints=breaks, intervals=tuple(intervals),
        constant_lambda_sections=tuple(constant), status=status)
    return replace(section, binding_sha256=_binding(section))


def _validate_cohort_section(section):
    if type(section) is not CohortMaterialSection or section.binding_sha256 != _binding(section):
        raise ValueError("Material cohort section differs from its immutable geometric binding")
    return section


def evaluate_cohort_material_section(section, lambda_value, *, fraction=.5, barycentric=None):
    """Evaluate a point on the declared plane, retaining its original q.

    ``fraction`` selects along a fixed-lambda segment.  A constant-lambda full
    triangle instead requires an explicit barycentric point.  At an isolated
    touching vertex the unique point is returned for every valid fraction.
    """
    _validate_cohort_section(section)
    lam = _number(lambda_value, "Section lambda")
    if not section.lambda_bounds[0] <= lam <= section.lambda_bounds[1]:
        raise ValueError("Section lambda is outside the live cohort")
    fraction = _number(fraction, "Section fraction")
    if not 0 <= fraction <= 1:
        raise ValueError("Section fraction must lie in [0, 1]")
    vertices, _, kind = _slice(section, lam)
    if barycentric is not None:
        N = _real_array(barycentric, (3,), "Section barycentric point")
        if np.any(N < 0) or not math.isclose(float(N.sum()), 1., rel_tol=0, abs_tol=8*np.finfo(float).eps):
            raise ValueError("Section barycentric point is outside the material triangle")
        signed, bounds = _scaled_signed(section, lam)
        if abs(float(N@signed)) > 2*float(np.max(bounds)):
            raise ValueError("Barycentric point is not on the declared material plane")
    elif kind == "triangle":
        raise ValueError("Constant-lambda triangle section requires explicit barycentric coordinates")
    elif kind == "empty":
        raise ValueError("Declared plane has no section at this material depth")
    elif kind == "point":
        N = vertices[0]
    else:
        N = (1-fraction)*vertices[0]+fraction*vertices[1]
    xyz = N@(lam*(section.triangle_m+section.slope_P_m)
              +(section.displacement_A_m-section.slope_P_m))
    if not np.isfinite(xyz).all():
        raise ValueError("Material section point is not finite")
    return MaterialSectionPoint(_immutable(N), lam, _q_at_lambda(section, lam),
        _immutable(xyz), float(section.normal_xyz@xyz-section.offset_m))


def prepare_volume_material_sections(geometry, displacement_m, normal_xyz, offset_m,
                                     *, face_indices=None):
    """Bind declared plane support to all selected original live cohorts.

    Every selected cohort is retained, including empty/touching sections.
    Neither an omitted cohort nor a plane miss can become a material birth.
    Material geometry is replay-validated; amplitudes are explicit inputs.
    """
    from dataclasses import replace
    validate_genesis_volume_geometry(geometry)
    vertex_count, face_count = len(geometry.vertices_m), len(geometry.faces)
    displacement = _real_array(displacement_m, (6*vertex_count,), "Volume nodal amplitudes")
    normal, offset = _plane(normal_xyz, offset_m)
    if face_indices is None:
        selected = tuple(range(face_count))
    else:
        raw = np.asarray(face_indices)
        if raw.ndim != 1 or (raw.size and raw.dtype.kind not in "iu"):
            raise ValueError("Selected face indices must be a one-dimensional integer sequence")
        selected = tuple(int(index) for index in raw)
        if (len(set(selected)) != len(selected) or any(index < 0 or index >= face_count for index in selected)):
            raise ValueError("Selected face indices are duplicate or outside the original mesh")
        selected = tuple(sorted(selected))
    rows = displacement.reshape(vertex_count, 6)
    face_lookup = set(selected)
    cohorts = []
    for index, cohort in enumerate(geometry.cohorts):
        if cohort.face_index not in face_lookup:
            continue
        face = np.asarray(geometry.faces[cohort.face_index], int)
        triangle = np.asarray(geometry.vertices_m)[face]
        _, _, coefficient = _chart(triangle, geometry.column_mass_kg[cohort.face_index],
                                     geometry.density_kg_m3)
        cohorts.append(prepare_cohort_material_section(cohort, rows[face], normal, offset,
            conical_mass_coefficient=coefficient, cohort_index=index))
    digest = hashlib.sha256()
    _hash_value(digest, geometry)
    result = VolumeMaterialSections(digest.hexdigest(), geometry.source_checkpoint_sha256,
        geometry.source_identity, geometry.time_myr,
        hashlib.sha256(displacement.tobytes()).hexdigest(), normal, offset, selected,
        tuple(cohorts), "")
    return replace(result, binding_sha256=_binding(result))


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "__dataclass_fields__"):
        return {field.name: _plain(getattr(value, field.name)) for field in fields(value)
                if field.name != "binding_sha256"}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


def volume_material_sections_to_dict(sections):
    if (type(sections) is not VolumeMaterialSections or sections.format != FORMAT
            or sections.scope != SCOPE or sections.binding_sha256 != _binding(sections)):
        raise ValueError("Volume material sections differ from their geometric binding")
    for cohort in sections.cohorts:
        _validate_cohort_section(cohort)
    body = _plain(sections)
    body.update(calculated_location_selected=False, source_creation_energy_law=None,
        physical_creation_admitted=False, physical_release_admitted=False,
        physical_contact_admitted=False, material_memory_changed=False,
        dofs_created_or_released=False, clock_advanced=False,
        creation_work_debited_from_Genesis_j=0.,
        integration_scope="analytic_slice_geometry_no_area_or_volume_quadrature_claim")
    body["sections_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
    return body


def volume_material_sections_from_dict(raw, *, geometry, displacement_m,
                                      normal_xyz, offset_m, face_indices=None):
    """Recompute an archive against independently supplied support inputs."""
    expected = prepare_volume_material_sections(geometry, displacement_m, normal_xyz,
        offset_m, face_indices=face_indices)
    try:
        equal = _canonical(raw) == _canonical(volume_material_sections_to_dict(expected))
    except (ValueError, TypeError) as error:
        raise ValueError("Volume material section archive is not a finite canonical packet") from error
    if not equal:
        raise ValueError("Volume material sections differ from full material/support reconstruction")
    return expected
