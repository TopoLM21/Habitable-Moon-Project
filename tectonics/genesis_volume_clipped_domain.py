"""Resolved finite-ball/cohort/plane-side integration domains.

The original material chart is sliced at fixed lambda.  Each slice is an
affine triangle, its intersection with the ball is an actual disk, and Gamma
clips that triangle by a linear half-plane.  Polynomial radial topology events
and analytic disk/polygon ray intervals replace sampling a ball and discarding
points outside the cohort.  Positive quadrature approximates integrals on the
resolved domain: this module does not certify their convergence or admit a
physical seed, release, history, or work debit.

All source X+P and A-P sums are performed once in the original float64 schema;
subsequent geometry uses longdouble and a centered local radial coordinate.
No polyhedral replacement of the original bilinear material chart is made.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math
from numbers import Integral, Real

import numpy as np

from .genesis_volume_checkpoint import _canonical
from .genesis_volume_embedded_surface import _current_chart_orientation
from .genesis_volume_finite_seed import _padd, _pdot, _pcross, _pmul, _roots
from .genesis_volume_material_sections import _validate_cohort_section


FORMAT = "genesis-finite-ball-original-cohort-side-domain-0.1"
SCOPE = "domain_geometry_resolved_integral_convergence_required"
LD = np.longdouble


class ClippedDomainRefusal(ValueError):
    """Declared domain geometry or a bounded quadrature is unresolved."""


def _plain(value):
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return float(value) if value.dtype.kind == "f" else value.item()
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    return value


def _readonly(value, shape=None, dtype=float):
    result = np.array(value, dtype=dtype, copy=True)
    if shape is not None:
        result = result.reshape(shape)
    result.setflags(write=False)
    return result


def _live_numeric_digest(arrays):
    """Seal exact numeric operands, never platform longdouble padding bytes.

    NumPy's integer-ratio conversion retains every represented significand
    bit.  The separately archived float64 table has a different, explicit
    arithmetic scope; it is not used to reconstruct live longdouble operands.
    """
    digest = hashlib.sha256()
    digest.update(b"genesis-clipped-domain-exact-logical-rational-operands-0.1\n")
    for name, values in arrays:
        array = np.asarray(values)
        digest.update(_canonical(dict(name=name, shape=list(array.shape))))
        digest.update(b"\n")
        for value in array.flat:
            if not np.isfinite(value):
                raise ClippedDomainRefusal("Live clipped geometry operand is nonfinite")
            numerator, denominator = value.as_integer_ratio()
            digest.update(f"{numerator}/{denominator}\n".encode("ascii"))
    return digest.hexdigest()


def _live_arrays(domain):
    return [(name, getattr(domain, name)) for name in (
        "barycentric", "radial_scale", "reference_volume_weights_m3",
        "birth_volume_weights_m3", "side", "centered_birth_offsets_m")]


def _source_record(section):
    return _plain(dict(triangle_m=section.triangle_m,
        displacement_A_m=section.displacement_A_m, slope_P_m=section.slope_P_m,
        lambda_bounds=section.lambda_bounds, normal_xyz=section.normal_xyz, offset_m=section.offset_m,
        cohort_id=section.cohort_id, face_index=section.face_index,
        owner_id=section.owner_id, material_id=section.material_id,
        mass_kg=section.mass_kg, volume_m3=section.volume_m3,
        q_bounds=section.q_bounds, conical_mass_coefficient=section.conical_mass_coefficient))


def _integer(value, name, lower, upper):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or not lower <= value <= upper:
        raise ClippedDomainRefusal(f"{name} must be an integer in [{lower}, {upper}]")
    return int(value)


def _finite_real(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ClippedDomainRefusal(f"{name} must be a finite real")
    result = float(value)
    if not math.isfinite(result):
        raise ClippedDomainRefusal(f"{name} must be a finite real")
    return result


def _cross2(a, b):
    return a[0]*b[1]-a[1]*b[0]


def _poly_scale_vector(polynomial, vector):
    return np.asarray(polynomial, LD)[:, None]*np.asarray(vector, LD)[None, :]


def _poly_vector_add(a, b):
    result = np.zeros((max(len(a), len(b)), 3), LD)
    result[:len(a)] += a
    result[:len(b)] += b
    return result


def _event_polynomials(C, D, center, normal, offset, radius, midpoint, halfwidth):
    """A complete superset of slice polygon/disk topology events, degree <=6.

    Z_i(t)=Y_i(lambda(t))-center.  Side polygons have original triangle
    vertices or Gamma/edge vertices.  Events are their sphere crossings,
    supporting-edge tangencies, the slice-plane sphere tangency and Gamma
    line sphere tangency, and every affine Gamma sign/denominator change.
    """
    Z = np.stack((midpoint*C+D-center, halfwidth*C))
    edges = [(i, j, Z[:, j]-Z[:, i]) for i, j in ((0, 1), (1, 2), (2, 0))]
    h = _pcross(Z[:, 1]-Z[:, 0], Z[:, 2]-Z[:, 0])
    hh = _pdot(h, h)
    g = _pdot(h, Z[:, 0])
    nn = normal@normal
    hn = h@normal
    plane_shift = LD(offset)-normal@center
    records = [("slice_plane_sphere_tangency", _padd(_pmul(g, g), -radius**2*hh))]
    signed = Z@normal
    signed[0] -= plane_shift
    for i in range(3):
        records.append((f"vertex_{i}_sphere_crossing", _padd(_pdot(Z[:, i], Z[:, i]), [-radius**2])))
        records.append((f"gamma_vertex_{i}_sign", signed[:, i]))
    for i, j, edge in edges:
        edge_square = _pdot(edge, edge)
        edge_cross = _pcross(Z[:, i], edge)
        records.append((f"edge_{i}_{j}_sphere_tangency",
                        _padd(_pdot(edge_cross, edge_cross), -radius**2*edge_square)))
        records.append((f"disk_center_edge_{i}_{j}_crossing", _pdot(h, edge_cross)))
        denominator = signed[:, i]-signed[:, j]
        numerator = np.zeros((3, 3), LD)
        for k in range(3):
            numerator[:, k] = np.convolve(signed[:, i], Z[:, j, k])-np.convolve(signed[:, j], Z[:, i, k])
        records.append((f"gamma_edge_{i}_{j}_sphere_crossing",
                        _padd(_pdot(numerator, numerator), -radius**2*_pmul(denominator, denominator))))
        records.append((f"gamma_edge_{i}_{j}_denominator", denominator))
    # Exact literal-normal formula; n.n is retained rather than assuming its
    # float64 normalization is algebraically one.  The distance of the line
    # h.z=g, n.z=plane_shift is numerator/denominator below.
    line_denominator = _padd(nn*hh, -_pmul(hn, hn))
    line_numerator = _padd(nn*_pmul(g, g), -2*plane_shift*_pmul(g, hn))
    line_numerator = _padd(line_numerator, plane_shift**2*hh)
    records.append(("gamma_slice_line_sphere_tangency", _padd(line_numerator, -radius**2*line_denominator)))
    records.append(("gamma_slice_parallel", line_denominator))
    records.append(("gamma_disk_center_crossing", _padd(_pmul(g, hn), -plane_shift*hh)))
    return Z, records


def _radial_events(records, midpoint, halfwidth, maximum):
    event_map = {LD(-1): ["cohort_lambda_lower"], LD(1): ["cohort_lambda_upper"]}
    polynomial_records = []
    for kind, polynomial in records:
        polynomial = np.asarray(polynomial, LD)
        if not np.isfinite(polynomial).all():
            raise ClippedDomainRefusal("Centered topology event polynomial overflowed")
        roots = _roots(polynomial)
        polynomial_records.append(dict(kind=kind, coefficients_local_t=_plain(polynomial),
                                       identically_zero=bool(np.all(polynomial == 0))))
        for root in roots:
            t = LD(root)
            event_map.setdefault(t, []).append(kind)
            if len(event_map) > maximum:
                raise ClippedDomainRefusal("Maximum radial event count exceeded")
    ts = tuple(sorted(event_map))
    events = []
    for t in ts:
        lam = midpoint+halfwidth*t
        events.append(dict(local_t=float(t), lambda_value=float(lam), kinds=tuple(event_map[t])))
    return ts, tuple(events), tuple(polynomial_records)


def _half_polygon(Z, signed, sign):
    """Clip the triangle by Gamma retaining original barycentric coordinates."""
    vertices = [(np.eye(3, dtype=LD)[i], Z[i], sign*signed[i]) for i in range(3)]
    result = []
    for previous, current in zip(vertices[-1:]+vertices[:-1], vertices):
        N0, p0, s0 = previous
        N1, p1, s1 = current
        inside0, inside1 = s0 >= 0, s1 >= 0
        if inside0 != inside1:
            denominator = s0-s1
            if denominator == 0 or not np.isfinite(denominator):
                raise ClippedDomainRefusal("Gamma side edge root is unresolved")
            t = s0/denominator
            if not 0 <= t <= 1:
                raise ClippedDomainRefusal("Gamma side edge root lies outside its edge")
            # The determinant numerator retains the actual plane root.  In
            # particular, an axis plane through the centered origin gives an
            # exactly-zero coordinate instead of a fictitious angular sliver
            # from (1-t)*p0+t*p1 cancellation.
            root_position = (s0*p1-s1*p0)/denominator
            result.append(((1-t)*N0+t*N1, root_position))
        if inside1:
            result.append((N1, p1))
    # Exact duplicate endpoints arise when Gamma passes a triangle vertex.
    unique = []
    for entry in result:
        if not unique or not np.array_equal(entry[0], unique[-1][0]):
            unique.append(entry)
    if len(unique) > 1 and np.array_equal(unique[0][0], unique[-1][0]):
        unique.pop()
    return unique


def _slice(Z, C, normal, plane_shift, radius, sign):
    e1, e2 = Z[1]-Z[0], Z[2]-Z[0]
    h = np.cross(e1, e2)
    hh = h@h
    if not np.isfinite(hh) or hh <= 0:
        raise ClippedDomainRefusal("Current lambda slice is collapsed")
    g = h@Z[0]
    disk_center = (g/hh)*h
    disk_square = radius**2-g*g/hh
    if disk_square <= 0:
        return None
    signed = Z@normal-plane_shift
    polygon = _half_polygon(Z, signed, sign)
    if len(polygon) < 3:
        return None
    unit = h/np.sqrt(hh)
    axis = np.eye(3, dtype=LD)[int(np.argmin(np.abs(unit)))]
    tangent = axis-unit*(unit@axis)
    tangent /= np.sqrt(tangent@tangent)
    tangent2 = np.cross(unit, tangent)
    basis = np.column_stack((tangent, tangent2))
    Q = np.array([(p-disk_center)@basis for _, p in polygon], LD)
    area = sum(_cross2(Q[i], Q[(i+1)%len(Q)]) for i in range(len(Q)))
    if area == 0:
        return None
    if area < 0:
        Q = Q[::-1]
    originalQ = (Z-disk_center)@basis
    matrix = np.column_stack((originalQ[1]-originalQ[0], originalQ[2]-originalQ[0]))
    determinant = _cross2(matrix[:, 0], matrix[:, 1])
    if determinant == 0 or not np.isfinite(determinant):
        raise ClippedDomainRefusal("Slice barycentric map is unresolved")
    inverse = np.array(((matrix[1, 1], -matrix[0, 1]),
                        (-matrix[1, 0], matrix[0, 0])), LD)/determinant
    return Q, originalQ, inverse, basis, disk_center, np.sqrt(disk_square), np.sqrt(hh), C


def _ray_pieces(polygon, radius):
    """All polygon vertex, parallel-ray and circle/edge angular events."""
    edges = np.roll(polygon, -1, axis=0)-polygon
    angles = {0., 2*math.pi}

    def add_angle(point):
        if point@point > 0:
            angles.add(float(np.arctan2(point[1], point[0])%(2*LD(math.pi))))

    for point, edge in zip(polygon, edges):
        squared = edge@edge
        if squared <= 0:
            raise ClippedDomainRefusal("Side polygon has an unresolved zero-length edge")
        add_angle(point)
        add_angle(edge)
        add_angle(-edge)
        nearest = -(point@edge)/squared
        perpendicular = point+nearest*edge
        margin = radius**2-perpendicular@perpendicular
        if margin >= 0:
            half = np.sqrt(margin/squared)
            for t in (nearest-half, nearest+half):
                if 0 <= t <= 1:
                    add_angle(point+t*edge)
    constants = np.array([_cross2(edge, -point) for point, edge in zip(polygon, edges)], LD)

    def clip(theta):
        direction = np.array((np.cos(LD(theta)), np.sin(LD(theta))), LD)
        lower, upper = LD(0), radius
        for edge, constant in zip(edges, constants):
            slope = _cross2(edge, direction)
            if slope > 0:
                lower = max(lower, -constant/slope)
            elif slope < 0:
                upper = min(upper, -constant/slope)
            elif constant < 0:
                return None
        if upper <= lower:
            return None
        return direction, lower, upper

    return tuple(sorted(angles)), clip


@dataclass(frozen=True, slots=True)
class CohortClippedDomain:
    section: object
    geometry_source: dict
    source_section_binding_sha256: str
    seed_center_world_m: np.ndarray
    seed_radius_m: float
    radial_order: int
    angular_order: int
    ray_order: int
    barycentric: np.ndarray
    radial_scale: np.ndarray
    reference_volume_weights_m3: np.ndarray
    birth_volume_weights_m3: np.ndarray
    side: np.ndarray
    centered_birth_offsets_m: np.ndarray
    lambda_event_records: tuple
    topology_event_polynomials: tuple
    topology_records: tuple
    live_quadrature_numeric_sha256: str
    binding_sha256: str
    format: str = FORMAT
    scope: str = SCOPE

    @property
    def physical_creation_admitted(self):
        return False

    @property
    def physical_release_admitted(self):
        return False

    @property
    def numerical_integral_convergence_certified(self):
        return False

    @property
    def point_count(self):
        return len(self.radial_scale)


def _body(domain):
    return _plain(dict(format=domain.format, scope=domain.scope,
        geometry_source=domain.geometry_source,
        source_section_binding_sha256=domain.source_section_binding_sha256,
        seed_center_world_m=domain.seed_center_world_m, seed_radius_m=domain.seed_radius_m,
        radial_order=domain.radial_order, angular_order=domain.angular_order, ray_order=domain.ray_order,
        barycentric=domain.barycentric, radial_scale=domain.radial_scale,
        reference_volume_weights_m3=domain.reference_volume_weights_m3,
        birth_volume_weights_m3=domain.birth_volume_weights_m3, side=domain.side,
        centered_birth_offsets_m=domain.centered_birth_offsets_m,
        lambda_event_records=domain.lambda_event_records,
        topology_event_polynomials=domain.topology_event_polynomials,
        topology_records=domain.topology_records, domain_geometry_resolved=True,
        live_quadrature_numeric_sha256=domain.live_quadrature_numeric_sha256,
        live_quadrature_numeric_binding="exact_integer_ratios_no_longdouble_storage_padding",
        public_point_arithmetic_scope="literal_float64_archive_separate_from_live_longdouble_operands",
        numerical_integral_convergence_certified=False, physical_creation_admitted=False,
        physical_release_admitted=False, physical_force_work_admission=False,
        available_for_creation_funding=False,
        quadrature_semantics="positive_Gauss_sine_endpoint_maps_on_analytic_lambda_and_angle_events_disk_polygon_ray_intervals",
        material_support_semantics="frozen_original_N_lambda_full_A_P_birth_chart_not_material_reinitialization",
        reference_weight_semantics="lambda_squared_det_X_over_slice_area_times_polar_ray_measure",
        float64_source_sums_retained=True,
        centered_geometry_arithmetic="longdouble_then_literal_float64_quadrature_archive"))


def clipped_domain_to_dict(domain):
    validate_cohort_clipped_domain(domain)
    return dict(_body(domain), binding_sha256=domain.binding_sha256)


def validate_cohort_clipped_domain(domain):
    if type(domain) is not CohortClippedDomain or domain.format != FORMAT or domain.scope != SCOPE:
        raise ClippedDomainRefusal("Invalid clipped domain format or scope")
    if hashlib.sha256(_canonical(_body(domain))).hexdigest() != domain.binding_sha256:
        raise ClippedDomainRefusal("Clipped domain differs from its immutable source/quadrature binding")
    if _live_numeric_digest(_live_arrays(domain)) != domain.live_quadrature_numeric_sha256:
        raise ClippedDomainRefusal("Clipped domain live numeric operands differ from their exact logical binding")
    try:
        _validate_cohort_section(domain.section)
    except (TypeError, ValueError, OverflowError) as error:
        raise ClippedDomainRefusal(str(error)) from error
    if domain.section.binding_sha256 != domain.source_section_binding_sha256:
        raise ClippedDomainRefusal("Clipped domain source section binding differs")
    if _canonical(domain.geometry_source) != _canonical(_source_record(domain.section)):
        raise ClippedDomainRefusal("Clipped domain declared source geometry differs from its original section")
    return True


def prepare_cohort_clipped_domain(section, seed_center_m, seed_radius_m, *,
        radial_order=8, angular_order=8, ray_order=8, max_lambda_events=256,
        max_points=250000, require_both_sides=False):
    """Prepare a bounded positive quadrature on actual two-side cohort support.

    Empty and single-side domains are valid geometry results.  A prospective
    enrichment caller may explicitly require both positive-volume sides.
    Inputs are declared geometry: accepted source/law activation is a separate
    caller responsibility, and increasing orders alone is no certificate.
    """
    try:
        _validate_cohort_section(section)
        _current_chart_orientation(section)
    except (TypeError, ValueError, OverflowError) as error:
        raise ClippedDomainRefusal(str(error)) from error
    radial_order = _integer(radial_order, "Radial order", 2, 64)
    angular_order = _integer(angular_order, "Angular order", 2, 64)
    ray_order = _integer(ray_order, "Ray order", 2, 64)
    max_lambda_events = _integer(max_lambda_events, "Maximum radial events", 2, 4096)
    max_points = _integer(max_points, "Maximum quadrature points", 1, 2000000)
    if type(require_both_sides) is not bool:
        raise ClippedDomainRefusal("Both-side requirement must be boolean")
    raw_center = np.asarray(seed_center_m)
    if raw_center.dtype.kind not in "ifu" or raw_center.shape != (3,):
        raise ClippedDomainRefusal("Seed center requires three finite real coordinates")
    center64 = np.asarray(raw_center, float)
    if not np.isfinite(center64).all():
        raise ClippedDomainRefusal("Seed center requires three finite real coordinates")
    radius64 = _finite_real(seed_radius_m, "Seed radius")
    if radius64 <= 64*np.finfo(float).eps*max(float(np.max(np.abs(center64))), 1.):
        raise ClippedDomainRefusal("Seed radius is below represented position resolution")
    center, radius = np.asarray(center64, LD), LD(radius64)
    C = np.asarray(section.triangle_m+section.slope_P_m, LD)
    D = np.asarray(section.displacement_A_m-section.slope_P_m, LD)
    X = np.asarray(section.triangle_m, LD)
    normal, offset = np.asarray(section.normal_xyz, LD), LD(section.offset_m)
    lo, hi = map(LD, section.lambda_bounds)
    midpoint, halfwidth = lo+(hi-lo)/2, (hi-lo)/2
    Z_polynomial, polynomials = _event_polynomials(C, D, center, normal, offset, radius, midpoint, halfwidth)
    ts, events, polynomial_records = _radial_events(polynomials, midpoint, halfwidth, max_lambda_events)
    detX = np.dot(X[0], np.cross(X[1], X[2]))
    if detX <= 0 or not np.isfinite(detX):
        raise ClippedDomainRefusal("Reference cohort determinant is unresolved")
    plane_shift = offset-normal@center
    radial_nodes, radial_weights = map(lambda a: np.asarray(a, LD), np.polynomial.legendre.leggauss(radial_order))
    angular_nodes, angular_weights = map(lambda a: np.asarray(a, LD), np.polynomial.legendre.leggauss(angular_order))
    ray_nodes, ray_weights = map(lambda a: np.asarray(a, LD), np.polynomial.legendre.leggauss(ray_order))
    records, topology = [], []
    for interval_index, (left, right) in enumerate(zip(ts, ts[1:])):
        if not left < right:
            raise ClippedDomainRefusal("Radial topology interval is unresolved")
        interval_mid, interval_half = (left+right)/2, (right-left)/2
        midpoint_lambda = midpoint+halfwidth*interval_mid
        midpoint_Z = Z_polynomial[0]+interval_mid*Z_polynomial[1]
        for sign in (-1, 1):
            middle_slice = _slice(midpoint_Z, C, normal, plane_shift, radius, sign)
            middle_positive = False
            if middle_slice is not None:
                breaks, clip = _ray_pieces(middle_slice[0], middle_slice[5])
                middle_positive = any(clip((a+b)/2) is not None for a, b in zip(breaks, breaks[1:]))
            topology.append(dict(lambda_interval_index=interval_index,
                lambda_bounds=(float(midpoint+halfwidth*left), float(midpoint+halfwidth*right)),
                side=.5*sign, positive_volume_interval=middle_positive))
            if not middle_positive:
                continue  # Analytically subdivided empty interval, never a node mask.
            for radial_node, radial_weight in zip(radial_nodes, radial_weights):
                radial_angle = LD(math.pi/2)*radial_node
                t = interval_mid+interval_half*np.sin(radial_angle)
                lam = midpoint+halfwidth*t
                Z = Z_polynomial[0]+t*Z_polynomial[1]
                data = _slice(Z, C, normal, plane_shift, radius, sign)
                if data is None:
                    raise ClippedDomainRefusal("Positive radial topology interval contains an empty quadrature slice")
                polygon, originalQ, inverse, basis, disk_center, disk_radius, area_scale, _ = data
                breaks, clip = _ray_pieces(polygon, disk_radius)
                lambda_weight = halfwidth*interval_half*LD(math.pi/2)*np.cos(radial_angle)*radial_weight
                positive_pieces = 0
                for angle_left, angle_right in zip(breaks, breaks[1:]):
                    if not angle_left < angle_right:
                        raise ClippedDomainRefusal("Angular topology interval is unresolved")
                    angle_mid = (LD(angle_left)+LD(angle_right))/2
                    angle_half = (LD(angle_right)-LD(angle_left))/2
                    if clip(angle_mid) is None:
                        continue  # Exact disk/polygon ray interval is empty.
                    positive_pieces += 1
                    for angle_node, angle_weight in zip(angular_nodes, angular_weights):
                        mapped_angle = LD(math.pi/2)*angle_node
                        theta = angle_mid+angle_half*np.sin(mapped_angle)
                        ray = clip(theta)
                        if ray is None:
                            raise ClippedDomainRefusal("Positive angular topology interval contains an empty quadrature ray")
                        direction, ray_lo, ray_hi = ray
                        ray_mid, ray_half = (ray_lo+ray_hi)/2, (ray_hi-ray_lo)/2
                        for ray_node, ray_weight in zip(ray_nodes, ray_weights):
                            rho = ray_mid+ray_half*ray_node
                            polar = rho*direction
                            uv = inverse@(polar-originalQ[0])
                            N = np.r_[1-uv.sum(), uv]
                            relative = disk_center+basis@polar
                            phi = normal@relative-plane_shift
                            determinant_birth = np.dot(np.cross(Z[1]-Z[0], Z[2]-Z[0]), N@C)
                            weight = (lambda_weight*angle_half*LD(math.pi/2)*np.cos(mapped_angle)*
                                      angle_weight*ray_half*ray_weight*rho/area_scale)
                            reference_weight, birth_weight = weight*lam**2*detX, weight*determinant_birth
                            if not (np.min(N) >= 0 and np.max(N) <= 1 and lo < lam < hi
                                    and relative@relative < radius**2 and sign*phi > 0
                                    and reference_weight > 0 and birth_weight > 0
                                    and np.isfinite(N).all() and np.isfinite(relative).all()
                                    and np.isfinite(reference_weight) and np.isfinite(birth_weight)):
                                raise ClippedDomainRefusal("Resolved-domain quadrature point or positive weight is unresolved")
                            records.append((N, lam, reference_weight, birth_weight, .5*sign, relative))
                            if len(records) > max_points:
                                raise ClippedDomainRefusal("Maximum quadrature point count exceeded")
                if positive_pieces == 0:
                    raise ClippedDomainRefusal("Positive radial topology interval has no positive angular pieces")
    if require_both_sides and {r[4] for r in records} != {-.5, .5}:
        raise ClippedDomainRefusal("Both positive-volume Gamma sides are required")
    geometry_source = _source_record(section)
    columns = [_readonly([r[i] for r in records], (-1, 3) if i in (0, 5) else (-1,), dtype=LD) for i in range(6)]
    domain = CohortClippedDomain(section, geometry_source, section.binding_sha256, _readonly(center64), radius64,
        radial_order, angular_order, ray_order, *columns, events, polynomial_records, tuple(topology), "", "")
    domain = replace(domain, live_quadrature_numeric_sha256=_live_numeric_digest(_live_arrays(domain)))
    return replace(domain, binding_sha256=hashlib.sha256(_canonical(_body(domain))).hexdigest())
