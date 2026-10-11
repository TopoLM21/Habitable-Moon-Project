"""Whole-time separating-support proofs for current native material pairs.

Unlike componentwise outer normal corners, this proof keeps one physical
clock in the scalar ``n(t) dot v``.  A native incidence authenticates an exact
zero at birth, so division by that clock removes the zero before enclosure.
No small area is rounded to zero.  Each true implicit material component is
contained in its declared upper native union; every pair of union pieces must
have a certified separating support.  The resulting proof excludes material
intersection over the complete interval, without admitting force mechanics.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction as F
import hashlib
import json
import math

from .geometric_native_implicit_epoch_admission import _portable_fingerprint_sha256
from .geometric_native_implicit_material_continuation import _base_components, _normal_series
from .geometric_native_implicit_material_epoch import (
    NativeImplicitMaterialEpoch, validate_native_implicit_material_epoch,
)
from .geometric_native_physical_history import _literal_json, _typed_fingerprint
from .geometric_native_polygons import (
    NativePlane, NativeRay, _incident, _plane_values, _ray_values, _world,
    native_plane_to_dict, native_ray_to_dict, validate_native_polygon,
)
from .geometric_native_pose import FractionInterval as I, _rounded_interval


FORMAT = "native-full-time-correlated-pair-exclusion-1"


class UnresolvedNativePairExclusion(ValueError):
    """No complete all-time separating-support proof was obtained."""


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()


def _hash(value):
    return hashlib.sha256(_encoded(value)).hexdigest()


def _r(value):
    return [value.numerator, value.denominator]


def _i(value):
    return dict(lo=_r(value.lo), hi=_r(value.hi))


def _vector(value):
    if (type(value) is not tuple or len(value) != 3
            or any(type(item) is not float or not math.isfinite(item) for item in value)):
        raise ValueError("Pair exclusion needs each literal complete finite Euler vector")
    return value


def _resources(dt_myr, precision, order, time_cells):
    if type(dt_myr) is not float or not math.isfinite(dt_myr) or dt_myr <= 0:
        raise ValueError("Pair exclusion needs the actual positive physical duration")
    if (type(precision) is not int or not 128 <= precision <= 512
            or type(order) is not int or not 12 <= order <= 40
            or type(time_cells) is not int or not 1 <= time_cells <= 256):
        raise ValueError("Pair exclusion needs explicit bounded scalar proof resources")


@dataclass(frozen=True, slots=True)
class NativeSupportRayNonpositive:
    ray: NativeRay
    inward_plane: NativePlane
    point_owner_omega: tuple
    plane_owner_omega: tuple
    dt_myr: float
    precision: int
    order: int
    time_cells: int
    exact_initial_incidence: bool
    divided_by_birth_clock: bool
    scalar_coefficients: tuple
    scalar_remainder_bound: F
    cell_sign_bounds: tuple


def certify_native_support_ray_nonpositive(ray, inward_plane,
        point_owner_omega, plane_owner_omega, dt_myr, *, precision=256,
        order=28, time_cells=1):
    """Prove this moving ray lies outside this moving inward halfspace.

    The current ordered poses remain in ``ray`` and ``inward_plane``.  Fresh
    rotations are the full saved Euler vectors; their scalar Taylor remainder
    holds over every time, including the continuous birth-clock limit.
    """
    if type(ray) is not NativeRay or type(inward_plane) is not NativePlane:
        raise ValueError("Support exclusion needs exact typed native lineage")
    point_owner_omega, plane_owner_omega = _vector(point_owner_omega), _vector(plane_owner_omega)
    _resources(dt_myr, precision, order, time_cells)
    rate = F.from_float(dt_myr)*sum((abs(F.from_float(x))
        for vector in (point_owner_omega, plane_owner_omega) for x in vector), F(0))
    if rate >= F(1, 4):
        raise ValueError("Scalar support proof requires its declared small full-rotation domain")
    _, coordinates = _ray_values(ray, precision)
    normal_coefficients, normal_tail = _normal_series(inward_plane,
        point_owner_omega, plane_owner_omega, F.from_float(dt_myr), order, precision)
    rows = tuple(_rounded_interval(sum((a*b for a, b in zip(row, coordinates)), I(0)), precision)
        for row in normal_coefficients)
    incidence = _incident(ray, inward_plane)
    if incidence:
        # This is an exact native algebraic identity, not a near-zero bound.
        rows = (I(0),)+rows[1:]
    scale = sum((max(abs(value.lo), abs(value.hi)) for value in coordinates), F(0))
    tail = max(abs(normal_tail.lo), abs(normal_tail.hi))*scale
    if point_owner_omega == plane_owner_omega:
        # exp(-[omega]t) exp([omega]t) is the exact identity at every time.
        # An artificial nonzero Taylor tail would obscure unchanged incidence.
        rows, tail = (rows[0],)+(I(0),)*order, F(0)
    selected = rows[1:] if incidence else rows
    tail_power = order if incidence else order+1
    cells = []
    for index in range(time_cells):
        domain = I(F(index, time_cells), F(index+1, time_cells))
        value = I(0)
        for coefficient in reversed(selected):
            value = _rounded_interval(value*domain+coefficient, precision)
        remainder = tail*domain.hi**tail_power
        value = _rounded_interval(value+I(-remainder, remainder), precision)
        if value.hi > 0:
            raise UnresolvedNativePairExclusion("The complete scalar support sign is not proved nonpositive")
        cells.append((domain, value))
    return NativeSupportRayNonpositive(ray, inward_plane, point_owner_omega,
        plane_owner_omega, dt_myr, precision, order, time_cells,
        incidence, incidence, rows, tail, tuple(cells))


@dataclass(frozen=True, slots=True)
class NativePairSeparationPiece:
    donor_piece_sequence: int
    receiver_piece_sequence: int
    separating_bank: str
    separating_support_sequence: int
    separating_support_weights: tuple
    vertex_proofs: tuple


@dataclass(frozen=True, slots=True)
class NativePairTemporalExclusion:
    source_epoch: object
    donor_component_id: str
    receiver_component_id: str
    omega_rad_per_myr: tuple
    dt_myr: float
    precision: int
    order: int
    time_cells: int
    piece_proofs: tuple
    provenance: str
    proof_sha256: str


def _vertex_body(value):
    return dict(ray=native_ray_to_dict(value.ray),
        inward_plane=native_plane_to_dict(value.inward_plane),
        point_owner_omega=list(value.point_owner_omega), plane_owner_omega=list(value.plane_owner_omega),
        exact_initial_incidence=value.exact_initial_incidence,
        divided_by_birth_clock=value.divided_by_birth_clock,
        scalar_coefficients=[_i(row) for row in value.scalar_coefficients],
        scalar_remainder_bound=_r(value.scalar_remainder_bound),
        cell_sign_bounds=[dict(time_fraction=_i(clock), sign_bounds=_i(bounds))
            for clock, bounds in value.cell_sign_bounds])


def _body(value):
    return dict(format=FORMAT, source_epoch_sha256=value.source_epoch.epoch_sha256,
        complete_source_epoch_fingerprint_sha256=_portable_fingerprint_sha256(value.source_epoch),
        donor_component_id=value.donor_component_id, receiver_component_id=value.receiver_component_id,
        omega_rad_per_myr=[[owner, list(vector)] for owner, vector in value.omega_rad_per_myr],
        dt_myr=value.dt_myr, precision=value.precision, order=value.order,
        time_cells=value.time_cells, provenance=value.provenance,
        pieces=[dict(donor_piece_sequence=row.donor_piece_sequence,
            receiver_piece_sequence=row.receiver_piece_sequence,
            separating_bank=row.separating_bank,
            separating_support_sequence=row.separating_support_sequence,
            separating_support_weights=[dict(support_sequence=index, weight=_r(weight))
                for index, weight in row.separating_support_weights],
            vertex_proofs=[_vertex_body(proof) for proof in row.vertex_proofs])
            for row in value.piece_proofs],
        temporal_material_area_bounds_sr=_i(I(0)),
        all_outer_union_piece_pairs_excluded=True,
        exact_implicit_material_mask_retained=True,
        positive_area_tolerance_used=False, full_saved_euler_vectors_preserved=True,
        force_solve_authority_supplied=False, force_admission_authorized=False)


def _combined_support_candidates(points, supports, precision):
    """Propose positive support combinations; every proposal is then proved.

    Midpoints only choose a candidate separating plane.  They make no sign
    decision: the full scalar ray proof must independently certify every ray.
    Common exact pose is necessary to represent a weighted sum as one native
    plane.  Each coefficient is positive, so every receiving point also lies
    in the resulting inward halfspace.
    """
    values = []
    for plane in supports:
        normal = _plane_values(plane, precision)[1]
        values.append(tuple(F(0) if _incident(ray, plane) else
            sum((a*b for a, b in zip(normal, _ray_values(ray, precision)[1])), I(0)).midpoint
            for ray in points))
    for first in range(len(supports)):
        for second in range(first+1, len(supports)):
            a, b = supports[first], supports[second]
            if a.rotations != b.rotations:
                continue
            low, high, feasible = F(0), None, True
            for av, bv in zip(values[first], values[second]):
                if bv > 0:
                    bound = -av/bv
                    high = bound if high is None else min(high, bound)
                elif bv < 0:
                    low = max(low, -av/bv)
                elif av > 0:
                    feasible = False
                    break
            if not feasible or high is not None and (high <= 0 or low > high):
                continue
            if low > 0:
                weight = 2*low if high is None or 2*low <= high else (low+high)/2
            else:
                weight = F(1) if high is None else high/2
            if weight <= 0:
                continue
            an = tuple(F(value, max(map(abs, a.normal))) for value in a.normal)
            bn = tuple(F(value, max(map(abs, b.normal))) for value in b.normal)
            vector = tuple(x+weight*y for x, y in zip(an, bn))
            if not any(vector):
                continue
            yield NativePlane(vector, a.rotations), ((first, F(1)), (second, weight))


def certify_native_birth_pair_exclusion(epoch, donor_component_id, receiver_component_id,
        omega_rad_per_myr, dt_myr, *, precision=256, order=28, time_cells=1, provenance):
    """Exclude one complete pair by its independently supplied source masks."""
    if type(epoch) is not NativeImplicitMaterialEpoch:
        raise ValueError("This current birth proof needs the typed first complete implicit epoch")
    epoch = validate_native_implicit_material_epoch(epoch)
    if epoch.phase != "partition":
        raise ValueError("Pair exclusion requires the exact complete prior partition")
    if (type(omega_rad_per_myr) is not tuple or len(omega_rad_per_myr) != 2
            or any(type(row) is not tuple or len(row) != 2 or type(row[0]) is not int
                for row in omega_rad_per_myr)
            or tuple(owner for owner, _ in omega_rad_per_myr) != (0, 1)):
        raise ValueError("Pair exclusion needs all literal canonical owner records")
    motions = {owner: _vector(vector) for owner, vector in omega_rad_per_myr}
    _resources(dt_myr, precision, order, time_cells)
    if any(type(value) is not str or not value.strip()
            for value in (donor_component_id, receiver_component_id, provenance)):
        raise ValueError("Pair exclusion needs explicit component identities and provenance")
    components = {row.component_id: row for row in _base_components(epoch)}
    if donor_component_id not in components or receiver_component_id not in components:
        raise ValueError("Pair exclusion component identity is absent from its complete source")
    donor, receiver = components[donor_component_id], components[receiver_component_id]
    if donor.owner != 1 or receiver.owner != 0:
        raise ValueError("This declared physical direction is donor1 to receiver0")
    proofs = []
    for di, dg in enumerate(donor.upper_geometries):
        dg = _world(validate_native_polygon(dg))
        for ri, rg in enumerate(receiver.upper_geometries):
            rg = _world(validate_native_polygon(rg))
            selected = None
            for bank, points, supports, point_omega, plane_omega in (
                    ("receiver", dg.vertices, rg.supports, motions[1], motions[0]),
                    ("donor", rg.vertices, dg.supports, motions[0], motions[1])):
                candidates = tuple((plane, ((pi, F(1)),)) for pi, plane in enumerate(supports))
                candidates += tuple(_combined_support_candidates(points, supports, precision))
                for plane, weights in candidates:
                    try:
                        vertices = tuple(certify_native_support_ray_nonpositive(ray, plane,
                            point_omega, plane_omega, dt_myr, precision=precision,
                            order=order, time_cells=time_cells) for ray in points)
                    except UnresolvedNativePairExclusion:
                        continue
                    selected = NativePairSeparationPiece(di, ri, bank, weights[0][0], weights, vertices)
                    break
                if selected is not None:
                    break
            if selected is None:
                raise UnresolvedNativePairExclusion("A complete upper-piece pair has no certified all-time separator")
            proofs.append(selected)
    result = NativePairTemporalExclusion(epoch, donor_component_id, receiver_component_id,
        omega_rad_per_myr, dt_myr, precision, order, time_cells, tuple(proofs), provenance, "")
    return replace(result, proof_sha256=_hash(_body(result)))


def validate_native_birth_pair_exclusion(value):
    if type(value) is not NativePairTemporalExclusion:
        raise ValueError("Expected the exact typed full-time pair proof")
    replay = certify_native_birth_pair_exclusion(value.source_epoch,
        value.donor_component_id, value.receiver_component_id, value.omega_rad_per_myr,
        value.dt_myr, precision=value.precision, order=value.order,
        time_cells=value.time_cells, provenance=value.provenance)
    if _typed_fingerprint(value) != _typed_fingerprint(replay):
        raise ValueError("Typed pair exclusion differs from its complete independent semantic replay")
    return replay


def native_birth_pair_exclusion_to_dict(value):
    value = validate_native_birth_pair_exclusion(value)
    return dict(_body(value), proof_sha256=value.proof_sha256)


def native_birth_pair_exclusion_from_dict(raw, *, expected_epoch, expected_donor_component_id,
        expected_receiver_component_id, expected_omega_rad_per_myr, expected_dt_myr,
        expected_precision, expected_order, expected_time_cells, expected_provenance):
    """A selfseal grants no proof authority; replay every independent input."""
    _literal_json(raw)
    replay = certify_native_birth_pair_exclusion(expected_epoch, expected_donor_component_id,
        expected_receiver_component_id, expected_omega_rad_per_myr, expected_dt_myr,
        precision=expected_precision, order=expected_order, time_cells=expected_time_cells,
        provenance=expected_provenance)
    expected = dict(_body(replay), proof_sha256=replay.proof_sha256)
    # Literal JSON has no tuples or dataclasses.  Its canonical bytes retain
    # bool/int, int/float and signed-zero distinctions after _literal_json.
    if _encoded(raw) != _encoded(expected):
        raise ValueError("Serialized pair exclusion differs from independently supplied full source replay")
    return replay
