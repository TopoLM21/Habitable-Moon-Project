"""Whole-time exclusion of a signed, implicit curved birth correction.

The positive correction is the swept archive minus the endpoint archive.
Consequently its points satisfy the endpoint receiver's outside halfspace at
the old endpoint exactly.  Its containing rectangle cannot reproduce that
correlation: a tiny positive rectangular upper bound at time zero is not a
positive material witness.  A strictly negative derivative of the full new
SO(3) support on the containing chart preserves the exact outside constraint
for every later time.  The rectangle remains an enclosure, never a material
polygon, an area debit, or an inclined contact carrier.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction as F
from math import factorial
import hashlib
import json

from .geometric_intrinsic_geometry import RotationIncrement
from .geometric_native_birth_pair_exclusion import (
    NativePairSeparationPiece, UnresolvedNativePairExclusion,
    _resources, _vector, _vertex_body, certify_native_support_ray_nonpositive,
)
from .geometric_native_implicit_epoch_admission import (
    NativeImplicitImmediateIncomingEdge, _edge_body, _immediate_edges,
    _portable_fingerprint_sha256, _unit_ray,
)
from .geometric_native_implicit_material_continuation import (
    _base_components, _intersection, _normal_series,
)
from .geometric_native_implicit_material_epoch import (
    NativeImplicitMaterialEpoch, validate_native_implicit_material_epoch,
)
from .geometric_native_physical_history import _literal_json, _typed_fingerprint
from .geometric_native_polygons import (
    NativeArc, NativePlane, NativeRay, _arc_length, _arc_order, _cross, _dot,
    _plane_values, _ray_values, _same_plane, _world, native_plane_to_dict,
    native_polygon_to_dict, native_ray_to_dict, native_shared_boundary_arcs,
    validate_native_polygon,
)
from .geometric_native_pose import FractionInterval as I, _rounded_interval, certify_rotation_history


FORMAT = "native-signed-curved-birth-pair-exclusion-1"


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()


def _hash(value):
    return hashlib.sha256(_encoded(value)).hexdigest()


def _r(value):
    return [value.numerator, value.denominator]


def _i(value):
    return dict(lo=_r(value.lo), hi=_r(value.hi))


def _polynomial_bounds(coefficients, domain, precision):
    value = I(0)
    for coefficient in reversed(coefficients):
        value = _rounded_interval(value*domain+coefficient, precision)
    return value


def _same_oriented_plane(first, second, precision):
    if not _same_plane(first, second):
        return False
    return _dot(_plane_values(first, precision)[1],
        _plane_values(second, precision)[1]).lo > 0


@dataclass(frozen=True, slots=True)
class NativeCurvedChartSeparation:
    receiver_piece_sequence: int
    separating_support_sequence: int
    receiver_inward_plane: NativePlane
    endpoint_outside_plane: NativePlane
    world_chart_bounds: tuple
    scalar_coefficients: tuple
    scalar_remainder_bound: F
    derivative_remainder_bound: F
    whole_time_support_bounds: I
    whole_time_derivative_bounds: I
    initial_support_bounds: I
    initial_support_constraint: str
    method: str


def _chart_plane_polynomial(world_chart, inward_plane, donor_omega,
        receiver_omega, dt_myr, precision, order):
    """Return a correlated scalar clock polynomial and derivative tails."""
    duration = F.from_float(dt_myr)
    rows, normal_tail = _normal_series(inward_plane, donor_omega,
        receiver_omega, duration, order, precision)
    coefficients = tuple(_rounded_interval(_dot(row, world_chart), precision) for row in rows)
    chart_scale = sum((max(abs(value.lo), abs(value.hi)) for value in world_chart), F(0))
    value_tail = max(abs(normal_tail.lo), abs(normal_tail.hi))*chart_scale
    _, normal = _plane_values(inward_plane, precision)
    normal_scale = sum((max(abs(value.lo), abs(value.hi)) for value in normal), F(0))
    rate = duration*(sum((abs(F.from_float(value)) for value in donor_omega), F(0))
        +sum((abs(F.from_float(value)) for value in receiver_omega), F(0)))
    if rate >= F(1, 4):
        raise UnresolvedNativePairExclusion("Curved support derivative needs bounded complete relative motion")
    # For the analytic vector series |c_k|_1 <= scale*rate**k/k!.
    # Differentiation leaves the tail beginning at k=order+1.  Subsequent
    # terms have ratio at most rate/(order+1), so this geometric sum bounds
    # every derivative on the complete physical fractional clock [0,1].
    derivative_tail = normal_scale*chart_scale*rate**(order+1)/factorial(order)/(1-rate/F(order+1))
    whole = I(0, 1)
    support = _rounded_interval(_polynomial_bounds(coefficients, whole, precision)
        +I(-value_tail, value_tail), precision)
    derivative = _rounded_interval(_polynomial_bounds(
        tuple(value*index for index, value in enumerate(coefficients) if index), whole, precision)
        +I(-derivative_tail, derivative_tail), precision)
    return coefficients, value_tail, derivative_tail, support, derivative


def _prove_chart_plane(world_chart, inward_plane, endpoint_plane,
        donor_omega, receiver_omega, dt_myr, precision, order, piece, support):
    coefficients, tail, derivative_tail, whole, derivative = _chart_plane_polynomial(
        world_chart, inward_plane, donor_omega, receiver_omega, dt_myr, precision, order)
    if whole.hi <= 0:
        method = "whole_chart_nonpositive_support"
        initial = coefficients[0]
        constraint = "containing_chart_support_enclosure"
    elif _same_oriented_plane(inward_plane, endpoint_plane, precision) and derivative.hi < 0:
        method = "exact_outside_endpoint_archive_and_negative_derivative"
        initial = I(min(F(0), coefficients[0].lo), 0)
        constraint = "signed_correction_equals_complete_swept_archive_minus_endpoint_archive"
    else:
        raise UnresolvedNativePairExclusion("Neither whole-chart nor exact-mask derivative separation was certified")
    return NativeCurvedChartSeparation(piece, support, inward_plane,
        endpoint_plane, world_chart, coefficients, tail, derivative_tail,
        whole, derivative, initial, constraint, method)


def _proxy_pair_proof(donor_geometry, receiver_geometry, donor_sequence,
        receiver_sequence, motions, dt_myr, precision, order):
    donor = _world(validate_native_polygon(donor_geometry))
    receiver = _world(validate_native_polygon(receiver_geometry))
    for bank, points, supports, point_omega, plane_omega in (
            ("receiver", donor.vertices, receiver.supports, motions[1], motions[0]),
            ("donor", receiver.vertices, donor.supports, motions[0], motions[1])):
        for sequence, plane in enumerate(supports):
            try:
                proofs = tuple(certify_native_support_ray_nonpositive(point,
                    plane, point_omega, plane_omega, dt_myr, precision=precision,
                    order=order, time_cells=1) for point in points)
            except UnresolvedNativePairExclusion:
                continue
            return NativePairSeparationPiece(donor_piece_sequence=donor_sequence,
                receiver_piece_sequence=receiver_sequence, separating_bank=bank,
                separating_support_sequence=sequence,
                separating_support_weights=((sequence, F(1)),), vertex_proofs=proofs)
    raise UnresolvedNativePairExclusion("A raw proxy piece pair is not excluded over the complete new clock")


@dataclass(frozen=True, slots=True)
class NativeCurvedBirthPairExclusion:
    source_epoch: NativeImplicitMaterialEpoch
    donor_component_id: str
    receiver_component_id: str
    omega_rad_per_myr: tuple
    dt_myr: float
    precision: int
    order: int
    correction_id: str
    negative_component_id: str
    original_chart_bounds: tuple
    proxy_piece_proofs: tuple
    curved_piece_proofs: tuple
    provenance: str
    proof_sha256: str


def certify_native_curved_pair_support(epoch, omega_rad_per_myr, dt_myr,
        donor_component_id, receiver_component_id, *, precision=256,
        order=28, provenance):
    """Exclude the complete positive correction plus every raw HOT piece.

    The source epoch supplies the exact signed mask and the immutable old
    chronology.  No caller-provided initial-sign assertion can grant proof.
    Force authority and constitutive admission remain external.
    """
    if type(epoch) is not NativeImplicitMaterialEpoch:
        raise ValueError("Curved support needs the typed first complete implicit material epoch")
    epoch = validate_native_implicit_material_epoch(epoch)
    if epoch.phase != "partition":
        raise ValueError("Curved support needs the complete accepted source partition")
    _resources(dt_myr, precision, order, 1)
    if (type(omega_rad_per_myr) is not tuple or len(omega_rad_per_myr) != 2
            or any(type(row) is not tuple or len(row) != 2 or type(row[0]) is not int for row in omega_rad_per_myr)
            or tuple(owner for owner, _ in omega_rad_per_myr) != (0, 1)):
        raise ValueError("Curved support needs all literal canonical owner motions")
    motions = {owner: _vector(vector) for owner, vector in omega_rad_per_myr}
    if any(type(value) is not str or not value.strip()
            for value in (donor_component_id, receiver_component_id, provenance)):
        raise ValueError("Curved support needs explicit component identities and provenance")
    components = {row.component_id: row for row in epoch.components}
    if donor_component_id not in components or receiver_component_id not in components:
        raise ValueError("Curved component identity is absent from its independent source")
    donor, receiver = components[donor_component_id], components[receiver_component_id]
    if (donor.owner != 1 or receiver.owner != 0 or donor.material_kind != "hot_birth"
            or receiver.material_kind != "original_live"):
        raise ValueError("This proof requires the positive HOT correction into original receiver0")
    corrections = tuple(row for row in epoch.paired_corrections
        if row.positive_component_id == donor_component_id)
    if len(corrections) != 1:
        raise ValueError("A unique source-authenticated signed curved correction is required")
    correction = corrections[0]
    certificate = epoch.source_material_state.sweep_certificate
    source_motions = dict(certificate.journal.omega_rad_per_myr)
    histories = dict(epoch.additional_owner_histories)
    if certificate.donor_owner != 1 or certificate.receiver_owner != 0:
        raise ValueError("The exact first-entry signed mask must retain its proved owner direction")
    if histories[0] != histories[1]:
        raise ValueError("The source must preserve one exact complete material partition")
    chart = (I(0, certificate.terminal_height_bounds.hi),
        I(-certificate.maximum_ratio_bounds.hi, -certificate.terminal_slope_bounds.lo), I(1))
    chart_history = (RotationIncrement(source_motions[1], certificate.journal.dt_myr),)+histories[1]
    pose = certify_rotation_history(chart_history, precision).intervals
    world_chart = tuple(_rounded_interval(sum((pose[i][j]*chart[j] for j in range(3)), I(0)), precision)
        for i in range(3))
    endpoint_plane = NativePlane((-1, 0, 0)).rotated(
        (RotationIncrement(source_motions[0], certificate.journal.dt_myr),)+histories[0])
    proxy_proofs = tuple(_proxy_pair_proof(dg, rg, di, ri,
        motions, dt_myr, precision, order)
        for di, dg in enumerate(donor.proxy_geometries)
        for ri, rg in enumerate(receiver.proxy_geometries))
    curved_proofs = []
    for ri, geometry in enumerate(receiver.proxy_geometries):
        selected = None
        for pi, plane in enumerate(geometry.support_planes):
            try:
                selected = _prove_chart_plane(world_chart, plane, endpoint_plane,
                    motions[1], motions[0], dt_myr, precision, order, ri, pi)
            except UnresolvedNativePairExclusion:
                continue
            break
        if selected is None:
            raise UnresolvedNativePairExclusion("The true signed curved mask has no all-time receiver separator")
        curved_proofs.append(selected)
    value = NativeCurvedBirthPairExclusion(epoch, donor_component_id,
        receiver_component_id, omega_rad_per_myr, dt_myr, precision, order,
        correction.correction_id, correction.negative_component_id, chart,
        proxy_proofs, tuple(curved_proofs), provenance, "")
    return replace(value, proof_sha256=_hash(_body(value)))


def _curved_piece_body(value):
    return dict(receiver_piece_sequence=value.receiver_piece_sequence,
        separating_support_sequence=value.separating_support_sequence,
        receiver_inward_plane=native_plane_to_dict(value.receiver_inward_plane),
        endpoint_outside_plane=native_plane_to_dict(value.endpoint_outside_plane),
        world_chart_bounds=[_i(row) for row in value.world_chart_bounds],
        scalar_coefficients=[_i(row) for row in value.scalar_coefficients],
        scalar_remainder_bound=_r(value.scalar_remainder_bound),
        derivative_remainder_bound=_r(value.derivative_remainder_bound),
        whole_time_support_bounds=_i(value.whole_time_support_bounds),
        whole_time_derivative_bounds=_i(value.whole_time_derivative_bounds),
        initial_support_bounds=_i(value.initial_support_bounds),
        initial_support_constraint=value.initial_support_constraint,
        method=value.method, physical_fractional_clock=_i(I(0, 1)))


def _body(value):
    correction = next(row for row in value.source_epoch.paired_corrections
        if row.correction_id == value.correction_id)
    return dict(format=FORMAT, source_epoch_sha256=value.source_epoch.epoch_sha256,
        complete_source_epoch_fingerprint_sha256=_portable_fingerprint_sha256(value.source_epoch),
        source_material_state_sha256=value.source_epoch.source_material_state.state_sha256,
        donor_component_id=value.donor_component_id, receiver_component_id=value.receiver_component_id,
        omega_rad_per_myr=[[owner, list(vector)] for owner, vector in value.omega_rad_per_myr],
        dt_myr=value.dt_myr, precision=value.precision, order=value.order, provenance=value.provenance,
        signed_correction_id=value.correction_id, negative_component_id=value.negative_component_id,
        positive_component_id=correction.positive_component_id,
        positive_curved_measure_bounds_sr=_i(correction.area_bounds_sr),
        original_gnomonic_chart_bounds=[_i(row) for row in value.original_chart_bounds],
        original_curved_mask="complete_first_entry_swept_archive minus endpoint_archive",
        proxy_pieces=[dict(donor_piece_sequence=row.donor_piece_sequence,
            receiver_piece_sequence=row.receiver_piece_sequence, separating_bank=row.separating_bank,
            separating_support_sequence=row.separating_support_sequence,
            separating_support_weights=[[index, _r(weight)] for index, weight in row.separating_support_weights],
            vertex_proofs=[_vertex_body(proof) for proof in row.vertex_proofs]) for row in value.proxy_piece_proofs],
        curved_pieces=[_curved_piece_body(row) for row in value.curved_piece_proofs],
        whole_material_pair_excluded_for_whole_interval=True,
        temporal_material_area_bounds_sr=_i(I(0)),
        containing_rectangle_is_actual_material_mask=False,
        positive_curved_correction_deleted=False, negative_curved_correction_deleted=False,
        curved_area_assigned_to_polygon=False, physical_curve_carrier_needed=False,
        constitutive_time_admission_supplied=False, force_solve_authority_supplied=False,
        force_admission_authorized=False)


def validate_native_curved_pair_support(value):
    if type(value) is not NativeCurvedBirthPairExclusion:
        raise ValueError("Expected the exact immutable signed curved pair proof")
    replay = certify_native_curved_pair_support(value.source_epoch,
        value.omega_rad_per_myr, value.dt_myr, value.donor_component_id,
        value.receiver_component_id, precision=value.precision,
        order=value.order, provenance=value.provenance)
    if _typed_fingerprint(value) != _typed_fingerprint(replay):
        raise ValueError("Curved pair proof differs from its independent typed semantic replay")
    return replay


def native_curved_pair_support_to_dict(value):
    value = validate_native_curved_pair_support(value)
    return json.loads(_encoded(dict(_body(value), proof_sha256=value.proof_sha256)))


def native_curved_pair_support_from_dict(raw, *, expected_epoch):
    if type(raw) is not dict or raw.get("format") != FORMAT:
        raise ValueError("Invalid signed curved pair support schema")
    try:
        records = raw["omega_rad_per_myr"]
        if type(records) is not list or any(type(row) is not list or len(row) != 2
                or type(row[0]) is not int or type(row[1]) is not list for row in records):
            raise ValueError("Curved support requires canonical literal JSON motions")
        value = certify_native_curved_pair_support(expected_epoch,
            tuple((owner, tuple(vector)) for owner, vector in records), raw["dt_myr"],
            raw["donor_component_id"], raw["receiver_component_id"],
            precision=raw["precision"], order=raw["order"], provenance=raw["provenance"])
        expected = json.loads(_encoded(dict(_body(value), proof_sha256=value.proof_sha256)))
        _literal_json(raw)
        if _encoded(raw) != _encoded(expected):
            raise ValueError("Saved curved proof differs from its complete independent source replay")
        return value
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError("Invalid signed curved pair support fields") from exc


def certify_native_curved_related_cold_front(epoch, omega_rad_per_myr,
        dt_myr, *, donor_original_parent_id="octant:5", receiver_original_parent_id="octant:3",
        sample_fraction=F(1, 100000), precision=256, provenance):
    """Prove an actual cold5-to-cold3 front on an inner material subset.

    The negative curved component is retained: donor polygons are the source's
    certified LOWER union after subtraction of the complete containing chart.
    An interior sample in the current interval can reveal a contact that has
    disappeared by its endpoint.  Positive native initial-edge flux, when
    present, separately proves that contact starts at arbitrarily small time.
    This witness supplies neither its terminal clock nor force permission.
    """
    if type(epoch) is not NativeImplicitMaterialEpoch:
        raise ValueError("The cold front needs the typed complete first material epoch")
    epoch = validate_native_implicit_material_epoch(epoch)
    _resources(dt_myr, precision, 28, 1)
    if (type(omega_rad_per_myr) is not tuple or len(omega_rad_per_myr) != 2
            or any(type(row) is not tuple or len(row) != 2 or type(row[0]) is not int for row in omega_rad_per_myr)
            or tuple(owner for owner, _ in omega_rad_per_myr) != (0, 1)):
        raise ValueError("The cold front needs every literal full canonical owner motion")
    motions = {owner: _vector(vector) for owner, vector in omega_rad_per_myr}
    if (type(sample_fraction) is not F or not 0 < sample_fraction <= 1
            or type(provenance) is not str or not provenance.strip()):
        raise ValueError("The cold front needs an explicit exact interior sample and provenance")
    pieces = _base_components(epoch)
    selected = {}
    target_parents = (donor_original_parent_id, receiver_original_parent_id)
    if any(type(parent) is not str for parent in target_parents) or len(set(target_parents)) != 2:
        raise ValueError("The cold front needs two different literal source parent identities")
    for component in pieces:
        if component.material_kind == "original_live" and component.original_parent_id in target_parents:
            selected[component.original_parent_id] = component
    if set(selected) != set(target_parents):
        raise ValueError("The declared exact cold source parent components are absent")
    donor, receiver = selected[donor_original_parent_id], selected[receiver_original_parent_id]
    if donor.owner != 1 or receiver.owner != 0:
        raise ValueError("The cold source must preserve donor1 into receiver0")
    negative = tuple(row for row in epoch.paired_corrections if row.negative_component_id == donor.component_id)
    if len(negative) > 1:
        raise ValueError("The cold source has ambiguous negative curved correction provenance")
    # This exact saved binary64 duration is itself within the requested
    # physical interval; the nominal rational sample is only a selection.
    sample_dt_myr = float(F.from_float(dt_myr)*sample_fraction)
    actual_fraction = F.from_float(sample_dt_myr)/F.from_float(dt_myr)
    if not 0 < actual_fraction <= 1:
        raise ValueError("The literal sampled physical time is outside the actual interval")
    fractions = {owner: tuple(F.from_float(x) for x in vector) for owner, vector in motions.items()}
    witnesses = []
    for di, dg in enumerate(donor.lower_geometries):
        for ri, rg in enumerate(receiver.lower_geometries):
            overlap = _intersection(dg.rotated(motions[1], sample_dt_myr),
                rg.rotated(motions[0], sample_dt_myr))
            if overlap is None:
                continue
            area = overlap.area_bounds()
            if area.lo <= 0:
                continue
            immediate = _immediate_edges(dg, rg, fractions[1], fractions[0], precision)
            endpoint = _intersection(dg.rotated(motions[1], dt_myr), rg.rotated(motions[0], dt_myr))
            witnesses.append(dict(donor_piece_sequence=di, receiver_piece_sequence=ri,
                donor_inner_geometry=native_polygon_to_dict(dg), receiver_inner_geometry=native_polygon_to_dict(rg),
                sample_intersection_geometry=native_polygon_to_dict(overlap),
                sample_intersection_area_bounds_sr=_i(area),
                full_interval_endpoint_has_positive_inner_subset=(endpoint is not None and endpoint.area_bounds().lo > 0),
                immediate_incoming_edges=[_edge_body(edge) for edge in immediate]))
    if not witnesses:
        raise UnresolvedNativePairExclusion("No actual positive cold inner subset is proved at this physical sample")
    body = dict(format="native-cold-interior-front-witness-1",
        source_epoch_sha256=epoch.epoch_sha256,
        complete_source_epoch_fingerprint_sha256=_portable_fingerprint_sha256(epoch),
        donor_component_id=donor.component_id, receiver_component_id=receiver.component_id,
        donor_original_parent_id=donor.original_parent_id, receiver_original_parent_id=receiver.original_parent_id,
        omega_rad_per_myr=[[owner, list(vector)] for owner, vector in omega_rad_per_myr],
        dt_myr=dt_myr, nominal_sample_fraction=_r(sample_fraction),
        actual_sample_dt_myr=sample_dt_myr, actual_sample_fraction=_r(actual_fraction),
        actual_sample_absolute_time_myr=_r(epoch.time_myr+F.from_float(sample_dt_myr)),
        negative_curved_correction_id=(negative[0].correction_id if negative else None),
        negative_curved_correction_measure_bounds_sr=(_i(negative[0].area_bounds_sr) if negative else _i(I(0))),
        complete_signed_curved_mask_retained=True,
        sampled_geometries_are_certified_inner_material_subsets=True,
        whole_material_pair_has_positive_temporal_subset=True,
        initial_contact_starts_at_arbitrarily_small_positive_time=any(row["immediate_incoming_edges"] for row in witnesses),
        witnesses=witnesses, provenance=provenance,
        sample_area_used_as_inclined_contact_measure=False,
        terminal_contact_clock_proved=False, whole_pair_time_domain_certified=False,
        constitutive_time_admission_supplied=False, force_admission_authorized=False)
    return json.loads(_encoded(dict(body, proof_sha256=_hash(body))))


def certify_native_cold5_receiver1_immediate_slice(epoch, omega_rad_per_myr,
        dt_myr, *, original_y_over_z=(F(-2, 10**23), F(-1, 10**23)),
        precision=256, provenance):
    """An actual positive-flux interior slice of the mixed-sign cold5 arc.

    Its endpoints are exact native intersections on the actual shared plane.
    The specified donor material Y/Z cuts are explicit choices of a witness,
    not a replacement for the full incoming carrier.  Pullback rays and
    intervals in receiver1's original frame authenticate the receiving
    spatial memory separately.  Strict positive flux proves entry begins
    arbitrarily close to the current physical clock; the complete front is
    still an external obligation.
    """
    if type(epoch) is not NativeImplicitMaterialEpoch:
        raise ValueError("The immediate cold slice needs the complete typed first epoch")
    epoch = validate_native_implicit_material_epoch(epoch)
    _resources(dt_myr, precision, 28, 1)
    if (type(omega_rad_per_myr) is not tuple or len(omega_rad_per_myr) != 2
            or any(type(row) is not tuple or len(row) != 2 or type(row[0]) is not int for row in omega_rad_per_myr)
            or tuple(owner for owner, _ in omega_rad_per_myr) != (0, 1)):
        raise ValueError("The immediate cold slice needs every full literal canonical owner motion")
    motions = {owner: _vector(vector) for owner, vector in omega_rad_per_myr}
    if (type(original_y_over_z) is not tuple or len(original_y_over_z) != 2
            or any(type(value) is not F for value in original_y_over_z)
            or not original_y_over_z[0] < original_y_over_z[1] < 0
            or type(provenance) is not str or not provenance.strip()):
        raise ValueError("The immediate cold slice needs two ordered exact negative material Y/Z cuts")
    selected = {(row.material_kind, row.original_parent_id): row for row in _base_components(epoch)}
    donor = selected[("original_live", "octant:5")]
    receiver = selected[("original_live", "octant:1")]
    certificate = epoch.source_material_state.sweep_certificate
    source_motions = dict(certificate.journal.omega_rad_per_myr)
    extra = dict(epoch.additional_owner_histories)
    histories = {owner: (RotationIncrement(source_motions[owner], certificate.journal.dt_myr),)+extra[owner]
        for owner in (0, 1)}
    receiver_inverse = tuple(RotationIncrement(tuple(-x for x in step.omega_rad_per_myr), step.dt_myr)
        for step in reversed(histories[0]))
    difference = tuple(F.from_float(a)-F.from_float(b) for a, b in zip(motions[1], motions[0]))
    for di, dg in enumerate(donor.lower_geometries):
        for ri, rg in enumerate(receiver.lower_geometries):
            for edge in native_shared_boundary_arcs(dg, rg):
                inward = next((plane for plane in rg.support_planes
                    if _same_plane(plane, edge.plane)), None)
                if inward is None:
                    continue
                normal = tuple(_rounded_interval(value, precision+24)
                    for value in _plane_values(inward, precision)[1])
                rays = []
                for ratio in original_y_over_z:
                    cut = NativePlane((0, ratio.denominator, -ratio.numerator)).rotated(histories[1])
                    ray = NativeRay("cross", planes=(edge.plane, cut))
                    if _ray_values(ray, precision)[1][2].hi < 0:
                        ray = replace(ray, sign=-1)
                    rays.append(ray)
                if any(_arc_order(edge.start, ray, edge.plane) < 0
                        or _arc_order(ray, edge.end, edge.plane) < 0 for ray in rays):
                    continue
                if _arc_order(rays[0], rays[1], edge.plane) < 0:
                    rays.reverse()
                points = tuple(tuple(_rounded_interval(value, precision+24)
                    for value in _unit_ray(ray, precision)) for ray in rays)
                other_donor = tuple(plane for plane in dg.support_planes if not _same_plane(plane, edge.plane))
                other_receiver = tuple(plane for plane in rg.support_planes if not _same_plane(plane, edge.plane))
                if any(_dot(_plane_values(plane, precision)[1], point).lo <= 0
                        for plane in other_donor+other_receiver for point in points):
                    continue
                flux = tuple(_rounded_interval(_dot(normal, _cross(difference, point)), precision)
                    for point in points)
                if any(value.lo <= 0 for value in flux):
                    continue
                interior = tuple(_rounded_interval((a+b)/2, precision+24) for a, b in zip(*points))
                donor_bounds = tuple((index, _rounded_interval(_dot(_plane_values(plane, precision)[1], interior), precision))
                    for index, plane in enumerate(dg.support_planes) if not _same_plane(plane, edge.plane))
                receiver_bounds = tuple((index, _rounded_interval(_dot(_plane_values(plane, precision)[1], interior), precision))
                    for index, plane in enumerate(rg.support_planes) if not _same_plane(plane, edge.plane))
                incoming = _rounded_interval(_dot(normal, _cross(difference, interior)), precision)
                subarc = NativeArc(rays[0], rays[1], edge.plane, _arc_length(*rays))
                witness = NativeImplicitImmediateIncomingEdge(subarc, inward, points, flux,
                    interior, donor_bounds, receiver_bounds, incoming)
                pulled = tuple(ray.rotated(receiver_inverse) for ray in rays)
                receiver_coordinates = tuple(_ray_values(ray, precision)[1] for ray in pulled)
                ratios = tuple(tuple(_rounded_interval(value/point[2], precision) for value in point)
                    for point in receiver_coordinates)
                body = dict(format="native-cold5-cold1-immediate-incoming-slice-1",
                    source_epoch_sha256=epoch.epoch_sha256,
                    complete_source_epoch_fingerprint_sha256=_portable_fingerprint_sha256(epoch),
                    donor_component_id=donor.component_id, receiver_component_id=receiver.component_id,
                    omega_rad_per_myr=[[owner, list(vector)] for owner, vector in omega_rad_per_myr],
                    dt_myr=dt_myr, donor_piece_sequence=di, receiver_piece_sequence=ri,
                    original_donor_y_over_z_cuts=[_r(value) for value in original_y_over_z],
                    immediate_incoming_edge=_edge_body(witness),
                    receiver_original_pullback_rays=[native_ray_to_dict(ray) for ray in pulled],
                    receiver_original_gnomonic_coordinate_bounds=[[_i(value) for value in row] for row in ratios],
                    first_entry_clock_is_current_epoch_boundary=True,
                    first_entry_absolute_time_myr=_r(epoch.time_myr),
                    contact_exists_at_arbitrarily_small_positive_time=True,
                    certified_lower_material_mask_used=True,
                    negative_curved_correction_retained=True,
                    complete_incoming_arc_proved=False, spatial_memory_admission_proved=False,
                    force_admission_authorized=False, provenance=provenance)
                return json.loads(_encoded(dict(body, proof_sha256=_hash(body))))
    raise UnresolvedNativePairExclusion("No source-authenticated positive-flux cold5/receiver1 witness slice was proved")
