"""Independent stdlib SO(3), scalar-clock and signed-mask curve audit.

The original complete first-entry certificate and actual force packet are
independently supplied authority.  This audit rebuilds ordered old poses,
fresh relative Taylor coefficients and their analytic remainders.  It then
checks the outside-endpoint premise and whole-time derivative.  Raw HOT proxy
piece proofs require the companion generic scalar audit; this receipt makes
no independent full-pair or mechanical-admission claim.
"""
from __future__ import annotations

from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
from math import factorial


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()


def _r(value):
    if (type(value) is not list or len(value) != 2 or any(type(x) is not int for x in value)
            or value[1] <= 0):
        raise ValueError("Audit rational is not a literal positive-denominator integer pair")
    result = F(*value)
    if [result.numerator, result.denominator] != value:
        raise ValueError("Audit rational is not canonical")
    return result


def _i(value):
    if type(value) is not dict or set(value) != {"lo", "hi"}:
        raise ValueError("Audit interval requires its complete canonical endpoints")
    result = (_r(value["lo"]), _r(value["hi"]))
    if result[0] > result[1]:
        raise ValueError("Audit interval is unordered")
    return result


def _add(a, b):
    return (a[0]+b[0], a[1]+b[1])


def _neg(a):
    return (-a[1], -a[0])


def _mul(a, b):
    products = tuple(x*y for x in a for y in b)
    return (min(products), max(products))


def _sub(a, b):
    return _add(a, _neg(b))


def _scale(a, value):
    return _mul(a, (value, value))


def _sum(values):
    result = (F(0), F(0))
    for value in values:
        result = _add(result, value)
    return result


def _contains(outer, inner):
    return outer[0] <= inner[0] <= inner[1] <= outer[1]


def _compress(value, bits=1312):
    """Exact outward relative dyadic compression, without integer-string IO."""
    maximum = max(map(abs, value))
    if not maximum:
        return value
    exponent = maximum.numerator.bit_length()-maximum.denominator.bit_length()
    if exponent >= 0:
        below = maximum.numerator < maximum.denominator << exponent
    else:
        below = maximum.numerator << -exponent < maximum.denominator
    exponent -= int(below)
    shift = bits-exponent
    grid = F(1 << shift) if shift >= 0 else F(1, 1 << -shift)
    low, high = value[0]*grid, value[1]*grid
    return (F(low.numerator//low.denominator)/grid,
        F(-((-high.numerator)//high.denominator))/grid)


def _cross(vector, intervals):
    return tuple(_compress(_sub(_scale(intervals[k], vector[j]), _scale(intervals[j], vector[k])))
        for j, k in ((1, 2), (2, 0), (0, 1)))


def _dot(left, right):
    return _compress(_sum(_mul(a, b) for a, b in zip(left, right)))


def _vector(raw):
    if (type(raw) is not list or len(raw) != 3
            or any(type(x) is not float or not math.isfinite(x) for x in raw)):
        raise ValueError("Audit needs all actual literal binary64 Euler components")
    return tuple(F.from_float(x) for x in raw)


def _rotation(vector, seed, *, precision=320):
    """Enclose exp([vector]cross) seed with an independent vector series."""
    rate = sum(map(abs, vector), F(0))
    if rate >= F(1, 4):
        raise ValueError("Independent pose audit requires the declared small-rotation domain")
    scale = sum((max(map(abs, row)) for row in seed), F(0))
    term, total = seed, seed
    for degree in range(1, 193):
        term = tuple(_compress(_scale(row, F(1, degree))) for row in _cross(vector, term))
        total = tuple(_compress(_add(a, b)) for a, b in zip(total, term))
        tail = scale*rate**(degree+1)/factorial(degree+1)/(1-rate/F(degree+2))
        if tail <= F(1, 10**precision):
            return tuple(_compress(_add(row, (-tail, tail))) for row in total)
    raise ValueError("Independent pose audit exhausted its explicit 192-term budget")


def _pose(history, seed):
    result = seed
    for step in history:
        if type(step) is not dict or set(step) != {"omega_rad_per_myr", "dt_myr"}:
            raise ValueError("Independent pose audit requires the entire ordered saved rotation")
        if type(step["dt_myr"]) is not float or not math.isfinite(step["dt_myr"]):
            raise ValueError("Independent pose clock must be the literal finite physical duration")
        duration = F.from_float(step["dt_myr"])
        vector = tuple(x*duration for x in _vector(step["omega_rad_per_myr"]))
        columns = tuple(_rotation(vector, tuple((F(int(i == j)), F(int(i == j))) for i in range(3)))
            for j in range(3))
        # Map each fixed chart coordinate once through the enclosed proper
        # matrix; reusing a chart interval in every vector Taylor term would
        # unnecessarily lose cancellation between its alternating terms.
        result = tuple(_compress(_sum(_mul(columns[j][i], result[j]) for j in range(3))) for i in range(3))
    return result


def _relative_coefficients(normal, donor, receiver, order):
    receiver_rows = [normal]
    for degree in range(1, order+1):
        receiver_rows.append(tuple(_compress(_scale(value, F(1, degree)))
            for value in _cross(receiver, receiver_rows[-1])))
    result = [[(F(0), F(0)) for _ in range(3)] for _ in range(order+1)]
    for first_degree, row in enumerate(receiver_rows):
        term = row
        for donor_degree in range(order-first_degree+1):
            if donor_degree:
                term = tuple(_compress(_scale(value, F(1, donor_degree)))
                    for value in _cross(tuple(-x for x in donor), term))
            degree = first_degree+donor_degree
            result[degree] = [_compress(_add(a, b)) for a, b in zip(result[degree], term)]
    return tuple(tuple(row) for row in result)


def _horner(coefficients):
    value = (F(0), F(0))
    for row in reversed(coefficients):
        value = _add(_mul(value, (F(0), F(1))), row)
    return value


def audit_curved_pair(proof_path, *, source_epoch_payload, candidate_payload,
        output_path=None):
    """Audit one saved curve proof against externally authenticated inputs."""
    proof = json.loads(Path(proof_path).read_text())
    if proof.get("format") != "native-signed-curved-birth-pair-exclusion-1":
        raise ValueError("Unexpected curved proof format")
    source = source_epoch_payload
    if proof["source_epoch_sha256"] != source["epoch_sha256"]:
        raise ValueError("Curve proof refers to another complete source epoch")
    if _encoded(proof["omega_rad_per_myr"]) != _encoded(candidate_payload["omega_rad_per_myr"]):
        raise ValueError("Curve proof does not use the actual current candidate Euler vectors")
    expected_dt = candidate_payload["dt_s"]/31557600000000.
    if type(proof["dt_myr"]) is not float or proof["dt_myr"] != expected_dt:
        raise ValueError("Curve proof is not on the current literal physical interval")
    order = proof["order"]
    if type(order) is not int or not 12 <= order <= 40:
        raise ValueError("Curve audit order is not within its explicit budget")
    motions = {owner: _vector(vector) for owner, vector in proof["omega_rad_per_myr"]}
    duration = F.from_float(proof["dt_myr"])
    donor, receiver = tuple(x*duration for x in motions[1]), tuple(x*duration for x in motions[0])
    rate = sum(map(abs, donor), F(0))+sum(map(abs, receiver), F(0))
    if rate >= F(1, 4):
        raise ValueError("Curve relative motion exceeds its declared complete-series domain")
    certificate = source["source_material_state"]["sweep_certificate"]
    envelope = certificate["envelope"]
    original = ((F(0), _i(envelope["terminal_height_bounds"])[1]),
        (-_i(envelope["maximum_ratio_bounds"])[1], -_i(envelope["terminal_slope_bounds"])[0]), (F(1), F(1)))
    if tuple(_i(row) for row in proof["original_gnomonic_chart_bounds"]) != original:
        raise ValueError("Curved enclosing chart does not come from the trusted complete envelope")
    old_motions = dict(certificate["journal"]["omega_rad_per_myr"])
    additional = dict(source["additional_owner_histories"])
    histories = {owner: [dict(omega_rad_per_myr=old_motions[owner], dt_myr=certificate["journal"]["dt_myr"])]+additional[owner]
        for owner in (0, 1)}
    if _encoded(additional[0]) != _encoded(additional[1]):
        raise ValueError("Curve source does not preserve one exact complete partition")
    corrections = [row for row in source["paired_corrections"] if row["correction_id"] == proof["signed_correction_id"]]
    if len(corrections) != 1:
        raise ValueError("The actual signed correction is absent or duplicated")
    correction = corrections[0]
    for field in ("negative_component_id", "positive_component_id"):
        if correction[field] != proof[field]:
            raise ValueError("A signed material component was replaced")
    if _i(proof["positive_curved_measure_bounds_sr"]) != _i(correction["area_bounds_sr"]):
        raise ValueError("The positive curved material measure was changed")
    if _i(correction["area_bounds_sr"])[0] <= 0:
        raise ValueError("The source correction has no proven positive curved measure")
    rebuilt_chart = _pose(histories[1], original)
    observations = []
    for piece in proof["curved_pieces"]:
        world_chart = tuple(_i(row) for row in piece["world_chart_bounds"])
        if not all(_contains(saved, actual) for saved, actual in zip(world_chart, rebuilt_chart)):
            raise ValueError("Saved world chart does not contain the independent full old pose")
        plane = piece["receiver_inward_plane"]
        if any(type(x) is not int for x in plane["normal"]):
            raise ValueError("Curve receiver plane normal requires literal exact integer components")
        normal = _pose(plane["rotations"], tuple((F(x), F(x)) for x in plane["normal"]))
        rebuilt = _relative_coefficients(normal, donor, receiver, order)
        coefficients = tuple(_i(row) for row in piece["scalar_coefficients"])
        if len(coefficients) != order+1:
            raise ValueError("Saved scalar series dropped a full-clock coefficient")
        for degree, (saved, actual_vector) in enumerate(zip(coefficients, rebuilt)):
            if not _contains(saved, _dot(actual_vector, world_chart)):
                raise ValueError("Scalar Taylor coefficient differs from independent SO(3) source reconstruction at "+str(degree))
        scale = sum((max(map(abs, row)) for row in normal), F(0))
        chart_scale = sum((max(map(abs, row)) for row in world_chart), F(0))
        required_tail = scale*chart_scale*rate**(order+1)/factorial(order+1)/(1-rate/F(order+2))
        required_derivative_tail = scale*chart_scale*rate**(order+1)/factorial(order)/(1-rate/F(order+1))
        tail, derivative_tail = _r(piece["scalar_remainder_bound"]), _r(piece["derivative_remainder_bound"])
        if tail < required_tail or derivative_tail < required_derivative_tail:
            raise ValueError("Saved analytic value or derivative tail is not conservative")
        value = _add(_horner(coefficients), (-tail, tail))
        derivative = _add(_horner(tuple(_scale(row, F(index)) for index, row in enumerate(coefficients) if index)),
            (-derivative_tail, derivative_tail))
        if not _contains(_i(piece["whole_time_support_bounds"]), value):
            raise ValueError("Full-clock value bound does not contain independent interval arithmetic")
        if not _contains(_i(piece["whole_time_derivative_bounds"]), derivative):
            raise ValueError("Full-clock derivative bound does not contain independent interval arithmetic")
        method = piece["method"]
        if method == "whole_chart_nonpositive_support":
            if value[1] > 0:
                raise ValueError("The whole chart is not outside its receiving halfspace")
        elif method == "exact_outside_endpoint_archive_and_negative_derivative":
            expected_plane = dict(normal=[-1, 0, 0], rotations=histories[0], label="")
            if _encoded(plane) != _encoded(expected_plane) or _encoded(piece["endpoint_outside_plane"]) != _encoded(expected_plane):
                raise ValueError("The mask premise lost its oriented actual endpoint receiver halfspace")
            if piece["initial_support_constraint"] != "signed_correction_equals_complete_swept_archive_minus_endpoint_archive":
                raise ValueError("The exact initial mask premise was replaced")
            if _i(piece["initial_support_bounds"])[1] != 0 or derivative[1] >= 0:
                raise ValueError("The exact outside premise is not preserved by a strictly negative full-time derivative")
        else:
            raise ValueError("Unsupported signed curve separation method")
        if _i(piece["physical_fractional_clock"]) != (F(0), F(1)):
            raise ValueError("Curve proof no longer covers the complete current physical clock")
        observations.append(dict(receiver_piece_sequence=piece["receiver_piece_sequence"],
            independently_reconstructed_scalar_coefficients=len(coefficients), method=method,
            whole_time_signed_curve_exclusion_proved=True))
    if not observations:
        raise ValueError("Curve proof has no actual receiver-piece exclusion")
    for key in ("positive_curved_correction_deleted", "negative_curved_correction_deleted",
            "curved_area_assigned_to_polygon", "force_admission_authorized"):
        if proof[key] is not False:
            raise ValueError("The saved curve source or admission scope was changed")
    body = dict(format="independent-stdlib-signed-curved-pair-audit-1",
        source_epoch_sha256=source["epoch_sha256"],
        actual_candidate_sha256=candidate_payload["candidate_sha256"],
        proof_raw_sha256=hashlib.sha256(Path(proof_path).read_bytes()).hexdigest(),
        proof_sha256=proof["proof_sha256"], observations=observations,
        ordered_old_pose_independently_reconstructed=True,
        full_current_relative_so3_coefficients_independently_reconstructed=True,
        analytic_value_and_derivative_tails_independently_checked=True,
        exact_signed_mask_premise_and_orientation_checked=True,
        whole_signed_curved_component_exclusion_independently_proved=True,
        original_full_first_entry_certificate_is_external_authenticated_authority=True,
        original_curve_area_reintegration_performed=False,
        raw_hot_proxy_exclusion_requires_companion_scalar_audit=True,
        whole_material_pair_exclusion_independently_reproved=False,
        force_admission_authorized=False)
    body["receipt_sha256"] = hashlib.sha256(_encoded(body)).hexdigest()
    if output_path is not None:
        with Path(output_path).open("xb") as stream:
            stream.write(_encoded(body)+b"\n")
    return body
