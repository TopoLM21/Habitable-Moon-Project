"""Source-bound complete compact-jump kernel for computational release probes.

The bulk enrichment has eighteen coefficients, but a compact interface may
observe fewer directions.  Every direction in its COMPLETE weighted jump
kernel remains tied.  Rank is diagnosed after column scaling and is refused
when singular values lie in an unresolved band.  This preparation neither
admits a physical release nor provides fracture funding.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json
import math

import numpy as np


FORMAT = "genesis-compact-release-modes-0.1"
SCOPE = "source_bound_computational_complete_jump_kernel_no_physical_release"
_EPS = np.finfo(float).eps
_NDOF = 18


class CompactReleaseModesRefusal(ValueError):
    """The complete jump kernel or a proposed released increment is unresolved."""


def _immutable(value):
    a = np.asarray(value, dtype=float)
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


def _numeric(value, shape, name):
    try:
        original = np.asarray(value)
        if any(isinstance(item, (bool, np.bool_)) for item in np.asarray(value, dtype=object).flat):
            raise ValueError
        if original.dtype.kind not in "iuf" or original.dtype.kind == "b":
            raise ValueError
        a = np.asarray(value, dtype=float)
    except (ValueError, TypeError, OverflowError) as error:
        raise CompactReleaseModesRefusal(f"{name} must contain finite real values, without bool") from error
    if a.shape != shape or not np.isfinite(a).all():
        raise CompactReleaseModesRefusal(f"{name} shape or finite values mismatch")
    return a


def _positive(value, name, *, minimum=0.):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value,
            (int, float, np.integer, np.floating)):
        raise CompactReleaseModesRefusal(f"{name} must be a finite positive real")
    value = float(value)
    if not math.isfinite(value) or value <= minimum:
        raise CompactReleaseModesRefusal(f"{name} must be a finite positive real")
    return value


def _source_pin(value):
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise CompactReleaseModesRefusal("Source binding must be lowercase SHA256")
    return value


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    return value


def _canonical(value):
    return json.dumps(_plain(value), sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode()


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _stable_norm(a, axis=None):
    if axis is None:
        scale = float(np.max(np.abs(a), initial=0.))
        return 0. if scale == 0. else float(scale*np.linalg.norm(a/scale))
    scale = np.max(np.abs(a), axis=axis)
    divisor = np.where(scale == 0., 1., scale)
    return scale*np.linalg.norm(a/np.expand_dims(divisor, axis), axis=axis)


@dataclass(frozen=True, slots=True)
class CompactReleaseModes:
    trace_xyz: np.ndarray
    reference_area_weights_m2: np.ndarray
    source_binding_sha256: str
    analytic_null_modes: np.ndarray
    relative_rank_tolerance: float
    rank_separation_factor: float
    column_scales_m: np.ndarray
    normalized_singular_values: np.ndarray
    rank: int
    tied_null_modes: np.ndarray
    release_basis: np.ndarray
    rank_threshold: float
    retained_rank_margin: float | None
    rejected_rank_margin: float | None
    minimum_retained_weighted_singular_value_m: float | None
    retained_weighted_condition_number: float | None
    numerical_kernel_relative_residual: float
    weighted_kernel_relative_residual: float
    analytic_null_relative_residual: float
    analytic_null_subspace_error: float
    analytic_null_subspace_error_bound: float
    binding_sha256: str

    def __post_init__(self):
        for name in ("trace_xyz", "reference_area_weights_m2", "analytic_null_modes",
                "column_scales_m", "normalized_singular_values", "tied_null_modes", "release_basis"):
            object.__setattr__(self, name, _immutable(getattr(self, name)))
        _source_pin(self.source_binding_sha256)
        if type(self.rank) is not int or not 0 <= self.rank <= _NDOF:
            raise CompactReleaseModesRefusal("Compact release rank is invalid")
        if self.binding_sha256 != _digest(_body(self)):
            raise CompactReleaseModesRefusal("Compact release modes immutable binding differs")

    @property
    def nullity(self):
        return _NDOF-self.rank

    @property
    def physical_release_admitted(self):
        return False

    @property
    def funding_admitted(self):
        return False


def _body(value):
    return {"format": FORMAT, "scope": SCOPE,
        **{field.name: _plain(getattr(value, field.name)) for field in fields(CompactReleaseModes)
           if field.name != "binding_sha256"},
        "nullity": value.nullity, "physical_release_admitted": False,
        "funding_admitted": False, "complete_kernel_remains_tied": True,
        "analytic_witnesses_pinned_into_tied_kernel": value.analytic_null_modes.shape[1] > 0,
        "weighted_kernel_is_numerical_rank_diagnostic": True,
        "bulk_volume_rank_does_not_admit_surface_release": True}


def resolve_compact_release_modes(trace_xyz, reference_area_weights_m2,
        source_binding_sha256, *, analytic_null_modes=None,
        relative_rank_tolerance=1e-10, rank_separation_factor=10.):
    """Resolve all eighteen directions without adding a stiffness/weight floor.

    Exact zero columns enter the tied kernel directly.  For nonzero columns,
    singular values must lie above factor*tolerance or below tolerance/factor.
    The scaled right kernel is transformed back BEFORE Euclidean orthogonal
    completion, so unequal A/P column sizes do not change the tied subspace.
    """
    try:
        n = np.asarray(trace_xyz).shape[0]
    except (ValueError, TypeError, IndexError) as error:
        raise CompactReleaseModesRefusal("Compact trace shape is invalid") from error
    if n <= 0:
        raise CompactReleaseModesRefusal("Compact trace requires positive point count")
    trace = _numeric(trace_xyz, (n, 3, _NDOF), "Compact trace")
    weights = _numeric(reference_area_weights_m2, (n,), "Reference area weights")
    if np.any(weights < 0.) or not np.any(weights > 0.):
        raise CompactReleaseModesRefusal("Reference area weights must be nonnegative with positive total")
    pin = _source_pin(source_binding_sha256)
    tolerance = _positive(relative_rank_tolerance, "Relative rank tolerance")
    separation = _positive(rank_separation_factor, "Rank separation factor", minimum=1.)
    if tolerance/separation <= 64*_EPS:
        raise CompactReleaseModesRefusal("Relative rank separation lies below floating arithmetic resolution")
    if tolerance*separation >= 1.:
        raise CompactReleaseModesRefusal("Relative rank tolerance and separation leave no retained rank")
    if analytic_null_modes is None:
        analytic = np.empty((_NDOF, 0))
    else:
        try:
            k = np.asarray(analytic_null_modes).shape[1]
        except (ValueError, TypeError, IndexError) as error:
            raise CompactReleaseModesRefusal("Analytic null witness shape is invalid") from error
        if k > _NDOF:
            raise CompactReleaseModesRefusal("Too many analytic null witnesses")
        analytic = _numeric(analytic_null_modes, (_NDOF, k), "Analytic null witnesses")

    # Uniform scaling cancels from both rank and kernel.  Applying it BEFORE
    # square/norm avoids overflow while preserving each nonzero column size.
    positive = weights > 0.
    trace_scale = float(np.max(np.abs(trace[positive]), initial=0.))
    weight_scale = float(np.max(weights))
    if trace_scale == 0.:
        weighted = np.zeros((3*n, _NDOF))
    else:
        weighted_trace = np.zeros_like(trace)
        ratios = weights[positive]/weight_scale
        if np.any((ratios == 0.) & np.any(trace[positive] != 0., axis=(1, 2))):
            raise CompactReleaseModesRefusal("Positive reference weight scaling underflow is unresolved")
        weighted_trace[positive] = np.sqrt(ratios)[:, None, None]*(trace[positive]/trace_scale)
        weighted = weighted_trace.reshape(-1, _NDOF)
    norms = _stable_norm(weighted, axis=0)
    active = np.flatnonzero(norms > 0.)
    inactive = np.flatnonzero(norms == 0.)
    # Underflow must not silently turn a source-supported column into zero.
    genuinely_active = np.any(trace[positive] != 0., axis=(0, 1))
    if np.any(genuinely_active & (norms == 0.)):
        raise CompactReleaseModesRefusal("Weighted compact trace column underflow is unresolved")
    column_scales = norms*trace_scale*math.sqrt(weight_scale)
    if not np.isfinite(column_scales).all() or np.any((norms > 0.) & (column_scales == 0.)):
        raise CompactReleaseModesRefusal("Weighted compact column scale overflow/underflow is unresolved")
    singular = np.zeros(_NDOF)
    if len(active):
        scaled = weighted[:, active]/norms[active]
        _, values, vh = np.linalg.svd(scaled,
            full_matrices=scaled.shape[0] < scaled.shape[1])
        singular[:len(values)] = values
        threshold = tolerance*float(values[0])
        if np.any((values >= threshold/separation) & (values <= threshold*separation)):
            raise CompactReleaseModesRefusal("Compact jump rank is ambiguous inside the separation band")
        rank = int(np.count_nonzero(values > threshold*separation))
        raw_null = np.zeros((_NDOF, _NDOF-rank))
        if len(active) > rank:
            raw_null[active, :len(active)-rank] = vh[rank:].T/norms[active, None]
        raw_null[inactive, len(active)-rank:] = np.eye(len(inactive))
    else:
        threshold = 0.
        rank = 0
        raw_null = np.eye(_NDOF)

    if raw_null.shape[1]:
        column_lengths = _stable_norm(raw_null, axis=0)
        if np.any(column_lengths == 0.) or not np.isfinite(column_lengths).all():
            raise CompactReleaseModesRefusal("Back-transformed complete jump kernel is unresolved")
        normalized_null = raw_null/column_lengths
        condition_values = np.linalg.svd(normalized_null, compute_uv=False)
        if condition_values[-1] <= 256*_EPS*condition_values[0]:
            raise CompactReleaseModesRefusal("Back-transformed complete jump kernel is numerically dependent")
        complete, _ = np.linalg.qr(normalized_null, mode="complete")
        z = complete[:, :_NDOF-rank]
        b = complete[:, _NDOF-rank:]
    else:
        z, b = np.empty((_NDOF, 0)), np.eye(_NDOF)
    operator_norm = _stable_norm(weighted)
    numerical_residual = 0. if operator_norm == 0. else _stable_norm(weighted@z)/operator_norm
    if numerical_residual > tolerance/separation+256*_EPS:
        raise CompactReleaseModesRefusal("Complete tied kernel failed weighted compact-jump residual")

    physical_values = np.linalg.svd(weighted@b, compute_uv=False) if rank else np.empty(0)
    if rank and (not np.isfinite(physical_values).all() or physical_values[-1] <= 0.):
        raise CompactReleaseModesRefusal("Retained weighted jump singular values are unresolved")
    physical_minimum = None if rank == 0 else float(
        physical_values[-1]*trace_scale*math.sqrt(weight_scale))
    physical_condition = None if rank == 0 else float(physical_values[0]/physical_values[-1])
    if rank and (not math.isfinite(physical_minimum) or physical_minimum <= 0.
                 or not math.isfinite(physical_condition)):
        raise CompactReleaseModesRefusal("Retained weighted jump conditioning overflow/underflow is unresolved")

    analytic_residual = analytic_error = witness_bound = 0.
    if analytic.shape[1]:
        if analytic.shape[1] > _NDOF-rank:
            raise CompactReleaseModesRefusal("Analytic null witnesses exceed complete kernel dimension")
        analytic_lengths = _stable_norm(analytic, axis=0)
        if np.any(analytic_lengths == 0.):
            raise CompactReleaseModesRefusal("Analytic null witnesses must be nonzero")
        normalized_analytic = analytic/analytic_lengths
        witness_values = np.linalg.svd(normalized_analytic, compute_uv=False)
        if witness_values[-1] <= 256*_EPS*witness_values[0]:
            raise CompactReleaseModesRefusal("Analytic null witnesses must be independent")
        analytic_residual = 0. if operator_norm == 0. else _stable_norm(weighted@normalized_analytic)/operator_norm
        analytic_error = _stable_norm(b.T@normalized_analytic)
        if analytic_residual > 256*_EPS*_stable_norm(normalized_analytic):
            raise CompactReleaseModesRefusal("Analytic null witness is not in the weighted compact-jump kernel")
        # The subspace error is retained separately: conditioning of the
        # smallest retained jump direction amplifies source arithmetic error.
        if rank:
            witness_bound = (analytic_residual*operator_norm/physical_values[-1]
                             +256*_EPS)*math.sqrt(analytic.shape[1])
            if analytic_error > witness_bound:
                raise CompactReleaseModesRefusal("Analytic witnesses differ from the complete tied subspace")

        # Source-derived phi refinements are exact structural witnesses; tiny
        # A/P jump directions amplify roundoff in the unscaled SVD kernel.
        # Pin the verified witnesses, but use the FULL independently resolved
        # nullity to complete every remaining direction instead of assuming
        # that the supplied witnesses exhaust the kernel.
        witness_q, _ = np.linalg.qr(normalized_analytic, mode="complete")
        known = witness_q[:, :analytic.shape[1]]
        remaining_count = _NDOF-rank-analytic.shape[1]
        if remaining_count:
            projected = z-known@(known.T@z)
            completion, completion_values, _ = np.linalg.svd(projected, full_matrices=False)
            if completion_values[remaining_count-1] <= 256*_EPS:
                raise CompactReleaseModesRefusal("Analytic-pinned complete kernel completion is unresolved")
            candidate = np.column_stack((known, completion[:, :remaining_count]))
        else:
            candidate = known
        complete, _ = np.linalg.qr(candidate, mode="complete")
        z, b = complete[:, :_NDOF-rank], complete[:, _NDOF-rank:]
    residual = 0. if operator_norm == 0. else _stable_norm(weighted@z)/operator_norm
    if residual > tolerance/separation+256*_EPS:
        raise CompactReleaseModesRefusal("Analytic-pinned complete kernel failed weighted compact-jump residual")
    # Report conditioning of the final, source-witness-pinned release basis.
    if rank:
        final_values = np.linalg.svd(weighted@b, compute_uv=False)
        if not np.isfinite(final_values).all() or final_values[-1] <= 0.:
            raise CompactReleaseModesRefusal("Pinned release-basis jump conditioning is unresolved")
        physical_minimum = float(final_values[-1]*trace_scale*math.sqrt(weight_scale))
        physical_condition = float(final_values[0]/final_values[-1])
        if not math.isfinite(physical_minimum) or physical_minimum <= 0. or not math.isfinite(physical_condition):
            raise CompactReleaseModesRefusal("Pinned weighted jump conditioning overflow/underflow is unresolved")

    retained_margin = None if rank == 0 else float(singular[rank-1]/threshold)
    rejected_margin = None if rank == _NDOF or threshold == 0. else float(singular[rank]/threshold)
    arguments = dict(trace_xyz=trace, reference_area_weights_m2=weights,
        source_binding_sha256=pin, analytic_null_modes=analytic,
        relative_rank_tolerance=tolerance, rank_separation_factor=separation,
        column_scales_m=column_scales, normalized_singular_values=singular,
        rank=rank, tied_null_modes=z, release_basis=b, rank_threshold=threshold,
        retained_rank_margin=retained_margin, rejected_rank_margin=rejected_margin,
        minimum_retained_weighted_singular_value_m=physical_minimum,
        retained_weighted_condition_number=physical_condition,
        numerical_kernel_relative_residual=numerical_residual,
        weighted_kernel_relative_residual=residual,
        analytic_null_relative_residual=analytic_residual,
        analytic_null_subspace_error=analytic_error,
        analytic_null_subspace_error_bound=witness_bound)
    body = {"format": FORMAT, "scope": SCOPE, **_plain(arguments),
        "nullity": _NDOF-rank, "physical_release_admitted": False,
        "funding_admitted": False, "complete_kernel_remains_tied": True,
        "analytic_witnesses_pinned_into_tied_kernel": analytic.shape[1] > 0,
        "weighted_kernel_is_numerical_rank_diagnostic": True,
        "bulk_volume_rank_does_not_admit_surface_release": True}
    return CompactReleaseModes(**arguments, binding_sha256=_digest(body))


def validate_compact_release_modes(value):
    if type(value) is not CompactReleaseModes:
        raise CompactReleaseModesRefusal("Typed complete compact release modes are required")
    rebuilt = resolve_compact_release_modes(value.trace_xyz, value.reference_area_weights_m2,
        value.source_binding_sha256, analytic_null_modes=value.analytic_null_modes,
        relative_rank_tolerance=value.relative_rank_tolerance,
        rank_separation_factor=value.rank_separation_factor)
    if value.binding_sha256 != rebuilt.binding_sha256 or _canonical(_body(value)) != _canonical(_body(rebuilt)):
        raise CompactReleaseModesRefusal("Complete compact release modes differ from source reconstruction")
    return value


def validate_release_increment(value, increment_m, velocity_m_per_s=None):
    """Check that supplied vectors have no complete-kernel component; never project."""
    validate_compact_release_modes(value)
    for vector, name in ((increment_m, "Release increment"), (velocity_m_per_s, "Release velocity")):
        if vector is None:
            continue
        a = _numeric(vector, (_NDOF,), name)
        scale = float(np.max(np.abs(a), initial=0.))
        scaled = a if scale == 0. else a/scale
        norm = _stable_norm(scaled)
        leakage = _stable_norm(value.tied_null_modes.T@scaled)
        if leakage > 256*_EPS*norm:
            raise CompactReleaseModesRefusal(f"{name} contains a tied complete jump-kernel component")
    if increment_m is None:
        raise CompactReleaseModesRefusal("Release increment is required")


def compact_release_modes_to_dict(value):
    validate_compact_release_modes(value)
    return {**_body(value), "release_modes_sha256": value.binding_sha256}


def compact_release_modes_from_dict(raw, *, trace_xyz, reference_area_weights_m2,
        source_binding_sha256, analytic_null_modes=None, relative_rank_tolerance=1e-10,
        rank_separation_factor=10.):
    """Replay only from trusted source trace/weights, not the archive's own inputs."""
    expected = resolve_compact_release_modes(trace_xyz, reference_area_weights_m2,
        source_binding_sha256, analytic_null_modes=analytic_null_modes,
        relative_rank_tolerance=relative_rank_tolerance,
        rank_separation_factor=rank_separation_factor)
    try:
        equal = type(raw) is dict and _canonical(raw) == _canonical(
            {**_body(expected), "release_modes_sha256": expected.binding_sha256})
    except (ValueError, TypeError, OverflowError):
        equal = False
    if not equal:
        raise CompactReleaseModesRefusal("Saved compact release modes differ from trusted source replay")
    return expected
