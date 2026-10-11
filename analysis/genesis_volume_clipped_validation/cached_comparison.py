"""Small immutable operands for the unchanged finite-order comparisons.

Point projections and declared work probes are calculated once by the frozen
public production functions.  This adapter retains their native LD values and
budgets verbatim, then applies the same component/probe comparison arithmetic.
It neither retains a domain/operator point table nor reconstructs a rounded
Gram, enlarges a budget, or changes any physical admission flag.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json

import numpy as np

from tectonics.genesis_volume_clipped_bulk import (
    ClippedBulkRefusal, _agreement, _plain, _precision_digest, _seal,
    validate_clipped_bulk_operator,
)
from tectonics.genesis_volume_operator import _immutable

LD = np.longdouble
NAMES = ("stiffness_cross", "stiffness_enriched", "mass_cross", "mass_enriched", "enrichment_force_n")
A = np.array((0, 1, 2, 6, 7, 8, 12, 13, 14)); P = A+3
FORMAT = "genesis-clipped-native-comparison-operands-cache-0.1"


def _require(value, message):
    if not value:
        raise ClippedBulkRefusal(message)


def _json(value):
    return json.dumps(_plain(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _basis(value, name):
    raw = np.asarray(value)
    _require(raw.dtype.kind in "ifu" and raw.ndim == 2 and raw.shape[0] == 18
        and 1 <= raw.shape[1] <= 18 and np.isfinite(raw).all(), name+" declared basis differs")
    return _immutable(raw, LD)


@dataclass(frozen=True, slots=True)
class CachedClippedComparisonOperand:
    operator_binding_sha256: str
    source_section_binding_sha256: str
    seed_center_world_m: np.ndarray
    seed_radius_m: float
    orders: tuple
    source_fields_binding_sha256: str
    full_blocks: tuple
    projected_blocks: tuple
    added_basis: np.ndarray | None
    parent_basis: np.ndarray | None
    probes_json: bytes
    probe_results_json: bytes
    binding_sha256: str
    format: str = FORMAT


def _body(record):
    arrays = [("seed_center_world_m", record.seed_center_world_m)]
    for prefix, blocks in (("full", record.full_blocks), ("projected", record.projected_blocks)):
        for name, value, budget in blocks:
            arrays.extend(((prefix+"/"+name, value), (prefix+"/"+name+"/budget", budget)))
    for name in ("added_basis", "parent_basis"):
        value = getattr(record, name)
        if value is not None:
            arrays.append((name, value))
    return dict(format=record.format, operator_binding_sha256=record.operator_binding_sha256,
        source_section_binding_sha256=record.source_section_binding_sha256,
        source_fields_binding_sha256=record.source_fields_binding_sha256,
        seed_radius_m=record.seed_radius_m, orders=list(record.orders),
        exact_native_operands_sha256=_precision_digest(arrays),
        probes_sha256=hashlib.sha256(record.probes_json).hexdigest(),
        probe_results_sha256=hashlib.sha256(record.probe_results_json).hexdigest(),
        point_tables_retained=False, budgets_changed=False, available_for_creation_funding=False)


def validate_cached_comparison_operand(record):
    _require(type(record) is CachedClippedComparisonOperand and record.format == FORMAT,
             "Cached comparison operand format differs")
    _require(record.binding_sha256 == _seal(_body(record)), "Cached native comparison operand binding differs")
    return True


def prepare_cached_comparison_operand(op, projected_blocks_native, declared_probe_contractions, *,
        added_basis=None, parent_basis=None, probes=()):
    """Copy already computed public operands once, without retaining points."""
    validate_clipped_bulk_operator(op)
    full = tuple((name, _immutable(getattr(op, name), LD), _immutable(getattr(op, name+"_budget"), LD)) for name in NAMES)
    projected = ()
    W, V = None, None
    if added_basis is not None:
        W = _basis(added_basis, "Added")
        V = _basis(np.eye(18) if parent_basis is None else parent_basis, "Parent")
        _require(type(projected_blocks_native) is dict and projected_blocks_native.get("projection_before_Gram") is True
            and projected_blocks_native.get("physical_contraction_certified") is False,
            "Cached source projection must come from the public native point projection")
        rows = []
        for name in NAMES:
            value, budget = np.asarray(projected_blocks_native[name], LD), np.asarray(projected_blocks_native[name+"_budget"], LD)
            shape = ((W.shape[1],) if name == "enrichment_force_n" else
                     (V.shape[1], W.shape[1]) if name.endswith("cross") else (W.shape[1], W.shape[1]))
            _require(value.shape == budget.shape == shape and np.isfinite(value).all()
                     and np.isfinite(budget).all() and np.all(budget >= 0), "Cached projected component shape/budget differs")
            rows.append((name, _immutable(value, LD), _immutable(budget, LD)))
        projected = tuple(rows)
    else:
        _require(projected_blocks_native is None, "Undeclared cached projection is not allowed")
    probes = tuple(probes)
    rows = tuple(declared_probe_contractions)
    _require(len(rows) == len(probes), "Cached declared probe count differs")
    normalized = []
    for row, probe in zip(rows, probes):
        _require(isinstance(probe, dict) and isinstance(probe.get("name"), str)
            and row.get("name") == probe["name"], "Cached declared probe identity differs")
        result = row.get("contractions", row.get("result"))
        _require(type(result) is dict and result.get("projection_before_Gram") is True
            and result.get("physical_contraction_certified") is False
            and result.get("available_for_creation_funding") is False, "Cached probe public arithmetic/scope differs")
        normalized.append(dict(name=probe["name"], contractions=result))
    record = CachedClippedComparisonOperand(op.binding_sha256, op.domain.source_section_binding_sha256,
        _immutable(op.domain.seed_center_world_m, LD), op.domain.seed_radius_m,
        (op.domain.radial_order, op.domain.angular_order, op.domain.ray_order),
        _seal({name:getattr(op, name) for name in ("elasticity_pa", "basis_xyz", "memory0", "memory1")}),
        full, projected, W, V, _json(probes), _json(normalized), "")
    return replace(record, binding_sha256=_seal(_body(record)))


def compare_cached_clipped_orders(sequence, relative_tolerance=2e-6, probes=(), *,
        added_basis=None, parent_basis=None):
    """Return exactly the frozen comparator DTO using cached native operands."""
    sequence, probes = tuple(sequence), tuple(probes)
    _require(len(sequence) >= 2, "At least two declared finite orders are required")
    _require(not isinstance(relative_tolerance, (bool, np.bool_)), "Finite-order relative tolerance must be positive and finite")
    tolerance = LD(relative_tolerance)
    _require(np.isfinite(tolerance) and 0 < tolerance < 1, "Finite-order relative tolerance must be positive and finite")
    W = None if added_basis is None else _basis(added_basis, "Added")
    V = None if W is None else _basis(np.eye(18) if parent_basis is None else parent_basis, "Parent")
    for record in sequence:
        validate_cached_comparison_operand(record)
        _require(record.probes_json == _json(probes), "Cached probe definitions differ from the declared comparison")
        _require((record.added_basis is None) == (W is None), "Cached operative basis presence differs")
        if W is not None:
            _require(np.array_equal(record.added_basis, W) and np.array_equal(record.parent_basis, V),
                     "Cached operative source basis differs")
    first = sequence[0]
    for record in sequence[1:]:
        _require(record.source_section_binding_sha256 == first.source_section_binding_sha256
            and record.source_fields_binding_sha256 == first.source_fields_binding_sha256
            and record.seed_radius_m == first.seed_radius_m
            and np.array_equal(record.seed_center_world_m, first.seed_center_world_m),
            "Finite orders must share identical original geometry and source fields")
    pairs = []
    for lo, hi in zip(sequence, sequence[1:]):
        _require(all(h >= l for l, h in zip(lo.orders, hi.orders)) and hi.orders != lo.orders,
                 "Successive finite orders must increase at least one declared order")
        blocks, agreed = {}, True
        for (name, lv, lb), (other, hv, hb) in zip(lo.full_blocks, hi.full_blocks):
            _require(name == other, "Cached full component order differs")
            mask, report = _agreement(lv, hv, lb, hb, tolerance)
            if mask.ndim == 2:
                report["A_P_blocks"] = {label:dict(agreed=bool(np.all(mask[np.ix_(rows, cols)])),
                    failed_component_count=int(np.count_nonzero(~mask[np.ix_(rows, cols)])))
                    for label, rows, cols in (("AA", A, A), ("AP", A, P), ("PA", P, A), ("PP", P, P))}
            blocks[name] = report
            agreed = agreed and report["agreed"]
        projected_reports = {}
        for (name, lv, lb), (other, hv, hb) in zip(lo.projected_blocks, hi.projected_blocks):
            _require(name == other, "Cached projected component order differs")
            _, report = _agreement(lv, hv, lb, hb, tolerance)
            projected_reports[name] = report
            agreed = agreed and report["agreed"]
        probe_results = []
        lowrows, highrows = json.loads(lo.probe_results_json), json.loads(hi.probe_results_json)
        for lowrow, highrow in zip(lowrows, highrows):
            low, high = lowrow["contractions"], highrow["contractions"]
            term_reports = {}
            for term in low["terms"]:
                l, h = low["terms"][term], high["terms"][term]
                _, report = _agreement(LD(l["value"]), LD(h["value"]), LD(l["absolute_roundoff_budget"]), LD(h["absolute_roundoff_budget"]), tolerance)
                term_reports[term] = report
                agreed = agreed and report["agreed"]
            for force, budget in (("enrichment_force_n", "enrichment_force_budget_n"), ("added_parent_force_n", "added_parent_force_budget_n")):
                _, report = _agreement(np.asarray(low[force], LD), np.asarray(high[force], LD),
                    np.asarray(low[budget], LD), np.asarray(high[budget], LD), tolerance)
                term_reports[force] = report
                agreed = agreed and report["agreed"]
            probe_results.append(dict(name=lowrow["name"], contractions=term_reports))
        pairs.append(dict(low_orders=lo.orders, high_orders=hi.orders, finite_order_agreement=agreed,
            components=blocks, declared_projected_components=projected_reports, declared_probe_results=probe_results))
    body = _plain(dict(format="genesis-clipped-bulk-finite-order-comparison-0.1",
        scope="declared_finite_orders_own_component_and_probe_scales_no_rigorous_error_certificate",
        operator_bindings=[record.operator_binding_sha256 for record in sequence], relative_tolerance=tolerance,
        probes=probes, declared_added_basis=added_basis, declared_parent_basis=parent_basis,
        projection_before_Gram=W is not None, pair_results=pairs,
        finite_order_agreement=all(p["finite_order_agreement"] for p in pairs),
        geometry_resolved=True, arbitrary_contraction_certified=False, physical_contraction_certified=False,
        physical_creation_admitted=False, physical_release_admitted=False,
        available_for_creation_funding=False, roundoff_available_for_funding=False))
    return dict(body, binding_sha256=_seal(body))


def compare_cached_clipped_pair(previous, current, relative_tolerance=2e-6, probes=(), *,
        added_basis=None, parent_basis=None):
    return compare_cached_clipped_orders((previous, current), relative_tolerance, probes,
        added_basis=added_basis, parent_basis=parent_basis)
