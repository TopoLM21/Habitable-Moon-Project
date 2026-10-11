"""Cached comparisons preserve the complete frozen comparator DTO."""
from dataclasses import fields, replace

import numpy as np
import pytest

from analysis.genesis_volume_clipped_validation.cached_comparison import (
    _body, compare_cached_clipped_orders, compare_cached_clipped_pair,
    prepare_cached_comparison_operand, validate_cached_comparison_operand,
)
from tectonics.genesis_volume_clipped_bulk import (
    ClippedBulkRefusal, _seal, clipped_bulk_contractions,
    clipped_bulk_projected_blocks, compare_clipped_bulk_orders,
)
from tectonics.genesis_volume_operator import _immutable
from test_genesis_volume_clipped_bulk import _op


@pytest.fixture(scope="module")
def operands():
    ops = (_op(order=3), _op(order=4), _op(order=5))
    W = np.eye(18)[:, (0, 3, 7, 10)]
    probes = (dict(name="full_mean", parent_increment_m=(1e-4*np.sin(.37*np.arange(18))).tolist(),
        enrichment_increment_m=(2e-4*np.cos(.23*np.arange(18))).tolist(),
        mean_e0=(ops[0].memory0*.7).tolist(), mean_e1=(ops[0].memory1*1.1).tolist(),
        mean_z_m=(3e-4*np.sin(.19*np.arange(18))).tolist()),
        dict(name="exact_zero", parent_increment_m=np.zeros(18).tolist(), enrichment_increment_m=np.zeros(18).tolist()))
    records = []
    for op in ops:
        projected = clipped_bulk_projected_blocks(op, W)
        rows = [dict(name=p["name"], contractions=clipped_bulk_contractions(op,
            p["parent_increment_m"], p["enrichment_increment_m"],
            **{k:p[k] for k in ("mean_e0", "mean_e1", "mean_z_m") if k in p})) for p in probes]
        records.append(prepare_cached_comparison_operand(op, projected, rows, added_basis=W, probes=probes))
    return ops, W, probes, records


def test_cached_full_native_projected_and_probe_DTO_is_exactly_equivalent(operands):
    ops, W, probes, records = operands
    original = compare_clipped_bulk_orders(ops, probes=probes, added_basis=W)
    cached = compare_cached_clipped_orders(records, probes=probes, added_basis=W)
    assert cached == original
    assert cached["binding_sha256"] == original["binding_sha256"]
    assert any(p["finite_order_agreement"] is False for p in cached["pair_results"])
    assert not cached["physical_contraction_certified"]


def test_neighbor_pair_DTO_matches_frozen_pair_without_retaining_old_points(operands):
    ops, W, probes, records = operands
    assert compare_cached_clipped_pair(records[1], records[2], probes=probes, added_basis=W) == \
        compare_clipped_bulk_orders(ops[1:], probes=probes, added_basis=W)
    record = records[-1]
    assert not any(f.name in {"domain", "operator", "parent_trace", "enrichment_trace"} for f in fields(record))
    assert sum(v.nbytes+b.nbytes for _, v, b in record.full_blocks+record.projected_blocks) < 50000


def test_full18_comparison_without_optional_projections_or_probes_is_exact(operands):
    ops, _, _, _ = operands
    records = [prepare_cached_comparison_operand(op, None, ()) for op in ops[:2]]
    assert compare_cached_clipped_orders(records) == compare_clipped_bulk_orders(ops[:2])


def test_native_LD_operands_and_budgets_are_verbatim_and_immutable(operands):
    ops, W, _, records = operands
    for op, record in zip(ops, records):
        for name, value, budget in record.full_blocks:
            np.testing.assert_array_equal(value, getattr(op, name))
            np.testing.assert_array_equal(budget, getattr(op, name+"_budget"))
            assert value.dtype == budget.dtype == np.dtype(np.longdouble)
            with pytest.raises(ValueError):
                value.setflags(write=True)
    changed = np.array(records[0].full_blocks[0][1], np.longdouble)
    changed[0, 0] = np.nextafter(changed[0, 0], np.longdouble(np.inf))
    blocks = ((records[0].full_blocks[0][0], _immutable(changed, np.longdouble), records[0].full_blocks[0][2]),)+records[0].full_blocks[1:]
    forged = replace(records[0], full_blocks=blocks)
    with pytest.raises(ClippedBulkRefusal, match="binding differs"):
        validate_cached_comparison_operand(forged)


def test_changed_declared_probe_or_basis_cannot_reuse_another_cached_result(operands):
    _, W, probes, records = operands
    wrong = [dict(p) for p in probes]
    wrong[0] = dict(wrong[0], mean_z_m=np.zeros(18).tolist())
    with pytest.raises(ClippedBulkRefusal, match="probe definitions differ"):
        compare_cached_clipped_orders(records, probes=wrong, added_basis=W)
    changed = W.copy(); changed[0, 0] *= 1.01
    with pytest.raises(ClippedBulkRefusal, match="source basis differs"):
        compare_cached_clipped_orders(records, probes=probes, added_basis=changed)


def test_zero_and_small_component_refusals_keep_their_own_scale(operands):
    _, W, probes, records = operands
    lo, hi = records[:2]
    # Resealing a deliberately changed cached work record is a negative
    # comparison test, not a claim that a producer emitted this result.
    import json
    rows = json.loads(hi.probe_results_json)
    rows[1]["contractions"]["terms"]["self_work_j"]["value"] = 1e-200
    modified = replace(hi, probe_results_json=json.dumps(rows, sort_keys=True, separators=(",", ":")).encode(), binding_sha256="")
    modified = replace(modified, binding_sha256=_seal(_body(modified)))
    result = compare_cached_clipped_pair(lo, modified, probes=probes, added_basis=W)
    zero = result["pair_results"][0]["declared_probe_results"][1]["contractions"]["self_work_j"]
    assert zero["agreed"] is False and zero["positive_scale_floor_used"] is False


def test_source_change_and_duplicate_orders_are_refused(operands):
    _, W, probes, records = operands
    changed = replace(records[1], source_fields_binding_sha256="changed", binding_sha256="")
    changed = replace(changed, binding_sha256=_seal(_body(changed)))
    with pytest.raises(ClippedBulkRefusal, match="identical original geometry"):
        compare_cached_clipped_pair(records[0], changed, probes=probes, added_basis=W)
    with pytest.raises(ClippedBulkRefusal, match="increase at least one"):
        compare_cached_clipped_pair(records[0], records[0], probes=probes, added_basis=W)


def test_sub_float64_native_basis_change_is_bound_exactly(operands):
    _, _, _, records = operands
    record = records[0]
    changed = np.array(record.added_basis, np.longdouble)
    changed[0, 0] += 4*np.finfo(np.longdouble).eps
    assert float(changed[0, 0]) == float(record.added_basis[0, 0])
    forged = replace(record, added_basis=_immutable(changed, np.longdouble))
    with pytest.raises(ClippedBulkRefusal, match="binding differs"):
        validate_cached_comparison_operand(forged)
