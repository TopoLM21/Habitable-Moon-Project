"""Bounded angular checks with exact radial evaluators and explicit refusals.

The immutable clipped-domain auditor supplies conservative angular support
bounds.  Sharing these bounds does not make this a second independent
geometry oracle.  Embedded angular differences are numerical evidence;
they are never a rigorous error enclosure or a physical admission.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import numpy as np

from analysis.genesis_volume_clipped_validation.independent_geometry_audit import (
    read_source, _angular_support, _cell_bounds, _children, _triangle_area,
)

LD = np.longdouble


class AngularControlRefusal(ValueError):
    pass


def _gamma(count):
    product = LD(count)*np.finfo(LD).eps
    if product >= 1:
        raise AngularControlRefusal("Angular arithmetic bound is unresolved")
    return product/(1-product)


@dataclass
class AngularResult:
    names: tuple[str, ...]
    coarse: np.ndarray
    fine: np.ndarray
    coarse_roundoff: np.ndarray
    fine_roundoff: np.ndarray
    embedded_error: np.ndarray
    allowance: np.ndarray
    lower_volume: np.ndarray
    upper_volume: np.ndarray
    unresolved_volume: np.ndarray
    evaluated_cells: int
    line_evaluations: int
    active_cells: int
    unresolved_cells: int
    maximum_depth: int
    reason: str
    relative_tolerance: float
    max_cells: int
    max_depth: int
    max_line_evaluations: int
    angular_orders: tuple[int, int]

    def to_dict(self):
        failed = self.embedded_error > self.allowance
        ratios = np.divide(self.embedded_error, self.allowance,
                           out=np.full_like(self.allowance, np.inf),
                           where=self.allowance != 0)
        ratios[(self.embedded_error == 0) & (self.allowance == 0)] = 0
        worst = int(np.argmax(ratios)) if len(ratios) else 0
        converged = not np.any(failed) and self.unresolved_cells == 0
        result = dict(
            format="genesis-exact-lambda-bounded-angular-control-0.1",
            scope="embedded_angular_evidence_no_rigorous_or_physical_certificate",
            status="FINITE_AGREEMENT" if converged else "OPEN",
            reason=self.reason,
            component_names=list(self.names),
            coarse_values=np.asarray(self.coarse, float).tolist(),
            fine_values=np.asarray(self.fine, float).tolist(),
            coarse_integral_arithmetic_budgets=np.asarray(self.coarse_roundoff, float).tolist(),
            fine_integral_arithmetic_budgets=np.asarray(self.fine_roundoff, float).tolist(),
            observed_embedded_errors=np.asarray(self.embedded_error, float).tolist(),
            component_allowances=np.asarray(self.allowance, float).tolist(),
            failed_component_count=int(np.count_nonzero(failed)),
            component_count=len(self.names),
            worst_component=self.names[worst] if self.names else None,
            worst_error_ratio=float(ratios[worst]) if self.names else None,
            reference_side_volume_envelope_lower_m3=np.asarray(self.lower_volume, float).tolist(),
            reference_side_volume_envelope_upper_m3=np.asarray(self.upper_volume, float).tolist(),
            unresolved_positive_candidate_volume_m3=np.asarray(self.unresolved_volume, float).tolist(),
            evaluated_angular_cells=self.evaluated_cells,
            retained_angular_cells=self.active_cells,
            unresolved_positive_candidate_cells=self.unresolved_cells,
            radial_line_evaluations=self.line_evaluations,
            maximum_evaluated_depth=self.maximum_depth,
            angular_orders=list(self.angular_orders),
            max_cells=self.max_cells, max_depth=self.max_depth,
            max_line_evaluations=self.max_line_evaluations,
            relative_tolerance=self.relative_tolerance,
            finite_order_numerical_agreement=converged,
            original_material_support_conservatively_covered=True,
            unresolved_candidate_cells_never_discarded=True,
            angular_geometry_bounds_shared_with_previous_auditor=True,
            angular_refinement_priority="cell_error_over_global_own_component_allowance",
            own_signed_component_relative_scale=True,
            positive_scale_floor_used=False,
            geometry_operand_accuracy_certified=False,
            general_contraction_accuracy_certified=False,
            physical_source_selected=False,
            physical_creation_admitted=False,
            physical_release_admitted=False,
            available_for_creation_funding=False,
            accepted_steps=0, new_material_cohorts=0, funding_debit_j=0.,
            automated_goal_enabled=False, point_one_complete=False,
        )
        # JSON has no infinity literal.  An infinite ratio is an explicit
        # zero own-scale refusal, rather than a manufactured positive floor.
        if result["worst_error_ratio"] is not None and not np.isfinite(result["worst_error_ratio"]):
            result["worst_error_ratio"] = None
            result["worst_error_ratio_unbounded"] = True
        return result


def integrate_angular(raw, evaluator, component_names, *, relative_tolerance=2e-6,
                      max_cells=2000, max_depth=24, angular_orders=(3, 5),
                      max_line_evaluations=68000):
    """Integrate a fixed-N exact-line vector on conservative candidate cells.

    ``evaluator(N)`` returns ``(values, arithmetic_budgets)``; its first two
    values MUST be the negative/positive reference side volumes per unit
    angular area.  All subsequent values keep their declared own scales.
    A candidate with no sampled positive volume stays unresolved, even when
    its embedded numerical difference is exactly zero.
    """
    source = read_source(raw)
    names = tuple(component_names)
    if (len(names) < 2 or len(set(names)) != len(names) or
            names[:2] != ("reference_volume_negative_m3", "reference_volume_positive_m3")):
        raise AngularControlRefusal("First components must be the two literal reference-side volumes")
    if (isinstance(relative_tolerance, bool) or not np.isfinite(relative_tolerance)
            or not 0 < relative_tolerance < .1):
        raise AngularControlRefusal("Invalid fixed angular tolerance")
    low_order, high_order = angular_orders
    for value in (max_cells, max_depth, max_line_evaluations, low_order, high_order):
        if isinstance(value, bool) or int(value) != value or value < 1:
            raise AngularControlRefusal("Angular resource/order bounds must be positive integers")
    if low_order >= high_order:
        raise AngularControlRefusal("Angular orders must increase")
    shape = (len(names),)
    total_low, total_high, budget_low, budget_high, error, operands = (
        np.zeros(shape, LD) for _ in range(6))
    lower, upper, unresolved = (np.zeros(2, LD) for _ in range(3))
    sequence = evaluated = line_count = deepest = missing_cells = 0
    heap = []
    rules = {}
    for order in angular_orders:
        nodes, weights = np.polynomial.legendre.leggauss(order)
        rules[order] = (np.asarray((nodes+1)/2, LD), np.asarray(weights/2, LD))

    def rule(vertices, order):
        nonlocal line_count
        nodes, weights = rules[order]
        value, budget, own = (np.zeros(shape, LD) for _ in range(3))
        jacobian = 2*_triangle_area(vertices)
        for s, sw in zip(nodes, weights):
            for t, tw in zip(nodes, weights):
                N = (1-s)*vertices[0]+s*(1-t)*vertices[1]+s*t*vertices[2]
                raw_value, raw_budget = evaluator(N)
                line_count += 1
                v, b = np.asarray(raw_value, LD), np.asarray(raw_budget, LD)
                if (v.shape != shape or b.shape != shape or not np.isfinite(v).all()
                        or not np.isfinite(b).all() or np.any(b < 0) or np.any(v[:2] < 0)):
                    raise AngularControlRefusal("Invalid source-bound radial vector or arithmetic budget")
                weight = jacobian*s*sw*tw
                contribution = weight*v
                value += contribution
                own += np.abs(contribution)
                budget += weight*b
        budget += _gamma(order*order+12)*own
        return value, budget, own

    def can_add(count):
        return (evaluated+count <= max_cells and
                line_count+count*(low_order**2+high_order**2) <= max_line_evaluations)

    def add(vertices, depth):
        nonlocal sequence, evaluated, deepest, missing_cells
        bounds = _cell_bounds(source, vertices)
        lo = np.array([bounds[s][0] for s in (-1, 1)], LD)
        hi = np.array([bounds[s][1] for s in (-1, 1)], LD)
        if not np.any(hi):
            return
        coarse, cb, coarse_own = rule(vertices, low_order)
        fine, fb, fine_own = rule(vertices, high_order)
        # Each component's coarse reduction can be larger than its fine
        # reduction.  Both accumulate against their own shared upper bound.
        own = np.maximum(coarse_own, fine_own)
        local_error = 8*np.abs(fine-coarse)
        missing = np.where((fine[:2] == 0) & (hi > 0), hi, 0)
        local_error[:2] = np.maximum(local_error[:2], missing)
        total_low[:] += coarse; total_high[:] += fine
        budget_low[:] += cb; budget_high[:] += fb
        operands[:] += own; error[:] += local_error
        lower[:] += lo; upper[:] += hi; unresolved[:] += missing
        missing_cells += int(np.any(missing))
        local_scale = LD(relative_tolerance)*np.maximum(np.abs(coarse), np.abs(fine))+cb+fb
        ratio = np.divide(local_error, local_scale, out=np.zeros(shape, LD), where=local_scale != 0)
        ratio[(local_scale == 0) & (local_error > 0)] = np.inf
        priority = float(np.max(ratio))
        heapq.heappush(heap, (-priority, sequence, vertices, depth,
                             coarse, fine, cb, fb, own, local_error, lo, hi, missing))
        sequence += 1; evaluated += 1; deepest = max(deepest, depth)

    support = _angular_support(source)
    if not can_add(len(support)):
        raise AngularControlRefusal("Resource cap cannot cover the initial conservative support")
    for vertices in support:
        add(vertices, 0)
    reason = "No positive candidate support"
    while heap:
        reduction = _gamma(4*evaluated+32)*operands
        allowance = LD(relative_tolerance)*np.maximum(np.abs(total_low), np.abs(total_high))
        allowance += np.maximum(budget_low, 0)+np.maximum(budget_high, 0)+2*reduction
        if np.all(error <= allowance) and missing_cells == 0:
            # Re-reduce positive leaf errors before any agreement claim.
            # Subtraction during refinement must not erase a small residual.
            error[:] = np.sum([record[9] for record in heap], axis=0, dtype=LD)
        if np.all(error <= allowance) and missing_cells == 0:
            reason = "All declared embedded angular components agree on their own scales"
            break
        if not can_add(4):
            reason = "Predeclared angular cell or radial-line work budget exhausted"
            break
        # Refinement pays down each complete integral's error.  A local
        # relative ratio can keep refining a vanishing cell while larger
        # contributors elsewhere stay untouched.  Use the same GLOBAL own
        # allowance as the stopping check, without borrowing another scale.
        best_index, best_key = 0, None
        for index, candidate in enumerate(heap):
            candidate_error = candidate[9]
            ratio = np.divide(candidate_error, allowance, out=np.zeros(shape, LD), where=allowance != 0)
            ratio[(allowance == 0) & (candidate_error > 0)] = np.inf
            key = (float(np.max(ratio)), float(np.sum(candidate[12])), float(_triangle_area(candidate[2])))
            if best_key is None or key > best_key:
                best_index, best_key = index, key
        record = heap.pop(best_index)
        heapq.heapify(heap)
        _, _, vertices, depth, coarse, fine, cb, fb, own, local_error, lo, hi, missing = record
        if depth >= max_depth:
            heapq.heappush(heap, record)
            reason = "Predeclared maximum angular depth reached with unresolved components"
            break
        total_low[:] -= coarse; total_high[:] -= fine
        budget_low[:] -= cb; budget_high[:] -= fb; operands[:] -= own
        error[:] -= local_error; lower[:] -= lo; upper[:] -= hi; unresolved[:] -= missing
        missing_cells -= int(np.any(missing))
        for vertices in _children(vertices):
            add(vertices, depth+1)
    if heap:
        # The final public arrays come from retained leaves, including all
        # unsampled candidates; no clipped negative cancellation is evidence.
        records = list(heap)
        total_low = np.sum([r[4] for r in records], axis=0, dtype=LD)
        total_high = np.sum([r[5] for r in records], axis=0, dtype=LD)
        budget_low = np.sum([r[6] for r in records], axis=0, dtype=LD)
        budget_high = np.sum([r[7] for r in records], axis=0, dtype=LD)
        operands = np.sum([r[8] for r in records], axis=0, dtype=LD)
        error = np.sum([r[9] for r in records], axis=0, dtype=LD)
        lower = np.sum([r[10] for r in records], axis=0, dtype=LD)
        upper = np.sum([r[11] for r in records], axis=0, dtype=LD)
        unresolved = np.sum([r[12] for r in records], axis=0, dtype=LD)
        missing_cells = sum(int(np.any(r[12])) for r in records)
    reduction = _gamma(4*evaluated+32)*np.maximum(operands, 0)
    lb, hb = np.maximum(budget_low, 0)+reduction, np.maximum(budget_high, 0)+reduction
    allowance = LD(relative_tolerance)*np.maximum(np.abs(total_low), np.abs(total_high))+lb+hb
    return AngularResult(names, total_low, total_high, lb, hb, np.maximum(error, 0),
                         allowance, np.maximum(lower, 0), np.maximum(upper, 0),
                         np.maximum(unresolved, 0), evaluated, line_count, len(heap), missing_cells,
                         deepest, reason, float(relative_tolerance), max_cells, max_depth,
                         max_line_evaluations, tuple(angular_orders))
