"""Strict bounded angular observations with source integral uncertainties.

The callback supplies a represented value and a proven absolute distance to
the literal-source radial integral at that same represented material N.
Source uncertainty is paid from each component's original relative tolerance;
it never enlarges the acceptance allowance.  Embedded Gauss differences and
the shared candidate-cell geometry remain observations, not angular proofs.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from analysis.genesis_volume_clipped_validation.independent_geometry_audit import (
    read_source, _angular_support, _cell_bounds, _children, _triangle_area,
)
from tectonics.genesis_volume_operand_bounds import (
    DirectedInterval, point, add, subtract, multiply,
)

LD = np.longdouble
VOLUME_NAMES = ("reference_volume_negative_m3", "reference_volume_positive_m3")


class AngularControlRefusal(ValueError):
    pass


class NumericalLineBoundRefusal(AngularControlRefusal):
    """A source-bound line could not be measured; it must never become zero."""

    def __init__(self, reason, *, barycentric=None, completed_line_evaluations=0,
                 attempted_line_evaluations=0):
        super().__init__(str(reason))
        self.reason = str(reason)
        self.barycentric = None if barycentric is None else np.array(barycentric, LD, copy=True)
        self.completed_line_evaluations = int(completed_line_evaluations)
        self.attempted_line_evaluations = int(attempted_line_evaluations)

    def to_dict(self):
        n = None if self.barycentric is None else [
            [str(a), str(b)] for a, b in (LD(v).as_integer_ratio() for v in self.barycentric)]
        return dict(status="OPEN", reason=self.reason,
                    offending_barycentric_exact_binary_ratios=n,
                    completed_source_bound_line_evaluations=self.completed_line_evaluations,
                    attempted_line_evaluations=self.attempted_line_evaluations,
                    unmeasured_line_replaced_with_zero=False,
                    angular_accuracy_certified=False, physical_contraction_certified=False)


def _abs_interval(value):
    lo, hi = value.lower, value.upper
    upper = np.maximum(np.abs(lo), np.abs(hi))
    lower = np.where((lo <= 0) & (hi >= 0), LD(0), np.minimum(np.abs(lo), np.abs(hi)))
    return DirectedInterval(lower, upper)


def _triangle_jacobian(vertices):
    uv = point(np.asarray(vertices, LD)[:, 1:])
    a, b = subtract(uv[1], uv[0]), subtract(uv[2], uv[0])
    return _abs_interval(subtract(multiply(a[0], b[1]), multiply(a[1], b[0])))


def _sum_enclosures(values, shape):
    total = point(np.zeros(shape, LD))
    for value in values:
        total = add(total, value)
    return total


def _strict_allowance(tolerance, coarse, fine):
    scale = np.maximum(np.abs(coarse), np.abs(fine))
    return multiply(point(LD(tolerance)), point(scale)).lower


@dataclass
class _Cell:
    vertices: np.ndarray
    depth: int
    coarse: np.ndarray
    fine: np.ndarray
    coarse_enclosure: DirectedInterval
    fine_enclosure: DirectedInterval
    error: np.ndarray
    lower_volume: np.ndarray
    upper_volume: np.ndarray
    missing: np.ndarray


@dataclass
class AngularResult:
    names: tuple[str, ...]
    coarse: np.ndarray
    fine: np.ndarray
    coarse_roundoff: np.ndarray
    fine_roundoff: np.ndarray
    embedded_error: np.ndarray
    allowance: np.ndarray
    combined_error: np.ndarray
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
        failed = self.combined_error > self.allowance
        ratios = np.divide(self.combined_error, self.allowance,
                           out=np.full_like(self.allowance, np.inf), where=self.allowance != 0)
        ratios[(self.combined_error == 0) & (self.allowance == 0)] = 0
        worst = int(np.argmax(ratios)) if len(ratios) else 0
        finite = not np.any(failed) and self.unresolved_cells == 0
        worst_ratio = float(ratios[worst]) if self.names else None
        result = dict(
            format="genesis-source-integral-strict-angular-control-0.1",
            scope="source_bound_sampled_lines_and_embedded_angular_observation",
            status="FINITE_AGREEMENT" if finite else "OPEN", reason=self.reason,
            component_names=list(self.names), component_count=len(self.names),
            coarse_values=np.asarray(self.coarse, float).tolist(),
            fine_values=np.asarray(self.fine, float).tolist(),
            coarse_integral_arithmetic_budgets=np.asarray(self.coarse_roundoff, float).tolist(),
            fine_integral_arithmetic_budgets=np.asarray(self.fine_roundoff, float).tolist(),
            coarse_source_integral_and_aggregation_error_bounds=np.asarray(self.coarse_roundoff, float).tolist(),
            fine_source_integral_and_aggregation_error_bounds=np.asarray(self.fine_roundoff, float).tolist(),
            observed_embedded_errors=np.asarray(self.embedded_error, float).tolist(),
            combined_observed_and_source_error_bounds=np.asarray(self.combined_error, float).tolist(),
            original_own_relative_component_allowances=np.asarray(self.allowance, float).tolist(),
            component_allowances=np.asarray(self.allowance, float).tolist(),
            failed_component_count=int(np.count_nonzero(failed)),
            source_uncertainty_failed_component_count=int(np.count_nonzero(
                self.coarse_roundoff+self.fine_roundoff > self.allowance)),
            observed_angular_failed_component_count=int(np.count_nonzero(self.embedded_error > self.allowance)),
            worst_component=self.names[worst] if self.names else None,
            worst_error_ratio=worst_ratio if worst_ratio is None or np.isfinite(worst_ratio) else None,
            reference_side_volume_envelope_lower_m3=np.asarray(self.lower_volume, float).tolist(),
            reference_side_volume_envelope_upper_m3=np.asarray(self.upper_volume, float).tolist(),
            unresolved_positive_candidate_volume_m3=np.asarray(self.unresolved_volume, float).tolist(),
            evaluated_angular_cells=self.evaluated_cells, retained_angular_cells=self.active_cells,
            unresolved_positive_candidate_cells=self.unresolved_cells,
            radial_line_evaluations=self.line_evaluations, maximum_evaluated_depth=self.maximum_depth,
            angular_orders=list(self.angular_orders), max_cells=self.max_cells,
            max_depth=self.max_depth, max_line_evaluations=self.max_line_evaluations,
            relative_tolerance=self.relative_tolerance,
            finite_order_numerical_agreement=finite,
            acceptance_rule="observed_embedded_error_plus_coarse_fine_source_bounds_le_original_own_relative_allowance",
            source_bounds_added_to_acceptance_allowance=False,
            all_evaluated_lines_source_error_bound_supplied=True,
            source_bounds_aggregated_with_directed_arithmetic=True,
            enumerated_literal_native_material_nodes=True,
            gauss_nodes_and_weights_retained_as_literal_binary_operands=True,
            angular_node_mapping_accuracy_certified=False,
            original_material_support_conservatively_covered=True,
            unresolved_candidate_cells_never_discarded=True,
            angular_geometry_bounds_shared_with_previous_auditor=True,
            angular_candidate_geometry_newly_directed_certified=False,
            angular_refinement_priority="cell_combined_error_over_global_own_component_allowance",
            own_signed_component_relative_scale=True, positive_scale_floor_used=False,
            angular_accuracy_certified=False, geometry_operand_accuracy_certified=False,
            general_contraction_accuracy_certified=False, physical_source_selected=False,
            physical_creation_admitted=False, physical_release_admitted=False,
            available_for_creation_funding=False, accepted_steps=0, new_material_cohorts=0,
            funding_debit_j=0., automated_goal_enabled=False, point_one_complete=False,
        )
        if worst_ratio is not None and not np.isfinite(worst_ratio):
            result["worst_error_ratio_unbounded"] = True
        return result


def integrate_angular(raw, evaluator, component_names, *, relative_tolerance=2e-6,
                      max_cells=2000, max_depth=24, angular_orders=(3, 5),
                      max_line_evaluations=3400):
    """Bound native quadrature sums without turning uncertainties into credit.

    ``evaluator(N)`` returns ``(native_values, absolute_source_error_bounds)``.
    Its first two entries are negative/positive literal reference-side volumes.
    An evaluator unable to bound a node must raise NumericalLineBoundRefusal;
    it is propagated with that node and completed call count for a case OPEN.
    """
    source = read_source(raw)
    names = tuple(component_names)
    if len(names) < 2 or len(set(names)) != len(names) or names[:2] != VOLUME_NAMES:
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
    rules = {}
    for order in angular_orders:
        nodes, weights = np.polynomial.legendre.leggauss(order)
        rules[order] = (np.asarray((nodes+1)/2, LD), np.asarray(weights/2, LD))
    evaluated = line_count = completed = deepest = 0
    cells = []

    def rule(vertices, order):
        nonlocal line_count, completed
        nodes, weights = rules[order]
        value = np.zeros(shape, LD)
        enclosure = point(value)
        native_jacobian = 2*_triangle_area(vertices)
        jacobian = _triangle_jacobian(vertices)
        for s, sw in zip(nodes, weights):
            for t, tw in zip(nodes, weights):
                N = (1-s)*vertices[0]+s*(1-t)*vertices[1]+s*t*vertices[2]
                line_count += 1
                try:
                    raw_value, raw_bound = evaluator(N)
                except NumericalLineBoundRefusal as error:
                    raise NumericalLineBoundRefusal(error.reason, barycentric=N,
                        completed_line_evaluations=completed,
                        attempted_line_evaluations=line_count) from error
                v, b = np.asarray(raw_value, LD), np.asarray(raw_bound, LD)
                if (v.shape != shape or b.shape != shape or not np.isfinite(v).all()
                        or not np.isfinite(b).all() or np.any(b < 0) or np.any(v[:2] < 0)):
                    raise AngularControlRefusal("Invalid source-bound radial vector or absolute bound")
                completed += 1
                value += (native_jacobian*s*sw*tw)*v
                bounded_value = DirectedInterval(subtract(point(v), point(b)).lower,
                                                add(point(v), point(b)).upper)
                weight = multiply(multiply(multiply(jacobian, s), sw), tw)
                enclosure = add(enclosure, multiply(weight, bounded_value))
        return value, enclosure

    def can_add(count):
        return (evaluated+count <= max_cells and
                line_count+count*(low_order**2+high_order**2) <= max_line_evaluations)

    def add_cell(vertices, depth):
        nonlocal evaluated, deepest
        bounds = _cell_bounds(source, vertices)
        lo = np.array([bounds[s][0] for s in (-1, 1)], LD)
        hi = np.array([bounds[s][1] for s in (-1, 1)], LD)
        if not np.any(hi):
            return
        coarse, cb = rule(vertices, low_order)
        fine, fb = rule(vertices, high_order)
        observed = multiply(_abs_interval(subtract(point(fine), point(coarse))), 8).upper
        missing = np.where((fine[:2] == 0) & (hi > 0), hi, 0)
        observed = np.array(observed, copy=True)
        observed[:2] = np.maximum(observed[:2], missing)
        cells.append(_Cell(vertices, depth, coarse, fine, cb, fb, observed, lo, hi, missing))
        evaluated += 1
        deepest = max(deepest, depth)

    def reduce_cells():
        if not cells:
            zeros = np.zeros(shape, LD)
            return (zeros.copy(), zeros.copy(), zeros.copy(), zeros.copy(), zeros.copy(),
                    zeros.copy(), zeros.copy(), np.zeros(2, LD), np.zeros(2, LD), np.zeros(2, LD), 0)
        coarse = np.sum([c.coarse for c in cells], axis=0, dtype=LD)
        fine = np.sum([c.fine for c in cells], axis=0, dtype=LD)
        cb = _sum_enclosures((c.coarse_enclosure for c in cells), shape).absolute_error_for(coarse)
        fb = _sum_enclosures((c.fine_enclosure for c in cells), shape).absolute_error_for(fine)
        observed = _sum_enclosures((point(c.error) for c in cells), shape).upper
        combined = add(add(point(observed), point(cb)), point(fb)).upper
        allowance = _strict_allowance(relative_tolerance, coarse, fine)
        lo = np.sum([c.lower_volume for c in cells], axis=0, dtype=LD)
        hi = np.sum([c.upper_volume for c in cells], axis=0, dtype=LD)
        missing = np.sum([c.missing for c in cells], axis=0, dtype=LD)
        missing_count = sum(int(np.any(c.missing)) for c in cells)
        return coarse, fine, cb, fb, observed, combined, allowance, lo, hi, missing, missing_count

    support = _angular_support(source)
    if not can_add(len(support)):
        raise AngularControlRefusal("Resource cap cannot cover the initial conservative support")
    for vertices in support:
        add_cell(vertices, 0)
    reason = "No positive candidate support"
    while cells:
        arrays = reduce_cells()
        allowance, missing_count = arrays[6], arrays[10]
        if np.all(arrays[5] <= allowance) and missing_count == 0:
            reason = "All embedded angular observations and source uncertainties fit original own tolerances"
            break
        if not can_add(4):
            reason = "Predeclared angular cell or radial-line work budget exhausted"
            break
        best_index, best_key = 0, None
        for index, candidate in enumerate(cells):
            cb = candidate.coarse_enclosure.absolute_error_for(candidate.coarse)
            fb = candidate.fine_enclosure.absolute_error_for(candidate.fine)
            total = add(add(point(candidate.error), point(cb)), point(fb)).upper
            ratio = np.divide(total, allowance, out=np.zeros(shape, LD), where=allowance != 0)
            ratio[(allowance == 0) & (total > 0)] = np.inf
            key = (float(np.max(ratio)), float(np.sum(candidate.missing)), float(_triangle_area(candidate.vertices)))
            if best_key is None or key > best_key:
                best_index, best_key = index, key
        candidate = cells[best_index]
        if candidate.depth >= max_depth:
            reason = "Predeclared maximum angular depth reached with unresolved components"
            break
        cells.pop(best_index)
        for vertices in _children(candidate.vertices):
            add_cell(vertices, candidate.depth+1)
    coarse, fine, cb, fb, observed, combined, allowance, lo, hi, missing, missing_count = reduce_cells()
    return AngularResult(names, coarse, fine, cb, fb, observed, allowance, combined,
                         np.maximum(lo, 0), np.maximum(hi, 0), np.maximum(missing, 0),
                         evaluated, line_count, len(cells), missing_count, deepest,
                         reason, float(relative_tolerance), max_cells, max_depth,
                         max_line_evaluations, tuple(angular_orders))
