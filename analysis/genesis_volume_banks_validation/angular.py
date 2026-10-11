"""Bank-specific bounds and event-guided mesh with exact source measure.

Adapted from the frozen strict angular control, retaining its (3,5) rules,
2e-6 own-component criterion, source-error accounting and work budgets.
Embedded differences are still observations, not angular error certificates.
"""
from fractions import Fraction as F
import numpy as np
from analysis.genesis_volume_integral_validation.angular_control import (
    AngularControlRefusal, NumericalLineBoundRefusal, AngularResult, _Cell,
    _abs_interval, _sum_enclosures, _strict_allowance, VOLUME_NAMES,
)
from tectonics.genesis_volume_operand_bounds import DirectedInterval,point,add,subtract,multiply,rational_enclosure
from tectonics.genesis_volume_rational_fields import rational,fraction
from tectonics.genesis_volume_source_coverage import source_coverage,exact_node
from tectonics.genesis_volume_event_partition import (
    partition_events,split_cell,signed_double_area,source_events,sign_certificate,radial_support_classification,
)
from tectonics.genesis_volume_bank_bounds import PreparedBankBounds
from tectonics.genesis_volume_event_mesh import event_mesh
from tectonics.genesis_volume_event_partition import ratio
LD=np.longdouble


def decode_cell(value):
    return tuple(tuple(F(int(a),int(b)) for a,b in row) for row in value)


def integrate_angular(raw, evaluator, component_names, *, relative_tolerance=2e-6,
                      max_cells=2000, max_depth=24, angular_orders=(3, 5),
                      max_line_evaluations=3400):
    """Bound native quadrature sums without turning uncertainties into credit.

    ``evaluator(N)`` returns ``(native_values, absolute_source_error_bounds)``.
    Its first two entries are negative/positive literal reference-side volumes.
    An evaluator unable to bound a node must raise NumericalLineBoundRefusal;
    it is propagated with that node and completed call count for a case OPEN.
    """
    if relative_tolerance != 2e-6:
        raise AngularControlRefusal("Original 2e-6 tolerance required")
    support, coverage = source_coverage(raw)
    original_area = sum((signed_double_area(v) for v in support),F())
    geometry = PreparedBankBounds(raw)
    discarded = []
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
        rules[order] = (tuple(map(fraction,(nodes+1)/2)), tuple(map(fraction,weights/2)))
    evaluated = line_count = completed = deepest = 0
    cells = []

    def rule(vertices, order):
        nonlocal line_count, completed
        nodes, weights = rules[order]
        value = np.zeros(shape, LD)
        enclosure = point(value)
        jacobian = signed_double_area(vertices)
        for s, sw in zip(nodes, weights):
            for t, tw in zip(nodes, weights):
                N = exact_node(vertices,s,t)
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
                exact_weight = jacobian*s*sw*tw
                weight = rational_enclosure(exact_weight)
                value += (weight.lower/LD(2)+weight.upper/LD(2))*v
                bounded_value = DirectedInterval(subtract(point(v), point(b)).lower,
                                                add(point(v), point(b)).upper)
                enclosure = add(enclosure, multiply(weight, bounded_value))
        return value, enclosure

    def can_add(count):
        return (evaluated+count <= max_cells and
                line_count+count*(low_order**2+high_order**2) <= max_line_evaluations)

    def add_cell(vertices, depth):
        nonlocal evaluated, deepest
        bounds = geometry.bounds(vertices)
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
        lo = _sum_enclosures((point(c.lower_volume) for c in cells), (2,)).lower
        hi = _sum_enclosures((point(c.upper_volume) for c in cells), (2,)).upper
        missing = _sum_enclosures((point(c.missing) for c in cells), (2,)).upper
        missing_count = sum(int(np.any(c.missing)) for c in cells)
        return coarse, fine, cb, fb, observed, combined, allowance, lo, hi, missing, missing_count

    support, mesh = event_mesh(raw,support,max_depth=max_depth)
    initial_depths = {decode_cell(c['vertices']):c['depth'] for c in mesh['cells']}
    retained_support = tuple(v for v in support if any(geometry.bounds(v)[s][1]>0 for s in (-1,1)))
    if not can_add(len(retained_support)):
        raise AngularControlRefusal("Resource cap cannot cover the initial conservative support")
    discarded.extend(v for v in support if v not in retained_support)
    for vertices in retained_support:
        add_cell(vertices, initial_depths[vertices])
    reason = "No positive candidate support"
    while cells:
        arrays = reduce_cells()
        allowance, missing_count = arrays[6], arrays[10]
        if np.all(arrays[5] <= allowance) and missing_count == 0:
            reason = "All embedded angular observations and source uncertainties fit original own tolerances"
            break
        best_index, best_key = 0, None
        for index, candidate in enumerate(cells):
            cb = candidate.coarse_enclosure.absolute_error_for(candidate.coarse)
            fb = candidate.fine_enclosure.absolute_error_for(candidate.fine)
            total = add(add(point(candidate.error), point(cb)), point(fb)).upper
            error_ratios = np.divide(total, allowance, out=np.zeros(shape, LD), where=allowance != 0)
            error_ratios[(allowance == 0) & (total > 0)] = np.inf
            key = (bool(np.any(candidate.missing)), float(np.sum(candidate.missing)),
                   float(np.max(error_ratios)), float(signed_double_area(candidate.vertices)/2))
            if best_key is None or key > best_key:
                best_index, best_key = index, key
        candidate = cells[best_index]
        if candidate.depth >= max_depth:
            reason = "Predeclared maximum angular depth reached with unresolved components"
            break
        children = split_cell(candidate.vertices)
        retained = tuple(v for v in children if any(hi>0 for lo,hi in geometry.bounds(v).values()))
        if not can_add(len(retained)):
            reason = 'Predeclared angular cell or radial-line work budget exhausted'
            break
        discarded.extend(v for v in children if v not in retained)
        cells.pop(best_index)
        for vertices in retained:
            add_cell(vertices, candidate.depth+1)
    coarse, fine, cb, fb, observed, combined, allowance, lo, hi, missing, missing_count = reduce_cells()
    result = AngularResult(names, coarse, fine, cb, fb, observed, allowance, combined,
                         np.maximum(lo, 0), np.maximum(hi, 0), np.maximum(missing, 0),
                         evaluated, line_count, len(cells), missing_count, deepest,
                         reason, float(relative_tolerance), max_cells, max_depth,
                         max_line_evaluations, tuple(angular_orders))

    retained_area = sum((signed_double_area(c.vertices) for c in cells),F())
    discarded_area = sum((signed_double_area(c) for c in discarded),F())
    if retained_area+discarded_area!=original_area:
        raise AssertionError('Initial parameter measure was lost')
    report=result.to_dict()
    report.update(format='genesis-bank-bounded-event-guided-angular-control-1',
        scope='exact_source_cover_and_exact_literal_parameter_map_with_embedded_angular_observations',
        enumerated_literal_native_material_nodes=False,
        angular_node_mapping_accuracy_certified=True,
        node_mapping_method='Exact rational Duffy map evaluated without rounding N',
        previous_native_mapper_accuracy_certified=False,
        gauss_nodes_and_weights_retained_as_literal_binary_operands=True,
        exact_gauss_root_error_certified=False,
        initial_source_support_coverage_certified=True,
        original_material_support_conservatively_covered=True,
        angular_geometry_bounds_shared_with_previous_auditor=False,
        angular_candidate_geometry_newly_directed_certified=True,
        exact_partition_of_initial_support=True,
        event_partition_initial_leaves=len(support),
        event_guided_mesh_sha256=mesh['sha256'],
        exact_event_alignment=False,
        event_cut_edge_parameter_bits=40,
        bank_volume_bounds='Exact centroid-line radial tubes plus Gamma/sphere separation',
        angular_refinement_priority='Resolve unsampled candidate banks first, then original own-component error',
        final_cells=[dict(vertices=[[ratio(v) for v in row] for row in c.vertices],depth=c.depth,
            lower_bank_volume=[ratio(fraction(v)) for v in c.lower_volume],
            upper_bank_volume=[ratio(fraction(v)) for v in c.upper_volume],
            coarse_bank_volume=[ratio(fraction(v)) for v in c.coarse[:2]],
            fine_bank_volume=[ratio(fraction(v)) for v in c.fine[:2]],
            missing_bank_volume=[ratio(fraction(v)) for v in c.missing],
            bank_bound_certificate=geometry.certificate(c.vertices)) for c in cells],
        certified_empty_cells_skipped=len(discarded),
        exact_initial_parameter_measure_preserved=True,
        initial_double_area=ratio(original_area),retained_double_area=ratio(retained_area),
        discarded_double_area=ratio(discarded_area),
        discarded_cells=[geometry.certificate(v) for v in discarded],
        coverage_sha256=coverage['sha256'],
        angular_accuracy_certified=False)
    return report
