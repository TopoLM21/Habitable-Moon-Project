"""Exact source cover, event-cell subdivision and rational quadrature mapping.

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
LD=np.longdouble


def decode_cell(value):
    return tuple(tuple(F(int(a),int(b)) for a,b in row) for row in value)


class CellGeometry:
    def __init__(self,raw):
        center = np.asarray(raw['seed_center_world_m'])
        if any(fraction(a)!=fraction(b) for a,b in zip(center,np.asarray(center,float))):
            raise AngularControlRefusal('Event implementation requires literal binary64 source center')
        self.events,_=source_events(raw)
        X=rational(np.asarray(raw['section']['triangle_m'],float))
        det=sum(X[0,j]*(X[1,(j+1)%3]*X[2,(j+2)%3]-X[1,(j+2)%3]*X[2,(j+1)%3]) for j in range(3))
        lo,hi=map(fraction,np.asarray(raw['section']['lambda_bounds'],float))
        if det<=0:
            raise AngularControlRefusal('Source reference chart orientation differs')
        self.factor=det*(hi**3-lo**3)/6
        self.cache={}; self.skipped=0

    def bounds(self,vertices):
        if vertices in self.cache:
            return self.cache[vertices]
        certs={k:sign_certificate(v,vertices) for k,v in self.events.items()}
        empty=radial_support_classification(certs)=='EMPTY_CERTIFIED_ON_SUPPLIED_CELL'
        if not empty and all(
                F(*map(int,certs[k][side])) == 0
                for k in ('Gamma_cohort_lower','Gamma_cohort_upper')
                for side in ('lower','upper')):
            raise AngularControlRefusal(
                'Literal Gamma contains an unresolved candidate cell')
        upper=F() if empty else signed_double_area(vertices)*self.factor
        result={s:(LD(0),rational_enclosure(upper).upper) for s in (-1,1)}
        if empty:
            self.skipped+=1
        # Affine Gamma has no interior extremum on a triangle x lambda slab.
        for side in (-1,1):
            field='lower' if side<0 else 'upper'
            values=[F(*map(int,certs[k][field])) for k in ('Gamma_cohort_lower','Gamma_cohort_upper')]
            impossible=min(values)>=0 if side<0 else max(values)<=0
            if impossible:
                result[side]=(LD(0),LD(0))
        self.cache[vertices]=result
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
    if relative_tolerance != 2e-6:
        raise AngularControlRefusal("Original 2e-6 tolerance required")
    support, coverage = source_coverage(raw)
    geometry = CellGeometry(raw)
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

    initial_depths = {}
    if support:
        partition = partition_events(raw,support,max_depth=min(2,max_depth),max_cells=256)
        support = tuple(decode_cell(leaf["vertices"]) for leaf in partition["leaves"])
        initial_depths = {decode_cell(leaf['vertices']):leaf['depth'] for leaf in partition['leaves']}
    retained_support = tuple(v for v in support if any(geometry.bounds(v)[s][1]>0 for s in (-1,1)))
    if not can_add(len(retained_support)):
        raise AngularControlRefusal("Resource cap cannot cover the initial conservative support")
    for vertices in retained_support:
        add_cell(vertices, initial_depths[vertices])
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
            key = (float(np.max(ratio)), float(np.sum(candidate.missing)), float(signed_double_area(candidate.vertices)/2))
            if best_key is None or key > best_key:
                best_index, best_key = index, key
        candidate = cells[best_index]
        if candidate.depth >= max_depth:
            reason = "Predeclared maximum angular depth reached with unresolved components"
            break
        cells.pop(best_index)
        for vertices in split_cell(candidate.vertices):
            add_cell(vertices, candidate.depth+1)
    coarse, fine, cb, fb, observed, combined, allowance, lo, hi, missing, missing_count = reduce_cells()
    result = AngularResult(names, coarse, fine, cb, fb, observed, allowance, combined,
                         np.maximum(lo, 0), np.maximum(hi, 0), np.maximum(missing, 0),
                         evaluated, line_count, len(cells), missing_count, deepest,
                         reason, float(relative_tolerance), max_cells, max_depth,
                         max_line_evaluations, tuple(angular_orders))

    report=result.to_dict()
    report.update(format='genesis-exact-mapped-angular-control-1',
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
        certified_empty_cells_skipped=geometry.skipped,
        coverage_sha256=coverage['sha256'],
        angular_accuracy_certified=False)
    return report
