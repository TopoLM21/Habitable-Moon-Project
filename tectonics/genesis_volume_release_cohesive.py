"""Mean cohesive force for a declared straight path in a fixed birth chart.

This is a computational work kernel.  Its explicit toughness, initial
reference traction and quadrature do not select a physical fracture or admit
normal contact.  The traction is evaluated along the path; endpoint traction
and a force reconstructed from scalar work are never substituted for it.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import hashlib
import json
import math

import numpy as np

from .genesis_volume_creation_law import (
    CreationLawRefusal, ProspectiveCreationLaw, _immutable, _numeric)
from .genesis_volume_prestress_creation import (
    PrestressCreationSeed, evaluate_prestress_creation, prepare_prestress_creation)


class VolumeReleaseCohesiveRefusal(CreationLawRefusal):
    """The fixed chart or independently integrated path is inadmissible."""


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _digest(value):
    return hashlib.sha256(json.dumps(_plain(value), sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class VolumeReleaseCohesiveChart:
    seed: PrestressCreationSeed
    reference_area_weights_m2: np.ndarray
    compact_jump_trace_world: np.ndarray
    birth_basis_xyz: np.ndarray
    compact_jump_trace_local: np.ndarray
    binding_sha256: str
    physical_release_admitted: bool = field(init=False, default=False)
    normal_contact_admitted: bool = field(init=False, default=False)

    def __post_init__(self):
        for name in ("reference_area_weights_m2", "compact_jump_trace_world",
                     "birth_basis_xyz", "compact_jump_trace_local"):
            object.__setattr__(self, name, _immutable(getattr(self, name)))


def _chart_body(chart):
    return dict(format="genesis-volume-release-cohesive-0.1",
        scope="computational_fixed_birth_chart_straight_path_no_physical_release",
        seed=_plain(asdict(chart.seed)),
        reference_area_weights_m2=_plain(chart.reference_area_weights_m2),
        compact_jump_trace_world=_plain(chart.compact_jump_trace_world),
        birth_basis_xyz=_plain(chart.birth_basis_xyz),
        compact_jump_trace_local=_plain(chart.compact_jump_trace_local),
        fracture_area_measure="original_reference_area_Gc_counts_both_banks_once",
        trajectory="literal_straight_enrichment_and_gap_increment",
        physical_release_admitted=chart.physical_release_admitted,
        normal_contact_admitted=chart.normal_contact_admitted)


def prepare_volume_release_cohesive(law, initial_reference_traction_world_pa,
        reference_area_weights_m2, compact_jump_trace_world, birth_basis_xyz):
    """Own explicit reference traction/area/trace and an orthonormal birth basis.

    Basis columns are the positive opening normal and the two tangents.  All
    traction and cohesive energy densities use original reference area.
    Source binding and work admission belong to the caller's separate gate.
    """
    if type(law) is not ProspectiveCreationLaw:
        raise VolumeReleaseCohesiveRefusal("An explicit ProspectiveCreationLaw is required")
    traction = _numeric(initial_reference_traction_world_pa, "reference traction")
    weights = _numeric(reference_area_weights_m2, "reference area weights")
    trace = _numeric(compact_jump_trace_world, "compact jump trace")
    basis = _numeric(birth_basis_xyz, "birth basis")
    if (weights.ndim != 1 or len(weights) == 0 or np.any(weights <= 0)
            or traction.shape != (len(weights), 3)
            or trace.ndim != 3 or trace.shape[:2] != (len(weights), 3)
            or trace.shape[2] == 0):
        raise VolumeReleaseCohesiveRefusal("Reference traction, positive areas and compact trace shapes differ")
    if basis.shape == (3, 3):
        basis = np.broadcast_to(basis, (len(weights), 3, 3))
    if basis.shape != (len(weights), 3, 3):
        raise VolumeReleaseCohesiveRefusal("Birth basis shape differs from quadrature points")
    eps = np.finfo(float).eps
    gram = np.einsum("pji,pjk->pik", basis, basis)
    if (np.any(np.abs(gram-np.eye(3)) > 128*eps)
            or np.any(np.abs(np.linalg.det(basis)-1.) > 256*eps)):
        raise VolumeReleaseCohesiveRefusal("Birth basis must be a right-handed orthonormal normal/tangent frame")
    local_traction = np.einsum("pji,pj->pi", basis, traction)
    local_trace = np.einsum("pji,pjk->pik", basis, trace)
    seed = prepare_prestress_creation(law, local_traction)
    chart = VolumeReleaseCohesiveChart(seed, weights, trace, basis, local_trace, "")
    return VolumeReleaseCohesiveChart(seed, weights, trace, basis, local_trace,
                                    _digest(_chart_body(chart)))


def validate_volume_release_cohesive(chart):
    if (type(chart) is not VolumeReleaseCohesiveChart
            or type(chart.seed) is not PrestressCreationSeed
            or chart.physical_release_admitted or chart.normal_contact_admitted
            or chart.binding_sha256 != _digest(_chart_body(chart))):
        raise VolumeReleaseCohesiveRefusal("Cohesive chart differs from its immutable declared binding")
    return chart


def volume_release_cohesive_to_dict(chart):
    validate_volume_release_cohesive(chart)
    return {**_chart_body(chart), "cohesive_chart_sha256": chart.binding_sha256}


def volume_release_cohesive_from_dict(raw, *, law,
        initial_reference_traction_world_pa, reference_area_weights_m2,
        compact_jump_trace_world, birth_basis_xyz):
    """Require literal agreement with independently supplied trusted inputs."""
    expected = prepare_volume_release_cohesive(law,
        initial_reference_traction_world_pa, reference_area_weights_m2,
        compact_jump_trace_world, birth_basis_xyz)
    if raw != volume_release_cohesive_to_dict(expected):
        raise VolumeReleaseCohesiveRefusal("Cohesive chart packet differs from trusted birth inputs")
    return expected


@dataclass(frozen=True, slots=True)
class VolumeReleaseCohesiveResponse:
    chart_binding_sha256: str
    old_enrichment_m: np.ndarray
    new_enrichment_m: np.ndarray
    old_gap_local_m: np.ndarray
    new_gap_local_m: np.ndarray
    new_maximum_separation_m: np.ndarray
    endpoint_traction_world_pa: np.ndarray
    mean_traction_local_pa: np.ndarray
    mean_traction_world_pa: np.ndarray
    mean_generalized_force_n: np.ndarray
    endpoint_generalized_force_n: np.ndarray
    mean_force_component_order_error_n: np.ndarray
    mean_force_component_roundoff_bound_n: np.ndarray
    mean_traction_component_order_error_pa: np.ndarray
    mean_traction_component_roundoff_bound_pa: np.ndarray
    mean_traction_integrated_absolute_pa: np.ndarray
    old_stored_energy_j: float
    stored_energy_j: float
    old_fracture_dissipation_j: float
    fracture_dissipation_j: float
    fracture_dissipation_increment_j: float
    mechanical_work_j: float
    numerically_integrated_work_j: float
    force_work_j: float
    integrated_absolute_work_j: float
    work_order_error_j: float
    work_duality_error_j: float
    work_kernel_error_j: float
    work_integral_roundoff_bound_j: float
    work_kernel_roundoff_bound_j: float
    work_roundoff_bound_j: float
    endpoint_work_agrees_at_integral_precision: bool
    relative_tolerance: float
    quadrature_orders: tuple[int, int]
    point_event_fractions: tuple[tuple[float, ...], ...]
    accepted_subinterval_count: int
    maximum_subdivision_depth: int
    kernel_only: bool = field(init=False, default=True)
    physical_release_admitted: bool = field(init=False, default=False)
    normal_contact_admitted: bool = field(init=False, default=False)
    funding_admitted: bool = field(init=False, default=False)

    def __post_init__(self):
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, np.ndarray):
                object.__setattr__(self, item.name, _immutable(value))

    @property
    def numerically_resolved(self):
        """Whether endpoint work agrees without cumulative-ledger cancellation credit."""
        return self.endpoint_work_agrees_at_integral_precision

    @property
    def exact_work_arithmetic_dominated(self):
        return not self.endpoint_work_agrees_at_integral_precision


def volume_release_cohesive_response_to_dict(response):
    if type(response) is not VolumeReleaseCohesiveResponse:
        raise VolumeReleaseCohesiveRefusal("A typed cohesive response is required")
    body = _plain(asdict(response))
    body["numerically_resolved"] = response.numerically_resolved
    body["exact_work_arithmetic_dominated"] = response.exact_work_arithmetic_dominated
    return {**body, "cohesive_response_sha256": _digest(body)}


def _events(seed, old_gap, gap_increment, old_maximum):
    """Resolve normal kinks, radial minima and all history/damage thresholds."""
    internal = seed.reference_offset_m + old_gap
    normal_events = [0., 1.]
    if gap_increment[0] != 0:
        root = -internal[0]/gap_increment[0]
        if 0 < root < 1:
            normal_events.append(float(root))
    normal_events.sort()
    events = list(normal_events)
    for lo, hi in zip(normal_events[:-1], normal_events[1:]):
        selected = np.array([internal[0] + (lo+hi)*.5*gap_increment[0] >= 0,
                             True, True])
        start = internal[selected]
        step = gap_increment[selected]
        # Scaling prevents overflow in the squared-radius quadratic.
        scale = max(float(np.max(np.abs(start), initial=0.)),
                    float(np.max(np.abs(step), initial=0.)),
                    float(old_maximum), seed.peak_separation_m,
                    float(seed.final_separation_m))
        start, step = start/scale, step/scale
        aa = float(step@step)
        bb = float(start@step)
        if aa == 0:
            continue
        center = -bb/aa
        if lo < center < hi:
            events.append(center)
        residual = start+center*step
        minimum_squared = float(residual@residual)
        for radius in (old_maximum, seed.peak_separation_m,
                       float(seed.final_separation_m)):
            scaled_radius_squared = (float(radius)/scale)**2
            distance_squared = scaled_radius_squared-minimum_squared
            # A radius tangent to the path has a double root.  Roundoff in
            # the two squared norms must not invent a microscopic pair of
            # history crossings adjacent to that turning point.
            if abs(distance_squared) <= 64*np.finfo(float).eps*(
                    scaled_radius_squared+minimum_squared):
                distance_squared = 0.
            if distance_squared < 0:
                continue
            distance = math.sqrt(distance_squared/aa)
            for root in (center-distance, center+distance):
                if lo < root < hi:
                    events.append(float(root))
    return tuple(sorted(set(events)))


def _integral(seed, old_gap, gap_increment, old_maximum, lo, hi, nodes, weights):
    fractions = lo+(hi-lo)*(nodes+1.)*.5
    gaps = old_gap+fractions[:, None]*gap_increment
    response = evaluate_prestress_creation(seed, old_gap, gaps, old_maximum)
    traction = response.traction_pa
    point_work = traction@gap_increment
    integral = (hi-lo)*.5*np.r_[weights@traction, weights@point_work]
    absolute = (hi-lo)*.5*np.r_[weights@np.abs(traction), weights@np.abs(point_work)]
    if not (np.isfinite(integral).all() and np.isfinite(absolute).all()):
        raise VolumeReleaseCohesiveRefusal("Cohesive mean path integral overflowed")
    # Cancellation in reference+old+s*increment can dominate a tiny normal
    # traction immediately next to its exact zero.  Bound that arithmetic
    # using the actual tangent, not a macroscopic traction floor.  The normal
    # kink is continuous and its two one-sided stiffnesses are bounded here.
    gap_scale = np.abs(seed.reference_offset_m)+np.abs(old_gap)+np.abs(
        fractions[:, None]*gap_increment)
    tangent = np.abs(response.tangent_pa_m)
    near_normal_kink = (np.abs(seed.reference_offset_m[0]+gaps[:, 0])
                       <= 8*np.finfo(float).eps*gap_scale[:, 0])
    kink_stiffness = np.maximum(seed.law.cohesive_stiffness_pa_m*(1-response.damage),
                              seed.law.compression_stiffness_pa_m)
    tangent[near_normal_kink, 0, 0] = np.maximum(
        tangent[near_normal_kink, 0, 0], kink_stiffness[near_normal_kink])
    traction_roundoff = 16*np.finfo(float).eps*np.einsum("qij,qj->qi", tangent, gap_scale)
    # Remaining stiffness is obtained by subtracting damage from one.  Near
    # complete separation its arithmetic error is bounded by the actual
    # softening formula operands, rather than by the tiny resulting traction.
    softening = ((response.maximum_separation_m > seed.peak_separation_m)
                 & (response.maximum_separation_m < seed.final_separation_m))
    if np.any(softening):
        effective = seed.reference_offset_m+gaps
        effective[:, 0] = np.maximum(effective[:, 0], 0.)
        ratio = float(seed.final_separation_m)/(float(seed.final_separation_m)-seed.peak_separation_m)
        traction_roundoff[softening] += (32*np.finfo(float).eps
            *seed.law.cohesive_stiffness_pa_m*(1+ratio)*np.abs(effective[softening]))
    value_roundoff = np.c_[traction_roundoff,
                           traction_roundoff@np.abs(gap_increment)]
    roundoff = ((hi-lo)*.5*(weights@value_roundoff)
                +128*np.finfo(float).eps*absolute)
    return integral, absolute, roundoff


def _point_integrals(seed, old_gap, increment, history, events, low_rule,
                     high_rule, tolerance, max_depth, forced_depth):
    totals_low, totals_high, absolute, roundoff_total = (
        np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4))
    count = depth_used = 0
    pending = [(lo, hi, 0) for lo, hi in zip(events[:-1], events[1:])]
    while pending:
        lo, hi, depth = pending.pop()
        low, _, low_roundoff = _integral(seed, old_gap, increment, history, lo, hi, *low_rule)
        high, magnitude, high_roundoff = _integral(seed, old_gap, increment, history, lo, hi, *high_rule)
        roundoff = low_roundoff+high_roundoff
        converged = np.all(np.abs(high-low) <= tolerance*np.abs(high)+roundoff)
        if converged and depth >= forced_depth:
            totals_low += low
            totals_high += high
            absolute += magnitude
            roundoff_total += roundoff
            count += 1
            depth_used = max(depth_used, depth)
        else:
            midpoint = (lo+hi)*.5
            if depth >= max_depth or midpoint == lo or midpoint == hi:
                raise VolumeReleaseCohesiveRefusal(
                    "Cohesive mean traction/work quadrature did not converge component by component")
            pending.extend(((lo, midpoint, depth+1), (midpoint, hi, depth+1)))
    return totals_low, totals_high, absolute, roundoff_total, count, depth_used


def evaluate_volume_release_cohesive(chart, old_enrichment_m, new_enrichment_m,
        old_maximum_separation_m, *, relative_tolerance=2e-10, low_order=16,
        high_order=32, max_subdivisions=16):
    """Evaluate exact endpoint energy/history and integrate the actual mean force.

    Gauss rules are compared separately for every traction/force component
    and work.  Event subdivision resolves compression, radial turning,
    history loading, peak and final separation.  Additional bisection is
    adaptive.  Cancellation allowances use machine epsilons times integrated
    absolute quantities, the actual tangent times gap-addition operands and
    the softening formula operands, never a force/work floor.  The separate
    endpoint-kernel allowance includes cumulative dissipation subtraction;
    this arithmetic witness never supplies fracture funding.
    The old maximum is retained on unloading and partial reloading.
    """
    validate_volume_release_cohesive(chart)
    nmode = chart.compact_jump_trace_world.shape[2]
    old = _numeric(old_enrichment_m, "old enrichment")
    new = _numeric(new_enrichment_m, "new enrichment")
    history = _numeric(old_maximum_separation_m, "old cohesive maximum")
    if old.shape != (nmode,) or new.shape != (nmode,) or history.shape != chart.reference_area_weights_m2.shape:
        raise VolumeReleaseCohesiveRefusal("Cohesive path enrichment/history shapes differ from chart")
    tolerance = _numeric(relative_tolerance, "quadrature relative tolerance")
    if tolerance.shape != () or not 0 < tolerance < 1:
        raise VolumeReleaseCohesiveRefusal("Cohesive quadrature relative tolerance must lie in (0,1)")
    for value, name in ((low_order, "low order"), (high_order, "high order"),
                        (max_subdivisions, "maximum subdivisions")):
        if type(value) is not int or value < (0 if name == "maximum subdivisions" else 1):
            raise VolumeReleaseCohesiveRefusal(f"Cohesive {name} requires an integer in its valid range")
    if high_order <= low_order:
        raise VolumeReleaseCohesiveRefusal("Cohesive high quadrature order must exceed low order")
    old_gap = np.einsum("pai,i->pa", chart.compact_jump_trace_local, old)
    new_gap = np.einsum("pai,i->pa", chart.compact_jump_trace_local, new)
    increment = new_gap-old_gap
    exact = evaluate_prestress_creation(chart.seed, old_gap, new_gap, history)
    # This normalises the same bounded history-roundoff accepted by the kernel.
    old_response = evaluate_prestress_creation(chart.seed, old_gap, old_gap, history)
    history = old_response.maximum_separation_m
    seeds = [prepare_prestress_creation(chart.seed.law, traction)
             for traction in chart.seed.initial_traction_pa]
    events = tuple(_events(seed, gap, step, maximum)
        for seed, gap, step, maximum in zip(seeds, old_gap, increment, history))
    low_rule = np.polynomial.legendre.leggauss(low_order)
    high_rule = np.polynomial.legendre.leggauss(high_order)
    areas = chart.reference_area_weights_m2
    trace = chart.compact_jump_trace_local
    exact_work = float(areas@exact.mechanical_work_j_m2)
    eps = np.finfo(float).eps
    # A small number of global refinements also resolves cancellation between
    # quadrature points in assembled force components.  Exhaustion refuses.
    for forced_depth in range(min(max_subdivisions, 8)+1):
        point_low, point_high, absolute, point_roundoff = [], [], [], []
        count = depth_used = 0
        for seed, gap, step, maximum, point_events in zip(seeds, old_gap,
                                                        increment, history, events):
            lo, hi, mag, arithmetic, number, depth = _point_integrals(seed, gap, step,
                maximum, point_events, low_rule, high_rule, float(tolerance),
                max_subdivisions, forced_depth)
            point_low.append(lo)
            point_high.append(hi)
            absolute.append(mag)
            point_roundoff.append(arithmetic)
            count += number
            depth_used = max(depth_used, depth)
        point_low, point_high, absolute, point_roundoff = map(np.asarray,
            (point_low, point_high, absolute, point_roundoff))
        low_force = np.einsum("pai,pa,p->i", trace, point_low[:, :3], areas)
        high_force = np.einsum("pai,pa,p->i", trace, point_high[:, :3], areas)
        force_magnitude = np.einsum("pai,pa,p->i", np.abs(trace), absolute[:, :3], areas)
        force_roundoff = (128*eps*force_magnitude
            +np.einsum("pai,pa,p->i", np.abs(trace), point_roundoff[:, :3], areas))
        force_error = np.abs(high_force-low_force)
        integral_work = float(areas@point_high[:, 3])
        low_work = float(areas@point_low[:, 3])
        force_work = float(high_force@(new-old))
        work_roundoff = float(areas@point_roundoff[:, 3])+128*eps*float(areas@absolute[:, 3])
        # Endpoint potential differences have their own arithmetic cancellation
        # scale; this is declared separately from integral/order certification.
        kernel_roundoff = 128*eps*float(areas@(
            np.abs(exact.stored_energy_j_m2)+np.abs(exact.old_stored_energy_j_m2)
            +np.abs(exact.fracture_dissipation_j_m2)
            +np.abs(exact.old_fracture_dissipation_j_m2)))
        force_ok = np.all(force_error <= float(tolerance)*np.abs(high_force)+force_roundoff)
        order_ok = abs(integral_work-low_work) <= float(tolerance)*abs(integral_work)+work_roundoff
        duality_ok = abs(force_work-integral_work) <= float(tolerance)*abs(integral_work)+work_roundoff
        kernel_ok = abs(integral_work-exact_work) <= float(tolerance)*abs(exact_work)+work_roundoff+kernel_roundoff
        if force_ok and order_ok and duality_ok and kernel_ok:
            break
    else:
        raise VolumeReleaseCohesiveRefusal(
            "Cohesive assembled mean force/work failed component convergence or exact path duality")
    mean_local = point_high[:, :3]
    mean_world = np.einsum("pij,pj->pi", chart.birth_basis_xyz, mean_local)
    endpoint_world = np.einsum("pij,pj->pi", chart.birth_basis_xyz, exact.traction_pa)
    endpoint_force = np.einsum("pai,pa,p->i", trace, exact.traction_pa, areas)
    return VolumeReleaseCohesiveResponse(chart.binding_sha256, old, new,
        old_gap, new_gap, exact.maximum_separation_m, endpoint_world,
        mean_local, mean_world, high_force, endpoint_force, force_error,
        force_roundoff, np.abs(point_high[:, :3]-point_low[:, :3]),
        point_roundoff[:, :3], absolute[:, :3],
        float(areas@exact.old_stored_energy_j_m2),
        float(areas@exact.stored_energy_j_m2),
        float(areas@exact.old_fracture_dissipation_j_m2),
        float(areas@exact.fracture_dissipation_j_m2),
        float(areas@exact.fracture_dissipation_increment_j_m2), exact_work,
        integral_work, force_work, float(areas@absolute[:, 3]), abs(integral_work-low_work),
        abs(force_work-integral_work), abs(integral_work-exact_work),
        work_roundoff, kernel_roundoff, work_roundoff+kernel_roundoff,
        bool(abs(integral_work-exact_work) <= float(tolerance)*abs(integral_work)+work_roundoff),
        float(tolerance), (low_order, high_order),
        events, count, depth_used)
