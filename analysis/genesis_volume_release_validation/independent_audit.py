"""Producer-free checks of private compact release work on unchanged Genesis.

The numerical sources are preserved independent material, Maxwell and cohesive
oracles.  No tectonics module or public planetary loader is imported.  A passed
computational balance does not select a physical cut, release it, or advance
the accepted planetary history.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis.genesis_volume_memory_validation import independent_audit_v2 as memory
from analysis.genesis_volume_seed_validation import independent_audit as seed
from analysis.genesis_volume_insertion_validation import independent_audit as insertion
from analysis.genesis_volume_creation_validation import independent_audit as creation

PARENT_RAW_SHA256 = memory.PARENT_RAW_SHA256
PARENT_CHECKPOINT_SHA256 = memory.PARENT_CHECKPOINT_SHA256
FORMATION_SHA256 = memory.FORMATION_SHA256
PREVIOUS_MEMORY_FIXTURE_RAW_SHA256 = "a98ff7cc89aeb59eb004c05b8ac87f75f2e39a4f10d4454fdf93c288153c9c0e"
PREVIOUS_MEMORY_FIXTURE_SHA256 = "2ba89795666f61987e513fa78971b9605db0d08bd6d83406eb9ebbe6295587c3"
FORMAT = "independent-genesis-volume-release-work-audit-0.1"
EPS = np.finfo(float).eps
QUADRATURE_SCOPE = "literal_archived_float64_point_fields_restricted_before_Gram_longdouble_computational_arithmetic_only"


def require(condition, message):
    if not condition:
        raise ValueError("Independent compact release audit: " + message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def numeric_array_pin(value):
    a = np.ascontiguousarray(value, dtype=float)
    return hashlib.sha256(repr(a.shape).encode()+a.tobytes()).hexdigest()


def seal(raw, key):
    require(type(raw) is dict and raw.get(key) == digest(
        {k: v for k, v in raw.items() if k != key}), key + " seal differs")


def array(value, shape=None):
    original = np.asarray(value)
    require(not any(isinstance(x, (bool, np.bool_)) for x in np.asarray(value, dtype=object).flat)
            and original.dtype.kind in "fiu" and np.isfinite(original).all()
            and (shape is None or original.shape == shape), "finite real numeric array required")
    return np.asarray(original, float)


def close(actual, expected, name, *, rtol=3e-8, atol=1e-10):
    a, b = array(actual), array(expected)
    require(a.shape == b.shape, name + " dimensions differ")
    error = float(np.max(abs(a-b), initial=0.))
    scale = float(np.max(abs(b), initial=0.))
    require(error <= atol + rtol*scale, name + " differs")
    return error/max(scale, atol, np.finfo(float).tiny)


def exact(actual, expected, name):
    require(canonical(actual) == canonical(expected), name + " differs")


def mode_close(actual, expected, name, *, rtol=3e-6, atol=1e-35):
    """Keep weak A/P forces and inertial coupling out of the large bulk scale."""
    a, b = array(actual).reshape(-1, 6), array(expected).reshape(-1, 6)
    require(a.shape == b.shape, name + " mode dimensions differ")
    return [close(a[:, j:j+3], b[:, j:j+3], name + " " + family,
                  rtol=rtol, atol=atol) for j, family in ((0, "A"), (3, "P"))]


def scalar_zero_credit(raw):
    """A re-sealed receipt cannot convert history or roundoff into funding."""
    for key in ("historical_dissipation_available_for_creation_j",
                "historical_dissipation_credit_j", "arithmetic_roundoff_credit_j",
                "creation_work_debited_from_Genesis_j"):
        if key in raw:
            require(type(raw[key]) in (int, float) and raw[key] == 0.,
                    key + " cannot fund a private prospective path")


def independent_jump_modes(trace, weights, *, tolerance=1e-10, separation=10.):
    """Weighted column-normalized rank and the complete physical tied kernel.

    This computes the projector, without copying the producer's singular-vector
    convention.  Back transformation precedes physical orthogonal completion.
    """
    trace, weights = array(trace), array(weights)
    require(trace.shape == (len(weights), 3, 18) and len(weights) > 0
            and np.all(weights >= 0) and np.any(weights > 0), "jump/area support differs")
    require(0 < tolerance*separation < 1 and separation > 1, "rank declaration differs")
    B = (trace*np.sqrt(weights)[:, None, None]).reshape(-1, 18)
    lengths = np.linalg.norm(B, axis=0)
    active, zero = np.flatnonzero(lengths > 0), np.flatnonzero(lengths == 0)
    singular = np.zeros(18)
    if len(active):
        _, s, vh = np.linalg.svd(B[:, active]/lengths[active], full_matrices=True)
        singular[:len(s)] = s
        threshold = tolerance*s[0]
        require(not np.any((s >= threshold/separation) & (s <= threshold*separation)),
                "jump rank has unresolved singular values")
        rank = int(np.count_nonzero(s > threshold*separation))
        Z = np.zeros((18, 18-rank))
        Z[active, :len(active)-rank] = vh[rank:].T/lengths[active, None]
        Z[zero, len(active)-rank:] = np.eye(len(zero))
    else:
        rank, threshold, Z = 0, 0., np.eye(18)
    if Z.shape[1]:
        Z /= np.linalg.norm(Z, axis=0)
        Q, _ = np.linalg.qr(Z, mode="complete")
        null, released = Q[:, :18-rank], Q[:, 18-rank:]
    else:
        null, released = np.empty((18, 0)), np.eye(18)
    residual = 0. if not np.any(B) else np.linalg.norm(B@null)/np.linalg.norm(B)
    require(residual <= tolerance/separation + 512*EPS, "complete tied kernel residual differs")
    return dict(rank=rank, nullity=18-rank, column_scales_m=lengths,
        normalized_singular_values=singular, rank_threshold=float(threshold),
        tied_null_modes=null, release_basis=released, null_projector=null@null.T,
        release_projector=released@released.T, weighted_kernel_relative_residual=float(residual))


def audit_jump_modes(raw, trace, weights):
    seal(raw, "release_modes_sha256")
    require(raw.get("complete_kernel_remains_tied") is True
            and raw.get("physical_release_admitted") is False
            and raw.get("funding_admitted") is False, "complete jump kernel cannot be released")
    expected = independent_jump_modes(trace, weights,
        tolerance=float(raw["relative_rank_tolerance"]),
        separation=float(raw["rank_separation_factor"]))
    require(0 < raw["relative_rank_tolerance"] <= 1e-10 and raw["rank_separation_factor"] >= 10,
            "jump rank tolerance cannot weaken complete-kernel admission")
    require(raw["rank"] == expected["rank"] and raw["nullity"] == expected["nullity"],
            "complete tied jump-kernel dimension differs")
    close(raw["trace_xyz"], trace, "source compact jump trace", rtol=2e-8, atol=1e-13)
    close(raw["reference_area_weights_m2"], weights, "source reference area", rtol=2e-8, atol=1e-12)
    close(raw["column_scales_m"], expected["column_scales_m"], "jump column scales", rtol=2e-8)
    # Null singular values themselves are roundoff; the resolved rank and
    # physical subspace, rather than null-value equality, carry admission.
    rank = expected["rank"]
    close(array(raw["normalized_singular_values"], (18,))[:rank],
        expected["normalized_singular_values"][:rank], "retained jump singular values", rtol=2e-7, atol=1e-13)
    N = array(raw["tied_null_modes"], (18, expected["nullity"]))
    R = array(raw["release_basis"], (18, rank))
    close(N.T@N, np.eye(expected["nullity"]), "kernel orthonormality", rtol=0, atol=3e-13)
    close(R.T@R, np.eye(rank), "release orthonormality", rtol=0, atol=3e-13)
    close(N@N.T, expected["null_projector"], "physical tied kernel projector", rtol=2e-7, atol=2e-8)
    close(R@R.T, expected["release_projector"], "physical release projector", rtol=2e-7, atol=2e-8)
    close(N@N.T+R@R.T, np.eye(18), "complete physical subspaces", rtol=0, atol=4e-13)
    expected["declared_tied_null_modes"] = N
    expected["declared_release_basis"] = R
    return expected


def require_tied_kernel(modes, vector, name):
    a = array(vector, (18,))
    leakage = float(np.linalg.norm(modes["declared_tied_null_modes"].T@a))
    require(leakage <= 512*EPS*float(np.linalg.norm(a)),
            name + " contains a released complete jump-kernel component")


def source_surface(parent, definition, *, order=8):
    surface = seed.finite_quadrature(parent, definition["normal_xyz"], definition["offset_m"],
        definition["seed_center_world_m"], definition["seed_radius_m"],
        cohort_id=definition["original_cohort_id"], order=order)
    chart = seed.cohort_chart(parent, definition["original_cohort_id"],
        definition["normal_xyz"], definition["offset_m"])
    fields = seed.compact_fields(chart, surface["barycentric"], surface["radial_scale"],
        definition["seed_center_world_m"], definition["seed_radius_m"])
    prestress = seed.prior.prestress_fields(parent, surface)
    basis = seed.prior.birth_basis(surface["current_normal_xyz"])
    return dict(surface=surface, trace=fields["surface_enrichment_trace"],
        weights=surface["reference_area_weights_m2"],
        initial_traction_world=prestress["reference_traction_world_pa"], basis=basis)


def independent_cohesive_path(law, initial_world, weights, trace_world, basis,
        old_a, new_a, old_maximum, *, order=48):
    """Direct event-split vector traction quadrature from older scalar oracles."""
    creation.validate_law(law)
    weights, trace_world, basis = array(weights), array(trace_world), array(basis)
    npoint, nmode = len(weights), trace_world.shape[-1]
    old_a, new_a = array(old_a, (nmode,)), array(new_a, (nmode,))
    require(trace_world.shape == (npoint, 3, nmode) and basis.shape == (npoint, 3, 3)
            and np.all(weights > 0), "cohesive reference chart differs")
    local_trace = np.einsum("pji,pjk->pik", basis, trace_world)
    initial_local = np.einsum("pji,pj->pi", basis, array(initial_world, (npoint, 3)))
    old_gap = np.einsum("pai,i->pa", local_trace, old_a)
    new_gap = np.einsum("pai,i->pa", local_trace, new_a)
    maximum = array(old_maximum, (npoint,))
    shifted = insertion.prestress_segment(law, initial_local, old_gap, new_gap, maximum)
    reference = shifted["seed"]
    # Read energies at the actual old history independently of the new state.
    old = insertion.prestress_segment(law, initial_local, old_gap, old_gap, maximum)
    nodes, gauss = np.polynomial.legendre.leggauss(order)
    mean = np.zeros((npoint, 3))
    absolute_mean = np.zeros((npoint, 3))
    absolute_work = np.zeros(npoint)
    value_work_roundoff = np.zeros(npoint)
    value_traction_roundoff = np.zeros((npoint, 3))
    absolute_traction_upper = np.zeros((npoint, 3))
    for j in range(npoint):
        point_law = {**law, "fracture_energy_j_m2": float(reference["auxiliary_fracture_energy_j_m2"][j])}
        h0 = old_gap[j]+reference["reference_offset_m"][j]
        h1 = new_gap[j]+reference["reference_offset_m"][j]
        history = max(float(maximum[j]), float(reference["initial_maximum_separation_m"][j]),
            float(np.linalg.norm(np.r_[max(h0[0], 0), h0[1:]])))
        cuts = creation._path_breaks(point_law, h0, h1, history)
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            fractions = (lo+hi)/2+(hi-lo)/2*nodes
            gaps = h0+(h1-h0)*fractions[:, None]
            point = creation.law_point(point_law, gaps, np.full(order, history))
            traction = point["traction_pa"]
            mean[j] += (hi-lo)/2*(gauss@traction)
            absolute_mean[j] += (hi-lo)/2*(gauss@abs(traction))
            delta = new_gap[j]-old_gap[j]
            absolute_work[j] += (hi-lo)/2*float(gauss@abs(traction@delta))
            # This bounds only arithmetic on this actual path.  It excludes
            # old cumulative D subtraction and inherited bulk-energy ULPs.
            gap_scale = abs(reference["reference_offset_m"][j])+abs(old_gap[j])+abs(fractions[:, None]*delta)
            tangent = abs(point["tangent_pa_m"])
            near_kink = abs(gaps[:, 0]) <= 8*EPS*gap_scale[:, 0]
            kink_k = np.maximum(law["cohesive_stiffness_pa_m"]*(1-point["damage"]), law["compression_stiffness_pa_m"])
            tangent[near_kink, 0, 0] = np.maximum(tangent[near_kink, 0, 0], kink_k[near_kink])
            value_error = 16*EPS*np.einsum("qij,qj->qi", tangent, gap_scale)
            onset = law["peak_traction_pa"]/law["cohesive_stiffness_pa_m"]
            final = float(reference["final_separation_m"][j])
            soft = (point["maximum_separation_m"] > onset) & (point["maximum_separation_m"] < final)
            effective = gaps.copy(); effective[:, 0] = np.maximum(effective[:, 0], 0)
            value_error[soft] += 32*EPS*law["cohesive_stiffness_pa_m"]*(1+final/(final-onset))*abs(effective[soft])
            value_work_roundoff[j] += (hi-lo)/2*float(gauss@(value_error@abs(delta)))
            value_traction_roundoff[j] += (hi-lo)/2*(gauss@value_error)
            start, end = h0+(h1-h0)*lo, h0+(h1-h0)*hi
            stiffness_upper = np.full(3, law["cohesive_stiffness_pa_m"])
            stiffness_upper[0] = max(law["cohesive_stiffness_pa_m"], law["compression_stiffness_pa_m"])
            absolute_traction_upper[j] += (hi-lo)*stiffness_upper*(abs(start)+abs(end))/2
    mean_world = np.einsum("pij,pj->pi", basis, mean)
    end_world = np.einsum("pij,pj->pi", basis, shifted["traction_pa"])
    force = np.einsum("pai,pa,p->i", local_trace, mean, weights)
    endpoint = np.einsum("pai,pa,p->i", local_trace, shifted["traction_pa"], weights)
    force_abs = np.einsum("pai,pa,p->i", abs(local_trace), absolute_mean, weights)
    work = float(weights@shifted["mechanical_work_j_m2"])
    close(float(force@(new_a-old_a)), work, "independent path force/work duality",
        rtol=4e-10, atol=256*EPS*float(abs(force_abs)@abs(new_a-old_a)))
    return dict(old_gap_local_m=old_gap, new_gap_local_m=new_gap,
        new_maximum_separation_m=shifted["maximum_separation_m"],
        mean_traction_local_pa=mean, mean_traction_world_pa=mean_world,
        endpoint_traction_world_pa=end_world, mean_generalized_force_n=force,
        endpoint_generalized_force_n=endpoint, force_absolute_integral_n=force_abs,
        old_stored_energy_j=float(weights@old["stored_energy_j_m2"]),
        stored_energy_j=float(weights@shifted["stored_energy_j_m2"]),
        old_fracture_dissipation_j=float(weights@old["fracture_dissipation_j_m2"]),
        fracture_dissipation_j=float(weights@shifted["fracture_dissipation_j_m2"]),
        fracture_dissipation_increment_j=float(weights@shifted["fracture_dissipation_increment_j_m2"]),
        mechanical_work_j=work, force_work_j=float(force@(new_a-old_a)),
        integrated_absolute_work_j=float(weights@absolute_work),
        mean_traction_integrated_absolute_pa=absolute_mean,
        mean_traction_absolute_operand_upper_pa=absolute_traction_upper,
        mean_traction_roundoff_operand_upper_pa=2*value_traction_roundoff+256*EPS*absolute_traction_upper,
        mean_force_roundoff_operand_upper_n=np.einsum("pai,pa,p->i", abs(local_trace),
            2*value_traction_roundoff+384*EPS*absolute_traction_upper, weights),
        work_integral_roundoff_bound_j=float(weights@(2*value_work_roundoff+384*EPS*absolute_work)))


def audit_cohesive_path(raw, expected):
    seal(raw, "cohesive_response_sha256")
    for key in ("kernel_only",):
        require(raw.get(key) is True, "private cohesive path must retain kernel scope")
    for key in ("physical_release_admitted", "normal_contact_admitted", "funding_admitted"):
        require(raw.get(key) is False, "cohesive path cannot admit physical release or funding")
    for key in ("old_gap_local_m", "new_gap_local_m", "new_maximum_separation_m"):
        close(raw[key], expected[key], "cohesive "+key, rtol=2e-8, atol=2e-13)
    for key in ("mean_traction_local_pa", "mean_traction_world_pa", "endpoint_traction_world_pa"):
        close(raw[key], expected[key], "direct cohesive path "+key, rtol=3e-9, atol=1e-5)
    for key in ("mean_generalized_force_n", "endpoint_generalized_force_n"):
        if len(expected[key]) == 18:
            mode_close(raw[key], expected[key], "direct cohesive path "+key, rtol=3e-7, atol=1e-8)
        else:
            for j in range(len(expected[key])):
                close(array(raw[key])[j], expected[key][j], "direct cohesive path "+key+" "+str(j),
                      rtol=3e-7, atol=1e-25)
    for key in ("old_stored_energy_j", "stored_energy_j", "old_fracture_dissipation_j",
                "fracture_dissipation_j", "fracture_dissipation_increment_j", "mechanical_work_j", "force_work_j"):
        close(raw[key], expected[key], "cohesive "+key, rtol=3e-9, atol=1e-9)
    for key in ("numerically_integrated_work_j",):
        close(raw[key], expected["mechanical_work_j"], "direct cohesive "+key, rtol=3e-9, atol=1e-9)
    require(raw["fracture_dissipation_increment_j"] >= 0, "fracture history cannot heal")
    if "work_integral_roundoff_bound_j" in raw:
        close(raw["work_integral_roundoff_bound_j"], expected["work_integral_roundoff_bound_j"],
            "actual-path arithmetic allowance", rtol=2e-7, atol=1e-25)
        close(raw["integrated_absolute_work_j"], expected["integrated_absolute_work_j"],
            "actual-path absolute work", rtol=3e-9, atol=1e-12)
        require(0 < raw["relative_tolerance"] <= 2e-10,
                "cohesive path tolerance cannot weaken numerical admission")
        resolved = bool(abs(raw["numerically_integrated_work_j"]-raw["mechanical_work_j"])
            <= raw["relative_tolerance"]*abs(raw["numerically_integrated_work_j"])+raw["work_integral_roundoff_bound_j"])
        require(raw["endpoint_work_agrees_at_integral_precision"] is resolved,
                "endpoint subtraction must resolve actual tiny path work")
    if "mean_traction_component_roundoff_bound_pa" in raw:
        magnitude = array(raw["mean_traction_integrated_absolute_pa"])
        point_bound = array(raw["mean_traction_component_roundoff_bound_pa"])
        upper = expected["mean_traction_absolute_operand_upper_pa"]
        roundoff_upper = expected["mean_traction_roundoff_operand_upper_pa"]
        require(magnitude.shape == point_bound.shape == upper.shape
                and np.all(magnitude >= 0) and np.all(point_bound >= 0)
                and np.all(magnitude <= 1.01*upper)
                and np.all(point_bound <= 1.01*roundoff_upper),
                "cohesive precision allowance must use only actual path operands")
        require(np.all(abs(array(raw["mean_traction_local_pa"])) <= magnitude+128*EPS*upper),
                "cohesive absolute path traction cannot be understated")
        force_bound = array(raw["mean_force_component_roundoff_bound_n"])
        order_error = array(raw["mean_force_component_order_error_n"])
        require(np.all(force_bound >= 0) and np.all(order_error >= 0)
                and np.all(force_bound <= 1.02*expected["mean_force_roundoff_operand_upper_n"])
                and np.all(order_error <= raw["relative_tolerance"]*abs(array(raw["mean_generalized_force_n"]))+force_bound),
                "cohesive vector integration error cannot supply arbitrary momentum allowance")
    return dict(direct_vector_traction_path_checked=True, endpoint_force_not_substituted_for_mean=True,
        fracture_area_counts_both_banks_once=True, signed_relative_interface_energy_retained=True)


def full_mass_drag_force(context):
    n, local = context["parent_ndof"], context["local"]
    M, D = np.zeros((n+18, n+18)), np.zeros((n+18, n+18))
    M[:n, :n], D[:n, :n] = context["original_mass"], context["original_drag"]
    dofs = context["selected_parent_dofs"]
    M[np.ix_(dofs, np.arange(n, n+18))] = local["mass_cross"]
    M[np.ix_(np.arange(n, n+18), dofs)] = local["mass_cross"].T
    M[n:, n:] = local["mass_enriched"]
    return M, D, np.r_[context["original_external"], np.zeros(18)]


def discrete_path_budget(M, D, external, increment, before_velocity, dt_s, bulk_mean_force,
        cohesive_mean_force, bulk_work_parts, cohesive_work):
    """Prospective midpoint work identity without subtracting large energies."""
    d, v0, external = array(increment), array(before_velocity), array(external)
    require(d.shape == v0.shape == external.shape and type(dt_s) in (int, float)
            and np.isfinite(dt_s) and dt_s > 0, "prospective trajectory/clock differs")
    M, D = array(M, (len(d), len(d))), array(D, (len(d), len(d)))
    v1 = 2*d/dt_s-v0
    inertia = 2*(M@(d-dt_s*v0))/(dt_s*dt_s)
    drag = D@d/dt_s
    drive = d-dt_s*v0
    kinetic_parts = np.array([
        d[:-18]@(2*(M[:-18, :-18]@drive[:-18])/(dt_s*dt_s)),
        d[:-18]@(2*(M[:-18, -18:]@drive[-18:])/(dt_s*dt_s))
          +d[-18:]@(2*(M[-18:, :-18]@drive[:-18])/(dt_s*dt_s)),
        d[-18:]@(2*(M[-18:, -18:]@drive[-18:])/(dt_s*dt_s))])
    kinetic = float(d@inertia)
    basal, work_external = float(drag@d), float(external@d)
    bulk_mean_force = array(bulk_mean_force, d.shape)
    cohesive_mean_force = array(cohesive_mean_force, (18,))
    bulk_parts = array(bulk_work_parts, (3,))
    require(np.isfinite(kinetic_parts).all() and basal >= 0, "kinetic/basal path work differs")
    r = inertia+drag+bulk_mean_force+np.r_[np.zeros(len(d)-18), cohesive_mean_force]-external
    # Small compact work stays separate before it is combined with the old
    # budget.  Historical Maxwell losses and stored-energy ULPs are absent.
    balance_parts = np.r_[kinetic_parts, basal, bulk_parts, cohesive_work, -work_external]
    balance = float(np.sum(balance_parts, dtype=np.longdouble))
    force_work = float(np.sum(np.asarray(r, np.longdouble)*d))
    return dict(after_velocity_m_s=v1, inertia_force_n=inertia, basal_drag_force_n=drag,
        external_force_n=external, residual_n=r, kinetic_change_decomposition_j=kinetic_parts,
        kinetic_energy_change_j=kinetic, basal_dissipation_j=basal,
        external_work_j=work_external, bulk_work_decomposition_j=bulk_parts,
        cohesive_work_j=float(cohesive_work), work_balance_residual_j=balance,
        residual_dual_work_j=force_work)


def restrict_point_operators(G, E, B, T, weights, C, density, basis):
    """Restrict literal source point fields before any Gram contraction.

    Projecting an already rounded Gram matrix loses genuine weak directions.
    These long-double contractions describe the archived float64 point field;
    they do not assert new physical accuracy of that field or invent a floor.
    """
    G, E, B, T = (np.asarray(array(a), np.longdouble) for a in (G, E, B, T))
    weights, C, W = (np.asarray(array(a), np.longdouble) for a in (weights, C, basis))
    require(G.ndim == 3 and G.shape[1:] == (6, 18) and E.shape == (len(G), 3, 18)
            and B.shape == G.shape and T.shape == E.shape and weights.shape == (len(G),)
            and C.shape == (6, 6) and W.ndim == 2 and W.shape[0] == 18
            and np.all(weights >= 0) and np.any(weights > 0)
            and type(density) in (int, float) and density > 0,
            "literal source restricted point dimensions/material differ")
    Gw = np.einsum("pai,ij->paj", G, W)
    Ew = np.einsum("pai,ij->paj", E, W)
    gram = lambda A, L, R: np.einsum("pki,kl,plj,p->ij", L, A, R, weights)
    return dict(basis_w=W, G_w=Gw, E_w=Ew, parent_strain_operator=B,
        parent_trace=T, reference_volume_weights_m3=weights, elasticity_pa=C,
        density_kg_m3=np.longdouble(density),
        stiffness_cross=gram(C, B, Gw), stiffness_enriched=gram(C, Gw, Gw),
        mass_cross=np.longdouble(density)*np.einsum("pki,pkj,p->ij", T, Ew, weights),
        mass_enriched=np.longdouble(density)*np.einsum("pki,pkj,p->ij", Ew, Ew, weights))


def independent_restricted_source_quadrature(context, prior_case, W, chart):
    """Audit source inputs freshly, then reuse their immutable represented data."""
    p = prior_case["embedded_partition"]
    report = memory.audit_partition_basis(p, context)
    local = p["local"]
    quad = restrict_point_operators(local["enrichment_strain_operator"], local["enrichment_trace"],
        local["parent_strain_operator"], local["parent_trace"], local["reference_volume_weights_m3"],
        local["elasticity_pa"], local["section"]["mass_kg"]/local["section"]["volume_m3"], W)
    close(float(quad["density_kg_m3"]), context["parent"]["runtime_state"]["geometry"]["density_kg_m3"],
          "literal source section density", rtol=3e-15, atol=0)
    quad["barycentric"] = np.asarray(array(local["barycentric"]), np.longdouble)
    quad["radial_scale"] = np.asarray(array(local["radial_scale"]), np.longdouble)
    # The source angular nodes are an exact constant convention of the frozen
    # memory representation; only the literal point locations are archived.
    quad["interpolation"] = quad["barycentric"]@np.asarray(
        np.linalg.inv(seed.prior.SOURCE_ANGULAR_POINTS), np.longdouble)
    quad["surface_trace_local_w"] = np.asarray(array(chart["compact_jump_trace_local"]), np.longdouble)
    quad["surface_area_weights_m2"] = np.asarray(array(chart["reference_area_weights_m2"]), np.longdouble)
    quad["surface_gram"] = np.einsum("pki,pkj,p->ij", quad["surface_trace_local_w"],
        quad["surface_trace_local_w"], quad["surface_area_weights_m2"])
    # Matrix descriptions pin the public float64 reduction of a stated
    # long-double summation order.  Forces and energy below are also rebuilt
    # directly from point integrals, independently of these described Grams.
    Cg = np.einsum("ab,pbi->pai", quad["elasticity_pa"], quad["G_w"])
    descriptor_arrays = {key: quad[key] for key in ("basis_w", "G_w", "E_w",
        "parent_strain_operator", "parent_trace", "reference_volume_weights_m3", "elasticity_pa",
        "mass_cross", "mass_enriched", "surface_trace_local_w", "surface_area_weights_m2", "surface_gram")}
    descriptor_arrays["stiffness_cross"] = np.einsum("pai,paj,p->ij",
        quad["parent_strain_operator"], Cg, quad["reference_volume_weights_m3"])
    descriptor_arrays["stiffness_enriched"] = np.einsum("pai,paj,p->ij",
        quad["G_w"], Cg, quad["reference_volume_weights_m3"])
    quad["array_sha256"] = {key: numeric_array_pin(value) for key, value in descriptor_arrays.items()}
    n = context["parent_ndof"]
    # Fresh source reconstruction above certifies these unchanged QQ blocks.
    # Their literal saved arithmetic, rather than a second quadrature's last
    # few bits, is the operator used by the prospective original DOFs.
    quad["original_mass"] = seed.dense_csr(p["mass_matrix"], "saved source mass")[:n, :n]
    quad["original_drag"] = seed.dense_csr(p["basal_drag_matrix"], "saved source drag")[:n, :n]
    quad["original_external"] = array(p["external_force_n"])[:n]
    quad["source_report"] = dict(fresh_geometry_material_memory_point_input_comparisons=report,
        immutable_prior_literal_float64_point_arrays_reused=True,
        restrictions_applied_before_gram_in_longdouble=True,
        fresh_physical_weak_source_accuracy_certified=False,
        physical_clipped_bulk_accuracy_certified=False)
    return quad


def restricted_inner_parts(quad, parent_a, added_a, parent_b, added_b):
    """Signed cross/self integrals, separately from the enormous old ledger."""
    w, C = quad["reference_volume_weights_m3"], quad["elasticity_pa"]
    inner = lambda x, y: np.einsum("pi,ij,pj,p->", x, C, y, w)
    return np.asarray((inner(parent_a, added_b)+inner(added_a, parent_b),
                       inner(added_a, added_b)), np.longdouble)


def archived_legacy_small_response(context, prior_case, previous, bulk, inc0, inc1, d, ratio):
    """Literal old rounded actions, preserved only as arithmetic witnesses."""
    local, ci = prior_case["embedded_partition"]["local"], context["selected_index"]
    ld = np.longdouble
    interpolation = np.asarray(array(local["barycentric"]), ld)@np.asarray(
        np.linalg.inv(seed.prior.SOURCE_ANGULAR_POINTS), ld)
    lam, weights = np.asarray(array(local["radial_scale"]), ld), np.asarray(array(local["reference_volume_weights_m3"]), ld)
    G, C = np.asarray(array(local["enrichment_strain_operator"]), ld), np.asarray(array(local["elasticity_pa"]), ld)
    Kqa, Kaa = array(local["stiffness_cross"]), array(local["stiffness_enriched"])
    def moment(e0, e1):
        point = interpolation@np.asarray(e0[ci], ld)+(interpolation@np.asarray(e1[ci], ld))/lam[:, None]
        return np.asarray(np.einsum("pai,pa,p->i", G, point@C.T, weights), float)
    def small(a0, a1, za, b0, b1, zb):
        cross = 0.
        if np.any(za): cross += float(za@moment(b0, b1))
        if np.any(zb): cross += float(zb@moment(a0, a1))
        return np.asarray((cross, float(za@Kaa@zb)))
    old0, old1, z, da = previous["e0"], previous["e1"], previous["z"], d[context["parent_ndof"]:]
    end0, end1, zend = bulk["strain0"], bulk["strain1"], array(bulk["coefficients_m"])
    mean0, mean1, zm = array(bulk["mean0"]), array(bulk["mean1"]), array(bulk["mean_coefficients_m"])
    before = .5*small(old0, old1, z, old0, old1, z)
    after = .5*small(end0, end1, zend, end0, end1, zend)
    work = small(inc0, inc1, da, mean0, mean1, zm)
    x = np.asarray(ratio, float)
    beta = np.divide(-np.expm1(-x), x, out=np.ones_like(x), where=x != 0)
    rr = np.divide(-np.expm1(-2*x), 2*x, out=np.ones_like(x), where=x != 0)
    aa, ab = x*rr, 2*(beta-rr)
    bb = np.divide(1-2*beta+rr, x, out=np.zeros_like(x), where=x != 0)
    tiny = x < 1e-3
    ab[tiny] = x[tiny]-x[tiny]**2+7*x[tiny]**3/12-x[tiny]**4/4
    bb[tiny] = x[tiny]/3-x[tiny]**2/4+7*x[tiny]**3/60-x[tiny]**4/24
    shift = np.divide(ab, 2*aa, out=np.zeros_like(x), where=aa != 0)
    remainder = np.maximum(bb-aa*shift*shift, 0.)
    shifted0, shifted1, zs = old0+shift[:, None, None]*inc0, old1+shift[:, None, None]*inc1, z+shift[ci]*da
    loss = aa[ci]*small(shifted0, shifted1, zs, shifted0, shifted1, zs)+remainder[ci]*small(inc0, inc1, da, inc0, inc1, da)
    return dict(before=before, after=after, work=work, loss=loss,
        cross_parent=Kqa@zm, cross_added=moment(mean0, mean1), self_added=Kaa@zm)


def journal_interface_maximum(checked, raw, *, operative):
    source = checked["operative_source"] if operative else checked["legacy_source"]
    local_trace = np.einsum("pji,pjk->pik", source["basis"], source["trace"])
    maximum, a = checked["initial_maximum"].copy(), np.zeros(18)
    reference = checked["reference_offset"]
    for item in raw["state"]["journal"]:
        a += array(item["enrichment_increment_m"])
        value = np.asarray(np.asarray(checked["modes"]["observable_force_basis"], np.longdouble).T@
            np.asarray(a, np.longdouble), float) if operative else a
        h = reference+np.einsum("pai,i->pa", local_trace, value)
        positive = h.copy(); positive[:, 0] = np.maximum(positive[:, 0], 0)
        maximum = np.maximum(maximum, np.linalg.norm(positive, axis=1))
    return maximum


def restricted_current_path(context, quad, previous, bulk, inc0, inc1, d, v0, dt, ratio,
        mean_traction_local, original_work_parts, original_loss_parts,
        original_before_energy_parts, original_after_energy_parts):
    """Current represented-field work and forces; no subtraction of old totals."""
    ld = np.longdouble
    W, G, E = quad["basis_w"], quad["G_w"], quad["E_w"]
    n, ci = context["parent_ndof"], context["selected_index"]
    d, v0 = np.asarray(d, ld), np.asarray(v0, ld)
    q, a = d[:n], W.T@d[n:]
    v = np.r_[v0[:n], W.T@v0[n:]]
    operative_d, drive = np.r_[q, a], np.r_[q, a]-ld(dt)*v
    selected = context["selected_parent_dofs"]
    point_parent = lambda e0, e1: quad["interpolation"]@np.asarray(e0[ci], ld)+(quad["interpolation"]@np.asarray(e1[ci], ld))/quad["radial_scale"][:, None]
    old_parent = point_parent(previous["e0"], previous["e1"])
    inc_parent = np.einsum("pai,i->pa", quad["parent_strain_operator"], q[selected])
    mean_parent = point_parent(bulk["mean0"], bulk["mean1"])
    end_parent = point_parent(bulk["strain0"], bulk["strain1"])
    old_added = G@(W.T@np.asarray(previous["z"], ld))
    inc_added = G@a
    mean_added = G@(W.T@np.asarray(bulk["mean_coefficients_m"], ld))
    end_added = G@(W.T@np.asarray(bulk["coefficients_m"], ld))
    parts = lambda pa, ga, pb, gb: restricted_inner_parts(quad, pa, ga, pb, gb)
    small_before = parts(old_parent, old_added, old_parent, old_added)/2
    small_after = parts(end_parent, end_added, end_parent, end_added)/2
    small_work = parts(inc_parent, inc_added, mean_parent, mean_added)
    temporal = np.asarray(memory.volume.temporal_factors(ratio[ci]), ld)
    small_loss = (temporal[2]*parts(old_parent, old_added, old_parent, old_added)
        +temporal[3]*parts(old_parent, old_added, inc_parent, inc_added)
        +temporal[4]*parts(inc_parent, inc_added, inc_parent, inc_added))
    work_parts = np.r_[ld(original_work_parts[0]), small_work]
    loss_parts = np.r_[ld(original_loss_parts[0]), small_loss]
    before_parts = np.r_[ld(original_before_energy_parts[0]), small_before]
    after_parts = np.r_[ld(original_after_energy_parts[0]), small_after]
    C, weights = quad["elasticity_pa"], quad["reference_volume_weights_m3"]
    force = lambda B, e: np.einsum("pki,kl,pl,p->i", B, C, e, weights)
    original_force = np.asarray(memory.full_generalized_force(context, bulk["mean0"],
        bulk["mean1"], np.zeros(18))[:n], ld)
    cross_parent = force(quad["parent_strain_operator"], mean_added)
    cross_added, self_added = force(G, mean_parent), force(G, mean_added)
    bulk_force = np.r_[original_force.copy(), cross_added+self_added]
    bulk_force[selected] += cross_parent
    cross_parent_inertia = 2*(quad["mass_cross"]@drive[n:])/ld(dt)**2
    cross_added_inertia = 2*(quad["mass_cross"].T@drive[selected])/ld(dt)**2
    self_added_inertia = 2*(quad["mass_enriched"]@drive[n:])/ld(dt)**2
    old_inertia = np.asarray(2*(quad["original_mass"]@np.asarray(drive[:n], float))/float(dt)**2, ld)
    inertia = np.r_[old_inertia.copy(), cross_added_inertia+self_added_inertia]
    inertia[selected] += cross_parent_inertia
    kinetic_parts = np.asarray((q@old_inertia,
        q[selected]@cross_parent_inertia+a@cross_added_inertia, a@self_added_inertia), ld)
    drag = np.r_[np.asarray(quad["original_drag"]@np.asarray(q, float)/float(dt), ld), np.zeros(W.shape[1], ld)]
    external = np.r_[np.asarray(quad["original_external"], ld), np.zeros(W.shape[1], ld)]
    gamma = np.r_[np.zeros(n, ld), np.einsum("pki,pk,p->i", quad["surface_trace_local_w"],
        np.asarray(mean_traction_local, ld), quad["surface_area_weights_m2"])]
    force_work_parts = np.asarray((q@original_force,
        q[selected]@cross_parent+a@cross_added, a@self_added), ld)
    return dict(operative_increment_m=operative_d, operative_before_velocity_m_s=v,
        operative_after_velocity_m_s=2*operative_d/ld(dt)-v,
        operative_inertia_force_n=inertia, operative_basal_drag_force_n=drag,
        operative_bulk_mean_force_n=bulk_force, operative_cohesive_mean_force_n=gamma,
        operative_external_force_n=external,
        bulk_work_decomposition_j=work_parts, bulk_dissipation_decomposition_j=loss_parts,
        bulk_energy_before_decomposition_j=before_parts, bulk_energy_after_decomposition_j=after_parts,
        bulk_energy_increment_decomposition_j=np.r_[work_parts[0]-loss_parts[0],
            np.einsum("pi,ij,pj,p->", end_parent-old_parent, C, end_added, weights)
                +np.einsum("pi,ij,pj,p->", old_parent, C, end_added-old_added, weights),
            .5*np.einsum("pi,ij,pj,p->", end_added-old_added, C, end_added+old_added, weights)],
        bulk_force_work_decomposition_j=force_work_parts,
        bulk_parent_mean_force_n=original_force, bulk_cross_parent_force_n=cross_parent,
        operative_bulk_cross_enrichment_force_n=cross_added, operative_bulk_self_enrichment_force_n=self_added,
        kinetic_change_decomposition_j=kinetic_parts,
        kinetic_change_stable_j=np.sum(kinetic_parts, dtype=ld),
        basal_dissipation_j=operative_d@drag, external_work_j=operative_d@external,
        cohesive_work_j=operative_d@gamma)


def reduction_roundoff(magnitude, count, *, epsilon=None):
    eps = np.longdouble(np.finfo(np.longdouble).eps if epsilon is None else epsilon)
    factor = count*eps
    require(type(count) is int and count > 0 and factor < 1,
            "actual operand reduction precision is unresolved")
    return factor/(1-factor)*np.asarray(magnitude, np.longdouble)


def source_expansion_witnesses(full_fields, restricted_fields, W, vectors):
    """Check represented field preservation at its own lift operands."""
    W = np.asarray(array(W), np.longdouble)
    reports, all_passed = {}, True
    for name, value in vectors.items():
        a = np.asarray(array(value, (18,)), np.longdouble)
        coordinates = W.T@a
        lift_magnitude = abs(W)@(abs(W).T@abs(a))
        for field_name, full_raw in full_fields.items():
            array(full_raw); array(restricted_fields[field_name])
            full, restricted = np.asarray(full_raw, np.longdouble), np.asarray(restricted_fields[field_name], np.longdouble)
            literal = np.einsum("pai,i->pa", full, a)
            operative = np.einsum("pai,i->pa", restricted, coordinates)
            magnitude = (np.einsum("pai,i->pa", abs(full), abs(a)+lift_magnitude)
                +np.einsum("pai,i->pa", abs(restricted), abs(coordinates)))
            bound, error = 256*EPS*magnitude, abs(literal-operative)
            passed = bool(np.all(error <= bound))
            all_passed = all_passed and passed
            relative = np.divide(error, bound, out=np.zeros_like(error), where=bound != 0)
            reports[name+"_"+field_name] = dict(maximum_absolute_difference=float(np.max(error, initial=0)),
                maximum_operand_arithmetic_bound=float(np.max(bound, initial=0)),
                maximum_difference_to_arithmetic_bound_ratio=float(np.max(relative, initial=0)), passed=passed)
    return reports, all_passed


def independent_current_operand_bounds(context, quad, previous, bulk, d, v0, dt, gamma_raw, parent_force):
    """Roundoff from current force/integral operands, without a force floor."""
    ld = np.longdouble
    n, ci, W = context["parent_ndof"], context["selected_index"], quad["basis_w"]
    q, da = np.asarray(d[:n], ld), W.T@np.asarray(d[n:], ld)
    drive = np.r_[q, da]-ld(dt)*np.r_[np.asarray(v0[:n], ld), W.T@np.asarray(v0[n:], ld)]
    selected = context["selected_parent_dofs"]
    G, E, B, T, C, weights = (abs(quad[key]) for key in ("G_w", "E_w", "parent_strain_operator",
        "parent_trace", "elasticity_pa", "reference_volume_weights_m3"))
    Kqa = np.einsum("pai,ab,pbj,p->ij", B, C, G, weights)
    Kaa = np.einsum("pai,ab,pbj,p->ij", G, C, G, weights)
    Mqa = quad["density_kg_m3"]*np.einsum("pai,paj,p->ij", T, E, weights)
    Maa = quad["density_kg_m3"]*np.einsum("pai,paj,p->ij", E, E, weights)
    count = 32+36*len(weights)
    zm = W.T@np.asarray(bulk["mean_coefficients_m"], ld)
    interpolation, lam = quad["interpolation"], quad["radial_scale"]
    emean = interpolation@np.asarray(bulk["mean0"][ci], ld)+(interpolation@np.asarray(bulk["mean1"][ci], ld))/lam[:, None]
    cross_q_bound = reduction_roundoff(Kqa@abs(zm), count+W.shape[1])
    cross_w_bound = reduction_roundoff(np.einsum("pai,ab,pb,p->i", G, C, abs(emean), weights), count)
    self_w_bound = reduction_roundoff(Kaa@abs(zm), count+W.shape[1])
    # Original analytic force operands are independently formed from fresh
    # geometry/material and the unchanged stored angular coefficient fields.
    radial, rw = context["radial_scale"], context["radial_weights"]
    invmean = np.sum(rw/radial, axis=1)
    invvar = np.sum(rw*(1/radial-invmean[:, None])**2, axis=1)
    b0, b1 = abs(np.asarray(context["B0"], ld)), abs(np.asarray(context["B1"], ld))
    e0, e1 = abs(np.asarray(bulk["mean0"], ld)), abs(np.asarray(bulk["mean1"], ld))
    elasticity = abs(np.asarray(context["elasticity_pa"], ld))
    im, var = np.asarray(invmean, ld), np.asarray(invvar, ld)
    stress = np.einsum("cab,cqb->cqa", elasticity, e0+im[:, None, None]*e1)
    inverse = np.einsum("cab,cqb->cqa", elasticity, e1)
    parent_local = (np.einsum("cqai,cqa->ci", b0+im[:, None, None, None]*b1, stress)
        +var[:, None]*np.einsum("cqai,cqa->ci", b1, inverse))*np.asarray(context["volume_m3"], ld)[:, None]/3
    parent_magnitude = np.zeros(n, ld)
    np.add.at(parent_magnitude, context["dofs"].ravel(), parent_local.ravel())
    parent_bound = reduction_roundoff(parent_magnitude, 128+3*len(context["cohort_ids"]), epsilon=EPS)
    bulk_bound = np.r_[parent_bound.copy(), cross_w_bound+self_w_bound]
    np.add.at(bulk_bound, selected, cross_q_bound)
    factor = 2/ld(dt)**2
    old_mass_magnitude = np.asarray(abs(quad["original_mass"])@np.asarray(abs(drive[:n]), float), ld)*factor
    inertia = np.r_[reduction_roundoff(old_mass_magnitude, n+32, epsilon=EPS),
        reduction_roundoff(factor*(Mqa.T@abs(drive[selected])), count+len(selected)+32)
        +reduction_roundoff(factor*(Maa@abs(drive[n:])), count+W.shape[1]+32)]
    np.add.at(inertia, selected, reduction_roundoff(factor*(Mqa@abs(drive[n:])), count+W.shape[1]+32))
    drag_magnitude = np.asarray(abs(quad["original_drag"])@np.asarray(abs(q), float), ld)/ld(dt)
    drag = np.r_[reduction_roundoff(drag_magnitude, n+32, epsilon=EPS), np.zeros(W.shape[1], ld)]
    J, area = abs(quad["surface_trace_local_w"]), quad["surface_area_weights_m2"]
    gamma_magnitude = np.einsum("pai,pa,p->i", J,
        np.asarray(gamma_raw["mean_traction_integrated_absolute_pa"], ld), area)
    point_gamma_roundoff = np.einsum("pai,pa,p->i", J,
        np.asarray(gamma_raw["mean_traction_component_roundoff_bound_pa"], ld), area)
    gamma = np.r_[np.zeros(n, ld), reduction_roundoff(gamma_magnitude, 32+3*len(area))
        +point_gamma_roundoff+np.asarray(gamma_raw["mean_force_component_order_error_n"], ld)]
    external = np.zeros(n+W.shape[1], ld)  # genuine source has literal zero added load
    signed_G, signed_C = quad["G_w"], quad["elasticity_pa"]
    eq = np.einsum("pai,i->pa", quad["parent_strain_operator"], q[selected])
    edw, ezm = np.einsum("pai,i->pa", signed_G, da), np.einsum("pai,i->pa", signed_G, zm)
    absolute_inner = lambda x, y: np.einsum("pa,ab,pb,p->", abs(x), C, abs(y), weights)
    work_magnitudes = np.asarray((abs(np.asarray(parent_force, ld))@abs(q),
        absolute_inner(eq, ezm)+absolute_inner(edw, emean), absolute_inner(edw, ezm)), ld)
    cross_q = quad["stiffness_cross"]@zm
    cross_w = np.einsum("pai,ab,pb,p->i", signed_G, signed_C, emean, weights)
    self_w = quad["stiffness_enriched"]@zm
    dot_magnitudes = np.asarray((work_magnitudes[0], abs(cross_q)@abs(q[selected])+abs(cross_w)@abs(da), abs(self_w)@abs(da)), ld)
    old_dot_bound = 128*EPS*float(work_magnitudes[0])
    work_bounds = np.r_[old_dot_bound, reduction_roundoff(work_magnitudes[1:], count)]
    dot_bounds = np.r_[old_dot_bound, reduction_roundoff(dot_magnitudes[1:], n+18+32)]
    dot_bounds[0] += parent_bound@abs(q)
    dot_bounds[1] += cross_q_bound@abs(q[selected])+cross_w_bound@abs(da)
    dot_bounds[2] += self_w_bound@abs(da)
    return dict(operative_inertia_component_roundoff_bound_n=inertia,
        operative_basal_drag_component_roundoff_bound_n=drag, operative_bulk_component_roundoff_bound_n=bulk_bound,
        operative_cohesive_component_roundoff_bound_n=gamma, operative_external_component_roundoff_bound_n=external,
        bulk_work_operand_magnitude_decomposition_j=work_magnitudes,
        bulk_force_dot_operand_magnitude_decomposition_j=dot_magnitudes,
        bulk_work_roundoff_bound_decomposition_j=work_bounds,
        bulk_force_dot_roundoff_bound_decomposition_j=dot_bounds)


def audit_source_proposal(raw, parent):
    """Preserve the original no-event proposal independently of the new law."""
    require(type(raw) is dict, "source nucleation proposal required")
    seal(raw, "nucleation_sha256")
    onset = raw["source_onset"]
    insertion.audit_onset(onset, parent)
    # Observer byte bindings are opaque links.  Numerical surface stresses
    # below are independently obtained from the complete authentic memory.
    observer = {"binding_sha256": raw["observer_binding_sha256"]}
    seed.prior.audit_nucleation(raw, parent, observer, onset, None)
    require(raw["declared_prospective_law"] is None
            and raw["source_recovered_creation_energy_law"] is None,
            "computational cohesive law cannot become recovered Genesis history")
    return dict(source_no_calculated_activation_event=True,
        original_unconfigured_source_law_retained=True, physical_selection_inferred=False)


def audit_release_context(raw, parent, prior_case, *, context=None):
    seal(raw, "release_work_context_sha256")
    scalar_zero_credit(raw)
    for key in ("physical_creation_admitted", "physical_release_admitted", "funding_admitted"):
        require(raw.get(key) is False, "private release context physical scope differs")
    p = prior_case["embedded_partition"]
    birth = prior_case["source_memory_birth"]
    expected_pins = dict(partition_binding_sha256=p["embedded_partition_sha256"],
        finite_seed_binding_sha256=prior_case["finite_seed_binding_sha256"],
        source_memory_birth_binding_sha256=digest(birth))
    for key, value in expected_pins.items():
        exact(raw[key], value, "source context "+key)
    exact(raw["release_modes"]["source_binding_sha256"], p["embedded_partition_sha256"],
          "complete kernel operator-source link")
    for key, value in (("accepted_parent_checkpoint_sha256", PARENT_CHECKPOINT_SHA256),
                       ("formation_checkpoint_sha256", FORMATION_SHA256)):
        exact(raw[key], value, "genuine source "+key)
    proposal = audit_source_proposal(raw["source_nucleation_proposal"], parent)
    expected_refusals = ["accepted_Genesis_has_no_new_source_activation",
        "no_source_selected_prospective_nucleation_law",
        "computational_seed_not_selected_by_Genesis_source_proposal",
        "prospective_creation_parameters_are_explicit_uncalibrated_computational_input",
        "source_selected_clipped_bulk_accuracy_not_certified",
        "moving_thermal_material_boundaries_require_mass_heat_and_work_transfer",
        "full_time_normal_contact_and_all_pair_underthrust_not_admitted",
        "archived_interval_reaction_not_replaced_by_initial_surface_traction"]
    exact(raw["physical_refusals"], expected_refusals, "genuine physical admission refusals")
    exact(raw["parameter_provenance"], "explicit_new_computational_input_not_historical_Genesis_toughness",
          "new cohesive law provenance")
    require(raw["complete_jump_kernel_remains_tied"] is True
            and raw["arithmetic_difference_is_available_work"] is False,
            "kernel/roundoff cannot pay for physical release")
    exact(raw["fault_history_sha256"], digest(parent["runtime_state"]["fault_history"]),
          "complete authentic fault memory link")
    exact(raw["source_identity"], parent["runtime_state"]["geometry"]["source_identity"], "genuine material identity")
    exact(raw["accepted_time_myr"], parent["runtime_state"]["geometry"]["time_myr"], "unaltered accepted time")
    definition = prior_case["seed_definition"]
    source = source_surface(parent, definition)
    for key, wanted in (("surface_trace_xyz", source["trace"]),
                        ("reference_area_weights_m2", source["weights"]),
                        ("initial_reference_traction_world_pa", source["initial_traction_world"]),
                        ("birth_basis_xyz", source["basis"])):
        close(raw[key], wanted, "source release chart "+key, rtol=2e-8,
            atol=1e-5 if "traction" in key else 2e-12)
    # Geometry and each input field were freshly compared above.  The new
    # numerical restriction acts on the literal represented source trace;
    # weak geometry accuracy beyond those input comparisons is not certified.
    literal_trace, literal_weights = array(raw["surface_trace_xyz"]), array(raw["reference_area_weights_m2"])
    modes = audit_jump_modes(raw["release_modes"], literal_trace, literal_weights)
    require(modes["nullity"] >= 3, "material-plane null modes cannot be released")
    W = array(raw["observable_force_basis"], (18, modes["rank"]))
    close(W.T@W, np.eye(modes["rank"]), "observable-force orthonormality", rtol=0, atol=4e-13)
    close(W@W.T, modes["release_projector"], "complete observable force subspace", rtol=2e-7, atol=2e-8)
    weighted = (np.sqrt(literal_weights)[:, None, None]*literal_trace).reshape(-1, 18)
    singular = np.linalg.svd(weighted@modes["declared_release_basis"], compute_uv=False)
    declared_singular = array(raw["observable_force_singular_values_m"], (modes["rank"],))
    for j in range(modes["rank"]):
        close(declared_singular[j], singular[j], "observable weak force singular value "+str(j),
              rtol=2e-6, atol=1e-16)
    sampled = weighted@W
    for j in range(modes["rank"]):
        close(np.linalg.norm(sampled[:, j]), declared_singular[j], "observable force column norm "+str(j),
              rtol=2e-6, atol=1e-16)
        for k in range(j):
            product = float(sampled[:, j]@sampled[:, k])
            arithmetic = 256*EPS*float(np.sum(abs(sampled[:, j]*sampled[:, k])))
            require(abs(product) <= 2e-6*declared_singular[j]*declared_singular[k]+arithmetic,
                    "observable force directions cannot mix unresolved weak and strong modes")
    modes["observable_force_basis"] = W
    law = raw["prospective_creation_law"]
    creation.validate_law(law)
    cohesive = raw["cohesive_chart"]
    seal(cohesive, "cohesive_chart_sha256")
    require(cohesive.get("physical_release_admitted") is False
            and cohesive.get("normal_contact_admitted") is False,
            "cohesive chart cannot admit release/contact")
    local_trace = np.einsum("pji,pjk->pik", source["basis"], source["trace"])
    local_t0 = np.einsum("pji,pj->pi", source["basis"], source["initial_traction_world"])
    expected_seed = seed.prior.cohesive_prestress_seed(law, local_t0)
    exact(cohesive["seed"]["law"], law, "cohesive declared law")
    for key, wanted in expected_seed.items():
        if key == "law": continue
        if isinstance(wanted, np.ndarray):
            close(cohesive["seed"][key], wanted, "prestress seed "+key,
                rtol=2e-8, atol=1e-5 if "traction" in key else 1e-11)
        else:
            exact(cohesive["seed"][key], wanted, "prestress seed "+key)
    for key, wanted in (("compact_jump_trace_world", source["trace"]),
                        ("compact_jump_trace_local", local_trace),
                        ("reference_area_weights_m2", source["weights"]),
                        ("birth_basis_xyz", source["basis"])):
        close(cohesive[key], wanted, "cohesive fixed chart "+key, rtol=2e-8, atol=2e-12)
    initial_force = np.einsum("pai,pa,p->i", source["trace"],
                             source["initial_traction_world"], source["weights"])
    archived = array(prior_case["accepted_interval_reaction"]["constraint_multipliers_n"], (18,))
    exact(raw["archived_constraint_multipliers_n"], archived.tolist(), "archived last-interval tied reaction")
    mode_close(raw["initial_surface_force_n"], initial_force, "endpoint reference surface force", atol=1e-7)
    mode_close(raw["archived_reaction_minus_initial_surface_force_n"], archived-initial_force,
        "different-scope reaction minus surface diagnostic", atol=1e-7)
    v0 = np.r_[array(parent["runtime_state"]["velocity_m_s"]), np.zeros(18)]
    exact(raw["initial_velocity_m_s"], v0.tolist(), "genuine accepted velocity with new modes zero")
    require(type(raw.get("physical_refusals")) is list and len(raw["physical_refusals"]) >= 3,
            "physical no-event/missing-law/missing-accuracy refusals omitted")
    if context is None:
        context = memory.source_context(parent, definition, order=12)
    context["birth_operator_energy_roundoff_j"] = prior_case["birth_memory"]["state"]["birth_operator_energy_roundoff_j"]
    operative_chart = raw["operative_cohesive_chart"]
    seal(operative_chart, "cohesive_chart_sha256")
    world_w = np.asarray(np.einsum("pai,ij->paj", np.asarray(array(
        cohesive["compact_jump_trace_world"]), np.longdouble), np.asarray(W, np.longdouble)), float)
    local_w = np.einsum("pji,pjk->pik", array(cohesive["birth_basis_xyz"]), world_w)
    for key in cohesive:
        if key in {"cohesive_chart_sha256", "compact_jump_trace_world", "compact_jump_trace_local"}:
            continue
        exact(operative_chart[key], cohesive[key], "operative cohesive source chart "+key)
    close(operative_chart["compact_jump_trace_world"], world_w,
          "source trace restricted before surface Gram", rtol=0, atol=0)
    close(operative_chart["compact_jump_trace_local"], local_w,
          "restricted source trace in birth frame", rtol=0, atol=0)
    quad = independent_restricted_source_quadrature(context, prior_case, W, operative_chart)
    exact(raw["operative_bulk_field_scope"], QUADRATURE_SCOPE, "operative source arithmetic scope")
    require(raw["weak_geometry_accuracy_certified"] is False,
            "computational source restriction cannot assert physical weak accuracy")
    operative_source = dict(trace=world_w, weights=array(cohesive["reference_area_weights_m2"]),
        initial_traction_world=array(raw["initial_reference_traction_world_pa"]),
        basis=array(cohesive["birth_basis_xyz"]))
    legacy_source = {**operative_source, "trace": array(cohesive["compact_jump_trace_world"])}
    return dict(numerical_context=context, source=source, modes=modes,
        operative_source=operative_source, legacy_source=legacy_source, restricted_quadrature=quad,
        source_proposal=proposal, law=law, initial_force=initial_force,
        archived_reaction=archived, before_velocity=v0,
        initial_maximum=expected_seed["initial_maximum_separation_m"],
        reference_offset=expected_seed["reference_offset_m"],
        context_binding_sha256=digest(raw), physical_refusals=expected_refusals,
        context_report=dict(complete_kernel_tied=True, rank=modes["rank"], nullity=modes["nullity"],
            restricted_source_quadrature=quad["source_report"],
            archived_reaction_scope="hypothetical_frozen_endpoint_basis_over_saved_last_interval",
            surface_traction_scope="accepted_endpoint_small_strain_reference_area_boundary_traction",
            reaction_replacement_admitted=False, source_proposal=proposal))


def replay_memory_archive(raw, checked, parent, prior_case):
    """Replay all private increments from the authentic material birth."""
    context, modes, source = checked["numerical_context"], checked["modes"], checked["operative_source"]
    seal(raw, "packet_sha256")
    original = parent["runtime_state"]["mechanics"]["state"]
    e0, e1 = context["inherited0"].copy(), context["inherited1"].copy()
    z, a = np.zeros(18), np.zeros(18)
    time = parent["runtime_state"]["geometry"]["time_myr"]
    velocity = checked["before_velocity"].copy()
    maximum = checked["initial_maximum"].copy()
    work, loss, energy = original["mechanical_work_j"], original["viscous_dissipation_j"], original["elastic_energy_j"]
    require(type(raw["state"]["journal"]) is list, "source memory journal required")
    for item in raw["state"]["journal"]:
        require(set(item) == {"after_time_myr", "parent_increment_m", "enrichment_increment_m", "ratio"},
                "private journal fields differ")
        q = array(item["parent_increment_m"], (context["parent_ndof"],))
        da = array(item["enrichment_increment_m"], (18,))
        require_tied_kernel(modes, da, "journal compact increment")
        after = item["after_time_myr"]
        require(type(after) in (int, float) and after > time, "journal private clock differs")
        ratio = memory.source_maxwell_ratios(parent, time, after)
        close(item["ratio"], ratio, "journal source Maxwell factors", rtol=2e-9, atol=2e-13)
        inc0 = np.einsum("cqai,ci->cqa", context["B0"], q[context["dofs"]])
        inc1 = np.einsum("cqai,ci->cqa", context["B1"], q[context["dofs"]])
        step = memory.maxwell_step(context, e0, e1, z, inc0, inc1, da, ratio)
        local_trace = np.einsum("pji,pjk->pik", source["basis"], source["trace"])
        local_t0 = np.einsum("pji,pj->pi", source["basis"], source["initial_traction_world"])
        coordinates = lambda value: np.asarray(np.asarray(modes["observable_force_basis"],
            np.longdouble).T@np.asarray(value, np.longdouble), float)
        cohesive = insertion.prestress_segment(checked["law"], local_t0,
            np.einsum("pai,i->pa", local_trace, coordinates(a)),
            np.einsum("pai,i->pa", local_trace, coordinates(a+da)), maximum)
        maximum = cohesive["maximum_separation_m"]
        a = a+da
        e0, e1, z = step["strain0"], step["strain1"], step["coefficients_m"]
        velocity = 2*np.r_[q, da]/((after-time)*memory.SECONDS_PER_MYR)-velocity
        time = after
        work += step["mechanical_work_j"]; loss += step["viscous_dissipation_j"]
        energy = step["elastic_energy_j"]
    report = memory.audit_state_archive(raw, context,
        prior_case["embedded_partition"]["embedded_partition_sha256"],
        expected0=e0, expected1=e1, expected_z=z, expected_a=a, expected_time=time,
        expected_steps=len(raw["state"]["journal"]), expected_journal=raw["state"]["journal"],
        expected_ledgers=dict(elastic_energy_j=energy, mechanical_work_j=work, viscous_dissipation_j=loss),
        birth=not raw["state"]["journal"])
    require_tied_kernel(modes, a, "retained private displacement")
    require_tied_kernel(modes, z, "retained private elastic memory")
    require_tied_kernel(modes, velocity[context["parent_ndof"]:], "retained private velocity")
    represented0, represented1 = memory.state_coefficients(raw)
    return dict(e0=represented0, e1=represented1, z=array(raw["state"]["coefficients_m"]),
                a=array(raw["state"]["displacement_m"]), time=time, velocity=velocity, maximum=maximum,
                report=report)


def endpoint_addition_witness(W, old_full, new_full, full_increment, old_coordinates, new_coordinates, coordinate_increment):
    """Literal coefficient-addition and restriction precision, never work credit."""
    ld = np.longdouble
    W = np.asarray(W, ld)
    old_full, new_full, full_increment, old_coordinates, new_coordinates, coordinate_increment = (
        np.asarray(v, ld) for v in (old_full, new_full, full_increment, old_coordinates, new_coordinates, coordinate_increment))
    full_difference = new_full-old_full-full_increment
    full_bound = reduction_roundoff(abs(old_full)+abs(full_increment), 1, epsilon=EPS)
    bound = (abs(W).T@full_bound
        +reduction_roundoff(abs(W).T@(abs(old_full)+abs(new_full)+abs(full_increment)), 18)
        +reduction_roundoff(abs(W.T@old_full)+abs(W.T@new_full), 1, epsilon=EPS))
    endpoint_increment = new_coordinates-old_coordinates
    difference = endpoint_increment-coordinate_increment
    bound += reduction_roundoff(abs(old_coordinates)+abs(new_coordinates)+abs(coordinate_increment), 3)
    return dict(endpoint_increment=endpoint_increment, difference=difference, bound=bound,
        full_difference=full_difference, full_bound=full_bound,
        passed=bool(np.all(abs(difference) <= bound) and np.all(abs(full_difference) <= full_bound)))


def audit_release_trajectory(record, checked, parent, prior_case):
    require(type(record) is dict, "full private trajectory record required")
    raw = record["response"]
    seal(raw, "release_work_response_sha256")
    scalar_zero_credit(raw)
    for key in ("physical_creation_admitted", "physical_release_admitted", "funding_admitted",
                "weak_geometry_accuracy_certified", "legacy_witness_differences_available_for_funding"):
        require(raw.get(key) is False, "private trajectory cannot admit physical accuracy/release/funding")
    require(raw["arithmetic_difference_is_available_work"] is False
            and raw["complete_jump_kernel_remains_tied"] is True
            and raw["new_accepted_Genesis_time_myr"] == 0 and raw["new_material_birth_count"] == 0,
            "prospective receipt cannot alter accepted history or spend reduction error")
    exact(raw["full_force_fields_scope"], "legacy_float64_expanded_actions_arithmetic_witness_only",
          "legacy force witness scope")
    context, modes = checked["numerical_context"], checked["modes"]
    source, quad = checked["operative_source"], checked["restricted_quadrature"]
    exact(raw["context_binding_sha256"], checked["context_binding_sha256"], "complete release context link")
    descriptor = raw["restricted_source_quadrature"]
    seal(descriptor, "source_quadrature_sha256")
    exact(raw["source_quadrature_binding_sha256"], descriptor["source_quadrature_sha256"],
          "restricted source quadrature link")
    exact(descriptor["source_binding_sha256"], prior_case["embedded_partition"]["embedded_partition_sha256"],
          "restricted literal original point source")
    require(descriptor["weak_geometry_accuracy_certified"] is False
            and descriptor["source_selected_clipped_bulk_accuracy_certified"] is False
            and descriptor["physical_release_admitted"] is False and descriptor["funding_admitted"] is False,
            "restricted arithmetic cannot certify physical weak geometry")
    exact(descriptor["represented_source_field_dtype"], "archived_float64", "literal point representation")
    exact(descriptor["contraction_arithmetic"], "longdouble_before_Gram", "restriction arithmetic order")
    exact(descriptor["scope"], QUADRATURE_SCOPE, "restricted point operator scope")
    exact(raw["operative_bulk_field_scope"], QUADRATURE_SCOPE, "operative field scope")
    exact(descriptor["array_sha256"], quad["array_sha256"], "independently derived literal restricted point array pins")
    exact(descriptor["density_kg_m3"], float(quad["density_kg_m3"]), "literal restricted source density")
    n, r, W = context["parent_ndof"], modes["rank"], np.asarray(modes["observable_force_basis"], np.longdouble)
    previous = replay_memory_archive(record["before_memory"], checked, parent, prior_case)
    d, v0 = array(raw["increment_full_m"], (n+18,)), array(raw["before_velocity_m_s"], (n+18,))
    exact(v0.tolist(), previous["velocity"].tolist(), "genuine-birth private journal velocity")
    require_tied_kernel(modes, d[n:], "prospective increment")
    require_tied_kernel(modes, v0[n:], "prospective initial velocity")
    after, before = raw["after_time_myr"], previous["time"]
    require(type(after) in (int, float) and after > before, "private time must follow genuine birth")
    dt = (after-before)*memory.SECONDS_PER_MYR
    exact(raw["dt_s"], dt, "literal private duration")
    exact(raw["before_memory_sha256"], digest(record["before_memory"]["state"]), "source-replayed before memory")
    close(record["old_maximum_separation_m"], previous["maximum"], "replayed irreversible cohesive history",
          rtol=2e-8, atol=2e-13)
    ratio = memory.source_maxwell_ratios(parent, before, after)
    inc0 = np.einsum("cqai,ci->cqa", context["B0"], d[:n][context["dofs"]])
    inc1 = np.einsum("cqai,ci->cqa", context["B1"], d[:n][context["dofs"]])
    old_bulk = memory.maxwell_step(context, previous["e0"], previous["e1"], previous["z"],
        inc0, inc1, d[n:], ratio)
    bulk = raw["bulk_response_minimal"]
    for key in ("mean0", "mean1", "mean_coefficients_m", "coefficients_m"):
        close(bulk[key], old_bulk[key], "minimal source Maxwell "+key, rtol=2e-9, atol=2e-13)
    exact(bulk["displacement_m"], (previous["a"]+d[n:]).tolist(), "continued compact displacement")
    after_memory = record["after_memory"]
    after0, after1 = memory.state_coefficients(after_memory)
    close(after0, old_bulk["strain0"], "source constant endpoint field", rtol=2e-9, atol=2e-13)
    close(after1, old_bulk["strain1"], "source radial endpoint field", rtol=2e-9, atol=2e-13)
    represented_bulk = {**bulk, "strain0": after0, "strain1": after1}
    legacy_math = archived_legacy_small_response(context, prior_case, previous,
        represented_bulk, inc0, inc1, d, ratio)
    for key, wanted in (("legacy_rounded_bulk_energy_before_decomposition_j", np.r_[old_bulk["before_energy_decomposition_j"][0], legacy_math["before"]]),
                        ("legacy_rounded_bulk_energy_after_decomposition_j", np.r_[old_bulk["energy_decomposition_j"][0], legacy_math["after"]]),
                        ("legacy_rounded_bulk_work_decomposition_j", np.r_[old_bulk["work_decomposition_j"][0], legacy_math["work"]]),
                        ("legacy_rounded_bulk_dissipation_decomposition_j", np.r_[old_bulk["dissipation_decomposition_j"][0], legacy_math["loss"]])):
        for j, family in enumerate(("original", "signed cross", "enriched self")):
            close(array(raw[key], (3,))[j], wanted[j], "preserved arithmetic witness "+key+" "+family,
                  rtol=2e-9 if j == 0 else 3e-6, atol=1e-10)
    parent_force = array(raw["bulk_parent_mean_force_n"], (n,))
    independent_parent = memory.full_generalized_force(context, bulk["mean0"], bulk["mean1"], np.zeros(18))[:n]
    mode_close(parent_force, independent_parent, "unchanged original analytic mean force", rtol=5e-7, atol=1e-5)
    for key, wanted in (("bulk_cross_parent_force_n", legacy_math["cross_parent"]),
                        ("bulk_cross_enrichment_force_n", legacy_math["cross_added"]),
                        ("bulk_self_enrichment_force_n", legacy_math["self_added"])):
        close(raw[key], wanted, "literal rounded bulk force witness "+key, rtol=0, atol=0)
    legacy_parent = parent_force.copy()
    np.add.at(legacy_parent, context["selected_parent_dofs"], legacy_math["cross_parent"])
    exact(raw["bulk_mean_force_n"], np.r_[legacy_parent, legacy_math["cross_added"]+legacy_math["self_added"]].tolist(),
          "preserved complete rounded bulk force witness")
    legacy_force_work = np.asarray((parent_force@d[:n],
        legacy_math["cross_parent"]@d[context["selected_parent_dofs"]]+legacy_math["cross_added"]@d[n:],
        legacy_math["self_added"]@d[n:]))
    exact(raw["legacy_rounded_bulk_force_work_decomposition_j"], legacy_force_work.tolist(),
          "complete literal rounded force/work witness")
    coordinates = lambda value: np.asarray(W.T@np.asarray(value, np.longdouble), float)
    old_a, new_a = coordinates(previous["a"]), coordinates(previous["a"]+d[n:])
    for key, wanted in (("operative_enrichment_old_displacement_m", old_a),
                        ("operative_enrichment_new_displacement_m", new_a)):
        exact(raw[key], wanted.tolist(), "literal retained-field "+key)
    expected_gamma = independent_cohesive_path(checked["law"], source["initial_traction_world"],
        source["weights"], source["trace"], source["basis"], old_a, new_a, previous["maximum"])
    cohesive_report = audit_cohesive_path(raw["cohesive_response"], expected_gamma)
    legacy_gamma = raw["legacy_rounded_cohesive_response"]
    legacy_source = checked["legacy_source"]
    expected_legacy_gamma = independent_cohesive_path(checked["law"], legacy_source["initial_traction_world"],
        legacy_source["weights"], legacy_source["trace"], legacy_source["basis"], previous["a"], previous["a"]+d[n:],
        journal_interface_maximum(checked, record["before_memory"], operative=False))
    audit_cohesive_path(legacy_gamma, expected_legacy_gamma)
    # The independently replayed path uses the operative chart.  The literal
    # expanded response remains a separate vector witness and supplies no work.
    close(legacy_gamma["old_enrichment_m"], previous["a"], "legacy interface old field", rtol=0, atol=0)
    close(legacy_gamma["new_enrichment_m"], previous["a"]+d[n:], "legacy interface new field", rtol=0, atol=0)
    M = seed.dense_csr(prior_case["embedded_partition"]["mass_matrix"], "literal expanded source mass")
    D = seed.dense_csr(prior_case["embedded_partition"]["basal_drag_matrix"], "literal expanded source drag")
    ext = array(prior_case["embedded_partition"]["external_force_n"])
    legacy_budget = discrete_path_budget(M, D, ext, d, v0, dt, old_bulk["mean_generalized_force_n"],
        array(legacy_gamma["mean_generalized_force_n"]), old_bulk["work_decomposition_j"],
        legacy_gamma["force_work_j"])
    exact(raw["after_velocity_m_s"], legacy_budget["after_velocity_m_s"].tolist(), "midpoint kinematic identity")
    require_tied_kernel(modes, raw["after_velocity_m_s"][n:], "prospective final velocity")
    for key, matrix, vector, factor in (("inertia_force_n", M, d-dt*v0, 2/dt**2),
                                      ("basal_drag_force_n", D, d, 1/dt)):
        matrix, vector = np.asarray(matrix, np.longdouble), np.asarray(vector, np.longdouble)
        action = (matrix@vector)*np.longdouble(factor)
        operand_scale = (abs(matrix)@abs(vector))*abs(np.longdouble(factor))
        require(np.all(abs(np.asarray(array(raw[key]), np.longdouble)-action) <= 512*EPS*operand_scale),
                "literal expanded arithmetic witness "+key+" differs")
    exact(raw["external_force_n"], prior_case["embedded_partition"]["external_force_n"], "unaltered complete source load")
    exact(raw["cohesive_mean_force_n"], np.r_[np.zeros(n), array(legacy_gamma["mean_generalized_force_n"])].tolist(),
          "expanded interface witness")
    legacy_residual = (array(raw["inertia_force_n"])+array(raw["basal_drag_force_n"])
        +array(raw["bulk_mean_force_n"])+array(raw["cohesive_mean_force_n"])-array(raw["external_force_n"]))
    exact(raw["residual_n"], legacy_residual.tolist(), "literal expanded residual witness")
    close(raw["tied_kernel_residual_n"], modes["declared_tied_null_modes"].T@legacy_residual[n:],
          "retained complete-kernel reaction witness", rtol=0, atol=1e-7)
    expected = restricted_current_path(context, quad, previous, represented_bulk, inc0, inc1, d, v0, dt,
        ratio, expected_gamma["mean_traction_local_pa"], old_bulk["work_decomposition_j"],
        old_bulk["dissipation_decomposition_j"], old_bulk["before_energy_decomposition_j"],
        old_bulk["energy_decomposition_j"])
    operand_bounds = independent_current_operand_bounds(context, quad, previous, represented_bulk, d, v0, dt,
        raw["cohesive_response"], parent_force)
    local = prior_case["embedded_partition"]["local"]
    expansion, expansion_passed = source_expansion_witnesses(
        dict(displacement=local["enrichment_trace"], strain=local["enrichment_strain_operator"],
            interface_gap=checked["legacy_source"]["trace"]),
        dict(displacement=quad["E_w"], strain=quad["G_w"], interface_gap=source["trace"]), W,
        dict(old_displacement=previous["a"], new_displacement=bulk["displacement_m"], displacement_increment=d[n:],
            before_velocity=v0[n:], after_velocity=raw["after_velocity_m_s"][n:],
            old_elastic_memory=previous["z"], new_elastic_memory=bulk["coefficients_m"], mean_elastic_memory=bulk["mean_coefficients_m"]))
    require(raw["source_point_expansion_passed"] is expansion_passed and expansion_passed,
            "source point field cannot change under the numerical restriction")
    exact(raw["source_point_expansion_bound_scope"],
        "256_float64_epsilon_times_actual_source_lift_projection_operands_no_floor", "source field representation precision")
    require(set(raw["source_point_expansion_witnesses"]) == set(expansion), "complete represented field expansion witnesses required")
    for key, expected_witness in expansion.items():
        actual_witness = raw["source_point_expansion_witnesses"][key]
        require(actual_witness["passed"] is expected_witness["passed"], "source field expansion verdict differs")
        for field in expected_witness:
            if field != "passed": close(actual_witness[field], expected_witness[field], "source field expansion "+key+" "+field, rtol=3e-12, atol=0)
    exact(raw["operative_force_roundoff_scope"],
        "actual_source_point_and_original_parent_operands_gamma_n_reduction_bounds_no_floor",
        "actual operand momentum precision scope")
    for key, wanted in operand_bounds.items():
        actual = array(raw[key], np.asarray(wanted).shape)
        require(np.all(actual >= 0), "current operand precision cannot be negative")
        for j in np.ndindex(actual.shape):
            close(actual[j], wanted[j], "actual source operand allowance "+key+" "+str(j), rtol=3e-6, atol=0)
    # Original force approximation and the tiny restricted modes are compared
    # separately; the old force magnitude cannot hide missing weak actions.
    for key in ("operative_inertia_force_n", "operative_basal_drag_force_n", "operative_bulk_mean_force_n",
                "operative_cohesive_mean_force_n", "operative_external_force_n"):
        actual, wanted = array(raw[key], (n+r,)), np.asarray(expected[key], float)
        mode_close(actual[:n], wanted[:n], "restricted original "+key, rtol=5e-7, atol=1e-5)
        for j in range(r):
            close(actual[n+j], wanted[n+j], "restricted weak source "+key+" "+str(j), rtol=3e-6, atol=1e-25)
    exact(raw["operative_external_force_n"], np.r_[quad["original_external"], np.zeros(r)].tolist(),
          "source external force cannot be manufactured")
    for key in ("operative_increment_m", "operative_before_velocity_m_s", "operative_after_velocity_m_s"):
        exact(raw[key], np.asarray(expected[key], float).tolist(), "literal full-field restriction "+key)
    for key in ("bulk_energy_before_decomposition_j", "bulk_energy_after_decomposition_j",
                "bulk_energy_increment_decomposition_j", "bulk_work_decomposition_j",
                "bulk_dissipation_decomposition_j", "bulk_force_work_decomposition_j",
                "kinetic_change_decomposition_j"):
        for j, family in enumerate(("original", "signed cross", "enriched self")):
            close(array(raw[key], (3,))[j], expected[key][j], "restricted current "+key+" "+family,
                  rtol=2e-9 if j == 0 else 3e-6, atol=1e-25 if j else 1e-8)
    for key, wanted in (("operative_bulk_cross_parent_force_n", expected["bulk_cross_parent_force_n"]),
                        ("operative_bulk_cross_enrichment_force_n", expected["operative_bulk_cross_enrichment_force_n"]),
                        ("operative_bulk_self_enrichment_force_n", expected["operative_bulk_self_enrichment_force_n"])):
        actual = array(raw[key])
        for j in range(len(actual)):
            close(actual[j], wanted[j], "restricted separate force "+key+" "+str(j), rtol=3e-6, atol=1e-25)
    operative_d = array(raw["operative_increment_m"])
    force_keys = ("operative_inertia_force_n", "operative_basal_drag_force_n", "operative_bulk_mean_force_n",
                  "operative_cohesive_mean_force_n", "operative_external_force_n")
    parts = [np.asarray(array(raw[key]), np.longdouble) for key in force_keys]
    parts[-1] *= -1
    projected = sum(parts)
    scale = sum(abs(a) for a in parts)
    tolerance = raw["momentum_relative_tolerance"]
    require(type(tolerance) in (int, float) and 0 < tolerance <= 3e-8,
            "declared momentum tolerance cannot weaken the control")
    per_force_bounds = [np.asarray(array(raw[key]), np.longdouble) for key in (
        "operative_inertia_component_roundoff_bound_n", "operative_basal_drag_component_roundoff_bound_n",
        "operative_bulk_component_roundoff_bound_n", "operative_cohesive_component_roundoff_bound_n",
        "operative_external_component_roundoff_bound_n")]
    projection_roundoff = sum(per_force_bounds)+256*EPS*scale
    bound = np.longdouble(tolerance)*scale+projection_roundoff
    for key, wanted in (("operative_projected_residual_n", projected), ("projected_residual_scale_n", scale),
                        ("operative_force_component_roundoff_bound_n", projection_roundoff),
                        ("projected_momentum_component_bound_n", bound)):
        actual = array(raw[key], (n+r,))
        for j in range(n+r):
            require(abs(np.longdouble(actual[j])-wanted[j]) <= 16*EPS*(scale[j]+abs(wanted[j])),
                    "literal restricted component "+key+" differs")
    close(raw["parent_residual_n"], raw["operative_projected_residual_n"][:n], "operative parent residual", rtol=0, atol=0)
    close(raw["observable_force_residual_n"], raw["operative_projected_residual_n"][n:], "operative weak residual", rtol=0, atol=0)
    close(raw["released_residual_n"], np.asarray(modes["declared_release_basis"], np.longdouble).T@W@
        np.asarray(raw["operative_projected_residual_n"][n:], np.longdouble), "release-space residual", rtol=3e-13, atol=1e-7)
    momentum_passed = bool(np.all(abs(array(raw["operative_projected_residual_n"])) <= array(raw["projected_momentum_component_bound_n"])))
    require(raw["projected_momentum_passed"] is momentum_passed, "projected momentum verdict differs")
    relative = float(np.max(np.divide(abs(np.asarray(raw["operative_projected_residual_n"], np.longdouble)),
        np.asarray(raw["projected_residual_scale_n"], np.longdouble), out=np.zeros(n+r, np.longdouble),
        where=array(raw["projected_residual_scale_n"]) > 0), initial=0.))
    close(raw["projected_momentum_relative_error"], relative, "projected momentum relative error", rtol=1e-12, atol=0)
    scalar_expected = {key: expected[key] for key in ("external_work_j", "kinetic_change_stable_j",
        "basal_dissipation_j", "cohesive_work_j")}
    scalar_expected.update(bulk_work_j=np.sum(expected["bulk_work_decomposition_j"], dtype=np.longdouble),
        cohesive_stored_energy_increment_j=expected_gamma["stored_energy_j"]-expected_gamma["old_stored_energy_j"],
        fracture_dissipation_increment_j=expected_gamma["fracture_dissipation_increment_j"])
    for key, wanted in scalar_expected.items():
        close(raw[key], wanted, "prospective work budget "+key, rtol=3e-6, atol=1e-8)
    # Source forcing is pinned exactly and independently contracted.  Historical
    # D, stored-energy subtraction ULPs and witness differences are absent.
    external_work = float(np.asarray(raw["operative_external_force_n"], np.longdouble)@
                          np.asarray(operative_d, np.longdouble))
    exact(raw["external_work_j"], external_work, "actual source external work cannot be manufactured")
    required = math.fsum((raw["kinetic_change_stable_j"], raw["basal_dissipation_j"],
        *raw["bulk_work_decomposition_j"], raw["cohesive_work_j"]))
    exact(raw["budget_required_work_j"], required, "actual-path required work")
    exact(raw["work_balance_residual_j"], raw["external_work_j"]-required, "actual source work deficit")
    dual = float(np.asarray(raw["operative_projected_residual_n"], np.longdouble)@
                 np.asarray(operative_d, np.longdouble))
    dual_roundoff = 32*EPS*sum(float(np.sum(abs(a*np.asarray(operative_d, np.longdouble)))) for a in parts)
    require(abs(raw["force_residual_work_j"]-dual) <= dual_roundoff,
            "restricted residual dual work differs")
    work_tolerance = raw["work_relative_tolerance"]
    require(type(work_tolerance) in (int, float) and 0 < work_tolerance <= 3e-9,
            "declared work tolerance cannot weaken the control")
    identity_scale = math.fsum((abs(raw["external_work_j"]), abs(raw["kinetic_change_stable_j"]),
        abs(raw["basal_dissipation_j"]), *(abs(x) for x in raw["bulk_work_decomposition_j"]), abs(raw["cohesive_work_j"])))
    identity_bound = work_tolerance*identity_scale+sum(128*EPS*float(np.sum(abs(a*operative_d))) for a in parts)
    identity_bound += float(sum(per_force_bounds)@np.asarray(abs(operative_d), np.longdouble))
    identity_bound += float(np.sum(np.asarray(raw["bulk_work_roundoff_bound_decomposition_j"], np.longdouble)
        +np.asarray(raw["bulk_force_dot_roundoff_bound_decomposition_j"], np.longdouble)))
    identity_bound += raw["cohesive_response"]["work_integral_roundoff_bound_j"]
    close(raw["work_identity_bound_j"], identity_bound, "current-path work bound", rtol=3e-15, atol=0)
    balance_passed = bool(abs(raw["work_balance_residual_j"]) <= identity_bound)
    require(raw["discrete_work_balance_passed"] is balance_passed, "actual-path balance verdict differs")
    small = array(raw["bulk_energy_increment_decomposition_j"])[1:]-(
        array(raw["bulk_work_decomposition_j"])[1:]-array(raw["bulk_dissipation_decomposition_j"])[1:])
    small_scale = (abs(array(raw["bulk_energy_after_decomposition_j"])[1:])+abs(array(raw["bulk_energy_before_decomposition_j"])[1:])
        +abs(array(raw["bulk_work_decomposition_j"])[1:])+abs(array(raw["bulk_dissipation_decomposition_j"])[1:]))
    for j in range(2):
        require(abs(array(raw["bulk_small_component_balance_residual_j"])[j]-small[j]) <= 16*EPS*small_scale[j],
                "literal tiny field balance differs")
    small_passed = bool(np.all(abs(array(raw["bulk_small_component_balance_residual_j"])) <= (work_tolerance+256*EPS)*small_scale))
    fw, bw = array(raw["bulk_force_work_decomposition_j"]), array(raw["bulk_work_decomposition_j"])
    duality_bound = work_tolerance*(abs(fw)+abs(bw))+array(raw["bulk_work_roundoff_bound_decomposition_j"])+array(raw["bulk_force_dot_roundoff_bound_decomposition_j"])
    close(raw["bulk_force_work_duality_bound_decomposition_j"], duality_bound,
          "only current point operands bound bulk duality", rtol=3e-15, atol=0)
    duality_passed = bool(np.all(abs(fw-bw) <= duality_bound))
    require(raw["bulk_small_component_balance_passed"] is small_passed
            and raw["bulk_force_work_duality_passed"] is duality_passed, "separate small-field work verdict differs")
    addition = endpoint_addition_witness(W, previous["a"], bulk["displacement_m"], d[n:], old_a, new_a,
        W.T@np.asarray(d[n:], np.longdouble))
    exact(raw["operative_endpoint_addition_scope"],
        "actual_source_coefficient_addition_and_coordinate_operands_gamma_n_no_floor_no_credit",
        "literal endpoint addition arithmetic scope")
    for key, wanted in (("operative_endpoint_enrichment_increment_m", addition["endpoint_increment"]),
                        ("operative_endpoint_addition_difference_m", addition["difference"]),
                        ("operative_endpoint_addition_roundoff_bound_m", addition["bound"]),
                        ("full_enrichment_endpoint_addition_difference_m", addition["full_difference"]),
                        ("full_enrichment_endpoint_addition_roundoff_bound_m", addition["full_bound"])):
        exact(raw[key], np.asarray(wanted, float).tolist(), "own-scale endpoint addition "+key)
    require(raw["operative_endpoint_addition_passed"] is addition["passed"], "literal endpoint addition verdict differs")
    gamma_residual = raw["cohesive_work_j"]-raw["cohesive_response"]["numerically_integrated_work_j"]
    gamma_bound = (work_tolerance*(abs(raw["cohesive_work_j"])
        +abs(raw["cohesive_response"]["numerically_integrated_work_j"]))
        +128*EPS*float(np.sum(abs(parts[3]*np.asarray(operative_d, np.longdouble))))
        +raw["cohesive_response"]["work_integral_roundoff_bound_j"]
        +float(per_force_bounds[3]@np.asarray(abs(operative_d), np.longdouble)))
    exact(raw["cohesive_force_work_duality_residual_j"], gamma_residual, "actual literal cohesive path dual work")
    close(raw["cohesive_force_work_duality_bound_j"], gamma_bound,
          "current cohesive path arithmetic cannot use cumulative fracture loss", rtol=3e-15, atol=0)
    gamma_duality_passed = bool(abs(gamma_residual) <= gamma_bound and addition["passed"])
    require(raw["cohesive_force_work_duality_passed"] is gamma_duality_passed,
            "literal cohesive mean-force/source-path verdict differs")
    identity_passed = bool(abs(raw["work_balance_residual_j"]+raw["force_residual_work_j"]) <= identity_bound
                           and small_passed and duality_passed and gamma_duality_passed)
    require(raw["work_identity_passed"] is identity_passed, "complete actual-path work identity verdict differs")
    balanced = bool(momentum_passed and balance_passed and identity_passed)
    require(raw["balanced_trajectory"] is balanced, "scalar work alone cannot certify a balanced trajectory")
    endpoint = raw["cohesive_response"]["endpoint_work_agrees_at_integral_precision"]
    require(raw["endpoint_ledger_resolved"] is endpoint and raw["cohesive_endpoint_work_agrees_at_integral_precision"] is endpoint
            and raw["funding_numerically_resolved"] is bool(balanced and endpoint), "endpoint precision cannot authorize funding")
    exact(raw["physical_refusals"], checked["physical_refusals"]+([] if endpoint else [
        "cohesive_endpoint_cumulative_energy_dissipation_increment_is_arithmetic_unresolved"])+([] if gamma_duality_passed else [
        "cohesive_mean_force_work_disagrees_with_actual_source_path_increment"]), "physical release refusals")
    require(raw["kinematic_identity_passed"] is True and raw["cohesive_history_replayed_from_source_birth"] is True,
            "source kinematic/history gate omitted")
    expected_item = dict(after_time_myr=after, parent_increment_m=d[:n].tolist(),
        enrichment_increment_m=d[n:].tolist(), ratio=after_memory["state"]["journal"][-1]["ratio"])
    exact(after_memory["state"]["journal"], record["before_memory"]["state"]["journal"]+[expected_item],
          "private journal contains exactly the declared increment")
    exact(raw["after_memory_sha256"], digest(after_memory["state"]), "response after-memory link")
    resumed = replay_memory_archive(after_memory, checked, parent, prior_case)
    require(record["source_memory_restart_exact"] is True and record["source_work_restart_exact"] is True,
            "private restart witnesses missing")
    return dict(label=record["label"], private_after_time_myr=after, complete_kernel_tied=True,
        before_memory_replayed=previous["report"], after_memory_replayed=resumed["report"],
        direct_cohesive_path=cohesive_report, literal_source_point_current_budget_checked=True,
        legacy_expanded_actions_are_arithmetic_witnesses=True, fresh_physical_weak_source_accuracy_certified=False,
        historical_loss_and_roundoff_funding=False, balanced_trajectory=balanced,
        external_work_j=external_work, independent_kinetic_change_decomposition_j=np.asarray(expected["kinetic_change_decomposition_j"], float).tolist(),
        physical_release_admitted=False)


def audit_case(context_raw, trajectory_raw, parent, prior_case):
    memory.authenticate_parent(parent)
    checked = audit_release_context(context_raw, parent, prior_case)
    trajectories = trajectory_raw if type(trajectory_raw) is list else [trajectory_raw]
    require(len(trajectories) > 0, "nonempty prospective trajectories required")
    reports = [audit_release_trajectory(raw, checked, parent, prior_case) for raw in trajectories]
    return dict(context=checked["context_report"], trajectories=reports)


def audit_fixture(fixture, parent, previous_memory_fixture, control_directory=None):
    seal(fixture, "fixture_sha256")
    memory.authenticate_parent(parent)
    seal(previous_memory_fixture, "fixture_sha256")
    require(previous_memory_fixture["fixture_sha256"] == PREVIOUS_MEMORY_FIXTURE_SHA256,
            "preserved complete-memory numerical source differs")
    exact(fixture["parent_raw_sha256"], PARENT_RAW_SHA256, "parent raw source pin")
    exact(fixture["parent_checkpoint_sha256"], PARENT_CHECKPOINT_SHA256, "parent checkpoint pin")
    exact(fixture["previous_memory_fixture_raw_sha256"], PREVIOUS_MEMORY_FIXTURE_RAW_SHA256, "previous memory raw pin")
    exact(fixture["previous_memory_fixture_sha256"], PREVIOUS_MEMORY_FIXTURE_SHA256, "previous memory fixture pin")
    scalar_zero_credit(fixture)
    for key in ("physical_creation_admitted", "physical_release_admitted", "funding_admitted"):
        require(fixture.get(key) is False, "control physical scope differs")
    for key in ("history_reinitialized", "accepted_planet_memory_changed", "automated_goal_enabled", "point_one_complete"):
        require(fixture.get(key) is False, "private control cannot claim accepted history or completion")
    for key in ("new_Genesis_time_steps", "new_material_cohorts"):
        require(fixture.get(key) == 0, "control cannot advance accepted Genesis")
    require(type(fixture.get("cases")) is list and len(fixture["cases"]) > 0, "nonempty source cases required")
    prior = {case["name"]: case for case in previous_memory_fixture["cases"]}
    reports = []
    for case in fixture["cases"]:
        require(case["previous_case_name"] in prior, "source case identity differs")
        inherited = prior[case["previous_case_name"]]
        if "seed_definition" in case:
            exact(case["seed_definition"], inherited["seed_definition"], "inherited literal seed definition")
        reports.append(dict(name=case["name"], **audit_case(case["context"], case["trajectories"], parent, inherited)))
    if control_directory is not None:
        directory = Path(control_directory)
        require((directory/"parent_before.json").read_bytes() == (directory/"parent_after.json").read_bytes(),
                "private release probe changed accepted parent bytes")
        for filename in ("parent_before.json", "parent_after.json"):
            require(hashlib.sha256((directory/filename).read_bytes()).hexdigest() == PARENT_RAW_SHA256,
                    "unchanged genuine parent raw pin differs")
        require(canonical(json.loads((directory/"control_fixture.json").read_bytes())) == canonical(fixture),
                "control directory fixture differs")
    report = dict(format=FORMAT, status="PASS", producer_imports=False,
        fixture_sha256=fixture["fixture_sha256"],
        previous_memory_fixture_raw_sha256=PREVIOUS_MEMORY_FIXTURE_RAW_SHA256,
        previous_memory_fixture_sha256=PREVIOUS_MEMORY_FIXTURE_SHA256,
        parent_raw_sha256=PARENT_RAW_SHA256, parent_checkpoint_sha256=PARENT_CHECKPOINT_SHA256,
        formation_checkpoint_sha256=FORMATION_SHA256, actual_cohorts=800, actual_active_faults=0,
        actual_time_myr=parent["runtime_state"]["geometry"]["time_myr"],
        physical_release_admitted=False, funding_admitted=False,
        accepted_Genesis_changed=False, cases=reports)
    return {**report, "report_sha256": digest(report)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--previous-memory-fixture", type=Path, required=True)
    parser.add_argument("--control-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    parent_bytes = args.parent_checkpoint.read_bytes()
    previous_bytes = args.previous_memory_fixture.read_bytes()
    require(hashlib.sha256(parent_bytes).hexdigest() == PARENT_RAW_SHA256, "genuine parent raw pin differs")
    require(hashlib.sha256(previous_bytes).hexdigest() == PREVIOUS_MEMORY_FIXTURE_RAW_SHA256, "previous memory raw pin differs")
    fixture = json.loads(args.fixture.read_bytes())
    report = audit_fixture(fixture, json.loads(parent_bytes), json.loads(previous_bytes), args.control_directory)
    require(not args.output.exists(), "audit output must be fresh")
    args.output.write_bytes(canonical(report)+b"\n")
    print(json.dumps(dict(status="PASS", report_sha256=report["report_sha256"],
        report_raw_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
        cases=len(report["cases"]), producer_imports=False)), flush=True)


if __name__ == "__main__":
    main()
