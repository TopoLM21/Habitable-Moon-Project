"""Producer-free audit of source-born authoritative release-coordinate history.

The original 800-cohort parent archive remains unchanged.  Native coordinates
start at the authentic zero enriched birth and replay their own source Maxwell
journal; no mature full18 displacement or memory is projected to initialize
this history.  Literal full18 packets are independently replayed arithmetic
witnesses.  Point tensors are restricted before Gram formation in longdouble,
using exactly the earlier sealed source's represented float64 inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from analysis.genesis_volume_release_validation import independent_audit as literal
from analysis.genesis_volume_memory_validation import independent_audit_v2 as memory

# These are producer-free mathematical checks, not simulation imports.
require, canonical, digest, seal = literal.require, literal.canonical, literal.digest, literal.seal
array, close, exact, mode_close = literal.array, literal.close, literal.exact, literal.mode_close
scalar_zero_credit, require_tied_kernel = literal.scalar_zero_credit, literal.require_tied_kernel
insertion, seed, creation = literal.insertion, literal.seed, literal.creation
EPS, QUADRATURE_SCOPE = literal.EPS, literal.QUADRATURE_SCOPE
PARENT_RAW_SHA256 = literal.PARENT_RAW_SHA256
PARENT_CHECKPOINT_SHA256 = literal.PARENT_CHECKPOINT_SHA256
FORMATION_SHA256 = literal.FORMATION_SHA256
PREVIOUS_MEMORY_FIXTURE_RAW_SHA256 = literal.PREVIOUS_MEMORY_FIXTURE_RAW_SHA256
PREVIOUS_MEMORY_FIXTURE_SHA256 = literal.PREVIOUS_MEMORY_FIXTURE_SHA256
FORMAT = "independent-genesis-native-release-work-audit-0.1"
NATIVE_FORMAT = "genesis-source-born-native-release-history-0.1"
NATIVE_SCOPE = "source_born_authoritative_W_coordinates_fixed_material_private_journal"
FIXTURE_FORMAT = "genesis-volume-native-release-work-control-0.2"
FIXTURE_SCOPE = "source_born_native_coordinate_private_fixed_material_work_gate_no_physical_release"
audit_release_context = literal.audit_release_context
archived_legacy_small_response = literal.archived_legacy_small_response
independent_cohesive_path = literal.independent_cohesive_path
audit_cohesive_path = literal.audit_cohesive_path
discrete_path_budget = literal.discrete_path_budget
journal_interface_maximum = literal.journal_interface_maximum
reduction_roundoff = literal.reduction_roundoff

def replay_memory_archive(raw, checked, parent, prior_case):
    """Replay the unchanged parent/full18 source archive as a witness only."""
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
    # Every expanded step was checked above, and the entire archive was
    # independently replayed.  Cumulative float64 additions can leave a null
    # component after cancellation.  These retained full18 arrays are recorded
    # witnesses, not the authoritative native displacement/memory/velocity.
    represented0, represented1 = memory.state_coefficients(raw)
    return dict(e0=represented0, e1=represented1, z=array(raw["state"]["coefficients_m"]),
                a=array(raw["state"]["displacement_m"]), time=time, velocity=velocity, maximum=maximum,
                report=report)


def quadrature_binding(checked, prior_case):
    q = checked["restricted_quadrature"]
    body = dict(format="genesis-release-restricted-source-quadrature-0.1",
        scope=QUADRATURE_SCOPE,
        source_binding_sha256=prior_case["embedded_partition"]["embedded_partition_sha256"],
        density_kg_m3=float(q["density_kg_m3"]), array_sha256=q["array_sha256"],
        represented_source_field_dtype="archived_float64", contraction_arithmetic="longdouble_before_Gram",
        weak_geometry_accuracy_certified=False, source_selected_clipped_bulk_accuracy_certified=False,
        physical_release_admitted=False, funding_admitted=False)
    return digest(body)


def replay_native_history(raw, full_memory, checked, parent, prior_case):
    """Replay c/z/v from zero, independently of expanded full18 coefficients."""
    seal(raw, "native_release_state_sha256")
    context, source = checked["numerical_context"], checked["operative_source"]
    n, r = context["parent_ndof"], checked["modes"]["rank"]
    W = np.asarray(checked["modes"]["observable_force_basis"], np.longdouble)
    time = parent["runtime_state"]["geometry"]["time_myr"]
    c, z, v = np.zeros(r), np.zeros(r), np.zeros(r)
    maximum = checked["initial_maximum"].copy()
    journal = raw.get("journal")
    archived = full_memory["state"]["journal"]
    require(type(journal) is list and len(journal) == len(archived),
            "native and original full18 source journals differ")
    local_trace = np.einsum("pji,pjk->pik", source["basis"], source["trace"])
    local_t0 = np.einsum("pji,pj->pi", source["basis"], source["initial_traction_world"])
    expected_journal = []
    for item, witness in zip(journal, archived):
        dw = array(item["increment_w_m"], (r,))
        q = array(item["parent_increment_m"], (n,))
        after = item["after_time_myr"]
        require(type(after) in (float, int) and after > time, "native source clock must advance")
        dt = (after-time)*memory.SECONDS_PER_MYR
        ratio = memory.source_maxwell_ratios(parent, time, after)
        close(witness["ratio"], ratio, "native source Maxwell factor witness", rtol=2e-9, atol=2e-13)
        x = float(array(witness["ratio"])[context["selected_index"]])
        alpha, beta = float(np.exp(-x)), float(-np.expm1(-x)/x) if x else 1.
        expanded = np.asarray(W@np.asarray(dw, np.longdouble), float)
        expected = dict(after_time_myr=after, dt_s=dt, parent_increment_m=q.tolist(),
            increment_w_m=dw.tolist(), source_expanded_increment_full18_m=expanded.tolist(),
            selected_maxwell_ratio=x, selected_maxwell_alpha=alpha, selected_maxwell_beta=beta)
        exact(item, expected, "native step must use the original source Maxwell and exact expanded witness")
        exact(witness["after_time_myr"], after, "native/full18 source clock link")
        exact(witness["parent_increment_m"], q.tolist(), "native original parent increment link")
        exact(witness["enrichment_increment_m"], expanded.tolist(), "native rounded full18 witness lift")
        new_c = np.asarray(c+dw, float)
        path = insertion.prestress_segment(checked["law"], local_t0,
            np.einsum("pai,i->pa", local_trace, c), np.einsum("pai,i->pa", local_trace, new_c), maximum)
        maximum = path["maximum_separation_m"]
        c, z, v = new_c, np.asarray(alpha*z+beta*dw, float), np.asarray(2*dw/dt-v, float)
        time = after
        expected_journal.append(expected)
    source_initial = digest(prior_case["birth_memory"]["state"])
    if not journal:
        exact(digest(full_memory["state"]), source_initial,
              "native zero birth cannot bind or project a mature full18 state")
        require(not np.any(array(full_memory["state"]["displacement_m"]))
                and not np.any(array(full_memory["state"]["coefficients_m"])),
                "native birth requires authentic zero enrichment")
    body = dict(format=NATIVE_FORMAT, scope=NATIVE_SCOPE,
        context_binding_sha256=checked["context_binding_sha256"],
        source_memory_birth_binding_sha256=digest(prior_case["source_memory_birth"]),
        source_initial_full18_memory_sha256=source_initial,
        source_quadrature_binding_sha256=quadrature_binding(checked, prior_case),
        full18_memory_sha256=digest(full_memory["state"]), native_rank=r,
        accepted_source_birth_time_myr=parent["runtime_state"]["geometry"]["time_myr"],
        time_myr=float(time), displacement_w_m=c.tolist(), coefficients_w_m=z.tolist(),
        velocity_w_m_s=v.tolist(), journal=expected_journal,
        native_birth_from_authentic_zero_enrichment=True,
        mature_full18_projection_used_as_native_initialization=False,
        full18_fields_scope="expanded_float64_arithmetic_witness_only_not_authoritative_coordinates",
        cohort_parent_memory_scope="unchanged_original_source_Maxwell_parent_archive",
        accepted_planet_memory_changed=False, physical_release_admitted=False, funding_admitted=False,
        new_accepted_Genesis_time_myr=0., creation_work_debited_from_Genesis_j=0.,
        historical_dissipation_available_for_creation_j=0.)
    exact(raw, {**body, "native_release_state_sha256": digest(body)},
          "native coordinates must replay the authentic zero-birth journal")
    return dict(c=c, z=z, velocity=v, time=time, maximum=maximum,
        binding_sha256=digest(raw), report=dict(native_zero_birth_replayed=True,
            native_journal_steps=len(journal), original_parent_fields_unchanged=True,
            mature_full18_projection_used=False, native_state_sha256=raw["native_release_state_sha256"]))


def native_coordinate_math_view(quad):
    """Express the already restricted point tensors directly in native axes.

    The identity here changes only the helper's coordinate arity.  G_W/E_W,
    source material, weights and M/K blocks are untouched; there is no recovery
    from a rounded full18 state or recomputation of the physical source basis.
    """
    return {**quad, "basis_w": np.eye(quad["basis_w"].shape[1], dtype=np.longdouble)}


def source_expansion_witnesses(full_fields, restricted_fields, W, vectors, native_vectors):
    """Audit the honest expanded witness against independently replayed natives."""
    W = np.asarray(W, np.longdouble)
    reports, all_passed = {}, True
    for name, value in vectors.items():
        a = np.asarray(array(value, (18,)), np.longdouble)
        coordinates = np.asarray(native_vectors[name], np.longdouble)
        lift_magnitude = abs(W)@(abs(W).T@abs(a))
        for field_name, full_raw in full_fields.items():
            full = np.asarray(full_raw, np.longdouble)
            restricted = np.asarray(restricted_fields[field_name], np.longdouble)
            difference = abs(np.einsum("pai,i->pa", full, a)-np.einsum("pai,i->pa", restricted, coordinates))
            magnitude = (np.einsum("pai,i->pa", abs(full), abs(a)+lift_magnitude)
                +np.einsum("pai,i->pa", abs(restricted), abs(coordinates)))
            bound = 256*EPS*magnitude
            passed = bool(np.all(difference <= bound))
            all_passed = all_passed and passed
            relative = np.divide(difference, bound, out=np.zeros_like(difference), where=bound != 0)
            reports[name+"_"+field_name] = dict(maximum_absolute_difference=float(np.max(difference, initial=0)),
                maximum_operand_arithmetic_bound=float(np.max(bound, initial=0)),
                maximum_difference_to_arithmetic_bound_ratio=float(np.max(relative, initial=0)), passed=passed)
    return reports, all_passed


def legacy_kernel_witnesses(null_modes, vectors):
    """Report the original literal18 criterion without admitting native motion."""
    N = array(null_modes)
    stable_norm = lambda value: (0. if not np.any(value) else
        float(np.max(abs(value))*np.linalg.norm(value/np.max(abs(value)))))
    reports, all_passed = {}, True
    for name, value in vectors.items():
        a = array(value, (18,))
        scale = float(np.max(abs(a), initial=0))
        scaled = a if scale == 0. else a/scale
        norm, leakage = stable_norm(scaled), stable_norm(N.T@scaled)
        passed = bool(leakage <= 256*EPS*norm)
        reports[name] = dict(norm_magnitude=scale*norm,
            null_component_norm=scale*leakage,
            relative_null_component=0. if norm == 0 else leakage/norm,
            literal18_kernel_tolerance_passed=passed)
        all_passed = all_passed and passed
    return reports, all_passed


def endpoint_addition_witness(old_full, new_full, full_increment, old_native, new_native, native_increment):
    ld = np.longdouble
    old_full, new_full, full_increment, old_native, new_native, native_increment = (
        np.asarray(v, ld) for v in (old_full, new_full, full_increment, old_native, new_native, native_increment))
    full_difference = new_full-old_full-full_increment
    full_bound = reduction_roundoff(abs(old_full)+abs(full_increment), 1, epsilon=EPS)
    endpoint_increment = new_native-old_native
    difference = endpoint_increment-native_increment
    bound = (reduction_roundoff(abs(old_native)+abs(native_increment), 1, epsilon=EPS)
        +reduction_roundoff(abs(old_native)+abs(new_native)+abs(native_increment), 3))
    return dict(endpoint_increment=endpoint_increment, difference=difference, bound=bound,
        full_difference=full_difference, full_bound=full_bound,
        passed=bool(np.all(abs(difference) <= bound) and np.all(abs(full_difference) <= full_bound)))



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
    integrated_work = np.zeros(npoint)
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
            integrated_work[j] += (hi-lo)/2*float(gauss@(traction@delta))
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
    direct_work = float(weights@integrated_work)
    close(float(force@(new_a-old_a)), direct_work, "independent path force/work duality",
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
        mechanical_work_j=work, numerically_integrated_work_j=direct_work,
        force_work_j=float(force@(new_a-old_a)),
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
        close(raw[key], expected["numerically_integrated_work_j"], "direct cohesive "+key, rtol=3e-9, atol=1e-9)
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



def audit_solver_receipt(raw, record, checked, parent, balanced, gamma):
    """Validate solver lineage; primary source actions certify its actual result."""
    if raw is None:
        return None
    seal(raw, "native_release_solve_sha256")
    response = record["response"]
    for key, wanted in (
        ("format", "genesis-source-forced-native-virgin-release-solve-0.1"),
        ("scope", "source_birth_native_range_journal_private_virgin_no_physical_release"),
        ("context_binding_sha256", checked["context_binding_sha256"]),
        ("before_memory_sha256", digest(record["before_memory"]["state"])),
        ("before_native_history_binding_sha256", digest(record["before_native"])),
        ("after_native_history_binding_sha256", digest(record["after_native"])),
        ("response_binding_sha256", digest(response)),
        ("source_quadrature_binding_sha256", response["source_quadrature_binding_sha256"]),
        ("after_time_myr", response["after_time_myr"]), ("dt_s", response["dt_s"]),
        ("increment_native_m", response["operative_increment_m"]),
        ("parent_ndof", checked["numerical_context"]["parent_ndof"]),
        ("compact_jump_rank", checked["modes"]["rank"]),
        ("tied_kernel_dimension", checked["modes"]["nullity"]),
        ("operative_coordinates", "source_birth_complete_B_range_orthogonal_W_rotation"),
        ("full18_expansion_scope", "rounded_arithmetic_witness_not_operative_history")):
        exact(raw[key], wanted, "native solver full DTO/source link "+key)
    scalar_zero_credit(raw)
    for key in ("physical_release_admitted", "funding_admitted", "accepted_planet_memory_changed"):
        require(raw[key] is False, "native solver cannot admit physical funding/history")
    for key in ("native_displacement_memory_velocity_authoritative",
                "source_birth_native_displacement_memory_velocity_exact_zero",
                "weak_source_fields_contracted_before_Gram_assembly",
                "complete_jump_kernel_remains_tied", "no_full18_roundtrip_used_for_operative_path",
                "no_ULP_jitter_or_tied_kernel_freedom_used", "whole_straight_path_virgin_by_convex_effective_radius"):
        require(raw[key] is True, "native solve source-preserving scope differs")
    require(raw["balanced_trajectory"] is True and balanced,
            "solved receipt must pass independently audited momentum/work")
    peak = checked["law"]["peak_traction_pa"]/checked["law"]["cohesive_stiffness_pa_m"]
    require(gamma["fracture_dissipation_increment_j"] == 0., "bounded virgin solve cannot acquire fracture loss")
    radius = lambda gap: np.linalg.norm(np.c_[np.maximum(gap[:, 0], 0), gap[:, 1:]], axis=1)
    old_radius = radius(gamma["old_gap_local_m"]+checked["reference_offset"])
    new_radius = radius(gamma["new_gap_local_m"]+checked["reference_offset"])
    require(np.all(old_radius <= peak) and np.all(new_radius <= peak)
            and np.all(array(record["old_maximum_separation_m"]) <= peak),
            "complete straight solved path must remain computationally virgin")
    exact(raw["peak_separation_m"], peak, "source-declared onset separation")
    for key, wanted in (("maximum_old_history_m", np.max(record["old_maximum_separation_m"])),
                        ("maximum_old_effective_separation_m", np.max(old_radius)),
                        ("maximum_new_effective_separation_m", np.max(new_radius))):
        close(raw[key], wanted, "independent virgin path "+key, rtol=3e-8, atol=1e-15)
    factors = [memory.volume.temporal_factors(x)[1] for x in memory.source_maxwell_ratios(
        parent, record["before_native"]["time_myr"], response["after_time_myr"])]
    close(raw["Maxwell_mean_stiffness_factors"], factors, "solver source Maxwell means", rtol=2e-9, atol=2e-13)
    return dict(solver_receipt_source_links_checked=True,
        result_momentum_and_work_independently_checked=True,
        straight_path_virgin_independently_checked=True,
        solver_conditioning_diagnostics_independently_recomputed=False)


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
    native = replay_native_history(record["before_native"], record["before_memory"], checked, parent, prior_case)
    after_native = replay_native_history(record["after_native"], record["after_memory"], checked, parent, prior_case)
    require(raw.get("native_coordinate_history_used") is True
            and raw.get("full18_roundtrip_used_for_operative_coordinates") is False,
            "authoritative native history cannot be recovered from rounded full18 fields")
    exact(raw["native_coordinate_scope"], NATIVE_SCOPE, "native response scope")
    exact(raw["full18_archive_fields_scope"],
          "expanded_float64_arithmetic_witness_only_not_authoritative_coordinates", "expanded archive scope")
    exact(raw["before_native_history_binding_sha256"], digest(record["before_native"]), "before native history link")
    exact(raw["after_native_history_binding_sha256"], digest(record["after_native"]), "after native history link")
    exact(raw["after_native_history"], record["after_native"], "embedded after native history")
    native_d = array(raw["operative_increment_m"], (n+r,))
    dw = native_d[n:]
    exact(raw["native_active_coordinate_count"], r, "complete native active range count")
    exact(raw["native_tied_kernel_coordinate_count"], 0, "native history has no kernel coordinate")
    previous["maximum"] = native["maximum"]
    d, v0 = array(raw["increment_full_m"], (n+18,)), array(raw["before_velocity_m_s"], (n+18,))
    exact(v0.tolist(), previous["velocity"].tolist(), "genuine-birth private journal velocity")
    exact(d.tolist(), np.r_[native_d[:n], np.asarray(W@np.asarray(dw, np.longdouble), float)].tolist(),
          "full18 arithmetic witness must expand the declared native increment")
    require_tied_kernel(modes, d[n:], "prospective increment")
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
    old_a, new_a = native["c"], np.asarray(native["c"]+dw, float)
    exact(after_native["c"].tolist(), new_a.tolist(), "native endpoint displacement continuation")
    for key, wanted in (("operative_enrichment_old_displacement_m", old_a),
                        ("operative_enrichment_new_displacement_m", new_a)):
        exact(raw[key], wanted.tolist(), "literal retained-field "+key)
    expected_gamma = independent_cohesive_path(checked["law"], source["initial_traction_world"],
        source["weights"], source["trace"], source["basis"], old_a, new_a, previous["maximum"])
    cohesive_report = audit_cohesive_path(raw["cohesive_response"], expected_gamma)
    legacy_gamma = raw["legacy_rounded_cohesive_response"]
    legacy_source = checked["legacy_source"]
    exact(raw["legacy_full18_cohesive_witness_evaluation_budget"], dict(max_subdivisions=0,
        scope="bounded_optional_rounded_full18_witness_not_intrinsic_failure_certification"),
        "bounded optional legacy arithmetic witness budget")
    available = raw["legacy_full18_cohesive_path_ledger_available"]
    require(type(available) is bool, "explicit legacy full18 path availability required")
    if available:
        require(legacy_gamma is not None and raw["legacy_full18_cohesive_path_refusal"] is None,
                "available legacy path must have its original arithmetic response")
        exact(raw["legacy_full18_cohesive_force_scope"],
              "literal_full18_generic_path_arithmetic_witness", "legacy generic path scope")
        exact(raw["legacy_full18_cohesive_witness_force_scope"], None,
              "available generic path does not use the native-traction fallback")
        expected_legacy_gamma = independent_cohesive_path(checked["law"], legacy_source["initial_traction_world"],
            legacy_source["weights"], legacy_source["trace"], legacy_source["basis"], previous["a"], previous["a"]+d[n:],
            journal_interface_maximum(checked, record["before_memory"], operative=False))
        audit_cohesive_path(legacy_gamma, expected_legacy_gamma)
        close(legacy_gamma["old_enrichment_m"], previous["a"], "legacy interface old field", rtol=0, atol=0)
        close(legacy_gamma["new_enrichment_m"], previous["a"]+d[n:], "legacy interface new field", rtol=0, atol=0)
        legacy_mean_force = array(legacy_gamma["mean_generalized_force_n"])
    else:
        # A refused old generic path supplies no fabricated zero energies and
        # cannot certify or fund the operative history.  Validate its explicit
        # class/scope and still reconstruct the full18 force witness literally.
        exact(legacy_gamma, None, "unavailable legacy path cannot manufacture an energy ledger")
        refusal = raw["legacy_full18_cohesive_path_refusal"]
        require(type(refusal) is dict and set(refusal) == {"exception_class", "message"}
            and refusal["exception_class"] == "VolumeReleaseCohesiveRefusal"
            and refusal["message"] in (
                "Cohesive mean traction/work quadrature did not converge component by component",
                "Cohesive assembled mean force/work failed component convergence or exact path duality"),
            "scoped unavailable legacy generic path refusal differs")
        exact(raw["legacy_full18_cohesive_force_scope"],
            "literal_full18_trace_contraction_of_authoritative_native_point_tractions_arithmetic_witness_only",
            "native-only literal full18 traction contraction scope")
        exact(raw["legacy_full18_cohesive_witness_force_scope"],
            "expanded_full_trace_action_of_primary_native_point_mean_traction_without_legacy_path_ledger",
            "fallback force carries no legacy path ledger")
        legacy_mean_force = np.einsum("pai,pa,p->i", legacy_source["trace"],
            array(raw["cohesive_response"]["mean_traction_world_pa"]), legacy_source["weights"])
    endpoint_witness = np.r_[np.zeros(n), np.einsum("pai,pa,p->i", legacy_source["trace"],
        array(raw["cohesive_response"]["endpoint_traction_world_pa"]), legacy_source["weights"])]
    exact(raw["legacy_expanded_cohesive_endpoint_force_n"], endpoint_witness.tolist(),
          "literal expanded endpoint force witness")
    exact(raw["kinetic_energy_before_after_scope"],
        "literal_full18_rounded_kinetic_subtraction_arithmetic_witness_only", "legacy kinetic ledger scope")
    M = seed.dense_csr(prior_case["embedded_partition"]["mass_matrix"], "literal expanded source mass")
    D = seed.dense_csr(prior_case["embedded_partition"]["basal_drag_matrix"], "literal expanded source drag")
    ext = array(prior_case["embedded_partition"]["external_force_n"])
    legacy_budget = discrete_path_budget(M, D, ext, d, v0, dt, old_bulk["mean_generalized_force_n"],
        legacy_mean_force, old_bulk["work_decomposition_j"],
        float(legacy_mean_force@d[n:]))
    exact(raw["after_velocity_m_s"], legacy_budget["after_velocity_m_s"].tolist(), "midpoint kinematic identity")
    for key, matrix, vector, factor in (("inertia_force_n", M, d-dt*v0, 2/dt**2),
                                      ("basal_drag_force_n", D, d, 1/dt)):
        matrix, vector = np.asarray(matrix, np.longdouble), np.asarray(vector, np.longdouble)
        action = (matrix@vector)*np.longdouble(factor)
        operand_scale = (abs(matrix)@abs(vector))*abs(np.longdouble(factor))
        require(np.all(abs(np.asarray(array(raw[key]), np.longdouble)-action) <= 512*EPS*operand_scale),
                "literal expanded arithmetic witness "+key+" differs")
    exact(raw["external_force_n"], prior_case["embedded_partition"]["external_force_n"], "unaltered complete source load")
    exact(raw["cohesive_mean_force_n"], np.r_[np.zeros(n), legacy_mean_force].tolist(),
          "expanded interface witness")
    legacy_residual = (array(raw["inertia_force_n"])+array(raw["basal_drag_force_n"])
        +array(raw["bulk_mean_force_n"])+array(raw["cohesive_mean_force_n"])-array(raw["external_force_n"]))
    exact(raw["residual_n"], legacy_residual.tolist(), "literal expanded residual witness")
    close(raw["tied_kernel_residual_n"], modes["declared_tied_null_modes"].T@legacy_residual[n:],
          "retained complete-kernel reaction witness", rtol=0, atol=1e-7)
    # Parent source fields retain their original archive; only additional
    # coordinate arrays use the independently source-born native journal.
    math_quad = native_coordinate_math_view(quad)
    native_previous = {**previous, "z": native["z"], "a": native["c"]}
    factors = memory.volume.temporal_factors(ratio[context["selected_index"]])
    native_mean = factors[0]*native["z"]+factors[1]*dw
    native_bulk = {**represented_bulk, "coefficients_m": after_native["z"],
        "mean_coefficients_m": native_mean}
    native_v0 = np.r_[v0[:n], native["velocity"]]
    expected = literal.restricted_current_path(context, math_quad, native_previous, native_bulk,
        inc0, inc1, native_d, native_v0, dt, ratio, expected_gamma["mean_traction_local_pa"],
        old_bulk["work_decomposition_j"], old_bulk["dissipation_decomposition_j"],
        old_bulk["before_energy_decomposition_j"], old_bulk["energy_decomposition_j"])
    # The persisted velocity follows the original float64 source recurrence;
    # the same journal has been replayed independently above.
    expected["operative_after_velocity_m_s"] = np.r_[legacy_budget["after_velocity_m_s"][:n],
        after_native["velocity"]]
    operand_bounds = literal.independent_current_operand_bounds(context, math_quad, native_previous,
        native_bulk, native_d, native_v0, dt, raw["cohesive_response"], parent_force)
    for key, wanted in (("operative_enrichment_old_memory_m", native["z"]),
                        ("operative_enrichment_new_memory_m", after_native["z"]),
                        ("operative_enrichment_mean_memory_m", native_mean)):
        for j in range(r):
            close(array(raw[key], (r,))[j], wanted[j], "source-born native Maxwell "+key+" "+str(j),
                  rtol=3e-9, atol=1e-25)
    local = prior_case["embedded_partition"]["local"]
    expansion, expansion_passed = source_expansion_witnesses(
        dict(displacement=local["enrichment_trace"], strain=local["enrichment_strain_operator"],
            interface_gap=checked["legacy_source"]["trace"]),
        dict(displacement=quad["E_w"], strain=quad["G_w"], interface_gap=source["trace"]), W,
        dict(old_displacement=previous["a"], new_displacement=bulk["displacement_m"], displacement_increment=d[n:],
            before_velocity=v0[n:], after_velocity=raw["after_velocity_m_s"][n:],
            old_elastic_memory=previous["z"], new_elastic_memory=bulk["coefficients_m"], mean_elastic_memory=bulk["mean_coefficients_m"]),
        dict(old_displacement=old_a, new_displacement=new_a, displacement_increment=dw,
            before_velocity=native["velocity"], after_velocity=2*dw/dt-native["velocity"],
            old_elastic_memory=native["z"], new_elastic_memory=after_native["z"],
            mean_elastic_memory=np.asarray(raw["operative_enrichment_mean_memory_m"], np.longdouble)))
    require(raw["source_point_expansion_passed"] is expansion_passed,
            "reported full18 source point arithmetic discrepancy differs")
    require(raw["source_point_expansion_is_native_admission_requirement"] is False,
            "rounded full18 point witnesses cannot define native admission")
    exact(raw["source_point_expansion_bound_scope"],
        "256_float64_epsilon_times_actual_source_lift_projection_operands_no_floor", "source field representation precision")
    require(set(raw["source_point_expansion_witnesses"]) == set(expansion), "complete represented field expansion witnesses required")
    for key, expected_witness in expansion.items():
        actual_witness = raw["source_point_expansion_witnesses"][key]
        require(actual_witness["passed"] is expected_witness["passed"], "source field expansion verdict differs")
        for field in expected_witness:
            if field != "passed": close(actual_witness[field], expected_witness[field], "source field expansion "+key+" "+field, rtol=3e-12, atol=0)
    kernel_witnesses, kernel_all_passed = legacy_kernel_witnesses(modes["declared_tied_null_modes"],
        dict(old_displacement=previous["a"], new_displacement=bulk["displacement_m"],
            displacement_increment=d[n:], before_velocity=v0[n:], after_velocity=raw["after_velocity_m_s"][n:],
            old_elastic_memory=previous["z"], new_elastic_memory=bulk["coefficients_m"],
            mean_elastic_memory=bulk["mean_coefficients_m"]))
    exact(raw["legacy_full18_kernel_witness_scope"],
        "rounded_full18_cumulative_source_journal_arithmetic_witness_only_not_native_admission",
        "nonadmitting cumulative full18 kernel witness scope")
    require(raw["legacy_full18_kernel_witness_all_passed"] is kernel_all_passed,
            "reported cumulative full18 kernel diagnostic verdict differs")
    require(set(raw["legacy_full18_kernel_witnesses"]) == set(kernel_witnesses),
            "complete cumulative full18 kernel diagnostics required")
    for key, wanted in kernel_witnesses.items():
        actual = raw["legacy_full18_kernel_witnesses"][key]
        exact(actual["literal18_kernel_tolerance_passed"], wanted["literal18_kernel_tolerance_passed"],
              "literal kernel witness criterion "+key)
        for field in ("norm_magnitude", "null_component_norm", "relative_null_component"):
            close(actual[field], wanted[field], "literal kernel witness "+key+" "+field, rtol=3e-12, atol=0)
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
        exact(raw[key], np.asarray(expected[key], float).tolist(), "source-born native operative kinematics "+key)
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
    addition = endpoint_addition_witness(previous["a"], bulk["displacement_m"], d[n:], old_a, new_a, dw)
    exact(raw["operative_endpoint_addition_scope"],
        "actual_source_coefficient_addition_and_coordinate_operands_gamma_n_no_floor_no_credit",
        "native endpoint addition arithmetic scope")
    for key, wanted in (("operative_endpoint_enrichment_increment_m", addition["endpoint_increment"]),
                        ("operative_endpoint_addition_difference_m", addition["difference"]),
                        ("operative_endpoint_addition_roundoff_bound_m", addition["bound"]),
                        ("full_enrichment_endpoint_addition_difference_m", addition["full_difference"]),
                        ("full_enrichment_endpoint_addition_roundoff_bound_m", addition["full_bound"])):
        exact(raw[key], np.asarray(wanted, float).tolist(), "own-scale endpoint addition "+key)
    require(raw["operative_endpoint_addition_passed"] is addition["passed"], "native endpoint addition verdict differs")
    gamma_residual = raw["cohesive_work_j"]-raw["cohesive_response"]["numerically_integrated_work_j"]
    gamma_bound = (work_tolerance*(abs(raw["cohesive_work_j"])
        +abs(raw["cohesive_response"]["numerically_integrated_work_j"]))
        +128*EPS*float(np.sum(abs(parts[3]*np.asarray(operative_d, np.longdouble))))
        +raw["cohesive_response"]["work_integral_roundoff_bound_j"]
        +float(per_force_bounds[3]@np.asarray(abs(operative_d), np.longdouble)))
    exact(raw["cohesive_force_work_duality_residual_j"], gamma_residual, "actual native cohesive path dual work")
    close(raw["cohesive_force_work_duality_bound_j"], gamma_bound,
          "current cohesive path arithmetic cannot use cumulative fracture loss", rtol=3e-15, atol=0)
    gamma_duality_passed = bool(abs(gamma_residual) <= gamma_bound and addition["passed"])
    require(raw["cohesive_force_work_duality_passed"] is gamma_duality_passed,
            "native cohesive mean-force/source-path verdict differs")
    identity_passed = bool(abs(raw["work_balance_residual_j"]+raw["force_residual_work_j"]) <= identity_bound
                           and small_passed and duality_passed)
    identity_passed = bool(identity_passed and gamma_duality_passed)
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
    native_item = record["after_native"]["journal"][-1]
    exact(native_item["increment_w_m"], dw.tolist(), "native journal contains declared operative increment")
    exact(record["after_native"]["journal"], record["before_native"]["journal"]+[native_item],
          "native journal contains exactly one new source continuation step")
    exact(raw["after_memory_sha256"], digest(after_memory["state"]), "response after-memory link")
    resumed = replay_memory_archive(after_memory, checked, parent, prior_case)
    require(record["source_memory_restart_exact"] is True and record["source_work_restart_exact"] is True
            and record["source_native_restart_exact"] is True,
            "private restart witnesses missing")
    solver_report = audit_solver_receipt(record.get("solver"), record, checked, parent, balanced, expected_gamma)
    return dict(label=record["label"], private_after_time_myr=after, complete_kernel_tied=True,
        solver=solver_report,
        before_memory_replayed=previous["report"], after_memory_replayed=resumed["report"],
        before_native_replayed=native["report"], after_native_replayed=after_native["report"],
        source_native_restart_exact=True, authoritative_native_history_starts_at_source_birth_zero=True,
        mature_full18_projection_used=False, full18_expansion_is_arithmetic_witness_only=True,
        direct_cohesive_path=cohesive_report, literal_source_point_current_budget_checked=True,
        legacy_expanded_actions_are_arithmetic_witnesses=True,
        legacy_full18_kernel_witness_all_passed=kernel_all_passed,
        legacy_full18_point_expansion_witness_passed=expansion_passed,
        legacy_full18_kernel_and_point_witnesses_are_native_admission_requirements=False,
        native_history_has_no_complete_kernel_coordinates=True,
        native_active_coordinate_count=r, native_tied_kernel_coordinate_count=0,
        complete_source_kernel_dimension=modes["nullity"],
        legacy_full18_cohesive_path_ledger_available=available,
        legacy_full18_cohesive_refusal_has_no_funding_authority=not available,
        legacy_full18_unavailable_path_classification=(None if available else
            "reported_conservative_unavailable_witness_class_message_scope_sealed"),
        legacy_full18_path_failure_independently_verified=False,
        fresh_physical_weak_source_accuracy_certified=False,
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
    exact(fixture["format"], FIXTURE_FORMAT, "native control fixture format")
    exact(fixture["scope"], FIXTURE_SCOPE, "native computational control scope")
    require(fixture.get("operative_coordinate_history_starts_at_source_birth_zero") is True
            and fixture.get("operative_coordinate_history_recovered_from_full18") is False
            and fixture.get("full18_expansion_is_rounded_arithmetic_witness_only") is True,
            "native fixture must preserve the genuine source-born coordinate history")
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
        accepted_Genesis_changed=False,
        authoritative_native_history_starts_at_source_birth_zero=True,
        mature_full18_projection_used=False, full18_expansion_is_arithmetic_witness_only=True,
        fresh_physical_weak_source_accuracy_certified=False, source_selected_clipped_accuracy_certified=False,
        cases=reports)
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
