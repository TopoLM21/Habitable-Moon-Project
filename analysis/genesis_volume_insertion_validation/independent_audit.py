"""Producer-free interval reaction and prestress-aware Γ work oracle.

Reference cut supports and all new constitutive parameters are computational.
Recovering the last accepted discrete reaction does not select a physical
Genesis fracture or admit its release, contact or underthrust trajectory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analysis.genesis_volume_bank_validation.independent_audit import cohort_quadrature
from analysis.genesis_volume_creation_validation.independent_audit import (
    array, canonical, close, surface_oracle, surface_jump, surface_force,
    validate_law, law_point, integrate_traction_work,
    audit_surface_claims, audit_context,
)
from analysis.genesis_volume_continuation_validation.independent_audit import (
    SOURCE_ANGULAR_POINTS, basal_drag_and_force, independent_strain_operators,
    isotropic_matrix, mass_matrix, temporal_factors, endpoint_clock_binding,
    radial_rule, conical_scales, source_dto_binding,
)

PARENT_RAW_SHA256 = "62579a0bad128830a66e2f44579cf2b1f5d3835b691225b8d431102f3e4486d7"
PARENT_CHECKPOINT_SHA256 = "5d282ceeccc8d76acc38427d08a7e65575fddb509fc4b93115cbe6d318e91349"
FORMATION_SHA256 = "cd79ee2a5139c4b91abdb1b4714a761c9153cc9f4490f5aa63fa2facba458c89"
FORMAT = "genesis-volume-insertion-control-0.1"
SCOPE = "accepted_interval_reaction_replacement_and_seeded_creation_probes_not_physical_fracture"
REACTION_SCOPE = "accepted_interval_average_on_computational_support_not_instantaneous_fracture_release"


def canonical_jump_lift(J):
    J = array(J)
    if J.ndim != 2:
        raise ValueError("Independent insertion jump matrix must be two-dimensional")
    if not len(J):
        return np.empty((J.shape[1], 0))
    gram = J@J.T
    if np.linalg.matrix_rank(gram) != len(J):
        raise ValueError("Independent insertion jump constraints must be independent")
    return np.linalg.solve(gram, J).T


def interval_reaction(J, residual):
    """Cancel the jump-space residual while retaining the parent residual."""
    J, residual = array(J), array(residual)
    lift = canonical_jump_lift(J)
    if residual.shape != (J.shape[1],):
        raise ValueError("Independent insertion residual has wrong bank dimension")
    eta = -lift.T@residual
    force = J.T@eta
    remaining = residual+force
    close(J@remaining, np.zeros(len(J)), "remaining jump-space residual", rtol=0,
          atol=1e-13*max(np.max(abs(residual), initial=0.), 1.))
    return dict(canonical_reaction_n=eta, interface_force_n=force,
                remaining_residual_n=remaining)


def global_surface_trace(surface):
    points = 12*len(surface["patch_edge_indices"])
    B = np.zeros((points, surface["bank_ndof"]))
    for patch, dofs in enumerate(surface["trace_dofs"]):
        rows = np.arange(12*patch, 12*patch+12)
        local = surface["jump_trace"][patch].reshape(12, 24)
        # Coincident tip DOFs must accumulate rather than overwrite.
        for column, global_column in enumerate(dofs):
            B[rows, global_column] += local[:, column]
    return B


def minimum_area_traction(surface, eta):
    """Minimum physical L2 representative from a conditioned Gram solve.

    The producer uses SVD of the weighted rectangular trace.  This oracle
    solves the smaller positive jump-mode Gram system instead, then checks
    the full unscaled A/P force and physical work identities.
    """
    J = surface["topology"]["jump_operator"]
    eta = array(eta, (len(J),))
    B, lift = global_surface_trace(surface), canonical_jump_lift(J)
    H = B@lift
    close(B, H@J, "surface trace factors through canonical jumps", atol=2e-12)
    weights = np.repeat(surface["area_weights_m2"].ravel(), 3)
    if not len(J):
        return dict(traction_world_pa=np.zeros((len(surface["patch_edge_indices"]), 4, 3)),
            H=H, modal_scale=np.zeros(0), singular_values=np.zeros(0),
            rank=0, interface_force_n=np.zeros(surface["bank_ndof"]),
            physical_l2_norm_squared_pa2_m2=0.)
    depth = 1-float(np.min(surface["lambda_bounds"][:, 0]))
    if depth <= 0 or np.any(weights <= 0):
        raise ValueError("Independent insertion needs positive paired depth and measure")
    # Canonical jump rows have one duplicated global component per row.
    components = [int(np.flatnonzero(row > 0)[0]) % 6 for row in J]
    modal_scale = np.asarray([1. if c < 3 else 1/depth for c in components])
    weighted = modal_scale[:, None]*H.T*np.sqrt(weights)[None]
    gram = weighted@weighted.T
    eig, vectors = np.linalg.eigh(gram)
    if eig[0] <= 0 or eig[0] <= eig[-1]*1e-24:
        raise ValueError("Independent insertion Γ does not span all canonical jump modes")
    target = modal_scale*eta
    dual = vectors@((vectors.T@target)/eig)
    density = H@(modal_scale*dual)
    reproduced = B.T@(weights*density)
    force = J.T@eta
    for start, name in ((0, "A"), (3, "P")):
        close(reproduced.reshape(-1, 6)[:, start:start+3], force.reshape(-1, 6)[:, start:start+3],
              f"unscaled traction force {name}", rtol=5e-10, atol=1e-5)
    close(H.T@(weights*density), eta, "unscaled traction jump modes", rtol=5e-10, atol=1e-5)
    return dict(traction_world_pa=density.reshape(len(surface["patch_edge_indices"]), 4, 3),
        H=H, modal_scale=modal_scale, singular_values=np.sqrt(eig[::-1]), rank=len(J),
        interface_force_n=reproduced,
        physical_l2_norm_squared_pa2_m2=float(np.sum(weights*density*density)))


def prestress_seed(law, initial_traction):
    """Derive offsets and the added auxiliary area under the cohesive law."""
    Gc, peak, stiffness, compression, onset, _ = validate_law(law)
    traction = array(initial_traction)
    if traction.shape[-1:] != (3,):
        raise ValueError("Independent prestress traction must have three modes")
    offset = traction/stiffness
    offset[..., 0] = np.where(traction[..., 0] < 0, traction[..., 0]/compression, offset[..., 0])
    positive = offset.copy(); positive[..., 0] = np.maximum(positive[..., 0], 0.)
    maximum = np.hypot.reduce(positive, axis=-1)
    epsilon = np.finfo(float).eps
    tolerance = 32*epsilon*np.maximum(maximum, onset)
    if np.any(maximum > onset+tolerance):
        raise ValueError("Independent prestress seed exceeds the declared peak traction")
    factor = np.ones(maximum.shape)
    near_peak = maximum > onset
    factor[near_peak] = onset*(1-4*epsilon)/maximum[near_peak]
    offset[..., 1:] *= factor[..., None]
    offset[..., 0] = np.where(offset[..., 0] > 0, offset[..., 0]*factor, offset[..., 0])
    positive = offset.copy(); positive[..., 0] = np.maximum(positive[..., 0], 0.)
    maximum = np.hypot.reduce(positive, axis=-1)
    reference = stiffness*offset
    reference[..., 0] = np.where(offset[..., 0] < 0, compression*offset[..., 0], reference[..., 0])
    B = .5*stiffness*maximum*maximum
    C = .5*compression*np.minimum(offset[..., 0], 0.)**2
    auxiliary = Gc+B
    return dict(initial_traction_pa=traction, reference_traction_pa=reference,
        reference_peak_snap_factor=factor, reference_offset_m=offset,
        initial_maximum_separation_m=maximum, reference_cohesive_energy_j_m2=B,
        reference_compression_energy_j_m2=C, auxiliary_fracture_energy_j_m2=auxiliary,
        final_separation_m=2*auxiliary/peak)


def prestress_segment(law, initial_traction, old_gap, new_gap, old_maximum):
    """Separate shifted traction-path integration from reference energy."""
    seed = prestress_seed(law, initial_traction)
    old_gap, new_gap = array(old_gap), array(new_gap)
    if old_gap.shape != seed["reference_offset_m"].shape or new_gap.shape != old_gap.shape:
        raise ValueError("Independent prestress gap dimensions differ")
    old_h, new_h = old_gap+seed["reference_offset_m"], new_gap+seed["reference_offset_m"]
    maximum = array(old_maximum, old_gap.shape[:-1])
    effective_old = old_h.copy(); effective_old[..., 0] = np.maximum(effective_old[..., 0], 0.)
    needed = np.maximum(np.hypot.reduce(effective_old, axis=-1), seed["initial_maximum_separation_m"])
    tolerance = 32*np.finfo(float).eps*np.maximum(np.maximum(maximum, needed), validate_law(law)[4])
    if np.any(maximum < needed-tolerance):
        raise ValueError("Independent prestress history is below the inserted reference state")
    maximum = np.maximum(maximum, needed)
    shape = maximum.shape
    outputs = {key: np.empty(shape+((3,) if key == "traction_pa" else (3, 3) if key == "tangent_pa_m" else ()))
               for key in ("traction_pa", "tangent_pa_m", "maximum_separation_m", "damage",
                           "stored_energy_j_m2", "fracture_dissipation_j_m2",
                           "fracture_dissipation_increment_j_m2", "mechanical_work_j_m2")}
    for index in np.ndindex(shape):
        point_law = {**law, "fracture_energy_j_m2": float(seed["auxiliary_fracture_energy_j_m2"][index])}
        old = law_point(point_law, old_h[index], maximum[index])
        new = law_point(point_law, new_h[index], maximum[index])
        baseline = seed["reference_cohesive_energy_j_m2"][index]
        compression_ref = seed["reference_compression_energy_j_m2"][index]
        # Evaluate the change before summing compression and cohesive terms;
        # subtracting two absolute stored energies would lose zero-birth
        # precision and could hide an invented reference-energy debit.
        def relative_energy(h, point):
            opening = h.copy(); opening[0] = max(opening[0], 0.)
            r = np.hypot.reduce(opening)
            cohesive = .5*law["cohesive_stiffness_pa_m"]*r*r
            compressed = .5*law["compression_stiffness_pa_m"]*min(h[0], 0.)**2
            return (1-point["damage"])*(cohesive-baseline)+(compressed-compression_ref)
        psi0, psi1 = relative_energy(old_h[index], old), relative_energy(new_h[index], new)
        D0 = old["fracture_dissipation_j_m2"]-baseline*old["damage"]
        D1 = new["fracture_dissipation_j_m2"]-baseline*new["damage"]
        work = integrate_traction_work(point_law, old_h[index][None], new_h[index][None], np.asarray([maximum[index]]))[0]
        rounding = 256*np.finfo(float).eps*max(point_law["fracture_energy_j_m2"], baseline, compression_ref, 1.)
        close(work, psi1-psi0+D1-D0, "prestress-aware direct traction work", rtol=3e-10, atol=rounding)
        for key in ("traction_pa", "tangent_pa_m", "maximum_separation_m", "damage"):
            outputs[key][index] = new[key]
        outputs["stored_energy_j_m2"][index] = psi1
        outputs["fracture_dissipation_j_m2"][index] = D1
        outputs["fracture_dissipation_increment_j_m2"][index] = D1-D0
        outputs["mechanical_work_j_m2"][index] = work
    return {**outputs, "seed": seed}


def source_frames(parent):
    source = parent["source_packet"]
    first_two = array(source["current_geometry"]["face_frames_xyz"])
    return np.concatenate((first_two, np.cross(first_two[:, :, 0], first_two[:, :, 1])[:, :, None]), axis=2)


def accepted_interval_forces(parent, surface):
    """Rebuild the complete accepted average response on split connectivity."""
    runtime = parent["runtime_state"]; geometry = runtime["geometry"]
    report = runtime["last_report"]; inputs = report["audit_input"]
    slip = report["weak_plane_slip"]
    if any(slip["active"]) or any(runtime["fault_history"]["fault_active"]):
        raise ValueError("Independent insertion fixture requires the actual inactive source")
    if canonical(inputs["after_cohorts"]) != canonical(geometry["cohorts"]):
        raise ValueError("Independent insertion average response lost accepted material catalogue")
    dt = (report["after_time_myr"]-report["before_time_myr"])*(365.25*86400*1e6)
    if dt <= 0 or report["after_time_myr"] != geometry["time_myr"]:
        raise ValueError("Independent insertion requires the actual last accepted interval")
    faces = array(geometry["faces"], integer=True)
    vertices = array(geometry["vertices_m"])
    topology = surface["topology"]
    split_faces = topology["faces"]
    split_vertices = vertices[topology["parent_vertex"]]
    masses, rho = array(geometry["column_mass_kg"]), float(geometry["density_kg_m3"])
    cohorts = geometry["cohorts"]
    source = parent["source_packet"]; shell = source["source"]["parameters"]["shell"]
    frames = source_frames(parent)
    increment, v0 = array(inputs["increment_m"]), array(inputs["before_velocity_m_s"])
    q0 = array(inputs["before_displacement_m"])
    close(q0+increment, runtime["displacement_m"], "accepted interval displacement endpoint", rtol=0, atol=0)
    close(2*increment/dt-v0, runtime["velocity_m_s"], "accepted interval midpoint velocity", rtol=0, atol=0)
    mean0, mean1 = array(slip["average_strain0"]), array(slip["average_strain1"])
    if mean0.shape != (len(cohorts), 3, 6) or mean1.shape != mean0.shape:
        raise ValueError("Independent insertion needs all angular/radial average coefficients")
    parent_force, bank_force = np.zeros(6*len(vertices)), np.zeros(surface["bank_ndof"])
    worst_average = 0.
    for index, cohort in enumerate(cohorts):
        face = int(cohort["face_index"])
        frame = frames[face]
        C = isotropic_matrix(shell["young_modulus_pa"], shell["poisson_ratio"],
            runtime["fault_history"]["damage"][face], shell["residual_stiffness"])
        close(inputs["basis_xyz"][index], frame, "accepted source cohort frame", rtol=3e-12)
        close(inputs["elasticity_pa"][index], C, "accepted source cohort elasticity", rtol=3e-12, atol=1e-4)
        dofs = (6*faces[face, :, None]+np.arange(6)).ravel()
        B0, B1 = independent_strain_operators(vertices[faces[face]], frame, SOURCE_ANGULAR_POINTS)
        retention, driving = temporal_factors(inputs["maxwell_ratio"][index])[:2]
        computed0 = retention*array(inputs["old0"][index])+driving*(B0@increment[dofs]-array(inputs["thermal0"][index]))
        computed1 = retention*array(inputs["old1"][index])+driving*(B1@increment[dofs])
        worst_average = max(worst_average,
            close(mean0[index], computed0, "accepted average strain0", rtol=3e-10, atol=2e-13),
            close(mean1[index], computed1, "accepted average strain1", rtol=3e-10, atol=2e-13))
        local = cohort_quadrature(vertices[faces[face]], frame, masses[face], rho,
            cohort, mean0[index], mean1[index], C)["force_n"]
        np.add.at(parent_force, dofs, local)
        bank_dofs = (6*split_faces[face, :, None]+np.arange(6)).ravel()
        np.add.at(bank_force, bank_dofs, local)
    traction = array(inputs["basal_traction_at_vertices_pa"])
    drag = float(inputs["basal_drag_pa_s_m"])
    parameters = json.loads(runtime["contact_parameters_json"])
    if drag != parameters["basal_drag_pa_s_m"]:
        raise ValueError("Independent insertion changed the source basal resistance")
    M = mass_matrix(vertices, faces, masses, rho, cohorts)
    D, external = basal_drag_and_force(vertices, faces, masses, rho, cohorts, drag, traction[faces])
    Ms = mass_matrix(split_vertices, split_faces, masses, rho, cohorts)
    Ds, external_s = basal_drag_and_force(split_vertices, split_faces, masses, rho, cohorts,
        drag, traction[topology["parent_vertex"]][split_faces])
    R = topology["prolongation"]
    bank_increment, bank_v0 = R@increment, R@v0
    inertia = 2*(Ms@(bank_increment-dt*bank_v0))/dt**2
    drag_force = Ds@bank_increment/dt
    residual = inertia+drag_force+bank_force-external_s
    parent_inertia = 2*(M@(increment-dt*v0))/dt**2
    parent_drag = D@increment/dt
    parent_residual = parent_inertia+parent_drag+parent_force-external
    close(R.T@residual, parent_residual, "split residual parent pullback", rtol=0,
        atol=1e-12*max(np.max(abs(parent_force), initial=0.), 1.))
    recovered = interval_reaction(topology["jump_operator"], residual)
    return dict(dt_s=dt, parent_increment_m=increment, parent_before_velocity_m_s=v0,
        interval_average_elastic_constant=mean0, interval_average_elastic_inverse_scale=mean1,
        bank_increment_m=bank_increment, bank_before_velocity_m_s=bank_v0,
        parent_residual_n=parent_residual, bank_inertia_force_n=inertia,
        bank_drag_force_n=drag_force, bank_average_internal_force_n=bank_force,
        bank_external_force_n=external_s, bank_residual_n=residual,
        constraint_multipliers_n=recovered["canonical_reaction_n"],
        constraint_force_n=recovered["interface_force_n"],
        remaining_residual_n=recovered["remaining_residual_n"],
        worst_average_coefficient_relative_error=worst_average)


def accepted_face_stress(parent):
    """Integrate inherited stress then project onto current physical axes."""
    runtime, source = parent["runtime_state"], parent["source_packet"]
    geometry = runtime["geometry"]
    vertices, faces = array(geometry["vertices_m"]), array(geometry["faces"], integer=True)
    frames = source_frames(parent)
    masses, rho = array(geometry["column_mass_kg"]), float(geometry["density_kg_m3"])
    shell = source["source"]["parameters"]["shell"]
    memories = {int(c["id"]): c for c in runtime["mechanics"]["state"]["cohorts"]}
    stress_mean = np.zeros((len(faces), 3, 3)); volumes = np.zeros(len(faces))
    for cohort in geometry["cohorts"]:
        face, identity = int(cohort["face_index"]), int(cohort["id"])
        q, weights = radial_rule(cohort["q_lo"], cohort["live_q_hi"])
        radial, _ = conical_scales(vertices[faces[face]], masses[face], rho, q)
        mean = array(memories[identity]["strain0"]).mean(axis=0)+(weights@(1/radial))*array(memories[identity]["strain1"]).mean(axis=0)
        C = isotropic_matrix(shell["young_modulus_pa"], shell["poisson_ratio"],
            runtime["fault_history"]["damage"][face], shell["residual_stiffness"])
        s = C@mean
        tensor = np.array(((s[0], s[3], s[4]), (s[3], s[1], s[5]), (s[4], s[5], s[2])))
        world = frames[face]@tensor@frames[face].T
        volume = masses[face]*(cohort["live_q_hi"]-cohort["q_lo"])/rho
        stress_mean[face] += volume*world; volumes[face] += volume
    stress_mean[volumes > 0] /= volumes[volumes > 0, None, None]
    physical = vertices+array(runtime["displacement_m"]).reshape(-1, 6)[:, :3]
    result = []
    for face, triangle in enumerate(physical[faces]):
        tangent = triangle[1]-triangle[0]; tangent /= np.linalg.norm(tangent)
        normal = np.cross(triangle[1]-triangle[0], triangle[2]-triangle[0]); normal /= np.linalg.norm(normal)
        frame = np.column_stack((tangent, np.cross(normal, tangent), normal))
        local = frame.T@stress_mean[face]@frame
        result.append((local[0, 0], local[1, 1], local[0, 1]))
    return np.asarray(result)


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError("Independent insertion audit: "+message)


def seal(packet, key):
    require(type(packet) is dict and packet.get(key) == digest(
        {k: v for k, v in packet.items() if k != key}), key+" differs")


def fixed(packet, expected, name):
    for key, value in expected.items():
        require(canonical(packet.get(key)) == canonical(value), name+" "+key+" differs")


def force_families(actual, expected, references, name, *, tolerance=3e-9):
    a, b = array(actual).reshape(-1, 6), array(expected).reshape(-1, 6)
    require(a.shape == b.shape, name+" bank dimensions differ")
    refs = [array(x).reshape(-1, 6) for x in references]
    rounding = 1024*np.finfo(float).eps*max(
        (float(np.linalg.norm(x)) for x in refs), default=1.)
    errors = []
    for section, family in ((slice(0, 3), "A"), (slice(3, 6), "P")):
        scale = max((float(np.linalg.norm(x[:, section])) for x in refs), default=0.)
        error = float(np.linalg.norm((a-b)[:, section]))
        require(error <= tolerance*scale+rounding+1e-7, name+" "+family+" force differs")
        errors.append(error/max(scale, rounding, 1e-300))
    return errors


def actual_parent(path):
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == PARENT_RAW_SHA256, "literal parent raw pin differs")
    parent = json.loads(raw)
    seal(parent, "checkpoint_sha256")
    require(parent["checkpoint_sha256"] == PARENT_CHECKPOINT_SHA256, "accepted parent self pin differs")
    require(parent["source_checkpoint_sha256"] == FORMATION_SHA256, "formation source differs")
    source_dto_binding(parent)
    runtime = parent["runtime_state"]; geometry = runtime["geometry"]
    time = geometry["time_myr"]
    endpoint_clock_binding(runtime, time)
    require(abs(time-.8684) < 1e-14 and len(geometry["cohorts"]) == 800,
            "accepted birth endpoint or material catalogue differs")
    require(runtime["accepted_step_sizes_myr"] == [.0001]*4,
            "accepted four-interval ancestry differs")
    require(runtime["fault_history"]["time_myr"] == time
            and runtime["fault_history"]["accepted_steps"] == 4
            and runtime["mechanics"]["state"]["accepted_steps"] == 4,
            "mechanical and fault clocks differ")
    require(not any(runtime["fault_history"]["fault_active"]), "source has an unexpected active plane")
    for subsystem in (geometry, runtime["history"], runtime["mechanics"]["state"], runtime["fault_history"]):
        for key, reference in (("source_checkpoint_sha256", FORMATION_SHA256),
                               ("source_identity", geometry["source_identity"])):
            if key in subsystem:
                require(subsystem[key] == reference, "source ancestry differs in "+key)
    return parent


def audit_onset(onset, parent):
    seal(onset, "onset_sha256")
    runtime, source = parent["runtime_state"], parent["source_packet"]
    geometry, history = runtime["geometry"], runtime["fault_history"]
    faults = source["source"]["parameters"]["faults"]
    ready = ((~np.asarray(history["fault_active"], dtype=bool))
        & (array(history["damage"]) >= faults["activation_damage"])
        & (array(history["fault_candidate_age_myr"]) >= faults["activation_persistence_myr"]))
    require(not (faults["enabled"] and history["membrane_established"] and any(ready)),
            "actual onset oracle unexpectedly found a ready source plane")
    expected = dict(format="genesis-volume-calculated-fracture-onset-prerequisite-0.1",
        scope="accepted_start_weak_plane_event_not_located_sharp_surface_or_release",
        parent_checkpoint_sha256=parent["checkpoint_sha256"], formation_checkpoint_sha256=FORMATION_SHA256,
        source_identity=geometry["source_identity"], time_myr=geometry["time_myr"],
        accepted_step_sizes_myr=runtime["accepted_step_sizes_myr"], accepted_steps=4,
        event_evaluation_scope="next_interval_start_at_current_accepted_state",
        source_fault_parameters=faults, accepted_fault_history=history, prepared_fault_history=history,
        new_activation_event=None, newly_activating_faces=[], existing_active_faces=[],
        prepared_plane_normal_world=np.zeros((len(geometry["faces"]), 3)).tolist(),
        prepared_slip_direction_world=np.zeros((len(geometry["faces"]), 3)).tolist(),
        live_cohorts=[dict(cohort_id=int(c["id"]), face_index=int(c["face_index"]),
            owner_id=c["owner_id"], material_id=c["material_id"]) for c in geometry["cohorts"]],
        inherited_runtime_ledgers={k: v for k, v in runtime.items() if k.endswith("_j")},
        material_surface_status="no_calculated_activation_event", selected_sharp_material_surface=None,
        source_creation_energy_law=None, physical_creation_admitted=False, physical_release_admitted=False,
        physical_fault_contact_admission_claimed=False, creation_work_debited_from_Genesis_j=0.,
        history_reinitialized=False, activation_accepted=False, clock_advanced=False)
    require(set(onset) == set(expected)|{"computed_face_stress_pa", "onset_sha256"},
            "whole onset schema differs")
    fixed(onset, expected, "source-bound onset")
    stress_error = close(onset["computed_face_stress_pa"], accepted_face_stress(parent),
        "independently integrated accepted-start stress", rtol=2e-10, atol=1e-3)
    return dict(current_start_stress_independently_integrated=True,
        source_activation_rule_and_unchanged_history_checked=True, stress_relative_error=stress_error,
        new_activation_event=None, sharp_material_surface_selected=False)


def audit_reaction(raw, parent, surface, geometry):
    seal(raw, "reaction_sha256")
    oracle = accepted_interval_forces(parent, surface)
    runtime = parent["runtime_state"]; report = runtime["last_report"]
    expected_context = dict(format="genesis-accepted-interval-volume-bank-reaction-0.1", scope=REACTION_SCOPE,
        accepted_parent_checkpoint_sha256=parent["checkpoint_sha256"], formation_checkpoint_sha256=FORMATION_SHA256,
        source_identity=runtime["geometry"]["source_identity"], time_myr=runtime["geometry"]["time_myr"],
        before_time_myr=report["before_time_myr"], after_time_myr=report["after_time_myr"],
        accepted_step_sizes_myr=runtime["accepted_step_sizes_myr"], accepted_final_report_sha256=digest(report),
        full_memory_sha256=digest(runtime["mechanics"]), birth_history_sha256=digest(runtime["history"]),
        fault_history_sha256=digest(runtime["fault_history"]), bank_binding_sha256=geometry["bank_binding_sha256"],
        cohort_ids=[int(c["id"]) for c in runtime["geometry"]["cohorts"]],
        parent_momentum_relative_residual=report["momentum_equation_relative_residual"],
        mass_convention="after_mass_with_accepted_start_comoving_velocity",
        traction_scope="last_accepted_interval_average_not_instantaneous_endpoint",
        source_calculated_creation_law=None, physical_creation_admitted=False, physical_release_admitted=False,
        calculated_physical_fracture_surface=False, transition_time_increment_myr=0.,
        creation_work_debited_from_Genesis_j=0., source_memory_reinitialized=False, parent_residual_preserved=True)
    fixed(raw, expected_context, "accepted interval reaction")
    context_fields = set(expected_context)|{"force_pullback_relative_errors_A_P",
        "constraint_force_null_relative_errors_A_P", "remaining_pullback_relative_errors_A_P",
        "recomputed_parent_momentum_relative_residual", "tied_constraint_work_j", "exact_canonical_jump_work_j"}
    require(raw["reaction_context_sha256"] == digest({k: raw[k] for k in context_fields}),
            "reaction context seal differs")
    array_fields = set(oracle)-{"worst_average_coefficient_relative_error"}
    require(set(raw) == context_fields|array_fields|{"reaction_context_sha256",
        "prepared_reaction_binding_sha256", "reaction_sha256"}, "whole reaction schema differs")
    forces = [oracle[k] for k in ("bank_inertia_force_n", "bank_drag_force_n",
        "bank_average_internal_force_n", "bank_external_force_n")]
    errors = {}
    for key in array_fields:
        if key.endswith("force_n") or key in ("bank_residual_n", "remaining_residual_n"):
            errors[key] = force_families(raw[key], oracle[key], forces, "source-derived "+key)
        elif key == "parent_residual_n":
            pulled = [surface["topology"]["prolongation"].T@value for value in forces]
            errors[key] = force_families(raw[key], oracle[key], pulled, "source-derived "+key)
        elif key == "constraint_multipliers_n":
            errors[key] = close(raw[key], oracle[key], key, rtol=3e-9, atol=1e-3)
        else:
            errors[key] = close(raw[key], oracle[key], key, rtol=3e-10, atol=2e-13)
    # Exact algebra on the published force components prevents erasing the
    # small accepted numerical residual or adding a tied force a second time.
    force_values = [array(raw[k]) for k in ("bank_inertia_force_n", "bank_drag_force_n",
        "bank_average_internal_force_n", "bank_external_force_n")]
    residual = force_values[0]+force_values[1]+force_values[2]-force_values[3]
    close(raw["bank_residual_n"], residual, "literal reaction force balance", rtol=0, atol=0)
    J, R = surface["topology"]["jump_operator"], surface["topology"]["prolongation"]
    literal = interval_reaction(J, residual)
    close(raw["constraint_multipliers_n"], literal["canonical_reaction_n"], "literal reaction multiplier", rtol=2e-14)
    close(raw["constraint_force_n"], literal["interface_force_n"], "literal reaction dual", rtol=2e-14)
    close(raw["remaining_residual_n"], residual+array(raw["constraint_force_n"]),
          "literal accepted residual retained", rtol=0, atol=0)
    force_families(R.T@array(raw["remaining_residual_n"]), raw["parent_residual_n"],
        [R.T@v for v in force_values], "parent numerical residual pullback")
    require(not np.any(J@array(raw["bank_increment_m"])), "accepted source interval is not exactly tied")
    force = array(raw["constraint_force_n"]); motion = array(raw["bank_increment_m"])
    work_rounding = 256*np.finfo(float).eps*max(float(np.linalg.norm(force)*np.linalg.norm(motion)), 1.)
    require(abs(float(force@motion)) <= work_rounding, "tied reaction does work")
    close(raw["tied_constraint_work_j"], force@motion, "tied reaction reported work", rtol=0, atol=work_rounding)
    require(raw["exact_canonical_jump_work_j"] == 0., "tied canonical jump work differs")
    for key in ("force_pullback_relative_errors_A_P", "constraint_force_null_relative_errors_A_P",
                "remaining_pullback_relative_errors_A_P"):
        diagnostics = array(raw[key], (2,))
        require(np.all(diagnostics >= 0) and np.all(diagnostics < 3e-9), key+" outside admitted tolerance")
    relative = float(np.linalg.norm(oracle["parent_residual_n"]))/max(
        float(np.linalg.norm(surface["topology"]["prolongation"].T@oracle["bank_average_internal_force_n"])),
        float(np.linalg.norm(surface["topology"]["prolongation"].T@oracle["bank_external_force_n"])), 1.)
    equilibrium_tolerance = json.loads(runtime["contact_parameters_json"])["equilibrium_tolerance"]
    require(0 <= raw["recomputed_parent_momentum_relative_residual"] <= equilibrium_tolerance
            and relative <= equilibrium_tolerance, "accepted momentum equation fails")
    return oracle, dict(full_interval_averages_independently_derived=True,
        average_coefficient_relative_error=oracle["worst_average_coefficient_relative_error"],
        independent_parent_momentum_relative_residual=relative,
        published_parent_residual_preserved=True, force_relative_errors=errors)


def audit_traction(raw, reaction, surface, geometry):
    seal(raw, "traction_sha256")
    expected = minimum_area_traction(surface, reaction["constraint_multipliers_n"])
    fixed(raw, dict(format="genesis-accepted-interval-volume-reaction-traction-0.1", scope=REACTION_SCOPE,
        reaction_binding_sha256=reaction["prepared_reaction_binding_sha256"],
        surface_binding_sha256=geometry["binding_sha256"], accepted_parent_checkpoint_sha256=PARENT_CHECKPOINT_SHA256,
        representative="minimum_reference_area_L2_world_traction",
        canonical_trace_semantics="world_surface_trace_equals_H_times_canonical_J",
        modal_scaling="A_one_P_reciprocal_one_minus_minimum_reference_lambda",
        physical_creation_admitted=False, physical_release_admitted=False, pointwise_stress_normal_recovered=False,
        instantaneous_birth_traction_claimed=False, transition_time_increment_myr=0.,
        creation_work_debited_from_Genesis_j=0.), "reaction traction")
    mapping = dict(world_traction_pa="traction_world_pa", canonical_trace="H", modal_scale="modal_scale",
        singular_values="singular_values", rank="rank", area_l2_traction_squared_pa2_m2="physical_l2_norm_squared_pa2_m2")
    errors = {key: close(raw[key], expected[value], "independent minimum-area "+key,
        rtol=3e-9, atol=1e-7) for key, value in mapping.items()}
    weights = np.repeat(surface["area_weights_m2"].ravel(), 3)
    Lshape = (len(surface["topology"]["jump_operator"]), len(weights))
    close(raw["svd_relative_cutoff"], 64*np.finfo(float).eps*max(Lshape), "declared SVD cutoff", rtol=0, atol=0)
    world = array(raw["world_traction_pa"], surface["area_weights_m2"].shape+(3,))
    force = surface_force(surface, world)
    errors["independent_force_A_P"] = force_families(force, reaction["constraint_force_n"],
        [reaction["constraint_force_n"]], "complete unscaled traction replacement")
    close(raw["replacement_force_n"], force, "surface force integration", rtol=5e-11, atol=1e-3)
    close(raw["replacement_residual_n"], array(raw["replacement_force_n"])-array(reaction["constraint_force_n"]),
        "literal traction replacement difference", rtol=0, atol=0)
    probe = np.sin(.117*np.arange(surface["bank_ndof"])+.39)
    trace_work = np.einsum("pqa,pqa,pq->", world, surface_jump(surface, probe), surface["area_weights_m2"])
    close(force@probe, trace_work, "independent traction force virtual work", rtol=5e-11, atol=1e-3)
    for key in ("range_relative_error_A_P", "force_relative_error_A_P"):
        value = array(raw[key], (2,)); require(np.all(value >= 0) and np.all(value < 3e-9), key+" differs")
    motion = array(reaction["bank_increment_m"])
    rounding = 1024*np.finfo(float).eps*max(float(np.linalg.norm(force)*np.linalg.norm(motion)), 1.)
    require(abs(float(force@motion)) <= rounding, "traction replacement performs tied work")
    close(raw["tied_replacement_work_j"], array(raw["replacement_force_n"])@motion,
          "literal tied replacement work", rtol=0, atol=rounding)
    return world, dict(rank=expected["rank"], minimum_physical_area_L2_checked=True,
        full_A_P_work_and_force_checked=True, relative_errors=errors)


ORIGINAL_LAW = dict(fracture_energy_j_m2=1e8, peak_traction_pa=1e8,
    cohesive_stiffness_pa_m=1e10, compression_stiffness_pa_m=2e10)
WORK_LAW = dict(fracture_energy_j_m2=1e8, peak_traction_pa=1e10,
    cohesive_stiffness_pa_m=1e14, compression_stiffness_pa_m=2e14)
LAW_PROVENANCE = "computational_only_after_declared_overstrength_refusal"


def audit_original_refusal(raw, initial_traction):
    positive = array(initial_traction).copy(); positive[..., 0] = np.maximum(positive[..., 0], 0.)
    maximum = float(np.max(np.hypot.reduce(positive, axis=-1), initial=0.))
    nonempty = positive.size > 0
    expected = dict(status="typed_virgin_overstrength_refusal" if nonempty else "empty_surface_no_traction_to_refuse",
        declared_law=ORIGINAL_LAW, declared_peak_traction_pa=ORIGINAL_LAW["peak_traction_pa"],
        exception_type="PrestressCreationRefusal" if nonempty else None,
        original_parameters_retained=True, actual_initial_traction_unchanged=True,
        law_calibrated_from_Genesis=False, physical_creation_admitted=False, physical_release_admitted=False)
    fixed(raw, expected, "original prospective virgin refusal")
    require(set(raw) == set(expected)|{"largest_initial_cohesive_traction_pa", "reason"},
            "whole original prospective refusal schema differs")
    close(raw["largest_initial_cohesive_traction_pa"], maximum,
          "unchanged actual initial cohesive traction maximum", rtol=5e-14, atol=1e-6)
    if nonempty:
        try:
            prestress_seed(ORIGINAL_LAW, initial_traction)
        except ValueError:
            pass
        else:
            raise ValueError("Independent insertion audit: original virgin law was not overstrength")
        require(raw["reason"] == "Prestress creation requires a virgin initial cohesive traction no greater than peak strength",
                "original typed refusal reason differs")
    else:
        require(raw["reason"] is None, "empty surface acquired an overstrength refusal")
    return dict(original_explicit_peak_traction_pa=ORIGINAL_LAW["peak_traction_pa"],
        unchanged_initial_traction_maximum_pa=maximum,
        original_virgin_overstrength_independently_confirmed=nonempty,
        computational_work_law_is_not_source_calibration=True)


def audit_interface_replacement(raw, reaction, surface, initial_world):
    force = surface_force(surface, initial_world)
    h = array(reaction["constraint_force_n"])
    fixed(raw, dict(replaces_constraint_force_once=True, retains_parent_residual=True,
        relative_stored_interface_energy_j=0., fracture_work_j=0., mechanical_work_j=0.,
        existing_Genesis_memory_or_energy_reinitialized=False,
        physical_creation_admitted=False, physical_release_admitted=False), "single seeded insertion")
    force_families(raw["bank_force_n"], force, [h], "complete single seeded replacement")
    force_families(force, h, [h], "seeded force replaces full accepted reaction")
    close(raw["old_tied_residual_n"], reaction["remaining_residual_n"],
          "retained old tied residual", rtol=0, atol=0)
    reported_force = array(raw["bank_force_n"])
    expected_new = array(reaction["bank_residual_n"])+reported_force
    close(raw["new_seeded_residual_n"], expected_new, "single new seeded residual", rtol=0, atol=0)
    close(raw["bulk_plus_seeded_internal_force_n"],
        array(reaction["bank_average_internal_force_n"])+reported_force,
        "bulk plus single seeded internal force", rtol=0, atol=0)
    force_families(raw["new_seeded_residual_n"], reaction["remaining_residual_n"],
                   [h], "single seeded accepted residual retained")
    for key in ("force_replacement_relative_errors_A_P", "retained_residual_relative_errors_A_P"):
        diagnostics = array(raw[key], (2,))
        require(np.all(diagnostics >= 0) and np.all(diagnostics < 3e-9), key+" differs")
    return dict(full_reaction_replaced_once=True, existing_bulk_force_not_doubled=True,
        zero_relative_reference_energy_and_creation_work=True, accepted_parent_residual_retained=True)


def audit_seeded_paths(case, surface, initial_local, parent_checkpoint_sha256):
    require(type(case.get("paths")) is list and len(case["paths"]) > 0,
            "computational path catalogue must be nonempty")
    law, raw_seed = case["law"], case["seed"]
    validate_law(law); seal(raw_seed, "seed_sha256")
    fixed(raw_seed, dict(law=law, kernel_only=True, physical_release_admitted=False,
        normal_contact_admitted=False), "prestress seed")
    weights = surface["area_weights_m2"]
    literal_initial = array(raw_seed["initial_traction_pa"], weights.shape+(3,))
    close(literal_initial, initial_local, "seed unchanged accepted reaction traction", rtol=5e-11, atol=1e-5)
    seed = prestress_seed(law, literal_initial)
    require(set(raw_seed) == set(seed)|{"law", "kernel_only", "physical_release_admitted",
        "normal_contact_admitted", "seed_sha256"}, "whole prestress seed schema differs")
    seed_errors = {key: close(raw_seed[key], value, "source-bound prestress "+key,
        rtol=3e-12, atol=2e-12) for key, value in seed.items()}
    frames = np.stack((surface["edge_normals_xyz"], surface["edge_tangent1_xyz"],
        surface["edge_tangent2_xyz"]), axis=-1)[surface["patch_edge_indices"]]
    previous_motion = np.zeros(surface["bank_ndof"])
    previous_local = np.zeros(weights.shape+(3,))
    previous_kappa = array(raw_seed["initial_maximum_separation_m"], weights.shape)
    total_work, total_fracture = 0., 0.
    paths = []
    for path in case["paths"]:
        old_motion = array(path["old_bank_displacement_from_fixture_origin_m"], previous_motion.shape)
        new_motion = array(path["bank_displacement_from_fixture_origin_m"], previous_motion.shape)
        old_kappa = array(path["old_kappa_m"], weights.shape)
        close(old_motion, previous_motion, "path old motion continuity", rtol=0, atol=0)
        close(old_kappa, previous_kappa, "path irreversible maximum continuity", rtol=0, atol=0)
        world = surface_jump(surface, new_motion)
        local = np.einsum("pqa,pab->pqb", world, frames)
        old_world = surface_jump(surface, old_motion)
        old_local = np.einsum("pqa,pab->pqb", old_world, frames)
        close(previous_local, old_local, "independently reconstructed old shifted gap")
        literal_local = array(path["local_jump_m"], local.shape)
        close(literal_local, local, "independently reconstructed shifted gap")
        response = prestress_segment(law, literal_initial, previous_local, literal_local, old_kappa)
        world_traction = np.einsum("pqb,pab->pqa", response["traction_pa"], frames)
        force = surface_force(surface, world_traction)
        errors = dict(world_jump=close(path["world_jump_m"], world, "world gap"),
            traction_world=close(path["traction_world_pa"], world_traction, "seeded world traction", rtol=2e-10, atol=1e-5))
        for published, computed in (("kappa_m", "maximum_separation_m"), ("damage", "damage"),
                ("traction_local_pa", "traction_pa"), ("tangent_local_pa_m", "tangent_pa_m"),
                ("stored_energy_density_j_m2", "stored_energy_j_m2"),
                ("cumulative_dissipation_density_j_m2", "fracture_dissipation_j_m2"),
                ("dissipation_increment_density_j_m2", "fracture_dissipation_increment_j_m2"),
                ("mechanical_work_increment_density_j_m2", "mechanical_work_j_m2")):
            # Direct traction-path integration may use different arithmetic
            # from endpoint energy subtraction, especially during unloading.
            rounding = 256*np.finfo(float).eps*max(law["fracture_energy_j_m2"],
                float(np.max(seed["reference_compression_energy_j_m2"], initial=0.)), 1.)
            errors[published] = close(path[published], response[computed], published,
                rtol=3e-9, atol=rounding if "density" in published else 1e-8)
        errors["bank_force_A_P"] = force_families(path["bank_force_n"], force, [force], "seeded path force")
        probe = np.cos(.149*np.arange(surface["bank_ndof"])+.21)
        virtual = np.einsum("pqa,pqa,pq->", world_traction, surface_jump(surface, probe), weights)
        close(force@probe, virtual, "seeded path independent virtual work", rtol=5e-11, atol=1e-3)
        integrals = {}
        for published, computed in (("stored_energy_j", "stored_energy_j_m2"),
                ("cumulative_dissipation_j", "fracture_dissipation_j_m2"),
                ("dissipation_increment_j", "fracture_dissipation_increment_j_m2"),
                ("mechanical_work_increment_j", "mechanical_work_j_m2")):
            integral = float(np.sum(weights*response[computed])); integrals[published] = integral
            errors[published] = close(path[published], integral, published, rtol=3e-9, atol=1e-3)
        total_work += integrals["mechanical_work_increment_j"]
        total_fracture += integrals["dissipation_increment_j"]
        paths.append(dict(name=path["name"], direct_traction_path_work_checked=True,
            true_creation_cost_Gc_with_prestress_adjusted_auxiliary_law=True,
            relative_errors=errors, **integrals))
        previous_motion = new_motion
        previous_local = literal_local
        previous_kappa = array(path["kappa_m"], weights.shape)
    terminal = case["terminal_state"]; seal(terminal, "state_sha256")
    expected_terminal = dict(format="genesis-computational-prestress-insertion-history-0.1", scope=SCOPE,
        parent_checkpoint_sha256=parent_checkpoint_sha256,
        surface_binding_sha256=case["geometry"]["binding_sha256"], seed_sha256=raw_seed["seed_sha256"],
        stored_energy_semantics="signed_change_from_prestressed_reference_not_new_historical_energy",
        physical_creation_admitted=False, physical_release_admitted=False, normal_contact_admitted=False,
        accepted_path_segments=[p["bank_displacement_from_fixture_origin_m"] for p in case["paths"]])
    fixed(terminal, expected_terminal, "computational terminal state")
    require(set(terminal) == set(expected_terminal)|{"bank_displacement_from_fixture_origin_m", "local_gap_m",
        "maximum_separation_m", "stored_interface_energy_j", "cumulative_fracture_work_j",
        "cumulative_cohesive_work_j", "state_sha256"}, "whole computational terminal schema differs")
    for key, reference in (("bank_displacement_from_fixture_origin_m", previous_motion),
            ("local_gap_m", previous_local), ("maximum_separation_m", previous_kappa),
            ("stored_interface_energy_j", integrals["stored_energy_j"]),
            ("cumulative_fracture_work_j", total_fracture), ("cumulative_cohesive_work_j", total_work)):
        close(terminal[key], reference, "terminal "+key, rtol=3e-9, atol=1e-3)
    close(total_work, integrals["stored_energy_j"]+total_fracture,
          "global seeded work creation balance", rtol=3e-9, atol=1e-3)
    require(case.get("direct_resumed_computational_state_byte_equal") is True,
            "complete computational restart comparison omitted")
    return dict(seed_relative_errors=seed_errors, paths=paths,
        terminal_whole_schema_seal_and_path_catalogue_checked=True,
        nonlinear_spatial_integration="declared_four_point_surface_quadrature",
        signed_reference_energy_preserved=True, paired_interface_area_counted_once=True)


def audit_case(case, parent):
    require(case.get("computational_fixture") is True, "unmarked physical fixture")
    fixed(case, dict(law=WORK_LAW, law_provenance=LAW_PROVENANCE,
        law_calibrated_from_Genesis=False), "explicit computational work law")
    surface, geometry_errors = audit_surface_claims(case, parent["runtime_state"]["geometry"])
    oracle, reaction_report = audit_reaction(case["reaction"], parent, surface, case["geometry"])
    world, traction_report = audit_traction(case["reaction_traction"], case["reaction"], surface, case["geometry"])
    frames = np.stack((surface["edge_normals_xyz"], surface["edge_tangent1_xyz"],
        surface["edge_tangent2_xyz"]), axis=-1)[surface["patch_edge_indices"]]
    initial_local = np.einsum("pqa,pab->pqb", world, frames)
    refusal = audit_original_refusal(case["original_prospective_virgin_refusal"], initial_local)
    seed_world = np.einsum("pqb,pab->pqa",
        array(case["seed"]["reference_traction_pa"], surface["area_weights_m2"].shape+(3,)), frames)
    replacement = audit_interface_replacement(case["interface_replacement"], case["reaction"], surface, seed_world)
    path_report = audit_seeded_paths(case, surface, initial_local, parent["checkpoint_sha256"])
    return dict(name=case["name"], computational_fixture=True,
        paired_patch_count=len(surface["patch_edge_indices"]),
        paired_interface_area_m2=float(surface["patch_area_m2"].sum()),
        geometry_relative_errors=geometry_errors, reaction=reaction_report, reaction_traction=traction_report,
        original_prospective_virgin_refusal=refusal, single_insertion=replacement, **path_report)


def audit_fixture(fixture, parent, control_directory=None):
    seal(fixture, "fixture_sha256")
    fixed(fixture, dict(format=FORMAT, scope=SCOPE, parent_raw_sha256=PARENT_RAW_SHA256,
        traction_scope="last_accepted_interval_average_not_instantaneous_birth_stress",
        bank_displacement_semantics="from_zero_computational_fixture_origin_not_absolute_accepted_Genesis_pose",
        no_new_Genesis_time_steps=True, no_physical_fracture_or_contact_admission=True,
        no_source_creation_law_calibration_claim=True, parent_residual_preserved_without_forcing_it_to_zero=True,
        original_declared_prospective_law=ORIGINAL_LAW, computational_work_fixture_law=WORK_LAW,
        computational_work_fixture_law_provenance=LAW_PROVENANCE), "top computational scope")
    require(parent["checkpoint_sha256"] == PARENT_CHECKPOINT_SHA256
        and digest({k: v for k, v in parent.items() if k != "checkpoint_sha256"}) == PARENT_CHECKPOINT_SHA256,
        "accepted immutable parent semantic pin differs")
    require(canonical(fixture["parent_checkpoint"]) == canonical(parent), "literal accepted parent changed")
    require(type(fixture.get("fixture_cases")) is list and len(fixture["fixture_cases"]) == 2
        and [c.get("name") for c in fixture["fixture_cases"]] ==
        ["empty_reference_support", "supplied_closed_first_face_reference_cycle"],
        "required empty and nonempty computational cases differ")
    audit_context(fixture["actual_creation_context"], parent, None)
    onset = audit_onset(fixture["actual_onset_prerequisite"], parent)
    cases = [audit_case(case, parent) for case in fixture["fixture_cases"]]
    external_checked = False
    if control_directory is not None:
        before, after = (control_directory/name for name in ("parent_before.json", "parent_after.json"))
        require(before.read_bytes() == after.read_bytes()
            and hashlib.sha256(before.read_bytes()).hexdigest() == PARENT_RAW_SHA256,
            "parent before/after complete archive bytes differ")
        for case in fixture["fixture_cases"]:
            name = case["name"]
            direct = (control_directory/(name+"_direct_terminal.json")).read_bytes()
            resumed = (control_directory/(name+"_resumed_terminal.json")).read_bytes()
            require(direct == resumed and canonical(json.loads(direct)) == canonical(case["terminal_state"]),
                    "complete computational direct/resumed archive bytes differ")
        external_checked = True
    report = dict(format="independent-genesis-volume-prestress-insertion-audit-0.1", status="PASS",
        scope=SCOPE, producer_imports=False, parent_raw_sha256=PARENT_RAW_SHA256,
        parent_checkpoint_sha256=PARENT_CHECKPOINT_SHA256, formation_checkpoint_sha256=FORMATION_SHA256,
        actual_time_myr=parent["runtime_state"]["geometry"]["time_myr"], actual_cohorts=800,
        actual_active_planes=0, actual_creation_law_unknown=True, actual_activation_event=None,
        actual_sharp_surface_selected=False, actual_Genesis_time_advanced=False,
        actual_creation_work_debited_j=0., physical_release_admitted=False, normal_contact_admitted=False,
        historical_damage_or_slip_dissipation_credited_as_creation=False,
        source_calculated_onset=onset, cases=cases,
        external_parent_and_computational_restart_bytes_checked=external_checked,
        opaque_producer_object_identifiers_are_linked_not_reconstructed=True)
    report["report_sha256"] = digest(report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, default=ROOT/
        "analysis/genesis_volume_creation_validation/evidence/actual_continuation/direct_after_second.json")
    parser.add_argument("--control-directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    parent = actual_parent(arguments.parent_checkpoint)
    fixture = json.loads(arguments.fixture.read_bytes())
    report = audit_fixture(fixture, parent, arguments.control_directory)
    arguments.output.write_bytes(canonical(report)+b"\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
