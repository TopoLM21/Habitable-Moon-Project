"""Finite material seeds and tied embedded enrichment on authentic Genesis.

One public restore establishes the unchanged accepted planet.  The explicitly
supplied finite planes and sphere radii are computational probes of its saved
memory, not physical activation, calibrated fracture data or funded release.
"""
from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analysis.genesis_volume_slip_validation.actual_control import audit_state
from tectonics.genesis_volume_creation_law import ProspectiveCreationLaw
from tectonics.genesis_volume_embedded_partition import (
    prepare_volume_embedded_partition, volume_embedded_partition_to_dict,
    evaluate_embedded_partition, evaluate_cohort_enrichment,
    evaluate_embedded_enrichment_trace,
)
from tectonics.genesis_volume_embedded_surface import (
    evaluate_embedded_surface_positions, evaluate_embedded_surface_jump,
    integrate_embedded_surface_traction, integrate_embedded_surface_virtual_work,
)
from tectonics.genesis_volume_finite_seed import (
    prepare_volume_finite_seed, finite_seed_to_dict, finite_seed_from_dict,
)
from tectonics.genesis_volume_fracture_onset import (
    bind_calculated_volume_fracture_onset, calculated_volume_fracture_onset_to_dict,
)
from tectonics.genesis_volume_material_stress import (
    bind_volume_material_stress, volume_material_stress_observer_to_dict,
)
from tectonics.genesis_volume_nucleation import (
    ProspectiveVolumeNucleationLaw, bind_volume_nucleation_proposal,
    volume_nucleation_proposal_to_dict,
)
from tectonics.genesis_volume_operator import evaluate_genesis_volume_operator
from tectonics.genesis_volume_seed_insertion import (
    prepare_volume_seed_insertion, bind_proposed_volume_seed_insertion,
    volume_seed_insertion_to_dict, volume_seed_insertion_from_dict,
)
from tectonics.genesis_volume_slip_checkpoint import (
    load_volume_slip_checkpoint, save_volume_slip_checkpoint,
    volume_slip_checkpoint_to_dict,
)

PARENT_RAW_SHA256 = "62579a0bad128830a66e2f44579cf2b1f5d3835b691225b8d431102f3e4486d7"
PARENT_CHECKPOINT_SHA256 = "5d282ceeccc8d76acc38427d08a7e65575fddb509fc4b93115cbe6d318e91349"
FORMATION_SHA256 = "cd79ee2a5139c4b91abdb1b4714a761c9153cc9f4490f5aa63fa2facba458c89"
FORMAT = "genesis-volume-finite-seed-tied-enrichment-control-0.1"
SCOPE = "source_bound_finite_material_seed_and_tied_bulk_enrichment_not_funded_physical_release"
PREVIOUS_CASE_DIRECTORY = ROOT/"analysis/genesis_volume_location_validation/evidence/actual_control"
QUADRATURE_ORDERS = (4, 8, 12)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if is_dataclass(value):
        return {field.name: plain(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def require(condition, message):
    if not condition:
        raise ValueError(message)


def seal(body, key):
    raw = {name: value for name, value in body.items() if name != key}
    return {**raw, key: digest(canonical(raw))}


def csr_payload(matrix):
    matrix = matrix.tocsr()
    return dict(shape=list(matrix.shape), data=matrix.data.tolist(),
        indices=matrix.indices.tolist(), indptr=matrix.indptr.tolist())


def computational_laws():
    # Explicit prospective inputs.  Historical damage/slip dissipation and
    # contact penalty parameters are not sources of toughness or seed radius.
    creation = ProspectiveCreationLaw(fracture_energy_j_m2=1e6,
        peak_traction_pa=1e8, cohesive_stiffness_pa_m=1e10,
        compression_stiffness_pa_m=1e10)
    law = ProspectiveVolumeNucleationLaw(tensile_strength_pa=12e6,
        shear_cohesion_pa=4e6, friction_coefficient=.6, seed_radius_m=100.,
        creation_law=creation)
    return law, creation


def source_seed_definition(face, state):
    """Retain a prior explicit orientation; locate a new declared finite probe."""
    path = PREVIOUS_CASE_DIRECTORY/f"supplied_current_plane_face_{face}.json"
    data = path.read_bytes()
    prior = json.loads(data)
    require(prior["computational_fixture"] is True
        and prior["physical_surface_selected"] is False
        and prior["physical_creation_admitted"] is False
        and prior["physical_release_admitted"] is False
        and prior["face_indices"] == [face]
        and prior["probe_material_stress_samples"]["parent_checkpoint_sha256"] == PARENT_CHECKPOINT_SHA256,
        "Prior computational normal differs from the accepted parent")
    # The earliest original cohort is selected by its immutable source ID.
    # Its thickness is not enlarged or merged with neighbouring cohorts.
    cohort = min((c for c in state.geometry.cohorts if c.face_index == face), key=lambda c: c.id)
    barycentric = np.full(3, 1/3.)
    radial_scale = .5*(cohort.outer_scale+cohort.inner_scale)
    vertex_ids = np.asarray(state.geometry.faces[face], np.int64)
    pose = np.asarray(state.displacement_m, float).reshape(-1, 6)[vertex_ids]
    triangle = np.asarray(cohort.top_triangle_m, float)
    center = barycentric@(radial_scale*(triangle+pose[:, 3:])+(pose[:, :3]-pose[:, 3:]))
    normal = np.asarray(prior["normal_xyz"], float)
    offset = float(normal@center)
    q = cohort.q_lo+(cohort.live_q_hi-cohort.q_lo)*(
        (cohort.outer_scale**3-radial_scale**3)/
        (cohort.outer_scale**3-cohort.inner_scale**3))
    thickness = float((cohort.outer_scale-cohort.inner_scale)*np.linalg.norm(barycentric@(triangle+pose[:, 3:])))
    radius = .12*thickness
    return dict(original_face_index=face, original_cohort_id=cohort.id,
        original_layer_index=cohort.layer_index, original_owner_id=cohort.owner_id,
        original_material_id=cohort.material_id,
        normal_xyz=normal.tolist(), offset_m=offset,
        seed_center_world_m=center.tolist(), seed_radius_m=radius,
        current_centroid_radial_band_thickness_m=thickness,
        center_original_barycentric=barycentric.tolist(),
        center_original_radial_scale=radial_scale, center_original_mass_coordinate_q=q,
        plane_provenance="prior_explicit_normal_with_new_declared_current_cohort_midpoint_offset",
        radius_provenance="explicit_computational_0p12_fraction_of_current_centroid_band_thickness_not_material_calibration",
        previous_normal_source_pin=dict(path=str(path.relative_to(ROOT)), raw_sha256=digest(data)),
        computational_fixture=True, actual_nucleation_candidate=None,
        material_birth_history_replaced=False, physical_creation_admitted=False)


def force_family_errors(actual, expected):
    actual = np.asarray(actual, float).reshape(-1, 6)
    expected = np.asarray(expected, float).reshape(-1, 6)
    return [float(np.linalg.norm(actual[:, start:start+3]-expected[:, start:start+3])/
        max(np.linalg.norm(expected[:, start:start+3]), 1.)) for start in (0, 3)]


def matrix_family_errors(actual, expected):
    """Compare all original A/P row-column families without an A-scale floor."""
    actual, expected = np.asarray(actual, float), np.asarray(expected, float)
    require(actual.shape == expected.shape == (18, 18),
        "Compact convergence matrices require all eighteen original A/P modes")
    families = {"A": np.asarray([6*v+axis for v in range(3) for axis in range(3)]),
                "P": np.asarray([6*v+3+axis for v in range(3) for axis in range(3)])}
    errors = {}
    for row_name, row in families.items():
        for column_name, column in families.items():
            selection = np.ix_(row, column)
            scale = max(float(np.linalg.norm(expected[selection])), np.finfo(float).tiny)
            errors[row_name+column_name] = float(np.linalg.norm(actual[selection]-expected[selection])/scale)
    return errors


def surface_virtual_work(seed, pullback, displacement):
    """Same-measure duality of the retained original A and P trace components."""
    surface = seed.surface
    index = np.arange(surface.parent_ndof, dtype=float)
    paired = np.stack((1e-5*np.sin(.173*(index+1.)), 1e-5*np.cos(.119*(index+2.))))
    reference_traction = np.asarray(pullback["reference_traction_world_pa"], float)
    birth_traction = np.asarray(pullback["birth_traction_world_pa"], float)
    reference_force = integrate_embedded_surface_traction(surface, reference_traction,
        area_convention="reference")
    birth_force = integrate_embedded_surface_traction(surface, birth_traction,
        area_convention="current")
    errors = force_family_errors(reference_force, pullback["paired_parent_force_n"])
    errors += force_family_errors(birth_force, reference_force)
    require(max(errors) < 3e-12, "Finite seed lost separate original A/P traction forces")
    jump = evaluate_embedded_surface_jump(surface, paired)
    zero = evaluate_embedded_surface_jump(surface, np.stack((paired[0], paired[0])))
    require(not np.any(zero), "Finite seed original parent tying failed exact zero jump")
    reference_work = integrate_embedded_surface_virtual_work(surface, reference_traction,
        paired, area_convention="reference")
    birth_work = integrate_embedded_surface_virtual_work(surface, birth_traction,
        paired, area_convention="current")
    nodal_work = float(np.sum(reference_force*paired))
    scale = max(abs(reference_work), abs(birth_work), abs(nodal_work),
        float(np.sum(np.abs(reference_force*paired))), 1.)
    work_errors = [abs(reference_work-birth_work)/scale, abs(reference_work-nodal_work)/scale]
    require(max(work_errors) < 3e-12, "Finite seed virtual-work metric duality differs")
    positions = evaluate_embedded_surface_positions(surface, displacement)
    position_error = float(np.max(np.linalg.norm(positions-surface.current_position_m, axis=1), initial=0.))
    position_scale = max(float(np.max(np.abs(surface.current_position_m), initial=0.)), 1.)
    require(position_error <= 128*np.finfo(float).eps*position_scale,
        "Finite seed frozen material trace changed its accepted original pose")
    return dict(paired_original_parent_virtual_increments_m=paired.tolist(),
        virtual_jump_m=jump.tolist(), reference_paired_parent_force_n=reference_force.tolist(),
        equivalent_birth_paired_parent_force_n=birth_force.tolist(),
        force_relative_errors_A_P=errors, reference_virtual_work_j=reference_work,
        equivalent_birth_virtual_work_j=birth_work, nodal_virtual_work_j=nodal_work,
        virtual_work_relative_errors=work_errors,
        equal_parent_fields_zero_jump_exact=True,
        frozen_material_pose_reproduction_max_error_m=position_error,
        work_scope="discrete_trace_duality_not_funded_physical_release")


def sparse_exact(actual, expected):
    return actual.shape == expected.shape and (actual-expected).nnz == 0


def frozen_material_follow(seed, displacement):
    """Translate a pure virtual pose beyond the birth sphere without remeshing."""
    increment = np.zeros(seed.surface.parent_ndof)
    translation = np.asarray((1000., -600., 400.))
    increment.reshape(-1, 6)[:, :3] = translation
    followed = evaluate_embedded_surface_positions(seed.surface, np.asarray(displacement)+increment)
    expected = seed.surface.current_position_m+translation
    error = float(np.max(np.abs(followed-expected), initial=0.))
    scale = max(float(np.max(np.abs(expected), initial=0.)), 1.)
    require(error <= 128*np.finfo(float).eps*scale,
        "Finite seed frozen material points failed original parent translation")
    distances = np.linalg.norm(followed-seed.seed_center_world_m, axis=1)
    require(np.all(distances > seed.seed_radius_m),
        "Virtual material follow did not leave the fixed birth sphere")
    return dict(computational_pose_probe=True, runtime_state_changed=False,
        parent_pose_increment_m=increment.tolist(), translation_world_m=translation.tolist(),
        followed_material_position_m=followed.tolist(),
        minimum_distance_to_fixed_birth_center_m=float(distances.min()),
        maximum_original_trace_translation_error_m=error,
        all_followed_points_outside_fixed_birth_sphere=True,
        frozen_barycentric_and_q_retained=True, frozen_birth_enrichment_retained=True,
        world_sphere_or_plane_reintersection_performed=False)


def partition_virtual_work(partition, assembly, basal, endpoint, insertion):
    """Retained parent equality and virtual compact bulk work, with no release."""
    n, local = assembly.ndof, partition.local
    require(partition.ndof == n+18 and partition.parent_ndof == n
        and partition.released_dof_count == 0,
        "Finite embedded enrichment changed original physical degrees of freedom")
    identities = dict(stiffness=sparse_exact(partition.R.T@partition.K@partition.R, assembly.K),
        mass=sparse_exact(partition.R.T@partition.M@partition.R, basal.mass_matrix),
        basal_drag=sparse_exact(partition.R.T@partition.D@partition.R, basal.basal_drag_matrix),
        external_force=np.array_equal(np.asarray(partition.R.T@partition.external_force_n).reshape(n),
            basal.external_force_n),
        endpoint_elastic_force=np.array_equal(np.asarray(partition.R.T@partition.initial_force_n).reshape(n),
            endpoint.generalized_force_n),
        endpoint_elastic_energy=partition.initial_elastic_energy_j == endpoint.elastic_energy_j,
        tie_times_prolongation=(partition.J@partition.R).nnz == 0,
        memory0=np.array_equal(partition.memory0, np.asarray(insertion["embedded_partition"]["memory0"])),
        memory1=np.array_equal(partition.memory1, np.asarray(insertion["embedded_partition"]["memory1"])))
    require(all(identities.values()), "Constrained finite insertion changed a tied original parent operator")
    require(np.trace(local.stiffness_enriched) > 0 and np.trace(local.mass_enriched) > 0
        and set(local.side.tolist()) == {-.5, .5},
        "Localized seed volume quadrature failed positive two-sided enriched energy/inertia")
    q = 1e-5*np.sin(.217*(np.arange(n, dtype=float)+1.))
    a = 2e-5*np.cos(.173*(np.arange(18, dtype=float)+3.))
    tied = evaluate_embedded_partition(partition, q, np.zeros(18))
    original = evaluate_genesis_volume_operator(assembly, q,
        elastic_constant=partition.memory0, elastic_inverse_scale=partition.memory1)
    require(np.array_equal(tied.generalized_force_n[:n], original.generalized_force_n)
        and tied.elastic_energy_j == original.elastic_energy_j
        and np.array_equal(tied.parent_memory0, original.memory0)
        and np.array_equal(tied.parent_memory1, original.memory1),
        "Zero enriched amplitude failed exact original constitutive response")
    response = evaluate_embedded_partition(partition, q, a)
    delta = np.r_[q, a]
    midpoint_force = partition.initial_force_n+.5*(partition.K@delta)
    midpoint_work = float(midpoint_force@delta)
    quadratic_work = response.finite_increment_bulk_work_j
    work_scale = max(abs(midpoint_work), abs(quadratic_work),
        float(np.sum(np.abs(midpoint_force*delta))), 1.)
    algebraic_error = abs(midpoint_work-quadratic_work)/work_scale
    require(algebraic_error < 3e-12, "Finite compact bulk force/quadratic-work duality differs")
    energy_change = response.elastic_energy_j-partition.initial_elastic_energy_j
    energy_error = abs(energy_change-quadratic_work)
    rounding_allowance = 256*np.finfo(float).eps*max(abs(response.elastic_energy_j),
        abs(partition.initial_elastic_energy_j), abs(quadratic_work), 1.)
    require(energy_error <= rounding_allowance,
        "Finite compact virtual energy differs beyond complete parent energy rounding")
    expected_force = np.array(original.generalized_force_n, copy=True)
    np.add.at(expected_force, partition.selected_parent_dofs, local.stiffness_cross@a)
    expected_added = local.enrichment_force_n+local.stiffness_cross.T@q[partition.selected_parent_dofs]+local.stiffness_enriched@a
    require(np.array_equal(response.generalized_force_n[:n], expected_force)
        and np.array_equal(response.generalized_force_n[n:], expected_added)
        and np.array_equal(response.static_tie_reaction_n, -expected_added),
        "Finite compact response lost its actual static enrichment gradient")
    surface_force = np.asarray(insertion["surface_enrichment_force_n"], float)
    reaction = -local.enrichment_force_n
    differences = reaction-surface_force
    require(np.array_equal(reaction, np.asarray(insertion["endpoint_elastic_constraint_reaction_n"]))
        and np.array_equal(differences, np.asarray(insertion["constraint_minus_surface_force_n"])),
        "Source-bound insertion changed its retained endpoint/surface reaction difference")
    trace = evaluate_embedded_enrichment_trace(partition,
        np.asarray(insertion["finite_seed"]["surface"]["barycentric"]),
        np.asarray(insertion["finite_seed"]["surface"]["radial_scale"]))
    saved_trace = np.asarray(insertion["surface_enrichment_trace"], float)
    tolerance = 128*np.finfo(float).eps*np.maximum(np.abs(trace), 1.)
    require(np.all(np.abs(trace-saved_trace) <= tolerance),
        "Source surface jump trace differs from the same compact bulk enrichment")
    section = local.section
    lower, upper = section.lambda_bounds
    mid = .5*(lower+upper)
    definitions = (("inner_original_cohort_boundary", np.full(3, 1/3.), lower),
        ("outer_original_cohort_boundary", np.full(3, 1/3.), upper),
        ("original_angular_boundary", np.asarray((0., .5, .5)), mid),
        ("outside_birth_ball_inside_original_cohort", np.asarray((.34, .33, .33)), mid))
    support_probes = []
    for name, barycentric, lam in definitions:
        data = evaluate_cohort_enrichment(section, local.seed_center_world_m,
            local.seed_radius_m, local.basis_xyz, barycentric, lam)
        require(data["bubble"] == 0. and not np.any(data["enrichment_trace"])
            and not np.any(data["jump_trace"]),
            "Compact enrichment leaked outside its finite original material support")
        support_probes.append(dict(name=name, barycentric=barycentric.tolist(),
            radial_scale=lam, birth_position_m=data["birth_position_m"].tolist(),
            bubble=data["bubble"], enrichment_trace=data["enrichment_trace"].tolist(),
            jump_trace=data["jump_trace"].tolist()))
    return dict(tied_parent_identities_exact=identities,
        tied_original_response_exact=True,
        parent_virtual_increment_m=q.tolist(), enrichment_virtual_amplitude_m=a.tolist(),
        all_enrichment_amplitudes_in_actual_state_zero=True,
        enriched_probe_is_computational_not_accepted_material_history=True,
        virtual_response=plain(response), midpoint_generalized_force_n=midpoint_force.tolist(),
        midpoint_virtual_work_j=midpoint_work, quadratic_bulk_virtual_work_j=quadratic_work,
        force_work_relative_error=algebraic_error,
        virtual_complete_parent_energy_change_j=energy_change,
        complete_parent_energy_subtraction_error_j=energy_error,
        complete_parent_energy_rounding_allowance_j=rounding_allowance,
        static_endpoint_bulk_enrichment_gradient_n=local.enrichment_force_n.tolist(),
        static_endpoint_elastic_constraint_reaction_n=reaction.tolist(),
        surface_jump_generalized_force_n=surface_force.tolist(),
        constraint_minus_surface_force_n=differences.tolist(),
        constraint_surface_force_difference_norms_A_P=[float(np.linalg.norm(differences.reshape(3, 6)[:, start:start+3]))
            for start in (0, 3)],
        surface_force_is_discrete_bulk_reaction=False,
        static_endpoint_reaction_is_accepted_dynamic_birth_reaction=False,
        original_material_support_zero_trace_probes=support_probes,
        added_modes_released=False, funding_admitted=False)


def clipped_large_surface_probe(definition, state):
    """A separate 100 m clipped thin-sheet proof, without bulk accuracy claims."""
    normal, center = np.asarray(definition["normal_xyz"]), np.asarray(definition["seed_center_world_m"])
    rows, archived = [], None
    for order in QUADRATURE_ORDERS:
        seed = prepare_volume_finite_seed(state.geometry, state.displacement_m,
            normal, definition["offset_m"], center, 100.,
            cohort_id=definition["original_cohort_id"], quadrature_order=order)
        require(seed.surface.point_count > 0 and seed.surface.total_current_area_m2 < .95*np.pi*100.**2,
            "100 m thin-cohort probe failed to demonstrate finite original-cohort clipping")
        rows.append(dict(quadrature_order=order, surface_point_count=seed.surface.point_count,
            total_reference_area_m2=seed.surface.total_reference_area_m2,
            total_current_area_m2=seed.surface.total_current_area_m2))
        if order == 8:
            archived = finite_seed_to_dict(seed)
    errors = [abs(rows[1][key]-rows[-1][key])/max(abs(rows[-1][key]), 1.)
        for key in ("total_reference_area_m2", "total_current_area_m2")]
    require(max(errors) <= 2e-6,
        "100 m finite original-cohort surface lacks order-eight/twelve area agreement")
    return dict(name="separate_100_m_original_thin_cohort_surface_only_probe",
        computational_fixture=True, seed_radius_m=100., finite_seed=archived,
        quadrature_order_records=rows, order_eight_twelve_area_relative_errors=errors,
        unclipped_current_disk_area_m2=float(np.pi*100.**2),
        current_disk_clipped_by_original_cohort=True,
        volume_enrichment_or_operator_built=False,
        bulk_quadrature_convergence_claimed=False, physical_activation_admitted=False,
        physical_creation_admitted=False, physical_release_admitted=False,
        funding_admitted=False, creation_work_debited_from_Genesis_j=0.)


def finite_seed_case(definition, model, state, observer, assembly, basal, endpoint, creation_law,
                     *, diagnostic_directory=None):
    normal = np.asarray(definition["normal_xyz"], float)
    offset = definition["offset_m"]
    center = np.asarray(definition["seed_center_world_m"], float)
    radius, cohort_id = definition["seed_radius_m"], definition["original_cohort_id"]
    records, archived_seed, archived_insertion, archived_work = [], None, None, None
    for order in QUADRATURE_ORDERS:
        seed = prepare_volume_finite_seed(state.geometry, state.displacement_m, normal, offset,
            center, radius, cohort_id=cohort_id, quadrature_order=order)
        require(seed.surface.parent_ndof == 252 and seed.surface.point_count > 0
            and seed.surface.selected_cohort_ids == (cohort_id,),
            "Finite seed lost its positive area in exactly one original cohort")
        if order == 8:
            insertion_value = prepare_volume_seed_insertion(model, state, seed,
                observer=observer, creation_law=creation_law, partition_quadrature_order=order)
            insertion = volume_seed_insertion_to_dict(insertion_value)
            partition = prepare_volume_embedded_partition(assembly, basal, state.displacement_m,
                normal, offset, cohort_id=cohort_id, seed_center_m=center, seed_radius_m=radius,
                elastic_constant=observer.elastic_constant,
                elastic_inverse_scale=observer.elastic_inverse_scale, quadrature_order=order)
            require(canonical(volume_embedded_partition_to_dict(partition)) == canonical(insertion["embedded_partition"]),
                "Source-bound insertion differs from the same saved-memory bulk enrichment")
            archived_seed, archived_insertion = finite_seed_to_dict(seed), insertion
            # Full trusted-source roundtrip once per actual case.  This does
            # not restore or rerun the original planetary checkpoint.
            finite_seed_from_dict(archived_seed, geometry=state.geometry,
                displacement_m=state.displacement_m, normal_xyz=normal, offset_m=offset,
                seed_center_world_m=center, seed_radius_m=radius, cohort_id=cohort_id,
                quadrature_order=order)
            volume_seed_insertion_from_dict(insertion, model=model, state=state,
                finite_seed=seed, observer=observer, creation_law=creation_law,
                partition_quadrature_order=order)
            surface_work = surface_virtual_work(seed, insertion["prestress_pullback"], state.displacement_m)
            bulk_work = partition_virtual_work(partition, assembly, basal, endpoint, insertion)
            follow = frozen_material_follow(seed, state.displacement_m)
            archived_work = dict(surface=surface_work, compact_bulk=bulk_work, frozen_material_follow=follow)
        else:
            partition = prepare_volume_embedded_partition(assembly, basal, state.displacement_m,
                normal, offset, cohort_id=cohort_id, seed_center_m=center, seed_radius_m=radius,
                elastic_constant=observer.elastic_constant,
                elastic_inverse_scale=observer.elastic_inverse_scale, quadrature_order=order)
        local = partition.local
        require(np.trace(local.stiffness_enriched) > 0 and np.trace(local.mass_enriched) > 0
            and set(local.side.tolist()) == {-.5, .5}
            and len(local.barycentric) == 4*order**3,
            "Finite sphere-local quadrature failed positive support on both sides")
        records.append(dict(quadrature_order=order, surface_point_count=seed.surface.point_count,
            volume_point_count=len(local.barycentric),
            side_point_counts=[int(np.sum(local.side == side)) for side in (-.5, .5)],
            total_reference_area_m2=seed.surface.total_reference_area_m2,
            total_current_area_m2=seed.surface.total_current_area_m2,
            sampled_reference_support_volume_m3=float(np.sum(local.reference_volume_weights_m3)),
            sampled_birth_support_volume_m3=float(np.sum(local.birth_volume_weights_m3)),
            stiffness_cross=local.stiffness_cross.tolist(), stiffness_enriched=local.stiffness_enriched.tolist(),
            mass_cross=local.mass_cross.tolist(), mass_enriched=local.mass_enriched.tolist(),
            static_endpoint_bulk_enrichment_gradient_n=local.enrichment_force_n.tolist(),
            numerical_enrichment_trace_rank=local.numerical_enrichment_trace_rank,
            scaled_augmented_stiffness_minimum_eigenvalue=local.scaled_augmented_stiffness_minimum_eigenvalue,
            geometrical_integrals_claimed_exact=False, physical_release_admitted=False))
    require(archived_seed is not None and archived_insertion is not None and archived_work is not None,
        "Finite seed order-eight source-bound archive is absent")
    comparisons = []
    final = records[-1]
    for row in records[:-1]:
        area_errors = [abs(row[k]-final[k])/max(abs(final[k]), 1.) for k in
            ("total_reference_area_m2", "total_current_area_m2")]
        matrix_errors = {key: float(np.linalg.norm(np.asarray(row[key])-np.asarray(final[key]))/
            max(np.linalg.norm(np.asarray(final[key])), 1.)) for key in
            ("stiffness_cross", "stiffness_enriched", "mass_cross", "mass_enriched")}
        matrix_families = {key: matrix_family_errors(row[key], final[key]) for key in
            ("stiffness_cross", "stiffness_enriched", "mass_cross", "mass_enriched")}
        force_errors = force_family_errors(row["static_endpoint_bulk_enrichment_gradient_n"],
            final["static_endpoint_bulk_enrichment_gradient_n"])
        comparisons.append(dict(quadrature_order=row["quadrature_order"], comparison_order=12,
            area_relative_errors_reference_current=area_errors,
            bulk_matrix_relative_errors=matrix_errors,
            bulk_matrix_relative_errors_AA_AP_PA_PP=matrix_families,
            endpoint_enrichment_force_relative_errors_A_P=force_errors,
            finite_order_comparison_only=True, certified_continuum_error_bound=False))
    agreement = comparisons[1]
    agreement_errors = (agreement["area_relative_errors_reference_current"]+
        list(agreement["bulk_matrix_relative_errors"].values())+
        [error for families in agreement["bulk_matrix_relative_errors_AA_AP_PA_PP"].values()
            for error in families.values()]+
        agreement["endpoint_enrichment_force_relative_errors_A_P"])
    maximum_agreement_error = max(agreement_errors)
    if diagnostic_directory is not None:
        diagnostic = seal(dict(format="genesis-finite-seed-quadrature-agreement-diagnostic-0.1",
            status="PASS" if maximum_agreement_error <= 2e-6 else "FAIL",
            scope="computational_finite_seed_order_comparison_without_accepted_history_change",
            seed_definition=definition, quadrature_convergence_records=records,
            quadrature_convergence_comparisons=comparisons,
            maximum_order_eight_twelve_relative_agreement_error=maximum_agreement_error,
            intended_relative_agreement_tolerance=2e-6,
            archived_order=8, comparison_order=12,
            computational_fixture=True, certified_continuum_error_bound=False,
            physical_creation_admitted=False, physical_release_admitted=False,
            funding_admitted=False, material_memory_changed=False,
            history_reinitialized=False, new_Genesis_time_steps=0,
            creation_work_debited_from_Genesis_j=0.), "diagnostic_sha256")
        diagnostic_path = Path(diagnostic_directory)/(
            f"convergence_diagnostic_face_{definition['original_face_index']}_cohort_{cohort_id}.json")
        diagnostic_path.write_bytes(canonical(diagnostic)+b"\n")
    require(maximum_agreement_error <= 2e-6,
        "Contained-sphere finite seed lacks order-eight/twelve area/bulk work agreement")
    large_probe = clipped_large_surface_probe(definition, state)
    case = dict(name=f"finite_computational_seed_cohort_{cohort_id}_face_{definition['original_face_index']}",
        computational_fixture=True, seed_definition=definition,
        finite_seed=archived_seed, source_bound_insertion=archived_insertion,
        virtual_work=archived_work, quadrature_convergence_records=records,
        quadrature_convergence_comparisons=comparisons,
        order_eight_twelve_observed_relative_agreement_tolerance=2e-6,
        order_eight_twelve_observed_agreement_pass=True,
        separate_100_m_clipped_surface_only_probe=large_probe,
        finite_positive_material_surface_area_prepared=True,
        surrounding_ligament_retained=True, whole_cohort_disconnected_banks_created=False,
        source_activation_event=None, physical_creation_admitted=False,
        physical_release_admitted=False, physical_contact_admitted=False,
        funding_admitted=False, original_material_memory_changed=False,
        actual_dynamic_reaction_replaced=False,
        actual_creation_work_debited_from_Genesis_j=0.)
    summary = dict(name=case["name"], original_face_index=definition["original_face_index"],
        original_cohort_id=cohort_id, retained_original_cohorts=1, seed_radius_m=radius,
        quadrature_order=8, surface_points=records[1]["surface_point_count"],
        volume_points=records[1]["volume_point_count"],
        reference_area_m2=records[1]["total_reference_area_m2"],
        current_area_m2=records[1]["total_current_area_m2"],
        numerical_enrichment_trace_rank=records[1]["numerical_enrichment_trace_rank"],
        positive_enriched_stiffness_and_mass_verified=True, tied_parent_K_M_D_F_exact=True,
        force_work_relative_error=archived_work["compact_bulk"]["force_work_relative_error"],
        quadrature_order_comparisons=comparisons,
        order_eight_twelve_observed_agreement_pass=True,
        separate_100_m_surface_reference_area_m2=large_probe["finite_seed"]["surface"]["total_reference_area_m2"],
        separate_100_m_surface_current_area_m2=large_probe["finite_seed"]["surface"]["total_current_area_m2"],
        exact_continuum_integral_or_funded_release_claimed=False,
        full_separation_cost_computational_j=archived_insertion["prestress_pullback"]["full_separation_cost_on_explicit_surface_j"],
        actual_creation_work_debited_from_Genesis_j=0.)
    return case, summary


def run(parent_path, output):
    output.mkdir(parents=True, exist_ok=False)
    parent_bytes = parent_path.read_bytes()
    require(digest(parent_bytes) == PARENT_RAW_SHA256,
        "Finite seed control requires the sealed authentic accepted parent")
    parent_packet = json.loads(parent_bytes)
    require(parent_packet["checkpoint_sha256"] == PARENT_CHECKPOINT_SHA256,
        "Finite seed semantic parent pin differs")
    # Exactly one public restore.  All descriptors below are reconstructed
    # against this runtime; none calls another public planetary restore.
    model, state = load_volume_slip_checkpoint(parent_path,
        expected_source_checkpoint_sha256=FORMATION_SHA256)
    require(len(state.geometry.cohorts) == 800 and abs(state.time_myr-.8684) < 1e-14
        and len(state.accepted_step_sizes_myr) == 4,
        "Finite seed actual time, material ancestry or interval count differs")
    require(canonical(volume_slip_checkpoint_to_dict(model, state)) == canonical(parent_packet),
        "Complete accepted finite seed parent replay changed the archive")
    save_volume_slip_checkpoint(output/"parent_before.json", model, state)
    require((output/"parent_before.json").read_bytes() == parent_bytes,
        "Restored finite seed parent is not byte-identical")
    actual_audit = audit_state(model, state)
    observer = bind_volume_material_stress(model, state)
    observer_payload = volume_material_stress_observer_to_dict(observer)
    onset_payload = calculated_volume_fracture_onset_to_dict(
        bind_calculated_volume_fracture_onset(model, state))
    require(onset_payload["new_activation_event"] is None
        and onset_payload["newly_activating_faces"] == []
        and onset_payload["existing_active_faces"] == [],
        "Actual parent unexpectedly acquired a source activation event")
    law, creation_law = computational_laws()
    proposals, gated = {}, {}
    for name, declared in (("unconfigured", None), ("explicit_computational_law", law)):
        raw = volume_nucleation_proposal_to_dict(bind_volume_nucleation_proposal(
            model, state, declared, observer=observer))
        require(raw["status"] == "no_calculated_activation_event"
            and raw["newly_activating_faces"] == [] and raw["candidate"] is None
            and raw["selected_sharp_material_surface"] is None
            and raw["funding_admitted"] is False,
            "Explicit finite seed parameters bypassed the inherited birth gate")
        proposals[name] = raw
        insertion = volume_seed_insertion_to_dict(bind_proposed_volume_seed_insertion(
            model, state, declared, quadrature_order=8, partition_quadrature_order=8))
        require(insertion["status"] == "no_calculated_activation_event"
            and insertion["insertion"] is None
            and insertion["finite_positive_surface_area_prepared"] is False
            and insertion["constrained_bulk_modes_prepared"] is False
            and insertion["released_physical_dofs"] == 0,
            "Prospective source insertion invented finite material support on a noevent parent")
        gated[name] = insertion
        (output/f"nucleation_{name}.json").write_bytes(canonical(raw)+b"\n")
        (output/f"proposed_insertion_{name}.json").write_bytes(canonical(insertion)+b"\n")
    (output/"actual_onset_prerequisite.json").write_bytes(canonical(onset_payload)+b"\n")
    (output/"material_stress_observer.json").write_bytes(canonical(observer_payload)+b"\n")
    assembly, basal = model._assembly(state.geometry, state.fault_history.damage)
    endpoint = evaluate_genesis_volume_operator(assembly, np.zeros(assembly.ndof),
        elastic_constant=observer.elastic_constant,
        elastic_inverse_scale=observer.elastic_inverse_scale)
    parent_operator = dict(parent_ndof=assembly.ndof, cohort_ids=assembly.cohort_ids.tolist(),
        stiffness=csr_payload(assembly.stiffness), mass=csr_payload(basal.mass_matrix),
        basal_drag=csr_payload(basal.basal_drag_matrix), external_force_n=basal.external_force_n.tolist(),
        saved_endpoint_elastic_force_n=endpoint.generalized_force_n.tolist(),
        saved_endpoint_elastic_energy_j=endpoint.elastic_energy_j,
        endpoint_force_scope="saved_endpoint_memory_with_zero_increment_not_absolute_pose_added_again")
    (output/"parent_operator_reference.json").write_bytes(canonical(parent_operator)+b"\n")
    cases, summaries = [], []
    for face in (0, len(state.geometry.faces)//2, len(state.geometry.faces)-1):
        definition = source_seed_definition(face, state)
        case, summary = finite_seed_case(definition, model, state, observer,
            assembly, basal, endpoint, creation_law, diagnostic_directory=output)
        cases.append(case)
        summaries.append(summary)
        (output/f"{case['name']}.json").write_bytes(canonical(case)+b"\n")
    fixture = seal(dict(format=FORMAT, scope=SCOPE, parent_raw_sha256=PARENT_RAW_SHA256,
        parent_path=str(parent_path.resolve()), parent_checkpoint_sha256=PARENT_CHECKPOINT_SHA256,
        actual_onset_prerequisite=onset_payload, material_stress_observer=observer_payload,
        nucleation_unconfigured=proposals["unconfigured"],
        nucleation_explicit_computational_law=proposals["explicit_computational_law"],
        proposed_insertion_unconfigured=gated["unconfigured"],
        proposed_insertion_explicit_computational_law=gated["explicit_computational_law"],
        parent_operator_reference=parent_operator, finite_seed_cases=cases,
        original_parent_ndof=252, original_cohort_count=800,
        stress_scope="current_accepted_endpoint_full_cohort_elastic_memory_not_last_interval_average",
        stress_representation="inherited_fixed_frame_small_strain_reference_energy_stress_not_finite_Cauchy_pushforward",
        material_chart="x(N,lambda)=N@(lambda*(X+P)+(A-P))",
        constrained_compact_enrichment_modes_per_probe=18,
        computational_finite_seed_clipping_performed=True,
        physical_Gamma_selected_on_actual_source=False, actual_activation_accepted=False,
        new_Genesis_time_steps=0, new_material_cohorts=0,
        creation_work_debited_from_Genesis_j=0., physical_creation_admitted=False,
        physical_release_admitted=False, physical_contact_admitted=False,
        funding_admitted=False, actual_dynamic_reaction_replaced=False,
        actual_birth_energy_balance_accepted=False, material_memory_changed=False,
        physical_dofs_created_or_released=False, history_reinitialized=False,
        point_one_complete=False), "fixture_sha256")
    (output/"control_fixture.json").write_bytes(canonical(fixture)+b"\n")
    save_volume_slip_checkpoint(output/"parent_after.json", model, state)
    require((output/"parent_after.json").read_bytes() == parent_bytes
        and parent_path.read_bytes() == parent_bytes,
        "Finite seed computational insertion changed the accepted planet")
    report = seal(dict(format="actual-genesis-volume-finite-seed-control-0.1", status="PASS",
        scope=SCOPE, parent_raw_sha256=PARENT_RAW_SHA256,
        parent_checkpoint_sha256=PARENT_CHECKPOINT_SHA256,
        formation_checkpoint_sha256=FORMATION_SHA256, time_myr=state.time_myr,
        cohort_count=len(state.geometry.cohorts), accepted_Genesis_intervals_replayed=4,
        parent_replayed_once_from_formation_checkpoint=True,
        parent_before_after_entire_packets_byte_equal=True, actual_state_audit=actual_audit,
        fixture_sha256=fixture["fixture_sha256"], actual_new_activation_event=None,
        actual_active_weak_plane_count=0, actual_unconfigured_insertion_status=gated["unconfigured"]["status"],
        actual_declared_insertion_status=gated["explicit_computational_law"]["status"],
        actual_inherited_activation_gate_retained=True, actual_physical_nucleation_candidate=None,
        explicit_new_constitutive_examples_calibrated=False, source_toughness_recovered=False,
        finite_seed_cases=summaries, original_parent_ndof=252,
        extra_computational_modes_per_probe=18, extra_computational_modes_constrained_to_zero=True,
        new_Genesis_time_steps=0, new_material_cohorts=0, no_replacement_initialization=True,
        actual_creation_work_debited_j=0., sharp_physical_banks_created=False,
        ties_released=False, actual_dynamic_reaction_replaced=False,
        actual_birth_energy_balance_accepted=False, funding_admitted=False,
        physical_banks_contact_underthrust_admitted=False, point_one_complete=False), "report_sha256")
    (output/"control_report.json").write_bytes(canonical(report)+b"\n")
    print(json.dumps(report, sort_keys=True))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-checkpoint", type=Path, default=ROOT/
        "analysis/genesis_volume_creation_validation/evidence/actual_continuation/direct_after_second.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.parent_checkpoint, args.output)


if __name__ == "__main__":
    main()
