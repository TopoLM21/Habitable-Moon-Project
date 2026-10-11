"""Bounded exact-domain bulk integration of authenticated Genesis birth memory.

This control reads the already publicly reconstructed, sealed Genesis archive.
It does not call a planetary restore, initialize material, select a fracture,
advance accepted time, or execute the private release solver.  The 100 m ball
is a fixed declared computational probe in each original thin cohort.  Every
integration order and refusal is retained; neither radius nor tolerance is
adapted to obtain agreement.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tectonics.genesis_volume_clipped_domain import (
    clipped_domain_to_dict, prepare_cohort_clipped_domain,
)
from tectonics.genesis_volume_clipped_bulk import (
    clipped_bulk_contractions, clipped_bulk_operator_to_dict, clipped_bulk_projected_blocks,
    compare_clipped_bulk_orders,
    prepare_clipped_bulk_operator,
)
from tectonics.genesis_volume_finite_seed import prepare_cohort_finite_seed, volume_finite_seed_to_dict
from tectonics.genesis_volume_material_sections import (
    CohortMaterialSection, ConstantLambdaMaterialSection,
    MaterialSectionInterval, _plane, _validate_cohort_section,
)
from tectonics.genesis_volume_geometry import _chart
from tectonics.genesis_volume_operator import isotropic_elasticity
from tectonics.genesis_volume_release_modes import compact_release_modes_to_dict, resolve_compact_release_modes
from tectonics.genesis_volume_release_work import _analytic_null
from tectonics.genesis_volume_seed_insertion import _surface_enrichment

PARENT = ROOT / "analysis/genesis_volume_creation_validation/evidence/actual_continuation/direct_after_second.json"
PREVIOUS = ROOT / "analysis/genesis_volume_memory_validation/evidence/actual_control/control_fixture.json"
RELEASE_REPORT = ROOT / "analysis/genesis_volume_release_validation/evidence/actual_control/report.json"
MEMORY_READBACK = Path("/workspace/genesis_volume_memory_final_readback_20261010.json")
RELEASE_READBACK = Path("/workspace/genesis_volume_release_final_readback_20261010.json")
PARENT_RAW = "62579a0bad128830a66e2f44579cf2b1f5d3835b691225b8d431102f3e4486d7"
PARENT_SELF = "5d282ceeccc8d76acc38427d08a7e65575fddb509fc4b93115cbe6d318e91349"
FORMATION = "cd79ee2a5139c4b91abdb1b4714a761c9153cc9f4490f5aa63fa2facba458c89"
PREVIOUS_RAW = "a98ff7cc89aeb59eb004c05b8ac87f75f2e39a4f10d4454fdf93c288153c9c0e"
PREVIOUS_SELF = "2ba89795666f61987e513fa78971b9605db0d08bd6d83406eb9ebbe6295587c3"
RELEASE_REPORT_SELF = "3a05d850da0a0cf8b6b0144ac5d67483ce2c6429fed7546ccbddd0fdf8f687bd"
MEMORY_READBACK_SELF = "0eac9f171a3ffe91a7f03b8cff80dfdbf2bfa9b5c379d70a603857dd8450a723"
RELEASE_READBACK_SELF = "4980caa8bcac47ebc12353475adfaef32415008239045dd477dd4e9eb598cf01"
MEMORY_READBACK_RAW = "a9fdd35924ba722ac56d97c5b09288ca174b3062cc6e0233995148541f918c40"
RELEASE_READBACK_RAW = "cd16e45bb3c6e3f853ea330fd203640b615aa2ffab35ec8fbd998287cc8b8796"
FORMAT = "genesis-source-bound-resolved-clipped-bulk-control-0.1"
SCOPE = "declared_fixed_100m_probe_on_original_Genesis_material_no_physical_event"
ORDERS = (4, 8, 12, 16)
BASELINE_ORDERS = (12, 16)
RADIUS_M = 100.0
RELATIVE_TOLERANCE = 2e-6
MAX_POINTS = 250000


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def pin(path):
    sha = hashlib.sha256()
    size = 0
    with Path(path).open("rb") as stream:
        while data := stream.read(1 << 20):
            sha.update(data)
            size += len(data)
    return {"bytes": size, "raw_sha256": sha.hexdigest()}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def seal(body, key):
    return {**body, key: digest(body)}


def check_seal(raw, key, expected=None):
    require(type(raw) is dict and key in raw, "Missing sealed object: " + key)
    wanted = digest({k: v for k, v in raw.items() if k != key})
    require(raw[key] == wanted and (expected is None or wanted == expected),
            "Source archive seal differs: " + key)
    return raw


def readonly(raw):
    value = np.asarray(raw, dtype=float).copy()
    value.setflags(write=False)
    return value


def section_from_archive(raw):
    """Decode a pinned descriptor without selecting or changing its plane."""
    require(set(raw) == {f.name for f in fields(CohortMaterialSection)},
            "Archived section fields differ")
    values = dict(raw)
    for name in ("triangle_m", "displacement_A_m", "slope_P_m", "normal_xyz",
                 "projection_slopes", "projection_offsets"):
        values[name] = readonly(values[name])
    for name in ("q_bounds", "lambda_bounds", "lambda_breakpoints"):
        values[name] = tuple(values[name])
    values["intervals"] = tuple(MaterialSectionInterval(
        tuple(v["lambda_bounds"]), tuple(tuple(p) for p in v["edge_pairs"]), v["slice_kind"])
        for v in values["intervals"])
    values["constant_lambda_sections"] = tuple(ConstantLambdaMaterialSection(**v)
        for v in values["constant_lambda_sections"])
    result = CohortMaterialSection(**values)
    _validate_cohort_section(result)
    return result


def exact_array(a, b, message):
    require(np.array_equal(np.asarray(a), np.asarray(b)), message)


def plain(value):
    if isinstance(value, np.ndarray):
        return np.asarray(value, dtype=float).tolist()
    if isinstance(value, np.generic):
        return float(value) if value.dtype.kind == "f" else value.item()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def projected_blocks_archive(operator, added_basis):
    live = clipped_bulk_projected_blocks(operator, added_basis)
    rounding = {name: (np.finfo(float).eps*np.abs(np.asarray(value, dtype=np.longdouble))).astype(float).tolist()
                for name, value in live.items() if isinstance(value, np.ndarray)}
    names = ("stiffness_cross", "stiffness_enriched", "mass_cross", "mass_enriched", "enrichment_force_n")
    return seal(dict(blocks={name: plain(live[name]) for name in names},
        roundoff_component_bounds={name: plain(live[name+"_budget"]) for name in names},
        added_basis=plain(added_basis), parent_basis=np.eye(18).tolist(), projection_before_Gram=True,
        physical_contraction_certified=False,
        literal_float64_archive_rounding_bounds=rounding,
        archive_rounding_is_arithmetic_diagnostic_only=True,
        live_finite_order_comparisons_include_only_live_operand_budgets=True,
        available_for_creation_funding=False), "archive_sha256")


def prepare_virtual_surface_basis(section, center, radius):
    """Reuse surface/kernel code for a declared probe, without a birth law."""
    seed = prepare_cohort_finite_seed(section, center, radius, quadrature_order=8)
    bubble, trace = _surface_enrichment(seed)
    modes = resolve_compact_release_modes(trace, seed.surface.reference_area_weights_m2,
        seed.binding_sha256, analytic_null_modes=_analytic_null(seed))
    require(modes.rank == 15 and modes.nullity == 3, "Declared virtual surface complete kernel differs")
    weighted = (np.sqrt(seed.surface.reference_area_weights_m2)[:, None, None]*trace).reshape(-1, 18)
    _, singular_values, right = np.linalg.svd(weighted@modes.release_basis, full_matrices=False)
    require(len(singular_values) == 15 and np.isfinite(singular_values).all()
            and np.all(singular_values > 0), "Declared virtual observable directions are unresolved")
    W = modes.release_basis@right.T
    packet = seal(dict(scope="computational_virtual_Gamma_surface_and_basis_not_physical_activation_or_history",
        finite_seed=volume_finite_seed_to_dict(seed), jump_modes=compact_release_modes_to_dict(modes),
        surface_enrichment_bubble=bubble.tolist(), surface_enrichment_trace=trace.tolist(),
        observable_force_basis=W.tolist(), observable_force_singular_values_m=singular_values.tolist(),
        source_section_binding_sha256=section.binding_sha256, seed_radius_m=radius,
        analytic_complete_kernel_tied=True, full_added_coordinate_count=18,
        operative_coordinate_count=15, kernel_coordinate_count=3,
        physical_creation_admitted=False, physical_release_admitted=False,
        source_activation_accepted=False, physical_Gamma_selected=False,
        available_for_creation_funding=False), "virtual_basis_sha256")
    return W, packet


def parent_block(raw, n):
    """Extract the literal old CSR block; this does not assemble a new model."""
    require(raw["shape"][0] == raw["shape"][1] and raw["shape"][0] >= n,
            "Archived parent CSR shape differs")
    data, indices, indptr = [], [], [0]
    for row in range(n):
        for position in range(raw["indptr"][row], raw["indptr"][row + 1]):
            column = raw["indices"][position]
            if column < n:
                data.append(raw["data"][position])
                indices.append(column)
        indptr.append(len(data))
    return dict(shape=[n, n], data=data, indices=indices, indptr=indptr)


def declared_probes():
    """Fixed virtual directions; none is a calculated planetary trajectory."""
    j = 1 + np.arange(18)
    a_indices = np.array((0, 1, 2, 6, 7, 8, 12, 13, 14))
    p_indices = a_indices + 3
    a_only, p_only = np.zeros(18), np.zeros(18)
    a_only[a_indices] = 1e-4*np.sin(.19*j[a_indices])
    p_only[p_indices] = 7e-5*np.cos(.31*j[p_indices])
    return (
        dict(name="mixed_A_P", parent_increment_m=(1e-4*np.sin(.37*j)).tolist(),
             enrichment_increment_m=(2e-4*np.cos(.23*j)).tolist()),
        dict(name="A_only", parent_increment_m=(3e-5*np.cos(.41*j)).tolist(),
             enrichment_increment_m=a_only.tolist()),
        dict(name="P_only", parent_increment_m=(4e-5*np.sin(.29*j)).tolist(),
             enrichment_increment_m=p_only.tolist()),
        dict(name="small_mixed", parent_increment_m=(2e-7*np.sin(.17*j)).tolist(),
             enrichment_increment_m=(3e-7*np.cos(.43*j)).tolist()),
    )


def authenticate_inputs(parent, previous):
    runtime = parent["runtime_state"]
    geometry = runtime["geometry"]
    require(parent["source_checkpoint_sha256"] == FORMATION
            and previous["parent_raw_sha256"] == PARENT_RAW
            and previous["parent_checkpoint_sha256"] == PARENT_SELF
            and previous["formation_checkpoint_sha256"] == FORMATION,
            "Source lineage differs")
    require(geometry["time_myr"] == 0.8684000000000005
            and len(geometry["cohorts"]) == 800
            and len(runtime["accepted_step_sizes_myr"]) == 4
            and not any(runtime["fault_history"]["fault_active"]),
            "Accepted time, material history, or source activation differs")
    require(all(previous[k] is False for k in ("history_reinitialized", "physical_creation_admitted",
            "physical_release_admitted", "physical_Gamma_selected", "funding_admitted",
            "accepted_planet_memory_changed", "actual_dynamic_reaction_replaced", "automated_goal_enabled"))
            and previous["new_Genesis_time_steps"] == previous["new_material_cohorts"] == 0
            and previous["creation_work_debited_from_Genesis_j"] == 0,
            "Inherited physical refusal or material/work accounting differs")
    require(len(previous["cases"]) == 3, "Authenticated memory cases differ")
    require([c["seed_definition"]["original_cohort_id"] for c in previous["cases"]] == [1, 41, 80],
            "Three original cohort identities differ")
    return runtime, geometry


def bind_case(parent, prior):
    """Require the archived local inputs to retain actual material and memory."""
    runtime, geometry = parent["runtime_state"], parent["runtime_state"]["geometry"]
    definition = prior["seed_definition"]
    partition = check_seal(prior["embedded_partition"], "embedded_partition_sha256")
    reaction = check_seal(prior["accepted_interval_reaction"], "seed_reaction_sha256")
    birth = check_seal(prior["source_memory_birth"], "source_memory_birth_sha256")
    memory = check_seal(prior["birth_memory"], "packet_sha256")
    local = partition["local"]
    section = section_from_archive(local["section"])
    index = section.cohort_index
    cohort = geometry["cohorts"][index]
    mechanics = runtime["mechanics"]["state"]
    memory_by_id = {c["id"]: c for c in mechanics["cohorts"]}
    original = memory_by_id[cohort["id"]]
    face = np.asarray(geometry["faces"][cohort["face_index"]], dtype=int)
    vertices = np.asarray(geometry["vertices_m"], dtype=float)[face]
    displacement = np.asarray(runtime["displacement_m"], dtype=float).reshape(-1, 6)[face]
    exact_array(section.triangle_m, vertices, "Section replaced original triangle")
    exact_array(section.displacement_A_m, displacement[:, :3], "Section replaced calculated A")
    exact_array(section.slope_P_m, displacement[:, 3:], "Section replaced calculated P")
    require(section.cohort_id == cohort["id"] == definition["original_cohort_id"]
            and section.face_index == cohort["face_index"] == definition["original_face_index"]
            and section.material_id == cohort["material_id"] == definition["original_material_id"]
            and section.owner_id == cohort["owner_id"] == definition["original_owner_id"]
            and section.q_bounds == (cohort["q_lo"], cohort["live_q_hi"])
            and section.lambda_bounds == (cohort["inner_scale"], cohort["outer_scale"])
            and section.mass_kg == cohort["mass_kg"] and section.volume_m3 == cohort["volume_m3"],
            "Section replaced original birth material or support")
    _, _, coefficient = _chart(vertices, geometry["column_mass_kg"][section.face_index],
                                geometry["density_kg_m3"])
    require(section.conical_mass_coefficient == coefficient,
            "Section replaced the original material mass-coordinate chart")
    normalized_normal, normalized_offset = _plane(definition["normal_xyz"], definition["offset_m"])
    exact_array(section.normal_xyz, normalized_normal, "Archived effective plane normal differs from original normalization")
    require(section.offset_m == normalized_offset,
            "Archived effective plane offset differs from original normalization")
    tangent = np.asarray(parent["source_packet"]["current_geometry"]["face_frames_xyz"][section.face_index])
    Q = np.column_stack((tangent, np.cross(tangent[:, 0], tangent[:, 1])))
    exact_array(local["basis_xyz"], Q, "Inherited source frame differs")
    e0, e1 = np.asarray(original["strain0"]), np.asarray(original["strain1"])
    exact_array(local["memory0"], e0, "Inherited e0 was reinitialized")
    exact_array(local["memory1"], e1, "Inherited e1 was reinitialized")
    exact_array(partition["memory0"], [memory_by_id[c["id"]]["strain0"] for c in geometry["cohorts"]],
                "Full original e0 coverage differs")
    exact_array(partition["memory1"], [memory_by_id[c["id"]]["strain1"] for c in geometry["cohorts"]],
                "Full original e1 coverage differs")
    shell = parent["source_packet"]["source"]["parameters"]["shell"]
    damage = runtime["fault_history"]["damage"][section.face_index]
    factor = shell["residual_stiffness"] + (1-shell["residual_stiffness"])*(1-damage)**2
    C = factor*isotropic_elasticity(shell["young_modulus_pa"], shell["poisson_ratio"])
    exact_array(local["elasticity_pa"], C, "Calculated damaged source elasticity differs")
    require(partition["source_identity"] == geometry["source_identity"] == birth["source_identity"]
            and partition["source_checkpoint_sha256"] == FORMATION
            and birth["accepted_parent_checkpoint_sha256"] == PARENT_SELF
            and birth["formation_checkpoint_sha256"] == FORMATION
            and birth["initial_memory"] == memory
            and memory["state"]["parent_mechanics"] == runtime["mechanics"]
            and birth["accepted_step_sizes_myr"] == runtime["accepted_step_sizes_myr"]
            and reaction["accepted_parent_checkpoint_sha256"] == PARENT_SELF
            and reaction["activation_accepted"] is False
            and reaction["historical_seed_claimed"] is False,
            "Source-bound memory birth or accepted activation differs")
    require(not np.any(memory["state"]["coefficients_m"])
            and not np.any(memory["state"]["displacement_m"])
            and memory["state"]["journal"] == [], "Source-born added field was not zero")
    require(definition["actual_nucleation_candidate"] is None
            and definition["computational_fixture"] is True
            and definition["material_birth_history_replaced"] is False,
            "Declared computational support became a source-selected fracture")
    dofs = np.asarray(partition["selected_parent_dofs"], dtype=int)
    exact_array(dofs, (6*face[:, None]+np.arange(6)).ravel(), "Original source DOFs differ")
    old_blocks = {name: parent_block(partition[name], partition["parent_ndof"])
                  for name in ("stiffness", "mass_matrix", "basal_drag_matrix")}
    source = dict(previous_case_name=prior["name"], previous_seed_definition=definition,
        section=local["section"], elasticity_pa=C.tolist(), basis_xyz=Q.tolist(),
        original_declared_plane=dict(normal_xyz=definition["normal_xyz"], offset_m=definition["offset_m"]),
        original_archived_effective_plane=dict(normal_xyz=local["section"]["normal_xyz"],
                                              offset_m=section.offset_m),
        original_effective_plane_normalization_reproduced_exactly=True,
        plane_repositioned_for_integration=False,
        memory0=e0.tolist(), memory1=e1.tolist(), source_identity=geometry["source_identity"],
        inherited_temperature_k=original["temperature_k"], selected_parent_dofs=dofs.tolist(),
        selected_parent_displacement_m=np.asarray(runtime["displacement_m"])[dofs].tolist(),
        selected_parent_velocity_m_s=np.asarray(runtime["velocity_m_s"])[dofs].tolist(),
        accepted_interval_mean_e0=reaction["interval_average_elastic_constant"][index],
        accepted_interval_mean_e1=reaction["interval_average_elastic_inverse_scale"][index],
        parent_ndof=partition["parent_ndof"], literal_original_parent_blocks=old_blocks,
        original_parent_block_sha256={name: digest(value) for name, value in old_blocks.items()},
        original_parent_mechanics_sha256=digest(runtime["mechanics"]),
        original_source_memory_birth= birth, original_accepted_interval_reaction=reaction,
        previous_local_bulk_blocks={name: local[name] for name in (
            "parent_stiffness", "parent_mass", "stiffness_cross", "stiffness_enriched",
            "mass_cross", "mass_enriched", "enrichment_force_n")},
        source_inputs_checked_directly_against_raw_accepted_state=True,
        old_parent_K_M_D_retained_without_reassembly=True,
        old_800_cohort_memory_retained_exactly=True, source_selected_prospective_law=None,
        actual_nucleation_candidate=None, physical_Gamma_selected=False,
        source_activation_accepted=False, material_birth_history_replaced=False)
    return section, C, Q, e0, e1, source


def integrate_sequence(section, C, Q, e0, e1, center, radius, orders, probes, label, *, old_local):
    W, virtual_basis = prepare_virtual_surface_basis(section, center, radius)
    previous_operator, records, comparisons = None, [], []
    for order in orders:
        domain = prepare_cohort_clipped_domain(section, center, radius,
            radial_order=order, angular_order=order, ray_order=order,
            max_points=MAX_POINTS, require_both_sides=True)
        operator = prepare_clipped_bulk_operator(domain, C, Q, e0, e1)
        exact_array(operator.parent_stiffness, old_local["parent_stiffness"],
                    "Clipped integration changed the old selected full-cohort stiffness")
        exact_array(operator.parent_mass, old_local["parent_mass"],
                    "Clipped integration changed the old selected full-cohort mass")
        contractions = []
        for probe in probes:
            options = {k: probe[k] for k in ("mean_e0", "mean_e1", "mean_z_m") if k in probe}
            contractions.append(dict(name=probe["name"], contractions=clipped_bulk_contractions(operator,
                probe["parent_increment_m"], probe["enrichment_increment_m"], **options)))
        records.append(dict(order=order, domain=clipped_domain_to_dict(domain),
            bulk_operator=clipped_bulk_operator_to_dict(operator, include_points=(order == orders[-1])),
            declared_probe_contractions=contractions,
            declared_projected_blocks=projected_blocks_archive(operator, W),
            original_full_cohort_parent_stiffness_exact=True,
            original_full_cohort_parent_mass_exact=True,
            literal_point_operator_tables_included=(order == orders[-1])))
        if previous_operator is not None:
            comparisons.append(compare_clipped_bulk_orders((previous_operator, operator),
                relative_tolerance=RELATIVE_TOLERANCE, probes=probes, added_basis=W))
        # Coarse live point arrays are released after their neighboring pair.
        # Their literal geometry, blocks, budgets and contractions remain in
        # the already prepared archive record.
        previous_operator = operator
        print(json.dumps(dict(status="ORDER_COMPLETE", label=label, order=order,
            cohort_id=section.cohort_id, seed_radius_m=radius, point_count=domain.point_count)), flush=True)
    pair_results = [comparison["pair_results"][0] for comparison in comparisons]
    comparison = seal(dict(format="genesis-clipped-bulk-control-sequential-pair-comparison-0.1",
        scope="neighbor_live_operators_all_literal_pair_packets_retained_no_convergence_certificate",
        comparison_packets=comparisons, pair_results=pair_results,
        finite_order_agreement=bool(pair_results) and all(p["finite_order_agreement"] for p in pair_results),
        qualification_available=(len(orders) >= 2), relative_tolerance=RELATIVE_TOLERANCE,
        operative_basis_virtual_sha256=virtual_basis["virtual_basis_sha256"],
        full_18_coordinate_components_and_declared_15_coordinate_point_contractions_retained=True,
        physical_contraction_certified=False, available_for_creation_funding=False), "comparison_sha256")
    decay = []
    for left, right in zip(comparison["pair_results"], comparison["pair_results"][1:]):
        ratios = {}
        for name, before in left["components"].items():
            previous_difference = before["maximum_absolute_difference"]
            current_difference = right["components"][name]["maximum_absolute_difference"]
            ratios[name] = dict(previous_maximum_absolute_difference=previous_difference,
                current_maximum_absolute_difference=current_difference,
                difference_ratio=None if current_difference == 0 else previous_difference/current_difference,
                current_difference_exact_zero=(current_difference == 0),
                rigorous_convergence_rate_claimed=False)
        decay.append(dict(previous_low_orders=left["low_orders"], previous_high_orders=left["high_orders"],
            current_low_orders=right["low_orders"], current_high_orders=right["high_orders"],
            block_difference_decay=ratios, scope="diagnostic_finite_order_difference_decay_not_error_certificate"))
    return dict(label=label, seed_radius_m=radius, fixed_orders=list(orders), orders=records,
        virtual_surface_basis=virtual_basis,
        all_order_diagnostic_comparison=comparison,
        required_final_order_pair=None if not pair_results else pair_results[-1],
        diagnostic_convergence_difference_decay=decay,
        required_final_orders=list(orders[-2:]),
        lower_order_agreement_is_diagnostic_only=True, radius_unchanged_for_all_orders=True,
        relative_tolerance_fixed_before_execution=RELATIVE_TOLERANCE,
        highest_order_literal_point_tables_retained=True,
        lower_order_geometry_arrays_and_all_operator_blocks_retained=True)


def run(output, *, smoke_first_case_order4=False):
    output = output.resolve()
    require(output.is_relative_to(Path("/workspace")) and not output.is_relative_to(ROOT),
            "Control output must be fresh and external inside /workspace")
    output.mkdir(exist_ok=False)
    require(pin(PARENT)["raw_sha256"] == PARENT_RAW, "Accepted parent raw pin differs")
    parent_bytes = PARENT.read_bytes()
    parent = check_seal(json.loads(parent_bytes), "checkpoint_sha256", PARENT_SELF)
    require(pin(PREVIOUS)["raw_sha256"] == PREVIOUS_RAW, "Authenticated birth-memory fixture raw pin differs")
    previous = check_seal(json.loads(PREVIOUS.read_bytes()), "fixture_sha256", PREVIOUS_SELF)
    release_report = check_seal(json.loads(RELEASE_REPORT.read_bytes()), "report_sha256", RELEASE_REPORT_SELF)
    require(pin(MEMORY_READBACK)["raw_sha256"] == MEMORY_READBACK_RAW
            and pin(RELEASE_READBACK)["raw_sha256"] == RELEASE_READBACK_RAW,
            "Previously validated final readback raw pin differs")
    memory_readback = check_seal(json.loads(MEMORY_READBACK.read_bytes()), "sha256", MEMORY_READBACK_SELF)
    release_readback = check_seal(json.loads(RELEASE_READBACK.read_bytes()), "sha256", RELEASE_READBACK_SELF)
    require(memory_readback["status"] == release_readback["status"] == release_report["status"] == "PASS",
            "Prior source reconstruction validation is not PASS")
    require(release_report["parent_raw_sha256"] == PARENT_RAW
            and release_report["previous_memory_fixture_pin"]["raw_sha256"] == PREVIOUS_RAW
            and "accepted_Genesis_has_no_new_source_activation" in release_report["physical_refusals"]
            and "no_source_selected_prospective_nucleation_law" in release_report["physical_refusals"],
            "Accepted-source physical refusal was replaced")
    runtime, geometry = authenticate_inputs(parent, previous)
    # The private two-step numerical trajectories in the old portable fixture
    # are already authenticated by its raw pin and sealed readback.  This new
    # integration needs only the literal birth/operator records, so discard
    # parsed copies of the unrelated private trajectory tables to bound RAM.
    prior_cases = [{k: case[k] for k in ("name", "seed_definition", "embedded_partition",
                   "accepted_interval_reaction", "source_memory_birth", "birth_memory")}
                   for case in previous["cases"]]
    del previous
    (output / "parent_before.json").write_bytes(parent_bytes)
    probes = declared_probes()
    if smoke_first_case_order4:
        prior = prior_cases[0]
        section, C, Q, e0, e1, inputs = bind_case(parent, prior)
        primary = integrate_sequence(section, C, Q, e0, e1,
            prior["seed_definition"]["seed_center_world_m"], RADIUS_M, (4,), probes,
            "PRELIMINARY_first_case_order4_API_smoke", old_local=inputs["previous_local_bulk_blocks"])
        (output / "parent_after.json").write_bytes(PARENT.read_bytes())
        require((output / "parent_after.json").read_bytes() == parent_bytes,
                "Source changed during preliminary API smoke")
        smoke = seal(dict(format=FORMAT + "-PRELIMINARY-smoke", scope="PRELIMINARY_API_SMOKE_NO_CONVERGENCE_QUALIFICATION",
            status="PRELIMINARY_COMPLETED_NO_CONVERGENCE_QUALIFICATION",
            name=prior["name"], source_inputs=inputs, declared_probes=list(probes),
            fixed_100m_clipped_support=primary, numerical_convergence_status="NOT_EVALUATED",
            parent_raw_sha256=PARENT_RAW, previous_memory_fixture_raw_sha256=PREVIOUS_RAW,
            public_planetary_restore_count=0, accepted_planet_memory_changed=False,
            physical_release_admitted=False, physical_creation_admitted=False,
            funding_admitted=False, automated_goal_enabled=False,
            declared_radius_m=RADIUS_M, new_Genesis_time_steps=0, new_material_cohorts=0), "smoke_sha256")
        (output / "smoke_fixture.json").write_bytes(canonical(smoke) + b"\n")
        print(json.dumps(dict(status=smoke["status"], fixture=pin(output / "smoke_fixture.json"),
            point_count=len(primary["orders"][0]["domain"]["barycentric"]),
            smoke_sha256=smoke["smoke_sha256"])), flush=True)
        return
    cases, summaries = [], []
    for index, prior in enumerate(prior_cases):
        section, C, Q, e0, e1, inputs = bind_case(parent, prior)
        center = prior["seed_definition"]["seed_center_world_m"]
        case_probes = probes + (dict(name="saved_accepted_interval_mean",
            parent_increment_m=probes[0]["parent_increment_m"],
            enrichment_increment_m=probes[0]["enrichment_increment_m"],
            mean_e0=inputs["accepted_interval_mean_e0"], mean_e1=inputs["accepted_interval_mean_e1"]),)
        primary = integrate_sequence(section, C, Q, e0, e1, center, RADIUS_M,
            ORDERS, case_probes, "fixed_100m_clipped_support",
            old_local=inputs["previous_local_bulk_blocks"])
        baseline = None
        if index == 0:
            baseline = integrate_sequence(section, C, Q, e0, e1, center,
                prior["seed_definition"]["seed_radius_m"], BASELINE_ORDERS, case_probes,
                "original_contained_computational_radius_baseline",
                old_local=inputs["previous_local_bulk_blocks"])
        case = dict(name=prior["name"], source_inputs=inputs, declared_probes=list(case_probes),
            fixed_100m_clipped_support=primary, original_radius_baseline=baseline)
        summary = dict(name=prior["name"], original_cohort_id=section.cohort_id,
            original_face_index=section.face_index, seed_radius_m=RADIUS_M,
            orders=list(ORDERS), point_counts=[len(v["domain"]["barycentric"]) for v in primary["orders"]],
            required_final_orders=list(ORDERS[-2:]),
            finite_order_agreement=primary["required_final_order_pair"]["finite_order_agreement"],
            original_radius_baseline_finite_order_agreement=None if baseline is None
                else baseline["required_final_order_pair"]["finite_order_agreement"])
        summaries.append(summary)
        case_packet = seal(dict(format=FORMAT + "-case", scope=SCOPE, **case), "case_sha256")
        case_path = output / f"case_{index:03d}_cohort_{section.cohort_id}.json"
        case_path.write_bytes(canonical(case_packet) + b"\n")
        cases.append(dict(name=prior["name"], original_cohort_id=section.cohort_id,
            case_file=case_path.name, case_pin=pin(case_path), case_sha256=case_packet["case_sha256"],
            numerical_summary=summary))
        print(json.dumps(dict(status="CASE_ARCHIVED", name=prior["name"], case_file=case_path.name,
            case_pin=cases[-1]["case_pin"], finite_order_agreement=summary["finite_order_agreement"])), flush=True)
        del primary, baseline, case, case_packet, inputs
    (output / "parent_after.json").write_bytes(PARENT.read_bytes())
    require((output / "parent_before.json").read_bytes() == (output / "parent_after.json").read_bytes()
            == parent_bytes and pin(PARENT)["raw_sha256"] == PARENT_RAW
            and pin(PREVIOUS)["raw_sha256"] == PREVIOUS_RAW,
            "Original Genesis or authenticated source archive changed")
    agreement = all(s["finite_order_agreement"] and s["original_radius_baseline_finite_order_agreement"] is not False
                    for s in summaries)
    refusals = [dict(case=s["name"], support="fixed_100m_clipped_support")
                for s in summaries if not s["finite_order_agreement"]]
    refusals += [dict(case=s["name"], support="original_radius_baseline")
                 for s in summaries if s["original_radius_baseline_finite_order_agreement"] is False]
    flags = dict(physical_creation_admitted=False, physical_release_admitted=False,
        physical_contraction_certified=False, physical_force_work_admission=False,
        funding_admitted=False, available_for_creation_funding=False,
        history_reinitialized=False, accepted_planet_memory_changed=False,
        actual_dynamic_reaction_replaced=False, physical_Gamma_selected=False,
        automated_goal_enabled=False, point_one_complete=False,
        new_Genesis_time_steps=0, new_material_cohorts=0,
        creation_work_debited_from_Genesis_j=0.0)
    source_pins = dict(parent=pin(PARENT), previous_memory_fixture=pin(PREVIOUS),
        previous_release_report=pin(RELEASE_REPORT), previous_memory_final_readback=pin(MEMORY_READBACK),
        previous_release_final_readback=pin(RELEASE_READBACK))
    fixture = seal(dict(format=FORMAT, scope=SCOPE, parent_raw_sha256=PARENT_RAW,
        parent_checkpoint_sha256=PARENT_SELF, formation_checkpoint_sha256=FORMATION,
        previous_memory_fixture_sha256=PREVIOUS_SELF, source_pins=source_pins,
        source_time_myr=geometry["time_myr"], cohort_count=800, accepted_interval_count=4,
        public_planetary_restore_count=0,
        source_binding_method="already_publicly_reconstructed_immutable_birth_archive_plus_direct_raw_parent_checks",
        physical_refusals=release_report["physical_refusals"], declared_fixed_radius_m=RADIUS_M,
        declared_orders=list(ORDERS), declared_relative_tolerance=RELATIVE_TOLERANCE,
        declared_max_points_per_order=MAX_POINTS, cases=cases,
        numerical_convergence_status="PASS" if agreement else "OPEN",
        finite_order_numerical_refusals=refusals, **flags), "fixture_sha256")
    (output / "control_fixture.json").write_bytes(canonical(fixture) + b"\n")
    report = seal(dict(format=FORMAT + "-report", scope=SCOPE,
        status="COMPLETED" if agreement else "COMPLETED_WITH_NUMERICAL_REFUSALS",
        structure_and_source_binding_status="PASS", numerical_convergence_status="PASS" if agreement else "OPEN",
        parent_raw_sha256=PARENT_RAW, parent_checkpoint_sha256=PARENT_SELF,
        formation_checkpoint_sha256=FORMATION, fixture_sha256=fixture["fixture_sha256"],
        source_pins=source_pins, public_planetary_restore_count=0,
        source_time_myr=geometry["time_myr"], cohort_count=800, accepted_interval_count=4,
        whole_parent_before_after_byte_identical=True, all_800_birth_memories_preserved=True,
        original_parent_K_M_D_preserved_without_reassembly=True,
        source_activation_none_preserved=True, source_selected_prospective_law=None,
        declared_fixed_radius_m=RADIUS_M, declared_orders=list(ORDERS),
        declared_relative_tolerance=RELATIVE_TOLERANCE, finite_order_numerical_refusals=refusals,
        physical_refusals=release_report["physical_refusals"], cases=summaries, **flags), "report_sha256")
    (output / "report.json").write_bytes(canonical(report) + b"\n")
    print(json.dumps(dict(status=report["status"], numerical_convergence_status=report["numerical_convergence_status"],
        report=pin(output / "report.json"), report_sha256=report["report_sha256"],
        fixture=pin(output / "control_fixture.json"), fixture_sha256=fixture["fixture_sha256"])), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke-first-case-order4", action="store_true",
                        help="Retain a preliminary first-case one-order API smoke without any convergence qualification")
    args = parser.parse_args()
    run(args.output, smoke_first_case_order4=args.smoke_first_case_order4)
