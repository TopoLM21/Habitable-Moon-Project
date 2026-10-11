"""Compact memory and accepted-interval reaction on unchanged genuine Genesis.

One public source replay supplies the accepted planet. Nonzero enrichment
evolves only in private fixed-material constitutive probes and their journals.
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

from tectonics.genesis_volume_dynamics import coefficient_arrays
from tectonics.genesis_volume_embedded_partition import (
    prepare_embedded_partition, volume_embedded_partition_to_dict)
from tectonics.genesis_volume_enriched_memory import (
    prepare_enriched_loading, evaluate_enriched_loading, accept_enriched_loading,
    enriched_memory_to_dict)
from tectonics.genesis_volume_finite_seed import prepare_volume_finite_seed
from tectonics.genesis_volume_operator import evaluate_genesis_volume_operator
from tectonics.genesis_volume_seed_memory import (
    prepare_source_seed_memory, source_seed_memory_to_dict, source_seed_memory_loading,
    restart_source_seed_memory)
from tectonics.genesis_volume_seed_reaction import (
    prepare_volume_seed_reaction, volume_seed_reaction_to_dict,
    volume_seed_reaction_from_dict)
from tectonics.genesis_volume_slip_checkpoint import (
    load_volume_slip_checkpoint, save_volume_slip_checkpoint,
    volume_slip_checkpoint_to_dict)

PARENT_RAW = "62579a0bad128830a66e2f44579cf2b1f5d3835b691225b8d431102f3e4486d7"
PARENT_SELF = "5d282ceeccc8d76acc38427d08a7e65575fddb509fc4b93115cbe6d318e91349"
FORMATION = "cd79ee2a5139c4b91abdb1b4714a761c9153cc9f4490f5aa63fa2facba458c89"
PREVIOUS = ROOT / "analysis/genesis_volume_seed_validation/evidence/actual_control/control_fixture.json"
PREVIOUS_RAW = "9c9a669639bb8d7c08a15266481f883894c86227a26a6cc9d2bd0759e51509bb"
FORMAT = "genesis-source-bound-enriched-memory-and-full-tied-reaction-control-0.1"
SCOPE = "private_fixed_material_Maxwell_and_hypothetical_accepted_interval_tied_reaction"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()


def pin(path):
    data = Path(path).read_bytes()
    return dict(bytes=len(data), raw_sha256=hashlib.sha256(data).hexdigest())


def plain(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if is_dataclass(value): return {f.name: plain(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, dict): return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [plain(v) for v in value]
    return value


def seal(body, key):
    return {**body, key: hashlib.sha256(canonical(body)).hexdigest()}


def require(condition, message):
    if not condition: raise ValueError(message)


def family_error(a, b):
    a, b = np.asarray(a).reshape(-1, 6), np.asarray(b).reshape(-1, 6)
    return [float(np.linalg.norm(a[:, i:i+3]-b[:, i:i+3])/
        max(float(np.linalg.norm(b[:, i:i+3])), np.finfo(float).tiny)) for i in (0, 3)]


def private_steps(birth):
    p = birth.partition
    original = birth.initial_memory
    state = original
    first_raw = None
    result = []
    summaries = []
    for index, dt in enumerate((1e-4, 7e-5)):
        q = np.zeros(p.parent_ndof) if index == 0 else 1e-7*np.sin(.23*(1+np.arange(p.parent_ndof)))
        da = (1e-4*np.sin(.37*(1+np.arange(18))) if index == 0 else
              -6e-5*np.cos(.29*(2+np.arange(18))))
        after_time = state.time_myr+dt
        loading = source_seed_memory_loading(birth, state, after_time)
        trial = prepare_enriched_loading(p, state, loading)
        response = evaluate_enriched_loading(p, trial, q, da)
        after = accept_enriched_loading(p, trial, response)
        raw = enriched_memory_to_dict(after)
        require(np.any(after.coefficients_m) and np.any(after.displacement_m),
            "Private probe must exercise nonzero compact memory and displacement")
        if first_raw is None: first_raw = raw
        # A separately replayed restart must reproduce every coefficient and
        # ledger, including the second step after the first resumed state.
        replayed = restart_source_seed_memory(birth, raw)
        require(enriched_memory_to_dict(replayed) == raw,
            "Private source-journal restart differs from uninterrupted memory")
        force_work = float(response.mean_generalized_force_n@np.r_[q, da])
        scale = max(abs(force_work), abs(response.mechanical_work_j), 1.)
        require(abs(force_work-response.mechanical_work_j) <= 3e-10*scale,
            "Full mean-force dual work differs from full-field constitutive work")
        parent_mean = evaluate_genesis_volume_operator(p.parent_assembly, np.zeros(p.parent_ndof),
            elastic_constant=response.mean0, elastic_inverse_scale=response.mean1).force
        enriched_parent = np.zeros(p.parent_ndof)
        np.add.at(enriched_parent, p.selected_parent_dofs,
            p.local.stiffness_cross@response.mean_coefficients_m)
        require(np.array_equal(response.mean_generalized_force_n[:p.parent_ndof],
            parent_mean+enriched_parent), "Stored full parent mean force is not its literal old-plus-enriched sum")
        response_raw = {**plain(response), "parent_mean_force_n": parent_mean.tolist(),
            "enriched_parent_force_n": enriched_parent.tolist()}
        result.append(dict(after_time_myr=after_time, parent_increment_m=q.tolist(),
            enrichment_increment_m=da.tolist(), ratio=loading.ratio.tolist(),
            response=response_raw, after_memory=raw))
        summaries.append(dict(private_step=index+1, private_after_time_myr=after_time,
            maximum_abs_compact_memory_m=float(np.max(np.abs(after.coefficients_m))),
            energy_decomposition_j=list(response.energy_decomposition_j),
            work_decomposition_j=list(response.work_decomposition_j),
            dissipation_decomposition_j=list(response.dissipation_decomposition_j),
            full_field_viscous_dissipation_j=response.viscous_dissipation_j,
            balance_residual_j=response.balance_residual_j,
            full_mean_force_work_relative_error=abs(force_work-response.mechanical_work_j)/scale,
            complete_private_restart_exact=True))
        state = after
    first = restart_source_seed_memory(birth, first_raw)
    second = result[1]
    loading = source_seed_memory_loading(birth, first, second["after_time_myr"])
    trial = prepare_enriched_loading(p, first, loading)
    continued = accept_enriched_loading(p, trial, evaluate_enriched_loading(p, trial,
        second["parent_increment_m"], second["enrichment_increment_m"]))
    require(enriched_memory_to_dict(continued) == result[1]["after_memory"],
        "Continuing from the first private restart changed the second response")
    return result, first_raw, enriched_memory_to_dict(continued), summaries


def run(parent_path, output):
    output.mkdir(parents=True, exist_ok=False)
    parent_bytes = parent_path.read_bytes()
    require(hashlib.sha256(parent_bytes).hexdigest() == PARENT_RAW, "Authentic parent raw pin differs")
    packet = json.loads(parent_bytes)
    require(packet["checkpoint_sha256"] == PARENT_SELF, "Authentic parent self pin differs")
    require(pin(PREVIOUS)["raw_sha256"] == PREVIOUS_RAW, "Sealed previous finite seed fixture raw pin differs")
    previous = json.loads(PREVIOUS.read_bytes())
    require(previous["fixture_sha256"] == hashlib.sha256(canonical(
        {k: v for k, v in previous.items() if k != "fixture_sha256"})).hexdigest(),
        "Previously accepted finite seed fixture seal differs")
    require(previous["parent_raw_sha256"] == PARENT_RAW, "Previous seed parent differs")
    # The only public planetary restore in this control.
    model, state = load_volume_slip_checkpoint(parent_path,
        expected_source_checkpoint_sha256=FORMATION)
    require(len(state.geometry.cohorts) == 800 and len(state.accepted_step_sizes_myr) == 4
        and state.time_myr == packet["runtime_state"]["mechanics"]["state"]["time_myr"],
        "Accepted Genesis clock, material ancestry or journal differs")
    save_volume_slip_checkpoint(output/"parent_before.json", model, state)
    require((output/"parent_before.json").read_bytes() == parent_bytes,
        "Full Genesis source replay differs from accepted archive")
    parent, basal = model._assembly(state.geometry, state.fault_history.damage)
    e0, e1 = coefficient_arrays(state.mechanics)
    cases, summaries = [], []
    for prior_case in previous["finite_seed_cases"]:
        d = prior_case["seed_definition"]
        seed = prepare_volume_finite_seed(state.geometry, state.displacement_m,
            d["normal_xyz"], d["offset_m"], d["seed_center_world_m"], d["seed_radius_m"],
            cohort_id=d["original_cohort_id"], quadrature_order=8)
        partitions = {}
        reactions = {}
        for order in (8, 12):
            p = prepare_embedded_partition(parent, seed, e0, e1,
                basal=basal, quadrature_order=order)
            reaction = prepare_volume_seed_reaction(model, state, seed,
                partition=p, quadrature_order=order)
            partitions[order], reactions[order] = p, reaction
        p, reaction = partitions[12], reactions[12]
        inertia_errors = family_error(reactions[8].added_inertia_force_n, reaction.added_inertia_force_n)
        mean_errors = family_error(reactions[8].added_average_internal_force_n, reaction.added_average_internal_force_n)
        require(max(inertia_errors+mean_errors) <= 2e-6,
            "Added accepted-interval A/P inertia or average force lacks observed order agreement")
        reaction_raw = volume_seed_reaction_to_dict(reaction)
        rebound = volume_seed_reaction_from_dict(reaction_raw, model=model, state=state,
            finite_seed=seed, partition=p, quadrature_order=12)
        require(volume_seed_reaction_to_dict(rebound) == reaction_raw, "Source reaction roundtrip differs")
        birth = prepare_source_seed_memory(model, state, seed, partition=p,
            reaction=reaction, quadrature_order=12)
        birth_raw = enriched_memory_to_dict(birth.initial_memory)
        require(birth.initial_memory.parent_state is state.mechanics,
            "Compact memory birth replaced the inherited mechanics")
        require(not np.any(birth.initial_memory.coefficients_m)
            and not np.any(birth.initial_memory.displacement_m), "Birth invented compact strain")
        steps, first, second, private_summary = private_steps(birth)
        cases.append(dict(name=prior_case["name"], seed_definition=d,
            embedded_partition=volume_embedded_partition_to_dict(p),
            finite_seed_binding_sha256=seed.binding_sha256,
            birth_memory=birth_raw, source_memory_birth=source_seed_memory_to_dict(birth),
            private_steps=steps, restarted_after_first_memory=first,
            restarted_after_second_memory=second, accepted_interval_reaction=reaction_raw,
            convergence=dict(orders=[8, 12], observed_relative_tolerance=2e-6,
                added_inertia_errors_A_P=inertia_errors, added_average_force_errors_A_P=mean_errors,
                added_inertia_force_n_order8=reactions[8].added_inertia_force_n.tolist(),
                added_average_internal_force_n_order8=reactions[8].added_average_internal_force_n.tolist())))
        summaries.append(dict(name=prior_case["name"], original_cohort_id=p.selected_cohort_id,
            quadrature_order=12, original_memory_and_ledgers_at_birth_exact=True,
            initial_compact_memory_and_displacement_exact_zero=True,
            birth_operator_energy_roundoff_j=p.initial_elastic_energy_j-state.mechanics.elastic_energy_j,
            added_inertia_norms_A_P=[float(np.linalg.norm(reaction.added_inertia_force_n.reshape(-1, 6)[:, i:i+3])) for i in (0, 3)],
            average_reaction_minus_static_endpoint_norm_n=float(np.linalg.norm(
                reaction.constraint_multipliers_n-reaction.endpoint_static_constraint_reaction_n)),
            added_tied_work_j=reaction_raw["tied_constraint_work_j"],
            parent_residual_preserved_exactly=True, private_steps=private_summary))
        print(json.dumps(dict(status="CASE_PASS", name=prior_case["name"],
            compact_memory_restart_exact=True, accepted_interval_reaction_checked=True)), flush=True)
    save_volume_slip_checkpoint(output/"parent_after.json", model, state)
    require((output/"parent_after.json").read_bytes() == parent_bytes
        and parent_path.read_bytes() == parent_bytes, "Private probes changed the accepted planet")
    flags = dict(new_Genesis_time_steps=0, new_material_cohorts=0,
        creation_work_debited_from_Genesis_j=0., history_reinitialized=False,
        accepted_planet_memory_changed=False, physical_creation_admitted=False,
        physical_release_admitted=False, funding_admitted=False, point_one_complete=False,
        physical_Gamma_selected=False, actual_dynamic_reaction_replaced=False,
        automated_goal_enabled=False)
    fixture = seal(dict(format=FORMAT, scope=SCOPE, parent_raw_sha256=PARENT_RAW,
        parent_checkpoint_sha256=PARENT_SELF, formation_checkpoint_sha256=FORMATION,
        parent_before_sha256=pin(output/"parent_before.json")["raw_sha256"],
        parent_after_sha256=pin(output/"parent_after.json")["raw_sha256"],
        previous_seed_fixture_pin=pin(PREVIOUS), cases=cases, **flags), "fixture_sha256")
    fixture_path = output/"control_fixture.json"
    fixture_path.write_bytes(canonical(fixture)+b"\n")
    report = seal(dict(format=FORMAT, status="PASS", scope=SCOPE,
        time_myr=state.time_myr, cohort_count=800, accepted_intervals=4,
        public_planetary_restore_count=1, fixture_sha256=fixture["fixture_sha256"],
        parent_raw_sha256=PARENT_RAW, parent_checkpoint_sha256=PARENT_SELF,
        formation_checkpoint_sha256=FORMATION, cases=summaries,
        whole_parent_before_after_byte_identical=True,
        private_probe_steps_per_case=2, private_probe_clocks_are_not_accepted_Genesis=True,
        fixed_material_temperature_elasticity_boundaries=True, **flags), "report_sha256")
    report_path = output/"report.json"
    report_path.write_bytes(canonical(report)+b"\n")
    print(json.dumps(dict(status="PASS", report=pin(report_path), self_sha256=report["report_sha256"],
        fixture=pin(fixture_path), fixture_sha256=fixture["fixture_sha256"])), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.parent_checkpoint.resolve(), args.output.resolve())
