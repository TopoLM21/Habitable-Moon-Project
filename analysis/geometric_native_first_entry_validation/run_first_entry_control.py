"""Re-solve the real force and apply the complete declared material closure.

The separate implicit material state uses the original weak-plane direction.
Its archive and endpoint hot fill do not adopt the refused force candidate.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analysis.geometric_native_contact_event_validation.run_time_support_control import run_control as run_force_control
from analysis.geometric_native_material_history_validation.run_material_history_control import (
    encoded, memory_payloads, sha, sources, write,
)
from tectonics.geometric_native_domain_journal import (
    initialize_native_domain_journal, native_domain_journal_to_dict,
    native_domain_journal_from_dict,
)
from tectonics.geometric_native_first_entry_domains import (
    certify_native_first_entry_domains, native_first_entry_domains_to_dict,
    native_first_entry_domains_from_dict,
)
from tectonics.geometric_native_first_entry_material import (
    initialize_native_first_entry_material, native_first_entry_material_to_dict,
    native_first_entry_material_from_dict,
    certify_native_first_entry_material_point, evaluate_native_first_entry_material_point,
    native_first_entry_material_point_chronology_to_dict,
    native_first_entry_material_point_chronology_from_dict,
    native_first_entry_material_point_query_to_dict,
    native_first_entry_material_point_query_from_dict,
    evaluate_native_first_entry_material_endpoint,
    native_first_entry_material_endpoint_query_to_dict,
    native_first_entry_material_endpoint_query_from_dict,
)
from tectonics.geometric_native_material_history import make_native_material_memory_reference
from tectonics.geometric_native_surface import native_surface_from_dict
from tectonics.geometric_native_pose import FractionInterval as I
from tectonics.geometric_underthrust_motion import HotBirthParameters

PROVENANCE = "Declared complete once-only first-entry archive and endpoint hot fill; force candidate remains refused"


def run_control(out, *, reuse_force=None):
    started = time.perf_counter()
    out.mkdir(parents=True, exist_ok=False)
    before = sources()
    write(out / "sources_before.json", before)
    force = out / "original_force"
    if reuse_force is None:
        force.mkdir()
        force_report = run_force_control(force)
    else:
        shutil.copytree(reuse_force, force)
        force_report = json.loads((force / "report.json").read_bytes())
    initial = json.loads((force / "initial.json").read_bytes())
    candidate = json.loads((force / "force_candidate.json").read_bytes())
    support = json.loads((force / "constitutive_time_support.json").read_bytes())
    probe = ROOT / "analysis/geometric_native_underthrust_validation/force_control_release_02/force_candidate_refusal_probe.json"
    historical = json.loads(probe.read_bytes())
    assert all(candidate[key] == value for key, value in historical["candidate"].items())
    assert candidate["admission"] == historical["admission"]
    surface = native_surface_from_dict(initial["current_surface"])
    omega = {owner: tuple(vector) for owner, vector in candidate["omega_rad_per_myr"]}
    duration = candidate["admission"]["dt_myr"]
    mechanics = initial["native_underthrust_motion"]["initial_mechanics"]
    old, new = memory_payloads(initial, candidate, support)
    old.update(original_mechanics_payload=deepcopy(mechanics),
        original_surface_payload=deepcopy(initial["current_surface"]),
        original_force_checkpoint_sha256=sha(force / "initial.json"))
    memory = make_native_material_memory_reference(old, new,
        constitutive_path_sha256=support["path_sha256"])
    write(out / "original_source_surface.json", initial["current_surface"])
    print("Reconstructing all original domains and strengthened physical source references", flush=True)
    journal = initialize_native_domain_journal(surface, omega, duration, memory_reference=memory)
    journal_raw = native_domain_journal_to_dict(journal)
    write(out / "full_domain_journal.json", journal_raw)
    journal_replay = native_domain_journal_from_dict(deepcopy(journal_raw),
        expected_memory_reference=memory)
    write(out / "full_domain_journal_replayed.json", native_domain_journal_to_dict(journal_replay))
    print("Certifying the entire selected owner first-entry envelope and its curved excess", flush=True)
    sweep = certify_native_first_entry_domains(journal)
    sweep_raw = native_first_entry_domains_to_dict(sweep)
    write(out / "first_entry_sweep.json", sweep_raw)
    replayed_sweep = native_first_entry_domains_from_dict(deepcopy(sweep_raw))
    write(out / "first_entry_sweep_replayed.json", native_first_entry_domains_to_dict(replayed_sweep))
    print("Applying one-time original material debits and filling actual endpoint vacancies", flush=True)
    state = initialize_native_first_entry_material(journal, sweep,
        original_mechanics_payload=mechanics,
        birth_parameters=HotBirthParameters(**initial["native_underthrust_motion"]["birth_parameters"]),
        provenance=PROVENANCE)
    state_raw = native_first_entry_material_to_dict(state)
    loaded = native_first_entry_material_from_dict(deepcopy(state_raw),
        expected_journal=journal, expected_original_mechanics_payload=mechanics)
    write(out / "material_state.json", state_raw)
    write(out / "material_state_replayed.json", native_first_entry_material_to_dict(loaded))
    previous = ROOT / "analysis/geometric_native_full_domain_validation/control_release_01"
    saved = json.loads((previous / "point_certificates.json").read_bytes())
    original_cases = {record["label"]: record for record in saved
        if record["kind"] != "fixed_world_receiver"}
    saved_queries = json.loads((previous / "point_queries.json").read_bytes())
    certificates, queries, replayed_queries, endpoints, replayed_endpoints = [], [], [], [], []
    labels = ("main_plane_reentry", "transverse_masked_main_entry", "ordinary_entry",
              "no_entry", "swept_only_owner_entry_exit")
    cases = []
    for label in labels:
        record = original_cases[label]
        point = tuple(Fraction(*pair) for pair in record["certificate"]["trajectory"]["point"])
        times = tuple(Fraction(*query["time_fraction"]) for query in saved_queries
                      if query["label"] == label)
        cases.append((label, record["donor_fragment_id"], point, times))
    cases.extend((
        ("empty_donor_four", "octant:4", (Fraction(1), Fraction(-1), Fraction(-1)), (Fraction(0), Fraction(1))),
        ("retained_overriding_material", "octant:0", (Fraction(-1), Fraction(-1), Fraction(-1)), (Fraction(0), Fraction(1))),
    ))
    for label, parent_id, point, times in cases:
        print("Executing material chronology and true endpoint vacancy: " + label, flush=True)
        chronology = certify_native_first_entry_material_point(state, parent_id, point)
        raw = native_first_entry_material_point_chronology_to_dict(chronology)
        replay = native_first_entry_material_point_chronology_from_dict(deepcopy(raw), state=loaded)
        assert encoded(raw) == encoded(native_first_entry_material_point_chronology_to_dict(replay))
        certificates.append(dict(label=label, certificate=raw))
        first = chronology.first_entry_certificate
        window = None if first is None else first.first_entry_time_fraction
        if window is not None:
            times = tuple(sorted(set((*times, window.lo, window.midpoint, window.hi,
                                      (window.hi + Fraction(1)) / 2))))
        for instant in times:
            query = native_first_entry_material_point_query_to_dict(
                evaluate_native_first_entry_material_point(chronology, instant))
            restored = native_first_entry_material_point_query_to_dict(
                native_first_entry_material_point_query_from_dict(deepcopy(query), chronology=replay))
            assert encoded(query) == encoded(restored)
            queries.append(dict(label=label, query=query))
            replayed_queries.append(dict(label=label, query=restored))
        endpoint = native_first_entry_material_endpoint_query_to_dict(
            evaluate_native_first_entry_material_endpoint(chronology))
        restored = native_first_entry_material_endpoint_query_to_dict(
            native_first_entry_material_endpoint_query_from_dict(deepcopy(endpoint), chronology=replay))
        assert encoded(endpoint) == encoded(restored)
        endpoints.append(dict(label=label, query=endpoint))
        replayed_endpoints.append(dict(label=label, query=restored))
        if label == "swept_only_owner_entry_exit":
            assert endpoint["raw_coverage"] is True
            assert endpoint["actual_material_vacancy"] is True
            assert endpoint["retained_material"] is False
            assert endpoint["birth_material_ids"]
    write(out / "point_certificates.json", certificates)
    write(out / "material_point_queries.json", queries)
    write(out / "material_point_queries_replayed.json", replayed_queries)
    write(out / "endpoint_material_queries.json", endpoints)
    write(out / "endpoint_material_queries_replayed.json", replayed_endpoints)
    archive_area = sum((row.archived_area_bounds_sr for row in state.parent_allocations), I(0))
    vacancy_area = sum((row.area_bounds_sr for row in state.endpoint_vacancies), I(0))
    assert archive_area.lo <= vacancy_area.hi and vacancy_area.lo <= archive_area.hi
    assert len(state.archive_cohorts) == 3 and len(state.parent_allocations) == 8
    assert (force / "initial.json").read_bytes() == (force / "after_refusal.json").read_bytes()
    pairs = (
        ("full_domain_journal.json", "full_domain_journal_replayed.json"),
        ("first_entry_sweep.json", "first_entry_sweep_replayed.json"),
        ("material_state.json", "material_state_replayed.json"),
        ("material_point_queries.json", "material_point_queries_replayed.json"),
        ("endpoint_material_queries.json", "endpoint_material_queries_replayed.json"),
        ("original_force/initial.json", "original_force/after_refusal.json"),
    )
    for first, second in pairs:
        assert (out / first).read_bytes() == (out / second).read_bytes()
    after = sources()
    assert before == after, "Source changed while computing the material control"
    write(out / "sources_after.json", after)
    report = dict(format="native-whole-first-entry-material-control-1", status="PASS",
        seconds=time.perf_counter()-started, source_count=len(before), sources_unchanged=True,
        original_force_re_solved=reuse_force is None, original_candidate_all_fields_identical=True,
        source_probe_sha256=sha(probe), original_force_field_count=len(historical["candidate"]),
        source_parent_count=8, donor_domain_count=len(sweep.donor_domains),
        positive_archive_cohort_count=len(state.archive_cohorts),
        endpoint_vacancy_domain_count=len(state.endpoint_vacancies),
        endpoint_newborn_count=sum(row.newborn is not None for row in state.endpoint_vacancies),
        point_certificate_count=len(certificates), material_query_count=len(queries),
        endpoint_query_count=len(endpoints), byte_pairs=[list(pair) for pair in pairs],
        whole_selected_owner_swept_domain_certified=True, full_curved_extra_integrated=True,
        complete_standalone_material_closure_applied=True,
        actual_material_vacancies_computed=True, endpoint_hot_fill_applied=True,
        original_live_archive_reservoirs_symbolically_conserved=True,
        original_seed_direction_authenticated=True, original_mechanical_work_preserved=True,
        candidate_mechanics_adopted=False, physical_pair_admission_authorized=False,
        global_native_inventory_updated=False, accepted_loaded_native_intervals=0,
        accepted_source_work_j=0, force_refusal_reasons=force_report["refusal_reasons"],
        artifacts_sha256={str(path.relative_to(out)):sha(path)
            for path in sorted(out.rglob("*")) if path.is_file()})
    report["report_sha256"] = hashlib.sha256(encoded(report)).hexdigest()
    write(out / "report.json", report)
    print(json.dumps({key:report[key] for key in ("status", "seconds",
        "positive_archive_cohort_count", "endpoint_newborn_count", "material_query_count",
        "accepted_loaded_native_intervals")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reuse-force", type=Path,
        help="Development only: reuse saved force evidence; release consolidation refuses it")
    args = parser.parse_args()
    run_control(args.out.resolve(), reuse_force=args.reuse_force)
