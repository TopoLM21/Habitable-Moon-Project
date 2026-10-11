"""Physical line and support control using the shared authenticated source.

This callable is intended to run after the new pytest modules in the same
interpreter.  Its receipt covers the line and support blocks, and deliberately
does not declare an accepted force interval on that diagnostic source.
"""
from __future__ import annotations

from copy import deepcopy
from fractions import Fraction as F
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis.geometric_native_active_line_validation.validation_support import (
    BASE, PRIOR, digest, encoded, guard_sources, read, value_sha, verify_baseline, write,
)
from analysis.geometric_native_active_line_validation.independent_line_oracle import (
    independent_frozen_potential_integral,
)


CARRIERS = (
    ("positive_two_changes", "octant:3", (F(0), F(1), F(1, 10**55))),
    ("negative_retained", "octant:0", (F(0), F(-1), F(-1, 10**55))),
    ("ordinary_core", "octant:2", (F(0), F(1), F(-1))),
    ("mixed_fresh_negative_core_retained", "octant:2", (F(0), F(1, 10**20), F(-1))),
    ("closed_endpoint", "octant:3", (F(0), F(1), F(0))),
)


def _rehash_new_owning_seals(value):
    if type(value) is list:
        for row in value:
            _rehash_new_owning_seals(row)
    elif type(value) is dict:
        for name, row in value.items():
            if name != "source_history":
                _rehash_new_owning_seals(row)
        key = {
            "native-active-receiver-contact-whole-line-geometry-1": "certificate_sha256",
            "native-active-receiver-contact-whole-line-carrier-selection-1": "selection_sha256",
            "native-active-original-contact-line-work-1": "integral_sha256",
            "native-implicit-original-bank-live-clamp-support-1": "support_sha256",
        }.get(value.get("format"))
        if key is not None and key in value:
            value[key] = value_sha({name: row for name, row in value.items() if name != key})


def _refuse(raw, mutation, loader, label, changed_path):
    forged = deepcopy(raw)
    mutation(forged)
    _rehash_new_owning_seals(forged)
    try:
        loader(forged)
    except (TypeError, ValueError) as error:
        return dict(label=label, changed_path=changed_path, rejected=True,
            new_owning_checksums_resealed=True, externally_supplied_source_unchanged=True,
            exception_type=type(error).__name__, message=str(error),
            forged_dto_sha256=value_sha(forged))
    raise AssertionError("A semantic substitution was accepted: " + label)


def run_control(out):
    from tectonics.geometric_native_active_contact_line import (
        native_active_contact_line_to_dict, native_active_contact_line_from_dict,
        native_active_contact_line_carrier_to_dict, native_active_contact_line_carrier_from_dict,
        select_native_active_contact_line_carrier,
    )
    from tectonics.geometric_native_active_contact_line_work import (
        native_active_contact_line_work_to_dict, native_active_contact_line_work_from_dict,
    )
    from tectonics.geometric_native_implicit_force_support import (
        certify_native_implicit_force_support, native_implicit_force_support_to_dict,
        native_implicit_force_support_from_dict,
    )
    from tectonics.geometric_native_physical_history import native_physical_history_binding_to_dict
    from tectonics.geometric_native_underthrust_motion_io import save_native_underthrust_motion_checkpoint
    from tests.test_geometric_native_active_contact_line_work import get_active_line_work_case

    started = time.perf_counter()
    before, immutable = guard_sources(require_frozen=True)
    verify_baseline()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    write(out / "sources_before.json", before)
    prepared = get_active_line_work_case()
    case, work = prepared["case"], prepared["value"]
    history, source = case["history"], case["source"]
    if source != read(PRIOR / "control_release_02/source_history_before.json"):
        raise AssertionError("The shared actual source differs from its authenticated complete history")
    write(out / "source_history_before.json", source)
    write(out / "force_candidate_before.json", source["actual_candidate"])
    if len(source["actual_candidate"]) != 14:
        raise AssertionError("The actual original force candidate lost a literal field")
    source_checkpoint_path = ROOT / "analysis/geometric_native_first_entry_validation/control_release_02/original_force/initial.json"
    source_provenance = read(source_checkpoint_path)["provenance"]
    save_native_underthrust_motion_checkpoint(out / "source_checkpoint_before.json",
        case["original"]["mesh"], case["original"]["origin"], case["original"]["before"],
        provenance=source_provenance)
    if (out / "source_checkpoint_before.json").read_bytes() != source_checkpoint_path.read_bytes():
        raise AssertionError("Typed source checkpoint differs from its original bytes")
    certificate = case["certificate"]
    line_raw = native_active_contact_line_to_dict(certificate)
    line_args = dict(expected_history=history, expected_law=case["law"], expected_source_history_dto=source)
    line_replay = native_active_contact_line_from_dict(deepcopy(line_raw), **line_args)
    write(out / "line_certificate.json", line_raw)
    write(out / "line_certificate_replayed.json", native_active_contact_line_to_dict(line_replay))
    selections, selections_replayed, selection_loaders = [], [], {}
    for label, fragment, carrier in CARRIERS:
        selected = select_native_active_contact_line_carrier(certificate, fragment, carrier,
            expected_source_history_dto=source)
        raw = native_active_contact_line_carrier_to_dict(selected, expected_source_history_dto=source)
        args = dict(expected_certificate=certificate, expected_receiver_fragment_id=fragment,
            expected_receiver_reference_coordinates=carrier, expected_source_history_dto=source)
        loader = lambda value, args=args: native_active_contact_line_carrier_from_dict(value, **args)
        replay = loader(deepcopy(raw))
        selections.append(dict(label=label, selection=raw))
        selections_replayed.append(dict(label=label, selection=native_active_contact_line_carrier_to_dict(
            replay, expected_source_history_dto=source)))
        selection_loaders[label] = loader
    write(out / "line_carriers.json", selections)
    write(out / "line_carriers_replayed.json", selections_replayed)
    work_raw = native_active_contact_line_work_to_dict(work, expected_source_history_dto=source)
    work_args = dict(history=history, expected_source_history_dto=source, expected_line_certificate=certificate)
    work_replay = native_active_contact_line_work_from_dict(deepcopy(work_raw), **work_args)
    write(out / "weighted_pair_work.json", work_raw)
    write(out / "weighted_pair_work_replayed.json", native_active_contact_line_work_to_dict(
        work_replay, expected_source_history_dto=source))
    support = certify_native_implicit_force_support(history, expected_source_history_dto=source)
    support_raw = native_implicit_force_support_to_dict(support)
    support_args = dict(expected_history=history, expected_source_history_dto=source)
    support_replay = native_implicit_force_support_from_dict(deepcopy(support_raw), **support_args)
    write(out / "full_bank_clamp_support.json", support_raw)
    write(out / "full_bank_clamp_support_replayed.json", native_implicit_force_support_to_dict(support_replay))
    observations = []
    for quarter in certificate.quarters:
        pair = next(row for row in work.receiving_pair_budgets if row.receiver_trace_id == quarter.receiver_trace_id
            and row.donor_trace_id == quarter.original_donor_trace_id)
        core = quarter.core_computational_angle_bounds
        numerical = independent_frozen_potential_integral(source, quarter.receiver_trace_id,
            core.lo, core.hi, quarter.computational_extent_radians)
        if not pair.core_potential_increment_integral_bounds_m4.lo <= F(numerical) <= pair.core_potential_increment_integral_bounds_m4.hi:
            raise AssertionError("Independent real-harmonic core potential is outside its production enclosure")
        observations.append(dict(receiver_fragment_id=quarter.receiver_fragment_id,
            potential_increment_m4_decimal=str(numerical), enclosed=True,
            scope="independent real harmonics and full SO3 numerical core cross-check; whole-angle proof is separate"))
    write(out / "independent_harmonic_core_observations.json", observations)
    refusals = []
    line_loader = lambda raw: native_active_contact_line_from_dict(raw, **line_args)
    work_loader = lambda raw: native_active_contact_line_work_from_dict(raw, **work_args)
    support_loader = lambda raw: native_implicit_force_support_from_dict(raw, **support_args)
    def check(raw, loader, label, path, mutation):
        refusals.append(_refuse(raw, mutation, loader, label, path))
    check(line_raw, line_loader, "discard_tiny_positive_measure", "common_computational_strip_width_radians",
        lambda raw: raw.update(common_computational_strip_width_radians=[0, 1]))
    check(line_raw, line_loader, "negative_seam_reassignment", "negative_axis_seams",
        lambda raw: raw.update(negative_axis_seams="reassign_without_positive_fresh_gate"))
    check(line_raw, line_loader, "force_adopted_by_geometry", "accepted_native_force_intervals",
        lambda raw: raw.update(accepted_native_force_intervals=1))
    positive_raw = selections[0]["selection"]
    check(positive_raw, selection_loaders["positive_two_changes"], "wrong_incoming_original_bank",
        "implicit_branches[0].incoming_donor_fragment_id",
        lambda raw: raw["implicit_branches"][0].update(incoming_donor_fragment_id="octant:7"))
    check(positive_raw, selection_loaders["positive_two_changes"], "implicit_clock_midpoint",
        "exact_frozen_event_reference", lambda raw: raw.update(exact_frozen_event_reference="midpoint_rotation"))
    check(positive_raw, selection_loaders["positive_two_changes"], "assumed_tangent_crossing",
        "implicit_branches[0].exact_existence_condition",
        lambda raw: raw["implicit_branches"][0].update(exact_existence_condition="closed_range_including_tangent"))
    check(work_raw, work_loader, "coarea_as_contact_measure", "measure",
        lambda raw: raw.update(measure="swept_two_dimensional_material_area"))
    check(work_raw, work_loader, "old_nodal_work_added", "original_nodal_ledgers_summed_with_continuous_work",
        lambda raw: raw.update(original_nodal_ledgers_summed_with_continuous_work=True))
    check(work_raw, work_loader, "literal_signed_residual_zeroed", "compliance_balance_residual_bounds_j",
        lambda raw: raw.update(compliance_balance_residual_bounds_j=dict(lo=[0, 1], hi=[0, 1])))
    check(work_raw, work_loader, "positive_strip_measure_zeroed", "explicitly_enclosed_strip_interface_area_m2",
        lambda raw: raw.update(explicitly_enclosed_strip_interface_area_m2=[0, 1]))
    check(support_raw, support_loader, "support_force_adoption", "accepted_loaded_native_intervals",
        lambda raw: raw.update(accepted_loaded_native_intervals=1))
    check(support_raw, support_loader, "support_foreign_source", "source_history_sha256",
        lambda raw: raw.update(source_history_sha256="0"*64))
    write(out / "semantic_refusals.json", refusals)
    write(out / "source_history_after.json", native_physical_history_binding_to_dict(history))
    write(out / "force_candidate_after.json", native_physical_history_binding_to_dict(history)["actual_candidate"])
    save_native_underthrust_motion_checkpoint(out / "source_checkpoint_after.json",
        case["original"]["mesh"], case["original"]["origin"], case["original"]["before"],
        provenance=source_provenance)
    after, immutable_after = guard_sources(require_frozen=True)
    write(out / "sources_after.json", after)
    if before != after or immutable != immutable_after:
        raise AssertionError("A Python source changed during the physical line control")
    pairs = (("source_history_before.json", "source_history_after.json"),
        ("force_candidate_before.json", "force_candidate_after.json"),
        ("source_checkpoint_before.json", "source_checkpoint_after.json"),
        ("line_certificate.json", "line_certificate_replayed.json"),
        ("line_carriers.json", "line_carriers_replayed.json"),
        ("weighted_pair_work.json", "weighted_pair_work_replayed.json"),
        ("full_bank_clamp_support.json", "full_bank_clamp_support_replayed.json"))
    byte_pairs = []
    for first, second in pairs:
        if (out / first).read_bytes() != (out / second).read_bytes():
            raise AssertionError("A preserved/restarted pair differs: " + first)
        byte_pairs.append(dict(first=first, second=second, bytes=(out / first).stat().st_size,
            sha256=digest(out / first)))
    if history.accepted_before.receipts or history.accepted_before.cumulative_source_work_j != 0:
        raise AssertionError("The diagnostic original candidate was accepted by the line control")
    report = dict(format="native-active-line-physical-control-1", status="PASS",
        seconds=time.perf_counter()-started, source_count=len(before), prior_704_sources_unchanged=True,
        source_before_after_unchanged=True, shared_authenticated_actual_source=True,
        original_actual_force_fields_preserved=14, whole_line_selection_proved=True,
        whole_line_contact_compression_and_monotone_down_dip_proved=True,
        weighted_receiving_pair_count=len(work.receiving_pair_budgets), original_quarter_count=4,
        continuous_interface_measure_preserved=True, positive_width_strips_retained=True,
        independent_harmonic_core_observation_count=len(observations),
        line_carrier_selection_count=len(selections), semantic_refusal_count=len(refusals),
        byte_pairs=byte_pairs, signed_literal_compliance_residual_retained=True,
        original_gauss_atom_count=len(history.quadrature_histories), original_atoms_added_to_continuous_work=False,
        transfer_work_energy_and_dissipation_exactly_zero=True,
        block_1_completed_scope="Whole original receiving line under the explicit selector, original-measure numerical pair budgets, full-line constitutive branch and strict source-bound restart",
        full_bank_live_clamp_support_observed=True,
        accepted_native_force_intervals=0, accepted_source_work_j=0, candidate_mechanics_adopted=False,
        entire_point_1_completed=False,
        remaining_scope="Actual implicit force admission, later accepted intervals and joint material/motion restart are separate controls")
    saved = write(out / "report.json", report, "report_sha256")
    print(encoded({key: saved[key] for key in ("status", "seconds", "weighted_receiving_pair_count",
        "semantic_refusal_count", "accepted_native_force_intervals", "entire_point_1_completed")}).decode(), flush=True)
    return saved


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    run_control(parser.parse_args().out)
