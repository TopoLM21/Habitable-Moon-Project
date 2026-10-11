"""Re-solve the unchanged force and audit every original native source domain.

The journal records all endpoint overlap and raw kinematic vacancy domains.
Continuous certificates concern arbitrary individual original material points;
they neither integrate the swept loss set nor adopt a force/material step.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, is_dataclass
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
from tectonics.geometric_native_donor_sweep import (
    _normal_coefficients, _scalar_coefficients, _bounds, certify_native_donor_patch_sweep,
)
from tectonics.geometric_native_material_history import make_native_material_memory_reference
from tectonics.geometric_native_point_chronology import (
    certify_native_point_chronology, certify_native_point_receiver_chronology,
    evaluate_native_point_chronology, native_point_chronology_to_dict,
    native_point_chronology_from_dict, native_point_chronology_query_to_dict,
)
from tectonics.geometric_native_polygons import (
    NativeRay, NativePlane, NativePolygon, _cross, _ray_values, _side, native_subtract,
    native_intersection, native_polygon_to_dict, validate_native_polygon,
)
from tectonics.geometric_native_pose import FractionInterval
from tectonics.geometric_native_surface import native_surface_from_dict
from tectonics.fractional_surface import EXTENSIVE_FIELDS
from tectonics.geometric_native_swept_domain import (
    certify_native_swept_domain, native_swept_domain_to_dict, native_swept_domain_from_dict,
)


def json_value(value):
    if type(value) is Fraction:
        return [value.numerator, value.denominator]
    if type(value) is FractionInterval:
        return dict(lo=json_value(value.lo), hi=json_value(value.hi))
    if is_dataclass(value):
        return json_value(asdict(value))
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    if type(value) is dict:
        return {key: json_value(v) for key, v in value.items()}
    return value


def point_cases(donor, receiver, donor_omega, receiver_omega, duration):
    patch = certify_native_donor_patch_sweep(donor.geometry, receiver.geometry,
        donor_omega, receiver_omega, duration)
    dt = Fraction.from_float(duration)
    da = tuple(Fraction.from_float(v) * dt for v in donor_omega)
    ra = tuple(Fraction.from_float(v) * dt for v in receiver_omega)
    main = _normal_coefficients((-1, 0, 0), da, ra, 24)
    transverse = _normal_coefficients((0, 0, -1), da, ra, 24)
    c1, c2 = transverse[1][1], transverse[2][1]
    assert c1 < 0 < c2
    uz = -c1 / (2 * c2)
    z = c1 * c1 / (8 * c2)
    return (
        ("main_plane_reentry", (patch.x_bounds.midpoint, patch.y_bounds.midpoint, Fraction(-1)),
         tuple(patch.sample_fractions)),
        ("transverse_plane_reentry", (main[1][1] * uz / 1000, Fraction(1), -z),
         (Fraction(0), uz / 16, uz, 4 * uz, Fraction(1))),
        ("transverse_masked_main_entry", (main[1][1] * uz / 2, Fraction(1), -z),
         (Fraction(0), uz / 16, uz, 4 * uz, Fraction(1))),
        ("ordinary_entry", (main[1][1] / 2, Fraction(1), Fraction(-1)),
         (Fraction(0), Fraction(1, 4), Fraction(3, 4), Fraction(1))),
        ("no_entry", (Fraction(1), Fraction(1), Fraction(-1)),
         (Fraction(0), Fraction(1, 2), Fraction(1))),
        ("initial_boundary", (Fraction(0), Fraction(1), Fraction(-1)),
         (Fraction(0), Fraction(1, 2), Fraction(1))),
    )


def transient_material_patch(donor, receiver, donor_omega, receiver_omega, duration):
    """A whole positive 2D original-material patch enters a transient receiver.

    The receiver is octant:3, while the final receiving fragment is octant:2.
    Its Z interface is internal to owner 0. No new fault is inferred from it.
    """
    dt = Fraction.from_float(duration)
    da = tuple(Fraction.from_float(v) * dt for v in donor_omega)
    ra = tuple(Fraction.from_float(v) * dt for v in receiver_omega)
    main = _normal_coefficients((-1, 0, 0), da, ra, 24)
    transverse = _normal_coefficients((0, 0, -1), da, ra, 24)
    c1, c2 = transverse[1][1], transverse[2][1]
    u = -c1 / (2 * c2)
    x = FractionInterval(main[1][1] * u / 3, 2 * main[1][1] * u / 3)
    z = FractionInterval(c1 * c1 / (10 * c2), c1 * c1 / (6 * c2))
    assert 0 < x.lo < x.hi and 0 < z.lo < z.hi and 0 < 4 * u < 1
    rays = tuple(NativeRay("seed", (a, Fraction(1), -b)) for a, b in (
        (x.lo, z.lo), (x.hi, z.lo), (x.hi, z.hi), (x.lo, z.hi)))
    planes = tuple(NativePlane(_cross(a.reference, b.reference))
                   for a, b in zip(rays, rays[1:] + rays[:1]))
    patch = validate_native_polygon(NativePolygon(rays, planes))
    assert not native_subtract(patch, donor.geometry)
    rate = sum(map(abs, da + ra), Fraction(0))
    coordinates = (x, FractionInterval(1), -z)
    scale = x.hi + 1 + z.hi
    proofs = []
    for support in receiver.geometry.supports:
        coeff = _scalar_coefficients(_normal_coefficients(support.normal, da, ra, 24), coordinates)
        bounds = tuple(_bounds(coeff, FractionInterval(t), 0, rate, scale, 256)
                       for t in (Fraction(0), u, 4 * u, Fraction(1)))
        assert bounds[1].lo > 0
        proofs.append(dict(receiver_normal=list(support.normal), gap_bounds=json_value(bounds)))
    assert any(_bounds(_scalar_coefficients(_normal_coefficients(support.normal, da, ra, 24), coordinates),
                      FractionInterval(4 * u), 0, rate, scale, 256).hi < 0
               for support in receiver.geometry.supports)
    endpoint = native_intersection(receiver.geometry.rotated(receiver_omega, duration),
                                   donor.geometry.rotated(donor_omega, duration))
    assert endpoint is None
    return dict(format="native-transient-original-material-patch-witness-1",
        donor_fragment_id=donor.fragment_id, receiver_fragment_id=receiver.fragment_id,
        donor_geometry=native_polygon_to_dict(donor.geometry),
        receiver_geometry=native_polygon_to_dict(receiver.geometry),
        donor_omega=list(donor_omega), receiver_omega=list(receiver_omega), dt_myr=duration,
        original_patch=native_polygon_to_dict(patch), original_patch_area_sr_bounds=json_value(patch.area_bounds()),
        chart="p=(x,1,-z)", x_bounds=json_value(x), z_bounds=json_value(z),
        sample_time_fractions=json_value((Fraction(0), u, 4 * u, Fraction(1))),
        full_receiver_plane_proofs=proofs, order=24, precision=256,
        all_patch_points_inside_receiver_at_middle_time=True,
        endpoint_pair_intersection=None, positive_2d_transient_pair_absent_at_endpoint=True,
        receiver_z_interface_is_internal_to_same_owner=True,
        new_fault_plane_inferred=False, physical_pair_ancestry_authenticated=False,
        material_allocation_adopted=False, geometry_admission_authorized=False)


def swept_only_material_patch(donor, receiver, donor_omega, receiver_omega, duration, swept):
    """Positive 2D original material which enters owner 0 and leaves by the end.

    This is a genuinely missing endpoint loss under the declared conditional
    once-only passive archive, unlike an internal receiver-fragment switch.
    Its end footprint is raw-covered by the departed donor but would need hot
    fill if that original material had been archived at its first entry.
    """
    dt = Fraction.from_float(duration)
    da = tuple(Fraction.from_float(v) * dt for v in donor_omega)
    ra = tuple(Fraction.from_float(v) * dt for v in receiver_omega)
    coeff = _normal_coefficients((-1, 0, 0), da, ra, 24)
    delta = swept.initial_q_bounds.lo - swept.terminal_q_bounds.hi
    middle = (swept.initial_q_bounds.lo + swept.terminal_q_bounds.hi) / 2
    assert delta > 0
    x = FractionInterval(coeff[1][1] * delta / 100, coeff[1][1] * delta / 50)
    y = FractionInterval(middle - delta / 10, middle + delta / 10)
    coordinates = (x, -y, FractionInterval(1))
    rays = tuple(NativeRay("seed", (a, -b, Fraction(1))) for a, b in (
        (x.lo, y.lo), (x.lo, y.hi), (x.hi, y.hi), (x.hi, y.lo)))
    supports = tuple(NativePlane(_cross(a.reference, b.reference))
                     for a, b in zip(rays, rays[1:] + rays[:1]))
    patch = validate_native_polygon(NativePolygon(rays, supports))
    assert not native_subtract(patch, donor.geometry)
    rate = sum(map(abs, da + ra), Fraction(0))
    scale = x.hi + y.hi + 1
    times = (Fraction(0), Fraction(1, 4), Fraction(1, 2), Fraction(3, 4), Fraction(1))
    proofs = []
    for support in receiver.geometry.supports:
        scalar = _scalar_coefficients(_normal_coefficients(support.normal, da, ra, 24), coordinates)
        bounds = tuple(_bounds(scalar, FractionInterval(t), 0, rate, scale, 256) for t in times)
        whole = _bounds(scalar, FractionInterval(0, 1), 0, rate, scale, 256)
        assert bounds[2].lo > 0
        if support.normal == (-1, 0, 0):
            assert bounds[0].hi < 0 and bounds[-1].hi < 0
        else:
            assert whole.lo > 0
        proofs.append(dict(receiver_normal=list(support.normal), gap_bounds=json_value(bounds),
            whole_interval_bounds=json_value(whole)))
    from tectonics.geometric_native_buried_support import native_pullback
    from tectonics.geometric_intrinsic_geometry import RotationIncrement
    endpoint = native_intersection(receiver.geometry.rotated(receiver_omega, duration),
                                   donor.geometry.rotated(donor_omega, duration))
    assert endpoint is not None
    pulled = native_pullback(endpoint, (RotationIncrement(donor_omega, duration),))
    assert native_intersection(patch, pulled) is None
    return dict(format="native-positive-swept-only-original-material-witness-1",
        donor_fragment_id=donor.fragment_id, receiver_fragment_id=receiver.fragment_id,
        donor_geometry=native_polygon_to_dict(donor.geometry),
        receiver_geometry=native_polygon_to_dict(receiver.geometry),
        donor_omega=list(donor_omega), receiver_omega=list(receiver_omega), dt_myr=duration,
        original_patch=native_polygon_to_dict(patch), original_patch_area_sr_bounds=json_value(patch.area_bounds()),
        chart="p=(x,-y,1)", x_bounds=json_value(x), y_bounds=json_value(y),
        sample_time_fractions=json_value(times), full_receiver_plane_proofs=proofs,
        order=24, precision=256, all_patch_points_inside_receiver_at_middle_time=True,
        all_patch_points_outside_receiver_owner_at_endpoint=True,
        patch_disjoint_from_actual_endpoint_loss=True,
        actual_endpoint_donor_domain=native_polygon_to_dict(pulled),
        raw_endpoint_donor_still_covers_moved_patch=True,
        extra_material_vacancy_if_once_only_archive_adopted=True,
        global_endpoint_loss_equals_complete_swept_loss=False,
        physical_first_entry_admission_requires_separate_authentication=True,
        material_allocation_adopted=False, global_inventory_updated=False,
        geometry_admission_authorized=False)


def run_control(out, *, reuse_force=None):
    started = time.perf_counter()
    out.mkdir(parents=True, exist_ok=False)
    before = sources()
    write(out / "sources_before.json", before)
    force_dir = out / "original_force"
    if reuse_force is None:
        force_dir.mkdir()
        force_report = run_force_control(force_dir)
    else:
        shutil.copytree(reuse_force, force_dir)
        force_report = json.loads((force_dir / "report.json").read_bytes())
    prepared = json.loads((force_dir / "initial.json").read_bytes())
    candidate = json.loads((force_dir / "force_candidate.json").read_bytes())
    support = json.loads((force_dir / "constitutive_time_support.json").read_bytes())
    probe_path = ROOT / "analysis/geometric_native_underthrust_validation/force_control_release_02/force_candidate_refusal_probe.json"
    probe_bytes = probe_path.read_bytes()
    probe = json.loads(probe_bytes)
    assert all(candidate[k] == v for k, v in probe["candidate"].items())
    assert candidate["admission"] == probe["admission"]
    surface = native_surface_from_dict(prepared["current_surface"])
    fragments = {f.fragment_id: f for f in surface.fragments}
    omega = {owner: tuple(values) for owner, values in candidate["omega_rad_per_myr"]}
    duration = candidate["admission"]["dt_myr"]
    old, new = memory_payloads(prepared, candidate, support)
    reference = make_native_material_memory_reference(old, new,
        constitutive_path_sha256=support["path_sha256"])
    write(out / "original_source_surface.json", prepared["current_surface"])
    print("Reconstructing every endpoint pair and all raw vacancy domains", flush=True)
    journal = initialize_native_domain_journal(surface, omega, duration,
        memory_reference=reference)
    raw = native_domain_journal_to_dict(journal)
    loaded = native_domain_journal_from_dict(deepcopy(raw), expected_memory_reference=reference)
    replay = native_domain_journal_to_dict(loaded)
    assert encoded(raw) == encoded(replay)
    write(out / "full_domain_journal.json", raw)
    write(out / "full_domain_journal_replayed.json", replay)
    expected_pairs = {(r["fragment_a"], r["fragment_b"])
                      for r in candidate["admission"]["actual_overlap_assessments"]}
    actual_pairs = {(p.fragment_a, p.fragment_b) for p in journal.endpoint_pairs}
    assert actual_pairs == expected_pairs and len(actual_pairs) == 5
    print("Proving the whole blocking-pair swept donor footprint", flush=True)
    swept = certify_native_swept_domain(fragments["octant:6"].geometry,
        fragments["octant:2"].geometry, omega[1], omega[0], duration)
    swept_body = native_swept_domain_to_dict(swept)
    swept_replay = native_swept_domain_to_dict(native_swept_domain_from_dict(deepcopy(swept_body)))
    assert encoded(swept_body) == encoded(swept_replay)
    write(out / "full_swept_domain.json", swept_body)
    write(out / "full_swept_domain_replayed.json", swept_replay)
    pair, = [p for p in journal.endpoint_pairs
             if (p.fragment_a, p.fragment_b) == ("octant:2", "octant:6")]
    assert not native_subtract(swept.endpoint_donor_domain, pair.donor_b_original_domain)
    assert not native_subtract(pair.donor_b_original_domain, swept.endpoint_donor_domain)
    parent = fragments["octant:6"]
    source_area = parent.geometry.area_bounds()
    swept_area = swept.endpoint_donor_domain.area_bounds()
    fraction = swept_area / source_area
    assert 0 < fraction.lo < fraction.hi < 1
    potential_shares = []
    for name in EXTENSIVE_FIELDS:
        amount = Fraction.from_float(getattr(parent.parcel, name))
        potential_shares.append(dict(reservoir=name, original_amount=json_value(amount),
            potential_archived_amount_bounds=json_value(fraction * amount),
            potential_live_amount_bounds=json_value((FractionInterval(1) - fraction) * amount),
            potential_archive_expression="original_amount * native_area(full_swept_domain) / native_area(original_parent)",
            potential_live_expression="original_amount - potential_archive_amount",
            exact_symbolic_identity="potential_live_amount + potential_archive_amount = original_amount",
            physical_allocation_adopted=False))
    write(out / "full_donor_material_partition.json", dict(
        source_parent_id=parent.fragment_id, material_id=parent.parcel.material_id,
        source_parent_area_sr_bounds=json_value(source_area),
        whole_positive_area_swept_domain_sr_bounds=json_value(swept_area),
        potential_fraction_bounds=json_value(fraction), reservoirs=potential_shares,
        first_entry_times_require_individual_temporal_certificates=True,
        initial_boundary_contact_is_retained_separately=True,
        physical_pairs_and_direction_require_separate_authentication=True,
        mechanical_work_allocated_to_2d_mass=False, material_allocation_adopted=False))
    transient = transient_material_patch(fragments["octant:6"], fragments["octant:3"],
        omega[1], omega[0], duration)
    assert ("octant:3", "octant:6") not in actual_pairs
    write(out / "transient_receiver_patch.json", transient)
    swept_only = swept_only_material_patch(fragments["octant:5"], fragments["octant:1"],
        omega[1], omega[0], duration, swept)
    write(out / "swept_only_material_patch.json", swept_only)
    certificates, queries, replay_queries = [], [], []

    def record(label, receiver_id, point, instants, *, donor_id=None):
        receiver = fragments[receiver_id]
        if donor_id is None:
            cert = certify_native_point_receiver_chronology(receiver.geometry,
                (0., 0., 0.), omega[receiver.parcel.plate], duration, point)
        else:
            donor = fragments[donor_id]
            cert = certify_native_point_chronology(donor.geometry, receiver.geometry,
                omega[donor.parcel.plate], omega[receiver.parcel.plate], duration, point)
        body = native_point_chronology_to_dict(cert)
        reloaded = native_point_chronology_from_dict(deepcopy(body))
        assert encoded(body) == encoded(native_point_chronology_to_dict(reloaded))
        certificates.append(dict(label=label, kind="fixed_world" if donor_id is None else "donor",
            donor_fragment_id=donor_id, receiver_fragment_id=receiver_id, certificate=body))
        occupancy = []
        for instant in instants:
            q = native_point_chronology_query_to_dict(evaluate_native_point_chronology(cert, instant))
            rq = native_point_chronology_query_to_dict(evaluate_native_point_chronology(reloaded, instant))
            assert encoded(q) == encoded(rq)
            item = dict(label=label, time_fraction=json_value(instant), query=q)
            if donor_id is not None:
                item["original_material_age_myr"] = json_value(
                    Fraction.from_float(fragments[donor_id].parcel.age_myr)
                    + instant * Fraction.from_float(duration))
            queries.append(item)
            replay_queries.append({**item, "query": rq})
            occupancy.append(q["occupancy"])
        return occupancy

    for label, point, instants in point_cases(fragments["octant:6"], fragments["octant:2"],
            omega[1], omega[0], duration):
        print("Certifying all receiver planes for " + label, flush=True)
        states = record(label, "octant:2", point, instants, donor_id="octant:6")
        if label in ("main_plane_reentry", "transverse_plane_reentry"):
            assert states == ["outside", "inside", "outside", "inside", "inside"], (label, states)
        elif label == "ordinary_entry":
            assert states == ["outside", "outside", "inside", "inside"], states
        elif label == "transverse_masked_main_entry":
            assert states == ["outside", "outside", "outside", "inside", "inside"], states
        elif label == "no_entry":
            assert states == ["outside"] * 3, states
    masked_label, masked_point, masked_times = point_cases(fragments["octant:6"], fragments["octant:2"],
        omega[1], omega[0], duration)[2]
    assert masked_label == "transverse_masked_main_entry"
    owner_states = [["outside", "outside", "outside", "inside", "inside"]]
    for receiver_id in ("octant:0", "octant:1", "octant:3"):
        current = record(masked_label + ":" + receiver_id, receiver_id, masked_point,
            masked_times, donor_id="octant:6")
        if receiver_id == "octant:3":
            assert current == ["outside", "outside", "inside", "outside", "outside"]
        else:
            assert current == ["outside"] * len(masked_times)
        owner_states.append(current)
    owner_union = ["inside" if any(states[i] == "inside" for states in owner_states) else "outside"
                   for i in range(len(masked_times))]
    assert owner_union == ["outside", "outside", "inside", "inside", "inside"]
    write(out / "receiver_owner_union_queries.json", dict(
        donor_fragment_id="octant:6", receiver_owner=0,
        receiver_fragment_ids=["octant:2", "octant:0", "octant:1", "octant:3"],
        original_point=json_value(masked_point), time_fractions=json_value(masked_times),
        fragment_occupancies=owner_states, owner_occupancies=owner_union,
        internal_fragment_interface_is_not_new_fault_plane=True,
        original_material_pair_history_authenticated=False))
    delta = swept.initial_q_bounds.lo - swept.terminal_q_bounds.hi
    mid_y = (swept.initial_q_bounds.lo + swept.terminal_q_bounds.hi) / 2
    dt_fraction = Fraction.from_float(duration)
    da = tuple(Fraction.from_float(v) * dt_fraction for v in omega[1])
    ra = tuple(Fraction.from_float(v) * dt_fraction for v in omega[0])
    a1 = _normal_coefficients((-1, 0, 0), da, ra, 24)[1][1]
    states = record("swept_only_owner_entry_exit", "octant:1", (a1 * delta / 100, -mid_y, Fraction(1)),
        (Fraction(0), Fraction(1, 4), Fraction(1, 2), Fraction(3, 4), Fraction(1)), donor_id="octant:5")
    assert states == ["outside", "inside", "inside", "outside", "outside"]

    # Every positive raw endpoint vacancy contributes an actual rational
    # interior query. A disposable interval-vector midpoint is a new exact
    # point, checked strictly against every authoritative native support;
    # it never replaces those support planes or the vacancy domain.
    vacancy_occupancies = []
    for index, vacancy in enumerate(journal.raw_vacancies):
        geometry = vacancy.geometry
        rays = [_ray_values(ray.rotated(geometry.pose), 256)[1] for ray in geometry.vertices]
        point = tuple(sum((v[k].midpoint for v in rays), Fraction(0)) for k in range(3))
        ray = NativeRay("seed", point)
        assert all(_side(ray, plane) == 1 for plane in geometry.support_planes)
        states = []
        for fragment_id in fragments:
            label = f"raw_vacancy:{index}:{fragment_id}"
            current = record(label, fragment_id, point, (Fraction(0), Fraction(1)))
            states.append(dict(receiver_fragment_id=fragment_id, occupancy=current))
        assert sum(row["occupancy"][0] == "inside" for row in states) == 1
        assert all(row["occupancy"][1] == "outside" for row in states)
        vacancy_occupancies.append(dict(index=index, exact_point=json_value(point), source_states=states))
    assert vacancy_occupancies
    write(out / "point_certificates.json", certificates)
    write(out / "point_queries.json", queries)
    write(out / "point_queries_replayed.json", replay_queries)
    write(out / "raw_vacancy_point_queries.json", vacancy_occupancies)
    assert (force_dir / "initial.json").read_bytes() == (force_dir / "after_refusal.json").read_bytes()
    after = sources()
    assert before == after and probe_path.read_bytes() == probe_bytes
    write(out / "sources_after.json", after)
    artifacts = {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*")) if p.is_file()}
    report = dict(format="native-full-source-domain-control-1", status="PASS",
        seconds=time.perf_counter() - started, source_count=len(before), sources_unchanged=True,
        original_force_re_solved=reuse_force is None,
        source_probe_sha256=hashlib.sha256(probe_bytes).hexdigest(),
        original_candidate_all_fields_identical=True,
        source_parent_count=len(surface.fragments), endpoint_pair_count=len(actual_pairs),
        raw_endpoint_vacancy_count=len(journal.raw_vacancies), point_certificate_count=len(certificates),
        full_original_owner_components_retained=True,
        complete_original_domain_definitions=True, per_point_temporal_certification=True,
        whole_blocking_pair_positive_area_swept_footprint_proved=True,
        positive_area_transient_material_pair_absent_at_endpoint_proved=True,
        receiving_fragment_switch_distinguished_from_receiver_owner_contact=True,
        positive_area_swept_only_material_missing_endpoint_loss_proved=True,
        raw_gap_equals_once_only_archive_material_vacancy=False,
        complete_swept_material_volume_integrated=False,
        every_original_material_point_history_certified=False,
        true_once_only_archive_material_vacancies_computed=False,
        continuous_hot_birth_law_declared=False, newborn_material_transported=False,
        material_polarities_assumed=False, global_inventory_updated=False,
        accepted_loaded_native_intervals=0, accepted_source_work_j=0,
        candidate_memory_adopted=False, geometry_admission_authorized=False,
        all_endpoint_raw_vacancies_have_strict_interior_queries=True,
        full_journal_byte_exact=True, point_queries_byte_exact=True,
        force_refusal_reasons=force_report["refusal_reasons"], artifacts_sha256=artifacts)
    report["report_sha256"] = hashlib.sha256(encoded(report)).hexdigest()
    write(out / "report.json", report)
    print(json.dumps({k: report[k] for k in ("status", "seconds", "source_parent_count",
        "endpoint_pair_count", "raw_endpoint_vacancy_count", "point_certificate_count",
        "accepted_loaded_native_intervals")}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reuse-force", type=Path,
        help="Development replay only; the final control omits this option")
    args = parser.parse_args()
    run_control(args.out, reuse_force=args.reuse_force)
