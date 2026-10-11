"""Whole owner first-entry geometry retains swept-only original material.

The saved positive patch is a separate geometric lower bound, and the old
complete 6 -> 2 proof is an independently constructed comparison domain.
Neither endpoint occupancy nor a polygon fitted to the curved envelope is a
substitute for the whole first-entry sweep.
"""
from copy import deepcopy
from dataclasses import replace
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

import pytest

from tectonics.geometric_native_domain_journal import (
    native_domain_journal_from_dict,
)
from tectonics.geometric_native_polygons import (
    native_polygon_from_dict, native_subtract,
)
from tectonics.geometric_native_pose import FractionInterval as I
from tectonics.geometric_native_surface import NativeSurfaceState
from tectonics.geometric_native_swept_domain import native_swept_domain_from_dict
from tectonics.geometric_native_first_entry_domains import (
    certify_native_first_entry_domains, certify_native_first_entry_point,
    evaluate_native_first_entry_point, native_first_entry_domains_from_dict,
    native_first_entry_domains_to_dict, validate_native_first_entry_domains,
    native_first_entry_point_from_dict, native_first_entry_point_to_dict,
    native_first_entry_point_query_from_dict, native_first_entry_point_query_to_dict,
    validate_native_first_entry_point,
)


ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "analysis/geometric_native_full_domain_validation/control_release_01"


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def reseal(raw):
    """An attacker can recompute the checksum but cannot recompute the proof."""
    body = {key: value for key, value in raw.items() if key != "payload_sha256"}
    raw["payload_sha256"] = hashlib.sha256(encoded(body)).hexdigest()
    return raw


def interval(raw):
    return I(Fraction(*raw["lo"]), Fraction(*raw["hi"]))


def independent_curved_extra_area(journal):
    """Decimal small-chart oracle using a separately built Rodrigues series.

    This is a numerical comparison, not the production area certificate.  The
    chart Jacobian differs from one by less than 1e-38 relatively here.  Four
    Gauss nodes integrate the leading degree-five envelope integrand exactly;
    the full-vector higher terms are retained through order twelve.
    """
    def cross(a, b):
        return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2],
                a[0]*b[1]-a[1]*b[0])

    omega = dict(journal.omega_rad_per_myr)
    duration = Fraction.from_float(journal.dt_myr)
    donor = tuple(-Fraction.from_float(v)*duration for v in omega[1])
    receiver = tuple(Fraction.from_float(v)*duration for v in omega[0])
    receiver_terms = [(Fraction(-1), Fraction(0), Fraction(0))]
    for number in range(1, 13):
        receiver_terms.append(tuple(v/number for v in cross(receiver, receiver_terms[-1])))
    coefficients = []
    for number in range(13):
        total = [Fraction(0)]*3
        for left_number in range(number+1):
            term = receiver_terms[number-left_number]
            for power in range(1, left_number+1):
                term = tuple(v/power for v in cross(donor, term))
            total = [a+b for a, b in zip(total, term)]
        coefficients.append(tuple(total))

    with localcontext() as context:
        context.prec = 120
        def dec(value):
            return Decimal(value.numerator)/Decimal(value.denominator)

        c = tuple(dec(-v[0]) for v in coefficients)
        g = tuple(dec(v[1]) for v in coefficients[1:])
        h = tuple(dec(v[2]) for v in coefficients[1:])

        def polynomial(values, u, derivative=0):
            terms = []
            for number, value in enumerate(values[derivative:], derivative):
                factor = math.factorial(number)//math.factorial(number-derivative)
                exponent = number-derivative
                terms.append(value*factor*(u**exponent if exponent else Decimal(1)))
            return sum(terms, Decimal(0))

        def quotient(a, b, u):
            a0, a1, a2 = (polynomial(a, u, n) for n in range(3))
            b0, b1, b2 = (polynomial(b, u, n) for n in range(3))
            return (a0/b0, a1/b0-a0*b1/b0**2,
                a2/b0-2*a1*b1/b0**2-a0*b2/b0**2+2*a0*b1**2/b0**3)

        y = (Decimal(0),)+g
        def envelope(u):
            f, f1, f2 = quotient(y, c, u)
            q, q1, q2 = quotient(h, g, u)
            return -f*f*q1/f1, q+f*q1/f1, 2*q1+f*q2/f1-f*q1*f2/f1**2

        lo, hi = Decimal(0), Decimal(1)
        assert quotient(h, g, lo)[1] > 0 > quotient(h, g, hi)[1]
        for _ in range(300):
            middle = (lo+hi)/2
            if quotient(h, g, middle)[1] > 0:
                lo = middle
            else:
                hi = middle
        turn = (lo+hi)/2
        center, half = (turn+1)/2, (1-turn)/2
        a = ((3-2*(Decimal(6)/5).sqrt())/7).sqrt()
        b = ((3+2*(Decimal(6)/5).sqrt())/7).sqrt()
        wa, wb = (18+Decimal(30).sqrt())/36, (18-Decimal(30).sqrt())/36
        result = Decimal(0)
        for node, weight in ((-a, wa), (a, wa), (-b, wb), (b, wb)):
            x_value, _, y_derivative = envelope(center+half*node)
            result += weight*x_value*(-y_derivative)
        result *= half
        f, _, _ = quotient(y, c, Decimal(1))
        q, _, _ = quotient(h, g, Decimal(1))
        _, terminal_y, _ = envelope(Decimal(1))
        result -= f*(q-terminal_y)**2/2
        assert result > 0
        assert max(abs(q), abs(terminal_y)) < Decimal("1e-19")
        return Fraction(result)


@pytest.fixture(scope="module")
def actual():
    paths = tuple(CONTROL / name for name in (
        "full_domain_journal.json", "full_swept_domain.json",
        "swept_only_material_patch.json", "original_force/force_candidate.json",
        "transient_receiver_patch.json"))
    files = {path: path.read_bytes() for path in paths}
    journal, old_sweep, patch, candidate, internal = (json.loads(files[path]) for path in paths)
    journal = native_domain_journal_from_dict(journal)
    return dict(journal=journal, old_sweep=native_swept_domain_from_dict(old_sweep),
        patch=patch, candidate=candidate, internal=internal, files=files,
        parents={f.fragment_id: f for f in journal.source_surface.fragments})


@pytest.fixture(scope="module")
def certificate(actual):
    return certify_native_first_entry_domains(actual["journal"])


@pytest.fixture(scope="module")
def domains(certificate):
    return {row.fragment_id: row for row in certificate.donor_domains}


@pytest.fixture(scope="module")
def swept_only_point(actual, certificate):
    x, y = interval(actual["patch"]["x_bounds"]), interval(actual["patch"]["y_bounds"])
    return certify_native_first_entry_point(certificate, "octant:5",
        ((x.lo+x.hi)/2, -(y.lo+y.hi)/2, Fraction(1)))


@pytest.fixture(scope="module")
def internal_switch_point(actual, certificate):
    x = interval(actual["internal"]["x_bounds"])
    z = interval(actual["internal"]["z_bounds"])
    return certify_native_first_entry_point(certificate, "octant:6",
        ((x.lo+x.hi)/2, Fraction(1), -(z.lo+z.hi)/2))


def test_source_keeps_full_original_partition_and_complete_actual_euler_vectors(actual, certificate):
    assert certificate.journal == actual["journal"]
    assert (certificate.donor_owner, certificate.receiver_owner) == (1, 0)
    expected = {owner: tuple(vector) for owner, vector in
                actual["candidate"]["omega_rad_per_myr"]}
    assert dict(certificate.journal.omega_rad_per_myr) == expected
    assert all(component != 0 for vector in expected.values() for component in vector)
    assert len(certificate.journal.source_surface.fragments) == 8
    assert all(path.read_bytes() == data for path, data in actual["files"].items())


def test_whole_original_donor_hemisphere_has_exactly_four_material_parent_domains(domains):
    assert set(domains) == {"octant:4", "octant:5", "octant:6", "octant:7"}
    assert domains["octant:4"].domain_case == "empty"
    assert domains["octant:5"].domain_case == "endpoint_plus_curved_swept_extra"
    assert domains["octant:6"].domain_case == "endpoint_equals_whole_swept"
    assert domains["octant:7"].domain_case == "endpoint_equals_whole_swept"


@pytest.mark.parametrize("identity", [f"octant:{number}" for number in range(4, 8)])
def test_each_domain_keeps_its_complete_original_source_and_parent_area(actual, domains, identity):
    domain = domains[identity]
    assert domain.source_geometry == actual["parents"][identity].geometry
    assert domain.parent_area_bounds_sr == domain.source_geometry.area_bounds()
    assert domain.parent_area_bounds_sr.lo > 1
    assert 0 <= domain.swept_area_bounds_sr.lo <= domain.swept_area_bounds_sr.hi
    assert domain.swept_area_bounds_sr.hi < domain.parent_area_bounds_sr.lo


def test_negative_y_negative_z_material_never_enters_and_has_no_fictitious_endpoint(domains):
    domain = domains["octant:4"]
    assert domain.endpoint_domain is None
    assert domain.endpoint_area_bounds_sr == I(0)
    assert domain.extra_swept_area_bounds_sr == I(0)
    assert domain.swept_area_bounds_sr == I(0)


def test_complete_octant_six_sweep_matches_separate_exact_native_cone_proof(actual, domains):
    domain, old = domains["octant:6"], actual["old_sweep"]
    assert not native_subtract(domain.endpoint_domain, old.endpoint_donor_domain)
    assert not native_subtract(old.endpoint_donor_domain, domain.endpoint_domain)
    assert domain.extra_swept_area_bounds_sr == I(0)
    assert domain.swept_area_bounds_sr.lo <= old.endpoint_area_bounds_sr.hi
    assert old.endpoint_area_bounds_sr.lo <= domain.swept_area_bounds_sr.hi
    assert domain.swept_area_bounds_sr.lo > Fraction(2, 10**7)


def test_positive_y_positive_z_footprint_has_no_unnecessary_curved_archive_increment(domains):
    domain = domains["octant:7"]
    assert domain.endpoint_domain is not None
    assert domain.endpoint_area_bounds_sr.lo > 0
    assert domain.extra_swept_area_bounds_sr == I(0)
    assert domain.swept_area_bounds_sr.lo <= domain.endpoint_area_bounds_sr.hi
    assert domain.endpoint_area_bounds_sr.lo <= domain.swept_area_bounds_sr.hi


def test_curved_sweep_retains_positive_material_absent_from_the_endpoint(actual, domains):
    domain = domains["octant:5"]
    patch = native_polygon_from_dict(actual["patch"]["original_patch"])
    patch_area = patch.area_bounds()
    # This old positive two-dimensional patch is outside the true endpoint,
    # so it gives a separate strict lower bound for the new curved increment.
    assert native_subtract(patch, domain.endpoint_domain)
    from tectonics.geometric_native_polygons import native_intersection
    assert native_intersection(patch, domain.endpoint_domain) is None
    assert domain.extra_swept_area_bounds_sr.lo >= patch_area.hi
    assert domain.swept_area_bounds_sr.lo > domain.endpoint_area_bounds_sr.hi
    assert domain.extra_swept_area_bounds_sr.lo > 0
    assert domain.extra_swept_area_bounds_sr.hi < Fraction(1, 10**60)
    assert domain.swept_area_bounds_sr.lo <= (
        domain.endpoint_area_bounds_sr + domain.extra_swept_area_bounds_sr).hi
    assert (domain.endpoint_area_bounds_sr + domain.extra_swept_area_bounds_sr).lo <= (
        domain.swept_area_bounds_sr.hi)


def test_curved_area_matches_separate_decimal_envelope_integral(actual, domains):
    numerical = independent_curved_extra_area(actual["journal"])
    bounds = domains["octant:5"].extra_swept_area_bounds_sr
    tolerance = abs(numerical)/10**10
    assert bounds.lo-tolerance <= numerical <= bounds.hi+tolerance


def test_saved_trajectory_and_domain_proof_reconstruct_canonical_bytes(certificate):
    raw = native_first_entry_domains_to_dict(certificate)
    rebuilt = native_first_entry_domains_from_dict(deepcopy(raw))
    assert rebuilt == certificate
    assert encoded(native_first_entry_domains_to_dict(rebuilt)) == encoded(raw)
    assert validate_native_first_entry_domains(certificate) == certificate


def test_whole_curved_domain_is_implicit_and_has_no_physical_acceptance(certificate):
    raw = native_first_entry_domains_to_dict(certificate)
    assert raw["envelope"]["represented_as_great_circle_polygon"] is False
    assert "max_{0<u<=1}" in raw["envelope"]["implicit_domain"]
    assert all(raw[name] is False for name in (
        "physical_direction_authenticated", "physical_archive_adopted",
        "native_force_step_accepted", "global_inventory_updated"))
    assert raw["extra_endpoint_fixed_parent_id"] == "octant:5"
    assert all(value.lo > 0 for _, value in certificate.extra_endpoint_parent_support_bounds)


def test_integral_covers_the_whole_curved_branch_and_retains_the_root_window(certificate):
    cells = certificate.quadrature
    assert len(cells) == certificate.integration_cells
    assert cells[0].time_fraction.lo == certificate.maximum_ratio_time_fraction.hi
    assert cells[-1].time_fraction.hi == 1
    assert all(left.time_fraction.hi == right.time_fraction.lo
               for left, right in zip(cells, cells[1:]))
    assert certificate.uncertain_initial_integral_bounds.lo == 0
    assert certificate.uncertain_initial_integral_bounds.hi > 0
    assert certificate.maximum_ratio_bounds.lo > certificate.terminal_ratio_bounds.hi
    assert certificate.terminal_ratio_bounds.lo > certificate.terminal_slope_bounds.hi > 0
    assert certificate.excess_density_bounds.lo > 0
    assert certificate.excess_density_bounds.hi <= 1
    assert certificate.curved_planar_integral_bounds.lo > (
        certificate.endpoint_planar_triangle_bounds.hi)


@pytest.mark.parametrize("damage", [
    "zero_extra", "endpoint_only_case", "dropped_parent", "dropped_integration_cell",
    "missing_root_window", "false_curvature", "polygon_proxy", "adopt_direction",
    "adopt_archive", "adopt_force", "adopt_inventory", "float_precision",
    "bool_owner", "new_domain_claim", "false_fixed_parent", "false_fixed_support"])
def test_resealed_domain_forgery_cannot_replace_the_whole_curved_geometry(certificate, damage):
    raw = deepcopy(native_first_entry_domains_to_dict(certificate))
    donor5 = next(row for row in raw["donor_domains"] if row["fragment_id"] == "octant:5")
    if damage == "zero_extra":
        donor5["extra_swept_area_bounds_sr"] = {"lo": [0, 1], "hi": [0, 1]}
    elif damage == "endpoint_only_case":
        donor5["domain_case"] = "endpoint_equals_whole_swept"
    elif damage == "dropped_parent":
        raw["donor_domains"].pop()
    elif damage == "dropped_integration_cell":
        raw["quadrature"].pop()
    elif damage == "missing_root_window":
        raw["integration"]["uncertain_initial_integral_bounds"] = {
            "lo": [0, 1], "hi": [0, 1]}
    elif damage == "false_curvature":
        raw["proof_bounds"]["curvature_after_maximum_ratio_negative"] = {
            "lo": [1, 1], "hi": [1, 1]}
    elif damage == "polygon_proxy":
        raw["envelope"]["represented_as_great_circle_polygon"] = True
    elif damage.startswith("adopt_"):
        key = {"adopt_direction": "physical_direction_authenticated",
               "adopt_archive": "physical_archive_adopted",
               "adopt_force": "native_force_step_accepted",
               "adopt_inventory": "global_inventory_updated"}[damage]
        raw[key] = True
    elif damage == "float_precision":
        raw["precision"] = float(raw["precision"])
    elif damage == "bool_owner":
        raw["donor_owner"] = True
    elif damage == "false_fixed_parent":
        raw["extra_endpoint_fixed_parent_id"] = "octant:1"
    elif damage == "false_fixed_support":
        raw["extra_endpoint_parent_support_bounds"][0]["bounds"] = {
            "lo": [-1, 1], "hi": [1, 1]}
    else:
        raw["whole_receiver_ancestry_resolved"] = True
    with pytest.raises(ValueError):
        native_first_entry_domains_from_dict(reseal(raw))


@pytest.mark.parametrize("component", [0, 1, 2])
def test_even_one_float_ulp_changed_in_the_full_donor_rotation_requires_a_new_proof(certificate, component):
    raw = deepcopy(native_first_entry_domains_to_dict(certificate))
    donor, = [row for row in raw["journal"]["omega_rad_per_myr"] if row[0] == 1]
    donor[1][component] = math.nextafter(donor[1][component], math.inf)
    journal_body = {key: value for key, value in raw["journal"].items()
                    if key != "journal_sha256"}
    raw["journal"]["journal_sha256"] = hashlib.sha256(encoded(journal_body)).hexdigest()
    with pytest.raises(ValueError):
        native_first_entry_domains_from_dict(reseal(raw))


def test_frozen_domain_cannot_be_forged_into_endpoint_only_material(domains, certificate):
    damaged = replace(domains["octant:5"], extra_swept_area_bounds_sr=I(0),
                      swept_area_bounds_sr=domains["octant:5"].endpoint_area_bounds_sr)
    forged = replace(certificate, donor_domains=tuple(
        damaged if row.fragment_id == damaged.fragment_id else row
        for row in certificate.donor_domains))
    with pytest.raises(ValueError):
        validate_native_first_entry_domains(forged)
    with pytest.raises(AttributeError):
        certificate.donor_owner = 0


@pytest.mark.parametrize("donor_owner,receiver_owner", [
    (True, 0), (1., 0), (1, False), (1, 0.), (0, 1), (1, 1), (2, 0)])
def test_cached_certificate_does_not_admit_wrong_or_numerically_equal_owner_types(
        actual, certificate, donor_owner, receiver_owner):
    with pytest.raises(ValueError):
        certify_native_first_entry_domains(actual["journal"],
            donor_owner=donor_owner, receiver_owner=receiver_owner)


@pytest.mark.parametrize("argument,value", [
    ("precision", True), ("precision", 256.), ("precision", 63), ("precision", 1025),
    ("order", True), ("order", 24.), ("order", 7), ("order", 65),
    ("integration_cells", True), ("integration_cells", 256.),
    ("integration_cells", 0), ("integration_cells", 999999)])
def test_numeric_cache_equality_cannot_bypass_declared_proof_budgets(
        actual, certificate, argument, value):
    with pytest.raises(ValueError):
        certify_native_first_entry_domains(actual["journal"], **{argument: value})


@pytest.mark.parametrize("change", ["posed_parent", "projected_transverse", "changed_clock"])
def test_unsupported_or_changed_original_journal_refuses_before_area_adoption(actual, change):
    journal = actual["journal"]
    if change == "posed_parent":
        source = journal.source_surface
        # A common rotation preserves a genuine partition.  It changes the
        # original epoch rather than making an invalid one-fragment overlap.
        moved = tuple(replace(parent, geometry=parent.geometry.rotated(
            dict(journal.omega_rad_per_myr)[1], journal.dt_myr))
            for parent in source.fragments)
        surface = NativeSurfaceState(source.time_myr, source.radius_km, moved,
            source.known_material_ids, source.phase, source.version)
        forged = replace(journal, source_surface=surface)
    elif change == "projected_transverse":
        projected = tuple((owner, (0., 0., omega[2]))
                          for owner, omega in journal.omega_rad_per_myr)
        forged = replace(journal, omega_rad_per_myr=projected)
    else:
        forged = replace(journal, dt_myr=math.nextafter(journal.dt_myr, math.inf))
    with pytest.raises(ValueError):
        certify_native_first_entry_domains(forged)


def test_swept_only_material_has_two_real_owner_crossings_but_only_one_archive_debit(swept_only_point):
    point = swept_only_point
    assert point.swept_membership == "inside"
    assert [event.kind for event in point.plane_chronology.events] == ["entry", "exit"]
    assert point.first_entry_time_fraction == point.plane_chronology.events[0].time_fraction
    before = evaluate_native_first_entry_point(point, Fraction(0))
    inside = evaluate_native_first_entry_point(point, Fraction(1, 2))
    departed = evaluate_native_first_entry_point(point, Fraction(1))
    assert (before.material_allocation, before.contact_occupancy, before.archive_debits) == (
        "live", "outside", 0)
    assert (inside.material_allocation, inside.contact_occupancy, inside.archive_debits) == (
        "archived", "inside", 1)
    assert (departed.material_allocation, departed.contact_occupancy, departed.archive_debits) == (
        "archived", "outside", 1)


def test_first_entry_window_keeps_material_allocation_unresolved(swept_only_point):
    window = swept_only_point.first_entry_time_fraction
    assert 0 < window.lo < window.hi < 1
    query = evaluate_native_first_entry_point(swept_only_point, (window.lo+window.hi)/2)
    assert query.material_allocation == query.contact_occupancy == "event_window"
    assert query.archive_debits == 0


def test_departure_window_does_not_cancel_a_resolved_first_archive(swept_only_point):
    departure = swept_only_point.plane_chronology.events[-1].time_fraction
    query = evaluate_native_first_entry_point(swept_only_point, (departure.lo+departure.hi)/2)
    assert query.contact_occupancy == "event_window"
    assert (query.material_allocation, query.archive_debits) == ("archived", 1)


def test_internal_receiver_switch_cannot_create_extra_owner_crossings_or_debits(actual, internal_switch_point):
    point = internal_switch_point
    assert point.donor_fragment_id == "octant:6"
    assert [event.kind for event in point.plane_chronology.events] == ["entry"]
    middle_time = Fraction(*actual["internal"]["sample_time_fractions"][1])
    for time in (middle_time, 4*middle_time, Fraction(1)):
        query = evaluate_native_first_entry_point(point, time)
        assert (query.material_allocation, query.contact_occupancy, query.archive_debits) == (
            "archived", "inside", 1)
    raw = native_first_entry_point_to_dict(point)
    assert raw["receiver_internal_interfaces_do_not_create_new_owner_debits"] is True


@pytest.mark.parametrize("identity,point", [
    ("octant:4", (1, -1, -1)), ("octant:5", (1, -1, 1)),
    ("octant:6", (1, 1, -1)), ("octant:7", (1, 1, 1))])
def test_material_far_from_original_owner_interface_remains_live(certificate, identity, point):
    proved = certify_native_first_entry_point(certificate, identity, point)
    assert proved.swept_membership == "outside"
    assert proved.first_entry_time_fraction is None
    assert proved.plane_chronology.events == ()
    query = evaluate_native_first_entry_point(proved, Fraction(1))
    assert (query.material_allocation, query.contact_occupancy, query.archive_debits) == (
        "live", "outside", 0)


def test_original_point_and_its_authoritative_queries_reconstruct_exactly(swept_only_point):
    raw = native_first_entry_point_to_dict(swept_only_point)
    rebuilt = native_first_entry_point_from_dict(deepcopy(raw))
    assert rebuilt == swept_only_point
    assert encoded(native_first_entry_point_to_dict(rebuilt)) == encoded(raw)
    assert raw["finite_material_measure_assigned_to_point"] is False
    assert raw["physical_adoption"] is False
    for time in (Fraction(0), Fraction(1, 2), Fraction(1)):
        query = evaluate_native_first_entry_point(swept_only_point, time)
        saved = native_first_entry_point_query_to_dict(query)
        replayed = native_first_entry_point_query_from_dict(saved, certificate=rebuilt)
        assert replayed == query
        assert encoded(native_first_entry_point_query_to_dict(replayed)) == encoded(saved)


@pytest.mark.parametrize("damage", ["dropped_entry", "dropped_exit", "first_entry_zero",
    "claim_outside", "wrong_original_parent", "float_budget", "bool_rational",
    "false_invariant", "promote_point_area", "adopt_physical"])
def test_resealed_point_cannot_lose_owner_roots_or_override_material_provenance(swept_only_point, damage):
    raw = deepcopy(native_first_entry_point_to_dict(swept_only_point))
    if damage == "dropped_entry":
        raw["owner_plane"]["events"].pop(0)
    elif damage == "dropped_exit":
        raw["owner_plane"]["events"].pop()
    elif damage == "first_entry_zero":
        raw["first_entry_time_fraction"] = {"lo": [0, 1], "hi": [0, 1]}
    elif damage == "claim_outside":
        raw["swept_membership"] = "outside"
    elif damage == "wrong_original_parent":
        raw["donor_fragment_id"] = "octant:1"
    elif damage == "float_budget":
        raw["max_cells"] = float(raw["max_cells"])
    elif damage == "bool_rational":
        raw["original_point"][2][0] = True
    elif damage == "false_invariant":
        raw["generator_characteristic_polynomial"][1] = [0, 1]
    elif damage == "promote_point_area":
        raw["finite_material_measure_assigned_to_point"] = True
    else:
        raw["physical_adoption"] = True
    with pytest.raises(ValueError):
        native_first_entry_point_from_dict(reseal(raw))


@pytest.mark.parametrize("time", [True, False, 0., 0.5, -1, Fraction(4, 3)])
def test_query_requires_exact_typed_time_inside_the_declared_interval(swept_only_point, time):
    with pytest.raises(ValueError):
        evaluate_native_first_entry_point(swept_only_point, time)


@pytest.mark.parametrize("damage", ["unbury_after_exit", "double_debit", "bool_debit",
    "false_contact", "midpoint_promotion", "adopt_physical", "float_time"])
def test_resealed_query_must_follow_authoritative_first_entry_history(swept_only_point, damage):
    query = evaluate_native_first_entry_point(swept_only_point, Fraction(1))
    raw = deepcopy(native_first_entry_point_query_to_dict(query))
    if damage == "unbury_after_exit":
        raw["material_allocation"], raw["archive_debits"] = "live", 0
    elif damage == "double_debit":
        raw["archive_debits"] = 2
    elif damage == "bool_debit":
        raw["archive_debits"] = True
    elif damage == "false_contact":
        raw["contact_occupancy"] = "inside"
    elif damage == "midpoint_promotion":
        window = swept_only_point.first_entry_time_fraction
        time = (window.lo+window.hi)/2
        raw["time_fraction"] = [time.numerator, time.denominator]
    elif damage == "adopt_physical":
        raw["physical_adoption"] = True
    else:
        raw["time_fraction"] = 1.
    with pytest.raises(ValueError):
        native_first_entry_point_query_from_dict(reseal(raw), certificate=swept_only_point)


def test_query_serializer_and_context_guard_reject_frozen_forgery(swept_only_point, internal_switch_point):
    query = evaluate_native_first_entry_point(swept_only_point, Fraction(1))
    with pytest.raises(ValueError):
        native_first_entry_point_query_to_dict(replace(query, archive_debits=2))
    with pytest.raises(ValueError):
        native_first_entry_point_query_to_dict(replace(query, material_allocation="live"))
    saved = native_first_entry_point_query_to_dict(query)
    with pytest.raises(ValueError):
        native_first_entry_point_query_from_dict(saved, certificate=internal_switch_point)
    with pytest.raises(ValueError):
        validate_native_first_entry_point(replace(swept_only_point, swept_membership="outside"))


@pytest.mark.parametrize("argument,value", [
    ("max_depth", True), ("max_depth", 192.), ("max_depth", 0), ("max_depth", 769),
    ("max_cells", True), ("max_cells", 4096.), ("max_cells", 0), ("max_cells", 65537)])
def test_point_cache_cannot_bypass_typed_certification_budgets(
        certificate, swept_only_point, argument, value):
    with pytest.raises(ValueError):
        certify_native_first_entry_point(certificate, swept_only_point.donor_fragment_id,
            swept_only_point.original_point, **{argument: value})


@pytest.mark.parametrize("argument", ["max_cells", "max_depth"])
def test_exhausted_owner_root_budget_fails_closed(certificate, swept_only_point, argument):
    with pytest.raises(ValueError):
        certify_native_first_entry_point(certificate, swept_only_point.donor_fragment_id,
            swept_only_point.original_point, **{argument: 1})


@pytest.mark.parametrize("identity,point", [
    ("octant:1", (1, -1, 1)), ("octant:5", (-1, -1, 1)),
    ("octant:5", (1, 1, 1)), ("octant:5", (1, -1, -1)),
    ("octant:5", (0, 0, 0)), ("octant:5", [1, -1, 1]),
    ("octant:5", (1., -1, 1)), ("octant:5", (True, -1, 1))])
def test_original_donor_point_requires_exact_supported_material_provenance(certificate, identity, point):
    with pytest.raises(ValueError):
        certify_native_first_entry_point(certificate, identity, point)
