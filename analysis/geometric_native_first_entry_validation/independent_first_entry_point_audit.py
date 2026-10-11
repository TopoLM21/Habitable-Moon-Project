"""Independent point and endpoint adapters for the full first-entry audit.

Only stdlib exact arithmetic and the earlier independent arithmetic kernel
are used.  Source bindings, whole swept domains and material budgets must
already have been authenticated by independent_first_entry_audit.  The saved
point proof seals are integrity checks, never geometric authority.
"""
from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
from functools import lru_cache
import importlib.util
import json
from math import gcd, lcm
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "independent_first_entry_point_kernel",
    ROOT / "analysis/geometric_native_full_domain_validation/independent_domain_audit.py")
kernel = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(kernel)
F = Fraction
ZERO, ONE = F(0), F(1)
encoded, digest, rational, interval = kernel.encoded, kernel.digest, kernel.rational, kernel.interval
fdto, idto = kernel.fdto, kernel.idto
MODEL = "explicit_original_seed_direction_once_only_full_first_entry_hot_endpoint"


def _same(saved, expected, message):
    if encoded(saved) != encoded(expected):
        raise ValueError(message)


def _seal(raw, field):
    if raw.get(field) != digest({key: value for key, value in raw.items() if key != field}):
        raise ValueError("Independent point audit rejected inconsistent " + field)


def _fraction_pair(raw):
    value = rational(raw)
    if type(raw) is not list or encoded(raw) != encoded(fdto(value)):
        raise ValueError("Point proof requires a canonical exact rational pair")
    return value


def _proof_types(plane):
    if type(plane["support_index"]) is not int:
        raise ValueError("Point support index must retain its literal integer type")
    zero_order = plane["initial_zero_order"]
    if zero_order is not None and type(zero_order) is not int:
        raise ValueError("Point multiplicity must retain its literal integer type")
    for cell in plane["cells"]:
        if set(cell) != {"time_fraction", "status", "sign", "gap_bounds", "derivative_bounds",
                         "left_gap_bounds", "right_gap_bounds", "root_fraction"}:
            raise ValueError("Point proof leaf has an unsupported schema")
        if type(cell["sign"]) is not int:
            raise ValueError("Point proof leaf sign must retain its literal integer type")
        for key in ("time_fraction", "gap_bounds", "derivative_bounds", "left_gap_bounds",
                    "right_gap_bounds", "root_fraction"):
            bounds = cell[key]
            if bounds is not None:
                interval(bounds)
                _fraction_pair(bounds["lo"])
                _fraction_pair(bounds["hi"])


def _check_extra_bounds(raw, proof):
    """Verify optional derivative witnesses even when the gap alone suffices."""
    rate = sum(map(abs, proof["donor_rate"]), ZERO) + sum(map(abs, proof["receiver_rate"]), ZERO)
    for index, plane in enumerate(raw["plane_chronologies"]):
        info = proof["by_support"][index]
        identity = "nine_exact_initial_derivatives_zero_cayley_hamilton" if info["coincident"] else (
            "exact_initial_zero_factor_and_complete_time_cell_partition")
        if plane["identity_proof"] != identity:
            raise ValueError("Point proof changed its exact initial identity witness")
        for cell in plane["cells"]:
            low, high = interval(cell["time_fraction"])
            if info["coincident"]:
                for name in ("derivative_bounds", "left_gap_bounds", "right_gap_bounds"):
                    if interval(cell[name]) != (ZERO, ZERO):
                        raise ValueError("A coincident analytic plane acquired nonzero derivative evidence")
                if cell["root_fraction"] is not None:
                    raise ValueError("A coincident analytic plane acquired a finite root bracket")
                continue
            for name, a, b, derivative in (
                    ("derivative_bounds", low, high, 1),
                    ("left_gap_bounds", low, low, 0),
                    ("right_gap_bounds", high, high, 0)):
                if cell[name] is not None:
                    bound = kernel.factored_bounds(info["values"], rate, info["mass"],
                                                   info["order"], a, b, derivative)
                    kernel.encloses(cell[name], bound, "optional point " + name)
            if cell["status"] == "no_root" and cell["root_fraction"] is not None:
                raise ValueError("A root-free leaf acquired a hidden saved root bracket")


def _trajectory(parent, point, receiver_polygon, journal, binding, receiver_owner):
    return dict(donor_polygon=parent["geometry"], receiver_polygon=receiver_polygon,
        donor_omega=binding["omega"][parent["parcel"]["plate"]],
        receiver_omega=binding["omega"][receiver_owner], dt_myr=journal["dt_myr"],
        point=[fdto(value) for value in point])


def _audit_owner_plane(raw, parent, point, journal, sweep, binding):
    _seal(raw, "payload_sha256")
    _same(raw["domains"], sweep, "Point owner proof replaced the independently validated whole swept domain")
    expected_keys = {"format", "domains", "donor_fragment_id", "original_point", "max_depth", "max_cells",
        "generator_characteristic_polynomial", "processed_cells", "first_entry_time_fraction", "swept_membership",
        "owner_plane", "receiver_internal_interfaces_do_not_create_new_owner_debits", "boundary_convention",
        "finite_material_measure_assigned_to_point", "physical_adoption", "payload_sha256"}
    if set(raw) != expected_keys or raw["format"] != "native-selected-owner-first-entry-point-1":
        raise ValueError("Unsupported complete owner first-entry point schema")
    _same(raw["original_point"], [fdto(value) for value in point], "Owner proof changed the original exact point")
    if raw["donor_fragment_id"] != parent["fragment_id"]:
        raise ValueError("Owner proof changed its original material parent")
    if (raw["receiver_internal_interfaces_do_not_create_new_owner_debits"] is not True
            or raw["finite_material_measure_assigned_to_point"] is not False or raw["physical_adoption"] is not False
            or raw["boundary_convention"] != "positive_owner_hemisphere_contact_with_original_one_dimensional_interface_of_zero_two_dimensional_measure_retained"):
        raise ValueError("Owner point proof changed its contact/debit/admission law")
    for name, low, high in (("max_depth", 1, 768), ("max_cells", 1, 65536)):
        if type(raw[name]) is not int or not low <= raw[name] <= high:
            raise ValueError("Owner proof lost a typed bounded resource budget")
    if type(raw["processed_cells"]) is not int:
        raise ValueError("Owner proof cell count must retain its literal integer type")
    plane = deepcopy(raw["owner_plane"])
    if set(plane) != {"support_index", "support", "initial_zero_order", "coincident", "cells", "events"}:
        raise ValueError("Unsupported original owner-plane chronology schema")
    _proof_types(plane)
    _same(plane["support"], dict(normal=[-1, 0, 0], rotations=[], label=""),
          "First entry must use the full original overriding owner plane")
    if plane["support_index"] != 0:
        raise ValueError("First-entry plane index changed")
    plane["identity_proof"] = "nine_exact_initial_derivatives_zero_cayley_hamilton" if plane["coincident"] else (
        "exact_initial_zero_factor_and_complete_time_cell_partition")
    receiver_polygon = dict(version="geometric-native-polygons-1", pose=[], vertices=[], supports=[plane["support"]])
    events = deepcopy(plane["events"])
    occupancy = []
    for event in events:
        item = deepcopy(event)
        item.update(plane_event_sequence=event["sequence"], support_indices=[0],
                    plane_event_sequences=[event["sequence"]])
        occupancy.append(item)
    side = -point[0]
    synthesized = dict(format="native-full-point-chronology-1",
        model="complete_all_receiver_halfspace_point_history",
        trajectory=_trajectory(parent, point, receiver_polygon, journal, binding, sweep["receiver_owner"]),
        precision=sweep["precision"], order=sweep["order"], max_depth=raw["max_depth"], max_cells=raw["max_cells"],
        generator_characteristic_polynomial=raw["generator_characteristic_polynomial"],
        plane_chronologies=[plane], plane_events=events, occupancy_events=occupancy,
        initial_occupancy="inside" if side >= 0 else "outside", initial_boundary_supports=[0] if side == 0 else [],
        processed_cells=raw["processed_cells"], boundary_convention="closed_native_receiver_halfspaces_with_explicit_boundary_events",
        admission_authorized=False, global_inventory_updated=False,
        scope="complete_original_point_chronology_not_a_swept_area_or_material_transaction")
    synthesized["payload_sha256"] = digest(synthesized)
    proof = kernel.audit_certificate(synthesized, expected_trajectory=synthesized["trajectory"])
    _check_extra_bounds(synthesized, proof)
    first = next((event for event in proof["all_events"] if event["kind"] == "entry"), None)
    first_window = None if first is None else (first["low"], first["high"])
    saved_window = None if raw["first_entry_time_fraction"] is None else interval(raw["first_entry_time_fraction"])
    if saved_window != first_window:
        raise ValueError("Material archive did not use the first independently proved owner entry")
    if saved_window is not None:
        _same(raw["first_entry_time_fraction"], idto(*first_window), "First-entry window lost its canonical exact bounds")
    membership = "boundary" if plane["coincident"] else "inside" if (
        any(cell["sign"] == 1 for cell in plane["cells"]) or first is not None) else "outside"
    if raw["swept_membership"] != membership:
        raise ValueError("Saved swept membership contradicts complete owner-plane root proof")
    proof.update(first_window=first_window, swept_membership=membership)
    return proof


def _plane_sign(proof, index, time):
    sign = proof["germs"][index]
    for event in proof["all_events"]:
        if event["support_index"] != index:
            continue
        if event["low"] <= time <= event["high"]:
            return None
        if event["high"] < time:
            sign = 1 if event["kind"] == "entry" else -1
    return sign


def _first_receivers(first, receivers):
    window = first["first_window"]
    if window is None:
        return [], "no_first_entry"
    low, high = window
    definite, possible = [], []
    for identity, certificate, proof in receivers:
        signs, unresolved = [], False
        for index, plane in enumerate(certificate["plane_chronologies"]):
            if plane["support"]["normal"] == [-1, 0, 0]:
                signs.append(0)
                continue
            overlaps = any(event["low"] <= high and low <= event["high"]
                           for event in proof["all_events"] if event["support_index"] == index)
            left, right = _plane_sign(proof, index, low), _plane_sign(proof, index, high)
            if not overlaps and left is not None and left == right:
                signs.append(left)
            elif proof["by_support"][index]["coincident"]:
                signs.append(0)
            else:
                signs.append(None)
                unresolved = True
        if any(sign is not None and sign < 0 for sign in signs):
            continue
        possible.append(identity)
        if not unresolved:
            definite.append(identity)
    if not possible:
        raise ValueError("Certified owner first entry lost all original receiving material ancestors")
    return possible, "proved" if possible == definite else "event_window_ambiguous"


def audit_point_context(raw, journal, sweep, material, binding):
    """Authenticate one saved material context and all complete support proofs."""
    _seal(raw, "point_chronology_sha256")
    identity = raw["original_parent_id"]
    if identity not in binding["parents"]:
        raise ValueError("Material point context invented an original parent")
    parent = binding["parents"][identity]
    point = tuple(_fraction_pair(value) for value in raw["original_point"])
    if len(point) != 3 or not any(point) or any(kernel.dot(point, support["normal"]) < 0
            for support in parent["geometry"]["supports"]):
        raise ValueError("Material point left its complete authentic original source domain")
    if (raw["format"] != "native-full-first-entry-material-point-1" or raw["model"] != MODEL
            or raw["material_state_sha256"] != material["state_sha256"]):
        raise ValueError("Material point context replaced its full independently audited material state")
    expected_keys = {"format", "model", "material_state_sha256", "original_parent_id", "original_point",
        "first_entry_certificate", "receiver_chronology", "finite_material_reservoir_assigned_to_point",
        "physical_pair_admission_authorized", "candidate_mechanics_adopted", "point_chronology_sha256"}
    if set(raw) != expected_keys or any(raw[name] is not False for name in (
            "finite_material_reservoir_assigned_to_point", "physical_pair_admission_authorized", "candidate_mechanics_adopted")):
        raise ValueError("Point context assigned finite volume or adopted unresolved mechanical evidence")
    selected = parent["parcel"]["plate"] == sweep["donor_owner"]
    if not selected:
        if raw["first_entry_certificate"] is not None or raw["receiver_chronology"] is not None:
            raise ValueError("Retained overriding material acquired a donor archive history")
        return dict(raw=raw, parent=parent, point=point, owner=None, receivers=[],
                    first_receivers=[], ancestry_status="no_first_entry", leaf_count=0, root_count=0)
    first = _audit_owner_plane(raw["first_entry_certificate"], parent, point, journal, sweep, binding)
    owner_raw = raw["receiver_chronology"]
    _seal(owner_raw, "point_chronology_sha256")
    receiver_ids = [key for key, value in binding["parents"].items()
                    if value["parcel"]["plate"] == sweep["receiver_owner"]]
    expected_owner = dict(format="native-whole-domain-owner-point-chronology-1",
        model=journal["model"], journal_sha256=journal["journal_sha256"],
        source_surface_sha256=journal["source_surface_sha256"], donor_fragment_id=identity,
        receiver_owner=sweep["receiver_owner"], original_point=raw["original_point"],
        receiver_certificates=owner_raw["receiver_certificates"],
        owner_domain="union_of_ALL_original_fragments_on_named_receiver_owner",
        internal_material_boundaries_are_new_faults=False, material_loss_decided=False,
        admission_authorized=False, global_inventory_updated=False)
    expected_owner["point_chronology_sha256"] = digest(expected_owner)
    _same(owner_raw, expected_owner, "Receiver owner wrapper counterfeited original source, clock or adoption scope")
    rows = owner_raw["receiver_certificates"]
    if [row["receiver_fragment_id"] for row in rows] != receiver_ids:
        raise ValueError("First receiving ancestry omitted, duplicated or reordered an original receiving fragment")
    receivers = []
    for row in rows:
        if set(row) != {"receiver_fragment_id", "chronology_certificate"}:
            raise ValueError("Receiver ancestry has an unsupported schema")
        receiver_id, certificate = row["receiver_fragment_id"], row["chronology_certificate"]
        full_keys = {"format", "model", "trajectory", "precision", "order", "max_depth", "max_cells",
            "generator_characteristic_polynomial", "plane_chronologies", "plane_events", "occupancy_events",
            "initial_occupancy", "initial_boundary_supports", "processed_cells", "boundary_convention",
            "admission_authorized", "global_inventory_updated", "scope", "payload_sha256"}
        if (set(certificate) != full_keys or type(certificate["processed_cells"]) is not int
                or certificate["model"] != "complete_all_receiver_halfspace_point_history"
                or certificate["boundary_convention"] != "closed_native_receiver_halfspaces_with_explicit_boundary_events"
                or certificate["scope"] != "complete_original_point_chronology_not_a_swept_area_or_material_transaction"):
            raise ValueError("Receiver point certificate changed its complete independent proof schema or scope")
        expected = _trajectory(parent, point, binding["parents"][receiver_id]["geometry"], journal,
                               binding, sweep["receiver_owner"])
        for name in ("precision", "order"):
            if type(certificate[name]) is not int or certificate[name] != sweep[name]:
                raise ValueError("Receiver point proof changed whole-domain arithmetic resources")
        for name in ("max_depth", "max_cells"):
            if type(certificate[name]) is not int or certificate[name] != raw["first_entry_certificate"][name]:
                raise ValueError("Receiver point proof changed its explicit first-entry proof budget")
        for plane in certificate["plane_chronologies"]:
            _proof_types(plane)
        proof = kernel.audit_certificate(certificate, expected_trajectory=expected)
        _check_extra_bounds(certificate, proof)
        receivers.append((receiver_id, certificate, proof))
    ancestry, status = _first_receivers(first, receivers)
    return dict(raw=raw, parent=parent, point=point, owner=first, receivers=receivers,
                first_receivers=ancestry, ancestry_status=status,
                leaf_count=first["leaf_count"] + sum(proof["leaf_count"] for _, _, proof in receivers),
                root_count=len(first["roots"]) + sum(len(proof["roots"]) for _, _, proof in receivers))


def _material_query(context, time, material, journal):
    raw, parent, first = context["raw"], context["parent"], context["owner"]
    if not 0 <= time <= 1:
        raise ValueError("Material point query left its complete original clock interval")
    window = None if first is None else first["first_window"]
    if first is None:
        allocation, occupancy, debits = "live", "not_a_selected_donor", 0
    else:
        sign = _plane_sign(first, 0, time)
        occupancy = "boundary" if first["swept_membership"] == "boundary" else (
            "event_window" if sign is None else "inside" if sign > 0 else "outside")
        if first["swept_membership"] == "boundary":
            allocation, debits = "boundary", 0
        elif window is None or time < window[0]:
            allocation, debits = "live", 0
        elif time <= window[1]:
            allocation, debits = "event_window", 0
        else:
            allocation, debits = "archived", 1
    traces = material["trace_bindings"]
    original_traces = [row["trace_id"] for row in traces if row["donor_fragment_id"] == parent["fragment_id"]]
    receiver_traces = [row["trace_id"] for row in traces if row["receiver_fragment_id"] in context["first_receivers"]]
    allocation_row = next(row for row in material["parent_allocations"] if row["fragment_id"] == parent["fragment_id"])
    return dict(format="native-full-first-entry-material-point-query-1", model=MODEL,
        material_state_sha256=material["state_sha256"], point_chronology_sha256=raw["point_chronology_sha256"],
        original_parent_id=parent["fragment_id"], original_point=raw["original_point"], time_fraction=fdto(time),
        material_allocation=allocation, contact_occupancy=occupancy,
        first_entry_time_fraction=None if window is None else idto(*window),
        receiver_original_fragment_ids=context["first_receivers"], original_trace_ids=original_traces,
        receiver_trace_ids=receiver_traces, first_entry_receiver_ancestry_status=context["ancestry_status"],
        cohort_id=allocation_row["cohort_id"], material_age_myr=fdto(rational(parent["parcel"]["age_myr"])
            + time * rational(journal["dt_myr"])), material_id=parent["parcel"]["material_id"],
        material_owner=parent["parcel"]["plate"], archive_debits=debits,
        finite_material_reservoir_assigned_to_point=False, physical_pair_ancestry_authenticated=False,
        original_trace_references_preserved=True, donor_receiver_trace_difference_creates_new_fault=False,
        candidate_mechanics_adopted=False, physical_pair_admission_authorized=False)


def audit_material_point_query(raw, context, journal, material):
    time = _fraction_pair(raw["time_fraction"])
    expected = _material_query(context, time, material, journal)
    _same(raw, expected, "Material point query changed first entry, immutable ancestry, archive debit, age or unknown window")
    return expected


def _primitive(point):
    denominator = lcm(*(value.denominator for value in point))
    integers = tuple(value.numerator * (denominator // value.denominator) for value in point)
    divisor = gcd(*integers)
    if divisor <= 0:
        raise ValueError("Endpoint ray lost a nonzero primitive original point")
    return [value // divisor for value in integers]


@lru_cache(maxsize=8)
def _endpoint_rotation_series(rate, order):
    generator, matrices = kernel.skew(rate), [kernel.eye()]
    for degree in range(1, order + 1):
        matrices.append(tuple(tuple(value / degree for value in row)
                              for row in kernel.matmul(matrices[-1], generator)))
    return tuple(matrices)


def _endpoint_fixed_parents(context, journal, binding):
    """Exact Taylor bounds establish every endpoint fixed-world support side."""
    point = context["point"]
    duration = rational(journal["dt_myr"])
    owner = context["parent"]["parcel"]["plate"]
    rate = tuple(rational(value) * duration for value in binding["omega"][owner])
    total_rate = sum(map(abs, rate), ZERO)
    matrices = _endpoint_rotation_series(rate, 96)
    scaled_point = kernel.normalize_scale(point)
    # 96 terms comfortably resolve the actual transverse 1e-55 test feature;
    # any unresolved side is rejected instead of inferred from Decimal signs.
    sides = {}
    for parent in binding["parents"].values():
        for support in parent["geometry"]["supports"]:
            if support["rotations"] or parent["geometry"]["pose"]:
                raise ValueError("Endpoint fixed-world query requires the actual unposed original epoch")
            normal = tuple(support["normal"])
            if normal in sides:
                continue
            scaled_normal = kernel.normalize_scale(normal)
            values = tuple(kernel.dot(scaled_normal, tuple(kernel.dot(row, scaled_point) for row in matrix))
                           for matrix in matrices)
            mass = sum(map(abs, scaled_point), ZERO) * sum(map(abs, scaled_normal), ZERO)
            if not any(values[:9]):
                sides[normal] = 0
            else:
                bounds = kernel.factored_bounds(values, total_rate, mass, 0, ONE, ONE)
                sign = kernel.strict_sign(bounds)
                if not sign:
                    raise ValueError("Independent endpoint support incidence remained unresolved")
                sides[normal] = sign
    parents = [identity for identity, parent in binding["parents"].items()
               if all(sides[tuple(support["normal"])] >= 0 for support in parent["geometry"]["supports"])]
    if not parents:
        raise ValueError("Exact endpoint ray left the complete original fixed-world partition")
    return parents


def audit_endpoint_material_query(raw, context, journal, sweep, material, binding):
    parent, point = context["parent"], context["point"]
    query = _material_query(context, ONE, material, journal)
    endpoint_ray = dict(kind="seed", reference=_primitive(point), rotations=[dict(
        omega_rad_per_myr=binding["omega"][parent["parcel"]["plate"]], dt_myr=journal["dt_myr"])], planes=[], sign=1)
    fixed_parents = _endpoint_fixed_parents(context, journal, binding)
    if (parent["parcel"]["plate"] == sweep["receiver_owner"]
            or query["contact_occupancy"] in ("inside", "boundary")
            or query["material_allocation"] in ("live", "boundary")):
        retained, vacancy = True, False
    elif query["material_allocation"] == "archived" and query["contact_occupancy"] == "outside":
        retained, vacancy = False, True
    else:
        retained, vacancy = None, None
    births = []
    if vacancy is True:
        births = [row["newborn"]["material_id"] for row in material["endpoint_vacancies"]
                  if row["fixed_original_parent_id"] in fixed_parents and row["newborn"] is not None]
        if not births:
            raise ValueError("A complete endpoint material vacancy has no authentic hot material fill")
    expected = dict(format="native-full-first-entry-material-endpoint-query-1", model=MODEL,
        material_state_sha256=material["state_sha256"], point_chronology_sha256=context["raw"]["point_chronology_sha256"],
        original_parent_id=parent["fragment_id"], original_point=context["raw"]["original_point"],
        endpoint_ray=endpoint_ray, point_query=query, raw_coverage=True, retained_material=retained,
        actual_material_vacancy=vacancy, fixed_original_parent_ids=fixed_parents,
        birth_material_ids=births, material_age_myr=query["material_age_myr"],
        physical_pair_admission_authorized=False, candidate_mechanics_adopted=False,
        point_domain="exact endpoint ray of the original donor trajectory",
        raw_coverage_does_not_promote_archived_material_to_live=True)
    _same(raw, expected, "Endpoint material query changed full source ray, retained coverage, vacancy parent, hot fill or force refusal")
    return expected


def audit_first_entry_point_artifacts(control_root, journal_raw, sweep_raw, material_raw, binding):
    """Audit all exported executable point contexts and their material queries."""
    control = Path(control_root)
    certificates = json.loads((control / "point_certificates.json").read_bytes())
    queries = json.loads((control / "material_point_queries.json").read_bytes())
    endpoints = json.loads((control / "endpoint_material_queries.json").read_bytes())
    contexts = {}
    for record in certificates:
        if set(record) != {"label", "certificate"} or type(record["label"]) is not str or record["label"] in contexts:
            raise ValueError("Point material evidence has an unsupported or duplicate trajectory label")
        contexts[record["label"]] = audit_point_context(record["certificate"], journal_raw,
                                                       sweep_raw, material_raw, binding)
    counts, query_labels, duplicate_queries = {}, set(), set()
    for record in queries:
        if set(record) != {"label", "query"} or record["label"] not in contexts:
            raise ValueError("Material point query has no authentic original point context")
        expected = audit_material_point_query(record["query"], contexts[record["label"]], journal_raw, material_raw)
        key = record["label"], encoded(expected["time_fraction"])
        if key in duplicate_queries:
            raise ValueError("Material point query repeated an executable time instead of full evidence")
        duplicate_queries.add(key)
        query_labels.add(record["label"])
        status = expected["material_allocation"]
        counts[status] = counts.get(status, 0) + 1
    if query_labels != set(contexts):
        raise ValueError("A saved first-entry point context lacks its executable material queries")
    endpoint_labels, vacancy_labels = set(), []
    for record in endpoints:
        if (set(record) != {"label", "query"} or record["label"] not in contexts
                or record["label"] in endpoint_labels):
            raise ValueError("Endpoint query duplicated or replaced its authentic source point")
        expected = audit_endpoint_material_query(record["query"], contexts[record["label"]], journal_raw,
                                                 sweep_raw, material_raw, binding)
        endpoint_labels.add(record["label"])
        if expected["actual_material_vacancy"] is True:
            vacancy_labels.append(record["label"])
    if endpoint_labels != set(contexts):
        raise ValueError("A saved material point context lacks its actual endpoint vacancy query")
    if "swept_only_owner_entry_exit" in contexts and "swept_only_owner_entry_exit" not in vacancy_labels:
        raise ValueError("The true swept-only owner contact was promoted back into retained endpoint material")
    if "transverse_masked_main_entry" in contexts:
        context = contexts["transverse_masked_main_entry"]
        if context["first_receivers"] != ["octant:3"] or context["ancestry_status"] != "proved":
            raise ValueError("The 6 to 3 to 2 receiver switch erased first receiving ancestry")
    return dict(point_certificate_count=len(contexts), material_point_query_count=len(queries),
        endpoint_material_query_count=len(endpoints),
        independently_verified_proof_leaf_count=sum(context["leaf_count"] for context in contexts.values()),
        independently_certified_point_root_count=sum(context["root_count"] for context in contexts.values()),
        material_point_allocation_counts=counts, actual_endpoint_material_vacancy_labels=vacancy_labels,
        all_saved_point_complete_time_partitions_independently_proved=True,
        first_owner_entry_precedes_masked_receiver_events=True,
        all_original_receiver_ancestry_at_first_entry_authenticated=True,
        once_only_passive_archive_never_debited_on_exit_or_reentry=True,
        point_unknown_event_windows_preserved=True, exact_original_material_ages_preserved=True,
        endpoint_native_seed_rays_and_full_owner_rotation_authenticated=True,
        all_endpoint_fixed_parent_sides_independently_exact_certified=True,
        raw_covered_swept_only_material_vacancy_and_hot_fill_authenticated=True,
        finite_reservoir_volume_assigned_to_point=False, physical_pair_admission_authorized=False,
        candidate_mechanics_adopted=False, native_force_intervals_accepted=0,
        complete_all_original_material_points_individually_certified=False,
        limitation="Exact point chronology and source-ray endpoint proofs cover every exported individual point executable. Whole original swept domains and reservoir measures are independently audited separately. Receiving-material ancestry does not authorize a physical pair, assign nodal work to volume, adopt a force candidate, or define continuous hot birth.")
