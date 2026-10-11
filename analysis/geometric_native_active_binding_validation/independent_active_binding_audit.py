"""Standard-library audit of the frozen-material active contact reference.

Original literal atom work, contact-line work density and two-dimensional
archive measure are separate ledgers.  Numerical root observations below do
not choose a production transfer clock or admit a native force interval.
"""
from __future__ import annotations

import argparse
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
PREVIOUS = ROOT / "analysis/geometric_native_domain_contact_handover_validation"
PRIOR_CONTROL = PREVIOUS / "control_release_01"
PRECISION = 360
F = Fraction


def load_helper(name, path):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


prior = load_helper("independent_active_binding_prior_domain_audit",
    PREVIOUS / "independent_domain_contact_audit.py")
kernel = load_helper("independent_active_binding_chronology_kernel",
    ROOT / "analysis/geometric_native_full_domain_validation/independent_domain_audit.py")
point_adapter = load_helper("independent_active_binding_extra_leaf_adapter",
    ROOT / "analysis/geometric_native_first_entry_validation/independent_first_entry_point_audit.py")
require, keys, encoded, digest = prior.require, prior.keys, prior.encoded, prior.digest
file_sha, read, same, seal = prior.file_sha, prior.read, prior.same, prior.seal
fraction, rational, interval, exact_float = prior.fraction, prior.rational, prior.interval, prior.exact_float
dec, contains, check_vector = prior.dec, prior.contains, prior.check_vector
dot, cross, sin_cos, rotate = prior.dot, prior.cross, prior.sin_cos, prior.rotate
atan, atan2_positive, half_pi = prior.atan, prior.atan2_positive, prior.half_pi
encloses, relative_file = prior.encloses, prior.relative_file


def verify_prior():
    """Rehash all five sealed releases and all302 saved manifest records."""
    receipt = read(BASE / "starting_validation_receipt.json")
    seal(receipt, "receipt_sha256", "Active-binding initial source authority")
    require(receipt["format"] == "native-active-binding-starting-validation-1"
        and receipt["status"] == "PASS"
        and receipt["original_python_source_count"] == 694
        and receipt["verified_prior_manifest_records"] == 302
        and receipt["prior_tests_reexecuted"] is False
        and receipt["accepted_loaded_native_intervals"] == 0
        and receipt["accepted_source_work_j"] == 0,
        "Active-binding initial validated ancestry differs")
    baseline_path = BASE / "starting_sources.json"
    require(file_sha(baseline_path) == receipt["starting_sources_sha256"],
        "Original694-source byte authority changed")
    record_count, unique_paths = 0, set()
    require(len(receipt["ancestry"]) == 5, "All five preceding releases are required")
    for row in receipt["ancestry"]:
        summary_path, manifest_path = ROOT / row["path"], ROOT / row["manifest_path"]
        require(file_sha(summary_path) == row["sha256"]
            and file_sha(manifest_path) == row["manifest_sha256"],
            "Saved prior release documents changed")
        summary, manifest = read(summary_path), read(manifest_path)
        seal(summary, "summary_sha256", "Prior active-binding release summary")
        seal(manifest, "manifest_sha256", "Prior active-binding artifact manifest")
        require(summary["status"] == "PASS" and manifest["count"] == row["verified_manifest_records"]
            and len(manifest["artifacts"]) == manifest["count"]
            and summary["artifact_manifest"]["sha256"] == file_sha(manifest_path),
            "Prior release status or complete artifact inventory differs")
        for name, record in manifest["artifacts"].items():
            keys(record, "sha256 bytes", "Prior immutable evidence record")
            path = relative_file(ROOT, name)
            require(type(record["bytes"]) is int and path.stat().st_size == record["bytes"]
                and file_sha(path) == record["sha256"],
                "Prior immutable evidence changed: " + name)
            unique_paths.add(name)
        for key in ("fresh_scoped_gate", "release_gate", "actual_control", "independent_audit"):
            if key in summary:
                reference = summary[key]
                require(file_sha(relative_file(ROOT, reference["path"])) == reference["sha256"],
                    "Prior release evidence reference changed: " + key)
        record_count += manifest["count"]
    require(record_count == 302 and len(unique_paths) == receipt["verified_unique_prior_artifacts"],
        "Complete302-record ancestry was not independently verified")
    return dict(prior_manifest_records_verified=record_count,
        verified_unique_prior_artifacts=len(unique_paths), prior_tests_reexecuted=False,
        prior_tests_passed=[770, 253, 178, 265, 341],
        starting_validation_receipt_sha256=file_sha(BASE / "starting_validation_receipt.json"))


def verify_audit_only_source_amendment(control, execution, current):
    """Verify the precise two-file analysis repair after the passing runs.

    The original gate/control maps and their source bodies remain evidence of
    what was executed.  Only this auditor's input adapter and the consolidator
    may differ from those maps; physics, test and control sources cannot.
    """
    path = BASE / "audit_only_source_amendment.json"
    amendment = read(path)
    seal(amendment, "amendment_sha256", "Explicit audit-only source amendment")
    require(amendment["format"] == "native-active-binding-audit-only-source-amendment-1"
        and amendment["status"] == "PASS"
        and amendment["source_count"] == 704 and amendment["original_source_count"] == 694
        and amendment["changed_python_source_count"] == 2 and amendment["unchanged_python_source_count"] == 702
        and amendment["original_694_sources_unchanged"] is True
        and amendment["production_tests_gate_and_control_sources_unchanged"] is True
        and amendment["physics_control_and_tests_reexecuted"] is False,
        "The precise analysis-only source amendment changed scope")
    allowed = tuple("analysis/geometric_native_active_binding_validation/" + name for name in (
        "independent_active_binding_audit.py", "consolidate_active_binding_validation.py"))
    require(set(amendment["changes"]) == set(allowed)
        and set(execution) == set(current) and len(execution) == len(current) == 704,
        "The analysis amendment changed the source inventory or another file")
    changed = sorted(name for name in execution if execution[name] != current[name])
    require(changed == sorted(allowed), "Only the two declared NEW analysis sources may differ")

    def check_reference(reference, expected_path, label):
        keys(reference, "path sha256", label)
        target = relative_file(ROOT, reference["path"])
        require(target == expected_path and file_sha(target) == reference["sha256"],
            label + " raw artifact identity changed")
        return target

    archived_map = check_reference(amendment["execution_frozen_sources"],
        BASE / "source_freeze_attempt_02/frozen_release_sources.json", "Archived execution source map")
    final_map = check_reference(amendment["current_frozen_sources"],
        BASE / "frozen_release_sources.json", "Final analysis-amended source map")
    same(read(archived_map), execution, "Complete original passing-run source map")
    same(read(final_map), current, "Complete final analysis-amended source map")
    old_receipt_path = check_reference(amendment["execution_frozen_receipt"],
        BASE / "source_freeze_attempt_02/frozen_release_receipt.json", "Archived execution freeze receipt")
    old_receipt = read(old_receipt_path)
    seal(old_receipt, "receipt_sha256", "Archived original source-freeze receipt")
    require(old_receipt["status"] == "PASS" and old_receipt["source_count"] == 704
        and old_receipt["original_source_count"] == 694
        and old_receipt["frozen_sources_sha256"] == file_sha(archived_map),
        "Original execution freeze did not authenticate the passing source map")
    gate_path = check_reference(amendment["passing_gate"], BASE / "release_gates_02/receipt.json",
        "Passing original scoped gate")
    report_path = check_reference(amendment["passing_control"], control / "report.json",
        "Passing original actual-source control")
    gate, report = read(gate_path), read(report_path)
    seal(gate, "receipt_sha256", "Original scoped gate receipt")
    seal(report, "report_sha256", "Original actual-source control report")
    require(gate["status"] == report["status"] == "PASS"
        and gate["exit_code"] == 0 and gate["counts"]["passed"] == 136
        and gate["changed_paths"] == [], "The source amendment lacks its original passing gate/control")
    for directory in (gate_path.parent, control):
        for stage in ("before", "after"):
            same(read(directory / ("sources_" + stage + ".json")), execution,
                "Four immutable complete passing execution source maps")
    archive_path = check_reference(amendment["execution_source_archive"],
        BASE / "source_freeze_attempt_02/archive_receipt.json", "Complete original execution source archive")
    archive = read(archive_path)
    seal(archive, "receipt_sha256", "Original execution-source body/archive receipt")
    require(archive["format"] == "native-active-binding-audit-source-archive-1"
        and archive["status"] == "PASS" and archive["source_count"] == 704
        and archive["original_source_count"] == 694 and archive["physics_control_and_tests_reexecuted"] is False
        and set(archive["original_bodies"]) == set(allowed),
        "The original execution source archive changed its exact scope")
    for name in ("execution_frozen_sources", "execution_frozen_receipt", "passing_gate", "passing_control"):
        same(archive[name], amendment[name], "Amendment's original source archive reference: " + name)
    require(len(archive["execution_maps"]) == 4, "The source archive omitted an original execution source map")
    expected_maps = [directory / ("sources_" + stage + ".json")
        for directory in (gate_path.parent, control) for stage in ("before", "after")]
    for reference, expected_path in zip(archive["execution_maps"], expected_maps):
        target = check_reference(reference, expected_path, "Archived passing execution source map")
        same(read(target), execution, "Archived passing complete source map")
    for name in allowed:
        change = amendment["changes"][name]
        require(change["before_sha256"] == execution[name]
            and change["after_sha256"] == current[name]
            and type(change["purpose"]) is str and change["purpose"],
            "A declared analysis-only source revision changed its exact bytes or purpose")
        old_body = check_reference(change["archived_before_body"],
            BASE / "source_freeze_attempt_02" / (Path(name).name + ".txt"),
            "Archived exact original analysis-source body")
        record = archive["original_bodies"][name]
        require(record["path"] == change["archived_before_body"]["path"]
            and record["sha256"] == execution[name] == file_sha(old_body)
            and record["bytes"] == old_body.stat().st_size,
            "The source archive lost the original passing-run analysis body")
    projection = dict(path=path.relative_to(ROOT).as_posix(), sha256=file_sha(path),
        execution_frozen_sources=amendment["execution_frozen_sources"],
        current_frozen_sources=amendment["current_frozen_sources"],
        changed_python_sources=changed, changed_python_source_count=2,
        unchanged_python_source_count=702, original_694_sources_unchanged=True,
        production_tests_gate_and_control_sources_unchanged=True, physics_control_and_tests_reexecuted=False)
    return dict(source_amendment=projection,
        audit_only_source_amendment=dict(path=path.relative_to(ROOT).as_posix(), sha256=file_sha(path)),
        execution_frozen_sources_sha256=file_sha(archived_map), current_frozen_sources_sha256=file_sha(final_map),
        amended_analysis_python_sources=changed, unmodified_execution_python_source_count=702,
        production_tests_gate_and_control_sources_unchanged=True, physics_control_and_tests_reexecuted=False)


def check_control_sources(control, report):
    inventory = report["artifacts_sha256"]
    require(type(inventory) is dict and inventory, "Actual active-binding artifact inventory is required")
    for name, expected in inventory.items():
        require(type(name) is str and type(expected) is str
            and file_sha(relative_file(control, name)) == expected,
            "Actual saved active-binding evidence changed: " + str(name))
    original = read(BASE / "starting_sources.json")
    require(type(original) is dict and len(original) == 694, "Original694-source map changed")
    for directory in ("release_gates_02", "control_release_01"):
        for stage in ("before", "after"):
            same(original, read(PREVIOUS / directory / ("sources_" + stage + ".json")),
                "The four authoritative original694-source maps differ")
    before, after = read(control / "sources_before.json"), read(control / "sources_after.json")
    same(before, after, "Actual active-binding complete source freeze")
    current = {path.relative_to(ROOT).as_posix(): file_sha(path)
        for directory in ("tectonics", "tests", "scripts", "analysis")
        for path in sorted((ROOT / directory).rglob("*.py"))}
    provenance = verify_audit_only_source_amendment(control, after, current)
    same(current, read(BASE / "frozen_release_sources.json"), "Complete authoritative release source freeze")
    require(all(current.get(name) == expected for name, expected in original.items()),
        "An original694 Python source changed")
    require(report["original_source_count"] == 694 and report["source_count"] == len(current)
        and report["existing_sources_changed"] == []
        and report["added_python_sources"] == sorted(set(current) - set(original)),
        "Actual active-binding source count or preserved lineage differs")
    expected_pairs = (("source_history_before.json", "source_history_after.json"),
        ("source_checkpoint_before.json", "source_checkpoint_after.json"),
        ("force_candidate_before.json", "force_candidate_after.json"),
        ("active_paths.json", "active_paths_replayed.json"),
        ("active_queries.json", "active_queries_replayed.json"),
        ("active_segment_references.json", "active_segment_references_replayed.json"))
    same(report["byte_pairs"], [list(pair) for pair in expected_pairs],
        "Actual active-binding complete source/restart byte pairs differ")
    byte_pairs = []
    for first, second in expected_pairs:
        require((control / first).read_bytes() == (control / second).read_bytes(),
            "Actual active-binding strict source/restart bytes differ")
        byte_pairs.append(dict(first=first, second=second, bytes=(control / first).stat().st_size,
            sha256=file_sha(control / first)))
    for name in ("source_history_before.json", "source_checkpoint_before.json", "force_candidate_before.json"):
        require((control / name).read_bytes() == (PRIOR_CONTROL / name).read_bytes(),
            "Complete immutable original actual source bytes changed: " + name)
    history = read(control / "source_history_before.json")
    seal(history, "binding_sha256", "Complete original physical source history")
    checkpoint, candidate = read(control / "source_checkpoint_before.json"), read(control / "force_candidate_before.json")
    require(type(candidate) is dict and len(candidate) == 14, "All fourteen actual candidate fields are required")
    same(history["actual_candidate"], candidate, "All fourteen actual source candidate fields")
    same(history["accepted_before"], checkpoint["native_underthrust_motion"],
        "Original accepted source checkpoint")
    require(history["original_checkpoint_sha256"] == file_sha(control / "source_checkpoint_before.json"),
        "Original actual checkpoint byte identity changed")
    ledger = read(control / "original_quadrature_ledger.json")
    require((control / "original_quadrature_ledger.json").read_bytes()
        == (PRIOR_CONTROL / "original_quadrature_ledger.json").read_bytes(),
        "Original literal16-atom state, area and work ledger changed")
    energy_fields = prior.old.check_original_ledger(ledger, history)
    return history, checkpoint, candidate, current, byte_pairs, energy_fields, dict(inventory), provenance


def source_axes(history):
    dt = exact_float(history["literal_dt_myr"])
    return {owner: tuple(exact_float(value) * dt for value in values)
        for owner, values in history["actual_candidate"]["omega_rad_per_myr"]}


def independent_fresh_owner_bound(history, carrier):
    """Coarse exact positive proof for every event-clock u and owner-clock v.

    The homogeneous carrier avoids an unnecessary square-root approximation.
    Positive division by its norm preserves the certified sign.  For the L1
    induced skew-matrix norm, exp(a)-1 <= a/(1-a); all rotation words use their
    entire saved axis, including the very small transverse components.
    """
    axes = source_axes(history)
    donor, receiver = axes[1], axes[0]
    difference = tuple(a - b for a, b in zip(donor, receiver))
    donor_size, receiver_size = sum(map(abs, donor), F(0)), sum(map(abs, receiver), F(0))
    size = donor_size + receiver_size
    require(0 < size < F(1, 4), "Fresh-owner independent rotation-word bound is unsupported")
    mass = sum(map(abs, carrier), F(0))
    require(len(carrier) == 3 and carrier[0] == 0 and mass > 0, "A fixed homogeneous receiver carrier is required")
    p_error, p_mass = mass * size / (1 - size), mass / (1 - size)
    n_error = receiver_size / (1 - receiver_size)
    error = sum(map(abs, difference), F(0)) * (n_error * p_mass + p_error)
    baseline = difference[2] * carrier[1] - difference[1] * carrier[2]
    lower, upper = baseline - error, baseline + error
    require(lower > 0, "The complete independent (u,v) fresh-owner derivative is not positive")
    return lower, upper


def observe_quarter_root(carrier, receiver_axis, donor_axis, support, window):
    """Observe a saved strict quarter EXIT; never choose a production clock."""
    point = tuple(dec(value) for value in carrier)
    magnitude = dot(point, point).sqrt()
    point = tuple(value / magnitude for value in point)
    receiver, donor = tuple(map(dec, receiver_axis)), tuple(-dec(value) for value in donor_axis)
    normal = tuple(dec(value) for value in support)

    def relative(instant):
        return rotate(rotate(point, receiver, instant), donor, instant)

    low, high = map(dec, interval(window))
    left, right = dot(normal, relative(low)), dot(normal, relative(high))
    require(left >= 0 and right <= 0, "Actual quarter EXIT lacks independent opposite boundary signs")
    for _ in range(1200):
        if high - low < Decimal(10) ** -315:
            break
        middle = (low + high) / 2
        value = dot(normal, relative(middle))
        if value > 0:
            low = middle
        elif value < 0:
            high = middle
        else:
            low = high = middle
            break
    else:
        raise ValueError("Independent quarter-root observation did not converge")
    observed = (low + high) / 2
    return observed, relative(observed)


def geometry_observation(history, receiver_trace_id, donor_trace_id, carrier, frozen_label, instant):
    """True full-SO3 geometry and derivative on one frozen original label."""
    axes = {owner: tuple(map(dec, axis)) for owner, axis in source_axes(history).items()}
    traces = {trace["trace_id"]: trace
        for trace in history["contact_time_path"]["initial_mechanics"]["fault_traces"]["traces"]}
    banks = {bank["trace_id"]: bank for bank in history["material_state"]["trace_bindings"]}
    parameters = {row["trace_id"]: row for row in history["contact_time_path"]["initial_mechanics"]["bending_parameters"]}
    radius = dec(exact_float(history["origin_surface"]["radius_km"]) * 1000)
    carrier = tuple(dec(value) if type(value) is not Decimal else value for value in carrier)
    carrier_norm = dot(carrier, carrier).sqrt()
    carrier = tuple(value / carrier_norm for value in carrier)
    point = tuple(dec(value) if type(value) is not Decimal else value for value in frozen_label)
    point_norm = dot(point, point).sqrt()
    point = tuple(value / point_norm for value in point)
    receiver_owner, donor_owner = banks[receiver_trace_id]["overriding_plate"], banks[donor_trace_id]["subducting_plate"]
    receiver_axis, donor_axis = axes[receiver_owner], axes[donor_owner]
    trace = traces[donor_trace_id]
    start, end = (tuple(dec(exact_float(value)) for value in trace[name]) for name in ("start", "end"))
    inward = tuple(-value for value in cross(start, end))
    beta = atan(dot(inward, point) / (dot(start, point) ** 2 + dot(end, point) ** 2).sqrt())
    width = dec(exact_float(parameters[donor_trace_id]["sub"]["width_m"]))
    distance = radius * beta
    require(0 <= distance < width, "Observed frozen label is outside its own donor Ritz strip")
    z = distance / width
    shape = 1 - Decimal("1.5") * z + z ** 3 / 2
    before = {row[0]: row[1:] for row in history["accepted_before"]["snapshot"]["transverse_displacement_m"]}
    after = {row[0]: row[1:] for row in history["actual_candidate"]["transverse_displacement_m"]}

    def tip(trace_id, bank):
        initial, final = (dec(exact_float(source[trace_id][bank])) for source in (before, after))
        return initial + instant * (final - initial), final - initial

    ws, dws = tip(donor_trace_id, 0)
    wo, dwo = tip(receiver_trace_id, 1)
    ws, dws = shape * ws, shape * dws
    sub_direction, over_direction = rotate(point, donor_axis, instant), rotate(carrier, receiver_axis, instant)
    sub = tuple((radius + ws) * value for value in sub_direction)
    over = tuple((radius + wo) * value for value in over_direction)
    sub_rate = tuple(dws * q + (radius + ws) * v for q, v in zip(sub_direction, cross(donor_axis, sub_direction)))
    over_rate = tuple(dwo * q + (radius + wo) * v for q, v in zip(over_direction, cross(receiver_axis, over_direction)))
    node = next(node for node in history["quadrature_histories"] if node["trace_id"] == receiver_trace_id)
    sine, cosine = tuple(dec(exact_float(value)) for value in node["original_frame"])
    horizontal = rotate(tuple(dec(exact_float(value)) for value in node["original_point"]["horizontal"]), receiver_axis, instant)
    normal = tuple(sine * h + cosine * q for h, q in zip(horizontal, over_direction))
    down = tuple(cosine * h - sine * q for h, q in zip(horizontal, over_direction))
    delta = tuple(b - a for a, b in zip(sub, over))
    delta_rate = tuple(b - a for a, b in zip(sub_rate, over_rate))
    return dict(source_inward_distance_m=distance, source_shape=shape,
        source_sub_displacement_m=ws, receiver_over_displacement_m=wo,
        sub_position=sub, over_position=over, receiver_world_direction=over_direction,
        receiver_world_normal=normal, receiver_world_down_dip=down,
        gap=dot(normal, delta), jump=dot(down, delta),
        gap_rate=dot(cross(receiver_axis, normal), delta) + dot(normal, delta_rate),
        jump_rate=dot(cross(receiver_axis, down), delta) + dot(down, delta_rate))


def iadd(first, second):
    return first[0] + second[0], first[1] + second[1]


def imul(first, second):
    values = [a * b for a in first for b in second]
    return min(values), max(values)


def iscale(value, scale):
    return imul(value, (scale, scale))


def idiv(first, second):
    require(not second[0] <= 0 <= second[1], "Independent interval divisor contains zero")
    values = [a / b for a in first for b in second]
    return min(values), max(values)


def isub(first, second):
    return iadd(first, (-second[1], -second[0]))


def interval_cross(first, second):
    return tuple(isub(imul(first[j], second[k]), imul(first[k], second[j]))
        for j, k in ((1, 2), (2, 0), (0, 1)))


def interval_dot(first, second):
    result = (F(0), F(0))
    for a, b in zip(first, second):
        result = iadd(result, imul(a, b))
    return result


def independent_unit_bounds(carrier):
    norm = kernel._sqrt_rational(sum((value * value for value in carrier), F(0)))
    return tuple(idiv((value, value), norm) for value in carrier)


def independent_pullback_bounds(history, carrier, window, order):
    """Independent exact matrix-series enclosure of Qd^-1 Qr unit(c)."""
    axes = source_axes(history)
    donor, receiver = axes[0], axes[1]
    rate = sum(map(abs, donor), F(0)) + sum(map(abs, receiver), F(0))
    low, high = interval(window)
    scale = max(map(abs, carrier))
    norm = kernel._sqrt_rational(sum((value * value for value in carrier), F(0)))
    values = []
    for index in range(3):
        normal = tuple(F(i == index) for i in range(3))
        coefficients, mass = kernel.coefficients(donor, receiver, normal, carrier, order)
        bound = kernel.factored_bounds(coefficients, rate, mass, 0, low, high)
        values.append(idiv(iscale(bound, scale), norm))
    return tuple(values)


def exact_axis(normal):
    normal = tuple(F(value) if type(value) is int else fraction(value) for value in normal)
    indices = [index for index, value in enumerate(normal) if value]
    require(len(indices) == 1 and abs(normal[indices[0]]) == 1,
        "The original quarter must retain its exact primitive axis supports")
    return indices[0], 1 if normal[indices[0]] > 0 else -1


def check_binding_path(raw, specification, history, law):
    """Reconstruct every trigger from all four independent root partitions."""
    seal(raw, "path_sha256", "Actual receiving-carrier binding path")
    require(raw["format"] == "native-active-receiver-contact-binding-path-1"
        and raw["model"] == "fixed_receiver_quarter_exit_unique_fresh_right_germ_frozen_material",
        "Actual receiving-carrier path changed its explicit selection law")
    same(raw["source_history"], history, "Complete external active-binding source history")
    same(raw["explicit_law"], law, "Complete explicit active-binding constitutive transport law")
    receiver_id = specification["receiver_fragment_id"]
    carrier = tuple(fraction(value) for value in specification["receiver_reference_coordinates"])
    require(len(carrier) == 3 and carrier[0] == 0 and carrier[1] and carrier[2],
        "Single-valued active binding requires one exact open receiving quarter")
    same(raw["receiver_reference_coordinates"], [rational(value) for value in carrier],
        "Actual external fixed receiving material coordinates")
    require(raw["receiver_fragment_id"] == receiver_id, "Actual receiving material identity changed")
    banks = history["material_state"]["trace_bindings"]
    by_donor = {row["donor_fragment_id"]: row for row in banks}
    receiver = next(row for row in banks if row["receiver_fragment_id"] == receiver_id)
    require(raw["receiver_trace_id"] == receiver["trace_id"], "Original receiving contact state was exchanged")
    parents = {row["fragment_id"]: row for row in history["material_state"]["journal"]["source_surface"]["fragments"]}
    require(all(sum(((F(a) if type(a) is int else fraction(a)) * b for a, b in zip(plane["normal"], carrier)), F(0)) >= 0
        for plane in parents[receiver_id]["geometry"]["supports"]),
        "Fixed receiving carrier is outside its original source parent")
    for name, low, high in (("precision", 96, 1024), ("order", 16, 48),
        ("max_depth", 1, 768), ("max_cells", 1, 65536)):
        require(type(raw[name]) is int and low <= raw[name] <= high,
            "Active binding changed its literal bounded proof budget: " + name)
    unit_bounds = independent_unit_bounds(carrier)
    for saved, expected in zip(raw["receiver_unit_direction_bounds"], unit_bounds):
        encloses(saved, expected, "Original exact receiving unit-direction identity")
    decimal_carrier = tuple(map(dec, carrier))
    carrier_norm = dot(decimal_carrier, decimal_carrier).sqrt()
    unit_carrier = tuple(value / carrier_norm for value in decimal_carrier)
    trace = next(row for row in history["contact_time_path"]["initial_mechanics"]["fault_traces"]["traces"]
        if row["trace_id"] == receiver["trace_id"])
    start, end = (tuple(dec(exact_float(value)) for value in trace[name]) for name in ("start", "end"))
    native_angle = atan2_positive(dot(end, decimal_carrier), dot(start, decimal_carrier))
    contains(raw["receiver_native_angle_bounds"], native_angle, "Independent fixed receiving carrier angle")
    memory = next(row for row in history["contact_time_path"]["step"]["previous_memory"]["traces"]
        if row["trace_id"] == receiver["trace_id"])
    computational_angle = native_angle * dec(exact_float(memory["extent_radians"])) / half_pi()
    contains(raw["receiver_computational_angle_bounds"], computational_angle,
        "Independent original computational receiving angle")
    omega = {owner: values for owner, values in history["actual_candidate"]["omega_rad_per_myr"]}
    rows = raw["donor_chronologies"]
    require(type(rows) is list and [row["donor_fragment_id"] for row in rows]
        == [row["donor_fragment_id"] for row in banks],
        "All four original donor quarters must be included exactly once")
    proofs, groups, chronologies = {}, {}, {}
    for row in rows:
        keys(row, "donor_fragment_id chronology", "Complete reverse-quarter chronology")
        identity, chronology = row["donor_fragment_id"], row["chronology"]
        expected = dict(donor_polygon=parents[receiver_id]["geometry"],
            receiver_polygon=parents[identity]["geometry"],
            donor_omega=omega[receiver["overriding_plate"]],
            receiver_omega=omega[receiver["subducting_plate"]],
            dt_myr=history["literal_dt_myr"], point=[rational(value) for value in carrier])
        proof = kernel.audit_certificate(chronology, expected_trajectory=expected)
        point_adapter._check_extra_bounds(chronology, proof)
        proofs[identity], chronologies[identity] = proof, chronology
        for event in proof["all_events"]:
            coordinate, orientation = exact_axis(chronology["plane_chronologies"][event["support_index"]]["support"]["normal"])
            if coordinate == 0 or event["high"] == 0:
                continue
            require(event["kind"] in ("entry", "exit"), "A quarter tangency became a transfer trigger")
            sign = orientation * (1 if event["kind"] == "entry" else -1)
            groups.setdefault((coordinate, event["low"], event["high"]), []).append((identity, sign, event))
    expected_events = []
    for (coordinate, low, high), rows in groups.items():
        require(len(rows) == 4 and len({row[0] for row in rows}) == 4
            and len({row[1] for row in rows}) == 1,
            "The four exact shared-quarter scalar roots lost their common incoming right germ")
        expected_events.append((low, high, coordinate, rows[0][1]))
    expected_events.sort()
    require(all(a[1] < b[0] for a, b in zip(expected_events, expected_events[1:])),
        "Transfer roots lack an independently strict causal ordering")
    require(len(raw["events"]) == len(expected_events)
        and len(raw["segments"]) == len(expected_events) + 1,
        "The unique active path duplicated or omitted a quarter transfer")
    quarter_map = {}
    for identity in by_donor:
        signs = dict(exact_axis(plane["normal"]) for plane in parents[identity]["geometry"]["supports"])
        require(signs[0] == 1, "Original donor owner must retain the complete positive-X hemisphere")
        quarter_map[(signs[1], signs[2])] = identity
    signs = [1 if carrier[index] > 0 else -1 for index in (1, 2)]
    active_donor = quarter_map[tuple(signs)]
    require(active_donor == receiver["donor_fragment_id"], "Initial active attachment replaced its original bank")
    prefix = "active-carrier:" + history["original_checkpoint_sha256"] + ":" + digest([
        receiver_id, [rational(value) for value in carrier]])
    active_id = prefix + ":binding:0"
    previous_window, previous_point = dict(lo=[0, 1], hi=[0, 1]), unit_carrier
    frozen_points, event_observations, consumed = [unit_carrier], [], []
    axes = source_axes(history)
    fresh_bound = independent_fresh_owner_bound(history, carrier)
    for index, (saved, expected_event) in enumerate(zip(raw["events"], expected_events)):
        low, high, coordinate, incoming_sign = expected_event
        require(0 < low < high < 1 and signs[coordinate - 1] == -incoming_sign,
            "A declared transfer is not an interior EXIT of its preceding quarter")
        same(saved["time_fraction_bounds"], dict(lo=rational(low), hi=rational(high)),
            "Actual transfer clock replaced its independently certified first quarter root")
        old_donor = active_donor
        signs[coordinate - 1] = incoming_sign
        new_donor = quarter_map[tuple(signs)]
        require(saved["sequence"] == index + 1 and saved["old_donor_fragment_id"] == old_donor
            and saved["new_donor_fragment_id"] == new_donor
            and saved["old_donor_trace_id"] == by_donor[old_donor]["trace_id"]
            and saved["new_donor_trace_id"] == by_donor[new_donor]["trace_id"]
            and saved["seam_coordinate_index"] == coordinate and saved["incoming_sign"] == incoming_sign,
            "Actual transfer donor or incoming material germ changed")
        next_id = prefix + ":binding:" + str(index + 1)
        require(saved["consumed_binding_id"] == active_id and saved["active_binding_id"] == next_id
            and active_id not in consumed, "An active contact history was duplicated or consumed twice")
        old_plane = next(plane["normal"] for plane in parents[old_donor]["geometry"]["supports"]
            if exact_axis(plane["normal"])[0] == coordinate)
        root_time, frozen = observe_quarter_root(carrier, axes[0], axes[1], old_plane,
            saved["time_fraction_bounds"])
        frozen = tuple(Decimal(0) if k == coordinate else value for k, value in enumerate(frozen))
        independent_bounds = independent_pullback_bounds(history, carrier, saved["time_fraction_bounds"], raw["order"])
        for k, (bounds, expected) in enumerate(zip(saved["donor_reference_coordinates_bounds"], independent_bounds)):
            if k == coordinate:
                require(interval(bounds) == (F(0), F(0)), "Exact seam incidence was replaced by a boxed selected label")
            else:
                encloses(bounds, expected, "Independent exact frozen material pullback at the true root")
                contains(bounds, frozen[k], "Observed actual frozen event material coordinate")
        require(interval(saved["donor_reference_coordinates_bounds"][0])[0] > 0,
            "A seam transfer assigned already absent incoming material")
        require(saved["exact_seam_incidence_from_source_chronology"] is True
            and saved["first_entry_proved"] is True and saved["same_material_reentry"] is False,
            "A seam transfer duplicated a reentered material label")
        derivative = interval(saved["fresh_owner_depth_derivative_bounds"])
        require(derivative[0] > 0 and fresh_bound[0] > 0,
            "Fresh incoming material lacks an independent whole-clock positive first-entry proof")
        for instant in (Decimal(0), root_time / 2, root_time):
            normal = rotate((Decimal(1), Decimal(0), Decimal(0)), tuple(map(dec, axes[0])), instant)
            moved = rotate(frozen, tuple(map(dec, axes[1])), instant)
            value = dot(tuple(dec(a - b) for a, b in zip(axes[1], axes[0])), cross(normal, moved))
            contains(saved["fresh_owner_depth_derivative_bounds"], value,
                "Independent observed incoming first-owner derivative")
        incoming_proof = proofs[new_donor]
        plane_event = next(event for event in incoming_proof["all_events"]
            if event["low"] == low and event["high"] == high and event["kind"] == "entry"
            and exact_axis(chronologies[new_donor]["plane_chronologies"][event["support_index"]]["support"]["normal"])[0] == coordinate)
        occupancy = next(event for event in incoming_proof["occupancy_events"]
            if interval(event["time_fraction"]) == (low, high) and event["kind"] == "entry")
        require(saved["incoming_chronology_ref"] == new_donor
            and saved["incoming_chronology_event_sequence"] == occupancy["sequence"]
            and saved["incoming_plane_event_sequence"] == plane_event["sequence"]
            and saved["incoming_occupancy_event_sequence"] == occupancy["sequence"],
            "Transfer used a masked plane root instead of the actual incoming donor occupancy")
        segment = raw["segments"][index]
        require(segment["sequence"] == index and segment["donor_fragment_id"] == old_donor
            and segment["donor_trace_id"] == by_donor[old_donor]["trace_id"]
            and segment["active_binding_id"] == active_id
            and segment["anchor_event_sequence"] == (None if index == 0 else index),
            "The preceding frozen donor segment was replaced or reset")
        same(segment["start_time_fraction_bounds"], previous_window, "Frozen segment start clock")
        same(segment["end_time_fraction_bounds"], saved["time_fraction_bounds"], "Frozen segment end clock")
        check_vector(segment["frozen_reference_coordinates_bounds"], previous_point,
            "Preceding material label remained frozen through its full segment")
        require(segment["donor_material_label_frozen"] is True,
            "A changing donor segment was continuously reprojected")
        consumed.append(active_id)
        active_donor, active_id = new_donor, next_id
        previous_window, previous_point = saved["time_fraction_bounds"], frozen
        frozen_points.append(frozen)
        event_observations.append(dict(time=root_time, point=frozen, event=saved))
    final = raw["segments"][-1]
    require(final["sequence"] == len(expected_events) and final["donor_fragment_id"] == active_donor
        and final["donor_trace_id"] == by_donor[active_donor]["trace_id"]
        and final["active_binding_id"] == active_id
        and final["anchor_event_sequence"] == (None if not expected_events else len(expected_events))
        and final["donor_material_label_frozen"] is True,
        "The final receiving contact lost its unique frozen attachment")
    same(final["start_time_fraction_bounds"], previous_window, "Final frozen segment start")
    same(final["end_time_fraction_bounds"], dict(lo=[1, 1], hi=[1, 1]), "Final original source clock")
    check_vector(final["frozen_reference_coordinates_bounds"], previous_point, "Final frozen original donor label")
    for earlier, later in zip(frozen_points, frozen_points[1:]):
        require(abs(later[0] - earlier[0]) > Decimal(10) ** -290,
            "A same-donor return reused an already consumed original material label")
    for name in ("preceding_binding_consumed_once",):
        require(raw[name] is True, "Active path lost once-only transition ownership")
    for name in ("reentry_allocates_again", "receiving_carrier_reprojected_after_switch",
        "root_midpoint_selected", "interface_or_work_weight_assigned_to_point", "all_receiver_angles_scheduled",
        "physical_pair_admission_authorized", "candidate_mechanics_adopted"):
        require(raw[name] is False, "Active carrier path incorrectly grants " + name)
    require(raw["accepted_native_force_intervals"] == 0, "Active carrier path accepted a force interval")
    return dict(raw=raw, carrier=carrier, receiver=receiver, native_angle=native_angle,
        computational_angle=computational_angle, frozen_points=frozen_points,
        event_observations=event_observations, proofs=proofs,
        fresh_owner_independent_bounds=fresh_bound,
        chronology_leaf_count=sum(proof["leaf_count"] for proof in proofs.values()),
        chronology_root_count=sum(len(proof["roots"]) for proof in proofs.values()))


BUDGET_FIELDS = ("source_work_density_bounds_j_m2", "friction_dissipation_density_bounds_j_m2",
    "bulk_energy_increment_density_bounds_j_m2", "contact_energy_increment_density_bounds_j_m2",
    "total_energy_increment_density_bounds_j_m2", "compliance_balance_residual_density_bounds_j_m2")
ACCUMULATED_FIELDS = ("accumulated_segment_work_density_bounds_j_m2",
    "accumulated_segment_dissipation_density_bounds_j_m2",
    "accumulated_segment_bulk_energy_increment_density_bounds_j_m2",
    "accumulated_segment_contact_energy_increment_density_bounds_j_m2",
    "accumulated_segment_total_energy_increment_density_bounds_j_m2",
    "accumulated_segment_compliance_balance_residual_density_bounds_j_m2")
STATE_FIELDS = ("free_gap_bounds_m", "free_jump_bounds_m", "plastic_slip_bounds_m", "cumulative_slip_bounds_m",
    "normal_pressure_bounds_pa", "shear_traction_bounds_pa", "bulk_energy_density_bounds_j_m2",
    "contact_energy_density_bounds_j_m2", "total_energy_density_bounds_j_m2")
SEED_TO_STATE = {
    "source_gap_at_switch_bounds_m": "free_gap_bounds_m",
    "source_jump_at_switch_bounds_m": "free_jump_bounds_m",
    "source_plastic_slip_at_switch_bounds_m": "plastic_slip_bounds_m",
    "source_cumulative_slip_at_switch_bounds_m": "cumulative_slip_bounds_m",
    "initial_bulk_energy_density_bounds_j_m2": "bulk_energy_density_bounds_j_m2",
    "initial_contact_energy_density_bounds_j_m2": "contact_energy_density_bounds_j_m2",
    "initial_total_energy_density_bounds_j_m2": "total_energy_density_bounds_j_m2",
}


def independent_whole_segment_branch(history, carrier, receiver_trace_id, donor_trace_id, reference, end):
    """Coarse exact whole-segment proof, independent of saved narrow boxes."""
    axes = source_axes(history)
    donor, receiver = axes[1], axes[0]
    difference = tuple(a - b for a, b in zip(donor, receiver))
    dsize, rsize = sum(map(abs, donor), F(0)), sum(map(abs, receiver), F(0))
    total = dsize + rsize
    require(dsize + 2 * rsize < F(1, 4), "Independent whole-segment rotation-word budget exceeded")
    c = independent_unit_bounds(carrier)
    mass = sum((max(map(abs, value)) for value in c), F(0))
    change_error = mass * (dsize + 2 * rsize) / (1 - dsize - 2 * rsize)
    derivative_error = sum(map(abs, difference), F(0)) * mass * (
        total / (1 - total) + rsize / ((1 - rsize) * (1 - total)))
    baseline = interval_cross(tuple((value, value) for value in difference), c)
    node = next(row for row in history["quadrature_histories"] if row["trace_id"] == receiver_trace_id)
    sine, cosine = map(exact_float, node["original_frame"])
    horizontal = tuple((exact_float(value), exact_float(value)) for value in node["original_point"]["horizontal"])
    normal = tuple(iadd(iscale(h, sine), iscale(q, cosine)) for h, q in zip(horizontal, c))
    down = tuple(isub(iscale(h, cosine), iscale(q, sine)) for h, q in zip(horizontal, c))
    before = {row[0]: row[1:] for row in history["accepted_before"]["snapshot"]["transverse_displacement_m"]}
    after = {row[0]: row[1:] for row in history["actual_candidate"]["transverse_displacement_m"]}
    ws0, ws1 = (exact_float(source[donor_trace_id][0]) for source in (before, after))
    wo0, wo1 = (exact_float(source[receiver_trace_id][1]) for source in (before, after))
    dws, dwo = ws1 - ws0, wo1 - wo0
    # The own-strip proof licenses 0<=phi<=1; the audit does not need a
    # production enclosure of the specific transcendental shape to prove this.
    phi = (F(0), F(1))
    radial = iadd((exact_float(history["origin_surface"]["radius_km"]) * 1000,) * 2,
        imul(phi, (min(ws0, ws1), max(ws0, ws1))))
    rates = []
    for frame, coefficient in ((normal, cosine), (down, -sine)):
        frame_mass = sum((max(map(abs, value)) for value in frame), F(0))
        projection = interval_dot(frame, baseline)
        projection = (projection[0] - frame_mass * derivative_error,
            projection[1] + frame_mass * derivative_error)
        leading = iscale(isub((dwo, dwo), iscale(phi, dws)), coefficient)
        perturbation = abs(dws) * frame_mass * change_error
        rate = isub(leading, imul(radial, projection))
        rates.append((rate[0] - perturbation, rate[1] + perturbation))
    length = interval(end)[1] - interval(reference["switch_time_fraction_bounds"])[0]
    gap_upper = interval(reference["source_gap_at_switch_bounds_m"])[1] + max(rates[0][1], F(0)) * length
    require(gap_upper <= 0 and rates[1][1] <= 0,
        "The independent entire frozen segment lacks compressive monotone down-dip evolution")
    return dict(gap_upper=gap_upper, jump_rate_upper=rates[1][1])


def initial_source_state(history, path):
    trace_id, theta = path["receiver"]["trace_id"], path["computational_angle"]
    memory = next(row for row in history["contact_time_path"]["step"]["previous_memory"]["traces"]
        if row["trace_id"] == trace_id)
    pieces = [piece for piece in memory["pieces"]
        if dec(exact_float(piece["low_radians"])) <= theta <= dec(exact_float(piece["high_radians"]))]
    require(len(pieces) == 1, "Independent source seed must retain its unique original memory piece")
    fields = prior.dense_source_polynomials(history, trace_id, pieces[0], Decimal(0))
    return {name: prior.poly_value(fields[field], theta) for name, field in (
        ("g", "gap"), ("j", "jump"), ("p", "plastic"), ("c", "cumulative"),
        ("eb", "bulk_energy"), ("ec", "contact_energy"), ("et", "total_energy"))}


def check_work_query(row, path, history, reference_inventory):
    raw, clock = row["query"], row["clock"]
    seal(raw, "query_sha256", "Actual frozen-material path work query")
    require(raw["format"] == "native-active-contact-path-work-query-1"
        and raw["model"] == "sequential_full_rotation_frozen_material_segments_with_once_only_state_transfer",
        "Actual frozen-material path query changed its reference/work model")
    same(raw["binding"], path["raw"], "Actual work query used a different complete binding path")
    same(raw["time_fraction_bounds"], clock, "External exact query clock")
    low, high = interval(clock)
    observed_event = next((event for event in path["event_observations"]
        if event["event"]["time_fraction_bounds"] == clock), None)
    require(low == high or observed_event is not None,
        "Independent audit only observes an exact clock or its complete authenticated event expression")
    current = dec(low) if low == high else observed_event["time"]
    revision = sum(event["event"]["time_fraction_bounds"] == clock
        or interval(event["event"]["time_fraction_bounds"])[1] < low for event in path["event_observations"])
    require(raw["revision"] == revision and len(raw["segments"]) == revision + 1,
        "Actual work query reset, omitted or duplicated a frozen material segment")
    require(raw["active_binding_id"] == path["raw"]["segments"][revision]["active_binding_id"],
        "Actual query selected a different active material binding")
    same(raw["consumed_binding_ids"], [event["consumed_binding_id"] for event in path["raw"]["events"][:revision]],
        "Actual once-only consumed contact-binding lineage")
    require(len(set(raw["consumed_binding_ids"])) == revision
        and raw["active_binding_id"] not in raw["consumed_binding_ids"],
        "An already consumed contact remains active")
    receiver_trace = path["receiver"]["trace_id"]
    trace = next(row for row in history["contact_time_path"]["initial_mechanics"]["fault_traces"]["traces"]
        if row["trace_id"] == receiver_trace)
    kb, _, kc, _, kn, _, mu, cohesion = prior.old.source_column_coefficients(
        trace["columns"], trace["law"], trace["water"])
    require(mu == 0 and cohesion == 0 and trace["law"]["viscosity_pa_s_m"] == 0,
        "Actual audited work requires its explicitly retained zero-strength inviscid source")
    ab, ac = kb * (kn / kc - 1) ** 2, kn ** 2 / kc
    coefficients = (kn, Decimal(0), ab, ac, ab + ac, kn - ab - ac)
    state, cumulative_amounts = initial_source_state(history, path), [Decimal(0)] * 6
    field_checks, geometric_checks, branch_checks = 0, 0, 0
    for sequence, segment in enumerate(raw["segments"]):
        seal(segment, "work_sha256", "Actual frozen contact segment work")
        require(segment["format"] == "native-active-frozen-contact-segment-work-1"
            and segment["model"] == "actual_full_SO3_frozen_material_Ritz_contact_zero_strength_inviscid",
            "Actual segment used a different local constitutive path")
        reference = segment["reference"]
        seal(reference, "reference_sha256", "Actual evolved contact segment reference")
        same(reference, reference_inventory[row["label"], sequence], "Deduplicated complete original contact reference")
        require(segment["reference_sha256"] == reference["reference_sha256"]
            and reference["complete_original_source_history_sha256"] == digest(history),
            "Frozen reference lost its complete external source identity")
        schedule = path["raw"]["segments"][sequence]
        require(reference["receiver_trace_id"] == receiver_trace
            and reference["donor_trace_id"] == schedule["donor_trace_id"]
            and reference["revision"] == sequence,
            "Frozen reference changed the actual receiver/donor state lineage")
        same(reference["receiver_original_direction"], path["raw"]["receiver_reference_coordinates"],
            "Fixed receiver material direction in local reference")
        same(reference["switch_time_fraction_bounds"], schedule["start_time_fraction_bounds"],
            "Reference's implicit first switching clock")
        context = reference["root_proof_context"]
        same(context, dict(format="native-active-contact-path-segment-expression-1",
            binding_path_sha256=path["raw"]["path_sha256"], segment_sequence=sequence,
            event_sequence=None if sequence == 0 else sequence,
            clock_identity="original_exact_zero" if sequence == 0 else "certified_original_donor_quarter_seam_root",
            state_source="original_field_at_exact_zero" if sequence == 0 else "evolved_previous_frozen_material_segment",
            root_midpoint_selected=False), "Full independently authenticated reference clock/state expression")
        source = reference["original_source_field_reference"]
        require(source["trace_id"] == receiver_trace and interval(source["time_fraction_bounds"]) == (F(0), F(0)),
            "Actual active source seed changed its original receiving field or exact zero clock")
        require(source["contact_time_path_sha256"] == history["contact_time_path"]["path_sha256"],
            "Actual active state reloaded a later diagnostic affine source clock")
        if sequence:
            previous = raw["segments"][sequence - 1]
            require(reference["initial_state_kind"] == "validated_previous_actual_segment"
                and reference["initial_state_sha256"] == previous["work_sha256"],
                "Positive-clock transfer reset to a diagnostic original contact state")
            for seed_name, state_name in SEED_TO_STATE.items():
                same(reference[seed_name], previous[state_name], "Exact preserved active-state seed: " + seed_name)
            event = path["raw"]["events"][sequence - 1]
            chronology = next(row["chronology"] for row in path["raw"]["donor_chronologies"]
                if row["donor_fragment_id"] == event["new_donor_fragment_id"])
            require(reference["switch_chronology_sha256"] == chronology["payload_sha256"]
                and reference["switch_event_sequence"] == event["incoming_chronology_event_sequence"],
                "Reference did not consume the true incoming-quarter occupancy event")
        else:
            require(reference["initial_state_kind"] == "original_dense_source_at_zero"
                and reference["initial_state_sha256"] is None
                and reference["switch_chronology_sha256"] is None
                and reference["switch_event_sequence"] is None,
                "Initial full-rotation segment was replaced by a diagnostic affine prefix")
        for name in ("source_expression_clock_replayed_after_reference", "frozen_material_label_reprojected_during_segment",
            "initial_source_work_density_inferred_from_literal_atoms", "candidate_mechanics_adopted", "physical_pair_admission_authorized"):
            require(reference[name] is False, "Frozen reference incorrectly grants " + name)
        require(reference["no_finite_point_area_or_work"] is True,
            "Frozen material reference assigned a point a finite contact measure")
        for name in ("switch_additional_work_density_j_m2", "switch_energy_change_density_j_m2"):
            require(fraction(reference[name]) == 0, "Actual state transfer invented additional work or energy")
        frozen = path["frozen_points"][sequence]
        check_vector(reference["frozen_original_material_coordinates"], frozen,
            "Actual entire-segment frozen original material coordinate")
        start = Decimal(0) if sequence == 0 else path["event_observations"][sequence - 1]["time"]
        finish = current if sequence == revision else path["event_observations"][sequence]["time"]
        end_clock = clock if sequence == revision else path["raw"]["events"][sequence]["time_fraction_bounds"]
        same(segment["current_time_fraction_bounds"], end_clock, "Actual segment's complete original clock")
        same_event = finish == start
        require(segment["at_reference_event"] is same_event, "A same-event reference was replaced by a root midpoint")
        gstart, jstart = state["g"], state["j"]
        first = geometry_observation(history, receiver_trace, schedule["donor_trace_id"],
            path["carrier"], frozen, start)
        last = geometry_observation(history, receiver_trace, schedule["donor_trace_id"],
            path["carrier"], frozen, finish)
        contains(reference["frozen_shape_bounds"], first["source_shape"], "Actual frozen donor's own Ritz shape")
        contains(reference["inward_distance_bounds_m"], first["source_inward_distance_m"],
            "Actual frozen donor's own beam coordinate")
        dg, dj = last["gap"] - first["gap"], last["jump"] - first["jump"]
        if same_event:
            dg = dj = Decimal(0)
        potential = dg * (2 * gstart + dg) / 2
        amounts = [coefficient * potential for coefficient in coefficients]
        state = dict(g=gstart + dg, j=jstart + dj, p=jstart + dj,
            c=state["c"] - dj, eb=state["eb"] + amounts[2],
            ec=state["ec"] + amounts[3], et=state["et"] + amounts[4])
        values = dict(projected_gap_change_bounds_m=dg, projected_jump_change_bounds_m=dj,
            free_gap_bounds_m=state["g"], free_jump_bounds_m=state["j"], plastic_slip_bounds_m=state["p"],
            cumulative_slip_bounds_m=state["c"], normal_pressure_bounds_pa=-kn * state["g"],
            shear_traction_bounds_pa=Decimal(0), bulk_energy_density_bounds_j_m2=state["eb"],
            contact_energy_density_bounds_j_m2=state["ec"], total_energy_density_bounds_j_m2=state["et"],
            free_gap_rate_bounds_m_per_fraction=last["gap_rate"], free_jump_rate_bounds_m_per_fraction=last["jump_rate"],
            plastic_slip_rate_bounds_m_per_fraction=last["jump_rate"],
            cumulative_slip_rate_bounds_m_per_fraction=-last["jump_rate"])
        for name, value in values.items():
            contains(segment[name], value, "Independent actual frozen-material state/rate: " + name)
            field_checks += 1
        for name, value, accumulated_name in zip(BUDGET_FIELDS, amounts, ACCUMULATED_FIELDS):
            cumulative_amounts[BUDGET_FIELDS.index(name)] += value
            contains(segment[name], value, "Independent actual segment reaction work/energy balance: " + name)
            contains(segment[accumulated_name], cumulative_amounts[BUDGET_FIELDS.index(name)],
                "Independent sequential once-only accumulated work/energy: " + accumulated_name)
            field_checks += 2
        require(abs(amounts[0] - amounts[2] - amounts[3] - amounts[1] - amounts[5])
            <= (1 + abs(amounts[0])) * Decimal(10) ** -330,
            "Independent actual normal reaction one-form lost the signed binary64 compliance remainder")
        for name in ("finite_point_area_m2", "finite_point_work_j"):
            require(segment[name] is None, "Actual contact segment assigned an isolated point a finite ledger")
        for name in ("original_affine_diagnostic_work_summed_with_actual_segment_work",
            "original_mechanical_candidate_work_replaced", "candidate_mechanics_adopted", "physical_pair_admission_authorized"):
            require(segment[name] is False, "Actual local work incorrectly grants " + name)
        require(segment["original_literal_area_energy_work_atoms_unchanged"] is True
            and segment["full_saved_rotation_axes_preserved"] is True
            and segment["accepted_loaded_native_intervals"] == 0,
            "Actual local work changed its literal source ledger or adopted a global force interval")
        require(interval(segment["whole_segment_gap_bounds_m"])[1] <= 0
            and interval(segment["whole_segment_jump_rate_bounds_m_per_fraction"])[1] <= 0,
            "Actual local constitutive segment lacks its complete compressive down-dip source branch")
        independent_whole_segment_branch(history, path["carrier"], receiver_trace,
            schedule["donor_trace_id"], reference, end_clock)
        branch_checks += 1
        contains(segment["whole_segment_gap_bounds_m"], state["g"], "Observed endpoint of full compressive branch")
        contains(segment["whole_segment_jump_rate_bounds_m_per_fraction"], last["jump_rate"],
            "Observed endpoint of full monotone down-dip branch")
        geometric_checks += 2
        if same_event and sequence:
            previous = raw["segments"][sequence - 1]
            for name in STATE_FIELDS:
                same(segment[name], previous[name], "Exact state continuity at the common implicit handover: " + name)
            require(all(interval(segment[name]) == (F(0), F(0)) for name in BUDGET_FIELDS),
                "Common-clock state transfer created nonzero work, dissipation or energy")
    for name, value in zip(BUDGET_FIELDS, cumulative_amounts):
        contains(raw["budgets"][name], value, "Independent whole actual path work/energy: " + name)
        field_checks += 1
    for name in ("transfer_work_density_j_m2", "transfer_dissipation_density_j_m2",
        "transfer_energy_increment_density_j_m2"):
        require(fraction(raw[name]) == 0, "Actual binding path created finite transfer work or energy")
    for name in ("original_affine_source_work_included", "original_nodal_ledgers_summed_with_continuous_work",
        "receiving_pair_weighted_contact_integrals_numerically_enclosed", "candidate_mechanics_adopted"):
        require(raw[name] is False, "Actual work path incorrectly grants " + name)
    require(raw["initial_attachment_evolved_from_original_exact_zero"] is True
        and raw["measure"] == "joules_per_square_metre_of_original_interface_area"
        and raw["accepted_native_force_intervals"] == 0
        and raw["finite_point_area_m2"] is None and raw["finite_point_work_j"] is None,
        "Actual work path replaced its original clock or contact measure")
    return dict(state_rate_and_work_observation_count=field_checks,
        whole_segment_branch_count=branch_checks, geometry_endpoint_observation_count=geometric_checks,
        active_revision=revision)


EXPECTED_MUTATIONS = (
    "unknown_path_field", "path_precision_literal_type_changed", "receiver_bank_replaced", "receiver_ray_replaced",
    "donor_chronology_omitted", "transfer_event_omitted", "transfer_event_duplicated", "incoming_donor_changed",
    "seam_material_coordinate_shifted", "fresh_owner_derivative_changed", "preceding_binding_consumed_again",
    "material_reentry_fabricated", "frozen_bank_reprojected", "root_midpoint_fabricated",
    "all_angles_schedule_fabricated", "native_force_acceptance_fabricated",
    "unknown_query_field", "query_clock_replaced", "actual_prefix_segment_omitted", "actual_segment_duplicated",
    "active_binding_replaced", "consumed_lineage_duplicated", "actual_work_budget_changed",
    "signed_compliance_residual_changed", "plastic_memory_changed", "cumulative_memory_changed",
    "stored_energy_changed", "affine_source_work_prefix_fabricated", "literal_and_continuum_ledgers_summed",
    "point_area_fabricated", "transfer_work_fabricated", "unknown_reference_field",
    "source_field_replayed_at_later_switch", "previous_actual_state_replaced", "reference_frozen_material_shifted",
    "reference_own_beam_shape_changed", "reference_free_gap_changed", "reference_plastic_memory_changed",
    "reference_cumulative_memory_changed", "reference_current_material_reprojected",
)


def check_mutations(raw):
    seal(raw, "receipt_sha256", "Actual active-binding semantic mutation refusal receipt")
    require(raw["format"] == "native-active-binding-mutation-refusals-1"
        and raw["count"] == len(EXPECTED_MUTATIONS) == 40
        and [row["label"] for row in raw["mutations"]] == list(EXPECTED_MUTATIONS),
        "Actual40 scientific mutation refusal scope changed")
    for row in raw["mutations"]:
        require(row["rejected"] is True and row["valid_rehashed_declared_checksum"] is True
            and row["exception_type"] in ("ValueError", "TypeError")
            and type(row["changed_path"]) is str and row["changed_path"]
            and type(row["message"]) is str and row["message"]
            and type(row["mutated_dto_sha256"]) is str and len(row["mutated_dto_sha256"]) == 64,
            "A rehashed scientific mutation was not rejected")
    return raw["count"]


def _run_audit(control):
    started = time.perf_counter()
    control = Path(control).resolve()
    report = read(control / "report.json")
    seal(report, "report_sha256", "Actual active-binding source control report")
    require(report["format"] == "native-active-binding-control-1" and report["status"] == "PASS",
        "Actual active-binding control failed or changed its declared scope")
    ancestry = verify_prior()
    history, checkpoint, candidate, current, byte_pairs, energy_fields, artifacts, provenance = check_control_sources(control, report)
    same(read(control / "source_validation_receipt.json"), read(BASE / "starting_validation_receipt.json"),
        "The complete initial external source-validation authority")
    law = read(control / "declared_handover_law.json")
    keys(law, "model provenance", "Explicit reduced receiving-contact transport law")
    require(law["model"] == "receiver_attached_complete_lumped_contact_configuration"
        and type(law["provenance"]) is str and law["provenance"].strip() == law["provenance"] and law["provenance"],
        "Active contact correspondence lacks its explicit constitutive transport provenance")
    same(law, report["law"], "Explicit complete original contact transport law")
    specifications = read(control / "receiver_carriers.json")
    same(specifications, [dict(label="two_changes", receiver_fragment_id="octant:3",
        receiver_reference_coordinates=[[0, 1], [1, 1], [1, 10 ** 55]]),
        dict(label="unchanged", receiver_fragment_id="octant:2",
            receiver_reference_coordinates=[[0, 1], [1, 1], [-1, 1]])],
        "The two real exact fixed receiving carrier controls")
    path_rows = read(control / "active_paths.json")
    require([row["label"] for row in path_rows] == [row["label"] for row in specifications],
        "Actual fixed-receiver controls omitted or duplicated a path")
    paths = {}
    for row, specification in zip(path_rows, specifications):
        keys(row, "label path", "Actual unique receiving-contact path control")
        paths[row["label"]] = check_binding_path(row["path"], specification, history, law)
    changing, unchanged = paths["two_changes"], paths["unchanged"]
    require([segment["donor_fragment_id"] for segment in changing["raw"]["segments"]]
        == ["octant:7", "octant:6", "octant:7"]
        and len(unchanged["raw"]["segments"]) == 1
        and unchanged["raw"]["segments"][0]["donor_fragment_id"] == "octant:6",
        "The genuine7-to6-to7 switch or original unchanged6/2 control changed")
    references = read(control / "active_segment_references.json")
    require(len(references) == 4, "Three changing and one retained references must be owned exactly once")
    inventory = {}
    for row in references:
        keys(row, "label segment_sequence reference", "Deduplicated actual frozen segment reference")
        key = row["label"], row["segment_sequence"]
        require(key not in inventory and key[0] in paths
            and type(key[1]) is int and 0 <= key[1] < len(paths[key[0]]["raw"]["segments"]),
            "An actual frozen state reference was duplicated or attached to a different segment")
        inventory[key] = row["reference"]
    queries = read(control / "active_queries.json")
    expected_clocks = [("two_changes", name) for name in (
        "initial", "before_first", "first", "between", "second", "after_second", "final")]
    expected_clocks += [("unchanged", "initial"), ("unchanged", "final")]
    require([(row["label"], row["clock_label"]) for row in queries] == expected_clocks,
        "The nine actual frozen-material work/state observations changed")
    totals = dict(state_rate_and_work_observation_count=0, whole_segment_branch_count=0,
        geometry_endpoint_observation_count=0)
    for row in queries:
        keys(row, "label clock_label clock query", "Actual frozen material work/state observation")
        checked = check_work_query(row, paths[row["label"]], history, inventory)
        for name in totals:
            totals[name] += checked[name]
    mutations = check_mutations(read(control / "mutation_refusals.json"))
    counts = dict(original_source_count=694, original_candidate_field_count=14, receiving_carrier_count=2,
        active_transfer_event_count=2, actual_path_query_count=9, frozen_segment_reference_count=4,
        mutation_refusal_count=mutations)
    for name, expected in counts.items():
        require(type(report[name]) is int and report[name] == expected,
            "Actual active-binding source-control count changed: " + name)
    for name in ("source_before_after_unchanged", "original_typed_checkpoint_loaded", "original_force_re_solved",
        "history_builder_original_force_re_solved", "original_literal_quadrature_ledger_unchanged",
        "unique_active_binding_per_receiving_carrier", "event_transfer_increments_zero",
        "frozen_original_material_label_used_for_later_geometry", "first_switch_uses_evolved_initial_segment",
        "second_switch_uses_evolved_previous_segment", "source_history_bytes_unchanged",
        "source_checkpoint_bytes_unchanged", "source_candidate_bytes_unchanged"):
        require(report[name] is True, "Actual active-binding control lacks " + name)
    for name in ("original_affine_source_work_included", "all_receiver_angles_scheduled",
        "per_receiving_pair_weighted_contact_integrals_evaluated", "physical_pair_admission_authorized",
        "candidate_mechanics_adopted"):
        require(report[name] is False, "Actual active-binding control incorrectly grants " + name)
    require(report["accepted_loaded_native_intervals"] == 0 and report["accepted_source_work_j"] == 0,
        "Local contact reference path was adopted as a global native force solution")
    require(not any(name == "tectonics" or name.startswith("tectonics.") for name in sys.modules),
        "The independent active-binding audit imported production modules")
    artifacts["report.json"] = file_sha(control / "report.json")
    return dict(format="native-active-binding-independent-audit-1", status="PASS",
        seconds=time.perf_counter() - started, decimal_precision=PRECISION,
        standard_library_only=True, production_modules_imported=False,
        scope="independent exact complete reverse-quarter partitions and unique incoming germs for two fixed receiving carriers; exact whole-clock positive fresh-owner and whole-segment compressive monotone branch bounds; independent360Decimal actual frozen-label SO3 geometry, inherited p/c/energy and reaction-work density from original zero; once-only sequential transfer and strict replay; weighted receiving-pair line integrals and native force admission remain pending",
        original_contexts_reexecuted=False, prior_original_force_fields_preserved=14,
        receiving_carrier_count=2, active_transfer_event_count=2, actual_path_query_count=9,
        frozen_segment_reference_count=4, original_node_count=16,
        complete_reverse_chronology_count=8,
        chronology_leaf_count=sum(path["chronology_leaf_count"] for path in paths.values()),
        chronology_root_count=sum(path["chronology_root_count"] for path in paths.values()),
        true_quarter_root_observation_count=2, production_root_clock_selected=False,
        independent_whole_clock_fresh_owner_proved=True,
        frozen_original_material_label_used_for_later_geometry=True,
        first_switch_uses_evolved_initial_segment=True, second_switch_uses_evolved_previous_segment=True,
        initial_actual_attachment_evolved_from_exact_zero=True,
        signed_literal_compliance_residual_preserved=True, event_transfer_increments_zero=True,
        original_affine_source_work_included=False, all_receiver_angles_scheduled=False,
        per_receiving_pair_weighted_contact_integrals_evaluated=False,
        original_literal_quadrature_ledger_unchanged=True, original_energy_work_field_count=energy_fields,
        mutation_refusal_count=mutations, recorded_mutation_refusals_verified=True,
        mutation_tests_independently_reexecuted=False,
        original_source_count=694, source_count=len(current), existing_sources_changed=[],
        source_before_after_unchanged=True, exhaustive_current_source_map_verified=True,
        byte_pairs=byte_pairs, candidate_mechanics_adopted=False, physical_pair_admission_authorized=False,
        accepted_loaded_native_intervals=0, accepted_source_work_j=0,
        audit_source_sha256=file_sha(Path(__file__)), input_artifact_sha256=artifacts,
        **ancestry, **totals, **provenance)


def run_audit(control):
    with localcontext() as context:
        context.prec = PRECISION
        return _run_audit(control)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    receipt = run_audit(arguments.control)
    receipt["receipt_sha256"] = digest(receipt)
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    with arguments.out.open("xb") as stream:
        stream.write(encoded(receipt) + b"\n")
    print(json.dumps({name: receipt[name] for name in ("status", "seconds", "source_count",
        "active_transfer_event_count", "actual_path_query_count", "mutation_refusal_count",
        "chronology_leaf_count", "chronology_root_count")}, sort_keys=True))


if __name__ == "__main__":
    main()
