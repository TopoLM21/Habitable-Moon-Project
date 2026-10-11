"""Independently check saved physical bindings using only the standard library.

The source checkpoint and original candidate are authorities, not newly sealed
binding records.  Decimal Rodrigues checks are numerical containment checks at
declared rational times; they do not certify a whole time window, solve the
force system, create material pairs, or adopt diagnostic work or memory.
"""
from __future__ import annotations

import argparse
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
PRECISION = 360


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    def reject(value):
        raise ValueError("Nonfinite JSON number: " + value)
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "Duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(path.read_bytes(), parse_constant=reject, object_pairs_hook=unique)


def seal(raw, key, label):
    require(type(raw) is dict and key in raw, label + " lacks " + key)
    body = {name: value for name, value in raw.items() if name != key}
    require(digest(body) == raw[key], label + " selfseal mismatch")
    return body


def fraction(raw):
    require(type(raw) is list and len(raw) == 2
            and all(type(value) is int for value in raw) and raw[1] > 0,
            "Expected a literal rational pair")
    result = Fraction(*raw)
    require([result.numerator, result.denominator] == raw,
            "Rational pair must be reduced and canonical")
    return result


def vector(raw, size=3):
    require(type(raw) is list and len(raw) == size, "Wrong rational vector size")
    return tuple(fraction(value) for value in raw)


def interval(raw):
    require(type(raw) is dict and set(raw) == {"lo", "hi"},
            "Expected a literal rational interval")
    lo, hi = fraction(raw["lo"]), fraction(raw["hi"])
    require(lo <= hi, "Reversed rational interval")
    return lo, hi


def dec(value):
    if type(value) is Fraction:
        return Decimal(value.numerator) / Decimal(value.denominator)
    if type(value) is Decimal:
        return value
    return Decimal(value)


def exact_float(value):
    require(type(value) in (float, int) and math.isfinite(value),
            "Source value must be a finite literal number")
    return Fraction(value)


def dot(a, b):
    return sum((x*y for x, y in zip(a, b)), Decimal(0))


def cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2],
            a[0]*b[1]-a[1]*b[0])


def sin_cos(value):
    """Decimal alternating series, retaining tiny represented rotations."""
    require(abs(value) <= Decimal(4), "Independent trig budget exceeded")
    threshold = Decimal(10) ** -(PRECISION-16)
    values = []
    for sine in (True, False):
        term = value if sine else Decimal(1)
        total = term
        for index in range(10000):
            n = 2*index+(2 if sine else 1)
            term = -term*value*value / Decimal(n*(n+1))
            total += term
            if abs(term) < threshold:
                break
        else:
            raise ValueError("Independent trig series did not converge")
        values.append(total)
    return tuple(values)


def rotate(point, axis, instant):
    """Compute exp([axis] instant) point with unnormalized input labels."""
    angle_vector = tuple(value*instant for value in axis)
    magnitude = dot(angle_vector, angle_vector).sqrt()
    if not magnitude:
        return point
    sine, cosine = sin_cos(magnitude)
    direction = tuple(value/magnitude for value in angle_vector)
    perpendicular = cross(direction, point)
    projection = dot(direction, point)
    return tuple(value*cosine + tangent*sine
                 + unit*projection*(1-cosine)
                 for value, tangent, unit in zip(point, perpendicular, direction))


def contains(raw, value, label):
    lo, hi = interval(raw)
    # This allowance is solely the independent Decimal arithmetic guard. It
    # is much smaller than the exported production interval rounding budget.
    epsilon = Decimal(10) ** -(PRECISION-40) * max(Decimal(1), abs(value))
    require(dec(lo)-epsilon <= value <= dec(hi)+epsilon,
            label + " does not enclose independent numerical value")


def check_vector(raw, values, label):
    require(type(raw) is list and len(raw) == len(values), label + " shape mismatch")
    for index, (bound, value) in enumerate(zip(raw, values)):
        contains(bound, value, label + " component " + str(index))


def pair_source(raw, expected=None):
    seal(raw, "maps_sha256", "Original tip map")
    source = raw["source"]
    require(digest(source) == raw["source_sha256"], "Tip-map source seal mismatch")
    if expected is not None:
        require(encoded(source) == encoded(expected), "Tip-map source differs from original history")
    for name in ("physical_pair_admission_authorized", "candidate_mechanics_adopted",
                 "finite_pose_change_replaces_free_anchors", "material_handover_defined"):
        require(raw[name] is False, "Tip map claims unauthorized " + name)
    require(raw["accepted_work_increment_j"] == [0, 1]
            and raw["accepted_native_force_intervals"] == 0,
            "Tip map adopts force/work")
    require(raw["format"] == "native-original-physical-pair-tip-time-maps-1",
            "Unsupported original tip-map format")
    require(raw["model"] == "literal_original_tip_labels_full_SO3_affine_candidate_radial_diagnostic"
            and raw["scope"] == "original_pair_geometry_only_no_anchor_rebase_no_material_handover",
            "Tip-map model or physical scope changed")
    return source


def check_pair_map(raw, expected=None):
    source = pair_source(raw, expected)
    low, high = interval(source["time_fraction_bounds"])
    require(0 <= low <= high <= 1, "Tip-map time lies outside epoch")
    omega = {owner: vector(values) for owner, values in source["omega_records"]}
    require(len(omega) == len(source["omega_records"])
            and list(omega) == sorted(omega), "Incomplete or noncanonical owner Ω")
    require(source["sub_owner"] != source["over_owner"], "Tip pair has one owner")
    dt = fraction(source["dt_myr"])
    require(dt > 0, "Nonpositive tip-map duration")
    axes = {owner: tuple(dec(value*dt) for value in values) for owner, values in omega.items()}
    point = tuple(dec(value) for value in vector(source["reference_coordinates"]))
    normal = tuple(dec(value) for value in vector(source["original_normal"]))
    down = tuple(dec(value) for value in vector(source["original_down_direction"]))
    before = tuple(dec(value) for value in vector(source["radial_before"], 2))
    candidate = tuple(dec(value) for value in vector(source["radial_candidate"], 2))
    radius = dec(fraction(source["radius_m"]))
    require(radius > 0, "Nonpositive original tip radius")
    reference_gap = (before[1]-before[0])*dot(normal, point)
    reference_jump = (before[1]-before[0])*dot(down, point)
    for name, value in (("reference_gap_m", reference_gap), ("reference_jump_m", reference_jump)):
        require(abs(dec(fraction(raw[name]))-value) < Decimal(10)**-(PRECISION-40),
                name + " changed original literal label")
    instants = sorted({low, (3*low+high)/4, (low+high)/2, (low+3*high)/4, high})
    for instant in instants:
        u = dec(instant)
        sub = rotate(point, axes[source["sub_owner"]], u)
        over = rotate(point, axes[source["over_owner"]], u)
        moved_normal = rotate(normal, axes[source["over_owner"]], u)
        moved_down = rotate(down, axes[source["over_owner"]], u)
        radial = tuple(old+(new-old)*u for old, new in zip(before, candidate))
        sub_position = tuple((radius+radial[0])*value for value in sub)
        over_position = tuple((radius+radial[1])*value for value in over)
        separation = tuple(a-b for a, b in zip(over_position, sub_position))
        gap, jump = dot(separation, moved_normal), dot(separation, moved_down)
        values = dict(sub_direction=sub, over_direction=over,
            sub_position_m=sub_position, over_position_m=over_position,
            normal=moved_normal, down_direction=moved_down, radial_at_time=radial,
            separation_m=separation)
        for name, value in values.items():
            check_vector(raw[name], value, name)
        for name, value in (("gap_m", gap), ("jump_m", jump),
                ("exact_gap_change_m", gap-reference_gap),
                ("exact_jump_change_m", jump-reference_jump)):
            contains(raw[name], value, name)
    return len(instants)


def same(first, second, label):
    require(encoded(first) == encoded(second), label + " changed literal source payload")


def rational_float(value):
    item = exact_float(value)
    return [item.numerator, item.denominator]


def typed_mechanics(raw):
    result = {key: value for key, value in raw.items()
              if key not in ("format", "coordinate_convention")}
    result["fault_traces"] = {key: value for key, value in raw["fault_traces"].items()
        if key not in ("format", "coordinate_convention", "evolution")}
    return result


def history_sources(history, initial, candidate, source, support, material, checkpoint_sha):
    seal(history, "binding_sha256", "Original history")
    require(history["format"] == "native-original-physical-history-binding-1",
            "Unsupported physical history format")
    require(history["model"] == "replayed_original_pairs_affine_candidate_observation_no_handover",
            "Original history model changed")
    require(history["radial_path_model"] == "affine_previous_to_unaccepted_candidate_radial_displacement"
            and history["source_work_scope"] == "original force quadrature endpoint ledger only; never a volume density",
            "Original history radial/work scope changed")
    require(history["original_checkpoint_sha256"] == checkpoint_sha,
            "History checkpoint reference changed")
    same(history["accepted_before"], initial["native_underthrust_motion"], "Accepted history")
    same(history["origin_surface"], initial["origin_surface"], "Origin material")
    same(history["actual_candidate"], candidate, "Actual candidate")
    same(history["current_load"], source, "Current physical load")
    same(history["whole_time_support"], support, "Whole-time support")
    same(history["material_state"], material, "First-entry material state")
    for name in ("incoming_pair_handover_authorized", "candidate_mechanics_adopted",
                 "mechanical_work_assigned_to_material_volume"):
        require(history[name] is False, "History grants " + name)
    require(history["original_pair_history_authenticated"] is True,
            "History lost original pair references")
    before = initial["native_underthrust_motion"]
    snapshot, seed = before["snapshot"], before["initial_mechanics"]
    same(history["literal_dt_s"], candidate["dt_s"], "Original seconds")
    same(history["literal_dt_myr"], candidate["admission"]["dt_myr"], "Original Myr")
    same(material["journal"]["omega_rad_per_myr"], candidate["omega_rad_per_myr"], "Full journal Ω")
    same(material["journal"]["dt_myr"], history["literal_dt_myr"], "Journal clock")
    same(material["original_mechanics_payload"], seed, "Material preparation")
    reference = material["memory_reference"]
    require(digest(reference["before_payload"]) == reference["before_sha256"]
            and digest(reference["candidate_after_payload"]) == reference["candidate_after_sha256"],
            "Material memory reference seal mismatch")
    memory_before = {key: snapshot[key] for key in (
        "states", "spatial_memory", "free_anchor_fields", "elapsed_time_s",
        "cumulative_source_work_j", "cumulative_basal_dissipation_j",
        "cumulative_bending_numerical_dissipation_j")}
    memory_before.update(original_mechanics_payload=seed,
        original_surface_payload=initial["current_surface"],
        original_force_checkpoint_sha256=checkpoint_sha)
    same(reference["before_payload"], memory_before, "Material original memory")
    memory_after = dict(states=candidate["states"], spatial_memory=candidate["spatial_step"]["memory"],
        free_anchor_fields=candidate["spatial_step"]["anchor_fields"], dt_s=candidate["dt_s"],
        candidate_diagnostics=candidate["diagnostics"], candidate_sha256=digest(candidate),
        whole_time_support=support, candidate_adopted=False)
    same(reference["candidate_after_payload"], memory_after, "Material unaccepted memory")
    require(support["candidate_sha256"] == digest(candidate), "Support candidate reference changed")
    literal = history["actual_candidate_literal_fields"]
    for name, value in candidate.items():
        if name != "unresolved":
            same(literal[name], value, "Typed candidate " + name)
    require(literal["unresolved"] == [], "Typed literal candidate adds overlap records")
    for name, key in (("admission_audit_json_hex", "admission"), ("points_json_hex", "physical_points")):
        require(bytes.fromhex(literal[name]) == encoded(candidate[key]),
                "Candidate literal byte field differs: " + name)
    path = history["contact_time_path"]
    same(path["initial_mechanics"], typed_mechanics(seed), "Contact-time preparation")
    same(path["step"], candidate["spatial_step"], "Contact-time spatial step")
    require(path["prior_steps"] == [], "Original contact-time path skips prior accepted steps")
    previous = [[identity, {"coefficients": gap}, {"coefficients": jump}]
                for identity, gap, jump in snapshot["free_anchor_fields"]]
    same(path["previous_anchor_fields"], previous, "Original free anchors")
    path_body = dict(initial_mechanics=path["initial_mechanics"], step=path["step"],
        previous_anchor_fields=[[identity, gap["coefficients"], jump["coefficients"]]
            for identity, gap, jump in path["previous_anchor_fields"]],
        prior_steps=path["prior_steps"], path_model=path["path_model"])
    path_sha = digest(path_body)
    require(path_sha == path["path_sha256"] == history["contact_time_path_sha256"]
            == support["path_sha256"] == reference["constitutive_path_sha256"],
            "Original contact-time path selfseal/reference changed")
    old_work = [[name, seed[name] if name == "preparation_work_j" else snapshot[name]]
        for name in ("preparation_work_j", "cumulative_source_work_j",
                     "cumulative_basal_dissipation_j", "cumulative_bending_numerical_dissipation_j")]
    same(history["before_work_ledgers"], old_work, "Prepared work ledger")
    new_work = [[name, value] for name, value in sorted(candidate["diagnostics"].items())
                if name.endswith("_j") or "energy_balance" in name]
    same(sorted(history["unaccepted_candidate_work_ledgers"]), new_work, "Unaccepted work ledger")
    require(snapshot["receipts"] == [] and snapshot["cumulative_source_work_j"] == 0,
            "The original unaccepted epoch acquired receipts or accepted work")
    return seed, snapshot


def near_float(actual, expected, label, tolerance=Decimal("3e-15")):
    value = dec(exact_float(actual))
    require(abs(value-expected) <= tolerance*max(Decimal(1), abs(expected)),
            label + " differs from independent original geometry")


def check_nodes(history, seed, snapshot, candidate):
    nodes = history["quadrature_histories"]
    traces = {row["trace_id"]: row for row in seed["fault_traces"]["traces"]}
    prepared, previous, after = (dict(value) for value in
        (seed["states"], snapshot["states"], candidate["states"]))
    require(len(traces) == 4 and len(nodes) == len(prepared) == len(previous) == len(after) == 16,
            "Original quadrature cardinality changed")
    require([row["point_id"] for row in nodes] == list(previous),
            "Original quadrature order/identities changed")
    reference = history["original_reference"]
    require(all(type(reference[name]) is list and len(reference[name]) == 16
                for name in ("points","reference_points","materials","areas","frames")),
            "Original reference quadrature shape changed")
    same(reference["surface"], history["origin_surface"], "Original reference surface")
    roots = ((Decimal(3)+2*(Decimal(6)/5).sqrt())/7).sqrt(), ((Decimal(3)-2*(Decimal(6)/5).sqrt())/7).sqrt()
    gauss = (-roots[0], -roots[1], roots[1], roots[0])
    require(seed["fault_traces"]["quadrature"] == dict(gauss_order=4, panel_count=1),
            "Unsupported original quadrature")
    result = {}
    for index, node in enumerate(nodes):
        identity, trace_id = node["point_id"], node["trace_id"]
        trace = traces[trace_id]
        same(node["prepared_state"], prepared[identity], "Prepared nodal state")
        same(node["previous_state"], previous[identity], "Previous nodal state")
        same(node["unaccepted_candidate_state"], after[identity], "Unaccepted nodal state")
        same(node["original_material"], trace, "Nodal material law")
        point = node["original_point"]
        require(type(node["original_frame"]) is list and len(node["original_frame"]) == 2
                and all(type(point[name]) is list and len(point[name]) == 3
                        for name in ("position","horizontal","normal","down_dip")),
                "Original point/frame vector shape changed")
        require(point["point_id"] == identity and point["trace_id"] == trace_id
                and point["panel_index"] == 0 and point["node_index"] == index % 4,
                "Original quadrature label changed")
        for value in (node["original_area_m2"], point["area_m2"],
                      prepared[identity]["area_m2"], previous[identity]["area_m2"], after[identity]["area_m2"]):
            same(value, reference["areas"][index], "Original quadrature measure")
        same(point, reference["points"][index], "Reference original point")
        same(point, reference["reference_points"][index], "Reference immutable original point")
        same(trace, reference["materials"][index], "Reference original law")
        same(node["original_frame"], reference["frames"][index], "Reference original frame")
        start, end = tuple(dec(exact_float(v)) for v in trace["start"]), tuple(dec(exact_float(v)) for v in trace["end"])
        require(dot(start, start) == dot(end, end) == 1 and dot(start, end) == 0,
                "Audit requires the authentic original quarter traces")
        horizontal = cross(start, end)
        tangent = cross(horizontal, start)
        # The retained computational extent is a binary64 pi/2, not exact pi/2.
        extent = dec(exact_float(math.pi/2))
        theta = extent*(1+gauss[index % 4])/2
        near_float(float(fraction(node["computational_angle"])), theta, "Original computational angle")
        sine, cosine = sin_cos(dec(fraction(node["computational_angle"])))
        direction = tuple(a*cosine+b*sine for a, b in zip(start, tangent))
        dip_sine, dip_cosine = sin_cos(dec(exact_float(trace["dip_radians"])))
        for actual, expected in zip(node["original_frame"], (dip_sine, dip_cosine)):
            near_float(actual, expected, "Original dip frame")
        normal = tuple(dip_sine*a+dip_cosine*b for a,b in zip(horizontal,direction))
        down = tuple(dip_cosine*a-dip_sine*b for a,b in zip(horizontal,direction))
        for name, values in (("position", direction), ("horizontal", horizontal), ("normal", normal), ("down_dip", down)):
            for actual, expected in zip(point[name], values):
                near_float(actual, expected, "Original node " + name)
        result[identity] = node
    return result, traces


def field_value(raw, angle):
    coefficients = raw["coefficients"] if type(raw) is dict else raw
    require(len(coefficients) == 5, "Original field basis changed")
    sine, cosine = sin_cos(angle)
    double_sine, double_cosine = sin_cos(2*angle)
    basis = (Decimal(1), cosine, sine, double_cosine, double_sine)
    return dot(tuple(dec(exact_float(v)) for v in coefficients), basis)


def check_query(query, node, history, candidate):
    seal(query, "query_sha256", "Original nodal query")
    require(query["format"] == "native-original-quadrature-history-query-1",
            "Unsupported nodal query format")
    require(query["precision"] == 128
            and query["constitutive_path_model"] == "affine_free_anchors_fractional_backward_euler"
            and query["radial_path_model"] == "affine_previous_to_unaccepted_candidate_radial_displacement",
            "Original query diagnostic model or declared precision changed")
    require(query["work_scope"] == "unchanged original nodal before/candidate endpoint records only"
            and query["field_scope"] == "exact dyadic-coefficient return-map diagnostic; saved binary64 nodal endpoints retained separately under original roundoff-aware replay"
            and query["clock_scope"] == "elapsed_time_s includes original local preparation; native_elapsed_time_s is the global native clock; identical exact fraction with retained dt_s and dt_myr",
            "Original query field/clock/work scope changed")
    same(query["original_quadrature_history"], node, "Query inherited nodal history")
    require(query["binding_sha256"] == history["binding_sha256"]
            and query["point_id"] == node["point_id"] and query["trace_id"] == node["trace_id"],
            "Query original history reference changed")
    same(query["computational_angle"], node["computational_angle"], "Query original coordinate")
    for name in ("incoming_pair_handover_authorized", "candidate_mechanics_adopted",
                 "nodal_work_allocated_to_material_volume", "intermediate_work_computed"):
        require(query[name] is False, "Query grants " + name)
    require(query["original_pair_history_authenticated"] is True,
            "Query lost original physical authentication")
    low, high = interval(query["time_fraction_bounds"])
    require(low == high and 0 <= low <= 1, "Control query must retain its exact singleton time")
    trace, path = node["original_material"], history["contact_time_path"]
    theta, u = dec(fraction(node["computational_angle"])), dec(low)
    old_fields = {row[0]:row[1:] for row in path["previous_anchor_fields"]}
    new_fields = {row[0]:row[1:] for row in candidate["spatial_step"]["anchor_fields"]}
    old_gap, old_jump = [field_value(v, theta) for v in old_fields[node["trace_id"]]]
    new_gap, new_jump = [field_value(v, theta) for v in new_fields[node["trace_id"]]]
    gap, jump = old_gap+(new_gap-old_gap)*u, old_jump+(new_jump-old_jump)*u
    stored = next(row for row in path["step"]["previous_memory"]["traces"] if row["trace_id"] == node["trace_id"])
    piece = next(row for row in stored["pieces"] if exact_float(row["low_radians"]) <= fraction(node["computational_angle"]) <= exact_float(row["high_radians"]))
    old_p, old_c = field_value(piece["plastic_slip"], theta), field_value(piece["cumulative_slip"], theta)
    columns, law = trace["columns"], trace["law"]
    column_kn = 1./(columns["sub_length_m"]/columns["sub_normal_modulus_pa"]+columns["over_length_m"]/columns["over_normal_modulus_pa"])
    column_kt = 1./(columns["sub_length_m"]/columns["sub_shear_modulus_pa"]+columns["over_length_m"]/columns["over_shear_modulus_pa"])
    kn = dec(exact_float(1./(1./column_kn+1./law["normal_stiffness_pa_m"])))
    kt = dec(exact_float(1./(1./column_kt+1./law["tangential_stiffness_pa_m"])))
    mu = dec(exact_float(law["friction_dry"]+(law["friction_wet"]-law["friction_dry"])*trace["water"]))
    cohesion = dec(exact_float(law["cohesion_pa"]*(1.-(1.-law["wet_cohesion_fraction"])*trace["water"])))
    pressure, trial = -kn*gap, kt*(jump-old_p)
    strength = cohesion+mu*pressure
    numerator = max(Decimal(0),trial-strength)-max(Decimal(0),-trial-strength)
    eta, duration = dec(exact_float(law["viscosity_pa_s_m"])), dec(exact_float(history["literal_dt_s"]))
    require(query["constitutive_status"] == (
        "viscous_fractional_backward_euler_diagnostic_only" if eta
        else "inviscid_original_pair_dense_return_map_diagnostic"),
        "Original query return-map status changed")
    increment = numerator*u*duration/(kt*u*duration+eta) if eta else numerator/kt
    plastic, cumulative = old_p+increment, old_c+abs(increment)
    shear = kt*(jump-plastic)
    expected = dict(free_gap_bounds_m=gap, free_jump_bounds_m=jump,
        previous_plastic_slip_bounds_m=old_p, plastic_increment_bounds_m=increment,
        plastic_slip_bounds_m=plastic, cumulative_slip_bounds_m=cumulative,
        normal_pressure_bounds_pa=pressure, shear_traction_bounds_pa=shear,
        contact_gap_bounds_m=-pressure/dec(exact_float(law["normal_stiffness_pa_m"])),
        contact_jump_bounds_m=plastic+shear/dec(exact_float(law["tangential_stiffness_pa_m"])),
        elapsed_time_s_bounds=dec(exact_float(node["previous_state"]["elapsed_time_s"]))+duration*u,
        native_elapsed_time_s_bounds=dec(exact_float(history["accepted_before"]["snapshot"]["elapsed_time_s"]))+duration*u,
        physical_time_myr_bounds=dec(exact_float(history["origin_surface"]["time_myr"]))+dec(exact_float(history["literal_dt_myr"]))*u)
    for name, value in expected.items():
        contains(query[name], value, name)
    before_radial = next(row[1:] for row in history["accepted_before"]["snapshot"]["transverse_displacement_m"] if row[0] == node["trace_id"])
    after_radial = next(row[1:] for row in candidate["transverse_displacement_m"] if row[0] == node["trace_id"])
    point = node["original_point"]
    source = dict(trace_id=node["trace_id"],
        reference_coordinates=[rational_float(v) for v in point["position"]],
        original_normal=[rational_float(v) for v in point["normal"]],
        original_down_direction=[rational_float(v) for v in point["down_dip"]],
        sub_owner=trace["subducting_plate"], over_owner=trace["overriding_plate"],
        omega_records=[[owner,[rational_float(v) for v in values]] for owner,values in candidate["omega_rad_per_myr"]],
        dt_myr=rational_float(history["literal_dt_myr"]),
        radius_m=rational_float(float(history["origin_surface"]["radius_km"]*1000.)),
        radial_before=[rational_float(v) for v in before_radial],
        radial_candidate=[rational_float(v) for v in after_radial],
        time_fraction_bounds=query["time_fraction_bounds"], precision=query["precision"], order=32)
    return check_pair_map(query["tip_maps"], source)


def atan(value):
    sign = -1 if value < 0 else 1
    value = abs(value)
    doubles = 0
    while value > Decimal("0.2"):
        value = value/(1+(1+value*value).sqrt())
        doubles += 1
    term, total = value, value
    threshold = Decimal(10)**-(PRECISION-16)
    for index in range(1, 10000):
        term *= -value*value
        addition = term/Decimal(2*index+1)
        total += addition
        if abs(addition) < threshold:
            return sign*total*(2**doubles)
    raise ValueError("Independent atan series did not converge")


def atan2_positive(sine, cosine):
    require(sine >= 0 and cosine >= 0 and sine+cosine > 0,
            "Independent carrier coordinate has unresolved quadrant")
    half_pi = (16*atan(Decimal(1)/5)-4*atan(Decimal(1)/239))/2
    return atan(sine/cosine) if cosine >= sine else half_pi-atan(cosine/sine)


def check_incoming(rows, history, nodes, traces, certificates, material_queries):
    require(len(rows) == len(certificates) == 7, "Incoming point coverage changed")
    saved = {row["label"]: row["certificate"] for row in certificates}
    end_queries = {row["label"]: row["query"] for row in material_queries
                   if row["query"]["time_fraction"] == [1, 1]}
    bindings = {row["trace_id"]: row for row in history["material_state"]["trace_bindings"]}
    omega = {owner: tuple(dec(exact_float(value)*exact_float(history["literal_dt_myr"]))
                         for value in values)
             for owner, values in history["actual_candidate"]["omega_rad_per_myr"]}
    numerical_checks = mismatch_count = off_carrier_count = 0
    require([row["label"] for row in rows] == [row["label"] for row in certificates],
            "Incoming labels reordered or substituted")
    for record in rows:
        label, raw = record["label"], record["binding"]
        seal(raw, "binding_sha256", "Incoming binding " + label)
        require(raw["format"] == "native-first-entry-original-physical-bank-binding-1",
                "Unsupported incoming binding format")
        require(raw["model"] == "preserved_original_pairs_incoming_material_handover_unresolved"
                and raw["precision"] == 256 and raw["order"] == 32,
                "Incoming binding model or declared proof budgets changed")
        chronology, final = saved[label], end_queries[label]
        require(raw["material_state_sha256"] == history["material_state"]["state_sha256"]
                == chronology["material_state_sha256"]
                and raw["point_chronology_sha256"] == chronology["point_chronology_sha256"]
                and raw["physical_history_sha256"] == history["binding_sha256"],
                "Incoming context source references changed")
        same(raw["original_parent_id"], chronology["original_parent_id"], "Incoming original parent")
        same(raw["original_point"], chronology["original_point"], "Incoming original label")
        first = chronology["first_entry_certificate"]
        window = None if first is None else first["first_entry_time_fraction"]
        same(raw["time_fraction_bounds"], window, "Incoming original root window")
        same(raw["first_receiver_ancestry_status"], final["first_entry_receiver_ancestry_status"],
             "Incoming original receiver ancestry")
        for name in ("first_root_midpoint_used", "original_nodal_history_assigned_to_incoming_material",
                     "finite_material_reservoir_assigned_to_point", "nodal_work_distributed_to_incoming_material",
                     "candidate_mechanics_adopted", "physical_pair_admission_authorized"):
            require(raw[name] is False, "Incoming binding grants " + name)
        require(raw["accepted_loaded_native_intervals"] == raw["accepted_source_work_j"] == 0,
                "Incoming binding accepts original force/work")
        if window is None:
            require(raw["status"] == "no_first_entry" and raw["rows"] == [],
                    "No-entry context created a physical bank")
            continue
        require(raw["status"] == "unresolved_material_handover", "Incoming material handover was adopted")
        expected_ids = [(donor, receiver) for donor in final["original_trace_ids"]
                        for receiver in final["receiver_trace_ids"]]
        require([(row["donor_trace_id"], row["receiver_trace_id"]) for row in raw["rows"]] == expected_ids,
                "Incoming bank coverage differs from original ancestry")
        point = vector(raw["original_point"])
        low, high = interval(window)
        for row in raw["rows"]:
            donor_id, receiver_id = row["donor_trace_id"], row["receiver_trace_id"]
            donor_binding, receiver_binding = bindings[donor_id], bindings[receiver_id]
            donor, receiver = traces[donor_id], traces[receiver_id]
            desired = [raw["original_parent_id"], receiver_binding["receiver_fragment_id"]]
            donor_pair = [donor_binding["donor_fragment_id"], donor_binding["receiver_fragment_id"]]
            receiver_pair = [receiver_binding["donor_fragment_id"], receiver_binding["receiver_fragment_id"]]
            for name, expected in (("desired_fragment_pair", desired),
                    ("original_donor_trace_pair", donor_pair), ("original_receiver_trace_pair", receiver_pair)):
                same(row[name], expected, "Original bank " + name)
            require(row["receiver_fragment_id"] == desired[1]
                    and row["same_original_fragment_pair"] is (desired == donor_pair == receiver_pair),
                    "Original physical bank correspondence was relabeled")
            mismatch_count += int(desired != donor_pair or desired != receiver_pair)
            start, end = tuple(exact_float(v) for v in donor["start"]), tuple(exact_float(v) for v in donor["end"])
            plane = cross(start, end)
            value = sum((a*b for a,b in zip(plane,point)), Fraction(0))
            on_carrier = value == 0 and sum(a*b for a,b in zip(start,point)) >= 0 and sum(a*b for a,b in zip(end,point)) >= 0
            require(fraction(row["donor_carrier_plane_value"]) == value
                    and row["incoming_original_label_on_donor_carrier"] is on_carrier,
                    "Incoming original carrier incidence changed")
            off_carrier_count += int(not on_carrier)
            start, end = tuple(exact_float(v) for v in receiver["start"]), tuple(exact_float(v) for v in receiver["end"])
            same(row["receiver_carrier_plane"], [[v.numerator,v.denominator] for v in cross(start,end)],
                 "Receiver original carrier plane")
            for key, identity in (("original_donor_point_ids", donor_id), ("original_receiver_point_ids", receiver_id)):
                same(row[key], [point_id for point_id,node in nodes.items() if node["trace_id"] == identity],
                     "Incoming preserved quadrature references")
            require(row["handover_status"] == "unresolved_material_handover",
                    "Incoming context invents material handover")
            for instant in sorted({low, (low+high)/2, high}):
                dpoint = tuple(dec(v) for v in point)
                relative = rotate(rotate(dpoint,omega[donor_binding["subducting_plate"]],dec(instant)),
                    tuple(-v for v in omega[receiver_binding["overriding_plate"]]),dec(instant))
                check_vector(row["relative_entry_coordinates"],relative,"Incoming relative full Ω coordinates")
                if row["receiver_native_angle_bounds"] is not None:
                    cosine = dot(tuple(dec(v) for v in start),relative)
                    sine = dot(tuple(dec(v) for v in end),relative)
                    contains(row["receiver_native_angle_bounds"],atan2_positive(sine,cosine),
                             "Incoming receiver native carrier angle")
                    require(row["receiver_coordinate_status"] == "first_owner_root_receiver_carrier_coordinate_proved"
                            and raw["first_receiver_ancestry_status"] == "proved",
                            "Incoming carrier coordinate lacks original owner-root ancestry")
                else:
                    require(row["receiver_coordinate_status"] == "unresolved_carrier_coordinate",
                            "Incoming unresolved angle claims a proved coordinate")
                numerical_checks += 1
    require(mismatch_count > 0 and off_carrier_count > 0,
            "Saved contexts lost the physical bank mismatch or off-carrier evidence")
    return dict(numerical_relative_coordinate_checks=numerical_checks,
                original_bank_mismatch_count=mismatch_count, original_off_carrier_count=off_carrier_count)


def verify_prior(source_receipt):
    seal(source_receipt, "receipt_sha256", "Prior source validation")
    require(source_receipt["status"] == "PASS"
            and source_receipt["prior_artifact_count"] == 83
            and source_receipt["prior_tests_passed"] == 770
            and source_receipt["original_source_count"] == 654
            and source_receipt["prior_starting_source_count"] == 643,
            "Prior source validation ancestry changed")
    prior_relative="analysis/geometric_native_first_entry_validation/"
    expected_paths={prior_relative+name for name in (
        "release_validation_02/validation_summary.json", "release_validation_02/artifact_manifest.json",
        "release_gates_02/receipt.json", "control_release_02/report.json",
        "control_release_02/independent_first_entry_receipt.json", "control_release_02/original_force/initial.json",
        "control_release_02/original_force/force_candidate.json", "control_release_02/original_force/constitutive_time_support.json",
        "control_release_02/material_state.json", "control_release_02/point_certificates.json",
        "control_release_02/material_point_queries.json")}
    expected_paths.add("analysis/geometric_native_physical_binding_validation/starting_sources.json")
    require(set(source_receipt["source_artifacts"]) == expected_paths,
            "Prior source validation reference coverage changed")
    for name, record in source_receipt["source_artifacts"].items():
        path = ROOT/name
        require(path.is_file() and file_sha(path) == record["sha256"]
                and path.stat().st_size == record["bytes"], "Prior authority differs: " + name)
    prior = ROOT/"analysis/geometric_native_first_entry_validation"
    summary = read(prior/"release_validation_02/validation_summary.json")
    manifest_path = prior/"release_validation_02/artifact_manifest.json"
    manifest = read(manifest_path)
    seal(summary,"summary_sha256","Prior summary")
    seal(manifest,"manifest_sha256","Prior artifact manifest")
    require(summary["status"] == "PASS" and manifest["count"] == len(manifest["artifacts"]) == 83
            and summary["artifact_manifest"]["sha256"] == file_sha(manifest_path),
            "Prior authoritative release summary changed")
    for name, record in manifest["artifacts"].items():
        path = ROOT/name
        require(path.is_file() and file_sha(path) == record["sha256"]
                and path.stat().st_size == record["bytes"], "Prior artifact changed: " + name)
    for key in ("release_gate","actual_control","independent_audit"):
        reference=summary[key]
        require(file_sha(ROOT/reference["path"]) == reference["sha256"],
                "Prior release reference changed: " + key)
    gate=read(ROOT/summary["release_gate"]["path"])
    control=read(ROOT/summary["actual_control"]["path"])
    audit=read(ROOT/summary["independent_audit"]["path"])
    seal(control,"report_sha256","Prior control")
    seal(audit,"receipt_sha256","Prior independent audit")
    require(gate["status"] == control["status"] == audit["status"] == "PASS"
            and gate["counts"]["passed"] == 770 and gate["exit_code"] == 0
            and gate["changed_paths"] == [], "Prior PASS gate/control/audit changed")
    return prior/"control_release_02"


def run_audit(control):
    started=time.perf_counter()
    report=read(control/"report.json")
    seal(report,"report_sha256","Physical binding control")
    require(report["status"] == "PASS", "Physical binding control failed")
    expected_artifacts={"sources_before.json", "sources_after.json", "source_validation_receipt.json",
        "original_history.json", "original_history_replayed.json", "original_nodal_queries.json",
        "original_nodal_queries_replayed.json", "point_certificates.json", "incoming_first_entry_bindings.json",
        "incoming_first_entry_bindings_replayed.json"}
    expected_artifacts.update("original_force/"+name for name in (
        "initial.json", "after_refusal.json", "declared_source.json", "native_source.json",
        "force_candidate.json", "constitutive_time_support.json", "report.json"))
    require(set(report["artifacts_sha256"]) == expected_artifacts,
            "Control input artifact coverage changed")
    for name, claim in report["artifacts_sha256"].items():
        require(file_sha(control/name) == claim, "Control artifact hash mismatch: " + name)
    require(report["original_trace_count"] == 4 and report["original_node_count"] == 16
            and report["query_count"] == 48 and report["incoming_context_count"] == 7
            and report["query_times"] == [[0,1],[3,8],[1,1]], "Control coverage changed")
    for name in ("original_typed_checkpoint_loaded", "original_force_re_solved",
                 "original_candidate_all_fields_identical", "history_builder_candidate_replayed",
                 "history_dto_external_sources_replayed", "prior_material_state_replayed"):
        require(report[name] is True, "Control lost source replay evidence: " + name)
    for name in ("intermediate_work_computed", "work_assigned_to_material_volume",
                 "candidate_mechanics_adopted", "physical_pair_admission_authorized",
                 "global_native_inventory_updated"):
        require(report[name] is False, "Control claims " + name)
    require(report["accepted_loaded_native_intervals"] == report["accepted_source_work_j"] == 0,
            "Physical binding adopts original force/work")
    before, after=read(control/"sources_before.json"),read(control/"sources_after.json")
    same(before,after,"Control source before/after map")
    current={str(path.relative_to(ROOT)):file_sha(path)
             for directory in ("tectonics","tests","scripts","analysis")
             for path in sorted((ROOT/directory).rglob("*.py"))}
    same(current,after,"Complete current Python source enumeration")
    for name, claim in after.items():
        require(file_sha(ROOT/name) == claim, "Source changed during independent audit: " + name)
    prior=verify_prior(read(control/"source_validation_receipt.json"))
    starting=read(ROOT/"analysis/geometric_native_physical_binding_validation/starting_sources.json")
    same(starting,read(prior.parent/"release_gates_02/sources_after.json"),
         "Original authoritative 654-source baseline")
    require(len(starting) == 654 and set(starting) <= set(after), "Original source map ancestry changed")
    for name,claim in starting.items():
        require(after[name] == claim,"Original frozen source was edited: " + name)
    require(report["source_count"] == len(after) and report["existing_sources_changed"] == [],
            "Control source count/change claims differ")
    same(report["added_python_sources"],sorted(set(after)-set(starting)),"Added Python source set")
    expected_pairs=[["original_force/initial.json","original_force/after_refusal.json"],
        ["original_history.json","original_history_replayed.json"],
        ["original_nodal_queries.json","original_nodal_queries_replayed.json"],
        ["incoming_first_entry_bindings.json","incoming_first_entry_bindings_replayed.json"]]
    same(report["byte_pairs"],expected_pairs,"Physical replay pair set")
    byte_pairs=[]
    for first,second in expected_pairs:
        require((control/first).read_bytes() == (control/second).read_bytes(),"Replay bytes differ: " + first)
        byte_pairs.append(dict(first=first,second=second,sha256=file_sha(control/first),bytes=(control/first).stat().st_size))
    force=control/"original_force"
    initial,candidate,source,support=(read(force/name) for name in
        ("initial.json","force_candidate.json","native_source.json","constitutive_time_support.json"))
    seal(initial,"payload_sha256","Original force checkpoint")
    same(initial,read(prior/"original_force/initial.json"),"Prior original checkpoint")
    same(candidate,read(prior/"original_force/force_candidate.json"),"All prior original candidate fields")
    same(source,read(prior/"original_force/native_source.json"),"Original physical source")
    same(support,read(prior/"original_force/constitutive_time_support.json"),"Prior original constitutive support")
    force_report=read(force/"report.json")
    seal(force_report,"report_sha256","Fresh force receipt")
    require(force_report["status"] == "PASS" and force_report["original_candidate_field_count"] == len(candidate)
            == report["original_candidate_field_count"],"Fresh force original candidate count differs")
    probe=ROOT/force_report["source_probe_path"]
    require(file_sha(probe) == force_report["source_probe_sha256"],"Original refusal probe changed")
    historical=read(probe)
    for name,value in historical["candidate"].items():
        same(candidate[name],value,"Original refusal-probe candidate " + name)
    same(candidate["admission"],historical["admission"],"Original refusal probe admission")
    material=read(prior/"material_state.json")
    history=read(control/"original_history.json")
    seed,snapshot=history_sources(history,initial,candidate,source,support,material,file_sha(force/"initial.json"))
    nodes,traces=check_nodes(history,seed,snapshot,candidate)
    queries=read(control/"original_nodal_queries.json")
    require(len(queries) == 48,"Original nodal query count changed")
    expected={(identity,instant) for identity in nodes for instant in (Fraction(0),Fraction(3,8),Fraction(1))}
    seen=set(); tip_checks=0
    for record in queries:
        identity=record["point_id"]; instant=fraction(record["time_fraction"])
        require((identity,instant) not in seen,"Duplicate original nodal query")
        seen.add((identity,instant))
        node=nodes[identity]
        require(record["trace_id"] == node["trace_id"]
                and record["quadrature_index"] == node["original_point"]["node_index"],
                "Nodal query outer quadrature labels changed")
        require(interval(record["query"]["time_fraction_bounds"]) == (instant,instant),
                "Nodal query outer time differs from literal inner clock")
        tip_checks+=check_query(record["query"],node,history,candidate)
    require(seen == expected,"Original nodal time coverage incomplete")
    certificates=read(control/"point_certificates.json")
    same(certificates,read(prior/"point_certificates.json"),"Incoming original chronology export")
    incoming=check_incoming(read(control/"incoming_first_entry_bindings.json"),history,nodes,traces,
        certificates,read(prior/"material_point_queries.json"))
    input_names=sorted(set(report["artifacts_sha256"])|{"report.json"})
    result=dict(format="independent-native-original-physical-binding-audit-1",status="PASS",
        elapsed_seconds=time.perf_counter()-started,source_count=len(after),original_source_count=654,
        original_sources_unchanged=True,all_control_sources_unchanged=True,
        prior_authoritative_artifact_count=83,prior_authoritative_tests_passed=770,
        original_checkpoint_byte_identical=True,original_candidate_all_fields_identical=True,
        original_trace_count=4,original_node_count=16,original_query_count=48,incoming_context_count=7,
        complete_original_node_states_measures_and_endpoint_work_preserved=True,
        original_history_raw_seal_and_external_references_verified=True,
        original_contact_time_path_raw_seal_and_source_fields_verified=True,
        decimal_precision=PRECISION,numerical_tip_map_checks=tip_checks,numerical_point_checks_only=True,
        whole_time_interval_independently_certified=False,independent_force_solution_recomputed=False,
        intermediate_work_computed=False,nodal_work_assigned_to_material_volume=False,
        physical_pair_admission_authorized=False,candidate_mechanics_adopted=False,
        accepted_loaded_native_intervals=0,accepted_source_work_j=0,byte_pairs=byte_pairs,
        input_artifact_sha256={name:file_sha(control/name) for name in input_names},
        auditor_sha256=file_sha(Path(__file__)),
        limitation="Standard-library audit authenticates raw source checkpoints, complete candidate fields, memory/work ledgers, all original nodal states and exported rational-time observations. Independent 360-digit Decimal Rodrigues and trig checks verify numerical containment at singleton query times and first-root window endpoints/midpoints; they do not independently prove whole-window interval bounds or solve force balance. No incoming material handover, work-to-volume allocation, candidate adoption or native force acceptance is inferred.",
        **incoming)
    result["receipt_sha256"]=digest(result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    require(not args.out.exists(),"Independent audit receipt already exists")
    with localcontext() as context:
        context.prec=PRECISION
        result=run_audit(args.control.resolve())
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open("xb") as stream:
        stream.write(encoded(result)+b"\n")
    print(json.dumps({name:result[name] for name in
        ("status","original_query_count","incoming_context_count","numerical_tip_map_checks",
         "numerical_relative_coordinate_checks","accepted_loaded_native_intervals")}),flush=True)


if __name__ == "__main__":
    main()
