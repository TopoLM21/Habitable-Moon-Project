"""Producer-free high-precision checks of the four original selected lines.

This command reads an authenticated compact source fixture and waits for the
four small support records of the bounded angular control.  Direct source
quadrature, field checks, and consistently resealed mathematical negatives are
separate from angular convergence and physical admission.  No planet, clipped
point shard, polynomial producer, or source-extraction producer is imported.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis.genesis_volume_exact_lambda_validation import independent_line_audit as independent

PRECISION = 80
QUADRATURE_ORDERS = (12, 20)
WAIT_SECONDS = 3600
SCOPE = "independent_four_selected_source_lines_no_angular_or_physical_certificate"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def seal(value, key):
    result = dict(value)
    result.pop(key, None)
    result[key] = hashlib.sha256(canonical(result)).hexdigest()
    return result


def checked(value, key):
    require(type(value) is dict and isinstance(value.get(key), str), "Missing body seal: " + key)
    require(seal(value, key)[key] == value[key], "Body seal differs: " + key)
    return value


def pin(path):
    digest, size = hashlib.sha256(), 0
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
            size += len(block)
    return dict(bytes=size, raw_sha256=digest.hexdigest())


def _remember(inputs, path, expected=None):
    path = Path(path).resolve()
    value = pin(path)
    require(expected is None or value == expected, "Input raw pin differs: " + str(path))
    require(str(path) not in inputs or inputs[str(path)] == value,
            "Input changed between reads: " + str(path))
    inputs[str(path)] = value
    return value


def _read(path, key, inputs, *, wait=False):
    path = Path(path).resolve()
    deadline = time.monotonic() + (WAIT_SECONDS if wait else 0)
    announced = False
    while True:
        if path.is_file():
            data = path.read_bytes()
            try:
                value = json.loads(data)
            except json.JSONDecodeError:
                # A growing control writes each record once.  Do not hash a
                # partially published JSON document as its accepted input.
                if not wait or time.monotonic() >= deadline:
                    raise
            else:
                checked(value, key)
                expected = dict(bytes=len(data), raw_sha256=hashlib.sha256(data).hexdigest())
                _remember(inputs, path, expected)
                return value
        elif not wait or time.monotonic() >= deadline:
            raise FileNotFoundError(path)
        if not announced:
            print(json.dumps(dict(stage="WAITING_FOR_ORIGINAL_SUPPORT_RECORD", path=str(path)),
                             sort_keys=True), flush=True)
            announced = True
        time.sleep(.25)


def _false_flags(record):
    for key in ("physical_creation_admitted", "physical_release_admitted",
                "available_for_creation_funding", "automated_goal_enabled"):
        require(record.get(key) is False, "Source record physical gate differs: " + key)


def _source_archive_checks(raw, source):
    checked(source, "archive_sha256")
    source_body = {key: value for key, value in source.items()
                   if key not in ("archive_sha256", "source_binding_sha256")}
    require(hashlib.sha256(canonical(source_body)).hexdigest() == source["source_binding_sha256"],
            "Source archive binding body differs")
    arrays = source["arrays"]
    section = raw["section"]
    for key in ("elasticity_pa", "basis_xyz", "memory0", "memory1",
                "accepted_interval_mean_e0", "accepted_interval_mean_e1"):
        require(arrays[key] == raw[key], "Literal retained field differs: " + key)
    for key in ("triangle_m", "displacement_A_m", "slope_P_m", "normal_xyz", "offset_m"):
        require(arrays["section." + key] == section[key], "Literal geometry operand differs: " + key)
    require(source["section_binding"] == section["binding_sha256"], "Original section binding differs")
    require(source["source_identity"] == raw["source_identity"], "Original source identity differs")
    for key in ("seed_center_world_m", "seed_radius_m"):
        require(arrays["geometry." + key] == raw[key], "Original computational sphere differs: " + key)
    require(arrays["geometry.lambda_bounds"] == section["lambda_bounds"], "Original radial bounds differ")
    X, A, P = (section[key] for key in ("triangle_m", "displacement_A_m", "slope_P_m"))
    # Python float additions retain the declared literal binary64 convention.
    C = [[float(x + p) for x, p in zip(row, prow)] for row, prow in zip(X, P)]
    D = [[float(a - p) for a, p in zip(row, prow)] for row, prow in zip(A, P)]
    require(arrays["geometry.birth_C_m"] == C and arrays["geometry.birth_D_m"] == D,
            "Birth chart was recomputed with a different literal convention")
    for key in ("parent_stiffness", "parent_mass"):
        require(arrays[key] == raw["previous_local_bulk_blocks"][key],
                "Whole original parent block changed: " + key)
    return dict(status="PASS", retained_source_fields_and_literal_geometry_equal=True,
                whole_original_parent_K_M_equal=True, source_body_binding_checked=True)


def _native_fraction(packet, label, index):
    entry = packet["native_exact_binary_ratio"][label]
    numerator, denominator = entry["numerator_denominator"][index]
    require(isinstance(numerator, str) and isinstance(denominator, str), "Native ratio is not exact text")
    return Fraction(int(numerator), int(denominator))


def _set_flat(array, index, shape, value):
    require(bool(shape), "Mutation requires a non-scalar alias")
    coordinates = []
    for size in reversed(shape):
        coordinates.append(index % size)
        index //= size
    require(index == 0, "Mutation flat index leaves native shape")
    target = array
    coordinates.reverse()
    for coordinate in coordinates[:-1]:
        target = target[coordinate]
    target[coordinates[-1]] = float(value)


def _set_native(packet, label, index, value):
    entry = packet["native_exact_binary_ratio"][label]
    entry["numerator_denominator"][index] = [str(value.numerator), str(value.denominator)]


def _flatten_alias(value, shape):
    if not shape:
        require(type(value) in (int, float) and math.isfinite(value), "Public native alias is not finite real")
        return [value]
    require(type(value) is list and len(value) == shape[0], "Public/native alias shape differs")
    return [number for child in value for number in _flatten_alias(child, shape[1:])]


def _indices(shape):
    if not shape:
        yield ()
    else:
        for index in range(shape[0]):
            for tail in _indices(shape[1:]):
                yield (index,) + tail


def _alias_checks(packet):
    native = packet.get("native_exact_binary_ratio")
    require(type(native) is dict, "Complete native logical witness is absent")
    aliases = {"values": packet["values"], "budgets": packet["roundoff_component_bounds"],
               "own": packet["absolute_operand_scales"]}
    mappings = (("full", "blocks"), ("fullbudget", "block_roundoff_component_bounds"),
                ("projected", "projected_blocks"), ("projectedbudget", "projected_roundoff_component_bounds"),
                ("coefficient", "coefficients"), ("coefficient_bound", "coefficient_roundoff_bounds"))
    for prefix, member in mappings:
        aliases.update({prefix + "." + key: value for key, value in packet[member].items()})
    for name, probe in packet["probe_contractions"].items():
        for member, fields in probe.items():
            aliases.update({"probe." + name + "." + member + "." + key: value
                            for key, value in fields.items()})
    interval = packet["interval"]
    require(type(interval) is dict, "Selected source line must retain its positive interval")
    for label in native:
        if label.startswith("interval."):
            key = label.split(".", 1)[1]
            if key == "side":
                aliases[label] = [interval["gamma_root_xi"] if interval["gamma_root_xi"] is not None
                                  else interval["constant_side"]]
            else:
                require(key in interval and interval[key] is not None, "Native interval alias is absent")
                aliases[label] = interval[key]
    require(set(aliases) == set(native), "Native/public witness label inventory differs")
    decoded = {}
    for label, public in aliases.items():
        array = independent.decode_native_array(packet, label)
        shape = list(array.shape)
        numbers = list(array.flat)
        public_numbers = _flatten_alias(public, shape)
        require(all(float(number) == observed for number, observed in zip(numbers, public_numbers)),
                "Native/public terminal conversion differs: " + label)
        decoded[label] = (shape, numbers)
    component_values, component_budgets = {}, {}
    for label, (shape, numbers) in decoded.items():
        if label.startswith(("full.", "projected.")):
            prefix = label
            budget_label = label.replace("full.", "fullbudget.", 1) if label.startswith("full.") else label.replace("projected.", "projectedbudget.", 1)
        elif label.startswith("probe.") and ".values." in label:
            prefix = label.replace(".values.", ".", 1)
            budget_label = label.replace(".values.", ".roundoff_component_bounds.", 1)
        else:
            continue
        require(budget_label in decoded and decoded[budget_label][0] == shape,
                "Component arithmetic alias shape differs")
        for index, number, budget in zip(_indices(shape), numbers, decoded[budget_label][1]):
            name = prefix + ("[" + ",".join(map(str, index)) + "]" if index else "")
            require(name not in component_values, "Duplicate native component alias")
            component_values[name], component_budgets[name] = number, budget
    names = packet["component_names"]
    require(type(names) is list and len(names) == len(set(names)) and set(names) == set(component_values),
            "Flattened/direct native component schema differs")
    require(decoded["values"][0] == decoded["budgets"][0] == [len(names)],
            "Native flattened component shape differs")
    require(all(component_values[name] == value and component_budgets[name] == budget
                for name, value, budget in zip(names, decoded["values"][1], decoded["budgets"][1])),
            "Flattened/direct native component values differ")
    return dict(status="PASS", native_public_alias_count=len(aliases),
                exact_native_flattened_direct_component_count=len(names),
                public_terminal_binary64_conversions_checked=True)


def _coefficient_negative(witness, field):
    packet = deepcopy(witness)
    label = "coefficient." + field
    entry = packet["native_exact_binary_ratio"][label]
    # Constant coefficient: its effect is visible at both declared field
    # sample positions.  The retained memory negative changes one component.
    count = math.prod(entry["shape"][1:])
    index = max(range(count), key=lambda i: abs(_native_fraction(packet, label, i)))
    before = _native_fraction(packet, label, index)
    after = before + max(Fraction(1), abs(before))
    _set_native(packet, label, index, after)
    _set_flat(packet["coefficients"][field], index, entry["shape"], after)
    return seal(packet, "archive_sha256"), dict(native_label=label, flat_index=index,
        original_exact_ratio=str(before), changed_exact_ratio=str(after),
        unchanged_coefficients_except_one=True, native_and_public_aliases_changed_together=True)


def _projected_negative(witness):
    packet = deepcopy(witness)
    field = "stiffness_enriched"
    label = "projected." + field
    entry = packet["native_exact_binary_ratio"][label]
    index = max(range(math.prod(entry["shape"])),
                key=lambda i: abs(_native_fraction(packet, label, i)))
    before = _native_fraction(packet, label, index)
    after = before + max(Fraction(1), abs(before))
    _set_native(packet, label, index, after)
    _set_flat(packet["projected_blocks"][field], index, entry["shape"], after)
    width = entry["shape"][1]
    component = f"{label}[{index // width},{index % width}]"
    flat_index = packet["component_names"].index(component)
    _set_native(packet, "values", flat_index, after)
    packet["values"][flat_index] = float(after)
    return seal(packet, "archive_sha256"), dict(native_label=label, flat_index=index,
        component_name=component, flattened_index=flat_index,
        original_exact_ratio=str(before), changed_exact_ratio=str(after),
        public_and_native_projected_and_flattened_aliases_changed_together=True,
        private_W_projection_checked_against_original_declared_basis=True)


def _negative_record(name, packet, mutation, result, baseline, field=None):
    checked(packet, "archive_sha256")
    alias_checks = _alias_checks(packet)
    mapping_key = "checks" if field is not None else "all_component_checks"
    before_checks, after_checks = baseline[mapping_key], result[mapping_key]
    targets = ([name for name in after_checks if "." + field + "(" in name]
               if field is not None else [mutation["component_name"]])
    delta = abs(Fraction(mutation["changed_exact_ratio"]) - Fraction(mutation["original_exact_ratio"]))
    proof = {}
    for target in targets:
        if target not in before_checks:
            continue
        before, after = before_checks[target], after_checks[target]
        observed = Fraction(after["absolute_difference"])
        prior = Fraction(before["absolute_difference"])
        budget = Fraction(after["arithmetic_bound"])
        detected = after["passed"] is False and observed > budget and observed > prior + delta / 2
        proof[target] = dict(detected=detected, original_check=before, changed_check=after,
                             targeted_error_increase_exceeds_half_exact_mutation=observed > prior + delta / 2)
    detected = bool(proof) and any(item["detected"] for item in proof.values())
    # Preserve both numerical results.  Detection must be evaluated from the
    # targeted mathematical checks, not merely from an existing OPEN status.
    return dict(name=name, status="DETECTED" if detected else "OPEN",
        targeted_independent_mathematical_detection=detected, target_detection_checks=proof,
        native_public_alias_checks=alias_checks, archive_sha256=packet["archive_sha256"],
        valid_resealed_archive_body=True, producer_private_seals_retained_as_original_provenance=True,
        mutation=mutation, targeted_field=field, baseline_audit=baseline, changed_audit=result,
        altered_witness=packet, mathematical_negative_not_byte_tamper_only=True)


def _audit_selected_line(raw, N, witness, options):
    """Retain an unresolved quadrature observation without trusting its values."""
    oracle = None
    refusal = None
    try:
        oracle = independent.integrate_source_line(raw, N, quadrature_orders=QUADRATURE_ORDERS, **options)
    except independent.IndependentLineRefusal as error:
        diagnostics = getattr(error, "diagnostics", None)
        refusal = dict(exception_class=type(error).__name__, message=str(error),
            structured_diagnostics_available=isinstance(diagnostics, dict),
            unresolved_observations_used_as_trusted_oracle=False,
            diagnostic_archive=independent.archive_line_result(diagnostics)
                if isinstance(diagnostics, dict) and "precision_decimal_digits" in diagnostics else diagnostics)
    fields = independent.audit_coefficient_fields(raw, N, witness, precision=PRECISION,
                                                 raise_on_mismatch=False)
    geometry = independent.audit_geometry_witness(raw, N, witness, precision=PRECISION)
    if oracle is not None:
        arithmetic = independent.audit_line_witness(raw, N, witness, oracle=oracle,
                                                   raise_on_mismatch=False, **options)
        primitive = arithmetic.get("represented_polynomial_primitive_audit")
        if primitive is None:
            primitive = independent.audit_represented_primitive(raw, N, witness,
                raise_on_mismatch=False, **options)
        direct_archive = independent.archive_line_result(oracle)
        component_status = "PASS" if not arithmetic["mismatching_components"] else "OPEN"
    else:
        # Neither the coarse nor fine observation from a refused quadrature
        # supplies an expected integral or an error enclosure for this lane.
        primitive = independent.audit_represented_primitive(raw, N, witness,
            raise_on_mismatch=False, **options)
        arithmetic = dict(scope=SCOPE, source_direct_quadrature_status="OPEN",
            source_quadrature_refusal_marker="REFUSED", trusted_oracle_available=False,
            fixed_line_numeric_agreement_status="NOT_EVALUATED",
            checked_component_count=0, all_component_checks={}, mismatching_components=[],
            saved_arithmetic_operand_scope_status="UNRESOLVED",
            saved_arithmetic_operand_authentication_certified=False,
            cached_oracle_source_binding_verified=False,
            represented_polynomial_primitive_audit=primitive,
            geometry_status=geometry["geometry_status"], geometry_checks=geometry["geometry_checks"],
            angular_accuracy_certified=False, general_contraction_accuracy_certified=False,
            physical_contraction_certified=False, physical_creation_admitted=False,
            physical_release_admitted=False, available_for_creation_funding=False)
        direct_archive = refusal["diagnostic_archive"]
        component_status = "OPEN"
    return oracle, dict(source_field_audit=fields,
        selected_line_arithmetic_audit=arithmetic,
        selected_line_component_arithmetic_status=component_status,
        source_geometry_observation=geometry,
        represented_polynomial_primitive_audit=primitive,
        independent_direct_quadrature=direct_archive,
        trusted_oracle_available=oracle is not None,
        source_quadrature_refused=refusal is not None,
        source_quadrature_refusal=refusal)


def _aggregate_statuses(records, negatives):
    return dict(
        field="PASS" if all(item["source_field_audit"]["coefficient_source_status"] == "PASS"
                            for item in records) else "OPEN",
        arithmetic="PASS" if all(item["trusted_oracle_available"]
                                 and item["selected_line_component_arithmetic_status"] == "PASS"
                                 for item in records) else "OPEN",
        geometry="PASS" if all(item["source_geometry_observation"]["geometry_status"] == "PASS"
                               for item in records) else "OPEN",
        represented_primitive="PASS" if all(item["represented_polynomial_primitive_audit"]["represented_polynomial_primitive_status"] == "PASS"
                                            for item in records) else "OPEN",
        negatives="PASS" if len(negatives) == 3 and all(item["status"] == "DETECTED"
                                                       for item in negatives) else "OPEN")


def run(source_fixture, actual_control, output):
    started = time.perf_counter()
    source_fixture, actual_control, output = (Path(value).resolve()
                                              for value in (source_fixture, actual_control, output))
    require(output.is_relative_to(Path("/workspace")) and not output.is_relative_to(ROOT)
            and not output.exists(), "Audit output must be fresh and external")
    require(actual_control.is_dir(), "Actual control directory is absent")
    inputs = {}
    fixture = _read(source_fixture, "source_fixture_sha256", inputs)
    control_fixture = _read(actual_control / "source_fixture.json", "source_fixture_sha256", inputs, wait=True)
    require(pin(source_fixture) == pin(actual_control / "source_fixture.json")
            and fixture == control_fixture, "Audit fixture must be the exact actual-control source copy")
    require(fixture["previous_source_count"] == 434 and len(fixture["previous_source_sha256"]) == 434
            and fixture["cohort_count"] == 800 and fixture["accepted_interval_count"] == 4,
            "Original source coverage differs")
    _false_flags(fixture)
    require(fixture["history_reinitialized"] is False
            and fixture["accepted_planet_memory_changed"] is False,
            "Authoritative Genesis history was replaced")
    for source_pin in fixture["source_pins"].values():
        _remember(inputs, source_pin["path"], {key: source_pin[key] for key in ("bytes", "raw_sha256")})
    for source_pin in [fixture["compact_comparison_archive_pin"], *fixture["original_minimal_source_pins"]]:
        _remember(inputs, source_pin["path"], {key: source_pin[key] for key in ("bytes", "raw_sha256")})
    for relative, digest in fixture["previous_source_sha256"].items():
        require(not Path(relative).is_absolute() and ".." not in Path(relative).parts,
                "Previous source path leaves repository")
        value = _remember(inputs, ROOT / relative)
        require(value["raw_sha256"] == digest, "Previous frozen source changed: " + relative)
    require([case["original_cohort_id"] for case in fixture["cases"]] == [1, 41, 80],
            "Original selected cohorts differ")
    records, negatives = [], []
    for index, case in enumerate(fixture["cases"]):
        checked(case, "source_case_sha256")
        for support_name, support in case["supports"].items():
            checked(support, "support_sha256")
            filename = f"case_{index:03d}_{support_name}.json"
            record = _read(actual_control / filename, "support_report_sha256", inputs, wait=True)
            require(record["cohort_id"] == case["original_cohort_id"] and record["support"] == support_name,
                    "Selected support identity differs")
            for key, expected in (("source_case_sha256", case["source_case_sha256"]),
                                  ("source_support_sha256", support["support_sha256"]),
                                  ("source_fixture_sha256", fixture["source_fixture_sha256"])):
                require(record[key] == expected, "Source/control seal identity differs: " + key)
            require(record["original_computational_support"] == support,
                    "Declared W, parent basis, probes or sphere were changed")
            _false_flags(record)
            require(record["source_history_reinitialized"] is False
                    and record["accepted_planet_memory_changed"] is False,
                    "Control advanced or reinitialized original history")
            raw = {**case["source_inputs"], "seed_center_world_m": support["seed_center_world_m"],
                   "seed_radius_m": support["seed_radius_m"]}
            source_checks = _source_archive_checks(raw, record["source"])
            witness = checked(record["line_witness"], "archive_sha256")
            aliases = _alias_checks(witness)
            require(witness["source_binding_sha256"] == record["source"]["source_binding_sha256"],
                    "Line witness belongs to another source")
            N = record["line_witness_barycentric_exact_binary_ratios"]
            require(len(N) == 3 and all(isinstance(value, str) for value in N),
                    "Exact selected barycentric operands are absent")
            native_N = list(independent.decode_native_array(witness, "interval.barycentric").flat)
            require(native_N == [Fraction(value) for value in N]
                    and len(set(native_N)) == 1, "Original selected midpoint line operands differ")
            options = dict(added_basis=support["original_declared_added_basis"],
                           parent_basis=support["original_declared_parent_basis"],
                           probes=support["declared_probes"], precision=PRECISION)
            print(json.dumps(dict(stage="INDEPENDENT_SOURCE_LINE_QUADRATURE", filename=filename),
                             sort_keys=True), flush=True)
            oracle, line_audits = _audit_selected_line(raw, N, witness, options)
            arithmetic, fields = line_audits["selected_line_arithmetic_audit"], line_audits["source_field_audit"]
            records.append(dict(filename=filename, control_input_pin=inputs[str(actual_control / filename)],
                source_case_sha256=case["source_case_sha256"], support_sha256=support["support_sha256"],
                cohort_id=case["original_cohort_id"], support=support_name,
                exact_barycentric=N, witness_archive_sha256=witness["archive_sha256"],
                native_public_alias_checks=aliases,
                source_binding_checks=source_checks, **line_audits,
                angular_control_status=record["angular_control"]["status"],
                angular_control_reason=record["angular_control"]["reason"]))
            if not negatives and oracle is not None:
                for field in ("w", "memoryhat"):
                    altered, mutation = _coefficient_negative(witness, field)
                    result = independent.audit_coefficient_fields(raw, N, altered, precision=PRECISION,
                                                                 raise_on_mismatch=False)
                    negatives.append(_negative_record("resealed_source_coefficient_" + field,
                                      altered, mutation, result, fields, field))
                altered, mutation = _projected_negative(witness)
                result = independent.audit_line_witness(raw, N, altered, oracle=oracle,
                                                       raise_on_mismatch=False, **options)
                negatives.append(_negative_record("resealed_private_W_projected_component",
                                  altered, mutation, result, arithmetic))
                for negative in negatives:
                    negative["authentic_source_control_filename"] = filename
                    negative["oracle_input_binding_sha256"] = oracle["oracle_input_binding_sha256"]
            print(json.dumps(dict(stage="SELECTED_LINE_AUDITED", filename=filename,
                field_status=fields["coefficient_source_status"],
                direct_status=arithmetic["source_direct_quadrature_status"],
                source_quadrature_refused=line_audits["source_quadrature_refused"],
                geometry_status=arithmetic["geometry_status"]), sort_keys=True), flush=True)
    require(len(records) == 4 and (len(negatives) == 3 or not any(item["trusted_oracle_available"] for item in records)),
            "Four original lines or available trusted-source mathematical negatives missing")
    after = {path: pin(path) for path in inputs}
    require(after == inputs, "An authoritative source or selected control record changed during audit")
    statuses = _aggregate_statuses(records, negatives)
    report = seal(dict(format="genesis-original-source-selected-line-independent-control-0.1", scope=SCOPE,
        status="PASS" if all(value == "PASS" for value in statuses.values()) else "OPEN",
        source_binding_status="PASS", selected_line_source_field_status=statuses["field"],
        selected_line_direct_quadrature_status=statuses["arithmetic"], selected_line_geometry_status=statuses["geometry"],
        selected_line_represented_polynomial_primitive_status=statuses["represented_primitive"],
        resealed_mathematical_negative_status=statuses["negatives"],
        source_quadrature_refusal_count=sum(item["source_quadrature_refused"] for item in records),
        source_quadrature_refusals_retained_as_unresolved_observations=True,
        refused_observations_used_as_trusted_oracle=False,
        source_fixture_sha256=fixture["source_fixture_sha256"], source_fixture_pin=pin(source_fixture),
        selected_original_line_count=4, mathematical_negative_count=len(negatives),
        expected_mathematical_negative_count=3, cases=records,
        genuine_resealed_mathematical_negatives=negatives,
        input_raw_pins_before=inputs, input_raw_pins_after=after, input_bytes_unchanged=True,
        literal_barycentric_binary_values_retained_without_normalization=True,
        whole_original_parent_work_not_integrated_over_clipped_line=True,
        quadrature_orders=list(QUADRATURE_ORDERS), precision_decimal_digits=PRECISION,
        angular_convergence_status="OPEN", angular_accuracy_certified=False,
        geometry_operand_accuracy_certified=False, general_contraction_accuracy_certified=False,
        physical_contraction_certified=False, physical_creation_admitted=False,
        physical_release_admitted=False, available_for_creation_funding=False,
        source_history_reinitialized=False, accepted_planet_memory_changed=False,
        old_parent_K_M_D_unchanged=True, all_800_birth_memories_unchanged=True,
        public_planetary_restore_count=0, clipped_point_shards_read=False,
        accepted_steps=0, new_material_cohorts=0, funding_debit_j=0.,
        automated_goal_enabled=False, point_one_complete=False,
        elapsed_seconds=time.perf_counter() - started), "report_sha256")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(canonical(report) + b"\n")
    print(json.dumps(dict(status=report["status"], report_sha256=report["report_sha256"],
                         elapsed_seconds=report["elapsed_seconds"]), sort_keys=True), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-fixture", required=True, type=Path)
    parser.add_argument("--actual-control", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.source_fixture, args.actual_control, args.output)
