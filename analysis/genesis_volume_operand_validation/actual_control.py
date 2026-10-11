"""Exact operand proofs on the four unchanged, archived Genesis lines.

This read-only control authenticates genuine birth memory and the accepted
parent. It proves captured coefficients and contractions with rational
arithmetic, checks new directed coefficient enclosures, and retains source
geometry displacement separately. It does not rerun angular integration or
advance, initialize, activate, or admit a planetary state.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
from pathlib import Path
import time

import numpy as np

from .source_control import (ROOT, canonical, pin, require, seal,
                             load_authenticated_inputs, support_source_inputs)
from .independent_coefficients import (audit_coefficients, decode_native_array,
    encode_rational_array, exact_source_coefficients, verify_coefficient_proof)
from .independent_geometry_bounds import audit_geometry_operands
from .independent_contractions import (build_contraction_certificate,
                                      verify_contraction_certificate)
from tectonics.genesis_volume_exact_lambda import prepare_fixed_N_interval
from tectonics.genesis_volume_exact_lambda_bulk import prepare_source
from tectonics.genesis_volume_operand_bounds import prepare_operand_bounds

LD = np.longdouble


def _native(value):
    """Recover a captured native binary number, requiring exact recovery."""
    value = Fraction(value)
    result = LD(value.numerator)/LD(value.denominator)
    require(Fraction(*result.as_integer_ratio()) == value,
            'Captured native operand cannot be recovered exactly')
    return result


def _fractions(array):
    array = np.asarray(array, dtype=LD)
    return np.asarray([Fraction(*v.as_integer_ratio()) for v in array.flat],
                      dtype=object).reshape(array.shape)


def _write(path, value):
    with path.open('xb') as stream:
        stream.write(canonical(value)+b'\n')
    require(json.loads(path.read_bytes()) == value, 'Literal JSON readback differs')
    return dict(path=path.name, **pin(path))


def directed_proof(raw, N, witness):
    source = prepare_source(raw)
    native_N = np.asarray([_native(v) for v in N], dtype=LD)
    interval = prepare_fixed_N_interval(source.geometry, native_N)
    require(interval is not None, 'Authenticated positive line became empty')
    lm = decode_native_array(witness, 'interval.lambda_mid').item()
    lh = decode_native_array(witness, 'interval.lambda_halfwidth').item()
    require(Fraction(*interval.lambda_mid.as_integer_ratio()) == lm
            and Fraction(*interval.lambda_halfwidth.as_integer_ratio()) == lh,
            'Fresh represented interval differs from captured original')
    certificate = prepare_operand_bounds(source).evaluate(interval)
    exact = exact_source_coefficients(raw, N, lm, lh)
    native = {name: np.asarray([_native(v) for v in
        decode_native_array(witness, 'coefficient.'+name).flat], dtype=LD).reshape(exact[name].shape)
        for name in exact}
    errors = certificate.error_bounds_for(native)
    fields, failures, count, larger = {}, [], 0, 0
    for name, expected in exact.items():
        lower, upper = _fractions(certificate.fields[name].lower), _fractions(certificate.fields[name].upper)
        error = _fractions(errors[name])
        values = _fractions(native[name])
        old = decode_native_array(witness, 'coefficient_bound.'+name)
        indices = [list(i) for i in np.ndindex(expected.shape)
                   if not (lower[i] <= expected[i] <= upper[i]
                           and abs(values[i]-expected[i]) <= error[i])]
        failures += [dict(family=name, index=i) for i in indices]
        over = [list(i) for i in np.ndindex(expected.shape) if error[i] > old[i]]
        larger += len(over)
        count += expected.size
        fields[name] = dict(shape=list(expected.shape),
            lower=encode_rational_array(lower), upper=encode_rational_array(upper),
            native_coefficient_forward_error=encode_rational_array(error),
            exact_oracle_enclosed=not indices, failed_indices=indices,
            new_bound_larger_than_old_budget_indices=over)
    body = dict(format='genesis-directed-coefficient-control-0.1',
        status='PASS' if not failures else 'OPEN', scope=certificate.scope,
        source_binding_sha256=certificate.source_binding_sha256,
        line_operand_sha256=certificate.line_operand_sha256,
        coefficient_count=int(count), exact_oracle_enclosure_failure_count=len(failures),
        exact_oracle_enclosure_failures=failures,
        new_bound_larger_than_old_budget_count=larger,
        old_advertised_budget_authentication_certified=False,
        coefficient_construction_accuracy_certified=not failures,
        source_geometry_displacement_propagated=False,
        radial_integral_accuracy_certified=False,
        angular_node_contraction_forward_bounds_certified=False,
        angular_integral_accuracy_certified=False,
        physical_contraction_certified=False, physical_creation_admitted=False,
        physical_release_admitted=False, available_for_creation_funding=False,
        fields=fields)
    return seal(body, 'directed_bounds_sha256')


def run(output):
    require(np.finfo(LD).nmant == 63, 'This control requires 64-bit native significand')
    require(output.is_relative_to(Path('/workspace')) and not output.is_relative_to(ROOT)
            and not output.exists(), 'Control output must be fresh and external')
    output.mkdir()
    started = time.monotonic()
    inputs = load_authenticated_inputs()
    _write(output/'authentication.json', inputs['authentication'])
    _write(output/'source_fixture.json', inputs['source_fixture'])
    cases = []
    for ordinal, item in enumerate(inputs['supports']):
        case_started = time.monotonic()
        raw = support_source_inputs(item['case'], item['label'])
        witness = item['line_witness']
        N = [Fraction(value) for value in
             item['report']['line_witness_barycentric_exact_binary_ratios']]
        options = dict(added_basis=item['support']['original_declared_added_basis'],
            parent_basis=item['support']['original_declared_parent_basis'],
            probes=item['support']['declared_probes'])
        directory = output/f'case_{ordinal:03d}_{item["label"]}'
        directory.mkdir()
        coefficient = audit_coefficients(raw, N, witness)
        require(verify_coefficient_proof(raw, N, witness, coefficient), 'Coefficient proof refusal')
        geometry = seal(audit_geometry_operands(raw, N, witness), 'geometry_proof_sha256')
        directed = directed_proof(raw, N, witness)
        contraction_started = time.monotonic()
        contraction = build_contraction_certificate(raw, N, witness, **options)
        construction_seconds = time.monotonic()-contraction_started
        verification_started = time.monotonic()
        verification = verify_contraction_certificate(raw, N, witness, contraction, **options)
        verification_seconds = time.monotonic()-verification_started
        files = {}
        for name, value in (('coefficient_proof',coefficient), ('geometry_proof',geometry),
                            ('directed_bounds',directed), ('contraction_certificate',contraction)):
            files[name] = _write(directory/(name+'.json'), value)
        previous = item['report']
        row = dict(case_ordinal=ordinal, original_cohort_id=item['case']['original_cohort_id'],
            original_face_index=item['case']['original_face_index'], support=item['label'],
            original_radius_m=item['support']['seed_radius_m'],
            original_report_pin=item['report_pin'], files=files,
            original_N_exact_binary_ratios=[[str(v.numerator),str(v.denominator)] for v in N],
            coefficient_count=coefficient['coefficient_count'],
            exact_native_coefficient_count=coefficient['exact_native_coefficient_count'],
            coefficient_saved_budget_status=coefficient['saved_coefficient_budget_status'],
            coefficient_saved_budget_failure_count=coefficient['saved_coefficient_budget_failure_count'],
            coefficient_canonical_verification='PASS', geometry_proof_status=geometry['status'],
            geometry_source_error_enclosures_certified=geometry['source_geometry_error_bounds_certified'],
            directed_coefficient_status=directed['status'],
            directed_coefficient_failure_count=directed['exact_oracle_enclosure_failure_count'],
            new_directed_bound_larger_than_old_budget_count=directed['new_bound_larger_than_old_budget_count'],
            contraction_component_count=contraction['checked_component_count'],
            contraction_saved_represented_budget_status=contraction['historical_saved_bound_represented_sufficiency_status'],
            contraction_saved_selected_interval_budget_status=contraction['historical_saved_bound_selected_interval_sufficiency_status'],
            contraction_saved_selected_interval_budget_failure_count=len(contraction['historical_saved_bound_selected_interval_sufficiency_failures']),
            contraction_canonical_verification=verification,
            contraction_build_seconds=construction_seconds, contraction_verify_seconds=verification_seconds,
            total_case_seconds=time.monotonic()-case_started,
            old_angular_numerical_status=previous['numerical_status'] if 'numerical_status' in previous else 'OPEN',
            old_angular_report_reclassified=False,
            source_geometry_displacement_propagated=False,
            angular_node_full_contraction_bounds_certified=False,
            geometry_contraction_accuracy_certified=False, angular_accuracy_certified=False,
            physical_creation_admitted=False, physical_release_admitted=False,
            available_for_creation_funding=False)
        row = seal(row, 'case_report_sha256')
        _write(directory/'case_report.json', row)
        cases.append(row)
        print(json.dumps(dict(case_ordinal=ordinal, cohort=row['original_cohort_id'],
            support=row['support'], coefficient_status=row['coefficient_saved_budget_status'],
            geometry_status=row['geometry_proof_status'], directed_status=row['directed_coefficient_status'],
            contraction_status=row['contraction_saved_selected_interval_budget_status'],
            canonical_verification=verification['status'], seconds=row['total_case_seconds']),
            sort_keys=True), flush=True)
    checks = all(case['coefficient_saved_budget_failure_count'] == 0
        and case['geometry_proof_status'] == 'PASS' and case['directed_coefficient_failure_count'] == 0
        and case['contraction_saved_selected_interval_budget_failure_count'] == 0
        and case['contraction_canonical_verification']['status'] == 'PASS' for case in cases)
    report = seal(dict(format='genesis-volume-operand-actual-control-0.1',
        selected_operand_proof_status='PASS' if checks else 'OPEN', numerical_status='OPEN',
        source_authentication_sha256=inputs['authentication']['sha256'],
        source_fixture_sha256=inputs['source_fixture']['source_fixture_sha256'],
        prior_source_count=447, source_time_myr=0.8684000000000005,
        original_cohort_count=800, original_accepted_interval_count=4,
        selected_line_count=len(cases), checked_coefficient_count=sum(c['coefficient_count'] for c in cases),
        checked_contraction_component_count=sum(c['contraction_component_count'] for c in cases),
        cases=cases, body_seconds=time.monotonic()-started,
        proof_scope='Exact selected-line source algebra on captured represented endpoints; separate source geometry error enclosures; reusable directed coefficient bounds.',
        canonical_certificate_verification='Independent exact arithmetic recomputed for each of all four certificates; JSON write/readback also checked.',
        source_geometry_displacement_propagated=False,
        full_contraction_uniform_forward_bounds_certified=False,
        new_angular_line_calls=0, old_angular_refusals_reclassified=False,
        angular_accuracy_certified=False, general_contraction_accuracy_certified=False,
        new_Genesis_time_steps=0, new_material_cohorts=0, public_planetary_restore_count=0,
        history_reinitialized=False, accepted_planet_memory_changed=False,
        original_whole_parent_K_M_bytes_changed=False, automated_goal_enabled=False,
        physical_Gamma_activated=False, prospective_source_law_installed=False,
        physical_creation_admitted=False, physical_release_admitted=False,
        available_for_creation_funding=False), 'report_sha256')
    _write(output/'control_report.json', report)
    print(json.dumps(dict(selected_operand_proof_status=report['selected_operand_proof_status'],
        numerical_status=report['numerical_status'], coefficients=report['checked_coefficient_count'],
        contractions=report['checked_contraction_component_count'], body_seconds=report['body_seconds'],
        report_sha256=report['report_sha256']), sort_keys=True), flush=True)
    return 0 if checks else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    raise SystemExit(run(parser.parse_args().output.resolve()))


if __name__ == '__main__':
    main()
