"""Read-only source-integral control on the four retained Genesis supports.

Independent rational certificates authenticate selected lines. Every angular
callback uses a freshly constructed directed source-integral enclosure, and
its uncertainty is charged against the original component tolerance. Embedded
angular agreement remains an observation; this control admits no material.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .source_control import (ROOT, canonical, pin, require, seal,
                             load_authenticated_inputs, support_source_inputs)
from .independent_integrals import (build_true_source_integral_certificate,
    verify_true_source_integral_certificate, verify_runtime_enclosures)
from .angular_control import integrate_angular, NumericalLineBoundRefusal, VOLUME_NAMES
from tectonics.genesis_volume_exact_lambda import reference_side_volumes_m3
from tectonics.genesis_volume_exact_lambda_bulk import prepare_source, prepare_line_evaluator
from tectonics.genesis_volume_integral_bounds import prepare_integral_bounds

LD = np.longdouble
ANGULAR_OPTIONS = dict(relative_tolerance=2e-6, max_cells=2000, max_depth=24,
                       angular_orders=(3, 5), max_line_evaluations=3400)
WORKERS = 4


def _native(value):
    value = Fraction(value)
    result = LD(value.numerator)/LD(value.denominator)
    require(Fraction(*result.as_integer_ratio()) == value,
            'Original binary operand cannot be recovered exactly')
    return result


def _ratios(values):
    return [[str(a), str(b)] for a, b in
            (LD(v).as_integer_ratio() for v in np.asarray(values).flat)]


def _write(path, value):
    with path.open('xb') as stream:
        stream.write(canonical(value)+b'\n')
    require(json.loads(path.read_bytes()) == value, 'Literal JSON readback differs')
    return dict(path=path.name, **pin(path))


def _values(native):
    volume = (np.zeros(2, dtype=LD) if native.interval is None else
              np.array(list(reference_side_volumes_m3(native.interval).values()), dtype=LD))
    return np.concatenate((volume, native.values))


def _scope():
    return dict(source_endpoint_displacement_propagated=True,
        angular_accuracy_certified=False, general_contraction_accuracy_certified=False,
        physical_contraction_certified=False, physical_creation_admitted=False,
        physical_release_admitted=False, available_for_creation_funding=False,
        history_reinitialized=False, accepted_planet_memory_changed=False,
        automated_goal_enabled=False)


def _case(ordinal, item, output):
    started = time.monotonic()
    directory = Path(output)/f'case_{ordinal:03d}_{item["label"]}'
    directory.mkdir()
    raw = support_source_inputs(item['case'], item['label'])
    options = dict(added_basis=item['support']['original_declared_added_basis'],
                   parent_basis=item['support']['original_declared_parent_basis'],
                   probes=item['support']['declared_probes'])
    source = prepare_source(raw)
    parent_bytes = (source.parent_stiffness.tobytes(), source.parent_mass.tobytes())
    saved_parent_bytes = tuple(np.asarray(raw['previous_local_bulk_blocks'][key],
        dtype=np.float64).tobytes() for key in ('parent_stiffness','parent_mass'))
    require(parent_bytes == saved_parent_bytes,
            'Fresh parent K/M differs from retained accepted Genesis bytes')
    native_evaluator = prepare_line_evaluator(source, **options)
    bounded_evaluator = prepare_integral_bounds(source, raw, **options)
    names = VOLUME_NAMES+native_evaluator.names
    require(names == bounded_evaluator.names, 'Original full/W15/probe schema differs')
    N = [Fraction(value) for value in
         item['report']['line_witness_barycentric_exact_binary_ratios']]
    native_N = np.array([_native(v) for v in N], dtype=LD)
    native = native_evaluator(native_N)
    require(native.interval is not None, 'Original selected positive line became empty')
    selected_bounds = bounded_evaluator.evaluate(native_N, native.interval, witness=False)
    require(bounded_evaluator.verify(selected_bounds, native_N, native.interval),
            'Canonical runtime bounds differ')
    values = _values(native)
    errors = selected_bounds.absolute_error_for(values)
    proof_started = time.monotonic()
    proof = build_true_source_integral_certificate(raw, N, item['line_witness'], **options)
    proof_seconds = time.monotonic()-proof_started
    verify_started = time.monotonic()
    verification = verify_true_source_integral_certificate(raw, N, item['line_witness'],
                                                            proof, **options)
    verify_seconds = time.monotonic()-verify_started
    targets = {name: tuple(Fraction(int(v[0]), int(v[1])) for v in
                          (entry['true_source_integral_enclosure']['lower'],
                           entry['true_source_integral_enclosure']['upper']))
               for name, entry in proof['components'].items()}
    comparison = verify_runtime_enclosures(targets, selected_bounds.lower,
                                           selected_bounds.upper, names)
    runtime = seal(dict(format='genesis-selected-source-integral-runtime-bounds-0.1',
        status=comparison['status'], component_names=list(names),
        component_count=len(names), original_N_exact_binary_ratios=_ratios(native_N),
        source_binding_sha256=selected_bounds.source_binding_sha256,
        line_operand_sha256=selected_bounds.line_operand_sha256,
        geometry_operand_sha256=selected_bounds.geometry_operand_sha256,
        native_values_exact_binary_ratios=_ratios(values),
        lower_exact_binary_ratios=_ratios(selected_bounds.lower),
        upper_exact_binary_ratios=_ratios(selected_bounds.upper),
        source_integral_absolute_error_bounds_exact_binary_ratios=_ratios(errors),
        canonical_runtime_construction_verification='PASS',
        independent_source_enclosure_comparison=comparison,
        radial_integral_accuracy_certified=True,
        historical_saved_budget_authentication_certified=False,
        scope=selected_bounds.scope, **_scope()), 'integral_bounds_sha256')
    files = dict(true_source_certificate=_write(directory/'true_source_certificate.json', proof),
                 runtime_integral_bounds=_write(directory/'runtime_integral_bounds.json', runtime))
    print(json.dumps(dict(event='selected_source_integrals', case_ordinal=ordinal,
        status=comparison['status'], components=len(names),
        canonical_runtime='PASS', seconds=time.monotonic()-started), sort_keys=True), flush=True)
    require(comparison['status'] == verification['status'] == 'PASS',
            'Independent selected source proof or runtime containment refused')

    counts = dict(attempted=0, completed=0, positive_native=0, empty_native=0)
    rolling = hashlib.sha256()
    angular_started = time.monotonic()

    def evaluator(node):
        counts['attempted'] += 1
        try:
            line = native_evaluator(node)
            bounds = bounded_evaluator.evaluate(node, line.interval, witness=False)
            require(bounds.names == names and bounds.source_binding_sha256 == source.binding_sha256
                and bounds.radial_integral_accuracy_certified is True
                and bounds.angular_integral_convergence_certified is False
                and bounds.physical_contraction_certified is False
                and bounds.available_for_creation_funding is False,
                'Fresh source-bound construction scope differs')
            value = _values(line)
            error = bounds.absolute_error_for(value)
        except ValueError as failure:
            raise NumericalLineBoundRefusal(type(failure).__name__+': '+str(failure),
                                            barycentric=node) from failure
        counts['completed'] += 1
        counts['positive_native' if line.interval is not None else 'empty_native'] += 1
        # Literal ratios avoid nondeterministic padding in extended-float bytes.
        rolling.update(canonical(dict(N=_ratios(node), values=_ratios(value),
            lower=_ratios(bounds.lower), upper=_ratios(bounds.upper),
            absolute_source_errors=_ratios(error), binding=bounds.line_operand_sha256)))
        rolling.update(b'\n')
        if counts['completed'] % 100 == 0:
            print(json.dumps(dict(event='bounded_angular_nodes', case_ordinal=ordinal,
                completed=counts['completed'], positive=counts['positive_native'],
                seconds=time.monotonic()-angular_started), sort_keys=True), flush=True)
        return value, error

    try:
        angular = integrate_angular(raw, evaluator, names, **ANGULAR_OPTIONS).to_dict()
        require(angular['radial_line_evaluations'] == counts['completed'],
                'Angular and source-bound enumerations differ')
    except NumericalLineBoundRefusal as refusal:
        angular = refusal.to_dict()
        require(angular['completed_source_bound_line_evaluations'] == counts['completed']
            and angular['attempted_line_evaluations'] == counts['attempted'],
            'Refused angular and source-bound enumerations differ')
    angular = seal(angular, 'angular_report_sha256')
    files['angular_report'] = _write(directory/'angular_report.json', angular)
    require(parent_bytes == (source.parent_stiffness.tobytes(), source.parent_mass.tobytes()),
            'Original parent K/M bytes changed during observation')
    row = seal(dict(case_ordinal=ordinal,
        original_cohort_id=item['case']['original_cohort_id'],
        original_face_index=item['case']['original_face_index'], support=item['label'],
        original_radius_m=item['support']['seed_radius_m'],
        original_report_pin=item['report_pin'],
        preceding_operand_proof_pins=item['prior_operand_proof_pins'],
        original_N_exact_binary_ratios=_ratios(native_N), files=files,
        selected_source_integral_status=comparison['status'],
        checked_component_count=proof['checked_component_count'],
        checked_saved_component_count=proof['checked_saved_component_count'],
        source_certificate_canonical_verification=verification,
        runtime_bound_canonical_verification='PASS',
        retained_parent_K_M_literal_byte_comparison='PASS',
        independent_runtime_containment=comparison,
        historical_saved_budget_source_integral_sufficiency_status=
            proof['historical_saved_budget_source_integral_sufficiency_status'],
        historical_saved_budget_source_integral_sufficiency_failure_count=
            len(proof['historical_saved_budget_source_integral_sufficiency_failures']),
        source_certificate_build_seconds=proof_seconds,
        source_certificate_verify_seconds=verify_seconds,
        angular=angular, angular_seconds=time.monotonic()-angular_started,
        angular_node_counts=counts, evaluated_node_bounds_trace_sha256=rolling.hexdigest(),
        trace_scope='Ordered digest of evaluated literal nodes, values, fresh enclosures and errors; not stored node witnesses or global angular proof.',
        evaluated_angular_nodes_source_bounds_supplied=True,
        source_bounds_added_to_acceptance_allowance=False,
        old_angular_report_status=item['report'].get('numerical_status','OPEN'),
        old_angular_report_reclassified=False, original_whole_parent_K_M_bytes_changed=False,
        total_case_seconds=time.monotonic()-started, **_scope()), 'case_report_sha256')
    _write(directory/'case_report.json', row)
    print(json.dumps(dict(event='case_complete', case_ordinal=ordinal,
        selected=row['selected_source_integral_status'], angular=angular['status'],
        completed_nodes=counts['completed'], seconds=row['total_case_seconds']), sort_keys=True), flush=True)
    return row


def run(output):
    require(np.finfo(LD).nmant == 63, 'Control requires a 64-bit native significand')
    require(output.is_relative_to(Path('/workspace')) and not output.is_relative_to(ROOT)
            and not output.exists(), 'Output must be fresh and external')
    output.mkdir()
    started = time.monotonic()
    request = seal(dict(format='genesis-source-integral-control-request-0.1',
        selected_support_count=4, angular_options={**ANGULAR_OPTIONS, 'angular_orders':[3,5]},
        independent_case_processes=WORKERS, numeric_threads_per_process=1,
        component_schema='2 original reference-bank volumes + 2534 original full18/W15/5probe components',
        node_budget_and_original_tolerance_preserved=True,
        old_angular_refusals_reclassified=False, **_scope()), 'request_sha256')
    _write(output/'request.json', request)
    inputs = load_authenticated_inputs()
    _write(output/'authentication.json', inputs['authentication'])
    _write(output/'source_fixture.json', inputs['source_fixture'])
    items = [{k:v for k,v in item.items() if k != 'prior_operand_proofs'}
             for item in inputs['supports']]
    cases = []
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        futures = [executor.submit(_case, ordinal, item, str(output))
                   for ordinal, item in enumerate(items)]
        for future in as_completed(futures):
            cases.append(future.result())
    cases.sort(key=lambda c:c['case_ordinal'])
    checks = len(cases) == 4 and all(c['selected_source_integral_status'] == 'PASS'
        and c['source_certificate_canonical_verification']['status'] == 'PASS'
        and c['runtime_bound_canonical_verification'] == 'PASS' for c in cases)
    angular_finite = all(c['angular']['status'] == 'FINITE_AGREEMENT' for c in cases)
    report = seal(dict(format='genesis-source-integral-actual-control-0.1',
        status='COMPLETED_WITH_NUMERICAL_REFUSALS' if not angular_finite else 'COMPLETED',
        selected_source_integral_proof_status='PASS' if checks else 'OPEN',
        numerical_status='FINITE_AGREEMENT' if angular_finite else 'OPEN',
        source_authentication_sha256=inputs['authentication']['sha256'],
        source_fixture_sha256=inputs['source_fixture']['source_fixture_sha256'],
        control_request_sha256=request['request_sha256'],
        prior_source_count=458, source_time_myr=0.8684000000000005,
        original_cohort_count=800, original_accepted_interval_count=4,
        selected_line_count=len(cases),
        checked_source_integral_component_count=sum(c['checked_component_count'] for c in cases),
        checked_saved_component_count=sum(c['checked_saved_component_count'] for c in cases),
        new_angular_line_calls=sum(c['angular_node_counts']['attempted'] for c in cases),
        completed_angular_source_bound_line_calls=sum(c['angular_node_counts']['completed'] for c in cases),
        cases=cases, body_seconds=time.monotonic()-started,
        proof_scope='Canonical selected true-source integrals plus freshly directed bounds at every completed literal angular node; embedded angular observations only.',
        full_contraction_uniform_forward_bounds_certified=False,
        evaluated_angular_nodes_source_bounds_supplied=True,
        source_bounds_added_to_acceptance_allowance=False,
        old_angular_refusals_reclassified=False,
        new_Genesis_time_steps=0, new_material_cohorts=0, public_planetary_restore_count=0,
        original_whole_parent_K_M_bytes_changed=False,
        physical_Gamma_activated=False, prospective_source_law_installed=False,
        **_scope()), 'report_sha256')
    _write(output/'control_report.json', report)
    print(json.dumps(dict(selected=report['selected_source_integral_proof_status'],
        numerical_status=report['numerical_status'],
        source_integrals=report['checked_source_integral_component_count'],
        angular_nodes=report['completed_angular_source_bound_line_calls'],
        body_seconds=report['body_seconds'], report_sha256=report['report_sha256']), sort_keys=True), flush=True)
    return 0 if checks else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    raise SystemExit(run(parser.parse_args().output.resolve()))


if __name__ == '__main__':
    main()
