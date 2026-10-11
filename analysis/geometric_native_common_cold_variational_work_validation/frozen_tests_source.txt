"""Whole-source nonlocal cold law and actual affine modal-path work checks."""
from dataclasses import replace
from fractions import Fraction as F
import json
from pathlib import Path

import pytest

from tectonics.geometric_native_pose import FractionInterval as I
from tectonics import geometric_native_common_cold_variational_work as api


@pytest.fixture(scope='module')
def context():
    packet = json.loads((Path(__file__).resolve().parents[1]/api.SOURCE_PATH).read_bytes())
    source = api.bind_native_cold_variational_source(packet)
    return packet, source


def overlap(first, second):
    first = first if type(first) is I else I(first)
    second = second if type(second) is I else I(second)
    assert max(first.lo, second.lo) <= min(first.hi, second.hi)


@pytest.mark.parametrize('label', ['registered_cold_endpoint', 'prepared_zero_cold_reference'])
def test_complete_saved_whole_source_operator(context, label):
    packet, source = context
    old = next(q['response'] for q in packet['evaluation_queries'] if q['label'] == label)
    moments = api.saved_native_cold_strain_moments(source, label)
    current = api.evaluate_native_cold_variational_work(source, moments)
    assert current.whole_original_integral_certified
    assert current.source_beams_included_once
    assert len(current.gradient) == 56 and len(current.hessian) == 56
    overlap(current.energy_j, api._saved(old['energy_j']))
    for i in range(56):
        overlap(current.gradient[i], api._saved(old['gradient_n'][i]))
        for j in range(56):
            overlap(current.hessian[i][j], api._saved(old['hessian_n_m'][i][j]))
            assert current.hessian[i][j] == current.hessian[j][i]
    assert not current.reference_preparation_work_paid
    assert not current.historical_three_dimensional_prestress_recovered
    assert not current.whole_clock_all_pair_admission_authorized
    assert current.accepted_new_native_intervals == 0


def test_zero_retains_literal_old_energy_and_all_eight_forces_once(context):
    _, source = context
    result = api.evaluate_native_cold_variational_work(source,
        api.saved_native_cold_strain_moments(source, 'prepared_zero_cold_reference'))
    overlap(result.energy_j, sum(source.literal_reference_energies_j, F(0)))
    for index, force in zip(source.tip_coordinate_indices, source.literal_reference_forces_n):
        overlap(result.gradient[index], force)
        assert (result.gradient[index].width if type(result.gradient[index]) is I else F(0)) == 0


def declared_tip_curvature(source, n=100):
    z = F(0); one = F(1)
    zero = ((z,)*n,)*n
    dt = [(z,)*n for _ in range(8)]
    dt[0] = (one,)+(z,)*(n-1)
    ddt = [zero for _ in range(8)]
    mixed = [[z]*n for _ in range(n)]; mixed[0][1] = mixed[1][0] = F(3)
    ddt[0] = tuple(tuple(row) for row in mixed)
    return api.declare_native_cold_strain_moments(source,
        coordinate_order=tuple(('declared_mode', str(i)) for i in range(n)),
        tip_coefficients_m=(F(2),)+(z,)*7, tip_first=tuple(dt), tip_second=tuple(ddt),
        remainder_projection_n=(z,)*8, remainder_projection_first=((z,)*n,)*8,
        remainder_projection_second=(zero,)*8, half_remainder_norm_j=z,
        half_remainder_norm_first=(z,)*n, half_remainder_norm_second=zero,
        provenance='Pure tip-span physical strain t0=2+x+3xy; no new domain certificate')


def test_full_hundred_channel_curvature_includes_literal_prestress(context):
    _, source = context
    moments = declared_tip_curvature(source)
    result = api.evaluate_native_cold_variational_work(source, moments)
    force = source.literal_reference_forces_n[0]+2*source.tip_stiffnesses_n_m[0]
    expected_energy = sum(source.literal_reference_energies_j, F(0))\
        +2*source.literal_reference_forces_n[0]+2*source.tip_stiffnesses_n_m[0]
    overlap(result.energy_j, expected_energy)
    overlap(result.hessian[0][1], 3*force)
    overlap(result.global_projection_curvature_hessian[0][1], 3*force)
    overlap(result.hessian[0][0], source.tip_stiffnesses_n_m[0])
    assert len(result.hessian) == 100
    assert not result.whole_original_integral_certified
    assert result.authentic_saved_label is None
    for i in range(2,100):
        overlap(result.gradient[i], F(0))
        for j in range(100):
            assert result.hessian[i][j] == I(0)


@pytest.mark.parametrize('lam', [F(0), F(1,3), F(1)])
def test_whole_source_affine_path_work_and_reaction_sign(context, lam):
    packet, source = context
    result = api.evaluate_native_cold_affine_modal_path_work(source, lam)
    work = api._saved(result['potential_path_work_j'])
    assert api._saved(result['physical_reaction_path_work_j']) == -work
    coeff = tuple(api._saved(x) for x in result['energy_polynomial_coefficients_j'])
    overlap(api._saved(result['energy_j'])-coeff[0], work)
    if lam in (F(0),F(1)):
        label = 'prepared_zero_cold_reference' if not lam else 'registered_cold_endpoint'
        old = next(q['response'] for q in packet['evaluation_queries'] if q['label'] == label)
        overlap(api._saved(result['energy_j']), api._saved(old['energy_j']))
    if not lam:
        assert work == I(0)
    assert result['all_eight_original_parents_and_three_archive_cohorts_covered']
    assert result['fixed895_coordinate_envelope_preserved']
    assert result['physical_modal_motion_defined']
    assert not result['physical_temporal_path_admission_certified']
    assert not result['reference_preparation_work_paid']
    assert api.native_cold_affine_modal_path_work_from_dict(result, source=source,
        expected_lambda_fraction=lam) == result


def test_state_change_preserves_old_beam_energy_and_work_sign(context):
    _, source = context
    work = api.compare_saved_native_cold_state_work(source,
        'prepared_zero_cold_reference', 'registered_cold_endpoint')
    assert api._saved(work['physical_reaction_work_j']) == -api._saved(work['potential_energy_change_j'])
    assert not work['temporal_path_certified']
    assert not work['reference_preparation_work_paid']


def test_complete_saved_moment_and_response_restart(context):
    _, source = context
    moments = api.saved_native_cold_strain_moments(source, 'registered_cold_endpoint')
    dto = api.native_cold_strain_moments_to_dict(source, moments)
    restored = api.native_cold_strain_moments_from_dict(dto, source=source)
    assert restored == moments
    result = api.evaluate_native_cold_variational_work(source, restored)
    dto = api.native_cold_variational_work_to_dict(result)
    restored_result = api.native_cold_variational_work_from_dict(dto, source=source, moments=restored)
    assert restored_result == result


def test_source_complete_restart_and_typed_rejection(context):
    packet, source = context
    dto = api.native_cold_variational_source_to_dict(source)
    assert api.native_cold_variational_source_from_dict(dto, source_packet=packet) is source
    changed = dict(dto); changed['source_beams_included_once'] = True
    with pytest.raises(ValueError):
        api.native_cold_variational_source_from_dict(changed, source_packet=packet)
    with pytest.raises(ValueError):
        api.saved_native_cold_strain_moments(source, 'fictional_reference_preparation')


def test_altered_moment_claim_cannot_gain_whole_material_certification(context):
    _, source = context
    moments = declared_tip_curvature(source, 2)
    raw = api.native_cold_strain_moments_to_dict(source, moments)
    raw['whole_original_integral_certified'] = True
    with pytest.raises(ValueError):
        api.native_cold_strain_moments_from_dict(raw, source=source)
    forged = replace(moments, whole_original_integral_certified=True)
    with pytest.raises(ValueError):
        api.evaluate_native_cold_variational_work(source, forged)
    raw = api.native_cold_strain_moments_to_dict(source, moments)
    raw['tip_coefficients_m'][0][0] = True
    with pytest.raises(ValueError):
        api.native_cold_strain_moments_from_dict(raw, source=source)


@pytest.mark.parametrize('lam', [True,F(-1),F(2),0.5])
def test_path_rejects_nonliteral_or_outside_parameters(context, lam):
    _, source = context
    with pytest.raises(ValueError):
        api.evaluate_native_cold_affine_modal_path_work(source, lam)


def test_declared_nonphysical_negative_orthogonal_norm_is_rejected(context):
    _, source = context
    moments = declared_tip_curvature(source, 2)
    raw = api.native_cold_strain_moments_to_dict(source, moments)
    raw['half_remainder_norm_j'] = [-1,1]
    with pytest.raises(ValueError):
        api.native_cold_strain_moments_from_dict(raw, source=source)


def test_frozen_object_bypass_mutation_cannot_change_certified_moment(context):
    _, source = context
    moments = declared_tip_curvature(source, 2)
    original = moments.whole_original_integral_certified
    try:
        object.__setattr__(moments, 'whole_original_integral_certified', True)
        with pytest.raises(ValueError):
            api.evaluate_native_cold_variational_work(source, moments)
    finally:
        object.__setattr__(moments, 'whole_original_integral_certified', original)
    assert api.native_cold_strain_moments_to_dict(source, moments)['whole_original_integral_certified'] is False
