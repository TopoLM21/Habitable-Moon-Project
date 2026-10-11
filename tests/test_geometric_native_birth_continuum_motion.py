"""Independent spatial virtual work and source/memory replay checks."""
from dataclasses import replace
from decimal import Decimal, localcontext
import math

import numpy as np
import pytest

from tectonics.geometric_native_birth_continuum_motion import (
    _continuum_operator_rows, build_native_birth_continuum_inputs,
    native_birth_continuum_candidate_to_dict, solve_native_birth_continuum_candidate,
)
from tectonics.geometric_native_pose import FractionInterval as I


def _rows():
    frame = tuple(tuple(I(value) for value in row)
        for row in ((0, 0, -1), (0, 1, 0), (-1, 0, 0)))
    return _continuum_operator_rows(start=frame[0], tangent=frame[1], horizontal=frame[2],
        radius_m=11., donor_displacement_m=.4, dip_radians=.37, dimension=8,
        donor_angular_indices=(3, 4, 5), receiver_angular_indices=(0, 1, 2),
        donor_radial_index=6, receiver_radial_index=7, precision=120)


@pytest.mark.parametrize("angle", [0., .31, .91])
def test_whole_arc_coefficients_match_independent_point_virtual_work(angle):
    operators = _rows()
    radial = np.asarray((0., math.sin(angle), -math.cos(angle)))
    horizontal = np.asarray((-1., 0., 0.))
    normal = math.sin(.37)*horizontal+math.cos(.37)*radial
    tangent = math.cos(.37)*horizontal-math.sin(.37)*radial
    q = np.asarray((.003, -.007, .005, -.011, .002, -.013, .017, -.019))
    # Actual independent angular/radial point displacement. Both angular
    # banks share the prescribed objective lever radius of the donor port.
    displacement = 11.4*np.cross(q[:3]-q[3:6], radial)+11.*(q[7]-q[6])*radial
    basis = np.asarray((1., math.cos(angle), math.sin(angle)))
    for coefficients, axis in zip(operators, (normal, tangent)):
        operator = basis@np.asarray(coefficients, dtype=float)
        assert float(operator@q) == pytest.approx(float(axis@displacement), rel=2e-14, abs=1e-15)


def test_every_spatial_coefficient_has_exact_common_rotation_nullspace():
    with localcontext() as context:
        context.prec = 160
        for operator in _rows():
            for row in operator:
                assert all(Decimal(row[index])+Decimal(row[index+3]) == 0 for index in range(3))
                assert Decimal(row[6])+Decimal(row[7]) == 0


def _native_sources():
    from tests.test_geometric_native_implicit_underthrust_motion import get_implicit_second_trial_case
    from tests.test_geometric_native_birth_contact_support import get_birth_support_case
    trial = get_implicit_second_trial_case()
    return trial, get_birth_support_case()["support"]


def test_native_builder_retains_all_original_histories_and_full_carrier_density():
    trial, support = _native_sources()
    inputs, frames = build_native_birth_continuum_inputs(trial["first"]["original"]["mesh"],
        trial["before"], trial["load"], support, float(trial["candidate"].dt_s))
    assert inputs["old_states"] == tuple(state for _, state in trial["before"].states)
    assert len(inputs["old_point_ids"]) == 16
    assert len(inputs["old_stiffness"]) == 14
    assert len(inputs["born_foundations"]) == 5
    assert inputs["born_attachments"] == ()
    assert len(inputs["born_continuum_attachments"]) == len(frames) == 2
    from fractions import Fraction
    with localcontext() as context:
        context.prec = inputs["decimal_precision"]
        for attachment, carrier in zip(inputs["born_continuum_attachments"], support.interface_carriers):
            declaration = attachment.declaration
            density = carrier.inclined_measure_density_m2_per_radian
            assert Decimal(declaration.measure_density_decimal) == Decimal(density.numerator)/Decimal(density.denominator)
            assert declaration.angle_start_decimal == "0"
            assert attachment.previous_state.step_count == 0
            assert Fraction(Decimal(declaration.angle_end_decimal)) > 0
    assert all(frame["material_area_assigned_to_point"] is False for frame in frames)


def test_native_builder_refuses_changed_complete_support_source():
    trial, support = _native_sources()
    changed = replace(support, source_epoch=replace(support.source_epoch, time_myr=support.source_epoch.time_myr+1))
    with pytest.raises(ValueError):
        build_native_birth_continuum_inputs(trial["first"]["original"]["mesh"], trial["before"],
            trial["load"], changed, float(trial["candidate"].dt_s))


def test_first_epoch_adapter_refuses_nonzero_hot_birth_before_constructing_wrong_ports():
    trial, support = _native_sources()
    from tectonics.geometric_native_birth_contact_support import declare_native_birth_contact_support
    declarations = tuple(replace(row, radial_birth_m=.125) if index == 0 else row
        for index, row in enumerate(support.foundation_declarations))
    changed = declare_native_birth_contact_support(support.source_epoch, declarations,
        support.interface_declarations, incoming_geometry=support.incoming_geometry)
    with pytest.raises(ValueError, match="zero radial HOT birth"):
        build_native_birth_continuum_inputs(trial["first"]["original"]["mesh"], trial["before"],
            trial["load"], changed, float(trial["candidate"].dt_s))


def test_genuine_full_arc_candidate_replays_clocks_and_refuses_tampered_memory():
    trial, support = _native_sources()
    mesh, before, load = trial["first"]["original"]["mesh"], trial["before"], trial["load"]
    candidate = solve_native_birth_continuum_candidate(mesh, before, load, support,
        float(trial["candidate"].dt_s), provenance="Independent full-arc native memory/replay check")
    assert candidate.material_admission_authorized is False
    assert candidate.before_state_sha256 == before.state_sha256
    assert len(candidate.original_point_states) == 16
    for (_, previous), (_, current) in zip(before.states, candidate.original_point_states):
        assert current.elapsed_time_s == previous.elapsed_time_s+candidate.dt_s
        assert current.step_count == previous.step_count+1
        assert current.columns_signature == previous.columns_signature
        assert current.area_m2 == previous.area_m2
    packet = native_birth_continuum_candidate_to_dict(candidate, mesh=mesh, previous=before, load=load, support=support)
    assert packet["accepted_new_native_intervals"] == 0
    assert packet["born_finite_pose_branch_admission_proved"] is False
    changed = replace(candidate, spatial_step=replace(candidate.spatial_step,
        memory=replace(candidate.spatial_step.memory, stage_count=candidate.spatial_step.memory.stage_count+1)))
    with pytest.raises(ValueError, match="complete source and memory replay"):
        native_birth_continuum_candidate_to_dict(changed, mesh=mesh, previous=before, load=load, support=support)
