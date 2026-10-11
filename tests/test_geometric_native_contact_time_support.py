"""Independent controls for declared constitutive contact-time diagnostics."""
from dataclasses import asdict, replace
from fractions import Fraction
import json
import math

import pytest

from tectonics.geometric_contact_mechanics import solve_compressed_contact
from tectonics.geometric_native_contact_time_support import (
    ADMISSIBLE, MODEL, NativeContactTimePath, _bounds, _field, _sin_cos_point,
    assess_native_contact_time_support, build_native_contact_time_path,
    native_contact_time_support_to_dict, native_quarter_material_coordinate_bounds,
)
from tectonics.geometric_native_pose import FractionInterval as I, interval_pi
from tectonics.geometric_spatial_underthrust import (
    TrigPolynomial, _angles, advance_spatial_underthrust_memory,
    initialize_spatial_underthrust_memory,
)
from tests.underthrust_case import physical_inputs


def constant(value):
    return TrigPolynomial((value, 0., 0., 0., 0.))


def seed_for(**options):
    return physical_inputs(**options)[3]


def previous_fields(seed):
    return tuple((t.trace_id, constant(t.initial_gap_m), constant(t.initial_jump_m))
                 for t in seed.fault_traces.traces)


def endpoint(seed, gap, jump, *, duration=1e6, previous=None):
    memory = initialize_spatial_underthrust_memory(seed) if previous is None else previous.memory
    states = seed.states if previous is None else previous.current_point_states
    old = dict(states)
    fields = tuple((trace.trace_id, gap, jump) for trace in seed.fault_traces.traces)
    current = tuple((identity, solve_compressed_contact(columns=trace.columns,
        law=trace.law, area_m2=old[identity].area_m2, gap_free_m=gap(theta),
        jump_free_m=jump(theta), dt_s=duration, water=trace.water,
        previous=old[identity]).state) for identity, trace, theta in _angles(seed))
    return advance_spatial_underthrust_memory(seed, memory, fields, duration,
        previous_point_states=states, current_point_states=current)


def path_for(gap=-.1, jump=-2., *, duration=1e6, **options):
    seed = seed_for(**options)
    step = endpoint(seed, constant(gap), constant(jump), duration=duration)
    return build_native_contact_time_path(seed, step, previous_fields(seed))


def support(path, first=0, last=1, *, low=0, high=None, precision=80):
    trace = path.step.certificates[0]
    return assess_native_contact_time_support(path, trace.trace_id, low,
        trace.extent_radians if high is None else high, first, last, precision=precision)


def test_whole_interval_preserves_prepared_preload_memory_work_and_endpoint():
    path = path_for()
    before = asdict(path)
    receipt = support(path)
    assert receipt.status == ADMISSIBLE and receipt.admissible
    assert receipt.pressure_bounds_pa.lo > 0
    assert receipt.plastic_increment_bounds_m == I(-2, 0)
    assert receipt.plastic_rate_bounds_m_s == I(Fraction(-2, 1_000_000))
    assert asdict(path) == before
    assert path.step.previous_memory.stage_count == 0
    assert all(local.gap_free_m == -.1 and local.plastic_slip_m == 0.
               for _, local in path.step.previous_point_states)


def test_inviscid_time_zero_uses_continuous_right_limit_without_memory_update():
    receipt = support(path_for(), 0, 0)
    assert receipt.admissible
    assert receipt.plastic_increment_bounds_m == I(0)
    assert receipt.plastic_rate_bounds_m_s.hi < 0


def test_fraction_time_far_below_binary64_normal_is_not_rounded_to_zero():
    time = Fraction(1, 10**400)
    receipt = support(path_for(), time, time)
    assert receipt.admissible
    assert receipt.time_fraction_bounds == (time, time)
    assert receipt.plastic_increment_bounds_m == I(-2*time)
    assert receipt.down_dip_yield_excess_bounds_pa.lo > 0


def test_subnormal_scale_original_anchor_field_keeps_real_down_dip_rate():
    receipt = support(path_for(jump=-1e-300))
    assert receipt.admissible
    assert receipt.plastic_rate_bounds_m_s.hi < 0
    assert receipt.plastic_increment_bounds_m.lo == Fraction.from_float(-1e-300)


def test_material_interval_can_be_thinner_than_binary64_angles():
    left = Fraction(1, 2)
    right = left+Fraction(1, 10**400)
    receipt = support(path_for(), low=left, high=right)
    assert receipt.admissible
    assert receipt.material_angle_bounds == (left, right)


def test_actual_up_dip_path_is_rejected_even_with_compressive_preload():
    receipt = support(path_for(jump=2.))
    assert not receipt.admissible
    assert receipt.status == "no_down_dip_plastic_time_support"


def test_zero_rate_cannot_be_granted_from_pressure_alone():
    receipt = support(path_for(jump=0.), 0, 0)
    assert not receipt.admissible
    assert receipt.status == "no_strict_down_dip_plastic_rate"


def test_viscous_zero_time_limit_is_explicitly_unresolved_for_admission():
    receipt = support(path_for(viscosity_pa_s_m=1e12))
    assert not receipt.admissible
    assert receipt.status == "unresolved_viscous_contact_time_limit"
    assert receipt.plastic_rate_bounds_m_s is None


def test_viscous_family_uses_actual_fractional_duration_and_is_diagnostic_only():
    path = path_for(viscosity_pa_s_m=1e12)
    receipt = support(path, Fraction(1, 2), Fraction(1, 2))
    trace = path.initial_mechanics.fault_traces.traces[0]
    kt = Fraction.from_float(1./(1./trace.columns.shear_stiffness_pa_m
                                 +1./trace.law.tangential_stiffness_pa_m))
    expected = -kt/(kt+Fraction(1_000_000_000_000, 500_000))
    assert receipt.plastic_increment_bounds_m == I(expected)
    wrong_full_duration = -kt/(kt+1_000_000)
    assert expected != wrong_full_duration
    assert receipt.plastic_rate_bounds_m_s.hi < 0
    assert receipt.status == "viscous_fractional_backward_euler_diagnostic_only"
    assert not receipt.admissible


def test_negative_total_viscous_slip_does_not_hide_positive_instantaneous_rate():
    # An accepted viscous preparation leaves a positive trial excess E0.
    # Moving the next anchor back toward old plastic memory decreases E(t).
    # The BE-family total increment is still negative while its derivative
    # has become positive.  This is a rate reversal, not down-dip admission.
    plain = seed_for()
    trace = plain.fault_traces.traces[0]
    kt = 1./(1./trace.columns.shear_stiffness_pa_m
             +1./trace.law.tangential_stiffness_pa_m)
    seed = seed_for(viscosity_pa_s_m=kt)
    first = endpoint(seed, constant(-.1), constant(-2.), duration=1.)
    second = endpoint(seed, constant(-.1), constant(-1.25), duration=1., previous=first)
    path = build_native_contact_time_path(seed, second, first.anchor_fields, prior_steps=(first,))
    receipt = support(path, 1, 1)
    assert receipt.down_dip_yield_excess_bounds_pa.lo > 0
    assert receipt.plastic_increment_bounds_m.hi < 0
    assert receipt.plastic_rate_bounds_m_s.lo > 0
    assert receipt.status == "no_strict_down_dip_plastic_rate"
    assert not receipt.admissible


def test_zero_endpoint_pressure_keeps_the_whole_time_box_unresolved():
    receipt = support(path_for(gap=0.))
    assert receipt.pressure_bounds_pa.lo == 0
    assert receipt.status == "unresolved_strict_compressive_support"
    assert not receipt.admissible


def test_prior_endpoint_chain_authenticates_old_free_fields_and_plastic_memory():
    seed = seed_for()
    first = endpoint(seed, constant(-.1), constant(-1.))
    second = endpoint(seed, constant(-.1), constant(-2.), previous=first)
    with pytest.raises(ValueError, match="skip or reset"):
        build_native_contact_time_path(seed, second, first.anchor_fields)
    path = build_native_contact_time_path(seed, second, first.anchor_fields, prior_steps=(first,))
    receipt = support(path)
    assert receipt.admissible
    assert receipt.plastic_increment_bounds_m == I(-1, 0)
    assert path.step.previous_memory == first.memory


def test_same_candidate_cannot_be_repeated_as_its_own_accepted_predecessor():
    path = path_for()
    with pytest.raises(ValueError, match="skip or reset"):
        build_native_contact_time_path(path.initial_mechanics, path.step,
            path.previous_anchor_fields, prior_steps=(path.step,))


def test_initial_fields_are_bound_to_entire_preparation_not_only_four_gauss_nodes():
    path = path_for()
    altered = tuple((identity, gap, TrigPolynomial((0., 1e-200, 0., 0., 0.)))
                    for identity, gap, jump in path.previous_anchor_fields)
    with pytest.raises(ValueError, match="Previous contact-time anchors"):
        build_native_contact_time_path(path.initial_mechanics, path.step, altered)


def test_modified_endpoint_certificate_cannot_be_adopted_by_rehashing():
    path = path_for()
    forged_piece = replace(path.step.certificates[0].pieces[0], branch="locked")
    forged_certificate = replace(path.step.certificates[0], pieces=(forged_piece,))
    forged_step = replace(path.step, certificates=(forged_certificate, *path.step.certificates[1:]))
    with pytest.raises(ValueError, match="endpoint differs"):
        build_native_contact_time_path(path.initial_mechanics, forged_step, path.previous_anchor_fields)


def test_path_hash_and_model_are_verified_before_field_queries():
    path = path_for()
    with pytest.raises(ValueError, match="binding changed"):
        support(replace(path, path_sha256="0"*64))
    with pytest.raises(ValueError, match="path model"):
        build_native_contact_time_path(path.initial_mechanics, path.step,
            path.previous_anchor_fields, path_model="endpoint_permission")
    with pytest.raises(ValueError, match="replay-authenticated"):
        assess_native_contact_time_support({}, "unknown", 0, 1, 0, 1)


def test_native_axis_quarter_coordinate_mapping_preserves_saved_terminal_domain():
    path = path_for()
    trace = path.step.certificates[0]
    pi = interval_pi(100)
    whole = native_quarter_material_coordinate_bounds(path, trace.trace_id,
        I(0, pi.hi/2), precision=90)
    assert whole == I(0, Fraction.from_float(trace.extent_radians))
    endpoint_bounds = native_quarter_material_coordinate_bounds(path, trace.trace_id,
        pi/2, precision=90)
    assert endpoint_bounds.lo <= Fraction.from_float(trace.extent_radians) <= endpoint_bounds.hi
    # A tiny true angle does not collapse in the query conversion.
    tiny = native_quarter_material_coordinate_bounds(path, trace.trace_id,
        I(Fraction(1, 10**400)), precision=90)
    assert tiny.lo > 0 and tiny.hi < Fraction(1, 10**399)


def test_rounded_non_axis_trace_cannot_claim_exact_quarter_mapping():
    path = path_for()
    traces = path.initial_mechanics.fault_traces.traces
    changed = replace(traces[0], start=(0., -math.cos(.01), math.sin(.01)))
    # Preserve path authentication first; directly changing the seed is refused.
    forged = replace(path, initial_mechanics=replace(path.initial_mechanics,
        fault_traces=replace(path.initial_mechanics.fault_traces, traces=(changed, *traces[1:]))))
    with pytest.raises(ValueError):
        native_quarter_material_coordinate_bounds(forged, changed.trace_id, I(0, 1))


def test_interval_trig_bounds_include_stationary_pockets_not_only_endpoints():
    field = TrigPolynomial((0., 0., 0., -1., 0.))
    enclosure = _bounds(_field(field), Fraction(0), Fraction.from_float(math.pi/2.), 90)
    assert enclosure.lo <= -1 <= enclosure.hi
    assert enclosure.lo <= 1-1e-14 <= enclosure.hi
    for theta in (.1, .3, .7, 1.2):
        assert float(enclosure.lo) <= field(theta) <= float(enclosure.hi)


def test_exact_taylor_preserves_tiny_sine_and_encloses_nontrivial_stationary_endpoint():
    tiny = Fraction(1, 10**400)
    sine, cosine = _sin_cos_point(tiny, 90)
    assert 0 < sine.lo <= tiny <= sine.hi
    assert cosine.lo < 1 and cosine.hi == 1
    sine, cosine = _sin_cos_point(Fraction.from_float(math.pi/2.), 90)
    assert sine.lo < 1 and sine.hi <= 1
    assert cosine.lo > 0


def test_exact_receipt_serialization_records_all_bounds_and_diagnostic_scope():
    receipt = support(path_for(), Fraction(1, 10**400), Fraction(1, 3))
    raw = native_contact_time_support_to_dict(receipt)
    assert raw["time_fraction_bounds"][0] == [1, 10**400]
    assert raw["admissible"]
    assert raw["scope"].endswith("no_geometry_or_memory_adoption")
    assert json.loads(json.dumps(raw, allow_nan=False)) == raw
    with pytest.raises(ValueError, match="Expected native"):
        native_contact_time_support_to_dict({})


@pytest.mark.parametrize("first,last", [(-1, 0), (0, 2), (1, 0), (True, 1), (0, float("nan"))])
def test_invalid_time_windows_are_rejected(first, last):
    with pytest.raises(ValueError):
        support(path_for(), first, last)


@pytest.mark.parametrize("low,high", [(-1, 1), (0, 0), (1, 0), (0, 2), (True, 1)])
def test_invalid_material_query_does_not_fabricate_positive_support(low, high):
    with pytest.raises(ValueError):
        support(path_for(), low=low, high=high)


@pytest.mark.parametrize("precision", [0, 15, True, 80.])
def test_unsupported_precision_cannot_select_a_sign(precision):
    with pytest.raises(ValueError, match="precision"):
        support(path_for(), precision=precision)
