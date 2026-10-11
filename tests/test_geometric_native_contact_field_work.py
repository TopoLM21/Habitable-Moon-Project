"""Independent continuous work checks against the literal source constitutive law."""
from copy import deepcopy
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import json

import pytest

from tectonics.geometric_native_physical_history import native_physical_history_binding_to_dict
from tectonics.geometric_native_pose import FractionInterval as I
from tests.test_geometric_native_contact_field_history import (
    decimal, enclosed, polynomial, sine_cosine,
)
from tests.test_geometric_native_physical_history import get_physical_case


F = Fraction


@pytest.fixture(scope="module")
def continuum_work_case():
    case = get_physical_case()
    return dict(history=case["binding"],
        source=native_physical_history_binding_to_dict(case["binding"]))


def independent_work(history, trace_id, theta, initial, final):
    """Direct integration of the genuine frictionless source family.

    This keeps the separately stored bulk/contact stiffnesses and the saved
    binary64 effective stiffness.  Their mismatch must survive in the signed
    balance residual; it is not a new constitutive dissipation.
    """
    with localcontext() as context:
        # A 1e-400 time increment loses 400 digits in the independent
        # endpoint-square subtraction. Keep the surviving oracle accuracy
        # beyond the production's 256 decimal digits.
        context.prec = 1100
        index = next(index for index, trace in enumerate(
            history.initial_mechanics.fault_traces.traces) if trace.trace_id == trace_id)
        trace = history.initial_mechanics.fault_traces.traces[index]
        assert trace.law.friction_dry == trace.law.friction_wet == trace.law.cohesion_pa == 0
        _, gap0, _ = history.contact_time_path.previous_anchor_fields[index]
        _, gap1, _ = history.contact_time_path.step.anchor_fields[index]
        g0, g1 = polynomial(gap0, decimal(theta)), polynomial(gap1, decimal(theta))
        delta = g1-g0
        start, end = g0+decimal(initial)*delta, g0+decimal(final)*delta
        column, contact = (decimal(F.from_float(value)) for value in (
            trace.columns.normal_stiffness_pa_m, trace.law.normal_stiffness_pa_m))
        effective = decimal(F.from_float(1/(1/trace.columns.normal_stiffness_pa_m
            + 1/trace.law.normal_stiffness_pa_m)))
        normal_energy_coefficient = column*(effective/contact-1)**2+effective**2/contact
        change = (end*end-start*start)/2
        work = effective*change
        energy = normal_energy_coefficient*change
        return dict(normal_work=work, tangential_work=Decimal(0),
            external_work=work, dissipation=Decimal(0), energy_change=energy,
            signed_balance_residual=work-energy)


def independent_rates(history, trace_id, theta, time):
    with localcontext() as context:
        context.prec = 600
        index = next(index for index, trace in enumerate(
            history.initial_mechanics.fault_traces.traces) if trace.trace_id == trace_id)
        trace = history.initial_mechanics.fault_traces.traces[index]
        _, gap0, jump0 = history.contact_time_path.previous_anchor_fields[index]
        _, gap1, jump1 = history.contact_time_path.step.anchor_fields[index]
        g0, g1, j0, j1 = (polynomial(value, decimal(theta))
            for value in (gap0, gap1, jump0, jump1))
        derivative = g1-g0
        gap = g0+decimal(time)*derivative
        column, contact = (decimal(F.from_float(value)) for value in (
            trace.columns.normal_stiffness_pa_m, trace.law.normal_stiffness_pa_m))
        effective = decimal(F.from_float(1/(1/trace.columns.normal_stiffness_pa_m
            + 1/trace.law.normal_stiffness_pa_m)))
        bulk = column*(effective/contact-1)**2*gap*derivative
        spring = effective**2/contact*gap*derivative
        work = effective*gap*derivative
        return dict(source_work_rate_bounds_j_m2_per_fraction=work,
            friction_dissipation_rate_bounds_j_m2_per_fraction=Decimal(0),
            bulk_energy_rate_bounds_j_m2_per_fraction=bulk,
            contact_energy_rate_bounds_j_m2_per_fraction=spring,
            total_energy_rate_bounds_j_m2_per_fraction=bulk+spring,
            compliance_balance_residual_rate_bounds_j_m2_per_fraction=work-bulk-spring,
            free_gap_rate_bounds_m_per_fraction=derivative,
            free_jump_rate_bounds_m_per_fraction=j1-j0,
            plastic_slip_rate_bounds_m_per_fraction=j1-j0,
            cumulative_slip_rate_bounds_m_per_fraction=-(j1-j0))


def query_work(case, trace_id, angle, time, *, context=None):
    from tectonics.geometric_native_contact_field_work import query_native_contact_field_work
    return query_native_contact_field_work(case["history"], trace_id, angle, time,
        expected_source_history_dto=case["source"], source_expression_context=context)


def integral_work(case, trace_id, angle, initial, final, *, context=None):
    from tectonics.geometric_native_contact_field_work import integrate_native_contact_field_work
    return integrate_native_contact_field_work(case["history"], trace_id, angle, initial, final,
        expected_source_history_dto=case["source"], source_expression_context=context)


@pytest.mark.parametrize("angle,time", [(F(0), F(0)), (F(2, 9), F(3, 8)),
    (F(7, 5), F(1)), (F(1, 10**400), F(1, 10**400))])
def test_continuum_power_and_memory_rates_match_independent_literal_source(
        continuum_work_case, angle, time):
    case = continuum_work_case
    for trace in case["history"].initial_mechanics.fault_traces.traces:
        value = query_work(case, trace.trace_id, I(angle), I(time))
        expected = independent_rates(case["history"], trace.trace_id, angle, time)
        for name, amount in expected.items():
            enclosed(getattr(value, name), amount)
        duration = F.from_float(case["history"].literal_dt_s)
        for name, amount in expected.items():
            if name.endswith("_bounds_j_m2_per_fraction"):
                seconds_name = name.replace("_bounds_j_m2_per_fraction", "_bounds_w_m2")
                with localcontext() as context:
                    context.prec = 600
                    enclosed(getattr(value, seconds_name), amount/decimal(duration))
        assert value.friction_dissipation_rate_bounds_j_m2_per_fraction == I(F(0))
        assert value.source_field_query.source_previous_memory is case["history"].previous_memory
        assert value.source_field_query.cumulative_work_density_j_m2 is None


@pytest.mark.parametrize("initial,final", [(F(0), F(1)), (F(2, 7), F(5, 9)),
    (F(0), F(1, 10**400)), (F(3, 8), F(3, 8))])
def test_integrated_work_preserves_signed_compliance_defect_without_artificial_dissipation(
        continuum_work_case, initial, final):
    case = continuum_work_case
    angle = F(1, 3)
    names = dict(external_work="source_work_density_bounds_j_m2",
        dissipation="friction_dissipation_density_bounds_j_m2",
        energy_change="total_energy_increment_density_bounds_j_m2",
        signed_balance_residual="compliance_balance_residual_density_bounds_j_m2")
    for trace in case["history"].initial_mechanics.fault_traces.traces:
        value = integral_work(case, trace.trace_id, I(angle), initial, final)
        expected = independent_work(case["history"], trace.trace_id, angle, initial, final)
        for name, field in names.items():
            enclosed(getattr(value, field), expected[name])
        assert value.friction_dissipation_density_bounds_j_m2 == I(F(0))
        if initial == final:
            for field in names.values():
                assert getattr(value, field) == I(F(0))
        else:
            residual = value.compliance_balance_residual_density_bounds_j_m2
            assert residual.hi < 0 or residual.lo > 0
            if final == F(1, 10**400):
                assert float(value.source_work_density_bounds_j_m2.lo) == 0
                assert value.source_work_density_bounds_j_m2.hi < 0 or value.source_work_density_bounds_j_m2.lo > 0


def test_full_angle_clock_query_encloses_all_source_rates_without_fabricating_saved_work_density(
        continuum_work_case):
    case = continuum_work_case
    angle, time = I(F(1, 9), F(7, 5)), I(F(1, 10**400), F(1))
    for trace in case["history"].initial_mechanics.fault_traces.traces:
        value = query_work(case, trace.trace_id, angle, time)
        for theta in (angle.lo, F(4, 7), angle.hi):
            for fraction in (time.lo, F(3, 8), time.hi):
                for name, amount in independent_rates(case["history"], trace.trace_id,
                        theta, fraction).items():
                    enclosed(getattr(value, name), amount)
        field = value.source_field_query
        assert field.finite_point_area_m2 is field.finite_point_work_j is None
        assert field.cumulative_work_density_j_m2 is None
    assert native_physical_history_binding_to_dict(case["history"]) == case["source"]


def work_arguments(case):
    trace = case["history"].initial_mechanics.fault_traces.traces[0]
    return dict(history=case["history"], trace_id=trace.trace_id,
        computational_angle_bounds=I(F(1, 3), F(2, 3)),
        expected_source_history_dto=case["source"],
        source_expression_context={"kind": "external_implicit_first_root", "first_root": "original-root:work"})


@pytest.mark.parametrize("kind", ["query", "integral"])
def test_work_restart_requires_complete_source_and_external_implicit_expression(
        continuum_work_case, kind):
    from tectonics.geometric_native_contact_field_work import (
        native_contact_field_work_from_dict, native_contact_field_work_to_dict,
        native_contact_field_work_integral_from_dict, native_contact_field_work_integral_to_dict,
    )
    case, arguments = continuum_work_case, work_arguments(continuum_work_case)
    if kind == "query":
        arguments["time_fraction_bounds"] = I(F(2, 7), F(3, 7))
        value = query_work(case, arguments["trace_id"], arguments["computational_angle_bounds"],
            arguments["time_fraction_bounds"], context=arguments["source_expression_context"])
        dump, restore = native_contact_field_work_to_dict, native_contact_field_work_from_dict
    else:
        arguments.update(initial_time_fraction=F(2, 7), final_time_fraction=F(3, 7))
        value = integral_work(case, arguments["trace_id"], arguments["computational_angle_bounds"],
            arguments["initial_time_fraction"], arguments["final_time_fraction"],
            context=arguments["source_expression_context"])
        dump, restore = native_contact_field_work_integral_to_dict, native_contact_field_work_integral_from_dict
    kwargs = dict(history=case["history"], expected_source_history_dto=case["source"])
    raw = dump(value, **kwargs)
    assert dump(restore(deepcopy(raw), **arguments), **kwargs) == raw
    forged_context = dict(arguments, source_expression_context={"kind": "replacement-root"})
    with pytest.raises(ValueError):
        restore(deepcopy(raw), **forged_context)


@pytest.mark.parametrize("mutation", ["work", "dissipation", "balance", "memory", "source_work", "source_area"])
def test_resealed_work_payload_cannot_change_constitutive_result_or_saved_history(
        continuum_work_case, mutation):
    from tectonics.geometric_native_contact_field_work import (
        native_contact_field_work_integral_from_dict, native_contact_field_work_integral_to_dict,
    )
    case, arguments = continuum_work_case, work_arguments(continuum_work_case)
    arguments.update(initial_time_fraction=F(2, 7), final_time_fraction=F(3, 7))
    value = integral_work(case, arguments["trace_id"], arguments["computational_angle_bounds"],
        arguments["initial_time_fraction"], arguments["final_time_fraction"],
        context=arguments["source_expression_context"])
    raw = deepcopy(native_contact_field_work_integral_to_dict(value,
        history=case["history"], expected_source_history_dto=case["source"]))
    if mutation in ("work", "dissipation", "balance"):
        key = {"work": "source_work_density_bounds_j_m2",
            "dissipation": "friction_dissipation_density_bounds_j_m2",
            "balance": "compliance_balance_residual_density_bounds_j_m2"}[mutation]
        raw[key] = {"lo": [1, 1], "hi": [1, 1]}
    else:
        # The portable whole source is required independently of its own seal.
        source = raw["source_field_query"]["source_history"]
        if mutation == "memory":
            source["actual_candidate"]["states"][0][1]["plastic_slip_m"] += 1
        else:
            source["actual_candidate"]["states"][0][1][
                "cumulative_external_work_j" if mutation == "source_work" else "area_m2"] += 1
    if mutation in ("memory", "source_work", "source_area"):
        field = raw["source_field_query"]
        field["query_sha256"] = hashlib.sha256(json.dumps(
            {name: item for name, item in field.items() if name != "query_sha256"},
            sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False).encode()).hexdigest()
    seals = [key for key in raw if key.endswith("sha256")]
    assert seals
    # Rehashing a claimed result cannot replace replay from the external typed source.
    for key in seals:
        raw[key] = hashlib.sha256(json.dumps({name: item for name, item in raw.items() if name != key},
            sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    with pytest.raises(ValueError):
        native_contact_field_work_integral_from_dict(raw, **arguments)


@pytest.mark.parametrize("bad", ["reversed_time", "float_time", "boolean_time", "outside_angle"])
def test_integral_refuses_untyped_or_reversed_material_clock(continuum_work_case, bad):
    case = continuum_work_case
    trace = case["history"].initial_mechanics.fault_traces.traces[0]
    initial, final, angle = F(0), F(1), I(F(1, 3))
    if bad == "reversed_time":
        initial, final = F(2, 3), F(1, 3)
    elif bad == "float_time":
        final = .5
    elif bad == "boolean_time":
        initial = False
    else:
        angle = I(F(-1, 3))
    with pytest.raises(ValueError):
        integral_work(case, trace.trace_id, angle, initial, final)


def independent_square_integral(coefficients, low, high):
    """Integrate pairwise real harmonic products without production Fourier code."""
    def cosine_integral(frequency):
        if frequency == 0:
            return high-low
        frequency = abs(frequency)
        return (sine_cosine(high*frequency)[0]-sine_cosine(low*frequency)[0])/frequency

    def sine_integral(frequency):
        if frequency == 0:
            return Decimal(0)
        return (sine_cosine(low*frequency)[1]-sine_cosine(high*frequency)[1])/frequency

    basis = (("constant", 0), ("cosine", 1), ("sine", 1),
             ("cosine", 2), ("sine", 2))
    result = Decimal(0)
    for first, (kind, frequency) in enumerate(basis):
        for second, (other_kind, other_frequency) in enumerate(basis):
            if kind == "constant":
                primitive = (cosine_integral(other_frequency) if other_kind != "sine"
                    else sine_integral(other_frequency))
            elif other_kind == "constant":
                primitive = (cosine_integral(frequency) if kind != "sine"
                    else sine_integral(frequency))
            elif kind == other_kind:
                primitive = (cosine_integral(frequency-other_frequency)
                    + (1 if kind == "cosine" else -1)*cosine_integral(frequency+other_frequency))/2
            else:
                sine_frequency = frequency if kind == "sine" else other_frequency
                cosine_frequency = frequency if kind == "cosine" else other_frequency
                primitive = (sine_integral(sine_frequency+cosine_frequency)
                    + sine_integral(sine_frequency-cosine_frequency))/2
            result += coefficients[first]*coefficients[second]*primitive
    return result


def independent_interface_integral(history, trace_id, low, high, initial, final):
    with localcontext() as context:
        context.prec = 1100
        index = next(index for index, trace in enumerate(history.initial_mechanics.fault_traces.traces)
            if trace.trace_id == trace_id)
        trace = history.initial_mechanics.fault_traces.traces[index]
        _, old, _ = history.contact_time_path.previous_anchor_fields[index]
        _, new, _ = history.contact_time_path.step.anchor_fields[index]
        old_coefficients, new_coefficients = (tuple(decimal(F.from_float(value))
            for value in field.coefficients) for field in (old, new))
        at_time = lambda time: tuple(a+decimal(time)*(b-a)
            for a, b in zip(old_coefficients, new_coefficients))
        first_square, final_square = (independent_square_integral(at_time(time),
            decimal(low), decimal(high)) for time in (initial, final))
        column, contact = (decimal(F.from_float(value)) for value in (
            trace.columns.normal_stiffness_pa_m, trace.law.normal_stiffness_pa_m))
        effective = decimal(F.from_float(1/(1/trace.columns.normal_stiffness_pa_m
            + 1/trace.law.normal_stiffness_pa_m)))
        bulk, spring = column*(effective/contact-1)**2, effective**2/contact
        node = next(node for node in history.quadrature_histories if node.trace_id == trace_id)
        density = decimal(F.from_float(history.origin_surface.radius_km)*1000
            * F.from_float(trace.contact_height_m)/F.from_float(node.original_frame[0]))
        change = density*(final_square-first_square)/2
        work, energy = effective*change, (bulk+spring)*change
        return dict(source_work_bounds_j=work, friction_dissipation_bounds_j=Decimal(0),
            bulk_energy_increment_bounds_j=bulk*change,
            contact_energy_increment_bounds_j=spring*change,
            total_energy_increment_bounds_j=energy,
            compliance_balance_residual_bounds_j=work-energy,
            initial_bulk_energy_bounds_j=density*bulk*first_square/2,
            initial_contact_energy_bounds_j=density*spring*first_square/2,
            initial_total_energy_bounds_j=density*(bulk+spring)*first_square/2,
            final_bulk_energy_bounds_j=density*bulk*final_square/2,
            final_contact_energy_bounds_j=density*spring*final_square/2,
            final_total_energy_bounds_j=density*(bulk+spring)*final_square/2)


@pytest.mark.parametrize("kind,initial,final", [("whole", F(0), F(1)),
    ("partial", F(2, 7), F(5, 9)), ("whole", F(0), F(1, 10**400)),
    ("zero", F(0), F(1))])
def test_angular_work_integral_uses_independent_real_harmonic_antiderivatives_and_saved_measure(
        continuum_work_case, kind, initial, final):
    from tectonics.geometric_native_contact_field_work import (
        integrate_native_contact_field_work_over_original_interface,
    )
    case, history = continuum_work_case, continuum_work_case["history"]
    for trace in history.initial_mechanics.fault_traces.traces:
        extent = F.from_float(next(row for row in history.previous_memory.traces
            if row.trace_id == trace.trace_id).extent_radians)
        low, high = (F(0), extent) if kind == "whole" else (F(1, 3), F(2, 3))
        if kind == "zero":
            high = low
        value = integrate_native_contact_field_work_over_original_interface(history, trace.trace_id,
            low, high, initial, final, expected_source_history_dto=case["source"])
        for name, amount in independent_interface_integral(history, trace.trace_id,
                low, high, initial, final).items():
            enclosed(getattr(value, name), amount)
        atoms = [node for node in history.quadrature_histories if node.trace_id == trace.trace_id]
        density = F.from_float(history.origin_surface.radius_km)*1000*F.from_float(trace.contact_height_m)
        density /= F.from_float(atoms[0].original_frame[0])
        assert value.original_interface_area_density_m2_per_computational_radian == density
        assert value.continuum_interface_area_m2 == density*(high-low)
        assert value.original_nodal_interface_area_m2 == sum((F.from_float(node.original_area_m2)
            for node in atoms), F(0))
        assert value.continuum_minus_original_nodal_interface_area_m2 == (
            value.continuum_interface_area_m2-value.original_nodal_interface_area_m2)
        assert value.friction_dissipation_bounds_j == I(F(0))
        assert not value.original_nodal_ledgers_replaced
        assert not value.candidate_mechanics_adopted


def test_implicit_clock_primitive_encloses_all_possible_event_values_without_selecting_midpoint(
        continuum_work_case):
    from tectonics.geometric_native_contact_field_work import query_native_contact_field_work_primitive
    case = continuum_work_case
    angle, final, initial = I(F(1, 9), F(7, 5)), I(F(2, 7), F(3, 7)), F(0)
    context = {"kind": "frozen_original_first_event", "event_id": "external:true-first-root"}
    for trace in case["history"].initial_mechanics.fault_traces.traces:
        value = query_native_contact_field_work_primitive(case["history"], trace.trace_id,
            angle, initial, final, expected_source_history_dto=case["source"],
            source_expression_context=context)
        assert value.final_time_fraction_bounds == final
        assert value.source_field_query.source_expression_context == context
        assert value.source_field_query.finite_point_area_m2 is None
        for theta in (angle.lo, F(4, 7), angle.hi):
            for end in (final.lo, final.hi):
                expected = independent_work(case["history"], trace.trace_id, theta, initial, end)
                enclosed(value.source_work_density_bounds_j_m2, expected["external_work"])
                enclosed(value.total_energy_increment_density_bounds_j_m2, expected["energy_change"])
                enclosed(value.compliance_balance_residual_density_bounds_j_m2,
                    expected["signed_balance_residual"])


@pytest.mark.parametrize("kind", ["primitive", "interface"])
def test_primitive_and_interface_restart_keep_original_measures_and_frozen_clock(
        continuum_work_case, kind):
    from tectonics.geometric_native_contact_field_work import (
        query_native_contact_field_work_primitive,
        integrate_native_contact_field_work_over_original_interface,
        native_contact_field_work_primitive_from_dict, native_contact_field_work_primitive_to_dict,
        native_contact_field_work_interface_integral_from_dict,
        native_contact_field_work_interface_integral_to_dict,
    )
    case, arguments = continuum_work_case, work_arguments(continuum_work_case)
    if kind == "primitive":
        arguments.update(initial_time_fraction=F(0), final_time_fraction_bounds=I(F(2, 7), F(3, 7)))
        value = query_native_contact_field_work_primitive(**arguments)
        dump, restore = native_contact_field_work_primitive_to_dict, native_contact_field_work_primitive_from_dict
    else:
        arguments.pop("computational_angle_bounds")
        arguments.update(initial_computational_angle=F(1, 3), final_computational_angle=F(2, 3),
            initial_time_fraction=F(0), final_time_fraction=F(3, 7))
        value = integrate_native_contact_field_work_over_original_interface(**arguments)
        dump = native_contact_field_work_interface_integral_to_dict
        restore = native_contact_field_work_interface_integral_from_dict
    kwargs = dict(history=case["history"], expected_source_history_dto=case["source"])
    raw = dump(value, **kwargs)
    assert dump(restore(deepcopy(raw), **arguments), **kwargs) == raw
    forged = dict(arguments, source_expression_context={"kind": "a different event"})
    with pytest.raises(ValueError):
        restore(deepcopy(raw), **forged)
