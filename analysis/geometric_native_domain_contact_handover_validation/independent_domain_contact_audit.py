"""Standard-library audit of whole-domain first-entry contact transfer.

Exact source Fourier integrals below are independent analytical observations.
An observation never chooses a production root inside its exact enclosure or
grants force admission.  Literal quadrature atoms remain a separate ledger
from the dimensional continuum interface measure and its work functional.
"""
from __future__ import annotations

import argparse
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import importlib.util
from itertools import product
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
PRIOR = ROOT / "analysis/geometric_native_first_entry_handover_validation"
PRECISION = 360


def load_helper(name, path):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


old = load_helper("independent_original_first_entry_contact_audit",
                  PRIOR / "independent_first_entry_handover_audit.py")
require, keys, encoded, digest = old.require, old.keys, old.encoded, old.digest
file_sha, read, same, seal = old.file_sha, old.read, old.same, old.seal
fraction, rational, interval, exact_float = old.fraction, old.rational, old.interval, old.exact_float
dec, contains, check_vector = old.dec, old.contains, old.check_vector
dot, cross, sin_cos, rotate = old.dot, old.cross, old.sin_cos, old.rotate
atan, atan2_positive, half_pi = old.atan, old.atan2_positive, old.half_pi
iadd, isub, imul, iscale, ineg = old.iadd, old.isub, old.imul, old.iscale, old.ineg
encloses, relative_file = old.encloses, old.relative_file


def fourier_sin_cos(value):
    """Independent Decimal Taylor observation through harmonic four."""
    require(abs(value) <= 8, "Independent source Fourier harmonic budget exceeded")
    threshold = Decimal(10)**-(PRECISION-15)
    squared = value*value

    def series(sine):
        term = value if sine else Decimal(1)
        result, index = term, 0
        while True:
            factor = 2*index+(2 if sine else 1)
            term *= -squared/Decimal(factor*(factor+1))
            result += term
            if abs(term) <= threshold:
                return result
            index += 1
    return series(True), series(False)


def fourier(raw):
    """Lift the five literal source coefficients before independent algebra."""
    require(type(raw) is dict and set(raw) == {"coefficients"}
            and type(raw["coefficients"]) is list and len(raw["coefficients"]) == 5,
            "Original degree-two source Fourier field changed")
    return {key: dec(exact_float(value)) for key, value in zip(
        (("c", 0), ("c", 1), ("s", 1), ("c", 2), ("s", 2)), raw["coefficients"])}


def poly_add(*values):
    result = {}
    for value in values:
        for key, coefficient in value.items():
            result[key] = result.get(key, Decimal(0)) + coefficient
    return {key: value for key, value in result.items() if value}


def poly_scale(value, scale):
    return {key: coefficient * scale for key, coefficient in value.items() if coefficient * scale}


def poly_constant(value):
    return {("c", 0): value} if value else {}


def poly_mul(first, second):
    """Direct real trigonometric products, independently of production code."""
    result = {}

    def add(kind, harmonic, coefficient):
        if kind == "s" and not harmonic:
            return
        if harmonic < 0:
            harmonic = -harmonic
            if kind == "s":
                coefficient = -coefficient
        key = kind, harmonic
        result[key] = result.get(key, Decimal(0)) + coefficient

    for (kind_a, a), first_coefficient in first.items():
        for (kind_b, b), second_coefficient in second.items():
            coefficient = first_coefficient * second_coefficient / 2
            if kind_a == kind_b == "c":
                add("c", a + b, coefficient)
                add("c", a - b, coefficient)
            elif kind_a == kind_b == "s":
                add("c", a - b, coefficient)
                add("c", a + b, -coefficient)
            elif kind_a == "c":
                add("s", b + a, coefficient)
                add("s", b - a, coefficient)
            else:
                add("s", a + b, coefficient)
                add("s", a - b, coefficient)
    return {key: value for key, value in result.items() if value}


def poly_value(value, theta):
    result = Decimal(0)
    for (kind, harmonic), coefficient in value.items():
        if not harmonic:
            result += coefficient
        else:
            sine, cosine = fourier_sin_cos(theta * harmonic)
            result += coefficient * (sine if kind == "s" else cosine)
    return result


def poly_integral(value, low, high):
    """Analytical 360-Decimal observation, without nodes or area weights."""
    require(low <= high, "Independent continuum integration has reversed bounds")
    result = Decimal(0)
    for (kind, harmonic), coefficient in value.items():
        if not harmonic:
            result += coefficient * (high - low)
        else:
            low_sine, low_cosine = fourier_sin_cos(low * harmonic)
            high_sine, high_cosine = fourier_sin_cos(high * harmonic)
            primitive_difference = ((low_cosine - high_cosine) if kind == "s"
                                    else (high_sine - low_sine)) / harmonic
            result += coefficient * primitive_difference
    return result


def dense_source_polynomials(history, trace_id, original_piece, instant):
    """Reconstruct the whole admitted down-dip family from literal sources."""
    path = history["contact_time_path"]
    trace = next(trace for trace in path["initial_mechanics"]["fault_traces"]["traces"]
                 if trace["trace_id"] == trace_id)
    support = next(row for row in history["whole_time_support"]["certificates"] if row["trace_id"] == trace_id)
    require(support["status"] == "inviscid_contact_time_constitutive_admissible"
            and trace["law"]["viscosity_pa_s_m"] == 0,
            "Independent continuum work requires the original whole admitted inviscid branch")
    initial = next(row for row in path["previous_anchor_fields"] if row[0] == trace_id)
    final = next(row for row in path["step"]["anchor_fields"] if row[0] == trace_id)
    gap0, jump0 = fourier(initial[1]), fourier(initial[2])
    dgap = poly_add(fourier(final[1]), poly_scale(gap0, Decimal(-1)))
    djump = poly_add(fourier(final[2]), poly_scale(jump0, Decimal(-1)))
    gap, jump = poly_add(gap0, poly_scale(dgap, instant)), poly_add(jump0, poly_scale(djump, instant))
    old_p, old_c = fourier(original_piece["plastic_slip"]), fourier(original_piece["cumulative_slip"])
    kncol, ktcol, kncontact, ktcontact, kn, kt, mu, cohesion = old.source_column_coefficients(
        trace["columns"], trace["law"], trace["water"])
    pressure = poly_scale(gap, -kn)
    strength = poly_add(poly_constant(cohesion), poly_scale(pressure, mu))
    plastic = poly_add(jump, poly_scale(strength, 1 / kt))
    increment = poly_add(plastic, poly_scale(old_p, Decimal(-1)))
    cumulative = poly_add(old_c, poly_scale(increment, Decimal(-1)))
    shear = poly_scale(strength, Decimal(-1))
    contact_gap = poly_scale(pressure, -1 / kncontact)
    contact_elastic = poly_scale(shear, 1 / ktcontact)
    bulk_normal = poly_scale(gap, kn / kncontact - 1)
    bulk_shear = poly_scale(shear, 1 / ktcontact - 1 / kt)
    bulk_energy = poly_scale(poly_add(poly_scale(poly_mul(bulk_normal, bulk_normal), kncol),
                                     poly_scale(poly_mul(bulk_shear, bulk_shear), ktcol)), Decimal("0.5"))
    contact_energy = poly_scale(poly_add(poly_scale(poly_mul(contact_gap, contact_gap), kncontact),
                                        poly_scale(poly_mul(contact_elastic, contact_elastic), ktcontact)), Decimal("0.5"))
    plastic_rate = poly_add(djump, poly_scale(dgap, -mu * kn / kt))
    return dict(gap=gap, jump=jump, previous_plastic=old_p, previous_cumulative=old_c,
        plastic_increment=increment, plastic=plastic, cumulative=cumulative, pressure=pressure, shear=shear,
        contact_gap=contact_gap, contact_elastic_shear=contact_elastic,
        contact_jump=poly_add(plastic, contact_elastic), bulk_normal_strain=bulk_normal,
        bulk_shear_strain=bulk_shear,
        bulk_energy=bulk_energy, contact_energy=contact_energy,
        total_energy=poly_add(bulk_energy, contact_energy), strength=strength,
        plastic_rate=plastic_rate, gap_rate=dgap, jump_rate=djump)


def work_polynomials(history, trace_id, original_piece, instant):
    """Independent reaction one-form and the retained literal first law."""
    source = dense_source_polynomials(history, trace_id, original_piece, instant)
    trace = next(trace for trace in history["contact_time_path"]["initial_mechanics"]["fault_traces"]["traces"]
                 if trace["trace_id"] == trace_id)
    kncol, ktcol, kncontact, ktcontact, kn, kt, mu, cohesion = old.source_column_coefficients(
        trace["columns"], trace["law"], trace["water"])
    an_bulk, an_contact = kncol * (kn / kncontact - 1)**2, kn**2 / kncontact
    at_bulk, at_contact = ktcol * (1 / ktcontact - 1 / kt)**2, 1 / ktcontact
    normal_product = poly_mul(source["gap"], source["gap_rate"])
    shear_rate = poly_scale(source["gap_rate"], mu * kn)
    shear_product = poly_mul(source["shear"], shear_rate)
    bulk = poly_add(poly_scale(normal_product, an_bulk), poly_scale(shear_product, at_bulk))
    contact = poly_add(poly_scale(normal_product, an_contact), poly_scale(shear_product, at_contact))
    return dict(source_work=poly_add(poly_scale(normal_product, kn),
                                    poly_mul(source["shear"], source["jump_rate"])),
        friction_dissipation=poly_mul(source["shear"], source["plastic_rate"]),
        bulk_energy=bulk, contact_energy=contact, total_energy=poly_add(bulk, contact),
        compliance_balance_residual=poly_add(poly_scale(normal_product, kn-an_bulk-an_contact),
                                             poly_scale(shear_product, 1/kt-at_bulk-at_contact)),
        free_gap=source["gap_rate"], free_jump=source["jump_rate"],
        plastic_slip=source["plastic_rate"], cumulative_slip=poly_scale(source["plastic_rate"], Decimal(-1)))


WORK_FAMILIES = ("source_work", "friction_dissipation", "bulk_energy", "contact_energy",
                 "total_energy", "compliance_balance_residual")


def check_work_source(raw, history):
    seal(raw, "query_sha256", "Retained continuous-work source")
    same(raw["source_history"], history, "Full continuous-work source history")
    require(raw["format"] == "native-original-continuous-contact-field-query-1"
            and raw["model"] == "original_inviscid_dense_return_map_entire_piecewise_history"
            and raw["piece_selection_model"] == "all_closed_original_piece_intersections_no_node_or_midpoint_selection"
            and raw["energy_measure"] == "joules_per_square_metre_of_original_interface_area"
            and raw["constitutive_status"] == "inviscid_contact_time_constitutive_admissible",
            "Continuous-work source used a different field, angular measure or branch")
    for name in ("finite_point_area_m2", "finite_point_work_j", "cumulative_work_density_j_m2"):
        require(raw[name] is None, "Continuous-work source invents finite point work/measure")
    angle = interval(raw["computational_angle_bounds"])
    path = history["contact_time_path"]
    memory = next(row for row in path["step"]["previous_memory"]["traces"] if row["trace_id"] == raw["trace_id"])
    extent = exact_float(memory["extent_radians"])
    require(0 <= angle[0] <= angle[1] <= extent, "Continuous work left the computational source chart")
    expected = []
    for index, piece in enumerate(memory["pieces"]):
        low, high = max(angle[0], exact_float(piece["low_radians"])), min(angle[1], exact_float(piece["high_radians"]))
        if low <= high:
            expected.append(dict(piece_index=index, computational_angle_bounds=dict(lo=rational(low), hi=rational(high)),
                                 original_piece=piece))
    same(raw["previous_memory_intersections"], expected, "All original previous pieces retained for continuous work")
    require(expected and len(raw["pieces"]) == len(expected), "Continuous work omitted an original source piece")
    candidate = next(row for row in path["step"]["memory"]["traces"] if row["trace_id"] == raw["trace_id"])
    candidate_intersections = []
    for index, piece in enumerate(candidate["pieces"]):
        low, high = max(angle[0],exact_float(piece["low_radians"])), min(angle[1],exact_float(piece["high_radians"]))
        if low <= high:
            candidate_intersections.append(dict(piece_index=index,computational_angle_bounds=dict(lo=rational(low),hi=rational(high)),
                                                original_piece=piece))
    same(raw["unaccepted_candidate_memory_intersections"],candidate_intersections,
         "Complete unchanged endpoint memory source pieces")
    field_map = {
        "free_gap_bounds_m":"gap", "free_jump_bounds_m":"jump",
        "previous_plastic_slip_bounds_m":"previous_plastic", "previous_cumulative_slip_bounds_m":"previous_cumulative",
        "plastic_increment_bounds_m":"plastic_increment", "plastic_slip_bounds_m":"plastic",
        "cumulative_slip_bounds_m":"cumulative", "normal_pressure_bounds_pa":"pressure", "shear_traction_bounds_pa":"shear",
        "contact_gap_bounds_m":"contact_gap", "contact_jump_bounds_m":"contact_jump",
        "bulk_normal_strain_bounds_m":"bulk_normal_strain", "bulk_shear_strain_bounds_m":"bulk_shear_strain",
        "contact_elastic_shear_strain_bounds_m":"contact_elastic_shear", "bulk_energy_density_bounds_j_m2":"bulk_energy",
        "contact_energy_density_bounds_j_m2":"contact_energy", "total_energy_density_bounds_j_m2":"total_energy"}
    clock = interval(raw["time_fraction_bounds"])
    for row, source_row in zip(raw["pieces"],expected):
        require(row["previous_piece_index"] == source_row["piece_index"], "Continuous field source reordered memory pieces")
        same(row["computational_angle_bounds"],source_row["computational_angle_bounds"],"Unchanged source piece domain")
        for theta, instant in product(set(interval(row["computational_angle_bounds"])),set(clock)):
            polynomials = dense_source_polynomials(history,raw["trace_id"],source_row["original_piece"],dec(instant))
            for field, polynomial in field_map.items():
                value = poly_value(polynomials[polynomial],dec(theta))
                contains(row[field],value,"Independent unchanged continuum source configuration")
                contains(raw[field],value,"Independent unchanged continuum source union")
    nodes = [row for row in history["quadrature_histories"] if row["trace_id"] == raw["trace_id"]]
    require(nodes and all(row["previous_state"]["elapsed_time_s"] == nodes[0]["previous_state"]["elapsed_time_s"] for row in nodes),
            "Original local continuum source has inconsistent clocks")
    for instant in set(clock):
        for field, value in (
            ("elapsed_time_s_bounds",exact_float(nodes[0]["previous_state"]["elapsed_time_s"])+instant*exact_float(history["literal_dt_s"])),
            ("native_elapsed_time_s_bounds",exact_float(history["accepted_before"]["snapshot"]["elapsed_time_s"])+instant*exact_float(history["literal_dt_s"])),
            ("physical_time_myr_bounds",exact_float(history["origin_surface"]["time_myr"])+instant*exact_float(history["literal_dt_myr"]))):
            contains(raw[field],dec(value),"Complete retained source continuum clock")
    return expected


def check_field_work(raw, history, *, integrated):
    seal(raw, "work_sha256", "Continuous contact work functional")
    expected_format = "native-original-continuous-contact-field-work-" + ("integral-1" if integrated else "query-1")
    require(raw["format"] == expected_format
            and raw["model"] == "authenticated_inviscid_down_dip_reaction_one_form_and_literal_compliance_residual"
            and raw["work_measure"] == "joules_per_square_metre_of_original_interface_area"
            and raw["rate_measure"] == "joules_per_square_metre_per_original_time_fraction"
            and raw["first_law"] == "source_work = bulk_energy_increment + contact_energy_increment + plastic_dissipation + signed_literal_compliance_residual",
            "Continuous contact work measure or original one-form changed")
    require(raw["binary64_compliance_coefficients_preserved"] is True
            and raw["finite_point_area_m2"] is None and raw["original_nodal_work_density_j_m2"] is None,
            "Continuous work changed original compliance or reinterpreted atomic work")
    for name in ("angular_integration_performed", "endpoint_backward_euler_ledgers_replaced",
                 "signed_compliance_residual_suppressed", "native_candidate_mechanics_adopted",
                 "geometry_expression_authenticated_by_this_component", "material_transfer_work_adopted",
                 "material_transfer_dissipation_adopted"):
        require(raw[name] is False, "Continuous work incorrectly grants " + name)
    source = raw["source_field_query"]
    intersections = check_work_source(source, history)
    window = interval(source["time_fraction_bounds"])
    duration = dec(exact_float(history["literal_dt_s"]))
    if integrated:
        first, last = fraction(raw["initial_time_fraction"]), fraction(raw["final_time_fraction"])
        require(0 <= first <= last <= 1 and window == (first, last), "Continuous work picked a different original time interval")
        clocks, span = (dec((first+last)/2),), dec(last-first)
    else:
        clocks, span = tuple(dec(value) for value in set(window)), Decimal(1)
    require(len(raw["pieces"]) == len(intersections), "Continuous work omitted an original angular piece")
    checked = 0
    for row, source_row in zip(raw["pieces"], intersections):
        require(row["previous_piece_index"] == source_row["piece_index"], "Continuous work reordered original memory")
        same(row["computational_angle_bounds"], source_row["computational_angle_bounds"], "Continuous work angular partition")
        for theta, clock in product(set(interval(row["computational_angle_bounds"])), clocks):
            theta = dec(theta)
            polynomials = work_polynomials(history, source["trace_id"], source_row["original_piece"], clock)
            values = {name: poly_value(polynomials[name], theta)*span for name in WORK_FAMILIES}
            discrepancy = values["source_work"]-values["friction_dissipation"]-values["total_energy"]-values["compliance_balance_residual"]
            tolerance = Decimal(10)**-(PRECISION-40)*max(Decimal(1), abs(values["source_work"]))
            require(abs(discrepancy) <= tolerance and values["friction_dissipation"] >= -tolerance,
                    "Independent continuum reaction one-form violates its retained first law")
            for name, value in values.items():
                suffix = ("_energy_increment_density_bounds_j_m2" if integrated and name in ("bulk_energy", "contact_energy", "total_energy")
                          else "_density_bounds_j_m2" if integrated else "_rate_bounds_j_m2_per_fraction")
                if integrated and name in ("bulk_energy", "contact_energy", "total_energy"):
                    field = name.removesuffix("_energy") + suffix
                else:
                    field = name + suffix
                contains(row[field], value, "Independent continuous piece work " + field)
                contains(raw[field], value, "Independent continuous union work " + field)
                if not integrated:
                    physical = name + "_rate_bounds_w_m2"
                    contains(row[physical], value/duration, "Independent physical contact power")
                    contains(raw[physical], value/duration, "Independent physical contact power union")
            if not integrated:
                for name in ("free_gap", "free_jump", "plastic_slip", "cumulative_slip"):
                    field = name + "_rate_bounds_m_per_fraction"
                    value = poly_value(polynomials[name], theta)
                    contains(row[field], value, "Independent free/plastic source rate")
                    contains(raw[field], value, "Independent free/plastic source rate union")
            checked += 1
    return checked


def check_interface_work(raw, history):
    """Independently integrate all retained source pieces in θ and time."""
    seal(raw, "work_sha256", "Original continuum interface work integral")
    require(raw["format"] == "native-original-continuous-contact-field-work-interface-integral-1"
            and raw["model"] == "authenticated_inviscid_down_dip_reaction_one_form_and_literal_compliance_residual"
            and raw["angular_integration_model"] == "exact_degree_four_harmonic_antiderivative"
            and raw["measure_model"] == "original_saved_frame_radius_height_over_sine_times_computational_angle",
            "Whole interface work changed its source harmonic integral or dimensional measure")
    for name in ("continuum_measure_is_original_nodal_gauss_measure", "continuum_work_is_original_endpoint_backward_euler_work",
                 "original_nodal_ledgers_replaced", "signed_compliance_residual_suppressed", "native_candidate_mechanics_adopted",
                 "swept_archive_measure_integrated", "incoming_material_contact_area_adopted",
                 "material_transfer_work_adopted", "material_transfer_dissipation_adopted"):
        require(raw[name] is False, "Whole original interface work incorrectly grants " + name)
    time_integral = raw["source_time_integral"]
    checks = check_field_work(time_integral, history, integrated=True)
    source = time_integral["source_field_query"]
    trace_id = source["trace_id"]
    angle = interval(source["computational_angle_bounds"])
    first, last = fraction(time_integral["initial_time_fraction"]), fraction(time_integral["final_time_fraction"])
    trace = next(trace for trace in history["contact_time_path"]["initial_mechanics"]["fault_traces"]["traces"]
                 if trace["trace_id"] == trace_id)
    nodes = [row for row in history["quadrature_histories"] if row["trace_id"] == trace_id]
    require(nodes and all(row["original_frame"] == nodes[0]["original_frame"] for row in nodes),
            "Whole interface work lacks a common original saved dip")
    radius = exact_float(history["contact_time_path"]["initial_mechanics"]["fault_traces"]["radius_km"])*1000
    require(radius == exact_float(history["origin_surface"]["radius_km"])*1000,
            "Source contact and material radii differ")
    density = radius*exact_float(trace["contact_height_m"])/exact_float(nodes[0]["original_frame"][0])
    nodal_area = sum((exact_float(row["original_area_m2"]) for row in nodes), Fraction(0))
    continuum_area = density*(angle[1]-angle[0])
    for name, expected in (("original_interface_area_density_m2_per_computational_radian", density),
                           ("continuum_interface_area_m2", continuum_area),
                           ("original_nodal_interface_area_m2", nodal_area),
                           ("continuum_minus_original_nodal_interface_area_m2", continuum_area-nodal_area)):
        require(fraction(raw[name]) == expected, "Original continuum or atomic interface measure changed: " + name)
    values = {name: Decimal(0) for name in WORK_FAMILIES}
    energies = {prefix+"_"+name: Decimal(0) for prefix in ("initial", "final")
                for name in ("bulk_energy", "contact_energy", "total_energy")}
    for intersection in source["previous_memory_intersections"]:
        low, high = (dec(value) for value in interval(intersection["computational_angle_bounds"]))
        piece = intersection["original_piece"]
        rate = work_polynomials(history, trace_id, piece, dec((first+last)/2))
        for name in WORK_FAMILIES:
            values[name] += poly_integral(rate[name], low, high)*dec(last-first)*dec(density)
        for prefix, instant in (("initial", first), ("final", last)):
            polynomials = dense_source_polynomials(history, trace_id, piece, dec(instant))
            for name in ("bulk_energy", "contact_energy", "total_energy"):
                energies[prefix+"_"+name] += poly_integral(polynomials[name], low, high)*dec(density)
    for name, value in values.items():
        field = name.removesuffix("_energy")+"_energy_increment_bounds_j" if name in ("bulk_energy", "contact_energy", "total_energy") else name+"_bounds_j"
        contains(raw[field], value, "Independent exact-harmonic continuum work integral " + field)
    for name, value in energies.items():
        contains(raw[name+"_bounds_j"], value, "Independent full original continuum energy integral")
    scale = max(Decimal(1), *(abs(value) for value in values.values()), *(abs(value) for value in energies.values()))
    tolerance = Decimal(10)**-(PRECISION-40)*scale
    for name in ("bulk_energy", "contact_energy", "total_energy"):
        require(abs(energies["final_"+name]-energies["initial_"+name]-values[name]) <= tolerance,
                "Independent continuum endpoint energies do not match integrated changes")
    require(abs(values["source_work"]-values["friction_dissipation"]-values["total_energy"]-values["compliance_balance_residual"]) <= tolerance,
            "Whole source interface loses its explicit signed compliance balance residual")
    return dict(work_point_checks=checks, original_interface_harmonic_integrals=12,
                continuum_atomic_measure_difference=rational(continuum_area-nodal_area))


def check_work_primitive(raw, history, attachment, first_window):
    seal(raw, "work_sha256", "Implicit-first-root work primitive")
    require(raw["format"] == "native-original-continuous-contact-field-work-primitive-1"
            and raw["model"] == "authenticated_inviscid_down_dip_reaction_one_form_and_literal_compliance_residual"
            and raw["work_measure"] == "joules_per_square_metre_of_original_interface_area"
            and raw["final_clock_interval_is_primitive_expression_enclosure"] is True
            and raw["binary64_compliance_coefficients_preserved"] is True
            and raw["finite_point_area_m2"] is None,
            "First-root work primitive changed source, compliance or clock semantics")
    for name in ("root_midpoint_selected", "angular_integration_performed", "endpoint_backward_euler_ledgers_replaced",
                 "signed_compliance_residual_suppressed", "native_candidate_mechanics_adopted",
                 "geometry_expression_authenticated_by_this_component", "material_transfer_work_adopted",
                 "material_transfer_dissipation_adopted"):
        require(raw[name] is False, "First-root work primitive incorrectly grants " + name)
    first, final = fraction(raw["initial_time_fraction"]), interval(raw["final_time_fraction_bounds"])
    require(first == 0 and final == interval(first_window), "First-root primitive chose a point clock or later entry")
    source = raw["source_field_query"]
    require(source["trace_id"] == attachment["original_receiver_trace_id"], "First-root work borrowed donor spring history")
    same(source["computational_angle_bounds"], attachment["source_field_query"]["computational_angle_bounds"],
         "First-root work original material coordinate")
    same(source["source_expression_context"], attachment["expression_context"], "First-root work frozen expression identity")
    require(interval(source["time_fraction_bounds"]) == (first, final[1]), "Primitive lacks its whole integration clock envelope")
    intersections = check_work_source(source, history)
    require(len(raw["pieces"]) == len(intersections), "First-root work omitted a source memory piece")
    checks = 0
    for row, source_row in zip(raw["pieces"], intersections):
        require(row["previous_piece_index"] == source_row["piece_index"], "Primitive changed original memory identity")
        same(row["computational_angle_bounds"], source_row["computational_angle_bounds"], "Primitive memory-piece angle domain")
        for theta, final_clock in product(set(interval(row["computational_angle_bounds"])), set(final)):
            polynomials = work_polynomials(history, source["trace_id"], source_row["original_piece"], dec((first+final_clock)/2))
            span = dec(final_clock-first)
            for name in WORK_FAMILIES:
                value = poly_value(polynomials[name], dec(theta))*span
                field = name.removesuffix("_energy")+"_energy_increment_density_bounds_j_m2" if name in ("bulk_energy", "contact_energy", "total_energy") else name+"_density_bounds_j_m2"
                contains(row[field], value, "Independent implicit-first-root work primitive")
                contains(raw[field], value, "Independent implicit-first-root work union")
            checks += 1
    return checks


def check_whole_coarea(raw, history):
    """Check causal domain completeness independently of rectangle quadrature."""
    keys(raw, "format model scope source_history precision order radius_m carrier_map material_coarea_integrand owner_depth "
        "causal_mask_contract coarea_proof carrier_masks parent_measures original_interfaces total_archive_solid_angle_bounds_sr "
        "total_archive_surface_area_bounds_m2 once_only_material_measure_certified receiver_pair_integrals_numerically_enclosed "
        "weighted_contact_energy_integrals_evaluated contact_interface_to_swept_material_density_declared source_nodal_work_reassigned "
        "physical_pair_admission_authorized candidate_mechanics_adopted integral_sha256", "Whole causal coarea functional")
    seal(raw, "integral_sha256", "Whole causal coarea functional")
    require(raw["format"] == "native-whole-first-entry-interface-coarea-1"
            and raw["model"] == "causal_first_event_carrier_masks_authentic_whole_domain_coarea"
            and raw["scope"] == "whole_material_coarea_identity_and_separate_original_interface_measures"
            and raw["precision"] == 256 and raw["order"] == 32,
            "Whole causal coarea model or source proof budget changed")
    same(raw["source_history"], history, "Complete whole coarea source")
    require(raw["carrier_map"] == "q(theta,u)=Q_d(u)^T Q_r(u)*(S*cos(theta)+E*sin(theta))"
            and raw["material_coarea_integrand"] == "R^2*abs((a_d-a_r).Q_r(u)*c_theta(theta))*causal_mask(theta,u)"
            and raw["owner_depth"] == "-C(v)*q_x+Y(v)*q_y+Z(v)*q_z"
            and raw["causal_mask_contract"] == "earliest_positive_owner_entry_with_complete_original_supports"
            and raw["once_only_material_measure_certified"] is True,
            "Whole map omitted its causal first-event material integrand")
    for name in ("receiver_pair_integrals_numerically_enclosed", "weighted_contact_energy_integrals_evaluated",
                 "contact_interface_to_swept_material_density_declared", "source_nodal_work_reassigned",
                 "physical_pair_admission_authorized", "candidate_mechanics_adopted"):
        require(raw[name] is False, "Causal coarea incorrectly grants " + name)
    state, path = history["material_state"], history["contact_time_path"]
    banks = {row["trace_id"]: row for row in state["trace_bindings"]}
    traces = {row["trace_id"]: row for row in path["initial_mechanics"]["fault_traces"]["traces"]}
    fragments = {row["fragment_id"]: row for row in state["journal"]["source_surface"]["fragments"]}
    donor_owner, receiver_owner = state["sweep_certificate"]["donor_owner"], state["sweep_certificate"]["receiver_owner"]
    require(donor_owner == 1 and receiver_owner == 0 and len(banks) == len(traces) == 4,
            "Whole source direction or original quarter inventory changed")
    quarters, donors, receivers = set(), set(), set()
    source_supports, receiver_supports = {}, {}
    for trace_id, bank in banks.items():
        trace = traces[trace_id]
        start, end = (tuple(exact_float(value) for value in trace[name]) for name in ("start", "end"))
        require(dot(start, start) == dot(end, end) == 1 and dot(start, end) == 0
                and start[0] == end[0] == 0 and cross(start, end) == (Fraction(-1), Fraction(0), Fraction(0)),
                "Original quarters do not form the oriented exact owner boundary")
        require(bank["subducting_plate"] == donor_owner and bank["overriding_plate"] == receiver_owner,
                "Whole quarter uses a different donor or receiver owner")
        donor, receiver = fragments[bank["donor_fragment_id"]], fragments[bank["receiver_fragment_id"]]
        require(donor["parcel"]["plate"] == donor_owner and receiver["parcel"]["plate"] == receiver_owner,
                "Original bank identity conflicts with original material owner")
        pair = int(start[1]+end[1]), int(start[2]+end[2])
        require(pair not in quarters and bank["donor_fragment_id"] not in donors
                and bank["receiver_fragment_id"] not in receivers, "Original boundary quarter or bank was duplicated")
        quarters.add(pair)
        donors.add(bank["donor_fragment_id"])
        receivers.add(bank["receiver_fragment_id"])
        for collection, fragment in ((source_supports, donor), (receiver_supports, receiver)):
            normals = []
            for plane in fragment["geometry"]["supports"]:
                require(plane["rotations"] == [], "Original axis support unexpectedly rotates")
                normal = tuple(exact_float(value) for value in plane["normal"])
                require(dot(normal, start) >= 0 and dot(normal, end) >= 0,
                        "Original quarter leaves its authenticated closed bank support")
                normals.append([rational(value) for value in normal])
            collection[fragment["fragment_id"]] = normals
    require(quarters == {(-1,-1),(-1,1),(1,-1),(1,1)}
            and donors == {row["fragment_id"] for row in state["parent_allocations"] if row["owner"] == donor_owner}
            and receivers == {row["fragment_id"] for row in state["parent_allocations"] if row["owner"] == receiver_owner},
            "Closed receiver-quarter partition omits an original material bank")
    dt = exact_float(history["literal_dt_myr"])
    axes = {owner: tuple(exact_float(value)*dt for value in axis)
            for owner, axis in history["actual_candidate"]["omega_rad_per_myr"]}
    delta = tuple(a-b for a,b in zip(axes[donor_owner], axes[receiver_owner]))
    receiver_axis = axes[receiver_owner]
    # The receiver rotation is generated by its constant full SO(3) axis.
    # ||Q_r(u)e_z-e_z|| <= ||a_r x e_z|| for all0<=u<=1.  Exact L1
    # norm bounds avoid any independent sampling claim for regularity.
    change_bound = sum(abs(value) for value in delta)*(abs(receiver_axis[0])+abs(receiver_axis[1]))
    independent_projection = delta[2]-change_bound, delta[2]+change_bound
    require(independent_projection[0] > 0, "Independent whole-clock owner-axis regularity cannot be proved")
    proof = raw["coarea_proof"]
    require(proof["first_event_map_injective"] is True and proof["original_receiver_boundary_complete"] is True
            and proof["jacobian_critical_set_zero_material_measure_proved"] is True
            and proof["exact_identity"] == "sum_receivers integral(causal_mask*R^2*abs(J), dtheta du) = original_archive_measure"
            and proof["measure_authority"] == "authenticated complete first-entry domain certificate, including curved swept-only excess",
            "Causal mask no longer identifies the once-only original archive")
    for name in ("endpoint_polygon_substitution", "root_midpoint_used", "rectangle_multiplicity_used_as_archive_area"):
        require(proof[name] is False, "Whole coarea proof incorrectly adopts " + name)
    projection = interval(proof["owner_boundary_regular_axis_projection_bounds"])
    require(projection[0] > 0, "Saved coarea regularity bound is not strict")
    for clock in (Decimal(0), Decimal(1)):
        value = dot(tuple(dec(component) for component in delta), rotate((Decimal(0),Decimal(0),Decimal(1)),
            tuple(dec(component) for component in receiver_axis), clock))
        contains(proof["owner_boundary_regular_axis_projection_bounds"], value, "Independent projected full SO(3) owner axis")
    allocations = {row["fragment_id"]: row for row in state["parent_allocations"]}
    domains = {row["fragment_id"]: row for row in state["sweep_certificate"]["donor_domains"]}
    require(len(raw["carrier_masks"]) == 16, "Whole domain omits a donor/receiver carrier mask")
    observed_pairs = set()
    measure_points = 0
    for mask in raw["carrier_masks"]:
        keys(mask, "original_donor_fragment_id original_receiver_fragment_id original_donor_trace_id original_receiver_trace_id "
            "archive_domain_id domain_case source_supports receiver_supports causal_predicate first_entry_only "
            "reentry_allocates_again receiving_pair_area_numerically_enclosed measure_kernel", "Complete causal carrier mask")
        donor_id, receiver_id = mask["original_donor_fragment_id"], mask["original_receiver_fragment_id"]
        pair = donor_id, receiver_id
        require(pair not in observed_pairs and donor_id in donors and receiver_id in receivers,
                "Whole contact domain repeats a pair or invents a material bank")
        observed_pairs.add(pair)
        donor_bank, receiver_bank = banks[mask["original_donor_trace_id"]], banks[mask["original_receiver_trace_id"]]
        require(donor_bank["donor_fragment_id"] == donor_id and receiver_bank["receiver_fragment_id"] == receiver_id
                and mask["archive_domain_id"] == allocations[donor_id]["archive_domain_id"]
                and mask["domain_case"] == domains[donor_id]["domain_case"], "Causal mask changed original material archive ancestry")
        same(mask["source_supports"], source_supports[donor_id], "Complete original donor supports")
        same(mask["receiver_supports"], receiver_supports[receiver_id], "Complete original receiving bank supports")
        require(mask["causal_predicate"] == "q(theta,u) in original_donor; c(theta) in original_receiver_quarter; owner_depth(q,u)=0; owner_depth(q,v)<=0 for every 0<=v<u; owner_depth has a positive right germ at u"
                and mask["first_entry_only"] is True and mask["reentry_allocates_again"] is False
                and mask["receiving_pair_area_numerically_enclosed"] is False,
                "Causal mask turned endpoint overlap or repeated entry into a second allocation")
        kernel = mask["measure_kernel"]
        measure_points += old.check_measure(kernel, history, donor_bank, receiver_bank,
                                           kernel["source"]["native_angle_bounds"], kernel["source"]["time_fraction_bounds"])
    require(observed_pairs == set(product(donors, receivers)), "Whole donor/receiver product is incomplete")
    radius = exact_float(history["origin_surface"]["radius_km"])*1000
    require(fraction(raw["radius_m"]) == radius and len(raw["parent_measures"]) == 8,
            "Whole material coarea uses a different radius or original parent inventory")
    parent_ids = set()
    for row in raw["parent_measures"]:
        parent_id = row["original_parent_id"]
        require(parent_id not in parent_ids and parent_id in allocations, "Whole coarea repeats or invents a parent")
        parent_ids.add(parent_id)
        allocation = allocations[parent_id]
        for name, expected in (("owner",allocation["owner"]),("archive_domain_id",allocation["archive_domain_id"]),
                               ("measure_reference",allocation["measure_reference"]),
                               ("archive_solid_angle_bounds_sr",allocation["archived_area_bounds_sr"]),
                               ("parent_solid_angle_bounds_sr",allocation["parent_area_bounds_sr"])):
            same(row[name], expected, "Whole original parent measure " + name)
        same(row["archive_surface_area_bounds_m2"], dict(zip(("lo","hi"), (rational(value*radius*radius) for value in interval(allocation["archived_area_bounds_sr"])))),
             "Original archive solid angle and surface area association")
        parent_area, archive_area = interval(allocation["parent_area_bounds_sr"]), interval(allocation["archived_area_bounds_sr"])
        expected_live = max(Fraction(0),parent_area[0]-archive_area[1]), parent_area[1]-archive_area[0]
        require(interval(row["live_solid_angle_bounds_sr"]) == expected_live
                and row["exact_identity"] == "live_measure+archive_measure=original_parent_measure",
                "Whole parent archive/live partition lost once-only conservation")
    require(parent_ids == set(allocations), "Whole material parent coarea is incomplete")
    same(raw["total_archive_solid_angle_bounds_sr"], state["sweep_certificate"]["swept_area_bounds_sr"],
         "Complete original curved swept archive authority")
    require(interval(raw["total_archive_surface_area_bounds_m2"]) == iscale(interval(raw["total_archive_solid_angle_bounds_sr"]),radius*radius),
            "Whole coarea changed archive dimensional area")
    require(len(raw["original_interfaces"]) == 4, "Whole coarea lost an original interface family")
    interface_ids = set()
    for row in raw["original_interfaces"]:
        trace_id = row["trace_id"]
        require(trace_id not in interface_ids and trace_id in traces, "Whole coarea duplicates an original interface family")
        interface_ids.add(trace_id)
        trace, nodes = traces[trace_id], [item for item in history["quadrature_histories"] if item["trace_id"] == trace_id]
        extent = exact_float(next(item["extent_radians"] for item in path["step"]["previous_memory"]["traces"] if item["trace_id"] == trace_id))
        density = radius*exact_float(trace["contact_height_m"])/exact_float(nodes[0]["original_frame"][0])
        require(fraction(row["computational_extent_radians"]) == extent and fraction(row["area_density_m2_per_radian"]) == density
                and fraction(row["whole_interface_area_m2"]) == density*extent
                and row["continuum_measure_equals_literal_quadrature_total_asserted"] is False
                and row["source_family_owned_once"] is True, "Whole coarea merged continuum and atomic interface measures")
        same(row["quadrature_atoms"], [dict(point_id=node["point_id"],literal_area_m2=rational(exact_float(node["original_area_m2"]))) for node in nodes],
             "Original literal quadrature atom lineage")
    require(interface_ids == set(traces), "Original interface family ownership is incomplete")
    return dict(causal_carrier_mask_count=16,original_parent_measure_count=8,original_interface_family_count=4,
        coarea_measure_point_checks=measure_points,independent_whole_clock_critical_set_regularity_proved=True,
        receiver_quarter_coverage_independently_verified=True,whole_material_coarea_identity_independently_verified=True,
        per_pair_material_area_integrals_numerically_evaluated=False,weighted_material_contact_work_integrals_evaluated=False)


def verify_prior():
    inherited = old.verify_prior()
    summary_path = PRIOR / "release_validation_01/validation_summary.json"
    manifest_path = PRIOR / "release_validation_01/artifact_manifest.json"
    require(file_sha(summary_path) == "00c3c3ee4ac4923eb18a3f11938cfeb6a8de6d3c644e3dbf59f21341bdad5a2d"
            and file_sha(manifest_path) == "3775524aa7418cde42f833bb004fe0f55ba6d61f8e277d7d247628bffffca052",
            "Prior first-entry handover release authority changed")
    summary, manifest = read(summary_path), read(manifest_path)
    seal(summary, "summary_sha256", "Prior first-entry handover release summary")
    seal(manifest, "manifest_sha256", "Prior first-entry handover release manifest")
    require(summary["status"] == "PASS" and manifest["count"] == 48
            and len(manifest["artifacts"]) == 48, "Prior first-entry handover scope changed")
    for name, record in manifest["artifacts"].items():
        keys(record, "sha256 bytes", "Prior first-entry handover artifact record")
        path = relative_file(ROOT, name)
        require(type(record["bytes"]) is int and path.stat().st_size == record["bytes"]
                and file_sha(path) == record["sha256"], "Prior first-entry evidence changed: " + name)
    return dict(prior_manifest_records_verified=inherited["prior_artifacts_verified"] + 48,
        prior_tests_reexecuted=False, prior_tests_passed=[770, 253, 178, 265],
        prior_first_entry_summary_sha256=file_sha(summary_path),
        prior_first_entry_manifest_sha256=file_sha(manifest_path))


def check_control_sources(control, report):
    artifacts = report["artifacts_sha256"]
    require(type(artifacts) is dict and artifacts, "Domain control lacks its complete artifact inventory")
    for name, expected in artifacts.items():
        require(type(name) is str and type(expected) is str and file_sha(relative_file(control,name)) == expected,
                "Saved domain control artifact differs: " + str(name))
    authority = read(control/"source_validation_receipt.json")
    seal(authority,"receipt_sha256","Complete prior domain-control source authority")
    require(authority["format"] == "native-domain-contact-handover-source-validation-1"
            and authority["status"] == "PASS" and authority["original_source_count"] == 684
            and authority["verified_prior_manifest_records"] == 233 and authority["prior_tests_reexecuted"] is False,
            "Domain source ancestry or inherited validation scope changed")
    for name, record in authority["source_artifacts"].items():
        path = relative_file(ROOT,name)
        require(path.stat().st_size == record["bytes"] and file_sha(path) == record["sha256"],
                "Complete original source evidence differs: " + name)
    baseline = read(BASE/"starting_sources.json")
    require(type(baseline) is dict and len(baseline) == 684,"Original684Python baseline changed")
    for directory in ("release_gates_01","control_release_01"):
        for stage in ("before","after"):
            same(baseline,read(PRIOR/directory/("sources_"+stage+".json")),"Original684source release authority")
    before, after = read(control/"sources_before.json"),read(control/"sources_after.json")
    same(before,after,"Current domain-control source freeze")
    current = {path.relative_to(ROOT).as_posix():file_sha(path) for directory in ("tectonics","tests","scripts","analysis")
               for path in sorted((ROOT/directory).rglob("*.py"))}
    same(current,after,"Exhaustive current Python source enumeration")
    require(all(current.get(name) == expected for name,expected in baseline.items()),"A frozen original Python source changed")
    require(report["original_source_count"] == 684 and report["source_count"] == len(current)
            and report["added_python_sources"] == sorted(set(current)-set(baseline))
            and report["existing_sources_changed"] == [],"Current domain-control source count or lineage changed")
    pairs = (("source_history_before.json","source_history_after.json"),
             ("source_checkpoint_before.json","source_checkpoint_after.json"),
             ("force_candidate_before.json","force_candidate_after.json"),
             ("domain_contact_handover.json","domain_contact_handover_replayed.json"),
             ("first_entry_interface_integral.json","first_entry_interface_integral_replayed.json"),
             ("original_continuum_work.json","original_continuum_work_replayed.json"),
             ("domain_point_queries.json","domain_point_queries_replayed.json"))
    same(report["byte_pairs"],[list(pair) for pair in pairs],"Full source and typed restart bytepair references")
    byte_pairs = []
    for first,second in pairs:
        require((control/first).read_bytes() == (control/second).read_bytes(),"Original source or strict domain restart bytes changed")
        byte_pairs.append(dict(first=first,second=second,bytes=(control/first).stat().st_size,sha256=file_sha(control/first)))
    for name, original in (("source_history_before.json","source_history_before.json"),
                           ("source_checkpoint_before.json","source_checkpoint_before.json"),
                           ("force_candidate_before.json","force_candidate_before.json")):
        require((control/name).read_bytes() == (PRIOR/"control_release_01"/original).read_bytes(),
                "Full original immutable source bytes changed: " + name)
    history = read(control/"source_history_before.json")
    seal(history,"binding_sha256","Complete original physical history")
    checkpoint,candidate = read(control/"source_checkpoint_before.json"),read(control/"force_candidate_before.json")
    require(type(candidate) is dict and len(candidate) == 14,"Fourteen original force candidate fields required")
    same(history["actual_candidate"],candidate,"Full original actual force candidate")
    same(history["accepted_before"],checkpoint["native_underthrust_motion"],"Full authentic original force checkpoint")
    require(history["original_checkpoint_sha256"] == file_sha(control/"source_checkpoint_before.json"),
            "Original source checkpoint byte identity changed")
    ledger = read(control/"original_quadrature_ledger.json")
    same(ledger,read(PRIOR/"control_release_01/original_quadrature_ledger.json"),"Complete original once-only16atom ledger")
    energy_fields = old.check_original_ledger(ledger,history)
    return history,checkpoint,candidate,current,byte_pairs,energy_fields,dict(artifacts)


def check_mutation_receipt(raw):
    keys(raw,"format count mutations receipt_sha256","Recorded domain mutation refusals")
    seal(raw,"receipt_sha256","Recorded domain mutation refusals")
    expected_labels = (
        "unknown_domain_schema_field","domain_precision_literal_type_changed","receiving_history_family_duplicated",
        "receiving_family_original_history_count_changed","receiving_family_consumed_lineage_duplicated",
        "causal_branch_source_family_changed","causal_branch_finite_history_fabricated","whole_donor_beam_width_changed",
        "whole_donor_beam_shape_changed","whole_donor_native_clamp_support_fabricated","domain_source_work_budget_changed",
        "original_atom_lineage_duplicated","global_unpaid_transfer_work_fabricated","material_contact_work_density_fabricated",
        "literal_atom_and_continuum_work_summed","unknown_integral_schema_field","integral_precision_literal_type_changed",
        "carrier_mask_omitted","carrier_mask_duplicated","receiving_parent_changed","receiving_support_changed",
        "archive_parent_omitted","archive_area_changed","continuum_interface_density_changed","literal_quadrature_area_changed",
        "initial_event_map_injectivity_claim_changed","receiver_boundary_incomplete","owner_regular_axis_projection_changed",
        "rectangle_multiplicity_claimed_as_archive_area","contact_density_inferred_from_material_area",
        "numerical_receiving_pair_integration_fabricated","nodal_work_reassigned","unknown_work_schema_field",
        "whole_interface_source_work_changed","whole_interface_dissipation_changed","whole_interface_initial_energy_changed",
        "whole_interface_final_energy_changed","literal_compliance_residual_suppressed",
        "continuum_measure_replaced_by_quadrature_measure","continuum_work_replaced_by_backward_euler_work",
        "original_interface_area_density_changed","material_swept_area_substituted_into_work",
        "whole_interface_invented_native_force_acceptance","source_fraction_interval_changed")
    require(raw["format"] == "native-domain-contact-handover-mutation-refusals-1"
            and type(raw["count"]) is int and raw["count"] == len(raw["mutations"]) == len(expected_labels)
            and [row["label"] for row in raw["mutations"]] == list(expected_labels),
            "Recorded domain mutation coverage is incomplete")
    labels = set()
    for row in raw["mutations"]:
        keys(row,"label changed_path rejected valid_rehashed_declared_checksum exception_type message mutated_dto_sha256",
             "Recorded rehashed domain mutation")
        require(type(row["label"]) is str and row["label"] and row["label"] not in labels
                and type(row["changed_path"]) is str and row["changed_path"]
                and row["rejected"] is True and row["valid_rehashed_declared_checksum"] is True
                and row["exception_type"] in ("ValueError","TypeError") and type(row["message"]) is str and row["message"]
                and type(row["mutated_dto_sha256"]) is str and len(row["mutated_dto_sha256"]) == 64
                and all(character in "0123456789abcdef" for character in row["mutated_dto_sha256"]),
                "A recorded rehashed domain mutation was not refused")
        labels.add(row["label"])
    return raw["count"]


def check_domain_registry(raw, history, law, coarea_raw, work_rows):
    keys(raw,"format model source_history precision order explicit_law interface_integral families branches source_beam_supports "
        "source_work_budgets original_atom_lineages whole_domain_field_transfer_committed original_parent_count receiver_field_family_count "
        "causal_material_branch_count exact_global_transfer_balances source_line_work_integrals_evaluated "
        "receiving_pair_weighted_contact_integrals_numerically_enclosed interface_energy_or_work_density_on_archived_bulk_declared "
        "contactline_work_and_literal_atom_work_are_summed source_mechanical_candidate_work_replaced_by_contactline_budget "
        "physical_pair_admission_authorized candidate_mechanics_adopted accepted_native_force_intervals accepted_source_work_j state_sha256",
        "Whole once-owned contact-field registry")
    seal(raw,"state_sha256","Whole once-owned contact-field registry")
    require(raw["format"] == "native-whole-first-entry-contact-domain-handover-1"
            and raw["model"] == "four_once_owned_receiver_fields_sixteen_causal_material_pullbacks"
            and raw["precision"] == 256 and raw["order"] == 32,
            "Whole domain changed original family ownership or proof budget")
    same(raw["source_history"],history,"Complete whole registered contact source")
    same(raw["explicit_law"],law,"Complete explicit registered transport law")
    same(raw["interface_integral"],coarea_raw,"Whole registry's original causal coarea functional")
    require(raw["whole_domain_field_transfer_committed"] is True and raw["original_parent_count"] == 8
            and raw["receiver_field_family_count"] == 4 and raw["causal_material_branch_count"] == 16
            and raw["source_line_work_integrals_evaluated"] is True,
            "Whole domain did not commit its complete source-family transfer")
    for name in ("receiving_pair_weighted_contact_integrals_numerically_enclosed",
                 "interface_energy_or_work_density_on_archived_bulk_declared",
                 "contactline_work_and_literal_atom_work_are_summed",
                 "source_mechanical_candidate_work_replaced_by_contactline_budget",
                 "physical_pair_admission_authorized","candidate_mechanics_adopted"):
        require(raw[name] is False,"Whole domain incorrectly grants " + name)
    require(type(raw["accepted_native_force_intervals"]) is int and raw["accepted_native_force_intervals"] == 0
            and fraction(raw["accepted_source_work_j"]) == 0,"Whole registry adopted native force clock/work")
    balances = raw["exact_global_transfer_balances"]
    keys(balances,"external_work_j dissipation_j strain_energy_change_j proof source_history_owned_once first_event_mask_excludes_reentry",
         "Exact whole-domain reference-transfer balances")
    require(all(fraction(balances[name]) == 0 for name in ("external_work_j","dissipation_j","strain_energy_change_j"))
            and balances["proof"] == "identical source p,c,strains,tractions,energy at every common implicit first event with identical receiver interface measure"
            and balances["source_history_owned_once"] is True and balances["first_event_mask_excludes_reentry"] is True,
            "Whole registry lost pointwise conservative reference identities")
    banks = {row["trace_id"]:row for row in history["material_state"]["trace_bindings"]}
    traces = {row["trace_id"]:row for row in history["contact_time_path"]["initial_mechanics"]["fault_traces"]["traces"]}
    require(len(raw["families"]) == 4,"Whole registry lost a source receiving family")
    families = {}
    for family in raw["families"]:
        keys(family,"family_id trace_id original_binding_id active_binding_id original_banks consumed_binding_ids revision memory_reference "
             "reference_clock_expression original_history_count active_history_count inactive_or_reentry_branch",
             "Once-owned receiving field family")
        trace_id = family["trace_id"]
        require(trace_id not in families and trace_id in banks,"A receiving family was cloned or uses a different original trace")
        families[trace_id] = family
        identity = "domain-contact:"+history["original_checkpoint_sha256"]+":"+trace_id
        require(family["family_id"] == identity and family["original_binding_id"] == identity+":original"
                and family["active_binding_id"] == identity+":first-entry-field:1"
                and family["consumed_binding_ids"] == [identity+":original"] and type(family["revision"]) is int and family["revision"] == 1
                and family["original_history_count"] == family["active_history_count"] == 1,
                "Whole receiving field cloned or failed to replace its original reference")
        require(family["memory_reference"] == history["contact_time_path"]["path_sha256"]
                and family["reference_clock_expression"] == "each material reference is frozen at its unique implicit first_entry(q)"
                and family["inactive_or_reentry_branch"] == "retains original attachment; no second family or work debit",
                "Whole field forgot full source history or consumed reentry twice")
        correspondence = dict(model="original_receiver_attached_full_field",trace_id=trace_id,
                              source_path=history["contact_time_path"]["path_sha256"])
        same(family["original_banks"],old.bank_attachment(banks[trace_id],correspondence),
             "Once-owned original material banks")
    require(set(families) == set(banks),"Whole registry omits an original receiving field")
    masks = {(row["original_donor_fragment_id"],row["original_receiver_fragment_id"]):row for row in coarea_raw["carrier_masks"]}
    require(len(raw["branches"]) == 16,"Whole registry lost a causal pullback branch")
    branch_ids, branch_pairs = set(),set()
    for branch in raw["branches"]:
        keys(branch,"branch_id source_family_id donor_trace_id receiver_trace_id original_donor_fragment_id original_receiver_fragment_id "
             "archive_domain_id target_banks anchor_expression frozen_first_event_expression new_finite_history_created additional_area_or_work_created",
             "Complete functional contact pullback branch")
        pair = branch["original_donor_fragment_id"],branch["original_receiver_fragment_id"]
        require(pair in masks and pair not in branch_pairs and branch["branch_id"] not in branch_ids,
                "Whole registry duplicates or invents a contact branch")
        branch_pairs.add(pair)
        branch_ids.add(branch["branch_id"])
        mask = masks[pair]
        family = families[branch["receiver_trace_id"]]
        donor_bank,receiver_bank = banks[branch["donor_trace_id"]],banks[branch["receiver_trace_id"]]
        require(branch["source_family_id"] == family["family_id"]
                and branch["branch_id"] == family["active_binding_id"]+":"+pair[0]
                and branch["donor_trace_id"] == mask["original_donor_trace_id"]
                and branch["receiver_trace_id"] == mask["original_receiver_trace_id"]
                and branch["archive_domain_id"] == mask["archive_domain_id"],"Causal branch changed first receiver or source family")
        context = dict(model=raw["model"],family_id=family["family_id"],donor_parent_id=pair[0],first_receiver_id=pair[1],
            causal_predicate=mask["causal_predicate"],
            label_expression="original_q=Q_d(first_entry)^T*Q_r(first_entry)*c(theta(first_entry))",
            label_frozen_during_later_force_variations=True,reference_clock_frozen=True,midpoint_used=False)
        target_bank = dict(donor_bank,receiver_fragment_id=receiver_bank["receiver_fragment_id"],
                           receiver_material_id=receiver_bank["receiver_material_id"],overriding_plate=receiver_bank["overriding_plate"])
        same(branch["target_banks"],old.bank_attachment(target_bank,context),"Changed material attachment and frozen first-event correspondence")
        require(branch["anchor_expression"] == "source_free(theta*,u*) + h_current(original_q) - h_first_entry(original_q)"
                and branch["frozen_first_event_expression"] == "theta*=receiver_coordinate(first_entry(original_q)); u*=first_entry(original_q)"
                and branch["new_finite_history_created"] is False and branch["additional_area_or_work_created"] is False,
                "Whole branch duplicated source work or failed to freeze its material reference")
        for parameter in ("columns","law","water"):
            same(traces[branch["donor_trace_id"]][parameter],traces[branch["receiver_trace_id"]][parameter],
                 "Unchanged whole material transport parameter " + parameter)
    require(branch_pairs == set(masks),"Whole functional contact branch coverage is incomplete")
    radius = exact_float(history["origin_surface"]["radius_km"])*1000
    rotation_bound = exact_float(history["literal_dt_myr"])*sum((sum(abs(exact_float(value)) for value in axis)
                                      for _,axis in history["actual_candidate"]["omega_rad_per_myr"]),Fraction(0))
    angle = rotation_bound/(1-rotation_bound**2)
    require(0 < rotation_bound < Fraction(1,4),"Independent whole original donor-strip proof is outside its valid domain")
    parameters = {row["trace_id"]:row for row in history["contact_time_path"]["initial_mechanics"]["bending_parameters"]}
    require(len(raw["source_beam_supports"]) == 4,"Whole registry omits an original donor beam support")
    support_ids = set()
    for row in raw["source_beam_supports"]:
        trace_id = row["donor_trace_id"]
        require(trace_id in banks and trace_id not in support_ids,"Whole registry repeats or invents a source beam")
        support_ids.add(trace_id)
        width, distance = exact_float(parameters[trace_id]["sub"]["width_m"]),radius*angle
        z = distance/width
        shape = 1-Fraction(3,2)*z+Fraction(1,2)*z**3,Fraction(1)
        require(distance < width and row["donor_fragment_id"] == banks[trace_id]["donor_fragment_id"]
                and fraction(row["radius_m"]) == radius and fraction(row["width_m"]) == width
                and fraction(row["rotation_path_bound_rad"]) == rotation_bound
                and fraction(row["max_inward_angle_rad"]) == angle
                and fraction(row["max_inward_distance_m"]) == distance and interval(row["shape_bounds"]) == shape
                and row["strip_membership_proved"] is True and row["native_force_and_clamp_support_certified"] is False,
                "Whole incoming label left its own donor Ritz strip or acquired force/clamp admission")
    budgets = {row["trace_id"]:row["whole_original_interface_integral"] for row in work_rows}
    require(len(raw["source_work_budgets"]) == 4,"Whole registry lost a source line budget")
    budget_ids = set()
    for row in raw["source_work_budgets"]:
        trace_id = row["trace_id"]
        require(trace_id in budgets and trace_id not in budget_ids,"Whole registry duplicates an original continuum line budget")
        budget_ids.add(trace_id)
        full_budget = budgets[trace_id]
        for name in ("original_interface_area_density_m2_per_computational_radian","continuum_interface_area_m2",
                     "original_nodal_interface_area_m2","continuum_minus_original_nodal_interface_area_m2",
                     "source_work_bounds_j","friction_dissipation_bounds_j","bulk_energy_increment_bounds_j",
                     "contact_energy_increment_bounds_j","total_energy_increment_bounds_j","compliance_balance_residual_bounds_j",
                     "initial_bulk_energy_bounds_j","initial_contact_energy_bounds_j","initial_total_energy_bounds_j",
                     "final_bulk_energy_bounds_j","final_contact_energy_bounds_j","final_total_energy_bounds_j"):
            same(row[name],full_budget[name],"Whole registry's independent original line work budget " + name)
        context = dict(model=raw["model"],family_id=families[trace_id]["family_id"],scope="original_receiver_line_once_unaccepted_source_path")
        same(row["source_expression_context"],context,"Once-owned source line budget reference")
        require(row["source_path"] == history["contact_time_path"]["path_sha256"]
                and row["balance_identity"] == "source_work=total_energy_increment+friction_dissipation+signed_compliance_residual"
                and row["literal_atom_ledgers_replaced"] is False and row["is_swept_material_energy_density"] is False,
                "Whole registry confused original line work and material/atomic measures")
    expected_atoms = [dict(checkpoint_sha256=history["original_checkpoint_sha256"],point_id=row["point_id"],trace_id=row["trace_id"],
        original_area_m2=rational(exact_float(row["original_area_m2"])),state_preserved_once=True,
        full_state_reference="source_history.quadrature_histories") for row in history["quadrature_histories"]]
    same(raw["original_atom_lineages"],expected_atoms,"All16 original finite history/work lineages retained once")
    return dict(whole_first_entry_domain_registry_committed=True,once_owned_receiving_history_family_count=4,
        functional_contact_pullback_branch_count=16,whole_original_donor_ritz_strips_independently_verified=True,
        source_line_budget_count=4,original_atom_lineage_count=16,exact_global_zero_transfer_balances_verified=True,
        source_line_work_and_literal_atomic_work_summed=False,interface_energy_density_on_archived_bulk_declared=False)


def check_domain_point_query(raw,registry,binding,history,law):
    keys(raw,"format registry_sha256 point_handover branches source_family_ids source_work_primitives status registered_history_created "
        "finite_point_area_m2 finite_point_work_j additional_handover_work_j physical_pair_admission_authorized query_sha256",
        "Whole-domain material point query")
    seal(raw,"query_sha256","Whole-domain material point query")
    require(raw["format"] == "native-whole-first-entry-contact-domain-point-query-1"
            and raw["registry_sha256"] == registry["state_sha256"] and raw["registered_history_created"] is False
            and raw["finite_point_area_m2"] is None and raw["finite_point_work_j"] is None
            and fraction(raw["additional_handover_work_j"]) == 0 and raw["physical_pair_admission_authorized"] is False,
            "A point query copied finite history/area/work or selected an unauthenticated registry")
    handover = raw["point_handover"]
    require(raw["status"] == handover["status"],"Domain query lost its original first-event state")
    attachments = handover["attachments"]
    require(len(raw["branches"]) == len(raw["source_family_ids"]) == len(raw["source_work_primitives"]) == len(attachments),
            "A point query duplicated a branch, family or work primitive")
    for index,attachment in enumerate(attachments):
        pair = tuple(attachment["desired_fragment_pair"])
        branches = [row for row in registry["branches"] if (row["original_donor_fragment_id"],row["original_receiver_fragment_id"]) == pair]
        require(len(branches) == 1,"A point query lacks a unique whole causal first-entry branch")
        branch = branches[0]
        require(raw["branches"][index] == branch["branch_id"] and raw["source_family_ids"][index] == branch["source_family_id"],
                "A point query changed source family or selected a later receiver")
        check_work_primitive(raw["source_work_primitives"][index],history,attachment,binding["time_fraction_bounds"])
    return dict(),handover


def _run_audit(control):
    started = time.perf_counter()
    control = Path(control).resolve()
    require(control.is_dir(),"Complete domain handover control evidence is missing")
    report = read(control/"report.json")
    seal(report,"report_sha256","Complete domain contact control report")
    require(report["format"] == "native-domain-contact-handover-control-1" and report["status"] == "PASS",
            "Complete domain contact control failed or changed format")
    prior = verify_prior()
    history,checkpoint,candidate,current,byte_pairs,energy_fields,artifacts = check_control_sources(control,report)
    law = read(control/"declared_handover_law.json")
    keys(law,"model provenance","Explicit domain contact transport law")
    require(law["model"] == "receiver_attached_complete_lumped_contact_configuration"
            and type(law["provenance"]) is str and law["provenance"].strip() == law["provenance"] and law["provenance"],
            "Domain contact transport lacks its explicit source law")
    same(law,report["law"],"Explicit full domain control transport law")
    coarea_raw = read(control/"first_entry_interface_integral.json")
    coarea = check_whole_coarea(coarea_raw,history)
    work_rows = read(control/"original_continuum_work.json")
    expected_traces = [row["trace_id"] for row in history["contact_time_path"]["step"]["previous_memory"]["traces"]]
    require(type(work_rows) is list and [row["trace_id"] for row in work_rows] == expected_traces
            and len(work_rows) == 4,"Four original continuum work budgets changed or duplicated")
    work_checks, harmonic_integrals, measure_differences = 0,0,{}
    for row in work_rows:
        keys(row,"trace_id whole_clock_rate_query exact_clock_density_integral whole_original_interface_integral",
             "Original once-only continuum interface work budget")
        for name in ("whole_clock_rate_query","exact_clock_density_integral"):
            source = row[name]["source_field_query"]
            require(source["trace_id"] == row["trace_id"] and interval(source["time_fraction_bounds"]) == (Fraction(0),Fraction(1)),
                    "Complete original continuum work used a different field or source clock")
        work_checks += check_field_work(row["whole_clock_rate_query"],history,integrated=False)
        work_checks += check_field_work(row["exact_clock_density_integral"],history,integrated=True)
        checked = check_interface_work(row["whole_original_interface_integral"],history)
        work_checks += checked["work_point_checks"]
        harmonic_integrals += checked["original_interface_harmonic_integrals"]
        measure_differences[row["trace_id"]] = checked["continuum_atomic_measure_difference"]
    registry = read(control/"domain_contact_handover.json")
    registry_result = check_domain_registry(registry,history,law,coarea_raw,work_rows)
    records = read(control/"point_certificates.json")
    proofs = old.independently_audit_chronologies(records,history,checkpoint,candidate)
    chronologies = {row["label"]:row["certificate"] for row in records}
    bindings = read(control/"original_first_entry_bindings.json")
    same(bindings,read(PRIOR/"control_release_01/original_first_entry_bindings.json"),"Complete original first-root physical contexts")
    bindings = {row["label"]:row["binding"] for row in bindings}
    points = read(control/"domain_point_queries.json")
    require(type(points) is list and [row["label"] for row in points] == list(old.POINT_LABELS),
            "Complete seven-context domain query changed")
    point_results = dict(field_point_checks=0,measure_point_checks=0,source_chart_corner_checks=0,numerical_owner_roots=0)
    attachments = {}
    for row in points:
        keys(row,"label query","Full domain point observation")
        label = row["label"]
        checked,handover = check_domain_point_query(row["query"],registry,bindings[label],history,law)
        checked_old = old.check_handover(handover,bindings[label],chronologies[label],proofs[label],history,law)
        for name,value in checked_old.items():
            point_results[name] += value
        if label in old.ENTRY_LABELS:
            attachments[label] = handover["attachments"][0]
    mask_observations = read(control/"carrier_mask_observations.json")
    require([row["label"] for row in mask_observations] == list(old.POINT_LABELS),"Complete carrier-mask controls changed")
    mask_point_checks = 0
    for row in mask_observations:
        keys(row,"label masks source_family_count_before source_family_count_after first_root_midpoint_used finite_point_area_m2 finite_point_work_j",
             "Complete first-root carrier-mask observation")
        label = row["label"]
        require(row["source_family_count_before"] == row["source_family_count_after"] == 4
                and row["first_root_midpoint_used"] is False and row["finite_point_area_m2"] is None and row["finite_point_work_j"] is None,
                "Domain point query cloned an original family or created finite point measure")
        require(len(row["masks"]) == 16,"Complete point observation omitted a carrier mask")
        for observed,mask in zip(row["masks"],coarea_raw["carrier_masks"]):
            keys(observed,"original_donor_fragment_id original_receiver_fragment_id selected","Carrier-mask point result")
            expected = mask["original_donor_fragment_id"] == chronologies[label]["original_parent_id"] and mask["original_receiver_fragment_id"] in proofs[label]["first_receivers"]
            same(observed,dict(original_donor_fragment_id=mask["original_donor_fragment_id"],
                original_receiver_fragment_id=mask["original_receiver_fragment_id"],selected=expected),
                "Independently proved earliest receiver carrier mask")
            mask_point_checks += 1
    primitives = read(control/"first_entry_work_primitives.json")
    require([row["label"] for row in primitives] == list(old.ENTRY_LABELS),"Four implicit-first-root work primitives changed")
    primitive_checks = 0
    for row in primitives:
        keys(row,"label source_receiver_trace_id primitive","First-root work primitive observation")
        label = row["label"]
        require(row["source_receiver_trace_id"] == attachments[label]["original_receiver_trace_id"],
                "First-root primitive selected a neighbouring spring history")
        primitive_checks += check_work_primitive(row["primitive"],history,attachments[label],bindings[label]["time_fraction_bounds"])
    mutations = check_mutation_receipt(read(control/"mutation_refusals.json"))
    counts = dict(original_source_count=684,original_candidate_field_count=14,original_node_count=16,original_context_count=7,
        first_root_handover_count=4,negative_context_count=3,original_continuum_trace_count=4,domain_parent_count=8,
        causal_mask_count=16,mutation_refusal_count=mutations,remaining_work_block_count=3)
    for name,value in counts.items():
        require(type(report[name]) is int and report[name] == value,"Domain control count changed: " + name)
    for name in ("source_before_after_unchanged","original_typed_checkpoint_loaded","original_force_re_solved",
        "history_builder_original_force_re_solved","prior_material_state_replayed","original_candidate_all_fields_identical",
        "first_receiver_pair_6_to_3_committed","source_history_bytes_unchanged","source_checkpoint_bytes_unchanged",
        "source_candidate_bytes_unchanged","original_quadrature_measure_work_counted_once","unchanged_global_beam_dofs_and_stiffness",
        "continuum_and_original_quadrature_measures_distinct","continuum_work_first_law_retains_literal_compliance_residual",
        "whole_first_entry_domain_registry_committed","whole_original_interface_continuum_work_integrated"):
        require(report[name] is True,"Domain control lacks a source or functional guarantee: " + name)
    for name in ("first_root_midpoint_used","whole_first_entry_interface_domain_numerically_integrated",
        "per_pair_incoming_continuum_work_numerically_integrated","transported_candidate_is_force_equilibrium",
        "native_full_bank_and_clamp_support_certified","finite_measure_assigned_to_first_entry_point",
        "nodal_work_assigned_to_first_entry_point","candidate_mechanics_adopted","physical_pair_admission_authorized","global_native_inventory_updated"):
        require(report[name] is False,"Domain control incorrectly grants " + name)
    for name in ("transfer_external_work_j","transfer_dissipation_j","accepted_loaded_native_intervals","accepted_source_work_j"):
        require(type(report[name]) is int and report[name] == 0,"Domain control adopted force work or clock")
    artifacts["report.json"] = file_sha(control/"report.json")
    require(not any(name == "tectonics" or name.startswith("tectonics.") for name in sys.modules),
            "Independent domain audit imported production modules")
    return dict(format="native-domain-contact-handover-independent-audit-1",status="PASS",
        seconds=time.perf_counter()-started,decimal_precision=PRECISION,standard_library_only=True,production_modules_imported=False,
        scope="independent complete causal coarea/quarter partition and whole-clock critical-set regularity, once-owned field/work lineage and original donor Ritz-strip proofs; independent360Decimal exact harmonic original-interface work/energy budgets and true first-root expressions; receiving-pair weighted material budgets and accepted force admission remain outside this proof",
        exact_scope="complete causal donor/receiver product, original quarter coverage and whole-clock critical-set regularity; complete original archive/live measures; four once-only interface/history families; explicit pointwise zero-transfer identities and original literal16atom work ledger",
        numerical_scope="independent360Decimal source Fourier antiderivatives, continuum first-law and endpoint energies; source carrier Jacobian corners and proved owner-root observations; no receiving-pair weighted material work integration or force admission",
        original_energy_work_field_count=energy_fields,continuum_work_point_checks=work_checks,
        original_interface_harmonic_integral_count=harmonic_integrals,continuum_atomic_area_differences=measure_differences,
        first_root_work_primitive_count=4,first_root_work_primitive_point_checks=primitive_checks,carrier_mask_point_checks=mask_point_checks,
        mutation_refusal_count=mutations,recorded_mutation_refusals_verified=True,mutation_tests_independently_reexecuted=False,
        original_context_count=7,first_root_handover_count=4,negative_context_count=3,original_node_count=16,
        original_source_count=684,source_count=len(current),existing_sources_changed=[],source_before_after_unchanged=True,
        exhaustive_current_source_map_verified=True,byte_pairs=byte_pairs,
        chronology_leaf_count=sum(proof["leaf_count"] for proof in proofs.values()),chronology_root_count=sum(proof["root_count"] for proof in proofs.values()),
        first_root_midpoint_used=False,production_root_clock_selected=False,
        first_receiver_pair_6_to_3_committed=True,
        original_quadrature_measure_work_counted_once=True,continuum_and_original_quadrature_measures_distinct=True,
        signed_literal_compliance_residual_preserved=True,whole_original_interface_continuum_work_integrated=True,
        whole_first_entry_interface_domain_numerically_integrated=False,per_pair_incoming_continuum_work_numerically_integrated=False,
        native_full_bank_and_clamp_support_certified=False,finite_measure_assigned_to_first_entry_point=False,
        nodal_work_assigned_to_first_entry_point=False,candidate_mechanics_adopted=False,physical_pair_admission_authorized=False,
        global_native_inventory_updated=False,transfer_external_work_j=0,transfer_dissipation_j=0,
        accepted_loaded_native_intervals=0,accepted_source_work_j=0,
        audit_source_sha256=file_sha(Path(__file__)),input_artifact_sha256=artifacts,**prior,**coarea,**registry_result,**point_results)


def run_audit(control):
    with localcontext() as context:
        context.prec = PRECISION
        return _run_audit(control)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    arguments = parser.parse_args()
    receipt = run_audit(arguments.control)
    receipt["receipt_sha256"] = digest(receipt)
    arguments.out.parent.mkdir(parents=True,exist_ok=True)
    with arguments.out.open("xb") as stream:
        stream.write(encoded(receipt)+b"\n")
    print(json.dumps({name:receipt[name] for name in ("status","seconds","source_count","causal_carrier_mask_count",
        "first_root_handover_count","original_interface_harmonic_integral_count","original_node_count")},sort_keys=True))


if __name__ == "__main__":
    main()
