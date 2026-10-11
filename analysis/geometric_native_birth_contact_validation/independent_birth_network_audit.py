"""Independently contract complete declared force and work equations.

The stdlib auditor checks the supplied authoritative Decimal coordinates.  It
does not re-solve equilibrium, authenticate native geometry or admit a time
interval.  Original matrices and every tiny new family stay explicit.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal, localcontext
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis.geometric_native_birth_contact_validation.verify_baseline import (
    digest, encoded, read, write,
)
from analysis.geometric_native_birth_contact_validation.independent_birth_energy_audit import (
    ZERO, audit_contact_state, audit_foundation_state, close, normal_response, number,
)


def coordinate(value):
    if type(value) is str:
        result = Decimal(value)
        if not result.is_finite() or str(result) != value:
            raise ValueError("The authoritative coordinate string is not finite and canonical")
        return result
    return number(value, "Coordinate")


def dot(first, second):
    if len(first) != len(second):
        raise ValueError("A declared operator has incomplete dimension")
    return sum((a*b for a, b in zip(first, second)), ZERO)


def multiply(matrix, vector):
    return [dot(row, vector) for row in matrix]


def relative(residual, operands):
    scale = sum((abs(value) for value in operands), ZERO)
    if scale == 0:
        if residual != 0:
            raise ValueError("A nonzero force has no physical balancing operands")
        return ZERO
    return abs(residual)/scale


def computed_close(actual, expected, name, tolerance):
    error = relative(actual-expected, (actual, expected))
    if error > Decimal(tolerance):
        raise ValueError(name + " differs from its independently contracted value")
    return error


def audit_network(path):
    started = time.perf_counter()
    packet = read(path)
    if packet["force_geometry_admission_authorized"] is not False:
        raise ValueError("A generic network packet cannot claim native admission")
    inputs, result = packet["inputs"], packet["result"]
    radius = number(inputs["radius_m"], "Radius", positive=True)
    duration = number(inputs["dt_s"], "Physical duration", positive=True)
    prefix = len(inputs["old_stiffness"])
    dimension = prefix+len(packet["born_foundations"])
    new = [coordinate(value) for value in result["coordinates_decimal"]]
    old = [coordinate(value) for value in inputs["coordinates_before"]]
    if len(old) != dimension or len(new) != dimension:
        raise ValueError("The complete coordinate vector differs")
    contacts, foundations = packet["born_contacts"], packet["born_foundations"]
    if (len(set(item["reference"]["family_id"] for item in contacts)) != len(contacts)
            or len(set(item["reference"]["patch_id"] for item in foundations)) != len(foundations)
            or sorted(item["radial_dof_index"] for item in foundations) != list(range(prefix, dimension))):
        raise ValueError("A born contact or independent foundation is missing or duplicated")
    with localcontext() as context:
        context.prec = 200
        delta = [a-b for a, b in zip(new, old)]
        drag = [[number(value, "Source drag") for value in row] for row in inputs["drag_nm_s"]]
        cold_matrix = [[number(value, "Original stiffness") for value in row] for row in inputs["old_stiffness"]]
        source = [number(value, "Source torque") for value in inputs["source_torque_nm"]]
        basal_force = [value/duration for value in multiply(drag, delta)]
        cold_force = multiply(cold_matrix, new[:prefix])+[ZERO]*(dimension-prefix)
        foundation_force = [ZERO]*dimension
        actual_foundation_force = [ZERO]*dimension
        foundation_work = ZERO
        actual_foundation_work = ZERO
        foundation_rows = []
        for index, item in enumerate(foundations):
            reference, parameters = item["reference"], item["parameters"]
            coefficient = (number(parameters["density_kg_m3"], "Hydrostatic density", positive=True)
                *number(parameters["gravity_m_s2"], "Gravity", positive=True)
                *number(reference["basal_area_m2"], "Basal area", positive=True))
            dof = item["radial_dof_index"]
            birth = number(reference["radial_birth_m"], "Radial birth")
            reaction = coefficient*(new[dof]*radius-birth)
            foundation_force[dof] += reaction*radius
            expected_work = reaction*delta[dof]*radius
            foundation_work += expected_work
            state_before, state_after = item["state_before"], result["born_foundation_states"][index]
            actual_radial = number(state_after["radial_displacement_m"], "Actual foundation displacement")
            actual_foundation_force[dof] += coefficient*(actual_radial-birth)*radius
            actual_foundation_work += number(result["born_foundation_steps"][index]["external_work_j"], "Actual foundation work")
            before_receipt = audit_foundation_state(state_before)
            after_receipt = audit_foundation_state(state_after)
            if state_before["parameters"] != state_after["parameters"] or state_before["reference"] != state_after["reference"]:
                raise ValueError("A foundation changed its independently supplied physics")
            error = close(result["born_foundation_steps"][index]["external_work_j"], expected_work,
                "Foundation local work", tolerance="5e-10")
            foundation_rows.append(dict(patch_id=reference["patch_id"],
                local_work_relative_error=str(error), before=before_receipt, after=after_receipt))
        all_contact_forces, actual_contact_forces = [], []
        old_work = born_work = actual_old_work = actual_born_work = ZERO
        old_reaction_errors, born_reaction_errors = [], []
        for index, (material, area, operator, before_state) in enumerate(zip(packet["original_materials"],
                inputs["old_areas_m2"], inputs["old_anchor_operators"], packet["original_states_before"])):
            normal = [number(value, "Original normal operator") for value in operator[0]]+[ZERO]*(dimension-prefix)
            shear = [number(value, "Original shear operator") for value in operator[1]]+[ZERO]*(dimension-prefix)
            gap = number(before_state["gap_free_m"], "Original previous gap")+dot(normal, delta)
            if gap > 0:
                raise ValueError("An original closed contact acquired unsupported opening")
            local = normal_response(material["columns"], material["law"], area, float(gap))
            # Force coefficients use the high precision free gap rather than
            # its diagnostic binary64 representative.
            coefficient = number(area, "Original area", positive=True)*local["effective_stiffness_pa_m"]
            reaction = coefficient*gap
            all_contact_forces.append([reaction*value for value in normal])
            expected_work = reaction*dot(normal, delta)
            old_work += expected_work
            state = result["original_point_states"][index][1]
            actual_normal = number(area, "Original area")*number(state["traction_normal_pa"], "Actual original normal traction")
            actual_shear = number(area, "Original area")*number(state["traction_shear_pa"], "Actual original shear traction")
            actual_contact_forces.append([actual_normal*n+actual_shear*s for n, s in zip(normal, shear)])
            actual_old_work += number(result["original_point_steps"][index]["external_work_j"], "Actual original work")
            if (state["columns_signature"] != before_state["columns_signature"]
                    or state["law_signature"] != before_state["law_signature"]
                    or state["area_m2"] != before_state["area_m2"]
                    or state["step_count"] != before_state["step_count"]+1):
                raise ValueError("An original physical point history was replaced")
            error = computed_close(number(area, "Original area")*number(state["traction_normal_pa"], "Original normal traction"),
                reaction, "Original reaction", "5e-12")
            work_error = close(result["original_point_steps"][index]["external_work_j"], expected_work,
                "Original local work", tolerance="5e-10")
            old_reaction_errors.append(dict(point_id=inputs["old_point_ids"][index],
                reaction_relative_error=str(error), work_relative_error=str(work_error)))
        active = []
        for index, item in enumerate(contacts):
            normal = [number(value, "Born normal operator") for value in item["normal_operator"]]
            gap = number(item["state_before"]["gap_free_m"], "Born previous gap")+dot(normal, delta)
            if gap < 0:
                active.append(index)
            parameters, reference = item["parameters"], item["reference"]
            local = normal_response(parameters["columns"], parameters["law"], reference["interface_area_m2"], float(gap))
            coefficient = number(reference["interface_area_m2"], "Inclined reference area", positive=True)*local["effective_stiffness_pa_m"]
            reaction = coefficient*min(gap, ZERO)
            all_contact_forces.append([reaction*value for value in normal])
            expected_work = reaction*dot(normal, delta)
            born_work += expected_work
            before_receipt = audit_contact_state(item["state_before"])
            after_state = result["born_contact_states"][index]
            after_receipt = audit_contact_state(after_state)
            if after_state["parameters"] != parameters or after_state["reference"] != reference:
                raise ValueError("A born contact changed its independently supplied physics")
            actual_reaction = (number(reference["interface_area_m2"], "Inclined reference area")
                *number(after_state["normal_traction_pa"], "Born normal traction"))
            actual_contact_forces.append([actual_reaction*value for value in normal])
            actual_born_work += number(result["born_contact_steps"][index]["external_work_j"], "Actual born work")
            error = computed_close(actual_reaction, reaction, "Born reaction", "5e-12")
            work_error = close(result["born_contact_steps"][index]["external_work_j"], expected_work,
                "Born local work", tolerance="5e-10")
            born_reaction_errors.append(dict(family_id=reference["family_id"],
                reaction_relative_error=str(error), work_relative_error=str(work_error),
                before=before_receipt, after=after_receipt))
        if active != result["active_born_contacts"]:
            raise ValueError("The independently derived unilateral active set differs")
        residuals, actual_residuals = [], []
        for index in range(dimension):
            contact_forces = [row[index] for row in all_contact_forces]
            operands = [basal_force[index], cold_force[index], foundation_force[index], source[index], *contact_forces]
            residual = basal_force[index]+cold_force[index]+foundation_force[index]-source[index]+sum(contact_forces, ZERO)
            error = relative(residual, operands)
            if error > Decimal("1e-35"):
                raise ValueError("A complete independently contracted force coordinate did not balance")
            residuals.append(str(error))
            actual_forces = [row[index] for row in actual_contact_forces]
            actual_operands = [basal_force[index], cold_force[index], actual_foundation_force[index], source[index], *actual_forces]
            actual_residual = (basal_force[index]+cold_force[index]+actual_foundation_force[index]-source[index]
                +sum(actual_forces, ZERO))
            actual_error = relative(actual_residual, actual_operands)
            if actual_error > Decimal("2e-10"):
                raise ValueError("An independently contracted stored physical reaction coordinate did not balance")
            actual_residuals.append(str(actual_error))
        source_work = dot(source, delta)
        basal_work = dot(delta, basal_force)
        cold_increment = dot(delta[:prefix], multiply(cold_matrix, [a+b for a, b in zip(new[:prefix], old[:prefix])]))/2
        cold_remainder = dot(delta[:prefix], multiply(cold_matrix, delta[:prefix]))/2
        work_residual = source_work-basal_work-cold_increment-cold_remainder-old_work-born_work-foundation_work
        work_error = relative(work_residual, (source_work, basal_work, cold_increment,
            cold_remainder, old_work, born_work, foundation_work))
        if work_error > Decimal("1e-35") or basal_work < 0 or cold_remainder < 0:
            raise ValueError("The independently contracted complete work balance failed")
        actual_work_residual = (source_work-basal_work-cold_increment-cold_remainder
            -actual_old_work-actual_born_work-actual_foundation_work)
        actual_work_error = relative(actual_work_residual, (source_work, basal_work, cold_increment,
            cold_remainder, actual_old_work, actual_born_work, actual_foundation_work))
        if actual_work_error > Decimal("2e-10"):
            raise ValueError("The independently contracted actual stored local work balance failed")
        diagnostic_expected = dict(source_work_j=source_work, basal_dissipation_j=basal_work,
            cold_bending_energy_increment_j=cold_increment, cold_bending_numerical_dissipation_j=cold_remainder,
            original_contact_work_j=old_work, born_contact_work_j=born_work, foundation_work_j=foundation_work)
        diagnostic_errors = {name: str(close(result["diagnostics"][name], value, name))
            for name, value in diagnostic_expected.items()}
    return dict(format="native-birth-independent-complete-network-audit-1", status="PASS",
        verified_utc=datetime.now(timezone.utc).isoformat(), seconds=time.perf_counter()-started,
        input_packet=dict(path=str(Path(path).resolve().relative_to(ROOT)), sha256=digest(path)),
        auditor_source=dict(path=str(Path(__file__).resolve().relative_to(ROOT)), sha256=digest(__file__)),
        independent_energy_oracle=dict(path="analysis/geometric_native_birth_contact_validation/independent_birth_energy_audit.py",
            sha256=digest(ROOT / "analysis/geometric_native_birth_contact_validation/independent_birth_energy_audit.py")),
        decimal_precision=200, production_mechanics_imported=False,
        complete_coordinate_count=dimension, unchanged_original_point_count=len(old_reaction_errors),
        born_contact_count=len(contacts), unique_foundation_count=len(foundations),
        authoritative_decimal_coordinates_used=True, binary64_coordinates_used_as_force_authority=False,
        per_coordinate_relative_error=residuals, maximum_relative_error=str(max(map(Decimal, residuals), default=ZERO)),
        actual_per_coordinate_relative_error=actual_residuals,
        maximum_actual_relative_error=str(max(map(Decimal, actual_residuals), default=ZERO)),
        complete_work_balance_relative_error=str(work_error), diagnostic_relative_errors=diagnostic_errors,
        actual_stored_local_work_balance_relative_error=str(actual_work_error),
        actual_stored_normal_and_shear_reactions_contracted=True,
        original_reaction_and_work_checks=old_reaction_errors,
        born_reaction_and_work_checks=born_reaction_errors, foundation_checks=foundation_rows,
        force_equations_resolved_here=False, native_source_authenticated_here=False,
        temporal_contact_coverage_proved_here=False, whole_native_interval_admitted=False,
        scope="Independent contraction of all declared coordinates, contacts, one foundation per patch and energy journals; no native geometry/admission claim")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    options = parser.parse_args()
    receipt = write(options.out, audit_network(options.input), "receipt_sha256")
    print(encoded({key: receipt[key] for key in ("status", "seconds", "complete_coordinate_count",
        "maximum_relative_error", "complete_work_balance_relative_error")}).decode())
