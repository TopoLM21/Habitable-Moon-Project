"""Stdlib-only oracle for the explicitly declared unbonded newborn law.

No production mechanics module is imported.  Compliance and energy are derived
from the supplied column/contact and hydrostatic parameters at 200 decimals;
the tiny-area cases retain their own scale instead of a unit-energy floor.
"""
from __future__ import annotations

from decimal import Decimal, localcontext
from datetime import datetime, timezone
import argparse
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis.geometric_native_birth_contact_validation.verify_baseline import (
    digest, encoded, read, value_sha, write,
)


ZERO = Decimal(0)
HALF = Decimal(1) / 2


def number(value, name, *, positive=False, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(name + " must be a literal finite number")
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(name + " must be finite")
        result = Decimal.from_float(value)
    else:
        result = Decimal(value)
    if positive and result <= 0 or nonnegative and result < 0:
        raise ValueError(name + " has an invalid sign")
    return result


def close(observed, expected, name, *, operands=(), tolerance="2e-12"):
    actual = number(observed, name)
    scale = max(abs(actual), abs(expected), *(abs(value) for value in operands))
    error = abs(actual-expected)
    if scale == 0:
        relative = ZERO if error == 0 else Decimal("Infinity")
    else:
        relative = error/scale
    if relative > Decimal(tolerance):
        raise ValueError(name + " differs from the independent constitutive value")
    return relative


def normal_response(columns, law, area_m2, gap_free_m):
    with localcontext() as context:
        context.prec = 200
        sub_length = number(columns["sub_length_m"], "sub_length_m", positive=True)
        over_length = number(columns["over_length_m"], "over_length_m", positive=True)
        sub_modulus = number(columns["sub_normal_modulus_pa"], "sub_normal_modulus_pa", positive=True)
        over_modulus = number(columns["over_normal_modulus_pa"], "over_normal_modulus_pa", positive=True)
        interface_stiffness = number(law["normal_stiffness_pa_m"], "normal_stiffness_pa_m", positive=True)
        area = number(area_m2, "area_m2", positive=True)
        free_gap = number(gap_free_m, "gap_free_m")
        column_compliance = sub_length/sub_modulus + over_length/over_modulus
        effective = 1/(column_compliance + 1/interface_stiffness)
        traction = effective*min(free_gap, ZERO)
        gap = traction/interface_stiffness if free_gap < 0 else free_gap
        bulk_energy = HALF*area*traction*traction*column_compliance
        contact_energy = HALF*area*traction*traction/interface_stiffness
        return dict(effective_stiffness_pa_m=effective, traction_normal_pa=traction,
            gap_m=gap, bulk_energy_j=bulk_energy, contact_energy_j=contact_energy,
            recoverable_energy_j=bulk_energy+contact_energy,
            integrated_reaction_n=area*traction)


def normal_step(columns, law, area_m2, old_gap_free_m, gap_free_m):
    with localcontext() as context:
        context.prec = 200
        old = normal_response(columns, law, area_m2, old_gap_free_m)
        new = normal_response(columns, law, area_m2, gap_free_m)
        increment = number(gap_free_m, "gap_free_m")-number(old_gap_free_m, "old_gap_free_m")
        work = new["integrated_reaction_n"]*increment
        energy_increment = new["recoverable_energy_j"]-old["recoverable_energy_j"]
        remainder = work-energy_increment
        if remainder < 0:
            raise ValueError("The unilateral normal law lost convexity")
        return dict(new, external_work_j=work, recoverable_energy_increment_j=energy_increment,
            backward_euler_remainder_j=remainder)


def foundation_response(parameters, area_m2, w_birth_m, deflection_m):
    with localcontext() as context:
        context.prec = 200
        density = number(parameters["density_kg_m3"], "density_kg_m3", positive=True)
        gravity = number(parameters["gravity_m_s2"], "gravity_m_s2", positive=True)
        area = number(area_m2, "basal_area_m2", positive=True)
        delta = number(deflection_m, "deflection_m")-number(w_birth_m, "w_birth_m")
        stiffness = density*gravity
        reaction = area*stiffness*delta
        energy = HALF*reaction*delta
        return dict(stiffness_pa_m=stiffness, traction_pa=stiffness*delta,
            integrated_reaction_n=reaction, recoverable_energy_j=energy)


def foundation_step(parameters, area_m2, w_birth_m, old_deflection_m, deflection_m):
    with localcontext() as context:
        context.prec = 200
        old = foundation_response(parameters, area_m2, w_birth_m, old_deflection_m)
        new = foundation_response(parameters, area_m2, w_birth_m, deflection_m)
        increment = number(deflection_m, "deflection_m")-number(old_deflection_m, "old_deflection_m")
        work = new["integrated_reaction_n"]*increment
        energy_increment = new["recoverable_energy_j"]-old["recoverable_energy_j"]
        remainder = work-energy_increment
        if remainder < 0:
            raise ValueError("The hydrostatic foundation lost convexity")
        return dict(new, external_work_j=work, recoverable_energy_increment_j=energy_increment,
            backward_euler_remainder_j=remainder)


def check_state_seal(state):
    body = {key: value for key, value in state.items() if key != "state_sha256"}
    if state.get("state_sha256") != value_sha(body):
        raise ValueError("A local state selfseal differs")


def audit_contact_state(state):
    check_state_seal(state)
    parameters, reference = state["parameters"], state["reference"]
    columns, law, area = parameters["columns"], parameters["law"], reference["interface_area_m2"]
    if any(number(law[name], name) != 0 for name in
            ("friction_dry", "friction_wet", "cohesion_pa", "viscosity_pa_s_m")):
        raise ValueError("The newborn declaration is not frictionless and unbonded")
    gap, jump = reference["birth_gap_free_m"], reference["birth_jump_free_m"]
    activation = normal_response(columns, law, area, gap)["recoverable_energy_j"]
    work = dissipation = elapsed = slip = ZERO
    with localcontext() as context:
        context.prec = 200
        for row in state["history"]:
            result = normal_step(columns, law, area, gap, row["gap_free_m"])
            work += result["external_work_j"]
            dissipation += result["backward_euler_remainder_j"]
            elapsed += number(row["dt_s"], "Contact duration", positive=True)
            slip += abs(number(row["jump_free_m"], "Contact jump")-number(jump, "Prior contact jump"))
            gap, jump = row["gap_free_m"], row["jump_free_m"]
        final = normal_response(columns, law, area, gap)
        expected = dict(gap_free_m=number(gap, "Contact gap"),
            jump_free_m=number(jump, "Contact jump"),
            normal_traction_pa=final["traction_normal_pa"],
            plastic_slip_m=number(jump, "Contact jump")-number(reference["birth_jump_free_m"], "Birth jump"),
            cumulative_slip_m=slip, normal_contact_energy_j=final["recoverable_energy_j"],
            activation_work_j=activation, cumulative_external_work_j=work,
            cumulative_numerical_dissipation_j=dissipation, elapsed_time_s=elapsed)
        errors = {name: close(state[name], value, name) for name, value in expected.items()}
        if type(state["step_count"]) is not int or state["step_count"] != len(state["history"]):
            raise ValueError("Contact step count differs from the independent journal")
        balance = activation+work-final["recoverable_energy_j"]-dissipation
        scale = max(abs(activation), abs(work), abs(final["recoverable_energy_j"]), abs(dissipation))
        if balance != 0 and (scale == 0 or abs(balance)/scale > Decimal("1e-190")):
            raise ValueError("Independent normal energy journal did not close")
    return dict(family_id=reference["family_id"], state_sha256=state["state_sha256"],
        steps=len(state["history"]), negative_birth_gap=reference["birth_gap_free_m"] < 0,
        area_m2=area, activation_energy_rederived=True,
        max_relative_error=str(max(errors.values(), default=ZERO)),
        ideal_series_compliance_rederived=True, zero_shear_work=True,
        source_authentication_checked_here=False)


def audit_foundation_state(state):
    check_state_seal(state)
    parameters, reference = state["parameters"], state["reference"]
    area, birth = reference["basal_area_m2"], reference["radial_birth_m"]
    radial = birth
    work = dissipation = elapsed = ZERO
    with localcontext() as context:
        context.prec = 200
        for row in state["history"]:
            result = foundation_step(parameters, area, birth, radial, row["radial_displacement_m"])
            work += result["external_work_j"]
            dissipation += result["backward_euler_remainder_j"]
            elapsed += number(row["dt_s"], "Foundation duration", positive=True)
            radial = row["radial_displacement_m"]
        final = foundation_response(parameters, area, birth, radial)
        expected = dict(radial_displacement_m=number(radial, "Foundation displacement"),
            foundation_energy_j=final["recoverable_energy_j"], cumulative_external_work_j=work,
            cumulative_numerical_dissipation_j=dissipation, elapsed_time_s=elapsed)
        errors = {name: close(state[name], value, name) for name, value in expected.items()}
        if type(state["step_count"]) is not int or state["step_count"] != len(state["history"]):
            raise ValueError("Foundation step count differs from the independent journal")
        balance = work-final["recoverable_energy_j"]-dissipation
        scale = max(abs(work), abs(final["recoverable_energy_j"]), abs(dissipation))
        if balance != 0 and (scale == 0 or abs(balance)/scale > Decimal("1e-190")):
            raise ValueError("Independent hydrostatic energy journal did not close")
    return dict(patch_id=reference["patch_id"], state_sha256=state["state_sha256"],
        steps=len(state["history"]), area_m2=area, rho_g_rederived=True,
        max_relative_error=str(max(errors.values(), default=ZERO)),
        source_authentication_checked_here=False)


def audit_packet(path):
    started = time.perf_counter()
    packet = read(path)
    expected = packet.pop("packet_sha256")
    if expected != value_sha(packet):
        raise ValueError("The constitutive control packet selfseal differs")
    cases = []
    for case in packet["cases"]:
        contacts = [audit_contact_state(state) for state in case["contact_states"]]
        foundations = [audit_foundation_state(state) for state in case["foundation_states"]]
        families = [row["family_id"] for row in contacts]
        patches = [row["patch_id"] for row in foundations]
        if len(set(families)) != len(families) or len(set(patches)) != len(patches):
            raise ValueError("A physical contact or radial patch was counted twice in one case")
        cases.append(dict(label=case["label"], contact_states=contacts,
            foundation_states=foundations, unique_patch_accounting=True))
    if not cases or not any(row["contact_states"] for row in cases) or not any(row["foundation_states"] for row in cases):
        raise ValueError("Both newborn contact and foundation journals are required")
    return dict(format="native-birth-independent-constitutive-energy-audit-1", status="PASS",
        verified_utc=datetime.now(timezone.utc).isoformat(), seconds=time.perf_counter()-started,
        input_packet=dict(path=str(Path(path).resolve().relative_to(ROOT)), sha256=digest(path)),
        auditor_source=dict(path=str(Path(__file__).resolve().relative_to(ROOT)), sha256=digest(__file__)),
        decimal_precision=200, production_mechanics_imported=False,
        cases=cases, case_count=len(cases),
        force_equilibrium_resolved_here=False, temporal_contact_coverage_proved_here=False,
        whole_native_interval_admitted=False,
        scope="Independent series-column unilateral normal response, paid activation, virgin shear and rho*g foundation energy journals; no source/geometry/admission claim")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    options = parser.parse_args()
    receipt = write(options.out, audit_packet(options.input), "receipt_sha256")
    print(encoded({key: receipt[key] for key in ("status", "seconds", "case_count", "decimal_precision")}).decode())
