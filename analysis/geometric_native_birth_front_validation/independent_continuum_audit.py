"""Stdlib-only contraction of the full-arc affine force and energy packet.

The independent angular oracle multiplies Taylor coefficient polynomials and
integrates their powers. It does not use the production trigonometric moment
formula, root solver, tangent, mechanics implementation or binary64 velocity.
It verifies supplied equilibrium coordinates rather than solving them again.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal, localcontext
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis.geometric_native_birth_front_validation.verify_baseline import digest, read, write, encoded
from analysis.geometric_native_birth_contact_validation.independent_birth_energy_audit import (
    number, normal_response, audit_foundation_state)

ZERO = Decimal(0)
PRECISION = 240
DEGREE = 160


def decimal(value):
    if type(value) is Decimal:
        if not value.is_finite():
            raise ValueError("An independently computed decimal must be finite")
        return value
    if type(value) is str:
        result = Decimal(value)
        if not result.is_finite() or str(result) != value:
            raise ValueError("A continuum audit requires canonical finite Decimal strings")
        return result
    return number(value, "Physical literal")


def dot(left, right):
    if len(left) != len(right):
        raise ValueError("An independent contraction lost a coordinate")
    return sum((a*b for a, b in zip(left, right)), ZERO)


def matvec(matrix, vector):
    return [dot(row, vector) for row in matrix]


def relative(error, values):
    scale = sum((abs(value) for value in values), ZERO)
    if scale == 0:
        if error != 0:
            raise ValueError("A nonzero discrepancy has no physical scale")
        return ZERO
    return abs(error)/scale


def check(observed, expected, label, *, tolerance="1e-65", allowance=ZERO):
    actual = decimal(observed)
    scale = abs(actual)+abs(expected)
    if abs(actual-expected) > allowance+Decimal(tolerance)*scale:
        raise ValueError(label+" differs from its independent value")
    return relative(actual-expected, (actual, expected))


def series(coefficients):
    """Power coefficients of a+b*cos(theta)+c*sin(theta)."""
    a, b, c = coefficients
    result = [a+b]+[ZERO]*DEGREE
    factorial = 1
    for power in range(1, DEGREE+1):
        factorial *= power
        if power % 2:
            result[power] = c*Decimal(-1 if (power//2)%2 else 1)/Decimal(factorial)
        else:
            result[power] = b*Decimal(-1 if (power//2)%2 else 1)/Decimal(factorial)
    return result


def evaluate(coefficients, angle):
    result = ZERO
    for coefficient in reversed(series(coefficients)):
        result = result*angle+coefficient
    return result


def integrate_product(first, second, intervals):
    """Independent Taylor-power integral; supports declared angles <=1/8."""
    p, q = series(first), series(second)
    product = [ZERO]*(len(p)+len(q)-1)
    for i, a in enumerate(p):
        if a:
            for j, b in enumerate(q):
                if b:
                    product[i+j] += a*b
    total = ZERO
    for low, high in intervals:
        if not ZERO <= low <= high <= Decimal("0.125"):
            raise ValueError("The independent power-integral oracle requires its explicit minor-angle scope")
        low_power, high_power = low, high
        for power, coefficient in enumerate(product, 1):
            total += coefficient*(high_power-low_power)/Decimal(power)
            low_power *= low
            high_power *= high
    return total


def classify_monotone_field(gap, low, high, bands, parts):
    """Prove all zeros/closed pieces for a strictly monotone native field."""
    if not ZERO <= low < high <= Decimal("0.125"):
        raise ValueError("The native independent classification scope changed")
    # On [0,1/8], cos is positive and sin is nonnegative. Exact Taylor
    # endpoint values with the derivative's elementary interval majorant
    # establish one fixed derivative sign, including the native tiny arcs.
    b, c = gap[1:]
    cos_min = evaluate((ZERO, Decimal(1), ZERO), high)
    sin_max = evaluate((ZERO, ZERO, Decimal(1)), high)
    derivative_low = min(c, c*cos_min)-abs(b)*sin_max
    derivative_high = max(c, c*cos_min)+abs(b)*sin_max
    if not (derivative_high < 0 or derivative_low > 0):
        raise ValueError("This audit refuses a field without a proved monotone whole arc")
    left, right = evaluate(gap, low), evaluate(gap, high)
    roots = []
    if left*right < 0:
        if len(bands) != 1:
            raise ValueError("A monotone sign-changing field requires exactly one explicit root band")
        a, b = bands[0]
        if not low <= a <= b <= high or evaluate(gap, a)*evaluate(gap, b) > 0:
            raise ValueError("An exported root band does not contain the native field zero")
        roots.append((a+b)/2)
    elif bands:
        raise ValueError("A root band was invented on a monotone arc without a sign change")
    cuts = [low, *roots, high]
    expected = [(a,b) for a,b in zip(cuts,cuts[1:]) if a<b and evaluate(gap,(a+b)/2)<0]
    def matching_endpoint(actual, target):
        return actual == target or any(a <= actual <= b and a <= target <= b for a,b in bands)
    if len(parts)!=len(expected) or any(not matching_endpoint(a,c) or not matching_endpoint(b,d)
            for (a,b),(c,d) in zip(parts,expected)):
        raise ValueError("The exported closed subarcs lost a positive or negative material interval")
    return dict(derivative_lower=str(derivative_low), derivative_upper=str(derivative_high),
        left_gap=str(left), right_gap=str(right), root_count=len(roots),
        certified_monotone=True, positive_area_tolerance_used=False)


def compliance(parameters):
    return normal_response(parameters["columns"], parameters["law"], 1., 0.)["effective_stiffness_pa_m"]


def audit_continuum(path):
    started = time.perf_counter()
    source = read(path)
    packet = source.get("actual_network", source)
    if packet["force_geometry_admission_authorized"] is not False:
        raise ValueError("An affine force packet cannot claim native material admission")
    if "actual_network" in source and (source["material_admission_authorized"] is not False
            or source["born_finite_pose_branch_admission_proved"] is not False
            or source["accepted_new_native_intervals"] != 0):
        raise ValueError("The genuine source candidate's material scope changed")
    inputs, result = packet["inputs"], packet["result"]
    with localcontext() as context:
        context.prec = PRECISION
        new = list(map(decimal,result["coordinates_decimal"]))
        old = list(map(decimal,inputs["coordinates_before"]))
        dimension, prefix = len(new), len(inputs["old_stiffness"])
        if dimension != prefix+len(packet["foundations"]) or len(old)!=dimension:
            raise ValueError("The independent complete vector dimension changed")
        delta = [a-b for a,b in zip(new,old)]
        radius, duration = decimal(inputs["radius_m"]), decimal(inputs["dt_s"])
        matrix = [list(map(decimal,row)) for row in inputs["old_stiffness"]]
        drag = [[decimal(v)/duration for v in row] for row in inputs["drag_nm_s"]]
        source_force = list(map(decimal,inputs["source_torque_nm"]))
        drag_force = matvec(drag,delta)
        cold_force = matvec(matrix,new[:prefix])+[ZERO]*(dimension-prefix)
        foundation_force, actual_foundation_force = [ZERO]*dimension,[ZERO]*dimension
        foundation_work=actual_foundation_work=ZERO
        foundation_rows=[]
        patches=[]
        for index,item in enumerate(packet["foundations"]):
            before=item["state_before"]
            after=result["foundation_states"][index]
            ref,params=before["reference"],before["parameters"]
            if after["reference"]!=ref or after["parameters"]!=params:
                raise ValueError("An independent foundation changed source or parameters")
            patches.append(ref["patch_id"])
            dof=item["radial_dof_index"]
            coefficient=decimal(params["density_kg_m3"])*decimal(params["gravity_m_s2"])*decimal(ref["basal_area_m2"])
            birth=decimal(ref["radial_birth_m"])
            w,wp=new[dof]*radius,old[dof]*radius
            reaction=coefficient*(w-birth)
            foundation_force[dof]+=reaction*radius
            expected_work=reaction*(w-wp)
            foundation_work+=expected_work
            actualw=decimal(after["radial_displacement_m"])
            actualwp=decimal(before["radial_displacement_m"])
            actual_foundation_force[dof]+=coefficient*(actualw-birth)*radius
            local=result["foundation_steps"][index]
            actualwork=decimal(local["external_work_j"])
            actual_foundation_work+=actualwork
            e,ep=abs(actualw-w),abs(actualwp-wp)
            alias=coefficient*(e*abs(w-wp)+(abs(w-birth)+e)*(e+ep))
            energy_alias=coefficient*(abs(w-birth)*e+e*e/2)
            remainder_alias=coefficient*(abs(w-wp)*(e+ep)+(e+ep)**2/2)
            check(local["external_work_j"],expected_work,"Foundation work",tolerance="5e-12",allowance=alias)
            check(after["foundation_energy_j"],coefficient*(w-birth)**2/2,"Foundation energy",tolerance="5e-12",allowance=energy_alias)
            check(local["numerical_dissipation_j"],coefficient*(w-wp)**2/2,"Foundation BE",tolerance="5e-12",allowance=remainder_alias)
            foundation_rows.append(dict(patch_id=ref["patch_id"],dof=dof,
                independently_computed_work_alias_bound_j=str(alias),
                before=audit_foundation_state(before),after=audit_foundation_state(after)))
        if len(set(patches))!=len(patches):
            raise ValueError("A physical foundation patch was counted twice")
        forces,actual_forces=[],[]
        cold_work=actual_cold_work=ZERO
        original_rows=[]
        for index,(material,area,operator,before) in enumerate(zip(packet["original_materials"],
                inputs["old_areas_m2"],inputs["old_anchor_operators"],packet["original_states_before"])):
            normal=list(map(decimal,operator[0]))+[ZERO]*(dimension-prefix)
            shear=list(map(decimal,operator[1]))+[ZERO]*(dimension-prefix)
            gap=decimal(before["gap_free_m"])+dot(normal,delta)
            if gap>0:
                raise ValueError("An unchanged original contact opened outside its constitutive model")
            coefficient=decimal(area)*compliance(material)
            reaction=coefficient*gap
            forces.append([reaction*v for v in normal])
            expected_work=reaction*dot(normal,delta)
            cold_work+=expected_work
            state=result["original_point_states"][index][1]
            normal_actual=decimal(area)*decimal(state["traction_normal_pa"])
            shear_actual=decimal(area)*decimal(state["traction_shear_pa"])
            actual_forces.append([normal_actual*n+shear_actual*s for n,s in zip(normal,shear)])
            actual_work=decimal(result["original_point_steps"][index]["external_work_j"])
            actual_cold_work+=actual_work
            reaction_error=check(normal_actual,reaction,"Original actual reaction",tolerance="5e-12")
            work_error=check(actual_work,expected_work,"Original actual work",tolerance="5e-10")
            if (state["columns_signature"]!=before["columns_signature"]
                    or state["law_signature"]!=before["law_signature"]
                    or state["area_m2"]!=before["area_m2"]
                    or state["step_count"]!=before["step_count"]+1
                    or decimal(state["elapsed_time_s"])!=decimal(before["elapsed_time_s"])+duration):
                raise ValueError("An original mechanical point lost its complete physical history")
            original_rows.append(dict(point_id=inputs["old_point_ids"][index],
                actual_reaction_relative_error=str(reaction_error),actual_work_relative_error=str(work_error)))
        arc_rows=[]
        born_work=activation=ZERO
        families=[]
        for attachment,local in zip(packet["continuum_attachments"],result["continuum_steps"]):
            declaration=attachment["declaration"]
            before=attachment["previous_state"]
            state=local["state"]
            if before["declaration"]!=declaration or state["declaration"]!=declaration:
                raise ValueError("A complete continuum source declaration changed")
            families.append(declaration["reference"]["family_id"])
            low,high=map(decimal,(declaration["angle_start_decimal"],declaration["angle_end_decimal"]))
            operators=[list(map(decimal,row)) for row in declaration["normal_operator_coefficients"]]
            shears=[list(map(decimal,row)) for row in declaration["shear_operator_coefficients"]]
            oldgap=list(map(decimal,before["gap_coefficients_decimal"]))
            oldjump=list(map(decimal,before["jump_coefficients_decimal"]))
            gap=[value+dot(row,delta) for value,row in zip(oldgap,operators)]
            jump=[value+dot(row,delta) for value,row in zip(oldjump,shears)]
            for a,b in zip(state["gap_coefficients_decimal"],gap):
                check(a,b,"Continuum full normal coefficient")
            for a,b in zip(state["jump_coefficients_decimal"],jump):
                check(a,b,"Continuum full shear coefficient")
            parts=[tuple(map(decimal,row)) for row in local["closed_angle_parts_decimal"]]
            bands=[tuple(map(decimal,row)) for row in local["root_bands_decimal"]]
            classification=classify_monotone_field(gap,low,high,bands,parts)
            factor=decimal(declaration["measure_density_decimal"])*compliance(declaration["parameters"])
            energy=factor*integrate_product(gap,gap,parts)/2
            work=factor*integrate_product(gap,[a-b for a,b in zip(gap,oldgap)],parts)
            oldenergy=decimal(before["stored_energy_decimal"])
            remainder=work-energy+oldenergy
            check(state["stored_energy_decimal"],energy,"Whole-arc normal energy",tolerance="1e-60")
            check(local["external_work_decimal"],work,"Whole-arc end-reaction work",tolerance="1e-60")
            check(local["stored_energy_increment_decimal"],energy-oldenergy,"Whole-arc energy increment")
            check(local["numerical_dissipation_decimal"],remainder,"Whole-arc BE remainder")
            if remainder<0 or energy<0:
                raise ValueError("The independent continuum ledger lost convexity")
            born_work+=work
            activation+=decimal(before["activation_energy_decimal"])
            birth=decimal(declaration["reference"]["birth_gap_free_m"])
            birth_energy=factor*(high-low)*min(birth,ZERO)**2/2
            check(before["activation_energy_decimal"],birth_energy,"Paid full-arc activation")
            gradient=[]
            for i in range(dimension):
                basis=[row[i] for row in operators]
                gradient.append(factor*integrate_product(gap,basis,parts))
                check(local["gradient_decimal"][i],gradient[-1],"Integrated continuum force",tolerance="1e-60")
            for i in range(dimension):
                for j in range(dimension):
                    expected=factor*integrate_product([row[i] for row in operators],
                        [row[j] for row in operators],parts)
                    check(local["hessian_decimal"][i][j],expected,"Integrated continuum tangent",tolerance="1e-60")
            forces.append(gradient)
            actual_forces.append(list(map(decimal,local["gradient_decimal"])))
            if state["history"]!=before["history"]+[dict(gap_coefficients_decimal=state["gap_coefficients_decimal"],
                    jump_coefficients_decimal=state["jump_coefficients_decimal"],dt_s=inputs["dt_s"])]:
                raise ValueError("A whole-arc local journal lost a constitutive increment")
            if state["step_count"]!=before["step_count"]+1 or decimal(state["elapsed_time_s"])!=decimal(before["elapsed_time_s"])+duration:
                raise ValueError("A full-arc state lost its actual constitutive clock")
            # Root-band proof is genuinely retained. Its force/energy bounds
            # are computed on the family's own scale, with no unit floor.
            # The bracket belongs to the declared stored Decimal field. Its
            # independent coordinate contraction above can have additional
            # guard digits; those do not replace the bracket's exact input.
            stored_gap=list(map(decimal,state["gap_coefficients_decimal"]))
            derivative_bound=abs(stored_gap[1])+abs(stored_gap[2])
            energy_bound=ZERO
            for a,b in bands:
                width=b-a
                g_bound=max(abs(evaluate(stored_gap,a)),abs(evaluate(stored_gap,b)))+derivative_bound*width
                energy_bound+=factor*width*g_bound*g_bound/2
            exported_bound=decimal(local["constitutive_root_band_uncertainty"])
            if exported_bound+Decimal("1e-60")*energy_bound<energy_bound:
                raise ValueError("A full-arc constitutive root band lost its physical energy bound")
            arc_rows.append(dict(family_id=families[-1],classification=classification,
                actual_angular_measure=str(high-low),closed_angular_measure=str(sum((b-a for a,b in parts),ZERO)),
                activation_energy_j=str(birth_energy),stored_energy_j=str(energy),work_j=str(work),
                backward_euler_remainder_j=str(remainder),root_band_energy_bound_j=str(energy_bound),
                no_finite_measure_assigned_to_root=True,no_point_material_weights=True))
        if len(set(families))!=len(families):
            raise ValueError("A continuum interface family was duplicated")
        residuals,actual_residuals=[],[]
        for i in range(dimension):
            localforces=[row[i] for row in forces]
            values=[drag_force[i],cold_force[i],foundation_force[i],source_force[i],*localforces]
            residual=drag_force[i]+cold_force[i]+foundation_force[i]-source_force[i]+sum(localforces,ZERO)
            error=relative(residual,values)
            if error>Decimal("1e-35"):
                raise ValueError("An independently integrated complete continuum force coordinate failed")
            residuals.append(str(error))
            actual=[row[i] for row in actual_forces]
            actualerror=relative(drag_force[i]+cold_force[i]+actual_foundation_force[i]-source_force[i]+sum(actual,ZERO),
                [drag_force[i],cold_force[i],actual_foundation_force[i],source_force[i],*actual])
            if actualerror>Decimal("5e-10"):
                raise ValueError("An actual stored original/foundation/arc reaction coordinate failed")
            actual_residuals.append(str(actualerror))
        source_work=dot(source_force,delta)
        basal_work=dot(delta,drag_force)
        cold_increment=dot(delta[:prefix],matvec(matrix,[a+b for a,b in zip(new[:prefix],old[:prefix])]))/2
        cold_remainder=dot(delta[:prefix],matvec(matrix,delta[:prefix]))/2
        work_error=relative(source_work-basal_work-cold_increment-cold_remainder-cold_work-born_work-foundation_work,
            (source_work,basal_work,cold_increment,cold_remainder,cold_work,born_work,foundation_work))
        actual_work_error=relative(source_work-basal_work-cold_increment-cold_remainder-actual_cold_work-born_work-actual_foundation_work,
            (source_work,basal_work,cold_increment,cold_remainder,actual_cold_work,born_work,actual_foundation_work))
        if work_error>Decimal("1e-35") or actual_work_error>Decimal("5e-10") or min(basal_work,cold_remainder)<0:
            raise ValueError("The independently integrated full source work balance failed")
    return dict(format="native-birth-independent-full-arc-network-audit-1",status="PASS",
        verified_utc=datetime.now(timezone.utc).isoformat(),seconds=time.perf_counter()-started,
        input_packet=dict(path=str(Path(path).resolve().relative_to(ROOT)),sha256=digest(path)),
        auditor_source=dict(path=str(Path(__file__).resolve().relative_to(ROOT)),sha256=digest(__file__)),
        decimal_precision=PRECISION,power_series_degree=DEGREE,
        independent_arc_integration="Taylor power products integrated term by term",
        production_mechanics_imported=False,production_trigonometric_moment_formula_used=False,
        complete_coordinate_count=dimension,original_point_count=len(original_rows),
        full_continuum_arc_count=len(arc_rows),foundation_patch_count=len(foundation_rows),born_midpoint_atoms=0,
        maximum_per_coordinate_relative_error=str(max(map(Decimal,residuals),default=ZERO)),
        maximum_actual_per_coordinate_relative_error=str(max(map(Decimal,actual_residuals),default=ZERO)),
        per_coordinate_relative_error=residuals,actual_per_coordinate_relative_error=actual_residuals,
        complete_work_balance_relative_error=str(work_error),actual_stored_work_balance_relative_error=str(actual_work_error),
        source_work_j=str(source_work),basal_work_j=str(basal_work),born_work_j=str(born_work),foundation_work_j=str(foundation_work),
        paid_activation_energy_j=str(activation),original_checks=original_rows,continuum_checks=arc_rows,foundation_checks=foundation_rows,
        force_equations_resolved_here=False,native_source_authenticated_here=False,
        finite_pose_branch_admission_proved=False,temporal_contact_coverage_proved_here=False,
        accepted_new_native_intervals=0,whole_point_1_completed=False)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    receipt=write(args.out,audit_continuum(args.input),"receipt_sha256")
    print(encoded({key:receipt[key] for key in ("status","seconds","maximum_per_coordinate_relative_error",
        "complete_work_balance_relative_error")}).decode())
