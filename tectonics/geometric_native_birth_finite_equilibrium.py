"""Actual finite-pose unilateral born forces with independent radial correction.

The original ports, material histories and literal angular path are retained.
Only the independently registered born radial coordinates are corrected, and
every mixed coordinate must pass its own actual force balance afterwards.
This force calculation does not admit material or establish solid embedding.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction as F
from math import factorial

from .geometric_native_pose import FractionInterval as I, _rounded_interval
from .geometric_native_birth_finite_pose import BornFinitePoseFields
from .geometric_native_birth_finite_force_bounds import (
    BornFinitePoseJacobianBounds, _normal_stiffness)


def _maximum(value):
    return max(abs(value.lo), abs(value.hi))


def _number(value):
    return F(Decimal(value)) if type(value) is str else F.from_float(float(value))


def _polynomial(coefficients, limit, order, precision):
    """Complete five-term field, with an outward rational Taylor tail."""
    polynomial = [I(0) for _ in range(order+1)]
    polynomial[0] = coefficients[0]
    tail = F(0)
    for coefficient, frequency, sine in ((coefficients[1], 1, False),
            (coefficients[2], 1, True), (coefficients[3], 2, False),
            (coefficients[4], 2, True)):
        for degree in range(int(sine), order+1, 2):
            sign = (-1)**((degree-int(sine))//2)
            polynomial[degree] += coefficient*F(sign*frequency**degree, factorial(degree))
        following = order+1
        if following % 2 != int(sine):
            following += 1
        # Alternating sine/cosine tails decrease throughout the declared
        # angular domain (2*limit <= 1/2). This is a true function remainder.
        tail += _maximum(coefficient)*(frequency*limit)**following/factorial(following)
    return tuple(_rounded_interval(value, precision) for value in polynomial), tail


def _evaluate(polynomial, angle, tail, precision):
    value = I(0)
    for coefficient in reversed(polynomial):
        value = _rounded_interval(value*angle+coefficient, precision)
    return _rounded_interval(value+I(-tail, tail), precision)


def _integral_product(left, left_tail, right, right_tail, low, high, precision):
    """Exact polynomial moments plus both complete function remainders."""
    coefficients = [I(0) for _ in range(len(left)+len(right)-1)]
    for i, a in enumerate(left):
        if a.lo == a.hi == 0:
            continue
        for j, b in enumerate(right):
            if b.lo != 0 or b.hi != 0:
                coefficients[i+j] = _rounded_interval(coefficients[i+j]+a*b, precision)
    value = sum((coefficient*((high**(degree+1)-low**(degree+1))/F(degree+1))
        for degree, coefficient in enumerate(coefficients)), I(0))
    angle = I(low, high)
    sup_left = _maximum(_evaluate(left, angle, F(0), precision))
    sup_right = _maximum(_evaluate(right, angle, F(0), precision))
    error = (high-low)*(left_tail*sup_right+right_tail*sup_left+left_tail*right_tail)
    return _rounded_interval(value+I(-error, error), precision)


@dataclass(frozen=True, slots=True)
class FiniteBornResponse:
    energy_bounds_j: I
    integrated_pressure_bounds_n: I
    gradient_bounds: tuple
    gauss_newton_hessian_bounds: tuple
    gauss_newton_hessian_coordinate_indices: tuple
    gap_bounds_m: I
    closed_angle_parts: tuple
    unresolved_angle_parts: tuple
    polynomial_order: int
    precision: int

    @property
    def material_admission_authorized(self):
        return False


def integrate_born_finite_pose_response(fields, jacobian, *,
        measure_density_m2_per_radian, angle_end_bounds,
        normal_stiffness_pa_m, precision=160, max_depth=48,
        polynomial_order=24, hessian_coordinate_indices=None):
    """Enclose the actual complete-arc potential, signed gradient and PSD part.

    Open cells contribute exactly zero. Closed cells use analytic rational
    polynomial moments. Every unresolved root cell contributes its entire
    clipped pressure, including an uncertain native endpoint. There is no
    pressure, force or area cutoff.
    """
    if type(fields) is not BornFinitePoseFields or type(jacobian) is not BornFinitePoseJacobianBounds:
        raise ValueError("Complete finite fields and their differentiated native frame are required")
    if (type(measure_density_m2_per_radian) is not F or measure_density_m2_per_radian <= 0
            or type(normal_stiffness_pa_m) is not F or normal_stiffness_pa_m <= 0):
        raise ValueError("The physical measure and series stiffness must be positive exact rationals")
    if type(angle_end_bounds) is not I or not 0 < angle_end_bounds.lo <= angle_end_bounds.hi <= F(1, 4):
        raise ValueError("Finite polynomial integration needs a positive minor arc at most 1/4 radian")
    if type(precision) is not int or not 64 <= precision <= 1024:
        raise ValueError("Finite integration precision must be an integer from 64 to 1024")
    if type(max_depth) is not int or not 0 <= max_depth <= 256:
        raise ValueError("The unresolved root budget must be a literal integer from 0 to 256")
    if type(polynomial_order) is not int or not 4 <= polynomial_order <= 48:
        raise ValueError("The complete-field polynomial order must be an integer from 4 to 48")
    if fields.time_fraction_bounds != jacobian.time_fraction_bounds:
        raise ValueError("Fields and derivatives must have the same actual pose clock")
    count = len(jacobian.coefficients[0])
    indices = tuple(range(count)) if hessian_coordinate_indices is None else hessian_coordinate_indices
    if (type(indices) is not tuple or len(set(indices)) != len(indices)
            or any(type(index) is not int or not 0 <= index < count for index in indices)):
        raise ValueError("Every requested Hessian coordinate must be uniquely registered")
    limit = angle_end_bounds.hi
    gap, gap_tail = _polynomial(fields.gap_coefficients, limit, polynomial_order, precision)
    columns = tuple(_polynomial(tuple(row[index] for row in jacobian.coefficients),
        limit, polynomial_order, precision) for index in range(count))
    derivative = tuple((degree+1)*gap[degree+1] for degree in range(len(gap)-1))
    # Differentiating the true trig field, rather than a tail bound, gives
    # monotonicity tests with a fresh analytic derivative remainder.
    derivative_coefficients = (I(0), fields.gap_coefficients[2], -fields.gap_coefficients[1],
        2*fields.gap_coefficients[4], -2*fields.gap_coefficients[3])
    derivative, derivative_tail = _polynomial(derivative_coefficients, limit, polynomial_order, precision)
    factor = measure_density_m2_per_radian*normal_stiffness_pa_m
    energy, pressure = I(0), I(0)
    gradient = [I(0) for _ in range(count)]
    hessian = [[I(0) for _ in indices] for _ in indices]
    closed, unknown = [], []
    pending = [(F(0), angle_end_bounds.lo, 0, False)]
    if angle_end_bounds.hi > angle_end_bounds.lo:
        pending.append((angle_end_bounds.lo, angle_end_bounds.hi, max_depth, True))
    while pending:
        low, high, depth, uncertain_measure = pending.pop()
        angle = I(low, high)
        bounds = _evaluate(gap, angle, gap_tail, precision)
        derivative_bounds = _evaluate(derivative, angle, derivative_tail, precision)
        if derivative_bounds.lo >= 0 or derivative_bounds.hi <= 0:
            left, right = (_evaluate(gap, I(point), gap_tail, precision) for point in (low, high))
            bounds = I(min(left.lo, right.lo), max(left.hi, right.hi))
        if bounds.lo >= 0:
            continue
        if bounds.hi <= 0 and not uncertain_measure:
            closed.append((low, high))
            local_energy = factor*_integral_product(gap, gap_tail, gap, gap_tail, low, high, precision)/2
            local_pressure = factor*_integral_product(gap, gap_tail, (I(1),), F(0), low, high, precision)
            local_gradient = tuple(factor*_integral_product(gap, gap_tail, poly, tail,
                low, high, precision) for poly, tail in columns)
            local_hessian = tuple(tuple(factor*_integral_product(*columns[i], *columns[j],
                low, high, precision) for j in indices) for i in indices)
        elif depth < max_depth:
            middle = (low+high)/2
            pending.extend(((middle, high, depth+1, False), (low, middle, depth+1, False)))
            continue
        else:
            unknown.append((low, high))
            clipped = I(min(bounds.lo, F(0)), min(bounds.hi, F(0)))
            if uncertain_measure:
                clipped = I(clipped.lo, 0)
            minimum, maximum = max(F(0), -clipped.hi), max(F(0), -clipped.lo)
            local_energy = factor*(high-low)*I(minimum**2, maximum**2)/2
            local_pressure = factor*(high-low)*clipped
            column_bounds = tuple(_evaluate(poly, angle, tail, precision) for poly, tail in columns)
            local_gradient = tuple(factor*(high-low)*clipped*column for column in column_bounds)
            local_hessian = tuple(tuple(factor*(high-low)*I(0, 1)*column_bounds[i]*column_bounds[j]
                for j in indices) for i in indices)
        local_energy = I(max(F(0), local_energy.lo), max(F(0), local_energy.hi))
        energy = _rounded_interval(energy+local_energy, precision)
        pressure = _rounded_interval(pressure+local_pressure, precision)
        gradient = [_rounded_interval(old+new, precision) for old, new in zip(gradient, local_gradient)]
        hessian = [[_rounded_interval(old+new, precision) for old, new in zip(row, local_row)]
            for row, local_row in zip(hessian, local_hessian)]
    return FiniteBornResponse(energy, pressure, tuple(gradient), tuple(tuple(row) for row in hessian),
        indices, _evaluate(gap, I(0, limit), gap_tail, precision), tuple(sorted(closed)),
        tuple(sorted(unknown)), polynomial_order, precision)


@dataclass(frozen=True, slots=True)
class FiniteBornContactIncrement:
    gap_coefficients: tuple
    jump_coefficients: tuple
    dt_s: F


@dataclass(frozen=True, slots=True)
class FiniteBornContactLedger:
    previous_state: object
    gap_coefficients: tuple
    jump_coefficients: tuple
    elapsed_time_s: F
    step_count: int
    activation_energy_decimal: str
    activation_energy_representation_error_j: I
    stored_energy_bounds_j: I
    conservative_path_work_increment_bounds_j: I
    external_work_increment_bounds_j: I
    numerical_dissipation_increment_bounds_j: I
    activation_paid_again: bool = False

    @property
    def history(self):
        return self.previous_state.history+(FiniteBornContactIncrement(self.gap_coefficients,
            self.jump_coefficients, self.elapsed_time_s-F.from_float(self.previous_state.elapsed_time_s)),)

    @property
    def external_work_definition(self):
        return "normal backward-Euler gap work; separate from generalized-coordinate endpoint work"

    @property
    def generalized_coordinate_energy_closure_certified(self):
        return False


@dataclass(frozen=True, slots=True)
class FiniteEquilibriumCertificate:
    residual_bounds: tuple
    own_scale_lower_bounds: tuple
    relative_residual_upper_bounds: tuple
    relative_tolerance: F

    @property
    def equilibrium_certified(self):
        return all(value <= self.relative_tolerance for value in self.relative_residual_upper_bounds)


def certify_finite_equilibrium_operands(operand_rows, *, relative_tolerance, precision=160):
    """Prove each actual force sum against that coordinate's own operands.

    Interval midpoints and another coordinate's scale cannot certify a row.
    A force-free row must be exactly zero, including its whole uncertainty.
    """
    if (type(operand_rows) is not tuple or not operand_rows
            or any(type(row) is not tuple or not row or any(type(value) is not I for value in row)
                for row in operand_rows)):
        raise ValueError("Each coordinate requires its complete immutable interval operand row")
    if type(relative_tolerance) is not F or not 0 < relative_tolerance < 1:
        raise ValueError("A literal exact relative force budget strictly between zero and one is required")
    if type(precision) is not int or not 64 <= precision <= 1024:
        raise ValueError("Certificate precision must be an integer from 64 to 1024")
    residuals, scales, ratios = [], [], []
    for index, operands in enumerate(operand_rows):
        residual = _rounded_interval(sum(operands, I(0)), precision)
        scale = sum((min(abs(value.lo), abs(value.hi)) if value.lo*value.hi > 0 else F(0)
            for value in operands), F(0))
        ratio = _maximum(residual)/scale if scale else (F(0) if residual.lo == residual.hi == 0 else None)
        if ratio is None or ratio > relative_tolerance:
            raise ValueError("Actual finite force balance exceeds coordinate %d's own budget" % index)
        residuals.append(residual)
        scales.append(scale)
        ratios.append(ratio)
    return FiniteEquilibriumCertificate(tuple(residuals), tuple(scales), tuple(ratios), relative_tolerance)


@dataclass(frozen=True, slots=True)
class NativeBornFiniteEquilibrium:
    coordinates_decimal: tuple
    source_affine_step: object
    exact_radial_normalization: object
    literal_owner_axes: tuple
    omega_rad_per_myr: tuple
    responses: tuple
    fields: tuple
    jacobians: tuple
    born_contact_ledgers: tuple
    corrected_foundation_steps: tuple
    residual_bounds: tuple
    operand_scales: tuple
    relative_residual_upper: tuple
    corrected_radial_coordinate_indices: tuple
    iterations: int
    precision: int
    relative_tolerance: F
    before_state_sha256: str
    birth_support_sha256: str

    @property
    def force_equilibrium_certified(self):
        return all(value <= self.relative_tolerance for value in self.relative_residual_upper)

    @property
    def material_admission_authorized(self):
        return False

    @property
    def force_model(self):
        return "unchanged original affine ports with actual finite-SO3 born continuum forces"

    @property
    def angular_path_error_bounds(self):
        return tuple((owner, tuple(abs(F(Decimal(value))-actual) for value, actual in
            zip(self.source_affine_step.coordinates_decimal[3*index:3*index+3], axis)))
            for index, (owner, axis) in enumerate(self.literal_owner_axes))


def _decimal(value, precision):
    with localcontext() as context:
        context.prec = precision
        return str(Decimal(value.numerator)/Decimal(value.denominator))


def _linear_solve(matrix, rhs, precision):
    """Small SPD radial system, solved without a binary64 scale cutoff."""
    from .geometric_native_birth_contact_network import _ldl_solve
    with localcontext() as context:
        context.prec = precision
        values = [[Decimal(_decimal(value, precision)) for value in row] for row in matrix]
        target = [Decimal(_decimal(value, precision)) for value in rhs]
        solution, _ = _ldl_solve(values, target)
        return tuple(F(value) for value in solution)


def solve_native_born_finite_equilibrium(case, *, affine_step=None,
        relative_tolerance=F(5, 10**10), precision=160, max_iterations=24):
    """Correct born radial coordinates and certify every actual mixed force.

    The old fourteen coordinates remain exactly those of the canonical-before
    affine source solve. This bounded correction is valid only when *all*
    nineteen final balances pass. Failure raises without adopting any history.
    """
    import math
    from .geometric_native_birth_exact_radial import normalize_native_birth_radial_before
    from .geometric_native_birth_contact_continuum import solve_born_continuum_network
    from .geometric_native_birth_continuum_motion import native_birth_continuum_frame
    from .geometric_native_birth_continuum_motion import build_native_birth_continuum_inputs
    from .geometric_native_birth_finite_pose import bound_born_finite_pose_fields
    from .geometric_native_birth_finite_force_bounds import bound_born_finite_pose_jacobian
    from .geometric_native_birth_contact import advance_born_foundation
    from .geometric_native_physical_history import _typed_fingerprint
    from .thermal import SECONDS_PER_MYR
    if type(relative_tolerance) is not F or not 0 < relative_tolerance <= F(1, 10**6):
        raise ValueError("Each coordinate requires an explicit exact relative force budget")
    if type(precision) is not int or not 100 <= precision <= 512:
        raise ValueError("The genuine finite force solve requires 100 to 512 digits")
    if type(max_iterations) is not int or not 1 <= max_iterations <= 100:
        raise ValueError("The finite radial iteration budget must be explicit")
    source_inputs, _ = build_native_birth_continuum_inputs(case["mesh"], case["previous"],
        case["load"], case["support"], case["dt_s"], decimal_precision=case["inputs"]["decimal_precision"])
    inputs, normalization = normalize_native_birth_radial_before(source_inputs, case["previous"],
        decimal_precision=source_inputs["decimal_precision"])
    replay = solve_born_continuum_network(**inputs)
    if affine_step is not None and _typed_fingerprint(affine_step) != _typed_fingerprint(replay):
        raise ValueError("The supplied affine initialization differs from its complete canonical-before source replay")
    affine_step = replay
    if len(affine_step.coordinates_decimal) != len(inputs["coordinates_before"]):
        raise ValueError("The affine initial source has the wrong physical coordinate count")
    # The complete initialization has just been replayed against its actual
    # source. No supplied force-residual claim is consumed.
    previous, support = case["previous"], case["support"]
    before_fingerprint = _typed_fingerprint(previous)
    support_fingerprint = _typed_fingerprint(support)
    radius = F.from_float(inputs["radius_m"])
    before = tuple(_number(value) for value in inputs["coordinates_before"])
    coordinates = [F(Decimal(value)) for value in affine_step.coordinates_decimal]
    old_dimension = len(inputs["old_stiffness"])
    indices = tuple(item.radial_dof_index for item in inputs["born_foundations"])
    if len(coordinates) != old_dimension+len(indices) or indices != tuple(range(old_dimension, len(coordinates))):
        raise ValueError("Every independent born radial coordinate must be registered once")
    physical_before = {item.coordinate_index: item.physical_before_m for item in normalization.coordinates}
    dt_myr = inputs["dt_s"]/SECONDS_PER_MYR
    omega = tuple((owner, tuple(float(value)/dt_myr for value in affine_step.coordinates[3*index:3*index+3]))
        for index, owner in enumerate(previous.plate_ids))
    axes = tuple((owner, tuple(F.from_float(value)*F.from_float(dt_myr) for value in values))
        for owner, values in omega)
    owner_axes = dict(axes)
    angular = {owner: tuple(range(3*index, 3*index+3)) for index, owner in enumerate(previous.plate_ids)}
    foundation_constants = tuple(F.from_float(item.parameters.density_kg_m3)
        *F.from_float(item.parameters.gravity_m_s2)*F.from_float(item.reference.basal_area_m2)*radius**2
        for item in inputs["born_foundations"])
    foundation_rest = tuple(F.from_float(item.reference.radial_birth_m)/radius for item in inputs["born_foundations"])
    frames = tuple(native_birth_continuum_frame(carrier, precision=max(256, precision+32))
        for carrier in support.interface_carriers)
    ports = []
    for attachment in inputs["born_continuum_attachments"]:
        row = tuple(Decimal(value) for value in attachment.declaration.normal_operator_coefficients[0])
        donor = tuple(index for index, value in enumerate(row) if value < 0)
        receiver = tuple(index for index, value in enumerate(row) if value > 0)
        if len(donor) != 1 or len(receiver) != 1:
            raise ValueError("A genuine finite interface requires its two physical radial banks")
        ports.append((donor[0], receiver[0]))
    def evaluate(current):
        fields, jacobians, responses = [], [], []
        for carrier, attachment, frame, (donor, receiver) in zip(support.interface_carriers,
                inputs["born_continuum_attachments"], frames, ports):
            sine, cosine = (F.from_float(value) for value in
                (math.sin(carrier.declaration.dip_radians), math.cos(carrier.declaration.dip_radians)))
            kwargs = dict(start_unit_bounds=frame[0], tangent_unit_bounds=frame[1], horizontal_unit_bounds=frame[2],
                donor_axis=owner_axes[carrier.donor_component.owner], receiver_axis=owner_axes[carrier.receiver_component.owner],
                radius_m=radius, donor_radial_before_m=physical_before[donor],
                donor_radial_increment_m=current[donor]*radius-physical_before[donor],
                sine_dip=sine, cosine_dip=cosine, time_fraction_bounds=I(1), precision=precision, order=28)
            field = bound_born_finite_pose_fields(**kwargs,
                receiver_radial_before_m=physical_before[receiver],
                receiver_radial_increment_m=current[receiver]*radius-physical_before[receiver],
                birth_gap_free_m=F.from_float(attachment.declaration.reference.birth_gap_free_m),
                birth_jump_free_m=F.from_float(attachment.declaration.reference.birth_jump_free_m))
            jacobian = bound_born_finite_pose_jacobian(attachment.declaration, **kwargs,
                donor_angular_indices=angular[carrier.donor_component.owner],
                receiver_angular_indices=angular[carrier.receiver_component.owner],
                donor_radial_index=donor, receiver_radial_index=receiver,
                angle_bounds=I(0, carrier.native_angle_bounds_radians.hi))
            response = integrate_born_finite_pose_response(field, jacobian,
                measure_density_m2_per_radian=carrier.inclined_measure_density_m2_per_radian,
                angle_end_bounds=carrier.native_angle_bounds_radians,
                normal_stiffness_pa_m=_normal_stiffness(attachment.declaration.parameters),
                precision=precision, polynomial_order=8, hessian_coordinate_indices=indices)
            fields.append(field)
            jacobians.append(jacobian)
            responses.append(response)
        radial_force = [I(constant*(current[index]-rest)) for index, constant, rest in
            zip(indices, foundation_constants, foundation_rest)]
        radial_hessian = [[I(foundation_constants[i] if i == j else 0) for j in range(len(indices))]
            for i in range(len(indices))]
        for response in responses:
            radial_force = [force+response.gradient_bounds[index] for index, force in zip(indices, radial_force)]
            radial_hessian = [[before+after for before, after in zip(row, response_row)]
                for row, response_row in zip(radial_hessian, response.gauss_newton_hessian_bounds)]
        return tuple(fields), tuple(jacobians), tuple(responses), tuple(radial_force), tuple(tuple(row) for row in radial_hessian)
    final = None
    for iteration in range(max_iterations):
        final = evaluate(coordinates)
        fields, jacobians, responses, radial_force, radial_hessian = final
        own_scales = [abs(constant*(coordinates[index]-rest))
            +sum((_maximum(response.gradient_bounds[index]) for response in responses), F(0))
            for index, constant, rest in zip(indices, foundation_constants, foundation_rest)]
        if all(_maximum(force) <= relative_tolerance*scale for force, scale in zip(radial_force, own_scales)):
            break
        correction = _linear_solve(tuple(tuple(value.midpoint for value in row) for row in radial_hessian),
            tuple(-value.midpoint for value in radial_force), precision)
        updated = [F(Decimal(_decimal(coordinates[index]+change, precision))) for index, change in zip(indices, correction)]
        if all(updated_value == coordinates[index] for index, updated_value in zip(indices, updated)):
            raise ValueError("The finite radial residual exceeds its budget at the declared precision")
        for index, updated_value in zip(indices, updated):
            coordinates[index] = updated_value
    else:
        raise ValueError("The finite born radial force solve exhausted its iterations without adopting memory")
    foundation_steps = tuple(advance_born_foundation(item.previous_state, item.parameters, item.reference,
        radial_displacement_m=float(coordinates[item.radial_dof_index]*radius), dt_s=inputs["dt_s"])
        for item in inputs["born_foundations"])
    literal_coordinates = tuple(value for _, axis in axes for value in axis)+tuple(
        coordinates[inputs["rotation_dof_count"]:])
    def angular_alias_operand(row, *, damping=False):
        def evaluate_operand(point):
            return sum((F.from_float(float(value))*(point[j]-before[j] if damping else point[j])
                for j, value in enumerate(row)), F(0))/(F.from_float(inputs["dt_s"]) if damping else 1)
        mechanical, literal = evaluate_operand(coordinates), evaluate_operand(literal_coordinates)
        return I(min(mechanical, literal), max(mechanical, literal))
    operand_rows = []
    for index in range(len(coordinates)):
        operands = [angular_alias_operand(inputs["drag_nm_s"][index], damping=True),
            I(-F.from_float(inputs["source_torque_nm"][index]))]
        if index < old_dimension:
            operands.append(angular_alias_operand(inputs["old_stiffness"][index]))
        else:
            foundation_index = indices.index(index)
            exact_force = foundation_constants[foundation_index]*(coordinates[index]-foundation_rest[foundation_index])
            saved_force = F.from_float(foundation_steps[foundation_index].energy_gradient_n)*radius
            operands.append(I(min(exact_force, saved_force), max(exact_force, saved_force)))
        for area, operator, (_, state) in zip(inputs["old_areas_m2"], inputs["old_anchor_operators"],
                affine_step.original_point_states):
            operands.append(I(F.from_float(area)*sum((F.from_float(traction)*F.from_float(float(row[index]))
                for traction, row in zip((state.traction_normal_pa, state.traction_shear_pa), operator)), F(0)))
                if index < old_dimension else I(0))
        operands.extend(response.gradient_bounds[index] for response in responses)
        operand_rows.append(tuple(operands))
    certificate = certify_finite_equilibrium_operands(tuple(operand_rows),
        relative_tolerance=relative_tolerance, precision=precision)
    ledgers = []
    for carrier, attachment, field, response in zip(support.interface_carriers,
            inputs["born_continuum_attachments"], fields, responses):
        prior = attachment.previous_state
        if prior.step_count != 0 or prior.history:
            raise ValueError("The first-born finite ledger requires its complete virgin paid activation source")
        birth_gap = F.from_float(attachment.declaration.reference.birth_gap_free_m)
        stiffness = _normal_stiffness(attachment.declaration.parameters)
        birth_energy = carrier.inclined_measure_density_m2_per_radian*carrier.native_angle_bounds_radians*stiffness*min(birth_gap, F(0))**2/2
        work = 2*response.energy_bounds_j-birth_gap*response.integrated_pressure_bounds_n
        numerical = work-response.energy_bounds_j+birth_energy
        if numerical.hi < 0:
            raise ValueError("The finite normal endpoint work violates its unilateral energy inequality")
        numerical = I(max(F(0), numerical.lo), max(F(0), numerical.hi))
        ledgers.append(FiniteBornContactLedger(prior, field.gap_coefficients, field.jump_coefficients,
            F.from_float(prior.elapsed_time_s)+F.from_float(inputs["dt_s"]), prior.step_count+1,
            prior.activation_energy_decimal, birth_energy-F(Decimal(prior.activation_energy_decimal)),
            response.energy_bounds_j, response.energy_bounds_j-birth_energy, work, numerical))
    if _typed_fingerprint(previous) != before_fingerprint or _typed_fingerprint(support) != support_fingerprint:
        raise ValueError("A finite force trial changed its immutable accepted source")
    inputs["domain_check"](tuple(float(value) for value in coordinates))
    final_coordinates_decimal = affine_step.coordinates_decimal[:old_dimension]+tuple(
        _decimal(value, precision) for value in coordinates[old_dimension:])
    return NativeBornFiniteEquilibrium(final_coordinates_decimal, affine_step,
        normalization, axes, omega, responses, fields, jacobians, tuple(ledgers), foundation_steps,
        certificate.residual_bounds, certificate.own_scale_lower_bounds,
        certificate.relative_residual_upper_bounds, indices, iteration+1, precision,
        relative_tolerance, previous.state_sha256, support.support_sha256)
