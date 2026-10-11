"""Full-arc unilateral HOT potentials and a joint Decimal force solve.

Only the declared Born arcs are integrated here. Original contacts retain
their original discrete measures and histories. Analytic trigonometric
moments integrate every closed subarc; a root band is retained as numerical
uncertainty, never assigned a finite material point or silently discarded.
Opening carries zero normal traction and zero shear traction. This mechanical
owner does not authorize a material transaction or a temporal contact gate.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from decimal import Decimal, localcontext
from fractions import Fraction
import json
import math

from .geometric_native_birth_contact import (
    born_contact_parameters_to_dict, born_contact_reference_to_dict,
    born_foundation_state_to_dict, advance_born_foundation,
)
from .geometric_native_birth_contact_network import (
    _advance_original_closed_inviscid, _d, _dot, _ldl_solve, _matvec,
    _relative, _series, _sum, _upper_float, solve_birth_contact_network,
)
from .geometric_native_physical_history import _typed_fingerprint


FORMAT = "native-full-arc-born-continuum-force-1"
FIELD_BASIS = ("constant", "cos_theta", "sin_theta")


def _decimal(value):
    if type(value) is not str:
        raise ValueError("Continuum declarations require canonical finite Decimal strings")
    number = Decimal(value)
    if not number.is_finite() or str(number) != value:
        raise ValueError("Continuum declarations require canonical finite Decimal strings")
    return number


def _strings(values):
    return tuple(str(value) for value in values)


def _coefficient_strings(values):
    if type(values) is not tuple or len(values) != 3:
        raise ValueError("A continuum field requires exactly three immutable canonical coefficients")
    return tuple(map(_decimal, values))


def _coefficient_decimals(values):
    if type(values) is not tuple or len(values) != 3 or any(
            type(value) is not Decimal or not value.is_finite() for value in values):
        raise ValueError("An internal continuum field requires exactly three finite Decimal coefficients")
    return values


def _precision(value):
    if type(value) is not int or value < 100:
        raise ValueError("A continuum declaration requires an integer precision of at least 100 digits")
    return value


def _duration(value):
    if type(value) is not float or not math.isfinite(value) or value <= 0:
        raise ValueError("A continuum history requires a finite positive float duration")
    return value


def _sincos(angle):
    """Taylor evaluation on a declared minor arc, with guard digits."""
    with localcontext() as work:
        work.prec += 20
        square = angle*angle
        sine, cosine = angle, Decimal(1)
        st, ct = angle, Decimal(1)
        threshold = Decimal(10)**(-work.prec-5)
        for index in range(1, 1024):
            st *= -square/Decimal((2*index)*(2*index+1))
            ct *= -square/Decimal((2*index-1)*(2*index))
            sine += st
            cosine += ct
            if max(abs(st), abs(ct)) < threshold:
                return +sine, +cosine
    raise ValueError("The declared trigonometric arc did not converge")


def _field(coefficients, angle):
    sine, cosine = _sincos(angle)
    return coefficients[0]+coefficients[1]*cosine+coefficients[2]*sine


def _derivative(coefficients):
    return (Decimal(0), coefficients[2], -coefficients[1])


def _zero_band(coefficients, low, high, precision):
    first, last = _field(coefficients, low), _field(coefficients, high)
    if first == 0:
        return low, low
    if last == 0:
        return high, high
    if first*last >= 0:
        raise ValueError("A continuum root requires a complete sign-changing bracket")
    width_bound = (high-low)*Decimal(10)**(-precision//2)
    for _ in range(4096):
        if high-low <= width_bound:
            return low, high
        middle = (low+high)/2
        value = _field(coefficients, middle)
        if value == 0:
            return middle, middle
        if value*first > 0:
            low, first = middle, value
        else:
            high, last = middle, value
    raise ValueError("The continuum root did not reach its declared enclosure width")


def _critical_points(coefficients, low, high, precision):
    derivative = _derivative(coefficients)
    if derivative[1] == derivative[2] == 0:
        return ()
    first, last = _field(derivative, low), _field(derivative, high)
    if first*last < 0:
        band = _zero_band(derivative, low, high, precision)
        return ((band[0]+band[1])/2,)
    return ()


def _closed_parts(coefficients, low, high, precision):
    # A degree-one trigonometric field on a minor interval (<pi) has at
    # most one interior stationary point. Its monotone pieces isolate every
    # actual zero, including a positive pocket between negative endpoints.
    knots = (low,)+_critical_points(coefficients, low, high, precision)+(high,)
    bands, roots = [], []
    for first, last in zip(knots, knots[1:]):
        left, right = _field(coefficients, first), _field(coefficients, last)
        if left*right < 0:
            band = _zero_band(coefficients, first, last, precision)
            bands.append(band)
            roots.append((band[0]+band[1])/2)
        elif left == 0:
            roots.append(first)
        elif right == 0:
            roots.append(last)
    cuts = tuple(sorted(set((low, high, *roots))))
    closed = tuple((a, b) for a, b in zip(cuts, cuts[1:])
        if a < b and _field(coefficients, (a+b)/2) < 0)
    return closed, tuple(bands)


def _moments(low, high):
    width, center = high-low, (high+low)/2
    sh, _ = _sincos(width/2)
    sc, cc = _sincos(center)
    sw, _ = _sincos(width)
    s2, c2 = _sincos(2*center)
    return ((width, 2*cc*sh, 2*sc*sh),
        (2*cc*sh, (width+c2*sw)/2, s2*sw/2),
        (2*sc*sh, s2*sw/2, (width-c2*sw)/2))


def _integral_product(first, second, parts):
    return _sum(_dot(first, _matvec(_moments(low, high), second)) for low, high in parts)


@dataclass(frozen=True, slots=True)
class BornContinuumDeclaration:
    parameters: object
    reference: object
    angle_start_decimal: str
    angle_end_decimal: str
    measure_density_decimal: str
    normal_operator_coefficients: tuple
    shear_operator_coefficients: tuple
    provenance: str

    def __post_init__(self):
        born_contact_parameters_to_dict(self.parameters)
        born_contact_reference_to_dict(self.reference)
        low, high, density = map(_decimal, (self.angle_start_decimal,
            self.angle_end_decimal, self.measure_density_decimal))
        if not (Decimal(0) <= low < high < Decimal("3.1415926535897932384626433832795028841971693993751")) or density <= 0:
            raise ValueError("The declared continuum needs a positive minor arc and positive inclined measure density")
        dimension = None
        for operator in (self.normal_operator_coefficients, self.shear_operator_coefficients):
            if type(operator) is not tuple or len(operator) != 3:
                raise ValueError("Each whole-arc operator requires all three trigonometric coefficient rows")
            for row in operator:
                if type(row) is not tuple:
                    raise ValueError("Whole-arc coefficient rows must be immutable tuples")
                if dimension is None:
                    dimension = len(row)
                if len(row) != dimension or dimension == 0:
                    raise ValueError("All whole-arc operator rows must cover the same complete coordinates")
                for value in row:
                    _decimal(value)
        if type(self.provenance) is not str or not self.provenance.strip():
            raise ValueError("Full-arc continuum provenance must be explicitly declared")


@dataclass(frozen=True, slots=True)
class BornContinuumIncrement:
    gap_coefficients_decimal: tuple
    jump_coefficients_decimal: tuple
    dt_s: float

    def __post_init__(self):
        _coefficient_strings(self.gap_coefficients_decimal)
        _coefficient_strings(self.jump_coefficients_decimal)
        _duration(self.dt_s)


@dataclass(frozen=True, slots=True)
class BornContinuumContactState:
    declaration: BornContinuumDeclaration
    history: tuple
    gap_coefficients_decimal: tuple
    jump_coefficients_decimal: tuple
    activation_energy_decimal: str
    stored_energy_decimal: str
    cumulative_external_work_decimal: str
    cumulative_numerical_dissipation_decimal: str
    elapsed_time_s: float
    step_count: int
    decimal_precision: int

    def __post_init__(self):
        if type(self.declaration) is not BornContinuumDeclaration:
            raise ValueError("A continuum state requires its complete typed declaration")
        if type(self.history) is not tuple or any(type(row) is not BornContinuumIncrement for row in self.history):
            raise ValueError("A continuum history requires immutable typed increments")
        for row in self.history:
            row.__post_init__()
        _coefficient_strings(self.gap_coefficients_decimal)
        _coefficient_strings(self.jump_coefficients_decimal)
        for name in ("activation_energy_decimal", "stored_energy_decimal",
                "cumulative_external_work_decimal", "cumulative_numerical_dissipation_decimal"):
            value = _decimal(getattr(self, name))
            if name != "cumulative_external_work_decimal" and value < 0:
                raise ValueError("A continuum energy or dissipation ledger cannot be negative")
        if type(self.elapsed_time_s) is not float or not math.isfinite(self.elapsed_time_s) or self.elapsed_time_s < 0:
            raise ValueError("A continuum state requires a finite nonnegative float clock")
        if type(self.step_count) is not int or self.step_count < 0 or self.step_count != len(self.history):
            raise ValueError("A continuum step count must exactly count its typed history")
        _precision(self.decimal_precision)


@dataclass(frozen=True, slots=True)
class BornContinuumAttachment:
    declaration: BornContinuumDeclaration
    previous_state: BornContinuumContactState


@dataclass(frozen=True, slots=True)
class BornContinuumContactStep:
    state: BornContinuumContactState
    gradient_decimal: tuple
    hessian_decimal: tuple
    closed_angle_parts_decimal: tuple
    root_bands_decimal: tuple
    external_work_decimal: str
    stored_energy_increment_decimal: str
    numerical_dissipation_decimal: str
    gap_extrema_decimal: tuple
    jump_increment_extrema_decimal: tuple
    constitutive_root_band_uncertainty: str
    root_band_gradient_bounds_decimal: tuple
    root_band_hessian_bounds_decimal: tuple


@dataclass(frozen=True, slots=True)
class BornContinuumNetworkStep:
    coordinates: tuple
    coordinates_decimal: tuple
    coordinate_representative_error: tuple
    original_point_states: tuple
    original_point_steps: tuple
    born_foundation_steps: tuple
    continuum_contact_steps: tuple
    original_anchor_fields: tuple
    residual_decimal: tuple
    per_coordinate_relative_residual: tuple
    actual_per_coordinate_relative_residual: tuple
    diagnostics: tuple
    decimal_precision: int
    format: str = FORMAT


def continuum_field_extrema(coefficients_decimal, low_decimal, high_decimal, *, precision=160):
    _precision(precision)
    with localcontext() as context:
        context.prec = precision
        coefficients = _coefficient_strings(coefficients_decimal)
        low, high = map(_decimal, (low_decimal, high_decimal))
        points = (low, high)+_critical_points(coefficients, low, high, precision)
        values = tuple(_field(coefficients, angle) for angle in points)
        return str(min(values)), str(max(values))


def continuum_radial_ordering_increment_coefficients(gap_coefficients_decimal,
        jump_coefficients_decimal, *, birth_gap_free_m, birth_jump_free_m,
        sine_decimal, cosine_decimal, precision=160):
    """Objective free-anchor radial difference, not a material permission.

    In the declared normal/down-dip frame r=c*n-s*d. Birth precompression
    is a paid constitutive offset; subtracting its reference yields the
    actual incremental free-anchor radial ordering field.
    """
    _precision(precision)
    with localcontext() as context:
        context.prec = precision
        gap = list(_coefficient_strings(gap_coefficients_decimal))
        jump = list(_coefficient_strings(jump_coefficients_decimal))
        gap[0] -= _d(birth_gap_free_m)
        jump[0] -= _d(birth_jump_free_m)
        sine, cosine = map(_decimal, (sine_decimal, cosine_decimal))
        return _strings(tuple(cosine*g-sine*j for g, j in zip(gap, jump)))


def _root_band_bounds(declaration, gap, bands):
    factor = _decimal(declaration.measure_density_decimal)*_series(declaration.parameters)
    operator = tuple(tuple(map(_decimal, row)) for row in declaration.normal_operator_coefficients)
    sup = tuple(_sum(abs(row[index]) for row in operator) for index in range(len(operator[0])))
    gradient = [Decimal(0)]*len(sup)
    hessian = [[Decimal(0)]*len(sup) for _ in sup]
    energy = Decimal(0)
    derivative_bound = abs(gap[1])+abs(gap[2])
    for low, high in bands:
        width = high-low
        gap_bound = max(abs(_field(gap, low)), abs(_field(gap, high)))+derivative_bound*width
        energy += factor*width*gap_bound**2/2
        for i in range(len(sup)):
            gradient[i] += factor*width*gap_bound*sup[i]
            for j in range(len(sup)):
                hessian[i][j] += factor*width*sup[i]*sup[j]
    return energy, tuple(gradient), tuple(tuple(row) for row in hessian)


def _response(declaration, gap, precision):
    _precision(precision)
    _coefficient_decimals(gap)
    low, high = map(_decimal, (declaration.angle_start_decimal, declaration.angle_end_decimal))
    factor = _decimal(declaration.measure_density_decimal)*_series(declaration.parameters)
    parts, bands = _closed_parts(gap, low, high, precision)
    moment = [[Decimal(0)]*3 for _ in range(3)]
    for a, b in parts:
        local = _moments(a, b)
        for i in range(3):
            for j in range(3):
                moment[i][j] += local[i][j]
    operators = tuple(tuple(map(_decimal, row)) for row in declaration.normal_operator_coefficients)
    weighted_gap = _matvec(moment, gap)
    count = len(operators[0])
    gradient = tuple(factor*_sum(operators[basis][index]*weighted_gap[basis] for basis in range(3))
        for index in range(count))
    hessian = tuple(tuple(factor*_sum(operators[i][first]*moment[i][j]*operators[j][second]
        for i in range(3) for j in range(3)) for second in range(count)) for first in range(count))
    energy = factor*_dot(gap, weighted_gap)/2
    if energy < 0:
        raise ValueError("The full-arc normal potential lost positivity at the declared precision")
    # Finite zero brackets bound the omitted/misclassified sliver. This
    # endpoint-band majorant is exported even when far below the main signal.
    uncertainty = _root_band_bounds(declaration, gap, bands)[0]
    return energy, gradient, hessian, parts, bands, uncertainty


def activate_born_continuum_contact(declaration, *, decimal_precision=160):
    _precision(decimal_precision)
    if type(declaration) is not BornContinuumDeclaration:
        raise ValueError("An explicit full-arc declaration and at least 100 digits are required")
    with localcontext() as context:
        context.prec = decimal_precision
        gap = (_d(declaration.reference.birth_gap_free_m), Decimal(0), Decimal(0))
        jump = (_d(declaration.reference.birth_jump_free_m), Decimal(0), Decimal(0))
        energy = _response(declaration, gap, decimal_precision)[0]
        return BornContinuumContactState(declaration, (), _strings(gap), _strings(jump),
            str(energy), str(energy), "0", "0", 0., 0, decimal_precision)


def _advance_field(previous, gap, jump, dt, precision):
    _precision(precision)
    _coefficient_decimals(gap)
    _coefficient_decimals(jump)
    _duration(dt)
    declaration = previous.declaration
    old_gap = tuple(map(_decimal, previous.gap_coefficients_decimal))
    old_jump = tuple(map(_decimal, previous.jump_coefficients_decimal))
    response = _response(declaration, gap, precision)
    energy, gradient, hessian, parts, bands, uncertainty = response
    change = tuple(now-old for now, old in zip(gap, old_gap))
    factor = _decimal(declaration.measure_density_decimal)*_series(declaration.parameters)
    work = factor*_integral_product(gap, change, parts)
    old_energy = _decimal(previous.stored_energy_decimal)
    remainder = work-energy+old_energy
    round_bound = Decimal(10)**(-precision+20)*max(abs(work), energy+old_energy, Decimal("1e-300"))
    if remainder < -round_bound:
        raise ValueError("The full-arc unilateral backward-Euler remainder became negative")
    if remainder < 0:
        # The declared roundoff is recorded; no positive mechanical signal or
        # contact measure is removed by this arithmetic floor.
        uncertainty += -remainder
        remainder = Decimal(0)
    elapsed = math.fsum((previous.elapsed_time_s, dt))
    if elapsed <= previous.elapsed_time_s or not math.isfinite(elapsed):
        raise ValueError("The full-arc state cannot resolve its physical clock increment")
    history = previous.history+(BornContinuumIncrement(_strings(gap), _strings(jump), dt),)
    state = BornContinuumContactState(declaration, history, _strings(gap), _strings(jump),
        previous.activation_energy_decimal, str(energy),
        str(_decimal(previous.cumulative_external_work_decimal)+work),
        str(_decimal(previous.cumulative_numerical_dissipation_decimal)+remainder),
        elapsed, previous.step_count+1, precision)
    extrema = continuum_field_extrema(state.gap_coefficients_decimal,
        declaration.angle_start_decimal, declaration.angle_end_decimal, precision=precision)
    jump_change = _strings(tuple(now-old for now, old in zip(jump, old_jump)))
    jump_extrema = continuum_field_extrema(jump_change,
        declaration.angle_start_decimal, declaration.angle_end_decimal, precision=precision)
    _, gradient_bounds, hessian_bounds = _root_band_bounds(declaration, gap, bands)
    return BornContinuumContactStep(state, _strings(gradient), tuple(_strings(row) for row in hessian),
        tuple(_strings(row) for row in parts), tuple(_strings(row) for row in bands),
        str(work), str(energy-old_energy), str(remainder), extrema, jump_extrema, str(uncertainty),
        _strings(gradient_bounds), tuple(_strings(row) for row in hessian_bounds))


def validate_born_continuum_contact_state(state, *, expected_declaration):
    if type(state) is not BornContinuumContactState or _typed_fingerprint(state.declaration) != _typed_fingerprint(expected_declaration):
        raise ValueError("The full-arc state requires the complete independent source declaration")
    state.__post_init__()
    replay = activate_born_continuum_contact(expected_declaration, decimal_precision=state.decimal_precision)
    with localcontext() as context:
        context.prec = state.decimal_precision
        for row in state.history:
            if type(row) is not BornContinuumIncrement:
                raise ValueError("A full-arc history increment has an unsupported type")
            replay = _advance_field(replay, tuple(map(_decimal, row.gap_coefficients_decimal)),
                tuple(map(_decimal, row.jump_coefficients_decimal)), row.dt_s, state.decimal_precision).state
    if _typed_fingerprint(state) != _typed_fingerprint(replay):
        raise ValueError("The full-arc state differs from its complete independent field/work replay")
    return state


def coincident_birth_arc_operator_coefficients(*, start, tangent, horizontal, sine, cosine,
        radius_m, donor_displacement_m, dimension, angular_owner_dofs,
        donor_owner, receiver_owner, donor_radial_coupling, receiver_radial_coupling,
        decimal_precision=160):
    """Analytic (1,cosθ,sinθ) coefficients on a coincident reference arc.

    Native frame interval errors remain owned by the geometric declaration.
    This constructor neither samples the arc nor infers material support.
    """
    _precision(decimal_precision)
    with localcontext() as context:
        context.prec = decimal_precision
        a, b, h = [tuple(map(_decimal, row)) for row in (start, tangent, horizontal)]
        s, c = map(_decimal, (sine, cosine))
        radius = _d(radius_m)
        lever_radius = radius+_d(donor_displacement_m)
        def cross(first, second):
            return tuple(first[j]*second[k]-first[k]*second[j] for j, k in ((1,2), (2,0), (0,1)))
        owners = dict(angular_owner_dofs)
        normal, shear = [[Decimal(0)]*dimension for _ in range(3)], [[Decimal(0)]*dimension for _ in range(3)]
        for basis, vector in ((1, a), (2, b)):
            lever = cross(vector, h)
            for factor, operator in ((s, normal), (c, shear)):
                for axis, value in enumerate(lever):
                    contribution = lever_radius*factor*value
                    operator[basis][owners[donor_owner][axis]] -= contribution
                    operator[basis][owners[receiver_owner][axis]] += contribution
        for coupling, sign in ((donor_radial_coupling, -1), (receiver_radial_coupling, 1)):
            for index, weight in coupling:
                normal[0][index] += sign*radius*c*_d(weight)
                shear[0][index] -= sign*radius*s*_d(weight)
        return tuple(_strings(row) for row in normal), tuple(_strings(row) for row in shear)


def solve_born_continuum_network(*, born_continuum_attachments, **network_inputs):
    """Replace born point weights by complete integrated arc potentials.

    The untouched discrete solver validates the source/old/foundation inputs
    and gives the cold-only starting point. Every subsequent residual,
    tangent, Newton step and work certificate includes the full born arcs.
    Original local states and all independent foundations are advanced only
    after that new joint equilibrium has genuinely converged.
    """
    if network_inputs.get("born_attachments", ()) != ():
        raise ValueError("Continuum mechanics must not also include born midpoint contact weights")
    _duration(network_inputs.get("dt_s"))
    if type(born_continuum_attachments) is not tuple or any(type(item) is not BornContinuumAttachment for item in born_continuum_attachments):
        raise ValueError("All full-arc attachments must form an immutable complete tuple")
    precision = network_inputs.get("decimal_precision", 160)
    if type(precision) is not int or precision < 120:
        raise ValueError("The mixed continuum force solve requires at least 120 Decimal digits")
    inputs = dict(network_inputs, born_attachments=(), decimal_precision=precision)
    seed = solve_birth_contact_network(**inputs)
    count, old_dimension = len(seed.coordinates), len(inputs["old_stiffness"])
    rotation_count = inputs["rotation_dof_count"]
    foundation_bindings = {(item.reference.cohort_id, item.reference.material_id,
        item.reference.parent_id, item.reference.owner) for item in inputs["born_foundations"]}
    families = []
    for item in born_continuum_attachments:
        declaration = item.declaration
        validate_born_continuum_contact_state(item.previous_state, expected_declaration=declaration)
        if item.previous_state.decimal_precision != precision:
            raise ValueError("The source full-arc history and new solve must use the same declared Decimal precision")
        if len(declaration.normal_operator_coefficients[0]) != count:
            raise ValueError("Each continuum operator must cover every actual mixed coordinate")
        for operator in (declaration.normal_operator_coefficients, declaration.shear_operator_coefficients):
            for row in operator:
                for axis in range(3):
                    # Decimal declarations are already finite exact literals.
                    # Their common-rotation identity must survive exact sum.
                    if sum((Fraction(Decimal(value)) for value in row[axis:rotation_count:3]), Fraction(0)) != 0:
                        raise ValueError("Full-arc operators must exactly annihilate common angular rotation")
        ref = declaration.reference
        if not foundation_bindings.intersection(((ref.donor_cohort_id, ref.donor_material_id, ref.donor_parent_id, ref.donor_owner),
                (ref.receiver_cohort_id, ref.receiver_material_id, ref.receiver_parent_id, ref.receiver_owner))):
            raise ValueError("Each full-arc interface must retain its independently registered HOT material bank")
        families.append(ref.family_id)
    if len(set(families)) != len(families):
        raise ValueError("Full-arc physical interface identities cannot repeat")
    with localcontext() as context:
        context.prec = precision
        before = [(_decimal(value) if isinstance(value, str) else _d(value)) for value in inputs["coordinates_before"]]
        duration, radius = _d(inputs["dt_s"]), _d(inputs["radius_m"])
        source = list(map(_d, inputs["source_torque_nm"]))
        damping = [[_d(value)/duration for value in row] for row in inputs["drag_nm_s"]]
        stiffness = [[Decimal(0)]*count for _ in range(count)]
        for i, row in enumerate(inputs["old_stiffness"]):
            for j, value in enumerate(row):
                stiffness[i][j] = _d(value)
        rest = [Decimal(0)]*count
        foundation_constants = []
        for item in inputs["born_foundations"]:
            coefficient = _d(item.parameters.density_kg_m3)*_d(item.parameters.gravity_m_s2)*_d(item.reference.basal_area_m2)
            index = item.radial_dof_index
            stiffness[index][index] = coefficient*radius**2
            rest[index] = _d(item.reference.radial_birth_m)/radius
            foundation_constants.append(coefficient)
        original_operators = [tuple(tuple(map(_d, row))+(Decimal(0),)*(count-old_dimension) for row in operator)
            for operator in inputs["old_anchor_operators"]]
        old_offsets = [tuple(_d(getattr(state, name))-_dot(operator[mode], before)
            for mode, name in enumerate(("gap_free_m", "jump_free_m")))
            for state, operator in zip(inputs["old_states"], original_operators)]
        cold_coefficients = [_d(area)*_series(material)
            for area, material in zip(inputs["old_areas_m2"], inputs["old_materials"])]
        field_operators = [tuple(tuple(map(_decimal, row)) for row in item.declaration.normal_operator_coefficients)
            for item in born_continuum_attachments]
        shear_operators = [tuple(tuple(map(_decimal, row)) for row in item.declaration.shear_operator_coefficients)
            for item in born_continuum_attachments]
        normal_offsets = [tuple(_decimal(value)-_dot(row, before)
            for value, row in zip(item.previous_state.gap_coefficients_decimal, operator))
            for item, operator in zip(born_continuum_attachments, field_operators)]
        shear_offsets = [tuple(_decimal(value)-_dot(row, before)
            for value, row in zip(item.previous_state.jump_coefficients_decimal, operator))
            for item, operator in zip(born_continuum_attachments, shear_operators)]
        base = [[damping[i][j]+stiffness[i][j] for j in range(count)] for i in range(count)]
        for coefficient, operator in zip(cold_coefficients, original_operators):
            for i in range(count):
                for j in range(count):
                    base[i][j] += coefficient*operator[0][i]*operator[0][j]

        def evaluate(coordinates):
            delta = [value-old for value, old in zip(coordinates, before)]
            drag_force = _matvec(damping, delta)
            elastic_force = _matvec(stiffness, [value-reference for value, reference in zip(coordinates, rest)])
            cold_fields = [tuple(offset[mode]+_dot(operator[mode], coordinates) for mode in range(2))
                for offset, operator in zip(old_offsets, original_operators)]
            if any(fields[0] > 0 for fields in cold_fields):
                raise ValueError("The original cold continuum gate still forbids normal opening")
            cold_forces = [tuple(coefficient*fields[0]*value for value in operator[0])
                for coefficient, fields, operator in zip(cold_coefficients, cold_fields, original_operators)]
            born_fields = [tuple(offset[basis]+_dot(operator[basis], coordinates) for basis in range(3))
                for offset, operator in zip(normal_offsets, field_operators)]
            responses = [_response(item.declaration, field, precision)
                for item, field in zip(born_continuum_attachments, born_fields)]
            forces = cold_forces+[response[1] for response in responses]
            residual = tuple(drag_force[i]+elastic_force[i]-source[i]+_sum(row[i] for row in forces)
                for i in range(count))
            relatives = tuple(_relative(residual[i], (drag_force[i], elastic_force[i], source[i],
                *(row[i] for row in forces))) for i in range(count))
            hessian = [[base[i][j]+_sum(response[2][i][j] for response in responses)
                for j in range(count)] for i in range(count)]
            potential = (_dot(delta, _matvec(damping, delta))/2
                +_dot([value-reference for value, reference in zip(coordinates, rest)], elastic_force)/2
                -_dot(source, coordinates)
                +_sum(coefficient*fields[0]**2/2 for coefficient, fields in zip(cold_coefficients, cold_fields))
                +_sum(response[0] for response in responses))
            return residual, relatives, hessian, potential, cold_fields, born_fields, responses, drag_force, elastic_force, cold_forces

        coordinates = list(map(_decimal, seed.coordinates_decimal))
        current = evaluate(coordinates)
        attempts = 0
        for iteration in range(256):
            if max(current[1], default=Decimal(0)) <= Decimal("1e-40"):
                break
            correction, pivots = _ldl_solve(current[2], [-value for value in current[0]])
            descent = _dot(current[0], correction)
            if descent >= 0:
                raise ValueError("The full-arc conservative force Newton direction lost strict descent")
            fraction, accepted = Decimal(1), False
            for _ in range(160):
                attempts += 1
                trial_coordinates = [value+fraction*change for value, change in zip(coordinates, correction)]
                try:
                    callback = inputs.get("domain_check")
                    if callback is not None:
                        callback(tuple(float(value) for value in trial_coordinates))
                    trial = evaluate(trial_coordinates)
                    if trial[3] <= current[3]+Decimal("0.0001")*fraction*descent:
                        coordinates, current, accepted = trial_coordinates, trial, True
                        break
                except (ValueError, RuntimeError, OverflowError):
                    pass
                fraction /= 2
            if not accepted:
                raise ValueError("The full-arc force line search failed without adopting any history")
        else:
            raise ValueError("The full-arc force solve did not reach all per-coordinate physical residual gates")
        representatives = tuple(float(value) for value in coordinates)
        if not all(math.isfinite(value) for value in representatives):
            raise ValueError("The full-arc solution has no finite kinematic representatives")
        if inputs.get("max_rotation_radians") is not None:
            bound = inputs["max_rotation_radians"]
            if any(math.sqrt(math.fsum(representatives[start+axis]**2 for axis in range(3))) > bound
                    for start in range(0, rotation_count, 3)):
                raise ValueError("The full-arc solution exceeds the original angular validity bound")
        callback = inputs.get("domain_check")
        if callback is not None:
            callback(representatives)
        delta = [value-old for value, old in zip(coordinates, before)]
        old_trials = tuple(_advance_original_closed_inviscid(material, area,
            float(fields[0]), float(fields[1]), old_state, inputs["dt_s"])
            for material, area, fields, old_state in zip(inputs["old_materials"], inputs["old_areas_m2"],
                current[4], inputs["old_states"]))
        original_steps = tuple(step for step, _ in old_trials)
        continuum_steps = tuple(_advance_field(item.previous_state, gap,
            tuple(offset[basis]+_dot(operator[basis], coordinates) for basis in range(3)),
            inputs["dt_s"], precision)
            for item, gap, offset, operator in zip(born_continuum_attachments, current[5], shear_offsets, shear_operators))
        foundation_steps = tuple(advance_born_foundation(item.previous_state, item.parameters, item.reference,
            radial_displacement_m=float(coordinates[item.radial_dof_index]*radius), dt_s=inputs["dt_s"])
            for item in inputs["born_foundations"])
        actual_elastic = current[8].copy()
        foundation_alias_bounds, foundation_energy_alias_bounds, foundation_remainder_alias_bounds = [], [], []
        for item, coefficient, step in zip(inputs["born_foundations"], foundation_constants, foundation_steps):
            index = item.radial_dof_index
            actual_force = _d(step.energy_gradient_n)*radius
            theoretical = current[8][index]
            if _relative(actual_force-theoretical, (actual_force, theoretical)) > Decimal("5e-12"):
                raise ValueError("An independently stored foundation reaction failed its own continuum force scale")
            actual_elastic[index] = actual_force
            current_w, old_w = coordinates[index]*radius, before[index]*radius
            e_new = abs(_d(step.state.radial_displacement_m)-current_w)
            e_old = abs(_d(item.previous_state.radial_displacement_m)-old_w)
            born = _d(item.reference.radial_birth_m)
            expected_work = coefficient*(current_w-born)*(current_w-old_w)
            alias = coefficient*(e_new*abs(current_w-old_w)+(abs(current_w-born)+e_new)*(e_new+e_old))
            actual_work = _d(step.external_work_j)
            if abs(actual_work-expected_work) > alias+Decimal("5e-12")*(abs(actual_work)+abs(expected_work)):
                raise ValueError("An individual foundation work ledger exceeds its continuum radial representative bound")
            expected_energy = coefficient*(current_w-born)**2/2
            actual_energy = _d(step.state.foundation_energy_j)
            energy_alias = coefficient*(abs(current_w-born)*e_new+e_new**2/2)
            if abs(actual_energy-expected_energy) > energy_alias+Decimal("5e-12")*(abs(actual_energy)+abs(expected_energy)):
                raise ValueError("An individual foundation energy exceeds its continuum radial representative bound")
            expected_remainder = coefficient*(current_w-old_w)**2/2
            actual_remainder = _d(step.numerical_dissipation_j)
            error_delta = e_new+e_old
            remainder_alias = coefficient*(abs(current_w-old_w)*error_delta+error_delta**2/2)
            if abs(actual_remainder-expected_remainder) > remainder_alias+Decimal("5e-12")*(abs(actual_remainder)+abs(expected_remainder)):
                raise ValueError("An individual foundation backward-Euler remainder exceeds its continuum radial representative bound")
            foundation_alias_bounds.append(_upper_float(alias))
            foundation_energy_alias_bounds.append(_upper_float(energy_alias))
            foundation_remainder_alias_bounds.append(_upper_float(remainder_alias))
        actual_forces = [tuple(_d(area)*(_d(step.traction_pa[0])*operator[0][i]+_d(step.traction_pa[1])*operator[1][i])
            for i in range(count)) for area, step, operator in zip(inputs["old_areas_m2"], original_steps, original_operators)]
        actual_forces += [tuple(map(_decimal, step.gradient_decimal)) for step in continuum_steps]
        actual_residual = tuple(current[7][i]+actual_elastic[i]-source[i]+_sum(row[i] for row in actual_forces)
            for i in range(count))
        actual_relative = tuple(_relative(actual_residual[i], (current[7][i], actual_elastic[i], source[i],
            *(row[i] for row in actual_forces))) for i in range(count))
        if max(actual_relative, default=Decimal(0)) > Decimal("5e-10"):
            raise ValueError("An actual stored reaction failed its individual mixed continuum coordinate balance")
        source_work = _dot(source, delta)
        basal_work = _dot(delta, _matvec(damping, delta))
        cold_stiffness = [row[:old_dimension] for row in stiffness[:old_dimension]]
        cold_delta, cold_now, cold_before = delta[:old_dimension], coordinates[:old_dimension], before[:old_dimension]
        cold_increment = _dot(cold_delta, _matvec(cold_stiffness, [a+b for a, b in zip(cold_now, cold_before)]))/2
        cold_remainder = _dot(cold_delta, _matvec(cold_stiffness, cold_delta))/2
        cold_work = _sum(coefficient*fields[0]*_dot(operator[0], delta)
            for coefficient, fields, operator in zip(cold_coefficients, current[4], original_operators))
        born_work = _sum(_decimal(step.external_work_decimal) for step in continuum_steps)
        foundation_work = _sum(coefficient*(coordinates[item.radial_dof_index]*radius-_d(item.reference.radial_birth_m))
            *delta[item.radial_dof_index]*radius for coefficient, item in zip(foundation_constants, inputs["born_foundations"]))
        balance = source_work-basal_work-cold_increment-cold_remainder-cold_work-born_work-foundation_work
        relative_balance = _relative(balance, (source_work, basal_work, cold_increment, cold_remainder,
            cold_work, born_work, foundation_work))
        if relative_balance > Decimal("1e-35") or min(basal_work, cold_remainder) < 0:
            raise ValueError("The complete integrated source/basal/cold/born/foundation work balance failed")
        family_work_errors = []
        for actual_step, coefficient, fields, operator in zip(original_steps, cold_coefficients, current[4], original_operators):
            expected = coefficient*fields[0]*_dot(operator[0], delta)
            actual = _d(actual_step.external_work_j)
            error = _relative(actual-expected, (actual, expected))
            if error > Decimal("5e-10"):
                raise ValueError("An original local work ledger failed its own mixed continuum increment scale")
            family_work_errors.append(float(error))
        born_family_errors = []
        for step in continuum_steps:
            work = _decimal(step.external_work_decimal)
            increment = _decimal(step.stored_energy_increment_decimal)
            remainder = _decimal(step.numerical_dissipation_decimal)
            error = _relative(work-increment-remainder, (work, increment, remainder))
            if error > Decimal("1e-50"):
                raise ValueError("An individual whole-arc energy ledger failed on its own microscopic scale")
            born_family_errors.append(float(error))
        actual_balance = source_work-basal_work-cold_increment-cold_remainder-_sum(_d(step.external_work_j) for step in original_steps)
        actual_balance -= born_work+_sum(_d(step.external_work_j) for step in foundation_steps)
        actual_scale = _sum(abs(value) for value in (source_work, basal_work, cold_increment, cold_remainder, born_work,
            *(_d(step.external_work_j) for step in original_steps+foundation_steps)))
        alias_sum = _sum(_d(value) for value in foundation_alias_bounds)
        if abs(actual_balance) > alias_sum+Decimal("5e-10")*actual_scale:
            raise ValueError("The actual stored mixed continuum ledgers failed the complete work balance")
        diagnostics = dict(source_work_j=float(source_work), basal_dissipation_j=float(basal_work),
            original_contact_work_j=float(cold_work), born_continuum_work_decimal=str(born_work),
            foundation_work_j=float(foundation_work), cold_bending_energy_increment_j=float(cold_increment),
            cold_bending_numerical_dissipation_j=float(cold_remainder),
            mechanical_work_balance_relative_error=float(relative_balance),
            actual_local_work_balance_relative_error=float(_relative(actual_balance, (actual_scale,))),
            maximum_per_coordinate_relative_residual=float(max(current[1], default=Decimal(0))),
            maximum_actual_per_coordinate_relative_residual=float(max(actual_relative, default=Decimal(0))),
            maximum_original_local_work_relative_error=max(family_work_errors, default=0.),
            maximum_born_continuum_family_work_relative_error=max(born_family_errors, default=0.),
            foundation_work_representative_alias_bounds_j=tuple(foundation_alias_bounds),
            foundation_energy_representative_alias_bounds_j=tuple(foundation_energy_alias_bounds),
            foundation_remainder_representative_alias_bounds_j=tuple(foundation_remainder_alias_bounds),
            joint_actual_local_work_representative_alias_bound_j=_upper_float(alias_sum),
            original_zero_strength_brent_endpoint_fallback_count=sum(used for _, used in old_trials),
            original_point_count=len(original_steps), original_coordinate_count=old_dimension,
            born_continuum_arc_count=len(continuum_steps), independent_foundation_count=len(foundation_steps),
            born_activation_energy_decimal=str(_sum(_decimal(item.previous_state.activation_energy_decimal)
                for item in born_continuum_attachments)), born_activation_is_cumulative_once_per_family=True,
            original_stiffness_block_unchanged=True, foundation_patches_counted_once=True,
            born_midpoint_atoms_used=False, exact_trigonometric_moment_integration=True,
            newton_iterations=iteration, line_search_evaluations=attempts,
            authoritative_coordinates="Decimal", theoretical_force_balance_uses_ideal_series_compliance=True,
            force_geometry_admission_authorized=False, whole_temporal_interval_admitted=False)
        return BornContinuumNetworkStep(representatives, _strings(coordinates),
            tuple(_upper_float(abs(_d(value)-exact)) for value, exact in zip(representatives, coordinates)),
            tuple((identity, step.state) for identity, step in zip(inputs["old_point_ids"], original_steps)),
            original_steps, foundation_steps, continuum_steps,
            tuple(tuple(map(float, fields)) for fields in current[4]), _strings(current[0]),
            tuple(map(float, current[1])), tuple(map(float, actual_relative)), tuple(sorted(diagnostics.items())), precision)


def birth_continuum_network_to_dict(step, *, born_continuum_attachments, **network_inputs):
    """Strict full-context replay packet for independent continuum force audits."""
    if type(step) is not BornContinuumNetworkStep:
        raise ValueError("A complete genuine full-arc force result is required")
    replay = solve_born_continuum_network(born_continuum_attachments=born_continuum_attachments, **network_inputs)
    if _typed_fingerprint(step) != _typed_fingerprint(replay):
        raise ValueError("The full-arc result differs from independent source/declaration/force replay")
    def declaration_body(value):
        body = {name: item for name, item in asdict(value).items() if name not in ("parameters", "reference")}
        body.update(parameters=born_contact_parameters_to_dict(value.parameters),
            reference=born_contact_reference_to_dict(value.reference))
        return body
    def state_body(value):
        body = {name: item for name, item in asdict(value).items() if name != "declaration"}
        body["declaration"] = declaration_body(value.declaration)
        return body
    body = dict(format=FORMAT, force_geometry_admission_authorized=False,
        inputs={name: value for name, value in network_inputs.items()
            if name not in ("old_materials", "old_states", "born_attachments", "born_foundations", "domain_check")},
        original_materials=[dict(columns=asdict(item.columns), law=asdict(item.law), water=item.water)
            for item in network_inputs["old_materials"]],
        original_states_before=[asdict(state) for state in network_inputs["old_states"]],
        foundations=[dict(radial_dof_index=item.radial_dof_index,
            state_before=born_foundation_state_to_dict(item.previous_state,
                expected_parameters=item.parameters, expected_reference=item.reference))
            for item in network_inputs["born_foundations"]],
        continuum_attachments=[dict(declaration=declaration_body(item.declaration), previous_state=state_body(item.previous_state))
            for item in born_continuum_attachments],
        result=dict(coordinates=step.coordinates, coordinates_decimal=step.coordinates_decimal,
            coordinate_representative_error=step.coordinate_representative_error,
            original_point_states=[(identity, asdict(state)) for identity, state in step.original_point_states],
            original_point_steps=[asdict(value) for value in step.original_point_steps],
            foundation_states=[born_foundation_state_to_dict(local.state,
                expected_parameters=item.parameters, expected_reference=item.reference)
                for item, local in zip(network_inputs["born_foundations"], step.born_foundation_steps)],
            foundation_steps=[{name: value for name, value in asdict(local).items() if name != "state"}
                for local in step.born_foundation_steps],
            continuum_steps=[dict(state=state_body(local.state), **{name: value for name, value in asdict(local).items() if name != "state"})
                for local in step.continuum_contact_steps],
            original_anchor_fields=step.original_anchor_fields, residual_decimal=step.residual_decimal,
            per_coordinate_relative_residual=step.per_coordinate_relative_residual,
            actual_per_coordinate_relative_residual=step.actual_per_coordinate_relative_residual,
            diagnostics=dict(step.diagnostics), decimal_precision=step.decimal_precision))
    return json.loads(json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False))
