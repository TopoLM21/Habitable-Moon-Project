"""Conservative full-SO(3) reaction and energy errors for a born native arc.

The unilateral pressure map is globally Lipschitz, including opening and
closing.  These bounds therefore retain narrow closing tails without assuming
that the affine active set survives a finite rotation.  The actual exponential
Jacobian is differentiated with a rational Taylor remainder; receiving normal
rotation, radial ports, native frame intervals and measure aliases are retained.
An explicit per-coordinate residual budget is required for a force certificate.
No result here authorizes a material transaction or a three-dimensional layer.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction as F
from math import factorial

from .geometric_native_birth_finite_pose import BornFinitePoseFields
from .geometric_native_pose import FractionInterval as I, _rounded_interval, interval_pi


def _maximum(value):
    return max(abs(value.lo), abs(value.hi))


def _fraction(value, name, *, positive=False, nonnegative=False):
    if type(value) is not F or (positive and value <= 0) or (nonnegative and value < 0):
        raise ValueError(name + " requires an exact rational physical value")
    return value


def _clock(value):
    if type(value) is not I or not 0 <= value.lo <= value.hi <= 1:
        raise ValueError("The finite force clock must be an exact interval in [0, 1]")
    return value


def _budget(precision, order):
    if type(precision) is not int or not 64 <= precision <= 1024:
        raise ValueError("Finite force precision must be a literal integer from 64 to 1024")
    if type(order) is not int or not 8 <= order <= 64:
        raise ValueError("Finite force Taylor order must be a literal integer from 8 to 64")


def _cross(left, right):
    return (left[1]*right[2]-left[2]*right[1],
        left[2]*right[0]-left[0]*right[2], left[0]*right[1]-left[1]*right[0])


def _dot(left, right):
    return sum((a*b for a, b in zip(left, right)), I(0))


def _compress_vector(vector, precision):
    return tuple(_rounded_interval(value, precision+16) for value in vector)


def _polynomial(coefficients, time, tail, precision):
    value = I(0)
    for coefficient in reversed(coefficients):
        value = _rounded_interval(value*time+coefficient, precision+16)
    return _rounded_interval(value+I(-tail, tail), precision+16)


def _rotated_and_derivatives(vector, axis, time, precision, order):
    """Bound exp([axis]u)v and d exp([beta])v / d beta_i at beta=axis*u.

If x_n=[axis]^n v/n!, its Frechet derivative coefficient is
d_n=([e_i]x_(n-1)+[axis]d_(n-1))/n.  The latter multiplies
u^(n-1), since the differentiated coordinate is beta, rather than axis.
The induced l1 norm gives |d_n| <= |v|_1 |axis|_1^(n-1)/(n-1)!.
"""
    rate = sum(map(abs, axis), F(0))
    maximum_angle = rate*time.hi
    if maximum_angle/F(order+1) >= 1:
        raise ValueError("The actual angular axis exceeds its Taylor remainder budget")
    size = sum((_maximum(value) for value in vector), F(0))
    rotation_tail = size*maximum_angle**(order+1)/factorial(order+1)/(1-maximum_angle/F(order+2))
    derivative_tail = size*maximum_angle**order/factorial(order)/(1-maximum_angle/F(order+1))
    rotation = [vector]
    derivatives = [[] for _ in range(3)]
    previous_derivatives = [(I(0),)*3 for _ in range(3)]
    previous = vector
    for degree in range(1, order+1):
        following = _compress_vector(tuple(value/degree for value in _cross(axis, previous)), precision)
        for component in range(3):
            unit = tuple(F(int(index == component)) for index in range(3))
            first = _cross(unit, previous)
            second = _cross(axis, previous_derivatives[component])
            derivative = _compress_vector(tuple((a+b)/degree for a, b in zip(first, second)), precision)
            derivatives[component].append(derivative)
            previous_derivatives[component] = derivative
        rotation.append(following)
        previous = following
    result = tuple(_polynomial(tuple(row[index] for row in rotation), time, rotation_tail, precision)
        for index in range(3))
    jacobian = tuple(tuple(_polynomial(tuple(row[index] for row in rows), time, derivative_tail, precision)
        for index in range(3)) for rows in derivatives)
    return result, jacobian


def _trig_product(left_cosine, left_sine, right_constant, right_cosine, right_sine):
    """Coefficients of the vector dot product in the complete five-term basis."""
    aa, bb = _dot(left_cosine, right_cosine), _dot(left_sine, right_sine)
    return ((aa+bb)/2, _dot(left_cosine, right_constant), _dot(left_sine, right_constant),
        (aa-bb)/2, (_dot(left_cosine, right_sine)+_dot(left_sine, right_cosine))/2)


def _field_bound(coefficients, angle, precision):
    from .geometric_native_contact_time_support import _trig_ranges
    if type(angle) is not I or angle.lo < 0 or angle.hi >= interval_pi(precision).lo:
        raise ValueError("A finite reaction bound needs a proved nonnegative native minor arc")
    sine, cosine = _trig_ranges(angle.lo, angle.hi, precision)
    sine2, cosine2 = _trig_ranges(2*angle.lo, 2*angle.hi, precision)
    return _rounded_interval(coefficients[0]+cosine*coefficients[1]+sine*coefficients[2]
        +cosine2*coefficients[3]+sine2*coefficients[4], precision)


@dataclass(frozen=True, slots=True)
class BornFinitePoseJacobianBounds:
    declaration: object
    coefficients: tuple
    angle_bounds: I
    time_fraction_bounds: I
    affine_operator_norm_bounds: tuple
    finite_operator_norm_bounds: tuple
    operator_error_bounds: tuple
    precision: int
    order: int

    def __post_init__(self):
        from .geometric_native_birth_contact_continuum import BornContinuumDeclaration
        if type(self.declaration) is not BornContinuumDeclaration:
            raise ValueError("The finite Jacobian requires its complete continuum declaration")
        self.declaration.__post_init__()
        _budget(self.precision, self.order)
        _clock(self.time_fraction_bounds)
        count = len(self.declaration.normal_operator_coefficients[0])
        if (type(self.coefficients) is not tuple or len(self.coefficients) != 5
                or any(type(row) is not tuple or len(row) != count or any(type(value) is not I for value in row)
                    for row in self.coefficients)):
            raise ValueError("The actual finite Jacobian needs all five coefficient rows and every coordinate")
        for row in (self.affine_operator_norm_bounds, self.finite_operator_norm_bounds, self.operator_error_bounds):
            if type(row) is not tuple or len(row) != count or any(type(value) is not F or value < 0 for value in row):
                raise ValueError("Every Jacobian coordinate requires exact nonnegative operator bounds")
        for index in range(count):
            finite = tuple(row[index] for row in self.coefficients)
            affine = tuple(I(F(Decimal(row[index]))) for row in self.declaration.normal_operator_coefficients)+(I(0), I(0))
            required = (_maximum(_field_bound(affine, self.angle_bounds, self.precision)),
                _maximum(_field_bound(finite, self.angle_bounds, self.precision)),
                _maximum(_field_bound(tuple(now-old for now, old in zip(finite, affine)), self.angle_bounds, self.precision)))
            if any(actual < bound for actual, bound in zip((self.affine_operator_norm_bounds[index],
                    self.finite_operator_norm_bounds[index], self.operator_error_bounds[index]), required)):
                raise ValueError("A stored Jacobian bound cannot omit its actual coefficient uncertainty")


def bound_born_finite_pose_jacobian(declaration, *, start_unit_bounds, tangent_unit_bounds,
        horizontal_unit_bounds, donor_axis, receiver_axis, radius_m,
        donor_radial_before_m, donor_radial_increment_m, sine_dip, cosine_dip,
        donor_angular_indices, receiver_angular_indices, donor_radial_index,
        receiver_radial_index, angle_bounds, time_fraction_bounds, precision=128, order=32):
    """Differentiate the real finite gap with respect to all mixed coordinates.

Angular coordinates are literal SO(3) rotation-vector components. Radial
coordinates are w/R. The receiving normal follows its own owner; it must be
differentiated even when the donor is stationary. The native frame intervals
are not normalized, flattened or replaced by their mechanical midpoints.
"""
    from .geometric_native_birth_contact_continuum import BornContinuumDeclaration
    if type(declaration) is not BornContinuumDeclaration:
        raise ValueError("The actual continuum declaration is required")
    declaration.__post_init__()
    _budget(precision, order)
    time = _clock(time_fraction_bounds)
    vectors = (start_unit_bounds, tangent_unit_bounds, horizontal_unit_bounds)
    if any(type(row) is not tuple or len(row) != 3 or any(type(value) is not I for value in row) for row in vectors):
        raise ValueError("Full native frame interval triples are required")
    if any(type(axis) is not tuple or len(axis) != 3 or any(type(value) is not F for value in axis)
            for axis in (donor_axis, receiver_axis)):
        raise ValueError("The actual angular axes must be exact rational triples")
    for number, name in ((radius_m, "Radius"), (donor_radial_before_m, "Prior donor radial placement"),
            (donor_radial_increment_m, "Donor radial increment"), (sine_dip, "Dip sine"), (cosine_dip, "Dip cosine")):
        _fraction(number, name, positive=name == "Radius")
    dimension = len(declaration.normal_operator_coefficients[0])
    angular = (donor_angular_indices, receiver_angular_indices)
    if any(type(row) is not tuple or len(row) != 3 for row in angular):
        raise ValueError("Both actual angular bank index triples are required")
    indices = donor_angular_indices+receiver_angular_indices+(donor_radial_index, receiver_radial_index)
    if any(type(index) is not int or not 0 <= index < dimension for index in indices) or len(set(indices)) != 8:
        raise ValueError("The actual angular and radial bank coordinates must be distinct and in range")
    # Evaluating here also validates the complete angular domain, including
    # its strict minor-arc upper bound.
    _field_bound((I(0),)*5, angle_bounds, precision)
    a, b, h = vectors
    da, dda = _rotated_and_derivatives(a, donor_axis, time, precision, order)
    db, ddb = _rotated_and_derivatives(b, donor_axis, time, precision, order)
    ra, dra = _rotated_and_derivatives(a, receiver_axis, time, precision, order)
    rb, drb = _rotated_and_derivatives(b, receiver_axis, time, precision, order)
    rh, drh = _rotated_and_derivatives(h, receiver_axis, time, precision, order)
    normal = (tuple(sine_dip*value for value in rh),
        tuple(cosine_dip*value for value in ra), tuple(cosine_dip*value for value in rb))
    lever = I(radius_m+donor_radial_before_m)+time*donor_radial_increment_m
    if lever.lo <= 0:
        raise ValueError("The actual donor radial placement cannot cross the sphere centre")
    columns = [(I(0),)*5 for _ in range(dimension)]
    for component, index in enumerate(donor_angular_indices):
        columns[index] = tuple(_rounded_interval(-lever*value, precision)
            for value in _trig_product(dda[component], ddb[component], *normal))
    for component, index in enumerate(receiver_angular_indices):
        derivative_normal = (tuple(sine_dip*value for value in drh[component]),
            tuple(cosine_dip*value for value in dra[component]), tuple(cosine_dip*value for value in drb[component]))
        columns[index] = tuple(_rounded_interval(-lever*value, precision)
            for value in _trig_product(da, db, *derivative_normal))
    columns[donor_radial_index] = tuple(_rounded_interval(-radius_m*value, precision)
        for value in _trig_product(da, db, *normal))
    # The receiving radius and its normal rotate together, so their product
    # is exactly invariant. Avoid an unnecessary whole-clock cancellation.
    initial_normal = (tuple(sine_dip*value for value in h),
        tuple(cosine_dip*value for value in a), tuple(cosine_dip*value for value in b))
    columns[receiver_radial_index] = tuple(_rounded_interval(radius_m*value, precision)
        for value in _trig_product(a, b, *initial_normal))
    coefficients = tuple(tuple(column[basis] for column in columns) for basis in range(5))
    affine_norms, finite_norms, errors = [], [], []
    for index, column in enumerate(columns):
        affine = tuple(I(F(Decimal(row[index]))) for row in declaration.normal_operator_coefficients)+(I(0), I(0))
        affine_norms.append(_maximum(_field_bound(affine, angle_bounds, precision)))
        finite_norms.append(_maximum(_field_bound(column, angle_bounds, precision)))
        errors.append(_maximum(_field_bound(tuple(now-old for now, old in zip(column, affine)), angle_bounds, precision)))
    return BornFinitePoseJacobianBounds(declaration, coefficients, angle_bounds, time,
        tuple(affine_norms), tuple(finite_norms), tuple(errors), precision, order)


def _normal_stiffness(parameters):
    columns, law = parameters.columns, parameters.law
    exact = lambda value: F.from_float(float(value))
    return 1/(exact(columns.sub_length_m)/exact(columns.sub_normal_modulus_pa)
        +exact(columns.over_length_m)/exact(columns.over_normal_modulus_pa)
        +1/exact(law.normal_stiffness_pa_m))


def _validate_attachment_step(attachment, step):
    from .geometric_native_birth_contact_continuum import (BornContinuumAttachment,
        BornContinuumContactStep, validate_born_continuum_contact_state, _advance_field)
    from .geometric_native_physical_history import _typed_fingerprint
    if type(attachment) is not BornContinuumAttachment or type(step) is not BornContinuumContactStep:
        raise ValueError("The complete actual continuum attachment and contact step are required")
    previous, current = attachment.previous_state, step.state
    validate_born_continuum_contact_state(previous, expected_declaration=attachment.declaration)
    if (current.step_count != previous.step_count+1 or current.history[:-1] != previous.history
            or current.decimal_precision != previous.decimal_precision):
        raise ValueError("The actual continuum step must extend its complete prior history once")
    with localcontext() as context:
        context.prec = current.decimal_precision
        replay = _advance_field(previous, tuple(map(Decimal, current.gap_coefficients_decimal)),
            tuple(map(Decimal, current.jump_coefficients_decimal)), current.history[-1].dt_s, current.decimal_precision)
    if _typed_fingerprint(step) != _typed_fingerprint(replay):
        raise ValueError("The contact step differs from its own complete field and energy replay")


def _rational_moments(low, high, precision):
    """Independent outward moments; no Decimal stiffness or trigonometry."""
    from .geometric_native_contact_time_support import _trig_ranges
    width, center = high-low, (high+low)/2
    point = lambda value: tuple(_rounded_interval(item, precision) for item in _trig_ranges(value, value, precision))
    sh, _ = point(width/2)
    sc, cc = point(center)
    sw, _ = point(width)
    s2, c2 = point(2*center)
    return tuple(tuple(_rounded_interval(value, precision) for value in row) for row in
        ((I(width), 2*cc*sh, 2*sc*sh),
         (2*cc*sh, (I(width)+c2*sw)/2, s2*sw/2),
         (2*sc*sh, s2*sw/2, (I(width)-c2*sw)/2)))


def _affine_reference_bounds(declaration, step, precision):
    """Enclose the complete affine unilateral integral, independently of its active set.

Reported part/band endpoints serve only as useful partition knots. Every cell
gets a new rational sign proof. Closed cells use exact-stiffness analytic
moments, open cells have zero pressure, and unresolved cells retain direct
clipped-pressure bounds. Thus a missing tiny pocket or Decimal arithmetic
rounding is never assumed to have zero reaction or potential.
"""
    start, end = (F(Decimal(value)) for value in
        (declaration.angle_start_decimal, declaration.angle_end_decimal))
    knots = {start, end}
    for rows in (step.closed_angle_parts_decimal, step.root_bands_decimal):
        for low, high in rows:
            low, high = F(Decimal(low)), F(Decimal(high))
            if not start <= low <= high <= end:
                raise ValueError("An affine integration knot is outside its declared complete arc")
            knots.update((low, high))
    knots = sorted(knots)
    gap = tuple(F(Decimal(value)) for value in step.state.gap_coefficients_decimal)
    coefficients = tuple(I(value) for value in gap)+(I(0), I(0))
    derivative = (I(0), I(gap[2]), I(-gap[1]), I(0), I(0))
    operators = tuple(tuple(F(Decimal(value)) for value in row) for row in declaration.normal_operator_coefficients)
    count = len(operators[0])
    factor = F(Decimal(declaration.measure_density_decimal))*_normal_stiffness(declaration.parameters)
    energy, gradient = I(0), [I(0)]*count
    for low, high in zip(knots, knots[1:]):
        angle = I(low, high)
        gap_bounds = _field_bound(coefficients, angle, precision)
        derivative_bounds = _field_bound(derivative, angle, precision)
        if derivative_bounds.lo >= 0 or derivative_bounds.hi <= 0:
            left, right = (_field_bound(coefficients, I(point), precision) for point in (low, high))
            gap_bounds = I(min(left.lo, right.lo), max(left.hi, right.hi))
        if gap_bounds.lo >= 0:
            continue
        if gap_bounds.hi <= 0:
            moments = _rational_moments(low, high, precision)
            weighted = tuple(_rounded_interval(sum((value*coefficient for value, coefficient in zip(row, gap)), I(0)), precision)
                for row in moments)
            local_energy = _rounded_interval(factor*sum((value*coefficient for value, coefficient in zip(weighted, gap)), I(0))/2, precision)
            local_gradient = tuple(_rounded_interval(factor*sum((operators[basis][index]*weighted[basis]
                for basis in range(3)), I(0)), precision) for index in range(count))
        else:
            pressure = I(min(gap_bounds.lo, F(0)), min(gap_bounds.hi, F(0)))
            minimum, maximum = max(F(0), -gap_bounds.hi), max(F(0), -gap_bounds.lo)
            local_energy = factor*(high-low)*I(minimum**2, maximum**2)/2
            local_gradient = tuple(factor*(high-low)*pressure*_field_bound(
                tuple(I(row[index]) for row in operators)+(I(0), I(0)), angle, precision)
                for index in range(count))
        # Nonnegative potential is an exact mathematical property. Clamping
        # a negative lower enclosure preserves every possible positive signal.
        local_energy = I(max(F(0), local_energy.lo), max(F(0), local_energy.hi))
        energy = _rounded_interval(energy+local_energy, precision)
        gradient = [_rounded_interval(before+local, precision) for before, local in zip(gradient, local_gradient)]
    return energy, tuple(gradient)


@dataclass(frozen=True, slots=True)
class BornFinitePoseForceErrorBounds:
    finite_gap_bounds_m: I
    affine_gap_bounds_m: I
    gap_error_bound_m: F
    pressure_error_bound_pa: F
    finite_pressure_bounds_pa: I
    finite_area_upper_m2: F
    measure_alias_bound_m2: F
    affine_reaction_reference_error_bounds: tuple
    affine_energy_reference_error_bound_j: F
    reaction_error_bounds: tuple
    finite_potential_gradient_bounds: tuple
    energy_error_bound_j: F
    finite_energy_bounds_j: I
    time_fraction_bounds: I
    angle_bounds: I

    @property
    def material_admission_authorized(self):
        return False


def bound_born_finite_pose_force_error(fields, attachment, step, jacobian, *,
        finite_measure_density_m2_per_radian, finite_angle_end_bounds,
        affine_operator_norm_bounds=None):
    """Bound the constitutive error, without preserving an assumed active set.

For p(g)=k min(g,0), |p_f-p_a| <= k |g_f-g_a|. Therefore each
reaction error is at most A k (delta_g |Dg_a| + |min(g_f,0)| delta_Dg).
The energy error is bounded by A k delta_g (|g_f^-|+|g_a^-|)/2.
    The finite native density/end need not equal their Decimal affine aliases.
    Their measure difference is added separately. A complete independent
    rational affine integral also pays for stored Decimal arithmetic and all
    uncertain branches, including the reported root bands. A whole-clock
    field is compared with the stored solved endpoint, so its errors may be
    wider than an endpoint force error; interpolation never changes that
    reference gradient or energy.
"""
    from .geometric_native_physical_history import _typed_fingerprint
    if type(fields) is not BornFinitePoseFields or type(jacobian) is not BornFinitePoseJacobianBounds:
        raise ValueError("Actual typed finite-pose fields and their Jacobian bounds are required")
    fields.__post_init__()
    jacobian.__post_init__()
    _validate_attachment_step(attachment, step)
    declaration = attachment.declaration
    if (_typed_fingerprint(jacobian.declaration) != _typed_fingerprint(declaration)
            or jacobian.time_fraction_bounds != fields.time_fraction_bounds):
        raise ValueError("The finite gap and actual operator must bind the same source and clock")
    density = _fraction(finite_measure_density_m2_per_radian, "Actual native measure density", positive=True)
    if type(finite_angle_end_bounds) is not I:
        raise ValueError("The actual native angle endpoint needs exact outward bounds")
    start, affine_end = map(lambda value: F(Decimal(value)),
        (declaration.angle_start_decimal, declaration.angle_end_decimal))
    if finite_angle_end_bounds.lo <= start:
        raise ValueError("The actual native interface must have strictly positive length")
    angle = I(start, max(affine_end, finite_angle_end_bounds.hi))
    if jacobian.angle_bounds != angle:
        raise ValueError("The Jacobian must cover the complete finite and affine reference arcs")
    precision = min(fields.precision, jacobian.precision)
    finite_gap = _field_bound(fields.gap_coefficients, angle, precision)
    after = tuple(F(Decimal(value)) for value in step.state.gap_coefficients_decimal)
    # The stored gradient and potential below are the solved endpoint values.
    # Compare a whole-clock enclosure against that same final affine field;
    # interpolating it while retaining endpoint gradient/energy would be wrong.
    affine = tuple(I(value) for value in after)+(I(0), I(0))
    affine_gap = _field_bound(affine, angle, precision)
    gap_error = _maximum(_field_bound(tuple(now-old for now, old in zip(fields.gap_coefficients, affine)), angle, precision))
    norms = jacobian.affine_operator_norm_bounds
    if affine_operator_norm_bounds is not None:
        if (type(affine_operator_norm_bounds) is not tuple or len(affine_operator_norm_bounds) != len(norms)
                or any(type(value) is not F or value < actual for value, actual in zip(affine_operator_norm_bounds, norms))):
            raise ValueError("An independent objective operator bound cannot reduce the proven actual norm")
        norms = affine_operator_norm_bounds
    stiffness = _normal_stiffness(declaration.parameters)
    finite_negative = max(F(0), -finite_gap.lo)
    affine_negative = max(F(0), -affine_gap.lo)
    finite_area = density*(finite_angle_end_bounds.hi-start)
    affine_density = F(Decimal(declaration.measure_density_decimal))
    endpoint_error = max(abs(finite_angle_end_bounds.lo-affine_end), abs(finite_angle_end_bounds.hi-affine_end))
    measure_alias = abs(density-affine_density)*(angle.hi-start)+max(density, affine_density)*endpoint_error
    reference_precision = max(fields.precision, jacobian.precision, step.state.decimal_precision+32)
    affine_energy, affine_gradient = _affine_reference_bounds(declaration, step, reference_precision)
    gradients = tuple(F(Decimal(value)) for value in step.gradient_decimal)
    reference_errors = tuple(max(abs(value-enclosure.lo), abs(value-enclosure.hi))
        for value, enclosure in zip(gradients, affine_gradient))
    energy = F(Decimal(step.state.stored_energy_decimal))
    reference_energy_error = max(abs(energy-affine_energy.lo), abs(energy-affine_energy.hi))
    errors = tuple(finite_area*stiffness*(gap_error*norm+finite_negative*operator_error)
        +measure_alias*stiffness*affine_negative*norm+reference
        for norm, operator_error, reference in zip(norms, jacobian.operator_error_bounds, reference_errors))
    if any(value < 0 for value in errors):
        raise ValueError("Finite force errors must be nonnegative")
    energy_error = (finite_area*stiffness*gap_error*(finite_negative+affine_negative)/2
        +measure_alias*stiffness*affine_negative**2/2+reference_energy_error)
    return BornFinitePoseForceErrorBounds(finite_gap, affine_gap, gap_error,
        stiffness*gap_error, I(stiffness*min(finite_gap.lo, F(0)), stiffness*min(finite_gap.hi, F(0))),
        finite_area, measure_alias, reference_errors, reference_energy_error, errors,
        tuple(I(value-error, value+error) for value, error in zip(gradients, errors)),
        energy_error, I(max(F(0), energy-energy_error), energy+energy_error),
        fields.time_fraction_bounds, angle)


@dataclass(frozen=True, slots=True)
class BornFinitePoseForceBudget:
    finite_absolute_residual_bounds: tuple
    maximum_absolute_residual: tuple
    equilibrium_certified: bool

    @property
    def material_admission_authorized(self):
        return False


def certify_born_finite_pose_force_budget(bounds, *, existing_absolute_residual_bounds,
        maximum_absolute_residual):
    """Add the new errors to explicit prior residuals; wide bounds refuse PASS."""
    if type(bounds) is not BornFinitePoseForceErrorBounds:
        raise ValueError("The complete finite constitutive error bounds are required")
    count = len(bounds.reaction_error_bounds)
    if type(bounds.reaction_error_bounds) is not tuple or any(type(value) is not F or value < 0
            for value in bounds.reaction_error_bounds):
        raise ValueError("Every finite reaction error requires an exact nonnegative bound")
    for row in (existing_absolute_residual_bounds, maximum_absolute_residual):
        if type(row) is not tuple or len(row) != count or any(type(value) is not F or value < 0 for value in row):
            raise ValueError("Every actual coordinate needs its own explicit nonnegative rational force budget")
    residuals = tuple(old+new for old, new in zip(existing_absolute_residual_bounds, bounds.reaction_error_bounds))
    return BornFinitePoseForceBudget(residuals, maximum_absolute_residual,
        all(value <= limit for value, limit in zip(residuals, maximum_absolute_residual)))
