"""Source-preserving endpoint equilibrium for the declared curved face law.

The force rows contain the original affine ports, actual current loads,
independent HOT foundations, and the new current-area curved potential.
The constant-height born forces are used only for initialization.  A final
certificate covers every coordinate separately, including literal angular
and saved foundation representations.  This endpoint calculation neither
admits material nor supplies a three-dimensional strain-energy closure.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction as F

from .geometric_native_pose import FractionInterval as I, _rounded_interval
from .geometric_native_birth_finite_equilibrium import (
    FiniteEquilibriumCertificate, _decimal, _linear_solve,
    certify_finite_equilibrium_operands,
)


def _maxabs(value):
    return max(abs(value.lo), abs(value.hi))


def _literal(value):
    return F(Decimal(value)) if type(value) is str else F.from_float(float(value))


def map_curved_contact_covector(gradient_bounds, *, coordinate_count,
        donor_angular_indices, receiver_angular_indices, donor_radial_index,
        receiver_radial_index, radius_m):
    """Pull eight local covectors into the actual mixed-coordinate chart.

    The first six entries differentiate literal exponential owner axes; the
    final two differentiate physical top displacements in metres.  Shared
    owners are summed and radial dimensionless coordinates carry exactly R.
    """
    if (type(gradient_bounds) is not tuple or len(gradient_bounds) != 8
            or any(type(value) is not I for value in gradient_bounds)):
        raise ValueError("A complete immutable eight-coordinate interval covector is required")
    if type(coordinate_count) is not int or coordinate_count <= 0:
        raise ValueError("The actual coordinate count must be a positive integer")
    for indices in (donor_angular_indices, receiver_angular_indices):
        if (type(indices) is not tuple or len(indices) != 3 or len(set(indices)) != 3
                or any(type(index) is not int or not 0 <= index < coordinate_count for index in indices)):
            raise ValueError("Each owner requires three uniquely registered angular indices")
    for index in (donor_radial_index, receiver_radial_index):
        if type(index) is not int or not 0 <= index < coordinate_count:
            raise ValueError("Both physical radial banks must be registered")
    if donor_radial_index == receiver_radial_index:
        raise ValueError("A physical contact requires independently registered radial banks")
    if (set((donor_radial_index, receiver_radial_index))
            & (set(donor_angular_indices) | set(receiver_angular_indices))):
        raise ValueError("A physical radial bank cannot alias an angular coordinate")
    if type(radius_m) is not F or radius_m <= 0:
        raise ValueError("The physical radius must be a positive literal Fraction")
    result = [I(0) for _ in range(coordinate_count)]
    local_indices = donor_angular_indices+receiver_angular_indices+(donor_radial_index, receiver_radial_index)
    scales = (F(1),)*6+(radius_m, radius_m)
    for index, scale, value in zip(local_indices, scales, gradient_bounds):
        result[index] += scale*value
    return tuple(result)


def certify_curved_force_operands(operand_rows, *, relative_tolerance, precision=160):
    """Retain complete intervals and each coordinate's own force operands."""
    return certify_finite_equilibrium_operands(operand_rows,
        relative_tolerance=relative_tolerance, precision=precision)


def require_virgin_curved_activation_sources(attachments):
    """Reject adoption or reset of an already advanced born history."""
    from .geometric_native_birth_contact_continuum import (
        BornContinuumAttachment, validate_born_continuum_contact_state,
    )
    if (type(attachments) is not tuple or not attachments
            or any(type(item) is not BornContinuumAttachment for item in attachments)):
        raise ValueError("Complete typed first-birth attachments are required")
    for item in attachments:
        validate_born_continuum_contact_state(item.previous_state,
            expected_declaration=item.declaration)
        if (item.previous_state.step_count != 0 or item.previous_state.history
                or item.previous_state.elapsed_time_s != 0):
            raise ValueError("The new curved first-birth model cannot extend or reset adopted born history")
    return attachments


@dataclass(frozen=True, slots=True)
class CurvedFirstBirthLedger:
    family_id: str
    source_virgin_declaration: object
    activation_potential_bounds_j: I
    stored_potential_bounds_j: I
    conservative_endpoint_potential_change_bounds_j: I
    elapsed_time_s: F
    step_count: int = 1
    activation_payment_count: int = 1
    flat_activation_credit_reused: bool = False
    adopted_history_extended: bool = False

    @property
    def complete_three_dimensional_strain_energy_certified(self):
        return False

    @property
    def independently_integrated_path_work_certified(self):
        return False


@dataclass(frozen=True, slots=True)
class NativeCurvedEquilibrium:
    coordinates_decimal: tuple
    source_affine_step: object
    exact_radial_normalization: object
    source_joint_field: object
    literal_owner_axes: tuple
    omega_rad_per_myr: tuple
    responses: tuple
    mapped_curved_gradient_bounds: tuple
    force_operand_rows: tuple
    original_point_states: tuple
    original_contact_steps: tuple
    corrected_foundation_steps: tuple
    curved_first_birth_ledgers: tuple
    force_certificate: FiniteEquilibriumCertificate
    corrected_coordinate_indices: tuple
    iterations: int
    integration_precision: int
    before_state_sha256: str
    birth_support_sha256: str
    original_accepted_contact_count: int

    @property
    def force_equilibrium_certified(self):
        return self.force_certificate.equilibrium_certified

    @property
    def residual_bounds(self):
        return self.force_certificate.residual_bounds

    @property
    def operand_scales(self):
        return self.force_certificate.own_scale_lower_bounds

    @property
    def relative_residual_upper(self):
        return self.force_certificate.relative_residual_upper_bounds

    @property
    def relative_tolerance(self):
        return self.force_certificate.relative_tolerance

    @property
    def precision(self):
        return self.integration_precision

    @property
    def angular_path_error_bounds(self):
        return tuple((owner, tuple(abs(F(Decimal(self.coordinates_decimal[3*index+k]))-value)
            for k, value in enumerate(axis))) for index, (owner, axis) in enumerate(self.literal_owner_axes))

    @property
    def force_model(self):
        return "original affine ports and current loads plus shared-reference current-area curved face potentials"

    @property
    def constant_height_born_force_certificate_reused(self):
        return False

    @property
    def complete_three_dimensional_strain_energy_certified(self):
        return False

    @property
    def whole_time_interval_force_and_collision_certified(self):
        return False

    @property
    def physical_material_admission_authorized(self):
        return False

    @property
    def material_admission_authorized(self):
        return False


def _rows_certificate(rows, tolerance, precision):
    """Return None for an unbalanced trial; no trial certificate is retained."""
    try:
        return certify_curved_force_operands(rows, relative_tolerance=tolerance, precision=precision)
    except ValueError as error:
        if not str(error).startswith("Actual finite force balance exceeds coordinate "):
            raise
        return None


def _own_scales(rows):
    return tuple(sum((min(abs(value.lo), abs(value.hi)) if value.lo*value.hi > 0 else F(0)
        for value in row), F(0)) for row in rows)


def curved_force_width_scales(reference_rows, mapped_covectors):
    """Return physical lower and upper operand scales for every force row.

    Upper scales guide an initial integration request; only lower scales can
    justify its final force width.  Neither a coarse enclosure nor another
    coordinate supplies a permanent lower force-scale floor.
    """
    if (type(reference_rows) is not tuple or not reference_rows
            or any(type(row) is not tuple or not row
                or any(type(value) is not I for value in row) for row in reference_rows)
            or type(mapped_covectors) is not tuple
            or any(type(row) is not tuple or len(row) != len(reference_rows)
                or any(type(value) is not I for value in row) for row in mapped_covectors)):
        raise ValueError("Curved integration scales require all complete typed physical operand rows")
    rows = tuple(reference_rows[index]+tuple(value[index] for value in mapped_covectors)
        for index in range(len(reference_rows)))
    lower = _own_scales(rows)
    upper = tuple(sum((_maxabs(value) for value in row), F(0)) for row in rows)
    return lower, upper


def _chart_hessian(local_hessian, mapping, count):
    if (type(local_hessian) is not tuple or len(local_hessian) != 8
            or any(type(row) is not tuple or len(row) != 8
                or any(type(value) is not I for value in row) for row in local_hessian)):
        raise ValueError("A complete eight-coordinate interval Gauss-Newton tensor is required")
    result = [[F(0) for _ in range(count)] for _ in range(count)]
    for a, (i, si) in enumerate(mapping):
        for b, (j, sj) in enumerate(mapping):
            result[i][j] += si*sj*local_hessian[a][b].midpoint
    return tuple(tuple(row) for row in result)


def solve_native_curved_equilibrium(case, *, joint_field, initial_step=None,
        relative_tolerance=F(5, 10**10), precision=160, max_iterations=100,
        taylor_order=8, max_cells=4096, integration_progress=None,
        iteration_progress=None):
    """Solve actual coupled endpoint forces and certify all mixed coordinates.

    The supplied joint field is an explicit before-configuration declaration
    and must replay against the same complete authenticated support.  All
    original ports and loads are rebuilt from their native source.  Only a
    final own-coordinate force certificate authorizes returning the trial.
    """
    from .geometric_native_birth_exact_radial import normalize_native_birth_radial_before
    from .geometric_native_birth_continuum_motion import build_native_birth_continuum_inputs
    from .geometric_native_birth_contact_continuum import solve_born_continuum_network
    from .geometric_native_birth_contact_network import _advance_original_closed_inviscid
    from .geometric_native_birth_contact import advance_born_foundation
    from .geometric_native_joint_spherical_field import validate_native_joint_spherical_field, joint_spherical_component
    from .geometric_native_physical_history import _typed_fingerprint
    from .geometric_native_curved_correlated_kernel import prepare_native_curved_correlated_source
    from .geometric_native_curved_certified_integration import certified_native_curved_contact_integral
    from .thermal import SECONDS_PER_MYR
    if type(relative_tolerance) is not F or not 0 < relative_tolerance <= F(1, 10**6):
        raise ValueError("The actual curved solve requires an exact own-coordinate relative force budget")
    if type(precision) is not int or not 100 <= precision <= 512:
        raise ValueError("The actual curved force solve requires 100 to 512 declared digits")
    if type(max_iterations) is not int or not 1 <= max_iterations <= 100:
        raise ValueError("The coupled curved iteration budget must be explicit")
    previous, support = case["previous"], case["support"]
    before_fingerprint, support_fingerprint = _typed_fingerprint(previous), _typed_fingerprint(support)
    load_fingerprint = _typed_fingerprint(case["load"])
    field = validate_native_joint_spherical_field(joint_field)
    field_fingerprint = _typed_fingerprint(field)
    if _typed_fingerprint(field.support) != support_fingerprint:
        raise ValueError("The declared curved reference field differs from the complete genuine support")
    source_inputs, _ = build_native_birth_continuum_inputs(case["mesh"], previous,
        case["load"], support, case["dt_s"], decimal_precision=case["inputs"]["decimal_precision"])
    require_virgin_curved_activation_sources(source_inputs["born_continuum_attachments"])
    inputs, normalization = normalize_native_birth_radial_before(source_inputs, previous,
        decimal_precision=source_inputs["decimal_precision"])
    replay = solve_born_continuum_network(**inputs)
    if initial_step is not None and _typed_fingerprint(initial_step) != _typed_fingerprint(replay):
        raise ValueError("The supplied curved initialization differs from its complete canonical-before replay")
    seed = replay
    radius = F.from_float(inputs["radius_m"])
    before = tuple(_literal(value) for value in inputs["coordinates_before"])
    coordinates = [F(Decimal(value)) for value in seed.coordinates_decimal]
    count, old_dimension = len(coordinates), len(inputs["old_stiffness"])
    rotation_count = inputs["rotation_dof_count"]
    foundation_indices = tuple(item.radial_dof_index for item in inputs["born_foundations"])
    if foundation_indices != tuple(range(old_dimension, count)):
        raise ValueError("Every independent HOT foundation must be registered once after the original coordinates")
    physical_before = {item.coordinate_index: item.physical_before_m for item in normalization.coordinates}
    angular = {owner: tuple(range(3*index, 3*index+3)) for index, owner in enumerate(previous.plate_ids)}
    damping = tuple(tuple(F.from_float(float(value))/F.from_float(inputs["dt_s"]) for value in row)
        for row in inputs["drag_nm_s"])
    source = tuple(F.from_float(float(value)) for value in inputs["source_torque_nm"])
    stiffness = [[F(0) for _ in range(count)] for _ in range(count)]
    for index, row in enumerate(inputs["old_stiffness"]):
        stiffness[index][:old_dimension] = [F.from_float(float(value)) for value in row]
    rest = [F(0) for _ in range(count)]
    for item in inputs["born_foundations"]:
        index = item.radial_dof_index
        stiffness[index][index] = F.from_float(item.parameters.density_kg_m3)*F.from_float(
            item.parameters.gravity_m_s2)*F.from_float(item.reference.basal_area_m2)*radius**2
        rest[index] = F.from_float(item.reference.radial_birth_m)/radius
    operators = tuple(tuple(tuple(F.from_float(float(value)) for value in row)+(F(0),)*(count-old_dimension)
        for row in operator) for operator in inputs["old_anchor_operators"])
    old_offsets = tuple(tuple(F.from_float(getattr(state, name))-sum((row[j]*before[j] for j in range(count)), F(0))
        for row, name in zip(operator, ("gap_free_m", "jump_free_m")))
        for state, operator in zip(inputs["old_states"], operators))
    coefficients = tuple(F.from_float(area)/(F.from_float(material.columns.sub_length_m)/F.from_float(
        material.columns.sub_normal_modulus_pa)+F.from_float(material.columns.over_length_m)/F.from_float(
        material.columns.over_normal_modulus_pa)+1/F.from_float(material.law.normal_stiffness_pa_m))
        for area, material in zip(inputs["old_areas_m2"], inputs["old_materials"]))
    base = [[damping[i][j]+stiffness[i][j] for j in range(count)] for i in range(count)]
    for coefficient, operator in zip(coefficients, operators):
        for i in range(count):
            for j in range(count):
                base[i][j] += coefficient*operator[0][i]*operator[0][j]
    prepared, mappings, bank_indices = [], [], []
    for carrier, attachment in zip(support.interface_carriers, inputs["born_continuum_attachments"]):
        row = tuple(F(Decimal(value)) for value in attachment.declaration.normal_operator_coefficients[0])
        donor, receiver = tuple(index for index, value in enumerate(row) if value < 0), tuple(
            index for index, value in enumerate(row) if value > 0)
        if len(donor) != 1 or len(receiver) != 1:
            raise ValueError("Every curved carrier requires independently registered radial banks")
        di, ri = donor[0], receiver[0]
        donor_component = joint_spherical_component(field, carrier.donor_component.component_id)
        receiver_component = joint_spherical_component(field, carrier.receiver_component.component_id)
        if (donor_component.layer.top_radial_displacement_m != physical_before[di]
                or receiver_component.layer.top_radial_displacement_m != physical_before[ri]):
            raise ValueError("The fixed curved field roof does not match its actual saved physical bank")
        prepared.append(prepare_native_curved_correlated_source(field, carrier.declaration.family_id,
            donor_radial_before_m=physical_before[di], receiver_radial_before_m=physical_before[ri],
            precision=precision, order=28))
        mappings.append(tuple((index, F(1)) for index in angular[carrier.donor_component.owner]
            +angular[carrier.receiver_component.owner])+((di, radius), (ri, radius)))
        bank_indices.append((di, ri, carrier.donor_component.owner, carrier.receiver_component.owner))
    dt_myr = inputs["dt_s"]/SECONDS_PER_MYR
    def angular_representations(current):
        omega = tuple((owner, tuple(float(current[j])/dt_myr for j in angular[owner])) for owner in previous.plate_ids)
        axes = tuple((owner, tuple(F.from_float(value)*F.from_float(dt_myr) for value in values)) for owner, values in omega)
        literal = list(current)
        for owner, values in axes:
            for index, value in zip(angular[owner], values):
                literal[index] = value
        return omega, axes, tuple(literal)
    def old_parts(current):
        delta = tuple(value-prior for value, prior in zip(current, before))
        fields = tuple(tuple(offset[mode]+sum((operator[mode][j]*current[j] for j in range(count)), F(0))
            for mode in range(2)) for offset, operator in zip(old_offsets, operators))
        if any(fields[0] > 0 for fields in fields):
            raise ValueError("The original physical affine port gate still forbids normal opening")
        drag = tuple(sum((row[j]*delta[j] for j in range(count)), F(0)) for row in damping)
        elastic = tuple(sum((row[j]*(current[j]-rest[j]) for j in range(count)), F(0)) for row in stiffness)
        ports = tuple(tuple(coefficient*fields[0]*value for value in operator[0])
            for coefficient, fields, operator in zip(coefficients, fields, operators))
        return fields, drag, elastic, ports
    def evaluate(current, width_scale=F(1)):
        omega, axes, literal = angular_representations(current)
        owner_axes = dict(axes)
        fields, drag, elastic, ports = old_parts(current)
        reference_rows = tuple((I(drag[i]), I(elastic[i]), I(-source[i]))+tuple(I(port[i]) for port in ports)
            for i in range(count))
        responses, mapped = [], []
        hessian = [row[:] for row in base]
        for receipt, mapping, (di, ri, donor_owner, receiver_owner) in zip(prepared, mappings, bank_indices):
            response = certified_native_curved_contact_integral(receipt,
                donor_radial_increment_m=current[di]*radius-physical_before[di],
                receiver_radial_increment_m=current[ri]*radius-physical_before[ri],
                donor_owner_axis=owner_axes[donor_owner], receiver_owner_axis=owner_axes[receiver_owner],
                time_fraction=F(1), gradient_absolute_width_budgets=None,
                precision=precision, taylor_order=taylor_order, max_cells=max_cells)
            responses.append(response)
            covector = map_curved_contact_covector(response.literal_axis_and_radial_increment_gradient_bounds,
                coordinate_count=count, donor_angular_indices=angular[donor_owner],
                receiver_angular_indices=angular[receiver_owner], donor_radial_index=di,
                receiver_radial_index=ri, radius_m=radius)
            mapped.append(covector)
        def width_budgets(scales, mapping):
            return tuple(relative_tolerance*scales[index]*width_scale/
                (F(128)*scale*len(prepared)*sum(other_index == index for other_index, _ in mapping))
                for index, scale in mapping)
        # A wide coarse rectangle can bound a force many orders above its
        # actual thin active wedge.  It guides only the first refinement.
        # Recompute complete physical LOWER scales after every pass and prove
        # every local width against them before constructing a Newton tensor.
        for refinement in range(24):
            lower_scales, upper_scales = curved_force_width_scales(reference_rows, tuple(mapped))
            if all(value.width <= budget for response, mapping in zip(responses, mappings)
                    for value, budget in zip(response.literal_axis_and_radial_increment_gradient_bounds,
                        width_budgets(lower_scales, mapping))):
                break
            requested_scales = upper_scales if refinement == 0 else tuple(
                low if low else high for low, high in zip(lower_scales, upper_scales))
            for family_index, (receipt, mapping, (di, ri, donor_owner, receiver_owner)) in enumerate(
                    zip(prepared, mappings, bank_indices)):
                budgets = width_budgets(requested_scales, mapping)
                response = responses[family_index]
                if any(value.width > budget for value, budget in
                        zip(response.literal_axis_and_radial_increment_gradient_bounds, budgets)):
                    response = certified_native_curved_contact_integral(receipt,
                        donor_radial_increment_m=current[di]*radius-physical_before[di],
                        receiver_radial_increment_m=current[ri]*radius-physical_before[ri],
                        donor_owner_axis=owner_axes[donor_owner], receiver_owner_axis=owner_axes[receiver_owner],
                        time_fraction=F(1), gradient_absolute_width_budgets=budgets,
                        precision=precision, taylor_order=taylor_order, max_cells=max_cells)
                    responses[family_index] = response
                    mapped[family_index] = map_curved_contact_covector(
                        response.literal_axis_and_radial_increment_gradient_bounds,
                        coordinate_count=count, donor_angular_indices=angular[donor_owner],
                        receiver_angular_indices=angular[receiver_owner], donor_radial_index=di,
                        receiver_radial_index=ri, radius_m=radius)
                    if not response.own_coordinate_widths_satisfied:
                        raise ValueError("The actual curved contact integration exceeded a physical coordinate's force-width budget: "
                            +receipt.family_id)
        else:
            raise ValueError("Actual curved force enclosures did not resolve their own lower operand scales")
        for receipt, mapping, response in zip(prepared, mappings, responses):
            local_hessian = _chart_hessian(response.gauss_newton_hessian_bounds, mapping, count)
            hessian = [[old+new for old, new in zip(row, fresh)] for row, fresh in zip(hessian, local_hessian)]
            if integration_progress is not None:
                integration_progress(receipt.family_id, response)
        rows = tuple(reference_rows[i]+tuple(value[i] for value in mapped) for i in range(count))
        residual = tuple(_rounded_interval(sum(row, I(0)), precision) for row in rows)
        return dict(omega=omega, axes=axes, literal=literal, fields=fields, responses=tuple(responses),
            mapped=tuple(mapped), rows=rows, residual=residual, hessian=tuple(tuple(row) for row in hessian))
    corrected = set()
    current = None
    for iteration in range(max_iterations):
        current = evaluate(coordinates)
        if iteration_progress is not None:
            iteration_progress(iteration+1, current["residual"], _own_scales(current["rows"]))
        trial_certificate = _rows_certificate(current["rows"], relative_tolerance/F(4), precision)
        if trial_certificate is not None:
            break
        correction = _linear_solve(current["hessian"], tuple(-value.midpoint for value in current["residual"]), precision)
        if sum((value.midpoint*change for value, change in zip(current["residual"], correction)), F(0)) >= 0:
            raise ValueError("The coupled curved Gauss-Newton direction lost strict force descent")
        fraction, updated = F(1), None
        for _ in range(160):
            trial = tuple(F(Decimal(_decimal(value+fraction*change, precision)))
                for value, change in zip(coordinates, correction))
            if trial == tuple(coordinates):
                raise ValueError("The actual curved force budget cannot be resolved at the declared precision")
            try:
                old_parts(trial)
                inputs["domain_check"](tuple(float(value) for value in trial))
                updated = trial
                break
            except (ValueError, RuntimeError, OverflowError):
                fraction /= 2
        if updated is None:
            raise ValueError("No coupled curved update satisfied the unchanged physical source gates")
        corrected.update(index for index, (old, new) in enumerate(zip(coordinates, updated)) if old != new)
        coordinates[:] = updated
    else:
        raise ValueError("The coupled curved force solve exhausted its strict iteration budget without adopting memory")
    # Advance unchanged original histories exactly once at the new endpoint.
    original_trials = tuple(_advance_original_closed_inviscid(material, area, float(fields[0]), float(fields[1]),
        prior, inputs["dt_s"])[0] for material, area, fields, prior in zip(inputs["old_materials"],
            inputs["old_areas_m2"], current["fields"], inputs["old_states"]))
    foundation_steps = tuple(advance_born_foundation(item.previous_state, item.parameters, item.reference,
        radial_displacement_m=float(coordinates[item.radial_dof_index]*radius), dt_s=inputs["dt_s"])
        for item in inputs["born_foundations"])
    literal_fields, literal_drag, literal_elastic, literal_ports = old_parts(current["literal"])
    mechanical_fields, mechanical_drag, mechanical_elastic, mechanical_ports = old_parts(coordinates)
    final_rows = []
    for index in range(count):
        drag = I(min(mechanical_drag[index], literal_drag[index]), max(mechanical_drag[index], literal_drag[index]))
        elastic_values = [mechanical_elastic[index], literal_elastic[index]]
        if index in foundation_indices:
            elastic_values.append(F.from_float(foundation_steps[foundation_indices.index(index)].energy_gradient_n)*radius)
        row = [drag, I(min(elastic_values), max(elastic_values)), I(-source[index])]
        for area, operator, step, mechanical, literal_port in zip(inputs["old_areas_m2"], operators,
                original_trials, mechanical_ports, literal_ports):
            stored = F.from_float(area)*sum((F.from_float(traction)*coefficient
                for traction, coefficients in zip(step.traction_pa, operator) for coefficient in (coefficients[index],)), F(0))
            values = (stored, mechanical[index], literal_port[index])
            row.append(I(min(values), max(values)))
        row.extend(value[index] for value in current["mapped"])
        final_rows.append(tuple(row))
    certificate = certify_curved_force_operands(tuple(final_rows), relative_tolerance=relative_tolerance, precision=precision)
    ledgers = tuple(CurvedFirstBirthLedger(receipt.family_id, attachment.declaration,
        response.activation_potential_bounds_j, response.potential_bounds_j,
        response.conservative_endpoint_potential_change_bounds_j, F.from_float(inputs["dt_s"]))
        for receipt, attachment, response in zip(prepared, inputs["born_continuum_attachments"], current["responses"]))
    if (_typed_fingerprint(previous) != before_fingerprint or _typed_fingerprint(support) != support_fingerprint
            or _typed_fingerprint(case["load"]) != load_fingerprint or _typed_fingerprint(field) != field_fingerprint):
        raise ValueError("The coupled curved trial changed immutable accepted source or support")
    inputs["domain_check"](tuple(float(value) for value in coordinates))
    return NativeCurvedEquilibrium(tuple(_decimal(value, precision) for value in coordinates), seed, normalization,
        field, current["axes"], current["omega"], current["responses"], current["mapped"], tuple(final_rows),
        tuple((identity, step.state) for identity, step in zip(inputs["old_point_ids"], original_trials)),
        original_trials, foundation_steps, ledgers, certificate, tuple(sorted(corrected)), iteration+1, precision,
        previous.state_sha256, support.support_sha256, len(inputs["old_states"]))
