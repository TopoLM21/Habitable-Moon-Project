"""A genuine mixed original/HOT equilibrium without changing old histories.

The caller supplies every material attachment and support.  This module
provides no geometric admission.  The original elastic beam block is retained
exactly; separately registered HOT patches each have one Winkler foundation.
Contacts may share patches and may open.  Original contacts retain their
existing closed, frictionless column histories and are advanced by the old
local return-map owner after the common equilibrium has been solved.

Decimal coordinates are the authoritative mechanical solution.  Binary64
coordinate representatives are exported separately: contracting nearly
cancelling HOT operators with those representatives can lose a physically
positive compression.  No force certificate is based on that substitution.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from decimal import Decimal, localcontext
from fractions import Fraction
import itertools
import json
import math
from numbers import Real

from .geometric_contact_mechanics import LocalContactStep, solve_compressed_contact, validate_contact_state
from .geometric_native_physical_history import _typed_fingerprint


FORMAT = "native-original-and-born-contact-force-network-1"


class UnresolvedBornContactNetwork(ValueError):
    """A force, support, active-set or constitutive gate failed atomically."""


@dataclass(frozen=True, slots=True)
class BornNetworkAttachment:
    parameters: object
    reference: object
    previous_state: object
    normal_operator: tuple
    shear_operator: tuple


@dataclass(frozen=True, slots=True)
class BornNetworkFoundation:
    parameters: object
    reference: object
    previous_state: object
    radial_dof_index: int


@dataclass(frozen=True, slots=True)
class BornContactNetworkStep:
    coordinates: tuple
    coordinates_decimal: tuple
    coordinate_representative_error: tuple
    original_point_states: tuple
    original_point_steps: tuple
    born_contact_steps: tuple
    born_foundation_steps: tuple
    original_anchor_fields: tuple
    born_anchor_fields: tuple
    residual_decimal: tuple
    per_coordinate_relative_residual: tuple
    original_reaction_relative_residual: tuple
    born_reaction_relative_residual: tuple
    foundation_reaction_relative_residual: tuple
    actual_per_coordinate_relative_residual: tuple
    active_born_contacts: tuple
    diagnostics: tuple
    decimal_precision: int
    format: str = FORMAT


def _real(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real value")
    value = float(value)
    if not math.isfinite(value) or (positive and value <= 0.):
        raise ValueError(f"{name} must be finite" + (" and positive" if positive else ""))
    return value


def _vector(value, size, name):
    if not isinstance(value, tuple) or len(value) != size:
        raise ValueError(f"{name} must be an immutable tuple of length {size}")
    return tuple(_real(item, name) for item in value)


def _matrix(value, size, name):
    if not isinstance(value, tuple) or len(value) != size:
        raise ValueError(f"{name} must be an immutable square matrix")
    result = tuple(_vector(row, size, name) for row in value)
    if any(result[i][j] != result[j][i] for i in range(size) for j in range(i)):
        raise ValueError(f"{name} must be exactly symmetric")
    return result


def _d(value):
    return Decimal.from_float(float(value))


def _dot(first, second):
    return sum((a*b for a, b in zip(first, second)), Decimal(0))


def _matvec(matrix, vector):
    return [_dot(row, vector) for row in matrix]


def _sum(value):
    return sum(value, Decimal(0))


def _series(parameters):
    columns, law = parameters.columns, parameters.law
    return Decimal(1)/(_d(columns.sub_length_m)/_d(columns.sub_normal_modulus_pa)
        + _d(columns.over_length_m)/_d(columns.over_normal_modulus_pa)
        + Decimal(1)/_d(law.normal_stiffness_pa_m))


def _relative(residual, operands):
    scale = _sum(abs(value) for value in operands)
    if scale == 0:
        if residual != 0:
            raise UnresolvedBornContactNetwork("A nonzero force has no physical balancing operands")
        return Decimal(0)
    return abs(residual)/scale


def _upper_float(value):
    """An outward binary64 bound for a nonnegative Decimal quantity."""
    if value < 0 or not value.is_finite():
        raise ValueError("A finite nonnegative error bound is required")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("A finite exported error bound is required")
    if _d(result) < value:
        result = math.nextafter(result, math.inf)
    return result


def _advance_original_closed_inviscid(material, area, gap, jump, previous, dt):
    """Keep the old owner, with an exact zero-strength endpoint fallback.

    The old scalar Brent bracket occasionally has two residuals of the same
    sign because its mathematically zero endpoint shear acquires a last-bit
    subtraction residue.  Only that exact old numerical error triggers this
    adapter, and only the explicitly zero friction/cohesion/viscosity law is
    admitted.  Its analytical limit is j=p=j_free, tau=0.  The supplied
    anchors and every prior signature, clock and ledger stay unchanged.
    """
    try:
        return solve_compressed_contact(columns=material.columns, law=material.law,
            area_m2=area, gap_free_m=gap, jump_free_m=jump,
            dt_s=dt, water=material.water, previous=previous), False
    except ValueError as error:
        if str(error) != "f(a) and f(b) must have different signs":
            raise
    _strict_contact_law(material)
    columns, law = material.columns, material.law
    validate_contact_state(previous, columns=columns, law=law, area_m2=area, water=material.water)
    if gap > 0:
        raise UnresolvedBornContactNetwork("The original analytical inviscid fallback does not admit opening")
    kn, kt = columns.normal_stiffness_pa_m, columns.shear_stiffness_pa_m
    actual_gap = gap/(1.+law.normal_stiffness_pa_m/kn)
    traction = law.normal_stiffness_pa_m*actual_gap
    slip_increment = jump-previous.plastic_slip_m
    bulk = .5*area*kn*(actual_gap-gap)**2
    contact = .5*area*law.normal_stiffness_pa_m*actual_gap**2
    external = area*traction*(gap-previous.gap_free_m)
    dg = actual_gap-previous.gap_m
    dgn = (actual_gap-gap)-(previous.gap_m-previous.gap_free_m)
    djt = -(previous.jump_m-previous.jump_free_m)
    delastic = -(previous.jump_m-previous.plastic_slip_m)
    remainder = .5*area*math.fsum((kn*dgn**2, kt*djt**2,
        law.normal_stiffness_pa_m*dg**2, law.tangential_stiffness_pa_m*delastic**2))
    dbulk, dcontact = bulk-previous.bulk_energy_j, contact-previous.contact_energy_j
    balance = math.fsum((external, -dbulk, -dcontact, -remainder))
    energy_scale = max(abs(external), abs(dbulk)+abs(dcontact)+remainder,
        bulk+previous.bulk_energy_j+contact+previous.contact_energy_j, math.ulp(0.))
    energy_error = abs(balance)/energy_scale
    normal_residual = kn*(actual_gap-gap)+traction
    force_scale = max(abs(traction), kn*abs(actual_gap-gap), math.ulp(0.))
    force_error = abs(normal_residual)/force_scale
    if (not all(math.isfinite(value) for value in (actual_gap, traction, normal_residual,
            bulk, contact, external, remainder, energy_error, force_error))
            or energy_error > 2e-10 or force_error > 2e-10 or min(bulk, contact, remainder) < 0.):
        raise UnresolvedBornContactNetwork("The exact zero-strength original endpoint fallback failed its unchanged physical work/force gates")
    state = replace(previous, gap_free_m=gap, jump_free_m=jump,
        gap_m=actual_gap, jump_m=jump, plastic_slip_m=jump,
        cumulative_slip_m=math.fsum((previous.cumulative_slip_m, abs(slip_increment))),
        traction_normal_pa=traction, traction_shear_pa=0., bulk_energy_j=bulk,
        contact_energy_j=contact,
        cumulative_external_work_j=math.fsum((previous.cumulative_external_work_j, external)),
        cumulative_numerical_dissipation_j=math.fsum((previous.cumulative_numerical_dissipation_j, remainder)),
        elapsed_time_s=math.fsum((previous.elapsed_time_s, dt)), step_count=previous.step_count+1)
    validate_contact_state(state, columns=columns, law=law, area_m2=area, water=material.water)
    return LocalContactStep(state, (traction, 0.), slip_increment, 0., -traction,
        external, 0., 0., remainder, dbulk, dcontact, balance, energy_error,
        normal_residual, 0., force_error), True


def _ldl_solve(matrix, rhs):
    """Exact-form SPD LDL elimination with Decimal-rounded positive pivots.

    Small pivots are retained.  A global norm tolerance is never used to
    replace a small physical foundation or to accept its residual.
    """
    count = len(rhs)
    lower = [[Decimal(0)]*count for _ in range(count)]
    diagonal = []
    for i in range(count):
        lower[i][i] = Decimal(1)
        for j in range(i):
            numerator = matrix[i][j]-_sum(lower[i][k]*diagonal[k]*lower[j][k]
                for k in range(j))
            lower[i][j] = numerator/diagonal[j]
        pivot = matrix[i][i]-_sum(lower[i][k]**2*diagonal[k] for k in range(i))
        if not pivot.is_finite() or pivot <= 0:
            raise UnresolvedBornContactNetwork("Declared source drag and support stiffness are not positive definite at the requested Decimal precision")
        diagonal.append(pivot)
    forward = []
    for i in range(count):
        forward.append(rhs[i]-_sum(lower[i][j]*forward[j] for j in range(i)))
    backward = [Decimal(0)]*count
    for i in range(count-1, -1, -1):
        backward[i] = forward[i]/diagonal[i]-_sum(lower[j][i]*backward[j]
            for j in range(i+1, count))
    return backward, tuple(diagonal)


def _positive_semidefinite(matrix, name):
    count = len(matrix)
    lower = [[Decimal(0)]*count for _ in range(count)]
    diagonal = []
    for i in range(count):
        lower[i][i] = Decimal(1)
        for j in range(i):
            numerator = matrix[i][j]-_sum(lower[i][k]*diagonal[k]*lower[j][k] for k in range(j))
            if diagonal[j] == 0:
                if numerator != 0:
                    raise ValueError(f"{name} has an unsupported zero-pivot coupling")
            else:
                lower[i][j] = numerator/diagonal[j]
        pivot = matrix[i][i]-_sum(lower[i][k]**2*diagonal[k] for k in range(i))
        if pivot < 0:
            raise ValueError(f"{name} must be positive semidefinite")
        diagonal.append(pivot)


def _check_common_rotation(row, rotation_count):
    if rotation_count:
        for axis in range(3):
            if sum((Fraction.from_float(row[index])
                    for index in range(axis, rotation_count, 3)), Fraction(0)) != 0:
                raise ValueError("Every contact operator must exactly annihilate common angular rotation")


def _strict_contact_law(material):
    law = material.law
    law.validate()
    if any(float(getattr(law, name)) != 0. for name in
            ("friction_dry", "friction_wet", "cohesion_pa", "viscosity_pa_s_m")):
        raise ValueError("The composed closed original block requires the explicitly frictionless inviscid law")


def solve_birth_contact_network(*, radius_m, rotation_dof_count, old_point_ids,
        old_materials, old_areas_m2, old_anchor_operators, old_stiffness,
        old_states, born_attachments, born_foundations, coordinates_before,
        drag_nm_s, source_torque_nm, dt_s, decimal_precision=120,
        domain_check=None, max_rotation_radians=None):
    """Solve the joint force equations and advance independently owned ledgers.

    ``old_stiffness`` and each old operator cover the unchanged old coordinate
    prefix.  A foundation registers one extra scalar ``w/radius`` coordinate;
    all extra coordinates must be covered exactly once.  Born interface
    operators cover the complete coordinate vector, including shared HOT
    patches.  Foundation states and contact states are separate owners.

    ``coordinates_before`` uses finite real binary64 values or canonical
    Decimal strings returned by the preceding step.  Decimal strings retain
    small reference compressions between subsequent increments.
    """
    from .geometric_native_birth_contact import (
        advance_born_contact, advance_born_foundation,
        validate_born_contact_state, validate_born_foundation_state,
    )

    radius = _real(radius_m, "Radius", positive=True)
    dt = _real(dt_s, "Physical interval", positive=True)
    if isinstance(decimal_precision, bool) or not isinstance(decimal_precision, int) or decimal_precision < 80:
        raise ValueError("At least 80 explicitly declared Decimal digits are required")
    if not isinstance(old_stiffness, tuple) or not old_stiffness:
        raise ValueError("The unchanged old stiffness block is required")
    old_dimension = len(old_stiffness)
    cold_stiffness = _matrix(old_stiffness, old_dimension, "Old stiffness")
    if (isinstance(rotation_dof_count, bool) or not isinstance(rotation_dof_count, int)
            or rotation_dof_count < 0 or rotation_dof_count > old_dimension
            or rotation_dof_count % 3):
        raise ValueError("The angular prefix must contain complete owner triples")
    if not isinstance(born_attachments, tuple) or not isinstance(born_foundations, tuple):
        raise ValueError("Born network registries must be immutable tuples")
    if any(type(item) is not BornNetworkAttachment for item in born_attachments):
        raise ValueError("Each born contact needs an explicit immutable attachment")
    if any(type(item) is not BornNetworkFoundation for item in born_foundations):
        raise ValueError("Each born patch needs an independent immutable foundation")
    dimension = old_dimension+len(born_foundations)
    if not isinstance(coordinates_before, tuple) or len(coordinates_before) != dimension:
        raise ValueError("Coordinates must cover the unchanged prefix and each independent born patch")
    drag = _matrix(drag_nm_s, dimension, "Actual source drag")
    torque = _vector(source_torque_nm, dimension, "Actual source torque")
    if any(torque[index] != 0. for index in range(rotation_dof_count, dimension)):
        raise ValueError("The mantle source torque cannot prescribe transverse support forces")
    if any(drag[i][j] != 0. for i in range(dimension) for j in range(dimension)
            if i >= rotation_dof_count or j >= rotation_dof_count):
        raise ValueError("Born radial coordinates cannot acquire artificial mantle drag")
    if not all(isinstance(value, tuple) for value in
            (old_point_ids, old_materials, old_areas_m2, old_anchor_operators, old_states)):
        raise ValueError("All unchanged old network inputs must be immutable tuples")
    count = len(old_point_ids)
    if any(len(value) != count for value in (old_materials, old_areas_m2, old_anchor_operators, old_states)):
        raise ValueError("Old operators, material signatures, areas and histories must match")
    if (any(not isinstance(identity, str) or not identity for identity in old_point_ids)
            or len(set(old_point_ids)) != count):
        raise ValueError("Every original point identity must remain unique")
    cold_areas, cold_operators = [], []
    for material, area, operator, state in zip(old_materials, old_areas_m2, old_anchor_operators, old_states):
        _strict_contact_law(material)
        area = _real(area, "Original reference area", positive=True)
        validate_contact_state(state, columns=material.columns, law=material.law,
            area_m2=area, water=material.water)
        if not isinstance(operator, tuple) or len(operator) != 2:
            raise ValueError("Each original operator must contain normal and shear rows")
        rows = tuple(_vector(row, old_dimension, "Original anchor operator") for row in operator)
        for row in rows:
            _check_common_rotation(row, rotation_dof_count)
        cold_areas.append(area)
        cold_operators.append(tuple(row+(0.,)*len(born_foundations) for row in rows))
    born_operators, family_ids = [], []
    for item in born_attachments:
        validate_born_contact_state(item.previous_state, expected_parameters=item.parameters, expected_reference=item.reference)
        rows = tuple(_vector(row, dimension, "Born anchor operator")
            for row in (item.normal_operator, item.shear_operator))
        for row in rows:
            _check_common_rotation(row, rotation_dof_count)
        born_operators.append(rows)
        family_ids.append(item.reference.family_id)
    if len(set(family_ids)) != len(family_ids):
        raise ValueError("Born physical interface identities cannot repeat")
    patch_ids, indices = [], []
    for item in born_foundations:
        validate_born_foundation_state(item.previous_state, expected_parameters=item.parameters, expected_reference=item.reference)
        index = item.radial_dof_index
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError("Independent radial DOF indices must be integers")
        indices.append(index)
        patch_ids.append(item.reference.patch_id)
    if len(set(patch_ids)) != len(patch_ids) or sorted(indices) != list(range(old_dimension, dimension)):
        raise ValueError("Each independent HOT patch and extra radial DOF must be registered exactly once")
    references = tuple(item.reference for item in born_attachments+born_foundations)
    if len({(value.source_sha256, value.source_epoch_sha256) for value in references}) > 1:
        raise ValueError("All new physical references must retain the same independently supplied source and epoch context")
    patch_bindings = {(item.reference.cohort_id, item.reference.material_id,
        item.reference.parent_id, item.reference.owner) for item in born_foundations}
    for item in born_attachments:
        ref = item.reference
        banks = {(ref.donor_cohort_id, ref.donor_material_id, ref.donor_parent_id, ref.donor_owner),
            (ref.receiver_cohort_id, ref.receiver_material_id, ref.receiver_parent_id, ref.receiver_owner)}
        if not banks.intersection(patch_bindings):
            raise ValueError("Each born interface must name at least one independently registered HOT bank")
    if max_rotation_radians is not None:
        max_rotation_radians = _real(max_rotation_radians, "Rotation bound", positive=True)
    with localcontext() as context:
        context.prec = decimal_precision
        before = []
        for value in coordinates_before:
            if isinstance(value, str):
                number = Decimal(value)
                if not number.is_finite() or str(number) != value:
                    raise ValueError("Saved coordinates must use canonical finite Decimal strings")
            else:
                number = _d(_real(value, "Coordinate before"))
            before.append(number)
        radial_radius, duration = _d(radius), _d(dt)
        source = list(map(_d, torque))
        damping = [[_d(value)/duration for value in row] for row in drag]
        _positive_semidefinite(damping, "Actual source drag")
        _positive_semidefinite([[ _d(value) for value in row] for row in cold_stiffness], "Original conservative stiffness")
        stiffness = [[Decimal(0)]*dimension for _ in range(dimension)]
        for i, row in enumerate(cold_stiffness):
            for j, value in enumerate(row):
                stiffness[i][j] = _d(value)
        rest = [Decimal(0)]*dimension
        foundation_constants = []
        for item in born_foundations:
            coefficient = (_d(item.parameters.density_kg_m3)*_d(item.parameters.gravity_m_s2)
                *_d(item.reference.basal_area_m2))
            index = item.radial_dof_index
            stiffness[index][index] = coefficient*radial_radius**2
            rest[index] = _d(item.reference.radial_birth_m)/radial_radius
            expected_radial = before[index]*radial_radius
            stored_radial = _d(item.previous_state.radial_displacement_m)
            difference = abs(expected_radial-stored_radial)
            if difference > Decimal("2e-14")*max(abs(expected_radial), abs(stored_radial), Decimal("1e-300")):
                raise ValueError("Foundation coordinate and independent prior radial state disagree")
            foundation_constants.append(coefficient)
        base = [[damping[i][j]+stiffness[i][j] for j in range(dimension)] for i in range(dimension)]
        # A positive base gives a strict uniqueness certificate for every
        # unilateral active subset because every added tangent is PSD.
        _, base_pivots = _ldl_solve(base, [Decimal(0)]*dimension)
        old_ops = [[tuple(map(_d, row)) for row in operator] for operator in cold_operators]
        new_ops = [[tuple(map(_d, row)) for row in operator] for operator in born_operators]
        old_offsets = [(_d(state.gap_free_m)-_dot(operator[0], before),
            _d(state.jump_free_m)-_dot(operator[1], before))
            for state, operator in zip(old_states, old_ops)]
        new_offsets = [(_d(item.previous_state.gap_free_m)-_dot(operator[0], before),
            _d(item.previous_state.jump_free_m)-_dot(operator[1], before))
            for item, operator in zip(born_attachments, new_ops)]
        cold_constants = [_d(area)*_series(material) for area, material in zip(cold_areas, old_materials)]
        born_constants = [_d(item.reference.interface_area_m2)*_series(item.parameters)
            for item in born_attachments]
        right_base = [source[i]+_dot(damping[i], before)+_dot(stiffness[i], rest)
            for i in range(dimension)]
        def solve_subset(active):
            matrix = [row.copy() for row in base]
            rhs = right_base.copy()
            for coefficient, operator, offset in itertools.chain(
                    zip(cold_constants, old_ops, old_offsets),
                    ((born_constants[index], new_ops[index], new_offsets[index]) for index in active)):
                normal = operator[0]
                for i in range(dimension):
                    rhs[i] -= coefficient*offset[0]*normal[i]
                    for j in range(dimension):
                        matrix[i][j] += coefficient*normal[i]*normal[j]
            solution, pivots = _ldl_solve(matrix, rhs)
            gaps = [offset[0]+_dot(operator[0], solution)
                for offset, operator in zip(new_offsets, new_ops)]
            actual = tuple(index for index, gap in enumerate(gaps) if gap < 0)
            return solution, gaps, actual, pivots
        active = tuple(index for index, item in enumerate(born_attachments)
            if item.previous_state.gap_free_m < 0.)
        visited, accepted = set(), None
        for iteration in range(128):
            if active in visited:
                break
            visited.add(active)
            trial = solve_subset(active)
            if trial[2] == active:
                accepted = trial
                break
            active = trial[2]
        if accepted is None:
            if len(born_attachments) > 16:
                raise UnresolvedBornContactNetwork("The unilateral active-set iteration did not certify a unique branch")
            for bitset in itertools.product((False, True), repeat=len(born_attachments)):
                active = tuple(index for index, enabled in enumerate(bitset) if enabled)
                if active in visited:
                    continue
                trial = solve_subset(active)
                if trial[2] == active:
                    accepted = trial
                    break
        if accepted is None:
            raise UnresolvedBornContactNetwork("No complete unilateral active subset satisfied all actual gap signs")
        coordinates, born_gaps, active, pivots = accepted
        representatives = tuple(float(value) for value in coordinates)
        if not all(math.isfinite(value) for value in representatives):
            raise UnresolvedBornContactNetwork("The actual solution has no finite binary64 kinematic representative")
        if max_rotation_radians is not None:
            for start in range(0, rotation_dof_count, 3):
                if math.sqrt(math.fsum(representatives[start+axis]**2 for axis in range(3))) > max_rotation_radians:
                    raise UnresolvedBornContactNetwork("The genuine solution exceeds its explicit angular bound")
        if domain_check is not None:
            domain_check(representatives)
        delta = [now-old for now, old in zip(coordinates, before)]
        original_fields = [tuple(offset[mode]+_dot(operator[mode], coordinates) for mode in range(2))
            for offset, operator in zip(old_offsets, old_ops)]
        if any(fields[0] > 0 for fields in original_fields):
            raise UnresolvedBornContactNetwork("An original compressed contact requires unsupported opening")
        original_trials = tuple(_advance_original_closed_inviscid(material, area,
            float(fields[0]), float(fields[1]), state, dt)
            for material, area, fields, state in zip(old_materials, cold_areas, original_fields, old_states))
        original_steps = tuple(step for step, _ in original_trials)
        born_fields = [tuple(offset[mode]+_dot(operator[mode], coordinates) for mode in range(2))
            for offset, operator in zip(new_offsets, new_ops)]
        contact_steps = tuple(advance_born_contact(item.previous_state, item.parameters, item.reference,
            gap_free_m=float(fields[0]), jump_free_m=float(fields[1]), dt_s=dt)
            for item, fields in zip(born_attachments, born_fields))
        foundation_steps = tuple(advance_born_foundation(item.previous_state, item.parameters, item.reference,
            radial_displacement_m=float(coordinates[item.radial_dof_index]*radial_radius), dt_s=dt)
            for item in born_foundations)
        old_forces = [[coefficient*fields[0]*value for value in operator[0]]
            for coefficient, fields, operator in zip(cold_constants, original_fields, old_ops)]
        new_forces = [[coefficient*min(fields[0], Decimal(0))*value for value in operator[0]]
            for coefficient, fields, operator in zip(born_constants, born_fields, new_ops)]
        basal_force = _matvec(damping, delta)
        elastic_force = _matvec(stiffness, [value-reference for value, reference in zip(coordinates, rest)])
        residual = [basal_force[index]+elastic_force[index]-source[index]
            +_sum(row[index] for row in old_forces+new_forces) for index in range(dimension)]
        relative = tuple(_relative(residual[index], [basal_force[index], elastic_force[index], source[index],
            *(row[index] for row in old_forces+new_forces)]) for index in range(dimension))
        if any(value > Decimal("1e-40") for value in relative):
            raise UnresolvedBornContactNetwork("A complete per-coordinate Decimal force balance failed; increase the declared precision")
        cold_reaction_error, born_reaction_error = [], []
        for material, area, fields, coefficient, step in zip(old_materials, cold_areas, original_fields, cold_constants, original_steps):
            actual = _d(area)*_d(step.traction_pa[0])
            expected = coefficient*fields[0]
            error = _relative(actual-expected, (actual, expected))
            if error > Decimal("5e-12"):
                raise UnresolvedBornContactNetwork("An actual original constitutive normal reaction disagrees with the joint force solve")
            # The unchanged old return map may retain its own bounded
            # binary64 subtraction residue at zero-strength shear.
            shear_scale = (_d(area)*_d(material.law.tangential_stiffness_pa_m)
                *(abs(_d(step.state.jump_m))+abs(_d(step.state.plastic_slip_m))))
            if abs(_d(area)*_d(step.traction_pa[1])) > Decimal("2e-13")*shear_scale:
                raise UnresolvedBornContactNetwork("An original declared frictionless shear response is nonzero beyond its arithmetic bound")
            cold_reaction_error.append(float(error))
        for item, fields, coefficient, step in zip(born_attachments, born_fields, born_constants, contact_steps):
            actual = _d(item.reference.interface_area_m2)*_d(step.state.normal_traction_pa)
            expected = coefficient*min(fields[0], Decimal(0))
            error = _relative(actual-expected, (actual, expected))
            if error > Decimal("5e-12"):
                raise UnresolvedBornContactNetwork("An actual born normal reaction disagrees with the joint force solve")
            born_reaction_error.append(float(error))
        foundation_reaction_error = []
        actual_elastic = _matvec(stiffness, [value-reference for value, reference in zip(coordinates, rest)])
        for item, coefficient, step in zip(born_foundations, foundation_constants, foundation_steps):
            index = item.radial_dof_index
            actual = _d(step.energy_gradient_n)
            expected = coefficient*(coordinates[index]*radial_radius-_d(item.reference.radial_birth_m))
            error = _relative(actual-expected, (actual, expected))
            if error > Decimal("5e-12"):
                raise UnresolvedBornContactNetwork("An independently owned foundation reaction disagrees with its actual radial coordinate")
            foundation_reaction_error.append(float(error))
            actual_elastic[index] = actual*radial_radius
        actual_contact_forces = [[_d(area)*(_d(step.traction_pa[0])*operator[0][index]
            +_d(step.traction_pa[1])*operator[1][index]) for index in range(dimension)]
            for area, step, operator in zip(cold_areas, original_steps, old_ops)]
        actual_contact_forces += [[_d(step.energy_gradient_n[0])*operator[0][index]
            for index in range(dimension)] for step, operator in zip(contact_steps, new_ops)]
        actual_relative = []
        for index in range(dimension):
            actual_residual = basal_force[index]+actual_elastic[index]-source[index]+_sum(
                row[index] for row in actual_contact_forces)
            error = _relative(actual_residual, (basal_force[index], actual_elastic[index], source[index],
                *(row[index] for row in actual_contact_forces)))
            if error > Decimal("5e-10"):
                raise UnresolvedBornContactNetwork("An actual constitutive per-coordinate force balance failed on its own physical scale")
            actual_relative.append(float(error))
        source_work = _dot(source, delta)
        basal_work = _dot(delta, _matvec([[_d(value)/duration for value in row] for row in drag], delta))
        cold_matrix = [row[:old_dimension] for row in stiffness[:old_dimension]]
        old_prefix, new_prefix, delta_prefix = before[:old_dimension], coordinates[:old_dimension], delta[:old_dimension]
        cold_energy_increment = _dot(delta_prefix, _matvec(cold_matrix,
            [a+b for a, b in zip(new_prefix, old_prefix)]))/Decimal(2)
        cold_remainder = _dot(delta_prefix, _matvec(cold_matrix, delta_prefix))/Decimal(2)
        old_contact_work = _sum(coefficient*fields[0]*_dot(operator[0], delta)
            for coefficient, fields, operator in zip(cold_constants, original_fields, old_ops))
        born_contact_work = _sum(coefficient*min(fields[0], Decimal(0))*_dot(operator[0], delta)
            for coefficient, fields, operator in zip(born_constants, born_fields, new_ops))
        foundation_work = _sum(coefficient*(coordinates[item.radial_dof_index]*radial_radius
            -_d(item.reference.radial_birth_m))*delta[item.radial_dof_index]*radial_radius
            for coefficient, item in zip(foundation_constants, born_foundations))
        balance = source_work-basal_work-cold_energy_increment-cold_remainder-old_contact_work-born_contact_work-foundation_work
        balance_error = _relative(balance, (source_work, basal_work, cold_energy_increment, cold_remainder,
            old_contact_work, born_contact_work, foundation_work))
        if balance_error > Decimal("1e-40") or basal_work < 0 or cold_remainder < 0:
            raise UnresolvedBornContactNetwork("The exact-coordinate joint source/basal/bending/contact/foundation work balance failed")
        # Independently compare the actual local ledgers, not only algebraic
        # linear forces.  Each tiny family is checked on its own scale.
        family_work_errors = []
        for actual, expected in zip(original_steps, (
                coefficient*fields[0]*_dot(operator[0], delta)
                for coefficient, fields, operator in zip(cold_constants, original_fields, old_ops))):
            error = _relative(_d(actual.external_work_j)-expected, (_d(actual.external_work_j), expected))
            if error > Decimal("5e-10"):
                raise UnresolvedBornContactNetwork("An original local work ledger disagrees with its complete coordinate increment")
            family_work_errors.append(float(error))
        contact_alias_bounds, contact_energy_alias_bounds = [], []
        for step, item, coefficient, fields, operator in zip(contact_steps, born_attachments, born_constants, born_fields, new_ops):
            expected = coefficient*min(fields[0], Decimal(0))*_dot(operator[0], delta)
            actual = _d(step.external_work_j)
            error = _relative(actual-expected, (actual, expected))
            gap_error = abs(_d(step.state.gap_free_m)-fields[0])
            gap_delta = _dot(operator[0], delta)
            alias = coefficient*(gap_error*abs(gap_delta)
                +(abs(min(fields[0], Decimal(0)))+gap_error)*gap_error)
            arithmetic = Decimal("5e-12")*(abs(actual)+abs(expected))
            if abs(actual-expected) > alias+arithmetic:
                raise UnresolvedBornContactNetwork("An individual born contact work ledger exceeds its explicit normal anchor representative alias bound")
            old_gap = _d(item.previous_state.gap_free_m)
            old_energy = coefficient*min(old_gap, Decimal(0))**2/2
            new_energy = coefficient*min(fields[0], Decimal(0))**2/2
            energy_alias = coefficient*(abs(min(fields[0], Decimal(0)))*gap_error+gap_error**2/2)
            actual_energy = _d(step.state.normal_contact_energy_j)
            if abs(actual_energy-new_energy) > energy_alias+Decimal("5e-12")*(abs(actual_energy)+abs(new_energy)):
                raise UnresolvedBornContactNetwork("An individual born contact stored energy exceeds its explicit normal representative bound")
            expected_remainder = expected-new_energy+old_energy
            actual_remainder = _d(step.numerical_dissipation_j)
            if abs(actual_remainder-expected_remainder) > alias+energy_alias+Decimal("5e-12")*(
                    abs(actual_remainder)+abs(expected_remainder)+old_energy+new_energy):
                raise UnresolvedBornContactNetwork("An individual born contact backward-Euler remainder exceeds its explicit representative bound")
            contact_alias_bounds.append(_upper_float(alias))
            contact_energy_alias_bounds.append(_upper_float(energy_alias))
            family_work_errors.append(float(error))
        foundation_alias_bounds, foundation_energy_alias_bounds = [], []
        for step, item, coefficient in zip(foundation_steps, born_foundations, foundation_constants):
            index = item.radial_dof_index
            current = coordinates[index]*radial_radius
            old = before[index]*radial_radius
            born = _d(item.reference.radial_birth_m)
            displacement = current-old
            error_new = abs(_d(step.state.radial_displacement_m)-current)
            error_old = abs(_d(item.previous_state.radial_displacement_m)-old)
            expected = coefficient*(current-born)*displacement
            actual = _d(step.external_work_j)
            # This explicit local bound pays for independently stored binary64
            # foundation representatives.  It never deletes their Decimal DOF
            # or relaxes their force certificate by a global norm tolerance.
            alias = coefficient*(error_new*abs(displacement)
                +(abs(current-born)+error_new)*(error_new+error_old))
            arithmetic = Decimal("5e-12")*(abs(actual)+abs(expected))
            if abs(actual-expected) > alias+arithmetic:
                raise UnresolvedBornContactNetwork("An independently owned foundation work ledger exceeds its explicit radial representative alias bound")
            new_energy = coefficient*(current-born)**2/2
            old_energy = coefficient*(old-born)**2/2
            energy_alias = coefficient*(abs(current-born)*error_new+error_new**2/2)
            old_energy_alias = coefficient*(abs(old-born)*error_old+error_old**2/2)
            actual_energy = _d(step.state.foundation_energy_j)
            if abs(actual_energy-new_energy) > energy_alias+Decimal("5e-12")*(abs(actual_energy)+abs(new_energy)):
                raise UnresolvedBornContactNetwork("An independent foundation stored energy exceeds its explicit radial representative bound")
            expected_remainder = coefficient*displacement**2/2
            actual_remainder = _d(step.numerical_dissipation_j)
            if abs(actual_remainder-expected_remainder) > alias+energy_alias+old_energy_alias+Decimal("5e-12")*(
                    abs(actual_remainder)+abs(expected_remainder)+old_energy+new_energy):
                raise UnresolvedBornContactNetwork("An independent foundation backward-Euler remainder exceeds its explicit radial representative bound")
            foundation_alias_bounds.append(_upper_float(alias))
            foundation_energy_alias_bounds.append(_upper_float(energy_alias))
        actual_balance = source_work-basal_work-cold_energy_increment-cold_remainder-_sum(
            _d(step.external_work_j) for step in original_steps+contact_steps+foundation_steps)
        actual_balance_error = _relative(actual_balance, (source_work, basal_work, cold_energy_increment, cold_remainder,
            *(_d(step.external_work_j) for step in original_steps+contact_steps+foundation_steps)))
        actual_alias = _sum(_d(value) for value in contact_alias_bounds+foundation_alias_bounds)
        if abs(actual_balance) > actual_alias+Decimal("5e-10")*_sum(abs(value) for value in (
                source_work, basal_work, cold_energy_increment, cold_remainder,
                *(_d(step.external_work_j) for step in original_steps+contact_steps+foundation_steps))):
            raise UnresolvedBornContactNetwork("The independently stored actual local work ledgers failed the complete joint mechanical balance")
        diagnostics = dict(source_work_j=float(source_work), basal_dissipation_j=float(basal_work),
            original_contact_work_j=float(old_contact_work), born_contact_work_j=float(born_contact_work),
            foundation_work_j=float(foundation_work), cold_bending_energy_increment_j=float(cold_energy_increment),
            cold_bending_numerical_dissipation_j=float(cold_remainder),
            mechanical_work_balance_residual_j=float(balance), mechanical_work_balance_relative_error=float(balance_error),
            maximum_per_coordinate_relative_residual=max(map(float, relative), default=0.),
            maximum_actual_per_coordinate_relative_residual=max(actual_relative, default=0.),
            maximum_original_local_work_relative_error=max(family_work_errors[:len(original_steps)], default=0.),
            maximum_born_local_work_unadjusted_relative_error=max(family_work_errors[len(original_steps):], default=0.),
            born_contact_work_representative_alias_bounds_j=tuple(contact_alias_bounds),
            born_contact_energy_representative_alias_bounds_j=tuple(contact_energy_alias_bounds),
            foundation_work_representative_alias_bounds_j=tuple(foundation_alias_bounds),
            foundation_energy_representative_alias_bounds_j=tuple(foundation_energy_alias_bounds),
            actual_local_work_balance_residual_j=float(actual_balance),
            actual_local_work_balance_relative_error=float(actual_balance_error),
            joint_actual_local_work_representative_alias_bound_j=_upper_float(actual_alias),
            active_set_attempts=len(visited), minimum_base_pivot_decimal=str(min(base_pivots)),
            minimum_active_pivot_decimal=str(min(pivots)),
            original_point_count=count, born_contact_count=len(born_attachments), independent_foundation_count=len(born_foundations),
            original_zero_strength_brent_endpoint_fallback_count=sum(used for _, used in original_trials),
            original_fallback_changes_anchors_or_previous_histories=False,
            original_stiffness_block_unchanged=True, foundation_patches_counted_once=True,
            force_geometry_admission_authorized=False, authoritative_coordinates="Decimal",
            theoretical_force_balance_uses_ideal_series_compliance=True,
            actual_force_balance_uses_stored_constitutive_reactions=True,
            original_stiffness_energy_before_j=float(_dot(old_prefix, _matvec(cold_matrix, old_prefix))/2),
            original_stiffness_energy_after_j=float(_dot(new_prefix, _matvec(cold_matrix, new_prefix))/2),
            born_contact_energy_before_j=math.fsum(item.previous_state.normal_contact_energy_j for item in born_attachments),
            born_contact_energy_after_j=math.fsum(step.state.normal_contact_energy_j for step in contact_steps),
            foundation_energy_before_j=math.fsum(item.previous_state.foundation_energy_j for item in born_foundations),
            foundation_energy_after_j=math.fsum(step.state.foundation_energy_j for step in foundation_steps),
            born_activation_work_j=math.fsum(item.previous_state.activation_work_j for item in born_attachments),
            born_activation_work_is_cumulative_once_per_family=True,
            born_activation_work_is_interval_source_work=False)
        return BornContactNetworkStep(representatives, tuple(map(str, coordinates)),
            tuple(_upper_float(abs(_d(value)-exact)) for value, exact in zip(representatives, coordinates)),
            tuple((identity, step.state) for identity, step in zip(old_point_ids, original_steps)),
            original_steps, contact_steps, foundation_steps,
            tuple(tuple(map(float, fields)) for fields in original_fields),
            tuple(tuple(map(float, fields)) for fields in born_fields), tuple(map(str, residual)),
            tuple(map(float, relative)), tuple(cold_reaction_error), tuple(born_reaction_error),
            tuple(foundation_reaction_error), tuple(actual_relative),
            active, tuple(sorted(diagnostics.items())), decimal_precision)


def birth_contact_network_to_dict(step, **network_inputs):
    """Export full external declarations and an independently replayed solve.

    This diagnostic packet contains enough numbers for a stdlib force/work
    audit.  It confers no native temporal admission and cannot be used to
    replace the original mechanics or the caller's independently held inputs.
    """
    from .geometric_native_birth_contact import (
        born_contact_parameters_to_dict, born_contact_reference_to_dict,
        born_contact_state_to_dict, born_foundation_parameters_to_dict,
        born_foundation_reference_to_dict, born_foundation_state_to_dict,
    )
    if type(step) is not BornContactNetworkStep:
        raise ValueError("The complete genuine network step is required")
    replay = solve_birth_contact_network(**network_inputs)
    if _typed_fingerprint(step) != _typed_fingerprint(replay):
        raise ValueError("The supplied force result differs from the complete independent current input replay")
    born_contacts, born_foundations = network_inputs["born_attachments"], network_inputs["born_foundations"]
    def contact_state(value, attachment):
        return born_contact_state_to_dict(value, expected_parameters=attachment.parameters,
            expected_reference=attachment.reference)
    def foundation_state(value, attachment):
        return born_foundation_state_to_dict(value, expected_parameters=attachment.parameters,
            expected_reference=attachment.reference)
    body = dict(format=FORMAT, force_geometry_admission_authorized=False,
        inputs={name: value for name, value in network_inputs.items()
            if name not in ("old_materials", "old_states", "born_attachments", "born_foundations", "domain_check")},
        original_materials=[dict(columns=asdict(material.columns), law=asdict(material.law), water=material.water)
            for material in network_inputs["old_materials"]],
        original_states_before=[asdict(state) for state in network_inputs["old_states"]],
        born_contacts=[dict(parameters=born_contact_parameters_to_dict(item.parameters),
            reference=born_contact_reference_to_dict(item.reference),
            state_before=contact_state(item.previous_state, item),
            normal_operator=item.normal_operator, shear_operator=item.shear_operator)
            for item in born_contacts],
        born_foundations=[dict(parameters=born_foundation_parameters_to_dict(item.parameters),
            reference=born_foundation_reference_to_dict(item.reference),
            state_before=foundation_state(item.previous_state, item), radial_dof_index=item.radial_dof_index)
            for item in born_foundations],
        result=dict(coordinates=step.coordinates, coordinates_decimal=step.coordinates_decimal,
            coordinate_representative_error=step.coordinate_representative_error,
            original_point_states=[(identity, asdict(value)) for identity, value in step.original_point_states],
            original_point_steps=[asdict(value) for value in step.original_point_steps],
            born_contact_states=[contact_state(value.state, item) for value, item in zip(step.born_contact_steps, born_contacts)],
            born_foundation_states=[foundation_state(value.state, item) for value, item in zip(step.born_foundation_steps, born_foundations)],
            born_contact_steps=[{name: value for name, value in asdict(local).items() if name != "state"}
                for local in step.born_contact_steps],
            born_foundation_steps=[{name: value for name, value in asdict(local).items() if name != "state"}
                for local in step.born_foundation_steps],
            original_anchor_fields=step.original_anchor_fields, born_anchor_fields=step.born_anchor_fields,
            residual_decimal=step.residual_decimal, per_coordinate_relative_residual=step.per_coordinate_relative_residual,
            original_reaction_relative_residual=step.original_reaction_relative_residual,
            born_reaction_relative_residual=step.born_reaction_relative_residual,
            foundation_reaction_relative_residual=step.foundation_reaction_relative_residual,
            actual_per_coordinate_relative_residual=step.actual_per_coordinate_relative_residual,
            active_born_contacts=step.active_born_contacts, diagnostics=dict(step.diagnostics),
            decimal_precision=step.decimal_precision))
    return json.loads(json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False))
