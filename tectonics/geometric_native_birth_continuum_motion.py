"""Source-bound full-arc birth mechanics on the unchanged accepted epoch.

This adapter builds every operator from the actual native shared boundaries,
retains the original sixteen histories and five independent HOT supports, and
advances their mechanical clocks together.  A force candidate is deliberately
separate from material admission: neither a balanced force nor an open HOT
spring grants permission to archive a new material point.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal, localcontext
from fractions import Fraction as F
import hashlib
import json
import math

import numpy as np
from scipy.spatial.transform import Rotation

from .geometric_native_physical_history import _typed_fingerprint


FORMAT = "native-source-bound-birth-continuum-force-candidate-1"


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()


def _decimal(value):
    if type(value) is F:
        return Decimal(value.numerator)/Decimal(value.denominator)
    return Decimal.from_float(float(value))


def _interval(value):
    return {name: [number.numerator, number.denominator]
        for name, number in (("lo", value.lo), ("hi", value.hi))}


def native_birth_continuum_frame(carrier, *, precision=256):
    """Bound the actual oriented great-circle frame without endpoint blending."""
    from .geometric_native_birth_contact_front import native_birth_arc_frame
    frame = native_birth_arc_frame(carrier, precision=precision)
    return frame.start_unit_bounds, frame.tangent_unit_bounds, frame.horizontal_unit_bounds


def _continuum_operator_rows(*, start, tangent, horizontal, radius_m,
        donor_displacement_m, dip_radians, dimension, donor_angular_indices,
        receiver_angular_indices, donor_radial_index, receiver_radial_index,
        precision):
    """Return the objective rows in the exact (1, cos(theta), sin(theta)) basis."""
    def cross(a, b):
        return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2],
            a[0]*b[1]-a[1]*b[0])
    with localcontext() as context:
        context.prec = precision
        a, b, h = (tuple(_decimal(value.midpoint) for value in row)
            for row in (start, tangent, horizontal))
        radius = _decimal(radius_m)
        lever_radius = radius+_decimal(donor_displacement_m)
        sine, cosine = map(_decimal, (math.sin(dip_radians), math.cos(dip_radians)))
        result = []
        for angular_factor, radial_factor in ((sine, cosine), (cosine, -sine)):
            rows = [[Decimal(0)]*dimension for _ in range(3)]
            rows[0][donor_radial_index] = -radius*radial_factor
            rows[0][receiver_radial_index] = radius*radial_factor
            for row, direction in zip(rows[1:], (a, b)):
                lever = tuple(lever_radius*angular_factor*value for value in cross(direction, h))
                for index, value in zip(donor_angular_indices, lever):
                    row[index] = -value
                for index, value in zip(receiver_angular_indices, lever):
                    row[index] = value
            result.append(tuple(tuple(str(value) for value in row) for row in rows))
        return tuple(result)


def build_native_birth_continuum_inputs(mesh, previous, load, support, dt_s, *,
        decimal_precision=160):
    """Build a genuine full-arc trial; caller cannot replace a source or port."""
    from .geometric_native_implicit_underthrust_motion import validate_native_implicit_underthrust_motion_state
    from .geometric_native_implicit_mantle_loading import validate_native_implicit_physical_spatial_mantle_load
    from .geometric_native_birth_contact_support import (native_birth_contact_support_to_dict,
        native_born_foundation_references, native_born_contact_references)
    from .geometric_native_birth_contact import activate_born_foundation
    from .geometric_native_birth_contact_network import BornNetworkFoundation
    from .geometric_native_birth_contact_continuum import (BornContinuumDeclaration,
        BornContinuumAttachment, activate_born_continuum_contact)
    from .geometric_material_pair_reference import _material_pair_network
    from .geometric_bending_contact_dynamics import _transverse_coordinates
    from .geometric_underthrust_motion import _domain

    validate_native_implicit_underthrust_motion_state(previous)
    validate_native_implicit_physical_spatial_mantle_load(mesh, previous.current_surface, load)
    native_birth_contact_support_to_dict(support)
    if _typed_fingerprint(support.source_epoch) != _typed_fingerprint(previous.current_surface):
        raise ValueError("Birth continuum support differs from the complete accepted material epoch")
    if type(dt_s) is not float or not math.isfinite(dt_s) or dt_s <= 0:
        raise ValueError("The genuine continuum interval needs a positive literal float clock")
    if type(decimal_precision) is not int or decimal_precision < 100:
        raise ValueError("The genuine continuum source needs at least 100 digits")
    seed = previous.initial_mechanics
    original, _, points, _, _, operators = _material_pair_network(previous.history.origin_surface,
        seed, previous.owner_rotations, previous.transverse_displacement_m,
        roundoff_steps=len(previous.receipts))
    old_dimension, angular_count = len(original.stiffness), 3*len(previous.plate_ids)
    if old_dimension != 14 or len(points) != 16 or previous.plate_ids != (0, 1):
        raise ValueError("Full-arc birth mechanics requires the unchanged original sixteen-point source")
    radius = previous.current_surface.radius_km*1000.
    references = native_born_foundation_references(support)
    if any(reference.radial_birth_m != 0. for reference in references):
        raise ValueError("This first-epoch continuum adapter requires zero radial HOT birth references")
    indices = {reference.patch_id: old_dimension+index for index, reference in enumerate(references)}
    dimension = old_dimension+len(references)
    foundations = tuple(BornNetworkFoundation(row.declaration.parameters, reference,
        activate_born_foundation(row.declaration.parameters, reference), indices[reference.patch_id])
        for row, reference in zip(support.foundations, references))
    trace_indices = {trace.trace_id: angular_count+2*index
        for index, trace in enumerate(seed.fault_traces.traces)}
    bindings = {row.trace_id: row for row in previous.history.material_state.trace_bindings}
    radial = {identity: sub for identity, sub, _ in previous.transverse_displacement_m}
    angular_indices = {owner: tuple(range(3*index, 3*index+3))
        for index, owner in enumerate(previous.plate_ids)}
    attachments, frames = [], []
    for carrier, reference in zip(support.interface_carriers, native_born_contact_references(support)):
        if carrier.donor_component.material_kind == "original_live":
            matching = tuple(identity for identity, binding in bindings.items()
                if binding.donor_fragment_id == carrier.donor_component.original_parent_id)
            if len(matching) != 1:
                raise ValueError("The full native arc donor has no unique original physical bank")
            donor_index, donor_displacement = trace_indices[matching[0]], radial[matching[0]]
        else:
            donor_index, donor_displacement = indices[carrier.donor_component.component_id], 0.
        receiver_index = indices[carrier.receiver_component.component_id]
        frame = native_birth_continuum_frame(carrier, precision=max(256, decimal_precision+32))
        normal, shear = _continuum_operator_rows(start=frame[0], tangent=frame[1], horizontal=frame[2],
            radius_m=radius, donor_displacement_m=donor_displacement,
            dip_radians=carrier.declaration.dip_radians, dimension=dimension,
            donor_angular_indices=angular_indices[carrier.donor_component.owner],
            receiver_angular_indices=angular_indices[carrier.receiver_component.owner],
            donor_radial_index=donor_index, receiver_radial_index=receiver_index,
            precision=decimal_precision)
        with localcontext() as context:
            context.prec = decimal_precision
            density = carrier.inclined_measure_density_m2_per_radian
            end = carrier.fixed_reference_interface_area_m2/density
            declaration = BornContinuumDeclaration(carrier.declaration.parameters, reference,
                "0", str(_decimal(end)), str(_decimal(density)), normal, shear,
                carrier.declaration.provenance)
        attachments.append(BornContinuumAttachment(declaration,
            activate_born_continuum_contact(declaration, decimal_precision=decimal_precision)))
        with localcontext() as context:
            context.prec = decimal_precision
            nominal_frame = tuple(tuple(str(_decimal(value.midpoint)) for value in row) for row in frame)
        frame_errors = tuple(tuple(max(abs(F(Decimal(number))-bound.lo),
            abs(F(Decimal(number))-bound.hi)) for number, bound in zip(numbers, bounds))
            for numbers, bounds in zip(nominal_frame, frame))
        frames.append(dict(family_id=reference.family_id,
            native_frame_bounds=[[_interval(value) for value in row] for row in frame],
            mechanical_frame_decimal=nominal_frame,
            mechanical_frame_error_bounds=[[[value.numerator, value.denominator] for value in row]
                for row in frame_errors],
            native_angle_bounds=_interval(carrier.native_angle_bounds_radians),
            fixed_angle_rational=[end.numerator, end.denominator],
            measure_density_rational=[density.numerator, density.denominator],
            constitutive_frame="256-digit or finer native frame midpoint with retained outward native bounds",
            interface_area_float_alias=[carrier.float_reference_interface_area_error_m2.numerator,
                carrier.float_reference_interface_area_error_m2.denominator],
            material_area_assigned_to_point=False))
    old_radial = _transverse_coordinates(seed.fault_traces, previous.transverse_displacement_m)
    before = (0.,)*angular_count+tuple(map(float, old_radial.reshape(-1)/radius))+(0.,)*len(foundations)
    drag = np.zeros((dimension, dimension))
    drag[:angular_count, :angular_count] = np.asarray(load.drag_nm_s)
    def domain(coordinates):
        vector = np.asarray(coordinates, dtype=float)
        _domain(previous, vector[:angular_count].reshape((-1, 3)),
            vector[angular_count:old_dimension].reshape((-1, 2))*radius)
    inputs = dict(radius_m=radius, rotation_dof_count=angular_count,
        old_point_ids=tuple(point.point_id for point in points), old_materials=original.materials,
        old_areas_m2=tuple(map(float, original.areas)), old_anchor_operators=tuple(operators),
        old_stiffness=original.stiffness, old_states=tuple(local for _, local in previous.states),
        born_attachments=(), born_continuum_attachments=tuple(attachments),
        born_foundations=foundations, coordinates_before=before,
        drag_nm_s=tuple(tuple(map(float, row)) for row in drag),
        source_torque_nm=tuple(map(float, load.torque_nm))+(0.,)*(dimension-angular_count),
        dt_s=dt_s, decimal_precision=decimal_precision, domain_check=domain,
        max_rotation_radians=seed.max_rotation_radians)
    return inputs, tuple(frames)


@dataclass(frozen=True, slots=True)
class NativeBirthContinuumForceCandidate:
    before_state_sha256: str
    current_load_sha256: str
    birth_support_sha256: str
    dt_s: float
    provenance: str
    network_step: object
    rotation_increment_radians: tuple
    omega_rad_per_myr: tuple
    original_point_states: tuple
    transverse_displacement_m: tuple
    free_anchor_fields: tuple
    spatial_step: object
    native_frames_payload_json: bytes
    finite_anchor_error_m: float
    relative_owner_angle_radians: float
    angular_path_error_bounds: tuple
    original_affine_anchor_representation_error_m: float

    @property
    def material_admission_authorized(self):
        return False


def solve_native_birth_continuum_candidate(mesh, previous, load, support, dt_s, *,
        provenance, decimal_precision=160):
    """Solve actual source forces and transport every original mechanical memory."""
    from .geometric_native_birth_contact_continuum import solve_born_continuum_network
    from .geometric_underthrust_motion import _domain, _sum_fields
    from .geometric_bending_contact_dynamics import _displacements
    from .geometric_spatial_underthrust import advance_spatial_underthrust_memory, TrigPolynomial, trig_polynomial_extrema
    from .geometric_material_pair_reference import material_pair_exact_anchor_fields
    from .geometric_intrinsic_geometry import RotationIncrement
    from .geometric_native_underthrust_motion import _poses
    from .geometric_fault_traces import _arc_frame
    from .thermal import SECONDS_PER_MYR
    if type(provenance) is not str or not provenance.strip():
        raise ValueError("The full-arc candidate needs explicit source provenance")
    inputs, frames = build_native_birth_continuum_inputs(mesh, previous, load, support, dt_s,
        decimal_precision=decimal_precision)
    step = solve_born_continuum_network(**inputs)
    seed, radius = previous.initial_mechanics, inputs["radius_m"]
    angular_count, old_dimension = inputs["rotation_dof_count"], len(inputs["old_stiffness"])
    coordinates = np.asarray(step.coordinates, dtype=float)
    increments = coordinates[:angular_count].reshape((-1, 3))
    radial = coordinates[angular_count:old_dimension].reshape((-1, 2))*radius
    fields = _domain(previous, increments, radial)
    spatial = advance_spatial_underthrust_memory(seed, previous.spatial_memory, fields, dt_s,
        previous_point_states=previous.states, current_point_states=step.original_point_states)
    omega = tuple((owner, tuple(map(float, row/(dt_s/SECONDS_PER_MYR))))
        for owner, row in zip(previous.plate_ids, increments))
    literal_dt = F.from_float(dt_s/SECONDS_PER_MYR)
    path_errors = tuple((owner, tuple(abs(F(Decimal(exact))-F.from_float(value)*literal_dt)
        for exact, value in zip(step.coordinates_decimal[3*index:3*index+3], values)))
        for index, (owner, values) in enumerate(omega))
    representation_bound = F(0)
    for operator in inputs["old_anchor_operators"]:
        for row in operator:
            bound = sum((abs(F.from_float(float(coefficient)))*abs(F(Decimal(exact))-F.from_float(float(value)))
                for coefficient, exact, value in zip(row, step.coordinates_decimal[:old_dimension], coordinates[:old_dimension])), F(0))
            representation_bound = max(representation_bound, bound)
    representation_error = float(representation_bound)
    if F.from_float(representation_error) < representation_bound:
        representation_error = math.nextafter(representation_error, math.inf)
    histories = tuple((owner, history+(RotationIncrement(dict(omega)[owner], dt_s/SECONDS_PER_MYR),))
        for owner, history in previous.owner_histories)
    poses = _poses(seed, histories, len(previous.receipts)+1)
    matrices = [np.asarray(matrix) for _, matrix in poses]
    angle = max((np.linalg.norm(Rotation.from_matrix(first@second.T).as_rotvec())
        for first in matrices for second in matrices), default=0.)
    if angle > seed.max_rotation_radians:
        raise ValueError("Full-arc mechanics exceeds the original finite relative-pose bound")
    displacements = _displacements(seed.fault_traces, radial)
    exact = material_pair_exact_anchor_fields(seed, poses, displacements,
        roundoff_steps=len(previous.receipts)+1)
    prepared = tuple((trace.trace_id, (trace.initial_gap_m, 0., 0., 0., 0.),
        (trace.initial_jump_m, 0., 0., 0., 0.)) for trace in seed.fault_traces.traces)
    exact, defect = _sum_fields(prepared, exact), 0.
    for trace, actual, integrated in zip(seed.fault_traces.traces, exact, fields):
        extent = _arc_frame(trace.start, trace.end)[3]
        for first, second in zip(actual[1:], integrated[1:]):
            extrema = trig_polynomial_extrema(TrigPolynomial(tuple(x-y for x, y in zip(first, second))), 0., extent)
            defect = max(defect, *(abs(value) for value in extrema))
    if math.fsum((defect, representation_error)) > previous.max_anchor_transport_error_m:
        raise ValueError("Full-arc mechanics exceeds its original finite-anchor error bound")
    plain_fields = tuple((row[0], *(field.coefficients if type(field) is TrigPolynomial else tuple(field)
        for field in row[1:])) for row in fields)
    return NativeBirthContinuumForceCandidate(previous.state_sha256, load.load_sha256,
        support.support_sha256, dt_s, provenance, step, tuple(tuple(map(float, row)) for row in increments),
        omega, step.original_point_states, displacements, plain_fields, spatial,
        _encoded(frames), float(defect), float(angle), path_errors, representation_error)


def native_birth_continuum_candidate_to_dict(candidate, *, mesh, previous, load, support):
    """Replay a complete genuine source; a stored hash alone grants no trust."""
    from .geometric_native_birth_contact_continuum import birth_continuum_network_to_dict
    from .geometric_native_implicit_underthrust_motion import native_implicit_underthrust_motion_to_dict
    from .geometric_native_implicit_mantle_loading import native_implicit_spatial_mantle_load_to_dict
    from .geometric_native_birth_contact_support import native_birth_contact_support_to_dict
    if type(candidate) is not NativeBirthContinuumForceCandidate:
        raise ValueError("A complete typed genuine full-arc candidate is required")
    replay = solve_native_birth_continuum_candidate(mesh, previous, load, support, candidate.dt_s,
        provenance=candidate.provenance, decimal_precision=candidate.network_step.decimal_precision)
    if _typed_fingerprint(candidate) != _typed_fingerprint(replay):
        raise ValueError("The full-arc candidate differs from complete source and memory replay")
    inputs, _ = build_native_birth_continuum_inputs(mesh, previous, load, support, candidate.dt_s,
        decimal_precision=candidate.network_step.decimal_precision)
    body = dict(format=FORMAT, before=native_implicit_underthrust_motion_to_dict(previous),
        current_load=native_implicit_spatial_mantle_load_to_dict(mesh, previous.current_surface, load),
        birth_support=native_birth_contact_support_to_dict(support),
        actual_network=birth_continuum_network_to_dict(candidate.network_step, **inputs),
        dt_s=candidate.dt_s, provenance=candidate.provenance,
        rotation_increment_radians=candidate.rotation_increment_radians,
        omega_rad_per_myr=candidate.omega_rad_per_myr,
        original_point_states=[(identity, asdict(state)) for identity, state in candidate.original_point_states],
        transverse_displacement_m=candidate.transverse_displacement_m,
        free_anchor_fields=candidate.free_anchor_fields, spatial_memory=asdict(candidate.spatial_step.memory),
        native_frames=json.loads(candidate.native_frames_payload_json),
        finite_anchor_error_m=candidate.finite_anchor_error_m,
        relative_owner_angle_radians=candidate.relative_owner_angle_radians,
        angular_path_error_bounds=[dict(owner=owner,
            radians=[[value.numerator, value.denominator] for value in values])
            for owner, values in candidate.angular_path_error_bounds],
        original_affine_anchor_representation_error_m=candidate.original_affine_anchor_representation_error_m,
        born_finite_pose_branch_admission_proved=False,
        mechanical_model="objective first-order original ports and declared full-arc HOT operators",
        material_admission_authorized=False, accepted_new_native_intervals=0)
    body["candidate_sha256"] = hashlib.sha256(_encoded(body)).hexdigest()
    return json.loads(_encoded(body))
