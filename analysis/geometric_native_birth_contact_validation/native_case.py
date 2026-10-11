"""Declared two-atom native birth experiment on the genuine current source.

All choices precede the mixed force solve.  Each initial shared native arc
has one explicitly declared midpoint quadrature atom and its own inclined
carrier weight.  This experiment does not certify continuum quadrature,
vertex-created fronts, all-pairs coverage, or an accepted material interval.
"""
from __future__ import annotations

from fractions import Fraction as F
import hashlib
import json
import math

import numpy as np


def _i(value):
    return {name: [number.numerator, number.denominator]
        for name, number in (("lo", value.lo), ("hi", value.hi))}


def _unit_interval(values, precision=256):
    from tectonics.geometric_native_pose import interval_sqrt
    from tectonics.geometric_native_polygons import _dot
    norm = interval_sqrt(_dot(values, values), precision+8)
    if norm.lo <= 0:
        raise ValueError("Native birth point/frame is not proved nonzero")
    return tuple(value/norm for value in values)


def _representative(values):
    vector = np.asarray([float(value.midpoint) for value in values])
    vector /= math.sqrt(float(vector@vector))
    errors = tuple(max(abs(F.from_float(float(item))-bound.lo),
        abs(F.from_float(float(item))-bound.hi)) for item, bound in zip(vector, values))
    return tuple(map(float, vector)), tuple([value.numerator, value.denominator] for value in errors)


def build_native_birth_network_case(trial=None):
    """Compose unchanged cold mechanics, five HOT patches and two born atoms.

    ``trial`` may be the canonical cached genuine second-trial case.  The
    previous loaded state and actual mantle source remain independently
    authenticated.  A reduced discrete mechanical trial supplies no temporal
    native admission even when its force and energy equations balance.
    """
    from tests.test_geometric_native_implicit_underthrust_motion import get_implicit_second_trial_case
    from tests.test_geometric_native_birth_contact_support import get_birth_support_case
    from tectonics.geometric_native_birth_contact import activate_born_contact, activate_born_foundation
    from tectonics.geometric_native_birth_contact_network import BornNetworkAttachment, BornNetworkFoundation
    from tectonics.geometric_native_birth_contact_support import (
        native_birth_contact_support_to_dict, native_born_contact_references,
        native_born_foundation_references, native_birth_point_port,
    )
    from tectonics.geometric_native_implicit_mantle_loading import validate_native_implicit_physical_spatial_mantle_load
    from tectonics.geometric_native_implicit_underthrust_motion import validate_native_implicit_underthrust_motion_state, _domain
    from tectonics.geometric_material_pair_reference import _material_pair_network
    from tectonics.geometric_bending_contact_dynamics import _transverse_coordinates
    from tectonics.geometric_native_physical_history import _typed_fingerprint
    from tectonics.geometric_native_polygons import _ray_values, _plane_values

    trial = get_implicit_second_trial_case() if trial is None else trial
    previous, load = trial["before"], trial["load"]
    mesh = trial["first"]["original"]["mesh"]
    validate_native_implicit_underthrust_motion_state(previous)
    validate_native_implicit_physical_spatial_mantle_load(mesh, previous.current_surface, load)
    source = get_birth_support_case()
    support = source["support"]
    if _typed_fingerprint(support.source_epoch) != _typed_fingerprint(previous.current_surface):
        raise ValueError("Declared birth masks differ from the complete genuine accepted current material epoch")
    source_payload = native_birth_contact_support_to_dict(support)
    declaration_bytes = json.dumps(source_payload, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()
    seed = previous.initial_mechanics
    original, _, old_points, old_ports, _, old_operators = _material_pair_network(
        previous.history.origin_surface, seed, previous.owner_rotations,
        previous.transverse_displacement_m, roundoff_steps=len(previous.receipts))
    old_dimension = len(original.stiffness)
    if old_dimension != 14 or len(old_points) != 16 or previous.plate_ids != (0, 1):
        raise ValueError("Native reduced birth control requires the unchanged 16-point/14-coordinate cold source")
    radius = previous.current_surface.radius_km*1000.
    angular_count = 3*len(previous.plate_ids)
    foundation_refs = native_born_foundation_references(support)
    contact_refs = native_born_contact_references(support)
    hot_indices = {reference.patch_id: old_dimension+index for index, reference in enumerate(foundation_refs)}
    dimension = old_dimension+len(foundation_refs)
    foundations = tuple(BornNetworkFoundation(row.declaration.parameters, reference,
        activate_born_foundation(row.declaration.parameters, reference), hot_indices[reference.patch_id])
        for row, reference in zip(support.foundations, foundation_refs))
    trace_positions = {row.trace_id: index for index, row in enumerate(seed.fault_traces.traces)}
    bindings = {row.trace_id: row for row in previous.history.material_state.trace_bindings}
    radial = dict((identity, (sub, over)) for identity, sub, over in previous.transverse_displacement_m)
    angular_owners = tuple((owner, tuple(range(3*index, 3*index+3)))
        for index, owner in enumerate(previous.plate_ids))
    attachments, ports, node_audits = [], [], []
    for carrier, reference in zip(support.interface_carriers, contact_refs):
        edge = carrier.edge
        endpoints = tuple(_unit_interval(_ray_values(ray, 256)[1]) for ray in (edge.start, edge.end))
        # This analytic arc midpoint is a quadrature point only.  No removed
        # material area or per-point portion of a spring is assigned to it.
        midpoint = _unit_interval(tuple(a+b for a, b in zip(*endpoints)))
        normal = _unit_interval(_plane_values(carrier.receiver_inward_plane, 256)[1])
        point, point_errors = _representative(midpoint)
        horizontal, normal_errors = _representative(normal)
        cold_attachment = None
        if carrier.donor_component.material_kind == "original_live":
            matching = tuple(identity for identity, binding in bindings.items()
                if binding.donor_fragment_id == carrier.donor_component.original_parent_id)
            if len(matching) != 1:
                raise ValueError("Birth cold donor has no unique declared original physical bank")
            identity = matching[0]
            if "976387e77331a64ed1404a86708546d6" not in identity:
                raise ValueError("Native initial cold6 birth attachment changed its prescribed original bank")
            donor_coupling = ((angular_count+2*trace_positions[identity], 1.),)
            donor_displacement = radial[identity][0]
            cold_attachment = dict(trace_id=identity,
                original_parent_id=carrier.donor_component.original_parent_id,
                old_radial_dof_index=donor_coupling[0][0], actual_previous_displacement_m=donor_displacement,
                support="declared original donor bank; birth point on the advected R1 X native boundary")
        else:
            donor_coupling = ((hot_indices[carrier.donor_component.component_id], 1.),)
            donor_displacement = 0.
        receiver_coupling = ((hot_indices[carrier.receiver_component.component_id], 1.),)
        port = native_birth_point_port(donor_owner=carrier.donor_component.owner,
            receiver_owner=carrier.receiver_component.owner, donor_radial=point,
            receiver_radial=point, horizontal_sub_to_over=horizontal,
            dip_radians=carrier.declaration.dip_radians, radius_m=radius,
            donor_displacement_m=float(donor_displacement), receiver_displacement_m=0.,
            dimension=dimension, angular_owner_dofs=angular_owners,
            donor_radial_coupling=donor_coupling, receiver_radial_coupling=receiver_coupling)
        parameters = carrier.declaration.parameters
        attachments.append(BornNetworkAttachment(parameters, reference,
            activate_born_contact(parameters, reference), port.normal_operator, port.shear_operator))
        ports.append(port)
        node_audits.append(dict(family_id=reference.family_id,
            native_unit_arc_midpoint_bounds=[_i(value) for value in midpoint],
            world_unit_point_representative=list(point), representative_error_bounds=point_errors,
            native_receiver_inward_unit_normal_bounds=[_i(value) for value in normal],
            world_horizontal_representative=list(horizontal), horizontal_error_bounds=normal_errors,
            donor_radial_coupling=[list(row) for row in donor_coupling],
            receiver_radial_coupling=[list(row) for row in receiver_coupling],
            cold_attachment=cold_attachment, quadrature_rule="one midpoint atom per initial shared arc",
            point_has_material_area=False, whole_arc_constitutive_admission=False,
            native_first_entry_time_admission=False,
            paid_birth_gap_free_m=carrier.declaration.birth_gap_free_m))
    old_radial = _transverse_coordinates(seed.fault_traces, previous.transverse_displacement_m)
    before = tuple((0.,)*angular_count)+tuple(map(float, old_radial.reshape(-1)/radius))+(0.,)*len(foundations)
    drag = np.zeros((dimension, dimension))
    drag[:angular_count, :angular_count] = np.asarray(load.drag_nm_s)
    source_torque = tuple(map(float, load.torque_nm))+(0.,)*(dimension-angular_count)
    def domain(coordinates):
        vector = np.asarray(coordinates, dtype=float)
        _domain(previous, vector[:angular_count].reshape((-1, 3)),
            vector[angular_count:old_dimension].reshape((-1, 2))*radius)
    inputs = dict(radius_m=radius, rotation_dof_count=angular_count,
        old_point_ids=tuple(point.point_id for point in old_points), old_materials=original.materials,
        old_areas_m2=tuple(map(float, original.areas)), old_anchor_operators=tuple(old_operators),
        old_stiffness=original.stiffness, old_states=tuple(local for _, local in previous.states),
        born_attachments=tuple(attachments), born_foundations=foundations,
        coordinates_before=before, drag_nm_s=tuple(tuple(map(float, row)) for row in drag),
        source_torque_nm=source_torque, dt_s=float(trial["candidate"].dt_s), decimal_precision=120,
        domain_check=domain, max_rotation_radians=seed.max_rotation_radians)
    return dict(trial=trial, previous=previous, load=load, support=support,
        source_payload=source_payload, fixed_declaration_sha256=hashlib.sha256(declaration_bytes).hexdigest(),
        fixed_declaration_bytes=declaration_bytes, foundation_references=foundation_refs,
        contact_references=contact_refs, hot_radial_indices=hot_indices, born_ports=tuple(ports),
        original_pair_ports=old_ports, node_audits=tuple(node_audits), network_inputs=inputs,
        scope=dict(original_atoms=16, original_coordinates=14, independent_hot_foundations=5,
            born_quadrature_atoms=2, known_positive_newborn_witnesses=4, unresolved_vertex_fronts=2,
            continuum_quadrature_error_certified=False, complete_all_pairs_temporal_coverage=False,
            accepted_material_interval=False,
            physical_choice="explicit new unbonded solid-layer hydrostatic Winkler experiment",
            birth_preparation="declared negative gap paid at zero physical time; independent from old memory"))
