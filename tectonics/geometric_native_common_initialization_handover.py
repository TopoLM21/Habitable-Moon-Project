"""Exact source-to-declared-reference initialization diagnostics.

This module reads authenticated saved fields.  It neither replays a force
solve nor assigns missing historical continuum or normal-fracture data.
All new residual arithmetic uses unrounded rational interval endpoints.
"""
from dataclasses import dataclass, fields, replace
from fractions import Fraction as F
import hashlib
import json

SCOPE = "AUTHENTICATED_SOURCE_TO_DECLARED_REFERENCE_INITIALIZATION_DIAGNOSTICS_ONLY"
_SYSTEMS = {}
_CONTEXTS = {}
_RESPONSES = {}


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def _digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _source_api():
    from . import geometric_native_common_initialization_source as api
    return api


def _fraction(raw):
    if (type(raw) is not list or len(raw) != 2
            or any(type(x) is not int for x in raw) or raw[1] <= 0):
        raise ValueError("An exact saved scalar requires a canonical integer fraction")
    value = F(*raw)
    if [value.numerator, value.denominator] != raw:
        raise ValueError("A saved exact scalar is not canonical")
    return value


@dataclass(frozen=True, slots=True)
class _I:
    lo: F
    hi: F

    def __post_init__(self):
        if type(self.lo) is not F or type(self.hi) is not F or self.lo > self.hi:
            raise ValueError("Residual intervals require ordered exact Fraction endpoints")

    @staticmethod
    def coerce(value):
        if type(value) is _I:
            return value
        if type(value) is F:
            return _I(value, value)
        if type(value) is int:
            return _I(F(value), F(value))
        raise ValueError("Interval arithmetic accepts exact scalars only")

    def __add__(self, value):
        value = self.coerce(value)
        return _I(self.lo+value.lo, self.hi+value.hi)

    __radd__ = __add__

    def __neg__(self):
        return _I(-self.hi, -self.lo)

    def __sub__(self, value):
        return self+-self.coerce(value)

    def __rsub__(self, value):
        return self.coerce(value)+-self

    def __mul__(self, value):
        value = self.coerce(value)
        products = (self.lo*value.lo, self.lo*value.hi,
                    self.hi*value.lo, self.hi*value.hi)
        return _I(min(products), max(products))

    __rmul__ = __mul__

    def __truediv__(self, value):
        value = self.coerce(value)
        if value.lo <= 0 <= value.hi:
            raise ValueError("An inverse interval cannot contain zero")
        return self*_I(1/value.hi, 1/value.lo)


def _interval(raw):
    if type(raw) is not dict or set(raw) != {"lo", "hi"}:
        raise ValueError("An exact saved interval requires both endpoint fields")
    return _I(_fraction(raw["lo"]), _fraction(raw["hi"]))


def _payload(value):
    if type(value) is F:
        return [value.numerator, value.denominator]
    if type(value) is _I:
        return dict(lo=_payload(value.lo), hi=_payload(value.hi))
    if type(value) in (tuple, list):
        return [_payload(x) for x in value]
    if type(value) is dict:
        return {k: _payload(v) for k, v in value.items()}
    if value is None or type(value) in (str, int, bool, float):
        return value
    raise ValueError("A diagnostic receipt contains an unsupported scalar kind")


def _vector(raw):
    if type(raw) is not list or len(raw) != 3:
        raise ValueError("A saved Cartesian vector requires all three components")
    return tuple(_interval(x) for x in raw)


def _matrix(raw):
    if type(raw) is not list or len(raw) != 3:
        raise ValueError("A saved Cartesian matrix requires three complete rows")
    return tuple(_vector(row) for row in raw)


def _jet_values(raw):
    return _vector([x["value_bounds"] for x in raw])


def _jet_matrix_values(raw):
    return tuple(_jet_values(row) for row in raw)


def _vadd(a, b):
    return tuple(x+y for x, y in zip(a, b))


def _vsub(a, b):
    return tuple(x-y for x, y in zip(a, b))


def _transpose(a):
    return tuple(tuple(a[j][i] for j in range(3)) for i in range(3))


def _dot(a, b):
    return sum((x*y for x, y in zip(a, b)), _I(F(0), F(0)))


def _mv(a, b):
    return tuple(_dot(row, b) for row in a)


def _mm(a, b):
    return tuple(tuple(_dot(row, column) for column in _transpose(b)) for row in a)


def _msub(a, b):
    return tuple(_vsub(x, y) for x, y in zip(a, b))


def _identity():
    return tuple(tuple(_I(F(int(i == j)), F(int(i == j))) for j in range(3))
                 for i in range(3))


def _square(value):
    return _I(F(0) if value.lo <= 0 <= value.hi else min(value.lo**2, value.hi**2),
              max(value.lo**2, value.hi**2))


def _norm2(value):
    return sum((_square(x) for x in value), _I(F(0), F(0)))


def _contains_zero(value):
    if type(value) is _I:
        return value.lo <= 0 <= value.hi
    return all(_contains_zero(x) for x in value)


def _inverse(a):
    cofactors = []
    for i in range(3):
        row = []
        for j in range(3):
            remaining = [[a[k][l] for l in range(3) if l != j]
                         for k in range(3) if k != i]
            row.append(((-1)**(i+j))*(remaining[0][0]*remaining[1][1]
                                     -remaining[0][1]*remaining[1][0]))
        cofactors.append(tuple(row))
    cofactors = tuple(cofactors)
    determinant = _dot(a[0], cofactors[0])
    if determinant.lo <= 0:
        raise ValueError("A declared handover requires a positive complete Cartesian map")
    return tuple(tuple(x/determinant for x in row) for row in _transpose(cofactors))


def _body(value):
    return {f.name: getattr(value, f.name) for f in fields(value)
            if f.name != "binding_sha256"}


def _typed(value):
    if type(value) is not dict or any(type(k) is not str for k in value):
        raise ValueError("A restart requires a complete JSON object")
    # JSON encoding distinguishes integer zero from false, and one from true.
    return _json(value)


@dataclass(frozen=True, slots=True)
class NativeCommonInitializationHandover:
    source_context_binding_sha256: str
    provenance: str
    source_clocks_s: tuple
    accepted_original_native_intervals: int
    accepted_new_native_intervals: int
    binding_sha256: str
    scope: str = SCOPE
    physical_initialization_ready: bool = False
    physical_source_prestress_work_handover_certified: bool = False
    reference_preparation_work_paid: bool = False
    new_mixed_force_equilibrium_certified: bool = False
    whole_clock_all_pair_admission_authorized: bool = False
    full_system_temporal_path_work_certified: bool = False
    clocks_advanced: bool = False
    original_point_1_completed: bool = False


@dataclass(frozen=True, slots=True)
class NativeCommonInitializationHandoverResponse:
    system_binding_sha256: str
    virtual_material_increment_m: tuple
    receipt_json: str
    source_clocks_s: tuple
    remaining_conditions: tuple
    original_side_count: int
    original_atom_count: int
    original_beam_count: int
    cartesian_bridge_diagrams_verified: bool
    covector_virtual_work_diagrams_verified: bool
    all_admissible_declared_chart_receiver_radial_gap_certified: bool
    binding_sha256: str
    scope: str = SCOPE
    physical_initialization_ready: bool = False
    physical_source_prestress_work_handover_certified: bool = False
    reference_preparation_work_paid: bool = False
    new_mixed_force_equilibrium_certified: bool = False
    whole_clock_all_pair_admission_authorized: bool = False
    full_system_temporal_path_work_certified: bool = False
    clocks_advanced: bool = False
    original_point_1_completed: bool = False
    received_material_embedding_preserved: bool = False
    accepted_mechanical_initialization_compatible: bool = False
    received_material_embedding_comparison_is_declared_reference_only: bool = True
    accepted_original_native_intervals: int = 1
    accepted_new_native_intervals: int = 0


def bind_native_common_initialization_handover(context, *, provenance):
    api = _source_api()
    api.validate_native_common_initialization_source(context)
    evidence = api.native_common_initialization_source_evidence(context)
    if type(provenance) is not str or not provenance.strip():
        raise ValueError("Declare the source-to-reference initialization diagnostic provenance")
    clocks = tuple(_fraction(v) for v in evidence["target_reference"]["source_clocks_s"])
    if clocks != tuple(F(x) for x in (0, 2000000, 1000000, 1000000)):
        raise ValueError("The target reference cannot overwrite the original retained clocks")
    system = NativeCommonInitializationHandover(context.binding_sha256, provenance, clocks, 1, 0, "")
    system = replace(system, binding_sha256=_digest(_payload(_body(system))))
    _SYSTEMS[system.binding_sha256] = _json(_payload(_body(system)))
    _CONTEXTS[system.binding_sha256] = context
    return system


def _registered(system):
    if (type(system) is not NativeCommonInitializationHandover
            or _digest(_payload(_body(system))) != system.binding_sha256
            or _SYSTEMS.get(system.binding_sha256) != _json(_payload(_body(system)))):
        raise ValueError("The typed initialization diagnostic system was altered or resealed")
    context = _CONTEXTS.get(system.binding_sha256)
    if context is None or context.binding_sha256 != system.source_context_binding_sha256:
        raise ValueError("The complete authenticated source context is absent")
    _source_api().validate_native_common_initialization_source(context)
    return _source_api().native_common_initialization_source_evidence(context)


def native_common_initialization_handover_to_dict(system):
    _registered(system)
    return dict(format="native-common-initialization-handover-system-1",
                **_payload({f.name: getattr(system, f.name) for f in fields(system)}))


def native_common_initialization_handover_from_dict(raw, *, context):
    _typed(raw)
    system = bind_native_common_initialization_handover(context, provenance=raw.get("provenance"))
    if _typed(raw) != _typed(native_common_initialization_handover_to_dict(system)):
        raise ValueError("A system restart differs from the complete authenticated initialization source")
    return system


def native_common_initialization_handover_response_to_dict(response):
    if (type(response) is not NativeCommonInitializationHandoverResponse
            or _digest(_payload(_body(response))) != response.binding_sha256
            or _RESPONSES.get(response.binding_sha256) != _json(_payload(_body(response)))):
        raise ValueError("A typed initialization handover response was altered or resealed")
    return dict(format="native-common-initialization-handover-response-1",
                **_payload({f.name: getattr(response, f.name) for f in fields(response)}))


def native_common_initialization_handover_response_from_dict(raw, *, system,
        expected_virtual_material_increment_m):
    _typed(raw)
    response = evaluate_native_common_initialization_handover(system,
        virtual_material_increment_m=expected_virtual_material_increment_m)
    if _typed(raw) != _typed(native_common_initialization_handover_response_to_dict(response)):
        raise ValueError("A result restart differs from the independently recomputed initialization diagnostic")
    return response


def native_common_initialization_handover_receipt(response):
    native_common_initialization_handover_response_to_dict(response)
    return json.loads(response.receipt_json)


def _literal_vector(raw):
    if type(raw) is not list or len(raw) != 3 or any(type(x) is not float for x in raw):
        raise ValueError("A genuine binary64 world-force record requires three literal floats")
    return tuple(_I(F.from_float(x), F.from_float(x)) for x in raw)


def _source_reactions(atom, basis, side):
    name = "force_on_"+side+"_n"
    if basis == "literal_archival_producer":
        return _literal_vector(json.loads(atom["archival_literal_world_reactions_json"])[name])
    return _vector(dict(atom[basis]["source_traction_world_reactions"])[name])


def _contact_world_force(atom):
    normal = _jet_values(atom["world_receiver_normal_twojet"])
    area = _fraction(atom["original_area_m2"])
    traction = _interval(atom["loading"]["traction_bounds_pa"])
    return tuple(area*traction*x for x in normal)


def _side_geometry(row, source_atom, target_atom, delta):
    saved, target = row["source_side"], row["zero_side"]
    field = saved["field"]
    side, owner = target["side"], target["owner"]
    old_x = _vector(field["material_reference_position_bounds_m"])
    archival_tip = _vector(field["source_reference_position_bounds_m"])
    current_tip = _vector(source_atom["current_objective"][side+"_position_bounds_m"])
    old_arm = _vector(target["original_internal_arm_bounds_m"])
    old_v = _vector(target["original_virtual_radial_direction_bounds"])
    old_a = _matrix(target["old_original_cartesian_gradient_bounds"])
    old_inverse = _matrix(target["old_original_cartesian_inverse_bounds"])
    bridge = target["complete_original_source_input_frame_bridge"]
    B, Q = _matrix(saved["old_reference_gradient"]), _matrix(saved["material_owner_pose"])
    C = _matrix(bridge["roof_chart_correction_bounds"])
    complete_old = _mm(_mm(B, Q), C)
    new_x = _jet_values(target["prepared_point_position_twojet"])
    new_a = _jet_matrix_values(target["original_cartesian_gradient_twojet"])
    saved_t = _jet_matrix_values(target["material_transport_twojet"])
    transport = _mm(new_a, old_inverse)
    transported_v = _mv(transport, old_v)
    archival_arm = _vsub(archival_tip, old_x)
    current_arm = _vsub(current_tip, old_x)
    archive_y = _vadd(new_x, _mv(transport, archival_arm))
    current_y = _vadd(new_x, _mv(transport, current_arm))
    actual_y = _jet_values(target["world_attachment_position_twojet"])
    material_delta = _vsub(new_x, old_x)
    actual_current_delta = _vsub(actual_y, current_tip)
    bridge_residual = _msub(complete_old, old_a)
    inverse_residual = _msub(_mm(old_inverse, old_a), _identity())
    transport_residual = _msub(transport, saved_t)
    commute_residual = _msub(_mm(transport, old_a), new_a)
    arm_residual = _vsub(archival_arm, old_arm)
    archive_y_residual = _vsub(archive_y, actual_y)
    current_archive_difference = _vsub(current_y, archive_y)
    current_archive_expected = _mv(transport, _vsub(current_tip, archival_tip))
    current_archive_residual = _vsub(current_archive_difference, current_archive_expected)
    if not all(_contains_zero(x) for x in (bridge_residual, inverse_residual,
            transport_residual, commute_residual, arm_residual, archive_y_residual,
            current_archive_residual)):
        raise ValueError("A genuine Cartesian initialization diagram failed its interval enclosure")
    target_force = _contact_world_force(target_atom)
    if side == "over":
        target_force = tuple(-x for x in target_force)
    pullback_force = _mv(_transpose(transport), target_force)
    inverse_transport = _inverse(transport)
    transported_delta = _mv(transport, delta)
    new_virtual_work = _dot(target_force, transported_delta)
    pulled_virtual_work = _dot(pullback_force, delta)
    pairing_residual = new_virtual_work-pulled_virtual_work
    if not _contains_zero(pairing_residual):
        raise ValueError("The complete Cartesian covector lost its conjugate virtual work")
    force_rows = []
    for name in ("literal_archival_producer", "archival_old_reference", "current_objective"):
        old_force = _source_reactions(source_atom, name, side)
        conjugate_candidate = _mv(_transpose(inverse_transport), old_force)
        candidate_pullback_residual = _vsub(_mv(_transpose(transport), conjugate_candidate), old_force)
        candidate_work_residual = _dot(conjugate_candidate, transported_delta)-_dot(old_force, delta)
        if not _contains_zero(candidate_pullback_residual) or not _contains_zero(candidate_work_residual):
            raise ValueError("A source Cartesian covector candidate lost its exact conjugate pairing")
        force_rows.append(dict(basis=name, old_world_force_n=old_force,
            formal_conjugate_target_force_candidate_n=conjugate_candidate,
            candidate_force_pullback_commuting_residual_n=candidate_pullback_residual,
            candidate_virtual_work_commuting_residual_j=candidate_work_residual,
            target_force_minus_formal_conjugate_candidate_n=_vsub(target_force, conjugate_candidate),
            target_world_force_minus_old_world_force_n=_vsub(target_force, old_force),
            pulled_target_force_minus_old_force_n=_vsub(pullback_force, old_force),
            source_virtual_work_j=_dot(old_force, delta),
            target_minus_source_virtual_work_j=new_virtual_work-_dot(old_force, delta)))
    return dict(point_id=target["point_id"], side=side, owner=owner,
        original_parent_id=target["original_parent_id"],
        original_material_id=target["original_material_id"],
        allocation=target["allocation"], archive_debits=target["archive_debits"],
        target_internal_coordinate_index=target["internal_radial_coordinate_index"],
        old_material_reference_position_m=old_x,
        archived_source_tip_position_m=archival_tip,
        accepted_current_tip_position_m=current_tip,
        target_material_position_m=new_x,
        target_existing_archive_arm_attachment_position_m=actual_y,
        target_current_tip_affine_candidate_position_m=current_y,
        source_current_tip_minus_archived_tip_m=_vsub(current_tip, archival_tip),
        source_current_tip_minus_old_material_reference_m=_vsub(current_tip, old_x),
        material_position_change_m=material_delta,
        material_position_change_squared_norm_m2=_norm2(material_delta),
        existing_attachment_minus_current_tip_m=actual_current_delta,
        existing_attachment_minus_current_tip_squared_norm_m2=_norm2(actual_current_delta),
        current_tip_candidate_minus_current_tip_m=_vsub(current_y, current_tip),
        full_old_original_cartesian_gradient=old_a,
        full_new_original_cartesian_gradient=new_a,
        full_old_original_cartesian_inverse=old_inverse,
        roof_chart_correction=C, full_cartesian_transport=transport,
        old_virtual_radial_direction=old_v, transported_virtual_radial_direction=transported_v,
        old_original_arm_m=old_arm, current_tip_arm_m=current_arm,
        roof_bridge_residual=bridge_residual, inverse_residual=inverse_residual,
        stored_transport_residual=transport_residual,
        original_gradient_commuting_residual=commute_residual,
        archived_arm_source_residual_m=arm_residual,
        existing_attachment_commuting_residual_m=archive_y_residual,
        current_minus_archived_attachment_commuting_residual_m=current_archive_residual,
        target_world_force_n=target_force, pulled_target_world_force_n=pullback_force,
        virtual_original_material_increment_m=delta,
        transported_virtual_increment_m=transported_delta,
        target_virtual_work_j=new_virtual_work,
        pulled_target_virtual_work_j=pulled_virtual_work,
        covector_virtual_work_pairing_residual_j=pairing_residual,
        old_force_basis_comparisons=force_rows,
        matrix_algebra_identity_by_construction=True,
        interval_zero_enclosures_are_additional_diagnostics=True,
        material_position_change_certified=_norm2(material_delta).lo > 0,
        material_position_comparison_is_registered879_reference_diagnostic=True,
        registered879_reference_is_recovered_historical_material_placement=False,
        force_map_is_conjugate_coordinate_diagnostic=True,
        formal_virtual_work_is_physical_path_work=False,
        current_tip_affine_candidate_applied=False,
        historical_continuum_prestress_reconstructed=False)


def _contact_receipt(source, target):
    area = _fraction(source["original_area_m2"])
    traction = _fraction(source["literal_source_tractions_pa"][0])
    stiffness = _fraction(source["series_stiffness_exact_pa_m"][0])
    literal_energy = sum((_fraction(v) for v in source["stored_source_energy_components_j"]), F(0))
    stored_free = _fraction(source["stored_free_anchors_m"][0])
    anchor = traction/stiffness
    offset = literal_energy-area*traction*traction/(2*stiffness)
    archival_force = _source_reactions(source, "archival_old_reference", "sub")
    literal_force = _source_reactions(source, "literal_archival_producer", "sub")
    objective_force = _source_reactions(source, "current_objective", "sub")
    new_force = _contact_world_force(target)
    energy_zero = _interval(target["energy_bounds_j"])
    return dict(point_id=source["point_id"], trace_id=source["trace_id"],
        original_area_m2=area, source_literal_normal_traction_pa=traction,
        exact_series_normal_stiffness_pa_m=stiffness,
        literal_source_energy_j=literal_energy,
        source_stored_free_anchor_m=stored_free, target_exact_traction_anchor_m=anchor,
        target_anchor_minus_source_stored_free_anchor_m=anchor-stored_free,
        source_energy_rounding_offset_j=offset,
        target_zero_energy_minus_literal_source_energy_j=energy_zero-literal_energy,
        literal_archival_producer_world_force_n=literal_force,
        exact_archival_world_force_n=archival_force,
        current_objective_reconstructed_world_force_n=objective_force,
        target935_normal_world_force_n=new_force,
        archival_exact_minus_literal_world_force_n=_vsub(archival_force, literal_force),
        current_objective_minus_archival_world_force_n=_vsub(objective_force, archival_force),
        target_minus_current_objective_world_force_n=_vsub(new_force, objective_force),
        literal_force_basis_is_accepted_frozen_producer=True,
        objective_reconstruction_is_literal_accepted_producer_force=False,
        target_scalar_traction_retention_certifies_world_force_handover=False,
        initial_normal_max_opening_m=None, initial_normal_damage=None,
        missing_normal_history_replaced_by_virgin_seed=False,
        original_energy_work_store_json=source["source_local_state_json"],
        original_source_work_lineage_id=source["original_source_work_lineage_id"],
        source_state_memory_work_advanced=False)


def _source14_receipt(evidence, source_atoms, target_atoms):
    reference, target = evidence["source_mechanical_reference"], evidence["target_reference"]
    old_order = reference["coordinate_order"]
    excerpt = target["zero_operator_excerpt"]
    new_order = excerpt["coordinate_order"]
    beams = reference["beams"]
    indices = []
    for row in old_order:
        if row[0] == "owner":
            indices.append(new_order.index(row))
        else:
            beam = next(b for b in beams if b["trace_id"] == row[1] and b["side"] == row[2])
            indices.append(beam["target_internal_coordinate_index"])
    if len(indices) != 14 or len(set(indices)) != 14 or set(indices) != set(range(6)) | set(range(92, 100)):
        raise ValueError("The source14 diagnostic projection must retain all true parent internal indices")
    source_gradients = {name: [_I(F(0), F(0)) for _ in range(14)]
                        for name in ("archival_old_reference", "current_objective")}
    target_contact = [_I(F(0), F(0)) for _ in range(14)]
    literal_physical = [_I(F(0), F(0)) for _ in range(14)]
    atom_rows = []
    for source in source_atoms:
        target_atom = target_atoms[source["point_id"]]
        area = _fraction(source["original_area_m2"])
        traction = tuple(_fraction(x) for x in source["literal_source_tractions_pa"])
        new_first = {i: _interval(v) for i, v in target_atom["scalar_increment_twojet"]["first_derivative_bounds"]}
        new_traction = _interval(target_atom["loading"]["traction_bounds_pa"])
        target_row = tuple(area*new_traction*new_first.get(i, _I(F(0), F(0))) for i in indices)
        basis_rows = {}
        for basis in source_gradients:
            first = tuple(tuple(_interval(x) for x in row)
                          for row in source[basis]["scalar_first_derivative_bounds"])
            if len(first) != 2 or any(len(row) != 14 for row in first):
                raise ValueError("The original source scalar driver must retain both complete14 rows")
            gradient = tuple(area*sum((traction[k]*first[k][i] for k in range(2)),
                _I(F(0), F(0))) for i in range(14))
            source_gradients[basis] = [a+b for a, b in zip(source_gradients[basis], gradient)]
            basis_rows[basis] = dict(potential_gradient14=gradient,
                physical_force_covector14=tuple(-x for x in gradient),
                target935_contact_gradient14_minus_source=tuple(a-b for a, b in zip(target_row, gradient)))
        raw_reactions = json.loads(source["archival_literal_world_reactions_json"])
        for i, coordinate in enumerate(old_order):
            if coordinate[0] == "owner":
                if coordinate[1] == source["sub_owner"]:
                    raw = raw_reactions["torque_on_sub_nm"]["xyz".index(coordinate[2])]
                elif coordinate[1] == source["over_owner"]:
                    raw = raw_reactions["torque_on_over_nm"]["xyz".index(coordinate[2])]
                else:
                    raw = 0.0
            elif coordinate[1] == source["trace_id"]:
                raw = raw_reactions["radial_force_n"][0 if coordinate[2] == "sub" else 1]
            else:
                raw = 0.0
            if type(raw) is not float:
                raise ValueError("The preserved literal source reaction requires its genuine binary64 scalar")
            literal_physical[i] += F.from_float(raw)
        target_contact = [a+b for a, b in zip(target_contact, target_row)]
        atom_rows.append(dict(point_id=source["point_id"], target935_contact_gradient14=target_row,
                              source_basis_covectors=basis_rows))
    beam_gradient = [_I(F(0), F(0)) for _ in range(14)]
    for beam in beams:
        i = beam["source_coordinate_index"]
        if old_order[i] != ["radial", beam["trace_id"], beam["side"]]:
            raise ValueError("An eight-beam handover must preserve the genuine source coordinate")
        f = _fraction(beam["literal_force_n"])
        beam_gradient[i] += f
    original_gradient = {name: tuple(a+b for a, b in zip(values, beam_gradient))
                         for name, values in source_gradients.items()}
    literal_total = tuple(a-b for a, b in zip(literal_physical, beam_gradient))
    target14 = tuple(_interval(excerpt["conservative_gradient_bounds"][i]) for i in indices)
    target_source14 = tuple(a+b for a, b in zip(target_contact, beam_gradient))
    return dict(source_coordinate_order14=old_order, target_diagnostic_indices14=indices,
        angular_coordinate_units="rad", radial_coordinate_units="m",
        literal_accepted_producer_physical_covector14=literal_total,
        literal_producer_covector14_uses_retained_source_beam_scalar_coefficients=True,
        exact_archival_source_potential_gradient14=original_gradient["archival_old_reference"],
        reconstructed_current_objective_potential_gradient14=original_gradient["current_objective"],
        exact_archival_source_physical_covector14=tuple(-x for x in original_gradient["archival_old_reference"]),
        reconstructed_current_objective_physical_covector14=tuple(-x for x in original_gradient["current_objective"]),
        target935_saved_zero_full_operator_gradient_subset14=target14,
        target935_source_only_contact_and_beam_gradient14=target_source14,
        target935_full_minus_source_only_gradient14=tuple(a-b for a, b in zip(target14, target_source14)),
        target935_subset_minus_exact_archival_gradient14=tuple(a-b for a, b in zip(target14, original_gradient["archival_old_reference"])),
        target935_subset_minus_current_objective_gradient14=tuple(a-b for a, b in zip(target14, original_gradient["current_objective"])),
        current_objective_minus_exact_archival_gradient14=tuple(a-b for a, b in zip(original_gradient["current_objective"], original_gradient["archival_old_reference"])),
        exact_archival_physical_minus_literal_producer_covector14=tuple(-a-b for a, b in zip(original_gradient["archival_old_reference"], literal_total)),
        contact_atom_covectors=atom_rows,
        receiver_frame_spin_retained_in_scalar_driver=True,
        target_zero_coordinate_subset_is_accepted_physical_initialization=False,
        source14_projection_is_diagnostic_only=True,
        unmeasured86_body_coordinates_assigned_zero=False)


def _literal_saved_scalar(value):
    if type(value) is float:
        return F.from_float(value)
    if type(value) is list:
        return _fraction(value)
    raise ValueError("A genuine saved work ledger scalar requires an exact literal value")


def _beam_and_ledger_receipts(evidence, source_atoms):
    reference = evidence["source_mechanical_reference"]
    beams = reference["beams"]
    if len(beams) != 8 or len({b["original_parent_id"] for b in beams}) != 8:
        raise ValueError("The eight original beams must occur exactly once")
    beam_rows = []
    energy_sum = F(0)
    for beam in beams:
        w = _fraction(beam["source_displacement_m"])
        K = _fraction(beam["stiffness_n_m"])
        energy = _fraction(beam["literal_energy_j"])
        force = _fraction(beam["literal_force_n"])
        if K <= 0:
            raise ValueError("An original source beam requires its positive literal stiffness")
        energy_sum += energy
        beam_rows.append(dict(original_parent_id=beam["original_parent_id"],
            trace_id=beam["trace_id"], side=beam["side"],
            source_coordinate_index=beam["source_coordinate_index"],
            target_internal_coordinate_index=beam["target_internal_coordinate_index"],
            target_tip_index=beam["target_tip_index"],
            source_accepted_tip_displacement_m=w, literal_reference_energy_j=energy,
            literal_reference_force_n=force, source_stiffness_n_m=K,
            exact_quadratic_reference_energy_j=K*w*w/2,
            exact_quadratic_reference_force_n=K*w,
            literal_minus_exact_quadratic_energy_j=energy-K*w*w/2,
            literal_minus_exact_quadratic_force_n=force-K*w,
            beam_original_energy_added_again=False,
            scalar_coefficients_retained_without_replaying_beam_solve=True))
    ledger = reference["ledger_candidate"]
    if type(ledger) is list:
        ledger = dict(ledger)
    aggregate = _literal_saved_scalar(ledger["bending_energy_increment_j"])
    residual = energy_sum-aggregate
    if residual != -F(3, 131072):
        raise ValueError("The genuine eight-beam individual/aggregate rounding residual changed")
    contacts = sum((sum((_fraction(v) for v in a["stored_source_energy_components_j"]), F(0))
                    for a in source_atoms), F(0))
    old_total = contacts+aggregate
    target_literal = contacts+energy_sum
    zero_energy = _interval(evidence["target_reference"]["zero_operator_excerpt"]["energy_bounds_j"])
    return beam_rows, dict(source_literal_contact_energy_j=contacts,
        source_individual_beam_literal_energy_sum_j=energy_sum,
        source_aggregate_beam_literal_energy_j=aggregate,
        individual_minus_aggregate_beam_energy_j=residual,
        old_contact_plus_aggregate_beam_energy_j=old_total,
        target_contact_plus_individual_beam_literal_energy_j=target_literal,
        literal_source_reference_relabeling_energy_residual_j=target_literal-old_total,
        saved_target935_zero_energy_minus_old_source_energy_j=zero_energy-old_total,
        source_before_ledger_json=_json(reference["ledger_before"]),
        source_candidate_ledger_json=_json(reference["ledger_candidate"]),
        literal_aggregation_residual_hidden=False,
        contact_offsets_retained=True, work_ledgers_changed=False,
        physical_preparation_or_motion_work_j=None,
        assumed_physical_duration_s=None,
        additional_lambda_drive_work_j=None,
        source_beam_energy_replayed=False,
        saved_zero_energy_match_certifies_physical_preparation_work=False)


def _continuum_receipt(evidence):
    target = evidence["target_reference"]["zero_operator_excerpt"]
    cold, hot = target["cold_response_selected"], target["hot_response_selected"]
    unknown = evidence["unknown_physical_inputs"]
    return dict(saved_declared935_zero_cold_energy_j=cold["energy_j"],
        saved_declared935_zero_cold_beam_energy_j=cold["beam_energy_j"],
        saved_declared935_zero_cold_nonlocal_energy_j=cold["nonlocal_orthogonal_energy_j"],
        saved_declared935_zero_cold_remainder_projection_n=cold["remainder_projection_n"],
        saved_declared935_zero_cold_tip_projection_m=cold["tip_projection_m"],
        saved_declared935_zero_hot_energy_j=hot["energy_bounds_j"],
        saved_declared935_zero_hot_gradient30_n=hot["gradient_bounds_n"],
        historical_cold56_deformation_coordinates_m=None,
        historical_hot30_deformation_coordinates_m=None,
        historical_original_parent_3d_prestress_field=None,
        historical_hot_birth_material_prestress_field=None,
        historical_to_declared_continuum_energy_residual_j=None,
        historical_to_declared_continuum_force_residual=None,
        unknown_physical_inputs=unknown,
        saved_zero_operator_is_authenticated_diagnostic=True,
        missing_historical_fields_replaced_by_zero=False,
        zero_hot_energy_certifies_birth_handover=False,
        declared899_903_laws_are_recovered_historical_fields=False,
        physical_continuum_initialization_ready=False)


def _allq_receiver_gap_receipts(evidence):
    target = evidence["target_reference"]
    cert = target["whole_domain_certificate"]
    owner_cert = cert["receiver"]
    U = _fraction(owner_cert["maximum_displacement_m"])
    scale = _fraction(owner_cert["fixed_radial_scale"])
    envelope = tuple(_fraction(v) for v in target["coordinate_envelope_abs_m"])
    banks = target["whole_cold_banks"]
    order = target["zero_operator_excerpt"]["coordinate_order"][36:92]
    if (len(envelope) != 56 or len(banks) != 8 or U < 0 or scale <= 0
            or any(len(bank["coordinates_m"]) != 7 for bank in banks)):
        raise ValueError("The whole receiver chart needs its authentic eight-bank56-coordinate envelope")
    expected = []
    for coordinate in order:
        bank = next(b for b in banks if b["component_id"] == coordinate[1])
        local_index = bank["coordinate_order"].index(coordinate[2])
        expected.append(2*abs(_fraction(bank["coordinates_m"][local_index])))
    if tuple(expected) != envelope:
        raise ValueError("The certified895 displacement envelope differs from the actual935 cold chart")
    receiver_parents = set(owner_cert["parent_ids"])
    source_atoms = {a["point_id"]: a for a in evidence["source_mechanical_reference"]["atoms"]}
    rows = []
    for row in target["sides"]:
        point, side = row["zero_point"], row["zero_side"]
        if side["side"] != "over":
            continue
        if point["original_parent_id"] not in receiver_parents or _fraction(point["fixed_radial_scale"]) != scale:
            raise ValueError("An original receiving label changed its authentic whole-domain radial guard")
        archive = json.loads(point["source_archive_point_json"])["response"]
        radial = _interval(archive["radial_support_bounds_m"])
        lower = scale*radial.lo-U
        source = source_atoms[side["point_id"]]
        current = _vector(source["current_objective"]["over_position_bounds_m"])
        old_radius2 = _norm2(current)
        old_registered_x = _vector(row["source_side"]["field"]["material_reference_position_bounds_m"])
        old_registered_radius2 = _norm2(old_registered_x)
        squared_gap = lower*lower-old_radius2.hi
        registered_squared_gap = lower*lower-old_registered_radius2.hi
        if lower <= 0 or squared_gap <= 0 or registered_squared_gap <= 0:
            raise ValueError("A whole-chart reference incompatibility needs a strictly positive authentic radial separation")
        rows.append(dict(point_id=side["point_id"], original_parent_id=point["original_parent_id"],
            original_material_id=point["original_material_id"],
            fixed_receiver_radial_scale=scale,
            authenticated_whole_cold_displacement_upper_m=U,
            archive_carrier_radial_support_bounds_m=radial,
            all_admissible_cold_chart_radius_lower_m=lower,
            accepted_current_tip_squared_radius_bounds_m2=old_radius2,
            squared_radius_separation_lower_m2=squared_gap,
            old_registered879_material_reference_squared_radius_bounds_m2=old_registered_radius2,
            target_minus_old_registered_reference_squared_radius_separation_lower_m2=registered_squared_gap,
            whole895_cold_envelope_matches935_actual_cold_chart=True,
            declared_owner_rotations_preserve_radius=True,
            preparation935_radial_twist_preserves_radius=True,
            hot_and_internal_coordinates_do_not_enter_cold_material_position=True,
            physical_material_position_is_distinct_from_internal_virtual_attachment=True,
            accepted_current_mechanical_tip_representable_as_declared935_cold_material_point=False,
            registered879_material_reference_representable_as_declared935_cold_material_point=False,
            historical_physical_material_embedding_preserved=None,
            whole_historical_physical_material_embedding_nonrepresentability_certified=False,
            internal_virtual_source_attachments_nonrepresentability_certified=False,
            physical_preparation_path_ruled_out=False,
            radial_separation_is_conditional_declared_chart_result=True))
    if len(rows) != 16:
        raise ValueError("The whole-chart compatibility check requires all sixteen original receiving labels")
    return rows


def evaluate_native_common_initialization_handover(system, *, virtual_material_increment_m):
    if (type(virtual_material_increment_m) is not tuple or len(virtual_material_increment_m) != 3
            or any(type(v) is not F for v in virtual_material_increment_m)):
        raise ValueError("Specify all three exact Cartesian virtual material increments in metres")
    evidence = _registered(system)
    source_atoms = evidence["source_mechanical_reference"]["atoms"]
    target_atoms = evidence["target_reference"]["atoms"]
    source_map = {a["point_id"]: a for a in source_atoms}
    target_map = {a["point_id"]: a for a in target_atoms}
    sides = evidence["target_reference"]["sides"]
    if (len(source_atoms) != 16 or len(source_map) != 16 or set(source_map) != set(target_map)
            or len(sides) != 32 or len({(r["zero_side"]["point_id"], r["zero_side"]["side"]) for r in sides}) != 32):
        raise ValueError("The complete genuine original16/contact32 initialization catalogue changed")
    delta = tuple(_I(v, v) for v in virtual_material_increment_m)
    side_receipts = [_side_geometry(row, source_map[row["zero_side"]["point_id"]],
        target_map[row["zero_side"]["point_id"]], delta) for row in sides]
    if any(not row["material_position_change_certified"] for row in side_receipts):
        raise ValueError("The declared actual32 material motions lack a strict nonzero displacement receipt")
    contacts = [_contact_receipt(source, target_map[source["point_id"]]) for source in source_atoms]
    source14 = _source14_receipt(evidence, source_atoms, target_map)
    beams, ledger = _beam_and_ledger_receipts(evidence, source_atoms)
    receiver_gap = _allq_receiver_gap_receipts(evidence)
    receipt = dict(format="native-common-initialization-handover-receipt-1",
        virtual_material_increment_m=virtual_material_increment_m,
        side_geometry_receipts=side_receipts,
        contact_force_work_receipts=contacts,
        source14_covector_comparison=source14,
        beam_energy_receipts=beams,
        source_energy_work_ledger_receipt=ledger,
        continuum_initialization_receipt=_continuum_receipt(evidence),
        allq_receiver_reference_gap_receipts=receiver_gap,
        accepted_source=evidence["accepted_source"],
        target_retained_reference_clocks_s=system.source_clocks_s,
        remaining_conditions=evidence["remaining_conditions"],
        original_side_count=32, original_atom_count=16, original_beam_count=8,
        live_source_side_count=sum(r["allocation"] == "live" for r in side_receipts),
        archived_source_side_count=sum(r["allocation"] == "archived" for r in side_receipts),
        received_material_embedding_preserved=False,
        received_material_embedding_comparison_is_declared_reference_only=True,
        historical_physical_material_embedding_preserved=None,
        whole_historical_physical_material_embedding_nonrepresentability_certified=False,
        accepted_mechanical_initialization_compatible=False,
        conditional_all_admissible_declared_chart_receiver_radial_gap_certified=True,
        world_covector_compatibility_certified=False,
        actual_motion_path_work_known=False,
        physical_source_historical_prestress_recovered=False,
        normal_history_initialized=False,
        physical_source_prestress_work_handover_certified=False,
        physical_initialization_ready=False,
        source_state_memory_work_or_clocks_advanced=False,
        original_point_1_completed=False,
        historical_force_solve_reexecuted=False,
        previous_tensor_quadrature_reexecuted=False)
    response = NativeCommonInitializationHandoverResponse(system.binding_sha256,
        virtual_material_increment_m, _json(_payload(receipt)), system.source_clocks_s,
        tuple(evidence["remaining_conditions"]), 32, 16, 8, True, True, True, "")
    response = replace(response, binding_sha256=_digest(_payload(_body(response))))
    _RESPONSES[response.binding_sha256] = _json(_payload(_body(response)))
    return response
