"""Once-only source accounting on the authentic COMMON cold receiver supports.

The four continuous C dtheta families and the sixteen literal source atoms are
separate measures.  A symbolic endpoint graph closes each continuous budget;
outward endpoint intervals never replace its shared projective identities.
This receipt preserves source memory and work stores.  It does not transport
rod material, equate old world forces, solve equilibrium, or adopt an interval.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from fractions import Fraction as F
import hashlib
import json

from .geometric_native_pose import FractionInterval as I, interval_atan, interval_pi, _rounded_interval
from .geometric_native_common_reference import validate_native_common_reference
from .geometric_native_common_contact_cuts import (
    validate_common_contact_cuts, validate_common_contact_cut, _RUNTIME,
    validate_common_contact_cut_bounds, _inverse_map, _map, _dot, _cross,
)
from .geometric_native_common_contact_ports import _encoded, _payload, _read_fraction, _read_interval
from .geometric_native_common_contact_state import _source, NativeCommonCurrentMemoryPiece
from .geometric_native_contact_field_history import _field_bounds
from .geometric_native_common_contact_measure import (
    _profile, _profile_bounds, _square, validate_native_common_contact_reference_measure_declaration,
)
from .geometric_native_polygons import native_ray_from_dict, _ray_values

FORMAT = "native-common-whole-source-allocation-1"
MODEL = "once_only_original_receiver_source_accounting_with_declared_family_depth_profiles"
SCOPE = "WHOLE_SOURCE_ACCOUNTING_NO_PHYSICAL_FORCE_OR_MATERIAL_HANDOVER"
STORE_FIELDS = (
    "contact_energy_j", "bulk_energy_j", "cumulative_external_work_j",
    "cumulative_numerical_dissipation_j", "cumulative_friction_work_j",
    "cumulative_viscous_work_j",
)
SOURCE_EPOCHS = ("prepared_state", "previous_state", "unaccepted_candidate_state")
_REGISTRY = {}
_EXPECTED_PAIRS = frozenset((("octant:5", "octant:1"), ("octant:5", "octant:3"),
    ("octant:6", "octant:2"), ("octant:7", "octant:2"), ("octant:7", "octant:3")))


def _digest(value):
    return hashlib.sha256(_encoded(value)).hexdigest()


def _literal(value):
    if type(value) is not float:
        raise ValueError("Source allocation retains literal binary64 source numbers")
    return F.from_float(value)


def _rotation_word(rows):
    """Only adjacent exact inverse exponentials cancel; no commutation."""
    result = []
    for row in rows:
        step = (tuple(_literal(x) for x in row["omega_rad_per_myr"]), _literal(row["dt_myr"]))
        if result and step[1] == result[-1][1] and step[0] == tuple(-x for x in result[-1][0]):
            result.pop()
        else:
            result.append(step)
    return tuple(result)


def _oriented_vector(values):
    vector = tuple(F(x) for x in values)
    leading = next((x for x in vector if x), None)
    if leading is None:
        raise ValueError("An authentic projective ray or plane cannot be zero")
    return tuple(x / leading for x in vector), 1 if leading > 0 else -1


def _plane_key(raw):
    vector, sign = _oriented_vector(raw["normal"])
    return (vector, _rotation_word(raw["rotations"])), sign


def _canonical_ray_key(raw):
    """Exact oriented projective identity, independent of numeric enclosures."""
    word, sign = _rotation_word(raw["rotations"]), raw["sign"]
    if raw["kind"] == "seed":
        vector, own_sign = _oriented_vector(raw["reference"])
        return ("seed", vector, word, sign * own_sign)
    if raw["kind"] != "cross" or len(raw["planes"]) != 2:
        raise ValueError("A source boundary must retain its complete native projective ray")
    first, first_sign = _plane_key(raw["planes"][0])
    second, second_sign = _plane_key(raw["planes"][1])
    sign *= first_sign * second_sign
    if repr(first) > repr(second):
        first, second, sign = second, first, -sign
    if first == second:
        raise ValueError("A cross ray cannot be formed from the same projective plane")
    return ("cross", first, second, word, sign)


def _seed_key(direction, rows):
    return _canonical_ray_key(dict(kind="seed", reference=direction, rotations=rows, sign=1, planes=[]))


def _telescoped_endpoint_coefficients(expressions):
    coefficients = {}
    for expression in expressions:
        for endpoint_id, coefficient in expression:
            if type(endpoint_id) is not str or type(coefficient) is not F:
                raise ValueError("Endpoint budget expressions need exact named coefficients")
            coefficients[endpoint_id] = coefficients.get(endpoint_id, F(0)) + coefficient
    return tuple(sorted((key, value) for key, value in coefficients.items() if value))


def _angle_bounds(trace, extent, direction, precision):
    start, end = (tuple(I(_literal(x)) for x in trace[key]) for key in ("start", "end"))
    x, y = _dot(start, direction), _dot(end, direction)
    if x.lo <= 0 or y.lo <= 0:
        raise ValueError("An internal source endpoint must lie strictly in its original quarter")
    half_pi = interval_pi(precision + 8) / 2
    native = interval_atan(y / x, precision) if y.hi <= x.lo else half_pi - interval_atan(x / y, precision)
    return native * extent / half_pi


@dataclass(frozen=True, slots=True)
class NativeCommonSourceBoundary:
    endpoint_id: str
    trace_id: str
    endpoint_kind: str
    computational_angle_bounds_radians: I
    exact_rational_value_radians: F | None
    canonical_projective_expression_json: str
    authentic_endpoint_aliases: tuple
    source_direction_expression: str


@dataclass(frozen=True, slots=True)
class NativeCommonSourceAllocationCell:
    cell_id: str
    trace_id: str
    receiver_original_parent_id: str
    active: bool
    cut_id: str | None
    cut_snapshot_sha256: str | None
    lower_endpoint_id: str
    upper_endpoint_id: str
    computational_angle_endpoint_bounds_radians: tuple
    source_area_endpoint_linear_expression: tuple
    source_reference_area_bounds_m2: I
    common_physical_radius_cubed_bounds_m3: I | None
    depth_profile_coefficients: tuple
    source_memory_family_trace_sha256: str
    closed_seams_have_zero_continuous_measure: bool = True
    finite_original_atom_areas_assigned_to_cell: bool = False


@dataclass(frozen=True, slots=True)
class NativeCommonSourceFamily:
    trace_id: str
    receiver_original_parent_id: str
    donor_original_parent_id: str
    source_computational_extent_radians: F
    source_interface_area_density_m2_per_computational_radian: F
    source_reference_interface_area_m2: F
    source_radius_m: F
    source_interface_height_m: F
    source_literal_original_frame_sine: F
    depth_profile_coefficients: tuple
    depth_profile_integral_exact: F
    complete_source_trace_json: str
    complete_current_memory_trace_json: str
    complete_endpoint_anchor_fields_json: str
    source_memory_family_trace_sha256: str
    original_atom_ids: tuple
    cell_ids: tuple
    summed_area_endpoint_linear_expression: tuple
    full_area_endpoint_linear_expression: tuple
    native_angle_orientation_receipts: tuple


@dataclass(frozen=True, slots=True)
class NativeCommonOriginalAtomLedgerEntry:
    point_id: str
    trace_id: str
    receiver_original_parent_id: str
    donor_original_parent_id: str
    computational_angle_exact_radians: F
    original_area_m2: F
    complete_original_atom_history_json: str
    complete_current_source_state_json: str
    source_accounting_bucket: str
    strict_endpoint_comparison_reason: str
    source_current_node_epoch: str
    source_contact_elapsed_time_s: F
    source_contact_step_count: int
    assigned_to_new_physical_material_point_or_area: bool = False


@dataclass(frozen=True, slots=True)
class NativeCommonOriginalAtomBudget:
    point_ids: tuple
    count: int
    original_area_m2: F
    source_epoch_work_and_energy_j: tuple
    current_minus_previous_work_and_energy_j: tuple


_SCOPE_FLAGS = dict(
    once_only_four_continuous_source_families_accounted=True,
    symbolic_full_source_reference_area_closure_certified=True,
    whole_current_continuous_memory_retained=True,
    once_only_sixteen_literal_source_atom_ledger_retained=True,
    inactive_source_supports_retained=True,
    family_depth_profile_model_explicit=True,
    inherited_three_dimensional_source_depth_profile=False,
    old_mechanical_atom_area_assigned=False,
    literal_atom_area_equated_to_continuous_source_measure=False,
    full_face_physical_memory_measure_transfer_certified=False,
    full_face_memory_measure_transfer_certified=False,
    global_receiver_source_allocation_certified=False,
    old_gauss_weights_reused_for_new_support=False,
    old_rod_energy_or_prestress_transferred=False,
    old_world_force_equivalence_certified=False,
    global_work_handover_certified=False,
    hot_receiving_contact_law_recovered=False,
    whole_clock_all_pair_admission_certified=False,
    force_equilibrium_certified=False,
    physical_material_admission_authorized=False,
    source_state_or_thermal_state_advanced=False,
    new_native_interval_accepted=False,
)


@dataclass(frozen=True, slots=True)
class NativeCommonSourceAllocation:
    source_snapshot_sha256: str
    source_epoch_sha256: str
    reference_snapshot_sha256: str
    cut_set_snapshot_sha256: str
    source_history_binding_sha256: str
    source_current_memory_sha256: str
    source_current_atom_states_sha256: str
    complete_source_dto_json: tuple
    complete_historical_saved_field_json: str
    complete_current_memory_json: str
    complete_original_physical_history_json: str
    complete_current_atom_states_json: str
    complete_source_trace_bindings_json: str
    source_native_elapsed_time_s: F
    source_memory_elapsed_time_s: F
    source_memory_stage_count: int
    source_contact_elapsed_time_s: F
    source_contact_step_count: int
    source_current_node_epoch: str
    families: tuple
    boundaries: tuple
    cells: tuple
    original_atom_ledger: tuple
    all16_source_atom_budget: NativeCommonOriginalAtomBudget
    source_bucket_atom_budgets: tuple
    active_current_receiver_source_atom_count: int
    inactive_current_receiver_source_atom_count: int
    uncertain_endpoint_source_atom_count: int
    excluded_hot_cut_snapshot_sha256: str
    provenance: str
    precision: int
    binding_sha256: str

    @property
    def scope(self):
        return SCOPE

    @property
    def source_atom_ledger(self):
        return self.original_atom_ledger


for _flag_name, _flag_value in _SCOPE_FLAGS.items():
    setattr(NativeCommonSourceAllocation, _flag_name, property(lambda self, value=_flag_value: value))


def _seal(value):
    return _digest({field.name: getattr(value, field.name) for field in fields(value)
                    if field.name != "binding_sha256"})


def _budget(atoms):
    return NativeCommonOriginalAtomBudget(tuple(atom["point_id"] for atom in atoms), len(atoms),
        sum((_literal(atom["original_area_m2"]) for atom in atoms), F(0)),
        tuple((epoch, tuple((key, sum((_literal(atom[epoch][key]) for atom in atoms), F(0)))
                           for key in STORE_FIELDS)) for epoch in SOURCE_EPOCHS),
        tuple((key, sum((_literal(atom["unaccepted_candidate_state"][key])
                         - _literal(atom["previous_state"][key]) for atom in atoms), F(0)))
              for key in STORE_FIELDS))


def _classify(parent, angle, endpoints):
    if parent == "octant:0":
        return "inactive:octant:0", "the complete original receiver carrier has no current cold cut"
    if parent == "octant:1":
        gamma = endpoints["gamma7"]
        if angle < gamma.lo:
            return "inactive:octant:1", "literal theta is strictly below the true terminal-trim support"
        if angle > gamma.hi:
            return "retained-contact:octant:5:octant:1", "literal theta is strictly within the true terminal-trim support"
        return "uncertain:gamma7", "literal theta belongs to the unresolved exact-endpoint enclosure"
    if parent == "octant:2":
        sigma, alpha = endpoints["sigma8"], endpoints["alpha8"]
        if angle < sigma.lo:
            return "retained-contact:octant:7:octant:2", "literal theta is strictly within the microscopic support"
        if sigma.hi < angle < alpha.lo:
            return "retained-contact:octant:6:octant:2", "literal theta is strictly within the long support"
        if angle > alpha.hi:
            return "inactive:octant:2", "literal theta is strictly above the long-support terminal"
        return "uncertain:octant:2", "literal theta belongs to an unresolved shared or terminal endpoint enclosure"
    if parent == "octant:3":
        sigma = endpoints["sigma11"]
        if angle < sigma.lo:
            return "retained-contact:octant:5:octant:3", "literal theta is strictly within the microscopic support"
        if angle > sigma.hi:
            return "retained-contact:octant:7:octant:3", "literal theta is strictly within the long support"
        return "uncertain:sigma11", "literal theta belongs to the unresolved exact-endpoint enclosure"
    raise ValueError("A source atom cannot borrow an unknown receiver family")


def bind_native_common_source_allocation(reference, cuts, *, depth_profiles, provenance, precision=160):
    """Bind all source carriers once; profile declarations remain NEW models."""
    if type(precision) is not int or not 96 <= precision <= 512:
        raise ValueError("Source allocation precision must be an integer from 96 to 512")
    if type(provenance) is not str or not provenance.strip():
        raise ValueError("Whole-source conditional depth profiles need explicit provenance")
    reference = validate_native_common_reference(reference)
    cuts = validate_common_contact_cuts(cuts, reference=reference)
    cold = {(cut.donor_original_parent_id, cut.receiver_original_parent_id): cut
            for cut in cuts.cuts if cut.receiver_material_kind == "original_live"}
    hot = [cut for cut in cuts.cuts if cut.receiver_material_kind == "hot_birth"]
    if set(cold) != _EXPECTED_PAIRS or len(cuts.cuts) != 6 or len(hot) != 1 or hot[0].receiving_trace_id is not None:
        raise ValueError("Whole-source allocation requires all five true cold cuts and the separate HOT boundary")
    before, history, path = _source(reference.source_snapshot_sha256)
    snapshot = before["snapshot"]
    material = snapshot["current_material_epoch"]["source_material_state"]
    if (_encoded(reference.complete_source_dto_json) != _encoded(cuts.complete_source_dto_json)
            or reference.saved_field_json != cuts.complete_historical_saved_field_json):
        raise ValueError("Source allocation cannot mix complete source DTOs or saved historical fields")
    bindings = material["trace_bindings"]
    by_parent = {binding["receiver_fragment_id"]: binding for binding in bindings}
    traces = {trace["trace_id"]: trace for trace in path["initial_mechanics"]["fault_traces"]["traces"]}
    memory = path["step"]["memory"]
    current_traces = {trace["trace_id"]: trace for trace in memory["traces"]}
    anchors = {row[0]: row for row in path["step"]["anchor_fields"]}
    atoms, states = history["quadrature_histories"], dict(snapshot["states"])
    tids = {binding["trace_id"] for binding in bindings}
    if (set(by_parent) != {"octant:0", "octant:1", "octant:2", "octant:3"}
            or len(bindings) != 4 or len(tids) != 4 or set(traces) != tids or set(current_traces) != tids
            or set(anchors) != tids or type(depth_profiles) is not dict or set(depth_profiles) != tids):
        raise ValueError("Exactly four authentic original families each need one immutable depth profile")
    profiles = {tid: _profile(depth_profiles[tid]) for tid in sorted(tids)}
    if len(atoms) != len(states) or len(atoms) != 16 or len({atom["point_id"] for atom in atoms}) != 16:
        raise ValueError("The complete sixteen literal original atom identities must be retained once")
    runtime = _RUNTIME[next(iter(cold.values())).cut_snapshot_sha256]
    poses, analytic = runtime["poses"], runtime["analytic"]
    owner_histories = dict(snapshot["owner_histories"])
    owner0_word = _rotation_word(owner_histories[0])
    arcs = {pair: json.loads(cut.genuine_native_arc_json) for pair, cut in cold.items()}
    receipts = {}
    for pair, cut in cold.items():
        trace = traces[by_parent[pair[1]]["trace_id"]]
        start, end = (tuple(_literal(x) for x in trace[key]) for key in ("start", "end"))
        if (sum(x*x for x in start) != 1 or sum(x*x for x in end) != 1
                or sum(x*y for x, y in zip(start, end)) != 0
                or _cross(start, end) != (F(-1), F(0), F(0))
                or _plane_key(arcs[pair]["plane"]) != (((F(1), F(0), F(0)), owner0_word), 1)):
            raise ValueError("The exact native carrier must reduce to its original receiver quarter plane")
        normal = _cross(cut.start_unit_bounds, cut.tangent_unit_bounds)
        owner0_x = tuple(poses[0][i][0] for i in range(3))
        if _dot(normal, owner0_x).lo <= 0:
            raise ValueError("The source-native parameter orientation is not strictly authenticated")
        receipts[pair] = (cut.cut_id, cut.cut_snapshot_sha256, _encoded(_plane_key(arcs[pair]["plane"])).decode(),
                          "exact_original_quarter_plane_and_positive_source_orientation", F(-1))
    long8, tiny8 = arcs[("octant:6", "octant:2")], arcs[("octant:7", "octant:2")]
    long11, tiny11 = arcs[("octant:7", "octant:3")], arcs[("octant:5", "octant:3")]
    if (_canonical_ray_key(long8["end"]) != _canonical_ray_key(tiny8["start"])
            or _canonical_ray_key(long11["end"]) != _canonical_ray_key(tiny11["start"])
            or _canonical_ray_key(tiny8["end"]) != _seed_key((0, 1, 0), owner_histories[0])
            or _canonical_ray_key(long11["start"]) != _seed_key((0, 1, 0), owner_histories[0])
            or _canonical_ray_key(tiny11["end"]) != _seed_key((0, 0, 1), owner_histories[0])
            or _canonical_ray_key(arcs[("octant:5", "octant:1")]["start"]) != _seed_key((0, 0, 1), owner_histories[0])):
        raise ValueError("The source partition requires shared exact projective seams and authentic quarter endpoints")
    extents = {parent: _literal(current_traces[binding["trace_id"]]["extent_radians"])
               for parent, binding in by_parent.items()}
    if any(extent <= 0 for extent in extents.values()):
        raise ValueError("All four original source carriers need positive complete extents")
    def ray_angle(parent, raw):
        direction = _ray_values(native_ray_from_dict(raw), max(320, precision + 24))[1]
        return _angle_bounds(traces[by_parent[parent]["trace_id"]], extents[parent],
                             _inverse_map(poses[0], direction), precision)
    endpoint_bounds = dict(sigma8=ray_angle("octant:2", long8["end"]),
                           alpha8=ray_angle("octant:2", long8["start"]),
                           sigma11=ray_angle("octant:3", long11["end"]))
    k1 = analytic[8]
    h1 = analytic[6].bounds(I(1)) / analytic[3].bounds(I(1))
    # _curve_geometry uses b1-a1*k1=(b1*A1-a1*B1)/A1.  The exact
    # height_numerator polynomial is analytic[6], never a chosen endpoint.
    terminal = _inverse_map(poses[0], _map(poses[1], (h1, -k1, I(1))))
    endpoint_bounds["gamma7"] = _angle_bounds(traces[by_parent["octant:1"]["trace_id"]],
                                              extents["octant:1"], terminal, precision)
    s8, a8, s11, g7 = (endpoint_bounds[key] for key in ("sigma8", "alpha8", "sigma11", "gamma7"))
    if not (0 < s8.lo <= s8.hi < a8.lo <= a8.hi < extents["octant:2"]
            and 0 < s11.lo <= s11.hi < extents["octant:3"]
            and 0 < g7.lo <= g7.hi < extents["octant:1"]):
        raise ValueError("The microscopic supports, positive inactive tail, or true trim have unresolved strict order")
    R = _literal(history["origin_surface"]["radius_km"]) * 1000
    if R != reference.radius_m:
        raise ValueError("Continuous source measure must retain the actual original radius")
    boundaries, cells, families = [], [], []
    internal = {
        "octant:2": (("sigma8", long8["end"], ((cold[("octant:6", "octant:2")].cut_id, "end"),
                      (cold[("octant:7", "octant:2")].cut_id, "start"))),
                     ("alpha8", long8["start"], ((cold[("octant:6", "octant:2")].cut_id, "start"),))),
        "octant:3": (("sigma11", long11["end"], ((cold[("octant:7", "octant:3")].cut_id, "end"),
                       (cold[("octant:5", "octant:3")].cut_id, "start"))),),
        "octant:1": (("gamma7", None, ((cold[("octant:5", "octant:1")].cut_id, "true_terminal_end"),)),),
        "octant:0": (),
    }
    partitions = {
        "octant:0": (("zero", "extent", None),),
        "octant:1": (("zero", "gamma7", None), ("gamma7", "extent", ("octant:5", "octant:1"))),
        "octant:2": (("zero", "sigma8", ("octant:7", "octant:2")),
                     ("sigma8", "alpha8", ("octant:6", "octant:2")), ("alpha8", "extent", None)),
        "octant:3": (("zero", "sigma11", ("octant:5", "octant:3")),
                     ("sigma11", "extent", ("octant:7", "octant:3"))),
    }
    for parent in sorted(by_parent):
        binding = by_parent[parent]
        tid, extent = binding["trace_id"], extents[parent]
        family_atoms = [atom for atom in atoms if atom["trace_id"] == tid]
        if len(family_atoms) != 4 or any(atom["original_frame"] != family_atoms[0]["original_frame"] for atom in family_atoms):
            raise ValueError("Every source family retains four immutable original frame declarations")
        height = _literal(family_atoms[0]["original_material"]["contact_height_m"])
        sine = _literal(family_atoms[0]["original_frame"][0])
        if min(height, sine) <= 0 or height != _literal(traces[tid]["contact_height_m"]):
            raise ValueError("The continuous C coefficient must use the literal source height and frame sine")
        C, profile, memory_json = R * height / sine, profiles[tid], _encoded(current_traces[tid]).decode()
        trace_memory_hash = hashlib.sha256(memory_json.encode()).hexdigest()
        pieces = current_traces[tid]["pieces"]
        reached = F(0)
        for piece in pieces:
            low, high = _literal(piece["low_radians"]), _literal(piece["high_radians"])
            if low > reached or low > high or low < 0 or high > extent:
                raise ValueError("The whole current memory family must cover its original interval without gaps")
            for field_name in ("plastic_slip", "cumulative_slip"):
                coefficients = tuple(_literal(x) for x in piece[field_name]["coefficients"])
                if len(coefficients) != 5:
                    raise ValueError("Whole-family memory retains all five literal source coefficients")
            reached = max(reached, high)
        if reached != extent:
            raise ValueError("The complete source current memory family is uncovered")
        zero_id, extent_id = tid + ":zero", tid + ":extent"
        family_boundaries = {
            "zero": NativeCommonSourceBoundary(zero_id, tid, "original_quarter_start", I(0), F(0),
                                                 _encoded(("literal_source_zero", tid)).decode(), (), "0"),
            "extent": NativeCommonSourceBoundary(extent_id, tid, "original_quarter_end", I(extent), extent,
                                                   _encoded(("literal_source_extent", tid)).decode(), (), "literal_source_Gamma"),
        }
        for name, raw, aliases in internal[parent]:
            context = (_canonical_ray_key(raw) if raw is not None else
                       ("exact_terminal_curve_u1", _encoded(owner_histories[0]).decode(),
                        _encoded(owner_histories[1]).decode(), _digest(material["sweep_certificate"]),
                        "q=(height_numerator(1)/A1(1),-B1(1)/A1(1),1)"))
            family_boundaries[name] = NativeCommonSourceBoundary(tid + ":" + name, tid,
                "exact_projective_ray" if raw is not None else "exact_terminal_envelope_expression",
                endpoint_bounds[name], None, _encoded(context).decode(), aliases,
                "Gamma*atan2(original_T dot Q0_inverse_direction,original_S dot Q0_inverse_direction)/(analytic_pi/2)")
        boundaries.extend(family_boundaries.values())
        own_cells = []
        for low_name, high_name, pair in partitions[parent]:
            low, high = family_boundaries[low_name], family_boundaries[high_name]
            cut = cold[pair] if pair is not None else None
            cell_id = cut.cut_id if cut else "inactive:" + parent
            width = high.computational_angle_bounds_radians - low.computational_angle_bounds_radians
            if width.lo <= 0:
                raise ValueError("Every retained symbolic support has strictly positive continuous source measure")
            expression = ((low.endpoint_id, -C), (high.endpoint_id, C))
            cell = NativeCommonSourceAllocationCell(cell_id, tid, parent, cut is not None,
                cut.cut_id if cut else None, cut.cut_snapshot_sha256 if cut else None,
                low.endpoint_id, high.endpoint_id,
                (low.computational_angle_bounds_radians, high.computational_angle_bounds_radians),
                expression, C * width, cut.common_physical_radius_cubed_bounds_m3 if cut else None,
                profile, trace_memory_hash)
            own_cells.append(cell)
        full_expression = ((zero_id, -C), (extent_id, C))
        summed = _telescoped_endpoint_coefficients(cell.source_area_endpoint_linear_expression for cell in own_cells)
        if summed != _telescoped_endpoint_coefficients((full_expression,)):
            raise ValueError("The original continuous source budget must close by exact shared-endpoint cancellation")
        families.append(NativeCommonSourceFamily(tid, parent, binding["donor_fragment_id"], extent, C, C*extent,
            R, height, sine, profile, sum(profile, F(0)) / len(profile), _encoded(traces[tid]).decode(),
            memory_json, _encoded(anchors[tid]).decode(), trace_memory_hash,
            tuple(atom["point_id"] for atom in family_atoms), tuple(cell.cell_id for cell in own_cells),
            summed, _telescoped_endpoint_coefficients((full_expression,)),
            tuple(receipts[pair] for pair in sorted(cold) if pair[1] == parent)))
        cells.extend(own_cells)
    ledger, buckets = [], {}
    for atom in atoms:
        tid, point_id = atom["trace_id"], atom["point_id"]
        binding = next(binding for binding in bindings if binding["trace_id"] == tid)
        parent = binding["receiver_fragment_id"]
        angle = _read_fraction(atom["computational_angle"])
        if (not 0 < angle < extents[parent] or states.get(point_id) != atom["unaccepted_candidate_state"]
                or atom["original_area_m2"] != atom["original_point"]["area_m2"]
                or atom["original_area_m2"] != states[point_id]["area_m2"]):
            raise ValueError("Every literal atom area, angle, history and unaccepted endpoint must match the source")
        bucket, reason = _classify(parent, angle, endpoint_bounds)
        buckets.setdefault(bucket, []).append(atom)
        state = states[point_id]
        ledger.append(NativeCommonOriginalAtomLedgerEntry(point_id, tid, parent, binding["donor_fragment_id"],
            angle, _literal(atom["original_area_m2"]), _encoded(atom).decode(), _encoded(state).decode(),
            bucket, reason, "unaccepted_candidate", _literal(state["elapsed_time_s"]), state["step_count"]))
    contact_clocks = {(entry.source_contact_elapsed_time_s, entry.source_contact_step_count) for entry in ledger}
    if len(contact_clocks) != 1:
        raise ValueError("All source atoms must preserve their one authentic contact clock")
    all_budget = _budget(atoms)
    bucket_budgets = tuple((label, _budget(subset)) for label, subset in sorted(buckets.items()))
    if sum(budget.count for _, budget in bucket_budgets) != 16:
        raise ValueError("Source angular classification must retain every atom exactly once")
    for epoch_index in range(len(SOURCE_EPOCHS)):
        for field_index in range(len(STORE_FIELDS)):
            if sum((budget.source_epoch_work_and_energy_j[epoch_index][1][field_index][1]
                    for _, budget in bucket_budgets), F(0)) != all_budget.source_epoch_work_and_energy_j[epoch_index][1][field_index][1]:
                raise ValueError("Every literal original energy and work store must close exactly once")
    contact_time, contact_steps = next(iter(contact_clocks))
    value = NativeCommonSourceAllocation(reference.source_snapshot_sha256, reference.source_epoch_sha256,
        reference.reference_snapshot_sha256, cuts.set_snapshot_sha256, history["binding_sha256"],
        _digest(memory), _digest(snapshot["states"]), reference.complete_source_dto_json,
        reference.saved_field_json, _encoded(memory).decode(), _encoded(history).decode(),
        _encoded(snapshot["states"]).decode(), _encoded(bindings).decode(),
        _literal(snapshot["elapsed_time_s"]), _literal(memory["elapsed_time_s"]), memory["stage_count"],
        contact_time, contact_steps, "unaccepted_candidate", tuple(families), tuple(boundaries), tuple(cells),
        tuple(ledger), all_budget, bucket_budgets,
        sum(entry.source_accounting_bucket.startswith("retained-contact:") for entry in ledger),
        sum(entry.source_accounting_bucket.startswith("inactive:") for entry in ledger),
        sum(entry.source_accounting_bucket.startswith("uncertain:") for entry in ledger),
        hot[0].cut_snapshot_sha256, provenance, precision, "")
    value = replace(value, binding_sha256=_seal(value))
    _REGISTRY[value.binding_sha256] = _encoded(value)
    return value


def validate_native_common_source_allocation(value, *, reference=None, cuts=None):
    if (type(value) is not NativeCommonSourceAllocation or value.binding_sha256 != _seal(value)
            or _REGISTRY.get(value.binding_sha256) != _encoded(value)):
        raise ValueError("An altered or re-sealed source allocation lacks complete authentic replay")
    if reference is not None:
        reference = validate_native_common_reference(reference)
        if value.reference_snapshot_sha256 != reference.reference_snapshot_sha256:
            raise ValueError("Source allocation belongs to another COMMON reference")
    if cuts is not None:
        cuts = validate_common_contact_cuts(cuts, reference=reference)
        if value.cut_set_snapshot_sha256 != cuts.set_snapshot_sha256:
            raise ValueError("Source allocation cannot borrow another actual cut collection")
    return value


def native_common_source_allocation_cell_for_cut(allocation, cut_or_id):
    allocation = validate_native_common_source_allocation(allocation)
    cut_id = cut_or_id if type(cut_or_id) is str else validate_common_contact_cut(cut_or_id).cut_id
    matches = [cell for cell in allocation.cells if cell.active and cell.cut_id == cut_id]
    if len(matches) != 1:
        raise ValueError("An unallocated or HOT cut has no original receiver source budget")
    cell = matches[0]
    if type(cut_or_id) is not str and cell.cut_snapshot_sha256 != cut_or_id.cut_snapshot_sha256:
        raise ValueError("Source support belongs to another authenticated cut")
    return cell


def validate_native_common_source_allocation_measure(allocation, declaration, *, reference=None,
                                                   port_response=None, state=None):
    """Link the existing density directly; no second support-fraction weight."""
    allocation = validate_native_common_source_allocation(allocation, reference=reference)
    declaration = validate_native_common_contact_reference_measure_declaration(declaration,
        reference=reference, port_response=port_response, state=state)
    cells = [cell for cell in allocation.cells if cell.active and cell.cut_snapshot_sha256 == declaration.cut_snapshot_sha256]
    if len(cells) != 1:
        raise ValueError("A density requires exactly its assigned genuine cold receiver support")
    cell = cells[0]
    family = next(family for family in allocation.families if family.trace_id == cell.trace_id)
    if (declaration.source_snapshot_sha256 != allocation.source_snapshot_sha256
            or declaration.source_epoch_sha256 != allocation.source_epoch_sha256
            or declaration.reference_snapshot_sha256 != allocation.reference_snapshot_sha256
            or declaration.receiving_trace_id != family.trace_id
            or declaration.depth_profile_coefficients != family.depth_profile_coefficients
            or declaration.source_current_memory_sha256 != allocation.source_current_memory_sha256
            or declaration.source_current_atom_states_sha256 != allocation.source_current_atom_states_sha256
            or declaration.source_current_memory_trace_sha256 != family.source_memory_family_trace_sha256
            or declaration.source_history_binding_sha256 != allocation.source_history_binding_sha256
            or declaration.source_computational_extent_radians != family.source_computational_extent_radians
            or declaration.source_interface_area_density_m2_per_computational_radian != family.source_interface_area_density_m2_per_computational_radian
            or declaration.source_reference_interface_area_m2 != family.source_reference_interface_area_m2
            or declaration.common_physical_radius_cubed_bounds_m3 != cell.common_physical_radius_cubed_bounds_m3
            or (declaration.source_native_elapsed_time_s, declaration.source_memory_elapsed_time_s,
                declaration.source_memory_stage_count, declaration.source_contact_elapsed_time_s,
                declaration.source_contact_step_count, declaration.source_current_node_epoch)
               != (allocation.source_native_elapsed_time_s, allocation.source_memory_elapsed_time_s,
                   allocation.source_memory_stage_count, allocation.source_contact_elapsed_time_s,
                   allocation.source_contact_step_count, allocation.source_current_node_epoch)):
        raise ValueError("A measure cannot change family profile, source memory, support or authentic clocks")
    return cell


def native_common_source_allocation_to_dict(value):
    value = validate_native_common_source_allocation(value)
    return dict(format=FORMAT, model=MODEL, scope=SCOPE, **_payload(value), **_SCOPE_FLAGS)


def native_common_source_allocation_from_dict(raw, *, reference, cuts):
    if type(raw) is not dict or raw.get("format") != FORMAT or type(raw.get("precision")) is not int:
        raise ValueError("Expected a complete typed whole-source allocation receipt")
    profiles = {row["trace_id"]: tuple(_read_fraction(x) for x in row["depth_profile_coefficients"])
                for row in raw["families"]}
    value = bind_native_common_source_allocation(reference, cuts, depth_profiles=profiles,
                                               provenance=raw["provenance"], precision=raw["precision"])
    if _encoded(raw) != _encoded(native_common_source_allocation_to_dict(value)):
        raise ValueError("Serialized source allocation differs from complete source, geometry, memory or work replay")
    return value


@dataclass(frozen=True, slots=True)
class NativeAllocatedSourceReferenceDensity:
    allocation_binding_sha256: str
    source_snapshot_sha256: str
    reference_snapshot_sha256: str
    cut_snapshot_sha256: str
    geometry_bounds_sha256: str
    receiving_trace_id: str
    receiving_cell_id: str
    source_current_memory_sha256: str
    source_memory_family_trace_sha256: str
    source_interface_area_density_m2_per_computational_radian: F
    source_computational_extent_radians: F
    depth_profile_coefficients: tuple
    full_physical_volume_depth_bounds_m: I
    full_physical_volume_depth_span_m: F
    normalized_depth_bounds: I
    depth_profile_value_bounds: I
    receiver_projection_coordinate_bounds: tuple
    receiver_projection_coordinate_parameter_derivative_bounds: tuple
    general_receiving_native_angle_parameter_derivative_bounds: I
    receiving_native_angle_parameter_derivative_bounds: I
    receiving_computational_angle_parameter_derivative_bounds: I
    native_angle_derivative_identity: str
    reference_area_density_bounds_m2_per_parameter_per_physical_volume_depth_m: I
    physical_area_density_bounds_m2_per_parameter_per_physical_volume_depth_m: I
    reference_to_physical_area_density_ratio_bounds: I
    precision: int
    binding_sha256: str
    extra_support_fraction_multiplier_applied: bool = False
    finite_point_area_assigned: bool = False
    globally_integrated_force_solution: bool = False
    physical_material_admission_authorized: bool = False


def evaluate_allocated_source_reference_density(allocation, reference, cut_bounds, *, precision=160):
    """Enclose a point reference density using its once-only assigned support.

    The native angle derivative is exactly -1 by the authenticated receiver
    plane/orientation proof.  Its independent projective differential is also
    retained as an outward check.  Physical area is a distinct mapped measure.
    """
    if type(precision) is not int or not 96 <= precision <= 512:
        raise ValueError("Allocated source density precision must be an integer from 96 to 512")
    allocation = validate_native_common_source_allocation(allocation, reference=reference)
    reference = validate_native_common_reference(reference)
    geometry = validate_common_contact_cut_bounds(cut_bounds, reference=reference)
    cell = native_common_source_allocation_cell_for_cut(allocation, geometry.cut)
    if not geometry.genuine_query_support_certified:
        raise ValueError("An unresolved material band has no allocated point reference density")
    family = next(row for row in allocation.families if row.trace_id == cell.trace_id)
    runtime = _RUNTIME[geometry.cut.cut_snapshot_sha256]
    trace = json.loads(family.complete_source_trace_json)
    start, end = (tuple(I(_literal(x)) for x in trace[key]) for key in ("start", "end"))
    q = geometry.receiver_owner_local_unit_direction_bounds
    qp = _inverse_map(runtime["poses"][0], geometry.source_unit_world_direction_parameter_derivative_bounds)
    x, y, dx, dy = _dot(start, q), _dot(end, q), _dot(start, qp), _dot(end, qp)
    squared_radius = _square(x) + _square(y)
    if squared_radius.lo <= 0:
        raise ValueError("The genuine receiver projective coordinate must remain regular")
    general_derivative = (x * dy - y * dx) / squared_radius
    receipt = next((row for row in family.native_angle_orientation_receipts
                    if row[0] == geometry.cut.cut_id and row[1] == geometry.cut.cut_snapshot_sha256), None)
    if receipt is None or receipt[-1] != -1 or not general_derivative.lo <= -1 <= general_derivative.hi:
        raise ValueError("The allocated native derivative lacks its exact source plane/orientation identity")
    derivative = I(-1)
    computational_derivative = derivative * family.source_computational_extent_radians / (interval_pi(precision + 8) / 2)
    R, cube = family.source_radius_m, cell.common_physical_radius_cubed_bounds_m3
    full_tau = I((R**3 - cube.hi) / (3 * R**2), (R**3 - cube.lo) / (3 * R**2))
    span = full_tau.hi - full_tau.lo
    if span <= 0:
        raise ValueError("An allocated physical cut needs its complete positive volume-depth span")
    z = (geometry.common_physical_volume_depth_bounds_m - full_tau.lo) / span
    profile_value = _profile_bounds(family.depth_profile_coefficients, z)
    density = family.source_interface_area_density_m2_per_computational_radian * (-computational_derivative) * profile_value / span
    physical = geometry.area_density_bounds_m2_per_parameter_per_material_m
    if physical.lo <= 0:
        raise ValueError("The mapped physical area density must remain strictly positive")
    value = NativeAllocatedSourceReferenceDensity(allocation.binding_sha256, allocation.source_snapshot_sha256,
        allocation.reference_snapshot_sha256, geometry.cut.cut_snapshot_sha256, geometry.bounds_snapshot_sha256,
        family.trace_id, cell.cell_id, allocation.source_current_memory_sha256,
        family.source_memory_family_trace_sha256, family.source_interface_area_density_m2_per_computational_radian,
        family.source_computational_extent_radians, family.depth_profile_coefficients, full_tau, span, z, profile_value,
        (x, y), (dx, dy), general_derivative, derivative, computational_derivative,
        "exact_original_receiver_quarter_plane_and_positive_native_orientation_theta_prime_minus_one",
        density, physical, density / physical, precision, "")
    value = replace(value, binding_sha256=_seal(value))
    _REGISTRY[type(value), value.binding_sha256] = _encoded(value)
    return value


def validate_allocated_source_reference_density(value, *, allocation=None, reference=None, cut_bounds=None):
    if (type(value) is not NativeAllocatedSourceReferenceDensity or value.binding_sha256 != _seal(value)
            or _REGISTRY.get((type(value), value.binding_sha256)) != _encoded(value)):
        raise ValueError("An altered or re-sealed allocated source density lacks authentic replay")
    if allocation is not None:
        allocation = validate_native_common_source_allocation(allocation, reference=reference)
        if value.allocation_binding_sha256 != allocation.binding_sha256:
            raise ValueError("An allocated density belongs to another family model or source allocation")
    if reference is not None and value.reference_snapshot_sha256 != validate_native_common_reference(reference).reference_snapshot_sha256:
        raise ValueError("An allocated density belongs to another COMMON reference")
    if cut_bounds is not None:
        geometry = validate_common_contact_cut_bounds(cut_bounds, reference=reference)
        if value.geometry_bounds_sha256 != geometry.bounds_snapshot_sha256:
            raise ValueError("An allocated density belongs to another genuine cut query")
    return value


def allocated_source_reference_density_to_dict(value):
    value = validate_allocated_source_reference_density(value)
    return dict(format=FORMAT, model=MODEL, scope=SCOPE, **_payload(value))


def allocated_source_reference_density_from_dict(raw, *, allocation, reference, cut_bounds):
    if type(raw) is not dict or raw.get("format") != FORMAT or type(raw.get("precision")) is not int:
        raise ValueError("Expected a complete allocated source reference density receipt")
    value = evaluate_allocated_source_reference_density(allocation, reference, cut_bounds, precision=raw["precision"])
    if _encoded(raw) != _encoded(allocated_source_reference_density_to_dict(value)):
        raise ValueError("Serialized allocated source density differs from source, family profile or true cut geometry")
    return value


@dataclass(frozen=True, slots=True)
class NativeCommonAllocatedSourceMemoryQuery:
    allocation_binding_sha256: str
    source_snapshot_sha256: str
    source_epoch_sha256: str
    source_history_binding_sha256: str
    source_current_memory_sha256: str
    source_memory_family_trace_sha256: str
    trace_id: str
    source_computational_extent_radians: F
    computational_angle_bounds: I
    source_native_elapsed_time_s: F
    source_memory_elapsed_time_s: F
    source_memory_stage_count: int
    source_contact_elapsed_time_s: F
    source_contact_step_count: int
    source_current_node_epoch: str
    memory_piece_intersections: tuple
    plastic_slip_bounds_m: I
    cumulative_slip_bounds_m: I
    precision: int
    binding_sha256: str
    literal_current_memory_used: bool = True
    all_closed_current_piece_intersections_retained: bool = True
    current_memory_replaced_with_previous_or_dense_query: bool = False
    three_dimensional_material_memory_advection_certified: bool = False
    source_state_or_thermal_state_advanced: bool = False
    physical_material_admission_authorized: bool = False


def evaluate_native_common_source_memory(allocation, trace_id, computational_angle_bounds, *, precision=96):
    """Enclose the authentic stored CURRENT scalar family, including inactive.

    Closed source seams retain both original pieces.  This is a one-dimensional
    source-memory query; neither a depth density nor a new material state is
    inferred from the complete family accounting.
    """
    if type(precision) is not int or not 96 <= precision <= 512:
        raise ValueError("Allocated source memory precision must be an integer from 96 to 512")
    allocation = validate_native_common_source_allocation(allocation)
    matches = [family for family in allocation.families if family.trace_id == trace_id]
    if type(trace_id) is not str or len(matches) != 1:
        raise ValueError("A memory query requires one authentic original receiving family")
    family, angle = matches[0], computational_angle_bounds
    if type(angle) is not I or not 0 <= angle.lo <= angle.hi <= family.source_computational_extent_radians:
        raise ValueError("A source memory query must lie entirely in the original computational extent")
    current = json.loads(family.complete_current_memory_trace_json)
    intersections = []
    for index, piece in enumerate(current["pieces"]):
        low = max(angle.lo, _literal(piece["low_radians"]))
        high = min(angle.hi, _literal(piece["high_radians"]))
        if low > high:
            continue
        plastic, cumulative = (tuple(_literal(x) for x in piece[key]["coefficients"])
                               for key in ("plastic_slip", "cumulative_slip"))
        intersections.append(NativeCommonCurrentMemoryPiece(index, I(low, high), _encoded(piece).decode(),
            _rounded_interval(_field_bounds(plastic, low, high, precision), precision + 16),
            _rounded_interval(_field_bounds(cumulative, low, high, precision), precision + 16)))
    if not intersections:
        raise ValueError("No actual current source piece covers the memory query")
    reached = angle.lo
    for low, high in sorted((piece.computational_angle_bounds.lo, piece.computational_angle_bounds.hi)
                            for piece in intersections):
        if low > reached:
            raise ValueError("The actual current source memory query has an uncovered interval")
        reached = max(reached, high)
    if reached != angle.hi:
        raise ValueError("The entire actual current source memory interval must remain covered")
    plastic = I(min(piece.plastic_slip_bounds_m.lo for piece in intersections),
                max(piece.plastic_slip_bounds_m.hi for piece in intersections))
    cumulative = I(min(piece.cumulative_slip_bounds_m.lo for piece in intersections),
                   max(piece.cumulative_slip_bounds_m.hi for piece in intersections))
    if cumulative.lo < 0:
        raise ValueError("The whole current source cumulative-slip query must remain nonnegative")
    value = NativeCommonAllocatedSourceMemoryQuery(allocation.binding_sha256, allocation.source_snapshot_sha256,
        allocation.source_epoch_sha256, allocation.source_history_binding_sha256,
        allocation.source_current_memory_sha256, family.source_memory_family_trace_sha256,
        trace_id, family.source_computational_extent_radians, angle, allocation.source_native_elapsed_time_s,
        allocation.source_memory_elapsed_time_s, allocation.source_memory_stage_count,
        allocation.source_contact_elapsed_time_s, allocation.source_contact_step_count,
        allocation.source_current_node_epoch, tuple(intersections), plastic, cumulative, precision, "")
    value = replace(value, binding_sha256=_seal(value))
    _REGISTRY[type(value), value.binding_sha256] = _encoded(value)
    return value


def validate_native_common_source_memory_query(value, *, allocation=None):
    if (type(value) is not NativeCommonAllocatedSourceMemoryQuery or value.binding_sha256 != _seal(value)
            or _REGISTRY.get((type(value), value.binding_sha256)) != _encoded(value)):
        raise ValueError("An altered or re-sealed source memory query lacks actual current-family replay")
    if allocation is not None:
        allocation = validate_native_common_source_allocation(allocation)
        if value.allocation_binding_sha256 != allocation.binding_sha256:
            raise ValueError("A source memory query belongs to another full source allocation")
    return value


def native_common_source_memory_query_to_dict(value):
    value = validate_native_common_source_memory_query(value)
    return dict(format=FORMAT, model=MODEL, scope=SCOPE, **_payload(value))


def native_common_source_memory_query_from_dict(raw, *, allocation):
    if type(raw) is not dict or raw.get("format") != FORMAT or type(raw.get("precision")) is not int:
        raise ValueError("Expected a complete authentic CURRENT source-memory query")
    value = evaluate_native_common_source_memory(allocation, raw["trace_id"],
        _read_interval(raw["computational_angle_bounds"]), precision=raw["precision"])
    if _encoded(raw) != _encoded(native_common_source_memory_query_to_dict(value)):
        raise ValueError("Serialized source memory differs from actual current pieces, literal coefficients or clocks")
    return value
