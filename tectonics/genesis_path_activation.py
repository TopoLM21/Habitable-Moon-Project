"""One local, traction-preserving contact birth on a prescribed support.

The support and its quadrature are given, not inferred fracture localization.
At a resolved strength event only the two bank DOFs of one interior vertex
are released. Its incident endpoint tractions replace the measured constraint
reaction. The companion endpoint may be below peak strength; its constitutive
law retains an elastic segment to the unchanged peak.

This fixed-depth research owner implements one birth and subsequent local
mechanics. It does not propagate fronts or monitor a second strength event.
Birth in compression is deliberately unsupported. Material cohort clocks are
preserved; the distinct contact-activation clock is saved in this owner.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, fields, replace
import hashlib
import json
from numbers import Real

import numpy as np

from .genesis_contact_growth import CohortState, validate_cohorts
from .genesis_crack_path import CrackInterval
from .genesis_extrinsic_contact_law import (extrinsic_damage,
    extrinsic_dissipation, extrinsic_energy, extrinsic_return_map)
from .genesis_path_birth import recover_tied_tractions, classify_tied_onset
from .genesis_path_dynamics import PathMechanics, PathState, _PathRetry


VERSION = "genesis-extrinsic-path-0.1"


def _owned(value):
    array = np.ascontiguousarray(value, dtype=float)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _finite(value, shape, name):
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in "fiu" or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite real values with shape {shape}")
    return np.asarray(array, dtype=float)


def _selected_traces(model, interval):
    free = model.basis.free_dofs(interval)
    released = free[free >= model.basis.nparent]
    touched = np.asarray(model.jump_operator[:, released].power(2).sum(axis=1)).ravel()
    return released, touched.reshape(-1, 2).sum(axis=1) > 0


class ExtrinsicPathMechanics(PathMechanics):
    """Same bulk solver, with immutable initial tractions and a distinct law.

    Use ``activate_tied_onset`` to obtain a force-consistent initial state.
    Direct construction exists for checkpoint restoration, not activation.
    The signed interface potential is relative to birth; in particular the
    shear contribution can reach -tau0**2/(2*Kt). That reference is neither
    extracted bulk energy nor heat nor Mode-I fracture work.
    """

    def __init__(self, tied_model, birth_traction_pa, birth_water,
                 birth_interval, birth_time_years):
        if type(tied_model) is not PathMechanics:
            raise ValueError("Extrinsic birth requires an ordinary tied PathMechanics owner")
        if not isinstance(birth_interval, CrackInterval):
            raise ValueError("Birth interval must be a resolved CrackInterval")
        if (isinstance(birth_time_years, (bool, np.bool_))
                or not isinstance(birth_time_years, Real)
                or not np.isfinite(birth_time_years) or birth_time_years < 0):
            raise ValueError("Birth time must be finite and nonnegative")
        super().__init__(tied_model.basis, tied_model.depth_m, tied_model.parameters,
            tied_model.law_parameters, geometry_parameters=tied_model.geometry_parameters)
        self.tied_fingerprint = tied_model.fingerprint
        n = len(self.trace_depth_m)
        traction = _finite(birth_traction_pa, (n, 2), "birth_traction_pa")
        water = _finite(birth_water, (n,), "birth_water")
        released, selected = _selected_traces(self, birth_interval)
        if len(released) != 2 or not selected.any():
            raise ValueError("One birth must release exactly one resolved interior vertex")
        if (np.any(traction[~selected] != 0) or np.any(traction[:, 0] < 0)
                or np.any(traction[:, 0] > self.law_parameters.tensile_strength_pa)):
            raise ValueError("Birth normal tractions must be tensile, below peak, and local")
        if np.any((water < 0) | (water > 1)):
            raise ValueError("Birth water must lie in [0, 1]")
        cohesion = self.law_parameters.cohesion_pa*(1
            -(1-self.law_parameters.wet_cohesion_fraction)*water)
        if np.any(np.abs(traction[:, 1]) > cohesion):
            raise ValueError("Birth shear tractions exceed the unchanged local strength")
        self.birth_traction_pa, self.birth_water = _owned(traction), _owned(water)
        self.birth_interval, self.birth_time_years = birth_interval, float(birth_time_years)
        digest = hashlib.sha256(self.fingerprint.encode())
        digest.update(json.dumps({"version": VERSION, "interval": asdict(birth_interval),
            "birth_time_years": self.birth_time_years}, sort_keys=True).encode())
        digest.update(self.birth_traction_pa.tobytes())
        digest.update(self.birth_water.tobytes())
        self.fingerprint = digest.hexdigest()

    def initial(self, *args, **kwargs):
        raise ValueError("Use activate_tied_onset to replace measured tied reactions")

    def release(self, *args, **kwargs):
        raise ValueError("Additional contact birth or front propagation is not implemented")

    def _validate_state(self, state):
        super()._validate_state(state)
        if (state.active_interval != self.birth_interval or state.accepted_steps < 1
                or state.elapsed_years < self.birth_time_years):
            raise ValueError("State disagrees with its extrinsic birth interval or clock")
        clock_tolerance = 128*np.finfo(float).eps*max(abs(self.birth_time_years), 1.)
        if np.any(state.cohorts.birth_time_myr*1e6 > self.birth_time_years+clock_tolerance):
            raise ValueError("Fixed interface material must exist before contact activation")
        if state.elapsed_years == self.birth_time_years:
            if (np.any(self.jump_operator@state.displacement_m != 0)
                    or any(np.any(getattr(state.cohorts, name) != 0) for name in
                        ("plastic_slip_m", "cumulative_slip_m", "max_opening_m", "damage",
                         "friction_work_j", "viscous_work_j", "fracture_work_j", "shear_remainder_j"))):
                raise ValueError("Contact birth cannot contain post-birth motion or work")

    def _validate_contact_history(self, cohorts):
        validate_cohorts(cohorts, None, len(self.trace_depth_m))
        expected_damage = extrinsic_damage(cohorts.max_opening_m,
            self.birth_traction_pa, self.law_parameters)
        expected_work = cohorts.area_ref_m2*extrinsic_dissipation(
            cohorts.max_opening_m, self.birth_traction_pa, self.law_parameters)
        if not np.allclose(cohorts.damage, expected_damage, rtol=1e-12, atol=1e-12):
            raise ValueError("Extrinsic damage disagrees with opening history")
        if not np.allclose(cohorts.fracture_work_j, expected_work, rtol=2e-10, atol=1e-6):
            raise ValueError("Extrinsic fracture work disagrees with opening history")

    def _stored_contact_traction(self, cohorts, gap, jump):
        c, law = cohorts, self.law_parameters
        normal = (1-c.damage)*self.birth_traction_pa[:, 0]+law.normal_stiffness_pa_m*np.where(
            gap < 0, gap, (1-c.damage)*gap)
        shear = self.birth_traction_pa[:, 1]+law.tangential_stiffness_pa_m*(jump-c.plastic_slip_m)
        free = (gap >= 0) & (c.damage >= 1)
        reference = jump+self.birth_traction_pa[:, 1]/law.tangential_stiffness_pa_m
        if not np.allclose(c.plastic_slip_m[free], reference[free], rtol=1e-10, atol=1e-10):
            raise ValueError("Free-open extrinsic contact has an inconsistent plastic reference")
        shear[free] = 0.
        return np.column_stack((normal, shear))

    def _evaluate_contact(self, cohorts, gap, jump, dt, water):
        c = cohorts
        response = extrinsic_return_map(gap, jump, c.plastic_slip_m, c.cumulative_slip_m,
            c.max_opening_m, dt, water, self.birth_traction_pa, self.law_parameters,
            old_damage=c.damage)
        arrays = {name: response[name].copy() for name in
            ("plastic_slip_m", "cumulative_slip_m", "max_opening_m", "damage", "traction_pa")}
        for stored, local in (("friction_work_j", "friction_work_j_m2"),
                ("viscous_work_j", "viscous_work_j_m2"),
                ("fracture_work_j", "fracture_work_j_m2"),
                ("shear_remainder_j", "shear_relaxation_remainder_j_m2")):
            arrays[stored] = getattr(c, stored)+c.area_ref_m2*response[local]
        after = replace(c, **arrays)
        return (after, c.area_ref_m2[:, None]*response["traction_pa"],
                c.area_ref_m2[:, None, None]*response["tangent_pa_m"])

    def _contact_energy(self, cohorts, gap, jump):
        energy = extrinsic_energy(gap, jump, cohorts.plastic_slip_m,
            cohorts.max_opening_m, self.birth_traction_pa, self.law_parameters)
        return float(cohorts.area_ref_m2@energy)

    def save_state(self, path, state):
        """Self-describing birth law and mechanics; thermal loading stays external."""
        self._validate_state(state)
        arrays = {"birth_traction_pa": self.birth_traction_pa, "birth_water": self.birth_water}
        metadata = {"version": VERSION, "fingerprint": self.fingerprint,
            "tied_fingerprint": self.tied_fingerprint,
            "birth_time_years": self.birth_time_years,
            "birth_interval": asdict(self.birth_interval)}
        for field in fields(state):
            value = getattr(state, field.name)
            if isinstance(value, np.ndarray):
                arrays[field.name] = value
            elif field.name == "cohorts":
                arrays.update({"cohort_"+f.name: getattr(value, f.name) for f in fields(value)})
            elif field.name == "active_interval":
                metadata[field.name] = asdict(value)
            else:
                metadata[field.name] = value
        arrays["metadata"] = np.array(json.dumps(metadata, allow_nan=False, sort_keys=True))
        np.savez_compressed(path, **arrays)

    def load_state(self, path):
        metadata, arrays = _read_checkpoint(path)
        if (metadata.pop("fingerprint") != self.fingerprint
                or metadata.pop("tied_fingerprint") != self.tied_fingerprint
                or metadata.pop("birth_time_years") != self.birth_time_years
                or metadata.pop("birth_interval") != asdict(self.birth_interval)
                or not np.array_equal(arrays.pop("birth_traction_pa"), self.birth_traction_pa)
                or not np.array_equal(arrays.pop("birth_water"), self.birth_water)):
            raise ValueError("Checkpoint belongs to a different extrinsic birth or model")
        metadata.pop("version")
        try:
            interval = CrackInterval(**metadata.pop("active_interval"))
            cohorts = CohortState(**{f.name: arrays.pop("cohort_"+f.name) for f in fields(CohortState)})
            state = PathState(**metadata, **arrays, cohorts=cohorts, active_interval=interval)
            self._validate_state(state)
            self._check_geometry(state, state)
        except (TypeError, KeyError, AttributeError, _PathRetry) as exc:
            raise ValueError("Malformed or inadmissible extrinsic checkpoint") from exc
        return state


def _read_checkpoint(path):
    state_arrays = {"displacement_m", "elastic_strain", "constraint_reaction_n"}
    cohort_arrays = {"cohort_"+f.name for f in fields(CohortState)}
    birth_arrays = {"birth_traction_pa", "birth_water"}
    scalar_names = {f.name for f in fields(PathState)}-state_arrays-{"cohorts"}
    birth_names = {"version", "fingerprint", "tied_fingerprint", "birth_time_years", "birth_interval"}
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != state_arrays | cohort_arrays | birth_arrays | {"metadata"}:
            raise ValueError("Extrinsic checkpoint has missing or unexpected arrays")
        encoded = data["metadata"]
        if encoded.shape != () or encoded.dtype.kind not in "US":
            raise ValueError("Extrinsic checkpoint metadata must be scalar JSON")
        metadata = json.loads(str(encoded))
        if (not isinstance(metadata, dict) or set(metadata) != scalar_names | birth_names
                or metadata["version"] != VERSION):
            raise ValueError("Extrinsic checkpoint has invalid metadata or version")
        arrays = {name: data[name].copy() for name in data.files if name != "metadata"}
    return metadata, arrays


def load_born_path(tied_model, path):
    """Restore birth context against the independently reconstructed tied mesh."""
    metadata, arrays = _read_checkpoint(path)
    if metadata["tied_fingerprint"] != tied_model.fingerprint:
        raise ValueError("Extrinsic checkpoint belongs to different tied mechanics")
    try:
        model = ExtrinsicPathMechanics(tied_model, arrays["birth_traction_pa"], arrays["birth_water"],
            CrackInterval(**metadata["birth_interval"]), metadata["birth_time_years"])
    except (TypeError, KeyError) as exc:
        raise ValueError("Invalid extrinsic birth metadata") from exc
    return model, model.load_state(path)


@dataclass(frozen=True)
class PathBirthTransition:
    model: ExtrinsicPathMechanics
    state: PathState
    governing_trace: int
    governing_mode: str
    strength_ratio: float
    released_dof_count: int
    replacement_force_relative_error: float
    interface_energy_change_j: float


def activate_tied_onset(model, state, stress_pa, water_per_trace, *, strength_tolerance=1e-6):
    """Commit one admissible local event, without clipping a traction or peak.

    Use a lower bracket sample sufficiently close to strength. Even a tiny
    overshoot is refused; an under-resolved time bracket must be refined by
    its owner. Other simultaneously critical vertices remain constrained and
    are not interpreted as an evolved network. The owner must resolve those
    events before using this local experiment as a global evolution model.
    """
    if (isinstance(strength_tolerance, (bool, np.bool_))
            or not isinstance(strength_tolerance, Real)
            or not np.isfinite(strength_tolerance) or not 0 < strength_tolerance <= 1e-3):
        raise ValueError("Birth strength_tolerance must lie in (0, 1e-3]")
    if type(model) is not PathMechanics:
        raise ValueError("One contact birth requires an ordinary all-tied model")
    recovery = recover_tied_tractions(model, state, stress_pa)
    onset = classify_tied_onset(recovery, water_per_trace, model.law_parameters,
        relative_tolerance=strength_tolerance)
    score = np.maximum(onset.normal_ratio, onset.shear_ratio)
    observed = recovery.trace_observed
    maximum = onset.maximum_observed_ratio
    if not np.isfinite(maximum) or maximum > 1.:
        raise ValueError("Birth requires an admissible lower sample, without strength overshoot")
    if maximum < 1-strength_tolerance:
        raise ValueError("Birth sample has not reached the strength surface")
    trace = int(np.flatnonzero(observed & (score == maximum))[0])
    columns = model.jump_operator[2*trace:2*trace+2].nonzero()[1]
    vertices = np.unique((columns-model.basis.nparent)//2)
    if len(vertices) != 1 or vertices[0] < 0:
        raise ValueError("Governing trace does not select one interior enrichment vertex")
    vertex = model.basis.enrichment_vertices[vertices[0]]
    order = np.flatnonzero(model.basis.insertion.path_vertex_ids == vertex)
    if len(order) != 1 or order[0] in (0, len(model.basis.insertion.path_vertex_ids)-1):
        raise ValueError("Governing vertex has no resolved neighboring fronts")
    arc, index = model.basis.insertion.path_arclength_m, int(order[0])
    interval = CrackInterval(float(arc[index-1]), float(arc[index+1]))
    released, selected = _selected_traces(model, interval)
    traction = np.zeros_like(recovery.traction_pa)
    traction[selected] = recovery.traction_pa[selected]
    born_model = ExtrinsicPathMechanics(model, traction, water_per_trace, interval, state.elapsed_years)
    replacement_force = np.asarray(model.jump_operator.T@(
        model.geometry.interface_area_m2[:, None]*traction).ravel()).ravel()
    required = state.constraint_reaction_n[released]
    error = float(np.linalg.norm(replacement_force[released]-required)/max(np.linalg.norm(required), 1.))
    if error > 1e-10:
        raise ValueError("Birth traction fails to replace the released reaction")
    born = deepcopy(state)
    born.active_interval = interval
    born.constraint_reaction_n -= replacement_force
    born.constraint_reaction_n[born_model.basis.free_dofs(interval)] = 0.
    born.cohorts.traction_pa = traction.copy()
    born.released_reaction_norm_n = float(np.linalg.norm(required))
    born.released_reaction_measured = True
    born.equilibrium_residual = max(born.equilibrium_residual, error)
    born_model._validate_state(born)
    born_model._check_geometry(born, born)
    gap, jump = (model.jump_operator@state.displacement_m).reshape(-1, 2).T
    energy_change = born_model._contact_energy(born.cohorts, gap, jump)-model._contact_energy(state.cohorts, gap, jump)
    if energy_change != 0.:
        raise ValueError("Birth must not add interface energy at zero physical jump")
    mode = "tensile" if onset.normal_ratio[trace] >= onset.shear_ratio[trace] else "shear"
    return PathBirthTransition(born_model, born, trace, mode, float(maximum), len(released), error, energy_change)
