"""Experimental irreversible front under prescribed opening, with an energy ledger.

Only the ideal DCB elastic oracle is supported. A reference path stores material
coordinates; its shape does not turn the DCB formula into planetary mechanics.
There is one advancing (right) tip and an explicitly supplied finite seed notch.
Load steps are quasi-static experiments, not years or a propagation speed law.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from numbers import Integral, Real
from pathlib import Path

import numpy as np

from .genesis_crack_energy import DCBOracle
from .genesis_crack_path import CrackInterval, ReferenceCrackPath


VERSION = "genesis-crack-front-dcb-0.1"


def _finite(value, name, *, positive=False):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not np.isfinite(value) or value < 0 or (positive and value == 0)):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return float(value)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class FrontParameters:
    extension_m: float = 0.0001
    max_extensions_per_step: int = 100_000

    def __post_init__(self):
        object.__setattr__(self, "extension_m", _finite(self.extension_m, "Extension", positive=True))
        count = self.max_extensions_per_step
        if isinstance(count, (bool, np.bool_)) or not isinstance(count, Integral) or count < 1:
            raise ValueError("Maximum extensions must be a positive integer")
        object.__setattr__(self, "max_extensions_per_step", int(count))


@dataclass(frozen=True)
class FrontEvent:
    load_step: int
    opening_m: float
    old_right_m: float
    new_right_m: float
    released_energy_j: float
    fracture_work_j: float
    unresolved_release_j: float


@dataclass(frozen=True)
class FrontState:
    model_fingerprint: str
    interval: CrackInterval
    opening_m: float
    load_step: int
    stored_energy_j: float
    external_work_j: float
    fracture_work_j: float
    unresolved_release_j: float
    events: tuple[FrontEvent, ...]
    load_history_m: tuple[float, ...]
    stop_reason: str


@dataclass(frozen=True)
class CrackFrontModel:
    path: ReferenceCrackPath
    oracle: DCBOracle
    seed_interval: CrackInterval
    parameters: FrontParameters = FrontParameters()

    def __post_init__(self):
        if (not isinstance(self.path, ReferenceCrackPath)
                or type(self.oracle) is not DCBOracle
                or not isinstance(self.seed_interval, CrackInterval)
                or not isinstance(self.parameters, FrontParameters)):
            raise ValueError("Front needs a fixed path, explicit seed and supported DCB oracle")
        self.seed_interval.validate(self.path)
        if self.seed_interval.right_m + self.parameters.extension_m == self.seed_interval.right_m:
            raise ValueError("Extension is below coordinate floating-point resolution")
        # Reject an unusable energy scale at construction, before creating state.
        self.oracle.compliance_m_n(self.seed_interval.length_m)

    def metadata(self):
        return {"version": VERSION, "backend": "ideal_dcb_prescribed_opening",
                "advancing_tip": "right", "radius_km": self.path.radius_km,
                "points_xyz": self.path.points_xyz.tolist(),
                "path_fingerprint": self.path.fingerprint,
                "oracle": asdict(self.oracle), "seed_interval": asdict(self.seed_interval),
                "parameters": asdict(self.parameters)}

    @property
    def fingerprint(self):
        return hashlib.sha256(_json(self.metadata()).encode("utf-8")).hexdigest()

    def initial(self):
        """Start unloaded. The seed's historical creation cost is excluded."""
        return FrontState(self.fingerprint, self.seed_interval, 0., 0, 0., 0., 0., 0.,
                          (), (), "initial_seed")

    def energy_residual_j(self, state):
        """W_external = U + W_fracture + unresolved_release, from unloaded seed."""
        return (state.external_work_j-state.stored_energy_j-state.fracture_work_j
                -state.unresolved_release_j)

    def _validate_state(self, state):
        if not isinstance(state, FrontState) or state.model_fingerprint != self.fingerprint:
            raise ValueError("Front state belongs to another geometry or model")
        if not isinstance(state.interval, CrackInterval):
            raise ValueError("Invalid front interval")
        state.interval.validate(self.path)
        if (state.interval.left_m != self.seed_interval.left_m
                or state.interval.right_m < self.seed_interval.right_m):
            raise ValueError("The fixed tip or seed history was changed")
        if (type(state.load_step) is not int or state.load_step < 0
                or type(state.events) is not tuple or type(state.load_history_m) is not tuple
                or state.load_step != len(state.load_history_m)
                or state.opening_m != (state.load_history_m[-1] if state.load_step else 0.)):
            raise ValueError("Invalid front loading history")
        for opening in state.load_history_m:
            _finite(opening, "Historical opening")
        for value in (state.opening_m, state.stored_energy_j,
                      state.fracture_work_j, state.unresolved_release_j):
            _finite(value, "Front state")
        if not np.isfinite(state.external_work_j):
            raise ValueError("External work must be finite")
        expected_u = self.oracle.stored_energy_j(state.interval.length_m, state.opening_m)
        expected_cost = self.oracle.fracture_cost_j(self.seed_interval.length_m, state.interval.length_m)
        scale = max(abs(state.external_work_j), expected_u, expected_cost,
                    state.fracture_work_j, state.unresolved_release_j, np.finfo(float).tiny)
        if (abs(state.stored_energy_j-expected_u) > 1e-10*scale
                or abs(state.fracture_work_j-expected_cost) > 1e-10*scale
                or abs(self.energy_residual_j(state)) > 1e-10*scale):
            raise ValueError("Front state energy ledger is inconsistent")
        previous_right, previous_step = self.seed_interval.right_m, 0
        for event in state.events:
            if (not isinstance(event, FrontEvent) or type(event.load_step) is not int
                    or not 1 <= event.load_step <= state.load_step
                    or event.load_step < previous_step
                    or event.opening_m != state.load_history_m[event.load_step-1]
                    or event.old_right_m != previous_right
                    or event.new_right_m != min(previous_right+self.parameters.extension_m,
                                                self.path.length_m)
                    or event.new_right_m <= previous_right):
                raise ValueError("Invalid front event history")
            for value in (event.released_energy_j, event.fracture_work_j, event.unresolved_release_j):
                _finite(value, "Front event energy")
            if (event.fracture_work_j <= 0
                    or event.released_energy_j < event.fracture_work_j
                    or event.unresolved_release_j != event.released_energy_j-event.fracture_work_j):
                raise ValueError("Invalid front event energy")
            previous_right, previous_step = event.new_right_m, event.load_step
        if (previous_right != state.interval.right_m
                or abs(math.fsum(event.fracture_work_j for event in state.events)
                       -state.fracture_work_j) > 1e-10*scale
                or abs(math.fsum(event.unresolved_release_j for event in state.events)
                       -state.unresolved_release_j) > 1e-10*scale):
            raise ValueError("Front event history disagrees with the current ledger or interval")
        expected_stop = ("initial_seed" if not state.load_step else
                         "support_exhausted" if state.interval.right_m == self.path.length_m
                         else "energy_arrest")
        if state.stop_reason != expected_stop:
            raise ValueError("Invalid front stop reason")

    def advance(self, state, opening_m):
        """Load at the old crack, then test adjacent extensions at fixed opening.

        Each accepted extension spends Gc times *one* projected crack area. The
        oracle is reevaluated at every new front. Any surplus release is recorded
        separately; it is neither heat nor another fracture-energy payment.
        Exceeding the iteration budget raises without changing the input state.
        """
        self._validate_state(state)
        opening = _finite(opening_m, "Prescribed opening")
        interval = state.interval
        stored = self.oracle.stored_energy_j(interval.length_m, opening)
        external = state.external_work_j + stored-state.stored_energy_j
        fracture = state.fracture_work_j
        unresolved = state.unresolved_release_j
        additions = []
        stop = "support_exhausted"
        while interval.right_m < self.path.length_m:
            next_right = min(interval.right_m+self.parameters.extension_m, self.path.length_m)
            if next_right <= interval.right_m:
                raise ValueError("Extension is below coordinate floating-point resolution")
            trial = interval.grow(self.path, right_m=next_right)
            trial_energy = self.oracle.stored_energy_j(trial.length_m, opening)
            cost = self.oracle.fracture_cost_j(interval.length_m, trial.length_m)
            if cost <= 0:
                raise ValueError("Fracture energy is below floating-point resolution")
            release = stored-trial_energy
            if release < cost:
                stop = "energy_arrest"
                break
            if len(additions) >= self.parameters.max_extensions_per_step:
                raise RuntimeError("Front extension budget exceeded; no state accepted")
            excess = release-cost
            additions.append(FrontEvent(state.load_step+1, opening, interval.right_m,
                                        next_right, release, cost, excess))
            fracture += cost
            unresolved += excess
            interval, stored = trial, trial_energy
        result = FrontState(self.fingerprint, interval, opening, state.load_step+1,
                            stored, external, fracture, unresolved,
                            state.events+tuple(additions), state.load_history_m+(opening,), stop)
        self._validate_state(result)
        return result

    def checkpoint_data(self, state):
        """Standalone experimental format, never a Contact/Coupled checkpoint."""
        self._validate_state(state)
        # Verify every accepted/rejected load and event, not only the final sum.
        replay = self.initial()
        for opening in state.load_history_m:
            replay = self.advance(replay, opening)
        if state != replay:
            raise ValueError("Front history does not match deterministic energy replay")
        return {"model": self.metadata(), "state": asdict(state)}

    def save_checkpoint(self, filename, state):
        data = self.checkpoint_data(state)
        # Exclusive creation protects existing results, including physical saves.
        with Path(filename).open("x", encoding="utf-8") as stream:
            stream.write(_json(data)+"\n")

    @classmethod
    def load_checkpoint(cls, filename):
        def unique_pairs(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("Duplicate checkpoint key")
                result[key] = value
            return result

        def invalid_constant(value):
            raise ValueError(f"Nonfinite checkpoint constant: {value}")

        with Path(filename).open(encoding="utf-8") as stream:
            data = json.load(stream, object_pairs_hook=unique_pairs, parse_constant=invalid_constant)
        try:
            if set(data) != {"model", "state"}:
                raise ValueError("Invalid front checkpoint fields")
            metadata = data["model"]
            if (metadata["version"] != VERSION
                    or metadata["backend"] != "ideal_dcb_prescribed_opening"
                    or metadata["advancing_tip"] != "right"):
                raise ValueError("Unsupported front checkpoint")
            model = cls(ReferenceCrackPath(metadata["points_xyz"], metadata["radius_km"]),
                        DCBOracle(**metadata["oracle"]),
                        CrackInterval(**metadata["seed_interval"]),
                        FrontParameters(**metadata["parameters"]))
            if _json(metadata) != _json(model.metadata()):
                raise ValueError("Front model metadata or geometry identity was changed")
            # Reconstruct physics from the explicit load history. Comparing the
            # full canonical payload checks schemas, all events and all ledgers.
            history = data["state"]["load_history_m"]
            if not isinstance(history, list):
                raise ValueError("Invalid loading history")
            replay = model.initial()
            for opening in history:
                replay = model.advance(replay, opening)
            if _json(data["state"]) != _json(asdict(replay)):
                raise ValueError("Checkpoint front history or energy ledger failed replay")
        except (KeyError, TypeError, AttributeError, OverflowError) as error:
            raise ValueError("Malformed front checkpoint") from error
        return model, replay
