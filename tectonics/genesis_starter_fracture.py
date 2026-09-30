"""Material fracture memory carried from the starter into young plate dynamics.

The same effective loading/damage law keeps operating after the first split.
It does not create chemical material or assign daughter velocities. Existing
weak bands are consumed once, and can become new rupture candidates only after
healing below half the rupture threshold and subsequently loading again. This
hysteresis is a numerical event latch, not a new physical fatigue law.

The caller owns the thermal clock, transport, topology bookkeeping and mature
state. In particular it must disable the alternative mature tidal damage law
and advect this memory with the mature material donor map exactly once per step.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np

from .genesis_starter_topology import (
    canonicalize_plate_seeds, select_starter_cut, split_starter_band,
)
from .mesh import connected_components
from .topology import _component_span_km


FORMAT = "genesis-starter-fracture-0.1"
FIELDS = ("damage", "cooling_stress_pa", "water_access", "yield_ratio", "strength_pa")


@dataclass
class YoungFractureMemory:
    damage: np.ndarray
    cooling_stress_pa: np.ndarray
    water_access: np.ndarray
    yield_ratio: np.ndarray
    strength_pa: np.ndarray
    eligible: np.ndarray
    consumed_band: np.ndarray
    first_fracture_time_myr: float | None
    tidal_peak_mpa: float


class YoungShellFracture:
    """Shared starter damage plus conservative, kick-free rupture events.

    Arrays represent material on the current mesh. Smooth mantle/cooling
    loading and tidal geometry remain spatial forcing, as in the starter.
    Between raster commits this retains the mature solver's subcell-motion
    approximation; it is not a Lagrangian stress tensor/contact solver.
    """

    def __init__(self, model, source):
        model._validate(source)
        self.model = model
        self.time_myr = float(source.time_myr)
        self.memory = YoungFractureMemory(
            **{name: getattr(source, name).copy() for name in FIELDS},
            eligible=source.eligible.copy(), consumed_band=source.split_band.copy(),
            first_fracture_time_myr=source.first_fracture_time_myr,
            tidal_peak_mpa=source.tidal_peak_mpa)
        self.events = []
        self._validate()

    @property
    def damage(self):
        return self.memory.damage

    def __deepcopy__(self, memo):
        result = self.__class__.__new__(self.__class__)
        memo[id(self)] = result
        result.model = self.model
        result.time_myr = self.time_myr
        result.memory = deepcopy(self.memory, memo)
        result.events = deepcopy(self.events, memo)
        return result

    def set_damage(self, damage):
        """Accept mature material resets without adding a second damage law."""
        array = np.asarray(damage, dtype=float)
        if (array.shape != (self.model.mesh.cell_count,) or not np.isfinite(array).all()
                or np.any((array < 0) | (array > 1))):
            raise ValueError("Young fracture damage must be finite and within [0, 1]")
        self.memory.damage = array.copy()
        self.memory.eligible = array >= self.model.parameters.rupture_damage
        self.memory.consumed_band[array < .5*self.model.parameters.rupture_damage] = False

    def _validate(self):
        n, m = self.model.mesh.cell_count, self.memory
        if not math.isfinite(self.time_myr) or self.time_myr < 0:
            raise ValueError("Young fracture clock must be finite and nonnegative")
        for name in FIELDS:
            array = np.asarray(getattr(m, name))
            if array.shape != (n,) or not np.isfinite(array).all():
                raise ValueError(f"Invalid young fracture field {name}")
        if (np.any((m.damage < 0) | (m.damage > 1))
                or np.any((m.water_access < 0) | (m.water_access > 1))
                or np.any(m.yield_ratio < 0) or np.any(m.strength_pa <= 0)):
            raise ValueError("Young fracture fields are outside physical ranges")
        for name in ("eligible", "consumed_band"):
            array = np.asarray(getattr(m, name))
            if array.shape != (n,) or array.dtype != bool:
                raise ValueError(f"Invalid young fracture mask {name}")
        if not np.array_equal(m.eligible, m.damage >= self.model.parameters.rupture_damage):
            raise ValueError("Young fracture eligibility must follow actual damage")
        if (not math.isfinite(m.tidal_peak_mpa) or m.tidal_peak_mpa < 0
                or (m.first_fracture_time_myr is not None and not (
                    math.isfinite(m.first_fracture_time_myr)
                    and 0 <= m.first_fracture_time_myr <= self.time_myr))):
            raise ValueError("Invalid young fracture diagnostic metadata")
        if not isinstance(self.events, list) or any(
                not isinstance(event, dict) or event.get("kind") != "split"
                or not isinstance(event.get("time_myr"), (int, float))
                or not 0 <= event["time_myr"] <= self.time_myr for event in self.events):
            raise ValueError("Invalid young fracture event history")

    def advance(self, before, samples):
        """Integrate accepted thermal samples atomically using the starter law."""
        self._validate()
        if not math.isclose(before.time_myr, self.time_myr, rel_tol=0., abs_tol=1e-10):
            raise ValueError("Young fracture and thermal clocks disagree")
        proposed = deepcopy(self.memory)
        time = self.time_myr
        previous = before
        for after in samples:
            if not math.isfinite(after.time_myr) or after.time_myr <= time:
                raise ValueError("Young fracture samples must advance monotonically")
            self.model._material_sample(proposed, previous, after)
            # A persistent weak band is already an existing boundary. Requiring
            # healing before reuse prevents relabeling its broad raster strip.
            proposed.consumed_band[proposed.damage < .5*self.model.parameters.rupture_damage] = False
            time = float(after.time_myr)
            previous = after
        old_memory, old_time = self.memory, self.time_myr
        self.memory, self.time_myr = proposed, time
        try:
            self._validate()
        except Exception:
            self.memory, self.time_myr = old_memory, old_time
            raise

    def transport(self, material_source_index):
        """Use the mature transport's actual donor, resetting newly born crust."""
        self._validate()
        source = np.asarray(material_source_index)
        n = self.model.mesh.cell_count
        if (source.shape != (n,) or source.dtype.kind not in "iu"
                or np.any(source < -1) or np.any(source >= n)):
            raise ValueError("Young fracture donor map must contain cell IDs or -1")
        valid = source >= 0
        proposed = deepcopy(self.memory)
        for name in FIELDS:
            target = (self.model.shell.tensile_strength_pa*self.model.strength_factor.copy()
                      if name == "strength_pa" else np.zeros(n))
            target[valid] = getattr(self.memory, name)[source[valid]]
            setattr(proposed, name, target)
        proposed.consumed_band = np.zeros(n, dtype=bool)
        proposed.consumed_band[valid] = self.memory.consumed_band[source[valid]]
        proposed.eligible = proposed.damage >= self.model.parameters.rupture_damage
        self.memory = proposed
        self._validate()

    def attempt(self, system, time_myr=None):
        """Return at most one real cut; caller applies manager/state bookkeeping.

        Independent plates retain their motion; daughters inherit the parent's
        rotation exactly. The next normal dynamics solve provides their motion.
        Chemical material, crust ages and rift extension are never manufactured.
        """
        self._validate()
        if time_myr is not None and not math.isclose(
                float(time_myr), self.time_myr, rel_tol=0., abs_tol=1e-10):
            raise ValueError("Young fracture event must use the accepted clock")
        # Transport moves ownership while Plate.seed_cell is only a raster
        # representative. Refresh that metadata without changing the cut or
        # either independent plate's motion.
        system = canonicalize_plate_seeds(self.model.mesh, system)
        p = self.model.parameters
        # Previously ruptured cells can anchor a new cross-plate cut at the
        # existing boundary. Excluding their entire broad raster band would
        # leave an uncut rim and block otherwise valid subsequent fractures.
        # Each candidate must nevertheless contain a connected, fresh loaded
        # stretch satisfying the existing physical span criterion; a tiny new
        # patch cannot replay an old band as another topology event.
        available = np.zeros(self.model.mesh.cell_count, dtype=bool)
        for parent in range(len(system.plates)):
            cells = np.flatnonzero((system.cell_plate == parent) & self.memory.eligible)
            for component in connected_components(cells, self.model.mesh.neighbors):
                component = np.asarray(component, dtype=np.int32)
                fresh = component[~self.memory.consumed_band[component]]
                fresh_parts = connected_components(fresh, self.model.mesh.neighbors)
                if any(_component_span_km(self.model.mesh, np.asarray(part),
                        self.model.thermal.radius_km) >= p.min_band_span_km for part in fresh_parts):
                    available[component] = True
        cut = select_starter_cut(self.model.mesh, system, available, self.damage,
            self.model.thermal.radius_km, p.min_child_area_km2, p.min_band_span_km)
        if cut is None:
            return None, None
        result, event = split_starter_band(self.model.mesh, system, cut,
            self.model.thermal.radius_km, p.min_child_area_km2, p.min_band_span_km,
            time_myr=self.time_myr)
        if result is None:
            # Mature manager may repair connectedness, but this path only emits
            # immediately usable, connected daughter domains.
            return None, None
        event.detail = "continued young-shell loading; " + event.detail
        self.memory.consumed_band[cut] = True
        self.events.append(asdict(event))
        return result, event

    def diagnose(self):
        self._validate()
        m, areas = self.memory, self.model.areas
        return {"time_myr": self.time_myr, "continued_split_count": len(self.events),
            "max_damage": float(m.damage.max()), "max_yield_ratio": float(m.yield_ratio.max()),
            "damaged_area_fraction": float(areas@m.eligible/areas.sum()),
            "available_rupture_area_fraction": float(areas@(m.eligible & ~m.consumed_band)/areas.sum()),
            "consumed_band_area_fraction": float(areas@m.consumed_band/areas.sum()),
            "tidal_peak_mpa": m.tidal_peak_mpa}

    def save(self, path):
        self._validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {"format": FORMAT, "fingerprint": self.model.fingerprint,
            "time_myr": self.time_myr, "events": self.events,
            "first_fracture_time_myr": self.memory.first_fracture_time_myr,
            "tidal_peak_mpa": self.memory.tidal_peak_mpa}
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
                temporary = Path(stream.name)
                np.savez_compressed(stream,
                    metadata=np.array(json.dumps(metadata, allow_nan=False)),
                    **{name: getattr(self.memory, name)
                       for name in (*FIELDS, "eligible", "consumed_band")})
            os.replace(temporary, path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    @classmethod
    def load(cls, model, path):
        with np.load(path, allow_pickle=False) as saved:
            metadata = json.loads(str(saved["metadata"]))
            if metadata.get("format") != FORMAT or metadata.get("fingerprint") != model.fingerprint:
                raise ValueError("Young fracture checkpoint belongs to another model/configuration")
            result = cls.__new__(cls)
            result.model, result.time_myr = model, metadata["time_myr"]
            result.events = metadata["events"]
            result.memory = YoungFractureMemory(
                **{name: saved[name].copy() for name in (*FIELDS, "eligible", "consumed_band")},
                first_fracture_time_myr=metadata["first_fracture_time_myr"],
                tidal_peak_mpa=metadata["tidal_peak_mpa"])
        result._validate()
        return result


__all__ = ["YoungShellFracture", "YoungFractureMemory"]
