"""Opt-in CPU execution policy; no physical parameters or checkpoint state.

The normal v0.31 entry point never enables this policy. A numerical process
owns one context, one bounded geometry cache and one persistent thread pool.
The spherical mesh must remain fixed while the context is active (as it does
in v0.31). Workers read old state and return private results; only the caller
commits those results, in plate-ID order.
"""
from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Callable, Iterable, TypeVar

import numpy as np
from scipy.spatial import cKDTree

from .mesh import SphereMesh

T = TypeVar("T")
R = TypeVar("R")
_active: CpuExecution | None = None


@dataclass
class MeshGeometry:
    mesh: SphereMesh  # Retain identity: IDs cannot be reused while cached.
    tree: cKDTree
    spacing: float | None = None
    neighbors: np.ndarray | None = None
    rasters: OrderedDict = field(default_factory=OrderedDict)


class CpuExecution(AbstractContextManager):
    def __init__(self, workers: int = 1, *, cell_kernels: bool = False) -> None:
        if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 32:
            raise ValueError("CPU workers must be an integer between 1 and 32")
        self.workers = workers
        self.cell_kernels = bool(cell_kernels)
        self.pool: ThreadPoolExecutor | None = None
        self._meshes: OrderedDict[int, MeshGeometry] = OrderedDict()

    def __enter__(self) -> "CpuExecution":
        global _active
        if _active is not None:
            raise RuntimeError("A CPU execution context is already active")
        if self.workers > 1:
            self.pool = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="moon-plate")
        _active = self
        return self

    def __exit__(self, *exc) -> None:
        global _active
        try:
            if self.pool is not None:
                self.pool.shutdown(wait=True, cancel_futures=True)
        finally:
            self.pool = None
            self._meshes.clear()
            _active = None

    def geometry(self, mesh: SphereMesh) -> MeshGeometry:
        """Called on the coordinator thread before any worker is submitted."""
        key = id(mesh)
        if key not in self._meshes:
            self._meshes[key] = MeshGeometry(mesh, cKDTree(mesh.centroids, copy_data=True))
            if len(self._meshes) > 2:
                self._meshes.popitem(last=False)
        self._meshes.move_to_end(key)
        return self._meshes[key]

    def ordered_map(self, function: Callable[[T], R], items: Iterable[T]) -> list[R]:
        # Materialise the entire result before committing anything. A failed
        # worker must not leave a partly updated transport state.
        if self.pool is None:
            return list(map(function, items))
        return list(self.pool.map(function, items))


def current_execution() -> CpuExecution | None:
    return _active


def query_workers() -> int:
    # cKDTree spawns threads for each query. Do not nest all-core queries
    # inside the outer persistent plate pool, including on small arrays.
    return 1 if _active is not None else -1
