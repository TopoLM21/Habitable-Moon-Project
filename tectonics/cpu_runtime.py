"""Opt-in CPU execution policy; no physical parameters or checkpoint state.

The normal v0.31 entry point never enables this policy. A numerical process
owns one context, bounded geometry caches and optional persistent worker pools.
The spherical mesh must remain fixed while the context is active (as it does
in v0.31). Workers read old state and return private results; only the caller
commits those results, in plate-ID or target-cell order.
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
    def __init__(self, workers: int = 1, *, cell_kernels: bool = False,
                 reuse_initial_mesh: bool = True, numeric_kernels: bool = True,
                 single_source_cells: bool = True, cell_workers: int = 1,
                 arc_kernels: bool = True) -> None:
        if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 32:
            raise ValueError("CPU workers must be an integer between 1 and 32")
        if isinstance(cell_workers, bool) or cell_workers not in (1, 2, 4, 8) or not isinstance(cell_workers, int):
            raise ValueError("Cell workers must be one of 1, 2, 4, 8")
        self.workers = workers
        self.cell_kernels = bool(cell_kernels)
        self.reuse_initial_mesh = bool(reuse_initial_mesh)
        self.numeric_kernels = bool(numeric_kernels)
        self.single_source_cells = bool(single_source_cells)
        self.cell_workers = cell_workers
        self.arc_kernels = bool(arc_kernels)
        self.pool: ThreadPoolExecutor | None = None
        self.cell_pool: ThreadPoolExecutor | None = None
        self.cell_calls = self.cell_tasks = self.cells_prepared = 0
        self.cell_thread_ids: set[int] = set()
        self.arc_calls = self.arc_tasks = 0
        self._meshes: OrderedDict[int, MeshGeometry] = OrderedDict()
        self._initial_mesh: tuple[int, SphereMesh] | None = None

    def __enter__(self) -> "CpuExecution":
        global _active
        if _active is not None:
            raise RuntimeError("A CPU execution context is already active")
        try:
            if self.workers > 1:
                self.pool = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="moon-plate")
            if self.single_source_cells and self.cell_workers > 1:
                self.cell_pool = ThreadPoolExecutor(max_workers=self.cell_workers, thread_name_prefix="moon-cell")
        except BaseException:
            self.__exit__()
            raise
        _active = self
        return self

    def __exit__(self, *exc) -> None:
        global _active
        try:
            try:
                if self.pool is not None:
                    self.pool.shutdown(wait=True, cancel_futures=True)
            finally:
                if self.cell_pool is not None:
                    self.cell_pool.shutdown(wait=True, cancel_futures=True)
        finally:
            self.pool = None
            self.cell_pool = None
            self._meshes.clear()
            self._initial_mesh = None
            _active = None

    def initial_mesh(self, subdivisions: int, builder: Callable[[int], SphereMesh]) -> SphereMesh:
        """Share only fixed geometry during this process, never evolving state.

        The key covers every input of build_icosphere. Keep at most one mesh;
        changing resolution or leaving this context cannot reuse stale geometry.
        Called by the coordinator, before any numerical workers are submitted.
        The opt-out exists for paired performance diagnostics.
        """
        if not self.reuse_initial_mesh:
            return builder(subdivisions)
        if self._initial_mesh is None or self._initial_mesh[0] != subdivisions:
            mesh = builder(subdivisions)
            self._initial_mesh = (subdivisions, mesh)
        return self._initial_mesh[1]

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

    def ordered_cell_map(self, function: Callable[[T], R], items: Iterable[T]) -> list[R]:
        if self.cell_pool is None:
            return list(map(function, items))
        return list(self.cell_pool.map(function, items))

    def numerical_report(self) -> dict:
        return {"numeric_kernels": self.numeric_kernels, "single_source_cells": self.single_source_cells,
                "cell_workers": self.cell_workers, "cell_calls": self.cell_calls,
                "cell_tasks": self.cell_tasks, "cells_prepared": self.cells_prepared,
                "cell_thread_ids": sorted(self.cell_thread_ids), "arc_kernels": self.arc_kernels,
                "arc_calls": self.arc_calls, "arc_tasks": self.arc_tasks,
                "arc_query_workers": self.workers if self.arc_kernels else 1}


def current_execution() -> CpuExecution | None:
    return _active


def query_workers() -> int:
    # cKDTree spawns threads for each query. Do not nest all-core queries
    # inside the outer persistent plate pool, including on small arrays.
    return 1 if _active is not None else -1
