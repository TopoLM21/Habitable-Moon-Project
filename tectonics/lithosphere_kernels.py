"""Conservative single-source cell preparation, with no shared worker writes.

Overlaps, gaps and all material redistribution stay in the original integrator.
Worker outputs are private. The coordinator commits only after every task has
succeeded, in ascending target order; common diagnostics are unaffected because
single-source targets cannot contribute collision or subduction losses.
"""
from __future__ import annotations

from functools import partial
from threading import get_ident

import numpy as np


def _prepare_single_source(targets, *, covered, source, areas, crust_type, age,
                           thickness, fraction, volume, dt_myr, copied_fields):
    from .lithosphere import CrustType

    plates = np.argmax(covered[:, targets], axis=0)
    sources = source[plates, targets]
    continental = crust_type[sources] == int(CrustType.CONTINENTAL)
    result = {name: values[sources].copy() for name, values in copied_fields.items()}
    result["plate"] = plates
    result["type"] = np.where(continental, int(CrustType.CONTINENTAL), int(CrustType.OCEANIC))
    result["age"] = age[sources] + dt_myr
    # np.sum(singleton) starts with +0.0; retain its signed-zero behaviour.
    result["fraction"] = np.add(0.0, fraction[sources] * areas[sources]) / areas[targets]
    result["volume"] = np.add(0.0, volume[sources])
    result["source"] = sources.copy()
    new_h = np.empty(len(targets), dtype=np.float64)
    oceanic = ~continental
    ocean_sources = sources[oceanic]
    new_h[oceanic] = thickness[ocean_sources] * areas[ocean_sources] / areas[targets[oceanic]]
    continent_sources = sources[continental]
    wf = np.maximum(fraction[continent_sources], 1e-12)
    new_h[continental] = volume[continent_sources] / np.maximum(areas[continent_sources] * wf, 1e-30)
    result["thickness"] = new_h
    return targets, result, get_ident()


def fill_single_source_cells(execution, targets, *, covered, source, areas, state,
                             fraction, volume, dt_myr, copied_fields, outputs):
    if not len(targets):
        return
    prepare = partial(_prepare_single_source, covered=covered, source=source, areas=areas,
                      crust_type=state.crust_type, age=state.crust_age_myr,
                      thickness=state.crust_thickness_km, fraction=fraction, volume=volume,
                      dt_myr=dt_myr, copied_fields=copied_fields)
    chunks = np.array_split(targets, min(execution.cell_workers, len(targets)))
    # No writes to outputs until all futures are complete. A worker failure
    # cannot leave a partially committed cell result.
    prepared = execution.ordered_cell_map(prepare, chunks)
    for cells, fields, thread_id in prepared:
        for name, values in fields.items():
            outputs[name][cells] = values
        execution.cell_thread_ids.add(thread_id)
    execution.cell_calls += 1
    execution.cell_tasks += len(prepared)
    execution.cells_prepared += len(targets)
