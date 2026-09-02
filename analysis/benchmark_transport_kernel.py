"""Compare only transport preparation, in one process, with exact input copies."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from copy import deepcopy
import json
from pathlib import Path
import statistics
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import yaml
from tectonics.checkpoint import load_checkpoint
from tectonics.cpu_runtime import CpuExecution
from tectonics.mesh import build_icosphere
from tectonics.topology import PlateTopologyManager, PlateTopologyParameters
from tectonics.transport import SubgridTransportParameters, build_transport_map


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    mesh = build_icosphere(int(config["mesh"]["subdivisions"]))
    checkpoint = load_checkpoint(args.resume, PlateTopologyManager(PlateTopologyParameters()))
    fields = SubgridTransportParameters.__dataclass_fields__
    parameters = SubgridTransportParameters(**{key: value for key, value in config.get("subgrid_transport", {}).items() if key in fields})
    reference = None
    rows = []
    for workers in (0, 1, 2, 4, 8):
        with CpuExecution(workers) if workers else nullcontext():
            timings = []
            for _ in range(4):
                memory = deepcopy(checkpoint.transport_state)
                began = perf_counter()
                result = build_transport_map(mesh, checkpoint.system, checkpoint.state, 4.0, memory, parameters)
                timings.append(perf_counter() - began)
                if reference is None:
                    reference = deepcopy(result)
                assert np.array_equal(result.covered, reference.covered)
                assert np.array_equal(result.source, reference.source)
                assert np.array_equal(result.state.residual_quaternions, reference.state.residual_quaternions)
                assert result.diagnostics == reference.diagnostics
            row = {"mode": "baseline" if workers == 0 else f"cpu{workers}", "cold_seconds": timings[0],
                   "warm_median_seconds": statistics.median(timings[1:]), "all_seconds": timings, "exact": True}
            rows.append(row)
            print(json.dumps(row), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)


if __name__ == "__main__":
    main()
