"""Continue a coarse genesis candidate through the existing mature runner.

Run this entry point in its own process: the mature runner installs module
hooks. Duration is total elapsed time since the original starter partition,
including when resuming a previous continuation directory.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from execution_policy import PROCESS_PRIORITY_CHOICES, RENDER_WORKER_CHOICES


def run_continuation(source, output, duration_myr, step_myr, *, resume=False, mature_config=None,
                     **execution_options):
    """Lazy entry so the starter's isolated worker can continue without a child."""
    if not all(math.isfinite(value) and value > 0 for value in (duration_myr, step_myr)):
        raise ValueError("Continuation duration and step must be finite and positive")
    from tectonics.genesis_starter_continuation import run_starter_continuation
    return run_starter_continuation(Path(source), Path(output), duration_myr=float(duration_myr),
                                    step_myr=float(step_myr), resume=resume,
                                    mature_config=None if mature_config is None else Path(mature_config),
                                    **execution_options)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--starter-checkpoint", type=Path)
    source.add_argument("--resume", type=Path, help="Previous continuation output directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration-myr", type=float, default=10., help="Total elapsed time from the original starter partition, also on resume")
    parser.add_argument("--step-myr", type=float, default=1.)
    parser.add_argument("--mature-config", type=Path)
    parser.add_argument("--subdivisions", type=int, choices=range(2, 9), default=argparse.SUPPRESS,
                        help="Optionally refine the saved world onto a finer icosphere")
    parser.add_argument("--cpu-workers", type=int, choices=range(1, 33), default=argparse.SUPPRESS)
    parser.add_argument("--render-workers", type=int, choices=RENDER_WORKER_CHOICES, default=argparse.SUPPRESS)
    parser.add_argument("--process-priority", choices=PROCESS_PRIORITY_CHOICES, default=argparse.SUPPRESS)
    parser.add_argument("--cell-workers", type=int, choices=(1, 2, 4, 8), default=argparse.SUPPRESS)
    for option in ("cell-kernels", "numeric-kernels", "single-source-cells", "arc-kernels",
                   "assignment-columns", "assignment-optimized", "boundary-forces"):
        parser.add_argument(f"--{option}", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
    parser.add_argument("--frame-interval", dest="frame_interval_myr", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--surface-only-frames", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("--finalize", action="store_true", default=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    execution_options = {name: value for name, value in vars(args).items() if name not in
                         {"resume", "starter_checkpoint", "output", "duration_myr", "step_myr", "mature_config"}}
    try:
        run_continuation(args.resume or args.starter_checkpoint, args.output, args.duration_myr,
                         args.step_myr, resume=args.resume is not None, mature_config=args.mature_config,
                         **execution_options)
        print(f"GENESIS_STARTER_CONTINUATION_COMPLETE {args.output.resolve()}", flush=True)
        return 0
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as exc:
        print(f"Genesis starter continuation error: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
