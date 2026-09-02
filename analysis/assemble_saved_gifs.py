"""Assemble an intermediate animation from saved PNGs without running physics."""
from __future__ import annotations

import argparse
from pathlib import Path
import re

from PIL import Image

FRAME_TIME = re.compile(r"([0-9]+(?:\.[0-9]+)?)_Myr\.png$")
FRAME_SETS = {
    "surface": ("hydrosphere_frames", "surface_history"),
    "plates": ("plate_frames", "plate_history"),
    "hotspots": ("hotspot_track_frames", "hotspot_history"),
}


def assemble(source: Path, output: Path, through_myr: float, kinds: list[str], duration_ms: int = 350) -> list[Path]:
    if not source.is_dir():
        raise ValueError(f"Run folder does not exist: {source}")
    if through_myr < 0 or duration_ms <= 0:
        raise ValueError("Time must be nonnegative and frame duration positive")
    if output.resolve().is_relative_to(source.resolve()):
        raise ValueError("GIF output must be outside the preserved source run")
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for kind in kinds:
        directory, stem = FRAME_SETS[kind]
        frames = []
        for path in (source / directory).glob("*.png"):
            match = FRAME_TIME.search(path.name)
            if match and float(match.group(1)) <= through_myr:
                frames.append((float(match.group(1)), path))
        frames.sort(key=lambda item: (item[0], item[1].name))
        if len(frames) < 2:
            print(f"SKIP {kind}: fewer than two saved frames", flush=True)
            continue
        images = []
        try:
            for _, path in frames:
                with Image.open(path) as image:
                    images.append(image.convert("P", palette=Image.Palette.ADAPTIVE))
            target = output / f"{stem}_through_{through_myr:g}_Myr.gif"
            images[0].save(target, save_all=True, append_images=images[1:], duration=duration_ms, loop=0, optimize=False)
            with Image.open(target) as animation:
                animation.seek(animation.n_frames - 1)
                animation.load()
                print(f"SAVED {target}: {animation.n_frames} frames, t={frames[0][0]:g}..{frames[-1][0]:g} Myr", flush=True)
            results.append(target)
        finally:
            for image in images:
                image.close()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New output folder, never the original run")
    parser.add_argument("--through-myr", type=float, required=True)
    parser.add_argument("--kinds", nargs="+", choices=FRAME_SETS, default=["surface", "plates"])
    args = parser.parse_args()
    assemble(args.source_run, args.output, args.through_myr, args.kinds)


if __name__ == "__main__":
    main()
