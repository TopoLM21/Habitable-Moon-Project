"""Run the portable focused geometry suite from any working directory."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    "test_spherical_polygons.py", "test_geometry_roundoff.py",
    "test_fractional_surface.py", "test_fractional_surface_io.py",
    "test_fractional_transport.py", "test_fractional_event_transport.py",
    "test_fractional_event_thermal.py", "test_fractional_time_coupling.py",
]


if __name__ == "__main__":
    selected = [ROOT/"tests"/name for name in FILES]
    selected += sorted((ROOT/"tests").glob("test_geometric*.py"))
    selected.append(ROOT/"tests/test_cloud_handoff.py")
    raise SystemExit(subprocess.call(
        [sys.executable, "-m", "pytest", "-q", *map(str, selected), *sys.argv[1:]], cwd=ROOT))
