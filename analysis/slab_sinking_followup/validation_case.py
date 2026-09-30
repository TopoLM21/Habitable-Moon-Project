"""Traced mechanics-0.5 experiments using the historical isolation harness.

Fresh imports explicitly select young-mechanics-0.5. Resume and frozen probes
retain their saved version. In particular a frozen0.4 checkpoint is a0.4
baseline, never an implicitly migrated0.5 geological state.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis/slab_sinking_validation"))
from run_case import main


if __name__ == "__main__":
    main(default_mechanics_version="young-mechanics-0.5")
