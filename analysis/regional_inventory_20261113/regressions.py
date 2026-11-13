"""Compare unchanged modes and the non-exhausting regional path exactly."""
import importlib.util,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('h',ROOT/'analysis/reconstruction_exchange_20261113/verify_runtime.py')
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
h.HERE=HERE;h.OUT=ROOT/'outputs_v131_regional'
h.BASE=h.yaml.safe_load((ROOT/'analysis/integral_arcs_20261113/control.yaml').read_text())
h.run_case('regression_integral100',2,4.,100.,'initial_same_g2')
h.compare(h.OUT/'regression_integral100/restart',ROOT/'outputs_v131_integral_arcs/final_g2_dt4/restart','previous_integral_mode_comparison')
h.run_case('regression_legacy20',2,4.,20.,mode='legacy')
h.compare(h.OUT/'regression_legacy20/restart',ROOT/'outputs_v131_reconstruction/legacy_original/restart','original_legacy_mode_comparison')
h.BASE=h.yaml.safe_load((HERE/'control.yaml').read_text())
h.run_case('regression_regional20',2,4.,20.,'initial_same_g2')
h.compare(h.OUT/'regression_regional20/restart',h.OUT/'short_m3_g2_dt4/restart','exhaustion_fix_nonempty_comparison')
