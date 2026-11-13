"""Preserve the pre-area mode exactly on its saved 20 Myr control."""
import importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('area_checks',Path(__file__).with_name('verify_runtime.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
if __name__=='__main__':
    m.h.BASE['regional_material'].pop('active_area')
    m.h.run_case('point_regression20',2,4.,20.,'initial_same_g2')
    m.h.compare(m.OUT/'point_regression20/restart',ROOT/'analysis/regional_inventory_20261113/saved_states/short_m3_g2_dt4','point_regression_comparison')
