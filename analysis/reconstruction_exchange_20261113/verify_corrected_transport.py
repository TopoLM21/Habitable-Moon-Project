"""Repeat only affected inventory cases after rigid-coast source correction."""
from verify_runtime import run_case,compare,OUT

for name,sub,dt,end,resume in [
    ('corrected_g2_dt4',2,4.,100.,None),('corrected_g2_dt2',2,2.,100.,None),
    ('corrected_g2_dt1',2,1.,100.,None),('corrected_g1_dt2',1,2.,100.,None),
    ('corrected_g3_dt2',3,2.,100.,None),('corrected_first48',2,4.,48.,None),
    ('corrected_restart100',2,4.,100.,'corrected_first48'),
]:run_case(name,sub,dt,end,resume)
compare(OUT/'corrected_g2_dt4/restart',OUT/'corrected_restart100/restart','corrected_restart_comparison')
