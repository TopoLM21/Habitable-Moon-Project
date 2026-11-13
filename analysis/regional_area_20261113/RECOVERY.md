# Preserved in-progress reconstruction, 2026-11-13

The previous environment returned with the same working copy and branch
`reconstruction/regional-20261113`, HEAD
`a6e78948f1980555c0ca4d9c04d92d72ca4e4020`. No simulation process was running.
The old defunct Python process was not an active computation.

The interrupted final patch did **not** apply: `tectonics/strip_intersection.py`
is absent, and neither `_coarea_field` nor `strip_rule` exists in the saved code.
No replay occurred before that read-only check.

Preserved work: `tectonics/regional_area.py`, its optional v131 runtime entry
through v123, `tests/test_regional_area.py`, the frozen-strip diagnostic script
and all existing logs/results in this directory. Nine targeted tests passed
before interruption. The initial seven failures in `initial_tests.log` were a
new int32 tree-key overflow, fixed before `area_tests.log`; they are unrelated
to the historical seven failures from the unavailable old project version.

The frozen-source results remain negative convergence evidence: at maximum
edge 62.5 km and 32-point quadrature, mature-area errors are approximately
−8.3% and −8.6% for the two orientations. Coarser cases are worse. Summing the
source exactly does not establish convergence of the thresholded material area.
No coupled long run, full new-mode energy closure or paid fracture is claimed.

This recovery commit saves the in-progress implementation and evidence before
further edits. The next numerical investigation is direct spherical
strip/triangle intersection integration; it has not yet been implemented or
verified in this saved version. Old controls, archived data and main remain
unchanged.
