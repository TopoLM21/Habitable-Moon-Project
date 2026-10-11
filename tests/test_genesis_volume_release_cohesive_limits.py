"""Explicit refusal when the inherited relative potential loses a tiny change."""
import numpy as np
import pytest

from tectonics.genesis_volume_creation_law import ProspectiveCreationLaw
from tectonics.genesis_volume_release_cohesive import (
    VolumeReleaseCohesiveRefusal, evaluate_volume_release_cohesive,
    prepare_volume_release_cohesive)


@pytest.mark.parametrize("tangent_gap_m", [1e-10, 1e-12])
def test_unresolved_quadratic_tangent_work_is_refused_without_funding_floor(tangent_gap_m):
    law = ProspectiveCreationLaw(1e6, 4e6, 1e9, 2e9)
    chart = prepare_volume_release_cohesive(law, [[1e6, 0, 0]], [7.],
        np.eye(3)[None], np.eye(3))
    # The virgin path has positive quadratic work.  Computing its potential
    # through radius-reference_radius loses the tiny radial difference.  The
    # current scoped kernel must refuse rather than substitute scalar funding
    # or weaken the direct traction/path-work certificate.
    assert .5*7*law.cohesive_stiffness_pa_m*tangent_gap_m**2 > 0.
    assert np.hypot(.001, tangent_gap_m) < law.peak_separation_m
    with pytest.raises(VolumeReleaseCohesiveRefusal, match="exact path duality"):
        evaluate_volume_release_cohesive(chart, [0, 0, 0], [0, tangent_gap_m, 0],
                                        chart.seed.initial_maximum_separation_m)
