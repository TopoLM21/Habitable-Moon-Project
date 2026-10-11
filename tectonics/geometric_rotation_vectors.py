"""Compact immutable angular velocities addressed by actual material owner."""
from collections.abc import Mapping
from numbers import Integral, Real
from types import MappingProxyType
import math

import numpy as np


def compact_rotation_vectors(values, owners):
    """Validate an explicitly supplied owner map without dense ID allocation."""
    if not isinstance(values, Mapping):
        raise ValueError("Compact rotations require an explicit owner mapping")
    expected = tuple(sorted(set(owners)))
    for owner in values:
        if isinstance(owner, (bool, np.bool_)) or not isinstance(owner, Integral) or owner < 0:
            raise ValueError("Angular velocity owners must be nonnegative integers")
    if tuple(sorted(values)) != expected:
        raise ValueError("Compact rotations must cover every actual owner exactly once")
    result = {}
    for owner in expected:
        raw = values[owner]
        if (not isinstance(raw, (tuple, list, np.ndarray))
                or (isinstance(raw, np.ndarray) and raw.shape != (3,))
                or len(raw) != 3):
            raise ValueError("Each angular velocity must have three components")
        if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Real)
               or not math.isfinite(float(v)) for v in raw):
            raise ValueError("Angular velocity components must be finite real numbers")
        # Immutable bytes own the data: a caller cannot re-enable WRITEABLE
        # and alter the validated map after it has been prepared for use.
        vector = np.frombuffer(np.asarray(raw, dtype=float).tobytes(), dtype=float)
        result[int(owner)] = vector
    return MappingProxyType(result)


def rotation_vectors_payload(values):
    """Keep the original dense transaction encoding; tag compact owner maps."""
    if isinstance(values, Mapping):
        return dict(format="compact-material-owner-angular-velocities-1",
            owner_vectors=[[int(owner), list(map(float, values[owner]))]
                           for owner in sorted(values)])
    return np.asarray(values).tolist()
