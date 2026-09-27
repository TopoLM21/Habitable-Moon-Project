"""Analytical energy of an ideal displacement-controlled double cantilever beam.

The two equal Euler--Bernoulli arms have height ``arm_height_m`` each and
width ``width_m``. ``opening_m`` is their total load-point separation, and
``reaction_n`` is the force on either arm. The crack tip is ideally clamped;
shear deformation, root rotation and a finite cohesive process zone are absent.
These formulae are a reference problem, not a planet-shell energy estimator,
a nucleation criterion, or a law for the speed of a crack.

The crack area convention is width times advance, counted once. The supplied
critical energy release rate already includes the cost of both crack faces.
It must not be charged again by a subsequent cohesive softening calculation.

References for the beam/compliance equations, not the numerical defaults:
https://solidmechanics.org/Text/Chapter9_4/Chapter9_4.php
https://ntrs.nasa.gov/api/citations/20180004403/downloads/20180004403.pdf
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import math
from numbers import Real


def _scalar(value, name, *, positive):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"DCB {name} must be a finite real scalar")
    try:
        value = float(value)
    except (OverflowError, ValueError):
        raise ValueError(f"DCB {name} must be a finite real scalar") from None
    if not math.isfinite(value) or (value <= 0 if positive else value < 0):
        sign = "positive" if positive else "nonnegative"
        raise ValueError(f"DCB {name} must be finite and {sign}")
    return value


def _result(name, calculation, *, positive=False):
    try:
        result = calculation()
    except (OverflowError, ZeroDivisionError, FloatingPointError):
        raise ValueError(f"DCB {name} is not representable at these scales") from None
    if not math.isfinite(result) or (result <= 0 if positive else result < 0):
        raise ValueError(f"DCB {name} is not representable at these scales")
    return result


@dataclass(frozen=True)
class DCBOracle:
    """Pure scalar SI energy backend for an existing Mode-I DCB crack."""

    young_pa: float = 70e9
    arm_height_m: float = .002
    width_m: float = .025
    fracture_energy_j_m2: float = 500.

    def __post_init__(self):
        for field in fields(self):
            value = _scalar(getattr(self, field.name), field.name, positive=True)
            object.__setattr__(self, field.name, value)

    def validate(self):
        for field in fields(self):
            _scalar(getattr(self, field.name), field.name, positive=True)

    def compliance_m_n(self, crack_length_m):
        """Total opening per force on one arm: ``8 a³ / (E b h³)``."""
        a = _scalar(crack_length_m, "crack_length_m", positive=True)
        return _result("compliance", lambda: (
            8.*a**3/(float(self.young_pa)*float(self.width_m)*float(self.arm_height_m)**3)
        ), positive=True)

    def reaction_n(self, crack_length_m, opening_m):
        """Opening reaction on either arm, ``P = delta / C``."""
        opening = _scalar(opening_m, "opening_m", positive=False)
        compliance = self.compliance_m_n(crack_length_m)
        return _result("reaction", lambda: opening/compliance)

    def stored_energy_j(self, crack_length_m, opening_m):
        """Total elastic energy of both arms at the prescribed opening."""
        opening = _scalar(opening_m, "opening_m", positive=False)
        reaction = self.reaction_n(crack_length_m, opening)
        return _result("stored energy", lambda: .5*opening*reaction)

    def release_rate_j_m2(self, crack_length_m, opening_m):
        """``-dU/(b da)`` at fixed opening, per projected crack area."""
        a = _scalar(crack_length_m, "crack_length_m", positive=True)
        energy = self.stored_energy_j(a, opening_m)
        return _result("energy release rate", lambda: 3.*energy/(float(self.width_m)*a))

    def equilibrium_length_m(self, opening_m):
        """Unclipped length where ``G = Gc``; zero for zero opening.

        This method applies neither a seed length nor irreversible history.
        Its result is an analytical equilibrium, not a newly born crack.
        """
        opening = _scalar(opening_m, "opening_m", positive=False)
        if opening == 0.:
            return 0.
        return _result("equilibrium crack length", lambda: (
            3.*float(self.young_pa)*float(self.arm_height_m)**3*opening**2
            /(16.*float(self.fracture_energy_j_m2))
        )**.25, positive=True)

    def fracture_cost_j(self, old_length_m, new_length_m):
        """Cost ``Gc b (a_new-a_old)`` of nonnegative crack advance."""
        old = _scalar(old_length_m, "old_length_m", positive=True)
        new = _scalar(new_length_m, "new_length_m", positive=True)
        if new < old:
            raise ValueError("DCB fracture cost requires nonnegative growth")
        return _result("fracture cost", lambda: (
            float(self.fracture_energy_j_m2)*float(self.width_m)*(new-old)
        ))
