"""Uniform root-box mass bound for the three analytic test fixtures only.

This bound is NOT installed in the integrator and does not discard boxes.
It motivates a later interval treatment; tensor components need their own
bounds. See README for the geometric derivation and its restricted scope.
"""
from fractions import Fraction as F


def fixture_root_box_mass_bound(radius,bits=40):
    radius=F(radius)
    if not 0<radius<=F(1,4) or type(bits) is not int or bits<1:
        raise ValueError('Bound only covers the documented centered test family')
    rho_max=25*radius/(2*(9-radius))
    max_fiber=32*radius
    # Sum of the total degrees of A, Delta, both sphere/cohort events, g,
    # both Gamma/cohort events, and the Gamma/sphere resultant.
    root_count_upper=17
    # width <= lo*2^-bits; a box meeting physical support has lo<=rho_max.
    # Sum of fan double areas is one, and the Duffy factor rho<=rho_max.
    return max_fiber*root_count_upper*rho_max**2/2**bits
