"""The Chevalier & Clegg (1985) wind, written independently of the C code
(src/wind_recoupling_physics.h), for the expected values of the wind tests.

Mass and energy injected uniformly inside radius R at rates Mdot and
Edot = Mdot v_inf^2 / 2, gamma = 5/3, adiabatic, no gravity. With x = r / R
the Mach number solves (CC85)
  x < 1: [(3g + M^-2)/(1 + 3g)]^(-(3g+1)/(5g+1)) [(g-1+2M^-2)/(1+g)]^((g+1)/(2(5g+1))) = x
  x > 1: M^(2/(g-1)) [(g-1+2M^-2)/(1+g)]^((g+1)/(2(g-1))) = x^2
and P = Mdot v_inf / (4 pi R^2) g(x) w(M) / (gamma M^2), with g = x inside and
x^-2 outside, w = M / (M^2 + 2/(gamma-1))^1/2 (wind-design.md Sec. 9.2).
"""

import numpy as np
from scipy.optimize import brentq

G = 5.0 / 3.0


def mach(x):
    if x == 1.0:
        return 1.0
    if x < 1.0:
        def f(lm):
            m2 = np.exp(-2 * lm)
            return ((3 * G + m2) / (1 + 3 * G)) ** (-(3 * G + 1) / (5 * G + 1)) * \
                ((G - 1 + 2 * m2) / (1 + G)) ** ((G + 1) / (2 * (5 * G + 1))) - x
        return float(np.exp(brentq(f, -40.0, 0.0, xtol=1e-14)))

    def f(lm):
        return (2 / (G - 1)) * lm + ((G + 1) / (2 * (G - 1))) * \
            np.log((G - 1 + 2 * np.exp(-2 * lm)) / (1 + G)) - 2 * np.log(x)
    return float(np.exp(brentq(f, 0.0, 40.0, xtol=1e-14)))


def pressure_dimensionless(x):
    m = mach(x)
    w = m / np.sqrt(m * m + 2 / (G - 1))
    gx = x if x < 1 else 1 / x ** 2
    return gx * w / (G * m * m)


def pressure(r, R, mdot, v_inf):
    """P(r) in the units of mdot v_inf / R^2."""
    return mdot * v_inf / (4 * np.pi * R * R) * pressure_dimensionless(r / R)


def balance_radius(R, mdot, v_inf, p_amb, r_lo=1e-3, r_hi=1e4):
    """Radius where P(r) falls to p_amb (P decreases monotonically outside a
    small inner region; bracketed between r_lo and r_hi)."""
    return brentq(lambda r: np.log(pressure(r, R, mdot, v_inf) / p_amb), r_lo, r_hi,
                  xtol=1e-12 * R)
