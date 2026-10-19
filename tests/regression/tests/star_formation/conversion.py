"""Stochastic gas-to-star conversion reproduces the integrated SFR.

A gas particle converts in a step with probability p = SFR dt / m_g, so the
number of stars formed between two snapshots has expectation
sum_i SFR_i dt / m_g and, the conversions being independent, a variance of
about the same size. The run covers about 0.2 depletion times of the box
of sfr_law (n_H = 1 cm^-3, T = 10^3 K, no cooling), so about 18 per cent of
the 32768 particles convert.

Two checks:
  1. stars formed against the SFR integrated over the snapshots
     (trapezoid rule), within 4 standard deviations;
  2. the gas fraction left against the closed-box solution. Without cooling
     u is constant, so P is proportional to the gas density, and the rate
     per unit gas mass scales as rho^alpha with alpha = (n - 1)/2. Then
     dx/dt = -x^(1+alpha)/tau_0 for the gas fraction x, which integrates to
     x(t) = (1 + alpha t / tau_0)^(-1/alpha).
"""

import numpy as np

from swiftreg import units
from swiftreg.checks import Check
from swiftreg.snapshot import snapshots

from . import _common as c

DESCRIPTION = "stochastic conversion matches the integrated SFR and the closed-box depletion"
TIER = "quick"
BUILDS = {"eagle": c.EAGLE_BUILD}

NH = 1.0
T = 1e3
T_END = 0.3  # internal units, 293 Myr, about 0.19 depletion times
N_SNAP = 10


def run(ctx):
    c.write_box(ctx, NH, T)
    p = c.sf_params(ctx, T_END, T_END / N_SNAP)
    ctx.info["params"] = p["EAGLEStarFormation"]
    ctx.swift(p, c.SF_FLAGS, timeout=1800)


def analyze(ctx):
    snaps = snapshots(ctx.run_dir)
    n0 = c.N ** 3
    t = np.array([s.time_cgs for s in snaps])
    nstar = np.array([s.npart[4] for s in snaps], dtype=float)
    # Total conversion rate, sum of SFR / m over gas particles, at each output.
    # The output at t = 0 precedes the first SFR computation, so its rate is
    # taken from the first later output.
    rate = np.array([
        np.sum(np.clip(s.read(0, "StarFormationRates"), 0, None) / s.read(0, "Masses"))
        for s in snaps
    ])
    rate[0] = rate[1]
    expected_n = np.concatenate([[0.0], np.cumsum(0.5 * (rate[1:] + rate[:-1]) * np.diff(t))])

    sigma_n = np.sqrt(expected_n[-1])
    tau0 = n0 / rate[1]
    alpha = (ctx.info["params"]["KS_exponent"] - 1.0) / 2.0
    x_expected = (1.0 + alpha * t[-1] / tau0) ** (-1.0 / alpha)
    ctx.info["depletion_time_Gyr"] = tau0 / units.GYR
    ctx.info["stars_formed"] = nstar[-1]

    return [
        Check("stars formed", nstar[-1], expected_n[-1], kind="sigma", tol=4.0,
              sigma=sigma_n,
              source="sum over snapshots of sum_i SFR_i dt / m_g; Poisson sigma"),
        Check("gas fraction left", 1.0 - nstar[-1] / n0, x_expected, kind="sigma",
              tol=4.0, sigma=sigma_n / n0,
              source="closed box, P proportional to rho: x = (1 + alpha t/tau0)^(-1/alpha), "
                     "alpha = (n-1)/2"),
        Check("stars formed, first half vs second half ratio",
              nstar[N_SNAP // 2], expected_n[N_SNAP // 2], kind="sigma", tol=4.0,
              sigma=np.sqrt(expected_n[N_SNAP // 2]),
              source="integrated SFR to mid-run"),
    ]
