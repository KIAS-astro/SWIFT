"""EAGLE stochastic thermal feedback injects the CCSN energy budget.

Each heating event raises one gas particle of mass m_g by Delta T, i.e. by
    dE_heat = m_g k Delta T / ((gamma - 1) mu_ion m_p)
(Dalla Vecchia & Schaye 2012, eq. 2; mu_ion = 4/(8 - 5(1 - X)) in SWIFT).
The gas tracer HeatedBySNIIFeedback counts the events each particle
received, so two checks follow:
  1. energy gained by the gas = (number of events) x dE_heat. The scheme
     satisfies this exactly, but the tracer is incremented once per particle
     per step even when several events land on that particle in the same
     step (feedback_iact.h: u += delta_u * N, tracer hook called once), so
     the counted events are a lower bound. Measured on 2026-10-20: the ratio
     is 1.037 after 10 Myr and 1.022 after 50 Myr, falling as the event rate
     per star falls; particles near a star were hit up to 13 times. The
     tolerance, 5 per cent, covers this undercount.
  2. number of events = E_SN / dE_heat within 4 Poisson sigma.
Delta T = 10^7 K gives about 7300 events from 128 stars, a 1.2 per cent
Poisson scatter.
"""

import numpy as np

from swiftreg import units
from swiftreg.checks import Check
from swiftreg.snapshot import snapshots

from . import _common as c

DESCRIPTION = "EAGLE thermal SN feedback: injected energy equals events x dE_heat and the CCSN budget"
TIER = "quick"
BUILDS = {"thermal": ["--with-subgrid=EAGLE", "--with-ext-potential=constant",
                      "--with-number-of-SNII-rays=8"]}

DELTA_T = 1e7


def run(ctx):
    p, mg, ms = c.setup(ctx, {"SNII_delta_T_K": DELTA_T, "SNII_feedback_model": "Isotropic"})
    ctx.info["m_gas"], ctx.info["m_star"] = mg, ms
    ctx.swift(p, c.FLAGS, timeout=3600)


def analyze(ctx):
    snaps = snapshots(ctx.run_dir)
    first, last = snaps[0], snaps[-1]
    fb = ctx.info["feedback"]
    mg = ctx.info["m_gas"]

    dE_heat = mg * units.BOLTZMANN_K * DELTA_T / (
        (units.GAMMA - 1.0) * units.MU_IONISED * units.PROTON_MASS)
    n_events = np.sum(last.read(0, "HeatedBySNIIFeedback", cgs=False))
    gained = c.total_energy(last) - c.total_energy(first)
    budget = c.expected_sn_energy(fb, ctx.info["m_star"])
    n_expected = budget / dE_heat
    ctx.info.update(n_events=n_events, n_expected=n_expected,
                    gained_erg=gained, budget_erg=budget)

    return [
        Check("energy gained / (events x dE_heat)", gained / (n_events * dE_heat), 1.0,
              kind="rel", tol=0.05,
              source="DVS12 eq. 2 per event; the tracer undercounts events merged "
                     "on one particle in one step (2-4 per cent here)"),
        Check("number of heating events", n_events, n_expected, kind="sigma", tol=4.0,
              sigma=np.sqrt(n_expected),
              source="E_SN / dE_heat, E_SN from the Chabrier IMF over 8-100 Msun"),
        Check("energy gained / CCSN budget", gained / budget, 1.0, kind="sigma", tol=4.0,
              sigma=1.0 / np.sqrt(n_expected),
              source="f_E = 1; Poisson scatter of the event number"),
    ]
