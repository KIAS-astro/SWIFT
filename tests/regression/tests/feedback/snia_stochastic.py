"""SNIa energy through stochastic heating in EAGLE-thermal-kinetic (test T7).

COLIBRE injects SNIa energy with the stochastic thermal scheme of the SNII
channel (f_E = 1, f_kin = 0, the same Delta T; Schaye et al. 2026, eq. 21 and
Sec. 3.7); EAGLE spreads it over all neighbours every step. Both must inject
    E = 1e51 erg x m_* x nu [exp(-(t1 - t_d)/tau) - exp(-(t2 - t_d)/tau)]
over a run from star age t1 to t2, for SWIFT's exponential delay-time
distribution with nu = SNIa_DTD_exp_norm_p_Msun, tau, t_d.

The feedback box of feedback/_common.py with 128 stars of 20 m_g born at t = 0;
the run starts at t = 97.8 Myr (time_begin = 0.1) and lasts 51 Myr, so only SNIa
release energy (CCSN end at about 40 Myr; SNII feedback off). SNIa enrichment is
on, because SWIFT computes the SNIa energy in the enrichment step; AGB and SNII
enrichment are off and AGB ejecta carry no kinetic energy.

Status (2026-10-20): the continuous run injects 2.1 per cent less than the DTD
integral (3.3 per cent before the age cut below was raised); the cause is not yet
found, and the check is left failing until it is. The stochastic run agrees with
both within its 10 per cent Poisson scatter.

Runs and checks:
  continuous: energy gained / E within 1e-2 (SPH energy conservation);
  stochastic, Delta T = 10^6.5 K: energy gained / E within 4 Poisson sigma of
  the expected number of heating events E / dE_heat.
"""

import math
import os

import numpy as np

from swiftreg import units
from swiftreg.checks import Check
from swiftreg.snapshot import snapshots

from . import _common as c
from .thermal_kinetic_wiring import COMMON, _Sub

DESCRIPTION = "EAGLE-thermal-kinetic: stochastic and continuous SNIa inject the DTD energy"
TIER = "quick"
BUILDS = {"thermal_kinetic": COMMON + ["--with-feedback=EAGLE-thermal-kinetic"]}

T_BEGIN = 0.1  # internal units, 97.8 Myr
DURATION = 0.0522  # 51 Myr
DELTA_T = 10 ** 6.5
MODES = ("Continuous", "Stochastic")


def run(ctx):
    for mode in MODES:
        os.makedirs(ctx.path(mode))
        p, mg, ms = c.setup(_Sub(ctx, ctx.path(mode)), {
            "use_SNII_feedback": 0, "use_SNIa_feedback": 1,
            "use_SNIa_enrichment": 1,
            "SNIa_feedback_model": mode, "SNII_f_kinetic": 0.0,
            "SNII_delta_T_K": DELTA_T,
            # past this age EAGLE releases stellar evolution only every
            # stellar_evolution_sampling_rate steps, so the last interval would
            # still be held back when the run ends
            "stellar_evolution_age_cut_Gyr": 1.0}, t_end=T_BEGIN + DURATION)
        p["TimeIntegration"]["time_begin"] = T_BEGIN
        p["Snapshots"]["time_first"] = T_BEGIN
        p["Statistics"]["time_first"] = T_BEGIN
        p["Snapshots"]["delta_time"] = DURATION / 5
        ctx.info.update(m_gas=mg, m_star=ms)
        ctx.swift(p, c.FLAGS, subdir=mode, name=mode, timeout=3600)


def expected_energy(fb, m_star_cgs, age1_yr, age2_yr):
    nu = fb["SNIa_DTD_exp_norm_p_Msun"]
    tau = fb["SNIa_DTD_exp_timescale_Gyr"] * 1e9
    td = fb["SNIa_DTD_delay_Gyr"] * 1e9
    n_per_msun = nu * (math.exp(-(age1_yr - td) / tau) - math.exp(-(age2_yr - td) / tau))
    return fb["SNIa_energy_erg"] * c.N_STARS * m_star_cgs / units.SOLAR_MASS * n_per_msun


def analyze(ctx):
    fb = ctx.info["feedback"]
    dE_heat = ctx.info["m_gas"] * units.BOLTZMANN_K * DELTA_T / (
        (units.GAMMA - 1.0) * units.MU_IONISED * units.PROTON_MASS)
    checks = []
    for mode in MODES:
        snaps = snapshots(ctx.path(mode))
        age1, age2 = snaps[0].time_cgs / units.YEAR, snaps[-1].time_cgs / units.YEAR
        E = expected_energy(fb, ctx.info["m_star"], age1, age2)
        gained = c.total_energy(snaps[-1]) - c.total_energy(snaps[0])
        ctx.info[mode] = {"E_expected": E, "gained": gained, "ages_yr": [age1, age2]}
        if mode == "Continuous":
            checks.append(Check("continuous: energy gained / DTD energy", gained / E, 1.0,
                                kind="rel", tol=1e-2,
                                source="SWIFT exponential DTD integrated over the run"))
        else:
            n_exp = E / dE_heat
            checks.append(Check("stochastic: energy gained / DTD energy", gained / E, 1.0,
                                kind="sigma", tol=4.0, sigma=1 / np.sqrt(n_exp),
                                source=f"Poisson scatter of {n_exp:.0f} expected events"))
    return checks
