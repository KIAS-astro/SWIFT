"""SNIa energy through stochastic heating in EAGLE-thermal-kinetic (test T7).

COLIBRE injects SNIa energy with the stochastic thermal scheme of the SNII
channel (f_E = 1, f_kin = 0, the same Delta T; Schaye et al. 2026, eq. 21 and
Sec. 3.7); EAGLE spreads it over all neighbours every step. Both must inject
the energy of the exponential delay-time distribution, with
nu = SNIa_DTD_exp_norm_p_Msun, tau and t_d (eagle_feedback_number_of_SNIa in
src/feedback/EAGLE/enrichment.h), over a run from star age t1 to t2 > t_d.
EAGLEFeedback:SNIa_DTD_convention selects the form:
    EAGLE:   E = 1e51 erg x m_* x nu [exp(-t1/tau) - exp(-t2/tau)],
    COLIBRE: E = 1e51 erg x m_* x nu [exp(-(t1-t_d)/tau) - exp(-(t2-t_d)/tau)]
(Schaye et al. 2026, eq. 13). In EAGLE's a star makes nu exp(-t_d/tau) SNIa
per Msun in total, in COLIBRE's nu; the two differ by exp(-t_d/tau) = 0.980
for t_d = 40 Myr and tau = 2 Gyr.

The feedback box of feedback/_common.py with 128 stars of 20 m_g born at t = 0;
the run starts at t = 97.8 Myr (time_begin = 0.1) and lasts 51 Myr, so only SNIa
release energy (CCSN end at about 40 Myr; SNII feedback off). SNIa enrichment is
on, because SWIFT computes the SNIa energy in the enrichment step; AGB and SNII
enrichment are off and AGB ejecta carry no kinetic energy.

Status (2026-10-20): against the EAGLE form the continuous run measures 0.9985.
The continuous run draws no random numbers; the 0.15 per cent residual is the
same for three sets of star positions (ics SEED 2026, 7, 99).

Runs and checks (continuous in both conventions, stochastic in COLIBRE's):
  continuous: energy gained / E within 1e-2. All time-bins end at t2, so no
  star holds back energy at the last output; at intermediate outputs a star
  owes up to one 0.8 Myr step and the ratio dips to 0.94. The tolerance
  bounds SPH energy conservation and float rounding of the injected energy;
  the measured residual is 1.5e-3, and the 2 per cent difference between the
  conventions lies outside it;
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
# (SNIa_feedback_model, SNIa_DTD_convention)
RUNS = (("Continuous", "EAGLE"), ("Continuous", "COLIBRE"), ("Stochastic", "COLIBRE"))


def run(ctx):
    for mode, conv in RUNS:
        sub = f"{mode}_{conv}"
        os.makedirs(ctx.path(sub))
        p, mg, ms = c.setup(_Sub(ctx, ctx.path(sub)), {
            "use_SNII_feedback": 0, "use_SNIa_feedback": 1,
            "use_SNIa_enrichment": 1, "SNIa_DTD_convention": conv,
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
        ctx.swift(p, c.FLAGS, subdir=sub, name=sub, timeout=3600)


def expected_energy(fb, conv, m_star_cgs, age1_yr, age2_yr):
    nu = fb["SNIa_DTD_exp_norm_p_Msun"]
    tau = fb["SNIa_DTD_exp_timescale_Gyr"] * 1e9
    td = fb["SNIa_DTD_delay_Gyr"] * 1e9
    age1_yr = max(age1_yr, td)
    shift = td if conv == "COLIBRE" else 0.0
    n_per_msun = nu * (math.exp(-(age1_yr - shift) / tau)
                       - math.exp(-(age2_yr - shift) / tau))
    return fb["SNIa_energy_erg"] * c.N_STARS * m_star_cgs / units.SOLAR_MASS * n_per_msun


def analyze(ctx):
    fb = ctx.info["feedback"]
    dE_heat = ctx.info["m_gas"] * units.BOLTZMANN_K * DELTA_T / (
        (units.GAMMA - 1.0) * units.MU_IONISED * units.PROTON_MASS)
    checks = []
    for mode, conv in RUNS:
        sub = f"{mode}_{conv}"
        snaps = snapshots(ctx.path(sub))
        age1, age2 = snaps[0].time_cgs / units.YEAR, snaps[-1].time_cgs / units.YEAR
        E = expected_energy(fb, conv, ctx.info["m_star"], age1, age2)
        gained = c.total_energy(snaps[-1]) - c.total_energy(snaps[0])
        ctx.info[sub] = {"E_expected": E, "gained": gained, "ages_yr": [age1, age2]}
        if mode == "Continuous":
            checks.append(Check(f"continuous, {conv}: energy gained / DTD energy",
                                gained / E, 1.0, kind="rel", tol=1e-2,
                                source=f"{conv} exponential DTD integrated over the run"))
        else:
            n_exp = E / dE_heat
            checks.append(Check(f"stochastic, {conv}: energy gained / DTD energy",
                                gained / E, 1.0,
                                kind="sigma", tol=4.0, sigma=1 / np.sqrt(n_exp),
                                source=f"Poisson scatter of {n_exp:.0f} expected events"))
    return checks
