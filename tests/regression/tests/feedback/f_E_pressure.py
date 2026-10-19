"""Energy per SN from the birth pressure in EAGLE-thermal-kinetic (test T6).

COLIBRE (Schaye et al. 2026, eq. 20 and footnote 16):
    f_E = f_E,min + (f_E,max - f_E,min) / (1 + exp(-(log10 P_birth - log10 P_0)/sigma_P))
    P_birth / k_B = rho_birth T_birth / m_p   (no mean molecular weight)
with the m6 values f_E,min = 0.3, f_E,max = 4, P_0/k_B = 1e4 K cm^-3, sigma_P = 0.3.

The feedback box of feedback/_common.py (128 stars of 20 m_g, 50 Myr, CCSN energy
only, f_kin = 0, Delta T = 1e7 K, no cooling), with the stars' birth temperature set
to 1e4 K and birth density to give P_birth = P_0/10, P_0 and 10 P_0:
expected f_E = 0.427, 2.15 and 3.873.

Checks per run:
  - the f_E recorded on the stars (FeedbackEnergyFractions) equals eq. 20 to 1e-5;
  - energy gained by the gas / (f_E x CCSN budget) equals 1 within 4 Poisson sigma of
    the number of heating events.
"""

import math
import os

import numpy as np

from swiftreg import units
from swiftreg.checks import Check
from swiftreg.snapshot import snapshots

from . import _common as c
from .thermal_kinetic_wiring import COMMON, _Sub

DESCRIPTION = "EAGLE-thermal-kinetic: f_E from the birth pressure (S26 eq. 20)"
TIER = "quick"
BUILDS = {"thermal_kinetic": COMMON + ["--with-feedback=EAGLE-thermal-kinetic"]}

F_MIN, F_MAX, P0, SIGMA = 0.3, 4.0, 1e4, 0.3
T_BIRTH = 1e4
PRESSURES = (P0 / 10, P0, 10 * P0)
DELTA_T = 1e7


def f_E(P):
    return F_MIN + (F_MAX - F_MIN) / (1 + math.exp(-(math.log10(P) - math.log10(P0)) / SIGMA))


def run(ctx):
    for P in PRESSURES:
        sub = f"P{P:g}"
        os.makedirs(ctx.path(sub))
        p, mg, ms = c.setup(_Sub(ctx, ctx.path(sub)), {
            "SNII_f_kinetic": 0.0, "SNII_delta_T_K": DELTA_T,
            "SNII_energy_fraction_function": "Pressure",
            "SNII_energy_fraction_min": F_MIN, "SNII_energy_fraction_max": F_MAX,
            "SNII_energy_fraction_P_0_K_p_cm3": P0,
            "SNII_energy_fraction_sigma_P": SIGMA})
        rho_birth = P * units.PROTON_MASS / T_BIRTH  # g cm^-3, P/k = rho T / m_p
        p["Stars"].update({"overwrite_birth_density": 1,
                           "birth_density": rho_birth / units.UNIT_DENSITY,
                           "overwrite_birth_temperature": 1,
                           "birth_temperature": T_BIRTH})
        ctx.info["m_gas"], ctx.info["m_star"] = mg, ms
        ctx.swift(p, c.FLAGS, subdir=sub, name=sub, timeout=3600)


def analyze(ctx):
    mg, ms = ctx.info["m_gas"], ctx.info["m_star"]
    fb = dict(ctx.info["feedback"])
    fb["SNII_energy_fraction_max"] = 1.0  # budget per unit f_E
    budget_fE1 = c.expected_sn_energy(fb, ms)
    dE_heat = mg * units.BOLTZMANN_K * DELTA_T / (
        (units.GAMMA - 1.0) * units.MU_IONISED * units.PROTON_MASS)
    checks = []
    for P in PRESSURES:
        snaps = snapshots(ctx.path(f"P{P:g}"))
        fE_expected = f_E(P)
        fE_stars = snaps[-1].read(4, "FeedbackEnergyFractions", cgs=False)
        gained = c.total_energy(snaps[-1]) - c.total_energy(snaps[0])
        n_exp = fE_expected * budget_fE1 / dE_heat
        checks.append(Check(f"P_birth = {P:g}: max |f_E(stars) / eq. 20 - 1|",
                            np.max(np.abs(fE_stars / fE_expected - 1)), 1e-5, kind="max",
                            source=f"S26 eq. 20: f_E = {fE_expected:.4f}"))
        checks.append(Check(f"P_birth = {P:g}: energy gained / (f_E x budget)",
                            gained / (fE_expected * budget_fE1), 1.0, kind="sigma", tol=4.0,
                            sigma=1 / np.sqrt(n_exp),
                            source="Chabrier IMF 8-100 Msun; Poisson scatter of events"))
    return checks
