"""EAGLE-thermal-kinetic splits the CCSN budget between heating and kicks.

Tests T3 and T4 of the module design (ai-notes feedback-hybrid-colibre.md). The box
of feedback/_common.py (128 stars of 20 m_g, 50 Myr, CCSN energy only, f_E = 1),
heating by Delta T = 10^7 K and pair kicks at 600 km/s, for f_kin = 0 and 0.1.

The module counts every heating event in the star field
FeedbackNumberOfHeatingEvents (number_of_heating_events), so the thermal energy
it injected is known exactly: N_heat x dE_heat, with
dE_heat = m_g k Delta T / ((gamma - 1) mu_ion m_p).

Checks, for each f_kin:
  1. f_kin = 0: energy gained = N_heat x dE_heat to 1e-2. The tolerance is SPH
     energy conservation: the 32^3 Sedov blast (sn_blast/sedov) conserved energy
     to 3.1e-3 with one blast; this run, with about 7000 heating events at the
     same resolution, gave 3.5e-3 (2026-10-20);
  2. N_heat against (1 - f_kin) E_SN / dE_heat within 4 Poisson sigma;
  3. f_kin = 0.1: energy gained minus N_heat x dE_heat, i.e. the kinetic part,
     against f_kin E_SN within 4 sigma of the number of pairs (E_SN f_kin /
     (m_g dv^2), about 420), widened by 4e-3 of the thermal
     energy (check 1 measured 3.5e-3) for the energy-conservation error.
"""

import os

import numpy as np

from swiftreg import units
from swiftreg.checks import Check
from swiftreg.snapshot import snapshots

from . import _common as c
from .thermal_kinetic_wiring import COMMON, _Sub

DESCRIPTION = "EAGLE-thermal-kinetic: heating events x dE_heat and the kinetic share of the budget"
TIER = "quick"
BUILDS = {"thermal_kinetic": COMMON + ["--with-feedback=EAGLE-thermal-kinetic"]}

DELTA_T = 1e7
DELTA_V = 600.0
F_KIN = (0.0, 0.1)


def run(ctx):
    for fk in F_KIN:
        sub = f"fkin{fk:g}"
        os.makedirs(ctx.path(sub))
        p, mg, ms = c.setup(_Sub(ctx, ctx.path(sub)),
                            {"SNII_delta_T_K": DELTA_T, "SNII_delta_v_km_p_s": DELTA_V,
                             "SNII_f_kinetic": fk})
        ctx.info["m_gas"], ctx.info["m_star"] = mg, ms
        ctx.swift(p, c.FLAGS, subdir=sub, name=sub, timeout=3600)


def analyze(ctx):
    mg, ms = ctx.info["m_gas"], ctx.info["m_star"]
    fb = ctx.info["feedback"]
    budget = c.expected_sn_energy(fb, ms)
    dE_heat = mg * units.BOLTZMANN_K * DELTA_T / (
        (units.GAMMA - 1.0) * units.MU_IONISED * units.PROTON_MASS)
    dE_pair = mg * (DELTA_V * 1e5) ** 2
    checks = []
    for fk in F_KIN:
        snaps = snapshots(ctx.path(f"fkin{fk:g}"))
        gained = c.total_energy(snaps[-1]) - c.total_energy(snaps[0])
        n_heat = np.sum(snaps[-1].read(4, "FeedbackNumberOfHeatingEvents", cgs=False))
        e_heat = n_heat * dE_heat
        n_heat_exp = (1 - fk) * budget / dE_heat
        ctx.info[f"fkin{fk:g}"] = dict(gained=gained, n_heat=n_heat, n_heat_exp=n_heat_exp)
        checks.append(Check(f"f_kin={fk:g}: heating events", n_heat, n_heat_exp, kind="sigma",
                            tol=4.0, sigma=np.sqrt(n_heat_exp),
                            source="(1 - f_kin) E_SN / dE_heat, Chabrier IMF 8-100 Msun"))
        if fk == 0:
            checks.append(Check("f_kin=0: energy gained / (N_heat x dE_heat)",
                                gained / e_heat, 1.0, kind="rel", tol=1e-2,
                                source="each event adds dE_heat; SPH energy conservation"))
        else:
            n_pairs = fk * budget / dE_pair
            sig = np.sqrt(n_pairs) * dE_pair + 4e-3 * e_heat
            checks.append(Check(f"f_kin={fk:g}: kinetic share (gained - N_heat dE_heat)",
                                gained - e_heat, fk * budget, kind="sigma", tol=4.0,
                                sigma=sig,
                                source="f_kin E_SN; Poisson scatter of the pair number"))
    return checks
