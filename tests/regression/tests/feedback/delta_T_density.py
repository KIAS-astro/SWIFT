"""Density-dependent heating temperature of EAGLE-thermal-kinetic (test T5).

COLIBRE's temperature change (Schaye et al. 2026, eq. 18):
    Delta T = Delta T_0 (n_H,SN / n_pivot)^(2/3), clamped to [Delta T_min, Delta T_max]
with n_H,SN the SPH gas density at the star. Here the COLIBRE m6 values:
Delta T_0 = 10^6.5 K, n_pivot = 0.5 cm^-3, Delta T_min = 10^6.75 K,
Delta T_max = 10^8 K (Table 1 of S26).

Three uniform boxes (1 kpc, 32^3, 1e4 K, no cooling), 64 stars of one gas mass each,
f_kin = 0, CCSN energy only. Each star releases all its CCSN energy once, at
0.1 Myr (fixed delay), into undisturbed gas, so the density at the star is the box
density when Delta T is set. (With continuous release the first events blow hot
bubbles, later events see lower densities and use lower Delta T: a first version
over 10 Myr found 0.45 and 0.68 of the box-density value at n_H = 5 and 300,
2026-10-20.) The run ends at 0.2 Myr. The stars are one gas mass each so that the
heating probability stays below 1 at every density: above it the energy is shared by
all neighbours and Delta T no longer applies (with 20 m_g stars the n_H = 0.1 box
was in that regime and gave 2.5 times the clamp value).
  n_H = 0.1:   formula gives 10^6.04 K, below the clamp, so Delta T = 10^6.75 K
  n_H = 5:     formula gives 10^7.17 K, inside the bounds
  n_H = 300:   formula gives 10^8.35 K, above the clamp, so Delta T = 10^8 K
Each heating event adds m_g k Delta T / ((gamma - 1) mu_ion m_p), and the module
counts events (FeedbackNumberOfHeatingEvents), so the energy gained per event gives
the Delta T applied. Check: Delta T applied / Delta T expected within 2 per cent
(the SPH density at a star on a lattice differs from the box density by under
1 per cent, which changes Delta T by under 0.7 per cent inside the bounds; SPH
energy conservation adds under 0.4 per cent, see energy_budget_thermal_kinetic).
"""

import os

import numpy as np

from swiftreg import units
from swiftreg.checks import Check
from swiftreg.snapshot import snapshots

from . import _common as c
from .thermal_kinetic_wiring import COMMON, _Sub

DESCRIPTION = "EAGLE-thermal-kinetic: Delta T(n_H) of S26 eq. 18 with its clamps"
TIER = "quick"
BUILDS = {"thermal_kinetic": COMMON + ["--with-feedback=EAGLE-thermal-kinetic"]}

DT0, NPIV, DTMIN, DTMAX = 10 ** 6.5, 0.5, 10 ** 6.75, 1e8
DENSITIES = (0.1, 5.0, 300.0)


def expected_dT(nH):
    return min(max(DT0 * (nH / NPIV) ** (2.0 / 3.0), DTMIN), DTMAX)


def run(ctx):
    for nH in DENSITIES:
        sub = f"n{nH:g}"
        os.makedirs(ctx.path(sub))
        p, mg, _ = c.setup(_Sub(ctx, ctx.path(sub)), {
            "SNII_f_kinetic": 0.0, "SNII_delta_T_K": DT0,
            "SNII_delta_T_model": "DensityDependent",
            "SNII_delta_T_n_pivot_H_p_cm3": NPIV,
            "SNII_delta_T_min_K": DTMIN, "SNII_delta_T_max_K": DTMAX,
            "SNII_sampled_delay": 0, "SNII_wind_delay_Gyr": 1e-4},
            n_stars=64, t_end=2.05e-4, nH=nH, star_mass_in_gas_masses=1.0)
        p["TimeIntegration"]["dt_max"] = 1e-5
        ctx.info[f"m_gas_{sub}"] = mg
        ctx.swift(p, c.FLAGS, subdir=sub, name=sub, timeout=3600)


def analyze(ctx):
    k = units.BOLTZMANN_K / ((units.GAMMA - 1.0) * units.MU_IONISED * units.PROTON_MASS)
    checks = []
    for nH in DENSITIES:
        sub = f"n{nH:g}"
        snaps = snapshots(ctx.path(sub))
        gained = c.total_energy(snaps[-1]) - c.total_energy(snaps[0])
        n_heat = np.sum(snaps[-1].read(4, "FeedbackNumberOfHeatingEvents", cgs=False))
        dT = gained / (n_heat * ctx.info[f"m_gas_{sub}"] * k)
        ctx.info[f"{sub}_n_heat"] = n_heat
        checks.append(Check(f"n_H = {nH:g}: Delta T applied / expected", dT / expected_dT(nH),
                            1.0, kind="rel", tol=0.02,
                            source=f"S26 eq. 18 with m6 values: {expected_dT(nH):.3g} K"))
        checks.append(Check(f"n_H = {nH:g}: heating events", n_heat, 1, kind="min",
                            source="at least one event, else the ratio is undefined"))
    return checks
