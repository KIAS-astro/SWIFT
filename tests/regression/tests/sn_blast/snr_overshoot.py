"""Why does SPHENIX overshoot the terminal momentum of a resolved remnant? (long tier)

With EAGLE cooling, one SN of 1e51 erg in 2.1 Msun particles (64^3 glass, 256 pc box,
n_H = 1 cm^-3, 4400 K, solar metallicity) reaches 1.93 times the Kim & Ostriker (2015)
terminal momentum at 1.5 Myr with SPHENIX and keeps rising; GIZMO-MFM gives 0.98 and
Phantom 0.94 (sn-blast-tests single_sn_long, SWIFT_REG_SNR_RES=low). This test repeats
that set-up in variants that separate the candidate causes (ai-notes
SWIFT-star-formation/papers-part12-hopkins.md Sec. 2.3, tests 1 and 3):

  cool   the set-up above (EAGLE cooling), as the reference;
  adiab  no cooling: the radial momentum must follow the Sedov-Taylor solution;
  diff0  cooling, SPHENIX artificial conduction off (diffusion_alpha_max = 0);
  diff2  cooling, SPHENIX artificial conduction doubled (diffusion_alpha_max and
         diffusion_beta both 2: alpha_diff is capped by alpha_max and grows at a
         rate proportional to beta, so both bounds double).

SWIFT_REG_SNR_VARIANTS selects a comma-separated subset (default: all four for
SPHENIX; diff0 and diff2 need SPHENIX). Outputs are log-spaced from 2 kyr to 1.5 Myr
so that the early, pre-shell-formation phase is sampled.

Checks: the adiabatic momentum at 0.1 Myr within 10 per cent of the Sedov-Taylor
value (resolution: the shock radius is 8 particle spacings then). The overshoot
itself is recorded without a pass criterion: p / p_KO15 at 0.45 and 1.5 Myr.
"""

import os

import numpy as np

from swiftreg import params, units
from swiftreg.checks import Check

from . import _common as c
from .single_sn_cooling import BUILDS, HYDRO, KO15_P_FINAL, NH, T, history  # noqa: F401

DESCRIPTION = "single SN to 1.5 Myr: adiabatic and conduction variants (SPHENIX overshoot)"
TIER = "long"
BOX = 0.256  # kpc
T_END = 1.5339e-3  # 1.5 Myr
VARIANTS = {"cool": {}, "adiab": {},
            "diff0": {"diffusion_alpha_max": 0.0},
            "diff2": {"diffusion_alpha_max": 2.0, "diffusion_beta": 2.0}}
_only = os.environ.get("SWIFT_REG_SNR_VARIANTS")
RUNS = {k: v for k, v in VARIANTS.items() if not _only or k in _only.split(",")}


def sedov_momentum(t_yr, rho0, e=c.E_SN, gamma=5.0 / 3.0):
    """Radial momentum [Msun km/s] of the exact Sedov-Taylor solution at t_yr."""
    from swiftreg.sedov import sedov
    t = t_yr * units.YEAR
    r, _, rho, v = sedov(t, e, rho0, gamma, n=4000)[:4]
    order = np.argsort(r)
    r, rho, v = r[order], rho[order], v[order]
    return np.trapezoid(4 * np.pi * r ** 2 * rho * v, r) / (units.SOLAR_MASS * 1e5)


def run(ctx):
    ctx.table("coolingtables")
    from swiftreg import data
    glass = data.glass(64, ctx.data_dir, log=ctx.log)
    times = np.logspace(np.log10(2.0452e-6), np.log10(T_END), 80)
    for name, sph in RUNS.items():
        os.makedirs(ctx.path(name))
        _, rho0 = c.blast_ics(ctx.path(name, "ics.hdf5"), 64, BOX, NH, T, glass_file=glass)
        ctx.info["rho0"] = rho0
        np.savetxt(ctx.path(name, "output_list.txt"), times, header="Time", comments="# ")
        p = params.eagle_base("ics.hdf5", T_END, ctx.data_dir, 1.0, dt_max=1e-6)
        p["Snapshots"].update({"compression": 4, "output_list_on": 1,
                               "output_list": "output_list.txt"})
        p["EAGLEEntropyFloor"].update({"Jeans_temperature_norm_K": 10.0,
                                       "Cool_temperature_norm_K": 10.0})
        p["SPH"].update(sph)
        flags = ["--hydro", "--limiter"] + ([] if name == "adiab" else ["--cooling"])
        ctx.swift(p, flags, subdir=name, name=name, timeout=24 * 3600)


def analyze(ctx):
    checks = []
    rho0 = ctx.info["rho0"]
    for name in RUNS:
        rows = history(ctx.path(name), box_kpc=BOX)
        ctx.info[f"{name}: t_yr_Mhot_Msun_pr_Msunkms_E_erg"] = rows.tolist()
        t, p_r = rows[:, 0], rows[:, 2]
        if name == "adiab":
            checks.append(Check("adiabatic: radial momentum at 0.1 Myr / Sedov-Taylor",
                                np.interp(1e5, t, p_r) / sedov_momentum(1e5, rho0), 1.0,
                                kind="rel", tol=0.10, source="exact Sedov-Taylor solution"))
        else:
            for t_myr in (0.45, 1.5):
                ctx.info[f"{name}: p_over_KO15_final_at_{t_myr}Myr"] = float(
                    np.interp(t_myr * 1e6, t, p_r) / KO15_P_FINAL)
    return checks
