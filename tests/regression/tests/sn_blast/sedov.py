"""Adiabatic Sedov-Taylor blast of one SN in uniform gas, at two resolutions.

10^51 erg in the 32 particles nearest the centre of an N^3 lattice
(n_H = 1 cm^-3, T = 10^4 K, 100 pc box), N = 32 and 64 (m_g = 1.1 and
0.14 Msun, smoothing length h = 3.9 and 1.9 pc); hydro only, 49 kyr. The exact
solution (swiftreg/sedov.py, from the SWIFT Sedov example) gives the shock
radius R = 1.152 (E t^2/rho)^(1/5) for gamma = 5/3: 12.5 pc at 9.8 kyr and
23.9 pc at 48.9 kyr. It neglects the background pressure (10^4 K); the shock
Mach number stays above 10.

The measured shock radius is where the density, binned in shells of 0.5 pc,
falls outside its maximum to half-way between the maximum and rho_0. The exact
shell is about R/12 thick, comparable to h, so SPH spreads it: at 64^3 the
binned maximum is 1.7-2.2 rho_0 against 4 rho_0 exact, its radius lies 0.5-5 per
cent inside the exact front, and the half-height radius 5-6 per cent outside it
(2026-10-20). Both offsets are about half a smoothing length, so the test checks
that the error shrinks with resolution rather than a fixed band.

Checks:
  - total energy (thermal + kinetic) at every output within 1 per cent of the
    initial value, at both resolutions;
  - at the last output, |R/R_exact - 1| at 64^3 at most 0.81 times its value at
    32^3, i.e. convergence of order at least 0.3. Borrow et al. (2022, SPHENIX,
    Sec. 5.2.2 and Fig. 6) find Sedov convergence of at best L1 proportional to
    h^(1/2) for SPHENIX, a ratio of 0.71; measured here 0.68 (2026-10-20);
  - at the last output at 64^3, |R/R_exact - 1| < 0.10.
"""

import os

import numpy as np

from swiftreg import units
from swiftreg.checks import Check
from swiftreg.sedov import sedov
from swiftreg.snapshot import snapshots

from . import _common as c

DESCRIPTION = "adiabatic Sedov-Taylor blast: energy conservation and shock-radius convergence"
TIER = "quick"
BUILDS = {"hydro": []}

RESOLUTIONS = (32, 64)
BOX = 0.1  # kpc
NH = 1.0
T = 1e4
DT_SNAP = 1e-5  # internal units, 9.78 kyr
T_END = 5e-5


def run(ctx):
    for n in RESOLUTIONS:
        sub = f"n{n}"
        os.makedirs(ctx.path(sub))
        _, rho0 = c.blast_ics(ctx.path(sub, "ics.hdf5"), n, BOX, NH, T)
        ctx.info["rho0"] = rho0
        p = {
            "InternalUnitSystem": units.internal_unit_params(),
            "TimeIntegration": {"time_begin": 0.0, "time_end": T_END,
                                "dt_min": 1e-14, "dt_max": 1e-6},
            "Snapshots": {"basename": "snap", "time_first": 0.0, "delta_time": DT_SNAP},
            "Statistics": {"delta_time": DT_SNAP, "time_first": 0.0},
            "InitialConditions": {"file_name": "ics.hdf5", "periodic": 1},
            "SPH": {"resolution_eta": 1.2348, "CFL_condition": 0.1,
                    "minimal_temperature": 10.0},
        }
        ctx.swift(p, ["--hydro", "--limiter"], subdir=sub, name=sub, timeout=3600)


def half_height_radius(mid, rho_bin, rho0):
    """Radius outside the density maximum where rho falls to (max + rho0)/2."""
    i = int(np.nanargmax(rho_bin))
    half = 0.5 * (rho_bin[i] + rho0)
    for j in range(i, len(mid) - 1):
        if rho_bin[j + 1] <= half:
            f = (rho_bin[j] - half) / (rho_bin[j] - rho_bin[j + 1])
            return mid[j] + f * (mid[j + 1] - mid[j])
    return mid[-1]


def total_energy(d):
    return np.sum(d["m"] * d["u"]) + 0.5 * np.sum(d["m"] * np.sum(d["v"] ** 2, axis=1))


def analyze(ctx):
    box_cm = BOX * units.UNIT_LENGTH
    rho0 = ctx.info["rho0"]
    edges = np.arange(0, 0.5 * box_cm, 0.5 * units.PARSEC)
    mid = 0.5 * (edges[1:] + edges[:-1])

    checks, err_final, history = [], {}, {}
    rows = []
    for n in RESOLUTIONS:
        snaps = snapshots(ctx.path(f"n{n}"))
        E0 = total_energy(c.radial(snaps[0], box_cm))
        worst_dE, rows = 0.0, []
        for s in snaps[1:]:
            d = c.radial(s, box_cm)
            R_sim = half_height_radius(mid, c.binned(d["r"], d["rho"], edges), rho0)
            R_ex = sedov(s.time_cgs, c.E_SN, rho0, units.GAMMA, 2000, 3)[4]
            rows.append([s.time_cgs / units.YEAR / 1e3, R_sim / units.PARSEC,
                         R_ex / units.PARSEC])
            worst_dE = max(worst_dE, abs(total_energy(d) / E0 - 1.0))
        err_final[n] = abs(rows[-1][1] / rows[-1][2] - 1.0)
        history[f"n{n}_t_kyr_Rsim_Rexact"] = rows
        checks.append(Check(f"{n}^3: max |energy/initial - 1| over outputs", worst_dE, 0.01,
                            kind="max", source="adiabatic, periodic box"))
    ctx.info.update(history)
    lo, hi = RESOLUTIONS
    checks.append(Check(f"shock radius error {hi}^3 / {lo}^3 at {rows[-1][0]:.1f} kyr",
                        err_final[hi] / err_final[lo], 0.81, kind="max",
                        source="order >= 0.3; SPHENIX Sedov convergence ~h^0.5 "
                               "(Borrow et al. 2022, Fig. 6)"))
    checks.append(Check(f"{hi}^3: |R/R_exact - 1| at {rows[-1][0]:.1f} kyr", err_final[hi],
                        0.10, kind="max",
                        source="exact Sedov-Taylor solution; about h/2 offset of the estimator"))
    return checks
