"""L-T1: a static medium of heavy and light particles in pressure equilibrium.

A periodic unit cube holds 16^3 particles on a cubic lattice (one number
density). A random half are light, of mass m_g / R, with internal energy
R u_g, so each particle stands for gas of density m / V and the same
pressure P = 1 (rho_heavy = 1, gamma = 5/3, at rest): a two-phase medium
mixed at the particle scale, which is what a population of hot light wind
particles among ambient gas is. The exact solution is rest. Mass ratios
R = 1, 3, 10, 30, with SPHENIX and GIZMO-MFM (3D builds, --limiter; MFM
also --sync), run for two sound-crossing times of the box at the heavy
sound speed c_g = (gamma P / rho_heavy)^1/2.

Measured: the pressure error at t = 0 (rms of P_i / P - 1 over all
particles, from the first density estimate) and the largest rms speed over
c_g in snapshots every t_cross / 4 during the first crossing. The first
crossing is used because the equal-mass SPH lattice is itself unstable:
its rms speed grows from 4e-7 c_g at t_cross / 4 to 0.07 c_g at 2 t_cross
(measured), which would hide the mass-ratio error later.

Expected: SPH estimates every density as a kernel sum over neighbours of
both masses, about (1 + 1/R) / 2 of the heavy density, so a light particle's
pressure is too high by up to (R + 1) / 2 and the medium is set in motion;
MFM gives each particle the volume set by the number density, so densities
and pressures are exact on the lattice.

Checks: R = 1 is quiet in both schemes (peak rms speed below 1e-3 c_g);
MFM stays below 1e-2 c_g at every ratio (the pass criterion of design
Sec. 6, L-T1); for SPHENIX the pressure errors and speeds are the result,
checked to stay within a factor 1.5 of the values measured on 2026-10-20
(a regression guard).
"""

import os

import numpy as np

from swiftreg.checks import Check
from swiftreg.snapshot import snapshots

from . import _mixed as mx

DESCRIPTION = "L-T1: heavy and light particles in pressure equilibrium, R = 1-30"
TIER = "quick"
BUILDS = {k: mx.build(s) for k, s in mx.SCHEMES.items()}

N = 16
T_CROSS = 1.0 / np.sqrt(mx.GAMMA)
T_END = 2.0 * T_CROSS

# Measured 2026-10-20 (SPHENIX): R -> (rms P error at t = 0, peak rms
# speed / c_g in the first crossing)
SPH_REFERENCE = {3: (0.599, 0.164), 10: (2.56, 0.353), 30: (8.22, 0.419)}


def _ics(path, R):
    x = (np.arange(N) + 0.5) / N
    xx, yy, zz = np.meshgrid(x, x, x, indexing="ij")
    pos = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    n = N ** 3
    rng = np.random.default_rng(11)
    light = np.zeros(n, bool)
    light[rng.permutation(n)[: n // 2]] = True
    m_g = 1.0 / n
    m = np.where(light, m_g / R, m_g)
    u_g = 1.0 / (mx.GAMMA - 1.0)
    u = np.where(light, R * u_g, u_g)
    mx.write_ic(path, pos, np.zeros((n, 3)), m, u, np.full(n, 1.2348 / N), [1.0] * 3, 3)
    return light


def run(ctx):
    for b, scheme in mx.SCHEMES.items():
        for R in mx.RATIOS:
            sub = f"{b}_R{R}"
            os.makedirs(ctx.path(sub), exist_ok=True)
            _ics(ctx.path(sub, "ics.hdf5"), R)
            p = mx.params(T_END, n_snap=8, top_cells=3)
            # Twice the initial h; kernel_gamma h_max stays below the top-level
            # cell (1/3). Without the cap SPHENIX at R = 30 stopped at
            # t = 0.57 t_cross with smoothing lengths beyond the cell size.
            p["SPH"]["h_max"] = 2 * 1.2348 / N
            ctx.swift(p, mx.flags(scheme), binary=b, subdir=sub, name="run",
                      timeout=1800)


def measure(ctx, b, R):
    snaps = snapshots(ctx.path(f"{b}_R{R}"))
    P0 = snaps[0].read(0, "Pressures", cgs=False)
    c_g = np.sqrt(mx.GAMMA)
    v_rms = []
    for s in snaps:
        v = s.read(0, "Velocities", cgs=False)
        v_rms.append(float(np.sqrt(np.mean((v ** 2).sum(axis=1))) / c_g))
    return {"P_err0": float(np.sqrt(np.mean((P0 - 1.0) ** 2))),
            "v_peak": max(v_rms[1:5]), "v_rms_end": v_rms[-1], "v_rms_series": v_rms}


def analyze(ctx):
    checks = []
    for b in mx.SCHEMES:
        res = {R: measure(ctx, b, R) for R in mx.RATIOS}
        ctx.info[b] = res
        checks.append(Check(f"{b} R=1: peak rms speed / c_g", res[1]["v_peak"], 1e-3,
                            kind="max", source="equal masses on a lattice: equilibrium"))
        for R in mx.RATIOS[1:]:
            if b == "mfm":
                checks.append(Check(f"mfm R={R}: peak rms speed / c_g", res[R]["v_peak"],
                                    1e-2, kind="max",
                                    source="design L-T1 pass criterion"))
            elif R in SPH_REFERENCE:
                ref = SPH_REFERENCE[R]
                checks.append(Check(f"sph R={R}: rms P error at t=0", res[R]["P_err0"],
                                    1.5 * ref[0], kind="max",
                                    source=f"regression guard, measured {ref[0]:.3g}"))
                checks.append(Check(f"sph R={R}: peak rms speed / c_g", res[R]["v_peak"],
                                    1.5 * ref[1], kind="max",
                                    source=f"regression guard, measured {ref[1]:.3g}"))
    return checks
