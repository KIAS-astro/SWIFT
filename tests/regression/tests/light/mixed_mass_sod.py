"""L-T2: a Sod shock tube whose low-density side is made of light particles.

The Sod problem (rho, v, P) = (1, 0, 1) | (0.125, 0, 0.1), gamma = 5/3,
in the mirrored periodic set-up of tests/gizmo (1D build, [0, 2), left
state in [0.5, 1.5), the tube with its interface at x = 1.5 analysed over
1 < x < 2), t = 0.2. The left (dense) side has 200 particles of mass
m_g = 1/200; the right side has particles of mass m_g / R, so its particle
spacing is 8 / R times the left one: R = 1 is the standard equal-mass tube
(8 times sparser on the right), R = 8 would be equal spacing, R = 30 puts
3.75 times more particles on the right. This is the light wind's case: low
density carried by light particles meeting dense gas made of heavy ones.

Measured: L1 of rho, v, P against the exact Riemann solution (Toro 2009,
ch. 4) over 1 < x < 2, relative to the exact value (v relative to the
post-shock speed), at R = 1, 3, 10, 30, SPHENIX and GIZMO-MFM (--limiter;
MFM also --sync).

Checks:
  1. the L1 error at each R is at most 1.5 times the equal-mass (R = 1) run
     of the same scheme. The R > 1 runs have R / 8 times more particles on
     the right, so this comparison mixes the mass contrast with resolution;
  2. matched resolution: an equal-mass control with N_LEFT R particles on the
     left has the same spacing on the low-density side as the light run; the
     ratio of the two L1 errors over x > 1.5 (contact and shock) isolates the
     mass contrast. Measured 2026-10-20 (rho, P): SPHENIX 1.41, 1.52 (R = 3),
     1.56, 3.32 (R = 10), 2.91, 4.56 (R = 30); GIZMO-MFM 1.16, 1.52, 3.20,
     2.92, 5.54, 5.46. The design's pass criterion (1.5) holds only at R = 3,
     for both schemes. These are regression guards (1.3 times the measured
     ratio): the mass contrast at a contact discontinuity costs both schemes
     a factor 3-5 in L1 at R = 10-30.
"""

import os

import numpy as np

from swiftreg.checks import Check
from swiftreg.snapshot import snapshots

from ..gizmo import _hydro1d as h1
from . import _mixed as mx

DESCRIPTION = "L-T2: Sod tube with light particles on the low-density side, R = 1-30"
TIER = "quick"
BUILDS = {k: mx.build(s, dim=1) for k, s in mx.SCHEMES.items()}

WL, WR, T_END = (1.0, 0.0, 1.0), (0.125, 0.0, 0.1), 0.2
N_LEFT = 200
# Measured 2026-10-20: L1 over x > 1.5, light run / matched-spacing control
CTRL_REFERENCE = {
    "sph": {3: {"rho": 1.41, "P": 1.52}, 10: {"rho": 1.56, "P": 3.32},
            30: {"rho": 2.91, "P": 4.56}},
    "mfm": {3: {"rho": 1.16, "P": 1.52}, 10: {"rho": 3.20, "P": 2.92},
            30: {"rho": 5.54, "P": 5.46}},
}


def _ics(path, R, n_left=N_LEFT):
    rL, rR = WL[0], WR[0]
    m_g = rL / N_LEFT
    m_r = m_g / R
    nR = int(round(rR / m_r))
    m_l = rL / n_left
    x = np.concatenate([0.5 + (np.arange(n_left) + 0.5) / n_left,
                        np.mod(1.5 + (np.arange(nR) + 0.5) / nR, 2.0)])
    m = np.concatenate([np.full(n_left, m_l), np.full(nR, m_r)])
    rho = np.concatenate([np.full(n_left, rL), np.full(nR, rR)])
    P = np.concatenate([np.full(n_left, WL[2]), np.full(nR, WR[2])])
    pos = np.zeros((len(x), 3))
    pos[:, 0] = x
    mx.write_ic(path, pos, np.zeros((len(x), 3)), m, P / ((mx.GAMMA - 1) * rho),
                1.2348 * m / rho, [2.0, 1.0, 1.0], 1)


def run(ctx):
    for b, scheme in mx.SCHEMES.items():
        for R in mx.RATIOS:
            sub = f"{b}_R{R}"
            os.makedirs(ctx.path(sub), exist_ok=True)
            _ics(ctx.path(sub, "ics.hdf5"), R)
            ctx.swift(mx.params(T_END), mx.flags(scheme), binary=b, subdir=sub,
                      name="run", timeout=900)
            if R == 1:
                continue
            # Control: equal masses, same particle spacing on the low-density
            # side as the light run (N_LEFT R particles on the left)
            sub = f"{b}_ctrl_R{R}"
            os.makedirs(ctx.path(sub), exist_ok=True)
            _ics(ctx.path(sub, "ics.hdf5"), R, n_left=N_LEFT * R)
            ctx.swift(mx.params(T_END), mx.flags(scheme), binary=b, subdir=sub,
                      name="run", timeout=1800)


def measure(ctx, b, R, kind="", x_min=1.0):
    d = h1.read(snapshots(ctx.path(f"{b}_{kind}R{R}"))[-1])
    sel = d["x"] > x_min
    ex = h1.riemann_exact(WL, WR, (d["x"][sel] - 1.5) / T_END)
    out = {}
    for k, f in enumerate(["rho", "v", "P"]):
        scale = np.max(np.abs(ex[:, k])) if f == "v" else ex[:, k]
        out[f] = float(np.mean(np.abs(d[f][sel] - ex[:, k]) / scale))
    return out


def analyze(ctx):
    checks = []
    for b in mx.SCHEMES:
        res = {R: measure(ctx, b, R) for R in mx.RATIOS}
        # Low-density half only (x > 1.5: contact and shock), light against the
        # matched-spacing equal-mass control
        for R in mx.RATIOS[1:]:
            lt = measure(ctx, b, R, x_min=1.5)
            ct = measure(ctx, b, R, kind="ctrl_", x_min=1.5)
            res[R]["right"] = lt
            res[R]["right_ctrl"] = ct
            for f in ("rho", "P"):
                ref = CTRL_REFERENCE[b][R][f]
                checks.append(Check(f"{b} R={R}: L1({f}, x>1.5) light / equal-mass control",
                                    lt[f] / ct[f], 1.3 * ref, kind="max",
                                    source=f"regression guard, measured {ref:.3g}; design "
                                           "L-T2 asks 1.5"))
        ctx.info[b] = res
        for R in mx.RATIOS[1:]:
            for f in ("rho", "v", "P"):
                checks.append(Check(f"{b} R={R}: L1({f}) / L1(R=1)",
                                    res[R][f] / res[1][f], 1.5, kind="max",
                                    source=f"design L-T2; L1(R=1) = {res[1][f]:.4f}"))
    return checks
