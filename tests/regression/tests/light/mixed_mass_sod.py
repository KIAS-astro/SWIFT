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

Checks: the L1 error at each R is at most 1.5 times the equal-mass (R = 1)
run of the same scheme (design Sec. 6, L-T2 pass criterion); the R > 1
runs have more particles on the right, so a larger error is caused by the
mass contrast, not by resolution.
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


def _ics(path, R):
    rL, rR = WL[0], WR[0]
    m_g = rL / N_LEFT
    m_r = m_g / R
    nR = int(round(rR / m_r))
    x = np.concatenate([0.5 + (np.arange(N_LEFT) + 0.5) / N_LEFT,
                        np.mod(1.5 + (np.arange(nR) + 0.5) / nR, 2.0)])
    m = np.concatenate([np.full(N_LEFT, m_g), np.full(nR, m_r)])
    rho = np.concatenate([np.full(N_LEFT, rL), np.full(nR, rR)])
    P = np.concatenate([np.full(N_LEFT, WL[2]), np.full(nR, WR[2])])
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


def measure(ctx, b, R):
    d = h1.read(snapshots(ctx.path(f"{b}_R{R}"))[-1])
    sel = d["x"] > 1.0
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
        ctx.info[b] = res
        for R in mx.RATIOS[1:]:
            for f in ("rho", "v", "P"):
                checks.append(Check(f"{b} R={R}: L1({f}) / L1(R=1)",
                                    res[R][f] / res[1][f], 1.5, kind="max",
                                    source=f"design L-T2; L1(R=1) = {res[1][f]:.4f}"))
    return checks
