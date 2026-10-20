"""GIZMO 1D sound wave: second-order convergence.

Right-moving isentropic simple wave, rho0 = 1, c0 = 1, v = A sin(2 pi x)
with A = 1e-2, one box crossing (t = 1), equidistant particles with masses
rho dx, 1D build, CFL 0.1. Exact solution by characteristics: v - 2c/(gamma-1)
is uniform and v is constant along x = xi + (v + c) t; valid before the
wave steepens into a shock at t = 2 / (2 pi A (gamma + 1)) = 11.9.

Measured 2026-10-20 (MFM, HLLC, before any change): L1(rho)/A = 0.066,
0.025, 0.0075, 0.0027 for N = 32, 64, 128, 256 (order 1.4, 1.7, 1.5);
SPHENIX 0.030, 0.0083, 0.0023, 0.0033. Hopkins (2015, sec. 4.2.1) shows
second order for MFM. Above N = 256 single-precision round-off over 3e4
steps dominates (0.0078 at N = 512), so the test stops at N = 128.
Changing CFL from 0.05 to 0.8 at N = 128 changes L1 by 30 per cent only:
the error is spatial.

Checks: L1(rho)/A at N = 128 below 0.01; convergence order between N = 32
and 128 above 1.3.
"""

import os

import numpy as np
from scipy.optimize import brentq

from swiftreg.checks import Check
from swiftreg.snapshot import snapshots

from . import _hydro1d as h

DESCRIPTION = "GIZMO-MFM 1D sound wave: L1 error and convergence order"
TIER = "quick"
BUILDS = {"mfm1d": h.build("gizmo-mfm", "hllc")}

AMP = 1e-2
RESOLUTIONS = [32, 128]
G = h.GAMMA


def run(ctx):
    for n in RESOLUTIONS:
        sub = f"n{n}"
        os.makedirs(ctx.path(sub), exist_ok=True)
        dx = 1.0 / n
        x = (np.arange(n) + 0.5) * dx
        v = AMP * np.sin(2 * np.pi * x)
        c = 1.0 + 0.5 * (G - 1) * v
        rho = c ** (2 / (G - 1))
        P = rho * c * c / G
        h.write_ic(ctx.path(sub, "ics.hdf5"), x, v, rho * dx, P / ((G - 1) * rho),
                   np.full(n, 1.2348 * dx), 1.0)
        ctx.swift(h.params(1.0), ["--hydro", "--limiter"], subdir=sub, timeout=600)


def exact_v(x, t):
    out = np.empty_like(x)
    for k, X in enumerate(x):
        f = lambda xi: xi + (AMP * np.sin(2 * np.pi * xi) * 0.5 * (G + 1) + 1.0) * t - X
        lo = X - t * (1.0 + 2 * AMP) - 0.1
        out[k] = AMP * np.sin(2 * np.pi * brentq(f, lo, lo + 1.2))
    return out


def analyze(ctx):
    L1 = {}
    for n in RESOLUTIONS:
        d = h.read(snapshots(ctx.path(f"n{n}"))[-1])
        ve = exact_v(d["x"], 1.0)
        rhoe = (1.0 + 0.5 * (G - 1) * ve) ** (2 / (G - 1))
        # the volume estimate biases the mean density; compare perturbations
        L1[n] = float(np.mean(np.abs((d["rho"] - d["rho"].mean())
                                     - (rhoe - rhoe.mean()))) / AMP)
    order = np.log(L1[32] / L1[128]) / np.log(4.0)
    ctx.info.update(L1=L1, order=float(order))
    return [
        Check("L1(rho)/A at N = 128", L1[128], 0.01, kind="max",
              source="measured 0.0075 (2026-10-20)"),
        Check("convergence order N = 32 to 128", order, 1.3, kind="min",
              source="second-order scheme (Hopkins 2015 sec. 4.2.1); measured 1.6"),
    ]
