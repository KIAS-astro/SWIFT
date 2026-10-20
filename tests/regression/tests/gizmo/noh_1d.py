"""GIZMO 1D Noh problem: cold colliding flows.

Planar Noh (Noh 1987): rho = 1, v = +-1 towards x = 1, P ~ 0 (u = 1e-6),
gamma = 5/3, 400 equal-mass particles on a periodic [0, 2), t = 0.6, 1D
build. Exact: two shocks moving out at 1/3, rho = 4, P = 4/3, v = 0 inside.

Run with the time-step limiter (--limiter), which GIZMO requires (Hopkins
2015, appendix G): without it, cold pre-shock particles on long time-steps
are overrun by the shock mid-step and the time-step collapses below 1e-8 at
t = 0.1, with HLLC and with the exact solver alike (measured 2026-10-20).
With it: post-shock density 3.984 and shock at 0.2015 with the original
HLLC solver (28 s), 3.988 and 0.2013 with Roe wave speeds (8.5 s; the
linearised star pressure underestimates the pressure between colliding cold
particles, so they approach closer and need smaller steps).

Checks: run completes; mean density in the inner 60 per cent of the
shocked region 4 within 10 per cent (wall heating lowers it at the centre,
as in Hopkins 2015 sec. 4.3.4); shock position t/3 within 0.02, four initial
particle spacings.
"""

import os

import numpy as np

from swiftreg.checks import Check
from swiftreg.snapshot import snapshots

from . import _hydro1d as h

DESCRIPTION = "GIZMO-MFM 1D Noh problem: post-shock density and shock position"
TIER = "quick"
BUILDS = {"mfm1d": h.build("gizmo-mfm", "hllc")}

N = 400
T_END = 0.6


def run(ctx):
    x = (np.arange(N) + 0.5) * 2.0 / N
    v = np.where(x < 1.0, 1.0, -1.0)
    dx = 2.0 / N
    h.write_ic(ctx.path("ics.hdf5"), x, v, np.full(N, dx), np.full(N, 1e-6),
               np.full(N, 1.2348 * dx), 2.0)
    ctx.swift(h.params(T_END, dt_min=1e-8), ["--hydro", "--limiter"], timeout=600)


def analyze(ctx):
    d = h.read(snapshots(ctx.run_dir)[-1])
    xs = T_END / 3
    r = np.abs(d["x"] - 1.0)
    core = (r > 0.2 * xs) & (r < 0.8 * xs)
    rho_core = float(np.mean(d["rho"][core]))
    shocked = r[d["rho"] > 2.5]
    shock_x = float(np.max(shocked)) if len(shocked) else 0.0
    ctx.info.update(rho_core=rho_core, shock_x=shock_x,
                    P_core=float(np.mean(d["P"][core])))
    return [
        Check("post-shock density (0.2-0.8 of the shocked region)", rho_core, 4.0,
              kind="rel", tol=0.1, source="Noh: (gamma + 1)/(gamma - 1) = 4"),
        Check("shock position |x - 1|", shock_x, xs, kind="abs", tol=0.02,
              source="Noh: shock speed (gamma - 1)/2 = 1/3"),
    ]
