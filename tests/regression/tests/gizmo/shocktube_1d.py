"""GIZMO 1D shock tubes against the exact Riemann solution.

Two problems on a periodic [0, 2) with mirrored tubes, 1D build, equal
particle masses, CFL 0.1, the right-hand tube (interface at x = 1.5)
compared with the exact solution (Toro 2009, ch. 4) over 1 < x < 2:

  Sod: (rho, v, P) = (1, 0, 1) | (0.125, 0, 0.1), 200 particles on the left,
       t = 0.2. L1 of rho, v, P (relative to the exact value; v relative to
       the post-shock speed). Measured 2026-10-20 before any GIZMO change:
       MFM HLLC 0.0205, 0.0162, 0.0255; MFM exact solver 0.0209, 0.0166,
       0.0261; SPHENIX 0.0202, 0.0211, 0.0281. Tolerance: 0.03 for each,
       about 1.2-1.5 times the measured values, so a lost factor in the
       Riemann fluxes fails while noise does not.
  Strong blast: (1, 0, 1000) | (1, 0, 0.01), 200 particles, t = 0.01
       (pressure ratio 1e5). L1(rho) 0.034 (measured 2026-10-20, both HLLC
       variants); tolerance 0.15. Without --limiter the MFM run collapsed its
       time-step below 1e-10; the limiter is part of the method (Hopkins
       2015, appendix G) and is on here.
"""

import os

import numpy as np

from swiftreg.checks import Check
from swiftreg.snapshot import snapshots

from . import _hydro1d as h

DESCRIPTION = "GIZMO-MFM 1D Sod and strong (1e5) shock tubes against the exact solution"
TIER = "quick"
BUILDS = {"mfm1d": h.build("gizmo-mfm", "hllc")}

CASES = {
    "sod": ((1.0, 0.0, 1.0), (0.125, 0.0, 0.1), 0.2),
    "strong": ((1.0, 0.0, 1000.0), (1.0, 0.0, 0.01), 0.01),
}
N_LEFT = 200


def run(ctx):
    for name, (WL, WR, t_end) in CASES.items():
        os.makedirs(ctx.path(name), exist_ok=True)
        h.shocktube_ics(ctx.path(name, "ics.hdf5"), N_LEFT, WL, WR)
        ctx.swift(h.params(t_end, dt_min=1e-10), ["--hydro", "--limiter"], subdir=name, name="run",
                  timeout=600)


def analyze(ctx):
    out = []
    for name, (WL, WR, t_end) in CASES.items():
        d = h.read(snapshots(ctx.path(name))[-1])
        sel = d["x"] > 1.0
        ex = h.riemann_exact(WL, WR, (d["x"][sel] - 1.5) / t_end)
        L1 = {}
        for k, f in enumerate(["rho", "v", "P"]):
            scale = np.max(np.abs(ex[:, k])) if f == "v" else ex[:, k]
            L1[f] = float(np.mean(np.abs(d[f][sel] - ex[:, k]) / scale))
        ctx.info[f"L1 {name}"] = L1
        if name == "sod":
            for f in ["rho", "v", "P"]:
                out.append(Check(f"Sod L1({f})", L1[f], 0.03, kind="max",
                                 source="exact Riemann solution; measured 0.016-0.026"))
        else:
            out.append(Check("strong tube L1(rho)", L1["rho"], 0.15, kind="max",
                             source="exact Riemann solution, pressure ratio 1e5"))
    return out
