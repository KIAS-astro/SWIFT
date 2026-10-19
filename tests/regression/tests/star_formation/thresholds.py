"""Gas outside the EAGLE star-forming region forms no stars.

Two boxes, each run for 50 Myr:
  diffuse: n_H = 0.01 cm^-3, below the metallicity-dependent threshold
           n_H,th = 0.1 (Z/0.002)^-0.64 cm^-3 = 0.030 cm^-3 at Z = 0.0129;
  warm:    n_H = 1 cm^-3, above the threshold, but T = 10^5 K, more than
           EOS_entropy_margin_dex (0.5 dex) above the entropy floor
           (1.7e4 K at this density), so the entropy criterion rejects it.
If either box were star-forming it would form several hundred stars in
this time (depletion times of 1.5 to 4 Gyr from the pressure law).
"""

import numpy as np

from swiftreg.checks import Check
from swiftreg.snapshot import snapshots

from . import _common as c

DESCRIPTION = "no star formation below the density threshold or above the entropy margin"
TIER = "quick"
BUILDS = {"eagle": c.EAGLE_BUILD}

CASES = {"diffuse": (0.01, 1e3), "warm": (1.0, 1e5)}
T_END = 0.05


def run(ctx):
    for name, (nH, T) in CASES.items():
        c.write_box(ctx, nH, T, subdir=name)
        p = c.sf_params(ctx, T_END, T_END)
        ctx.info["params"] = p["EAGLEStarFormation"]
        ctx.swift(p, c.SF_FLAGS, subdir=name, name=name)


def analyze(ctx):
    sf = ctx.info["params"]
    nth = c.density_threshold(sf, c.Z)
    checks = []
    for name, (nH, T) in CASES.items():
        last = snapshots(ctx.path(name))[-1]
        sfr = last.read(0, "StarFormationRates")
        why = (f"n_H = {nH} < threshold {nth:.3g} cm^-3" if nH < nth
               else f"T = {T:g} K above floor x 10^{sf['EOS_entropy_margin_dex']}")
        checks.append(Check(f"{name}: particles with SFR > 0", np.sum(sfr > 0), 0,
                            kind="equal", source=why))
        checks.append(Check(f"{name}: stars formed", last.npart[4], 0, kind="equal",
                            source=why))
    return checks
