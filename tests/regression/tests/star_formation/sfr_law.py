"""Each gas particle's SFR equals the EAGLE pressure law at its own state.

A uniform cold box at n_H = 1 cm^-3, T = 10^3 K and solar metallicity is
above the density threshold (0.03 cm^-3) and below the entropy limit (the
floor at this density is 1.7e4 K), so every particle is star-forming. The
SFR written to the snapshot is compared, particle by particle, with the
pressure law evaluated from the density, mass and pressure in the same
snapshot.
"""

import numpy as np

from swiftreg import units
from swiftreg.checks import Check
from swiftreg.snapshot import snapshots

from . import _common as c

DESCRIPTION = "EAGLE SFR of every particle equals the Schaye & Dalla Vecchia (2008) pressure law"
TIER = "quick"
BUILDS = {"eagle": c.EAGLE_BUILD}

NH = 1.0
T = 1e3
T_END = 2e-2  # 20 Myr; a few hundred time-steps, a few per cent of stars


def run(ctx):
    c.write_box(ctx, NH, T)
    p = c.sf_params(ctx, T_END, T_END / 2)
    ctx.info["params"] = p["EAGLEStarFormation"]
    ctx.swift(p, c.SF_FLAGS)


def analyze(ctx):
    snap = snapshots(ctx.run_dir)[-1]
    sfr = snap.read(0, "StarFormationRates")
    m = snap.read(0, "Masses")
    P = snap.read(0, "Pressures")
    sf = ctx.info["params"]

    expected = c.sfr_pressure_law(m, P, sf)
    star_forming = sfr > 0
    rel = np.abs(sfr[star_forming] / expected[star_forming] - 1.0)

    return [
        Check("fraction of gas particles star-forming", star_forming.mean(), 1.0,
              kind="equal",
              source=f"n_H = {NH} > threshold {c.density_threshold(sf, c.Z):.3g} cm^-3; "
                     f"T = {T:g} K below floor x 10^{sf['EOS_entropy_margin_dex']}"),
        Check("max |SFR / pressure law - 1| over particles", rel.max(), 1e-4,
              kind="max",
              source="SD08 eq. 19 from each particle's own mass and pressure; "
                     "tolerance is single precision of the output fields"),
        Check("mean gas depletion time [Gyr]",
              (m.sum() / sfr.sum()) / units.GYR,
              (m.sum() / expected.sum()) / units.GYR, kind="rel", tol=1e-4,
              source="same law, summed"),
    ]
