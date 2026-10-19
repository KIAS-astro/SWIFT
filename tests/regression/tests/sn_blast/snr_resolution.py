"""Resolution series of the resolved single SN with cooling (long tier).

The set-up of sn_blast/single_sn_cooling (one 1e51 erg SN in 32 particles,
n_H = 1 cm^-3, 4400 K, solar metallicity, EAGLE cooling, 128 pc box, glass) at
32^3, 64^3 and 96^3 particles (m_g = 2.1, 0.26, 0.078 Msun; the 96^3 glass tiles
the 32^3 glass 3 x 3 x 3). KO15 (Sec. 4) find the momentum converged to 5 per cent
for Delta x < r_sf/10; the question here is whether SPH converges to KO15's
terminal momentum (2.95e5 Msun km/s, their Eq. 29) or to the 3.8e5 of the 64^3 run,
and whether the over-dense, over-pressured hot interior of the 64^3 run (n_H ~ 0.06
cm^-3 at 300 kyr, against ~1e-3 in KO15) shrinks with resolution.

Recorded per resolution and output: radial momentum, mass, mean density, mean
pressure and thermal energy of gas hotter than 5e4 K. Check: each run reached 450 kyr.
"""

import os

import numpy as np

from swiftreg import data, params, units
from swiftreg.checks import Check
from swiftreg.snapshot import snapshots

from . import _common as c

DESCRIPTION = "single SN with cooling at 32^3, 64^3, 96^3: momentum and hot interior"
TIER = "long"
BUILDS = {"eagle": ["--with-subgrid=EAGLE"]}

BOX = 0.128
NH = 1.0
T = 4.4e3
DT_SNAP = 1.0226e-5
T_END = 4.6e-4
RES = {32: (32, 1), 64: (64, 1), 96: (32, 3)}  # N: (glass, tile)


def run(ctx):
    ctx.table("coolingtables")
    for n, (g, tile) in RES.items():
        sub = f"n{n}"
        os.makedirs(ctx.path(sub))
        c.blast_ics(ctx.path(sub, "ics.hdf5"), n, BOX, NH, T,
                    glass_file=data.glass(g, ctx.data_dir, log=ctx.log), glass_tile=tile)
        p = params.eagle_base("ics.hdf5", T_END, ctx.data_dir, DT_SNAP, dt_max=1e-6)
        p["Snapshots"]["compression"] = 4
        p["EAGLEEntropyFloor"].update({"Jeans_temperature_norm_K": 10.0,
                                       "Cool_temperature_norm_K": 10.0})
        ctx.swift(p, ["--hydro", "--cooling", "--limiter"], subdir=sub, name=sub,
                  timeout=6 * 3600)


def analyze(ctx):
    box_cm = BOX * units.UNIT_LENGTH
    checks = []
    for n in RES:
        rows = []
        for s in snapshots(ctx.path(f"n{n}")):
            d = c.radial(s, box_cm)
            hot = d["T"] > 5e4
            m = d["m"][hot]
            w = m if m.sum() > 0 else None
            rows.append([
                s.time_cgs / units.YEAR / 1e3,
                np.sum(d["m"] * d["vr"]) / (units.SOLAR_MASS * 1e5),
                m.sum() / units.SOLAR_MASS,
                np.average(d["rho"][hot], weights=w) * 0.752 / units.PROTON_MASS if w is not None else 0,
                np.average(d["P"][hot], weights=w) / units.BOLTZMANN_K if w is not None else 0,
                np.sum(m * d["u"][hot]) / 1e51,
            ])
        ctx.info[f"n{n}_t_kyr_pr_Mhot_nhot_Phot_Ehot"] = rows
        checks.append(Check(f"{n}^3: final time [kyr]", rows[-1][0], 470.4, kind="rel",
                            tol=1e-2, source="run completed"))
    return checks
