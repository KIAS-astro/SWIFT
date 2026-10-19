"""Radiative remnant of one SN in uniform gas with EAGLE cooling (long tier).

10^51 erg in the 32 particles nearest the centre of a 64^3 lattice,
n_H = 1 cm^-3, T = 10^4 K, solar metallicity, 128 pc box (m_g = 0.30 Msun,
h = 2.5 pc); hydro and EAGLE cooling at z = 0 (Wiersma et al. 2009 tables). The
entropy floor is lowered to 10 K so that it does not hold up the cooled shell
(the EAGLE floor would keep shell gas at n_H = 10-100 cm^-3 at 40-80 kK).

Measured about the centre from snapshots every 10 kyr to 300 kyr:
  - hot-gas mass, T > 10^5.5 K; shell formation t_sf is taken at its maximum;
  - radial momentum, sum of m v.r_hat.

Reference: Kim & Ostriker (2015, KO15) for 1e51 erg, n_H = 1 cm^-3, solar
metallicity: t_sf ~ 4.4e4 yr, radial momentum at t_sf ~ 2.2e5 Msun km/s, rising to
~2.8e5 Msun km/s at late times. THESE VALUES ARE RECALLED, NOT READ: KO15 is not
in the local paper library. The checks use a 30 per cent tolerance for SPH against
KO15's grid code, a different cooling function, and the hot-mass definition; tighten
it, and cite the equation numbers, once the values are checked against the paper.
"""

import numpy as np

from swiftreg import params, units
from swiftreg.checks import Check
from swiftreg.snapshot import snapshots

from . import _common as c

DESCRIPTION = "single SN with EAGLE cooling: shell formation time and momentum against KO15"
TIER = "long"
BUILDS = {"eagle": ["--with-subgrid=EAGLE"]}

N = 64
BOX = 0.128  # kpc
NH = 1.0
T = 1e4
DT_SNAP = 1.0226e-5  # internal units, 10.0 kyr
T_END = 3.068e-4  # 300 kyr
T_HOT = 10 ** 5.5

KO15_T_SF_YR = 4.4e4
KO15_P_SF = 2.2e5  # Msun km/s
KO15_P_LATE = 2.8e5


def run(ctx):
    ctx.table("coolingtables")
    _, rho0 = c.blast_ics(ctx.path("ics.hdf5"), N, BOX, NH, T)
    ctx.info["rho0"] = rho0
    p = params.eagle_base("ics.hdf5", T_END, ctx.data_dir, DT_SNAP, dt_max=1e-6)
    p["Snapshots"]["compression"] = 4
    p["EAGLEEntropyFloor"].update({"Jeans_temperature_norm_K": 10.0,
                                   "Cool_temperature_norm_K": 10.0})
    ctx.swift(p, ["--hydro", "--cooling", "--limiter"], timeout=4 * 3600)


def analyze(ctx):
    box_cm = BOX * units.UNIT_LENGTH
    msun_kms = units.SOLAR_MASS * 1e5
    rows = []
    for s in snapshots(ctx.run_dir):
        d = c.radial(s, box_cm)
        hot = np.sum(d["m"][d["T"] > T_HOT]) / units.SOLAR_MASS
        p_r = np.sum(d["m"] * d["vr"]) / msun_kms
        rows.append([s.time_cgs / units.YEAR, hot, p_r])
    rows = np.array(rows)
    ctx.info["t_yr_Mhot_Msun_pr_Msunkms"] = rows.tolist()
    i_sf = int(np.argmax(rows[:, 1]))
    t_sf, p_sf, p_late = rows[i_sf, 0], rows[i_sf, 2], rows[-1, 2]
    return [
        Check("shell formation time [yr]", t_sf, KO15_T_SF_YR, kind="rel", tol=0.3,
              source="KO15 t_sf at n_H = 1 (RECALLED, not read); peak of M(T > 10^5.5 K) "
                     "at 10 kyr cadence"),
        Check("radial momentum at t_sf [Msun km/s]", p_sf, KO15_P_SF, kind="rel", tol=0.3,
              source="KO15 p_sf at n_H = 1 (RECALLED, not read)"),
        Check("radial momentum at 300 kyr [Msun km/s]", p_late, KO15_P_LATE, kind="rel",
              tol=0.3, source="KO15 late-time momentum at n_H = 1 (RECALLED, not read)"),
    ]
