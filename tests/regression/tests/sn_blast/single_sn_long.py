"""Single SN with cooling run to 1.5 Myr (36 t_sf) in a 256 pc box: does the momentum saturate?

single_sn_cooling stops at 450 kyr in a 128 pc box, where the radial momentum is
still rising (3.83e5 at 419 kyr against KO15's 2.95e5, which KO15 find flat after
4-5 t_sf). This test continues the same set-up (n_H = 1 cm^-3, solar metallicity,
4400 K, 1e51 erg in the 32 central particles, EAGLE cooling, no entropy floor) to
1.5 Myr in a box twice as large, so the shell (r ~ 46 pc at 420 kyr, r ~ t^0.3)
stays inside half the box.

Resolution from SWIFT_REG_SNR_RES: "high" = 64^3 glass tiled 2x2x2 (128^3, 0.26
Msun, as single_sn_cooling; 2.1e6 particles, run on a cluster), "low" (default) =
64^3 glass (2.1 Msun; spacing 3.9 pc, r_sf / spacing = 5.8, within KO15's
Delta < r_sf / 3).

Checks: momentum at 41.9 kyr (KO15 Table 1) within 15 per cent; and the measured
quantity of interest, recorded without a pass criterion: the momentum at 0.42, 0.8
and 1.5 Myr relative to KO15 Eq. 29.
"""

import os

import numpy as np

from swiftreg import params
from swiftreg.checks import Check

from . import _common as c
from .single_sn_cooling import BUILDS, KO15_P_FINAL, KO15_P_SF, KO15_T_SF_YR, NH, T, history  # noqa: F401

DESCRIPTION = "single SN with cooling to 1.5 Myr in a 256 pc box (momentum saturation)"
TIER = "long"
RES = os.environ.get("SWIFT_REG_SNR_RES", "low")
BOX = 0.256  # kpc
TILE = {"low": 1, "high": 2}[RES]
DT_SNAP = 2.0452e-5  # 20 kyr
T_END = 1.5339e-3  # 1.5 Myr


def run(ctx):
    ctx.table("coolingtables")
    from swiftreg import data
    glass = data.glass(64, ctx.data_dir, log=ctx.log)
    _, rho0 = c.blast_ics(ctx.path("ics.hdf5"), 64 * TILE, BOX, NH, T, glass_file=glass,
                          glass_tile=TILE)
    ctx.info["rho0"] = rho0
    ctx.info["res"] = RES
    p = params.eagle_base("ics.hdf5", T_END, ctx.data_dir, DT_SNAP, dt_max=1e-6)
    p["Snapshots"]["compression"] = 4
    p["EAGLEEntropyFloor"].update({"Jeans_temperature_norm_K": 10.0,
                                   "Cool_temperature_norm_K": 10.0})
    ctx.swift(p, ["--hydro", "--cooling", "--limiter"], timeout=24 * 3600)


def analyze(ctx):
    rows = history(ctx.run_dir, box_kpc=BOX)
    ctx.info["t_yr_Mhot_Msun_pr_Msunkms_E_erg"] = rows.tolist()
    t, p_r = rows[:, 0], rows[:, 2]
    checks = [Check("radial momentum at 41.9 kyr [Msun km/s]", np.interp(KO15_T_SF_YR, t, p_r),
                    KO15_P_SF, kind="rel", tol=0.15, source="KO15 Table 1 (Eq. 25), n_0 = 1")]
    for t_myr in (0.42, 0.8, 1.5):
        ctx.info[f"p_over_KO15_final_at_{t_myr}Myr"] = float(np.interp(t_myr * 1e6, t, p_r)
                                                              / KO15_P_FINAL)
    return checks
